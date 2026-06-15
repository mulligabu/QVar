"""Does the Markov+Boltzmann model's P(up) beat the MARKET's implied probability?

This is the honest test of the vol-mispricing edge. For each window we take a
snapshot at a moderate lead (default 5-12 min before close) -- late enough that
the contract trades on volatility/distance, early enough that the outcome is NOT
already determined (avoiding the lead-lag direction trap). We compare:

  * model P(up): Boltzmann terminal dist with Markov vol-regime sigma (causal spot)
  * market P(up): the contract mid
  * realized outcome (causal candle label, ~96% vs venue settlement)

If Brier(model) < Brier(market) -- especially on OFF-0.50 contracts where vol
drives the price -- the model carries information the market lacks => real edge.
We also report net-of-fee EV of trading the model's disagreement at the book ask.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.edge_proof.loaders import find_runs
from research.edge_proof.markets import iter_quote_windows
from research.edge_proof.outcomes import OutcomeResolver
from research.strategy.boltzmann_terminal import kelly_fraction, prob_up
from research.strategy.markov_vol import MarkovVol

DEFAULT_STRAT = Path("/home/user/crypto_strat/prediction_market_bots/data")


def taker_fee(p: float, rate: float = 0.072) -> float:
    p = max(0.0, min(1.0, p))
    return rate * p * (1.0 - p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-data", type=Path, default=DEFAULT_STRAT)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--venue", default="polymarket")
    ap.add_argument("--beta", type=float, default=1.4, help="Boltzmann tail exponent (<2 = fat tails)")
    ap.add_argument("--lead-lo", type=float, default=5.0, help="min minutes before close")
    ap.add_argument("--lead-hi", type=float, default=12.0, help="max minutes before close")
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE,BNB,HYPE")
    args = ap.parse_args()

    runs = find_runs(args.strat_data / "forward")[-args.runs:]
    candles = args.strat_data / "candles" / "binance_multiasset_model_dirs"
    since = datetime.now(timezone.utc) - timedelta(days=20)
    resolver = OutcomeResolver(candles, since=since)
    assets = args.assets.split(",")
    mv: dict[str, MarkovVol | None] = {}

    # calibration accumulators, split by |market - 0.5| band
    bands = {"near(.4-.6)": (0.0, 0.10), "mid(.25-.4/.6-.75)": (0.10, 0.25), "off(<.25/>.75)": (0.25, 0.51)}
    cal = {b: dict(n=0, mse_model=0.0, mse_mkt=0.0) for b in bands}
    # trading accumulators
    trade = dict(n=0, win=0, net=0.0, sized_net=0.0, sized_stake=0.0)
    n_obs = 0

    for w in iter_quote_windows(runs, resolver, assets=assets):
        if w.outcome is None or w.strike is None:
            continue
        series = resolver._get(w.asset)
        if series is None:
            continue
        if w.asset not in mv:
            m = MarkovVol(series)
            f0 = m.forecast(int(w.start))
            mv[w.asset] = m if f0 is not None else None
        model_vol = mv[w.asset]
        if model_vol is None:
            continue
        fc = model_vol.forecast(int(w.start))
        if fc is None:
            continue
        snaps = w.snaps_by_venue.get(args.venue, [])
        # pick the snapshot nearest the middle of the lead band
        target = w.end - 0.5 * (args.lead_lo + args.lead_hi) * 60
        best = None
        for s in snaps:
            mins = (w.end - s.ts) / 60.0
            if args.lead_lo <= mins <= args.lead_hi and s.up_mid is not None:
                if best is None or abs(s.ts - target) < abs(best.ts - target):
                    best = s
        if best is None:
            continue
        spot = series.price_at(datetime.fromtimestamp(best.ts, tz=timezone.utc))
        if spot is None:
            continue
        tau = w.end - best.ts
        mp = prob_up(spot, w.strike, fc.sigma_window, tau, beta=args.beta)
        mkt = best.up_mid
        won_up = 1 if w.outcome == "up" else 0
        n_obs += 1

        dist = abs(mkt - 0.5)
        for b, (lo, hi) in bands.items():
            if lo <= dist < hi:
                acc = cal[b]
                acc["n"] += 1
                acc["mse_model"] += (mp - won_up) ** 2
                acc["mse_mkt"] += (mkt - won_up) ** 2
                break

        # trade the model's disagreement, executed at the book ask
        edge_up = mp - mkt
        if edge_up > args.min_edge and best.up_ask is not None:
            price, won = best.up_ask, won_up
        elif -edge_up > args.min_edge and best.up_bid is not None:
            price, won = 1.0 - best.up_bid, 1 - won_up
        else:
            continue
        if not (0.0 < price < 1.0):
            continue
        fee = taker_fee(price)
        net = won - price - fee
        f = kelly_fraction(max(mp, 1 - mp), price, temperature=1.0 + 2.0 * fc.entropy)
        trade["n"] += 1
        trade["win"] += won
        trade["net"] += net
        trade["sized_net"] += f * net
        trade["sized_stake"] += f * price

    print("# Vol-edge validation — Markov+Boltzmann model vs market implied")
    print(f"runs={args.runs} venue={args.venue} beta={args.beta} lead={args.lead_lo}-{args.lead_hi}min")
    print(f"observations: {n_obs}\n")
    print("## Calibration: Brier(model) vs Brier(market) by distance-from-0.50")
    print(f"{'band':24} {'n':>6} {'Brier_model':>12} {'Brier_mkt':>11} {'model_better':>13}")
    for b in bands:
        a = cal[b]
        if a["n"] == 0:
            continue
        bm, bk = a["mse_model"] / a["n"], a["mse_mkt"] / a["n"]
        print(f"{b:24} {a['n']:6d} {bm:12.5f} {bk:11.5f} {('YES' if bm < bk else 'no'):>13}")
    print("\n## Trading the model's disagreement (taker at book ask)")
    t = trade
    if t["n"]:
        print(f"trades={t['n']}  win%={t['win']/t['n']:.4f}  net/contract={t['net']/t['n']:+.4f}")
        print(f"Kelly-sized: net={t['sized_net']:+.3f} on stake={t['sized_stake']:.2f} "
              f"=> ROI={ (t['sized_net']/t['sized_stake']) if t['sized_stake'] else 0:+.4f}")
    else:
        print("no qualifying trades")
    print("\nVerdict: model edge is real only if Brier_model < Brier_mkt (esp. off-0.50)")
    print("AND net/contract > 0. Otherwise the market's implied vol already wins.")


if __name__ == "__main__":
    main()
