"""Test 1: terminal-value / lead-lag mispricing (Brownian bridge vs the book).

For each window we know the strike (underlying at start) and the realized outcome.
At each point in time the parameter-free digital probability
    P(up) = Phi( ln(spot_now/strike) / (vol_per_min * sqrt(minutes_left)) )
is compared to the contract's implied P(up). If the book lags the underlying, the
bridge signal predicts the outcome AND there is positive net-of-fee EV in buying
the side the bridge says is underpriced.

We take one observation per (window, time-to-expiry bucket) — the latest snapshot
in that bucket — to avoid autocorrelation, and we only count EXECUTABLE trades
(the side we want to buy has a real ask). Net EV uses Polymarket taker fees.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.edge_proof.loaders import find_runs
from research.edge_proof.markets import bridge_prob_up, iter_quote_windows, per_minute_vol
from research.edge_proof.outcomes import OutcomeResolver

DEFAULT_STRAT = Path("/home/user/crypto_strat/prediction_market_bots/data")
BUCKETS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 12), (12, 16)]  # minutes remaining


def taker_fee(p: float, rate: float = 0.072) -> float:
    return rate * max(0.0, min(1.0, p)) * (1.0 - max(0.0, min(1.0, p)))


def bucket_of(minutes: float):
    for lo, hi in BUCKETS:
        if lo <= minutes < hi:
            return (lo, hi)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-data", type=Path, default=DEFAULT_STRAT)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--venue", default="polymarket")
    ap.add_argument("--min-signal", type=float, default=0.05, help="min |bridge - mid| to trade")
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE,BNB")
    args = ap.parse_args()

    runs = find_runs(args.strat_data / "forward")[-args.runs:]
    candles = args.strat_data / "candles" / "binance_multiasset_model_dirs"
    since = datetime.now(timezone.utc) - timedelta(days=20)
    resolver = OutcomeResolver(candles, since=since)
    assets = args.assets.split(",")
    vol_cache: dict[str, float] = {}

    # stats[bucket] -> dict of accumulators
    stats = {b: dict(n=0, exec_n=0, dir_correct=0, sum_signal=0.0, sum_net=0.0, sum_fee=0.0, traded=0, traded_win=0)
             for b in BUCKETS}
    n_windows = 0

    print("# Test 1 — Terminal-value / lead-lag bridge vs book")
    print(f"runs={args.runs} venue={args.venue} min_signal={args.min_signal} assets={assets}\n")

    for w in iter_quote_windows(runs, resolver, assets=assets):
        if w.outcome is None or w.strike is None:
            continue
        series = resolver._get(w.asset)
        if series is None:
            continue
        if w.asset not in vol_cache:
            vol_cache[w.asset] = per_minute_vol(series)
        vol = vol_cache[w.asset]
        won_up = 1 if w.outcome == "up" else 0
        snaps = w.snaps_by_venue.get(args.venue, [])
        if not snaps:
            continue
        n_windows += 1
        # one obs per (window, bucket): latest snap in bucket
        latest_in_bucket: dict[tuple, object] = {}
        for s in snaps:
            mins = (w.end - s.ts) / 60.0
            b = bucket_of(mins)
            if b is None:
                continue
            latest_in_bucket[b] = (s, mins)
        for b, (s, mins) in latest_in_bucket.items():
            spot = series.price_at(datetime.fromtimestamp(s.ts, tz=timezone.utc))
            if spot is None:
                continue
            bp = bridge_prob_up(spot, w.strike, mins, vol)
            mid = s.up_mid
            if mid is None:
                continue
            signal = bp - mid
            acc = stats[b]
            acc["n"] += 1
            acc["sum_signal"] += abs(signal)
            # directional: does bridge pick the winning side?
            bridge_side_up = bp >= 0.5
            if (bridge_side_up and won_up) or (not bridge_side_up and not won_up):
                acc["dir_correct"] += 1
            # tradeable: buy the side the bridge says is underpriced by > min_signal
            if signal > args.min_signal and s.up_ask is not None:      # buy UP at ask
                price, won = s.up_ask, won_up
            elif -signal > args.min_signal and s.up_bid is not None:    # buy DOWN at (1-up_bid)
                price, won = 1.0 - s.up_bid, (1 - won_up)
            else:
                continue
            if not (0.0 < price < 1.0):
                continue
            fee = taker_fee(price)
            acc["exec_n"] += 1
            acc["traded"] += 1
            acc["traded_win"] += won
            acc["sum_net"] += won - price - fee
            acc["sum_fee"] += fee

    print(f"windows with {args.venue} quotes: {n_windows}\n")
    print(f"{'mins_left':10} {'obs':>5} {'|signal|':>9} {'bridge_dir%':>11} {'trades':>7} {'trade_win%':>10} {'net/contract':>13}")
    tot_trades = tot_win = 0
    tot_net = 0.0
    for b in BUCKETS:
        a = stats[b]
        if a["n"] == 0:
            continue
        avg_sig = a["sum_signal"] / a["n"]
        dirpct = a["dir_correct"] / a["n"]
        tw = (a["traded_win"] / a["traded"]) if a["traded"] else None
        net = (a["sum_net"] / a["traded"]) if a["traded"] else None
        tot_trades += a["traded"]; tot_win += a["traded_win"]; tot_net += a["sum_net"]
        print(f"{f'{b[0]}-{b[1]}':10} {a['n']:5d} {avg_sig:9.4f} {dirpct:11.4f} {a['traded']:7d} "
              f"{(f'{tw:.4f}' if tw is not None else '-'):>10} {(f'{net:+.4f}' if net is not None else '-'):>13}")
    print("-" * 70)
    if tot_trades:
        print(f"{'TOTAL':10} {'':>5} {'':>9} {'':>11} {tot_trades:7d} {tot_win/tot_trades:10.4f} {tot_net/tot_trades:+13.4f}")
        print(f"\nAggregate executable net-of-fee PnL per contract: {tot_net/tot_trades:+.4f}  over {tot_trades} trades")
    else:
        print("no executable trades at this min_signal")


if __name__ == "__main__":
    main()
