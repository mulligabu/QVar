"""Test 3: cross-venue relative value (Polymarket vs Kalshi).

For windows quoted on BOTH venues we align snapshots in time and compare the
implied P(up). Polymarket settles on Chainlink; Kalshi on a CF Benchmarks BRTI
60s average -- so a persistent *basis* is expected and must be separated from
exploitable divergence.

Test: when the two venues' implied P(up) diverge by > threshold at the same time,
buy UP on the venue where UP is cheaper (and, implicitly, you'd buy DOWN on the
other). Measure realized net-of-fee EV of the cheap-venue UP leg against the
candle outcome, vs. the mean basis. If divergence beyond the basis predicts the
outcome, that's structural relative-value edge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from research.edge_proof.loaders import find_runs
from research.edge_proof.markets import iter_quote_windows
from research.edge_proof.outcomes import OutcomeResolver

DEFAULT_STRAT = Path("/home/user/crypto_strat/prediction_market_bots/data")


def taker_fee(p: float, rate: float = 0.072) -> float:
    p = max(0.0, min(1.0, p))
    return rate * p * (1.0 - p)


def aligned_pairs(poly, kalshi, tol=20.0):
    """Yield (poly_snap, kalshi_snap) nearest in time within tol seconds."""
    j = 0
    for ps in poly:
        # advance kalshi pointer to nearest
        best = None
        bestd = tol + 1
        for ks in kalshi:
            d = abs(ks.ts - ps.ts)
            if d < bestd:
                bestd, best = d, ks
        if best is not None and bestd <= tol:
            yield ps, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-data", type=Path, default=DEFAULT_STRAT)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--min-div", type=float, default=0.04, help="min |poly_up - kalshi_up| to trade")
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE,BNB")
    args = ap.parse_args()

    runs = find_runs(args.strat_data / "forward")[-args.runs:]
    candles = args.strat_data / "candles" / "binance_multiasset_model_dirs"
    since = datetime.now(timezone.utc) - timedelta(days=20)
    resolver = OutcomeResolver(candles, since=since)
    assets = args.assets.split(",")

    basis_samples: list[float] = []     # poly_up_mid - kalshi_up_mid
    n_both = 0
    trades = 0
    trade_win = 0
    sum_net = 0.0
    # one observation per window: the pair with the largest divergence in the last 8 min
    print("# Test 3 — Cross-venue relative value (Polymarket vs Kalshi)\n")

    for w in iter_quote_windows(runs, resolver, assets=assets):
        if w.outcome is None:
            continue
        poly = w.snaps_by_venue.get("polymarket", [])
        kal = w.snaps_by_venue.get("kalshi", [])
        if not poly or not kal:
            continue
        n_both += 1
        won_up = 1 if w.outcome == "up" else 0
        # find the max-divergence aligned pair where both mids exist
        best = None
        for ps, ks in aligned_pairs(poly, kal):
            if ps.up_mid is None or ks.up_mid is None:
                continue
            div = ps.up_mid - ks.up_mid
            basis_samples.append(div)
            if best is None or abs(div) > abs(best[2]):
                best = (ps, ks, div)
        if best is None:
            continue
        ps, ks, div = best
        if abs(div) < args.min_div:
            continue
        # buy UP on the venue where UP is cheaper (lower mid); use that venue's ask
        if div > 0:   # poly up richer -> up cheaper on kalshi -> buy up on kalshi
            price = ks.up_ask if ks.up_ask is not None else ks.up_mid
            won = won_up
        else:         # up cheaper on poly -> buy up on poly
            price = ps.up_ask if ps.up_ask is not None else ps.up_mid
            won = won_up
        if price is None or not (0.0 < price < 1.0):
            continue
        fee = taker_fee(price)
        trades += 1
        trade_win += won
        sum_net += won - price - fee

    # ---- market-neutral arb: buy UP on cheap venue + DOWN on rich venue ------
    # DOWN on rich venue costs (1 - up_bid_rich). If up_ask_cheap + (1-up_bid_rich)
    # + fees < 1, the pair locks $1 of payoff for less than $1 -> riskless profit,
    # EXCEPT when the two venues settle to different outcomes (feed basis risk).
    arb_ops = 0
    arb_profit = 0.0
    arb_checked = 0
    for w in iter_quote_windows(runs, resolver, assets=assets):
        poly = w.snaps_by_venue.get("polymarket", [])
        kal = w.snaps_by_venue.get("kalshi", [])
        if not poly or not kal:
            continue
        best = None
        for ps, ks in aligned_pairs(poly, kal):
            if None in (ps.up_bid, ps.up_ask, ks.up_bid, ks.up_ask):
                continue
            arb_checked += 1
            # try both orientations, take the cheaper total lock cost
            # A: up on kalshi (cheap if ks.up_ask low), down on poly (1 - ps.up_bid)
            costA = ks.up_ask + (1.0 - ps.up_bid)
            costA += taker_fee(ks.up_ask) + taker_fee(1.0 - ps.up_bid)
            # B: up on poly, down on kalshi
            costB = ps.up_ask + (1.0 - ks.up_bid)
            costB += taker_fee(ps.up_ask) + taker_fee(1.0 - ks.up_bid)
            cost = min(costA, costB)
            if best is None or cost < best:
                best = cost
        if best is None:
            continue
        if best < 1.0:
            arb_ops += 1
            arb_profit += (1.0 - best)

    print(f"windows quoted on both venues: {n_both}")
    if basis_samples:
        print(f"mean basis (poly_up - kalshi_up): {mean(basis_samples):+.4f}  (n={len(basis_samples)} aligned pairs)")
        absb = sorted(abs(x) for x in basis_samples)
        print(f"median |divergence|: {absb[len(absb)//2]:.4f}   p90 |divergence|: {absb[int(len(absb)*0.9)]:.4f}")
    print(f"\nbuy-cheaper-venue-UP when |divergence| >= {args.min_div}:")
    if trades:
        print(f"  trades={trades}  win%={trade_win/trades:.4f}  net/contract={sum_net/trades:+.4f}")
    else:
        print("  no qualifying trades")
    print(f"\nmarket-neutral arb (buy UP cheap + DOWN rich, both legs executable):")
    print(f"  windows with a two-sided pair on both venues: {sum(1 for _ in ()) or '-'}  checked_pairs={arb_checked}")
    if arb_checked:
        print(f"  windows where best lock-cost < $1 (riskless ex-basis): {arb_ops}")
        print(f"  avg locked profit per such window: {(arb_profit/arb_ops) if arb_ops else 0:+.4f}")
        print(f"  NOTE: real only if both venues settle the SAME outcome; ~4% candle-vs-venue")
        print(f"        basis => some windows the two feeds disagree and the lock breaks.")
    print("\nRead: net/contract > 0 AND beyond the mean basis = structural relative-value edge.")


if __name__ == "__main__":
    main()
