"""Test 2: settlement-feed basis / terminal determinism.

We don't capture the real venue settlement feeds (Chainlink for Poly, CF
Benchmarks BRTI 60s-avg for Kalshi), so we characterize the *magnitude* of the
edge that real-time feed access would unlock, using the underlying candles + book:

  (a) Terminal determinism: T minutes before close, how often does the current
      sign(spot - strike) already match the final outcome? If you can read the
      feed, betting that sign wins at that rate -- the gap above the book's
      implied accuracy is the latency edge.
  (b) Book lag: at the same instant, how accurate is the contract's own implied
      side (up_mid >= 0.5)? If spot-sign beats book-implied near expiry, the book
      is laggy and there is real-time-feed edge.
  (c) Last-60s flips: how often the final 60s flips sign(spot-strike). High flip
      rate = the 60s-average settlement (Kalshi) is materially different from a
      point read, i.e. genuine settlement-mechanic risk/edge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.edge_proof.loaders import find_runs
from research.edge_proof.markets import iter_quote_windows
from research.edge_proof.outcomes import OutcomeResolver

DEFAULT_STRAT = Path("/home/user/crypto_strat/prediction_market_bots/data")
LEADS = [1, 2, 3, 5, 8, 12]  # minutes before close


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-data", type=Path, default=DEFAULT_STRAT)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--venue", default="polymarket")
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE,BNB")
    args = ap.parse_args()

    runs = find_runs(args.strat_data / "forward")[-args.runs:]
    candles = args.strat_data / "candles" / "binance_multiasset_model_dirs"
    since = datetime.now(timezone.utc) - timedelta(days=20)
    resolver = OutcomeResolver(candles, since=since)
    assets = args.assets.split(",")

    # determinism[lead] = [spot_sign_correct, n]; booklag[lead] = [book_correct, n_with_book]
    det = {L: [0, 0] for L in LEADS}
    booklag = {L: [0, 0] for L in LEADS}
    flips = {"flip": 0, "n": 0}
    n_windows = 0

    for w in iter_quote_windows(runs, resolver, assets=assets):
        if w.outcome is None or w.strike is None:
            continue
        series = resolver._get(w.asset)
        if series is None:
            continue
        n_windows += 1
        won_up = w.outcome == "up"
        # (c) last-60s flip
        spot_end = series.price_at(datetime.fromtimestamp(w.end, tz=timezone.utc))
        spot_60 = series.price_at(datetime.fromtimestamp(w.end - 60, tz=timezone.utc))
        if spot_end is not None and spot_60 is not None:
            flips["n"] += 1
            if (spot_60 > w.strike) != (spot_end > w.strike):
                flips["flip"] += 1
        # (a)/(b) per lead
        snaps = w.snaps_by_venue.get(args.venue, [])
        for L in LEADS:
            t = w.end - L * 60
            spot = series.price_at(datetime.fromtimestamp(t, tz=timezone.utc))
            if spot is None:
                continue
            det[L][1] += 1
            if (spot > w.strike) == won_up:
                det[L][0] += 1
            # nearest snapshot at-or-before t (book implied side)
            near = None
            for s in snaps:
                if s.ts <= t + 5:
                    near = s
                else:
                    break
            if near is not None and near.up_mid is not None:
                booklag[L][1] += 1
                if (near.up_mid >= 0.5) == won_up:
                    booklag[L][0] += 1

    print("# Test 2 — Settlement-feed basis / terminal determinism")
    print(f"runs={args.runs} venue={args.venue} windows={n_windows}\n")
    print("Spot-sign vs book-implied accuracy by lead time (T min before close):")
    print(f"{'lead_min':9} {'n':>6} {'spot_sign_acc':>14} {'book_implied_acc':>17} {'edge(spot-book)':>16}")
    for L in LEADS:
        dn = det[L][1]; bn = booklag[L][1]
        sa = det[L][0] / dn if dn else None
        ba = booklag[L][0] / bn if bn else None
        edge = (sa - ba) if (sa is not None and ba is not None) else None
        print(f"{L:9d} {dn:6d} {(f'{sa:.4f}' if sa is not None else '-'):>14} "
              f"{(f'{ba:.4f}' if ba is not None else '-'):>17} {(f'{edge:+.4f}' if edge is not None else '-'):>16}")
    if flips["n"]:
        print(f"\nLast-60s sign flips: {flips['flip']}/{flips['n']} = {flips['flip']/flips['n']:.4f}")
    print("\nRead: spot_sign_acc near expiry = win rate if you could bet the live feed sign.")
    print("      edge(spot-book) > 0 = the book is laggy vs the underlying => real-time-feed edge.")


if __name__ == "__main__":
    main()
