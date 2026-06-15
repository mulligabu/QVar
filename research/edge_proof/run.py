"""Edge-proof harness entry point.

Usage:
  python -m research.edge_proof.run --runs 5 --sample 20

Answers, from the existing crypto_strat forward data:
  1. Is the probability model calibrated / does it have directional skill? (all decisions)
  2. Where does realized money go — fees vs bad picks? (settled trades)
  3. Is there an edge bucket with positive realized net-of-fee EV? (settled trades)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.edge_proof.analyze import (
    brier,
    bucketize_by_edge,
    reliability_table,
)
from research.edge_proof.loaders import find_runs, iter_decisions, iter_settled_trades
from research.edge_proof.outcomes import OutcomeResolver, parse_ts, validate_against_settled

DEFAULT_STRAT = Path("/home/user/crypto_strat/prediction_market_bots/data")

# Poly crypto taker fee ~= shares * 0.072 * p * (1-p); maker ~= 0. Kalshi taker ~0.07*p(1-p).
def taker_fee(price: float, fee_rate: float = 0.072) -> float:
    return fee_rate * price * (1.0 - price)


def fmt(x, nd=4):
    return "    -   " if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-data", type=Path, default=DEFAULT_STRAT)
    ap.add_argument("--runs", type=int, default=5, help="most recent N forward runs")
    ap.add_argument("--sample", type=int, default=20, help="keep 1 of every N decisions")
    ap.add_argument("--max-per-file", type=int, default=None)
    args = ap.parse_args()

    forward_dir = args.strat_data / "forward"
    candles_dir = args.strat_data / "candles" / "binance_multiasset_model_dirs"
    runs = find_runs(forward_dir)[-args.runs:]
    print(f"# Edge-Proof Report")
    print(f"strat_data : {args.strat_data}")
    print(f"runs       : {len(runs)} (most recent {args.runs})")
    for r in runs:
        print(f"             - {r.name}")
    print(f"decision sampling: 1/{args.sample}")
    print()

    since = datetime.now(timezone.utc) - timedelta(days=45)
    resolver = OutcomeResolver(candles_dir, since=since)

    # ---- 1. Settled trades: realized money attribution -----------------------
    settled = list(iter_settled_trades(runs))
    print(f"## Settled trades: {len(settled)}")
    if settled:
        val = validate_against_settled(resolver, settled)
        print(f"label cross-check vs engine: agreement={fmt(val['agreement'],4)} on {val['checked']} windows")
        if val["mismatches_sample"]:
            print(f"  (mismatch sample: {val['mismatches_sample'][:3]})")

        tot_pnl = sum(r["realized_pnl"] or 0 for r in settled)
        tot_contracts = sum(r["contracts"] or 0 for r in settled)
        tot_fees = sum((r["fee_paid"] or 0) for r in settled)
        wins = sum(1 for r in settled if r["settlement_winning_side"] == r["side"])
        print(f"realized PnL total : {tot_pnl:+.2f}")
        print(f"contracts total    : {tot_contracts:,.0f}")
        print(f"fees paid total    : {tot_fees:+.2f}")
        print(f"win rate (entered) : {wins/len(settled):.4f}  ({wins}/{len(settled)})")
        # Gross (fee-free) realized: won*1 - entry_price, per contract.
        gross = 0.0
        for r in settled:
            won = 1.0 if r["settlement_winning_side"] == r["side"] else 0.0
            ep = r["entry_price"] or 0.0
            c = r["contracts"] or 0.0
            gross += (won - ep) * c
        print(f"gross PnL (pre-fee) : {gross:+.2f}   -> fee drag turns it to {gross - tot_fees:+.2f}")
        print()

        # Per-asset settled breakdown
        print("### Settled win-rate & realized net per contract, by asset")
        print(f"{'asset':6} {'n':>4} {'winrate':>8} {'avg_entry':>10} {'net/contract':>13} {'realized_pnl':>13}")
        by_asset: dict[str, list[dict]] = {}
        for r in settled:
            by_asset.setdefault(r["asset"], []).append(r)
        for asset, rs in sorted(by_asset.items()):
            w = sum(1 for r in rs if r["settlement_winning_side"] == r["side"])
            avg_entry = sum(r["entry_price"] or 0 for r in rs) / len(rs)
            net_pc = sum(
                ((1.0 if r["settlement_winning_side"] == r["side"] else 0.0)
                 - (r["entry_price"] or 0) - (r["entry_fee_per_contract"] or 0))
                for r in rs
            ) / len(rs)
            pnl = sum(r["realized_pnl"] or 0 for r in rs)
            print(f"{asset:6} {len(rs):4d} {w/len(rs):8.4f} {avg_entry:10.4f} {net_pc:+13.4f} {pnl:+13.2f}")
        print()

        # EV by stated edge bucket (settled trades, real prices)
        recs = []
        for r in settled:
            p = r["model_prob_side"]
            ep = r["entry_price"]
            fee = r["entry_fee_per_contract"] or 0.0
            if p is None or ep is None:
                continue
            won = 1 if r["settlement_winning_side"] == r["side"] else 0
            recs.append({
                "edge": p - ep - fee,
                "p_side": p,
                "won": won,
                "ask": ep,
                "bid": ep,            # settled: only the real fill price is known
                "fee_taker": fee,
                "fee_maker": 0.0,
                "asset": r["asset"],
            })
        if recs:
            print("### Realized net-of-fee EV by stated edge bucket (settled trades)")
            print("  edge = model_prob(side) - entry_price - fee ; net = realized payoff - entry - fee")
            print(f"{'edge_bucket':18} {'n':>4} {'avg_pred':>9} {'win_rate':>9} {'net/contract':>13}")
            buckets = bucketize_by_edge(recs, [-1.0, -0.05, 0.0, 0.02, 0.05, 0.10, 1.0])
            for b in buckets:
                if b.n == 0:
                    continue
                print(f"{b.label:18} {b.n:4d} {fmt(b.avg_pred,4):>9} {fmt(b.win_rate,4):>9} {fmt(b.net_taker_per_contract,4):>13}")
            print()

    # ---- 2. Live entry-time decisions: calibration & directional skill -------
    # The engine logs many rows per market as the window evolves; most are
    # diagnostic/out-of-window (tts>16, reason 'unknown'/'stale_signal') with
    # degenerate prob_up~0/1. Restrict to genuine entry-time evaluations and
    # dedupe to one row per (asset, market window) to avoid autocorrelation.
    print("## Live entry-time decisions: probability calibration (entered + skipped)")
    EXCLUDE = {"unknown", "stale_signal", "market_expired", "duplicate_open_market"}
    best_per_window: dict[tuple, dict] = {}
    n_dec = n_kept = 0
    for d in iter_decisions(runs, sample_every=args.sample, max_per_file=args.max_per_file):
        n_dec += 1
        tts = d["time_to_settlement_minutes"]
        reason = d["decision_reason"]
        if d["probability_up"] is None or tts is None or not (0 < tts <= 16):
            continue
        if reason in EXCLUDE:
            continue
        key = (d["asset"], d["market_start_time"])
        # earliest valid entry evaluation = largest tts within the window
        prev = best_per_window.get(key)
        if prev is None or tts > prev["time_to_settlement_minutes"]:
            best_per_window[key] = d

    cal_rows: list[tuple[float, int]] = []   # (p_up, won_up)
    n_labeled = 0
    side_skill = {"pred_up_n": 0, "pred_up_correct": 0, "pred_down_n": 0, "pred_down_correct": 0}
    for d in best_per_window.values():
        st, et = d["market_start_time"], d["market_end_time"]
        if not st or not et:
            continue
        out = resolver.resolve(d["asset"], parse_ts(st), parse_ts(et))
        if out is None:
            continue
        p_up = d["probability_up"]
        won_up = 1 if out[0] == "up" else 0
        cal_rows.append((p_up, won_up))
        n_labeled += 1
        if p_up >= 0.5:
            side_skill["pred_up_n"] += 1
            side_skill["pred_up_correct"] += won_up
        else:
            side_skill["pred_down_n"] += 1
            side_skill["pred_down_correct"] += (1 - won_up)
    n_kept = len(best_per_window)
    print(f"decisions scanned: {n_dec}   live entry-time windows: {n_kept}   labeled: {n_labeled}")
    if cal_rows:
        b = brier([(p, y) for p, y in cal_rows])
        base = sum(y for _, y in cal_rows) / len(cal_rows)
        print(f"Brier score      : {b:.5f}   (base-rate Brier = {base*(1-base):.5f}; lower is better)")
        print(f"realized up-rate : {base:.4f}")
        pu, puc = side_skill["pred_up_n"], side_skill["pred_up_correct"]
        pd, pdc = side_skill["pred_down_n"], side_skill["pred_down_correct"]
        print(f"directional skill: when model says UP   -> correct {puc}/{pu} = {puc/pu if pu else 0:.4f}")
        print(f"                   when model says DOWN -> correct {pdc}/{pd} = {pdc/pd if pd else 0:.4f}")
        print()
        print("### Reliability table (predicted P(up) vs realized up-rate)")
        print(f"{'bin':12} {'n':>6} {'pred':>8} {'actual':>8}")
        for row in reliability_table(cal_rows, bins=10):
            print(f"{row['bin']:12} {row['n']:6d} {fmt(row['pred'],4):>8} {fmt(row['actual'],4):>8}")
    print()
    print("## Verdict")
    print("See net/contract columns: any consistently-positive edge bucket = real edge.")
    print("Flat reliability (actual ~ const regardless of pred) = no directional skill.")


if __name__ == "__main__":
    main()
