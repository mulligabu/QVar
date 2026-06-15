"""Counterfactual: what would realized PnL have been under selectivity rules?

Replays the *same* settled trades (same fills, same outcomes, same fees) but only
keeps the ones a candidate rule would have taken. This isolates how much of the
loss is overtrading vs genuinely bad picks. It does NOT model maker fills or new
entries — it's a strict subset of trades that actually happened, so it's a lower
bound on the benefit of selectivity (maker execution would add more).

Usage:
  python -m research.edge_proof.counterfactual --runs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research.edge_proof.loaders import find_runs, iter_settled_trades

DEFAULT_STRAT = Path("/home/user/crypto_strat/prediction_market_bots/data")


def realized_net_per_contract(r: dict) -> float | None:
    p = r["model_prob_side"]
    ep = r["entry_price"]
    fee = r["entry_fee_per_contract"] or 0.0
    if p is None or ep is None or r["settlement_winning_side"] not in ("up", "down"):
        return None
    won = 1.0 if r["settlement_winning_side"] == r["side"] else 0.0
    return won - ep - fee, p - ep - fee  # (net_per_contract, stated_edge)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-data", type=Path, default=DEFAULT_STRAT)
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()
    runs = find_runs(args.strat_data / "forward")[-args.runs:]
    settled = list(iter_settled_trades(runs))

    rules = {
        "ALL (baseline)": lambda r, e: True,
        "edge>=0.02": lambda r, e: e >= 0.02,
        "BTC only": lambda r, e: r["asset"] == "BTC",
        "BTC+ETH": lambda r, e: r["asset"] in ("BTC", "ETH"),
        "BTC+ETH & edge>=0.02": lambda r, e: r["asset"] in ("BTC", "ETH") and e >= 0.02,
        "BTC only & edge>=0.02": lambda r, e: r["asset"] == "BTC" and e >= 0.02,
        "edge>=0.02 (all assets)": lambda r, e: e >= 0.02,
    }

    print(f"# Selectivity counterfactual on {len(settled)} settled trades, {len(runs)} runs")
    print("Realized PnL per contract uses actual fills/fees/outcomes; per-trade PnL")
    print("scales net/contract by the contracts actually traded.\n")
    print(f"{'rule':28} {'trades':>7} {'win%':>6} {'net/contract':>13} {'~PnL(contracts)':>16}")
    for name, rule in rules.items():
        n = wins = 0
        sum_net = sum_pnl = 0.0
        for r in settled:
            res = realized_net_per_contract(r)
            if res is None:
                continue
            net, edge = res
            if not rule(r, edge):
                continue
            n += 1
            wins += 1 if r["settlement_winning_side"] == r["side"] else 0
            sum_net += net
            sum_pnl += net * (r["contracts"] or 0.0)
        if n == 0:
            continue
        print(f"{name:28} {n:7d} {wins/n*100:6.1f} {sum_net/n:+13.4f} {sum_pnl:+16.2f}")


if __name__ == "__main__":
    main()
