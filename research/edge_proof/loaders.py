"""Streaming loaders for the crypto_strat forward-run data.

The forward data lives under
  <strat>/prediction_market_bots/data/forward/<run>/<ASSET>/{trades,decisions}/...

Decision files are ~300 MB per asset per day, so everything here streams line by
line and keeps only a compact projection of each record.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE"]


def find_runs(forward_dir: Path, prefix: str = "dashboard_conservative-validation_boltzmann_") -> list[Path]:
    runs = sorted(p for p in forward_dir.glob(f"{prefix}*") if p.is_dir())
    return runs


def _f(d: dict, *keys, default=None):
    """First present, non-null value among keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def iter_settled_trades(runs: list[Path]) -> Iterator[dict]:
    """Yield compact settled-trade rows (entered trades that reached settlement)."""
    for run in runs:
        for asset in ASSETS:
            tdir = run / asset / "trades"
            if not tdir.is_dir():
                continue
            for path in tdir.glob("btc15m_paper_trades_*.jsonl"):
                with path.open() as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if d.get("record_type") != "paper_trade_settled":
                            continue
                        yield {
                            "run": run.name,
                            "asset": asset,
                            "venue": d.get("venue"),
                            "side": _f(d, "side", "selected_side", "edge_polarity_selected_side"),
                            "model_prob_side": _f(d, "entry_probability", "raw_model_probability"),
                            "raw_model_probability": d.get("raw_model_probability"),
                            "calibrated_directional_probability": d.get("calibrated_directional_probability"),
                            "entry_price": _f(d, "actual_fill_price", "entry_price"),
                            "entry_fee_per_contract": d.get("entry_fee_per_contract"),
                            "fee_paid": d.get("fee_paid"),
                            "contracts": _f(d, "filled_contracts", "contracts", default=0.0),
                            "realized_pnl": d.get("realized_pnl"),
                            "fill_slippage_per_contract": d.get("fill_slippage_per_contract"),
                            "settlement_winning_side": d.get("settlement_winning_side"),
                            "settlement_observed_return": d.get("settlement_observed_return"),
                            "market_start_time": d.get("market_start_time"),
                            "market_end_time": d.get("market_end_time"),
                            "signal_time": d.get("signal_time"),
                        }


def iter_decisions(runs: list[Path], sample_every: int = 1, max_per_file: int | None = None) -> Iterator[dict]:
    """Yield compact decision rows for ALL decisions (entered + skipped).

    ``sample_every`` keeps 1 of every N records; ``max_per_file`` caps records per file.
    """
    for run in runs:
        for asset in ASSETS:
            ddir = run / asset / "decisions"
            if not ddir.is_dir():
                continue
            for path in ddir.glob("*paper_decisions_*.jsonl"):
                kept = seen = 0
                with path.open() as fh:
                    for line in fh:
                        if sample_every > 1 and (seen := seen + 1) % sample_every != 0:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if d.get("record_type") != "paper_decision":
                            continue
                        prob_up = _f(d, "probability_up", "raw_probability_up", "model_probability")
                        yield {
                            "run": run.name,
                            "asset": asset,
                            "venue": d.get("venue"),
                            "signal_time": d.get("signal_time"),
                            "market_start_time": d.get("market_start_time"),
                            "market_end_time": d.get("market_end_time"),
                            "time_to_settlement_minutes": d.get("time_to_settlement_minutes"),
                            "selected_side": _f(d, "selected_side", "side"),
                            "probability_up": prob_up,
                            "calibrated_probability_up": d.get("calibrated_probability_up"),
                            "model_probability": d.get("model_probability"),
                            "confidence": d.get("confidence"),
                            "should_enter": d.get("should_enter"),
                            "ev_qualified": d.get("ev_qualified"),
                            "decision_reason": d.get("decision_reason"),
                            "entry_price": d.get("entry_price"),
                            "observed_best_ask": d.get("observed_best_ask"),
                            "observed_best_bid": d.get("observed_best_bid"),
                            "max_bid": d.get("max_bid"),
                            "max_taker_entry_price": d.get("max_taker_entry_price"),
                            "fee_per_contract": _f(d, "fee_per_contract", "fee_cost_per_contract", default=0.0),
                            "spread_cost_per_contract": d.get("spread_cost_per_contract"),
                            "exec_cost_per_contract": d.get("estimated_total_execution_cost_per_contract"),
                            "quote_spread": d.get("quote_spread"),
                            "edge_combined_adjusted": d.get("edge_combined_adjusted"),
                            "historical_count": d.get("historical_count"),
                            "historical_wins": d.get("historical_wins"),
                        }
                        kept += 1
                        if max_per_file is not None and kept >= max_per_file:
                            break
