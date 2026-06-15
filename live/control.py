"""Cross-process risk control plane (stdlib only).

One shared file, data/CONTROL.json, is the operator's switchboard — written
atomically by the dashboard's toggle endpoints (or by hand / `touch data/HALT`),
re-read by EVERY engine at the top of EVERY cycle, so a toggle takes effect on
the next cycle (~seconds). Two distinct kill switches per the operator spec:

  estop         EMERGENCY STOP — something is broken/bugged: engines freeze
                entirely (no decisions, no settlements, no learning, no DB
                writes except a status heartbeat). Recoverable: toggling off
                resumes the loop exactly where it was. NOTE: positions that
                expire while frozen settle AFTER resume at the then-current
                Chainlink price; reconciliation flags any divergence.
  halt_entries  soft halt — stop opening NEW positions; settling/learning
                continue (also triggered by the legacy data/HALT file).

Plus two wired risk controls, each independently toggleable + adjustable live:

  max_loss      global realized-PnL breaker across ALL lanes (UTC day window).
                On breach it LATCHES data/TRIPPED.json — trading stays blocked
                even if PnL recovers, until the operator clears the trip from
                the dashboard. Blocks new entries only (open positions must
                still settle out).
  exposure_cap  cross-lane open-exposure governor: total committed $ across
                all lanes ≤ max_total_frac of combined equity, and per-asset
                committed $ ≤ max_asset_frac (the 5-lanes-one-correlated-bet
                gap). Checked per new trade; breaching trades are SKIPped and
                logged with block_reason='exposure'.

Lane aggregation reads each lane's status.json (written every cycle by every
engine). Stale lanes (no heartbeat for STALE_S) are excluded — visibility is
reported so the dashboard can show coverage honestly.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

DEFAULTS: dict = {
    "estop": False,
    "estop_reason": None,   # who/why: "operator (dashboard)" or "auto: lane 'x' ..."
    "estop_ts": None,
    "halt_entries": False,
    "max_loss": {"enabled": True, "limit": 150.0},   # $ realized loss, all lanes, UTC day
    "exposure_cap": {"enabled": True, "max_total_frac": 0.10, "max_asset_frac": 0.05},
}
STALE_S = 600.0   # lane status older than this is excluded from aggregates


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


def control_path(data_root: Path) -> Path:
    return Path(data_root) / "CONTROL.json"


def trip_path(data_root: Path) -> Path:
    return Path(data_root) / "TRIPPED.json"


def load_control(data_root: Path) -> dict:
    """Current control state = DEFAULTS overlaid with data/CONTROL.json.
    Unreadable/missing file -> defaults (controls fail to their default state,
    never to 'everything off')."""
    try:
        return _merge(DEFAULTS, json.loads(control_path(data_root).read_text()))
    except Exception:
        return deepcopy(DEFAULTS)


def save_control(data_root: Path, updates: dict) -> dict:
    """Deep-merge `updates` into the persisted control file (atomic tmp+replace
    so engines never read a torn write). Returns the new state."""
    state = _merge(load_control(data_root), updates)
    p = control_path(data_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)
    return state


def read_trip(data_root: Path) -> dict | None:
    """The latched max-loss trip, or None. {ts, reason, day_pnl, limit}."""
    try:
        return json.loads(trip_path(data_root).read_text())
    except Exception:
        return None


def trip(data_root: Path, reason: str, day_pnl: float, limit: float) -> dict:
    """Latch the breaker (first writer wins; subsequent calls keep the original
    trip record so the root cause isn't overwritten)."""
    existing = read_trip(data_root)
    if existing:
        return existing
    rec = {"ts": datetime.now(UTC).isoformat(), "reason": reason,
           "day_pnl": round(day_pnl, 2), "limit": limit}
    p = trip_path(data_root)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(p)
    return rec


def clear_trip(data_root: Path) -> None:
    trip_path(data_root).unlink(missing_ok=True)


def lanes_snapshot(data_root: Path, exclude: str | None = None) -> dict:
    """Aggregate every FRESH lane's status.json under data/: realized PnL today,
    equity, and committed open exposure (total + per asset). `exclude` skips one
    lane by name (an engine uses its own live numbers instead of its possibly
    stale status file). Lanes running pre-control code lack pnl_today/cost —
    they contribute equity/exposure fallbacks and are listed in `partial`."""
    lanes, partial = [], []
    total_day_pnl, total_equity, total_cost = 0.0, 0.0, 0.0
    by_asset: dict[str, float] = {}
    now = time.time()
    for sp in sorted(Path(data_root).glob("forward_live*/status.json")):
        try:
            s = json.loads(sp.read_text())
        except Exception:
            continue
        name = (s.get("config") or {}).get("name") or sp.parent.name
        if exclude is not None and name == exclude:
            continue
        try:
            upd = datetime.fromisoformat(s.get("updated", "")).timestamp()
        except Exception:
            continue
        if now - upd > STALE_S:
            continue
        perf = s.get("performance") or {}
        day = perf.get("pnl_today")
        if day is None:
            partial.append(name)
        total_day_pnl += day or 0.0
        total_equity += perf.get("equity") or 0.0
        lane_cost = 0.0
        for pos in s.get("open_positions") or []:
            cost = pos.get("cost")
            if cost is None:   # pre-control engine: committed ≈ size × entry price
                cost = (pos.get("size") or 0.0) * (pos.get("fill_price") or 0.0)
            lane_cost += cost
            a = pos.get("asset") or "?"
            by_asset[a] = by_asset.get(a, 0.0) + cost
        total_cost += lane_cost
        lanes.append({"name": name, "pnl_today": day, "equity": perf.get("equity"),
                      "open_cost": round(lane_cost, 2),
                      "open_positions": len(s.get("open_positions") or [])})
    return {"lanes": lanes, "partial": partial,
            "day_pnl": round(total_day_pnl, 2), "equity": round(total_equity, 2),
            "open_cost": round(total_cost, 2),
            "by_asset": {a: round(c, 2) for a, c in sorted(by_asset.items())}}
