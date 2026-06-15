"""Kill-switch watchdog for the disagreement gate (cron-driven, stdlib only).

Background (see memory disagreement-gate-killswitch.md): the live engines trade
`side = sign(p_cal - mid)`. That signal currently reads ANTI-predictive
(disagreement-AUC 0.368) but the read is confounded — every asset is falling, so
nearly all leans are down-leans. The open question only a NON-DOWN regime resolves:
does disagreement-AUC climb above ~0.5 when the tape isn't falling?

This script runs on a 3-day cron. It stays QUIET while we're still in a down
regime (just refreshes status). It raises a loud flag ONLY once we reach a genuine
clean up/chop tape — non-down spot drift AND a balanced/up realized OUTCOME up-rate
(>= CLEAN_UP_RATE). The 2026-06-10 deep dive showed drift turns days before the
SETTLED outcomes do, so a drift-only flag fired prematurely on a still-down-tilted
outcome panel; we now hold both engines until the outcomes themselves balance.

Outputs (in data/):
  switch_status.json        — machine-readable latest read (always written)
  switch_check.log          — append-only history
  SWITCH_DECISION_DUE.txt   — written ONLY when a decision is due; the loud flag
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "forward_live" / "forward_engine.db"
DATA = ROOT / "data"
WINDOW_DAYS = 3
MIN_AUC_SAMPLE = 150       # need enough scored markets for a trustworthy AUC
DOWN_DRIFT_PCT = -1.0      # per-asset move below this = "falling"
DOWN_REGIME_FRAC = 0.6     # >=60% of assets falling = still a down tape
# CLEAN-TAPE gate (2026-06-10, user "hold both, run one clean up-tape"): drift turning
# non-down is NOT enough — the deep dive showed drift flips days before the SETTLED
# outcomes do, so the disagreement-AUC keeps being measured on a down-tilted OUTCOME
# panel. Only call the decision DUE once the realized outcome up-rate is balanced/up,
# i.e. we've actually observed the gate resolve off a falling tape.
CLEAN_UP_RATE = 0.47       # outcome up-rate >= this = no longer a down-tilted tape
# 2026-06-14 fix: the original up-rate POOLED every scored market over the 3d window with
# equal weight per market. Market volume is wildly uneven across days (a single down-tape
# day had ~475 markets vs ~70-100 on the up days that followed), so a high-volume down day
# DOMINATED the pooled mean and masked a turn that was unmistakable per day (40% -> 84% up
# over 4 days). We now read the tape PER DAY: equal-weight the daily up-rates (so one fat
# down day can't drown out the turn) and require the MOST RECENT day to itself be up. A day
# needs MIN_DAY_SAMPLE scored markets to count (drops thin partial days).
MIN_DAY_SAMPLE = 20


def auc(pairs: list[tuple[float, int]]) -> float | None:
    """Rank (Mann-Whitney) AUC of score vs binary outcome. 0.5 = no skill."""
    pos = sorted(s for s, o in pairs if o == 1)
    neg = sorted(s for s, o in pairs if o == 0)
    if not pos or not neg:
        return None
    wins = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def analyze() -> dict:
    cutoff = (datetime.now(UTC) - timedelta(days=WINDOW_DAYS)).isoformat()
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5.0)

    # per-asset spot drift over the window -> is the tape still falling?
    drift = {}
    for asset, in conn.execute("SELECT DISTINCT asset FROM decisions WHERE ts>=?", (cutoff,)):
        row = conn.execute(
            "SELECT (SELECT spot FROM decisions WHERE asset=? AND ts>=? AND spot IS NOT NULL ORDER BY ts LIMIT 1),"
            "       (SELECT spot FROM decisions WHERE asset=? AND ts>=? AND spot IS NOT NULL ORDER BY ts DESC LIMIT 1)",
            (asset, cutoff, asset, cutoff)).fetchone()
        if row and row[0] and row[1]:
            drift[asset] = round(100.0 * (row[1] - row[0]) / row[0], 2)
    falling = [a for a, p in drift.items() if p < DOWN_DRIFT_PCT]
    down_regime = bool(drift) and len(falling) / len(drift) >= DOWN_REGIME_FRAC

    # disagreement-AUC over the window (current-regime read), with all-time fallback
    def dis_auc(since: str | None):
        q = ("SELECT p_cal, market_mid, outcome_up FROM calib "
             "WHERE outcome_up IS NOT NULL AND p_cal IS NOT NULL AND market_mid IS NOT NULL")
        if since:
            q += " AND ts>=?"
            rows = conn.execute(q, (since,)).fetchall()
        else:
            rows = conn.execute(q).fetchall()
        pairs = [(p - m, o) for p, m, o in rows]
        up_rate = (sum(o for _, _, o in rows) / len(rows)) if rows else None
        return auc(pairs), len(pairs), up_rate

    a_win, n_win, up_win = dis_auc(cutoff)
    a_all, n_all, up_all = dis_auc(None)
    # prefer the windowed read if it has the sample, else fall back to all-time
    if n_win >= MIN_AUC_SAMPLE:
        a, n, scope, up_pooled = a_win, n_win, f"last {WINDOW_DAYS}d", up_win
    else:
        a, n, scope, up_pooled = a_all, n_all, "all-time", up_all

    # PER-DAY outcome up-rate over the window (the volume-weight-robust regime read).
    day_rows = conn.execute(
        "SELECT substr(ts,1,10) d, COUNT(*) n, AVG(outcome_up) up FROM calib "
        "WHERE outcome_up IS NOT NULL AND p_cal IS NOT NULL AND market_mid IS NOT NULL "
        "AND ts>=? GROUP BY d ORDER BY d", (cutoff,)).fetchall()
    conn.close()
    per_day = [(d, int(dn), round(du, 3)) for d, dn, du in day_rows if dn >= MIN_DAY_SAMPLE]
    day_rates = [du for _, _, du in per_day]
    # equal-weight the days (one fat down day can't dominate) + the most-recent day's own rate
    up_daily_mean = sum(day_rates) / len(day_rates) if day_rates else None
    up_recent = day_rates[-1] if day_rates else None
    up_rate = up_daily_mean  # the regime-representative figure used everywhere below

    enough = n >= MIN_AUC_SAMPLE and a is not None
    # CLEAN tape = the outcomes themselves are no longer down-tilted. Per-day now: the
    # equal-weighted daily mean is up AND the most recent qualifying day is itself up (so we
    # fire on a real turn, not a stale up-day while today rolled back over). Drift turning
    # non-down leads the outcome tape by days, so we still require non-down drift too (in `due`).
    clean_tape = (up_daily_mean is not None and up_daily_mean >= CLEAN_UP_RATE
                  and up_recent is not None and up_recent >= CLEAN_UP_RATE)
    # A decision is DUE only off a genuinely clean up/chop OUTCOME window with a trustworthy
    # AUC. non-down drift + still-down outcomes -> keep WAITing (the deep-dive scenario).
    # clean + AUC<0.5 -> flip (kill rule). clean + AUC>=0.5 -> edge validated forward.
    due = (not down_regime) and enough and clean_tape
    rate_str = f"daily-mean {up_daily_mean:.2f}, latest {up_recent:.2f}" if up_recent is not None else "n/a"
    if due:
        verdict = ("FLIP THE SWITCH — clean up/chop tape (per-day up-rate "
                   f"{rate_str}) but disagreement-AUC still {a:.3f} (<0.5): pull the gate "
                   "`side=sign(p_cal-mid)` in live/forward_engine.evaluate_trade, per the kill rule."
                   if a < 0.5 else
                   f"EDGE VALIDATED — clean up/chop tape (per-day up-rate {rate_str}) AND "
                   f"disagreement-AUC {a:.3f} (>=0.5): the disagreement signal works off a falling "
                   "tape. Consider sizing up (carefully).")
    elif down_regime:
        verdict = (f"HOLD — still a down tape ({len(falling)}/{len(drift)} assets falling). "
                   f"AUC {a:.3f} read is down-confounded; not actionable yet.")
    elif not clean_tape:
        verdict = (f"WAIT — drift turned non-down but the OUTCOME tape is not cleanly up yet "
                   f"(per-day up-rate {rate_str}, need both >= {CLEAN_UP_RATE}). Holding both "
                   "engines per the 2026-06-10 'clean up-tape' decision; re-read when outcomes balance.")
    else:
        verdict = f"WAIT — clean tape but only {n} scored markets (<{MIN_AUC_SAMPLE}); AUC not yet trustworthy."

    return {"ts": datetime.now(UTC).isoformat(), "down_regime": down_regime,
            "assets_drift_pct": drift, "assets_falling": falling,
            "outcome_up_rate": round(up_rate, 3) if up_rate is not None else None,
            "outcome_up_rate_pooled": round(up_pooled, 3) if up_pooled is not None else None,
            "up_rate_recent": round(up_recent, 3) if up_recent is not None else None,
            "per_day_up_rate": per_day, "clean_tape": clean_tape,
            "disagreement_auc": round(a, 4) if a is not None else None,
            "auc_n": n, "auc_scope": scope, "decision_due": due, "verdict": verdict}


def main():
    try:
        st = analyze()
    except Exception as e:  # never let cron noise; record the error
        st = {"ts": datetime.now(UTC).isoformat(), "error": str(e),
              "decision_due": False, "verdict": f"check failed: {e}"}
    (DATA / "switch_status.json").write_text(json.dumps(st, indent=2))
    with (DATA / "switch_check.log").open("a") as f:
        f.write(json.dumps(st) + "\n")
    flag = DATA / "SWITCH_DECISION_DUE.txt"
    if st.get("decision_due"):
        flag.write_text(f"{st['ts']}\n\n{st['verdict']}\n\nDrift: {st.get('assets_drift_pct')}\n")
    elif flag.exists():
        flag.unlink()   # clear stale flag once we're back in a down regime
    print(st["verdict"])


if __name__ == "__main__":
    main()
