"""SQLite store for the forward engine — one DB for clean analytics.

Replaces the sprawl of dated JSONL files with a single
``data/forward_live/forward_engine.db`` holding decisions, settlements, and the
per-cycle book/underlying path. WAL mode so the dashboard (separate process) can
read concurrently while the engine writes.

Query examples for analytics:
  SELECT asset, tf, COUNT(*), AVG(won) FROM settlements WHERE filled=1 GROUP BY asset, tf;
  SELECT regime, AVG(ev), COUNT(*) FROM decisions WHERE decision='TRADE' GROUP BY regime;
  SELECT ts, cum_pnl FROM settlements WHERE filled=1 ORDER BY rowid;
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY, ts TEXT, market TEXT, ticker TEXT, asset TEXT, venue TEXT, tf TEXT, kind TEXT,
  mins_left REAL, spot REAL, strike REAL, z_dist REAL, p_up REAL, p_cal REAL, side TEXT,
  move_prob REAL, dir_edge REAL, fv_resid REAL, regime TEXT, temp REAL, confidence REAL,
  book_mid_up REAL, limit_price REAL, ev REAL, decision TEXT,
  alpha_up REAL, alpha_naive REAL, struct_div REAL, p_anchored REAL, gate_margin REAL,
  sigma_s REAL, kappa REAL, struct_regime TEXT);
CREATE TABLE IF NOT EXISTS settlements(
  id INTEGER PRIMARY KEY, ts TEXT, market TEXT, ticker TEXT, asset TEXT, venue TEXT, tf TEXT, side TEXT,
  strike REAL, end_price REAL, outcome_up INTEGER, won INTEGER, filled INTEGER,
  fill_price REAL, venue_outcome_up INTEGER, recon TEXT, pnl REAL, cum_pnl REAL,
  settled INTEGER, fill_rate REAL, winrate REAL,
  agg_size REAL, pass_size REAL, pass_filled INTEGER,
  tp_pnl REAL, tp_cum_pnl REAL, tp_exited INTEGER, peak_bid REAL, trough_bid REAL,
  peak_mins_left REAL, post_peak_low REAL,
  peak_spot REAL, peak_spread REAL, peak_bid_size REAL, peak_ask_size REAL);
CREATE TABLE IF NOT EXISTS paths(
  id INTEGER PRIMARY KEY, ts REAL, market TEXT, asset TEXT, underlying REAL,
  up_bid REAL, up_ask REAL, side TEXT, limit_price REAL,
  up_bid_size REAL, up_ask_size REAL, mins_left REAL);
-- calibration: every EVALUATED market at settlement (not just traded) — the
-- markov+boltzmann edge test (does model/structural beat the market mid?).
CREATE TABLE IF NOT EXISTS calib(
  id INTEGER PRIMARY KEY, ts TEXT, market TEXT, asset TEXT, tf TEXT,
  market_mid REAL, p_cal REAL, alpha_up REAL, outcome_up INTEGER);
-- system health (institutional observability): cycle latency, API error budget,
-- feed staleness — one row per ~minute so stalls/degradation are visible in data,
-- not just in a scrolled-away console.
CREATE TABLE IF NOT EXISTS health(
  id INTEGER PRIMARY KEY, ts TEXT, cycle_ms REAL, cycle_p95_ms REAL,
  open_pos INTEGER, markets INTEGER, api_calls INTEGER, api_errors INTEGER,
  max_cl_staleness_s REAL, halted INTEGER);
CREATE INDEX IF NOT EXISTS ix_dec_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS ix_dec_asset ON decisions(asset, tf);
CREATE INDEX IF NOT EXISTS ix_set_ts ON settlements(ts);
CREATE INDEX IF NOT EXISTS ix_set_filled ON settlements(filled);
CREATE INDEX IF NOT EXISTS ix_path_market ON paths(market);
"""

DEC_COLS = ["ts", "market", "ticker", "asset", "venue", "tf", "kind", "mins_left", "spot", "strike",
            "z_dist", "p_up", "p_cal", "side", "move_prob", "dir_edge", "fv_resid", "regime", "temp",
            "confidence", "book_mid_up", "limit_price", "ev", "decision",
            "alpha_up", "alpha_naive", "struct_div", "p_anchored", "gate_margin",
            "sigma_s", "kappa", "struct_regime", "halted", "block_reason"]
SET_COLS = ["ts", "market", "ticker", "asset", "venue", "tf", "side", "strike", "end_price", "outcome_up",
            "won", "filled", "fill_price", "venue_outcome_up", "recon", "pnl", "cum_pnl",
            "settled", "fill_rate", "winrate", "agg_size", "pass_size", "pass_filled",
            "tp_pnl", "tp_cum_pnl", "tp_exited", "peak_bid", "trough_bid",
            "peak_mins_left", "post_peak_low",
            "peak_spot", "peak_spread", "peak_bid_size", "peak_ask_size"]
PATH_COLS = ["ts", "market", "asset", "underlying", "up_bid", "up_ask", "side", "limit_price",
             "up_bid_size", "up_ask_size", "mins_left"]
CALIB_COLS = ["ts", "market", "asset", "tf", "market_mid", "p_cal", "alpha_up", "outcome_up"]
HEALTH_COLS = ["ts", "cycle_ms", "cycle_p95_ms", "open_pos", "markets",
               "api_calls", "api_errors", "max_cl_staleness_s", "halted"]


class Store:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        # additive migrations for pre-existing DBs (ignore if column already exists)
        migrations = [("decisions", "ticker", "TEXT"), ("settlements", "ticker", "TEXT"),
                      ("decisions", "side", "TEXT")]
        # structural-anchor shadow columns (research/boltzmann_upgrade.md, qv_markov_upgrade.md)
        migrations += [("decisions", c, "REAL") for c in
                       ("alpha_up", "alpha_naive", "struct_div", "p_anchored",
                        "gate_margin", "sigma_s", "kappa")]
        migrations += [("decisions", "struct_regime", "TEXT")]
        # conviction-scaled execution split (2026-06-03): aggressive/passive leg sizes
        migrations += [("settlements", "agg_size", "REAL"), ("settlements", "pass_size", "REAL"),
                       ("settlements", "pass_filled", "INTEGER")]
        # ghost take-profit overlay (2026-06-03): shadow ledger exiting at tp_level.
        # tp_pnl = ghost pnl for this settlement, tp_cum_pnl = running ghost total
        # (seeded at real cum_pnl at first launch so ghost equity starts = real equity),
        # tp_exited = 1 if the TP fired (sold early) vs 0 held to settlement. See memory
        # take-profit-result (TP@0.90 = the one exit rule that beats holding).
        migrations += [("settlements", "tp_pnl", "REAL"), ("settlements", "tp_cum_pnl", "REAL"),
                       ("settlements", "tp_exited", "INTEGER")]
        # per-trade reversal excursion: peak_bid = max our-side sellable price reached
        # while open, trough_bid = min. With won/fill_price these let us sweep the
        # optimal TP (and SL) empirically — at what favorable price do winners/losers
        # actually peak before reversing. The `paths` table holds the time-resolved shape.
        migrations += [("settlements", "peak_bid", "REAL"), ("settlements", "trough_bid", "REAL")]
        # pivot characterization: peak_mins_left = mins-to-settle WHEN the peak (the turn)
        # occurred; post_peak_low = lowest our-side price reached AFTER that peak (reversal
        # depth = peak_bid − post_peak_low). paths gets order-flow sizes (imbalance is the
        # fast leading indicator of a pivot). Together they locate the pivot in price+time.
        migrations += [("settlements", "peak_mins_left", "REAL"), ("settlements", "post_peak_low", "REAL")]
        migrations += [("paths", "up_bid_size", "REAL"), ("paths", "up_ask_size", "REAL"),
                       ("paths", "mins_left", "REAL")]
        # full state snapshot AT THE PIVOT (the regression target for trades that reverse
        # against us): underlying, spread (liquidity), and book sizes at the moment the
        # peak formed. Everything else (z, gamma, OFI, microprice, RV) derives offline
        # from these + the raw paths series — log the substrate, don't pre-commit features.
        migrations += [("settlements", "peak_spot", "REAL"), ("settlements", "peak_spread", "REAL"),
                       ("settlements", "peak_bid_size", "REAL"), ("settlements", "peak_ask_size", "REAL")]
        # control plane: rows blocked by estop/halt/max-loss-trip/exposure-cap
        # keep the would-have-traded signal so blocked windows stay
        # counterfactually scored. block_reason = which control fired.
        migrations += [("decisions", "halted", "INTEGER"),
                       ("decisions", "block_reason", "TEXT")]
        for table, col, typ in migrations:
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    def settlement_totals(self) -> dict:
        """Restore running totals on restart so equity/warmup don't reset."""
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN filled=1 THEN pnl END),0), "
            "COALESCE(SUM(CASE WHEN filled=1 THEN won END),0), "
            "COALESCE(SUM(filled),0), COALESCE(SUM(CASE WHEN filled=0 THEN 1 END),0) "
            "FROM settlements")
        pnl, wins, fills, no_fills = cur.fetchone()
        return {"pnl": pnl or 0.0, "wins": int(wins or 0),
                "settled": int(fills or 0), "no_fills": int(no_fills or 0)}

    def pnl_today(self) -> float:
        """Realized PnL since UTC midnight (input to the max-loss breaker)."""
        from datetime import UTC, datetime
        midnight = datetime.now(UTC).strftime("%Y-%m-%dT00:00:00")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM settlements WHERE filled=1 AND ts >= ?",
            (midnight,)).fetchone()
        return float(row[0] or 0.0)

    def tp_totals(self) -> dict:
        """Resume the ghost take-profit ledger across restarts. cum = the latest
        running ghost total (None if the overlay has no settlements yet, so the
        engine can seed it to the real cum_pnl). A ghost trade 'won' if it took
        profit (tp_exited=1) or held to a real win."""
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(tp_exited),0), COUNT(tp_pnl), "
            "COALESCE(SUM(CASE WHEN tp_exited=1 OR (tp_exited=0 AND won=1) THEN 1 ELSE 0 END),0) "
            "FROM settlements WHERE tp_pnl IS NOT NULL")
        exits, settled, wins = cur.fetchone()
        row = self.conn.execute(
            "SELECT tp_cum_pnl FROM settlements WHERE tp_cum_pnl IS NOT NULL "
            "ORDER BY id DESC LIMIT 1").fetchone()
        return {"exits": int(exits or 0), "settled": int(settled or 0),
                "wins": int(wins or 0), "cum": row[0] if row else None}

    def _insert(self, table: str, cols: list[str], rec: dict):
        vals = [rec.get(c) for c in cols]
        ph = ",".join("?" * len(cols))
        self.conn.execute(f"INSERT INTO {table}({','.join(cols)}) VALUES({ph})", vals)
        self.conn.commit()

    def insert_decision(self, rec: dict):
        self._insert("decisions", DEC_COLS, rec)

    def insert_settlement(self, rec: dict):
        self._insert("settlements", SET_COLS, rec)

    def insert_path(self, rec: dict):
        self._insert("paths", PATH_COLS, rec)

    def insert_calib(self, rec: dict):
        self._insert("calib", CALIB_COLS, rec)

    def insert_health(self, rec: dict):
        self._insert("health", HEALTH_COLS, rec)


class ReadStore:
    """Read-only accessor for the dashboard (separate process; WAL allows it)."""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=5.0)
        self.conn.row_factory = sqlite3.Row

    def recent(self, table: str, limit: int = 50) -> list[dict]:
        try:
            cur = self.conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error:
            return []

    def recent_settlements(self, limit: int = 60) -> list[dict]:
        """Recent FILLED settlements only. The dashboard's settled-trades panel shows
        actual trades; pulling generic recent rows let no-fill rows fill the window and
        hide older real trades (e.g. the settle engine's up bets fell off the list)."""
        try:
            cur = self.conn.execute(
                "SELECT * FROM settlements WHERE filled=1 ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error:
            return []

    def throughput(self) -> dict:
        """All-time per-timeframe and per-asset activity for the dashboard coverage
        panel: decisions, trades, and filled settlements. Lets thin lanes (e.g. 1h)
        stay visible instead of falling off the recent-40 window."""
        out: dict[str, dict] = {"by_tf": {}, "by_asset": {}}
        try:
            cur = self.conn.execute(
                "SELECT tf, COUNT(*) n, SUM(decision='TRADE') trades FROM decisions GROUP BY tf")
            for r in cur.fetchall():
                out["by_tf"][r["tf"]] = {"decisions": r["n"], "trades": r["trades"] or 0,
                                         "settled": 0, "winrate": None, "pnl": 0.0}
            cur = self.conn.execute(
                "SELECT tf, COUNT(*) settled, AVG(won) winrate, SUM(pnl) pnl "
                "FROM settlements WHERE filled=1 GROUP BY tf")
            for r in cur.fetchall():
                d = out["by_tf"].setdefault(r["tf"], {"decisions": 0, "trades": 0})
                d["settled"] = r["settled"]; d["winrate"] = r["winrate"]; d["pnl"] = r["pnl"] or 0.0
            cur = self.conn.execute(
                "SELECT asset, COUNT(*) n, SUM(decision='TRADE') trades FROM decisions GROUP BY asset")
            for r in cur.fetchall():
                out["by_asset"][r["asset"]] = {"decisions": r["n"], "trades": r["trades"] or 0,
                                               "settled": 0, "winrate": None, "pnl": 0.0}
            cur = self.conn.execute(
                "SELECT asset, COUNT(*) settled, AVG(won) winrate, SUM(pnl) pnl "
                "FROM settlements WHERE filled=1 GROUP BY asset")
            for r in cur.fetchall():
                d = out["by_asset"].setdefault(r["asset"], {"decisions": 0, "trades": 0})
                d["settled"] = r["settled"]; d["winrate"] = r["winrate"]; d["pnl"] = r["pnl"] or 0.0
        except sqlite3.Error:
            pass
        return out

    def equity_curve(self) -> list[dict]:
        try:
            cur = self.conn.execute(
                "SELECT ts, cum_pnl, tp_cum_pnl FROM settlements WHERE filled=1 ORDER BY id")
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error:
            return []

    def calibration(self, limit: int = 8000) -> dict:
        """The markov+boltzmann edge test: Brier + log-loss of the market mid vs our
        ensemble (p_cal) vs the structural alpha, scored against realized outcomes
        over all evaluated markets. Lower beats. If model/alpha < market -> we price
        better than the market (edge). Computed in Python (no SQLite math funcs)."""
        import math
        try:
            rows = self.conn.execute(
                "SELECT market_mid, p_cal, alpha_up, outcome_up FROM calib "
                "WHERE outcome_up IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.Error:
            return {"n": 0, "ready": False}
        if not rows:
            return {"n": 0, "ready": False}

        def auc(po):
            """Rank (Mann-Whitney) AUC: P(score_up > score_down). 0.5 = no skill."""
            import bisect
            pos = sorted(p for p, o in po if o == 1)
            neg = sorted(p for p, o in po if o == 0)
            if not pos or not neg:
                return None
            wins = 0.0
            for p in pos:
                lo = bisect.bisect_left(neg, p)
                hi = bisect.bisect_right(neg, p)
                wins += lo + 0.5 * (hi - lo)
            return round(wins / (len(pos) * len(neg)), 4)

        def score(col):
            po = [(r[col], r["outcome_up"]) for r in rows if r[col] is not None]
            if not po:
                return None
            n = len(po)
            brier = sum((p - o) ** 2 for p, o in po) / n
            ll = sum(-(o * math.log(min(max(p, 1e-6), 1 - 1e-6))
                       + (1 - o) * math.log(min(max(1 - p, 1e-6), 1 - 1e-6))) for p, o in po) / n
            return {"n": n, "brier": round(brier, 4), "logloss": round(ll, 4), "auc": auc(po)}

        market = score("market_mid")
        ens = score("p_cal")
        struct = score("alpha_up")
        # DIRECTIONAL-SKILL readout: does the engine's traded signal — sign(p_cal − mid),
        # i.e. its disagreement with the market — actually predict the residual outcome?
        # This is the live edge test, and it scores up-side skill the filled-only
        # settlements table can't (only ~25 fills vs all evaluated markets here). AUC of
        # (p_cal − mid) > 0.5 ⇒ disagreement is predictive (real alpha); ≈0.5 ⇒ no edge,
        # P&L is regime beta; < 0.5 ⇒ anti-predictive (the signal is backwards).
        dis = [(r["p_cal"] - r["market_mid"], r["outcome_up"]) for r in rows
               if r["p_cal"] is not None and r["market_mid"] is not None]
        up_lean = [(p, o) for p, o in dis if p > 0]
        dn_lean = [(p, o) for p, o in dis if p <= 0]
        directional = {
            "n": len(dis),
            "disagreement_auc": auc(dis),
            "up_lean_n": len(up_lean),
            "up_lean_realized_up": round(sum(o for _, o in up_lean) / len(up_lean), 3) if up_lean else None,
            "dn_lean_n": len(dn_lean),
            "dn_lean_realized_up": round(sum(o for _, o in dn_lean) / len(dn_lean), 3) if dn_lean else None,
        }
        verdict = "accumulating…"
        if market and ens and ens["n"] >= 30:
            beats = []
            if ens["brier"] < market["brier"]:
                beats.append("ensemble")
            if struct and struct["brier"] < market["brier"]:
                beats.append("structural α")
            verdict = (f"{' & '.join(beats)} beat the market (Brier)" if beats
                       else "market still prices best — no edge yet")
        return {"n": len(rows), "ready": len(rows) >= 30,
                "market": market, "ensemble": ens, "structural": struct,
                "directional": directional, "verdict": verdict}
