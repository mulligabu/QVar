"""Lightweight live dashboard for the forward prediction engine (stdlib only).

Serves the auto-refreshing dashboard page plus the engine's live JSON/JSONL so you
can watch real decisions, open positions, settled outcomes, PnL, and the online
model's calibration update in real time.

    python -m live.dashboard_server           # http://127.0.0.1:8011
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from live.control import clear_trip, lanes_snapshot, load_control, read_trip, save_control
from live.db import ReadStore

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"   # control plane (CONTROL.json / TRIPPED.json / HALT)
DATA = ROOT.parent / "data" / "forward_live"
DB = DATA / "forward_engine.db"
COND_DATA = ROOT.parent / "data" / "forward_live_conditional"   # conditional-skill cell gate
COND_DB = COND_DATA / "forward_engine.db"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_engine_json(self, name: str, data_dir: Path, db: Path):
        """Serve one of {status,equity,settlements,decisions,throughput,calib} for a
        given engine's data dir / DB. Used for both main and the conditional engine."""
        if name == "status":
            f = data_dir / "status.json"
            self._send(f.read_bytes() if f.exists() else b"{}", "application/json")
            return
        body = b"{}" if name in ("throughput", "calib") else b"[]"
        if db.exists():
            try:
                rs = ReadStore(db)
                data: object
                if name == "equity":
                    data = rs.equity_curve()
                elif name == "throughput":
                    data = rs.throughput()
                elif name == "calib":
                    data = rs.calibration()
                elif name == "settlements":
                    data = rs.recent_settlements(limit=60)   # filled-only, full window
                else:                       # decisions
                    data = rs.recent(name, limit=60)
                body = json.dumps(data, default=float).encode()
                rs.conn.close()
            except Exception:
                pass
        self._send(body, "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        # control plane: switch states + latched trip + global aggregates (all lanes)
        if path == "/control.json":
            body = json.dumps({
                "control": load_control(DATA_ROOT),
                "tripped": read_trip(DATA_ROOT),
                "halt_file": (DATA_ROOT / "HALT").exists(),
                "agg": lanes_snapshot(DATA_ROOT),
            }, default=float).encode()
            self._send(body, "application/json")
            return
        # same dashboard page for all engines; the page detects which via its URL
        if path in ("/", "/index.html",
                    "/conditional", "/conditional/", "/conditional.html"):
            self._send((ROOT / "dashboard.html").read_bytes(), "text/html")
            return
        # engine-prefixed data: /x/ conditional, else main
        if path.startswith("/x/"):
            data_dir, db, name = COND_DATA, COND_DB, path[3:].split(".")[0]
        else:
            data_dir, db, name = DATA, DB, path.strip("/").split(".")[0]
        if name in ("status", "equity", "settlements", "decisions", "throughput", "calib"):
            self._serve_engine_json(name, data_dir, db)
            return
        if path == "/leadlag.json":
            # separate experiment: raw-feed-vs-book lead-lag probe (research/leadlag.db)
            body = b'{"ready": false}'
            try:
                import sys as _sys
                rs = str(ROOT.parent / "research" / "strategy")
                if rs not in _sys.path:
                    _sys.path.insert(0, rs)
                from leadlag_collect import summary as _ll
                body = json.dumps(_ll(), default=float).encode()
            except Exception as e:
                body = json.dumps({"ready": False, "error": str(e)}).encode()
            self._send(body, "application/json")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Operator toggles (localhost-only server). Engines re-read the control
        file every cycle, so changes take effect on the next cycle (~seconds)."""
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if n else {}
        except Exception:
            body = {}
        if path == "/control/set":
            # stamp who/when on an estop transition so the banner explains itself
            if "estop" in body:
                cur = load_control(DATA_ROOT)
                if body["estop"] and not cur.get("estop"):
                    body.setdefault("estop_reason", "operator (dashboard)")
                    body.setdefault("estop_ts", datetime.now(UTC).isoformat())
                elif not body["estop"]:
                    body["estop_reason"] = None
                    body["estop_ts"] = None
            state = save_control(DATA_ROOT, body)
            self._send(json.dumps(state).encode(), "application/json")
        elif path == "/control/clear-trip":
            clear_trip(DATA_ROOT)
            self._send(b'{"ok": true}', "application/json")
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"# Forward-engine dashboard: http://127.0.0.1:{args.port}  (data: {DATA})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
