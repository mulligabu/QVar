"""Engine-side adapters for the shared market-data service (live/md_service.py).

STDLIB ONLY — this module sits on the engines' decision path, which stays free
of third-party deps (hard rule; websockets/httpx live only in the standalone
services). Transport is a plain Unix-domain stream socket reading the broker's
NDJSON stream into an in-memory mirror on a background thread.

Control safety: every accessor degrades to None when the stream is stale or
dead, and MDFeedHub transparently falls back to the direct-REST FeedHub —
killing md_service must never stop an engine for more than one cycle. A
live-stream seq gap means dropped messages → reconnect (re-snapshots).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

from live.feed import FeedHub, Tick
from live.venues import BookTop


class MDMirror:
    """Local mirror of the md_service stream: ticks, chainlink rounds, books,
    strikes, discovery. Thread-safe reads (the engine's book fan-out pool calls
    book_top concurrently)."""

    def __init__(self, sock_path: str | Path = "data/md.sock", max_age: float = 10.0,
                 start: bool = True):
        self.sock_path = str(sock_path)
        self.max_age = max_age   # hb topic flows every ~2s — silence means dead
        self.lock = threading.Lock()
        self.connected = False
        self.last_msg_ts = 0.0
        self.msgs = 0
        self.reconnects = 0
        self.gaps = 0
        self.ticks: dict[str, dict] = {}
        self.cl: dict[str, dict] = {}
        self.books: dict[tuple[str, str], dict] = {}
        self.strikes: dict[tuple[str, int], dict] = {}
        self.disc: dict | None = None
        self.disc_ts = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="md-mirror")
        if start:
            self._thread.start()

    # ------------------------------------------------------------ transport
    def _run(self) -> None:
        while not self._stop:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(5.0)
                    s.connect(self.sock_path)
                    s.sendall(b'{"op":"subscribe","topics":[""]}\n')
                    s.settimeout(self.max_age)
                    self.connected = True
                    last_live_seq = 0
                    with s.makefile("rb") as f:
                        for line in f:
                            msg = json.loads(line)
                            if not msg.get("snapshot"):
                                seq = int(msg.get("seq") or 0)
                                if last_live_seq and seq != last_live_seq + 1:
                                    self.gaps += 1
                                    break   # dropped messages — resync via reconnect
                                last_live_seq = seq
                            self._apply(msg)
            except Exception:
                pass
            self.connected = False
            self.reconnects += 1
            time.sleep(1.0)

    def stop(self) -> None:
        self._stop = True

    def _apply(self, msg: dict) -> None:
        topic = str(msg.get("topic") or "")
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return
        now = time.time()
        with self.lock:
            self.last_msg_ts = now
            self.msgs += 1
            if topic.startswith("tick."):
                self.ticks[topic[5:]] = {**payload, "_rx": now, "_ts_src": msg.get("ts_src")}
            elif topic.startswith("chainlink."):
                self.cl[topic[10:]] = {**payload, "_rx": now}
            elif topic.startswith("book."):
                parts = topic.split(".", 2)
                if len(parts) == 3:
                    self.books[(parts[1], parts[2])] = {**payload, "_rx": now}
            elif topic.startswith("strike."):
                parts = topic.split(".")
                if len(parts) == 3:
                    try:
                        self.strikes[(parts[1], int(parts[2]))] = payload
                    except ValueError:
                        pass
            elif topic == "mkt.discovery":
                self.disc = payload
                self.disc_ts = now

    # ------------------------------------------------------------ accessors
    def fresh(self) -> bool:
        return self.connected and (time.time() - self.last_msg_ts) <= self.max_age

    def spot_tick(self, asset: str, max_age: float = 15.0) -> Tick | None:
        with self.lock:
            t = self.ticks.get(asset)
        if t is None or (time.time() - t["_rx"]) > max_age:
            return None
        try:
            return Tick(price=float(t["price"]), ts=float(t.get("_ts_src") or t["_rx"]),
                        source=str(t.get("source") or "md"))
        except Exception:
            return None

    def chainlink_tick(self, asset: str) -> Tick | None:
        """No age limit: rounds post on deviation, sparseness is normal — the
        composite blend keys off updated_at exactly like the direct feed."""
        with self.lock:
            t = self.cl.get(asset)
        if t is None:
            return None
        try:
            return Tick(price=float(t["price"]), ts=float(t["_rx"]), source="chainlink",
                        updated_at=float(t["updated_at"]))
        except Exception:
            return None

    def book_top(self, venue: str, market_id: str) -> BookTop | None:
        if not self.fresh():
            return None   # dead stream — caller falls back to direct REST
        with self.lock:
            b = self.books.get((venue, str(market_id)))
        if b is None:
            return None
        try:
            return BookTop(
                venue=venue, market_id=str(market_id),
                start_ts=float(b.get("start_ts") or 0.0), end_ts=float(b.get("end_ts") or 0.0),
                up_bid=None if b.get("up_bid") is None else float(b["up_bid"]),
                up_ask=None if b.get("up_ask") is None else float(b["up_ask"]),
                up_bid_size=float(b.get("up_bid_size") or 0.0),
                up_ask_size=float(b.get("up_ask_size") or 0.0),
                observed_at=float(b["_rx"]))
        except Exception:
            return None

    def strike(self, asset: str, window_start: int) -> float | None:
        with self.lock:
            s = self.strikes.get((asset, int(window_start)))
        if s is None:
            return None
        try:
            return float(s["price"])
        except Exception:
            return None

    def discovery(self, max_age: float = 180.0) -> dict | None:
        with self.lock:
            if self.disc is None or (time.time() - self.disc_ts) > max_age:
                return None
            return self.disc

    def stats(self) -> dict:
        return {"connected": self.connected,
                "age_s": round(time.time() - self.last_msg_ts, 1) if self.last_msg_ts else None,
                "msgs": self.msgs, "reconnects": self.reconnects, "gaps": self.gaps,
                "books": len(self.books), "strikes": len(self.strikes)}


class MDFeedHub:
    """FeedHub-compatible facade: same CompositeFeed anchor+drift blend, fed
    from the md_service mirror instead of per-engine HTTP, with automatic
    per-sample fallback to the direct-REST FeedHub when the stream is stale or
    dead. `.feeds` is shared with the inner hub, so last_chainlink/staleness
    and everything downstream behave identically on either transport."""

    def __init__(self, assets: list[str], mirror: MDMirror):
        self.inner = FeedHub(assets)
        self.assets = list(assets)
        self.mirror = mirror
        self.feeds = self.inner.feeds
        self.transport = "md"
        self.fallbacks = 0

    def sample(self) -> dict[str, Tick | None]:
        if not self.mirror.fresh():
            if self.transport != "rest":
                self.transport = "rest"
                self.fallbacks += 1
                print("# [md] stream stale/dead -> direct REST fallback")
            return self.inner.sample()
        if self.transport != "md":
            self.transport = "md"
            print("# [md] stream restored -> back on md_service")
        out: dict[str, Tick | None] = {}
        for a in self.assets:
            out[a] = self.feeds[a].get(cl=self.mirror.chainlink_tick(a),
                                       cb=self.mirror.spot_tick(a), prefetched=True)
        return out

    def staleness(self) -> dict[str, float | None]:
        return self.inner.staleness()
