"""Shared market-data service: ONE process owns ALL external market I/O.

Each of the 5 engines used to poll feeds/discovery/books independently (5×
redundant API load; fill checks see the book only once per 8-22s cycle).
md_service owns the venue connections — WebSocket-first — and republishes
everything over a Unix-domain-socket pub/sub stream plus a SQLite snapshot
(data/md.db) for late joiners and the dashboard. Engines consume it through
live/md_client.py adapters that fall back to today's direct REST automatically
if this process dies (control safety: killing this window degrades engines to
polling, it never stops them).

Sources:
  Polymarket books  CLOB WSS market channel (book / price_change /
                    last_trade_price), resubscribe on discovery change
  Kalshi books      REST poll ~2s through live.net (WS v2 needs RSA creds;
                    REST fallback until they exist)
  Spot              Coinbase WS ticker + Binance WS miniTicker (BNB/HYPE gaps)
  Chainlink         batched latestRoundData poll (1.5s) PLUS AnswerUpdated
                    eth_getLogs → the exact round sequence (roundId, answer,
                    updatedAt) bracketing every window boundary → ONE
                    authoritative Chainlink-round-exact strike table shared
                    by all lanes
  Discovery         the existing PolymarketClient/KalshiClient sweeps, one
                    60s loop for everyone

Topics: tick.{asset}, book.{venue}.{market_id}, trade.{venue}.{market_id},
chainlink.{asset}, strike.{asset}.{window_start}, mkt.discovery, hb.
Messages: NDJSON {seq, ts_src, ts_recv, topic, payload}; snapshot-on-subscribe
(flagged "snapshot": true); a live-stream seq gap on the client side means
messages were dropped → reconnect (which re-snapshots).

This is a NEW standalone service so it MAY use `websockets` (audit §2.4); the
engines' decision path stays stdlib — see live/md_client.py.

Run: python -m live.md_service            (boot_engines.sh window `md`)
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import contextlib
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect as ws_connect

from live import net
from live.feed import CHAINLINK_FEEDS, BinanceFeed, CoinbaseFeed, batch_chainlink
from live.venues import BookTop, KalshiClient, PolymarketClient

POLY_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
COINBASE_WSS = "wss://ws-feed.exchange.coinbase.com"
BINANCE_WSS = "wss://stream.binance.com:9443/stream"
ARB_RPC = CHAINLINK_FEEDS["BTC"][0]
# keccak256("AnswerUpdated(int256,uint256,uint256)") — emitted by the underlying
# aggregator (not the proxy) on every new round.
ANSWER_UPDATED_TOPIC0 = "0x0559884fd3a460db3073b7fc896cc77986f16e378210ded43186175bf646fc5f"
AGGREGATOR_SELECTOR = "0x245a7bfc"   # EACAggregatorProxy.aggregator()

GRIDS = (300, 900, 3600)      # 5m / 15m directional window grids + hourly threshold
STRIKE_DELAY_S = 10.0         # let lagging AnswerUpdated logs land before freezing a strike
STRIKE_KEEP_S = 2 * 86400.0
POLY_MAX_TOKENS = 95          # WSS market-channel subscription cap headroom
TICK_PUBLISH_MIN_S = 0.25     # spot ticks throttled to ≥250ms apart per asset


def _int256(word: str) -> int:
    v = int(word, 16)
    return v - 2**256 if v >= 2**255 else v


def decode_answer_updated(log: dict) -> tuple[float, int, float] | None:
    """AnswerUpdated(int256 indexed current, uint256 indexed roundId,
    uint256 updatedAt) → (price, round_id, updated_at); None if malformed."""
    try:
        topics = log.get("topics") or []
        if len(topics) < 3 or topics[0].lower() != ANSWER_UPDATED_TOPIC0:
            return None
        price = _int256(topics[1]) / 1e8
        round_id = int(topics[2], 16)
        updated_at = float(int(str(log.get("data", "0x0"))[:66], 16))
        return price, round_id, updated_at
    except Exception:
        return None


class TokenBook:
    """Local order-book mirror for one CLOB token (price level → size)."""

    __slots__ = ("asks", "bids")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply_snapshot(self, bids: list[dict], asks: list[dict]) -> None:
        self.bids = {float(lvl["price"]): float(lvl["size"]) for lvl in bids}
        self.asks = {float(lvl["price"]): float(lvl["size"]) for lvl in asks}

    def apply_change(self, price: Any, side: str, size: Any) -> None:
        levels = self.bids if str(side).upper() == "BUY" else self.asks
        p, s = float(price), float(size)
        if s <= 0:
            levels.pop(p, None)
        else:
            levels[p] = s

    def top(self) -> tuple[float | None, float | None, float, float]:
        bid = max(self.bids, default=None)
        ask = min(self.asks, default=None)
        return (bid, ask,
                self.bids.get(bid, 0.0) if bid is not None else 0.0,
                self.asks.get(ask, 0.0) if ask is not None else 0.0)


class RoundTable:
    """Per-asset Chainlink round sequence, merged from the latestRoundData poll
    and AnswerUpdated logs (deduped on updatedAt — the poll has no raw roundId).
    The authoritative strike for a window boundary = the round in effect at the
    boundary (latest updatedAt ≤ boundary)."""

    def __init__(self, keep: int = 600):
        self.rounds: dict[str, list[tuple[float, float, int | None]]] = {}
        self.keep = keep

    def add(self, asset: str, updated_at: float, price: float,
            round_id: int | None = None) -> bool:
        rs = self.rounds.setdefault(asset, [])
        for i, (ua, p, rid) in enumerate(rs):
            if abs(ua - updated_at) < 1e-9:
                if round_id is not None and rid is None:
                    rs[i] = (ua, p, round_id)   # log row upgrades a poll row
                return False
        bisect.insort(rs, (float(updated_at), float(price), round_id),
                      key=lambda r: r[0])
        if len(rs) > self.keep:
            del rs[: len(rs) - self.keep]
        return True

    def strike_for(self, asset: str, wstart: float) -> tuple[float, float, int | None] | None:
        """(price, round_updated_at, round_id) of the round in effect at wstart;
        None when no round ≤ wstart is known yet (no coverage → no guess)."""
        rs = self.rounds.get(asset) or []
        i = bisect.bisect_right(rs, wstart, key=lambda r: r[0])
        if i == 0:
            return None
        ua, price, rid = rs[i - 1]
        return price, ua, rid


class _Client:
    """One UDS subscriber: prefix-matched topics + a bounded outbound queue
    (a consumer that can't keep up is dropped, never blocks the publishers)."""

    def __init__(self, writer: asyncio.StreamWriter, topics: list[str]):
        self.writer = writer
        self.topics = tuple(topics)
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8000)
        self.alive = True

    def wants(self, topic: str) -> bool:
        return any(topic.startswith(t) for t in self.topics) if self.topics else True

    def send(self, line: bytes) -> None:
        if not self.alive:
            return
        try:
            self.queue.put_nowait(line)
        except asyncio.QueueFull:
            self.alive = False   # slow consumer — disconnect, it will resync


class Broker:
    """UDS pub/sub: NDJSON messages with a global seq, snapshot-on-subscribe.
    The latest message per topic is kept for snapshots and the md.db writer."""

    def __init__(self, sock_path: str | Path):
        self.sock_path = Path(sock_path)
        self.seq = 0
        self.latest: dict[str, dict] = {}
        self.dirty: set[str] = set()
        self.clients: set[_Client] = set()
        self._server: asyncio.Server | None = None

    def publish(self, topic: str, payload: dict, ts_src: float | None = None) -> dict:
        self.seq += 1
        msg = {"seq": self.seq, "ts_src": ts_src, "ts_recv": round(time.time(), 6),
               "topic": topic, "payload": payload}
        self.latest[topic] = msg
        self.dirty.add(topic)
        line = (json.dumps(msg, default=float) + "\n").encode()
        for c in list(self.clients):
            if c.alive and c.wants(topic):
                c.send(line)
        return msg

    async def start(self) -> None:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            self.sock_path.unlink()
        self._server = await asyncio.start_unix_server(self._handle, path=str(self.sock_path))

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            sub = json.loads(line or b"{}")
            topics = list(sub.get("topics") or [""])
        except Exception:
            with contextlib.suppress(Exception):
                writer.close()
            return
        client = _Client(writer, topics)
        # snapshot first (synchronous enqueue), THEN register — no await between
        # the two, so a concurrent publish can't be missed or duplicated.
        for topic in sorted(self.latest):
            if client.wants(topic):
                m = dict(self.latest[topic])
                m["snapshot"] = True
                client.send((json.dumps(m, default=float) + "\n").encode())
        self.clients.add(client)
        try:
            while client.alive:
                writer.write(await client.queue.get())
                await writer.drain()
        except Exception:
            pass
        finally:
            self.clients.discard(client)
            with contextlib.suppress(Exception):
                writer.close()


class SnapshotDB:
    """md.db: latest message per topic (late joiners / dashboard) + append-only
    strike and round history (the offline rollout diff + settlement audit)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS snapshots(
      topic TEXT PRIMARY KEY, seq INTEGER, ts_src REAL, ts_recv REAL, payload TEXT);
    CREATE TABLE IF NOT EXISTS strikes(
      asset TEXT, window_start INTEGER, price REAL, round_updated_at REAL,
      round_id TEXT, ts_recv REAL, PRIMARY KEY(asset, window_start));
    CREATE TABLE IF NOT EXISTS rounds(
      asset TEXT, updated_at REAL, price REAL, round_id TEXT, src TEXT,
      ts_recv REAL, PRIMARY KEY(asset, updated_at));
    """

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def write_topics(self, msgs: list[dict]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO snapshots(topic,seq,ts_src,ts_recv,payload) VALUES(?,?,?,?,?)",
            [(m["topic"], m["seq"], m["ts_src"], m["ts_recv"],
              json.dumps(m["payload"], default=float)) for m in msgs])
        self.conn.commit()

    def add_strike(self, asset: str, window_start: int, price: float,
                   round_updated_at: float, round_id: int | None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO strikes VALUES(?,?,?,?,?,?)",
            (asset, window_start, price, round_updated_at,
             str(round_id) if round_id is not None else None, time.time()))
        self.conn.commit()

    def add_round(self, asset: str, updated_at: float, price: float,
                  round_id: int | None, src: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO rounds VALUES(?,?,?,?,?,?)",
            (asset, updated_at, price,
             str(round_id) if round_id is not None else None, src, time.time()))
        self.conn.commit()


class MDService:
    def __init__(self, assets: list[str], poly_tfs: tuple[str, ...] = ("15m", "5m", "4h"),
                 thresh_tfs: tuple[str, ...] = ("1h", "1D"),
                 sock: str | Path = "data/md.sock", db: str | Path = "data/md.db",
                 status: str | Path = "data/md_status.json"):
        self.assets = list(assets)
        self.poly_tfs = poly_tfs
        self.thresh_tfs = thresh_tfs
        self.broker = Broker(sock)
        self.db = SnapshotDB(db)
        self.status_path = Path(status)
        self.rounds = RoundTable()
        self.poly = PolymarketClient()
        self.kalshi = KalshiClient()
        # state
        self.books: dict[str, TokenBook] = {}      # token_id -> book mirror
        self.token_meta: dict[str, dict] = {}      # token_id -> market meta
        self.poly_tokens: set[str] = set()         # WSS subscription set
        self.kalshi_markets: list[dict] = []       # raw dicts to poll
        self.spot: dict[str, dict] = {}            # asset -> latest spot tick
        self.strikes: dict[tuple[str, int], float] = {}
        self._tick_pub: dict[str, float] = {}      # asset -> last publish ts
        self._t0 = time.time()
        self.counts: dict[str, int] = defaultdict(int)
        self.last_event: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self._err_ts: dict[str, float] = {}

    # ---------------------------------------------------------------- utils
    def _err(self, src: str, e: Exception) -> None:
        self.errors[src] = f"{type(e).__name__}: {e}"[:200]
        now = time.time()
        if now - self._err_ts.get(src, 0.0) > 30.0:   # don't spam the log
            self._err_ts[src] = now
            print(f"# [{src}] {self.errors[src]}")

    def _mark(self, src: str) -> None:
        self.last_event[src] = time.time()
        self.counts[src] += 1

    def _rpc(self, method: str, params: list) -> Any:
        r = net.get_json(ARB_RPC, data=json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
            timeout=6.0)
        if "error" in r:
            raise RuntimeError(f"rpc {method}: {r['error']}")
        return r["result"]

    # ------------------------------------------------------------ discovery
    async def discovery_loop(self) -> None:
        while True:
            try:
                poly, hourly, kal, kalt = await asyncio.gather(
                    asyncio.to_thread(self.poly.discover, self.assets, self.poly_tfs),
                    asyncio.to_thread(self.poly.discover_hourly, self.assets),
                    asyncio.to_thread(self.kalshi.discover, self.assets),
                    asyncio.to_thread(self.kalshi.discover_threshold, self.assets,
                                      self.thresh_tfs))
                self.apply_discovery(poly, hourly, kal, kalt)
                self._mark("discovery")
            except Exception as e:
                self._err("discovery", e)
            await asyncio.sleep(60)

    @staticmethod
    def _up_token(m: dict) -> str | None:
        try:
            tokens = m.get("clobTokenIds")
            if isinstance(tokens, str):
                tokens = json.loads(tokens or "[]")
            return str(tokens[0]) if tokens else None   # outcome[0] = Up/Yes
        except Exception:
            return None

    def apply_discovery(self, poly: list[dict], hourly: list[dict],
                        kal: list[dict], kalt: list[dict]) -> None:
        """Rebuild the WSS token subscription set + Kalshi poll set, publish the
        raw sweep for engine adapters. Hourly-ladder tokens are kept near the
        money (engines never evaluate far strikes) and capped under the WSS
        subscription limit; up/down markets always make the cut."""
        meta: dict[str, dict] = {}

        def add(m: dict, prio: float) -> None:
            tok = self._up_token(m)
            if tok is None:
                return
            s = float(m.get("_event_start") or 0.0)
            ws = float(m.get("_window_s") or 900.0)
            meta[tok] = {"market_id": str(m.get("id")), "asset": m.get("_asset"),
                         "tf": m.get("_tf"), "start_ts": s, "end_ts": s + ws,
                         "_prio": prio}

        for m in poly:
            add(m, 0.0)
        for m in hourly:
            spot = (self.spot.get(str(m.get("_asset"))) or {}).get("price")
            strike = float(m.get("_strike") or 0.0)
            if spot and strike > 0:
                dist = abs(math.log(spot / strike))
                if dist <= 0.10:
                    add(m, 1.0 + dist)
            else:   # no spot yet (cold start) — include, the cap sorts it out
                add(m, 2.0)
        keep = sorted(meta, key=lambda t: meta[t]["_prio"])[:POLY_MAX_TOKENS]
        self.token_meta = {t: meta[t] for t in keep}
        self.poly_tokens = set(keep)
        self.books = {t: b for t, b in self.books.items() if t in self.poly_tokens}
        self.kalshi_markets = sorted(kal + kalt, key=lambda m: str(m.get("close_time", "")))
        self.broker.publish("mkt.discovery", {
            "ts": time.time(), "polymarket": poly, "polymarket_hourly": hourly,
            "kalshi": kal, "kalshi_threshold": kalt})

    # ------------------------------------------------------ polymarket books
    async def poly_books(self) -> None:
        while True:
            tokens = sorted(self.poly_tokens)
            if not tokens:
                await asyncio.sleep(2.0)
                continue
            try:
                async with ws_connect(
                        POLY_WSS, additional_headers={"User-Agent": net.UA["User-Agent"]},
                        ping_interval=20, ping_timeout=20, max_size=2**22,
                        open_timeout=10) as ws:
                    await ws.send(json.dumps({"type": "market", "assets_ids": tokens}))
                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        except TimeoutError:
                            await ws.send("PING")
                            raw = None
                        if raw and raw not in ("PONG", "PING"):
                            try:
                                evs = json.loads(raw)
                            except Exception:
                                evs = None
                            if isinstance(evs, dict):
                                evs = [evs]
                            if isinstance(evs, list):
                                for ev in evs:
                                    if isinstance(ev, dict):
                                        self.poly_event(ev)
                        if self.poly_tokens != set(tokens):
                            break   # discovery changed the set — resubscribe
            except Exception as e:
                self._err("poly_ws", e)
                await asyncio.sleep(3.0)

    def poly_event(self, ev: dict) -> None:
        et = ev.get("event_type")
        tid = str(ev.get("asset_id") or "")
        m = self.token_meta.get(tid)
        if m is None:
            return
        try:
            ts_src: float | None = float(ev.get("timestamp") or 0) / 1000.0 or None
        except Exception:
            ts_src = None
        if et == "book":
            book = self.books.setdefault(tid, TokenBook())
            book.apply_snapshot(ev.get("bids") or ev.get("buys") or [],
                                ev.get("asks") or ev.get("sells") or [])
        elif et == "price_change":
            book = self.books.setdefault(tid, TokenBook())
            changes = ev.get("changes")
            if not isinstance(changes, list):    # older flat single-change shape
                changes = [ev] if ev.get("price") is not None else []
            for ch in changes:
                try:
                    book.apply_change(ch["price"], ch.get("side", "BUY"), ch["size"])
                except Exception:
                    continue
        elif et == "last_trade_price":
            self._mark("poly_trade")
            self.broker.publish(f"trade.polymarket.{m['market_id']}", {
                "venue": "polymarket", "market_id": m["market_id"], "token_id": tid,
                "price": ev.get("price"), "size": ev.get("size"),
                "side": ev.get("side")}, ts_src=ts_src)
            return
        else:
            return
        self._mark("poly_book")
        bid, ask, bsz, asz = self.books[tid].top()
        self.broker.publish(f"book.polymarket.{m['market_id']}", {
            "venue": "polymarket", "market_id": m["market_id"], "token_id": tid,
            "asset": m["asset"], "tf": m["tf"], "start_ts": m["start_ts"],
            "end_ts": m["end_ts"], "up_bid": bid, "up_ask": ask,
            "up_bid_size": bsz, "up_ask_size": asz}, ts_src=ts_src)

    # --------------------------------------------------------- kalshi books
    async def kalshi_poll(self) -> None:
        while True:
            mkts = self.kalshi_markets[:12]   # bounded sweep, nearest expiries first
            if mkts:
                tops = await asyncio.gather(
                    *(asyncio.to_thread(self.kalshi.book_top, m) for m in mkts),
                    return_exceptions=True)
                for bt in tops:
                    if isinstance(bt, BookTop):
                        self._mark("kalshi_book")
                        self.broker.publish(f"book.kalshi.{bt.market_id}", {
                            "venue": "kalshi", "market_id": bt.market_id,
                            "asset": None, "tf": None,
                            "start_ts": bt.start_ts, "end_ts": bt.end_ts,
                            "up_bid": bt.up_bid, "up_ask": bt.up_ask,
                            "up_bid_size": bt.up_bid_size,
                            "up_ask_size": bt.up_ask_size}, ts_src=bt.observed_at)
            await asyncio.sleep(2.0)

    # ----------------------------------------------------------------- spot
    def spot_update(self, asset: str, price: float, ts_src: float | None,
                    source: str) -> None:
        """Coinbase is the blend's tick source of record (matches CompositeFeed
        priority); Binance fills only when Coinbase is dark/stale for the asset."""
        now = time.time()
        cur = self.spot.get(asset)
        if (source != "coinbase" and cur is not None and cur["source"] == "coinbase"
                and now - cur["_rx"] < 3.0):
            return
        self.spot[asset] = {"price": price, "source": source, "ts_src": ts_src, "_rx": now}
        self._mark(f"spot_{source}")
        if now - self._tick_pub.get(asset, 0.0) >= TICK_PUBLISH_MIN_S:
            self._tick_pub[asset] = now
            self.broker.publish(f"tick.{asset}", {
                "asset": asset, "price": price, "source": source}, ts_src=ts_src)

    async def coinbase_ws(self) -> None:
        products = {CoinbaseFeed.PRODUCT[a]: a for a in self.assets if a in CoinbaseFeed.PRODUCT}
        core = {f"{a}-USD" for a in ("BTC", "ETH", "SOL", "XRP", "DOGE")}
        while True:
            if not products:
                await asyncio.sleep(30.0)
                continue
            try:
                async with ws_connect(COINBASE_WSS, ping_interval=20, ping_timeout=20,
                                      open_timeout=10) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channels": [{"name": "ticker", "product_ids": sorted(products)}]}))
                    async for raw in ws:
                        msg = json.loads(raw)
                        t = msg.get("type")
                        if t == "ticker" and msg.get("price"):
                            a = products.get(str(msg.get("product_id")))
                            if a:
                                self.spot_update(a, float(msg["price"]), None, "coinbase")
                        elif t == "error":
                            # an unknown product fails the whole subscription —
                            # drop the offender (or fall back to the core set)
                            blob = f"{msg.get('message', '')} {msg.get('reason', '')}"
                            bad = [p for p in products if p in blob]
                            for p in bad or [p for p in products if p not in core]:
                                products.pop(p, None)
                            raise RuntimeError(f"subscribe rejected: {blob[:120]}")
            except Exception as e:
                self._err("coinbase_ws", e)
                await asyncio.sleep(3.0)

    async def binance_ws(self) -> None:
        syms = {BinanceFeed.SYMBOL[a].lower(): a for a in self.assets if a in BinanceFeed.SYMBOL}
        streams = "/".join(f"{s}@miniTicker" for s in sorted(syms))
        while True:
            try:
                async with ws_connect(f"{BINANCE_WSS}?streams={streams}", ping_interval=20,
                                      ping_timeout=20, open_timeout=10) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        d = msg.get("data") or {}
                        if d.get("e") == "24hrMiniTicker" and d.get("c"):
                            a = syms.get(str(d.get("s", "")).lower())
                            if a:
                                ts = float(d.get("E", 0)) / 1000.0 or None
                                self.spot_update(a, float(d["c"]), ts, "binance")
            except Exception as e:
                self._err("binance_ws", e)
                await asyncio.sleep(5.0)

    # ------------------------------------------------------------ chainlink
    def add_round(self, asset: str, updated_at: float, price: float,
                  round_id: int | None, src: str) -> bool:
        if not self.rounds.add(asset, updated_at, price, round_id):
            return False
        self._mark("chainlink")
        self.broker.publish(f"chainlink.{asset}", {
            "asset": asset, "price": price, "updated_at": updated_at,
            "round_id": round_id, "src": src}, ts_src=updated_at)
        self.db.add_round(asset, updated_at, price, round_id, src)
        return True

    async def chainlink_poll(self) -> None:
        while True:
            try:
                ticks = await asyncio.to_thread(batch_chainlink, self.assets, 3.0)
                for a, t in ticks.items():
                    if t.updated_at:
                        self.add_round(a, t.updated_at, t.price, None, "poll")
                if ticks:
                    self._mark("chainlink_poll")
            except Exception as e:
                self._err("chainlink_poll", e)
            await asyncio.sleep(1.5)

    def _resolve_aggregators(self) -> dict[str, str]:
        """proxy.aggregator() per asset → {aggregator_address_lower: asset};
        AnswerUpdated logs are emitted by the aggregator, not the proxy."""
        feeds = [(a, CHAINLINK_FEEDS[a][1]) for a in self.assets if a in CHAINLINK_FEEDS]
        payload = json.dumps([
            {"jsonrpc": "2.0", "id": i, "method": "eth_call",
             "params": [{"to": addr, "data": AGGREGATOR_SELECTOR}, "latest"]}
            for i, (_, addr) in enumerate(feeds)]).encode()
        resp = net.get_json(ARB_RPC, data=payload, timeout=6.0)
        out: dict[str, str] = {}
        by_id = {r.get("id"): r for r in resp if isinstance(r, dict)}
        for i, (asset, _) in enumerate(feeds):
            res = (by_id.get(i) or {}).get("result") or ""
            if len(res) >= 42:
                out["0x" + res[-40:].lower()] = asset
        return out

    async def chainlink_logs(self) -> None:
        aggs: dict[str, str] | None = None
        last_block: int | None = None
        while True:
            try:
                if aggs is None or not aggs:
                    aggs = await asyncio.to_thread(self._resolve_aggregators)
                    print(f"# [chainlink_logs] watching {len(aggs)} aggregators")
                head = int(await asyncio.to_thread(self._rpc, "eth_blockNumber", []), 16)
                if last_block is None:
                    last_block = head
                elif head > last_block:
                    logs = await asyncio.to_thread(self._rpc, "eth_getLogs", [{
                        "fromBlock": hex(last_block + 1), "toBlock": hex(head),
                        "address": sorted(aggs), "topics": [ANSWER_UPDATED_TOPIC0]}])
                    for lg in logs or []:
                        dec = decode_answer_updated(lg)
                        a = aggs.get(str(lg.get("address", "")).lower())
                        if dec and a:
                            price, rid, ua = dec
                            self.add_round(a, ua, price, rid, "log")
                    last_block = head
                    self._mark("chainlink_logs")
            except Exception as e:
                self._err("chainlink_logs", e)
                await asyncio.sleep(6.0)
            await asyncio.sleep(4.0)

    # -------------------------------------------------------------- strikes
    def resolve_strikes(self, now: float | None = None) -> int:
        """Freeze the Chainlink-round-exact strike for every passed window
        boundary (after a small delay so lagging logs land). Returns # new."""
        now = time.time() if now is None else now
        added = 0
        for a in self.assets:
            for grid in GRIDS:
                w = int(now) - int(now) % grid
                for ws_ in (w - grid, w):
                    if now < ws_ + STRIKE_DELAY_S or (a, ws_) in self.strikes:
                        continue
                    r = self.rounds.strike_for(a, ws_)
                    if r is None:
                        continue
                    price, ua, rid = r
                    self.strikes[(a, ws_)] = price
                    added += 1
                    self._mark("strike")
                    self.broker.publish(f"strike.{a}.{ws_}", {
                        "asset": a, "window_start": ws_, "price": price,
                        "round_updated_at": ua, "round_id": rid}, ts_src=ua)
                    self.db.add_strike(a, ws_, price, ua, rid)
        cutoff = now - STRIKE_KEEP_S
        for k in [k for k in self.strikes if k[1] < cutoff]:
            del self.strikes[k]
        return added

    async def strike_loop(self) -> None:
        while True:
            try:
                self.resolve_strikes()
            except Exception as e:
                self._err("strike", e)
            await asyncio.sleep(1.0)

    # --------------------------------------------------------- housekeeping
    def status_blob(self) -> dict:
        now = time.time()
        return {
            "updated": now, "uptime_s": round(now - self._t0, 1),
            "seq": self.broker.seq, "clients": len(self.broker.clients),
            "poly_tokens": len(self.poly_tokens),
            "kalshi_markets": len(self.kalshi_markets),
            "strikes": len(self.strikes),
            "spot_age_s": {a: round(now - v["_rx"], 1) for a, v in self.spot.items()},
            "chainlink_staleness_s": {
                a: (round(now - rs[-1][0], 1) if rs else None)
                for a, rs in self.rounds.rounds.items()},
            "event_age_s": {k: round(now - v, 1) for k, v in self.last_event.items()},
            "counts": dict(self.counts),
            "errors": self.errors,
        }

    async def housekeeping(self) -> None:
        n = 0
        while True:
            try:
                self.broker.publish("hb", {"ts": time.time(), "seq": self.broker.seq,
                                           "clients": len(self.broker.clients)})
                dirty = [self.broker.latest[t] for t in self.broker.dirty
                         if t in self.broker.latest]
                self.broker.dirty.clear()
                if dirty:
                    await asyncio.to_thread(self.db.write_topics, dirty)
                if n % 2 == 0:
                    tmp = self.status_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(self.status_blob(), default=float, indent=1))
                    tmp.replace(self.status_path)
            except Exception as e:
                self._err("housekeeping", e)
            n += 1
            await asyncio.sleep(2.0)

    async def run(self) -> None:
        await self.broker.start()
        print(f"# md_service | assets={self.assets} | sock={self.broker.sock_path} | "
              f"topics: tick/book/trade/chainlink/strike/mkt.discovery/hb")
        await asyncio.gather(
            self.discovery_loop(), self.poly_books(), self.kalshi_poll(),
            self.coinbase_ws(), self.binance_ws(), self.chainlink_poll(),
            self.chainlink_logs(), self.strike_loop(), self.housekeeping())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE,BNB,HYPE")
    ap.add_argument("--poly-tfs", default="15m,5m,4h")
    ap.add_argument("--thresh-tfs", default="1h,1D")
    ap.add_argument("--sock", default="data/md.sock")
    ap.add_argument("--db", default="data/md.db")
    ap.add_argument("--status", default="data/md_status.json")
    args = ap.parse_args()
    svc = MDService(args.assets.split(","), poly_tfs=tuple(args.poly_tfs.split(",")),
                    thresh_tfs=tuple(args.thresh_tfs.split(",")),
                    sock=args.sock, db=args.db, status=args.status)
    asyncio.run(svc.run())


if __name__ == "__main__":
    main()
