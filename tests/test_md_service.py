"""Brief 1 — shared market-data service (live/md_service.py) + the engines'
stdlib adapters (live/md_client.py): book mirroring, the Chainlink round →
strike bracketing, UDS pub/sub snapshot+stream, and the control-safety
fallbacks (kill md_service → engines degrade to direct REST within a cycle)."""

import asyncio
import json
import sqlite3
import threading
import time

import pytest

from live.md_client import MDFeedHub, MDMirror
from live.md_service import (
    ANSWER_UPDATED_TOPIC0,
    Broker,
    MDService,
    RoundTable,
    SnapshotDB,
    TokenBook,
    decode_answer_updated,
)


# ---------------------------------------------------------------- token book
def test_token_book_snapshot_and_changes():
    b = TokenBook()
    b.apply_snapshot([{"price": "0.48", "size": "120"}, {"price": "0.45", "size": "10"}],
                     [{"price": "0.52", "size": "80"}, {"price": "0.55", "size": "5"}])
    assert b.top() == (0.48, 0.52, 120.0, 80.0)
    b.apply_change("0.49", "BUY", "50")     # new best bid
    b.apply_change("0.52", "SELL", "0")     # best ask pulled
    bid, ask, bsz, asz = b.top()
    assert (bid, bsz) == (0.49, 50.0)
    assert (ask, asz) == (0.55, 5.0)
    b.apply_change("0.49", "BUY", "0")      # removed -> falls back to 0.48
    assert b.top()[0] == 0.48


# ------------------------------------------------------- rounds & strikes
def test_round_table_strike_bracketing():
    rt = RoundTable()
    w = 1_700_000_100.0
    assert rt.strike_for("BTC", w) is None          # no coverage -> no guess
    rt.add("BTC", w - 50, 99.0)
    rt.add("BTC", w - 5, 100.0)
    rt.add("BTC", w + 3, 101.0)                     # after the boundary
    price, ua, _rid = rt.strike_for("BTC", w)
    assert price == 100.0 and ua == w - 5           # round in effect AT the open
    assert rt.add("BTC", w - 5, 100.0) is False     # dedupe on updated_at
    assert rt.add("BTC", w - 5, 100.0, round_id=7) is False   # log upgrades poll row
    assert rt.strike_for("BTC", w)[2] == 7


def test_decode_answer_updated_log():
    price, rid = 50_000 * 10**8, 4242
    log = {"topics": [ANSWER_UPDATED_TOPIC0,
                      "0x" + format(price, "064x"),
                      "0x" + format(rid, "064x")],
           "data": "0x" + format(1_700_000_000, "064x")}
    assert decode_answer_updated(log) == (50_000.0, 4242, 1_700_000_000.0)
    assert decode_answer_updated({"topics": [], "data": "0x"}) is None
    assert decode_answer_updated({"topics": ["0xdead", "0x0", "0x0"], "data": "0x0"}) is None


def _svc(tmp_path, assets=("BTC",)):
    return MDService(list(assets), sock=tmp_path / "md.sock", db=tmp_path / "md.db",
                     status=tmp_path / "md_status.json")


def test_resolve_strikes_after_delay_only(tmp_path):
    svc = _svc(tmp_path)
    w = 1_700_000_100          # multiple of 300 and 900 -> one window key
    svc.rounds.add("BTC", w - 5, 100.0)
    svc.rounds.add("BTC", w + 3, 101.0)
    assert svc.resolve_strikes(now=w + 2.0) == 0          # inside the log-lag grace
    n = svc.resolve_strikes(now=w + 11.0)
    assert n >= 1 and svc.strikes[("BTC", w)] == 100.0    # round at open, not w+3
    msg = svc.broker.latest[f"strike.BTC.{w}"]
    assert msg["payload"]["price"] == 100.0
    rows = sqlite3.connect(tmp_path / "md.db").execute(
        "SELECT asset, window_start, price FROM strikes").fetchall()
    assert ("BTC", w, 100.0) in rows


# ------------------------------------------------------------- poly events
def test_poly_event_book_and_price_change_publish_top(tmp_path):
    svc = _svc(tmp_path)
    svc.token_meta["tok1"] = {"market_id": "M1", "asset": "BTC", "tf": "15m",
                              "start_ts": 0.0, "end_ts": 900.0}
    svc.poly_event({"event_type": "book", "asset_id": "tok1",
                    "bids": [{"price": "0.48", "size": "120"}],
                    "asks": [{"price": "0.52", "size": "80"}],
                    "timestamp": "1700000000000"})
    p = svc.broker.latest["book.polymarket.M1"]["payload"]
    assert (p["up_bid"], p["up_ask"]) == (0.48, 0.52)
    svc.poly_event({"event_type": "price_change", "asset_id": "tok1",
                    "changes": [{"price": "0.50", "side": "BUY", "size": "30"}],
                    "timestamp": "1700000001000"})
    p = svc.broker.latest["book.polymarket.M1"]["payload"]
    assert (p["up_bid"], p["up_bid_size"]) == (0.50, 30.0)
    svc.poly_event({"event_type": "book", "asset_id": "unknown"})   # unsubscribed: ignored
    svc.poly_event({"event_type": "last_trade_price", "asset_id": "tok1",
                    "price": "0.51", "size": "10", "side": "BUY",
                    "timestamp": "1700000002000"})
    assert svc.broker.latest["trade.polymarket.M1"]["payload"]["price"] == "0.51"


def test_apply_discovery_selects_tokens_near_money(tmp_path):
    svc = _svc(tmp_path)
    svc.spot["BTC"] = {"price": 100_000.0, "source": "coinbase", "ts_src": None,
                       "_rx": time.time()}
    updown = {"id": 1, "clobTokenIds": json.dumps(["t-up", "t-dn"]), "_asset": "BTC",
              "_tf": "15m", "_window_s": 900, "_event_start": 1000, "slug": "btc-updown"}
    near = {"id": 2, "clobTokenIds": json.dumps(["h-near", "h-no"]), "_asset": "BTC",
            "_tf": "1h", "_window_s": 3600, "_event_start": 2000, "_strike": 100_500.0}
    far = {"id": 3, "clobTokenIds": json.dumps(["h-far", "h-no2"]), "_asset": "BTC",
           "_tf": "1h", "_window_s": 3600, "_event_start": 2000, "_strike": 150_000.0}
    svc.apply_discovery([updown], [near, far], [], [])
    assert svc.poly_tokens == {"t-up", "h-near"}           # up-token only; far strike cut
    assert svc.token_meta["t-up"]["market_id"] == "1"
    assert svc.token_meta["t-up"]["end_ts"] == 1900.0
    d = svc.broker.latest["mkt.discovery"]["payload"]
    assert len(d["polymarket"]) == 1 and len(d["polymarket_hourly"]) == 2


def test_spot_update_coinbase_priority(tmp_path):
    svc = _svc(tmp_path)
    svc.spot_update("BTC", 100.0, None, "coinbase")
    svc.spot_update("BTC", 999.0, None, "binance")         # fresh coinbase wins
    assert svc.spot["BTC"]["price"] == 100.0
    svc.spot["BTC"]["_rx"] -= 10.0                          # coinbase goes stale
    svc.spot_update("BTC", 101.0, None, "binance")
    assert svc.spot["BTC"]["price"] == 101.0


# ------------------------------------------------------------ broker (UDS)
def test_broker_snapshot_then_filtered_stream(tmp_path):
    async def main():
        broker = Broker(tmp_path / "md.sock")
        await broker.start()
        broker.publish("tick.BTC", {"price": 1.0})
        broker.publish("book.polymarket.m1", {"up_bid": 0.4})
        reader, writer = await asyncio.open_unix_connection(str(tmp_path / "md.sock"))
        writer.write(b'{"op":"subscribe","topics":["tick."]}\n')
        await writer.drain()
        snap = json.loads(await asyncio.wait_for(reader.readline(), 2))
        assert snap["topic"] == "tick.BTC" and snap["snapshot"] is True
        await asyncio.sleep(0.05)                       # let the client register
        broker.publish("book.polymarket.m1", {"up_bid": 0.41})   # filtered out
        broker.publish("tick.BTC", {"price": 2.0})
        live = json.loads(await asyncio.wait_for(reader.readline(), 2))
        assert live["topic"] == "tick.BTC" and live["payload"]["price"] == 2.0
        assert "snapshot" not in live
        writer.close()
    asyncio.run(main())


def test_snapshot_db_roundtrip(tmp_path):
    db = SnapshotDB(tmp_path / "md.db")
    db.write_topics([{"topic": "tick.BTC", "seq": 5, "ts_src": 1.0, "ts_recv": 2.0,
                      "payload": {"price": 7.0}}])
    db.write_topics([{"topic": "tick.BTC", "seq": 6, "ts_src": 1.5, "ts_recv": 2.5,
                      "payload": {"price": 8.0}}])
    row = db.conn.execute("SELECT seq, payload FROM snapshots WHERE topic='tick.BTC'").fetchone()
    assert row[0] == 6 and json.loads(row[1])["price"] == 8.0


# --------------------------------------------- mirror client (integration)
@pytest.fixture
def broker_thread(tmp_path):
    """A real Broker on a real UDS socket, its event loop in a daemon thread."""
    sock = tmp_path / "md.sock"
    broker = Broker(sock)
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broker.start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    for _ in range(200):
        if sock.exists():
            break
        time.sleep(0.01)
    yield broker, loop, sock
    loop.call_soon_threadsafe(loop.stop)


def _pub(loop, broker, topic, payload, ts_src=None):
    loop.call_soon_threadsafe(broker.publish, topic, payload, ts_src)


def _wait(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_mirror_receives_snapshot_and_stream(broker_thread):
    broker, loop, sock = broker_thread
    _pub(loop, broker, "tick.BTC", {"asset": "BTC", "price": 100.5, "source": "coinbase"})
    _pub(loop, broker, "chainlink.BTC", {"asset": "BTC", "price": 100.0,
                                         "updated_at": 1_700_000_000.0, "round_id": 9})
    time.sleep(0.1)
    mirror = MDMirror(sock, max_age=5.0)
    try:
        assert _wait(lambda: mirror.fresh() and "BTC" in mirror.ticks)   # snapshot landed
        assert mirror.spot_tick("BTC").price == 100.5
        assert mirror.chainlink_tick("BTC").updated_at == 1_700_000_000.0
        _pub(loop, broker, "book.polymarket.M1",
             {"venue": "polymarket", "market_id": "M1", "start_ts": 0.0, "end_ts": 900.0,
              "up_bid": 0.48, "up_ask": 0.52, "up_bid_size": 10.0, "up_ask_size": 5.0})
        _pub(loop, broker, "strike.BTC.1700000100", {"asset": "BTC",
             "window_start": 1_700_000_100, "price": 100.0})
        assert _wait(lambda: mirror.book_top("polymarket", "M1") is not None)
        bt = mirror.book_top("polymarket", "M1")
        assert (bt.up_bid, bt.up_ask) == (0.48, 0.52)
        assert mirror.strike("BTC", 1_700_000_100) == 100.0
        assert mirror.strike("BTC", 1_700_000_400) is None
    finally:
        mirror.stop()


def test_mirror_goes_stale_when_service_dies(broker_thread):
    broker, loop, sock = broker_thread
    mirror = MDMirror(sock, max_age=0.3)
    try:
        _pub(loop, broker, "hb", {"ts": time.time()})
        assert _wait(lambda: mirror.msgs >= 1)
        assert _wait(lambda: not mirror.fresh(), timeout=2.0)   # silence -> stale
        assert mirror.book_top("polymarket", "M1") is None      # dead stream -> None
    finally:
        mirror.stop()


# ----------------------------------------------------- feed hub + fallback
def test_mdfeedhub_uses_mirror_when_fresh():
    mirror = MDMirror("/nonexistent/md.sock", start=False)
    now = time.time()
    mirror.connected = True
    mirror.last_msg_ts = now
    mirror.ticks["BTC"] = {"price": 100.5, "source": "coinbase", "_rx": now, "_ts_src": None}
    mirror.cl["BTC"] = {"price": 100.0, "updated_at": 1_700_000_000.0, "_rx": now}
    hub = MDFeedHub(["BTC"], mirror)
    out = hub.sample()
    assert hub.transport == "md"
    assert out["BTC"].price == 100.0          # anchored at the chainlink round
    assert hub.feeds["BTC"].last_chainlink.updated_at == 1_700_000_000.0
    assert hub.staleness()["BTC"] is not None


def test_mdfeedhub_falls_back_to_rest_when_stale(monkeypatch):
    mirror = MDMirror("/nonexistent/md.sock", start=False)   # never connected
    hub = MDFeedHub(["BTC"], mirror)
    sentinel = {"BTC": None}
    monkeypatch.setattr(hub.inner, "sample", lambda: sentinel)
    assert hub.sample() is sentinel
    assert hub.transport == "rest" and hub.fallbacks == 1
    assert hub.sample() is sentinel           # stays on REST, no flapping prints
    assert hub.fallbacks == 1


# -------------------------------------------------- engine chaos fallback
def test_engine_md_mode_falls_back_within_one_cycle(tmp_path, monkeypatch):
    """Kill-md_service chaos test: --md on with NO service running -> books and
    feeds come from the direct REST path immediately; health reports the state."""
    from live.forward_engine import ForwardEngine, Market
    from live.venues import BookTop

    eng = ForwardEngine(["BTC"], data_dir=str(tmp_path / "run"), md_mode=True)
    try:
        assert eng.md is not None and not eng.md.fresh()
        m = Market(venue="polymarket", market_id="m1", asset="BTC", kind="directional",
                   tf="15m", window_s=900, start_ts=0.0, end_ts=9e9, strike=100.0,
                   raw={}, ticker="t")
        rest_bt = BookTop("polymarket", "m1", 0.0, 9e9, 0.49, 0.51, 10.0, 10.0, 0.0)
        monkeypatch.setattr(eng.poly, "book_top", lambda raw: rest_bt)
        assert eng.book_for(m) is rest_bt                  # REST fallback, same cycle
        assert eng._md_book_miss == 1
        h = eng.health_snapshot()
        assert h["md"]["connected"] is False
        assert h["md"]["books_rest_fallback"] == 1
        # discovery: stale mirror -> None -> direct REST path would run
        assert eng._discover_md() is None
    finally:
        eng.md.stop()


def test_engine_md_mode_merges_authoritative_strikes(tmp_path):
    from live.forward_engine import ForwardEngine

    eng = ForwardEngine(["BTC"], data_dir=str(tmp_path / "run"), md_mode=True)
    try:
        now = int(time.time())
        w = now - now % 300
        eng.md.strikes[("BTC", w)] = {"asset": "BTC", "window_start": w, "price": 123.0}
        eng._merge_md_strikes()
        assert eng.strikes[("BTC", w)] == 123.0            # missed window filled
        eng.strikes[("BTC", w)] = 124.0                    # local capture present
        eng._merge_md_strikes()
        assert eng.strikes[("BTC", w)] == 124.0            # never overwritten
        assert ("BTC", w) in eng._md_strike_diff           # ...but the diff is logged
    finally:
        eng.md.stop()
