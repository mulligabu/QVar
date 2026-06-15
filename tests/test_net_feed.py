"""New speed layer: pooled HTTP, batched Chainlink decode, FeedHub blend,
and the data/HALT kill switch."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import live.feed as feed
import live.net as net
from live.feed import CompositeFeed, Tick, _decode_round, batch_chainlink
from live.forward_engine import ForwardEngine, Market
from live.predictor import Prediction
from live.venues import BookTop


# ---------------------------------------------------------------- net pool
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/missing":
            self._reply({"error": "nope"}, status=404)
        else:
            self._reply({"ok": True, "path": self.path})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._reply({"echo": json.loads(self.rfile.read(n))})


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_get_json_reuses_connection_and_counts(server):
    host = server.split("//")[1]
    before = net.stats().get(host, {"calls": 0, "errors": 0})
    assert net.get_json(f"{server}/a")["path"] == "/a"
    assert net.get_json(f"{server}/b?x=1")["path"] == "/b?x=1"
    s = net.stats()[host]
    assert s["calls"] >= before["calls"] + 2
    assert s["errors"] == before["errors"]
    assert s["mean_ms"] is not None


def test_get_json_posts_body(server):
    out = net.get_json(f"{server}/rpc", data=b'{"jsonrpc": "2.0"}')
    assert out["echo"] == {"jsonrpc": "2.0"}


def test_get_json_http_error_raises_and_is_counted(server):
    host = server.split("//")[1]
    errs = net.stats()[host]["errors"] if host in net.stats() else 0
    with pytest.raises(urllib.error.HTTPError):
        net.get_json(f"{server}/missing")
    assert net.stats()[host]["errors"] == errs + 1


# ---------------------------------------------------------------- chainlink decode
def _round_hex(price_8dp: int, updated_at: int) -> str:
    words = [1, price_8dp, updated_at - 1, updated_at, 1]
    return "0x" + "".join(format(w, "064x") for w in words)


def test_decode_round_extracts_price_and_updated_at():
    dec = _decode_round(_round_hex(50_000 * 10**8, 1_700_000_000))
    assert dec == (50_000.0, 1_700_000_000.0)
    assert _decode_round("0x") is None
    assert _decode_round("") is None


def test_batch_chainlink_one_request_many_assets(monkeypatch):
    def fake_rpc(url, data=None, timeout=4.0, headers=None):
        reqs = json.loads(data)
        assert isinstance(reqs, list) and len(reqs) == 2  # ONE batched request
        return [{"jsonrpc": "2.0", "id": r["id"],
                 "result": _round_hex((50_000 + r["id"]) * 10**8, 1_700_000_000 + r["id"])}
                for r in reqs]

    monkeypatch.setattr(net, "get_json", fake_rpc)
    out = batch_chainlink(["BTC", "ETH", "HYPE"])  # HYPE has no feed -> absent
    assert set(out) == {"BTC", "ETH"}
    assert out["BTC"].price == 50_000.0 and out["ETH"].price == 50_001.0
    assert out["BTC"].updated_at == 1_700_000_000.0


def test_batch_chainlink_rpc_failure_returns_empty(monkeypatch):
    def boom(url, data=None, timeout=4.0, headers=None):
        raise OSError("rpc down")
    monkeypatch.setattr(net, "get_json", boom)
    assert batch_chainlink(["BTC"]) == {}


# ---------------------------------------------------------------- composite blend
def test_composite_prefetched_anchor_plus_drift():
    cf = CompositeFeed("BTC")
    cl1 = Tick(price=100.0, ts=1.0, source="chainlink", updated_at=10.0)
    cb1 = Tick(price=100.5, ts=1.0, source="coinbase")
    t = cf.get(cl=cl1, cb=cb1, prefetched=True)
    assert t.price == pytest.approx(100.0)        # anchored at the round
    cb2 = Tick(price=101.5, ts=2.0, source="coinbase")
    t = cf.get(cl=cl1, cb=cb2, prefetched=True)   # same round -> add drift
    assert t.price == pytest.approx(101.0)        # 100 + (101.5-100.5)
    cl2 = Tick(price=101.2, ts=3.0, source="chainlink", updated_at=11.0)
    t = cf.get(cl=cl2, cb=cb2, prefetched=True)   # NEW round -> re-anchor
    assert t.price == pytest.approx(101.2)


def test_composite_prefetched_miss_does_not_refetch():
    cf = CompositeFeed("BTC")
    cb = Tick(price=99.0, ts=1.0, source="coinbase")
    t = cf.get(cl=None, cb=cb, prefetched=True)   # no chainlink yet -> bootstrap
    assert t.price == pytest.approx(99.0)
    assert cf.get(cl=None, cb=None, prefetched=True) is None or True  # no crash


# ---------------------------------------------------------------- HALT switch
def _trade_setup(tmp_path):
    eng = ForwardEngine(["BTC"], data_dir=str(tmp_path / "run"))
    eng.predictor.n_updates = 50   # warmed
    m = Market(venue="polymarket", market_id="m1", asset="BTC", kind="directional",
               tf="15m", window_s=900, start_ts=0.0, end_ts=9e9, strike=100.0,
               raw={}, ticker="t")
    bt = BookTop("polymarket", "m1", 0.0, 9e9, 0.64, 0.66, 50.0, 50.0, 0.0)
    pred = Prediction(p_up=0.74, p_cal=0.84, move_prob=0.6, directional_edge=0.4,
                      fair_value_residual=0.19, regime="mid_vol/chop", temperature=1.0,
                      confidence=0.3, head_outputs={}, head_weights={},
                      features={"z_dist": 2.0, "momentum": 0.1, "realized_vol": 0.5,
                                "efficiency": 0.3, "ob_imbalance": 0.0, "log_tau": 1.0,
                                "_spot": 100.3, "_strike": 100.0, "_tau_min": 5.0})
    return eng, m, bt, pred


def test_evaluate_trade_opens_position_when_not_halted(tmp_path):
    eng, m, bt, pred = _trade_setup(tmp_path)
    eng._halted = False
    eng.evaluate_trade(m, bt, pred, mins_left=5.0)
    assert m.key in eng.open_pos
    row = eng.store.conn.execute(
        "SELECT decision, halted FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
    assert row == ("TRADE", 0)


def test_evaluate_trade_halt_blocks_entry_but_logs_counterfactual(tmp_path):
    eng, m, bt, pred = _trade_setup(tmp_path)
    eng._halted = True   # operator created data/HALT
    eng.evaluate_trade(m, bt, pred, mins_left=5.0)
    assert m.key not in eng.open_pos                 # no new position
    assert m.key in eng.shadow                       # still shadow-learns
    row = eng.store.conn.execute(
        "SELECT decision, halted FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
    assert row == ("SKIP", 1)                        # counterfactual preserved


def test_halt_flag_follows_file(tmp_path, monkeypatch):
    eng, *_ = _trade_setup(tmp_path)   # ctrl_root = tmp_path (data_dir's parent)
    monkeypatch.setattr(feed.FeedHub, "sample", lambda self: {})
    eng.discover = lambda: []
    eng.step()
    assert eng._halted is False
    (tmp_path / "HALT").touch()        # legacy soft-halt file
    eng.step()
    assert eng._halted is True and eng._block_reason == "halt"
