"""Control plane: estop freeze, max-loss breaker latch, cross-lane exposure cap,
dashboard toggle persistence — all immediate-effect (re-read every cycle)."""

import json
import time
from datetime import UTC, datetime

import pytest

import live.feed as feed
from live.control import (
    DEFAULTS,
    clear_trip,
    lanes_snapshot,
    load_control,
    read_trip,
    save_control,
    trip,
)
from live.forward_engine import ForwardEngine, GhostPosition, Market
from live.predictor import Prediction
from live.venues import BookTop


# ---------------------------------------------------------------- control file
def test_load_defaults_when_missing(tmp_path):
    c = load_control(tmp_path)
    assert c == DEFAULTS and c["max_loss"]["enabled"] is True


def test_save_control_merges_and_roundtrips(tmp_path):
    save_control(tmp_path, {"max_loss": {"limit": 80.0}})
    save_control(tmp_path, {"estop": True, "estop_reason": "operator (dashboard)"})
    c = load_control(tmp_path)
    assert c["max_loss"] == {"enabled": True, "limit": 80.0}  # partial merge kept enabled
    assert c["estop"] is True and c["estop_reason"] == "operator (dashboard)"
    assert c["exposure_cap"]["max_total_frac"] == 0.10        # untouched section intact


def test_save_control_adjusts_exposure_caps(tmp_path):
    save_control(tmp_path, {"exposure_cap": {"enabled": False, "max_total_frac": 0.2}})
    c = load_control(tmp_path)
    assert c["exposure_cap"]["enabled"] is False
    assert c["exposure_cap"]["max_total_frac"] == 0.2
    assert c["exposure_cap"]["max_asset_frac"] == 0.05


def test_corrupt_control_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "CONTROL.json").write_text("{not json")
    assert load_control(tmp_path) == DEFAULTS


def test_trip_latches_first_writer_wins(tmp_path):
    a = trip(tmp_path, "lane main: day -160", -160.0, 150.0)
    b = trip(tmp_path, "lane alt: day -170", -170.0, 150.0)
    assert b == a                       # original root cause preserved
    assert read_trip(tmp_path)["day_pnl"] == -160.0
    clear_trip(tmp_path)
    assert read_trip(tmp_path) is None


# ---------------------------------------------------------------- lane aggregation
def _lane_status(tmp_path, name, pnl_today=0.0, equity=1000.0, positions=()):
    d = tmp_path / f"forward_live_{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({
        "updated": datetime.now(UTC).isoformat(),
        "config": {"name": name},
        "performance": {"pnl_today": pnl_today, "equity": equity},
        "open_positions": [{"asset": a, "cost": c} for a, c in positions],
    }))


def test_lanes_snapshot_aggregates_and_excludes(tmp_path):
    _lane_status(tmp_path, "alt", pnl_today=-40.0, equity=900.0,
                 positions=[("BTC", 30.0), ("XRP", 10.0)])
    _lane_status(tmp_path, "alt2", pnl_today=-20.0, equity=950.0,
                 positions=[("BTC", 15.0)])
    snap = lanes_snapshot(tmp_path, exclude="alt2")
    assert [ln["name"] for ln in snap["lanes"]] == ["alt"]
    assert snap["day_pnl"] == -40.0 and snap["equity"] == 900.0
    assert snap["by_asset"] == {"BTC": 30.0, "XRP": 10.0}
    full = lanes_snapshot(tmp_path)
    assert full["day_pnl"] == -60.0 and full["by_asset"]["BTC"] == 45.0


def test_lanes_snapshot_skips_stale_lane(tmp_path):
    _lane_status(tmp_path, "alt", pnl_today=-40.0)
    s = json.loads((tmp_path / "forward_live_alt" / "status.json").read_text())
    s["updated"] = "2026-01-01T00:00:00+00:00"   # long dead
    (tmp_path / "forward_live_alt" / "status.json").write_text(json.dumps(s))
    assert lanes_snapshot(tmp_path)["lanes"] == []


# ---------------------------------------------------------------- engine wiring
def _engine(tmp_path, **kw):
    eng = ForwardEngine(["BTC"], data_dir=str(tmp_path / "forward_live"), **kw)
    eng.predictor.n_updates = 50
    return eng


def _tradeable(eng):
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
    return m, bt, pred


def test_estop_freezes_step_entirely(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    sampled = []
    monkeypatch.setattr(feed.FeedHub, "sample", lambda self: sampled.append(1) or {})
    eng.discover = lambda: []
    save_control(tmp_path, {"estop": True, "estop_reason": "operator (dashboard)"})
    eng.step()
    assert sampled == []                      # no feeds, no work at all
    assert eng._block_reason == "estop"
    save_control(tmp_path, {"estop": False})  # toggle off -> next cycle resumes
    eng.step()
    assert sampled == [1]


def test_max_loss_breaker_trips_on_global_day_loss(tmp_path, monkeypatch):
    _lane_status(tmp_path, "alt", pnl_today=-120.0)
    eng = _engine(tmp_path)
    monkeypatch.setattr(feed.FeedHub, "sample", lambda self: {})
    eng.discover = lambda: []
    save_control(tmp_path, {"max_loss": {"enabled": True, "limit": 100.0}})
    eng.step()
    t = read_trip(tmp_path)
    assert t is not None and "max-loss" in t["reason"] and t["day_pnl"] == -120.0
    assert eng._halted and eng._block_reason == "maxloss-trip"
    # latched: recovery does not un-trip
    _lane_status(tmp_path, "alt", pnl_today=0.0)
    eng.step()
    assert eng._block_reason == "maxloss-trip"
    clear_trip(tmp_path)                       # operator clears from dashboard
    eng.step()
    assert eng._halted is False


def test_max_loss_disabled_does_not_trip(tmp_path, monkeypatch):
    _lane_status(tmp_path, "alt", pnl_today=-500.0)
    eng = _engine(tmp_path)
    monkeypatch.setattr(feed.FeedHub, "sample", lambda self: {})
    eng.discover = lambda: []
    save_control(tmp_path, {"max_loss": {"enabled": False, "limit": 100.0}})
    eng.step()
    assert read_trip(tmp_path) is None and eng._halted is False


def test_exposure_cap_blocks_when_lanes_are_loaded(tmp_path):
    # other lanes already hold $300 of BTC against ~$2000 combined equity
    _lane_status(tmp_path, "alt", equity=1000.0, positions=[("BTC", 300.0)])
    eng = _engine(tmp_path)
    save_control(tmp_path, {"exposure_cap": {"enabled": True,
                                             "max_total_frac": 0.10,
                                             "max_asset_frac": 0.05}})
    assert eng.check_controls() is False
    m, bt, pred = _tradeable(eng)
    eng.evaluate_trade(m, bt, pred, mins_left=5.0)
    assert m.key not in eng.open_pos
    row = eng.store.conn.execute(
        "SELECT decision, block_reason FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
    assert row == ("SKIP", "exposure")


def test_exposure_cap_off_lets_the_same_trade_through(tmp_path):
    _lane_status(tmp_path, "alt", equity=1000.0, positions=[("BTC", 300.0)])
    eng = _engine(tmp_path)
    save_control(tmp_path, {"exposure_cap": {"enabled": False}})
    assert eng.check_controls() is False
    m, bt, pred = _tradeable(eng)
    eng.evaluate_trade(m, bt, pred, mins_left=5.0)
    assert m.key in eng.open_pos


def test_own_open_positions_count_toward_the_cap(tmp_path):
    eng = _engine(tmp_path)
    save_control(tmp_path, {"exposure_cap": {"enabled": True,
                                             "max_total_frac": 0.05,
                                             "max_asset_frac": 0.05}})
    assert eng.check_controls() is False
    m, bt, pred = _tradeable(eng)
    m2 = Market(venue="polymarket", market_id="m0", asset="BTC", kind="directional",
                tf="15m", window_s=900, start_ts=0.0, end_ts=9e9, strike=100.0,
                raw={}, ticker="t0")
    eng.open_pos[m2.key] = GhostPosition(
        market_key=m2.key, asset="BTC", side="up", price=0.5, size=80,
        p_model=0.6, regime="r", placed_at=time.time(), end_ts=9e9, strike=100.0,
        pred={}, ref_up_mid=0.5, market=m2)          # $40 already committed
    eng.evaluate_trade(m, bt, pred, mins_left=5.0)   # +$20 would breach 5% of $1000
    assert m.key not in eng.open_pos


def test_status_reports_control_state(tmp_path, monkeypatch):
    _lane_status(tmp_path, "alt", pnl_today=-10.0, equity=990.0,
                 positions=[("BTC", 25.0)])
    eng = _engine(tmp_path)
    monkeypatch.setattr(feed.FeedHub, "sample", lambda self: {})
    eng.discover = lambda: []
    eng.step()
    cs = eng.control_snapshot()
    assert cs["estop"] is False and cs["tripped"] is None
    assert cs["max_loss"]["day_pnl_global"] == pytest.approx(-10.0)
    assert cs["exposure_cap"]["open_cost_global"] == pytest.approx(25.0)
    assert cs["exposure_cap"]["equity_global"] == pytest.approx(1990.0)
    assert set(cs["lanes_visible"]) == {"alt", "main"}
