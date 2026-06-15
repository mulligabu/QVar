"""Decision-path math: fees, Kelly sizing, fill simulation, settlement P&L.

These guard the money path — every formula here moves the ledger. Pure logic,
no network: ForwardEngine is constructed against a tmp data dir and fed
synthetic history.
"""

import json
import time

import pytest

from live.forward_engine import (
    ForwardEngine,
    GhostPosition,
    Market,
    kelly_fraction,
    taker_fee,
    tf_sort_key,
    venue_taker_fee,
)


# ---------------------------------------------------------------- fees
def test_taker_fee_peaks_at_half_and_vanishes_at_extremes():
    assert taker_fee(0.5) == pytest.approx(0.072 * 0.25)
    assert taker_fee(0.01) < taker_fee(0.5)
    assert taker_fee(0.99) < taker_fee(0.5)
    assert taker_fee(0.0) == 0.0
    assert taker_fee(1.0) == 0.0
    # out-of-range prices are clamped, never negative fees
    assert taker_fee(-0.5) == 0.0
    assert taker_fee(1.5) == 0.0


def test_venue_taker_fee_polymarket_free_kalshi_charged():
    assert venue_taker_fee("polymarket", 0.5) == 0.0
    assert venue_taker_fee("kalshi", 0.5) == pytest.approx(taker_fee(0.5))


# ---------------------------------------------------------------- kelly
def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.5, 0.5) == 0.0


def test_kelly_zero_when_negative_edge():
    assert kelly_fraction(0.4, 0.5) == 0.0


def test_kelly_positive_and_capped():
    f = kelly_fraction(0.6, 0.5, cap=0.02, frac=0.5)
    assert f == 0.02  # full kelly 0.2, half 0.1, capped at 0.02
    assert kelly_fraction(0.51, 0.5, cap=0.02, frac=0.5) == pytest.approx(0.01)


def test_kelly_invalid_price_is_zero():
    assert kelly_fraction(0.6, 0.0) == 0.0
    assert kelly_fraction(0.6, 1.0) == 0.0


# ---------------------------------------------------------------- tf ordering
def test_tf_sort_key_orders_timeframes():
    tfs = ["1D", "5m", "1h", "15m", "4h"]
    assert sorted(tfs, key=tf_sort_key) == ["5m", "15m", "1h", "4h", "1D"]
    assert tf_sort_key("bogus") > tf_sort_key("1D")  # unknown sorts last


# ---------------------------------------------------------------- fill simulation
def _pos(side: str, price: float, ref_up_mid: float, market=None) -> GhostPosition:
    return GhostPosition(
        market_key="k", asset="BTC", side=side, price=price, size=10,
        p_model=0.6, regime="r", placed_at=0.0, end_ts=1.0, strike=100.0,
        pred={}, ref_up_mid=ref_up_mid, market=market,
    )


def test_fill_check_up_fills_only_when_ask_trades_through():
    pos = _pos("up", price=0.55, ref_up_mid=0.55)
    assert not pos.fill_check(up_bid=0.54, up_ask=0.56)   # ask above our limit
    assert pos.fill_check(up_bid=0.50, up_ask=0.55)       # ask at our limit
    assert pos.fill_check(up_bid=0.50, up_ask=0.50)       # ask through our limit
    assert not pos.fill_check(up_bid=0.50, up_ask=None)   # no ask -> no fill


def test_fill_check_down_fills_when_up_bid_rises_to_ref_mid():
    pos = _pos("down", price=0.45, ref_up_mid=0.55)
    assert not pos.fill_check(up_bid=0.54, up_ask=0.60)
    assert pos.fill_check(up_bid=0.55, up_ask=0.60)
    assert not pos.fill_check(up_bid=None, up_ask=0.60)


# ---------------------------------------------------------------- engine fixtures
@pytest.fixture
def engine(tmp_path):
    return ForwardEngine(["BTC"], data_dir=str(tmp_path / "run"))


def _market(end_ts: float, raw=None) -> Market:
    return Market(
        venue="polymarket", market_id="m1", asset="BTC", kind="directional",
        tf="15m", window_s=900, start_ts=end_ts - 900, end_ts=end_ts,
        strike=100.0, raw=raw or {}, ticker="btc-updown-15m-test",
    )


# ---------------------------------------------------------------- settlement P&L
def test_settle_two_leg_pnl_and_tp_overlay(engine):
    now = time.time()
    engine.cl_hist["BTC"].append((now, 101.0))  # settle above strike -> UP
    m = _market(end_ts=now, raw={"outcomePrices": '["1", "0"]'})
    pos = GhostPosition(
        market_key=m.key, asset="BTC", side="up", price=0.55, size=5,
        p_model=0.6, regime="r", placed_at=now - 60, end_ts=now, strike=100.0,
        pred={}, ref_up_mid=0.55, market=m, status="filled",
        agg_size=10, agg_price=0.60, agg_fee_per=0.01, tp_hit=True,
    )
    engine.settle(pos)
    # agg leg (1 - 0.60 - 0.01)*10 = 3.9 ; passive leg (1 - 0.55)*5 = 2.25
    assert engine.pnl == pytest.approx(6.15)
    assert engine.settled == 1 and engine.wins == 1 and engine.fills == 1
    # ghost TP sold both legs at 0.90: (0.9-0.61)*10 + (0.9-0.55)*5 = 4.65
    assert engine.tp_cum == pytest.approx(4.65)
    assert engine.tp_exits == 1
    # reconciliation against venue-reported outcome agreed
    assert engine.recon_checked == 1 and engine.recon_mismatch == 0


def test_settle_unfilled_resting_order_is_a_no_trade(engine):
    now = time.time()
    engine.cl_hist["BTC"].append((now, 99.0))
    pos = GhostPosition(
        market_key="polymarket:m1", asset="BTC", side="up", price=0.55, size=5,
        p_model=0.6, regime="r", placed_at=now - 60, end_ts=now, strike=100.0,
        pred={}, ref_up_mid=0.55, market=_market(end_ts=now),
    )
    engine.settle(pos)
    assert engine.pnl == 0.0
    assert engine.no_fills == 1 and engine.settled == 0


def test_settle_loss_books_negative_pnl(engine):
    now = time.time()
    engine.cl_hist["BTC"].append((now, 99.0))  # settles DOWN, we are UP
    pos = GhostPosition(
        market_key="polymarket:m1", asset="BTC", side="up", price=0.55, size=10,
        p_model=0.6, regime="r", placed_at=now - 60, end_ts=now, strike=100.0,
        pred={}, ref_up_mid=0.55, market=_market(end_ts=now), status="filled",
    )
    engine.settle(pos)
    assert engine.pnl == pytest.approx(-5.5)
    assert engine.wins == 0 and engine.settled == 1


def test_equity_resumes_from_db_across_restart(engine, tmp_path):
    now = time.time()
    engine.cl_hist["BTC"].append((now, 101.0))
    pos = GhostPosition(
        market_key="polymarket:m1", asset="BTC", side="up", price=0.50, size=10,
        p_model=0.6, regime="r", placed_at=now - 60, end_ts=now, strike=100.0,
        pred={}, ref_up_mid=0.50, market=_market(end_ts=now), status="filled",
    )
    engine.settle(pos)
    pnl_before = engine.pnl
    assert pnl_before == pytest.approx(5.0)
    fresh = ForwardEngine(["BTC"], data_dir=str(tmp_path / "run"))  # same dir
    assert fresh.pnl == pytest.approx(pnl_before)
    assert fresh.settled == 1 and fresh.equity == pytest.approx(1005.0)


# ---------------------------------------------------------------- config drift guard
def test_launch_args_persisted_for_config_drift_guard(engine, tmp_path):
    cfg = json.loads((tmp_path / "run" / "launch_args.json").read_text())
    assert cfg["price_lo"] == 0.30
    assert cfg["sides"] == ["down", "up"]
