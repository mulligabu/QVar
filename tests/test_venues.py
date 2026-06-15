"""Venue book parsing: BookTop normalization, Kalshi/Polymarket payload handling.

Network is monkeypatched — these test the PARSERS against captured payload
shapes (the Kalshi orderbook_fp/yes_dollars shape broke us once already).
"""

import pytest

import live.venues as venues
from live.venues import BookTop, KalshiClient, PolymarketClient, _iso


# ---------------------------------------------------------------- BookTop
def _bt(bid, ask):
    return BookTop("v", "m", 0, 1, bid, ask, 1.0, 1.0, 0.0)


def test_up_mid_averages_both_sides():
    assert _bt(0.50, 0.56).up_mid == pytest.approx(0.53)


def test_up_mid_falls_back_to_one_side():
    assert _bt(None, 0.56).up_mid == 0.56
    assert _bt(0.50, None).up_mid == 0.50
    assert _bt(None, None).up_mid is None


# ---------------------------------------------------------------- _iso
def test_iso_parses_zulu_and_offset_timestamps():
    assert _iso("2026-06-11T10:00:00Z") == pytest.approx(1781172000.0)
    assert _iso("2026-06-11T10:00:00+00:00") == _iso("2026-06-11T10:00:00Z")


def test_iso_garbage_returns_zero():
    assert _iso(None) == 0.0
    assert _iso("not-a-date") == 0.0


# ---------------------------------------------------------------- Kalshi parsing
def test_kalshi_book_top_parses_orderbook_fp_dollars(monkeypatch):
    payload = {"orderbook_fp": {
        "yes_dollars": [["0.9120", "119.78"], ["0.9000", "50"]],
        "no_dollars": [["0.0820", "1000"], ["0.0700", "10"]],
    }}
    monkeypatch.setattr(venues, "_get", lambda url, timeout=8.0: payload)
    bt = KalshiClient().book_top({"ticker": "KXBTC15M-TEST",
                                  "open_time": "2026-06-11T10:00:00Z",
                                  "close_time": "2026-06-11T10:15:00Z"})
    assert bt.up_bid == pytest.approx(0.912)        # best YES bid
    assert bt.up_ask == pytest.approx(1 - 0.082)    # 1 - best NO bid
    assert bt.up_bid_size == pytest.approx(119.78)
    assert bt.end_ts - bt.start_ts == pytest.approx(900.0)


def test_kalshi_book_top_empty_book_yields_none_sides(monkeypatch):
    monkeypatch.setattr(venues, "_get", lambda url, timeout=8.0: {"orderbook_fp": {}})
    bt = KalshiClient().book_top({"ticker": "T", "open_time": None, "close_time": None})
    assert bt.up_bid is None and bt.up_ask is None
    assert bt.up_mid is None


def test_kalshi_book_top_network_error_returns_none(monkeypatch):
    def boom(url, timeout=8.0):
        raise OSError("connection reset")
    monkeypatch.setattr(venues, "_get", boom)
    assert KalshiClient().book_top({"ticker": "T"}) is None


# ---------------------------------------------------------------- Polymarket parsing
def test_polymarket_book_top_selects_best_levels(monkeypatch):
    payload = {
        "bids": [{"price": "0.52", "size": "100"}, {"price": "0.54", "size": "30"}],
        "asks": [{"price": "0.58", "size": "40"}, {"price": "0.56", "size": "20"}],
    }
    monkeypatch.setattr(venues, "_get", lambda url, timeout=8.0: payload)
    market = {"id": "123", "clobTokenIds": '["tokUP", "tokDOWN"]',
              "_window_s": 900, "_event_start": 1000}
    bt = PolymarketClient().book_top(market)
    assert bt.up_bid == pytest.approx(0.54)   # highest bid
    assert bt.up_ask == pytest.approx(0.56)   # lowest ask
    assert bt.up_bid_size == pytest.approx(30)
    assert bt.up_ask_size == pytest.approx(20)
    assert bt.up_mid == pytest.approx(0.55)


def test_polymarket_book_top_no_tokens_returns_none(monkeypatch):
    monkeypatch.setattr(venues, "_get", lambda url, timeout=8.0: {})
    assert PolymarketClient().book_top({"id": "1", "clobTokenIds": "[]"}) is None
