"""SQLite store: schema/migrations idempotency, ledger resume, analytics reads."""

import pytest

from live.db import ReadStore, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db")


def _settlement(**kw):
    base = {"ts": "2026-06-11T00:00:00Z", "market": "polymarket:1", "ticker": "t",
            "asset": "BTC", "venue": "polymarket", "tf": "15m", "side": "up",
            "strike": 100.0, "end_price": 101.0, "outcome_up": 1, "won": 1,
            "filled": 1, "fill_price": 0.55, "pnl": 4.5, "cum_pnl": 4.5}
    base.update(kw)
    return base


def test_migrations_are_idempotent(tmp_path):
    Store(tmp_path / "m.db")
    Store(tmp_path / "m.db")  # second open must not raise on ALTERs


def test_settlement_totals_resume_ledger(store):
    store.insert_settlement(_settlement(pnl=4.5, won=1))
    store.insert_settlement(_settlement(pnl=-5.5, won=0, market="polymarket:2"))
    store.insert_settlement(_settlement(filled=0, pnl=None, won=None,
                                        market="polymarket:3"))  # no-fill
    t = store.settlement_totals()
    assert t["pnl"] == pytest.approx(-1.0)
    assert t["wins"] == 1 and t["settled"] == 2 and t["no_fills"] == 1


def test_tp_totals_empty_then_resume(store):
    assert store.tp_totals()["cum"] is None  # lets the engine seed at real pnl
    store.insert_settlement(_settlement(tp_pnl=3.5, tp_cum_pnl=3.5, tp_exited=1))
    store.insert_settlement(_settlement(market="polymarket:2", won=1,
                                        tp_pnl=4.5, tp_cum_pnl=8.0, tp_exited=0))
    g = store.tp_totals()
    assert g["cum"] == pytest.approx(8.0)   # latest running total
    assert g["exits"] == 1 and g["settled"] == 2 and g["wins"] == 2


def test_readstore_equity_curve_and_recent(tmp_path):
    s = Store(tmp_path / "r.db")
    for i in range(3):
        s.insert_settlement(_settlement(market=f"polymarket:{i}", cum_pnl=float(i)))
    s.insert_settlement(_settlement(market="polymarket:nf", filled=0))
    rs = ReadStore(tmp_path / "r.db")
    curve = rs.equity_curve()
    assert [r["cum_pnl"] for r in curve] == [0.0, 1.0, 2.0]  # filled only, in order
    assert len(rs.recent_settlements()) == 3                  # no-fill rows excluded
    assert len(rs.recent("settlements")) == 4


def test_readstore_calibration_scores_perfect_signal(tmp_path):
    s = Store(tmp_path / "c.db")
    for i in range(40):
        o = i % 2
        s.insert_calib({"ts": "t", "market": f"m{i}", "asset": "BTC", "tf": "15m",
                        "market_mid": 0.5, "p_cal": 0.9 if o else 0.1,
                        "alpha_up": None, "outcome_up": o})
    c = ReadStore(tmp_path / "c.db").calibration()
    assert c["ready"] and c["n"] == 40
    assert c["ensemble"]["auc"] == pytest.approx(1.0)   # perfectly ranked
    assert c["ensemble"]["brier"] < c["market"]["brier"]
    assert c["directional"]["disagreement_auc"] == pytest.approx(1.0)
    assert "beat the market" in c["verdict"]


def test_readstore_throughput_aggregates(tmp_path):
    s = Store(tmp_path / "tp.db")
    s.insert_decision({"ts": "t", "market": "m", "asset": "BTC", "tf": "15m",
                       "decision": "TRADE"})
    s.insert_decision({"ts": "t", "market": "m2", "asset": "BTC", "tf": "15m",
                       "decision": "SKIP"})
    s.insert_settlement(_settlement())
    out = ReadStore(tmp_path / "tp.db").throughput()
    assert out["by_tf"]["15m"]["decisions"] == 2
    assert out["by_tf"]["15m"]["trades"] == 1
    assert out["by_asset"]["BTC"]["settled"] == 1
