"""Online predictor: heads learn, tilt cap binds, calibration decays,
conditional-skill gate flips/abstains correctly, state round-trips."""

import math
import random

import pytest

from live.predictor import (
    ConditionalSkill,
    Head,
    OnlinePredictor,
    Welford,
    build_features,
    classify_regime,
    entropy,
    sigmoid,
)


# ---------------------------------------------------------------- primitives
def test_sigmoid_symmetry_and_stability():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(5.0) + sigmoid(-5.0) == pytest.approx(1.0)
    assert 0.0 < sigmoid(-800.0) < 1e-300 or sigmoid(-800.0) == 0.0  # no overflow
    assert sigmoid(800.0) == pytest.approx(1.0)


def test_entropy_bounds():
    assert entropy(0.5) == pytest.approx(1.0)
    assert entropy(0.0) == pytest.approx(0.0, abs=1e-6)
    assert entropy(1.0) == pytest.approx(0.0, abs=1e-6)


def test_welford_matches_population_moments():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    w = Welford()
    for x in xs:
        w.update(x)
    assert w.mean == pytest.approx(3.0)
    assert w.std == pytest.approx(math.sqrt(2.0))  # population (divide-by-n)


# ---------------------------------------------------------------- heads
def test_head_learns_a_signed_signal():
    rng = random.Random(7)
    h = Head("t", ["x"])
    for _ in range(400):
        x = rng.gauss(0, 1)
        outcome = 1 if x > 0 else 0
        h.learn({"x": x}, outcome)
    assert h.w["x"] > 0.2
    assert h.predict({"x": 2.0}) > 0.6
    assert h.predict({"x": -2.0}) < 0.4
    assert h.reliability > 0.5  # better than coinflip log-score


# ---------------------------------------------------------------- tilt cap
def test_prediction_is_anchored_within_tilt_cap_of_mid():
    p = OnlinePredictor(tilt_cap=0.4)
    feats = {"z_dist": 5.0, "momentum": 5.0, "realized_vol": 0.5,
             "efficiency": 0.5, "ob_imbalance": 0.9, "log_tau": 1.0}
    for mid in (0.3, 0.5, 0.65):
        pred = p.predict(feats, market_mid=mid)
        mkt_logit = math.log(mid / (1 - mid))
        p_logit = math.log(pred.p_up / (1 - pred.p_up))
        assert abs(p_logit - mkt_logit) <= 0.4 + 1e-9


# ---------------------------------------------------------------- calibration decay
def test_calibration_load_clamps_stale_cumulative_counts():
    p = OnlinePredictor()
    cap = 1.0 / (1.0 - p.cal_decay)
    state = p.state()
    state["cal"]["5"] = [5000.0, 2500.0]  # huge pre-decay cumulative bucket
    fresh = OnlinePredictor()
    fresh.load_state(state)
    n, wins = fresh.cal[5]
    assert n == pytest.approx(cap)
    assert wins / n == pytest.approx(0.5)  # rate preserved


def test_learn_decays_calibration_buckets():
    p = OnlinePredictor()
    pred = p.predict({"z_dist": 0.0, "momentum": 0.0, "realized_vol": 0.5,
                      "efficiency": 0.3, "ob_imbalance": 0.0, "log_tau": 1.0},
                     market_mid=0.5)
    b = min(9, int(pred.p_up * 10))
    for _ in range(2000):
        p.learn(pred, outcome_up=1, market_mid=0.5)
    n, wins = p.cal[b]
    assert n <= 1.0 / (1.0 - p.cal_decay) + 1.0  # bounded by the decay window
    assert wins / n > 0.9  # tracked the all-up tape


# ---------------------------------------------------------------- conditional skill
def _feed_cell(cs: ConditionalSkill, tf, mid, lean_up_wins: bool, n=60):
    # up-leans resolve up, down-leans resolve down (raw signal predictive)...
    for _ in range(n):
        cs.learn(tf, mid, mid + 0.05, 1 if lean_up_wins else 0)
        cs.learn(tf, mid, mid - 0.05, 0 if lean_up_wins else 1)


def test_conditional_skill_warmup_then_trades_raw_side():
    cs = ConditionalSkill()
    assert cs.assess("15m", 0.5, 0.55)[1] == 0  # warmup abstains
    _feed_cell(cs, "15m", 0.5, lean_up_wins=True)
    status, factor, conv, _ = cs.assess("15m", 0.5, 0.55)
    assert status == "OK" and factor == +1 and conv > 0.08


def test_conditional_skill_flips_an_inverted_cell():
    cs = ConditionalSkill()
    _feed_cell(cs, "15m", 0.5, lean_up_wins=False)  # raw signal backwards
    status, factor, _, _ = cs.assess("15m", 0.5, 0.55)
    assert status == "OK" and factor == -1


def test_conditional_skill_no_skill_cell_abstains():
    cs = ConditionalSkill()
    for i in range(80):  # both leans resolve up half the time -> gap ~ 0
        cs.learn("15m", 0.5, 0.55, i % 2)
        cs.learn("15m", 0.5, 0.45, i % 2)
    status, factor, _, _ = cs.assess("15m", 0.5, 0.55)
    assert status == "NOSKILL" and factor == 0


def test_conditional_skill_blacklist_abstains_but_learns():
    cs = ConditionalSkill()
    ConditionalSkill.BLACKLIST.add("15m|mid")
    try:
        _feed_cell(cs, "15m", 0.5, lean_up_wins=True)
        status, factor, _, _ = cs.assess("15m", 0.5, 0.55)
        assert status == "BLACKLIST" and factor == 0
        assert cs.cells["15m|mid"]["up"][0] > 0  # still accumulated
    finally:
        ConditionalSkill.BLACKLIST.clear()  # currently empty by user decision


def test_conditional_skill_band_edges():
    cs = ConditionalSkill()
    assert cs._key("15m", 0.44) == "15m|lo"
    assert cs._key("15m", 0.45) == "15m|mid"
    assert cs._key("15m", 0.55) == "15m|hi"
    assert cs._key("15m", 0.99) == "15m|hi"


# ---------------------------------------------------------------- state round-trip
def test_predictor_state_roundtrip_preserves_predictions():
    rng = random.Random(3)
    p = OnlinePredictor()
    feats = {"z_dist": 0.4, "momentum": 0.2, "realized_vol": 0.6,
             "efficiency": 0.35, "ob_imbalance": 0.1, "log_tau": 1.2}
    for _ in range(120):
        f = dict(feats, z_dist=rng.gauss(0, 1))
        pred = p.predict(f, market_mid=0.5)
        p.learn(pred, rng.randint(0, 1), market_mid=0.5, bucket_key="r|15m")
    clone = OnlinePredictor()
    clone.load_state(p.state())
    a = p.predict(feats, market_mid=0.55)
    b = clone.predict(feats, market_mid=0.55)
    assert a.p_cal == pytest.approx(b.p_cal, abs=1e-9)
    assert a.regime == b.regime
    assert clone.n_updates == p.n_updates


# ---------------------------------------------------------------- regimes
def test_classify_regime_panic_needs_history():
    w = Welford()
    for x in (0.1, 0.1, 0.1, 0.12, 0.11, 0.1, 0.1, 0.11, 0.1, 0.12, 0.11):
        w.update(x)
    assert classify_regime(5.0, 0.5, w) == "panic"
    assert classify_regime(0.11, 0.5, w).endswith("/trend")
    assert classify_regime(0.11, 0.1, w).endswith("/mean_revert")


# ---------------------------------------------------------------- features
def test_build_features_is_causal():
    now = 1000.0
    hist = [(now - 60 + i * 2, 100.0 + i * 0.01) for i in range(30)]
    future = [*hist, (now + 10, 200.0), (now + 20, 300.0)]  # leak bait
    a = build_features(hist, 100.3, 100.0, now + 300, now, 0.0)
    b = build_features(future, 100.3, 100.0, now + 300, now, 0.0)
    assert a is not None
    assert a == b  # future points must not change anything


def test_build_features_rejects_thin_or_degenerate_input():
    now = 1000.0
    assert build_features([(now - 1, 100.0)], 100.0, 100.0, now + 60, now, 0.0) is None
    hist = [(now - 60 + i * 2, 100.0 + i * 0.01) for i in range(30)]
    assert build_features(hist, 100.0, 0.0, now + 60, now, 0.0) is None  # bad strike


def test_build_features_z_sign_follows_spot_vs_strike():
    now = 1000.0
    hist = [(now - 60 + i * 2, 100.0 + (i % 3) * 0.02) for i in range(30)]
    above = build_features(hist, 101.0, 100.0, now + 300, now, 0.0)
    below = build_features(hist, 99.0, 100.0, now + 300, now, 0.0)
    assert above["z_dist"] > 0 > below["z_dist"]
