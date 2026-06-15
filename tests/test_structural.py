"""Structural anchor math: digital probability, noise-robust vol, shrinkage."""

import math

import pytest

from live.structural import (
    AnchorConfig,
    NoiseRobustVol,
    StructuralHead,
    TerminalVarianceRegime,
    digital_prob_up,
    normal_cdf,
    realized_variance_ac1,
    realized_variance_naive,
    structural_anchor,
)


def test_normal_cdf_known_values():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.0) == pytest.approx(0.841345, abs=1e-5)
    assert normal_cdf(-1.0) == pytest.approx(1 - normal_cdf(1.0))


def test_digital_prob_at_the_money_is_half():
    assert digital_prob_up(100.0, 100.0, 60.0, 0.0, 1e-4) == pytest.approx(0.5)


def test_digital_prob_deep_in_and_out_of_money():
    assert digital_prob_up(110.0, 100.0, 60.0, 0.0, 1e-4) > 0.99
    assert digital_prob_up(90.0, 100.0, 60.0, 0.0, 1e-4) < 0.01


def test_digital_prob_degenerate_inputs_return_half():
    assert digital_prob_up(0.0, 100.0, 60.0, 0.0, 1e-4) == 0.5
    assert digital_prob_up(100.0, 100.0, 0.0, 0.0, 1e-4) == 0.5
    assert digital_prob_up(100.0, 100.0, 60.0, 0.0, 0.0) == 0.5


def test_ac1_variance_floored_at_quarter_of_naive():
    # alternating returns -> strongly negative autocov -> floor engages
    rets = [0.01, -0.01] * 20
    rv = realized_variance_naive(rets)
    assert realized_variance_ac1(rets) == pytest.approx(0.25 * rv)


def test_noise_robust_vol_positive_after_updates():
    v = NoiseRobustVol()
    px = 100.0
    for i in range(60):
        px *= 1.0 + (0.001 if i % 2 else -0.0008)
        v.update(float(i * 8), px)
    assert v.n > 50
    assert v.sigma_s() > 0
    v.update(59 * 8.0, px)  # dt <= 0 must not corrupt the buffer
    assert v.n <= 60


def test_terminal_variance_regime_needs_warmup_then_classifies():
    t = TerminalVarianceRegime(w=10)
    assert t.kappa() is None and t.sigma_multiplier() == 1.0
    for _ in range(20):
        t.update(0.001)          # quiet first
    for i in range(20):
        t.update(0.01 * (1 if i % 2 else -1))  # then loud
    assert t.kappa() is not None
    assert t.regime() in ("concentration", "diffusion", "expansion")


def test_structural_anchor_shrinks_toward_alpha_and_widens_gate():
    cfg = AnchorConfig()
    p, gate = structural_anchor(0.80, 0.50, cfg)
    assert 0.50 < p < 0.80                       # pulled toward alpha
    assert cfg.gate_base < gate <= cfg.gate_max  # divergence widened the gate
    _p2, gate2 = structural_anchor(0.51, 0.50, cfg)
    assert gate2 < gate                          # small divergence, small gate


def test_structural_head_evaluate_shape():
    h = StructuralHead()
    px = 100.0
    for i in range(80):
        px *= 1.0 + (0.0006 if i % 3 else -0.0005)
        h.update(float(i * 8), px)
    out = h.evaluate(px, px * 0.999, 300.0)
    assert 0.0 < out["alpha_up"] < 1.0
    assert out["sigma_s"] > 0 and out["n"] > 70
    assert math.isfinite(out["alpha_naive"])
