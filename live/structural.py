"""Structural digital-probability anchor for the near-expiry binary engine.

Production port of the two research prototypes (see research/boltzmann_upgrade.md
and research/qv_markov_upgrade.md). STDLIB ONLY.

Why this exists
---------------
Our markets are DIGITAL (payoff 1 iff S_T crosses the strike), so the only
structural quantity we need is alpha = P(S_T > k) = N(d2), the Gaussian digital
probability (this is the Boltzmann paper's `alpha` coefficient, verified there to
be identically 1 - N(d2)). Its job here is NOT to be another voting head — it is
a CALIBRATION ANCHOR against the live failure mode (overconfident, one-sided
favorite buying whose realized winrate < price).

Two pieces feed alpha:
  * NoiseRobustVol  -- per-second drift mu_s and a NOISE-ROBUST per-second vol
    sigma_s. sigma_s uses the lag-1 autocovariance (Zhou 1996) correction, which
    is the Hansen-Horel (2009) two-state/order-1 Markov QV estimator in closed
    form -- the only member of that family that is estimable on our short,
    irregular, continuous-float composite feed. De-inflates microstructure noise.
  * TerminalVarianceRegime -- a diffusion-vs-concentration multiplier on sigma
    (Boltzmann Eq. 20 motivation), used only as a mild +-25% shaping of the
    terminal dispersion, not a regime switch.

`structural_anchor` then shrinks the ensemble probability toward alpha in
log-odds (harder the more they diverge) and returns a divergence-widened EV gate.

The engine wires this in SHADOW mode first: it logs alpha / divergence / anchored
p without changing the trade decision, so alpha's calibration can be validated
out-of-sample before the gate is flipped (kill-criteria in the research notes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Digital probability math (alpha = N(d2)); identical convention to the protos #
# --------------------------------------------------------------------------- #
def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def digital_prob_up(spot: float, strike: float, tau_s: float,
                    mu_s: float, sigma_s: float) -> float:
    """P(S_T > k) = N(d2) with d2 = (log(S/k) + tau*mu_s)/(sigma_s*sqrt(tau)).

    mu_s, sigma_s are PER-SECOND log-return moments; tau_s = seconds to expiry,
    so all timeframe scaling lives in tau (mean ~ tau, vol ~ sqrt(tau))."""
    if spot <= 0 or strike <= 0 or tau_s <= 0 or sigma_s <= 0:
        return 0.5
    d2 = (math.log(spot / strike) + tau_s * mu_s) / (sigma_s * math.sqrt(tau_s))
    return max(1e-6, min(1 - 1e-6, normal_cdf(d2)))


# --------------------------------------------------------------------------- #
# Realized-variance primitives (naive RV vs noise-robust AC1 / Bartlett)       #
# --------------------------------------------------------------------------- #
def _autocov(rets: list[float], lag: int) -> float:
    n = len(rets)
    if n <= lag:
        return 0.0
    m = sum(rets) / n
    s = 0.0
    for i in range(lag, n):
        s += (rets[i] - m) * (rets[i - lag] - m)
    return s / n          # biased (divide-by-n), standard for RV/kernel estimators


def realized_variance_naive(rets: list[float]) -> float:
    return sum(r * r for r in rets)


def realized_variance_ac1(rets: list[float]) -> float:
    """Zhou (1996) / lag-1 autocovariance corrected RV = Hansen-Horel S=2,k=1
    Markov QV estimator: IV = RV + 2*n*gamma_1. Floored at 0.25*RV so a short or
    jump-dominated window can never feed a zero/negative variance downstream."""
    n = len(rets)
    if n < 2:
        return realized_variance_naive(rets)
    rv = realized_variance_naive(rets)
    iv = rv + 2.0 * n * _autocov(rets, 1)
    return max(iv, 0.25 * rv, 1e-18)


def realized_variance_kernel(rets: list[float], q: int = 2) -> float:
    """Bartlett realized kernel (q lags), PSD by construction. For our window
    lengths AC1 (q<=1) validated better; this is kept for completeness."""
    n = len(rets)
    if n < q + 1:
        return realized_variance_ac1(rets)
    k = n * _autocov(rets, 0)
    for h in range(1, q + 1):
        k += 2.0 * (1.0 - h / (q + 1.0)) * n * _autocov(rets, h)
    return max(k, 0.25 * realized_variance_naive(rets), 1e-18)


# --------------------------------------------------------------------------- #
# Online noise-robust per-second drift / vol (drop-in for a naive EWMA vol)    #
# --------------------------------------------------------------------------- #
@dataclass
class NoiseRobustVol:
    win: int = 120              # returns retained (~ a few minutes at ~8s cycles)
    kernel_q: int = 1           # 0/1 -> AC1; >=2 -> Bartlett kernel
    min_n: int = 12             # below this, fall back to naive RV (cannot correct)
    floor_sigma: float = 1e-7
    _last_t: float | None = None
    _last_p: float | None = None
    _r: list[float] = field(default_factory=list)
    _dt: list[float] = field(default_factory=list)

    def update(self, ts: float, price: float) -> None:
        if price is None or price <= 0:
            return
        if self._last_p is None or self._last_t is None:
            self._last_t, self._last_p = ts, price
            return
        dt = ts - self._last_t
        if dt <= 0:
            self._last_p = price
            return
        self._r.append(math.log(price / self._last_p))
        self._dt.append(dt)
        if len(self._r) > self.win:
            self._r.pop(0)
            self._dt.pop(0)
        self._last_t, self._last_p = ts, price

    @property
    def n(self) -> int:
        return len(self._r)

    @property
    def total_dt(self) -> float:
        return sum(self._dt) if self._dt else 0.0

    @property
    def mu_s(self) -> float:
        td = self.total_dt
        if self.n < 5 or td <= 0:
            return 0.0
        return sum(self._r) / td

    def _iv(self, robust: bool) -> float:
        if not robust or self.n < self.min_n:
            return realized_variance_naive(self._r)
        if self.kernel_q and self.kernel_q >= 2:
            return realized_variance_kernel(self._r, q=self.kernel_q)
        return realized_variance_ac1(self._r)

    def sigma_s(self, robust: bool = True) -> float:
        td = self.total_dt
        if self.n < 3 or td <= 0:
            return self.floor_sigma
        return max(math.sqrt(max(self._iv(robust) / td, 0.0)), self.floor_sigma)


# --------------------------------------------------------------------------- #
# Diffusion-vs-concentration terminal-variance shaper (Boltzmann Eq. 20)       #
# --------------------------------------------------------------------------- #
@dataclass
class TerminalVarianceRegime:
    """kappa = Var(last 2w returns) / Var(last w returns). ~1 diffusion, <1
    concentration (pinning), >1 expansion (breakout). Returns a sqrt(kappa)
    multiplier on sigma, clamped to a mild band so it shapes (not switches)."""
    w: int = 30
    cap: int = 240
    buf: list[float] = field(default_factory=list)

    def update(self, r: float) -> None:
        self.buf.append(r)
        if len(self.buf) > self.cap:
            self.buf.pop(0)

    def kappa(self) -> float | None:
        if len(self.buf) < 4 * self.w:
            return None
        recent = self.buf[-2 * self.w:]

        def var(xs):
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)

        v_w = var(recent[: self.w])
        if v_w <= 0:
            return None
        return var(recent) / v_w

    def sigma_multiplier(self, kmin: float = 0.6, kmax: float = 1.6) -> float:
        k = self.kappa()
        if k is None:
            return 1.0
        return math.sqrt(max(kmin, min(kmax, k)))

    def regime(self) -> str:
        k = self.kappa()
        if k is None:
            return "unknown"
        if k < 0.8:
            return "concentration"
        if k > 1.25:
            return "expansion"
        return "diffusion"


# --------------------------------------------------------------------------- #
# Shrinkage anchor + divergence-widened EV gate                               #
# --------------------------------------------------------------------------- #
@dataclass
class AnchorConfig:
    lam0: float = 0.35           # base shrink weight toward structural alpha
    div_scale: float = 6.0       # how fast lambda ramps with |p_ens - alpha|
    lam_max: float = 0.85
    gate_base: float = 0.01      # normal required EV margin
    gate_div_k: float = 0.40     # +0.40 * divergence added to required margin
    gate_max: float = 0.12


def structural_anchor(p_ens: float, alpha_up: float, cfg: AnchorConfig
                      ) -> tuple[float, float]:
    """Shrink ensemble up-prob toward structural alpha in log-odds (harder the
    more they diverge), and return (p_anchored, required_ev_margin)."""
    div = abs(p_ens - alpha_up)
    lam = min(cfg.lam_max, cfg.lam0 + cfg.div_scale * div * cfg.lam0)

    def lo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    p_anchored = 1.0 / (1.0 + math.exp(-((1 - lam) * lo(p_ens) + lam * lo(alpha_up))))
    gate = min(cfg.gate_max, cfg.gate_base + cfg.gate_div_k * div)
    return p_anchored, gate


# --------------------------------------------------------------------------- #
# Per-asset head bundling the above for the engine                            #
# --------------------------------------------------------------------------- #
@dataclass
class StructuralHead:
    """One per asset. Fed the composite (ts, price) ticks the engine already
    buffers; produces the structural up-probability and diagnostics for a market.
    """
    vol: NoiseRobustVol = field(default_factory=NoiseRobustVol)
    tvar: TerminalVarianceRegime = field(default_factory=TerminalVarianceRegime)
    # modest σ widening: measured σ_mkt/σ_struct≈1.23 on liquid markets (our α runs
    # slightly over-sharp). Data-derived nudge, NOT a fudge to force agreement.
    vol_scale: float = 1.25
    _last_p: float | None = None

    def update(self, ts: float, price: float) -> None:
        self.vol.update(ts, price)
        if self._last_p is not None and self._last_p > 0 and price and price > 0:
            self.tvar.update(math.log(price / self._last_p))
        if price and price > 0:
            self._last_p = price

    def evaluate(self, spot: float, strike: float, tau_s: float,
                 use_drift: bool = True) -> dict:
        """Structural digital probability for a market plus diagnostics.

        Returns alpha_up (noise-robust sigma, kappa-shaped) and alpha_naive
        (uncorrected sigma) so the shadow A/B can compare calibration."""
        mu = self.vol.mu_s if use_drift else 0.0
        sig_rb = self.vol.sigma_s(robust=True) * self.vol_scale
        sig_nv = self.vol.sigma_s(robust=False)
        kmult = self.tvar.sigma_multiplier()
        return {
            "alpha_up": digital_prob_up(spot, strike, tau_s, mu, sig_rb * kmult),
            "alpha_naive": digital_prob_up(spot, strike, tau_s, mu, sig_nv),
            "sigma_s": sig_rb,
            "sigma_naive": sig_nv,
            "mu_s": mu,
            "kappa": self.tvar.kappa(),
            "kappa_mult": kmult,
            "regime": self.tvar.regime(),
            "n": self.vol.n,
        }
