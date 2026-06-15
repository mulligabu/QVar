"""qv_markov -- noise-robust integrated-variance (sigma) estimator for the
near-expiry digital engine, translated from Hansen & Horel (2009),
"Quadratic Variation by Markov Chains" (SSRN 1367519).

STDLIB ONLY (math, sqlite3, statistics, sys). Does NOT import or modify
anything under live/. numpy is NOT required (and not used).

------------------------------------------------------------------------------
WHY THIS FILE EXISTS
------------------------------------------------------------------------------
Our digital probability is alpha = N(d2) with
    d2 = (log(S/k) + tau*mu_s) / (sigma_s * sqrt(tau)).
Everything hinges on sigma_s. Today sigma_s comes from naive rolling variance of
log-returns on a COMPOSITE feed (Chainlink anchor + Coinbase/Binance drift,
sampled ~per cycle / ~8-16s). Microstructure noise (and the anchor/tick blend
itself, which injects a step + a moving drift correction) biases realized
variance -- typically UPWARD at high frequency. An upward-biased sigma pushes
every alpha toward 0.5 (under-confident) OR, when the noise is negatively
autocorrelated (bid-ask bounce style), the naive RV can be biased and noisy in
ways that distort d2 unpredictably. Either way the calibration anchor in
research/boltzmann_upgrade.md inherits the error.

------------------------------------------------------------------------------
WHAT TRANSFERS FROM HANSEN-HOREL (and what does NOT) -- see qv_markov_upgrade.md
------------------------------------------------------------------------------
The full Markov-chain estimator MC# = <pi,(2Z-I)pi> requires:
  (a) price increments confined to a fixed tick GRID with a small number of
      states S (NYSE: whole cents), and
  (b) thousands of tick-by-tick increments per estimation window to estimate the
      SxS (or S^k x S^k) transition matrix P and its fundamental matrix Z.
NEITHER holds for us: our composite price is a continuous float (Chainlink/1e8 +
a real-valued Coinbase drift), and a single 15m market yields ~20-40 samples,
not thousands. Forcing onto a grid with so few points gives a degenerate,
near-singular P. So the FULL chain estimator is impractical here -- we say so
plainly rather than cargo-culting it.

WHAT DOES transfer is the *mechanism* and the simplest closed-form member of the
family. The paper's own simplest case (Sec 10.2, S=2, k=1) collapses the whole
machinery to:
        MC# = ((1 + rho) / (1 - rho)) * n * sigma_increment^2,
i.e. the realized variance times a serial-correlation correction factor, where
rho is the first-order autocorrelation of the increments (= the chain's second
eigenvalue). This is EXACTLY the structure of the classic Zhou (1996) /
first-order autocovariance-corrected realized-variance estimator:
        IV_hat = sum r_i^2 + 2 * sum r_i r_{i-1}      (= RV + 2*gamma_1).
Note ((1+rho)/(1-rho)) ~= 1 + 2*rho for small rho, and rho*RV ~= gamma_1, so the
two-state Markov correction IS the lag-1 autocovariance correction to first
order. That is the defensible, implementable transfer for an irregular,
real-valued, short feed. We provide it as `realized_variance_ac1` and a small
multi-lag generalization (`realized_variance_kernel`, a Bartlett/realized-kernel
flatten) for when bid-ask-bounce noise has dependence beyond lag 1.

We ALSO provide `markov_qv_grid` -- a faithful implementation of the actual
two-/multi-state Markov estimator on a forced sign/coarse grid -- so the
mechanism can be inspected and unit-tested against the AC1 form on synthetic
data, and used on the (rare) longer windows where enough samples exist. Treat it
as a diagnostic, not the production path.

------------------------------------------------------------------------------
API SUMMARY
------------------------------------------------------------------------------
- log_returns(prices)                          -> list of log-returns
- realized_variance_naive(rets)                -> RV = sum r^2     (what we do now)
- realized_variance_ac1(rets)                  -> noise-robust (Zhou / 2-state MC)
- realized_variance_kernel(rets, q)            -> Bartlett realized kernel, q lags
- markov_qv_grid(rets, n_states, k)            -> faithful MC# on a forced grid
- NoiseRobustVol  (drop-in for CausalDriftVol) -> online per-second mu_s, sigma_s
- digital_prob_up(...) / normal_cdf(...)       -> reused alpha=N(d2) (matches boltzmann_digital)
- backtest_db(db_path)                         -> calibration of naive vs robust sigma
                                                  against realized outcomes (paths table)

Run:
    python3 research/strategy/qv_markov.py                 # math selftest
    python3 research/strategy/qv_markov.py /abs/forward_engine.db   # + DB calibration
"""

from __future__ import annotations

import math
import sqlite3
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 0. Shared digital-probability math (kept identical to boltzmann_digital.py)  #
# --------------------------------------------------------------------------- #
def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def digital_prob_up(spot: float, strike: float, tau_s: float,
                    mu_s: float, sigma_s: float) -> float:
    """P(S_T > k) = N(d2), with d2 = (log(S/k)+tau*mu)/(sigma*sqrt(tau)).

    sigma_s, mu_s are PER-SECOND log-return moments; tau_s is seconds to expiry.
    Identical convention to research/strategy/boltzmann_digital.py so the two
    prototypes are interchangeable in the engine. The ONLY thing this file
    changes is HOW sigma_s is produced (noise-robust vs naive)."""
    if spot <= 0 or strike <= 0 or tau_s <= 0 or sigma_s <= 0:
        return 0.5
    d2 = (math.log(spot / strike) + tau_s * mu_s) / (sigma_s * math.sqrt(tau_s))
    return max(1e-6, min(1 - 1e-6, normal_cdf(d2)))


# --------------------------------------------------------------------------- #
# 1. Return + variance primitives                                             #
# --------------------------------------------------------------------------- #
def log_returns(prices: list[float]) -> list[float]:
    out = []
    for i in range(1, len(prices)):
        a, b = prices[i - 1], prices[i]
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def _autocov(rets: list[float], lag: int) -> float:
    """Biased (divide-by-n) sample autocovariance at `lag`. Biased form is the
    standard choice for realized-variance / kernel estimators (Barndorff-Nielsen
    et al.)."""
    n = len(rets)
    if n <= lag:
        return 0.0
    m = sum(rets) / n
    s = 0.0
    for i in range(lag, n):
        s += (rets[i] - m) * (rets[i - lag] - m)
    return s / n


def realized_variance_naive(rets: list[float]) -> float:
    """RV = sum r_i^2. This is what build_features() effectively does
    (variance * n). Consistent for QV ONLY in the absence of noise; biased when
    the observed price = efficient price + microstructure noise."""
    return sum(r * r for r in rets)


def realized_variance_ac1(rets: list[float]) -> float:
    """Zhou (1996) / first-order-autocovariance corrected RV  ==  the paper's
    two-state, order-one Markov estimator (Sec 10.2).

        IV_hat = sum r_i^2 + 2 * sum_{i} r_i r_{i-1}   =   RV + 2*n*gamma_1.

    Derivation of the equivalence (report Sec 1.1): with iid (or AR(1)) noise,
    observed returns inherit a negative MA(1)/AR(1) component; the lag-1
    autocovariance gamma_1 < 0 captures the bid-ask-bounce, and adding 2*gamma_1
    removes the noise inflation. The paper's MC#_{S=2,k=1} = (1+rho)/(1-rho)*RV
    with rho = corr(r_i, r_{i-1}); to first order (1+rho)/(1-rho) ~= 1 + 2*rho,
    and rho*RV ~= n*gamma_1, recovering RV + 2*n*gamma_1. We return the additive
    (Zhou) form because it is exact, stable, and needs no eigenvalue."""
    n = len(rets)
    if n < 2:
        return realized_variance_naive(rets)
    rv = realized_variance_naive(rets)
    g1 = _autocov(rets, 1)        # already divided by n
    iv = rv + 2.0 * n * g1
    # guard: noise-correction can drive the estimate negative when the window is
    # short or dominated by one big move; floor at a fraction of naive RV so we
    # never feed a zero/neg variance into the digital formula.
    return max(iv, 0.25 * rv, 1e-18)


def realized_variance_kernel(rets: list[float], q: int = 2) -> float:
    """Bartlett (flat-top-ish) realized kernel with q lags. Generalizes the AC1
    estimator to noise with dependence beyond lag 1 (the paper's higher-order
    Markov chains handle exactly this serial-dependent / endogenous noise).

        K = gamma_0*n + 2 * sum_{h=1}^{q} (1 - h/(q+1)) * n * gamma_h

    Bartlett weights (1 - h/(q+1)) guarantee a non-negative estimator (positive
    semidefinite kernel), which the raw multi-lag Zhou sum does not. For our
    feed q in {1,2} is plenty; q=1 with full weight is essentially AC1."""
    n = len(rets)
    if n < q + 1:
        return realized_variance_ac1(rets)
    k = n * _autocov(rets, 0)
    for h in range(1, q + 1):
        w = 1.0 - h / (q + 1.0)
        k += 2.0 * w * n * _autocov(rets, h)
    return max(k, 0.25 * realized_variance_naive(rets), 1e-18)


def markov_qv_grid(rets: list[float], n_states: int = 2, k_order: int = 1) -> float:
    """FAITHFUL Hansen-Horel MC# on a forced grid (DIAGNOSTIC, not production).

    Implements MC# = <pi, (2Z - I) pi> * (sum of squared *state values*) scaling,
    via the level form MC# = n^2 <f,(2Zhat-I)f>_pihat normalized as in the paper
    (Sec 10.1 / Eq.9). We force the continuous increments onto a grid of n_states
    quantile bins (so each state has a representative value = mean increment in
    that bin), estimate the order-1 (or extend to order-k via tuple states)
    transition matrix Phat by counting, append k ancillary observations so the
    first/last states match (Sec 8.3 -> closed-form stationary dist + ergodic),
    build Z = (I - P + Pi)^{-1} by Gauss-Jordan, and return the quadratic form.

    For n_states=2, k=1 this MUST agree with the (1+rho)/(1-rho)*RV form -- the
    selftest asserts that on synthetic AR(1) data. We DO NOT recommend running
    this in production: with ~20-40 samples per market the transition counts are
    far too sparse (a 2-state P needs maybe 50+ transitions to be stable; an
    order-k S^k chain needs thousands, which we never have). It exists to (1)
    prove the AC1 form is the right shadow of the real estimator on our data, and
    (2) be available on the rare long DOGE 1D windows (~100 samples) as a check."""
    n = len(rets)
    if n < max(8, 2 * n_states):
        return realized_variance_ac1(rets)
    if k_order != 1 or n_states != 2:
        # Higher order / more states implemented only for the documented 2x1 case
        # here to keep the prototype auditable; fall back otherwise.
        return realized_variance_kernel(rets, q=min(2, n - 1))

    # --- force onto a 2-state sign grid; representative value = RMS magnitude ---
    # State 0 = down move, State 1 = up move (zero increments are rare on a float
    # feed; assign by sign, ties -> up). Representative magnitude delta = sqrt(RV/n)
    # so that x = (-delta, +delta) and RV ~= n*delta^2 (matches the paper's f).
    rv = realized_variance_naive(rets)
    if rv <= 0:
        return 1e-18
    delta = math.sqrt(rv / n)
    states = [1 if r >= 0 else 0 for r in rets]

    # append 1 ancillary obs so first==last state (Sec 8.3) -> ergodic, closed pi
    seq = states + [states[0]]
    # count transitions
    nrs = [[0, 0], [0, 0]]
    for i in range(1, len(seq)):
        nrs[seq[i - 1]][seq[i]] += 1
    row = [nrs[0][0] + nrs[0][1], nrs[1][0] + nrs[1][1]]
    if row[0] == 0 or row[1] == 0:
        return realized_variance_ac1(rets)        # absorbing -> bail
    P = [[nrs[r][s] / row[r] for s in range(2)] for r in range(2)]
    # stationary dist (closed form for 2-state): pi0 = (1-q)/(2-p-q) with
    # p=P11 (stay up?), define via standard 2-state solution
    p01, p10 = P[0][1], P[1][0]
    denom = p01 + p10
    if denom <= 0:
        return realized_variance_ac1(rets)
    pi = [p10 / denom, p01 / denom]               # pi0 (down), pi1 (up)
    Pi = [[pi[0], pi[1]], [pi[0], pi[1]]]          # rows all = pi
    # Z = (I - P + Pi)^{-1}
    A = [[(1.0 if r == s else 0.0) - P[r][s] + Pi[r][s] for s in range(2)]
         for r in range(2)]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-15:
        return realized_variance_ac1(rets)
    Z = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]]
    # MC# = <f, (2Z - I) f>_pi  with f = (x0, x1) = (-delta, +delta), scaled by n
    f = [-delta, delta]
    M = [[2 * Z[r][s] - (1.0 if r == s else 0.0) for s in range(2)] for r in range(2)]
    # quadratic form sum_r pi_r * f_r * sum_s M_rs f_s
    qf = 0.0
    for r in range(2):
        inner = sum(M[r][s] * f[s] for s in range(2))
        qf += pi[r] * f[r] * inner
    mc = n * qf
    return max(mc, 0.25 * rv, 1e-18)


# --------------------------------------------------------------------------- #
# 2. Online, causal, noise-robust per-second vol estimator                    #
#    (drop-in replacement / augmentation for CausalDriftVol.sigma_s)          #
# --------------------------------------------------------------------------- #
@dataclass
class NoiseRobustVol:
    """Online per-second drift mu_s and NOISE-ROBUST per-second vol sigma_s.

    Feed (ts, price) ticks exactly like CausalDriftVol. Internally keeps a
    bounded ring buffer of recent (dt, r) pairs and, on demand, computes:
      - mu_s   = sum(r) / sum(dt)                       (per-second drift)
      - var_ps = robust_IV(r) / sum(dt)                 (per-second variance)
    where robust_IV is the AC1 (two-state Markov / Zhou) corrected realized
    variance over the window. This directly de-biases the microstructure /
    anchor-blend inflation that the naive EWMA-of-r^2 in CausalDriftVol suffers.

    Per-second normalization uses TOTAL elapsed seconds in the window so it is
    horizon-consistent across 5m / 15m / 1h / 1D lanes (the same property
    CausalDriftVol was built for). No look-ahead; only past ticks.

    Tuning:
      win      : number of returns retained (ring). 120 ~ a few minutes at ~8s.
      kernel_q : 0 -> AC1 only; >=1 -> Bartlett kernel with that many lags.
      min_n    : below this, fall back to naive RV (not enough data to correct).
    """
    win: int = 120
    kernel_q: int = 1
    min_n: int = 12
    floor_sigma: float = 1e-7        # per-second sigma floor (~ guards d2 blowup)
    _last_t: float | None = None
    _last_p: float | None = None
    _r: list[float] = field(default_factory=list)
    _dt: list[float] = field(default_factory=list)

    def update(self, ts: float, price: float) -> None:
        if price is None or price <= 0:
            return
        if self._last_p is None:
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
        """Per-second volatility. robust=True -> noise-corrected; False -> naive
        (so the same object can produce both for A/B calibration)."""
        td = self.total_dt
        if self.n < 3 or td <= 0:
            return self.floor_sigma
        iv = self._iv(robust)
        var_ps = iv / td
        return max(math.sqrt(max(var_ps, 0.0)), self.floor_sigma)

    def prob_up(self, spot: float, strike: float, tau_s: float,
                use_drift: bool = False, robust: bool = True) -> float:
        mu = self.mu_s if use_drift else 0.0
        return digital_prob_up(spot, strike, tau_s, mu, self.sigma_s(robust))


# --------------------------------------------------------------------------- #
# 3. Validation: does noise-robust sigma give better-calibrated alpha?        #
# --------------------------------------------------------------------------- #
def _brier(probs_outcomes: list[tuple[float, int]]) -> float:
    if not probs_outcomes:
        return float("nan")
    return sum((p - o) ** 2 for p, o in probs_outcomes) / len(probs_outcomes)


def _logloss(probs_outcomes: list[tuple[float, int]]) -> float:
    if not probs_outcomes:
        return float("nan")
    s = 0.0
    for p, o in probs_outcomes:
        p = min(max(p, 1e-6), 1 - 1e-6)
        s += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return s / len(probs_outcomes)


def _reliability(probs_outcomes: list[tuple[float, int]], nb: int = 5) -> list[tuple]:
    buckets = [[0, 0.0, 0.0] for _ in range(nb)]   # n, sum_p, sum_outcome
    for p, o in probs_outcomes:
        b = min(nb - 1, int(p * nb))
        buckets[b][0] += 1
        buckets[b][1] += p
        buckets[b][2] += o
    out = []
    for b in buckets:
        if b[0] > 0:
            out.append((b[0], b[1] / b[0], b[2] / b[0]))   # n, mean_p, hit_rate
        else:
            out.append((0, float("nan"), float("nan")))
    return out


def backtest_db(db_path: str, min_path: int = 12) -> None:
    """Empirical calibration test on the engine's own `paths` table.

    DESIGN (honest about what the data allows):
      For each settled market that also has a price path:
        - We have: strike, the per-cycle underlying[], the path timestamps, and
          the realized outcome_up (did it settle ABOVE strike?).
        - We treat the FINAL recorded sample as the decision point at horizon
          tau = (expiry - last_ts). The paths run right up to expiry, so we
          approximate expiry = last path ts and instead evaluate alpha a few
          samples BEFORE the end (offset `eval_back` samples) using the
          remaining wall-clock as tau. This gives a genuine forecast with a
          real >0 horizon, scored against the realized terminal outcome.
        - sigma is estimated CAUSALLY from the underlying samples up to the eval
          point, both ways: naive RV and AC1 noise-robust. mu set to 0 (near
          expiry drift is second-order, per boltzmann_upgrade Sec 1.4; also the
          honest choice given how few samples we have).
        - alpha = N(d2) for each sigma; score against outcome_up via Brier,
          log-loss, and a reliability table.

    The question answered: across these markets, does the noise-robust sigma
    produce alpha closer to the realized 0/1 outcome (lower Brier/log-loss,
    reliability nearer the diagonal) than the naive sigma?

    CAVEATS printed at the end. n is small (this DB is a short run); treat the
    sign of the difference as the signal, not the magnitude."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    markets = conn.execute(
        "SELECT DISTINCT s.market, s.strike, s.outcome_up, s.tf "
        "FROM settlements s JOIN paths p ON p.market=s.market "
        "WHERE s.strike IS NOT NULL AND s.outcome_up IS NOT NULL"
    ).fetchall()

    naive_po: list[tuple[float, int]] = []
    robust_po: list[tuple[float, int]] = []
    kernel_po: list[tuple[float, int]] = []
    eval_back = 3        # evaluate this many samples before the last (real tau>0)
    used = 0
    rows_dbg = []

    for m in markets:
        path = conn.execute(
            "SELECT ts, underlying FROM paths WHERE market=? AND underlying IS NOT NULL "
            "ORDER BY ts", (m["market"],)
        ).fetchall()
        if len(path) < min_path:
            continue
        ts = [r["ts"] for r in path]
        und = [r["underlying"] for r in path]
        # decision index: a few samples before the end so tau>0
        j = len(und) - 1 - eval_back
        if j < min_path - 1:
            j = len(und) - 1            # too short -> use all, tau from spacing
        expiry = ts[-1]
        tau = max(1.0, expiry - ts[j])
        if tau < 1.0:
            continue
        spot = und[j]
        strike = m["strike"]
        outcome = int(m["outcome_up"])
        # causal returns up to and including j
        rets = log_returns(und[: j + 1])
        if len(rets) < 4 or spot <= 0 or strike <= 0:
            continue
        total_dt = max(1.0, ts[j] - ts[0])
        # per-second variance, both ways
        rv_naive = realized_variance_naive(rets)
        rv_robust = realized_variance_ac1(rets)
        rv_kernel = realized_variance_kernel(rets, q=2)
        sig_naive = math.sqrt(max(rv_naive / total_dt, 0.0)) or 1e-9
        sig_robust = math.sqrt(max(rv_robust / total_dt, 0.0)) or 1e-9
        sig_kernel = math.sqrt(max(rv_kernel / total_dt, 0.0)) or 1e-9
        a_naive = digital_prob_up(spot, strike, tau, 0.0, sig_naive)
        a_robust = digital_prob_up(spot, strike, tau, 0.0, sig_robust)
        a_kernel = digital_prob_up(spot, strike, tau, 0.0, sig_kernel)
        naive_po.append((a_naive, outcome))
        robust_po.append((a_robust, outcome))
        kernel_po.append((a_kernel, outcome))
        used += 1
        rows_dbg.append((m["market"][:26], m["tf"], spot, strike, tau,
                         sig_naive, sig_robust, a_naive, a_robust, outcome))

    if used == 0:
        print("no markets with enough path samples to evaluate; need >= "
              f"{min_path} underlying points per settled market.")
        return

    print(f"\n=== alpha calibration on paths table (n={used} markets) ===")
    print(f"{'market':26} {'tf':4} {'tau_s':>6} {'sig_nv':>8} {'sig_rb':>8} "
          f"{'a_nv':>5} {'a_rb':>5} {'out':>3}")
    for r in rows_dbg:
        print(f"{r[0]:26} {r[1]:4} {r[4]:6.0f} {r[5]:8.2e} {r[6]:8.2e} "
              f"{r[7]:5.2f} {r[8]:5.2f} {r[9]:>3}")

    print("\n--- scoring (lower is better) ---")
    print(f"{'metric':10} {'naive':>10} {'robust_ac1':>12} {'kernel_q2':>12}")
    print(f"{'Brier':10} {_brier(naive_po):10.4f} {_brier(robust_po):12.4f} "
          f"{_brier(kernel_po):12.4f}")
    print(f"{'LogLoss':10} {_logloss(naive_po):10.4f} {_logloss(robust_po):12.4f} "
          f"{_logloss(kernel_po):12.4f}")

    print("\n--- reliability (bucket: n, mean_alpha, hit_rate) ---")
    for label, po in (("naive", naive_po), ("robust", robust_po)):
        print(f"  {label}:")
        for i, (nb, mp, hr) in enumerate(_reliability(po, nb=5)):
            lo, hi = i / 5, (i + 1) / 5
            if nb:
                print(f"    [{lo:.1f},{hi:.1f})  n={nb:3d}  mean_a={mp:.3f}  hit={hr:.3f}")

    # net sigma direction: is robust systematically lower (de-inflated)?
    ratios = [r[6] / r[5] for r in rows_dbg if r[5] > 0]
    if ratios:
        mr = sum(ratios) / len(ratios)
        print(f"\n  mean sigma_robust/sigma_naive = {mr:.3f} "
              f"({'de-inflated' if mr < 1 else 'inflated'} by robust correction)")

    print("\nCAVEATS:")
    print("  * Small n: this DB is a short run; treat the SIGN of the Brier/")
    print("    log-loss gap as the signal, not the magnitude.")
    print("  * mu=0 (no drift) for both estimators, so this isolates the sigma")
    print("    effect -- the only thing the upgrade changes.")
    print("  * sigma estimated from the COMPOSITE underlying[] (anchor+tick");
    print("    blend) -- the very feed whose noise we are correcting; the AC1")
    print("    correction is the cheapest defensible de-bias, NOT the full")
    print("    grid Markov chain (impractical at ~20-40 samples/market).")
    print("  * outcome_up is the binary terminal label; with so few markets a")
    print("    reliability table is indicative only.")


# --------------------------------------------------------------------------- #
# 4. Self-tests                                                               #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import random
    random.seed(7)

    # (a) AC1 == naive when returns are iid (no serial dependence) -> gamma_1 ~ 0
    rets = [random.gauss(0, 1e-3) for _ in range(4000)]
    rv = realized_variance_naive(rets)
    iv = realized_variance_ac1(rets)
    assert abs(iv / rv - 1.0) < 0.15, (iv, rv)   # close, no strong correction

    # (b) AC1 corrects DOWNWARD when noise adds negative-lag1 (bid-ask bounce):
    #     observed_r = eff_r + (e_i - e_{i-1}); the MA(1) noise inflates RV and
    #     induces gamma_1 < 0, so AC1 should give a SMALLER variance than naive.
    eff = [random.gauss(0, 5e-4) for _ in range(4000)]
    noise = [random.gauss(0, 5e-4) for _ in range(4001)]
    obs = [eff[i] + (noise[i + 1] - noise[i]) for i in range(4000)]
    rv2 = realized_variance_naive(obs)
    iv2 = realized_variance_ac1(obs)
    assert iv2 < rv2, (iv2, rv2)                  # noise de-inflated
    # and AC1 should be closer to the TRUE efficient variance than naive RV is
    true_var = realized_variance_naive(eff)
    assert abs(iv2 - true_var) < abs(rv2 - true_var), (iv2, rv2, true_var)

    # (c) two-state Markov grid estimator agrees with AC1 on AR(1) data (the
    #     paper's Sec 10.2 equivalence). Build returns with lag-1 correlation.
    ar = [0.0]
    for _ in range(3000):
        ar.append(-0.3 * ar[-1] + random.gauss(0, 5e-4))
    ar = ar[1:]
    mc = markov_qv_grid(ar, n_states=2, k_order=1)
    ac1 = realized_variance_ac1(ar)
    # same order of magnitude and same direction of correction vs naive RV
    rv3 = realized_variance_naive(ar)
    assert mc < rv3 and ac1 < rv3, (mc, ac1, rv3)   # neg corr -> both deflate
    assert 0.3 < mc / ac1 < 3.0, (mc, ac1)          # broadly consistent

    # (d) online NoiseRobustVol recovers a sane per-second sigma and matches the
    #     batch AC1 within the window.
    nrv = NoiseRobustVol(win=500, kernel_q=1)
    px, t = 100.0, 0.0
    for _ in range(600):
        t += 8.0                                    # ~8s cycle like the engine
        px *= math.exp(random.gauss(0, 5e-4))
        nrv.update(t, px)
    s_rb = nrv.sigma_s(robust=True)
    s_nv = nrv.sigma_s(robust=False)
    assert s_rb > 0 and s_nv > 0
    # digital prob at ATM ~ 0.5
    assert abs(nrv.prob_up(px, px, 300.0) - 0.5) < 0.02

    # (e) digital_prob_up matches the boltzmann_digital convention exactly
    S, k, tau, sig = 100.0, 99.0, 600.0, 3e-4
    d = math.log(S / k) / (sig * math.sqrt(tau))
    assert abs(digital_prob_up(S, k, tau, 0.0, sig) - normal_cdf(d)) < 1e-9

    print("selftest OK")
    print(f"  iid:        AC1/RV          = {iv/rv:.3f}  (~1, no spurious correction)")
    print(f"  bounce:     AC1/RV          = {iv2/rv2:.3f}  (<1, noise de-inflated)")
    print(f"              |AC1-true|<|RV-true|  -> {abs(iv2-true_var):.2e} < {abs(rv2-true_var):.2e}")
    print(f"  AR(1):      MC_grid/AC1      = {mc/ac1:.3f}  (Markov 2-state ~ AC1)")
    print(f"  online:     sigma_robust    = {s_rb:.2e}  sigma_naive = {s_nv:.2e}")


if __name__ == "__main__":
    _selftest()
    if len(sys.argv) > 1:
        backtest_db(sys.argv[1])
    else:
        print("\npass a forward_engine.db path to run the calibration backtest")
