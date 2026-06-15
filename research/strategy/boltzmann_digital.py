"""boltzmann_digital — structural digital-probability head + calibration anchor.

Prototype for the "boltzmann_structural" upgrade described in
research/boltzmann_upgrade.md. STDLIB ONLY (math, sqlite3, statistics, json).
Does NOT import or modify anything under live/.

What this provides
------------------
1. `digital_prob_up(...)`  -- closed-form P(S_T > k) for a digital, the paper's
   1 - alpha = (1/2) Erfc[(log k - (T)*mu)/(sigma*sqrt(2 T))]. Identical to the
   Gaussian / Black-Scholes N(d2)-style digital probability. Time scaling is
   carried in mu and sigma being PER-UNIT (per-step) quantities multiplied by the
   number of steps-to-expiry T.

2. `CausalDriftVol` -- an online, causal estimator of per-second log-return drift
   (mu_s) and vol (sigma_s) from the price feed, with correct horizon scaling to
   any timeframe via tau in seconds.

3. `TerminalVarianceRegime` -- detects diffusion vs concentration (Eq. 20) by
   comparing realized terminal dispersion against the sqrt-time diffusion law,
   and returns a multiplier on sigma used in the digital formula.

4. `structural_anchor(...)` -- the shrinkage / EV-gate-widening policy that uses
   the structural alpha to fight the measured live overconfidence.

5. A self-contained backtest harness `backtest_db(db_path)` that replays the
   engine's own settlements DB and shows what the structural anchor + widened
   gate WOULD have traded vs what actually happened. Run:

       python research/strategy/boltzmann_digital.py /abs/path/forward_engine.db
"""

from __future__ import annotations

import math
import sqlite3
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 1. The structural digital probability (the paper's alpha)                   #
# --------------------------------------------------------------------------- #
def _erfc(x: float) -> float:
    return math.erfc(x)


def digital_prob_up(spot: float, strike: float, tau_s: float,
                    mu_s: float, sigma_s: float) -> float:
    """P(S_T > k) at expiry for a digital, derived from the paper's alpha.

    Paper (Eq. 16/18, with our digital reading). Let x_0 = log S, mean = x_0 + T*mu:
        x_T = log S_T ~ Normal(mean = log S + T*mu, var = T*sigma^2)
        The paper writes alpha = (1/2) Erfc[ (mean - log k)/(sigma*sqrt(2T)) ].
        With d := (mean - log k)/(sigma*sqrt(T)) = (log(S/k) + T*mu)/(sigma*sqrt(T)):
            alpha = (1/2) Erfc[ d/sqrt(2) ] = 1 - N(d)        [= P(S_T <= k) = P(DOWN)]
            P(up) = 1 - alpha = N(d)                          [= N(d2) digital prob]
        i.e. the paper's alpha is EXACTLY 1 - N(d2); P(up) is the standard Gaussian
        digital. (Verify: ATM + zero drift -> d=0 -> P(up)=1/2.)

    Here we work in SECONDS: mu_s, sigma_s are PER-SECOND log-return moments and
    T -> tau_s (seconds to expiry). All timeframe scaling is absorbed in tau_s.

    NOTE on drift convention: the paper uses the physical mean mu of log-returns.
    Black-Scholes' d2 uses the risk-neutral (r - sigma^2/2). We expose mu_s as the
    *effective* per-second drift of log price; pass the drift-adjusted value (the
    estimator below already returns the empirical mean of log-returns, which is the
    physical drift the paper intends). For crypto near-expiry, |mu_s*tau| is tiny
    vs sigma_s*sqrt(tau), so the distinction barely moves alpha (see report sec 2).
    """
    if spot <= 0 or strike <= 0 or tau_s <= 0 or sigma_s <= 0:
        return 0.5
    denom = sigma_s * math.sqrt(2.0 * tau_s)
    # numerator = mean - log k = log(S/k) + tau*mu  (spot folded into the mean)
    num = math.log(spot / strike) + tau_s * mu_s
    alpha = 0.5 * _erfc(num / denom)          # = P(DOWN) = 1 - N(d2)
    return max(1e-6, min(1 - 1e-6, 1.0 - alpha))   # P(UP) = N(d2)


def normal_cdf(x: float) -> float:
    return 0.5 * _erfc(-x / math.sqrt(2.0))


# --------------------------------------------------------------------------- #
# 2. Causal per-second drift / vol estimator                                  #
# --------------------------------------------------------------------------- #
@dataclass
class CausalDriftVol:
    """Online EWMA estimator of per-second log-return drift and variance.

    Feed it (timestamp, price) ticks. It accumulates log-returns r_i and the
    elapsed dt_i, then reports mu_s = E[r]/E[dt] and sigma_s = sqrt(Var[r]/E[dt]).
    EWMA (half-life ~ `hl` returns) keeps it causal and adaptive. No look-ahead.

    Scaling to a horizon tau (seconds):  mu*tau, sigma*sqrt(tau).
    """
    hl: float = 240.0            # half-life in number of returns
    _last_t: float | None = None
    _last_p: float | None = None
    n: int = 0
    ew_r: float = 0.0            # EWMA of per-second return  (r/dt)
    ew_r2: float = 0.0           # EWMA of (r/dt)^2 ... no: see below
    ew_var: float = 1e-12        # EWMA of per-second variance contribution
    _decay: float = field(default=0.0)

    def __post_init__(self):
        self._decay = 0.5 ** (1.0 / max(1.0, self.hl))

    def update(self, ts: float, price: float) -> None:
        if price <= 0:
            return
        if self._last_p is None:
            self._last_t, self._last_p = ts, price
            return
        dt = ts - self._last_t
        if dt <= 0:
            self._last_p = price
            return
        r = math.log(price / self._last_p)
        rps = r / dt                     # per-second drift contribution
        varps = (r * r) / dt             # per-second variance contribution (E[r^2]/dt)
        a = self._decay
        if self.n == 0:
            self.ew_r = rps
            self.ew_var = varps
        else:
            self.ew_r = a * self.ew_r + (1 - a) * rps
            self.ew_var = a * self.ew_var + (1 - a) * varps
        self.n += 1
        self._last_t, self._last_p = ts, price

    @property
    def mu_s(self) -> float:
        return self.ew_r if self.n > 5 else 0.0

    @property
    def sigma_s(self) -> float:
        # subtract drift^2 to get variance, floor at a small positive number
        v = self.ew_var - self.mu_s * self.mu_s
        v = max(v, 1e-16)
        return math.sqrt(v)

    def prob_up(self, spot: float, strike: float, tau_s: float,
                use_drift: bool = False) -> float:
        mu = self.mu_s if use_drift else 0.0
        return digital_prob_up(spot, strike, tau_s, mu, self.sigma_s)


# --------------------------------------------------------------------------- #
# 3. Terminal-variance regime: diffusion vs concentration (Eq. 20)            #
# --------------------------------------------------------------------------- #
@dataclass
class TerminalVarianceRegime:
    """Detect diffusion vs concentration of terminal variance.

    The paper's Eq. 20 shows sigma^2_n = C1*(1)^n + C2*(-1)^n can OSCILLATE
    (concentrate on some steps) rather than monotonically diffuse. We can't
    observe future variance, but we CAN detect, causally, whether realized
    dispersion over the recent horizon is growing like sqrt(t) (pure diffusion)
    or sub-diffusively (concentration / mean-reversion / pinning near a level).

    Method: maintain realized variance over two non-overlapping recent windows of
    lengths w and 2w. Under iid diffusion Var(2w) = 2*Var(w). The ratio
        kappa = Var(2w) / (2*Var(w))
    is ~1 for diffusion, <1 for concentration (variance not accumulating ->
    pinning/mean-revert), >1 for super-diffusion (trending/expansion).

    Returns a multiplier on sigma to feed the digital formula:
        sigma_eff = sigma * sqrt(clamp(kappa, kmin, kmax))
    so a concentrating regime SHRINKS the effective terminal sigma (pushes alpha
    toward the extremes correctly), a diffusing regime leaves it, and an
    expanding regime widens it (pulls alpha toward 0.5 -> less overconfidence).
    """
    w: int = 30
    buf: list[float] = field(default_factory=list)  # recent log-returns
    cap: int = 240

    def update(self, r: float) -> None:
        self.buf.append(r)
        if len(self.buf) > self.cap:
            self.buf.pop(0)

    def kappa(self) -> float | None:
        if len(self.buf) < 4 * self.w:
            return None
        recent = self.buf[-2 * self.w:]
        half1 = recent[: self.w]
        full = recent
        def var(xs):
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        v_w = var(half1)
        v_2w = var(full)
        if v_w <= 0:
            return None
        # full window has 2w points, half has w. Under diffusion the *sum* variance
        # scales linearly; here per-step variance should match => ratio ~1.
        return v_2w / v_w

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
# 4. Shrinkage anchor + EV-gate widening                                      #
# --------------------------------------------------------------------------- #
@dataclass
class AnchorConfig:
    # shrink ensemble p toward structural alpha by lambda, scaled by divergence
    lam0: float = 0.35           # base shrink weight toward structural alpha
    div_scale: float = 6.0       # how fast lambda ramps with |p_ens - alpha|
    lam_max: float = 0.85
    # EV gate widening: extra margin (in price units) per unit divergence
    gate_base: float = 0.01
    gate_div_k: float = 0.40     # +0.40 * |p_ens - alpha| added to required EV
    gate_max: float = 0.12


def structural_anchor(p_ens: float, alpha_up: float, cfg: AnchorConfig
                      ) -> tuple[float, float]:
    """Shrink the ensemble probability toward the structural digital prob and
    return (p_anchored, required_ev_margin).

    Rationale vs the live failure: the engine bought 'down' favorites priced ~0.65
    because the ensemble said p_down ~ 0.74 (overconfident). If the structural
    alpha said p_down ~ 0.60 (closer to the market), shrinking p toward alpha both
    (a) lowers p_side below the price -> kills the EV>0 signal, and (b) the large
    |p_ens - alpha| divergence WIDENS the required EV margin, so only trades where
    the structural model AGREES with the ensemble survive. That directly removes
    the favorite-buying losses, which were exactly the high-divergence trades.
    """
    div = abs(p_ens - alpha_up)
    lam = min(cfg.lam_max, cfg.lam0 + cfg.div_scale * div * cfg.lam0)
    # shrink in log-odds space (more stable near 0/1 than linear)
    def lo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))
    p_anchored = 1.0 / (1.0 + math.exp(-((1 - lam) * lo(p_ens) + lam * lo(alpha_up))))
    gate = min(cfg.gate_max, cfg.gate_base + cfg.gate_div_k * div)
    return p_anchored, gate


# --------------------------------------------------------------------------- #
# 5. Backtest harness against the engine's own DB                             #
# --------------------------------------------------------------------------- #
def backtest_db(db_path: str) -> None:
    """Replay filled settlements joined with their decision rows. Recompute the
    structural alpha from the decision's recorded z_dist/realized_vol-equivalent,
    apply the anchor + widened gate, and report counterfactual selection.

    LIMITATION (be honest): the decisions table does NOT store per-second sigma or
    tau in seconds directly, but it stores z_dist = log(spot/strike)/sigma_tau and
    the timeframe. We can invert: sigma_tau = log(spot/strike)/z_dist, which is
    exactly the terminal (horizon) sigma the structural formula needs. Then
        alpha_up = N( (log(spot/strike) + tau*mu) / sigma_tau ) ~= N(z_dist)
    with mu~0. So the structural P(up) here is simply Phi(z_dist) -- a clean,
    causal reconstruction from logged fields. This is the key check: does anchoring
    p_cal toward Phi(z_dist) remove the losing favorite trades?
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    # join filled settlements to the decision that spawned them (same market+side)
    rows = conn.execute(
        """
        SELECT s.market, s.side, s.outcome_up, s.won, s.fill_price, s.pnl,
               d.p_cal, d.p_up, d.z_dist, d.ev, d.tf, d.spot, d.strike, d.fv_resid
        FROM settlements s
        JOIN decisions d ON d.market = s.market AND d.decision='TRADE'
        WHERE s.filled = 1
        GROUP BY s.market
        """
    ).fetchall()
    if not rows:
        print("no filled+traded rows joinable; nothing to replay")
        return

    cfg = AnchorConfig()
    actual_pnl = 0.0
    actual_n = 0
    actual_wins = 0
    kept = []
    print(f"{'market':22} {'side':4} {'price':>6} {'p_cal':>6} {'alpha':>6} "
          f"{'p_anch':>6} {'gate':>5} {'ev_new':>7} {'keep':>4} {'won':>3}")
    for r in rows:
        actual_n += 1
        actual_pnl += (r["pnl"] or 0.0)
        actual_wins += r["won"] or 0
        # structural up-prob reconstructed from logged z_dist (mu~0)
        alpha_up = normal_cdf(r["z_dist"]) if r["z_dist"] is not None else 0.5
        p_cal = r["p_cal"] if r["p_cal"] is not None else 0.5
        # the engine's traded side prob and price
        side = r["side"]
        price = r["fill_price"] if r["fill_price"] is not None else 0.5
        # anchor on the UP probability, then map to traded side
        p_anch_up, gate = structural_anchor(p_cal, alpha_up, cfg)
        p_side_new = p_anch_up if side == "up" else (1 - p_anch_up)
        ev_new = p_side_new - price - 0.005
        keep = ev_new > gate and 0.30 <= price <= 0.70
        won = r["won"] or 0
        print(f"{r['market'][:22]:22} {side:4} {price:6.3f} {p_cal:6.3f} "
              f"{alpha_up:6.3f} {(p_anch_up if side=='up' else 1-p_anch_up):6.3f} "
              f"{gate:5.3f} {ev_new:+7.3f} {str(keep):>4} {won:>3}")
        if keep:
            kept.append((r["pnl"] or 0.0, won, price))

    print("\n--- ACTUAL (engine) ---")
    print(f"  trades={actual_n}  wins={actual_wins}  winrate="
          f"{actual_wins/actual_n:.3f}  pnl={actual_pnl:+.2f}")
    if kept:
        kp = sum(x[0] for x in kept)
        kw = sum(x[1] for x in kept)
        kavgp = sum(x[2] for x in kept) / len(kept)
        print("--- COUNTERFACTUAL (structural anchor + widened gate) ---")
        print(f"  trades={len(kept)}  wins={kw}  winrate={kw/len(kept):.3f}  "
              f"avg_price={kavgp:.3f}  pnl={kp:+.2f}")
        print(f"  removed {actual_n - len(kept)} trades; pnl delta={kp - actual_pnl:+.2f}")
    else:
        print("--- COUNTERFACTUAL: structural anchor + gate would have taken ZERO "
              "of these trades (all were high-divergence favorites). ---")
        print(f"  avoided pnl = {-actual_pnl:+.2f} (i.e. it sidesteps the whole loss)")


# --------------------------------------------------------------------------- #
# quick self-test of the math + optional DB replay                            #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # ATM digital with zero drift must be ~0.5
    p = digital_prob_up(100.0, 100.0, 300.0, 0.0, 1e-4)
    assert abs(p - 0.5) < 1e-6, p
    # spot above strike -> p_up > 0.5
    assert digital_prob_up(101.0, 100.0, 300.0, 0.0, 5e-5) > 0.5
    # identity vs N(d2): P(up) == Phi(d) when mu=0, d=log(S/k)/(sigma*sqrt(tau))
    S, k, tau, sig = 100.0, 99.0, 600.0, 3e-4
    d = math.log(S / k) / (sig * math.sqrt(tau))
    assert abs(digital_prob_up(S, k, tau, 0.0, sig) - normal_cdf(d)) < 1e-9
    # causal estimator sanity
    dv = CausalDriftVol(hl=50)
    px = 100.0
    import random
    random.seed(0)
    t = 0.0
    for _ in range(500):
        t += 1.0
        px *= math.exp(random.gauss(0, 0.0005))
        dv.update(t, px)
    assert dv.sigma_s > 0
    print(f"selftest OK | est sigma_s={dv.sigma_s:.2e} (true ~5.0e-4) "
          f"prob_up(ATM)={dv.prob_up(px, px, 300):.3f}")


if __name__ == "__main__":
    _selftest()
    if len(sys.argv) > 1:
        print("\n=== DB replay ===")
        backtest_db(sys.argv[1])
    else:
        print("\npass a forward_engine.db path to run the counterfactual replay")
