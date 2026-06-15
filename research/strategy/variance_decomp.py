"""Variance/jump decomposition — Lee-Mykland (2008) jumps, bipower, signed jump var.

Module A of the full-stack research engine (see research/fullstack_research_engine.md).
Implements the realized-variance decomposition behind Lee & Wang (2024) [1] and the
jump machinery of Catania & Grassi (2019) [2], faithfully but at the ASSET/DAILY level
our feed supports (~thousands of obs), not per-15m-market (~20-40 obs, too short).

    RV  =  JumpRobustVar (continuous, bipower)  +  PosJumpVar  +  NegJumpVar

[1]'s key result: the negative return-predictability loads on PosJumpVar + JumpRobustVar,
NOT NegJumpVar (lottery/retail interpretation). So we keep the signed parts separate.

stdlib only (no numpy). Run with no args for the synthetic selftest (known truth);
pass a forward_engine.db path for the live-history backtest.

    python3 research/strategy/variance_decomp.py
    python3 research/strategy/variance_decomp.py data/forward_live/forward_engine.db
"""

from __future__ import annotations

import math
import sys

MU1 = math.sqrt(2.0 / math.pi)   # E|Z| for Z~N(0,1); bipower scaling constant


# ----------------------------------------------------------------------------- core
def log_returns(prices: list[float]) -> list[float]:
    out = []
    for a, b in zip(prices, prices[1:]):
        if a and b and a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def realized_variance(rets: list[float]) -> float:
    """RV = Σ rᵢ² (total quadratic variation, incl. jumps)."""
    return math.fsum(r * r for r in rets)


def bipower_variation(rets: list[float]) -> float:
    """BV = (π/2)·Σ|rᵢ||rᵢ₋₁| — Barndorff-Nielsen–Shephard jump-ROBUST variance.
    Consistent for the continuous (diffusive) part of QV; |r||r_prev| down-weights a
    lone jump because a jump rarely lands on two adjacent returns."""
    if len(rets) < 2:
        return 0.0
    s = math.fsum(abs(a) * abs(b) for a, b in zip(rets[1:], rets[:-1]))
    return (1.0 / (MU1 * MU1)) * s     # 1/μ1² = π/2


def _gumbel_constants(n: int) -> tuple[float, float]:
    """Lee-Mykland Gumbel centering/scaling (C_n, S_n) for n tested returns."""
    ln = math.log(n)
    c = MU1
    sqrt2ln = math.sqrt(2.0 * ln)
    C = sqrt2ln / c - (math.log(math.pi) + math.log(ln)) / (2.0 * c * sqrt2ln)
    S = 1.0 / (c * sqrt2ln)
    return C, S


def lee_mykland_jumps(rets: list[float], K: int | None = None,
                      alpha: float = 0.05) -> list[bool]:
    """Lee-Mykland (2008) nonparametric jump test. For each return rᵢ, standardize by a
    LOCAL bipower spot-vol over the preceding K returns; flag |Lᵢ| in the Gumbel tail.

    Returns a list[bool] aligned to `rets` (True = jump arrival at i). The first K
    returns are un-testable (no local window) → False.

    WINDOW SENSITIVITY (important): the local σ̂ is estimated from only K returns, so a
    too-small K makes σ̂ noisy → it under-estimates vol in quiet patches → spurious
    flags. Empirically (pure-diffusion Monte Carlo, n=2000) the spurious-flag rate runs
    ~1.6/run at K=16 vs the ~0.04 asymptotic target, and only settles toward target by
    K≳120. Lee-Mykland themselves recommend K≈√(252·obs_per_day) (typically a few
    hundred intraday). We therefore default K to a fraction of the series length
    (≈√n, floored at 50) rather than a fixed 16. Pass K explicitly to tune the
    bias/variance of σ̂ vs the loss of testable head returns.
    """
    n = len(rets)
    if K is None:
        # √n-style window, floored at 50 so the local bipower σ̂ is stable (see above).
        K = max(50, int(math.sqrt(n)))
        K = min(K, max(2, n // 4))                # never eat >¼ of the series as warm-up
    if n <= K + 2:
        return [False] * n
    # significance threshold on the Gumbel variable: reject if (|L|-C)/S > beta_star.
    # NOTE: C_n, S_n already absorb the sample size, so this beta_star is the MAX-stat
    # (family-wise) level. Applying it per-return — standard LM usage — controls the
    # spurious-detection probability per the asymptotics; finite-K σ̂ noise is the main
    # residual inflation (mitigated by the larger default K above).
    beta_star = -math.log(-math.log(1.0 - alpha))
    n_test = n - K
    C, S = _gumbel_constants(max(n_test, 2))
    flags = [False] * n
    for i in range(K, n):
        window = rets[i - K:i]                     # K returns strictly BEFORE i
        bp = math.fsum(abs(window[j]) * abs(window[j - 1]) for j in range(1, len(window)))
        # Lee-Mykland (2008) Eq.(7): the LOCAL vol estimate is sigmahat = sqrt of the
        # raw bipower mean WITHOUT the (π/2)=1/μ1² inflation, so E[sigmahat] = μ1·σ.
        # The test stat L_i = r_i/sigmahat then converges to N(0,1)/μ1, which is exactly
        # the variable the c=μ1 Gumbel constants (C_n, S_n) below are built for. Dividing
        # the bipower by μ1² here (recovering the TRUE σ) would make L_i standard normal
        # and leave the c=μ1 constants ~1/μ1≈1.25× too large — an over-conservative test
        # that under-detects jumps (effective cutoff ~5.9σ vs the paper's ~4.7σ at α=.01).
        sigma_hat = math.sqrt(bp / (K - 2)) if bp > 0 else 0.0   # ≈ μ1·σ (paper's σ̂)
        if sigma_hat <= 0:
            continue
        L = rets[i] / sigma_hat
        if (abs(L) - C) / S > beta_star:
            flags[i] = True
    return flags


def decompose(rets: list[float], K: int | None = None, alpha: float = 0.05) -> dict:
    """Full decomposition. Returns total RV, bipower jump-robust var, and signed jump
    variances (from Lee-Mykland-flagged returns), plus counts/intensity."""
    rv = realized_variance(rets)
    bv = bipower_variation(rets)
    flags = lee_mykland_jumps(rets, K=K, alpha=alpha)
    pos_jv = math.fsum(r * r for r, f in zip(rets, flags) if f and r > 0)
    neg_jv = math.fsum(r * r for r, f in zip(rets, flags) if f and r < 0)
    n_jumps = sum(flags)
    # continuous part: prefer bipower (BNS), but never exceed RV-jumpvar (numerical guard)
    jump_var_flagged = pos_jv + neg_jv
    jump_robust = min(bv, max(rv - jump_var_flagged, 0.0)) if bv > 0 else max(rv - jump_var_flagged, 0.0)
    return {
        "n_rets": len(rets),
        "rv": rv,
        "bipower_var": bv,
        "jump_robust_var": jump_robust,
        "pos_jump_var": pos_jv,
        "neg_jump_var": neg_jv,
        "jump_var_total": jump_var_flagged,
        "bns_jump_var": max(rv - bv, 0.0),     # alt: total jump var via RV-BV (BNS)
        "n_jumps": n_jumps,
        "jump_intensity": n_jumps / max(len(rets), 1),
        # the [1] "lottery" signal: positive-jump share of total variance
        "pos_jump_share": (pos_jv / rv) if rv > 0 else 0.0,
        "signed_jump_var": pos_jv - neg_jv,    # net signed jump variation
    }


# --------------------------------------------------------------------------- selftest
def _seeded_normals(n: int, seed: int = 7) -> list[float]:
    """Box-Muller normals from a deterministic LCG (stdlib-only, reproducible)."""
    state = seed & 0xFFFFFFFF
    def u():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return (state + 1) / (0x7FFFFFFF + 2)
    out = []
    while len(out) < n:
        u1, u2 = u(), u()
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2))
        out.append(r * math.sin(2 * math.pi * u2))
    return out[:n]


def selftest() -> None:
    print("=== variance_decomp selftest (synthetic, known truth) ===")
    n = 2000
    sigma = 0.01                          # per-step diffusive vol
    z = _seeded_normals(n)
    diffusion = [sigma * zi for zi in z]
    # inject 6 known jumps (4 positive, 2 negative), each ~8σ — unmistakable
    jump_idx = {300: +0.08, 700: +0.09, 1100: +0.085, 1500: +0.075, 900: -0.08, 1700: -0.085}
    rets = list(diffusion)
    for i, j in jump_idx.items():
        rets[i] += j
    d = decompose(rets, alpha=0.01)
    flags = lee_mykland_jumps(rets, alpha=0.01)
    detected = {i for i, f in enumerate(flags) if f}
    truth = set(jump_idx)
    tp = len(detected & truth); fp = len(detected - truth); fn = len(truth - detected)
    diffusion_var_true = sigma * sigma * n
    print(f"  injected jumps: {sorted(truth)}  (4 pos, 2 neg)")
    print(f"  detected:       {sorted(detected)}")
    print(f"  jump detection: TP={tp} FP={fp} FN={fn}  (want TP=6, FP=0)")
    print(f"  true diffusion var ≈ {diffusion_var_true:.4f}")
    print(f"  RV (incl jumps)    = {d['rv']:.4f}   (inflated by jumps)")
    print(f"  bipower (robust)   = {d['bipower_var']:.4f}   (should ≈ diffusion var)")
    print(f"  pos_jump_var       = {d['pos_jump_var']:.4f}   neg_jump_var = {d['neg_jump_var']:.4f}")
    print(f"  pos share of RV    = {d['pos_jump_share']:.3f}   signed JV = {d['signed_jump_var']:+.4f}")
    ok_detect = (tp == 6 and fp == 0)
    ok_robust = abs(d["bipower_var"] - diffusion_var_true) < 0.25 * diffusion_var_true
    ok_signed = d["pos_jump_var"] > d["neg_jump_var"]   # 4 pos vs 2 neg, similar mag
    print(f"  [{'PASS' if ok_detect else 'FAIL'}] jump detection exact")
    print(f"  [{'PASS' if ok_robust else 'FAIL'}] bipower recovers diffusion var (jump-robust)")
    print(f"  [{'PASS' if ok_signed else 'FAIL'}] pos_jump_var > neg_jump_var (sign preserved)")
    print(f"  RESULT: {'ALL PASS' if (ok_detect and ok_robust and ok_signed) else 'CHECK ABOVE'}")


# --------------------------------------------------------------------------- backtest
def backtest_db(db_path: str) -> None:
    """Per-asset variance/jump decomposition from the live engine's stored spot history
    (decisions.spot). Honest about coverage; this is a foundation sanity check, not a
    return-prediction test (that's module D, cross_sectional.py)."""
    import sqlite3
    print(f"=== variance_decomp backtest on {db_path} ===")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    assets = [a for (a,) in conn.execute(
        "SELECT DISTINCT asset FROM decisions WHERE spot IS NOT NULL ORDER BY asset")]
    print(f"  assets: {assets}")
    print(f"  {'asset':6s} {'n_obs':>6s} {'RV':>10s} {'bipower':>10s} {'posJV':>9s} "
          f"{'negJV':>9s} {'jumps':>6s} {'pos_share':>9s}")
    for a in assets:
        rows = [s for (s,) in conn.execute(
            "SELECT spot FROM decisions WHERE asset=? AND spot IS NOT NULL ORDER BY ts", (a,))]
        # dedupe consecutive identical spots (flat composite ticks → spurious zero-returns)
        prices = [rows[0]] + [p for p, q in zip(rows[1:], rows[:-1]) if p != q] if rows else []
        rets = log_returns(prices)
        if len(rets) < 30:
            print(f"  {a:6s} {len(rets):>6d}  (too few returns)")
            continue
        d = decompose(rets)
        print(f"  {a:6s} {d['n_rets']:>6d} {d['rv']:>10.5f} {d['bipower_var']:>10.5f} "
              f"{d['pos_jump_var']:>9.5f} {d['neg_jump_var']:>9.5f} {d['n_jumps']:>6d} "
              f"{d['pos_jump_share']:>9.3f}")
    conn.close()
    print("  NOTE: composite-tick history, not clean OHLC — treat as a wiring sanity check.\n"
          "        Real signal test = cross_sectional.py (predicts outcome beyond the mid).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        backtest_db(sys.argv[1])
    else:
        selftest()
