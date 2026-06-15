"""HAR-RV — Corsi (2009) Heterogeneous AutoRegressive realized-variance forecaster.

Module B of the full-stack research engine (see research/fullstack_research_engine.md).
Implements the cascade model of Corsi (2009), "A Simple Approximate Long-Memory Model of
Realized Volatility" (J Financial Econometrics 7(2), DOI: 10.1093/jjfinec/nbp001), plus
the negative-return LEVERAGE asymmetry of Catania & Grassi (2019)/Corsi-Renò (HAR-RV-L):

    RV_{t+1}  =  β0  +  β_d·RV_d  +  β_w·RV_w  +  β_m·RV_m  +  β_lev·LEV_t  +  ε

  RV_d = the current bar's realized variance
  RV_w = average RV over the last 5 bars     (Corsi's weekly cascade component)
  RV_m = average RV over the last 22 bars    (monthly component)
  LEV  = a leverage term that loads on negative cumulative returns
         ( min(ret_d, 0)² , the down-move "bad variance" — drives the asymmetric
           vol response that [2] documents; positive returns carry no leverage weight )

LEVELLING (read first — same constraint as Module A). Corsi's d/w/m are daily/weekly/
monthly on CLEAN daily RV. Our live feed has only ~6 calendar days of composite-tick spot,
so DAILY RV gives ~6 points — far too few to fit a 22-lag monthly component. We therefore
build RV at a sub-daily BAR level: the deduped spot log-return series is chopped into
fixed-size bars of `bar_len` returns, RV is computed per bar (Module A's realized_variance,
optionally the AC1 noise-robust variant), and the Corsi cascade ratios (1 / 5 / 22 bars) are
preserved verbatim. "Daily/weekly/monthly" thus means "1 / 5 / 22 bars"; with ~30 returns/
bar (~4-8 min of wall clock on an ~8s feed) this yields ~60-70 bars/asset — enough to fit
the 5 coefficients honestly, while the cascade STRUCTURE (the thing the paper is about) is
identical. We state n openly; this is in-sample term-structure estimation, not an OOS claim.

stdlib only (no numpy). Run with no args for the synthetic selftest (known truth); pass a
forward_engine.db path for the live-history backtest.

    python3 research/strategy/har_rv.py
    python3 research/strategy/har_rv.py data/forward_live/forward_engine.db
"""

from __future__ import annotations

import math
import sys

# Reuse Module A's primitives so RV is defined identically across the stack.
try:
    from variance_decomp import log_returns, realized_variance
except ImportError:                                    # when run from repo root
    from research.strategy.variance_decomp import log_returns, realized_variance


# Cascade horizons (in BARS). Corsi's canonical 1 / 5 / 22 — see module docstring for
# why a "bar" is the levelled unit here rather than a calendar day.
LAG_W = 5
LAG_M = 22


# ----------------------------------------------------------------------------- RV series
def realized_variance_ac1(rets: list[float]) -> float:
    """Zhou (1996) lag-1 autocovariance-corrected RV (noise-robust). Mirrors
    qv_markov.realized_variance_ac1 — duplicated here (a few lines) to keep har_rv
    importable stdlib-only without pulling the whole qv_markov module. Floors at ¼·RV
    so microstructure correction can never drive the variance non-positive."""
    n = len(rets)
    rv = realized_variance(rets)
    if n < 2:
        return rv
    m = math.fsum(rets) / n
    g1 = math.fsum((rets[i] - m) * (rets[i - 1] - m) for i in range(1, n)) / n
    return max(rv + 2.0 * n * g1, 0.25 * rv, 1e-18)


def bar_rv_series(prices: list[float], bar_len: int = 30,
                  robust: bool = False) -> tuple[list[float], list[float]]:
    """Chop a (deduped) price series into fixed-size bars and return per-bar
    (RV, signed cumulative bar-return). The signed bar return feeds the leverage term.

    Returns (rv_list, ret_list), aligned bar-by-bar. RV uses Module A's realized_variance
    (or the AC1 noise-robust variant if robust=True). Bars with <2 returns are skipped.
    """
    rets_all = log_returns(prices)
    rv_list: list[float] = []
    ret_list: list[float] = []
    for start in range(0, len(rets_all) - bar_len + 1, bar_len):
        chunk = rets_all[start:start + bar_len]
        if len(chunk) < 2:
            continue
        rv = realized_variance_ac1(chunk) if robust else realized_variance(chunk)
        rv_list.append(rv)
        ret_list.append(math.fsum(chunk))              # cumulative log-return over the bar
    return rv_list, ret_list


# ----------------------------------------------------------------------------- HAR design
def har_design(rv: list[float], ret: list[float] | None = None,
               lag_w: int = LAG_W, lag_m: int = LAG_M) -> tuple[list[list[float]], list[float]]:
    """Build the HAR-RV(+leverage) design matrix X and target y.

    For each t in [lag_m, len(rv)-1):
        target  y_t   = RV_{t+1}                            (next-bar realized variance)
        features      = [1, RV_d=RV_t, RV_w=mean(RV_{t-4..t}), RV_m=mean(RV_{t-21..t}),
                         LEV = min(ret_t, 0)²]              (leverage on down-moves only)
    The first lag_m bars are warm-up (no full monthly window). If `ret` is None the
    leverage column is omitted (pure Corsi HAR).

    Returns (X, y) with X a list of feature rows (including the intercept column).
    """
    X: list[list[float]] = []
    y: list[float] = []
    use_lev = ret is not None
    for t in range(lag_m, len(rv) - 1):
        rv_d = rv[t]
        rv_w = math.fsum(rv[t - lag_w + 1:t + 1]) / lag_w
        rv_m = math.fsum(rv[t - lag_m + 1:t + 1]) / lag_m
        row = [1.0, rv_d, rv_w, rv_m]
        if use_lev:
            down = ret[t] if ret[t] < 0 else 0.0
            row.append(down * down)                        # down-move "bad" variance
        X.append(row)
        y.append(rv[t + 1])
    return X, y


# ----------------------------------------------------------------------------- OLS (stdlib)
def ols_fit(X: list[list[float]], y: list[float], ridge: float = 0.0) -> list[float]:
    """Ordinary least squares via the normal equations (XᵀX) β = Xᵀy, solved with
    Gauss-Jordan elimination. Optional ridge (λ on the non-intercept diagonal) keeps the
    system well-conditioned when the d/w/m components are collinear (they often are —
    RV_w and RV_m overlap RV_d by construction). stdlib only."""
    if not X:
        return []
    p = len(X[0])
    # XtX and Xty
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for xi, yi in zip(X, y):
        for a in range(p):
            Xty[a] += xi[a] * yi
            xa = xi[a]
            row = XtX[a]
            for b in range(p):
                row[b] += xa * xi[b]
    if ridge > 0.0:
        for a in range(1, p):                              # don't penalise the intercept
            XtX[a][a] += ridge
    return _solve(XtX, Xty)


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b by Gauss-Jordan with partial pivoting (stdlib). Returns least-squares
    fallback (zeros for unsolvable columns) if A is singular."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-18:
            continue                                        # singular column → leave 0
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        M[col] = [v / pivval for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col]
                M[r] = [v - factor * mc for v, mc in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def predict(X: list[list[float]], beta: list[float]) -> list[float]:
    return [math.fsum(b * xij for b, xij in zip(beta, xi)) for xi in X]


def r_squared(y: list[float], yhat: list[float]) -> float:
    if not y:
        return float("nan")
    m = math.fsum(y) / len(y)
    ss_tot = math.fsum((yi - m) ** 2 for yi in y)
    ss_res = math.fsum((yi - hi) ** 2 for yi, hi in zip(y, yhat))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ----------------------------------------------------------------------------- online HAR
class OnlineHAR:
    """Rolling/online HAR-RV via recursive normal-equations (Sherman-Morrison-free batch
    refit on a sliding window). Kept simple and causal: feed (rv, ret) bar by bar; it
    maintains a window of the last `window` bars and refits on demand. Used for a genuine
    walk-forward forecast in the backtest (each forecast uses only past bars)."""

    def __init__(self, window: int = 400, lag_w: int = LAG_W, lag_m: int = LAG_M,
                 ridge: float = 1e-12, leverage: bool = True):
        self.window = window
        self.lag_w = lag_w
        self.lag_m = lag_m
        self.ridge = ridge
        self.leverage = leverage
        self._rv: list[float] = []
        self._ret: list[float] = []
        self.beta: list[float] | None = None

    def update(self, rv: float, ret: float) -> None:
        self._rv.append(rv)
        self._ret.append(ret)
        if len(self._rv) > self.window:
            self._rv.pop(0)
            self._ret.pop(0)

    def fit(self) -> list[float] | None:
        if len(self._rv) < self.lag_m + 5:
            return None
        ret = self._ret if self.leverage else None
        X, y = har_design(self._rv, ret, self.lag_w, self.lag_m)
        if len(X) < len(X[0]) + 2 if X else True:
            return None
        self.beta = ols_fit(X, y, ridge=self.ridge)
        return self.beta

    def forecast_next(self) -> float | None:
        """One-step-ahead RV forecast from the most recent fitted beta and the current
        tail of the RV/ret history (causal: uses only data already fed)."""
        if self.beta is None or len(self._rv) < self.lag_m:
            return None
        t = len(self._rv) - 1
        rv_d = self._rv[t]
        rv_w = math.fsum(self._rv[t - self.lag_w + 1:t + 1]) / self.lag_w
        rv_m = math.fsum(self._rv[t - self.lag_m + 1:t + 1]) / self.lag_m
        row = [1.0, rv_d, rv_w, rv_m]
        if self.leverage:
            down = self._ret[t] if self._ret[t] < 0 else 0.0
            row.append(down * down)
        return math.fsum(b * x for b, x in zip(self.beta, row))


# --------------------------------------------------------------------------- selftest
def _seeded_normals(n: int, seed: int = 13) -> list[float]:
    """Box-Muller normals from a deterministic LCG (stdlib, reproducible)."""
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
    print("=== har_rv selftest (synthetic, known HAR coefficients) ===")
    # Generate an RV series that obeys a KNOWN HAR recursion, then check OLS recovers it.
    n = 3000
    lag_w, lag_m = LAG_W, LAG_M
    b0, bd, bw, bm = 0.0002, 0.30, 0.35, 0.25            # true coefficients (sum<1, stationary)
    z = _seeded_normals(n, seed=13)
    rv = [0.001] * lag_m                                 # warm-up seed
    for t in range(lag_m, n):
        rv_d = rv[t - 1]
        rv_w = math.fsum(rv[t - lag_w:t]) / lag_w
        rv_m = math.fsum(rv[t - lag_m:t]) / lag_m
        mean = b0 + bd * rv_d + bw * rv_w + bm * rv_m
        # multiplicative log-normal innovation keeps RV strictly positive (vol is positive)
        rv.append(max(1e-9, mean * math.exp(0.15 * z[t] - 0.5 * 0.15 ** 2)))

    # Fit WITHOUT leverage (true DGP has none) and recover (b0,bd,bw,bm).
    X, y = har_design(rv, None, lag_w, lag_m)
    beta = ols_fit(X, y, ridge=0.0)
    yhat = predict(X, beta)
    r2 = r_squared(y, yhat)
    names = ["β0", "β_d", "β_w", "β_m"]
    truth = [b0, bd, bw, bm]
    print(f"  fitted on n={len(y)} bars (no-leverage DGP):")
    for nm, est, tr in zip(names, beta, truth):
        print(f"    {nm:4s} = {est:+.4f}   (true {tr:+.4f})")
    print(f"  in-sample R² = {r2:.4f}")
    # coefficient recovery: the slope sum and each slope close to truth
    ok_slopes = all(abs(beta[i + 1] - truth[i + 1]) < 0.12 for i in range(3))
    ok_sum = abs(sum(beta[1:]) - sum(truth[1:])) < 0.10
    ok_r2 = r2 > 0.20

    # Leverage recovery: inject a DGP where down-bar returns raise next RV, confirm β_lev>0.
    z2 = _seeded_normals(n, seed=21)
    zr = _seeded_normals(n, seed=29)                     # independent return shocks
    ret = [0.0] * lag_m
    rv2 = [0.001] * lag_m
    # smaller, stationary leverage loading: the RV recursion slopes still sum <1 and the
    # leverage feedback is bounded (returns are drawn with a FIXED scale, not sqrt(RV), so
    # the system can't blow up). β_lev just needs to be recovered with the right sign.
    blev = 0.8
    ret_scale = 0.04
    for t in range(lag_m, n):
        rv_d = rv2[t - 1]
        rv_w = math.fsum(rv2[t - lag_w:t]) / lag_w
        rv_m = math.fsum(rv2[t - lag_m:t]) / lag_m
        down = ret[t - 1] if ret[t - 1] < 0 else 0.0
        mean = b0 + bd * rv_d + bw * rv_w + bm * rv_m + blev * (down * down)
        rv2.append(max(1e-9, mean * math.exp(0.15 * z2[t] - 0.5 * 0.15 ** 2)))
        ret.append(ret_scale * zr[t])                    # fixed-scale return shocks (stationary)
    Xl, yl = har_design(rv2, ret, lag_w, lag_m)
    betal = ols_fit(Xl, yl, ridge=0.0)
    r2l = r_squared(yl, predict(Xl, betal))
    print(f"\n  leverage DGP (true β_lev={blev:+.2f}):")
    for nm, est in zip(["β0", "β_d", "β_w", "β_m", "β_lev"], betal):
        print(f"    {nm:5s} = {est:+.4f}")
    print(f"  in-sample R² = {r2l:.4f}")
    ok_lev = betal[4] > 0.0                               # leverage loads positive

    # Online walk-forward sanity: forecasts are finite and positive on the no-lev series.
    oh = OnlineHAR(window=800, leverage=False)
    fc_ok = True
    for t in range(len(rv)):
        oh.update(rv[t], 0.0)
        if t % 50 == 0 and t > lag_m + 10:
            oh.fit()
        f = oh.forecast_next()
        if f is not None and (f != f or f < 0):          # NaN or negative
            fc_ok = False
    print()
    print(f"  [{'PASS' if ok_slopes else 'FAIL'}] HAR slopes recovered within 0.12 of truth")
    print(f"  [{'PASS' if ok_sum else 'FAIL'}] slope-sum recovered within 0.10")
    print(f"  [{'PASS' if ok_r2 else 'FAIL'}] in-sample R² > 0.20")
    print(f"  [{'PASS' if ok_lev else 'FAIL'}] β_lev > 0 on leverage DGP")
    print(f"  [{'PASS' if fc_ok else 'FAIL'}] online walk-forward forecasts finite & ≥0")
    allok = ok_slopes and ok_sum and ok_r2 and ok_lev and fc_ok
    print(f"  RESULT: {'ALL PASS' if allok else 'CHECK ABOVE'}")


# --------------------------------------------------------------------------- backtest
def backtest_db(db_path: str, bar_len: int = 30, robust: bool = False) -> None:
    """Per-asset HAR-RV fit from the live engine's stored spot history (decisions.spot).

    Pipeline: dedupe consecutive flat composite ticks → bar the return series (bar_len
    returns/bar) → per-bar RV (Module A) → HAR(+leverage) design → in-sample OLS fit.
    Reports the (β_d, β_w, β_m, β_lev) term structure + in-sample R² + a walk-forward
    one-step R² (causal OnlineHAR). Honest about n: ~6 days of feed ⇒ tens of bars/asset.
    """
    import sqlite3
    print(f"=== har_rv backtest on {db_path}  (bar_len={bar_len}, robust={robust}) ===")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    assets = [a for (a,) in conn.execute(
        "SELECT DISTINCT asset FROM decisions WHERE spot IS NOT NULL ORDER BY asset")]
    hdr = (f"  {'asset':6s} {'bars':>5s} {'fitN':>5s} {'β0':>9s} {'β_d':>7s} {'β_w':>7s} "
           f"{'β_m':>7s} {'β_lev':>8s} {'R²_is':>7s} {'R²_wf':>7s}")
    print(hdr)
    for a in assets:
        rows = [s for (s,) in conn.execute(
            "SELECT spot FROM decisions WHERE asset=? AND spot IS NOT NULL ORDER BY ts", (a,))]
        prices = [rows[0]] + [p for p, q in zip(rows[1:], rows[:-1]) if p != q] if rows else []
        rv, ret = bar_rv_series(prices, bar_len=bar_len, robust=robust)
        if len(rv) < LAG_M + 6:
            print(f"  {a:6s} {len(rv):>5d}  (too few bars for a {LAG_M}-bar monthly window)")
            continue
        X, y = har_design(rv, ret, LAG_W, LAG_M)
        # mild ridge: RV_d/RV_w/RV_m are collinear by construction on short samples
        beta = ols_fit(X, y, ridge=1e-12 * max((max(yy for yy in y) ** 2), 1e-18))
        r2_is = r_squared(y, predict(X, beta))

        # walk-forward (causal) one-step R²: refit every 5 bars on the trailing window.
        # Forecasts are clamped to [0, RV_cap] — variance can't be negative, and an
        # unstable short-sample fit must not emit an absurd value that swamps R². The cap
        # is a generous multiple of the largest RV seen so far (still causal).
        oh = OnlineHAR(window=len(rv), leverage=True)
        preds: list[float] = []
        actuals: list[float] = []
        rv_cap = 0.0
        for t in range(len(rv)):
            f = oh.forecast_next() if t > 0 else None
            oh.update(rv[t], ret[t])
            rv_cap = max(rv_cap, rv[t])
            if t % 5 == 0:
                oh.fit()
            if f is not None and t >= LAG_M + 1:
                preds.append(min(max(f, 0.0), 5.0 * rv_cap))   # economic clamp
                actuals.append(rv[t])                          # forecast at t-1 for bar t
        r2_wf = r_squared(actuals, preds) if len(actuals) > 3 else float("nan")

        blev = beta[4] if len(beta) > 4 else float("nan")
        print(f"  {a:6s} {len(rv):>5d} {len(y):>5d} {beta[0]:>9.2e} {beta[1]:>7.3f} "
              f"{beta[2]:>7.3f} {beta[3]:>7.3f} {blev:>8.3f} {r2_is:>7.3f} {r2_wf:>7.3f}")
    conn.close()
    print("\n  β_d/β_w/β_m = daily/weekly/monthly cascade loadings (bars = 1/5/22).")
    print("  β_lev = negative-return leverage loading (down-move bad-variance, [2]).")
    print("  R²_is = in-sample, R²_wf = causal walk-forward one-step (the honest number).")
    print("  NOTE: ~6 days of composite-tick feed ⇒ tens of bars/asset; in-sample term")
    print("        structure is indicative. Real predictive gate = cross_sectional.py (D).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        backtest_db(sys.argv[1])
    else:
        selftest()
