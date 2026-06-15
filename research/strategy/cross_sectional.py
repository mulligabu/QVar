"""Cross-sectional validator — Fama-MacBeth logistic for binary outcomes.

Module D of the full-stack research engine (see research/fullstack_research_engine.md).
The VALIDATION GATEKEEPER: measures incremental predictive power of variance/jump features
(Module A) BEYOND the market mid. A feature earns a live slot ONLY if it predicts outcome_up
significantly better than price alone.

Adapted Fama-MacBeth:
  - For each settlement period (daily window), fit a cross-market LOGISTIC regression of
    outcome_up on candidate features, ALWAYS controlling for logit(market_mid).
  - Aggregate the per-period coefficients: mean, time-series SE, t-stat.
  - Report OOS AUC / Brier for (mid alone) vs (mid + feature).

Features tested (all computed causally per-asset up to each market's decision time):
  - variance/jump features from variance_decomp.py: RV, bipower_var, jump_robust_var,
    pos_jump_var, neg_jump_var, jump_intensity, pos_jump_share, signed_jump_var
  - existing columns: z_dist, (p_cal - mid), regime (encoded).

CRITICAL: NO LOOK-AHEAD. For each market evaluated at time T, variance features are computed
from the asset's spot history STRICTLY BEFORE T (filtered by ts < market_ts), using a
lookback window (default 4 hours on 5m/15m, longer on 4h/1D).

stdlib only (no numpy/pandas/sklearn). Logistic regression via IRLS (Iteratively Reweighted
Least Squares). Run with no args for synthetic selftest; pass DB path for live backtest.

    python3 research/strategy/cross_sectional.py
    python3 research/strategy/cross_sectional.py data/forward_live/forward_engine.db
"""

from __future__ import annotations

import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass

# Import variance decomposition from Module A
import os
sys.path.insert(0, os.path.dirname(__file__))
from variance_decomp import decompose, log_returns


# ----------------------------------------------------------------------------- logistic regression (IRLS)
def _logistic(x: float) -> float:
    """σ(x) = 1 / (1 + exp(-x)), numerically stable."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _logit(p: float) -> float:
    """Inverse logistic: logit(p) = log(p/(1-p)). Guard extremes."""
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def _matrix_vec(A: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in A]


def _vec_outer(a: list[float], b: list[float]) -> list[list[float]]:
    """Outer product a ⊗ b."""
    return [[x * y for y in b] for x in a]


def _add_matrix(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _scale_matrix(A: list[list[float]], c: float) -> list[list[float]]:
    return [[c * x for x in row] for row in A]


def _cholesky(A: list[list[float]]) -> list[list[float]] | None:
    """Cholesky decomposition A = L L^T. Returns None if A is not positive definite."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = math.fsum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 1e-12:
                    return None
                L[i][j] = math.sqrt(val)
            else:
                if abs(L[j][j]) < 1e-12:
                    return None
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _forward_sub(L: list[list[float]], b: list[float]) -> list[float]:
    """Solve L x = b (lower triangular)."""
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        s = math.fsum(L[i][j] * x[j] for j in range(i))
        x[i] = (b[i] - s) / L[i][i]
    return x


def _backward_sub(LT: list[list[float]], b: list[float]) -> list[float]:
    """Solve L^T x = b (upper triangular in the transposed view)."""
    n = len(LT)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum(LT[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / LT[i][i]
    return x


def _solve_cholesky(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve A x = b via Cholesky A = LL^T."""
    L = _cholesky(A)
    if L is None:
        return None
    y = _forward_sub(L, b)
    x = _backward_sub(L, y)
    return x


def logistic_regression_irls(X: list[list[float]], y: list[int],
                              max_iter: int = 25, tol: float = 1e-6) -> list[float] | None:
    """Fit logistic regression via IRLS. X is n x p (each row is a feature vector for one sample),
    y is binary {0,1}. Returns coefficient vector β (length p), or None on failure.

    Iteratively solves:  (X^T W X) β_new = X^T W z,  where W = diag(π(1-π)), z = Xβ + W^{-1}(y-π).
    """
    n = len(X)
    if n == 0:
        return None
    p = len(X[0])
    beta = [0.0] * p
    for iteration in range(max_iter):
        eta = [_dot(X[i], beta) for i in range(n)]
        pi = [_logistic(e) for e in eta]
        # weights w_i = π_i(1-π_i)
        w = [p * (1 - p) for p in pi]
        # guard: if any w too small, numerical issue; floor it
        w = [max(wi, 1e-9) for wi in w]
        # working response z = Xβ + (y - π)/w
        z = [eta[i] + (y[i] - pi[i]) / w[i] for i in range(n)]
        # build X^T W X  and  X^T W z
        XtWX = [[0.0] * p for _ in range(p)]
        XtWz = [0.0] * p
        for i in range(n):
            for j in range(p):
                XtWz[j] += w[i] * X[i][j] * z[i]
                for k in range(p):
                    XtWX[j][k] += w[i] * X[i][j] * X[i][k]
        # solve for β_new
        beta_new = _solve_cholesky(XtWX, XtWz)
        if beta_new is None:
            return None
        # convergence check
        delta = math.sqrt(math.fsum((b - bn) ** 2 for b, bn in zip(beta, beta_new)))
        beta = beta_new
        if delta < tol:
            break
    return beta


def predict_logistic(X: list[list[float]], beta: list[float]) -> list[float]:
    """Return predicted probabilities π = σ(Xβ)."""
    return [_logistic(_dot(row, beta)) for row in X]


# ----------------------------------------------------------------------------- evaluation metrics
def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return math.fsum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _logloss(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    s = 0.0
    for p, o in zip(probs, outcomes):
        p = max(1e-6, min(1 - 1e-6, p))
        s += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return s / len(probs)


def _auc_roc(probs: list[float], outcomes: list[int]) -> float:
    """Wilcoxon-Mann-Whitney AUC: fraction of (pos, neg) pairs correctly ordered."""
    if not probs or len(set(outcomes)) < 2:
        return float("nan")
    pairs = sorted(zip(probs, outcomes), key=lambda x: x[0])
    n_pos = sum(outcomes)
    n_neg = len(outcomes) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    for i, (_, o) in enumerate(pairs):
        if o == 1:
            rank_sum += i + 1
    # AUC = (rank_sum - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ----------------------------------------------------------------------------- causal variance features
@dataclass
class VarianceFeatures:
    """Variance/jump features for a single market, computed causally from asset history."""
    rv: float = 0.0
    bipower_var: float = 0.0
    jump_robust_var: float = 0.0
    pos_jump_var: float = 0.0
    neg_jump_var: float = 0.0
    jump_intensity: float = 0.0
    pos_jump_share: float = 0.0
    signed_jump_var: float = 0.0
    n_rets: int = 0

    def to_list(self) -> list[float]:
        """Return features as a list (for regression matrix). Exclude n_rets (just metadata)."""
        return [
            self.rv, self.bipower_var, self.jump_robust_var,
            self.pos_jump_var, self.neg_jump_var, self.jump_intensity,
            self.pos_jump_share, self.signed_jump_var,
        ]


def compute_variance_features_causal(conn: sqlite3.Connection, asset: str, market_ts: str,
                                      lookback_seconds: float) -> VarianceFeatures:
    """Compute variance/jump features for `asset` using spot history STRICTLY BEFORE `market_ts`,
    within a lookback window. NO LOOK-AHEAD enforced by ts < market_ts filter.

    Returns VarianceFeatures (zeros if insufficient data).
    """
    # Parse market_ts to get cutoff time (strictly before)
    # Query: SELECT spot FROM decisions WHERE asset=? AND ts < ? AND spot IS NOT NULL
    #        ORDER BY ts DESC LIMIT <enough for lookback>
    # We'll pull the last N samples and filter by time delta.
    cutoff = market_ts
    rows = conn.execute(
        "SELECT ts, spot FROM decisions WHERE asset=? AND ts < ? AND spot IS NOT NULL "
        "ORDER BY ts DESC LIMIT 500",
        (asset, cutoff)
    ).fetchall()
    if len(rows) < 10:
        return VarianceFeatures()

    # Filter by lookback window
    from datetime import datetime, timezone
    try:
        cutoff_dt = datetime.fromisoformat(cutoff.replace('+00:00', '')).replace(tzinfo=timezone.utc)
    except:
        return VarianceFeatures()

    valid = []
    for ts_str, spot in rows:
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace('+00:00', '')).replace(tzinfo=timezone.utc)
            delta = (cutoff_dt - ts_dt).total_seconds()
            if 0 < delta <= lookback_seconds:
                valid.append(spot)
        except:
            continue

    if len(valid) < 10:
        return VarianceFeatures()

    # Reverse to chronological order for returns
    valid.reverse()
    rets = log_returns(valid)
    if len(rets) < 5:
        return VarianceFeatures()

    # Decompose (guard against short windows that break Lee-Mykland)
    try:
        d = decompose(rets, K=None, alpha=0.05)
    except (ZeroDivisionError, ValueError):
        # Window too short for jump detection; fall back to basic RV/bipower only
        from variance_decomp import realized_variance, bipower_variation
        rv = realized_variance(rets)
        bv = bipower_variation(rets)
        return VarianceFeatures(
            rv=rv, bipower_var=bv, jump_robust_var=min(bv, rv),
            pos_jump_var=0.0, neg_jump_var=0.0, jump_intensity=0.0,
            pos_jump_share=0.0, signed_jump_var=0.0, n_rets=len(rets)
        )
    return VarianceFeatures(
        rv=d["rv"],
        bipower_var=d["bipower_var"],
        jump_robust_var=d["jump_robust_var"],
        pos_jump_var=d["pos_jump_var"],
        neg_jump_var=d["neg_jump_var"],
        jump_intensity=d["jump_intensity"],
        pos_jump_share=d["pos_jump_share"],
        signed_jump_var=d["signed_jump_var"],
        n_rets=d["n_rets"],
    )


# ----------------------------------------------------------------------------- cross-sectional logic
@dataclass
class MarketObs:
    """A single market observation with features + outcome."""
    ts: str
    asset: str
    tf: str
    market_mid: float
    p_cal: float
    alpha_up: float
    outcome_up: int
    z_dist: float | None
    regime: str | None
    var_feats: VarianceFeatures


def _encode_regime(regime: str | None) -> list[float]:
    """One-hot encode regime into [is_high_vol, is_trending] (simple 2-bit encoding).
    high_vol/mean_revert -> [1, 0]; low_vol/trending -> [0, 1]; else [0, 0].
    """
    if regime is None:
        return [0.0, 0.0]
    if "high_vol" in regime:
        return [1.0, 0.0]
    if "trending" in regime:
        return [0.0, 1.0]
    return [0.0, 0.0]


def build_feature_vector(obs: MarketObs, include_mid: bool = True,
                          include_variance: bool = False,
                          include_z_dist: bool = False,
                          include_p_cal_delta: bool = False,
                          include_regime: bool = False) -> list[float]:
    """Build feature vector for logistic regression. ALWAYS include intercept at position 0.
    Control flags specify which groups to include.
    """
    feats = [1.0]  # intercept
    if include_mid:
        feats.append(_logit(obs.market_mid))
    if include_variance:
        feats.extend(obs.var_feats.to_list())
    if include_z_dist and obs.z_dist is not None:
        feats.append(obs.z_dist)
    if include_p_cal_delta:
        feats.append(obs.p_cal - obs.market_mid)
    if include_regime:
        feats.extend(_encode_regime(obs.regime))
    return feats


def feature_names(include_mid: bool = True,
                  include_variance: bool = False,
                  include_z_dist: bool = False,
                  include_p_cal_delta: bool = False,
                  include_regime: bool = False) -> list[str]:
    """Return feature names matching build_feature_vector."""
    names = ["intercept"]
    if include_mid:
        names.append("logit_mid")
    if include_variance:
        names.extend(["rv", "bipower_var", "jump_robust_var", "pos_jump_var",
                      "neg_jump_var", "jump_intensity", "pos_jump_share", "signed_jump_var"])
    if include_z_dist:
        names.append("z_dist")
    if include_p_cal_delta:
        names.append("p_cal_delta")
    if include_regime:
        names.extend(["regime_high_vol", "regime_trending"])
    return names


def _standardize_features(X: list[list[float]], skip_intercept: bool = True) -> list[list[float]]:
    """Standardize features (z-score) to prevent numerical issues. Skip intercept column."""
    if not X:
        return X
    n = len(X)
    p = len(X[0])

    # Compute mean and std for each feature
    means = []
    stds = []
    for j in range(p):
        if skip_intercept and j == 0:
            means.append(0.0)
            stds.append(1.0)
            continue
        vals = [X[i][j] for i in range(n)]
        m = math.fsum(vals) / n
        v = math.fsum((x - m) ** 2 for x in vals) / max(n - 1, 1)
        s = math.sqrt(v) if v > 0 else 1.0
        # Guard: if std is tiny (near-zero variance), keep scale=1 to avoid inflation
        if s < 1e-10:
            s = 1.0
        means.append(m)
        stds.append(s)

    # Standardize
    X_std = []
    for i in range(n):
        row = []
        for j in range(p):
            if skip_intercept and j == 0:
                row.append(X[i][j])
            else:
                row.append((X[i][j] - means[j]) / stds[j])
        X_std.append(row)

    return X_std


def fama_macbeth_logistic(periods: list[list[MarketObs]],
                           include_mid: bool = True,
                           include_variance: bool = False,
                           include_z_dist: bool = False,
                           include_p_cal_delta: bool = False,
                           include_regime: bool = False,
                           standardize: bool = True) -> dict:
    """Fama-MacBeth cross-sectional logistic regression across multiple periods.

    For each period, fit a logistic regression of outcome_up on the selected features.
    Aggregate coefficients across periods: mean, time-series SE, t-stat.

    Returns:
        {
            "n_periods": int,
            "feature_names": list[str],
            "mean_coefs": list[float],
            "se_coefs": list[float],        # time-series SE
            "t_stats": list[float],
            "failed_periods": int,          # periods where regression failed
        }
    """
    feature_names_list = feature_names(include_mid, include_variance, include_z_dist,
                                        include_p_cal_delta, include_regime)
    n_feats = len(feature_names_list)
    period_coefs = []

    for period_obs in periods:
        if len(period_obs) < 5:
            continue
        X = [build_feature_vector(obs, include_mid, include_variance, include_z_dist,
                                   include_p_cal_delta, include_regime) for obs in period_obs]
        y = [obs.outcome_up for obs in period_obs]
        # Require some variation in outcome
        if len(set(y)) < 2:
            continue
        # Standardize features to prevent numerical issues
        if standardize:
            X = _standardize_features(X, skip_intercept=True)
        beta = logistic_regression_irls(X, y)
        if beta is not None and len(beta) == n_feats:
            period_coefs.append(beta)

    if not period_coefs:
        return {
            "n_periods": 0,
            "feature_names": feature_names_list,
            "mean_coefs": [float("nan")] * n_feats,
            "se_coefs": [float("nan")] * n_feats,
            "t_stats": [float("nan")] * n_feats,
            "failed_periods": len(periods),
        }

    n_periods = len(period_coefs)
    # Mean coefficient across periods
    mean_coefs = [math.fsum(c[j] for c in period_coefs) / n_periods for j in range(n_feats)]
    # Time-series SE (standard deviation / sqrt(n_periods))
    var_coefs = [math.fsum((c[j] - mean_coefs[j]) ** 2 for c in period_coefs) / max(n_periods - 1, 1)
                 for j in range(n_feats)]
    se_coefs = [math.sqrt(v / n_periods) for v in var_coefs]
    # t-stat
    t_stats = [mean_coefs[j] / se_coefs[j] if se_coefs[j] > 1e-12 else float("nan")
               for j in range(n_feats)]

    return {
        "n_periods": n_periods,
        "feature_names": feature_names_list,
        "mean_coefs": mean_coefs,
        "se_coefs": se_coefs,
        "t_stats": t_stats,
        "failed_periods": len(periods) - n_periods,
    }


def oos_validation(all_obs: list[MarketObs], n_folds: int = 5, standardize: bool = True, **feat_flags) -> dict:
    """Out-of-sample AUC/Brier via time-ordered cross-validation.

    Split observations into n_folds chronological chunks. For each fold as test, train on all
    earlier folds, predict on test fold. Aggregate predictions and compute metrics.

    Returns: {"auc": float, "brier": float, "logloss": float, "n_test": int}
    """
    if len(all_obs) < n_folds * 5:
        return {"auc": float("nan"), "brier": float("nan"), "logloss": float("nan"), "n_test": 0}

    # Sort by timestamp
    all_obs = sorted(all_obs, key=lambda o: o.ts)
    fold_size = len(all_obs) // n_folds
    all_preds = []
    all_outcomes = []

    for i in range(1, n_folds):
        train = all_obs[:i * fold_size]
        test = all_obs[i * fold_size:(i + 1) * fold_size] if i < n_folds - 1 else all_obs[i * fold_size:]
        if len(train) < 5 or len(test) < 1:
            continue
        X_train = [build_feature_vector(o, **feat_flags) for o in train]
        X_test = [build_feature_vector(o, **feat_flags) for o in test]
        y_train = [o.outcome_up for o in train]
        if len(set(y_train)) < 2:
            continue

        # Standardize using training set statistics (prevent leakage)
        if standardize and len(X_train[0]) > 1:
            n_tr = len(X_train)
            p = len(X_train[0])
            means = []
            stds = []
            for j in range(p):
                if j == 0:  # skip intercept
                    means.append(0.0)
                    stds.append(1.0)
                    continue
                vals = [X_train[k][j] for k in range(n_tr)]
                m = math.fsum(vals) / n_tr
                v = math.fsum((x - m) ** 2 for x in vals) / max(n_tr - 1, 1)
                s = math.sqrt(v) if v > 0 else 1.0
                if s < 1e-10:
                    s = 1.0
                means.append(m)
                stds.append(s)

            # Apply to both train and test
            X_train = [[(x[j] - means[j]) / stds[j] if j > 0 else x[j] for j in range(p)] for x in X_train]
            X_test = [[(x[j] - means[j]) / stds[j] if j > 0 else x[j] for j in range(p)] for x in X_test]

        beta = logistic_regression_irls(X_train, y_train)
        if beta is None:
            continue
        preds = predict_logistic(X_test, beta)
        all_preds.extend(preds)
        all_outcomes.extend([o.outcome_up for o in test])

    if not all_preds:
        return {"auc": float("nan"), "brier": float("nan"), "logloss": float("nan"), "n_test": 0}

    return {
        "auc": _auc_roc(all_preds, all_outcomes),
        "brier": _brier(all_preds, all_outcomes),
        "logloss": _logloss(all_preds, all_outcomes),
        "n_test": len(all_preds),
    }


# ----------------------------------------------------------------------------- selftest
def selftest() -> None:
    """Synthetic selftest: inject a KNOWN signal that predicts beyond a noisy mid, verify
    the harness recovers it and rejects a pure-noise feature.
    """
    import random
    random.seed(42)

    print("=== cross_sectional selftest (synthetic, known signal) ===")

    # Generate synthetic data: outcome_up is Bernoulli(true_prob),
    # true_prob = σ(0.5*logit(noisy_mid) + 1.2*signal_feat + noise).
    # The "noisy_mid" alone is informative but imperfect; signal_feat has INCREMENTAL power.
    n_periods = 20
    n_per_period = 50
    all_obs = []
    periods = []

    for _ in range(n_periods):
        period = []
        for _ in range(n_per_period):
            # True underlying probability
            u = random.uniform(0.2, 0.8)
            # Signal feature (ranges -1 to +1)
            signal = random.uniform(-1, 1)
            # Noisy mid: correlated with u but with noise
            noisy_mid = max(0.05, min(0.95, u + random.gauss(0, 0.15)))
            # True outcome probability combines noisy_mid and signal
            logit_mid = _logit(noisy_mid)
            true_logit = 0.5 * logit_mid + 1.2 * signal
            true_prob = _logistic(true_logit)
            outcome = 1 if random.random() < true_prob else 0

            # Build a synthetic MarketObs
            vf = VarianceFeatures()
            vf.rv = signal  # abuse rv field to hold our signal feature
            vf.bipower_var = random.gauss(0, 0.1)  # pure noise feature
            obs = MarketObs(
                ts=f"2026-01-01T{_:02d}:00:00",
                asset="SYN", tf="5m",
                market_mid=noisy_mid,
                p_cal=noisy_mid,
                alpha_up=noisy_mid,
                outcome_up=outcome,
                z_dist=None, regime=None,
                var_feats=vf,
            )
            period.append(obs)
            all_obs.append(obs)
        periods.append(period)

    # Test 1: mid alone should be predictive
    print("\n--- Test 1: mid alone (baseline) ---")
    fm_mid = fama_macbeth_logistic(periods, include_mid=True, include_variance=False)
    print(f"  periods: {fm_mid['n_periods']}")
    for i, name in enumerate(fm_mid["feature_names"]):
        print(f"  {name:15s}  coef={fm_mid['mean_coefs'][i]:+.3f}  SE={fm_mid['se_coefs'][i]:.3f}  "
              f"t={fm_mid['t_stats'][i]:+.2f}")
    oos_mid = oos_validation(all_obs, include_mid=True, include_variance=False)
    print(f"  OOS: AUC={oos_mid['auc']:.3f}  Brier={oos_mid['brier']:.3f}")

    # Test 2: mid + signal feature (RV) should improve
    # We'll use include_variance=True but only RV is the signal; bipower is noise
    print("\n--- Test 2: mid + RV (signal feature) ---")
    # For this test, we'll manually build a feature set with just mid + RV
    # Actually, let's use the variance flag and check if RV coefficient is significant
    # But our variance_feats.to_list() includes all 8 features. For clean test, we need
    # to modify. Let's instead use a simpler approach: manually inject it.
    # Actually, let's redefine the test to just check if adding variance features helps.

    # Simplified: we'll define a custom feature builder for this test
    def build_feat_signal_only(obs: MarketObs) -> list[float]:
        return [1.0, _logit(obs.market_mid), obs.var_feats.rv]

    # Manual FM for this test
    period_coefs = []
    for period in periods:
        if len(period) < 5:
            continue
        X = [build_feat_signal_only(o) for o in period]
        y = [o.outcome_up for o in period]
        if len(set(y)) < 2:
            continue
        beta = logistic_regression_irls(X, y)
        if beta is not None:
            period_coefs.append(beta)

    n_per = len(period_coefs)
    mean_c = [math.fsum(c[j] for c in period_coefs) / n_per for j in range(3)]
    var_c = [math.fsum((c[j] - mean_c[j]) ** 2 for c in period_coefs) / max(n_per - 1, 1) for j in range(3)]
    se_c = [math.sqrt(v / n_per) for v in var_c]
    t_c = [mean_c[j] / se_c[j] if se_c[j] > 1e-12 else 0 for j in range(3)]

    print(f"  periods: {n_per}")
    for i, name in enumerate(["intercept", "logit_mid", "signal_feat"]):
        print(f"  {name:15s}  coef={mean_c[i]:+.3f}  SE={se_c[i]:.3f}  t={t_c[i]:+.2f}")

    # OOS for mid+signal
    all_preds = []
    all_outcomes = []
    fold_size = len(all_obs) // 5
    for i in range(1, 5):
        train = all_obs[:i * fold_size]
        test = all_obs[i * fold_size:(i + 1) * fold_size] if i < 4 else all_obs[i * fold_size:]
        X_tr = [build_feat_signal_only(o) for o in train]
        y_tr = [o.outcome_up for o in train]
        if len(set(y_tr)) < 2:
            continue
        beta = logistic_regression_irls(X_tr, y_tr)
        if beta is None:
            continue
        X_te = [build_feat_signal_only(o) for o in test]
        preds = predict_logistic(X_te, beta)
        all_preds.extend(preds)
        all_outcomes.extend([o.outcome_up for o in test])

    auc_signal = _auc_roc(all_preds, all_outcomes)
    brier_signal = _brier(all_preds, all_outcomes)
    print(f"  OOS: AUC={auc_signal:.3f}  Brier={brier_signal:.3f}")

    # Test 3: mid + noise feature (bipower_var) should NOT help significantly
    def build_feat_noise_only(obs: MarketObs) -> list[float]:
        return [1.0, _logit(obs.market_mid), obs.var_feats.bipower_var]

    period_coefs_n = []
    for period in periods:
        if len(period) < 5:
            continue
        X = [build_feat_noise_only(o) for o in period]
        y = [o.outcome_up for o in period]
        if len(set(y)) < 2:
            continue
        beta = logistic_regression_irls(X, y)
        if beta is not None:
            period_coefs_n.append(beta)

    n_per_n = len(period_coefs_n)
    mean_cn = [math.fsum(c[j] for c in period_coefs_n) / n_per_n for j in range(3)]
    var_cn = [math.fsum((c[j] - mean_cn[j]) ** 2 for c in period_coefs_n) / max(n_per_n - 1, 1) for j in range(3)]
    se_cn = [math.sqrt(v / n_per_n) for v in var_cn]
    t_cn = [mean_cn[j] / se_cn[j] if se_cn[j] > 1e-12 else 0 for j in range(3)]

    print("\n--- Test 3: mid + noise feature (expect no signal) ---")
    print(f"  periods: {n_per_n}")
    for i, name in enumerate(["intercept", "logit_mid", "noise_feat"]):
        print(f"  {name:15s}  coef={mean_cn[i]:+.3f}  SE={se_cn[i]:.3f}  t={t_cn[i]:+.2f}")

    # Assertions
    ok_mid = abs(t_c[1]) > 2.0  # logit_mid should be significant in signal model
    ok_signal = abs(t_c[2]) > 2.0  # signal_feat should be significant (|t| > 2)
    ok_noise = abs(t_cn[2]) < 2.0  # noise_feat should NOT be significant
    ok_lift = auc_signal > oos_mid["auc"] + 0.02  # signal model should have higher AUC

    print(f"\n  [{'PASS' if ok_mid else 'FAIL'}] logit_mid significant in signal model (|t|={abs(t_c[1]):.1f}>2)")
    print(f"  [{'PASS' if ok_signal else 'FAIL'}] signal_feat significant (|t|={abs(t_c[2]):.1f}>2)")
    print(f"  [{'PASS' if ok_noise else 'FAIL'}] noise_feat NOT significant (|t|={abs(t_cn[2]):.1f}<2)")
    print(f"  [{'PASS' if ok_lift else 'FAIL'}] signal model lifts AUC ({auc_signal:.3f} > {oos_mid['auc']:.3f})")
    print(f"\n  RESULT: {'ALL PASS' if all([ok_mid, ok_signal, ok_noise, ok_lift]) else 'CHECK ABOVE'}")


# ----------------------------------------------------------------------------- live backtest
def backtest_db(db_path: str) -> None:
    """Cross-sectional validation on the live forward_engine.db.

    Load settled markets from calib table, compute variance features CAUSALLY per-asset
    (lookback window, strict ts < market_ts), run Fama-MacBeth and OOS tests.

    HONEST REPORTING: state sample size, multiple testing, and the kill criterion.
    """
    print(f"\n=== cross_sectional backtest on {db_path} ===")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # Load settled markets with outcome
    rows = conn.execute("""
        SELECT c.ts, c.asset, c.tf, c.market_mid, c.p_cal, c.alpha_up, c.outcome_up,
               d.z_dist, d.regime, c.market
        FROM calib c
        LEFT JOIN decisions d ON c.market = d.market
        WHERE c.outcome_up IS NOT NULL
        ORDER BY c.ts
    """).fetchall()

    print(f"  total settled markets: {len(rows)}")

    # Define lookback windows by timeframe (in seconds)
    lookback_map = {
        "5m": 4 * 3600,      # 4 hours
        "15m": 6 * 3600,     # 6 hours
        "1h": 24 * 3600,     # 24 hours
        "4h": 7 * 24 * 3600, # 7 days
        "1D": 30 * 24 * 3600, # 30 days
    }

    # Build MarketObs with causal variance features
    all_obs = []
    print("  computing causal variance features per market...")
    for i, row in enumerate(rows):
        ts, asset, tf, mid, p_cal, alpha, outcome, z_dist, regime, market = row
        if i % 2000 == 0 and i > 0:
            print(f"    processed {i}/{len(rows)}...")
        lookback = lookback_map.get(tf, 4 * 3600)
        var_feats = compute_variance_features_causal(conn, asset, ts, lookback)
        obs = MarketObs(
            ts=ts, asset=asset, tf=tf,
            market_mid=mid, p_cal=p_cal, alpha_up=alpha, outcome_up=outcome,
            z_dist=z_dist, regime=regime,
            var_feats=var_feats,
        )
        all_obs.append(obs)

    print(f"  built {len(all_obs)} observations with variance features")

    # Split into daily periods for Fama-MacBeth
    from datetime import datetime, timezone
    periods_dict = defaultdict(list)
    for obs in all_obs:
        try:
            dt = datetime.fromisoformat(obs.ts.replace('+00:00', '')).replace(tzinfo=timezone.utc)
            day = dt.date()
            periods_dict[day].append(obs)
        except:
            continue

    periods = [v for k, v in sorted(periods_dict.items()) if len(v) >= 5]
    print(f"  periods (daily): {len(periods)}  (avg {sum(len(p) for p in periods) / len(periods):.1f} markets/day)")

    # DATA REALITY CHECK
    # Count how many obs have non-zero variance features
    n_with_var = sum(1 for o in all_obs if o.var_feats.n_rets >= 5)
    print(f"  markets with variance features (>=5 rets): {n_with_var} / {len(all_obs)}")

    if n_with_var < 100:
        print("\n  WARNING: very few markets have variance features; results will be noisy.")

    # --- Baseline: market mid alone ---
    print("\n--- Baseline: market mid alone (logit transform) ---")
    fm_mid = fama_macbeth_logistic(periods, include_mid=True)
    print(f"  periods: {fm_mid['n_periods']}  failed: {fm_mid['failed_periods']}")
    for i, name in enumerate(fm_mid["feature_names"]):
        c, se, t = fm_mid["mean_coefs"][i], fm_mid["se_coefs"][i], fm_mid["t_stats"][i]
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.65 else ""))
        print(f"  {name:20s}  coef={c:+7.3f}  SE={se:6.3f}  t={t:+6.2f} {sig}")

    oos_mid = oos_validation(all_obs, include_mid=True)
    print(f"  OOS (5-fold): AUC={oos_mid['auc']:.4f}  Brier={oos_mid['brier']:.4f}  "
          f"LogLoss={oos_mid['logloss']:.4f}  n_test={oos_mid['n_test']}")

    # --- Test 1: mid + all variance features ---
    print("\n--- mid + variance/jump features (all 8) ---")
    fm_var = fama_macbeth_logistic(periods, include_mid=True, include_variance=True)
    print(f"  periods: {fm_var['n_periods']}  failed: {fm_var['failed_periods']}")
    for i, name in enumerate(fm_var["feature_names"]):
        c, se, t = fm_var["mean_coefs"][i], fm_var["se_coefs"][i], fm_var["t_stats"][i]
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.65 else ""))
        print(f"  {name:20s}  coef={c:+7.3f}  SE={se:6.3f}  t={t:+6.2f} {sig}")

    oos_var = oos_validation(all_obs, include_mid=True, include_variance=True)
    print(f"  OOS (5-fold): AUC={oos_var['auc']:.4f}  Brier={oos_var['brier']:.4f}  "
          f"LogLoss={oos_var['logloss']:.4f}  n_test={oos_var['n_test']}")
    print(f"  LIFT vs mid alone: AUC {oos_var['auc'] - oos_mid['auc']:+.4f}  "
          f"Brier {oos_var['brier'] - oos_mid['brier']:+.4f}")

    # --- Test 2: mid + z_dist ---
    print("\n--- mid + z_dist ---")
    fm_z = fama_macbeth_logistic(periods, include_mid=True, include_z_dist=True)
    print(f"  periods: {fm_z['n_periods']}  failed: {fm_z['failed_periods']}")
    for i, name in enumerate(fm_z["feature_names"]):
        c, se, t = fm_z["mean_coefs"][i], fm_z["se_coefs"][i], fm_z["t_stats"][i]
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.65 else ""))
        print(f"  {name:20s}  coef={c:+7.3f}  SE={se:6.3f}  t={t:+6.2f} {sig}")

    oos_z = oos_validation(all_obs, include_mid=True, include_z_dist=True)
    print(f"  OOS (5-fold): AUC={oos_z['auc']:.4f}  Brier={oos_z['brier']:.4f}  "
          f"LogLoss={oos_z['logloss']:.4f}  n_test={oos_z['n_test']}")
    print(f"  LIFT vs mid alone: AUC {oos_z['auc'] - oos_mid['auc']:+.4f}  "
          f"Brier {oos_z['brier'] - oos_mid['brier']:+.4f}")

    # --- Test 3: mid + (p_cal - mid) ---
    print("\n--- mid + (p_cal - mid) calibration delta ---")
    fm_cal = fama_macbeth_logistic(periods, include_mid=True, include_p_cal_delta=True)
    print(f"  periods: {fm_cal['n_periods']}  failed: {fm_cal['failed_periods']}")
    for i, name in enumerate(fm_cal["feature_names"]):
        c, se, t = fm_cal["mean_coefs"][i], fm_cal["se_coefs"][i], fm_cal["t_stats"][i]
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.65 else ""))
        print(f"  {name:20s}  coef={c:+7.3f}  SE={se:6.3f}  t={t:+6.2f} {sig}")

    oos_cal = oos_validation(all_obs, include_mid=True, include_p_cal_delta=True)
    print(f"  OOS (5-fold): AUC={oos_cal['auc']:.4f}  Brier={oos_cal['brier']:.4f}  "
          f"LogLoss={oos_cal['logloss']:.4f}  n_test={oos_cal['n_test']}")
    print(f"  LIFT vs mid alone: AUC {oos_cal['auc'] - oos_mid['auc']:+.4f}  "
          f"Brier {oos_cal['brier'] - oos_mid['brier']:+.4f}")

    # --- Test 4: mid + regime ---
    print("\n--- mid + regime ---")
    fm_reg = fama_macbeth_logistic(periods, include_mid=True, include_regime=True)
    print(f"  periods: {fm_reg['n_periods']}  failed: {fm_reg['failed_periods']}")
    for i, name in enumerate(fm_reg["feature_names"]):
        c, se, t = fm_reg["mean_coefs"][i], fm_reg["se_coefs"][i], fm_reg["t_stats"][i]
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.65 else ""))
        print(f"  {name:20s}  coef={c:+7.3f}  SE={se:6.3f}  t={t:+6.2f} {sig}")

    oos_reg = oos_validation(all_obs, include_mid=True, include_regime=True)
    print(f"  OOS (5-fold): AUC={oos_reg['auc']:.4f}  Brier={oos_reg['brier']:.4f}  "
          f"LogLoss={oos_reg['logloss']:.4f}  n_test={oos_reg['n_test']}")
    print(f"  LIFT vs mid alone: AUC {oos_reg['auc'] - oos_mid['auc']:+.4f}  "
          f"Brier {oos_reg['brier'] - oos_mid['brier']:+.4f}")

    # --- Full model: mid + variance + z_dist + p_cal_delta + regime ---
    print("\n--- Full model: mid + variance + z_dist + p_cal_delta + regime ---")
    fm_full = fama_macbeth_logistic(periods, include_mid=True, include_variance=True,
                                     include_z_dist=True, include_p_cal_delta=True,
                                     include_regime=True)
    print(f"  periods: {fm_full['n_periods']}  failed: {fm_full['failed_periods']}")
    for i, name in enumerate(fm_full["feature_names"]):
        c, se, t = fm_full["mean_coefs"][i], fm_full["se_coefs"][i], fm_full["t_stats"][i]
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.65 else ""))
        print(f"  {name:20s}  coef={c:+7.3f}  SE={se:6.3f}  t={t:+6.2f} {sig}")

    oos_full = oos_validation(all_obs, include_mid=True, include_variance=True,
                               include_z_dist=True, include_p_cal_delta=True,
                               include_regime=True)
    print(f"  OOS (5-fold): AUC={oos_full['auc']:.4f}  Brier={oos_full['brier']:.4f}  "
          f"LogLoss={oos_full['logloss']:.4f}  n_test={oos_full['n_test']}")
    print(f"  LIFT vs mid alone: AUC {oos_full['auc'] - oos_mid['auc']:+.4f}  "
          f"Brier {oos_full['brier'] - oos_mid['brier']:+.4f}")

    conn.close()

    # --- INTERPRETATION & KILL CRITERION ---
    print("\n" + "=" * 80)
    print("SUMMARY & KILL CRITERION")
    print("=" * 80)
    print(f"Sample size: {len(all_obs)} markets across {len(periods)} daily periods")
    print(f"Markets with variance features: {n_with_var} ({100 * n_with_var / len(all_obs):.1f}%)")
    print()
    print("BASELINE (mid alone):")
    print(f"  AUC={oos_mid['auc']:.4f}  Brier={oos_mid['brier']:.4f}")
    print()
    print("INCREMENTAL TESTS (beyond mid):")
    tests = [
        ("z_dist", oos_z['auc'] - oos_mid['auc'], fm_z['t_stats'][2] if len(fm_z['t_stats']) > 2 else float('nan')),
        ("p_cal_delta", oos_cal['auc'] - oos_mid['auc'], fm_cal['t_stats'][2] if len(fm_cal['t_stats']) > 2 else float('nan')),
        ("variance/jump", oos_var['auc'] - oos_mid['auc'], float('nan')),  # multi-feature, no single t
        ("regime", oos_reg['auc'] - oos_mid['auc'], float('nan')),
        ("full model", oos_full['auc'] - oos_mid['auc'], float('nan')),
    ]
    for name, lift, t in tests:
        print(f"  {name:20s}  AUC lift: {lift:+.4f}  {'  (t=' + f'{t:+.2f})' if not math.isnan(t) else ''}")

    print()
    print("KILL CRITERION (per-feature, before live integration):")
    print("  1. |t-stat| > 1.96 in Fama-MacBeth (p<0.05, two-tailed)")
    print("  2. OOS AUC lift > +0.005 vs mid alone (0.5 bps, practical threshold)")
    print("  3. Sign of effect stable across at least 2/3 of periods (robustness check)")
    print()
    print("HONEST CAVEATS:")
    print(f"  - Small n: {len(periods)} periods is THIN; treat SIGN not magnitude.")
    print(f"  - Multiple testing: {8 + 3} features tested; Bonferroni => p<0.05/{8 + 3}≈0.0045.")
    print("  - Variance features sparse: only computed where lookback has >=10 returns.")
    print("  - Down-regime bias: this data is 2026-06-05 to 2026-06-10, check regime mix.")
    print()
    print("NEXT STEP: if any feature passes the kill criterion, validate in shadow A/B")
    print("           (fork engine, log signals, measure live calibration before sizing).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        backtest_db(sys.argv[1])
    else:
        selftest()
