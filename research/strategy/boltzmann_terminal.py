"""Boltzmann / max-entropy terminal-price distribution for binary pricing.

The terminal log-return over the remaining window is modeled as a generalized
normal (Subbotin / exponential-power) distribution -- a Gibbs/Boltzmann form

    p(r) proportional to exp( - |r / s|^beta / beta )

where:
  * beta = 2 recovers the Gaussian (what naive pricing assumes),
  * beta < 2 gives FAT TAILS (crypto reality) -> more mass on large moves,
  * s (scale, the "temperature") is set by the Markov vol forecast for the
    remaining time: s = sigma_window * sqrt(tau_remaining / window).

P(up) = P(end >= strike) = P(r >= ln(strike/spot)). Because the edge is meant to
come from volatility/tails (not drift), the distribution is centered at the
current spot (driftless) -- consistent with the finding that direction is a
coinflip. The model's advantage over the market is a better *shape/scale*, i.e.
better implied vol and tail behavior, which matters most for off-0.50 contracts.

Also exposes fractional-Kelly sizing with a Boltzmann temperature that shrinks
size as regime entropy rises.
"""

from __future__ import annotations

import math

WINDOW_SECONDS = 15 * 60


def _gennorm_sf(z: float, beta: float) -> float:
    """Survival function P(R/s >= z) for a standard generalized-normal(beta).

    Symmetric about 0. Uses the regularized lower incomplete gamma via series/
    continued fraction (stdlib only).
    """
    if z == 0:
        return 0.5
    t = (abs(z) ** beta) / beta
    # CDF of |R/s|^beta/beta ~ Gamma(shape=1/beta, scale=1): P(|.| <= z) = gammainc(1/beta, t)
    p_abs = _gammainc(1.0 / beta, t)  # P(|R/s| <= |z|)
    tail = 0.5 * (1.0 - p_abs)        # one-sided tail beyond |z|
    return tail if z > 0 else 1.0 - tail


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x). Stdlib-only implementation."""
    if x <= 0:
        return 0.0
    if a <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        # series expansion
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    # continued fraction for upper gamma Q, then P = 1 - Q
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return 1.0 - q


def prob_up(spot: float, strike: float, sigma_window: float, tau_seconds: float, beta: float = 1.5) -> float:
    """P(terminal price >= strike) under the driftless Boltzmann terminal dist."""
    if spot <= 0 or strike <= 0 or sigma_window <= 0 or tau_seconds <= 0:
        return 0.5
    scale = sigma_window * math.sqrt(tau_seconds / WINDOW_SECONDS)
    if scale <= 0:
        return 0.5
    r_strike = math.log(strike / spot)
    z = r_strike / scale
    # P(R >= r_strike) = survival at z
    return max(0.0, min(1.0, _gennorm_sf(z, beta)))


def implied_scale_from_market(spot: float, strike: float, market_p_up: float, tau_seconds: float,
                              beta: float = 1.5) -> float | None:
    """Invert prob_up to back out the market's implied window-sigma (vol)."""
    if not (0.0 < market_p_up < 1.0) or spot <= 0 or strike <= 0:
        return None
    r_strike = math.log(strike / spot)
    if abs(r_strike) < 1e-9:
        return None
    # solve _gennorm_sf(r_strike/scale, beta) = market_p_up for scale via bisection
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        p = _gennorm_sf(r_strike / mid, beta)
        if p > market_p_up:
            # too much mass above strike -> scale too large (if r_strike<0) ... monotonic handling
            hi = mid
        else:
            lo = mid
    scale = math.sqrt(lo * hi)
    return scale / math.sqrt(tau_seconds / WINDOW_SECONDS)


def kelly_fraction(model_p: float, price: float, temperature: float = 1.0, cap: float = 0.25) -> float:
    """Fractional Kelly for a binary at `price` with model win prob `model_p`.

    `temperature` >= 1 shrinks the bet (set from regime entropy); cap limits size.
    Payout is 1 per contract; b = (1-price)/price.
    """
    if not (0.0 < price < 1.0) or not (0.0 <= model_p <= 1.0):
        return 0.0
    b = (1.0 - price) / price
    f = (model_p * (b + 1.0) - 1.0) / b  # = (p*(1+b)-1)/b
    f = max(0.0, f) / max(1e-6, temperature)
    return min(cap, f)
