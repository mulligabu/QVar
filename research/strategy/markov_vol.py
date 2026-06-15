"""Markov regime-switching VOLATILITY forecaster (direction-agnostic).

Direction is a coinflip in 15m crypto binaries; volatility is not (it clusters).
This models the *scale* of the next 15m move, never its sign.

State = discretized recent realized-vol regime (optionally crossed with a trend
regime, though trend is not used for the sign). We build a walk-forward transition
matrix over the realized-vol-regime sequence and forecast the next window's sigma
as the regime-conditional expected realized vol, plus an entropy/confidence score
(how concentrated the transition row is) that feeds Boltzmann sizing.

All inputs are causal: regime at window i uses only candles completed by the start
of window i; the realized vol it predicts is window i's own forward move.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

from research.edge_proof.outcomes import CandleSeries

WINDOW_SECONDS = 15 * 60


@dataclass
class VolForecast:
    sigma_window: float       # forecast std of the 15m fractional return
    regime: int               # current vol regime index
    entropy: float            # transition-row entropy (0 = certain, 1 = max)
    confidence: float         # 1 - normalized entropy
    n_obs: int                # samples behind the forecast


def _minute_index(series: CandleSeries, epoch: int) -> int:
    """Index of the last candle with timestamp <= epoch."""
    return bisect_right(series.epochs, epoch) - 1


def realized_vol_15m(series: CandleSeries, end_epoch: int, lookback_windows: int = 1) -> float | None:
    """Std of 1m fractional returns over the ``lookback_windows`` * 15m before end_epoch."""
    i_end = _minute_index(series, end_epoch)
    n = 15 * lookback_windows
    if i_end - n < 1:
        return None
    closes = series.closes[i_end - n: i_end + 1]
    rets = [(closes[k] - closes[k - 1]) / closes[k - 1] for k in range(1, len(closes)) if closes[k - 1] > 0]
    if len(rets) < 5:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(1, len(rets) - 1)
    # scale 1m std to a 15m-window std (sqrt-time)
    return math.sqrt(var) * math.sqrt(15)


class MarkovVol:
    """Walk-forward Markov chain over realized-vol regimes for one asset."""

    def __init__(self, series: CandleSeries, n_regimes: int = 5, train_windows: int = 2000):
        self.series = series
        self.n = n_regimes
        self.train_windows = train_windows
        self._edges: list[float] | None = None
        self._regime_mean_vol: list[float] = []
        self._trans: list[list[float]] = []
        self._built = False

    def _build(self, anchor_epoch: int) -> None:
        """Fit regime boundaries + transition matrix on windows ending before anchor."""
        s = self.series
        # sample one realized-vol per 15m window stepping back from anchor
        vols: list[float] = []
        e = (anchor_epoch // WINDOW_SECONDS) * WINDOW_SECONDS
        for _ in range(self.train_windows):
            v = realized_vol_15m(s, e)
            if v is not None and v > 0:
                vols.append(v)
            e -= WINDOW_SECONDS
            if _minute_index(s, e) - 16 < 1:
                break
        if len(vols) < 50:
            self._built = False
            return
        vols.reverse()  # chronological
        ordered = sorted(vols)
        # quantile edges -> regimes
        self._edges = [ordered[int(len(ordered) * q / self.n)] for q in range(1, self.n)]
        regimes = [self._regime_of(v) for v in vols]
        # regime-conditional mean vol
        sums = [0.0] * self.n
        cnts = [0] * self.n
        for v, r in zip(vols, regimes):
            sums[r] += v
            cnts[r] += 1
        self._regime_mean_vol = [sums[r] / cnts[r] if cnts[r] else ordered[len(ordered) // 2] for r in range(self.n)]
        # transition counts with Laplace smoothing
        trans = [[1.0] * self.n for _ in range(self.n)]
        for a, b in zip(regimes, regimes[1:]):
            trans[a][b] += 1.0
        self._trans = [[c / sum(row) for c in row] for row in trans]
        self._built = True

    def _regime_of(self, vol: float) -> int:
        assert self._edges is not None
        return bisect_right(self._edges, vol)

    def forecast(self, window_start_epoch: int) -> VolForecast | None:
        """Forecast the sigma of the window starting at window_start_epoch."""
        if not self._built or self._edges is None:
            self._build(window_start_epoch)
        if not self._built or self._edges is None:
            return None
        cur_vol = realized_vol_15m(self.series, window_start_epoch)
        if cur_vol is None:
            return None
        r = self._regime_of(cur_vol)
        row = self._trans[r]
        # expected next-window sigma = sum_j P(r->j) * mean_vol(j)
        sigma = sum(p * mv for p, mv in zip(row, self._regime_mean_vol))
        ent = -sum(p * math.log(p) for p in row if p > 0) / math.log(self.n)
        return VolForecast(
            sigma_window=sigma,
            regime=r,
            entropy=ent,
            confidence=1.0 - ent,
            n_obs=self.train_windows,
        )
