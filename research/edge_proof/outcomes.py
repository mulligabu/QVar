"""Outcome labeling for BTC-style 15m up/down prediction markets.

The engine resolves each window from local Binance 1m candles by comparing the
underlying price at ``market_start_time`` to the price at ``market_end_time``.
We replicate that here so we can attach a realized outcome to *every* decision
(entered or skipped), not just the trades that were actually opened. This avoids
the selection bias of only looking at settled trades.

The labeler is cross-validated against the engine's own ``settlement_winning_side``
field in settled-trade records (see ``validate_against_settled``); if agreement is
high, the candle convention is correct and can be trusted for skipped markets.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Asset -> Binance symbol. The clean per-symbol files live in
# data/candles/binance/<SYMBOL>_1m.csv.
ASSET_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "BNB": "BNBUSDT",
    "HYPE": "HYPEUSDT",
}


def parse_ts(value: str) -> datetime:
    """Parse either ISO ('2026-05-31T03:45:00Z') or candle ('2026-05-31 03:45:00')."""
    text = value.strip().replace("Z", "+00:00")
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class CandleSeries:
    """Sorted 1m OHLC series for one asset, queryable by causal 'price at instant T'.

    Binance kline timestamp = OPEN time; candle@m covers [m, m+1) and its OPEN is
    the price at instant m (verified: engine settlement start_price == open of the
    start-minute candle). So price_at(T) returns the OPEN of the candle covering T.
    This is causal (open@m is known at m <= T) and avoids the 1-minute look-ahead
    that using the close would introduce near expiry.
    """

    epochs: list[int]
    opens: list[float]
    closes: list[float]

    def price_at(self, dt: datetime) -> float | None:
        """Causal price at instant ``dt`` = open of the candle covering dt."""
        target = int(dt.timestamp())
        idx = bisect_right(self.epochs, target) - 1
        if idx < 0:
            return None
        # Guard against huge gaps (missing data): don't reach back more than ~30 min.
        if target - self.epochs[idx] > 30 * 60:
            return None
        return self.opens[idx]


def load_candles(symbol: str, candles_dir: Path, since: datetime | None = None) -> CandleSeries:
    return _load_path(candles_dir / f"{symbol}_1m.csv", since)


def _load_path(path: Path, since: datetime | None = None) -> CandleSeries:
    epochs: list[int] = []
    opens: list[float] = []
    closes: list[float] = []
    since_epoch = int(since.timestamp()) if since else None
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header: timestamp,open,high,low,close,volume
        for row in reader:
            if not row:
                continue
            ts = parse_ts(row[0])
            epoch = int(ts.timestamp())
            if since_epoch is not None and epoch < since_epoch:
                continue
            epochs.append(epoch)
            opens.append(float(row[1]))
            closes.append(float(row[4]))
    return CandleSeries(epochs, opens, closes)


class OutcomeResolver:
    """Resolves 15m windows to 'up'/'down' using per-asset 1m candles.

    Convention (matches the engine): up wins iff price@end > price@start; ties go
    to 'down' (settlement_tie_side observed as 'down' in settled records).

    Two candle layouts are supported:
      * multiasset (default, what the live engine writes & keeps fresh):
        ``<candles_dir>/<ASSET>/BTCUSDT_1m.csv`` -- the file is always named
        BTCUSDT_1m.csv but contains the folder's asset data.
      * per-symbol: ``<candles_dir>/<SYMBOL>_1m.csv`` (e.g. ETHUSDT_1m.csv).
        These were stale in the snapshot we inspected; prefer multiasset.
    """

    def __init__(self, candles_dir: Path, since: datetime | None = None, layout: str = "multiasset"):
        self.candles_dir = Path(candles_dir)
        self.since = since
        self.layout = layout
        self._series: dict[str, CandleSeries | None] = {}

    def _path_for(self, asset: str) -> Path | None:
        if self.layout == "multiasset":
            return self.candles_dir / asset / "BTCUSDT_1m.csv"
        symbol = ASSET_SYMBOL.get(asset)
        return self.candles_dir / f"{symbol}_1m.csv" if symbol else None

    def _get(self, asset: str) -> CandleSeries | None:
        if asset not in self._series:
            path = self._path_for(asset)
            if path and path.exists():
                self._series[asset] = _load_path(path, self.since)
            else:
                self._series[asset] = None
        return self._series[asset]

    def resolve(self, asset: str, start: datetime, end: datetime) -> tuple[str, float] | None:
        """Return (winning_side, return_fraction) or None if unresolvable."""
        series = self._get(asset)
        if series is None:
            return None
        p0 = series.price_at(start)
        p1 = series.price_at(end)
        if p0 is None or p1 is None or p0 <= 0:
            return None
        ret = (p1 - p0) / p0
        side = "up" if p1 > p0 else "down"
        return side, ret


def validate_against_settled(resolver: OutcomeResolver, settled_rows: list[dict]) -> dict:
    """Cross-check candle-derived labels vs the engine's settlement_winning_side."""
    agree = total = 0
    mismatches: list[dict] = []
    for r in settled_rows:
        eng = r.get("settlement_winning_side")
        if eng not in ("up", "down"):
            continue
        start, end = r.get("market_start_time"), r.get("market_end_time")
        if not start or not end:
            continue
        got = resolver.resolve(r["asset"], parse_ts(start), parse_ts(end))
        if got is None:
            continue
        total += 1
        if got[0] == eng:
            agree += 1
        elif len(mismatches) < 10:
            mismatches.append({"asset": r["asset"], "start": start, "engine": eng, "ours": got[0]})
    return {
        "checked": total,
        "agreement": agree / total if total else None,
        "mismatches_sample": mismatches,
    }
