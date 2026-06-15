"""Shared market/quote foundation for the structural-edge tests.

Quote snapshots only ever show ONE side's book (``selected_side``). We reconstruct
the UP-token prices via binary complementarity (up_ask = 1 - down_best_bid,
up_bid = 1 - down_best_ask) so every snapshot yields a comparable implied P(up).
Windows are keyed by market_slug where present (Polymarket) and market_id
otherwise (Kalshi lumps everything under one slug).

All three tests (bridge / feed-basis / cross-venue) consume ``iter_quote_windows``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from research.edge_proof.loaders import ASSETS
from research.edge_proof.outcomes import CandleSeries, OutcomeResolver, parse_ts


@dataclass
class Snap:
    venue: str
    ts: float            # epoch seconds of quote_timestamp
    up_bid: float | None  # reconstructed best bid for the UP token
    up_ask: float | None  # reconstructed best ask for the UP token
    bid_size: float
    ask_size: float

    @property
    def up_mid(self) -> float | None:
        if self.up_bid is not None and self.up_ask is not None:
            return 0.5 * (self.up_bid + self.up_ask)
        # one-sided book: fall back to the single side we have
        return self.up_ask if self.up_ask is not None else self.up_bid

    @property
    def spread(self) -> float | None:
        if self.up_bid is not None and self.up_ask is not None:
            return self.up_ask - self.up_bid
        return None


@dataclass
class Window:
    asset: str
    key: str
    start: float          # epoch seconds market_start_time
    end: float            # epoch seconds market_end_time
    snaps_by_venue: dict[str, list[Snap]] = field(default_factory=dict)
    outcome: str | None = None       # 'up'/'down' from candles
    ret: float | None = None
    strike: float | None = None      # underlying at window start


def _orient_up(side: str | None, bid: float | None, ask: float | None) -> tuple[float | None, float | None]:
    """Return (up_bid, up_ask) given a one-sided quote for ``side``."""
    if side == "down":
        # up_ask = 1 - down_bid ; up_bid = 1 - down_ask
        up_bid = None if ask is None else 1.0 - ask
        up_ask = None if bid is None else 1.0 - bid
        return up_bid, up_ask
    return bid, ask  # treat unknown/up as up-oriented


def iter_quote_windows(
    runs: list[Path],
    resolver: OutcomeResolver,
    assets: list[str] | None = None,
) -> Iterator[Window]:
    """Yield one Window per (asset, market) with all venues' snapshots attached.

    Windows are grouped per (run, asset) to bound memory — one trading day of
    quote files at a time.
    """
    assets = assets or ASSETS
    for run in runs:
        for asset in assets:
            qdir = run / asset / "quotes"
            if not qdir.is_dir():
                continue
            windows: dict[str, Window] = {}
            for path in qdir.glob("*quotes_*.jsonl"):
                with path.open() as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if d.get("record_type") != "quote_snapshot":
                            continue
                        st, et, qt = d.get("market_start_time"), d.get("market_end_time"), d.get("quote_timestamp")
                        if not st or not et or not qt:
                            continue
                        start_ep, end_ep = parse_ts(st).timestamp(), parse_ts(et).timestamp()
                        if abs((end_ep - start_ep) - 900) > 1:   # 15m windows only
                            continue
                        # key by TIME window so Poly and Kalshi for the same 15m bin join
                        key = f"{asset}|{st}|{et}"
                        w = windows.get(key)
                        if w is None:
                            w = Window(asset=asset, key=key, start=start_ep, end=end_ep)
                            windows[key] = w
                        up_bid, up_ask = _orient_up(
                            d.get("selected_side"), d.get("observed_best_bid"), d.get("observed_best_ask")
                        )
                        snap = Snap(
                            venue=d.get("venue") or "unknown",
                            ts=parse_ts(qt).timestamp(),
                            up_bid=up_bid,
                            up_ask=up_ask,
                            bid_size=float(d.get("observed_bid_size") or 0.0),
                            ask_size=float(d.get("observed_ask_size") or 0.0),
                        )
                        w.snaps_by_venue.setdefault(snap.venue, []).append(snap)
            # finalize: attach outcome + strike, then yield
            for w in windows.values():
                from datetime import datetime, timezone

                start_dt = datetime.fromtimestamp(w.start, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(w.end, tz=timezone.utc)
                out = resolver.resolve(w.asset, start_dt, end_dt)
                if out is not None:
                    w.outcome, w.ret = out
                series = resolver._get(w.asset)
                if series is not None:
                    w.strike = series.price_at(start_dt)
                for snaps in w.snaps_by_venue.values():
                    snaps.sort(key=lambda s: s.ts)
                yield w


# --- volatility ------------------------------------------------------------
def per_minute_vol(series: CandleSeries, lookback: int = 720) -> float:
    """Std of 1m simple returns over the most recent ``lookback`` candles."""
    closes = series.closes[-lookback:]
    if len(closes) < 30:
        return 0.0
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not rets:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(1, len(rets) - 1)
    return math.sqrt(var)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bridge_prob_up(spot: float, strike: float, minutes_remaining: float, vol_per_min: float) -> float:
    """Driftless digital-call probability that price ends > strike.

    P(end > strike | spot_now) = Phi( ln(spot/strike) / (vol_per_min * sqrt(T)) ).
    """
    if strike <= 0 or spot <= 0 or vol_per_min <= 0 or minutes_remaining <= 0:
        return 0.5
    z = math.log(spot / strike) / (vol_per_min * math.sqrt(minutes_remaining))
    return norm_cdf(z)
