"""Live venue clients: discover the current BTC 15m market + its order book.

Polymarket: series btc-up-or-down-15m, slug btc-updown-15m-{epoch}. Settles on
Chainlink. CLOB /book gives bids/asks per token.
Kalshi: series KXBTC15M, "BTC price up in next 15 mins?". Settles on CF Benchmarks
BRTI 60s avg. Orderbook returns YES bids and NO bids; YES ask = 1 - best NO bid.

We normalize both to an UP-token view: (up_bid, up_ask, sizes, window times).
Stdlib only; browser UA (Cloudflare blocks default urllib UA with 403).
"""

from __future__ import annotations

import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC
from typing import ClassVar

from live import net

UA = net.UA

# discovery fan-out pool: per-(asset,timeframe) lookups are independent — run
# them concurrently instead of ~40 sequential round trips per discovery sweep
_DISC_POOL = ThreadPoolExecutor(max_workers=10, thread_name_prefix="discover")


def _get(url: str, timeout: float = 4.0):
    return net.get_json(url, timeout=timeout)


@dataclass
class BookTop:
    venue: str
    market_id: str
    start_ts: float
    end_ts: float
    up_bid: float | None
    up_ask: float | None
    up_bid_size: float
    up_ask_size: float
    observed_at: float

    @property
    def up_mid(self):
        if self.up_bid is not None and self.up_ask is not None:
            return 0.5 * (self.up_bid + self.up_ask)
        return self.up_ask if self.up_ask is not None else self.up_bid


# ---------------- Kalshi ----------------
class KalshiClient:
    HOST = "https://api.elections.kalshi.com/trade-api/v2"
    SERIES_15M: ClassVar[dict[str, str]] = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M"}

    def discover(self, assets) -> list[dict]:
        def one(asset):
            series = self.SERIES_15M.get(asset)
            if not series:
                return None
            try:
                r = _get(f"{self.HOST}/markets?series_ticker={series}&status=open&limit=10")
            except Exception:
                return None
            mk = r.get("markets", [])
            if not mk:
                return None
            mk.sort(key=lambda m: m.get("close_time", ""))
            m = dict(mk[0]); m["_asset"] = asset; m["_tf"] = "15m"; m["_window_s"] = 900
            return m

        return [m for m in _DISC_POOL.map(one, assets) if m is not None]

    # threshold ladders: hourly (KX{A}) + daily (KX{A}D) -> "price >= strike at T"
    THRESHOLD_SERIES: ClassVar[dict[str, str]] = {"1h": "KX{A}", "1D": "KX{A}D"}

    def discover_threshold(self, assets, timeframes=("1h", "1D")) -> list[dict]:
        import re

        def one(pair):
            asset, tf = pair
            st = self.THRESHOLD_SERIES[tf].format(A=asset)
            try:
                r = _get(f"{self.HOST}/markets?series_ticker={st}&status=open&limit=60")
            except Exception:
                return []
            got = []
            for m in r.get("markets", []):
                tk = m.get("ticker", "")
                mm = re.search(r"-T([0-9.]+)$", tk)
                if not mm or m.get("market_type") not in (None, "binary"):
                    continue
                m = dict(m); m["_asset"] = asset; m["_tf"] = tf
                m["_strike"] = float(mm.group(1)); m["_kind"] = "threshold"
                got.append(m)
            return got

        pairs = [(a, tf) for a in assets for tf in timeframes]
        return [m for got in _DISC_POOL.map(one, pairs) for m in got]

    def current_market(self) -> dict | None:  # back-compat (BTC)
        ms = self.discover(["BTC"])
        return ms[0] if ms else None

    def book_top(self, market: dict) -> BookTop | None:
        ticker = str(market.get("ticker") or "")
        try:
            r = _get(f"{self.HOST}/markets/{ticker}/orderbook?depth=2")
        except Exception:
            return None
        # actual shape: {"orderbook_fp": {"yes_dollars": [["0.9120","119.78"],...],
        #                                 "no_dollars":  [["0.0820","1000"],...]}}
        # prices are DOLLARS (0-1) strings; yes_dollars = YES bids, no_dollars = NO bids.
        ob = r.get("orderbook_fp") or r.get("orderbook") or {}
        yes = ob.get("yes_dollars") or ob.get("yes") or []
        no = ob.get("no_dollars") or ob.get("no") or []

        def best(levels):  # highest-priced bid (top of book)
            bids = [(float(p), float(s)) for p, s in levels]
            return max(bids, key=lambda x: x[0]) if bids else (None, 0.0)

        up_bid, up_bid_size = best(yes)                 # best YES bid = up bid
        no_bid, no_bid_size = best(no)                  # best NO bid
        up_ask = (1.0 - no_bid) if no_bid is not None else None   # up ask = 1 - best no bid
        return BookTop(
            venue="kalshi", market_id=ticker,
            start_ts=_iso(market.get("open_time")), end_ts=_iso(market.get("close_time")),
            up_bid=up_bid, up_ask=up_ask,
            up_bid_size=float(up_bid_size), up_ask_size=float(no_bid_size),
            observed_at=time.time(),
        )


# ---------------- Polymarket ----------------
class PolymarketClient:
    GAMMA = "https://gamma-api.polymarket.com"
    CLOB = "https://clob.polymarket.com"
    ASSET_SLUG: ClassVar[dict[str, str]] = {"BTC": "btc", "ETH": "eth", "SOL": "sol", "XRP": "xrp",
                  "DOGE": "doge", "BNB": "bnb", "HYPE": "hype"}
    TF_SECONDS: ClassVar[dict[str, int]] = {"5m": 300, "15m": 900, "4h": 14400}
    # hourly threshold ladder: series "{name}-multi-strikes-hourly", ~20 Yes/No
    # strikes ($200 apart) densely centered on spot. Only BTC/ETH exist.
    MULTI_STRIKE_HOURLY: ClassVar[dict[str, str]] = {"BTC": "bitcoin", "ETH": "ethereum"}

    def discover(self, assets, timeframes=("15m", "5m", "4h")) -> list[dict]:
        """Current up/down market per (asset, timeframe). Slug = {a}-updown-{tf}-{epoch}."""
        now = int(time.time())

        def one(pair):
            asset, tf = pair
            slug_a = self.ASSET_SLUG.get(asset)
            if not slug_a:
                return None
            step = self.TF_SECONDS[tf]
            start = now - (now % step)
            for s in (start, start + step):  # current, then next (current may be expiring)
                try:
                    ev = _get(f"{self.GAMMA}/events/slug/{slug_a}-updown-{tf}-{s}")
                except Exception:
                    continue
                mkts = ev.get("markets") if isinstance(ev, dict) else None
                if mkts:
                    m = dict(mkts[0])
                    m["_event_start"] = s
                    m["_asset"] = asset
                    m["_tf"] = tf
                    m["_window_s"] = step
                    return m
            return None

        pairs = [(a, tf) for a in assets for tf in timeframes]
        return [m for m in _DISC_POOL.map(one, pairs) if m is not None]

    def discover_hourly(self, assets) -> list[dict]:
        """Polymarket hourly threshold ladder (BTC/ETH only). 'X above {strike} on
        {date} {hour} ET?' Yes/No markets, ~20 strikes densely centered on spot —
        the real 1h source (Kalshi's hourly KX{A} ladder is sparse + off-spot)."""
        import re
        out = []
        now = time.time()

        def fetch(name):
            try:
                return _get(f"{self.GAMMA}/events?" + urllib.parse.urlencode(
                    {"series_slug": f"{name}-multi-strikes-hourly", "closed": "false",
                     "limit": 50, "order": "endDate", "ascending": "true"}))
            except Exception:
                return None

        names = [(a, self.MULTI_STRIKE_HOURLY[a]) for a in assets if a in self.MULTI_STRIKE_HOURLY]
        fetched = dict(zip((a for a, _ in names),
                           _DISC_POOL.map(fetch, (n for _, n in names)), strict=True))
        for asset, _name in names:
            evs = fetched.get(asset)
            if evs is None:
                continue
            if isinstance(evs, dict):
                evs = evs.get("data") or []
            # the two nearest future-expiry hours (avoids a gap at the roll boundary)
            fut = sorted(((e, _iso(e.get("endDate"))) for e in evs if e.get("endDate")),
                         key=lambda x: x[1])
            fut = [(e, t) for e, t in fut if t > now][:2]
            for ev, end in fut:
                start = _iso(ev.get("startDate")) or (end - 3600)
                for m in ev.get("markets") or []:
                    if m.get("closed"):
                        continue
                    mm = re.search(r"above-([0-9.]+)-on", m.get("slug", "") or "")
                    if not mm:
                        continue
                    m = dict(m)
                    m["_asset"] = asset; m["_tf"] = "1h"; m["_window_s"] = max(60.0, end - start)
                    m["_event_start"] = start; m["_strike"] = float(mm.group(1)); m["_kind"] = "threshold"
                    out.append(m)
        return out

    def current_market(self) -> dict | None:  # back-compat (BTC 15m)
        ms = self.discover(["BTC"], ("15m",))
        return ms[0] if ms else None

    def book_top(self, market: dict) -> BookTop | None:
        try:
            token_ids = json.loads(market.get("clobTokenIds") or "[]")
        except Exception:
            token_ids = market.get("clobTokenIds") or []
        if not token_ids:
            return None
        up_token = str(token_ids[0])  # outcome[0] = "Up"
        try:
            book = _get(f"{self.CLOB}/book?{urllib.parse.urlencode({'token_id': up_token})}")
        except Exception:
            return None
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        up_bid = max((float(b["price"]) for b in bids), default=None)
        up_ask = min((float(a["price"]) for a in asks), default=None)
        bsz = next((float(b["size"]) for b in bids if float(b["price"]) == up_bid), 0) if up_bid is not None else 0
        asz = next((float(a["size"]) for a in asks if float(a["price"]) == up_ask), 0) if up_ask is not None else 0
        ws = float(market.get("_window_s") or 900)
        start = market.get("_event_start", int(time.time()) // int(ws) * int(ws))
        return BookTop(
            venue="polymarket", market_id=str(market.get("id")),
            start_ts=float(start), end_ts=float(start + ws),
            up_bid=up_bid, up_ask=up_ask, up_bid_size=bsz, up_ask_size=asz,
            observed_at=time.time(),
        )


def _iso(s) -> float:
    if not s:
        return 0.0
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(UTC).timestamp()
    except Exception:
        return 0.0
