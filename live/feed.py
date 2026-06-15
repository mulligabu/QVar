"""Real-time price feeds for the forward engine.

Chainlink is the *settlement-grade* reference: Polymarket's BTC 15m markets resolve
on the Chainlink BTC/USD stream, so pricing/settling against Chainlink removes the
~4% Binance-candle basis we measured. The on-chain Arbitrum aggregator is readable
free via public RPC (updates on ~0.05% deviation). A fast exchange tick (Coinbase)
fills the gaps between Chainlink updates for sub-second prediction.

All stdlib. Transport = live.net (keep-alive pool + health counters); browser UA
required or Cloudflare returns 403. FeedHub samples every asset in one batched
Chainlink RPC + parallel exchange ticks instead of ~3 sequential calls/asset.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

from live import net

UA = net.UA

# Chainlink on-chain USD aggregators (Arbitrum proxies), readable via eth_call.
# All verified live to return sane prices. HYPE has no Chainlink feed -> tick only.
_ARB = "https://arb1.arbitrum.io/rpc"
CHAINLINK_FEEDS = {
    "BTC": (_ARB, "0x6ce185860a4963106506C203335A2910413708e9"),
    "ETH": (_ARB, "0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612"),
    "SOL": (_ARB, "0x24ceA4b8ce57cdA5058b924B9B9987992450590c"),
    "XRP": (_ARB, "0xB4AD57B52aB9141de9926a3e0C8dc6264c2ef205"),
    "DOGE": (_ARB, "0x9A7FB1b3950837a8D9b40517626E11D4127C098C"),
    "BNB": (_ARB, "0x6970460aabF80C5BE983C6b74e5D06dEDCA95D4A"),
}
LATEST_ROUND_DATA = "0xfeaf968c"


@dataclass
class Tick:
    price: float
    ts: float          # wall-clock epoch we observed it
    source: str
    updated_at: float | None = None  # feed's own update time (Chainlink roundData)


def _get_json(url: str, data: bytes | None = None, timeout: float = 4.0):
    return net.get_json(url, data=data, timeout=timeout)


def _decode_round(res: str) -> tuple[float, float] | None:
    """Decode latestRoundData hex -> (price, updated_at); None if malformed."""
    if not res or len(res) < 2 + 64 * 5:
        return None
    words = [res[2 + 64 * i: 2 + 64 * (i + 1)] for i in range(5)]
    answer = int(words[1], 16)
    if answer >= 2 ** 255:  # int256 negative guard
        answer -= 2 ** 256
    return answer / 1e8, float(int(words[3], 16))


class ChainlinkFeed:
    """Real Chainlink price via on-chain aggregator (settlement-grade reference).

    Only configured for assets in CHAINLINK_FEEDS (BTC/ETH verified). For others
    get() returns None and the composite falls back to the exchange tick.
    """

    def __init__(self, asset: str = "BTC"):
        self.asset = asset
        self.rpc: str | None
        self.addr: str | None
        self.rpc, self.addr = CHAINLINK_FEEDS.get(asset, (None, None))

    def get(self) -> Tick | None:
        if self.addr is None or self.rpc is None:
            return None
        try:
            payload = json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "eth_call",
                "params": [{"to": self.addr, "data": LATEST_ROUND_DATA}, "latest"],
            }).encode()
            r = _get_json(self.rpc, data=payload)
            dec = _decode_round(r.get("result", ""))
            if dec is None:
                return None
            price, updated_at = dec
            return Tick(price=price, ts=time.time(), source="chainlink", updated_at=updated_at)
        except Exception:
            return None


def batch_chainlink(assets: list[str], timeout: float = 4.0) -> dict[str, Tick]:
    """All configured assets' latestRoundData in ONE JSON-RPC batch request
    (one HTTP round trip instead of one per asset). Assets without a feed, or
    whose result is malformed, are simply absent from the returned dict."""
    feeds = [(a, CHAINLINK_FEEDS[a][1]) for a in assets if a in CHAINLINK_FEEDS]
    if not feeds:
        return {}
    payload = json.dumps([
        {"jsonrpc": "2.0", "id": i, "method": "eth_call",
         "params": [{"to": addr, "data": LATEST_ROUND_DATA}, "latest"]}
        for i, (_, addr) in enumerate(feeds)
    ]).encode()
    out: dict[str, Tick] = {}
    try:
        resp = _get_json(_ARB, data=payload, timeout=timeout)
    except Exception:
        return out
    if not isinstance(resp, list):
        return out
    now = time.time()
    by_id = {r.get("id"): r for r in resp if isinstance(r, dict)}
    for i, (asset, _) in enumerate(feeds):
        dec = _decode_round((by_id.get(i) or {}).get("result", ""))
        if dec is not None:
            out[asset] = Tick(price=dec[0], ts=now, source="chainlink", updated_at=dec[1])
    return out


class CoinbaseFeed:
    """Fast spot tick to fill gaps between Chainlink deviation updates."""

    PRODUCT: ClassVar[dict[str, str]] = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
               "XRP": "XRP-USD", "DOGE": "DOGE-USD", "BNB": "BNB-USD", "HYPE": "HYPE-USD"}

    def __init__(self, asset: str = "BTC"):
        self.product = self.PRODUCT.get(asset, f"{asset}-USD")

    def get(self) -> Tick | None:
        try:
            r = _get_json(f"https://api.coinbase.com/v2/prices/{self.product}/spot")
            return Tick(price=float(r["data"]["amount"]), ts=time.time(), source="coinbase")
        except Exception:
            return None


class BinanceFeed:
    """Fast spot tick fallback (covers BNB/HYPE and others Coinbase lacks)."""

    SYMBOL: ClassVar[dict[str, str]] = {a: f"{a}USDT" for a in ("BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE")}

    def __init__(self, asset: str = "BTC"):
        self.symbol = self.SYMBOL.get(asset, f"{asset}USDT")

    def get(self) -> Tick | None:
        for host in ("https://api.binance.com", "https://api.binance.us"):
            try:
                r = _get_json(f"{host}/api/v3/ticker/price?symbol={self.symbol}")
                return Tick(price=float(r["price"]), ts=time.time(), source="binance")
            except Exception:
                continue
        return None


class CompositeFeed:
    """Chainlink as the reference truth; Coinbase for freshness between updates.

    Returns the Chainlink price when it is fresh; otherwise blends in the Coinbase
    move since the last Chainlink read so the prediction sees live drift while still
    anchoring to the settlement source.
    """

    def __init__(self, asset: str = "BTC"):
        self.chainlink = ChainlinkFeed(asset)
        self.coinbase = CoinbaseFeed(asset)
        self.binance = BinanceFeed(asset)
        self._last_cl: Tick | None = None     # raw chainlink anchor (settlement-grade)
        self._cb_at_cl: float | None = None    # tick price when the anchor was set

    def _tick(self) -> Tick | None:
        return self.coinbase.get() or self.binance.get()

    @property
    def last_chainlink(self) -> Tick | None:
        return self._last_cl

    def get(self, cl: Tick | None = None, cb: Tick | None = None,
            prefetched: bool = False) -> Tick | None:
        """Composite tick. Standalone it fetches its own chainlink + exchange
        tick; FeedHub passes both pre-fetched (`prefetched=True`, so a fetch
        miss is not re-fetched serially). Blend logic is identical either way."""
        if not prefetched:
            cl = self.chainlink.get()
            cb = self._tick()
        # Re-anchor ONLY when chainlink posts a NEW round (updated_at changes);
        # otherwise the anchor is stale and we add coinbase drift since then.
        # (Bug fixed: re-anchoring every call zeroed the drift -> frozen price.)
        if cl is not None and (self._last_cl is None or cl.updated_at != self._last_cl.updated_at):
            self._last_cl = cl
            self._cb_at_cl = cb.price if cb else None
        if self._last_cl is None:
            return cb  # bootstrap before first chainlink read
        ref = self._last_cl
        if cb is not None and self._cb_at_cl:
            drift = cb.price - self._cb_at_cl
            return Tick(price=ref.price + drift, ts=time.time(), source="chainlink+cb",
                        updated_at=ref.updated_at)
        return ref


def make_feed(asset: str = "BTC", kind: str = "composite"):
    return {"chainlink": ChainlinkFeed, "coinbase": CoinbaseFeed, "composite": CompositeFeed}[kind](asset)


class FeedHub:
    """Samples every asset's composite feed in parallel: one batched Chainlink
    RPC for all assets + concurrent exchange ticks, then feeds them into each
    CompositeFeed's (unchanged) anchor+drift blend. Replaces N_assets×(2-3)
    sequential HTTP calls per engine cycle with ~1 batch + N parallel ticks."""

    def __init__(self, assets: list[str]):
        self.assets = list(assets)
        self.feeds: dict[str, CompositeFeed] = {a: CompositeFeed(a) for a in self.assets}
        self._pool = ThreadPoolExecutor(max_workers=min(8, len(self.assets) + 1),
                                        thread_name_prefix="feedhub")

    def sample(self) -> dict[str, Tick | None]:
        cl_fut = self._pool.submit(batch_chainlink, self.assets)
        tick_futs = {a: self._pool.submit(self.feeds[a]._tick) for a in self.assets}
        try:
            cls = cl_fut.result()
        except Exception:
            cls = {}
        out: dict[str, Tick | None] = {}
        for a in self.assets:
            try:
                cb = tick_futs[a].result()
            except Exception:
                cb = None
            out[a] = self.feeds[a].get(cl=cls.get(a), cb=cb, prefetched=True)
        return out

    def staleness(self) -> dict[str, float | None]:
        """Seconds since each asset's last Chainlink ROUND (the feed's own
        updated_at) — the settlement-integrity health metric. None = no
        chainlink feed configured / never read."""
        now = time.time()
        out = {}
        for a in self.assets:
            cl = self.feeds[a].last_chainlink
            out[a] = round(now - cl.updated_at, 1) if cl is not None and cl.updated_at else None
        return out
