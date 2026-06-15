"""Forward-running prediction engine (NOT a backtest).

Loop, live:
  1. sample the Chainlink-anchored feed -> per-asset rolling price history
  2. record the underlying price at each expiry-window boundary (strike for
     directional up/down markets); threshold markets carry their own fixed strike
  3. discover all feasible markets across venues / assets / expiries
  4. near expiry, build causal features, run the online RenTech predictor
  5. EV-gate after costs; ghost a LIMIT order at the book mid on the predicted side
  6. at expiry, settle on the Chainlink price, mark the ghost fill, and let the
     predictor learn online (Bayesian/SGD/calibration update)

Generic over market kind:
  - directional: strike = underlying price at window open (observed forward)
  - threshold  : strike = fixed level from market metadata (tradeable immediately)

Stdlib only. Persists to data/forward_live/. Run:
  python -m live.forward_engine --assets BTC,ETH --minutes 30
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from live.md_client import MDFeedHub, MDMirror

from live import net
from live.control import lanes_snapshot, load_control, read_trip, save_control, trip
from live.db import Store
from live.feed import FeedHub
from live.predictor import OnlinePredictor, Prediction, build_features
from live.structural import AnchorConfig, StructuralHead, structural_anchor
from live.venues import KalshiClient, PolymarketClient, _iso

OUT = Path("data/forward_live")
# Control plane (live/control.py): data/CONTROL.json + data/TRIPPED.json +
# legacy data/HALT, re-read EVERY cycle so dashboard toggles take effect on the
# next cycle. Two kill switches: estop (freeze everything — for bugs/breakage)
# and halt_entries (block NEW entries; settling/learning continue), plus the
# wired max-loss breaker (latches TRIPPED.json) and the cross-lane exposure cap.
TAKER_FEE_RATE = 0.072  # polymarket crypto; kalshi similar

# canonical short->seconds for ordering timeframe lanes in the dashboard
TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1D": 86400}


def tf_sort_key(tf: str) -> float:
    return TF_SECONDS.get(tf, 1e12)


@dataclass
class Market:
    venue: str
    market_id: str
    asset: str
    kind: str           # 'directional' | 'threshold'
    tf: str             # '5m','15m','1h','1D'
    window_s: float
    start_ts: float
    end_ts: float
    strike: float | None  # threshold: known now; directional: filled at open
    raw: dict = field(default_factory=dict)
    ticker: str = ""    # human-readable slug/ticker

    @property
    def key(self):
        return f"{self.venue}:{self.market_id}"


@dataclass
class GhostPosition:
    market_key: str
    asset: str
    side: str           # 'up' | 'down'
    price: float        # limit price we rested at (mid at decision)
    size: float
    p_model: float
    regime: str
    placed_at: float
    end_ts: float
    strike: float
    pred: dict
    ref_up_mid: float            # up-mid observed at decision (for fill test)
    market: Market             # to re-fetch the book while resting
    status: str = "resting"      # 'resting' -> 'filled' (PASSIVE leg only)
    fill_ts: float | None = None
    # conviction-scaled AGGRESSIVE leg: filled immediately at decision by crossing the
    # spread (paid `agg_price`, cost `agg_fee_per`/contract). `size` above is the
    # PASSIVE resting leg. agg_size=0 => pure-passive trade (today's behaviour).
    agg_size: float = 0.0
    agg_price: float = 0.0
    agg_fee_per: float = 0.0
    # ghost take-profit overlay: set True once our-side bid touches tp_level while the
    # position is open (a resting maker sell at tp_level would have been lifted).
    tp_hit: bool = False
    # reversal excursion + PIVOT (our-side sellable price), tracked while entered:
    #   peak_bid       = highest sellable price reached (the turning point for reversers)
    #   peak_mins_left = mins-to-settle WHEN that peak occurred (when the pivot happened)
    #   post_peak_low  = lowest price AFTER the peak (reversal depth = peak_bid−post_peak_low)
    #   trough_bid     = global min; last_bid = current (live reversal-in-progress view)
    peak_bid: float = 0.0
    peak_mins_left: float | None = None
    post_peak_low: float | None = None
    trough_bid: float | None = None
    last_bid: float | None = None
    # full state captured AT the pivot (the moment peak_bid was set): underlying spot,
    # bid-ask spread (liquidity at the turn), and book sizes. z/gamma/OFI derive offline.
    peak_spot: float | None = None
    peak_spread: float | None = None
    peak_bid_size: float | None = None
    peak_ask_size: float | None = None

    def fill_check(self, up_bid, up_ask) -> bool:
        """Honest fill: a resting limit fills only when the book trades THROUGH it.
        Buy UP@mid fills when ask drops to/below our price; buy DOWN@mid (= sell UP
        @ ref_up_mid) fills when up_bid rises to/above ref_up_mid. This makes fills
        adverse-selected (we get filled when price comes to us)."""
        if self.side == "up":
            return up_ask is not None and up_ask <= self.price + 1e-9
        return up_bid is not None and up_bid >= self.ref_up_mid - 1e-9


def taker_fee(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return TAKER_FEE_RATE * p * (1 - p)


def venue_taker_fee(venue: str, p: float) -> float:
    """Per-contract cost of CROSSING the spread, by venue. Polymarket charges no
    trading fee (only the half-spread is paid); Kalshi charges ~0.072*p*(1-p),
    which vanishes near p->0/1 (deep favorites cross almost free). 86% of our
    traded markets are Polymarket. See memory unfilled-winners-capture."""
    return 0.0 if venue == "polymarket" else taker_fee(p)


def kelly_fraction(p_win: float, price: float, cap: float = 0.02, frac: float = 0.5) -> float:
    """Fractional-Kelly fraction of bankroll for a binary at `price` with win prob
    `p_win`, capped. `frac`=0.5 is half-Kelly (default); candle engine uses 0.25
    (quarter-Kelly). Sizes stake to the bankroll instead of a flat contract count."""
    if not (0.0 < price < 1.0):
        return 0.0
    b = (1.0 - price) / price
    f = (p_win * (b + 1.0) - 1.0) / b   # full Kelly
    return max(0.0, min(cap, frac * f))   # fractional-Kelly, capped


class ForwardEngine:
    def __init__(self, assets: list[str], min_ev=0.01, venues=("polymarket", "kalshi"),
                 poly_tfs=("15m", "5m", "4h"), thresh_tfs=("1h", "1D"),
                 data_dir="data/forward_live", price_lo=0.30, price_hi=0.65,
                 late_frac=0.45, use_anchor=True, name="main",
                 early_frac=0.05, risk_cap=0.02, agg_risk_cap=0.01,
                 kelly_frac=0.5, tilt_cap=0.4, tp_level=0.90,
                 flat_size=True, zdist_gate=True, conditional_gate=False,
                 sides=("up", "down"), md_mode=False):
        self.assets = assets
        self.poly_tfs = poly_tfs
        self.thresh_tfs = thresh_tfs
        self.out = Path(data_dir)
        self.price_lo, self.price_hi = price_lo, price_hi   # price-discipline band
        self.late_frac = late_frac    # trade window = last `late_frac` fraction of the market
        # entry time-band = mins_left in [early_frac*window, late_frac*window]. early_frac
        # raised (candle engine: 0.12) gets us in EARLIER (more time left -> near-money
        # price ~0.50) instead of main's deep near-expiry point. See memory candle-engine.
        self.early_frac = early_frac
        self.kelly_frac = kelly_frac   # 0.5 half-Kelly (main); 0.25 quarter (candle)
        self.use_anchor = use_anchor  # True => gate on the structural-anchored prob (Phase 2)
        self.name = name
        # FeedHub: one batched Chainlink RPC + parallel exchange ticks per cycle
        # (was ~3 sequential TLS round trips per asset). self.feeds kept as the
        # per-asset CompositeFeed map for everything downstream.
        # md_mode (--md on, Brief 1 rollout): consume the shared md_service
        # stream (WS-first books/ticks/strikes/discovery over data/md.sock)
        # through interface-identical adapters that AUTOMATICALLY fall back to
        # the direct REST paths below whenever the stream is stale or dead —
        # killing md_service can never stop a lane for more than one cycle.
        # Default OFF: transport plumbing only, zero behavior change.
        self.md: MDMirror | None = None
        self.hub: FeedHub | MDFeedHub
        if md_mode:
            from live.md_client import MDFeedHub as _Hub
            from live.md_client import MDMirror as _Mirror
            self.md = _Mirror(Path(data_dir).parent / "md.sock")
            self.hub = _Hub(assets, self.md)
        else:
            self.hub = FeedHub(assets)
        self.feeds = self.hub.feeds
        self._md_books = 0          # books served from the mirror
        self._md_book_miss = 0      # mirror miss -> direct REST fallback
        self._md_strike_diff: set = set()
        # book-poll fan-out: open positions + candidate markets fetch their books
        # concurrently; results are consumed sequentially on the main thread.
        self._io_pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="books")
        # cycle latency + halt observability (status.json "health" section)
        self._cycle_ms: deque = deque(maxlen=100)
        # control plane: shared switchboard lives in the lanes' common data/
        # root (self.out.parent), so all engines + the dashboard see one state
        self.ctrl_root = self.out.parent
        self.halt_file = self.ctrl_root / "HALT"   # legacy soft-halt file
        self.control = load_control(self.ctrl_root)
        self.tripped: dict | None = read_trip(self.ctrl_root)
        self._lanes: dict | None = None   # other lanes' snapshot (refreshed per cycle)
        self._halted = False
        self._block_reason: str | None = None
        self.hist: dict[str, deque] = {a: deque(maxlen=3000) for a in assets}     # composite (prediction)
        self.cl_hist: dict[str, deque] = {a: deque(maxlen=3000) for a in assets}  # raw chainlink (settlement)
        self.strikes: dict[tuple[str, int], float] = {}   # (asset, window_start) -> chainlink price
        self.predictor = OnlinePredictor(tilt_cap=tilt_cap)
        # structural digital-probability anchor (SHADOW mode: logged, not yet gating).
        # fed the same composite ticks; see research/boltzmann_upgrade.md §5.4.
        self.struct: dict[str, StructuralHead] = {a: StructuralHead() for a in assets}
        self.anchor_cfg = AnchorConfig()
        self.poly = PolymarketClient()
        self.kalshi = KalshiClient()
        self.venues = venues
        self.min_ev = min_ev
        self.starting_bankroll = 1000.0
        # Execution = passive resting limit leg @ mid + a conviction-scaled aggressive taker
        # leg that only crosses when the edge clears the cross cost (spread/2 + venue fee).
        self.risk_cap = risk_cap  # PASSIVE limit leg: max fraction of equity (half-Kelly)
        # DECOUPLED per-leg execution sizing (2026-06-03, user). A resting limit at mid
        # is adverse-selected (fills on the losing tail, misses winners that run away);
        # an aggressive taker leg crosses immediately to participate. Each leg has its
        # OWN risk budget rather than splitting one stake: the passive limit gets the
        # larger 2% (risk_cap), the aggressive taker gets 1% (agg_risk_cap), and the
        # taker leg only fires when edge=|p_cal-mid| clears the cross cost
        # (spread/2 + venue_taker_fee) so we never pay to cross a thin edge. Kelly's own
        # edge-scaling within each cap does the conviction-scaling. Net risk up to ~3%.
        # See memory unfilled-winners-capture / deferred-engine-fixes.
        self.agg_risk_cap = agg_risk_cap  # AGGRESSIVE taker leg: max fraction of equity (half-Kelly)
        self.tilt_cap = tilt_cap
        # 2026-06-05 validated tweaks (see memory main-engine-live-postmortem / zdist-directional-edge):
        # flat_size: Kelly edge-scaling oversized LOSERS (corr(won,total_size)=-0.33); |p_cal-mid|
        #   doesn't predict realized PnL, so flat fraction-of-equity dominates Kelly (+$707 main subset,
        #   replicates 3/3). zdist_gate: never trade against sign(z_dist) -- raw z_dist beats the market
        #   mid (AUC 0.63 main / 0.74 settle OOS); agreeing trades +$125 vs fighting -$727 on main.
        self.flat_size = flat_size
        self.zdist_gate = zdist_gate
        # conditional_gate: trade the disagreement signal ONLY in (tf,mid-band) cells that
        # have earned drift-controlled skill, flipping the side where the cell says the
        # raw tilt is backwards. Side+EV come from the cell, not raw p_cal−mid. Main runs
        # this OFF (control); the 4th/settle/candle engines run it ON. See predictor.ConditionalSkill.
        self.conditional_gate = conditional_gate
        # sides: which FINAL sides (after any conditional flip) the engine may trade.
        # USER POLICY 2026-06-11: regime-agnostic engines only — every live lane runs
        # both sides (engines.conf pins "up,down"); skew problems are addressed with
        # side-neutral conditioning (cell blacklist, drift agreement), not side locks.
        # The flag exists for research/counterfactual runs.
        self.sides = tuple(sides)
        self.open_pos: dict[str, GhostPosition] = {}
        self.shadow: dict[str, dict] = {}   # all evaluated markets -> online learning at settlement
        self.decided: set[str] = set()
        self.settled = 0          # filled trades that reached settlement
        self.pnl = 0.0
        self.wins = 0
        self.fills = 0            # resting orders that got filled
        self.no_fills = 0         # resting orders that expired unfilled
        self.recon_checked = 0    # settlements reconciled vs venue-reported outcome
        self.recon_mismatch = 0
        self.out.mkdir(parents=True, exist_ok=True)
        # CONFIG-DRIFT GUARD: lane-defining flags must live in live/engines.conf (the single
        # source of truth), never in ad-hoc launch commands. This guard persists the lane
        # config next to the DB and SCREAMS on relaunch if it changed, so a silent config
        # revert (e.g. a boot that drops a flag) can never pass unnoticed.
        lane_cfg = {"price_lo": price_lo, "price_hi": price_hi,
                    "conditional_gate": conditional_gate, "zdist_gate": zdist_gate,
                    "sides": sorted(self.sides), "late_frac": late_frac,
                    "early_frac": early_frac, "risk_cap": risk_cap}
        cfg_path = self.out / "launch_args.json"
        if cfg_path.exists():
            try:
                prev = json.loads(cfg_path.read_text())
                if prev != lane_cfg:
                    diff = {k: (prev.get(k), lane_cfg[k]) for k in lane_cfg if prev.get(k) != lane_cfg[k]}
                    print("#" * 78)
                    print(f"# !!! CONFIG DRIFT [{name}]: lane flags CHANGED vs last launch !!!")
                    for k, (old, new) in diff.items():
                        print(f"# !!!   {k}: {old} -> {new}")
                    print("# !!! intended? update live/engines.conf; accidental? fix the launch")
                    print("#" * 78)
            except Exception:
                pass
        cfg_path.write_text(json.dumps(lane_cfg, indent=1, default=float))
        self.store = Store(self.out / "forward_engine.db")   # one DB for decisions/settlements/paths
        # resume running totals from the DB so equity is continuous across restarts
        t = self.store.settlement_totals()
        self.pnl, self.wins, self.settled, self.no_fills = t["pnl"], t["wins"], t["settled"], t["no_fills"]
        self.fills = t["settled"]
        if self.settled or self.no_fills:
            print(f"# resumed ledger: pnl={self.pnl:+.1f} settled={self.settled} fills={self.fills} no_fills={self.no_fills}")
        # --- ghost take-profit overlay (shadow; never affects real pnl) ---
        # On a filled position, if our-side bid touches tp_level before expiry, the ghost
        # sells the whole position there (resting maker exit) instead of holding to settle.
        # tp_cum is a running total seeded at the real cum_pnl at first launch so the ghost
        # equity ($1000+tp_cum) starts EQUAL to the engine's current equity and only
        # diverges on exits. See memory take-profit-result.
        self.tp_level = tp_level
        self.tp_cum = 0.0
        self.tp_settled = self.tp_exits = self.tp_wins = 0
        if self.tp_level > 0:
            g = self.store.tp_totals()
            self.tp_settled, self.tp_exits, self.tp_wins = g["settled"], g["exits"], g["wins"]
            self.tp_cum = g["cum"] if g["cum"] is not None else self.pnl   # seed = real equity seam
            print(f"# ghost take-profit overlay ON @ {self.tp_level:.2f} | "
                  f"tp_equity={self.starting_bankroll + self.tp_cum:+.1f} "
                  f"exits={self.tp_exits} settled={self.tp_settled}")
        self.status_path = self.out / "status.json"
        self.state_path = self.out / "model_state.json"
        self.recent_decisions: deque = deque(maxlen=40)
        # resume online learning across restarts/crashes (durable 48h run)
        if self.state_path.exists():
            try:
                self.predictor.load_state(json.loads(self.state_path.read_text()))
                print(f"# resumed model state: {self.predictor.n_updates} prior updates")
            except Exception as e:
                print(f"# model state load failed ({e}); cold start")

    def save_model(self):
        try:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.predictor.state(), default=float))
            tmp.replace(self.state_path)
        except Exception:
            pass

    def _log(self, path: Path, rec: dict):
        with path.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def sample_feeds(self):
        ticks = self.hub.sample()   # all assets concurrently (one batched CL RPC)
        for a in self.assets:
            t = ticks.get(a)
            if t is None:
                continue
            self.hist[a].append((t.ts, t.price))
            self.struct[a].update(t.ts, t.price)   # structural vol/regime anchor (shadow)
            cl = self.feeds[a].last_chainlink
            if cl is not None:
                self.cl_hist[a].append((cl.ts, cl.price))
            # record strike ONLY if we observe the window within ~25s of its open,
            # using the CHAINLINK price (the real settlement source). Record for both
            # the 5m and 15m grids so directional markets of either timeframe resolve.
            price = cl.price if cl is not None else t.price
            for ws_size in (300, 900):
                wstart = int(t.ts) - (int(t.ts) % ws_size)
                if (t.ts - wstart) <= 25 and (a, wstart) not in self.strikes:
                    self.strikes[(a, wstart)] = price
        if self.md is not None:
            self._merge_md_strikes()

    def _merge_md_strikes(self):
        """Chainlink-round-exact strikes from md_service's shared table: FILL
        windows the 25s local capture missed (the real gap — lanes used to go
        dark on missed opens), never overwrite a strike already in use. A
        local-vs-exact disagreement is logged once per window — it is the
        rollout diff data for the cutover gate."""
        assert self.md is not None
        now = int(time.time())
        for a in self.assets:
            for ws_size in (300, 900):
                w = now - now % ws_size
                for wstart in (w - ws_size, w):
                    sp = self.md.strike(a, wstart)
                    if sp is None:
                        continue
                    k = (a, wstart)
                    if k not in self.strikes:
                        self.strikes[k] = sp
                    elif abs(self.strikes[k] - sp) > 1e-9 and k not in self._md_strike_diff:
                        self._md_strike_diff.add(k)
                        print(f"  [md] strike diff {a}@{wstart}: local {self.strikes[k]} "
                              f"vs chainlink-exact {sp} (kept local; logged for rollout diff)")

    def spot(self, asset: str) -> float | None:
        return self.hist[asset][-1][1] if self.hist[asset] else None

    def cl_spot(self, asset: str) -> float | None:
        """Chainlink price = settlement-grade reference for strike/settlement."""
        return self.cl_hist[asset][-1][1] if self.cl_hist[asset] else self.spot(asset)

    def _discover_md(self) -> list[Market] | None:
        """Markets from md_service's shared 60s discovery sweep (one sweep for
        all lanes instead of one per lane). Returns None when the snapshot is
        stale/absent — the caller then runs the direct REST path unchanged.
        Market-building mirrors discover() below on the same raw dicts; the
        duplication is deliberate so the control path stays byte-identical."""
        assert self.md is not None
        d = self.md.discovery(max_age=180.0)
        if d is None:
            return None
        markets: list[Market] = []
        if "polymarket" in self.venues:
            for m in d.get("polymarket") or []:
                if m.get("_tf") not in self.poly_tfs or m.get("_asset") not in self.assets:
                    continue
                s, ws, a = m["_event_start"], m["_window_s"], m["_asset"]
                ticker = m.get("slug") or f"{a.lower()}-updown-{m['_tf']}-{int(s)}"
                markets.append(Market("polymarket", str(m.get("id")), a, "directional",
                                      m["_tf"], float(ws), float(s), float(s + ws),
                                      self.strikes.get((a, int(s))), raw=m, ticker=ticker))
            if "1h" in self.thresh_tfs:
                for m in d.get("polymarket_hourly") or []:
                    if m.get("_asset") not in self.assets:
                        continue
                    a = m["_asset"]; s = m["_event_start"]; ws = m["_window_s"]
                    ticker = m.get("slug") or f"{a.lower()}-hourly-{int(s)}"
                    markets.append(Market("polymarket", str(m.get("id")), a, "threshold", "1h",
                                          float(ws), float(s), float(s + ws),
                                          m["_strike"], raw=m, ticker=ticker))
        if "kalshi" in self.venues:
            for m in d.get("kalshi") or []:
                if m.get("_asset") not in self.assets:
                    continue
                a = m["_asset"]; ot = _iso(m.get("open_time")); ct = _iso(m.get("close_time"))
                ws = (ct - ot) if ct > ot else 900
                markets.append(Market("kalshi", str(m.get("ticker") or ""), a, "directional", "15m",
                                      ws, ot, ct, self.strikes.get((a, int(ot))), raw=m,
                                      ticker=m.get("ticker", "")))
            for m in d.get("kalshi_threshold") or []:
                if m.get("_tf") not in self.thresh_tfs or m.get("_asset") not in self.assets:
                    continue
                a = m["_asset"]; ot = _iso(m.get("open_time")); ct = _iso(m.get("close_time"))
                ws = (ct - ot) if ct > ot else (3600 if m["_tf"] == "1h" else 86400)
                markets.append(Market("kalshi", str(m.get("ticker") or ""), a, "threshold", m["_tf"],
                                      ws, ot, ct, m["_strike"], raw=m, ticker=m.get("ticker", "")))
        return markets

    def discover(self) -> list[Market]:
        if self.md is not None:
            got = self._discover_md()
            if got is not None:
                return got
        markets: list[Market] = []
        if "polymarket" in self.venues:
            try:
                for m in self.poly.discover(self.assets, self.poly_tfs):
                    s, ws, a = m["_event_start"], m["_window_s"], m["_asset"]
                    ticker = m.get("slug") or f"{a.lower()}-updown-{m['_tf']}-{int(s)}"
                    markets.append(Market("polymarket", str(m.get("id")), a, "directional",
                                          m["_tf"], float(ws), float(s), float(s + ws),
                                          self.strikes.get((a, int(s))), raw=m, ticker=ticker))
                # Polymarket hourly threshold ladder (dense, spot-centered) — the real
                # 1h source for BTC/ETH (Kalshi's hourly ladder is sparse + off-spot)
                if "1h" in self.thresh_tfs:
                    for m in self.poly.discover_hourly(self.assets):
                        a = m["_asset"]; s = m["_event_start"]; ws = m["_window_s"]
                        ticker = m.get("slug") or f"{a.lower()}-hourly-{int(s)}"
                        markets.append(Market("polymarket", str(m.get("id")), a, "threshold", "1h",
                                              float(ws), float(s), float(s + ws),
                                              m["_strike"], raw=m, ticker=ticker))
            except Exception:
                pass
        if "kalshi" in self.venues:
            try:
                for m in self.kalshi.discover(self.assets):  # 15m directional
                    a = m["_asset"]; ot = _iso(m.get("open_time")); ct = _iso(m.get("close_time"))
                    ws = (ct - ot) if ct > ot else 900
                    markets.append(Market("kalshi", str(m.get("ticker") or ""), a, "directional", "15m",
                                          ws, ot, ct, self.strikes.get((a, int(ot))), raw=m,
                                          ticker=m.get("ticker", "")))
                for m in self.kalshi.discover_threshold(self.assets, self.thresh_tfs):
                    a = m["_asset"]; ot = _iso(m.get("open_time")); ct = _iso(m.get("close_time"))
                    ws = (ct - ot) if ct > ot else (3600 if m["_tf"] == "1h" else 86400)
                    markets.append(Market("kalshi", str(m.get("ticker") or ""), a, "threshold", m["_tf"],
                                          ws, ot, ct, m["_strike"], raw=m, ticker=m.get("ticker", "")))
            except Exception:
                pass
        return markets

    def book_for(self, m: Market):
        if self.md is not None:
            bt = self.md.book_top(m.venue, m.market_id)
            if bt is not None:
                self._md_books += 1
                return bt
            self._md_book_miss += 1   # not mirrored (yet) -> direct REST below
        try:
            if m.venue == "polymarket":
                return self.poly.book_top(m.raw)
            return self.kalshi.book_top(m.raw)
        except Exception:
            return None

    def check_controls(self) -> bool:
        """Refresh the shared control plane (CONTROL.json / TRIPPED.json / HALT),
        run the wired max-loss breaker, and set the entry-block state. Returns
        True when the engine must FREEZE entirely (emergency stop)."""
        self.control = load_control(self.ctrl_root)
        if self.control.get("estop"):
            self._halted, self._block_reason = True, "estop"
            return True
        # other lanes' heartbeat aggregate (own live numbers used instead of
        # own possibly-stale status file)
        self._lanes = lanes_snapshot(self.ctrl_root, exclude=self.name)
        self.tripped = read_trip(self.ctrl_root)
        ml = self.control.get("max_loss") or {}
        if ml.get("enabled") and not self.tripped:
            day = self._lanes["day_pnl"] + self.pnl_today
            if day <= -abs(float(ml.get("limit", 0) or 0)):
                self.tripped = trip(self.ctrl_root,
                                    f"max-loss breaker: all-lane day PnL {day:+.2f} breached "
                                    f"-{float(ml['limit']):.2f} (detected by lane '{self.name}')",
                                    day, float(ml["limit"]))
                print("#" * 78)
                print(f"# !!! MAX-LOSS BREAKER TRIPPED: {self.tripped['reason']}")
                print("# !!! new entries BLOCKED on all lanes until cleared from the dashboard")
                print("#" * 78)
        if self.tripped:
            self._halted, self._block_reason = True, "maxloss-trip"
        elif self.control.get("halt_entries") or self.halt_file.exists():
            self._halted, self._block_reason = True, "halt"
        else:
            self._halted, self._block_reason = False, None
        return False

    def step(self):
        cyc_t0 = time.monotonic()
        if self.check_controls():
            return   # EMERGENCY STOP: frozen — no feeds/decisions/settles/learning
        self.sample_feeds()
        now = time.time()
        # resting limit orders: poll book, fill if the market trades through us,
        # log the path; settle (PnL) at expiry — filled -> outcome, unfilled -> no trade.
        # Books for every position that needs a poll this cycle are fetched
        # CONCURRENTLY (each worker has its own keep-alive conn); the position
        # state machine below stays single-threaded.
        need_poll = [pos for pos in self.open_pos.values()
                     if now < pos.end_ts and (pos.status == "resting" or
                        (self.tp_level > 0 and (pos.agg_size > 0 or pos.status == "filled")))]
        books = dict(zip((p.market_key for p in need_poll),
                         self._io_pool.map(self.book_for, (p.market for p in need_poll)),
                         strict=True)) if need_poll else {}
        for key, pos in list(self.open_pos.items()):
            if now < pos.end_ts:
                bt = books.get(key)
                if bt is not None:
                    spot_now = self.spot(pos.asset)
                    mins_left_now = (pos.end_ts - now) / 60.0
                    self.store.insert_path({"ts": now, "market": key, "asset": pos.asset,
                              "underlying": spot_now, "up_bid": bt.up_bid,
                              "up_ask": bt.up_ask, "side": pos.side, "limit_price": pos.price,
                              "up_bid_size": bt.up_bid_size, "up_ask_size": bt.up_ask_size,
                              "mins_left": round(mins_left_now, 3)})
                    if pos.status == "resting" and pos.size > 0 and pos.fill_check(bt.up_bid, bt.up_ask):
                        pos.status = "filled"   # PASSIVE leg crossed
                        pos.fill_ts = now
                        # fills/no_fills are now counted at settle (position-level),
                        # so the aggressive-leg trade and the passive fill don't
                        # double-count the same market.
                    # our-side sellable price = price we could exit (lift the bid) at:
                    # up -> up_bid; down -> 1-up_ask. Track excursion + ghost TP trigger
                    # only once ENTERED (agg leg, or passive leg just filled this step).
                    if pos.agg_size > 0 or pos.status == "filled":
                        our_bid = bt.up_bid if pos.side == "up" else (
                            (1.0 - bt.up_ask) if bt.up_ask is not None else None)
                        if our_bid is not None:
                            pos.last_bid = our_bid
                            if our_bid > pos.peak_bid:        # new high -> this is the pivot so far
                                pos.peak_bid = our_bid
                                pos.peak_mins_left = mins_left_now
                                pos.post_peak_low = our_bid   # reversal window restarts at the high
                                # snapshot the FULL state at the pivot (regression target
                                # for trades that reverse against us)
                                pos.peak_spot = spot_now
                                pos.peak_spread = (bt.up_ask - bt.up_bid) if (
                                    bt.up_ask is not None and bt.up_bid is not None) else None
                                pos.peak_bid_size = bt.up_bid_size
                                pos.peak_ask_size = bt.up_ask_size
                            elif pos.post_peak_low is not None:
                                pos.post_peak_low = min(pos.post_peak_low, our_bid)  # drawdown off the peak
                            pos.trough_bid = our_bid if pos.trough_bid is None else min(pos.trough_bid, our_bid)
                            if self.tp_level > 0 and not pos.tp_hit and our_bid >= self.tp_level - 1e-9:
                                pos.tp_hit = True   # a resting maker sell @ tp_level fills here
            if now >= pos.end_ts:
                self.settle(pos)
                del self.open_pos[key]
        for key, sh in list(self.shadow.items()):
            if now >= sh["end_ts"] + 5:   # small grace for the settlement price
                self.settle_shadow(key, sh)
                del self.shadow[key]
        # evaluate live markets (cache discovery ~60s; markets roll every 5-15m+)
        if not hasattr(self, "_mkts") or now - getattr(self, "_disc_at", 0) > 60:
            self._mkts = self.discover()
            self._disc_at = now
        # phase 1: cheap causal filters select the candidates worth a book fetch
        candidates: list[tuple[Market, float, float]] = []   # (market, mins_left, spot)
        for m in self._mkts:
            if m.strike is None:
                continue  # directional window whose open we didn't observe yet
            if m.key in self.decided:
                continue
            mins_left = (m.end_ts - now) / 60.0
            wm = m.window_s / 60.0
            # near-expiry trade band: bias to the LAST ~45% of the window, where the
            # edge was proven (predicting at the open is a coinflip). Cap at 180m so we
            # don't poll day-long markets all day.
            lo = max(0.5, self.early_frac * wm)
            hi = min(self.late_frac * wm, 180.0)
            if not (lo <= mins_left <= hi):
                continue
            spot = self.spot(m.asset)
            if spot is None:
                continue
            # bound book fetches on threshold ladders: only near-the-money strikes.
            # the hourly (1h) ladder is sparse (~4 strikes vs the daily's ~195), so a
            # tight band almost never has a qualifying strike -> widen it for 1h so the
            # lane actually fires; the 0.30-0.70 price discipline downstream still gates.
            if m.kind == "threshold" and m.strike > 0:
                nm = 0.08 if m.tf == "1h" else 0.04
                if abs(math.log(spot / m.strike)) > nm:
                    continue
            candidates.append((m, mins_left, spot))
        # phase 2: fetch all candidate books concurrently
        cand_books = list(self._io_pool.map(self.book_for,
                                            (c[0] for c in candidates))) if candidates else []
        # phase 3: evaluate sequentially (predictor + ledger stay single-threaded)
        for (m, mins_left, spot), bt in zip(candidates, cand_books, strict=True):
            if m.strike is None or m.key in self.decided:   # duplicate key within one sweep
                continue
            if bt is None or bt.up_mid is None:
                continue
            # liquidity filter: a wide book mid is not a real probability (stale /
            # illiquid threshold ladders sit ~0.5). Skip so we neither trade on it nor
            # pollute the calibration with spurious model-vs-market divergence.
            if bt.up_bid is not None and bt.up_ask is not None and (bt.up_ask - bt.up_bid) > 0.12:
                continue
            imb = 0.0
            if bt.up_bid_size + bt.up_ask_size > 0:
                imb = (bt.up_bid_size - bt.up_ask_size) / (bt.up_bid_size + bt.up_ask_size)
            feats = build_features(self.hist[m.asset], spot, m.strike, m.end_ts, now, imb)
            if feats is None:
                continue
            pred = self.predictor.predict(feats, market_mid=bt.up_mid)
            self.evaluate_trade(m, bt, pred, mins_left)
        self._cycle_ms.append(round((time.monotonic() - cyc_t0) * 1000.0, 1))

    def evaluate_trade(self, m: Market, bt, pred: Prediction, mins_left: float):
        assert m.strike is not None   # callers filter strikeless markets upstream
        p_ens = pred.p_cal           # ensemble (markov+boltzmann) calibrated up-prob
        mid = bt.up_mid
        if mid is None:
            return
        # --- structural digital-probability anchor: LOGGED ONLY (no longer gates) ---
        # Drift-controlled analysis (2026-06-03): alpha=N(d2) is overconfident in EVERY
        # probability bin; its low Brier was drift+extremeness, not skill. The real,
        # drift-robust edge is the ENSEMBLE's disagreement with the MARKET MID (within a
        # fixed mid-band, sign(p_cal-mid) predicts the residual outcome). So we trade the
        # side the ensemble favors RELATIVE TO THE MARKET PRICE (sign of p_cal-mid, NOT
        # p_cal-0.5) and gate on EV=|p_cal-mid|-fees. The alpha/anchor fields below are
        # kept for the calib edge-test logging only. See memory edge-ensemble-vs-mid.
        sh = self.struct.get(m.asset)
        st = sh.evaluate(pred.features["_spot"], m.strike, max(1.0, m.end_ts - time.time())) if sh else None
        alpha_up = st["alpha_up"] if st else None
        p_anchored, gate_margin, struct_div = None, None, None
        if alpha_up is not None:
            p_anchored, gate_margin = structural_anchor(p_ens, alpha_up, self.anchor_cfg)
            struct_div = abs(p_ens - alpha_up)
        # GATE on the ensemble's edge vs the market price (the drift-robust signal).
        p, req_margin = p_ens, self.min_ev
        raw_side = "up" if p > mid else "down"
        side = raw_side
        # CONDITIONAL GATE (4th/settle/candle engines): consult the per-(tf,mid-band)
        # drift-controlled skill tracker. It may FLIP the side (cell shows the raw tilt is
        # backwards in this band) and supplies the edge (cell conviction) in place of the
        # raw model-EV. Cells that haven't earned skill abstain. Main runs this OFF and is
        # unchanged. See predictor.ConditionalSkill / memory edge-ensemble-vs-mid.
        cond_ok, cs = True, None
        if self.conditional_gate:
            cs = self.predictor.cond_skill.assess(m.tf, mid, p)   # (status, factor, conv, n)
            if cs[1] == -1:
                side = "down" if raw_side == "up" else "up"
            cond_ok = cs[1] != 0
        p_side = p if side == "up" else 1 - p
        # execute as a LIMIT at mid on the predicted side (we proved taker kills it)
        price = mid if side == "up" else 1 - mid
        if not (0.02 < price < 0.98):
            return
        # EV: cell conviction drives the conditional engine (its side may disagree with
        # p_cal, so raw model-EV would be mis-signed); raw model-edge drives the others.
        # edge_gross = the raw edge BEFORE costs, in the mode's own currency: cell conviction
        # for the conditional gate, model edge (p_side−price = |p_cal−mid|) otherwise. Used
        # both for ev and for the taker cross-cost guard (so conditional isn't blocked by a
        # small |p_cal−mid| when its real edge is the cell conviction).
        if self.conditional_gate:
            edge_gross = cs[2] if cs else 0.0
        else:
            edge_gross = p_side - price
        ev = edge_gross - 0.005 - 0.5 * (1 - pred.confidence) * 0.02
        warmed = self.predictor.n_updates >= 40
        sane = abs(pred.fair_value_residual) <= 0.20   # stale-mid markets are spread-filtered upstream
        # price-discipline band (parameterized): main = mid-book 0.30-0.65; the separate
        # close-to-settlement engine uses a deep-favorite band. See leadlag-result.md.
        priced_ok = self.price_lo <= price <= self.price_hi
        # z_dist direction gate: don't bet against where spot has already moved vs strike.
        # z_dist>0 (spot above strike) favours UP; the sticky binary mid under-reacts to it.
        zd = pred.features["z_dist"]
        z_agree = (not self.zdist_gate) or (zd >= 0 if side == "up" else zd <= 0)
        side_ok = side in self.sides   # final side (post-flip) allowed for this lane
        decision = "TRADE" if (warmed and sane and priced_ok and z_agree
                               and side_ok and cond_ok and ev > req_margin
                               and pred.confidence > 0.05 and pred.move_prob > 0.35) else "SKIP"
        # entry blocks (control plane): estop / halt / max-loss trip force SKIP.
        # The row still logs the would-have-traded signal so every blocked window
        # stays counterfactually measurable.
        block_reason = (self._block_reason or "halt") if (self._halted and decision == "TRADE") else None
        if block_reason:
            decision = "SKIP"
        bucket_key = f"{pred.regime}|{m.tf}"
        pred_blob = asdict(pred)  # full prediction incl features/head dicts
        # SHADOW-LEARN every evaluated market at settlement (not just traded ones).
        # p_cal stored = the ENSEMBLE prob (for the calib edge test vs market/alpha).
        self.shadow[m.key] = {"pred": pred_blob, "end_ts": m.end_ts, "strike": m.strike,
                              "asset": m.asset, "market_mid": bt.up_mid, "bucket_key": bucket_key,
                              "tf": m.tf, "p_cal": p_ens, "alpha_up": alpha_up}
        # position sizing: half-Kelly fraction of CURRENT equity, capped (no flat 100)
        equity = self.equity
        contracts = 0.0           # PASSIVE resting leg (at mid), 2% half-Kelly budget
        agg_contracts = 0.0       # AGGRESSIVE taker leg (cross at decision), 1% budget
        agg_price = 0.0; agg_fee_per = 0.0
        if decision == "TRADE":
            # PASSIVE limit leg: flat fraction-of-equity (risk_cap) when flat_size, else half-Kelly
            # @ mid capped at risk_cap. Flat removes Kelly's edge-scaling, which oversized losers.
            f_pass = self.risk_cap if self.flat_size else kelly_fraction(p_side, price, cap=self.risk_cap, frac=self.kelly_frac)
            contracts = (equity * f_pass) / price if (f_pass > 0 and price > 0) else 0.0
            if contracts < 1.0:
                contracts = 0.0   # too small to rest
            # AGGRESSIVE taker leg: half-Kelly @ the ASK, capped at agg_risk_cap (1%),
            # placed only when the edge clears the cross cost (spread/2 + venue fee).
            edge = abs(p - mid)                      # |p_cal - mid|
            spread = (bt.up_ask - bt.up_bid) if (bt.up_ask is not None and bt.up_bid is not None) else 0.06
            aprice = bt.up_ask if side == "up" else ((1.0 - bt.up_bid) if bt.up_bid is not None else None)
            if aprice is not None and 0.02 < aprice < 0.98:
                cross_cost = spread / 2.0 + venue_taker_fee(m.venue, aprice)
                if edge > cross_cost:
                    f_agg = self.agg_risk_cap if self.flat_size else kelly_fraction(p_side, aprice, cap=self.agg_risk_cap, frac=self.kelly_frac)
                    ac = (equity * f_agg) / aprice if (f_agg > 0 and aprice > 0) else 0.0
                    if ac >= 1.0:
                        agg_contracts = ac
                        agg_price = aprice
                        agg_fee_per = venue_taker_fee(m.venue, aprice)
            if contracts < 1.0 and agg_contracts < 1.0:
                decision = "SKIP"   # neither leg worth placing
        # CROSS-LANE EXPOSURE CAP (toggleable + adjustable live from the
        # dashboard): committed $ across ALL lanes (resting + filled legs at
        # entry cost) must stay under max_total_frac of combined equity, and
        # this asset's committed $ under max_asset_frac. Sized AFTER both legs
        # so the check sees the true marginal cost of this trade.
        ec = (self.control or {}).get("exposure_cap") or {}
        if decision == "TRADE" and ec.get("enabled"):
            new_cost = contracts * price + agg_contracts * agg_price
            own_cost = sum(p.size * p.price + p.agg_size * p.agg_price
                           for p in self.open_pos.values())
            own_asset = sum(p.size * p.price + p.agg_size * p.agg_price
                            for p in self.open_pos.values() if p.asset == m.asset)
            lanes = self._lanes or {"open_cost": 0.0, "equity": 0.0, "by_asset": {}}
            comb_eq = lanes["equity"] + self.equity
            tot = lanes["open_cost"] + own_cost + new_cost
            asset_tot = lanes["by_asset"].get(m.asset, 0.0) + own_asset + new_cost
            if comb_eq > 0 and (tot > float(ec["max_total_frac"]) * comb_eq
                                or asset_tot > float(ec["max_asset_frac"]) * comb_eq):
                decision, block_reason = "SKIP", "exposure"
                print(f"  [CAP ] exposure cap blocked {m.asset} {m.tf}: "
                      f"total ${tot:.0f} / asset ${asset_tot:.0f} vs combined equity ${comb_eq:.0f} "
                      f"(caps {float(ec['max_total_frac']):.0%}/{float(ec['max_asset_frac']):.0%})")
        rec = {
            "ts": datetime.now(UTC).isoformat(), "market": m.key, "ticker": m.ticker,
            "asset": m.asset, "venue": m.venue, "tf": m.tf,
            "alpha_up": round(alpha_up, 4) if alpha_up is not None else None,
            "alpha_naive": round(st["alpha_naive"], 4) if st else None,
            "struct_div": round(struct_div, 4) if struct_div is not None else None,
            "p_anchored": round(p_anchored, 4) if p_anchored is not None else None,
            "gate_margin": round(gate_margin, 4) if gate_margin is not None else None,
            "sigma_s": st["sigma_s"] if st else None,
            "kappa": round(st["kappa"], 4) if (st and st["kappa"] is not None) else None,
            "struct_regime": st["regime"] if st else None,
            "kind": m.kind, "mins_left": round(mins_left, 2), "spot": pred.features["_spot"],
            "strike": m.strike, "z_dist": round(pred.features["z_dist"], 3),
            "p_up": round(pred.p_up, 4), "p_cal": round(p_ens, 4), "side": side,
            "move_prob": round(pred.move_prob, 3), "dir_edge": round(pred.directional_edge, 4),
            "fv_resid": round(pred.fair_value_residual, 4),
            "regime": pred.regime, "temp": round(pred.temperature, 3),
            "confidence": round(pred.confidence, 3), "book_mid_up": round(bt.up_mid, 4),
            "limit_price": round(price, 4), "ev": round(ev, 4), "decision": decision,
            "halted": 1 if block_reason else 0, "block_reason": block_reason,
        }
        self.store.insert_decision(rec)
        self.recent_decisions.appendleft(rec)
        split = f" +agg={agg_contracts:.0f}@{agg_price:.3f}" if agg_contracts > 0 else ""
        print(f"  [{decision}] {(m.ticker or m.key)[:38]:38} {m.asset} t-{mins_left:4.1f}m  "
              f"z={pred.features['z_dist']:+.2f} p={p:.3f} {side:4} mid={bt.up_mid:.3f} "
              f"ev={ev:+.4f} sz={contracts:.0f}{split} {pred.regime}")
        if decision == "TRADE":
            self.open_pos[m.key] = GhostPosition(
                market_key=m.key, asset=m.asset, side=side, price=price, size=contracts,
                p_model=p, regime=pred.regime, placed_at=time.time(), end_ts=m.end_ts,
                strike=m.strike, pred=pred_blob, ref_up_mid=bt.up_mid, market=m,
                agg_size=agg_contracts, agg_price=agg_price, agg_fee_per=agg_fee_per,
            )
        self.decided.add(m.key)

    @property
    def equity(self) -> float:
        return self.starting_bankroll + self.pnl

    @property
    def pnl_today(self) -> float:
        """This lane's realized PnL since UTC midnight (max-loss breaker input)."""
        return self.store.pnl_today()

    def settle_shadow(self, key: str, sh: dict):
        """Learn online from any evaluated market at settlement (PnL-free).

        Updates every specialist head, the fair-value head, calibration buckets,
        regime transitions, and bucket-level realized-edge feedback (the ML/RL loop).
        """
        end_price = self.cl_spot(sh["asset"])
        if end_price is None or sh["strike"] is None:
            return
        outcome_up = 1 if end_price > sh["strike"] else 0
        pr = Prediction(**sh["pred"])
        self.predictor.learn(pr, outcome_up, market_mid=sh.get("market_mid"),
                             bucket_key=sh.get("bucket_key"))
        # drift-controlled per-(tf,mid-band) skill of the disagreement signal (always
        # learned, every engine; only the conditional engine gates on it)
        self.predictor.cond_skill.learn(sh.get("tf"), sh.get("market_mid"),
                                        sh.get("p_cal"), outcome_up)
        # persist the markov+boltzmann edge test: model/structural/market vs outcome
        self.store.insert_calib({"ts": datetime.now(UTC).isoformat(), "market": key,
                                 "asset": sh["asset"], "tf": sh.get("tf"),
                                 "market_mid": sh.get("market_mid"), "p_cal": sh.get("p_cal"),
                                 "alpha_up": sh.get("alpha_up"), "outcome_up": outcome_up})

    def venue_reported_outcome(self, pos: GhostPosition) -> int | None:
        """Best-effort venue-reported resolution for reconciliation vs our Chainlink
        settle. Returns 1 (up/yes), 0 (down/no), or None if not yet resolved."""
        try:
            raw = pos.market.raw
            if pos.market.venue == "polymarket":
                prices = raw.get("outcomePrices")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if prices and float(prices[0]) in (0.0, 1.0):
                    return int(float(prices[0]))  # outcome[0] = Up
            else:  # kalshi
                res = raw.get("result")
                if res in ("yes", "no"):
                    return 1 if res == "yes" else 0
        except Exception:
            return None
        return None

    def settle(self, pos: GhostPosition):
        # settle on the CHAINLINK price (real settlement source), not the composite
        end_price = self.cl_spot(pos.asset)
        if end_price is None:
            return
        outcome_up = 1 if end_price > pos.strike else 0
        # reconciliation vs venue-reported outcome (best-effort; may be pending)
        venue_up = self.venue_reported_outcome(pos)
        recon = "n/a"
        if venue_up is not None:
            self.recon_checked += 1
            if venue_up != outcome_up:
                self.recon_mismatch += 1
                recon = "MISMATCH"
            else:
                recon = "match"
        won = outcome_up if pos.side == "up" else (1 - outcome_up)
        # two legs: aggressive (crossed at decision, always entered) + passive (rested
        # at mid, entered only if it crossed). A market "traded" if either leg entered.
        pass_filled = pos.status == "filled"
        pass_size = pos.size if pass_filled else 0.0
        entered = pos.agg_size + pass_size
        if entered <= 0:
            self.no_fills += 1   # pure-passive trade that never crossed -> no position
            rec = {"ts": datetime.now(UTC).isoformat(), "market": pos.market_key,
                   "ticker": pos.market.ticker, "asset": pos.asset, "venue": pos.market.venue, "tf": pos.market.tf,
                   "side": pos.side, "strike": pos.strike, "end_price": end_price,
                   "outcome_up": outcome_up, "filled": 0, "venue_outcome_up": venue_up, "recon": recon}
            self.store.insert_settlement(rec)
            print(f"  ··· NOFILL {(pos.market.ticker or pos.market_key)[:32]:32} {pos.side} limit={pos.price:.3f} "
                  f"(resting order never crossed)  fills={self.fills} nofills={self.no_fills}")
            return
        agg_pnl = (won - pos.agg_price - pos.agg_fee_per) * pos.agg_size if pos.agg_size > 0 else 0.0
        pass_pnl = (won - pos.price) * pass_size  # maker fill ~0 fee
        pnl = agg_pnl + pass_pnl
        eff_entry = (pos.agg_size * pos.agg_price + pass_size * pos.price) / entered
        self.pnl += pnl
        self.wins += won
        self.settled += 1
        self.fills += 1
        # --- ghost take-profit overlay (shadow ledger; real pnl above is untouched) ---
        # Mirror the real entry exactly; only the EXIT differs. If the TP fired, the ghost
        # sold both legs at tp_level (maker exit ~0 fee, entry fees retained); else it held
        # to settlement so its pnl == real pnl. tp_cum runs from the real-equity seam.
        tp_fields = {}
        if self.tp_level > 0:
            if pos.tp_hit:
                tp_pnl = ((self.tp_level - pos.agg_price - pos.agg_fee_per) * pos.agg_size
                          + (self.tp_level - pos.price) * pass_size)
                tp_exited, tp_won = 1, 1   # banked a profit (tp_level > entry by construction)
            else:
                tp_pnl, tp_exited, tp_won = pnl, 0, won
            self.tp_cum += tp_pnl
            self.tp_settled += 1
            self.tp_exits += tp_exited
            self.tp_wins += tp_won
            tp_fields = {"tp_pnl": round(tp_pnl, 3), "tp_cum_pnl": round(self.tp_cum, 3),
                         "tp_exited": tp_exited, "peak_bid": round(pos.peak_bid, 4),
                         "trough_bid": round(pos.trough_bid, 4) if pos.trough_bid is not None else None,
                         "peak_mins_left": round(pos.peak_mins_left, 2) if pos.peak_mins_left is not None else None,
                         "post_peak_low": round(pos.post_peak_low, 4) if pos.post_peak_low is not None else None,
                         "peak_spot": round(pos.peak_spot, 6) if pos.peak_spot is not None else None,
                         "peak_spread": round(pos.peak_spread, 4) if pos.peak_spread is not None else None,
                         "peak_bid_size": pos.peak_bid_size, "peak_ask_size": pos.peak_ask_size}
        rec = {"ts": datetime.now(UTC).isoformat(), "market": pos.market_key,
               "ticker": pos.market.ticker, "asset": pos.asset, "venue": pos.market.venue, "tf": pos.market.tf,
               "side": pos.side, "strike": pos.strike, "end_price": end_price,
               "outcome_up": outcome_up, "won": won, "filled": 1, "fill_price": round(eff_entry, 4),
               "venue_outcome_up": venue_up, "recon": recon, "pnl": round(pnl, 3),
               "cum_pnl": round(self.pnl, 3), "settled": self.settled,
               "fill_rate": round(self.fills / (self.fills + self.no_fills), 3) if (self.fills + self.no_fills) else None,
               "winrate": round(self.wins / self.settled, 3),
               "agg_size": round(pos.agg_size, 2), "pass_size": round(pass_size, 2),
               "pass_filled": 1 if pass_filled else 0, **tp_fields}
        self.store.insert_settlement(rec)
        legs = f"agg={pos.agg_size:.0f}@{pos.agg_price:.3f}+pass={pass_size:.0f}@{pos.price:.3f}" if pos.agg_size > 0 else f"sz={pass_size:.0f}@{pos.price:.3f}"
        tp_tag = ""
        if self.tp_level > 0:
            tp_tag = (f" | TP {'EXIT@'+format(self.tp_level,'.2f') if pos.tp_hit else 'held'} "
                      f"peak={pos.peak_bid:.2f} tp_eq={self.starting_bankroll + self.tp_cum:.1f}")
        print(f"  >>> SETTLE {(pos.market.ticker or pos.market_key)[:32]:32} {pos.side} {legs} -> "
              f"{'WIN' if won else 'LOSS'} (mkt {'UP' if outcome_up else 'DOWN'}) recon={recon} "
              f"pnl={pnl:+.1f} equity={self.equity:.1f} wr={self.wins}/{self.settled}{tp_tag}")

    def write_status(self):
        p = self.predictor
        cal = []
        for i in range(10):
            n, wins = p.cal[i]
            cal.append({"bucket": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": round(n - 1.0, 0),
                        "hit_rate": round(wins / n, 3) if n > 0 else None})
        # every timeframe lane the engine is configured to trade (so the dashboard can
        # show all lanes incl. ones that are dark this window — thin != inactive)
        configured_tfs = sorted(
            set(self.poly_tfs) | ({"15m"} if "kalshi" in self.venues else set())
            | (set(self.thresh_tfs) if "kalshi" in self.venues else set()),
            key=tf_sort_key)
        status = {
            "updated": datetime.now(UTC).isoformat(),
            "engine": "live forward prediction engine",
            "feed": "chainlink (settlement-grade) + coinbase drift",
            "config": {"assets": self.assets, "venues": list(self.venues),
                       "poly_tfs": list(self.poly_tfs), "thresh_tfs": list(self.thresh_tfs),
                       "timeframes": configured_tfs, "name": self.name,
                       "anchor": self.use_anchor, "band": [self.price_lo, self.price_hi],
                       "gate": ("conditional skill (tf,band)" if self.conditional_gate else "edge=|p_cal−mid|"),
                       "conditional_gate": self.conditional_gate,
                       "sides": list(self.sides), "late_frac": self.late_frac},
            "assets": {a: {"spot": self.spot(a), "chainlink": self.cl_spot(a),
                           "n_history": len(self.hist[a])} for a in self.assets},
            "open_positions": [
                {"market": x.market_key, "ticker": x.market.ticker, "asset": x.asset,
                 "side": x.side, "status": x.status, "size": round(x.size, 0),
                 "cost": round(x.size * x.price + x.agg_size * x.agg_price, 2),
                 "fill_price": round(x.price, 4), "p_model": round(x.p_model, 4),
                 "regime": x.regime, "strike": round(x.strike, 4),
                 "mins_to_settle": round((x.end_ts - time.time()) / 60.0, 1),
                 "peak_bid": round(x.peak_bid, 3), "tp_hit": x.tp_hit,
                 "now_bid": round(x.last_bid, 3) if x.last_bid is not None else None,
                 "drawdown": round(x.peak_bid - x.last_bid, 3) if (x.last_bid is not None and x.peak_bid > 0) else None}
                for x in self.open_pos.values()],
            "performance": {"settled": self.settled, "wins": self.wins,
                            "pnl_today": round(self.pnl_today, 2),
                            "winrate": round(self.wins / self.settled, 3) if self.settled else None,
                            "cum_pnl": round(self.pnl, 2),
                            "starting_bankroll": self.starting_bankroll,
                            "equity": round(self.equity, 2),
                            "model_updates": p.n_updates,
                            "markets_tracked": len(getattr(self, "_mkts", [])),
                            "decisions": len(self.decided),
                            "fills": self.fills, "no_fills": self.no_fills,
                            "fill_rate": round(self.fills / (self.fills + self.no_fills), 3) if (self.fills + self.no_fills) else None,
                            "recon_checked": self.recon_checked, "recon_mismatch": self.recon_mismatch},
            "model": {
                "heads": [{"name": n, "reliability": round(h.reliability, 3),
                           "n": h.n, "weights": {f: round(w, 3) for f, w in h.w.items()}}
                          for n, h in p.heads.items()]
                         + [{"name": "fair_value", "reliability": round(p.fair.reliability, 3),
                             "n": p.fair.n, "weights": {"k": round(p.fair.k, 3)}}],
                "calibration": cal,
                "calib_err": round(p._calib_err, 4),
                "regimes_seen": list(p.trans.keys()),
                "cond_skill": p.cond_skill.summary(),
                "bucket_edge": [
                    {"bucket": k, "n": int(n), "win_rate": round(w / n, 3)}
                    for k, (n, w) in sorted(p.bucket.items(), key=lambda kv: -kv[1][0])
                    if n >= 3][:14],
            },
            "recent_decisions": list(self.recent_decisions)[:20],
            "health": self.health_snapshot(),
            "control": self.control_snapshot(),
        }
        # ghost take-profit overlay monitor: equity starts at the real-equity seam and
        # diverges only on TP exits. delta_vs_real = ghost − real PnL since the seam.
        if self.tp_level > 0:
            status["tp"] = {
                "level": self.tp_level,
                "equity": round(self.starting_bankroll + self.tp_cum, 2),
                "pnl": round(self.tp_cum, 2),
                "delta_vs_real": round(self.tp_cum - self.pnl, 2),
                "settled": self.tp_settled, "exits": self.tp_exits,
                "held": self.tp_settled - self.tp_exits,
                "winrate": round(self.tp_wins / self.tp_settled, 3) if self.tp_settled else None,
            }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, default=float, indent=1))
        tmp.replace(self.status_path)

    def control_snapshot(self) -> dict:
        """This lane's view of the shared control plane for status.json/dashboard:
        switch states, the latched trip, and the GLOBAL aggregates (other fresh
        lanes + this lane's live numbers) that the wired controls act on."""
        c = self.control or {}
        lanes = self._lanes or {"lanes": [], "partial": [], "day_pnl": 0.0,
                                "equity": 0.0, "open_cost": 0.0, "by_asset": {}}
        own_cost = sum(p.size * p.price + p.agg_size * p.agg_price
                       for p in self.open_pos.values())
        by_asset = dict(lanes["by_asset"])
        for p in self.open_pos.values():
            by_asset[p.asset] = round(by_asset.get(p.asset, 0.0)
                                      + p.size * p.price + p.agg_size * p.agg_price, 2)
        return {
            "estop": bool(c.get("estop")),
            "estop_reason": c.get("estop_reason"),
            "estop_ts": c.get("estop_ts"),
            "halt_entries": bool(c.get("halt_entries")) or self.halt_file.exists(),
            "block_reason": self._block_reason,
            "tripped": self.tripped,
            "max_loss": {**(c.get("max_loss") or {}),
                         "day_pnl_global": round(lanes["day_pnl"] + self.pnl_today, 2)},
            "exposure_cap": {**(c.get("exposure_cap") or {}),
                             "open_cost_global": round(lanes["open_cost"] + own_cost, 2),
                             "equity_global": round(lanes["equity"] + self.equity, 2),
                             "by_asset": by_asset},
            "lanes_visible": [ln["name"] for ln in lanes["lanes"]] + [self.name],
            "lanes_partial": lanes["partial"],
        }

    def health_snapshot(self) -> dict:
        """System-health readout for status.json: cycle latency distribution,
        per-host API error budget (live.net counters), settlement-feed staleness
        (seconds since each asset's last Chainlink round), and the kill switch."""
        cyc = sorted(self._cycle_ms)
        stale = self.hub.staleness()
        known = [s for s in stale.values() if s is not None]
        out = {
            "halted": self._halted,
            "cycle_ms_last": self._cycle_ms[-1] if self._cycle_ms else None,
            "cycle_ms_p50": cyc[len(cyc) // 2] if cyc else None,
            "cycle_ms_p95": cyc[int(0.95 * (len(cyc) - 1))] if cyc else None,
            "chainlink_staleness_s": stale,
            "chainlink_staleness_max_s": max(known) if known else None,
            "api": net.stats(),
        }
        if self.md is not None:
            out["md"] = {**self.md.stats(),
                         "transport": getattr(self.hub, "transport", "rest"),
                         "feed_fallbacks": getattr(self.hub, "fallbacks", 0),
                         "books_md": self._md_books,
                         "books_rest_fallback": self._md_book_miss,
                         "strike_diffs": len(self._md_strike_diff)}
        return out

    def log_health(self):
        """Persist a health row (~once a minute) so stalls and API degradation
        are queryable history, not just a scrolled-away console."""
        try:
            h = self.health_snapshot()
            api = h["api"]
            self.store.insert_health({
                "ts": datetime.now(UTC).isoformat(),
                "cycle_ms": h["cycle_ms_last"], "cycle_p95_ms": h["cycle_ms_p95"],
                "open_pos": len(self.open_pos), "markets": len(getattr(self, "_mkts", [])),
                "api_calls": sum(v["calls"] for v in api.values()),
                "api_errors": sum(v["errors"] for v in api.values()),
                "max_cl_staleness_s": h["chainlink_staleness_max_s"],
                "halted": 1 if h["halted"] else 0,
            })
        except Exception:
            pass

    def run(self, minutes: float, interval: float = 6.0):
        # minutes <= 0 -> run indefinitely until killed
        end = float("inf") if minutes <= 0 else time.time() + minutes * 60
        print(f"# Forward engine [{self.name}] | assets={self.assets} venues={self.venues} "
              f"poly_tfs={self.poly_tfs} thresh_tfs={self.thresh_tfs} | band=[{self.price_lo},{self.price_hi}] "
              f"late_frac={self.late_frac} anchor={'ON' if self.use_anchor else 'shadow'} | "
              f"duration={'unlimited' if minutes <= 0 else f'{minutes}m'} | logging to {self.out}/")
        cyc = 0
        err_streak = 0
        was_estopped = False
        while time.time() < end:
            cyc += 1
            try:
                self.step()
                self.write_status()
                if cyc % 5 == 0:
                    self.save_model()   # checkpoint learning every ~40s
                if cyc % 8 == 0:
                    self.log_health()   # health row ~once a minute
                err_streak = 0
                if self.control.get("estop") and not was_estopped:
                    print(f"# !!! EMERGENCY STOP ACTIVE — engine '{self.name}' frozen "
                          f"({self.control.get('estop_reason') or 'operator'})")
                was_estopped = bool(self.control.get("estop"))
            except Exception as e:
                err_streak += 1
                print(f"  ! step error ({err_streak} consecutive): {type(e).__name__}: {e}")
                # AUTO-ESTOP: a persistent crash loop means the engine is
                # dysfunctional — freeze ALL lanes and put the cause on the
                # dashboard banner rather than grinding a broken loop.
                if err_streak == 5:
                    try:
                        save_control(self.ctrl_root, {
                            "estop": True,
                            "estop_reason": (f"auto: lane '{self.name}' hit {err_streak} consecutive "
                                             f"cycle errors — last: {type(e).__name__}: {e}"),
                            "estop_ts": datetime.now(UTC).isoformat(),
                        })
                        print("#" * 78)
                        print(f"# !!! AUTO EMERGENCY STOP: lane '{self.name}' crash loop — all engines frozen")
                        print("#" * 78)
                    except Exception:
                        pass
            sp = self.spot("BTC")
            if cyc % 5 == 1:
                print(f"[{datetime.now(UTC):%H:%M:%S}] cyc={cyc} BTC={sp and f'${sp:,.1f}'} "
                      f"open={len(self.open_pos)} settled={self.settled} cum_pnl={self.pnl:+.1f} "
                      f"model_updates={self.predictor.n_updates}")
            time.sleep(interval)
        print(f"\n# done. settled={self.settled} cum_pnl={self.pnl:+.2f} "
              f"winrate={(self.wins/self.settled) if self.settled else 0:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,DOGE,BNB,HYPE")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--interval", type=float, default=8.0)
    ap.add_argument("--poly-tfs", default="15m,5m,4h")
    ap.add_argument("--thresh-tfs", default="1h,1D")
    ap.add_argument("--data-dir", default="data/forward_live")
    ap.add_argument("--name", default="main")
    ap.add_argument("--price-lo", type=float, default=0.50)   # fill floor: <0.50 fills won 26% (toxic)
    ap.add_argument("--price-hi", type=float, default=0.65)
    ap.add_argument("--late-frac", type=float, default=0.45)
    ap.add_argument("--early-frac", type=float, default=0.05)   # mins_left>=early_frac*window
    ap.add_argument("--risk-cap", type=float, default=0.02)     # passive leg Kelly cap
    ap.add_argument("--agg-risk-cap", type=float, default=0.01)  # aggressive leg Kelly cap
    ap.add_argument("--kelly-frac", type=float, default=0.5)    # 0.5 half / 0.25 quarter
    ap.add_argument("--tilt-cap", type=float, default=0.4)      # max |logit nudge| over mid
    ap.add_argument("--tp-level", type=float, default=0.90)     # ghost take-profit exit price (0 disables)
    ap.add_argument("--anchor", default="on")   # on|shadow
    ap.add_argument("--flat-size", default="on")   # on|off: flat fraction-of-equity vs Kelly edge-scaling
    ap.add_argument("--zdist-gate", default="on")  # on|off: never trade against sign(z_dist)
    ap.add_argument("--conditional-gate", default="off")  # on|off: per-(tf,band) skill gate
    ap.add_argument("--sides", default="up,down")   # final sides the lane may trade
    # --md: consume the shared md_service stream (data/md.sock) with automatic REST
    # fallback. TRANSPORT only (not lane-defining decision config, so it is deliberately
    # NOT in launch_args.json's config-drift guard).
    ap.add_argument("--md", default="off")  # on|off
    args = ap.parse_args()
    eng = ForwardEngine(args.assets.split(","), poly_tfs=tuple(args.poly_tfs.split(",")),
                        thresh_tfs=tuple(args.thresh_tfs.split(",")), data_dir=args.data_dir,
                        name=args.name, price_lo=args.price_lo, price_hi=args.price_hi,
                        late_frac=args.late_frac, use_anchor=(args.anchor == "on"),
                        early_frac=args.early_frac, risk_cap=args.risk_cap,
                        agg_risk_cap=args.agg_risk_cap, kelly_frac=args.kelly_frac,
                        tilt_cap=args.tilt_cap, tp_level=args.tp_level,
                        flat_size=(args.flat_size == "on"), zdist_gate=(args.zdist_gate == "on"),
                        conditional_gate=(args.conditional_gate == "on"),
                        sides=tuple(s.strip() for s in args.sides.split(",") if s.strip()),
                        md_mode=(args.md == "on"))
    eng.run(args.minutes, args.interval)


if __name__ == "__main__":
    main()
