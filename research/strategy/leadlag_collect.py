"""leadlag_collect -- high-frequency RAW-feed-vs-book collector + CCF analyzer.

Decisive disambiguation for leadlag-result.md: the composite feed (Chainlink
anchor + Coinbase drift) LAGS the venue book by ~1 cycle at 8s sampling. Is that
because the book is genuinely faster (HFT), or because our composite is smoothed
by the slow on-chain Chainlink anchor? This logs the RAW Coinbase tick (no
anchoring) next to the book mid at ~2s, then re-runs the cross-correlation.

  raw Coinbase LEADS book  -> the edge is real, our composite was just too slow
                              (fixable infra) -> deep-favorite strategy is viable
  raw Coinbase still LAGS   -> the book is genuinely faster -> near-expiry
                              directional is dead, pivot to cross-venue

Standalone: reuses live.feed (CoinbaseFeed/BinanceFeed) + live.venues for books;
writes its own SQLite (research/leadlag.db), never touches forward_engine.db.

  collect:  python3 research/strategy/leadlag_collect.py collect [seconds]
  analyze:  python3 research/strategy/leadlag_collect.py analyze
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import time

# make the `live` package importable when run as a plain script
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from live.feed import CoinbaseFeed, BinanceFeed          # noqa: E402
from live.venues import PolymarketClient                 # noqa: E402

DB = os.path.join(_ROOT, "research", "leadlag.db")
ASSETS = ["BTC", "ETH"]            # most liquid Polymarket up/down books
TFS = ("5m", "15m")               # fast-repricing near-expiry markets
POLL_S = 2.0                       # ~2s sampling (4x finer than the 8s engine)
REDISCOVER_S = 45.0


def _store():
    c = sqlite3.connect(DB)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS ll(
        id INTEGER PRIMARY KEY, ts_raw REAL, ts_book REAL, asset TEXT, market TEXT,
        tf TEXT, raw_price REAL, up_bid REAL, up_ask REAL, mid REAL)""")
    c.commit()
    return c


def collect(duration: float = 2400.0) -> None:
    c = _store()
    poly = PolymarketClient()
    feeds = {a: CoinbaseFeed(a) for a in ASSETS}
    bins = {a: BinanceFeed(a) for a in ASSETS}
    mkts: list = []
    disc_at = 0.0
    end = time.time() + duration
    n, errs = 0, 0
    print(f"# leadlag collect -> {DB} | assets={ASSETS} tfs={TFS} poll={POLL_S}s "
          f"dur={duration:.0f}s", flush=True)
    while time.time() < end:
        loop0 = time.time()
        if loop0 - disc_at > REDISCOVER_S:
            try:
                mkts = poly.discover(ASSETS, TFS)
                disc_at = loop0
            except Exception:
                errs += 1
        # one raw tick per asset per cycle (reused across that asset's markets)
        raw = {}
        for a in ASSETS:
            t = feeds[a].get() or bins[a].get()
            if t is not None:
                raw[a] = (t.ts, t.price)
        for m in mkts:
            a = m.get("_asset")
            if a not in raw:
                continue
            try:
                bt = poly.book_top(m)
            except Exception:
                errs += 1
                continue
            if bt is None or bt.up_bid is None or bt.up_ask is None:
                continue
            ts_raw, rp = raw[a]
            mid = 0.5 * (bt.up_bid + bt.up_ask)
            c.execute("INSERT INTO ll(ts_raw,ts_book,asset,market,tf,raw_price,up_bid,up_ask,mid)"
                      " VALUES(?,?,?,?,?,?,?,?,?)",
                      (ts_raw, bt.observed_at, a, m.get("slug") or str(m.get("id")),
                       m.get("_tf"), rp, bt.up_bid, bt.up_ask, mid))
            n += 1
        c.commit()
        if n and n % 100 < len(mkts):
            print(f"  [{time.strftime('%H:%M:%S')}] rows={n} mkts={len(mkts)} errs={errs}", flush=True)
        dt = time.time() - loop0
        if dt < POLL_S:
            time.sleep(POLL_S - dt)
    print(f"# done. rows={n} errs={errs}", flush=True)


# ----------------------------- analysis ----------------------------------- #
def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, n
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs); sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return 0.0, n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(sx * sy), n


def _std(xs):
    n = len(xs)
    if n < 2:
        return xs
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return [0.0] * n if sd <= 0 else [(x - m) / sd for x in xs]


def summary(db_path: str = DB) -> dict:
    """Compute the lead-lag CCF + 80-85c band check as a JSON-able dict (reused by
    the dashboard). peak_lag>0 => raw feed leads book (edge); <=0 => book leads."""
    try:
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {"rows": 0, "markets": 0, "ccf": [], "ready": False}
    c.row_factory = sqlite3.Row
    try:
        mkts = [r["market"] for r in c.execute(
            "SELECT market, COUNT(*) n FROM ll GROUP BY market HAVING n>=20")]
        total = c.execute("SELECT COUNT(*) n FROM ll").fetchone()["n"]
        tspan = c.execute("SELECT MIN(ts_book) a, MAX(ts_book) b FROM ll").fetchone()
    except sqlite3.OperationalError:
        return {"rows": 0, "markets": 0, "ccf": [], "ready": False}
    LAGS = list(range(-5, 6))
    pooled = {k: ([], []) for k in LAGS}
    used, band_big, band_hit = 0, 0, 0
    for mk in mkts:
        rows = c.execute("SELECT raw_price rp, mid FROM ll WHERE market=? ORDER BY ts_book", (mk,)).fetchall()
        rp = [r["rp"] for r in rows]; mid = [r["mid"] for r in rows]
        du = [math.log(rp[i] / rp[i - 1]) if rp[i] > 0 and rp[i - 1] > 0 else 0.0 for i in range(1, len(rp))]
        dm = [mid[i] - mid[i - 1] for i in range(1, len(mid))]
        if sum(1 for x in du if x != 0) < 5:
            continue
        zdu, zdm = _std(du), _std(dm)
        for k in LAGS:
            for i in range(len(zdu)):
                j = i + k
                if 0 <= j < len(zdm):
                    pooled[k][0].append(zdu[i]); pooled[k][1].append(zdm[j])
        for i in range(len(du) - 1):
            if 0.80 <= mid[i + 1] <= 0.85 and du[i] != 0 and dm[i + 1] != 0:
                band_big += 1
                if (du[i] > 0) == (dm[i + 1] > 0):
                    band_hit += 1
        used += 1
    ccf, best = [], None
    for k in LAGS:
        r, n = _pearson(*pooled[k])
        ccf.append({"lag": k, "r": round(r, 3), "n": n})
        if best is None or abs(r) > abs(best[1]):
            best = (k, r)
    verdict = "collecting…"
    if best and used:
        if best[0] > 0:
            verdict = "RAW FEED LEADS book — edge real, composite too slow"
        elif best[0] == 0:
            verdict = "contemporaneous — book reprices within ~2s, no lead"
        else:
            verdict = "book STILL leads raw feed — directional dead, pivot cross-venue"
    return {
        "rows": total, "markets": used, "poll_s": POLL_S,
        "elapsed_s": round((tspan["b"] - tspan["a"]) if (tspan and tspan["a"]) else 0, 0),
        "ccf": ccf, "peak_lag": best[0] if best else None,
        "peak_r": round(best[1], 3) if best else None, "verdict": verdict,
        "band_big": band_big, "band_hit": band_hit,
        "band_rate": round(band_hit / band_big, 3) if band_big else None,
        "ready": used > 0,
    }


def analyze() -> None:
    s = summary()
    print(f"total rows={s['rows']}  markets used={s['markets']}  elapsed={s['elapsed_s']}s")
    print(f"\n{'lag k':>6} {'r(du_t,dm_t+k)':>15} {'n':>7}   meaning")
    for row in s["ccf"]:
        k = row["lag"]
        mean = "raw LEADS book" if k > 0 else ("contemporaneous" if k == 0 else "book leads raw")
        print(f"{k:>6} {row['r']:>15.3f} {row['n']:>7}   {mean}")
    print(f"\npeak |r| at lag k={s['peak_lag']} (r={s['peak_r']})  [POLL={s.get('poll_s')}s/cycle]")
    print(f"=> {s['verdict']}")
    if s["band_big"]:
        print(f"\n80-85c band: raw move predicts next mid direction "
              f"{s['band_hit']}/{s['band_big']} = {s['band_rate']:.1%} (50% = no lead)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "collect":
        collect(float(sys.argv[2]) if len(sys.argv) > 2 else 2400.0)
    else:
        analyze()
