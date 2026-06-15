"""leadlag -- does the underlying move BEFORE the venue book mid reprices?

The near-expiry thesis only has an edge if the book LAGS the underlying: when the
underlying moves, if the book hasn't yet repriced P(up), we can act before it does.
This tests that directly on the engine's `paths` table (logged per cycle while a
limit rests: underlying price + up_bid/up_ask, ~8-16s apart, near expiry).

Method (pooled cross-correlation of first differences):
  du_t  = log(underlying_t / underlying_{t-1})        # underlying move
  dm_t  = mid_t - mid_{t-1},  mid = (up_bid+up_ask)/2  # book P(up) move
  For lag k, correlate du_t with dm_{t+k}, standardized WITHIN each market then
  pooled. k>0  => underlying leads the book by k cycles  => exploitable edge.
  k=0   => contemporaneous (book reprices within our sampling) => no edge at 8s.
  k<0   => book leads underlying => definitely no edge for us.

Honest limits: (a) `underlying` is the COMPOSITE feed (Chainlink + Coinbase ticks
the market also sees), so this UNDERSTATES a pure-Chainlink lead. (b) ~8-16s
sampling: a lead faster than a cycle shows up as lag 0. (c) paths exist only for
traded near-expiry markets (selection = the regime we trade). (d) small n.

Run:  python3 research/strategy/leadlag.py data/forward_live/forward_engine.db
"""
from __future__ import annotations
import math
import sqlite3
import sys


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, n
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return 0.0, n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(sx * sy), n


def standardize(xs):
    n = len(xs)
    if n < 2:
        return xs
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    if sd <= 0:
        return [0.0] * n
    return [(x - m) / sd for x in xs]


def main(db):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    mkts = [r["market"] for r in conn.execute(
        "SELECT market, COUNT(*) n FROM paths GROUP BY market HAVING n>=20").fetchall()]

    LAGS = list(range(-4, 5))
    pooled = {k: ([], []) for k in LAGS}     # lag -> (du list, dm list)
    used, skipped_flat = 0, 0
    mid_change_frac, und_change_frac = [], []

    for mk in mkts:
        rows = conn.execute(
            "SELECT ts, underlying u, up_bid b, up_ask a FROM paths "
            "WHERE market=? AND underlying IS NOT NULL AND up_bid IS NOT NULL "
            "AND up_ask IS NOT NULL ORDER BY ts", (mk,)).fetchall()
        if len(rows) < 20:
            continue
        u = [r["u"] for r in rows]
        mid = [0.5 * (r["b"] + r["a"]) for r in rows]
        du = [math.log(u[i] / u[i - 1]) if u[i] > 0 and u[i - 1] > 0 else 0.0
              for i in range(1, len(u))]
        dm = [mid[i] - mid[i - 1] for i in range(1, len(mid))]
        # require the underlying to actually vary in this market (precision/staleness)
        if sum(1 for x in du if x != 0) < 5:
            skipped_flat += 1
            continue
        und_change_frac.append(sum(1 for x in du if x != 0) / len(du))
        mid_change_frac.append(sum(1 for x in dm if x != 0) / len(dm))
        zdu, zdm = standardize(du), standardize(dm)
        for k in LAGS:
            for i in range(len(zdu)):
                j = i + k
                if 0 <= j < len(zdm):
                    pooled[k][0].append(zdu[i])
                    pooled[k][1].append(zdm[j])
        used += 1

    print(f"markets used={used}  skipped_flat_underlying={skipped_flat}")
    if und_change_frac:
        print(f"avg fraction of cycles underlying moved = "
              f"{sum(und_change_frac)/len(und_change_frac):.2f}")
        print(f"avg fraction of cycles book mid moved    = "
              f"{sum(mid_change_frac)/len(mid_change_frac):.2f}   "
              f"(low => stale/illiquid book => test underpowered)")
    print(f"\n{'lag k':>6} {'r(du_t, dm_t+k)':>16} {'n':>7}   interpretation")
    print(f"{'':>6} {'(k>0: und leads)':>16}")
    best = None
    for k in LAGS:
        r, n = pearson(pooled[k][0], pooled[k][1])
        tag = ""
        if k == 0:
            tag = "contemporaneous"
        elif k > 0:
            tag = "underlying LEADS book"
        else:
            tag = "book leads underlying"
        bar = "#" * int(abs(r) * 40)
        print(f"{k:>6} {r:>16.3f} {n:>7}   {tag:24} {('+' if r>=0 else '-')}{bar}")
        if best is None or abs(r) > abs(best[1]):
            best = (k, r)
    print(f"\npeak |correlation| at lag k={best[0]} (r={best[1]:.3f})")
    if best[0] <= 0:
        print("=> NO exploitable lead: book reprices at/before our sampling. The")
        print("   near-expiry directional thesis has no latency edge at this resolution.")
    else:
        print(f"=> underlying leads the book by ~{best[0]} cycle(s): POTENTIAL edge,")
        print("   worth a finer-resolution confirmation before trading it.")

    # actionable check: does a big underlying move predict the NEXT mid move's sign?
    big, hit, base_up = 0, 0, 0
    for k in [1]:
        xs, ys = pooled[k]
        if xs:
            thr = sorted(abs(x) for x in xs)[int(0.66 * len(xs))]
            for i in range(len(xs)):
                if ys[i] != 0:
                    base_up += 1 if ys[i] > 0 else 0
                if abs(xs[i]) >= thr and ys[i] != 0:
                    big += 1
                    if (xs[i] > 0) == (ys[i] > 0):
                        hit += 1
    if big:
        print(f"\ndirectional: when |du_t| is large (top tercile), next-cycle mid moves")
        print(f"  the SAME direction {hit}/{big} = {hit/big:.1%} of the time "
              f"(50% = no predictive lead).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 research/strategy/leadlag.py <db>")
        sys.exit(1)
    main(sys.argv[1])
