"""Edge diagnostics: calibration, Brier, and net-of-fee EV by bucket.

Everything is stdlib-only. The central question this answers:

  Is there any slice of decisions where the model's stated edge actually predicts
  positive realized net-of-fee expected value, out of sample?

If no bucket is positive after costs, there is no edge to amplify and no amount of
ensembling / RL / sizing will make the strategy profitable.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class Bucket:
    label: str
    n: int = 0
    wins: int = 0
    sum_p: float = 0.0          # sum of model prob for selected side
    sum_price: float = 0.0      # sum of taker entry price
    sum_fee: float = 0.0        # sum of taker fee per contract
    sum_net_taker: float = 0.0  # realized (won - price - fee)
    sum_net_maker: float = 0.0  # realized (won - bid - maker_fee)  [assumes fill]

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.n if self.n else None

    @property
    def avg_pred(self) -> float | None:
        return self.sum_p / self.n if self.n else None

    @property
    def avg_price(self) -> float | None:
        return self.sum_price / self.n if self.n else None

    @property
    def net_taker_per_contract(self) -> float | None:
        return self.sum_net_taker / self.n if self.n else None

    @property
    def net_maker_per_contract(self) -> float | None:
        return self.sum_net_maker / self.n if self.n else None


def brier(rows: list[tuple[float, int]]) -> float | None:
    pts = [(p, y) for p, y in rows if p is not None and y in (0, 1)]
    if not pts:
        return None
    return mean((p - y) ** 2 for p, y in pts)


def reliability_table(rows: list[tuple[float, int]], bins: int = 10) -> list[dict]:
    """rows: (predicted_prob_for_selected_side, won). Returns per-bin calibration."""
    edges = [i / bins for i in range(bins + 1)]
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = [(p, y) for p, y in rows if p is not None and (lo <= p < hi or (i == bins - 1 and p == hi))]
        if not sel:
            out.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": 0, "pred": None, "actual": None})
            continue
        out.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": len(sel),
            "pred": mean(p for p, _ in sel),
            "actual": mean(y for _, y in sel),
        })
    return out


def bucketize_by_edge(records: list[dict], edges: list[float]) -> list[Bucket]:
    """Group decision records by stated taker edge into [edges[i], edges[i+1]) buckets.

    Each record must carry: p_side, won (0/1), ask (taker price), bid (maker price),
    fee_taker, fee_maker. Net EV is realized: payoff(=won) minus entry cost.
    """
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"[{edges[i]:+.3f},{edges[i+1]:+.3f})")
    buckets = [Bucket(label=lab) for lab in labels]
    for r in records:
        e = r["edge"]
        if e is None:
            continue
        idx = None
        for i in range(len(edges) - 1):
            if edges[i] <= e < edges[i + 1]:
                idx = i
                break
        if idx is None:
            continue
        b = buckets[idx]
        won = r["won"]
        b.n += 1
        b.wins += won
        b.sum_p += r["p_side"]
        b.sum_price += r["ask"]
        b.sum_fee += r["fee_taker"]
        b.sum_net_taker += won - r["ask"] - r["fee_taker"]
        b.sum_net_maker += won - r["bid"] - r["fee_maker"]
    return buckets


def group_stat(records: list[dict], key: str) -> dict[str, Bucket]:
    """Group records into one Bucket per distinct value of records[key]."""
    out: dict[str, Bucket] = {}
    for r in records:
        k = str(r.get(key))
        b = out.setdefault(k, Bucket(label=k))
        won = r["won"]
        b.n += 1
        b.wins += won
        b.sum_p += r["p_side"]
        b.sum_price += r["ask"]
        b.sum_fee += r["fee_taker"]
        b.sum_net_taker += won - r["ask"] - r["fee_taker"]
        b.sum_net_maker += won - r["bid"] - r["fee_maker"]
    return out
