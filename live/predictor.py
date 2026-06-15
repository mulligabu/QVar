"""RenTech-style online prediction stack for near-expiry binary outcome.

Architecture (per the decision-stack reframe — Markov is ONE head, Boltzmann is
the action/size policy, with calibration + reliability feedback in between):

    features
      -> move-probability head            P(tradable_move)
      -> specialist direction heads        each emits p_up + online reliability
           - z_distance head  (the bridge/Markov-style distance-to-strike signal)
           - momentum head
           - microstructure head (orderbook imbalance)
           - fair_value_residual head (model vs market mid; underlying-not-reflected)
      -> reliability-weighted ensemble      weight_i propto exp(EWMA log-score_i)
      -> online calibration (bucket recal)  honest p
      -> dynamic Boltzmann temperature      regime stability x calib-error x move-unc
      -> directional_edge = P(move)*(2p-1)  -> EV gate + size

Everything is causal/online. Each head is a 1-2 feature online logistic
(max-entropy/Gibbs) with running standardization. No look-ahead, no batch fit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

FEATURES = ["z_dist", "momentum", "realized_vol", "efficiency", "ob_imbalance", "log_tau"]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def entropy(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        v = math.sqrt(self.m2 / self.n) if self.n > 1 else 1.0
        return v if v > 1e-9 else 1.0


class Head:
    """Online logistic (Gibbs) direction head over a subset of features.

    Tracks an EWMA log-score reliability so the ensemble can weight it by recent
    out-of-sample performance.
    """

    def __init__(self, name: str, feats: list[str], w0: dict[str, float] | None = None,
                 b0: float = 0.0, lr: float = 0.04):
        self.name = name
        self.feats = feats
        self.w = {f: (w0 or {}).get(f, 0.0) for f in feats}
        self.b = b0
        self.lr = lr
        self.stats = {f: Welford() for f in feats}
        self.log_score = -0.6931  # EWMA of log P(correct); init ~ log(0.5)
        self.n = 0
        self._last_p = 0.5

    def _std(self, x: dict) -> dict:
        out = {}
        for f in self.feats:
            s = self.stats[f]
            out[f] = (x[f] - s.mean) / s.std if s.n > 5 else x[f]
        return out

    def predict(self, feats: dict) -> float:
        xs = self._std(feats)
        self._last_p = sigmoid(self.b + sum(self.w[f] * xs[f] for f in self.feats))
        return self._last_p

    @property
    def reliability(self) -> float:
        # map EWMA log-score (<=0) to (0,1]; log(0.5)=-0.69 -> ~0.5 baseline
        return math.exp(self.log_score)

    def learn(self, feats: dict, outcome_up: int):
        for f in self.feats:
            self.stats[f].update(feats[f])
        xs = self._std(feats)
        p = sigmoid(self.b + sum(self.w[f] * xs[f] for f in self.feats))
        err = p - outcome_up
        for f in self.feats:
            self.w[f] -= self.lr * err * xs[f]
        self.b -= self.lr * err
        pc = p if outcome_up == 1 else (1 - p)
        ls = math.log(min(max(pc, 1e-6), 1 - 1e-6))
        self.log_score = 0.97 * self.log_score + 0.03 * ls
        self.n += 1


class FairValueHead:
    """Fair-value residual head: anchors to the market mid and tilts by the
    underlying-not-reflected signal (z_dist). p_up = sigmoid(logit(mid) + k*z),
    with k learned online. Captures 'underlying moved, contract hasn't yet'."""

    def __init__(self, lr: float = 0.03):
        self.k = 0.5
        self.lr = lr
        self.zstat = Welford()
        self.log_score = -0.6931
        self.n = 0

    def predict(self, feats: dict, mid: float) -> float:
        mid = min(max(mid, 1e-4), 1 - 1e-4)
        z = (feats["z_dist"] - self.zstat.mean) / self.zstat.std if self.zstat.n > 5 else feats["z_dist"]
        return sigmoid(math.log(mid / (1 - mid)) + self.k * z)

    @property
    def reliability(self) -> float:
        return math.exp(self.log_score)

    def learn(self, feats: dict, mid: float, outcome_up: int):
        self.zstat.update(feats["z_dist"])
        p = self.predict(feats, mid)
        z = (feats["z_dist"] - self.zstat.mean) / self.zstat.std if self.zstat.n > 5 else feats["z_dist"]
        self.k -= self.lr * (p - outcome_up) * z
        self.k = max(0.0, min(4.0, self.k))
        pc = p if outcome_up == 1 else (1 - p)
        self.log_score = 0.97 * self.log_score + 0.03 * math.log(min(max(pc, 1e-6), 1 - 1e-6))
        self.n += 1


class ConditionalSkill:
    """Per-(timeframe, mid-band) tracker of the disagreement signal's DRIFT-CONTROLLED
    directional skill (2026-06-06 finding). Globally `sign(p_cal − mid)` looks
    anti-predictive (AUC 0.37) but that is a drift artifact; conditioned on the market's
    own mid-band the tilt predicts again (~0.65). For each cell we track the decayed
    realized up-rate WHEN THE MODEL LEANS UP (p_cal>mid) vs WHEN IT LEANS DOWN, on the
    SAME mid-band — so the gap (r_up − r_dn) is the within-cell discrimination with the
    market level controlled out:
        gap > 0  -> raw sign(p_cal−mid) predicts the residual  (trade the model's side)
        gap < 0  -> the INVERSE predicts                       (flip the side)
        gap ~ 0  -> no skill in this cell                      (abstain)
    The conditional engine trades only cells that have earned |gap| >= margin with enough
    samples on BOTH legs, flipping the side where the cell says so. EWMA decay keeps each
    cell tracking the current regime."""

    BANDS: ClassVar[list[tuple[float, float, str]]] = [(-1.0, 0.45, "lo"), (0.45, 0.55, "mid"), (0.55, 2.0, "hi")]
    # Cell blacklist mechanism: listed cells always ABSTAIN but still LEARN. Currently
    # EMPTY — a cell is only worth blocking with weeks of out-of-sample evidence, not days.
    BLACKLIST: ClassVar[set[str]] = set()

    def __init__(self, decay: float = 0.99, min_n: float = 20.0, margin: float = 0.08):
        self.decay, self.min_n, self.margin = decay, min_n, margin
        # key "tf|band" -> {"up": [n, sum_outcome_up], "dn": [n, sum_outcome_up]}
        self.cells: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"up": [0.0, 0.0], "dn": [0.0, 0.0]})

    def _key(self, tf: str, mid: float) -> str:
        band = next((nm for lo, hi, nm in self.BANDS if lo <= mid < hi), "hi")
        return f"{tf}|{band}"

    def learn(self, tf, mid, p_cal, outcome_up):
        if tf is None or mid is None or p_cal is None:
            return
        leg = "up" if p_cal > mid else "dn"
        c = self.cells[self._key(tf, mid)][leg]
        c[0] = c[0] * self.decay + 1.0
        c[1] = c[1] * self.decay + outcome_up

    def assess(self, tf, mid, p_cal) -> tuple[str, int, float, float]:
        """-> (status, side_factor, conviction, min_leg_n). side_factor: +1 trade the raw
        model side, -1 flip to the inverse, 0 abstain (warmup / no demonstrated skill)."""
        if mid is None or p_cal is None:
            return ("WARMUP", 0, 0.0, 0.0)
        key = self._key(tf, mid)
        if key in self.BLACKLIST:
            return ("BLACKLIST", 0, 0.0, 0.0)
        c = self.cells.get(key)
        if c is None:
            return ("WARMUP", 0, 0.0, 0.0)
        nu, wu = c["up"]; nd, wd = c["dn"]
        if min(nu, nd) < self.min_n:
            return ("WARMUP", 0, 0.0, min(nu, nd))
        gap = (wu + 0.5) / (nu + 1.0) - (wd + 0.5) / (nd + 1.0)
        if abs(gap) < self.margin:
            return ("NOSKILL", 0, abs(gap), min(nu, nd))
        return ("OK", (1 if gap > 0 else -1), abs(gap), min(nu, nd))

    def summary(self) -> list[dict]:
        out = []
        for k, c in self.cells.items():
            nu, wu = c["up"]; nd, wd = c["dn"]
            if min(nu, nd) < 1:
                continue
            r_up, r_dn = (wu + 0.5) / (nu + 1.0), (wd + 0.5) / (nd + 1.0)
            gap = r_up - r_dn
            earned = min(nu, nd) >= self.min_n and abs(gap) >= self.margin
            side = ("raw" if gap > 0 else "flip") if earned else ("warm" if min(nu, nd) < self.min_n else "—")
            if k in self.BLACKLIST:
                side = "blk"   # blacklisted: learns but never trades
            out.append({"cell": k, "n_up": round(nu, 0), "n_dn": round(nd, 0),
                        "r_up": round(r_up, 3), "r_dn": round(r_dn, 3), "gap": round(gap, 3),
                        "side": side})
        return sorted(out, key=lambda x: -abs(x["gap"]))

    def state(self) -> dict:
        return {k: {"up": v["up"], "dn": v["dn"]} for k, v in self.cells.items()}

    def load_state(self, d: dict) -> None:
        for k, v in (d or {}).items():
            self.cells[k] = {"up": list(v["up"]), "dn": list(v["dn"])}


def classify_regime(realized_vol: float, efficiency: float, vol_stats: Welford) -> str:
    hi = vol_stats.mean + 0.5 * vol_stats.std if vol_stats.n > 5 else realized_vol
    lo = vol_stats.mean - 0.5 * vol_stats.std if vol_stats.n > 5 else realized_vol
    if vol_stats.n > 10 and realized_vol > vol_stats.mean + 2.0 * vol_stats.std:
        return "panic"
    vol_state = "high_vol" if realized_vol >= hi else ("low_vol" if realized_vol <= lo else "mid_vol")
    trend = "trend" if efficiency >= 0.45 else ("mean_revert" if efficiency <= 0.20 else "chop")
    return f"{vol_state}/{trend}"


@dataclass
class Prediction:
    p_up: float
    p_cal: float
    move_prob: float
    directional_edge: float       # P(move)*(2p-1)
    fair_value_residual: float    # p_cal - market_mid
    regime: str
    temperature: float
    confidence: float
    head_outputs: dict
    head_weights: dict
    features: dict


class OnlinePredictor:
    """Specialist-head ensemble + move-prob + calibration + dynamic temperature."""

    def __init__(self, tilt_cap: float = 0.4):
        self.tilt_cap = tilt_cap   # max |logit nudge| over the market mid (per-engine)
        self.heads = {
            "z_markov": Head("z_markov", ["z_dist", "log_tau"], w0={"z_dist": 1.8}),
            "momentum": Head("momentum", ["momentum", "efficiency"], w0={"momentum": 0.4}),
            "microstructure": Head("microstructure", ["ob_imbalance"], w0={"ob_imbalance": 0.6}),
        }
        self.fair = FairValueHead()
        self.vol_stats = Welford()
        self.move_thresh_stat = Welford()   # realized |move|/window scale
        # vol-conditioned Bayesian (Dirichlet) regime transitions
        self.trans: dict[str, dict[str, float]] = {}
        self.prev_regime: str | None = None
        # online calibration buckets. DECAYING counts (not cumulative): each bucket
        # holds an exponentially-weighted [n, wins] with effective window ~1/(1-decay)
        # ≈ 200 samples. Cumulative (infinite-memory) buckets anchored p_cal to the
        # base rate of whatever regime dominated the run — so a calibrator trained in a
        # down tape stayed below the mid and could never bet up even after the market
        # turned. Decay makes p_cal track the CURRENT regime's base rate. See memory
        # main-engine-live-postmortem / edge-ensemble-vs-mid.
        self.cal_decay = 0.995
        self.cal = {i: [1.0, 0.5] for i in range(10)}
        # bucket-level realized edge feedback: (regime|tf) -> [n, wins]
        self.bucket: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.n_updates = 0
        self._calib_err = 0.1   # EWMA |p - outcome| proxy
        # drift-controlled per-(tf,mid-band) skill of the disagreement signal. Always
        # learned (cheap, harmless); only the conditional engine consults it to gate/flip.
        self.cond_skill = ConditionalSkill()

    def _move_prob(self, feats: dict) -> float:
        """P(tradable move): realized vol over remaining horizon vs a learned scale.

        sigma_tau (in z units) large -> more likely the window separates from strike
        enough to matter. Calibrated by the running distribution of move scales."""
        scale = feats.get("realized_vol", 0.0) * math.sqrt(max(1.0, math.exp(feats["log_tau"])))
        if self.move_thresh_stat.n > 10:
            zscore = (scale - self.move_thresh_stat.mean) / (self.move_thresh_stat.std or 1.0)
        else:
            zscore = 0.0
        return sigmoid(0.9 * zscore)

    def _regime_temperature(self, regime: str) -> tuple[float, float]:
        row = self.trans.get(self.prev_regime or "", {})
        total = sum(row.values())
        if total > 3:
            probs = [c / total for c in row.values()]
            te = -sum(p * math.log(p) for p in probs if p > 0) / math.log(max(2, len(probs)))
            stability = 1.0 - te
        else:
            stability = 0.3
        # dynamic temperature: hotter when unstable, when calibration is poor
        temperature = 1.0 + 2.0 * (1.0 - stability) + 3.0 * max(0.0, self._calib_err - 0.1)
        if regime.startswith("panic"):
            temperature *= 1.5
        return temperature, stability

    def predict(self, feats: dict, market_mid: float | None = None) -> Prediction:
        regime = classify_regime(feats["realized_vol"], feats["efficiency"], self.vol_stats)
        temperature, stability = self._regime_temperature(regime)
        # heads
        outs = {name: h.predict(feats) for name, h in self.heads.items()}
        if market_mid is not None:
            outs["fair_value"] = self.fair.predict(feats, market_mid)
        # reliability weights (softmax of EWMA log-scores)
        rel = {name: h.reliability for name, h in self.heads.items()}
        if market_mid is not None:
            rel["fair_value"] = self.fair.reliability
        zsum = sum(rel.values()) or 1.0
        weights = {k: v / zsum for k, v in rel.items()}
        # ensemble in log-odds space, temperature-scaled
        logit = 0.0
        for k, p in outs.items():
            p = min(max(p, 1e-6), 1 - 1e-6)
            logit += weights[k] * math.log(p / (1 - p))
        ens_logit = logit / temperature
        # MARKET-ANCHORED bounded tilt: the market mid is a strong prior (these
        # markets are efficient). We only nudge it by our edge, capped — never
        # wholesale-disagree. Prevents warmup-overfit heads producing absurd
        # "48-cent EV" trades that fight a liquid market. Edge = small bounded tilt.
        if market_mid is not None:
            mid = min(max(market_mid, 1e-4), 1 - 1e-4)
            mkt_logit = math.log(mid / (1 - mid))
            # cap shrunk 0.8->0.4 (2026-06-03): the ensemble was OVERCONFIDENT on the
            # up side in every reliability bin — confident-wrong UP calls drove the
            # Kelly blowup. A tighter tilt both de-biases p_cal and shrinks the implied
            # stake. Per-engine via tilt_cap (candle engine uses 0.25 — logit is
            # steepest near 0.5). See memory edge-ensemble-vs-mid / deferred-engine-fixes.
            tilt = max(-self.tilt_cap, min(self.tilt_cap, ens_logit - mkt_logit))
            p_up = sigmoid(mkt_logit + tilt)
        else:
            p_up = sigmoid(ens_logit)
        # online calibration (decaying decile buckets, track the current regime)
        b = min(9, int(p_up * 10))
        n, wins = self.cal[b]
        p_cal_bucket = (wins + 0.5) / (n + 1.0)
        wts = min(1.0, n / 30.0)
        p_cal = wts * p_cal_bucket + (1 - wts) * p_up
        move_prob = self._move_prob(feats)
        directional_edge = move_prob * (2 * p_cal - 1)
        fvr = (p_cal - market_mid) if market_mid is not None else 0.0
        conf = (1.0 - entropy(p_cal)) * (0.4 + 0.6 * stability) * (0.5 + 0.5 * move_prob)
        return Prediction(p_up=p_up, p_cal=p_cal, move_prob=move_prob,
                          directional_edge=directional_edge, fair_value_residual=fvr,
                          regime=regime, temperature=temperature, confidence=conf,
                          head_outputs=outs, head_weights=weights, features=feats)

    def learn(self, pred: Prediction, outcome_up: int, market_mid: float | None = None,
              bucket_key: str | None = None):
        feats = pred.features
        self.vol_stats.update(feats["realized_vol"])
        scale = feats.get("realized_vol", 0.0) * math.sqrt(max(1.0, math.exp(feats["log_tau"])))
        self.move_thresh_stat.update(scale)
        for h in self.heads.values():
            h.learn(feats, outcome_up)
        if market_mid is not None:
            self.fair.learn(feats, market_mid, outcome_up)
        # calibration buckets (exponentially-weighted so they track the current regime)
        b = min(9, int(pred.p_up * 10))
        d = self.cal_decay
        self.cal[b][0] = self.cal[b][0] * d + 1.0
        self.cal[b][1] = self.cal[b][1] * d + outcome_up
        self._calib_err = 0.97 * self._calib_err + 0.03 * abs(pred.p_cal - outcome_up)
        # regime transitions
        if self.prev_regime is not None:
            self.trans.setdefault(self.prev_regime, {})
            self.trans[self.prev_regime][pred.regime] = self.trans[self.prev_regime].get(pred.regime, 0.0) + 1.0
        self.prev_regime = pred.regime
        # bucket-level realized edge feedback
        if bucket_key is not None:
            won = outcome_up if pred.p_cal >= 0.5 else (1 - outcome_up)
            self.bucket[bucket_key][0] += 1
            self.bucket[bucket_key][1] += won
        self.n_updates += 1

    # ---- persistence (survive restarts/crashes over a multi-day run) --------
    def state(self) -> dict:
        def wf(s: Welford):
            return {"n": s.n, "mean": s.mean, "m2": s.m2}
        return {
            "heads": {n: {"w": h.w, "b": h.b, "log_score": h.log_score, "n": h.n,
                          "stats": {f: wf(s) for f, s in h.stats.items()}}
                      for n, h in self.heads.items()},
            "fair": {"k": self.fair.k, "log_score": self.fair.log_score, "n": self.fair.n,
                     "zstat": wf(self.fair.zstat)},
            "vol_stats": wf(self.vol_stats), "move_thresh_stat": wf(self.move_thresh_stat),
            "trans": self.trans, "prev_regime": self.prev_regime,
            "cal": {str(k): v for k, v in self.cal.items()},
            "bucket": {k: v for k, v in self.bucket.items()},
            "calib_err": self._calib_err, "n_updates": self.n_updates,
            "cond_skill": self.cond_skill.state(),
        }

    def load_state(self, d: dict) -> None:
        def setwf(s: Welford, sd: dict):
            s.n, s.mean, s.m2 = sd["n"], sd["mean"], sd["m2"]
        for n, hd in d.get("heads", {}).items():
            if n in self.heads:
                h = self.heads[n]
                h.w.update(hd["w"]); h.b = hd["b"]; h.log_score = hd["log_score"]; h.n = hd["n"]
                for f, sd in hd.get("stats", {}).items():
                    if f in h.stats:
                        setwf(h.stats[f], sd)
        if "fair" in d:
            self.fair.k = d["fair"]["k"]; self.fair.log_score = d["fair"]["log_score"]
            self.fair.n = d["fair"]["n"]; setwf(self.fair.zstat, d["fair"]["zstat"])
        if "vol_stats" in d:
            setwf(self.vol_stats, d["vol_stats"])
        if "move_thresh_stat" in d:
            setwf(self.move_thresh_stat, d["move_thresh_stat"])
        self.trans = {k: dict(v) for k, v in d.get("trans", {}).items()}
        self.prev_regime = d.get("prev_regime")
        # clamp loaded bucket counts to the effective window so decay takes effect
        # immediately after a restart (older state has huge cumulative counts that
        # would otherwise take thousands of updates to forget). Preserve the rate.
        cap = 1.0 / (1.0 - self.cal_decay)
        def _clamp(n, wins):
            if n > cap:
                wins = wins * (cap / n)
                n = cap
            return [n, wins]
        for k, v in d.get("cal", {}).items():
            n, wins = v
            self.cal[int(k)] = _clamp(n, wins)
        for k, v in d.get("bucket", {}).items():
            self.bucket[k] = v
        self._calib_err = d.get("calib_err", 0.1)
        self.n_updates = d.get("n_updates", 0)
        self.cond_skill.load_state(d.get("cond_skill", {}))

    @property
    def weights_summary(self) -> dict:
        out = {f"{n}.{f}": round(v, 3) for n, h in self.heads.items() for f, v in h.w.items()}
        out["fair.k"] = round(self.fair.k, 3)
        return out


# ---- feature builder (unchanged contract; causal) -------------------------
def build_features(price_hist, spot: float, strike: float, end_ts: float, now: float,
                   ob_imbalance: float) -> dict | None:
    from collections import deque  # noqa
    tau_s = max(1.0, end_ts - now)
    tau_min = tau_s / 60.0
    pts = [(t, p) for t, p in price_hist if t <= now]
    if len(pts) < 5 or strike <= 0 or spot <= 0:
        return None
    prices = [p for _, p in pts]
    rets = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] > 0]
    if len(rets) < 4:
        return None
    n = len(rets)
    mean = sum(rets) / n
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(1, n - 1)) or 1e-6
    sigma_tau = vol * math.sqrt(max(1.0, tau_s / max(1.0, (pts[-1][0] - pts[0][0]) / max(1, n))))
    z = math.log(spot / strike) / (sigma_tau or 1e-6)
    momentum = (prices[-1] - prices[0]) / prices[0]
    net = abs(prices[-1] - prices[0])
    path = sum(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices)))
    efficiency = net / path if path > 0 else 0.0
    return {
        "z_dist": z, "momentum": momentum * 100, "realized_vol": vol * 100,
        "efficiency": efficiency, "ob_imbalance": ob_imbalance, "log_tau": math.log(tau_min + 1e-6),
        "_spot": spot, "_strike": strike, "_tau_min": tau_min,
    }
