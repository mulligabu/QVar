# QVar — System Overview

**What this is.** A live, forward-running (NOT backtest) paper/ghost trading system for
**short-dated crypto binary options** — "will ASSET be up / above STRIKE at time T?" — listed on
**Polymarket** and **Kalshi**, settling on **Chainlink** on-chain price. It runs **two engine
"lanes"** off one shared codebase (a control and a conditional-gated variant), plus a shared
market-data service and a dashboard.

**The honest bottom line (read this first).** After extended live measurement, the **direction
of these markets is efficient** — the model does **not** out-predict the market mid on which side
wins (disagreement-AUC ≈ 0.50 out-of-sample). The one signal that replicates is **calibration**
(the model prices *confident* markets slightly better in Brier score, not direction). The program
therefore emphasizes **execution quality, liquidity/calibration, and infrastructure** rather than
directional alpha. The math below is the machinery; the verdict above is what the machinery has
actually shown.

File:line references are to `live/` — verify before relying on exact line numbers.

---

## 1. End-to-end data flow (one engine cycle, ~8s)

```
            ┌─────────────── md_service.py (1 process, WS-first) ────────────────┐
            │  Polymarket CLOB WSS books/trades · Coinbase/Binance spot WS       │
            │  Kalshi REST poll · Chainlink batched RPC + AnswerUpdated logs     │
            │  60s discovery sweep · → UDS data/md.sock + md.db snapshot         │
            └───────────────────────────────┬───────────────────────────────────┘
                                            │  (lanes consume this with REST fallback)
   per lane (forward_engine.py .step()):    ▼
   1. sample_feeds()      Chainlink-anchored composite price → rolling history
   2. record strikes      underlying price AT each expiry-window boundary
   3. discover()          all live markets across venues / assets / expiries
   4. build_features()    causal features from price history + book
   5. predictor.predict() specialist-head ensemble → calibrated p_up (p_cal)
   6. structural anchor   digital prob α=N(d2) (logged, shadow)
   7. evaluate_trade()    EV-gate after costs → ghost LIMIT order
   8. fill / settle       resting order fills if book trades through; settle on Chainlink
   9. learn()             every settled market updates the online model (no look-ahead)
```

Everything is **online and causal** — no batch fit, no look-ahead. State (model weights,
calibration, ledger) persists to disk and **resumes** across restarts.

---

## 2. The data / feed layer

### 2.1 Price feeds (`live/feed.py`)
- **ChainlinkFeed** — the **settlement-grade** reference. Reads the on-chain Arbitrum
  aggregator's `latestRoundData()` via JSON-RPC `eth_call`. Polymarket's crypto markets resolve
  on Chainlink, so pricing/settling against it removes the ~4% Binance-candle basis. Returns
  `(price, updated_at)`; rounds post only on ~0.05% deviation, so they can be sparse.
- **CoinbaseFeed / BinanceFeed** — fast spot ticks to fill the gaps *between* Chainlink rounds.
- **CompositeFeed** — the price the model sees. **Anchors** to the last Chainlink round and adds
  the **Coinbase drift since that anchor**: `price = chainlink_anchor + (coinbase_now −
  coinbase_at_anchor)`. Re-anchors only when Chainlink posts a *new* round (keyed on
  `updated_at`), so the model sees live sub-second drift while staying tied to the settlement
  source.
- **FeedHub** — samples **all assets in parallel** in one batched Chainlink RPC + concurrent
  exchange ticks. `staleness()` reports seconds since each asset's last Chainlink round.

### 2.2 Pooled HTTP (`live/net.py`)
Per-(thread, host) keep-alive `HTTPSConnection` pool with per-host latency/error counters (the
"API error budget" in status.json). Stdlib only, with a one-shot urllib fallback.

### 2.3 Shared market-data service (`live/md_service.py`)
One asyncio process owns **all external market I/O**, WebSocket-first, republished over a
Unix-domain pub/sub socket + SQLite snapshot. Engines consume it via stdlib adapters
(`live/md_client.py`) behind `--md on`, with automatic REST fallback. See §10.

---

## 3. Market discovery & venues (`live/venues.py`)

Normalizes both venues to an **UP-token view** `(up_bid, up_ask, sizes, window)`:
- **PolymarketClient** — `discover()` finds the current up/down market per (asset, timeframe);
  slug `{asset}-updown-{tf}-{epoch}` where `epoch` = window open. `discover_hourly()` finds the
  hourly **threshold ladder** (BTC/ETH). `book_top()` reads the CLOB `/book`.
- **KalshiClient** — 15m directional + hourly/daily threshold ladders. **UP ask = 1 − best NO bid.**

**Two market kinds:**
- **directional** — strike = the underlying price **at the window open**, observed forward.
- **threshold** — strike is a fixed level in the market metadata (tradeable immediately).

---

## 4. Feature engineering (`build_features`, predictor.py)

From the causal price history + book imbalance, all standardized online:

| Feature | Definition | Captures |
|---|---|---|
| **z_dist** | `log(spot/strike) / σ_τ` | distance to strike in **σ units over the remaining horizon** — the dominant signal. |
| **momentum** | `(p_last − p_first)/p_first` ×100 | recent directional drift |
| **realized_vol** | stdev of per-step log returns ×100 | the vol state |
| **efficiency** | `|net move| / Σ|step moves|` | trend (→1) vs chop/mean-revert (→0) |
| **ob_imbalance** | `(bid_sz − ask_sz)/(bid_sz + ask_sz)` | order-book pressure |
| **log_tau** | `log(minutes to expiry)` | time-decay scaling |

`z_dist` is the "distance-to-barrier" signal: how many standard deviations the spot sits from the
strike given the time left. A binary's fair value is essentially a function of this distance.

---

## 5. The prediction stack (`live/predictor.py`)

A **RenTech-style online decision stack**: specialist "heads" each emit an up-probability and a
self-rated reliability; they're combined in log-odds, anchored to the market, calibrated, and
temperature-scaled.

### 5.1 Specialist heads (`Head`)
Each head is an **online logistic regression** (one-layer max-entropy / Gibbs classifier) over
1-2 standardized features, trained by SGD on every settled outcome:
- **z_markov** — features `[z_dist, log_tau]`, init `w[z_dist]=1.8`. The dominant directional head.
- **momentum** — `[momentum, efficiency]`.
- **microstructure** — `[ob_imbalance]`.
- **FairValueHead** — anchors to the market mid and tilts by z_dist: `p = sigmoid(logit(mid) +
  k·z)`, `k` learned online (clamped 0-4). "Underlying moved, the contract hasn't repriced yet."

**Standardization** uses **Welford's** online algorithm. **Reliability** = `exp(EWMA of
log-score)` — an exponentially-weighted average of `log P(it called correct)`; a head right
recently gets more weight. This is the online out-of-sample score, not training accuracy.

### 5.2 Ensemble
Reliability-weighted average **in log-odds space**, divided by a **temperature**:
```
ens_logit = ( Σ_i  weight_i · logit(p_i) ) / temperature      weight_i = reliability_i / Σ reliability
```

### 5.3 Market-anchored bounded tilt (the key discipline)
These markets are efficient, so the model does **not** wholesale-disagree with the mid. It takes
the market log-odds as a **prior** and nudges by its edge, **capped** at `±tilt_cap` (0.4):
```
tilt  = clip(ens_logit − mid_logit, −tilt_cap, +tilt_cap)
p_up  = sigmoid(mid_logit + tilt)
```
This de-biases the (historically over-confident) ensemble and shrinks the implied stake. Without
the cap, warmup-overfit heads produced absurd "48-cent EV" trades fighting a liquid market.

### 5.4 Online calibration (`cal` buckets)
`p_up` is mapped through 10 **decile buckets** holding a **decaying** `[n, wins]` (EWMA,
effective window ≈ 200 samples). `p_cal = w·bucket_rate + (1−w)·p_up`, blending in as the bucket
fills. **Why decaying, not cumulative:** an infinite-memory calibrator trained in a down tape
stayed permanently below the mid and could never bet up even after the market turned. Decay makes
`p_cal` track the **current** regime's base rate.

### 5.5 Move-probability head
`P(tradable move)` = `sigmoid(0.9 · z-score of (realized_vol · √remaining_horizon))` vs a learned
distribution of move scales. Low when the window is unlikely to separate from the strike. Feeds
`directional_edge = P(move)·(2·p_cal − 1)` and the confidence score.

### 5.6 Dynamic temperature (Markov regime entropy)
Temperature **hots up** (shrinks the edge toward the market) when the regime is unstable or
calibration is poor:
```
temperature = 1 + 2·(1 − stability) + 3·max(0, calib_err − 0.1)     (×1.5 in "panic")
```
`stability = 1 − normalized_entropy(regime-transition row)`. The **regime transition matrix** is a
**vol-conditioned Markov chain**: `trans[prev_regime][next_regime]` are **Dirichlet**
pseudo-counts. When the chain says the current regime reliably persists (low transition entropy),
`stability→1` and the model is allowed to be more confident. This is the literal Markov chain in
the system — over *regimes*, not prices.

### 5.7 Regime classification
`classify_regime(realized_vol, efficiency)` → `"{vol_state}/{trend}"`, e.g. `high_vol/trend`,
`mid_vol/chop`, or `panic` (vol > mean + 2σ).

---

## 6. The structural anchor — Boltzmann / quadratic-variation math (`live/structural.py`)

A second, **stochastic-calculus-grounded** probability estimate, kept in **SHADOW mode** (logged
as `alpha_up`, does not gate). Production port of two research prototypes
(`research/boltzmann_upgrade.md`, `research/qv_markov_upgrade.md`).

### 6.1 Digital probability α = N(d2) (the "Boltzmann alpha")
Our payoff is **digital** (1 iff S_T crosses strike), so the only structural quantity needed is
the Gaussian digital probability:
```
α = P(S_T > k) = N(d2),   d2 = (log(S/k) + τ·μ_s) / (σ_s·√τ)
```
- S = spot, k = strike, τ = **seconds** to expiry, μ_s / σ_s = **per-second** log-return drift /
  vol. All timeframe scaling lives in τ (mean ∝ τ, vol ∝ √τ) — the Bachelier/Black-Scholes digital.
- This is the **Boltzmann paper's α coefficient** (verified there to be identically 1 − N(d2)).
  The name "Boltzmann" comes from that paper's max-entropy / Gibbs-measure derivation of the
  terminal price distribution; for a digital the relevant moment collapses to N(d2).

### 6.2 Noise-robust volatility via **quadratic variation** (the QV / Markov estimator)
σ_s is **not** a naive realized vol. Microstructure noise inflates naive RV (`Σ r²`), so we use a
**lag-1 autocovariance correction**:
```
naive RV          = Σ r_i²
AC1-corrected IV  = RV + 2·n·γ₁        where γ₁ = lag-1 autocovariance of returns
σ_s = √(IV / Σdt)                      (floored at 0.25·RV so a jump window can't go ≤0)
```
- This is the **Zhou (1996)** estimator, which equals the **Hansen–Horel (2009) two-state /
  order-1 Markov quadratic-variation estimator** in closed form — the only member of that family
  estimable on our short, irregular, continuous-float composite feed. It de-inflates the
  observation noise that would otherwise make σ too high and α too timid.
- A **Bartlett realized kernel** (q lags, PSD by construction) is also implemented; AC1 (q≤1)
  validated better at our window lengths.
- **Quadratic variation** is the probabilistic object: for a continuous semimartingale the QV *is*
  the integrated variance; these estimators consistently estimate `[X]_T` from discretely/noisily
  sampled prices. That integrated variance feeds σ_s·√τ in d2. *(The project is named for it.)*

### 6.3 Terminal-variance regime shaper κ (Boltzmann Eq. 20 motivation)
```
κ = Var(last 2w returns) / Var(last w returns)
```
- κ ≈ 1 → **diffusion**; κ < 1 → **concentration / pinning**; κ > 1 → **expansion / breakout**.
  Applied only as a **mild ±25% shaping** of σ via `√(clip(κ, 0.6, 1.6))` — it *shapes* terminal
  dispersion, it is **not** a regime switch.

### 6.4 Shrinkage anchor + divergence-widened gate (`structural_anchor`)
Blends the ensemble prob toward α in log-odds, **harder the more they diverge**, and widens the
required EV margin with divergence:
```
λ = min(0.85, 0.35 + 6·|p_ens − α|·0.35)
p_anchored = sigmoid( (1−λ)·logit(p_ens) + λ·logit(α) )
gate       = min(0.12, 0.01 + 0.40·|p_ens − α|)
```

### 6.5 Status: SHADOW only
α was found to be **overconfident in every probability bin** (its low Brier was drift+extremeness,
not skill). So `structural_anchor` no longer gates — `alpha_up`, `struct_div`, `sigma_s`, `kappa`
are **logged** for the calibration A/B and nothing else.

---

## 7. The conditional-skill tracker (`ConditionalSkill`, predictor.py)

A **drift-controlled** measure of whether the model's disagreement with the market predicts the
residual outcome. Globally `sign(p_cal − mid)` looks anti-predictive (a drift artifact);
conditioned on the market's own **mid-band** (lo <0.45, mid 0.45-0.55, hi >0.55) × **timeframe**,
for each cell it tracks the **decayed realized up-rate when the model leans up** vs **when it
leans down**, same band:
```
gap = r_up − r_dn      (Laplace-smoothed, EWMA-decayed)
gap > 0 → trade the model's side ;  gap < 0 → FLIP ;  gap ≈ 0 → abstain
```
The **conditional** lane gates/flips on cells that have earned `|gap| ≥ 0.08` with ≥20 samples on
**both** legs; side and EV come from the **cell**, not raw `p_cal − mid`. A **blacklist** mechanism
exists (cells that always abstain but still learn) — currently empty.

---

## 8. Decision, sizing, execution & settlement (`forward_engine.py`)

### 8.1 The trade gate (`evaluate_trade`)
A market becomes a TRADE only if **all** hold:
- model **warmed** (≥40 updates) and `|fair_value_residual| ≤ 0.20`
- **price-discipline band**: `price_lo ≤ price ≤ price_hi` (per lane)
- **z_dist gate**: never bet against `sign(z_dist)` (don't bet down when spot is already above
  strike).
- **conditional gate** (conditional lane): the cell must have earned skill (may flip the side).
- **EV after costs** `> min_ev`, `confidence > 0.05`, `move_prob > 0.35`, side allowed by `--sides`.
- not blocked by the **control plane** (estop / halt / max-loss trip / exposure cap).

Every evaluated market — traded or not — is **shadow-learned** at settlement (the model learns
from all outcomes, the ledger only from trades).

### 8.2 Sizing
- **Flat fraction-of-equity** (`flat_size`, default on): stake = `risk_cap`·equity (2%). Replaced
  Kelly because **Kelly edge-scaling oversized losers** (`|p_cal−mid|` doesn't predict realized PnL).
- **Two-leg execution**: a **passive** limit resting at mid (2% budget) + a conviction-scaled
  **aggressive taker** crossing at the ask (1% budget), the taker firing only when
  `edge > spread/2 + venue_fee`.

### 8.3 Fill model (honest adverse selection)
A resting limit fills **only when the book trades through it** (`GhostPosition.fill_check`): buy
UP@mid fills when the ask drops to your price; buy DOWN fills when the up-bid rises to your
reference mid. Fills are **adverse-selected** (you're filled when price comes to you) — honest,
because a naive "filled at mid" overstates maker performance.

### 8.4 Settlement & reconciliation
At expiry, settle on the **Chainlink** price (`cl_spot`): `outcome_up = end_price > strike`. PnL
per leg = `(won − entry_price − fee)·size`. **Reconciliation** compares our Chainlink settle to
the venue-reported outcome (`recon = match / MISMATCH / n/a`). Polymarket charges **no** trading
fee; Kalshi ≈ `0.072·p·(1−p)`.

### 8.5 Ghost take-profit overlay + pivot logging
A **shadow ledger** mirroring each real entry but exiting at `tp_level` (0.90) if our-side bid
touches it before expiry. The engine also logs the full **pivot state** of every position
(peak/trough sellable price, mins-left at the peak, spread and book sizes at the turn) so the
optimal exit can be swept offline. `paths` = the time-resolved book/underlying path per position.

---

## 9. Risk control plane (`live/control.py`)

A shared switchboard (`data/CONTROL.json`, re-read every cycle by every lane) + latched trips:
- **estop** — full freeze, **auto-trips** on 5 consecutive cycle errors (crash-loop guard).
- **halt_entries** — soft: block new entries, keep settling/learning.
- **max-loss breaker** — latches `data/TRIPPED.json` when all-lane day PnL breaches the limit.
- **cross-lane exposure cap** — committed $ across lanes vs combined equity, total + per-asset.
- Dashboard toggles + flashing alarm banner; blocked windows still log the would-have-traded
  signal so they stay counterfactually measurable. A `health` table persists cycle latency, API
  error budget, and feed staleness once/minute.

---

## 10. md_service (`live/md_service.py`, `live/md_client.py`)

One asyncio process owns all external market I/O, **WebSocket-first**, republished over a
Unix-domain pub/sub socket (`data/md.sock`, NDJSON `{seq, ts_src, ts_recv, topic, payload}`,
snapshot-on-subscribe, seq-gap → resync) plus a SQLite snapshot (`data/md.db`):
- **Polymarket** CLOB WSS books (`book`/`price_change`) + trade prints; **Coinbase/Binance** spot
  WS; **Kalshi** REST poll; **Chainlink** batched `latestRoundData` poll + `eth_getLogs` on the
  aggregators' `AnswerUpdated` events → the **exact round sequence** bracketing every window
  boundary → one **authoritative, round-exact strike table** shared by all lanes. 60s discovery
  sweep for everyone.
- Engines consume via **stdlib** adapters (`MDMirror`/`MDFeedHub`) behind `--md on`, with
  **automatic REST fallback** within one cycle if the stream dies (control safety). A lane's
  status.json `health.md` reports transport, fallbacks, and books served from the mirror vs REST.

---

## 11. The two live lanes (`live/engines.conf` — single source of truth)

Both lanes run the **same code**, differing only by config. `main` is the untouched CONTROL —
never change its behavior. Configs are pinned in `engines.conf` (never hand-typed); a
CONFIG-DRIFT banner fires if a relaunch changes lane flags.

| Lane | Band | Entry | Distinguishing config | Role |
|---|---|---|---|---|
| **main** | 0.50-0.65 | maker | plain `sign(p_cal−mid)` gate, zdist-gate on | **CONTROL.** The fixed reference for every A/B. |
| **conditional** | 0.50-0.60 | maker | `--conditional-gate on` (cell-skill flip), capped 0.60 | trades only earned (timeframe, mid-band) cells; the drift-controlled-skill lane. |

---

## 12. What the live experiment has established

The system is run as an experiment; results are reported as they are, including negatives — which
is the point of keeping a fixed control lane:

1. **Direction is efficient.** `sign(p_cal − mid)` does not predict the residual outcome OOS
   (disagreement-AUC ≈ 0.50), under cluster-robust, drift-controlled testing.
2. **The replicated edge is calibration**, not direction — the model prices *confident* markets
   slightly better in Brier score (sharpness), a narrower claim than side-picking.
3. **The real bleed is execution** — maker-at-mid is adverse-selected. The event-driven
   market-data service exists to measure and remove that; liquidity provision plus calibration is
   the surviving thesis.

**Map of the code:** `feed.py` (price), `net.py` (HTTP pool), `venues.py` (markets/books),
`predictor.py` (online model + heads + calibration + Markov regime temperature + conditional
skill), `structural.py` (Boltzmann α / QV vol / κ), `forward_engine.py` (loop, gate, sizing,
fills, settle, TP, control), `control.py` (risk plane), `db.py` (SQLite stores), `md_service.py`
+ `md_client.py` (market-data service), `dashboard_server.py` + `dashboard.html` (:8011).
Foundational research is in `research/strategy/` and `research/edge_proof/`.
