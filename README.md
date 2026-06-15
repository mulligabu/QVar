# QVar

### Forward prediction engine — short-dated crypto binary markets

> *Maxwell's Demon stands at a gate in a gas at equilibrium, observes each molecule, and opens
> the door selectively — extracting usable order from pure noise. It isn't free: the demon pays
> in measurement. So does this. The market's direction is ~a coin flip (efficient); QVar doesn't
> bet on it. It **measures** (calibration, realized volatility via quadratic variation, regime)
> and **gates** (z-distance gate, conditional-skill gate, kill switches) to skim a thin,
> paid-for, real edge out of equilibrium.*

A live, **forward-running** (paper/ghost) research and execution system for short-dated
crypto **binary options** on **Polymarket** and **Kalshi** — markets of the form *"will ASSET
be up / above STRIKE at time T?"* across BTC, ETH, SOL, XRP, DOGE, BNB, HYPE on the 5m / 15m /
1h / 1D horizons, settling on **Chainlink** on-chain price.

It is **not a backtest.** Every component runs against live venue and on-chain data in real
time: an online learner predicts each market's settlement near expiry, rests passive limit
orders, settles against the Chainlink round that the venue itself resolves on, and updates its
model from the realized outcome — all causal, no look-ahead, with state that persists and
resumes across restarts.

The system runs **two strategy lanes in parallel** off one codebase — one is an untouched
**control** — so every configuration change is a live A/B test against a fixed reference rather
than a backtest narrative.

> **Full technical walkthrough:** [`research/audit/SYSTEM_OVERVIEW.md`](research/audit/SYSTEM_OVERVIEW.md)
> — feeds, the prediction stack, the structural pricer, execution, risk, and the empirical
> verdicts, grounded line-by-line in the code.

---

## Architecture

```
  ┌──────────────── market-data service (md_service.py, asyncio, WS-first) ─────────────┐
  │  Polymarket CLOB WSS books + trade prints · Coinbase/Binance spot WS                │
  │  Kalshi REST · Chainlink batched RPC + AnswerUpdated logs → round-exact strikes     │
  │  60s discovery sweep → Unix-domain pub/sub (data/md.sock) + SQLite snapshot         │
  └───────────────────────────────────┬────────────────────────────────────────────────┘
                                       │  stdlib adapters, automatic REST fallback
        engine lanes (forward_engine.py), each ~1 cycle:
        feed → strike capture → discover → causal features → online ensemble
             → structural digital-prob anchor → EV gate → ghost fill → settle → learn
                                       │
                          risk control plane (control.py): estop · entry-halt
                          · max-loss breaker · cross-lane exposure cap
                                       │
                          SQLite analytics (db.py) → dashboard (:8011)
```

**Design principles.** The decision path is **stdlib-only** (urllib/http.client, sqlite3) — no
runtime dependencies, no hidden state. Standalone services (the market-data service) may use
`websockets`/`httpx`, but the trading loop never does. The market mid is treated as a **strong,
near-efficient prior**; the model only tilts it within a bounded edge. The control lane's
behavior is **sacred** — it is never changed, so it remains a valid baseline.

---

## Mathematical foundations

The predictor is an **online decision stack** in the spirit of a reliability-weighted expert
ensemble. All estimators are streaming (Welford moments, EWMA scores, decaying buckets) and
update on each settled outcome.

**Specialist direction heads** (`predictor.py`). Each is an online **logistic / max-entropy
(Gibbs)** classifier over 1–2 standardized features, trained by stochastic gradient descent:
- **z-distance head** — the dominant signal: `z = log(S/K) / σ_τ`, the spot-to-strike distance in
  standard deviations over the remaining horizon (a binary's fair value is essentially a function
  of this barrier distance).
- **momentum**, **microstructure (order-book imbalance)**, and a **fair-value-residual** head that
  anchors to the market mid and tilts by z (`p = σ(logit(mid) + k·z)`, k learned online).

**Ensemble & calibration.** Heads combine in **log-odds**, weighted by each head's reliability
(`exp` of an EWMA out-of-sample log-score), divided by a **dynamic temperature**, then mapped
through **decaying decile calibration buckets** (effective window ≈ 200 samples, so the calibrator
tracks the *current* regime's base rate rather than anchoring to a stale one). The result is
nudged off the market mid by a **bounded tilt** (`±0.4` in log-odds) — the model never
wholesale-disagrees with a liquid market.

**Markov regime conditioning.** A vol-conditioned **Markov chain over market regimes** (volatility
× path-efficiency states) is maintained as **Dirichlet** transition counts. The normalized entropy
of the current regime's transition row sets the ensemble temperature: when the chain says the
regime persists, the model is permitted more confidence; in unstable or "panic" regimes the edge
is shrunk toward the market.

**Structural digital-option pricer** (`structural.py`, shadow mode). A second, stochastic-calculus
estimate of the up-probability for cross-checking calibration:
- **Digital probability** `α = P(S_T > K) = N(d₂)` with `d₂ = (log(S/K) + τμ) / (σ√τ)` — the
  Bachelier/Black-Scholes digital, equal to the *Boltzmann* terminal-pricing α coefficient
  (`1 − N(d₂)`).
- **Noise-robust volatility via quadratic variation.** σ is estimated with a **lag-1
  autocovariance (Zhou 1996) correction** to realized variance — equivalently the **Hansen–Horel
  order-1 Markov quadratic-variation estimator** in closed form — which de-inflates the
  microstructure noise that naive `Σr²` carries on a high-frequency composite feed. A Bartlett
  realized kernel is also implemented. *(This estimator is what the project is named for.)*
- **Terminal-variance regime shaper** `κ = Var(2w) / Var(w)`: κ<1 concentration/pinning, ≈1
  diffusion, >1 expansion — applied as a mild ±25% shaping of terminal dispersion (the
  diffusion-vs-concentration multiplier), not a regime switch.

**Conditional-skill tracker.** A drift-controlled measure of whether the model's *disagreement*
with the market predicts the residual outcome, tracked per (timeframe × mid-band) cell with
Laplace-smoothed, EWMA-decayed up-rates — the basis for the conditional lane's gating.

---

## Execution & risk

- **Honest fill model.** A resting limit fills only when the book *trades through it* — fills are
  deliberately adverse-selected (you are filled when price comes to you), so maker performance is
  never overstated.
- **Two-leg sizing.** A passive limit at mid plus a conviction-scaled aggressive taker leg that
  fires only when the edge clears the cross cost (`spread/2 + venue fee`). Default sizing is a
  **flat fraction of equity** (Kelly edge-scaling was found to oversize losers).
- **Settlement & reconciliation.** Settled on the Chainlink price the venue resolves on; every
  settlement is reconciled against the venue-reported outcome and flagged on mismatch.
- **Risk control plane** (`control.py`, re-read every cycle by every lane): emergency stop (with
  auto-trip on a crash loop), soft entry-halt, an all-lane **max-loss breaker** (latched), and a
  **cross-lane exposure cap** (total and per-asset) — all dashboard-toggleable, with blocked
  windows still logged counterfactually.

---

## Repository layout

| Path | Contents |
|---|---|
| `live/feed.py` | Chainlink on-chain aggregators + Coinbase/Binance ticks; Chainlink-anchored composite; parallel `FeedHub`. |
| `live/net.py` | Stdlib keep-alive HTTP pool with per-host latency/error budget. |
| `live/venues.py` | Polymarket + Kalshi discovery and order books, normalized to an up-token view. |
| `live/predictor.py` | Online ensemble: heads, reliability weighting, calibration, Markov-regime temperature, conditional-skill. |
| `live/structural.py` | Digital pricer: `α = N(d₂)`, quadratic-variation vol, κ terminal-variance shaper. |
| `live/forward_engine.py` | The cycle: discover → predict → EV-gate → ghost-fill → settle → learn; sizing, TP overlay, control. |
| `live/control.py` | Shared risk control plane (kill switches, breakers, exposure caps). |
| `live/md_service.py`, `live/md_client.py` | Shared WS-first market-data service + stdlib engine adapters. |
| `live/db.py` | SQLite stores (decisions / settlements / paths / calibration / health). |
| `live/dashboard_server.py`, `live/dashboard.html` | Live dashboard, one tab per lane. |
| `research/strategy/` | Foundational studies: digital/terminal Boltzmann pricer, quadratic-variation & Markov vol, HAR-RV, HMM regime, variance decomposition, lead-lag. |
| `research/edge_proof/` | Net-of-fee EV, calibration, and cross-venue edge studies. |
| `data/` | Live run artifacts (DBs, model state, status) — gitignored. |

---

## Running

```bash
make check                                   # tests + ruff + mypy — the pre-merge gate
bash live/boot_engines.sh                    # launch lanes + market-data service + dashboard (tmux 'crypto')
python -m live.dashboard_server --port 8011  # http://127.0.0.1:8011
```

Lane configuration is pinned in **`live/engines.conf`** — the single source of truth; engines are
never launched with hand-typed flags, and a config-drift guard refuses to silently change a lane.

---

## What the live experiment has established

The system is run as an experiment, and the results are reported as they are — including the
negative ones, which is the point of keeping a fixed control lane:

- **Market direction is efficiently priced.** Out-of-sample, the model's disagreement with the
  mid does not predict which side wins (disagreement-AUC ≈ 0.50). Apparent directional signals do
  not survive cluster-robust, drift-controlled testing.
- **The replicated edge is calibration, not direction** — the model prices *confident* markets
  slightly better in Brier score (sharpness), which is a different and narrower claim than picking
  sides.
- **The binding constraint is execution, not signal.** Maker-at-mid is adverse-selected; the
  event-driven market-data service exists to measure and remove that, and liquidity provision plus
  calibration is the surviving thesis.

See [`research/audit/SYSTEM_OVERVIEW.md`](research/audit/SYSTEM_OVERVIEW.md) for the full design.

---

## License

Released under the **[PolyForm Noncommercial License 1.0.0](LICENSE.md)**. You may use, fork,
modify, and redistribute this software for any **noncommercial** purpose (research, study,
personal and academic work) provided notices are preserved. **Commercial use is not permitted**
under this license — contact the copyright holder for commercial terms.
