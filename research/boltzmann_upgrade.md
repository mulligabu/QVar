# Boltzmann structural digital-probability upgrade

Advisory note on integrating "The Boltzmann Equation in Finance" (Bogliardi,
Charif Khalifi, Kitapbayev, Noguer Alonso, Occhionero, Zubelli) into the live
near-expiry binary engine. Source paper: `/tmp/paper.txt`. Engine read but NOT
modified: `live/predictor.py`, `live/forward_engine.py`. Prototype (stdlib-only):
`research/strategy/boltzmann_digital.py`.

Bottom line up front: the paper's headline result (a closed-form *call PDF* at
maturity) does **not** transfer to our markets — binaries don't have a payoff
density. But the one scalar inside that PDF, the coefficient `alpha`, **is**
exactly the digital probability we need, and it is the standard Gaussian /
N(d2) probability. Its real value here is not as a new alpha source but as a
**calibration anchor** that directly attacks the live failure mode
(overconfident, 100%-one-sided favorite buying). Replaying the engine's own DB,
anchoring to the structural `alpha` removes the seven biggest losers and flips
the joinable filled book from **−46.0 to +82.0** (with honest caveats in §5).

---

## 0. The live problem, quantified from the DB

From `data/forward_live/forward_engine.db` (filled+traded settlements joined to
their decisions):

| metric | value | implication |
|---|---|---|
| filled trades | 33 | all `side="down"` (100% directional bias) |
| winrate | 0.606 | **< avg price 0.647** → negative edge per contract |
| realized PnL | −46.0 | losing |
| 0.70-price bucket | 17 trades, −47.1 | the favorites are the whole loss |
| 0.60-price bucket | 16 trades, +1.1 | roughly break-even |
| avg `p_cal` on trades | 0.261 (→ p_down 0.74) | model thinks p_down ≫ price 0.65 |
| avg `fv_resid` on trades | −0.094 | model thinks UP is *systematically overpriced* |

For a binary, EV>0 ⇔ winrate > price. The engine's gate fires on
`p_side − price > min_ev`, but realized `winrate (0.606) < price (0.647)`. So the
gate is being fed an **overconfident, biased** `p_side`. The single most
informative cut: split the 33 trades by how far the calibrated ensemble diverges
from the structural digital prob `alpha` (computed below):

| group | n | PnL | avg PnL |
|---|---|---|---|
| `|p_cal − alpha| > 0.30` | 7 | **−81.0** | −11.6 |
| `|p_cal − alpha| ≤ 0.30` | 26 | +35.0 | +1.3 |

The losses are concentrated *exactly* where the ensemble most disagrees with the
structural Gaussian. That is the lever this upgrade pulls.

---

## 1. What transfers, and what does not (be skeptical)

### 1.1 The call-PDF result does NOT transfer
The paper's novelty (Eq. 11/18) is the full **probability density of a European
call's payoff** at maturity, `Ψ(c,N) = α·δ(c) + (1−α)·γ(c,N)`:
- `α·δ(c)`: point mass at payoff 0 (option expires worthless).
- `(1−α)·γ(c,N)`: continuous density over positive payoffs `c>0`, where
  `γ(c,N) = 1/(1−α) · 1/(k+c) · 1/√(2π(N−n)σ²) · exp(−[log(k+c)−(N−n)µ]²/(2(N−n)σ²))`.

Our markets are **digital**: payoff is `1` if `S_T > k` (or `S_T ≥ k`), else `0`.
There is no continuous payoff axis. `γ(c,N)` — the part the paper actually plots,
the part it calls its contribution — is **irrelevant** to us. The `1/(k+c)`
Jacobian, the fat-tailed exercise density, the Black-Scholes-via-expectation
derivation (Eq. 13–15): none of it touches a binary. We should not pretend the
"50-years-unexplored PDF" is doing work for us. **It isn't.**

### 1.2 What DOES transfer: the scalar `α`
The *only* quantity that survives for a digital is the Dirac coefficient
`α = P(S_T ≤ k)` = P(DOWN/OTM). The paper (Eq. 16):

```
α = ½ · Erfc[ ((N−n)µ − log k) / (σ·√(2(N−n))) ]
```

This is the textbook Gaussian digital probability. **Derivation / sanity check
(reproduced in the prototype's `_selftest`)**: under the paper's own assumption
`x_T = log S_T ~ Normal(mean = log S + (N−n)µ, var = (N−n)σ²)` (spot folded into
the mean), define

```
d := (mean − log k)/(σ√(N−n)) = (log(S/k) + (N−n)µ)/(σ√(N−n))
```

Then `α = ½·Erfc[d/√2] = 1 − Φ(d)` and therefore

```
P(up) = 1 − α = Φ(d) = N(d2).
```

So the paper's `α` is **identically** `1 − N(d2)`, and `P(up)` is the standard
N(d2) digital probability. (At-the-money with zero drift: `d=0 → P(up)=½`, which
the prototype asserts.) This is *not new mathematics* — Breeden–Litzenberger
(ref [6]) and risk-neutral valuation give the same thing — but it gives us a
**clean, parameter-light, causal** probability to anchor against. That is its
entire value here.

> OCR note: the extracted numerator reads `(N−n)µ − log k`. The spot enters
> through the mean `log S`, so the operative numerator is `log(S/k) + (N−n)µ`.
> The original code had this sign inverted; the prototype's selftest catches it
> (spot above strike must give `P(up)>½`). Verified against `Φ` directly.

### 1.3 What also transfers, more speculatively: the variance result
The stationary-growth result (Eq. 20), `σ²_n = C̃₁·(1)ⁿ + C̃₂·(−1)ⁿ`, says terminal
variance can **oscillate/concentrate** rather than diffuse monotonically as
`σ²·t`. This is qualitatively useful (§4) — markets near expiry often *pin* to a
level (concentration) or break out (expansion) rather than diffuse. But Eq. 20
is a toy two-initial-condition difference equation, not a fitted model; we use it
only as **motivation** for a data-driven diffusion-vs-concentration detector, not
as a literal `(−1)ⁿ` law. Don't oversell it.

### 1.4 Drift convention caveat
The paper uses the *physical* drift `µ` (empirical mean log-return). N(d2) in
risk-neutral pricing uses `(r − ½σ²)`. We are not pricing; we are predicting the
*physical* probability of crossing, so the paper's physical `µ` is the correct
choice. In practice, near expiry, `|µ·τ| ≪ σ·√τ` for crypto (drift is
second-order over minutes), so `P(up) ≈ Φ(log(S/k)/(σ√τ))` and the drift term
barely moves `α`. We keep it for the longer (1h/1D) lanes where it matters more.

---

## 2. The `boltzmann_structural` head (closed form + estimators)

### 2.1 Closed form (seconds-native)
Work in seconds so all timeframes share one code path. Let `S`=spot, `k`=strike,
`τ`=seconds to expiry, `µ_s`=per-second drift of log price, `σ_s`=per-second vol
of log returns. Then:

```
num   = log(S/k) + τ·µ_s
denom = σ_s·√(2τ)
α     = ½·Erfc[ num/denom ]        # = P(DOWN)
P(up) = 1 − α                       # = N(d2)
```

Time scaling lives entirely in `τ`: a 5m and a 1D market use the *same* `µ_s,σ_s`
and differ only by `τ ∈ {300, 86400}` seconds. Mean scales `∝τ`, vol `∝√τ`. This
is `digital_prob_up()` in the prototype.

### 2.2 Causal estimation of `µ_s`, `σ_s`
`CausalDriftVol` (prototype) consumes the same `(ts, price)` ticks the engine
already buffers in `self.hist[asset]`. For each new tick it forms a log-return
`r = log(p/p_prev)` over elapsed `dt`, then maintains EWMA (half-life ≈ 240
returns) of the **per-second** contributions `r/dt` (drift) and `r²/dt`
(variance), so:

```
µ_s = EWMA(r/dt)
σ_s = √( max( EWMA(r²/dt) − µ_s², floor ) )
```

This is causal (only past ticks), adaptive (EWMA tracks vol regime), and
*per-second* so it scales to any `τ`. Selftest recovers `σ_s ≈ 5.1e-4` from a
synthetic `5.0e-4` process. **Note**: this is genuinely independent of the
ensemble's `realized_vol` feature, which is computed per-window and not
horizon-consistent — `CausalDriftVol` fixes the time-scaling that
`build_features` does ad hoc via `sigma_tau`.

### 2.3 Relationship to the existing `z_dist` feature
The engine already computes `z_dist = log(spot/strike)/sigma_tau` where
`sigma_tau` is the (rough) terminal vol. Therefore **`P(up) = Φ(z_dist)`** when
`µ≈0` — the structural head is, to first order, just `Φ(z_dist)`. This is why the
backtest in the prototype can reconstruct `alpha` from logged `z_dist` without
re-running the feed. The upgrade replaces the *implicit, miscalibrated* use of
`z_dist` (fed through a learned logistic with weight 1.8 and a temperature) with
the *explicit, parameter-free* `Φ(z_dist_seconds)` as a probability anchor.

---

## 3. Using `α` as a calibration anchor (the actual fix)

The structural `α` is **not** added as just another reliability-weighted head
(that would let the overconfident ensemble out-vote it again). It is used two
ways, both in `structural_anchor()`:

### 3.1 Shrinkage toward `α` (log-odds), divergence-scaled
```
div = |p_ens − α_up|
λ   = min(λ_max, λ0 + div_scale·div·λ0)          # λ0=0.35, div_scale=6, λ_max=0.85
p*  = σ( (1−λ)·logit(p_ens) + λ·logit(α_up) )
```
The more the ensemble disagrees with the structural Gaussian, the harder we pull
it back toward `α`. A 0.74 ensemble p_down against a 0.56 structural p_down gets
shrunk most of the way back — killing the EV signal that drove the favorite buy.

### 3.2 Divergence-widened EV gate
```
required_margin = min(gate_max, gate_base + gate_div_k·div)   # 0.01, +0.40·div, cap 0.12
TRADE iff  p_side* − price − costs  >  required_margin
```
A trade where ensemble and structural agree (`div≈0`) needs only the normal
1-cent edge. A trade where they diverge by 0.3 needs 13 cents — effectively
forbidden. This is a *second* line of defense beyond shrinkage: even if shrinkage
leaves a marginal edge, high divergence raises the bar.

### 3.3 Why this would have removed the favorite-buying losses
The losing trades were precisely the high-divergence ones (§0 table: the 7 trades
with `|p_cal − α|>0.30` were −81.0). Mechanically: the engine bought `down` at
~0.65 because `p_cal_down ≈ 0.74`, but the structural `α` (P(down)) for those was
often **< 0.50** (price was actually drifting up into the strike). Shrinkage
pulls `p_down` back below the price → `EV<0` → SKIP. The divergence gate
independently blocks them. Row-by-row in the prototype output, every trade with
`alpha ≥ 0.7` (structural P(up) high while engine bet down) is killed, and those
carried PnL of −0.53, −0.47, −0.34, −0.35, −0.21, −0.16.

---

## 4. Variance concentration → terminal-σ model (Eq. 20)

`TerminalVarianceRegime` (prototype) operationalizes Eq. 20's diffusion-vs-
concentration insight without taking the `(−1)ⁿ` form literally. It maintains
recent log-returns and computes a sub-/super-diffusion ratio:

```
κ = Var(returns over last 2w) / Var(returns over last w)
```
- `κ ≈ 1`  → diffusion (variance accumulates ∝ time, the BS assumption).
- `κ < 0.8` → **concentration**: variance is *not* accumulating; price is pinning
  / mean-reverting near a level (the paper's "contraction"). Terminal dispersion
  is smaller than `σ√τ` implies → `α` should be **sharper** (further from 0.5).
- `κ > 1.25` → **expansion**: trending / breakout; terminal dispersion larger →
  `α` should be **softer** (closer to 0.5), reducing overconfidence.

The head feeds `σ_eff = σ_s · √(clamp(κ, 0.6, 1.6))` into `digital_prob_up`. This
ties directly into the existing stack:
- It **replaces/augments the `move_prob` head**: low `κ` (concentration) ≈ low
  tradable move ≈ the regime should *widen* the gate, not narrow it. The current
  `_move_prob` uses `realized_vol·√τ` vs a learned scale; `κ` adds the missing
  *shape* (is that vol diffusing or pinning?).
- It maps onto the existing `classify_regime` axes: `efficiency≥0.45`
  (trend) ↔ expansion (`κ>1`); `efficiency≤0.20` (mean-revert) ↔ concentration
  (`κ<1`). So `κ` is a continuous, horizon-correct version of the `efficiency`
  feature, and can be cross-checked against it.

Honest caveat: `κ` from short windows is noisy; treat it as a mild σ multiplier
(±25%), not a regime switch. Its main job is to stop the structural `α` from
being over-sharp in pinning regimes (the false "high conviction" that, combined
with adverse-selected fills, hurts most).

---

## 5. Integration plan, expected impact, risks, validation

### 5.1 Concrete changes to `predictor.py` / `forward_engine.py`
Ordered by impact-to-effort. **None require touching the ensemble heads' math.**

1. **Add `BoltzmannStructuralHead`** to `predictor.py` (a thin class wrapping
   `CausalDriftVol` + `TerminalVarianceRegime` + `digital_prob_up`). It is fed the
   raw `(ts, price)` history and `(spot, strike, tau_s)` — *not* standardized
   features — so it stays parameter-free and immune to warmup overfit. Output:
   `alpha_up`, `kappa`, `regime`.

2. **In `OnlinePredictor.predict`**, after computing `p_up`/`p_cal`, call
   `structural_anchor(p_cal, alpha_up, cfg)` and return `p_anchored` plus
   `struct_div = |p_cal − alpha_up|` and the gate margin in the `Prediction`
   dataclass (new fields; additive, won't break persistence). `p_cal` used
   downstream becomes `p_anchored`.

3. **In `evaluate_trade`** (forward_engine), replace the fixed `self.min_ev` with
   the divergence-widened `required_margin` from the prediction, and add
   `struct_div ≤ 0.25` as a hard gate alongside the existing `sane`/`priced_ok`.
   This is a ~5-line change to the `decision = "TRADE" if (...)` predicate.

4. **Log** `alpha_up`, `struct_div`, `kappa`, `sigma_s` into the `decisions` table
   (additive columns, mirror the `ticker`/`side` migration pattern in `db.py`) so
   the next review can measure the head's standalone calibration.

5. **(Optional, later)** Add a directional de-bias guard: the 100%-`down` bias is
   itself a red flag. Track rolling `side` balance; if >80% one-sided over the
   last N trades, raise the gate on the majority side. The structural anchor
   *already* fixes this indirectly (it's symmetric in `S/k`), but an explicit
   guard is cheap insurance.

### 5.2 Expected impact — honest
- **Counterfactual on the current DB** (prototype replay, n=33 joinable filled
  trades): keeps 10/33, winrate 0.606→**0.900**, PnL −46.0→**+82.0**.
- **Discount this heavily.** Caveats: (a) n=10 kept is tiny; 0.90 is not a
  reliable winrate estimate. (b) The replay reconstructs `α` from the *logged*
  `z_dist`, so it is partly in-sample. (c) It does not re-simulate fills — it
  assumes the kept trades fill as they did. (d) Survivorship: removing 23 trades
  also removes their (mostly winning, small) PnL.
- **What I'm confident in**: the *direction* and *mechanism*. The clean, robust
  finding (§0) is that loss is monotone in structural divergence (high-div 7
  trades = −81; low-div 26 = +35). Any anchor that down-weights high-divergence
  trades improves the book. A realistic expectation is **fewer trades, materially
  higher winrate, and removal of the fat-tail favorite losses** — not necessarily
  the full +82 swing.

### 5.3 Risks
- **Over-anchoring**: if `σ_s` is mis-estimated low, `α` becomes over-sharp and
  could *reinforce* a bad bet. Mitigated by the `κ` multiplier and by anchoring in
  log-odds with capped `λ`.
- **The market is the best estimator**: `predictor.py`'s own comment already notes
  these markets are efficient and the existing `fair_value` head anchors to mid.
  The structural `α` and the market mid should usually *agree*; when they don't,
  trust the mid more than `α` (the mid prices in order flow `α` can't see). Suggest
  a final blend `p_final = β·mid + (1−β)·p_anchored`, β≈0.5, which the existing
  bounded-tilt logic already approximates.
- **Throughput collapse**: widening the gate on divergence will cut trade count
  sharply (here, 33→10). At current volumes that means slow validation. Accept it;
  the alternative is bleeding.

### 5.4 Validation protocol before going live
1. **Static replay** (done, prototype): `python3 research/strategy/boltzmann_digital.py
   data/forward_live/forward_engine.db`. Confirms divergence flags the losers.
2. **Shadow re-scoring**: add the head + logging only (steps 1,2,4 above) with the
   gate change DISABLED. Run 48h. Then query the DB:
   - `α` calibration: bucket `alpha_up`, compare hit-rate to `alpha_up` (should
     lie on the diagonal far better than the ensemble's `p_cal` does today).
   - Confirm `|p_cal − alpha_up|` still separates winners from losers
     out-of-sample (the §0 split, now on fresh data).
3. **Gate A/B**: only after (2) shows `α` is well-calibrated, enable the
   divergence-widened gate (step 3). Compare next-window realized winrate vs price
   on traded contracts; require **winrate > price** (positive per-contract edge)
   over ≥50 trades before sizing up.
4. **Kill criteria**: if shadow `α` calibration is worse than the ensemble, or the
   gate doesn't lift winrate above price in 50 trades, revert — the structural
   anchor is a calibration tool, not a new edge, and must earn its place.

---

## Files

- `research/boltzmann_upgrade.md` — this note.
- `research/strategy/boltzmann_digital.py` — stdlib-only prototype:
  `digital_prob_up` (the verified `α`/N(d2) form), `CausalDriftVol`,
  `TerminalVarianceRegime`, `structural_anchor`, and `backtest_db` (DB replay).
  Run with no args for the math selftest; pass the DB path for the counterfactual.
