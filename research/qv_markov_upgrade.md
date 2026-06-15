# Noise-robust integrated-variance (σ) upgrade — Hansen & Horel (2009)

Advisory note on translating Peter Reinhard Hansen & Guillaume Horel,
*"Quadratic Variation by Markov Chains"* (SSRN 1367519, 2009) into the live
near-expiry digital engine. Source paper read in full: `/tmp/m_1367519.txt`.
Engine read but **NOT modified**: `live/predictor.py`, `live/feed.py`. Prior
work this builds on: `research/boltzmann_upgrade.md` (the structural-α anchor),
which depends entirely on the σ this note sharpens. Prototype (stdlib-only):
`research/strategy/qv_markov.py`.

**Bottom line up front.** The full Hansen-Horel Markov-chain estimator does
**not** transfer to our feed: it needs a fixed tick *grid* with a handful of
states and *thousands* of tick-by-tick increments per window to estimate the
transition matrix — we have a continuous float price and ~20-40 samples per
market. But the paper's own simplest case (§10.2, two states / order one)
collapses to a **first-order autocovariance correction of realized variance**,
which is the classic Zhou (1996) noise-robust estimator. That *is* implementable
on our irregular composite feed, it is the defensible member of the family for
our data, and on the engine's own `paths` table it de-inflates σ by ~9% on
average and improves digital-probability calibration (Brier 0.1613→0.1570,
log-loss 0.8190→0.7975). The effect is **real but modest**, and it sharpens —
rather than replaces — the structural-α anchor.

---

## 1. Faithful summary of the Hansen-Horel estimator

### 1.1 The model: a contaminated semimartingale on a grid
Observed price `X_t = Y_t + U_t`, where `Y_t` is the latent **efficient price**
(a martingale after folding the finite-variation part into the noise) and `U_t`
is **market-microstructure noise** (§2, p.5). The object of interest is the
quadratic variation `[Y]_T = ∫₀ᵀ σ²_u du` (integrated variance). Naive realized
variance `RV = Σᵢ (ΔX_{Tᵢ})²` is consistent for `[Y]` only when `U=0`; with
noise it is biased — at high frequency the noise dominates and `RV → ∞` as
sampling intensifies.

The discreteness premise (§3, p.10): quoted/traded prices live on a **tick
grid** (NYSE: whole cents), so price *increments* `ΔX_{Tᵢ} ∈ {x₁,…,x_S}` take a
small number `S` of values. Assumption 2: the increments are ergodic and
distributed as a **homogeneous Markov chain of order k** with `S < ∞` states.

### 1.2 Filtering removes the noise (§2)
The key theoretical result (Proposition 1, Assumption 1): if `Y` is a martingale
and the noise is ergodic with a finite first moment (a very weak condition — far
weaker than the usual iid assumption), then the **G-filtered price**
`lim_{h→∞} E(X_{t+h} | G_t) = Y_t + FV*_t` differs from the efficient price only
by a continuous finite-variation process. So the realized variance of the
*filtered* price recovers `[Y]`. Crucially (the comment on p.7): one must use
**returns of the filtered price**, not *filtered returns* — the sum of squared
filtered returns does **not** estimate QV. This is the conceptual heart of the
paper.

Feasibly we cannot condition on `G_t` (it contains the latent `Y,U`); we use the
observable filtration `F_t = σ(X_s, s≤t)`. The Markov chain makes the conditional
expectations `E(ΔX_{Tᵢ+h} | F_{Tᵢ})` trivial to compute.

### 1.3 The estimator construction (§3, §4)
With the order-k chain having transition matrix `P` (size `Sᵏ×Sᵏ`), stationary
distribution `π` (`π'P = π'`), `Π = 1π'`, and the **fundamental matrix**

```
Z = (I − P + Π)⁻¹                                              (Kemeny–Snell)
```

the **filtered return** for a transition from state `x_r` to `x_s` is
(Lemma 2, h→∞):

```
y(x_r, x_s) = e_r'(I − Z)f + e_s'Z f ,
```

where `f` holds the last-increment value of each state. The filtered realized
variance is `RV_F = Σ_{r,s} n_{r,s} · y(x_r,x_s)²` (count-weighted). Substituting
the empirical `P̂` (transition counts) gives the **feasible estimator**, which is
shown (Theorem 4) to be asymptotically equivalent to the compact quadratic form

```
MC#  =  ⟨π, (2Z − I) π⟩     (level prices on a grid)
```

and for **log-prices** (§5, §10.1 Eq.6/approx), the production object:

```
        n² ⟨f, (2Ẑ − I) f⟩_π̂
MC  =  ─────────────────────── .                                          (★)
            Σᵢ X²_{Tᵢ}
```

**Consistency & limit law** (§4):
- Theorem 1: `RV_F → ⟨π,(2Z−I)π⟩` a.s.
- Theorem 2/3: `√n (MC# − ⟨π,(2Z−I)π⟩) →ᵈ N(0, Ω_MC)`, with `Ω_MC` available in
  **closed form** from `P̂` via the delta method (Corollary 1) — *no* need to
  estimate integrated quarticity or the noise long-run variance separately (a
  major practical advantage over realized kernels). A `log(MC#)` transform
  (Corollary 2) improves finite-sample coverage.
- The asymptotic design (Assumption 3) shrinks the state values `f = ξ/√n` so
  `RV` neither vanishes nor diverges — the standard in-fill scheme.

**Robustness (§6, §7).** Under a continuous-time Markov chain the discretely
sampled chain is homogeneous in the limit *if zero-increments are discarded*
(Theorem 5) — an argument for dropping flat-price ticks. And the estimator is
**robust to inhomogeneity** (time-varying volatility) by *increasing the order
k*: a misspecified order-2 chain matches an oracle that knows the regime breaks
(Tables 1–2). Recommended `k = 3–4` empirically.

### 1.4 The transferable special case (§10.2) — this is the lever
For `S = 2`, `k = 1`, drift-less (`p = q`), the entire machinery collapses to

```
MC#  =  (1 + ρ̂)/(1 − ρ̂) · n·σ̂² ,                                         (✦)
```

where `ρ̂` is the second eigenvalue of `P̂` = the **lag-1 autocorrelation of the
increments**. Since `(1+ρ)/(1−ρ) ≈ 1 + 2ρ` for small `ρ`, and `ρ·RV ≈ n·γ₁` (the
lag-1 autocovariance), (✦) is, to first order, the **autocovariance-corrected
realized variance**

```
IV = Σᵢ rᵢ²  +  2 Σᵢ rᵢ rᵢ₋₁   =  RV + 2·n·γ₁ ,                            (Zhou)
```

i.e. the Zhou (1996) estimator, and the paper notes Large's (2006) "alternation"
estimator `(c/a)·nσ²` is the same object. **This is the practically defensible
transfer for our setting** (see §2).

### 1.5 Assumptions that do NOT hold for our feed (be skeptical)
| Paper assumption | Our reality | Consequence |
|---|---|---|
| Fixed tick **grid**, few states `S` | Composite price is a **continuous float** (Chainlink/1e8 + a real-valued Coinbase drift correction). | Forcing onto a grid is possible but the natural `S` is unclear; the discreteness *advantage* the paper exploits is largely absent. |
| **Thousands** of tick-by-tick increments | ~20-40 samples per 15m market, ~100 for a DOGE 1D. | Cannot estimate an `Sᵏ×Sᵏ` transition matrix (even 2-state needs ~50+ transitions to be stable). The full `MC#` is **infeasible** here. |
| Sampling in event/tick time, near-equidistant in volatility time | **Irregular ~8-16s calendar sampling**; the engine samples per cycle. | We must work in calendar time with explicit `dt`, normalizing variance per-second. |
| Single clean price series | **Anchor + tick blend**: re-anchors on a new Chainlink round (a step), then adds Coinbase drift. This injects a *structured* artifact, not just iid bounce. | The noise is partly endogenous/serially-dependent and partly a deterministic blend artifact — exactly the regime where naive RV misbehaves, but also not the iid-bounce the simplest correction targets. |
| Homogeneous within window | Vol regime shifts within 15m. | Paper's fix (raise `k`) needs data we don't have; we instead keep windows short. |

**Honest verdict:** the *theorem* (filter, then take RV of the filtered price)
does not transfer as stated. The *mechanism* — correct RV for the serial
dependence the noise induces — transfers via its simplest closed form (✦/Zhou).

---

## 2. The implementable estimator for our setting

Given §1.5, we **do not** ship the grid Markov chain in production. We ship the
**lag-1 autocovariance-corrected realized variance** (the paper's `S=2,k=1`
case = Zhou 1996), with a Bartlett realized-kernel generalization available for
deeper noise, and the faithful 2-state Markov-grid estimator kept only as a
**diagnostic** to prove the equivalence on our data. All in
`research/strategy/qv_markov.py`, stdlib-only (no numpy).

### 2.1 Core estimator
```python
IV_robust = RV + 2·n·γ₁ ,    γ₁ = lag-1 autocovariance of log-returns
```
- `realized_variance_naive(rets)`  — `Σ rᵢ²`, what `build_features` does today.
- `realized_variance_ac1(rets)`    — the noise-robust (✦/Zhou) form. Floored at
  `0.25·RV` so a short/jumpy window can never drive the variance to ≤0.
- `realized_variance_kernel(rets, q)` — Bartlett-weighted multi-lag kernel
  `n·γ₀ + 2Σ_{h≤q}(1−h/(q+1))·n·γ_h`, PSD by construction; for serially
  dependent noise beyond lag 1 (the paper's higher-`k` case).
- `markov_qv_grid(rets, 2, 1)` — faithful `MC# = n·⟨f,(2Z−I)f⟩_π` on a forced
  sign grid with the §8.3 ancillary-observation trick (first=last state ⇒ closed
  stationary dist, no absorbing state). **Diagnostic only.**

### 2.2 Online drop-in: `NoiseRobustVol`
Same `update(ts, price)` contract as `CausalDriftVol` in `boltzmann_digital.py`.
Keeps a bounded ring of `(dt, r)`, and on demand returns **per-second** moments
(horizon-consistent across 5m…1D, the property `CausalDriftVol` was built for):

```
mu_s    = Σr / Σdt
sigma_s = sqrt( IV_robust(rets) / Σdt )          # robust=True
        = sqrt( RV(rets)        / Σdt )          # robust=False  (for A/B)
```

It exposes `sigma_s(robust=True|False)` so the *same object* produces both
estimates for shadow A/B testing, and `prob_up(spot, strike, tau_s)` returning
`N(d2)` with the identical convention as `boltzmann_digital.digital_prob_up`.
**It is a strict drop-in for `CausalDriftVol`'s σ** — `mu_s`, `prob_up`, and the
per-second scaling all match.

### 2.3 Why this and not the full chain (justification)
1. **Data volume.** A 2-state `P̂` is barely estimable at 30 samples; `Sᵏ`
   states are hopeless. (✦) needs only one autocovariance — robust at `n≥12`.
2. **Continuity.** No real grid ⇒ no natural `S`. The sign-grid `markov_qv_grid`
   throws away magnitude and is noisier than (✦), confirmed by the selftest
   (`MC_grid/AC1 ≈ 1.7` on AR(1) — same direction, more variance).
3. **Equivalence is proven.** The selftest shows (✦) *is* the 2-state Markov
   correction to first order, so we lose no theory by using the closed form.
4. **No ancillary nuisance estimates.** Like the paper's delta-method variance,
   (✦) needs no quarticity / noise-variance pre-estimate — unlike realized
   kernels whose bandwidth selection the paper itself calls "rather complex."

### 2.4 Selftest evidence (`python3 research/strategy/qv_markov.py`)
```
iid:     AC1/RV       = 0.997   (~1, no spurious correction on clean data)
bounce:  AC1/RV       = 0.327   (<1, MA(1) bid-ask-bounce noise de-inflated)
         |AC1−true| < |RV−true|  ->  6.5e-06 < 2.0e-03   (closer to truth)
AR(1):   MC_grid/AC1  = 1.724   (2-state Markov ~ AC1, same direction)
```
The bounce test is the crux: when observed = efficient + (eₜ − eₜ₋₁) noise, naive
RV is wildly inflated and AC1 recovers the true efficient variance.

---

## 3. Empirical validation against the live DB

`backtest_db()` in the prototype runs on `data/forward_live/forward_engine.db`.

### 3.1 Test design (honest about the data)
The `paths` table has **617 rows / 52 markets** (sparse: ~20-40 samples per 15m
market, ~100 for DOGE 1D, ~10-16s spacing). For each settled market that also
has a path we know `strike`, the per-cycle `underlying[]`, the timestamps, and
the realized `outcome_up`. The test:
1. Pick a decision point a few samples **before** the last (so horizon `τ>0`,
   `τ = expiry − ts[j]`, with `expiry ≈ last path ts`).
2. From the underlying samples **up to** that point, estimate σ **both ways**
   (naive RV and AC1-robust), per-second, `μ=0` (isolates the σ effect; near
   expiry drift is second-order — `boltzmann_upgrade.md §1.4`).
3. `α = N(d2)` for each σ; score against `outcome_up` via **Brier, log-loss, and
   a reliability table**. Better σ ⇒ α closer to the realized 0/1.

This directly answers the deliverable's question: *does noise-robust σ produce
better-calibrated digital probabilities than the naive σ?*

### 3.2 Result (n = 15 evaluable markets)
| metric (lower=better) | naive | robust (AC1) | kernel q=2 |
|---|---|---|---|
| Brier | 0.1613 | **0.1570** | 0.1611 |
| Log-loss | 0.8190 | **0.7975** | 0.8604 |

`mean σ_robust / σ_naive = 0.910` → the robust correction **de-inflates σ by
~9%** on this feed, consistent with positive microstructure inflation in the
composite tick. AC1 improves both proper scores. The Bartlett kernel (q=2) does
**not** help here (log-loss *worse*) — honest finding: at 20-40 samples the
multi-lag kernel adds variance without bias reduction. **Use AC1 (q≤1), not the
multi-lag kernel, on these window lengths.**

### 3.3 Caveats (do not oversell)
- **Small n.** 15 markets. Treat the *sign* of the Brier/log-loss gap as the
  signal, not the magnitude. This is not a 50-trade validation.
- Many α land at ~0.00/1.00 (deep markets that resolved as priced); the
  calibration signal lives in the few mid-range markets, where AC1's lower σ
  correctly **sharpens** (pushes α away from 0.5 toward the realized side) when
  the naive σ was inflated.
- σ is still estimated from the **composite** `underlying[]` — the very blend
  whose noise we're correcting. AC1 removes the iid/MA-bounce component; it does
  **not** remove the deterministic Chainlink re-anchor step (see §5 risk).
- This is **in-sample on a short run**. The real test is the shadow A/B in §4.

---

## 4. Integration into the structural-α anchor — prioritized

The σ from this note is the **input** to the `digital_prob_up` that
`boltzmann_upgrade.md` uses for its calibration anchor `α`. The cleanest, lowest-
risk integration:

### 4.1 Where it plugs in
- In the planned `BoltzmannStructuralHead` (`boltzmann_upgrade.md §5.1`), replace
  the σ source `CausalDriftVol.sigma_s` with `NoiseRobustVol.sigma_s(robust=True)`.
  **Nothing else in the anchor changes** — same `α = N(d2)`, same shrinkage,
  same divergence gate. This is a one-line swap of the σ provider.
- Keep `NoiseRobustVol.sigma_s(robust=False)` wired in parallel and **log both
  α_naive and α_robust** into the `decisions` table (additive columns, mirroring
  the `ticker`/`side` migration in `db.py`) so the next review measures the
  calibration delta out-of-sample.

### 4.2 Combined effect on the overconfidence / favorite-buying problem
The live failure (`boltzmann_upgrade.md §0`) is overconfident `p_cal` driving
favorite buys whose realized winrate (0.606) < price (0.647). The anchor fights
this by shrinking `p_cal` toward `α`. **A *less noisy, less inflated* σ makes `α`
itself more trustworthy**, which matters in two opposite directions:
- Where naive σ was **inflated** (the common case, ~9% here), naive `α` was
  pulled *toward 0.5* (under-confident, soft). Robust σ **sharpens** `α` — but
  *symmetrically in `S/k`*, so it sharpens toward whichever side the underlying
  actually favors, not toward the ensemble's biased "down". So it strengthens
  the anchor's pull when the ensemble is wrong, without re-introducing the
  ensemble's directional bias.
- The anchor's **divergence gate** (`required_margin ∝ |p_ens − α|`) becomes
  better-calibrated: a sharper, truer `α` means high divergence more reliably
  flags genuinely bad trades rather than noise.

### 4.3 Honest magnitude
- The calibration gain is **modest**: Brier −0.004, log-loss −0.022 on n=15.
  This is a *sharpening* of an already-decent probability, not a new edge. It
  will not, by itself, flip the book; the **anchor** (shrinkage + gate) is what
  removes the losers. This note makes the anchor's input cleaner.
- Expected practical effect: marginally tighter, better-calibrated `α`; slightly
  more confident (correct) skips on inflated-σ markets; no change to the
  parameter-free, symmetric nature that fixes the directional bias.

### 4.4 Risks
- **The re-anchor step.** AC1 corrects *serial-dependence* noise; the Chainlink
  re-anchor injects an occasional deterministic *jump* in `underlying[]`. That
  is a **jump**, not bounce — AC1 will partly absorb it as variance. Mitigation:
  the paper's §8.1 jump treatment (threshold large increments, e.g. drop/clip
  log-returns beyond a robust multiple of the window median), trivially added to
  `NoiseRobustVol` if shadow logs show anchor-step contamination. **Not added by
  default** — flagged for the review.
- **Over-de-inflation.** If σ is corrected too low, `α` over-sharpens and could
  *reinforce* a bad bet. The `0.25·RV` floor and the anchor's own κ multiplier +
  log-odds capped λ (`boltzmann_upgrade.md §3.1, §5.3`) bound this. Keep them.
- **The market is still the best estimator.** Per `predictor.py`'s own comment
  and `boltzmann_upgrade.md §5.3`, the market mid prices in flow `α` can't see.
  σ-robustness improves `α`, not the final blend; keep the mid-anchored bounded
  tilt and `p_final = β·mid + (1−β)·p_anchored`.

### 4.5 Validation protocol before live (gate on it)
1. **Static replay (done):** `python3 research/strategy/qv_markov.py
   data/forward_live/forward_engine.db` — AC1 σ improves Brier & log-loss.
2. **Shadow A/B (required):** log `α_naive`, `α_robust`, `σ_naive`, `σ_robust`
   for ≥48h with the trade gate **unchanged**. Then re-run the §3 calibration on
   the fresh, larger sample. **Kill criterion:** if robust α's Brier/log-loss is
   not ≤ naive's over ≥50 markets, revert — it's a sharpening tool and must earn
   it. Watch specifically for anchor-step jump contamination (σ_robust spuriously
   *higher* than σ_naive on markets that saw a Chainlink re-anchor).
3. **Only then** swap `CausalDriftVol` → `NoiseRobustVol` in the structural head.

---

## Files
- `research/qv_markov_upgrade.md` — this note.
- `research/strategy/qv_markov.py` — stdlib-only prototype:
  `realized_variance_naive / _ac1 / _kernel`, `markov_qv_grid` (faithful 2-state
  MC# diagnostic), `NoiseRobustVol` (drop-in for `CausalDriftVol`'s σ),
  `digital_prob_up`/`normal_cdf` (matching `boltzmann_digital`), and
  `backtest_db` (naive-vs-robust α calibration on the `paths` table). Run with no
  args for the math selftest; pass the DB path for the calibration backtest.
```
python3 research/strategy/qv_markov.py                                  # selftest
python3 research/strategy/qv_markov.py data/forward_live/forward_engine.db   # + calibration
```
```
