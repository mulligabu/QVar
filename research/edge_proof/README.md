# Edge-Proof Harness

Answers one question from the existing `crypto_strat` forward data:
**is there a slice of decisions with positive realized net-of-fee EV?**

Stdlib-only (no pandas/numpy). Reads the 220GB forward runs read-only.

## Run

```bash
python -m research.edge_proof.run --runs 4 --sample 6        # full report
python -m research.edge_proof.counterfactual --runs 4         # selectivity PnL
```

## Modules
- `outcomes.py`  — label each 15m window up/down from fresh 1m candles
                   (`binance_multiasset_model_dirs/<ASSET>/`), cross-checked
                   against the engine's own `settlement_winning_side`.
- `loaders.py`   — stream compact rows from settled trades & decisions.
- `analyze.py`   — calibration / Brier / net-EV-by-bucket primitives.
- `run.py`       — settled-trade money attribution + entry-time calibration.
- `counterfactual.py` — replay settled trades under selectivity rules.

## Headline finding (last 4 runs, 380 settled trades)
- Trade everything: **-$549**, 47.4% win, fees ≈ -$260 on -$289 gross.
- Filter stated edge >= 2%: **+$207**, 53.3% win.
- BTC+ETH & edge >= 2%: **+$217**, 63.4% win (41 trades).
The engine has a thin real edge it destroys by overtrading marginal/negative
buckets and alts. Needs out-of-sample (walk-forward) confirmation.

## Structural-edge tests (bridge / feed_basis / cross_venue)
```bash
python -m research.edge_proof.bridge --runs 4        # Test 1: terminal/lead-lag
python -m research.edge_proof.feed_basis --runs 4    # Test 2: feed basis/determinism
python -m research.edge_proof.cross_venue --runs 4   # Test 3: cross-venue + arb
```
Shared foundation in `markets.py`. Outcome labels are CAUSAL (open of candle@T,
matching the engine's settlement convention; 96% agreement). `price_at` must NOT
use candle close (1-min look-ahead — it faked a +0.095/contract bridge edge).

### Findings (last 4 runs)
- Test 1 (bridge): NO edge after removing look-ahead (-0.058/contract). The book
  is faster than 1m candles; when our stale estimate disagrees, the book is right.
- Test 2 (feed basis): book lags underlying sign by ~+5-10pts, but Test 1 shows
  it's not capturable as a taker; would need real-time feed + maker (unproven).
- Test 3 (cross-venue): the survivor. Directional cheap-UP +0.047/contract;
  market-neutral arb lockable in 67% of dual-venue windows, +0.076 avg locked
  profit after fees. Open risks: execution simultaneity, top-of-book size,
  Chainlink-vs-CF settlement disagreement (~4%).
