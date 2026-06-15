# Live Forward Prediction Engine

Forward-running (NOT backtest) near-expiry binary prediction engine.

- `feed.py`    — real-time price: Chainlink BTC/USD on-chain (settlement-grade) +
                 Coinbase fast tick. Chainlink-anchored composite.
- `venues.py`  — Polymarket + Kalshi live market discovery + order book (browser UA).
- `predictor.py` — online RenTech predictor: feature-weighted Boltzmann/logistic
                 (=online max-entropy), regime + vol-conditioned Bayesian-Dirichlet
                 transitions, entropy/temperature scheduling, online calibration,
                 EV-after-costs gate. Causal/online only.
- `forward_engine.py` — loop: sample feed -> record strikes -> discover markets ->
                 near-expiry predict -> ghost LIMIT at mid on predicted side ->
                 settle on Chainlink -> learn online. Generic over directional &
                 threshold markets, multi-asset/expiry/venue.

Run: `python -m live.forward_engine --assets BTC --minutes 45`
Logs: data/forward_live/decisions_*.jsonl, settlements_*.jsonl
