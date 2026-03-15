# Directive: Ensemble Strategy Framework

> Last updated: 2026-03-14

## Goal

Run **multiple uncorrelated strategies** simultaneously, combining their signals into a weighted
ensemble to reduce single-strategy risk and improve capital efficiency.

## Current Strategy Status

| Strategy | Status | Ensemble-Ready? |
|---|---|---|
| ORB (7 US equities) | Live (deployed Mar 2026) | Yes — after ≥ 1 month live track record |
| MTF (EUR/USD H1/H4/D/W) | Deployment-ready (OOS Sharpe 2.936) | After ≥ 1 week clean paper session |
| ML (XGBoost, EUR/USD) | Experimental (OOS Sharpe 1.142) | No — below 1.5 threshold |

> [!IMPORTANT]
> MTF must complete at least one full week of clean paper trading before ensemble registration.
> ORB and MTF trade different instruments (equities vs forex) — naturally low correlation,
> making them strong ensemble candidates.

---

## Inputs

- Trained `.joblib` models in `models/` (for ML strategies only)
- Strategy configs in `config/` (one per strategy)
- Ensemble registry in `config/ensemble.toml`

## Architecture

```
┌────────────────────────────────────────────────────┐
│               Ensemble Engine                       │
│  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │ ORB Strategy  │  │ MTF Confluence (EUR/USD)     │ │
│  │ 7 US equities │  │ H1/H4/D/W, SMA, OOS 2.936   │ │
│  └──────┬────────┘  └─────────────┬────────────────┘ │
│         ↓                         ↓                  │
│  ┌─────────────────────────────────────────────┐     │
│  │     Weighted Signal Aggregation (≥0.3)      │     │
│  └─────────────────────────────────────────────┘     │
│         ↓                                            │
│  ┌─────────────────────────────────────────────┐     │
│  │   Correlation Filter + Weight Rebalancer    │     │
│  └─────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────┘
```

---

## Execution Steps

### 1. Strategy Discovery

For each strategy type, complete the Alpha Research Loop and pass all robustness gates.
Save each trained model (if ML) with a descriptive name in `models/`.

### 2. Register Strategies

Add each strategy to `config/ensemble.toml` with initial weights.
Set `active = true` only after the strategy passes live paper trading validation.

### 3. Correlation Check

Run the ensemble correlation check:
```bash
uv run python research/ml/run_ensemble.py
```

Strategies with pairwise correlation > `correlation_threshold` (default 0.70) have weights
automatically reduced. ORB (equities) and MTF (EUR/USD forex) are expected to be low-correlation
by design — different instruments, different market structures.

### 4. Signal Generation

Each active strategy independently generates signals on the latest features/bars.
Signals are combined using performance-weighted voting.
Ensemble only trades when weighted consensus exceeds ±0.3 threshold.

### 5. Rebalancing

Weights are recalculated at `rebalance_frequency` (default: weekly).
Strategies that underperform are downweighted; strong performers are upweighted.

---

## Safety Constraints

> [!IMPORTANT]
> - Minimum 2 active strategies required to trade
> - No single strategy may hold > 60% of capital allocation
> - All strategies must independently pass OOS validation (Sharpe ≥ 1.5)
> - ML strategies additionally require OOS Sharpe ≥ 1.5 and model freshness < 1 month

---

## Outputs

- Ensemble signal (BUY / SELL / HOLD)
- Per-strategy signal breakdown
- Correlation matrix report
- Rebalanced weight allocations
