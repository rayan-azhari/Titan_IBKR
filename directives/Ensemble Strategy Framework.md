# Directive: Ensemble Strategy Framework

> Last updated: 2026-04-03

## Goal

Run **multiple uncorrelated strategies** simultaneously under a shared portfolio risk layer
(`PortfolioRiskManager`) to reduce single-strategy risk and improve capital efficiency.

## Current Strategy Status

| Strategy | Status | Instruments | PortfolioRM |
|---|---|---|---|
| IC Equity Daily (mean-reversion) | Live | 7 US equities | Wired |
| MTF Confluence (FX trend) | Live | EUR/USD | Wired |
| ORB (intraday breakout) | Live | 7 US equities | Wired |
| ETF Trend (daily trend) | Live | SPY, QQQ, IWB, TQQQ, EFA, GLD, DBC | Wired |
| MR FX (intraday mean-reversion) | Live | EUR/USD M5 | Wired |
| ML Classifier (PSKY, QQQ, EUR_USD) | Validation | Daily | Tier 2 deployment |

All 5 live strategies are connected to the shared `PortfolioRiskManager` singleton
(April 2026). Portfolio-level drawdown kill switch (15%) and proportional scaling (10%)
are active across all strategies.

> [!IMPORTANT]
> See `directives/System Status and Roadmap.md` for the full system status, target
> portfolio composition (8 buckets), and implementation roadmap (Tier 1-3).

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
