# Directive: Ensemble Strategy Framework

> Last updated: 2026-04-21 (post portfolio-risk rewrite)

## Goal

Run **multiple uncorrelated strategies** simultaneously under a shared portfolio risk layer
to reduce single-strategy risk and improve capital efficiency. As of April 21, 2026 that
layer is the rewritten `PortfolioRiskManager` + `PortfolioAllocator` + per-strategy
`StrategyEquityTracker`.

## Champion Portfolio (paper, April 2026)

Runner: `uv run python scripts/run_portfolio.py --strategies champion_portfolio`

| Strategy | Class | Instruments | WFO Sharpe |
|---|---|---|---|
| MR AUD/JPY | `MRAUDJPYStrategy` (vwap_anchor=46) | AUD/JPY.IDEALPRO H1 | +2.10 OOS |
| MR AUD/USD | `MRAUDJPYStrategy` (vwap_anchor=36) | AUD/USD.IDEALPRO H1 | +2.00 research |
| Bond-Equity IHYU→CSPX | `BondGoldStrategy` reused | CSPX.LSEETF D | +1.64 OOS (25 folds 2013-2026) |

All three use the full `StrategyEquityTracker` integration -- they report real per-strategy
equity (seed + realised P&L), not whole-account NLV. For AUD/JPY the FX unit conversion
path (`convert_notional_to_units`) is live; for AUD/USD and CSPX the trivial USD path is used.

## Other Live Strategies

| Strategy | Status | Instruments | Integration pattern |
|---|---|---|---|
| IC Equity Daily | Live | 7 US equities | Deterministic USD fallback |
| MTF Confluence | Live (edge invalidated) | EUR/USD | Deterministic USD fallback |
| ORB | Live | 7 US equities | Deterministic USD fallback + explicit FX conversion |
| ETF Trend | Live | SPY, QQQ, IWB, TQQQ, EFA, GLD, DBC | Deterministic USD fallback |
| MR FX | Live | EUR/USD M5 | Deterministic USD fallback |
| GLD Confluence | Live-ready | GLD.ARCA H1 | Deterministic USD fallback |
| Gold Macro | Live-ready | GLD.ARCA D | Deterministic USD fallback |
| Bond->Gold | Live-ready | GLD (IEF signal) | Full tracker (via BondGoldStrategy) |
| FX Carry AUD/JPY | Live-ready | AUD/JPY.IDEALPRO D | Deterministic USD fallback |
| Pairs GLD/EFA | Live-ready | GLD + EFA D | Deterministic USD fallback |
| Gap Fade | Live-ready | EUR/USD M5 | Deterministic USD fallback |
| ML Classifier | Validation | Daily | Deterministic USD fallback |

**Two integration patterns coexist during the migration:**

1. **Full tracker** (`MRAUDJPYStrategy`, `BondGoldStrategy`): owns a
   `StrategyEquityTracker`, reports per-strategy equity, accumulates realised P&L in
   `on_position_closed`. This is the target state for every strategy.
2. **Deterministic-USD fallback** (everyone else): uses
   `get_base_balance(account, "USD")` + explicit `bar.ts_event` timestamp, but still
   feeds whole-account NLV to the PRM. Safe in isolation (single strategy per process)
   but defeats the inverse-vol allocator when multiple strategies run simultaneously.

Migration from pattern (2) to pattern (1) is a ~one-day task per strategy tracked in
the roadmap. The champion portfolio is entirely on pattern (1).

Portfolio-level drawdown kill switch (15%) and proportional scaling (10%), plus the
vol-target and regime-scale overlays, are active across all strategies regardless of
integration pattern.

> [!IMPORTANT]
> See `directives/System Status and Roadmap.md` section 4 for full portfolio-risk details
> and `C:/Users/rayan/.claude/skills/titan-orchestrator/references/portfolio-risk-architecture.md`
> for the definitive architecture reference.

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
