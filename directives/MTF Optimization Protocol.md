# MTF Strategy Optimization Protocol (VectorBT)

> **Status: COMPLETE — Round 4 Validated (2026-03-15)**
> Full 6-stage pipeline complete for EUR/USD. GBP/USD also validated (Combined Sharpe 1.331).
> Locked configs: `config/mtf_eurusd.toml`, `config/mtf_gbpusd.toml`.
> Do NOT re-run stages unless starting a new research cycle with fresh data.

---

## Overview

The MTF Confluence strategy is optimized in six sequential stages using VectorBT backtesting
on 10 years of historical OHLCV data (H1/H4/D/W timeframes). Each stage locks a layer of
parameters before the next stage begins, preventing look-ahead contamination across the search space.

**Split:** 70% In-Sample (IS) / 30% Out-of-Sample (OOS) — by bar count
**Friction modelled:** fees (1.5 pip/side), slippage (1.0 pip/side), next-bar execution, swap cost
**Evaluation metric:** OOS Sharpe Ratio (combined long + short)
**Anti-look-ahead:** `.shift(1)` applied to confluence signal before all VBT backtests

Pipeline is fully pair-agnostic. Any pair with parquet files in `data/` can be swept.

---

## Running the Pipeline

### Option A — Full automated run (recommended)

```bash
uv run python scripts/run_mtf_pipeline.py --pair EUR_USD
uv run python scripts/run_mtf_pipeline.py --pair GBP_USD
```

This runs all 6 stages sequentially, passing outputs between stages via the state manager.
Gate summary is printed at the end.

### Option B — Stage by stage

```bash
uv run python research/mtf/run_optimisation.py --pair GBP_USD
uv run python research/mtf/run_stage2.py        --pair GBP_USD --load-state
uv run python research/mtf/run_pair_sweep.py    --pair GBP_USD --load-state
uv run python research/mtf/run_stage4_atr.py    --pair GBP_USD
uv run python research/mtf/run_portfolio.py     --pair GBP_USD
uv run python scripts/robustness_mtf.py         --pair GBP_USD
```

### Resume from a specific stage

```bash
uv run python scripts/run_mtf_pipeline.py --pair EUR_USD --from-stage 4
```

---

## Stage 1 — Moving Average Type and Confirmation Threshold

**Objective:** Determine the best MA type and entry threshold across all timeframes simultaneously.

**Script:** `research/mtf/run_optimisation.py --pair PAIR`

**Parameters swept:**
- MA Type: `SMA`, `EMA`, `WMA`
- Confirmation Threshold: 0.05 to 0.85 (step 0.05)

**Output:**
- `.tmp/reports/mtf_stage1_{pair}_scoreboard.csv`
- `.tmp/reports/mtf_stage1_{pair}_heatmap.html`
- State saved to `.tmp/mtf_state_{pair}.json`

**EUR/USD result (locked):** MA=WMA, Threshold=0.10
**GBP/USD result (locked):** MA=SMA, Threshold=0.10

> [!IMPORTANT]
> WMA won EUR/USD OOS. SMA won GBP/USD OOS. This is pair-specific — always validate per pair.

---

## Stage 2 — Timeframe Weights

**Objective:** Determine how much influence each timeframe has on the composite score.

**Script:** `research/mtf/run_stage2.py --pair PAIR --load-state`

**Auto-loads:** Stage 1 MA type and threshold from state.

**Parameters swept:** Grid of H1/H4/D/W weight combinations that sum to 1.0.

**Output:**
- `.tmp/reports/mtf_stage2_{pair}_scoreboard.csv`
- State updated with winning weights.

**EUR/USD result (locked):**

| Timeframe | Weight | Role |
|---|---|---|
| D (Daily) | 0.55 | Primary trend bias |
| W (Weekly) | 0.30 | Long-term macro regime |
| H1 | 0.10 | Entry timing |
| H4 | 0.05 | Minimal — swing noise at EUR/USD scale |

**GBP/USD result (locked):**

| Timeframe | Weight | Role |
|---|---|---|
| D (Daily) | 0.55 | Primary trend bias |
| H4 | 0.30 | Swing confirmation |
| H1 | 0.10 | Entry timing |
| W (Weekly) | 0.05 | Long-term regime context |

---

## Stage 3 — Indicator Tuning (Per Timeframe)

**Objective:** Tune `fast_ma`, `slow_ma`, and `rsi_period` for each timeframe individually
using a greedy approach (optimize one timeframe at a time in order of weight).

**Script:** `research/mtf/run_pair_sweep.py --pair PAIR --load-state`

**Auto-loads:** Stage 1 + Stage 2 results from state.

**Greedy optimization order:** D → H4 → H1 → W (highest weight first)

**Output:**
- `.tmp/reports/mtf_stage3_{pair}_scoreboard.csv`
- Config written to `config/mtf_{pair_lower}.toml`

**EUR/USD result (locked):**

| TF | fast_ma | slow_ma | rsi_period |
|---|---|---|---|
| D  | 10 | 20 | 7 |
| H4 | 10 | 40 | 14 |
| H1 | 10 | 50 | 14 |
| W  | 8  | 21 | 14 |

**GBP/USD result (locked):**

| TF | fast_ma | slow_ma | rsi_period |
|---|---|---|---|
| D  | 5  | 20 | 10 |
| H4 | 10 | 30 | 14 |
| H1 | 20 | 100 | 21 |
| W  | 10 | 21 | 7 |

---

## Stage 4 — ATR Stop Sensitivity Sweep

**Objective:** Find the optimal ATR stop multiplier for the OOS period.

**Script:** `research/mtf/run_stage4_atr.py --pair PAIR`

**Multipliers swept:** 1.0, 1.5, 2.0, 2.5, 3.0, 4.0

**Output:**
- `.tmp/reports/mtf_stage4_{pair}_atr_sweep.csv`
- `atr_stop_mult` updated in `config/mtf_{pair_lower}.toml`

**EUR/USD result (locked): `atr_stop_mult = 4.0`**
**GBP/USD result (locked): `atr_stop_mult = 4.0`**

> [!IMPORTANT]
> The trailing ATR stop consistently hurts trend-following strategies at tight multiples.
> At 4.0×, the stop acts as catastrophic-loss insurance only — the primary exit is signal reversal.
> The market often moves temporarily against a position before the trend confirms.
> Signal-only (no stop) produces the best Sharpe; 4.0× stop is the live compromise.

---

## Stage 5 — Portfolio Simulation

**Objective:** Full P&L simulation with all friction: fees, slippage, next-bar execution, swap costs.
Also compares signal-only vs ATR stop scenarios side by side.

**Script:** `research/mtf/run_portfolio.py --pair PAIR`

**What it does:**
- Applies swap cost drag from `config/spread.toml [swap]` (annual % on open position value)
- Reads `atr_stop_mult` from `config/mtf_{pair_lower}.toml` (no hardcoding)
- Plots 4 equity curves: raw stop, raw signal-only, swap-adj signal, swap-adj stop
- Outputs extended comparison table: CAGR, Sharpe, MaxDD, Swap Cost, Adjusted Return, Final Equity

**Output:** `.tmp/reports/mtf_{pair_lower}_comparison.html`

**EUR/USD signal-only result:**

| Metric | Value |
|---|---|
| OOS Sharpe (long) | 2.252 |
| OOS Sharpe (short) | 1.958 |
| Combined Sharpe | 1.943 |
| Swap-adjusted CAGR | ~8%/yr |
| Max Drawdown | ~10% |
| Swap cost (10yr) | ~1.0%/yr drag |

---

## Stage 6 — Robustness (Monte Carlo + WFO)

**Objective:** Stress-test the strategy against random trade-order shuffling and rolling OOS windows.

**Script:** `scripts/robustness_mtf.py --pair PAIR`

**Monte Carlo (N=1,000 trade shuffles):**
- Gate: 5th-pct Sharpe > 0.5 AND >80% simulations profitable

**Rolling WFO (2yr anchor / 6mo windows):**
- Gate: >70% of windows positive AND max consecutive negative ≤ 2

---

## Quality Gates (7 gates — all must pass)

| Gate | Threshold | Scope |
|---|---|---|
| OOS/IS Sharpe ratio | ≥ 0.5 | Both long and short |
| OOS Sharpe | ≥ 1.0 | Both long and short |
| Trades in OOS | ≥ 30 | Both long and short |
| Win Rate | ≥ 40% | Both long and short |
| Max Drawdown | ≤ 25% | Both long and short |
| Monte Carlo 5th-pct Sharpe | > 0.5 AND >80% profitable | Combined |
| WFO positive windows | >70% AND max consec. negative ≤ 2 | Combined |

---

## Final Locked Configs

### EUR/USD — `config/mtf_eurusd.toml`

```toml
ma_type = "WMA"
confirmation_threshold = 0.10
atr_stop_mult = 4.0

[weights]
H1 = 0.10
H4 = 0.05
D  = 0.55
W  = 0.30

[H1]
fast_ma = 10
slow_ma = 50
rsi_period = 14

[H4]
fast_ma = 10
slow_ma = 40
rsi_period = 14

[D]
fast_ma = 10
slow_ma = 20
rsi_period = 7

[W]
fast_ma = 8
slow_ma = 21
rsi_period = 14
```

### GBP/USD — `config/mtf_gbpusd.toml`

```toml
ma_type = "SMA"
confirmation_threshold = 0.10
atr_stop_mult = 4.0

[weights]
H1 = 0.10
H4 = 0.30
D  = 0.55
W  = 0.05

[H1]
fast_ma = 20
slow_ma = 100
rsi_period = 21

[H4]
fast_ma = 10
slow_ma = 30
rsi_period = 14

[D]
fast_ma = 5
slow_ma = 20
rsi_period = 10

[W]
fast_ma = 10
slow_ma = 21
rsi_period = 7
```

---

## State Manager

Per-pair state is persisted at `.tmp/mtf_state_{pair_lower}.json`.
Each stage reads prior stage results automatically when `--load-state` is passed.
This allows stages to be resumed independently without re-running the full pipeline.

---

## When to Re-Run Optimization

Re-run all stages for a pair if:
1. **Model is stale > 6 months** with degrading live performance
2. **New timeframe is being added** (e.g., M15 as entry trigger)
3. **New instrument** beyond existing validated pairs

> [!CAUTION]
> Always run with **fresh OOS data** (extend data window forward, maintain the 70/30 split).
> Never re-optimize on the period previously used as OOS — that is look-ahead bias by another name.
