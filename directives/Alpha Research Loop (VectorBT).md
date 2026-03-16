# Directive: Alpha Research Loop (VectorBT)

> Last updated: 2026-03-14

## Goal

Identify and optimise a viable **swing trading** strategy for **EUR/USD** using vectorised
backtesting on higher timeframes. The MTF Confluence strategy is the primary validated output
of this loop (Round 3, OOS Sharpe 2.936 — see `directives/MTF Optimization Protocol.md`).

## Timeframes

| Granularity | Use Case |
|---|---|
| **H1** | Entry timing |
| **H4** | Swing confirmation |
| **D** | Primary trend bias |
| **W** | Long-term regime filter |

## Inputs

- Parameter ranges (e.g., RSI 10–25, MA 20–100, thresholds, weights)
- Historical Parquet files from `data/` (downloaded by `scripts/download_data.py`)

---

## Execution Steps

### 1. Data Ingestion

Pull 2+ years of H1/H4/D/W OHLC data:
```bash
uv run python scripts/download_data_mtf.py
```

Stores in `data/` as Parquet files (`EUR_USD_H1.parquet`, `EUR_USD_H4.parquet`, etc.).

### 2. Data Validation

Check for gaps, duplicates, and outlier spikes:
```bash
uv run python titan/data/validation.py
```

### 3. Strategy Optimisation (In-Sample)

Run the MTF optimisation pipeline (3 stages, greedy):
```bash
# Stage 1: MA type + threshold
uv run python research/mtf/run_optimisation.py --pair EUR_USD

# Stage 2: Timeframe weights
uv run python research/mtf/run_stage2.py --pair EUR_USD --load-state

# Stage 3: Per-timeframe indicator periods
uv run python research/mtf/run_pair_sweep.py --pair EUR_USD --load-state
```

Data is split **70% in-sample / 30% out-of-sample**. Optimisation runs on IS data only.
Best parameters are auto-saved to `.tmp/mtf_state_{pair}.json` between stages.

### 4. Out-of-Sample Validation

Best IS candidates are tested on the held-out OOS data.

**Reject** any candidate whose OOS Sharpe drops below 50% of IS Sharpe (overfitting signal).

Additional gates (see `resources/VectorBT Credible Backtesting Guide.md`):
- Slippage stress test (5 levels)
- ATR stop sensitivity sweep (8 multipliers)
- Monte Carlo (N=1,000)
- Rolling Walk-Forward Optimisation (2yr anchor / 6mo windows)

### 5. ATR Stop Sensitivity Sweep

Separate sweep to find optimal hard stop distance. Run after Stage 3:
```bash
uv run python research/mtf/run_stage4_atr.py --pair EUR_USD
```

Result: `atr_stop_mult = 4.0` (see `directives/MTF Optimization Protocol.md`).

### 6. MTF Confluence Backtest Report

Generate the full IS/OOS report:
```bash
uv run python scripts/run_backtest_meta.py
```

Generates HTML reports in `.tmp/reports/`.

### 6.5. Gaussian Channel (Research Complete)

The Gaussian Channel indicator research is complete:
- Implementation: `titan/indicators/gaussian_filter.py` (Numba `@njit`)
- Optimisation script: `research/gaussian/run_optimisation.py`
- Config: `config/legacy/gaussian_channel_config.toml`
- **Current use:** Deployed as a per-ticker filter in the ORB strategy (`use_gauss` flag).
- **Not a standalone strategy** — Gaussian Channel Confluence strategy was deprecated after
  the MTF strategy proved superior on the same data.

### 7. Parameter Lock

Once validation passes, write the final configuration:
```bash
# Parameters auto-saved by Stage 3; manually verify config/mtf.toml matches .tmp/mtf_state.json
cat config/mtf.toml
```

### 8. VBT → ML Bridge (Feature Selection)

If pursuing the ML strategy after locking MTF parameters:
```bash
uv run python research/ml/run_feature_selection.py
```

Sweeps 7 indicator families. Writes winning parameters to `config/legacy/features.toml` for ML pipeline.

> [!NOTE]
> An ML meta-overlay on top of MTF was tested and **rejected** (OOS Sharpe 0.83 vs raw 2.73).
> The ML pipeline is independent of MTF and targets its own instrument/timeframe setup.

---

## Outputs

- `config/mtf_{pair}.toml` — locked optimised parameters per pair (Round 4 validated)
- `.tmp/reports/mtf_stage*.csv` — per-stage scoreboards
- `.tmp/reports/mtf_meta_*.html` — IS/OOS equity curves
- `config/legacy/features.toml` — ML feature parameters (if ML path taken)
