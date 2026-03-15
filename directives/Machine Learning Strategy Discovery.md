# Directive: Machine Learning Strategy Discovery

> Last updated: 2026-03-14

## Status

**OOS Sharpe: 1.142 (XGBoost)** — Experimental. Below the 1.5 threshold for live deployment.
Treat as a research prototype. Do not deploy without retraining on fresh data and achieving OOS Sharpe ≥ 1.5.

> [!IMPORTANT]
> An ML meta-overlay was also tested on top of the MTF Confluence strategy and **rejected**.
> The meta-model destroyed OOS Sharpe from 2.73 → 0.83. Do not add an ML filter to MTF.
> See `directives/MTF Optimization Protocol.md` for the full experiment results.

---

## Goal

Train and validate a Machine Learning model to predict **price direction** (3-class classification:
LONG / SHORT / FLAT) using VectorBT for pipeline evaluation.

## Inputs

- Raw Parquet data from `data/`
- Feature definitions from `config/features.toml` (auto-tuned by feature selection sweep)

---

## Execution Steps

### 1. Unified Pipeline

Run the full pipeline:
```bash
uv run python research/ml/run_pipeline.py
```

This handles:
1. **Feature Engineering**: Builds 30+ features (RSI, SMA cross, EMA cross, MACD, Bollinger Bands, Stochastic, ADX, MTF bias D/W SMA direction). All features lagged by 1 bar (`shift(1)`) to prevent look-ahead.
2. **Target Engineering**: 3-class target (LONG/SHORT/FLAT) — next-bar direction vs. current close ± 0.5×ATR threshold.
3. **Model Training**: Walk-forward cross-validation (5 folds). Models evaluated: XGBoost, HistGradientBoosting, RandomForest.
4. **VBT Backtest**: Tests signals on OOS data with full friction.

> [!CAUTION]
> **Look-ahead Bias:** All features MUST use `.shift(1)`. If OOS Sharpe > 3.0, suspect data leakage.
> The pipeline handles this automatically — verify before any manual modifications.

### 2. Performance Evaluation

Review console output and JSON report in `.tmp/reports/`.

Key metrics (report in this order):
1. IS Sharpe
2. OOS Sharpe
3. OOS/IS ratio (reject if < 0.5)
4. Win Rate
5. Max Drawdown

**Current best (XGBoost):** OOS Sharpe 1.142 — below deployment threshold.

### 3. Feature Re-Selection (if data is stale > 1 month)

```bash
uv run python research/ml/run_feature_selection.py
```

Sweeps 7 indicator families across parameter ranges. Scores candidates by
`Stability = min(IS, OOS) / max(IS, OOS)`. Writes winning parameters to `config/features.toml`.

### 4. Model Serialisation

Best model (highest Validation Sharpe) is automatically saved to `models/` as a `.joblib` file.

### 5. Deployment

The saved model is picked up by `scripts/run_live_ml.py`.
Requires OOS Sharpe ≥ 1.5 before deploying.

---

## Common Errors

### `FileNotFoundError: models/ml_strategy_*.joblib`
No trained model. Run `uv run python research/ml/run_pipeline.py` first.

### Feature mismatch at inference
`config/features.toml` changed after training. Retrain model with current feature set.

### Accuracy degrading over time
Market regime shift. Re-run `run_feature_selection.py` + `run_pipeline.py` with fresh data.

### `TypeError: Argument 'position_id' has incorrect type`
Use `cache.positions(instrument_id=...)` to list positions, not `cache.position(instrument_id)`:
```python
positions = self.cache.positions(instrument_id=self.instrument_id)
position = positions[-1] if positions else None
```
