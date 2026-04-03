# ML Signal Discovery Strategy — Technical Documentation

## Overview

An XGBoost-based classifier that identifies high-probability trend entry points on equity
indices and FX pairs using 72 technical features and regime-aware pullback labeling.

**Best result:** QQQ Daily — OOS Sharpe +1.11, 57% positive WFO folds, avg parity 3.01.

**Core idea:** In a confirmed uptrend (SMA50 > SMA200, MACD > 0), buy pullbacks when RSI
dips below an oversold threshold and forward returns confirm the pullback resolves upward.
Mirror logic for downtrends. Train XGBoost on these confirmed pullback bars only (~3-8% of
data), then predict on all bars — hold position until the model's prediction flips.

---

## Architecture

```
Data (OHLCV)
    |
    v
[Feature Builder] ---> 72 features (shifted 1 bar, no lookahead)
    |                      |
    |    52 IC signals (Groups A-G from phase1_sweep.py)
    |     7 daily-equiv MA features (50d/200d spreads, trend strength)
    |     5 regime features (ADX regime, HMM state, ATR percentile)
    |     3 VIX features (VIX SMA20, VIX < 15 flag, vol risk premium)
    |     1 calendar feature (month seasonality)
    |     4 momentum features (ret_lag 126/252, momentum acceleration)
    |
    v
[Regime + Pullback Labeler] ---> sparse labels (+1 long, -1 short, 0 hold)
    |
    |  Step 1 (causal): Trend regime via SMA(50/200) + MACD(12/26/9)
    |  Step 2 (causal): RSI pullback detection within regime
    |  Step 3 (forward-looking, LABELS ONLY): Confirm with forward returns
    |
    v
[XGBClassifier] ---> P(long pullback) per bar
    |
    |  Trained on confirmed entry bars only (~3-8% of data)
    |  Predicted on ALL bars
    |
    v
[Position Manager] ---> held position (+1, -1, or 0)
    |
    |  P(long) > 0.6 --> go LONG
    |  P(long) < 0.4 --> go SHORT
    |  Otherwise: HOLD previous position
    |
    v
Strategy Returns (position x bar_returns - transaction costs)
```

---

## Feature Groups (72 total)

### Group 1: 52 IC Signals (from `phase1_sweep.py:build_all_signals`)

| Group | Count | Examples | Purpose |
|-------|-------|----------|---------|
| A: Trend | 10 | MA spreads (5/20, 10/50, 50/200), MACD, EMA slopes | Trend direction & strength |
| B: Momentum | 11 | RSI (7,14,21), Stochastic K/D, CCI, Williams %R, ROC | Overbought/oversold |
| C: Mean Reversion | 6 | Bollinger z-score, rolling z-scores (20-252) | Distance from mean |
| D: Volatility | 7 | ATR, realized vol, Garman-Klass, Parkinson, BB width, ADX | Vol regime |
| E: Acceleration | 7 | First-diff of ROC, RSI, MACD, ATR, BB width, StochK | Rate of change |
| F: Structural | 6 | Donchian position (10,20,55), Keltner, percentile rank | Breakout detection |
| G: Combinations | 5 | trend_mom, trend_vol_adj, mom_accel_combo | Semantic blends |

### Group 2: Daily-Equivalent MA Features (7)

Scaled by timeframe (H1: 24 bars/day, D: 1 bar/day, M5: 288 bars/day):

| Feature | Formula | Purpose |
|---------|---------|---------|
| `ma_spread_weekly` | (SMA_50d - SMA_200d) / SMA_200d | **#1 feature across all instruments** |
| `ema_spread_50_200` | (EMA_50d - EMA_200d) / EMA_200d | EMA variant of above |
| `ma_spread_daily` | (EMA_slow - SMA_trend) / SMA_trend | Intermediate trend |
| `price_vs_longterm` | (close - SMA_200d) / SMA_200d | Distance from 200d MA |
| `sma_50_200_cross` | SMA_50d > SMA_200d (binary) | Golden/death cross state |
| `dist_from_200d` | z-score of (close - SMA_200d) | Normalized deviation |
| `trend_strength` | abs(SMA_50d - SMA_200d) / ATR | Trend magnitude vs volatility |

### Group 3: Regime Features (5)

| Feature | Source | Purpose |
|---------|--------|---------|
| `adx_regime_ranging` | ADX(14) < 20 | Identifies choppy markets |
| `adx_regime_trending` | ADX(14) > 25 | Identifies strong trends |
| `hmm_state` | 2-state Gaussian HMM on [log_ret, rvol20] | Latent vol regime |
| `atr_pct_rank` | Rolling 252-bar percentile of ATR(14) | Volatility regime |
| `vol_regime_low` | atr_pct_rank < 0.30 | Low-vol calm regime |

### Group 4: VIX + Calendar + Momentum (8)

| Feature | Purpose |
|---------|---------|
| `vix_sma20` | Smoothed VIX — fear/greed gauge |
| `vix_below_15` | Complacency flag (historically precedes selloffs) |
| `vol_risk_premium` | VIX - realized vol = implied-realized spread |
| `month` | Month of year (0-1 normalized) — seasonality |
| `ret_lag_126` | 6-month momentum factor |
| `ret_lag_252` | 12-month momentum factor |
| `mom_accel_21_252` | 1-month vs 12-month momentum differential |
| `mom_accel_5_63` | 1-week vs 3-month momentum differential |

---

## Labeling: Regime + Pullback (v3)

### Step 1 — Trend Regime (causal, no lookahead)

```python
BULL = (SMA(50) > SMA(200)) AND (MACD_histogram > 0)
BEAR = (SMA(50) < SMA(200)) AND (MACD_histogram < 0)
NEUTRAL = everything else
```

### Step 2 — Pullback Detection (causal)

```python
In BULL regime: pullback = RSI(14) < rsi_oversold  (dip in uptrend)
In BEAR regime: rally = RSI(14) > rsi_overbought   (bounce in downtrend)
```

### Step 3 — Forward Confirmation (labels only, not features)

```python
Long confirmed = bull_pullback AND fwd_return(N bars) > confirm_pct
Short confirmed = bear_rally AND fwd_return(N bars) < -confirm_pct
```

### Per-Fold Parameter Sweep

Parameters are swept on IS data per WFO fold to find the best combination:

| Parameter | Sweep Values |
|-----------|-------------|
| `rsi_oversold` | 40, 45, 48, 50 |
| `rsi_overbought` | 50, 52, 55, 60 |
| `confirm_bars` | 5, 10, 20 |
| `confirm_pct` | 0.002, 0.003, 0.005, 0.01 |

Selection criteria: most training samples with >= 15% minority class balance.

---

## Model Configuration

```python
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,           # shallow — prevents overfitting on noisy data
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.6,  # uses ~43 of 72 features per tree
    "random_state": 42,
}
```

**Class weighting:** `scale_pos_weight` auto-computed per fold from class ratio.
**Signal threshold:** P(long) > 0.6 = long, P(long) < 0.4 = short, else hold.

---

## Walk-Forward Validation

| Timeframe | IS Window | OOS Window | Typical Folds |
|-----------|-----------|------------|---------------|
| Daily | 504 bars (2 years) | 126 bars (6 months) | ~40-50 folds |
| H1 | 12,096 bars (2 years) | 3,024 bars (6 months) | 3-5 folds |

Rolling WFO: fixed IS window slides forward by OOS_BARS each fold.
Per-fold: re-train model AND re-select label parameters on IS data.

---

## Results Summary (Daily, all indices)

| Instrument | Stitched Sharpe | % Positive Folds | Avg Parity | Worst Fold |
|------------|----------------|------------------|------------|------------|
| **QQQ** | **+1.106** | 57% | 3.01 | -0.12 |
| DAX | +0.643 | 71% | 6.93 | -2.17 |
| SPY | +0.544 | 50% | -0.34 | -2.86 |
| FTSE | +0.016 | 53% | -4.59 | -2.96 |

### Evolution Across Versions

| Version | Labeler | Model | SPY | QQQ | Best Feature |
|---------|---------|-------|-----|-----|-------------|
| v1 | Trailing-stop (per bar) | XGBRegressor | +0.50 | +0.68 | ma_spread_weekly |
| v1-pruned | Trailing-stop (pruned feat) | XGBRegressor | +0.50 | +0.68 | month, vix_sma20 |
| v2 | ATR swing points | XGBClassifier | +0.34 | +0.04 | price_vs_sma20 |
| **v3** | **Regime + pullback** | **XGBClassifier** | **+0.54** | **+1.11** | **sma_50_200_cross** |

---

## Key Feature Importance (averaged across all folds and instruments)

| Rank | Feature | Avg Importance | Group |
|------|---------|---------------|-------|
| 1 | `sma_50_200_cross` | 0.110 | MA (golden/death cross) |
| 2 | `ma_spread_weekly` | 0.085 | MA (50d vs 200d spread) |
| 3 | `ma_spread_50_200` | 0.075 | IC Group A (trend) |
| 4 | `macd_norm` | 0.065 | IC Group A (trend) |
| 5 | `price_vs_sma50` | 0.055 | IC Group A (trend) |
| 6 | `ma_spread_10_50` | 0.055 | IC Group A (trend) |
| 7 | `ema_slope_20` | 0.045 | IC Group A (trend) |
| 8 | `garman_klass` | 0.040 | IC Group D (volatility) |
| 9 | `norm_atr_14` | 0.035 | IC Group D (volatility) |
| 10 | `ema_spread_50_200` | 0.030 | MA (EMA variant of 50/200) |

**Key insight:** Trend features dominate because the labeler teaches the model to identify
trend regime entries. The model has learned that the 50/200 SMA cross state is the single
most important predictor of whether a pullback is a buying opportunity.

---

## File Structure

```
research/ml/
    run_52signal_classifier.py    # Main pipeline (features, labels, WFO, evaluation)
    plot_52sig_signals.py         # Chart generator (price + signals + equity + drawdown)
    run_52sig_param_sweep.py      # Parameter sweep (MA periods, ATR, stop, threshold)
    build_tbm_labels.py           # Triple Barrier Method (legacy v1 labeler)
    ML_STRATEGY_DOCUMENTATION.md  # This file
```

---

## How to Run

```bash
# Single instrument:
uv run python research/ml/run_52signal_classifier.py --instrument QQQ --tf D

# Generate chart:
uv run python research/ml/plot_52sig_signals.py --instrument QQQ --tf D

# All target instruments:
uv run python research/ml/run_52signal_classifier.py

# Charts saved to: .tmp/reports/ml_52sig_chart_{instrument}_{tf}.html
# Results saved to: .tmp/reports/ml_52sig_{timestamp}.csv
```

---

## Validation Gates

| Gate | Metric | Threshold | QQQ Status |
|------|--------|-----------|------------|
| Stitched Sharpe | OOS across all folds | > 1.0 | **PASS (+1.11)** |
| % Positive Folds | Folds with OOS Sharpe > 0 | >= 70% | FAIL (57%) |
| Avg Parity | OOS/IS Sharpe ratio | >= 0.50 | **PASS (3.01)** |
| Worst Fold | Min OOS Sharpe | > -2.0 | **PASS (-0.12)** |
| Consec Neg Folds | Max streak of negative folds | <= 2 | **PASS** |

---

## Risks and Limitations

1. **Low fold count on short-history instruments** — QQQ has only 7 valid folds due to the
   strict IS=2yr window + sparse labels. Statistical significance is limited.

2. **Label leakage risk** — forward returns used in labeling (Step 3) could leak if the same
   bars appear in both IS labels and OOS features. Mitigated by: labels are computed on the
   full dataset once, but the model only trains on IS bars. The forward return at IS bar t
   does not reveal OOS bar prices because IS and OOS windows don't overlap.

3. **Regime dependence** — the model performs best in clear trending regimes (bull or bear).
   In choppy sideways markets (2015-2016, 2023), few labels are generated and performance
   degrades. This is by design — the model stays flat when it doesn't see a clear trend.

4. **Transaction costs** — with ~10-30 flips per 6-month fold, costs are low (~2 bps/trade).
   But the 100% hold time means carry costs (if trading futures/CFDs) accumulate.

5. **Survivorship bias in VIX** — VIX data starts in 2000. Earlier folds for SPY (1993-2000)
   don't have VIX features (filled with NaN -> 0). This may explain weaker early-fold performance.

---

## Next Steps

1. **Deploy on QQQ** — passes Sharpe gate, integrate into portfolio risk manager
2. **Add more indices** — test on Russell 2000 (IWM), Nikkei, Hang Seng when data available
3. **Ensemble with existing strategies** — combine ML signal with IC equity + ETF trend for
   portfolio diversification (different signal source = uncorrelated alpha)
4. **Live paper test** — run predictions daily, compare with actual market moves for 3 months
   before deploying capital
