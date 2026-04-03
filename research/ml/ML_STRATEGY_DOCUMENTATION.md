# ML Signal Discovery Strategy — Technical Documentation

## Overview

An XGBoost-based classifier that identifies high-probability trend entry points across
FX, equity indices, gold, silver, and miners using ~79 technical features (52 IC signals
+ 7 MA + 5 regime + 3 VIX + 1 calendar + 4 momentum + 7 stochastic oscillator) and
regime-aware pullback labeling.

**Universe:** 28 instruments x 3 timeframes (D, H1, H4) = 38 unique signals scanned.
Daily is the dominant timeframe -- all viable signals are on Daily bars.

**Best results (April 2026):**
- EUR_USD Daily: Sharpe +1.58, 100% positive folds (3 folds)
- QQQ Daily: Sharpe +1.11, 57% positive folds (7 folds)
- PSKY Daily (gold miners): Sharpe +0.55, 69% positive folds (13 folds -- most robust)

**Core idea:** In a confirmed uptrend (SMA50 > SMA200, MACD > 0), buy pullbacks when RSI
dips below an oversold threshold and forward returns confirm the pullback resolves upward.
Mirror logic for downtrends. Train XGBoost on these confirmed pullback bars only (~3-8% of
data), then predict on all bars -- hold position until the model's prediction flips.

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

## Feature Groups (~79 total)

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

### Group 5: Stochastic Oscillator (7)

| Feature | Purpose |
|---------|---------|
| `stoch_k_raw` | Fast stochastic %K (14-period) -- raw overbought/oversold |
| `stoch_d_raw` | Slow stochastic %D (3-period smoothed %K) |
| `stoch_k_minus_d` | %K - %D momentum -- positive = bullish crossover |
| `stoch_overbought` | %K > 80 flag (binary) |
| `stoch_oversold` | %K < 20 flag (binary) |
| `stoch_k_slow` | Slow stochastic %K (21-period) -- longer-term view |
| `stoch_k_slow_minus_fast` | Divergence between slow and fast stochastic |

Note: The 52 IC signals already include `stoch_k_dev` and `stoch_d_dev` (Group B, deviation
from 50). The raw stochastic features complement these by providing absolute levels (overbought
/oversold at 80/20) and multi-period divergence that the z-scored IC versions don't capture.

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
| H4 | 3,024 bars (2 years) | 756 bars (6 months) | ~18 folds |
| H1 | 12,096 bars (2 years) | 3,024 bars (6 months) | 3-38 folds |

Rolling WFO: fixed IS window slides forward by OOS_BARS each fold.
Per-fold: re-train model AND re-select label parameters on IS data.

---

## Cross-Asset Signal Map (April 2026, 28 instruments x 3 timeframes)

Full universe scan: 6 FX pairs, 7 equity indices, 3 gold, 4 silver, 4 miners, DXY.
Each instrument tested on Daily, H1, and H4 where data is available.

### Tier A -- Deploy candidates (Sharpe >= 1.0, >=50% positive, >=3 folds)

| Instrument | TF | Category | Sharpe | %Pos | Folds |
|------------|:--:|----------|-------:|-----:|------:|
| **EUR_USD** | D | FX | **+1.584** | 100% | 3 |
| **QQQ** | D | Index | **+1.106** | 57% | 7 |

### Tier B -- Strong signals (Sharpe >= 0.5, >=50% positive)

| Instrument | TF | Category | Sharpe | %Pos | Folds |
|------------|:--:|----------|-------:|-----:|------:|
| TQQQ | D | Index | +0.961 | 67% | 3 |
| USD_CHF | D | FX | +0.862 | 60% | 5 |
| ES | D | Index | +0.777 | 62% | 13 |
| IWB | D | Index | +0.694 | 50% | 4 |
| SPY | D | Index | +0.555 | 75% | 4 |
| PSKY | D | Miner | +0.554 | 69% | 13 |
| SIL | D | Miner | +0.536 | 100% | 1 |
| AUD_USD | D | FX | +0.534 | 60% | 5 |

### Tier C -- Weak positive (not tradeable)

| Instrument | TF | Category | Sharpe | %Pos | Folds |
|------------|:--:|----------|-------:|-----:|------:|
| USD_JPY | D | FX | +0.337 | 100% | 1 |
| AUD_JPY | D | FX | +0.333 | 100% | 1 |
| SLV | D | Silver | +0.231 | 67% | 6 |
| PSLV | D | Silver | +0.179 | 50% | 8 |
| GDXJ | D | Miner | +0.147 | 57% | 14 |
| AUD_JPY | H4 | FX | +0.136 | 56% | 18 |
| ^FTSE | D | Index | +0.099 | 65% | 17 |
| ^GDAXI | D | Index | +0.097 | 71% | 7 |
| QQQ | H1 | Index | +0.068 | 50% | 4 |
| GBP_USD | H4 | FX | +0.005 | 50% | 18 |

### Tier D -- No signal (18 signals, all negative Sharpe)

All H1 FX, all H4 FX (except marginal AUD_JPY/GBP_USD), all gold (GLD, IAU, GC=F),
and some silver (SI=F, SIVR).

### Key Findings

1. **Daily is the dominant timeframe.** All Tier A and B signals are Daily bars. H1 and
   H4 produced zero signals with Sharpe > 0.5. The regime+pullback approach requires
   daily-scale trend regimes to be effective.

2. **Four uncorrelated signal buckets:**

   | Bucket | Instruments | Correlation | Best Pick |
   |--------|-------------|-------------|-----------|
   | US Equity | QQQ, SPY, ES, IWB, TQQQ | ~0.95 (treat as ONE) | QQQ (Sharpe 1.11, 7 folds) |
   | FX | EUR_USD, USD_CHF, AUD_USD | Moderate | EUR_USD (Sharpe 1.58, 3 folds) |
   | Miners | PSKY, SIL | Low to equity/FX | PSKY (Sharpe 0.55, **13 folds** -- most robust) |
   | Gold/Silver | GLD, IAU, SLV, GC=F, SI=F | N/A | **No signal** -- model cannot predict |

3. **PSKY (gold miners) is the most statistically robust signal.** 13 folds with 69%
   positive -- the highest fold count of any Tier B signal. Uncorrelated to equity
   and FX, making it the best diversifier for portfolio construction.

4. **Gold and silver are untradeable** with this approach. GLD, IAU, GC=F all have
   negative Sharpe. Gold doesn't follow SMA50/200 + MACD regime patterns.

5. **European indices (FTSE, DAX) are marginal.** Near-zero stitched Sharpe despite
   long histories (10K+ bars, 17+ folds). The trend signal exists but is too weak.

### Comprehensive Evaluation Stats (Tier A instruments)

| Metric | QQQ D | EUR_USD D | SPY D |
|--------|------:|----------:|------:|
| Total Return | +109.2% | +19.3% | +20.1% |
| CAGR | +23.5% | +12.5% | +9.6% |
| Sharpe | +1.11 | +1.58 | +0.56 |
| Sortino | +1.40 | +2.36 | +0.71 |
| Calmar | 1.24 | 3.12 | 0.47 |
| Max Drawdown | -19.0% | -4.0% | -20.5% |
| Max DD Duration | 371d | 121d | 389d |
| Trades | 12 | 3 | 4 |
| Win Rate | 75% | 67% | 75% |
| Profit Factor | 7.71 | 9.04 | 2.87 |
| Avg Capital Invested | 100% | 94% | 100% |
| Risk of Ruin (50%) | 0.6% | 0.0% | 0.5% |

---

### Evolution Across Versions

| Version | Labeler | Model | SPY | QQQ | Best Feature |
|---------|---------|-------|-----|-----|-------------|
| v1 | Trailing-stop (per bar) | XGBRegressor | +0.50 | +0.68 | ma_spread_weekly |
| v2 | ATR swing points | XGBClassifier | +0.34 | +0.04 | price_vs_sma20 |
| **v3** | **Regime+pullback** | **XGBClassifier** | **+0.54** | **+1.11** | **sma_50_200_cross** |
| v3+stoch | Regime+pullback | XGBClassifier (79 feat) | +0.55 | +1.11 | ema_slope_20 |

---

## Key Feature Importance (averaged across all folds and instruments)

| Rank | Feature | Avg Importance | Group |
|------|---------|---------------|-------|
| 1 | `ema_slope_20` | 0.091-0.116 | IC Group A (trend) |
| 2 | `ma_spread_10_50` | 0.053-0.089 | IC Group A (trend) |
| 3 | `sma_50_200_cross` | 0.088 | MA (golden/death cross) |
| 4 | `ma_spread_weekly` | 0.082 | MA (50d vs 200d spread) |
| 5 | `macd_norm` | 0.065-0.096 | IC Group A (trend) |
| 6 | `ma_spread_daily` | 0.072 | MA (intermediate) |
| 7 | `ma_spread_50_200` | 0.043-0.061 | IC Group A (trend) |
| 8 | `bb_zscore_50` | 0.053 | IC Group C (mean rev) |
| 9 | `price_vs_sma50` | 0.050 | IC Group A (trend) |
| 10 | `mom_accel_21_252` | 0.030 | Momentum (acceleration) |

**Key insight:** Trend features dominate (7 of top 10) because the labeler teaches the
model to identify trend regime entries. EMA slope and MA spreads are the most consistent
predictors across all instruments.

---

## File Structure

```
research/ml/
    run_52signal_classifier.py    # Main pipeline (features, labels, WFO, evaluation)
    run_ml_full_eval.py           # Comprehensive evaluation (equity, stats, risk of ruin, charts)
    run_52sig_param_sweep.py      # Parameter sweep (MA periods, ATR, stop, threshold)
    plot_52sig_signals.py         # Chart generator (price + signals + equity + drawdown)
    run_pipeline.py               # Alternate pipeline (3-class target, VBT backtest)
    run_metalabeling.py           # Meta-labeling (MTF confluence + TBM + walk-forward CV)
    build_tbm_labels.py           # Triple Barrier Method labeler
    build_features.py             # Legacy feature builder (older pipeline)
    train_model.py                # Legacy model trainer (older pipeline)
    run_ensemble.py               # Multi-strategy ensemble framework
    ML_STRATEGY_DOCUMENTATION.md  # This file
```

---

## How to Run

```bash
# Single instrument (daily):
uv run python research/ml/run_52signal_classifier.py --instrument QQQ --tf D

# All instruments on daily:
uv run python research/ml/run_52signal_classifier.py --tf D

# All instruments on default timeframe (H1 for FX, D for indices):
uv run python research/ml/run_52signal_classifier.py

# Full evaluation with comprehensive stats + charts (QQQ, SPY, EUR_USD Daily):
uv run python research/ml/run_ml_full_eval.py
uv run python research/ml/run_ml_full_eval.py --instrument QQQ

# Parameter sweep (daily, fast mode):
uv run python research/ml/run_52sig_param_sweep.py --tf D --fast
uv run python research/ml/run_52sig_param_sweep.py --instrument QQQ --tf D

# Generate signal chart:
uv run python research/ml/plot_52sig_signals.py --instrument QQQ --tf D

# Results saved to: .tmp/reports/ml_52sig_{timestamp}.csv
# Eval charts:      .tmp/reports/ml_eval_{instrument}_{tf}.html
# Signal charts:    .tmp/reports/ml_52sig_chart_{instrument}_{tf}.html
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

## Full Evaluation Stats + Parameter Sweep (Daily)

Comprehensive evaluation via `run_ml_full_eval.py` (Tier A instruments):

| Metric | QQQ D | EUR_USD D | SPY D |
|--------|------:|----------:|------:|
| Total Return | +109.2% | +19.3% | +20.1% |
| CAGR | +23.5% | +12.5% | +9.6% |
| Sharpe | +1.11 | +1.58 | +0.56 |
| Sortino | +1.40 | +2.36 | +0.71 |
| Calmar | 1.24 | 3.12 | 0.47 |
| Max Drawdown | -19.0% | -4.0% | -20.5% |
| Max DD Duration | 371d | 121d | 389d |
| Trades | 12 | 3 | 4 |
| Win Rate | 75% | 67% | 75% |
| Profit Factor | 7.71 | 9.04 | 2.87 |
| Avg Capital Invested | 100% | 94% | 100% |
| Risk of Ruin (50%) | 0.6% | 0.0% | 0.5% |

Parameter sweep (daily, fast mode -- trailing-stop labels):

| Instrument | Stitched Sharpe | % Positive Folds | Folds | Best Params |
|------------|----------------|------------------|-------|-------------|
| QQQ | +0.829 | 86% | 14 | EMA(20/50) trend=1200 lt=4800 |
| SPY | +0.568 | 74% | 27 | EMA(20/50) trend=1200 lt=4800 |
| EUR_USD | +0.438 | 64% | 11 | SMA(15/100) trend=1200 lt=4800 |

### Portfolio Construction Candidates (from full universe scan)

Four uncorrelated signal buckets for portfolio construction:

| Bucket | Best Pick | Sharpe | Folds | Other Candidates |
|--------|-----------|-------:|------:|------------------|
| **US Equity** | QQQ D | +1.11 | 7 | SPY, ES, IWB, TQQQ (all correlated ~0.95) |
| **FX** | EUR_USD D | +1.58 | 3 | USD_CHF (+0.86), AUD_USD (+0.53) |
| **Miners** | PSKY D | +0.55 | **13** | SIL (+0.54, 1 fold only) |
| **Gold/Silver** | -- | -- | -- | No signal (GLD, IAU, SLV, GC=F all negative) |

> **PSKY is the most statistically robust signal** (13 folds, 69% positive) and the
> best portfolio diversifier (low correlation to equity and FX buckets).

---

## Backtest Assumptions and Known Problems

> **CRITICAL: The current backtest is a signal quality test, not a portfolio simulation.**
> The returns reported above are NOT achievable as-is. They measure whether the ML model
> can predict direction, not what a real trading account would produce.

### What the backtest assumes

| Assumption | Reality |
|------------|---------|
| 100% of capital in one instrument | No portfolio allocation across assets |
| Position is always +1.0 or -1.0 | No position sizing -- always "all in" long or short |
| No leverage modelling | Position of 1.0 = 1x notional, no margin calculation |
| Each instrument runs in isolation | QQQ, SPY, EUR_USD treated as 3 independent universes |
| No concurrent trade limit | Irrelevant because each instrument is solo |
| No portfolio construction | Summary table stacks results, does not model shared capital |
| Transaction costs = flat bps | No slippage model, no market impact, no spread widening |
| Instant fills at close price | No execution delay, no partial fills |

### Problems this creates

1. **Overstated returns** — the +109% QQQ return assumes you put 100% of capital into QQQ
   for 3.5 years, going long or short every single day. No real account would do this.

2. **Capital double-counting** — the summary table implies you could get QQQ (+109%) AND
   SPY (+20%) AND EUR_USD (+19%) simultaneously, but that would require 3x capital.

3. **No risk budgeting** — when QQQ and SPY both signal long (which they often do, since
   they're correlated), the portfolio would be 200% long US equities with no diversification.

4. **Missing position sizing** — a 1R stop at 2x ATR on QQQ ($500) is ~$30 risk per share.
   Without sizing, we don't know how many shares to trade for a given account size.

5. **Carry costs ignored** — 100% invested at all times means short positions incur borrow
   costs, and leveraged positions incur margin interest. Not modelled.

---

## What's Missing for a Real Portfolio

To turn these signals into a deployable system, the following portfolio risk layer is needed:

### 1. Position Sizing
- **ATR-based sizing**: Risk a fixed % of capital (e.g. 1-2%) per trade's stop distance.
  `shares = (account * risk_pct) / (ATR * stop_mult)`.
- **Kelly criterion**: Optimal fraction based on win rate and payoff ratio.
  With QQQ's 75% WR and 2.57 W/L ratio, Kelly = 0.75 - (0.25/2.57) = 65%.
  Half-Kelly (~33%) is the practical maximum.

### 2. Multi-Asset Allocation
- When QQQ, SPY, EUR_USD all signal simultaneously, how to split capital.
- Options: equal weight, risk parity (inverse-vol), or signal-strength-weighted.
- Must enforce a max gross exposure cap (e.g. 150% notional).

### 3. Correlation Filter
- QQQ and SPY are highly correlated (~0.95). Treating them as independent doubles
  the effective equity exposure when both signal the same direction.
- Solution: when correlated assets agree, treat as one position with combined weight
  capped at the single-asset maximum.

### 4. Risk Limits
- **Per-position max**: No single position > 30% of capital.
- **Portfolio max drawdown**: If portfolio DD exceeds 15%, reduce all positions by 50%.
- **Daily loss limit**: If daily P&L < -3%, flatten all positions until next session.
- **Sector concentration**: Cap equity exposure at 60% of portfolio.

### 5. Execution Model
- Slippage: 0.5-2 bps for liquid ETFs (QQQ/SPY), 1-3 bps for FX.
- Fills at next-bar open (not current-bar close) to simulate realistic entry.
- Partial fill modelling for large position changes.

### 6. Implementation Priority
1. ATR-based position sizing (required before any live deployment)
2. Multi-asset allocation with gross exposure cap
3. Correlation-aware weight adjustment
4. Portfolio-level drawdown circuit breaker
5. Realistic execution model (next-bar-open fills + slippage)

---

## Risks and Limitations

1. **Low fold count on short-history instruments** -- EUR_USD Daily has only 3 valid folds
   (Sharpe +1.58 but low statistical confidence). QQQ has 7 folds. Only PSKY (13 folds),
   ES (13 folds), and ^FTSE (17 folds) have truly robust sample sizes.

2. **Label leakage risk** -- forward returns used in labeling (Step 3) could leak if the
   same bars appear in both IS labels and OOS features. Mitigated by: labels are computed
   on the full dataset once, but the model only trains on IS bars. The forward return at
   IS bar t does not reveal OOS bar prices because IS and OOS windows don't overlap.

3. **Regime dependence** -- the model performs best in clear trending regimes (bull or bear).
   In choppy sideways markets (2015-2016, 2023), few labels are generated and performance
   degrades. This is by design -- the model stays flat when it doesn't see a clear trend.

4. **Transaction costs** -- with ~10-30 flips per 6-month fold, costs are low (~2 bps/trade).
   But the 100% hold time means carry costs (if trading futures/CFDs) accumulate.

5. **Survivorship bias in VIX** -- VIX data starts in 2000. Earlier folds for SPY (1993-2000)
   don't have VIX features (filled with NaN -> 0). This may explain weaker early-fold perf.

6. **No portfolio risk management** -- see "Backtest Assumptions" section above. The current
   evaluation is a signal quality test, not a portfolio simulation. Returns are not
   achievable without position sizing and capital allocation.

7. **Gold/silver untradeable** -- the regime+pullback approach produces no signal on GLD,
   IAU, GC=F, SI=F. These assets don't follow SMA50/200 + MACD regime patterns. A
   different labeling approach (e.g. momentum or mean-reversion) may be needed for metals.

8. **H1/H4 dead for this labeler** -- all intraday timeframes produce negative or marginal
   stitched Sharpe. The regime+pullback signal requires daily-scale trends to be effective.

9. **High correlation within equity bucket** -- QQQ, SPY, ES, IWB, TQQQ all produce
   positive signals but are ~0.95 correlated. A portfolio holding all 5 has essentially
   one position, not five. Must be treated as a single allocation bucket.

---

## Next Steps

1. **Build portfolio risk layer** -- ATR-based position sizing + multi-asset allocation +
   correlation filter + drawdown circuit breaker (see "What's Missing" above)
2. **Re-evaluate with realistic sizing** -- run full eval with 1-2% risk per trade,
   max 150% gross exposure, next-bar-open fills to get realistic P&L
3. **Construct 3-bucket portfolio** -- QQQ (equity) + EUR_USD (FX) + PSKY (miner) as
   the minimum viable diversified portfolio, with correlation-aware allocation
4. **Investigate gold labeling** -- test momentum or mean-reversion labelers on GLD/GC=F
   since the regime+pullback approach produces no signal on metals
5. **Ensemble with existing strategies** -- combine ML signal with IC equity + ETF trend
   for portfolio diversification (different signal source = uncorrelated alpha)
6. **Live paper test** -- run predictions daily, compare with actual market moves for
   3 months before deploying capital
