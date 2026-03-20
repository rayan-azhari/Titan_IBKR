# IC Signal Analysis Pipeline

**Version:** 4.0 | **Last Updated:** 2026-03-20

---

## Overview

The IC Signal Analysis pipeline answers a single question before any strategy is built:

> **Does this indicator actually predict forward returns — and how consistently?**

It treats every technical indicator as a continuous output at bar `t` and measures its Spearman rank correlation with log returns at `t+h`. This is the Information Coefficient (IC) framework, standard in quantitative equity research and increasingly applied to FX and futures.

The pipeline is deliberately upstream of any strategy. It tells you what signals have alpha before you design entries, exits, or position sizing. Building a strategy around a signal with |IC| < 0.03 is wasted effort. Building one around a signal with ICIR > 0.5 gives you a principled starting point.

---

## Position in the Full Pipeline

```
Phase 0  — Regime Identification
  Regime labelling (ADX + HMM)        → per-bar state labels
  Fractional differencing (optional)   → stationarity pre-processing

Phase 1  — Signal Discovery
  phase1_sweep.py        → 52-signal IC/ICIR leaderboard (unconditional + per-regime, FLIP detection)
  phase1_param_sweep.py  → parameter grid search per signal family

Phase 2  — Signal Combination (MANDATORY)
  phase2_combination.py  → correlation matrix, orthogonality, composite building

Phase 3  — Backtesting
  phase3_backtest.py     → full-friction IS/OOS backtest: spread + slippage + swap + RoR gate

Phase 4  — Walk-Forward Optimisation
  phase4_wfo.py          → rolling or anchored walk-forward: 2yr IS / 6mo OOS, per-fold recalibration

Phase 5  — Robustness
  run_ic_robustness.py   → Monte Carlo · top-N removal · 3× slippage · WFO folds

Phase 6  — Live Deployment
  titan/strategies/       → NautilusTrader live implementation
```

For the complete specification of Phases 3–6, see `directives/IC MTF Backtesting Guide.md`.
For specific strategy results and instrument findings, see `research/ic_analysis/FINDINGS.md`.

### What Phase 1 delivers to Phase 3

- **Signal identity:** which signals to use in the composite
- **Sign orientation:** Spearman IC sign per (signal, TF) pair — used to orient all signals bullishly before averaging
- **Natural horizon:** the horizon h at which each signal peaks — informs the strategy's expected holding period
- **Regime profile:** whether the signal is unconditional, trend-only, or range-only (from Phase 0 labels)

### What Phase 1 does NOT deliver

Phase 1 does not pick a threshold, test transaction costs, or evaluate Sharpe ratios. Those are computed in Phase 3 where the full cost model (spread + slippage + swap) is applied.

---

## Directional Constraints

> [!IMPORTANT]
> **Equities / ETFs:** Long-only. Negative-IC signals are used as *dip-buy* indicators
> (buy when signal is low / oversold), never as short entries. The sign-normalisation in the
> composite builder flips these signals so that composite > 0 always means "long."
>
> **FX / Futures:** Both directions. Positive IC → long, negative IC → short. The full
> composite can produce short entries when composite < −threshold.

This constraint must be enforced in both Phase 1 (IC interpretation) and Phase 3 (trade generation). For equities, discard the short side entirely when computing backtest statistics.

---

## Core Concepts

### Information Coefficient (IC)

The IC at horizon `h` is the Spearman rank correlation between a signal vector and a forward return vector:

```
IC(h) = SpearmanCorr( signal[t], log(close[t+h] / close[t]) )
```

**Why Spearman, not Pearson?**
Return distributions are fat-tailed and non-normal. Pearson correlation is sensitive to extreme observations. Spearman converts both series to ranks first, making it robust to outliers without requiring distributional assumptions.

**Interpretation:**
- IC > 0: signal predicts positive returns (trend-following signal)
- IC < 0: signal predicts negative returns (mean-reversion signal — still useful, just flip the trade)
- |IC| < 0.03: indistinguishable from noise
- |IC| >= 0.05: usable edge (institutional quant threshold)
- |IC| > 0.10: strong edge (rare in liquid markets)

The IC is computed per bar using all available history, giving one number per (signal, horizon) pair.

**Statistical significance:** `compute_ic_table()` returns raw p-values from `scipy.stats.spearmanr` alongside IC values. Signals passing IC thresholds but failing p < 0.001 should be treated with caution, especially with small sample sizes. With 52 signals × 5 horizons = 260 simultaneous tests, apply Benjamini-Hochberg FDR correction for conservative selection.

### IC Information Ratio (ICIR)

The ICIR measures whether the IC is consistent over time, not just on average:

```
ICIR = rolling_mean(IC_series) / rolling_std(IC_series)
```

This is computed on a rolling window of 60 bars **at each signal's best horizon** (not at a fixed horizon). A signal might have a high average IC but swing wildly between positive and negative — that is not a tradeable edge. The ICIR filters for consistency.

**Interpretation:**
- ICIR > 0.5: consistent edge — the signal works reliably
- ICIR 0.3–0.5: moderate consistency — viable with confirmation
- ICIR < 0.3: inconsistent — signal may be regime-dependent

> [!WARNING]
> **Overlapping Forward Returns:** At long horizons (e.g., h=60), every forward return
> overlaps with the next 59 bars' returns. This creates serial correlation in the IC series,
> inflating the apparent ICIR. The ICIR at h=60 is not directly comparable to ICIR at h=1.
> When interpreting long-horizon results, apply Newey-West standard errors with `h-1` lags,
> or treat ICIR as an upper bound.

### Forward Returns (Volatility-Adjusted)

Forward returns at horizon `h` are computed as volatility-standardised log returns:

```python
raw_return[t] = log(close[t+h] / close[t])
vol[t]        = rolling_std(log_returns, 20) * sqrt(h)
fwd_return[t] = raw_return[t] / vol[t]
```

Volatility standardisation prevents macro-driven vol expansions (NFP, earnings) from dominating the Spearman rank. A 50-pip move during Asian session is statistically massive; a 50-pip move during NFP is noise. Standardising yields far more stable ICs.

### MFE / MAE (Path-Dependency)

Standard IC assumes the trader holds through all intermediary volatility. A signal could correctly predict a price 20 bars later, but the price might hit a 3R stop-loss on bar 2.

**Maximum Favorable Excursion (MFE):** `max(high[t+1:t+h]) - close[t]` — the best price you *could* have gotten.
**Maximum Adverse Excursion (MAE):** `min(low[t+1:t+h]) - close[t]` — the worst drawdown before reaching the target.

Both use reverse-roll-reverse for correct forward-looking windows. Ideal signals have high MFE correlation and low (negative) MAE correlation — asymmetric, safe entries.

### Alpha Decay (Signal Half-Life)

The **Half-Life** is the horizon at which IC drops to 50% of its peak. Computed by scanning IC at every bar from `h=1` through `2× max(requested horizon)`.

- Short half-life (< 5 bars): signal decays too fast for retail execution
- Long half-life (> 30 bars): signal has durable predictive power

### Signal Autocorrelation (AR1)

AR1 = `Spearman(signal[t], signal[t-1])`. Measures turnover friction.

- AR1 > 0.9: very slow-moving (low turnover, good)
- AR1 0.3–0.9: moderate turnover
- AR1 < 0.3: flips rapidly every bar — will bleed on bid/ask spread

### Recency-Weighted IC (RcntIC)

An exponentially-weighted IC with halflife=500 bars gives more weight to recent market conditions. Comparing RcntIC to full-sample IC reveals structural edge evolution:

- `RcntIC >> IC`: edge is **strengthening** recently
- `RcntIC ≈ IC`: edge is **stable**
- `RcntIC << IC`: edge is **dying** — do not deploy

> [!CAUTION]
> **IC Decay Warning:** If `RcntIC < 0.5 × FullIC`, the signal's alpha has structurally
> decayed. Treat it as no longer actionable regardless of the full-sample verdict.

### Monotonicity Score

Spearman rank correlation of quantile bin index vs. mean forward return. 

- ±1.0: perfectly monotonic edge (every quantile is ordered correctly)
- ±0.7: strong monotonic relationship
- Near 0: non-linear or chaotic — IC may be driven by tail bins only

### Horizons

Five horizons are evaluated simultaneously: **h = 1, 5, 10, 20, 60** bars.

On H4 data this corresponds to: 4h, 20h (1 trading day), 2 days, 4 days, 12 days.
On D data: 1 day, 1 week, 2 weeks, 1 month, 3 months.

The horizon at which a signal peaks tells you its natural holding period. A signal that peaks at h=5 on H4 is suggesting 20-hour trades. A signal that peaks at h=60 on D is suggesting 3-month positions.

---

## Verdict Classification

Each signal receives a verdict based on its best IC (across all horizons), corresponding ICIR, and signal autocorrelation:

| Verdict | Condition | Meaning |
|---|---|---|
| **STRONG** | \|IC\| >= 0.05 AND \|ICIR\| >= 0.5 AND AR1 > 0.3 | Consistent, meaningful predictive edge with low turnover. Pipeline candidate. |
| **USABLE** | \|IC\| >= 0.05 but fails ICIR or AR1 gate | Real signal, but inconsistent or high-turnover. Use with confirmation filter. |
| **WEAK** | 0.03 <= \|IC\| < 0.05 | Marginal signal. Only use in ensemble, not standalone. |
| **NOISE** | \|IC\| < 0.03 | Indistinguishable from random. Discard. |

The threshold of |IC| = 0.03 is standard in quantitative research — at this level the signal cannot overcome realistic transaction costs. The ICIR threshold of 0.5 ensures the signal is not purely a single-regime artifact. The AR1 gate (> 0.3) ensures the signal does not flip fast enough to bleed on execution friction.

---

## Phase 0 — Regime Identification

Before computing IC, every bar is labelled with its market regime. This enables Phase 1 to evaluate signals *per-regime*, preventing cancellation effects where trending and ranging edges cancel to zero.

### ADX Axis (3 states)

ADX measures trend *strength* (not direction). A strong downtrend and a strong uptrend both produce ADX > 25.

| Regime | Condition | Typical bar % |
|--------|-----------|---------------|
| Ranging | ADX < 20 | ~20% |
| Neutral | 20 ≤ ADX ≤ 25 | ~15% |
| Trending | ADX > 25 | ~65% |

### HMM Axis (2 states)

A 2-state Gaussian Hidden Markov Model fit on `[log_returns, realised_vol_20]` identifies latent regimes beyond what ADX can capture:

- **State 0:** Typically low-vol, mean-reverting behaviour
- **State 1:** Typically high-vol, trending or crisis behaviour

HMM is fit on the IS window only and applied forward to OOS bars. In WFO, the HMM is re-fit each fold.

### FLIP Detection

A signal is flagged as **FLIP** if its IC sign reverses across ADX regimes:
- IC_trending > +0.03 AND IC_ranging < −0.03 (or vice versa)

FLIP signals are the most important discovery in regime IC. A signal with unconditional IC ≈ 0 could have IC = +0.08 trending / −0.08 ranging. Without Phase 0, this signal is classified as NOISE. With Phase 0, it becomes STRONG in both regimes (with opposite trade directions).

### Fractional Differencing (Optional Pre-Processing)

Price series are non-stationary. Standard IC on raw prices can produce spurious correlations. Fractional differencing with d ≈ 0.3–0.4 maximally removes the unit root while preserving memory — unlike full differencing (d=1) which destroys the signal.

Script: `research/ic_analysis/run_frac_diff.py`

**When to use:** For signals derived from raw price levels (z-scores, Bollinger). Not needed for already-stationary signals (RSI, ROC, ATR).

### Phase 0 Outputs

Phase 0 produces a DataFrame with regime labels appended to each bar:
- `adx_regime`: "ranging" / "neutral" / "trending"
- `hmm_state`: 0 / 1
- `frac_diff_close`: (optional) fractionally differenced close

These columns are consumed by Phase 1 to partition the IC computation.

---

## Phase 1 — Signal Discovery

### Signal Groups (52 Signals)

#### Group A — Trend (10 signals)

Measure the direction and magnitude of price trends via moving average relationships.

| Signal | Formula | Interpretation |
|---|---|---|
| `ma_spread_5_20` | `(EMA5 - EMA20) / EMA20` | Short-term vs medium-term trend |
| `ma_spread_10_50` | `(EMA10 - EMA50) / EMA50` | Medium-term trend |
| `ma_spread_20_100` | `(EMA20 - EMA100) / EMA100` | Intermediate trend |
| `ma_spread_50_200` | `(EMA50 - EMA200) / EMA200` | Long-term trend (golden/death cross) |
| `wma_spread_5_20` | `(WMA5 - WMA20) / WMA20` | Linearly-weighted variant of short trend |
| `price_vs_sma20` | `(close - SMA20) / SMA20` | Price deviation from 20-bar mean |
| `price_vs_sma50` | `(close - SMA50) / SMA50` | Price deviation from 50-bar mean |
| `macd_norm` | `(EMA12 - EMA26) / rolling_std(close, 20)` | MACD normalised by local volatility |
| `ema_slope_10` | `(EMA10 - EMA10.shift(5)) / EMA10.shift(5)` | Rate of change of short MA |
| `ema_slope_20` | `(EMA20 - EMA20.shift(10)) / EMA20.shift(10)` | Rate of change of medium MA |

All signals are normalised (ratios or returns), never raw price levels.

#### Group B — Momentum (11 signals)

Measure the velocity of price movement without direct reference to moving average relationships.

| Signal | Formula | Interpretation |
|---|---|---|
| `rsi_7_dev` | `RSI(7) - 50` | Fast RSI centred at 0 |
| `rsi_14_dev` | `RSI(14) - 50` | Standard RSI centred at 0 |
| `rsi_21_dev` | `RSI(21) - 50` | Slow RSI centred at 0 |
| `stoch_k_dev` | `StochK(14,3) - 50` | Stochastic %K centred at 0 |
| `stoch_d_dev` | `StochD(14,3) - 50` | Stochastic %D centred at 0 |
| `cci_20` | `(close - SMA20) / (0.015 * MAD(close, 20))` | Commodity Channel Index |
| `williams_r_dev` | `Williams%R(14) + 50` | Williams %R centred at 0 |
| `roc_3` | `log(close / close.shift(3))` | 3-bar log return |
| `roc_10` | `log(close / close.shift(10))` | 10-bar log return |
| `roc_20` | `log(close / close.shift(20))` | 20-bar log return |
| `roc_60` | `log(close / close.shift(60))` | 60-bar log return |

RSI and Stochastic signals are centred at 0 (by subtracting 50) so they behave as symmetric signals.

#### Group C — Mean Reversion (6 signals)

Measure how far price has deviated from a mean — negative IC expected if the instrument mean-reverts.

| Signal | Formula | Interpretation |
|---|---|---|
| `bb_zscore_20` | `(close - SMA20) / (2 * std(close, 20))` | Bollinger Band position |
| `bb_zscore_50` | `(close - SMA50) / (2 * std(close, 50))` | Wide Bollinger position |
| `zscore_20` | `(close - SMA20) / std(close, 20)` | 20-bar z-score |
| `zscore_50` | `(close - SMA50) / std(close, 50)` | 50-bar z-score |
| `zscore_100` | `(close - SMA100) / std(close, 100)` | 100-bar z-score |
| `zscore_expanding` | `(close - expanding_mean) / expanding_std` | Deviation from full-history mean |

A negative IC for these signals indicates the instrument mean-reverts (stretched prices retrace). A positive IC indicates trends persist. The sign of IC for Group C signals is diagnostic of the instrument's statistical regime.

#### Group D — Volatility State (7 signals)

| Signal | Formula | Interpretation |
|---|---|---|
| `norm_atr_14` | `ATR(14) / close` | Normalised true range — regime identifier |
| `realized_vol_5` | `std(log_returns, 5) * sqrt(252)` | 5-bar realised vol, annualised |
| `realized_vol_20` | `std(log_returns, 20) * sqrt(252)` | 20-bar realised vol, annualised |
| `garman_klass` | `sqrt(0.5*(log(H/L))^2 - (2*ln2-1)*(log(C/O))^2)`, rolling 20 | OHLC-efficient vol estimator |
| `parkinson_vol` | `sqrt(1/(4*ln2) * mean((log(H/L))^2, 20))` | High-low range vol estimator |
| `bb_width` | `(4 * std(close, 20)) / SMA20` | Bollinger Band width |
| `adx_14` | `ADX(14)` | Average Directional Index (trend strength) |

#### Group E — Acceleration / Deceleration (7 signals)

First differences of base signals — measure the *rate of change* of the indicator, not its level.

| Signal | Formula | Base signal |
|---|---|---|
| `accel_roc10` | `roc_10.diff(1)` | Rate of change of 10-bar momentum |
| `accel_rsi14` | `rsi_14_dev.diff(1)` | RSI velocity |
| `accel_macd` | `macd_hist(close,12,26,9).diff(1)` | MACD histogram velocity |
| `accel_atr` | `norm_atr_14.diff(1)` | Volatility expansion/contraction |
| `accel_bb_width` | `bb_width.diff(1)` | Bandwidth expansion/contraction |
| `accel_rvol20` | `realized_vol_20.diff(1)` | Realised vol change |
| `accel_stoch_k` | `stoch_k_dev.diff(1)` | Stochastic velocity |

#### Group F — Structural / Breakout (6 signals)

| Signal | Formula | Interpretation |
|---|---|---|
| `donchian_pos_10` | `(close - min(low,10)) / (max(high,10) - min(low,10)) - 0.5` | 10-bar range position |
| `donchian_pos_20` | same, window=20 | 20-bar range position |
| `donchian_pos_55` | same, window=55 | 55-bar range position (Turtle system) |
| `keltner_pos` | `(close - EMA20) / (2 * ATR(10))` | Keltner Channel position |
| `price_pct_rank_20` | `close.rolling(20).rank(pct=True) - 0.5` | Percentile rank in 20-bar window |
| `price_pct_rank_60` | `close.rolling(60).rank(pct=True) - 0.5` | Percentile rank in 60-bar window |

#### Group G — Semantic Combinations (5 signals)

| Signal | Formula | Rationale |
|---|---|---|
| `trend_mom` | `sign(ma_spread_5_20) * abs(rsi_14_dev) / 50` | Trend direction × momentum magnitude |
| `trend_vol_adj` | `ma_spread_5_20 / (norm_atr_14 + 1e-9)` | Trend signal normalised by current volatility |
| `mom_accel_combo` | `rsi_14_dev * sign(accel_rsi14)` | Momentum × whether it is accelerating |
| `donchian_rsi` | `donchian_pos_20 * (rsi_14_dev / 50)` | Structural position gated by momentum |
| `vol_regime_trend` | `ma_spread_5_20 * (1 - norm_atr_14 / norm_atr_14.rolling(60).mean())` | Trend attenuated in high-vol regimes |

### Phase 1 Execution

Phase 1 runs IC computation **three times** using Phase 0 regime labels:

1. **Unconditional:** all bars (baseline)
2. **Trending-only:** bars where `adx_regime == "trending"`
3. **Ranging-only:** bars where `adx_regime == "ranging"`

The leaderboard shows all three rows for each signal. Signals marked NOISE unconditionally but STRONG in one regime are flagged as **FLIP** — these are the highest-value discoveries.

### Phase 1 Output Columns

| Column | Description |
|---|---|
| `signal` | Signal name |
| `group` | Group label (Trend, Momen, MnRev, Vol, Accel, Struct, Combo) |
| `best_h` | Horizon with highest \|IC\| |
| `best_ic` | IC value at best horizon |
| `icir` | ICIR at best horizon (rolling 60-bar) |
| `p_val` | Spearman p-value at best horizon |
| `ar1` | Signal Autocorrelation |
| `mfe` | IC correlation with MFE at best horizon |
| `mae` | IC correlation with MAE at best horizon |
| `half_life` | Alpha decay half-life (bars) |
| `mono` | Monotonicity score (±1.0 = perfect) |
| `rcnt_ic` | Recency-weighted IC (halflife=500 bars) |
| `verdict` | STRONG / USABLE / WEAK / NOISE |

---

## Phase 2 — Signal Combination (Mandatory)

> [!IMPORTANT]
> Phase 2 is **mandatory** before proceeding to backtesting. Evaluating signals independently
> (Marginal IC) creates an illusion of diversification. Two STRONG signals at 95% correlation
> represent the *same* edge — combining them adds noise, not alpha.

### Step 1: Correlation Matrix

Compute the pairwise Spearman correlation of all STRONG and USABLE signals:

```
corr_matrix = signals[strong_usable_list].corr(method='spearman')
```

Any pair with `|corr| > 0.7` must be resolved: keep the signal with higher ICIR, drop the other.

### Step 2: Partial IC (Orthogonality Test)

Before adding a signal to the composite, regress it against the existing dominant signals and compute IC on the *residuals*:

```
residual = signal - β₁ × dominant_signal_1 - β₂ × dominant_signal_2
Partial_IC = SpearmanCorr(residual, fwd_return)
```

A signal with high Marginal IC but near-zero Partial IC adds no incremental alpha. Only signals with **Partial IC > 0.02** should enter the composite.

### Step 3: Composite Building

Six combination methods are tested:
1. **Equal-weight** — simple average of sign-normalised signals
2. **ICIR-weighted** — weight by each signal's ICIR
3. **PCA** — first principal component of signal matrix
4. **Regime-gated** — different weights per ADX regime
5. **Stacked** — equal-weight across multiple timeframes
6. **Custom** — manual weights based on domain knowledge

The composite that maximises IS IC without diverging from equal-weight by more than 0.003 on OOS is selected (Occam's razor: prefer simplicity unless complexity measurably helps).

### Cross-Sectional IC (Equity Universe)

For equity portfolios with multiple instruments, compute **cross-sectional IC** at each bar: rank all symbols by signal value, then correlate with their forward returns. This is how institutional equity factors are evaluated and avoids per-instrument overfitting.

```
CS_IC[t] = SpearmanCorr( signal_rank[all_symbols, t], fwd_return[all_symbols, t] )
```

---

## Look-Ahead Safety

All signal computations are strictly causal. No signal uses future information.

| Operation type | Example | Safe? |
|---|---|---|
| Rolling window | `.rolling(n).mean()` | Yes — uses only bars t through t-n+1 |
| Exponential smoothing | `.ewm(span=n).mean()` | Yes — decays geometrically into past |
| Positive shift | `.shift(+n)` | Yes — shifts data backward, uses older values |
| Diff | `.diff(1)` | Yes — bar t minus bar t-1 |
| Forward return (TARGET ONLY) | `close.shift(-h)` | Intentional — this is the label, never a feature |
| Negative shift in feature | `.shift(-n)` | **FORBIDDEN** — look-ahead bias |

> [!WARNING]
> `close.shift(-h)` appears only inside `compute_forward_returns()` in `run_ic.py`.
> Every signal factory function must contain zero instances of `shift(-n)`.
> Verify with: `grep -n "shift(-" research/ic_analysis/phase1_sweep.py`

---

## Pipeline Architecture

### Code Files

| File | Role |
|---|---|
| `research/ic_analysis/run_ic.py` | Core IC functions — `compute_ic_table`, `compute_forward_returns`, `compute_icir`, `compute_mfe_mae_targets`, `compute_alpha_decay`, `compute_monotonicity`, `compute_recency_weighted_ic` |
| `research/ic_analysis/phase0_regime.py` | Phase 0 — ADX + HMM regime labelling; optional fractional differencing |
| `research/ic_analysis/phase1_sweep.py` | Phase 1 — 52-signal IC/ICIR sweep with regime-conditional IC and FLIP detection |
| `research/ic_analysis/phase1_param_sweep.py` | Phase 1 extension — parameter grid search per signal family |
| `research/ic_analysis/phase2_combination.py` | Phase 2 — correlation matrix, partial IC, composite building |
| `research/ic_analysis/pipeline_discovery.py` | Orchestrator: Phases 0 → 1 → 2 for any instrument(s) / TF |
| `titan/strategies/ml/features.py` | Technical indicator implementations (used by signal registry in ic_generic) |

### Data Requirements

Data must be available as Parquet files:

```
data/{INSTRUMENT}_{TIMEFRAME}.parquet          ← Phase 0–2
.tmp/data/raw/{INSTRUMENT}_{TIMEFRAME}.parquet ← Phase 3+
```

Required columns (case-insensitive, standardised to lowercase at load): `open`, `high`, `low`, `close`. `volume` is optional.

Minimum bars: 200 (to allow 100-bar rolling windows with a buffer).

### Execution Flow

```
Phase 0:
  _load_ohlcv()  →  compute ADX(14) + HMM(2-state)  →  regime labels
    |
    v
Phase 1:
  build_all_signals()       → 52 signals (Groups A–G)
    |
    v
  compute_forward_returns()       [vol-adjusted log returns per horizon]
  compute_mfe_mae_targets()       [forward-looking path excursion]
    |
    v
  compute_ic_table()              [Spearman rank corr + p-values × 3 regime splits]
  compute_recency_weighted_ic()   [exponentially-weighted IC, halflife=500 bars]
    |
    v
  compute_icir()                  [rolling mean/std, at best horizon per signal]
  compute_alpha_decay()           [dense h=1..2×max(h) scan]
  compute_autocorrelation()       [AR1 per signal]
  compute_monotonicity()          [quantile spread linearity]
    |
    v
  _print_leaderboard()            [ranked by |IC|, FLIP detection]
    |
    v
Phase 2:
  corr_matrix()                   [pairwise signal correlation]
  partial_ic()                    [orthogonality test]
  composite_builder()             [6 combination methods]
    |
    v
  CSV export to .tmp/reports/
```

---

## Running the Pipeline

### Full Discovery Pipeline (Phases 0 → 1 → 2)

```bash
# Single instrument
uv run python research/ic_analysis/pipeline_discovery.py \
  --instrument EUR_USD --timeframe H4 --tfs W,D,H4,H1

# Batch — equities
uv run python research/ic_analysis/pipeline_discovery.py \
  --instruments SPY QQQ CSCO NOC --timeframe D

# Phase 1 only (skip Phase 2)
uv run python research/ic_analysis/pipeline_discovery.py \
  --instrument EUR_USD --timeframe H4 --phase 0,1
```

### Phase 0 (Regime Labels Only)

```bash
uv run python research/ic_analysis/phase0_regime.py --instrument EUR_USD --timeframe H4
uv run python research/ic_analysis/phase0_regime.py --instrument SPY --timeframe D --frac-diff
```

### Phase 1 (Signal Sweep Only)

```bash
uv run python research/ic_analysis/phase1_sweep.py --instrument EUR_USD --timeframe H4
uv run python research/ic_analysis/phase1_sweep.py --instrument SPY --timeframe D --horizons 1,5,10,20,60
```

### Phase 2 (Signal Combination Only)

```bash
uv run python research/ic_analysis/phase2_combination.py --instrument SPY --timeframe D
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--instrument` | `EUR_USD` | Instrument slug (must match Parquet filename) |
| `--timeframe` | `H4` | Timeframe (H1, H4, D, W) |
| `--horizons` | `1,5,10,20,60` | Comma-separated forward return horizons in bars |
| `--n_bins` | `10` | Number of quantile bins for decile plots |

---

## Output Format

### Console Leaderboard

```
==================================================================================
  IC SIGNAL SWEEP -- EUR_USD H4
  Signals: 52  |  Horizons: [1, 5, 10, 20, 60]  |  Bars: 1,133
==================================================================================

  LEADERBOARD (ranked by |IC| at best horizon)
  --------------------------------------------------------------------------------
  Rank  Signal                    Grp   BestH        IC     ICIR     AR1  Verdict
  --------------------------------------------------------------------------------
```

### Signal Summary (run_ic.py)

```
-- Signal Summary -------------------------------------------------------------------------------------
  Signal           BestH    BestIC     ICIR    p-val     MFE     MAE  HalfL   Mono   RcntIC  Verdict
  -------------------------------------------------------------------------------------------------------
```

### CSV Outputs

Two CSV files per run, saved to `.tmp/reports/`:
- `ic_sweep_{instrument}_{timeframe}.csv` — Full IC table with all metrics
- `icir_sweep_{instrument}_{timeframe}.csv` — Rolling ICIR time series

---

## Interpreting Results

### What to do with STRONG signals

Promote to Phase 2 (combination). A STRONG signal is a justified basis for an entry rule, a filter, or an ML feature. Document the horizon at which it peaks — this is the strategy's natural holding period.

### What to do with USABLE signals

Use as confirmation or in ensembles. A single USABLE signal should not be traded standalone, but combining two independent USABLE signals is often equivalent to a STRONG signal.

### What to do with WEAK signals

Keep for reference. Run on additional timeframes and instruments. If consistently WEAK across all data, discard.

### What to do with NOISE signals

Discard immediately. Their presence degrades composite performance through noise amplification.

### FLIP signals

The most valuable discovery. A FLIP signal appears as NOISE unconditionally but is STRONG in one or both regimes. These signals require Phase 0 regime gating to unlock their edge — without the gate, the two regime effects cancel each other out.

### IC Decay Warning

If `RcntIC < 0.5 × FullIC`, the signal's alpha has structurally decayed. This means the edge that historically existed has been crowded out or the market microstructure has changed. Do not deploy dying signals even if their full-sample verdict is STRONG.

---

## Risk of Ruin (Phase 3 Gate)

Risk of Ruin (RoR) is computed in Phase 3 once actual trade results are available. It answers: **what is the probability that this strategy will hit a catastrophic drawdown before realising its edge?**

### Balsara Formula

```
edge     = (win_rate × avg_win) − (loss_rate × avg_loss)
cap_units = 1 / risk_per_trade    # e.g., 1% risk → 100 units
P(ruin)  = ((1 − edge) / (1 + edge)) ^ cap_units
```

### Three Inputs

| Input | Source | Typical Range |
|---|---|---|
| **Win Rate** | OOS trade results from Phase 3 | 55–75% for IC-derived strategies |
| **Payoff Ratio** | Avg win / Avg loss from Phase 3 | 0.8–2.0× for mean-reversion |
| **Position Sizing** | Risk per trade (% of capital) | 1–2% recommended |

### Gate Criteria

| Level | Target | Interpretation |
|---|---|---|
| **Retail** | P(ruin at 25% DD) < 5% | Survives normal losing streaks |
| **Professional** | P(ruin at 25% DD) < 1% | High confidence in survival |
| **Institutional** | P(ruin at 50% DD) < 0.1% | Extreme stress tolerance |

### Why It Matters

**Survival over performance.** A trader who cannot survive a losing streak never realises their long-term edge. Geometric attrition means losses compound: a 50% drawdown requires a 100% gain to recover. The Balsara formula quantifies this risk *before* live deployment.

> [!CAUTION]
> A strategy with high Sharpe but P(ruin) > 5% is a ticking time bomb. Phase 3 must
> compute and display the RoR alongside Sharpe, annual return, and max drawdown.

---

## From IC to Strategy — What Comes After Phase 2

### Cost Model Applied in Phase 3

| Cost component | Default | Applied at |
|----------------|---------|-----------| 
| Spread (half bid-ask) | 0.5–1.0 pip per pair | Each fill (entry + exit) |
| Slippage | 0.5 pip per fill | Each fill |
| Swap (overnight carry) | pair-specific pips/night | Each 21:00 UTC bar while in position |

An IC sweep finding of STRONG can still fail Phase 3 if the signal fires too frequently. The holding period implied by the peak horizon determines whether the IC is large enough to survive friction:

```
Minimum IC to cover round-trip costs ≈ 2 × total_friction / avg_holding_bars
```

### Monte Carlo: Why It Matters

The IC measures correlation over the full sample. A signal with IC=0.06 could achieve that through 6 months of strong correlation and 6 months of near-zero. The Monte Carlo shuffle in Phase 5 tests whether the *same trades* in random order would produce similar Sharpe — confirming the edge is in individual trade returns, not lucky sequencing.

### Walk-Forward: IC Sign Recalibration

In live deployment, the IC sign is computed from a warmup window. WFO validates that re-calibrating the sign and re-selecting the threshold every 2 years produces consistently positive OOS results.

---

## Statistical Caveats

> [!WARNING]
> **Phase 1 Selection Bias:** IC is computed on the full dataset with no IS/OOS split.
> Signals advanced to Phase 3 have already been selected based on full-sample IC, which
> includes the periods that will later be used for OOS validation. This creates a subtle
> selection bias. Phase 4 (WFO) partially mitigates this, but consider running Phase 1
> on only the first 70% of data and validating the shortlist on the last 30%.

> [!WARNING]
> **Multiple Testing:** With 52 signals × 5 horizons × 3 regime splits = 780 simultaneous
> tests, approximately 39 false positives are expected at the 5% level. The p-values in the
> summary table are raw (unadjusted). For conservative selection, apply Benjamini-Hochberg
> FDR correction or require p < 0.001.

---

## Quality Gates

Before trusting any sweep output:

- [ ] Confirm zero `shift(-n)` calls in any signal factory: `grep -n "shift(-" research/ic_analysis/phase1_sweep.py`
- [ ] All 52 signals present in the leaderboard (no KeyError / group mismatch)
- [ ] CSV row count = 52
- [ ] At least 200 bars available after NaN drop
- [ ] ICIR_WINDOW (60) < available bars / 3 (otherwise ICIR is unreliable)
- [ ] Phase 2 correlation matrix computed before proceeding to Phase 3
- [ ] Ruff passes: `uv run ruff check research/ic_analysis/phase1_sweep.py`

---

## Parameter Sweep (Phase 1 Extension)

**Script:** `research/ic_analysis/phase1_param_sweep.py`

The parameter sweep tests whether the default indicator parameters are near-optimal or whether tuned values meaningfully improve IC.

### Overfitting Warning

> Parameter sweeping against IC is in-sample optimisation. Before acting on tuned parameters:
> 1. Validate on a held-out OOS period (last 20% of bars)
> 2. Check that the best and second-best parameters have similar IC (smooth optimum = robust)
> 3. Only upgrade defaults if OOS IC improvement > 0.005

### Parameter Grids

| Family | Parameters swept |
|--------|----------------|
| RSI | period ∈ [5, 7, 9, 12, 14, 21, 28] |
| MA Spread | fast ∈ [3,5,8,10], slow ∈ [15,20,30,50] (fast < slow) |
| ROC | period ∈ [2, 3, 5, 10, 15, 20, 30, 60] |
| Bollinger/Z-score | window ∈ [10, 15, 20, 30, 50, 100] |
| ATR | period ∈ [7, 10, 14, 21] |
| Stochastic | k ∈ [5,9,14,21], d ∈ [2,3,5] |
| Donchian | window ∈ [5, 10, 20, 40, 55, 100] |
| ADX | period ∈ [7, 10, 14, 21, 28] |
| MACD | fast ∈ [8,10,12], slow ∈ [20,26,30], sig ∈ [7,9,12] |
| Realized Vol | window ∈ [5, 10, 20, 40, 60] |

```bash
uv run python research/ic_analysis/run_param_sweep.py --instrument EUR_USD --timeframe H4
```

---

## Pending Enhancements

### Newey-West Standard Errors for Long-Horizon ICIR
At h=60, the ICIR is inflated by overlapping returns. Implementing Newey-West corrected standard errors with `h-1` lags would provide unbiased significance estimates.

### Benjamini-Hochberg FDR Automation
Currently p-values are reported raw. Automating the BH correction across the 780-test matrix (52 × 5 × 3 regimes) and reporting adjusted p-values alongside raw would prevent false discovery.

---

## Extending the Pipeline

### Adding a new signal

1. Choose the appropriate group function (`_compute_group_a` through `_compute_group_g`)
2. Compute the signal as a `pd.Series` aligned to `df.index`
3. Call `out[_tag("signal_name", "GroupLabel")] = series` to register and store it
4. If it is a Group G combination, ensure all component signals are in `sigs` before accessing them
5. Update the signal count in the header comment and in this document

### Adding a new group

1. Create `_compute_group_h(...)` following the existing pattern
2. Call it in `build_all_signals()` after all signals it depends on
3. Merge its output into the running DataFrame before passing to dependent groups
4. Register all signals with `_tag()` using a new group label

### Adding a new instrument or timeframe

No code changes required. The data loader reads any file matching `data/{INSTRUMENT}_{TIMEFRAME}.parquet`. Ensure the file has lowercase OHLC column names.

---

## Version History

| Version | Date | Changes |
|---|---|---|
| **4.0** | 2026-03-20 | Major restructuring: Added Phase 0 (regime pre-identification + frac diff), mandatory Phase 2 (Partial IC + correlation matrix), directional constraints (equities long-only / FX both ways), Risk of Ruin (Balsara), cross-sectional IC, IC decay warnings, fractional differencing. Extracted all strategy findings to `FINDINGS.md`. Removed implemented roadmap items. |
| **3.4** | 2026-03-20 | Re-run with HMM gate added. 6 validated symbols. |
| **3.2** | 2026-03-19 | Fixed MFE/MAE rolling direction. ICIR at best horizon. p-values, monotonicity, recency-weighted IC. AR1 verdict gate. |
| **3.0** | 2026-03-19 | AR1 autocorrelation, vol-standardized targets. Institutional-grade roadmap. |
| **2.0** | 2026-03-17 | 52-signal sweep with 7 groups. Regime-conditional IC. |
| **1.0** | 2026-03-15 | Initial IC scanner with 6 base signals and ICIR. |
