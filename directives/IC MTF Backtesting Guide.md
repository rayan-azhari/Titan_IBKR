# IC MTF Backtesting Guide

**Version:** 2.1 | **Last Updated:** 2026-03-19

---

## Overview

The IC MTF (Information Coefficient Multi-Timeframe) strategy is a systematic forex strategy
that trades six currency pairs by combining two momentum signals across four timeframes. It was
validated through a six-phase research pipeline before live implementation in NautilusTrader.

**Strategy:**
- **Pairs:** EUR/USD, GBP/USD, USD/JPY, AUD/USD, AUD/JPY, USD/CHF
- **Timeframes:** W, D, H4, H1
- **Signals:** `accel_stoch_k` + `accel_rsi14` (equal-weight composite, z-score normalised)
- **Entry:** composite_z > threshold → long; composite_z < -threshold → short
- **Exit:** composite_z crosses zero
- **Sizing:** 1% risk / (1.5 × ATR14_H1), capped at 20× leverage
- **OOS Sharpe range (fresh run 2026-03-19):** 6.84–8.48 across all pairs

---

## Pipeline at a Glance

```
Phase 1: IC Signal Sweep        → screen 52 signals for genuine forward-return predictability
Phase 2: Signal Combination     → confirm equal-weight composite beats individual signals
Phase 3: IS/OOS Backtest        → full-friction backtest with 70/30 split, threshold search
Phase 4: Walk-Forward (WFO)     → rolling 2yr IS / 6mo OOS to validate temporal stability
Phase 5: Robustness Validation  → Monte Carlo · top-N removal · 3× slippage · WFO folds
Phase 6: Live Implementation    → NautilusTrader strategy in titan/strategies/ic_mtf/
```

---

## Data

### Source

All data is downloaded via Databento (Historical API) at H1 granularity. Higher timeframes
(H4, D, W) are resampled within each research script using pandas `resample()`.

```bash
uv run python scripts/download_data_databento.py --instrument EUR_USD
```

Files land in `.tmp/data/raw/` as Parquet:

```
.tmp/data/raw/EUR_USD_H1.parquet
.tmp/data/raw/EUR_USD_H4.parquet
.tmp/data/raw/EUR_USD_D.parquet
.tmp/data/raw/EUR_USD_W.parquet
```

### Available History (as of 2026-03-19)

| Pair    | From       | To         | H1 Bars |
|---------|-----------|-----------|---------|
| EUR/USD | 2005-01-02 | 2026-03-16 | 134,971 |
| GBP/USD | 2016-03-17 | 2026-03-13 |  62,053 |
| USD/JPY | 2016-03-17 | 2026-03-13 |  62,074 |
| AUD/USD | 2016-03-17 | 2026-03-13 |  62,073 |
| AUD/JPY | 2016-03-17 | 2026-03-13 |  62,077 |
| USD/CHF | 2016-03-17 | 2026-03-13 |  62,078 |

---

## Phase 1 — IC Signal Sweep

**Script:** `research/ic_analysis/run_signal_sweep.py`

**Purpose:** Screen every candidate indicator for genuine forward-return predictability using
the Information Coefficient (IC) framework before building any strategy. A signal with
|IC| < 0.03 is statistically indistinguishable from noise and should not be traded.

### What is the Information Coefficient?

```
IC(h) = SpearmanRank( signal[t],  log(close[t+h] / close[t]) )
```

Spearman rank correlation is used instead of Pearson because FX return distributions are
fat-tailed. Spearman converts both series to ranks first, making it robust to outliers without
any distributional assumption.

| IC range        | Verdict | Meaning |
|-----------------|---------|---------|
| \|IC\| >= 0.05, ICIR >= 0.5 | STRONG | Build strategy around this |
| \|IC\| >= 0.05, ICIR < 0.5  | USABLE | IC present but inconsistent |
| 0.03 <= \|IC\| < 0.05       | WEAK   | Try regime conditioning |
| \|IC\| < 0.03               | NOISE  | Discard |

**ICIR** (IC Information Ratio) measures consistency over time:

```
ICIR = rolling_mean(IC_series, window=60) / rolling_std(IC_series, window=60)
```

A signal with high average IC but high variance in IC is not reliably tradeable. ICIR filters
for signals whose edge is stable across different market regimes.

### The 52 Signals — Full Catalogue

All 52 signals are grouped into 7 families (A–G). Every computation uses only past data
(no `.shift(-h)` in signal factories — only in `compute_forward_returns`).

**Group A — Trend (10 signals)**

These measure the direction and strength of the current price trend via moving average spreads
and slope derivatives.

| Signal | Formula |
|--------|---------|
| `ma_spread_5_20` | `(EMA5 - EMA20) / EMA20` |
| `ma_spread_10_50` | `(EMA10 - EMA50) / EMA50` |
| `ma_spread_20_100` | `(EMA20 - EMA100) / EMA100` |
| `ma_spread_50_200` | `(EMA50 - EMA200) / EMA200` |
| `wma_spread_5_20` | `(WMA5 - WMA20) / WMA20` |
| `price_vs_sma20` | `(close - SMA20) / SMA20` |
| `price_vs_sma50` | `(close - SMA50) / SMA50` |
| `macd_norm` | `(EMA12 - EMA26) / rolling_std(20)` |
| `ema_slope_10` | `(EMA10 - EMA10.shift(5)) / EMA10.shift(5)` |
| `ema_slope_20` | `(EMA20 - EMA20.shift(10)) / EMA20.shift(10)` |

**Group B — Momentum (11 signals)**

Oscillators measuring buying/selling pressure. Deviated from their neutral value (50 for RSI,
stochastic; 0 for ROC/Williams) so positive = bullish bias, negative = bearish.

| Signal | Formula |
|--------|---------|
| `rsi_7_dev` | `RSI(7) - 50` |
| `rsi_14_dev` | `RSI(14) - 50` |
| `rsi_21_dev` | `RSI(21) - 50` |
| `stoch_k_dev` | `Stochastic_%K(14, 3) - 50` |
| `stoch_d_dev` | `Stochastic_%D(14, 3) - 50` |
| `cci_20` | `(close - SMA20) / (0.015 × MAD20)` |
| `williams_r_dev` | `Williams_%R(14) + 50` |
| `roc_3` | `log(close / close.shift(3))` |
| `roc_10` | `log(close / close.shift(10))` |
| `roc_20` | `log(close / close.shift(20))` |
| `roc_60` | `log(close / close.shift(60))` |

**Group C — Mean Reversion (6 signals)**

Z-score distance of price from its rolling mean. Negative IC expected (price reverts to mean) —
sign-normalised to +1 before combining with trend signals.

| Signal | Formula |
|--------|---------|
| `bb_zscore_20` | `(close - SMA20) / (2 × std20)` |
| `bb_zscore_50` | `(close - SMA50) / (2 × std50)` |
| `zscore_20` | `(close - SMA20) / std20` |
| `zscore_50` | `(close - SMA50) / std50` |
| `zscore_100` | `(close - SMA100) / std100` |
| `zscore_expanding` | `(close - expanding_mean) / expanding_std` |

**Group D — Volatility State (7 signals)**

Measure current volatility regime. Useful for sizing and regime-conditioning but typically
have low IC as directional signals by themselves.

| Signal | Formula |
|--------|---------|
| `norm_atr_14` | `ATR(14) / close` |
| `realized_vol_5` | `rolling_std(log_returns, 5) × √252` |
| `realized_vol_20` | `rolling_std(log_returns, 20) × √252` |
| `garman_klass` | OHLC-based vol estimator (rolling 20) |
| `parkinson_vol` | High-low range vol estimator (rolling 20) |
| `bb_width` | `(4 × std20) / SMA20` (Bollinger Band width) |
| `adx_14` | Average Directional Index, period=14 |

**Group E — Acceleration / Deceleration (7 signals)**

First difference of Group B/D signals. Captures the *rate of change* of momentum, not its
level. These turned out to be the strongest predictors (accel_rsi14 and accel_stoch_k
selected for the strategy).

| Signal | Formula |
|--------|---------|
| `accel_roc10` | `roc_10.diff(1)` |
| `accel_rsi14` | `rsi_14_dev.diff(1)` = diff(RSI(14) - 50) |
| `accel_macd` | `MACD_hist(12,26,9).diff(1)` |
| `accel_atr` | `norm_atr_14.diff(1)` |
| `accel_bb_width` | `bb_width.diff(1)` |
| `accel_rvol20` | `realized_vol_20.diff(1)` |
| `accel_stoch_k` | `stoch_k_dev.diff(1)` = diff(Stoch_%K - 50) |

**Group F — Structural / Breakout (6 signals)**

Position of price within its recent high/low range. Positive = near the top of range (bullish
breakout potential).

| Signal | Formula |
|--------|---------|
| `donchian_pos_10` | `(close - low_10) / (high_10 - low_10) - 0.5` |
| `donchian_pos_20` | `(close - low_20) / (high_20 - low_20) - 0.5` |
| `donchian_pos_55` | `(close - low_55) / (high_55 - low_55) - 0.5` |
| `keltner_pos` | `(close - EMA20) / (2 × ATR10)` |
| `price_pct_rank_20` | `rolling_percentile_rank(close, 20) - 0.5` |
| `price_pct_rank_60` | `rolling_percentile_rank(close, 60) - 0.5` |

**Group G — Semantic Combinations (5 signals)**

Hand-crafted compound signals that blend information from multiple groups.

| Signal | Formula |
|--------|---------|
| `trend_mom` | `sign(ma_spread_5_20) × |rsi_14_dev| / 50` |
| `trend_vol_adj` | `ma_spread_5_20 / (norm_atr_14 + ε)` |
| `mom_accel_combo` | `rsi_14_dev × sign(accel_rsi14)` |
| `donchian_rsi` | `donchian_pos_20 × (rsi_14_dev / 50)` |
| `vol_regime_trend` | `ma_spread_5_20 × (1 - norm_atr_14 / ma(norm_atr_14, 60))` |

### Run

```bash
uv run python research/ic_analysis/run_signal_sweep.py \
    --instrument EUR_USD \
    --timeframe H4
```

CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--instrument` | `EUR_USD` | Instrument slug |
| `--timeframe` | `H4` | Timeframe for native signal computation |
| `--horizons` | `1,5,10,20,60` | Forward-return horizons in bars |
| `--icir_window` | `60` | Rolling window for ICIR computation |

### Output

```
.tmp/reports/ic_sweep_{slug}.csv
```

Columns: `signal, group, horizon, ic, icir, verdict, quantile_spread`

### Selected Signals for IC MTF

After sweeping EUR/USD H4 at all horizons, and validating findings on other pairs and
timeframes, two signals were selected:

| Signal | IC at h=20 | ICIR at h=20 | Character |
|--------|-----------|-------------|-----------|
| `accel_stoch_k` | +0.058 | +0.74 | Stochastic %K acceleration — speed of momentum change |
| `accel_rsi14` | +0.061 | +0.71 | RSI(14) acceleration — second derivative of buying pressure |

**Why acceleration signals?** The sweep consistently showed that the *rate of change* of
momentum (Group E) outperformed the momentum level itself (Group B). This makes intuitive
sense in trending FX markets: it is not enough that RSI is above 50; what matters is that
momentum is still accelerating upward. When RSI begins decelerating (diff turns negative)
while still above 50, the trend is exhausting — and the signal captures this early.

**Why these two specifically?**
- Low mutual correlation (0.31) — they carry complementary information
- Edge is consistent across all 6 pairs and all 4 timeframes (not EUR/USD-specific)
- Both have ICIR > 0.5, meaning the edge is stable, not just a lucky average

---

## Phase 2 — Signal Combination

**Script:** `research/ic_analysis/run_signal_combination.py`

**Purpose:** Test 6 combination methods on the candidate signals to determine whether
compounding signals improves IC beyond the best individual. For IC MTF, this phase confirmed
that equal-weight averaging of the two selected signals was optimal.

### Methods tested

| Method | Description |
|--------|-------------|
| Correlation clustering | Identify redundant pairs (Pearson ≥ 0.7) — diagnostic only |
| Equal-weight top-N | Mean of top-N signals by \|IC\|, sign-normalised (N=3, 5, 10) |
| ICIR-weighted top-N | Weighted mean using \|ICIR_i\| / Σ\|ICIR\| (N=3, 5) |
| Group-diversified | Best signal from each group A–G averaged |
| PCA (PC1) | First principal component of usable signals after StandardScaler |
| AND-gating | Signal A active only when signal B confirms direction |

**Sign normalisation:** Before averaging, each signal is multiplied by `sign(IC)` at the
target horizon. This ensures mean-reversion signals (negative IC) contribute bullishly when
they reach extreme lows, rather than pulling the composite in the wrong direction.

### Result for IC MTF

Equal-weight of `accel_stoch_k` + `accel_rsi14` produced:
- ΔIC vs. best individual: +0.003 — negligible improvement from adding more signals
- PCA and ICIR-weighting: ΔIC < 0.003 vs. equal-weight
- Decision: keep equal-weight, 2-signal composite (minimal complexity, same performance)

```bash
uv run python research/ic_analysis/run_signal_combination.py \
    --instrument EUR_USD --timeframe H4 --horizon 20
```

Output: `.tmp/reports/ic_combo_{slug}.csv`

---

## Phase 3 — IS/OOS Backtest

**Script:** `research/ic_analysis/run_ic_backtest.py`

**Purpose:** Full-friction backtest of the IC MTF composite strategy. Uses a single 70/30
time-ordered IS/OOS split. The optimal entry threshold is selected on IS only; OOS evaluation
is strictly blind.

### IS/OOS Split

```
IS  = bars 0 → int(0.70 × n)    ← threshold selection only
OOS = bars int(0.70 × n) → n    ← all reported performance metrics
```

No shuffling. OOS is always the chronologically most recent 30% of data.

### Composite Signal Construction

```
1. Load H1, H4, D, W bars for the instrument.
2. For each timeframe, compute accel_stoch_k and accel_rsi14.
3. On the IS period, compute IC sign for each (signal, TF) pair:
       sign = Spearman( signal_IS[t], log_return_IS[t+1] ) → +1 or -1
4. Multiply each signal by its sign → all oriented bullishly.
5. Forward-fill each TF signal onto the H1 timeline (no look-ahead):
       higher_tf.reindex(h1_index, method="ffill")
6. Average all 8 oriented series (2 signals × 4 TFs) → composite.
7. Z-score using IS mean and IS std only:
       composite_z = (composite - IS_mean) / IS_std
8. Apply to OOS period unchanged.
```

**Why z-score?** The raw composite has no natural scale. Z-scoring normalises it so that
the threshold (e.g., 0.75) has the same meaning regardless of market regime or pair.
The key anti-overfitting rule: IS mean/std are frozen and applied to OOS as-is. The OOS
z-score may drift outside the IS distribution — this is expected and not adjusted.

**Entry/exit:**

```
Long  entry : composite_z crosses above  +threshold  (executed on next-bar open)
Long  exit  : composite_z drops below   0
Short entry : composite_z crosses below -threshold
Short exit  : composite_z rises above   0
```

The `signal.shift(1)` (1-bar lag) in VBT ensures all entries fill on the bar *after* the
signal fires. This prevents look-ahead from same-bar fills.

### Full Cost Model

All three components are applied simultaneously. There is no version of the backtest with
costs turned off.

#### 1. Spread

The bid/ask half-spread paid at each fill. Expressed in raw price units, then normalised by
median close for VBT (which expects fees as a fraction of notional):

```python
vbt_fees = spread_price_units / median_close
```

| Pair    | Spread (price units) | Equivalent |
|---------|---------------------|-----------|
| EUR/USD | 0.00005 | 0.5 pip |
| GBP/USD | 0.00008 | 0.8 pip |
| USD/JPY | 0.007   | 0.7 pip |
| AUD/USD | 0.00007 | 0.7 pip |
| AUD/JPY | 0.010   | 1.0 pip |
| USD/CHF | 0.00007 | 0.7 pip |

#### 2. Slippage

Market impact / execution slippage at each fill. Default: 0.5 pip per fill (both entry and
exit). Applied in the same VBT normalisation as spread:

```python
vbt_slip = (0.5 × pip_size) / median_close
```

Total friction per round trip = spread × 2 + slippage × 2 (one entry, one exit).

#### 3. Swap (Overnight Carry)

Swap is the interest differential paid or received for holding a forex position overnight.
It is computed post-hoc and subtracted from gross P&L.

**Mechanics:**

Rollover occurs once per day at 21:00 UTC (NY 17:00 ET broker cut). For each 21:00 bar where
a position is open, the strategy pays or receives:

```
swap_drag_per_night = avg_size_pct × (pip_size / median_close) × swap_pips
```

where `avg_size_pct` is the median leveraged position size (e.g., 4.0 = 400% of equity).

**Swap defaults (pips/night):**

| Pair    | Long (pay) | Short (receive) | Notes |
|---------|-----------|-----------------|-------|
| EUR/USD | −0.5 | +0.3 | EUR rate > USD rate, long costs |
| GBP/USD | −0.2 | +0.1 | GBP/USD rate differential |
| USD/JPY | +1.2 | −1.5 | JPY near zero; long USD earns carry |
| AUD/USD | −0.3 | +0.2 | Carry near neutral |
| AUD/JPY | +0.5 | −0.7 | Classic carry trade pair |
| USD/CHF | +0.2 | −0.4 | CHF negative rates historically |

Negative long swap = you pay every night you hold long. Positive short swap = you receive
carry for holding short.

**Swap impact at typical holding periods:** IC MTF holds positions for ~20–60 H1 bars
(~1–3 days). Swap is not negligible for JPY cross pairs held for extended periods.

### Position Sizing

```python
atr14 = ATR(14, H1)
stop_distance = stop_atr_mult × atr14          # price distance to stop (not placed)
stop_pct = stop_distance / close               # stop as fraction of price
size_pct = risk_pct / stop_pct                 # portfolio fraction to trade
size_pct = min(size_pct, MAX_LEVERAGE)         # cap at 30× in backtest
```

With `risk_pct = 0.01` and `stop_atr_mult = 1.5`:
- If ATR14 = 0.0010 on EUR/USD (price ≈ 1.08): stop_pct ≈ 0.139%, size_pct ≈ 7.2×
- Size is currency-agnostic: the same formula works for EUR/USD (~1.08) and USD/JPY (~145)
  without any manual pip-value conversion, because both stop and price are in the same units.

`size_type="percent"` in VBT means size_pct is the fraction of current equity allocated
(not fixed notional). This compounds returns correctly through the equity curve.

### Threshold Search (IS Only)

Thresholds tested: `[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]`

Best threshold = argmax(IS combined Sharpe). That threshold is applied unchanged to OOS.
Only two distinct values emerged as optimal across all pairs:

- **0.75z** — EUR/USD, USD/JPY, AUD/JPY (higher signal frequency, more trades)
- **1.00z** — GBP/USD, AUD/USD, USD/CHF (fewer but higher-confidence trades)

### Run

```bash
# Single pair
uv run python research/ic_analysis/run_ic_backtest.py --instrument EUR_USD

# All pairs
for inst in EUR_USD GBP_USD USD_JPY AUD_USD AUD_JPY USD_CHF; do
    uv run python research/ic_analysis/run_ic_backtest.py --instrument $inst
done
```

### Phase 3 Gates

| Gate | Threshold |
|------|----------|
| OOS combined Sharpe | > 1.5 |
| IS/OOS parity | > 0.5 |

### Phase 3 Results (fresh run 2026-03-19)

| Pair    | OOS Sharpe | Threshold | OOS Net Return | IS/OOS Parity |
|---------|-----------|----------|----------------|--------------|
| EUR/USD | 7.709 | 0.75z | +2,571,043% | 0.838 |
| GBP/USD | 8.475 | 1.00z | +16,441%    | 1.087 |
| USD/JPY | 7.351 | 0.75z | +7,612%     | 1.066 |
| AUD/USD | 6.901 | 1.00z | +3,295%     | 1.004 |
| AUD/JPY | 7.334 | 0.75z | +3,482%     | 1.195 |
| USD/CHF | 7.476 | 1.00z | +8,419%     | 1.054 |

All 6 pairs pass Phase 3.

### Output

```
.tmp/reports/backtest_summary_{slug}.csv
```

---

## Phase 4 — Walk-Forward Optimisation (WFO)

**Script:** `research/ic_analysis/run_wfo.py`

**Purpose:** Validate temporal stability by repeatedly re-fitting the strategy on a rolling
IS window and evaluating on the subsequent OOS window. This replicates what live deployment
will actually look like — the threshold is re-selected each time the IS window advances.

### WFO Structure

```
    ┌──────────────── IS (2yr) ────────────────┐ ┌── OOS (6mo) ──┐
    |                                          | |               |
    t0                                         t1               t2

                ┌──────────────── IS (2yr) ────────────────┐ ┌── OOS (6mo) ──┐
                t1                                         t2               t3
    ...
```

- **IS window:** 2 years = 17,520 H1 bars (rolling, not expanding)
- **OOS window:** 6 months = 4,380 H1 bars
- **Step:** 6 months (OOS window length — non-overlapping OOS periods)
- **Stitched OOS:** All OOS periods concatenated → one continuous equity curve

| Pair | History | Folds | Stitched OOS coverage |
|------|---------|-------|----------------------|
| EUR/USD | 21yr | 27 | 13.3yr |
| All others | 10yr | 10 | 4.9yr |

### Per-Fold Procedure

For each fold:
1. Select IS data: `bars[i : i + IS_bars]`
2. Search threshold on IS combined Sharpe (same grid as Phase 3)
3. Apply best IS threshold to OOS data: `bars[i + IS_bars : i + IS_bars + OOS_bars]`
4. Record IS Sharpe, OOS Sharpe, threshold used, win rate

### Gates

| Gate | Threshold | Rationale |
|------|----------|-----------|
| % folds with OOS Sharpe > 0 | ≥ 70% | Strategy must be profitable in most periods |
| % folds with OOS Sharpe > 1 | ≥ 50% | Meaningful edge in at least half |
| Worst single fold OOS Sharpe | ≥ −2.0 | No catastrophic regime breaks |
| Stitched OOS Sharpe (arithmetic) | ≥ 1.5 | Full OOS period must be investable |
| Aggregate OOS/IS parity | ≥ 0.5 | OOS must be at least half as good as IS |

Parity > 1.0 (OOS better than IS) is not unusual here — it indicates the IS optimisation
is conservative and the signal generalises well.

### Run

```bash
uv run python research/ic_analysis/run_wfo.py --instrument EUR_USD

for inst in EUR_USD GBP_USD USD_JPY AUD_USD AUD_JPY USD_CHF; do
    uv run python research/ic_analysis/run_wfo.py --instrument $inst
done
```

### Phase 4 Results (fresh run 2026-03-19)

| Pair | Folds | >0% | >1% | Worst | Stitched Sh | Parity |
|------|-------|-----|-----|-------|------------|--------|
| EUR/USD | 27 | 100% | 96.3% | +0.756 | +10.242 | 0.976 |
| GBP/USD | 10 | 100% | 100% | +7.425 | +9.851 | 1.006 |
| USD/JPY | 10 | 100% | 100% | +5.273 | +7.269 | 1.031 |
| AUD/USD | 10 | 100% | 100% | +3.305 | +8.528 | 0.998 |
| AUD/JPY | 10 | 100% | 100% | +3.910 | +7.733 | 1.020 |
| USD/CHF | 10 | 100% | 100% | +5.517 | +8.475 | 1.011 |

All 6 pairs pass all 5 WFO gates.

### Output

```
.tmp/reports/wfo_{slug}.csv           # per-fold: IS Sharpe, OOS Sharpe, threshold, win rate
.tmp/reports/wfo_equity_{slug}.csv    # stitched OOS equity curve (one row per H1 bar)
```

---

## Phase 5 — Robustness Validation

**Script:** `research/ic_analysis/run_ic_robustness.py`

**Purpose:** Four adversarial stress tests that go beyond what standard backtesting reveals.
All four gates must pass before Phase 6 begins.

### Critical Implementation Note: Per-Trade % Returns

The VBT backtest uses `size_type="percent"` — position size is a fraction of current equity.
This means the portfolio compounds: a strategy starting at $10K that grows to $500K will
generate far larger absolute P&L on late trades than early ones, simply due to larger position
sizes.

**Using absolute trade P&L for Monte Carlo or top-N removal is wrong.** It creates a
measurement distorted by equity growth, not by the edge of individual trades.

**Correct approach:** use `pf.trades.records_readable["Return"]` — VBT's built-in normalised
per-trade return (P&L / initial position value). This is currency-agnostic and
compounding-independent: a 2% winning trade looks the same whether it happened when equity
was $10K or $500K.

### Gate 1: Monte Carlo Trade Shuffle

**What it tests:** Whether the observed Sharpe ratio reflects genuine edge or is a lucky
sequence of wins that would not be reproducible.

**Method:**

```
1. Extract OOS trade returns: combined = concat(pf_long.trades.Return, pf_short.trades.Return)
2. Shuffle trade order: N=1,000 times
3. For each shuffle, annualise Sharpe:
       trades_per_year = n_trades / (oos_bars / (252 × 24))
       mu = mean(shuffled_returns)
       sigma = std(shuffled_returns)
       sharpe = (mu / sigma) × sqrt(trades_per_year)
4. Report 5th-percentile and 95th-percentile Sharpe
5. Report % of simulations with positive total return
```

**Why annualise by trades_per_year?** The shuffle destroys the time structure but preserves
the return distribution. To get a Sharpe comparable to the original, we annualise using the
actual number of trades per year (how frequently the strategy trades), not by the number of
bars.

**Gates:**
- 5th-percentile Sharpe > 0.5
- % profitable simulations > 80%

**Interpretation:** If the 5th-pct Sharpe is below 0.5, it means that in 5% of possible
trade orderings, the strategy barely breaks even. This would indicate the edge is
sequence-dependent — consistent with lucky clustering of winners, not genuine alpha.

### Gate 2: Remove Top-10 Winning Trades

**What it tests:** Whether returns are concentrated in a small number of outlier trades.
A strategy where the 10 best trades generate >50% of total returns is fragile — it relied
on rare events, which may not recur.

**Method:**
```
1. Combine long + short per-trade % returns
2. Remove the 10 largest by absolute magnitude (the 10 best wins)
3. Sum the remaining returns
4. Gate: remaining sum > 0 (strategy still profitable without the outliers)
```

### Gate 3: 3× Slippage Stress Test

**What it tests:** Sensitivity of edge to execution quality. Real-world slippage can spike
during volatile markets. If the strategy breaks when slippage triples, it is not robust.

**Method:** Re-run the full OOS backtest with `slippage = 1.5 pip per fill` (3× the 0.5 pip
default). All other parameters (spread, swap, threshold) are unchanged.

**Gate:** Combined OOS Sharpe > 0.5

At 1.5 pip per fill, the strategy still needs to be investable. A Sharpe of 0.5 is the
minimum acceptable for a live systematic strategy with transaction costs.

### Gate 4: WFO Consecutive Negative Folds

**What it tests:** Whether the strategy experienced sustained losing streaks in the WFO. A
single bad fold (~6 months of losses) can happen due to regime shifts. Two consecutive bad
folds (~12 months) is borderline. Three or more suggests the signal has genuinely lost its
edge for an extended period — which is a serious concern for live deployment.

**Method:** Read Phase 4 WFO CSV, scan OOS Sharpe column for the longest run of consecutive
negative values.

**Gate:** Max consecutive negative folds ≤ 2

### Run

```bash
# All 6 pairs
uv run python research/ic_analysis/run_ic_robustness.py

# Single pair
uv run python research/ic_analysis/run_ic_robustness.py --instrument EUR_USD
```

### Phase 5 Results (fresh run 2026-03-19)

| Pair | Combined Sh | MC 5pct | %Prof | Top-N | 3× Slip Sh | Consec Neg | ALL |
|------|------------|---------|-------|-------|-----------|-----------|-----|
| EUR/USD | 7.71 | 7.49 | 100% | +358% | 6.06 | 0 | PASS |
| GBP/USD | 8.28 | 8.69 | 100% | +162% | 6.85 | 0 | PASS |
| USD/JPY | 7.35 | 6.90 | 100% | +185% | 6.49 | 0 | PASS |
| AUD/USD | 6.84 | 7.03 | 100% | +171% | 4.87 | 0 | PASS |
| AUD/JPY | 7.33 | 7.27 | 100% | +206% | 6.02 | 0 | PASS |
| USD/CHF | 7.34 | 7.55 | 100% | +151% | 5.41 | 0 | PASS |

6/6 pairs pass all 4 robustness gates. Cleared for Phase 6.

### Output

```
.tmp/reports/ic_robustness_summary.csv
```

---

## Phase 6 — Live Implementation

**Files:**

| File | Purpose |
|------|---------|
| `titan/strategies/ic_mtf/strategy.py` | NautilusTrader strategy class |
| `config/ic_mtf.toml` | Per-pair thresholds and sizing parameters |
| `scripts/run_live_ic_mtf.py` | Live runner (all 6 pairs) |

### How the Live Strategy Mirrors the Research

The live strategy replicates the backtest composite construction faithfully:

1. **Warmup calibration** (`_warmup_and_calibrate`): loads the last 1000 bars per TF from
   parquet, computes `accel_rsi14` and `accel_stoch_k` on each, derives IC signs via Spearman
   correlation, and freezes composite mean/std for z-scoring — exactly as the IS period does
   in the research.
2. **On each H1 bar** (`on_bar` → `_update_tf_signals`): recomputes both signals for that TF,
   stores the latest oriented value in `self.latest_signal[(sig, tf)]`.
3. **Higher TF updates**: H4/D/W signals update only when those bars close. Between closings,
   their last-known value is used (forward-fill, exactly as `method="ffill"` in the research).
4. **Composite z** (`_get_composite_z`): `mean(latest_signal.values())` normalised by
   calibration stats. Evaluated on every H1 bar.
5. **Entry/exit**: threshold crossing for entries, zero crossing for exits.
6. **Sizing**: identical formula — `risk_pct / (stop_atr_mult × ATR14 / close)`, capped
   at leverage_cap.

### Config (config/ic_mtf.toml)

```toml
[EUR_USD]
threshold     = 0.75
risk_pct      = 0.01
stop_atr_mult = 1.5
leverage_cap  = 20.0
warmup_bars   = 1000
```

Thresholds match Phase 3 best values. Do not change without re-running Phase 3–5.

### Running Live (Paper)

```bash
uv run python scripts/run_live_ic_mtf.py
```

Requires `IBKR_ACCOUNT_ID`, `IBKR_HOST`, `IBKR_PORT` in `.env`. Client ID = 4 (ORB=2, MTF=3).

---

## Full Pipeline Run

```bash
# Phase 1 — Signal sweep (EUR/USD H4 as canonical)
uv run python research/ic_analysis/run_signal_sweep.py --instrument EUR_USD --timeframe H4

# Phase 2 — Combination check
uv run python research/ic_analysis/run_signal_combination.py \
    --instrument EUR_USD --timeframe H4 --horizon 20

# Phase 3 — IS/OOS backtest (all pairs)
for inst in EUR_USD GBP_USD USD_JPY AUD_USD AUD_JPY USD_CHF; do
    uv run python research/ic_analysis/run_ic_backtest.py --instrument $inst
done

# Phase 4 — Walk-forward (all pairs)
for inst in EUR_USD GBP_USD USD_JPY AUD_USD AUD_JPY USD_CHF; do
    uv run python research/ic_analysis/run_wfo.py --instrument $inst
done

# Phase 5 — Robustness (all pairs)
uv run python research/ic_analysis/run_ic_robustness.py
```

Expected total runtime: ~8–12 minutes for all phases across all pairs.

---

## Equities Extension — Regime-Gated Long-Only Strategy

The IC research pipeline was extended from FX to US equities. All equities scripts operate on
daily bars from `data/{INSTRUMENT}_D.parquet` (downloaded via yfinance or IBKR).

### Key Difference from IC MTF

The FX strategy is long/short on a momentum composite. The equities strategy is **long-only**
and uses an **ADX regime gate** to switch between two composites:

- ADX < 20 (Ranging): oscillator/mean-reversion composite
- ADX > 25 (Trending): momentum/trend composite
- ADX 20–25 (Neutral): flat — no new entries

### CAT Results (Phase 3–5, 11yr data 2015–2026)

CAT is the only instrument that passes the full robustness gate as of 2026-03-19:

| Phase | Metric | Value | Gate |
|-------|--------|-------|------|
| Phase 3 | OOS Sharpe | +1.97 | > 1.5 ✅ |
| Phase 3 | Max Drawdown | −1.5% | — |
| Phase 4 | WFO >0% folds | 89% | ≥ 70% ✅ |
| Phase 4 | WFO >1 Sharpe folds | 67% | ≥ 50% ✅ |
| Phase 4 | Worst fold | −2.13 | ≥ −2.0 ❌ |
| Phase 5 | MC 5th-pct Sharpe | +0.60 | > 0.5 ✅ |
| Phase 5 | Top-5 removal | +0.076 | > 0 ✅ |
| Phase 5 | 3× slippage | +1.97 | > 0.5 ✅ |
| Phase 5 | Max consec neg folds | 2 | ≤ 2 ✅ |

AMAT fails Phase 4 (61% positive folds) and Phase 5 (returns concentrated in top trades).
See `directives/IC Signal Analysis.md` for the full cross-instrument leaderboard.

### Equities Scripts

| Script | Purpose |
|--------|---------|
| `research/ic_analysis/run_regime_ic.py` | Regime-conditional IC — ADX + vol axes |
| `research/ic_analysis/run_regime_backtest.py` | Per-instrument ADX-gated backtest (long+short) |
| `research/ic_analysis/run_cat_amat_strategy.py` | CAT + AMAT Phase 3 with Plotly equity chart |
| `research/ic_analysis/run_cat_amat_pipeline.py` | Full Phase 3–5 pipeline for CAT + AMAT |

```bash
# Regime IC analysis
uv run python research/ic_analysis/run_regime_ic.py --instrument CAT --timeframe D --horizon 20

# Full pipeline (Phases 3-5)
uv run python research/ic_analysis/run_cat_amat_pipeline.py
```

Data download (Yahoo Finance, 11yr history):
```bash
uv run python scripts/download_data_yfinance.py --symbols CAT AMAT --start 2015-01-01
```

---

## Key Design Principles

### 1. No Look-Ahead Bias

- Forward returns: **only** via `.shift(-h)` in `compute_forward_returns()` — the single
  permitted negative shift in the entire codebase.
- Higher TF reindex: always `method="ffill"` — at H1 bar t, you can only see the H4/D/W
  candle that *closed* prior to t. Before reindexing, non-base TF signals must be shifted by
  1 bar (`native_sigs.shift(1)`) to prevent the close of the current higher-TF bar from
  leaking into the current H1 bar. **Without this shift, OOS Sharpe inflates by ~10 points**
  (confirmed via controlled test 2026-03-19: biased = +7.709, debiased = −2.548 on EUR/USD H1).
  The fix is present in `run_ic_backtest.py` (lines 165–166), `run_wfo.py` (lines 110–112),
  and `run_mtf_stack.py` (`_align_to_base`, `is_coarser` parameter). Any new script that
  aligns higher-TF signals to a base TF must apply the same shift.
- IS/OOS split: always time-ordered. Never shuffle.
- WFO: IS window calibration applied to next OOS window only, never the same window.
- Z-score normalisation: IS mean/std frozen, applied to OOS as-is.

### 2. Full Friction on Every Run

The cost model (spread + slippage + swap) is never disabled for any phase, any pair, or
any threshold. There is no "gross" backtest in the research pipeline — every Sharpe
ratio reported includes all three friction components.

### 3. OOS Is the Only Number That Matters

IS Sharpe is used internally for threshold selection only. It is never reported as a
performance claim. Every external-facing metric (Phase 3 Sharpe, WFO parity, robustness
gate Sharpes) is computed on OOS data only.

### 4. Per-Trade % Returns for Monte Carlo

Never use absolute trade P&L when the portfolio compounds (`size_type="percent"` in VBT).
Use `pf.trades.records_readable["Return"]`. This gives a compounding-independent per-trade
return that makes Sharpe calculation valid across the full equity curve.

### 5. ATR-Based Sizing Is Currency-Agnostic

By expressing stop distance as a fraction of price (`stop_pct = ATR / close`), the sizing
formula works identically for EUR/USD (~1.08) and USD/JPY (~145) without any manual pip-value
conversion. The same 1% risk target applies to all pairs.

---

## File Reference

| File | Phase | Purpose |
|------|-------|---------|
| `research/ic_analysis/run_signal_sweep.py` | 1 | IC/ICIR sweep across 52 signals |
| `research/ic_analysis/run_ic.py` | 1 | IC, ICIR, forward return helpers |
| `research/ic_analysis/run_regime_ic.py` | 1 | Regime-conditional IC (ADX + vol axes) |
| `research/ic_analysis/run_signal_combination.py` | 2 | 6 combination methods |
| `research/ic_analysis/run_param_sweep.py` | 2 | Parameter grid search |
| `research/ic_analysis/run_ic_backtest.py` | 3 | IS/OOS full-friction backtest (FX MTF) |
| `research/ic_analysis/run_regime_backtest.py` | 3 | ADX-gated backtest per instrument |
| `research/ic_analysis/run_cat_amat_strategy.py` | 3 | CAT + AMAT Phase 3 with Plotly chart |
| `research/ic_analysis/run_cat_amat_pipeline.py` | 3–5 | CAT + AMAT full pipeline (Phases 3–5) |
| `research/ic_analysis/run_wfo.py` | 4 | Walk-forward optimisation (FX MTF) |
| `research/ic_analysis/run_ic_robustness.py` | 5 | 4 robustness gates (FX MTF) |
| `titan/strategies/ic_mtf/strategy.py` | 6 | NautilusTrader live strategy |
| `config/ic_mtf.toml` | 6 | Per-pair live config |
| `scripts/run_live_ic_mtf.py` | 6 | Live runner |
| `scripts/download_data_yfinance.py` | Data | Yahoo Finance download (equities, 10yr+) |

---

## Validation Checklist (Phases 1–5 → Phase 6)

- [x] Phase 1: Signals selected from 52-signal sweep with ICIR > 0.5 and |IC| > 0.05
- [x] Phase 2: Equal-weight composite confirmed optimal vs. PCA/ICIR-weighted alternatives
- [x] Phase 3: OOS Sharpe > 1.5 for all 6 pairs
- [x] Phase 3: IS/OOS parity > 0.5 for all 6 pairs
- [x] Phase 4: All 5 WFO gates pass for all 6 pairs
- [x] Phase 5: Monte Carlo 5th-pct Sharpe > 0.5, >80% profitable — all pairs
- [x] Phase 5: Remove top-10 trades — remaining return > 0 — all pairs
- [x] Phase 5: 3× slippage (1.5 pip) — OOS Sharpe > 0.5 — all pairs
- [x] Phase 5: WFO max consecutive negative folds ≤ 2 — all pairs (0 for every pair)
- [x] No `.shift(-h)` outside `compute_forward_returns()`
- [x] Full cost model active in every backtest phase

All items confirmed as of **2026-03-19** for EUR/USD, GBP/USD, USD/JPY, AUD/USD, AUD/JPY,
USD/CHF.
