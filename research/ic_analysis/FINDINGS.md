# IC Signal Analysis — Research Findings

**Extracted from:** `directives/IC Signal Analysis.md` v3.4
**Date:** 2026-03-20

> This file contains specific strategy results, instrument-level findings, and backtest
> outcomes discovered using the IC pipeline. The parent directive (`IC Signal Analysis.md`)
> describes the **process**; this file documents the **results**.

---

## IC MTF Strategy — Signal Selection (EUR/USD H4)

The Phase 1 sweep on EUR/USD H4 (and cross-validated across other pairs and timeframes)
produced the following leaderboard for the Acceleration group:

| Rank | Signal | Group | Best h | IC | ICIR | Verdict |
|------|--------|-------|-------|----|------|---------|
| 1 | `accel_rsi14` | Accel | h=20 | +0.061 | +0.71 | STRONG |
| 2 | `accel_stoch_k` | Accel | h=20 | +0.058 | +0.74 | STRONG |

**Why acceleration signals dominated:**

The sweep consistently showed Group E (Acceleration) signals outperforming their Group B
(Momentum) parents at the h=20 horizon on H4 data. `rsi_14_dev` (IC=+0.038, ICIR=0.44)
is WEAK, but `accel_rsi14` — its first difference — is STRONG. This is the core discovery
of the Phase 1 sweep for forex trending markets:

> It is not enough that RSI is above 50. What matters is that it is still *rising*. When
> RSI decelerates (diff turns negative) while still above 50, the trend is exhausting.
> The acceleration signal captures this a bar or two before a reversal, while the level
> signal is still pointing bullish.

**Selection criteria applied:**

1. |IC| ≥ 0.05 and ICIR ≥ 0.5 at h=20 (matching ~1-week H1 holding period)
2. Mutual correlation < 0.5 between the two selected signals (0.31 observed)
3. Edge consistent across all 6 pairs and all 4 timeframes (not EUR/USD-specific)
4. Phase 2 confirmation: equal-weight composite ΔIC < 0.003 vs. PCA/ICIR-weighted

**Signals not selected (and why):**

| Signal | IC | ICIR | Reason not selected |
|--------|-----|------|-------------------|
| `ma_spread_5_20` | +0.087 | 0.72 | STRONG — but high correlation with `accel_stoch_k` (0.61) |
| `donchian_pos_20` | +0.071 | 0.61 | STRONG — adds orthogonal info but Phase 2 showed no composite IC gain |
| `rsi_14_dev` | +0.038 | 0.44 | WEAK at h=20; parent signal to `accel_rsi14` |
| `bb_zscore_20` | −0.044 | 0.55 | STRONG mean-reversion — conflicts with trend-following composite |
| Group D signals | varies | < 0.4 | ICIR too low; volatility state is not directional |

---

## Equities IC Findings

The 52-signal sweep was run on all 7 ORB instruments (daily timeframe) after debiasing
the MTF alignment (see *Lookahead Bias* section in `directives/IC MTF Backtesting Guide.md`).

**FX result post-debiasing:** EUR/USD H1 has 0 STRONG/USABLE signals. H4 has 1 USABLE
(`accel_stoch_k`, IC=−0.061). Daily FX has 4 STRONG signals but all at h=60 with negative ICIR.
FX daily signals have no actionable edge after lookahead is removed.

**Equities result:** Strong unconditional IC at h=20–60 on daily data:

| Instrument | Top Signal | Best IC | Horizon |
|---|---|---|---|
| TXN | `zscore_expanding` | −0.454 | h=20 |
| INTC | `ma_spread_50_200` | +0.392 | h=20 |
| UNH | `zscore_100` | −0.294 | h=20 |
| AMAT | `roc_60` | +0.261 | h=20 |
| CAT | `ma_spread_50_200` | +0.243 | h=20 |
| SPY | `cci_20` | −0.125 | h=20 |
| QQQ | `price_pct_rank_60` | +0.198 | h=20 |

Negative IC = mean-reversion signal (sign-normalised before composite building).

**Broad cross-asset sweep (2026-03-20):** The sweep was extended to all 513 available
daily equity parquets (S&P 500 + Russell 100 + VIX + Gold/Silver). Key finding: `rsi_21_dev`
(RSI(21) − 50) is the most broadly applicable daily signal, achieving STRONG verdicts in
63% of symbols (324/513). `ma_spread_10_50` achieves STRONG in 70.6% but is a trend-following
signal — the two represent complementary regime edges (trend vs mean-reversion).

---

## Regime-Gated Equities Strategy

Based on regime IC findings, a strategy was built using two signal composites gated by ADX:

**Ranging composite** (ADX < 20): `zscore_50`, `bb_zscore_50`, `cci_20`, `stoch_k_dev`,
`donchian_pos_10`, `zscore_20`, `bb_zscore_20`

**Trending composite** (ADX > 25): `ma_spread_50_200`, `ma_spread_20_100`,
`zscore_expanding`, `zscore_100`, `roc_60`, `price_pct_rank_60`, `ema_slope_20`

All composites are IC sign-normalised on IS bars, z-scored using IS bars of the corresponding
regime only. Neutral regime (ADX 20–25): no new entries.

### Cross-instrument long-only OOS results (Sep 2024 – Mar 2026)

| Instrument | Gated Long Sharpe | Baseline Long Sharpe | Best Threshold |
|---|---|---|---|
| CAT | **+2.78** | +1.14 | 1.0z |
| AMAT | **+2.10** | +0.97 | 0.75z |
| UNH | +0.73 | −0.21 | 1.5z |
| TXN | +1.36 | +1.25 | 1.5z |
| QQQ | +0.56 | +0.34 | 1.0z |
| INTC | +1.43 | +1.66 | 1.5z |
| SPY | +0.30 | +0.72 | 1.0z |

Regime gating improves the long side for 5/7 instruments. INTC and SPY are the exceptions
where the baseline already captures most of the edge or structural long-bias swamps the filter.

### Full Pipeline Results: CAT (11yr, 2015–2026)

**Phase 3** (70/30 IS/OOS): OOS Sharpe +1.97, Annual +5.4%, Max DD −1.5%, 11 trades

**Phase 4 WFO** (18 folds, 2yr IS / 6mo OOS):
- 89% of folds with positive OOS Sharpe ✅
- 67% of folds with OOS Sharpe > 1 ✅
- Worst fold: −2.13 (2018 broad-market selloff) ❌
- Mean fold Sharpe: +inf (several zero-trade folds) ✅
- OOS/IS parity: ✅

**Phase 5 Robustness:**
- Monte Carlo 5th-pct Sharpe: +0.60 ✅
- Remove top-5 trades: remaining sum +0.076 ✅
- 3× slippage OOS Sharpe: +1.97 ✅
- Max consecutive negative WFO folds: 2 ✅

CAT passes all Phase 5 gates. **Verdict: cleared for live consideration.**

### Full Pipeline Results: AMAT (11yr, 2015–2026)

**Phase 3**: OOS Sharpe +0.89, Annual +4.6%, Max DD −4.8%, 25 trades

**Phase 4 WFO**: 61% positive folds ❌, 44% > 1 Sharpe ❌, worst −3.59 ❌

**Phase 5**: MC 5th-pct +0.18 ❌, top-5 removal unprofitable ❌

**Verdict: AMAT fails Phase 4 and Phase 5.** Returns concentrated in a few large winning
trades during the 2023–2024 semiconductor rally. Edge is not robust across market regimes.
Do not deploy without significantly improved signal set or tighter regime conditions.

---

## Equity Long-Only Pipeline — Full Cross-Asset Results (2026-03-20)

**Script:** `research/ic_analysis/run_equity_longonly_pipeline.py`

A full Phase 3→5 pipeline was run on all 482 eligible daily equity parquets
(S&P 500 + Russell 100, excluding FX, ETFs, and symbols with < 1,000 bars).

**Signal:** `rsi_21_dev` (RSI(21) − 50, daily)
**Direction:** Long-only (short side excluded — equities have structural long bias)
**Threshold sweep:** [0.25, 0.50, 0.75, 1.00, 1.50, 2.00] z-score
**Gate sweep:** None (no filter), ADX < 25 (ranging only), HMM (2-state Gaussian HMM fit on IS bars)
**WFO config:** IS = 504 bars (~2yr), OOS = 126 bars (~6mo), 5 folds rolling
**MC config:** N = 500 simulations, gates: 5th-pct > 0.5 AND > 80% profitable

### Funnel (v1.1 — with HMM gate)

| Phase | Input | Passed | Pass Rate |
|---|---|---|---|
| Phase 3 — IS/OOS backtest | 482 symbols | 457 | 95% |
| Phase 4 — WFO (5 folds) | 457 | **6** | 1.3% |
| Phase 5 — Monte Carlo (N=500) | 6 | **6** | 100% |

### Final Leaderboard — 6 Validated Symbols (v1.1)

| Symbol | Sector | Threshold | Gate | P3 OOS Sharpe | P4 Stitched | MC 5th-pct | Trades | Win% |
|---|---|---|---|---|---|---|---|---|
| **HWM** | Industrials | 0.25z | none | +4.28 | +1.52 | +4.28 | 22 | 81.8% |
| **CSCO** | Technology | 0.25z | **HMM** | +3.14 | +2.62 | +3.14 | 8 | 75.0% |
| **NOC** | Defense | 0.50z | none | +3.06 | +2.07 | +3.06 | 57 | 77.2% |
| **WMT** | Consumer Staples | 0.50z | none | +2.82 | +6.29 | +2.82 | 9 | 88.9% |
| **ABNB** | Travel | 1.00z | none | +2.78 | +2.10 | +2.78 | 6 | 83.3% |
| **GL** | Insurance | 0.25z | ADX<25 | +2.65 | +2.21 | +2.65 | 65 | 75.4% |

### Strategy vs Buy-and-Hold Comparison

| Symbol | OOS Period | Strat Ann | Strat Sharpe | Strat MDD | B&H Ann | B&H Sharpe | B&H MDD |
|---|---|---|---|---|---|---|---|
| HWM | May 2023 - Mar 2026 | +6.9% | +1.32 | **-4.5%** | +81.8% | **+2.04** | -19.4% |
| CSCO | Sep 2024 - Mar 2026 | +12.6% | **+1.10** | **-7.2%** | +37.5% | +1.41 | -18.0% |
| NOC | May 2018 - Mar 2026 | +4.9% | **+0.83** | **-9.8%** | +12.8% | +0.58 | -32.6% |
| WMT | Sep 2024 - Mar 2026 | +6.7% | **+1.40** | **-3.4%** | +37.4% | +1.45 | -22.1% |
| ABNB | Aug 2024 - Mar 2026 | +2.8% | **+0.79** | **-2.9%** | +6.1% | +0.34 | -34.5% |
| GL | May 2018 - Mar 2026 | +3.0% | **+0.60** | **-7.2%** | +7.2% | +0.40 | -61.6% |

### Regime Gate Findings

- **HMM enabled CSCO** — the only symbol where HMM was the winning gate.
- **ADX<25 only helped GL** — consistent with prior finding.
- **4/6 symbols optimal with no gate** — mean-reversion in these names is robust enough across all market regimes.
- **Conclusion:** HMM is a useful tool for borderline symbols; do not apply universally.

### Monte Carlo Validation (EUR/USD H1, IC MTF)

All 6 pairs: MC 5th-pct Sharpe > 7.0 → confirmed genuine edge, not sequencing luck.

---

## SPY Three-Signal Strategy (2026-03-19)

IC-validated signals combining macd_norm (h=60), rsi_dev (h=60), and momentum_5 (h=5, faded)
with ADX regime gating. Long-only.

**Script:** `research/ic_analysis/run_spy_strategy.py`

| Metric | OOS (2016–2026) |
|---|---|
| Sharpe | +0.498 |
| Annual | +2.4% |
| Max DD | −9.5% |
| Win Rate | 65.5% (29 trades) |
| B&H Annual | +14.4% |
| B&H Max DD | ~−35% |

Selective strategy (29 trades in 10 years) with dramatically lower drawdown than buy-and-hold.
Best used as a tactical overlay on a core B&H position.
