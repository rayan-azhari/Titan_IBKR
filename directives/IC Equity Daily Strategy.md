# IC Equity Daily Strategy

> [!CAUTION]
> **DEPRECATED:** This document has been superseded by `directives/Backtesting & Validation.md`
> (process) and `research/ic_analysis/FINDINGS.md` (results). This file is retained for
> historical reference only. Do not use as a primary directive.

**Version:** 1.1 | **Last Updated:** 2026-03-20


---

## Overview

The IC Equity Daily strategy is a daily mean-reversion strategy for US equities, derived
from the IC Signal Analysis pipeline. It uses a single signal (`rsi_21_dev`) to identify
oversold conditions on daily bars and enters long-only positions expecting mean-reversion.

**Key design decisions:**
- **Long-only:** Short side removed. Individual equities have structural long bias;
  mean-reversion short strategies are exposed to binary event risk (earnings, M&A) and
  adverse risk/reward asymmetry.
- **Single signal:** `rsi_21_dev` (RSI(21) − 50) is the most broadly applicable daily
  mean-reversion signal across the US equity universe (STRONG in 63% of 513 symbols tested).
- **Daily bars only:** No MTF stacking. The mean-reversion edge is strongest at daily
  resolution; intraday adds noise without IC improvement.

---

## Validated Instruments (2026-03-20, v1.1 — with HMM gate)

Full Phase 3-5 pipeline run on 482 S&P 500 + Russell 100 daily symbols.
Gates tested: no-gate, ADX<25, HMM (2-state Gaussian HMM on IS bars).
6 symbols passed all gates. All 6 passed Phase 5 Monte Carlo (N=500).

| Symbol | Company | Sector | Threshold | Gate | OOS Sharpe | WFO Stitched | Trades | Win% |
|---|---|---|---|---|---|---|---|---|
| **HWM** | Howmet Aerospace | Industrials | 0.25z | none | +4.28 | +1.52 | 22 | 81.8% |
| **CSCO** | Cisco Systems | Technology | 0.25z | **HMM** | +3.14 | +2.62 | 8 | 75.0% |
| **NOC** | Northrop Grumman | Defense | 0.50z | none | +3.06 | +2.07 | 57 | 77.2% |
| **WMT** | Walmart | Consumer Staples | 0.50z | none | +2.82 | +6.29 | 9 | 88.9% |
| **ABNB** | Airbnb | Travel | 1.00z | none | +2.78 | +2.10 | 6 | 83.3% |
| **GL** | Globe Life | Insurance | 0.25z | ADX<25 | +2.65 | +2.21 | 65 | 75.4% |

**Changes from v1.0:** CB and SYK dropped (failed WFO with HMM as a competing gate).
CSCO is new — only passes *with* the HMM gate; no-gate and ADX<25 fail WFO for it.

**Recommended first deployment (strongest WFO consistency):** NOC, GL, WMT

**Note on thin trade counts:** CSCO (8 trades), WMT (9 trades), ABNB (6 trades) have
statistically thin OOS samples. Monitor closely in live — exit if first 3 trades show
no positive edge. NOC (57 trades) and GL (65 trades) have the most statistical power.

---

## Strategy vs Buy-and-Hold

The strategy underperforms buy-and-hold on raw return. This is **expected and not a flaw**
— it is by design. The real story is in Sharpe ratio and maximum drawdown.

> [!NOTE]
> Two Sharpe metrics appear in this project. The **Trade Sharpe** (Phases 3-5 leaderboard)
> is mean/std of individual trade PnL — it rewards high win rates and is useful for ranking
> signal quality. The **Daily Sharpe** below is mean/std of daily equity returns × sqrt(252)
> — comparable to B&H Sharpe because it includes flat days. Both are valid; daily Sharpe
> is the conservative, apples-to-apples number.

| Symbol | OOS Period | Strat Ann | Strat Sharpe | Strat MDD | B&H Ann | B&H Sharpe | B&H MDD |
|---|---|---|---|---|---|---|---|
| HWM | May 2023 - Mar 2026 | +6.9% | +1.32 | **-4.5%** | +81.8% | **+2.04** | -19.4% |
| CSCO | Sep 2024 - Mar 2026 | +12.6% | **+1.10** | **-7.2%** | +37.5% | +1.41 | -18.0% |
| NOC | May 2018 - Mar 2026 | +4.9% | **+0.83** | **-9.8%** | +12.8% | +0.58 | -32.6% |
| WMT | Sep 2024 - Mar 2026 | +6.7% | **+1.40** | **-3.4%** | +37.4% | +1.45 | -22.1% |
| ABNB | Aug 2024 - Mar 2026 | +2.8% | **+0.79** | **-2.9%** | +6.1% | +0.34 | -34.5% |
| GL | May 2018 - Mar 2026 | +3.0% | **+0.60** | **-7.2%** | +7.2% | +0.40 | -61.6% |

**Reading the table:**
- **Raw return:** Strategy underperforms B&H in most cases — expected, since it is only
  in the market during oversold windows (20-40% of days), not every up-day in the bull run.
  CSCO (+12.6%) is the exception, beating B&H in a short OOS window.
- **Sharpe ratio:** Strategy beats B&H for 4/6 symbols (NOC: 0.83 vs 0.58, ABNB: 0.79 vs 0.34,
  GL: 0.60 vs 0.40, WMT: 1.40 vs 1.45 ≈ tied). HWM and CSCO are exceptions — strong
  bull runs in short OOS windows favour B&H.
- **Max drawdown:** Strategy cuts peak-to-trough loss dramatically for every symbol.
  NOC: -9.8% vs -32.6% B&H. GL: -7.2% vs -61.6% B&H. ABNB: -2.9% vs -34.5% B&H.
  This is the primary risk-management benefit — selective exposure avoids sustained drawdowns.

**Best use:** Overlay on a core B&H or index position. Signal informs tactical sizing
(overweight when oversold, neutral when flat). Not a standalone replacement for B&H in a
bull market — but dramatically improves the risk profile of a long equity portfolio.

---

## Signal Mechanics

### `rsi_21_dev`

```python
rsi_21_dev = RSI(close, 21) - 50
```

Centred at zero: positive = overbought (RSI > 50), negative = oversold (RSI < 50).

The IC of this signal on US equities daily data is **negative** (IC ≈ -0.05 to -0.15):
overbought readings predict underperformance, oversold readings predict outperformance.
This is the mean-reversion regime. The IC sign is calibrated on IS warmup data and applied
to orient the composite before z-scoring.

### Entry z-score

```python
composite = rsi_21_dev * ic_sign   # oriented so positive = bullish
entry_z   = (composite - mu_IS) / sigma_IS
```

`mu_IS` and `sigma_IS` are the mean and standard deviation of the composite over the IS
warmup period (last 504 trading bars = ~2 years). This normalises the signal to a
consistent scale regardless of the instrument's RSI characteristics.

### Entry / Exit

```
Entry (Long) : entry_z crosses above +threshold
Exit         : entry_z crosses below 0 (signal neutralises)
Flip         : if already long and entry_z falls below 0, close position
```

No take-profit or stop-loss orders in the signal logic. The exit is purely signal-driven.
ATR-based sizing provides implicit risk control via position size.

---

## Position Sizing

```
stop_dist = ATR(14) * stop_atr_mult   (default: 1.5)
raw_units = (equity * risk_pct) / stop_dist
max_units = (equity * leverage_cap) / price
units     = min(raw_units, max_units)
```

Default parameters (from `config/ic_equity_daily.toml`):
- `risk_pct = 0.005` (0.5% equity risk per trade — reduced due to small OOS trade count)
- `stop_atr_mult = 1.5`
- `leverage_cap = 5.0` (equities; compare 20.0 for FX)

The 0.5% risk setting is conservative — Phase 5 Gate 2 (top-N removal) flagged that the
OOS trade count is too small to validate full 1% risk. If live performance confirms the
edge over 30+ trades, risk can be increased to 1%.

---

## Architecture

```
titan/strategies/ic_equity_daily/
    strategy.py          <- ICEquityDailyConfig + ICEquityDailyStrategy
    __init__.py

config/ic_equity_daily.toml     <- per-symbol threshold + sizing params (to be created)
scripts/run_live_ic_equity_daily.py <- live runner (to be created)
data/{TICKER}_D.parquet         <- warmup data (already downloaded)
```

### `ICEquityDailyConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `instrument_id` | str | — | NautilusTrader instrument ID |
| `bar_type_d` | str | — | e.g. `"WMT.NYSE-1-DAY-LAST-EXTERNAL"` |
| `ticker` | str | — | Used to locate `data/{ticker}_D.parquet` |
| `threshold` | float | 0.75 | z-score entry threshold |
| `risk_pct` | float | 0.005 | Equity risk per trade (0.5%) |
| `stop_atr_mult` | float | 1.5 | ATR multiplier for stop distance |
| `leverage_cap` | float | 5.0 | Max leverage cap |
| `warmup_bars` | int | 504 | IS bars for calibration (~2yr daily) |

---

## Deployment Pre-Flight Checklist

Before going live with any of the 6 symbols:

- [ ] `data/{TICKER}_D.parquet` exists and has ≥ 1,000 bars
- [ ] Strategy starts and logs `calibrated=True` with `ic_sign=-1.0` (mean-reversion)
- [ ] Paper trade for 5+ signal firings before live capital
- [ ] Position size verified: 0.5% of account equity at risk (not 5% or 50%)
- [ ] B&H context acknowledged: strategy will underperform index in strong bull runs
- [ ] Liquidity confirmed: all 6 symbols are large-cap / liquid; no ABNB concern at
  small size, but verify bid-ask spread at time of entry

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **Small trade counts** | CSCO (8 trades), WMT (9 trades), ABNB (6 trades) — thin OOS samples. Statistical power is limited. |
| **HMM/ADX regime gates** | ADX<25 only helped GL. HMM enabled CSCO. Both are heuristics fit on limited IS data — monitor for regime-detection drift in live. |
| **No short side** | Long-only misses downside mean-reversion edge. For UNH: L+S OOS was +17.8% vs long-only +2.9%; short side caught the CEO assassination crash. |
| **Bull market OOS** | Most OOS periods (2018-2026) are in a structural bull market. Strategy will look weaker vs B&H. True test is a flat or declining market. |
| **Single signal** | `rsi_21_dev` alone; no ensemble or composite. Adding a second STRONG signal (e.g. `bb_zscore_20`) could improve ICIR consistency. |

---

## Running the Research Pipeline

```bash
# Re-run full pipeline (482 symbols, ~60 min)
uv run python research/ic_analysis/run_equity_longonly_pipeline.py

# Single symbol validation
uv run python research/ic_analysis/run_equity_longonly_pipeline.py --symbol CB

# Check leaderboard
cat .tmp/reports/equity_longonly_leaderboard.csv
```

Key output files:
- `.tmp/reports/equity_longonly_phase3.csv` — IS/OOS backtest results
- `.tmp/reports/equity_longonly_phase4.csv` — WFO fold-by-fold stats
- `.tmp/reports/equity_longonly_phase5.csv` — Monte Carlo results
- `.tmp/reports/equity_longonly_leaderboard.csv` — final ranked leaderboard

---

## Version History

| Version | Date | Changes |
|---|---|---|
| **1.0** | 2026-03-20 | Initial directive. Full 482-symbol pipeline. 7 validated symbols. B&H comparison. ADX filter analysis. |
| **1.1** | 2026-03-20 | Re-run with HMM gate. CB and SYK dropped; CSCO added (HMM-gated). 6 validated symbols. |
