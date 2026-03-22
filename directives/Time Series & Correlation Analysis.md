# Time Series & Correlation Analysis

**Status:** Complete — research phase
**Scripts:** `research/spy_ts_analysis.py`, `research/ts_analysis_batch.py`, `research/correlation_analysis/run_correlation.py`
**Output:** `.tmp/*.png`, `.tmp/*.csv`

---

## Purpose

Systematic characterisation of statistical properties across the full asset universe (equities, FX, indices, commodities) to inform:
- Signal selection (trending vs mean-reverting assets)
- Portfolio construction (avoid correlated duplicates)
- Pair/spread strategy candidate identification

---

## Part 1 — Time Series Analysis

### Scripts

| Script | Purpose |
|---|---|
| `research/spy_ts_analysis.py` | Single-symbol deep-dive: 6-panel figure + console stats |
| `research/ts_analysis_batch.py` | Batch runner across all `*_D.parquet` files |

### Four tests applied

**A. Seasonal Decomposition** (`statsmodels.seasonal_decompose`)
- Model: multiplicative, period=252 (daily) or 120 (H1 weekly cycle)
- Outputs: seasonal amplitude (max−min), residual std

**B. ACF / PACF on log returns**
- Lag-1 ACF threshold: >+0.02 = trending, <−0.02 = mean-reverting
- Ljung-Box Q test at lags 10 and 20

**C. Hurst Exponent (R/S analysis)** — numba JIT
- H > 0.55 = trending/persistent
- H < 0.45 = mean-reverting
- 0.45–0.55 = random walk
- Implementation: chunked R/S averaging across all non-overlapping windows per lag (corrected from single-chunk bug), manual OLS slope in numba to avoid `np.polyfit`
- Rolling Hurst: 252-bar window, step=21

**D. ADF + KPSS stationarity**
- Applied to price levels (testing non-stationarity) and log returns (confirming stationarity)

### Usage

```bash
# Single symbol
uv run python research/spy_ts_analysis.py --symbol SPY_D
uv run python research/spy_ts_analysis.py --symbol GBP_USD_H1 --period 120

# Batch (all *_D.parquet)
uv run python research/ts_analysis_batch.py --workers 4 --min-bars 504
```

### Key findings (daily universe, ~526 symbols)

| Metric | Value |
|---|---|
| Mean Hurst H | 0.505 |
| Median Hurst H | 0.503 |
| Trending (H > 0.55) | ~2% of symbols |
| Mean-reverting (H < 0.45) | ~0.2% of symbols |
| Random walk (0.45–0.55) | ~98% of symbols |
| Lag-1 ACF mean | −0.025 (slight mean-reversion) |
| Mean-reverting by ACF | ~40% of symbols |

**Implication:** Almost all daily equity log returns are statistical random walks by Hurst. The slight negative lag-1 ACF (~−0.025) suggests mild mean-reversion at the daily frequency — consistent with market microstructure noise. Momentum strategies require at least weekly holding periods to escape the mean-reversion zone.

**Notable exceptions:**
- DAX (`^GDAXI_D`): H = 0.557 — most trending index
- Several commodity/sector ETFs show H > 0.55

---

## Part 2 — Correlation Analysis

### Script

`research/correlation_analysis/run_correlation.py`

### Method

1. Load log returns for each symbol, strip timezone, normalize daily timestamps to midnight (to align different session close times — equities use 05:00 UTC, FX uses 22:00 UTC)
2. Inner-join on common dates; report coverage loss per symbol
3. Pearson + Spearman correlation matrices
4. Ward hierarchical clustering on distance matrix `1 − r`
5. Pair classification: positive (r > threshold), negative (r < −threshold), uncorrelated
6. Outputs: heatmap PNG, dendrogram PNG, full matrix CSV, pairs CSV

### Usage

```bash
# Default 18-symbol curated set (daily)
uv run python research/correlation_analysis/run_correlation.py

# Full H1 universe (16 symbols — FTSE/DAX H1 unavailable)
uv run python research/correlation_analysis/run_correlation.py \
  --symbols SPY_H1 QQQ_H1 UNH_H1 AMAT_H1 TXN_H1 INTC_H1 CAT_H1 WMT_H1 TMO_H1 GLD_H1 \
  EUR_USD_H1 GBP_USD_H1 USD_JPY_H1 AUD_USD_H1 USD_CHF_H1 AUD_JPY_H1 \
  --tag h1_full

# FX M5 spread analysis
uv run python research/correlation_analysis/run_correlation.py \
  --symbols EUR_USD_M5 USD_CHF_M5 EUR_CHF_M5 --min-bars 100 --tag fx_m5

# All daily symbols
uv run python research/correlation_analysis/run_correlation.py --all

# Custom threshold / min history
uv run python research/correlation_analysis/run_correlation.py --threshold 0.4 --min-bars 504
```

### Key findings

#### Cluster structure (H1, 16 symbols, 2018–2024)

| Cluster | Assets | Interpretation |
|---|---|---|
| Tech/Semi core | SPY, QQQ, AMAT, TXN, INTC | High-beta US growth |
| Defensive equities | UNH, CAT, TMO | Lower-beta, idiosyncratic |
| AUD bloc | AUD/USD, AUD/JPY | Risk-on carry |
| USD safe-havens | USD/JPY, USD/CHF | Risk-off |
| EUR/GBP bloc | EUR/USD, GBP/USD | USD-negative crosses |
| Gold + EUR | GLD, EUR/USD | USD-negative macro |
| Singleton | WMT | Most idiosyncratic equity |

#### Strongest pair correlations (H1)

| Pair | Pearson r | Interpretation |
|---|---|---|
| SPY ↔ QQQ | +0.918 | Near-identical exposure |
| AUD/USD ↔ AUD/JPY | +0.764 | Shared AUD leg |
| GBP/USD ↔ AUD/USD | +0.675 | Both USD-negative |
| EUR/USD ↔ USD/CHF | **−0.700** (H1) / **−0.692** (M5) | Shared CHF leg (negative) |
| GBP/USD ↔ USD/CHF | −0.589 | USD-negative vs USD-positive |

#### EUR/USD ↔ USD/CHF spread analysis (M5, 2024–2026)

The negative correlation holds at **−0.692 on M5** — consistent across timeframes (M5, H1, Daily). The synthetic spread `log(EUR/USD) + log(USD/CHF)` = `log(EUR/CHF)`, making this a triangular relationship rather than a pure stat arb. The spread is too efficiently arbed at M5 by market makers. More promising: use **correlation regime breakdown** (rolling r rising above −0.5) as a macro dislocation signal.

#### Portfolio construction implications

- SPY + QQQ in the same portfolio = nearly identical exposure — pick one or halve sizes
- ORB instruments (7 equities) cluster tightly at H1 (r ≈ 0.5–0.7): when multiple fire simultaneously, effective position count is ~2–3, not 7
- GLD is uncorrelated with equities at bar level but correlated in drawdowns — useful as a hedge

---

## Part 3 — Data Downloads Added

### New download scripts

| Script | Purpose |
|---|---|
| `scripts/download_fx_m5.py` | FX 5-min bars from IBKR (6-month chunks, any FX pair) |
| `scripts/download_index_h1.py` | Foreign index H1 bars from IBKR (IND contract type) |

### New data files

| File | Bars | Range | Source |
|---|---|---|---|
| `data/QQQ_H1.parquet` | 30,990 | 2018–2026 | Databento XNAS.ITCH |
| `data/GLD_H1.parquet` | 13,283 | 2018–2026 | Databento XNYS.PILLAR |
| `data/EUR_USD_M5.parquet` | 266,920 | 2005–2026 | IBKR (merged with old data) |
| `data/USD_CHF_M5.parquet` | 146,943 | 2024–2026 | IBKR |
| `data/EUR_CHF_M5.parquet` | 146,943 | 2024–2026 | IBKR |

### Notes on data availability

- **FTSE / DAX H1:** Not available via yfinance (indices blocked at intraday) or IBKR (requires market data subscription). Use daily data only.
- **FX M5:** IBKR caps 5-min bars at 6 months per request. For data older than ~2 years, use Dukascopy or similar.
- **yfinance H1:** Capped at last 730 days. Must pass `--start` within that window.

---

## Next Research Steps

1. **Cointegration analysis** (`research/cointegration_analysis/`) — Engle-Granger + Johansen tests, hedge ratio estimation, half-life calculation for candidate pairs
2. **Gold vs miners spread** — Download GDX/GDXJ H1, test GLD ↔ GDX cointegration (economic relationship, not mechanical)
3. **Rolling correlation regime signal** — Flag when EUR/USD ↔ USD/CHF rolling r breaks above −0.5 as a macro regime filter for equity entries
4. **Correlation-adjusted ORB position sizing** — Scale each instrument's size by `1 − avg_corr(instrument, active_positions)`
