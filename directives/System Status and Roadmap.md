# Titan-IBKR System Status and Roadmap

**Last updated:** 2026-04-03
**Author:** Architect (Claude Code)
**Status:** Active development, 5 live strategies, 3 in validation, portfolio risk layer deployed

---

## 1. System Overview

Titan-IBKR is a NautilusTrader-based quantitative trading system connected to Interactive Brokers. It combines multiple independent strategy types (trend-following, mean-reversion, ML classifiers, intraday breakout) into a portfolio managed by a shared risk layer.

### Architecture

```
directives/     18 SOPs (source of truth for rules and processes)
titan/          Core library: strategies, adapters, indicators, risk management
research/       Experiments: VectorBT sweeps, ML pipeline, mean reversion, IC analysis
scripts/        CLI entry points: runners, kill switch, data download
config/         TOML parameters: written by research pipelines, read by titan/
data/           Historical Parquet files (OHLCV, auto-downloaded)
models/         Trained ML artifacts (.joblib)
.tmp/           Logs and transient reports
```

Code flows one direction: `research/` discovers -> `config/` captures -> `titan/` implements -> `scripts/` executes.

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Execution engine | NautilusTrader (event-driven, Rust core) |
| Broker | Interactive Brokers (TWS API via IB Gateway) |
| Research | VectorBT Pro (vectorized backtesting) |
| ML | XGBoost, scikit-learn, hmmlearn |
| Data | Yahoo Finance (daily), Databento (tick), IBKR (live) |
| Language | Python 3.11, fully type-hinted in titan/ |
| Package manager | uv (dependency management) |
| Linting | Ruff (E, F, I, W rules) |

---

## 2. Live Strategies (5 deployed)

### 2.1 IC Equity Daily (Mean-Reversion, Long-Only)

**Status:** LIVE | **Validated:** 2026-03-20

| Property | Value |
|----------|-------|
| Signal | RSI(21) deviation from 50 (negative IC = buy oversold dips) |
| Instruments | HWM, CB, SYK, NOC, WMT, ABNB, GL (7 of 482 screened) |
| Timeframe | Daily |
| OOS Sharpe | +2.65 to +3.41 per symbol |
| Win rate | 75-89% |
| Sizing | ATR-based, 0.5% equity risk per trade |
| Portfolio integration | PortfolioRiskManager wired (April 2026) |
| Config | Inline per-symbol parameters |
| Files | `titan/strategies/ic_equity_daily/strategy.py` |

**April 2026 improvements applied:**
- Registered with `PortfolioRiskManager` singleton
- Equity updates on every daily bar, halt check before evaluation
- Position sizes multiplied by `portfolio_risk_manager.scale_factor`

**Planned improvements (Tier 2):**
- Regime-conditional sizing (VIX tiers: <17.8 full, 17.8-23.1 at 75%, >23.1 at 50%)
- 3-tier scale-in logic (1/3 at z>0.75, 1/3 at z>1.0, 1/3 at z>1.5)
- Monthly rotation: re-rank by rolling 60-day Sharpe, activate top-5 only

---

### 2.2 MTF Confluence (FX Trend-Following)

**Status:** LIVE | **Validated:** Round 4, 2026-03

| Property | Value |
|----------|-------|
| Signal | Weighted MA crossover + RSI across 4 timeframes (H1/H4/D/W) |
| Instruments | EUR/USD |
| Timeframe | Multi (H1 primary, H4/D/W context) |
| OOS Sharpe | +1.94 (long+short combined) |
| Sizing | 1% equity risk / (ATR * stop_mult) |
| Portfolio integration | PortfolioRiskManager wired (April 2026) |
| Config | `config/mtf_eurusd.toml` |
| Files | `titan/strategies/mtf/strategy.py` |

**April 2026 improvements applied:**
- **Continuous forecasts**: Replaced binary +0.5/-0.5 per component with continuous normalized values. MA component uses tanh-capped normalized spread; RSI uses linear (rsi-50)/100. Total per-TF signal now ranges [-1.0, +1.0] with conviction proportional to signal strength.
- **Position inertia**: Rebalancing suppressed when score delta < 10% and direction unchanged (`position_inertia_pct = 0.10` in TOML). Reduces FX turnover costs.
- **Conviction-scaled sizing**: `risk_amount = equity * risk_pct * conviction` where conviction = abs(score), clipped [0.3, 1.0]. Stronger signals get larger positions.
- PortfolioRiskManager wired (register, update, halt check, scale_factor)

---

### 2.3 ORB (Intraday Equity Breakout)

**Status:** LIVE | **Validated:** Per-ticker optimization

| Property | Value |
|----------|-------|
| Signal | 5-minute opening range high/low breakout with daily filters |
| Instruments | UNH, AMAT, TXN, INTC, CAT, WMT, TMO (7 equities) |
| Timeframe | 5M entry, 1D filters (SMA50, RSI14, optional Gaussian channel) |
| Entry | Opening range breakout with bracket orders (TP/SL via OCO) |
| Exit | Bracket hit or flatten at 15:50 ET |
| Sizing | 1% equity risk, ATR-based |
| Portfolio integration | PortfolioRiskManager wired (April 2026) |
| Config | `config/orb_live.toml` (per-ticker) |
| Files | `titan/strategies/orb/strategy.py` |

**April 2026 improvements applied:**
- PortfolioRiskManager wired (register, update on daily bars, halt check, scale_factor on sizing)

---

### 2.4 ETF Trend (Daily Trend-Following, Long-Only)

**Status:** LIVE | **Validated:** 4-stage pipeline

| Property | Value |
|----------|-------|
| Signal | Close > slow MA entry, MA break exit, optional decel composite |
| Instruments | SPY, QQQ, IWB, TQQQ, **EFA**, **GLD**, **DBC** (expanded April 2026) |
| Timeframe | Daily |
| Execution | MOC orders at 15:30 ET next day (mirrors backtest shift-1) |
| Sizing | Binary (100% when in trade) or vol-target or dynamic decel |
| Portfolio integration | PortfolioRiskManager wired (April 2026) |
| Configs | `config/etf_trend_{spy,qqq,iwb,tqqq,efa,gld,dbc}.toml` |
| Files | `titan/strategies/etf_trend/strategy.py` |

**April 2026 expansion:**
Downloaded daily data for EFA, EEM, TLT, IEF, DBC via Yahoo Finance. Ran full 4-stage optimization pipeline on each:

| Instrument | Category | Stage 1 | OOS Sharpe | OOS Return | MaxDD | Config |
|------------|----------|---------|-----------|-----------|-------|--------|
| SPY | US large-cap | (existing) | -- | -- | -- | Yes |
| QQQ | US tech | (existing) | -- | -- | -- | Yes |
| IWB | US broad | (existing) | -- | -- | -- | Yes |
| TQQQ | US tech 3x | (existing) | -- | -- | -- | Yes |
| **EFA** | International eq | 9/20 pass | **+0.89** | +87.3% | -14.6% | **Yes** |
| **GLD** | Gold | 0/20* | **+1.35** | +184.3% | -14.2% | **Yes** |
| **DBC** | Commodities | 1/20 pass | **+1.37** | +187.0% | -19.3% | **Yes** |
| EEM | Emerging mkts | 0/20 | +0.41 | +29.9% | -27.4% | No (marginal) |
| TLT | Long bonds | 0/20 | -0.18 | -14.2% | -41.4% | No (FAIL) |
| IEF | Int. bonds | 0/20 | +0.31 | +8.1% | -16.6% | No (marginal) |

*GLD failed formal Stage 1 gates but excelled in Stages 2-4.

**Key findings from expansion:**
- Bonds (TLT, IEF) don't work with trend-following (negative OOS Sharpe)
- Gold (GLD) works very well with SMA(250) -- Sharpe +1.35
- DBC (commodities) is the surprise winner -- Sharpe +1.37 with decel signals
- EFA adds international equity diversification -- Sharpe +0.89

---

### 2.5 MR FX (Mean-Reversion, Intraday) -- NEW

**Status:** DEPLOYED (April 2026) | **Validated:** Research pipeline complete (Stages 1-3)

| Property | Value |
|----------|-------|
| Signal | Session-anchored VWAP deviation with 4-tier exponential grid |
| Instruments | EUR/USD |
| Timeframe | M5 (5-minute bars) |
| Session | London core (07:00-12:00 UTC entries only) |
| Regime gate | ATR percentile < 30th over 500-bar window (low-vol = ranging) |
| Entry | 4 tiers at 90th, 95th, 98th, 99th percentile of deviation |
| Tier sizes | [1, 2, 4, 8] (exponential -- deeper tiers carry more weight) |
| Exit | Partial reversion TP (50% back to VWAP) + hard invalidation (99.99th pct) + NY close (21:00 UTC) |
| Sizing | 1% equity risk, ATR-based, per-tier weighted |
| Portfolio integration | PortfolioRiskManager wired |
| Config | `config/mr_fx_eurusd.toml` |
| Files | `titan/strategies/mr_fx/strategy.py` |

**Architecture:**
- Subscribes to M5 bars, warm-up from `data/EUR_USD_M5.parquet`
- Maintains rolling VWAP (resets at London 07:00, NY 13:00 UTC)
- Tracks deviation percentile distribution over 2000-bar rolling window
- Basket tracking per direction (long/short separately)
- Integrated with PortfolioRiskManager

**Research pipeline:** `research/mean_reversion/` (signals.py, regime.py, risk.py, execution.py, state_manager.py, run_stage1-4.py)

---

## 3. ML Signal Discovery Pipeline

### 3.1 Architecture

```
Data (OHLCV) -> [Feature Builder] -> 79 features -> [Regime+Pullback Labeler] -> sparse labels
  -> [XGBClassifier] -> P(long) per bar -> [Position Manager] -> held position
  -> Strategy Returns (position * bar_returns - costs)
```

**Features:** 52 IC signals (Groups A-G) + 7 MA + 5 regime (ADX, HMM, ATR pctile) + 3 VIX + 1 calendar + 4 momentum + 7 stochastic

**Labeling (v3):** SMA(50/200) + MACD regime detection -> RSI pullback within regime -> forward return confirmation. Per-fold parameter sweep on IS data.

**Model:** XGBClassifier, depth=4, lr=0.03, 300 trees, scale_pos_weight auto-computed.

**Validation:** Walk-Forward (rolling IS=2yr, OOS=6mo), 5 quality gates.

### 3.2 Cross-Asset Signal Map (28 instruments x 3 timeframes)

#### Tier A -- Deploy candidates (Sharpe >= 1.0)

| Instrument | TF | Category | Sharpe | %Pos | Folds |
|------------|:--:|----------|-------:|-----:|------:|
| EUR_USD | D | FX | +1.584 | 100% | 3 |
| QQQ | D | Index | +1.106 | 57% | 7 |

#### Tier B -- Strong signals (Sharpe >= 0.5)

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

#### Key findings
- Daily is the only viable timeframe (H1 and H4 all negative)
- 4 uncorrelated buckets: US Equity (QQQ best), FX (EUR_USD best), Miners (PSKY most robust), Gold (no signal)
- QQQ/SPY/ES/IWB/TQQQ are ~0.95 correlated -- treat as ONE allocation bucket
- PSKY is the most statistically robust signal (13 folds, 69% positive)
- Gold (GLD, IAU, GC=F) and silver futures (SI=F) produce zero ML signal

### 3.3 Comprehensive Evaluation Stats

| Metric | QQQ D | EUR_USD D | SPY D |
|--------|------:|----------:|------:|
| Total Return | +109.2% | +19.3% | +20.1% |
| CAGR | +23.5% | +12.5% | +9.6% |
| Sharpe | +1.11 | +1.58 | +0.56 |
| Sortino | +1.40 | +2.36 | +0.71 |
| Calmar | 1.24 | 3.12 | 0.47 |
| Max Drawdown | -19.0% | -4.0% | -20.5% |
| Trades | 12 | 3 | 4 |
| Win Rate | 75% | 67% | 75% |
| Profit Factor | 7.71 | 9.04 | 2.87 |
| Avg Capital Invested | 100% | 94% | 100% |
| Risk of Ruin (50%) | 0.6% | 0.0% | 0.5% |

### 3.4 Bugs Fixed (April 2026 audit)

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | CRITICAL | train_model.py | Sharpe used backward returns -> now uses forward returns |
| 2 | CRITICAL | run_ensemble.py | 3-class signal mapping inverted -> now handles both binary and 3-class |
| 3 | CRITICAL | run_pipeline.py | Saved model IS-only trained -> retrained on full data + performance gate |
| 4 | HIGH | run_52signal_classifier.py | HMM trained on 70% of all data -> now uses first IS window only |
| 5 | HIGH | run_52signal_classifier.py | VIX same-day lookahead -> shifted by 1 day |
| 6 | HIGH | run_metalabeling.py | Purged K-Fold trained on future -> walk-forward expanding splits |
| 7 | HIGH | run_metalabeling.py | Forward returns on gapped bars -> computed on full series first |
| 8 | MEDIUM | build_tbm_labels.py | Same-bar tiebreaker always favored SL -> now labels as inconclusive |
| 9 | MEDIUM | run_pipeline.py | Sharpe annualization wrong -> per-timeframe bars_per_year |
| 10 | MEDIUM | run_52signal_classifier.py | SIGNAL_THRESHOLD 0.5 (no dead zone) -> 0.6 (matching docs) |
| 11 | MEDIUM | run_52signal_classifier.py | NaN-to-zero bias -> XGBoost native NaN handling |
| 12 | MEDIUM | build_features.py | PROJECT_ROOT wrong + SMA RSI -> Wilder's smoothing |
| 13 | MEDIUM | run_metalabeling.py | SMA RSI -> Wilder's smoothing |

---

## 4. Portfolio Risk Management

### 4.1 PortfolioRiskManager (deployed April 2026)

**File:** `titan/risk/portfolio_risk_manager.py`

Module-level singleton shared by all live strategies. Auto-loads config from `config/risk.toml [portfolio]`.

**Integration pattern (all 4 live strategies wired):**
1. `on_start()`: `portfolio_risk_manager.register_strategy(id, initial_equity)`
2. `on_bar()`: `portfolio_risk_manager.update(id, current_equity)` + halt check
3. Sizing: `computed_size * portfolio_risk_manager.scale_factor`

**Risk controls:**

| Trigger | Action | Reset |
|---------|--------|-------|
| Portfolio DD > 10% | Linear scale-down from 1.0 to 0.25 floor | Auto when DD recovers |
| Portfolio DD > 15% | Kill switch: `halt_all = True`, flatten everything | Manual `reset_halt()` |
| Strategy pair r > 0.85 | WARNING log (informational) | N/A |
| Single strategy > 60% of portfolio | WARNING log | N/A |

**Config:** `config/risk.toml`
```toml
[portfolio]
portfolio_max_dd_pct       = 15.0
portfolio_heat_scale_pct   = 10.0
correlation_window_days    = 60
correlation_halt_threshold = 0.85
portfolio_max_single_pct   = 60.0
```

### 4.2 Backtest Limitations (documented)

The current ML backtest is a **signal quality test, not a portfolio simulation**:
- Each instrument assumes 100% of capital in isolation
- No position sizing, no multi-asset allocation, no leverage modelling
- Returns are NOT achievable as-is -- they measure direction prediction quality
- Full documentation in `research/ml/ML_STRATEGY_DOCUMENTATION.md`

---

## 5. Research Pipelines

### 5.1 IC Analysis (Phases 0-6)

| Phase | Script | Purpose |
|-------|--------|---------|
| 0 | `phase0_regime.py` | ADX + HMM regime labelling |
| 1 | `phase1_sweep.py` | 52-signal IC/ICIR leaderboard |
| 2 | `phase2_combination.py` | Correlation matrix, composite building |
| 3 | `phase3_backtest.py` | Full-friction IS/OOS backtest |
| 4 | `phase4_wfo.py` | Rolling walk-forward validation |
| 5 | `phase5_robustness.py` | Monte Carlo + top-N removal + 3x slippage stress |

**Directive:** `directives/IC Signal Analysis.md` (v4.2)

### 5.2 ETF Trend (Stages 1-4)

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `run_optimisation.py` | MA type + slow period sweep |
| 2 | `run_stage2_decel.py` | Deceleration composite signals |
| 3 | `run_stage3_exits.py` | Exit logic variants (MA break, decel threshold, hybrid) |
| 4 | `run_stage4_sizing.py` | Position sizing (binary, vol-target, dynamic decel) |

**Directive:** `directives/ETF Trend Strategy.md`

### 5.3 Mean Reversion (Stages 1-4)

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `run_stage1_signals.py` | VWAP percentile tier entry signals |
| 2 | `run_stage2_regime.py` | ATR + Hurst regime gates |
| 3 | `run_stage3_full.py` | Full backtest with IS/OOS, Monte Carlo, slippage stress |
| 4 | `run_stage4_pairs.py` | Pairs trading (correlated legs) |

**Config:** `config/eurusd_mr.toml`, `config/mr_fx_eurusd.toml`

### 5.4 ML Signal Discovery

| Script | Purpose |
|--------|---------|
| `run_52signal_classifier.py` | Main pipeline: 28 instruments, WFO, regime+pullback labels |
| `run_ml_full_eval.py` | Comprehensive evaluation: equity curves, stats, risk of ruin, charts |
| `run_52sig_param_sweep.py` | Parameter sweep (MA, ATR, stop, threshold) |
| `plot_52sig_signals.py` | Interactive Plotly charts (price + signals + equity + drawdown) |
| `run_pipeline.py` | Alternate pipeline (3-class target, VBT backtest) |
| `run_metalabeling.py` | Meta-labeling (MTF + TBM + walk-forward CV) |
| `build_tbm_labels.py` | Triple Barrier Method labeler |
| `run_ensemble.py` | Multi-strategy ensemble framework |

**Documentation:** `research/ml/ML_STRATEGY_DOCUMENTATION.md`

### 5.5 Intraday Profiles (Exploratory)

| Script | Purpose |
|--------|---------|
| `explore_eurusd_2024.py` | EUR/USD M5 hourly profiles, day-of-week seasonality |
| `am_pm_signal.py` | AM/PM session correlation |
| `pca_archetypes.py` | PCA on intraday patterns |
| `seasonality.py` | Month/week-of-year effects |

**Status:** EDA only, no trading model. Outputs in `.tmp/reports/eda_eurusd/`.

---

## 6. Data Inventory

### 6.1 FX Pairs

| Instrument | D | H1 | H4 | M5 | W |
|------------|:-:|:--:|:--:|:--:|:-:|
| EUR_USD | 6,412 | 134,971 | 35,925 | 266,920 | 1,154 |
| GBP_USD | 2,592 | 62,053 | 18,102 | -- | 522 |
| USD_JPY | 2,592 | 62,074 | 18,106 | -- | 522 |
| AUD_JPY | 2,592 | 62,077 | 18,106 | -- | 522 |
| AUD_USD | 2,592 | 62,073 | 18,106 | -- | 522 |
| USD_CHF | 2,592 | 62,078 | 18,106 | 146,943 | 522 |

### 6.2 Indices

| Instrument | D | H1 | Notes |
|------------|:-:|:--:|-------|
| SPY | 8,338 | 26,642 | From 1993 |
| QQQ | 6,797 | 30,990 | From 1999 |
| IWB | 6,494 | -- | Broad market |
| TQQQ | 4,048 | -- | 3x leveraged |
| ^FTSE | 10,665 | -- | From 1984 |
| ^GDAXI | 9,665 | -- | From 1988 |
| ES | 6,592 | -- | S&P futures |
| DXY | 4,079 | -- | Dollar index |

### 6.3 Gold, Silver, Miners

| Instrument | D | H1 | Category |
|------------|:-:|:--:|----------|
| GLD | 5,366 | 13,283 | Gold ETF |
| IAU | 5,318 | -- | Gold ETF |
| GC=F | 6,411 | -- | Gold futures |
| SLV | 5,004 | -- | Silver ETF |
| SIVR | 4,189 | -- | Silver ETF |
| PSLV | 3,869 | -- | Silver |
| SI=F | 6,413 | -- | Silver futures |
| GDX | 4,988 | -- | Gold miners |
| GDXJ | 4,112 | -- | Jr gold miners |
| SIL | 4,004 | -- | Silver miners |
| PSKY | 5,103 | -- | Miners |

### 6.4 New (April 2026)

| Instrument | D | Source | Category |
|------------|:-:|--------|----------|
| EFA | 6,186 | Yahoo Finance | International equity ETF |
| EEM | 5,780 | Yahoo Finance | Emerging markets ETF |
| TLT | 5,958 | Yahoo Finance | Long-term bonds |
| IEF | 5,958 | Yahoo Finance | Intermediate bonds |
| DBC | 5,071 | Yahoo Finance | Commodities ETF |

---

## 7. Configuration Files

| Config | Strategy | Status |
|--------|----------|--------|
| `config/risk.toml` | Portfolio risk manager (all strategies) | Active |
| `config/orb_live.toml` | ORB (per-ticker params) | Active |
| `config/mtf_eurusd.toml` | MTF Confluence EUR/USD | Active |
| `config/mr_fx_eurusd.toml` | MR FX EUR/USD (VWAP) | Active (new) |
| `config/etf_trend_spy.toml` | ETF Trend SPY | Active |
| `config/etf_trend_qqq.toml` | ETF Trend QQQ | Active |
| `config/etf_trend_iwb.toml` | ETF Trend IWB | Active |
| `config/etf_trend_tqqq.toml` | ETF Trend TQQQ | Active |
| `config/etf_trend_efa.toml` | ETF Trend EFA | Active (new) |
| `config/etf_trend_gld.toml` | ETF Trend GLD | Active (new) |
| `config/etf_trend_dbc.toml` | ETF Trend DBC | Active (new) |
| `config/mtf.toml` | MTF research config | Research |
| `config/eurusd_mr.toml` | MR research config | Research |

---

## 8. What Was Done (April 2026 Session)

### 8.1 ML Pipeline Audit and Bug Fixes (13 issues fixed)

Comprehensive code review of all 9 files in `research/ml/`. Found and fixed 3 CRITICAL, 4 HIGH, and 6 MEDIUM severity bugs including:
- Sharpe computed on backward returns (train_model.py)
- Ensemble signal mapping inverted for 3-class models (run_ensemble.py)
- HMM lookahead in WFO (run_52signal_classifier.py)
- Purged K-Fold training on future data (run_metalabeling.py)
- NaN-to-zero substitution masking warmup artifacts (run_52signal_classifier.py)

All pipelines re-run after fixes. Previous results were overstated.

### 8.2 Full Universe ML Scan (28 instruments x 3 timeframes)

Expanded TARGET_INSTRUMENTS from 9 to 28 (6 FX, 7 indices, 3 gold, 4 silver, 4 miners, DXY, ES, DBC, GC=F, SI=F). Added H4 WFO config. Ran classifier on Daily, H1, H4.

Result: 2 Tier A + 8 Tier B signals, all on Daily timeframe. H1/H4 dead.

### 8.3 Comprehensive Evaluation Script

Built `research/ml/run_ml_full_eval.py` with:
- Full WFO with regime+pullback labels
- Comprehensive stats (CAGR, Sharpe, Sortino, Calmar, max DD, DD duration)
- Trade analysis (long/short breakdown, win rate, avg win/loss, profit factor)
- Risk of ruin (Monte Carlo, 5000 simulations, 3 threshold levels)
- Capital invested tracking (time-series + average)
- 5-panel interactive Plotly charts (price+signals, equity, drawdown, exposure, annual returns)

### 8.4 Strategy Innovation Plan (approved)

Reviewed 12 resource documents + codebase + web research to design:
- Part A: 4 strategy improvements (IC Equity, MTF, ORB, ML Pipeline)
- Part B: 5 new strategies (Gold Macro, EUR/USD VWAP MR, FX Carry, Session Gap Fade, Equity Pairs)
- Part C: Portfolio composition framework (3-layer sizing, risk parity, regime scaling, drawdown breakers, return stacking)
- Part D: Implementation priority (Tier 1-3, ranked by impact/effort)

### 8.5 Tier 1 Implementation (4 priorities, all delivered)

| # | Priority | What was done |
|---|----------|---------------|
| 1 | PortfolioRiskManager | Wired into all 4 live strategies (IC Equity, MTF, ORB, ETF Trend). Auto-loads risk.toml config. |
| 2 | EUR/USD VWAP MR | New strategy class (`titan/strategies/mr_fx/strategy.py`), new config, M5 bars, tiered grid, regime gate. |
| 3 | MTF improvements | Continuous forecasts, position inertia (10% buffer), conviction-scaled sizing. |
| 4 | ETF Trend expansion | Downloaded 5 new datasets, ran 4-stage pipeline, added EFA + GLD + DBC (7 total instruments). |

---

## 9. Tier 2 Implementation (delivered April 2026)

| # | Task | Status | What was done |
|---|------|--------|---------------|
| 5 | IC Equity regime sizing + scale-in | **DONE** | 3-tier scale-in (1/3 at threshold, +0.25, +0.75). ATR percentile fed to PortfolioRiskManager. Tiers reset on exit/flip. |
| 6 | ML FractionalKelly + HealthMonitor | **DONE** | Quarter-Kelly sizer (max 3%), WinLossTracker, ModelHealthMonitor (prob drift + calibration drift). Sizes halved when model degraded. PortfolioRiskManager wired in. |
| 7 | Gold Macro Strategy | **DEFERRED to Tier 3** | Needs TIP/UUP data + new research pipeline. |
| 8 | Portfolio vol-targeting overlay | **DONE** | EWMA(lambda=0.94) realized vol tracking. `vol_scale = target_vol(12%) / realized_vol`, clipped [0.25, 2.0]. Config in `risk.toml`. |
| 9 | VIX-based regime scaling | **DONE** | VIX 4-tier scaling + ATR percentile gate. `regime_scale = min(vix_scale, atr_scale)`. VIX is optional (fallback to ATR-only). Config in `risk.toml`. |

### Scale factor composition (new)

```
scale_factor = min(dd_scale, vol_scale, regime_scale)
```

- `dd_scale`: Linear [0.25, 1.0] from 10-15% portfolio DD (existing).
- `vol_scale`: `target_vol / EWMA_vol`, clipped [0.25, 2.0] (Tier 2 #8).
- `regime_scale`: `min(vix_scale, atr_scale)`, range [0.25, 1.25] (Tier 2 #9).

## 9.5 Tier 3 Implementation (delivered April 2026)

| # | Task | Status | What was done |
|---|------|--------|---------------|
| 10 | FX Carry Trade | **DONE** | `titan/strategies/fx_carry/` — AUD/JPY daily, long when carry+SMA confirm. Vol-targeted sizing (8%), VIX halving. Config: `config/fx_carry_audjpy.toml`. |
| 11 | Session Gap Fade | **DONE** | `titan/strategies/gap_fade/` — EUR/USD M5, fade overnight gaps > 1.5x ATR at London open. 50% gap fill TP, bracket orders, EOD hard close. AM/PM archetype filter (optional). Config: `config/gap_fade_eurusd.toml`. |
| 13 | PortfolioAllocator | **DONE** | `titan/risk/portfolio_allocator.py` — inverse-vol weighting, monthly rebalance (21 days), min 5%/max 60% constraints, correlation penalty (|r| > 0.70). Config: `config/risk.toml [allocation]`. |
| 7 | Gold Macro Strategy | **DONE** | `titan/strategies/gold_macro/` — 3-component cross-asset signal (real rates + dollar + momentum). OOS Sharpe +0.603, OOS/IS ratio 2.06. Vol-targeted sizing (10%), ATR stop. Research: `research/gold_macro/run_backtest.py`. Config: `config/gold_macro_gld.toml`. Data: TIP, TLT, DXY, GLD daily. |

| 12 | Equity Pairs Trading | **DONE** | `titan/strategies/pairs/` -- generic pairs framework. Scanned 26 instruments, selected GLD/EFA (p=0.01 cointegration). OOS Sharpe +1.14. Walk-forward beta refit every 126 bars. Note: INTC/TXN and QQQ/SPY FAILED cointegration in recent data -- rejected. Config: `config/pairs_gld_efa.toml`. |

## 9.6 What's Next (remaining)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 14 | Return stacking (T-bill collateral + futures overlay) | Medium | High |

---

## 11. Target Portfolio Composition

| Bucket | Strategy | Instrument(s) | Expected Alloc | Correlation |
|--------|----------|---------------|:-----------:|-------------|
| US Equity Trend | ETF Trend + ML QQQ | QQQ (or SPY) | 25-30% | -- |
| IC Mean-Rev | IC Equity Daily | UNH, TXN, WMT, etc. | 20-25% | Low (MR vs trend) |
| FX Trend | MTF Confluence + ML EUR_USD | EUR/USD | 15-20% | Low (different asset) |
| FX Mean-Rev | MR FX (VWAP) | EUR/USD M5 | 10-15% | Negative (MR vs trend) |
| Miners | ML PSKY | PSKY | 5-10% | Low (uncorrelated) |
| Gold/Commodity | ETF Trend GLD+DBC | GLD, DBC | 5-10% | Low (macro drivers) |
| Carry | FX Carry (future) | AUD/JPY | 5% | Near-zero (structural) |
| Intraday | ORB + Gap Fade (future) | SPY + EUR/USD | 5-10% | Near-zero (intraday) |

Expected portfolio Sharpe: 1.5-2.5 (diversification multiplier ~1.5x on individual Sharpe ~1.0)

---

## 12. Quality Gates

### Per-Strategy Deployment Gates

| Gate | Metric | Threshold |
|------|--------|-----------|
| Stitched Sharpe | OOS across all folds | > 1.0 |
| % Positive Folds | Folds with OOS Sharpe > 0 | >= 70% |
| Avg Parity | OOS/IS Sharpe ratio | >= 0.50 |
| Worst Fold | Min OOS Sharpe | > -2.0 |
| Consec Neg Folds | Max streak | <= 2 |

### Portfolio-Level Gates

| Gate | Metric | Threshold |
|------|--------|-----------|
| Combined Sharpe | All strategies combined | >= 1.5 |
| Sortino | Downside-adjusted | >= 2.0 |
| Max DD | Portfolio drawdown | < 15% |
| Avg Correlation | Pairwise strategy correlation | < 0.30 |

### Code Quality Gates (pre-push)

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest tests/ -v
```

---

## 13. Key Risks and Limitations

1. **Backtest != reality:** ML backtest assumes 100% capital in one instrument. Not a portfolio simulation.
2. **Low fold counts:** EUR_USD Daily has only 3 WFO folds. Statistical confidence is limited for short-history instruments.
3. **Regime dependence:** Trend strategies struggle in choppy sideways markets (by design).
4. **Gold/silver untradeable by ML:** Regime+pullback produces no signal. ETF Trend approach works for GLD but not for ML classifier.
5. **High equity correlation:** QQQ/SPY/ES/IWB/TQQQ are ~0.95 correlated. Must be treated as one allocation bucket.
6. **No portfolio-level backtesting yet:** Individual strategies validated, but combined portfolio performance not yet simulated.
7. **Carry costs not modelled:** Short positions incur borrow costs; leveraged positions incur margin interest.
8. **H1/H4 intraday timeframes dead for ML:** All produce negative Sharpe with current approach.

---

## 14. File Reference

### Core Strategies
- `titan/strategies/ic_equity_daily/strategy.py` -- IC mean-reversion
- `titan/strategies/mtf/strategy.py` -- MTF confluence
- `titan/strategies/orb/strategy.py` -- Opening range breakout
- `titan/strategies/etf_trend/strategy.py` -- ETF trend-following
- `titan/strategies/mr_fx/strategy.py` -- Mean reversion FX (NEW)
- `titan/strategies/ml/strategy.py` -- ML classifier (beta)

### Risk Management
- `titan/risk/portfolio_risk_manager.py` -- Portfolio-level singleton

### Research Pipelines
- `research/ml/run_52signal_classifier.py` -- Main ML pipeline
- `research/ml/run_ml_full_eval.py` -- Comprehensive evaluation
- `research/etf_trend/run_optimisation.py` through `run_stage4_sizing.py` -- ETF pipeline
- `research/mean_reversion/run_stage1_signals.py` through `run_stage4_pairs.py` -- MR pipeline
- `research/ic_analysis/phase0_regime.py` through `phase5_robustness.py` -- IC pipeline

### Documentation
- `directives/System Status and Roadmap.md` -- THIS FILE (master reference)
- `research/ml/ML_STRATEGY_DOCUMENTATION.md` -- ML strategy details
- `directives/IC Signal Analysis.md` -- IC pipeline methodology
- `directives/Backtesting & Validation.md` -- Validation framework
- `directives/ETF Trend Strategy.md` -- ETF strategy spec
