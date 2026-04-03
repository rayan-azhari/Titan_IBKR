# Titan-IBKR Workspace Structure

> Last updated: 2026-04-03

This document outlines the file organization of the Titan-IBKR project.

## Root Directory

| Directory/File | Description |
|---|---|
| **`titan/`** | **Core Package**. Strategies, risk management, indicators, adapters. |
| **`research/`** | **Research Lab**. VectorBT sweeps, ML pipeline, IC analysis, mean reversion. |
| **`scripts/`** | **Entry Points**. Live runners, data downloaders, utilities. |
| **`config/`** | **Configuration**. TOML files for strategy parameters and risk. |
| **`data/`** | **Data Store**. Historical market data in Parquet format (~50 instruments). |
| **`models/`** | **ML Artifacts**. Trained .joblib models (HMM, XGBoost, RandomForest). |
| **`directives/`** | **SOPs**. 23 standard operating procedures and strategy specs. |
| **`resources/`** | **Reference**. Strategy research papers, framework guides, PDFs. |
| **`tests/`** | **Test Suite**. Unit and integration tests. |
| **`.tmp/`** | **Transient**. Reports, equity curves, sweep results (not committed). |

---

## Detailed Structure

### 1. `titan/` (The Engine)
*Library code only. No executable scripts. Fully type-hinted.*

```text
titan/
  adapters/ibkr/           IBKR adapter bridge
  costs/swap_curve.py      Historical swap/carry costs
  data/validation.py       Data integrity checks
  indicators/
    gaussian_filter.py     Numba Gaussian channel (used by ORB)
  models/spread.py         Session-aware spread model
  risk/
    portfolio_risk_manager.py   Shared singleton (all strategies)
  strategies/
    ic_equity_daily/       IC mean-reversion (7 US equities, daily)
    ic_generic/            Asset-agnostic IC strategy
    ic_mtf/                IC multi-timeframe FX
    mtf/                   MTF Confluence (EUR/USD, 4 timeframes)
    orb/                   Opening Range Breakout (7 equities, 5M)
    etf_trend/             ETF Trend (7 ETFs, daily)
    mr_fx/                 Mean Reversion FX (EUR/USD, M5)
    ml/                    ML Signal Classifier (daily)
    turtle/                Turtle Trading (Donchian+ATR)
  utils/notification.py    Slack alerting
```

### 2. `research/` (The Lab)
*Experimental code. Output feeds into config/ and models/.*

```text
research/
  ic_analysis/             IC signal discovery (Phases 0-5, 52 signals)
  etf_trend/               ETF Trend pipeline (Stages 1-4, 7 instruments)
  mean_reversion/          VWAP MR pipeline (Stages 1-4, EUR/USD)
  ml/                      ML classifier (28 instruments, 79 features)
  ml_regime/               HMM regime + XGBoost meta-labeling
  mtf/                     MTF confluence optimization (6-stage)
  orb/                     ORB backtesting and optimization
  intraday_profiles/       EUR/USD M5 pattern EDA
  alpha_loop/              Feature selection experiments
  turtle/                  Turtle universe scan (136 instruments)
  _archive/                Archived exploratory code (portfolio, correlation)
```

### 3. `scripts/` (The Control Panel)
*Executable entry points.*

```text
scripts/
  run_live_mtf.py          Deploy MTF Confluence
  run_live_etf_trend.py    Deploy ETF Trend
  run_live_orb.py          Deploy ORB Breakout
  run_live_ml.py           Deploy ML Classifier
  run_live_ic_mtf.py       Deploy IC Multi-Timeframe
  download_data*.py        Data downloaders (IBKR, Databento, yfinance)
  kill_switch.py           Emergency flatten all
  close_all_positions.py   Manual position close
  check_*.py               Environment verification
  test_*.py                Order/bracket testing
```

### 4. `config/` (The Controls)

| File | Strategy |
|---|---|
| `risk.toml` | Portfolio risk manager (kill switch, scaling, correlation) |
| `orb_live.toml` | ORB strategy (7 equities, per-ticker params) |
| `mtf_eurusd.toml` | MTF Confluence EUR/USD (Round 4 locked) |
| `mr_fx_eurusd.toml` | Mean Reversion FX EUR/USD (VWAP + grid) |
| `etf_trend_{spy,qqq,iwb,tqqq,efa,gld,dbc}.toml` | ETF Trend (7 instruments) |
| `mtf.toml` | MTF research defaults |
| `eurusd_mr.toml` | MR research defaults |
| `turtle_h1.toml` | Turtle strategy params |

### 5. `directives/` (The Rules)

| Category | Files |
|----------|-------|
| Master reference | System Status and Roadmap.md |
| Process | Backtesting & Validation.md, IC Signal Analysis.md, Strategy Deployment Protocol.md |
| Strategy specs | ETF Trend Strategy.md, Multi-Timeframe Confluence.md, ORB Trading Strategy.md |
| Operations | MTF Strategy User Guide.md, ORB Strategy User Guide.md, Deployment & Operations.md |
| Infrastructure | IBKR & NautilusTrader API Reference.md, Titan-IBKR Adapter Guide.md, Titan Library Reference.md |
| Research | ML Strategy Discovery.md, Turtle Trading Strategy Analysis.md, Time Series & Correlation Analysis.md |

See `directives/System Status and Roadmap.md` for the full system status and implementation roadmap.
