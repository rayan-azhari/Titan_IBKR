# Titan-IBKR Workspace Structure

This document outlines the file organization of the Titan-IBKR project.

## 📦 Root Directory

| Directory/File | Description |
|---|---|
| **`titan/`** | **Core Package**. Contains all reusable logic, models, and adapters. |
| **`research/`** | **Research Lab**. Experimental code, backtesting, and ML pipelines. |
| **`scripts/`** | **Entry Points**. User-facing scripts for running the system. |
| **`config/`** | **Configuration**. TOML files for strategy parameters and risk. |
| **`data/`** | **Data Store**. Historical market data in Parquet format. |
| **`tests/`** | **Test Suite**. Unit and integration tests. |
| `README.md` | Project overview and quick start guide. |
| `USER_GUIDE.md` | Detailed manual for operators. |

---

## 🏗️ Detailed Structure

### 1. `titan/` (The Engine)
*Library code only. No executable scripts.*

```text
titan/
├── adapters/
│   └── ibkr/          # NautilusTrader IBKR Adapter
├── config/             # Config loading utilities
├── data/
│   ├── ibkr.py        # IBKR API fetching logic
│   └── validation.py   # Data integrity checks
├── indicators/         # High-performance indicators (Numba)
├── models/             # Quant models (Spread, Slippage)
├── strategies/         # Production-ready strategies
│   ├── mtf/            # Multi-Timeframe Confluence
│   └── ml/             # Machine Learning execution
└── utils/              # Logging and notifications
```

### 2. `research/` (The Lab)
*Experimental code. Output feeds into config/ or titan/ models.*

```text
research/
├── alpha_loop/         # VectorBT optimization loop
├── gaussian/           # Gaussian Channel research
├── ml/                 # ML training pipeline & Feature selection
└── mtf/                # MTF strategy optimization
    └── legacy/         # Archived 5m stage variants (superseded by run_pair_sweep.py)
```

### 3. `scripts/` ( The Control Panel)
*Executable scripts to run the system.*

```text
scripts/
├── download_data.py        # Fetch history
├── download_sp100.py       # Download S&P 100 symbols
├── check_env.py            # Verify environment
├── run_backtest_mtf.py     # Run MTF backtest
├── run_live_mtf.py         # Deploy MTF strategy Live
├── run_live_ml.py          # Deploy ML strategy Live
├── list_instruments.py     # List available pairs
├── spread_analysis.py      # Spread & Cost Analysis
├── validate_data.py        # Data Integrity Check
├── kill_switch.py          # Emergency Stop
├── inspect_factory.py      # NautilusTrader order factory inspector
└── verify_titan_install.py # Installation Check
```

### 4. `config/` (The Controls)
*Parameterizing the system.*

| File | Purpose |
|---|---|
| `instruments.toml` | Pairs to trade and download. |
| `risk.toml` | Position sizing and drawdown limits. |
| `mtf.toml` | Parameters for the MTF strategy. |
| `mtf_eurusd.toml`, `mtf_gbpusd.toml`, etc. | Per-pair MTF locked configs (6 forex pairs). |
| `orb_live.toml` | ORB strategy parameters (7 equities). |

Inactive / historical configs are archived in `config/legacy/`.
