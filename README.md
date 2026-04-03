# Titan-IBKR-Algo

> A quantitative trading system for Interactive Brokers -- NautilusTrader execution, VectorBT research, ML signal discovery, and portfolio-level risk management.

---

## Live Strategies (5 deployed)

| Strategy | Instruments | Timeframe | OOS Sharpe | Runner |
|---|---|---|---|---|
| **IC Equity Daily** | HWM, CB, SYK, NOC, WMT, ABNB, GL | D | +2.65 to +4.28 | `scripts/run_live_ic_mtf.py` |
| **MTF Confluence** | EUR/USD | H1/H4/D/W | +1.94 | `scripts/run_live_mtf.py` |
| **ORB** | UNH, AMAT, TXN, INTC, CAT, WMT, TMO | 5M | Validated | `scripts/run_live_orb.py` |
| **ETF Trend** | SPY, QQQ, IWB, TQQQ, EFA, GLD, DBC | D | +0.89 to +1.37 | `scripts/run_live_etf_trend.py` |
| **MR FX** | EUR/USD | M5 | Research validated | -- |

All strategies are wired to a shared `PortfolioRiskManager` (drawdown scaling + kill switch).

**ML Signal Discovery** (beta): 28 instruments scanned. Tier A: EUR/USD D (+1.58 Sharpe), QQQ D (+1.11). PSKY D (+0.55, 13 folds) next for deployment.

---

## Architecture

```
directives/    <- 18 SOPs (source of truth for all strategy/deployment rules)
titan/         <- Core library: 9 strategies, risk management, indicators, adapters
research/      <- Experiments: VectorBT sweeps, ML pipeline, IC analysis, mean reversion
scripts/       <- CLI entry points: runners, watchdog, kill switch, data download
config/        <- TOML parameters (written by research, read by titan/)
data/          <- Historical Parquet files (OHLCV, ~50 instruments)
models/        <- Trained ML artifacts (.joblib)
.tmp/          <- Logs and transient reports
```

Code flows one direction: `research/` discovers -> `config/` captures -> `titan/` implements -> `scripts/` executes.

---

## Quick Start

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure credentials
```bash
cp .env.example .env   # then edit with your IBKR account details
```

Key `.env` variables:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # 4002=paper gateway, 4001=live gateway
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### 3. Verify connection
```bash
uv run python scripts/verify_connection.py
```

### 4. Download data
```bash
uv run python scripts/download_data.py
```

### 5. Run a strategy (paper)

Use the **watchdog** for production (handles IB nightly restarts):
```bash
uv run python scripts/watchdog_mtf.py
```

Or run strategies directly:
```bash
uv run python scripts/run_live_mtf.py          # MTF EUR/USD
uv run python scripts/run_live_orb.py           # ORB equities
uv run python scripts/run_live_etf_trend.py     # ETF Trend
uv run python scripts/run_live_ml.py            # ML Classifier
```

See `directives/Deployment & Operations.md` for full deployment guide (VPS, Docker, systemd).

---

## Emergency Stop

```bash
uv run python scripts/kill_switch.py
```

Cancels all orders and closes all positions immediately via IBKR API. Does not require the strategy process to be running.

---

## Research Workflow

```bash
# VectorBT parameter sweep
uv run python research/alpha_loop/run_vbt_optimisation.py

# ML signal discovery (28 instruments)
uv run python research/ml/run_52signal_classifier.py

# Full ML evaluation
uv run python research/ml/run_ml_full_eval.py --instrument QQQ

# MTF confluence optimisation
uv run python research/mtf/run_optimisation.py
```

---

## Pre-Push Checklist

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest tests/ -v
```

All three must pass before pushing.

---

## Key Rules

- **`uv` only** -- no bare `pip` installs
- **`decimal.Decimal`** for all financial types (or NautilusTrader `Price`/`Quantity`)
- **`random_state=42`** on every ML training call
- **No look-ahead bias** -- features lagged, targets future-derived
- **Factory methods** for NautilusTrader objects (`Price.from_str()`, not constructors)
- **Portfolio risk** -- all strategies must register with `PortfolioRiskManager`

Full rules: `directives/Titan Library Reference.md`

---

## Documentation

All operational knowledge lives in `directives/` (18 SOPs). Key files:

| Topic | Directive |
|---|---|
| System overview | `System Status and Roadmap.md` |
| Deployment | `Deployment & Operations.md` |
| ORB strategy | `ORB Trading Strategy.md` + `ORB Strategy User Guide.md` |
| MTF strategy | `Multi-Timeframe Confluence.md` + `MTF Strategy User Guide.md` |
| ETF Trend | `ETF Trend Strategy.md` |
| ML pipeline | `Machine Learning Strategy Discovery.md` |
| IC signals | `IC Signal Analysis.md` |
| Backtesting | `Backtesting & Validation.md` |
| API reference | `Titan Library Reference.md` + `IBKR & NautilusTrader API Reference.md` |
| Adapter guide | `Titan-IBKR Adapter Guide.md` |
