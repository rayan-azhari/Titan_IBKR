# Titan Library Reference

> Last updated: 2026-04-03

**Package Name:** `titan-ibkr-algo`
**Import Name:** `titan`
**Version:** 0.1.0

The `titan` package is the core library powering the Titan IBKR Algo system. It contains reusable
business logic, quantitative models, and infrastructure adapters, decoupled from execution scripts.

---

## Architecture Philosophy

The Titan library follows a **Strict Layered Architecture** to ensure stability and reproducibility.

### 1. The "No Scripts" Rule

- `titan/` contains **only** functions, classes, and constants.
- It **never** contains code that runs immediately on import (no `if __name__ == "__main__"`).
- It **never** modifies `sys.path`.

### 2. Dependency Flow

Dependencies flow **inwards** towards core models:
- `titan.strategies` imports from `titan.models` and `titan.indicators` ✓
- `titan.models` does NOT import from `titan.strategies` (no circles) ✗
- `scripts/` may import from everything ✓

### 3. Configuration Injection

Functions accept configuration as arguments (dicts or config objects). They avoid loading
files from `config/` directly where possible — keeps the library testable.

---

## Installation

```bash
# From the project root — always use uv, never bare pip
uv sync
```

> [!IMPORTANT]
> Never use `pip install -e .` or `pip install`. All dependency management in this project
> goes through `uv`. This is a non-negotiable workspace rule.

---

## Modules Reference

### 1. `titan.adapters` — IBKR via NautilusTrader

IBKR connectivity is handled by **NautilusTrader's built-in Interactive Brokers adapter** —
not a custom REST client. Do not use `ibkrpyV20` or any standalone IBKR REST library.

The adapter is configured via `InteractiveBrokersDataClientConfig` and
`InteractiveBrokersExecClientConfig` from `nautilus_trader.adapters.interactive_brokers.config`.

Key objects:
- `IBContract` — defines the instrument (secType, symbol, currency, exchange)
- `IB` — venue key constant (use this, not the string `"IBKR"`)
- `IBMarketDataTypeEnum.REALTIME` / `DELAYED_FROZEN` — market data mode

> **Full documentation:** See `directives/Titan-IBKR Adapter Guide.md` and
> `directives/IBKR & NautilusTrader API Reference.md`.

---

### 2. `titan.data`

Utilities for fetching, validating, and managing historical data.

- **`titan.data.ibkr`**: Primitives for IBKR history requests via the NautilusTrader adapter.
  - Used by `scripts/download_data.py` (called automatically by live runners at startup).
- **`titan.data.validation`**: Data integrity checks.
  - `check_gaps(df)`: Detects missing candles.
  - `check_outliers(df)`: Flags suspicious price spikes.

---

### 3. `titan.indicators`

High-performance technical indicators optimized for both Numba (backtesting) and Python (live).

- **`titan.indicators.gaussian_filter`**: Ehlers-based Gaussian Channel indicator (Numba `@njit`).
  - Used as a filter in ORB strategy (`use_gauss` per-ticker config).
  - Params: `period`, `poles`, `sigma` (from `config/gaussian_channel_config.toml`).
- **`titan.indicators.common`**: Shared standard indicators (SMA, EMA, RSI). [In progress]

---

### 4. `titan.models`

Quantitative models for market physics and trading costs.

- **`titan.models.spread`**: Time-varying spread and slippage estimation.
  - `build_spread_series(df, pair)`: Session-based spread model (Tokyo/London/NY).
  - `estimate_slippage(units, volume)`: Square-root impact model.

---

### 5. `titan.risk`

Portfolio-level risk management shared by all live strategies.

- **`titan.risk.portfolio_risk_manager`**: Module-level singleton, auto-loads from `config/risk.toml [portfolio]`.
  - `register_strategy(id, equity)` -- call in on_start()
  - `update(id, equity)` -- call on every bar
  - `update_vix(level)` -- feed VIX for regime scaling (optional)
  - `update_atr_percentile(id, pct)` -- feed ATR percentile for regime scaling
  - `halt_all` -- True when portfolio DD exceeds 15% (kill switch)
  - `scale_factor` -- min(dd_scale, vol_scale, regime_scale) multiplier
  - `check_correlation_regime()` -- pairwise correlation alerts (r > 0.85)

- **`titan.risk.portfolio_allocator`**: Module-level singleton, auto-loads from `config/risk.toml [allocation]`.
  - `tick()` -- call once per bar; triggers monthly rebalance when due
  - `get_weight(id)` -- returns inverse-vol allocation weight for sizing
  - `force_rebalance()` -- trigger immediate rebalance

---

### 6. `titan.strategies`

Production-grade strategy logic, separated from the execution harness.

- **`titan.strategies.ic_equity_daily`**: IC mean-reversion -- 7 US equities, daily RSI(21) deviation.
- **`titan.strategies.ic_generic`**: Asset-agnostic IC strategy framework.
- **`titan.strategies.ic_mtf`**: IC multi-timeframe FX signals.
- **`titan.strategies.mtf`**: MTF Confluence -- EUR/USD, H1/H4/D/W, continuous forecasts, conviction sizing.
- **`titan.strategies.orb`**: Opening Range Breakout -- 7 US equities, bracket orders, EOD flatten.
- **`titan.strategies.etf_trend`**: ETF Trend -- 7 ETFs (SPY/QQQ/IWB/TQQQ/EFA/GLD/DBC), MOC orders.
- **`titan.strategies.mr_fx`**: Mean Reversion FX -- EUR/USD M5, VWAP deviation grid, regime-gated.
- **`titan.strategies.ml`**: ML signal classifier + shared feature engineering.
  - `titan.strategies.ml.features` must match training code exactly to avoid feature drift.
- **`titan.strategies.turtle`**: Turtle Trading -- Donchian breakout + ATR sizing.
- **`titan.strategies.gap_fade`**: Session Gap Fade -- EUR/USD M5, fade overnight gaps at London open, bracket orders, EOD hard close.
- **`titan.strategies.fx_carry`**: FX Carry Trade -- AUD/JPY daily, collect carry premium with SMA trend filter, vol-targeted sizing.
- **`titan.strategies.gold_macro`**: Gold Macro -- GLD daily, 3-component cross-asset signal (real rates + dollar + momentum), vol-targeted sizing.
- **`titan.strategies.pairs`**: Pairs Trading -- market-neutral spread mean reversion with walk-forward beta refit. Validated: GLD/EFA.

All strategies are integrated with `PortfolioRiskManager` (register, update, halt check, scale_factor)
and `PortfolioAllocator` (inverse-vol allocation weights).

---

### 7. `titan.utils`

Operational utilities for production handling.

- **`titan.utils.ops`**: Emergency operations (cancel orders, close positions).
- **`titan.utils.notification`**: Slack alerting integration.

---

## Development Guidelines

1. **Strict Separation**: Never import from `scripts/` or `research/` into `titan/`.
2. **No `sys.path`**: Do not use `sys.path.insert` in library code. Use proper package installation.
3. **Type Hints**: All `titan/` functions must be fully type-hinted.
4. **Config Injection**: Pass configuration to functions; do not load files from global paths inside library code.
5. **NautilusTrader API**: Use factory methods only — `Price.from_str()`, `Quantity.from_str()`, `Quantity.from_int()`. No direct constructors.
6. **Financial types**: `decimal.Decimal` or NautilusTrader native types (`Price`, `Quantity`). No bare floats for price or volume.

---

## Pre-Push Gates

Run these before every push:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest tests/ -v
```

All three must pass.
