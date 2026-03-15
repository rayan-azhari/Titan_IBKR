# Titan Library Reference

> Last updated: 2026-03-14

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

### 5. `titan.strategies`

Production-grade strategy logic, separated from the execution harness.

- **`titan.strategies.orb`**: Opening Range Breakout — 7 US equities, bracket orders, EOD flatten.
- **`titan.strategies.mtf`**: Multi-Timeframe Confluence — EUR/USD, H1/H4/D/W, 2-layer exit.
- **`titan.strategies.ml`**: Machine Learning signal generation + feature engineering.
  - `titan.strategies.ml.features` must match training code exactly to avoid feature drift.

---

### 6. `titan.utils`

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
