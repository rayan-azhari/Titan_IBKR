# Titan Library Reference

> Last updated: 2026-04-21 (post portfolio-risk rewrite)

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

### 5. `titan.risk` (**rewritten April 21, 2026**)

Portfolio-level risk management shared by all live strategies. The layer was
rewritten on April 21, 2026 after an audit found four structural defects in
the original implementation. Architecture details:
`C:/Users/rayan/.claude/skills/titan-orchestrator/references/portfolio-risk-architecture.md`.

- **`titan.risk.strategy_equity`** (**NEW**): Per-strategy equity tracking + deterministic base-currency resolution + FX unit conversion.
  - `StrategyEquityTracker(prm_id, initial_equity, base_ccy="USD")` -- dataclass owned by each strategy. Accumulates realised P&L in base ccy on `on_position_closed`. Exposes `current_equity()`.
  - `get_base_balance(account, base_ccy="USD")` -> `float | None` -- returns account balance in an explicit currency, or `None` if that ccy is absent. **Never** picks a non-deterministic `ccys[0]` fallback.
  - `convert_notional_to_units(notional_base, price, quote_ccy, base_ccy, fx_rate_quote_to_base)` -> `int` -- FX-aware unit conversion. Raises `ValueError` if `quote_ccy != base_ccy` and no rate supplied.
  - `split_fx_pair("AUD/JPY.IDEALPRO")` -> `("AUD", "JPY")` -- convenience parser.
  - `report_equity_and_check(strategy, prm_id, bar, tracker=None)` -> `(equity, halted)` -- drop-in replacement for the legacy `balance_total` block. Feeds PRM with explicit `bar.ts_event` timestamp.

- **`titan.risk.portfolio_risk_manager`**: Module-level singleton, auto-loads from `config/risk.toml [portfolio]`. Timestamp-aware state.
  - `register_strategy(id, initial_equity)` -- call in `on_start()`.
  - `update(id, equity, ts)` -- call on every bar. `ts` accepts nanosecond-epoch int, `datetime`, or `pd.Timestamp`.
  - `update_vix(level)` -- feed VIX for regime scaling (optional).
  - `update_atr_percentile(id, pct)` -- feed ATR percentile for regime scaling.
  - `halt_all` -- True when portfolio DD exceeds 15% (kill switch). **Persisted to `.tmp/portfolio_halt.json`.**
  - `scale_factor` -- `min(dd_scale, vol_scale, regime_scale)` multiplier.
  - `get_equity_histories()` -> `dict[str, pd.Series]` -- **public accessor**. Allocator and external code must use this instead of private `_strategies`.
  - `get_summary()` -- full snapshot dict for monitoring / logging.
  - `check_correlation_regime()` -- pairwise correlation alerts (r > 0.85). Runs automatically once per calendar date; callable manually.
  - `reset_halt(operator="<name>")` -- clear halt flag. **Preserves pre-halt HWM.** Writes a cleared-halt record to `.tmp/portfolio_halt.json`.
  - `reset_hwm(operator="<name>")` -- explicit HWM re-anchoring to current total equity.

- **`titan.risk.portfolio_allocator`**: Module-level singleton, auto-loads from `config/risk.toml [allocation]`. Wall-clock gated.
  - `tick(now: date | None = None)` -- call once per bar. Rebalances when `(now - last_rebalance) >= interval_days`. **Measures wall-clock calendar days, not tick count.** Pass `bar_ts.date()` from your strategy.
  - `get_weight(id)` -> `float` -- inverse-vol allocation weight for sizing.
  - `get_all_weights()` -> `dict[str, float]` -- full snapshot.
  - `force_rebalance()` -- triggers rebalance on next `tick()` regardless of calendar distance.

**Integration contract for every live strategy** (see `references/portfolio-risk-architecture.md` for the full template):

```python
# on_start
portfolio_risk_manager.register_strategy(self._prm_id, self.config.initial_equity)
self._equity_tracker = StrategyEquityTracker(prm_id=self._prm_id,
                                              initial_equity=self.config.initial_equity)

# on_bar
_, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
if halted: self._flatten(); return
portfolio_allocator.tick(now=unix_nanos_to_dt(bar.ts_event).date())

# sizing
units = convert_notional_to_units(notional, px, quote_ccy, "USD",
                                   fx_rate_quote_to_base=fx_rate)

# on_position_closed
self._equity_tracker.on_position_closed(pnl_quote, fx_to_base=fx_rate)
```

**Regression coverage:** `tests/test_portfolio_risk_april2026_fixes.py` -- 13 tests covering per-strategy-equity honour, wall-clock gating, timestamp alignment, halt persistence, and FX unit conversion.

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
