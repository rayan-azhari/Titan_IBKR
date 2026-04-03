# Strategy Deployment Protocol

> Last updated: 2026-03-14

This document outlines the systematic process for moving a strategy from the **Research Phase**
(VectorBT backtesting) to **Live Deployment** (NautilusTrader/IBKR).

---

## 1. Prerequisites (Research Completion)

Before deployment, a strategy must pass all robustness gates:

- **OOS/IS Sharpe ratio ≥ 0.5** (MTF: 0.98 ✓, ORB: verified ✓)
- **OOS Sharpe ≥ 1.5** for live deployment (MTF Round 3: 2.936 ✓)
- Parameters locked in `config/<strategy>.toml` — do NOT deploy with in-memory values

See `directives/MTF Optimization Protocol.md` for the MTF optimization process.

---

## 2. Code Adaptation (Research → Production)

Research code (vectorized `research/`) must be ported to the event-driven engine (`titan/strategies/`).

### A. Strategy Logic Class

- **Location**: `titan/strategies/<name>/strategy.py`
- **Key requirements**:
  1. Load parameters via `Config` object and TOML file (e.g., `config/mtf.toml`)
  2. Subscribe to the correct `BarType`s for each timeframe
  3. Implement `on_bar` handler — buffer multi-timeframe bars, recalculate signal on each new bar
  4. Signal logic must exactly replicate the research backtest (same MA, same threshold, same weights)

*MTF example (production)*: `titan/strategies/mtf/strategy.py` — H1/H4/D/W bars, SMA, weighted score ≥ ±0.10.

### B. Configuration

- **Location**: `config/<strategy_name>.toml`
- **Content**: global params (MA type, threshold), weights, per-timeframe indicator settings, risk parameters

*MTF live config*: `config/mtf.toml` — validated Round 3 parameters.

---

## 3. Data Hygiene & Warmup

Strategies require historical data to calculate initial indicators before the first live tick.
Without warmup, the strategy will either crash or show `??` in the dashboard.

### Data Download

```bash
# ORB stocks (M5 + D)
uv run python scripts/download_data.py

# EUR/USD for MTF (H1, H4, D, W)
uv run python scripts/download_data_mtf.py
```

Saves Parquet files to `data/` using naming convention `data/{SYMBOL}_{GRANULARITY}.parquet`
(e.g., `EUR_USD_H1.parquet`, `EUR_USD_D.parquet`).

### Warmup Logic

The runner script calls the appropriate download script before starting the NautilusTrader node (`download_data_mtf.py` for MTF, `download_data.py` for ORB).
The strategy's `_warmup_all()` method loads the tail of these files at startup.

**MTF expected warmup sequence:**
```
Loading H1 warmup from data/EUR_USD_H1.parquet
Loading H4 warmup from data/EUR_USD_H4.parquet
Loading D  warmup from data/EUR_USD_D.parquet
Loading W  warmup from data/EUR_USD_W.parquet
Warmup complete. ATR stop mult: 2.5x. Ready for signals.
```

---

## 4. Execution Infrastructure

### Runner Script

**Location**: `scripts/run_live_<strategy>.py`

Required elements in every runner:
1. **Logging**: File handler to `.tmp/logs/` + stream handler for stdout
2. **Data warmup**: Call the download script before node.build() (`download_data_mtf.py` for MTF, `download_data.py` for ORB)
3. **IBContract**: Define the correct contract for each instrument
4. **Node ordering**: `node.build()` → `node.trader.add_strategy()` → `node.run()`
5. **market_data_type**: Infer from port — REALTIME for live (4001/7496), DELAYED_FROZEN for paper (4002/7497)
6. **Venue key**: Use `IB` constant from `nautilus_trader.adapters.interactive_brokers.common`, not the string `"IBKR"`

*MTF runner*: `scripts/run_live_mtf.py` (H1/H4/D/W bars, `config/mtf.toml`, client_id=3)
*ORB runner*: `scripts/run_live_orb.py` (5M + 1D bars per ticker, `config/orb_live.toml`, client_id from `.env`)

### Environment

```ini
# .env — required variables
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # 4002=Gateway Paper, 4001=Gateway Live, 7497=TWS Paper, 7496=TWS Live
IBKR_CLIENT_ID=3        # MTF uses 3; ORB uses value from .env (default 20)
IBKR_ACCOUNT_ID=DUxxxxxxx
```

---

## 5. Verification & Launch

### Paper Trading (always first)

```bash
# MTF
uv run python scripts/run_live_mtf.py

# ORB
uv run python scripts/run_live_orb.py
```

Check:
1. Warmup completes with no `[ERROR]` lines
2. Dashboard shows populated indicators (not `??`)
3. First trade opens and closes correctly (check logs for `ATR stop placed`, position close events)
4. `Ctrl+C` shuts down gracefully

### Live Trading (after clean paper session)

Switch `.env` to live port, confirm IBKR account type:
```ini
IBKR_PORT=4001   # Gateway Live
```

> [!CAUTION]
> Never switch to live while an open position exists on the paper account.
> Always verify account type in TWS before running.

---

## 6. Server Persistence

For long-term deployment, use one of these methods (see `directives/Deployment & Operations.md` for full details):

### Option A: tmux (testing / first week)

```bash
tmux new -s titan
uv run python scripts/run_live_mtf.py
# Ctrl+B then D to detach; tmux attach -t titan to re-attach
```

### Option B: systemd (production VPS)

```ini
# /etc/systemd/system/titan-mtf.service
[Service]
WorkingDirectory=/path/to/Titan-IBKR
ExecStart=/path/to/.venv/bin/python scripts/run_live_mtf.py
Restart=always
RestartSec=15
EnvironmentFile=/path/to/Titan-IBKR/.env
```

---

## Deployment Checklist

- [ ] OOS Sharpe ≥ 1.5 and OOS/IS ratio ≥ 0.5 confirmed in backtest
- [ ] Parameters locked in `config/<strategy>.toml`
- [ ] Warmup data downloaded (`scripts/download_data_mtf.py` for MTF, `scripts/download_data.py` for ORB)
- [ ] Paper session completed — startup sequence clean, at least one trade
- [ ] `.env` correct for target environment (paper vs live port)
- [ ] Kill switch tested: `uv run python scripts/kill_switch.py`
- [ ] Log monitoring in place (`tail -f .tmp/logs/...`)
