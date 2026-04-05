# Paper Trading Guide

> Last updated: 2026-04-05
> System: Titan-IBKR-Algo v3 -- 17 strategies, unified portfolio runner

---

## 1. System Overview

Titan-IBKR is a multi-strategy quantitative trading system that connects to Interactive Brokers via NautilusTrader. It runs 9 independent strategy processes, each subscribing to bar data and executing orders through the IBKR API.

### Strategy Map

| # | Strategy | Instrument(s) | Timeframe | Style | Runner | WFO Sharpe |
|---|----------|---------------|-----------|-------|--------|-----------|
| 1 | ~~MTF Confluence~~ | ~~EUR/USD~~ | ~~H1/H4/D/W~~ | ~~FX trend~~ | `run_live_mtf.py` | **INVALIDATED** |
| 2 | **ORB** | 7 US equities | 5M | Intraday breakout | `run_live_orb.py` | Validated |
| 3 | **ETF Trend** | SPY (expandable) | D | Trend following | `run_live_etf_trend.py` | +0.89 |
| 4 | **IC Equity Daily** | 7 US equities | D | Mean reversion | `run_live_ic_mtf.py` | +2.65-3.41 |
| 5 | **ML Classifier** | EUR/USD (or others) | D/H1 | ML signal | `run_live_ml.py` | +0.55-1.58 |
| 6 | **Gold Macro** | GLD | D | Cross-asset macro | `run_live_gold_macro.py` | +0.60 |
| 7 | **Gap Fade** | EUR/USD | M5 | Intraday MR | `run_live_gap_fade.py` | Validated |
| 8 | **FX Carry** | AUD/JPY | D | Carry premium | `run_live_fx_carry.py` | Validated |
| 9 | **Pairs Trading** | GLD / EFA | D | Market-neutral | `run_live_pairs.py` | +1.14 |
| 10 | **GLD Confluence** | GLD | H1 | Multi-scale trend | `run_live_gld_confluence.py` | **+1.46** |
| 11 | **AUD/JPY MR** | AUD/JPY | H1 | MR + confluence regime | `run_live_mr_audjpy.py` | **+2.08** |
| 12 | **Bond->Gold** | GLD (signal: IEF) | D | Cross-asset momentum | `run_live_bond_gold.py` | **+1.17** |
| 13 | **MR FX** | EUR/USD | M5 | VWAP MR tiered grid | `run_live_mr_fx.py` | Validated |

> **Recommended for paper trading:** Use `run_portfolio.py --strategies gold_core` (strategies 6, 10, 12) as the starting set. These have the highest WFO Sharpe and lowest correlation.

### Risk Architecture

All strategies share two risk layers:

- **PortfolioRiskManager** (`titan/risk/portfolio_risk_manager.py`):
  `scale_factor = min(dd_scale, vol_scale, regime_scale)`
  - DD > 10%: linear scale-down to 25% floor. DD > 15%: kill all.
  - Vol-targeting: EWMA realized vol vs 12% target. Clip [0.25, 2.0].
  - Regime: VIX tiers + ATR percentile. Min of both.

- **PortfolioAllocator** (`titan/risk/portfolio_allocator.py`):
  Inverse-volatility weighting, rebalanced monthly. Min 5%, max 60% per strategy.

---

## 2. Prerequisites

### Software
- **Python 3.11+** installed
- **uv** package manager: `pip install uv` then `uv sync`
- **IBKR Gateway** or **TWS** installed and running

### IBKR Setup
1. Log in to IBKR Gateway (or TWS) with your **paper trading** account.
2. Go to **Settings > API > Settings**:
   - Check "Enable ActiveX and Socket Clients"
   - **Uncheck** "Read-Only API" (we need to place orders)
   - Note the Socket port: **4002** (Gateway paper) or **7497** (TWS paper)
3. Leave Gateway running -- it must stay open while strategies run.

### Environment File
```bash
cp .env.example .env
```

Edit `.env`:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4002              # Paper: 4002 (Gateway) or 7497 (TWS)
IBKR_CLIENT_ID=1            # Each runner uses its own ID (see table below)
IBKR_ACCOUNT_ID=DUxxxxxxx   # Your paper account ID (starts with DU)
```

### Verify Connection
```bash
uv run python scripts/verify_connection.py
```

Expected output: `IBKR Connection Verified` with your account ID.

---

## 3. Data Setup

Most strategies warm up from local Parquet files. Download before first run:

```bash
# FX data (EUR/USD, GBP/USD, AUD/JPY, etc.) -- from IBKR
uv run python scripts/download_data.py

# MTF warmup (EUR/USD all timeframes)
uv run python scripts/download_data_mtf.py --pair EUR_USD --years 10

# ETF + equity daily data -- from Yahoo Finance (deep history, no API key)
uv run python scripts/download_data_yfinance.py --symbols SPY QQQ GLD EFA TIP TLT DXY

# Update the data manifest
uv run python scripts/build_data_manifest.py
```

Verify key files exist:
```bash
ls data/EUR_USD_H1.parquet data/EUR_USD_D.parquet data/GLD_D.parquet data/EFA_D.parquet
```

---

## 4. Running Strategies

### Quick Start -- Run the Portfolio (RECOMMENDED)

The unified portfolio runner manages all strategies in a single process:

```bash
# Run all strategies in one process (recommended)
uv run python scripts/run_portfolio.py --strategies all

# Run only daily strategies (lower bar volume, simpler)
uv run python scripts/run_portfolio.py --strategies daily_only

# Run only gold-focused strategies
uv run python scripts/run_portfolio.py --strategies gold_core

# Run specific strategies by name
uv run python scripts/run_portfolio.py --strategies gld_confluence bond_gold etf_trend_spy
```

The portfolio runner:
- Shares a single IBKR connection (one client ID)
- Manages all instrument subscriptions automatically
- PortfolioRiskManager controls portfolio-level risk (DD breaker, vol-targeting)
- PortfolioAllocator handles cross-strategy allocation (inverse-vol weighting)
- Press `Ctrl+C` to gracefully stop all strategies

#### Available Strategy Sets

| Set | Strategies | Description |
|-----|-----------|-------------|
| `all` | etf_trend_spy, gold_macro, bond_gold, fx_carry_audjpy, gld_confluence, mr_audjpy, ic_equity_noc | Everything |
| `daily_only` | etf_trend_spy, gold_macro, bond_gold, fx_carry_audjpy, ic_equity_noc | Daily timeframe only |
| `gold_core` | gld_confluence, gold_macro, bond_gold | Gold-focused (highest Sharpe) |
| `h1_only` | gld_confluence, mr_audjpy | H1 timeframe strategies |

#### Available Individual Strategies

| Name | Instrument | Timeframe | Signal | WFO Sharpe |
|------|-----------|-----------|--------|-----------|
| `etf_trend_spy` | SPY | Daily | Slow MA trend | +0.89 |
| `gold_macro` | GLD | Daily | Cross-asset (TIP/TLT/DXY) | +0.60 |
| `bond_gold` | GLD (signal: IEF) | Daily | Bond momentum -> gold | +1.17 |
| `fx_carry_audjpy` | AUD/JPY | Daily | SMA trend + carry | Validated |
| `gld_confluence` | GLD | H1 | AND-gated multi-scale trend_mom | +1.46 |
| `mr_audjpy` | AUD/JPY | H1 | VWAP MR + confluence regime | +2.08 |
| `ic_equity_noc` | NOC | Daily | RSI(21) mean reversion | +3.06 |

### Alternative: Run Individual Strategies

You can still run strategies individually in separate terminals:

```bash
# Terminal 1: MTF Confluence (EUR/USD H1/H4/D/W) -- NOTE: edge invalidated, kept for reference
uv run python scripts/watchdog_mtf.py

# Terminal 2: ORB (7 US equities, 5M bars, market hours only)
uv run python scripts/run_live_orb.py

# Terminal 3: ETF Trend (SPY daily)
uv run python scripts/run_live_etf_trend.py

# Terminal 4: IC Equity Daily (7 US equities)
uv run python scripts/run_live_ic_mtf.py

# Terminal 5: Gold Macro (GLD daily, cross-asset signal)
uv run python scripts/run_live_gold_macro.py

# Terminal 6: Pairs Trading (GLD/EFA daily, market-neutral)
uv run python scripts/run_live_pairs.py

# Terminal 7: FX Carry (AUD/JPY daily)
uv run python scripts/run_live_fx_carry.py

# Terminal 8: Gap Fade (EUR/USD M5, London session only)
uv run python scripts/run_live_gap_fade.py

# Terminal 9: ML Classifier (needs trained model in models/)
uv run python scripts/run_live_ml.py
```

### Client ID Assignment

**Portfolio runner** uses a single client ID (default: 1 from `.env`).

**Individual runners** each need a unique client ID:

| Runner | Default Client ID |
|--------|:-:|
| **run_portfolio.py** | **1 (from .env)** |
| watchdog_mtf.py | 1 |
| run_live_orb.py | 2 |
| run_live_etf_trend.py | 3 |
| run_live_ic_mtf.py | 4 |
| run_live_ml.py | 1 |
| run_live_gold_macro.py | 10 |
| run_live_pairs.py | 11 |
| run_live_fx_carry.py | 12 |
| run_live_gap_fade.py | 13 |
| **run_live_gld_confluence.py** | **14** |
| **run_live_mr_audjpy.py** | **15** |
| **run_live_bond_gold.py** | **16** |
| **run_live_mr_fx.py** | **17** |
| kill_switch.py | 98 |

> **Important:** Do NOT run the portfolio runner and individual runners at the same time -- they will conflict on client IDs and instrument subscriptions. Use one or the other.

---

## 5. What To Expect After Startup

### Healthy Startup Sequence

```
============================================================
  GOLD MACRO STRATEGY -- IBKR GATEWAY
  Host: 127.0.0.1:4002  |  Mode: PAPER
  Account: DU1234567  |  ClientID: 10
============================================================
  GLD: 252 D bars loaded
  TIP: 252 D bars loaded
  TLT: 252 D bars loaded
  DXY: 252 D bars loaded
Gold Macro Strategy attached for GLD.ARCA.
Starting Gold Macro Trading Node...
```

### When Do Trades Happen?

| Strategy | Trading Hours | Signal Frequency |
|----------|--------------|-----------------|
| MTF Confluence | 24/5 (FX) | Every H1 bar close |
| ORB | 09:30-15:55 ET (US equities) | First 5M breakout after open |
| ETF Trend | Daily at 15:30 ET (MOC) | Daily bar close |
| IC Equity Daily | Daily bar close | When z-score crosses threshold |
| Gold Macro | Daily bar close | When composite signal flips |
| Pairs Trading | Daily bar close | When spread z > 2.0 |
| FX Carry | Daily bar close (24/5) | When SMA filter changes |
| Gap Fade | 07:00-07:30 UTC (London open) | Only on gap days |
| ML Classifier | Per model timeframe | Each bar close |

**Be patient.** Daily strategies may not trade for days or weeks. This is normal -- they are selective.

### Key Log Messages

| Message | Meaning |
|---------|---------|
| `Warmup complete` | Indicators ready, strategy is trading |
| `ENTRY BUY / SELL` | Order submitted |
| `FILLED` | Order executed by IBKR |
| `CLOSED: PnL=...` | Position closed, P&L reported |
| `Portfolio kill switch` | DD > 15%, all strategies flattening |
| `Scale XX%` | Risk manager reducing position sizes |
| `Model degraded` | ML health monitor detected drift |

---

## 6. Monitoring

### Log Files

All logs are in `.tmp/logs/`:

```bash
# Watch a strategy's log in real time
tail -f .tmp/logs/gold_macro_live_*.log

# Check for errors across all strategies
grep -i "error\|rejected\|exception" .tmp/logs/*.log | tail -20

# Check portfolio risk manager state
grep "PortfolioRM" .tmp/logs/*.log | tail -10
```

### Account State

```bash
# Check positions and orders (dry run -- reads only)
uv run python scripts/kill_switch.py --dry-run

# Check account balance
uv run python scripts/check_balance.py
```

### IBKR TWS/Gateway

Open the TWS order panel to see:
- Active positions (should match strategy logs)
- Pending orders (stop losses, take profits)
- Account equity and margin usage

---

## 7. Stopping Strategies

### Graceful Stop (one strategy)
Press `Ctrl+C` in the terminal running the strategy. It will:
1. Cancel all pending orders for that strategy's instruments
2. Close all open positions
3. Exit cleanly

### Emergency Stop (all strategies)
```bash
uv run python scripts/kill_switch.py
```

This connects directly to IBKR (client ID 98) and:
1. Cancels ALL pending orders across ALL instruments
2. Closes ALL open positions at market price
3. Does NOT require the strategy processes to be running

Then stop all strategy processes:
```bash
# Windows
taskkill /F /IM python.exe

# Linux/Mac
pkill -f "run_live_"
```

### After Emergency Stop

1. Verify in TWS that no positions remain open
2. Check `.tmp/logs/` for the error that caused the issue
3. Do NOT restart until you understand what happened
4. Run `scripts/verify_connection.py` to confirm IBKR is still connected

---

## 8. Recommended Paper Testing Plan (Using Portfolio Runner)

### Week 1: Gold Core (highest Sharpe, simplest)

Start with the gold-focused strategies (3 strategies, all proven):

```bash
uv run python scripts/run_portfolio.py --strategies gold_core
```

This runs: GLD Confluence (H1), Gold Macro (daily), Bond->Gold (daily).

Monitor for:
- Clean startup with all 3 strategies attached
- Warmup data loaded for GLD, TIP, TLT, DXY, IEF
- GLD Confluence: confluence score printed each H1 bar
- Gold Macro: composite signal printed each daily bar
- Bond->Gold: IEF momentum z-score printed each daily bar

### Week 2: Add Daily Strategies

Switch to daily_only set (adds ETF Trend, FX Carry, IC Equity):

```bash
uv run python scripts/run_portfolio.py --strategies daily_only
```

Monitor for:
- All 5 strategies attached and producing signals
- ETF Trend: position changes on MA crossovers
- FX Carry: AUD/JPY entries when price > SMA(50)
- IC Equity: RSI-based entries on NOC

### Week 3: Full Portfolio

Run everything:

```bash
uv run python scripts/run_portfolio.py --strategies all
```

This adds the H1 strategies (GLD Confluence, AUD/JPY MR) to the daily set.

Monitor for:
- All 7 strategies producing signals
- AUD/JPY MR: tiered entries during 07:00-12:00 UTC, hard closes at 21:00 UTC
- Portfolio risk manager: check logs for `Scale` messages (DD/vol/regime scaling)
- No conflicting orders on shared instruments (GLD is traded by 3 strategies)

### Week 4: Stress Test

Test the kill switch and recovery:

```bash
# In another terminal while portfolio is running:
uv run python scripts/kill_switch.py --dry-run   # preview only
uv run python scripts/kill_switch.py              # flatten everything
```

Then restart:
```bash
uv run python scripts/run_portfolio.py --strategies all
```

### Success Criteria Before Live

- [ ] Portfolio runner starts all strategies without `[ERROR]` lines
- [ ] At least one trade opened and closed per strategy
- [ ] Kill switch tested -- confirmed it flattens all positions
- [ ] No orphaned stop orders after clean shutdown (`Ctrl+C`)
- [ ] Portfolio risk manager scaling works (check logs for `Scale` messages)
- [ ] PortfolioAllocator rebalances monthly (check logs for weight updates)
- [ ] No conflicting trades on shared instruments (GLD, AUD/JPY)
- [ ] Logs clean during IBKR nightly maintenance (23:45-00:05 ET)
- [ ] At least 30 days of continuous paper operation
- [ ] Portfolio P&L tracked in `.tmp/logs/portfolio_live_*.log`

---

## 9. Switching to Live

> **DO NOT** switch to live until all success criteria above are met.

1. Stop all paper strategies
2. Log in to IBKR Gateway with your **live** account
3. Update `.env`:
   ```ini
   IBKR_PORT=4001              # Live Gateway port
   IBKR_ACCOUNT_ID=Uxxxxxxx   # Your live account ID (starts with U)
   ```
4. Start one strategy at a time. Monitor for 48 hours before adding the next.
5. Use the watchdog for any strategy running 24/7.

---

## 10. Configuration Reference

All strategy parameters are in `config/*.toml`. Key files:

| File | Strategy | Key Parameters |
|------|----------|---------------|
| `risk.toml` | All | DD thresholds, vol-target, VIX tiers, allocator |
| `mtf_eurusd.toml` | MTF (invalidated) | Timeframe weights, MA periods, RSI periods |
| `orb_live.toml` | ORB | Ticker list, ORB window, ATR multiplier, RR ratio |
| `etf_trend_spy.toml` | ETF Trend | Slow MA, deceleration mode, vol-target |
| `gold_macro_gld.toml` | Gold Macro | SMA period, real rate window, vol target |
| `pairs_gld_efa.toml` | Pairs | Entry/exit z, refit window, max leverage |
| `fx_carry_audjpy.toml` | FX Carry | SMA period, vol target, VIX halve threshold |
| `gap_fade_eurusd.toml` | Gap Fade | Gap ATR mult, TP fill %, ATR period |
| **`gld_confluence.toml`** | **GLD Confluence** | **Threshold, exit buffer, TF weights, warmup** |
| **`bond_gold.toml`** | **Bond->Gold** | **Lookback, threshold, hold days, vol target** |
| **`mr_audjpy.toml`** | **AUD/JPY MR** | **Percentile tiers, tier sizes, session hours, regime filter** |
| `mr_fx_eurusd.toml` | MR FX | Percentile window, tiers, reversion target, regime gate |

**Never change parameters while a strategy is running.** Stop the strategy, edit the TOML, then restart.

---

## 11. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Connection timed out` | IBKR Gateway not running | Start Gateway, check port |
| `IBKR_ACCOUNT_ID not set` | Missing .env | `cp .env.example .env` and edit |
| `Warmup file missing` | Data not downloaded | Run download scripts (Section 3) |
| `ORDER REJECTED` | Insufficient margin or wrong instrument | Check TWS order panel for reason |
| `Model not found` | No trained ML model | Run `uv run python research/ml/run_52signal_classifier.py` |
| `ConnectionResetError` | IB nightly maintenance | Watchdog handles this automatically |
| Strategy not trading | Market closed or no signal | Check trading hours table (Section 5) |
| `Portfolio kill switch` | DD > 15% triggered | Review positions, call `reset_halt()` manually |

---

## 12. File Reference

```
scripts/
  run_portfolio.py         UNIFIED PORTFOLIO RUNNER (recommended)
  watchdog_mtf.py          MTF with auto-restart
  run_live_mtf.py          MTF direct (edge invalidated)
  run_live_orb.py          ORB equities
  run_live_etf_trend.py    ETF Trend
  run_live_ic_mtf.py       IC Equity Daily
  run_live_ml.py           ML Classifier
  run_live_gold_macro.py   Gold Macro
  run_live_pairs.py        Pairs Trading
  run_live_fx_carry.py     FX Carry
  run_live_gap_fade.py     Gap Fade
  run_live_gld_confluence.py  GLD AND-gated Confluence (H1)
  run_live_mr_audjpy.py    AUD/JPY MR + Confluence (H1)
  run_live_bond_gold.py    Bond->Gold Momentum (daily)
  run_live_mr_fx.py        MR FX VWAP (M5)
  kill_switch.py           Emergency stop (cancel all + flatten)
  verify_connection.py     Test IBKR connection
  check_balance.py         Show account balance
  download_gld_h1.py       Download GLD/IEF H1 from IBKR

config/
  risk.toml                Portfolio risk + allocation parameters
  gld_confluence.toml      GLD Confluence parameters
  bond_gold.toml           Bond->Gold parameters
  mr_audjpy.toml           AUD/JPY MR parameters
  mr_fx_eurusd.toml        MR FX parameters
  orb_live.toml            ORB strategy parameters
  etf_trend_spy.toml       ETF Trend parameters
  gold_macro_gld.toml      Gold Macro parameters
  pairs_gld_efa.toml       Pairs Trading parameters
  fx_carry_audjpy.toml     FX Carry parameters
  gap_fade_eurusd.toml     Gap Fade parameters

.tmp/logs/                 Strategy logs (timestamped)
data/                      Historical Parquet files (warmup data)
models/                    Trained ML models (.joblib)
```
