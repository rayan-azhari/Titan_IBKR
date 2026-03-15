# MTF Strategy User Guide

> Last updated: 2026-03-14

## Overview

The Multi-Timeframe Confluence (MTF) Strategy trades **EUR/USD** on IDEALPRO by computing a
weighted score across four timeframes (H1, H4, Daily, Weekly). When the score exceeds a threshold
in either direction, the strategy enters a position and places a hard ATR stop. Positions are
closed when the score returns to neutral, reverses, or the ATR stop is triggered.

**Instrument:** EUR/USD on IDEALPRO (`EUR/USD.IDEALPRO`)
**Timeframes:** H1 (entry timing), H4 (swing), D/Daily (primary trend), W/Weekly (regime)
**Validated OOS Sharpe:** 2.936 | OOS CAGR: +29.44% | Max Drawdown: −5.12%
**Config:** `config/mtf.toml` | **Runner:** `scripts/run_live_mtf.py`

Unlike ORB (intraday, EOD flatten), MTF holds positions across sessions. There is **no end-of-day
flatten** — positions close when the signal changes or the stop fires.

---

## Terminal and Environment

> **All commands in this guide are run from the Antigravity terminal (the integrated terminal
> inside the Antigravity IDE/VSCode), NOT a standalone Windows terminal.**
>
> The Antigravity terminal uses **bash** (Git Bash under the hood on Windows). Do not use
> PowerShell or Windows CMD unless explicitly noted.

To open the integrated terminal in Antigravity:
- `Ctrl + `` ` (backtick) or **View → Terminal**

You should see a `bash` prompt:
```
rayan@MACHINE MINGW64 /c/Users/rayan/Desktop/Antigravity/Titan-IBKR (main)
$
```

If the working directory is wrong:
```bash
cd /c/Users/rayan/Desktop/Antigravity/Titan-IBKR
```

---

## Prerequisites

### 1. IBKR Gateway must be running

Open Interactive Brokers Gateway (or TWS) before starting the strategy.

**Port reference:**
| Mode | Application | Port |
|---|---|---|
| Live trading | TWS | 7496 |
| Paper trading | TWS | 7497 |
| Live trading | Gateway | 4001 |
| Paper trading | Gateway | 4002 |

Check your `.env` file:
```bash
cat .env
```

You should see `IBKR_PORT=4002` (paper) or `IBKR_PORT=4001` (live).

### 2. API settings in Gateway/TWS

In TWS/Gateway: **Edit → Global Configuration → API → Settings**

- **Enable ActiveX and Socket Clients** — must be checked
- **Read-Only API** — must be **unchecked** (if checked, all orders silently rejected with Error 321)
- **Socket port** — must match `IBKR_PORT` in `.env`

### 3. Environment variables

```bash
cat .env
```

Required variables:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # Paper gateway — change to 4001 for live
IBKR_CLIENT_ID=3        # MTF uses client ID 3 (do not reuse for other strategies)
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### 4. Verify IBKR connection

```bash
python scripts/check_balance.py
```

Expected:
```
Connecting to IBKR on 127.0.0.1:4002 for Account DUxxxxxxx...

✅ Connection Successful!
----------------------------------------
Account ID:      DUxxxxxxx
Net Liquidation: $XX,XXX.XX
Available Funds: $XX,XXX.XX
----------------------------------------
```

If this fails, check: Gateway is running, API is enabled, Read-Only is unchecked.

---

## Daily Workflow

### Starting the Strategy

EUR/USD trades nearly 24/5 — you can start the MTF strategy any time the forex market is open
(Sunday 17:00 ET through Friday 17:00 ET).

Open **two terminal tabs** in Antigravity.

In **Tab 1**, start the strategy with `tee` to capture full output:

```bash
# bash (Antigravity terminal) — Tab 1
uv run python scripts/run_live_mtf.py 2>&1 | tee .tmp/logs/mtf_stdout_$(date +%Y%m%d_%H%M%S).log
```

> **Why `tee`?** NautilusTrader logs all framework events (connections, bar subscriptions, order
> submissions, fills, ATR stops) to **stdout only** — they do not appear in `mtf_live_*.log`.
> Using `tee` captures the full output to a timestamped file for post-session review.

Do not close Tab 1 — closing it kills the strategy.

### Monitor Live — Tab 2

```bash
# bash (Antigravity terminal) — Tab 2
# Find the latest stdout log
ls -t .tmp/logs/mtf_stdout_*.log | head -1

# Stream it live (replace filename with actual)
tail -f .tmp/logs/mtf_stdout_20260314_091500.log
```

Filter for only important events:
```bash
tail -f .tmp/logs/mtf_stdout_20260314_091500.log | grep -i "Signal\|ATR stop\|FILLED\|REJECTED\|ERROR\|WARN"
```

---

## Startup Sequence — What to Expect

**Step 1 — Data download:**
```
Checking for latest data...
Data download finished.
```

**Step 2 — Node and instrument load:**
```
[INFO] TITAN-MTF.TradingNode: BUILDING
[INFO] TITAN-MTF.TradingNode: RUNNING
Added MTF Strategy for EUR/USD.
```

**Step 3 — Bar subscriptions (4 feeds):**
```
[INFO] DataClient-INTERACTIVE_BROKERS: Subscribed EUR/USD.IDEALPRO-1-HOUR-MID-EXTERNAL bars
[INFO] DataClient-INTERACTIVE_BROKERS: Subscribed EUR/USD.IDEALPRO-4-HOUR-MID-EXTERNAL bars
[INFO] DataClient-INTERACTIVE_BROKERS: Subscribed EUR/USD.IDEALPRO-1-DAY-MID-EXTERNAL bars
[INFO] DataClient-INTERACTIVE_BROKERS: Subscribed EUR/USD.IDEALPRO-1-WEEK-MID-EXTERNAL bars
```

**Step 4 — Warmup from historical data:**
```
MTF Strategy Started. Warming up...
  Loading H1 warmup from data/EUR_USD_H1.parquet
  Loading H4 warmup from data/EUR_USD_H4.parquet
  Loading D  warmup from data/EUR_USD_D.parquet
  Loading W  warmup from data/EUR_USD_W.parquet
Warmup complete. ATR stop mult: 2.5x. Ready for signals.
```

If all 4 steps complete without `[ERROR]` lines, the strategy is healthy and watching for signals.

The dashboard prints on every H1 bar close with current scores. If MA/RSI values show `??`,
warmup data is missing — see Common Errors below.

---

## Strategy Logic — How Trades Are Taken

### Confluence Score

The strategy computes a weighted score across all timeframes:

```
Score = 0.60 × Signal_D + 0.25 × Signal_H4 + 0.10 × Signal_H1 + 0.05 × Signal_W
```

Each timeframe signal ∈ {−1.0, −0.5, 0.0, +0.5, +1.0}:
- **MA component**: fast SMA > slow SMA → +0.5, else −0.5
- **RSI component**: RSI > 50 → +0.5, else −0.5
- **Sum**: −1.0 to +1.0 per timeframe

**Entry:** Long if Score ≥ +0.10 | Short if Score ≤ −0.10

### Indicator Parameters (from `config/mtf.toml`)

| TF | MA Type | Fast | Slow | RSI Period |
|---|---|---|---|---|
| H1 | SMA | 10 | 30 | 21 |
| H4 | SMA | 10 | 50 | 21 |
| D | SMA | 13 | 20 | 14 |
| W | SMA | 13 | 21 | 10 |

### Exit Logic (Two-Layer)

**Layer 1 — Signal reversal (primary):**
Score returns to neutral zone (−0.10 to +0.10) or flips direction. ATR stop is cancelled first,
then position is closed at market.

**Layer 2 — Hard ATR stop (secondary):**
A GTC `STOP_MARKET` order is placed immediately after the entry fills at:
- Long: `fill_price − 2.5 × ATR(14, H1)`
- Short: `fill_price + 2.5 × ATR(14, H1)`

This order remains open until cancelled by a signal exit or triggered by price.

### Position Sizing

```
stop_dist = 2.5 × ATR(14, H1)
units     = (equity × 0.01) / stop_dist          # 1% equity risk per trade
units     = min(units, equity × 5.0 / price)     # 5× leverage cap
```

This sizes each trade so the 2.5× ATR stop represents exactly 1% of equity loss if hit.

---

## Reading the Logs — Key Events

Full lifecycle of a trade from entry to exit:

```
# 1. Signal fires
Score=0.73 [D=+1.0 H4=+1.0 H1=+0.5 W=+0.5]. Opening LONG.
Submitting LONG 87000 EUR/USD.IDEALPRO @ market.

# 2. Entry fills
ORDER FILLED: O-20260314-... side=BUY qty=87000 px=1.08450 commission=3.22

# 3. ATR stop placed
ATR stop placed @ 1.08200 (2.5x ATR = 0.00250). Order: O-20260314-...-stop

# 4a. Signal reversal exit (winning or scratch trade)
Score=0.05 [D=+0.5 H4=+0.5 H1=-0.5 W=+0.5]. Signal Neutral. Closing position.
# or:
Score=-0.13. Signal Flip: Long -> Short. Closing position.
Cancelling ATR stop order O-20260314-...-stop
ORDER FILLED: O-20260314-... side=SELL qty=87000 px=1.09120 commission=3.22
POSITION CLOSED: realized_pnl=+$582.90

# 4b. ATR stop triggered (losing trade)
ATR stop triggered. Fill @ 1.08200.
POSITION CLOSED: realized_pnl=-$217.50
```

### Log Levels

| Level | Meaning |
|---|---|
| `[INFO]` | Normal operation — bars, scores, orders, fills |
| `[WARN]` | Non-critical — missing warmup data, expired orders |
| `[ERROR]` | Requires attention — rejected orders, sizing failures |

Filter for errors:
```bash
grep "ERROR\|REJECTED\|WARN" .tmp/logs/mtf_stdout_YYYYMMDD_HHMMSS.log
```

See all fills and P&L:
```bash
grep "ORDER FILLED\|POSITION CLOSED" .tmp/logs/mtf_stdout_YYYYMMDD_HHMMSS.log
```

---

## Stopping the Strategy

### Normal Stop

Press `Ctrl+C` in Tab 1. NautilusTrader shuts down gracefully — it cancels pending orders and
logs a clean shutdown. **Any open position remains open on the IB side**, held by the existing
ATR stop order. This is intentional — the stop continues protecting the position after shutdown.

> **Important:** Unlike ORB, MTF has no end-of-day flatten. If you stop the process while holding
> a position, the ATR stop remains live in TWS as a GTC order. Log into TWS to verify.

### Force Kill (if the process is stuck)

```bash
# bash (Antigravity terminal) — find the PID
python -c "
import subprocess
r = subprocess.run(['wmic', 'process', 'where', 'name=\"python.exe\"', 'get', 'ProcessId,CommandLine', '/value'], capture_output=True, text=True)
print(r.stdout)
"
```

Find the line containing `run_live_mtf.py` and note its PID:
```bash
python -c "import subprocess; subprocess.run(['taskkill', '/F', '/PID', 'REPLACE_WITH_PID'])"
```

After a force kill, check TWS to confirm the GTC stop order is still active on your position.

---

## Restarting

Before restarting after a crash, verify the old process is dead — otherwise you get Error 326
(client ID already in use):

```bash
python -c "
import subprocess
r = subprocess.run(['tasklist'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'python' in line.lower():
        print(line)
"
```

If you see multiple Python processes, check which are MTF-related and kill them first.

**After restart**, the strategy will re-read the current score on the next bar. If you were
holding a position when the strategy crashed, the ATR stop GTC order on IB's side is still
protecting you. The strategy will detect the open position at startup and resume managing it.

---

## Common Errors and Fixes

### Dashboard shows `??` for MA/RSI values

```
H1: fast=?? slow=?? rsi=?? | Score=??
```

Warmup parquet files are missing or empty. Run:
```bash
uv run python scripts/download_data_mtf.py
```

Then restart the strategy. All four parquet files must exist:
`data/EUR_USD_H1.parquet`, `data/EUR_USD_H4.parquet`, `data/EUR_USD_D.parquet`, `data/EUR_USD_W.parquet`

### No trades in long sessions

This is **expected behaviour**. The threshold of ±0.10 is conservative. When EUR/USD is in a
sideways/ranging regime, the Daily and H4 signals often cancel each other and the score stays
in the neutral zone (−0.10 to +0.10). Wait for a trending regime.

Average trade frequency: ~78 trades/year (~1.5 per week).

### Error 326 — Client ID already in use

```
ERROR: 326 Unable to connect as the client id is already in use.
```

An old strategy process is still holding the socket. Kill it (see Force Kill above), then restart.

### Error 321 — Read-Only API

```
ERROR: 321 API interface is currently in Read-Only mode.
```

In TWS/Gateway: **Edit → Global Configuration → API → Settings → uncheck "Read-Only API"**.

### `No account in cache`

IBKR Gateway/TWS is not connected or not running on the configured port. Check:
- Gateway is running
- Port in `.env` matches Gateway port
- API is enabled

### ATR stop placed but not cancelled on signal exit

Check logs for `Cancelling ATR stop` lines. If missing, the `_cancel_stops()` call may have failed.
Verify via TWS order panel — if an orphaned stop order exists, cancel it manually to avoid
an unwanted position reversal.

### Terminal floods with `Portfolio: Updated AccountState` lines

Normal. IB pushes every account field every ~3 minutes. Filter them out when monitoring:
```bash
tail -f .tmp/logs/mtf_stdout_YYYYMMDD_HHMMSS.log | grep -v "Updated AccountState"
```

---

## Checking Account Balance Mid-Session

Run this in a new tab — it uses a separate client ID and does not conflict with the running strategy:
```bash
python scripts/check_balance.py
```

---

## Configuration Reference

### `.env` — Connection Settings

```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # 4002=Gateway Paper, 4001=Gateway Live, 7497=TWS Paper, 7496=TWS Live
IBKR_CLIENT_ID=3        # MTF uses 3 — do not reuse for other strategies
IBKR_ACCOUNT_ID=DUxxxxxxx
```

Client ID assignments:

| Script | Client ID | Notes |
|---|---|---|
| `run_live_mtf.py` | 3 (hardcoded) | MTF strategy |
| `run_live_orb.py` | 20 (from `.env`) | ORB strategy |
| `check_balance.py` | 99 (hardcoded) | Safe to run alongside any strategy |
| `download_data_mtf.py` | 11 (hardcoded) | EUR/USD warmup data — called automatically by runner |
| `download_data.py` | 10 (hardcoded) | ORB stocks data — run separately pre-market |

### `config/mtf.toml` — Strategy Parameters

```toml
confirmation_threshold = 0.10   # Score must exceed ±0.10 to enter
atr_stop_mult = 2.5             # Hard stop = 2.5 × ATR(14, H1) from entry

[weights]
H1 = 0.10   # Entry timing (10%)
H4 = 0.25   # Swing confirmation (25%)
D  = 0.60   # Primary trend — dominant driver (60%)
W  = 0.05   # Long-term regime context (5%)

[H1]
fast_ma = 10 | slow_ma = 30 | rsi_period = 21

[H4]
fast_ma = 10 | slow_ma = 50 | rsi_period = 21

[D]
fast_ma = 13 | slow_ma = 20 | rsi_period = 14

[W]
fast_ma = 13 | slow_ma = 21 | rsi_period = 10
```

> [!CAUTION]
> Do NOT manually edit `config/mtf.toml` to change weights or MA periods. These values were
> validated through 3 rounds of optimization (OOS Sharpe 2.936). Any change requires a full
> re-backtest with OOS validation before deploying.

### `scripts/run_live_mtf.py` — Runner Settings

```python
risk_pct=0.01       # 1% of equity risked per trade
leverage_cap=5.0    # Maximum 5× leverage cap on position size
warmup_bars=1000    # Historical bars loaded per timeframe at startup
```

---

## Log File Location

Two log files are created per run:

| File | What it captures |
|---|---|
| `.tmp/logs/mtf_live_YYYYMMDD_HHMMSS.log` | Runner startup messages (Python logger) |
| `.tmp/logs/mtf_stdout_YYYYMMDD_HHMMSS.log` | **Full output** — connections, subscriptions, scores, orders, fills, stops (requires `tee` at startup — see above) |

> **Important:** NautilusTrader strategy events (`self.log.info()` calls) go to **stdout only**.
> Always use `tee` so you have the full session record.

Log files are timestamped at startup. Clean up periodically:
```bash
# bash (Antigravity terminal) — delete logs older than 7 days
find .tmp/logs/ -name "*.log" -mtime +7 -delete
```
