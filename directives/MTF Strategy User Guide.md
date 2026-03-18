# MTF Strategy User Guide

> Last updated: 2026-03-18

## Overview

The Multi-Timeframe Confluence (MTF) Strategy trades **EUR/USD** on IDEALPRO by computing a
weighted score across four timeframes (H1, H4, Daily, Weekly). When the score exceeds a threshold
in either direction, the strategy enters a position and places a hard ATR stop. Positions are
closed when the score reverses past the exit buffer, or the ATR stop is triggered.

**Instrument:** EUR/USD on IDEALPRO (`EUR/USD.IDEALPRO`)
**Timeframes:** H1 (entry timing), H4 (swing), D/Daily (primary trend), W/Weekly (macro regime)
**Validated OOS Sharpe (signal-only):** 2.14 | Swap-adj return: ~334% over 21yr | Max Drawdown: ~4%
**Config:** `config/mtf_eurusd.toml` | **Runner:** `scripts/run_live_mtf.py` | **Watchdog:** `scripts/watchdog_mtf.py`

Unlike ORB (intraday, EOD flatten), MTF holds positions across sessions. There is **no end-of-day
flatten** — positions close when the signal reverses past the exit buffer or the stop fires.

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

### Starting the Strategy (Recommended — Watchdog Mode)

EUR/USD trades nearly 24/5 — you can start the MTF strategy any time the forex market is open
(Sunday 17:00 ET through Friday 17:00 ET).

Open **two terminal tabs** in Antigravity.

In **Tab 1**, start the **watchdog** (which auto-restarts the strategy on crash or IB disconnect):

```bash
# bash (Antigravity terminal) — Tab 1
uv run python scripts/watchdog_mtf.py 2>&1 | tee .tmp/logs/watchdog_stdout_$(date +%Y%m%d_%H%M%S).log
```

The watchdog:
- Starts `run_live_mtf.py` as a subprocess
- Waits **3 minutes** and restarts if it exits for any reason (crash, IB maintenance window)
- If the process dies in under 30 seconds (config error), waits 60 seconds and warns before retrying
- Logs all restart events to `.tmp/logs/watchdog.log`
- Stops cleanly on `Ctrl+C` (will not restart after receiving the signal)

Stop the watchdog with `Ctrl+C`. The current strategy process exits gracefully.

> **Manual run (without watchdog):**
> ```bash
> uv run python scripts/run_live_mtf.py 2>&1 | tee .tmp/logs/mtf_stdout_$(date +%Y%m%d_%H%M%S).log
> ```
> Use this for debugging only — the watchdog is preferred for unattended operation.

Do not close Tab 1 — closing it kills the watchdog and strategy.

### Monitor Live — Tab 2

```bash
# bash (Antigravity terminal) — Tab 2
# Find the latest stdout log
ls -t .tmp/logs/mtf_stdout_*.log | head -1

# Stream it live (replace filename with actual)
tail -f .tmp/logs/mtf_stdout_20260315_091500.log
```

Filter for only important events:
```bash
tail -f .tmp/logs/mtf_stdout_20260315_091500.log | grep -i "Signal\|Exit\|FILLED\|REJECTED\|ERROR\|WARN"
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
Warmup complete. ATR size mult: 4.0x. Exit: signal reversal only.
```

If all 4 steps complete without `[ERROR]` lines, the strategy is healthy and watching for signals.

The dashboard prints on every H1 bar close with current scores. If MA/RSI values show `??`,
warmup data is missing — see Common Errors below.

---

## Strategy Logic — How Trades Are Taken

### Confluence Score

The strategy computes a weighted score across all timeframes:

```
Score = 0.55 × Signal_D + 0.30 × Signal_W + 0.10 × Signal_H1 + 0.05 × Signal_H4
```

Each timeframe signal ∈ {−1.0, −0.5, 0.0, +0.5, +1.0}:
- **MA component**: fast WMA > slow WMA → +0.5, else −0.5
- **RSI component**: RSI > 50 → +0.5, else −0.5
- **Sum**: −1.0 to +1.0 per timeframe

**Entry (next bar):** Long if Score ≥ +0.10 | Short if Score ≤ −0.10

### Why the Strategy Trades H4 Bars But Is Driven by Daily/Weekly

The strategy **subscribes to H4 bars** to get price updates throughout the day, but the confluence
score is dominated by **Daily (55%) and Weekly (30%) signals**. H4 has only 5% weight.

In practice, the score rarely changes within a single day — Daily and Weekly MAs move slowly.
Most entries happen when the Daily WMA crossover fires, and H4/H1 happen to agree.
Expect roughly 1–2 signals per week, not per day.

### Indicator Parameters (from `config/mtf_eurusd.toml`)

| TF | MA Type | Fast | Slow | RSI Period |
|---|---|---|---|---|
| H1 | WMA | 10 | 50 | 14 |
| H4 | WMA | 10 | 40 | 14 |
| D  | WMA | 10 | 20 | 7 |
| W  | WMA | 8  | 21 | 14 |

### Exit Logic — Signal Reversal with Exit Buffer

The strategy exits on **signal reversal only** — no hard stop order is placed. ATR is used
exclusively for position sizing.

**Exit rule:**
- **Long:** close when `Score < −exit_buffer` (default: Score < −0.10)
- **Short:** close when `Score > +exit_buffer` (default: Score > +0.10)
- **Direction flip:** if Long and score drops to ≤ −threshold, flip directly to Short (and vice versa)

The `exit_buffer` creates a **hysteresis dead band**: the position is held through the neutral
zone (score between −0.10 and +0.10) and only closes when the score genuinely crosses to the
**opposite side** by at least 0.10. This prevents premature exits caused by the score briefly
dipping into neutral during a trend continuation.

> [!IMPORTANT]
> There is **no hard stop order** placed on IBKR. If the strategy process crashes while in a
> position, the position will remain open with no protection. The watchdog restarts the process
> and the strategy will resume managing the open position on the next bar. Always run via
> `watchdog_mtf.py` in production.

### Position Sizing

ATR is used **only for sizing** — it is not used to place a stop order.

```
stop_dist = 4.0 × ATR(14, H1)
units     = (equity × 0.01) / stop_dist          # 1% equity risk per trade
units     = min(units, equity × 5.0 / price)     # 5× leverage cap
```

---

## Reading the Logs — Key Events

Full lifecycle of a trade from entry to exit:

```
# 1. Signal fires (on bar close — executes on next bar)
Signal Entry: Long.
Submitted BUY 87000 | ATR=0.00400 | stop_dist=0.01600

# 2. Entry fills
Entry filled @ 1.08450.
POSITION OPENED: ... side=BUY qty=87000 avg_px=1.08450

# 3. Position held through neutral zone (score 0.0 to -0.09 — no exit)
CONFLUENCE: +0.04  |  Entry: +-0.1  |  Exit buf: +-0.1  |  Signal: FLAT
Position: LONG  (held — score not below -exit_buffer)

# 4a. Exit Long: score crosses below -exit_buffer
Exit Long: score=-0.11 < -0.10.
POSITION CLOSED: realized_pnl=+$582.90

# 4b. Direct flip: score drops to <= -threshold (same threshold as exit_buffer)
Signal Flip: Long -> Short.
POSITION CLOSED: realized_pnl=-$120.00
Signal Entry: Short.
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
logs a clean shutdown. **Any open position remains open on the IB side, unprotected by a stop.**

> [!CAUTION]
> Unlike ORB, MTF has no end-of-day flatten and **no hard stop order**. If you stop the process
> while holding a position, the position will sit unprotected on IB's side. Either:
> - Restart the strategy (watchdog will do this automatically), or
> - Manually close or add a stop in TWS if you do not intend to restart.

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

After a force kill, check TWS to confirm your position status. Since no stop order is placed,
manually add a protective stop in TWS if you do not plan to restart the strategy immediately.

---

## Autonomous Operation — Surviving the IB Nightly Maintenance Window

IB Gateway/TWS performs a daily restart around **23:45 ET** (approx. 04:45 UTC). The strategy
process will disconnect and exit. Three layers handle this automatically:

| Layer | What it does | Setup required |
|---|---|---|
| **IB Gateway auto-restart** | Restarts Gateway after maintenance | `Configure → Settings → Auto restart` → enable, time = `23:30 ET` |
| **NautilusTrader reconnect** | Retries connection for 60s on disconnect | Built into `run_live_mtf.py` (`connection_timeout=60`) |
| **Watchdog** | Restarts the full process after exit | Run `watchdog_mtf.py` instead of `run_live_mtf.py` |

With all three layers active, the strategy recovers without intervention.

**On restart after a disconnect:**
- NautilusTrader reconciliation (enabled by default) re-reads open positions and orders from IB
- The strategy detects the existing position via `self.cache.positions()` — no duplicate entry
- Open ATR stop orders are re-registered via `on_order_accepted` during reconciliation — signal exits work correctly

---

## Restarting Manually

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

Warmup parquet files are missing or empty. Run:
```bash
uv run python scripts/download_data_mtf.py
```

Then restart the strategy. All four parquet files must exist:
`data/EUR_USD_H1.parquet`, `data/EUR_USD_H4.parquet`, `data/EUR_USD_D.parquet`, `data/EUR_USD_W.parquet`

### No trades in long sessions

This is **expected behaviour**. The threshold of ±0.10 is conservative. EUR/USD Daily and Weekly
MAs move slowly — the score often stays in the neutral zone during range-bound regimes.

Average trade frequency: ~1.5 trades/week.

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

### `config/mtf_eurusd.toml` — Strategy Parameters

```toml
ma_type = "WMA"
confirmation_threshold = 0.10   # Score must exceed ±0.10 to enter
exit_buffer = 0.10              # Exit long when score < -0.10 (hold through neutral zone)
atr_stop_mult = 4.0             # ATR multiplier for position SIZING only (no stop order placed)

[weights]
H1 = 0.10   # Entry timing (10%)
H4 = 0.05   # Swing confirmation (5%)
D  = 0.55   # Primary trend — dominant driver (55%)
W  = 0.30   # Long-term macro regime filter (30%)

[H1]
fast_ma = 10 | slow_ma = 30 | rsi_period = 28

[H4]
fast_ma = 10 | slow_ma = 40 | rsi_period = 7

[D]
fast_ma = 5  | slow_ma = 20 | rsi_period = 10

[W]
fast_ma = 13 | slow_ma = 26 | rsi_period = 10
```

> [!CAUTION]
> Do NOT manually edit `config/mtf_eurusd.toml` to change weights or MA periods. These values
> were validated through a 6-stage optimization pipeline (OOS Signal-only Sharpe 2.14,
> Max DD 4.0%, 38/38 WFO windows positive). Any change requires a full re-backtest with OOS
> validation before deploying.

### `scripts/run_live_mtf.py` — Runner Settings

```python
risk_pct=0.01       # 1% of equity risked per trade
leverage_cap=5.0    # Maximum 5× leverage cap on position size
warmup_bars=1000    # Historical bars loaded per timeframe at startup
```

---

## Log File Location

Three log files are created per run:

| File | What it captures |
|---|---|
| `.tmp/logs/watchdog.log` | Watchdog restart events — attempt count, exit codes, restart delays |
| `.tmp/logs/mtf_live_YYYYMMDD_HHMMSS.log` | Runner startup messages (Python logger) |
| `.tmp/logs/mtf_stdout_YYYYMMDD_HHMMSS.log` | **Full output** — connections, subscriptions, scores, orders, fills, stops (requires `tee` at startup — see above) |

> **Important:** NautilusTrader strategy events (`self.log.info()` calls) go to **stdout only**.
> Always use `tee` so you have the full session record.

Log files are timestamped at startup. Clean up periodically:
```bash
# bash (Antigravity terminal) — delete logs older than 7 days
find .tmp/logs/ -name "*.log" -mtime +7 -delete
```
