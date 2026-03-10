# ORB Strategy User Guide

## Overview

The Opening Range Breakout (ORB) Strategy trades 7 large-cap US equities by detecting breakouts from the first 10–15 minutes of the session. It runs on NautilusTrader connected to Interactive Brokers, submits bracket orders with automatic TP/SL management, and flattens all positions before market close.

**Instruments:** UNH (NYSE), AMAT (NASDAQ), TXN (NASDAQ), INTC (NASDAQ), CAT (NYSE), WMT (NASDAQ), TMO (NYSE)

---

## Terminal and Environment

> **All commands in this guide are run from the Antigravity terminal (the integrated terminal inside the Antigravity IDE/VSCode), NOT a standalone Windows terminal.**
>
> The Antigravity terminal uses **bash** (Git Bash under the hood on Windows). Do not use PowerShell or Windows CMD unless explicitly noted — path separators and commands differ.

To open the integrated terminal in Antigravity:
- `Ctrl + `` ` (backtick) or **View → Terminal**

You should see a `bash` prompt with the working directory set to the project root:
```
rayan@MACHINE MINGW64 /c/Users/rayan/Desktop/Antigravity/Titan-IBKR (main)
$
```

If the working directory is wrong, navigate to the project root:
```bash
cd /c/Users/rayan/Desktop/Antigravity/Titan-IBKR
```

---

## Prerequisites

### 1. IBKR Gateway must be running

Open Interactive Brokers Gateway (or TWS) on your machine before starting the strategy. The strategy connects over the local socket — it will fail immediately if Gateway is not running.

**Port reference:**
| Mode | Application | Port |
|---|---|---|
| Live trading | TWS | 7496 |
| Paper trading | TWS | 7497 |
| Live trading | Gateway | 4001 |
| Paper trading | Gateway | 4002 |

Check your `.env` file to confirm which port you are targeting:
```bash
cat .env
```
You should see `IBKR_PORT=4002` (paper) or `IBKR_PORT=4001` (live).

### 2. API settings in Gateway/TWS

In TWS/Gateway: **Edit → Global Configuration → API → Settings**

- **Enable ActiveX and Socket Clients** — must be checked
- **Read-Only API** — must be **unchecked** (if checked, all order submissions will be silently rejected with Error 321)
- **Socket port** — must match `IBKR_PORT` in your `.env`

### 3. Environment variables

The strategy reads credentials from `.env` in the project root. Verify it is populated:

```bash
# bash (Antigravity terminal)
cat .env
```

Required variables:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=20
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### 4. Verify IBKR connection

Before running the strategy, verify the connection is healthy:
```bash
# bash (Antigravity terminal)
python scripts/check_balance.py
```

Expected output:
```
Connecting to IBKR on 127.0.0.1:4002 for Account DUxxxxxxx...
Connection Successful!
----------------------------------------
Account ID:      DUxxxxxxx
Net Liquidation: $XX,XXX.XX
Available Funds: $XX,XXX.XX
----------------------------------------
```

If this fails, Gateway is not running or the port/account ID in `.env` is wrong.

---

## Daily Workflow

### Pre-Market (~9:00 ET)

1. Start IBKR Gateway
2. Verify API settings (Read-Only unchecked)
3. Run `python scripts/check_balance.py` to confirm connectivity
4. Optionally refresh historical data for indicator warmup:
   ```bash
   # bash (Antigravity terminal)
   python scripts/download_data.py
   ```

### Start the Strategy (~9:15 ET)

Open **two terminal tabs** in Antigravity. In the first tab, start the strategy:

```bash
# bash (Antigravity terminal) — Tab 1
python scripts/run_live_orb.py
```

The strategy runs in the **foreground**. You will see live logs directly in this terminal. Do not close this tab — closing it kills the strategy.

### Monitor the Live Log — Tab 2

In the second tab, tail the log file that the strategy writes to `.tmp/logs/`:

```bash
# bash (Antigravity terminal) — Tab 2
# Find the latest log file
ls -t .tmp/logs/ | head -3

# Stream it live (replace the filename with the actual one)
tail -f .tmp/logs/orb_live_20260310_091500.log
```

If you are on **PowerShell** (not recommended, but if needed):
```powershell
# PowerShell — NOT the Antigravity terminal
Get-Content .tmp\logs\orb_live_20260310_091500.log -Wait -Tail 50
```

---

## Startup Sequence — What to Expect

When the strategy starts, you will see this sequence in the logs. Each step confirms the system is healthy:

**Step 1 — Node and instruments load:**
```
[INFO] TITAN-ORB.TradingNode: BUILDING
[INFO] TITAN-ORB.TradingNode: RUNNING
```

**Step 2 — 7 strategies start, one per ticker:**
```
[INFO] TITAN-ORB.ORBStrategy: ORB Strategy Started for UNH. Parameters: ATR:2.0 RR:2.0 SMA:True Gauss:False RSI:True ORB:09:45:00 Cutoff:11:00:00
[INFO] TITAN-ORB.ORBStrategy: ORB Strategy Started for AMAT. Parameters: ATR:2.5 RR:1.5 SMA:True Gauss:True RSI:True ORB:09:40:00 Cutoff:11:00:00
... (5 more)
```

**Step 3 — 14 bar feeds subscribed (2 per ticker: 1D + 5M):**
```
[INFO] DataClient-INTERACTIVE_BROKERS: Subscribed UNH.NYSE-1-DAY-LAST-EXTERNAL bars
[INFO] DataClient-INTERACTIVE_BROKERS: Subscribed UNH.NYSE-5-MINUTE-LAST-EXTERNAL bars
... (12 more lines)
```

**Step 4 — Indicator warmup from historical data:**
```
[INFO] ORBStrategy: [UNH] Daily warmup complete. SMA50: 512.34
[INFO] ORBStrategy: [UNH] 5M warmup complete. ATR: 3.21, Gauss Mid: 509.10
```
If warmup data is missing you will see a warning — the strategy still runs but starts without pre-loaded indicators:
```
[WARN] ORBStrategy: No Daily historical data found at data/UNH_D.parquet for UNH
```

If all 4 steps complete without `[ERROR]` lines, the strategy is healthy and waiting for bars.

---

## Strategy Logic — How Trades Are Taken

Understanding the logic helps you interpret log output and know when to expect activity.

### Opening Range Formation (09:30–09:40 or 09:45 ET, per ticker)

The strategy watches 5-minute bars from 09:30 ET. It tracks the highest high and lowest low across all bars in the opening range window. At the final bar of the window (09:40 or 09:45 depending on the ticker), the Opening Range is locked:

```
[INTC] ORB Formed - High: 23.85, Low: 23.41
```

No trades are placed before this line appears.

### Entry Triggers (09:40/09:45 → 11:00 ET)

After the ORB is formed, each new 5-minute bar is checked for a breakout:

- **LONG trigger:** bar closes **above** the Opening Range High AND all enabled filters pass
- **SHORT trigger:** bar closes **below** the Opening Range Low AND all enabled filters pass

Filters per ticker (configured in `config/orb_live.toml`):

| Ticker | ORB End | SMA50 Filter | RSI14 Filter | Gaussian Filter |
|--------|---------|-------------|-------------|----------------|
| UNH | 09:45 | Yes | Yes | No |
| AMAT | 09:40 | Yes | Yes | Yes |
| TXN | 09:40 | Yes | Yes | Yes |
| INTC | 09:40 | No | Yes | Yes |
| CAT | 09:40 | Yes | Yes | Yes |
| WMT | 09:45 | Yes | No | Yes |
| TMO | 09:45 | Yes | Yes | Yes |

- **SMA50:** price must be above SMA50 for longs, below for shorts
- **RSI14:** RSI must be below 70 for longs, above 30 for shorts
- **Gaussian Channel:** price must be above the Gaussian midline for longs, below for shorts

When a trigger fires, you will see:
```
[AMAT] LONG Trigger: Close 187.42 > ORH 186.91. SMA=181.20, RSI=58.3, Gauss=185.10
```
Or if filters blocked it, there will be no log entry (the bar is simply skipped silently). Only one trade per ticker per day is taken.

Entry cutoff is 11:00 ET for all tickers — no new entries after that.

### Position Sizing

The strategy risks exactly **1% of account equity** per trade, capped at **4× leverage**.

```
Risk amount    = Account Equity × 1%
Stop distance  = ATR14 × ATR multiplier (per ticker)
Position size  = Risk amount ÷ Stop distance
             (capped at Equity × 4 ÷ entry price)
```

### Bracket Order Structure

Every trade is submitted as a **linked bracket order**:
- **Entry:** Market order (`TimeInForce.DAY`)
- **Stop Loss:** Stop order at `entry − (ATR × multiplier)` for longs (`TimeInForce.GTC`)
- **Take Profit:** Limit order at `entry + (ATR × multiplier × RR ratio)` for longs (`TimeInForce.GTC`)

NautilusTrader wires these as OTO (entry triggers TP+SL) and OCO (TP and SL cancel each other when one fills). You do not need to manage them manually.

> **Verified live (Mar 10 2026):** All 3 bracket legs submit, are accepted by IB with venue order IDs, and the entry fills correctly. On a **paper account with DELAYED_FROZEN data**, TP/SL fills may not simulate naturally mid-session (price feed is frozen for new subscriptions). On a **live account** with REALTIME data, TP and SL fill normally when the price crosses the level. The EOD flatten at 15:55 ET and the fallback cancel+close path are both confirmed working regardless of account type.

### End of Day Flatten (15:55 ET)

At 15:55 ET, all open positions and pending orders for each ticker are cancelled and closed automatically:
```
[UNH] EOD Flatten @ 15:55 ET — Closed all positions and cancelled orders.
```
This prevents overnight gap risk. The strategy holds no positions overnight.

---

## Reading the Logs — Key Events

Every order and position change is logged. Here is the full lifecycle of a trade from trigger to close:

```
# 1. Breakout detected
[AMAT] LONG Trigger: Close 187.42 > ORH 186.91. SMA=181.20, RSI=58.3, Gauss=185.10
[AMAT] Bracket submitted -> BUY 53 @ ~187.42
[AMAT] SL=182.17  TP=195.67  Risk/share=5.25  RR=1.5

# 2. Orders sent and confirmed by IB
[AMAT] ORDER SUBMITTED: O-20260310-000001-001-001-1
[AMAT] ORDER ACCEPTED: O-20260310-000001-001-001-1 venue_id=1024
[AMAT] ORDER SUBMITTED: O-20260310-000001-001-001-2   ← SL leg
[AMAT] ORDER ACCEPTED: O-20260310-000001-001-001-2 venue_id=1025
[AMAT] ORDER SUBMITTED: O-20260310-000001-001-001-3   ← TP leg
[AMAT] ORDER ACCEPTED: O-20260310-000001-001-001-3 venue_id=1026

# 3. Entry fills
[AMAT] ORDER FILLED: O-20260310-000001-001-001-1 side=BUY qty=53 px=187.48 commission=1.06

# 4. Position opened
[AMAT] POSITION OPENED: P-20260310-000001 side=BUY qty=53 avg_px=187.48

# 5a. Take Profit hit (winning trade)
[AMAT] ORDER FILLED: O-20260310-000001-001-001-3 side=SELL qty=53 px=195.67 commission=1.06
[AMAT] ORDER CANCELLED: O-20260310-000001-001-001-2   ← SL cancelled (OCO)
[AMAT] POSITION CLOSED: P-20260310-000001 realized_pnl=432.71 duration=2340s

# OR 5b. Stop Loss hit (losing trade)
[AMAT] ORDER FILLED: O-20260310-000001-001-001-2 side=SELL qty=53 px=182.17 commission=1.06
[AMAT] ORDER CANCELLED: O-20260310-000001-001-001-3   ← TP cancelled (OCO)
[AMAT] POSITION CLOSED: P-20260310-000001 realized_pnl=-278.31 duration=1800s

# OR 5c. End of day reached before TP/SL
[AMAT] EOD Flatten @ 15:55 ET — Closed all positions and cancelled orders.
```

### Log levels

| Level | Meaning |
|---|---|
| `[INFO]` | Normal operation — orders, fills, position events |
| `[WARN]` | Non-critical issue — e.g. missing warmup data, expired orders |
| `[ERROR]` | Problem requiring attention — rejected orders, sizing failures |

To filter for errors only:
```bash
# bash (Antigravity terminal)
grep "ERROR\|REJECTED\|WARN" .tmp/logs/orb_live_YYYYMMDD_HHMMSS.log
```

To see all fills and closed P&L for the day:
```bash
# bash (Antigravity terminal)
grep "ORDER FILLED\|POSITION CLOSED" .tmp/logs/orb_live_YYYYMMDD_HHMMSS.log
```

---

## Stopping the Strategy

### Normal Stop

Press `Ctrl+C` in the strategy terminal (Tab 1). NautilusTrader shuts down gracefully — it cancels all pending orders and logs a clean shutdown. Open positions are **left open** on the IB side, managed by the existing SL/TP bracket orders. This is the correct way to stop at end of day after 15:55.

### Force Kill (if the process is stuck)

If `Ctrl+C` does not work, find and kill the process:

```bash
# bash (Antigravity terminal) — find the PID
python -c "
import subprocess
r = subprocess.run(['wmic', 'process', 'where', 'name=\"python.exe\"', 'get', 'ProcessId,CommandLine', '/value'], capture_output=True, text=True)
print(r.stdout)
"
```

Look for the line containing `run_live_orb.py` and note its PID. Then kill it:

```bash
# bash (Antigravity terminal)
python -c "import subprocess; subprocess.run(['taskkill', '/F', '/PID', 'REPLACE_WITH_PID'])"
```

> **Important:** After a force kill, open positions remain open on IB's side. Log into TWS/Gateway to verify position status and manually close if needed.

---

## Restarting

Before restarting after a crash or force kill, always check that the old process is dead — otherwise you will get Error 326 (client ID already in use):

```bash
# bash (Antigravity terminal)
python -c "
import subprocess
r = subprocess.run(['tasklist'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'python' in line.lower():
        print(line)
"
```

If you see multiple Python processes, check which ones are ORB-related (as above) and kill them before restarting.

---

## Common Errors and Fixes

### Error 326 — Client ID already in use
```
ERROR: 326 Unable to connect as the client id is already in use.
```
An old strategy process is still holding the socket. Find and kill it (see Force Kill above), then restart.

### Error 321 — Read-Only API
```
ERROR: 321 API interface is currently in Read-Only mode.
```
In TWS/Gateway: **Edit → Global Configuration → API → Settings → uncheck "Read-Only API"**. Orders will be silently rejected until this is fixed.

### No fills / strategy seems idle

This is normal. The strategy only trades when:
1. The ORB is formed (after 09:40 or 09:45 ET depending on ticker)
2. A bar closes beyond the Opening Range High/Low
3. All configured filters pass (SMA, RSI, Gaussian Channel)
4. The time is before 11:00 ET

If it is before 09:45 ET you will not see any trade logs — the strategy is waiting for the ORB to form.

### `[ERROR] No configuration found for TICKER`

The ticker key is missing from `config/orb_live.toml`. The strategy will not trade that instrument. Add a `[TICKER]` section to the TOML file.

### `[WARN] No Daily historical data found`

The warmup parquet file for the ticker is missing from `data/`. The strategy runs without pre-loaded indicators — SMA and RSI will not be valid until enough live bars accumulate. Run `python scripts/download_data.py` before the next session to fix this.

### `[ERROR] Sizing calculated 0 units`

The ATR value was too large relative to account equity, making the risk-adjusted position size round down to zero. The trade is skipped. This usually means the ATR multiplier is set too high in `config/orb_live.toml` relative to the current account size.

### `[ERROR] ORDER REJECTED`

IB rejected an order. The reason string will follow in the same log line. Common causes:
- Market is not open yet
- Instrument is halted
- Account lacks buying power
- Order parameters out of range

When an entry order is rejected, `trade_taken_today` is reset so the strategy can attempt another entry on the next bar if conditions still hold.

---

## Checking Account Balance Mid-Session

You can check the account balance at any time without disturbing the running strategy. This script uses a separate client ID and does not conflict:

```bash
# bash (Antigravity terminal) — new tab, strategy keeps running
python scripts/check_balance.py
```

---

## Configuration Reference

### `.env` — Connection Settings
```ini
IBKR_HOST=127.0.0.1        # Gateway/TWS host (local)
IBKR_PORT=4002             # 4002=Gateway Paper, 4001=Gateway Live, 7497=TWS Paper, 7496=TWS Live
IBKR_CLIENT_ID=20          # Socket client ID for the ORB strategy — must be unique per process
IBKR_ACCOUNT_ID=DUxxxxxxx  # Your IB paper/live account ID
```

### `config/orb_live.toml` — Per-Ticker Strategy Parameters
```toml
[UNH]
atr_multiplier = 2.0    # Stop distance = ATR × this value
rr_ratio = 2.0          # Take profit = stop distance × this value
use_sma = true          # Filter: price must be above/below SMA50
use_rsi = true          # Filter: RSI14 must be < 70 (long) or > 30 (short)
use_gauss = false       # Filter: price must be above/below Gaussian midline
orb_window_end = "09:45"   # Opening range locks at this time (ET)
entry_cutoff = "11:00"     # No new entries after this time (ET)
```

### `scripts/run_live_orb.py` — Runner Settings
```python
risk_pct=0.01      # 1% of equity risked per trade
leverage_cap=4.0   # Maximum 4× leverage cap on position size
warmup_bars_1d=60  # Daily bars loaded from parquet for SMA/RSI warmup
warmup_bars_5m=200 # 5M bars loaded for ATR/Gaussian warmup (Gaussian needs 144+)
```

---

## Log File Location

All log files are written to:
```
.tmp/logs/orb_live_YYYYMMDD_HHMMSS.log
```

Log files are timestamped at startup. Each new run of the strategy creates a new file. Old files accumulate — clean them up periodically:

```bash
# bash (Antigravity terminal) — delete logs older than 7 days
find .tmp/logs/ -name "*.log" -mtime +7 -delete
```
