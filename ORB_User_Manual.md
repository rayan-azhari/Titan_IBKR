# Titan ORB Live Trading Manual

This manual explains how to run the updated 5-Minute Opening Range Breakout (ORB) Strategy, which now takes advantage of a **Gaussian Channel** filter to improve Sharpe ratios and filter out choppy entries.

## Strategy Overview
- **Symbols traded:** UNH, AMAT, TXN, INTC, CAT, WMT, TMO.
- **Data Subscriptions:** Live 1-Day & 5-Minute internal bars via Interactive Brokers.
- **Entry Rules:** 
  - Opening Range defined from `09:30` to `09:40`/`09:45` (varies by ticker).
  - Price must break out of the Opening Range before `11:00 AM`.
  - The breakout direction must align with the Daily SMA50, Daily RSI14, and the 5-Minute Gaussian Channel (Close vs Midline).
- **Execution:** Full Bracket Orders (Market Entry, Stop Loss, Take Profit).
- **Position Sizing:** 1% of total account Equity, risked via the ATR Stop Loss distance.

---

## 🚀 How to Start the Strategy

### 1. Preparing the Environment
Ensure your local `.env` file at the project root is properly configured for the account you wish to trade:
```env
IBKR_HOST=127.0.0.1
IBKR_PORT=7497           # 7497 for Paper, 7496 for Live (if using TWS)
IBKR_CLIENT_ID=2         # Must be a unique ID
IBKR_ACCOUNT_ID=DUXXXXX  # Your IBKR Paper Account Number
```

### 2. Pre-loading Historical Data
To avoid any pricing discrepancies that could cause indicator misalignment, the strategy strictly warms up using **Interactive Brokers' own pristine historical data.**

Before launching the strategy each morning, quickly run the data downloader to cache the latest pre-market bars to your local drive. Because it queries your live IBKR socket, it perfectly matches the live execution feed:
```bash
python scripts/download_data.py
```
*(This fetches the required 5-Minute and Daily bars for all configured pairs and ORB stocks into the `data/` folder).*

### 3. Launching the Live Node
With TWS or the IBKR Gateway running locally:
```bash
python scripts/run_live_orb.py
```
You will see the node connect, load the local parquet files, instantly calculate the Gaussian Channel and SMA/RSI indicators up to the current moment, and then wait for live 5-minute bar closes.

---

## 🛑 How to Stop the Strategy

### Normal Shutdown
If you need to stop the strategy (e.g., at the end of the day or to restart the application):
1. Navigate to the terminal running `run_live_orb.py`.
2. Press **`CTRL + C`**.
3. Nautilus will catch the signal, gracefully disconnect from the IBKR feeds, and shut down the Trading Node safely. 
> *Note: Shutting down the node does **not** close your open positions at IBKR or cancel your open Bracket limits/stops.*

### Emergency Kill Switch
If the node is completely stuck, hangs, or you need an immediate total halt to all Titan trading across MTF and ORB:
Open a secondary terminal in the project directory and run:
```bash
python scripts/kill_switch.py
```
*(This closes all positions and cancels all pending orders across the active IBKR account.)*

---

## 📊 Monitoring

- **Console:** The `run_live_orb.py` process will output raw log data, including signal formations ("`ORB Formed - High: 150.2, Low: 149.8`") and bracket executions.
- **Log Files:** Everything is permanently recorded in `.tmp/logs/orb_live_<timestamp>.log`.
- **IBKR TWS/Gateway:** Bracket orders will appear instantly in your TWS Orders window once triggered.
