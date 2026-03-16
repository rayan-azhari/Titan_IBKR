# Directive: Live Deployment and Monitoring

> Last updated: 2026-03-16

---

## Pre-Flight Checklist

> [!CAUTION]
> **All items must pass before going live.**

- [ ] `config/risk.toml` reviewed — max drawdown, position limits, daily loss cap
- [ ] `scripts/kill_switch.py` tested on paper account (`uv run python scripts/kill_switch.py --dry-run`)
- [ ] OOS Sharpe ≥ 50% of IS Sharpe (MTF EUR/USD: Combined OOS 1.943 ✓)
- [ ] Paper trading session completed — at least one trade opened and closed cleanly
- [ ] IBKR Gateway running on correct port, API settings confirmed (Read-Only API unchecked)
- [ ] No orphaned stop orders visible in TWS order panel after a clean shutdown

---

## Local (Practice) Deployment

Always start on paper. Verify the full startup → trade → shutdown cycle before
switching to a live account.

### Configuration

`.env` for paper:
```ini
IBKR_PORT=7497         # TWS paper (or 4002 for IB Gateway paper)
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### Run the strategy

**Recommended: use the watchdog**, not the runner directly. The watchdog handles
the IB nightly maintenance window (23:45–00:05 ET) automatically.

```bash
# MTF EUR/USD — primary live strategy
uv run python scripts/watchdog_mtf.py 2>&1 | tee .tmp/logs/watchdog_$(date +%Y%m%d_%H%M%S).log

# ORB strategy (7 large-cap equities)
uv run python scripts/run_live_orb.py 2>&1 | tee .tmp/logs/orb_$(date +%Y%m%d_%H%M%S).log
```

If you want to run the strategy directly (no auto-restart):
```bash
uv run python scripts/run_live_mtf.py
```

### Expected startup sequence (MTF)

```
[watchdog] Attempt 1: Starting MTF strategy...
  Checking for latest data...
  Data download finished.
  MTF Strategy Started. Warming up...
    Loading H1 warmup from data/EUR_USD_H1.parquet  (134,962 bars)
    Loading H4 warmup from data/EUR_USD_H4.parquet  (35,923 bars)
    Loading D  warmup from data/EUR_USD_D.parquet   (6,412 bars)
    Loading W  warmup from data/EUR_USD_W.parquet   (1,154 bars)
  Warmup complete. ATR stop mult: 4.0x. Ready for signals.
```

If MA/RSI values show `??` — warmup parquet is missing. Run:
```bash
uv run python scripts/download_data_mtf.py --pair EUR_USD --years 10
```

### Log monitoring

```bash
# bash
tail -f .tmp/logs/mtf_live_$(ls -t .tmp/logs/mtf_live_*.log | head -1 | xargs basename)

# PowerShell
Get-Content .tmp/logs/mtf_live_*.log -Wait -Tail 50
```

Key log strings:
- `ATR stop placed @ 1.08234 (4.0x ATR = 0.00250)` — stop order active
- `ATR stop triggered. Fill @ ...` — hard stop fired; next bar re-evaluates
- `Signal Flip: Long -> Short` — primary exit + reversal
- `Signal Neutral. Closing position.` — primary exit, going flat
- `[watchdog] Process exited with code 0 after 843s` — watchdog is about to restart

---

## VPS (Autonomous Production) Deployment

For fully autonomous operation, deploy the strategy on a Linux VPS using systemd.
The watchdog + systemd stack provides:
- Auto-start on VPS reboot
- Auto-restart after any crash or IB disconnect
- Graceful shutdown via SIGTERM (no orphaned positions)

**Full setup instructions:** `directives/VPS Deployment Guide.md`

**Quick reference:**

```bash
# Deploy update from local machine
ssh titan@YOUR_VPS_IP
cd ~/Titan-IBKR && git pull && uv sync
sudo systemctl restart mtf-watchdog

# Check status
sudo systemctl status mtf-watchdog ibgateway

# View live logs
journalctl -u mtf-watchdog -f
```

---

## Emergency Stop

> [!CAUTION]
> Use the kill switch for any position emergency. It does **not** require
> NautilusTrader or the strategy process to be running.

```bash
# Cancel all open orders and close all positions immediately
uv run python scripts/kill_switch.py

# Then stop the watchdog so the strategy doesn't restart
sudo systemctl stop mtf-watchdog    # on VPS
# or Ctrl+C on the watchdog process  # local
```

The kill switch connects directly to IB Gateway via the IBKR API, independent
of the strategy. See `directives/MTF Strategy User Guide.md` for MTF-specific
emergency procedures.

---

## Monitoring Health

```bash
# Restart history (how many times has the watchdog restarted the strategy?)
grep "Attempt" .tmp/logs/watchdog.log

# Check for errors in the most recent strategy log
grep -i "error\|exception\|rejected\|failed" .tmp/logs/mtf_live_*.log | tail -20

# Current open positions (via IBKR)
uv run python scripts/kill_switch.py --dry-run   # shows positions without acting
```

Automated health check (cron, every 5 minutes):
```bash
*/5 * * * * systemctl is-active --quiet mtf-watchdog || \
    echo "TITAN: mtf-watchdog DOWN at $(date)" | mail -s "TITAN ALERT" you@email.com
```

---

## Success Criteria for Paper Session

Before switching to live:
- [ ] Startup sequence completes with no `[ERROR]` lines
- [ ] Dashboard shows populated indicator values (not `??`) on every bar
- [ ] At least one trade opens and closes cleanly
- [ ] No orphaned stop orders after a clean shutdown (check TWS order panel)
- [ ] Watchdog restarts cleanly after killing `run_live_mtf.py` manually
- [ ] `Ctrl+C` on watchdog propagates SIGTERM and exits without restarting
