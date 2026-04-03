# Deployment & Operations Guide

> Last updated: 2026-04-03
> Merged from: Deployment Options.md, VPS Deployment Guide.md, Live Deployment and Monitoring.md

---

## 1. Pre-Flight Checklist

> [!CAUTION]
> **All items must pass before going live.**

- [ ] `config/risk.toml` reviewed -- max drawdown, position limits, daily loss cap
- [ ] `scripts/kill_switch.py` tested on paper account (`uv run python scripts/kill_switch.py --dry-run`)
- [ ] OOS Sharpe >= 50% of IS Sharpe for each strategy being deployed
- [ ] Paper trading session completed -- at least one trade opened and closed cleanly
- [ ] IBKR Gateway running on correct port, API settings confirmed (Read-Only API unchecked)
- [ ] No orphaned stop orders visible in TWS order panel after a clean shutdown
- [ ] PortfolioRiskManager registers all strategies and halt check is working

---

## 2. Deployment Method Comparison

| Feature | **Systemd** (recommended) | **Docker** | **Tmux** |
|---|---|---|---|
| Best for | Production VPS | Multi-strategy scale-out | Testing / first 48h monitoring |
| Reliability | Native OS service, dependency ordering | Isolated environment | No auto-restart |
| IB Gateway headless | Native (Xvfb + IBC) | Complex (display forwarding) | Native |
| Auto-start on boot | Yes (`systemctl enable`) | Yes (`--restart always`) | No |
| Graceful shutdown | SIGTERM -> watchdog -> no-restart | SIGTERM -> container stop | Ctrl+C |
| Log access | `journalctl -u mtf-watchdog` | `docker logs titan` | Live in terminal |
| Setup complexity | Medium | High | Low |

---

## 3. Local (Practice) Deployment

### Configuration

`.env` for paper:
```ini
IBKR_PORT=7497         # TWS paper (or 4002 for IB Gateway paper)
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### Run strategies

**Recommended: use the watchdog**, not the runner directly. The watchdog handles
the IB nightly maintenance window (23:45-00:05 ET) automatically.

```bash
# MTF EUR/USD
uv run python scripts/watchdog_mtf.py 2>&1 | tee .tmp/logs/watchdog_$(date +%Y%m%d).log

# ORB (7 equities)
uv run python scripts/run_live_orb.py 2>&1 | tee .tmp/logs/orb_$(date +%Y%m%d).log

# ETF Trend (7 ETFs)
uv run python scripts/run_live_etf_trend.py

# IC Equity Daily
# (started via ic_mtf runner or standalone)

# ML Classifier
uv run python scripts/run_live_ml.py
```

### Expected startup sequence (MTF example)

```
[watchdog] Attempt 1: Starting MTF strategy...
  MTF Strategy Started. Warming up...
    Loading H1 warmup from data/EUR_USD_H1.parquet
    Loading H4 warmup from data/EUR_USD_H4.parquet
    Loading D  warmup from data/EUR_USD_D.parquet
    Loading W  warmup from data/EUR_USD_W.parquet
  Warmup complete. ATR stop mult: 4.0x. Ready for signals.
```

If MA/RSI values show `??` -- warmup parquet is missing:
```bash
uv run python scripts/download_data_mtf.py --pair EUR_USD --years 10
```

### Paper success criteria

Before switching to live:
- [ ] Startup sequence completes with no `[ERROR]` lines
- [ ] Dashboard shows populated indicator values (not `??`) on every bar
- [ ] At least one trade opens and closes cleanly
- [ ] No orphaned stop orders after a clean shutdown
- [ ] Watchdog restarts cleanly after killing the runner manually
- [ ] `Ctrl+C` on watchdog propagates SIGTERM and exits without restarting

---

## 4. VPS (Production) Deployment

### VPS Specification

| Item | Recommendation |
|---|---|
| Provider | Hetzner CX22 (~4 EUR/mo), DigitalOcean Basic, Vultr |
| OS | Ubuntu 22.04 LTS |
| CPU | 2 vCPU |
| RAM | 4 GB |
| Storage | 40 GB SSD |

Latency is irrelevant for daily/H4 strategies. Pick the cheapest datacenter.

### Architecture

```
Ubuntu 22.04 VPS
  ibgateway.service    (IB Gateway headless via Xvfb + IBC auto-login)
      |
      Requires
      |
  mtf-watchdog.service (scripts/watchdog_mtf.py)
      |
      spawns subprocess
      |
  scripts/run_live_mtf.py  (restarted by watchdog on crash/disconnect)
```

### Step 1 -- VPS initial setup

```bash
adduser titan
usermod -aG sudo titan
su - titan

sudo apt update && sudo apt install -y \
    git curl unzip python3.11 python3.11-venv \
    xvfb x11vnc default-jre mailutils
```

### Step 2 -- Install uv and clone repo

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/YOUR_USERNAME/Titan-IBKR.git /home/titan/Titan-IBKR
cd /home/titan/Titan-IBKR && uv sync

# Copy .env from local machine
scp .env titan@YOUR_VPS_IP:/home/titan/Titan-IBKR/.env
```

`.env` for live:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4001        # Live Gateway port (4002 = paper)
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_ID=Uxxxxxxx
```

### Step 3 -- Install IB Gateway + IBC

```bash
# Download IB Gateway from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
chmod +x ibgateway-*.sh && ./ibgateway-*.sh

# Download IBC from: https://github.com/IbcAlpha/IBC/releases
unzip IBCLinux-*.zip -d /opt/ibc
chmod +x /opt/ibc/scripts/*.sh
cp /opt/ibc/config.ini /opt/ibc/config_live.ini
```

Edit `/opt/ibc/config_live.ini`:
```ini
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=live
AcceptIncomingConnectionAction=accept
ReadOnlyApi=no
SocketPort=4001
```

> [!CAUTION]
> Plaintext credentials. Set `chmod 600 /opt/ibc/config_live.ini` and block
> port 4001 from internet via firewall.

Test startup:
```bash
DISPLAY=:1 Xvfb :1 -screen 0 1024x768x24 &
DISPLAY=:1 /opt/ibc/scripts/ibcstart.sh live
```

### Step 4 -- systemd services

**ibgateway.service** (`/etc/systemd/system/ibgateway.service`):
```ini
[Unit]
Description=IB Gateway (headless via IBC + Xvfb)
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
User=titan
Environment=HOME=/home/titan
Environment=DISPLAY=:1
ExecStartPre=/usr/bin/Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
ExecStart=/opt/ibc/scripts/ibcstart.sh live
Restart=on-failure
RestartSec=60s
TimeoutStartSec=120s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ibgateway

[Install]
WantedBy=multi-user.target
```

**mtf-watchdog.service** (`/etc/systemd/system/mtf-watchdog.service`):
```ini
[Unit]
Description=MTF Confluence Strategy Watchdog
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=titan
WorkingDirectory=/home/titan/Titan-IBKR
ExecStart=/home/titan/Titan-IBKR/.venv/bin/python scripts/watchdog_mtf.py
Restart=on-failure
RestartSec=30s
KillSignal=SIGTERM
TimeoutStopSec=90s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mtf-watchdog

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ibgateway mtf-watchdog
sudo systemctl start ibgateway && sleep 90 && sudo systemctl start mtf-watchdog
sudo systemctl status ibgateway mtf-watchdog
```

### Step 5 -- Firewall

```bash
sudo ufw allow 22/tcp
sudo ufw deny 4001/tcp   # Block Gateway from internet
sudo ufw deny 4002/tcp
sudo ufw enable
```

### Step 6 -- Telegram alerting (recommended)

```python
# scripts/alert.py
import os, requests
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
def send(msg: str) -> None:
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  json={"chat_id": CHAT_ID, "text": msg}, timeout=5)
```

Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env`.

### Pre-live checklist (VPS-specific)

- [ ] `systemctl status ibgateway` shows `active (running)` for > 24 hours
- [ ] Strategy log shows `Warmup complete` with no `[ERROR]` lines
- [ ] At least one trade opened and closed cleanly on paper
- [ ] `kill_switch.py` tested on paper -- confirmed it flattens positions
- [ ] `.env` updated to `IBKR_PORT=4001` (live) and live `IBKR_ACCOUNT_ID`
- [ ] Firewall confirmed: port 4001 blocked from internet
- [ ] Telegram/email alert tested and working

---

## 5. Docker (Alternative)

```bash
docker run -d --name mtf-live --restart always --env-file .env \
  -v $(pwd)/data:/app/data -v $(pwd)/.tmp/logs:/app/.tmp/logs \
  titan-ibkr-algo python scripts/watchdog_mtf.py
```

> [!IMPORTANT]
> Container CMD should be `watchdog_mtf.py`, not `run_live_mtf.py`.
> Docker `--restart always` handles container crashes; watchdog handles IB disconnects.

---

## 6. Tmux (Testing only)

```bash
tmux new -s titan
.venv/bin/python scripts/watchdog_mtf.py
# Detach: Ctrl+B then D
# Re-attach: tmux attach -t titan
```

> [!WARNING]
> Tmux does **not** auto-restart on VPS reboot. Use for first 24-48h only, then migrate to systemd.

---

## 7. Monitoring & Operations

### Log monitoring

```bash
# Systemd logs
journalctl -u mtf-watchdog -f
journalctl -u ibgateway -f

# Strategy logs
tail -f .tmp/logs/mtf_live_*.log

# Restart history
grep "Attempt" .tmp/logs/watchdog.log | tail -20

# Check for errors
grep -i "error\|exception\|rejected\|failed" .tmp/logs/mtf_live_*.log | tail -20

# Current positions (dry-run)
uv run python scripts/kill_switch.py --dry-run
```

### Key log strings
- `ATR stop placed @ ...` -- stop order active
- `Signal Flip: Long -> Short` -- primary exit + reversal
- `[watchdog] Process exited with code X after Ys` -- watchdog restarting
- `[PortfolioRM] Portfolio heat: DD=X%` -- risk manager scaling down

### Health check cron

```bash
*/5 * * * * systemctl is-active --quiet mtf-watchdog || \
    echo "TITAN: mtf-watchdog DOWN at $(date)" | mail -s "TITAN ALERT" you@email.com
```

### Operational commands

```bash
# Status
sudo systemctl status mtf-watchdog ibgateway

# Graceful stop
sudo systemctl stop mtf-watchdog

# Emergency stop
.venv/bin/python scripts/kill_switch.py && sudo systemctl stop mtf-watchdog

# Deploy update
git pull && uv sync && sudo systemctl restart mtf-watchdog
```

---

## 8. Nightly Maintenance & Weekends

**Nightly maintenance** (23:45-00:05 ET): IB Gateway disconnects. Watchdog handles
this automatically -- 3-minute restart delay covers the window. No human intervention.

**Weekends**: FX markets close Friday ~5 pm ET, reopen Sunday ~5 pm ET. IB Gateway
stays connected. Strategy process runs but receives no bar events. Normal behaviour.

---

## 9. Emergency Stop

> [!CAUTION]
> Use the kill switch for any position emergency. It does NOT require
> NautilusTrader or the strategy process to be running.

```bash
# Cancel all open orders and close all positions immediately
uv run python scripts/kill_switch.py

# Then stop the watchdog so the strategy doesn't restart
sudo systemctl stop mtf-watchdog    # on VPS
# or Ctrl+C on the watchdog process  # local
```

The kill switch connects directly to IB Gateway via the IBKR API, independent
of the strategy process. See `directives/MTF Strategy User Guide.md` for
MTF-specific emergency procedures.
