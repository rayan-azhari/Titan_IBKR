# VPS Deployment Guide — MTF Confluence Strategy

> Last updated: 2026-03-16

This guide covers end-to-end autonomous deployment of the MTF strategy on a Linux VPS,
including IB Gateway headless setup, systemd service management, and monitoring.

---

## Architecture Overview

```
Ubuntu 22.04 VPS
├── IB Gateway (headless via Xvfb + IBC auto-login)
│     └─ systemd: ibgateway.service  (auto-starts on boot, auto-restarts on crash)
│
└── MTF Watchdog (scripts/watchdog_mtf.py)
      └─ systemd: mtf-watchdog.service  (depends on ibgateway, auto-restarts)
            └─ spawns subprocess: scripts/run_live_mtf.py
                    └─ restarts with 3-min delay after IB nightly maintenance window
```

The strategy is **fully autonomous**:
- IB Gateway logs in automatically on boot via IBC
- The watchdog restarts the strategy after any crash or disconnect
- The 3-min restart delay in `watchdog_mtf.py` covers the IB nightly maintenance window (23:45–00:05 ET)
- SIGTERM propagates cleanly: `systemctl stop mtf-watchdog` → watchdog sets `_shutdown=True`
  → waits for current strategy process → exits without restarting

---

## VPS Specification

| Item | Recommendation |
|---|---|
| Provider | Hetzner CX22 (~€4/mo), DigitalOcean Basic, Vultr |
| OS | Ubuntu 22.04 LTS |
| CPU | 2 vCPU |
| RAM | 4 GB |
| Storage | 40 GB SSD |
| Region | Any — EUR/USD on H4 bars has no meaningful latency requirement |

> [!NOTE]
> Latency is irrelevant for this strategy. Signals fire on 4H bar closes; a 50ms
> ping to IBKR changes nothing. Pick the cheapest datacenter.

---

## Step 1 — VPS Initial Setup

```bash
# Create dedicated user (never run as root)
adduser titan
usermod -aG sudo titan
su - titan

# Install system packages
sudo apt update && sudo apt install -y \
    git curl unzip \
    python3.11 python3.11-venv \
    xvfb x11vnc default-jre \
    mailutils  # optional, for email alerts
```

---

## Step 2 — Install uv and Clone Repo

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the repo
git clone https://github.com/YOUR_USERNAME/Titan-IBKR.git /home/titan/Titan-IBKR
cd /home/titan/Titan-IBKR

# Install Python dependencies
uv sync

# Copy .env from local machine
scp .env titan@YOUR_VPS_IP:/home/titan/Titan-IBKR/.env
```

`.env` should point to the local IB Gateway:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4001        # Live Gateway port (4002 = paper)
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_ID=Uxxxxxxx
```

---

## Step 3 — Install IB Gateway + IBC

**IB Gateway** is IBKR's headless trading client (lighter than TWS, no GUI needed).
**IBC** (open-source) handles automatic login so no human is needed on restart.

```bash
# 1. Download IB Gateway from IBKR website
#    https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
#    Choose: Linux installer (standalone)

chmod +x ibgateway-10.29.1-standalone-linux-x64.sh
./ibgateway-10.29.1-standalone-linux-x64.sh   # installs to ~/Jts/ibgateway/

# 2. Download and install IBC
#    https://github.com/IbcAlpha/IBC/releases
unzip IBCLinux-3.18.0.zip -d /opt/ibc
chmod +x /opt/ibc/scripts/*.sh

# 3. Configure IBC credentials
cp /opt/ibc/config.ini /opt/ibc/config_live.ini
```

Edit `/opt/ibc/config_live.ini`:
```ini
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=live          # or 'paper' for paper trading
AcceptIncomingConnectionAction=accept
ReadOnlyApi=no
SocketPort=4001
```

> [!CAUTION]
> The IBC config file contains plaintext credentials. Set permissions:
> `chmod 600 /opt/ibc/config_live.ini`
> and ensure the VPS has firewall rules blocking port 4001 from the internet.

Test IBC + Gateway starts correctly:
```bash
DISPLAY=:1 Xvfb :1 -screen 0 1024x768x24 &
DISPLAY=:1 /opt/ibc/scripts/ibcstart.sh live
# Should see: "IB Gateway started" in logs — takes ~30-60s
```

---

## Step 4 — systemd Service: IB Gateway

Create `/etc/systemd/system/ibgateway.service`:

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

# Start virtual display first
ExecStartPre=/usr/bin/Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &

# Launch IB Gateway via IBC (handles auto-login)
ExecStart=/opt/ibc/scripts/ibcstart.sh live

Restart=on-failure
RestartSec=60s

# Give Gateway time to authenticate on restart
TimeoutStartSec=120s

StandardOutput=journal
StandardError=journal
SyslogIdentifier=ibgateway

[Install]
WantedBy=multi-user.target
```

---

## Step 5 — systemd Service: MTF Watchdog

Create `/etc/systemd/system/mtf-watchdog.service`:

```ini
[Unit]
Description=MTF Confluence Strategy Watchdog
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=titan
WorkingDirectory=/home/titan/Titan-IBKR

# Use the uv-managed venv python directly
ExecStart=/home/titan/Titan-IBKR/.venv/bin/python scripts/watchdog_mtf.py

# Restart the watchdog itself if it crashes (unusual but possible)
Restart=on-failure
RestartSec=30s

# SIGTERM → watchdog._shutdown = True → graceful exit without restart
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

# Start Gateway first, wait for it to authenticate (~60s), then start watchdog
sudo systemctl start ibgateway
sleep 90
sudo systemctl start mtf-watchdog

# Verify both are running
sudo systemctl status ibgateway mtf-watchdog
```

---

## Step 6 — Firewall

IB Gateway should only be reachable from localhost:

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Block Gateway port from internet (only localhost needs it)
sudo ufw deny 4001/tcp
sudo ufw deny 4002/tcp

sudo ufw enable
```

---

## Monitoring

### View live logs

```bash
# Watchdog restart log
journalctl -u mtf-watchdog -f

# IB Gateway log
journalctl -u ibgateway -f

# Strategy detailed log (written by run_live_mtf.py)
tail -f /home/titan/Titan-IBKR/.tmp/logs/mtf_live_*.log | head -100
# or: latest log file only
tail -f $(ls -t /home/titan/Titan-IBKR/.tmp/logs/mtf_live_*.log | head -1)
```

### Health check cron

Add to `crontab -e` (runs as `titan` user):

```bash
# Alert if mtf-watchdog is not active — check every 5 minutes
*/5 * * * * systemctl is-active --quiet mtf-watchdog || \
    echo "TITAN ALERT: mtf-watchdog is DOWN at $(date)" | \
    mail -s "TITAN ALERT" your@email.com
```

### Telegram alerting (recommended over email)

```python
# scripts/alert.py — send a Telegram message
import os, requests
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def send(msg: str) -> None:
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg},
        timeout=5,
    )
```

Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env`. The watchdog can be extended
to call `send()` on each restart event.

---

## Operational Commands

```bash
# Check status
sudo systemctl status mtf-watchdog ibgateway

# Graceful stop (no position closed — let the strategy finish its current bar)
sudo systemctl stop mtf-watchdog

# Emergency stop (cancel all orders + flatten positions, then stop)
cd /home/titan/Titan-IBKR
.venv/bin/python scripts/kill_switch.py
sudo systemctl stop mtf-watchdog

# Restart after config change
sudo systemctl restart mtf-watchdog

# View recent restarts
grep "Attempt" /home/titan/Titan-IBKR/.tmp/logs/watchdog.log | tail -20

# Deploy a code update
cd /home/titan/Titan-IBKR
git pull
uv sync
sudo systemctl restart mtf-watchdog
```

---

## Nightly Maintenance Window

IB Gateway goes offline for maintenance nightly between approximately **23:45–00:05 ET**.
The watchdog handles this automatically:

1. Gateway disconnects → NautilusTrader raises a disconnect error
2. `run_live_mtf.py` exits (non-zero exit code)
3. Watchdog logs: `Process exited with code X after Ys`
4. Watchdog waits **180 seconds** (`RESTART_DELAY_SECS` in `watchdog_mtf.py`)
5. Gateway has finished its restart by then → watchdog re-launches `run_live_mtf.py`
6. Strategy reconnects, reloads warmup data, resumes trading

No human intervention required.

---

## Weekend Behaviour

FX markets close Friday ~5 pm ET, reopen Sunday ~5 pm ET. IB Gateway stays connected
over the weekend. The strategy process runs continuously but receives no bar events
while the market is closed. This is normal — no action needed.

---

## Pre-Live Checklist

> [!CAUTION]
> Complete all items before switching from paper to live.

- [ ] `systemctl status ibgateway` shows `active (running)` for > 24 hours
- [ ] `systemctl status mtf-watchdog` shows `active (running)`
- [ ] Strategy log shows `Warmup complete` with no `[ERROR]` lines
- [ ] At least one trade opened and closed cleanly on paper
- [ ] `kill_switch.py` tested on paper — confirmed it flattens positions
- [ ] `.env` updated to `IBKR_PORT=4001` (live Gateway port) and live `IBKR_ACCOUNT_ID`
- [ ] `/opt/ibc/config_live.ini` updated with live account credentials
- [ ] Firewall confirmed: port 4001 blocked from internet
- [ ] Telegram/email alert tested and working
