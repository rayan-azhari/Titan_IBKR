# Deployment Options: Systemd vs Docker vs Tmux

> Last updated: 2026-03-16

---

## Comparison Matrix

| Feature | **Systemd** ⚙️ | **Docker** 🐳 | **Tmux** 🖥️ |
|---|---|---|---|
| **Best for** | Production VPS (recommended) | Multi-strategy scale-out | Testing / first 48h monitoring |
| **Reliability** | ★★★★★ — native OS service, IB Gateway dependency ordering | ★★★★★ — isolated environment | ★★ — no auto-restart |
| **IB Gateway headless** | Native (Xvfb + IBC) | Complex (display forwarding in container) | Native |
| **Auto-start on boot** | Yes (`systemctl enable`) | Yes (`--restart always`) | No |
| **Graceful shutdown** | SIGTERM → watchdog → no-restart | SIGTERM → container stop | Ctrl+C |
| **Log access** | `journalctl -u mtf-watchdog` | `docker logs titan` | Live in terminal |
| **Setup complexity** | Medium | High (IB Gateway in Docker is painful) | Low |

---

## Recommended: Systemd on VPS

The **watchdog + systemd** stack is the production standard for this codebase.

```
ibgateway.service   (IB Gateway via IBC auto-login + Xvfb)
      ↓ Requires
mtf-watchdog.service   (scripts/watchdog_mtf.py)
      ↓ spawns subprocess
scripts/run_live_mtf.py   (restarted by watchdog on any crash/disconnect)
```

**Full setup instructions:** `directives/VPS Deployment Guide.md`

Quick reference — service unit skeleton:

```ini
# /etc/systemd/system/mtf-watchdog.service
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

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ibgateway mtf-watchdog
sudo systemctl start ibgateway && sleep 90 && sudo systemctl start mtf-watchdog
```

---

## Docker (Multi-Strategy Scale-Out)

Docker is the right choice if you are running multiple strategies simultaneously
and want environment isolation between them. IB Gateway headless in Docker requires
a virtual display passthrough, which adds complexity.

```bash
# Build
uv run python scripts/build_docker.py

# Run with restart policy
docker run -d \
  --name mtf-live \
  --restart always \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.tmp/logs:/app/.tmp/logs \
  titan-ibkr-algo \
  python scripts/watchdog_mtf.py
```

> [!IMPORTANT]
> The container's `CMD` should be `watchdog_mtf.py`, not `run_live_mtf.py`.
> Docker's `--restart always` handles container-level crashes; the watchdog
> handles IB disconnect/maintenance-window restarts within the container.

---

## Tmux (Testing / First 48h)

Use tmux when you want to **watch the live logs interactively** during initial
deployment before handing off to systemd.

```bash
# Start a named session
tmux new -s titan

# Inside the session — start the watchdog (not run_live_mtf.py directly)
cd ~/Titan-IBKR
.venv/bin/python scripts/watchdog_mtf.py

# Detach (leave running): Ctrl+B then D
# Re-attach later: tmux attach -t titan
```

> [!WARNING]
> Tmux does **not** auto-restart if the server reboots. Use it for the first
> 24–48 hours to observe behaviour, then migrate to systemd.

---

## Operational Commands (Systemd)

```bash
# Status
sudo systemctl status mtf-watchdog ibgateway

# Graceful stop
sudo systemctl stop mtf-watchdog

# Emergency flat + stop
.venv/bin/python scripts/kill_switch.py && sudo systemctl stop mtf-watchdog

# Deploy update
git pull && uv sync && sudo systemctl restart mtf-watchdog

# View logs
journalctl -u mtf-watchdog -f
tail -f .tmp/logs/mtf_live_*.log
grep "Attempt" .tmp/logs/watchdog.log | tail -20   # restart history
```
