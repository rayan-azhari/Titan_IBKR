# Docker Paper Trading Guide

> A complete walkthrough for running the Titan-IBKR champion portfolio on
> paper using Docker — no TWS, no GUI, no nightly 23:45 ET reset interrupting
> trades. Read this top-to-bottom the first time. After that the
> "Day-to-day operations" section is your reference.
>
> For the host-Python and VPS systemd alternatives, see
> `directives/Deployment & Operations.md`.

---

## 0. What you're setting up

Two Docker containers, started together with one command:

```
┌──────────────────────────────────────────────────────────────┐
│                  docker compose up -d                        │
│                                                              │
│  ┌──────────────────┐         ┌─────────────────────────┐   │
│  │   ib-gateway     │ ◄────── │   titan-portfolio       │   │
│  │  (port 4004)     │         │  (run_portfolio.py      │   │
│  │   IB Gateway +   │         │   under watchdog)       │   │
│  │   IBC + Xvfb     │         │                         │   │
│  └──────────────────┘         └─────────────────────────┘   │
│         ↑                              ↑                     │
│   Sunday 03:00 ET                Auto-restart on crash       │
│   weekly auto-restart            within 180 seconds          │
└──────────────────────────────────────────────────────────────┘
                       Internet → IBKR
```

What you get vs. running TWS on Windows:

| Thing | TWS on Windows | This Docker stack |
|---|---|---|
| Daily 23:45 ET hard restart | ~20 min outage every weekday | Suppressed; one ~3 min outage Sunday 03:00 ET |
| Manual GUI login after Windows reboot | Yes | No — IBC auto-logs in |
| Manual GUI login after Windows Update killed it | Yes | No — `restart: unless-stopped` |
| Strategy crash recovery | Manual | Watchdog restarts within 180 s |
| Halt / kill-switch state across restart | Lost | Persisted in named volume |

### Port topology (worth knowing once)

Inside the gateway container, the actual IB Gateway process listens on
**4001 (live)** and **4002 (paper)** — IBKR's hard-coded ports.

The `gnzsnz/ib-gateway` image then runs `socat` to forward those to
**4003 (live)** and **4004 (paper)** for external clients. We connect
to the socat-forwarded ports (`IBKR_PORT=4004` for paper).

If you see `4001` or `4002` in gateway logs (e.g. *"API server listening
on port 4002"*), that's the gateway's internal log line — the strategy
container reaches it via `4004`. Everything is correct; the dual port
numbering is just how the image is built.

Hard limits this **does not** fix:
- Live trading still requires 2FA on every login (paper does not).
- IBKR can still take the gateway down for actual scheduled maintenance.
- A bad strategy bug will still lose money on paper — paper is for testing
  the *plumbing*, not the trade logic. Strategy edge has already been
  validated upstream.

---

## 1. Prerequisites

You need three things on your Windows machine:

### 1.1 Docker Desktop with WSL2 backend

If you've never installed Docker on this machine:

1. Download Docker Desktop from <https://www.docker.com/products/docker-desktop/>.
2. Run the installer. When prompted, leave **"Use WSL 2 instead of Hyper-V"**
   checked (this is the default on Windows 11).
3. Reboot when the installer asks.
4. After reboot, launch Docker Desktop. You should see a green "Engine
   running" indicator in the bottom-left of the Docker Desktop window.

If you already have Docker Desktop, just confirm it's up:

```bash
docker version
docker compose version
```

You should see both `Client:` and `Server:` blocks for `docker version`,
and `Docker Compose version v2.x.x` for the second command. If the server
section is missing or you get "Cannot connect to the Docker daemon" —
launch Docker Desktop and wait for the green indicator before continuing.

### 1.2 An IBKR paper account

Log into <https://www.interactivebrokers.com/portal/> with your live
account. Click your profile in the top-right → "Settings" → look for
"Paper Trading Account". If you don't have one, you can create one for
free; it takes a few minutes to provision.

You need three things from the paper account:

- **Username** — the same username pattern as your live login but usually
  prefixed with something like `paper-` or with a separate paper username.
  Open the Client Portal on the paper side once to confirm what string
  works for login.
- **Password** — the paper account's password. Often the same as live;
  sometimes different. Test it by logging into the paper Client Portal.
- **Account ID** — starts with `DU` (e.g. `DU1234567`). Visible at the top
  of the paper Client Portal.

> [!CAUTION]
> **Test the credentials by logging into IBKR's web Client Portal first.**
> If they don't work there, they won't work in the Docker stack either,
> and you'll burn 10 minutes debugging the wrong layer.

### 1.3 The Titan-IBKR repo on disk

You're presumably reading this file from the repo, so this is already
done. Confirm by opening a terminal and running:

```bash
ls Dockerfile docker-compose.yml .env.docker.example
```

All three should exist. If any are missing, you're in the wrong directory
or on a stale branch.

---

## 2. First-time setup (one-time, ~5 minutes of your time + ~5 minutes of build)

### Step 1 — Open a terminal at the repo root

PowerShell, Git Bash, WSL terminal — any of them work as long as Docker
Desktop is running and you're in the directory that contains the
`Dockerfile`. Confirm:

```bash
pwd
# expect:  C:\Users\rayan\Desktop\Antigravity\Titan-IBKR
#    or:   /mnt/c/Users/rayan/Desktop/Antigravity/Titan-IBKR  (WSL)
```

All commands below run from that directory.

### Step 2 — Create your credentials file

```bash
cp .env.docker.example .env.docker
```

Open `.env.docker` in any editor and fill in the four required values:

```ini
TWS_USERID=your_paper_username       # ← from step 1.2
TWS_PASSWORD=your_paper_password     # ← from step 1.2
IBKR_ACCOUNT_ID=DUxxxxxxx            # ← from step 1.2 (DU... for paper)

# Leave these as defaults unless you have a specific reason:
TRADING_MODE=paper
IBKR_PORT=4004
IBKR_CLIENT_ID=7
```

Save and close the file.

> [!CAUTION]
> `.env.docker` contains your IBKR password in plaintext. The repo's
> `.gitignore` excludes `.env*` files so you can't accidentally commit it,
> but double-check with `git status` before any commit. Do NOT paste the
> contents of this file into chat, screenshots, or pastebins.

### Step 3 — Build the images and start the stack

```bash
docker compose --env-file .env.docker up -d --build
```

What happens during this command (first time):

1. **Pull `gnzsnz/ib-gateway:stable`** — about 800 MB, takes 1-3 minutes
   depending on your connection.
2. **Pull `python:3.11-slim`** — about 50 MB.
3. **Build the `titan-portfolio` image** — installs Python deps via `uv
   sync`, copies `titan/` and `scripts/`. About 2-3 minutes the first
   time, then ~10 seconds for incremental rebuilds (the dep install
   is cached).
4. **Start `ib-gateway` container** — IBC logs in to IBKR (60-90 s).
5. **Start `titan-portfolio` container** — waits for the gateway
   healthcheck to go green, then starts the watchdog, which starts
   `run_portfolio.py`.

When the command returns to your prompt, the stack is up but may still
be initialising. Move on to verification.

### Step 4 — First verification (~2 minutes)

```bash
docker compose ps
```

Expected output (after about 2 minutes):

```
NAME              IMAGE                       STATUS                  PORTS
titan-ib-gateway  gnzsnz/ib-gateway:stable    Up 2 minutes (healthy)  4003-4004/tcp
titan-portfolio   titan-ibkr-titan-portfolio  Up 90 seconds
```

Two things to check:

- **`(healthy)` next to `titan-ib-gateway`** — means the gateway is logged
  in and the API socket is accepting connections. If it's stuck on
  `(starting)` for more than 3 minutes, jump to **Troubleshooting → 1**.
- **`titan-portfolio` is `Up`** — means the watchdog is running. If it's
  in `Restarting` loop, jump to **Troubleshooting → 2**.

Now confirm the gateway logged in cleanly:

```bash
docker compose logs --tail=80 ib-gateway
```

Look for two key lines (somewhere in that output):

```
IBC: Login has completed
API server listening on port 4002
```

If you see "Login failed", "Already logged in elsewhere", or "Invalid
username", jump to **Troubleshooting → 1**.

Now confirm the strategy connected:

```bash
docker compose logs --tail=200 titan-portfolio
```

You should see, in order:

```
Watchdog starting. Child command: ...run_portfolio.py --strategies champion_portfolio
[Attempt 1] Starting portfolio runner at 2026-04-28T...
   Loading H1 warmup from data/AUD_JPY_H1.parquet
   Loading H4 warmup from data/AUD_JPY_H4.parquet
   ...
   Connected to ib-gateway:4004 as client 7, account DU1234567
   Subscribed: AUD/JPY.IDEALPRO 1-HOUR-MID
   Subscribed: CSPX.LSEETF 1-DAY-LAST
   Subscribed: IHYU.LSEETF 1-DAY-LAST
   Warmup complete. Ready for signals.
```

The exact wording will differ a little — what matters is:

- No `[ERROR]` or `Traceback` lines.
- Account ID matches the `DU...` value you put in `.env.docker`.
- Three subscription lines (one for AUD/JPY, two for the bond-equity pair).
- A "Warmup complete" or equivalent indication that bars loaded.

You're paper-trading. Tail the logs live to watch the first hour:

```bash
docker compose logs -f titan-portfolio
```

Press Ctrl+C to detach. The container keeps running.

---

## 3. Day-to-day operations

### See what's running

```bash
docker compose ps
```

### Live tail the strategy

```bash
docker compose logs -f titan-portfolio
```

### Live tail just the watchdog (restart history)

```bash
docker compose exec titan-portfolio tail -f /app/.tmp/logs/watchdog_portfolio.log
```

### Sanity-check the IBKR connection

```bash
docker compose exec titan-portfolio python scripts/verify_connection.py
```

This connects with a different client ID, verifies the API socket, and
exits. Useful when you suspect the gateway is the problem rather than the
strategy.

### Restart the strategy after editing a TOML

`config/` is bind-mounted, so editing `config/risk.toml` (or any other
TOML in `config/`) and running:

```bash
docker compose restart titan-portfolio
```

is enough — no rebuild required. The watchdog will start a fresh
`run_portfolio.py` with the new config in 5-10 seconds.

### Rebuild after a code change

If you edited anything under `titan/` or `scripts/`, rebuild the
strategy image:

```bash
docker compose up -d --build titan-portfolio
```

The gateway container is untouched — only the strategy stops and
restarts.

### Stop everything

```bash
docker compose down
```

This sends SIGTERM to the watchdog, which forwards it to
`run_portfolio.py` (60-second grace period, configured in
`docker-compose.yml`). The strategy gets a chance to flatten paper
positions and persist halt state before the container actually exits.

The `.tmp/` named volume is preserved, so logs and `portfolio_halt.json`
survive until you explicitly delete them:

```bash
docker compose down -v       # ← also wipes .tmp/ — only do this on a clean teardown
```

### Start everything again

```bash
docker compose --env-file .env.docker up -d
```

(Same command as the first time, just without `--build` — Docker reuses
the existing images.)

### Inspect persisted halt state

```bash
docker compose exec titan-portfolio cat /app/.tmp/portfolio_halt.json
```

If `"halted": true` and you want to clear it, the safest path is to
restart the strategy with the operator-reset flow described in
`directives/Deployment & Operations.md` section 9.

### Emergency flatten — kill switch

If a paper position is misbehaving and you want to flatten everything
immediately:

```bash
docker compose exec -it titan-portfolio python scripts/kill_switch.py
```

The `-it` flags make the shell interactive — the kill switch lists
positions and asks you to type `KILL` to confirm. After it fires, stop
the watchdog so the strategy doesn't immediately re-enter:

```bash
docker compose stop titan-portfolio
```

### Image management — build once, version, save to file

The compose file uses both `image: titan-portfolio:latest` (what gets
run) and `build: { context: ., dockerfile: Dockerfile }` (how to
rebuild when you want to). With this setup `docker compose up -d`
**never rebuilds** — it always uses the existing tagged image. You
control the rebuild explicitly.

#### Build (or rebuild) the image

```bash
# Quick: build + tag :latest
scripts/build_image.sh

# Versioned: build + tag :1.0.0 + tag :latest
scripts/build_image.sh 1.0.0
```

The script wraps `docker build` and tags with both the version you
specify and `:latest`, so the compose default keeps working.

#### Save the image to a portable file (backup or transfer)

```bash
# Build + save to dist/titan-portfolio-1.0.0.tar.gz (~250 MB compressed)
scripts/build_image.sh 1.0.0 --save

# Or save without rebuilding (image must already exist)
mkdir -p dist
docker save titan-portfolio:1.0.0 | gzip > dist/titan-portfolio-1.0.0.tar.gz
```

The `dist/` directory is gitignored. The tarball contains all image
layers — drop it on a USB stick, copy to a VPS, anywhere Docker runs.

#### Restore from a saved file

```bash
gunzip -c dist/titan-portfolio-1.0.0.tar.gz | docker load
# Image is now available locally as titan-portfolio:1.0.0
docker compose --env-file .env.docker up -d
```

#### Pin a specific version in compose

The compose file resolves `${TITAN_IMAGE:-titan-portfolio:latest}`. To
freeze on a specific version (e.g., for a known-good rollback):

```bash
echo 'TITAN_IMAGE=titan-portfolio:1.0.0' >> .env.docker
docker compose --env-file .env.docker up -d
```

To revert to floating `:latest`, remove that line.

#### Save the IB Gateway image too (fully offline bundle)

The `gnzsnz/ib-gateway:stable` image is pulled from Docker Hub. If you
want a self-contained offline bundle (e.g., to bring up the stack on a
machine without internet during the initial boot):

```bash
docker save gnzsnz/ib-gateway:stable | gzip > dist/ib-gateway-stable.tar.gz
# Restore on target machine:
gunzip -c dist/ib-gateway-stable.tar.gz | docker load
```

#### When to rebuild

Rebuild when you change anything baked into the image:
- `pyproject.toml` or `uv.lock` (dependency changes)
- `titan/` (strategy code)
- `scripts/` (runner code)
- `Dockerfile` itself

You do NOT need to rebuild when you change:
- `config/*.toml` — bind-mounted, just `docker compose restart titan-portfolio`
- `data/*.parquet` — bind-mounted
- `.env.docker` — read at compose-up time, restart sufficient

### Notifications (Slack and/or Telegram)

The strategies emit a notification on **four** events:

| Event | When | What |
|---|---|---|
| 🎯 Signal | Right before `submit_order` | Strategy id, side, qty, price, indicator readings, equity, risk |
| 📝 Order accepted | IBKR `OrderAccepted` event | venue_order_id, client_order_id, side, qty, type |
| 🚫 Order rejected | IBKR `OrderRejected` event | client_order_id, reason from IBKR |
| ✅ Position closed | NautilusTrader `PositionClosed` | Direction, entry/exit, PnL, equity-after, cumulative-% |

Two backends supported. **Either, both, or neither** — set the env vars in
`.env.docker` and the notification module dispatches to whichever is
configured. If neither is set, the calls are a silent no-op (no errors, no
logs after the first warning).

#### Slack (recommended — simplest)

1. Open <https://api.slack.com/messaging/webhooks> and create an Incoming
   Webhook for the channel you want messages in.
2. Copy the URL (it looks like `https://hooks.slack.com/services/T.../B.../...`).
3. Paste into `.env.docker`:
   ```ini
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
   ```
4. `docker compose --env-file .env.docker up -d` (or `restart titan-portfolio`).

#### Telegram (alternative or in addition)

1. In Telegram, message `@BotFather`, send `/newbot`, follow the prompts.
   Save the **bot token** it gives you.
2. Send any message to your new bot from your personal Telegram account.
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser.
   Find your chat in the JSON response and copy `result[0].message.chat.id`.
4. Paste into `.env.docker`:
   ```ini
   TELEGRAM_BOT_TOKEN=123456789:ABC-DEF...
   TELEGRAM_CHAT_ID=987654321
   ```
5. `docker compose --env-file .env.docker up -d`.

Both `BOT_TOKEN` and `CHAT_ID` must be set; either alone is treated as
unconfigured.

#### Send a test message without waiting for a real signal

```bash
docker compose exec titan-portfolio python -m titan.utils.notification signal
docker compose exec titan-portfolio python -m titan.utils.notification order
docker compose exec titan-portfolio python -m titan.utils.notification position
```

Each fires a sample message of that type to whichever backend(s) are
configured. Use this to verify the webhook before any real trading.

#### What the messages look like

```
🎯 *Signal* — `bond_gold_CSPX`
`BUY` 36 CSPX.LSEETF
  • Price: 770.5000   Notional: +27,738.00 USD
  • Reason: ief_z=-0.8900  threshold=+0.5000  lookback=60  hold_days=20
  • Risk: +1.00% of +10,000.00 USD = +100.00 USD
  • Account: `DUP958545`

📝 *Order accepted* — `bond_gold_CSPX`
`BUY` 36 CSPX.LSEETF (MARKET) @ 770.5000
  • venue_order_id: `101`
  • client_order_id: `O-20260430-164248-PORTFOLIO-001-1`

✅ *Position closed* — `bond_gold_CSPX`
`LONG` CSPX.LSEETF
  • Held: 8.0d
  • Entry: 770.5000   Exit: 785.2000
  • PnL: +528.40 USD   Strategy equity: +10,528.40 USD
  • Cumulative: +5.28% on +10,000.00 USD initial
```

#### Failure handling — important

Notifications are **fire-and-forget with a 2-second timeout**. If Slack
or Telegram is down, the strategy logs a one-line `notify failed: ...`
warning and keeps trading. The trade itself never depends on a successful
notification.

This means:

- The trade log in `.tmp/logs/portfolio_live_*.log` is the source of
  truth, not your phone.
- You can lose individual messages to network blips with no operational
  impact.
- For critical events (kill switch fires, PRM halts the portfolio), the
  log file is the only guaranteed record. Monitor the logs separately.

---

## 4. The weekly restart (what "healthy" looks like)

Sunday at 03:00 America/New_York (06:00 UTC during DST, 08:00 UTC
otherwise — `TZ=America/New_York` in compose handles this correctly):

```
Sun 03:00:00 ib-gateway       IBC: scheduling restart at 03:00:00
Sun 03:00:01 ib-gateway       IBC: stopping IB Gateway...
Sun 03:00:02 titan-portfolio  [ERROR] Connection to ib-gateway:4004 lost
Sun 03:00:03 titan-portfolio  [Attempt N] Child exited code=1 after Xs
Sun 03:00:04 titan-portfolio  Restarting in 180s...
Sun 03:01:30 ib-gateway       IBC: starting IB Gateway...
Sun 03:02:15 ib-gateway       IBC: Login has completed
Sun 03:02:30 ib-gateway       API server listening on port 4002
Sun 03:03:04 titan-portfolio  [Attempt N+1] Starting portfolio runner...
Sun 03:03:10 titan-portfolio  Connected to ib-gateway:4004 as client 7
```

Total downtime ≈ 3 minutes, once a week. FX positions are open over the
weekend gap anyway, so this is benign for the AUD/JPY strategy. For the
bond-equity pair, this happens between trading sessions — also benign.

You can simulate this without waiting for Sunday:

```bash
# Force a "weekly restart" right now
docker compose stop ib-gateway

# Watch the strategy crash and the watchdog log a retry
docker compose logs --tail=20 titan-portfolio

# Bring the gateway back
docker compose start ib-gateway

# Within 180 seconds, the watchdog spawns a new attempt and reconnects
docker compose logs -f titan-portfolio
```

If that whole sequence works, the weekly cycle will work too.

---

## 5. Troubleshooting

### Troubleshooting → 1: gateway never goes healthy

Symptom: `docker compose ps` shows `titan-ib-gateway` as `(starting)`
or `(unhealthy)` after 3+ minutes.

Diagnose:

```bash
docker compose logs --tail=120 ib-gateway
```

Common causes and fixes:

| Log message contains | Cause | Fix |
|---|---|---|
| `Login failed` / `Invalid username or password` | Wrong creds in `.env.docker` | Test on Client Portal web first; correct `.env.docker`; `docker compose down && up -d` |
| `Already logged in elsewhere` | Another TWS/Gateway is running with the same login | Close TWS on Windows; close any other Gateway sessions on other machines |
| `Account temporarily locked` | Too many failed logins | Wait 15 minutes; verify creds via Client Portal; retry |
| `Cannot connect to host` | Network firewall blocking IBKR | Check your machine can reach `gdcdyn.interactivebrokers.com` on port 4001/4002 |
| Hangs at `Starting Xvfb...` | Image bug or platform issue | `docker compose pull ib-gateway` to refresh; rare |

If the log tails rotate too fast to read:

```bash
docker compose logs ib-gateway > gateway.log 2>&1
# then open gateway.log in your editor and search for "fail", "error", "login"
```

### Troubleshooting → 2: titan-portfolio in restart loop

Symptom: `docker compose ps` shows `titan-portfolio` constantly
`Restarting`.

Diagnose:

```bash
docker compose logs --tail=200 titan-portfolio
```

Common causes:

| Log message contains | Cause | Fix |
|---|---|---|
| `IBKR_ACCOUNT_ID is required but not set` | Missing in `.env.docker` | Add it; `docker compose up -d` |
| `Connection refused` / `client id already in use` | Another script holding client ID 7 | Change `IBKR_CLIENT_ID` in `.env.docker` to something else (e.g. 17); restart |
| `data/AUD_JPY_H1.parquet missing` | The `data/` bind-mount didn't pick up the file | Verify the file exists on the host; check no path-translation issue between Windows and WSL2 |
| `ImportError` or `ModuleNotFoundError` | Build cached an old layer | `docker compose down && docker compose build --no-cache titan-portfolio && docker compose up -d` |
| Watchdog says `Child died in <30s` repeatedly | The runner itself is crashing fast | Read the actual `run_portfolio.py` traceback above the watchdog message; usually a config error |

### Troubleshooting → 3: stack starts fine but no trades happen for hours

This is normal for slow strategies. The bond-equity pair is daily; it
fires on the daily close. The AUD/JPY mean-reversion strategy fires
when its filter conditions align — typically once or twice a day at
most.

Verify the strategy is alive (not silently dead):

```bash
docker compose logs --since 10m titan-portfolio | grep -iE "bar|tick|signal|warmup"
```

If you see periodic bar-arrival messages, it's working — there just
isn't a trade signal right now.

If you see *nothing* in the last 10 minutes:

```bash
docker compose exec titan-portfolio python scripts/verify_connection.py
```

If that succeeds, the gateway is fine and the strategy is just quiet.
If it fails, jump to **Troubleshooting → 1**.

### Troubleshooting → 4: build itself fails

Symptom: `docker compose up -d --build` errors out before containers start.

Common cases:

- `error sending tarball: ... no space left on device` → Free up disk;
  `docker system prune -a` will reclaim a lot but also delete unused
  images.
- `failed to solve: pull access denied for ghcr.io/astral-sh/uv:0.5.11` → 
  The pinned uv version was renamed/removed. Edit `Dockerfile` to a
  newer tag from <https://github.com/astral-sh/uv/pkgs/container/uv>.
- `Killed` during `uv sync` → Docker Desktop running out of memory.
  Open Docker Desktop → Settings → Resources → bump RAM to ≥ 4 GB.

### Troubleshooting → 5: Docker Desktop stops on Windows reboot

Open Docker Desktop → Settings → General → tick **"Start Docker Desktop
when you sign in to your computer"**. The compose stack has
`restart: unless-stopped`, so once Docker Desktop is up, the containers
auto-resume.

---

## 6. Going to live (out of scope here, read this before you flip)

The same compose stack runs live with three changes to `.env.docker`:

```ini
TRADING_MODE=live
IBKR_PORT=4003                 # socat-forwarded live port (internal gateway is 4001)
IBKR_ACCOUNT_ID=Uxxxxxxx       # live account, U… not DU…
TWS_USERID=your_live_username
TWS_PASSWORD=your_live_password
```

But — and this is the important part — **live forces 2FA on every
fresh login**. After Sunday's weekly auto-restart, IB Gateway will
prompt for an IBKR Mobile push approval or SMS code. Unattended
weekly restarts will hang the gateway until a human approves the push.

Before flipping to live:

1. Run the paper stack for at least one full week including a clean
   Sunday cycle.
2. Decide who is on call Sunday 03:00 ET and how they get the push
   notification.
3. Re-read `directives/Deployment & Operations.md` Section 1 (pre-flight
   checklist).
4. Re-validate the strategies' OOS Sharpe ≥ 50% of IS Sharpe in
   `directives/System Status and Roadmap.md`.

`TWOFA_TIMEOUT_ACTION=restart` (already set in `docker-compose.yml`)
makes the gateway re-prompt rather than silently stall when the push
times out — small mitigation, not a substitute for being available.

---

## 7. Rolling back to host-Python

If the Docker stack misbehaves and you need the strategy running NOW,
you can fall back to the original host-Python workflow without touching
the Docker setup:

```bash
# Stop the docker stack so it doesn't fight you for client IDs / port
docker compose down

# Make sure your host .env (NOT .env.docker) is configured with
# IBKR_HOST=127.0.0.1, IBKR_PORT=7497 (TWS paper) or 4002 (Gateway paper)
# and a TWS or local IB Gateway is logged in.

uv run python scripts/run_portfolio.py --strategies champion_portfolio
```

The Docker config is unaffected; come back with `docker compose
--env-file .env.docker up -d` whenever you want.

---

## 8. Quick reference card

```bash
# First time
cp .env.docker.example .env.docker     # then edit it
scripts/build_image.sh                  # builds + tags titan-portfolio:latest
docker compose --env-file .env.docker up -d

# Check status
docker compose ps
docker compose logs -f titan-portfolio

# Restart strategy after TOML edit (no rebuild needed)
docker compose restart titan-portfolio

# Rebuild image after code change (titan/, scripts/, pyproject.toml)
scripts/build_image.sh                  # bumps :latest in place
docker compose up -d titan-portfolio    # picks up new image

# Build a versioned snapshot + save to dist/ as a .tar.gz file
scripts/build_image.sh 1.0.0 --save     # → dist/titan-portfolio-1.0.0.tar.gz

# Restore an image from a saved file (e.g. on a fresh machine)
gunzip -c dist/titan-portfolio-1.0.0.tar.gz | docker load
docker compose --env-file .env.docker up -d

# Sanity check IBKR connection
docker compose exec titan-portfolio python scripts/verify_connection.py

# Emergency flatten
docker compose exec -it titan-portfolio python scripts/kill_switch.py
docker compose stop titan-portfolio

# Stop the stack
docker compose down                   # graceful, .tmp/ preserved
docker compose down -v                # also wipe .tmp/ volume

# Inspect halt state
docker compose exec titan-portfolio cat /app/.tmp/portfolio_halt.json

# Force the weekly-restart cycle to test resilience
docker compose stop ib-gateway && sleep 5 && docker compose start ib-gateway
```
