# Directive: Live Deployment and Monitoring

> Last updated: 2026-03-14

## Pre-Flight Checklist

> [!CAUTION]
> **All items must pass before going live.**

- [ ] `config/risk.toml` reviewed — max drawdown, position limits, daily loss cap
- [ ] `scripts/kill_switch.py` tested on practice account (run `uv run python scripts/kill_switch.py --dry-run`)
- [ ] OOS Sharpe ≥ 50% of IS Sharpe (verified in backtest — MTF: 0.98 ✓, ORB: verified ✓)
- [ ] Paper trading session completed with no errors (at least one full day)
- [ ] IBKR Gateway running on correct port, API settings confirmed (Read-Only unchecked)

---

## Practice (Local) Deployment

Always start on paper before switching to live. Verify the strategy in the local Practice Environment.

### 1. Configuration

Ensure `.env` is set to practice mode (port 4002 for Gateway, 7497 for TWS):
```ini
IBKR_PORT=4002
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### 2. Execution

**A. ORB Strategy (7 large-cap equities — production)**
```bash
uv run python scripts/run_live_orb.py 2>&1 | tee .tmp/logs/orb_stdout_$(date +%Y%m%d_%H%M%S).log
```
Connects to TWS/Gateway, loads 7 instruments (`UNH.NYSE`, `AMAT.NASDAQ`, `TXN.NASDAQ`, `INTC.NASDAQ`, `CAT.NYSE`, `WMT.NASDAQ`, `TMO.NYSE`), subscribes to 5-MINUTE + 1-DAY bars for each, and starts trading.

**B. Multi-Timeframe Confluence — EUR/USD (deployment-ready)**
```bash
uv run python scripts/run_live_mtf.py 2>&1 | tee .tmp/logs/mtf_stdout_$(date +%Y%m%d_%H%M%S).log
```
Connects to IDEALPRO, loads EUR/USD, subscribes to H1 + H4 + D + W bars, and starts trading.
Config: `config/mtf.toml` (H1/H4/D/W, SMA, threshold=0.10, ATR stop 2.5×).

**C. ML Strategy (XGBoost — experimental, OOS Sharpe 1.142)**
```bash
uv run python scripts/run_live_ml.py
```
Requires trained `.joblib` model in `models/`. See `directives/Machine Learning Strategy Discovery.md`.

### 3. Validation — MTF Startup Sequence

```
Checking for latest data...
Data download finished.
MTF Strategy Started. Warming up...
  Loading H1 warmup from data/EUR_USD_H1.parquet
  Loading H4 warmup from data/EUR_USD_H4.parquet
  Loading D  warmup from data/EUR_USD_D.parquet
  Loading W  warmup from data/EUR_USD_W.parquet
Warmup complete. ATR stop mult: 2.5x. Ready for signals.
```

Dashboard prints on every bar. If MA/RSI show `??`, warmup parquet is missing — run `uv run python scripts/download_data_mtf.py`.

### 4. Log Monitoring

```bash
# PowerShell
Get-Content .tmp/logs/mtf_live_*.log -Wait -Tail 50

# bash (Antigravity terminal)
tail -f .tmp/logs/mtf_stdout_$(ls -t .tmp/logs/mtf_stdout_*.log | head -1 | xargs basename)
```

Key MTF log strings:
- `ATR stop placed @ 1.08234 (2.5x ATR = 0.00250)` — stop order active
- `ATR stop triggered. Fill @ ...` — hard stop fired; strategy re-evaluates next bar
- `Signal Flip: Long -> Short.` — primary exit + reversal
- `Signal Neutral. Closing position.` — primary exit, going flat

---

## Cloud Execution (Aspirational — VPS / Docker)

The recommended architecture for long-term deployment:

### Infrastructure

- **Provider**: Google Cloud Platform (GCE) or AWS EC2
- **Region**: `us-east1` (New York) — closest to IBKR matching engine
- **Spec**: 2 vCPU, 4 GB RAM minimum

### Containerisation

Build and run via Docker for environment consistency:
```bash
# Build (from project root)
uv run python scripts/build_docker.py

# Run with restart policy
docker run -d \
  --name titan-live \
  --restart always \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.tmp/logs:/app/.tmp/logs \
  titan-ibkr-algo
```

> [!IMPORTANT]
> Copy the `models/` directory into the container if the ML strategy is active.

### Monitoring — The "Guardian" Pattern

Run a lightweight monitor that tails logs for `ERROR` strings and alerts via Slack:
- Implementation: `titan/utils/notification.py`
- Alert if no new log lines for > 5 minutes (strategy stalled or crashed)

---

## Emergency Stop

If the strategy misbehaves, use the kill switch:
```bash
uv run python scripts/kill_switch.py
```

This cancels all open orders and closes all positions directly via TWS API (does not require NautilusTrader to be running). See `directives/ORB Strategy User Guide.md` and `directives/MTF Strategy User Guide.md` for strategy-specific emergency procedures.

---

## Success Criteria for Paper Session

Before switching to live:
- [ ] Startup sequence completes with no `[ERROR]` lines
- [ ] Dashboard shows populated indicator values (not `??`) on every bar
- [ ] At least one trade opens and closes cleanly (logs show `ATR stop placed`, `Signal Neutral` or `Signal Flip`)
- [ ] No orphaned stop orders (verify via TWS order panel after a close)
- [ ] `Ctrl+C` shutdowns gracefully with no hanging processes
