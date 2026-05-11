# Market Data Refresh Strategy

> Design doc + operational plan for keeping warmup parquets current on the
> live (paper or production) Docker stack. Companion to
> `directives/Docker Paper Trading Guide.md` and
> `directives/Deployment & Operations.md`.

---

## 1. The problem

Both champion-portfolio strategies need historical bars at startup to
populate indicators *before* the first live bar arrives:

| Strategy | Files read | Bars needed |
|---|---|---|
| `mr_audjpy` | `data/AUD_JPY_{H1,H4,D,W}.parquet` | 3000 H1 (~17 weeks) |
| `bond_equity_ihyu_cspx` | `data/CSPX_D.parquet`, `data/IHYU_D.parquet` | ~120 daily (~6 months) |

The compose stack bind-mounts `./data` (read-write) into the container —
so the parquets are host-managed, not baked into the image, and writes
from inside the container land on host disk. That's the right
architecture, but it leaves an open question: **how do the parquets
stay current** on a long-running unattended VPS deployment, especially
across multi-day outages?

Today the answer is "manually run download scripts on the host". On a
laptop that's fine. On a VPS without a human at the console it's not.

---

## 2. Three scenarios that matter

| Scenario | Acceptable staleness | Frequency |
|---|---|---|
| **Sunday weekly Gateway restart** | Hours | Weekly |
| **Multi-day outage / VPS reboot** | Days | Rare but possible |
| **Fresh VPS deploy** | None — no data exists | One-time per host |

The first is trivially handled by daily-or-better refresh. The second is
the painful one — strategies boot with stale indicator state. The third
is a one-time bootstrap problem.

---

## 3. Options considered

### Option A — Nightly cron refresh (CHOSEN for Phase 1)

A scheduled job runs the existing `scripts/download_data*.py` once a day
and overwrites/merges the parquets in place.

| Pros | Cons |
|---|---|
| Decoupled — failure doesn't crash trading | Up to 24h staleness |
| Reuses code we already trust | Requires data-source connectivity (IBKR or external API) |
| systemd timers are simple | Doesn't capture the *exact* bars IBKR fed the strategy |
| Restart-safe: parquet always coherent | |

### Option B — In-process bar journal

Strategy (or sibling) subscribes to every bar event and writes it to disk
as it arrives. Two flavors:

- **B1 — Append-only journal** (CSV/JSONL): each bar appended on arrival,
  monthly file rollover, warmup concatenates canonical parquet + journals
- **B2 — Monthly parquet rollover**: buffer N bars in memory, write a
  fresh per-month parquet on each flush

| Pros | Cons |
|---|---|
| Always perfectly current | Code change in `titan/` — affects strategy |
| Captures *exact* bars IBKR delivered (audit trail) | Parquet doesn't natively append; need rollover |
| Survives any restart with zero blind window | Concurrency: persistence can't slow down strategy |
| Free (no extra API calls) | Reconciliation logic if strategy crashes mid-write |

> [!IMPORTANT]
> Parquet is not designed for append-as-you-go. It's columnar and wants
> whole batches written at once. Naive appending corrupts files. Safe
> patterns: per-month rollover (B2), or write to journal (CSV/JSONL) and
> reconcile to parquet later. Anyone "appending to parquet" is doing one
> of those, or using a DB that happens to write parquet (DuckDB).

### Option C — Hybrid (A + B)

Cron downloads canonical parquet weekly (deep history). Strategy writes
a journal of received bars (live tail). Warmup reads both, concatenates,
deduplicates by timestamp.

| Pros | Cons |
|---|---|
| Both deep history and zero blind window | More moving parts |
| Cron failure recoverable from journal | Reconciliation logic ~150 LOC |

### Option D — Real time-series DB

Replace parquet with TimescaleDB / DuckDB / ClickHouse.

| Pros | Cons |
|---|---|
| Proper write-append semantics | All `titan/` code reads parquet today — major refactor |
| Built-in dedup, range queries | Another container to operate |
| Scales to 50+ instruments | Overkill for current scale |

**Verdict:** don't. Revisit if portfolio grows past ~20 instruments.

---

## 4. Phased plan

### Phase 1 — Cron refresh + freshness check (DONE, current state)

**What ships in Phase 1:**

| Artefact | Purpose |
|---|---|
| `scripts/refresh_market_data.sh` | Wrapper that calls existing download scripts in sequence for the champion portfolio's instruments |
| `directives/operations/titan-data-refresh.service.example` | systemd service template — runs `docker compose exec` to fire the refresh inside the strategy container |
| `directives/operations/titan-data-refresh.timer.example` | systemd timer — daily 02:00 America/New_York (after market close, before Sunday 03:00 ET Gateway weekly restart) |
| `_check_data_freshness()` in `scripts/run_portfolio.py` | Logs WARN/ERROR at startup if any required parquet is older than 2/7 days; non-blocking |
| Updates to `directives/Deployment & Operations.md` | Install steps for the timer on a VPS |

**Data sources (Phase 1):**
- AUD/JPY (H1/H4/D/W): `download_data_mtf.py --pair AUD_JPY` — IBKR via the running Gateway. Merge-safe (concat + dedup on timestamp).
- CSPX_D, IHYU_D: `download_data_yfinance.py --symbols ... --interval D` — Yahoo Finance, free. **Overwrites** the parquet, but we always pass `--start 2015-01-01` so the rewritten file has 10+ years of history.

**Cadence:** daily 02:00 ET. Most market events done; 1h before any
weekly Gateway restart on Sunday.

**Why this is enough:** for two strategies on H1+D timeframes, 24h
staleness is rounding error. Real concern would be a multi-day VPS
outage, but the freshness check catches that and warns the operator
before live trading resumes.

### Phase 2 — Live bar journal (NOT BUILT)

Trigger to build: Phase 1 in production reveals an actual operational
problem — e.g., recurring multi-day VPS outages where the cron-refresh
window can't catch up before the strategy needs to trade.

If we get to Phase 2:

- New `titan/journaling/bar_journal.py` Strategy that subscribes to every
  bar of every instrument and writes monthly parquet files
  (`data/_live/AUD_JPY_H1_2026_05.parquet`)
- Modify `mr_audjpy` and `bond_gold` warmup to glob `data/_live/*.parquet`
  and concat with canonical
- Periodic merge job (weekly cron) folds journal files into canonical
  parquet and prunes old months
- Update freshness check to look at the union of canonical + journal

Estimated work: half a day. Not justified yet.

### Phase 3 — DB migration (DEFERRED INDEFINITELY)

Trigger to build: portfolio grows past ~20 instruments and parquet
proliferation becomes operationally painful.

If we get there: migrate to DuckDB (embedded, zero-ops, parquet-friendly)
not TimescaleDB (Postgres-heavy).

---

## 5. Phase 1 install steps (VPS)

### 5.1 Confirm refresh script works manually

From the host (or any machine with the stack running):

```bash
docker compose exec titan-portfolio bash /app/scripts/refresh_market_data.sh
```

Expected output: log lines from each download script, then a summary
listing the parquets that were updated. Run twice in a row — second run
should be near-instant (the merge-safe scripts only fetch new bars).

### 5.2 Install systemd timer (VPS only — assumes Linux + Docker Compose)

```bash
sudo cp directives/operations/titan-data-refresh.service.example \
       /etc/systemd/system/titan-data-refresh.service
sudo cp directives/operations/titan-data-refresh.timer.example \
       /etc/systemd/system/titan-data-refresh.timer

# Edit the service file to point at the right WorkingDirectory
sudo nano /etc/systemd/system/titan-data-refresh.service
#   set:  WorkingDirectory=/home/titan/Titan-IBKR

sudo systemctl daemon-reload
sudo systemctl enable --now titan-data-refresh.timer

# Verify
sudo systemctl list-timers | grep titan-data-refresh
sudo systemctl status titan-data-refresh.timer
```

The timer fires daily at 02:00 America/New_York. To force an immediate
run for testing:

```bash
sudo systemctl start titan-data-refresh.service
sudo journalctl -u titan-data-refresh.service -n 100
```

### 5.3 Confirm freshness check is alive

After the cron has run once, restart the strategy and look at startup
logs:

```bash
docker compose restart titan-portfolio
docker compose logs --tail=80 titan-portfolio | grep -A1 "freshness"
```

Expected:

```
[freshness] AUD_JPY_H1.parquet              fresh (1d old, last bar 2026-05-01)
[freshness] AUD_JPY_H4.parquet              fresh (1d old, last bar 2026-05-01)
[freshness] AUD_JPY_D.parquet               fresh (1d old, last bar 2026-05-01)
[freshness] AUD_JPY_W.parquet               fresh (1d old, last bar 2026-05-01)
[freshness] CSPX_D.parquet                  fresh (1d old, last bar 2026-05-01)
[freshness] IHYU_D.parquet                  fresh (1d old, last bar 2026-05-01)
```

If any line says `stale` (>2d) or `VERY stale` (>7d), check
`journalctl -u titan-data-refresh.service` — the cron has been failing.

---

## 6. Failure modes Phase 1 doesn't cover

These are explicitly accepted risks with the chosen design:

1. **Cron silently fails for >24h between checks.** Mitigation: the
   freshness check at strategy startup logs ERROR if any file >7d old.
   Operator notices on the next strategy restart. Could be tightened
   with a Prometheus / Grafana alert later, not in scope for Phase 1.

2. **Yahoo Finance breaks or rate-limits.** Yfinance reliability has
   historically been spotty. Mitigation: `download_data_databento.py`
   exists as an alternative if `DATABENTO_API_KEY` is set. The refresh
   script can be edited to switch sources.

3. **IBKR Gateway down at cron time.** AUD_JPY refresh fails for that
   day. Mitigation: the next day's cron will catch up since the
   merge-safe scripts only fetch the gap. Freshness check warns.

4. **The strategy receives bars between two cron runs that are never
   captured to parquet.** Phase 2 fixes this; Phase 1 accepts ~24h of
   "live-only" bars not on disk. Acceptable for current scale.

---

## 7. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-30 | Phase 1 (cron) over Phase 2 (journal) for VPS migration | Operational simplicity; current scale doesn't justify the journal complexity |
| 2026-04-30 | yfinance for ETFs, IBKR for FX | yfinance is free and adequate for daily bars; IBKR is the only reliable source for FX merge-safe |
| 2026-04-30 | 02:00 ET cron cadence | After US market close, before Sunday 03:00 ET Gateway weekly restart |
| 2026-04-30 | Freshness check non-blocking | Don't let a missing parquet prevent emergency live restarts; warn loudly instead |
