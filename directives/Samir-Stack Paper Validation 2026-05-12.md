# Samir-Stack 4-Week Paper Validation Runbook

**Created:** 2026-05-12
**Strategy:** `samir_stack_paper` (registry key in `scripts/run_portfolio.py`)
**Strategy set:** `samir_validation`
**Promotion target:** add to `champion_portfolio` after 4 clean weeks

---

## 1. Purpose

The Samir-Stack research is locked in (PR #11) and the live strategy
class is feature-complete (PRs #19, #20). What's missing is **time on
real broker infrastructure**: paper-account fills, daily bar
reconciliation, container restarts, IBKR maintenance windows, and
edge-case behaviour the backtest can't simulate.

This runbook defines the paper-validation gate that promotes
`samir_stack_paper` from a registered-but-unused entry into the live
champion portfolio.

---

## 2. v1 deployment configuration

The paper-validation entry uses the **simplest possible config** — Phase 1
behaviour only. Phase 2 (MES futures) and Phase 3 (bond rotation) are
enabled in code but turned **off** in the paper-validation registry
entry. Layering them in comes after a clean v1 baseline.

| Component | v1 Choice | Why |
|---|---|---|
| Equity sleeve | CSPX (margin L=3) | Already deployed; same ETF the bond_equity strategies use |
| Bond sleeve | IEF (single instrument) | US 7-10Y Treasuries; ARCA-listed; rotation off |
| Equity engine | `equity_is_future=False` | Defer MES futures rollover handling |
| Bond rotation | `bond_rotation_instruments=()` | Defer for v1 |
| VIX subscription | **none** | Avoids paid IBKR market-data tier; regime score NaN-skips VIX cleanly |
| Vol target | 8% annualised | Champion config |
| Capital split | 10% equity / 90% bond | Champion config |
| L_max | 3.0 | Champion config |

---

## 3. Pre-deployment checklist

Run each of these before kicking off the validation period.

| # | Check | How |
|---|---|---|
| 1 | Confirm IEF, SPY, HYG ETFs are tradeable on the IBKR paper account | Manually search for them in TWS or the IB gateway log |
| 2 | Refresh warmup parquets for SPY, IEF, HYG | `uv run python scripts/download_data_yfinance.py --symbols SPY IEF HYG --start 2003-01-01` |
| 3 | Verify `^VIX_D.parquet` exists for warmup (live VIX subscription is not used) | `ls data/^VIX_D.parquet` — if missing, run yfinance with `--symbols ^VIX` |
| 4 | Set `TITAN_FX_USD_TO_GBP` env var if account base ccy != USD | `export TITAN_FX_USD_TO_GBP=0.78` (current rate) — required for FX-aware sizing |
| 5 | Smoke-test the registry entry locally | `uv run python -c "from scripts.run_portfolio import STRATEGY_REGISTRY as R; print(R['samir_stack_paper']['contracts'])"` |
| 6 | Run `tests/test_samir_stack_strategy.py` and the broader suite | `uv run pytest tests/ -q` should be 230+ tests passing |
| 7 | Confirm `samir_stack_paper` is **not** in `champion_portfolio` (only in `samir_validation`) | grep run_portfolio.py |
| 8 | Snapshot pre-deployment broker positions | `uv run python scripts/smoke_double_restart.py --dry-run` (snapshot only, no chaos) |
| 9 | Verify NLV on paper account is within target (recommend > $30k for clean PDT headroom) | TWS account window |
| 10 | Confirm operational watchdogs are running | `docker logs titan-portfolio --since 10m | grep "Reconciliation watchdog started"` |

---

## 4. Deployment procedure

```bash
# 1. Bring down the current container (graceful — watchdog forwards SIGTERM)
docker compose stop titan-portfolio

# 2. Switch to the validation strategy set in compose env
echo 'TITAN_STRATEGIES=samir_validation' >> .env.docker
# OR pass --strategies samir_validation explicitly via the watchdog command

# 3. Start the container
docker compose up -d titan-portfolio

# 4. Tail the startup logs for the new strategy's banner
docker logs -f titan-portfolio | grep -E "Samir-Stack started|REHYDRATED Samir-Stack"

# 5. Verify all expected bar subscriptions are active
docker logs titan-portfolio --since 5m | grep "Subscribed.*-1-DAY-LAST-EXTERNAL"
# Should see: CSPX, IEF, SPY, HYG, AUD/JPY, IHYU, VUSD, IHYG, EIMI

# 6. Run the smoke double-restart to confirm rehydration is clean
uv run python scripts/smoke_double_restart.py --healthy-timeout 120
# Expect exit code 0 with snapshot stable across both restarts

# 7. Confirm reconciliation watchdog finds no D5 alerts
docker logs titan-portfolio --since 5m | grep -E "\[D5"
# Should be empty
```

---

## 5. Daily monitoring (week 1-4)

**Slack alerts to watch:**
- `[D5 shadow samir_stack_paper]` — strategy/cache disagreement
- `[D1 multi-position]` — orphan + strategy double-up
- `Order REJECTED` — strategy entered but IBKR refused
- `cost_model_drift` (Sun 02:00) — modelled vs realised commission
- `replay_audit_mismatch` (Sun 03:00) — backtest-vs-live decision diff

**Weekly review (every Sunday after the cron audits run):**

```bash
# 1. Pull the cost-audit summary
ls .tmp/reports/cost_audit/ | tail -1
cat ".tmp/reports/cost_audit/$(ls -t .tmp/reports/cost_audit/ | head -1)"

# 2. Pull the replay-audit summary
uv run python scripts/replay_audit.py --strategy samir_stack_paper --days 7 --verbose

# 3. Snapshot positions and compare to last week
uv run python scripts/smoke_double_restart.py --dry-run
diff .tmp/smoke_double_restart.json .tmp/smoke_double_restart_lastweek.json

# 4. Run the chaos harness (confirms no degradation in resilience)
uv run python scripts/chaos_harness.py --scenario S2 --healthy-timeout 120
```

**Per-week log to keep:**

| Week | Days clean | D5 alerts | Replay mismatches | Cost drift % | Notes |
|---|---|---|---|---|---|
| 1 | / 7 |   |   |   |   |
| 2 | / 7 |   |   |   |   |
| 3 | / 7 |   |   |   |   |
| 4 | / 7 |   |   |   |   |

---

## 6. Promotion gate

After 4 weeks, the strategy promotes to `champion_portfolio` if **all
five conditions** hold:

1. **Zero unrecovered D5 alerts** in `docker logs` over the 4-week
   window. (Transient alerts that resolve in <2 hours are tolerable;
   persistent alerts that require manual intervention are not.)
2. **Zero actionable replay-audit mismatches** across the 4 weekly
   `replay_audit.py --strategy samir_stack_paper` runs.
3. **Cost-model drift ≤ 30%** in 3 of the 4 weekly cost-audit runs
   (one outlier week tolerable for early-validation noise).
4. **No PDT or other broker rejections** of the strategy's orders
   (operator manually verifies via the global rejection notifier from T1.4).
5. **Manual eyeballing of strategy actions vs research expectations**:
   pick one rebalance per week, check that the tier / vol-scale / order
   sizes match what the research backtest would have produced for the
   same regime score and NLV.

If any gate fails, **do NOT promote**. Document the failure in this
file's section 9, fix root cause, restart the 4-week clock.

---

## 7. Promotion procedure

```bash
# 1. Edit scripts/run_portfolio.py — add "samir_stack_paper" to
#    champion_portfolio set
# 2. Open PR titled "feat(samir_stack): promote to champion_portfolio
#    after clean 4-week validation"
# 3. After merge: docker compose restart titan-portfolio
# 4. Update directives/System Status and Roadmap.md to reflect the
#    new live deployment
```

---

## 8. Rollback procedure

If validation is unstable mid-period:

```bash
# 1. Stop the container
docker compose stop titan-portfolio

# 2. Revert TITAN_STRATEGIES to champion_portfolio (drops samir_stack)
sed -i 's/TITAN_STRATEGIES=samir_validation/TITAN_STRATEGIES=champion_portfolio/' .env.docker

# 3. Manually flatten any open Samir-Stack positions before restarting
uv run python scripts/close_orphans.py CSPX IEF
# (only if Samir-Stack actually opened positions; check first)

# 4. Restart on the original champion set
docker compose up -d titan-portfolio
```

---

## 9. Known-incident log

Use this section to track any issue that arises during the validation
window. Each entry is evidence for the promotion-gate decision.

_(empty — first validation cycle starts on deployment date)_

---

## 10. After promotion: layering Phase 2/3

Once `samir_stack_paper` is in `champion_portfolio`, the path forward:

- **Month 2:** Add `equity_is_future=True` + manual quarterly rollover
  for MES futures. Re-run a 2-week paper sub-validation focused on
  contract sizing + roll handling.
- **Month 3:** Add `bond_rotation_instruments=("IEF.ARCA", "HYG.ARCA")`
  + `bond_rotation_lookback_days=60`. Re-run a 2-week sub-validation
  focused on rotation transitions.
- **Month 4+:** Capitulation overlay re-tune + multi-strategy
  combination work (see directive §13.8).

Each phase is its own PR, its own paper-validation cycle, its own
known-incident log entry in this file.
