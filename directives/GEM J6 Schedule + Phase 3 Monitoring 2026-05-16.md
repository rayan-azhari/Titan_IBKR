# GEM J6 Re-audit Schedule + Phase 3 Live-Monitoring Plan

**Author:** rayanazhari (planner) + Claude orchestrator (Operator + Architect)
**Date committed:** 2026-05-16
**Type:** Operations memo. Defines the J5 → J6 audit cadence and the live-monitoring checks between now and the J6 audit date.
**Predecessors:** `directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md`, `.tmp/reports/gem_j5_reaudit/findings.md`.

---

## Context

GEM J5 `P_hl60_vt05` is **LIVE on paper** since 2026-05-16 17:10 UTC (replaced J4 A1_ewma_hl40 via in-place config overwrite + container restart). Per the J5 migration memo, Phase 3 is a J6 re-audit after **4 weeks of live operation** to confirm the J5 verdict holds with the post-cutover data slice. This memo sets the schedule + the in-between monitoring checks.

**J6 target audit date: 2026-06-13** (28 days from cutover).

---

## Phase 3 monitoring — between 2026-05-16 and 2026-06-13

### Daily checks (operator)

| Check | Method | What success looks like |
|---|---|---|
| Container health | `docker ps` | Both `titan-portfolio` and `titan-ib-gateway` Up |
| Strategy attached | `docker compose logs --since 24h titan-portfolio \| grep "GEM started"` | Boot-line with `vol_target=0.05 \| max_leverage=2.0 \| warmup_bars=378` present |
| No errors | `docker compose logs --since 24h titan-portfolio \| grep -E "Traceback\|ERROR\|FATAL"` | empty |
| Halt state | `docker compose exec titan-portfolio cat .tmp/portfolio_halt.json 2>/dev/null` | file missing OR `halted=false` |

### Weekly checks (Operator → Architect handoff)

| Check | Method | Action if abnormal |
|---|---|---|
| PnL realism | Look at IBKR account NLV; compare to baseline +/- 10% noise band | If NLV drops > 10% in a week without market context, escalate |
| Position size sanity | `docker compose logs --since 7d titan-portfolio \| grep "RECONCILE"` | Position deltas should be << J4's because vol_target=0.05 |
| Sharpe-tracking diff | Compute realised Sharpe over the live window; compare to OOS Sharpe +1.00 with CI95 [+0.47, +1.49] | If realised Sharpe < +0.47 after 2 weeks of live data, flag for early J6 |

### One-time checks (within 7 days of cutover, by 2026-05-23)

- [ ] Verify FIRST monthly rebalance fires correctly. J5 inherited J4's position sizes (CSPX qty=27 + IDTM qty=45) which were sized for `vol_target=0.10`; at month-end the strategy should TRIM equity by ~50% (target weights for vt=0.05).
- [ ] Confirm the rebalance order respects the 5% `rebalance_threshold_weight` floor (no spurious 1-share commission-burning orders).
- [ ] Sanity-check the first vol-target update — EWMA halflife=60 means it takes ~60 bars to fully ramp; the strategy will gradually shift size as the new EWMA estimate matures.

---

## J6 re-audit pre-conditions (must hold before 2026-06-13)

1. **Live operation:** J5 must have completed ≥ 4 weeks of paper-trading with no manual intervention (other than market data refresh).
2. **At least 1 monthly rebalance fired** (so we have post-cutover live trade data). If no monthly rebalance has fired by 2026-06-13 (e.g., the entire period stayed at flat positions), defer J6 to 2026-07-04.
3. **No halt events.** If `portfolio_halt.json` fires during the period, treat as an out-of-band incident; root-cause BEFORE running J6 (the audit should be on undisturbed data).
4. **Data freshness:** `data/SPY_D.parquet`, `data/EFA_D.parquet`, `data/IEF_D.parquet` updated within 2 days of the J6 run.

---

## J6 audit scope (pre-committed)

**J6 IS a confirmation audit, NOT a fresh L52 hybrid sweep.** The 2D sweep was already done in J5; the question for J6 is: "does the J5 verdict hold with 4 weeks of additional data?"

### Audit harness

Run `research/gem/run_gem_j5_reaudit.py` AS-IS (no code changes). The harness re-loads fresh data, runs the same WFO + 5-axis audit on the SAME cells. The sanctuary window will naturally roll forward by 4 weeks (now 13mo instead of 12mo).

### Pass/fail criteria

| Axis | Pass criterion | If fail |
|---|---|---|
| C1_canonical stitched OOS Sharpe | ≥ +0.50 (vs J5 OOS +0.99) | If 0 < Sharpe < +0.50: re-classify as CONDITIONAL_WATCHPOINT; if Sharpe ≤ 0: emergency revert to J4 |
| C1 CI_lo | ≥ +0.20 (vs J5 CI_lo +0.47) | If CI_lo drops below +0.20: investigate; possible regime shift |
| C1 5-axis verdict | DEPLOY | If demotes to CONDITIONAL: keep live but flag for J7 |
| Plateau spread | ≤ 30% | If > 30%: serious — L43 knife-edge emerging; emergency revert |
| Live config in result_log | matches the audit-promoted canonical | If they diverge: config drift incident; halt and reconcile |

### Reporting

Write `.tmp/reports/gem_j6_reaudit/result_log.md` and `.tmp/reports/gem_j6_reaudit/findings.md`. If J6 confirms J5:
- Update `docs/strategies/gem-dual-momentum.md` to v2.2 noting J6-confirmation.
- Schedule J7 for **2026-09-13** (3 months after J6) — at that cadence the strategy gets a quarterly re-audit going forward.

If J6 demotes / rejects J5:
- Apply the rollback procedure documented in `docs/strategies/gem-dual-momentum.md` §5 (git revert config + container restart).
- Run a fresh L52 hybrid sweep (this becomes J7 by emergency).
- Document the failure mode in the lessons catalogue.

---

## Rollback procedure (emergency)

If J5 misbehaves during Phase 3 (large drawdown, repeated halts, or J6 emergency demotion), revert to J4 immediately:

```bash
# 1. Restore J4 config from git history
cd /path/to/Titan-IBKR
git log --oneline config/gem_voltarget_lev2.toml | head -5
# Identify the J4 commit (pre-2026-05-16); copy the SHA
git show <SHA>:config/gem_voltarget_lev2.toml > config/gem_voltarget_lev2.toml

# 2. Restart the container
docker compose restart titan-portfolio

# 3. Verify J4 params in the boot log
docker compose logs --since 2m titan-portfolio | grep "GEM started"
# Expected: vol_target=0.10 | max_leverage=2.0
```

The 2 IBKR-side positions stay; only the rebalance logic reverts.

---

## Status

- **Phase 0 + 2 (cutover):** DONE 2026-05-16.
- **Phase 3 (live monitoring):** in progress, 2026-05-16 → 2026-06-13.
- **J6 audit:** scheduled 2026-06-13. Run `research/gem/run_gem_j5_reaudit.py` as-is.
