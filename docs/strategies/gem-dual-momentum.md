# GEM — Dual Momentum

**Version:** 2.1 | **Last updated:** 2026-05-16
**Status:** **LIVE on paper (J5 `P_hl60_vt05`)** — cutover 2026-05-16 17:10 UTC. 2 existing positions adopted cleanly from IBKR (no double-fill).
**Source:**
- Research mechanics: [research/gem/gem_strategy.py](../../research/gem/gem_strategy.py)
- Audit harnesses: [research/gem/run_gem_audit.py](../../research/gem/run_gem_audit.py) (J3/J4), [research/gem/run_gem_j5_reaudit.py](../../research/gem/run_gem_j5_reaudit.py) (J5)
- Live class: [titan/strategies/gem/](../../titan/strategies/gem/)
- Pre-reg directives: [Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md](../../directives/Pre-Reg%20J5%20GEM%20Hybrid%20Re-audit%202026-05-16.md) (current), [Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md](../../directives/Pre-Reg%20J4%20GEM%20Noise-Robust%20Redesign%202026-05-15.md), [Pre-Reg GEM Dual Momentum 2026-05-14.md](../../directives/Pre-Reg%20GEM%20Dual%20Momentum%202026-05-14.md)
- TOML configs: [config/gem_voltarget_lev2.toml](../../config/gem_voltarget_lev2.toml) (**LIVE J5**), [config/gem_voltarget_lev2_j5.toml](../../config/gem_voltarget_lev2_j5.toml) (sidecar snapshot for reference)
- Migration memo: [.tmp/reports/gem_j5_reaudit/findings.md](../../.tmp/reports/gem_j5_reaudit/findings.md)

---

## Executive Summary

GEM is Gary Antonacci's **Global Equity Momentum** — a slow, monthly-rebalanced cross-asset rotation between US equity (SPY), non-US equity (EFA), and US Treasury bonds (IEF) — extended with V2.0-framework risk overlays (multi-speed lookback blend, continuous vol-targeting, controlled leverage). The strategy holds exactly one of the three legs at a time, scaled by a trailing-vol overlay.

**J5 audit headline (2026-05-16, full V3.6 protocol incl. L17 rel-MC, 22y SPY/EFA/IEF, 60 WFO folds):**

| | J4 live `(halflife=40, vol_target=0.10)` | **J5 PROMOTED `(60, 0.05)`** | Delta |
|---|---|---|---|
| Stitched OOS Sharpe | +0.74 | **+1.00** | +0.26 (+34%) |
| CI95 lower bound | +0.24 | **+0.51** | 2.1x tighter |
| Rel-MC DD reduction vs 60/40 SPY/IEF | 0.88 (FAIL) | **0.62 (PASS)** | gate flip |
| P(strategy MaxDD < benchmark MaxDD) | 0.68 | 0.95 | — |
| Noise axis | best | best | unchanged |
| 5-axis verdict | CONDITIONAL_WATCHPOINT | **DEPLOY** | upgrade |
| Effective deployed capital | 100% (vt=0.10) | 50% (vt=0.05) | allocator action |

**Verdict: J5 PROMOTED — DEPLOY**, supersedes J4. Migration pending allocator sign-off (50% capital reduction needs rotation plan; recommend bond_gold CONDITIONAL or cash buffer).

---

## 0. Audit history

GEM has been through 4 framework audits:

| Audit | Date | Cell | Status |
|---|---|---|---|
| **C12** (15-cell sweep) | 2026-05-14 | `(halflife=N/A, rolling-std vol, vol_target=0.10, max_leverage=2.0)` | DEPLOY at the time |
| **J3** (5-axis re-audit) | 2026-05-15 | C12 same | DEMOTED to CONDITIONAL_WATCHPOINT (noise axis = mid) |
| **J4** (noise-robust redesign) | 2026-05-15 | `A1_ewma_hl40 (halflife=40, vt=0.10)` | PROMOTED to DEPLOY → **LIVE since 2026-05-15 13:07 UTC** |
| **J5** (hybrid re-audit, **this version**) | 2026-05-16 | `P_hl60_vt05 (halflife=60, vt=0.05)` | PROMOTED to DEPLOY, supersedes J4 (migration pending) |

The J5 audit applied the V3.6 hybrid framework (L52) — a 2D sweep over `(vol_estimator_halflife × ann_vol_target)`. J4's 1D sweep at fixed `vol_target=0.10` missed that the vol_target axis has steeper Sharpe sensitivity than the halflife axis. **L57** documents the mechanism: the `max_leverage=2.0` cap binds asymmetrically at high vol_target (truncating upside while leaving downside intact). At vol_target=0.05 the cap rarely binds → vol-targeting works as designed → Sharpe is preserved.

---

## 1. What it trades

GEM is **leg-aware**: the SPY leg can optionally route to MES futures for leverage, while the EFA and IEF legs stay as ETFs. The current live config uses ETF mode (`execution_mode="etf"`) so all three legs trade as ETFs.

| Logical role | US universe (default) | UK universe (UCITS, PRIIPs-compliant) |
|---|---|---|
| US equity | SPY (ARCA) | **CSPX.LSEETF** (iShares Core S&P 500 UCITS, USD, ISIN IE00B5BMR087) |
| Non-US equity | EFA (ARCA) | **IWDA.LSEETF** (iShares Core MSCI World UCITS, USD, ISIN IE00B4L5Y983) — *imperfect; ~65% US* |
| Safety asset | IEF (NASDAQ) | **IDTM.LSEETF** (iShares USD Treasury 7-10y UCITS, USD; yfinance ticker `IBTM.L`) |
| Regime signal (read-only) | VIX (CBOE), HYG (ARCA) | VIX (CBOE) — *HYG omitted: would hit PRIIPs* |

The UK substitution is required because UK retail accounts cannot trade US-listed ETFs under PRIIPs/KID rules. IWDA is documented as an imperfect EFA substitute — the strategy is effectively *CSPX-or-cash* under the UK universe because the EFA leg rarely wins this comparison.

---

## 2. How it trades

Decision pipeline, evaluated once at month-end, executed next session. **Mechanism is unchanged across all four audits (C12 → J3 → J4 → J5).** Only the vol-estimator path and vol-target value differ.

1. **Relative momentum (pick the equity winner)** — Compute trailing return over a multi-speed blend (3, 6, 12 months). Rank SPY and EFA by each lookback's return; average ranks; the winner is the candidate risk asset. A 0.5% buffer (`buffer_pct`) suppresses tiny incumbent-vs-challenger swings.
2. **Absolute momentum (defensive switch)** — If the candidate's trailing 12-month return is below IEF's trailing 12-month return, allocate 100% to IEF instead. This is what saved Antonacci's strategy in 2008.
3. **Continuous vol-target overlay** — Scale the chosen position so that the strategy's trailing realised vol (EWMA estimator, halflife = `vol_estimator_halflife`) equals `ann_vol_target` annualised. Scaling is bounded by `max_leverage` (2.0). Any unused capital sits in IEF earning carry.
4. **Live submission filter** — Only emit orders when any per-asset weight delta exceeds 5% (`rebalance_threshold_weight`). Suppresses ~70% of the daily vol-target tweaks that would otherwise hit the IBKR commission floor.

The strategy is **causal by construction** — every signal at bar `t` uses data through `t-1` only. A `gem_assert_causal` smoke test corrupts future closes and verifies past weights are unchanged.

---

## 3. Parameters — production cell (LIVE J5 `P_hl60_vt05`)

| Parameter | J4 (prior live, 2026-05-15 → 2026-05-16) | **J5 LIVE** (`P_hl60_vt05`, since 2026-05-16) | Notes |
|---|---:|---:|---|
| `lookback_blend_str` | "3,6,12" | "3,6,12" | Multi-speed momentum blend (unchanged) |
| `absolute_gate_lookback_months` | 12 | 12 | Canonical Antonacci defensive switch (unchanged) |
| `buffer_pct` | 0.005 | 0.005 | 0.5% buffer to reduce churn (unchanged) |
| `defensive_switch` | true | true | Allow IEF fallback when risk-off (unchanged) |
| **`ann_vol_target`** | 0.10 | **0.05** | **J5 CHANGE — 50% capital reduction** |
| `vol_lookback_days` | 20 | 20 | Unused at EWMA setting (unchanged) |
| `max_leverage` | 2.0 | 2.0 | 2× cap; **rarely binds at vt=0.05 (L57)** |
| `vol_estimator_kind` | "ewma" | "ewma" | Smoother vol estimator (J4 redesign) |
| **`vol_estimator_halflife`** | 40 | **60** | **J5 CHANGE — longer halflife = noise axis "best"** |
| `stress_gate_enabled` | false | false | Continuous vol-target dominates a binary gate (L19) |
| `dd_breaker_enabled` | false | false | Redundant on top of vol-target (L22) |
| `execution_mode` | "etf" | "etf" | Trade as ETFs (unchanged) |
| `rebalance_threshold_weight` | 0.05 | 0.05 | Don't submit unless per-asset delta > 5% |
| `warmup_bars` | 380 | 380 | 18 months — covers the longest lookback |
| `initial_equity` | 30 000 USD | 30 000 USD | Notional used for sizing + commission-floor math |

The J5 changes are confined to two knobs (`ann_vol_target`, `vol_estimator_halflife`). Everything else is preserved verbatim from J4. Rollback to J4 is a one-line git revert of `config/gem_voltarget_lev2.toml` + container restart.

---

## 4. Cost model

Calibrated against the live IBKR fill audit in [directives/Cost Model Audit 2026-05-11.md](../../directives/Cost%20Model%20Audit%202026-05-11.md). The model is **leg-aware** and includes per-fill commission floors. See [V3.6 L23](../../directives/V3.6%20Lessons%20Catalogue.md) for the lesson behind this.

| Component | ETF leg (SPY/CSPX, EFA/IWDA, IEF/IDTM) | SPY-as-MES leg (when execution_mode="mes") |
|---|---:|---:|
| Variable (spread + slip), one-way | 6.0 bps/turnover | 1.0 bps/turnover |
| Per-fill commission floor (IBKR Pro) | $1.00 / fill | $1.19 / contract / side |
| Per-fill commission floor (IBKR Lite alternative) | $4.00 / fill | (same as Pro for futures) |

**J5 cost-drag check (`C4_gross_no_costs` cell in the J5 audit):** OOS Sharpe gross = +1.028, net = +0.993 → cost drag at vt=0.05 is **+0.03 Sharpe**, well below the 0.10 H6 gate. **Lower vol_target = lower turnover → lower absolute cost drag in basis points.** Cost is not a binding constraint at J5.

---

## 5. Live deployment

### Containers

| Container | Image | Role | Port |
|---|---|---|---|
| `titan-ib-gateway` | `gnzsnz/ib-gateway:stable` | IBKR Gateway | 4004 (paper) / 4003 (live) |
| `titan-portfolio` | `titan-portfolio:latest` | Runs `scripts/watchdog_gem.py` → `scripts/run_live_gem.py` | — |

Bring-up: `docker compose --env-file .env.docker up -d --build`
Tail logs: `docker compose logs -f titan-portfolio`
Stop: `docker compose down`

See [directives/Docker Paper Trading Guide.md](../../directives/Docker%20Paper%20Trading%20Guide.md) for the full runbook.

### J5 migration plan — current status: **Phase 2 COMPLETE (live)**

| Phase | Action | Status |
|---|---|---|
| Phase 0 | Write `config/gem_voltarget_lev2_j5.toml` sidecar snapshot | **DONE 2026-05-16** |
| Phase 1 | 1-week paper shadow of J5 alongside J4 live | **SKIPPED per user instruction** (paper-only deployment makes the change reversible) |
| **Phase 2** | Live cutover via overwrite of `config/gem_voltarget_lev2.toml` + `docker compose restart titan-portfolio` | **DONE 2026-05-16 17:10 UTC** |
| Phase 3 | J6 audit after 4 weeks of J5 live operation | pending (target 2026-06-13) |

**Cutover verification (2026-05-16 17:10 UTC):**

- Container restart clean (`Up 1 minute` after `docker compose restart titan-portfolio`)
- Boot log confirms J5 params: `GEM started | execution=etf | blend=(3, 6, 12) | vol_target=0.05 | max_leverage=2.0 | warmup_bars=378`
- **2 existing positions adopted from IBKR** (no double-fill, no flip, no manual reconcile needed):
  - `CSPX.LSEETF net_position=27` (avg px 798.15 USD)
  - `IDTM.LSEETF net_position=45` (avg px 172.99 USD)
- ExecEngine reconciliation: `report.signed_decimal_qty == position_signed_decimal_qty` for both legs — IBKR truth and Nautilus cache match exactly.
- Daily bars subscribed for CSPX, IWDA, IDTM. Next rebalance evaluation: month-end + 15:30 ET timer.

**Expected first J5 trade:** at month-end the strategy will recompute target weights using vol_target=0.05 and halflife=60. The pre-J5 positions were sized for vol_target=0.10 → J5 will likely want to TRIM the equity leg (CSPX) by ~50%. Any per-asset delta exceeding the 5% `rebalance_threshold_weight` will emit a MOC order on the next session.

**Rollback procedure** if J5 misbehaves: restore `config/gem_voltarget_lev2.toml` from git history (the J4 A1_ewma_hl40 version was committed prior to 2026-05-16) and `docker compose restart titan-portfolio`. The 2 adopted positions stay; only the rebalance logic reverts.

### Required environment ([.env.docker](../../.env.docker))

| Variable | Example | Notes |
|---|---|---|
| `TWS_USERID` / `TWS_PASSWORD` | — | IBKR Gateway login. Paper credentials. |
| `TRADING_MODE` | `paper` | Set `live` only after the 30-day paper validation. |
| `IBKR_PORT` | `4004` | Paper API. Live uses `4003`. |
| `IBKR_CLIENT_ID_GEM` | `21` | Avoid 0, 1, 20 (ORB), 25, 98, 99. |
| `IBKR_CLIENT_ID_GEM_J5` | `22` *(proposed)* | For the Phase 1 paper shadow. |
| `IBKR_ACCOUNT_ID` | `DUxxxxxxx` | Paper account ID. |
| `UNIVERSE` | `uk` or `us` | UK swaps SPY/EFA/IEF → CSPX/IWDA/IDTM. |
| `TITAN_PORTFOLIO_USD_EQUITY` | optional | Override the 30k default at startup. |
| `SLACK_WEBHOOK_URL` | optional | Halt/alert notifications. |

### IBKR contract resolution

Under `UNIVERSE=uk` the runner submits these to the InstrumentProvider:
- `STK CSPX LSEETF USD` → ConId 76023663
- `STK IWDA LSEETF USD` → ConId 78999785
- `STK IDTM LSEETF USD` → ConId 68489992 *(not `IBTM` — that resolves to an unrelated US fund on IBKR)*
- `IND VIX CBOE USD` → ConId 13455763 (regime signal only, never traded)

---

## 6. Operations

### Warmup

On `on_start` the strategy loads the last 380 daily bars from
`data/{ticker}_D.parquet` (where ticker matches `cfg.ticker_spy/efa/ief/vix`).
Under UK: `data/CSPX_D.parquet`, `data/IWDA_D.parquet`, `data/IDTM_D.parquet`,
`data/VIX_D.parquet`. The loader renames physical-ticker columns to logical
SPY/EFA/IEF role keys before handing off to `GemLiveLogic`. Missing parquets
abort warmup with a `WARN` (the strategy then waits for live bars).

To refresh warmup parquets from yfinance: `uv run python scripts/refresh_market_data.py`
(append the UCITS tickers IBTM.L, CSPX.L, IWDA.L for the UK universe).

### Logs

Container stdout + a rolling file at `/.tmp/logs/gem_live_YYYYMMDD_HHMMSS.log`
inside the named volume. Headline lines to watch:

- `GEM Strategy attached (execution_mode=etf, max_leverage=2.0).` — boot.
- `Contract qualified for CSPX.LSEETF with ConId=…` — instrument resolution.
- `Warmup loaded N bars from parquets (VIX=yes, HYG=no).` — warmup OK.
- `RECONCILE <SYM>: BUY|SELL n (target=t, current=c, weight=w)` — order submission.
- `Order Message: BUY n CSPX LSEETF Warning: Your order will not be placed at the exchange until …` — benign LSE-session-not-open warning (code 399).

### Kill switch

`docker compose down` sends SIGTERM; the watchdog forwards to the trading node and the strategy exits cleanly within `stop_grace_period=60s`.

For an emergency flatten (no restart) without stopping the container:
```
docker compose exec titan-portfolio python -m titan.scripts.kill_switch
```
(uses a separate client id `98` so it doesn't collide with the live strategy's `21`).

### Restart safety

Position reconciliation is via broker query, not local cache. On
`on_start` the strategy adopts whatever positions IBKR reports for
its three instrument ids — research isn't rebuilt from local
state, so a crash-restart will not double-fill.

---

## 7. Audit summary (J5, 2026-05-16)

J5 hybrid re-audit on US universe (SPY+EFA+IEF), 2003-01-02 → 2026-04-02, 5,850 bars, 60 rolling WFO folds, L17 relative MC vs **60/40 SPY/IEF** buy-and-hold (NOT just B&H SPY — GEM rotates cross-asset).

Full result log: [.tmp/reports/gem_j5_reaudit/result_log.md](../../.tmp/reports/gem_j5_reaudit/result_log.md).
Migration memo: [.tmp/reports/gem_j5_reaudit/findings.md](../../.tmp/reports/gem_j5_reaudit/findings.md).

### Per-cell verdict matrix

| Cell | Sharpe | CI_lo | CI_hi | Rel-MC DD red | P(strat better) | Rel-MC pass | Sanc Sharpe | Sanc %ile | Noise | Verdict |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---|
| C1_canonical (hl20, vt0.05) | +0.99 | +0.47 | +1.49 | 0.61 | 0.95 | YES | +0.94 | 0.40 | mid | CONDITIONAL |
| P_hl10_vt05 | +1.01 | +0.50 | +1.51 | 0.61 | 0.94 | YES | +0.93 | 0.37 | mid | CONDITIONAL |
| P_hl40_vt05 | +0.99 | +0.49 | +1.49 | 0.60 | 0.95 | YES | +0.98 | 0.42 | best | DEPLOY |
| **P_hl60_vt05** ← PROMOTED | **+1.00** | **+0.51** | **+1.51** | **0.62** | **0.95** | **YES** | **+0.97** | **0.40** | **best** | **DEPLOY** |
| P_hl20_vt075 | +0.87 | +0.35 | +1.35 | 0.73 | 0.87 | YES | +0.83 | 0.44 | best | DEPLOY |
| C2_constrained (hl20, vt0.10) | +0.76 | +0.23 | +1.27 | 0.85 | 0.71 | no | +0.81 | 0.46 | best | CONDITIONAL |
| **C3_J4_live** (hl40, vt0.10) | **+0.74** | **+0.24** | **+1.24** | **0.88** | **0.68** | **no** | **+0.88** | **0.54** | **best** | **CONDITIONAL** |
| C4_gross (vt0.05, no costs) | +1.03 | +0.51 | +1.52 | 0.61 | 0.96 | YES | +0.97 | 0.40 | mid | CONDITIONAL |

### Falsification hypotheses

All six pre-committed hypotheses SUPPORTED. Plateau spread on stitched OOS = 13.72% (passes strict L27 30% gate). Canonical OOS Sharpe +0.99 (passes H2 ≥+0.50 gate). C1 OOS Sharpe (+0.99) > C3 J4 live OOS Sharpe (+0.74) — H3 supported. CI_lo +0.47 > 0 — H4 supported. Rel-MC ratio 0.62 ≤ 0.80 with p_better 0.95 ≥ 0.50 — H5 supported. Noise axis = best on the promoted cell — H6 supported.

### Why P_hl60_vt05 was promoted (not C1)

- Both have nearly identical Sharpe (~1.00).
- P_hl60_vt05 has noise axis = **best** (longer EWMA halflife → smoother vol estimate → noise-robust position scaling).
- C1 has noise axis = **mid** (halflife=20 is short enough that noise injection occasionally flips positions).
- 5-axis matrix: P_hl60 gets DEPLOY (5/5 best), C1 gets CONDITIONAL_WATCHPOINT (4/5 best).
- §3 selection rule picks highest CI_lo among DEPLOY/CONDITIONAL eligible cells; P_hl60_vt05 at CI_lo=+0.510 wins.

The hybrid framework worked as designed: **sweep was a prior**, **audit was the gate**, **audit refined the canonical within the OOS plateau** by surfacing the noise-axis differentiator that the IS-only sweep couldn't see.

---

## 8. Known caveats

1. **J4 live remains until allocator approves Phase 2.** The J5 verdict is research-promoted; live cutover requires the migration sign-off in [.tmp/reports/gem_j5_reaudit/findings.md](../../.tmp/reports/gem_j5_reaudit/findings.md). Currently in Phase 0.
2. **L57 (max_leverage cap asymmetry).** The `max_leverage=2.0` cap binds asymmetrically at high vol_target. At vol_target=0.10 (J4 live) this costs ~0.26 Sharpe vs the J5 (vt=0.05) operating point. Future strategies with vol-target overlays must sweep vol_target as one axis. See [V3.6 L57](../../directives/V3.6%20Lessons%20Catalogue.md).
3. **50% capital reduction on J5 migration.** vol_target=0.05 deploys ~50% of the capital that vt=0.10 does. Absolute PnL is ~50% of J4 despite higher Sharpe. Allocator must decide on the freed-capital destination before Phase 2. Recommended: rotate to `bond_gold` CONDITIONAL_WATCHPOINT (Wave A.1, see L52 hybrid workflow) or cash buffer.
4. **UK universe history starts 2010-09-14** (CSPX inception). Loses 2008 GFC stress test. Per-universe Sharpe is ~0.1 lower than US universe.
5. **IWDA is not pure EFA.** IWDA includes ~65% US large-caps, so the EFA leg in the UK universe is highly correlated with the SPY (CSPX) leg. In practice the relative-momentum step rarely picks EFA — the strategy is "CSPX or IDTM" most of the time.
6. **VIX bar is the only regime signal under UK.** HYG (US-listed) would hit PRIIPs even though the strategy never trades it, so the subscription is dropped. The credit-spread sub-signal of the stress gate is silently disabled — fine, because all promoted cells (C12/J4/J5) keep `stress_gate_enabled=false`.
7. **MES leverage path is paper-only until permissions are granted.** All promoted cells include `max_leverage=2.0` only matters when `execution_mode="mes"`. In ETF mode the effective cap is 1.0 (you cannot lever an ETF intraday without a margin loan). To unlock 2× live, switch `execution_mode="mes"`, populate `mes_instrument_id` with the current front-month contract id, and grant CME futures permissions on the account. Note: with J5's vt=0.05, the 2× cap is even less relevant — the strategy rarely scales above 1× anyway.
8. **Costs are calibrated for $30k notional.** Per-fill commission as a fraction of equity scales inversely with notional. At $10k the realistic drag is ~2× higher; at $100k it's ~0.4×. Re-run the audit with `COST_NOTIONAL_USD` set to your actual equity before reading the verdict.
9. **Rebalance threshold creates path dependency.** The 5% threshold means two runs with slightly different warmup periods can carry different "incumbent" weights for a few weeks. The strategy converges back to the target weight quickly; this is not a parity bug.

---

## 9. References

- Antonacci, Gary. *Dual Momentum Investing.* McGraw Hill, 2014.
- [directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md](../../directives/Pre-Reg%20J5%20GEM%20Hybrid%20Re-audit%202026-05-16.md) — Current pre-reg directive (V3.6 + L52 hybrid).
- [directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md](../../directives/Pre-Reg%20J4%20GEM%20Noise-Robust%20Redesign%202026-05-15.md) — J4 pre-reg (L31 EWMA estimator).
- [directives/Pre-Reg GEM Dual Momentum 2026-05-14.md](../../directives/Pre-Reg%20GEM%20Dual%20Momentum%202026-05-14.md) — Original 15-cell sweep (C12 lineage).
- [directives/Cost Model Audit 2026-05-11.md](../../directives/Cost%20Model%20Audit%202026-05-11.md) — live IBKR fill data behind the cost calibration.
- [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md) — L17 (rel-MC), L31 (J4 EWMA), L52 (hybrid framework), L57 (max_leverage cap asymmetry).
- [.tmp/reports/gem_j5_reaudit/findings.md](../../.tmp/reports/gem_j5_reaudit/findings.md) — J5 migration memo with allocator-approval checklist.
- [.tmp/reports/sweep_gem_hybrid/findings.md](../../.tmp/reports/sweep_gem_hybrid/findings.md) — L52 hybrid sweep findings.
