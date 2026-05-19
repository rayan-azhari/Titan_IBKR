# Pre-Registration — F3: FOMC Pre-Announcement Drift (Lucca-Moench 2015)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-19
**Branch:** v2-main → research/f3-fomc-preannouncement
**Type:** Strategy audit (proposed class `CALENDAR_ANOMALY` — first instance; falls back to `DAILY_MEAN_REVERSION` with calendar gate)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE any FOMC-conditional return statistic has been computed on our SPY dataset. Lucca & Moench's published findings + SPY daily prices are public; our prior knowledge of broad equity markets is common knowledge. No FOMC-window Sharpe, hit-rate, or drift magnitude has been inspected for OUR specific universe slice. The 2015-2025 sub-period is in our visible data.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` row F3 — FOMC pre-announcement drift (1d, free data, `CALENDAR_ANOMALY`).

**Source.** Lucca, D., & Moench, E. (2015). *"The Pre-FOMC Announcement Drift."* Journal of Finance, 70(1), 329–371.

**Key claim (1994-2011 sample in original paper):** SPX cash-index abnormal cumulative log return averages roughly +33 bps in the 24 hours preceding scheduled FOMC announcements (specifically: 14:00 ET on day before → 14:00 ET on announcement day). The drift is concentrated in the overnight session before the announcement; the announcement-day morning contributes a smaller share. Statistically significant t > 5 in the original sample. **Post-2014 decay:** mixed reports — some authors (Cieslak-Vissing-Jørgensen 2021) report the effect attenuates after the policy regime shift around 2014 + Fed transparency reforms.

**Hypothesis (falsifiable).** A long-only SPY position entered at the close of the day BEFORE a scheduled FOMC announcement and exited at the close of the announcement day (or the immediately following day, depending on the cell) earns positive risk-adjusted returns over 2010-01-01 → sanctuary cutoff, with per-trade Sharpe materially above zero AND aggregate annualised Sharpe at the proposed deployment weight clearing the L66 + L17 + L65 + L67 gates.

**Specific falsification triggers:**

- If `per_trade_Sharpe(canonical_cell) <= 0` or the cell's annualised Sharpe `CI_lo <= 0`: RETIRE-tentative pending broader window.
- If `L52 plateau spread > 50%` across the 3×3 cell grid in §2: RETIRE per L52 strict.
- If `L65 single-strategy ruin` fails at proposed weight: RETIRE.
- If `L67 10-metric` adds fewer than +0.05 portfolio Sharpe vs current GEM/Turtle/IC_top3 baseline AND drops any currently-passing metric below benchmark: bench (CONDITIONAL/diversifier-only verdict).
- **L72 reminder:** AUC/event-hit-rate is NOT a deployment signal; the per-trade strategy-return distribution must clear the 5-axis matrix.

**Mechanism.**

1. **Universe.** SPY ETF (`SPY.ARCA-1-DAY-LAST-EXTERNAL` daily bars). Lucca-Moench used SPX cash; SPY is the natural retail-implementable proxy with negligible tracking error at daily frequency.

2. **Calendar feed.** FOMC scheduled-meeting announcement dates from the official Federal Reserve website (`federalreserve.gov/monetarypolicy/fomccalendars.htm`). These dates are public and stable — no data quality risk. Universe of "scheduled" announcements only (8 per year). **EXCLUDE intermeeting / emergency announcements** (e.g., 2020-03-03, 2020-03-15) — the strategy is for known anticipated meetings, not crisis surprises.

3. **Signal.** Binary calendar gate:
   - `signal_t = 1` if `t` is the close BEFORE a scheduled FOMC announcement (i.e., `t+1` IS an FOMC announcement day).
   - `signal_t = 0` otherwise.

4. **Position rule (per cell):**
   - **Entry**: at `close_t` of the day before announcement (signal = 1).
   - **Exit**: at `close_{t+hold_days}` where `hold_days ∈ {1, 2, 3}`. Per Lucca-Moench, `hold_days=1` (exit at announcement-day close) is the canonical configuration; longer holds test whether drift persists post-announcement.
   - **Size**: fixed notional fraction of portfolio NAV per trade (vol-targeted variant tested as a sub-cell).

5. **Causality.** Position effective from `t` (the day before announcement) AT CLOSE forward to `t+hold_days` AT CLOSE. The signal is computed from the FOMC calendar (frozen at audit time, no look-ahead). L21 smoke: corrupting future SPY prices must not change past trade outcomes.

6. **Class assignment.** Provisionally `CALENDAR_ANOMALY` (NEW StrategyClass to add to `titan/research/framework/typology.py` if F3 promotes). Falls back to `DAILY_MEAN_REVERSION` (closest existing class) for the framework defaults (`wfo`, MC config) — using DAILY_MEAN_REVERSION's class defaults for the §2 sweep keeps the framework pre-flight runnable WITHOUT introducing a new class as a side-effect of the audit.

**Why this is novel for our stack.** No calendar-anomaly strategy exists in `titan/strategies/`. The closest pattern is the `daily_summary` passive ops strategy, which uses calendar gating to trigger Slack messages — but does not trade. F3 introduces:

- First scheduled-event-driven trading strategy.
- Tests whether the V3.7 framework primitives (L52 sweep, DSR, L65 ruin) generalise to sparse-trade strategies (~8 events/year ≈ ~64 events across 8-year sanctuary-visible window).
- Bridges to F2 (CoT) and F4 (ETF-flow) — both also calendar/event-driven.

---

## §2. Cell sweep (V3.7 hybrid mandatory per L75)

3×3 grid pre-committed BEFORE any FOMC-conditional return is computed:

| dim | values |
|---|---|
| `hold_days` | {1, 2, 3} |
| `entry_offset_days` | {0, 1, 2} — entry at close of day before announcement (canonical=0), or two/three days before |

Total: **9 cells.** Canonical (Lucca-Moench reference): `(hold_days=1, entry_offset_days=0)` — enter at close of T-1, exit at close of T where T is the announcement date.

**Plateau set:** canonical + its 4 immediate neighbours (one step in each dimension). Plateau spread = (max − min) / |mean|. L52 gates:

- H1 plateau gate: spread ≤ 50%
- Strict plateau gate: spread ≤ 30%

**Per-cell metrics computed:**

- Per-trade Sharpe (sparse-trade primary metric, per L06).
- Annualised Sharpe (with explicit `periods_per_year=252`, per L60).
- Bootstrap CI 95% (n_resamples=2000, seed=42).
- Hit rate (fraction of trades with net_ret > 0).
- Max drawdown.

**Per-cell costs.** SPY at retail IBKR commissions: `cost_bps = 0.5` per turnover (conservative). With ~8 entries per year and `hold_days=1`, total turnover ~16/year — negligible drag.

**DSR.** Bailey & Lopez de Prado 2014 deflation across the 9 cells using `sr_var_from_sweep`. Canonical cell's `dsr.dsr_prob >= 0.95` required.

---

## §3. Gates & decision rule

The audit is the V3.7 hybrid + L65 + L67 closure in one script, per L75:

### §3.1 Pre-flight L52 plateau gate (per L75)

Canonical cell + 4 neighbours form the plateau set. Compute their stitched-OOS Sharpes; require:

- Plateau **spread ≤ 50%** (H1 gate).
- If `≤ 30%` (strict), upgrade plateau status to "robust".

**Failure mode.** If `spread > 50%`, abort to RETIRE: the canonical cell's Sharpe is a point-selection artefact.

### §3.2 5-axis decision matrix (V3.6 standard, L46)

| Axis | Gate | Source |
|---|---|---|
| 1. CI_lo | annualised Sharpe CI_lo > 0 | bootstrap_sharpe_ci |
| 2. DSR | `dsr.dsr_prob >= 0.95` (deflated over 9 cells) | deflated_sharpe |
| 3. MC abs DD | P(MaxDD > 35%) < 10% | run_block_mc |
| 4. Sanctuary divergence | Sanctuary Sharpe vs visible Sharpe within ±0.3 | sanctuary_divergence_test |
| 5. Noise robustness | mean degradation < 30% at σ=0.5; worst-case > 0 | run_noise_robustness |

### §3.3 Baseline & relative MC (L17 / L66)

**L66 baseline class:** Per the pre-flight, F3 is a long-only equity exposure during a specific window. The baseline is NOT 60/40 (the strategy is barely-deployed; portfolio-context comparison is L67's job). The L66 baseline is **cash (zero)** + the requirement that `Sharpe_annualised > 0 AND CI_lo > -0.5` (the standard L66 long-only-equity-class minimum).

**L17 relative MC vs 60/40 SPY/IEF:** APPLIED if §3.2 gates pass. Bootstrap blocks size 63, 200 paths. Gate: median dd_reduction ≤ 0.80 AND p_strategy_better ≥ 0.50. **Critical for L17:** F3 is in-market only ~16 days/year out of 252 ≈ 6.3%; the rel-MC dd_reduction comparison is over the FULL CALENDAR (not just FOMC days), so the strategy's cash exposure for 94% of the time naturally produces dd_reduction close to 1.0. This is a **known L17 quirk for sparse-trade strategies** — interpret the gate ACCORDINGLY: an F3 dd_reduction of 0.95 vs B&H 1.00 is materially different from a continuously-deployed strategy at 0.95. We will report rel-MC but **NOT use it as a hard gate** for F3 specifically; the relevant DD test is the absolute MC gate (§3.2 axis 3) on the actual trade-day returns.

This is a pre-registered interpretation, not a post-hoc relaxation.

### §3.4 L65 single-strategy ruin

At proposed deployment weights {0.02, 0.05, 0.10} (low — sparse trades, 8 events/year, so even at 10% portfolio weight the per-year exposure is small). Gate per assess_strategy_ruin: P_kill_trip < 1% AND p95 DD at size > -25%.

### §3.5 L65 joint ruin

With current LIVE strategies (gem_j5 + turtle_cat_c3peak + ic_top3 basket — i1v2 is SHADOW, excluded). Candidate weights: {(G75/T15/IC10/F0), (G75/T15/IC10/F2), (G73/T13/IC10/F4)}.

### §3.6 L67 10-metric portfolio inclusion

Vs 60/40 SPY/IEF. CURRENT (GEM80/T20) baseline = 7/10, portfolio Sharpe +0.93. PROPOSED = add F3 at the §3.4-passing weight. Verdict gate: **≥ 8/10 OR a Sharpe lift ≥ 0.05 with no metric regression** (matching the ic_top3 deployment threshold).

### §3.7 Decision tree (pre-committed)

```
plateau spread > 50%          → RETIRE (L52 strict)
DSR p < 0.95                  → RETIRE (selection bias dominates)
CI_lo <= 0                    → RETIRE (no edge)
MC absolute DD fails          → RETIRE
sanctuary divergence outside  → RETIRE (look-ahead suspected)
noise mean fails              → CONDITIONAL_WATCHPOINT (J3 axis)
L65 single FAIL               → RETIRE
L65 joint FAIL                → RETIRE
L67 < 7/10 AND Sharpe lift < 0 → bench (CONDITIONAL_WATCHPOINT, paper-only)
L67 maintains 7/10 + Sharpe lift ≥ 0.05 → DEPLOY-eligible
otherwise                     → CONDITIONAL_WATCHPOINT
```

### §3.8 Sanctuary

12-month hold-out at the END of the visible window (per V3.6 standard L15). Visible = 2010-01-01 → 2025-04-30; sanctuary = 2025-05-01 → 2026-04-30. Sweep + plateau + DSR + L65 + L67 all run on VISIBLE only; sanctuary divergence test then validates the canonical cell.

---

## §4. Result log

(To be appended after audit run. Empty at pre-reg commit.)

---

## §5. Known caveats (acknowledged at pre-reg)

1. **Post-2014 decay risk.** Cieslak-Vissing-Jørgensen (2021) and others suggest the pre-FOMC drift attenuated after 2014. The audit window (2010 → sanctuary) splits 50/50 across the pre-2014 and post-2014 regimes; expect lower magnitude than Lucca-Moench's headline ~33 bps. Decay would manifest as: lower per-trade Sharpe in the back half of the sample, possibly negative CI_lo, possibly plateau failure if the cells diverge across regimes.

2. **Survivorship-free SPY.** SPY is an index ETF — no survivorship bias risk at the security level. The only data-quality risk is the FOMC calendar source. The Fed publishes scheduled-meeting dates back to 1994; we cross-check with Wikipedia for any discrepancies.

3. **Intermeeting / emergency announcements excluded.** Lucca-Moench's paper focuses on SCHEDULED meetings; the strategy follows that scope. Emergency cuts (2020-03-03, 2020-03-15) are NOT in the trading universe — including them would be a different strategy.

4. **Sparse-trade L17 quirk.** As noted in §3.3, the L17 rel-MC dd_reduction interpretation must account for cash exposure 94% of the time. Pre-committed: not a hard gate for F3.

5. **Small effective sample.** 8 events/year × 15 visible years × ~50% post-2014 efficacy concern = ~60-120 useful trades. The bootstrap CI95 width will reflect this; expect wider CIs than continuous strategies.

6. **No new StrategyClass commit YET.** §1 mechanism uses `DAILY_MEAN_REVERSION` defaults via the framework. If F3 promotes, a follow-up adds `CALENDAR_ANOMALY` to `typology.py` with class-specific defaults (per L11 — class-specific MC gates and WFO shapes).

---

## §6. Implementation plan

1. **Pre-reg commit (THIS commit).** Freeze §1–§3.
2. **FOMC calendar fetch** (`scripts/fetch_fomc_calendar.py` or inline in audit script): pull scheduled-meeting dates from a stable public source. Cache to `data/fomc_announcements.csv`.
3. **Audit script** (`research/exploration/audit_f3_fomc.py`): runs §2 sweep + §3 gates end-to-end. Outputs `.tmp/reports/f3_fomc/result_log.md`.
4. **Joint L65 + L67** (per L75 mandatory hybrid): integrated into the audit script.
5. **Append §4 result log** to this directive with the audit verdict.
6. **Catalogue + Retirement Registry** update if RETIRE; deploy + L73 cutover-diff if DEPLOY.

---

## §7. Discipline crosscheck (V3.6 / V3.7 lessons)

- **L11 (class defaults):** F3 uses `DAILY_MEAN_REVERSION` defaults provisionally; promotes to `CALENDAR_ANOMALY` only if it deploys.
- **L21 (causality smoke):** mandatory before any sweep. Corrupt future SPY prices, assert past trade outputs bit-exact unchanged.
- **L52 (plateau pre-flight):** mandatory per L75.
- **L60 (annualisation):** `periods_per_year=252` explicit on every Sharpe / vol / annualisation. NO hardcoded `sqrt(252)`; route through `titan.research.metrics`.
- **L66 (baseline class declaration):** cash baseline for the standalone trade returns; 60/40 baseline for the L67 portfolio test.
- **L67 (portfolio inclusion):** ≥7/10 OR +0.05 Sharpe lift, no metric regression.
- **L72 (AUC ≠ Sharpe):** hit-rate is informational, not a gate.
- **L73 (cutover diff):** if F3 deploys, the L73 checklist applies for the v37_live cutover.
- **L74 (carry-premium):** N/A — F3 is not a carry strategy.
- **L75 (hybrid pre-flight mandatory):** §2 sweep + §3 plateau + DSR run BEFORE any deployment-eligible claim.
