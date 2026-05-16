# Pre-Registration — J5 GEM Hybrid Re-audit (V3.6 + L52)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** Confirmed-live strategy re-test under L52 hybrid framework.
**Strategy class:** `CROSS_ASSET_MOMENTUM`
**Predecessors:**

- Pre-Reg GEM Dual Momentum 2026-05-14 (C12 selected via 15-cell sweep).
- Pre-Reg GEM J3 Noise-Robust Audit 2026-05-15 (C12 demoted to CONDITIONAL_WATCHPOINT — noise axis "mid").
- Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15 (A1_ewma_hl40 promoted to DEPLOY; live since 13:07 UTC).
- L52 hybrid sweep 2026-05-16 (`.tmp/reports/sweep_gem_hybrid/findings.md`) — identified plateau at `(halflife=20, vol_target=0.05)` with +9.8% IS Sharpe over live canonical.

**Status:** §1–§3 frozen at this commit. §4 result log appended after the audit harness runs.

> **V3.1 pre-registration.** §1–§3 are frozen BEFORE the audit examines the 12-month sanctuary window. **Hypothesis being tested: a V3.6-correct GEM canonical chosen from the L52 hybrid sweep — operating at `(vol_estimator_halflife=20, ann_vol_target=0.05)` — achieves higher Sharpe AND tighter CI_lo than the J4 live canonical `(40, 0.10)` on the held-out sanctuary, while passing all 5 axes.**
>
> **L52 hybrid step:** sweep performed on IS-only data with 12mo sanctuary held out. Pre-reg follows directly; audit on sanctuary is the deployment gate. The live config is the parity baseline.

---

## §1. Motivation & mechanism

**Why a hybrid re-audit on a confirmed-live strategy.** J4 was a 1D sweep over `vol_estimator_halflife` at fixed `ann_vol_target=0.10`. The L52 hybrid sweep (2026-05-16) extended to 2D over `(halflife × vol_target)` and found:

1. **The vol_target axis is steeper than the halflife axis.** Lower vol_target uniformly produces higher Sharpe across all halflife values.
2. **The plateau is at `vol_target=0.05`.** Spread along the vol_target=0.05 column is only 5% across halflife ∈ {10, 20, 40, 60, 100, 160} — flat plateau.
3. **The live `(40, 0.10)` is OFF the plateau** — IS Sharpe 0.750 vs plateau centre (20, 0.05) at 0.824 (+9.8% gap).

**Mechanism (unchanged from J4 — same `gem_strategy.py`).** Per bar t:

1. Multi-speed momentum signal: rank SPY / EFA / IEF by 3mo, 6mo, 12mo returns; average ranks; pick winner.
2. Defensive switch: if neither risk asset (SPY, EFA) beats IEF on 12mo, allocate to IEF.
3. Buffer: switch only if new winner beats incumbent by ≥ 0.5% on rank-average return.
4. Vol-target overlay: scale position so trailing realised vol matches `ann_vol_target`. Scale capped at `max_leverage=2.0`. Excess capital → IEF.
5. EWMA realised-vol estimator with halflife `vol_estimator_halflife` (J4 mitigation A — replaces 20-day rolling-std).

**Causality (L04 / A1).** Decision at close[t] earns return at t+1 via `weights = raw_decisions.shift(1)`. Unchanged from J4.

**Why the vol_target plateau exists (mechanistic explanation):** the `max_leverage=2.0` cap binds asymmetrically. At low vol_target the strategy rarely needs to lever up; at high vol_target the strategy WANTS to lever above 2x in low-vol regimes but is capped → upside truncated, downside intact → Sharpe degrades. The (20, 0.05) plateau corresponds to the strategy's "unconstrained" operating regime.

## §2. Universe + audit configurations

**Universe.** Same as J4: SPY + EFA + IEF daily TR closes from yfinance (`data/{SPY,EFA,IEF}_D.parquet`).

**Date range.** 2003-01-02 → 2026-04-02 (common intersection, 5,850 bars).

**Sanctuary.** Last 12 months held out: 2025-04-02 → 2026-04-02 (252 bars). Identical to the J4 sanctuary boundary. **The L52 sweep saw the IS window only.**

**Visible window for WFO.** 5,599 bars (2003-01-02 → 2025-04-02), ~22.2 years.

**WFO.** `CROSS_ASSET_MOMENTUM` class default: `is_min_years=2.0, oos_years=0.5, fold_count=8, is_mode="rolling", stride_overlap_allowed=True`. With ~22y visible and auto_fold_count, expect ~35+ rolling folds.

**MC.** `CROSS_ASSET_MOMENTUM` class default: `block_size_bars=63, n_paths=200, bootstrap_method="shared_block", max_dd_threshold_pct=0.35, max_dd_pass_prob=0.10`. **L17 relative MC** required (long-only equity-class) — benchmark = 60/40 SPY/IEF buy-and-hold (the "do nothing" alternative for a cross-asset momentum strategy).

**Pre-registered cells (V3.1, frozen).**

| Cell | `vol_estimator_halflife` | `ann_vol_target` | Notes |
|---|---:|---:|---|
| **C1_canonical** | **20** | **0.05** | L52 sweep plateau centre (IS Sharpe 0.824). Candidate new live canonical. |
| P_hl10_vt05 | 10 | 0.05 | Plateau neighbour. IS Sharpe 0.817. |
| P_hl40_vt05 | 40 | 0.05 | Plateau neighbour (J4-halflife at low vt). IS Sharpe 0.818. |
| P_hl60_vt05 | 60 | 0.05 | Plateau neighbour. IS Sharpe 0.817. |
| P_hl20_vt075 | 20 | 0.075 | Plateau neighbour on the vol_target axis. IS Sharpe 0.815. |
| C2_constrained_best | 20 | 0.10 | Best cell IF allocator constraint `vol_target=0.10` is binding. IS Sharpe 0.768. Migration option B. |
| **C3_J4_live_baseline** | 40 | 0.10 | Current LIVE config. IS Sharpe 0.750. Parity baseline. EXCLUDED from promotion. |
| C4_gross_no_costs | 20 | 0.05 | Canonical with `apply_costs=False`. Gross reference. EXCLUDED from promotion. |

8 cells total. C3 / C4 EXCLUDED from §3 selection rule.

**Falsification hypotheses (pre-committed, V3.1).**

- **H1 (plateau holds OOS, spread ≤ 30%).** "The vol_target=0.05 plateau (C1 + 3 hl-neighbours + 1 vt-neighbour) has stitched-OOS Sharpe spread ≤ 30%." Falsifiable.
- **H2 (canonical OOS Sharpe ≥ +0.50 net).** "C1 stitched OOS Sharpe ≥ +0.50." Falsifiable. Calibrated: J4 live earned +0.78 IS; expecting +0.50 OOS is a modest 65% retention.
- **H3 (new canonical beats live on stitched OOS).** "C1 stitched OOS Sharpe > C3 stitched OOS Sharpe." Falsifiable. **Key migration-decision hypothesis.**
- **H4 (CI_lo > 0).** "C1 bootstrap 95% lower bound > 0." Falsifiable. V3.6 deployment gate.
- **H5 (rel-MC vs 60/40 SPY/IEF).** "C1 median DD reduction vs 60/40 benchmark ≤ 0.80 AND p_strategy_better ≥ 0.50." Falsifiable per L17.
- **H6 (noise axis at least mid).** "C1 noise axis is `mid` or `best`." The J4 redesign specifically targeted noise robustness via the EWMA estimator. Falsifiable — if the lower vol_target with halflife=20 degrades noise robustness, the upgrade is not deployment-eligible.

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds:** `CROSS_ASSET_MOMENTUM` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (DEPLOY, or CONDITIONAL_WATCHPOINT with noise=best), EXCLUDING C3/C4, pick the cell with the highest CI_lo.

**Pre-flight gate (L27 + L52 + L53).**

1. **L52 plateau pre-flight.** Stitched OOS spread across (C1, P_hl10_vt05, P_hl40_vt05, P_hl60_vt05, P_hl20_vt075) must be ≤ 30%. If > 30%, ABORT — IS plateau did not hold OOS.
2. **L53 early gate.** Skip Pass 2 if no cell can plausibly clear CI_lo > 0.
3. **Pass 2 + 5-axis decision matrix.** Only run on cells that pass L52 + L53.

**Migration rule (Decision 3 of findings).** If C1 (plateau centre) is PROMOTED:
- Migrate live config to `(halflife=20, vol_target=0.05)`.
- Allocator note: capital deployed at vol_target=0.05 is 50% of live; rotate freed capital to bond_gold CONDITIONAL allocation + cash buffer.
- 1-month shadow comparison before final cutover.

If only C2 (constrained best at vol_target=0.10) is PROMOTED:
- Migrate live config to `(halflife=20, vol_target=0.10)` (smaller change; preserves vol_target=0.10 mandate).
- No allocator rebalance needed.

If C1 fails H3 OR H5 OR H6:
- Live config remains J4 canonical. Sweep finding is informational only; no migration.

**L46 tighter-constraint rule.** If a cell passes the 5-axis matrix but CI_lo ≤ 0, do NOT promote.

**L55 sanctuary caveat.** If sanctuary `lucky_flag=True`, cite stitched OOS as deployment-relevant.

## §4. Result log

To be appended AFTER the audit harness runs.

## §5. Failure modes to watch

- **L04 / A1 (causality).** `gem_strategy.gem_returns` already enforces; `gem_assert_causal` smoke test in audit harness.
- **L17 (relative MC).** Benchmark = 60/40 SPY/IEF B&H (NOT just B&H SPY — GEM rotates across 3 assets so the relevant comparator is a static cross-asset allocation, not a single index).
- **L18 (shift discipline).** `gem_returns` does this; sweep + audit must agree.
- **L27 (plateau).** H1 tests this on OOS.
- **L46 (sample-size CI bottleneck).** With ~22y visible + 35+ folds, sample size is ample.
- **L55 (sanctuary regime-favourable).** 2025-04 → 2026-04 sanctuary is post-2024 bull continuation; expect possible lucky-flag.
- **L57 reservation (max_leverage cap asymmetry).** If J5 confirms the (20, 0.05) Sharpe holds OOS, formalise L57 in the catalogue.

## §6. What "complete" looks like

- §4 fully appended with all 8 cells' matrix.
- Verdict outcomes:
  - **PROMOTED — migrate to (20, 0.05)**: live config change scheduled at allocator window with 50% capital reduction notice.
  - **PROMOTED — migrate to (20, 0.10)**: live config change scheduled, no allocator rebalance.
  - **NOT PROMOTED**: J4 canonical remains live; L52 sweep finding documented as informational only.
- Update `.tmp/dashboard/audit_results.json` with the J5 entry.
- Update `directives/V1-era Re-audit Sweep Roster 2026-05-16.md` with the GEM hybrid result.
- Catalogue update (L57 if applicable).
