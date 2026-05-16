# Pre-Registration — etf_trend SPY V1-era Re-audit (V3.6 + L52 hybrid)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** V1-era live strategy re-audit. Wave A.2 of the V1-era re-audit roster.
**Strategy class:** `DAILY_TREND`
**Predecessors:**

- V1-era `titan/strategies/etf_trend/` deployment with `config/etf_trend_spy.toml`. V1 audit (now deleted) reportedly claimed Sharpe ~+0.70 net.
- L52 — V3.6 hybrid Pardo + V3.6 workflow. Wave A.1 (bond_gold, 2026-05-16) successfully promoted a new canonical under CONDITIONAL_WATCHPOINT.
- L55 — sanctuary in regime-favourable windows is upper-bound (added 2026-05-16 after bond_gold).
- Sweep findings: `.tmp/reports/sweep_etf_trend_spy/findings.md` (2026-05-16).

**Status:** §1–§3 frozen at this commit. §4 result log appended after the audit harness runs.

> **V3.1 pre-registration.** §1–§3 are frozen BEFORE the audit examines the 24-month sanctuary window. **Hypothesis being tested: a V3.6-correct etf_trend_spy canonical, chosen from a Pardo-style sweep on the 21y IS, achieves CI_lo > 0 on the stitched OOS AND demonstrates MaxDD reduction vs buy-and-hold SPY under relative MC.**
>
> **L52 canonical-selection note.** Selected `(slow_ma=300, exit_confirm_days=5)` — sweep plateau detector rank #2 with 3 valid neighbours (vs rank #1 at `(300, 10)` with only 2 edge neighbours). Trades a marginally lower IS Sharpe (0.713 vs 0.714) for stronger neighbourhood stability evidence (spread 8.6% vs 1.4% but with 3 vs 2 neighbours). Choice pre-committed; falsification gate H1 below tests whether the chosen plateau actually holds OOS.

---

## §1. Motivation & mechanism

**Why re-audit etf_trend_spy now.** Three reasons:

1. **The strategy is live** in `titan/strategies/etf_trend/` with config `(slow_ma=150, exit_confirm_days=1)`. Its Sharpe claim from the deleted V1 audit (~+0.70) is approximately reproducible under V3.6 (+0.62 IS) but the canonical sits OFF the plateau identified by the sweep.
2. **L52 sweep (2026-05-16) on 21y IS data found the plateau at slow_ma=300.** Live canonical IS Sharpe 0.619 vs plateau row best 0.723 (+17% gap). The slow_ma=300 row is a clean plateau (4% spread across all 5 exit_confirm_days cells); the slow_ma=150 row is less stable.
3. **24 months of sanctuary** (2024-05-13 → 2026-05-12) is sufficient to test whether the IS plateau holds OOS. The sanctuary period is likely regime-favourable for long-SPY (bull market continuation) → L55 caveat will apply.

**Mechanism (per daily bar t, single-instrument signal on SPY).**

1. **Trend regime.** Compute `sma(t) = mean(spy.close[t-slow_ma+1 .. t])`. Regime is "long-eligible" when `spy.close[t] > sma(t)`.

2. **Exit confirmation.** Once long, stay long while `spy.close > sma`. When `spy.close < sma`, increment a counter. Exit when counter reaches `exit_confirm_days` consecutive bars. Counter resets on any `close > sma` day.

3. **Vol-target sizing.**

   ```
   var(t)   = EWMA(spy_log_ret^2, span=20)[t]
   real_vol = sqrt(var(t) * 252)
   scale(t) = min(vol_target / real_vol, max_leverage)
   position(t) = signal(t) * scale(t)
   ```

   `vol_target=0.20`, `max_leverage=2.0` (frozen at live config).

4. **Per-bar return.** Position effective at close[t] earns return from t → t+1:

   ```
   gross(t) = position(t-1) * spy_log_ret(t)
   cost(t)  = |position(t) - position(t-1)| * cost_bps / 1e4
   net(t)   = gross(t) - cost(t)
   ```

   `cost_bps_per_turnover=1.0` (matches B2 / B4 / bond_gold cost model for SPY-class instruments).

**Causality (L04 / A1).** SMA at close[t] uses [t-slow_ma+1, t] inclusive — known by EOD. Position at t earns t→t+1 return via `.shift(1)`. Asserted by `etf_trend_spy_assert_causal()` smoke test.

**Note on simplification.** The live strategy has additional knobs: `decel_signals`, `atr_stop_mult`, `fast_reentry_ma`. The current live config sets `decel_signals=[]` (no decel composite) and `fast_reentry_ma=None` (not asymmetric), so the audit's simplified mechanism above is a TRUE subset of the live behaviour. The `atr_stop_mult=5.0` hard stop is a risk-management overlay, not a signal-edge feature; the audit deliberately excludes it because the deployment-relevant question is "does the SIGNAL produce CI_lo > 0?" — the stop is invariant across the parameter grid.

## §2. Universe + audit configurations

**Universe.** SPY daily closes from `data/SPY_D.parquet`.

**Date range.** 2003-01-02 → 2026-05-12, 5,877 bars.

**Sanctuary.** Last 24 months held out: 2024-05-13 → 2026-05-12 (501 bars). Applied via `slice_sanctuary()` BEFORE WFO. Same sweep boundary as the L52 informant.

**Visible window for WFO.** 5,376 bars (2003-01-02 → 2024-05-10), ~21.3 years.

**WFO.** `DAILY_TREND` class default: `is_min_years=3.0, oos_years=1.0, fold_count=5, is_mode="expanding", stride_overlap_allowed=False`. With ~21y visible, `auto_fold_count` should produce ~17-18 folds. No per-fold tuning (L13/L14).

**MC.** `DAILY_TREND` class default: `block_size_bars=21, n_paths=200, bootstrap_method="block", max_dd_threshold_pct=0.35, max_dd_pass_prob=0.10`.

**Relative MC (L17 — REQUIRED for long-only equity).** Use `run_relative_block_mc()` with `benchmark_fn = buy-and-hold SPY (vol-targeted)`. Absolute P(MaxDD>35%) is expected to fail because B&H SPY in any bootstrap drawing from 2008-09 will easily exceed 35%. The deployment-relevant gate is `median_dd_reduction ≤ 0.80` AND `p_strategy_better ≥ 0.50` per L17.

**Pre-registered cells (V3.1, frozen).**

| Cell | `slow_ma` | `exit_confirm_days` | Notes |
|---|---:|---:|---|
| **C1_canonical** | **300** | **5** | Plateau detector rank #2 (3 valid neighbours, hood mean 0.692). The new V3.6 canonical. |
| P_ec1 | 300 | 1 | IS Sharpe 0.702. |
| P_ec2 | 300 | 2 | IS Sharpe 0.706. |
| P_ec3 | 300 | 3 | IS Sharpe 0.684 (low edge of the row). |
| P_ec10 | 300 | 10 | IS Sharpe 0.714 (high edge of the row). |
| C2_live_canonical | 150 | 1 | V1-era live config (IS Sharpe 0.619). Parity baseline. |
| C3_buy_and_hold | — | — | Buy-and-hold SPY vol-targeted (no signal). Economic baseline. |
| C4_gross_no_costs | 300 | 5 | Canonical with `apply_costs=False`. Gross reference. |

8 cells total. C2 / C3 / C4 are baselines and EXCLUDED from the §3 selection rule.

**Falsification hypotheses (pre-committed, V3.1).**

- **H1 (plateau holds OOS, spread ≤ 30%).** "The slow_ma=300 plateau persists OOS: spread across (C1, P_ec1, P_ec2, P_ec3, P_ec10) stitched OOS Sharpes ≤ 30% (matches L27 strict gate)." **Falsifiable** — if spread > 30%, the IS plateau was IS-specific.
- **H2 (canonical OOS Sharpe ≥ +0.30 net).** "C1's stitched OOS Sharpe is ≥ +0.30." Falsifiable — well above buy-and-hold-style noise floor.
- **H3 (new canonical beats live on stitched OOS).** "C1 stitched OOS Sharpe > C2 stitched OOS Sharpe." Falsifiable — tests whether the L52-recommended canonical actually improves over the V1 choice on unseen data.
- **H4 (CI_lo > 0).** "C1 bootstrap 95% lower bound across stitched OOS > 0." Falsifiable — V3.6 deployment gate (L46).
- **H5 (MaxDD reduction vs B&H).** "C1 median MaxDD across MC paths is ≤ 80% of buy-and-hold SPY's median MaxDD across the same paths." Falsifiable per L17 relative-MC test.
- **H6 (cost drag ≤ 0.10).** "C4 gross Sharpe − C1 net Sharpe ≤ 0.10." Falsifiable — etf_trend_spy is a slow strategy; cost drag should be small.

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds:** `DAILY_TREND` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (DEPLOY, or CONDITIONAL_WATCHPOINT with noise=best), EXCLUDING C2/C3/C4, pick the cell with the highest CI_lo.

**Pre-flight gate (L27 + L52 + L53).**

1. **L52 plateau pre-flight.** Stitched OOS spread across (C1, P_ec1, P_ec2, P_ec3, P_ec10) must be ≤ 30% (matches L27 strict gate; tighter than bond_gold's 50% since the SPY data has 4x the bars and the IS plateau was much flatter at 4%). If > 30%, ABORT and document the L52 first-mistake on this audit.

2. **L53 early gate.** For each cell, compute `pass1_can_clear_ci_gate(headline_sharpe, n_oos_bars, block_size=21)`. If NO cell can plausibly clear CI_lo > 0, skip Pass 2 and verdict = RETIRED.

3. **Pass 2 + 5-axis decision matrix.** Only run on cells that pass L52 + L53 gates.

**Tighter-constraint rule (L46).** If a cell passes the 5-axis matrix but has CI_lo ≤ 0, it does NOT promote.

**L17 relative MC.** Use `run_relative_block_mc()` for the MC axis. Strategy passes the MC axis iff `median_dd_reduction ≤ 0.80` AND `p_strategy_better ≥ 0.50`. The absolute P(MaxDD>35%) field is reported but not gating (long-only equity will fail it under bootstrap).

**L55 sanctuary caveat.** If sanctuary_divergence_test returns `lucky_flag=True`, the result log MUST cite stitched OOS Sharpe (NOT sanctuary Sharpe) as the deployment-relevant number. Sanctuary 2024-26 is expected to be regime-favourable (continuing bull market), so this caveat is expected to bind.

## §4. Result log

To be appended AFTER the audit harness runs. Sections expected:

- §4.1 — Pass 1 stitched OOS Sharpe + CI per cell
- §4.2 — L52 plateau pre-flight + L53 early gate
- §4.3 — Per-cell 5-axis matrix (including relative MC)
- §4.4 — H1 / H2 / H3 / H4 / H5 / H6 verdicts
- §4.5 — Promotion verdict + L55 caveat application
- §4.6 — Live-config impact + backlog (Wave A.3 next)

## §5. Failure modes to watch

- **L04 / A1 (causality).** SMA + position must shift by 1. `assert_causal` smoke in harness.
- **L17 (relative MC for long-only equity).** REQUIRED. Absolute MaxDD gate WILL fail.
- **L18 (shift discipline).** Sweep parity test in `tests/`.
- **L25 (class default WFO).** No override needed — DAILY_TREND defaults applied directly with auto_fold_count.
- **L27 (plateau).** Tested as H1 with strict 30% gate.
- **L43 (knife-edge plateau).** Sweep ruled this out on IS; H1 tests on OOS.
- **L46 (sample-size CI bottleneck).** With 5,376 IS bars + 17+ folds, sample size should be ample.
- **L49 (regime-artifact).** 2003-2024 spans multiple cycles (2003-07 bull, 2008-09 crash, 2009-20 bull, 2020 crash, 2020-22 bull, 2022 bear, 2023-24 bull). Diagnostic: per-decade rolling Sharpe in §4.6 to confirm signal isn't regime-localised.
- **L55 (sanctuary regime-favourable).** Expected to bind for 2024-26 long-SPY. Apply caveat in result log.

## §6. What "complete" looks like

- §4 fully appended with all 8 cells' matrix.
- One of three outcomes:
  - **DEPLOY** (C1 passes all gates including relative MC vs B&H): live config can be migrated to `(slow_ma=300, exit_confirm_days=5)`. Migration requires a separate `config/etf_trend_spy.toml` change + parity test against the new audit cell. Live deployment delayed until migration test passes.
  - **CONDITIONAL_WATCHPOINT** (C1 passes decision matrix but blocked by CI_lo ≤ 0 or relative-MC marginal): 6-month shadow comparison alongside live; re-audit when sanctuary extends.
  - **RETIRED** (C1 fails plateau / sanctuary divergence / relative-MC test): etf_trend_spy V1 config remains in production; flag as "V1-claim unconfirmed; further data required."

In all three cases: update `directives/V1-era Re-audit Sweep Roster 2026-05-16.md` Wave A.2 status to COMPLETE with verdict.
