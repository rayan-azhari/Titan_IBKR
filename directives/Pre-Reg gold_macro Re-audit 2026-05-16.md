# Pre-Registration — gold_macro V1-era Re-audit (V3.6 + L52 hybrid)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** V1-era live strategy re-audit. Wave B full audit (first P2 strategy to clear the Wave B triage gate). See `directives/V1-era Re-audit Sweep Roster 2026-05-16.md` + `.tmp/reports/wave_b_triage/findings.md`.
**Strategy class:** `DAILY_TREND` (with cross-asset composite gate — closer to `CROSS_ASSET_MOMENTUM` for MC purposes since the signal uses non-target series).
**Predecessors:**

- V1-era `titan/strategies/gold_macro/` deployment (config: `config/gold_macro_gld.toml`). V1 audit (now deleted) claimed OOS Sharpe +0.603, OOS/IS ratio 2.06.
- Wave B triage 2026-05-16: bare-SMA(200) signal-layer on 21y GLD returns +0.69 Sharpe (CI_lo +0.26) — strongest Wave B candidate. **Caveat**: triage tested only component-3 (the SMA gate), not the full 3-component composite. The full audit below tests the composite.
- L52 sweep (`research/exploration/sweep_gold_macro.py`, 2026-05-16): IS plateau on 14y data found at `(slow_ma=100, real_rate_window=60)` with IS Sharpe +0.424, hood mean +0.421, spread 28.8%. **Live canonical `(slow_ma=200, real_rate_window=20)` IS Sharpe +0.133 — 275% below the plateau-row best.**
- L21 causality smoke on the pure-research implementation: **PASS** (max past-return diff = 0.00 when 20 future bars across all 4 series are multiplied by 100).

**Status:** §1–§3 frozen at this commit. §4 result log appended after the audit harness runs.

> **V3.1 pre-registration.** §1–§3 are frozen BEFORE the audit examines the 24-month sanctuary window. **Hypothesis being tested: the V3.6-correct gold_macro canonical `(slow_ma=100, real_rate_window=60)`, chosen from a Pardo-style sweep on the 14y IS, achieves CI_lo > 0 on the held-out sanctuary divergence test AND survives the 5-axis decision matrix.**
>
> **L52 canonical-selection note.** The canonical `(100, 60)` is the plateau center with the strongest neighbourhood mean (0.421 across 3 neighbours, spread 28.8%). The single-best cell `(150, 60)` has IS 0.497 (+17% over canonical) but borderline plateau structure (only 2 neighbours in spread bound). Picking the lower-IS-but-better-supported (100, 60) is the L43 anti-knife-edge discipline applied as the V3.6 default.
>
> **Why the live config is so far off the plateau.** The live `(200, 20)` was tuned in V1 against a much shorter dataset (composite signal cannot use data before DXY's start in 2010). The live's longer SMA + shorter real-rate window asks for very stale momentum confirmation paired with reactive rate signal — they fight each other. The plateau favours faster momentum gate + slower rate signal, which is internally coherent.

---

## §1. Motivation & mechanism

**Why re-audit gold_macro now.** Three converging signals:

1. The strategy is **currently in `titan/strategies/gold_macro/`** with a config whose Sharpe claim of +0.603 comes from a V1 audit whose harness has documented failure modes (filtered `rets != 0`, no shift discipline, no L21-style causality smoke). Per V2.0 README, all such strategies need a V3.6 re-audit before their Sharpe claims can be trusted.
2. **Wave B triage (2026-05-16) found gold_macro POSSIBLY VIABLE** (bare-SMA signal layer +0.69 Sharpe with CI_lo +0.26 on 21y GLD-D). It cleared the L58 signal-layer gate, justifying the full audit.
3. **L52 sweep on 14y composite-signal IS data found the live canonical sits significantly OFF the plateau.** The plateau at `(slow_ma=100, real_rate_window=60)` produces +0.424 IS Sharpe vs the live `(200, 20)`'s +0.133 (+219% gap). This mirrors the GEM J4→J5 pattern: a 1D-tuned V1 canonical can miss the 2D-plateau structure.

**Mechanism (per bar t, 3-component composite signal → GLD long-only position).**

1. **Component 1 — Real-rate proxy.**

   ```
   rr_signal(t) = -(log(TIP[t]/TLT[t]) - log(TIP[t-W_rr]/TLT[t-W_rr]))
   ```

   Inverted log-ratio change: positive when TIP underperforms TLT (real rates falling = gold bullish). `W_rr` is the swept `real_rate_window`.

2. **Component 2 — Dollar weakness.**

   ```
   d_signal(t) = -(log(DXY[t]) - log(DXY[t-W_d]))
   ```

   Inverted DXY log-return. `W_d = 20` is frozen at the live value.

3. **Causal expanding z-score normalisation.**

   ```
   rr_z(t) = (rr_signal(t) - expanding_mean[0..t](rr_signal)) / expanding_std[0..t](rr_signal)
   d_z(t)  = (d_signal(t)  - expanding_mean[0..t](d_signal))  / expanding_std[0..t](d_signal)
   composite_z(t) = (rr_z(t) + d_z(t)) / 2
   ```

   Z-score uses past-only data through t. Warm-up of 60 observations before first non-zero output.

4. **Component 3 — Momentum gate.**

   ```
   momentum(t) = 1 if GLD[t] > SMA(GLD, slow_ma)[t] else 0
   ```

   `slow_ma` is the swept axis.

5. **Entry / exit.**

   ```
   signal(t) = 1 if (composite_z(t) > 0) AND (momentum(t) == 1) else 0
   ```

6. **Vol-target sizing on GLD.**

   ```
   var(t)              = EWMA(gld_ret^2, span=20)[t]
   realised_vol_ann(t) = sqrt(var(t) * 252)
   scale(t)            = min(0.10 / realised_vol_ann(t), 1.5)
   position(t)         = signal(t) * scale(t)
   ```

7. **Per-bar return.**

   ```
   gross(t+1) = position(t) * log(GLD[t+1] / GLD[t])
   cost(t+1)  = |position(t+1) - position(t)| * 1.5 / 1e4
   net(t)     = gross(t) - cost(t)
   ```

   Implemented as `position.shift(1) * gld_log_return`.

**Causality (L04 / A1 / L21).** All component signals at t use data known by EOD t. Z-score is causal expanding (past-only). Momentum SMA uses `GLD[t-slow_ma+1 .. t]`. Position EFFECTIVE for the t→t+1 return is the t-1 close decision (via `position.shift(1)`). L21 smoke test in the audit harness corrupts the last 20 bars across all 4 series and asserts past returns are bit-exact identical (already PASSED in the sweep at line `[gm-sweep] L21 causality smoke PASS (max past-return diff = 0.00e+00)`).

## §2. Universe + audit configurations

**Universe.** GLD (target) + TIP, TLT (real-rate proxy) + DXY (dollar). All yfinance daily adjusted closes (`data/{GLD,TIP,TLT,DXY}_D.parquet`).

**Date range.** Common intersection: **2010-01-04 → 2026-04-02 (4,085 bars)**. DXY's 2010 start is the binding constraint.

**Sanctuary.** **Last 24 months** held out: 2024-04-02 → 2026-04-02 (503 bars). Applied via `slice_sanctuary()` BEFORE WFO fold construction. **The sweep that informed §2's cell selection was run on the visible (IS) slice; the sanctuary was NEVER seen.**

**Visible window for WFO.** 3,583 bars (2010-01-04 → 2024-04-02), ~14.2 years.

**WFO.** `CROSS_ASSET_MOMENTUM` class default (the strategy has cross-asset signal so MC must bootstrap with `extra_series`): `is_min_years=2.0, oos_years=0.5, fold_count=auto, is_mode="rolling", stride_overlap_allowed=True`. With ~14y visible, expect ~20+ rolling folds. No per-fold tuning (L13/L14).

**MC.** `CROSS_ASSET_MOMENTUM` defaults: `block_size_bars=63, n_paths=200, bootstrap_method="shared_block", max_dd_threshold_pct=0.35, max_dd_pass_prob=0.10`. `extra_series={"TIP": ..., "TLT": ..., "DXY": ...}` to preserve the GLD↔TIP/TLT/DXY correlation under the shared-block bootstrap.

**Pre-registered cells (V3.1, frozen).** Sweep informed the canonical + 4 plateau cells + 2 references.

| Cell | `slow_ma` | `real_rate_window` | Notes |
|---|---:|---:|---|
| **C1_canonical** | **100** | **60** | Plateau center (sweep #1, hood mean 0.421 across 3 nb, spread 28.8%). The new V3.6 canonical. |
| P_north (faster MA) | 50 | 60 | Plateau neighbour, IS Sharpe 0.389. |
| P_south (slower MA) | 150 | 60 | Plateau neighbour AND grid maximum, IS Sharpe 0.497. |
| P_east (faster RR) | 100 | 40 | Plateau neighbour, IS Sharpe 0.375. |
| P_corner (faster both) | 50 | 40 | Adjacent top-left high-Sharpe zone, IS Sharpe 0.445. |
| C2_live_canonical | 200 | 20 | V1-era live config (IS Sharpe 0.133). Parity baseline + V1-vs-V3.6 comparator. |
| C3_gross_no_costs | 100 | 60 | Canonical with `apply_costs=False`. Gross-economics reference. |
| C4_pure_sma | 200 | 20 | Bare-SMA(200) ONLY (composite_z gate disabled) — the Wave B triage cell. Tests whether the composite adds value over the bare gate. |

8 cells total. Cells C2 / C3 / C4 are baselines / references and are EXCLUDED from the §3 selection rule.

**Falsification hypotheses (pre-committed, V3.1).**

- **H1 (plateau holds OOS).** Relative spread across (C1, P_north, P_south, P_east, P_corner) on the SANCTUARY remains ≤ 50%. **Falsifiable** — if spread > 50%, the IS plateau was IS-specific and the canonical choice is fragile.
- **H2 (canonical sanctuary Sharpe ≥ +0.30).** **Falsifiable** — translates the IS plateau into a sanctuary-economics threshold. (Lower than bond_gold's H2 because composite-gated long-only equity typically has lower realised Sharpe than the gross-momentum bond_gold setup, and 2024-2026 has been a strong-gold regime.)
- **H3 (new canonical beats live).** "C1 (100, 60) achieves higher sanctuary Sharpe than C2 (200, 20)." **Falsifiable** — if C2 ≥ C1, the L52-recommended canonical does not improve over the live config, and the V1 choice should stand.
- **H4 (CI_lo > 0 on stitched OOS).** "C1's bootstrap 95% lower bound across the full WFO stitched OOS is > 0." **Falsifiable** — if CI_lo ≤ 0, the strategy fails the V3.6 deployment gate regardless of decision-matrix verdict (L46 precedent: tighter constraint wins).
- **H5 (composite adds value over bare SMA).** "C1 (composite + SMA(100)) achieves higher CI_lo than C4 (bare SMA(200), no composite)." **Falsifiable** — if C4 ≥ C1, the cross-asset composite contributes nothing and the strategy should be simplified to a pure trend rule (or routed into the etf_trend family — but note GLD is the only L56-PASSING unleveraged single-leg ETF per Wave A.2; this branch would warrant separate consideration).
- **H6 (cost drag is not the bottleneck).** "C3 gross Sharpe − C1 net Sharpe ≤ 0.25." **Falsifiable** — if the gap is > 0.25, costs are the binding constraint and the strategy is uneconomic at retail-cost levels.

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds:** `CROSS_ASSET_MOMENTUM` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (DEPLOY, or CONDITIONAL_WATCHPOINT with noise=best), EXCLUDING C2/C3/C4 (live-baseline + gross reference + bare-SMA control), pick the cell with the highest CI_lo.

**Pre-flight gate (L27 + L52).**

1. **L52 plateau pre-flight (run FIRST).** Compute Pass-1 stitched OOS Sharpe for C1 + 4 plateau cells. Relative spread `(max − min) / |mean|` must be ≤ 50%. If > 50%, the IS plateau did NOT hold OOS — abort Pass 2 and verdict = RETIRED (L52 H1 plateau-fail).
2. **L53 early gate** (run SECOND). For each cell, compute `pass1_can_clear_ci_gate`. If NO cell can plausibly clear CI_lo > 0, skip Pass 2 and verdict = RETIRED.
3. **Pass 2 + 5-axis decision matrix.** Only run on cells that pass L52 + L53 gates.

**Tighter-constraint rule (L46).** If a cell passes the 5-axis matrix but has CI_lo ≤ 0, it does NOT promote. Bootstrap CI is the binding deployment gate.

**L17 relative-MC application.** GLD is long-only equity-class; under absolute-MaxDD threshold MC, even buy-and-hold GLD can fail the 35% gate (gold drawdowns are deep — 2012-2015 saw ~45%). The audit applies `run_relative_block_mc()` with `benchmark_fn=buy_and_hold_GLD` as the L17-correct test. Pass criterion: `median_dd_reduction ≤ 0.80` AND `p_strategy_better ≥ 0.50`. This is the same gate that distinguished GEM J5 from J4 (and which `etf_trend_spy` failed in Wave A.2).

**Sanctuary divergence test.** Standard `sanctuary_divergence_test()` 4th axis. Canonical's sanctuary Sharpe must be within the percentile range derived from the OOS-fold Sharpe distribution.

**Noise robustness.** Standard `run_noise_robustness()` 5th axis with `NoiseConfig()` defaults.

## §4. Result log

To be appended AFTER the audit harness runs. Sections expected:

- §4.1 — Baseline reproduction (C2 vs V1 +0.603 claim; document the bid-ask)
- §4.2 — Plateau pre-flight (L27 + L52 on stitched OOS)
- §4.3 — L53 early-gate report
- §4.4 — Per-cell 5-axis decision matrix
- §4.5 — L17 relative MC (GLD buy-and-hold benchmark)
- §4.6 — H1 / H2 / H3 / H4 / H5 / H6 verdicts
- §4.7 — Promotion verdict + sample-size caveat (L46 pattern)
- §4.8 — Live-config impact + backlog
- §4.9 — New lessons (if any)

## §5. Failure modes to watch

- **L04 / A1 (causality).** Verified at the sweep stage — composite z-score is causal expanding, momentum gate uses past-only SMA, position shift-1 before earning return. `assert_causal` in audit harness re-verifies.
- **L13 / L14 (IS/OOS separation).** The expanding z-score uses ALL past data, which includes the IS portion when computing z-score on sanctuary. This is the **V3.6-correct causal-expanding pattern** — the alternative (re-fit z-stats per fold) introduces L14 leakage if done naively. Sanctuary still tests an unseen 24-month period; only the z-stats grow naturally with time.
- **L17 (relative MC for long-only equity-like targets).** GLD is long-only equity-class. Applied as a primary 5-axis input per §3.
- **L18 (shift discipline).** Position must shift by 1 bar; parity test in `tests/` against sweep implementation.
- **L21 (multi-instrument causality).** Already PASSED at sweep stage — corrupted future closes across all 4 instruments left past returns bit-exact unchanged.
- **L27 (plateau).** Tested as H1 on OOS.
- **L43 (knife-edge plateau).** Sweep showed reasonable hood mean (0.421) at the canonical with 3 supporting neighbours; H1 tests on OOS.
- **L46 (sample-size CI bottleneck).** May bind — only 14y of usable data (DXY-constrained) is shorter than bond_gold's 21y. Effective WFO folds may be limited.
- **L49 (regime-artifact).** Gold has had several distinct regimes over the 2010-2024 IS: 2011-2015 secular bear, 2019-2020 COVID surge, 2022-2024 inflation/rate-cut anticipation. Diagnostic: per-regime breakdown in §4.7.
- **L52 (hybrid workflow self-test).** Second Wave-B applied case. If H1 fails (IS plateau does not hold OOS), update L52 catalogue note.
- **L56 (long-only-MA-crossover rel-MC fail).** Five etf_trend variants RETIRED on this pattern. The composite gate distinguishes gold_macro from a pure MA-crossover — H5 explicitly tests whether the composite adds value over the bare gate. If H5 is REJECTED, gold_macro reduces to an L56-vulnerable strategy.
- **L58 (signal-layer-first audit pattern).** Wave B triage established the signal layer is positive (bare-SMA +0.69 Sharpe). The full audit tests whether the full composite preserves or degrades that edge.

## §6. What "complete" looks like

- §4 fully appended with all 8 cells' 5-axis matrix + L17 rel-MC for plateau cells.
- One of four outcomes:
  - **DEPLOY** (C1 passes all gates, CI_lo > 0): live config migrates to `(slow_ma=100, real_rate_window=60)`. Migration requires updated `config/gold_macro_gld.toml` + parity test + 6-month shadow comparison.
  - **CONDITIONAL_WATCHPOINT** (C1 passes 5-axis but CI_lo ≤ 0, or noise axis < best): shadow-deploy canonical alongside the live one for 6-12 months; re-audit when CI_lo crosses zero or noise improves.
  - **L17 FAIL → RETIRE** (canonical fails L17 rel-MC despite other axes passing): RETIRE consistent with L56 precedent. Live `gold_macro` de-allocated at next allocator window.
  - **RETIRED** (plateau-fail OR sanctuary divergence OR sanctuary Sharpe): gold_macro V1-era config remains in code but flag as "V1-claim unconfirmed"; de-allocation depending on allocator priorities.

In all four cases: update [V1-era Re-audit Sweep Roster 2026-05-16.md](V1-era%20Re-audit%20Sweep%20Roster%202026-05-16.md) to mark gold_macro's Wave B full-audit status as COMPLETE.
