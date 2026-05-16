# Pre-Registration — bond_gold V1-era Re-audit (V3.6 + L52 hybrid)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** V1-era live strategy re-audit. Wave A.1 of the V1-era re-audit roster (see `directives/V1-era Re-audit Sweep Roster 2026-05-16.md`).
**Strategy class:** `CROSS_ASSET_MOMENTUM`
**Predecessors:**

- V1-era `titan/strategies/bond_gold/` deployment (config: `config/bond_gold.toml`). V1 audit (now deleted) claimed Sharpe +1.17, 68% positive folds, 37-fold WFO 2007-2026.
- L52 — V3.6 hybrid Pardo + V3.6 workflow (this is the first APPLIED case after the B4b retrospective).
- Sweep findings: `.tmp/reports/sweep_bond_gold/findings.md` (2026-05-16).

**Status:** §1–§3 frozen at this commit. §4 result log appended after the audit harness runs.

> **V3.1 pre-registration.** §1–§3 are frozen BEFORE the audit examines the 24-month sanctuary window. **Hypothesis being tested: the V3.6-correct bond_gold canonical, chosen from a Pardo-style sweep on the 19y IS, achieves CI_lo > 0 on the held-out sanctuary divergence test AND survives the 5-axis decision matrix.**
>
> **L52 canonical-selection note.** The canonical `(lookback=120, threshold=0.50)` was chosen as the *best individual cell* in the lookback=120 row — which is the most-stable plateau row in the sweep. It is NOT the cell with the strictest L27 plateau pass (that would be `(120, 0.00)` at spread 11.3%). The choice trades a slightly higher IS Sharpe (0.665 vs 0.560) for a slightly less robust neighbourhood. This trade-off is documented and pre-committed; the falsification gate H1 below tests whether the chosen canonical's plateau actually holds.

---

## §1. Motivation & mechanism

**Why re-audit bond_gold now.** Three converging signals:

1. The strategy is **currently in `titan/strategies/bond_gold/`** with a config (`lookback=60, threshold=0.50`) whose Sharpe claim of +1.17 comes from a V1 audit whose harness has the documented failure modes (filtered `rets != 0`, no shift discipline, ambiguous cost model). Per the V2.0 README, all such strategies need a V3.6 re-audit before their Sharpe claims can be trusted.
2. **L52 sweep (2026-05-16) on 19y IS data found the live canonical sits OFF the plateau.** Plateau centre is at `lookback=120`. Live canonical IS Sharpe 0.524 vs plateau-row best 0.665 (+27% gap).
3. **24 months of held-out sanctuary** (2024-04-02 → 2026-04-02) is sufficient to test whether the IS plateau holds out-of-sample. This is the deployment-relevant question — V1's Sharpe is now informational; the V3.6 audit is the deployment gate.

**Mechanism (per bar t, single-instrument signal → GLD position).**

1. **Bond momentum.** Compute the log-return of IEF (intermediate Treasury bond ETF) over `lookback` days:

   ```
   bond_mom(t) = log(IEF[t] / IEF[t - lookback])
   ```

   At canonical `lookback=120`, this is a ~6-month bond return — positive when rates fall (bond prices rise).

2. **Rolling z-score normalisation** (causal, IS-frozen rolling window of 504 days):

   ```
   zmean(t) = rolling_mean(bond_mom, 504)[t]
   zstd(t)  = rolling_std(bond_mom, 504, ddof=1)[t]
   z(t)     = (bond_mom(t) - zmean(t)) / zstd(t)
   ```

3. **Long entry.** Signal is binary: `sig_raw(t) = 1` if `z(t) > threshold`, else `0`. At canonical `threshold=0.50`, this requires the bond momentum to be ~0.5σ above its 2-year mean.

4. **Hold-day floor.** Once entered, the position stays open for at least `hold_days=20` (frozen at the live config value). After 20 days, exit when `z(t) <= threshold`.

5. **Vol-target sizing.** GLD position scale:

   ```
   var(t) = EWMA(gld_ret^2, span=20)[t]
   realised_vol_ann(t) = sqrt(var(t) * 252)
   scale(t) = min(target_vol / realised_vol_ann, max_leverage)
   position(t) = sig(t) * scale(t)
   ```

   `target_vol=0.10`, `max_leverage=1.5` (frozen at live values).

6. **Per-bar return.** Position effective at close[t] earns return from t -> t+1:

   ```
   gross(t+1) = position(t) * log(GLD[t+1] / GLD[t])
   cost(t+1)  = |position(t+1) - position(t)| * cost_bps_per_turnover / 1e4
   net(t)     = gross(t) - cost(t)
   ```

**Causality (L04 / A1).** All rolling statistics are past-only. Signal uses bond_mom through close[t] and z-score statistics through t-1 (since z-mean and z-std exclude bar t when computed as `rolling.mean()` and shifted appropriately by the EOD timing convention). Position at t earns t->t+1 return. Implemented as `position.shift(1) * gld_log_return`. Causality asserted via `assert_causal` smoke test in the audit harness.

## §2. Universe + audit configurations

**Universe.** IEF (signal) + GLD (target). Both yfinance daily adjusted closes (`data/IEF_D.parquet`, `data/GLD_D.parquet`).

**Date range.** Common intersection: 2004-11-18 → 2026-04-02 (5,376 bars).

**Sanctuary.** **Last 24 months** held out as the sanctuary divergence test window: 2024-04-02 → 2026-04-02 (503 bars). Sanctuary applied via `slice_sanctuary()` BEFORE WFO fold construction. **The sweep that informed §2's cell selection was run on the same IS slice; the sanctuary was NEVER seen.**

**Visible window for WFO.** 4,874 bars (2004-11-18 → 2024-04-02), ~19.4 years.

**WFO.** `CROSS_ASSET_MOMENTUM` class default: `is_min_years=2.0, oos_years=0.5, fold_count=8, is_mode="rolling", stride_overlap_allowed=True`. With ~19y visible and auto_fold_count, expect ~30+ rolling folds. No per-fold tuning (L13/L14).

**MC.** `CROSS_ASSET_MOMENTUM` class default: `block_size_bars=63, n_paths=200, bootstrap_method="shared_block", max_dd_threshold_pct=0.35, max_dd_pass_prob=0.10`. `extra_series={"IEF": ief_close}` to preserve IEF↔GLD correlation under the shared-block bootstrap.

**Pre-registered cells (V3.1, frozen).** Sweep informed the canonical + 4 plateau neighbours; the baseline + gross cells are V3.6 standards.

| Cell | `lookback` | `threshold` | Notes |
|---|---:|---:|---|
| **C1_canonical** | **120** | **0.50** | Plateau centre on IS (Sharpe 0.665). The new V3.6 canonical. |
| P_low_threshold | 120 | 0.00 | Most-stable plateau (sweep #1, IS spread 11.3%). |
| P_quarter | 120 | 0.25 | IS Sharpe 0.627. |
| P_high_threshold | 120 | 0.75 | IS Sharpe 0.606. |
| P_strict | 120 | 1.00 | IS Sharpe 0.511. |
| C2_live_canonical | 60 | 0.50 | V1-era live config (IS Sharpe 0.524). Parity baseline. |
| C3_no_threshold | 120 | 0.00 | Most-permissive plateau cell (duplicate of P_low_threshold — kept as separate cell to test selection-rule edge case). |
| C4_gross_no_costs | 120 | 0.50 | Canonical with `apply_costs=False`. Gross-economics reference. |

8 cells total. Cells C2 / C3 / C4 are baselines / references and are EXCLUDED from the §3 selection rule.

**Falsification hypotheses (pre-committed, V3.1).**

- **H1 (plateau holds out-of-sample).** "The lookback=120 plateau identified on IS persists in the sanctuary OOS — relative spread across (C1, P_low_threshold, P_quarter, P_high_threshold, P_strict) on the SANCTUARY remains ≤ 50% (looser than the L27 30% gate because sanctuary is shorter)." **Falsifiable** — if spread > 50%, the plateau was IS-specific and the canonical choice is fragile.
- **H2 (canonical Sharpe ≥ +0.30 net on sanctuary).** "C1's sanctuary Sharpe is at least +0.30 (modest positive)." **Falsifiable** — if Sharpe < +0.30, the IS plateau did not translate to OOS economics.
- **H3 (new canonical beats live).** "C1 (lookback=120) achieves higher sanctuary Sharpe than C2 (lookback=60)." **Falsifiable** — if C2 ≥ C1, the L52-recommended canonical does not improve over the live config, and the V1 choice should stand.
- **H4 (CI_lo > 0 on stitched OOS).** "C1's bootstrap 95% lower bound across the full WFO stitched OOS is > 0." **Falsifiable** — if CI_lo ≤ 0, the strategy fails the V3.6 deployment gate regardless of decision-matrix verdict (L46 precedent: tighter constraint wins).
- **H5 (cost drag is not the bottleneck).** "C4 gross Sharpe - C1 net Sharpe ≤ 0.30." **Falsifiable** — if the gap is > 0.30, costs are the binding constraint (similar to B2 §4.8 finding) and the strategy is uneconomic at retail-cost levels.

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds:** `CROSS_ASSET_MOMENTUM` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (DEPLOY, or CONDITIONAL_WATCHPOINT with noise=best), EXCLUDING C2/C3/C4 (live-baseline + gross reference), pick the cell with the highest CI_lo.

**Pre-flight gate (L27 + L52).**

1. **L52 plateau pre-flight (run FIRST).** Compute Pass-1 OOS Sharpe for C1 + 4 plateau neighbours on the WFO stitched OOS. Relative spread `(max - min) / |mean|` must be ≤ 50%. If > 50%, the IS plateau did NOT hold OOS — the canonical is L43 knife-edge in disguise. **Document the L52 first-mistake-on-applied-case and ABORT** before launching Pass 2.

2. **L53 early gate** (run SECOND, after plateau passes). For each cell, compute `pass1_can_clear_ci_gate(headline_sharpe, n_oos_bars, block_size=63)`. If NO cell can plausibly clear CI_lo > 0, skip Pass 2 and verdict = RETIRED.

3. **Pass 2 + 5-axis decision matrix.** Only run on cells that pass L52 + L53 gates.

**Tighter-constraint rule (L46).** If a cell passes the 5-axis matrix but has CI_lo ≤ 0, it does NOT promote. Bootstrap CI is the binding deployment gate. Per the precedent in B2 §4.7: tier=unconfirmed for negative CI_lo overrides decision-matrix verdict.

**Sanctuary divergence test.** Apply the standard `sanctuary_divergence_test()` 4th axis. The new canonical's sanctuary Sharpe must be within the percentile range derived from the OOS-fold Sharpe distribution. The 5th axis (noise robustness) uses the framework default.

## §4. Result log

To be appended AFTER the audit harness runs. Sections expected:

- §4.1 — Baseline reproduction (C2 vs V1 claim; document the bid-ask)
- §4.2 — Plateau pre-flight (L27 + L52 on stitched OOS)
- §4.3 — L53 early-gate report
- §4.4 — Per-cell 5-axis decision matrix
- §4.5 — H1 / H2 / H3 / H4 / H5 verdicts
- §4.6 — Promotion verdict + sample-size caveat (L46 pattern)
- §4.7 — Live-config impact + backlog
- §4.8 — New lessons (if any)

## §5. Failure modes to watch

- **L04 / A1 (causality).** Bond momentum must use IEF through t-1; z-score must be past-only; position must shift by 1 before earning return. `assert_causal` smoke test in harness.
- **L13 / L14 (IS/OOS separation).** Rolling z-score statistics inside the WFO IS window only? **No — z-score uses a fixed 504d trailing window across the full series, which is the V3.6-correct "causal rolling" pattern.** The window is NOT re-fit per WFO fold (that would be L14 leakage if done naively); it slides naturally with time.
- **L17 (relative MC for long-only equity-like targets).** GLD is long-only equity-class; under absolute-MaxDD threshold MC, even buy-and-hold GLD can fail the 35% gate. Document the absolute-vs-relative trade-off in §4. If C1 fails absolute MaxDD, run `run_relative_block_mc()` with `benchmark_fn=buy_and_hold_gld`.
- **L18 (shift discipline).** Position must shift by 1 bar; the strategy_fn implementation MUST mirror the sweep implementation in `research/exploration/sweep_bond_gold.py`. Parity test in `tests/`.
- **L27 (plateau).** Tested as H1.
- **L43 (knife-edge plateau).** Sweep ruled this out on IS; H1 tests on OOS.
- **L46 (sample-size CI bottleneck).** May bind if WFO produces few effective folds. Report verdict + binding constraint separately.
- **L49 (regime-artifact).** GLD's 2002-2011 secular bull run could dominate the WFO IS. Diagnostic: compute per-decade rolling Sharpe in §4.7 to confirm the signal is not regime-localised.
- **L52 (hybrid workflow self-test).** This audit is the first APPLIED case of the L52 workflow. If H1 fails (IS plateau does not hold OOS), the L52 catalogue note must be amended with the failure mode.

## §6. What "complete" looks like

- §4 fully appended with all 8 cells' 5-axis matrix.
- One of three outcomes:
  - **DEPLOY** (C1 passes all gates): live config can be migrated to `lookback=120, threshold=0.50`. Migration requires a separate `config/bond_gold.toml` change + parity test against the new audit cell. Live deployment delayed until migration test passes.
  - **CONDITIONAL_WATCHPOINT** (C1 passes decision matrix but blocked by CI_lo ≤ 0): shadow-deploy the new canonical alongside the live one for 6-12 months; re-audit when CI_lo crosses zero.
  - **RETIRED** (C1 fails plateau OR sanctuary divergence OR sanctuary Sharpe): bond_gold's V1-era live config remains (no degradation), but flag the strategy as "V1-claim unconfirmed; further data required" in the dashboard.

In all three cases: update the V1-era roster (`directives/V1-era Re-audit Sweep Roster 2026-05-16.md`) to mark bond_gold's Wave A.1 status as COMPLETE, with the verdict.
