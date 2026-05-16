# Pre-Registration — J4: GEM Noise-Robust Redesign

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy-redesign audit (production cell C12 noise-robustness recovery)
**Predecessor:** Pre-Reg J3 §4.3 — production cell C12 demoted DEPLOY → CONDITIONAL_WATCHPOINT under the 5-axis matrix because the vol-target overlay's `min(1, target_vol / realised_vol_20d)` arithmetic amplifies short-window noise into position swings (L30).
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> This is a V3.1 pre-registration committed BEFORE running the J4 audit. The empirical evidence from J3 (the FAILURE of C12 on the noise axis with axis=`mid`) is the *motivation* for this pre-reg, but the specific cells, gates, and selection rule below have not been tested against this design's parameter grid.

---

## §1. Motivation & mechanism

**J3 outcome (L30).** GEM cells with `ann_vol_target=0.10` (C8, C9, C12, C14, C15) uniformly demoted from noise axis = `best` to `mid` under the 5-axis matrix. C13 (`ann_vol_target=0.20`) demoted to `worst`. The base un-overlaid Antonacci variants (C1, C3, C4, C5, C7 — no `ann_vol_target`) all held noise axis = `best`. **The vol-target arithmetic is what introduces the noise fragility.**

The L30 mitigation patterns each address a different mechanical noise-source in the overlay:

- **Mitigation A — EWMA / longer-window vol estimator.** The 20-day rolling-std denominator is sensitive to single-bar noise. EWMA with half-life ≥ 40 or a 60-day window reduces estimator noise.
- **Mitigation B — Per-bar position-change cap.** Even with a noisy vol estimator, the position cannot move faster than the cap allows. Decouples position trajectory from vol-estimator trajectory; noise can't propagate into large weight swings within a single bar.
- **Mitigation C — Rolling-percentile vol target.** Replace fixed `target_vol = 0.10` with `target_vol = rolling_quantile(realised_vol, window=252, q=q*)`. The target moves with the data; input noise affects both numerator and denominator, partially cancelling.

**Hypothesis (J4).** At least one of the three mitigations (or the right combination) recovers GEM's noise axis to `best` while preserving the 4 statistical axes (CI_lo, DSR, MC, Sanctuary) at the `best` levels achieved by C12. Falsifiable: if every J4 cell still fails the noise axis at `best` (i.e. all `mid` or `worst`), the vol-target overlay is intrinsically noise-fragile at this asset frequency and the production stance must permanently accept CONDITIONAL_WATCHPOINT (or remove the overlay, regressing to non-levered C6-blend performance).

**Why three mitigations in one pre-reg.** Each mitigation isolates a different mechanism in the overlay. Running them as ORTHOGONAL cells lets us attribute *which mechanism* drives the noise demotion. The result will inform a future redesign PR that may combine the winning mitigations.

## §2. Universe + cells + data

**Universe:** unchanged from J3 / GEM canonical — SPY, EFA, IEF daily total-return parquets in `data/`. Optional regime extras (VIX, HYG) used in the existing harness; not modified for J4.

**Date range:** 2003-01-02 → present (limited by EFA inception). Sanctuary: trailing 12 months. Visible window ≈ 22 years.

**Bar timeframe:** Daily. `BARS_PER_YEAR["D"] = 252`. Strategy class: `CROSS_ASSET_MOMENTUM`.

**Cells (V3.1 frozen, 7 cells).** Baseline is C12 (`config/gem_voltarget_lev2.toml` — the current production cell). Each mitigation has 2 dose levels to test sensitivity.

| Cell | vol_estimator_kind | vol_estimator_halflife | vol_lookback_days | max_weight_delta_per_bar | vol_target_kind | vol_target_quantile | vol_target_quantile_window | ann_vol_target | max_leverage |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| **C0_baseline** (C12 reproduction) | rolling_std | — | 20 | None | fixed | — | — | 0.10 | 2.0 |
| **A1_ewma_hl40** | ewma | 40 | — | None | fixed | — | — | 0.10 | 2.0 |
| **A2_window_60** | rolling_std | — | 60 | None | fixed | — | — | 0.10 | 2.0 |
| **B1_cap_5pct** | rolling_std | — | 20 | 0.05 | fixed | — | — | 0.10 | 2.0 |
| **B2_cap_10pct** | rolling_std | — | 20 | 0.10 | fixed | — | — | 0.10 | 2.0 |
| **C1_qtile_q40** | rolling_std | — | 20 | None | rolling_quantile | 0.40 | 252 | — | 2.0 |
| **C2_qtile_q50** | rolling_std | — | 20 | None | rolling_quantile | 0.50 | 252 | — | 2.0 |

**7 cells total.** DSR adjustment applies (N=7 > 5).

**C0_baseline** is mechanically C12 — exists ONLY to verify the new harness reproduces J3's C12 numbers (sanity check; not a candidate for promotion). All other parameters (lookback_blend=(3,6,12), buffer_pct=0.005, defensive_switch=True, etc.) are FIXED at C12's values across all 7 cells. The audit isolates the effect of each mitigation.

## §3. Decision rule (pre-committed, V3.1)

**Class defaults:** `defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)` — unchanged.

**Per-axis thresholds (5-axis matrix, L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo | > 0 | ≤ −0.2 |
| DSR-prob | ≥ 0.95 | < 0.50 |
| MC (relative gate, L17) — median DD ratio | ≤ 0.80 | ≥ 1.6 |
| Sanctuary Sharpe | > 0 | ≤ −0.3 |
| Noise (Varma, J3) | passes mean AND worst-case at (0.1, 0.3, 0.5)σ × 10 trials | fails mean gate at any level |

**Selection rule (V3.1 + V3.2).** A cell is eligible for **PROMOTION TO PRODUCTION** iff:

1. Verdict = DEPLOY (all 5 axes at `best`), OR
2. Verdict = CONDITIONAL_WATCHPOINT AND the failing axis is **NOT** the noise axis (per J3 pre-reg rule).

Among eligible cells, pick the one with the **highest CI_lo** as the tie-breaker (most statistically robust). If multiple mitigations qualify, document the COMBINATION as a follow-up pre-reg candidate.

**Plateau pre-flight (L27).** Before the full audit, compute stitched-OOS Sharpe for C0_baseline + each of the 6 mitigation cells. Report the relative spread. If spread > 30%, ABORT and document — the mitigations themselves introduce too much parameter sensitivity. **DO NOT auto-abort like E1b**: this is a redesign audit where some Sharpe variation is expected (the whole point is to find the variant that's most noise-robust). Instead, flag the spread in the result log and proceed; the verdict map below is the deployment gate.

**No retroactive cell-favouring (V3.1).** If C0_baseline (the C12 reproduction) doesn't match C12's J3 numbers within tolerance (Sharpe ±0.05, noise axis identical), the audit fails its sanity check and we STOP — there's a harness bug to debug before we can trust the new cells' results.

**Causality test (A10 / L04).** Existing `gem_assert_causal` test extended to cover the new GemConfig fields. Pre-commit assertion before audit runs.

## §4. Result log

**Audit run:** 2026-05-15. Full output in `.tmp/reports/gem_j4/result_log.md`. WFO: 22 folds, ~17m runtime on 8-core MC parallel.

### §4.1 Sanity check (C0_baseline vs C12 from J3)

**PASS.** C0_baseline Sharpe = +0.8016, identical to J3 C12 (|Δ| = 0.0000, tolerance 0.05). Noise axis = `mid`, matching J3 C12. The harness is correct; J4 results are trusted.

### §4.2 Per-cell verdicts (5-axis)

| Cell | Sharpe | CI95 lo | CI95 hi | DSR | Rel MC ratio | Sanc Sharpe | Noise base | Noise axis | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| C0_baseline (=C12 prod) | +0.8016 | +0.402 | +1.164 | 1.0000 | 0.5748 | +0.8717 | +0.7596 | mid | COND_WP |
| **A1_ewma_hl40** | **+0.7773** | **+0.387** | **+1.140** | **1.0000** | **0.5657** | **+0.8856** | **+0.7549** | **best** | **DEPLOY** |
| A2_window_60 | +0.7904 | +0.392 | +1.143 | 1.0000 | 0.6044 | +0.7643 | +0.7812 | mid | COND_WP |
| **B1_cap_5pct** | **+0.5621** | **+0.192** | **+0.943** | **1.0000** | **0.5977** | **+0.9273** | **+0.5599** | **best** | **DEPLOY** |
| **B2_cap_10pct** | **+0.6394** | **+0.270** | **+1.013** | **1.0000** | **0.5763** | **+0.9058** | **+0.6316** | **best** | **DEPLOY** |
| C1_qtile_q40 | +0.7937 | +0.410 | +1.150 | 1.0000 | 0.6186 | +0.8889 | +0.7456 | worst | COND_WP |
| C2_qtile_q50 | +0.7641 | +0.381 | +1.126 | 1.0000 | 0.6685 | +0.9511 | +0.7419 | worst | COND_WP |

### §4.3 Plateau spread

29.88% across the 7 cells (informational, not an abort gate). Just under the 30% threshold — the seven cells are similar enough on Sharpe that the headline metric is parameter-robust. The differentiation comes through the NOISE axis, not the point Sharpe.

### §4.4 Mitigation attribution

| Mitigation | Cells | Outcome |
|---|---|---|
| **A (smoother vol estimator)** | A1 (EWMA hl=40), A2 (rolling window 60) | **A1 WINS** (noise axis: mid → best). A2 fails (still mid). EWMA recovers DEPLOY; a longer rolling window does NOT. |
| **B (per-bar Δw cap)** | B1 (cap=0.05), B2 (cap=0.10) | **BOTH WIN** (mid → best). Tighter cap = lower Sharpe (more leftover bleed into IEF), but both recover DEPLOY. |
| **C (rolling-percentile vol target)** | C1 (q=0.40), C2 (q=0.50) | **BOTH FAIL** (mid → worst). The rolling quantile makes the target itself path-dependent on noise — degrades, not improves, the noise axis. Lesson: the mitigation pattern that helped on VRP regime gates (E1b L26) does NOT transfer to vol-target denominators here. |

**Conclusion.** Two of three mitigation MECHANISMS work — and they work via different routes:

- **Mitigation A (EWMA)** attacks the noise at the source: smoother vol estimator → smoother position trajectory.
- **Mitigation B (Δw cap)** attacks the noise downstream: even if the vol estimate is noisy, the position can't move faster than the cap allows.

**Mitigation C** intuitively SHOULD have worked (it's the L26 pattern that the user requested for percentile-gating elsewhere), but rolling-percentile of realised vol amplifies single-bar noise into the target itself — both numerator AND denominator move together with noise but with phase lag, producing worse position swings than the fixed target. L26's pattern is correct for THRESHOLD-style signals (where the noise unilaterally crosses the gate); it fails for DENOMINATOR-style scaling overlays where the noisy quantity is on both sides.

### §4.5 Recommended production change

**Promote A1_ewma_hl40 to production.** It has the highest CI_lo among DEPLOY cells (the J3 §3 tie-breaker rule). Concrete config change to `config/gem_voltarget_lev2.toml`:

```toml
# J4 (2026-05-15): replace 20-day rolling-std vol estimator with EWMA half-life=40.
# Recovers GEM C12 from CONDITIONAL_WATCHPOINT to DEPLOY under the 5-axis matrix
# (J3, L24). Sharpe trade-off: +0.7773 (A1) vs +0.8016 (baseline) — 3% lower point
# Sharpe in exchange for noise-axis = best (was mid). CI_lo +0.387 stays above zero
# with comfortable margin. See directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md.
vol_estimator_kind = "ewma"
vol_estimator_halflife = 40
# (all other parameters unchanged from current C12 production)
```

**Trade-off acknowledged.** A1 has ~3% lower point Sharpe than C0_baseline (+0.7773 vs +0.8016) but materially better noise robustness (axis = best vs mid). Per the J3 spirit — "a strategy that fails the noise gate is by construction fragile to small data perturbations" — A1 is the right deployment.

**Why not B1/B2.** Both achieve DEPLOY but at materially lower point Sharpe (+0.56, +0.64 vs +0.78). The position-change cap mechanically slows the strategy's response to genuine regime changes too, not just noise. EWMA is the more surgical fix — it only changes the vol estimator, not the strategy's response speed to real moves.

**Why not C1/C2.** Both FAIL the noise axis. Rolling-quantile vol target is the wrong tool for this job; documented as the negative arm of L31 below.

**Operational rollout** (separate PR per J3 §3 — this directive only documents the audit + recommendation):

1. Update `config/gem_voltarget_lev2.toml` with the two new fields.
2. Verify `titan/strategies/gem/strategy.py` reads them through `GemConfig` (it does — `gem_strategy.GemConfig` is the source of truth and the live class consumes it).
3. Add `tests/test_gem_live_parity.py` case asserting the live class produces the same per-bar weights as `gem_returns(..., cfg=A1_config)` on a fixed window.
4. Paper-trade for 5 sessions before going live to confirm the EWMA path doesn't surprise the live infrastructure.

### §4.6 New lessons (appended to V3.6)

- **L31 (new)**: Of the three L30 mitigations for noise-fragile vol-target overlays, **EWMA vol estimator (A) is strictly preferred over per-bar Δw cap (B) or rolling-percentile target (C)** for the CROSS_ASSET_MOMENTUM class. EWMA recovers noise=best at minimal point-Sharpe cost (~3% drag); Δw cap recovers noise=best but at ~20–30% Sharpe drag (because it also slows the response to genuine regime shifts); rolling-quantile target FAILS the noise axis entirely (target itself becomes path-dependent on noise). Lesson generalisation: when adding a vol-target overlay, prefer SMOOTHER ESTIMATORS over downstream CAPS; AVOID rolling-quantile vol targets. Apply this pattern when scoping any new noise-robust strategy designs. **How to apply.** Default `vol_estimator_kind = "ewma"`, `vol_estimator_halflife = 40` whenever `ann_vol_target` is set. Document the choice in the pre-reg's §3. Only switch to per-bar cap when EWMA is infeasible (e.g. operationally — a hard per-trade size limit might be required by risk; that's a different kind of constraint). **Source:** J4 audit 2026-05-15 — three winning cells: A1 (DEPLOY, +0.387 CI_lo), B1 (DEPLOY, +0.192 CI_lo), B2 (DEPLOY, +0.270 CI_lo). A1 dominates on CI_lo + Sharpe. C1/C2 both FAIL the noise axis (noise = `worst` — quantile target degrades the axis vs the baseline's `mid`).

- **L32 (new — falsification)**: The L26 percentile-gating pattern (which addresses noise fragility in THRESHOLD signals) does NOT transfer to DENOMINATOR-style overlays. The mechanism is different: a percentile threshold on a signal moves the gate alongside the signal so noise affects both sides symmetrically; a rolling-quantile vol target replaces a constant denominator with a noise-derived quantile — both numerator and denominator move with noise but with phase lag, producing AMPLIFIED position swings (not damped). Empirically: GEM C1_qtile_q40 / C2_qtile_q50 demoted from mid → worst on the noise axis vs C0 baseline. **How to apply.** When a strategy's noise fragility is due to a SCALING DENOMINATOR (vol target, sizing kernel, etc.), reach for L31's mitigation set (EWMA / Δw cap) — NOT for L26's percentile pattern. The two cover disjoint regions of the design space. Document the distinction in any future redesign pre-reg.

---

## §5. Failure modes to watch

- **L04 / A1** — Existing causality test extended to cover the 3 new mitigation paths (EWMA vol estimator, position-change cap, rolling-quantile target). Each must satisfy: corrupting close[t] does not change weights[t' < t].
- **L17** — GEM is long-only equity; use the RELATIVE MC gate (median DD ratio ≤ 0.80) not the absolute one.
- **L20 — Index normalisation.** No new cross-series merges introduced.
- **L24 — Per-cell noise gate.** Every cell receives the full Varma sweep (0.1, 0.3, 0.5)σ × 10 trials.
- **L27 — Plateau pre-flight.** Run + report, but do NOT auto-abort (see §3 rationale).
- **L28 — Two-axis fragility.** A mitigation might rescue the noise axis (input-noise robustness) but introduce parameter-spread fragility (e.g. `vol_target_quantile = 0.40` vs `0.50` gives very different verdicts). Both gates must hold.
- **L30 — The thing we're fighting.** Track noise-axis class transitions cell-by-cell. If ANY cell moves to `best`, that's a positive result regardless of overall verdict.
- **A4 — WFO honesty.** Cells pre-registered; per-fold parameter selection is NOT applicable (no IS-trained parameters change across cells in this design).
- **A5 / V3.1 — DSR for N=7.** Apply at the actual sweep size.
- **V3.6 / L16 — Negative-result discipline.** If no cell moves noise axis to `best`, document the mechanism and recommend permanent acceptance of CONDITIONAL_WATCHPOINT for C12. Failure is not an outcome we hide.

## §6. Implementation plan

1. **Extend `GemConfig`** in `research/gem/gem_strategy.py` with the new fields (defaults preserving existing behaviour):
   - `vol_estimator_kind: str = "rolling_std"` (or `"ewma"`)
   - `vol_estimator_halflife: int = 40` (only used when `ewma`)
   - `max_weight_delta_per_bar: float | None = None` (None = no cap)
   - `vol_target_kind: str = "fixed"` (or `"rolling_quantile"`)
   - `vol_target_quantile: float = 0.40` (only used when `rolling_quantile`)
   - `vol_target_quantile_window: int = 252`
2. **Update `_apply_vol_target`** to honour the new fields. Each mitigation is a small, localised modification to the vol-estimator computation or the post-scaling weight processing.
3. **Build the audit harness** in `research/gem_noise_redesign/run_j4_audit.py`. Re-use the existing `run_gem_audit.py` framework primitives (cell looping, MC, noise, decision); just swap CELLS to the 7 J4 cells. Output to `.tmp/reports/gem_j4/result_log.md`.
4. **Tests in `tests/test_gem.py`** (extend existing): one test per new field exercising the mitigation; verify defaults preserve existing behaviour (no regression on existing GEM cells).
5. **Run audit, append §4 result log.**
6. **If a cell promotes:** draft the config change to `config/gem_voltarget_lev2.toml` in a SEPARATE PR (J4 only documents the audit + recommendation). Existing live class in `titan/strategies/gem/` is not touched in this pre-reg.

After J4 lands, the next backlog step is **D2 — Commodity futures carry** (5d, needs 24-commodity data acquisition).
