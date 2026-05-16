# Pre-Registration — J3: Noise-Injection Robustness 5th Decision Axis

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Infrastructure upgrade (framework axis), not a strategy audit
**Status:** §1–§3 frozen at commit; §4 result log appended after re-audits

> This is a V3.1 pre-registration. §1–§3 stay frozen for the lifetime of the rollout. The gate (threshold values) can only be RELAXED in a separate PR explaining why the original was unimplementable.

---

## §1. Motivation & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` step 2 of the recommended execution order:

> **J3 — Noise-injection robustness gate (2d) — becomes 5th axis we apply to all subsequent strategies — NEXT**

**Source.** `resources/Signal vs. Noise_ A Masterclass in Quantitative Robustness.md` (Samir Varma's noise-injection methodology). Distilled into the V3.6 framework discipline as:

> A robust signal is a reflection of a persistent market inefficiency, whereas noise is a temporary alignment of data that will never repeat. To survive, you must distinguish between the two by observing how your system degrades under pressure.

The framework's 4-axis decision matrix (CI_lo / DSR / MC / Sanctuary) tests **statistical confidence** in a Sharpe estimate, **multiple-testing correction**, **tail-risk under regime-shuffled MC**, and **out-of-sample stability**. None of those axes catch the failure mode where a strategy is positioned in a **narrow parameter slot** that any reasonable price-path perturbation evaporates.

**Hypothesis.** A strategy that fails Varma's noise-injection gate at ≤0.3σ relative noise has overfit to a specific data realisation. Adding this gate as a 5th axis catches a class of overfit that V3.2's plateau rule catches *if* the operator remembers to run a ±1-step grid sweep around the canonical cell — but does NOT catch when the operator only runs the canonical cell. The noise gate is **automatic**: it runs once per cell and emits a binary pass/fail without needing a grid sweep.

**Mechanism.**
1. For each cell, after producing stitched OOS returns, run `run_noise_robustness(visible_data, strategy_fn, periods_per_year, cfg=NoiseConfig(...))`.
2. The gate perturbs each input close series by adding Gaussian noise with σ = `noise_level × σ_log_returns`, re-runs the strategy on each perturbed sample, and reports degradation = `(sr_base - sr_at_noise_level) / |sr_base|` at each level.
3. Classify the resulting axis as `best` / `mid` / `worst`:
   - **best**: PASSES at every noise level on BOTH mean Sharpe AND 5th-percentile Sharpe (worst-case across trials).
   - **mid**: PASSES at every noise level on mean Sharpe BUT FAILS the worst-case (some trial degraded >30%).
   - **worst**: FAILS the mean-Sharpe gate at any noise level (strategy is fragile to input noise).

**Why a 5th axis and not just a kill-switch?** Some strategies (path-dependent, with stops, with regime gates) genuinely have non-linear responses to small input noise without being "overfit" in the classic sense — for example a trend-follower may flicker between long and flat under small perturbations near the trend boundary. The 5-axis decision matrix treats such a strategy as `mid` rather than `worst`, allowing it to still reach `CONDITIONAL_WATCHPOINT` if the other four axes are strong. This preserves the V3.2 principle that an overall robust strategy can have one weak axis without being retired.

**Causality / look-ahead concerns.** None. The noise gate operates on COMPLETED stitched OOS returns plus the underlying close series — no information from the noise-perturbed input flows back into the strategy's IS calibration.

## §2. Specification

### §2.1 API contract changes (`titan/research/framework/decision.py`)

**`DecisionInputs` gains TWO new fields** (no `Optional[]` — these are now load-bearing):

```python
@dataclass(frozen=True)
class DecisionInputs:
    ci_lo: float
    dsr_prob: float
    p_maxdd_gt_threshold: float
    pass_threshold_prob: float
    sanctuary_sharpe: float
    # NEW (J3):
    noise_passes_mean: bool      # True iff degradation_mean < cfg.max_degradation at every level
    noise_passes_worst: bool     # True iff degradation_p5  < cfg.max_degradation at every level
```

**`GateThresholds`** is unchanged — the noise axis uses the binary pass flags directly. Future tightening can move the boolean classification into thresholds (e.g. allow `max_degradation_best_pp = 0.20` for a tighter "best" tier).

**`classify_axis_noise(noise_passes_mean, noise_passes_worst)`** returns:
- `"best"` if `noise_passes_worst` (which implies `noise_passes_mean`).
- `"mid"` if `noise_passes_mean` but not `noise_passes_worst`.
- `"worst"` otherwise.

**`DecisionResult`** gains `noise_axis: AxisLevel` field, populated by `classify_axis_noise`.

**`decide(...)` verdict mapping** is updated from 4-axis to 5-axis:

| `n_axes_best` | Verdict |
|:---:|:---|
| 5 | DEPLOY |
| 4 | CONDITIONAL_WATCHPOINT |
| 3 | TIER_UNCONFIRMED |
| 2 | SUSPECT |
| 0–1 | RETIRE |

This collapses the worst two buckets (0 and 1) into `RETIRE` because with 5 axes the marginal information of "1 axis passes" vs "0 axes pass" is small — both indicate fundamental fragility. The mid buckets retain their meaning: `CONDITIONAL_WATCHPOINT` is still "one weak axis", `TIER_UNCONFIRMED` is "two weak axes". (Equivalent to: with 5 axes, you need a majority `best` to clear `SUSPECT` and unanimity to `DEPLOY`.)

> NOTE: this changes the **bar for DEPLOY**: a strategy with 4-of-4 best under the old matrix is now 4-of-5 under the new matrix UNLESS it also passes the noise gate. Re-audits below will surface any GEM cells that demote from `DEPLOY` to `CONDITIONAL_WATCHPOINT`.

### §2.2 Audit harness contract

Every strategy audit harness MUST:
1. Run `run_noise_robustness(...)` on the same data slice used for the cell's stitched OOS evaluation (typically the `visible` window — i.e. excluding sanctuary).
2. Pass `noise_passes_mean` and `noise_passes_worst` to `DecisionInputs` per cell.
3. Report the noise sweep table in the result log alongside the other 4 axes.

For multi-cell audits, the noise gate runs **per cell**, not just on the canonical one. Cost: each cell's noise gate runs `n_trials × len(noise_levels)` strategy invocations (default 10 × 3 = 30) on the visible window. GEM's full 15-cell sweep is ≈ 450 invocations; at the existing strategy_fn speed (~50 ms per invocation on visible window) that's ≈ 25 seconds added end-to-end. Acceptable.

### §2.3 Defaults

```python
NoiseConfig(
    noise_levels=(0.1, 0.3, 0.5),
    n_trials=10,
    max_degradation=0.30,
    seed=42,
    method="additive",
)
```

These were the Varma-recommended defaults already used by GEM in its pre-J3 standalone noise pass. We keep them. Per-strategy-class overrides are reserved for the future (no class has a strong reason to deviate today).

### §2.4 Pre-committed thresholds (frozen, V3.1)

| Axis pre-J3 | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe) | `> 0` | `≤ -0.2` |
| DSR-prob (deflated at N trials, actual skew/kurt) | `≥ 0.95` | `< 0.50` |
| MC P(MaxDD > class_threshold) | `≤ pass_threshold_prob` | `≥ 2 × pass_threshold_prob` |
| Sanctuary Sharpe (on held-out 12mo) | `> 0` | `≤ -0.3` |

| **Axis NEW (J3)** | **Best** | **Worst** |
|---|---|---|
| **Noise robustness (Varma)** | **`degradation_mean < 0.3` AND `degradation_p5 < 0.3` at every noise_level in (0.1, 0.3, 0.5)** | **`degradation_mean ≥ 0.3` at any noise_level** |

### §2.5 Tests (V3.6 discipline)

- `tests/test_framework_synthetic.py::test_decision_matrix_totality` updated from 81 cells (3⁴) to **243 cells (3⁵)**. Asserts every combination produces a verdict (no `UNDETERMINED`).
- New: `test_decision_5axis_verdict_thresholds` verifies the n_axes_best → verdict mapping (5 → DEPLOY, 0 → RETIRE, etc.) explicitly.
- New: `test_classify_axis_noise` asserts the truth table for (passes_mean, passes_worst) → axis-level.
- All existing tests in `tests/test_robustness.py` remain green — the underlying noise gate API is unchanged.

## §3. Decision rule (pre-committed, V3.1)

**Cell selection unchanged.** The plateau rule (V3.2) and parsimony tie-breaker still apply. The 5-axis matrix only affects the VERDICT assigned to each cell.

**Re-audit obligation.** After the framework change lands, the GEM result log §4 must be re-emitted using the 5-axis verdict. Any cell that demoted from `DEPLOY` to `CONDITIONAL_WATCHPOINT` MUST be called out in the appended log entry; if the production cell C12 demotes, the deployment status of `titan/strategies/gem/` is reviewed in a separate PR. (This is the V3.6 discipline: a methodology change CAN invalidate prior verdicts; we honour that even when the strategy is already live.)

**Selection rule for NEW strategies** (everything from J3 onward): a cell is only eligible for `DEPLOY` if it passes ALL 5 axes at `best`. `CONDITIONAL_WATCHPOINT` allowed for production deployment ONLY if the failing axis is NOT the noise axis. (Rationale: a strategy that fails the noise gate is by construction fragile to small data perturbations — there's no operational fix; the algorithm itself needs revision.)

## §4. Result log

### §4.1 Framework tests

`pytest tests/test_framework_synthetic.py tests/test_robustness.py -v` → **31/31 PASS**. Notable:

- `test_decision_total_function_covers_all_combinations` — 243-cell (3⁵) totality, all 5 verdict levels reachable.
- `test_decision_4_of_5_returns_conditional_watchpoint` — the canonical J3 case (4 stats axes best + noise worst → CONDITIONAL_WATCHPOINT).
- `test_classify_axis_noise_truth_table` — truth table for `(passes_mean, passes_worst) -> {best, mid, worst}`.
- `test_decision_5axis_verdict_thresholds` — explicit n_best → verdict mapping (5→DEPLOY, 4→COND_WP, 3→TIER_UNCONFIRMED, 2→SUSPECT, 1→RETIRE, 0→RETIRE).

Full repo: **372/372 tests pass**. Ruff: clean.

### §4.2 GEM re-audit under 5-axis matrix

`uv run python research/gem/run_gem_audit.py` completed (~30 min on 8-core MC parallel). Results from `.tmp/reports/gem_us/result_log.md`:

| Cell | 4-axis (pre-J3) | 5-axis (post-J3) | Noise axis | Verdict change |
|---|---|---|:---:|---|
| C1_canonical | COND_WP | COND_WP | best | unchanged |
| C2_no_buffer | COND_WP | TIER_UNCONFIRMED | mid | DEMOTED (−1) |
| C3_short_lookback | COND_WP | COND_WP | best | unchanged |
| C4_long_lookback | COND_WP | COND_WP | best | unchanged |
| C5_no_defensive | COND_WP | COND_WP | best | unchanged |
| C6_blend_3_6_12 | COND_WP | TIER_UNCONFIRMED | mid | DEMOTED (−1) |
| C7_blend_1_3_6 | COND_WP | COND_WP | best | unchanged |
| **C8_blend_voltarget10** | **DEPLOY** | **COND_WP** | **mid** | **DEMOTED (−1)** |
| C9_stress_gated | DEPLOY | COND_WP | mid | DEMOTED (−1) |
| C10_stress_gated_lev_1p5 | COND_WP | TIER_UNCONFIRMED | mid | DEMOTED (−1) |
| C11_composite_stress | COND_WP | TIER_UNCONFIRMED | mid | DEMOTED (−1) |
| **C12_voltarget_lev2 (prod)** | **DEPLOY** | **COND_WP** | **mid** | **DEMOTED (−1)** |
| C13_target20_lev2 | COND_WP | TIER_UNCONFIRMED | worst | DEMOTED (−1) |
| C14_voltarget_dd_breaker | DEPLOY | COND_WP | mid | DEMOTED (−1) |
| C15_voltarget_lev2_dd_breaker | DEPLOY | COND_WP | mid | DEMOTED (−1) |

**No cell holds DEPLOY under the 5-axis matrix.** Every formerly-DEPLOY cell (C8, C9, C12, C14, C15) demotes to CONDITIONAL_WATCHPOINT because of noise-axis = `mid` (passes mean degradation gate but fails worst-case at one of the 0.1/0.3/0.5σ levels). C13 demotes to TIER_UNCONFIRMED specifically because noise axis = `worst` (the 0.20 vol-target version is more fragile than the 0.10 baseline).

**Noise-gate observations:**

- The 5 cells with noise axis = `best` (C1, C3, C4, C5, C7) are exactly the cells WITHOUT a vol-target overlay (their `ann_vol_target` is None). The base GEM Antonacci design is noise-robust.
- The vol-target overlay (`ann_vol_target=0.10`) consistently produces noise axis = `mid` — the position-scaling math amplifies small input perturbations into larger weight changes, breaking some bad-luck trials.
- This is consistent with L19: continuous vol-targeting Sharpe-dominates binary stress gating, but ALSO adds path-dependence that the noise gate detects as worst-case fragility.

### §4.3 Production deployment review (C12)

**Current state:** C12 (`config/gem_voltarget_lev2.toml`) is live at `titan/strategies/gem/` via `scripts/run_live_gem.py`. The original GEM pre-reg's §4.4 verdict was **DEPLOY** under the 4-axis matrix.

**Under the 5-axis matrix:** C12 verdict is **CONDITIONAL_WATCHPOINT** (4 of 5 axes best; failing axis = noise). The mean-degradation gate PASSES (Sharpe survives 30%-degradation threshold at all noise levels in expectation). The worst-case gate FAILS at 0.5σ (5th-percentile across 10 noise trials breaches the threshold).

**J3 pre-reg §3 rule for new strategies:** *"`CONDITIONAL_WATCHPOINT` allowed for production deployment ONLY if the failing axis is NOT the noise axis."* C12's failing axis IS the noise axis. Under that rule, C12 would NOT be eligible for NEW deployment. For EXISTING deployment, the pre-reg defers to "a separate PR".

**Initial recommendation (2026-05-15, AM):** KEEP C12 LIVE with watchpoint instrumentation, pending a follow-up redesign PR. Reasoning preserved below for historical record:

1. C12 has been running well in paper trading; the noise-axis `mid` (not `worst`) indicates the strategy's MEAN behaviour is robust — it's the tail of input-noise trials that breaches the threshold. This is "stochastic fragility under input noise", not a structural design defect like E1's bare-threshold gates.
2. The 4 statistical-confidence axes (CI_lo +0.40, DSR 1.0, MC 0.06, Sanctuary +0.87) are STRONGER than what C12 had pre-J3.
3. The deployment risk is bounded by the existing kill-switch + vol-target overlay + portfolio risk manager.
4. Pausing C12 means losing capital-at-work without a better alternative; no other cell holds DEPLOY either.

**Three redesign options proposed (each its own fresh pre-reg under V3.1):**

- Option A: smooth the `realised_vol_20d` denominator (EWMA or longer window).
- Option B: cap the position-scaling rate per bar (`|Δw| < 0.05` per day).
- Option C: percentile-based vol-target (rolling quantile of realised vol).

### §4.3.1 Resolution (2026-05-15, PM)

The follow-up PR landed the same day as **`directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md`**. All three mitigations (A/B/C) were tested as parallel cells in the J4 audit. Outcome:

| Mitigation | Cells | Outcome |
|---|---|---|
| A (EWMA / longer window) | A1_ewma_hl40, A2_window_60 | **A1 WINS** — DEPLOY, noise axis recovered to `best`, +0.387 CI_lo. A2 still `mid`. |
| B (per-bar Δw cap) | B1_cap_5pct, B2_cap_10pct | Both DEPLOY but at materially lower CI_lo (+0.19, +0.27) and Sharpe (-30%). |
| C (rolling-quantile target) | C1_qtile_q40, C2_qtile_q50 | Both demoted noise axis from `mid` → **`worst`** (L32 — the L26 percentile pattern does not transfer to denominator-style overlays). |

**Final production decision: A1_ewma_hl40 promoted to production.** Deployed on the Docker paper account at 2026-05-15 13:07 UTC. The C12 → A1 trajectory now closes the J3 deployment-review loop with a true 5/5-axis DEPLOY verdict. See J4 pre-reg §4 for full audit numbers, V3.6 Catalogue L30/L31/L32 for the lesson set, and `.tmp/reports/gem_j4/dashboard.html` for visual inspection.

### §4.4 New lessons (appended to V3.6)

- **L30 (new)**: Vol-target overlays produce noise-axis = `mid` even when noise-axis = `best` holds for the un-overlaid base strategy. Mechanism: scaling position by `min(1, vol_target / realised_vol_20d)` amplifies short-window noise on the realised-vol estimate into larger position swings on the bars where realised vol is itself noise-perturbed. The mean over trials still passes (the vol estimate noise is unbiased), but the 5th-percentile worst case does not. **Mitigation patterns:** smooth the realised-vol estimator (EWMA, longer window), cap per-bar position-change rate, or replace fixed-value vol target with a rolling-percentile target (L26 generalised). **How to apply:** when adding a vol-target overlay to a noise-best base strategy, expect the noise axis to demote to `mid`; either accept the CONDITIONAL_WATCHPOINT outcome (acknowledging the bounded-trade-but-stochastic-fragility status) or apply one of the mitigation patterns. **Source.** GEM J3 5-axis re-audit, 2026-05-15: cells C8, C9, C12, C14, C15 (all with `ann_vol_target=0.10`) demoted from `best` to `mid` on the noise axis vs C1, C3, C4, C5, C7 (no vol-target overlay) which remained `best`.

---

## §5. Failure modes to watch

- **L04 / A1 — Same-bar look-ahead.** The noise gate must NOT leak any of the perturbed close into the strategy's IS calibration. `run_noise_robustness` is purely a wrapper around the strategy_fn — confirm strategy_fn itself enforces causality (already covered by existing `gem_assert_causal` tests).
- **L08 — Class-specific calibration.** The noise thresholds (`max_degradation=0.3` at 0.5σ) are tuned for daily-bar long-only equity. Intraday microstructure strategies may need a tighter threshold (their edge is more sensitive to input noise by definition). Pre-J3, we keep one global threshold; a class-specific override is on the J5 backlog if any strategy class fails this gate "for the wrong reasons".
- **A4 — Honesty.** This pre-reg lands BEFORE any noise gate result is collected on a non-GEM strategy. The existing GEM Step-3 noise result (C8 PASSES) is informational; the J3 axis decision lives going forward, not retroactively (the pre-reg is the contract).
- **V3.1 — No retroactive cell-favouring.** If the 5-axis matrix demotes a previously-DEPLOY cell, the next PR shall NOT relax the noise gate to re-promote it. The PR shall instead document the demotion and address it via algorithm change (or accept the lower tier).
- **V3.2 — Plateau-not-peak.** The noise gate is COMPLEMENTARY to the plateau rule, not a substitute. A cell can pass the noise gate while failing the plateau rule (e.g. an isolated parameter spike where small input noise doesn't unseat the local maximum but adjacent parameter values do). Both gates must hold for a `DEPLOY` verdict.
- **A8 — "Validated against X" requires an artifact.** This directive's §4.1 result log + the committed `.tmp/reports/gem_us/result_log.md` together constitute the artifact for the claim "GEM C12 passes the 5-axis matrix".

## §6. Implementation plan

1. **Extend `titan/research/framework/decision.py`:** add `classify_axis_noise`, two new `DecisionInputs` fields, update `decide(...)` to 5 axes, update `DecisionResult.noise_axis`, update rationale formatting.
2. **Update `tests/test_framework_synthetic.py`:** 81 → 243 cells; new `test_classify_axis_noise`; new `test_decision_5axis_verdict_thresholds`.
3. **Update `research/gem/run_gem_audit.py`:** run `run_noise_robustness` per cell (not just canonical), thread `noise_passes_mean` + `noise_passes_worst` into `DecisionInputs`, expand the §4.1 result-log table to include the noise column.
4. **Append L23 to `directives/V3.6 Lessons Catalogue.md`:** "Noise-injection robustness is the 5th axis of the decision matrix. A strategy that passes 4 axes but fails the noise gate is `CONDITIONAL_WATCHPOINT`, not `DEPLOY`."
5. **Run pre-push gates:** `uv run ruff check . --fix` → `uv run ruff format .` → `uv run pytest tests/test_framework_synthetic.py tests/test_robustness.py tests/test_gem.py -v`.
6. **Re-run GEM audit:** `uv run python research/gem/run_gem_audit.py`; if C12 demotes, file follow-up PR per §3.

After J3 lands, the framework's `decide()` function is the single source of truth for verdict assignment across all 14+ live strategies and every future audit. The next backlog step (E1 — VRP capture) consumes this 5-axis matrix from day one.
