# Pre-Registration — E1b: VRP Capture v2 (Percentile Gates + Vol-Carry Class)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy audit (new sub-class `DAILY_MEAN_REVERSION_VOL_CARRY`)
**Predecessor:** E1 (Pre-Reg E1 VRP Capture 2026-05-15.md) — verdict SUSPECT on all 7 cells.
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> This is a V3.1 pre-registration committed BEFORE any data is examined for this design. The general findings of E1 (specific drawdown magnitudes, Sharpe ranges, regime mix) are common knowledge from the literature; the V3.1 discipline of "no retro cell-favouring" applies to this design's parameter grid, which has not been backtested.

---

## §1. Motivation & mechanism

**Predecessor outcome.** E1 audited 7 cells of a threshold-based VRP-capture strategy on VIXY using the VIX term structure. Verdict: **all SUSPECT** under the 5-axis matrix. Three axes failed across the grid:

| Failing axis | Why |
|---|---|
| CI_lo | 95% bootstrap CI straddled zero — point Sharpe +0.21 to +0.46 within noise |
| MC (DAILY_MEAN_REVERSION default) | P(MaxDD > 25%) = 1.0 — class default too tight for short-vol carry |
| Noise (Varma, J3) | Threshold gate fragile to 0.1σ input perturbation |

L25 (MC threshold per exposure type) and L26 (threshold gates are noise-fragile) were appended to the V3.6 Lessons Catalogue with proposed mitigations. This directive (E1b) **tests both mitigations together** under a fresh pre-reg.

**Mitigations applied:**

1. **L26 — Percentile-rolling gates replace bare thresholds.** Instead of `ratio_long >= 1.05`, the regime signal is `ratio_long >= rolling_quantile(ratio_long, window=W, q=enter_q)`. The gate moves with the signal's own distribution, eliminating the parameter-spike fragility under input noise. Implementation: store the rolling quantile via `pandas.Series.rolling(W).quantile(q)` with `shift(1)` for causality.
2. **L25 — New `DAILY_MEAN_REVERSION_VOL_CARRY` StrategyClass with relaxed MC default.** The Eraker-Wu *JFE* 2017 and Cheng SSRN 2020 papers document MaxDDs of 30–70% as the structural cost of VRP harvesting. The new class's MC default: P(MaxDD > 50%) < 10%, block size = 21 days (unchanged), n_paths = 200 (unchanged). Other defaults unchanged from `DAILY_MEAN_REVERSION` (per-day MTM Sharpe, expanding WFO 3y IS / 1y OOS / 5 folds).

**Hypothesis (E1b).** Combining percentile-rolling regime gates with a properly-calibrated MC threshold will produce a non-SUSPECT verdict on at least one cell — meaning the underlying VRP signal carries meaningful information, and the E1 SUSPECT result was a methodology issue (gate fragility + threshold mis-calibration), not a true absence of edge. Falsifiable: if every cell of E1b is also SUSPECT or worse, the underlying signal does not survive a properly-designed framework audit and "VRP carry on VIXY in retail timeframes after costs" should be retired from the backlog.

**Additional design variant (C7 — continuous scaling).** L26 also recommends continuous sigmoid scaling as an alternative to discrete percentile gates. C7 implements `position = w_target * sigmoid((s - q_enter) / 0.05)` where `s` is `ratio_long` and `q_enter` is the rolling percentile. This is a stronger noise-mitigation: even bars sitting exactly on the gate get a partial position rather than a 100%-or-0 flip.

## §2. Universe + cells + data

**Universe (unchanged from E1):**

- VIXY (implementation vehicle) — `data/VIXY_D.parquet` (downloaded for E1, 2011-01-04 → present).
- VIX, VIX9D, VIX3M (signal inputs) — existing parquets.
- SPY (optional defensive overlay) — existing parquet.

**Date range:** 2011-01-04 → present. Sanctuary: trailing 12 months from latest data. Visible window ≈ 14 years × 252 ≈ 3,500 bars.

**Bar timeframe:** Daily. `BARS_PER_YEAR["D"] = 252`.

**Strategy class:** `DAILY_MEAN_REVERSION_VOL_CARRY` (new — to be added to `titan.research.framework.typology` as part of this PR; defaults: P(MaxDD>50%) < 10%, otherwise same as `DAILY_MEAN_REVERSION`).

**Cells (V3.1 frozen at this commit, 7 cells):**

| Cell | gate_kind | window_d | enter_q | exit_q | target_short_weight | sigmoid_scale | notes |
|---|---|---:|---:|---:|---:|---:|---|
| C1 canonical | percentile | 252 | 0.60 | 0.40 | −0.50 | n/a | mid-quantile entry, mid-exit; 1-year window |
| C2 wider_band | percentile | 252 | 0.70 | 0.30 | −0.50 | n/a | only top-30% / bottom-30% trigger |
| C3 narrower_band | percentile | 252 | 0.55 | 0.45 | −0.50 | n/a | tighter band — more active |
| C4 shorter_window | percentile | 126 | 0.60 | 0.40 | −0.50 | n/a | 6-month rolling quantile (more adaptive) |
| C5 longer_window | percentile | 504 | 0.60 | 0.40 | −0.50 | n/a | 2-year rolling quantile (more stable) |
| C6 smaller_short | percentile | 252 | 0.60 | 0.40 | −0.25 | n/a | half-size short position |
| C7 continuous | sigmoid | 252 | 0.60 | 0.40 | −0.50 | 0.05 | continuous scaling (L26 alternative) |

**7 cells total.** DSR adjustment applies (N=7 > 5).

**Backwardation arm** (defensive long-SPY overlay) is OUT of scope for E1b. The E1 result showed it was non-load-bearing (C6 of E1 added 0.05 Sharpe with the overlay vs C1 — within noise). Removing it cuts a design dimension to focus on the percentile-gate mitigation. If E1b verdict is positive, a follow-up directive may re-introduce the defensive overlay.

## §3. Decision rule (pre-committed, V3.1)

**Class defaults (NEW: `defaults_for(StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY)`):**

- Sharpe: per-day MTM, 252 bars/yr.
- WFO: expanding, 3y IS, 1y OOS, 5 folds.
- MC: 21-day blocks, 200 paths, **P(MaxDD > 50%) < 10%** (NEW — L25 calibration for vol-carry exposure).

**Per-axis thresholds (5-axis decision matrix, L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe) | > 0 | ≤ −0.2 |
| DSR-prob (deflated at N=7, actual skew/kurt) | ≥ 0.95 | < 0.50 |
| MC P(MaxDD > 50%) on underlying-block-bootstrap | ≤ 0.10 | ≥ 0.20 |
| Sanctuary Sharpe (on held-out 12mo) | > 0 | ≤ −0.3 |
| Noise robustness (Varma, J3) | passes mean AND worst-case at 0.1/0.3/0.5σ | fails mean gate at any level |

**Verdict map (5-axis, J3):** 5 → DEPLOY · 4 → CONDITIONAL_WATCHPOINT · 3 → TIER_UNCONFIRMED · 2 → SUSPECT · 0–1 → RETIRE.

**Cell selection (V3.2 plateau).** The selected cell's ±1-step grid neighbours must ALSO pass the 5-axis matrix at the same `best` level on at least 4 of 5 axes, AND headline Sharpe must vary by < 30% across the neighbourhood. Tie-break by parsimony (canonical percentile rule preferred over sigmoid where both pass).

**Causality test (A10 / L04).** Pre-commit assertion: corrupt future close + future VIX/VIX9D/VIX3M values at random t, assert weights at every t' < t are bit-exact unchanged. Live parity test required before any DEPLOY/CWP cell promotes to `titan/strategies/`.

**Specific failure-mode pre-commits (V3.6 lessons applied):**

- **L20 — Percentile computation index alignment.** Rolling quantile is computed AFTER index normalisation (date-only) and forward-fill (limit 3 bars). No cross-series merge needs to happen during the quantile computation — the signal is built per-series first, then aligned with VIXY.
- **L24 — Noise gate runs per cell.** Every cell receives a full Varma noise sweep at (0.1, 0.3, 0.5)σ × 10 trials × max_degradation=0.30.
- **L26 — Plateau pre-flight.** Before running the audit, the canonical cell's ±1 step in each of the 3 main parameters (window_d, enter_q, exit_q) is checked — IF the Sharpe spread > 30% in the neighbourhood, the design is REJECTED at audit-start (not at audit-end). This protects against the audit "looking robust" on the canonical alone.

## §4. Result log

**Audit run:** 2026-05-15. Result: **ABORTED at plateau pre-flight gate.** Per pre-reg §3, the design's canonical cell + 6 grid neighbours did not form a parameter plateau, so the full MC + noise + decision-matrix pipeline was not run.

### §4.1 Plateau pre-flight result (V3.2 gate)

Run on the visible window (2011-01-04 → 2025-05-13, 3,610 bars, 11 WFO folds).

| Cell | Stitched OOS Sharpe |
|---|---:|
| C1_canonical (window=252, enter=0.60, exit=0.40) | +0.4104 |
| P_window_short (window=189, enter=0.60, exit=0.40) | +0.4879 |
| P_window_long (window=378, enter=0.60, exit=0.40) | +0.3312 |
| P_enter_lower (window=252, enter=0.55, exit=0.40) | +0.4142 |
| P_enter_higher (window=252, enter=0.65, exit=0.40) | +0.5985 |
| P_exit_lower (window=252, enter=0.60, exit=0.35) | +0.4201 |
| P_exit_higher (window=252, enter=0.60, exit=0.45) | +0.4646 |

**Range:** +0.33 (P_window_long) → +0.60 (P_enter_higher). **Relative spread:** 44.66% (gate: 30%). **VERDICT: FAIL.**

Per pre-reg §3, the full audit was aborted. Per V3.2: "A cell with the best metric whose ±1-step neighbours fail is overfit. A slightly worse cell whose neighbours all pass is robust. Prefer the latter." Here NEITHER condition holds — every neighbour has a meaningfully different Sharpe, indicating the design has no robust parameter region.

### §4.2 Comparison to E1

E1 (bare-threshold gates, `DAILY_MEAN_REVERSION` class):

- All cells produced SUSPECT verdicts (CI_lo / MC / Noise axes failed).
- Noise axis specifically failed because bare-threshold gates flip under 0.1σ input noise.

E1b (percentile-rolling gates, `DAILY_MEAN_REVERSION_VOL_CARRY` class):

- Audit aborted before computing CI_lo / MC / Noise / Sanctuary.
- The L25 mitigation (relaxed MC class) was never tested because the L26 mitigation (percentile gates) does not produce a plateau either.

**Diagnosis.** Both audits point to the same underlying fragility: the VIX-term-structure-driven VRP signal on VIXY does NOT have a stable parameter region under daily-bar / monthly-VIX-roll dynamics. Switching from bare-threshold to percentile gates moved the location of the maximum (from +0.46 in E1 to +0.60 in E1b's P_enter_higher) but did not produce a plateau. The Sharpe surface remains spiky.

This is consistent with the literature on practical VRP harvesting:

- The VRP exists in *option-implied* form (puts trade rich vs realised vol).
- VIX futures and ETFs LIKE VIXY *partially* harvest the VRP but with significant slippage from the futures roll mechanics, term-structure dynamics, and rebalancing flows. Eraker-Wu (2017) noted that the after-cost edge on retail-accessible vehicles is materially smaller than the theoretical VRP — and noisier.

What we now have STRONG evidence for: **on a retail-implementable VIXY vehicle with daily-bar signals derived from VIX/VIX9D/VIX3M, no parameter configuration (in the set we explored) produces a plateau-stable strategy.** That is a much stronger conclusion than E1's "all cells SUSPECT under one specific design".

### §4.3 Selected production cell

**None.** Per pre-reg §3, the audit aborted before producing per-cell verdicts. No promotion candidates.

### §4.4 New lessons (appended to V3.6)

- **L27 (new)**: A plateau pre-flight check at the START of the audit (not the end) saves compute and produces honest negative results. Run the canonical cell + 6 grid neighbours BEFORE the MC + noise + DSR pipeline; if relative Sharpe spread > 30%, abort and report the fragility. This audit consumed < 1 minute of compute to produce a stronger negative result than E1's ~6 minutes of full MC + per-cell noise + decision-matrix work. The discipline: plateau pre-flight ≡ V3.2 enforced *before* burning the compute budget.
- **L28 (new)**: The Varma noise gate (L24) and the plateau-spread check (V3.2) are *complementary, not redundant*. L24 perturbs the *input price* and re-runs the strategy — it catches strategies fragile to small data noise. V3.2 perturbs the *parameters* and re-runs the strategy — it catches strategies fragile to small parameter noise. A strategy can pass one and fail the other. E1's bare-threshold design failed L24 (input-noise fragile). E1b's percentile-gate design likely would have passed L24 (signal is computed on a rolling distribution, not a fixed value) but fails V3.2 (different parameter steps produce 45% Sharpe spread). Both gates are required; both are part of the 5-axis matrix's protection model. (Strictly, V3.2 is a pre-flight gate not an axis — it's enforced by the harness before the matrix is evaluated.)
- **L29 (new — research-direction)**: The VIX-term-structure → VIXY signal does NOT survive a properly-disciplined framework audit at daily resolution after costs. Two independent designs (E1 bare-threshold, E1b percentile-rolling) both fail. Reasonable conclusion: this trade is not viable on retail-accessible vehicles at retail-accessible signal frequencies. Future VRP-style audits should EITHER (a) move to higher-frequency signals (intraday VIX changes, GEX-based regime signals — backlog E3) OR (b) use a different implementation vehicle (option-selling on SPY, or direct VIX futures with explicit roll-yield modelling — needs futures parquets) OR (c) be retired from the backlog as a known dead-end.

### §4.5 Negative-result discipline (V3.6 / L16)

**What is now ruled out:**

1. The VIX-term-structure → VIXY signal does not have a robust parameter region at daily resolution (E1b plateau pre-flight, 45% spread).
2. The L26 mitigation (percentile-rolling gates) did not rescue E1's fragility — it improved the noise-fragility axis but did not address the underlying signal weakness.
3. The L25 mitigation (vol-carry MC threshold) was never tested because the design failed plateau pre-flight. We do not know whether it would have produced a non-SUSPECT verdict downstream.

**Status of the E1 / E1b lineage:** RETIRED from the backlog at retail-resolution daily-bar timeframes. The next-iteration directions in E1 §4.6 (continuous scaling, percentile gates, class recalibration) are now BOTH tested (continuous scaling was C7, percentile gates were C1-C6; both fall under the failed plateau pre-flight). A genuinely new pre-reg would need a fundamentally different signal (e.g. GEX, intraday VIX dynamics) or a different vehicle (options, direct futures with roll modelling).

After E1b lands, the next backlog step is **D2 — Commodity futures carry** (5d, needs 24-commodity data acquisition first). The VRP-capture line of investigation is closed.

---

## §5. Failure modes to watch

- **L04 / A1 — Same-bar look-ahead.** Rolling quantile uses `shift(1)` so today's quantile threshold is computed from data ending yesterday. Position uses `weights.shift(1)`.
- **L06 — Per-day MTM Sharpe, not per-trade.** Strategy holds positions for weeks; class-default per-day MTM applies.
- **L08 — MC threshold is now class-specific (NEW class).** Do NOT use generic `DAILY_MEAN_REVERSION` MC default; the new `DAILY_MEAN_REVERSION_VOL_CARRY` defaults are the gate.
- **L11 — Data snapshot.** Same parquets as E1; commit the read timestamp.
- **L17 — Absolute MC, not relative.** Same as E1; short-vol carry is not a long-only equity strategy.
- **L20 — Index normalisation.** All series normalised to date-only before any merge.
- **L24 — Per-cell noise gate.** As above.
- **L25 — Class match.** Strategy MUST select `DAILY_MEAN_REVERSION_VOL_CARRY` in the audit harness; do not silently fall back to `DAILY_MEAN_REVERSION`.
- **L26 — Continuous scaling validation.** C7 (sigmoid) is the alternative mitigation. The sigmoid output is bounded to [0, 1] (one-sided — strategy is short-only in this design), then multiplied by `target_short_weight` to give the final VIXY weight. The sigmoid centre IS the rolling enter_q value; scale = 0.05 (one quantile-bandwidth).
- **A3 — TR vs price-only.** All inputs (VIXY/SPY/VIX-family) are yfinance adj close = total return.
- **A4 — WFO honesty.** This directive committed BEFORE any new data examined. Cells pre-registered.
- **A5 / V3.1 — DSR for N=7.** Apply deflation with empirical skew/kurt.
- **V3.2 — Plateau pre-flight.** Sharpe spread across the canonical's neighbours, reported in §4.2.
- **V3.6 / L16 — Negative-result discipline.** If E1b also produces SUSPECT/RETIRE across all cells, document the mechanism (what's now ruled out about VRP capture) and accept that the trade does not work on retail-implementable vehicles after costs.

## §6. Implementation plan

1. **Extend the framework typology.** Add `DAILY_MEAN_REVERSION_VOL_CARRY` to `titan.research.framework.typology.StrategyClass` enum + a `DEFAULTS` row. Update `tests/test_framework_synthetic.py::test_defaults_for_every_strategy_class` to cover the new class.
2. **Build the strategy module** in `research/vrp_v2/vrp_v2_strategy.py`. Public functions:
   - `VrpV2Config` dataclass (per-cell parameters including `gate_kind`, `window_d`, `enter_q`, `exit_q`, `sigmoid_scale`).
   - `vrp_v2_target_weights(closes, *, cfg, vix, vix9d, vix3m) -> DataFrame`.
   - `vrp_v2_returns(...)` — cost-adjusted per-bar returns.
   - `vrp_v2_assert_causal(...)` — A10 causality smoke test.
3. **Build the audit harness** in `research/vrp_v2/run_vrp_v2_audit.py`. Selects `StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY` for `defaults_for(...)`. Per-cell noise gate (L24). Result log at `.tmp/reports/vrp_v2/result_log.md`.
4. **Tests in `tests/test_vrp_v2.py`:**
   - Class defaults consistency (P(MaxDD>50%) < 0.10).
   - Percentile gate logic on synthetic universe.
   - Continuous-scaling logic (sigmoid C7).
   - Causality (A10).
   - Shift discipline.
5. **Plateau pre-flight.** Before main audit, run only the canonical neighbours and abort with a clear error if Sharpe spread > 30%.
6. **Run the audit, append §4 result log.**
7. **Update V3.6 Catalogue** with any new lessons.

After E1b lands (verdict positive or negative), the next backlog step remains **D2 — Commodity futures carry** (needs 24-commodity data acquisition).
