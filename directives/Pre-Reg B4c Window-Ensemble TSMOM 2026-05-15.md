# Pre-Registration — B4c: Window-Ensemble TSMOM

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** L43 ensemble-mitigation follow-up to B4b.
**Predecessors:** Pre-Reg B4 (yfinance, RETIRED), Pre-Reg D2b B4b IBKR Roll-Stitched 2026-05-15 (B4b: H1-SUPPORTED for L40, RETIRED at plateau pre-flight with knife-edge `P_window_9=+0.45` vs `C1_canonical=+1.63`). L43 is the lesson: knife-edge plateau means the parameterisation is brittle even if the thesis is right.
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE the audit examines new data. The hypothesis being tested is **L43's mitigation claim** — that a window-ensemble TSMOM dissolves the brittle knife-edge that killed B4b at plateau, while preserving the L40-cleanup Sharpe lift.

---

## §1. Motivation & mechanism

**L43 in plain language.** B4b proved L40 was real (clean roll-stitched data lifted TSMOM Sharpe 10x vs yfinance). But it also showed the strategy at canonical settings (12-month window, 1-month skip) is plateau-fragile: a small lookback perturbation (12 → 9 months) collapses Sharpe from +1.63 to +0.45. The 5-axis matrix correctly RETIRED this — deploying a knife-edge would be a parameter overfit. The natural mitigation is to STOP relying on a single lookback. If we combine multiple windows, the signal is no longer pinned to a specific window's regime fit; it averages across them.

**Mechanism.** Per asset i, per rebalance date t, instead of computing a single 12-month sign signal, compute:

```
signal_i(t) = aggregate over W in WINDOW_SET of:
                sign( cum_log_return_W(asset_i, ending at t-skip) )
```

Two aggregation methods to test:

- **`vote`** — `signal_i(t) = mean of sign() values across W`. Always in [-1, +1]. A 3-window ensemble with all three agreeing gives ±1.0; mixed signals give 0 or ±0.33 or ±0.67.
- **`weighted_sum`** — `signal_i(t) = mean of sign(W) × |z(cum_ret_W)|` where z is the cross-asset z-score over the same date. Heavier weight to windows where the signal is strong.

`vote` is the simpler, more interpretable default. `weighted_sum` is the alternative with one additional knob.

The portfolio construction layer (inverse-vol sizing, target-vol overlay, costs) is unchanged from B4b. **Only the signal is ensembled.**

**Why this dissolves the knife-edge.** B4b's P_window_9 was bad because the 9-month window happened to miss a 2024 trend. P_window_12 captured it. The ensemble holds the strategy at "mostly long the assets that trended over 9 AND 12 AND 15 months" — robust to which specific window was favourable in this regime.

## §2. Universe + audit configurations

**Universe.** Same 24 commodities as B4b. Same stitched M1 inputs (`data/{ROOT}_M1_stitched_D.parquet`, back-adjusted).

**Date range.** Same as B4b: 2023-05-12 → 2026-05-15 (~3 years, L41 IBKR depth ceiling).

**WFO override (L25 transparency).** Same as B4b: 1.5y IS / 0.5y OOS / fold_count=5, expanding. With 3 years of data, this yields ~2 effective folds. Statistical power is limited (a known cost of L41); the audit's job is to show the ENSEMBLE makes the per-cell Sharpe robust to plateau perturbations, not to produce a higher Sharpe than B4b at canonical.

**Pre-registered cells (V3.1).** All cells inherit B4b's `signal_mode="sign"`, `skip_months=1` (canonical). The new knob is `window_set`:

| Cell                | window_set       | aggregation     | notes                                                  |
|---------------------|------------------|-----------------|--------------------------------------------------------|
| **C1_canonical**    | (9, 12, 15)      | vote            | The L43-suggested 3-window blend                       |
| C2_pair             | (12, 15)         | vote            | 2-window minimal blend                                 |
| C3_wide             | (6, 12, 18)      | vote            | Wider window spread                                    |
| C4_classic          | (3, 6, 12)       | vote            | Carver-style speed range                               |
| C5_weighted         | (9, 12, 15)      | weighted_sum    | C1 with magnitude weighting                            |
| C6_singleton_12     | (12,)            | vote            | Degenerate (= B4b C1_canonical); reference baseline    |
| C7_singleton_9      | (9,)             | vote            | Degenerate (= B4b P_window_9); the knife-edge baseline |
| C8_gross_no_costs   | (9, 12, 15)      | vote            | C1 with `apply_costs=False`                            |

C6 and C7 are baselines: they should reproduce B4b's C1_canonical (~+1.6) and P_window_9 (~+0.4) respectively. **If C6 and C7 do not reproduce B4b, the audit harness has a bug** — abort and debug.

**Plateau pre-flight neighbours (V3.2, L27 gate).** Around C1 = (9, 12, 15) vote:

| Neighbour            | What changes              |
|----------------------|---------------------------|
| P_shift_short        | (6, 9, 12) vote           |
| P_shift_long         | (12, 15, 18) vote         |
| P_drop_short         | (12, 15) vote             |
| P_drop_long          | (9, 12) vote              |

If C1's Sharpe is +1.0 and neighbours' span [+0.9, +1.1], that's plateau-robust (L43 mitigation works). If neighbours span [+0.4, +1.6], that's still knife-edge — ensembling didn't help.

**Falsification hypotheses (pre-committed, V3.1).**

- **B4c H1 (L43 mitigation).** "The window-ensemble dissolves the plateau knife-edge." Falsifiable: if C1's relative plateau spread ≥ 30% (L27 gate fails), the ensemble did NOT mitigate. Subtler test: if C1's plateau spread is materially lower than B4b's 72.48%, partial support.
- **B4c H2 (Sharpe preservation).** "The ensemble preserves at least 70% of B4b's canonical Sharpe (+1.63)." Falsifiable: C1 Sharpe < +1.14 means the ensemble averaged AWAY the edge instead of stabilising it.

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds (5-axis matrix, L24):** standard `CROSS_ASSET_MOMENTUM` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (verdict = DEPLOY, OR CONDITIONAL_WATCHPOINT with noise axis = `best`), pick the one with the highest CI_lo. Excludes C6_singleton_12, C7_singleton_9, C8_gross_no_costs from deployment (they are baselines / cost reference).

**Plateau pre-flight (L27).** C1 + 4 neighbours above. Relative spread < 30% to proceed. **If C1 fails plateau but H1 still shows clear improvement over B4b's spread, document the partial-mitigation finding and add to L43.**

**Sanctuary:** same convention as B4b — last 6 months held out (~124 bars).

**L43 carry-over.** If the audit promotes a window-ensemble cell, the ensemble approach gets generalised: every future TSMOM/EWMAC pre-reg defaults to multi-speed/multi-window unless single-window is justified.

## §4. Result log

### §4.1 Baseline reproduction (C6, C7)

| Cell                | B4c Sharpe   | B4b reference | Match? |
|---------------------|-------------:|--------------:|:------:|
| C6_singleton_12     | +1.6309      | +1.6309       | ✓ exact |
| C7_singleton_9      | +0.4488      | +0.4488       | ✓ exact |

**Audit harness integrity: VALIDATED.** Singleton-tuple windows reproduce B4b's single-int results bit-exactly, confirming `_normalize_windows` and the aggregation logic are correct when `window_set = (W,)`.

### §4.2 Plateau pre-flight + per-cell sweep

| Cell                 | Window set        | Sharpe   |
|----------------------|-------------------|---------:|
| **C1_canonical**     | (9, 12, 15) vote  | +1.4328  |
| P_shift_short        | (6, 9, 12) vote   | +0.8819  |
| P_shift_long         | (12, 15, 18) vote | +1.5942  |
| P_drop_short         | (12, 15) vote     | +1.6937  |
| P_drop_long          | (9, 12) vote      | +1.1854  |

**Relative plateau spread: 47.93%** (B4b reference: 72.48%). L27 30% gate FAILS — audit aborts before 5-axis matrix.

### §4.3 H1 verdict — PARTIALLY SUPPORTED

> "The window-ensemble dissolves the plateau knife-edge."

- B4b plateau spread: **72.48%**
- B4c plateau spread: **47.93%**
- Mitigation: **−33.9%** (reduction in spread)

The window-ensemble materially reduces brittleness but does NOT fully dissolve it under the L41 short-window constraint. The remaining fragility traces to short-end windows: P_shift_short (6, 9, 12) = +0.88 is the worst neighbour, while P_drop_short (12, 15) = +1.69 is the best. The 6-month and 9-month lookbacks individually pull the ensemble Sharpe down; dropping them lifts it. **Interpretation:** in the 2023-2025 regime, short-horizon momentum on this 24-commodity universe was structurally weaker than 12-18 month momentum. Ensembling can dampen but not erase this asymmetry.

### §4.4 H2 verdict — SUPPORTED

> "The ensemble preserves at least 70% of B4b's canonical Sharpe (+1.63)."

- B4b canonical: +1.6309
- B4c C1_canonical (9, 12, 15) vote: **+1.4328**
- Preservation: **87.9%** (well above 70% threshold)

The ensemble retains the L40 roll-stitching lift; the 12% Sharpe loss is the price of robustness (volatility-of-Sharpe-across-cells dropped from B4b's 72.48% to 47.93% spread). The trade-off is real but favourable.

### §4.5 Recommendation + new lessons

**Decision (per pre-reg §3 + L27).** Plateau gate failed (47.93% > 30%). No cell qualifies for promotion to deployment per the standard 5-axis matrix.

**However**, the audit produced two valuable research findings:
1. **L43 mitigation is real but partial.** Ensembling 3 nearby windows reduces plateau brittleness ~34%, not the 100% needed to pass L27. This is a meaningful but insufficient mitigation.
2. **Regime fingerprint.** The remaining plateau fragility is concentrated at the short-window end (6-9 months). Future TSMOM/EWMAC pre-regs on similar data should either:
   - Use only 12+ month lookbacks, OR
   - Add a regime-aware overlay that down-weights short-window signals in regimes where they are unstable.

**New lesson: L45 — Plateau-mitigation ensembles are partial fixes; report mitigation pct and abort if still > gate.** Detail below in V3.6 catalogue.

**Backlog next step.** With both B4b and B4c plateau-failing, single-window AND window-ensemble TSMOM on the IBKR 3y window are both retired-with-research-alive. The Carver-style **EWMAC ensemble (B2)** is the natural escalation — it averages signals across many SPEEDS (4-8 EWMA-pair speeds), is the canonical robust trend-following construction, and explicitly handles the short-window fragility B4c surfaced. Proceed to B2.

---

## §5. Failure modes to watch

- **L04 / A1 (causality).** All windows must `.shift(1)` properly. Causality smoke test covers it.
- **L23 / A5 (DSR).** 8 cells. DSR at N=8 trials applied.
- **L25 (class override transparency).** WFO override is L41 contingency, not L43 cheat.
- **L27 (plateau pre-flight).** The whole audit's purpose is to test L27 robustness.
- **L37 (TSMOM persistence).** Already partially confirmed by B4b. Reconfirm at C6 baseline.
- **L40 (yfinance roll contamination).** Already fixed by using stitched M1. No re-litigation.
- **L43 (knife-edge plateau).** **The thing being tested.**
- **L44 (back-adjust bias in carry signals).** Not applicable — TSMOM uses single-series cumulative return, back-adjustment is the CORRECT input.
- **A6 (MC bootstrap).** Class default, same as B4b.

## §6. Implementation plan

1. **Extend `TsmomConfig`** in `research/tsmom/tsmom_strategy.py` with `momentum_window_months: int | tuple[int, ...]` and `ensemble_aggregation: Literal["vote", "weighted_sum"]`. Keep the existing single-int form for B4b compat.
2. **Refactor `compute_tsmom_signal`** to handle the tuple case: compute sign per window, aggregate.
3. **Add unit tests** to `tests/test_tsmom.py`:
   - Singleton tuple `(12,)` == int `12` (B4b parity).
   - Vote average of 3 unanimous +1 signs == +1.
   - Vote of mixed signs averages correctly.
   - Weighted-sum with one strong window != equal-weight average.
4. **Write `research/tsmom/run_b4c_audit.py`** mirroring `run_b4b_audit.py` with the new cell set + plateau neighbours.
5. **Run audit.** Report in result log.
6. **If C1 promotes:** add to recommended-next ports list. If C1 fails plateau but shows mitigation, document as L43 refinement.
