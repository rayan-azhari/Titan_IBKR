"""4-axis decision-matrix template.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.6. Fixes gaps G1 (incomplete matrices), G2 (UNDETERMINED verdicts),
G4 (no template).

Every empirical audit outcome maps to one of 81 cells (3 levels × 4
axes), and every cell maps deterministically to a verdict in:

    DEPLOY                  -- all 4 axes at "best"
    CONDITIONAL_WATCHPOINT  -- 3 of 4 axes at "best" (worst on a single axis)
    TIER_UNCONFIRMED        -- 2 of 4 axes at "best"
    SUSPECT                 -- 1 of 4 axes at "best"
    RETIRE                  -- 0 axes at "best"

UNDETERMINED is impossible by construction -- the matrix is total.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Verdict(Enum):
    DEPLOY = "DEPLOY"
    CONDITIONAL_WATCHPOINT = "CONDITIONAL_WATCHPOINT"
    TIER_UNCONFIRMED = "TIER_UNCONFIRMED"
    SUSPECT = "SUSPECT"
    RETIRE = "RETIRE"


AxisLevel = Literal["best", "mid", "worst"]


@dataclass(frozen=True)
class GateThresholds:
    """Per-axis thresholds. Defaults are the framework's recommended values
    but each strategy class can override via a pre-registration directive."""

    # CI_lo axis: thresholds on the 95% bootstrap CI lower bound on
    # stitched OOS Sharpe.
    ci_lo_best: float = 0.0      # best: CI_lo > 0
    ci_lo_worst: float = -0.2    # worst: CI_lo < -0.2

    # DSR axis: deflated_sharpe.dsr_prob thresholds.
    dsr_best: float = 0.95
    dsr_worst: float = 0.50

    # MC axis: P(MaxDD > X) ratio to the pass threshold.
    # best = P <= pass_threshold; worst = P >= 2 * pass_threshold.
    # The pass_threshold is class-specific (from typology's McConfig).
    mc_worst_ratio: float = 2.0  # multiplier on the pass_threshold

    # Sanctuary axis: realised Sharpe on the held-out window.
    sanctuary_best: float = 0.0  # best: sanctuary Sharpe > 0
    sanctuary_worst: float = -0.3


def classify_axis_ci_lo(ci_lo: float, thr: GateThresholds = GateThresholds()) -> AxisLevel:
    if ci_lo > thr.ci_lo_best:
        return "best"
    if ci_lo > thr.ci_lo_worst:
        return "mid"
    return "worst"


def classify_axis_dsr(dsr_prob: float, thr: GateThresholds = GateThresholds()) -> AxisLevel:
    if dsr_prob >= thr.dsr_best:
        return "best"
    if dsr_prob >= thr.dsr_worst:
        return "mid"
    return "worst"


def classify_axis_mc(
    p_maxdd_gt_threshold: float,
    pass_threshold_prob: float,
    thr: GateThresholds = GateThresholds(),
) -> AxisLevel:
    if p_maxdd_gt_threshold <= pass_threshold_prob:
        return "best"
    if p_maxdd_gt_threshold <= pass_threshold_prob * thr.mc_worst_ratio:
        return "mid"
    return "worst"


def classify_axis_sanctuary(sanctuary_sharpe: float, thr: GateThresholds = GateThresholds()) -> AxisLevel:
    if sanctuary_sharpe > thr.sanctuary_best:
        return "best"
    if sanctuary_sharpe > thr.sanctuary_worst:
        return "mid"
    return "worst"


@dataclass(frozen=True)
class DecisionInputs:
    """Per-cell inputs to the 4-axis classifier."""

    ci_lo: float
    dsr_prob: float
    p_maxdd_gt_threshold: float
    pass_threshold_prob: float
    sanctuary_sharpe: float


@dataclass(frozen=True)
class DecisionResult:
    """The verdict + per-axis explainability fields for audit logs."""

    verdict: Verdict
    ci_lo_axis: AxisLevel
    dsr_axis: AxisLevel
    mc_axis: AxisLevel
    sanctuary_axis: AxisLevel
    n_axes_best: int
    rationale: str


def decide(
    inputs: DecisionInputs,
    *,
    thresholds: GateThresholds | None = None,
) -> DecisionResult:
    """Map the 4-axis input vector to one of the 5 verdicts deterministically."""
    thr = thresholds or GateThresholds()
    ci = classify_axis_ci_lo(inputs.ci_lo, thr)
    dsr = classify_axis_dsr(inputs.dsr_prob, thr)
    mc = classify_axis_mc(inputs.p_maxdd_gt_threshold, inputs.pass_threshold_prob, thr)
    sanc = classify_axis_sanctuary(inputs.sanctuary_sharpe, thr)
    n_best = sum(1 for a in (ci, dsr, mc, sanc) if a == "best")

    verdict_by_n = {
        4: Verdict.DEPLOY,
        3: Verdict.CONDITIONAL_WATCHPOINT,
        2: Verdict.TIER_UNCONFIRMED,
        1: Verdict.SUSPECT,
        0: Verdict.RETIRE,
    }
    verdict = verdict_by_n[n_best]
    axis_names = [
        ("CI_lo", ci, f"{inputs.ci_lo:.3f}"),
        ("DSR", dsr, f"{inputs.dsr_prob:.3f}"),
        ("MC", mc, f"P(MaxDD>X)={inputs.p_maxdd_gt_threshold:.3f}, threshold={inputs.pass_threshold_prob:.3f}"),
        ("Sanctuary", sanc, f"{inputs.sanctuary_sharpe:.3f}"),
    ]
    best_axes = [n for n, lvl, _ in axis_names if lvl == "best"]
    worst_axes = [n for n, lvl, _ in axis_names if lvl == "worst"]
    if best_axes:
        rationale_best = f"PASS: {', '.join(best_axes)}"
    else:
        rationale_best = "no axes PASS"
    if worst_axes:
        rationale_worst = f"FAIL: {', '.join(worst_axes)}"
    else:
        rationale_worst = "no axes FAIL"
    rationale = f"{verdict.value} | {rationale_best} | {rationale_worst}"
    return DecisionResult(
        verdict=verdict,
        ci_lo_axis=ci, dsr_axis=dsr, mc_axis=mc, sanctuary_axis=sanc,
        n_axes_best=n_best, rationale=rationale,
    )
