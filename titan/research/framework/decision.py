"""5-axis decision-matrix template.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.6 + directives/Pre-Reg J3 Noise Robustness 5th Axis 2026-05-15.md.
Fixes gaps G1 (incomplete matrices), G2 (UNDETERMINED verdicts),
G4 (no template).

Every empirical audit outcome maps to one of 243 cells (3 levels × 5
axes), and every cell maps deterministically to a verdict in:

    DEPLOY                  -- all 5 axes at "best"
    CONDITIONAL_WATCHPOINT  -- 4 of 5 axes at "best" (worst on a single axis)
    TIER_UNCONFIRMED        -- 3 of 5 axes at "best"
    SUSPECT                 -- 2 of 5 axes at "best"
    RETIRE                  -- 0 or 1 axes at "best"

UNDETERMINED is impossible by construction -- the matrix is total.

The 5th axis (noise robustness) was added per J3 backlog: a strategy
that passes the other 4 axes but is fragile to small input-price noise
(Varma's noise-injection test) is `CONDITIONAL_WATCHPOINT`, not DEPLOY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Verdict(Enum):
    """Five-valued verdict returned by ``decide(DecisionInputs)``."""

    DEPLOY = "DEPLOY"
    CONDITIONAL_WATCHPOINT = "CONDITIONAL_WATCHPOINT"
    TIER_UNCONFIRMED = "TIER_UNCONFIRMED"
    SUSPECT = "SUSPECT"
    RETIRE = "RETIRE"


AxisLevel = Literal["best", "mid", "worst"]


@dataclass(frozen=True)
class GateThresholds:
    """Per-axis thresholds. Defaults are the framework's recommended values
    but each strategy class can override via a pre-registration directive.
    """

    # CI_lo axis: thresholds on the 95% bootstrap CI lower bound on
    # stitched OOS Sharpe.
    ci_lo_best: float = 0.0  # best: CI_lo > 0
    ci_lo_worst: float = -0.2  # worst: CI_lo < -0.2

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


def classify_axis_sanctuary(
    sanctuary_sharpe: float, thr: GateThresholds = GateThresholds()
) -> AxisLevel:
    if sanctuary_sharpe > thr.sanctuary_best:
        return "best"
    if sanctuary_sharpe > thr.sanctuary_worst:
        return "mid"
    return "worst"


def classify_axis_noise(noise_passes_mean: bool, noise_passes_worst: bool) -> AxisLevel:
    """Noise-injection robustness axis (J3).

    Inputs come from `run_noise_robustness(...)`:
        noise_passes_mean  -- degradation_mean < cfg.max_degradation at every level
        noise_passes_worst -- degradation_p5  < cfg.max_degradation at every level

    Truth table:
        worst_pass=True  -> best   (passes mean AND 5th-percentile gates)
        mean_pass=True   -> mid    (passes mean but some trial degraded too much)
        otherwise        -> worst  (fragile: mean degradation breaches threshold)

    Note: worst_pass=True implies mean_pass=True (deg_p5 >= deg_mean is generally
    false: p5 of Sharpe across trials is the WORST observed Sharpe, so
    degradation_p5 >= degradation_mean -- a strategy can pass mean but fail worst).
    """
    if noise_passes_worst:
        return "best"
    if noise_passes_mean:
        return "mid"
    return "worst"


@dataclass(frozen=True)
class DecisionInputs:
    """Per-cell inputs to the 5-axis classifier.

    The first 5 fields are the original 4-axis inputs (`pass_threshold_prob`
    is the class-specific gate for the MC axis, not a separate axis).
    The last 2 fields (J3) are the noise-injection robustness gate's
    binary outputs from `run_noise_robustness(...)`.
    """

    ci_lo: float
    dsr_prob: float
    p_maxdd_gt_threshold: float
    pass_threshold_prob: float
    sanctuary_sharpe: float
    # J3 — 5th axis (Varma noise-injection robustness):
    noise_passes_mean: bool
    noise_passes_worst: bool


@dataclass(frozen=True)
class DecisionResult:
    """The verdict + per-axis explainability fields for audit logs."""

    verdict: Verdict
    ci_lo_axis: AxisLevel
    dsr_axis: AxisLevel
    mc_axis: AxisLevel
    sanctuary_axis: AxisLevel
    noise_axis: AxisLevel
    n_axes_best: int
    rationale: str


def decide(
    inputs: DecisionInputs,
    *,
    thresholds: GateThresholds | None = None,
) -> DecisionResult:
    """Map the 5-axis input vector to one of the 5 verdicts deterministically.

    Verdict mapping (J3 — 5 axes):
        5 -> DEPLOY
        4 -> CONDITIONAL_WATCHPOINT
        3 -> TIER_UNCONFIRMED
        2 -> SUSPECT
        0,1 -> RETIRE

    The 0/1 collapse (vs. distinct buckets pre-J3) reflects that with 5 axes,
    "1 axis at best" is as fundamentally fragile as "0 axes at best".
    """
    thr = thresholds or GateThresholds()
    ci = classify_axis_ci_lo(inputs.ci_lo, thr)
    dsr = classify_axis_dsr(inputs.dsr_prob, thr)
    mc = classify_axis_mc(inputs.p_maxdd_gt_threshold, inputs.pass_threshold_prob, thr)
    sanc = classify_axis_sanctuary(inputs.sanctuary_sharpe, thr)
    noise = classify_axis_noise(inputs.noise_passes_mean, inputs.noise_passes_worst)
    n_best = sum(1 for a in (ci, dsr, mc, sanc, noise) if a == "best")

    verdict_by_n = {
        5: Verdict.DEPLOY,
        4: Verdict.CONDITIONAL_WATCHPOINT,
        3: Verdict.TIER_UNCONFIRMED,
        2: Verdict.SUSPECT,
        1: Verdict.RETIRE,
        0: Verdict.RETIRE,
    }
    verdict = verdict_by_n[n_best]
    axis_names = [
        ("CI_lo", ci, f"{inputs.ci_lo:.3f}"),
        ("DSR", dsr, f"{inputs.dsr_prob:.3f}"),
        (
            "MC",
            mc,
            f"P(MaxDD>X)={inputs.p_maxdd_gt_threshold:.3f}, "
            f"threshold={inputs.pass_threshold_prob:.3f}",
        ),
        ("Sanctuary", sanc, f"{inputs.sanctuary_sharpe:.3f}"),
        (
            "Noise",
            noise,
            f"mean_pass={inputs.noise_passes_mean}, worst_pass={inputs.noise_passes_worst}",
        ),
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
        ci_lo_axis=ci,
        dsr_axis=dsr,
        mc_axis=mc,
        sanctuary_axis=sanc,
        noise_axis=noise,
        n_axes_best=n_best,
        rationale=rationale,
    )
