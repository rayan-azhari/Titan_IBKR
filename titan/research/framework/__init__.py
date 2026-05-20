"""Unified research framework for the Titan-IBKR-Algo project.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md.

Every NEW audit + every RE-AUDIT uses these primitives. The framework
standardises:

    * Strategy-class typology (titan.research.framework.typology)
    * Walk-forward fold construction (titan.research.framework.wfo)
    * Sanctuary discipline (titan.research.framework.sanctuary)
    * Monte Carlo block bootstrap (titan.research.framework.mc)
    * Deflated Sharpe Ratio (titan.research.framework.dsr)
    * 4-axis decision matrix (titan.research.framework.decision)

The existing audit scripts in research/strategies/, research/cross_asset/,
research/ml/, research/orb/, etc. remain as historical records. They
should NOT be mass-refactored; instead they are RE-RUN under the
framework as part of Phase 2 of the methodology directive.
"""

from titan.research.framework.allocator_erc import ErcResult, compute_erc_weights
from titan.research.framework.amortised_mc import (
    InferFn,
    PrefitFn,
    run_block_mc_amortised,
)
from titan.research.framework.crisis_stress import (
    NAMED_CRISES,
    CrisisStressResult,
    CrisisWindowResult,
    run_crisis_stress,
)
from titan.research.framework.dashboard import (
    AuditResult,
    CellSummary,
    render_dashboard,
)
from titan.research.framework.decision import (
    DecisionInputs,
    DecisionResult,
    GateThresholds,
    Verdict,
    classify_axis_noise,
    decide,
)
from titan.research.framework.drift_cusum import CusumResult, run_cusum_drift
from titan.research.framework.dsr import DsrResult, deflated_sharpe, sr_var_from_sweep
from titan.research.framework.early_gate import (
    Pass1GateResult,
    format_gate_report,
    pass1_can_clear_any_cell,
    pass1_can_clear_ci_gate,
    pass1_can_clear_from_returns,
)
from titan.research.framework.fdm import (
    DEFAULT_FDM_CAP,
    FdmResult,
    fdm_from_uniform_correlation,
    forecast_diversification_multiplier,
)
from titan.research.framework.kelly import KellyFraction, compute_kelly_fraction
from titan.research.framework.mc import (
    McResult,
    RelativeMcResult,
    run_block_mc,
    run_relative_block_mc,
)
from titan.research.framework.robustness import (
    NoiseConfig,
    NoiseLevelResult,
    NoiseRobustnessResult,
    run_noise_robustness,
)
from titan.research.framework.ruin import (
    RuinAssessment,
    assess_joint_ruin,
    assess_strategy_ruin,
)
from titan.research.framework.sanctuary import (
    DivergenceTest,
    SanctuarySlice,
    sanctuary_divergence_test,
    slice_sanctuary,
)
from titan.research.framework.typology import (
    COST_CME_FUTURES_LIQUID,
    COST_FX_MAJOR,
    COST_IG_DFB_INDEX,
    COST_UCITS_ETF,
    COST_US_EQUITY_LARGE_CAP,
    COST_US_ETF_LIQUID,
    DEFAULTS,
    CostModel,
    McConfig,
    SharpeReporting,
    StrategyClass,
    StrategyClassDefaults,
    WfoConfig,
    defaults_for,
)
from titan.research.framework.wfo import Fold, build_folds, iter_folds

__all__ = [
    # V3.7 — Risk-of-ruin (L65) + Kelly (L67) + ERC + Crisis + Drift
    "RuinAssessment",
    "assess_strategy_ruin",
    "assess_joint_ruin",
    "KellyFraction",
    "compute_kelly_fraction",
    "ErcResult",
    "compute_erc_weights",
    "CrisisStressResult",
    "CrisisWindowResult",
    "NAMED_CRISES",
    "run_crisis_stress",
    "CusumResult",
    "run_cusum_drift",
    # Typology
    "StrategyClass",
    "StrategyClassDefaults",
    "WfoConfig",
    "McConfig",
    "SharpeReporting",
    "CostModel",
    "DEFAULTS",
    "defaults_for",
    "COST_CME_FUTURES_LIQUID",
    "COST_US_EQUITY_LARGE_CAP",
    "COST_US_ETF_LIQUID",
    "COST_UCITS_ETF",
    "COST_FX_MAJOR",
    "COST_IG_DFB_INDEX",
    # WFO
    "Fold",
    "build_folds",
    "iter_folds",
    # Sanctuary
    "SanctuarySlice",
    "slice_sanctuary",
    "DivergenceTest",
    "sanctuary_divergence_test",
    # MC
    "McResult",
    "run_block_mc",
    "RelativeMcResult",
    "run_relative_block_mc",
    # Amortised MC (IS-frozen model state cached across paths)
    "PrefitFn",
    "InferFn",
    "run_block_mc_amortised",
    # DSR
    "DsrResult",
    "deflated_sharpe",
    "sr_var_from_sweep",
    # FDM (Carver Forecast Diversification Multiplier; backlog J5)
    "FdmResult",
    "forecast_diversification_multiplier",
    "fdm_from_uniform_correlation",
    "DEFAULT_FDM_CAP",
    # Early gate (Pass-1-gates-Pass-2 speed-up)
    "Pass1GateResult",
    "pass1_can_clear_ci_gate",
    "pass1_can_clear_from_returns",
    "pass1_can_clear_any_cell",
    "format_gate_report",
    # Decision
    "Verdict",
    "DecisionInputs",
    "DecisionResult",
    "GateThresholds",
    "decide",
    "classify_axis_noise",
    # Robustness (noise-injection gate -- Varma)
    "NoiseConfig",
    "NoiseLevelResult",
    "NoiseRobustnessResult",
    "run_noise_robustness",
    # Dashboard
    "AuditResult",
    "CellSummary",
    "render_dashboard",
]
