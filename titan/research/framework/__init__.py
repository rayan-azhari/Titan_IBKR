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
    decide,
)
from titan.research.framework.dsr import DsrResult, deflated_sharpe, sr_var_from_sweep
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
    # DSR
    "DsrResult",
    "deflated_sharpe",
    "sr_var_from_sweep",
    # Decision
    "Verdict",
    "DecisionInputs",
    "DecisionResult",
    "GateThresholds",
    "decide",
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
