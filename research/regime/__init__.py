"""Per-asset regime detection — I1 HMM gate.

Pre-registered in ``directives/Pre-Reg I1 HMM Per-Asset Regime + EWMAC Gate
2026-05-16.md``.

A 2-state (or 3-state) Gaussian HMM is fitted per-asset on a configurable
training window, frozen, then Viterbi-decoded forward over the full
audit window. The "trend-friendly" state is identified IN-SAMPLE via
return-autocorrelation or vol heuristic, and that state-label is frozen
for OOS use.

L48/L49 escalation: B2c (broad-trend gate) and B2d (broad-vol gate) both
failed because the regime is per-asset, not universe-wide. I1 tests
whether per-asset granularity rescues B2.
"""

from research.regime.hmm_gate import (
    HMMGateConfig,
    PerAssetRegimeGateConfig,
    compute_per_asset_regime_gate,
    fit_one_hmm,
    identify_trend_state,
)

__all__ = [
    "HMMGateConfig",
    "PerAssetRegimeGateConfig",
    "compute_per_asset_regime_gate",
    "fit_one_hmm",
    "identify_trend_state",
]
