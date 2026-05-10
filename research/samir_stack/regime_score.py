"""Regime score combiner.

Combines the validated indicator panel into a single benign-probability
score in [0, 1]. Two methods:

* equal-weight: simple mean across available (non-NaN) indicators per bar
* vol-weighted: weight each indicator by its hostile/benign vol ratio
  (indicators that produce more volatility-discrimination get higher weight)

The combiner skips indicators on bars where they're NaN — this is critical
because the credit indicator only starts in 2008. Pre-2008 the ensemble
operates on the 5 trend/vol/momentum indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Indicators kept after IC validation (tlt_trend dropped: vol_ratio < 1)
DEFAULT_INDICATORS = [
    "vix",
    "rv_regime",
    "credit",
    "trend",
    "dd_velocity",
    "momentum_12_1",
]

# Production-grade indicators (with HMM enabled)
PRODUCTION_INDICATORS = DEFAULT_INDICATORS + ["hmm_risk"]

# Vol-ratio weights (from validate_indicators run; can be re-fit per WFO fold)
DEFAULT_VOL_RATIO_WEIGHTS = {
    "vix": 2.43,
    "rv_regime": 2.24,
    "credit": 1.93,
    "trend": 2.08,
    "dd_velocity": 2.20,
    "momentum_12_1": 2.00,
    "hmm_risk": 2.50,  # estimated from causal HMM vol ratio ~2.5x
}


def regime_score_equal(
    panel: pd.DataFrame, indicators: list[str] = DEFAULT_INDICATORS
) -> pd.Series:
    """Equal-weight mean of available indicators per bar.

    Bars where an indicator is NaN are dropped from that bar's mean
    (so the score is always over the indicators that *are* available).
    """
    cols = [c for c in indicators if c in panel.columns]
    sub = panel[cols]
    return sub.mean(axis=1, skipna=True).clip(0.0, 1.0).rename("regime_score")


def regime_score_weighted(
    panel: pd.DataFrame,
    indicators: list[str] = DEFAULT_INDICATORS,
    weights: dict[str, float] = DEFAULT_VOL_RATIO_WEIGHTS,
) -> pd.Series:
    """Weighted mean of available indicators per bar.

    Weights are renormalised per bar over the indicators that are
    non-NaN, so missing indicators don't bias the score downward.
    """
    cols = [c for c in indicators if c in panel.columns]
    sub = panel[cols].copy()
    w = pd.Series({c: weights.get(c, 1.0) for c in cols})

    mask = sub.notna().astype(float)
    weighted_sum = (sub.fillna(0.0) * w).sum(axis=1)
    weight_total = (mask * w).sum(axis=1).replace(0.0, np.nan)
    return (weighted_sum / weight_total).clip(0.0, 1.0).rename("regime_score")


def correlation_matrix(
    panel: pd.DataFrame, indicators: list[str] = DEFAULT_INDICATORS
) -> pd.DataFrame:
    """Spearman correlation of indicators on common-non-NaN bars."""
    cols = [c for c in indicators if c in panel.columns]
    sub = panel[cols].dropna()
    return sub.corr(method="spearman").round(3)
