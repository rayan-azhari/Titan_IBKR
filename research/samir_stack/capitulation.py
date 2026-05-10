"""Capitulation overlay — opportunistic re-entry near crash bottoms.

The base regime gate is structurally LATE on re-entry: it requires a
20-bar quiet period at score >= 0.50. By the time that fires, recovery
rallies (which cluster with worst days) are already mostly over.

This overlay adds a contrarian re-entry path that fires when:
  1. A capitulation event has registered in the last 60 bars
     (extreme VIX z-score, deep 21d drawdown, or credit-spread blowout)
  2. AND a stabilisation signal is currently active
     (5d bounce, vol mean-reverting, or rv_regime recovering)

When both conditions hit while the strategy is in cash, it enters at
TIER 1 (1x leverage at normal sleeve weight) — asymmetric sizing keeps
risk contained vs. a regime-confirmed full-tier-3 entry.

A failed-bounce stop (-5% within 10 bars from entry) ejects false dawns
before they cause material damage. Once regime score recovers to >= 0.50,
the strategy graduates back to normal tier logic.

Design intent
-------------
* No change to defensive logic (when to EXIT) — the existing regime gate
  + DD circuit breaker handle that.
* Strict additive — this overlay only adds new entry paths.
* RoR-preserving — asymmetric sizing + failed-bounce stop limit damage
  from false signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CapitulationConfig:
    """Configuration for the capitulation overlay."""

    enabled: bool = False
    """Master switch. False = behaviour identical to v1."""

    # Capitulation detection — at least N indicators must hit extreme low
    capitulation_lookback: int = 60
    """How many bars to look back for a recent capitulation event."""

    vix_score_capitulation_threshold: float = 0.05
    """VIX score below this counts as capitulation. 0.05 ≈ VIX z-score
    in the most extreme 5% of history ≈ VIX > 35-40 in normal regimes."""

    dd_velocity_capitulation_threshold: float = 0.05
    """dd_velocity score below this = SPY 21d return ≤ approximately -10%."""

    credit_capitulation_threshold: float = 0.05
    """Credit-spread score below this = HYG/IEF z-score in extreme 5%
    of distribution (deep credit panic)."""

    min_capitulation_indicators: int = 2
    """How many indicators must hit extreme low simultaneously within the
    lookback window. 2 = stricter (only true crises); 1 = looser (any
    single shock)."""

    spy_dd_required: float = 0.12
    """SPY must be at least this far below its 60-bar high for the
    overlay to consider firing. Filters routine vol spikes from real
    crashes. The point is to detect that the asset itself has crashed
    (regardless of whether our portfolio took the hit — the regime gate
    may have correctly avoided most damage).

    OPTIMIZED 2026-05-02 from sweep: 0.12 sits in the stable plateau
    [0.12, 0.22]. Below 0.12 fires too often; above adds nothing."""

    spy_dd_lookback: int = 60
    """How many bars to look back for SPY's recent high."""

    # Stabilisation detection — must be confirmed with a minimum quality
    bounce_5d_threshold: float = 0.03
    """SPY 5-day return >= +3% = bounce confirmation.

    OPTIMIZED 2026-05-02 from sweep: 0.03 captures the COVID V-rebound
    earlier than 0.05; both 0.02 and 0.03 form a stable plateau."""

    vix_recovery_fraction: float = 0.50
    """VIX score has improved by this fraction of its recent low's
    distance to 1.0 (50% recovery from extreme = meaningful de-stress)."""

    rv_regime_recovery_threshold: float = 0.50
    """rv_regime score recovering above this level after a low excursion
    (vol mean-reverting toward normal)."""

    # Sizing on opportunistic entry
    opportunistic_tier: float = 2.0
    """Tier to enter at when capitulation overlay fires. 2.0 = 2x equity,
    so 40% sleeve x 2x = 80% notional effective S&P exposure.

    OPTIMIZED 2026-05-02 from sweep: tier=2 doubles COVID rebound capture
    (+10.5pp -> +23.5pp) while the failed-bounce stop preserves RoR
    (P(DD>50%) actually IMPROVES vs v1: 0.40% -> 0.16%). Tier=3 explodes
    GFC false-bounce damage (-32% vs -2.4% at tier=2). Tier=2 is the
    balanced optimum."""

    # Failed-bounce stop
    failed_bounce_drawdown: float = 0.05
    """If equity drops by this fraction from opportunistic entry within
    failed_bounce_lookback bars, exit (false dawn)."""

    failed_bounce_lookback: int = 10

    # Graduation
    graduation_score: float = 0.50
    """Once regime score reaches this, exit opportunistic state and
    revert to normal tier logic at whatever tier the score supports."""


@dataclass
class CapitulationState:
    """Mutable state for the capitulation overlay."""

    active: bool = False
    """True when the strategy is in opportunistic re-entry mode."""

    entry_equity: float = 1.0
    """Portfolio equity level at the opportunistic entry — used for
    failed-bounce stop computation."""

    bars_since_entry: int = 0
    """Counter for the failed-bounce window."""


def detect_capitulation(
    panel: pd.DataFrame,
    cfg: CapitulationConfig,
) -> pd.Series:
    """Returns a boolean Series — True if at least ``min_capitulation_indicators``
    capitulation signals fired within the last ``capitulation_lookback`` bars.

    Uses the indicator-score panel (each column in [0, 1] with 0=hostile,
    1=benign). Capitulation = score reached an extreme low recently.
    Requiring multiple indicators to agree filters routine vol spikes
    from genuine crises.
    """
    if not cfg.enabled:
        return pd.Series(False, index=panel.index)

    n = cfg.capitulation_lookback
    flags = pd.DataFrame(0, index=panel.index, columns=["vix", "dd_velocity", "credit"])

    if "vix" in panel.columns:
        vix_min = panel["vix"].rolling(n, min_periods=1).min()
        flags["vix"] = (vix_min < cfg.vix_score_capitulation_threshold).astype(int)
    if "dd_velocity" in panel.columns:
        dd_min = panel["dd_velocity"].rolling(n, min_periods=1).min()
        flags["dd_velocity"] = (dd_min < cfg.dd_velocity_capitulation_threshold).astype(int)
    if "credit" in panel.columns:
        credit_min = panel["credit"].rolling(n, min_periods=1).min()
        flags["credit"] = (credit_min < cfg.credit_capitulation_threshold).astype(int)

    n_fired = flags.sum(axis=1)
    return (n_fired >= cfg.min_capitulation_indicators).fillna(False)


def detect_stabilisation(
    panel: pd.DataFrame,
    spy_close: pd.Series,
    cfg: CapitulationConfig,
) -> pd.Series:
    """Returns a boolean Series — True if a stabilisation signal is
    currently active (any of: 5-day bounce, VIX recovering, rv_regime
    recovering).
    """
    if not cfg.enabled:
        return pd.Series(False, index=panel.index)

    common = panel.index.intersection(spy_close.index)
    out = pd.Series(False, index=panel.index)

    # 5-day bounce
    spy_5d_ret = spy_close.pct_change(5).reindex(common)
    bounce = (spy_5d_ret >= cfg.bounce_5d_threshold).reindex(panel.index, fill_value=False)
    out |= bounce

    # VIX recovering: the score has come back from the recent low
    if "vix" in panel.columns:
        n = cfg.capitulation_lookback
        vix_min = panel["vix"].rolling(n, min_periods=1).min()
        # Recovery threshold = vix_min + frac * (1 - vix_min)
        recovery_target = vix_min + cfg.vix_recovery_fraction * (1.0 - vix_min)
        vix_recovering = (panel["vix"] >= recovery_target) & (vix_min < 0.20)
        out |= vix_recovering.fillna(False)

    # rv_regime recovering above threshold
    if "rv_regime" in panel.columns:
        n = cfg.capitulation_lookback
        rv_min = panel["rv_regime"].rolling(n, min_periods=1).min()
        rv_recovering = (panel["rv_regime"] >= cfg.rv_regime_recovery_threshold) & (rv_min < 0.20)
        out |= rv_recovering.fillna(False)

    return out.fillna(False)


def opportunistic_entry_signal(
    panel: pd.DataFrame,
    spy_close: pd.Series,
    cfg: CapitulationConfig,
) -> pd.Series:
    """Combined entry signal: capitulation has registered AND stabilisation
    is currently active. The strategy should enter the opportunistic tier
    when this is True AND it is currently in cash.
    """
    if not cfg.enabled:
        return pd.Series(False, index=panel.index)

    cap = detect_capitulation(panel, cfg)
    stab = detect_stabilisation(panel, spy_close, cfg)
    return cap & stab
