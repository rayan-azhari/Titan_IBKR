"""signals.py — Anchored VWAP, empirical percentile levels, ATR levels.

All functions are pure (no VectorBT dependency).  Every series returned is
already shifted by 1 bar to prevent look-ahead bias — callers must NOT add
an additional shift.

Session anchors (UTC):
  London open : 07:00
  NY open     : 13:00
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.strategies.ml.features import atr

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

LONDON_HOUR = 7  # 07:00 UTC
NY_HOUR = 13  # 13:00 UTC
US_OPEN_HOUR = 14  # 14:00 UTC — nearest H1 bar to NYSE open (09:30 ET)


def _utc_hours(index: pd.DatetimeIndex) -> pd.Series:
    if index.tzinfo is not None:
        return pd.Series(index.tz_convert("UTC").hour, index=index)
    return pd.Series(index.hour, index=index)


def session_group(index: pd.DatetimeIndex, anchor_sessions: list[str]) -> pd.Series:
    """Assign a monotonically increasing group ID that resets at each anchor.

    Each new session start increments the group counter so cumsum operations
    inside a group stay anchored to the most recent session open.

    Args:
        index: DatetimeIndex of the OHLCV DataFrame.
        anchor_sessions: Sessions to anchor at — any subset of ["london", "ny"].

    Returns:
        Integer Series (group IDs aligned to index).
    """
    hours = _utc_hours(index)
    is_anchor = pd.Series(False, index=index)
    if "london" in anchor_sessions:
        is_anchor |= hours == LONDON_HOUR
    if "ny" in anchor_sessions:
        is_anchor |= hours == NY_HOUR
    if "us" in anchor_sessions:
        is_anchor |= hours == US_OPEN_HOUR  # NYSE open anchor
    return is_anchor.cumsum()


def session_mask(index: pd.DatetimeIndex, sessions: list[str]) -> pd.Series:
    """Boolean mask: True for bars within active trading sessions.

    Args:
        index: DatetimeIndex of the OHLCV DataFrame.
        sessions: Active sessions — any subset of ["london", "ny"].

    Returns:
        Boolean Series aligned to index.
    """
    hours = _utc_hours(index)
    mask = pd.Series(False, index=index)
    if "london_core" in sessions:
        mask |= (hours >= 7) & (hours < 12)  # 07:00-12:00 UTC
    elif "london" in sessions:
        mask |= (hours >= 7) & (hours < 16)
    if "ny" in sessions:
        mask |= (hours >= 13) & (hours < 21)
    if "us_regular" in sessions:
        mask |= (hours >= 14) & (hours < 21)  # 09:30-16:00 ET = 14:30-21:00 UTC
    if "us_open" in sessions:
        mask |= (hours >= 14) & (hours < 17)  # first 3 hours: highest MR tendency
    return mask


# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------


def compute_anchored_vwap(
    df: pd.DataFrame,
    anchor_sessions: list[str] | None = None,
) -> pd.Series:
    """Compute session-anchored VWAP.

    Resets the cumulative sum at each anchor session open.  The result is
    shifted by 1 bar so bar N only sees information from bars 0..N-1.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        anchor_sessions: Sessions to anchor at. Defaults to ["london", "ny"].

    Returns:
        VWAP Series (shift-1 applied, look-ahead safe).
    """
    if anchor_sessions is None:
        anchor_sessions = ["london", "ny"]

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"].astype(float)

    grp = session_group(df.index, anchor_sessions)

    cum_tp_vol = tp_vol.groupby(grp).cumsum()
    cum_vol = df["volume"].astype(float).groupby(grp).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.shift(1)


def compute_deviation(df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
    """Price deviation from anchored VWAP (in price units, not pips).

    Args:
        df: OHLCV DataFrame.
        vwap: Anchored VWAP series (already shift-1).

    Returns:
        deviation = close - vwap
    """
    return df["close"] - vwap


# ---------------------------------------------------------------------------
# Entry levels — Option A: Empirical percentiles
# ---------------------------------------------------------------------------


def percentile_levels(
    deviation: pd.Series,
    window: int,
    pcts: list[float],
) -> pd.DataFrame:
    """Rolling empirical percentile bands of the VWAP deviation distribution.

    These define the SHORT entry tiers.  For LONG entries use the negatives.
    Already shifted by 1 bar — do NOT shift again.

    Args:
        deviation: close - vwap series.
        window: Rolling lookback in bars.
        pcts: Upper-tail percentiles, e.g. [0.90, 0.95, 0.98, 0.99, 0.999].

    Returns:
        DataFrame with columns labelled by percentile value.
    """
    cols = {}
    for p in pcts:
        cols[p] = deviation.rolling(window).quantile(p).shift(1)
    return pd.DataFrame(cols, index=deviation.index)


def atr_percentile_gate(
    df: pd.DataFrame,
    atr_period: int = 14,
    gate_window: int = 500,
    gate_pct: float = 0.30,
) -> pd.Series:
    """Regime gate based on ATR percentile.

    Allow entries only when ATR(atr_period) is below its rolling gate_pct
    percentile — i.e., when the market is in a low-volatility (ranging) state.
    High-ATR periods are typically trending and eat mean-reversion P&L.

    Replaces the HMM gate: simpler, causal, no training required.

    Args:
        df: OHLCV DataFrame.
        atr_period: ATR lookback period.
        gate_window: Rolling window for the ATR percentile reference.
        gate_pct: Only trade when ATR < this percentile of its own history.

    Returns:
        Boolean Series (shift-1 applied — look-ahead safe).
    """
    a = atr(df, atr_period)
    threshold = a.rolling(gate_window).quantile(gate_pct)
    return (a < threshold).shift(1).fillna(False)


def invalidation_level(
    deviation: pd.Series,
    window: int,
    pct: float = 0.9999,
) -> pd.Series:
    """Hard invalidation threshold (99.99th percentile by default).

    When |deviation| exceeds this, the mean-reverting premise is broken and
    the entire basket should be closed immediately.

    Args:
        deviation: close - vwap series.
        window: Rolling lookback in bars.
        pct: Percentile for the upper tail. Default 0.9999.

    Returns:
        Series of threshold values (shift-1, look-ahead safe).
    """
    return deviation.rolling(window).quantile(pct).shift(1)


# ---------------------------------------------------------------------------
# Entry levels — Option B: ATR multiples
# ---------------------------------------------------------------------------


def atr_levels(
    df: pd.DataFrame,
    vwap: pd.Series,
    atr_period: int,
    mults: list[float],
) -> pd.DataFrame:
    """ATR-multiple bands around the anchored VWAP.

    Already shift-1 via atr() internals + explicit shift on vwap.

    Args:
        df: OHLCV DataFrame.
        vwap: Anchored VWAP (shift-1 already applied).
        atr_period: ATR lookback period.
        mults: ATR multipliers for each tier, e.g. [1.0, 1.5, 2.0, 2.5, 3.0].

    Returns:
        DataFrame with columns labelled by multiplier value.
    """
    a = atr(df, atr_period).shift(1)
    cols = {}
    for m in mults:
        cols[m] = vwap + m * a
    return pd.DataFrame(cols, index=df.index)
