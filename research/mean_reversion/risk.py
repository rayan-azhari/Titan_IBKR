"""risk.py — Hard invalidation and time-based exit signals.

Two circuit-breaker mechanisms:
  1. Hard invalidation: |deviation| exceeds 99.99th percentile threshold.
     The mean-reverting premise has broken down (e.g., NFP surprise, flash
     crash).  Entire basket is closed immediately via market order.
  2. NY session close: All positions closed at 21:00 UTC to avoid rollover
     spread widening and swap fees.  Intraday MR strategies should never
     hold risk through the overnight gap.

Both return boolean Series that are OR'd with the basket TP exit in
combined_exit = basket_exit | invalidation_exit | time_exit.
"""

from __future__ import annotations

import pandas as pd


def hard_invalidation_signal(
    deviation: pd.Series,
    inv_level: pd.Series,
) -> pd.Series:
    """Fire True when |deviation| exceeds the hard invalidation level.

    The invalidation level is the 99.99th percentile of the rolling deviation
    distribution (shift-1 applied in signals.py).  When triggered, the entire
    basket should be closed with a market order — no TP hunting.

    Args:
        deviation: close - anchored_vwap.
        inv_level: Hard invalidation threshold (from signals.invalidation_level).

    Returns:
        Boolean Series — True means close everything immediately.
    """
    return deviation.abs() > inv_level.abs()


def ny_session_close_exit(
    index: pd.DatetimeIndex,
    cutoff_hour_utc: int = 21,
) -> pd.Series:
    """Fire True on the last bar of each NY session.

    Closes all positions before 21:00 UTC rollover to avoid:
      - Widened overnight spreads
      - Swap/financing charges on carried positions
      - Gap risk at the Asian open

    Fires on EVERY bar at the cutoff hour (not just once per session) to ensure
    positions are closed even if multiple bars land in that hour at a coarser
    granularity.

    Args:
        index: DatetimeIndex of the OHLCV DataFrame.
        cutoff_hour_utc: Hour (UTC) at which to force-close. Default 21.

    Returns:
        Boolean Series aligned to index.
    """
    if index.tzinfo is not None:
        hours = index.tz_convert("UTC").hour
    else:
        hours = index.hour
    return pd.Series(hours == cutoff_hour_utc, index=index, name="ny_close_exit")


def partial_reversion_exit_short(
    deviation: pd.Series,
    tier1_level: pd.Series,
    reversion_pct: float = 0.5,
) -> pd.Series:
    """Short TP: exit when deviation returns to reversion_pct of entry level.

    Short was entered at deviation > tier1_level.  Exit once it has fallen back
    to tier1_level * reversion_pct (e.g., 50% of the way back to VWAP).

    Args:
        deviation: close - anchored_vwap.
        tier1_level: Tier-1 percentile level Series (positive, shift-1 applied).
        reversion_pct: Fraction of tier1_level at which to take profit.

    Returns:
        Boolean Series — True means close short this bar.
    """
    return deviation < tier1_level.abs() * reversion_pct


def partial_reversion_exit_long(
    deviation: pd.Series,
    tier1_level: pd.Series,
    reversion_pct: float = 0.5,
) -> pd.Series:
    """Long TP: exit when deviation returns to reversion_pct of entry level.

    Long was entered at deviation < -tier1_level.  Exit once it has risen back
    to -tier1_level * reversion_pct (e.g., 50% of the way back to VWAP).

    Args:
        deviation: close - anchored_vwap.
        tier1_level: Tier-1 percentile level Series (positive, shift-1 applied).
        reversion_pct: Fraction of tier1_level at which to take profit.

    Returns:
        Boolean Series — True means close long this bar.
    """
    return deviation > -tier1_level.abs() * reversion_pct


def build_combined_exit(
    basket_exit: pd.Series,
    deviation: pd.Series,
    inv_level: pd.Series,
    index: pd.DatetimeIndex,
    tier1_level: pd.Series | None = None,
    reversion_pct: float = 0.5,
    cutoff_hour_utc: int = 21,
) -> tuple[pd.Series, pd.Series]:
    """Combine all exit triggers into direction-aware (long_exit, short_exit).

    long_exit  = basket_TP | partial_reversion_long  | invalidation | time_exit
    short_exit = basket_TP | partial_reversion_short | invalidation | time_exit

    Splitting by direction prevents a short-reversion signal from spuriously
    closing a long position (and vice versa).

    Args:
        basket_exit: Basket TP from execution.compute_basket_vwap_exit().
        deviation: close - anchored_vwap.
        inv_level: Hard invalidation level from signals.invalidation_level().
        index: DatetimeIndex of the DataFrame.
        tier1_level: Tier-1 percentile level (for partial reversion TP).
                     If None, partial reversion is skipped.
        reversion_pct: Target reversion fraction (default 0.5 = 50%).
        cutoff_hour_utc: NY session close hour (UTC).

    Returns:
        (long_combined_exit, short_combined_exit) — Boolean Series pair.
    """
    invalidation = hard_invalidation_signal(deviation, inv_level)
    time_exit    = ny_session_close_exit(index, cutoff_hour_utc)
    base         = basket_exit | invalidation | time_exit

    if tier1_level is not None:
        long_exit  = base | partial_reversion_exit_long(deviation, tier1_level, reversion_pct)
        short_exit = base | partial_reversion_exit_short(deviation, tier1_level, reversion_pct)
    else:
        long_exit  = base
        short_exit = base

    return long_exit, short_exit
