"""Causal swing-point detection.

A swing high at bar i with lookback N is defined as:
    high[i] > high[i-1..i-N] AND high[i] > high[i+1..i+N]

The point is *confirmed* only at the close of bar i+N (we need N bars after
i to know it was a real swing). For trading purposes the swing's existence
is therefore knowable at i+N, not i. Code that uses swing[i] at bar i (or
even at bar i+1..i+N-1) is look-ahead and must be rejected.

This module returns a `confirmation_idx` for every detected swing so the
strategy code can lag correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Swing:
    idx: int  # bar index where the swing occurred
    confirm_idx: int  # bar index where the swing was confirmable (idx + N)
    price: float
    is_high: bool  # True for swing high, False for swing low


def detect_swings(
    high: pd.Series,
    low: pd.Series,
    n: int,
) -> list[Swing]:
    """Detect all swing highs/lows with lookback N on each side.

    Parameters
    ----------
    high, low:
        OHLC high and low series, integer-indexed (0..len-1) or with the same
        DatetimeIndex. We use positional indexing internally; caller maps
        idx back to timestamps.
    n:
        Lookback on each side (e.g., 2 for daily, 6 for 15M per the spec).

    Returns
    -------
    list[Swing] sorted by idx ascending. A bar can be both a swing high and
    a swing low (rare but possible) — both are returned.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    h = high.to_numpy()
    lo = low.to_numpy()
    m = len(h)
    if m < 2 * n + 1:
        return []

    swings: list[Swing] = []
    for i in range(n, m - n):
        # Strict inequality on both sides — ties don't qualify as a swing,
        # matching the convention "higher than the N candles on both sides".
        if h[i] > h[i - n : i].max() and h[i] > h[i + 1 : i + n + 1].max():
            swings.append(Swing(idx=i, confirm_idx=i + n, price=float(h[i]), is_high=True))
        if lo[i] < lo[i - n : i].min() and lo[i] < lo[i + 1 : i + n + 1].min():
            swings.append(Swing(idx=i, confirm_idx=i + n, price=float(lo[i]), is_high=False))
    swings.sort(key=lambda s: s.idx)
    return swings


def trend_state_series(
    swings: list[Swing],
    n_bars: int,
) -> pd.Series:
    """Compute the trend-state series: +1 (up), -1 (down), 0 (neutral) per bar.

    Trend rule: 2 consecutive HH+HL (up) or 2 consecutive LH+LL (down),
    evaluated on swings *as they become confirmed*.

    The state at bar i reflects only swings with confirm_idx <= i (so no
    look-ahead). Once set, it persists until a contrary 2-swing structure
    confirms in the opposite direction.

    Implementation:
        Walk bars left-to-right. Maintain rolling lists of recent confirmed
        swing highs and swing lows (most recent at end). At each bar, ingest
        any newly-confirmed swings; then check the most recent two highs
        and two lows for HH/HL or LH/LL pattern.
    """
    state = np.zeros(n_bars, dtype=np.int8)
    # Sort swings by confirmation time so we ingest in causal order
    confirmed = sorted(swings, key=lambda s: s.confirm_idx)
    cursor = 0  # index into `confirmed`
    highs: list[Swing] = []
    lows: list[Swing] = []
    cur_state = 0
    for i in range(n_bars):
        while cursor < len(confirmed) and confirmed[cursor].confirm_idx <= i:
            sw = confirmed[cursor]
            if sw.is_high:
                highs.append(sw)
            else:
                lows.append(sw)
            cursor += 1
        # Evaluate trend from the last 2 highs and last 2 lows (if we have them)
        if len(highs) >= 2 and len(lows) >= 2:
            h2, h1 = highs[-2], highs[-1]
            l2, l1 = lows[-2], lows[-1]
            # Uptrend: HH and HL — and the swing lows must alternate with highs
            # (we don't enforce strict alternation here; the structural check
            # is just "most recent two highs are higher and most recent two
            # lows are higher", which is the spirit of the rule.)
            if h1.price > h2.price and l1.price > l2.price:
                cur_state = 1
            elif h1.price < h2.price and l1.price < l2.price:
                cur_state = -1
            # else: keep previous state (no flip on a single non-confirming swing)
        state[i] = cur_state
    return pd.Series(state, index=range(n_bars))
