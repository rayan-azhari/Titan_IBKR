"""fibonacci_features.py -- Fibonacci Retracement Feature Computation.

Auto-detects swing highs/lows and computes Fibonacci retracement levels
as features for ML models. All features are causal (no look-ahead).

Features produced:
  fib_retrace_depth: how deep into the range price has retraced [0, 1]
  fib_dist_382: distance to 38.2% level (normalized by ATR)
  fib_dist_500: distance to 50.0% level
  fib_dist_618: distance to 61.8% level
  fib_near_level: 1 if close is within 0.5*ATR of any Fib level
  fib_zone: categorical (0-4): below_382, 382_500, 500_618, above_618
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_swing_points(
    high: pd.Series,
    low: pd.Series,
    order: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Rolling causal swing high/low detection.

    A swing high at bar t is confirmed at bar t+order (needs bars on each side).
    Result is shifted by `order` bars to be causal -- the swing is only
    known after confirmation.

    Returns (swing_high, swing_low) Series with NaN where no swing detected.
    """
    n = len(high)
    swing_highs = pd.Series(np.nan, index=high.index)
    swing_lows = pd.Series(np.nan, index=low.index)

    high_vals = high.values
    low_vals = low.values

    for i in range(order, n - order):
        # Swing high: bar i is higher than `order` bars on each side
        is_high = True
        for j in range(1, order + 1):
            if high_vals[i] <= high_vals[i - j] or high_vals[i] <= high_vals[i + j]:
                is_high = False
                break
        if is_high:
            # Confirmed at bar i+order (causal)
            swing_highs.iloc[i + order] = high_vals[i]

        # Swing low
        is_low = True
        for j in range(1, order + 1):
            if low_vals[i] >= low_vals[i - j] or low_vals[i] >= low_vals[i + j]:
                is_low = False
                break
        if is_low:
            swing_lows.iloc[i + order] = low_vals[i]

    # Forward-fill to carry the most recent confirmed swing
    swing_highs = swing_highs.ffill()
    swing_lows = swing_lows.ffill()

    return swing_highs, swing_lows


def compute_fibonacci_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    order: int = 5,
) -> pd.DataFrame:
    """Compute Fibonacci retracement features.

    All features shifted by 1 bar for no look-ahead.
    """
    swing_high, swing_low = detect_swing_points(high, low, order=order)

    rng = swing_high - swing_low
    rng_safe = rng.replace(0, np.nan).clip(lower=1e-8)

    # Fibonacci levels
    fib_382 = swing_low + rng * 0.382
    fib_500 = swing_low + rng * 0.500
    fib_618 = swing_low + rng * 0.618

    # ATR for normalization
    atr_safe = atr.clip(lower=1e-8)

    out = pd.DataFrame(index=close.index)

    # How deep into the range price has retraced (0 = at swing low, 1 = at swing high)
    out["fib_retrace_depth"] = (close - swing_low) / rng_safe

    # Distance to each Fib level (normalized by ATR, signed)
    out["fib_dist_382"] = (close - fib_382) / atr_safe
    out["fib_dist_500"] = (close - fib_500) / atr_safe
    out["fib_dist_618"] = (close - fib_618) / atr_safe

    # Near any Fib level (within 0.5 ATR)
    near_382 = (close - fib_382).abs() < 0.5 * atr_safe
    near_500 = (close - fib_500).abs() < 0.5 * atr_safe
    near_618 = (close - fib_618).abs() < 0.5 * atr_safe
    out["fib_near_level"] = (near_382 | near_500 | near_618).astype(float)

    # Fib zone (categorical: 0-4)
    depth = out["fib_retrace_depth"]
    out["fib_zone"] = np.where(
        depth < 0.382, 0, np.where(depth < 0.500, 1, np.where(depth < 0.618, 2, 3))
    ).astype(float)

    # Shift by 1 for no look-ahead
    return out.shift(1)
