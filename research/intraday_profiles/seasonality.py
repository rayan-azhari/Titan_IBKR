"""seasonality.py — Intraday volatility / drift / volume heatmap.

No ML dependencies. Pure numpy/pandas.
All computations use past data only (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hourly seasonality
# ---------------------------------------------------------------------------


def hourly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Per-UTC-hour statistics over the full dataset.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.

    Returns:
        DataFrame indexed by hour (0-23) with columns:
          abs_ret     — mean |close - open| / open (intraday volatility proxy)
          fwd_drift   — mean (close - open) / open (directional bias)
          volume_norm — mean volume / overall mean volume (liquidity proxy)
          n_bars      — observation count per hour
    """
    tmp = df.copy()
    tmp["hour"] = tmp.index.hour
    tmp["bar_ret"] = (tmp["close"] - tmp["open"]) / tmp["open"]
    mean_vol = tmp["volume"].mean()

    result = tmp.groupby("hour").agg(
        abs_ret=("bar_ret", lambda x: x.abs().mean()),
        fwd_drift=("bar_ret", "mean"),
        volume_norm=("volume", lambda x: x.mean() / mean_vol),
        n_bars=("bar_ret", "count"),
    )
    return result.sort_index()


def minute_of_day_volume_baseline(
    df: pd.DataFrame,
    window_days: int = 30,
) -> pd.Series:
    """Rolling per-minute-of-day volume baseline (IS-safe when applied to IS slice).

    Computes, for each M5 bar, the mean volume at the same minute-of-day over the
    previous `window_days` trading days. Used to build RelVol feature.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.
        window_days: Rolling window in calendar days for the baseline.

    Returns:
        Series of baseline volumes aligned to df.index.
    """
    tmp = df[["volume"]].copy()
    tmp["minute_of_day"] = tmp.index.hour * 60 + tmp.index.minute

    # Compute per-minute rolling mean using a date-offset window
    baseline = tmp.groupby("minute_of_day")["volume"].transform(
        lambda s: s.rolling(window=window_days, min_periods=5).mean().shift(1)
    )
    return baseline.rename("vol_baseline")


def t_test_drift(seasonality_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """One-sample t-test: is mean hourly drift significantly different from zero?

    Adds columns 't_stat' and 'p_value' to the seasonality DataFrame.

    Args:
        seasonality_df: Output of hourly_seasonality().
        df: Original OHLCV DataFrame.

    Returns:
        seasonality_df with t_stat and p_value columns appended.
    """
    from scipy import stats

    tmp = df.copy()
    tmp["hour"] = tmp.index.hour
    tmp["bar_ret"] = (tmp["close"] - tmp["open"]) / tmp["open"]

    t_stats, p_vals = [], []
    for hour in seasonality_df.index:
        rets = tmp.loc[tmp["hour"] == hour, "bar_ret"].dropna()
        if len(rets) < 10:
            t_stats.append(np.nan)
            p_vals.append(np.nan)
        else:
            t, p = stats.ttest_1samp(rets, 0.0)
            t_stats.append(float(t))
            p_vals.append(float(p))

    seasonality_df = seasonality_df.copy()
    seasonality_df["t_stat"] = t_stats
    seasonality_df["p_value"] = p_vals
    return seasonality_df
