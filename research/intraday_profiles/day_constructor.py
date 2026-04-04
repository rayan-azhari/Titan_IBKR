"""day_constructor.py — Build aligned per-day OHLCV feature matrices.

Each trading day is represented as a fixed-length sequence of 3 microstructure features:
  - TR  (True Range):         absolute volatility / bar expansion
  - CLV (Close Location Value): who won the 5-min auction, in [-1, +1]
  - RelVol (Relative Volume):  volume vs. same-minute-of-day baseline

The result is a 3D tensor of shape (n_days, n_bars_per_session, 3), suitable for
both sklearn PCA (after flattening) and tslearn multivariate DTW (natively 3D).

Look-ahead safety:
  - RelVol baseline computed from IS data only, then frozen for OOS.
  - Days with < MIN_BARS bars in a session are excluded from training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AM_START_UTC = 0    # 00:00 UTC inclusive
AM_END_UTC   = 12   # 12:00 UTC exclusive  -> 144 M5 bars
PM_START_UTC = 12   # 12:00 UTC inclusive
PM_END_UTC   = 22   # 22:00 UTC exclusive  -> 120 M5 bars
MIN_BARS     = 100  # Drop days with fewer bars than this from training


# ---------------------------------------------------------------------------
# Per-bar feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame, vol_baseline: pd.Series | None = None) -> pd.DataFrame:
    """Compute TR, CLV, RelVol per M5 bar.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.
        vol_baseline: Pre-computed per-minute-of-day volume baseline (from IS).
            If None, RelVol = 1.0 for all bars (no volume normalisation).

    Returns:
        DataFrame with columns ['tr', 'clv', 'relvol'] aligned to df.index.
    """
    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Close Location Value: (C-L - H-C) / (H-L), in [-1, +1]
    hl = df["high"] - df["low"]
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    clv = clv.where(hl > 0, 0.0)  # doji bars: CLV = 0

    # Relative Volume
    if vol_baseline is not None:
        relvol = df["volume"] / vol_baseline.reindex(df.index).replace(0, np.nan)
        relvol = relvol.fillna(1.0).clip(0.0, 10.0)
    else:
        relvol = pd.Series(1.0, index=df.index)

    return pd.DataFrame({"tr": tr, "clv": clv, "relvol": relvol})


def compute_vol_baseline(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """Per-minute-of-day rolling volume baseline.

    Must be computed on IS data only and frozen for OOS.

    Args:
        df: IS-only OHLCV DataFrame.
        window_days: Rolling lookback in days (number of past occurrences of
            the same minute-of-day).

    Returns:
        Series of baseline volume values, same index as df.
    """
    tmp = df[["volume"]].copy()
    tmp["minute_of_day"] = tmp.index.hour * 60 + tmp.index.minute

    baseline = tmp.groupby("minute_of_day")["volume"].transform(
        lambda s: s.rolling(window=window_days, min_periods=5).mean().shift(1)
    )
    return baseline.fillna(df["volume"].mean()).rename("vol_baseline")


def normalise_tr(features: pd.DataFrame, is_mean_tr: float | None = None) -> pd.DataFrame:
    """Normalise TR by IS mean (makes values dimensionless for clustering).

    Args:
        features: DataFrame with 'tr', 'clv', 'relvol'.
        is_mean_tr: Mean TR from IS dataset. If None, compute from features.

    Returns:
        Copy with TR divided by is_mean_tr.
    """
    out = features.copy()
    mean_tr = is_mean_tr if is_mean_tr is not None else out["tr"].mean()
    if mean_tr > 0:
        out["tr"] = out["tr"] / mean_tr
    return out


# ---------------------------------------------------------------------------
# Daily matrix construction
# ---------------------------------------------------------------------------


def _extract_session(
    features: pd.DataFrame,
    date: pd.Timestamp,
    start_hour: int,
    end_hour: int,
    n_bars: int,
) -> np.ndarray | None:
    """Extract one session's feature matrix for a given date.

    Returns:
        Array of shape (n_bars, 3) or None if too few bars.
    """
    day_slice = features[
        (features.index.date == date.date()) &
        (features.index.hour >= start_hour) &
        (features.index.hour < end_hour)
    ]
    if len(day_slice) < MIN_BARS:
        return None

    # Take first n_bars; pad with last value if slightly short
    vals = day_slice[["tr", "clv", "relvol"]].values  # (k, 3)
    if len(vals) >= n_bars:
        return vals[:n_bars]
    # Pad: repeat last row
    pad = np.tile(vals[-1], (n_bars - len(vals), 1))
    return np.vstack([vals, pad])


def build_daily_matrices(
    features: pd.DataFrame,
    session: str = "am",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (n_days, n_bars, 3) feature tensor for one session.

    Args:
        features: Output of engineer_features() + normalise_tr() with UTC DatetimeIndex.
        session: "am" (00:00-12:00 UTC, 144 bars) or "pm" (12:00-22:00 UTC, 120 bars).

    Returns:
        Tuple of:
          matrices: np.ndarray of shape (n_days, n_bars, 3)
          dates:    np.ndarray of datetime.date objects (one per row)
    """
    if session == "am":
        start_h, end_h, n_bars = AM_START_UTC, AM_END_UTC, 144
    else:
        start_h, end_h, n_bars = PM_START_UTC, PM_END_UTC, 120

    all_dates = np.unique(features.index.date)
    matrices, valid_dates = [], []

    for d in all_dates:
        mat = _extract_session(features, pd.Timestamp(d), start_h, end_h, n_bars)
        if mat is not None:
            matrices.append(mat)
            valid_dates.append(d)

    if not matrices:
        return np.empty((0, n_bars, 3)), np.array([])

    return np.stack(matrices, axis=0), np.array(valid_dates)
