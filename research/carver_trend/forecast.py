"""Carver-style EWMAC forecast computation.

Pure functions, no state. Each function takes a price series and returns
a forecast Series (or DataFrame) suitable for downstream backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Carver's published forecast scaling factors. These are the multipliers
# that bring abs(mean) of a raw EWMAC/sigma signal to ~10 across his
# diversified universe. They are NOT tuned per-instrument — that would be
# in-sample fitting. We adopt his published constants verbatim.
EWMAC_SCALARS: dict[tuple[int, int], float] = {
    (2, 8): 12.10,
    (4, 16): 8.53,
    (8, 32): 5.95,
    (16, 64): 4.10,
    (32, 128): 2.65,
    (64, 256): 1.69,
}

# Forecast cap (Carver convention). Anything above ±20 is clipped.
FORECAST_CAP: float = 20.0

# Default ladder — Carver's three medium-speed EWMACs.
DEFAULT_LADDER: tuple[tuple[int, int], ...] = ((16, 64), (32, 128), (64, 256))

# Forecast diversification multiplier for the default 3-variant ladder.
# Carver derives this from the correlation of forecast variants — a
# 3-variant ladder with correlations ~0.6-0.7 gets FDM ~1.4.
DEFAULT_FDM: float = 1.4

# Daily vol estimation: 36-day equivalent EWMA half-life. Carver uses a
# blend of short (~25-day) and long (~10-year) but the simple 36-day
# EWMA is close enough for a first pass.
VOL_HALFLIFE_DAYS: int = 36


def daily_vol(close: pd.Series, halflife: int = VOL_HALFLIFE_DAYS) -> pd.Series:
    """EWMA daily vol estimate. Returns a Series in the same units as price.

    Carver uses 36-day half-life on daily returns. We use price differences
    in the same units as price (so vol can be divided into price-difference
    forecasts cleanly).
    """
    diff = close.diff()
    var = diff.pow(2).ewm(halflife=halflife, min_periods=halflife // 2).mean()
    return var.pow(0.5)


def ewmac_raw(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Raw EWMAC signal: EMA(close, fast) - EMA(close, slow).

    Same units as price. Positive = uptrend, negative = downtrend.
    """
    if fast >= slow:
        raise ValueError(f"fast must be < slow, got fast={fast}, slow={slow}")
    fast_ema = close.ewm(span=fast, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, min_periods=slow).mean()
    return fast_ema - slow_ema


def ewmac_forecast(
    close: pd.Series,
    fast: int,
    slow: int,
    vol_halflife: int = VOL_HALFLIFE_DAYS,
) -> pd.Series:
    """Carver's normalised EWMAC forecast.

    Steps:
      1. raw = EWMAC(fast, slow) on price                              [same units as price]
      2. vol = EWMA daily vol of price changes                          [same units]
      3. risk-adjusted = raw / vol                                      [unitless]
      4. scaled = risk-adjusted * Carver's published scalar             [target abs(mean)=10]
      5. capped = clip to ±FORECAST_CAP                                 [final forecast]

    Returns a Series of forecasts where +20 = max long view,
    -20 = max short view, 0 = no view.
    """
    scalar = EWMAC_SCALARS.get((fast, slow))
    if scalar is None:
        raise ValueError(
            f"No published scalar for EWMAC({fast},{slow}). Known: {list(EWMAC_SCALARS.keys())}"
        )
    raw = ewmac_raw(close, fast, slow)
    vol = daily_vol(close, halflife=vol_halflife)
    risk_adj = raw / vol.replace(0.0, np.nan)
    scaled = risk_adj * scalar
    capped = scaled.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)
    return capped


def ladder_forecast(
    close: pd.Series,
    ladder: tuple[tuple[int, int], ...] = DEFAULT_LADDER,
    fdm: float = DEFAULT_FDM,
    vol_halflife: int = VOL_HALFLIFE_DAYS,
) -> pd.Series:
    """Combined forecast from an EWMAC ladder.

    Average the per-variant forecasts, apply FDM (diversification multiplier),
    then re-cap at ±FORECAST_CAP. The final cap is important — a high-FDM
    average of three already-near-cap forecasts could blow past the limit
    and amplify tail risk.
    """
    variants = [
        ewmac_forecast(close, fast, slow, vol_halflife=vol_halflife) for fast, slow in ladder
    ]
    avg = pd.concat(variants, axis=1).mean(axis=1)
    combined = (avg * fdm).clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)
    return combined
