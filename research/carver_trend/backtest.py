"""Carver-style position sizing and PnL simulation.

Given a continuous forecast (Series in [-20, +20]) and a price series,
produce per-bar returns under vol-targeted continuous sizing.

Convention: forecast at close[t] sets position for the bar t -> t+1.
i.e. signal lags. ``return[t+1] = position[t] * (close[t+1]/close[t] - 1)``.
This is shifted via .shift(1) on the position before multiplying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.carver_trend.forecast import daily_vol


def vol_target_position(
    forecast: pd.Series,
    close: pd.Series,
    vol_target_pct: float = 0.25,
    capital: float = 1.0,
    vol_halflife_days: int = 36,
    periods_per_year: int = 252,
) -> pd.Series:
    """Convert a continuous forecast (-20..+20) to a fractional position.

    Returns *fractional notional* (i.e. position size as a fraction of
    `capital`). For a Sharpe / portfolio-vol calculation the absolute
    `capital` cancels out — only the time-series of position-fractions
    matters.

    Formula::

        pct_vol = daily_price_vol / price                  # daily return-vol per bar
        ann_vol = pct_vol * sqrt(periods_per_year)
        notional_target = capital * vol_target_pct / ann_vol
        position = (forecast / 10.0) * notional_target / capital

    A forecast of 10 (Carver's "average" view) maps to a notional of
    ``vol_target_pct / ann_vol`` (i.e. full-vol-target sizing). +20 = 2×
    that, -20 = -2× (short).

    Args:
        forecast: Series in [-20, +20], indexed like ``close``.
        close: Price series.
        vol_target_pct: Annualised vol target (default 25% per Carver).
        capital: Notional capital (cancels in returns; left for clarity).
        vol_halflife_days: EWMA half-life for vol estimate (Carver: 36).
        periods_per_year: Annualisation factor for the bar timeframe.
    """
    price_vol = daily_vol(close, halflife=vol_halflife_days)
    pct_vol = price_vol / close.replace(0.0, np.nan)
    ann_vol = pct_vol * np.sqrt(periods_per_year)
    notional_target = capital * vol_target_pct / ann_vol.replace(0.0, np.nan)
    position_fraction = (forecast / 10.0) * notional_target / capital
    return position_fraction


def backtest_returns(
    forecast: pd.Series,
    close: pd.Series,
    *,
    vol_target_pct: float = 0.25,
    cost_bps_per_turn: float = 1.0,
    vol_halflife_days: int = 36,
    periods_per_year: int = 252,
) -> dict:
    """Run the backtest. Returns a dict with the per-bar return series and
    summary stats.

    Cost model: ``cost_bps_per_turn`` charged on the absolute change in
    position fraction each bar. A turn from +1 to -1 costs 2 * cost_bps.

    Returns
    -------
    dict with keys:
        net_returns   : pd.Series of per-bar returns net of cost
        gross_returns : pd.Series of per-bar returns gross of cost
        position      : pd.Series of position fractions (signed)
        n_bars        : int
        n_turns       : float (sum of |position changes|)
    """
    position = vol_target_position(
        forecast,
        close,
        vol_target_pct=vol_target_pct,
        vol_halflife_days=vol_halflife_days,
        periods_per_year=periods_per_year,
    )
    # Bar-to-bar return on price
    bar_ret = close.pct_change().shift(-1)  # ret from t to t+1, indexed at t
    # Position at close[t] earns return t -> t+1
    gross = position.fillna(0.0) * bar_ret.fillna(0.0)
    # Costs: |pos[t] - pos[t-1]| * (cost_bps / 10000)
    pos_change = position.fillna(0.0).diff().abs().fillna(0.0)
    cost = pos_change * (cost_bps_per_turn / 1e4)
    net = gross - cost
    return {
        "net_returns": net,
        "gross_returns": gross,
        "position": position,
        "cost": cost,
        "n_bars": int(net.notna().sum()),
        "n_turns": float(pos_change.sum()),
    }
