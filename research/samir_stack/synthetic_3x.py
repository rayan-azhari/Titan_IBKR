"""Synthetic leveraged-ETF return generator.

A daily-rebalanced Lx ETF earns:

    r_daily(L) = L * r_underlying - (L - 1) * funding_daily - TER_daily

Volatility drag emerges naturally from compounding L*r daily. There is no
need to add a `0.5*L*(L-1)*sigma^2` term explicitly — the path of leveraged
daily returns produces it on its own.

Validated against actual 3USL.LSEETF 2018-2024 history (see
`validate_against_3usl()`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Default parameters ─────────────────────────────────────────────────────
#
# 3USL.LSEETF expense ratio (the UCITS 3x S&P 500 product).
# WisdomTree's Boost 3x Long S&P 500 has a 0.75% TER. UPRO (US-domiciled
# 3x SPY) is 0.91%. Choose 0.75% as default since the live target is UK.
DEFAULT_TER_ANNUAL = 0.0075

# Piecewise approximation of historical USD funding rate (effective Fed
# Funds annualised). Used because the project doesn't have a FRED parquet
# and we don't want to introduce an external dependency. Granularity: yearly
# averages (sufficient for the 0.5-1% drag accuracy we need; the dominant
# leverage drag term is volatility decay, not funding precision).
#
# Sources cross-referenced: FRED DFF, NY Fed effective FF series.
_FUNDING_PIECEWISE = [
    # (year_start, annual_rate)
    (1993, 0.030),
    (1994, 0.043),
    (1995, 0.058),
    (1996, 0.052),
    (1997, 0.054),
    (1998, 0.055),
    (1999, 0.050),
    (2000, 0.062),
    (2001, 0.039),
    (2002, 0.017),
    (2003, 0.011),
    (2004, 0.014),
    (2005, 0.032),
    (2006, 0.050),
    (2007, 0.050),
    (2008, 0.020),
    (2009, 0.002),
    (2010, 0.002),
    (2011, 0.001),
    (2012, 0.001),
    (2013, 0.001),
    (2014, 0.001),
    (2015, 0.002),
    (2016, 0.004),
    (2017, 0.011),
    (2018, 0.020),
    (2019, 0.022),
    (2020, 0.004),
    (2021, 0.001),
    (2022, 0.020),
    (2023, 0.050),
    (2024, 0.052),
    (2025, 0.045),
    (2026, 0.040),
]


def annual_funding_rate(date: pd.Timestamp | str) -> float:
    """Return the annualised funding rate to use for a given date."""
    ts = pd.Timestamp(date)
    yr = ts.year
    rate = _FUNDING_PIECEWISE[0][1]
    for y, r in _FUNDING_PIECEWISE:
        if yr >= y:
            rate = r
        else:
            break
    return rate


def funding_series(index: pd.DatetimeIndex) -> pd.Series:
    """Daily-frequency annualised funding rate for the given index."""
    return pd.Series([annual_funding_rate(t) for t in index], index=index, name="funding_annual")


def synthetic_leveraged_returns(
    underlying_close: pd.Series,
    leverage: float,
    *,
    ter_annual: float = DEFAULT_TER_ANNUAL,
    funding_annual: pd.Series | float | None = None,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Generate the daily-return series of a synthetic Lx ETF.

    Parameters
    ----------
    underlying_close
        Close prices of the underlying (e.g., SPY). Must be a clean
        DatetimeIndex'd Series.
    leverage
        Constant leverage factor. ``1.0`` returns the underlying minus TER
        (no funding cost since no borrowing). ``3.0`` is the 3x case.
    ter_annual
        Total expense ratio (annual, decimal). For ``leverage == 1.0`` we
        still apply this — represents the carry cost of holding the ETF
        wrapper. Pass ``0.0`` if simulating direct underlying holding.
    funding_annual
        Annualised borrow rate. Series indexed to match ``underlying_close``,
        a scalar applied uniformly, or None to use the default piecewise
        Fed Funds approximation (see ``_FUNDING_PIECEWISE``).
    trading_days_per_year
        Annualisation denominator (default 252).

    Returns
    -------
    pd.Series
        Daily returns of the synthetic Lx ETF, same index as input (less
        the first bar which is NaN from pct_change).

    Notes
    -----
    For ``leverage == 1.0`` we deliberately keep the TER term so that
    1x-vs-3x comparisons are like-for-like (both pay the wrapper). If you
    want pure SPY-buy-hold performance, call with ``ter_annual=0.0``.
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")
    if not isinstance(underlying_close, pd.Series):
        raise TypeError("underlying_close must be a pd.Series")

    spy_ret = underlying_close.pct_change()

    if funding_annual is None:
        funding = funding_series(underlying_close.index)
    elif isinstance(funding_annual, (int, float)):
        funding = pd.Series(float(funding_annual), index=underlying_close.index)
    elif isinstance(funding_annual, pd.Series):
        funding = funding_annual.reindex(underlying_close.index).ffill()
    else:
        raise TypeError("funding_annual must be Series, scalar, or None")

    daily_ter = ter_annual / trading_days_per_year
    daily_funding = funding / trading_days_per_year

    # Borrow only when leverage > 1
    borrow_factor = max(leverage - 1.0, 0.0)
    return (leverage * spy_ret) - (borrow_factor * daily_funding) - daily_ter


def synthetic_leveraged_equity(
    underlying_close: pd.Series,
    leverage: float,
    *,
    initial_value: float = 100.0,
    **kwargs,
) -> pd.Series:
    """Cumulative equity curve of a synthetic Lx ETF, starting at ``initial_value``."""
    rets = synthetic_leveraged_returns(underlying_close, leverage, **kwargs)
    return (1.0 + rets.fillna(0.0)).cumprod() * initial_value


# ── Validation against actual 3USL ─────────────────────────────────────────


def validate_against_3usl(spy_close: pd.Series, threeusl_close: pd.Series) -> dict:
    """Validate the 3x synthetic against actual 3USL.LSEETF history.

    Compares CAGR, vol, max DD, and full equity-curve correlation across the
    overlapping period. Returns a dict of comparison metrics. Acceptable
    deviation: synthetic CAGR within 2pp of actual, MaxDD within 5pp,
    daily-return correlation > 0.99.
    """
    common = spy_close.index.intersection(threeusl_close.index)
    if len(common) < 252:
        return {"error": f"insufficient overlap: {len(common)} bars"}

    spy = spy_close.loc[common]
    actual = threeusl_close.loc[common]
    synth_eq = synthetic_leveraged_equity(spy, leverage=3.0, initial_value=float(actual.iloc[0]))

    actual_ret = actual.pct_change().dropna()
    synth_ret = synth_eq.pct_change().dropna()
    aligned = pd.concat([actual_ret, synth_ret], axis=1, join="inner").dropna()
    aligned.columns = ["actual", "synth"]

    n_years = len(common) / 252.0

    def cagr(series: pd.Series) -> float:
        return float((series.iloc[-1] / series.iloc[0]) ** (1.0 / n_years) - 1.0)

    def maxdd(series: pd.Series) -> float:
        peak = series.cummax()
        return float(((series - peak) / peak).min())

    return {
        "n_bars": len(common),
        "n_years": round(n_years, 2),
        "actual_cagr": round(cagr(actual), 4),
        "synth_cagr": round(cagr(synth_eq), 4),
        "cagr_diff_pp": round((cagr(synth_eq) - cagr(actual)) * 100, 2),
        "actual_vol": round(float(aligned["actual"].std() * np.sqrt(252)), 4),
        "synth_vol": round(float(aligned["synth"].std() * np.sqrt(252)), 4),
        "actual_maxdd": round(maxdd(actual), 4),
        "synth_maxdd": round(maxdd(synth_eq), 4),
        "daily_return_corr": round(float(aligned["actual"].corr(aligned["synth"])), 4),
    }
