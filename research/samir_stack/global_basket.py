"""Synthetic global equity basket for testing diversification benefit.

Constructs a daily-rebalanced 60/25/15 basket of SPY + EFA + EEM that
approximates MSCI ACWI cap weights (US / Developed-ex-US / Emerging
Markets). Then applies the synthetic Lx leveraged-ETF model to that
basket.

For LIVE deployment on IBKR UK paper, the equivalent tradables are:
    * 3USL.LSEETF (3x S&P 500 UCITS) — US sleeve
    * 3EUL.LSEETF (3x Euro Stoxx 50 UCITS) — EU sleeve
    * 3JPL.LSEETF (3x Nikkei 225 UCITS) — JP sleeve
A pure 3x global ETF does not exist in UCITS form. The closest tradable
strategy is to weight the regional 3x ETFs to match this synthetic basket.
"""

from __future__ import annotations

import pandas as pd

from research.samir_stack.synthetic_3x import (
    DEFAULT_TER_ANNUAL,
    funding_series,
)


def build_global_basket(
    spy_close: pd.Series,
    efa_close: pd.Series,
    eem_close: pd.Series,
    *,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Build a daily-rebalanced global equity basket.

    Default weights mirror MSCI ACWI (approximately):
        SPY: 60% (US large-cap)
        EFA: 25% (developed ex-US)
        EEM: 15% (emerging markets)

    Returns a synthetic price series indexed by the common dates,
    starting at 100.0 and drifting per the basket's daily returns.
    """
    if weights is None:
        weights = {"spy": 0.60, "efa": 0.25, "eem": 0.15}

    common = (
        spy_close.index.intersection(efa_close.index).intersection(eem_close.index)
    )
    if len(common) < 252:
        raise ValueError(f"Insufficient overlap: {len(common)} bars")

    spy = spy_close.loc[common]
    efa = efa_close.loc[common]
    eem = eem_close.loc[common]

    # Daily returns of each sleeve
    r_spy = spy.pct_change().fillna(0.0)
    r_efa = efa.pct_change().fillna(0.0)
    r_eem = eem.pct_change().fillna(0.0)

    # Daily-rebalanced basket return
    r_basket = (
        weights["spy"] * r_spy
        + weights["efa"] * r_efa
        + weights["eem"] * r_eem
    )
    # Cumulate to a price-like series starting at 100
    basket_price = (1.0 + r_basket).cumprod() * 100.0
    basket_price.name = "global_basket"
    return basket_price


def synthetic_leveraged_global(
    basket_price: pd.Series,
    leverage: float,
    *,
    ter_annual: float = DEFAULT_TER_ANNUAL,
    funding_annual: pd.Series | float | None = None,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Apply Lx leveraged-ETF math to the global basket.

    Same formula as ``synthetic_leveraged_returns`` in synthetic_3x.py:
        r_daily(L) = L * r_basket - (L-1) * funding_daily - TER_daily

    This implicitly assumes a swap-based 3x ETF on the global basket
    exists, which it doesn't directly — but the math models the
    *expected* behaviour of an equally-weighted basket of 3x regional
    ETFs (since 3x is linear in the underlying return).
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")

    basket_ret = basket_price.pct_change()

    if funding_annual is None:
        funding = funding_series(basket_price.index)
    elif isinstance(funding_annual, (int, float)):
        funding = pd.Series(float(funding_annual), index=basket_price.index)
    else:
        funding = funding_annual.reindex(basket_price.index).ffill()

    daily_ter = ter_annual / trading_days_per_year
    daily_funding = funding / trading_days_per_year
    borrow_factor = max(leverage - 1.0, 0.0)

    return (leverage * basket_ret) - (borrow_factor * daily_funding) - daily_ter
