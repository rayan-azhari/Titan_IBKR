"""Margin-financed equity returns — alternative to daily-rebalanced leveraged ETFs.

Two models are exposed:

1. ``constant_leverage_margin_returns(spy, L, ...)``: holder targets a
   constant L every day and rebalances at the close. Mathematically
   equivalent to ``synthetic_3x.synthetic_leveraged_returns`` except the
   funding rate uses the BROKER margin rate (= benchmark + spread, e.g.
   IBKR Pro tier is roughly SOFR + 1.5%) instead of the wholesale rate
   the ETF issuer pays. Also uses a single 1× ETF TER (e.g. CSPX 0.07%)
   instead of the leveraged-ETF TER (3USL 0.75%).

2. ``drift_margin_returns(spy, L_initial, ...)``: holder borrows a fixed
   dollar amount on day 0 and lets leverage drift with the underlying.
   Less volatility drag than the constant-L case, but exposes the holder
   to margin calls when the underlying drops sharply. Includes an
   optional auto-deleverage ("margin call") trigger.

The economic intuition behind the comparison vs leveraged ETFs:

  - Leveraged ETF (e.g. 3USL): rebalances daily by construction → max vol
    drag. Issuer pays wholesale funding (~Fed Funds), high TER (~0.75%).
  - Constant-L margin: same daily rebalancing → same vol drag. But pays
    retail margin (~SOFR + 1.5%) and only the underlying ETF's low TER
    (~0.07%). Net cost is roughly a wash with leveraged ETFs at L=3, but
    favours margin at L=2 because the borrow scales linearly with (L-1).
  - Drift margin: no rebalancing → no daily vol drag, but risks a margin
    call in a sharp drawdown. Best for tactical positions held briefly.

Reference: ``synthetic_3x.py`` for the leveraged-ETF model this file
mirrors. ``directives/Cost Model Audit 2026-05-11.md`` for the live
cost-model audit context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.synthetic_3x import funding_series

# IBKR retail margin rates above benchmark (annualised, decimal). Cross-
# reference: interactivebrokers.com/en/trading/margin-rates.php (US tier
# 0-100k → benchmark + 1.5% Pro / + 2.5% Lite, drops with size).
IBKR_PRO_SPREAD_OVER_BENCHMARK = 0.015
IBKR_LITE_SPREAD_OVER_BENCHMARK = 0.025

# CSPX (iShares S&P 500 UCITS Acc, USD) management fee. Reference:
# ishares.com/uk/individual/en/products/253743 — TER 0.07%.
CSPX_TER_ANNUAL = 0.0007

# Standard US equity initial margin = 50% (Reg T). Maintenance margin =
# 25% (FINRA), but IBKR uses 30% intraday for most US ETFs. Holding L=2
# means initial equity = 50% of position; a 50% drop in underlying takes
# equity to 0% and triggers liquidation. With 30% maintenance, leverage
# drift to L > 1/0.30 ≈ 3.33 triggers a call.
DEFAULT_MAINTENANCE_MARGIN = 0.30


def margin_funding_series(
    index: pd.DatetimeIndex,
    *,
    broker: str = "ibkr_pro",
) -> pd.Series:
    """Per-bar annualised margin rate paid by the borrower.

    = wholesale benchmark (Fed Funds approx) + broker spread.
    """
    base = funding_series(index)
    spread = (
        IBKR_PRO_SPREAD_OVER_BENCHMARK if broker == "ibkr_pro" else IBKR_LITE_SPREAD_OVER_BENCHMARK
    )
    return base + spread


def constant_leverage_margin_returns(
    underlying_close: pd.Series,
    leverage: float,
    *,
    ter_annual: float = CSPX_TER_ANNUAL,
    broker: str = "ibkr_pro",
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Daily returns of a constant-L margined position rebalanced at close.

    Mathematically identical to the leveraged-ETF model except for funding
    rate (broker margin instead of wholesale) and TER (single underlying
    ETF instead of leveraged wrapper).
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")
    spy_ret = underlying_close.pct_change()
    funding = margin_funding_series(underlying_close.index, broker=broker)
    daily_ter = ter_annual / trading_days_per_year
    daily_funding = funding / trading_days_per_year
    borrow_factor = max(leverage - 1.0, 0.0)
    return (leverage * spy_ret) - (borrow_factor * daily_funding) - daily_ter


def drift_margin_returns(
    underlying_close: pd.Series,
    *,
    initial_leverage: float = 2.0,
    ter_annual: float = CSPX_TER_ANNUAL,
    broker: str = "ibkr_pro",
    maintenance_margin: float = DEFAULT_MAINTENANCE_MARGIN,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Margin position with FIXED borrow notional. Leverage drifts.

    Day 0: equity = E_0; borrow = E_0 * (L - 1); position = L * E_0.
    Each day:
        position_t = position_{t-1} * (1 + spy_ret_t)
        equity_t   = position_t - borrow * (1 + cumulative_funding)
        leverage_t = position_t / equity_t

    If equity_t / position_t drops below ``maintenance_margin`` we
    auto-deleverage: sell enough shares to bring leverage back to
    1 / maintenance_margin. The deleverage is recorded as a margin call.

    Returns a DataFrame with columns ``equity_ret`` (daily return on
    starting equity), ``leverage``, ``margin_call``, ``cum_funding_pct``.
    """
    if initial_leverage <= 0:
        raise ValueError(f"initial_leverage must be > 0, got {initial_leverage}")
    n = len(underlying_close)
    spy_ret = underlying_close.pct_change().fillna(0.0).to_numpy()
    funding = margin_funding_series(underlying_close.index, broker=broker).to_numpy()
    daily_funding = funding / trading_days_per_year
    daily_ter = ter_annual / trading_days_per_year

    equity = np.empty(n)
    leverage = np.empty(n)
    margin_call = np.zeros(n, dtype=bool)
    borrow = np.empty(n)

    equity[0] = 1.0
    borrow[0] = max(initial_leverage - 1.0, 0.0)
    position = initial_leverage  # = equity[0] * L
    leverage[0] = initial_leverage

    for i in range(1, n):
        # Borrow accrues funding daily
        borrow[i] = borrow[i - 1] * (1.0 + daily_funding[i])
        # Position drifts with the underlying
        position = position * (1.0 + spy_ret[i])
        # TER charged on the gross position (proxy: ETF wrapper fee on
        # held shares). For un-leveraged underlying like CSPX this is
        # ~0.07% per year on ALL the shares held, not just the equity slice.
        position *= 1.0 - daily_ter
        equity[i] = position - borrow[i]

        # Margin call check
        if equity[i] <= 0 or (equity[i] / position) < maintenance_margin:
            # Forced deleverage: sell shares to bring leverage to 1/maintenance.
            target_lev = 1.0 / maintenance_margin
            # If equity already wiped, lock in 100% loss and stop
            if equity[i] <= 0:
                equity[i:] = 0.0
                leverage[i:] = 0.0
                margin_call[i] = True
                break
            new_position = equity[i] * target_lev
            new_borrow = new_position - equity[i]
            position = new_position
            borrow[i] = new_borrow
            margin_call[i] = True

        leverage[i] = position / equity[i] if equity[i] > 0 else 0.0

    eq_ret = np.zeros(n)
    eq_ret[1:] = (equity[1:] - equity[:-1]) / equity[:-1]
    cum_funding_pct = (borrow / equity[0] - max(initial_leverage - 1.0, 0.0)) * 100.0

    return pd.DataFrame(
        {
            "equity_ret": eq_ret,
            "leverage": leverage,
            "margin_call": margin_call,
            "cum_funding_pct": cum_funding_pct,
        },
        index=underlying_close.index,
    )
