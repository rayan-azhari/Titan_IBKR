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

# SPX trailing 12-month dividend yield. Long-run average ~1.8%; current
# ~1.4%. Used in the futures cost model: basis carry ≈ funding_rate -
# dividend_yield (cost-of-carry minus what spot holders earn).
SPX_DIVIDEND_YIELD_DEFAULT = 0.015

# IBKR overnight initial margin for ES/MES Micro E-mini S&P 500 as
# fraction of notional. CME exchange minimum is ~3-5%; IBKR holds a
# buffer above that. ES initial ≈ $13.2k / $290k ≈ 4.6%; conservative
# value chosen here for headroom.
ES_OVERNIGHT_MARGIN_PCT_DEFAULT = 0.06

# Quarterly roll slippage cost on SPX futures. Calendar-spread quote
# typically 0.25-0.50 ES points = ~5bps of notional per roll. Four
# rolls per year (Mar/Jun/Sep/Dec) → ~20bps/yr drag.
SPX_FUTURES_ROLL_SLIPPAGE_BPS_DEFAULT = 5.0
SPX_FUTURES_ROLLS_PER_YEAR = 4

# IBKR US500 CFD overnight financing spread (long side). Per IBKR
# website: SOFR + 1.5% on long index CFDs (matches Pro margin).
IBKR_CFD_LONG_SPREAD_OVER_BENCHMARK = 0.015

# IBKR US500 CFD commission: 0.005% of notional per side. For a
# turnover-light strategy this is negligible vs financing cost.
IBKR_CFD_COMMISSION_PCT_PER_SIDE = 0.00005


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


def futures_returns(
    underlying_close: pd.Series,
    leverage: float,
    *,
    dividend_yield: float = SPX_DIVIDEND_YIELD_DEFAULT,
    margin_pct: float = ES_OVERNIGHT_MARGIN_PCT_DEFAULT,
    rolls_per_year: int = SPX_FUTURES_ROLLS_PER_YEAR,
    roll_slippage_bps: float = SPX_FUTURES_ROLL_SLIPPAGE_BPS_DEFAULT,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Daily equity-return series of a constant-L SPX futures position.

    Cost decomposition (per $E equity, per year):

      ``L × spx_price_return``                 # gross levered exposure
      ``- L × (funding - dividend_yield)``     # basis decay (cost of carry)
      ``+ T_bill_yield``                       # IBKR pays interest on free cash
      ``- L × roll_slippage``                  # quarterly roll bid/ask cost

    Sanity check at L=1: futures_return + T-bill on cash ≈ spx_total_return,
    which matches direct holding of the index. The model assumes IBKR pays
    the full benchmark rate on the cash balance net of posted margin (true
    above the $10k threshold for IBKR Pro). For ``L × margin_pct >= 1`` the
    holder would need to borrow against margin which this simple model
    rejects — use a smaller L or a CFD/margin engine for that range.

    The futures wrapper has no TER (no daily-rebalanced ETF carrying a
    management fee), no per-night borrow interest (financing is implicit
    in the basis), and no dividend cashflow (dividends are reflected in
    the spot-vs-futures spread).

    Parameters
    ----------
    underlying_close
        Close prices of the SPX cash index proxy (e.g. SPY total-return-
        adjusted, or CSPX, or a synthetic SPX series). The model treats
        this as the cash-index price; basis decay is added on top.
    leverage
        Constant target leverage (notional / equity). 1.0 = unlevered.
    dividend_yield
        SPX trailing 12-month dividend yield, decimal. Default 1.5%.
    margin_pct
        IBKR overnight initial margin / notional for ES/MES. Default 6%.
    rolls_per_year
        Number of contract rolls per year (default 4 — quarterly).
    roll_slippage_bps
        Bid/ask cost per roll, in basis points of notional. Default 5bps.
    trading_days_per_year
        Annualisation denominator (default 252).
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")
    if leverage * margin_pct >= 1.0:
        raise ValueError(
            f"leverage × margin_pct = {leverage * margin_pct:.2f} >= 1; "
            f"insufficient cash to support {leverage}x leverage at "
            f"{margin_pct:.0%} margin requirement. Use a CFD or margin engine."
        )
    spy_ret = underlying_close.pct_change()
    funding = funding_series(underlying_close.index)
    daily_funding = funding / trading_days_per_year
    daily_div = dividend_yield / trading_days_per_year
    daily_basis = daily_funding - daily_div
    daily_roll = (rolls_per_year * roll_slippage_bps / 10_000.0) / trading_days_per_year
    return (
        (leverage * spy_ret)
        - (leverage * daily_basis)
        + daily_funding  # T-bill earned on cash equity
        - (leverage * daily_roll)
    )


def cfd_returns(
    underlying_close: pd.Series,
    leverage: float,
    *,
    cfd_spread_over_benchmark: float = IBKR_CFD_LONG_SPREAD_OVER_BENCHMARK,
    commission_pct_per_side: float = IBKR_CFD_COMMISSION_PCT_PER_SIDE,
    annual_turnover: float = 1.0,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Daily equity-return series of a constant-L IBKR US500 CFD position.

    Cost decomposition (per $E equity, per year):

      ``L × spx_total_return``           # CFD pays/receives dividends as cash adj
      ``- (L - 1) × cfd_funding_rate``   # financing on the borrowed portion
      ``- L × commission × turnover``    # round-trip commission cost

    CFDs differ from leveraged ETFs in three ways:
      1. No TER (no fund wrapper).
      2. Financing is on the BORROWED portion only — at L=1 financing = 0.
      3. Dividends are paid/received as price adjustments → use
         underlying_total_return (price + dividend_yield) as the gross.

    The model approximates underlying_total_return as ``spx_price_return``
    if the input series is total-return-adjusted (e.g. CSPX) or as
    ``spx_price_return + dividend_yield/year`` if the input is price-only.
    To keep the function simple we assume the input is total-return like
    the existing ``constant_leverage_margin_returns`` does — pass a
    total-return series in.

    Cost ranking at typical strategy turnover:
      - vs ETF margin: CFD saves the TER (~7bps/yr), pays similar
        financing (IBKR Pro spread = CFD spread = SOFR+1.5%).
      - vs futures L=2: CFD costs ~6.5% on borrowed portion (= 6.5%
        per equity); futures cost ~5% × 2 - T-bill = 5% per equity.
        Futures slightly cheaper. Gap widens at higher L.
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")
    spy_ret = underlying_close.pct_change()
    funding = funding_series(underlying_close.index)
    cfd_rate = funding + cfd_spread_over_benchmark
    daily_cfd_rate = cfd_rate / trading_days_per_year
    borrow_factor = max(leverage - 1.0, 0.0)
    daily_commission = (
        leverage * commission_pct_per_side * 2.0 * annual_turnover
    ) / trading_days_per_year
    return (leverage * spy_ret) - (borrow_factor * daily_cfd_rate) - daily_commission
