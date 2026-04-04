"""pairs_trading.py — EUR/USD vs GBP/USD cointegrated spread module.

Strategy: long EUR/USD while short GBP/USD (or vice versa) when their
spread diverges beyond an empirical percentile threshold.  The two-legged
structure hedges out directional USD exposure — P&L accrues only from the
spread normalising, not from USD direction.

Uses the same VWAP-percentile-regime infrastructure as the single-pair
mean reversion strategy.

Cointegration:
  - Engle-Granger test on IS data to confirm cointegration.
  - Hedge ratio β estimated via OLS regression: EURUSD ~ β × GBPUSD + ε
  - Spread = EURUSD_close - β × GBPUSD_close
  - β re-estimated in each WFO IS window to track drift.

IMPORTANT: β must be re-estimated per IS window in WFO.  A fixed β
over the full history is a common failure mode in pairs strategies.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# ---------------------------------------------------------------------------
# Cointegration test
# ---------------------------------------------------------------------------


def test_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
    significance: float = 0.05,
) -> dict:
    """Engle-Granger cointegration test on IS data.

    Args:
        series_a: Primary series (e.g., EUR/USD close).
        series_b: Secondary series (e.g., GBP/USD close).
        significance: P-value threshold (reject H0 of no cointegration).

    Returns:
        Dict with t_stat, p_value, crit_values, and is_cointegrated.
    """
    t_stat, p_value, crit_values = coint(series_a, series_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "crit_values": [float(c) for c in crit_values],
        "is_cointegrated": float(p_value) < significance,
    }


# ---------------------------------------------------------------------------
# Hedge ratio estimation
# ---------------------------------------------------------------------------


def estimate_hedge_ratio(
    series_a: pd.Series,
    series_b: pd.Series,
) -> float:
    """OLS regression: series_a ~ β × series_b + ε.

    Returns:
        β coefficient (hedge ratio).
    """
    X = sm.add_constant(series_b.values)
    result = sm.OLS(series_a.values, X).fit()
    beta = float(result.params[1])
    return beta


def compute_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    beta: float,
) -> pd.Series:
    """Compute the cointegrated spread: a - β × b.

    Args:
        series_a: Primary series (already aligned).
        series_b: Secondary series (already aligned).
        beta: Hedge ratio from estimate_hedge_ratio().

    Returns:
        Spread series.
    """
    return series_a - beta * series_b


# ---------------------------------------------------------------------------
# Spread percentile entry levels (reuses signals.py logic)
# ---------------------------------------------------------------------------


def spread_percentile_levels(
    spread: pd.Series,
    window: int,
    pcts: list[float],
) -> pd.DataFrame:
    """Rolling empirical percentile bands of the spread distribution.

    Identical to signals.percentile_levels() but operates on the spread
    rather than the VWAP deviation.  Already shift-1.

    Args:
        spread: Cointegrated spread series.
        window: Rolling lookback in bars.
        pcts: Upper-tail percentiles.

    Returns:
        DataFrame of threshold levels (columns = percentile values).
    """
    cols = {}
    for p in pcts:
        cols[p] = spread.rolling(window).quantile(p).shift(1)
    return pd.DataFrame(cols, index=spread.index)


# ---------------------------------------------------------------------------
# Entry and exit signal builder
# ---------------------------------------------------------------------------


def build_pairs_signals(
    spread: pd.Series,
    levels: pd.DataFrame,
    regime_gate: pd.Series,
    tier1_pct: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build entry/exit signals for the pairs trade.

    Spread > 95th pct -> pair is too wide -> short the spread
      (short EUR/USD, long GBP/USD).
    Spread < 5th pct  -> pair is too narrow -> long the spread
      (long EUR/USD, short GBP/USD).
    Exit when spread reverts toward zero.

    Args:
        spread: Cointegrated spread series.
        levels: From spread_percentile_levels().
        regime_gate: Boolean Series from regime.regime_gate().
        tier1_pct: First-tier percentile (e.g., 0.90).

    Returns:
        (short_entry, short_exit, long_entry, long_exit) Boolean Series.
    """
    upper = levels[tier1_pct]

    # Invert for lower tail: lower percentile = 1 - tier1_pct
    lower_pct = round(1.0 - tier1_pct, 3)
    if lower_pct in levels.columns:
        lower = levels[lower_pct]
    else:
        lower = spread.rolling(int(spread.rolling(1).count().max())).quantile(lower_pct).shift(1)

    short_entry = (spread > upper) & regime_gate
    short_exit = spread < 0

    long_entry = (spread < lower) & regime_gate
    long_exit = spread > 0

    return short_entry, short_exit, long_entry, long_exit


# ---------------------------------------------------------------------------
# Pairs backtest runner (VectorBT)
# ---------------------------------------------------------------------------


def run_pairs_backtest(
    spread: pd.Series,
    short_entry: pd.Series,
    short_exit: pd.Series,
    long_entry: pd.Series,
    long_exit: pd.Series,
    spread_cost: pd.Series,
    label: str = "pairs",
) -> dict:
    """VectorBT backtest on the spread series.

    In a real pairs execution, two orders are placed simultaneously.
    Here we model the P&L on the spread series directly, with double
    transaction costs (two legs to enter and exit).

    Args:
        spread: Spread series (acts as the 'price' in the portfolio).
        short_entry / short_exit: Short spread signals.
        long_entry / long_exit: Long spread signals.
        spread_cost: 2× combined spread from both legs.
        label: Human-readable label.

    Returns:
        Dict of performance metrics.
    """
    import vectorbt as vbt

    pf = vbt.Portfolio.from_signals(
        spread,
        entries=long_entry,
        exits=long_exit,
        short_entries=short_entry,
        short_exits=short_exit,
        init_cash=10_000,
        fees=spread_cost.values,
        freq="5min",
    )
    n_trades = pf.trades.count()
    weeks = (spread.index[-1] - spread.index[0]).days / 7
    return {
        "label": label,
        "sharpe": round(float(pf.sharpe_ratio()), 3),
        "max_dd": round(float(pf.max_drawdown()), 4),
        "n_trades": int(n_trades),
        "trades_per_week": round(n_trades / max(weeks, 1), 2),
        "win_rate": round(float(pf.trades.win_rate()), 3),
    }
