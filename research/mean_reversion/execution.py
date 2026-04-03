"""execution.py — Exponential grid builder and basket VWAP exit logic.

Grid design:
  - n tiers, each triggered at a progressively more extreme percentile level.
  - Lot sizes follow an exponential schedule [1, 2, 4, 8, ...] so that
    deeper, higher-conviction trades carry more weight.
  - The break-even price of the combined basket is pulled toward the extremes,
    meaning less mean-reversion is needed before the whole basket is profitable.

Basket exit:
  - Track the volume-weighted average entry price of the entire open basket.
  - Issue a single combined exit signal when price crosses:
      basket_vwap_entry ± profit_margin  (direction-aware)
  - This is pre-computed as a boolean Series before being fed to VectorBT.

VectorBT approach:
  - N sub-portfolios, one per tier, with init_cash proportional to tier size.
  - combined_exit = basket_exit | invalidation_exit | time_exit
  - Each sub-portfolio receives the same combined_exit boolean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tier size schedules
# ---------------------------------------------------------------------------

# Exponential (aggressive break-even compression)
TIER_SIZES_EXP = [1, 2, 4, 8, 8, 8, 8]

# Linear (moderate)
TIER_SIZES_LIN = [1, 2, 3, 4, 5, 6, 7]

# Flat (equal weight baseline — for comparison)
TIER_SIZES_FLAT = [1, 1, 1, 1, 1, 1, 1]


def get_tier_sizes(schedule: str, n_tiers: int) -> list[int]:
    """Return tier sizes for a given schedule name.

    Args:
        schedule: One of "exp", "lin", "flat".
        n_tiers: Number of tiers (truncates or pads schedule to fit).

    Returns:
        List of integer lot sizes, length == n_tiers.
    """
    schedules = {
        "exp": TIER_SIZES_EXP,
        "lin": TIER_SIZES_LIN,
        "flat": TIER_SIZES_FLAT,
    }
    base = schedules.get(schedule, TIER_SIZES_EXP)
    # Extend by repeating last value if needed
    while len(base) < n_tiers:
        base = base + [base[-1]]
    return base[:n_tiers]


# ---------------------------------------------------------------------------
# Grid entry signals
# ---------------------------------------------------------------------------


def build_grid_entries(
    deviation: pd.Series,
    levels: pd.DataFrame,
    gate: pd.Series,
    tier_sizes: list[int],
) -> list[dict]:
    """Build per-tier entry/exit signals gated by the regime filter.

    Each tier is a dict with keys:
      long_entry   : Boolean Series — open long (price too far below VWAP)
      short_entry  : Boolean Series — open short (price too far above VWAP)
      size         : int — lot weight for this tier

    The deviation series and levels DataFrame are already shift-1.
    Entry is triggered on a CROSS (not a level-hold), so we detect the bar
    where deviation first exceeds the threshold.

    Args:
        deviation: close - anchored_vwap (shift-1 applied).
        levels: DataFrame of positive thresholds (columns = tier labels or pcts).
                Short side uses these directly; long side uses their negatives.
        gate: Boolean Series — True when regime allows entries.
        tier_sizes: Lot weights per tier.

    Returns:
        List of dicts, one per tier, ordered from lowest to highest threshold.
    """
    cols = list(levels.columns)
    entries = []
    for i, col in enumerate(cols):
        lvl = levels[col]
        size = tier_sizes[i] if i < len(tier_sizes) else tier_sizes[-1]

        # Cross detection: deviation crosses UP through lvl (short entry)
        short_cross = (deviation > lvl) & (deviation.shift(1) <= lvl.shift(1))
        # Cross detection: deviation crosses DOWN through -lvl (long entry)
        long_cross = (deviation < -lvl) & (deviation.shift(1) >= -lvl.shift(1))

        entries.append({
            "tier": col,
            "short_entry": short_cross & gate,
            "long_entry": long_cross & gate,
            "size": size,
        })
    return entries


# ---------------------------------------------------------------------------
# Basket VWAP exit (bar-by-bar simulation)
# ---------------------------------------------------------------------------


def compute_basket_vwap_exit(
    close: pd.Series,
    grid_entries: list[dict],
    profit_margin: float,
    direction: str = "both",
) -> pd.Series:
    """Simulate basket VWAP tracking and generate exit signal.

    For each direction (short/long) independently:
      - Track open lots and their entry prices bar by bar.
      - Compute basket_entry_vwap = Σ(size_i × price_i) / Σ(size_i).
      - Emit exit signal when:
          short basket: close <= basket_entry_vwap - profit_margin
          long  basket: close >= basket_entry_vwap + profit_margin
      - Exit also fires when the basket is empty (no open positions).

    This cannot be fully vectorised because basket state is path-dependent.

    Args:
        close: Close price series.
        grid_entries: Output of build_grid_entries().
        profit_margin: Profit target in price units (e.g., 0.0005 = 5 pips).
        direction: "both", "short_only", or "long_only".

    Returns:
        Boolean Series — True means exit entire basket this bar.
    """
    n = len(close)
    close_vals = close.values
    exit_signal = np.zeros(n, dtype=bool)

    for side in (["short", "long"] if direction == "both" else [direction.replace("_only", "")]):
        cum_cost = 0.0
        cum_units = 0
        basket_active = False

        for i in range(n):
            price = close_vals[i]

            # Check new entries this bar
            for tier in grid_entries:
                entry_key = f"{side}_entry"
                if entry_key not in tier:
                    continue
                entry_arr = tier[entry_key].values
                if i < len(entry_arr) and entry_arr[i]:
                    size = tier["size"]
                    cum_cost += size * price
                    cum_units += size
                    basket_active = True

            if not basket_active or cum_units == 0:
                continue

            basket_vwap_entry = cum_cost / cum_units

            # TP condition
            if side == "short":
                tp_hit = price <= basket_vwap_entry - profit_margin
            else:
                tp_hit = price >= basket_vwap_entry + profit_margin

            if tp_hit:
                exit_signal[i] = True
                cum_cost = 0.0
                cum_units = 0
                basket_active = False

    return pd.Series(exit_signal, index=close.index, name="basket_exit")


# ---------------------------------------------------------------------------
# VectorBT sub-portfolio builder
# ---------------------------------------------------------------------------


def build_subportfolios(
    close: pd.Series,
    grid_entries: list[dict],
    long_combined_exit: pd.Series,
    short_combined_exit: pd.Series,
    spread_series: pd.Series,
    total_cash: float = 10_000.0,
) -> list:  # list[vbt.Portfolio]
    """Build one VectorBT Portfolio per tier, cash-weighted by lot size.

    Returns a list of portfolios that can be combined (sum daily returns)
    for aggregate performance metrics.

    NOTE: Imports vectorbt lazily to avoid hard dependency at module import time.

    Args:
        close: Close price series.
        grid_entries: Output of build_grid_entries().
        long_combined_exit: Exit signal for long positions (partial reversion
            + basket TP + invalidation + time exit).
        short_combined_exit: Exit signal for short positions (direction-aware).
        spread_series: Session-aware spread Series from build_spread_series().
        total_cash: Total notional capital to distribute across tiers.

    Returns:
        List of vbt.Portfolio objects, one per tier.
    """
    import vectorbt as vbt  # lazy import

    total_units = sum(t["size"] for t in grid_entries)
    sub_pfs = []

    for tier in grid_entries:
        weight = tier["size"] / total_units
        cash = total_cash * weight

        pf = vbt.Portfolio.from_signals(
            close,
            entries=tier["long_entry"],
            exits=long_combined_exit,
            short_entries=tier["short_entry"],
            short_exits=short_combined_exit,
            init_cash=cash,
            fees=spread_series.values,
            freq="5min",
        )
        sub_pfs.append(pf)

    return sub_pfs


def combine_portfolio_returns(sub_pfs: list) -> pd.Series:
    """Combine sub-portfolio equity curves into a single daily return series.

    Args:
        sub_pfs: List of vbt.Portfolio objects.

    Returns:
        Daily return Series of the combined basket.
    """
    daily_returns = [pf.returns().resample("1D").sum() for pf in sub_pfs]
    combined = pd.concat(daily_returns, axis=1).sum(axis=1) / len(sub_pfs)
    return combined


def compute_combined_sharpe(daily_returns: pd.Series, freq: int = 252) -> float:
    """Sharpe ratio from daily returns.

    Args:
        daily_returns: Daily return Series.
        freq: Annualisation factor (252 trading days).

    Returns:
        Annualised Sharpe ratio.
    """
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return float(mu / sigma * np.sqrt(freq))
