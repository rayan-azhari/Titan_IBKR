"""Cross-sectional country-momentum strategy with WFO + bootstrap CI.

Classical Asness-Moskowitz country momentum: at each rebalance date,
rank the universe by N-day return, long the top-k, short the bottom-k,
equal-weighted. Monthly rebalance by default.

WFO structure:
  IS: estimate nothing (cross-sectional ranking is stateless)
  OOS: run the rank/trade rule on held-out period, compute Sharpe
  Fold width: ~1 year each, stitched returns used for the overall
    Sharpe + bootstrap CI.

API:

    run_country_wfo(
        instruments={"EWG": series, "EWU": series, ...},
        lookback_days=252,           # 12m momentum
        top_k=2,                     # long the top 2
        bottom_k=2,                  # short the bottom 2 (set 0 for long-only)
        rebalance_days=21,           # monthly
        cost_bps=5.0,
        is_days=504, oos_days=252,
    ) -> dict

Result dict keys mirror `run_bond_wfo`:
  stitched_sharpe, sharpe_ci_95_lo, sharpe_ci_95_hi, stitched_dd_pct,
  n_folds, pct_positive, total_trades, stitched_returns
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _rank_momentum(
    prices: pd.DataFrame, lookback: int, top_k: int, bottom_k: int
) -> pd.DataFrame:
    """At each row, rank instruments by `lookback`-day log return. Output
    rows are the same as prices.index, cols the same as prices.columns.
    Values: +1/top_k for top-k, -1/bottom_k for bottom-k, 0 otherwise."""
    logp = np.log(prices)
    mom = logp - logp.shift(lookback)

    n = prices.shape[1]
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ts, row in mom.iterrows():
        vals = row.dropna()
        if len(vals) < (top_k + bottom_k):
            continue
        sorted_idx = vals.sort_values(ascending=False).index
        if top_k > 0:
            for c in sorted_idx[:top_k]:
                weights.at[ts, c] = 1.0 / top_k
        if bottom_k > 0:
            for c in sorted_idx[-bottom_k:]:
                weights.at[ts, c] = -1.0 / bottom_k
    return weights


def _stitched_oos_returns(
    prices: pd.DataFrame,
    lookback: int,
    top_k: int,
    bottom_k: int,
    rebalance_days: int,
    is_days: int,
    oos_days: int,
    cost_bps: float,
) -> tuple[pd.Series, list[float], int]:
    """Returns (stitched daily return series on OOS folds, per-fold Sharpe
    list, total trade count)."""
    dates = prices.index
    n = len(dates)
    if n < is_days + oos_days:
        return pd.Series(dtype=float), [], 0

    # Daily returns, NaN→0
    rets = prices.pct_change().fillna(0.0)

    stitched_parts: list[pd.Series] = []
    fold_sharpes: list[float] = []
    total_trades = 0

    oos_start = is_days
    while oos_start + oos_days <= n:
        oos_slice = slice(oos_start, oos_start + oos_days)
        # Compute weights for all dates up to end of OOS — needed because
        # lookback stretches back into IS.
        px_upto = prices.iloc[: oos_start + oos_days]
        weights = _rank_momentum(px_upto, lookback, top_k, bottom_k)
        # Shift positions by 1 day: decision at close t earns return t->t+1.
        # Hold constant between rebalance dates.
        rebalance_mask = pd.Series(False, index=px_upto.index)
        rebalance_mask.iloc[::rebalance_days] = True
        # Forward-fill weights between rebalance dates.
        held_weights = weights.where(rebalance_mask).ffill().fillna(0.0)
        # Shift for causality.
        held_weights = held_weights.shift(1).fillna(0.0)

        # Portfolio returns = sum(weight * instrument_return)
        port_rets_full = (held_weights * rets.loc[held_weights.index]).sum(axis=1)
        port_rets = port_rets_full.iloc[oos_slice]

        # Costs: charge on weight change.
        weight_change = held_weights.diff().abs().sum(axis=1).iloc[oos_slice]
        cost = weight_change * (cost_bps / 10000.0)
        port_rets_net = port_rets - cost

        # Count trades: any bar where any weight changed.
        n_trades = int((weight_change > 0).sum())
        total_trades += n_trades

        stitched_parts.append(port_rets_net)
        sd = port_rets_net.std()
        if sd > 1e-12:
            fold_sharpes.append(
                float(port_rets_net.mean() / sd * np.sqrt(252))
            )
        oos_start += oos_days

    stitched = pd.concat(stitched_parts) if stitched_parts else pd.Series(dtype=float)
    return stitched, fold_sharpes, total_trades


def run_country_wfo(
    instruments: dict[str, pd.Series],
    lookback: int = 252,
    top_k: int = 2,
    bottom_k: int = 2,
    rebalance_days: int = 21,
    is_days: int = 504,
    oos_days: int = 252,
    cost_bps: float = 5.0,
) -> dict:
    """Walk-forward country-momentum rotation.

    Returns dict with the same keys as `run_bond_equity_wfo.run_bond_wfo`:
      stitched_sharpe, sharpe_ci_95_lo, sharpe_ci_95_hi, stitched_dd_pct,
      n_folds, pct_positive, total_trades, stitched_returns.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci
    from titan.research.metrics import sharpe as _sharpe

    # Align all series on common index.
    prices = pd.concat(instruments, axis=1)
    prices = prices.dropna(how="all")
    # Forward-fill short gaps (ETF holiday mismatches).
    prices = prices.ffill().dropna(how="any")

    stitched, fold_sharpes, total_trades = _stitched_oos_returns(
        prices,
        lookback,
        top_k,
        bottom_k,
        rebalance_days,
        is_days,
        oos_days,
        cost_bps,
    )

    if stitched.empty:
        return {
            "stitched_sharpe": 0.0,
            "sharpe_ci_95_lo": 0.0,
            "sharpe_ci_95_hi": 0.0,
            "stitched_dd_pct": 0.0,
            "n_folds": 0,
            "pct_positive": 0.0,
            "total_trades": 0,
            "stitched_returns": stitched,
        }

    pp = BARS_PER_YEAR["D"]
    sharpe = float(_sharpe(stitched, periods_per_year=pp))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=pp, n_resamples=1000, seed=42
    )
    eq = (1.0 + stitched).cumprod()
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    pct_pos = float(np.mean([s > 0 for s in fold_sharpes])) if fold_sharpes else 0.0

    return {
        "stitched_sharpe": sharpe,
        "sharpe_ci_95_lo": ci_lo,
        "sharpe_ci_95_hi": ci_hi,
        "stitched_dd_pct": dd,
        "n_folds": len(fold_sharpes),
        "pct_positive": pct_pos,
        "total_trades": total_trades,
        "stitched_returns": stitched,
    }
