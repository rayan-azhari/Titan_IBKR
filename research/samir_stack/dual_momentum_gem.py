"""GEM (Global Equities Momentum) signal layer.

Implements the dual-momentum framework from Antonacci's GEM model
(see ``resources/dual_momentum.md``):

  1. **Absolute momentum** (regime gate): is SPY's trailing-12-month
     EXCESS return (over the risk-free rate) positive? If yes, the
     equity regime is "ON". If no, default to safe-harbour (bonds).

  2. **Relative momentum** (asset selection): when "ON", pick whichever
     of SPY or EFA (international-developed proxy for ACWI ex-US) had
     the higher trailing-12-month total return.

GEM as a standalone runs ~1.5 trades/year and has historically reduced
SPY's max-DD from ~-51% to ~-18% with higher CAGR.

This module exposes:

  - ``gem_signal(spy, efa, tbill_annual_rate)`` → DataFrame with the
    daily signal (asset choice + abs/rel momentum scores)
  - ``compose_with_samir_regime(gem_df, samir_score)`` → combined gate

The combination logic is the new contribution: GEM's 12-month signal
is much SLOWER than Samir's 21-day regime score. They're complementary
— GEM filters out broad bear regimes that the faster gate may not catch
on its own; Samir's faster gate catches sharp drawdowns within an
otherwise-bullish 12-month window (e.g. February 2018, March 2020).

Reference: ``resources/dual_momentum.md``,
``directives/Samir-Stack Margin Variant 2026-05-11.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.synthetic_3x import funding_series


def absolute_momentum(
    close: pd.Series,
    *,
    lookback_days: int = 252,
    tbill_annual_rate: pd.Series | None = None,
) -> pd.DataFrame:
    """12-month excess return vs risk-free rate.

    Parameters
    ----------
    close
        Close-price series (daily).
    lookback_days
        Window for the trailing return. Default 252 = ~12 months.
    tbill_annual_rate
        Annualised risk-free rate, indexed to ``close``. If None, uses
        the ``funding_series`` (Fed Funds approx) as a T-bill proxy.

    Returns
    -------
    pd.DataFrame
        Columns: ``ret_12m``, ``rf_12m``, ``excess``, ``in_market`` (bool).
    """
    if tbill_annual_rate is None:
        tbill_annual_rate = funding_series(close.index)
    ret_12m = close.pct_change(lookback_days)
    # 12-month accumulated risk-free return ≈ rate * (lookback_days / 252)
    rf_12m = tbill_annual_rate * (lookback_days / 252.0)
    excess = ret_12m - rf_12m
    return pd.DataFrame(
        {
            "ret_12m": ret_12m,
            "rf_12m": rf_12m,
            "excess": excess,
            "in_market": excess > 0,
        }
    )


def relative_momentum(
    spy_close: pd.Series,
    efa_close: pd.Series,
    *,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Pick the higher 12m total-return between SPY and EFA.

    Returns DataFrame with ``spy_12m``, ``efa_12m``, ``winner`` ('SPY'|'EFA').
    """
    spy_12m = spy_close.pct_change(lookback_days)
    efa_12m = efa_close.pct_change(lookback_days)
    # Align
    common = spy_12m.index.intersection(efa_12m.index)
    spy_12m = spy_12m.reindex(common)
    efa_12m = efa_12m.reindex(common)
    winner = pd.Series(
        np.where(spy_12m >= efa_12m, "SPY", "EFA"),
        index=common,
        name="winner",
    )
    # NaN handling: if either is NaN, pick the non-NaN one; if both NaN, "SPY"
    both_na = spy_12m.isna() & efa_12m.isna()
    winner.loc[both_na] = "SPY"
    only_spy = spy_12m.isna() & efa_12m.notna()
    only_efa = spy_12m.notna() & efa_12m.isna()
    winner.loc[only_spy] = "EFA"
    winner.loc[only_efa] = "SPY"
    return pd.DataFrame({"spy_12m": spy_12m, "efa_12m": efa_12m, "winner": winner})


def gem_signal(
    spy_close: pd.Series,
    efa_close: pd.Series,
    *,
    tbill_annual_rate: pd.Series | None = None,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Full GEM signal: combines absolute + relative momentum.

    Returns DataFrame with columns:

      - ``asset_choice``: 'SPY' / 'EFA' / 'BONDS'
      - ``equity_in_market``: bool (abs-mom gate on SPY)
      - ``rel_winner``: 'SPY' / 'EFA' (rel-mom winner — only meaningful
        when ``equity_in_market`` is True)
      - ``spy_excess``: SPY's 12m excess return
      - ``efa_excess``: EFA's 12m excess return
    """
    if tbill_annual_rate is None:
        tbill_annual_rate = funding_series(spy_close.index)
    abs_spy = absolute_momentum(
        spy_close, lookback_days=lookback_days, tbill_annual_rate=tbill_annual_rate
    )
    abs_efa = absolute_momentum(
        efa_close, lookback_days=lookback_days, tbill_annual_rate=tbill_annual_rate
    )
    rel = relative_momentum(spy_close, efa_close, lookback_days=lookback_days)

    common = abs_spy.index.intersection(abs_efa.index).intersection(rel.index)
    abs_spy = abs_spy.reindex(common)
    abs_efa = abs_efa.reindex(common)
    rel = rel.reindex(common)

    # Antonacci's original GEM uses ONLY SPY's abs-mom as the gate. The
    # rel-mom winner is then picked from {SPY, EFA}. Some variants
    # require BOTH abs-mom > 0 to enter equities at all; we use the
    # original (single-gate) form for fidelity to the framework.
    equity_in = abs_spy["in_market"]
    asset = pd.Series("BONDS", index=common)
    asset.loc[equity_in] = rel["winner"][equity_in].values

    return pd.DataFrame(
        {
            "asset_choice": asset,
            "equity_in_market": equity_in,
            "rel_winner": rel["winner"],
            "spy_excess": abs_spy["excess"],
            "efa_excess": abs_efa["excess"],
        }
    )


def compose_with_samir_regime(
    gem_df: pd.DataFrame,
    samir_score: pd.Series,
    *,
    samir_threshold: float = 0.30,
) -> pd.Series:
    """Combine GEM's slow gate with Samir's faster regime score.

    Returns an "effective regime score" for use in Samir-Stack's tier
    logic. Two gates must BOTH agree to give the equity sleeve any
    weight:

      - GEM's 12m abs-mom must be in-market (slow gate)
      - Samir's regime score must clear ``samir_threshold`` (fast gate)

    When GEM kicks out, the effective score is forced to 0 (regime gate
    cash). When GEM is in but Samir is bearish, Samir wins.

    The result feeds directly into ``run_stacked_strategy`` as if it
    were the Samir score, so all of Samir's downstream logic
    (capitulation, DD breaker, hysteresis) keeps operating on the
    composed signal.
    """
    common = gem_df.index.intersection(samir_score.index)
    out = samir_score.reindex(common).copy()
    gem_in = gem_df["equity_in_market"].reindex(common).fillna(False)
    out.loc[~gem_in] = 0.0
    return out
