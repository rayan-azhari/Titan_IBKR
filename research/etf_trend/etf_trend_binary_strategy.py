"""Generic etf_trend binary-sizing strategy module.

Used for Wave A.2 SPOT-CHECK audits on DBC and GLD (and any future
spot-check on the 5 unleveraged etf_trend variants with `sizing_mode=
"binary"` — IWB, EFA, DBC, GLD).

Mechanism (per bar t):
    1. ``sma(t) = rolling_mean(close, slow_ma)[t]``           — trend boundary.
    2. ``in_regime(t) = close[t] > sma(t)``.
    3. Exit-confirm state machine: stay long until close < SMA for
       ``exit_confirm_days`` consecutive bars.
    4. **Binary** sizing — position is 0 or 1.0 (the L17 question is
       whether the signal beats B&H; vol-target sizing is a separate
       concern handled inside the live strategy class).
    5. ``net = position.shift(1) * log_return - cost``.

Causality (L04 / A1 / L18): SMA at t uses [t-slow_ma+1, t]; position
shift by 1.

Same shape as ``etf_trend_spy_strategy.py`` and ``etf_trend_tqqq_strategy
.py`` — chosen interface so the audit harness can mix-and-match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EtfTrendBinaryConfig:
    """Config for the binary-sizing variant.

    The audit cells supply ``symbol`` to name the underlying. The
    strategy_fn ignores any extra columns in ``closes_df`` outside
    ``symbol``.
    """

    symbol: str  # e.g. "DBC", "GLD"
    slow_ma: int = 200
    exit_confirm_days: int = 1
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0


def _apply_exit_confirm(above_sma: pd.Series, *, exit_confirm_days: int) -> pd.Series:
    """Exit-confirm state machine (same as SPY/TQQQ variants)."""
    out = above_sma.copy()
    arr = above_sma.to_numpy()
    sig = np.zeros(len(above_sma), dtype=float)
    pos = 0
    days_below = 0
    for i in range(len(above_sma)):
        if np.isnan(arr[i]):
            sig[i] = float(pos)
            continue
        if pos == 0 and arr[i] == 1.0:
            pos = 1
            days_below = 0
        elif pos == 1:
            if arr[i] == 0.0:
                days_below += 1
                if days_below >= exit_confirm_days:
                    pos = 0
                    days_below = 0
            else:
                days_below = 0
        sig[i] = float(pos)
    out.iloc[:] = sig
    return out


def etf_trend_binary_returns(
    closes_df: pd.DataFrame, *, cfg: EtfTrendBinaryConfig | None = None
) -> pd.Series:
    """Per-bar net return for the binary-sizing trend strategy."""
    if cfg is None:
        raise ValueError("EtfTrendBinaryConfig required (symbol must be set)")
    # Accept either symbol-named column OR 'close' (for MC bootstrap dfs).
    if cfg.symbol in closes_df.columns:
        price = closes_df[cfg.symbol]
    elif "close" in closes_df.columns:
        price = closes_df["close"]
    else:
        raise ValueError(
            f"etf_trend_binary_returns requires column '{cfg.symbol}' or 'close'; "
            f"got {list(closes_df.columns)}"
        )

    sma = price.rolling(cfg.slow_ma, min_periods=cfg.slow_ma).mean()
    above = (price > sma).astype(float)
    position = _apply_exit_confirm(above, exit_confirm_days=cfg.exit_confirm_days)

    log_ret = np.log(price / price.shift(1)).fillna(0.0)
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * log_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def buy_and_hold_binary_returns(
    closes_df: pd.DataFrame, *, cfg: EtfTrendBinaryConfig | None = None
) -> pd.Series:
    """Always-long buy-and-hold returns of the underlying — L17 benchmark."""
    if cfg is None:
        raise ValueError("EtfTrendBinaryConfig required (symbol must be set)")
    if cfg.symbol in closes_df.columns:
        price = closes_df[cfg.symbol]
    elif "close" in closes_df.columns:
        price = closes_df["close"]
    else:
        raise ValueError(
            f"buy_and_hold_binary_returns requires column '{cfg.symbol}' or 'close'; "
            f"got {list(closes_df.columns)}"
        )
    log_ret = np.log(price / price.shift(1)).fillna(0.0)
    held_lagged = pd.Series(1.0, index=log_ret.index)
    held_lagged.iloc[0] = 0.0
    return (held_lagged * log_ret).rename("ret")


def etf_trend_binary_assert_causal(
    closes_df: pd.DataFrame, *, cfg: EtfTrendBinaryConfig | None = None
) -> None:
    """L04 smoke test — corrupting future closes must not change past returns."""
    if cfg is None:
        raise ValueError("EtfTrendBinaryConfig required")
    base = etf_trend_binary_returns(closes_df, cfg=cfg)
    rng = np.random.default_rng(42)
    t_corrupt = int(rng.integers(cfg.slow_ma + 50, len(closes_df) - 50))
    corrupted = closes_df.copy()
    corrupted.iloc[t_corrupt:] = np.nan
    corrupted_ret = etf_trend_binary_returns(corrupted, cfg=cfg)

    base_past = base.iloc[: t_corrupt - 1].dropna()
    corrupted_past = corrupted_ret.iloc[: t_corrupt - 1].dropna()
    common = base_past.index.intersection(corrupted_past.index)
    if len(common) == 0:
        raise AssertionError("Causality test could not find common past index")
    diffs = (base_past.reindex(common) - corrupted_past.reindex(common)).abs()
    max_diff = float(diffs.max())
    if max_diff > 1e-12:
        n_changed = int((diffs > 1e-12).sum())
        raise AssertionError(
            f"Causality smoke failed: corrupting closes at t={t_corrupt} "
            f"changed {n_changed} past returns (max |delta|={max_diff:.2e})"
        )


__all__ = [
    "EtfTrendBinaryConfig",
    "etf_trend_binary_returns",
    "buy_and_hold_binary_returns",
    "etf_trend_binary_assert_causal",
]
