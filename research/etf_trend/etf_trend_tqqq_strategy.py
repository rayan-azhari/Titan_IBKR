"""etf_trend TQQQ — pure-research strategy module for V3.6 re-audit.

Specified in (forthcoming) ``directives/Pre-Reg etf_trend TQQQ Re-audit 2026-05-16.md``.

Mechanism (per bar t):

    1. ``sma(t) = rolling_mean(qqq.close, slow_ma)[t]``       — trend boundary on QQQ.
    2. ``in_regime(t) = qqq.close[t] > sma(t)``.
    3. Exit-confirm state machine on QQQ signal.
    4. **Binary** sizing (no vol-target) — position is 0 or 1.0 in TQQQ.
    5. TQQQ's 3x leverage provides position sizing implicitly.
    6. ``net = position.shift(1) * tqqq_log_return - cost``.

Causality (L04 / A1 / L18): SMA at t-1, position shift by 1.

Parity contract: the sweep
(`research/exploration/sweep_etf_trend_tqqq.py::etf_trend_tqqq_returns`)
and the audit (`etf_trend_tqqq_returns()` here) MUST produce identical
per-bar returns for identical inputs.

Buy-and-hold baseline (`buy_and_hold_tqqq_returns`) is the benchmark for
the L17 relative MC test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EtfTrendTqqqConfig:
    """One row of the etf_trend_tqqq V3.6 pre-reg cell grid."""

    slow_ma: int = 150
    exit_confirm_days: int = 1
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0


def _apply_exit_confirm(above_sma: pd.Series, *, exit_confirm_days: int) -> pd.Series:
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


def etf_trend_tqqq_returns(
    closes_df: pd.DataFrame, *, cfg: EtfTrendTqqqConfig | None = None
) -> pd.Series:
    """Per-bar net return — signal from QQQ, binary position in TQQQ (3x)."""
    if cfg is None:
        cfg = EtfTrendTqqqConfig()
    if "QQQ" not in closes_df.columns or "TQQQ" not in closes_df.columns:
        raise ValueError(
            f"etf_trend_tqqq_returns requires columns 'QQQ' and 'TQQQ'; "
            f"got {list(closes_df.columns)}"
        )

    qqq = closes_df["QQQ"]
    tqqq = closes_df["TQQQ"]
    sma = qqq.rolling(cfg.slow_ma, min_periods=cfg.slow_ma).mean()
    above = (qqq > sma).astype(float)
    position = _apply_exit_confirm(above, exit_confirm_days=cfg.exit_confirm_days)

    tqqq_ret = np.log(tqqq / tqqq.shift(1)).fillna(0.0)
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * tqqq_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def buy_and_hold_tqqq_returns(
    closes_df: pd.DataFrame, *, cfg: EtfTrendTqqqConfig | None = None
) -> pd.Series:
    """Always-long TQQQ — L17 relative-MC benchmark.

    No vol-target (TQQQ has internal 3x leverage). Position = 1.0 always.
    """
    if cfg is None:
        cfg = EtfTrendTqqqConfig()
    if "TQQQ" not in closes_df.columns:
        raise ValueError(
            f"buy_and_hold_tqqq_returns requires column 'TQQQ'; got {list(closes_df.columns)}"
        )
    tqqq = closes_df["TQQQ"]
    tqqq_ret = np.log(tqqq / tqqq.shift(1)).fillna(0.0)
    # Long-only at 1.0; held_lagged is 1.0 except for the first bar.
    held_lagged = pd.Series(1.0, index=tqqq_ret.index)
    held_lagged.iloc[0] = 0.0
    return (held_lagged * tqqq_ret).rename("ret")


def etf_trend_tqqq_assert_causal(
    closes_df: pd.DataFrame, *, cfg: EtfTrendTqqqConfig | None = None
) -> None:
    """L04 smoke test: corrupting future closes must not change past returns."""
    if cfg is None:
        cfg = EtfTrendTqqqConfig()
    base = etf_trend_tqqq_returns(closes_df, cfg=cfg)
    rng = np.random.default_rng(42)
    t_corrupt = int(rng.integers(cfg.slow_ma + 50, len(closes_df) - 50))
    corrupted = closes_df.copy()
    corrupted.iloc[t_corrupt:] = np.nan
    corrupted_ret = etf_trend_tqqq_returns(corrupted, cfg=cfg)

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
    "EtfTrendTqqqConfig",
    "etf_trend_tqqq_returns",
    "buy_and_hold_tqqq_returns",
    "etf_trend_tqqq_assert_causal",
]
