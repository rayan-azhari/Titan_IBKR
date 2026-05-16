"""turtle -- pure-research strategy module for the V3.6 re-audit.

Specified in ``directives/Pre-Reg turtle Re-audit 2026-05-16.md``.

Mechanism (per bar t, long-only Donchian breakout on H1):

    1. Entry: close > Donchian_high(entry_period).shift(1) flips state 0 -> 1.
    2. Exit:  close < Donchian_low(exit_period).shift(1)  flips state 1 -> 0.
    3. Position: binary (single unit; pyramiding is a downstream sizing
       overlay tested separately).
    4. Cost: 1 bp per turnover (US equity H1, conservative).
    5. net = position.shift(1) * log_return - cost.

Causality (L04 / A1 / L21):
    - Donchian channels use .shift(1) so the breakout test at t uses
      past-only OHLC.
    - Position decided at close[t] earns t->t+1 return.
    - Smoke test asserts past returns are bit-exact unchanged when
      future bars are corrupted.

Annualisation: US equity RTH H1 cadence = 1,764 bars/year (7 bars/day
x 252 days). Do NOT use BARS_PER_YEAR["H1"] = 6048 (FX 24/7).

Parity contract: the sweep (`research/exploration/sweep_turtle.py`)
and the audit (`turtle_returns()`) MUST produce identical per-bar
returns for identical inputs at any cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS_PER_YEAR_H1_EQ = 1764


@dataclass(frozen=True)
class TurtleConfig:
    """One row of the turtle V3.6 pre-reg cell grid.

    The two SWEPT axes are ``entry_period`` and ``exit_period``;
    everything else is frozen.
    """

    entry_period: int = 20
    exit_period: int = 30
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0


def turtle_returns(
    bars: pd.DataFrame, *, cfg: TurtleConfig | None = None
) -> pd.Series:
    """Per-bar net return of the long-only Donchian breakout signal.

    Parameters:
        bars:
            OHLC DataFrame with columns ``open, high, low, close`` aligned on
            a DatetimeIndex. Index timezone is normalised by caller.
        cfg:
            ``TurtleConfig``. Default = canonical (entry=20, exit=30).

    Returns:
        Per-bar return Series, named ``ret``.
    """
    if cfg is None:
        cfg = TurtleConfig()
    required = {"high", "low", "close"}
    if not required.issubset(bars.columns):
        raise ValueError(
            f"turtle_returns requires columns {sorted(required)}; got {list(bars.columns)}"
        )

    high = bars["high"]
    low = bars["low"]
    close = bars["close"]

    donch_high = high.rolling(cfg.entry_period, min_periods=cfg.entry_period).max().shift(1)
    donch_low = low.rolling(cfg.exit_period, min_periods=cfg.exit_period).min().shift(1)

    arr_close = close.to_numpy()
    arr_high = donch_high.to_numpy()
    arr_low = donch_low.to_numpy()
    pos = np.zeros(len(close), dtype=float)
    state = 0
    for i in range(len(close)):
        if np.isnan(arr_high[i]) or np.isnan(arr_low[i]):
            pos[i] = float(state)
            continue
        if state == 0 and arr_close[i] > arr_high[i]:
            state = 1
        elif state == 1 and arr_close[i] < arr_low[i]:
            state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=close.index)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        net = gross - cost
    else:
        net = gross

    return net.rename("ret")


def turtle_assert_causal(
    bars: pd.DataFrame, *, cfg: TurtleConfig | None = None
) -> None:
    """L21 smoke: corrupting future bars must not change past returns."""
    if cfg is None:
        cfg = TurtleConfig()
    base = turtle_returns(bars, cfg=cfg)

    corrupted = bars.copy()
    n = len(corrupted)
    cutoff = n - 50
    if cutoff <= cfg.entry_period + 100:
        raise RuntimeError("Series too short for causality smoke")
    for col in ("open", "high", "low", "close"):
        if col in corrupted.columns:
            corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] = (
                corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] * 100.0
            )
    perturbed = turtle_returns(corrupted, cfg=cfg)

    safe_end = cutoff - 100
    base_past = base.iloc[:safe_end].dropna()
    pert_past = perturbed.iloc[:safe_end].dropna()
    diffs = (base_past - pert_past).abs()
    max_diff = float(diffs.max())
    if max_diff > 1e-12:
        n_changed = int((diffs > 1e-12).sum())
        raise AssertionError(
            f"L21 causality smoke FAILED: changed {n_changed} past returns "
            f"(max |delta|={max_diff:.2e})"
        )


__all__ = [
    "PERIODS_PER_YEAR_H1_EQ",
    "TurtleConfig",
    "turtle_assert_causal",
    "turtle_returns",
]
