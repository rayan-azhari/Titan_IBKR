"""mr_audjpy — pure-research minimal VWAP-MR strategy module.

**SCOPE NOTE:** This is the SIGNAL-LAYER abstraction of the live
`titan/strategies/mr_audjpy/strategy.py`. It captures the core VWAP-
deviation mean-reversion edge WITHOUT the live strategy's:

- 4-tier grid entries (live: tier_sizes=[1,2,4,8])
- Multi-scale regime filter (donchian H1/H4/D/W disagreement)
- 07:00–12:00 UTC session entry filter
- 21:00 UTC hard close
- FX-pair sizing complexity (JPY-quote)

The audit question this module answers is:

  > **Does VWAP-deviation mean-reversion on H1 AUD/JPY produce a
    meaningful Sharpe edge with V3.6-correct math?**

If yes → the live strategy's tier-grid + regime-filter + session
machinery is a refinement on a real signal, worth a full audit.
If no → the signal-layer fails L52; no need to audit the refinements
above it.

Wave A.3 of the V1-era re-audit roster.

Causality (L04 / A1 / L18):
    - VWAP at t uses bars [t-anchor+1, t] (EOD-known).
    - Percentile bands at t use rolling [t-pct_window+1, t].
    - Position effective at t earns return t -> t+1 via .shift(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MrAudjpyConfig:
    """Minimal pure-research config.

    Live config (config/mr_audjpy.toml) has additional knobs we don't
    sweep here: tier_sizes, regime filter, session windows, NY close.
    """

    vwap_anchor: int = 24          # H1 bars (24 = 1 trading day)
    pct_window: int = 500          # rolling percentile window for bands
    entry_pct: float = 0.95        # entry threshold (single-tier, conservative)
    reversion_pct: float = 0.50    # exit threshold (50% reversion to VWAP)
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.5  # 1.5 bps FX spread + slip (matches V3.6 FX model)


def _rolling_vwap(close: pd.Series, volume: pd.Series, *, anchor: int) -> pd.Series:
    """Rolling VWAP over `anchor` bars. Past-only by construction."""
    pv = (close * volume).rolling(anchor, min_periods=anchor).sum()
    v = volume.rolling(anchor, min_periods=anchor).sum().replace(0, np.nan)
    return pv / v


def mr_audjpy_returns(
    bars_df: pd.DataFrame, *, cfg: MrAudjpyConfig | None = None
) -> pd.Series:
    """Per-bar net log-return of the simplified VWAP-MR signal on AUD/JPY H1.

    Mechanism (single-tier, no grid, no regime filter):
        1. Compute rolling VWAP at vwap_anchor bars.
        2. Compute deviation = (close - vwap) / vwap.
        3. Compute rolling percentile bands at pct_window.
        4. Enter SHORT when deviation > +entry_pct band; LONG when < -entry_pct.
        5. Exit when deviation has reverted by reversion_pct toward 0.
        6. Cost = bps per turnover (1.5bps for AUD/JPY FX).

    `bars_df` must have columns ``['close', 'volume']`` (volume may be -1
    sentinel for unknown — we fall back to equal-weighted in that case).
    """
    if cfg is None:
        cfg = MrAudjpyConfig()

    close = bars_df["close"].astype(float)
    if "volume" in bars_df.columns:
        vol = bars_df["volume"].astype(float)
        # Replace -1 sentinel (FX bars often have no real volume) with 1.0.
        vol = vol.where(vol > 0, 1.0)
    else:
        vol = pd.Series(1.0, index=close.index)

    vwap = _rolling_vwap(close, vol, anchor=cfg.vwap_anchor)
    dev = (close - vwap) / vwap

    # Rolling percentile bands (past-only).
    upper = dev.rolling(cfg.pct_window, min_periods=cfg.pct_window).quantile(cfg.entry_pct)
    lower = dev.rolling(cfg.pct_window, min_periods=cfg.pct_window).quantile(1.0 - cfg.entry_pct)

    # State machine — single-position MR (no tier grid).
    # state: 0 (flat), +1 (long), -1 (short)
    # Entry: dev < lower -> +1 (price below VWAP, expect mean-revert UP)
    #        dev > upper -> -1
    # Exit:  dev crosses reversion-band toward 0
    arr_dev = dev.to_numpy()
    arr_upper = upper.to_numpy()
    arr_lower = lower.to_numpy()
    pos = np.zeros(len(close), dtype=float)
    state = 0
    entry_dev = 0.0
    for i in range(len(close)):
        if np.isnan(arr_dev[i]) or np.isnan(arr_upper[i]):
            pos[i] = float(state)
            continue
        if state == 0:
            if arr_dev[i] < arr_lower[i]:
                state = 1
                entry_dev = arr_dev[i]
            elif arr_dev[i] > arr_upper[i]:
                state = -1
                entry_dev = arr_dev[i]
        elif state == 1:
            # Long; exit when deviation reverts by reversion_pct toward 0.
            if arr_dev[i] >= entry_dev * (1.0 - cfg.reversion_pct):
                state = 0
        elif state == -1:
            if arr_dev[i] <= entry_dev * (1.0 - cfg.reversion_pct):
                state = 0
        pos[i] = float(state)

    position = pd.Series(pos, index=close.index)
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * log_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def mr_audjpy_assert_causal(
    bars_df: pd.DataFrame, *, cfg: MrAudjpyConfig | None = None
) -> None:
    """L04 smoke test — corrupting future bars must not change past returns."""
    if cfg is None:
        cfg = MrAudjpyConfig()
    base = mr_audjpy_returns(bars_df, cfg=cfg)
    rng = np.random.default_rng(42)
    t_corrupt = int(rng.integers(cfg.pct_window + 50, len(bars_df) - 50))
    corrupted = bars_df.copy()
    corrupted.iloc[t_corrupt:] = np.nan
    corrupted_ret = mr_audjpy_returns(corrupted, cfg=cfg)

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
            f"Causality smoke failed: corrupting bars at t={t_corrupt} "
            f"changed {n_changed} past returns (max |delta|={max_diff:.2e})"
        )


__all__ = [
    "MrAudjpyConfig",
    "mr_audjpy_returns",
    "mr_audjpy_assert_causal",
]
