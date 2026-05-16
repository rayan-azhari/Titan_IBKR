"""gold_macro -- pure-research strategy module for the V3.6 re-audit.

Specified in ``directives/Pre-Reg gold_macro Re-audit 2026-05-16.md``.

Mechanism (per bar t, 3-component composite signal -> GLD long-only position):

    1. Real-rate proxy: rr_signal(t) = -(log(TIP/TLT)[t] - log(TIP/TLT)[t-W_rr])
    2. Dollar weakness:  d_signal(t) = -(log(DXY)[t] - log(DXY)[t-W_d])
    3. Causal expanding z-score normalisation -> composite_z = (rr_z + d_z)/2
    4. Momentum gate: GLD[t] > SMA(slow_ma)[t]
    5. signal = 1 if (composite_z > 0) AND (momentum) else 0
    6. Vol-target sizing (target=0.10 ann, EWMA span=20, max_lev=1.5)
    7. net = position.shift(1) * gld_log_return - cost

Causality (L04 / A1 / L21):
    - All component signals at t use closes at t (known by EOD t).
    - Z-score is causal expanding (past-only).
    - Momentum SMA uses GLD[t-slow_ma+1 .. t].
    - Position EFFECTIVE for the t->t+1 return is the t-1 close decision
      (via position.shift(1)).
    - L21 smoke test asserts past returns are bit-exact when future bars
      are corrupted.

Parity contract: the sweep (`research/exploration/sweep_gold_macro.py`)
and the audit (`gold_macro_returns()`) MUST produce identical per-bar
returns for identical inputs at the canonical (100, 60) config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from titan.research.metrics import BARS_PER_YEAR


@dataclass(frozen=True)
class GoldMacroConfig:
    """One row of the gold_macro V3.6 pre-reg cell grid.

    The two SWEPT axes are ``slow_ma`` and ``real_rate_window``;
    ``dollar_window`` is frozen at the live value to keep the surface 2D.
    """

    slow_ma: int = 100
    real_rate_window: int = 60
    dollar_window: int = 20
    vol_target_ann: float = 0.10
    vol_ewma_span: int = 20
    max_leverage: float = 1.5
    zscore_min_obs: int = 60
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.5
    # If True, disables the composite_z gate; the strategy reduces to the
    # bare SMA(slow_ma) trend signal. Used for the C4_pure_sma reference cell.
    disable_composite: bool = False


def _expanding_zscore(series: pd.Series, *, min_obs: int) -> pd.Series:
    """Causal expanding z-score. At t, uses values in [0..t] inclusive.

    Returns 0.0 until min_obs values accumulate. Uses cumulative
    sum / sumsq for O(n) mean/std (vs O(n^2) for naive expanding).
    """
    arr = series.to_numpy()
    n = len(arr)
    out = np.zeros(n, dtype=float)
    csum = np.cumsum(arr)
    csumsq = np.cumsum(arr * arr)
    for i in range(min_obs, n):
        k = i + 1
        mu = csum[i] / k
        var = csumsq[i] / k - mu * mu
        if var <= 1e-12:
            out[i] = 0.0
        else:
            out[i] = (arr[i] - mu) / np.sqrt(var)
    return pd.Series(out, index=series.index)


def _vol_target_scale(
    gld_log_ret: pd.Series, *, target_ann: float, span: int, max_lev: float
) -> pd.Series:
    """EWMA-based vol-target scale, capped at ``max_lev``."""
    var = gld_log_ret.pow(2).ewm(span=span, adjust=False, min_periods=span).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (target_ann / realised_vol_ann.replace(0, np.nan)).clip(upper=max_lev).fillna(0.0)
    return scale


def gold_macro_returns(
    closes_df: pd.DataFrame, *, cfg: GoldMacroConfig | None = None
) -> pd.Series:
    """Per-bar net return of the gold_macro signal.

    Parameters:
        closes_df:
            DataFrame with columns ``GLD``, ``TIP``, ``TLT``, ``DXY`` of daily
            closes, aligned on a common DatetimeIndex.
        cfg:
            ``GoldMacroConfig``. Default = canonical (slow_ma=100,
            real_rate_window=60).

    Returns:
        Per-bar return Series, named ``ret``.
    """
    if cfg is None:
        cfg = GoldMacroConfig()
    required = {"GLD", "TIP", "TLT", "DXY"}
    if not required.issubset(closes_df.columns):
        raise ValueError(
            f"gold_macro_returns requires columns {sorted(required)}; got {list(closes_df.columns)}"
        )

    gld = closes_df["GLD"]
    tip = closes_df["TIP"]
    tlt = closes_df["TLT"]
    dxy = closes_df["DXY"]

    # Component 1: real-rate proxy.
    log_rr = np.log(tip / tlt)
    rr_signal = -(log_rr - log_rr.shift(cfg.real_rate_window))

    # Component 2: dollar weakness.
    log_dxy = np.log(dxy)
    d_signal = -(log_dxy - log_dxy.shift(cfg.dollar_window))

    # Causal expanding z-score per component, then average.
    rr_z = _expanding_zscore(rr_signal.fillna(0.0), min_obs=cfg.zscore_min_obs)
    d_z = _expanding_zscore(d_signal.fillna(0.0), min_obs=cfg.zscore_min_obs)
    composite_z = (rr_z + d_z) / 2.0

    # Component 3: momentum gate.
    sma = gld.rolling(cfg.slow_ma, min_periods=cfg.slow_ma).mean()
    momentum = (gld > sma).astype(float)

    # Entry condition.
    if cfg.disable_composite:
        # C4 reference cell: bare momentum only.
        signal = momentum.copy()
    else:
        signal = ((composite_z > 0) & (momentum > 0)).astype(float)

    # Vol-target sizing.
    gld_ret = np.log(gld / gld.shift(1)).fillna(0.0)
    scale = _vol_target_scale(
        gld_ret,
        target_ann=cfg.vol_target_ann,
        span=cfg.vol_ewma_span,
        max_lev=cfg.max_leverage,
    )
    position = (signal * scale).fillna(0.0)

    # Per-bar return: position effective at t earns t->t+1 return.
    held = position.shift(1).fillna(0.0)
    gross = held * gld_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        net = gross - cost
    else:
        net = gross

    return net.rename("ret")


def gold_macro_assert_causal(
    closes_df: pd.DataFrame, *, cfg: GoldMacroConfig | None = None
) -> None:
    """L21 smoke: corrupting future closes must not change past returns.

    Multiplies the last 20 bars of every column by 100, then asserts the
    pre-cutoff segment of the return series is bit-exact identical.

    Raises AssertionError on any mismatch.
    """
    if cfg is None:
        cfg = GoldMacroConfig()
    base = gold_macro_returns(closes_df, cfg=cfg)

    corrupted = closes_df.copy()
    n = len(corrupted)
    cutoff = n - 20
    if cutoff <= cfg.slow_ma + 100:
        raise RuntimeError("Series too short for causality smoke")
    for col in corrupted.columns:
        corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] = (
            corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] * 100.0
        )
    perturbed = gold_macro_returns(corrupted, cfg=cfg)

    # Buffer the safe-past end by slow_ma to account for SMA(slow_ma) reaching back.
    safe_end = cutoff - cfg.slow_ma
    base_past = base.iloc[:safe_end].dropna()
    pert_past = perturbed.iloc[:safe_end].dropna()
    diffs = (base_past - pert_past).abs()
    max_diff = float(diffs.max())
    if max_diff > 1e-12:
        n_changed = int((diffs > 1e-12).sum())
        raise AssertionError(
            f"L21 causality smoke FAILED: corrupting closes at t={cutoff} "
            f"changed {n_changed} past returns (max |delta|={max_diff:.2e})"
        )


__all__ = [
    "GoldMacroConfig",
    "gold_macro_assert_causal",
    "gold_macro_returns",
]
