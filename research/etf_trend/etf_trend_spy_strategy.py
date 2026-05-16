"""etf_trend SPY — pure-research strategy module for the V3.6 re-audit.

Specified in ``directives/Pre-Reg etf_trend SPY Re-audit 2026-05-16.md``.

Mechanism (per bar t):

    1. ``sma(t) = rolling_mean(spy.close, slow_ma)[t]``       — trend boundary.
    2. ``in_regime(t) = spy.close[t] > sma(t)``.
    3. Exit-confirm state machine: once long, stay long until
       ``close < sma`` for ``exit_confirm_days`` consecutive bars.
    4. Vol-target sizing on SPY (target_vol_ann=0.20, EWMA span=20, max_lev=2.0).
    5. ``net = position.shift(1) * spy_log_return - cost``.

Causality (L04 / A1 / L18):
    - SMA at close[t] uses [t-slow_ma+1..t] inclusive (known EOD t).
    - Position effective at close[t] earns return from t -> t+1 via
      ``position.shift(1) * log_ret``.

Parity contract: the sweep
(`research/exploration/sweep_etf_trend_spy.py::etf_trend_spy_returns`) and
the audit (`etf_trend_spy_returns()` here) MUST produce identical per-bar
returns for identical inputs. Enforced by
``tests/test_etf_trend_spy_reaudit.py::test_sweep_audit_parity``.

Buy-and-hold baseline (`buy_and_hold_spy_returns`) is the benchmark for
the L17 relative MC test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from titan.research.metrics import BARS_PER_YEAR


@dataclass(frozen=True)
class EtfTrendSpyConfig:
    """One row of the etf_trend_spy V3.6 pre-reg cell grid.

    Swept axes: ``slow_ma`` and ``exit_confirm_days``. Sizing knobs and
    cost model are frozen at live-config values.
    """

    slow_ma: int = 300
    exit_confirm_days: int = 5
    vol_target_ann: float = 0.20
    vol_ewma_span: int = 20
    max_leverage: float = 2.0
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0


def _apply_exit_confirm(above_sma: pd.Series, *, exit_confirm_days: int) -> pd.Series:
    """Exit-confirm state machine.

    State: ``pos`` (0 = flat, 1 = long); ``days_below`` (consecutive bars
    with close < sma while long).
    """
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


def _vol_target_scale(
    log_ret: pd.Series, *, target_ann: float, span: int, max_lev: float
) -> pd.Series:
    var = log_ret.pow(2).ewm(span=span, adjust=False, min_periods=span).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (target_ann / realised_vol_ann.replace(0, np.nan)).clip(upper=max_lev).fillna(0.0)
    return scale


def etf_trend_spy_returns(
    closes_df: pd.DataFrame, *, cfg: EtfTrendSpyConfig | None = None
) -> pd.Series:
    """Per-bar net return of the simplified SPY trend signal.

    Parameters:
        closes_df:
            DataFrame with column ``SPY`` (or ``close``) of daily closes.
        cfg:
            ``EtfTrendSpyConfig``. Default = canonical
            ``(slow_ma=300, exit_confirm_days=5)``.

    Returns:
        Per-bar return Series, named ``ret``.
    """
    if cfg is None:
        cfg = EtfTrendSpyConfig()
    # Accept either 'SPY' or 'close' columns (MC bootstraps use 'close').
    if "SPY" in closes_df.columns:
        spy = closes_df["SPY"]
    elif "close" in closes_df.columns:
        spy = closes_df["close"]
    else:
        raise ValueError(
            f"etf_trend_spy_returns requires column 'SPY' or 'close'; got {list(closes_df.columns)}"
        )

    sma = spy.rolling(cfg.slow_ma, min_periods=cfg.slow_ma).mean()
    above = (spy > sma).astype(float)
    sig = _apply_exit_confirm(above, exit_confirm_days=cfg.exit_confirm_days)

    log_ret = np.log(spy / spy.shift(1)).fillna(0.0)
    scale = _vol_target_scale(
        log_ret,
        target_ann=cfg.vol_target_ann,
        span=cfg.vol_ewma_span,
        max_lev=cfg.max_leverage,
    )
    position = (sig * scale).fillna(0.0)

    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * log_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        net = gross - cost
    else:
        net = gross

    return net.rename("ret")


def buy_and_hold_spy_returns(
    closes_df: pd.DataFrame, *, cfg: EtfTrendSpyConfig | None = None
) -> pd.Series:
    """Vol-targeted buy-and-hold SPY returns — L17 relative-MC benchmark.

    Same vol-target / leverage / cost knobs as the strategy so the
    comparison isolates the SIGNAL effect (not the sizing or cost).
    """
    if cfg is None:
        cfg = EtfTrendSpyConfig()
    if "SPY" in closes_df.columns:
        spy = closes_df["SPY"]
    elif "close" in closes_df.columns:
        spy = closes_df["close"]
    else:
        raise ValueError(
            f"buy_and_hold_spy_returns requires column 'SPY' or 'close'; got {list(closes_df.columns)}"
        )

    log_ret = np.log(spy / spy.shift(1)).fillna(0.0)
    scale = _vol_target_scale(
        log_ret,
        target_ann=cfg.vol_target_ann,
        span=cfg.vol_ewma_span,
        max_lev=cfg.max_leverage,
    )
    # Always long (signal = 1).
    position = scale.fillna(0.0)
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * log_ret
    # B&H has near-zero turnover (only when vol-target rebalances) — costs negligible
    # but apply for symmetry.
    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def etf_trend_spy_assert_causal(
    closes_df: pd.DataFrame, *, cfg: EtfTrendSpyConfig | None = None
) -> None:
    """L04 smoke test: corrupting future closes must not change past returns."""
    if cfg is None:
        cfg = EtfTrendSpyConfig()
    base = etf_trend_spy_returns(closes_df, cfg=cfg)
    rng = np.random.default_rng(42)
    t_corrupt = int(rng.integers(cfg.slow_ma + cfg.vol_ewma_span + 50, len(closes_df) - 50))
    corrupted = closes_df.copy()
    corrupted.iloc[t_corrupt:] = np.nan
    corrupted_ret = etf_trend_spy_returns(corrupted, cfg=cfg)

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
    "EtfTrendSpyConfig",
    "etf_trend_spy_returns",
    "buy_and_hold_spy_returns",
    "etf_trend_spy_assert_causal",
]
