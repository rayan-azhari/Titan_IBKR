"""bond_gold — pure-research strategy module for the V3.6 re-audit.

Specified in ``directives/Pre-Reg bond_gold Re-audit 2026-05-16.md``.

Mechanism (per bar t, single-instrument signal → GLD position):

    1. ``bond_mom(t) = log(IEF[t] / IEF[t - lookback])``        — bond momentum.
    2. Causal rolling z-score over 504 days.
    3. ``sig_raw = 1`` when ``z > threshold``, else 0.
    4. Hold-day floor: minimum 20 bars in position.
    5. Vol-target sizing on GLD (target_vol_ann=0.10, EWMA span=20, max_lev=1.5).
    6. ``net = position.shift(1) * gld_log_return - cost``.

Causality (L04 / A1 / L18):
    - Rolling stats use past-only windows.
    - Position effective at close[t] earns return from t -> t+1
      via ``position.shift(1) * log_ret``.

Parity contract: the sweep (`research/exploration/sweep_bond_gold.py::
bond_gold_strategy_fn`) and the audit (`bond_gold_returns()`) MUST
produce identical per-bar returns for identical inputs. Enforced by
``tests/test_bond_gold_reaudit.py::test_sweep_audit_parity``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from titan.research.metrics import BARS_PER_YEAR


@dataclass(frozen=True)
class BondGoldConfig:
    """One row of the bond_gold V3.6 pre-reg cell grid.

    The two SWEPT axes are ``lookback`` and ``threshold``; everything else
    is frozen at the live-config values per the pre-reg directive.
    """

    lookback: int = 120
    threshold: float = 0.50
    hold_days: int = 20
    zscore_window: int = 504
    vol_target_ann: float = 0.10
    vol_ewma_span: int = 20
    max_leverage: float = 1.5
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0


def _causal_rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    """Past-only rolling z-score. NaN until ``window`` bars accumulate.

    Note (causality): pandas' ``rolling(window).mean()`` at index t uses
    bars [t-window+1 .. t] inclusive — INCLUDES bar t. For the signal at
    t to be EOD-causal, this is fine because all inputs are EOD-known.
    The downstream ``position.shift(1)`` step ensures the position
    EFFECTIVE for the t -> t+1 return uses ONLY information through t-1.
    """
    m = series.rolling(window, min_periods=window).mean()
    s = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - m) / s.replace(0, np.nan)


def _apply_hold_floor(raw_sig: pd.Series, *, hold_days: int) -> pd.Series:
    """Enforce minimum hold period after entry.

    State machine: flat -> long when ``raw_sig`` flips to 1; stay long for
    at least ``hold_days`` bars; exit when ``raw_sig`` flips to 0 AND the
    hold floor is satisfied.
    """
    out = raw_sig.copy()
    pos = 0
    held = 0
    arr_raw = raw_sig.to_numpy()
    arr_out = np.empty(len(raw_sig), dtype=float)
    for i in range(len(raw_sig)):
        if pos == 0 and arr_raw[i] == 1.0:
            pos = 1
            held = 0
        elif pos == 1:
            held += 1
            if held >= hold_days and arr_raw[i] == 0.0:
                pos = 0
                held = 0
        arr_out[i] = float(pos)
    out.iloc[:] = arr_out
    return out


def _vol_target_scale(
    gld_log_ret: pd.Series, *, target_ann: float, span: int, max_lev: float
) -> pd.Series:
    """EWMA-based vol-target scale, capped at ``max_lev``."""
    var = gld_log_ret.pow(2).ewm(span=span, adjust=False, min_periods=span).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (target_ann / realised_vol_ann.replace(0, np.nan)).clip(upper=max_lev).fillna(0.0)
    return scale


def bond_gold_returns(closes_df: pd.DataFrame, *, cfg: BondGoldConfig | None = None) -> pd.Series:
    """Per-bar net return of the bond_gold signal.

    Parameters:
        closes_df:
            DataFrame with columns ``IEF`` and ``GLD`` of daily closes,
            aligned on a common DatetimeIndex.
        cfg:
            ``BondGoldConfig``. Default = canonical (lookback=120,
            threshold=0.50, costs ON).

    Returns:
        Per-bar return Series, named ``ret``.
    """
    if cfg is None:
        cfg = BondGoldConfig()
    if "IEF" not in closes_df.columns or "GLD" not in closes_df.columns:
        raise ValueError(
            f"bond_gold_returns requires columns 'IEF' and 'GLD'; got {list(closes_df.columns)}"
        )

    ief = closes_df["IEF"]
    gld = closes_df["GLD"]

    bond_mom = np.log(ief / ief.shift(cfg.lookback))
    z = _causal_rolling_zscore(bond_mom, window=cfg.zscore_window)
    raw_sig = (z > cfg.threshold).astype(float).fillna(0.0)
    sig = _apply_hold_floor(raw_sig, hold_days=cfg.hold_days)

    gld_ret = np.log(gld / gld.shift(1)).fillna(0.0)
    scale = _vol_target_scale(
        gld_ret,
        target_ann=cfg.vol_target_ann,
        span=cfg.vol_ewma_span,
        max_lev=cfg.max_leverage,
    )
    position = (sig * scale).fillna(0.0)

    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * gld_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        net = gross - cost
    else:
        net = gross

    return net.rename("ret")


def bond_gold_assert_causal(closes_df: pd.DataFrame, *, cfg: BondGoldConfig | None = None) -> None:
    """Smoke test: corrupting future closes must not change past returns.

    Picks a random date T in the middle of the series, sets IEF[T:] and
    GLD[T:] to NaN, and asserts that bond_gold_returns()[:T-1] is
    bit-exact identical to the un-corrupted run.

    Raises AssertionError on any mismatch.
    """
    if cfg is None:
        cfg = BondGoldConfig()
    base = bond_gold_returns(closes_df, cfg=cfg)

    rng = np.random.default_rng(42)
    t_corrupt = int(rng.integers(cfg.zscore_window + cfg.lookback + 100, len(closes_df) - 100))

    corrupted = closes_df.copy()
    corrupted.iloc[t_corrupt:] = np.nan
    corrupted_ret = bond_gold_returns(corrupted, cfg=cfg)

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
    "BondGoldConfig",
    "bond_gold_returns",
    "bond_gold_assert_causal",
]
