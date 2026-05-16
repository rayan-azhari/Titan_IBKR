"""Tests for the etf_trend SPY V3.6 re-audit pipeline (Wave A.2).

Critical invariants (pre-reg §5):
    * L04 / A1 — etf_trend_spy_assert_causal must pass on synthetic data.
    * L18 — position is .shift(1)'d before earning return.
    * SWEEP ↔ AUDIT parity — sweep's strategy fn and audit module produce
      bit-exact returns for identical inputs.
    * Buy-and-hold baseline returns are causal (no future leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.etf_trend.etf_trend_spy_strategy import (
    EtfTrendSpyConfig,
    buy_and_hold_spy_returns,
    etf_trend_spy_assert_causal,
    etf_trend_spy_returns,
)
from research.exploration.sweep_etf_trend_spy import etf_trend_spy_returns as sweep_fn


def _toy_spy(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic SPY-like price series with weak positive drift."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    ret = rng.normal(0.0004, 0.012, size=n)
    return pd.DataFrame({"SPY": 100.0 * np.exp(np.cumsum(ret))}, index=idx)


def test_etf_trend_spy_returns_runs_clean() -> None:
    closes = _toy_spy(n=1500)
    rets = etf_trend_spy_returns(closes, cfg=EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5))
    assert isinstance(rets, pd.Series)
    assert rets.dropna().shape[0] > 0
    assert np.isfinite(rets.dropna().to_numpy()).all()


def test_etf_trend_spy_causal() -> None:
    """L04 / A1 — corrupting future closes must NOT change past returns."""
    closes = _toy_spy(n=1500)
    etf_trend_spy_assert_causal(closes, cfg=EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5))


def test_accepts_close_column_alias() -> None:
    """The strategy fn accepts 'close' column too (for MC bootstrap dfs)."""
    closes = _toy_spy(n=1500).rename(columns={"SPY": "close"})
    rets = etf_trend_spy_returns(closes, cfg=EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5))
    assert rets.dropna().shape[0] > 0


def test_requires_spy_or_close_column() -> None:
    closes = _toy_spy(n=1500).rename(columns={"SPY": "X"})
    with pytest.raises(ValueError, match="SPY"):
        etf_trend_spy_returns(closes)


def test_sweep_audit_parity() -> None:
    """SWEEP ↔ AUDIT parity for the canonical cell."""
    closes = _toy_spy(n=1500, seed=11)
    sweep_ret = sweep_fn(closes, slow_ma=300, exit_confirm_days=5)
    audit_ret = etf_trend_spy_returns(
        closes, cfg=EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5)
    )
    common = sweep_ret.dropna().index.intersection(audit_ret.dropna().index)
    assert len(common) > 100, "parity test could not find common index"
    diff = (sweep_ret.reindex(common) - audit_ret.reindex(common)).abs()
    max_diff = float(diff.max())
    assert max_diff < 1e-12, (
        f"sweep vs audit divergence: max |diff| = {max_diff:.2e} on {len(common)} bars"
    )


def test_position_shift_discipline() -> None:
    """L18 — position must NOT earn the same-bar return.

    Construct a series with one large positive bar after a long flat
    period. With SMA flat, regime fires ONLY on the spike bar. The shift
    contract guarantees that even if the position fires at bar T, the
    return at bar T uses position[T-1] (= 0), so gross return on the
    spike bar must be exactly 0.

    Note: costs ARE charged on the position transition at the spike bar
    (legitimate transaction cost), so we disable costs to isolate the
    shift-discipline check.
    """
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    vals = np.full(n, 100.0)
    vals[-1] = 120.0  # +20% spike
    spy = pd.Series(vals, index=idx)
    closes = pd.DataFrame({"SPY": spy})
    rets = etf_trend_spy_returns(
        closes,
        cfg=EtfTrendSpyConfig(slow_ma=200, exit_confirm_days=1, apply_costs=False),
    )
    nonzero = rets[rets.abs() > 1e-12]
    assert len(nonzero) == 0, (
        f"shift discipline broken: {len(nonzero)} non-zero gross returns "
        f"on flat-then-spike series"
    )


def test_buy_and_hold_baseline_causal() -> None:
    closes = _toy_spy(n=1500)
    rets = buy_and_hold_spy_returns(closes)
    assert isinstance(rets, pd.Series)
    assert rets.dropna().shape[0] > 0
    # Causality: position at t-1 earns return at t. The first bar's return
    # must be 0 (no prior position).
    assert abs(rets.iloc[0]) < 1e-12


def test_buy_and_hold_is_long_only() -> None:
    """B&H baseline must be long-only — after warmup, returns are correlated +1
    with SPY's underlying returns (perfect long exposure, ignoring vol-target rescaling).
    """
    closes = _toy_spy(n=1500)
    rets = buy_and_hold_spy_returns(closes)
    spy_ret = np.log(closes["SPY"] / closes["SPY"].shift(1)).fillna(0.0)
    # Skip warmup (first 100 bars to let EWMA vol stabilise).
    common = rets.iloc[100:].dropna().index.intersection(spy_ret.iloc[100:].dropna().index)
    bh_align = rets.reindex(common)
    spy_align = spy_ret.reindex(common)
    # Correlation must be very high (B&H is just SPY scaled by EWMA-vol-targeted leverage).
    corr = float(bh_align.corr(spy_align))
    # Correlation < 1.0 is normal — vol-target rescaling is a slowly-varying
    # scalar so per-bar correlation drops slightly below unity.
    assert corr > 0.95, f"B&H baseline not strongly correlated with SPY: corr={corr:.4f}"
