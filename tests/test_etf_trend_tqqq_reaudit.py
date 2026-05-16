"""Tests for the etf_trend TQQQ V3.6 re-audit (Wave A.2-confirm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.etf_trend.etf_trend_tqqq_strategy import (
    EtfTrendTqqqConfig,
    buy_and_hold_tqqq_returns,
    etf_trend_tqqq_assert_causal,
    etf_trend_tqqq_returns,
)
from research.exploration.sweep_etf_trend_tqqq import etf_trend_tqqq_returns as sweep_fn


def _toy_universe(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    qqq_ret = rng.normal(0.0006, 0.015, size=n)
    qqq = 100.0 * np.exp(np.cumsum(qqq_ret))
    # TQQQ ≈ 3x QQQ daily returns minus decay; simulate with 3x scaled noise.
    tqqq_ret = 3.0 * qqq_ret - 0.5 * qqq_ret**2  # simple decay model
    tqqq = 100.0 * np.exp(np.cumsum(tqqq_ret))
    return pd.DataFrame({"QQQ": qqq, "TQQQ": tqqq}, index=idx)


def test_tqqq_returns_runs_clean() -> None:
    closes = _toy_universe(n=1500)
    rets = etf_trend_tqqq_returns(closes, cfg=EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=1))
    assert isinstance(rets, pd.Series)
    assert np.isfinite(rets.dropna().to_numpy()).all()


def test_tqqq_causal() -> None:
    closes = _toy_universe(n=1500)
    etf_trend_tqqq_assert_causal(closes, cfg=EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=1))


def test_tqqq_requires_both_columns() -> None:
    closes = _toy_universe(n=1500).drop(columns=["QQQ"])
    with pytest.raises(ValueError, match="QQQ"):
        etf_trend_tqqq_returns(closes)


def test_sweep_audit_parity_tqqq() -> None:
    closes = _toy_universe(n=1500, seed=11)
    sweep_ret = sweep_fn(closes, slow_ma=150, exit_confirm_days=1)
    audit_ret = etf_trend_tqqq_returns(
        closes, cfg=EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=1)
    )
    common = sweep_ret.dropna().index.intersection(audit_ret.dropna().index)
    assert len(common) > 100
    diff = (sweep_ret.reindex(common) - audit_ret.reindex(common)).abs()
    assert float(diff.max()) < 1e-12


def test_buy_and_hold_tqqq_is_long_only_no_costs() -> None:
    closes = _toy_universe(n=1500)
    rets = buy_and_hold_tqqq_returns(closes)
    # B&H is just TQQQ returns shifted; correlation with raw TQQQ ret must be ~1.0.
    tqqq_ret = np.log(closes["TQQQ"] / closes["TQQQ"].shift(1)).fillna(0.0)
    # Shift by 1 to compare.
    common_idx = rets.dropna().index.intersection(tqqq_ret.index)
    bh_align = rets.reindex(common_idx).iloc[1:]
    tqqq_align = tqqq_ret.reindex(common_idx).iloc[1:]
    corr = float(bh_align.corr(tqqq_align))
    assert corr > 0.99


def test_position_shift_discipline_tqqq() -> None:
    """L18 — verify no same-bar leakage with flat-then-spike series."""
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    qqq_vals = np.full(n, 100.0)
    qqq_vals[-1] = 110.0
    tqqq_vals = np.full(n, 50.0)
    tqqq_vals[-1] = 65.0
    closes = pd.DataFrame({"QQQ": qqq_vals, "TQQQ": tqqq_vals}, index=idx)
    rets = etf_trend_tqqq_returns(
        closes, cfg=EtfTrendTqqqConfig(slow_ma=200, exit_confirm_days=1, apply_costs=False)
    )
    nonzero = rets[rets.abs() > 1e-12]
    assert len(nonzero) == 0, f"shift discipline broken: {len(nonzero)} non-zero returns"
