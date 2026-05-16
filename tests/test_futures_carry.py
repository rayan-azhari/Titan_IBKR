"""Unit tests for D2 commodity futures carry.

Specified in directives/Pre-Reg D2 Commodity Futures Carry 2026-05-15.md §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.futures_carry.carry_strategy import (
    CarryConfig,
    _align_universe,
    _month_end_mask,
    _week_end_mask,
    build_portfolio_weights,
    carry_assert_causal,
    carry_returns,
    compute_carry_signal,
)
from titan.research.framework import StrategyClass, defaults_for


def _synthetic_universe(n_commodities: int = 10, n_bars: int = 600, seed: int = 0):
    """Synthetic M1/M2 with embedded carry-then-return correlation.

    For each commodity, half the bars are backwardated (M1 > M2) with
    positive subsequent drift; half are contangoed (M1 < M2) with
    negative drift. This is the BGR cross-sectional pattern in miniature
    so the strategy CAN earn positive returns on this universe.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    M1 = pd.DataFrame(index=idx, dtype=float)
    M2 = pd.DataFrame(index=idx, dtype=float)
    for i in range(n_commodities):
        # Some commodities are persistently backwardated (M1>M2), some contangoed.
        is_backwardated = i % 2 == 0
        drift = 0.0008 if is_backwardated else -0.0008
        m1_price = 100 * np.cumprod(1 + rng.normal(drift, 0.012, n_bars))
        # M2 trades at a small offset; backwardated => M2 < M1, contango => M2 > M1
        offset = -0.02 if is_backwardated else +0.02
        m2_price = m1_price * (1 + offset + rng.normal(0, 0.003, n_bars))
        M1[f"C{i}"] = m1_price
        M2[f"C{i}"] = m2_price
    return M1, M2


# ── (1) Class defaults ────────────────────────────────────────────────────


def test_carry_class_defaults():
    d = defaults_for(StrategyClass.CARRY)
    assert d.sharpe.primary == "per_day_mtm"
    assert d.sharpe.primary_periods_per_year == 252
    assert d.mc.max_dd_threshold_pct == 0.30
    assert d.mc.max_dd_pass_prob == 0.10


# ── (2) Universe alignment + signal ──────────────────────────────────────


def test_align_universe_intersects_columns_and_dates():
    M1 = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    M2 = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4], "D": [5, 6]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    a, b = _align_universe(M1, M2)
    assert list(a.columns) == ["A", "B"]
    assert list(b.columns) == ["A", "B"]
    assert len(a) == len(b) == 2


def test_carry_signal_positive_for_backwardation():
    """raw_carry = log(M1/M2); when M1 > M2 (backwardated), raw_carry > 0."""
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    M1 = pd.DataFrame({"X": np.full(10, 102.0)}, index=idx)
    M2 = pd.DataFrame({"X": np.full(10, 100.0)}, index=idx)
    signal = compute_carry_signal(M1, M2, smooth_days=1)
    # After the .shift(1), all post-warmup carry values should be log(102/100) > 0
    valid = signal["X"].dropna()
    assert (valid > 0).all()
    assert np.isclose(valid.iloc[0], np.log(102.0 / 100.0))


def test_carry_signal_negative_for_contango():
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    M1 = pd.DataFrame({"X": np.full(10, 98.0)}, index=idx)
    M2 = pd.DataFrame({"X": np.full(10, 100.0)}, index=idx)
    signal = compute_carry_signal(M1, M2, smooth_days=1)
    valid = signal["X"].dropna()
    assert (valid < 0).all()
    assert np.isclose(valid.iloc[0], np.log(98.0 / 100.0))


def test_carry_signal_smoothing_reduces_variance():
    M1, M2 = _synthetic_universe(n_commodities=5, n_bars=300, seed=1)
    raw = compute_carry_signal(M1, M2, smooth_days=1)
    sm = compute_carry_signal(M1, M2, smooth_days=5)
    # Smoothed signal should have lower per-bar variance.
    assert sm.std().mean() < raw.std().mean()


def test_rolling_yield_signal_does_not_need_m2():
    """Rolling-yield mode is M1-only. Empty M2 must NOT cause a failure."""
    M1, _ = _synthetic_universe(n_commodities=5, n_bars=500, seed=10)
    empty_m2 = pd.DataFrame(index=M1.index, columns=M1.columns, dtype=float)
    signal = compute_carry_signal(
        M1, empty_m2, smooth_days=1, signal_mode="rolling_yield", yield_lookback=126
    )
    # After warmup (yield_lookback bars), the signal should have finite values.
    valid = signal.iloc[200:]
    assert valid.notna().any().any(), "Rolling-yield signal produced no valid values"


def test_rolling_yield_signal_positive_for_uptrending_commodity():
    """A commodity with persistent positive drift should have positive
    rolling-yield carry; the strategy will rank it in the long quintile."""
    idx = pd.date_range("2020-01-02", periods=500, freq="B")
    up_trend = pd.DataFrame({"X": np.cumprod(1 + np.full(500, 0.001))}, index=idx)
    empty_m2 = pd.DataFrame(index=idx, columns=["X"], dtype=float)
    signal = compute_carry_signal(
        up_trend, empty_m2, signal_mode="rolling_yield", yield_lookback=252
    )
    # Post-warmup (after yield_lookback bars), signal must be > 0 for an uptrend.
    valid = signal["X"].dropna()
    assert (valid > 0).all(), "Rolling-yield should be positive for uptrending price"


def test_rolling_yield_signal_negative_for_downtrending_commodity():
    idx = pd.date_range("2020-01-02", periods=500, freq="B")
    down_trend = pd.DataFrame({"X": np.cumprod(1 + np.full(500, -0.001))}, index=idx)
    empty_m2 = pd.DataFrame(index=idx, columns=["X"], dtype=float)
    signal = compute_carry_signal(
        down_trend, empty_m2, signal_mode="rolling_yield", yield_lookback=252
    )
    valid = signal["X"].dropna()
    assert (valid < 0).all(), "Rolling-yield should be negative for downtrending price"


def test_carry_signal_rejects_unknown_mode():
    M1, M2 = _synthetic_universe(n_commodities=3, n_bars=100)
    try:
        compute_carry_signal(M1, M2, signal_mode="bogus")
    except ValueError as e:
        assert "signal_mode" in str(e)
    else:
        raise AssertionError("Expected ValueError on bogus signal_mode")


def test_carry_returns_runs_in_rolling_yield_mode_with_empty_m2():
    """End-to-end: carry_returns must work when M2 is empty + rolling_yield."""
    M1, _ = _synthetic_universe(n_commodities=10, n_bars=600, seed=11)
    empty_m2 = pd.DataFrame(index=M1.index, columns=M1.columns, dtype=float)
    cfg = CarryConfig(signal_mode="rolling_yield", yield_lookback=252, apply_costs=False)
    rets = carry_returns(M1, empty_m2, cfg=cfg)
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(M1)


# ── (3) Rebalance masks ──────────────────────────────────────────────────


def test_month_end_mask_marks_last_business_day():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    me = _month_end_mask(idx)
    assert me.tolist() == [False, True, False, True, True]


def test_week_end_mask_marks_last_day_of_week():
    # Mon-Fri week: Friday should be True.
    idx = pd.DatetimeIndex(
        ["2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-29"]
    )
    we = _week_end_mask(idx)
    # Last day of week 4 (Fri 2024-01-26) is True; the new week's Mon is also True (its own last seen so far).
    assert we[4] is np.True_ or we[4]  # Friday
    assert we[5] is np.True_ or we[5]  # Mon = last of its week so far


# ── (4) Portfolio construction ───────────────────────────────────────────


def test_portfolio_long_short_is_dollar_neutral():
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=300, seed=2)
    signal = compute_carry_signal(M1, M2, smooth_days=1)
    w = build_portfolio_weights(signal, cfg=CarryConfig(long_only=False, breadth_pct=0.20))
    # On rebalance days post-warmup, weights should sum to ~0.
    sums = w.sum(axis=1)
    rebal_days = sums[sums.abs() > 1e-9]
    if len(rebal_days) > 0:
        assert rebal_days.abs().max() < 1e-9 + 0.01, (
            f"Long-short weights not net-neutral: max |sum| = {rebal_days.abs().max():.4f}"
        )


def test_portfolio_long_only_has_no_short_legs():
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=300, seed=3)
    signal = compute_carry_signal(M1, M2, smooth_days=1)
    w = build_portfolio_weights(signal, cfg=CarryConfig(long_only=True, breadth_pct=0.20))
    assert (w >= -1e-9).all().all(), "Long-only must never produce negative weights"


def test_quintile_breadth_produces_smaller_legs_than_tercile():
    """quintile (0.20) selects fewer names per leg than tercile (0.333)."""
    M1, M2 = _synthetic_universe(n_commodities=15, n_bars=300, seed=4)
    signal = compute_carry_signal(M1, M2, smooth_days=1)
    w_q = build_portfolio_weights(signal, cfg=CarryConfig(breadth_pct=0.20))
    w_t = build_portfolio_weights(signal, cfg=CarryConfig(breadth_pct=0.333))
    n_long_q = (w_q.iloc[-1] > 0).sum()
    n_long_t = (w_t.iloc[-1] > 0).sum()
    assert n_long_q <= n_long_t


# ── (5) Returns + causality ──────────────────────────────────────────────


def test_carry_returns_proper_shape():
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=300, seed=5)
    rets = carry_returns(M1, M2, cfg=CarryConfig())
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(M1)


def test_carry_returns_positive_on_designed_universe():
    """The synthetic universe is RIGGED so backwardated names trend up and
    contangoed names trend down. A long-short carry strategy SHOULD earn."""
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=600, seed=6)
    rets = carry_returns(M1, M2, cfg=CarryConfig(apply_costs=False))
    sharpe_ratio = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio > 0.3, (
        f"Expected positive Sharpe on designed universe; got {sharpe_ratio:.3f}"
    )


def test_costs_reduce_carry_returns():
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=300, seed=7)
    gross = carry_returns(M1, M2, cfg=CarryConfig(apply_costs=False))
    net = carry_returns(M1, M2, cfg=CarryConfig(apply_costs=True))
    # On rebalance days the net cumulative return should be lower than gross.
    assert net.sum() < gross.sum() + 1e-9


def test_carry_assert_causal_passes():
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=400, seed=8)
    carry_assert_causal(M1, M2, n_trials=3, seed=42)


def test_corrupting_future_does_not_affect_past_returns():
    M1, M2 = _synthetic_universe(n_commodities=10, n_bars=300, seed=9)
    cfg = CarryConfig(apply_costs=False)
    base = carry_returns(M1, M2, cfg=cfg)
    M1c = M1.copy()
    M2c = M2.copy()
    M1c.iloc[-1] = M1c.iloc[-1] * 5
    M2c.iloc[-1] = M2c.iloc[-1] * 5
    altered = carry_returns(M1c, M2c, cfg=cfg)
    pd.testing.assert_series_equal(base.iloc[:-1], altered.iloc[:-1], check_names=False)
