"""Unit tests for run_relative_block_mc (V3.6 L17 gate).

Contract:
  * A strategy that BEATS its benchmark on MaxDD across most paths should pass.
  * A strategy that MATCHES its benchmark (because it IS the benchmark) should
    score a median DD ratio ~= 1.0 and FAIL the gate (we want strict reduction).
  * A strategy that's UNIFORMLY WORSE than its benchmark should fail decisively.
  * Reproducibility: same seed → same result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.research.framework.mc import (
    RelativeMcResult,
    run_relative_block_mc,
)
from titan.research.framework.typology import McConfig


def _synthetic_volatile_series(n: int = 5000, seed: int = 0) -> pd.Series:
    """A series with realistic drawdowns -- random walk with regime-switching vol."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    # Inject two crisis windows scaled to series length so short tests still work.
    crisis_len = max(20, n // 50)
    c1_start = n // 5
    c2_start = (3 * n) // 5
    if c1_start + crisis_len <= n:
        rets[c1_start : c1_start + crisis_len] += rng.normal(-0.005, 0.025, crisis_len)
    if c2_start + crisis_len <= n:
        rets[c2_start : c2_start + crisis_len] += rng.normal(-0.003, 0.020, crisis_len)
    px = 100 * np.cumprod(1 + rets)
    return pd.Series(px, index=pd.date_range("2008-01-02", periods=n, freq="B"), name="close")


def _identity_strategy(df: pd.DataFrame) -> pd.Series:
    """Identity: strategy == benchmark (buy-and-hold). Tests the ratio ~= 1.0 case."""
    return df["close"].pct_change().fillna(0.0)


def _half_position_strategy(df: pd.DataFrame) -> pd.Series:
    """Always hold 50% in the asset, 50% in cash. Reduces vol AND drawdown."""
    return 0.5 * df["close"].pct_change().fillna(0.0)


def _inverse_strategy(df: pd.DataFrame) -> pd.Series:
    """Short the asset 100%. Strictly worse than long benchmark."""
    return -df["close"].pct_change().fillna(0.0)


def _benchmark_long(df: pd.DataFrame) -> pd.Series:
    """Standard buy-and-hold benchmark."""
    return df["close"].pct_change().fillna(0.0)


# ── (1) Basic contract ────────────────────────────────────────────────────


def test_relative_mc_returns_proper_struct():
    primary = _synthetic_volatile_series(n=2000)
    cfg = McConfig(
        block_size_bars=21,
        n_paths=30,
        bootstrap_method="block",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )
    result = run_relative_block_mc(
        primary_close=primary,
        cfg=cfg,
        strategy_fn=_identity_strategy,
        benchmark_fn=_benchmark_long,
        periods_per_year=252,
        seed=42,
    )
    assert isinstance(result, RelativeMcResult)
    assert result.n_paths_completed == 30
    # Both halves run on the SAME paths, so their median MaxDDs should be
    # similar (modulo the fact that strategy and benchmark are identical here).
    assert result.median_dd_reduction == 1.0  # identity


# ── (2) Identity strategy: ratio == 1.0, gate fails ───────────────────────


def test_identity_strategy_fails_strict_relative_gate():
    """Strategy == benchmark → ratio = 1.0 everywhere → fails the 0.8 gate."""
    primary = _synthetic_volatile_series()
    cfg = McConfig(
        block_size_bars=63,
        n_paths=50,
        bootstrap_method="block",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )
    result = run_relative_block_mc(
        primary_close=primary,
        cfg=cfg,
        strategy_fn=_identity_strategy,
        benchmark_fn=_benchmark_long,
        periods_per_year=252,
        seed=1,
        median_ratio_gate=0.80,
        p_strategy_better_gate=0.50,
    )
    # Identity strategy: drawdowns are bit-exact equal to benchmark.
    assert result.median_dd_reduction == 1.0
    # p_better counts >= as "better" (not worse), so identity gets credit for every path.
    assert result.p_strategy_better == 1.0
    # But median ratio == 1.0 > 0.80 -- fails the gate.
    assert not result.passes


# ── (3) Half-position strategy: ratio ~ 0.5 (lower MaxDD), passes ─────────


def test_half_position_strategy_passes_relative_gate():
    """Holding only half a position halves the drawdown vs full long.
    Median DD ratio ~ 0.5, comfortably under the 0.8 gate.
    """
    primary = _synthetic_volatile_series()
    cfg = McConfig(
        block_size_bars=63,
        n_paths=50,
        bootstrap_method="block",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )
    result = run_relative_block_mc(
        primary_close=primary,
        cfg=cfg,
        strategy_fn=_half_position_strategy,
        benchmark_fn=_benchmark_long,
        periods_per_year=252,
        seed=2,
    )
    # 50% position reduces drawdown by approximately 50%.
    assert 0.40 <= result.median_dd_reduction <= 0.60, (
        f"Expected ~0.5 DD ratio, got {result.median_dd_reduction}"
    )
    assert result.p_strategy_better >= 0.95
    assert result.passes


# ── (4) Inverse strategy: strictly worse, fails decisively ────────────────


def test_inverse_strategy_fails_decisively():
    """Shorting the underlying when it has positive drift → strategy's
    drawdown is bigger than benchmark's on essentially every path.
    """
    primary = _synthetic_volatile_series()
    cfg = McConfig(
        block_size_bars=63,
        n_paths=50,
        bootstrap_method="block",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )
    result = run_relative_block_mc(
        primary_close=primary,
        cfg=cfg,
        strategy_fn=_inverse_strategy,
        benchmark_fn=_benchmark_long,
        periods_per_year=252,
        seed=3,
    )
    # Strategy MaxDDs are big (it's short a drifting asset). Median ratio > 1.
    assert result.median_dd_reduction > 1.0
    assert not result.passes


# ── (5) Reproducibility under seed ────────────────────────────────────────


def test_relative_mc_reproducible_under_seed():
    primary = _synthetic_volatile_series()
    cfg = McConfig(
        block_size_bars=21,
        n_paths=20,
        bootstrap_method="block",
        max_dd_threshold_pct=0.35,
        max_dd_pass_prob=0.10,
    )
    r1 = run_relative_block_mc(
        primary_close=primary,
        cfg=cfg,
        strategy_fn=_half_position_strategy,
        benchmark_fn=_benchmark_long,
        periods_per_year=252,
        seed=99,
    )
    r2 = run_relative_block_mc(
        primary_close=primary,
        cfg=cfg,
        strategy_fn=_half_position_strategy,
        benchmark_fn=_benchmark_long,
        periods_per_year=252,
        seed=99,
    )
    assert r1.median_dd_reduction == r2.median_dd_reduction
    assert r1.p_strategy_better == r2.p_strategy_better
    assert r1.strategy_median_maxdd == r2.strategy_median_maxdd
