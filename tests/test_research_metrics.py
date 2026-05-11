"""Regression tests for ``titan.research.metrics``.

These tests specifically lock in the behaviours the April 2026 audit fixes
depend on:

* Sharpe does not filter zero-return bars (the old ``nz`` filter overstated
  Sharpe by ``sqrt(1/active_ratio)``).
* ``periods_per_year`` is required — callers must pass the correct factor
  for the bar frequency.
* Rolling z-score is strictly causal — a value at bar ``t`` depends only on
  bars through ``t``.
* Bootstrap CI contains the point estimate on typical sample sizes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from titan.research.metrics import (
    BARS_PER_YEAR,
    annualize_vol,
    bootstrap_sharpe_ci,
    calmar,
    ewm_vol_last,
    is_frozen_zscore,
    max_drawdown,
    rolling_zscore,
    sharpe,
    sortino,
    trade_sharpe,
)

# ── sharpe: basic contracts ────────────────────────────────────────────────


def test_sharpe_constant_returns_zero():
    """Zero-variance series must return 0.0, not NaN or a division error."""
    rets = pd.Series([0.001] * 100)
    assert sharpe(rets, periods_per_year=252) == 0.0


def test_sharpe_insufficient_samples():
    assert sharpe(pd.Series([0.01, 0.02]), periods_per_year=252) == 0.0


def test_sharpe_matches_manual_formula():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 500)
    expected = rets.mean() / rets.std(ddof=1) * math.sqrt(252)
    assert sharpe(rets, periods_per_year=252) == pytest.approx(expected, rel=1e-10)


def test_sharpe_frequency_scaling_identity():
    """If the same P&L shows up as D vs H1 (resampled), Sharpe should be
    self-consistent: daily Sharpe from daily returns equals the H1 Sharpe
    when those H1 returns actually were the daily returns divided across
    24 bars with zero variance filler. This is the invariant the old
    filter-then-annualize bug broke."""
    rng = np.random.default_rng(1)
    # 500 trading days, daily returns
    daily = pd.Series(rng.normal(0.001, 0.01, 500))
    daily_sharpe = sharpe(daily, periods_per_year=252)

    # Now embed those same daily returns in an H1 stream where only 1 bar
    # per day has the return and the other 23 are zero. The H1 Sharpe
    # computed with the WRONG annualisation (sqrt(252)) would equal daily.
    # Computed with the CORRECT H1 annualisation it should be larger by
    # ~sqrt(24) — but only if we keep the zero bars in the denominator.
    h1 = np.zeros(500 * 24)
    h1[::24] = daily.to_numpy()
    h1_sharpe_correct = sharpe(h1, periods_per_year=BARS_PER_YEAR["H1"])

    # The correct H1 Sharpe on a series that is zero 23/24 of the time is
    # lower than the daily Sharpe because the zero bars increase sample
    # size without changing the mean. Specifically:
    #   mean_h1 = mean_daily / 24
    #   std_h1  ≈ std_daily / sqrt(24)   (approximately, since non-zero bars
    #                                     are exactly daily values scaled)
    # So Sharpe_h1 = (mean_d / 24) / (std_d / sqrt(24)) * sqrt(252 * 24)
    #              = mean_d / std_d * sqrt(252) = daily Sharpe
    #
    # i.e. the CORRECT formulation gives identical annual Sharpe regardless
    # of sampling frequency — that's the whole point of annualisation.
    assert h1_sharpe_correct == pytest.approx(daily_sharpe, rel=0.05)


def test_sharpe_filter_zero_days_bias_is_gone():
    """The old bug: filtering ``rets != 0`` before annualising overstated
    Sharpe for sparse strategies by ``sqrt(1/active_ratio)``.

    To isolate the bias, we use the SAME active-day P&L distribution in both
    a dense and a sparse series — the sparse series is the dense one with
    zeros inserted. The correct (unfiltered) Sharpe ratio between the two
    should be ``sqrt(active_ratio)`` (because the zeros dilute the mean by
    that factor and the std by sqrt of that factor, leaving the ratio
    smaller by sqrt). If we were filtering zeros the ratio would be ~1.0.
    """
    rng = np.random.default_rng(2)
    active_ratio = 0.10
    n_active = 500
    n_total = int(n_active / active_ratio)

    active_pnl = rng.normal(0.001, 0.01, n_active)

    # Dense: every bar is active (trade N_active bars back-to-back)
    dense = pd.Series(active_pnl)
    dense_sharpe = sharpe(dense, periods_per_year=252)

    # Sparse: same active-day P&L, but padded to n_total with zeros
    sparse = np.zeros(n_total)
    sparse[:n_active] = active_pnl
    sparse_sharpe = sharpe(sparse, periods_per_year=252)

    # Theoretical ratio: sparse / dense ≈ sqrt(active_ratio) ≈ sqrt(0.1) ≈ 0.316
    # Allow generous tolerance for finite-sample variation.
    ratio = sparse_sharpe / dense_sharpe
    assert 0.2 < ratio < 0.45, (
        f"Sparse/dense Sharpe ratio {ratio:.3f} is not near sqrt({active_ratio})"
        f" ≈ {math.sqrt(active_ratio):.3f} — filtering bias may have returned."
    )


# ── trade_sharpe ────────────────────────────────────────────────────────────


def test_trade_sharpe_basic():
    rng = np.random.default_rng(3)
    trades = rng.normal(0.005, 0.02, 100)
    # e.g. 50 trades per year
    s = trade_sharpe(trades, trades_per_year=50)
    assert s == pytest.approx(trades.mean() / trades.std(ddof=1) * math.sqrt(50), rel=1e-10)


# ── annualize_vol / ewm_vol ────────────────────────────────────────────────


def test_annualize_vol_matches_sqrt():
    assert annualize_vol(0.01, 252) == pytest.approx(0.01 * math.sqrt(252))


def test_ewm_vol_frequency_scaling():
    """EWMA vol on a constant-variance Gaussian stream should roughly match
    the theoretical stdev * sqrt(periods_per_year), and scale correctly
    between frequencies."""
    rng = np.random.default_rng(4)
    sigma = 0.01
    rets = pd.Series(rng.normal(0, sigma, 5000))
    vol_daily = ewm_vol_last(rets, lam=0.94, periods_per_year=252)
    vol_h1 = ewm_vol_last(rets, lam=0.94, periods_per_year=252 * 24)
    # vol_h1 / vol_daily should equal sqrt(24) — the critical invariant the
    # live strategies violated by always using 252.
    assert vol_h1 / vol_daily == pytest.approx(math.sqrt(24), rel=1e-6)


def test_ewm_vol_requires_positive_lam():
    with pytest.raises(ValueError):
        ewm_vol_last([0.0], lam=0.0, periods_per_year=252)
    with pytest.raises(ValueError):
        ewm_vol_last([0.0], lam=1.0, periods_per_year=252)


def test_ewm_vol_last_empty_zero():
    assert ewm_vol_last(pd.Series(dtype=float), lam=0.94, periods_per_year=252) == 0.0


# ── z-score causality ─────────────────────────────────────────────────────


def test_rolling_zscore_is_causal():
    """The value at bar t must depend only on bars [0..t]. We assert this
    by mutating bar t+1 after computing and verifying the first t+1 values
    are unchanged."""
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(0, 1, 200))
    z_original = rolling_zscore(x, window=50)

    # Now perturb bar 100 by a huge amount
    x_perturbed = x.copy()
    x_perturbed.iloc[100] = 1e6
    z_perturbed = rolling_zscore(x_perturbed, window=50)

    # Bars 0..99 must be identical; bars 100+ may differ.
    pd.testing.assert_series_equal(z_original.iloc[:100], z_perturbed.iloc[:100])
    # And bar 100 should clearly differ now:
    assert z_original.iloc[100] != z_perturbed.iloc[100]


def test_is_frozen_zscore_uses_only_is_stats():
    """Appending bars to the end must not change the IS slice's z-scores."""
    rng = np.random.default_rng(6)
    x = pd.Series(rng.normal(0, 1, 200))
    z1 = is_frozen_zscore(x, is_end_idx=100).iloc[:100]

    # Append OOS bars with very different distribution
    x_extended = pd.concat([x.iloc[:100], pd.Series(rng.normal(5, 3, 100))]).reset_index(drop=True)
    z2 = is_frozen_zscore(x_extended, is_end_idx=100).iloc[:100]

    pd.testing.assert_series_equal(z1, z2)


def test_is_frozen_zscore_rejects_bad_idx():
    with pytest.raises(ValueError):
        is_frozen_zscore([1.0, 2.0, 3.0], is_end_idx=0)
    with pytest.raises(ValueError):
        is_frozen_zscore([1.0, 2.0, 3.0], is_end_idx=10)


# ── drawdown / calmar / sortino ─────────────────────────────────────────────


def test_max_drawdown_monotone_negative():
    rets = pd.Series([-0.01, -0.02, -0.01, -0.05])
    dd = max_drawdown(rets)
    assert dd < 0
    # Cumulative product: 0.99 * 0.98 * 0.99 * 0.95 ≈ 0.912; trough at the last
    # bar, peak at start (1.0), so dd ≈ -0.088.
    assert dd == pytest.approx(-0.0880, abs=1e-3)


def test_max_drawdown_empty_zero():
    assert max_drawdown([]) == 0.0


def test_calmar_and_sortino_smoke():
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.001, 0.01, 2000))
    c = calmar(rets, periods_per_year=252)
    s = sortino(rets, periods_per_year=252)
    # Just shape checks — both should be finite floats.
    assert math.isfinite(c)
    assert math.isfinite(s)


# ── bootstrap CI ───────────────────────────────────────────────────────────


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(8)
    rets = pd.Series(rng.normal(0.0005, 0.01, 2000))
    point = sharpe(rets, periods_per_year=252)
    lo, hi = bootstrap_sharpe_ci(rets, periods_per_year=252, n_resamples=500, seed=8)
    assert lo < point < hi


def test_bootstrap_ci_includes_zero_for_noise():
    """White noise should produce a CI that straddles zero at 95%."""
    rng = np.random.default_rng(9)
    rets = pd.Series(rng.normal(0.0, 0.01, 1000))
    lo, hi = bootstrap_sharpe_ci(rets, periods_per_year=252, n_resamples=1000, seed=9)
    assert lo < 0 < hi


def test_bootstrap_ci_excludes_zero_for_strong_signal():
    """With a strong positive mean the 95% CI should be strictly above zero."""
    rng = np.random.default_rng(10)
    # Sharpe ≈ 0.002/0.005 * sqrt(252) ≈ 6.3 — very strong
    rets = pd.Series(rng.normal(0.002, 0.005, 2000))
    lo, hi = bootstrap_sharpe_ci(rets, periods_per_year=252, n_resamples=1000, seed=10)
    assert lo > 0
    assert hi > lo


def test_bootstrap_ci_empty_input():
    assert bootstrap_sharpe_ci([], periods_per_year=252) == (0.0, 0.0)


# ── BARS_PER_YEAR sanity ───────────────────────────────────────────────────


def test_bars_per_year_ratios():
    assert BARS_PER_YEAR["H1"] == BARS_PER_YEAR["D"] * 24
    assert BARS_PER_YEAR["H4"] == BARS_PER_YEAR["D"] * 6
    assert BARS_PER_YEAR["M5"] == BARS_PER_YEAR["H1"] * 12
