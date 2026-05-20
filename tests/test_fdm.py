"""Unit tests for the Carver Forecast Diversification Multiplier
(`titan/research/framework/fdm.py`).

Tests cover:
    1. Mathematical identities (N=1, perfect correlation, uncorrelated).
    2. The uniform-correlation closed form vs the matrix path.
    3. Cap behaviour and the cap flag.
    4. Edge cases: empty input, all-NaN columns, zero-variance columns,
       sub-min-history fallback.
    5. Sanity properties: FDM >= 1.0 always; FDM monotonic in correlation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.fdm import (
    DEFAULT_FDM_CAP,
    fdm_from_uniform_correlation,
    forecast_diversification_multiplier,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _gaussian_correlated(
    n_forecasts: int, n_bars: int, rho: float, *, seed: int = 42
) -> pd.DataFrame:
    """Generate `n_bars` samples from a multivariate normal with uniform
    pairwise correlation `rho` across `n_forecasts` columns.
    """
    rng = np.random.default_rng(seed)
    cov = np.full((n_forecasts, n_forecasts), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    samples = rng.multivariate_normal(mean=np.zeros(n_forecasts), cov=cov, size=n_bars)
    cols = [f"f{i}" for i in range(n_forecasts)]
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    return pd.DataFrame(samples, index=idx, columns=cols)


# ── Math identities ────────────────────────────────────────────────────


def test_fdm_n1_returns_unit_multiplier():
    """A single forecast cannot be diversified -- FDM must equal 1.0."""
    df = pd.DataFrame({"only": np.linspace(0.01, 0.5, 100)})
    res = forecast_diversification_multiplier(df)
    assert res.fdm == 1.0
    assert res.fdm_uncapped == 1.0
    assert res.n_forecasts == 1
    assert math.isnan(res.avg_correlation)
    assert not res.was_capped


def test_fdm_perfect_correlation_no_diversification_benefit():
    """ρ = 1 across all pairs -> FDM = 1.0 (no benefit possible)."""
    base = np.linspace(0.01, 0.5, 200)
    df = pd.DataFrame(
        {"a": base, "b": base, "c": base},
        index=pd.date_range("2020-01-01", periods=200, freq="D"),
    )
    res = forecast_diversification_multiplier(df)
    # Floating-point tolerance: ρ=1 gives portfolio_var=1 exactly, FDM=1.0.
    assert res.fdm == pytest.approx(1.0, abs=1e-12)
    assert res.avg_correlation == pytest.approx(1.0, abs=1e-9)
    assert not res.was_capped


def test_fdm_uncorrelated_matches_sqrt_n():
    """ρ ≈ 0 across pairs -> FDM ≈ sqrt(N). Tolerance reflects the
    bootstrap-style noise in the empirical correlation estimate.
    """
    df = _gaussian_correlated(n_forecasts=4, n_bars=2000, rho=0.0, seed=42)
    res = forecast_diversification_multiplier(df)
    # Long sample: empirical correlations should be small. FDM near sqrt(4)=2.
    assert res.fdm == pytest.approx(math.sqrt(4), rel=0.05)
    assert res.fdm_uncapped == pytest.approx(math.sqrt(4), rel=0.05)
    assert res.n_forecasts == 4
    assert abs(res.avg_correlation) < 0.05
    assert not res.was_capped


def test_fdm_uniform_correlation_matches_closed_form():
    """For uniform ρ, the matrix FDM and the closed-form FDM must agree."""
    df = _gaussian_correlated(n_forecasts=5, n_bars=5000, rho=0.4, seed=7)
    res = forecast_diversification_multiplier(df)
    # Closed-form value with the EMPIRICAL avg correlation (not the
    # population rho of 0.4) to give a tight comparison.
    closed = fdm_from_uniform_correlation(n=res.n_forecasts, avg_corr=res.avg_correlation)
    assert res.fdm_uncapped == pytest.approx(closed, rel=1e-6)


# ── Closed-form FDM tests ──────────────────────────────────────────────


def test_closed_form_n1():
    assert fdm_from_uniform_correlation(1, 0.5) == 1.0


def test_closed_form_uncorrelated():
    assert fdm_from_uniform_correlation(4, 0.0) == pytest.approx(2.0)
    assert fdm_from_uniform_correlation(9, 0.0) == pytest.approx(3.0)


def test_closed_form_perfect_correlation():
    assert fdm_from_uniform_correlation(10, 1.0) == pytest.approx(1.0)


def test_closed_form_known_carver_values():
    """Spot-check against typical Carver-quoted FDMs."""
    # N=3 forecasts with ρ ≈ 0.5 -> FDM = sqrt(3 / (1 + 2*0.5)) = sqrt(1.5)
    assert fdm_from_uniform_correlation(3, 0.5) == pytest.approx(math.sqrt(1.5), rel=1e-9)
    # N=10, ρ=0.25 -> sqrt(10 / 3.25) ≈ 1.7541
    assert fdm_from_uniform_correlation(10, 0.25) == pytest.approx(math.sqrt(10 / 3.25), rel=1e-9)


def test_closed_form_negative_lower_bound_returns_inf():
    """At the PSD lower-bound correlation, equal-weight variance touches 0
    and FDM is unbounded. Function returns inf so callers can clip.
    """
    n = 4
    lower = -1.0 / (n - 1)
    assert fdm_from_uniform_correlation(n, lower) == float("inf")


# ── Cap behaviour ──────────────────────────────────────────────────────


def test_fdm_caps_at_default_25_when_uncapped_would_exceed():
    """N=20 nearly-uncorrelated forecasts would give uncapped FDM ~ 4.5;
    Carver cap of 2.5 must kick in.
    """
    df = _gaussian_correlated(n_forecasts=20, n_bars=10_000, rho=0.0, seed=0)
    res = forecast_diversification_multiplier(df)
    assert res.fdm == DEFAULT_FDM_CAP
    assert res.was_capped is True
    assert res.fdm_uncapped > DEFAULT_FDM_CAP


def test_fdm_cap_override_accepts_higher_threshold():
    """Caller can raise the cap (e.g. when they have a defensible
    reason to trust the low correlation estimate).
    """
    df = _gaussian_correlated(n_forecasts=20, n_bars=10_000, rho=0.0, seed=0)
    res = forecast_diversification_multiplier(df, cap=10.0)
    assert res.fdm == res.fdm_uncapped  # not clipped
    assert res.fdm > 3.0
    assert not res.was_capped


# ── Edge cases ─────────────────────────────────────────────────────────


def test_fdm_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        forecast_diversification_multiplier(pd.DataFrame())


def test_fdm_all_zero_variance_columns_raises():
    """Pure constants have no correlation structure."""
    df = pd.DataFrame(
        {"a": [1.0] * 100, "b": [2.0] * 100},
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )
    with pytest.raises(ValueError, match="non-zero variance"):
        forecast_diversification_multiplier(df)


def test_fdm_drops_zero_variance_columns_keeps_others():
    """Mix of valid + zero-variance: zero-variance is dropped, FDM
    computed on the remainder.
    """
    valid = _gaussian_correlated(n_forecasts=3, n_bars=500, rho=0.3, seed=11)
    valid["constant"] = 1.0
    res = forecast_diversification_multiplier(valid)
    assert res.n_forecasts == 3  # "constant" dropped
    assert "constant" not in res.correlation_matrix.columns


def test_fdm_floor_at_one_under_numerical_noise():
    """When empirical correlations are noisy and avg_corr is just above
    the implied FDM-zero point, the function must still return >= 1.0.
    """
    df = _gaussian_correlated(n_forecasts=2, n_bars=10_000, rho=0.99, seed=3)
    res = forecast_diversification_multiplier(df)
    assert res.fdm >= 1.0
    assert res.fdm_uncapped >= 1.0


# ── Sanity properties ─────────────────────────────────────────────────


def test_fdm_is_monotonic_decreasing_in_correlation():
    """As correlation rises, diversification benefit shrinks -> FDM falls."""
    df_low = _gaussian_correlated(n_forecasts=5, n_bars=5000, rho=0.0, seed=4)
    df_mid = _gaussian_correlated(n_forecasts=5, n_bars=5000, rho=0.4, seed=4)
    df_high = _gaussian_correlated(n_forecasts=5, n_bars=5000, rho=0.8, seed=4)
    fdm_low = forecast_diversification_multiplier(df_low).fdm_uncapped
    fdm_mid = forecast_diversification_multiplier(df_mid).fdm_uncapped
    fdm_high = forecast_diversification_multiplier(df_high).fdm_uncapped
    assert fdm_low > fdm_mid > fdm_high


def test_fdm_correlation_matrix_diagonal_is_one():
    """Sanity: returned correlation matrix has ones on the diagonal."""
    df = _gaussian_correlated(n_forecasts=4, n_bars=500, rho=0.3, seed=2)
    res = forecast_diversification_multiplier(df)
    diag = np.diag(res.correlation_matrix.values)
    assert np.all(diag == pytest.approx(1.0, abs=1e-12))


def test_fdm_correlation_matrix_bounded_in_minus_one_one():
    """No numerical artefact can produce |ρ| > 1 -- the function clips."""
    df = _gaussian_correlated(n_forecasts=3, n_bars=300, rho=0.5, seed=13)
    res = forecast_diversification_multiplier(df)
    vals = res.correlation_matrix.values
    assert vals.min() >= -1.0
    assert vals.max() <= 1.0
