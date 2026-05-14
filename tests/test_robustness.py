"""Unit tests for the noise-injection robustness gate.

Specified in directives/Strategy Backlog 2026-05-14.md §J3.

Contract:
    * A strategy with a real, durable edge passes the gate (degradation < threshold).
    * A strategy whose Sharpe is a single-realisation artefact fails the gate.
    * The base Sharpe under no perturbation must match what `sharpe(...)` would return.
    * The function tolerates strategy_fn raising on perturbed input
      (it scores those trials as 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.robustness import (
    NoiseConfig,
    NoiseRobustnessResult,
    run_noise_robustness,
)
from titan.research.metrics import sharpe


def _synthetic_trending_closes(n: int = 504, seed: int = 0, drift: float = 0.15) -> pd.DataFrame:
    """2y daily series with a strong drift. Sharpe ~1.5 for a trivial long-and-hold."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    px = 100 * np.cumprod(1 + rng.normal(drift / 252, 0.012, n))
    return pd.DataFrame({"PX": px}, index=idx)


def _long_and_hold_strategy(closes: pd.DataFrame) -> pd.Series:
    """Trivial strategy: long the one column every day, no costs."""
    col = closes.columns[0]
    rets = closes[col].pct_change().fillna(0.0)
    return rets


def _fragile_threshold_strategy(closes: pd.DataFrame) -> pd.Series:
    """A tightly-tuned MR threshold strategy.

    Long only when a 20-bar causal z-score is in a NARROW band
    [-1.05, -1.00]. On any specific seed/realisation a curve-fitted narrow
    band can produce a positive Sharpe, but the band is so tight that
    any noise on the prices shuffles which bars trigger -- the "edge"
    evaporates. This is the canonical parameter-spike-not-plateau pattern
    that V3.2 warns about; the noise-injection gate must catch it.
    """
    col = closes.columns[0]
    px = closes[col]
    # Causal: lagged mean and std (shift(1) on the inputs).
    mean = px.shift(1).rolling(20, min_periods=20).mean()
    std = px.shift(1).rolling(20, min_periods=20).std()
    z = (px.shift(1) - mean) / std
    pos = ((z >= -1.05) & (z <= -1.00)).astype(float)
    rets = px.pct_change().fillna(0.0)
    return pos * rets


# ── (1) Basic contract ────────────────────────────────────────────────────


def test_noise_robustness_returns_proper_struct():
    closes = _synthetic_trending_closes()
    result = run_noise_robustness(
        closes,
        _long_and_hold_strategy,
        periods_per_year=252,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3), n_trials=5),
    )
    assert isinstance(result, NoiseRobustnessResult)
    assert len(result.per_level) == 2
    assert result.per_level[0].n_trials == 5
    # Base Sharpe must match the direct call.
    direct = float(sharpe(_long_and_hold_strategy(closes), periods_per_year=252))
    assert result.base_sharpe == pytest.approx(direct, abs=1e-3)


def test_noise_robustness_rejects_invalid_method():
    closes = _synthetic_trending_closes()
    with pytest.raises(ValueError, match="method"):
        run_noise_robustness(
            closes,
            _long_and_hold_strategy,
            periods_per_year=252,
            cfg=NoiseConfig(method="bogus"),
        )


# ── (2) Robust strategy passes ────────────────────────────────────────────


def test_robust_long_and_hold_passes_gate():
    """Long-and-hold on a strong-drift series is the canonical 'robust to
    small input noise' baseline. Should pass at the default max_degradation=0.3.
    """
    closes = _synthetic_trending_closes(n=504, seed=1, drift=0.15)
    result = run_noise_robustness(
        closes,
        _long_and_hold_strategy,
        periods_per_year=252,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.3),
    )
    assert result.passes, (
        f"Long-and-hold failed the gate. base={result.base_sharpe}, "
        f"per-level={[(r.noise_level, r.degradation_mean) for r in result.per_level]}"
    )


# ── (3) Fragile tightly-tuned strategy fails ──────────────────────────────


def test_fragile_threshold_fails_gate():
    """A tightly-tuned MR threshold strategy may have a positive base Sharpe
    on a specific seed but is fragile because the narrow band is essentially
    curve-fit. Under noise its edge collapses. The gate must catch this.

    We search a small seed grid to find one where the base Sharpe is
    non-trivially positive (the curve-fit "edge"); we then verify the gate
    rejects it under noise. This is realistic — researchers do select seeds
    that look good and the gate is designed precisely to catch that.
    """
    found = False
    for seed in range(20):
        closes = _synthetic_trending_closes(n=1008, seed=seed, drift=0.03)
        base = float(sharpe(_fragile_threshold_strategy(closes), periods_per_year=252))
        if base > 0.5:
            found = True
            break
    if not found:
        pytest.skip("No seed produced a positive base Sharpe for the fragile threshold strategy.")

    result = run_noise_robustness(
        closes,
        _fragile_threshold_strategy,
        periods_per_year=252,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=15, max_degradation=0.3),
    )
    assert result.base_sharpe > 0.5
    # Must FAIL — degradation at the highest level should breach the 0.3 threshold.
    assert not result.passes, (
        f"Fragile strategy passed the gate! base={result.base_sharpe} "
        f"per-level={[(r.noise_level, r.degradation_mean) for r in result.per_level]}"
    )


# ── (4) Monotonicity expectation: bigger noise = more degradation ──────────


def test_degradation_is_monotone_in_noise_level():
    """For a generic strategy, degradation should weakly increase with σ.

    This isn't a hard mathematical guarantee for path-dependent strategies,
    but for long-and-hold over Gaussian noise it should be cleanly monotone.
    """
    closes = _synthetic_trending_closes(n=504, seed=3, drift=0.15)
    result = run_noise_robustness(
        closes,
        _long_and_hold_strategy,
        periods_per_year=252,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=15),
    )
    degs = [r.degradation_mean for r in result.per_level]
    # Permit small numerical jitter via tolerance.
    assert degs[1] >= degs[0] - 0.05
    assert degs[2] >= degs[1] - 0.05


# ── (5) Strategy failures under noise score as Sharpe=0, no crash ─────────


def test_strategy_raising_under_noise_scores_zero():
    """If the strategy raises on perturbed input, the gate must NOT crash —
    it scores that trial as Sharpe=0 (graceless degradation).
    """
    closes = _synthetic_trending_closes()

    raise_count = {"n": 0}

    def flaky_strategy(perturbed: pd.DataFrame) -> pd.Series:
        # Raise on every call after the first (base call).
        raise_count["n"] += 1
        if raise_count["n"] == 1:
            return _long_and_hold_strategy(perturbed)
        raise RuntimeError("simulated strategy crash under noise")

    result = run_noise_robustness(
        closes,
        flaky_strategy,
        periods_per_year=252,
        cfg=NoiseConfig(noise_levels=(0.3,), n_trials=3),
    )
    # All three trials at the noise level should have scored 0.
    assert result.per_level[0].sharpes == (0.0, 0.0, 0.0)
