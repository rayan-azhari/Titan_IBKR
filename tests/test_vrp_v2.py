"""Unit tests for VRP v2 (percentile-gate redesign, E1b).

Specified in directives/Pre-Reg E1b VRP Capture v2 Percentile Gates 2026-05-15.md.

Contract:
    * Causality: corrupted-future does not leak into past weights.
    * Class defaults: DAILY_MEAN_REVERSION_VOL_CARRY has P(MaxDD>50%)<10%.
    * Percentile gate: enters short when ratio_long >= rolling Q[enter_q].
    * Continuous sigmoid (C7): position fades smoothly across the threshold.
    * Shift discipline: last bar's signal cannot affect prior-bar weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.vrp_v2.vrp_v2_strategy import (
    VRP_V2_UNIVERSE,
    VrpV2Config,
    vrp_v2_assert_causal,
    vrp_v2_returns,
    vrp_v2_target_weights,
)
from titan.research.framework import StrategyClass, defaults_for

# ── Fixtures ──────────────────────────────────────────────────────────────


def _synthetic_universe(n: int = 800, seed: int = 0):
    """Build a longer synthetic universe (need >= window_d + warmup bars)
    with a noisy regime structure that produces both contango + backw'n bars."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    vixy = pd.Series(100 * np.cumprod(1 + rng.normal(-0.0005, 0.03, n)), index=idx, name="VIXY")
    # Build ratio_long ~ trending around 1.05 with regime regions.
    base = np.linspace(1.0, 1.10, n)
    noise = rng.normal(0.0, 0.02, n)
    regime_pulse = np.zeros(n)
    regime_pulse[n // 2 : n // 2 + 50] = -0.20  # backwardation pulse mid-series
    target_ratio = base + noise + regime_pulse
    vix = pd.Series(20.0 / target_ratio, index=idx, name="VIX")
    vix3m = pd.Series(20.0 * np.ones(n), index=idx, name="VIX3M")
    vix9d = pd.Series(18.0 * np.ones(n), index=idx, name="VIX9D")
    closes = pd.DataFrame({"VIXY": vixy})
    return closes, vix, vix9d, vix3m


# ── (1) Class defaults ────────────────────────────────────────────────────


def test_vol_carry_class_defaults():
    """L25: DAILY_MEAN_REVERSION_VOL_CARRY has relaxed MC threshold."""
    d = defaults_for(StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY)
    assert d.sharpe.primary == "per_day_mtm"
    assert d.sharpe.primary_periods_per_year == 252
    # The class-specific point of this sub-class:
    assert d.mc.max_dd_threshold_pct == 0.50
    assert d.mc.max_dd_pass_prob == 0.10
    # WFO same as parent class.
    assert d.wfo.fold_count == 5
    assert d.wfo.is_min_years == 3.0


def test_vol_carry_class_distinct_from_parent():
    """DAILY_MEAN_REVERSION_VOL_CARRY and DAILY_MEAN_REVERSION have
    distinct MC thresholds — this is the whole point of L25."""
    a = defaults_for(StrategyClass.DAILY_MEAN_REVERSION)
    b = defaults_for(StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY)
    assert a.mc.max_dd_threshold_pct != b.mc.max_dd_threshold_pct


def test_vrp_v2_universe_constant():
    assert VRP_V2_UNIVERSE == ("VIXY",)


# ── (2) Percentile gate logic ─────────────────────────────────────────────


def test_percentile_gate_produces_some_positions():
    """With a strongly trending ratio_long, the rolling-percentile gate
    should produce at least some short positions in the upper-quantile region."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=800)
    cfg = VrpV2Config(gate_kind="percentile", window_d=252, enter_q=0.60, exit_q=0.40)
    w = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    # After warmup (~252 bars), there should be some non-zero positions.
    post_warmup = w.iloc[300:]
    assert (post_warmup["VIXY"] < 0).any(), (
        f"No short positions after warmup. Unique weights: {post_warmup['VIXY'].unique()}"
    )


def test_percentile_gate_zero_during_warmup():
    """Rolling quantile is NaN during warmup -> weight stays at 0."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=400)
    cfg = VrpV2Config(window_d=252, enter_q=0.60, exit_q=0.40)
    w = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    # First few bars (definitely before warmup completes at 252) -> 0.
    assert (w.iloc[:50]["VIXY"] == 0.0).all()


def test_wider_band_produces_fewer_regime_flips_than_narrower():
    """Wider band (enter=0.70, exit=0.30) has MORE hysteresis -> fewer
    state transitions than the narrower band (0.55/0.45). Total in-regime
    BAR count is not monotone in band width (once entered, wider band
    stays in longer), but the count of FLIPS is.
    """
    closes, vix, vix9d, vix3m = _synthetic_universe(n=800)
    w_wide = vrp_v2_target_weights(
        closes,
        cfg=VrpV2Config(enter_q=0.70, exit_q=0.30),
        vix=vix,
        vix9d=vix9d,
        vix3m=vix3m,
    )
    w_narrow = vrp_v2_target_weights(
        closes,
        cfg=VrpV2Config(enter_q=0.55, exit_q=0.45),
        vix=vix,
        vix9d=vix9d,
        vix3m=vix3m,
    )
    flips_wide = (w_wide["VIXY"].diff().abs() > 1e-9).sum()
    flips_narrow = (w_narrow["VIXY"].diff().abs() > 1e-9).sum()
    assert flips_wide <= flips_narrow


# ── (3) Continuous sigmoid (C7) ──────────────────────────────────────────


def test_sigmoid_gate_produces_continuous_weights():
    """C7's position should take on a CONTINUOUS range of values, not just
    {0, target_short_weight} like the discrete percentile gate."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=800)
    cfg = VrpV2Config(gate_kind="sigmoid", sigmoid_scale=0.05)
    w = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    post_warmup = w.iloc[300:]["VIXY"]
    # Should have more than 2 unique values (i.e. not just 0 and -0.5)
    unique_count = post_warmup.round(4).unique().size
    assert unique_count > 5, (
        f"Sigmoid gate produced only {unique_count} unique weights; expected continuous range"
    )


def test_sigmoid_weights_bounded():
    """Sigmoid output * target_short_weight ∈ [target_short_weight, 0]."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=800)
    cfg = VrpV2Config(gate_kind="sigmoid", target_short_weight=-0.50, sigmoid_scale=0.05)
    w = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    assert (w["VIXY"] >= -0.50 - 1e-9).all()
    assert (w["VIXY"] <= 0.0 + 1e-9).all()


# ── (4) Causality (A10) ──────────────────────────────────────────────────


def test_vrp_v2_assert_causal_passes():
    closes, vix, vix9d, vix3m = _synthetic_universe(n=800)
    vrp_v2_assert_causal(closes, vix=vix, vix9d=vix9d, vix3m=vix3m, n_trials=5, seed=42)


def test_last_bar_perturbation_does_not_affect_prior_weights():
    """Corrupting the last bar of every signal series must not change any
    prior bar's weight."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=600)
    cfg = VrpV2Config(window_d=252, enter_q=0.60, exit_q=0.40)
    w_base = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    vix2 = vix.copy()
    vix2.iloc[-1] = vix2.iloc[-1] * 5
    vix9d2 = vix9d.copy()
    vix9d2.iloc[-1] = vix9d2.iloc[-1] * 5
    vix3m2 = vix3m.copy()
    vix3m2.iloc[-1] = vix3m2.iloc[-1] * 5
    w_corrupt = vrp_v2_target_weights(closes, cfg=cfg, vix=vix2, vix9d=vix9d2, vix3m=vix3m2)
    assert w_base.iloc[:-1].equals(w_corrupt.iloc[:-1])


# ── (5) Returns calculation ──────────────────────────────────────────────


def test_returns_has_proper_index():
    closes, vix, vix9d, vix3m = _synthetic_universe(n=600)
    rets = vrp_v2_returns(closes, cfg=VrpV2Config(), vix=vix, vix9d=vix9d, vix3m=vix3m)
    assert isinstance(rets, pd.Series)
    assert rets.index.equals(closes.index)


def test_returns_zero_when_signal_never_fires():
    """If ratio_long is STRICTLY DECREASING, the rolling-percentile gate
    never fires (today's ratio < all historical ratios at the chosen
    quantile). Weights stay at 0 and returns are 0.
    """
    idx = pd.date_range("2020-01-02", periods=600, freq="B")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame({"VIXY": 100 * np.cumprod(1 + rng.normal(0, 0.02, 600))}, index=idx)
    # ratio_long = vix3m / vix is strictly decreasing -> rolling Q[0.6] is
    # always above the current ratio -> gate never fires.
    decreasing = pd.Series(np.linspace(2.0, 1.0, 600), index=idx)
    flat = pd.Series(np.full(600, 1.0), index=idx)
    rets = vrp_v2_returns(closes, cfg=VrpV2Config(), vix=flat, vix9d=flat, vix3m=decreasing)
    assert rets.abs().sum() < 1e-6
