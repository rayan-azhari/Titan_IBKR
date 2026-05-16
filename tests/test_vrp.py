"""Unit tests for VRP capture strategy (E1).

Specified in directives/Pre-Reg E1 VRP Capture 2026-05-15.md §6.4.

Contract:
    * Causality: corrupted-future does not leak into past weights.
    * Contango regime: when ratio_long >= gate, VIXY weight = target_short_weight.
    * Backwardation regime: when ratio_long < backwardation_ratio_long,
      VIXY weight = 0 (defensive_long_spy_w applies on the SPY leg if set).
    * Hysteresis: the regime_buffer_pct prevents flip-flops near the gate.
    * Class defaults: defaults_for(DAILY_MEAN_REVERSION) returns the row
      this audit relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.vrp.vrp_strategy import (
    VRP_UNIVERSE,
    VrpConfig,
    vrp_assert_causal,
    vrp_returns,
    vrp_target_weights,
)
from titan.research.framework import StrategyClass, defaults_for

# ── Fixtures ──────────────────────────────────────────────────────────────


def _synthetic_universe(
    n: int = 500, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build a synthetic universe with controllable regime structure.

    First half: stable contango (VIX < VIX3M, VIX9D < VIX). VIXY drifts down.
    Second half: backwardation flare-up (VIX > VIX3M, VIX > VIX9D). VIXY spikes.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    vixy = pd.Series(
        100
        * np.cumprod(
            1
            + np.concatenate(
                [
                    rng.normal(-0.001, 0.02, n // 2),  # contango decay
                    rng.normal(+0.003, 0.04, n - n // 2),  # backwardation rally
                ]
            )
        ),
        index=idx,
        name="VIXY",
    )
    spy = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)),
        index=idx,
        name="SPY",
    )
    vix = pd.Series(
        np.concatenate([rng.uniform(13, 18, n // 2), rng.uniform(25, 45, n - n // 2)]),
        index=idx,
        name="VIX",
    )
    vix9d = pd.Series(
        np.concatenate(
            [
                rng.uniform(11, 16, n // 2),  # < VIX in contango
                rng.uniform(28, 50, n - n // 2),  # > VIX in stress
            ]
        ),
        index=idx,
        name="VIX9D",
    )
    vix3m = pd.Series(
        np.concatenate(
            [
                rng.uniform(15, 22, n // 2),  # > VIX in contango
                rng.uniform(22, 35, n - n // 2),  # < VIX in stress
            ]
        ),
        index=idx,
        name="VIX3M",
    )
    closes = pd.concat([vixy, spy], axis=1)
    return closes, vix, vix9d, vix3m


# ── (1) Basic contracts ───────────────────────────────────────────────────


def test_vrp_universe_constant():
    assert VRP_UNIVERSE == ("VIXY",)


def test_class_defaults_consistency():
    """The audit relies on DAILY_MEAN_REVERSION defaults: per-day MTM
    Sharpe, expanding WFO with 3y IS / 1y OOS / 5 folds, P(MaxDD>25%)<10%.
    """
    d = defaults_for(StrategyClass.DAILY_MEAN_REVERSION)
    assert d.sharpe.primary == "per_day_mtm"
    assert d.sharpe.primary_periods_per_year == 252
    assert d.wfo.fold_count == 5
    assert d.wfo.is_min_years == 3.0
    assert d.wfo.oos_years == 1.0
    assert d.mc.max_dd_threshold_pct == 0.25
    assert d.mc.max_dd_pass_prob == 0.10


# ── (2) Regime logic ──────────────────────────────────────────────────────


def test_contango_regime_assigns_short_vixy_weight():
    """Steep contango (VIX3M/VIX >= 1.05 AND VIX/VIX9D >= 1.0) should
    SHORT VIXY at target_short_weight."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=400)
    cfg = VrpConfig(target_short_weight=-0.50)
    w = vrp_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    # First half is contango -- expect some bars with VIXY weight = -0.50.
    contango_half = w.iloc[50:150]["VIXY"]  # skip warmup
    assert (contango_half == -0.50).any(), (
        f"No contango short positions in first half. Unique weights: {contango_half.unique()}"
    )


def test_backwardation_regime_zeroes_vixy_weight():
    """Strong backwardation (ratio_long < 0.98) should put VIXY weight at 0."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=400)
    cfg = VrpConfig(target_short_weight=-0.50, defensive_long_spy_w=0.5)
    w = vrp_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)
    # Second half is backwardation; expect VIXY=0 in places + SPY=+0.5 where regime is -1.
    backw_half = w.iloc[250:380]
    assert (backw_half["VIXY"] == 0.0).any()
    # SPY overlay should fire at least once in the regime
    assert (backw_half["SPY"] == 0.50).any(), "Defensive SPY overlay never engaged"


def test_buffer_reduces_flipflops():
    """A small regime_buffer_pct >= 0 keeps the strategy stable near gates."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=400)
    w_no_buffer = vrp_target_weights(
        closes, cfg=VrpConfig(regime_buffer_pct=0.0), vix=vix, vix9d=vix9d, vix3m=vix3m
    )
    w_buffer = vrp_target_weights(
        closes,
        cfg=VrpConfig(regime_buffer_pct=0.05),
        vix=vix,
        vix9d=vix9d,
        vix3m=vix3m,
    )
    # Buffer should never INCREASE the number of regime flips.
    flips_no_buf = (w_no_buffer["VIXY"].diff().abs() > 0).sum()
    flips_buf = (w_buffer["VIXY"].diff().abs() > 0).sum()
    assert flips_buf <= flips_no_buf


# ── (3) Causality (A10) ───────────────────────────────────────────────────


def test_vrp_assert_causal_passes_on_clean_data():
    closes, vix, vix9d, vix3m = _synthetic_universe(n=400)
    # Should not raise.
    vrp_assert_causal(closes, vix=vix, vix9d=vix9d, vix3m=vix3m, n_trials=5, seed=42)


def test_weights_use_shift1_signal():
    """The decision at t must use info through t-1 only. Last bar's weight
    can NEVER depend on last bar's close."""
    closes, vix, vix9d, vix3m = _synthetic_universe(n=300)
    cfg = VrpConfig()
    w_base = vrp_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)

    # Corrupt LAST bar's VIX values; weights at all PRIOR bars must be unchanged.
    vix2 = vix.copy()
    vix2.iloc[-1] = vix2.iloc[-1] * 5
    vix9d2 = vix9d.copy()
    vix9d2.iloc[-1] = vix9d2.iloc[-1] * 5
    vix3m2 = vix3m.copy()
    vix3m2.iloc[-1] = vix3m2.iloc[-1] * 5
    w_corrupt = vrp_target_weights(closes, cfg=cfg, vix=vix2, vix9d=vix9d2, vix3m=vix3m2)
    assert w_base.iloc[:-1].equals(w_corrupt.iloc[:-1])


# ── (4) Returns calculation ───────────────────────────────────────────────


def test_vrp_returns_has_proper_index():
    closes, vix, vix9d, vix3m = _synthetic_universe(n=300)
    rets = vrp_returns(closes, cfg=VrpConfig(), vix=vix, vix9d=vix9d, vix3m=vix3m)
    assert isinstance(rets, pd.Series)
    assert rets.index.equals(closes.index)


def test_vrp_returns_zero_when_no_regime_signal():
    """Construct a universe where VIX/VIX9D/VIX3M are all equal -- the
    strategy should never see contango or backwardation -> weight stays 0
    -> returns are 0 (no costs from no rebalances)."""
    idx = pd.date_range("2020-01-02", periods=200, freq="B")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(
        {
            "VIXY": 100 * np.cumprod(1 + rng.normal(0, 0.02, 200)),
            "SPY": 100 * np.cumprod(1 + rng.normal(0, 0.01, 200)),
        },
        index=idx,
    )
    flat = pd.Series(np.full(200, 20.0), index=idx)
    rets = vrp_returns(closes, cfg=VrpConfig(), vix=flat, vix9d=flat, vix3m=flat)
    # Strategy never enters a position -> all rets ~ 0 (modulo first-bar bootstrap).
    assert rets.abs().sum() < 1e-9
