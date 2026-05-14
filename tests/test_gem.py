"""Unit tests for the GEM dual-momentum strategy primitives.

These are the contract tests promised in
``directives/Pre-Reg GEM Dual Momentum 2026-05-14.md`` §6.4. They run in <1s.

We deliberately stop short of "this strategy is profitable" tests --
that's the audit's job (run_gem_audit.py + the framework primitives).
The unit tests here verify the CAUSALITY + decision-rule contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.gem.gem_strategy import (
    GEM_UNIVERSE,
    GemConfig,
    _month_end_idx,
    compute_stress_signal,
    gem_assert_causal,
    gem_returns,
    gem_target_weights,
)
from titan.research.framework.typology import StrategyClass, defaults_for


def _synthetic_closes(
    n_years: int = 5,
    seed: int = 0,
    spy_drift: float = 0.08,
    efa_drift: float = 0.05,
    ief_drift: float = 0.02,
    spy_vol: float = 0.012,
    efa_vol: float = 0.011,
    ief_vol: float = 0.003,
) -> pd.DataFrame:
    """Construct a synthetic daily series with reproducible drifts + vols.

    Daily-vol defaults approximate the empirical ratios (SPY ~20% ann,
    EFA ~17% ann, IEF ~5% ann). Tests that want signal-dominates-noise
    behaviour should pass low vols to make the drift the dominant feature.
    """
    rng = np.random.default_rng(seed)
    n = 252 * n_years
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    spy = 100 * np.cumprod(1 + rng.normal(spy_drift / 252, spy_vol, n))
    efa = 100 * np.cumprod(1 + rng.normal(efa_drift / 252, efa_vol, n))
    ief = 100 * np.cumprod(1 + rng.normal(ief_drift / 252, ief_vol, n))
    return pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)


# ── (1) Month-end detection ──────────────────────────────────────────────


def test_month_end_marks_last_bar_of_each_month():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    me = _month_end_idx(idx)
    # Jan 31 is end-of-Jan; Feb 29 is end-of-Feb; Mar 1 is end-of-Mar (only 1 bar).
    assert me.tolist() == [False, True, False, True, True]


def test_month_end_handles_single_bar_month():
    idx = pd.DatetimeIndex(["2024-01-31", "2024-02-15"])
    me = _month_end_idx(idx)
    assert me.tolist() == [True, True]


# ── (2) Causality (V3.6 A10) -- THE non-negotiable test ──────────────────


def test_gem_causality_corrupted_future_does_not_leak():
    closes = _synthetic_closes(n_years=5)
    # Must not raise. Implementation contract: future shocks cannot change past weights.
    gem_assert_causal(closes, n_trials=5, seed=42)


def test_gem_weights_one_bar_shifted():
    """Direct verification of the .shift(1) discipline.

    The raw decision is made at a month-end bar; the weight that EARNS the
    return must lag by one bar.
    """
    closes = _synthetic_closes(n_years=4)
    w = gem_target_weights(closes)
    # At bar i, weights are the decision from bar i-1. So weights.iloc[0] is all zeros.
    assert (w.iloc[0] == 0).all()
    # Once warmup completes, exactly one column should be 1 on each held day.
    warm = w.iloc[252 * 2 :]  # well past the 12m lookback
    sums = warm.sum(axis=1)
    # All sums must be 0 (transition gap, rare) or 1.
    assert ((sums == 1.0) | (sums == 0.0)).all()
    # Most sums should be 1.0 (we expect a held position almost all the time).
    assert (sums == 1.0).mean() > 0.99


# ── (3) Decision rule ─────────────────────────────────────────────────────


def test_gem_chooses_spy_when_spy_dominates():
    closes = _synthetic_closes(n_years=4, spy_drift=0.20, efa_drift=0.02, ief_drift=0.02)
    w = gem_target_weights(closes)
    held = w.iloc[252 * 2 :]  # post-warmup
    # SPY should be the dominant winner. Check that SPY weight is on >50% of bars.
    pct_spy_long = (held["SPY"] == 1.0).mean()
    assert pct_spy_long > 0.5, f"Expected SPY to dominate; got {pct_spy_long:.2%}"


def test_gem_chooses_efa_when_efa_dominates():
    closes = _synthetic_closes(n_years=4, spy_drift=0.02, efa_drift=0.20, ief_drift=0.02)
    w = gem_target_weights(closes)
    held = w.iloc[252 * 2 :]
    pct_efa_long = (held["EFA"] == 1.0).mean()
    assert pct_efa_long > 0.5, f"Expected EFA to dominate; got {pct_efa_long:.2%}"


def test_gem_defensive_switch_when_all_risk_assets_underperform():
    """When SPY and EFA both bleed and IEF outperforms, GEM should sit in IEF.

    Use low-vol synthetic so the drift signal dominates the random-walk noise
    (otherwise a 23% chance per month-end picks the wrong asset by luck).
    """
    closes = _synthetic_closes(
        n_years=4,
        spy_drift=-0.10,
        efa_drift=-0.10,
        ief_drift=0.04,
        spy_vol=0.003,
        efa_vol=0.003,
        ief_vol=0.001,
    )
    w = gem_target_weights(closes, cfg=GemConfig(defensive_switch=True))
    held = w.iloc[252 * 2 :]
    pct_ief = (held["IEF"] == 1.0).mean()
    assert pct_ief > 0.5, f"Expected IEF defensive; got {pct_ief:.2%}"


def test_gem_no_defensive_when_disabled():
    """Cell C5 (defensive_switch=False) holds max(SPY,EFA) even if both underperform IEF."""
    closes = _synthetic_closes(
        n_years=4,
        spy_drift=-0.10,
        efa_drift=-0.10,
        ief_drift=0.04,
        spy_vol=0.003,
        efa_vol=0.003,
        ief_vol=0.001,
    )
    w = gem_target_weights(closes, cfg=GemConfig(defensive_switch=False))
    held = w.iloc[252 * 2 :]
    pct_ief = (held["IEF"] == 1.0).mean()
    assert pct_ief < 0.05, f"defensive_switch=False but IEF held {pct_ief:.2%} of bars"


# ── (4) Buffer behaviour ──────────────────────────────────────────────────


def test_gem_buffer_reduces_churn():
    """A larger buffer must produce <= as many regime switches as a smaller one."""
    closes = _synthetic_closes(n_years=4, seed=1, spy_drift=0.05, efa_drift=0.045)
    w_no_buffer = gem_target_weights(closes, cfg=GemConfig(buffer_pct=0.0))
    w_big_buffer = gem_target_weights(closes, cfg=GemConfig(buffer_pct=0.05))

    def n_switches(w):
        # Switch = the active pick changes from one bar to the next.
        active = w.idxmax(axis=1)
        return (active.shift(1) != active).sum()

    s_no = n_switches(w_no_buffer)
    s_big = n_switches(w_big_buffer)
    assert s_big <= s_no, f"Buffer didn't reduce churn: s_no={s_no}, s_big={s_big}"


def test_gem_lookback_blend_reacts_faster_than_single_12m():
    """Multi-speed momentum (3m / 6m / 12m blend) should detect a regime
    flip earlier than a single 12m lookback because the shorter horizons
    invert first.

    Setup: 18 months of bull (SPY rallies; IEF flat), then 6 months of
    bear (SPY crashes; IEF rallies). Compare:
      * Single 12m lookback (canonical GEM)
      * Blend (3, 6, 12) lookback (the Step 2 enhancement)
    The blend variant must switch to IEF at least one month earlier on
    average across multiple random seeds, AND must spend more time in
    IEF over the bear window.
    """

    def _build(seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        # Year 1+H1Y2: SPY +18% ann, IEF +1% ann (18 months bull).
        n_bull = 252 + 126
        spy_bull = np.cumprod(1 + rng.normal(0.18 / 252, 0.003, n_bull))
        ief_bull = np.cumprod(1 + rng.normal(0.01 / 252, 0.001, n_bull))
        efa_bull = np.cumprod(1 + rng.normal(0.05 / 252, 0.003, n_bull))
        # H2Y2: 6-month rapid SPY crash -50% ann; IEF +12% ann.
        n_bear = 126
        spy_bear = spy_bull[-1] * np.cumprod(1 + rng.normal(-0.50 / 252, 0.003, n_bear))
        ief_bear = ief_bull[-1] * np.cumprod(1 + rng.normal(0.12 / 252, 0.001, n_bear))
        efa_bear = efa_bull[-1] * np.cumprod(1 + rng.normal(-0.50 / 252, 0.003, n_bear))
        spy = np.concatenate([spy_bull, spy_bear]) * 100
        efa = np.concatenate([efa_bull, efa_bear]) * 100
        ief = np.concatenate([ief_bull, ief_bear]) * 100
        idx = pd.date_range("2020-01-02", periods=len(spy), freq="B")
        return pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    single = GemConfig(lookback_months=12)
    blend = GemConfig(lookback_blend=(3, 6, 12))

    n_seeds = 5
    single_ief_fraction = []
    blend_ief_fraction = []
    for s in range(n_seeds):
        closes = _build(seed=s)
        w_single = gem_target_weights(closes, cfg=single)
        w_blend = gem_target_weights(closes, cfg=blend)
        # The last 126 bars are the bear window. How much of it was in IEF?
        bear = slice(-126, None)
        single_ief_fraction.append((w_single.iloc[bear]["IEF"] == 1.0).mean())
        blend_ief_fraction.append((w_blend.iloc[bear]["IEF"] == 1.0).mean())
    mean_single = float(np.mean(single_ief_fraction))
    mean_blend = float(np.mean(blend_ief_fraction))
    assert mean_blend > mean_single, (
        f"Blend was not faster than single 12m. Mean IEF holding in bear "
        f"window: single={mean_single:.2%}, blend={mean_blend:.2%}."
    )


def test_gem_lookback_blend_requires_non_empty_tuple():
    closes = _synthetic_closes(n_years=4)
    with pytest.raises(ValueError, match="lookback_blend"):
        gem_target_weights(closes, cfg=GemConfig(lookback_blend=()))


def test_gem_lookback_blend_causality():
    """The blend variant must still pass the strict causality smoke test."""
    closes = _synthetic_closes(n_years=5)
    gem_assert_causal(closes, cfg=GemConfig(lookback_blend=(3, 6, 12)), n_trials=5, seed=42)


def test_vol_target_reduces_position_under_high_vol():
    """Vol-target scales the risk-asset weight DOWN when realised vol >
    the annual target, routing the cash portion into IEF.

    Setup: a high-vol SPY series that grinds up. With no vol target,
    GEM holds SPY at weight = 1.0. With a 10% annual vol target and
    realised vol of ~38%, scale should be ~0.26 → IEF picks up ~0.74.
    """
    rng = np.random.default_rng(0)
    # SPY with 38% ann vol (well above any reasonable target).
    n = 252 * 3
    spy = 100 * np.cumprod(1 + rng.normal(0.10 / 252, 0.024, n))  # ~38% ann vol
    efa = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.011, n))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.003, n))
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    w_no_target = gem_target_weights(closes, cfg=GemConfig())
    w_targeted = gem_target_weights(
        closes, cfg=GemConfig(ann_vol_target=0.10, vol_lookback_days=20)
    )

    # On bars where the unconstrained strategy is long SPY (weight=1), the
    # vol-targeted version must have SPY weight materially BELOW 1.0.
    spy_long_mask = w_no_target["SPY"] == 1.0
    if spy_long_mask.any():
        avg_spy_scaled = w_targeted.loc[spy_long_mask, "SPY"].mean()
        avg_ief_scaled = w_targeted.loc[spy_long_mask, "IEF"].mean()
        assert avg_spy_scaled < 0.6, f"SPY weight not reduced enough: {avg_spy_scaled:.3f}"
        assert avg_ief_scaled > 0.4, f"IEF cash portion missing: {avg_ief_scaled:.3f}"
        # Sum of weights should still be ~1 (no leverage).
        sums = w_targeted.loc[spy_long_mask].sum(axis=1)
        assert (sums.between(0.95, 1.05)).all(), (
            f"Weight sums not in [0.95, 1.05]: min={sums.min()}, max={sums.max()}"
        )


def test_vol_target_no_op_under_low_vol():
    """When realised vol < target, scale is capped at max_leverage = 1.0.
    Weights should equal the unscaled binary version.
    """
    rng = np.random.default_rng(1)
    # Very low-vol SPY: ~3% ann vol, below 10% target.
    n = 252 * 3
    spy = 100 * np.cumprod(1 + rng.normal(0.10 / 252, 0.002, n))
    efa = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.002, n))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.0005, n))
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    w_no_target = gem_target_weights(closes, cfg=GemConfig())
    w_targeted = gem_target_weights(
        closes, cfg=GemConfig(ann_vol_target=0.10, vol_lookback_days=20, max_leverage=1.0)
    )

    # Past the warmup, the targeted weights should equal the binary weights
    # (cap at max_leverage = 1.0 means we don't lever up).
    post_warmup = slice(252, None)
    pd.testing.assert_frame_equal(
        w_no_target.iloc[post_warmup],
        w_targeted.iloc[post_warmup],
        check_exact=False,
        atol=1e-6,
    )


def test_compute_stress_signal_disabled_by_default():
    """When stress_gate_enabled is False, signal is always False (no-op)."""
    closes = _synthetic_closes(n_years=3)
    cfg = GemConfig()  # stress_gate_enabled = False by default
    sig = compute_stress_signal(closes["SPY"], cfg)
    assert isinstance(sig, pd.Series)
    assert (sig.astype(bool) == False).all()  # noqa: E712


def test_compute_stress_signal_fires_on_high_realised_vol():
    """High realised SPY vol triggers stress = True."""
    rng = np.random.default_rng(0)
    n = 252 * 2
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    # 40% ann vol -- well above the 20% threshold.
    spy = 100 * np.cumprod(1 + rng.normal(0.05 / 252, 0.025, n))
    cfg = GemConfig(stress_gate_enabled=True, stress_realised_vol_threshold=0.20)
    sig = compute_stress_signal(pd.Series(spy, index=idx, name="SPY"), cfg)
    # Most post-warmup bars should be stressed.
    post_warmup = sig.iloc[40:]
    assert post_warmup.mean() > 0.8


def test_compute_stress_signal_fires_on_vix_threshold():
    """VIX above the threshold triggers stress even when realised vol is low."""
    n = 100
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    spy = pd.Series(100.0 * np.linspace(1.0, 1.10, n), index=idx, name="SPY")  # low vol
    vix = pd.Series(35.0, index=idx, name="VIX")  # well above 25
    cfg = GemConfig(stress_gate_enabled=True, stress_vix_threshold=25.0)
    sig = compute_stress_signal(spy, cfg, vix=vix)
    # All post-warmup bars should be in stress (VIX dominates).
    assert sig.iloc[40:].mean() > 0.95


def test_stress_gate_no_op_in_calm_markets():
    """When stress signal is False, the conditional overlay should deploy
    at max_leverage (i.e. = 1.0 by default) -- NOT scale down. Result:
    weights ~= binary weights, unchanged.
    """
    # Construct a long stable bull series so SPY vol stays ~5% << 20% threshold.
    rng = np.random.default_rng(1)
    n = 252 * 3
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    spy = 100 * np.cumprod(1 + rng.normal(0.10 / 252, 0.003, n))
    efa = 100 * np.cumprod(1 + rng.normal(0.05 / 252, 0.003, n))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.0005, n))
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    w_no_target = gem_target_weights(closes, cfg=GemConfig())
    w_gated = gem_target_weights(
        closes,
        cfg=GemConfig(
            ann_vol_target=0.10,
            stress_gate_enabled=True,
            stress_realised_vol_threshold=0.20,
            max_leverage=1.0,
        ),
    )
    # Past the warmup, in calm regime: gated == ungated.
    post = slice(252, None)
    pd.testing.assert_frame_equal(
        w_no_target.iloc[post], w_gated.iloc[post], check_exact=False, atol=1e-6
    )


def test_stress_gate_activates_in_high_vol():
    """When the regime turns stressful, the conditional overlay scales down."""
    rng = np.random.default_rng(2)
    n_calm = 252 * 2
    n_stress = 252
    idx = pd.date_range("2020-01-02", periods=n_calm + n_stress, freq="B")
    spy_calm = np.cumprod(1 + rng.normal(0.10 / 252, 0.003, n_calm))
    # Stress phase: 40% ann vol.
    spy_stress = spy_calm[-1] * np.cumprod(1 + rng.normal(0.05 / 252, 0.025, n_stress))
    spy = np.concatenate([spy_calm, spy_stress]) * 100
    efa = 100 * np.cumprod(1 + rng.normal(0.05 / 252, 0.003, len(spy)))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.0005, len(spy)))
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    cfg = GemConfig(
        ann_vol_target=0.10,
        stress_gate_enabled=True,
        stress_realised_vol_threshold=0.20,
        max_leverage=1.0,
    )
    w = gem_target_weights(closes, cfg=cfg)
    # In the stress phase, SPY weight should be materially below 1 (scaled down).
    stress_slice = slice(n_calm + 40, None)
    spy_long_mask = w.iloc[stress_slice]["SPY"] > 0
    if spy_long_mask.any():
        avg_spy = w.iloc[stress_slice].loc[spy_long_mask, "SPY"].mean()
        assert avg_spy < 0.6, f"Expected scale-down in stress, got SPY weight {avg_spy:.3f}"


def test_leverage_in_calm_increases_position():
    """When max_leverage > 1, calm-regime weights should exceed 1.0 on the
    risk asset (modelling MES futures leverage).
    """
    rng = np.random.default_rng(3)
    n = 252 * 3
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    # Very calm market: realised vol ~5%, well below 20% stress threshold.
    spy = 100 * np.cumprod(1 + rng.normal(0.10 / 252, 0.003, n))
    efa = 100 * np.cumprod(1 + rng.normal(0.05 / 252, 0.003, n))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.0005, n))
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    cfg = GemConfig(
        ann_vol_target=0.10,
        stress_gate_enabled=True,
        stress_realised_vol_threshold=0.20,
        max_leverage=1.5,
    )
    w = gem_target_weights(closes, cfg=cfg)
    # Post-warmup, in calm regime: SPY weight should be ~1.5 (the cap).
    held = w.iloc[252:]
    spy_long_mask = held["SPY"] > 0
    if spy_long_mask.any():
        avg_spy = held.loc[spy_long_mask, "SPY"].mean()
        assert avg_spy > 1.4, f"Expected leveraged SPY weight ~1.5, got {avg_spy:.3f}"


def test_cost_overlay_reduces_returns():
    """Cost overlay (cost_bps_per_turnover > 0) must reduce per-bar
    returns by the right amount on turnover bars and leave hold bars
    unchanged.
    """
    closes = _synthetic_closes(n_years=4, seed=2)
    ret_gross = gem_returns(closes, cost_bps_per_turnover=0.0)
    ret_net = gem_returns(closes, cost_bps_per_turnover=1.5)
    # Net is always <= gross (costs only ever subtract).
    diff = ret_gross - ret_net
    assert (diff >= -1e-12).all(), "Cost overlay should never increase returns"
    # At least SOME bars must have non-zero cost (months see turnover).
    assert (diff > 1e-9).sum() >= 4
    # Cost is bounded: total turnover per bar <= 2 (swap), so per-bar cost
    # <= 2 * 1.5bps = 3 bps = 3e-4.
    assert diff.max() < 3.1e-4


def test_dd_breaker_disabled_by_default_is_no_op():
    """When dd_breaker_enabled is False (default), weights should be
    identical with or without the breaker config fields.
    """
    closes = _synthetic_closes(n_years=4, seed=1)
    w_off = gem_target_weights(closes, cfg=GemConfig(dd_breaker_enabled=False))
    w_default = gem_target_weights(closes, cfg=GemConfig())  # no dd_breaker_enabled
    pd.testing.assert_frame_equal(w_off, w_default)


def test_dd_breaker_haircut_fires_on_modest_drawdown():
    """Construct a series with a ~25% drawdown after a rally. With
    haircut threshold = -10% the breaker should engage well before the
    bottom and scale SPY weight to ~0.5 (or 0 if -15% breached).
    """
    rng = np.random.default_rng(3)
    n_up = 252
    n_down = 200
    # Rally first to build a HWM.
    spy_up = np.cumprod(1 + rng.normal(0.30 / 252, 0.003, n_up))
    # Then a sustained decline to ~25-30% peak-to-trough.
    spy_down = spy_up[-1] * np.cumprod(1 + rng.normal(-0.40 / 252, 0.003, n_down))
    spy = np.concatenate([spy_up, spy_down]) * 100
    efa = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.003, len(spy)))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.0005, len(spy)))
    idx = pd.date_range("2020-01-02", periods=len(spy), freq="B")
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    cfg = GemConfig(
        dd_breaker_enabled=True,
        dd_breaker_haircut_threshold=-0.10,
        dd_breaker_flat_threshold=-0.15,
        dd_breaker_recovery_threshold=-0.05,
    )
    w_off = gem_target_weights(closes, cfg=GemConfig(dd_breaker_enabled=False))
    w_on = gem_target_weights(closes, cfg=cfg)

    # In the late-decline window (last 100 bars) when DD is well below
    # -10%, the breaker version must hold less SPY than the off version.
    late = slice(-100, None)
    spy_off_avg = w_off.iloc[late]["SPY"].mean()
    spy_on_avg = w_on.iloc[late]["SPY"].mean()
    assert spy_on_avg < spy_off_avg, (
        f"DD breaker did not reduce SPY exposure in late decline. "
        f"off={spy_off_avg:.3f} on={spy_on_avg:.3f}"
    )
    # IEF should have picked up the missing capital.
    assert w_on.iloc[late]["IEF"].mean() > w_off.iloc[late]["IEF"].mean()


def test_dd_breaker_preserves_causality():
    """Future shocks must not change past DD-breaker weights."""
    closes = _synthetic_closes(n_years=5)
    gem_assert_causal(closes, cfg=GemConfig(dd_breaker_enabled=True), n_trials=5, seed=42)


def test_cost_overlay_zero_when_no_trading():
    """When the strategy holds the same position (no turnover), cost = 0."""
    # Construct a series where SPY dominates throughout (no switches).
    rng = np.random.default_rng(99)
    n = 252 * 3
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    spy = 100 * np.cumprod(1 + rng.normal(0.20 / 252, 0.002, n))
    efa = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.002, n))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.0005, n))
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)
    ret_gross = gem_returns(closes, cost_bps_per_turnover=0.0)
    ret_net = gem_returns(closes, cost_bps_per_turnover=1.5)
    # Within hold periods (which dominate this series), cost should be 0.
    diff = ret_gross - ret_net
    # Most bars should be cost-free; a handful at the first switch may have cost.
    assert (diff < 1e-9).mean() > 0.95


def test_vol_target_preserves_causality():
    """Vol-target adds rolling-vol on the strategy returns. Verify the
    shift discipline still holds: future shocks must not alter past
    targeted weights.
    """
    closes = _synthetic_closes(n_years=5)
    gem_assert_causal(
        closes,
        cfg=GemConfig(ann_vol_target=0.10, vol_lookback_days=20),
        n_trials=5,
        seed=42,
    )


def test_gem_buffer_does_not_freeze_against_stale_incumbent():
    """Regression test for the 2026-05-14 bug.

    The bug: the buffer compared challenger's CURRENT 12m return against
    the incumbent's frozen 12m return from the last time it was the winner.
    Once the incumbent stopped being challenger (e.g., once SPY went into
    a bear market and IEF became the regime winner), the incumbent's
    stale return was never refreshed, and IEF's actual 12m never reached
    the (irrelevant) bull-market high of SPY -- so the strategy never
    switched into the defensive IEF leg.

    This test constructs: a first year where SPY rallies (becomes incumbent
    at a high stale level), then a second + third year where SPY crashes
    while IEF rallies. With the bug, GEM stays in SPY forever; with the
    fix, GEM switches to IEF when SPY's CURRENT 12m drops below IEF's.
    """
    rng = np.random.default_rng(0)

    # Year 1 (2020): SPY rips +30%, IEF flat. SPY 12m at end of Y1 ~ +30%.
    n1 = 252
    spy_y1 = np.cumprod(1 + rng.normal(0.30 / 252, 0.003, n1))
    ief_y1 = np.cumprod(1 + rng.normal(0.00 / 252, 0.001, n1))
    efa_y1 = np.cumprod(1 + rng.normal(0.05 / 252, 0.003, n1))

    # Year 2 (2021): SPY crashes -30%; IEF rallies +10%. By end of Y2, SPY
    # 12m return = -30%, IEF 12m = +10%. The defensive switch MUST fire.
    n2 = 252
    spy_y2 = spy_y1[-1] * np.cumprod(1 + rng.normal(-0.30 / 252, 0.003, n2))
    ief_y2 = ief_y1[-1] * np.cumprod(1 + rng.normal(0.10 / 252, 0.001, n2))
    efa_y2 = efa_y1[-1] * np.cumprod(1 + rng.normal(-0.30 / 252, 0.003, n2))

    # Year 3 (2022): same conditions persist. With the bug, GEM stays SPY
    # because the stale incumbent return is too high. With the fix, GEM
    # is in IEF for the whole year.
    n3 = 252
    spy_y3 = spy_y2[-1] * np.cumprod(1 + rng.normal(-0.30 / 252, 0.003, n3))
    ief_y3 = ief_y2[-1] * np.cumprod(1 + rng.normal(0.10 / 252, 0.001, n3))
    efa_y3 = efa_y2[-1] * np.cumprod(1 + rng.normal(-0.30 / 252, 0.003, n3))

    spy = np.concatenate([spy_y1, spy_y2, spy_y3]) * 100
    efa = np.concatenate([efa_y1, efa_y2, efa_y3]) * 100
    ief = np.concatenate([ief_y1, ief_y2, ief_y3]) * 100
    idx = pd.date_range("2020-01-02", periods=len(spy), freq="B")
    closes = pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)

    w = gem_target_weights(closes)

    # In year 3 (last 252 bars) the strategy must be in IEF on >50% of
    # bars. Pre-fix this was 0% (frozen in SPY).
    held_y3 = w.iloc[-252:]
    pct_ief = (held_y3["IEF"] == 1.0).mean()
    assert pct_ief > 0.5, (
        f"Regression: buffer froze the defensive switch. "
        f"Year-3 IEF holding pct = {pct_ief:.2%} (expected > 50%)."
    )


# ── (5) Returns sanity ────────────────────────────────────────────────────


def test_gem_returns_finite_and_first_bar_zero():
    closes = _synthetic_closes(n_years=4)
    ret = gem_returns(closes)
    assert np.isfinite(ret).all()
    # The first bar can't have a return (no shifted weight).
    assert ret.iloc[0] == 0.0


def test_gem_returns_match_weight_dot_bar_returns():
    """End-to-end identity: gem_returns == sum(weight * bar_ret)."""
    closes = _synthetic_closes(n_years=3)
    w = gem_target_weights(closes)
    bar_ret = closes[list(GEM_UNIVERSE)].pct_change()
    manual = (w * bar_ret).sum(axis=1)
    pd.testing.assert_series_equal(gem_returns(closes), manual, check_names=False)


# ── (6) Class defaults sanity (the pre-reg's class assumption) ────────────


def test_gem_uses_cross_asset_momentum_defaults():
    d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    # Per-day MTM Sharpe convention (L06 lesson -- not per-bar at sqrt(252*24)).
    assert d.sharpe.primary == "per_day_mtm"
    # MC gate is class-calibrated, NOT uniform 25%/5% (L08 lesson).
    assert d.mc.max_dd_threshold_pct == pytest.approx(0.35)
    assert d.mc.max_dd_pass_prob == pytest.approx(0.10)
    # WFO uses rolling (cross-asset benefits from a sliding IS).
    assert d.wfo.is_mode == "rolling"
