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
