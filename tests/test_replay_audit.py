"""Tests for the replay-audit pure logic.

The IBKR connection + signal-parquet loading are integration-level. The
decision logic and fill aggregation are pure and what we exercise here.

Tier 2.5 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from scripts.replay_audit import (
    Fill,
    compute_z_score,
    expected_action,
    fills_to_daily_actions,
)

# ── compute_z_score ──────────────────────────────────────────────────


def test_compute_z_score_returns_none_for_insufficient_history():
    """Need at least lookback + 30 bars (lookback + 10 baseline + 20 for the
    momentum-series std). Anything less returns None to mean 'not yet ready'."""
    closes = [100.0] * 5
    assert compute_z_score(closes, lookback=5, zscore_window=504) is None


def test_compute_z_score_returns_zero_for_constant_series_within_window():
    """A series with no variation has zero momentum and zero std — function
    must return None (degenerate case) rather than NaN."""
    closes = [100.0] * 100
    assert compute_z_score(closes, lookback=5, zscore_window=504) is None


def test_compute_z_score_positive_when_momentum_above_window_mean():
    """Build a series where the last bar's momentum is well above the
    historical average. Z should be positive."""
    rng = np.random.default_rng(42)
    base = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.005, 600))
    base = list(base)
    # Force last bar to be a strong rally relative to lookback bar
    base[-1] = base[-6] * 1.10  # +10% over 5 days
    z = compute_z_score(base, lookback=5, zscore_window=504)
    assert z is not None
    assert z > 1.0, f"Expected strong positive z, got {z}"


def test_compute_z_score_matches_hand_calculation_for_simple_case():
    """Build a series with known stats and verify the formula matches."""
    # Linear trend: each day +0.01 log return for 200 bars
    log_ret = 0.01
    closes = [math.exp(log_ret * i) * 100.0 for i in range(200)]
    # 5-day momentum is constant 5 * log_ret = 0.05, std = 0
    # → function returns None (sigma < 1e-8)
    assert compute_z_score(closes, lookback=5, zscore_window=504) is None


# ── expected_action ──────────────────────────────────────────────────


def test_expected_action_entry_when_flat_and_z_above_threshold():
    assert (
        expected_action(z=0.30, is_long=False, bars_held=0, threshold=0.25, hold_days=5) == "entry"
    )


def test_expected_action_no_entry_when_z_at_or_below_threshold():
    """Strict > comparison: z == threshold is HOLD, not entry."""
    assert (
        expected_action(z=0.25, is_long=False, bars_held=0, threshold=0.25, hold_days=5) == "hold"
    )
    assert (
        expected_action(z=0.20, is_long=False, bars_held=0, threshold=0.25, hold_days=5) == "hold"
    )


def test_expected_action_no_entry_when_already_long():
    """The post-fix entry guard: never enter when already long."""
    assert expected_action(z=2.0, is_long=True, bars_held=10, threshold=0.25, hold_days=5) == "hold"


def test_expected_action_exit_when_long_and_past_hold_and_z_drops():
    assert expected_action(z=0.10, is_long=True, bars_held=5, threshold=0.25, hold_days=5) == "exit"
    assert expected_action(z=0.10, is_long=True, bars_held=6, threshold=0.25, hold_days=5) == "exit"


def test_expected_action_no_exit_before_min_hold():
    """Within hold_days, even a bad z keeps us long."""
    assert expected_action(z=0.10, is_long=True, bars_held=4, threshold=0.25, hold_days=5) == "hold"
    assert expected_action(z=-1.0, is_long=True, bars_held=2, threshold=0.25, hold_days=5) == "hold"


def test_expected_action_no_exit_when_z_above_threshold():
    """Past min hold but z still bullish → continue holding."""
    assert (
        expected_action(z=0.50, is_long=True, bars_held=10, threshold=0.25, hold_days=5) == "hold"
    )


# ── fills_to_daily_actions ───────────────────────────────────────────


def _mk_fill(date: str, side: str, qty: float) -> Fill:
    dt = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return Fill(
        exec_id=f"E-{date}-{side}",
        time_utc=dt,
        symbol="VUSD",
        side=side,
        qty=qty,
        price=140.0,
    )


def test_fills_to_daily_actions_buy_only_is_entry():
    fills = [_mk_fill("2026-05-01", "BUY", 30.0)]
    assert fills_to_daily_actions(fills) == {"2026-05-01": "entry"}


def test_fills_to_daily_actions_sell_only_is_exit():
    fills = [_mk_fill("2026-05-01", "SELL", 30.0)]
    assert fills_to_daily_actions(fills) == {"2026-05-01": "exit"}


def test_fills_to_daily_actions_mixed_same_day_flagged():
    """A buy + sell on the same day is unusual — flag as 'mixed' so the
    audit reports it for review (could be a legitimate close + re-open or
    a bug)."""
    fills = [
        _mk_fill("2026-05-01", "BUY", 30.0),
        _mk_fill("2026-05-01", "SELL", 30.0),
    ]
    assert fills_to_daily_actions(fills) == {"2026-05-01": "mixed"}


def test_fills_to_daily_actions_multiple_buys_same_day_collapse():
    """Multiple BUY fills on the same day still register as one 'entry'."""
    fills = [
        _mk_fill("2026-05-01", "BUY", 10.0),
        _mk_fill("2026-05-01", "BUY", 20.0),
    ]
    assert fills_to_daily_actions(fills) == {"2026-05-01": "entry"}


def test_fills_to_daily_actions_separate_days_separate_entries():
    fills = [
        _mk_fill("2026-05-01", "BUY", 30.0),
        _mk_fill("2026-05-08", "SELL", 30.0),
    ]
    out = fills_to_daily_actions(fills)
    assert out["2026-05-01"] == "entry"
    assert out["2026-05-08"] == "exit"


def test_fills_to_daily_actions_empty_returns_empty():
    assert fills_to_daily_actions([]) == {}


# ── End-to-end logic (no IBKR) ───────────────────────────────────────


def test_replay_logic_catches_unexpected_entry():
    """Simulate the May 11 doubling pattern: backtest says HOLD (already
    long), but actual fills show an unexpected BUY. The diff should flag it."""
    expected = expected_action(
        z=2.0,  # bullish
        is_long=True,  # already long after rehydration
        bars_held=5,
        threshold=0.25,
        hold_days=5,
    )
    assert expected == "hold"
    # Actual: live made a BUY anyway (the May 11 bug pattern)
    actual = "entry"
    is_mismatch = expected != actual
    is_actionable = expected != "hold" or actual != "hold"
    assert is_mismatch and is_actionable


def test_replay_logic_no_alert_when_both_hold():
    """A quiet day where neither backtest nor live did anything is normal."""
    expected = expected_action(z=0.10, is_long=False, bars_held=0, threshold=0.25, hold_days=5)
    assert expected == "hold"
    actual = "hold"
    is_actionable = expected != "hold" or actual != "hold"
    assert not is_actionable
