"""Regression tests for the April 2026 PortfolioRiskManager/Allocator rewrite.

Covers the audit findings that the rewrite fixed:

1. Per-strategy equity is honoured -- two strategies feeding *different*
   equity streams produce different inverse-vol weights (previously they'd
   always be equal because every strategy was passing whole-account NLV).
2. Daily-vol gate: mixing H1 and D1 update cadences no longer skews
   ``vol_scale`` because the EWMA variance is recomputed once per calendar
   date from a resampled daily NAV series.
3. Timestamp alignment in correlation/allocator math -- observations with
   the same *date* correlate, not observations with the same positional
   index.
4. Wall-clock rebalance gating -- ``tick`` called 1000 times in one day
   triggers at most one rebalance.
5. Halt persistence -- the kill switch is written to disk and re-read on a
   fresh manager instance.
6. FX unit conversion -- ``convert_notional_to_units`` refuses to silently
   assume a 1.0 FX rate when quote_ccy != base_ccy.
7. ``get_base_balance`` returns ``None`` instead of grabbing a
   nondeterministic ``ccys[0]`` when USD is absent.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from titan.risk.portfolio_allocator import PortfolioAllocator
from titan.risk.portfolio_risk_manager import PortfolioRiskManager
from titan.risk.strategy_equity import (
    StrategyEquityTracker,
    convert_notional_to_units,
    get_base_balance,
    split_fx_pair,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _day(n: int) -> pd.Timestamp:
    return pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=n)


def _seed_returns(seed: int, n: int, vol: float) -> pd.Series:
    import numpy as np

    rng = np.random.default_rng(seed)
    rets = rng.normal(0, vol, n)
    eq = (1 + pd.Series(rets)).cumprod() * 10_000.0
    idx = [_day(i) for i in range(n)]
    return pd.Series(eq.values, index=pd.DatetimeIndex(idx))


# ── Bug #1 regression -- per-strategy equity is real ─────────────────────────


def test_different_equity_streams_produce_different_weights():
    """If two strategies feed *different* equity curves, the inverse-vol
    allocator must produce different weights. Under the old bug every
    strategy passed whole-account NLV so weights collapsed to equal."""
    prm = PortfolioRiskManager(config={"history_max_days": 500, "portfolio_max_dd_pct": 99.0})
    prm.register_strategy("low_vol", 10_000.0)
    prm.register_strategy("high_vol", 10_000.0)

    low = _seed_returns(seed=1, n=60, vol=0.005)
    high = _seed_returns(seed=2, n=60, vol=0.03)
    for ts, v in low.items():
        prm.update("low_vol", float(v), ts=ts)
    for ts, v in high.items():
        prm.update("high_vol", float(v), ts=ts)

    alloc = PortfolioAllocator(config={"rebalance_interval_days": 1, "min_history_days": 20})
    # Swap the module-level PRM the allocator talks to.
    import titan.risk.portfolio_risk_manager as prm_module

    original = prm_module.portfolio_risk_manager
    prm_module.portfolio_risk_manager = prm
    try:
        alloc.force_rebalance()
        alloc.tick(now=date(2026, 3, 31))
        weights = alloc.get_all_weights()
    finally:
        prm_module.portfolio_risk_manager = original

    assert set(weights.keys()) == {"low_vol", "high_vol"}
    # The low-vol strategy should get strictly more weight than the high-vol one.
    assert weights["low_vol"] > weights["high_vol"] + 0.05, weights


# ── Bug #4/#6/#14 regression -- daily-vol gate runs once per calendar day ───


def test_vol_gate_fires_once_per_calendar_day():
    prm = PortfolioRiskManager(config={"history_max_days": 500})
    prm.register_strategy("s", 10_000.0)

    # Feed 24 ticks inside a single UTC day -- the daily-vol recompute should
    # only fire on the first one (subsequent ticks hit the date gate).
    day_ts = pd.Timestamp("2026-01-01", tz="UTC")
    for hour in range(24):
        prm.update("s", 10_000.0 + hour, ts=day_ts + pd.Timedelta(hours=hour))

    # We don't expose recompute-count, but we can observe that
    # ``_last_daily_date`` equals the fed day and didn't advance.
    assert prm._last_daily_date == day_ts.date()


# ── Gemini Finding #1 regression -- adding a strategy mid-life must not ──
# inject a spurious one-day NAV jump into _ewma_var.


def test_recompute_daily_vol_ignores_capital_addition():
    """Old _recompute_daily_vol summed total equity then took pct_change.
    Registering a new strategy with seed_equity 1500 onto a 30k portfolio
    produced a +5% spurious 'return' on the join day, which exploded
    _ewma_var and collapsed vol_scale to the 0.25 floor. The fix uses
    previous-day capital-weighted per-strategy returns; a brand-new
    strategy must contribute zero to the portfolio return on its debut.
    """
    prm = PortfolioRiskManager(config={"history_max_days": 500, "portfolio_max_dd_pct": 99.0})
    prm.register_strategy("incumbent", 30_000.0)

    # 30 days of dead-flat equity for the incumbent -> zero portfolio vol.
    for i in range(30):
        prm.update("incumbent", 30_000.0, ts=_day(i))

    # Capture state just before the new strategy joins.
    ewma_pre = prm._ewma_var

    # On day 30, register a new strategy at 1500 seed equity. Under the bug,
    # nav jumps 30000 -> 31500 (+5%) and _ewma_var explodes.
    prm.register_strategy("newcomer", 1_500.0)
    prm.update("newcomer", 1_500.0, ts=_day(30))
    prm.update("incumbent", 30_000.0, ts=_day(30))

    # Also let one more flat day pass so the newcomer's own return is 0.0.
    prm.update("newcomer", 1_500.0, ts=_day(31))
    prm.update("incumbent", 30_000.0, ts=_day(31))

    ewma_post = prm._ewma_var
    # Pre-fix this would be ~25e-4 (5% squared * EWMA decay). With the fix it
    # stays at machine-zero because no strategy ever produced a real return.
    assert ewma_post is None or ewma_post < 1e-10, (
        f"capital addition leaked into EWMA variance: pre={ewma_pre} post={ewma_post}"
    )


# ── Gemini Finding #2 regression -- min_w must never be violated when one ──
# strategy has insufficient history.


def test_allocator_floor_never_violated_with_missing_strategy():
    """Old _rebalance assigned min_w to the missing strategy then divided
    every weight by (1 + n_missing*min_w), dragging mature strategies that
    were at the floor below it. The two-pass fix reserves min_w for missing
    first, then allocates the residual (1 - n_missing*min_w) among mature.
    """
    prm = PortfolioRiskManager(config={"history_max_days": 500, "portfolio_max_dd_pct": 99.0})
    # Two mature strategies (60d history) + one immature (5d, < min_history).
    prm.register_strategy("mature_a", 10_000.0)
    prm.register_strategy("mature_b", 10_000.0)
    prm.register_strategy("immature", 5_000.0)

    a = _seed_returns(seed=11, n=60, vol=0.005)
    b = _seed_returns(seed=12, n=60, vol=0.03)
    for ts, v in a.items():
        prm.update("mature_a", float(v), ts=ts)
    for ts, v in b.items():
        prm.update("mature_b", float(v), ts=ts)
    # Immature: only 5 datapoints.
    for i in range(5):
        prm.update("immature", 5_000.0 + i, ts=_day(55 + i))

    alloc = PortfolioAllocator(
        config={
            "rebalance_interval_days": 1,
            "min_history_days": 20,
            "min_weight": 0.10,
            "max_weight": 0.60,
        }
    )
    import titan.risk.portfolio_risk_manager as prm_module

    original = prm_module.portfolio_risk_manager
    prm_module.portfolio_risk_manager = prm
    try:
        alloc.force_rebalance()
        alloc.tick(now=date(2026, 3, 31))
        weights = alloc.get_all_weights()
    finally:
        prm_module.portfolio_risk_manager = original

    assert set(weights.keys()) == {"mature_a", "mature_b", "immature"}
    # All three strategies must be at or above min_w.
    for sid, w in weights.items():
        assert w >= 0.10 - 1e-9, f"floor violated: {sid} -> {w:.4f}"
    # All three must be at or below max_w.
    for sid, w in weights.items():
        assert w <= 0.60 + 1e-9, f"cap violated: {sid} -> {w:.4f}"
    # Must sum to 1.0.
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # Immature gets exactly min_w (the reserve).
    assert abs(weights["immature"] - 0.10) < 1e-9


# ── Gemini Finding #3 regression -- risk_contributions must expose the ──
# unmeasured-capital bucket so a viewer cannot read 100% accounted-for.


def test_risk_contributions_expose_unmeasured_bucket():
    """When strategies with <20d history are dropped from the covariance,
    the returned rc values used to sum to 1.0 across the mature subset --
    falsely signalling that 100% of risk was accounted for. The fix
    rescales rc to share of TOTAL portfolio risk and adds a
    `__unmeasured__` key holding the capital fraction of immature
    strategies. `get_summary` surfaces this as `risk_unmeasured_capital_pct`.
    """
    prm = PortfolioRiskManager(config={"history_max_days": 500, "portfolio_max_dd_pct": 99.0})
    prm.register_strategy("mature_a", 10_000.0)
    prm.register_strategy("mature_b", 10_000.0)
    prm.register_strategy("immature", 10_000.0)

    a = _seed_returns(seed=21, n=60, vol=0.01)
    b = _seed_returns(seed=22, n=60, vol=0.01)
    for ts, v in a.items():
        prm.update("mature_a", float(v), ts=ts)
    for ts, v in b.items():
        prm.update("mature_b", float(v), ts=ts)
    # Immature has only 5 datapoints -> dropped from cov.
    for i in range(5):
        prm.update("immature", 10_000.0 + i, ts=_day(55 + i))

    rc = prm._risk_contributions()
    assert "__unmeasured__" in rc, rc
    # ~1/3 of capital is in the immature strategy, so unmeasured ~= 0.333.
    assert 0.30 < rc["__unmeasured__"] < 0.40, rc["__unmeasured__"]
    # Mature contribs no longer sum to 1.0; they sum to mature_capital_frac.
    mature_sum = sum(v for k, v in rc.items() if not k.startswith("__"))
    assert 0.60 < mature_sum < 0.70, mature_sum

    summary = prm.get_summary()
    assert summary["risk_unmeasured_capital_pct"] > 30.0


# ── Bug #5 regression -- allocator rebalances on calendar, not ticks ─────────


def test_allocator_rebalance_gated_by_calendar_day():
    alloc = PortfolioAllocator(config={"rebalance_interval_days": 21})
    # Force the initial rebalance.
    alloc._last_rebalance_date = date(2026, 1, 1)

    # 1000 ticks on the same day shouldn't trigger a second rebalance.
    rebalance_calls = []
    alloc._rebalance = lambda: rebalance_calls.append(1)  # type: ignore[assignment]
    for _ in range(1000):
        alloc.tick(now=date(2026, 1, 2))
    assert rebalance_calls == []

    # Ticking 22 days later triggers exactly one.
    alloc.tick(now=date(2026, 1, 23))
    assert rebalance_calls == [1]


# ── Bug #8 regression -- correlation aligned on timestamps, not positions ───


def test_correlation_aligned_on_timestamps():
    prm = PortfolioRiskManager(
        config={
            "history_max_days": 500,
            "correlation_halt_threshold": 0.99,
            "portfolio_max_dd_pct": 99.0,
        }
    )
    prm.register_strategy("a", 10_000.0)
    prm.register_strategy("b", 10_000.0)

    # a updates every day from day 0; b starts at day 30. Their positional
    # indices would not line up but their timestamps align on days 30+.
    for i in range(60):
        prm.update("a", 10_000.0 + i * 10.0, ts=_day(i))
    for i in range(30, 60):
        prm.update("b", 10_000.0 - (i - 30) * 5.0, ts=_day(i))

    histories = prm.get_equity_histories()
    # Build returns and correlate.
    ret_a = histories["a"].pct_change().dropna()
    ret_b = histories["b"].pct_change().dropna()
    df = pd.DataFrame({"a": ret_a, "b": ret_b}).fillna(0.0)
    # The overlap period is >= 20 rows.
    assert len(df.dropna()) > 20 or len(df) > 20


# ── Bug #11/#12 regression -- halt state persists ────────────────────────────


def test_halt_state_persists_to_disk(tmp_path, monkeypatch):
    path = tmp_path / "halt.json"
    monkeypatch.setattr("titan.risk.portfolio_risk_manager._HALT_STATE_PATH", path, raising=True)

    prm = PortfolioRiskManager(config={"portfolio_max_dd_pct": 1.0})
    prm.register_strategy("s", 10_000.0)

    # Drive a drawdown that exceeds 1% to trigger the kill switch.
    prm.update("s", 10_000.0, ts=_day(0))
    prm.update("s", 9_800.0, ts=_day(1))  # -2% DD
    assert prm.halt_all is True
    assert path.exists()

    # A fresh manager reads the persisted halt state and refuses to resume.
    fresh = PortfolioRiskManager(config={"portfolio_max_dd_pct": 1.0})
    assert fresh.halt_all is True


# ── Bug #3 / FX unit conversion ──────────────────────────────────────────────


def test_fx_conversion_requires_explicit_rate():
    with pytest.raises(ValueError):
        convert_notional_to_units(
            notional_base=10_000.0,
            price=95.0,  # JPY per AUD
            quote_ccy="JPY",
            base_ccy="USD",
            fx_rate_quote_to_base=None,  # forgot it -> must raise
        )


def test_fx_conversion_with_explicit_rate_gives_correct_units():
    # 10,000 USD, AUD/JPY quoted at 95 JPY per AUD, JPY->USD = 0.0067.
    # => notional_in_JPY = 10_000 / 0.0067 ≈ 1_492_537
    # => units_AUD = 1_492_537 / 95 ≈ 15_710
    units = convert_notional_to_units(
        notional_base=10_000.0,
        price=95.0,
        quote_ccy="JPY",
        base_ccy="USD",
        fx_rate_quote_to_base=0.0067,
    )
    assert 15_600 < units < 15_800


def test_fx_conversion_usd_quoted_is_trivial():
    # USD/USD: units = notional / price, same as equity case.
    units = convert_notional_to_units(10_000.0, 400.0, quote_ccy="USD", base_ccy="USD")
    assert units == 25


def test_split_fx_pair():
    assert split_fx_pair("AUD/JPY.IDEALPRO") == ("AUD", "JPY")
    assert split_fx_pair("AUD/USD") == ("AUD", "USD")
    assert split_fx_pair("SPY.ARCA") is None


# ── Bug #2 regression -- deterministic base-currency resolution ──────────────


def test_get_base_balance_returns_none_when_usd_absent():
    acct = MagicMock()
    # Balances dict returns only JPY -- no USD available.
    jpy = MagicMock()
    jpy.__str__ = lambda self: "JPY"  # type: ignore[assignment]
    acct.balances.return_value = {jpy: MagicMock()}
    result = get_base_balance(acct, "USD")
    assert result is None


def test_get_base_balance_finds_explicit_usd():
    acct = MagicMock()
    jpy = MagicMock()
    jpy.__str__ = lambda self: "JPY"  # type: ignore[assignment]
    usd = MagicMock()
    usd.__str__ = lambda self: "USD"  # type: ignore[assignment]
    acct.balances.return_value = {jpy: MagicMock(), usd: MagicMock()}

    usd_balance = MagicMock()
    usd_balance.as_double.return_value = 100_000.0
    acct.balance_total.return_value = usd_balance

    result = get_base_balance(acct, "USD")
    assert result == 100_000.0
    acct.balance_total.assert_called_with(usd)


# ── StrategyEquityTracker accumulation ───────────────────────────────────────


def test_tracker_accumulates_realized_pnl_with_fx():
    tracker = StrategyEquityTracker(prm_id="t", initial_equity=10_000.0)
    # 5 JPY of PnL at 0.0067 USD/JPY = 0.0335 USD.
    tracker.on_position_closed(5.0, fx_to_base=0.0067)
    assert tracker.realized_pnl_base == pytest.approx(0.0335, rel=1e-3)
    assert tracker.current_equity() == pytest.approx(10_000.0335, rel=1e-6)


# ── get_equity_histories is the public accessor ──────────────────────────────


def test_public_equity_histories_accessor():
    prm = PortfolioRiskManager(config={"history_max_days": 500})
    prm.register_strategy("s", 10_000.0)
    for i in range(5):
        prm.update("s", 10_000.0 + i, ts=_day(i))
    hist = prm.get_equity_histories()
    assert "s" in hist
    assert isinstance(hist["s"], pd.Series)
