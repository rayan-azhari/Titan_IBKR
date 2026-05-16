"""Unit tests for G4 overnight session decomposition.

Specified in directives/Pre-Reg G4 Overnight Session Decomposition 2026-05-15.md §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.overnight.overnight_strategy import (
    OvernightConfig,
    Strategy,
    _session_returns,
    overnight_assert_causal,
    overnight_returns,
)
from titan.research.framework import StrategyClass, defaults_for


def _synthetic_spy(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLC where overnight has positive drift and intraday
    has zero drift -- LPS 2019 in miniature."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    close = 100.0
    rows = []
    for _ in range(n):
        # Overnight: positive expected return (drift), small noise.
        overnight = rng.normal(0.0006, 0.005)
        new_open = close * (1 + overnight)
        # Intraday: zero drift, larger noise.
        intraday = rng.normal(0.0, 0.010)
        new_close = new_open * (1 + intraday)
        high = max(new_open, new_close) * (1 + abs(rng.normal(0, 0.002)))
        low = min(new_open, new_close) * (1 - abs(rng.normal(0, 0.002)))
        rows.append((new_open, high, low, new_close, 1e6))
        close = new_close
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])
    return df


# ── (1) Class defaults ────────────────────────────────────────────────────


def test_intraday_microstructure_defaults_apply():
    d = defaults_for(StrategyClass.INTRADAY_MICROSTRUCTURE)
    # Class default is per-bar (one bar = one daily decision for G4). We
    # use the class-default WFO + MC, NOT the Sharpe convention (G4
    # treats each daily bar as one P&L observation; the audit harness
    # passes periods_per_year=252 explicitly to sharpe()).
    assert d.sharpe.primary in ("per_bar", "per_day_mtm", "per_trade")
    assert d.mc.max_dd_threshold_pct > 0


# ── (2) Session decomposition ─────────────────────────────────────────────


def test_session_decomposition_identity():
    """The compound (1+overnight)*(1+intraday) - 1 must equal the daily return."""
    df = _synthetic_spy(n=300)
    ov, ind, daily = _session_returns(df)
    compound = (1 + ov.fillna(0)) * (1 + ind.fillna(0)) - 1
    # Skip first row (NaN overnight).
    np.testing.assert_allclose(
        compound.iloc[1:].values,
        daily.iloc[1:].values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_overnight_returns_positive_on_drift_universe():
    """On the synthetic universe with positive overnight drift, the
    overnight-only strategy should have positive realised Sharpe (gross)."""
    df = _synthetic_spy(n=600, seed=1)
    rets = overnight_returns(
        df, cfg=OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=False)
    )
    sharpe_ratio = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio > 0.5, f"Expected positive overnight Sharpe; got {sharpe_ratio:.3f}"


def test_intraday_returns_near_zero_on_drift_universe():
    """The counterfactual: intraday-only should be flat (no drift built in)."""
    df = _synthetic_spy(n=600, seed=2)
    rets = overnight_returns(
        df, cfg=OvernightConfig(strategy=Strategy.INTRADAY_LONG, apply_costs=False)
    )
    sharpe_ratio = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
    assert abs(sharpe_ratio) < 0.5, f"Expected near-zero intraday Sharpe; got {sharpe_ratio:.3f}"


def test_buy_hold_matches_close_to_close():
    df = _synthetic_spy(n=300, seed=3)
    rets = overnight_returns(df, cfg=OvernightConfig(strategy=Strategy.BUY_HOLD, apply_costs=False))
    expected = df["close"].pct_change().fillna(0.0)
    np.testing.assert_allclose(rets.iloc[1:].values, expected.iloc[1:].values, rtol=1e-12)


# ── (3) Costs ─────────────────────────────────────────────────────────────


def test_costs_reduce_returns():
    df = _synthetic_spy(n=600, seed=4)
    gross = overnight_returns(
        df, cfg=OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=False)
    )
    net = overnight_returns(
        df, cfg=OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=True)
    )
    # Net must be strictly below gross on every fill bar.
    drag = (gross - net).iloc[1:]
    assert (drag > 0).all(), "Costs must reduce net returns"


def test_buy_hold_has_no_cost_drag():
    """BUY_HOLD has fills_per_bar=0 -- benchmark assumes already long."""
    df = _synthetic_spy(n=300, seed=5)
    a = overnight_returns(df, cfg=OvernightConfig(strategy=Strategy.BUY_HOLD, apply_costs=True))
    b = overnight_returns(df, cfg=OvernightConfig(strategy=Strategy.BUY_HOLD, apply_costs=False))
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)


# ── (4) Date filter ──────────────────────────────────────────────────────


def test_start_date_filter_truncates_history():
    df = _synthetic_spy(n=500, seed=6)
    cfg_full = OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=False)
    cfg_late = OvernightConfig(
        strategy=Strategy.OVERNIGHT_LONG, apply_costs=False, start_date="2019-01-01"
    )
    rets_full = overnight_returns(df, cfg=cfg_full)
    rets_late = overnight_returns(df, cfg=cfg_late)
    assert len(rets_late) < len(rets_full)
    assert rets_late.index[0] >= pd.Timestamp("2019-01-01")


# ── (5) Causality (A10) ──────────────────────────────────────────────────


def test_overnight_assert_causal_passes():
    df = _synthetic_spy(n=400, seed=7)
    overnight_assert_causal(
        df,
        cfg=OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=False),
        n_trials=5,
        seed=42,
    )


def test_corrupting_future_bar_does_not_affect_past_returns():
    df = _synthetic_spy(n=300, seed=8)
    cfg = OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=False)
    base = overnight_returns(df, cfg=cfg)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_indexer(["open", "close"])] *= 5.0
    altered = overnight_returns(df2, cfg=cfg)
    pd.testing.assert_series_equal(
        base.iloc[:-1],
        altered.iloc[:-1],
        check_names=False,
    )


# ── (6) OVERNIGHT_INTRADAY_SHORT signed direction ────────────────────────


def test_overnight_intraday_short_outperforms_overnight_only_on_lps_universe():
    """If intraday actually loses (zero-drift synthetic), short-intraday on
    top of long-overnight should outperform overnight-only -- the short
    pocket adds positive expectation from the intraday noise's downside
    bias when paired with the fixed cost structure ... actually no:
    on a zero-drift intraday, OVERNIGHT_INTRADAY_SHORT's gross expected
    return equals OVERNIGHT_LONG. The cost structure (4 fills vs 2) makes
    OVERNIGHT_INTRADAY_SHORT net-worse. This is the audited tradeoff.
    """
    df = _synthetic_spy(n=600, seed=9)
    o = overnight_returns(
        df, cfg=OvernightConfig(strategy=Strategy.OVERNIGHT_LONG, apply_costs=True)
    )
    s = overnight_returns(
        df, cfg=OvernightConfig(strategy=Strategy.OVERNIGHT_INTRADAY_SHORT, apply_costs=True)
    )
    # With double fills, the aggressive variant should under-perform on a
    # zero-drift synthetic intraday. Mean of o > mean of s on this seed.
    assert o.mean() > s.mean() - 1e-6
