"""tests/test_etf_trend_strategy.py — Unit tests for ETF Trend Strategy helpers.

Tests pure math functions that don't require a NautilusTrader test harness.
Covers: ADX computation, ATR computation, regime detection logic,
decel composite calculation, next-trading-day helper.

Instance methods are tested via a plain mock class that delegates to
ETFTrendStrategy's unbound implementations — avoids the NautilusTrader
Strategy base-class __new__ restriction.
"""

from datetime import date

import numpy as np
import pandas as pd

from titan.strategies.etf_trend.strategy import ETFTrendStrategy

# ── Duck-typed mock ───────────────────────────────────────────────────────────


class _MockStrat:
    """Plain object that exposes ETFTrendStrategy's instance helpers.

    Attributes must be set before calling any method that reads them.
    """

    ma_type: str = "EMA"
    fast_ma: int = 50
    slow_ma: int = 200
    decel_signals: list = []
    history: list = []

    # Delegate to ETFTrendStrategy's unbound implementations
    def _closes(self):
        return ETFTrendStrategy._closes(self)  # type: ignore[arg-type]

    def _ma(self, s: pd.Series, period: int) -> pd.Series:
        return ETFTrendStrategy._ma(self, s, period)  # type: ignore[arg-type]

    def _compute_regime(self) -> bool:
        return ETFTrendStrategy._compute_regime(self)  # type: ignore[arg-type]

    def _compute_decel_composite(self) -> float:
        return ETFTrendStrategy._compute_decel_composite(self)  # type: ignore[arg-type]

    # Delegate static helpers called inside _compute_decel_composite
    _compute_adx_last = staticmethod(ETFTrendStrategy._compute_adx_last)


def _make_mock(
    prices: list[float],
    ma_type: str = "EMA",
    fast: int = 50,
    slow: int = 200,
    signals: list | None = None,
) -> _MockStrat:
    m = _MockStrat()
    m.ma_type = ma_type
    m.fast_ma = fast
    m.slow_ma = slow
    m.decel_signals = signals if signals is not None else []
    m.history = [{"close": p, "high": p + 0.5, "low": p - 0.5} for p in prices]
    return m


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _trending(n: int = 300, drift: float = 0.001) -> list[float]:
    rng = np.random.default_rng(42)
    returns = rng.normal(drift, 0.012, n)
    return (100.0 * np.cumprod(1 + returns)).tolist()


# ── Static method tests: ADX ──────────────────────────────────────────────────


class TestADX:
    def test_returns_float(self):
        p = np.array(_trending(100))
        assert isinstance(ETFTrendStrategy._compute_adx_last(p + 0.5, p - 0.5, p, 14), float)

    def test_range(self):
        p = np.array(_trending(200))
        result = ETFTrendStrategy._compute_adx_last(p + 0.5, p - 0.5, p, 14)
        assert 0.0 <= result <= 100.0

    def test_trending_higher_than_flat(self):
        n = 250
        trend = np.linspace(100, 200, n)
        flat = np.full(n, 150.0) + np.random.default_rng(0).normal(0, 0.1, n)
        adx_t = ETFTrendStrategy._compute_adx_last(trend + 0.5, trend - 0.5, trend, 14)
        adx_f = ETFTrendStrategy._compute_adx_last(flat + 0.5, flat - 0.5, flat, 14)
        assert adx_t > adx_f

    def test_insufficient_data_returns_default(self):
        p = np.array([100.0, 101.0, 102.0])
        assert ETFTrendStrategy._compute_adx_last(p + 0.5, p - 0.5, p, 14) == 20.0


# ── Static method tests: ATR ──────────────────────────────────────────────────


class TestATR:
    def test_returns_positive_float(self):
        p = np.array(_trending(50))
        result = ETFTrendStrategy._compute_atr_last(p + 1, p - 1, p, 14)
        assert isinstance(result, float) and result > 0

    def test_insufficient_data_returns_zero(self):
        p = np.array([100.0, 101.0])
        assert ETFTrendStrategy._compute_atr_last(p + 1, p - 1, p, 14) == 0.0

    def test_wider_range_higher_atr(self):
        p = np.array(_trending(100))
        narrow = ETFTrendStrategy._compute_atr_last(p + 0.5, p - 0.5, p, 14)
        wide = ETFTrendStrategy._compute_atr_last(p + 5.0, p - 5.0, p, 14)
        assert wide > narrow


# ── Static method tests: next trading day ────────────────────────────────────


class TestNextTradingDay:
    def test_friday_to_monday(self):
        assert ETFTrendStrategy._next_trading_day(date(2024, 3, 1)) == date(2024, 3, 4)

    def test_monday_to_tuesday(self):
        assert ETFTrendStrategy._next_trading_day(date(2024, 3, 4)) == date(2024, 3, 5)

    def test_saturday_to_monday(self):
        assert ETFTrendStrategy._next_trading_day(date(2024, 3, 2)) == date(2024, 3, 4)

    def test_sunday_to_monday(self):
        assert ETFTrendStrategy._next_trading_day(date(2024, 3, 3)) == date(2024, 3, 4)


# ── Instance method tests: MA ─────────────────────────────────────────────────


class TestMA:
    def test_sma_length(self):
        m = _make_mock([], ma_type="SMA")
        s = pd.Series(np.arange(1, 101, dtype=float))
        assert len(m._ma(s, 20)) == 100

    def test_ema_length(self):
        m = _make_mock([], ma_type="EMA")
        s = pd.Series(np.arange(1, 101, dtype=float))
        assert len(m._ma(s, 20)) == 100

    def test_sma_value(self):
        m = _make_mock([], ma_type="SMA")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(m._ma(s, 3).iloc[-1] - 4.0) < 1e-6

    def test_ema_reacts_faster_than_sma(self):
        # Flat series with a spike at the end: EMA pulls higher than SMA
        s = pd.Series([100.0] * 49 + [200.0])
        ema = float(_make_mock([], ma_type="EMA")._ma(s, 20).iloc[-1])
        sma = float(_make_mock([], ma_type="SMA")._ma(s, 20).iloc[-1])
        assert ema > sma


# ── Instance method tests: regime ────────────────────────────────────────────


class TestRegime:
    def test_uptrend_regime_true(self):
        prices = list(np.linspace(50, 300, 300))
        m = _make_mock(prices)
        assert m._compute_regime() is True

    def test_downtrend_regime_false(self):
        prices = list(np.linspace(300, 50, 300))
        m = _make_mock(prices)
        assert m._compute_regime() is False

    def test_insufficient_history_false(self):
        prices = list(np.linspace(100, 200, 100))
        m = _make_mock(prices, fast=50, slow=200)
        assert m._compute_regime() is False


# ── Instance method tests: decel composite ───────────────────────────────────


class TestDecelComposite:
    def test_no_signals_returns_one(self):
        m = _make_mock(_trending(300), signals=[])
        assert m._compute_decel_composite() == 1.0

    def test_in_range(self):
        prices = _trending(300)
        for sigs in [["d_pct"], ["rv_20"], ["adx_14"], ["d_pct", "rv_20", "adx_14"]]:
            m = _make_mock(prices, signals=sigs)
            val = m._compute_decel_composite()
            assert -1.0 <= val <= 1.0, f"Out of [-1,1] for {sigs}: {val}"

    def test_strong_uptrend_d_pct_positive(self):
        prices = list(np.linspace(100, 400, 300))
        m = _make_mock(prices, signals=["d_pct"])
        assert m._compute_decel_composite() > 0
