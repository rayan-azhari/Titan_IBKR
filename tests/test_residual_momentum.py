"""Unit tests for A1 residual momentum.

Specified in directives/Pre-Reg A1 Residual Momentum 2026-05-15.md §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.residual_momentum.residual_strategy import (
    ResidualConfig,
    _compute_residuals_for_window,
    _month_end_mask,
    build_portfolio_weights,
    compute_residual_signal,
    residual_assert_causal,
    residual_returns,
)
from titan.research.framework import StrategyClass, defaults_for


def _synthetic_universe(
    n_stocks: int = 20, n_bars: int = 252 * 6, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic FF3 + stock universe.

    Each stock i has true factor loadings (beta_mkt_i, beta_smb_i, beta_hml_i)
    + idiosyncratic drift (alpha_i). Half the stocks have positive idio
    drift, half have negative -- so a residual-momentum strategy SHOULD
    be able to earn on this universe.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    # Factors.
    mkt = rng.normal(0.0003, 0.012, n_bars)
    smb = rng.normal(0.0001, 0.006, n_bars)
    hml = rng.normal(0.0001, 0.006, n_bars)
    rf = np.full(n_bars, 0.00005)  # ~1.25% annual
    ff3 = pd.DataFrame({"mkt_rf": mkt, "smb": smb, "hml": hml, "rf": rf}, index=idx)
    # Stocks with idiosyncratic drift split half/half.
    stocks: dict[str, np.ndarray] = {}
    for i in range(n_stocks):
        beta_mkt = rng.uniform(0.7, 1.3)
        beta_smb = rng.uniform(-0.5, 0.5)
        beta_hml = rng.uniform(-0.5, 0.5)
        alpha = 0.0005 if i % 2 == 0 else -0.0005
        # Daily log returns:
        idio = rng.normal(alpha, 0.008, n_bars)
        ret = beta_mkt * mkt + beta_smb * smb + beta_hml * hml + idio + rf
        # Build a price series.
        price = 100 * np.cumprod(1 + ret)
        stocks[f"S{i:03d}"] = price
    stocks_close = pd.DataFrame(stocks, index=idx)
    return stocks_close, ff3


# ── (1) Class defaults ────────────────────────────────────────────────────


def test_uses_cross_asset_momentum_defaults():
    d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    assert d.sharpe.primary == "per_day_mtm"
    assert d.mc.max_dd_threshold_pct > 0


# ── (2) Internals ─────────────────────────────────────────────────────────


def test_month_end_mask_marks_last_business_day():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    me = _month_end_mask(idx)
    assert me.tolist() == [False, True, False, True, True]


def test_compute_residuals_for_window_recovers_known_betas():
    """If we generate stock returns as 1*mkt + 0.3*smb + epsilon, the OLS
    residuals should be approximately epsilon."""
    rng = np.random.default_rng(0)
    T = 252
    factors = np.column_stack(
        [
            np.ones(T),
            rng.normal(0, 0.012, T),
            rng.normal(0, 0.006, T),
            rng.normal(0, 0.006, T),
        ]
    )
    eps = rng.normal(0, 0.005, T)
    stock_excess = 1.0 * factors[:, 1] + 0.3 * factors[:, 2] + eps
    resid, resid_std = _compute_residuals_for_window(stock_excess, factors)
    # Recovered residuals should correlate strongly with the planted noise.
    valid = np.isfinite(resid)
    corr = np.corrcoef(resid[valid], eps[valid])[0, 1]
    assert corr > 0.99, f"Residual recovery failed; correlation = {corr:.4f}"


def test_compute_residuals_returns_nan_on_insufficient_data():
    factors = np.column_stack([np.ones(3), [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    stock_excess = np.array([0.1, 0.2, 0.3])
    resid, std = _compute_residuals_for_window(stock_excess, factors)
    assert np.isnan(std)


# ── (3) Signal ─────────────────────────────────────────────────────────────


def test_residual_signal_only_populated_on_month_ends():
    stocks, ff3 = _synthetic_universe(n_stocks=10, n_bars=252 * 4)
    cfg = ResidualConfig(signal_mode="residual", regression_months=24)
    sig = compute_residual_signal(stocks, ff3, cfg=cfg)
    me = _month_end_mask(sig.index)
    # Sample a few non-month-end rows -- they should be all-NaN.
    non_me = np.where(~me)[0]
    for i in non_me[:20]:
        assert sig.iloc[i].isna().all()


def test_raw_signal_mode_computes_cumulative_log_returns():
    stocks, ff3 = _synthetic_universe(n_stocks=10, n_bars=252 * 4)
    cfg = ResidualConfig(signal_mode="raw")
    sig = compute_residual_signal(stocks, ff3, cfg=cfg)
    # Should produce finite values on month-ends after warmup.
    me_idx = sig.index[_month_end_mask(sig.index)]
    if len(me_idx) > 14:  # need at least 12+1 months of warmup
        late = sig.loc[me_idx[14:]]
        assert late.notna().any().any()


def test_residual_signal_rejects_unknown_mode():
    stocks, ff3 = _synthetic_universe(n_stocks=5, n_bars=300)
    cfg = ResidualConfig(signal_mode="bogus")  # type: ignore[arg-type]
    try:
        compute_residual_signal(stocks, ff3, cfg=cfg)
    except ValueError as e:
        assert "signal_mode" in str(e)
    else:
        raise AssertionError("Expected ValueError for bogus signal_mode")


# ── (4) Portfolio construction ───────────────────────────────────────────


def test_long_short_portfolio_is_dollar_neutral():
    stocks, ff3 = _synthetic_universe(n_stocks=15, n_bars=252 * 4)
    cfg = ResidualConfig(long_only=False, regression_months=24)
    sig = compute_residual_signal(stocks, ff3, cfg=cfg)
    w = build_portfolio_weights(sig, cfg=cfg)
    # Whenever the position is non-zero, the long+short legs sum to ~0.
    sums = w.sum(axis=1)
    held_days = sums[sums.abs() > 1e-9]
    if len(held_days) > 0:
        assert held_days.abs().max() < 1e-9 + 0.01


def test_long_only_has_no_short_legs():
    stocks, ff3 = _synthetic_universe(n_stocks=15, n_bars=252 * 4)
    cfg = ResidualConfig(long_only=True, regression_months=24)
    sig = compute_residual_signal(stocks, ff3, cfg=cfg)
    w = build_portfolio_weights(sig, cfg=cfg)
    assert (w >= -1e-9).all().all()


# ── (5) Returns + causality ──────────────────────────────────────────────


def test_residual_returns_proper_shape():
    stocks, ff3 = _synthetic_universe(n_stocks=10, n_bars=252 * 4)
    cfg = ResidualConfig(regression_months=24, apply_costs=False)
    rets = residual_returns(stocks, ff3, cfg=cfg)
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(stocks)


def test_residual_returns_positive_on_designed_universe():
    """On the synthetic universe with split idiosyncratic drift, the
    residual-momentum strategy SHOULD earn positive returns (the
    cross-section separates winners and losers by idio drift)."""
    stocks, ff3 = _synthetic_universe(n_stocks=20, n_bars=252 * 6, seed=11)
    cfg = ResidualConfig(regression_months=36, apply_costs=False)
    rets = residual_returns(stocks, ff3, cfg=cfg)
    # Use the late portion (post-warmup) for the Sharpe check.
    late = rets.iloc[252 * 4 :]
    sharpe = late.mean() / late.std(ddof=1) * np.sqrt(252) if late.std() > 0 else 0.0
    # Threshold 0.15 is a sanity check that the strategy is earning the
    # planted alpha; tighter thresholds are brittle on a small synthetic
    # universe (20 stocks, quintile L/S = 4 names per leg = noisy).
    assert sharpe > 0.15, f"Expected positive Sharpe on designed universe; got {sharpe:.3f}"


def test_residual_assert_causal_passes():
    stocks, ff3 = _synthetic_universe(n_stocks=10, n_bars=252 * 6, seed=12)
    cfg = ResidualConfig(regression_months=36, apply_costs=False)
    residual_assert_causal(stocks, ff3, cfg=cfg, n_trials=2, seed=42)


def test_corrupting_future_does_not_affect_past_returns():
    stocks, ff3 = _synthetic_universe(n_stocks=8, n_bars=252 * 5, seed=13)
    cfg = ResidualConfig(regression_months=24, apply_costs=False)
    base = residual_returns(stocks, ff3, cfg=cfg)
    s2 = stocks.copy()
    s2.iloc[-1] = s2.iloc[-1] * 5.0
    f2 = ff3.copy()
    f2.iloc[-1] = f2.iloc[-1] * 5.0
    altered = residual_returns(s2, f2, cfg=cfg)
    pd.testing.assert_series_equal(base.iloc[:-1], altered.iloc[:-1], check_names=False)


def test_costs_reduce_returns_when_position_held():
    stocks, ff3 = _synthetic_universe(n_stocks=10, n_bars=252 * 5)
    cfg_gross = ResidualConfig(regression_months=24, apply_costs=False)
    cfg_net = ResidualConfig(regression_months=24, apply_costs=True)
    gross = residual_returns(stocks, ff3, cfg=cfg_gross)
    net = residual_returns(stocks, ff3, cfg=cfg_net)
    assert net.sum() < gross.sum() + 1e-9
