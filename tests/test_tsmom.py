"""Unit tests for B4 TSMOM.

Specified in directives/Pre-Reg B4 TSMOM 2026-05-15.md §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.tsmom.tsmom_strategy import (
    TsmomConfig,
    _month_end_mask,
    _realised_vol,
    build_portfolio_weights,
    compute_tsmom_signal,
    tsmom_assert_causal,
    tsmom_returns,
)
from titan.research.framework import StrategyClass, defaults_for


def _synthetic_universe(n_assets: int = 10, n_bars: int = 252 * 6, seed: int = 0):
    """Synthetic basket with mixed trend signs.

    Half the assets trend up (alpha=+0.001 daily), half trend down
    (alpha=-0.001 daily). A TSMOM strategy with sign-of-cumulative-return
    SHOULD long the up-trenders, short the down-trenders, earn positive.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    closes: dict[str, np.ndarray] = {}
    for i in range(n_assets):
        alpha = 0.001 if i % 2 == 0 else -0.001
        log_ret = rng.normal(alpha, 0.012, n_bars)
        price = 100 * np.cumprod(1 + log_ret)
        closes[f"A{i:02d}"] = price
    return pd.DataFrame(closes, index=idx)


# ── (1) Class defaults ────────────────────────────────────────────────────


def test_uses_cross_asset_momentum_defaults():
    d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    assert d.sharpe.primary == "per_day_mtm"
    assert d.mc.max_dd_threshold_pct > 0


# ── (2) Signal ────────────────────────────────────────────────────────────


def test_tsmom_signal_only_on_month_ends():
    closes = _synthetic_universe(n_assets=5, n_bars=252 * 3)
    cfg = TsmomConfig(momentum_window_months=12, skip_months=1)
    sig = compute_tsmom_signal(closes, cfg=cfg)
    me = _month_end_mask(sig.index)
    non_me = np.where(~me)[0]
    for i in non_me[:20]:
        assert sig.iloc[i].isna().all()


def test_tsmom_signal_positive_for_uptrending():
    idx = pd.date_range("2020-01-02", periods=252 * 3, freq="B")
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0.001, 0.005, len(idx))
    price = 100 * np.cumprod(1 + log_ret)
    closes = pd.DataFrame({"UP": price}, index=idx)
    cfg = TsmomConfig(momentum_window_months=12, skip_months=1)
    sig = compute_tsmom_signal(closes, cfg=cfg)
    me_idx = sig.index[_month_end_mask(sig.index)]
    # Late month-ends should have positive sign.
    late = sig.loc[me_idx[-12:]]
    assert (late["UP"].dropna() > 0).all()


def test_tsmom_signal_negative_for_downtrending():
    idx = pd.date_range("2020-01-02", periods=252 * 3, freq="B")
    rng = np.random.default_rng(1)
    log_ret = rng.normal(-0.001, 0.005, len(idx))
    price = 100 * np.cumprod(1 + log_ret)
    closes = pd.DataFrame({"DOWN": price}, index=idx)
    cfg = TsmomConfig(momentum_window_months=12, skip_months=1)
    sig = compute_tsmom_signal(closes, cfg=cfg)
    me_idx = sig.index[_month_end_mask(sig.index)]
    late = sig.loc[me_idx[-12:]]
    assert (late["DOWN"].dropna() < 0).all()


def test_tsmom_signal_rejects_unknown_mode():
    closes = _synthetic_universe(n_assets=3, n_bars=300)
    cfg = TsmomConfig(signal_mode="bogus")  # type: ignore[arg-type]
    try:
        compute_tsmom_signal(closes, cfg=cfg)
    except ValueError as e:
        assert "signal_mode" in str(e)
    else:
        raise AssertionError("Expected ValueError")


# ── (3) Vol weighting ─────────────────────────────────────────────────────


def test_inv_vol_weights_smaller_for_higher_vol_assets():
    """Two assets, same trend direction, different vols. The higher-vol
    asset should get a SMALLER absolute weight under inv-vol."""
    idx = pd.date_range("2020-01-02", periods=252 * 3, freq="B")
    rng = np.random.default_rng(0)
    # Low-vol asset.
    lv_ret = rng.normal(0.001, 0.005, len(idx))
    # High-vol asset.
    hv_ret = rng.normal(0.001, 0.015, len(idx))
    lv = 100 * np.cumprod(1 + lv_ret)
    hv = 100 * np.cumprod(1 + hv_ret)
    closes = pd.DataFrame({"LV": lv, "HV": hv}, index=idx)
    cfg = TsmomConfig(momentum_window_months=12, skip_months=1, weighting="inv_vol")
    sig = compute_tsmom_signal(closes, cfg=cfg)
    vol = _realised_vol(closes, lookback_days=60)
    w = build_portfolio_weights(sig, vol, cfg=cfg)
    # On a rebalance day post-warmup, |w_LV| > |w_HV|.
    me_mask = _month_end_mask(w.index)
    me_rows = w[me_mask]
    valid_rows = me_rows.dropna(how="all").iloc[-3:]
    for _, row in valid_rows.iterrows():
        if abs(row["LV"]) > 0 and abs(row["HV"]) > 0:
            assert abs(row["LV"]) > abs(row["HV"]), (
                f"inv-vol failed: |w_LV|={row['LV']:.4f}, |w_HV|={row['HV']:.4f}"
            )


def test_equal_weight_mode_gives_uniform_magnitudes():
    closes = _synthetic_universe(n_assets=4, n_bars=252 * 3)
    cfg = TsmomConfig(weighting="equal")
    sig = compute_tsmom_signal(closes, cfg=cfg)
    vol = _realised_vol(closes, lookback_days=60)
    w = build_portfolio_weights(sig, vol, cfg=cfg)
    me_rows = w.iloc[_month_end_mask(w.index)]
    valid_rows = me_rows.dropna(how="all")
    if len(valid_rows) > 0:
        last_row = valid_rows.iloc[-1]
        nonzero = last_row[last_row.abs() > 1e-9].abs()
        if len(nonzero) > 1:
            # All non-zero weights should have the same magnitude.
            assert (nonzero.max() - nonzero.min()) < 1e-9


# ── (4) Returns + causality ──────────────────────────────────────────────


def test_tsmom_returns_proper_shape():
    closes = _synthetic_universe(n_assets=10, n_bars=252 * 4)
    rets = tsmom_returns(closes, cfg=TsmomConfig())
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(closes)


def test_tsmom_returns_positive_on_designed_universe():
    """Mixed-trend basket: half up, half down. TSMOM should long the
    uppers, short the downers, earn positive net of noise."""
    closes = _synthetic_universe(n_assets=10, n_bars=252 * 6, seed=11)
    cfg = TsmomConfig(apply_costs=False)
    rets = tsmom_returns(closes, cfg=cfg)
    late = rets.iloc[252 * 3 :]
    sharpe_ratio = late.mean() / late.std(ddof=1) * np.sqrt(252) if late.std() > 0 else 0.0
    assert sharpe_ratio > 0.5, (
        f"Expected positive Sharpe on designed universe; got {sharpe_ratio:.3f}"
    )


def test_tsmom_assert_causal_passes():
    closes = _synthetic_universe(n_assets=8, n_bars=252 * 6, seed=12)
    tsmom_assert_causal(closes, cfg=TsmomConfig(apply_costs=False), n_trials=2, seed=42)


def test_corrupting_future_does_not_affect_past():
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 5, seed=13)
    cfg = TsmomConfig(apply_costs=False)
    base = tsmom_returns(closes, cfg=cfg)
    c2 = closes.copy()
    c2.iloc[-1] = c2.iloc[-1] * 5
    altered = tsmom_returns(c2, cfg=cfg)
    pd.testing.assert_series_equal(base.iloc[:-1], altered.iloc[:-1], check_names=False)


def test_costs_reduce_returns():
    closes = _synthetic_universe(n_assets=8, n_bars=252 * 4)
    gross = tsmom_returns(closes, cfg=TsmomConfig(apply_costs=False))
    net = tsmom_returns(closes, cfg=TsmomConfig(apply_costs=True))
    assert net.sum() < gross.sum() + 1e-9


# ── (5) Rebalance frequency ───────────────────────────────────────────────


def test_weekly_rebalance_produces_more_rebalances_than_monthly():
    closes = _synthetic_universe(n_assets=5, n_bars=252 * 3)
    sig_m = compute_tsmom_signal(closes, cfg=TsmomConfig(rebalance="monthly"))
    sig_w = compute_tsmom_signal(closes, cfg=TsmomConfig(rebalance="weekly"))
    n_m_rebal = sig_m.notna().any(axis=1).sum()
    n_w_rebal = sig_w.notna().any(axis=1).sum()
    assert n_w_rebal > n_m_rebal


def test_rebalance_rejects_unknown():
    closes = _synthetic_universe(n_assets=3, n_bars=300)
    cfg = TsmomConfig(rebalance="bogus")  # type: ignore[arg-type]
    try:
        compute_tsmom_signal(closes, cfg=cfg)
    except ValueError as e:
        assert "rebalance" in str(e)
    else:
        raise AssertionError("Expected ValueError")


# ── (6) B4c window-ensemble — pre-Reg B4c Window-Ensemble TSMOM ──────────


def test_singleton_tuple_window_is_b4b_parity():
    """`momentum_window_months=(12,)` must produce identical signal to
    `momentum_window_months=12`. Required for B4c C6 baseline test."""
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 4)
    sig_int = compute_tsmom_signal(closes, cfg=TsmomConfig(momentum_window_months=12))
    sig_tup = compute_tsmom_signal(closes, cfg=TsmomConfig(momentum_window_months=(12,)))
    # Drop all-NaN rows then compare values.
    sig_int_valid = sig_int.dropna(how="all")
    sig_tup_valid = sig_tup.dropna(how="all")
    pd.testing.assert_frame_equal(sig_int_valid, sig_tup_valid, check_dtype=False)


def test_vote_aggregation_averages_signs():
    """When 3 windows all yield +1 sign on an asset, vote gives +1.
    When they split 2:1, vote gives 1/3."""
    # Construct an asset that's strongly up-trending: any window will see +.
    rng = np.random.default_rng(0)
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    up = 100 * np.cumprod(1 + rng.normal(0.002, 0.005, n_bars))  # strong uptrend
    closes = pd.DataFrame({"UP": up}, index=idx)
    cfg = TsmomConfig(momentum_window_months=(9, 12, 15), ensemble_aggregation="vote")
    sig = compute_tsmom_signal(closes, cfg=cfg)
    last_rebal = sig.dropna(how="all").iloc[-1]
    # All three windows should agree on +1 sign -> vote = 1.0.
    assert abs(last_rebal["UP"] - 1.0) < 1e-9, f"expected +1.0, got {last_rebal['UP']}"


def test_vote_aggregation_partial_disagreement():
    """If only 1 of 3 windows sees -1 and the other 2 see +1, vote = +1/3."""
    n_bars = 252 * 4
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    # Build a series that:
    # - has been DOWN over the last 9 months (negative cum return [end-9m, end))
    # - has been UP over the last 12 AND 15 months  (positive cum return)
    # Construction: strong early uptrend, modest recent downtrend. The
    # 15-month and 12-month windows see big early + smaller recent = net +.
    # The 9-month window sees only the modest downtrend = net -.
    n_recent = 21 * 9  # ~189 bars
    # 9m down at -0.001/day -> cum log ret ~ -0.189
    # Need 12m: 63 bars (3m) of strong + minus 189 bars (9m) of mild -.
    #   For 12m + : 63*X > 0.189 -> X > 0.003
    log_ret = np.concatenate(
        [
            np.full(n_bars - n_recent, 0.005),  # strong early uptrend
            np.full(n_recent, -0.001),  # mild recent downtrend
        ]
    )
    closes = pd.DataFrame({"MIX": 100 * np.cumprod(1 + log_ret)}, index=idx)
    cfg = TsmomConfig(
        momentum_window_months=(9, 12, 15), ensemble_aggregation="vote", skip_months=0
    )
    sig = compute_tsmom_signal(closes, cfg=cfg)
    last_rebal = sig.dropna(how="all").iloc[-1]
    # Expect 9-month sees down (-1), 12 and 15 see up (+1, +1) -> vote = +1/3.
    expected = (-1 + 1 + 1) / 3.0
    assert abs(last_rebal["MIX"] - expected) < 1e-9, f"expected {expected}, got {last_rebal['MIX']}"


def test_weighted_sum_aggregation_differs_from_vote():
    """When one window has a much larger |cum_return| than the others,
    weighted_sum should put more emphasis on it than vote."""
    rng = np.random.default_rng(1)
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    # Asset is strongly +trending so all windows agree on sign, but their
    # magnitudes differ (longer window has bigger cum return).
    log_ret = rng.normal(0.002, 0.005, n_bars)
    closes = pd.DataFrame({"X": 100 * np.cumprod(1 + log_ret)}, index=idx)
    cfg_vote = TsmomConfig(
        momentum_window_months=(6, 12), ensemble_aggregation="vote", signal_mode="sign"
    )
    cfg_wsum = TsmomConfig(
        momentum_window_months=(6, 12), ensemble_aggregation="weighted_sum", signal_mode="sign"
    )
    sig_vote = compute_tsmom_signal(closes, cfg=cfg_vote).dropna(how="all").iloc[-1]
    sig_wsum = compute_tsmom_signal(closes, cfg=cfg_wsum).dropna(how="all").iloc[-1]
    # Vote: mean of two +1 signs = +1.0
    assert abs(sig_vote["X"] - 1.0) < 1e-9
    # Weighted: mean of (+1 * |cum_6|, +1 * |cum_12|) = mean of two positive
    # magnitudes — should be > 0 but generally != 1.0.
    assert sig_wsum["X"] > 0.0
    assert abs(sig_wsum["X"] - 1.0) > 1e-6, (
        "weighted_sum should differ from vote when |cum_ret| varies"
    )


def test_ensemble_rejects_unknown_aggregation():
    closes = _synthetic_universe(n_assets=3, n_bars=300)
    cfg = TsmomConfig(momentum_window_months=(9, 12), ensemble_aggregation="bogus")  # type: ignore[arg-type]
    try:
        compute_tsmom_signal(closes, cfg=cfg)
    except ValueError as e:
        assert "ensemble_aggregation" in str(e)
    else:
        raise AssertionError("Expected ValueError")
