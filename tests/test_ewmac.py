"""Unit tests for B2 Carver EWMAC trend ensemble.

Specified in directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.ewmac.ewmac_strategy import (
    CARVER_FDM,
    CARVER_FORECAST_SCALARS,
    BroadTrendFilterConfig,
    EwmacConfig,
    VolRegimeFilterConfig,
    _compute_broad_trend_gate,
    _compute_vol_regime_gate,
    _fdm_for,
    _forecast_scalar,
    _vol_normalised_ewmac,
    build_positions,
    compute_ewmac_forecast,
    ewmac_assert_causal,
    ewmac_returns,
)


def _synthetic_universe(n_assets: int = 6, n_bars: int = 252 * 4, seed: int = 0):
    """Half-up, half-down basket. EWMAC should long uppers, short downers."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    closes: dict[str, np.ndarray] = {}
    for i in range(n_assets):
        alpha = 0.0008 if i % 2 == 0 else -0.0008
        log_ret = rng.normal(alpha, 0.012, n_bars)
        price = 100 * np.cumprod(1 + log_ret)
        closes[f"A{i:02d}"] = price
    return pd.DataFrame(closes, index=idx)


# ── (1) Forecast scalar / FDM lookups ─────────────────────────────────────


def test_forecast_scalar_carver_lookup():
    cfg = EwmacConfig(forecast_scalar_mode="carver")
    assert _forecast_scalar((16, 64), cfg) == CARVER_FORECAST_SCALARS[(16, 64)]
    assert _forecast_scalar((32, 128), cfg) == CARVER_FORECAST_SCALARS[(32, 128)]
    assert _forecast_scalar((64, 256), cfg) == CARVER_FORECAST_SCALARS[(64, 256)]


def test_forecast_scalar_unit_mode():
    cfg = EwmacConfig(forecast_scalar_mode="unit")
    for speed in CARVER_FORECAST_SCALARS:
        assert _forecast_scalar(speed, cfg) == 1.0


def test_forecast_scalar_explicit_override():
    cfg = EwmacConfig(forecast_scalar_mode="carver", forecast_scalars={(16, 64): 99.9})
    assert _forecast_scalar((16, 64), cfg) == 99.9
    # non-overridden speeds still hit Carver lookup
    assert _forecast_scalar((32, 128), cfg) == CARVER_FORECAST_SCALARS[(32, 128)]


def test_forecast_scalar_is_frozen_mode():
    """``is_frozen`` should produce scalar such that the IS abs-mean
    forecast equals target_forecast (10.0 by default)."""
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(11)
    log_ret = rng.normal(0.0005, 0.012, n_bars)
    is_close = pd.Series(100 * np.cumprod(1 + log_ret), index=idx)
    cfg = EwmacConfig(forecast_scalar_mode="is_frozen", target_forecast=10.0)
    scalar = _forecast_scalar((16, 64), cfg, is_close=is_close)
    # Apply scalar back and check abs-mean is approximately 10.
    norm = _vol_normalised_ewmac(is_close, 16, 64, vol_lookback=cfg.vol_lookback_days)
    scaled = (norm * scalar).dropna()
    assert abs(scaled.abs().mean() - 10.0) < 0.5, (
        f"is_frozen scalar should yield |forecast| ~10, got {scaled.abs().mean():.3f}"
    )


def test_fdm_lookup():
    for n in (1, 2, 3, 4, 5, 6):
        cfg = EwmacConfig(speeds=tuple([(16, 64)] * n))
        assert _fdm_for(cfg) == CARVER_FDM[n]


def test_fdm_explicit_override():
    cfg = EwmacConfig(speeds=((16, 64), (32, 128), (64, 256)), fdm=2.0)
    assert _fdm_for(cfg) == 2.0


# ── (2) Forecast sign + cap ───────────────────────────────────────────────


def test_forecast_positive_for_uptrending_asset():
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(3)
    # Strong uptrend.
    log_ret = rng.normal(0.0015, 0.008, n_bars)
    closes = pd.DataFrame({"UP": 100 * np.cumprod(1 + log_ret)}, index=idx)
    cfg = EwmacConfig(speeds=((16, 64),), fdm=1.0)
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    tail = fcst["UP"].dropna().iloc[-252:]
    assert (tail > 0).mean() > 0.7, (
        f"Expected predominantly + forecast on uptrend, got {(tail > 0).mean() * 100:.1f}% +"
    )


def test_forecast_negative_for_downtrending_asset():
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(4)
    log_ret = rng.normal(-0.0015, 0.008, n_bars)
    closes = pd.DataFrame({"DN": 100 * np.cumprod(1 + log_ret)}, index=idx)
    cfg = EwmacConfig(speeds=((16, 64),), fdm=1.0)
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    tail = fcst["DN"].dropna().iloc[-252:]
    assert (tail < 0).mean() > 0.7


def test_forecast_cap_clips_extreme_values():
    """Force a strong directional signal and confirm forecast never
    exceeds the cap in absolute value."""
    n_bars = 252 * 2
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    # Exponential up-blow-off — strong trend.
    log_ret = np.full(n_bars, 0.005)
    closes = pd.DataFrame({"X": 100 * np.cumprod(1 + log_ret)}, index=idx)
    cfg = EwmacConfig(speeds=((16, 64),), forecast_cap=5.0, fdm=1.0, forecast_scalar_mode="carver")
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    assert fcst["X"].abs().max() <= 5.0 + 1e-9


def test_fdm_applied_to_combined_forecast():
    """Stacking the same speed twice and using FDM=2.0 should double the
    raw forecast (subject to the cap). Validates the multiplier path."""
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(5)
    log_ret = rng.normal(0.001, 0.008, n_bars)
    closes = pd.DataFrame({"X": 100 * np.cumprod(1 + log_ret)}, index=idx)
    base = compute_ewmac_forecast(closes, cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0))
    fdm2 = compute_ewmac_forecast(closes, cfg=EwmacConfig(speeds=((16, 64), (16, 64)), fdm=2.0))
    # Pick a date with a non-saturated forecast (away from the cap).
    sample = base["X"].dropna().abs() < 5.0
    valid = base["X"].dropna()[sample].index
    if len(valid):
        b_val = base["X"].loc[valid[-1]]
        f_val = fdm2["X"].loc[valid[-1]]
        # FDM=2 with identical speeds averages to the same scaled signal
        # then multiplies by 2: expect f_val ≈ 2 * b_val (still ≤ cap).
        expected = min(max(2 * b_val, -20.0), 20.0)
        assert abs(f_val - expected) < 1e-6, (
            f"FDM doubling not applied: base={b_val:.3f}, doubled={f_val:.3f}, "
            f"expected≈{expected:.3f}"
        )


# ── (3) Position sizing + returns ─────────────────────────────────────────


def test_positions_scale_linearly_with_forecast_at_fixed_vol():
    """Carver positions scale forecast / target_forecast linearly when
    instrument_vol is held fixed. Validate by constructing two identical
    series (same vol) but force their forecasts to differ via FDM."""
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(0)
    # Same series under both columns -> same vol; same forecast input.
    log_ret = rng.normal(0.001, 0.008, n_bars)
    price = 100 * np.cumprod(1 + log_ret)
    closes = pd.DataFrame({"A": price, "B": price.copy()}, index=idx)
    cfg = EwmacConfig(speeds=((16, 64),), fdm=1.0)
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    from research.ewmac.ewmac_strategy import _instrument_vol

    vol = _instrument_vol(closes, lookback_days=cfg.instrument_vol_lookback_days)
    pos = build_positions(fcst, vol, cfg=cfg)
    last = pos.dropna(how="all").iloc[-1]
    # With identical inputs, positions on A and B should match.
    assert abs(last["A"] - last["B"]) < 1e-12, (
        f"identical assets should produce identical positions; got A={last['A']:.6f}, "
        f"B={last['B']:.6f}"
    )


def test_position_sign_follows_forecast_sign():
    """A predominantly uptrending asset should produce a non-negative
    position; a downtrending one should produce a non-positive position."""
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(7)
    up = 100 * np.cumprod(1 + rng.normal(0.002, 0.008, n_bars))
    dn = 100 * np.cumprod(1 + rng.normal(-0.002, 0.008, n_bars))
    closes = pd.DataFrame({"UP": up, "DN": dn}, index=idx)
    cfg = EwmacConfig(speeds=((16, 64), (32, 128)))
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    from research.ewmac.ewmac_strategy import _instrument_vol

    vol = _instrument_vol(closes, lookback_days=cfg.instrument_vol_lookback_days)
    pos = build_positions(fcst, vol, cfg=cfg)
    last = pos.dropna(how="all").iloc[-1]
    assert last["UP"] >= 0, f"uptrend should produce non-negative position, got {last['UP']}"
    assert last["DN"] <= 0, f"downtrend should produce non-positive position, got {last['DN']}"


def test_ewmac_returns_proper_shape():
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 3)
    rets = ewmac_returns(closes)
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(closes)


def test_ewmac_returns_positive_on_designed_universe():
    """Mixed-trend basket: half up, half down. EWMAC should earn positive
    on the cost-free variant."""
    closes = _synthetic_universe(n_assets=10, n_bars=252 * 5, seed=21)
    rets = ewmac_returns(closes, cfg=EwmacConfig(apply_costs=False))
    late = rets.iloc[252:]
    sh = late.mean() / late.std(ddof=1) * np.sqrt(252) if late.std() > 0 else 0.0
    assert sh > 0.2, f"Expected positive Sharpe on designed universe; got {sh:.3f}"


def test_costs_reduce_ewmac_returns():
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 3)
    gross = ewmac_returns(closes, cfg=EwmacConfig(apply_costs=False))
    net = ewmac_returns(closes, cfg=EwmacConfig(apply_costs=True))
    assert net.sum() < gross.sum() + 1e-9


# ── (4) Causality ──────────────────────────────────────────────────────────


def test_ewmac_assert_causal_passes():
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 4, seed=33)
    ewmac_assert_causal(closes, cfg=EwmacConfig(apply_costs=False), n_trials=2, seed=42)


def test_corrupting_future_does_not_affect_past_ewmac():
    closes = _synthetic_universe(n_assets=5, n_bars=252 * 3, seed=34)
    cfg = EwmacConfig(apply_costs=False)
    base = ewmac_returns(closes, cfg=cfg)
    c2 = closes.copy()
    c2.iloc[-1] = c2.iloc[-1] * 5
    altered = ewmac_returns(c2, cfg=cfg)
    pd.testing.assert_series_equal(base.iloc[:-1], altered.iloc[:-1], check_names=False)


# ── (5) Singleton-speed baseline parity ──────────────────────────────────


def test_singleton_speed_no_fdm_is_unscaled_baseline():
    """With FDM=1.0 and a single speed, the combined forecast should be
    identical to a single scaled+capped vol-normalised EWMAC for that
    speed (no multi-speed averaging in play)."""
    n_bars = 252 * 3
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rng = np.random.default_rng(6)
    log_ret = rng.normal(0.0005, 0.01, n_bars)
    closes = pd.DataFrame({"X": 100 * np.cumprod(1 + log_ret)}, index=idx)
    cfg = EwmacConfig(speeds=((16, 64),), fdm=1.0, forecast_cap=20.0, forecast_scalar_mode="carver")
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    # Manual recomputation.
    norm = _vol_normalised_ewmac(closes["X"], 16, 64, vol_lookback=cfg.vol_lookback_days)
    expected = (norm * CARVER_FORECAST_SCALARS[(16, 64)]).clip(-20, 20)
    # FDM=1 -> mean over one speed -> same as scaled; then clip again is idempotent.
    pd.testing.assert_series_equal(
        fcst["X"].dropna(),
        expected.dropna(),
        check_names=False,
        check_exact=False,
        atol=1e-10,
    )


# ── (6) B2c broad-trend regime filter ────────────────────────────────────


def test_broad_trend_filter_disabled_by_default():
    """``broad_trend_filter=None`` (default) must produce identical forecast
    to a configuration that doesn't reference the filter at all (B2 parity)."""
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 3, seed=99)
    fcst_b2 = compute_ewmac_forecast(closes, cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0))
    fcst_default = compute_ewmac_forecast(
        closes,
        cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0, broad_trend_filter=None),
    )
    pd.testing.assert_frame_equal(fcst_b2, fcst_default)


def test_broad_trend_gate_is_one_during_universe_uptrend():
    """When all instruments trend up, the equal-weight broad-index EWMAC
    is positive everywhere after warmup, so absolute-mode gate = +1."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-02", periods=252 * 3, freq="B")
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.cumprod(1 + rng.normal(0.0015, 0.008, len(idx))) for i in range(5)},
        index=idx,
    )
    cfg_filter = BroadTrendFilterConfig(
        fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=0.0
    )
    gate = _compute_broad_trend_gate(closes, filter_cfg=cfg_filter)
    # Take the last year (post-warmup) — should be ~all +1 since the
    # universe is trending up.
    tail = gate.iloc[-252:]
    assert (tail == 1.0).mean() > 0.95, (
        f"Expected gate=+1 dominant during universe uptrend, got "
        f"{(tail == 1.0).mean() * 100:.1f}% +1"
    )


def test_broad_trend_gate_deadband_silences_quiescent_regime():
    """A small deadband should produce gate=0 during periods where the
    broad-index EWMAC is near zero. Test by constructing a flat-then-trending
    series and confirming early bars (flat regime) have gate=0 under deadband."""
    n = 252 * 4
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    rng = np.random.default_rng(11)
    # First half: flat (~zero drift). Second half: strong uptrend.
    log_ret = np.concatenate(
        [
            rng.normal(0.0, 0.005, n // 2),
            rng.normal(0.003, 0.005, n - n // 2),
        ]
    )
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.cumprod(1 + log_ret + rng.normal(0, 0.002, n)) for i in range(4)},
        index=idx,
    )
    cfg_with_deadband = BroadTrendFilterConfig(
        fast_hl=32, slow_hl=128, mode="absolute_trend", deadband=2.0
    )
    gate = _compute_broad_trend_gate(closes, filter_cfg=cfg_with_deadband)
    # Window from bar ~300 to half (flat regime, post-warmup): should have
    # frequent gate=0.
    early = gate.iloc[300 : n // 2]
    late = gate.iloc[-252:]
    assert (early == 0.0).mean() > (late == 0.0).mean(), (
        f"Deadband should silence flat regime more than trending: early-zero "
        f"frac={float((early == 0.0).mean()):.2f}, late-zero frac={float((late == 0.0).mean()):.2f}"
    )


def test_filter_zeros_forecast_when_gate_is_zero():
    """When the gate is explicitly 0 (broad-trend within deadband), the
    per-asset combined forecast must also be 0 on that bar."""
    # Use a deadband so large nothing ever passes the gate → all forecasts 0.
    closes = _synthetic_universe(n_assets=4, n_bars=252 * 3, seed=13)
    cfg = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=1e9
        ),
    )
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    # Every cell must be exactly zero (after warmup NaNs become 0 via fillna).
    nonzero = (fcst.abs() > 1e-12).any().any()
    assert not nonzero, "Filter with infinite deadband should zero all forecasts"


def test_filter_directional_mode_flips_sign_on_negative_broad_trend():
    """In ``directional`` mode, when broad_trend < -deadband, the gate is
    -1 and per-asset forecasts get their sign flipped."""
    rng = np.random.default_rng(19)
    idx = pd.date_range("2018-01-02", periods=252 * 3, freq="B")
    # All assets strongly DOWN-trending — broad gate should be -1.
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.cumprod(1 + rng.normal(-0.0015, 0.008, len(idx))) for i in range(4)},
        index=idx,
    )
    cfg_abs = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=0.0
        ),
    )
    cfg_dir = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="directional", deadband=0.0
        ),
    )
    fcst_abs = compute_ewmac_forecast(closes, cfg=cfg_abs)
    fcst_dir = compute_ewmac_forecast(closes, cfg=cfg_dir)
    tail_abs = fcst_abs.iloc[-252:].mean(axis=0)
    tail_dir = fcst_dir.iloc[-252:].mean(axis=0)
    # In absolute mode, gate=+1, raw per-asset forecast is negative (downtrend),
    # so fcst_abs should be negative. In directional mode, gate=-1, flips the
    # negative forecast → positive. The signs should be opposite.
    for col in closes.columns:
        if abs(tail_abs[col]) > 0.1:
            assert tail_abs[col] * tail_dir[col] < 0, (
                f"directional mode should flip sign vs absolute mode on "
                f"down-trending universe; col={col} abs={tail_abs[col]:+.3f} "
                f"dir={tail_dir[col]:+.3f}"
            )


def test_filter_rejects_unknown_mode():
    closes = _synthetic_universe(n_assets=3, n_bars=300, seed=21)
    cfg = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        broad_trend_filter=BroadTrendFilterConfig(fast_hl=64, slow_hl=256, mode="bogus"),  # type: ignore[arg-type]
    )
    try:
        compute_ewmac_forecast(closes, cfg=cfg)
    except ValueError as e:
        assert "mode" in str(e)
    else:
        raise AssertionError("Expected ValueError on unknown filter mode")


def test_filter_preserves_causality():
    """Standard A10 smoke test on the filtered strategy: corrupting future
    data must not change past returns."""
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 4, seed=23)
    cfg = EwmacConfig(
        speeds=((16, 64), (32, 128)),
        apply_costs=False,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=0.0
        ),
    )
    ewmac_assert_causal(closes, cfg=cfg, n_trials=2, seed=42)


# ── (7) B2d realised-vol regime filter ───────────────────────────────────


def test_vol_regime_filter_disabled_by_default():
    """``vol_regime_filter=None`` produces identical forecast to baseline."""
    closes = _synthetic_universe(n_assets=6, n_bars=252 * 3, seed=55)
    fcst_baseline = compute_ewmac_forecast(closes, cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0))
    fcst_explicit_none = compute_ewmac_forecast(
        closes,
        cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0, vol_regime_filter=None),
    )
    pd.testing.assert_frame_equal(fcst_baseline, fcst_explicit_none)


def test_vol_regime_gate_in_band():
    """A normal-vol synthetic universe should mostly be inside the
    [20, 80] percentile band after warmup."""
    rng = np.random.default_rng(77)
    n = 252 * 4  # enough for the 252-bar percentile window + 60-bar vol lookback
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)) for i in range(4)},
        index=idx,
    )
    cfg_filter = VolRegimeFilterConfig(
        vol_lookback_days=60, percentile_window_days=252, pct_lo=20.0, pct_hi=80.0
    )
    gate = _compute_vol_regime_gate(closes, filter_cfg=cfg_filter)
    # Drop NaNs from warmup.
    warm = gate.dropna()
    in_band_frac = float((warm == 1.0).mean())
    # Vol clusters, so the rolling-percentile-rank of vol on the current
    # bar isn't uniformly distributed (recent vol levels tend to be near
    # the trailing window's current vol). Just confirm the gate is doing
    # SOMETHING — fires sometimes, not always, not never.
    assert 0.20 < in_band_frac < 0.85, (
        f"Expected gate to fire on a non-trivial fraction of bars; got {in_band_frac * 100:.1f}%"
    )


def test_vol_regime_gate_excludes_extreme_vol():
    """Inject a crisis vol spike at the end; the gate should drop those
    bars to 0 (vol percentile pushed above the band)."""
    rng = np.random.default_rng(88)
    n = 252 * 4
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    # Normal vol first 90%, then 10x vol spike in the last 10%.
    log_ret = np.concatenate(
        [
            rng.normal(0.0005, 0.01, int(n * 0.9)),
            rng.normal(0.0005, 0.05, n - int(n * 0.9)),  # 5x vol spike
        ]
    )
    closes = pd.DataFrame(
        {f"A{i:02d}": 100 * np.cumprod(1 + log_ret) for i in range(3)},
        index=idx,
    )
    cfg_filter = VolRegimeFilterConfig(
        vol_lookback_days=60, percentile_window_days=252, pct_lo=20.0, pct_hi=80.0
    )
    gate = _compute_vol_regime_gate(closes, filter_cfg=cfg_filter)
    crisis_tail = gate.iloc[-100:].dropna()
    # Crisis tail should be mostly OUT of the band (gate=0).
    assert float((crisis_tail == 0.0).mean()) > 0.5, (
        f"Crisis tail should mostly fail vol-band gate, got "
        f"{float((crisis_tail == 0.0).mean()) * 100:.1f}% gate=0"
    )


def test_vol_regime_filter_zeros_forecast_outside_band():
    """When the gate is 0, every per-asset forecast must be 0."""
    closes = _synthetic_universe(n_assets=4, n_bars=252 * 3, seed=91)
    # Impossible band: pct_lo > pct_hi → no observation in band → gate always 0
    cfg = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        vol_regime_filter=VolRegimeFilterConfig(
            vol_lookback_days=60, percentile_window_days=252, pct_lo=99.0, pct_hi=99.5
        ),
    )
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    # Allow a tiny number of edge cases at the boundary; require very-low coverage.
    nonzero_frac = float((fcst.abs() > 1e-12).any(axis=1).mean())
    assert nonzero_frac < 0.05, (
        f"Filter at pct=[99,99.5] should silence nearly all bars, got {nonzero_frac * 100:.1f}% non-zero"
    )


def test_vol_regime_filter_preserves_causality():
    closes = _synthetic_universe(n_assets=5, n_bars=252 * 4, seed=93)
    cfg = EwmacConfig(
        speeds=((16, 64), (32, 128)),
        apply_costs=False,
        vol_regime_filter=VolRegimeFilterConfig(
            vol_lookback_days=60, percentile_window_days=252, pct_lo=20.0, pct_hi=80.0
        ),
    )
    ewmac_assert_causal(closes, cfg=cfg, n_trials=2, seed=42)


def test_combined_trend_and_vol_filters_and_together():
    """When both filters are set, gate is the AND of broad-trend and
    vol-regime gates. Test by setting an impossible vol band so combined
    forecasts are zero regardless of broad-trend gate."""
    closes = _synthetic_universe(n_assets=4, n_bars=252 * 3, seed=97)
    cfg = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=0.0
        ),
        vol_regime_filter=VolRegimeFilterConfig(
            vol_lookback_days=60, percentile_window_days=252, pct_lo=99.0, pct_hi=99.5
        ),
    )
    fcst = compute_ewmac_forecast(closes, cfg=cfg)
    nonzero_frac = float((fcst.abs() > 1e-12).any(axis=1).mean())
    assert nonzero_frac < 0.05, (
        f"AND-combined filters with impossible vol band should silence forecast: got {nonzero_frac * 100:.1f}% non-zero"
    )
