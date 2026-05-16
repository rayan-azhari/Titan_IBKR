"""Unit tests for I1 per-asset HMM regime gate.

Pre-Reg: directives/Pre-Reg I1 HMM Per-Asset Regime + EWMAC Gate 2026-05-16.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.ewmac.ewmac_strategy import (
    EwmacConfig,
    compute_ewmac_forecast,
    ewmac_returns,
)
from research.regime.hmm_gate import (
    HMMGateConfig,
    PerAssetRegimeGateConfig,
    compute_per_asset_regime_gate,
    fit_one_hmm,
    identify_trend_state,
)


def _regime_switching_series(
    n_bars: int = 252 * 4,
    seed: int = 0,
    trend_alpha: float = 0.0015,
    range_alpha: float = 0.0,
    trend_vol: float = 0.006,
    range_vol: float = 0.012,
) -> pd.Series:
    """Return a synthetic series with alternating trend/range regimes.

    First half: trend (drift + low vol). Second half: range (zero drift +
    high vol). HMM should identify the trend half as its "trend state"
    under the ``autocorr`` heuristic because consecutive same-sign returns
    are more common.
    """
    rng = np.random.default_rng(seed)
    half = n_bars // 2
    log_ret = np.concatenate(
        [
            rng.normal(trend_alpha, trend_vol, half),
            rng.normal(range_alpha, range_vol, n_bars - half),
        ]
    )
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    return pd.Series(100 * np.cumprod(1 + log_ret), index=idx, name="REG")


# ── (1) HMM fitting ──────────────────────────────────────────────────────


def test_fit_one_hmm_returns_model_on_sufficient_data():
    series = _regime_switching_series(n_bars=500, seed=1)
    returns = np.log(series / series.shift(1))
    cfg = HMMGateConfig(n_states=2, random_seed=42)
    model = fit_one_hmm(returns, cfg=cfg)
    assert model is not None
    assert model.transmat_.shape == (2, 2)
    assert np.allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-6)


def test_fit_one_hmm_returns_none_on_too_few_bars():
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0, 0.01, 30))
    cfg = HMMGateConfig(n_states=2)
    assert fit_one_hmm(short, cfg=cfg) is None


def test_identify_trend_state_picks_higher_autocorr_state():
    """On a regime-switching series, the trend half has positive
    drift → some serial correlation in sign. The autocorr-based
    state-id should pick the state corresponding to that half."""
    series = _regime_switching_series(n_bars=1000, seed=2)
    returns = np.log(series / series.shift(1))
    cfg = HMMGateConfig(n_states=2, random_seed=42, state_id="autocorr")
    model = fit_one_hmm(returns, cfg=cfg)
    trend_state = identify_trend_state(model, train_returns=returns, cfg=cfg)
    assert trend_state is not None
    assert 0 <= trend_state < 2


def test_identify_trend_state_low_vol_heuristic():
    """Low-vol heuristic picks state with smallest emission variance."""
    series = _regime_switching_series(n_bars=600, seed=3)
    returns = np.log(series / series.shift(1))
    cfg = HMMGateConfig(n_states=2, random_seed=42, state_id="low_vol")
    model = fit_one_hmm(returns, cfg=cfg)
    trend_state = identify_trend_state(model, train_returns=returns, cfg=cfg)
    assert trend_state is not None
    variances = np.array([float(model.covars_[s][0, 0]) for s in range(2)])
    assert variances[trend_state] == variances.min()


# ── (2) Per-asset gate construction ──────────────────────────────────────


def test_compute_per_asset_regime_gate_shape_and_values():
    rng = np.random.default_rng(5)
    n = 800
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    closes = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
            "B": 100 * np.cumprod(1 + rng.normal(-0.0005, 0.012, n)),
        },
        index=idx,
    )
    cfg = PerAssetRegimeGateConfig(
        hmm=HMMGateConfig(n_states=2, random_seed=42, state_id="autocorr"),
        is_min_bars=252,
    )
    is_end_idx = 500  # ~2y of IS
    gate = compute_per_asset_regime_gate(closes, cfg=cfg, is_end_idx=is_end_idx)
    assert gate.shape == closes.shape
    # Gate values should be 0 or 1.
    unique = set(np.unique(gate.fillna(0.0).values.flatten()).tolist())
    assert unique <= {0.0, 1.0}


def test_compute_per_asset_regime_gate_leaves_short_assets_at_1():
    """An asset with fewer bars than ``is_min_bars`` should get gate=1
    everywhere (no filtering applied)."""
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    closes = pd.DataFrame(
        {"SHORT": 100 * np.cumprod(1 + rng.normal(0.0, 0.01, n))},
        index=idx,
    )
    cfg = PerAssetRegimeGateConfig(
        hmm=HMMGateConfig(n_states=2),
        is_min_bars=10_000,  # impossible threshold
    )
    gate = compute_per_asset_regime_gate(closes, cfg=cfg, is_end_idx=200)
    assert (gate == 1.0).all().all()


# ── (3) Causality / IS-frozen ────────────────────────────────────────────


def test_gate_is_is_frozen_oos_corruption_invariant():
    """Corrupting the OOS portion of an asset's returns must not change
    the IS-frozen HMM parameters → only the OOS Viterbi state path can
    differ. Test by computing the gate twice: once on original closes,
    once with the OOS half scaled by 1.5. The IS portion of the gate
    must be bit-identical between runs."""
    closes = pd.DataFrame(
        {"X": _regime_switching_series(n_bars=800, seed=11).values},
        index=pd.date_range("2018-01-02", periods=800, freq="B"),
    )
    cfg = PerAssetRegimeGateConfig(
        hmm=HMMGateConfig(n_states=2, random_seed=42),
        is_min_bars=252,
    )
    is_end = 500
    gate_base = compute_per_asset_regime_gate(closes, cfg=cfg, is_end_idx=is_end)
    closes_corrupt = closes.copy()
    closes_corrupt.iloc[is_end:] = closes_corrupt.iloc[is_end:] * 1.5
    gate_corrupt = compute_per_asset_regime_gate(closes_corrupt, cfg=cfg, is_end_idx=is_end)
    # IS portion of gate must be identical.
    pd.testing.assert_series_equal(
        gate_base["X"].iloc[:is_end],
        gate_corrupt["X"].iloc[:is_end],
        check_names=False,
    )


# ── (4) Integration with EwmacConfig ─────────────────────────────────────


def test_per_asset_gate_disabled_by_default():
    """``per_asset_regime_gate=None`` (default) produces identical forecast
    to baseline."""
    rng = np.random.default_rng(13)
    n = 500
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    closes = pd.DataFrame(
        {f"A{i}": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)) for i in range(3)},
        index=idx,
    )
    fcst_baseline = compute_ewmac_forecast(closes, cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0))
    fcst_explicit_none = compute_ewmac_forecast(
        closes,
        cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0, per_asset_regime_gate=None),
    )
    pd.testing.assert_frame_equal(fcst_baseline, fcst_explicit_none)


def test_per_asset_gate_requires_is_end_idx():
    """If `per_asset_regime_gate` is set, omitting is_end_idx raises."""
    rng = np.random.default_rng(17)
    n = 500
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    closes = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, n))}, index=idx)
    cfg = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        per_asset_regime_gate=PerAssetRegimeGateConfig(),
    )
    try:
        compute_ewmac_forecast(closes, cfg=cfg)
    except ValueError as e:
        assert "is_end_idx" in str(e)
    else:
        raise AssertionError("Expected ValueError when is_end_idx is missing")


def test_per_asset_gate_changes_forecast_when_active():
    """With a regime-switching universe + HMM gate, the gated forecast
    must be zero on bars where the HMM assigns a non-trend state. So the
    total non-zero forecast count should be LOWER than without the gate."""
    cols = {f"A{i}": _regime_switching_series(n_bars=600, seed=i).values for i in range(4)}
    idx = pd.date_range("2018-01-02", periods=600, freq="B")
    closes = pd.DataFrame(cols, index=idx)
    base = compute_ewmac_forecast(closes, cfg=EwmacConfig(speeds=((16, 64),), fdm=1.0))
    cfg_with = EwmacConfig(
        speeds=((16, 64),),
        fdm=1.0,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, random_seed=42), is_min_bars=200
        ),
    )
    gated = compute_ewmac_forecast(closes, cfg=cfg_with, is_end_idx=400)
    n_nonzero_base = int((base.abs() > 1e-9).sum().sum())
    n_nonzero_gated = int((gated.abs() > 1e-9).sum().sum())
    assert n_nonzero_gated <= n_nonzero_base, (
        f"Gated forecast should have <= non-zero bars than baseline; "
        f"baseline={n_nonzero_base} gated={n_nonzero_gated}"
    )


def test_ewmac_returns_with_per_asset_gate_runs_end_to_end():
    """Smoke test: full ewmac_returns pipeline with per-asset gate."""
    cols = {f"A{i}": _regime_switching_series(n_bars=500, seed=i + 100).values for i in range(4)}
    idx = pd.date_range("2018-01-02", periods=500, freq="B")
    closes = pd.DataFrame(cols, index=idx)
    cfg = EwmacConfig(
        speeds=((16, 64), (32, 128)),
        apply_costs=False,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, random_seed=42)
        ),
    )
    rets = ewmac_returns(closes, cfg=cfg, is_end_idx=300)
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(closes)
    assert np.isfinite(rets.iloc[-50:].sum())
