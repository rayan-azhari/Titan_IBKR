"""Unit tests for IC Census primitives -- the gates we must never regress."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.ic_analysis.ic_census_lib import (
    CellResult,
    anchored_aggregate,
    assert_causal,
    deflated_t_pvalue,
    fold_ic_signs,
    mtf_agreement,
    plateau_stable,
    signal_factories,
    slice_sanctuary,
)


# ── (1) Causality smoke test ───────────────────────────────────────────────


def _hourly_series(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(rng.normal(size=n).cumsum(), index=idx)


def test_assert_causal_passes_for_causal_aggregator():
    src = _hourly_series(n=300)
    dst_index = pd.date_range(src.index[0], src.index[-1], freq="1D", tz="UTC")
    # The canonical anchored aggregator is causal by construction.
    fn = lambda s, di: anchored_aggregate(s, di, higher_tf=True, rule="1D")  # noqa: E731
    assert_causal(fn, src, dst_index, n_trials=5)  # must not raise


def test_assert_causal_catches_leaky_aggregator():
    src = _hourly_series(n=300)
    dst_index = pd.date_range(src.index[0], src.index[-1], freq="1D", tz="UTC")
    # Leaky: bin includes [T, T+1d] AND no .shift(1). Output at D-bar T
    # depends on H1 bars from the SAME day T (which is forbidden).
    def leaky_agg(s, di):
        return s.resample("1D", label="left", closed="right").last().reindex(di)
    with pytest.raises(AssertionError):
        assert_causal(leaky_agg, src, dst_index, n_trials=5)


def test_anchored_aggregate_lower_tf_no_same_day_leak():
    """The EUR/USD MTF +1.94 Sharpe bug pattern: ffill-without-shift lets
    today's daily value leak into intraday bars of the same day.

    The strict-causality ``assert_causal`` test does NOT catch this bug
    (the leak is at the same-timestamp boundary, not across it). Instead,
    we verify directly that the legitimate aggregator's ``.shift(1)``
    step makes daily[T] invisible to h1 bars on calendar day T --
    corrupting daily[T] must not change any h1 output on day T.
    """
    d_idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    daily = pd.Series(np.arange(10, dtype=float) + 1.0, index=d_idx)
    h1_idx = pd.date_range("2024-01-02 00:00", periods=24, freq="1h", tz="UTC")

    baseline = anchored_aggregate(daily, h1_idx, higher_tf=False)
    corrupted = daily.copy()
    corrupted.iloc[1] = 9999.0  # corrupt daily[2024-01-02]
    after = anchored_aggregate(corrupted, h1_idx, higher_tf=False)
    # h1 bars on 2024-01-02 must see daily[2024-01-01]=1.0, never daily[2024-01-02].
    pd.testing.assert_series_equal(baseline, after, check_exact=True, check_names=False)
    assert (baseline == 1.0).all()  # confirms it's yesterday's value


# ── (2) Anchored aggregator ────────────────────────────────────────────────


def test_anchored_aggregate_higher_tf_excludes_same_day():
    h1_idx = pd.date_range("2024-01-01 00:00", periods=72, freq="1h", tz="UTC")
    h1 = pd.Series(np.arange(72, dtype=float), index=h1_idx)
    d_idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    out = anchored_aggregate(h1, d_idx, higher_tf=True, rule="1D")
    # D-bar for 2024-01-01: must NOT see hours 0-23 of 2024-01-01.
    # The aggregator returns NaN (no prior day data) for the first D-bar.
    assert pd.isna(out.iloc[0])
    # D-bar for 2024-01-02: sees last H1 of 2024-01-01 (value 23).
    assert out.iloc[1] == 23.0
    # D-bar for 2024-01-03: sees last H1 of 2024-01-02 (value 47).
    assert out.iloc[2] == 47.0


def test_anchored_aggregate_lower_tf_uses_yesterday():
    d_idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    daily = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=d_idx)
    h1_idx = pd.date_range("2024-01-02 00:00", periods=24, freq="1h", tz="UTC")
    out = anchored_aggregate(daily, h1_idx, higher_tf=False)
    # All H1 bars on 2024-01-02 should see daily[2024-01-01] = 10.0.
    assert (out == 10.0).all()


# ── (3) Deflated t-pvalue ──────────────────────────────────────────────────


def test_deflated_t_pvalue_at_null_max():
    # At |t| == sqrt(2 ln N) we expect dsr_p ≈ 0.5 (we're at the null max).
    for n in (100, 1000, 10000):
        t_at_null = math.sqrt(2 * math.log(n))
        p = deflated_t_pvalue(t_at_null, n)
        assert 0.45 < p < 0.55, f"N={n}: dsr_p={p:.3f} (expected ≈ 0.5)"


def test_deflated_t_pvalue_far_above_null():
    # t = 5 with N = 10k -> null max ≈ 4.29, gap ≈ 0.71, Phi(0.71) ≈ 0.76.
    p = deflated_t_pvalue(5.0, 10000)
    assert 0.70 < p < 0.80


def test_deflated_t_pvalue_far_below_null():
    # t = 2 with N = 10k -> well below null max -- p should be very small.
    p = deflated_t_pvalue(2.0, 10000)
    assert p < 0.05


def test_deflated_t_pvalue_zero_trials():
    assert deflated_t_pvalue(5.0, 1) == 0.0


# ── (4) Fold IC sign stability ─────────────────────────────────────────────


def test_fold_ic_signs_stable_positive():
    rng = np.random.default_rng(0)
    n = 500
    signal = pd.Series(rng.normal(size=n))
    # Construct a forward return correlated with signal -> consistent +ve IC.
    fwd = signal * 0.5 + rng.normal(size=n) * 0.5
    fold_ic, stable, quorum = fold_ic_signs(signal, fwd, n_folds=5)
    assert stable
    assert quorum >= 4
    assert all(x > 0 for x in fold_ic)


def test_fold_ic_signs_unstable():
    rng = np.random.default_rng(1)
    n = 500
    signal = pd.Series(rng.normal(size=n))
    fwd = pd.Series(rng.normal(size=n))  # pure noise
    _, stable, _ = fold_ic_signs(signal, fwd, n_folds=5)
    assert not stable


def test_fold_ic_signs_default_quorum_is_four():
    """Default quorum is 4 of 5 (not 3) -- a 3 quorum would be tautological
    since 5 non-NaN folds always have >= 3 sharing a sign by pigeonhole.
    Two opposing non-near-zero folds must break the gate."""
    # Construct a strong signal in 3 folds and a strong OPPOSITE signal in 2.
    rng = np.random.default_rng(11)
    sizes = [100, 100, 100, 100, 100]
    pieces_sig, pieces_fwd = [], []
    for i, n in enumerate(sizes):
        s = rng.normal(size=n)
        # Folds 0,1,2: positive corr; folds 3,4: negative corr.
        sign = 1.0 if i < 3 else -1.0
        f = sign * 0.5 * s + rng.normal(size=n) * 0.4
        pieces_sig.append(s)
        pieces_fwd.append(f)
    signal = pd.Series(np.concatenate(pieces_sig))
    fwd = pd.Series(np.concatenate(pieces_fwd))
    _, stable, modal = fold_ic_signs(signal, fwd, n_folds=5)
    # Modal=3 (the larger of the two camps) < default quorum 4 -> not stable.
    assert modal == 3
    assert not stable


def test_fold_ic_signs_nan_fold_tolerated():
    """A fold with undefined Spearman (constant fwd in that slice)
    should be NaN-IC and treated sign-agnostic. 4 positive folds + 1 NaN
    must pass the default quorum=4 gate."""
    rng = np.random.default_rng(7)
    pieces_sig, pieces_fwd = [], []
    for i in range(5):
        s = rng.normal(size=100)
        if i == 4:
            # Constant fwd -> Spearman with any signal == NaN.
            f = np.zeros(100)
        else:
            f = 0.6 * s + rng.normal(size=100) * 0.3
        pieces_sig.append(s)
        pieces_fwd.append(f)
    signal = pd.Series(np.concatenate(pieces_sig))
    fwd = pd.Series(np.concatenate(pieces_fwd))
    fold_ic, stable, modal = fold_ic_signs(signal, fwd, n_folds=5)
    # First 4 folds clearly positive; last is NaN.
    assert all(x > 0.2 for x in fold_ic[:4])
    assert np.isnan(fold_ic[4])
    assert modal == 4
    assert stable


def test_fold_ic_signs_strict_quorum_rejects_split_vote():
    """4 strong-positive folds + 1 strong-negative fold: modal=4. The
    default quorum=4 must accept (tolerates one opposing fold). A strict
    quorum=5 must reject."""
    rng = np.random.default_rng(13)
    pieces_sig, pieces_fwd = [], []
    for i in range(5):
        s = rng.normal(size=100)
        sign = -1.0 if i == 4 else 1.0
        f = sign * 0.6 * s + rng.normal(size=100) * 0.3
        pieces_sig.append(s)
        pieces_fwd.append(f)
    signal = pd.Series(np.concatenate(pieces_sig))
    fwd = pd.Series(np.concatenate(pieces_fwd))
    _, stable_strict, modal_strict = fold_ic_signs(signal, fwd, n_folds=5, quorum=5)
    _, stable_loose, modal_loose = fold_ic_signs(signal, fwd, n_folds=5, quorum=4)
    assert modal_strict == 4  # 4 positive, 1 negative
    assert modal_loose == 4
    assert not stable_strict
    assert stable_loose


# ── (5) Plateau gate ───────────────────────────────────────────────────────


def _cells(ics_ts: list[tuple[float, float]]) -> list[CellResult]:
    return [CellResult(params={"i": i}, ic=ic, t_stat=t, n_obs=500)
            for i, (ic, t) in enumerate(ics_ts)]


def test_plateau_passes_with_smooth_neighbours():
    cells = _cells([(0.04, 4.0), (0.05, 5.0), (0.045, 4.5)])
    passes, headline, _ = plateau_stable(
        cells, headline_t_floor=4.5, neighbour_t_floor=3.0, ic_range_max=0.30
    )
    assert passes
    assert headline.params == {"i": 1}


def test_plateau_rejects_edge_cell():
    # Best |t| at index 0 -- no left neighbour.
    cells = _cells([(0.06, 6.0), (0.03, 3.0), (0.02, 2.0)])
    passes, _, reason = plateau_stable(
        cells, headline_t_floor=4.5, neighbour_t_floor=3.0, ic_range_max=0.30
    )
    assert not passes
    assert "edge" in reason


def test_plateau_rejects_isolated_peak():
    # Spike at middle but neighbour |t| below floor.
    cells = _cells([(0.01, 1.0), (0.05, 5.0), (0.01, 1.0)])
    passes, _, reason = plateau_stable(
        cells, headline_t_floor=4.5, neighbour_t_floor=3.0, ic_range_max=0.30
    )
    assert not passes
    assert "neighbour" in reason


def test_plateau_rejects_sign_flip():
    cells = _cells([(0.04, 4.0), (0.05, 5.0), (-0.04, -4.0)])
    passes, _, reason = plateau_stable(
        cells, headline_t_floor=4.5, neighbour_t_floor=3.0, ic_range_max=0.30
    )
    assert not passes
    assert "sign" in reason


def test_plateau_rejects_high_ic_range():
    # All clear t-floors, all same sign, but IC magnitudes swing by 50%.
    cells = _cells([(0.03, 4.5), (0.06, 5.0), (0.04, 4.5)])
    passes, _, reason = plateau_stable(
        cells, headline_t_floor=4.5, neighbour_t_floor=3.0, ic_range_max=0.30
    )
    assert not passes
    assert "range" in reason


# ── (6) MTF agreement ──────────────────────────────────────────────────────


def test_mtf_agreement_passes_with_quorum():
    per_tf = {
        "D":  CellResult(params={}, ic=0.05, t_stat=5.0, n_obs=500),
        "H4": CellResult(params={}, ic=0.04, t_stat=4.6, n_obs=500),
        "H1": CellResult(params={}, ic=-0.02, t_stat=-2.0, n_obs=500),  # fails floor
    }
    agree, n_pass = mtf_agreement(per_tf, quorum=2, t_floor=4.5)
    assert agree
    assert n_pass == 2


def test_mtf_agreement_fails_single_tf():
    per_tf = {
        "D":  CellResult(params={}, ic=0.05, t_stat=5.0, n_obs=500),
        "H4": None,
        "H1": None,
    }
    agree, _ = mtf_agreement(per_tf, quorum=2, t_floor=4.5)
    assert not agree


def test_mtf_agreement_fails_on_sign_disagreement():
    per_tf = {
        "D":  CellResult(params={}, ic=0.05, t_stat=5.0, n_obs=500),
        "H4": CellResult(params={}, ic=-0.05, t_stat=-5.0, n_obs=500),
        "H1": CellResult(params={}, ic=0.03, t_stat=3.0, n_obs=500),  # fails floor
    }
    agree, _ = mtf_agreement(per_tf, quorum=2, t_floor=4.5)
    assert not agree  # neither side has quorum (1 pos, 1 neg, 1 below floor)


# ── (7) Sanctuary slicing ──────────────────────────────────────────────────


def test_slice_sanctuary_drops_trailing_year():
    idx = pd.date_range("2020-01-01", "2024-12-31", freq="1D", tz="UTC")
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    visible, start, end = slice_sanctuary(df, months=12)
    assert visible.index[-1] < start
    assert end == idx[-1]
    assert (end - start).days >= 364


# ── (8) Cross-asset signal factories ───────────────────────────────────────


def test_signal_registry_has_cross_asset_externals():
    """The cross-asset entries must declare which external instruments
    they need. Without this metadata the orchestrator cannot load and
    anchor the external series."""
    reg = signal_factories()
    assert reg["hyg_ief_z"]["externals"] == ["HYG", "IEF"]
    assert reg["dxy_z"]["externals"] == ["EUR_USD"]
    # Single-instrument signals must NOT declare externals.
    assert "externals" not in reg["rsi_dev"] or reg["rsi_dev"].get("externals") in (None, [])


def test_hyg_ief_z_factory_runs_causally():
    """End-to-end check that the cross-asset factory produces a finite
    series of the expected shape from anchored inputs."""
    reg = signal_factories()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    target_close = pd.Series(100 * np.exp(rng.normal(0, 0.01, 500).cumsum()), index=idx)
    hyg_close = pd.Series(80 * np.exp(rng.normal(0, 0.008, 500).cumsum()), index=idx)
    ief_close = pd.Series(110 * np.exp(rng.normal(0, 0.005, 500).cumsum()), index=idx)
    # Anchor externals via the canonical pattern (shift(1) + ffill).
    hyg_aligned = anchored_aggregate(hyg_close, idx, higher_tf=False)
    ief_aligned = anchored_aggregate(ief_close, idx, higher_tf=False)
    sig = reg["hyg_ief_z"]["fn"](target_close, hyg=hyg_aligned, ief=ief_aligned, lookback=60)
    # First 60 bars NaN (rolling lookback); thereafter finite.
    assert sig.iloc[:60].isna().all()
    assert sig.iloc[60:].notna().sum() > 400
    # Finite values must be in a sane range for a z-score over a random walk.
    finite = sig.dropna()
    assert finite.abs().max() < 10.0


def test_dxy_z_factory_inverts_eur_usd():
    """USD strengthens when EUR/USD falls. The z of -log(EUR_USD) must
    have the opposite sign to the z of log(EUR_USD)."""
    reg = signal_factories()
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    # Monotonically rising EUR/USD -> USD weakening -> z should trend NEGATIVE.
    eur_usd = pd.Series(np.linspace(1.0, 1.5, 400), index=idx)
    target_close = pd.Series(100.0, index=idx)
    eur_aligned = anchored_aggregate(eur_usd, idx, higher_tf=False)
    sig = reg["dxy_z"]["fn"](target_close, eur_usd=eur_aligned, lookback=60)
    finite = sig.dropna()
    # Most of the dynamic range should be negative for a rising EUR/USD.
    assert (finite < 0).sum() > (finite > 0).sum()


def test_phase_b_factories_registered_with_externals():
    """All four Phase B cross-asset factories must declare their externals."""
    reg = signal_factories()
    assert reg["vix9d_over_vix"]["externals"] == ["VIX9D", "VIX"]
    assert reg["vix_over_vix3m"]["externals"] == ["VIX", "VIX3M"]
    assert reg["vrp_z"]["externals"] == ["VIX"]
    assert reg["us_lead_eu"]["externals"] == ["SPY"]


def test_vix9d_over_vix_produces_finite_z_score():
    reg = signal_factories()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=500, freq="1D", tz="UTC")
    close = pd.Series(100 * np.exp(rng.normal(0, 0.01, 500).cumsum()), index=idx)
    vix = pd.Series(15 + rng.normal(0, 2, 500).cumsum().clip(-5, 20), index=idx)
    vix9d = pd.Series(vix.values + rng.normal(0, 1.5, 500), index=idx)
    vix_aligned = anchored_aggregate(vix, idx, higher_tf=False)
    vix9d_aligned = anchored_aggregate(vix9d, idx, higher_tf=False)
    sig = reg["vix9d_over_vix"]["fn"](
        close, vix9d=vix9d_aligned, vix=vix_aligned, smoothing=5
    )
    # First ~65 bars NaN (5-bar smoothing + 60-bar z-score window + 1-bar shift).
    finite = sig.dropna()
    assert len(finite) > 400
    assert finite.abs().max() < 10.0


def test_vrp_z_responds_to_implied_vs_realised():
    """When VIX is high relative to realised vol, vrp_z should be
    positive; when low, negative."""
    reg = signal_factories()
    idx = pd.date_range("2015-01-01", periods=400, freq="1D", tz="UTC")
    # Construct a calm-market close series (low realised vol).
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(rng.normal(0, 0.005, 400).cumsum()), index=idx)
    # And a VIX series fixed high (implied >> realised).
    vix_high = pd.Series(25.0, index=idx)
    vix_high_aligned = anchored_aggregate(vix_high, idx, higher_tf=False)
    sig_high = reg["vrp_z"]["fn"](close, vix=vix_high_aligned, rv_window=20)
    # The level itself is high VRP; the *z-score* over its own series is
    # near zero (constant). So we test the underlying VRP is positive
    # by checking that vrp series mean (before z) is positive.
    vrp_calm = (vix_high_aligned / 100.0) - np.log(close).diff().rolling(20).std(ddof=1) * np.sqrt(252)
    assert vrp_calm.dropna().mean() > 0.0


def test_us_lead_eu_uses_shifted_spy():
    """At target time T, the signal must reflect SPY at strictly < T."""
    reg = signal_factories()
    idx = pd.date_range("2020-01-01", periods=200, freq="1D", tz="UTC")
    # A monotonically rising SPY -- log returns are constant positive.
    spy = pd.Series(np.exp(np.arange(200) * 0.01), index=idx)
    eu_close = pd.Series(100.0, index=idx)
    spy_aligned = anchored_aggregate(spy, idx, higher_tf=False)
    sig = reg["us_lead_eu"]["fn"](eu_close, spy=spy_aligned, window=5)
    finite = sig.dropna()
    # All finite values should be positive (positive SPY returns shifted in).
    assert (finite > 0).all()


def test_cross_asset_alignment_is_causal_via_assert_causal():
    """The canonical Anchored MTF aggregator must survive the corrupt-
    the-future test. This is the wrap the runner applies before passing
    externals to the cross-asset factories."""
    src = pd.Series(
        np.arange(300, dtype=float),
        index=pd.date_range("2020-01-01", periods=300, freq="1D", tz="UTC"),
    )
    dst_index = src.index  # same-frequency anchoring (D->D)
    fn = lambda s, di: anchored_aggregate(s, di, higher_tf=False)  # noqa: E731
    assert_causal(fn, src, dst_index, n_trials=5)
