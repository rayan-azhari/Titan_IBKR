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
