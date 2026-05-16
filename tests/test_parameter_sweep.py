"""Tests for research/exploration/parameter_sweep.py.

The sweep module is exploratory infrastructure (L52). Tests must lock in:
    * Cell-iteration order (deterministic, row-major).
    * Sharpe annualisation honours ``periods_per_year`` (L18 / metrics rule).
    * Plateau detection finds a known synthetic plateau and rejects a
      knife-edge peak (the L43 failure mode this module is designed to
      avoid).
    * IS-only enforcement: the module raises when handed too few bars.
    * The report carries the EXPLORATORY banner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.exploration.parameter_sweep import (
    PlateauCandidate,
    SweepResult,
    detect_plateau,
    format_plateau_report,
    run_parameter_sweep,
)


def _toy_closes(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Two-column random-walk price series (deterministic given seed)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.01, size=(n, 2))
    closes = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=["A", "B"])
    return closes


def _toy_strategy(closes: pd.DataFrame, *, fast: int, slow: int) -> pd.Series:
    """MA crossover toy strategy — long when fast > slow, else 0. Causal."""
    if fast >= slow:
        return pd.Series(0.0, index=closes.index)
    log_ret = np.log(closes / closes.shift(1)).fillna(0.0)
    ma_f = closes.rolling(fast, min_periods=fast).mean()
    ma_s = closes.rolling(slow, min_periods=slow).mean()
    sig = (ma_f > ma_s).astype(float)
    held = sig.shift(1).fillna(0.0)  # L18 shift discipline
    return (held * log_ret).mean(axis=1)


# ── run_parameter_sweep ───────────────────────────────────────────────────


def test_sweep_iteration_is_row_major() -> None:
    """Cells are emitted in row-major order of param_grid keys."""
    closes = _toy_closes(n=300)
    res = run_parameter_sweep(
        closes,
        strategy_fn=_toy_strategy,
        param_grid={"fast": [5, 10], "slow": [20, 40, 60]},
        periods_per_year=252,
    )
    assert res.n_cells == 6
    expected = [
        {"fast": 5, "slow": 20},
        {"fast": 5, "slow": 40},
        {"fast": 5, "slow": 60},
        {"fast": 10, "slow": 20},
        {"fast": 10, "slow": 40},
        {"fast": 10, "slow": 60},
    ]
    assert res.cells == expected


def test_sweep_metadata_records_is_window() -> None:
    closes = _toy_closes(n=300)
    res = run_parameter_sweep(
        closes,
        strategy_fn=_toy_strategy,
        param_grid={"fast": [5], "slow": [20]},
        periods_per_year=252,
        meta={"data_source": "toy"},
    )
    assert res.meta["n_bars"] == 300
    assert res.meta["periods_per_year"] == 252
    assert res.meta["exploratory_only"] is True
    assert res.meta["promotion_eligible"] is False
    assert res.meta["data_source"] == "toy"


def test_sweep_rejects_short_is_window() -> None:
    closes = _toy_closes(n=100)
    with pytest.raises(ValueError, match="need >= 252"):
        run_parameter_sweep(
            closes,
            strategy_fn=_toy_strategy,
            param_grid={"fast": [5], "slow": [20]},
            periods_per_year=252,
        )


def test_sweep_rejects_empty_grid() -> None:
    closes = _toy_closes(n=300)
    with pytest.raises(ValueError, match="param_grid is empty"):
        run_parameter_sweep(
            closes,
            strategy_fn=_toy_strategy,
            param_grid={},
            periods_per_year=252,
        )


def test_sweep_allow_cell_skips_dominated() -> None:
    """``allow_cell`` predicate skips cells; their Sharpe is NaN."""
    closes = _toy_closes(n=300)
    res = run_parameter_sweep(
        closes,
        strategy_fn=_toy_strategy,
        param_grid={"fast": [10, 20], "slow": [5, 30]},
        periods_per_year=252,
        allow_cell=lambda c: c["fast"] < c["slow"],
    )
    # Cells where fast >= slow must be NaN; cells where fast < slow must be finite.
    df = res.to_dataframe()
    skipped = df[df["fast"] >= df["slow"]]
    kept = df[df["fast"] < df["slow"]]
    assert skipped["sharpe"].isna().all()
    assert kept["sharpe"].notna().all()


def test_sweep_strategy_fn_must_return_series() -> None:
    """Wrong return type raises TypeError with a useful message."""
    closes = _toy_closes(n=300)
    with pytest.raises(TypeError, match="must return pd.Series"):
        run_parameter_sweep(
            closes,
            strategy_fn=lambda df, **_: 0.0,  # type: ignore[arg-type, return-value]
            param_grid={"fast": [5], "slow": [20]},
            periods_per_year=252,
        )


def test_sweep_annualisation_honours_periods_per_year() -> None:
    """Same returns at different periods_per_year give sqrt(ratio) Sharpe ratio."""
    closes = _toy_closes(n=300)
    res_daily = run_parameter_sweep(
        closes,
        strategy_fn=_toy_strategy,
        param_grid={"fast": [5], "slow": [20]},
        periods_per_year=252,
    )
    res_hourly = run_parameter_sweep(
        closes,
        strategy_fn=_toy_strategy,
        param_grid={"fast": [5], "slow": [20]},
        periods_per_year=252 * 24,
    )
    # Sharpe is mean/std * sqrt(ppy); the data and strategy are identical,
    # so the ratio must be sqrt(24).
    s_d = res_daily.sharpes[0]
    s_h = res_hourly.sharpes[0]
    if abs(s_d) < 1e-6:
        pytest.skip("toy data produced ~0 Sharpe; ratio test undefined")
    assert s_h / s_d == pytest.approx(np.sqrt(24), rel=1e-9)


# ── to_surface ─────────────────────────────────────────────────────────────


def test_to_surface_requires_2d_grid() -> None:
    closes = _toy_closes(n=300)

    def toy_with_extra(closes: pd.DataFrame, *, fast: int, slow: int, extra: str) -> pd.Series:
        del extra
        return _toy_strategy(closes, fast=fast, slow=slow)

    res = run_parameter_sweep(
        closes,
        strategy_fn=toy_with_extra,
        param_grid={"fast": [5], "slow": [20], "extra": ["x"]},
        periods_per_year=252,
    )
    with pytest.raises(ValueError, match="exactly 2 swept params"):
        res.to_surface()


def test_to_surface_shape_and_values() -> None:
    closes = _toy_closes(n=300)
    res = run_parameter_sweep(
        closes,
        strategy_fn=_toy_strategy,
        param_grid={"fast": [5, 10], "slow": [20, 40]},
        periods_per_year=252,
    )
    surf = res.to_surface()
    assert surf.shape == (2, 2)
    assert list(surf.index) == [5, 10]
    assert list(surf.columns) == [20, 40]
    # Cell (5, 20) in the surface must equal the matching cell in res.cells.
    target_idx = next(i for i, c in enumerate(res.cells) if c["fast"] == 5 and c["slow"] == 20)
    assert surf.loc[5, 20] == pytest.approx(res.sharpes[target_idx])


# ── detect_plateau ────────────────────────────────────────────────────────


def _synthetic_result(surface: np.ndarray, p1_vals: list[int], p2_vals: list[int]) -> SweepResult:
    """Build a SweepResult directly from a 2-D Sharpe surface (no strategy_fn)."""
    cells = []
    sharpes = []
    for v1 in p1_vals:
        for v2 in p2_vals:
            i = p1_vals.index(v1)
            j = p2_vals.index(v2)
            cells.append({"p1": v1, "p2": v2})
            sharpes.append(float(surface[i, j]))
    return SweepResult(
        param_names=("p1", "p2"),
        param_grid={"p1": list(p1_vals), "p2": list(p2_vals)},
        cells=cells,
        sharpes=np.array(sharpes, dtype=float),
        vols=np.full(len(cells), 0.1, dtype=float),
        n_obs=np.full(len(cells), 252, dtype=int),
        meta={"n_bars": 252, "periods_per_year": 252},
    )


def test_plateau_detect_finds_flat_high_region() -> None:
    """3x3 surface with a clear plateau at the centre."""
    # Plateau: centre + 4 axis-neighbours all ~1.0; corners are negative.
    # The centre's neighbourhood (5 cells) should be selected as #1.
    surf = np.array(
        [
            [-0.5, 1.00, -0.5],
            [1.00, 1.05, 1.02],
            [-0.5, 1.00, -0.5],
        ]
    )
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=3)
    assert len(candidates) >= 1
    best = candidates[0]
    assert best.center == {"p1": 20, "p2": 200}
    assert best.rank == 1
    assert best.spread_pct <= 0.30
    assert best.mean_neighbourhood_sharpe > 0.9


def test_plateau_detect_rejects_knife_edge_peak() -> None:
    """A single high peak surrounded by losses must NOT be selected (L43).

    This is the failure mode the hybrid workflow exists to prevent.
    """
    surf = np.array(
        [
            [-0.3, -0.3, -0.3],
            [-0.3, 2.00, -0.3],  # knife-edge peak
            [-0.3, -0.3, -0.3],
        ]
    )
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2)
    # The peak fails the spread gate AND the neighbours fail positivity, so
    # no candidate may be returned.
    assert candidates == []


def test_plateau_detect_rejects_uniformly_negative_region() -> None:
    """A stable but uniformly losing region must NOT be a plateau candidate."""
    surf = np.full((3, 3), -0.25)
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2)
    assert candidates == []


def test_plateau_detect_handles_nan_cells() -> None:
    """NaN cells (from allow_cell skip) are ignored; valid neighbours still count."""
    surf = np.array(
        [
            [np.nan, 1.0, np.nan],
            [1.0, 1.05, 1.0],
            [np.nan, 1.0, np.nan],
        ]
    )
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2)
    assert len(candidates) >= 1
    assert candidates[0].center == {"p1": 20, "p2": 200}


def test_plateau_detect_returns_empty_on_empty_grid() -> None:
    res = SweepResult(
        param_names=("p1",),
        param_grid={"p1": []},
        cells=[],
        sharpes=np.array([]),
        vols=np.array([]),
        n_obs=np.array([], dtype=int),
        meta={},
    )
    assert detect_plateau(res) == []


# ── format_plateau_report ─────────────────────────────────────────────────


def test_report_carries_exploratory_banner() -> None:
    """The L52 banner MUST appear so the report can't be cited as a deployment gate."""
    surf = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.05, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    cands = detect_plateau(res)
    report = format_plateau_report(res, cands, audit_label="DEMO")
    assert "EXPLORATORY ONLY" in report
    assert "NOT a deployment gate" in report
    assert "PRIOR" in report


def test_report_handles_no_candidates() -> None:
    surf = np.full((3, 3), -0.25)
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    cands = detect_plateau(res)
    report = format_plateau_report(res, cands)
    assert "No plateau candidates" in report


def test_report_lists_candidates_in_rank_order() -> None:
    """Candidates appear under headers #1, #2, ..."""
    surf = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.05, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    res = _synthetic_result(surf, p1_vals=[10, 20, 30], p2_vals=[100, 200, 300])
    cands = detect_plateau(res, top_k=3)
    if len(cands) >= 2:
        report = format_plateau_report(res, cands)
        i1 = report.find("#1 — center")
        i2 = report.find("#2 — center")
        assert 0 <= i1 < i2


# ── PlateauCandidate ──────────────────────────────────────────────────────


def test_plateau_candidate_is_frozen() -> None:
    """Immutable so it can't be mutated after being attached to a report."""
    c = PlateauCandidate(
        center={"p1": 1},
        neighbour_cells=[],
        center_sharpe=0.5,
        neighbour_sharpes=[],
        mean_neighbourhood_sharpe=0.5,
        spread_pct=0.0,
        rank=1,
    )
    with pytest.raises((AttributeError, Exception)):
        c.rank = 2  # type: ignore[misc]
