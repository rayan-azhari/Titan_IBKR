"""Tests for titan/research/framework/amortised_mc.py.

Speed-up #2 invariants (L52):

    * ``prefit`` is called EXACTLY ONCE — never per MC path.
    * ``prefit`` sees ONLY the IS portion of the original data (no
      synthetic/bootstrap leakage into the model fit).
    * The resulting McResult is shape-compatible with the standard
      ``run_block_mc`` output so the decision matrix consumes it
      identically.
    * Per-path ``infer`` is invoked with both the synthetic DataFrame
      and the cached fitted state.
    * ``is_end_idx`` validation rejects out-of-range indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.amortised_mc import run_block_mc_amortised
from titan.research.framework.typology import McConfig


def _toy_closes(n: int = 600, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.01, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name="close")


def _mc_cfg() -> McConfig:
    return McConfig(
        n_paths=30,
        block_size_bars=20,
        max_dd_threshold_pct=0.30,
        max_dd_pass_prob=0.20,
        bootstrap_method="block",
    )


# ── core invariants ───────────────────────────────────────────────────────


def test_prefit_invoked_exactly_once() -> None:
    """The whole point of speed-up #2 — prefit must NOT run per path."""
    closes = _toy_closes(n=600)
    call_count = {"prefit": 0, "infer": 0}

    def prefit(is_df: pd.DataFrame) -> dict:
        call_count["prefit"] += 1
        return {"mean": float(is_df["close"].pct_change().mean())}

    def infer(df: pd.DataFrame, state: dict) -> pd.Series:
        call_count["infer"] += 1
        # Toy strategy: long when synthetic return is positive on average.
        sign = 1.0 if state["mean"] > 0 else -1.0
        return df["close"].pct_change().fillna(0.0) * sign

    cfg = _mc_cfg()
    res = run_block_mc_amortised(
        closes,
        cfg,
        prefit=prefit,
        infer=infer,
        is_end_idx=400,
        periods_per_year=252,
    )

    assert call_count["prefit"] == 1, "prefit must run exactly once"
    assert call_count["infer"] == cfg.n_paths
    assert res.n_paths_completed > 0


def test_prefit_sees_only_is_window() -> None:
    """prefit must NOT receive any synthetic data — IS-frozen rule."""
    closes = _toy_closes(n=600)
    is_end = 400
    received_indices: list[pd.DatetimeIndex] = []

    def prefit(is_df: pd.DataFrame) -> int:
        received_indices.append(is_df.index)
        return 1

    def infer(df: pd.DataFrame, _state: int) -> pd.Series:
        return df["close"].pct_change().fillna(0.0)

    run_block_mc_amortised(
        closes,
        _mc_cfg(),
        prefit=prefit,
        infer=infer,
        is_end_idx=is_end,
        periods_per_year=252,
    )

    # prefit must have been called exactly once with the IS slice.
    assert len(received_indices) == 1
    idx = received_indices[0]
    expected_last = closes.index[is_end - 1]
    assert idx[-1] == expected_last
    assert len(idx) == is_end


def test_infer_receives_fitted_state_unchanged() -> None:
    """The opaque state object must reach each ``infer`` call by reference."""
    closes = _toy_closes(n=600)
    sentinel = object()
    seen_states: list[object] = []

    def prefit(_is_df: pd.DataFrame) -> object:
        return sentinel

    def infer(df: pd.DataFrame, state: object) -> pd.Series:
        seen_states.append(state)
        return df["close"].pct_change().fillna(0.0)

    run_block_mc_amortised(
        closes,
        _mc_cfg(),
        prefit=prefit,
        infer=infer,
        is_end_idx=400,
        periods_per_year=252,
    )
    assert len(seen_states) > 0
    assert all(s is sentinel for s in seen_states)


def test_result_shape_matches_run_block_mc() -> None:
    """Amortised MC returns the SAME ``McResult`` shape — decision matrix swap-in."""
    closes = _toy_closes(n=600)

    def prefit(_is_df: pd.DataFrame) -> None:
        return None

    def infer(df: pd.DataFrame, _state: None) -> pd.Series:
        return df["close"].pct_change().fillna(0.0)

    cfg = _mc_cfg()
    res = run_block_mc_amortised(
        closes, cfg, prefit=prefit, infer=infer, is_end_idx=400, periods_per_year=252
    )
    # Mirror the run_block_mc McResult schema check.
    assert hasattr(res, "median_sharpe")
    assert hasattr(res, "p5_sharpe")
    assert hasattr(res, "p95_sharpe")
    assert hasattr(res, "median_maxdd")
    assert hasattr(res, "p_maxdd_gt_threshold")
    assert hasattr(res, "passes")
    assert res.method == cfg.bootstrap_method
    assert res.block_size == cfg.block_size_bars


def test_failing_infer_is_skipped_not_raised() -> None:
    """A strategy that raises on a particular synthetic path is skipped, not propagated."""
    closes = _toy_closes(n=600)

    def prefit(_is_df: pd.DataFrame) -> None:
        return None

    call_count = {"n": 0}

    def infer(df: pd.DataFrame, _state: None) -> pd.Series:
        call_count["n"] += 1
        if call_count["n"] % 3 == 0:
            raise RuntimeError("simulated strategy failure")
        return df["close"].pct_change().fillna(0.0)

    cfg = _mc_cfg()
    res = run_block_mc_amortised(
        closes, cfg, prefit=prefit, infer=infer, is_end_idx=400, periods_per_year=252
    )
    # Some paths failed but the run still completed; n_paths_completed reflects only success.
    assert 0 < res.n_paths_completed < cfg.n_paths


# ── input validation ─────────────────────────────────────────────────────


def test_is_end_idx_validation() -> None:
    closes = _toy_closes(n=600)

    def prefit(_is_df: pd.DataFrame) -> None:
        return None

    def infer(df: pd.DataFrame, _state: None) -> pd.Series:
        return df["close"].pct_change().fillna(0.0)

    with pytest.raises(ValueError, match="is_end_idx"):
        run_block_mc_amortised(
            closes, _mc_cfg(), prefit=prefit, infer=infer, is_end_idx=0, periods_per_year=252
        )
    with pytest.raises(ValueError, match="is_end_idx"):
        run_block_mc_amortised(
            closes,
            _mc_cfg(),
            prefit=prefit,
            infer=infer,
            is_end_idx=10_000,
            periods_per_year=252,
        )


def test_extra_series_alignment_for_prefit() -> None:
    """When extras supplied, prefit receives them aligned on the IS window."""
    closes = _toy_closes(n=600)
    extra = closes.pct_change().fillna(0.0).rename("aux")  # noqa: F841 (kept in extras dict below)
    received: list[pd.DataFrame] = []

    def prefit(is_df: pd.DataFrame) -> None:
        received.append(is_df)
        return None

    def infer(df: pd.DataFrame, _state: None) -> pd.Series:
        return df["close"].pct_change().fillna(0.0)

    run_block_mc_amortised(
        closes,
        _mc_cfg(),
        prefit=prefit,
        infer=infer,
        is_end_idx=400,
        periods_per_year=252,
        extra_series={"aux": closes.pct_change().fillna(0.0)},
    )
    assert len(received) == 1
    df = received[0]
    assert "close" in df.columns
    assert "aux" in df.columns
    assert len(df) <= 400  # constrained to IS window
