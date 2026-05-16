"""Tests for the GEM J5 hybrid re-audit pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.exploration.sweep_gem_hybrid import gem_strategy_fn
from research.gem.gem_strategy import gem_returns
from research.gem.run_gem_j5_reaudit import _make_cfg, buy_and_hold_60_40


def _toy_gem_universe(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    spy_ret = rng.normal(0.0006, 0.012, size=n)
    efa_ret = rng.normal(0.0003, 0.013, size=n)
    ief_ret = rng.normal(0.0001, 0.004, size=n)
    return pd.DataFrame(
        {
            "SPY": 100.0 * np.exp(np.cumsum(spy_ret)),
            "EFA": 100.0 * np.exp(np.cumsum(efa_ret)),
            "IEF": 100.0 * np.exp(np.cumsum(ief_ret)),
        },
        index=idx,
    )


def test_make_cfg_matches_live_knobs() -> None:
    """`_make_cfg` produces a GemConfig with live-frozen knobs + swept axes."""
    cfg = _make_cfg(halflife=20, vol_target=0.05)
    assert cfg.vol_estimator_halflife == 20
    assert cfg.ann_vol_target == 0.05
    assert cfg.vol_estimator_kind == "ewma"
    assert cfg.lookback_blend == (3, 6, 12)
    assert cfg.max_leverage == 2.0
    assert cfg.defensive_switch is True


def test_buy_and_hold_60_40_causal_and_long_only() -> None:
    """60/40 benchmark returns a Series, causal, with reasonable magnitude."""
    closes = _toy_gem_universe(n=600)
    rets = buy_and_hold_60_40(closes)
    assert isinstance(rets, pd.Series)
    # First bar must be exactly 0 (no prior position).
    assert abs(rets.iloc[0]) < 1e-12
    # The 60/40 benchmark correlates strongly with the SPY+IEF combo.
    spy_ret = np.log(closes["SPY"] / closes["SPY"].shift(1)).fillna(0.0)
    ief_ret = np.log(closes["IEF"] / closes["IEF"].shift(1)).fillna(0.0)
    composite = (0.6 * spy_ret + 0.4 * ief_ret).shift(0)  # un-shifted
    common = rets.iloc[1:].dropna().index.intersection(composite.iloc[1:].dropna().index)
    rets_a = rets.reindex(common).iloc[1:]
    comp_a = composite.reindex(common).iloc[1:]
    corr = float(rets_a.corr(comp_a))
    assert corr > 0.95


def test_sweep_audit_parity_gem() -> None:
    """SWEEP ↔ AUDIT parity for the J5 canonical cell.

    `sweep_gem_hybrid.gem_strategy_fn` and the audit `_make_cfg` + `gem_returns`
    must produce identical per-bar returns for identical inputs.
    """
    closes = _toy_gem_universe(n=1200, seed=11)
    # Sweep invocation.
    sweep_ret = gem_strategy_fn(closes, vol_estimator_halflife=20, ann_vol_target=0.05)
    # Audit invocation — needs identical cost params to the sweep.
    cfg = _make_cfg(halflife=20, vol_target=0.05)
    audit_ret = gem_returns(
        closes,
        cfg=cfg,
        cost_bps_per_turnover=6.0,
        cost_fixed_usd_per_fill=1.0,
        notional_usd=30_000.0,
        execution_mode="etf",
        rebalance_threshold=0.05,
    )
    common = sweep_ret.dropna().index.intersection(audit_ret.dropna().index)
    assert len(common) > 50
    diff = (sweep_ret.reindex(common) - audit_ret.reindex(common)).abs()
    max_diff = float(diff.max())
    assert max_diff < 1e-12, f"sweep vs audit divergence: max |diff| = {max_diff:.2e}"


def test_j5_canonical_cell_is_finite() -> None:
    """The J5 canonical config produces finite returns on toy data."""
    closes = _toy_gem_universe(n=1200)
    cfg = _make_cfg(halflife=20, vol_target=0.05)
    rets = gem_returns(closes, cfg=cfg)
    assert rets.dropna().shape[0] > 0
    assert np.isfinite(rets.dropna().to_numpy()).all()


def test_j5_canonical_differs_from_j4_live() -> None:
    """Sanity check: the J5 canonical and J4 live config produce different returns
    (otherwise the sweep finding would be vacuous)."""
    closes = _toy_gem_universe(n=1200)
    j5 = gem_returns(closes, cfg=_make_cfg(halflife=20, vol_target=0.05))
    j4 = gem_returns(closes, cfg=_make_cfg(halflife=40, vol_target=0.10))
    diff = (j5.dropna() - j4.dropna()).abs()
    # On 1200 toy bars, the two strategies must produce materially different daily
    # returns (vol_target differs by 2x).
    assert float(diff.max()) > 1e-4
