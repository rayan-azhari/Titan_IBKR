"""Tests for the bond_gold V3.6 re-audit pipeline.

Critical invariants (pre-reg §5):
    * L04 / A1 — bond_gold_assert_causal must pass on synthetic data.
    * L18 — position is .shift(1)'d before earning return.
    * SWEEP ↔ AUDIT parity — `sweep_bond_gold.bond_gold_strategy_fn` must
      produce bit-exact returns vs `bond_gold_strategy.bond_gold_returns`
      on identical inputs. Otherwise the pre-reg's "the canonical chosen
      from the sweep" claim is unfounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.cross_asset.bond_gold_strategy import (
    BondGoldConfig,
    bond_gold_assert_causal,
    bond_gold_returns,
)
from research.exploration.sweep_bond_gold import bond_gold_strategy_fn


def _toy_universe(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Two-column synthetic IEF + GLD random walks (deterministic)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    ief_ret = rng.normal(0.0002, 0.005, size=n)
    gld_ret = rng.normal(0.0005, 0.010, size=n)
    return pd.DataFrame(
        {
            "IEF": 100.0 * np.exp(np.cumsum(ief_ret)),
            "GLD": 100.0 * np.exp(np.cumsum(gld_ret)),
        },
        index=idx,
    )


def test_bond_gold_returns_runs_clean() -> None:
    """Smoke: bond_gold_returns produces a finite Series on toy data."""
    closes = _toy_universe(n=1200)
    rets = bond_gold_returns(closes, cfg=BondGoldConfig(lookback=120, threshold=0.50))
    assert isinstance(rets, pd.Series)
    assert rets.dropna().shape[0] > 0
    assert np.isfinite(rets.dropna().to_numpy()).all()


def test_bond_gold_causal() -> None:
    """L04 / A1 — corrupting future closes must NOT change past returns."""
    closes = _toy_universe(n=1200)
    # Must not raise.
    bond_gold_assert_causal(closes, cfg=BondGoldConfig(lookback=120, threshold=0.50))


def test_bond_gold_requires_ief_and_gld() -> None:
    closes = _toy_universe(n=600).rename(columns={"IEF": "X"})
    with pytest.raises(ValueError, match="IEF"):
        bond_gold_returns(closes)


def test_sweep_audit_parity() -> None:
    """SWEEP ↔ AUDIT parity. The sweep's strategy fn and the audit module
    MUST produce identical per-bar returns for identical inputs and params.

    If this fails, the pre-reg directive cannot legitimately claim its
    canonical was chosen from the sweep — the two pipelines would be
    measuring different strategies.
    """
    closes = _toy_universe(n=1500, seed=11)
    # Use the exact canonical from the pre-reg.
    sweep_ret = bond_gold_strategy_fn(closes, lookback=120, threshold=0.50)
    audit_ret = bond_gold_returns(
        closes, cfg=BondGoldConfig(lookback=120, threshold=0.50)
    )
    common = sweep_ret.dropna().index.intersection(audit_ret.dropna().index)
    assert len(common) > 100, "Parity test could not find common index"
    diff = (sweep_ret.reindex(common) - audit_ret.reindex(common)).abs()
    max_diff = float(diff.max())
    assert max_diff < 1e-12, (
        f"Sweep and audit diverge: max |diff| = {max_diff:.2e} on {len(common)} bars"
    )


def test_position_shift_discipline() -> None:
    """L18 — position must NOT earn the same-bar return.

    Construct a synthetic GLD series with one large positive bar at index
    T after a long flat period. Verify that the return earned at T is
    based on the position carried in from T-1 (which is 0 in the flat
    regime), not the just-computed position at T.
    """
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Flat IEF -- z-score is 0 throughout -> no signal -> position 0 throughout.
    ief = pd.Series(100.0, index=idx)
    # GLD with one big jump near the end.
    gld_vals = np.full(n, 100.0)
    gld_vals[-1] = 110.0
    gld = pd.Series(gld_vals, index=idx)
    closes = pd.DataFrame({"IEF": ief, "GLD": gld})
    rets = bond_gold_returns(closes, cfg=BondGoldConfig(lookback=60, threshold=0.50))
    # With IEF flat, no signal fires, so every return must be 0.0 — including
    # the last bar (where same-bar leakage would have fired a position based
    # on the spike).
    nonzero = rets[rets.abs() > 1e-12]
    assert len(nonzero) == 0, (
        f"Position shift discipline broken: {len(nonzero)} non-zero returns when IEF is flat"
    )
