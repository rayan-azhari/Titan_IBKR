"""Parity test: frozen-artefact gate == audit-time gate, bar-for-bar.

Validates that `compute_panel_regime_gate_frozen` (live runtime path,
using the I1v2 C6 frozen artefact) produces the SAME gate as
`compute_panel_regime_gate` (audit-time path, fitting HMM on IS) when
both are fed the same IS slice.

This is the V3.6 "live == research" rule: a strategy that doesn't match
its audit bar-for-bar is a strategy that doesn't yet have an honest
deployment claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ewmac.run_b2e_audit import UNIVERSE, _load_close
from research.regime.hmm_gate_v2 import (
    PanelHMMGateConfig,
    compute_panel_regime_gate,
    compute_panel_regime_gate_frozen,
)
from titan.research.framework import slice_sanctuary
from titan.strategies.ewmac_regime.frozen_artefact import load_frozen_artefact

DATA_DIR = PROJECT_ROOT / "data"
PANEL_FP = DATA_DIR / "i1_regime_panel.parquet"
ARTEFACT_FP = DATA_DIR / "i1v2_c6_frozen.json"


@pytest.mark.skipif(not ARTEFACT_FP.exists(), reason="freeze artefact missing")
@pytest.mark.skipif(not PANEL_FP.exists(), reason="regime panel missing")
def test_frozen_gate_matches_research_gate() -> None:
    """The frozen-artefact gate must equal the audit-time gate on every bar.

    Audit-time path: refit HMM on IS slice, recompute trend-friendly mapping.
    Frozen path: load IS-frozen HMM + mapping, apply gate.

    Both should produce the same per-asset gate DataFrame on the visible
    window when the artefact was frozen on the same IS slice.
    """
    # Load universe + panel exactly as the audit does.
    parts = [_load_close(sym, fname) for sym, fname in UNIVERSE.items()]
    closes = pd.concat(parts, axis=1).sort_index()
    first_valid = closes.dropna(how="any").index
    closes = closes.loc[first_valid[0] :].ffill(limit=5)
    panel = pd.read_parquet(PANEL_FP).sort_index().dropna(how="any")

    sanc = slice_sanctuary(closes, months=12)
    closes_v = sanc.visible
    is_end_idx = len(closes_v)

    cfg = PanelHMMGateConfig(
        n_states=2,
        state_id="mean_return",
        min_state_bars=60,
        smoothing_days=5,
        random_seed=42,
    )

    # 1) Audit-time gate (refit).
    audit_gate = compute_panel_regime_gate(
        closes_v,
        panel,
        cfg=cfg,
        is_end_idx=is_end_idx,
    )

    # 2) Frozen-runtime gate (load artefact, apply).
    art = load_frozen_artefact()
    runtime_gate = compute_panel_regime_gate_frozen(
        closes_v,
        panel,
        hmm_model=art.hmm_model,
        trend_friendly_per_asset=art.trend_friendly_per_asset,
        smoothing_days=art.smoothing_days,
        require_broad_trend=art.require_broad_trend,
    )

    # 3) Compare.
    assert audit_gate.shape == runtime_gate.shape, (
        f"shape mismatch: audit={audit_gate.shape} vs runtime={runtime_gate.shape}"
    )
    assert (audit_gate.columns == runtime_gate.columns).all(), "column mismatch"
    diff = audit_gate.values - runtime_gate.values
    max_abs = float(np.max(np.abs(diff)))
    mismatches_per_col = (audit_gate != runtime_gate).sum(axis=0)
    n_mismatch = int(mismatches_per_col.sum())
    total = audit_gate.size
    # Allow zero mismatch -- both paths should be bit-exact (same HMM model,
    # same forward-filter algorithm, same trend-friendly map).
    assert max_abs < 1e-9, (
        f"frozen gate diverges from audit gate: max_abs={max_abs}, "
        f"n_mismatch={n_mismatch}/{total} "
        f"({mismatches_per_col[mismatches_per_col > 0].to_dict()})"
    )
