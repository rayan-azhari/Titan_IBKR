"""Load the I1v2 C6 frozen artefact and reconstruct the HMM model.

The artefact is produced by `scripts/freeze_i1v2_c6_artefact.py`. This
module is the runtime counterpart -- it deserialises the JSON into the
shape `compute_panel_regime_gate_frozen` expects (an hmmlearn-compatible
model object + per-asset trend-friendly set).

Reconstruction uses an unfitted `GaussianHMM` and assigns the frozen
parameters directly. This avoids re-running EM and guarantees bit-exact
parity with the audit-verdict cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hmmlearn import hmm  # type: ignore[import-untyped]

DEFAULT_ARTEFACT_PATH = Path(__file__).resolve().parents[3] / "data" / "i1v2_c6_frozen.json"


class FrozenI1v2C6:
    """Container for the IS-frozen artefact used at runtime."""

    def __init__(
        self,
        *,
        hmm_model: hmm.GaussianHMM,
        feature_names: list[str],
        asset_order: list[str],
        trend_friendly_per_asset: dict[str, set[int]],
        smoothing_days: int,
        require_broad_trend: bool,
        is_end_date: str,
        freeze_ts: str,
        audit_ref: str,
    ) -> None:
        self.hmm_model = hmm_model
        self.feature_names = feature_names
        self.asset_order = asset_order
        self.trend_friendly_per_asset = trend_friendly_per_asset
        self.smoothing_days = smoothing_days
        self.require_broad_trend = require_broad_trend
        self.is_end_date = is_end_date
        self.freeze_ts = freeze_ts
        self.audit_ref = audit_ref


def load_frozen_artefact(path: Path | str | None = None) -> FrozenI1v2C6:
    """Load + deserialise the I1v2 C6 frozen artefact."""
    fp = Path(path) if path else DEFAULT_ARTEFACT_PATH
    if not fp.exists():
        raise FileNotFoundError(
            f"Frozen I1v2 C6 artefact not found at {fp}. "
            f"Run scripts/freeze_i1v2_c6_artefact.py first."
        )
    raw = json.loads(fp.read_text(encoding="utf-8"))

    n_states = int(raw["hmm_model"]["n_components"])
    n_features = len(raw["feature_names"])
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        random_state=int(raw["hmm_config"]["random_seed"]),
    )
    # Direct parameter assignment -- bypasses EM. hmmlearn supports this
    # for "scoring/decoding only" use, which is exactly the runtime case.
    model.startprob_ = np.array(raw["hmm_model"]["startprob"], dtype=float)
    model.transmat_ = np.array(raw["hmm_model"]["transmat"], dtype=float)
    model.means_ = np.array(raw["hmm_model"]["means"], dtype=float)
    model.covars_ = np.array(raw["hmm_model"]["covars"], dtype=float)
    # n_features_ is required by some hmmlearn paths; set explicitly.
    model.n_features = n_features

    # trend_friendly_per_asset: JSON dumps sets as lists; convert back.
    trend_friendly: dict[str, set[int]] = {
        asset: {int(s) for s in states}
        for asset, states in raw["trend_friendly_per_asset"].items()
    }

    return FrozenI1v2C6(
        hmm_model=model,
        feature_names=list(raw["feature_names"]),
        asset_order=list(raw["asset_order"]),
        trend_friendly_per_asset=trend_friendly,
        smoothing_days=int(raw["hmm_config"]["smoothing_days"]),
        require_broad_trend=bool(raw["hmm_config"]["require_broad_trend"]),
        is_end_date=str(raw["is_window"]["end"]),
        freeze_ts=str(raw["freeze_ts"]),
        audit_ref=str(raw["audit_ref"]),
    )
