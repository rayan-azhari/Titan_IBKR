"""freeze_i1v2_c6_artefact.py -- IS-freeze the I1v2 C6_smoothed HMM + mapping.

Produces a single JSON artefact at ``data/i1v2_c6_frozen.json`` containing:
    - HMM model parameters (means, covars, transmat, startprob)
    - Per-asset trend-friendly state sets (IS-frozen mapping)
    - Feature names + ordering (for live panel reconstruction)
    - Freeze metadata (date, audit verdict, etc.)

The live ewmac_regime strategy loads this artefact at on_start and uses
the frozen params for the entire 12-month paper validation period. The
artefact is RE-CREATED only by re-running this script (e.g., after the
2026-11-17 re-audit).

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/freeze_i1v2_c6_artefact.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.ewmac.run_b2e_audit import UNIVERSE, _load_close  # noqa: E402
from research.regime.hmm_gate_v2 import (  # noqa: E402
    PanelHMMGateConfig,
    _causal_forward_states,
    _smooth,
    fit_panel_hmm,
    identify_trend_friendly_per_asset,
)
from titan.research.framework import slice_sanctuary  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
PANEL_FP = DATA_DIR / "i1_regime_panel.parquet"
OUT_FP = DATA_DIR / "i1v2_c6_frozen.json"


def main() -> None:
    print("=" * 80)
    print("Freeze I1v2 C6_smoothed artefact for live deployment")
    print("=" * 80)

    # 1. Load 11-symbol universe + regime panel (same as audit).
    parts = [_load_close(sym, fname) for sym, fname in UNIVERSE.items()]
    closes = pd.concat(parts, axis=1).sort_index()
    first_valid = closes.dropna(how="any").index
    closes = closes.loc[first_valid[0] :].ffill(limit=5)
    panel = pd.read_parquet(PANEL_FP).sort_index().dropna(how="any")
    print(
        f"\nUniverse: {len(closes.columns)} symbols, "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)"
    )
    print(
        f"Panel: {len(panel.columns)} features, "
        f"{panel.index[0].date()} -> {panel.index[-1].date()} ({len(panel)} bars)"
    )

    # 2. Use the FULL visible window (12mo sanctuary held out) as the IS
    #    slice for the freeze. This matches the audit's deployment-window
    #    config: the live strategy starts trading using the IS-frozen HMM
    #    fit on everything up to the 12mo cutoff (i.e. sanctuary acts as
    #    paper-validation start).
    sanc = slice_sanctuary(closes, months=12)
    closes_v = sanc.visible
    is_end_idx = len(closes_v)
    is_index = closes.index[:is_end_idx]
    print(f"\nIS slice: {is_end_idx} bars ({is_index[0].date()} -> {is_index[-1].date()})")
    print(f"Sanctuary held out: {len(closes) - is_end_idx} bars (paper validation starts here)")

    # 3. C6_smoothed cell config (audit-verdict cell).
    cfg = PanelHMMGateConfig(
        n_states=2,
        state_id="mean_return",
        train_window_bars=None,
        min_state_bars=60,
        smoothing_days=5,
        random_seed=42,
        n_iter=200,
        require_broad_trend=False,
    )

    # 4. Align panel to closes_v, fit HMM on IS slice.
    panel_aligned = panel.reindex(closes.index).ffill(limit=5).dropna(how="any")
    panel_is = panel_aligned.reindex(is_index).dropna(how="any")
    if len(panel_is) < cfg.n_states * 60:
        print("ERROR: IS panel too short for HMM fit", file=sys.stderr)
        sys.exit(1)

    model = fit_panel_hmm(panel_is, cfg=cfg)
    if model is None:
        print("ERROR: HMM fit failed", file=sys.stderr)
        sys.exit(1)
    print(
        f"\nHMM fitted: n_states={cfg.n_states}, "
        f"transmat diag={[round(float(d), 3) for d in np.diag(model.transmat_)]}"
    )

    # 5. Compute state path on IS slice for trend-friendly mapping.
    # CRITICAL: smoothing MUST be applied BEFORE deriving the mapping, to
    # match the audit's `compute_panel_regime_gate` path (which smooths
    # state_path_full and then computes per-asset friendly sets on the
    # smoothed series). Parity test enforces this.
    full_obs = panel_aligned.to_numpy()
    state_path_full = _causal_forward_states(model, full_obs)
    if cfg.smoothing_days > 1:
        state_path_full = _smooth(state_path_full, cfg.smoothing_days)
    state_path_full_series = pd.Series(state_path_full, index=panel_aligned.index)
    state_path_is = (
        state_path_full_series.reindex(is_index).ffill().fillna(0).astype(int).to_numpy()
    )

    # 6. Per-asset trend-friendly mapping using IS log-returns.
    log_ret_is = np.log(closes / closes.shift(1)).iloc[:is_end_idx]
    per_asset_friendly = identify_trend_friendly_per_asset(
        state_path_is,
        log_ret_is,
        cfg=cfg,
    )
    print("\nPer-asset trend-friendly states (IS-frozen):")
    for asset in closes.columns:
        friendly = sorted(per_asset_friendly.get(asset, set()))
        print(f"  {asset:>5s}: {friendly if friendly else '(NONE -- gate=0 always)'}")

    # 7. Verify gate fractions on visible window match audit (sanity check).
    # gate_on[t] = 1 if state_path[t] in friendly_set[asset] else 0
    gate_fractions = {}
    for asset in closes.columns:
        friendly = per_asset_friendly.get(asset, set())
        if not friendly:
            gate_fractions[asset] = 0.0
            continue
        gate = np.isin(state_path_full[:is_end_idx], list(friendly)).astype(float)
        gate_fractions[asset] = float(gate.mean())
    print("\nGate fractions on IS slice (sanity vs audit's 0.79/1.0/0.21/0.0):")
    for asset, frac in gate_fractions.items():
        print(f"  {asset:>5s}: {frac:.3f}")

    # 8. Serialise.
    artefact = {
        "schema_version": 1,
        "freeze_ts": datetime.now(timezone.utc).isoformat(),
        "audit_ref": (
            "directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate 2026-05-17.md "
            "Section 4 -- C6_smoothed DEPLOY (Sharpe +0.52, CI_lo +0.049, noise=best)"
        ),
        "is_window": {
            "start": str(is_index[0].date()),
            "end": str(is_index[-1].date()),
            "n_bars": int(is_end_idx),
        },
        "hmm_config": {
            "n_states": cfg.n_states,
            "state_id": cfg.state_id,
            "min_state_bars": cfg.min_state_bars,
            "smoothing_days": cfg.smoothing_days,
            "random_seed": cfg.random_seed,
            "n_iter": cfg.n_iter,
            "require_broad_trend": cfg.require_broad_trend,
        },
        "hmm_model": {
            "n_components": int(model.n_components),
            "means": model.means_.tolist(),
            "covars": model.covars_.tolist(),
            "transmat": model.transmat_.tolist(),
            "startprob": model.startprob_.tolist(),
        },
        "feature_names": list(panel_aligned.columns),
        "asset_order": list(closes.columns),
        "trend_friendly_per_asset": {
            asset: sorted(per_asset_friendly.get(asset, set())) for asset in closes.columns
        },
        "gate_fractions_is": gate_fractions,
    }
    OUT_FP.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    print(f"\nFrozen artefact: {OUT_FP}")
    print(f"  size: {OUT_FP.stat().st_size} bytes")


if __name__ == "__main__":
    main()
