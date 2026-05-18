"""I1 v2 -- Multi-feature panel HMM with per-asset trend-friendly state mapping.

Pre-registered in ``directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate
2026-05-17.md``.

Key differences from v1 (`hmm_gate.py`):
- v1: per-asset 1D Gaussian HMM on raw daily log-returns -> L51 no-op.
- v2: single GLOBAL multi-feature Gaussian HMM on a cross-asset macro
  regime panel (vix_z, term_spread_z, credit_spread_z, rv20_z,
  spy_above_sma200, dxy_z, dd_velocity_21). Per-asset state-to-trend
  mapping derived IS-only from each asset's mean daily log-return in
  each global state.

Causality (L04/L50):
- HMM fit uses IS panel rows only.
- State decoding uses CAUSAL forward filtering (no Viterbi).
- Per-asset trend-friendly mapping is IS-frozen and applied unchanged
  to OOS.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn import hmm  # type: ignore[import-untyped]

from research.regime.hmm_gate import (
    _causal_forward_states,
    _smooth,
)


@dataclass(frozen=True)
class PanelHMMGateConfig:
    """I1 v2 -- multi-feature HMM regime gate."""

    n_states: int = 2
    state_id: Literal["mean_return", "low_vol"] = "mean_return"
    train_window_bars: int | None = None
    min_state_bars: int = 60
    smoothing_days: int = 0
    random_seed: int = 42
    n_iter: int = 200
    require_broad_trend: bool = False


def fit_panel_hmm(panel_is: pd.DataFrame, *, cfg: PanelHMMGateConfig) -> hmm.GaussianHMM | None:
    """Fit a multivariate Gaussian HMM on the IS regime panel."""
    clean = panel_is.dropna(how="any")
    if cfg.train_window_bars is not None:
        clean = clean.iloc[-cfg.train_window_bars :]
    if len(clean) < max(cfg.n_states * 60, 252):
        return None
    obs = clean.to_numpy()
    model = hmm.GaussianHMM(
        n_components=cfg.n_states,
        covariance_type="full",
        n_iter=cfg.n_iter,
        random_state=cfg.random_seed,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(obs)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(model.transmat_)):
        return None
    return model


def identify_trend_friendly_per_asset(
    state_path_is: np.ndarray,
    asset_log_returns_is: pd.DataFrame,
    *,
    cfg: PanelHMMGateConfig,
) -> dict[str, set[int]]:
    """Per-asset set of trend-friendly state indices. IS-frozen.

    State s is trend-friendly for asset i when the asset's mean IS daily
    log-return restricted to bars where global state == s is > 0 AND
    that subset has >= cfg.min_state_bars observations.
    """
    out: dict[str, set[int]] = {}
    is_len = min(len(state_path_is), len(asset_log_returns_is))
    sp = state_path_is[:is_len]
    rt = asset_log_returns_is.iloc[:is_len].reset_index(drop=True)
    for col in asset_log_returns_is.columns:
        trend_set: set[int] = set()
        for s in range(cfg.n_states):
            mask = sp == s
            if int(mask.sum()) < cfg.min_state_bars:
                continue
            r_s = rt[col][mask].dropna()
            if len(r_s) < cfg.min_state_bars:
                continue
            if float(r_s.mean()) > 0:
                trend_set.add(s)
        out[col] = trend_set
    return out


def identify_trend_friendly_low_vol(
    model: hmm.GaussianHMM,
    feature_names: list[str],
    *,
    cfg: PanelHMMGateConfig,
) -> set[int]:
    """For state_id == "low_vol": single state with the lowest emission
    variance on the rv20_z feature is deemed trend-friendly for ALL
    assets (Carver-style; less per-asset granular).
    """
    if "rv20_z" in feature_names:
        rv_idx = feature_names.index("rv20_z")
        vols = [float(model.covars_[s, rv_idx, rv_idx]) for s in range(cfg.n_states)]
    else:
        vols = [float(np.trace(c)) for c in model.covars_]
    return {int(np.argmin(vols))}


def compute_panel_regime_gate(
    closes_df: pd.DataFrame,
    regime_panel: pd.DataFrame,
    *,
    cfg: PanelHMMGateConfig,
    is_end_idx: int,
) -> pd.DataFrame:
    """I1 v2 -- per-asset gate from a global multi-feature panel HMM.

    Returns a T x N float gate DataFrame in {0.0, 1.0}, aligned to
    closes_df.index. Missing-panel rows return gate=1.0 (no filter).
    """
    out = pd.DataFrame(1.0, index=closes_df.index, columns=closes_df.columns, dtype=float)
    panel_aligned = regime_panel.reindex(closes_df.index).ffill(limit=5).dropna(how="any")
    if len(panel_aligned) < cfg.n_states * 60:
        return out
    is_index = closes_df.index[:is_end_idx]
    panel_is = panel_aligned.reindex(is_index).dropna(how="any")
    if len(panel_is) < max(cfg.n_states * 60, 252):
        return out
    model = fit_panel_hmm(panel_is, cfg=cfg)
    if model is None:
        return out
    full_obs = panel_aligned.to_numpy()
    state_path_full = _causal_forward_states(model, full_obs)
    if cfg.smoothing_days > 1:
        state_path_full = _smooth(state_path_full, cfg.smoothing_days)
    state_path_full_series = pd.Series(state_path_full, index=panel_aligned.index)
    # Per-asset trend-friendly mapping (IS-frozen).
    log_ret = np.log(closes_df / closes_df.shift(1))
    log_ret_is = log_ret.iloc[:is_end_idx]
    state_path_is_series = state_path_full_series.reindex(is_index).ffill().fillna(0).astype(int)
    if cfg.state_id == "low_vol":
        feature_names = list(panel_aligned.columns)
        single_set = identify_trend_friendly_low_vol(model, feature_names, cfg=cfg)
        per_asset_friendly: dict[str, set[int]] = {col: single_set for col in closes_df.columns}
    else:
        per_asset_friendly = identify_trend_friendly_per_asset(
            state_path_is_series.to_numpy(),
            log_ret_is,
            cfg=cfg,
        )
    if cfg.require_broad_trend:
        broad_mean_log_ret = log_ret.mean(axis=1).fillna(0.0)
        broad_trend = broad_mean_log_ret.ewm(halflife=64, adjust=False).mean()
        broad_ok = (broad_trend > 0).reindex(closes_df.index).fillna(False).to_numpy()
    else:
        broad_ok = np.ones(len(closes_df), dtype=bool)
    state_path_on_close = (
        state_path_full_series.reindex(closes_df.index).ffill().fillna(0).astype(int).to_numpy()
    )
    for col in closes_df.columns:
        friendly = per_asset_friendly.get(col, set())
        if not friendly:
            out[col] = 0.0
            continue
        in_friendly = np.isin(state_path_on_close, list(friendly))
        out[col] = (in_friendly & broad_ok).astype(float)
    return out


def compute_panel_regime_gate_frozen(
    closes_df: pd.DataFrame,
    regime_panel: pd.DataFrame,
    *,
    hmm_model: hmm.GaussianHMM,
    trend_friendly_per_asset: dict[str, set[int]],
    smoothing_days: int = 0,
    require_broad_trend: bool = False,
) -> pd.DataFrame:
    """Live variant: apply a PRE-FROZEN HMM + per-asset mapping.

    Use this for live/shadow execution after `scripts/freeze_i1v2_c6_artefact.py`
    has produced the artefact. The HMM is NOT refit; the trend-friendly
    state sets are NOT recomputed. Both come from the artefact. Only the
    causal forward state path and the per-bar gate application happen at
    runtime.

    This ensures live = research bar-for-bar on the same data: there is
    no in-memory model that drifts between sessions.
    """
    out = pd.DataFrame(1.0, index=closes_df.index, columns=closes_df.columns, dtype=float)
    panel_aligned = regime_panel.reindex(closes_df.index).ffill(limit=5).dropna(how="any")
    if len(panel_aligned) == 0:
        return out
    full_obs = panel_aligned.to_numpy()
    state_path_full = _causal_forward_states(hmm_model, full_obs)
    if smoothing_days > 1:
        state_path_full = _smooth(state_path_full, smoothing_days)
    state_path_full_series = pd.Series(state_path_full, index=panel_aligned.index)

    log_ret = np.log(closes_df / closes_df.shift(1))
    if require_broad_trend:
        broad_mean_log_ret = log_ret.mean(axis=1).fillna(0.0)
        broad_trend = broad_mean_log_ret.ewm(halflife=64, adjust=False).mean()
        broad_ok = (broad_trend > 0).reindex(closes_df.index).fillna(False).to_numpy()
    else:
        broad_ok = np.ones(len(closes_df), dtype=bool)
    state_path_on_close = (
        state_path_full_series.reindex(closes_df.index).ffill().fillna(0).astype(int).to_numpy()
    )
    for col in closes_df.columns:
        friendly = trend_friendly_per_asset.get(col, set())
        if not friendly:
            out[col] = 0.0
            continue
        in_friendly = np.isin(state_path_on_close, list(friendly))
        out[col] = (in_friendly & broad_ok).astype(float)
    return out
