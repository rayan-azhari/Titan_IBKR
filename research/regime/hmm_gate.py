"""I1 — Per-asset HMM regime gate for EWMAC.

Pre-registered in ``directives/Pre-Reg I1 HMM Per-Asset Regime + EWMAC
Gate 2026-05-16.md``.

API overview:
    - ``HMMGateConfig`` — knobs for a single HMM (state count, training,
      identification heuristic, seed, smoothing).
    - ``PerAssetRegimeGateConfig`` — wrapper used by ``EwmacConfig``.
    - ``fit_one_hmm()`` — fit a Gaussian HMM on a 1-D returns Series.
    - ``identify_trend_state()`` — pick which state is "trend-friendly"
      based on in-sample autocorr / vol heuristic. Frozen on IS.
    - ``compute_per_asset_regime_gate()`` — main entrypoint: per-asset
      forward-filtered gate Series across the full audit window. IS-frozen
      training; OOS state decoding uses CAUSAL forward filtering.

Causality (L04/A1): HMM training is restricted to the IS window. State
decoding uses CAUSAL forward filtering (not Viterbi — Viterbi is
forward-backward and would leak future observations into past state
labels). The forward algorithm computes log P(state at t | obs through t)
and picks the argmax. No information from OOS bars leaks into the gate
at any prior bar.

**L50 (new):** Viterbi decoding is NON-CAUSAL because it's a
forward-backward most-likely-sequence algorithm. For online/causal use,
implement forward-only filtering manually (the helper
``_causal_forward_states()`` below). The hmmlearn ``predict()`` API is
Viterbi by default and MUST NOT be used for live/causal trading.

L13/L14 carry-over: no per-fold hyperparameter tuning. Cells are
pre-committed; the audit harness sweeps cells, not hyperparameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# hmmlearn is the de-facto HMM library used in the project.
from hmmlearn import hmm  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]


@dataclass(frozen=True)
class HMMGateConfig:
    """Single per-asset HMM configuration.

    n_states: 2 or 3. 2 = bull/bear; 3 = bull/range/bear.
    train_window_bars: how many bars to train on per refit (default = full
        IS portion provided by caller).
    state_id: how to identify which state is "trend-friendly".
        "autocorr"   = state with highest sample autocorr of returns is trend.
        "low_vol"    = state with lowest emission variance is trend.
        "high_mean"  = state with highest emission mean is trend.
    smoothing_days: post-Viterbi median-filter window. 0 = no smoothing.
    random_seed: HMM init seed.
    n_iter: max EM iterations.
    """

    n_states: int = 2
    train_window_bars: int | None = None  # None = use all IS data
    state_id: Literal["autocorr", "low_vol", "high_mean"] = "autocorr"
    smoothing_days: int = 0
    random_seed: int = 42
    n_iter: int = 100


@dataclass(frozen=True)
class PerAssetRegimeGateConfig:
    """Wrapper for ``EwmacConfig.per_asset_regime_gate``.

    ``is_min_bars`` is the minimum bars needed in the IS window to fit the
    HMM. Assets with fewer bars get gate=1 everywhere (no filter).
    """

    hmm: HMMGateConfig = HMMGateConfig()
    is_min_bars: int = 252


def fit_one_hmm(returns: pd.Series, *, cfg: HMMGateConfig) -> hmm.GaussianHMM | None:
    """Fit a Gaussian HMM on a 1-D return Series.

    Returns the fitted model, or None on insufficient/degenerate data.
    Catches the EM-convergence and degenerate-state warnings as info.
    """
    clean = returns.dropna().to_numpy().reshape(-1, 1)
    if cfg.train_window_bars is not None:
        clean = clean[-cfg.train_window_bars :]
    if len(clean) < max(cfg.n_states * 30, 60):  # need enough samples per state
        return None
    model = hmm.GaussianHMM(
        n_components=cfg.n_states,
        covariance_type="full",
        n_iter=cfg.n_iter,
        random_state=cfg.random_seed,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(clean)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(model.transmat_)):
        return None
    return model


def identify_trend_state(
    model: hmm.GaussianHMM, *, train_returns: pd.Series, cfg: HMMGateConfig
) -> int | None:
    """Pick which state index is "trend-friendly". IS-frozen.

    Returns the integer state index (0..n_states-1) or None on failure.
    """
    if model is None:
        return None
    train_clean = train_returns.dropna()
    if cfg.train_window_bars is not None:
        train_clean = train_clean.iloc[-cfg.train_window_bars :]
    if len(train_clean) < 60:
        return None
    obs = train_clean.to_numpy().reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states = model.predict(obs)
    except (ValueError, np.linalg.LinAlgError):
        return None

    if cfg.state_id == "low_vol":
        # State with smallest emission variance is the calm-trend state.
        variances = np.array([float(model.covars_[s][0, 0]) for s in range(cfg.n_states)])
        return int(np.argmin(variances))
    if cfg.state_id == "high_mean":
        means = model.means_.flatten()
        return int(np.argmax(means))
    # autocorr: which state's in-sample-returns has highest lag-1 autocorr?
    autocorrs: dict[int, float] = {}
    train_series = pd.Series(train_clean.values, index=train_clean.index)
    for s in range(cfg.n_states):
        mask = states == s
        if mask.sum() < 20:
            autocorrs[s] = -1.0  # not enough data, exclude
            continue
        s_returns = train_series[mask].dropna()
        if len(s_returns) < 20:
            autocorrs[s] = -1.0
            continue
        ac = s_returns.autocorr(lag=1)
        autocorrs[s] = float(ac) if np.isfinite(ac) else -1.0
    if not autocorrs:
        return None
    return int(max(autocorrs.items(), key=lambda kv: kv[1])[0])


def _smooth(states: np.ndarray, window: int) -> np.ndarray:
    """Median-filter the state sequence to suppress single-bar flickers."""
    if window <= 1:
        return states
    s = pd.Series(states).rolling(window, min_periods=1, center=False).median()
    return s.ffill().fillna(0).astype(int).to_numpy()


def _gaussian_log_likelihood(obs: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """Per-bar per-state Gaussian log-likelihood. Manual replacement for
    hmmlearn's private ``_compute_log_likelihood``. obs shape (T, 1),
    means shape (K, 1), covars shape (K, 1, 1). Returns shape (T, K)."""
    n_states = means.shape[0]
    log_likelihoods = np.empty((len(obs), n_states), dtype=float)
    for s in range(n_states):
        mu = float(means[s, 0])
        var = float(covars[s, 0, 0])
        var = max(var, 1e-12)
        # log N(x | mu, var) = -0.5 * (log(2*pi*var) + (x-mu)^2 / var)
        diff = obs[:, 0] - mu
        log_likelihoods[:, s] = -0.5 * (np.log(2 * np.pi * var) + (diff * diff) / var)
    return log_likelihoods


def _causal_forward_states(
    model: hmm.GaussianHMM, obs: np.ndarray
) -> np.ndarray:
    """Forward-filtered most-likely-state sequence — CAUSAL.

    Computes log P(state_t = j | obs_{1..t}) for each t and returns
    argmax over j. Unlike Viterbi (forward-backward), this does NOT use
    any observation after time t when assigning the state at time t.

    L50: This is the correct decoder for online/live trading where
    future observations are unavailable. Viterbi's most-likely-sequence
    decoding looks at the full series and is therefore NON-causal.
    """
    n_obs = len(obs)
    n_states = model.n_components
    log_emit = _gaussian_log_likelihood(obs, model.means_, model.covars_)
    log_transmat = np.log(np.maximum(model.transmat_, 1e-300))
    log_startprob = np.log(np.maximum(model.startprob_, 1e-300))
    log_alpha = np.full((n_obs, n_states), -np.inf, dtype=float)
    log_alpha[0] = log_startprob + log_emit[0]
    for t in range(1, n_obs):
        # log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_transmat[:, j]) + log_emit[t, j]
        # Vectorise across j.
        prev = log_alpha[t - 1][:, None]  # (K, 1)
        combined = prev + log_transmat  # (K, K)
        log_alpha[t] = logsumexp(combined, axis=0) + log_emit[t]
    return np.argmax(log_alpha, axis=1).astype(int)


def compute_per_asset_regime_gate(
    closes_df: pd.DataFrame,
    *,
    cfg: PerAssetRegimeGateConfig,
    is_end_idx: int,
) -> pd.DataFrame:
    """Per-asset Viterbi-decoded regime gate.

    Parameters
    ----------
    closes_df : DataFrame
        Full audit window (IS + OOS), columns = assets.
    cfg : PerAssetRegimeGateConfig
        Per-asset gate config.
    is_end_idx : int
        Number of rows from the start of ``closes_df`` that constitute
        the IS window. The HMM is fit on closes_df.iloc[:is_end_idx]
        per asset; the state-label identification uses the same window;
        Viterbi-decoded predictions are emitted for the full series but
        only the OOS portion is consumed by the strategy. The strategy
        harness slices the gate by f.oos_start..f.oos_end_excl per fold.

    Returns
    -------
    DataFrame indexed like closes_df.index, columns = closes_df.columns,
    values = float gate in {0.0, 1.0}. NaN for unfittable assets =>
    treated as gate=1 by the multiplication step (no filter).
    """
    log_ret = np.log(closes_df / closes_df.shift(1))
    gate = pd.DataFrame(1.0, index=closes_df.index, columns=closes_df.columns, dtype=float)
    for col in closes_df.columns:
        series = log_ret[col]
        # IS portion for training + state-id (frozen).
        is_returns = series.iloc[:is_end_idx]
        if is_returns.dropna().shape[0] < cfg.is_min_bars:
            # Not enough IS data — leave gate = 1 (no filter).
            continue
        model = fit_one_hmm(is_returns, cfg=cfg.hmm)
        if model is None:
            continue
        trend_state = identify_trend_state(model, train_returns=is_returns, cfg=cfg.hmm)
        if trend_state is None:
            continue
        # CAUSAL forward-filter the full series (IS + OOS) using IS-frozen
        # params. We do NOT use model.predict() (Viterbi) because Viterbi
        # is forward-backward and would leak future observations into past
        # state labels — see L50.
        full_obs = series.dropna().to_numpy().reshape(-1, 1)
        if len(full_obs) < 10:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_path = _causal_forward_states(model, full_obs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if cfg.hmm.smoothing_days > 1:
            state_path = _smooth(state_path, cfg.hmm.smoothing_days)
        # Build the per-asset gate Series aligned to closes_df.index.
        full_index = series.dropna().index
        gate_vals = (state_path == trend_state).astype(float)
        per_col = pd.Series(gate_vals, index=full_index)
        gate[col] = per_col.reindex(closes_df.index).fillna(1.0)
    return gate
