"""Causal VIX-HMM regime classifier (Samir V3 Layer 1).

Fits a 2-state Gaussian HMM on log(VIX) with rolling annual re-fit, and
emits a per-bar **filtering probability** of the low-VIX (benign) state.
Filtering = P(state_t | observations_0..t) — strictly causal, no
look-ahead.

The output ``regime_score`` is a Series in [0, 1] where:
  - 1.0 = high confidence the current bar is in the low-VIX (benign) state
  - 0.0 = high confidence the current bar is in the high-VIX (hostile) state

State labels are anchored by mean log(VIX) within each IS window: the
state with the lower mean is always "low-VIX" regardless of the HMM's
internal numbering for that refit period.

Rolling-refit schedule (matches the original ``research/samir_stack/hmm_risk.py``
``hmm_benign_score_causal`` for comparability):
  - Warmup: 504 trading days (~2 years) of IS history before the first fit.
  - Refit cadence: every 252 bars (~1 year).
  - At each refit: fit HMM on expanding window [start, fit_point]; emit
    filtering probability for [fit_point, fit_point + 252).

See ``directives/Samir V3 — VIX-HMM Strategy Design 2026-05-13.md`` §4.

References
----------
- Rabiner, L. (1989). A tutorial on hidden Markov models. Proc. IEEE.
- hmmlearn — https://hmmlearn.readthedocs.io
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from titan.research.metrics import BARS_PER_YEAR


def _build_features(vix_close: pd.Series) -> pd.DataFrame:
    """Construct the HMM feature matrix from a VIX close series.

    Baseline: log(VIX) only. Single feature keeps the HMM identification
    problem maximally constrained (2 states × 1 mean × 1 variance per state).
    """
    log_vix = np.log(vix_close.clip(lower=1e-6))
    return pd.DataFrame({"log_vix": log_vix})


def _filtering_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Forward-pass filtering: P(state_t | data_0..t) for each t in X.

    hmmlearn's ``predict_proba`` returns the smoothed posterior (using all
    observations, including future ones) which is anti-causal. For real-
    time / causal use we need the filtered posterior, which is the
    normalised forward variable α(t, s) = P(state_t = s, obs_0..t).

    We implement the forward pass manually (rather than depending on
    hmmlearn private methods) so the math is auditable and portable
    across hmmlearn versions.

    Math (log space, numerically stable):

        log_alpha[0, s] = log π[s] + log_emit[0, s]
        log_alpha[t, s] = log_emit[t, s]
                        + logsumexp_{s'}(log_alpha[t-1, s'] + log_transmat[s', s])

    Filtering prob:

        P(state_t = s | obs_0..t) = exp(log_alpha[t, s]) / Σ_s' exp(log_alpha[t, s'])

    Returns array of shape (T, n_states) summing to 1 per row.
    """
    framelogprob = model._compute_log_likelihood(X)
    T, n_states = framelogprob.shape
    log_pi = np.log(np.asarray(model.startprob_) + 1e-300)
    log_A = np.log(np.asarray(model.transmat_) + 1e-300)

    log_alpha = np.empty((T, n_states))
    log_alpha[0] = log_pi + framelogprob[0]
    for t in range(1, T):
        # log_alpha[t, s] = log_emit[t, s]
        #                 + logsumexp_s'(log_alpha[t-1, s'] + log_A[s', s])
        log_alpha[t] = framelogprob[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

    log_normaliser = logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha - log_normaliser)


def _fit_one_window(feats_is: np.ndarray, n_states: int, seed: int = 42):
    """Fit a Gaussian HMM on the in-sample feature matrix.

    Returns the fitted model. Caller is responsible for state-relabelling.
    """
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",  # one feature → diag == full
        n_iter=200,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(feats_is)
    return model


def _identify_low_vol_state(model, feats_is: np.ndarray) -> int:
    """Return the state index whose IS-period mean is the lowest.

    For our single feature (log VIX) this is just argmin of model.means_.
    Wrapped in a function so the caller's logic is independent of how
    states were numbered by the EM solver.
    """
    means = model.means_[:, 0]  # log VIX mean per state (1 feature)
    return int(np.argmin(means))


def vix_hmm_regime_score(
    vix_close: pd.Series,
    *,
    warmup_bars: int = 504,
    refit_freq_bars: int = 252,
    n_states: int = 2,
    seed: int = 42,
) -> pd.Series:
    """Causal rolling VIX-HMM filtering probability of the benign state.

    Parameters
    ----------
    vix_close : pd.Series
        Daily VIX close prices, datetime-indexed.
    warmup_bars : int
        Initial IS window before the first fit (default 504 = ~2 years).
    refit_freq_bars : int
        Refit cadence (default 252 = annual).
    n_states : int
        HMM state count. Default 2 (binary regime classification per Samir).
    seed : int
        RNG seed for hmmlearn's EM initialisation (reproducibility).

    Returns
    -------
    pd.Series
        regime_score ∈ [0, 1], indexed to ``vix_close.index``.
        Bars before the first fit point return 0.5 (neutral — no
        information yet, strategy should treat this as "do not deploy").
        NaN VIX bars produce NaN scores.

    Notes
    -----
    The HMM is fit on log(VIX) only — a single feature. Two consequences:

    1. The model has only ``n_states × 2`` parameters (mean + variance per
       state). EM converges robustly even on short IS windows.
    2. There is no information in the model about ``Δ VIX`` or trend; if
       the strategy needs that, add it as a layer-2+ feature with care
       (more features = more risk of overfitting).
    """
    feats = _build_features(vix_close).dropna()
    n = len(feats)
    if n < warmup_bars + refit_freq_bars:
        raise ValueError(
            f"Need at least {warmup_bars + refit_freq_bars} bars for the "
            f"VIX-HMM rolling fit, got {n}."
        )

    # Score series initialised to neutral 0.5 (pre-warmup bars).
    scores = pd.Series(0.5, index=feats.index, dtype=float)

    feats_array = feats.to_numpy()
    fit_point = warmup_bars
    while fit_point < n:
        next_fit = min(fit_point + refit_freq_bars, n)
        is_feats = feats_array[:fit_point]
        try:
            model = _fit_one_window(is_feats, n_states=n_states, seed=seed)
        except Exception:
            # If EM fails to converge for this window (rare with log VIX),
            # leave scores at 0.5 (neutral) for the OOS chunk and move on.
            fit_point = next_fit
            continue

        low_state = _identify_low_vol_state(model, is_feats)

        # Filtering for the OOS chunk on the EXPANDING window [start, next_fit].
        # We then slice out the scores for the OOS bars only (fit_point :
        # next_fit). This ensures each OOS bar's score is computed using
        # observations strictly up to that bar (the forward-pass property).
        chunk_end_for_filtering = next_fit
        feats_to_filter = feats_array[:chunk_end_for_filtering]
        probs = _filtering_probabilities(model, feats_to_filter)
        oos_probs = probs[fit_point:next_fit, low_state]
        scores.iloc[fit_point:next_fit] = oos_probs

        fit_point = next_fit

    # Reindex to the original VIX index (some NaN bars may have been dropped).
    return scores.reindex(vix_close.index).rename("vix_hmm_regime_score")


def vix_hmm_forward_vol_discrimination(
    vix_close: pd.Series,
    spy_close: pd.Series,
    regime_score: pd.Series,
    *,
    forward_window: int = 21,
    threshold: float = 0.5,
) -> dict:
    """Quick sanity check on the HMM classifier — does the "benign" state
    actually have lower forward realised vol than the "hostile" state?

    Returns a dict with per-state forward-vol statistics. If the benign
    state's forward 21-day vol is NOT meaningfully lower than the hostile
    state's, the HMM isn't classifying real regimes and layer 1 fails the
    diagnostic gate.

    Per Samir's framework: the regime gate's value is reducing forward
    *dispersion* (variance compounding against you), not reducing mean
    return. So forward-vol discrimination is the right metric.
    """
    common = vix_close.index.intersection(spy_close.index).intersection(regime_score.index)
    rets = spy_close.reindex(common).pct_change()
    fwd_vol = rets.shift(-forward_window).rolling(forward_window).std() * math.sqrt(
        BARS_PER_YEAR["D"]
    )
    # Align: at bar t we know the score; the FWD vol is computed from
    # rets[t+1 : t+forward_window+1] (true forward). The .shift(-forward_window)
    # then .rolling(forward_window) produces the forward window ending forward_window
    # bars from now — i.e. uses obs after t. That's appropriate here because
    # this is a *diagnostic*, not a tradable signal — we measure "given
    # today's regime classification, what did vol actually do over the
    # next 21 days?"

    df = pd.DataFrame({"score": regime_score.reindex(common), "fwd_vol": fwd_vol}).dropna()

    benign_mask = df["score"] > threshold
    hostile_mask = df["score"] <= threshold

    benign = df.loc[benign_mask, "fwd_vol"]
    hostile = df.loc[hostile_mask, "fwd_vol"]

    return {
        "n_bars": len(df),
        "frac_benign": float(benign_mask.mean()),
        "frac_hostile": float(hostile_mask.mean()),
        "fwd_vol_benign_mean": float(benign.mean()),
        "fwd_vol_hostile_mean": float(hostile.mean()),
        "fwd_vol_benign_median": float(benign.median()),
        "fwd_vol_hostile_median": float(hostile.median()),
        "ratio_hostile_over_benign": (
            float(hostile.mean() / benign.mean()) if benign.mean() > 0 else float("nan")
        ),
    }
