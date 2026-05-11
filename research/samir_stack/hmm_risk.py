"""Pure-risk HMM for regime classification.

Unlike the existing `phase0_regime.py` HMM which is fit on
``[log_returns, realised_vol]`` (mixing price direction with risk),
this HMM is fit on **only dispersion / risk features** — no returns.

The intent: classify *risk regimes* (vol clusters), not *price regimes*
(bull/bear). Samir's framework explicitly classifies risk, so the HMM
features should match that intent.

Features:
    * realised_vol_20 (z-scored)
    * abs(log_return) (z-scored — vol surprise magnitude)
    * normalised true range (high-low)/close (z-scored)
    * normalised ATR(14) (z-scored)
    * 5-day max drawdown from rolling peak

All features are POSITIVE / DISPERSION-ONLY. No directional signal.

Implementation supports rolling re-fit for proper OOS testing — though
the convenience method ``fit_full_sample`` is provided for diagnostic
runs (with explicit look-ahead caveat in the docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_risk_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Build the pure-risk feature panel (NO directional features).

    Inputs:
        ohlc — DataFrame with columns ``open``, ``high``, ``low``, ``close``.

    Output:
        DataFrame indexed like ohlc, columns:
            rv20 — 20-day realised vol of log returns
            absret — |log_return| (vol surprise magnitude)
            ntr — normalised true range (high-low)/close
            atr14 — 14-day ATR / close
            dd5 — depth below 5-day rolling high (≤ 0)
    """
    log_ret = np.log(ohlc["close"]).diff()
    rv20 = log_ret.rolling(20).std()
    absret = log_ret.abs()
    ntr = (ohlc["high"] - ohlc["low"]) / ohlc["close"]
    # ATR(14) without a proper TR — close-to-close vol approximation
    tr = np.maximum.reduce(
        [
            (ohlc["high"] - ohlc["low"]).values,
            (ohlc["high"] - ohlc["close"].shift(1)).abs().values,
            (ohlc["low"] - ohlc["close"].shift(1)).abs().values,
        ]
    )
    atr14 = pd.Series(tr, index=ohlc.index).rolling(14).mean() / ohlc["close"]
    rolling_max5 = ohlc["close"].rolling(5).max()
    dd5 = (ohlc["close"] - rolling_max5) / rolling_max5  # ≤ 0
    dd5 = -dd5  # flip so larger = more risk

    return pd.DataFrame(
        {
            "rv20": rv20,
            "absret": absret,
            "ntr": ntr,
            "atr14": atr14,
            "dd5": dd5,
        }
    )


def fit_full_sample(
    ohlc: pd.DataFrame,
    n_states: int = 2,
    *,
    is_frac: float = 0.7,
) -> tuple[pd.Series, dict]:
    """Fit risk HMM with IS/OOS split (look-ahead bias warning — for
    diagnostic use, not deployment).

    Returns
    -------
    states : pd.Series
        Per-bar state labels (0..n_states-1).
    info : dict
        Includes which state corresponds to highest realised vol — that's
        the "hostile" state.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn required. Install: uv add hmmlearn") from e

    feats = _build_risk_features(ohlc).dropna()
    n = len(feats)
    is_n = max(100, int(n * is_frac))

    is_feats = feats.iloc[:is_n]
    mu = is_feats.mean()
    sd = is_feats.std()
    z = (feats - mu) / sd
    X_full = z.to_numpy()
    X_is = X_full[:is_n]

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X_is)
    states = model.predict(X_full)
    states = pd.Series(states, index=feats.index, dtype=int).reindex(ohlc.index).ffill()

    # Identify hostile state = state with highest mean realised vol
    rv = feats["rv20"]
    state_rv = pd.DataFrame({"state": states.reindex(rv.index), "rv": rv}).dropna()
    hostile_state = int(state_rv.groupby("state")["rv"].mean().idxmax())

    info = {
        "n_bars": int(n),
        "is_n": int(is_n),
        "n_states": int(n_states),
        "hostile_state": hostile_state,
        "state_distribution": states.value_counts().sort_index().to_dict(),
        "state_mean_rv": state_rv.groupby("state")["rv"].mean().to_dict(),
    }
    return states, info


def hmm_benign_score(
    ohlc: pd.DataFrame,
    n_states: int = 2,
    *,
    is_frac: float = 0.7,
) -> pd.Series:
    """Convenience: returns benign-probability series in [0, 1] from
    pure-risk HMM. 1.0 if in benign state, 0.0 if in hostile state.

    For multi-state HMM (n_states > 2), states are ranked by mean
    realised vol — the LOWEST-vol state gets 1.0, highest-vol gets 0.0,
    intermediate states linearly interpolated.

    NOTE: This is the FULL-SAMPLE fit version (look-ahead bias). For
    deployment use ``hmm_benign_score_causal()`` instead.
    """
    states, info = fit_full_sample(ohlc, n_states=n_states, is_frac=is_frac)
    feats = _build_risk_features(ohlc).dropna()
    rv = feats["rv20"]

    # Rank states by mean rv (low → benign, high → hostile)
    state_rv = pd.DataFrame({"state": states.reindex(rv.index), "rv": rv}).dropna()
    state_rv_mean = state_rv.groupby("state")["rv"].mean().sort_values()
    rank = {s: i / max(1, n_states - 1) for i, s in enumerate(state_rv_mean.index)}
    score_map = {s: 1.0 - rank[s] for s in state_rv_mean.index}
    return states.map(score_map).rename("hmm_risk_score")


def _fit_and_label(
    feats: pd.DataFrame,
    is_end: int,
    label_start: int,
    label_end: int,
    n_states: int,
):
    """Fit HMM on feats[:is_end] and predict states for feats[label_start:label_end].

    Returns (state_series, hostile_state_id) where hostile_state_id is
    determined by mean realised vol within the IS slice.
    """
    from hmmlearn.hmm import GaussianHMM

    is_feats = feats.iloc[:is_end]
    if len(is_feats) < 100:
        # Not enough data to fit
        return None, None

    mu = is_feats.mean()
    sd = is_feats.std().replace(0.0, 1.0)
    z_full = (feats - mu) / sd
    X_is = z_full.iloc[:is_end].to_numpy()
    X_predict = z_full.iloc[label_start:label_end].to_numpy()

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    try:
        model.fit(X_is)
    except Exception:
        return None, None

    pred = model.predict(X_predict)
    state_series = pd.Series(pred, index=feats.index[label_start:label_end], dtype=int)

    # Identify hostile state on the IS slice (so the labelling is
    # determined causally from the same data the model was trained on)
    is_states = pd.Series(model.predict(X_is), index=feats.index[:is_end])
    rv_is = feats["rv20"].iloc[:is_end]
    state_rv = pd.DataFrame({"state": is_states, "rv": rv_is}).dropna()
    if len(state_rv) < 30:
        return None, None
    hostile_state = int(state_rv.groupby("state")["rv"].mean().idxmax())

    return state_series, hostile_state


def hmm_benign_score_causal(
    ohlc: pd.DataFrame,
    n_states: int = 2,
    *,
    warmup_bars: int = 504,
    refit_freq_bars: int = 252,
) -> pd.Series:
    """Causal pure-risk HMM with rolling re-fit.

    Fits the HMM on an expanding window: at each ``refit_freq_bars``
    interval, re-train on all data up to that point and predict the next
    ``refit_freq_bars`` of states. No bar uses information from after
    its own date.

    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC frame.
    n_states : int
        HMM components (default 2).
    warmup_bars : int
        Bars of history required before the first HMM fit. Bars before
        warmup are returned as 0.5 (neutral) — the score doesn't push
        the strategy in either direction during warmup.
    refit_freq_bars : int
        Re-fit interval (default 252 = 1 year). Lower = more adaptive
        but potentially less stable.

    Returns
    -------
    pd.Series
        Benign-probability score in [0, 1] for each bar of ``ohlc``.
        NaN before any features are computable (~25 bars of warmup for
        rv20).
    """
    feats = _build_risk_features(ohlc).dropna()
    n = len(feats)
    if n < warmup_bars + refit_freq_bars:
        raise ValueError(
            f"Need at least {warmup_bars + refit_freq_bars} bars for causal HMM, got {n}"
        )

    # Build state series, one chunk at a time
    all_scores = pd.Series(0.5, index=feats.index, dtype=float)

    # First fit at warmup_bars; predict [warmup_bars, warmup_bars + refit_freq_bars)
    # Then refit at warmup_bars + k*refit_freq_bars
    fit_point = warmup_bars
    while fit_point < n:
        next_fit = min(fit_point + refit_freq_bars, n)
        state_series, hostile_state = _fit_and_label(
            feats,
            is_end=fit_point,
            label_start=fit_point,
            label_end=next_fit,
            n_states=n_states,
        )
        if state_series is not None and hostile_state is not None:
            # Build score: benign state = 1.0, hostile = 0.0; rank by IS rv
            # Get IS state-rv mapping for ranking
            from hmmlearn.hmm import GaussianHMM  # noqa: F401 (re-import OK)

            # Re-derive score map by ranking IS states by rv
            is_feats_now = feats.iloc[:fit_point]
            mu = is_feats_now.mean()
            sd = is_feats_now.std().replace(0.0, 1.0)
            z_is = ((is_feats_now - mu) / sd).to_numpy()
            model_tmp = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model_tmp.fit(z_is)
            is_states_now = pd.Series(
                model_tmp.predict(z_is),
                index=is_feats_now.index,
            )
            rv_is = feats["rv20"].iloc[:fit_point]
            state_rv_mean = (
                pd.DataFrame({"state": is_states_now, "rv": rv_is})
                .dropna()
                .groupby("state")["rv"]
                .mean()
                .sort_values()
            )
            rank = {int(s): i / max(1, n_states - 1) for i, s in enumerate(state_rv_mean.index)}
            score_map = {s: 1.0 - rank[s] for s in rank}

            # Re-predict OOS chunk with this same model (consistent state IDs)
            oos_feats = feats.iloc[fit_point:next_fit]
            z_oos = ((oos_feats - mu) / sd).to_numpy()
            oos_states = pd.Series(model_tmp.predict(z_oos), index=oos_feats.index)
            chunk_scores = oos_states.map(score_map).astype(float)
            all_scores.loc[chunk_scores.index] = chunk_scores

        fit_point = next_fit

    # Reindex to original ohlc (NaN for pre-feature warmup ~ first 20 bars)
    return all_scores.reindex(ohlc.index).rename("hmm_risk_score_causal")
