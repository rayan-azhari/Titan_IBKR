"""regime.py — 3-state HMM + Hurst exponent dual-filter execution gate.

Regime classification pipeline:
  1. Build smoothed observation vector [log_return, realized_vol] via
     Savitzky-Golay filter (preserves extrema, avoids EMA lag).
  2. Train a 3-state Gaussian HMM on IS data only (random_state=42).
  3. Post-hoc label states: lowest-mean-vol state = "Ranging".
  4. Compute rolling Hurst exponent via R/S analysis.
  5. Gate: allow entries only when P(Ranging) >= p_thresh AND H < hurst_thresh.

CRITICAL: HMM must be trained on IS data, then used in rolling-predict mode
on OOS.  Training on the full dataset introduces look-ahead bias.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.signal import savgol_filter
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# Observation vector
# ---------------------------------------------------------------------------


def build_observations(
    close: pd.Series,
    vol_window: int = 20,
    sg_window: int = 11,
    sg_poly: int = 3,
) -> np.ndarray:
    """Build and smooth the HMM observation matrix.

    Features: [log_return, realized_vol, vol_of_vol, abs_return].
    - log_return   : direction signal
    - realized_vol : level of volatility
    - vol_of_vol   : rolling std of realized_vol — distinguishes regime
                     transitions (high) from stable ranging/trending (low)
    - abs_return   : |log_return| — separates ranging (small) from
                     trending (consistently larger) without sign confusion

    Savitzky-Golay preserves local extrema while suppressing bar-by-bar noise,
    preventing the HMM from whipsawing between states on every tick.

    Args:
        close: Close price series.
        vol_window: Rolling window for realised volatility.
        sg_window: SG filter window length (must be odd).
        sg_poly: SG polynomial order (< sg_window).

    Returns:
        2D numpy array shape (n_bars, 4).
    """
    log_ret = np.log(close / close.shift(1)).fillna(0).values
    s_log_ret = pd.Series(log_ret)
    real_vol = s_log_ret.rolling(vol_window).std().fillna(0).values
    vol_of_vol = pd.Series(real_vol).rolling(vol_window).std().fillna(0).values
    abs_ret = np.abs(log_ret)

    sg_win = sg_window if sg_window % 2 == 1 else sg_window + 1  # must be odd
    smoothed_ret     = savgol_filter(log_ret,    window_length=sg_win, polyorder=sg_poly)
    smoothed_vol     = savgol_filter(real_vol,   window_length=sg_win, polyorder=sg_poly)
    smoothed_vov     = savgol_filter(vol_of_vol, window_length=sg_win, polyorder=sg_poly)
    smoothed_abs_ret = savgol_filter(abs_ret,    window_length=sg_win, polyorder=sg_poly)
    return np.column_stack([smoothed_ret, smoothed_vol, smoothed_vov, smoothed_abs_ret])


# ---------------------------------------------------------------------------
# HMM training
# ---------------------------------------------------------------------------


def train_hmm(
    obs: np.ndarray,
    n_states: int = 3,
    random_state: int = 42,
) -> hmm.GaussianHMM:
    """Train a Gaussian HMM on IS observations.

    States (post-labelling):
      Ranging      — low vol, mean-reverting price action
      Trending     — directional, sustained moves
      High-Vol     — crisis / news spike (high vol, fast-ranging)

    Args:
        obs: Observation array from build_observations() on IS slice.
        n_states: Number of hidden states.
        random_state: Seed for reproducibility.

    Returns:
        Fitted GaussianHMM model.
    """
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        verbose=False,
    )
    model.fit(obs)
    return model


def label_states(model: hmm.GaussianHMM) -> dict[int, str]:
    """Map HMM state indices to human-readable labels.

    The state with the lowest mean realised volatility (index 1 of means)
    is labelled "ranging".  Highest vol = "high_vol".  The remaining = "trending".

    Args:
        model: Fitted HMM model.

    Returns:
        Dict mapping state_index -> label string.
    """
    mean_vols = model.means_[:, 1]   # column 1 = smoothed realised vol
    sorted_idx = np.argsort(mean_vols)
    labels = {}
    labels[sorted_idx[0]] = "ranging"
    labels[sorted_idx[-1]] = "high_vol"
    if len(sorted_idx) == 3:
        labels[sorted_idx[1]] = "trending"
    return labels


def ranging_state_index(model: hmm.GaussianHMM) -> int:
    """Return the HMM state index corresponding to the ranging regime."""
    labels = label_states(model)
    return next(k for k, v in labels.items() if v == "ranging")


# ---------------------------------------------------------------------------
# Rolling posterior (bar-by-bar, no look-ahead)
# ---------------------------------------------------------------------------


def rolling_regime_posterior(
    model: hmm.GaussianHMM,
    obs: np.ndarray,
    ranging_idx: int,
    min_bars: int = 200,
) -> np.ndarray:
    """Compute rolling P(ranging) via incremental HMM forward algorithm.

    Replaces the O(n²) score_samples-on-prefix approach with an O(n) causal
    forward pass.  At each bar t, the filtered posterior P(q_t | o_0..t) is
    computed using only past observations — identical look-ahead guarantee,
    ~1000x faster on 266k-bar M5 datasets.

    Forward recursion (log-space to avoid underflow):
        log_alpha_t[j] = logsumexp(log_alpha_{t-1} + log_A[:, j]) + log_b_j(o_t)
    Posterior: softmax(log_alpha_t)[ranging_idx]

    Args:
        model: Fitted HMM (trained on IS only).
        obs: Full observation array (IS + OOS), shape (n, n_features).
        ranging_idx: State index corresponding to "ranging".
        min_bars: Minimum history before producing a non-NaN posterior.

    Returns:
        1D array of P(ranging) values, NaN for first min_bars rows.
    """
    n, K = len(obs), model.n_components

    # Pre-compute all emission log-probs in one vectorised call: (n, K)
    log_emit = np.zeros((n, K))
    for k in range(K):
        log_emit[:, k] = multivariate_normal.logpdf(
            obs, mean=model.means_[k], cov=model.covars_[k]
        )

    log_A = np.log(model.transmat_ + 1e-300)           # (K, K)
    log_alpha = np.log(model.startprob_ + 1e-300) + log_emit[0]  # (K,)

    posteriors = np.full(n, np.nan)
    if 0 >= min_bars:
        posteriors[0] = np.exp(log_alpha[ranging_idx] - logsumexp(log_alpha))

    for i in range(1, n):
        # Transition: log_alpha[j] = logsumexp_i(log_alpha[i] + log_A[i,j])
        log_alpha = logsumexp(log_alpha[:, None] + log_A, axis=0) + log_emit[i]
        if i >= min_bars:
            posteriors[i] = np.exp(log_alpha[ranging_idx] - logsumexp(log_alpha))

    return posteriors


# ---------------------------------------------------------------------------
# Hurst exponent (R/S analysis)
# ---------------------------------------------------------------------------


def _hurst_rs(series: np.ndarray) -> float:
    """Compute Hurst exponent via rescaled-range (R/S) analysis.

    Includes Anis-Lloyd bias correction for small sample sizes.

    Args:
        series: 1D price array (raw close, not returns).

    Returns:
        Hurst exponent H.  H < 0.5 = mean-reverting, H > 0.5 = trending.
    """
    n = len(series)
    if n < 20:
        return np.nan

    # Work with log-returns
    log_ret = np.log(series[1:] / series[:-1])
    mean_ret = np.mean(log_ret)
    deviations = np.cumsum(log_ret - mean_ret)
    r = np.max(deviations) - np.min(deviations)
    s = np.std(log_ret, ddof=1)
    if s == 0:
        return np.nan
    rs = r / s

    # Anis-Lloyd correction
    if n <= 340:
        correction = (
            (n - 0.5) / n
            * np.sqrt(np.pi / 2)
            / (
                np.sum(1.0 / np.sqrt(np.arange(1, n)))
                if n > 1
                else 1.0
            )
        )
        rs = rs / correction if correction > 0 else rs

    h = np.log(rs) / np.log(n)
    return float(np.clip(h, 0.0, 1.0))


def rolling_hurst(
    close: pd.Series,
    window: int = 500,
) -> pd.Series:
    """Rolling Hurst exponent, shift-1 applied for look-ahead safety.

    TIP: On M5 data, window=500 is ~41 hours of trading.  For faster
    computation during research sweeps, subsample to H1 and forward-fill.

    Args:
        close: Close price series.
        window: Rolling lookback in bars.

    Returns:
        Hurst series (shift-1, look-ahead safe).
    """
    h = close.rolling(window).apply(_hurst_rs, raw=True)
    return h.shift(1)


# ---------------------------------------------------------------------------
# Execution gate
# ---------------------------------------------------------------------------


def regime_gate(
    hmm_posterior: pd.Series | np.ndarray,
    hurst: pd.Series,
    p_thresh: float = 0.65,
    hurst_thresh: float = 0.50,
) -> pd.Series:
    """Dual-filter execution gate.

    Allows entries only when BOTH conditions hold:
      - HMM P(ranging) >= p_thresh
      - Hurst exponent < hurst_thresh (mean-reverting)

    Args:
        hmm_posterior: P(ranging) series (from rolling_regime_posterior).
        hurst: Rolling Hurst series.
        p_thresh: Minimum HMM ranging probability.
        hurst_thresh: Maximum Hurst exponent (below = mean-reverting).

    Returns:
        Boolean Series — True means entries are allowed.
    """
    post = pd.Series(hmm_posterior, index=hurst.index) if isinstance(hmm_posterior, np.ndarray) else hmm_posterior
    return (post >= p_thresh) & (hurst < hurst_thresh)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_hmm(model: hmm.GaussianHMM, path: str) -> None:
    """Save fitted HMM to disk via joblib."""
    joblib.dump(model, path)
    print(f"[HMM] Saved to {path}")


def load_hmm(path: str) -> hmm.GaussianHMM:
    """Load a previously saved HMM model."""
    model = joblib.load(path)
    print(f"[HMM] Loaded from {path}")
    return model
