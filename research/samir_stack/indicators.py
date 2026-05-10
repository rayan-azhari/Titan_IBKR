"""Candidate regime indicators for the Samir-Stack strategy.

Each indicator outputs a per-bar **benign-probability score in [0, 1]**
where 1.0 = maximum benign (full risk-on) and 0.0 = maximum hostile
(full cash). All computations are causal — no future bars used.

The indicators are weakly correlated by design (different families):
volatility, trend, momentum, dispersion, credit, drawdown velocity.

The IC pipeline validates each one before it enters the regime score
ensemble. Indicators that don't pass `bh_significant=True` AND
`ICIR > 0.4` get dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.research.metrics import rolling_zscore

# ── Helpers ────────────────────────────────────────────────────────────────


def _normalise_to_unit(x: pd.Series, low: float, high: float, *, invert: bool = False) -> pd.Series:
    """Clip x to [low, high] and rescale to [0, 1]. ``invert=True`` flips so
    that high values map to 0 (hostile)."""
    clipped = x.clip(low, high)
    scaled = (clipped - low) / (high - low)
    if invert:
        scaled = 1.0 - scaled
    return scaled.astype(float)


# ── Tier-1 indicators (always available, 2003+) ───────────────────────────


def vix_regime(vix_close: pd.Series, *, zscore_window: int = 756) -> pd.Series:
    """VIX-regime score from rolling z-score.

    High VIX (or rising VIX) = hostile (low score). Uses 3-year rolling
    z-score so the indicator is regime-relative, not absolute.

    Returns a Series in [0, 1] aligned to ``vix_close``.
    """
    z = rolling_zscore(vix_close, window=zscore_window, min_periods=63)
    # z > +1 (vol elevated above 3y norm) → hostile; z < -1 → benign
    return _normalise_to_unit(z, low=-1.5, high=2.0, invert=True)


def spy_trend(spy_close: pd.Series) -> pd.Series:
    """Trend status: combine SMA(200) and SMA(50) gates.

    Score values:
        1.0 if close > SMA50 > SMA200 (full uptrend)
        0.66 if close > SMA200 only
        0.33 if close > SMA50 only
        0.0 otherwise (close below both MAs)
    """
    sma50 = spy_close.rolling(50).mean()
    sma200 = spy_close.rolling(200).mean()
    above_50 = (spy_close > sma50).astype(float)
    above_200 = (spy_close > sma200).astype(float)
    score = (above_50 * 0.34 + above_200 * 0.66).clip(0.0, 1.0)
    return score


def momentum_12_1(spy_close: pd.Series) -> pd.Series:
    """12-1 momentum: return over t-252..t-21 bars.

    Skips the most recent month to avoid short-term reversal contamination
    (Asness convention). Positive 12-1 = benign trend.
    """
    # Return from 252 days ago to 21 days ago
    mom = spy_close.shift(21) / spy_close.shift(252) - 1.0
    # +20% over the year = full benign; -20% = full hostile
    return _normalise_to_unit(mom, low=-0.20, high=0.20)


def realised_vol_regime(
    spy_close: pd.Series, *, vol_window: int = 21, percentile_window: int = 756
) -> pd.Series:
    """Realised-vol regime: where does current 21d vol sit in the rolling 3-yr distribution?

    Low percentile (current vol below historical) = benign. High = hostile.
    """
    rets = spy_close.pct_change()
    rv = rets.rolling(vol_window).std() * np.sqrt(252)
    # Rolling percentile rank
    pct = rv.rolling(percentile_window, min_periods=63).rank(pct=True)
    return (1.0 - pct).fillna(0.5).clip(0.0, 1.0)


def drawdown_velocity(spy_close: pd.Series, *, window: int = 21) -> pd.Series:
    """Crash-in-progress detector: 21-day cumulative return.

    Rapid drawdowns are the signature of forced-liquidation cascades that
    Samir specifically targets. If 21d return < -8% → hostile.
    """
    r21 = spy_close.pct_change(window)
    # -10% in 21 days → 0 (hostile); +5% → 1 (benign); 0% → 0.66
    return _normalise_to_unit(r21, low=-0.10, high=0.05)


# ── Tier-2 indicators (post-2008) ─────────────────────────────────────────


def credit_spread(
    hyg_close: pd.Series, ief_close: pd.Series, *, zscore_window: int = 756
) -> pd.Series:
    """Credit-spread regime via HYG/IEF ratio z-score.

    Falling HYG/IEF ratio (high yield underperforms Treasuries) = credit
    market panic = hostile. Rising ratio = risk-on credit conditions =
    benign.
    """
    common = hyg_close.index.intersection(ief_close.index)
    ratio = hyg_close.loc[common] / ief_close.loc[common]
    z = rolling_zscore(ratio, window=zscore_window, min_periods=63)
    # Negative z → ratio compressed → credit stress → hostile
    return _normalise_to_unit(z, low=-2.0, high=1.5)


# ── Tier-3 indicators (bond regime proxy) ─────────────────────────────────


def tlt_trend(tlt_close: pd.Series) -> pd.Series:
    """TLT trend score — proxy for rate / yield-curve dynamics.

    TLT > 200d MA AND not falling rapidly (< 5% drop in 30d) = benign rate
    environment. TLT cratering (2022-style) flags rate-shock regime.
    """
    sma200 = tlt_close.rolling(200).mean()
    above_ma = (tlt_close > sma200).astype(float)
    r30 = tlt_close.pct_change(30)
    # -8% in 30d is a severe rate shock; -3% is moderate
    velocity_score = _normalise_to_unit(r30, low=-0.08, high=0.0)
    return (0.5 * above_ma + 0.5 * velocity_score).clip(0.0, 1.0)


# ── Build full indicator panel ─────────────────────────────────────────────


def build_indicator_panel(
    spy_close: pd.Series,
    *,
    vix_close: pd.Series | None = None,
    hyg_close: pd.Series | None = None,
    ief_close: pd.Series | None = None,
    tlt_close: pd.Series | None = None,
    spy_ohlc: pd.DataFrame | None = None,
    enable_hmm: bool = False,
    hmm_n_states: int = 2,
) -> pd.DataFrame:
    """Build the full panel of regime indicators on the SPY index.

    Returns a DataFrame with one column per indicator, all in [0, 1] where
    1 = benign and 0 = hostile. NaN values mean the indicator is not yet
    computable for that bar (warm-up period or data not available).

    Indicators with insufficient data (e.g., HYG/IEF before 2008) will have
    NaN before their start date — the regime-score combiner handles this
    by averaging only over available indicators per bar.

    Optional HMM: if ``enable_hmm=True`` and ``spy_ohlc`` is provided,
    adds a pure-risk causal HMM column. Requires ``hmmlearn`` package.
    """
    panel = pd.DataFrame(index=spy_close.index)

    panel["trend"] = spy_trend(spy_close)
    panel["momentum_12_1"] = momentum_12_1(spy_close)
    panel["rv_regime"] = realised_vol_regime(spy_close)
    panel["dd_velocity"] = drawdown_velocity(spy_close)

    if vix_close is not None:
        v = vix_close.reindex(spy_close.index).ffill(limit=2)
        panel["vix"] = vix_regime(v)
    if hyg_close is not None and ief_close is not None:
        cs = credit_spread(hyg_close, ief_close)
        panel["credit"] = cs.reindex(spy_close.index).ffill(limit=2)
    if tlt_close is not None:
        t = tlt_close.reindex(spy_close.index).ffill(limit=2)
        panel["tlt_trend"] = tlt_trend(t)

    if enable_hmm and spy_ohlc is not None:
        try:
            from research.samir_stack.hmm_risk import hmm_benign_score_causal

            hmm_score = hmm_benign_score_causal(
                spy_ohlc,
                n_states=hmm_n_states,
                warmup_bars=504,
                refit_freq_bars=252,
            )
            panel["hmm_risk"] = hmm_score.reindex(spy_close.index).ffill(limit=2)
        except ImportError:
            # hmmlearn not installed — silently skip
            pass

    return panel
