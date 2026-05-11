"""Runtime regime computation for the Samir-Stack live strategy.

Mirrors the research module ``research/samir_stack/`` but operates on
rolling buffers of daily bars (one update per signal source per day).
Designed for the NautilusTrader live loop where each indicator must be
recomputable from the current signal-buffer snapshot.

The HMM is the only indicator that's not trivial to compute online — it
requires hmmlearn and a fitted model. For deployment we cache the model
and re-fit annually using only past data, identical to the causal
research version (`research/samir_stack/hmm_risk.py::hmm_benign_score_causal`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from titan.research.metrics import BARS_PER_YEAR

# ── Indicator computations (snapshot-based) ────────────────────────────────


def _rolling_zscore_last(series: pd.Series, window: int) -> float:
    """Z-score of the last value vs trailing-window mean/std."""
    if len(series) < max(20, window // 4):
        return float("nan")
    win = series.iloc[-window:] if len(series) >= window else series
    mu = float(win.mean())
    sd = float(win.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return (float(series.iloc[-1]) - mu) / sd


def _normalise_to_unit(x: float, low: float, high: float, *, invert: bool = False) -> float:
    """Clip x to [low, high] then rescale to [0, 1]. invert flips."""
    if not np.isfinite(x):
        return float("nan")
    clipped = max(low, min(high, x))
    scaled = (clipped - low) / (high - low) if high > low else 0.5
    return 1.0 - scaled if invert else scaled


def vix_score(vix_buffer: pd.Series, *, window: int = 756) -> float:
    z = _rolling_zscore_last(vix_buffer, window)
    return _normalise_to_unit(z, low=-1.5, high=2.0, invert=True)


def trend_score(spy_buffer: pd.Series) -> float:
    if len(spy_buffer) < 200:
        return float("nan")
    sma50 = spy_buffer.iloc[-50:].mean()
    sma200 = spy_buffer.iloc[-200:].mean()
    last = spy_buffer.iloc[-1]
    above_50 = 1.0 if last > sma50 else 0.0
    above_200 = 1.0 if last > sma200 else 0.0
    return min(1.0, above_50 * 0.34 + above_200 * 0.66)


def momentum_12_1_score(spy_buffer: pd.Series) -> float:
    if len(spy_buffer) < 252:
        return float("nan")
    mom = spy_buffer.iloc[-21] / spy_buffer.iloc[-252] - 1.0
    return _normalise_to_unit(mom, low=-0.20, high=0.20)


def realised_vol_regime_score(
    spy_buffer: pd.Series, *, vol_window: int = 21, percentile_window: int = 756
) -> float:
    if len(spy_buffer) < percentile_window + vol_window:
        return float("nan")
    rets = spy_buffer.pct_change()
    rv_window_size = vol_window
    rv_recent = rets.iloc[-percentile_window:].rolling(rv_window_size).std() * math.sqrt(
        BARS_PER_YEAR["D"]
    )
    rv_recent = rv_recent.dropna()
    if len(rv_recent) < 50:
        return float("nan")
    current = rv_recent.iloc[-1]
    pct = float((rv_recent <= current).mean())
    return 1.0 - pct


def drawdown_velocity_score(spy_buffer: pd.Series, *, window: int = 21) -> float:
    if len(spy_buffer) < window + 1:
        return float("nan")
    r = spy_buffer.iloc[-1] / spy_buffer.iloc[-1 - window] - 1.0
    return _normalise_to_unit(r, low=-0.10, high=0.05)


def credit_score(hyg_buffer: pd.Series, ief_buffer: pd.Series, *, window: int = 756) -> float:
    common = hyg_buffer.index.intersection(ief_buffer.index)
    if len(common) < window // 2:
        return float("nan")
    ratio = (hyg_buffer.loc[common] / ief_buffer.loc[common]).dropna()
    if len(ratio) < 100:
        return float("nan")
    z = _rolling_zscore_last(ratio, window)
    return _normalise_to_unit(z, low=-2.0, high=1.5)


# ── Aggregate score ────────────────────────────────────────────────────────


@dataclass
class RegimeBuffers:
    """Rolling buffers for indicator computation. Updated daily."""

    spy: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    vix: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    hyg: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ief: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spy_high: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spy_low: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def add_spy(self, ts: pd.Timestamp, close: float, high: float, low: float) -> None:
        self.spy.loc[ts] = close
        self.spy_high.loc[ts] = high
        self.spy_low.loc[ts] = low
        self.spy = self.spy.sort_index()
        self.spy_high = self.spy_high.sort_index()
        self.spy_low = self.spy_low.sort_index()

    def add_vix(self, ts: pd.Timestamp, close: float) -> None:
        self.vix.loc[ts] = close
        self.vix = self.vix.sort_index()

    def add_hyg(self, ts: pd.Timestamp, close: float) -> None:
        self.hyg.loc[ts] = close
        self.hyg = self.hyg.sort_index()

    def add_ief(self, ts: pd.Timestamp, close: float) -> None:
        self.ief.loc[ts] = close
        self.ief = self.ief.sort_index()

    def trim(self, max_bars: int = 1500) -> None:
        """Keep only the last ``max_bars`` bars to bound memory."""
        for attr in ("spy", "vix", "hyg", "ief", "spy_high", "spy_low"):
            s = getattr(self, attr)
            if len(s) > max_bars:
                setattr(self, attr, s.iloc[-max_bars:])


def compute_regime_score(
    buffers: RegimeBuffers,
    *,
    enable_credit: bool = True,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the ensemble regime score (0=hostile, 1=benign).

    Returns (score, indicator_breakdown). The breakdown dict shows each
    indicator's individual contribution — useful for live diagnostics.
    """
    indicators: dict[str, float] = {}

    indicators["trend"] = trend_score(buffers.spy)
    indicators["momentum_12_1"] = momentum_12_1_score(buffers.spy)
    indicators["rv_regime"] = realised_vol_regime_score(buffers.spy)
    indicators["dd_velocity"] = drawdown_velocity_score(buffers.spy)
    if len(buffers.vix) > 100:
        indicators["vix"] = vix_score(buffers.vix)
    if enable_credit and len(buffers.hyg) > 100 and len(buffers.ief) > 100:
        indicators["credit"] = credit_score(buffers.hyg, buffers.ief)

    valid = {k: v for k, v in indicators.items() if v is not None and np.isfinite(v)}
    if len(valid) < 3:
        return float("nan"), indicators

    if weights is None:
        score = float(np.mean(list(valid.values())))
    else:
        # Weighted mean over valid indicators, normalised
        w_total = 0.0
        s_sum = 0.0
        for k, v in valid.items():
            w = weights.get(k, 1.0)
            s_sum += w * v
            w_total += w
        score = s_sum / w_total if w_total > 0 else float("nan")

    return float(np.clip(score, 0.0, 1.0)), indicators


# ── Tier mapping with hysteresis ───────────────────────────────────────────


def target_tier_from_score(
    score: float,
    current_tier: float,
    *,
    tier_thresholds: tuple[float, ...] = (0.30, 0.50, 0.75),
    hysteresis_buffer: float = 0.05,
    L_max: float = 3.0,
) -> float:
    """Convert score → target tier with hysteresis (variable number of tiers).

    Same logic as ``research/samir_stack/stacked_strategy.py::_equity_target_tier``.
    Going UP requires threshold + buffer; going DOWN uses bare threshold.
    """
    if not np.isfinite(score):
        return current_tier
    target = current_tier
    if score < tier_thresholds[0]:
        target = 0.0
    else:
        for k in range(len(tier_thresholds) - 1, 0, -1):
            if score < tier_thresholds[k] and current_tier > k:
                target = float(k)
                break

    if current_tier == 0.0 and score >= tier_thresholds[0] + hysteresis_buffer:
        target = 1.0
    for k in range(1, len(tier_thresholds)):
        if current_tier <= k and score >= tier_thresholds[k] + hysteresis_buffer:
            target = float(k + 1)

    return min(target, L_max)
