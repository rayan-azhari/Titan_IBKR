"""am_pm_signal.py — Conditional probability signal: P(PM archetype | AM archetype).

At 12:00 UTC, classify the AM session using IS-frozen PCA+K-means.
Generate a directional bias (+1 long / -1 short / 0 flat) for the PM session
based on the empirical transition matrix built from IS data.

Look-ahead guarantee:
  - AM classification uses only bars from 00:00–11:55 UTC.
  - Signal applied starting at 12:05 UTC bar (next bar after AM close).
  - Transition matrix trained on IS labels only; frozen for OOS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_profiles.day_constructor import (
    PM_END_UTC,
    PM_START_UTC,
    build_daily_matrices,
)
from research.intraday_profiles.pca_archetypes import PCAArchetypes, describe_archetypes


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


def build_transition_matrix(
    am_labels: np.ndarray,
    pm_labels: np.ndarray,
    n_am: int,
    n_pm: int,
) -> np.ndarray:
    """Empirical P(PM=j | AM=i) from IS co-occurrences.

    Laplace smoothing (+1 pseudocount) prevents zero-probability cells
    on short WFO folds where some AM/PM pairs never co-occur.

    Args:
        am_labels: AM cluster assignments, shape (n_days,).
        pm_labels: PM cluster assignments, shape (n_days,).
        n_am: Number of AM clusters.
        n_pm: Number of PM clusters.

    Returns:
        Probability matrix, shape (n_am, n_pm), rows sum to 1.
    """
    counts = np.ones((n_am, n_pm), dtype=float)   # Laplace prior
    for a, p in zip(am_labels, pm_labels):
        counts[int(a), int(p)] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def align_am_pm_labels(
    am_dates: np.ndarray,
    am_labels: np.ndarray,
    pm_dates: np.ndarray,
    pm_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner join AM and PM on matching calendar dates.

    Args:
        am_dates, am_labels: AM date array and corresponding cluster labels.
        pm_dates, pm_labels: PM date array and corresponding cluster labels.

    Returns:
        Tuple of (common_dates, aligned_am_labels, aligned_pm_labels).
    """
    am_map = {d: l for d, l in zip(am_dates, am_labels)}
    pm_map = {d: l for d, l in zip(pm_dates, pm_labels)}
    common = sorted(set(am_dates) & set(pm_dates))
    if not common:
        return np.array([]), np.array([]), np.array([])
    dates  = np.array(common)
    a_labs = np.array([am_map[d] for d in common])
    p_labs = np.array([pm_map[d] for d in common])
    return dates, a_labs, p_labs


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def make_daily_signals(
    am_labels: np.ndarray,
    am_dates: np.ndarray,
    transition: np.ndarray,
    trend_up_indices: list[int],
    trend_down_indices: list[int],
    threshold: float = 0.40,
) -> pd.Series:
    """Generate a directional signal for each day's PM session.

    Args:
        am_labels: AM cluster assignment per day, shape (n_days,).
        am_dates: Calendar date per AM day, shape (n_days,).
        transition: P(PM | AM) matrix, shape (n_am, n_pm).
        trend_up_indices: PM cluster indices that represent directional up moves.
        trend_down_indices: PM cluster indices that represent directional down moves.
        threshold: Minimum probability to generate a signal.

    Returns:
        Series of {+1, -1, 0} indexed by pd.Timestamp (at 12:00 UTC each day).
    """
    signals = {}
    for date, am_label in zip(am_dates, am_labels):
        probs = transition[int(am_label)]
        p_up   = float(probs[trend_up_indices].sum())
        p_down = float(probs[trend_down_indices].sum())
        if p_up >= threshold and p_up > p_down:
            sig = 1
        elif p_down >= threshold and p_down > p_up:
            sig = -1
        else:
            sig = 0
        # Anchor the signal at the 12:00 UTC bar of each day
        ts = pd.Timestamp(date, tz="UTC").replace(hour=12, minute=0)
        signals[ts] = sig

    return pd.Series(signals, name="signal").sort_index()


def expand_signal_to_bars(
    daily_signals: pd.Series,
    df_index: pd.DatetimeIndex,
) -> pd.Series:
    """Expand daily 12:00 signals to all PM bars via forward fill.

    Signal is active from 12:05 UTC to 21:55 UTC (next bar after signal bar).
    Outside that window the signal is forced to 0 (flat).

    Args:
        daily_signals: pd.Series from make_daily_signals().
        df_index: Full M5 DatetimeIndex to align against.

    Returns:
        Series of {+1, -1, 0} aligned to df_index.
    """
    signal_ff = daily_signals.reindex(df_index).ffill().fillna(0).astype(int)

    # Mask to PM session only: 12:05 UTC to 22:00 UTC (exclusive)
    hours = df_index.hour
    pm_mask = (hours >= PM_START_UTC) & (hours < PM_END_UTC)
    # The 12:00 UTC bar itself is the AM-close bar; signal applies from 12:05 onward
    at_cut  = (hours == PM_START_UTC) & (df_index.minute == 0)
    active  = pm_mask & ~at_cut

    signal_ff[~active] = 0
    return signal_ff


# ---------------------------------------------------------------------------
# Full fitting interface
# ---------------------------------------------------------------------------


class AMPMPipeline:
    """End-to-end AM/PM archetype classification and signal generation.

    Encapsulates:
      - IS fitting: PCAArchetypes for AM + PM, transition matrix
      - OOS scoring: classify new AM sessions, generate PM signals

    Usage:
        pipe = AMPMPipeline(n_am=5, n_pm=5, threshold=0.40)
        pipe.fit(features_is)
        signal_series = pipe.score(features_oos)
    """

    def __init__(
        self,
        n_am: int = 5,
        n_pm: int = 5,
        threshold: float = 0.40,
        n_pca_components: int = 15,
        random_state: int = 42,
    ) -> None:
        self.n_am         = n_am
        self.n_pm         = n_pm
        self.threshold    = threshold
        self.n_pca        = n_pca_components
        self.random_state = random_state

        self.am_model_: PCAArchetypes | None = None
        self.pm_model_: PCAArchetypes | None = None
        self.transition_: np.ndarray | None  = None
        self.trend_up_idx_:   list[int] = []
        self.trend_down_idx_: list[int] = []
        self.am_descriptions_: list[dict] = []
        self.pm_descriptions_: list[dict] = []

    def fit(self, features_is: pd.DataFrame) -> "AMPMPipeline":
        """Fit AM and PM archetypes + transition matrix on IS features.

        Args:
            features_is: Output of engineer_features() on IS slice, UTC DatetimeIndex.

        Returns:
            self
        """
        am_mats, am_dates = build_daily_matrices(features_is, session="am")
        pm_mats, pm_dates = build_daily_matrices(features_is, session="pm")

        self.am_model_ = PCAArchetypes(self.n_pca, self.n_am, self.random_state)
        self.am_model_.fit(am_mats)
        am_labels = self.am_model_.predict(am_mats)

        self.pm_model_ = PCAArchetypes(self.n_pca, self.n_pm, self.random_state)
        self.pm_model_.fit(pm_mats)
        pm_labels = self.pm_model_.predict(pm_mats)

        # Describe archetypes to determine directional indices
        actual_n_am = int(self.am_model_.km_.n_clusters)
        actual_n_pm = int(self.pm_model_.km_.n_clusters)
        self.am_descriptions_ = describe_archetypes(am_mats, am_labels, actual_n_am)
        self.pm_descriptions_ = describe_archetypes(pm_mats, pm_labels, actual_n_pm)

        self.trend_up_idx_   = [
            d["label"] for d in self.pm_descriptions_
            if d.get("suggested_name") == "TrendUp"
        ]
        self.trend_down_idx_ = [
            d["label"] for d in self.pm_descriptions_
            if d.get("suggested_name") == "TrendDown"
        ]

        # Align dates and build transition matrix
        _, a_labs, p_labs = align_am_pm_labels(am_dates, am_labels, pm_dates, pm_labels)
        self.transition_ = build_transition_matrix(a_labs, p_labs, self.n_am, self.n_pm)

        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Score new data: classify AM sessions and generate PM signals.

        Args:
            features: Output of engineer_features() for OOS period, UTC DatetimeIndex.

        Returns:
            Daily signal Series at 12:00 UTC bars (+1 / -1 / 0).
        """
        assert self.am_model_ is not None, "Call fit() first."
        am_mats, am_dates = build_daily_matrices(features, session="am")
        if len(am_mats) == 0:
            return pd.Series(dtype=int)

        am_labels = self.am_model_.predict(am_mats)
        return make_daily_signals(
            am_labels, am_dates,
            self.transition_,
            self.trend_up_idx_,
            self.trend_down_idx_,
            self.threshold,
        )
