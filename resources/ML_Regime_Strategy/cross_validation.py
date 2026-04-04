"""cross_validation.py

Purged and embargoed K-Fold cross-validation for financial time series.

Standard K-Fold leaks in two ways that are severe for triple-barrier-labelled data:

  1. Serial correlation: an observation at t and t+1 share macroeconomic context.
     A model trained on t can effectively "see" t+1 through correlation.

  2. Label look-ahead: each label spans [entry_time, exit_time]. If any part of
     a label's span falls in the test period, training that observation leaks
     forward-looking information about the test window into the model.

Purging removes training samples whose label spans overlap with the test period.
Embargoing removes a buffer of samples immediately AFTER the test fold to prevent
leakage from trailing correlations.

Reference: de Prado (2018), Advances in Financial Machine Learning, Ch. 7
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        """Args:
        n_splits:     Number of folds (test folds are sequential, not shuffled)
        embargo_pct:  Fraction of total observations to block after each test fold.
                      At 0.01 on 1000 bars, that is 10 bars blocked per fold.
                      Set based on your expected autocorrelation decay horizon.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        exit_times: pd.Series,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index splits with purging and embargoing.

        Args:
            X:           Feature DataFrame. Index must be entry times (DatetimeIndex).
            y:           Labels (same index as X).
            exit_times:  Series where exit_times.iloc[i] is the barrier exit time
                         of observation i. Index matches X.

        Yields:
            (train_indices, test_indices) as numpy integer arrays.

        Notes:
            - Test folds are sequential and non-overlapping.
            - Training data is always strictly before the test fold start
              (plus earlier folds), minus purged and embargoed observations.
            - Earlier test folds CAN appear as training data for later folds
              (walk-forward style), which is realistic for live deployment.
        """
        n = len(X)
        embargo_n = int(n * self.embargo_pct)
        all_indices = np.arange(n)
        test_folds = np.array_split(all_indices, self.n_splits)

        for test_idx in test_folds:
            t0 = test_idx[0]
            t1 = test_idx[-1]
            test_start_time = X.index[t0]
            test_end_time = X.index[t1]

            # Embargo window: [t1+1, t1+embargo_n]
            embargo_end = min(t1 + embargo_n, n - 1)

            train_idx = []
            for i in all_indices:
                # Exclude test set
                if t0 <= i <= t1:
                    continue
                # Exclude embargo zone
                if t1 < i <= embargo_end:
                    continue
                # Purge: exclude if label span overlaps test window
                # Overlap condition: entry_time <= test_end AND exit_time >= test_start
                entry_time = X.index[i]
                exit_time = exit_times.iloc[i]
                if entry_time <= test_end_time and exit_time >= test_start_time:
                    continue

                train_idx.append(i)

            yield np.array(train_idx, dtype=int), test_idx

    def n_purged_per_fold(self, X: pd.DataFrame, exit_times: pd.Series) -> list:
        """Diagnostic: count how many observations are purged per fold."""
        counts = []
        n = len(X)
        test_folds = np.array_split(np.arange(n), self.n_splits)
        for test_idx in test_folds:
            t0, t1 = test_idx[0], test_idx[-1]
            test_start_time = X.index[t0]
            test_end_time = X.index[t1]
            purge_count = 0
            for i in range(n):
                if t0 <= i <= t1:
                    continue
                entry_time = X.index[i]
                exit_time = exit_times.iloc[i]
                if entry_time <= test_end_time and exit_time >= test_start_time:
                    purge_count += 1
            counts.append(purge_count)
        return counts


def compute_sample_uniqueness(
    entry_times: pd.DatetimeIndex,
    exit_times: pd.Series,
    bar_times: pd.DatetimeIndex,
) -> pd.Series:
    """Compute average label uniqueness for each observation.

    Uniqueness measures how much of a label's bar-span is shared with
    other concurrent labels. Observations with low uniqueness should
    receive lower sample weight in training.

    Returns: Series of mean uniqueness per observation, index = entry_times.
    """
    # For each bar, count how many labels are active
    label_counts = pd.Series(0, index=bar_times, dtype=float)

    for entry, exit_t in zip(entry_times, exit_times):
        mask = (bar_times >= entry) & (bar_times <= exit_t)
        label_counts[mask] += 1

    # Uniqueness of each observation = mean(1 / concurrent_labels) over its span
    uniqueness = []
    for entry, exit_t in zip(entry_times, exit_times):
        mask = (bar_times >= entry) & (bar_times <= exit_t)
        concurrent = label_counts[mask]
        if len(concurrent) == 0 or concurrent.sum() == 0:
            uniqueness.append(1.0)
        else:
            uniqueness.append((1.0 / concurrent[concurrent > 0]).mean())

    return pd.Series(uniqueness, index=entry_times)
