"""
regime_detection.py

Fixes two bugs from a prior HMM implementation:

1. INFERENCE BUG: predict() on a single observation discards the transition
   probability context. The live strategy must maintain a rolling window and
   call predict_proba() on the sequence, taking the last step's posterior via
   the forward-backward algorithm.

2. LABEL SWITCHING: Unsupervised models have no ordinal stability. State 0 today
   may become State 1 after retraining. Fix: sort states by mean volatility feature
   ascending at fit time. State 0 = lowest volatility (mean-reverting),
   State 1 = highest volatility (trending). Deterministic and physically interpretable.
"""

import warnings
from typing import Optional, Tuple

import joblib
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class RegimeDetector:

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 200,
        min_seq_len: int = 60,
        vol_feature_idx: int = 1,   # index of volatility in feature vector
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.min_seq_len = min_seq_len
        self.vol_feature_idx = vol_feature_idx

        self._model: Optional[hmm.GaussianHMM] = None
        self._scaler = StandardScaler()
        self._canonical_to_raw: Optional[np.ndarray] = None

    # -- Fitting -------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "RegimeDetector":
        """
        Fit HMM and establish canonical state ordering.
        features: [n_samples x n_features], typically [log_ret, ewm_vol, frac_diff]
        """
        X = self._scaler.fit_transform(features)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=42,
            )
            self._model.fit(X)

        # Establish canonical ordering by volatility feature mean, ascending.
        raw_vol_means = self._model.means_[:, self.vol_feature_idx]
        self._canonical_to_raw = np.argsort(raw_vol_means)

        return self

    # -- Inference -----------------------------------------------------------

    def predict_sequence(self, features: np.ndarray) -> np.ndarray:
        """
        Predict canonical regime labels for a full historical sequence.
        Used in research pipeline to label training data.
        """
        self._assert_fitted()
        X = self._scaler.transform(features)
        raw_states = self._model.predict(X)
        return self._to_canonical(raw_states)

    def predict_current_state(
        self, feature_window: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Predict regime for the CURRENT bar given a rolling window of history.

        Returns:
            current_state: canonical state label (int)
            posteriors: [n_states] array of state probabilities for the last timestep,
                        reordered to canonical labels.

        Requires len(feature_window) >= min_seq_len.
        """
        self._assert_fitted()
        if len(feature_window) < self.min_seq_len:
            raise ValueError(
                f"Feature window length {len(feature_window)} < min_seq_len "
                f"{self.min_seq_len}."
            )

        X = self._scaler.transform(feature_window)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_posteriors = self._model.predict_proba(X)

        last_raw = raw_posteriors[-1]   # [n_states]
        canonical_posteriors = last_raw[self._canonical_to_raw]
        current_state = int(np.argmax(canonical_posteriors))

        return current_state, canonical_posteriors

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str):
        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "canonical_to_raw": self._canonical_to_raw,
                "n_states": self.n_states,
                "min_seq_len": self.min_seq_len,
                "vol_feature_idx": self.vol_feature_idx,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        data = joblib.load(path)
        obj = cls(
            n_states=data["n_states"],
            min_seq_len=data["min_seq_len"],
            vol_feature_idx=data["vol_feature_idx"],
        )
        obj._model = data["model"]
        obj._scaler = data["scaler"]
        obj._canonical_to_raw = data["canonical_to_raw"]
        return obj

    # -- Helpers -------------------------------------------------------------

    def _to_canonical(self, raw_states: np.ndarray) -> np.ndarray:
        raw_to_canonical = np.zeros(self.n_states, dtype=int)
        for canonical, raw in enumerate(self._canonical_to_raw):
            raw_to_canonical[raw] = canonical
        return raw_to_canonical[raw_states]

    def _assert_fitted(self):
        if self._model is None or self._canonical_to_raw is None:
            raise RuntimeError("RegimeDetector must be fitted before calling predict.")

    @property
    def state_descriptions(self) -> dict:
        """Human-readable description of each canonical state."""
        self._assert_fitted()
        means = self._scaler.inverse_transform(self._model.means_)
        desc = {}
        for canonical, raw in enumerate(self._canonical_to_raw):
            vol = means[raw, self.vol_feature_idx]
            desc[canonical] = f"State {canonical}: mean_vol={vol:.6f}"
        return desc
