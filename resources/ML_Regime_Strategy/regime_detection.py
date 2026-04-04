"""regime_detection.py

Fixes two bugs from Gemini's HMM implementation:

1. INFERENCE BUG: predict() on a single observation discards the transition
   probability context that makes HMMs useful. The live strategy must maintain
   a rolling window and call predict_proba() on the sequence, taking the last
   step's posterior via the forward-backward algorithm.

2. LABEL SWITCHING: Unsupervised models have no ordinal stability. State 0 today
   may become State 1 after retraining. We fix this by establishing a canonical
   ordering at fit time, sorting states by their mean volatility feature ascending:
   State 0 = lowest volatility (mean-reverting), State 1 = highest volatility (trending).
   This ordering is deterministic and physically interpretable.
"""

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
        # Maps position in canonical ordering -> original HMM state index
        self._canonical_to_raw: Optional[np.ndarray] = None

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray) -> "RegimeDetector":
        """Fit HMM and establish canonical state ordering.
        features: [n_samples x n_features], typically [log_ret, ewm_vol, frac_diff]
        """
        X = self._scaler.fit_transform(features)

        self._model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
        )
        self._model.fit(X)

        # Establish canonical ordering by volatility feature mean, ascending.
        # For n_states > 2: argsort gives [lowest_vol_state, ..., highest_vol_state]
        # in canonical (0, 1, ...) order. We store the mapping raw->canonical.
        raw_vol_means = self._model.means_[:, self.vol_feature_idx]
        # _canonical_to_raw[canonical_label] = raw_hmm_label
        self._canonical_to_raw = np.argsort(raw_vol_means)

        return self

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_sequence(self, features: np.ndarray) -> np.ndarray:
        """Predict canonical regime labels for a full historical sequence.
        Used in research pipeline to label training data.
        """
        self._assert_fitted()
        X = self._scaler.transform(features)
        raw_states = self._model.predict(X)
        return self._to_canonical(raw_states)

    def predict_current_state(
        self, feature_window: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """Predict regime for the CURRENT bar given a rolling window of history.

        Returns:
            current_state: canonical state label (int)
            posteriors: [n_states] array of state probabilities for the last timestep,
                        reordered to canonical labels

        Requires len(feature_window) >= min_seq_len.
        The quality of the posterior degrades for very short windows relative to
        the model's learned transition timescales — min_seq_len should be validated
        empirically against your training sequences.
        """
        self._assert_fitted()
        if len(feature_window) < self.min_seq_len:
            raise ValueError(
                f"Feature window length {len(feature_window)} < min_seq_len {self.min_seq_len}. "
                "Increase rolling buffer or reduce min_seq_len (check posterior quality)."
            )

        X = self._scaler.transform(feature_window)

        # predict_proba runs forward-backward; we take the last timestep
        # Shape: [n_steps, n_states]
        raw_posteriors = self._model.predict_proba(X)
        last_raw = raw_posteriors[-1]  # [n_states]

        # Reorder to canonical labels
        # canonical[0] = prob of state with lowest vol, canonical[1] = highest vol, etc.
        canonical_posteriors = last_raw[self._canonical_to_raw]
        current_state = int(np.argmax(canonical_posteriors))

        return current_state, canonical_posteriors

    # ── Persistence ─────────────────────────────────────────────────────────

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

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _to_canonical(self, raw_states: np.ndarray) -> np.ndarray:
        """Convert raw HMM state labels to canonical ordered labels."""
        # Build reverse map: raw_label -> canonical_label
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
