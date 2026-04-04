"""meta_labeller.py

Fixes Gemini's threshold problem.

Hardcoding 0.6 is overfitting-by-hand: the developer manually tunes a threshold
on test data they can see. The threshold must be:

  1. Treated as a hyperparameter with a defined optimisation target
  2. Tuned on out-of-fold (OOF) predictions from purged CV, not on the test set
  3. Stored alongside the model so production uses the same value

This module also exposes predict_proba() for continuous position sizing
(the meta-label probability is MORE useful than a binary threshold in production).
"""

from typing import Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from .cross_validation import PurgedKFold, compute_sample_uniqueness


class MetaLabeller:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        threshold_target: Literal["f1", "precision", "recall"] = "f1",
        use_sample_weights: bool = True,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.threshold_target = threshold_target
        self.use_sample_weights = use_sample_weights

        self._model: Optional[XGBClassifier] = None
        self._optimal_threshold: float = 0.5
        self._cv_results: Dict = {}
        self._feature_names: Optional[list] = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        exit_times: pd.Series,
        bar_times: Optional[pd.DatetimeIndex] = None,
    ) -> "MetaLabeller":
        """Fit meta-labeller with purged/embargoed CV.

        Args:
            X:           Features, indexed by entry time.
            y:           Binary meta-labels (1 = hit upper barrier, 0 = did not).
            exit_times:  Triple barrier exit times (for purging). Same index as X.
            bar_times:   Full price bar DatetimeIndex for computing sample uniqueness.
                         If None, all samples receive equal weight.
        """
        self._feature_names = list(X.columns)

        # Compute sample uniqueness weights
        if self.use_sample_weights and bar_times is not None:
            uniqueness = compute_sample_uniqueness(X.index, exit_times, bar_times)
            sample_weights = uniqueness.reindex(X.index).fillna(1.0)
        else:
            sample_weights = pd.Series(1.0, index=X.index)

        cv = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)

        oof_probs = np.full(len(y), np.nan)
        fold_aucs = []

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, exit_times)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]
            sw_tr = sample_weights.iloc[train_idx]

            model = self._build_model()
            model.fit(X_tr, y_tr, sample_weight=sw_tr.values)

            probs = model.predict_proba(X_te)[:, 1]
            oof_probs[test_idx] = probs

            if len(np.unique(y_te)) > 1:
                auc = roc_auc_score(y_te, probs)
                fold_aucs.append(auc)

        # Threshold tuning on OOF predictions only
        valid_mask = ~np.isnan(oof_probs)
        y_valid = y.values[valid_mask]
        p_valid = oof_probs[valid_mask]

        self._optimal_threshold = self._tune_threshold(y_valid, p_valid)

        oof_preds = (p_valid >= self._optimal_threshold).astype(int)
        self._cv_results = {
            "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
            "std_auc": float(np.std(fold_aucs)) if fold_aucs else None,
            "oof_auc": float(roc_auc_score(y_valid, p_valid))
            if len(np.unique(y_valid)) > 1
            else None,
            "oof_f1": float(f1_score(y_valid, oof_preds, zero_division=0)),
            "oof_precision": float(precision_score(y_valid, oof_preds, zero_division=0)),
            "oof_recall": float(recall_score(y_valid, oof_preds, zero_division=0)),
            "optimal_threshold": self._optimal_threshold,
            "positive_rate": float(y.mean()),
            "n_train": len(y),
        }

        # Final model on full data
        self._model = self._build_model()
        self._model.fit(X, y, sample_weight=sample_weights.values)

        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of success (upper barrier hit). Shape [n,]."""
        self._assert_fitted()
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary prediction using CV-tuned threshold."""
        return (self.predict_proba(X) >= self._optimal_threshold).astype(int)

    @property
    def optimal_threshold(self) -> float:
        return self._optimal_threshold

    @property
    def cv_results(self) -> Dict:
        return self._cv_results

    def feature_importance(self) -> pd.Series:
        self._assert_fitted()
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(
            {
                "model": self._model,
                "threshold": self._optimal_threshold,
                "cv_results": self._cv_results,
                "feature_names": self._feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MetaLabeller":
        data = joblib.load(path)
        obj = cls()
        obj._model = data["model"]
        obj._optimal_threshold = data["threshold"]
        obj._cv_results = data["cv_results"]
        obj._feature_names = data["feature_names"]
        return obj

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_model() -> XGBClassifier:
        return XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,  # prevents overfitting on small financial samples
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    def _tune_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find threshold maximising target metric on OOF predictions.
        Grid search over [0.30, 0.80] at 0.01 increments.
        """
        best_score = -1.0
        best_t = 0.5
        for t in np.linspace(0.30, 0.80, 51):
            y_pred = (y_prob >= t).astype(int)
            if self.threshold_target == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif self.threshold_target == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            else:
                score = recall_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    def _assert_fitted(self):
        if self._model is None:
            raise RuntimeError("MetaLabeller must be fitted or loaded before predicting.")
