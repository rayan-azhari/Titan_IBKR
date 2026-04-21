"""ensemble_stacking.py -- Phase 1B: XGBoost + LSTM Ensemble Stacking.

Trains XGBoost and LSTM independently, then stacks predictions via a
logistic regression meta-learner. The meta-learner has only 3 inputs
(P_xgb, P_lstm, disagreement) = 4 parameters total, preventing overfit.

Level 0: XGBoost(72 features) → P_xgb, LSTM(20×72 seqs) → P_lstm
Level 1: LogisticRegression([P_xgb, P_lstm, |P_xgb-P_lstm|]) → P_final

Nested CV within IS (3 temporal sub-folds) generates unbiased Level 0
predictions for meta-learner training.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from research.ml.lstm_classifier import (
    predict_lstm_classifier,
    train_lstm_classifier,
)
from research.ml.run_52signal_classifier import XGB_PARAMS


class StackedEnsemble:
    """XGBoost + LSTM stacking with logistic meta-learner."""

    def __init__(
        self,
        xgb_params: dict | None = None,
        lstm_hidden: int = 32,
        lstm_lookback: int = 20,
        lstm_epochs: int = 50,
        n_nested_folds: int = 3,
        model_type: str = "lstm",
    ):
        self.xgb_params = xgb_params or XGB_PARAMS.copy()
        self.lstm_hidden = lstm_hidden
        self.lstm_lookback = lstm_lookback
        self.lstm_epochs = lstm_epochs
        self.n_nested_folds = n_nested_folds
        self.model_type = model_type  # "lstm" or "tcn"

        self.xgb_model = None
        self.lstm_model = None
        self.meta_model = None

    def _train_seq_model(self, X, y, patience=5):
        """Train sequence model (LSTM or TCN) based on model_type."""
        if self.model_type == "tcn":
            from research.ml.tcn_classifier import train_tcn_classifier

            return train_tcn_classifier(
                X,
                y,
                lookback=self.lstm_lookback,
                hidden_dim=self.lstm_hidden,
                epochs=self.lstm_epochs,
                patience=patience,
            )
        return train_lstm_classifier(
            X,
            y,
            lookback=self.lstm_lookback,
            hidden_dim=self.lstm_hidden,
            epochs=self.lstm_epochs,
            patience=patience,
        )

    def _predict_seq_model(self, model, X):
        """Predict with sequence model (LSTM or TCN)."""
        if self.model_type == "tcn":
            from research.ml.tcn_classifier import predict_tcn_classifier

            return predict_tcn_classifier(model, X, self.lstm_lookback)
        return predict_lstm_classifier(model, X, self.lstm_lookback)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        entry_mask: np.ndarray,
    ) -> "StackedEnsemble":
        """Fit ensemble on IS data.

        Args:
            X: Full IS feature matrix (n_is, n_features).
            y: Full IS labels (binary, n_is).
            entry_mask: Boolean mask of labeled entry bars within IS.
        """
        n = len(X)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Nested CV for meta-learner training ---
        fold_size = n // self.n_nested_folds
        oof_xgb = np.full(n, np.nan)
        oof_lstm = np.full(n, np.nan)

        for fold_i in range(self.n_nested_folds):
            val_start = fold_i * fold_size
            val_end = (fold_i + 1) * fold_size if fold_i < self.n_nested_folds - 1 else n

            train_mask = np.ones(n, dtype=bool)
            train_mask[val_start:val_end] = False

            # Entry bars in training portion
            train_entries = np.where(train_mask & entry_mask)[0]
            if len(train_entries) < 20:
                continue

            # XGBoost on entry bars
            X_xgb_train = X_clean[train_entries]
            y_xgb_train = y[train_entries]
            pos = y_xgb_train.sum()
            neg = len(y_xgb_train) - pos
            spw = neg / max(pos, 1)

            xgb = XGBClassifier(**self.xgb_params, scale_pos_weight=spw, eval_metric="logloss")
            xgb.fit(X_xgb_train, y_xgb_train)
            oof_xgb[val_start:val_end] = xgb.predict_proba(X_clean[val_start:val_end])[:, 1]

            # Sequence model on entry bars (need full IS for sequences)
            X_train_lstm = X_clean[train_mask]
            y_train_lstm = y[train_mask]
            seq_model = self._train_seq_model(X_train_lstm, y_train_lstm, patience=5)
            if seq_model is not None:
                oof_lstm[val_start:val_end] = self._predict_seq_model(
                    seq_model, X_clean[val_start:val_end]
                )[: val_end - val_start]

        # Fill any remaining NaN with 0.5 (neutral)
        oof_xgb = np.nan_to_num(oof_xgb, nan=0.5)
        oof_lstm = np.nan_to_num(oof_lstm, nan=0.5)

        # Meta-features: [P_xgb, P_lstm, disagreement]
        disagreement = np.abs(oof_xgb - oof_lstm)
        meta_X = np.column_stack([oof_xgb, oof_lstm, disagreement])

        # Meta-learner on entry bars only
        entry_idx = np.where(entry_mask)[0]
        meta_X_entries = meta_X[entry_idx]
        meta_y_entries = y[entry_idx]

        if len(meta_y_entries) >= 20:
            self.meta_model = LogisticRegression(C=1.0, penalty="l2", max_iter=200)
            self.meta_model.fit(meta_X_entries, meta_y_entries)

        # --- Final models on full IS ---
        all_entries = np.where(entry_mask)[0]
        X_final = X_clean[all_entries]
        y_final = y[all_entries]

        pos = y_final.sum()
        neg = len(y_final) - pos
        spw = neg / max(pos, 1)
        self.xgb_model = XGBClassifier(
            **self.xgb_params, scale_pos_weight=spw, eval_metric="logloss"
        )
        self.xgb_model.fit(X_final, y_final)

        self.lstm_model = self._train_seq_model(X_clean, y, patience=5)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(long) for OOS bars. Returns 1D array."""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        p_xgb = self.xgb_model.predict_proba(X_clean)[:, 1]

        if self.lstm_model is not None:
            p_lstm = self._predict_seq_model(self.lstm_model, X_clean)[: len(X)]
        else:
            p_lstm = np.full(len(X), 0.5)

        if self.meta_model is not None:
            disagreement = np.abs(p_xgb - p_lstm)
            meta_X = np.column_stack([p_xgb, p_lstm, disagreement])
            return self.meta_model.predict_proba(meta_X)[:, 1]

        # Fallback: simple average
        return (p_xgb + p_lstm) / 2

    def get_meta_coefficients(self) -> dict | None:
        """Return meta-learner coefficients for diagnostics."""
        if self.meta_model is None:
            return None
        coefs = self.meta_model.coef_[0]
        return {
            "coef_xgb": round(float(coefs[0]), 4),
            "coef_lstm": round(float(coefs[1]), 4),
            "coef_disagreement": round(float(coefs[2]), 4),
            "intercept": round(float(self.meta_model.intercept_[0]), 4),
        }
