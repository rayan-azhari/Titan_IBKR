"""quantile_features.py -- Phase 0D: Quantile Regression Feature Layer.

Trains quantile regressors (10th, 50th, 90th percentile) on IS data to
predict next-bar return distribution. The quantile predictions become
features for the downstream XGBoost classifier.

Features produced (3):
    qr_p10:   predicted 10th percentile of next-bar return
    qr_p50:   predicted median of next-bar return
    qr_p90:   predicted 90th percentile of next-bar return

Derived (2):
    qr_spread: qr_p90 - qr_p10 (predicted volatility / uncertainty)
    qr_skew:   (qr_p90 + qr_p10 - 2*qr_p50) / qr_spread (distribution asymmetry)

Uses LightGBM's built-in quantile objective for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_quantile_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
):
    """Fit a single quantile regression model."""
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        # Fallback to sklearn GradientBoostingRegressor (slower but no extra dep)
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model


class QuantileFeatureExtractor:
    """Fits quantile regressors on IS data, predicts on any data."""

    QUANTILES = [0.10, 0.50, 0.90]

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models: dict[float, object] = {}

    def fit(self, X: np.ndarray, y_returns: np.ndarray) -> "QuantileFeatureExtractor":
        """Fit quantile regressors on training data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y_returns: Next-bar returns (n_samples,).
        """
        # Clean inputs
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_returns)
        X_clean = X[mask]
        y_clean = y_returns[mask]

        for alpha in self.QUANTILES:
            self.models[alpha] = _fit_quantile_model(
                X_clean,
                y_clean,
                alpha,
                self.n_estimators,
                self.max_depth,
                self.learning_rate,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quantiles. Returns (n_samples, 3) array [p10, p50, p90]."""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.column_stack([self.models[alpha].predict(X_clean) for alpha in self.QUANTILES])
        return preds

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """Predict quantiles and compute derived features.

        Returns DataFrame with 5 columns.
        """
        raw = self.predict(X)
        n = raw.shape[0]

        out = pd.DataFrame(index=range(n))
        out["qr_p10"] = raw[:, 0]
        out["qr_p50"] = raw[:, 1]
        out["qr_p90"] = raw[:, 2]

        spread = raw[:, 2] - raw[:, 0]
        out["qr_spread"] = spread

        safe_spread = np.where(np.abs(spread) > 1e-10, spread, np.nan)
        out["qr_skew"] = (raw[:, 2] + raw[:, 0] - 2 * raw[:, 1]) / safe_spread

        return out
