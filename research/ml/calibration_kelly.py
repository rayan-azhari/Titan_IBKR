"""calibration_kelly.py -- Phase 0E: Probability Calibration + Kelly Sizing.

Applies isotonic regression to XGBoost predict_proba() to produce calibrated
probabilities, then uses fractional Kelly criterion for position sizing.

Kelly formula: f* = (p * (b+1) - 1) / b
  where p = calibrated P(win), b = win/loss ratio

Position sizing:
    - Full Kelly is too aggressive for noisy financial data
    - Use half-Kelly (f*/2) as default
    - Clip to [0, 1] range

Calibration:
    - IsotonicRegression from sklearn (monotonic, no parametric assumption)
    - Trained on IS validation split (last 20% of IS)
    - Applied to OOS predictions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class CalibratedKellySizer:
    """Calibrates XGBoost probabilities and applies Kelly position sizing."""

    def __init__(self, kelly_fraction: float = 0.5, min_edge: float = 0.02):
        """
        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly).
            min_edge: Minimum calibrated edge (|p - 0.5|) to trade.
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.calibrator_long = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self.calibrator_short = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._fitted = False

    def fit(
        self,
        raw_proba: np.ndarray,
        actual_returns: np.ndarray,
        threshold: float = 0.0,
    ) -> "CalibratedKellySizer":
        """Fit calibrators on IS validation data.

        Args:
            raw_proba: XGBoost P(long) predictions on validation set.
            actual_returns: Realized next-bar returns on validation set.
            threshold: Return threshold for "win" (default 0 = any positive).
        """
        mask = np.isfinite(raw_proba) & np.isfinite(actual_returns)
        proba = raw_proba[mask]
        rets = actual_returns[mask]

        # Binary outcomes: did a long position win?
        long_win = (rets > threshold).astype(float)
        short_win = (rets < -threshold).astype(float)

        self.calibrator_long.fit(proba, long_win)
        # For short: higher raw_proba means less likely to win short
        self.calibrator_short.fit(1.0 - proba, short_win)

        # Estimate average win/loss ratio for Kelly
        long_wins = rets[rets > threshold]
        long_losses = rets[rets <= threshold]
        short_wins = -rets[rets < -threshold]
        short_losses = -rets[rets >= -threshold]

        self.b_long = (
            float(np.mean(np.abs(long_wins)) / np.mean(np.abs(long_losses)))
            if len(long_wins) > 10 and len(long_losses) > 10
            else 1.0
        )
        self.b_short = (
            float(np.mean(np.abs(short_wins)) / np.mean(np.abs(short_losses)))
            if len(short_wins) > 10 and len(short_losses) > 10
            else 1.0
        )

        self._fitted = True
        return self

    def kelly_size(self, calibrated_p: float, b: float) -> float:
        """Compute fractional Kelly bet size."""
        f_star = (calibrated_p * (b + 1) - 1) / b if b > 0 else 0.0
        return max(0.0, min(1.0, f_star * self.kelly_fraction))

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        """Convert raw XGBoost probabilities to Kelly-sized positions.

        Returns array of positions in [-1, 1] where magnitude = Kelly size.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        n = len(raw_proba)
        positions = np.zeros(n, dtype=np.float64)

        cal_long = self.calibrator_long.predict(raw_proba)
        cal_short = self.calibrator_short.predict(1.0 - raw_proba)

        for i in range(n):
            p_long = cal_long[i]
            p_short = cal_short[i]

            edge_long = p_long - 0.5
            edge_short = p_short - 0.5

            if edge_long > self.min_edge and edge_long > edge_short:
                positions[i] = self.kelly_size(p_long, self.b_long)
            elif edge_short > self.min_edge and edge_short > edge_long:
                positions[i] = -self.kelly_size(p_short, self.b_short)
            # else: flat (no edge)

        return positions


def apply_calibrated_kelly(
    raw_proba: np.ndarray,
    returns: np.ndarray,
    cal_split: float = 0.8,
    kelly_fraction: float = 0.5,
    min_edge: float = 0.02,
) -> tuple[np.ndarray, CalibratedKellySizer]:
    """Full pipeline: fit calibrator on IS-val, apply Kelly sizing.

    Args:
        raw_proba: Full IS probabilities from XGBoost.
        returns: Full IS bar returns.
        cal_split: Fraction of IS for calibrator training.
        kelly_fraction: Half-Kelly = 0.5.
        min_edge: Minimum edge to trade.

    Returns:
        (sized_positions, fitted_sizer)
    """
    n = len(raw_proba)
    split = int(n * cal_split)

    sizer = CalibratedKellySizer(kelly_fraction, min_edge)
    sizer.fit(raw_proba[:split], returns[:split])

    # Apply to second half of IS (validation-like)
    positions = sizer.predict(raw_proba[split:])

    return positions, sizer
