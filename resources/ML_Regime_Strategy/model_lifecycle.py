"""
model_lifecycle.py

Addresses the model degradation problem Gemini omitted entirely.

Financial regimes drift. A model trained in one market environment will degrade
as that environment changes. This module provides:

  1. ModelHealthMonitor: tracks live regime distribution, probability distribution,
     and calibration drift against training-time baselines.

  2. RetrainingScheduler: defines when and how to retrain, including warm-up
     period management when a new model goes live.

The philosophy: instrument everything, alert loudly, never retrain silently.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, List, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    timestamp: datetime
    alert_type: str
    message: str
    value: float
    threshold: float


class ModelHealthMonitor:
    """
    Monitors a deployed model for three forms of degradation:

    1. Regime distribution drift: if the live regime frequency diverges from
       training, the HMM is encountering states it wasn't calibrated for.

    2. Probability distribution drift: if XGBoost's output distribution shifts
       (mean, std), the feature inputs have drifted away from training distribution.

    3. Calibration drift: if the model predicts 70% success but only 50% of trades
       succeed, the model's probability estimates are no longer meaningful.
    """

    def __init__(
        self,
        training_regime_dist: np.ndarray,    # [n_states] normalised frequencies
        training_prob_mean: float,
        training_prob_std: float,
        n_states: int = 2,
        kl_threshold: float = 0.2,
        prob_z_threshold: float = 2.0,
        calibration_threshold: float = 0.15,  # allowed abs deviation in hit rate
        calibration_window: int = 50,
        check_interval_bars: int = 20,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ):
        self.training_regime_dist = training_regime_dist
        self.training_prob_mean = training_prob_mean
        self.training_prob_std = training_prob_std
        self.n_states = n_states
        self.kl_threshold = kl_threshold
        self.prob_z_threshold = prob_z_threshold
        self.calibration_threshold = calibration_threshold
        self.calibration_window = calibration_window
        self.check_interval_bars = check_interval_bars
        self.on_alert = on_alert

        self._live_regimes: List[int] = []
        self._live_probs: List[float] = []
        self._trade_outcomes: List[tuple] = []  # (prob, outcome)
        self.alerts: List[Alert] = []
        self._bar_count = 0

    def record_bar(self, regime: int, prob_success: float):
        self._live_regimes.append(regime)
        self._live_probs.append(prob_success)
        self._bar_count += 1

        if self._bar_count % self.check_interval_bars == 0:
            self._run_checks()

    def record_trade(self, prob_success: float, outcome: int):
        """
        Record completed trade for calibration tracking.
        outcome: 1 if upper barrier hit, 0 if lower or time barrier.
        """
        self._trade_outcomes.append((prob_success, outcome))
        if len(self._trade_outcomes) >= self.calibration_window:
            self._check_calibration()

    def _run_checks(self):
        self._check_regime_drift()
        self._check_prob_drift()

    def _check_regime_drift(self):
        window = self._live_regimes[-200:] if len(self._live_regimes) >= 200 else self._live_regimes
        counts = np.bincount(window, minlength=self.n_states)
        live_dist = counts / counts.sum()

        eps = 1e-10
        p = self.training_regime_dist + eps
        q = live_dist + eps
        kl = float(np.sum(p * np.log(p / q)))

        if kl > self.kl_threshold:
            alert = Alert(
                timestamp=datetime.utcnow(),
                alert_type="regime_drift",
                message=(
                    f"KL divergence {kl:.4f} > threshold {self.kl_threshold}. "
                    f"Training dist: {np.round(self.training_regime_dist, 3).tolist()}, "
                    f"Live dist: {np.round(live_dist, 3).tolist()}"
                ),
                value=kl,
                threshold=self.kl_threshold,
            )
            self._raise_alert(alert)

    def _check_prob_drift(self):
        window = self._live_probs[-200:] if len(self._live_probs) >= 200 else self._live_probs
        if len(window) < 30:
            return
        live_mean = np.mean(window)
        z = abs(live_mean - self.training_prob_mean) / (self.training_prob_std + 1e-10)

        if z > self.prob_z_threshold:
            alert = Alert(
                timestamp=datetime.utcnow(),
                alert_type="prob_drift",
                message=(
                    f"Live prob mean {live_mean:.3f} deviates {z:.1f} std from "
                    f"training mean {self.training_prob_mean:.3f}"
                ),
                value=z,
                threshold=self.prob_z_threshold,
            )
            self._raise_alert(alert)

    def _check_calibration(self):
        recent = self._trade_outcomes[-self.calibration_window:]
        probs = np.array([t[0] for t in recent])
        outcomes = np.array([t[1] for t in recent])
        expected = probs.mean()
        realized = outcomes.mean()
        dev = abs(expected - realized)

        if dev > self.calibration_threshold:
            alert = Alert(
                timestamp=datetime.utcnow(),
                alert_type="calibration_drift",
                message=(
                    f"Expected hit rate {expected:.2%}, realized {realized:.2%}. "
                    f"Deviation {dev:.2%} > threshold {self.calibration_threshold:.2%}"
                ),
                value=dev,
                threshold=self.calibration_threshold,
            )
            self._raise_alert(alert)

    def _raise_alert(self, alert: Alert):
        self.alerts.append(alert)
        logger.warning(f"[ModelHealthMonitor] {alert.alert_type}: {alert.message}")
        if self.on_alert:
            self.on_alert(alert)

    def summary(self) -> Dict:
        if not self._live_regimes:
            return {"status": "no data"}
        counts = np.bincount(self._live_regimes, minlength=self.n_states)
        return {
            "bars_observed": len(self._live_regimes),
            "live_regime_dist": (counts / counts.sum()).round(4).tolist(),
            "live_prob_mean": round(float(np.mean(self._live_probs)), 4),
            "live_prob_std": round(float(np.std(self._live_probs)), 4),
            "n_alerts": len(self.alerts),
            "alert_types": [a.alert_type for a in self.alerts],
            "trades_recorded": len(self._trade_outcomes),
        }


class RetrainingScheduler:
    """
    Defines when retraining should occur and manages the model warm-up period.

    Retraining triggers:
      1. Scheduled: every N bars (time-based)
      2. Alert-triggered: when ModelHealthMonitor fires an alert

    Warm-up: after a new model goes live, it cannot be evaluated or used for
    position sizing until min_history bars have accumulated for features,
    and min_seq_len bars for HMM inference. The scheduler tracks this.
    """

    def __init__(
        self,
        scheduled_interval_bars: int = 252,  # ~1 year daily bars
        min_bars_for_trigger: int = 100,       # don't retrain until we have enough data
        feature_warmup_bars: int = 150,        # FeatureEngine.min_history
        hmm_warmup_bars: int = 60,             # RegimeDetector.min_seq_len
    ):
        self.scheduled_interval_bars = scheduled_interval_bars
        self.min_bars_for_trigger = min_bars_for_trigger
        self.warmup_bars = feature_warmup_bars + hmm_warmup_bars

        self._bars_since_retrain = 0
        self._total_bars = 0
        self._in_warmup = True
        self._retrain_requested = False
        self._retrain_reason: Optional[str] = None

    def tick(self):
        self._bars_since_retrain += 1
        self._total_bars += 1

        if self._in_warmup and self._total_bars >= self.warmup_bars:
            self._in_warmup = False
            logger.info(f"Warm-up complete after {self._total_bars} bars.")

        if self._bars_since_retrain >= self.scheduled_interval_bars:
            self._retrain_requested = True
            self._retrain_reason = "scheduled"

    def request_retrain(self, reason: str = "alert"):
        if self._total_bars >= self.min_bars_for_trigger:
            self._retrain_requested = True
            self._retrain_reason = reason

    def should_retrain(self) -> bool:
        return self._retrain_requested and not self._in_warmup

    def in_warmup(self) -> bool:
        return self._in_warmup

    def acknowledge_retrain(self):
        """Call after retraining completes to reset the scheduler."""
        self._retrain_requested = False
        self._retrain_reason = None
        self._bars_since_retrain = 0
        self._in_warmup = True  # new model needs warm-up
        logger.info("Retraining acknowledged. Warm-up period restarted.")
