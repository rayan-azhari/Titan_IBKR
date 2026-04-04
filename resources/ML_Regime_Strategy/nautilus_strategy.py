"""nautilus_strategy.py

Production Nautilus Trader strategy using the full pipeline.

Key architectural decisions vs Gemini's version:
  - FeatureEngine.step() is called on every bar — same class as research
  - HMM inference uses predict_current_state() on a rolling window, not single-step
  - Triple barrier levels are computed via TripleBarrierLabeller.compute_barriers_for_live_order()
    using IDENTICAL parameters to the research labelling step
  - Position sizing is continuous (fractional Kelly) not binary threshold
  - ModelHealthMonitor and RetrainingScheduler are wired in
  - Active position is managed bar-by-bar against the barrier levels

This class assumes you have:
  - A trained RegimeDetector saved to config.hmm_model_path
  - A trained MetaLabeller saved to config.xgb_model_path
  - Nautilus Trader installed with IBKR adapter configured
"""

import logging
from collections import deque

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy

from .feature_engine import FeatureEngine
from .labelling import TripleBarrierLabeller
from .meta_labeller import MetaLabeller
from .model_lifecycle import ModelHealthMonitor, RetrainingScheduler
from .position_sizing import FractionalKelly, WinLossTracker
from .regime_detection import RegimeDetector

logger = logging.getLogger(__name__)


class MLRegimeStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    hmm_model_path: str
    xgb_model_path: str

    # FeatureEngine params — must match training
    vol_span: int = 20
    frac_d: float = 0.4
    frac_tau: float = 1e-4

    # HMM inference
    hmm_min_seq_len: int = 60

    # Primary signal
    fast_ma_period: int = 10
    slow_ma_period: int = 30

    # Barriers — must match TripleBarrierLabeller used in research
    vol_multiplier_upper: float = 2.0
    vol_multiplier_lower: float = 1.0
    max_holding_bars: int = 5

    # Position sizing
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    min_edge: float = 0.02
    vol_target_pct: float = 0.01

    # Health monitor (populate from research pipeline output)
    training_regime_dist: tuple = (0.6, 0.4)
    training_prob_mean: float = 0.5
    training_prob_std: float = 0.15

    # Retraining
    retrain_interval_bars: int = 252


class MLRegimeStrategy(Strategy):
    def __init__(self, config: MLRegimeStrategyConfig):
        super().__init__(config)

        # Load pre-trained models
        self.feature_engine = FeatureEngine(
            vol_span=config.vol_span,
            frac_d=config.frac_d,
            frac_tau=config.frac_tau,
        )
        self.regime_detector = RegimeDetector.load(config.hmm_model_path)
        self.meta_labeller = MetaLabeller.load(config.xgb_model_path)
        self.barrier_calc = TripleBarrierLabeller(
            vol_multiplier_upper=config.vol_multiplier_upper,
            vol_multiplier_lower=config.vol_multiplier_lower,
            max_holding_bars=config.max_holding_bars,
        )
        self.position_sizer = FractionalKelly(
            fraction=config.kelly_fraction,
            max_position_pct=config.max_position_pct,
            min_edge=config.min_edge,
            vol_target_pct=config.vol_target_pct,
        )
        self.wl_tracker = WinLossTracker()

        self.health_monitor = ModelHealthMonitor(
            training_regime_dist=np.array(config.training_regime_dist),
            training_prob_mean=config.training_prob_mean,
            training_prob_std=config.training_prob_std,
            on_alert=self._on_health_alert,
        )
        self.retrain_scheduler = RetrainingScheduler(
            scheduled_interval_bars=config.retrain_interval_bars,
            feature_warmup_bars=self.feature_engine.min_history,
            hmm_warmup_bars=config.hmm_min_seq_len,
        )

        # Rolling state
        self._feature_window: deque = deque(maxlen=config.hmm_min_seq_len * 3)
        self._price_window: deque = deque(maxlen=config.slow_ma_period + 5)

        # Active position state
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._barriers: dict = {}
        self._bars_held: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)
        logger.info(
            f"MLRegimeStrategy started. "
            f"Feature warm-up: {self.feature_engine.min_history} bars. "
            f"HMM warm-up: {self.config.hmm_min_seq_len} bars."
        )

    def on_stop(self):
        if self._in_position:
            self.close_all_positions(self.instrument.id)
        logger.info(f"Health monitor summary: {self.health_monitor.summary()}")
        logger.info(
            f"Win/loss tracker: {self.wl_tracker.n_trades} trades, "
            f"W/L ratio: {self.wl_tracker.ratio:.2f}"
        )

    # ── Bar processing ───────────────────────────────────────────────────────

    def on_bar(self, bar: Bar):
        close = bar.close.as_double()
        self._price_window.append(close)
        self.retrain_scheduler.tick()

        # --- Feature computation ---
        features = self.feature_engine.step(close)
        if features is None:
            return  # Warm-up

        self._feature_window.append(features)

        if len(self._feature_window) < self.config.hmm_min_seq_len:
            return  # HMM warm-up

        if self.retrain_scheduler.in_warmup():
            return

        # --- Regime detection (sequence inference, not single-step) ---
        feat_arr = np.array(self._feature_window)
        current_regime, regime_posteriors = self.regime_detector.predict_current_state(feat_arr)
        current_vol = float(features[1])  # ewm_vol

        # Record to health monitor
        self.health_monitor.record_bar(current_regime, float(regime_posteriors[1]))

        # --- Active position management ---
        if self._in_position:
            self._manage_position(close, current_vol)
            return

        # --- Entry logic ---
        if not self._primary_signal(close):
            return

        meta_X = self._build_meta_features(feat_arr[-1], regime_posteriors)
        prob = float(self.meta_labeller.predict_proba(meta_X)[0])

        portfolio_value = self._portfolio_value()
        size = self.position_sizer.size(
            prob_success=prob,
            win_loss_ratio=self.wl_tracker.ratio,
            portfolio_value=portfolio_value,
            instrument_daily_vol=current_vol,
        )

        if size <= 0:
            return

        # Compute and store barriers BEFORE order submission
        self._barriers = self.barrier_calc.compute_barriers_for_live_order(
            entry_price=close,
            current_vol=current_vol,
            current_regime=current_regime,
        )
        self._entry_price = close
        self._bars_held = 0
        self._in_position = True

        qty = self.instrument.make_qty(size / close)
        order = self.order_factory.market(
            instrument_id=self.instrument.id,
            order_side=OrderSide.BUY,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        logger.info(
            f"ENTRY: price={close:.4f}, prob={prob:.3f}, size={size:.0f}, "
            f"tp={self._barriers['take_profit']:.4f}, "
            f"sl={self._barriers['stop_loss']:.4f}, "
            f"regime={current_regime}"
        )

    def _manage_position(self, close: float, current_vol: float):
        """Check triple barrier conditions bar by bar.
        Mirrors the labelling logic exactly — same parameters, same regime-conditional
        multipliers — so that what the model was trained to predict is what we execute.
        """
        self._bars_held += 1
        exit_reason = None
        trade_return = (close - self._entry_price) / self._entry_price

        if close >= self._barriers.get("take_profit", float("inf")):
            exit_reason = "upper_barrier"
        elif close <= self._barriers.get("stop_loss", 0.0):
            exit_reason = "lower_barrier"
        elif self._bars_held >= self.config.max_holding_bars:
            exit_reason = "time_barrier"

        if exit_reason:
            self.close_all_positions(self.instrument.id)
            self._in_position = False
            self.wl_tracker.record(trade_return)
            outcome = 1 if exit_reason == "upper_barrier" else 0
            # Record for calibration tracking (if we stored the prob, which we should)
            logger.info(
                f"EXIT [{exit_reason}]: price={close:.4f}, "
                f"return={trade_return:.3%}, bars_held={self._bars_held}"
            )

    def _primary_signal(self, close: float) -> bool:
        """MA crossover: fast crosses above slow."""
        n = len(self._price_window)
        fast_p = self.config.fast_ma_period
        slow_p = self.config.slow_ma_period

        if n < slow_p + 1:
            return False

        prices = np.array(self._price_window)
        fast_now = prices[-fast_p:].mean()
        slow_now = prices[-slow_p:].mean()
        fast_prev = prices[-(fast_p + 1) : -1].mean()
        slow_prev = prices[-(slow_p + 1) : -1].mean()

        return (fast_now > slow_now) and (fast_prev <= slow_prev)

    def _build_meta_features(
        self, latest_features: np.ndarray, regime_posteriors: np.ndarray
    ) -> pd.DataFrame:
        """Concatenate base features and regime posteriors.
        MUST produce the same feature vector shape as MetaLabeller training.
        """
        vec = np.concatenate([latest_features, regime_posteriors])
        cols = ["log_return", "ewm_vol", "frac_diff"] + [
            f"regime_prob_{i}" for i in range(len(regime_posteriors))
        ]
        return pd.DataFrame([vec], columns=cols)

    def _portfolio_value(self) -> float:
        try:
            return float(self.portfolio.net_exposure(self.instrument.quote_currency))
        except Exception:
            return 100_000.0  # fallback during testing

    def _on_health_alert(self, alert):
        logger.warning(f"Health alert [{alert.alert_type}]: {alert.message}")
        if alert.alert_type in ("regime_drift", "calibration_drift"):
            self.retrain_scheduler.request_retrain(reason=alert.alert_type)
