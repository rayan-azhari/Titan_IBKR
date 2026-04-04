"""ml_strategy.py
-----------------

ML-driven strategy for NautilusTrader.
Loads a trained Joblib model, warms up with local data, calculates features
on streaming bars, and executes trades based on model predictions.

Tier 2 #6 (April 2026): FractionalKelly sizer + ModelHealthMonitor.
"""

from __future__ import annotations

import logging
from collections import deque
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.strategies.ml.features import build_features, load_feature_config

_logger = logging.getLogger(__name__)


# ── FractionalKelly Sizer (Tier 2 #6) ────────────────────────────────────────


class WinLossTracker:
    """Tracks live win/loss ratio from realised trades."""

    def __init__(self, prior_ratio: float = 1.5, min_trades: int = 10) -> None:
        self._wins: int = 0
        self._losses: int = 0
        self._prior: float = prior_ratio
        self._min_trades: int = min_trades

    def record(self, pnl: float) -> None:
        if pnl > 0:
            self._wins += 1
        elif pnl < 0:
            self._losses += 1

    @property
    def ratio(self) -> float:
        """Win/loss ratio. Returns prior until min_trades reached."""
        if self._wins + self._losses < self._min_trades:
            return self._prior
        if self._losses == 0:
            return self._prior * 2  # Cap at 2x prior
        return self._wins / self._losses

    @property
    def win_rate(self) -> float:
        total = self._wins + self._losses
        if total == 0:
            return 0.5  # prior
        return self._wins / total


class FractionalKelly:
    """Quarter-Kelly position sizer with vol-targeting and hard caps.

    effective_size = min(kelly_size, vol_cap, hard_cap)

    Args:
        fraction:         Kelly fraction (0.25 = quarter-Kelly).
        max_position_pct: Hard cap on position as % of equity.
        vol_target_pct:   Single-trade daily vol contribution cap.
    """

    def __init__(
        self,
        fraction: float = 0.25,
        max_position_pct: float = 0.03,
        vol_target_pct: float = 0.01,
    ) -> None:
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.vol_target_pct = vol_target_pct

    def size(
        self,
        prob_success: float,
        wl_ratio: float,
        equity: float,
        daily_vol: float,
        price: float,
    ) -> int:
        """Compute position size in units (shares/lots).

        Returns 0 if Kelly is negative (no edge).
        """
        if prob_success <= 0 or wl_ratio <= 0 or equity <= 0 or price <= 0:
            return 0

        # Kelly formula: f* = p - (1-p)/b where b = win/loss ratio
        kelly_f = prob_success - (1.0 - prob_success) / wl_ratio
        if kelly_f <= 0:
            return 0  # No edge — don't trade

        # Fractional Kelly
        position_frac = kelly_f * self.fraction

        # Vol-targeting cap: position_frac <= vol_target / daily_vol
        if daily_vol > 0:
            vol_cap_frac = self.vol_target_pct / daily_vol
            position_frac = min(position_frac, vol_cap_frac)

        # Hard cap
        position_frac = min(position_frac, self.max_position_pct)

        # Convert to units
        notional = equity * position_frac
        units = int(notional / price)
        return max(0, units)


# ── Model Health Monitor (Tier 2 #6) ─────────────────────────────────────────


class ModelHealthMonitor:
    """Tracks model prediction quality and fires degradation alerts.

    Monitors:
      1. Probability drift: z-score of live mean prob vs training baseline.
      2. Calibration drift: |predicted_win_rate - actual_win_rate| over window.
    """

    def __init__(
        self,
        training_mean_prob: float = 0.5,
        training_std_prob: float = 0.1,
        prob_drift_threshold: float = 2.0,
        calibration_threshold: float = 0.15,
        window: int = 100,
    ) -> None:
        self._train_mean = training_mean_prob
        self._train_std = training_std_prob
        self._prob_drift_threshold = prob_drift_threshold
        self._calibration_threshold = calibration_threshold
        self._probs: deque[float] = deque(maxlen=window)
        self._outcomes: deque[int] = deque(maxlen=window)  # 1=win, 0=loss
        self.degraded: bool = False
        self._alert_reason: str = ""

    def record_prediction(self, prob: float) -> None:
        self._probs.append(prob)

    def record_outcome(self, won: bool) -> None:
        self._outcomes.append(1 if won else 0)

    def check(self) -> bool:
        """Run health checks. Returns True if model is healthy."""
        self.degraded = False
        self._alert_reason = ""

        if len(self._probs) < 20:
            return True  # Not enough data

        live_mean = float(np.mean(list(self._probs)))

        # Check 1: Probability drift
        if self._train_std > 0:
            z = abs(live_mean - self._train_mean) / self._train_std
            if z > self._prob_drift_threshold:
                self.degraded = True
                self._alert_reason = (
                    f"Probability drift: z={z:.2f} (live_mean={live_mean:.3f},"
                    f" train_mean={self._train_mean:.3f})"
                )
                _logger.warning("[ModelHealth] %s", self._alert_reason)
                return False

        # Check 2: Calibration drift
        if len(self._outcomes) >= 20:
            predicted_wr = live_mean
            actual_wr = float(np.mean(list(self._outcomes)))
            deviation = abs(predicted_wr - actual_wr)
            if deviation > self._calibration_threshold:
                self.degraded = True
                self._alert_reason = (
                    f"Calibration drift: predicted_wr={predicted_wr:.3f},"
                    f" actual_wr={actual_wr:.3f}, gap={deviation:.3f}"
                )
                _logger.warning("[ModelHealth] %s", self._alert_reason)
                return False

        return True

    @property
    def alert_reason(self) -> str:
        return self._alert_reason


class MLSignalStrategyConfig(StrategyConfig):
    """Configuration for MLSignalStrategy."""

    model_path: str
    instrument_id: str
    bar_type: str  # e.g. "EUR/USD-H4"
    risk_pct: float = 0.02  # 2% risk per trade (fallback if Kelly returns 0)
    warmup_bars: int = 500  # Number of bars to load for history
    # FractionalKelly parameters (Tier 2 #6)
    kelly_fraction: float = 0.25  # Quarter-Kelly
    max_position_pct: float = 0.03  # Hard cap: 3% of equity per trade
    vol_target_pct: float = 0.01  # Daily vol contribution cap: 1%
    # Health monitor
    health_check_interval: int = 50  # Check every N bars


class MLSignalStrategy(Strategy):
    """ML-driven strategy with FractionalKelly sizing and health monitoring.

    Tier 2 #6: Replaces fixed lot size with quarter-Kelly position sizing,
    integrates PortfolioRiskManager, and monitors model health.
    """

    def __init__(self, config: MLSignalStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)

        # Load resources
        self.model = self._load_model(config.model_path)
        self.feature_config = load_feature_config(self.log)

        # History buffer (list of dicts, converted to DF for inference)
        self.history: list[dict] = []

        # FractionalKelly sizer (Tier 2 #6)
        self._sizer = FractionalKelly(
            fraction=config.kelly_fraction,
            max_position_pct=config.max_position_pct,
            vol_target_pct=config.vol_target_pct,
        )
        self._wl_tracker = WinLossTracker(prior_ratio=1.5, min_trades=10)

        # Model health monitor (Tier 2 #6)
        self._health = ModelHealthMonitor(
            training_mean_prob=0.5,
            training_std_prob=0.1,
            window=config.health_check_interval * 2,
        )
        self._bar_count: int = 0

        # Portfolio risk manager ID
        self._prm_id: str = ""

    def _load_model(self, path: str):
        """Load the trained .joblib model."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found at {p}")
        self.log.info(f"Loading model from {p}...")
        return joblib.load(p)

    def on_start(self) -> None:
        """Called when the strategy starts."""
        self.log.info("MLStrategy started (FractionalKelly + HealthMonitor).")

        # Register with portfolio risk manager
        self._prm_id = f"ml_{self.instrument_id.value.replace('/', '_')}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        # Warmup History
        self._warmup_history()

        # Subscribe to Bars
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Subscribed to bars: {self.bar_type}")

    def _warmup_history(self) -> None:
        """Load recent historical data from local Parquet to initialize indicators."""
        pair = self.instrument_id.value.replace("/", "_")
        bt = str(self.bar_type)
        gran = bt.split("-")[-1] if "-" in bt else "H4"

        project_root = Path(__file__).resolve().parents[3]
        parquet_path = project_root / "data" / f"{pair}_{gran}.parquet"

        if not parquet_path.exists():
            self.log.warning(f"Warmup file missing: {parquet_path}. Trading delayed.")
            return

        self.log.info(f"Loading warmup data from {parquet_path}...")
        try:
            df = pd.read_parquet(parquet_path)
            df = df.sort_index().tail(self.config.warmup_bars)

            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required_cols):
                self.log.error(f"Warmup data missing columns. Found: {list(df.columns)}")
                return

            df_reset = df.reset_index()
            for _, row in df_reset.iterrows():
                self.history.append(
                    {
                        "time": row.iloc[0],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )
            self.log.info(f"Warmed up with {len(self.history)} bars.")

        except Exception as e:
            self.log.error(f"Failed to load warmup data: {e}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a new bar closes."""
        self._bar_count += 1

        # 1. Update History
        self.history.append(
            {
                "time": bar.close_time_as_datetime(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

        max_bars = self.config.warmup_bars + 200
        if len(self.history) > max_bars:
            self.history = self.history[-max_bars:]

        # 2. Check Warmup
        if len(self.history) < 200:
            self.log.info(f"Warming up... ({len(self.history)}/200)")
            return

        # 3. Portfolio risk manager: update equity + check halt
        accounts = self.cache.accounts()
        if accounts:
            acct = accounts[0]
            ccys = list(acct.balances().keys())
            if ccys:
                equity = float(acct.balance_total(ccys[0]).as_double())
                portfolio_risk_manager.update(self._prm_id, equity)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch active — flattening.")
            self.close_all_positions(self.instrument_id)
            return

        # 4. Health check (periodic)
        if self._bar_count % self.config.health_check_interval == 0:
            healthy = self._health.check()
            if not healthy:
                self.log.warning(f"Model degraded: {self._health.alert_reason}. Halving sizes.")

        # 5. Calculate Features + Predict
        try:
            df = pd.DataFrame(self.history)
            df.set_index("time", inplace=True)

            context_data: dict[str, pd.DataFrame] = {}
            X = build_features(df, context_data, self.feature_config)
            latest_features = X.iloc[[-1]]

            signal = self.model.predict(latest_features)[0]

            # Get probability if model supports predict_proba
            prob = 0.5
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(latest_features)[0]
                # For signal=1 (long), use P(class=1); for signal=-1, use P(class=-1)
                if signal == 1 and len(proba) > 1:
                    prob = float(proba[1]) if len(proba) == 2 else float(proba[-1])
                elif signal == -1:
                    prob = float(proba[0])
                else:
                    prob = float(max(proba))

            self._health.record_prediction(prob)

            # 6. Execute
            self._execute_signal(signal, prob, bar.close)

        except Exception as e:
            self.log.error(f"Error in on_bar: {e}", exc_info=True)

    def _get_daily_vol(self) -> float:
        """Estimate daily return volatility from recent history."""
        if len(self.history) < 20:
            return 0.01  # Default 1%
        closes = [h["close"] for h in self.history[-60:]]
        returns = pd.Series(closes).pct_change().dropna()
        if len(returns) < 5:
            return 0.01
        return float(returns.std())

    def _execute_signal(self, signal: int, prob: float, price: Decimal) -> None:
        """Execute trades using FractionalKelly sizing."""
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None

        current_dir = 0
        if position and position.is_open:
            current_dir = 1 if position.side == OrderSide.BUY else -1

        self.log.info(
            f"Signal={signal} prob={prob:.3f} current={current_dir}"
            f" kelly_wl={self._wl_tracker.ratio:.2f}"
            f" scale={portfolio_risk_manager.scale_factor:.2f}"
        )

        if signal == 1:
            if current_dir == 1:
                return
            if current_dir == -1:
                self.close_all_positions(self.instrument_id)
            qty = self._compute_quantity(prob, price)
            if qty > 0:
                self._submit_market(OrderSide.BUY, qty)

        elif signal == -1:
            if current_dir == -1:
                return
            if current_dir == 1:
                self.close_all_positions(self.instrument_id)
            qty = self._compute_quantity(prob, price)
            if qty > 0:
                self._submit_market(OrderSide.SELL, qty)

        elif signal == 0:
            if current_dir != 0:
                self.close_all_positions(self.instrument_id)

    def _compute_quantity(self, prob: float, price: Decimal) -> int:
        """Compute position size via FractionalKelly + portfolio scale."""
        accounts = self.cache.accounts()
        if not accounts:
            return 0
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return 0

        equity = float(acct.balance_total(ccys[0]).as_double())
        px = float(price)
        daily_vol = self._get_daily_vol()

        # Kelly sizing
        units = self._sizer.size(
            prob_success=prob,
            wl_ratio=self._wl_tracker.ratio,
            equity=equity,
            daily_vol=daily_vol,
            price=px,
        )

        # Apply portfolio-level scale factor
        units = int(units * portfolio_risk_manager.scale_factor)

        # If health monitor flagged degradation, halve the size
        if self._health.degraded:
            units = max(1, units // 2)

        return units

    def _submit_market(self, side: OrderSide, units: int) -> None:
        """Submit a market order."""
        qty = Quantity.from_int(units)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.FOK,
        )
        self.submit_order(order)
        self.log.info(f"{'BUY' if side == OrderSide.BUY else 'SELL'} {qty} {self.instrument_id}")

    def on_position_closed(self, event) -> None:
        """Track win/loss for Kelly ratio and health monitor."""
        pnl = float(event.realized_pnl)
        self._wl_tracker.record(pnl)
        self._health.record_outcome(pnl > 0)
        self.log.info(
            f"Position closed: PnL={pnl:.2f}"
            f" WR={self._wl_tracker.win_rate:.1%}"
            f" W/L={self._wl_tracker.ratio:.2f}"
        )

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("MLStrategy stopped — flat.")
