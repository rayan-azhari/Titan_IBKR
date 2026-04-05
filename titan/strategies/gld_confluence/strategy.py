"""gld_confluence/strategy.py -- GLD AND-Gated Multi-Scale Confluence.

Computes trend_mom signal at H1/H4/D/W virtual scales on single H1 stream.
Only enters when ALL 4 scales agree on direction (AND-gate).

WFO validated: Sharpe +1.46, 80% positive folds.

Signal:
    trend_mom = sign(ma_spread_5_20) * |rsi_14_dev| / 50
    Computed at 4 scales: H1 (native), H4 (x4), D (x24), W (x120).
    Weighted: H1=10%, H4=5%, D=55%, W=30%.
    AND-gate: only enter when all 4 scales agree on sign.
    Z-score normalised, entry at threshold.

April 2026. Research: research/ic_analysis/run_confluence_wfo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager

# Virtual TF scale multipliers (H1 bars per TF bar)
SCALE_MAP = {"H1": 1, "H4": 4, "D": 24, "W": 120}
TF_WEIGHTS = {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30}


class GLDConfluenceConfig(StrategyConfig):
    """Configuration for GLD AND-Gated Confluence strategy."""

    instrument_id: str  # "GLD.ARCA"
    bar_type_h1: str  # "GLD.ARCA-1-HOUR-LAST-EXTERNAL"
    ticker: str = "GLD"
    threshold: float = 0.75  # Z-score entry threshold
    exit_buffer: float = 0.10
    vol_target_pct: float = 0.10
    ewma_span: int = 20
    max_leverage: float = 1.5
    warmup_bars: int = 5000  # Need W-scale warmup (~3120 H1 bars)
    zscore_window: int = 5000  # Expanding z-score calibration window


class GLDConfluenceStrategy(Strategy):
    """GLD AND-gated multi-scale confluence on H1 bars."""

    def __init__(self, config: GLDConfluenceConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_h1 = BarType.from_str(config.bar_type_h1)

        self._closes: list[float] = []
        self._prm_id: str = ""
        self._current_dir: int = 0  # +1 long, -1 short, 0 flat
        self._confluence_scores: list[float] = []

    def on_start(self) -> None:
        self._prm_id = f"gld_confluence_{self.config.ticker}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        self._warmup()
        self.subscribe_bars(self.bar_type_h1)
        self.log.info(
            f"GLDConfluence started | threshold={self.config.threshold}"
            f" | warmup={len(self._closes)} H1 bars"
        )

    def _warmup(self) -> None:
        """Load H1 parquet for signal warmup."""
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "data" / f"{self.config.ticker}_H1.parquet"
        if not path.exists():
            self.log.warning(f"Warmup missing: {path}")
            return
        df = pd.read_parquet(path).sort_index()
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        tail = df.tail(self.config.warmup_bars)
        for _, row in tail.iterrows():
            self._closes.append(float(row["close"]))
        self.log.info(f"  {self.config.ticker}: {len(self._closes)} H1 bars")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("GLDConfluence stopped -- flat.")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_h1:
            return

        px = float(bar.close)
        self._closes.append(px)
        # Keep buffer for W-scale computation
        max_keep = self.config.warmup_bars + 2000
        if len(self._closes) > max_keep:
            self._closes = self._closes[-max_keep:]

        # Portfolio risk
        accounts = self.cache.accounts()
        if accounts:
            acct = accounts[0]
            ccys = list(acct.balances().keys())
            if ccys:
                equity = float(acct.balance_total(ccys[0]).as_double())
                portfolio_risk_manager.update(self._prm_id, equity)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch -- flattening.")
            self.close_all_positions(self.instrument_id)
            return

        portfolio_allocator.tick()

        # Need enough bars for W-scale computation
        min_bars = 120 * 200 + 100  # W slow_ma(200) * W_scale(120)
        if len(self._closes) < min(min_bars, self.config.warmup_bars):
            return

        # Compute trend_mom at each scale
        close_arr = np.array(self._closes)
        per_scale_signals = {}

        for label, mult in SCALE_MAP.items():
            sig = self._compute_trend_mom(close_arr, mult)
            if sig is not None:
                per_scale_signals[label] = sig

        if len(per_scale_signals) < 4:
            return

        # AND-gate: check if all scales agree on direction
        signs = [np.sign(v) for v in per_scale_signals.values()]
        all_positive = all(s > 0 for s in signs)
        all_negative = all(s < 0 for s in signs)

        if all_positive or all_negative:
            # Weighted sum
            score = sum(per_scale_signals[label] * TF_WEIGHTS[label] for label in per_scale_signals)
        else:
            score = 0.0

        # Z-score on expanding window
        self._confluence_scores.append(score)
        if len(self._confluence_scores) > self.config.zscore_window:
            self._confluence_scores = self._confluence_scores[-self.config.zscore_window :]

        if len(self._confluence_scores) < 100:
            return

        mu = np.mean(self._confluence_scores)
        sigma = np.std(self._confluence_scores)
        if sigma < 1e-8:
            return
        z = (score - mu) / sigma

        # Position logic
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        pos_dir = 0
        if position and position.is_open:
            pos_dir = 1 if str(position.side) == "LONG" else -1

        threshold = self.config.threshold
        exit_buf = self.config.exit_buffer

        # Exit: score flips past exit buffer
        if pos_dir != 0:
            if pos_dir > 0 and z < -exit_buf:
                self.close_all_positions(self.instrument_id)
                self._current_dir = 0
                self.log.info(f"EXIT LONG: z={z:.2f}")
                return
            if pos_dir < 0 and z > exit_buf:
                self.close_all_positions(self.instrument_id)
                self._current_dir = 0
                self.log.info(f"EXIT SHORT: z={z:.2f}")
                return

        # Entry
        if pos_dir == 0:
            if z >= threshold:
                units = self._compute_size(px)
                if units > 0:
                    qty = Quantity.from_int(units)
                    order = self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=OrderSide.BUY,
                        quantity=qty,
                        time_in_force=TimeInForce.DAY,
                    )
                    self.submit_order(order)
                    self._current_dir = 1
                    self.log.info(f"ENTRY BUY {qty} @ ~{px:.2f} | z={z:.2f}")
            elif z <= -threshold:
                units = self._compute_size(px)
                if units > 0:
                    qty = Quantity.from_int(units)
                    order = self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=OrderSide.SELL,
                        quantity=qty,
                        time_in_force=TimeInForce.DAY,
                    )
                    self.submit_order(order)
                    self._current_dir = -1
                    self.log.info(f"ENTRY SELL {qty} @ ~{px:.2f} | z={z:.2f}")

    def _compute_trend_mom(self, close: np.ndarray, scale: int) -> float | None:
        """Compute trend_mom signal at a given scale.

        trend_mom = sign(ma_spread_5_20) * |rsi_14_dev| / 50
        Periods are multiplied by scale factor.
        """
        fast_p = 5 * scale
        slow_p = 20 * scale
        rsi_p = 14 * scale

        if len(close) < slow_p + 50:
            return None

        # EMA-based MA spread (faster than WMA for live)
        alpha_fast = 2.0 / (fast_p + 1)
        alpha_slow = 2.0 / (slow_p + 1)
        fast_ema = close[-1]
        slow_ema = close[-1]
        for i in range(len(close) - 2, max(len(close) - slow_p * 3, -1), -1):
            fast_ema = alpha_fast * close[i] + (1 - alpha_fast) * fast_ema
            slow_ema = alpha_slow * close[i] + (1 - alpha_slow) * slow_ema
        # Recompute forward for proper EMA
        fast_ema = close[max(0, len(close) - slow_p * 3)]
        slow_ema = close[max(0, len(close) - slow_p * 3)]
        for i in range(max(0, len(close) - slow_p * 3) + 1, len(close)):
            fast_ema = alpha_fast * close[i] + (1 - alpha_fast) * fast_ema
            slow_ema = alpha_slow * close[i] + (1 - alpha_slow) * slow_ema

        if abs(slow_ema) < 1e-8:
            return None
        ma_spread = (fast_ema - slow_ema) / abs(slow_ema)

        # RSI
        deltas = np.diff(close[-(rsi_p + 1) :])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        alpha_rsi = 2.0 / (rsi_p + 1)
        avg_gain = gains[0]
        avg_loss = losses[0]
        for i in range(1, len(gains)):
            avg_gain = alpha_rsi * gains[i] + (1 - alpha_rsi) * avg_gain
            avg_loss = alpha_rsi * losses[i] + (1 - alpha_rsi) * avg_loss
        if avg_loss < 1e-8:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_dev = (rsi - 50.0) / 50.0  # [-1, +1]

        # trend_mom = sign(ma_spread) * |rsi_dev|
        return float(np.sign(ma_spread) * abs(rsi_dev))

    def _compute_size(self, price: float) -> int:
        """Vol-targeted position sizing."""
        if len(self._closes) < 60 or price <= 0:
            return 0
        accounts = self.cache.accounts()
        if not accounts:
            return 0
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return 0
        equity = float(acct.balance_total(ccys[0]).as_double())

        rets = pd.Series(self._closes[-60:]).pct_change().dropna()
        if len(rets) < 10:
            return 0
        ewma_var = rets.ewm(span=self.config.ewma_span, adjust=False).var().iloc[-1]
        ann_vol = math.sqrt(max(0.0, ewma_var) * 252)
        if ann_vol <= 0:
            return 0

        notional = equity * (self.config.vol_target_pct / ann_vol)
        notional = min(notional, equity * self.config.max_leverage)

        alloc = portfolio_allocator.get_weight(self._prm_id)
        notional *= alloc * portfolio_risk_manager.scale_factor

        return max(0, int(notional / price))

    def on_position_closed(self, event) -> None:
        self._current_dir = 0
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
