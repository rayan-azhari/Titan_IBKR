"""orb_strategy.py
-----------------

Opening Range Breakout Strategy for NautilusTrader.
Implements the 5m ORB with Daily SMA50/RSI14 filters and 1% Equity Risk Bracket Orders.

Bracket Management:
  Uses order_factory.bracket() with entry_order_type=MARKET.
  NautilusTrader wires OTO (entry triggers TP+SL) and OCO (TP/SL cancel each other)
  contingency automatically — no manual counterpart tracking needed.
  Positions are flattened at 15:55 ET to avoid overnight gap risk.
"""

import tomllib
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

# Reuse indicators from the ML features module (which has SMA, RSI, ATR)
from titan.indicators.gaussian_filter import _gaussian_channel_kernel
from titan.strategies.ml.features import atr, rsi, sma

ET = ZoneInfo("America/New_York")


class ORBConfig(StrategyConfig):
    """Configuration for ORB Strategy."""

    instrument_id: str
    bar_type_5m: str
    bar_type_1d: str
    config_path: str = "config/orb_live.toml"
    risk_pct: float = 0.01  # 1% Risk per trade
    leverage_cap: float = 4.0
    warmup_bars_1d: int = 60  # For SMA50 and RSI14
    warmup_bars_5m: int = 100  # For ATR14


class ORBStrategy(Strategy):
    """Executes ORB Logic.

    1. Subscribes to 5M and 1D bars.
    2. Calculates Daily SMA50 and RSI14 (from 1D bars).
    3. Calculates 5M ATR14 (from 5M bars).
    4. Triggers if price breaks Opening Range High/Low with matching filters before Cutoff time.
    5. Submits bracket orders with exactly risk-adjusted position sizing.
    """

    def __init__(self, config: ORBConfig):
        super().__init__(config)

        # Identifier
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        # We need the naked ticker for looking up the TOML config
        self.ticker = (
            self.instrument_id.symbol.value.replace("/", "")
            .replace("USD", "")
            .replace(".", "")
            .strip()
        )

        # Load optimization params for this specific ticker
        self.toml_cfg = self._load_toml(config.config_path).get(self.ticker, {})
        if not self.toml_cfg:
            self.log.error(f"No configuration found for {self.ticker} in {config.config_path}!")

        # Strategy Parameters
        self.atr_multiplier = self.toml_cfg.get("atr_multiplier", 2.0)
        self.rr_ratio = self.toml_cfg.get("rr_ratio", 2.0)
        self.use_sma = self.toml_cfg.get("use_sma", True)
        self.use_rsi = self.toml_cfg.get("use_rsi", True)
        self.use_gauss = self.toml_cfg.get("use_gauss", False)

        # Parse times
        orb_window_str = self.toml_cfg.get("orb_window_end", "09:45")
        cutoff_str = self.toml_cfg.get("entry_cutoff", "11:00")

        self.orb_end_time = datetime.strptime(orb_window_str, "%H:%M").time()
        self.cutoff_time = datetime.strptime(cutoff_str, "%H:%M").time()

        # Bar Types
        self.bt_5m = BarType.from_str(config.bar_type_5m)
        self.bt_1d = BarType.from_str(config.bar_type_1d)

        # State Buffers
        self.history_1d = []
        self.history_5m = []

        # Indicators
        self.daily_sma = 0.0
        self.daily_rsi = 50.0
        self.current_atr = 0.0
        self.current_gauss_mid = 0.0

        # Daily Session State
        self.current_date = None
        self.or_high = float("-inf")
        self.or_low = float("inf")
        self.trade_taken_today = False
        self.orb_formed = False
        self.eod_flattened = False

        # No manual bracket tracking needed — NautilusTrader manages OTO/OCO contingency.

    def _load_toml(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            # For testing with Nautilus where path might be relative to execution dir
            project_root = Path(__file__).resolve().parents[3]
            p = project_root / path
            if not p.exists():
                raise FileNotFoundError(f"Config not found at {p}")

        with open(p, "rb") as fobj:
            return tomllib.load(fobj)

    def on_start(self):
        """Lifecycle: Strategy Started."""
        self.log.info(
            f"ORB Strategy Started for {self.ticker}. Parameters: "
            f"ATR:{self.atr_multiplier} RR:{self.rr_ratio} "
            f"SMA:{self.use_sma} Gauss:{self.use_gauss} RSI:{self.use_rsi} "
            f"ORB:{self.orb_end_time} Cutoff:{self.cutoff_time}"
        )

        # Subscribe
        self.subscribe_bars(self.bt_1d)
        self.subscribe_bars(self.bt_5m)

        # Warmup
        self._warmup_all()

    def _warmup_all(self):
        """Load history to pre-calculate daily indicators."""
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"

        # 1. Warmup Daily Data
        pair_str = self.instrument_id.symbol.value.replace("/", "_")
        parquet_1d = data_dir / f"{pair_str}_D.parquet"

        if parquet_1d.exists():
            self.log.info(f"Loading 1D warmup from {parquet_1d}")
            try:
                df_1d = pd.read_parquet(parquet_1d).sort_index().tail(self.config.warmup_bars_1d)
                for t, row in df_1d.iterrows():
                    self.history_1d.append(
                        {
                            "time": t,
                            "close": float(row["close"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                        }
                    )
                self._update_daily_indicators()
                self.log.info(f"[{self.ticker}] Daily warmup complete. SMA50: {self.daily_sma:.2f}")
            except Exception as e:
                self.log.error(f"Failed to load 1D parquet: {e}")
        else:
            self.log.warning(f"No Daily historical data found at {parquet_1d} for {self.ticker}")

        # 2. Warmup 5M Data
        parquet_5m = data_dir / f"{pair_str}_M5.parquet"
        if parquet_5m.exists():
            self.log.info(f"Loading 5M warmup from {parquet_5m}")
            try:
                df_5m = pd.read_parquet(parquet_5m).sort_index().tail(self.config.warmup_bars_5m)
                for t, row in df_5m.iterrows():
                    self.history_5m.append(
                        {
                            "time": t,
                            "close": float(row["close"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                        }
                    )
                self._update_5m_indicators()
                self.log.info(
                    f"[{self.ticker}] 5M warmup complete. ATR: {self.current_atr:.2f}, "
                    f"Gauss Mid: {self.current_gauss_mid:.2f}"
                )
            except Exception as e:
                self.log.error(f"Failed to load 5M parquet: {e}")
        else:
            self.log.warning(f"No 5M historical data found at {parquet_5m} for {self.ticker}")

    def on_bar(self, bar: Bar):
        """Lifecycle: New Bar Closed."""
        dt = unix_nanos_to_dt(bar.ts_event)

        if bar.bar_type == self.bt_1d:
            self._handle_daily_bar(bar, dt)
        elif bar.bar_type == self.bt_5m:
            self._handle_5m_bar(bar, dt)

    def _handle_daily_bar(self, bar: Bar, dt: datetime):
        """Process a Daily bar close to update SMA and RSI."""
        self.history_1d.append(
            {"time": dt, "close": float(bar.close), "high": float(bar.high), "low": float(bar.low)}
        )

        if len(self.history_1d) > self.config.warmup_bars_1d + 20:
            self.history_1d = self.history_1d[-self.config.warmup_bars_1d :]

        self._update_daily_indicators()

    def _handle_5m_bar(self, bar: Bar, dt: datetime):
        """Process a 5-minute bar."""
        self.history_5m.append(
            {"time": dt, "close": float(bar.close), "high": float(bar.high), "low": float(bar.low)}
        )

        if len(self.history_5m) > self.config.warmup_bars_5m + 20:
            self.history_5m = self.history_5m[-self.config.warmup_bars_5m :]

        self._update_5m_indicators()

        # Convert UTC timestamp to Eastern Time for ORB hour comparisons
        dt_et = dt.astimezone(ET)
        bar_date = dt_et.date()
        bar_time = dt_et.time()

        # New day reset
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.or_high = float("-inf")
            self.or_low = float("inf")
            self.trade_taken_today = False
            self.orb_formed = False
            self.eod_flattened = False

        # Form Opening Range (09:30 to orb_end_time)
        trading_start = time(9, 30)

        if trading_start <= bar_time <= self.orb_end_time:
            self.or_high = max(self.or_high, float(bar.high))
            self.or_low = min(self.or_low, float(bar.low))

            if bar_time == self.orb_end_time:
                self.orb_formed = True
                self.log.info(
                    f"[{self.ticker}] ORB Formed - High: {self.or_high:.2f}, Low: {self.or_low:.2f}"
                )

            return  # Still in the ORB window, no trading yet

        # EOD Flatten: Close all positions and cancel orders at 15:55 ET
        eod_time = time(15, 55)
        if bar_time >= eod_time and not self.eod_flattened:
            self._flatten_eod()
            return

        # Out of bounds check
        if not self.orb_formed or self.trade_taken_today:
            return

        # Time Cutoff check
        if bar_time > self.cutoff_time:
            return

        # ── Execution Logic ──
        c = float(bar.close)

        # Filter evaluation
        bull_sma = (c > self.daily_sma) if self.use_sma else True
        bear_sma = (c < self.daily_sma) if self.use_sma else True

        bull_rsi = (self.daily_rsi < 70) if self.use_rsi else True
        bear_rsi = (self.daily_rsi > 30) if self.use_rsi else True

        bull_gauss = (c > self.current_gauss_mid) if self.use_gauss else True
        bear_gauss = (c < self.current_gauss_mid) if self.use_gauss else True

        target_side = None

        if c > self.or_high and bull_sma and bull_rsi and bull_gauss:
            target_side = OrderSide.BUY
            self.log.info(
                f"[{self.ticker}] LONG Trigger: Close {c} > ORH {self.or_high}. "
                f"SMA={self.daily_sma:.2f}, RSI={self.daily_rsi:.2f}, "
                f"Gauss={self.current_gauss_mid:.2f}"
            )
        elif c < self.or_low and bear_sma and bear_rsi and bear_gauss:
            target_side = OrderSide.SELL
            self.log.info(
                f"[{self.ticker}] SHORT Trigger: Close {c} < ORL {self.or_low}. "
                f"SMA={self.daily_sma:.2f}, RSI={self.daily_rsi:.2f}, "
                f"Gauss={self.current_gauss_mid:.2f}"
            )

        if target_side:
            self._execute_bracket(target_side, Decimal(str(c)))

    def _flatten_eod(self):
        """Close all positions and cancel all orders at end of day."""
        self.eod_flattened = True
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info(
            f"[{self.ticker}] EOD Flatten @ 15:55 ET — Closed all positions and cancelled orders."
        )

    def on_order_submitted(self, event):
        """Fired when order is sent to the broker."""
        self.log.info(
            f"[{self.ticker}] ORDER SUBMITTED: {event.client_order_id}"
        )

    def on_order_accepted(self, event):
        """Fired when IB acknowledges the order."""
        self.log.info(
            f"[{self.ticker}] ORDER ACCEPTED: {event.client_order_id} "
            f"venue_id={event.venue_order_id}"
        )

    def on_order_rejected(self, event):
        """Fired when IB rejects the order — always log with ERROR level."""
        self.log.error(
            f"[{self.ticker}] ORDER REJECTED: {event.client_order_id} — {event.reason}"
        )
        self.trade_taken_today = False  # allow re-entry if the entry was rejected

    def on_order_filled(self, event):
        """Fired on every fill. NautilusTrader handles OTO/OCO cancellation automatically."""
        self.log.info(
            f"[{self.ticker}] ORDER FILLED: {event.client_order_id} "
            f"side={event.order_side.name} qty={event.last_qty} px={event.last_px} "
            f"commission={event.commission}"
        )

    def on_order_canceled(self, event):
        """Fired when an order is cancelled (by strategy, EOD flatten, or OCO counterpart)."""
        self.log.info(
            f"[{self.ticker}] ORDER CANCELLED: {event.client_order_id}"
        )

    def on_order_expired(self, event):
        """Fired when a DAY order expires at end of session."""
        self.log.warning(
            f"[{self.ticker}] ORDER EXPIRED: {event.client_order_id}"
        )

    def on_position_opened(self, event):
        """Fired when a new position is opened."""
        self.log.info(
            f"[{self.ticker}] POSITION OPENED: {event.position_id} "
            f"side={event.entry.name} qty={event.quantity} avg_px={event.avg_px_open:.2f}"
        )

    def on_position_changed(self, event):
        """Fired when a position is partially closed or size changes."""
        self.log.info(
            f"[{self.ticker}] POSITION CHANGED: {event.position_id} "
            f"qty={event.quantity} unrealized_pnl={event.unrealized_pnl}"
        )

    def on_position_closed(self, event):
        """Fired when a position is fully closed."""
        self.log.info(
            f"[{self.ticker}] POSITION CLOSED: {event.position_id} "
            f"realized_pnl={event.realized_pnl} duration={event.duration_ns // 1_000_000_000}s"
        )

    def _update_daily_indicators(self):
        """Update SMA50 and RSI14."""
        if len(self.history_1d) < 50:
            return

        df = pd.DataFrame(self.history_1d)
        close = df["close"]

        # Note: the features return a Series. We take the last value.
        sma_series = sma(close, 50)
        rsi_series = rsi(close, 14)

        self.daily_sma = float(sma_series.iloc[-1])
        self.daily_rsi = float(rsi_series.iloc[-1])

    def _update_5m_indicators(self):
        """Update ATR14."""
        if len(self.history_5m) < 14:
            return

        df = pd.DataFrame(self.history_5m)
        df.set_index("time", inplace=True)

        # feature.atr expects a DataFrame with High/Low/Close
        # Because we're using vectorbt internally, we format it nicely
        # atr() expects lowercase column names — call before renaming
        atr_series = atr(df, 14)
        if not atr_series.empty:
            self.current_atr = float(atr_series.iloc[-1])

        # Gaussian Channel uses uppercase — rename after atr is done
        df = df.rename(columns={"close": "Close", "high": "High", "low": "Low"})
        high_arr = df["High"].values
        low_arr = df["Low"].values
        close_arr = df["Close"].values
        # 144 period, 4 poles, 2.0 sigma
        _, _, mid_arr = _gaussian_channel_kernel(high_arr, low_arr, close_arr, 144.0, 4, 2.0)
        self.current_gauss_mid = float(mid_arr[-1])

    def _execute_bracket(self, side: OrderSide, fill_price: Decimal):
        """Submit a market-entry bracket order (entry → OTO → TP/SL as OCO pair)."""
        # 1. Size Position
        if self.current_atr <= 0:
            self.current_atr = self.or_high - self.or_low
            if self.current_atr <= 0:
                self.log.error(f"[{self.ticker}] Cannot calculate Risk (ATR=0). Aborting trade.")
                return

        accounts = self.cache.accounts()
        if not accounts:
            self.log.error(f"[{self.ticker}] No account found! Cannot size position.")
            return

        account = accounts[0]
        currencies = list(account.balances().keys())
        if not currencies:
            self.log.error(f"[{self.ticker}] Account has no balances. Cannot size position.")
            return
        equity = float(account.balance_total(currencies[0]).as_double())
        risk_amount = equity * self.config.risk_pct

        risk_per_share = self.current_atr * self.atr_multiplier
        raw_units = risk_amount / risk_per_share

        max_units = (equity * self.config.leverage_cap) / float(fill_price)
        units = int(min(raw_units, max_units))

        if units == 0:
            self.log.warning(f"[{self.ticker}] Sizing calculated 0 units. Trade skipped.")
            return

        qty = Quantity.from_int(units)

        # 2. Bracket Prices
        price_f = float(fill_price)
        if side == OrderSide.BUY:
            sl_price = price_f - risk_per_share
            tp_price = price_f + (risk_per_share * self.rr_ratio)
        else:
            sl_price = price_f + risk_per_share
            tp_price = price_f - (risk_per_share * self.rr_ratio)

        instrument = self.cache.instrument(self.instrument_id)
        precision = instrument.price_precision if instrument else 2

        # 3. Mark trade taken BEFORE submitting — prevents re-entry on the next bar
        #    if the order is still pending or the bar closes above/below the OR again.
        self.trade_taken_today = True

        # 4. Submit as a linked bracket via order_factory.bracket().
        #    NautilusTrader wires OTO (entry triggers TP+SL) and OCO (TP/SL cancel each
        #    other) automatically — no manual _bracket_pairs tracking needed.
        #    entry_order_type=MARKET uses TimeInForce.DAY (the only valid TIF for MKT on IB).
        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            sl_trigger_price=Price(round(sl_price, precision), precision=precision),
            tp_price=Price(round(tp_price, precision), precision=precision),
            entry_order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,  # entry TIF (not 'entry_time_in_force')
            tp_post_only=False,  # default True causes IB to reject TP in brackets
            tp_time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.submit_order_list(bracket)

        self.log.info(f"[{self.ticker}] Bracket submitted -> {side} {units} @ ~{price_f:.2f}")
        self.log.info(
            f"[{self.ticker}] SL={sl_price:.2f}  TP={tp_price:.2f}  "
            f"Risk/share={risk_per_share:.2f}  RR={self.rr_ratio}"
        )

    def on_stop(self):
        """Lifecycle: Strategy Stopped. Clean up all orders and positions."""
        self.cancel_all_orders(self.instrument_id)
        self.log.info(f"[{self.ticker}] ORB Strategy stopped — all orders cancelled.")
