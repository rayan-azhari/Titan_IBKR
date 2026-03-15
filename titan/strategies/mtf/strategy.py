"""mtf_strategy.py
-----------------

Multi-Timeframe Confluence Strategy for NautilusTrader.
Validated in Round 3 backtesting (OOS Sharpe 2.936 at 2.5x ATR stop).

Exit logic:
  - Primary: signal reversal (confluence score crosses threshold)
  - Secondary: 2.5x ATR hard stop from entry price (placed as STOP_MARKET order)

Sizing: 1% equity risk per 2.5x ATR move (matches actual stop distance).
"""

import tomllib
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.strategies.ml.features import atr, ema, rsi, sma, wma


class MTFConfluenceConfig(StrategyConfig):
    """Configuration for MTF Strategy."""

    instrument_id: str
    bar_types: dict[str, str]  # Map: "H1": "EUR/USD.IDEALPRO-1-HOUR-MID-EXTERNAL"
    config_path: str = "config/mtf.toml"
    risk_pct: float = 0.01
    leverage_cap: float = 5.0
    warmup_bars: int = 1000


class MTFConfluenceStrategy(Strategy):
    """Executes trades based on Multi-Timeframe Confluence.

    Round 3 validated logic (H1/H4/D/W, SMA, OOS Sharpe 2.936):
    1. Subscribe to H1, H4, D, W bars.
    2. On each bar, update history and recalculate signal for that TF.
    3. Compute weighted confluence score.
    4. Entry: score crosses ±threshold.
    5. Primary exit: signal reversal (score returns to neutral or flips).
    6. Hard stop: 2.5x ATR stop_market order placed on entry fill.
    7. Sizing: 1% equity risk / (2.5 × ATR).
    """

    def __init__(self, config: MTFConfluenceConfig):
        super().__init__(config)

        self.toml_cfg = self._load_toml(config.config_path)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        self.bar_type_map: dict[BarType, str] = {}
        for tf, periodicity in config.bar_types.items():
            bt = BarType.from_str(periodicity)
            self.bar_type_map[bt] = tf

        self.history: dict[str, list] = {tf: [] for tf in ["M5", "H1", "H4", "D", "W"]}
        self.signals: dict[str, float] = {tf: 0.0 for tf in ["M5", "H1", "H4", "D", "W"]}
        self.indicator_state: dict[str, dict] = {
            tf: {"fast_ma": None, "slow_ma": None, "rsi": None}
            for tf in ["M5", "H1", "H4", "D", "W"]
        }

        self.latest_atr: float | None = None

        # ATR stop multiplier — read from TOML, default 2.5 (Round 3 optimal)
        self._atr_stop_mult: float = float(self.toml_cfg.get("atr_stop_mult", 2.5))

        # Track order IDs to distinguish entries from stops in on_order_filled
        self._entry_order_ids: set = set()
        self._stop_order_ids: set = set()

    def _load_toml(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found at {p}")
        with open(p, "rb") as fobj:
            return tomllib.load(fobj)

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def on_start(self) -> None:
        self.log.info("MTF Strategy Started. Warming up...")
        for bt in self.bar_type_map.keys():
            self.subscribe_bars(bt)
        self._warmup_all()
        self.log.info(f"Warmup complete. ATR stop mult: {self._atr_stop_mult}x. Ready for signals.")

    def _warmup_all(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"
        self.log.info(f"Looking for warmup data in: {data_dir}")

        for bt, tf in self.bar_type_map.items():
            spec = bt.spec
            agg = spec.aggregation
            interval = spec.step

            agg_str = str(agg)
            if hasattr(agg, "name"):
                agg_str = agg.name
            elif hasattr(agg, "value"):
                agg_str = str(agg.value)
            agg_str = agg_str.upper()

            suffix = "UNKNOWN"
            if "MINUTE" in agg_str:
                if interval == 5:
                    suffix = "M5"
            elif "HOUR" in agg_str:
                if interval == 1:
                    suffix = "H1"
                elif interval == 4:
                    suffix = "H4"
            elif "DAY" in agg_str:
                suffix = "D"
            elif "WEEK" in agg_str:
                suffix = "W"

            if suffix == "UNKNOWN":
                bt_str = str(bt).upper()
                if "5-MINUTE" in bt_str:
                    suffix = "M5"
                elif "1-HOUR" in bt_str:
                    suffix = "H1"
                elif "4-HOUR" in bt_str:
                    suffix = "H4"
                elif "1-DAY" in bt_str:
                    suffix = "D"
                elif "1-WEEK" in bt_str:
                    suffix = "W"

            if suffix == "UNKNOWN":
                self.log.warning(f"Unknown BarType: {bt} (Agg: {agg_str}, Step: {interval})")
                continue

            pair_str = self.instrument_id.symbol.value.replace("/", "_")
            parquet_path = data_dir / f"{pair_str}_{suffix}.parquet"

            if not parquet_path.exists():
                self.log.warning(f"Missing warmup file: {parquet_path}")
                continue

            self.log.info(f"Loading {tf} warmup from {parquet_path}")
            try:
                df = pd.read_parquet(parquet_path).sort_index().tail(self.config.warmup_bars)
            except Exception as e:
                self.log.error(f"Failed to load parquet: {e}")
                continue

            for t, row in df.iterrows():
                self.history[tf].append(
                    {
                        "time": t,
                        "close": float(row["close"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                    }
                )
            self._update_signal(tf)

        self.log.info("Warmup loop complete.")

    # ---------------------------------------------------------------------------
    # Bar handler
    # ---------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        tf = self.bar_type_map.get(bar.bar_type)
        if not tf:
            return

        self.history[tf].append(
            {
                "time": unix_nanos_to_dt(bar.ts_event),
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
            }
        )

        if len(self.history[tf]) > self.config.warmup_bars + 100:
            self.history[tf] = self.history[tf][-self.config.warmup_bars :]

        self._update_signal(tf)
        self._evaluate_confluence(price=bar.close)

    # ---------------------------------------------------------------------------
    # Signal calculation
    # ---------------------------------------------------------------------------

    def _update_signal(self, tf: str) -> None:
        data = self.history[tf]
        if len(data) < 50:
            return

        df = pd.DataFrame(data)
        close = df["close"]

        params = self.toml_cfg.get(tf, {})
        fast_p = params.get("fast_ma", 10)
        slow_p = params.get("slow_ma", 20)
        rsi_p = params.get("rsi_period", 14)
        ma_type = self.toml_cfg.get("ma_type", "SMA").upper()

        if ma_type == "EMA":
            s_fast = ema(close, fast_p)
            s_slow = ema(close, slow_p)
        elif ma_type == "WMA":
            s_fast = wma(close, fast_p)
            s_slow = wma(close, slow_p)
        else:
            s_fast = sma(close, fast_p)
            s_slow = sma(close, slow_p)

        r = rsi(close, rsi_p)

        if tf == "H1":
            df["high"] = pd.Series([d["high"] for d in data], index=df.index)
            df["low"] = pd.Series([d["low"] for d in data], index=df.index)
            atr_s = atr(df, 14)
            if not atr_s.empty:
                self.latest_atr = float(atr_s.iloc[-1])

        last_fast = s_fast.iloc[-1]
        last_slow = s_slow.iloc[-1]
        last_rsi = r.iloc[-1]

        self.indicator_state[tf] = {
            "fast_ma": last_fast,
            "slow_ma": last_slow,
            "rsi": last_rsi,
        }
        self.signals[tf] = (0.5 if last_fast > last_slow else -0.5) + (
            0.5 if last_rsi > 50 else -0.5
        )

    # ---------------------------------------------------------------------------
    # Confluence evaluation & execution
    # ---------------------------------------------------------------------------

    def _evaluate_confluence(self, price: Decimal) -> None:
        weights = self.toml_cfg.get("weights", {"H1": 0.1, "H4": 0.25, "D": 0.6, "W": 0.05})
        threshold = self.toml_cfg.get("confirmation_threshold", 0.10)

        score = sum(self.signals[tf] * w for tf, w in weights.items())

        if score >= threshold:
            bias, signal_label = 1, "LONG"
        elif score <= -threshold:
            bias, signal_label = -1, "SHORT"
        else:
            bias, signal_label = 0, "FLAT"

        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        pos_label = "FLAT"
        if position and position.is_open:
            pos_label = "LONG" if str(position.side) == "LONG" else "SHORT"

        self._log_status_dashboard(price, score, threshold, signal_label, pos_label, weights)
        self._execute_bias(bias, price)

    def _execute_bias(self, bias: int, price: Decimal) -> None:
        instrument_id = self.instrument_id
        positions = self.cache.positions(instrument_id=instrument_id)
        position = positions[-1] if positions else None

        current_dir = 0
        if position and position.is_open:
            current_dir = 1 if str(position.side) == "LONG" else -1

        if bias == 1:
            if current_dir == 1:
                return
            elif current_dir == -1:
                self.log.info("Signal Flip: Short -> Long.")
                self._cancel_stops()
                self.close_all_positions(instrument_id)
                self._open_position(OrderSide.BUY, price)
            else:
                self.log.info("Signal Entry: Long.")
                self._open_position(OrderSide.BUY, price)

        elif bias == -1:
            if current_dir == -1:
                return
            elif current_dir == 1:
                self.log.info("Signal Flip: Long -> Short.")
                self._cancel_stops()
                self.close_all_positions(instrument_id)
                self._open_position(OrderSide.SELL, price)
            else:
                self.log.info("Signal Entry: Short.")
                self._open_position(OrderSide.SELL, price)

        elif bias == 0:
            if current_dir != 0:
                self.log.info(f"Signal Neutral. Closing position (dir={current_dir}).")
                self._cancel_stops()
                self.close_all_positions(instrument_id)

    # ---------------------------------------------------------------------------
    # Order management
    # ---------------------------------------------------------------------------

    def _open_position(self, side: OrderSide, price: Decimal) -> None:
        """Size and submit market entry. Stop is placed in on_order_filled."""
        if self.latest_atr is None or self.latest_atr == 0:
            self.log.warning("ATR not ready. Skipping trade.")
            return

        accounts = self.cache.accounts()
        if not accounts:
            self.log.error("No account in cache. Cannot size position.")
            return

        account = accounts[0]
        equity_money = account.balance_total(Currency.from_str("USD"))
        equity = float(equity_money.as_double())

        risk_amount = equity * self.config.risk_pct
        # Size against actual stop distance so risk% is accurate
        stop_dist = self.latest_atr * self._atr_stop_mult
        if stop_dist == 0:
            return

        raw_units = risk_amount / stop_dist
        if float(price) > 0:
            max_units = (equity * self.config.leverage_cap) / float(price)
            units = min(raw_units, max_units)
        else:
            units = 0

        if int(units) <= 0:
            self.log.warning("Calculated size is 0. Skipping trade.")
            return

        qty = Quantity.from_int(int(units))
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.FOK,
        )
        self._entry_order_ids.add(order.client_order_id)
        self.submit_order(order)
        self.log.info(
            f"Submitted {side} {qty} | ATR={self.latest_atr:.5f} | stop_dist={stop_dist:.5f}"
        )

    def on_order_filled(self, event: OrderFilled) -> None:
        """Place ATR stop after entry fill; log when stop fires."""
        if event.instrument_id != self.instrument_id:
            return

        cid = event.client_order_id

        if cid in self._entry_order_ids:
            self._entry_order_ids.discard(cid)
            self._submit_atr_stop(
                entry_side=event.order_side,
                fill_px=event.last_px.as_double(),
                qty=event.last_qty.as_double(),
            )
        elif cid in self._stop_order_ids:
            self._stop_order_ids.discard(cid)
            self.log.info(f"ATR stop triggered. Fill @ {event.last_px}.")

    def _submit_atr_stop(self, entry_side: OrderSide, fill_px: float, qty: float) -> None:
        """Submit a GTC STOP_MARKET at fill_px ± atr_stop_mult × ATR."""
        if self.latest_atr is None or self.latest_atr == 0:
            self.log.warning("ATR not ready — stop not placed.")
            return

        dist = self._atr_stop_mult * self.latest_atr

        if entry_side == OrderSide.BUY:
            stop_px = fill_px - dist
            stop_side = OrderSide.SELL
        else:
            stop_px = fill_px + dist
            stop_side = OrderSide.BUY

        instrument = self.cache.instrument(self.instrument_id)
        prec = instrument.price_precision
        stop_price = Price.from_str(f"{stop_px:.{prec}f}")
        stop_qty = Quantity.from_int(int(qty))

        order = self.order_factory.stop_market(
            instrument_id=self.instrument_id,
            order_side=stop_side,
            quantity=stop_qty,
            trigger_price=stop_price,
            time_in_force=TimeInForce.GTC,
        )
        self._stop_order_ids.add(order.client_order_id)
        self.submit_order(order)
        self.log.info(f"ATR stop placed @ {stop_price} ({self._atr_stop_mult}x ATR = {dist:.5f})")

    def _cancel_stops(self) -> None:
        """Cancel all open ATR stop orders before signal-based close."""
        open_orders = self.cache.orders_open(instrument_id=self.instrument_id)
        for order in open_orders:
            if order.client_order_id in self._stop_order_ids:
                self.cancel_order(order)
                self._stop_order_ids.discard(order.client_order_id)
                self.log.info(f"Cancelled ATR stop {order.client_order_id}.")

    # ---------------------------------------------------------------------------
    # Status dashboard
    # ---------------------------------------------------------------------------

    def _log_status_dashboard(
        self,
        price: Decimal,
        score: float,
        threshold: float,
        signal_label: str,
        pos_label: str,
        weights: dict,
    ) -> None:
        sep = "═" * 55
        lines = [f"\n{sep}", f"  MTF STATUS @ Price: {float(price):.5f}", f"{'─' * 55}"]

        display_tfs = [tf for tf in ["W", "D", "H4", "H1", "M5"] if tf in weights]
        for tf in display_tfs:
            st = self.indicator_state.get(tf)
            if not st:
                continue
            sig = self.signals[tf]
            w = weights.get(tf, 0)
            if st["fast_ma"] is not None:
                ma_dir = "BULL" if st["fast_ma"] > st["slow_ma"] else "BEAR"
                rsi_val = f"{st['rsi']:.1f}"
            else:
                ma_dir = " ?? "
                rsi_val = " ? "
            lines.append(
                f"  {tf:>2}  │  MA: {ma_dir}  │  RSI: {rsi_val:>5}"
                f"  │  Sig: {sig:+.1f}  │  Wtd: {sig * w:+.3f}"
            )

        lines.append(f"{'─' * 55}")
        lines.append(
            f"  CONFLUENCE: {score:+.3f}  │  Threshold: ±{threshold}  │  Signal: {signal_label}"
        )
        atr_str = f"{self.latest_atr:.5f}" if self.latest_atr else "pending"
        lines.append(f"  Position: {pos_label}  │  ATR(14)H1: {atr_str}")
        lines.append(sep)
        self.log.info("\n".join(lines))
