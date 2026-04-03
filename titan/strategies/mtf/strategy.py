"""mtf_strategy.py
-----------------

Multi-Timeframe Confluence Strategy for NautilusTrader.
Validated in Round 4 backtesting — EUR/USD, 10yr IS/OOS, full friction pipeline.

Config: config/mtf_eurusd.toml
  MA type : WMA (Weighted Moving Average)
  Weights : D=0.55, H4=0.05, H1=0.10, W=0.30  (daily + weekly macro bias)
  OOS Sharpe (combined long+short): 1.943
  OOS CAGR   (signal-only):  ~8%/yr after swap costs

Exit logic:
  - Signal reversal only: score returns to neutral or flips direction.
  - No hard stop order. Position is held until the signal exits.

Sizing: 1% equity risk per atr_stop_mult × ATR (sets position size; no stop placed).
"""

import tomllib
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, PriceType, TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency, Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.strategies.ml.features import atr, ema, rsi, sma, wma


class MTFConfluenceConfig(StrategyConfig):
    """Configuration for MTF Strategy."""

    instrument_id: str
    bar_types: dict[str, str]  # Map: "H1": "EUR/USD.IDEALPRO-1-HOUR-MID-EXTERNAL"
    config_path: str = "config/mtf_eurusd.toml"
    risk_pct: float = 0.01
    leverage_cap: float = 5.0
    warmup_bars: int = 1000


class MTFConfluenceStrategy(Strategy):
    """Executes trades based on Multi-Timeframe Confluence.

    Round 4 validated logic (H1/H4/D/W, WMA, EUR/USD OOS Combined Sharpe 1.943):
    1. Subscribe to H1, H4, D, W bars.
    2. On each bar, update history and recalculate signal for that TF.
    3. Compute weighted confluence score.
    4. Entry: score crosses ±threshold on the *next* bar (no look-ahead).
    5. Exit: signal reversal only (score returns to neutral or flips direction).
    6. Sizing: 1% equity risk / (atr_stop_mult × ATR).
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

        # ATR size multiplier — used for position sizing only (no stop placed)
        self._atr_stop_mult: float = float(self.toml_cfg.get("atr_stop_mult", 2.5))

        # Track pending entry order IDs to clear _entry_pending on fill
        self._entry_order_ids: set = set()
        # Guard against multiple bars firing entries before position cache updates
        self._entry_pending: bool = False

        # Position inertia: only rebalance when score changes direction or
        # target size differs from current by > inertia threshold (reduces turnover).
        self._prev_score: float = 0.0
        self._position_inertia_pct: float = float(self.toml_cfg.get("position_inertia_pct", 0.10))

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

        # Register with portfolio risk manager
        self._prm_id = "mtf_confluence"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        self.log.info(
            f"Warmup complete. ATR size mult: {self._atr_stop_mult}x. Exit: signal reversal only."
        )

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

        # Portfolio risk manager: update equity + check halt
        accounts = self.cache.accounts()
        if accounts:
            acct = accounts[0]
            ccys = list(acct.balances().keys())
            if ccys:
                equity = float(acct.balance_total(ccys[0]).as_double())
                portfolio_risk_manager.update(self._prm_id, equity)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch active -- flattening.")
            for instrument_id in self.bar_type_map.values():
                self.close_all_positions(self.instrument_id)
            return

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

        # Continuous forecast: scale with conviction instead of binary +/-0.5.
        # MA component: normalized spread, tanh-capped to [-0.5, +0.5].
        # RSI component: linear (rsi-50)/50, range [-0.5, +0.5].
        # Total per-TF signal: [-1.0, +1.0].
        if last_slow != 0:
            ma_spread = (last_fast - last_slow) / abs(last_slow)
            # Compute rolling std of spread for normalization (use 20-bar window)
            if len(close) >= 20:
                spread_series = (s_fast - s_slow) / s_slow.abs().clip(lower=1e-10)
                spread_std = float(spread_series.tail(20).std())
                if spread_std > 1e-10:
                    ma_spread = ma_spread / spread_std
            ma_signal = float(np.tanh(ma_spread)) * 0.5
        else:
            ma_signal = 0.0
        rsi_signal = (last_rsi - 50.0) / 100.0  # range [-0.5, +0.5]

        self.signals[tf] = ma_signal + rsi_signal

    # ---------------------------------------------------------------------------
    # Confluence evaluation & execution
    # ---------------------------------------------------------------------------

    def _evaluate_confluence(self, price: Decimal) -> None:
        weights = self.toml_cfg.get("weights", {"H1": 0.1, "H4": 0.25, "D": 0.6, "W": 0.05})
        threshold = float(self.toml_cfg.get("confirmation_threshold", 0.10))
        exit_buffer = float(self.toml_cfg.get("exit_buffer", 0.0))

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

        self._log_status_dashboard(
            price, score, threshold, exit_buffer, signal_label, pos_label, weights
        )
        self._execute_bias(bias, score, exit_buffer, price)

    def _execute_bias(self, bias: int, score: float, exit_buffer: float, price: Decimal) -> None:
        instrument_id = self.instrument_id
        positions = self.cache.positions(instrument_id=instrument_id)
        position = positions[-1] if positions else None

        current_dir = 0
        if position and position.is_open:
            current_dir = 1 if str(position.side) == "LONG" else -1

        # Position inertia: skip if score hasn't changed direction and the
        # magnitude shift is < inertia threshold (reduces whipsaw turnover).
        score_delta = abs(score - self._prev_score)
        same_direction = (bias == current_dir) and (current_dir != 0)
        if same_direction and score_delta < self._position_inertia_pct:
            return  # position already aligned, change too small to rebalance

        # Conviction scalar: |score| / max_possible (1.0) for position sizing.
        # Higher conviction = larger position. Minimum 0.3 to avoid dust orders.
        conviction = max(0.3, min(1.0, abs(score)))

        self._prev_score = score

        if bias == 1:
            if current_dir == 1:
                return
            elif current_dir == -1:
                self.log.info("Signal Flip: Short -> Long.")
                self.cancel_all_orders(instrument_id)
                self.close_all_positions(instrument_id)
                self._open_position(OrderSide.BUY, price, conviction)
            else:
                self.log.info("Signal Entry: Long.")
                self._open_position(OrderSide.BUY, price, conviction)

        elif bias == -1:
            if current_dir == -1:
                return
            elif current_dir == 1:
                self.log.info("Signal Flip: Long -> Short.")
                self.cancel_all_orders(instrument_id)
                self.close_all_positions(instrument_id)
                self._open_position(OrderSide.SELL, price, conviction)
            else:
                self.log.info("Signal Entry: Short.")
                self._open_position(OrderSide.SELL, price, conviction)

        elif bias == 0:
            # With exit_buffer: hold through the neutral zone.
            # Exit long only when score crosses below -exit_buffer (not just neutral).
            # Exit short only when score crosses above +exit_buffer.
            if current_dir == 1 and score < -exit_buffer:
                self.log.info(f"Exit Long: score={score:+.3f} < -{exit_buffer}.")
                self.cancel_all_orders(instrument_id)
                self.close_all_positions(instrument_id)
            elif current_dir == -1 and score > exit_buffer:
                self.log.info(f"Exit Short: score={score:+.3f} > +{exit_buffer}.")
                self.cancel_all_orders(instrument_id)
                self.close_all_positions(instrument_id)

    # ---------------------------------------------------------------------------
    # Order management
    # ---------------------------------------------------------------------------

    def _open_position(self, side: OrderSide, price: Decimal, conviction: float = 1.0) -> None:
        """Size and submit market entry. Stop is placed in on_order_filled."""
        if self._entry_pending:
            self.log.debug("Entry already pending — skipping duplicate bar signal.")
            return
        if self.latest_atr is None or self.latest_atr == 0:
            self.log.warning("ATR not ready. Skipping trade.")
            return

        accounts = self.cache.accounts()
        if not accounts:
            self.log.error("No account in cache. Cannot size position.")
            return

        account = accounts[0]
        currencies = list(account.balances().keys())
        if not currencies:
            self.log.error("Account has no balances. Cannot size position.")
            return
        acct_ccy = currencies[0]
        equity_raw = float(account.balance_total(acct_ccy).as_double())

        # Convert to USD for consistent sizing; fall back to raw balance with warning.
        equity = equity_raw
        if str(acct_ccy) != "USD":
            try:
                usd = Currency.from_str("USD")
                fx = self.portfolio.exchange_rate(acct_ccy, usd, PriceType.MID)
                if fx and fx > 0:
                    equity = equity_raw * fx
                else:
                    raise ValueError("zero rate")
            except Exception:
                self.log.warning(f"No {acct_ccy}/USD FX rate cached; using raw balance for sizing.")

        # Conviction-scaled risk: stronger signal = larger position.
        risk_amount = equity * self.config.risk_pct * conviction
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

        # Apply portfolio-level scale factor (drawdown heat reduction)
        units *= portfolio_risk_manager.scale_factor

        if int(units) <= 0:
            self.log.warning("Calculated size is 0. Skipping trade.")
            return

        qty = Quantity.from_int(int(units))
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self._entry_order_ids.add(order.client_order_id)
        self._entry_pending = True
        self.submit_order(order)
        self.log.info(
            f"Submitted {side} {qty} | ATR={self.latest_atr:.5f} | stop_dist={stop_dist:.5f}"
        )

    def on_order_filled(self, event: OrderFilled) -> None:
        """Clear entry-pending state when entry fill arrives."""
        if event.instrument_id != self.instrument_id:
            return
        cid = event.client_order_id
        if cid in self._entry_order_ids:
            self._entry_order_ids.discard(cid)
            self._entry_pending = False
            self.log.info(f"Entry filled @ {event.last_px}.")

    def on_stop(self) -> None:
        """Cancel all open stops and close positions on shutdown."""
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("MTF Strategy stopped — orders cancelled, positions closed.")

    # ---------------------------------------------------------------------------
    # Order / position lifecycle event handlers
    # ---------------------------------------------------------------------------

    def on_order_submitted(self, event) -> None:
        self.log.info(f"ORDER SUBMITTED: {event.client_order_id}")

    def on_order_accepted(self, event) -> None:
        self.log.info(f"ORDER ACCEPTED: {event.client_order_id} venue={event.venue_order_id}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"ORDER REJECTED: {event.client_order_id} — {event.reason}")
        if event.client_order_id in self._entry_order_ids:
            self._entry_order_ids.discard(event.client_order_id)
            self._entry_pending = False

    def on_order_canceled(self, event) -> None:
        self.log.info(f"ORDER CANCELLED: {event.client_order_id}")

    def on_order_expired(self, event) -> None:
        self.log.warning(f"ORDER EXPIRED: {event.client_order_id}")

    def on_position_opened(self, event) -> None:
        self.log.info(
            f"POSITION OPENED: {event.position_id} "
            f"side={event.entry.name} qty={event.quantity} avg_px={event.avg_px_open:.5f}"
        )

    def on_position_changed(self, event) -> None:
        self.log.info(
            f"POSITION CHANGED: {event.position_id} "
            f"qty={event.quantity} unrealized_pnl={event.unrealized_pnl}"
        )

    def on_position_closed(self, event) -> None:
        self.log.info(
            f"POSITION CLOSED: {event.position_id} "
            f"realized_pnl={event.realized_pnl} "
            f"duration={event.duration_ns // 1_000_000_000}s"
        )

    # ---------------------------------------------------------------------------
    # Status dashboard
    # ---------------------------------------------------------------------------

    def _log_status_dashboard(
        self,
        price: Decimal,
        score: float,
        threshold: float,
        exit_buffer: float,
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
            f"  CONFLUENCE: {score:+.3f}  │  Entry: ±{threshold}  │  Exit buf: ±{exit_buffer}"
            f"  │  Signal: {signal_label}"
        )
        atr_str = f"{self.latest_atr:.5f}" if self.latest_atr else "pending"
        lines.append(f"  Position: {pos_label}  │  ATR(14)H1: {atr_str}")
        lines.append(sep)
        self.log.info("\n".join(lines))
