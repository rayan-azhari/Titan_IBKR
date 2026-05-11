"""etf_trend/strategy.py -- NautilusTrader ETF Trend Strategy.

Slow-MA trend boundary + decel-composite entry timing. Long-only.

Logic:
  Entry (decel_positive): close > slow_MA AND decel >= 0
  Entry (asymmetric):     close > slow_MA (standard entry)
                          OR  close > fast_reentry_MA AND close > slow_MA * 0.90
                          (exits slow on slow_MA break; re-enters faster via fast_reentry_MA)
  Exit mode A: close < slow_MA (sole trend-reversal gate)
  Exit mode C: decel < exit_thresh
  Exit mode D: A OR C (whichever fires first)

Fast MA is NOT used as an exit gate -- slow MA is the only trend boundary.
fast_reentry_ma is ONLY used for asymmetric re-entry.

Execution flow (mirrors backtest .shift(1) logic):
  1. Daily bar closes at ~16:00 ET -> compute regime + decel composite -> cache.
  2. Timer fires at 15:30 ET the *next* trading day -> evaluate cached signals
     -> submit MOC delta if position change required.

Orders:
  - Entry/exit: MOC (TimeInForce.AT_THE_CLOSE) on SPY.ARCA.
  - Hard stop: STOP_MARKET GTC alongside open position (ATR x stop_mult).

Config driven by config/etf_trend_{instrument}.toml (written by Stage 4 pipeline).
"""

import tomllib
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce, TriggerType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.research.metrics import BARS_PER_YEAR, annualize_vol
from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import StrategyEquityTracker, get_base_balance

ET = ZoneInfo("America/New_York")

# 15:30 ET = 30-min buffer before IBKR's 15:40 MOC cutoff for ARCA
MOC_EVAL_HOUR = 15
MOC_EVAL_MINUTE = 30


class ETFTrendConfig(StrategyConfig):
    """Configuration for the ETF Trend Strategy."""

    instrument_id: str
    bar_type_1d: str
    config_path: str = "config/etf_trend_spy.toml"
    warmup_bars: int = 350  # enough for slow MA 300 + vol/ATR windows


class ETFTrendStrategy(Strategy):
    """Daily ETF Trend Strategy -- long-only, MOC execution on ARCA.

    Entry (decel_positive): close > slow_MA AND decel >= 0.
    Entry (decel_cross):    close > slow_MA AND decel crosses 0 from below.
    Exit mode A: close < slow_MA (sole trend-reversal gate).
    Exit mode C: decel < exit_thresh.
    Exit mode D: A OR C (whichever first).
    Sizing: decel scalar (dynamic) or fully invested (binary).
    """

    def __init__(self, config: ETFTrendConfig) -> None:
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bt_1d = BarType.from_str(config.bar_type_1d)

        cfg = self._load_toml(config.config_path)

        # Stage 1 -- MA params (fast MA removed; slow MA is sole trend boundary)
        self.ma_type: str = str(cfg.get("ma_type", "SMA"))
        self.slow_ma: int = int(cfg.get("slow_ma", 200))

        # Stage 2 -- decel signals
        self.decel_signals: list[str] = list(cfg.get("decel_signals", []))

        # Stage 3 -- exit/entry config
        self.entry_mode: str = str(cfg.get("entry_mode", "decel_positive"))
        self.fast_reentry_ma: int | None = (
            int(cfg["fast_reentry_ma"]) if cfg.get("fast_reentry_ma") else None
        )
        self.exit_mode: str = str(cfg.get("exit_mode", "D"))
        self.exit_decel_thresh: float = float(cfg.get("exit_decel_thresh", -0.3))
        self.exit_confirm_days: int = int(cfg.get("exit_confirm_days", 1))
        # 10% buffer -- block asymmetric re-entry in deep bear markets
        self._asymmetric_bear_buffer: float = 0.90

        # Stage 4 -- sizing config
        self.sizing_mode: str = str(cfg.get("sizing_mode", "dynamic_decel"))
        self.atr_stop_mult: float = float(cfg.get("atr_stop_mult", 4.0))
        # Vol-target leverage params (used when sizing_mode == "vol_target")
        self.vol_target: float = float(cfg.get("vol_target", 0.15))
        self.max_leverage: float = float(cfg.get("max_leverage", 1.0))

        # History buffer: list of {"close", "high", "low"} dicts
        self.history: list[dict[str, float]] = []

        # Cached signal state (set on bar close, consumed at MOC eval timer)
        self.cached_regime: bool = False
        self.cached_decel: float = 0.0
        self.cached_close: float = 0.0
        self._prev_decel: float = 0.0  # for decel_cross entry detection
        self.cached_above_fast_reentry: bool = False  # for asymmetric re-entry
        self._days_below_slow_ma: int = 0  # consecutive days below slow_ma (confirmation filter)

        # ATR stop order tracking
        self._stop_order_id: str | None = None

        # Per-strategy equity (populated in on_start)
        self._equity_tracker: StrategyEquityTracker | None = None
        self._initial_equity: float = float(cfg.get("initial_equity", 10_000.0))
        self._base_ccy: str = str(cfg.get("base_ccy", "USD"))

    # ── Config loading ─────────────────────────────────────────────────────

    def _load_toml(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            root = Path(__file__).resolve().parents[3]
            p = root / path
        if not p.exists():
            self.log.warning(f"Config not found at {p} — using defaults.")
            return {}
        with open(p, "rb") as fobj:
            return tomllib.load(fobj)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def on_start(self) -> None:
        fast_info = f" fast_reentry={self.fast_reentry_ma}" if self.fast_reentry_ma else ""
        self.log.info(
            f"ETF Trend Strategy Started -- {self.instrument_id} | "
            f"MA={self.ma_type}(slow={self.slow_ma}){fast_info} | "
            f"entry={self.entry_mode} | exit={self.exit_mode} | "
            f"sizing={self.sizing_mode} | atr_mult={self.atr_stop_mult} | "
            f"decel_sigs={self.decel_signals}"
        )
        self.subscribe_bars(self.bt_1d)
        self._warmup()

        # Register with portfolio risk manager
        symbol = self.instrument_id.symbol.value
        self._prm_id = f"etf_trend_{symbol}"
        portfolio_risk_manager.register_strategy(self._prm_id, self._initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self._initial_equity,
            base_ccy=self._base_ccy,
        )

    def _warmup(self) -> None:
        """Pre-populate history from the daily parquet file."""
        root = Path(__file__).resolve().parents[3]
        symbol = self.instrument_id.symbol.value  # e.g. "SPY"
        parquet = root / "data" / f"{symbol}_D.parquet"
        if not parquet.exists():
            self.log.warning(f"No warmup data at {parquet}. Indicators cold-starting.")
            return
        try:
            df = pd.read_parquet(parquet).sort_index().tail(self.config.warmup_bars)
            for _, row in df.iterrows():
                self.history.append(
                    {
                        "close": float(row["close"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                    }
                )
            self.log.info(f"Warmup complete: {len(self.history)} bars loaded.")
        except Exception as exc:
            self.log.error(f"Warmup failed: {exc}")

    # ── Bar handler ────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        """Fires on daily bar close (~16:00 ET).

        Computes and caches signals, then schedules a MOC evaluation timer
        for 15:30 ET the *next* trading day (the bar-shift equivalent of
        the .shift(1) anti-look-ahead applied in the backtest).
        """
        if bar.bar_type != self.bt_1d:
            return

        # Portfolio risk manager (explicit bar timestamp so daily vol math
        # uses the right calendar day rather than wall-clock at bar-replay).
        equity = self._account_equity()
        if equity > 0:
            portfolio_risk_manager.update(self._prm_id, equity, ts=bar.ts_event)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch active -- flattening.")
            self.close_all_positions(self.instrument_id)
            return

        # Append bar to history
        self.history.append(
            {
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
            }
        )
        if len(self.history) > self.config.warmup_bars + 20:
            self.history = self.history[-self.config.warmup_bars :]

        # Compute and cache today's signals
        self.cached_close = float(bar.close)
        self._prev_decel = self.cached_decel  # snapshot before update (for decel_cross)
        self.cached_regime = self._compute_regime()
        self.cached_decel = self._compute_decel_composite()
        self.cached_above_fast_reentry = self._compute_above_fast_reentry()
        # Update consecutive-days-below counter for confirmation filter
        if self.cached_regime:
            self._days_below_slow_ma = 0
        else:
            self._days_below_slow_ma += 1

        dt_et = unix_nanos_to_dt(bar.ts_event).astimezone(ET)
        self.log.info(
            f"Daily bar {dt_et.date()}: close={bar.close} "
            f"regime={self.cached_regime} decel={self.cached_decel:.3f}"
        )

        # Schedule MOC evaluation for next trading day at 15:30 ET
        next_trading_day = self._next_trading_day(dt_et.date())
        eval_dt = datetime(
            next_trading_day.year,
            next_trading_day.month,
            next_trading_day.day,
            MOC_EVAL_HOUR,
            MOC_EVAL_MINUTE,
            0,
            tzinfo=ET,
        )
        eval_ns = int(eval_dt.timestamp() * 1_000_000_000)
        alert_name = f"MOC_EVAL_{next_trading_day}"
        self.clock.set_time_alert_ns(name=alert_name, alert_time_ns=eval_ns)

    @staticmethod
    def _next_trading_day(from_date: date) -> date:
        """Return next Mon–Fri calendar day (does not account for US holidays)."""
        candidate = from_date + timedelta(days=1)
        while candidate.weekday() >= 5:  # Sat=5, Sun=6
            candidate += timedelta(days=1)
        return candidate

    # ── Timer handler ──────────────────────────────────────────────────────

    def on_time_event(self, event) -> None:  # noqa: ANN001
        """Fires at 15:30 ET the day after each bar close.

        Uses the cached signals from yesterday's close to evaluate whether
        a position change is required, then submits a MOC order.
        """
        if not str(event.name).startswith("MOC_EVAL"):
            return
        self._evaluate_and_order()

    # ── Core logic ─────────────────────────────────────────────────────────

    def _evaluate_and_order(self) -> None:
        """Determine desired position and submit MOC order if delta needed."""
        in_regime = self.cached_regime  # close > slow_ma
        decel_ok = self.cached_decel >= 0.0

        # Entry logic
        if self.entry_mode == "asymmetric":
            # Exit slow (slow_ma break), re-enter fast (fast_reentry_ma crossover).
            # Bear market filter: block re-entry if price is >10% below slow_ma.
            not_deep_bear = self.cached_close > (
                self._slow_ma_last() * self._asymmetric_bear_buffer
            )
            should_be_long = in_regime or (self.cached_above_fast_reentry and not_deep_bear)
        elif self.entry_mode == "decel_cross":
            # Only enter when decel CROSSES from negative to positive
            decel_crossing = decel_ok and self._prev_decel < 0.0
            should_be_long = in_regime and decel_crossing
        else:
            # decel_positive: above slow MA and decel non-negative
            should_be_long = in_regime and decel_ok

        # Exit overrides when currently long
        # sma_exit: True only after N consecutive days below slow_ma
        sma_exit = self._days_below_slow_ma >= self.exit_confirm_days
        if self._is_long():
            if self.exit_mode == "A":
                should_be_long = not sma_exit
            elif self.exit_mode == "C":
                should_be_long = self.cached_decel >= self.exit_decel_thresh
            else:  # D: confirmed SMA break OR immediate decel collapse
                should_be_long = (not sma_exit) and (self.cached_decel >= self.exit_decel_thresh)

        currently_long = self._is_long()

        if should_be_long and not currently_long:
            qty = self._size_position()
            if qty is not None and int(qty) > 0:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.AT_THE_CLOSE,
                )
                self.submit_order(order)
                self.log.info(
                    f"MOC BUY {qty} submitted | "
                    f"regime={self.cached_regime} decel={self.cached_decel:.3f}"
                )
        elif not should_be_long and currently_long:
            self._submit_flat_moc()

    def _submit_flat_moc(self) -> None:
        """Submit MOC sell to flatten position."""
        pos = self._open_position()
        if pos is None:
            return
        qty = Quantity.from_str(str(pos.quantity))
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
            time_in_force=TimeInForce.AT_THE_CLOSE,
        )
        self.submit_order(order)
        self.log.info(
            f"MOC SELL {qty} submitted — exiting | "
            f"regime={self.cached_regime} decel={self.cached_decel:.3f}"
        )

    def _submit_atr_stop(self, entry_price: float) -> None:
        """Place a GTC ATR stop order alongside the position (hard stop)."""
        closes = self._closes()
        highs = [h["high"] for h in self.history]
        lows = [h["low"] for h in self.history]
        atr_val = self._compute_atr_last(np.array(highs), np.array(lows), np.array(closes), 14)
        if atr_val <= 0:
            return
        stop_price = entry_price - atr_val * self.atr_stop_mult
        instrument = self.cache.instrument(self.instrument_id)
        precision = instrument.price_precision if instrument else 2

        stop_order = self.order_factory.stop_market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self._open_position().quantity,  # type: ignore[union-attr]
            trigger_price=Price(round(stop_price, precision), precision=precision),
            trigger_type=TriggerType.DEFAULT,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(stop_order)
        self._stop_order_id = stop_order.client_order_id.value
        self.log.info(
            f"ATR stop placed at {stop_price:.2f} "
            f"(entry={entry_price:.2f} ATR={atr_val:.2f} mult={self.atr_stop_mult})"
        )

    # ── Position helpers ───────────────────────────────────────────────────

    def _is_long(self) -> bool:
        pos = self._open_position()
        return pos is not None and pos.is_open

    def _open_position(self):  # noqa: ANN201
        positions = self.cache.positions_open(instrument_id=self.instrument_id)
        return positions[0] if positions else None

    # ── Sizing ─────────────────────────────────────────────────────────────

    def _compute_leverage(self) -> float:
        """Compute vol-targeted leverage from recent 20-day realized volatility.

        Returns a value in [0.5, max_leverage]. Falls back to 1.0 if history
        is too short.
        """
        closes = self._closes()
        if len(closes) < 22:
            return 1.0
        rets = pd.Series(closes[-21:]).pct_change().dropna()
        # Daily bars -> 252 per year. Route through shared metrics.
        realized_vol = annualize_vol(float(rets.std()), periods_per_year=BARS_PER_YEAR["D"])
        if realized_vol <= 1e-9:
            return 1.0
        return float(np.clip(self.vol_target / realized_vol, 0.5, self.max_leverage))

    def _size_position(self) -> Quantity | None:
        """Return target share count based on sizing mode and decel scalar.

        binary:        100% of equity deployed whenever in a trade.
        dynamic_decel: fraction scales with decel composite.
                       decel=+1 -> 100%, decel=0 -> 50%, decel=-1 -> 0%.
        vol_target:    leverage = clip(vol_target / realized_vol_20d, 0.5, max_leverage).
                       Target fraction can exceed 1.0 (leveraged) in calm markets.
        """
        closes = self._closes()
        if len(closes) < 2:
            return None

        current_price = closes[-1]
        equity = self._account_equity()
        if equity <= 0:
            return None

        if self.sizing_mode == "vol_target":
            target_fraction = self._compute_leverage()
        elif self.sizing_mode == "dynamic_decel":
            target_fraction = float(np.clip((self.cached_decel + 1) / 2, 0.0, 1.0))
        else:
            target_fraction = 1.0

        # Apply portfolio-level scale factor (drawdown heat reduction)
        target_fraction *= portfolio_risk_manager.scale_factor

        units = int((equity * target_fraction) / current_price)
        return Quantity.from_int(units) if units > 0 else None

    def _account_equity(self) -> float:
        """Per-strategy equity from the tracker (authoritative); account NLV
        is only used as a last-resort fallback before the tracker is wired."""
        if self._equity_tracker is not None:
            eq = self._equity_tracker.current_equity()
            if eq > 0:
                return float(eq)
        accounts = self.cache.accounts()
        if not accounts:
            return 0.0
        equity = get_base_balance(accounts[0], self._base_ccy)
        return float(equity) if equity is not None else 0.0

    # ── Indicator helpers ──────────────────────────────────────────────────

    def _closes(self) -> list[float]:
        return [h["close"] for h in self.history]

    def _ma(self, s: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "EMA":
            return s.ewm(span=period, adjust=False).mean()
        return s.rolling(period).mean()

    def _compute_regime(self) -> bool:
        """True when close > slow_MA (sole trend boundary)."""
        closes = self._closes()
        if len(closes) < self.slow_ma:
            return False
        s = pd.Series(closes)
        return bool(s.iloc[-1] > self._ma(s, self.slow_ma).iloc[-1])

    def _slow_ma_last(self) -> float:
        """Return the most recent slow MA value (for bear buffer calculation)."""
        closes = self._closes()
        if len(closes) < self.slow_ma:
            return float("inf")
        s = pd.Series(closes)
        return float(self._ma(s, self.slow_ma).iloc[-1])

    def _compute_above_fast_reentry(self) -> bool:
        """True when close > fast_reentry_MA (asymmetric mode only).

        Returns False if no fast_reentry_ma is configured.
        """
        if self.fast_reentry_ma is None:
            return False
        closes = self._closes()
        if len(closes) < self.fast_reentry_ma:
            return False
        s = pd.Series(closes)
        return bool(s.iloc[-1] > self._ma(s, self.fast_reentry_ma).iloc[-1])

    def _compute_decel_composite(self) -> float:
        """Weighted composite of selected decel signals ∈ [-1, 1].

        Returns 1.0 (always bullish) when no signals are configured.
        """
        if not self.decel_signals:
            return 1.0

        closes = self._closes()
        n = len(closes)
        if n < max(30, self.slow_ma):
            return 1.0

        s = pd.Series(closes)
        slow_series = self._ma(s, self.slow_ma)
        signals: list[float] = []

        if "d_pct" in self.decel_signals:
            d = (s.iloc[-1] - slow_series.iloc[-1]) / (slow_series.iloc[-1] + 1e-8)
            signals.append(float(np.tanh(d * 10)))

        if "rv_20" in self.decel_signals and n >= 21:
            rets = np.diff(closes[-21:]) / np.maximum(closes[-21:-1], 1e-8)
            rv = annualize_vol(float(np.std(rets)), periods_per_year=BARS_PER_YEAR["D"])
            signals.append(float(-np.tanh(max(rv - 0.15, 0) * 5)))

        if "adx_14" in self.decel_signals:
            h = np.array([x["high"] for x in self.history])
            lo = np.array([x["low"] for x in self.history])
            adx_val = self._compute_adx_last(h, lo, np.array(closes), 14)
            signals.append(float(np.tanh((adx_val - 25.0) / 15.0)))

        if "macd_hist" in self.decel_signals and n >= 35:
            ema_fast = s.ewm(span=12, adjust=False).mean()
            ema_slow = s.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            hist = float((macd_line - signal_line).iloc[-1])
            ref = float(s.iloc[-1]) or 1.0
            signals.append(float(np.tanh(hist / ref * 1000)))

        return float(np.mean(signals)) if signals else 1.0

    @staticmethod
    def _compute_adx_last(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Compute the last ADX value using Wilder smoothing."""
        n = len(close)
        if n < period * 2 + 1:
            return 20.0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        dm_plus = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0.0),
            0.0,
        ).astype(float)
        dm_minus = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0.0),
            0.0,
        ).astype(float)

        # Wilder initialisation
        atr_w = np.zeros(len(tr))
        dmp_w = np.zeros(len(tr))
        dmm_w = np.zeros(len(tr))
        atr_w[period - 1] = tr[:period].sum()
        dmp_w[period - 1] = dm_plus[:period].sum()
        dmm_w[period - 1] = dm_minus[:period].sum()
        for i in range(period, len(tr)):
            atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
            dmp_w[i] = dmp_w[i - 1] - dmp_w[i - 1] / period + dm_plus[i]
            dmm_w[i] = dmm_w[i - 1] - dmm_w[i - 1] / period + dm_minus[i]

        with np.errstate(divide="ignore", invalid="ignore"):
            di_plus = np.where(atr_w > 0, 100.0 * dmp_w / atr_w, 0.0)
            di_minus = np.where(atr_w > 0, 100.0 * dmm_w / atr_w, 0.0)
            denom = di_plus + di_minus
            dx = np.where(denom > 0, 100.0 * np.abs(di_plus - di_minus) / denom, 0.0)

        adx_arr = np.zeros(len(dx))
        start = 2 * period - 2
        if start >= len(dx):
            return 20.0
        adx_arr[start] = dx[period - 1 : 2 * period - 1].mean()
        for i in range(start + 1, len(dx)):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
        return float(adx_arr[-1])

    @staticmethod
    def _compute_atr_last(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Return the last ATR value using Wilder smoothing."""
        if len(close) < period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr_w = tr[:period].mean()
        for val in tr[period:]:
            atr_w = (atr_w * (period - 1) + val) / period
        return float(atr_w)

    # ── NautilusTrader event callbacks ─────────────────────────────────────

    def on_order_submitted(self, event) -> None:  # noqa: ANN001
        self.log.info(f"ORDER SUBMITTED: {event.client_order_id}")

    def on_order_accepted(self, event) -> None:  # noqa: ANN001
        self.log.info(f"ORDER ACCEPTED: {event.client_order_id} venue={event.venue_order_id}")

    def on_order_rejected(self, event) -> None:  # noqa: ANN001
        self.log.error(f"ORDER REJECTED: {event.client_order_id} — {event.reason}")

    def on_order_filled(self, event) -> None:  # noqa: ANN001
        self.log.info(
            f"ORDER FILLED: {event.client_order_id} "
            f"side={event.order_side.name} qty={event.last_qty} px={event.last_px}"
        )
        # Place ATR stop after a buy fill
        if event.order_side.name == "BUY":
            self._submit_atr_stop(float(event.last_px))

    def on_order_canceled(self, event) -> None:  # noqa: ANN001
        self.log.info(f"ORDER CANCELLED: {event.client_order_id}")

    def on_position_opened(self, event) -> None:  # noqa: ANN001
        self.log.info(
            f"POSITION OPENED: {event.position_id} "
            f"qty={event.quantity} avg_px={event.avg_px_open:.2f}"
        )

    def on_position_closed(self, event) -> None:  # noqa: ANN001
        if self._equity_tracker is not None:
            try:
                pnl = float(event.realized_pnl.as_double())
                self._equity_tracker.on_position_closed(pnl, fx_to_base=1.0)
            except Exception as e:
                self.log.warning(f"tracker on_position_closed failed: {e}")
        self.log.info(f"POSITION CLOSED: {event.position_id} realized_pnl={event.realized_pnl}")
        self._stop_order_id = None

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("ETF Trend Strategy stopped — orders cancelled, positions closed.")
