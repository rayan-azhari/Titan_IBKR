"""gap_fade/strategy.py — Session Gap Fade Strategy (EUR/USD M5).

Fades overnight gaps at London open when the gap exceeds a volatility
threshold. Takes profit at 50% gap fill, hard closes at NY close.

Logic:
    1. At 07:00 UTC (London open), compute gap = open - prev_ny_close.
    2. If |gap| > gap_atr_mult * ATR(14, H1): enter opposite direction.
    3. TP at 50% gap fill (limit order).
    4. SL at gap_atr_mult * ATR(14, H1) from entry.
    5. Hard close all at 21:00 UTC (no overnight holds).

Signal confirmation (optional):
    AM/PM archetype signal from research/intraday_profiles/ can be fed via
    update_am_signal() to add a directional filter. When enabled, only take
    gap fades that align with the AM archetype prediction.

Tier 3 #11 (April 2026).
"""

from __future__ import annotations

from datetime import time, timezone
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager

# ── Session boundaries (UTC) ─────────────────────────────────────────────────

LONDON_OPEN = time(7, 0, tzinfo=timezone.utc)
NY_CLOSE = time(21, 0, tzinfo=timezone.utc)
ENTRY_WINDOW_END = time(7, 30, tzinfo=timezone.utc)  # Only enter in first 30 min


# ── Config ────────────────────────────────────────────────────────────────────


class GapFadeConfig(StrategyConfig):
    """Configuration for the Gap Fade strategy."""

    instrument_id: str  # e.g. "EUR/USD.IDEALPRO"
    bar_type_m5: str  # e.g. "EUR/USD.IDEALPRO-5-MINUTE-MID-EXTERNAL"
    risk_pct: float = 0.005  # 0.5% equity risk per trade
    gap_atr_mult: float = 1.5  # Min gap size in ATR multiples
    tp_fill_pct: float = 0.50  # Take profit at 50% gap fill
    stop_atr_mult: float = 2.0  # Stop loss in ATR multiples
    atr_period: int = 14
    warmup_bars: int = 300  # ~1 day of M5 bars
    use_am_filter: bool = False  # Enable AM/PM archetype confirmation


# ── Strategy ──────────────────────────────────────────────────────────────────


class GapFadeStrategy(Strategy):
    """Intraday gap fade at London open with EOD hard close."""

    def __init__(self, config: GapFadeConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_m5 = BarType.from_str(config.bar_type_m5)

        # Price history for ATR computation
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []

        # Session state
        self._ny_close_price: float | None = None  # Previous day's 21:00 close
        self._today_gap: float | None = None
        self._entered_today: bool = False
        self._prm_id: str = ""
        self._precision: int = 5  # Price precision for FX

        # AM archetype signal (optional filter)
        self._am_signal: int = 0  # +1 long bias, -1 short bias, 0 neutral

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._prm_id = f"gap_fade_{self.instrument_id.value.replace('/', '_')}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        self.subscribe_bars(self.bar_type_m5)
        self.log.info(
            f"GapFade started | {self.instrument_id}"
            f" | gap_mult={self.config.gap_atr_mult}"
            f" | tp_fill={self.config.tp_fill_pct:.0%}"
        )

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("GapFade stopped — flat.")

    # ── Public API for AM/PM filter ───────────────────────────────────────

    def update_am_signal(self, signal: int) -> None:
        """Feed the AM archetype directional signal.

        Call from an external actor or the runner script after computing
        the AM classification each morning.

        Args:
            signal: +1 (long bias), -1 (short bias), 0 (neutral/no signal).
        """
        self._am_signal = signal

    # ── Bar handler ───────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_m5:
            return

        px_close = float(bar.close)
        px_high = float(bar.high)
        px_low = float(bar.low)
        bar_time = unix_nanos_to_dt(bar.ts_event)
        bar_utc = bar_time.time().replace(tzinfo=timezone.utc)

        # Update price buffers
        self._highs.append(px_high)
        self._lows.append(px_low)
        self._closes.append(px_close)
        max_len = self.config.warmup_bars + 100
        if len(self._closes) > max_len:
            self._highs = self._highs[-max_len:]
            self._lows = self._lows[-max_len:]
            self._closes = self._closes[-max_len:]

        # Portfolio risk manager update
        accounts = self.cache.accounts()
        if accounts:
            acct = accounts[0]
            ccys = list(acct.balances().keys())
            if ccys:
                equity = float(acct.balance_total(ccys[0]).as_double())
                portfolio_risk_manager.update(self._prm_id, equity)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch — flattening.")
            self.close_all_positions(self.instrument_id)
            return

        # Tick the allocator
        portfolio_allocator.tick()

        # ── Track NY close (21:00 UTC) ────────────────────────────────────
        if bar_utc.hour == 21 and bar_utc.minute == 0:
            self._ny_close_price = px_close
            self._entered_today = False
            self._today_gap = None
            self._am_signal = 0  # Reset for next day
            return

        # ── EOD hard close (21:00 UTC) ────────────────────────────────────
        if bar_utc >= NY_CLOSE:
            positions = self.cache.positions(instrument_id=self.instrument_id)
            if positions and positions[-1].is_open:
                self.log.info("EOD hard close at 21:00 UTC.")
                self.cancel_all_orders(self.instrument_id)
                self.close_all_positions(self.instrument_id)
            return

        # ── London open gap detection (07:00-07:30 UTC) ───────────────────
        if (
            not self._entered_today
            and self._ny_close_price is not None
            and LONDON_OPEN <= bar_utc <= ENTRY_WINDOW_END
        ):
            atr = self._compute_atr()
            if atr is None or atr <= 0:
                return

            gap = px_close - self._ny_close_price
            self._today_gap = gap

            gap_threshold = self.config.gap_atr_mult * atr

            if abs(gap) >= gap_threshold:
                # AM filter check
                if self.config.use_am_filter and self._am_signal != 0:
                    # Gap up + AM says long -> don't fade (aligned with gap)
                    # Gap up + AM says short -> fade (confirms mean reversion)
                    if gap > 0 and self._am_signal == 1:
                        self.log.info(f"Gap +{gap:.5f} but AM={self._am_signal} aligns — skip.")
                        return
                    if gap < 0 and self._am_signal == -1:
                        self.log.info(f"Gap {gap:.5f} but AM={self._am_signal} aligns — skip.")
                        return

                self._enter_fade(gap, atr, bar.close)

    # ── Entry ─────────────────────────────────────────────────────────────

    def _enter_fade(self, gap: float, atr: float, price: Decimal) -> None:
        """Enter a gap fade: sell if gap up, buy if gap down."""
        # Fade direction: opposite of gap
        side = OrderSide.SELL if gap > 0 else OrderSide.BUY
        px = float(price)

        # Size: risk_pct of equity / stop distance
        accounts = self.cache.accounts()
        if not accounts:
            return
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return

        equity = float(acct.balance_total(ccys[0]).as_double())
        stop_dist = self.config.stop_atr_mult * atr
        if stop_dist <= 0:
            return

        raw_units = (equity * self.config.risk_pct) / stop_dist

        # Apply portfolio allocation + risk scaling
        alloc = portfolio_allocator.get_weight(self._prm_id)
        units = int(raw_units * alloc * portfolio_risk_manager.scale_factor)

        if units < 1:
            self.log.warning(f"Gap fade size < 1 (units={raw_units:.1f}) — skip.")
            return

        # TP at 50% gap fill
        tp_price = px - gap * self.config.tp_fill_pct  # Fade direction
        # SL at stop_atr_mult * ATR from entry
        if side == OrderSide.SELL:
            sl_price = px + stop_dist
        else:
            sl_price = px - stop_dist

        # Submit bracket order
        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=Quantity.from_int(units),
            entry_order_type=OrderType.MARKET,
            time_in_force=TimeInForce.FOK,
            tp_price=Price.from_str(f"{tp_price:.{self._precision}f}"),
            sl_trigger_price=Price.from_str(f"{sl_price:.{self._precision}f}"),
            tp_post_only=False,
            tp_time_in_force=TimeInForce.GTC,
            sl_time_in_force=TimeInForce.GTC,
        )
        self.submit_order_list(bracket)
        self._entered_today = True

        self.log.info(
            f"GAP FADE {'SELL' if side == OrderSide.SELL else 'BUY'}"
            f" {units} @ ~{px:.5f}"
            f" | gap={gap:+.5f} ({abs(gap) / atr:.1f}x ATR)"
            f" | TP={tp_price:.5f} SL={sl_price:.5f}"
            f" | alloc={alloc:.1%} scale={portfolio_risk_manager.scale_factor:.2f}"
        )

    # ── ATR computation ───────────────────────────────────────────────────

    def _compute_atr(self) -> float | None:
        """Compute ATR from M5 price buffers.

        Uses the last atr_period * 12 M5 bars (~14 hours) as a proxy for
        H1-equivalent ATR. This avoids needing a separate H1 bar subscription.
        """
        period = self.config.atr_period
        n = period * 12  # 12 M5 bars per hour -> period hours of data
        if len(self._closes) < n + 1:
            return None

        tr_sum = 0.0
        for i in range(-n, 0):
            h = self._highs[i]
            low = self._lows[i]
            prev_c = self._closes[i - 1]
            tr = max(h - low, abs(h - prev_c), abs(low - prev_c))
            tr_sum += tr

        return tr_sum / n

    # ── Event handlers ────────────────────────────────────────────────────

    def on_order_filled(self, event) -> None:
        if event.instrument_id != self.instrument_id:
            return
        self.log.info(f"FILLED: {event.order_side.name} @ {event.last_px}")

    def on_position_closed(self, event) -> None:
        self.log.info(
            f"CLOSED: PnL={event.realized_pnl} duration={event.duration_ns // 60_000_000_000}min"
        )

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} — {event.reason}")
