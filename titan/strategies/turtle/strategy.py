from enum import Enum
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.average_true_range import AverageTrueRange
from nautilus_trader.indicators.donchian_channel import DonchianChannel
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.strategy_equity import get_base_balance


class DirectionFlag(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"


class TurtleConfig(StrategyConfig):
    instrument_id: str
    bar_type: str
    entry_period: int = 45
    exit_period: int = 30
    atr_period: int = 20
    risk_pct: float = 0.01
    stop_atr_mult: float = 2.0

    # Gap/Earnings Risk Mitigations
    max_leverage: float = 1.5  # Reduced per audit to avoid overnight gaps
    flat_before_earnings: bool = False  # Stub for earnings-avoidance logic
    use_moc_fill: bool = True  # Submit orders on bar close vs open
    direction: str = "long_only"

    # Pyramiding & Trailing
    max_units: int = 4  # Max units allowed (Classic = 4, S2 conservative = 1)
    pyramid_atr_mult: float = 0.5  # Scale in every 0.5 ATR
    use_trailing_stop: bool = True  # Classic Turtle trails Hard Stop up upon pyramiding


class TurtleStrategy(Strategy):
    def __init__(self, config: TurtleConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.direction = DirectionFlag(config.direction)

        # Indicators
        self.donch_entry = DonchianChannel(period=config.entry_period)
        self.donch_exit = DonchianChannel(period=config.exit_period)
        self.atr = AverageTrueRange(period=config.atr_period)

        # Pyramiding state
        self._units_held = 0
        self._current_pos_dir = 0  # 1 for long, -1 for short
        self._last_entry_price: Optional[float] = None
        self._entry_atr: Optional[float] = None
        self._hard_stop_price: Optional[float] = None

        # To prevent lookahead, keep a 1-bar delay for indicators
        self._prev_entry_upper: float = 0.0
        self._prev_entry_lower: float = 0.0
        self._prev_exit_upper: float = 0.0
        self._prev_exit_lower: float = 0.0
        self._prev_atr: float = 0.0

        self._halted = False

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

        # Indicators update manually to control t-1 access vs current close
        self.log.info(
            f"Turtle Started | {self.instrument_id} | Dir={self.direction.value} "
            f"| Pyramiding up to {self.config.max_units} units."
        )

    def on_bar(self, bar: Bar) -> None:
        if self._halted:
            return

        if self.config.flat_before_earnings and self._is_imminent_earnings(bar):
            self.log.warning("Imminent earnings detected. Flattening positions.")
            self._close_all("Earnings Guard")
            self._halt_indicators_update(bar)
            return

        is_ready = (
            self.donch_entry.initialized and self.donch_exit.initialized and self.atr.initialized
        )

        if is_ready:
            self._evaluate_bar(bar)

        # Update indicators for NEXT bar evaluation (preserving strictly t-1 logic)
        self.donch_entry.handle_bar(bar)
        self.donch_exit.handle_bar(bar)
        self.atr.handle_bar(bar)

        # Cache values
        if self.donch_entry.initialized:
            self._prev_entry_upper = self.donch_entry.upper
            self._prev_entry_lower = self.donch_entry.lower
        if self.donch_exit.initialized:
            self._prev_exit_upper = self.donch_exit.upper
            self._prev_exit_lower = self.donch_exit.lower
        if self.atr.initialized:
            self._prev_atr = self.atr.value

    def _evaluate_bar(self, bar: Bar) -> None:
        close = float(bar.close)

        # 1. Manage Active Position
        if self._units_held > 0:
            if self._current_pos_dir == 1:
                # Check Exits (Hard Stop OR Donchian 30-day Exit)
                if close < self._hard_stop_price or close < self._prev_exit_lower:
                    self._close_all("Long Exit or Stop Hit")
                    return
                # Check Pyramiding
                elif self._units_held < self.config.max_units:
                    pyramid_threshold = self._last_entry_price + (
                        self.config.pyramid_atr_mult * self._entry_atr
                    )
                    if close > pyramid_threshold:
                        self._add_unit(1, close)

            elif self._current_pos_dir == -1:
                # Check Exits
                if close > self._hard_stop_price or close > self._prev_exit_upper:
                    self._close_all("Short Exit or Stop Hit")
                    return
                # Check Pyramiding
                elif self._units_held < self.config.max_units:
                    pyramid_threshold = self._last_entry_price - (
                        self.config.pyramid_atr_mult * self._entry_atr
                    )
                    if close < pyramid_threshold:
                        self._add_unit(-1, close)

        # 2. Look for Initial Entry
        else:
            if self.direction in [DirectionFlag.LONG_ONLY, DirectionFlag.BOTH]:
                if close > self._prev_entry_upper:
                    self._current_pos_dir = 1
                    self._add_unit(1, close)
                    return

            if self.direction in [DirectionFlag.SHORT_ONLY, DirectionFlag.BOTH]:
                if close < self._prev_entry_lower:
                    self._current_pos_dir = -1
                    self._add_unit(-1, close)
                    return

    def _add_unit(self, direction: int, price: float) -> None:
        """Calculate single Unit size and submit market order."""
        accounts = self.cache.accounts()
        if not accounts:
            return

        # Deterministic USD balance. The previous ``balances.keys()[0]`` anti-
        # pattern returned a non-deterministic currency on multi-ccy accounts
        # (audit finding #2) — if USD is absent, we bail rather than silently
        # size off EUR/JPY balance.
        equity = get_base_balance(accounts[0], "USD")
        if equity is None or equity <= 0:
            self.log.warning("Turtle: no USD balance on account; skipping unit.")
            return
        equity = float(equity)

        # 1 unit risk
        stop_dist = self._prev_atr * self.config.stop_atr_mult
        raw_units = (equity * self.config.risk_pct) / stop_dist

        # Cap leverage
        max_notional = equity * self.config.max_leverage

        # We only allocate a fraction of our max leverage per unit step
        # Technically max_leverage applies to the PORTFOLIO not just this unit,
        # so we ensure (existing_units + new_unit) * price <= max_notional

        current_notional = 0
        positions = self.cache.positions(instrument_id=self.instrument_id)
        if positions and positions[0].is_open:
            current_notional = float(positions[0].quantity.as_double()) * price

        remaining_notional = max(0, max_notional - current_notional)
        units = min(raw_units, remaining_notional / price)

        if int(units) <= 0:
            self.log.warning(
                f"Unit size 0 calculated. Notional cap reached? {current_notional}/{max_notional}"
            )
            return

        qty = Quantity.from_int(int(units))
        side = OrderSide.BUY if direction == 1 else OrderSide.SELL

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        # Update State
        self._units_held += 1
        self._last_entry_price = price
        if self._units_held == 1:
            self._entry_atr = self._prev_atr

        # Update Hard Stop (Trailing up slightly on pyramids)
        if self.config.use_trailing_stop:
            if direction == 1:
                self._hard_stop_price = price - (self.config.stop_atr_mult * self._entry_atr)
            else:
                self._hard_stop_price = price + (self.config.stop_atr_mult * self._entry_atr)

        self.log.info(
            f"Scaled IN [{self._units_held}/{self.config.max_units}]"
            f" | {side.value} {qty} @ ~{price:.2f}"
            f" | Stop={self._hard_stop_price:.2f}"
        )

    def _close_all(self, reason: str) -> None:
        positions = self.cache.positions(instrument_id=self.instrument_id)
        if positions and positions[-1].is_open:
            pos = positions[-1]
            side = OrderSide.SELL if str(pos.side) == "LONG" else OrderSide.BUY
            qty = pos.quantity
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.log.info(f"Closed ALL units ({self._units_held}) | Reason: {reason}")

        self._units_held = 0
        self._current_pos_dir = 0
        self._last_entry_price = None
        self._entry_atr = None
        self._hard_stop_price = None

    def _is_imminent_earnings(self, bar: Bar) -> bool:
        # STUB: In a live system, query an alternate data source or calendar
        return False

    def _halt_indicators_update(self, bar: Bar) -> None:
        self.donch_entry.handle_bar(bar)
        self.donch_exit.handle_bar(bar)
        self.atr.handle_bar(bar)
