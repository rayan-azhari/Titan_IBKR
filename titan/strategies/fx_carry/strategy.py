"""fx_carry/strategy.py — FX Carry Trade with Momentum Filter.

Captures the interest rate differential (carry premium) on high-yielding
FX pairs, with an SMA trend filter for crash protection.

Logic:
    1. Daily bar close: compute SMA(sma_period) as trend filter.
    2. If carry is positive (long swap > 0) AND price > SMA: go LONG.
    3. If carry is negative (short swap > 0) AND price < SMA: go SHORT.
    4. Exit when price crosses SMA against position.
    5. Size: vol-targeted (target_vol / realized_vol * equity), halved if VIX > 25.

The carry direction is configured per instrument. AUD/JPY typically has
positive carry for longs (AUD rates > JPY rates). This is set via the
``carry_direction`` config (+1 = long carry, -1 = short carry).

Tier 3 #10 (April 2026).
"""

from __future__ import annotations

import math

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager


class FXCarryConfig(StrategyConfig):
    """Configuration for the FX Carry Trade strategy."""

    instrument_id: str  # e.g. "AUD/JPY.IDEALPRO"
    bar_type_d: str  # e.g. "AUD/JPY.IDEALPRO-1-DAY-MID-EXTERNAL"
    ticker: str  # e.g. "AUD_JPY" (for parquet file lookup)
    carry_direction: int = 1  # +1 = long carry, -1 = short carry
    sma_period: int = 50  # Trend filter lookback
    vol_target_pct: float = 0.08  # 8% annualized vol target per position
    ewma_span: int = 20  # EWMA span for realized vol
    vix_halve_threshold: float = 25.0  # Halve size when VIX > this
    warmup_bars: int = 60  # Min bars before trading


class FXCarryStrategy(Strategy):
    """FX Carry Trade: collect swap premium with trend crash protection."""

    def __init__(self, config: FXCarryConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_d = BarType.from_str(config.bar_type_d)

        self._closes: list[float] = []
        self._prm_id: str = ""
        self._current_dir: int = 0  # +1 long, -1 short, 0 flat

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._prm_id = f"carry_{self.config.ticker}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        self._warmup()
        self.subscribe_bars(self.bar_type_d)
        self.log.info(
            f"FXCarry started | {self.config.ticker}"
            f" | carry_dir={self.config.carry_direction:+d}"
            f" | SMA={self.config.sma_period}"
            f" | vol_target={self.config.vol_target_pct:.0%}"
        )

    def _warmup(self) -> None:
        """Load daily parquet for SMA warmup."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "data" / f"{self.config.ticker}_D.parquet"
        if not path.exists():
            self.log.warning(f"Warmup file missing: {path}")
            return

        try:
            df = pd.read_parquet(path).sort_index().tail(self.config.warmup_bars)
            for _, row in df.iterrows():
                self._closes.append(float(row["close"]))
            self.log.info(f"Warmed up with {len(self._closes)} D bars.")
        except Exception as e:
            self.log.error(f"Warmup failed: {e}")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info(f"FXCarry stopped — {self.config.ticker} flat.")

    # ── Bar handler ───────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_d:
            return

        px = float(bar.close)
        self._closes.append(px)
        if len(self._closes) > self.config.warmup_bars + 100:
            self._closes = self._closes[-(self.config.warmup_bars + 100) :]

        # Portfolio risk
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

        portfolio_allocator.tick()

        # Need enough bars for SMA
        if len(self._closes) < self.config.sma_period:
            return

        sma_val = sum(self._closes[-self.config.sma_period :]) / self.config.sma_period

        # Determine carry signal
        carry_dir = self.config.carry_direction
        above_sma = px > sma_val

        # Current position
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        pos_dir = 0
        if position and position.is_open:
            pos_dir = 1 if str(position.side) == "LONG" else -1

        # Signal logic:
        # carry_direction=+1: go long when price > SMA (trend confirms carry)
        # carry_direction=-1: go short when price < SMA
        target = 0
        if carry_dir == 1 and above_sma:
            target = 1
        elif carry_dir == -1 and not above_sma:
            target = -1

        # Exit: price crosses SMA against position
        if pos_dir != 0 and target != pos_dir:
            self.log.info(f"Exit {pos_dir:+d}: price={px:.5f} SMA={sma_val:.5f}")
            self.close_all_positions(self.instrument_id)
            self._current_dir = 0
            pos_dir = 0

        # Entry
        if target != 0 and pos_dir == 0:
            side = OrderSide.BUY if target == 1 else OrderSide.SELL
            units = self._compute_size(px)
            if units > 0:
                qty = Quantity.from_int(units)
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)
                self._current_dir = target
                self.log.info(
                    f"CARRY {'BUY' if target == 1 else 'SELL'} {qty}"
                    f" @ ~{px:.5f} | SMA={sma_val:.5f}"
                )

        # Status log
        self.log.info(
            f"  {self.config.ticker} @ {px:.5f} SMA={sma_val:.5f}"
            f" pos={pos_dir:+d} target={target:+d}"
        )

    # ── Sizing ────────────────────────────────────────────────────────────

    def _compute_size(self, price: float) -> int:
        """Vol-targeted position sizing, halved if VIX > threshold."""
        if len(self._closes) < 20 or price <= 0:
            return 0

        accounts = self.cache.accounts()
        if not accounts:
            return 0
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return 0
        equity = float(acct.balance_total(ccys[0]).as_double())

        # Realized vol (EWMA)
        rets = pd.Series(self._closes[-60:]).pct_change().dropna()
        if len(rets) < 10:
            return 0
        ewma_var = rets.ewm(span=self.config.ewma_span, adjust=False).var().iloc[-1]
        ann_vol = math.sqrt(max(0.0, ewma_var) * 252)

        if ann_vol <= 0:
            return 0

        # Vol-targeted notional
        target_vol = self.config.vol_target_pct
        notional = equity * (target_vol / ann_vol)

        # VIX halving (from portfolio risk manager)
        if portfolio_risk_manager._vix_level is not None:
            if portfolio_risk_manager._vix_level > self.config.vix_halve_threshold:
                notional *= 0.5

        # Apply allocator + portfolio scale
        alloc = portfolio_allocator.get_weight(self._prm_id)
        notional *= alloc * portfolio_risk_manager.scale_factor

        units = int(notional / price)
        return max(0, units)

    # ── Events ────────────────────────────────────────────────────────────

    def on_position_closed(self, event) -> None:
        self._current_dir = 0
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} — {event.reason}")
