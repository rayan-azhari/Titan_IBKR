"""pairs/strategy.py -- Equity Pairs Trading Strategy (Market-Neutral).

Trades the spread between two cointegrated instruments using z-score
mean reversion. The hedge ratio (beta) is re-estimated periodically
to handle structural drift.

Validated pair: GLD/EFA (OOS Sharpe +1.14, April 2026).

Logic:
    1. Spread = price_A - beta * price_B (OLS hedge ratio).
    2. Z-score = (spread - expanding_mean) / expanding_std.
    3. Entry: z > entry_z -> SHORT spread (sell A, buy B).
             z < -entry_z -> LONG spread (buy A, sell B).
    4. Exit: |z| < exit_z (spread reverted).
    5. Invalidation: |z| > max_z -> force close (spread blew out).
    6. Beta re-estimated every refit_window bars (~6 months).

The strategy submits two simultaneous orders (legs A and B) to achieve
market neutrality. Both legs are sized so that:
    notional_A ~= beta * notional_B

Tier 3 #12 (April 2026).
"""

from __future__ import annotations

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
from titan.risk.strategy_equity import StrategyEquityTracker, report_equity_and_check


class PairsConfig(StrategyConfig):
    """Configuration for the Pairs Trading strategy."""

    instrument_a: str  # e.g. "GLD.ARCA"
    instrument_b: str  # e.g. "EFA.ARCA"
    bar_type_a: str  # Daily bar for leg A
    bar_type_b: str  # Daily bar for leg B
    ticker_a: str = "GLD"  # For parquet warmup
    ticker_b: str = "EFA"
    # Signal parameters
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_z: float = 4.0
    refit_window: int = 126  # Re-estimate beta every ~6 months
    # Sizing
    risk_pct: float = 0.005  # 0.5% equity risk per trade
    max_leverage: float = 2.0  # Max combined notional / equity
    # Warmup
    warmup_bars: int = 504  # 2 years of daily bars
    initial_equity: float = 10_000.0
    base_ccy: str = "USD"


class PairsStrategy(Strategy):
    """Market-neutral pairs trading with walk-forward beta refit."""

    def __init__(self, config: PairsConfig) -> None:
        super().__init__(config)
        self.instrument_a = InstrumentId.from_str(config.instrument_a)
        self.instrument_b = InstrumentId.from_str(config.instrument_b)
        self.bar_type_a = BarType.from_str(config.bar_type_a)
        self.bar_type_b = BarType.from_str(config.bar_type_b)

        # Price histories
        self._closes_a: list[float] = []
        self._closes_b: list[float] = []
        self._bar_count: int = 0

        # Hedge ratio
        self._beta: float | None = None
        self._bars_since_refit: int = 0

        # Spread stats (expanding)
        self._spread_vals: list[float] = []

        # Position state
        self._position: int = 0  # +1 long spread, -1 short spread, 0 flat

        self._prm_id: str = ""
        self._last_bar_a: float | None = None
        self._last_bar_b: float | None = None
        # Bar synchronisation by date — the previous ``_bars_{a,b}_seen``
        # counters permanently de-synced the strategy if one leg had a
        # missing bar (holiday, data gap). Tracking by date lets both legs
        # fall out of sync for a day and then rejoin on the next common date.
        self._last_date_a: "pd.Timestamp | None" = None
        self._last_date_b: "pd.Timestamp | None" = None
        self._last_eval_date: "pd.Timestamp | None" = None
        self._equity_tracker: StrategyEquityTracker | None = None

    # -- Lifecycle -------------------------------------------------------------

    def on_start(self) -> None:
        self._prm_id = f"pairs_{self.config.ticker_a}_{self.config.ticker_b}"
        portfolio_risk_manager.register_strategy(self._prm_id, self.config.initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self.config.initial_equity,
            base_ccy=self.config.base_ccy,
        )

        self._warmup()
        self.subscribe_bars(self.bar_type_a)
        self.subscribe_bars(self.bar_type_b)
        self.log.info(
            f"Pairs started | {self.config.ticker_a}/{self.config.ticker_b}"
            f" | entry_z={self.config.entry_z}"
            f" | beta={'pending' if self._beta is None else f'{self._beta:.4f}'}"
        )

    def _warmup(self) -> None:
        """Load daily parquet warmup for both legs."""
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"

        for sym, buf in [
            (self.config.ticker_a, self._closes_a),
            (self.config.ticker_b, self._closes_b),
        ]:
            path = data_dir / f"{sym}_D.parquet"
            if not path.exists():
                self.log.warning(f"Warmup missing: {path}")
                continue
            try:
                df = pd.read_parquet(path).sort_index().tail(self.config.warmup_bars)
                for _, row in df.iterrows():
                    buf.append(float(row["close"]))
                self.log.info(f"  {sym}: {len(buf)} D bars loaded")
            except Exception as e:
                self.log.error(f"  {sym} warmup failed: {e}")

        # Initial beta estimate
        self._refit_beta()

    def on_stop(self) -> None:
        self._close_both_legs("strategy stop")
        self.log.info("Pairs stopped -- flat.")

    # -- Bar handler -----------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        from nautilus_trader.core.datetime import unix_nanos_to_dt as _u

        bar_date = _u(bar.ts_event).date()

        # Track which leg this bar belongs to
        if bar.bar_type == self.bar_type_a:
            self._last_bar_a = float(bar.close)
            self._closes_a.append(float(bar.close))
            self._last_date_a = bar_date
        elif bar.bar_type == self.bar_type_b:
            self._last_bar_b = float(bar.close)
            self._closes_b.append(float(bar.close))
            self._last_date_b = bar_date
        else:
            return

        # Evaluate only when both legs have delivered for this same date. If
        # one leg missed a session (holiday, data gap), we skip this date
        # and resume when both legs align again — rather than permanently
        # losing sync as the old counter-based gate did.
        if self._last_date_a != bar_date or self._last_date_b != bar_date:
            return
        if self._last_eval_date == bar_date:
            # Idempotent: a re-emitted bar must not double-evaluate.
            return
        self._last_eval_date = bar_date

        self._bar_count += 1

        # Trim buffers
        max_len = self.config.warmup_bars + 100
        if len(self._closes_a) > max_len:
            self._closes_a = self._closes_a[-max_len:]
        if len(self._closes_b) > max_len:
            self._closes_b = self._closes_b[-max_len:]

        # Portfolio risk: per-strategy tracker equity, explicit bar timestamp
        _, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
        if halted:
            self.log.warning("Portfolio kill switch -- flattening pair.")
            self._close_both_legs("kill switch")
            return

        portfolio_allocator.tick(now=_u(bar.ts_event).date())

        # Need enough data
        if len(self._closes_a) < 60 or len(self._closes_b) < 60:
            return

        # Periodic beta refit
        self._bars_since_refit += 1
        if self._bars_since_refit >= self.config.refit_window or self._beta is None:
            self._refit_beta()

        if self._beta is None or self._last_bar_a is None or self._last_bar_b is None:
            return

        # Compute spread and z-score
        spread = self._last_bar_a - self._beta * self._last_bar_b
        self._spread_vals.append(spread)

        if len(self._spread_vals) < 30:
            return

        arr = np.array(self._spread_vals, dtype=float)
        mu = arr.mean()
        sigma = arr.std()
        if sigma < 1e-8:
            return
        z = float((spread - mu) / sigma)

        # Signal logic
        self._evaluate(z)

        # Status
        self.log.info(
            f"  {self.config.ticker_a}/{self.config.ticker_b}"
            f" spread={spread:.4f} z={z:+.3f}"
            f" beta={self._beta:.4f} pos={self._position:+d}"
        )

    # -- Signal ----------------------------------------------------------------

    def _evaluate(self, z: float) -> None:
        """Z-score mean reversion logic."""
        entry_z = self.config.entry_z
        exit_z = self.config.exit_z
        max_z = self.config.max_z

        # Invalidation: force close
        if self._position != 0 and abs(z) > max_z:
            self.log.info(f"INVALIDATION: |z|={abs(z):.2f} > {max_z}")
            self._close_both_legs("invalidation")
            return

        # Exit: z reverted
        if self._position == 1 and z > -exit_z:
            self.log.info(f"EXIT long spread: z={z:+.3f}")
            self._close_both_legs("exit")
            return
        if self._position == -1 and z < exit_z:
            self.log.info(f"EXIT short spread: z={z:+.3f}")
            self._close_both_legs("exit")
            return

        # Entry
        if self._position == 0:
            if z > entry_z:
                # Spread too wide -> short spread (sell A, buy B)
                self._enter_pair(-1, z)
            elif z < -entry_z:
                # Spread too narrow -> long spread (buy A, sell B)
                self._enter_pair(1, z)

    # -- Order management ------------------------------------------------------

    def _enter_pair(self, direction: int, z: float) -> None:
        """Enter a pair trade (two simultaneous legs)."""
        if self._last_bar_a is None or self._last_bar_b is None or self._beta is None:
            return
        if self._equity_tracker is None:
            return

        equity = self._equity_tracker.current_equity()
        if equity <= 0:
            return

        # Risk budget per trade
        risk_notional = equity * self.config.risk_pct

        # Apply allocator + portfolio scale
        alloc = portfolio_allocator.get_weight(self._prm_id)
        risk_notional *= alloc * portfolio_risk_manager.scale_factor

        # Size: units_A such that notional_A ~= risk_notional
        # units_B = beta * units_A * (price_A / price_B) for dollar neutrality
        units_a = max(1, int(risk_notional / self._last_bar_a))
        units_b = max(1, int(abs(self._beta) * units_a * self._last_bar_a / self._last_bar_b))

        if units_a < 1 or units_b < 1:
            self.log.warning("Pair size too small -- skip.")
            return

        # direction=+1: long spread = BUY A, SELL B
        # direction=-1: short spread = SELL A, BUY B
        side_a = OrderSide.BUY if direction == 1 else OrderSide.SELL
        side_b = OrderSide.SELL if direction == 1 else OrderSide.BUY

        order_a = self.order_factory.market(
            instrument_id=self.instrument_a,
            order_side=side_a,
            quantity=Quantity.from_int(units_a),
            time_in_force=TimeInForce.DAY,
        )
        order_b = self.order_factory.market(
            instrument_id=self.instrument_b,
            order_side=side_b,
            quantity=Quantity.from_int(units_b),
            time_in_force=TimeInForce.DAY,
        )
        self.submit_order(order_a)
        self.submit_order(order_b)
        self._position = direction

        self.log.info(
            f"PAIR ENTRY {'LONG' if direction == 1 else 'SHORT'} spread"
            f" | z={z:+.3f}"
            f" | {side_a.name} {units_a} {self.config.ticker_a}"
            f" + {side_b.name} {units_b} {self.config.ticker_b}"
            f" | alloc={alloc:.1%}"
        )

    def _close_both_legs(self, reason: str) -> None:
        """Close positions in both instruments."""
        self.cancel_all_orders(self.instrument_a)
        self.cancel_all_orders(self.instrument_b)
        self.close_all_positions(self.instrument_a)
        self.close_all_positions(self.instrument_b)
        if self._position != 0:
            self.log.info(f"PAIR CLOSED ({reason})")
        self._position = 0

    # -- Beta estimation -------------------------------------------------------

    def _refit_beta(self) -> None:
        """OLS hedge ratio: A = beta * B + epsilon."""
        n = min(len(self._closes_a), len(self._closes_b))
        window = min(n, self.config.refit_window * 2)
        if window < 30:
            return

        a = np.array(self._closes_a[-window:], dtype=float)
        b = np.array(self._closes_b[-window:], dtype=float)

        # Simple OLS (no statsmodels dependency in titan/)
        b_mean = b.mean()
        a_mean = a.mean()
        cov = np.mean((b - b_mean) * (a - a_mean))
        var = np.mean((b - b_mean) ** 2)
        if var < 1e-12:
            return

        self._beta = float(cov / var)
        self._bars_since_refit = 0
        self.log.info(
            f"Beta refit: {self.config.ticker_a}/{self.config.ticker_b}"
            f" beta={self._beta:.4f} (window={window})"
        )

    # -- Events ----------------------------------------------------------------

    def on_position_closed(self, event) -> None:
        if self._equity_tracker is not None:
            try:
                pnl = float(event.realized_pnl.as_double())
                self._equity_tracker.on_position_closed(pnl, fx_to_base=1.0)
            except Exception as e:
                self.log.warning(f"tracker on_position_closed failed: {e}")
        self.log.info(f"LEG CLOSED: {event.instrument_id} PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
