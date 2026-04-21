"""bond_gold/strategy.py -- Bond->Gold Cross-Asset Momentum Strategy.

Uses IEF (intermediate bond) momentum to time GLD (gold) entries.
When bond momentum is positive (rates falling), go long gold.

WFO validated: Sharpe +1.17, 68% positive folds across 37 rolling folds.

Signal:
    1. Compute IEF 60-day log-return (bond momentum).
    2. Z-score normalise on expanding window.
    3. Long GLD when z-score > threshold (0.50).
    4. Hold for minimum hold_days (20).
    5. Exit when z-score drops below threshold after hold period.

Sizing: vol-targeted (target_vol / realized_vol), portfolio-scaled.

April 2026. Research: research/cross_asset/run_bond_equity_wfo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.research.metrics import BARS_PER_YEAR, ewm_vol_last
from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import StrategyEquityTracker, report_equity_and_check


class BondGoldConfig(StrategyConfig):
    """Configuration for Bond->Gold Momentum strategy."""

    instrument_id: str  # GLD (traded instrument)
    signal_instrument_id: str  # IEF (signal source)
    bar_type_d: str  # GLD daily bar type
    signal_bar_type_d: str  # IEF daily bar type
    ticker_gld: str = "GLD"
    ticker_ief: str = "IEF"
    lookback: int = 60  # Bond momentum lookback (days)
    threshold: float = 0.50  # Z-score entry threshold
    hold_days: int = 20  # Minimum holding period
    vol_target_pct: float = 0.10
    ewma_span: int = 20
    max_leverage: float = 1.5
    warmup_bars: int = 120
    initial_equity: float = 10_000.0  # Seed capital in base ccy
    base_ccy: str = "USD"
    # Z-score normalisation window. Research WFO uses IS-frozen stats (504d
    # per fold); live mirrors that with a rolling trailing window so the
    # threshold behaviour stays aligned. The old expanding-window z-score
    # (mean/std over *all* observed history) drifted further from backtest
    # behaviour the longer the strategy ran.
    zscore_window: int = 504


class BondGoldStrategy(Strategy):
    """Bond->Gold: long GLD when IEF momentum is positive."""

    def __init__(self, config: BondGoldConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.signal_instrument_id = InstrumentId.from_str(config.signal_instrument_id)
        self.bar_type_d = BarType.from_str(config.bar_type_d)
        self.signal_bar_type_d = BarType.from_str(config.signal_bar_type_d)

        self._gld_closes: list[float] = []
        self._ief_closes: list[float] = []
        self._z_scores: list[float] = []
        self._current_pos: int = 0  # 0=flat, 1=long
        self._bars_held: int = 0
        self._prm_id: str = ""
        self._equity_tracker: StrategyEquityTracker | None = None
        # Bar-synchronisation: IB does not guarantee GLD and IEF daily bars
        # arrive in the same callback order. The signal requires both legs
        # for the same calendar date before it fires — otherwise a GLD bar
        # delivered before IEF would use a one-day-stale IEF close.
        self._last_gld_date: pd.Timestamp | None = None
        self._last_ief_date: pd.Timestamp | None = None
        self._last_signal_date: pd.Timestamp | None = None

    def on_start(self) -> None:
        self._prm_id = f"bond_gold_{self.config.ticker_gld}"
        portfolio_risk_manager.register_strategy(self._prm_id, self.config.initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self.config.initial_equity,
            base_ccy=self.config.base_ccy,
        )

        self._warmup()
        self.subscribe_bars(self.bar_type_d)
        self.subscribe_bars(self.signal_bar_type_d)
        self.log.info(
            f"BondGold started | IEF LB={self.config.lookback}"
            f" | threshold={self.config.threshold}"
            f" | hold={self.config.hold_days}d"
            f" | GLD bars={len(self._gld_closes)}"
            f" | IEF bars={len(self._ief_closes)}"
        )

    def _warmup(self) -> None:
        """Load daily parquet for warmup."""
        project_root = Path(__file__).resolve().parents[3]
        for ticker, store in [
            (self.config.ticker_gld, self._gld_closes),
            (self.config.ticker_ief, self._ief_closes),
        ]:
            path = project_root / "data" / f"{ticker}_D.parquet"
            if not path.exists():
                self.log.warning(f"Warmup missing: {path}")
                continue
            df = pd.read_parquet(path).sort_index()
            tail = df.tail(self.config.warmup_bars)
            for _, row in tail.iterrows():
                store.append(float(row["close"]))
            self.log.info(f"  {ticker}: {len(store)} D bars loaded")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("BondGold stopped -- GLD flat.")

    def on_bar(self, bar: Bar) -> None:
        bar_date = unix_nanos_to_dt(bar.ts_event).date()

        # Collect IEF bars for signal
        if bar.bar_type == self.signal_bar_type_d:
            self._ief_closes.append(float(bar.close))
            if len(self._ief_closes) > self.config.warmup_bars + 200:
                self._ief_closes = self._ief_closes[-(self.config.warmup_bars + 200) :]
            self._last_ief_date = bar_date
            # IEF may deliver first or after GLD. If GLD already arrived for
            # this same date, run the signal now.
            if self._last_gld_date == bar_date:
                self._run_signal(bar)
            return

        if bar.bar_type != self.bar_type_d:
            return

        px = float(bar.close)
        self._gld_closes.append(px)
        if len(self._gld_closes) > self.config.warmup_bars + 200:
            self._gld_closes = self._gld_closes[-(self.config.warmup_bars + 200) :]
        self._last_gld_date = bar_date

        # Only run signal when both legs have delivered for this date. If IEF
        # is still pending, skip — the IEF on_bar will call _run_signal when
        # it arrives. Previously, an out-of-order GLD-before-IEF callback
        # would evaluate the signal with a one-day-stale IEF close.
        if self._last_ief_date != bar_date:
            return
        self._run_signal(bar)

    def _run_signal(self, bar: Bar) -> None:
        """Run the bond-momentum signal once both legs have reported for the bar date."""
        bar_date = unix_nanos_to_dt(bar.ts_event).date()
        # Idempotent: if we already ran for this date (e.g. a re-emitted bar)
        # skip to avoid double orders.
        if self._last_signal_date == bar_date:
            return
        self._last_signal_date = bar_date

        # Portfolio risk check (per-strategy equity + explicit timestamp)
        _, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
        if halted:
            self.log.warning("Portfolio kill switch -- flattening.")
            self.close_all_positions(self.instrument_id)
            return

        bar_ts = unix_nanos_to_dt(bar.ts_event)
        portfolio_allocator.tick(now=bar_ts.date())
        px = float(bar.close)

        # Need enough IEF bars for momentum
        if len(self._ief_closes) < self.config.lookback + 10:
            return

        # Compute IEF momentum z-score on a rolling trailing window that
        # matches the WFO IS length (zscore_window). The old code used an
        # expanding window — as history grew, the live threshold behaviour
        # drifted away from the fixed IS-length behaviour in the backtest.
        ief = self._ief_closes
        lb = self.config.lookback
        mom = math.log(ief[-1] / ief[-1 - lb])

        # Build momentum series for all bars we have enough history for,
        # truncated to the trailing zscore_window.
        first_valid = lb + 10
        all_moms = [math.log(ief[i] / ief[i - lb]) for i in range(first_valid, len(ief))]
        if len(all_moms) < 20:
            return
        window = min(self.config.zscore_window, len(all_moms))
        window_moms = np.asarray(all_moms[-window:], dtype=float)
        mu = float(window_moms.mean())
        sigma = float(window_moms.std())
        if sigma < 1e-8:
            return
        z = (mom - mu) / sigma

        # Position logic
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        is_long = position and position.is_open and str(position.side) == "LONG"

        if is_long:
            self._bars_held += 1

        # Exit: z drops below threshold after hold period
        if is_long and self._bars_held >= self.config.hold_days:
            if z <= self.config.threshold:
                self.close_all_positions(self.instrument_id)
                self._current_pos = 0
                self._bars_held = 0
                self.log.info(
                    f"EXIT: z={z:.2f} < {self.config.threshold} after {self._bars_held}d hold"
                )
                return

        # Entry: z > threshold when flat
        if not is_long and z > self.config.threshold:
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
                self._current_pos = 1
                self._bars_held = 0
                self.log.info(f"ENTRY BUY {qty} GLD @ ~{px:.2f} | IEF z={z:.2f}")

        self.log.info(
            f"  GLD={px:.2f} IEF_z={z:.2f}"
            f" pos={'LONG' if is_long else 'FLAT'}"
            f" held={self._bars_held}d"
        )

    def _compute_size(self, price: float) -> int:
        """Vol-targeted position sizing."""
        if len(self._gld_closes) < 20 or price <= 0:
            return 0
        if self._equity_tracker is None:
            return 0
        equity = self._equity_tracker.current_equity()
        if equity <= 0:
            return 0

        rets = pd.Series(self._gld_closes[-60:]).pct_change().dropna()
        if len(rets) < 10:
            return 0
        # Daily bars -> 252 per year.
        span = self.config.ewma_span
        rm_lambda = (span - 1.0) / (span + 1.0)
        ann_vol = ewm_vol_last(
            rets,
            lam=rm_lambda,
            periods_per_year=BARS_PER_YEAR["D"],
        )
        if ann_vol <= 0:
            return 0

        notional = equity * (self.config.vol_target_pct / ann_vol)
        notional = min(notional, equity * self.config.max_leverage)

        alloc = portfolio_allocator.get_weight(self._prm_id)
        notional *= alloc * portfolio_risk_manager.scale_factor

        return max(0, int(notional / price))

    def on_position_closed(self, event) -> None:
        self._current_pos = 0
        self._bars_held = 0
        if self._equity_tracker is not None:
            try:
                # GLD/IWB/CSPX etc are USD-quoted -- no FX conversion needed.
                pnl = float(event.realized_pnl.as_double())
                self._equity_tracker.on_position_closed(pnl, fx_to_base=1.0)
            except Exception as e:
                self.log.warning(f"tracker on_position_closed failed: {e}")
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
