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
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager


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

    def on_start(self) -> None:
        self._prm_id = f"bond_gold_{self.config.ticker_gld}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

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
        # Collect IEF bars for signal
        if bar.bar_type == self.signal_bar_type_d:
            self._ief_closes.append(float(bar.close))
            if len(self._ief_closes) > self.config.warmup_bars + 200:
                self._ief_closes = self._ief_closes[-(self.config.warmup_bars + 200) :]
            return

        if bar.bar_type != self.bar_type_d:
            return

        px = float(bar.close)
        self._gld_closes.append(px)
        if len(self._gld_closes) > self.config.warmup_bars + 200:
            self._gld_closes = self._gld_closes[-(self.config.warmup_bars + 200) :]

        # Portfolio risk check
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

        # Need enough IEF bars for momentum
        if len(self._ief_closes) < self.config.lookback + 10:
            return

        # Compute IEF momentum z-score
        ief = self._ief_closes
        lb = self.config.lookback
        mom = math.log(ief[-1] / ief[-1 - lb])

        # Expanding z-score
        all_moms = []
        for i in range(lb + 10, len(ief)):
            all_moms.append(math.log(ief[i] / ief[i - lb]))
        if len(all_moms) < 20:
            return
        mu = np.mean(all_moms)
        sigma = np.std(all_moms)
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
        accounts = self.cache.accounts()
        if not accounts:
            return 0
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return 0
        equity = float(acct.balance_total(ccys[0]).as_double())

        rets = pd.Series(self._gld_closes[-60:]).pct_change().dropna()
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
        self._current_pos = 0
        self._bars_held = 0
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
