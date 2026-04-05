"""mr_audjpy/strategy.py -- AUD/JPY MR + Confluence Regime Filter.

Intraday mean reversion on AUD/JPY using VWAP deviation grid entries
with multi-scale confluence disagreement as regime filter.

WFO validated: Sharpe +2.08, 75% positive folds.

Signal:
    1. Compute rolling VWAP (24-bar anchor = 1 day).
    2. Track price deviation from VWAP.
    3. Compute rolling percentile bands (95/98/99/99.9).
    4. Enter tiered grid [1,2,4,8] when deviation > percentile threshold.
    5. Regime gate: rsi_14_dev computed at H1/H4/D/W scales must DISAGREE
       (scales don't agree on direction = ranging = allow MR entries).
    6. Session filter: 07:00-12:00 UTC entries only.
    7. Exit: 50% reversion TP or 21:00 UTC hard close.

April 2026. Research: research/mean_reversion/run_confluence_regime_wfo.py
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

from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager

SCALE_MAP = {"H1": 1, "H4": 4, "D": 24, "W": 120}
TIERS_PCT = [0.95, 0.98, 0.99, 0.999]
TIER_SIZES = [1, 2, 4, 8]


class MRAUDJPYConfig(StrategyConfig):
    """Configuration for AUD/JPY MR + Confluence strategy."""

    instrument_id: str  # "AUD/JPY.IDEALPRO"
    bar_type_h1: str  # "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL"
    ticker: str = "AUD_JPY"
    vwap_anchor: int = 24  # Rolling VWAP anchor (H1 bars = 1 day)
    pct_window: int = 500  # Rolling percentile window
    reversion_pct: float = 0.50  # Exit at 50% reversion
    ny_close_hour: int = 21  # Hard close at 21:00 UTC
    entry_start_hour: int = 7  # Session entry start
    entry_end_hour: int = 12  # Session entry end
    vol_target_pct: float = 0.08
    ewma_span: int = 20
    max_leverage: float = 1.5
    warmup_bars: int = 1000


class MRAUDJPYStrategy(Strategy):
    """AUD/JPY MR with confluence disagreement regime filter."""

    def __init__(self, config: MRAUDJPYConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_h1 = BarType.from_str(config.bar_type_h1)

        self._closes: list[float] = []
        self._prm_id: str = ""

        # VWAP state
        self._vwap_sum: float = 0.0
        self._vwap_count: int = 0

        # Position state
        self._entry_price: float = 0.0
        self._position_size: float = 0.0
        self._tiers_hit: set[int] = set()

        # Confluence state (rsi_14_dev at each scale)
        self._rsi_devs: dict[str, float] = {}

    def on_start(self) -> None:
        self._prm_id = f"mr_audjpy_{self.config.ticker}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        self._warmup()
        self.subscribe_bars(self.bar_type_h1)
        self.log.info(
            f"MR AUD/JPY started | pct_window={self.config.pct_window}"
            f" | reversion={self.config.reversion_pct}"
            f" | bars={len(self._closes)}"
        )

    def _warmup(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "data" / f"{self.config.ticker}_H1.parquet"
        if not path.exists():
            self.log.warning(f"Warmup missing: {path}")
            return
        df = pd.read_parquet(path).sort_index()
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        tail = df.tail(self.config.warmup_bars)
        for _, row in tail.iterrows():
            self._closes.append(float(row["close"]))
        self.log.info(f"  {self.config.ticker}: {len(self._closes)} H1 bars")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("MR AUD/JPY stopped -- flat.")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_h1:
            return

        px = float(bar.close)
        self._closes.append(px)
        max_keep = self.config.warmup_bars + 2000
        if len(self._closes) > max_keep:
            self._closes = self._closes[-max_keep:]

        ts = unix_nanos_to_dt(bar.ts_event)
        hour = ts.hour

        # Portfolio risk
        accounts = self.cache.accounts()
        if accounts:
            acct = accounts[0]
            ccys = list(acct.balances().keys())
            if ccys:
                equity = float(acct.balance_total(ccys[0]).as_double())
                portfolio_risk_manager.update(self._prm_id, equity)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch -- flattening.")
            self._flatten()
            return

        portfolio_allocator.tick()

        if len(self._closes) < self.config.pct_window + 50:
            return

        # VWAP and deviation
        anchor = self.config.vwap_anchor
        vwap = np.mean(self._closes[-anchor:])
        deviation = (px - vwap) / max(abs(vwap), 1e-8)
        abs_dev = abs(deviation)

        # Percentile levels
        devs = [
            abs(
                (self._closes[i] - np.mean(self._closes[max(0, i - anchor) : i]))
                / max(abs(np.mean(self._closes[max(0, i - anchor) : i])), 1e-8)
            )
            for i in range(len(self._closes) - self.config.pct_window, len(self._closes))
        ]
        dev_arr = np.array(devs)
        levels = [float(np.percentile(dev_arr, p * 100)) for p in TIERS_PCT]

        # Check existing position
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        has_pos = position and position.is_open

        # Exit checks
        if has_pos:
            # Hard close at NY session
            if hour >= self.config.ny_close_hour:
                self._flatten()
                self.log.info(f"NY CLOSE: flatten @ {px:.5f}")
                return

            # Reversion TP: deviation < lowest tier level * reversion_pct
            if levels and abs_dev < levels[0] * (1 - self.config.reversion_pct):
                self._flatten()
                self.log.info(
                    f"REVERSION TP: dev={abs_dev:.6f}"
                    f" < {levels[0] * (1 - self.config.reversion_pct):.6f}"
                )
                return

        # Entry checks: session filter + regime gate
        in_session = self.config.entry_start_hour <= hour < self.config.entry_end_hour
        if not in_session:
            return

        # Confluence regime gate: rsi_14_dev must disagree across scales
        ranging = self._check_confluence_regime()
        if not ranging:
            return

        # Tiered grid entries
        if not has_pos:
            self._tiers_hit = set()

        for tier_idx, (level, size) in enumerate(zip(levels, TIER_SIZES)):
            if tier_idx in self._tiers_hit:
                continue
            if abs_dev > level:
                direction = OrderSide.SELL if deviation > 0 else OrderSide.BUY
                units = self._compute_size(px, size)
                if units > 0:
                    qty = Quantity.from_int(units)
                    order = self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=direction,
                        quantity=qty,
                        time_in_force=TimeInForce.GTC,
                    )
                    self.submit_order(order)
                    self._tiers_hit.add(tier_idx)
                    self.log.info(
                        f"TIER {tier_idx + 1} {'SELL' if direction == OrderSide.SELL else 'BUY'}"
                        f" {qty} @ ~{px:.5f} | dev={abs_dev:.6f} > {level:.6f}"
                    )

    def _check_confluence_regime(self) -> bool:
        """Return True if scales DISAGREE (ranging market = allow MR)."""
        if len(self._closes) < 120 * 21 + 100:
            return True  # Not enough data for W-scale, allow entries

        close_arr = np.array(self._closes)
        signs = []
        for label, mult in SCALE_MAP.items():
            rsi_p = 14 * mult
            if len(close_arr) < rsi_p + 10:
                continue
            # Fast RSI computation
            deltas = np.diff(close_arr[-(rsi_p + 1) :])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            alpha = 2.0 / (rsi_p + 1)
            avg_g = gains[0]
            avg_l = losses[0]
            for i in range(1, len(gains)):
                avg_g = alpha * gains[i] + (1 - alpha) * avg_g
                avg_l = alpha * losses[i] + (1 - alpha) * avg_l
            if avg_l < 1e-8:
                rsi_dev = 50.0
            else:
                rsi_dev = (100.0 - 100.0 / (1.0 + avg_g / avg_l)) - 50.0
            signs.append(np.sign(rsi_dev))

        if len(signs) < 2:
            return True

        # Ranging = scales disagree (not all same sign)
        all_pos = all(s > 0 for s in signs)
        all_neg = all(s < 0 for s in signs)
        return not (all_pos or all_neg)

    def _flatten(self) -> None:
        """Close all positions and reset state."""
        self.close_all_positions(self.instrument_id)
        self._tiers_hit = set()
        self._entry_price = 0.0
        self._position_size = 0.0

    def _compute_size(self, price: float, tier_mult: int) -> int:
        """Vol-targeted sizing per tier."""
        if len(self._closes) < 60 or price <= 0:
            return 0
        accounts = self.cache.accounts()
        if not accounts:
            return 0
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return 0
        equity = float(acct.balance_total(ccys[0]).as_double())

        rets = pd.Series(self._closes[-60:]).pct_change().dropna()
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

        # Scale by tier multiplier
        base_units = int(notional / price)
        return max(0, base_units * tier_mult // max(sum(TIER_SIZES), 1))

    def on_position_closed(self, event) -> None:
        self._tiers_hit = set()
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
