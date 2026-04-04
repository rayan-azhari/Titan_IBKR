"""gold_macro/strategy.py -- Gold Macro Cross-Asset Strategy (GLD Daily).

3-component cross-asset gold signal validated through research backtest:
    OOS Sharpe +0.603, OOS/IS ratio 2.06 (April 2026).

Components:
    1. Real rate proxy: log(TIP/TLT) 20-day change (falling = gold bullish)
    2. Dollar weakness: DXY (or UUP) 20-day log-return (falling = gold bullish)
    3. Momentum: GLD close > SMA(200) (trend confirmation)

Entry: composite z-score > 0 AND momentum confirms -> LONG GLD.
Exit:  composite <= 0 OR price < SMA(slow_ma) OR ATR hard stop.
Sizing: vol-targeted (target_vol / realized_vol), capped.

Requires daily bar subscriptions for GLD, TIP, TLT, and context data for DXY.
Context instruments are fed via warmup parquet files (data/*_D.parquet).

Tier 3 #7 (April 2026).
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


class GoldMacroConfig(StrategyConfig):
    """Configuration for the Gold Macro strategy."""

    instrument_id: str  # GLD instrument (e.g. "GLD.ARCA")
    bar_type_d: str  # Daily bar type (e.g. "GLD.ARCA-1-DAY-LAST-EXTERNAL")
    ticker: str = "GLD"
    # Signal parameters (from research validation)
    slow_ma: int = 200
    real_rate_window: int = 20
    dollar_window: int = 20
    # Sizing
    vol_target_pct: float = 0.10  # 10% annualized vol target
    vol_ewma_span: int = 20
    max_leverage: float = 1.5
    # Risk
    stop_atr_mult: float = 2.0
    atr_period: int = 14
    warmup_bars: int = 252  # 1 year of daily bars


class GoldMacroStrategy(Strategy):
    """Gold Macro: cross-asset composite signal on GLD daily."""

    def __init__(self, config: GoldMacroConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_d = BarType.from_str(config.bar_type_d)

        # Price histories (daily closes)
        self._gld_closes: list[float] = []
        self._tip_closes: list[float] = []
        self._tlt_closes: list[float] = []
        self._dxy_closes: list[float] = []
        self._gld_highs: list[float] = []
        self._gld_lows: list[float] = []

        # Signal state
        self._composite_z: float = 0.0
        self._momentum: bool = False
        self._latest_atr: float | None = None
        self._entry_price: float | None = None
        self._stop_price: float | None = None

        # Expanding stats for z-score normalisation
        self._real_rate_vals: list[float] = []
        self._dollar_vals: list[float] = []

        self._prm_id: str = ""

    # -- Lifecycle -------------------------------------------------------------

    def on_start(self) -> None:
        self._prm_id = f"gold_macro_{self.config.ticker}"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

        self._warmup()
        self.subscribe_bars(self.bar_type_d)
        self.log.info(
            f"GoldMacro started | SMA={self.config.slow_ma}"
            f" | vol_target={self.config.vol_target_pct:.0%}"
            f" | bars warmed={len(self._gld_closes)}"
        )

    def _warmup(self) -> None:
        """Load daily parquet warmup for GLD + context instruments."""
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"

        for sym, buf in [
            ("GLD", self._gld_closes),
            ("TIP", self._tip_closes),
            ("TLT", self._tlt_closes),
            ("DXY", self._dxy_closes),
        ]:
            path = data_dir / f"{sym}_D.parquet"
            if not path.exists():
                self.log.warning(f"Warmup missing: {path}")
                continue
            try:
                df = pd.read_parquet(path).sort_index().tail(self.config.warmup_bars)
                for _, row in df.iterrows():
                    buf.append(float(row["close"]))
                    if sym == "GLD":
                        self._gld_highs.append(float(row.get("high", row["close"])))
                        self._gld_lows.append(float(row.get("low", row["close"])))
                self.log.info(f"  {sym}: {len(buf)} D bars loaded")
            except Exception as e:
                self.log.error(f"  {sym} warmup failed: {e}")

        # Seed expanding z-score stats from warmup
        self._seed_zscore_stats()

    def _seed_zscore_stats(self) -> None:
        """Pre-compute expanding mean/std for z-scores from warmup data."""
        n = min(
            len(self._gld_closes),
            len(self._tip_closes),
            len(self._tlt_closes),
            len(self._dxy_closes),
        )
        w_rr = self.config.real_rate_window
        w_d = self.config.dollar_window

        if n < max(w_rr, w_d) + 10:
            return

        for i in range(max(w_rr, w_d), n):
            # Real rate signal
            log_rr_now = math.log(self._tip_closes[i] / self._tlt_closes[i])
            log_rr_prev = math.log(self._tip_closes[i - w_rr] / self._tlt_closes[i - w_rr])
            self._real_rate_vals.append(-(log_rr_now - log_rr_prev))

            # Dollar signal
            dxy_log_ret = math.log(self._dxy_closes[i]) - math.log(self._dxy_closes[i - w_d])
            self._dollar_vals.append(-dxy_log_ret)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("GoldMacro stopped -- flat.")

    # -- Bar handler -----------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_d:
            return

        px = float(bar.close)
        self._gld_closes.append(px)
        self._gld_highs.append(float(bar.high))
        self._gld_lows.append(float(bar.low))

        # Trim buffers
        max_len = self.config.warmup_bars + 100
        for buf in [
            self._gld_closes,
            self._gld_highs,
            self._gld_lows,
            self._tip_closes,
            self._tlt_closes,
            self._dxy_closes,
        ]:
            if len(buf) > max_len:
                del buf[: len(buf) - max_len]

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
            self.close_all_positions(self.instrument_id)
            return

        portfolio_allocator.tick()

        # Need enough context data
        if (
            len(self._tip_closes) < self.config.real_rate_window + 5
            or len(self._dxy_closes) < self.config.dollar_window + 5
            or len(self._gld_closes) < self.config.slow_ma + 5
        ):
            return

        # Compute signal
        self._compute_signal()
        self._update_atr()

        # Position state
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        is_long = position is not None and position.is_open

        bar_time = unix_nanos_to_dt(bar.ts_event)

        # Check stop
        if is_long and self._stop_price is not None and px < self._stop_price:
            self.log.info(f"STOP HIT: price={px:.2f} < stop={self._stop_price:.2f}")
            self.close_all_positions(self.instrument_id)
            self._entry_price = None
            self._stop_price = None
            return

        # Exit: composite turns negative OR momentum lost
        if is_long and (self._composite_z <= 0 or not self._momentum):
            self.log.info(f"EXIT: composite_z={self._composite_z:.3f} momentum={self._momentum}")
            self.close_all_positions(self.instrument_id)
            self._entry_price = None
            self._stop_price = None
            return

        # Entry: composite > 0 AND momentum AND not already long
        if not is_long and self._composite_z > 0 and self._momentum:
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
                self._entry_price = px
                if self._latest_atr and self._latest_atr > 0:
                    self._stop_price = px - self.config.stop_atr_mult * self._latest_atr
                self.log.info(
                    f"ENTRY BUY {qty} GLD @ ~{px:.2f}"
                    f" | composite_z={self._composite_z:.3f}"
                    f" | stop={self._stop_price:.2f}"
                    f" | {bar_time.date()}"
                )

        # Status
        self.log.info(
            f"  GLD={px:.2f} composite_z={self._composite_z:+.3f}"
            f" mom={'UP' if self._momentum else 'DN'}"
            f" pos={'LONG' if is_long else 'FLAT'}"
        )

    # -- Signal ----------------------------------------------------------------

    def _compute_signal(self) -> None:
        """Compute the 3-component composite z-score."""
        w_rr = self.config.real_rate_window
        w_d = self.config.dollar_window

        # Component 1: Real rate (inverted change in log(TIP/TLT))
        tip_now = self._tip_closes[-1]
        tlt_now = self._tlt_closes[-1]
        tip_prev = self._tip_closes[-1 - w_rr]
        tlt_prev = self._tlt_closes[-1 - w_rr]

        log_rr_now = math.log(tip_now / tlt_now)
        log_rr_prev = math.log(tip_prev / tlt_prev)
        real_rate_signal = -(log_rr_now - log_rr_prev)
        self._real_rate_vals.append(real_rate_signal)

        # Component 2: Dollar weakness (inverted DXY log return)
        dxy_now = self._dxy_closes[-1]
        dxy_prev = self._dxy_closes[-1 - w_d]
        dollar_signal = -(math.log(dxy_now) - math.log(dxy_prev))
        self._dollar_vals.append(dollar_signal)

        # Z-score normalisation (expanding)
        rr_z = self._expanding_zscore(self._real_rate_vals)
        d_z = self._expanding_zscore(self._dollar_vals)

        # Component 3: Momentum
        sma_val = sum(self._gld_closes[-self.config.slow_ma :]) / self.config.slow_ma
        self._momentum = self._gld_closes[-1] > sma_val

        # Composite (average of z-scores)
        self._composite_z = (rr_z + d_z) / 2.0

    @staticmethod
    def _expanding_zscore(vals: list[float], min_obs: int = 60) -> float:
        """Compute z-score of latest value using expanding mean/std."""
        if len(vals) < min_obs:
            return 0.0
        arr = np.array(vals, dtype=float)
        mu = arr.mean()
        sigma = arr.std()
        if sigma < 1e-8:
            return 0.0
        return float((arr[-1] - mu) / sigma)

    def _update_atr(self) -> None:
        """Compute ATR from GLD OHLC buffers."""
        p = self.config.atr_period
        if len(self._gld_closes) < p + 1:
            return

        tr_sum = 0.0
        for i in range(-p, 0):
            h = self._gld_highs[i]
            low = self._gld_lows[i]
            prev_c = self._gld_closes[i - 1]
            tr = max(h - low, abs(h - prev_c), abs(low - prev_c))
            tr_sum += tr
        self._latest_atr = tr_sum / p

    # -- Sizing ----------------------------------------------------------------

    def _compute_size(self, price: float) -> int:
        """Vol-targeted position sizing."""
        if price <= 0 or len(self._gld_closes) < 20:
            return 0

        accounts = self.cache.accounts()
        if not accounts:
            return 0
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return 0
        equity = float(acct.balance_total(ccys[0]).as_double())

        # EWMA realized vol
        rets = pd.Series(self._gld_closes[-60:]).pct_change().dropna()
        if len(rets) < 10:
            return 0
        ewma_var = rets.ewm(span=self.config.vol_ewma_span, adjust=False).var().iloc[-1]
        ann_vol = math.sqrt(max(0.0, ewma_var) * 252)
        if ann_vol <= 0:
            return 0

        # Vol-targeted notional
        vol_scale = min(self.config.max_leverage, self.config.vol_target_pct / ann_vol)
        notional = equity * vol_scale

        # Apply allocator + portfolio scale
        alloc = portfolio_allocator.get_weight(self._prm_id)
        notional *= alloc * portfolio_risk_manager.scale_factor

        return max(0, int(notional / price))

    # -- Events ----------------------------------------------------------------

    def on_position_closed(self, event) -> None:
        self._entry_price = None
        self._stop_price = None
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
