"""mr_fx/strategy.py — EUR/USD Intraday Mean Reversion via Session-Anchored VWAP.

NautilusTrader live strategy ported from the research pipeline at
research/mean_reversion/ (Stages 1-3 validated).

Signal logic:
  1. Compute session-anchored VWAP (resets at London 07:00, NY 13:00 UTC).
  2. Track price deviation from VWAP.
  3. Compute rolling percentile bands of the deviation distribution.
  4. Enter tiered grid positions when deviation crosses percentile thresholds.
  5. Exit on partial reversion (50% back to VWAP), hard invalidation (99.99th pct),
     or NY session close (21:00 UTC).

Regime gate: ATR percentile < 30th over 500-bar window (low-vol = ranging regime).
Session filter: London core (07:00-12:00 UTC) for entries only.

Config: config/mr_fx_eurusd.toml
Research: research/mean_reversion/ (signals.py, regime.py, risk.py, execution.py)
"""

from __future__ import annotations

import tomllib
from collections import deque
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

from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.strategies.ml.features import atr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MRFXConfig(StrategyConfig):
    """Configuration for the Mean Reversion FX Strategy."""

    instrument_id: str
    bar_type_m5: str  # e.g. "EUR/USD.IDEALPRO-5-MINUTE-MID-EXTERNAL"
    config_path: str = "config/mr_fx_eurusd.toml"
    warmup_bars: int = 3000  # ~12.5 trading days of M5 data


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class MRFXStrategy(Strategy):
    """Session-anchored VWAP Mean Reversion with exponential grid entries.

    Architecture:
      - Subscribes to M5 bars.
      - Warm-up from parquet to pre-populate VWAP, deviation, and ATR.
      - On each bar: update VWAP, check regime gate, evaluate grid entries.
      - Exits: partial reversion TP, hard invalidation, NY session close.
      - Integrated with PortfolioRiskManager for portfolio-level risk control.
    """

    def __init__(self, config: MRFXConfig) -> None:
        super().__init__(config)

        self.toml_cfg = self._load_toml(config.config_path)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_m5 = BarType.from_str(config.bar_type_m5)

        # Signal config
        sig_cfg = self.toml_cfg.get("signal", {})
        self.pct_window: int = sig_cfg.get("percentile_window", 2000)
        self.tiers_pct: list[float] = sig_cfg.get("tiers_pct", [0.90, 0.95, 0.98, 0.99])
        self.tier_sizes: list[int] = sig_cfg.get("tier_sizes", [1, 2, 4, 8])
        self.reversion_target: float = sig_cfg.get("reversion_target_pct", 0.5)
        self.invalidation_pct: float = sig_cfg.get("invalidation_pct", 0.9999)
        self.session_filter: str = sig_cfg.get("session_filter", "london_core")

        # Regime config
        reg_cfg = self.toml_cfg.get("regime", {})
        self.atr_gate_window: int = reg_cfg.get("atr_gate_window", 500)
        self.atr_gate_pct: float = reg_cfg.get("atr_gate_pct", 0.30)

        # Risk config
        risk_cfg = self.toml_cfg.get("risk", {})
        self.ny_close_utc: int = risk_cfg.get("ny_close_utc", 21)
        self.risk_pct: float = risk_cfg.get("risk_pct", 0.01)
        self.leverage_cap: float = risk_cfg.get("leverage_cap", 20.0)

        # VWAP anchor config
        vwap_cfg = self.toml_cfg.get("vwap", {})
        self.anchor_sessions: list[str] = vwap_cfg.get("anchor_sessions", ["london", "ny"])

        # Rolling state
        self.history: deque[dict] = deque(maxlen=config.warmup_bars + 200)
        self._vwap_cum_tp_vol: float = 0.0
        self._vwap_cum_vol: float = 0.0
        self._vwap_session_id: int = 0
        self._current_vwap: float = 0.0
        self._deviation_history: deque[float] = deque(maxlen=self.pct_window + 10)
        self._atr_history: deque[float] = deque(maxlen=self.atr_gate_window + 10)

        # Basket tracking (per direction)
        self._long_basket_cost: float = 0.0
        self._long_basket_units: int = 0
        self._short_basket_cost: float = 0.0
        self._short_basket_units: int = 0

        # Entry tier tracking (prevent re-entry at same tier in same session)
        self._long_tiers_hit: set[int] = set()
        self._short_tiers_hit: set[int] = set()
        self._entry_pending: bool = False

        # ATR for sizing
        self._latest_atr: float = 0.0

    def _load_toml(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MR FX config not found at {p}")
        with open(p, "rb") as fobj:
            return tomllib.load(fobj)

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def on_start(self) -> None:
        self.log.info(
            f"MR FX Strategy Started -- {self.instrument_id} | "
            f"tiers={len(self.tiers_pct)} | session={self.session_filter} | "
            f"reversion={self.reversion_target:.0%} | NY close={self.ny_close_utc}h"
        )
        self.subscribe_bars(self.bar_type_m5)
        self._warmup()

        # Register with portfolio risk manager
        self._prm_id = "mr_fx_eurusd"
        portfolio_risk_manager.register_strategy(self._prm_id, 10_000.0)

    def _warmup(self) -> None:
        """Pre-populate history from EUR_USD M5 parquet."""
        root = Path(__file__).resolve().parents[3]
        parquet = root / "data" / "EUR_USD_M5.parquet"
        if not parquet.exists():
            self.log.warning(f"No warmup data at {parquet}. Cold-starting.")
            return
        try:
            df = pd.read_parquet(parquet).sort_index().tail(self.config.warmup_bars)
            for _, row in df.iterrows():
                self.history.append(
                    {
                        "time": row.name if hasattr(row.name, "hour") else None,
                        "open": float(row.get("open", row["close"])),
                        "high": float(row.get("high", row["close"])),
                        "low": float(row.get("low", row["close"])),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0)),
                    }
                )
            self._recompute_indicators()
            self.log.info(f"Warmup complete: {len(self.history)} bars loaded.")
        except Exception as exc:
            self.log.error(f"Warmup failed: {exc}")

    def _recompute_indicators(self) -> None:
        """Recompute VWAP, deviation, ATR from full history after warmup."""
        if len(self.history) < 50:
            return

        df = pd.DataFrame(list(self.history))
        if "time" in df.columns and df["time"].iloc[0] is not None:
            df.index = pd.to_datetime(df["time"], utc=True)

        # ATR
        atr_series = atr(df, 14)
        valid_atr = atr_series.dropna()
        if len(valid_atr) > 0:
            self._latest_atr = float(valid_atr.iloc[-1])
            for v in valid_atr.values[-self.atr_gate_window :]:
                self._atr_history.append(float(v))

        # VWAP (simplified: just compute current session VWAP from last anchor)
        if hasattr(df.index, "hour"):
            hours = df.index.hour
            london_anchors = hours == 7
            ny_anchors = hours == 13
            anchors = london_anchors | ny_anchors
            last_anchor_idx = anchors.values.nonzero()[0]
            if len(last_anchor_idx) > 0:
                session_start = last_anchor_idx[-1]
                session_df = df.iloc[session_start:]
                typical = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
                vol = session_df["volume"].astype(float)
                cum_tp_vol = (typical * vol).cumsum()
                cum_vol = vol.cumsum()
                if cum_vol.iloc[-1] > 0:
                    self._current_vwap = float(cum_tp_vol.iloc[-1] / cum_vol.iloc[-1])
                    self._vwap_cum_tp_vol = float(cum_tp_vol.iloc[-1])
                    self._vwap_cum_vol = float(cum_vol.iloc[-1])

        # Deviation history
        if self._current_vwap > 0:
            for row in list(self.history)[-self.pct_window :]:
                dev = row["close"] - self._current_vwap
                self._deviation_history.append(dev)

    # ---------------------------------------------------------------------------
    # Bar handler
    # ---------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_m5:
            return

        dt = unix_nanos_to_dt(bar.ts_event)
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        volume = float(bar.volume) if hasattr(bar, "volume") else 0.0
        hour = dt.hour if hasattr(dt, "hour") else 0

        # Append to history
        self.history.append(
            {
                "time": dt,
                "open": float(bar.open),
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        # Portfolio risk manager: update equity every ~12 bars (hourly)
        if len(self.history) % 12 == 0:
            accounts = self.cache.accounts()
            if accounts:
                acct = accounts[0]
                ccys = list(acct.balances().keys())
                if ccys:
                    equity = float(acct.balance_total(ccys[0]).as_double())
                    portfolio_risk_manager.update(self._prm_id, equity)
        if portfolio_risk_manager.halt_all:
            self.log.warning("Portfolio kill switch active -- flattening.")
            self._flatten_all()
            return

        # --- VWAP update ---
        # Check for session anchor (VWAP reset)
        is_anchor = False
        if "london" in self.anchor_sessions and hour == 7:
            is_anchor = True
        if "ny" in self.anchor_sessions and hour == 13:
            is_anchor = True

        if is_anchor and dt.minute == 0:
            # Reset VWAP accumulators for new session
            self._vwap_cum_tp_vol = 0.0
            self._vwap_cum_vol = 0.0
            self._vwap_session_id += 1
            # Reset tier tracking for new session
            self._long_tiers_hit.clear()
            self._short_tiers_hit.clear()

        typical = (high + low + close) / 3.0
        self._vwap_cum_tp_vol += typical * volume
        self._vwap_cum_vol += volume
        if self._vwap_cum_vol > 0:
            self._current_vwap = self._vwap_cum_tp_vol / self._vwap_cum_vol

        if self._current_vwap <= 0:
            return

        # --- Deviation ---
        deviation = close - self._current_vwap
        self._deviation_history.append(deviation)

        # --- ATR update (every 12 bars = hourly equivalent) ---
        if len(self.history) >= 20 and len(self.history) % 12 == 0:
            df = pd.DataFrame(list(self.history)[-100:])
            atr_val = atr(df, 14)
            valid = atr_val.dropna()
            if len(valid) > 0:
                self._latest_atr = float(valid.iloc[-1])
                self._atr_history.append(self._latest_atr)

        # --- Exit checks (always evaluate, regardless of session) ---
        self._check_exits(close, deviation, hour)

        # --- Entry checks (session-filtered) ---
        if not self._in_session(hour):
            return

        if not self._regime_gate_pass():
            return

        self._check_entries(close, deviation)

    # ---------------------------------------------------------------------------
    # Session filter
    # ---------------------------------------------------------------------------

    def _in_session(self, hour: int) -> bool:
        """Check if the current bar is within the allowed trading session."""
        if self.session_filter == "london_core":
            return 7 <= hour < 12
        elif self.session_filter == "london":
            return 7 <= hour < 16
        elif self.session_filter == "london+ny":
            return 7 <= hour < 21
        return True  # "none"

    # ---------------------------------------------------------------------------
    # Regime gate
    # ---------------------------------------------------------------------------

    def _regime_gate_pass(self) -> bool:
        """ATR percentile gate: allow entries only when ATR < 30th percentile."""
        if len(self._atr_history) < 50:
            return False  # not enough data

        atr_arr = np.array(self._atr_history)
        current_atr = atr_arr[-1]
        threshold = np.percentile(atr_arr, self.atr_gate_pct * 100)
        return current_atr < threshold

    # ---------------------------------------------------------------------------
    # Entry logic
    # ---------------------------------------------------------------------------

    def _check_entries(self, close: float, deviation: float) -> None:
        """Check tiered grid entries based on percentile levels."""
        if len(self._deviation_history) < self.pct_window:
            return

        dev_arr = np.array(self._deviation_history)

        for i, pct in enumerate(self.tiers_pct):
            upper_level = float(np.percentile(dev_arr[:-1], pct * 100))
            lower_level = -abs(upper_level)

            # Short entry: deviation crosses above upper level
            if deviation > upper_level and i not in self._short_tiers_hit:
                self._short_tiers_hit.add(i)
                size = self.tier_sizes[i] if i < len(self.tier_sizes) else self.tier_sizes[-1]
                self._enter_tier(OrderSide.SELL, close, size, tier=i)

            # Long entry: deviation crosses below lower level
            if deviation < lower_level and i not in self._long_tiers_hit:
                self._long_tiers_hit.add(i)
                size = self.tier_sizes[i] if i < len(self.tier_sizes) else self.tier_sizes[-1]
                self._enter_tier(OrderSide.BUY, close, size, tier=i)

    def _enter_tier(self, side: OrderSide, price: float, tier_size: int, tier: int) -> None:
        """Submit a market order for one grid tier."""
        accounts = self.cache.accounts()
        if not accounts:
            return
        acct = accounts[0]
        ccys = list(acct.balances().keys())
        if not ccys:
            return
        equity = float(acct.balance_total(ccys[0]).as_double())

        # Size: risk_pct * equity / (ATR * leverage_cap), scaled by tier weight
        if self._latest_atr <= 0:
            return

        total_tier_weight = sum(self.tier_sizes[: len(self.tiers_pct)])
        tier_fraction = tier_size / total_tier_weight if total_tier_weight > 0 else 0.25
        risk_amount = equity * self.risk_pct * tier_fraction
        raw_units = risk_amount / self._latest_atr

        max_units = (equity * self.leverage_cap) / price if price > 0 else 0
        units = min(raw_units, max_units)

        # Apply portfolio-level scale factor
        units *= portfolio_risk_manager.scale_factor

        units = int(units)
        if units <= 0:
            return

        qty = Quantity.from_int(units)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        # Track basket
        if side == OrderSide.BUY:
            self._long_basket_cost += units * price
            self._long_basket_units += units
        else:
            self._short_basket_cost += units * price
            self._short_basket_units += units

        self.log.info(
            f"MR ENTRY tier={tier} {side.name} {units} @ {price:.5f} "
            f"(basket L={self._long_basket_units} S={self._short_basket_units})"
        )

    # ---------------------------------------------------------------------------
    # Exit logic
    # ---------------------------------------------------------------------------

    def _check_exits(self, close: float, deviation: float, hour: int) -> None:
        """Check all exit conditions: partial reversion, invalidation, time."""
        # 1. NY session close (force flatten)
        if hour == self.ny_close_utc:
            if self._long_basket_units > 0 or self._short_basket_units > 0:
                self.log.info("NY session close -- flattening all MR positions.")
                self._flatten_all()
            return

        # 2. Hard invalidation (99.99th percentile)
        if len(self._deviation_history) >= self.pct_window:
            dev_arr = np.array(self._deviation_history)
            inv_level = float(np.percentile(dev_arr[:-1], self.invalidation_pct * 100))
            if abs(deviation) > abs(inv_level):
                self.log.warning(
                    f"INVALIDATION: |dev|={abs(deviation):.5f} > inv={abs(inv_level):.5f}"
                )
                self._flatten_all()
                return

        # 3. Partial reversion TP (long basket)
        if self._long_basket_units > 0:
            basket_vwap = self._long_basket_cost / self._long_basket_units
            # Long entered below VWAP; TP when deviation returns to reversion_target
            # of the entry deviation
            entry_dev = basket_vwap - self._current_vwap  # negative for longs
            target_dev = entry_dev * self.reversion_target
            if deviation > target_dev:
                self.log.info(f"MR LONG TP: dev={deviation:.5f} target={target_dev:.5f}")
                self._close_long_basket()

        # 4. Partial reversion TP (short basket)
        if self._short_basket_units > 0:
            basket_vwap = self._short_basket_cost / self._short_basket_units
            entry_dev = basket_vwap - self._current_vwap  # positive for shorts
            target_dev = entry_dev * self.reversion_target
            if deviation < target_dev:
                self.log.info(f"MR SHORT TP: dev={deviation:.5f} target={target_dev:.5f}")
                self._close_short_basket()

    def _flatten_all(self) -> None:
        """Close all positions and reset basket tracking."""
        self.close_all_positions(self.instrument_id)
        self._long_basket_cost = 0.0
        self._long_basket_units = 0
        self._short_basket_cost = 0.0
        self._short_basket_units = 0
        self._long_tiers_hit.clear()
        self._short_tiers_hit.clear()

    def _close_long_basket(self) -> None:
        """Close long positions only."""
        # In NautilusTrader, close_all_positions closes the net position.
        # For simplicity, we close all and let the short basket re-enter if needed.
        # A more sophisticated approach would track individual order IDs.
        if self._short_basket_units == 0:
            self.close_all_positions(self.instrument_id)
        else:
            # Net position — submit a sell order to flatten the long portion
            if self._long_basket_units > 0:
                qty = Quantity.from_int(self._long_basket_units)
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)
        self._long_basket_cost = 0.0
        self._long_basket_units = 0
        self._long_tiers_hit.clear()

    def _close_short_basket(self) -> None:
        """Close short positions only."""
        if self._long_basket_units == 0:
            self.close_all_positions(self.instrument_id)
        else:
            if self._short_basket_units > 0:
                qty = Quantity.from_int(self._short_basket_units)
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)
        self._short_basket_cost = 0.0
        self._short_basket_units = 0
        self._short_tiers_hit.clear()

    # ---------------------------------------------------------------------------
    # Lifecycle events
    # ---------------------------------------------------------------------------

    def on_stop(self) -> None:
        self._flatten_all()
        self.log.info("MR FX Strategy stopped. All positions flattened.")

    def on_order_filled(self, event) -> None:
        self.log.info(f"FILL: {event.order_side.name} {event.last_qty} @ {event.last_px}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"ORDER REJECTED: {event.client_order_id} -- {event.reason}")
