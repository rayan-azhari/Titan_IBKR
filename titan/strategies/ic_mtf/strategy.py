"""ic_mtf strategy.py
--------------------

NautilusTrader live strategy for the IC MTF (Information Coefficient
Multi-Timeframe) system.

Validated through 5-phase research pipeline (2026-03-19):
  OOS Sharpe  : EUR/USD 7.71 | GBP/USD 8.28 | USD/JPY 7.35
                AUD/USD 6.84 | AUD/JPY 7.33 | USD/CHF 7.34
  WFO parity  : 0.98–1.03 across all pairs
  Robustness  : MC, top-N removal, 3x slippage, WFO folds — all PASS

Signals:
  accel_rsi14   = diff(rsi(close, 14) - 50)   RSI(14) acceleration
  accel_stoch_k = diff(stoch_k(df, 14) - 50)  Stochastic %K acceleration

Composite:
  equal-weight mean of sign-normalised (signal × TF) pairs across W, D, H4, H1
  z-score normalised using warmup-period mean/std (IS calibration)

Entry / Exit:
  Long  : composite_z crosses above  +threshold (default 0.75)
  Short : composite_z crosses below  -threshold
  Exit  : composite_z crosses zero

Sizing:
  1% equity risk / (1.5 × ATR14_H1) — capped at 20x leverage
"""

from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, PriceType, TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Currency, Quantity
from nautilus_trader.trading.strategy import Strategy
from scipy.stats import spearmanr

from titan.strategies.ml.features import atr, rsi, stochastic

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _accel_rsi14(close: pd.Series) -> pd.Series:
    """RSI(14) acceleration: first difference of RSI deviation from 50."""
    return (rsi(close, 14) - 50.0).diff(1)


def _accel_stoch_k(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K acceleration: first difference of %K deviation from 50."""
    k, _ = stochastic(df, k_period=14, d_period=3)
    return (k - 50.0).diff(1)


def _ic_sign(signal: pd.Series, close: pd.Series) -> float:
    """Spearman IC sign at h=1 on provided data. Returns +1 or -1."""
    fwd = np.log(close.shift(-1) / close)
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0.0 else 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ICMTFConfig(StrategyConfig):
    """Configuration for the IC MTF Strategy."""

    instrument_id: str
    bar_types: dict[str, str]  # {"H1": "EUR/USD.IDEALPRO-1-HOUR-MID-EXTERNAL", ...}
    threshold: float = 0.75  # z-score entry threshold (Phase 3 best for EUR/USD)
    risk_pct: float = 0.01
    stop_atr_mult: float = 1.5
    leverage_cap: float = 20.0
    warmup_bars: int = 1000


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class ICMTFStrategy(Strategy):
    """IC Multi-Timeframe Confluence Strategy.

    Phase 1-5 validated. See directives/IC MTF Backtesting Guide.md.

    Architecture:
      - Loads warmup parquet data per TF to calibrate IC signs + composite stats.
      - On each H1 bar: recomputes composite z-score from latest TF signal values.
      - Enters/exits based on z-score crossing ±threshold / zero.
      - Higher TF signals (H4, D, W) update when their bar closes; ffilled between.
    """

    _SIGNALS = ("accel_rsi14", "accel_stoch_k")
    _TFS = ("W", "D", "H4", "H1")

    def __init__(self, config: ICMTFConfig) -> None:
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        # BarType -> TF label
        self.bar_type_map: dict[BarType, str] = {}
        for tf, spec in config.bar_types.items():
            self.bar_type_map[BarType.from_str(spec)] = tf

        # Rolling OHLC history per TF
        self.history: dict[str, list[dict]] = {tf: [] for tf in self._TFS}

        # Latest oriented signal value per (signal_name, TF)
        self.latest_signal: dict[tuple[str, str], float] = {}

        # IS calibration (frozen after warmup)
        self._ic_signs: dict[tuple[str, str], float] = {}
        self._comp_mu: float = 0.0
        self._comp_sigma: float = 1.0
        self._calibrated: bool = False

        # Previous z for zero-cross detection
        self._prev_z: float = 0.0

        # H1 ATR for sizing
        self.latest_atr: float | None = None

        # Entry guard
        self._entry_pending: bool = False
        self._entry_order_ids: set = set()

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def on_start(self) -> None:
        self.log.info("IC MTF Strategy starting — loading warmup data...")
        for bt in self.bar_type_map:
            self.subscribe_bars(bt)
        self._warmup_and_calibrate()
        self.log.info(
            f"Ready | threshold=±{self.config.threshold}z"
            f" | risk={self.config.risk_pct:.1%}"
            f" | stop={self.config.stop_atr_mult}×ATR"
            f" | calibrated={self._calibrated}"
        )

    def _warmup_and_calibrate(self) -> None:
        """Load parquet warmup data and calibrate IC signs + composite z-score stats."""
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / ".tmp" / "data" / "raw"
        pair_str = self.instrument_id.symbol.value.replace("/", "_")

        tf_df: dict[str, pd.DataFrame] = {}

        for bt, tf in self.bar_type_map.items():
            path = data_dir / f"{pair_str}_{tf}.parquet"
            if not path.exists():
                self.log.warning(f"Warmup file not found: {path}")
                continue
            try:
                df = pd.read_parquet(path).sort_index().tail(self.config.warmup_bars)
            except Exception as e:
                self.log.error(f"Failed to load {path}: {e}")
                continue

            tf_df[tf] = df
            for t, row in df.iterrows():
                self.history[tf].append(
                    {
                        "time": t,
                        "close": float(row["close"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                    }
                )
            self.log.info(f"Loaded {len(df)} warmup bars for {tf}")

        if "H1" not in tf_df:
            self.log.error("H1 warmup data missing — cannot calibrate.")
            return

        base_close = tf_df["H1"]["close"].astype(float)
        base_index = base_close.index

        # Compute signals per (signal, TF), align to H1, calibrate sign
        composite_parts: list[pd.Series] = []
        for sig in self._SIGNALS:
            for tf in self._TFS:
                if tf not in tf_df:
                    continue
                df = tf_df[tf]
                close = df["close"].astype(float)

                if sig == "accel_rsi14":
                    raw = _accel_rsi14(close)
                else:
                    raw = _accel_stoch_k(df)

                aligned = raw.reindex(base_index, method="ffill")
                sign = _ic_sign(aligned.dropna(), base_close.loc[aligned.dropna().index])
                self._ic_signs[(sig, tf)] = sign

                oriented = aligned * sign
                oriented.name = f"{sig}_{tf}"
                composite_parts.append(oriented)

                last = oriented.dropna()
                self.latest_signal[(sig, tf)] = float(last.iloc[-1]) if not last.empty else 0.0

        if not composite_parts:
            self.log.error("No composite parts computed — calibration failed.")
            return

        composite = pd.concat(composite_parts, axis=1).mean(axis=1)
        self._comp_mu = float(composite.mean())
        self._comp_sigma = float(composite.std()) or 1.0
        self._calibrated = True
        self._prev_z = (float(composite.iloc[-1]) - self._comp_mu) / self._comp_sigma

        # Seed ATR
        atr_s = atr(tf_df["H1"], 14)
        if not atr_s.empty:
            self.latest_atr = float(atr_s.dropna().iloc[-1])

        self.log.info(
            f"Calibrated | mu={self._comp_mu:.4f} sigma={self._comp_sigma:.4f}"
            f" | {len(composite_parts)} (signal,TF) pairs | ATR={self.latest_atr}"
        )

    # ---------------------------------------------------------------------------
    # Bar handler
    # ---------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        tf = self.bar_type_map.get(bar.bar_type)
        if not tf:
            return

        self.history[tf].append(
            {
                "time": unix_nanos_to_dt(bar.ts_event),
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
            }
        )
        if len(self.history[tf]) > self.config.warmup_bars + 100:
            self.history[tf] = self.history[tf][-self.config.warmup_bars :]

        self._update_tf_signals(tf)

        # Evaluate only on H1 bars (entry timeframe)
        if tf == "H1" and self._calibrated and self.latest_atr is not None:
            z = self._get_composite_z()
            self._evaluate(z, bar.close)
            self._prev_z = z

    # ---------------------------------------------------------------------------
    # Signal computation
    # ---------------------------------------------------------------------------

    def _update_tf_signals(self, tf: str) -> None:
        """Recompute accel_rsi14 and accel_stoch_k for the given TF."""
        data = self.history[tf]
        if len(data) < 20:
            return

        df = pd.DataFrame(data)
        close = df["close"].astype(float)

        for sig in self._SIGNALS:
            if sig == "accel_rsi14":
                raw = _accel_rsi14(close)
            else:
                raw = _accel_stoch_k(df)

            sign = self._ic_signs.get((sig, tf), 1.0)
            val = raw.iloc[-1]
            if not np.isnan(val):
                self.latest_signal[(sig, tf)] = float(val) * sign

        # Update H1 ATR
        if tf == "H1":
            atr_s = atr(df, 14)
            last_atr = atr_s.iloc[-1]
            if not np.isnan(last_atr):
                self.latest_atr = float(last_atr)

    def _get_composite_z(self) -> float:
        """Composite z-score from latest oriented signal values."""
        values = [v for v in self.latest_signal.values() if not np.isnan(v)]
        if not values:
            return 0.0
        comp = float(np.mean(values))
        return (comp - self._comp_mu) / self._comp_sigma

    # ---------------------------------------------------------------------------
    # Entry / exit logic
    # ---------------------------------------------------------------------------

    def _evaluate(self, z: float, price) -> None:
        threshold = self.config.threshold
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None

        current_dir = 0
        if position and position.is_open:
            current_dir = 1 if str(position.side) == "LONG" else -1

        # Determine new signal bias
        if z > threshold:
            new_bias = 1
        elif z < -threshold:
            new_bias = -1
        else:
            new_bias = 0

        # Entry / flip
        if new_bias != 0 and new_bias != current_dir:
            if current_dir != 0:
                self.log.info(f"Signal flip: {current_dir:+d} -> {new_bias:+d} | z={z:+.3f}")
                self.cancel_all_orders(self.instrument_id)
                self.close_all_positions(self.instrument_id)
            side = OrderSide.BUY if new_bias == 1 else OrderSide.SELL
            self._open_position(side, price)
            return

        # Exit: z crosses zero
        if current_dir == 1 and z < 0 and self._prev_z >= 0:
            self.log.info(f"Exit long — z={self._prev_z:+.3f} -> {z:+.3f}")
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)
        elif current_dir == -1 and z > 0 and self._prev_z <= 0:
            self.log.info(f"Exit short — z={self._prev_z:+.3f} -> {z:+.3f}")
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)

        self._log_status(z, threshold, current_dir, price)

    # ---------------------------------------------------------------------------
    # Order management
    # ---------------------------------------------------------------------------

    def _open_position(self, side: OrderSide, price) -> None:
        if self._entry_pending:
            self.log.debug("Entry pending — skipping duplicate signal.")
            return
        if self.latest_atr is None or self.latest_atr == 0:
            self.log.warning("ATR not ready — skipping entry.")
            return

        accounts = self.cache.accounts()
        if not accounts:
            self.log.error("No account in cache.")
            return
        account = accounts[0]
        currencies = list(account.balances().keys())
        if not currencies:
            self.log.error("Account has no balances.")
            return

        acct_ccy = currencies[0]
        equity_raw = float(account.balance_total(acct_ccy).as_double())
        equity = equity_raw

        if str(acct_ccy) != "USD":
            try:
                usd = Currency.from_str("USD")
                fx = self.portfolio.exchange_rate(acct_ccy, usd, PriceType.MID)
                if fx and fx > 0:
                    equity = equity_raw * fx
            except Exception:
                self.log.warning(f"No {acct_ccy}/USD FX — using raw balance for sizing.")

        stop_dist = self.latest_atr * self.config.stop_atr_mult
        if stop_dist == 0:
            return

        raw_units = (equity * self.config.risk_pct) / stop_dist
        px = float(price)
        if px > 0:
            max_units = (equity * self.config.leverage_cap) / px
            units = min(raw_units, max_units)
        else:
            units = 0

        if int(units) <= 0:
            self.log.warning("Calculated size is 0 — skipping entry.")
            return

        qty = Quantity.from_int(int(units))
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self._entry_order_ids.add(order.client_order_id)
        self._entry_pending = True
        self.submit_order(order)
        self.log.info(
            f"{'BUY' if side == OrderSide.BUY else 'SELL'} {qty}"
            f" | ATR={self.latest_atr:.5f} stop_dist={stop_dist:.5f}"
        )

    def on_order_filled(self, event: OrderFilled) -> None:
        if event.instrument_id != self.instrument_id:
            return
        if event.client_order_id in self._entry_order_ids:
            self._entry_order_ids.discard(event.client_order_id)
            self._entry_pending = False
            self.log.info(f"Entry filled @ {event.last_px}")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("IC MTF stopped — orders cancelled, positions closed.")

    # ---------------------------------------------------------------------------
    # Order / position event handlers
    # ---------------------------------------------------------------------------

    def on_order_submitted(self, event) -> None:
        self.log.info(f"ORDER SUBMITTED: {event.client_order_id}")

    def on_order_accepted(self, event) -> None:
        self.log.info(f"ORDER ACCEPTED: {event.client_order_id} venue={event.venue_order_id}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"ORDER REJECTED: {event.client_order_id} — {event.reason}")
        if event.client_order_id in self._entry_order_ids:
            self._entry_order_ids.discard(event.client_order_id)
            self._entry_pending = False

    def on_order_canceled(self, event) -> None:
        self.log.info(f"ORDER CANCELLED: {event.client_order_id}")

    def on_order_expired(self, event) -> None:
        self.log.warning(f"ORDER EXPIRED: {event.client_order_id}")

    def on_position_opened(self, event) -> None:
        self.log.info(
            f"POSITION OPENED: {event.position_id}"
            f" side={event.entry.name} qty={event.quantity}"
            f" avg_px={event.avg_px_open:.5f}"
        )

    def on_position_changed(self, event) -> None:
        self.log.info(
            f"POSITION CHANGED: {event.position_id} qty={event.quantity}"
            f" unrealized_pnl={event.unrealized_pnl}"
        )

    def on_position_closed(self, event) -> None:
        self.log.info(
            f"POSITION CLOSED: {event.position_id}"
            f" realized_pnl={event.realized_pnl}"
            f" duration={event.duration_ns // 1_000_000_000}s"
        )

    # ---------------------------------------------------------------------------
    # Status log
    # ---------------------------------------------------------------------------

    def _log_status(self, z: float, threshold: float, current_dir: int, price) -> None:
        dir_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[current_dir]
        atr_str = f"{self.latest_atr:.5f}" if self.latest_atr else "pending"
        sep = "═" * 52
        self.log.info(
            f"\n{sep}\n"
            f"  IC MTF @ {float(price):.5f}  |  z={z:+.3f}  |  thresh=±{threshold}\n"
            f"  Position: {dir_str}  |  ATR(14)H1: {atr_str}\n"
            f"{sep}"
        )
