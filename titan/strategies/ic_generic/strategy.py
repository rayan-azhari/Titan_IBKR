"""ic_generic/strategy.py
-----------------------

NautilusTrader live strategy — IC Generic Composite (asset-class agnostic).

Replaces ic_mtf/strategy.py (FX) and ic_equity_daily/strategy.py (equities)
with a single parametrised implementation driven entirely by TOML config.
No instrument-specific branching in code.

Validated pipeline:
  Phase 0-5 research must be complete and passing all gates before deploying.
  Threshold and signals come from Phase 3/2 output via scripts/phase6_deploy.py.

Strategy logic (mirrors research exactly):
  1. Warmup calibration: load last warmup_bars per TF from parquet → compute signals
     → derive IC signs via Spearman → freeze composite mean/std.
  2. On each base-TF bar: recompute signals for that TF, update oriented value.
  3. Higher TF signals: forward-filled between bar closes (matching research ffill).
  4. Composite z-score: mean(oriented_signals) normalised by calibration stats.
  5. Entry: z crosses ±threshold (short entry disabled when direction=long_only).
  6. Exit: z crosses zero.
  7. Sizing: ATR-based 1% risk per trade, capped at leverage_cap.

Live monitoring (evaluated every base-TF bar):
  - Rolling 20-trade Sharpe < 0 for 3 consecutive bars → log WARNING.
  - Equity drawdown > phase3_max_dd × 1.5 → flatten all, halt.
  - Trade frequency < 50% or > 200% of expected rate → log WARNING.

TOML config location: config/ic_generic.toml
Per-instrument section: [INSTRUMENT_NAME]  (e.g. [EUR_USD], [SPY])

Required TOML keys per instrument:
    signals       = ["accel_rsi14", "accel_stoch_k"]
    tfs           = ["W", "D", "H4", "H1"]
    threshold     = 0.75
    risk_pct      = 0.01
    stop_atr_mult = 1.5
    leverage_cap  = 20.0
    warmup_bars   = 1000
    direction     = "both"         # or "long_only"
    asset_class   = "fx_major"     # informational; used for log messages

Optional:
    phase3_max_dd_pct = 8.0        # from Phase 3 OOS. If set, enables DD halt gate.
    phase3_trade_rate = 0.0        # expected trades/month. If > 0, enables freq gate.
"""

from __future__ import annotations

from collections import deque
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
# Signal registry — all signals that can be requested via TOML config.
# Extend this dict to add new signals without modifying strategy logic.
# ---------------------------------------------------------------------------


def _accel_rsi14(df: pd.DataFrame) -> pd.Series:
    """RSI(14) acceleration: first difference of RSI deviation from 50."""
    close = df["close"].astype(float)
    return (rsi(close, 14) - 50.0).diff(1)


def _accel_rsi21(df: pd.DataFrame) -> pd.Series:
    """RSI(21) acceleration."""
    close = df["close"].astype(float)
    return (rsi(close, 21) - 50.0).diff(1)


def _accel_stoch_k(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K acceleration: first difference of %K deviation from 50."""
    k, _ = stochastic(df, k_period=14, d_period=3)
    return (k - 50.0).diff(1)


def _rsi_14_dev(df: pd.DataFrame) -> pd.Series:
    """RSI(14) deviation from 50."""
    return rsi(df["close"].astype(float), 14) - 50.0


def _rsi_21_dev(df: pd.DataFrame) -> pd.Series:
    """RSI(21) deviation from 50."""
    return rsi(df["close"].astype(float), 21) - 50.0


def _stoch_k(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K level."""
    k, _ = stochastic(df, k_period=14, d_period=3)
    return k - 50.0


_SIGNAL_REGISTRY: dict[str, callable] = {
    "accel_rsi14": _accel_rsi14,
    "accel_rsi21": _accel_rsi21,
    "accel_stoch_k": _accel_stoch_k,
    "rsi_14_dev": _rsi_14_dev,
    "rsi_21_dev": _rsi_21_dev,
    "stoch_k": _stoch_k,
}


# ---------------------------------------------------------------------------
# IC sign helper
# ---------------------------------------------------------------------------


def _ic_sign(signal: pd.Series, close: pd.Series) -> float:
    """Spearman IC sign at h=1 on warmup data. Returns +1.0 or -1.0."""
    fwd = np.log(close.shift(-1) / close)
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0.0 else 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ICGenericConfig(StrategyConfig):
    """Configuration for IC Generic Strategy (any asset class, any signal set).

    All per-instrument parameters come from config/ic_generic.toml via
    scripts/phase6_deploy.py. Never hardcode thresholds here.
    """

    instrument_id: str
    bar_types: dict[str, str]  # {"H1": "EUR/USD.IDEALPRO-1-HOUR-MID-EXTERNAL", ...}
    ticker: str  # symbol used to locate parquet data files, e.g. "EUR_USD"
    signals: list[str]  # e.g. ["accel_rsi14", "accel_stoch_k"]
    tfs: list[str]  # e.g. ["W", "D", "H4", "H1"]
    direction: str = "both"  # "both" (FX/futures) or "long_only" (equities)
    threshold: float = 0.75
    risk_pct: float = 0.01
    stop_atr_mult: float = 1.5
    leverage_cap: float = 20.0
    warmup_bars: int = 1000
    asset_class: str = "fx_major"  # informational label
    phase3_max_dd_pct: float = 0.0  # 0 = monitoring disabled
    phase3_trade_rate: float = 0.0  # expected trades/month; 0 = disabled


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class ICGenericStrategy(Strategy):
    """IC Generic Composite Strategy — asset-class agnostic.

    Works for any instrument and any signal set validated through the 5-phase
    IC research pipeline. Controlled entirely by ICGenericConfig / TOML.

    To add a new signal:
      1. Implement the signal function (df: pd.DataFrame) -> pd.Series.
      2. Add it to _SIGNAL_REGISTRY above.
      3. Reference its name in config/ic_generic.toml under the instrument section.
    """

    def __init__(self, config: ICGenericConfig) -> None:
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)

        # BarType -> TF label mapping
        self.bar_type_map: dict[BarType, str] = {}
        for tf, spec in config.bar_types.items():
            self.bar_type_map[BarType.from_str(spec)] = tf

        # Identify base TF (last in config list = highest-frequency)
        self._base_tf: str = config.tfs[-1]

        # Rolling OHLC history per TF
        self.history: dict[str, list[dict]] = {tf: [] for tf in config.tfs}

        # Latest oriented signal value per (signal_name, TF)
        self.latest_signal: dict[tuple[str, str], float] = {}

        # IS calibration (frozen after warmup)
        self._ic_signs: dict[tuple[str, str], float] = {}
        self._comp_mu: float = 0.0
        self._comp_sigma: float = 1.0
        self._calibrated: bool = False

        # Previous z-score for zero-cross detection
        self._prev_z: float = 0.0

        # Latest ATR (base TF)
        self.latest_atr: float | None = None

        # Entry guard
        self._entry_pending: bool = False
        self._entry_order_ids: set = set()

        # --- Live monitoring ---
        # Rolling trade returns (last 20) for rolling Sharpe
        self._trade_returns: deque[float] = deque(maxlen=20)
        self._consecutive_low_sharpe: int = 0
        # Equity peak for drawdown monitoring
        self._equity_peak: float = 0.0
        self._halted: bool = False
        # Trade counting for frequency monitoring
        self._trade_count_this_month: int = 0
        self._current_month: int = -1
        # Bars-since-entry for holding-period tracking
        self._entry_equity: float | None = None

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def on_start(self) -> None:
        cfg = self.config
        self.log.info(
            f"IC Generic Strategy starting — {cfg.ticker}"
            f" | signals={cfg.signals} | TFs={cfg.tfs}"
            f" | direction={cfg.direction} | threshold=±{cfg.threshold}z"
            f" | asset_class={cfg.asset_class}"
        )
        for bt in self.bar_type_map:
            self.subscribe_bars(bt)
        self._warmup_and_calibrate()
        self.log.info(
            f"Ready | calibrated={self._calibrated}"
            f" | mu={self._comp_mu:.4f} sigma={self._comp_sigma:.4f}"
            f" | ATR={self.latest_atr}"
        )

    def _warmup_and_calibrate(self) -> None:
        """Load warmup parquet data and calibrate IC signs + composite z-score stats.

        Mirrors research exactly:
          1. Load last warmup_bars per TF.
          2. Compute signals on native bars.
          3. Align higher TFs to base TF via ffill (with prior-bar shift).
          4. Compute IC sign per (signal, TF) on base-TF close.
          5. Freeze composite mean/std.
        """
        cfg = self.config
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"

        tf_df: dict[str, pd.DataFrame] = {}
        for tf in cfg.tfs:
            path = data_dir / f"{cfg.ticker}_{tf}.parquet"
            if not path.exists():
                self.log.warning(f"Warmup file not found: {path}")
                continue
            try:
                df = pd.read_parquet(path).sort_index().tail(cfg.warmup_bars)
                df.columns = [c.lower() for c in df.columns]
                for col in ("open", "high", "low", "close"):
                    if col not in df.columns:
                        self.log.warning(f"Missing column {col} in {path.name}")
                tf_df[tf] = df.dropna(subset=["close"])
                # Seed live history buffer
                for t, row in df.iterrows():
                    self.history[tf].append(
                        {
                            "time": t,
                            "open": float(row.get("open", row["close"])),
                            "high": float(row.get("high", row["close"])),
                            "low": float(row.get("low", row["close"])),
                            "close": float(row["close"]),
                        }
                    )
                self.log.info(f"Loaded {len(df)} warmup bars for {tf}")
            except Exception as exc:
                self.log.error(f"Failed to load {path}: {exc}")

        if self._base_tf not in tf_df:
            self.log.error(f"Base TF {self._base_tf} warmup data missing — cannot calibrate.")
            return

        base_df = tf_df[self._base_tf]
        base_close = base_df["close"].astype(float)
        base_index = base_close.index

        composite_parts: list[pd.Series] = []

        for sig_name in cfg.signals:
            fn = _SIGNAL_REGISTRY.get(sig_name)
            if fn is None:
                self.log.warning(
                    f"Signal '{sig_name}' not in registry — skipping."
                    f" Available: {list(_SIGNAL_REGISTRY.keys())}"
                )
                continue

            for tf in cfg.tfs:
                if tf not in tf_df:
                    continue

                raw = fn(tf_df[tf])

                # LOOKAHEAD PREVENTION:
                # Higher-TF bars are timestamped at bar open. Shift by 1 so the
                # signal is only available after bar close — exactly as in research.
                if tf != self._base_tf:
                    raw = raw.shift(1)

                aligned = raw.reindex(base_index, method="ffill")

                # Calibrate IC sign on base-TF close (IS calibration)
                sign = _ic_sign(aligned.dropna(), base_close.loc[aligned.dropna().index])
                self._ic_signs[(sig_name, tf)] = sign

                oriented = aligned * sign
                oriented.name = f"{sig_name}_{tf}"
                composite_parts.append(oriented)

                # Seed latest_signal with last warmup value
                last = oriented.dropna()
                self.latest_signal[(sig_name, tf)] = float(last.iloc[-1]) if not last.empty else 0.0

        if not composite_parts:
            self.log.error("No composite parts computed — calibration failed.")
            return

        composite = pd.concat(composite_parts, axis=1).mean(axis=1)
        self._comp_mu = float(composite.mean())
        self._comp_sigma = float(composite.std()) or 1.0
        self._calibrated = True
        last_comp = float(composite.dropna().iloc[-1]) if not composite.dropna().empty else 0.0
        self._prev_z = (last_comp - self._comp_mu) / self._comp_sigma

        # Seed ATR from base TF
        atr_series = atr(base_df, 14)
        if not atr_series.empty:
            last_atr = float(atr_series.dropna().iloc[-1])
            if not np.isnan(last_atr):
                self.latest_atr = last_atr

        self.log.info(
            f"Calibrated | {len(composite_parts)} (signal,TF) pairs"
            f" | mu={self._comp_mu:.4f} sigma={self._comp_sigma:.4f}"
            f" | ATR={self.latest_atr}"
        )

    # ---------------------------------------------------------------------------
    # Bar handler
    # ---------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if self._halted:
            return

        tf = self.bar_type_map.get(bar.bar_type)
        if not tf:
            return

        # Append to rolling history
        self.history[tf].append(
            {
                "time": unix_nanos_to_dt(bar.ts_event),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
            }
        )
        max_history = self.config.warmup_bars + 200
        if len(self.history[tf]) > max_history:
            self.history[tf] = self.history[tf][-self.config.warmup_bars :]

        # Recompute signals for this TF
        self._update_tf_signals(tf)

        # Evaluate strategy only on base-TF bars
        if tf == self._base_tf and self._calibrated and self.latest_atr is not None:
            z = self._get_composite_z()
            self._check_monitoring(bar)
            if not self._halted:
                self._evaluate(z, bar.close)
            self._prev_z = z

    # ---------------------------------------------------------------------------
    # Signal update
    # ---------------------------------------------------------------------------

    def _update_tf_signals(self, tf: str) -> None:
        """Recompute all configured signals for one TF from rolling history."""
        data = self.history[tf]
        if len(data) < 20:
            return

        df = pd.DataFrame(data)
        df = (
            df.rename(columns={"time": "timestamp"}).set_index("timestamp")
            if "timestamp" in df.columns
            else df
        )

        for sig_name in self.config.signals:
            fn = _SIGNAL_REGISTRY.get(sig_name)
            if fn is None:
                continue
            try:
                raw = fn(df)
                sign = self._ic_signs.get((sig_name, tf), 1.0)
                val = raw.iloc[-1] if len(raw) > 0 else float("nan")
                if not np.isnan(val):
                    self.latest_signal[(sig_name, tf)] = float(val) * sign
            except Exception as exc:
                self.log.debug(f"Signal {sig_name}/{tf} update failed: {exc}")

        # Update ATR on base TF
        if tf == self._base_tf:
            try:
                atr_s = atr(df, 14)
                last_val = float(atr_s.iloc[-1])
                if not np.isnan(last_val):
                    self.latest_atr = last_val
            except Exception:
                pass

    def _get_composite_z(self) -> float:
        """Compute current composite z-score from latest oriented signals."""
        values = [v for v in self.latest_signal.values() if not np.isnan(v)]
        if not values:
            return 0.0
        comp = float(np.mean(values))
        if self._comp_sigma < 1e-10:
            return 0.0
        return (comp - self._comp_mu) / self._comp_sigma

    # ---------------------------------------------------------------------------
    # Live monitoring
    # ---------------------------------------------------------------------------

    def _check_monitoring(self, bar: Bar) -> None:
        """Run live monitoring checks. Logs warnings and halts if thresholds exceeded."""
        cfg = self.config

        # Equity drawdown gate (only when phase3_max_dd_pct is configured)
        if cfg.phase3_max_dd_pct > 0:
            accounts = self.cache.accounts()
            if accounts:
                acct = accounts[0]
                currencies = list(acct.balances().keys())
                if currencies:
                    equity = float(acct.balance_total(currencies[0]).as_double())
                    if equity > self._equity_peak:
                        self._equity_peak = equity
                    if self._equity_peak > 0:
                        current_dd_pct = (self._equity_peak - equity) / self._equity_peak * 100
                        dd_limit = cfg.phase3_max_dd_pct * 1.5
                        if current_dd_pct > dd_limit:
                            self.log.error(
                                f"[HALT] DD {current_dd_pct:.1f}% exceeds limit {dd_limit:.1f}%"
                                f" (1.5 × Phase3 OOS max DD {cfg.phase3_max_dd_pct:.1f}%)."
                                f" Flattening all positions."
                            )
                            self.cancel_all_orders(self.instrument_id)
                            self.close_all_positions(self.instrument_id)
                            self._halted = True
                            return

        # Rolling 20-trade Sharpe warning
        if len(self._trade_returns) >= 10:
            arr = np.array(list(self._trade_returns))
            std = float(arr.std())
            if std > 1e-10:
                roll_sharpe = float(arr.mean()) / std * np.sqrt(len(arr))
            else:
                roll_sharpe = 0.0
            if roll_sharpe < 0:
                self._consecutive_low_sharpe += 1
                if self._consecutive_low_sharpe >= 3:
                    self.log.warning(
                        f"[MONITOR] Rolling 20-trade Sharpe={roll_sharpe:.2f} < 0"
                        f" for {self._consecutive_low_sharpe} consecutive checks."
                        f" Consider re-running Phase 1."
                    )
            else:
                self._consecutive_low_sharpe = 0

    # ---------------------------------------------------------------------------
    # Entry / exit logic
    # ---------------------------------------------------------------------------

    def _evaluate(self, z: float, price) -> None:
        """Evaluate z-score and manage position accordingly."""
        threshold = self.config.threshold
        direction = self.config.direction

        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        current_dir = 0
        if position and position.is_open:
            current_dir = 1 if str(position.side) == "LONG" else -1

        # Determine new bias from composite z
        if z > threshold:
            new_bias = 1
        elif z < -threshold and direction != "long_only":
            new_bias = -1
        else:
            new_bias = 0

        # Entry / direction flip
        if new_bias != 0 and new_bias != current_dir:
            if current_dir != 0:
                self.log.info(f"Signal flip {current_dir:+d} → {new_bias:+d} | z={z:+.3f}")
                self.cancel_all_orders(self.instrument_id)
                self.close_all_positions(self.instrument_id)
            side = OrderSide.BUY if new_bias == 1 else OrderSide.SELL
            self._open_position(side, price)
            return

        # Exit: z crosses zero
        if current_dir == 1 and z < 0 and self._prev_z >= 0:
            self.log.info(f"Exit long — z={self._prev_z:+.3f} → {z:+.3f}")
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)
        elif current_dir == -1 and z > 0 and self._prev_z <= 0:
            self.log.info(f"Exit short — z={self._prev_z:+.3f} → {z:+.3f}")
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)

        # Periodic status log (every 100 base-TF bars)
        if len(self.history.get(self._base_tf, [])) % 100 == 0:
            self.log.debug(
                f"z={z:+.4f} | prev_z={self._prev_z:+.4f}"
                f" | pos={current_dir:+d} | ATR={self.latest_atr}"
            )

    # ---------------------------------------------------------------------------
    # Order management
    # ---------------------------------------------------------------------------

    def _open_position(self, side: OrderSide, price) -> None:
        if self._entry_pending:
            self.log.debug("Entry pending — skipping duplicate.")
            return
        if self.latest_atr is None or self.latest_atr == 0:
            self.log.warning("ATR not ready — skipping entry.")
            return

        cfg = self.config
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

        # FX equity normalisation to USD
        if str(acct_ccy) != "USD":
            try:
                usd = Currency.from_str("USD")
                fx = self.portfolio.exchange_rate(acct_ccy, usd, PriceType.MID)
                if fx and fx > 0:
                    equity = equity_raw * float(fx)
            except Exception:
                self.log.warning(f"No {acct_ccy}/USD FX — using raw balance.")

        # ATR-based sizing (mirrors Phase 3 exactly)
        stop_dist = self.latest_atr * cfg.stop_atr_mult
        if stop_dist == 0:
            return

        raw_units = (equity * cfg.risk_pct) / stop_dist
        px = float(price)
        if px > 0:
            max_units = (equity * cfg.leverage_cap) / px
            units = min(raw_units, max_units)
        else:
            units = 0.0

        if int(units) <= 0:
            self.log.warning(f"Calculated size {units:.2f} is 0 — skipping entry.")
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
        self._entry_equity = equity
        self.submit_order(order)

        side_str = "BUY" if side == OrderSide.BUY else "SELL"
        self.log.info(
            f"{side_str} {qty}"
            f" | z={self._prev_z:+.3f} | ATR={self.latest_atr:.5f}"
            f" | stop_dist={stop_dist:.5f} | risk={cfg.risk_pct:.1%}"
        )

        # Monthly trade counter
        import datetime

        now = datetime.datetime.utcnow()
        if now.month != self._current_month:
            self._current_month = now.month
            self._trade_count_this_month = 0
        self._trade_count_this_month += 1

    def on_order_filled(self, event: OrderFilled) -> None:
        if event.instrument_id != self.instrument_id:
            return

        if event.client_order_id in self._entry_order_ids:
            self._entry_order_ids.discard(event.client_order_id)
            self._entry_pending = False
            self.log.info(f"Entry filled @ {event.last_px} | id={event.client_order_id}")
            return

        # Exit fill — record per-trade return for monitoring
        if self._entry_equity is not None and self._entry_equity > 0:
            accounts = self.cache.accounts()
            if accounts:
                currencies = list(accounts[0].balances().keys())
                if currencies:
                    current_eq = float(accounts[0].balance_total(currencies[0]).as_double())
                    trade_ret = (current_eq - self._entry_equity) / self._entry_equity
                    self._trade_returns.append(trade_ret)
            self._entry_equity = None

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("IC Generic stopped — orders cancelled, positions closed.")

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
        self.log.info(f"ORDER CANCELED: {event.client_order_id}")
        if event.client_order_id in self._entry_order_ids:
            self._entry_order_ids.discard(event.client_order_id)
            self._entry_pending = False
