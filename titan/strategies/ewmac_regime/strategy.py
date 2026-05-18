"""I1 v2 -- Multi-feature HMM regime gate + EWMAC ensemble (LIVE/SHADOW).

Live wrapper for the audit-verdict cell C6_smoothed
(`research/ewmac/run_i1v2_audit.py`, pre-reg in
`directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate 2026-05-17.md`).

Design (V3.6 / V3.7 compliant):

  1. **IS-FROZEN ARTEFACT**: HMM model params + per-asset trend-friendly
     mapping are loaded from `data/i1v2_c6_frozen.json` at on_start. They
     are NEVER refit at runtime. To re-freeze, run
     `scripts/freeze_i1v2_c6_artefact.py` (typically after the 12mo
     re-audit at 2026-11-17).
  2. **CAUSAL FORWARD-FILTER**: the global regime state at each bar is
     decoded using the frozen HMM + the cumulative panel through that
     bar. No Viterbi (L50).
  3. **PANEL UPDATE**: the regime panel parquet is refreshed daily by a
     cron job (see `scripts/cusum_daemon.py` pattern); this strategy
     reads the latest snapshot at on_start and on each bar.
  4. **PER-ASSET GATE**: at each bar, each asset's EWMAC forecast is
     multiplied by gate=1 if global_state in trend_friendly_per_asset,
     else 0. Mapping is IS-frozen.
  5. **PARITY**: validated by `tests/test_ewmac_regime_parity.py` --
     frozen-runtime gate == audit-time gate bit-for-bit on the visible
     window.

Shadow mode (config.shadow_mode=True, default):
  - Computes EWMAC forecasts + gates + would-be portfolio weights every
    bar but submits NO orders.
  - Synthetic positions tracked internally; paper PnL flows through
    `StrategyEquityTracker.on_position_closed` calls aggregated daily.
  - 12mo paper validation period before any live capital allocation.

V3.6 contract applied:
  - StrategyEquityTracker for per-strategy equity (not whole-account NLV)
  - Halt persistence via portfolio_risk_manager
  - report_equity_and_check on every bar
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from titan.research.metrics import BARS_PER_YEAR
from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import StrategyEquityTracker, report_equity_and_check

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EwmacRegimeConfig(StrategyConfig):
    """I1 v2 EWMAC + regime gate configuration -- maps to C6_smoothed."""

    instrument_ids_str: str = (
        "ES.CME,NQ.CME,CL.NYMEX,BZ.IFEU,HG.COMEX,SI.COMEX,GC.COMEX,"
        "ZN.CBOT,ZB.CBOT,6E.CME,6J.CME"
    )
    bar_type_template: str = "{}-1-DAY-LAST-EXTERNAL"

    # EWMAC ensemble (C1 baseline = 16/64, 32/128, 64/256, FDM=1.35).
    speeds_str: str = "16/64,32/128,64/256"
    fdm: float = 1.35
    forecast_cap: float = 20.0
    vol_lookback_days: int = 20
    instrument_vol_lookback_days: int = 60
    target_vol_annual: float = 0.10
    target_forecast: float = 10.0

    # Costs (CME futures).
    cost_bps_per_turnover: float = 1.0
    cost_fixed_usd_per_fill: float = 1.0
    notional_usd_per_leg: float = 30_000.0

    # Artefact + panel paths.
    frozen_artefact_path: str = "data/i1v2_c6_frozen.json"
    regime_panel_path: str = "data/i1_regime_panel.parquet"

    # Rolling history cap (per-instrument). 600 ~= 2.4y daily, enough for
    # the slowest EWMAC speed (slow_hl=256).
    history_bars: int = 600

    # PRM / equity.
    initial_equity: float = 1_500.0  # 5% of 30k baseline per L65/L67
    base_ccy: str = "USD"

    # Shadow mode: True = no orders submitted, paper PnL only.
    shadow_mode: bool = True


class EwmacRegimeStrategy(Strategy):
    """I1 v2 EWMAC + regime gate, shadow-mode default.

    Submits no orders by default. Each bar:
      1. Append close to per-instrument deque.
      2. Re-read regime panel parquet (cron updates it).
      3. Compute global state path via frozen HMM (causal forward).
      4. Compute EWMAC forecast per instrument from the deques.
      5. Apply per-asset gate.
      6. Forecast -> notional position; record day-over-day PnL via
         the equity tracker.

    Bar synchronisation: signals are computed when ALL 11 instruments
    have produced a bar with the SAME (or strictly greater) date. The
    last fully-synced date drives the signal.
    """

    def __init__(self, config: EwmacRegimeConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._instrument_ids: list[InstrumentId] = [
            InstrumentId.from_str(s.strip())
            for s in config.instrument_ids_str.split(",")
            if s.strip()
        ]
        self._bar_types: list[BarType] = [
            BarType.from_str(config.bar_type_template.format(str(iid)))
            for iid in self._instrument_ids
        ]
        # Speeds parsed once.
        self._speeds: list[tuple[int, int]] = [
            (int(p.strip().split("/")[0]), int(p.strip().split("/")[1]))
            for p in config.speeds_str.split(",")
            if p.strip()
        ]
        # Per-symbol rolling close (timestamp -> close). Symbol = bar_type.instrument_id string.
        self._closes: dict[str, dict[pd.Timestamp, float]] = {
            str(iid): {} for iid in self._instrument_ids
        }
        # Frozen artefact (loaded at on_start).
        self._frozen = None
        # Regime panel snapshot (last-read).
        self._panel: pd.DataFrame | None = None
        # Per-symbol synthetic position (notional USD, signed).
        self._synthetic_notional: dict[str, float] = {
            str(iid): 0.0 for iid in self._instrument_ids
        }
        # Last close per symbol used for MTM.
        self._last_close: dict[str, float] = {
            str(iid): float("nan") for iid in self._instrument_ids
        }
        # Last fully-synced date (signals computed up to this date).
        self._last_signal_date: pd.Timestamp | None = None

        self._halted = False
        self._prm_id: str = ""
        self._equity_tracker: StrategyEquityTracker | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_start(self) -> None:
        self._prm_id = "ewmac_regime_i1v2_c6"
        portfolio_risk_manager.register_strategy(self._prm_id, self._cfg.initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self._cfg.initial_equity,
            base_ccy=self._cfg.base_ccy,
        )

        # Load frozen IS artefact (model + per-asset trend-friendly map).
        from titan.strategies.ewmac_regime.frozen_artefact import load_frozen_artefact
        artefact_fp = PROJECT_ROOT / self._cfg.frozen_artefact_path
        try:
            self._frozen = load_frozen_artefact(artefact_fp)
            self.log.info(
                f"loaded frozen artefact: is_end={self._frozen.is_end_date}, "
                f"freeze_ts={self._frozen.freeze_ts[:19]}"
            )
        except FileNotFoundError as e:
            self.log.error(f"{e}; strategy will not produce signals.")
            self._frozen = None

        # Load panel snapshot.
        self._reload_panel()

        # Subscribe to bars for all instruments.
        for bt in self._bar_types:
            self.subscribe_bars(bt)

        self.log.info(
            f"EwmacRegimeStrategy started | shadow_mode={self._cfg.shadow_mode} "
            f"| {len(self._instrument_ids)} instruments | "
            f"panel_rows={0 if self._panel is None else len(self._panel)}"
        )

    def _reload_panel(self) -> None:
        panel_fp = PROJECT_ROOT / self._cfg.regime_panel_path
        if not panel_fp.exists():
            self.log.warning(f"Regime panel missing at {panel_fp}; gate disabled.")
            self._panel = None
            return
        try:
            df = pd.read_parquet(panel_fp).sort_index().dropna(how="any")
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            self._panel = df
        except Exception as e:  # noqa: BLE001
            self.log.warning(f"panel reload failed: {e}")

    # ------------------------------------------------------------------
    # Bar handling
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        if self._halted:
            return
        # Per-bar equity report (V3.6 contract). Kill-switch flattens.
        _, halted = report_equity_and_check(
            self, self._prm_id, bar, tracker=self._equity_tracker
        )
        if halted:
            self._halted = True
            self._flatten_shadow()
            return

        sym = str(bar.bar_type.instrument_id)
        ts = pd.Timestamp(bar.close_time_as_datetime()).tz_localize(None).normalize()
        close = float(bar.close)

        # Mark-to-market against prior close (synthetic PnL bookkeeping).
        prior_close = self._last_close.get(sym)
        if prior_close is not None and np.isfinite(prior_close) and prior_close > 0:
            ret = close / prior_close - 1.0
            notional = self._synthetic_notional.get(sym, 0.0)
            mtm_pnl = notional * ret
            if abs(mtm_pnl) > 1e-9 and self._equity_tracker is not None:
                self._equity_tracker.on_position_closed(mtm_pnl, fx_to_base=1.0)
        self._last_close[sym] = close

        # Append to history (deduplicate by date, last-write-wins).
        self._closes[sym][ts] = close
        if len(self._closes[sym]) > self._cfg.history_bars:
            # Drop oldest.
            oldest = min(self._closes[sym])
            self._closes[sym].pop(oldest, None)

        # Bar sync: compute signal when all instruments have a bar at ts
        # (or strictly past it) AND ts is later than last signal date.
        if self._all_synced_at(ts) and ts != self._last_signal_date:
            self._last_signal_date = ts
            self._compute_and_apply_signals(ts)

    def _all_synced_at(self, ts: pd.Timestamp) -> bool:
        for sym in [str(iid) for iid in self._instrument_ids]:
            if ts not in self._closes.get(sym, {}):
                return False
        return True

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _compute_and_apply_signals(self, asof: pd.Timestamp) -> None:
        if self._frozen is None or self._panel is None:
            return
        from research.regime.hmm_gate_v2 import compute_panel_regime_gate_frozen

        # Build closes_df aligned to the frozen artefact's asset order.
        asset_order = self._frozen.asset_order
        closes_df = self._build_closes_df(asset_order, asof)
        if closes_df is None or len(closes_df) < max(s[1] for s in self._speeds) + 5:
            return

        # Compute gate using frozen artefact.
        gate = compute_panel_regime_gate_frozen(
            closes_df, self._panel,
            hmm_model=self._frozen.hmm_model,
            trend_friendly_per_asset=self._frozen.trend_friendly_per_asset,
            smoothing_days=self._frozen.smoothing_days,
            require_broad_trend=self._frozen.require_broad_trend,
        )
        if gate.empty:
            return

        # Compute EWMAC forecast per asset on the closes_df.
        forecast = self._compute_ewmac_forecast(closes_df)
        if forecast is None or forecast.empty:
            return

        # Apply gate -> gated forecast (latest bar only).
        latest_forecast = forecast.iloc[-1]
        latest_gate = gate.iloc[-1]
        gated_forecast = latest_forecast * latest_gate

        # Convert forecast to target notional per asset.
        # target = forecast/target_forecast * (target_vol/instrument_vol_ann) * equity
        equity = (
            self._equity_tracker.current_equity()
            if self._equity_tracker else self._cfg.initial_equity
        )
        inst_vol_ann = self._instrument_vol(closes_df).iloc[-1]
        # Avoid div-by-zero.
        inst_vol_ann = inst_vol_ann.replace(0, np.nan)
        sized = (
            gated_forecast / float(self._cfg.target_forecast)
        ) * (self._cfg.target_vol_annual / inst_vol_ann) * equity
        sized = sized.fillna(0.0)
        # Cap leverage per asset at 2x equity (sanity).
        sized = sized.clip(-2 * equity, 2 * equity)

        # Update synthetic positions.
        for asset in asset_order:
            sym = self._sym_for_asset(asset)
            if sym is None:
                continue
            self._synthetic_notional[sym] = float(sized.get(asset, 0.0))

        # Log a one-liner per signal.
        on_assets = [a for a in asset_order if abs(float(sized.get(a, 0.0))) > 1e-6]
        self.log.info(
            f"[shadow] {asof.date()} | equity={equity:.0f} {self._cfg.base_ccy} | "
            f"gated_on={len(on_assets)}/{len(asset_order)} | "
            f"max_notional={float(sized.abs().max()):.0f}"
        )

    def _build_closes_df(
        self, asset_order: list[str], asof: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Stack per-symbol close history into a DataFrame indexed by date.

        Tolerant of missing assets: if an asset in `asset_order` has no
        corresponding subscribed instrument, it is silently skipped. The
        gate function (compute_panel_regime_gate_frozen) handles a partial
        column set correctly. Missing assets simply don't contribute to
        the portfolio. Returns None if NO assets have history.
        """
        cols = {}
        for asset in asset_order:
            sym = self._sym_for_asset(asset)
            if sym is None:
                continue  # asset not subscribed (e.g., 6E/6J without TWS sub)
            hist = self._closes.get(sym, {})
            if not hist:
                continue
            s = pd.Series(hist).sort_index()
            s = s[s.index <= asof]
            cols[asset] = s
        if not cols:
            return None
        df = pd.DataFrame(cols).sort_index()
        df = df.dropna(how="any").ffill(limit=5)
        return df if len(df) > 0 else None

    def _sym_for_asset(self, asset: str) -> str | None:
        """Map a frozen-artefact asset code (e.g. 'ES') to an instrument-id
        string (e.g. 'ES.CME').
        """
        for iid in self._instrument_ids:
            if str(iid).split(".")[0] == asset:
                return str(iid)
        return None

    def _compute_ewmac_forecast(self, closes_df: pd.DataFrame) -> pd.DataFrame | None:
        """Carver multi-speed EWMAC ensemble.

        Per-asset, per-bar:
          forecast = clip( FDM * mean_s( clip( scalar_s * ewmac_s / vol, +-cap ) ), +-cap )
        """
        try:
            from research.ewmac.ewmac_strategy import (
                CARVER_FDM,
                CARVER_FORECAST_SCALARS,
                _vol_normalised_ewmac,
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning(f"ewmac import failed: {e}")
            return None
        speeds = self._speeds
        cap = self._cfg.forecast_cap
        fdm = self._cfg.fdm if self._cfg.fdm > 0 else CARVER_FDM.get(len(speeds), 1.0)
        per_asset_combined = pd.DataFrame(
            index=closes_df.index, columns=closes_df.columns, dtype=float,
        )
        for asset in closes_df.columns:
            close = closes_df[asset]
            accum = []
            for s in speeds:
                norm = _vol_normalised_ewmac(
                    close, s[0], s[1], vol_lookback=self._cfg.vol_lookback_days,
                )
                scalar = float(CARVER_FORECAST_SCALARS.get(s, 1.0))
                scaled = (norm * scalar).clip(-cap, cap)
                accum.append(scaled)
            mean_forecast = pd.concat(accum, axis=1).mean(axis=1)
            per_asset_combined[asset] = (mean_forecast * fdm).clip(-cap, cap)
        return per_asset_combined.fillna(0.0)

    def _instrument_vol(self, closes_df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(closes_df / closes_df.shift(1))
        return log_ret.rolling(
            self._cfg.instrument_vol_lookback_days,
            min_periods=self._cfg.instrument_vol_lookback_days,
        ).std(ddof=1) * np.sqrt(BARS_PER_YEAR["D"])

    # ------------------------------------------------------------------
    # Shadow / live order interface
    # ------------------------------------------------------------------
    def _flatten_shadow(self) -> None:
        """Zero out synthetic positions on kill-switch."""
        self._synthetic_notional = {k: 0.0 for k in self._synthetic_notional}
        self.log.warning("[shadow] flattened synthetic positions (kill-switch).")

    def on_position_closed(self, event) -> None:  # noqa: ANN001
        # Shadow mode submits no orders -> this should never fire on a
        # real trade. Kept for forward-compatibility with live wiring.
        if self._equity_tracker is None:
            return
        try:
            pnl = float(event.realized_pnl.as_double())
            self._equity_tracker.on_position_closed(pnl, fx_to_base=1.0)
        except Exception as e:  # noqa: BLE001
            self.log.warning(f"tracker on_position_closed failed: {e}")
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")
