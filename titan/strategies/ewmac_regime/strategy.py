"""I1 v2 -- Multi-feature HMM regime gate + EWMAC ensemble.

Live wrapper for the audit-verdict cell C6_smoothed
(`research/ewmac/run_i1v2_audit.py`, pre-reg in
`directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate 2026-05-17.md`).

V3.6 / V3.7 discipline applied:
  * Per-strategy equity tracker (StrategyEquityTracker)
  * Halt persistence via portfolio_risk_manager
  * Causal forward-filtered state path (no Viterbi, L50)
  * IS-frozen HMM + per-asset trend-friendly mapping
  * Equity-level kill-switch + per-bar PRM update

NOT YET LIVE-WIRED INTO v37_live STRATEGY_SET.
This file is a scaffold for the shadow port. To take live:
  1. Wire IBKR contract subscriptions for all 11 trading instruments +
     6 regime-feature instruments (VIX/TLT/IEF/HYG/SPY/DXY).
  2. Implement live regime panel reconstruction (currently loads static
     `data/i1_regime_panel.parquet` and reads the last-known row).
  3. Implement multi-instrument bar synchronisation (similar to GEM
     SPY/EFA/IEF sync).
  4. Add a parity test in tests/test_ewmac_regime_parity.py that
     compares a single-day signal from this Strategy vs the same day's
     output from `gated_ewmac_returns()` in run_i1v2_audit.py.
  5. Add to STRATEGY_REGISTRY in scripts/run_portfolio.py.
  6. Add to v37_live STRATEGY_SETS in scripts/run_portfolio.py after
     parity passes + L65/L67 are re-run with refreshed window.

Per L65 + L67 (2026-05-17):
  - L65 single PASS at 5/10/15%
  - L65 joint PASS at >=5% I1v2 (rescues current 80/20 mix which marginally
    fails on 2019-2025 window)
  - L67 unchanged at PORTFOLIO_CONDITIONAL
  - VERDICT: risk reducer, not return enhancer. Shadow port at 5% weight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import StrategyEquityTracker, report_equity_and_check

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EwmacRegimeConfig(StrategyConfig):
    """I1 v2 EWMAC + regime gate configuration.

    Cells map to C6_smoothed of the I1v2 pre-reg (2-state HMM, mean-return
    state ID, 5-day median smoothing, seed=42). Default values mirror the
    audit-verdict cell.
    """

    # Universe -- 11 instruments, comma-separated NautilusTrader instrument IDs.
    # Order matches the audit's UNIVERSE dict in run_b2e_audit.py.
    instrument_ids_str: str = (
        "ES.CME,NQ.CME,CL.NYMEX,BZ.IFEU,HG.COMEX,SI.COMEX,GC.COMEX,"
        "ZN.CBOT,ZB.CBOT,6E.CME,6J.CME"
    )
    bar_type_template: str = "{}-1-DAY-LAST-EXTERNAL"

    # EWMAC ensemble (B2 C1 baseline = 16/64, 32/128, 64/256, FDM=1.35).
    speeds_str: str = "16/64,32/128,64/256"
    fdm: float = 1.35
    forecast_cap: float = 20.0
    vol_lookback_days: int = 20
    instrument_vol_lookback_days: int = 60
    target_vol_annual: float = 0.10
    target_forecast: float = 10.0

    # Regime gate (C6_smoothed config).
    hmm_n_states: int = 2
    hmm_state_id: str = "mean_return"
    hmm_smoothing_days: int = 5
    hmm_random_seed: int = 42
    hmm_min_state_bars: int = 60

    # Regime panel: load from this parquet at on_start, then live-extend.
    # The panel must contain columns matching the audit (vix_z, term_spread_z,
    # credit_spread_z, rv20_z, spy_above_sma200, dxy_z, dd_velocity_21).
    regime_panel_path: str = "data/i1_regime_panel.parquet"

    # Costs (CME futures).
    cost_bps_per_turnover: float = 1.0
    cost_fixed_usd_per_fill: float = 1.0
    notional_usd_per_leg: float = 30_000.0

    # PRM / equity.
    initial_equity: float = 1_500.0  # 5% of 30k baseline -- see L65/L67 2026-05-17
    base_ccy: str = "USD"

    # Rebalance frequency (Carver default = monthly target).
    rebalance: str = "monthly"

    # Shadow mode: when True, the strategy computes signals + paper PnL
    # but does NOT submit orders. Defaults to True for first deployment.
    shadow_mode: bool = True


class EwmacRegimeStrategy(Strategy):
    """I1 v2 EWMAC + regime gate live strategy.

    SCAFFOLD ONLY -- live signal computation + order submission deferred to
    a follow-up session. This class establishes the V3.6 contract pattern
    (PRM registration, equity tracker, kill-switch) but does not yet wire
    EWMAC forecasts to live orders.
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
        # Speeds tuple parsed from "16/64,32/128,64/256".
        self._speeds: list[tuple[int, int]] = []
        for cell in config.speeds_str.split(","):
            parts = cell.strip().split("/")
            self._speeds.append((int(parts[0]), int(parts[1])))

        # Per-instrument rolling close arrays. Keyed by instrument symbol.
        self._closes: dict[str, list[float]] = {
            str(iid): [] for iid in self._instrument_ids
        }
        # Regime panel (static at on_start; live extension deferred to next iteration).
        self._regime_panel: pd.DataFrame | None = None
        # IS-frozen HMM and per-asset trend-friendly mapping.
        self._hmm_model: object | None = None
        self._trend_friendly_per_asset: dict[str, set[int]] = {}
        # Last computed state (causally forward-filtered).
        self._last_state: int | None = None
        self._last_signal_date: pd.Timestamp | None = None
        # Per-instrument synthetic position (for shadow-mode paper PnL).
        self._synthetic_positions: dict[str, float] = {
            str(iid): 0.0 for iid in self._instrument_ids
        }

        self._halted = False
        self._prm_id: str = ""
        self._equity_tracker: StrategyEquityTracker | None = None

    def on_start(self) -> None:
        self._prm_id = "ewmac_regime_i1v2_c6"
        portfolio_risk_manager.register_strategy(self._prm_id, self._cfg.initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self._cfg.initial_equity,
            base_ccy=self._cfg.base_ccy,
        )

        # Subscribe to bars for all 11 instruments.
        for bt in self._bar_types:
            self.subscribe_bars(bt)

        # Load static regime panel + fit IS-frozen HMM.
        panel_fp = PROJECT_ROOT / self._cfg.regime_panel_path
        if not panel_fp.exists():
            self.log.error(
                f"Regime panel not found at {panel_fp}. Strategy will run "
                f"WITHOUT gate (degenerates to bare B2e). Re-run "
                f"research/exploration/build_i1_regime_panel.py."
            )
            self._regime_panel = None
        else:
            self._regime_panel = pd.read_parquet(panel_fp).sort_index().dropna(how="any")
            self._fit_is_frozen_hmm()

        self.log.info(
            f"EwmacRegimeStrategy started | shadow_mode={self._cfg.shadow_mode} "
            f"| n_instruments={len(self._instrument_ids)} "
            f"| panel_rows={0 if self._regime_panel is None else len(self._regime_panel)}"
        )

    def _fit_is_frozen_hmm(self) -> None:
        """Fit the multi-feature HMM on the loaded panel.

        Uses `research.regime.hmm_gate_v2.fit_panel_hmm` so live behaviour
        matches the audit harness bar-for-bar.
        """
        from research.regime.hmm_gate_v2 import (
            PanelHMMGateConfig,
            fit_panel_hmm,
        )

        if self._regime_panel is None:
            return

        gate_cfg = PanelHMMGateConfig(
            n_states=self._cfg.hmm_n_states,
            state_id=self._cfg.hmm_state_id,  # type: ignore[arg-type]
            min_state_bars=self._cfg.hmm_min_state_bars,
            smoothing_days=self._cfg.hmm_smoothing_days,
            random_seed=self._cfg.hmm_random_seed,
        )
        self._hmm_model = fit_panel_hmm(self._regime_panel, cfg=gate_cfg)
        if self._hmm_model is None:
            self.log.warning("HMM fit failed; gate disabled.")
            return

        # Trend-friendly mapping requires per-asset close history. For the
        # scaffold, we defer this until enough live bars accumulate; assume
        # all states are trend-friendly initially (degrades to no-gate).
        # The live wiring should LOAD the audit-computed mapping from a
        # frozen artefact (e.g., data/i1v2_trend_friendly.json) so the
        # live mapping matches the audit's IS-frozen choice bit-for-bit.
        self.log.info(
            f"IS-frozen HMM fitted: n_states={self._cfg.hmm_n_states}, "
            f"transmat diag={[round(float(d), 3) for d in np.diag(self._hmm_model.transmat_)]}"
        )

    def on_bar(self, bar: Bar) -> None:
        if self._halted:
            return
        # Per-bar equity report; kill-switch flattens locally if PRM trips.
        _, halted = report_equity_and_check(
            self, self._prm_id, bar, tracker=self._equity_tracker
        )
        if halted:
            self._halted = True
            self.log.warning("PRM kill-switch tripped -- shadow strategy paused.")
            return

        # Append close to rolling array.
        sym = str(bar.bar_type.instrument_id)
        self._closes[sym].append(float(bar.close))
        # Cap memory at 600 bars (~2.4y daily).
        if len(self._closes[sym]) > 600:
            self._closes[sym] = self._closes[sym][-600:]

        # Live signal computation deferred to next iteration:
        # 1. Reconstruct today's regime panel row from VIX/TLT/IEF/HYG/SPY/DXY
        #    bar subscriptions.
        # 2. Causally forward-filter to get today's global state.
        # 3. Compute per-instrument EWMAC forecast from rolling close arrays.
        # 4. Apply trend-friendly gate per asset.
        # 5. Convert forecast to notional + sign -> synthetic position.
        # 6. Shadow PnL = synthetic position * (close[t] - close[t-1]) / close[t-1].

    def on_position_closed(self, event) -> None:  # noqa: ANN001
        # Shadow mode submits no orders, so this should never fire. Keep
        # the hook for parity with the live-wiring contract.
        if self._equity_tracker is None:
            return
        try:
            pnl = float(event.realized_pnl.as_double())
            self._equity_tracker.on_position_closed(pnl, fx_to_base=1.0)
        except Exception as e:  # noqa: BLE001
            self.log.warning(f"tracker on_position_closed failed: {e}")
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")
