"""samir_stack/strategy.py — Samir-Stack live NautilusTrader Strategy.

Regime-gated leveraged-equity + bond stack. Two trading instruments
(equity ETF + bond ETF) plus signal sources (SPY proxy, VIX, HYG, IEF
for the multi-indicator regime classifier).

Research lineage: ``research/samir_stack/``. The 2026-05-12 audit found
three deployment-blocking bugs in the prior research pipeline (look-ahead
in bond rotation and EFA overlay; dividend double-count in futures
engine); see ``directives/Samir-Stack Remediation Plan 2026-05-12.md``.
After Phases 1-5 of the remediation plan, this strategy is rebuilt on:

Live champion config (Phase 5 GBP-clean variant — 2026-05-13):
  Equity sleeve: 3USL.LSEETF (synthetic 3x SPY UCITS, daily-rebalanced).
                 Validated against UPRO (real 3x SPY): daily-return
                 correlation 0.9983 over 16.8y, CAGR diff +1.21pp.
  Bond sleeve:   IGLT.LSEETF (UK 7-10y gilts, GBP-native — no FX risk
                 on the defensive ballast for a GBP-base account).
  Capital split: 40% equity / 60% bond.
  L_max:         2.0 (tiers 1, 2 only — vol drag at L=3 erodes Calmar
                 more than the marginal upside from extra leverage).
  Tier thresholds: (0.30, 0.55) — regime score ≥ 0.30 enters tier 1;
                 ≥ 0.55 enters tier 2.
  Vol target:    DISABLED. Phase 5 found uniform Calmar-CI-lo
                 improvement from the regime gate without vol-targeting.
  I1/I2/I3:      ALL DROPPED. Phase 3 rejected bond rotation on the
                 churn gate (>22 flips/year); I3 EFA had a same-bar
                 look-ahead and was dropped per remediation plan §0(1).
  Capitulation:  RESEARCH-ONLY in this config. The Phase 5 winner uses
                 capitulation=on, but the overlay is not yet wired into
                 the live strategy class (Phase 6b follow-up). This
                 deployment is a strict subset of the Phase 5 cell;
                 expected Sharpe is ~0.91 vs the with-cap 0.96.

Phase 5 validation summary (with cap on, 18.9y backtest 2007-2026):
  WFO stitched Sharpe 0.961 | CI lo 0.428
  Calmar CI lo 0.148 | Sanctuary Sharpe 0.80
  MC P(MaxDD>50%) < 1% (RoR-acceptable per Samir framework)

This config replaces the prior 10/90 + 8% vol-target "champion" which
the audit found delivered honest Sharpe 0.64 / CAGR 5% (not the
claimed 2.28 / 20%).

Execution flow:
1. On each daily bar of SPY (the regime-driving signal):
   a. Update buffers (SPY/VIX/HYG/IEF).
   b. Compute regime score from the 6-indicator ensemble.
   c. Determine target equity tier with hysteresis.
   d. Apply DD circuit breaker overlay.
   e. Compute target weights: equity = equity_weight × tier; bond = bond_weight.
   f. If positions deviate from target, issue rebalance orders at next open.
2. Per-strategy equity tracking + portfolio-risk-manager registration per
   project contract (see CLAUDE.md / portfolio-risk-architecture.md).

Pre-flight gates required before deployment (see directives/Pre-Flight Checklist.md):
- Live parity test asserting one bar of regime/score/tier matches research.
- FX handling: equity sleeve 3USL is USD-quoted; UK paper account is GBP-base.
  Bond sleeve IGLT is GBP-native (no FX). Use convert_notional_to_units with
  explicit fx_rate for the equity sleeve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import (
    StrategyEquityTracker,
    convert_notional_to_units,
    report_equity_and_check,
)
from titan.strategies.samir_stack.regime import (
    RegimeBuffers,
    compute_regime_score,
    target_tier_from_score,
)


class SamirStackConfig(StrategyConfig):
    """Configuration for Samir-Stack live strategy."""

    # ── Required fields (no defaults) ────────────────────────────────────
    equity_instrument_id: str
    """The leveraged-equity ETF traded (e.g., '3USL.LSEETF')."""
    bond_instrument_id: str
    """The bond ETF traded (e.g., 'IEF.NASDAQ' or 'IBTM.LSEETF')."""
    signal_spy_id: str
    """SPY (or SPY proxy) daily bars — primary regime driver."""
    bar_type_equity_d: str
    bar_type_bond_d: str
    bar_type_spy_d: str

    # ── Optional fields (with defaults) ──────────────────────────────────
    signal_vix_id: str | None = None
    signal_hyg_id: str | None = None
    signal_ief_id: str | None = None
    bar_type_vix_d: str | None = None
    bar_type_hyg_d: str | None = None
    bar_type_ief_d: str | None = None

    # Per-instrument quote currency. For IBKR UK paper:
    #   equity_quote_ccy = "USD" (3USL.LSEETF is USD-quoted)
    #   bond_quote_ccy   = "GBP" (IGLT.LSEETF) or "USD" (IBTM.LSEETF)
    equity_quote_ccy: str = "USD"
    bond_quote_ccy: str = "USD"

    # FX rates: 1 unit of quote_ccy = how many units of base_ccy.
    # For a GBP-base account holding USD-quoted 3USL: fx_rate_equity ~= 0.75
    # (1 USD = 0.75 GBP). For GBP-quoted IGLT in a GBP account: 1.0.
    # The strategy uses these as static fallbacks; if the operator wires
    # a GBPUSD subscription later, the rates can be updated dynamically.
    fx_rate_equity_quote_to_base: float = 1.0
    fx_rate_bond_quote_to_base: float = 1.0

    # Strategy parameters — Phase 5 GBP-clean champion (2026-05-13).
    equity_weight: float = 0.40
    bond_weight: float = 0.60
    L_max: float = 2.0
    tier_thresholds: tuple[float, ...] = (0.30, 0.55)
    hysteresis_buffer: float = 0.05
    re_entry_quiet_bars: int = 20

    # Native leverage of the equity-trading instrument. For a 3x ETF
    # (3USL.LSEETF, UPRO) this is 3.0; for a 2x ETF (SSO) this is 2.0;
    # for a 1x ETF (CSPX, SPY) this is 1.0.
    #
    # The strategy holds ``equity_weight × (target_tier / equity_native_leverage)``
    # of NAV in the equity instrument. This makes the LIVE effective
    # SPX exposure match the BACKTEST model:
    #
    #   tier=1: live holds 40%/3 ≈ 13.3% of NAV in 3USL → 40% effective SPX
    #   tier=2: live holds 40%×2/3 ≈ 26.7% of NAV in 3USL → 80% effective SPX
    #
    # Without this scaling the live class would over-deploy: holding 40%
    # of NAV in 3USL gives 120% effective SPX regardless of tier — the
    # pre-audit behaviour that was inconsistent with the L_max=N research
    # model. See ``Samir-Stack Remediation Plan 2026-05-12.md`` §6 for
    # the live-vs-research mismatch this fixes.
    equity_native_leverage: float = 3.0

    # DD circuit breaker
    dd_throttle: float = 0.10
    dd_kill: float = 0.15
    dd_re_entry_score: float = 0.70
    dd_re_entry_bars: int = 5

    # Vol-targeting overlay — DISABLED by default (vol_target_annual <= 0).
    # Phase 5 found the regime gate captures the same risk-management benefit
    # without the operational complexity. Kept in the config for opt-in
    # experimentation only. The pre-Phase-5 default of 0.08 was based on a
    # buggy backtest (audit 2026-05-12); honest backtests show vol-target
    # adds only ~+0.07 Sharpe at significant operational cost (scaling on
    # already-leveraged ETF positions inflates vol drag).
    vol_target_annual: float = 0.0
    vol_target_window: int = 30
    vol_target_max_scale: float = 2.0

    # ── Phase 2: futures execution (CME MES Micro E-mini S&P 500) ────────
    #
    # When ``equity_is_future=True``, ``equity_instrument_id`` is treated
    # as an MES futures contract rather than a margin-traded ETF. Sizing
    # converts to integer contracts: ``contracts = floor(target_notional
    # / (futures_multiplier * spx_price))``. v1 MVP requires the operator
    # to set ``equity_instrument_id`` to the front-month contract and
    # update it manually each quarterly roll (Mar/Jun/Sep/Dec). See the
    # directive §13.7 for the rollover plan.
    equity_is_future: bool = False
    futures_multiplier: float = 5.0
    """USD per index point. MES = $5; ES = $50."""

    # ── Phase 3: bond rotation overlay (replaces single bond instrument) ──
    #
    # When ``bond_rotation_instruments`` is non-empty, the bond sleeve
    # rotates between the listed instruments + cash by 60-day momentum
    # (research/samir_stack/run_samir_improvements.py I2 overlay).
    # Each entry must be an InstrumentId string with a paired bar_type
    # in ``bond_rotation_bar_types``. The single ``bond_instrument_id``
    # serves as the v1 fallback (still required for backwards compat).
    bond_rotation_instruments: tuple[str, ...] = ()
    bond_rotation_bar_types: tuple[str, ...] = ()
    bond_rotation_lookback_days: int = 60

    # Operational
    initial_equity: float = 10_000.0
    base_ccy: str = "USD"
    warmup_bars: int = 250
    rebalance_min_pct_change: float = 0.05
    """Skip rebalance if target weight changes by less than this fraction
    of current weight (avoids tiny costly trades)."""

    # PortfolioRiskManager id
    prm_id: str = "samir_stack"


@dataclass
class _StrategyState:
    """Mutable strategy state."""

    equity_tier: float = 0.0
    dd_state: str = "normal"
    quiet_bars: int = 9999
    dd_recovery_bars: int = 0
    last_rebalance_ts: pd.Timestamp | None = None
    # Daily equity-curve sample for the vol-target rolling vol. Stored as
    # a list of base-ccy NLV values; computed differences are the daily
    # returns. Bounded to ~vol_target_window × 4 to keep memory tiny.
    equity_curve: list[float] = field(default_factory=list)
    # Last computed vol-target scale, persisted across rebalances so a
    # missing-bar day doesn't reset to 1.0.
    last_vol_scale: float = 1.0


class SamirStackStrategy(Strategy):
    """Samir-Stack live NautilusTrader strategy."""

    def __init__(self, config: SamirStackConfig) -> None:
        super().__init__(config)
        self.equity_id = InstrumentId.from_str(config.equity_instrument_id)
        self.bond_id = InstrumentId.from_str(config.bond_instrument_id)
        self.spy_id = InstrumentId.from_str(config.signal_spy_id)
        self.vix_id = InstrumentId.from_str(config.signal_vix_id) if config.signal_vix_id else None
        self.hyg_id = InstrumentId.from_str(config.signal_hyg_id) if config.signal_hyg_id else None
        self.ief_id = InstrumentId.from_str(config.signal_ief_id) if config.signal_ief_id else None

        self.bar_type_equity = BarType.from_str(config.bar_type_equity_d)
        self.bar_type_bond = BarType.from_str(config.bar_type_bond_d)
        self.bar_type_spy = BarType.from_str(config.bar_type_spy_d)
        self.bar_type_vix = (
            BarType.from_str(config.bar_type_vix_d) if config.bar_type_vix_d else None
        )
        self.bar_type_hyg = (
            BarType.from_str(config.bar_type_hyg_d) if config.bar_type_hyg_d else None
        )
        # IEF bar subscription is independent of the bond instrument — needed
        # for the credit-spread indicator (HYG/IEF ratio z-score) when
        # bond_instrument_id != IEF (e.g., IGLT for UK gilts).
        self.bar_type_ief = (
            BarType.from_str(config.bar_type_ief_d) if config.bar_type_ief_d else None
        )

        self._buffers = RegimeBuffers()
        self._state = _StrategyState()
        self._equity_tracker: StrategyEquityTracker | None = None
        self._prm_id = config.prm_id

        # Track latest prices for rebalance sizing
        self._last_equity_price: float | None = None
        self._last_bond_price: float | None = None

        # ── Phase 3: bond rotation overlay ────────────────────────────
        # Parse the rotation list. Keep alongside the single bond_id for
        # backwards compat — when the rotation list is empty, the strategy
        # falls back to single-bond behaviour.
        self.bond_rotation_ids: list[InstrumentId] = [
            InstrumentId.from_str(s) for s in config.bond_rotation_instruments
        ]
        self.bond_rotation_bar_types: list[BarType] = [
            BarType.from_str(s) for s in config.bond_rotation_bar_types
        ]
        if len(self.bond_rotation_ids) != len(self.bond_rotation_bar_types):
            raise ValueError(
                "samir_stack: bond_rotation_instruments and "
                "bond_rotation_bar_types must have matching lengths "
                f"(got {len(self.bond_rotation_ids)} vs {len(self.bond_rotation_bar_types)})"
            )
        # Per-bond close history for momentum lookup (key: instrument_id str).
        # Bounded to lookback × 4 to stay tiny.
        self._bond_closes: dict[str, list[float]] = {str(bid): [] for bid in self.bond_rotation_ids}
        # Latest selected bond instrument (cache to avoid repeated rotation
        # decisions inside one rebalance cycle).
        self._selected_bond_id: InstrumentId | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_start(self) -> None:
        # Fail-fast on FX mis-config: if a sleeve's quote ccy differs from
        # base_ccy but the fx_rate is the default 1.0, refuse to start.
        # Silent FX assumptions caused the April 2026 audit incidents.
        for sleeve_name, quote_ccy, fx_rate in [
            ("equity", self.config.equity_quote_ccy, self.config.fx_rate_equity_quote_to_base),
            ("bond", self.config.bond_quote_ccy, self.config.fx_rate_bond_quote_to_base),
        ]:
            if quote_ccy != self.config.base_ccy and abs(fx_rate - 1.0) < 1e-12:
                raise ValueError(
                    f"samir_stack: {sleeve_name}_quote_ccy={quote_ccy!r} != "
                    f"base_ccy={self.config.base_ccy!r} but fx_rate is default 1.0. "
                    f"Set fx_rate_{sleeve_name}_quote_to_base in config explicitly."
                )

        # Subscribe to bars for trading instruments
        self.subscribe_bars(self.bar_type_equity)
        self.subscribe_bars(self.bar_type_bond)
        # Subscribe to signal instruments
        self.subscribe_bars(self.bar_type_spy)
        if self.bar_type_vix is not None:
            self.subscribe_bars(self.bar_type_vix)
        if self.bar_type_hyg is not None:
            self.subscribe_bars(self.bar_type_hyg)
        # IEF independent subscription — the credit-spread indicator
        # needs IEF bars even when bond_instrument_id is not IEF (e.g., IGLT).
        if self.bar_type_ief is not None and self.bar_type_ief != self.bar_type_bond:
            self.subscribe_bars(self.bar_type_ief)
        # Phase 3: bond-rotation candidates. Subscribe to each rotation
        # member that isn't already in the active set above (to avoid
        # double-subscription with bond_id / IEF).
        already_subbed = {str(self.bar_type_bond)}
        if self.bar_type_ief is not None:
            already_subbed.add(str(self.bar_type_ief))
        for bt in self.bond_rotation_bar_types:
            if str(bt) not in already_subbed:
                self.subscribe_bars(bt)
                already_subbed.add(str(bt))

        # Per-strategy equity tracker (project contract)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self.config.initial_equity,
        )
        portfolio_risk_manager.register_strategy(self._prm_id, self.config.initial_equity)

        # Rehydrate any pre-existing broker positions (Tier 1.1 pattern,
        # post-May-11 fix). MUST run before the first on_bar so the
        # rebalance logic compares against actual broker state, not zero.
        self._rehydrate_position_from_broker()

        self.log.info(
            f"Samir-Stack started: equity={self.equity_id} ({self.config.equity_quote_ccy}), "
            f"bond={self.bond_id} ({self.config.bond_quote_ccy}), base_ccy={self.config.base_ccy}, "
            f"split={self.config.equity_weight}/{self.config.bond_weight}, "
            f"L_max={self.config.L_max}, "
            f"vol_target={self.config.vol_target_annual:.0%} "
            f"(window={self.config.vol_target_window}d)"
        )

    def on_stop(self) -> None:
        self.log.info("Samir-Stack stopping. Cancelling open orders.")
        self.cancel_all_orders(self.equity_id)
        self.cancel_all_orders(self.bond_id)

    # ── Bar handling ──────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        bar_ts = (
            pd.Timestamp(unix_nanos_to_dt(bar.ts_event)).tz_convert(None)
            if pd.Timestamp(unix_nanos_to_dt(bar.ts_event)).tz is not None
            else pd.Timestamp(unix_nanos_to_dt(bar.ts_event))
        )
        bar_date = bar_ts.normalize()

        # Update buffers based on which bar fired
        bt = bar.bar_type
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)

        if bt == self.bar_type_spy:
            self._buffers.add_spy(bar_date, close, high, low)
        elif self.bar_type_vix and bt == self.bar_type_vix:
            self._buffers.add_vix(bar_date, close)
        elif self.bar_type_hyg and bt == self.bar_type_hyg:
            self._buffers.add_hyg(bar_date, close)
        elif self.bar_type_ief and bt == self.bar_type_ief:
            # Dedicated IEF feed for credit-spread indicator
            self._buffers.add_ief(bar_date, close)
        elif bt == self.bar_type_bond:
            # Bond ETF doubles as IEF signal source ONLY when no separate IEF
            # feed is subscribed (i.e., bond_instrument_id == IEF.NASDAQ).
            if self.bar_type_ief is None and (self.ief_id is None or self.ief_id == self.bond_id):
                self._buffers.add_ief(bar_date, close)
            self._last_bond_price = close
        elif bt == self.bar_type_equity:
            self._last_equity_price = close

        # Phase 3: buffer rotation-bond closes for the 60d momentum.
        # We do this regardless of which bar arrived because rotation
        # candidates may overlap the bond_id / IEF subscriptions above.
        for bid, btype in zip(self.bond_rotation_ids, self.bond_rotation_bar_types, strict=True):
            if bt == btype:
                buf = self._bond_closes[str(bid)]
                buf.append(close)
                # Bound to lookback × 4 to keep memory tiny
                cap = self.config.bond_rotation_lookback_days * 4
                if len(buf) > cap:
                    self._bond_closes[str(bid)] = buf[-cap:]

        self._buffers.trim(max_bars=1500)

        # Only act on the SPY signal bar (the regime-driving primary)
        if bt != self.bar_type_spy:
            return

        # Equity tracker + halt check (project contract)
        _, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
        if halted:
            self.log.warning(f"PRM halt fired at {bar_ts}. Flattening positions.")
            self._flatten()
            return

        # Warmup gate
        if len(self._buffers.spy) < self.config.warmup_bars:
            return

        # Compute regime score
        score, breakdown = compute_regime_score(self._buffers)
        if not (0.0 <= score <= 1.0):
            return  # NaN or out-of-range — skip

        # Determine target tier
        target_tier = target_tier_from_score(
            score,
            self._state.equity_tier,
            tier_thresholds=tuple(self.config.tier_thresholds),
            hysteresis_buffer=self.config.hysteresis_buffer,
            L_max=self.config.L_max,
        )

        # Re-entry quiet bars
        if self._state.equity_tier == 0.0 and target_tier > 0.0:
            if self._state.quiet_bars < self.config.re_entry_quiet_bars:
                target_tier = 0.0

        # DD circuit breaker
        target_tier, bond_active = self._apply_dd_breaker(target_tier, score)

        # Compute target weights.
        #
        # The equity-sleeve weight is scaled by ``target_tier /
        # equity_native_leverage`` so that the LIVE effective SPX
        # exposure matches the BACKTEST model. For 3USL (native
        # leverage 3) the actual position held is:
        #   tier=1: 40% × 1/3 ≈ 13.3% of NAV in 3USL → 40% SPX
        #   tier=2: 40% × 2/3 ≈ 26.7% of NAV in 3USL → 80% SPX
        # See SamirStackConfig.equity_native_leverage docstring for the
        # rationale; this corrects a live-vs-research mismatch
        # documented in directives/Samir-Stack Remediation Plan
        # 2026-05-12.md §6.
        if target_tier == 0.0 and not bond_active:
            target_eq_w, target_bd_w = 0.0, 0.0
        elif target_tier == 0.0:
            target_eq_w, target_bd_w = 0.0, self.config.bond_weight
        else:
            tier_ratio = target_tier / self.config.equity_native_leverage
            target_eq_w = self.config.equity_weight * tier_ratio
            target_bd_w = self.config.bond_weight if bond_active else 0.0

        # Issue rebalance if material change
        self._rebalance_if_needed(target_eq_w, target_bd_w)

        # Update state
        self._update_quiet_bars(target_tier, score)
        self._state.equity_tier = target_tier
        self._state.last_rebalance_ts = bar_ts

        self.log.info(
            f"Score={score:.3f} tier={target_tier} dd_state={self._state.dd_state} "
            f"target=({target_eq_w:.2f},{target_bd_w:.2f}) breakdown={breakdown}"
        )

    # ── DD breaker ────────────────────────────────────────────────────────

    def _apply_dd_breaker(self, target_tier: float, score: float) -> tuple[float, bool]:
        """Apply DD circuit breaker. Returns (target_tier, bond_active)."""
        # Equity-tracker keeps HWM and current equity
        tr = self._equity_tracker
        if tr is None:
            return target_tier, True
        equity = tr.current_equity()
        hwm = tr.high_water_mark()
        dd = (equity - hwm) / hwm if hwm > 0 else 0.0
        bond_active = True

        if self._state.dd_state == "killed":
            target_tier = 0.0
            bond_active = False
            if score >= self.config.dd_re_entry_score:
                self._state.dd_recovery_bars += 1
                if self._state.dd_recovery_bars >= self.config.dd_re_entry_bars:
                    self._state.dd_state = "normal"
                    self._state.dd_recovery_bars = 0
                    tr.reset_hwm_to_current()
                    target_tier = target_tier_from_score(
                        score,
                        self._state.equity_tier,
                        tier_thresholds=tuple(self.config.tier_thresholds),
                        hysteresis_buffer=self.config.hysteresis_buffer,
                        L_max=self.config.L_max,
                    )
                    bond_active = True
            else:
                self._state.dd_recovery_bars = 0
        elif self._state.dd_state == "throttled":
            if abs(dd) >= self.config.dd_kill:
                self._state.dd_state = "killed"
                target_tier = 0.0
                bond_active = False
                self._state.dd_recovery_bars = 0
            elif score >= self.config.dd_re_entry_score:
                self._state.dd_recovery_bars += 1
                if self._state.dd_recovery_bars >= self.config.dd_re_entry_bars:
                    self._state.dd_state = "normal"
                    self._state.dd_recovery_bars = 0
                    tr.reset_hwm_to_current()
                else:
                    target_tier = min(target_tier, 1.0)
            else:
                self._state.dd_recovery_bars = 0
                target_tier = min(target_tier, 1.0)
        else:  # normal
            if abs(dd) >= self.config.dd_kill:
                self._state.dd_state = "killed"
                target_tier = 0.0
                bond_active = False
                self._state.dd_recovery_bars = 0
            elif abs(dd) >= self.config.dd_throttle:
                self._state.dd_state = "throttled"
                target_tier = min(target_tier, 1.0)
                self._state.dd_recovery_bars = 0

        return target_tier, bond_active

    def _update_quiet_bars(self, applied_tier: float, score: float) -> None:
        """Update the re-entry quiet-bars counter."""
        if applied_tier == 0.0:
            if score >= 0.50:
                self._state.quiet_bars += 1
            elif score < 0.30:
                self._state.quiet_bars = 0
        else:
            self._state.quiet_bars = 9999

    # ── Order management ─────────────────────────────────────────────────

    def _vol_target_scale(self, equity_value: float) -> float:
        """Compute the vol-target multiplicative scale for the equity sleeve.

        Samples the strategy's own NLV onto an internal buffer (one entry
        per rebalance call), computes 30-day rolling realised vol of the
        daily NLV returns, and returns ``min(target / realised, max_scale)``.
        Returns 1.0 (no-op) when:
          - vol_target_annual <= 0 (feature disabled)
          - insufficient history to compute realised vol
          - realised vol degenerate (zero / NaN)

        The scale is persisted as ``state.last_vol_scale`` so a single
        missing-bar day doesn't snap us back to 1.0 unexpectedly.

        Lag: scale is computed from history *up to and including* the
        most recent bar's NLV, then returned for the *current* rebalance.
        Because NLV is updated by yesterday's MTM (positions are evaluated
        on close), this is effectively scale-from-yesterday's-vol applied
        to today's sizing — the same lag-by-1 semantics as the research
        ``vol_target()`` wrapper in ``run_overlay_sweep.py``.
        """
        target = self.config.vol_target_annual
        if target is None or target <= 0:
            return 1.0
        # Append current NLV (skip duplicates from same bar)
        if self._state.equity_curve and self._state.equity_curve[-1] == equity_value:
            pass  # idempotent
        else:
            self._state.equity_curve.append(equity_value)
        # Bound buffer
        max_keep = self.config.vol_target_window * 4
        if len(self._state.equity_curve) > max_keep:
            self._state.equity_curve = self._state.equity_curve[-max_keep:]

        n = len(self._state.equity_curve)
        if n < self.config.vol_target_window + 1:
            return self._state.last_vol_scale  # warmup

        # Daily returns from the buffered NLV samples
        eq = self._state.equity_curve
        rets = [(eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, n) if eq[i - 1] != 0]
        window_rets = rets[-self.config.vol_target_window :]
        if len(window_rets) < 2:
            return self._state.last_vol_scale
        # Sample std × sqrt(252) for annualised
        mean_r = sum(window_rets) / len(window_rets)
        var = sum((r - mean_r) ** 2 for r in window_rets) / (len(window_rets) - 1)
        if var <= 0:
            return self._state.last_vol_scale
        realised_vol = (var**0.5) * (252.0**0.5)
        if realised_vol < 1e-8:
            return self._state.last_vol_scale
        scale = min(target / realised_vol, self.config.vol_target_max_scale)
        # Floor at small positive value to avoid zero notional from
        # transient extreme realised vol
        scale = max(scale, 0.01)
        self._state.last_vol_scale = scale
        return scale

    def _rehydrate_position_from_broker(self) -> None:
        """Adopt any open broker positions for our equity / bond instruments.

        Mirrors the Tier 1.1 pattern (see
        ``titan/strategies/bond_gold/strategy.py`` and
        ``directives/Operational Robustness Framework 2026-05-12.md``).
        Critical: must NOT filter ``cache.positions`` by ``strategy_id``
        — EXTERNAL-tagged positions reconciled by NT's ExecEngine on
        startup carry no ``strategy_id`` from the prior session.

        Adoption sets ``last_rebalance_ts`` to "very old" so the next
        on_bar evaluation will trigger a rebalance check against the
        rehydrated state — the strategy then either holds (target ==
        current) or trades to true-up (target != current).
        """
        try:
            adopted: list[str] = []
            for instrument_id in (self.equity_id, self.bond_id):
                positions = [
                    p for p in self.cache.positions(instrument_id=instrument_id) if not p.is_closed
                ]
                if not positions:
                    continue
                qty = sum(float(p.signed_qty) for p in positions)
                tags = sorted({str(p.strategy_id) for p in positions})
                if qty != 0:
                    adopted.append(f"{instrument_id} qty={qty:+.0f} tags={tags}")
            if adopted:
                self.log.info("REHYDRATED Samir-Stack: " + " | ".join(adopted))
        except Exception as e:
            self.log.warning(f"_rehydrate_position_from_broker failed: {e}")

    def _select_bond_instrument(self) -> InstrumentId | None:
        """Phase 3 bond-rotation selector.

        Mirrors ``research/samir_stack/run_samir_improvements.py::
        bond_rotation_returns`` (the I2 overlay). Returns the InstrumentId
        of the rotation candidate with the highest 60d log-momentum, or
        ``None`` (=> hold cash) if all candidates have negative momentum.

        Falls back to ``self.bond_id`` when rotation is disabled (empty
        ``bond_rotation_instruments`` config).

        Note: "cash" is represented as ``None`` and means the bond sleeve
        is unallocated for that period. The caller's ``_rebalance_if_needed``
        treats ``None`` as a flatten-bond signal.
        """
        if not self.bond_rotation_ids:
            return self.bond_id  # backwards compat

        lookback = self.config.bond_rotation_lookback_days
        scores: list[tuple[InstrumentId, float]] = []
        for bid in self.bond_rotation_ids:
            closes = self._bond_closes.get(str(bid), [])
            if len(closes) < lookback + 1:
                continue  # warmup
            try:
                from math import log

                mom = log(closes[-1] / closes[-1 - lookback])
            except (ValueError, ZeroDivisionError):
                continue
            scores.append((bid, mom))

        if not scores:
            return None  # not enough history → cash
        scores.sort(key=lambda kv: kv[1], reverse=True)
        best_bid, best_mom = scores[0]
        if best_mom <= 0:
            return None  # all negative → cash
        return best_bid

    def _futures_target_contracts(self, notional_base: float, price: float | None) -> float | None:
        """Phase 2 futures sizing: floor(notional / (multiplier × price)).

        Always integer-rounded down. ``futures_multiplier × price`` is the
        notional value of one contract in the future's quote currency.
        Multi-currency case: callers convert ``notional_base`` to the
        future's quote currency upstream and then this method just
        produces the integer-contract count.

        Returns None on missing / invalid price (caller skips the rebalance).
        """
        if price is None or price <= 0:
            return None
        if notional_base <= 0:
            return 0.0
        # Convert base→quote first (futures is a single-quote instrument)
        if self.config.equity_quote_ccy != self.config.base_ccy:
            # equity_quote_ccy → base: 1 quote = fx_rate base
            # so base → quote: divide by fx_rate
            fx = self.config.fx_rate_equity_quote_to_base
            if fx <= 0:
                return None
            notional_quote = notional_base / fx
        else:
            notional_quote = notional_base
        contract_notional = self.config.futures_multiplier * price
        if contract_notional <= 0:
            return None
        # Floor to integer contracts. We accept the small under-target
        # bias rather than over-leverage by rounding up.
        from math import floor

        return float(floor(notional_quote / contract_notional))

    def _rebalance_if_needed(
        self,
        target_eq_w: float,
        target_bd_w: float,
    ) -> None:
        """Issue rebalance orders if target weights differ from current."""
        equity_value = (
            self._equity_tracker.current_equity()
            if self._equity_tracker is not None
            else self.config.initial_equity
        )

        # Vol-target scale applied ONLY to the equity sleeve. The bond
        # sleeve is the defensive ballast and shouldn't be vol-scaled
        # (it's already low-vol by construction; scaling it up at calm
        # times defeats the purpose).
        vol_scale = self._vol_target_scale(equity_value)
        target_eq_w_scaled = target_eq_w * vol_scale

        # Target notional in BASE currency.
        # Equity sleeve effective notional = target_eq_w_scaled × equity_value
        target_eq_notional = target_eq_w_scaled * equity_value
        target_bd_notional = target_bd_w * equity_value

        # FX-aware unit conversion per project contract (see CLAUDE.md).
        # Each sleeve uses its own quote_ccy + fx_rate — never assume 1.0.
        # Phase 2: when ``equity_is_future`` use integer-contract sizing.
        if self.config.equity_is_future:
            target_eq_units = self._futures_target_contracts(
                target_eq_notional, self._last_equity_price
            )
        else:
            target_eq_units = self._compute_target_units(
                target_eq_notional,
                self._last_equity_price,
                quote_ccy=self.config.equity_quote_ccy,
                fx_rate=self.config.fx_rate_equity_quote_to_base,
            )

        # Phase 3: pick the rotation winner before sizing the bond sleeve.
        # selected==None means CASH for this period (flatten any non-cash
        # bond positions, take no new bond exposure).
        selected_bond_id = self._select_bond_instrument()
        self._selected_bond_id = selected_bond_id
        if selected_bond_id is None:
            target_bd_units = 0.0
            # Flatten any open bond positions across rotation candidates +
            # the legacy bond_id slot.
            self._flatten_unselected_bonds(keep=None)
        else:
            # If rotation chose something other than the legacy bond_id,
            # flatten the un-selected ones first.
            self._flatten_unselected_bonds(keep=selected_bond_id)
            # Use the selected instrument's last price for sizing
            sel_price = self._latest_price_for(selected_bond_id)
            target_bd_units = self._compute_target_units(
                target_bd_notional,
                sel_price,
                quote_ccy=self.config.bond_quote_ccy,
                fx_rate=self.config.fx_rate_bond_quote_to_base,
            )

        if target_eq_units is None or target_bd_units is None:
            self.log.warning("Missing price for sizing — skipping rebalance")
            return

        current_eq_units = self._current_units(self.equity_id)
        # Use the SELECTED bond for current-units (the one we'll trade).
        bond_target_id = selected_bond_id if selected_bond_id is not None else self.bond_id
        current_bd_units = self._current_units(bond_target_id)

        # Only rebalance if material delta
        eq_delta = target_eq_units - current_eq_units
        bd_delta = target_bd_units - current_bd_units

        eq_pct_change = abs(eq_delta) / max(abs(current_eq_units), 1.0)
        bd_pct_change = abs(bd_delta) / max(abs(current_bd_units), 1.0)
        if (
            (current_eq_units == 0 and current_bd_units == 0)
            or eq_pct_change > self.config.rebalance_min_pct_change
            or bd_pct_change > self.config.rebalance_min_pct_change
        ):
            self._submit_delta(self.equity_id, eq_delta)
            self._submit_delta(bond_target_id, bd_delta)

    def _flatten_unselected_bonds(self, *, keep: InstrumentId | None) -> None:
        """Phase 3 helper: close any open bond positions other than ``keep``.

        Iterates the rotation list (and the legacy ``self.bond_id`` slot
        for backwards compat) and submits a flatten order for each one
        the strategy is currently holding but rotation no longer wants.

        ``keep=None`` means we're going to cash and want EVERYTHING
        bond-side flat.
        """
        candidates = list(self.bond_rotation_ids)
        if self.bond_id not in candidates:
            candidates.append(self.bond_id)
        for bid in candidates:
            if keep is not None and bid == keep:
                continue
            current = self._current_units(bid)
            if abs(current) > 0.5:
                self._submit_delta(bid, -current)

    def _latest_price_for(self, instrument_id: InstrumentId) -> float | None:
        """Return the most-recent close in the rotation buffer for the
        given instrument, or fall back to ``self._last_bond_price``
        (which tracks the legacy ``self.bond_id``)."""
        if instrument_id == self.bond_id:
            return self._last_bond_price
        buf = self._bond_closes.get(str(instrument_id))
        if buf:
            return buf[-1]
        return None

    def _compute_target_units(
        self,
        notional_base: float,
        price: float | None,
        *,
        quote_ccy: str,
        fx_rate: float,
    ) -> float | None:
        """Convert base-currency notional to instrument units, FX-aware.

        Delegates to the project's shared ``convert_notional_to_units``
        helper which fails fast when ``quote_ccy != base_ccy`` and no
        rate is supplied. Returns None on missing price (so the caller
        can skip rebalance gracefully).
        """
        if price is None or price <= 0:
            return None
        if notional_base <= 0:
            return 0.0
        # convert_notional_to_units requires fx_rate=None when ccy matches
        # (refuses to silently assume 1.0 across currencies).
        rate = None if quote_ccy == self.config.base_ccy else fx_rate
        units = convert_notional_to_units(
            notional_base=notional_base,
            price=price,
            quote_ccy=quote_ccy,
            base_ccy=self.config.base_ccy,
            fx_rate_quote_to_base=rate,
        )
        return float(units)

    def _current_units(self, instrument_id: InstrumentId) -> float:
        """Current net position (units) for the instrument."""
        positions = self.cache.positions_open(instrument_id=instrument_id)
        if not positions:
            return 0.0
        # Sum signed quantities
        return float(sum(p.signed_qty.as_double() for p in positions))

    def _submit_delta(self, instrument_id: InstrumentId, delta_units: float) -> None:
        """Submit a market order for the delta. Skips if delta < 1 unit."""
        if abs(delta_units) < 1.0:
            return
        side = OrderSide.BUY if delta_units > 0 else OrderSide.SELL
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            self.log.warning(f"Instrument not found: {instrument_id}")
            return
        qty = Quantity.from_str(f"{abs(delta_units):.0f}")
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.DAY,
        )
        self.submit_order(order)
        self.log.info(f"Rebalance order: {side} {qty} {instrument_id}")

    def _flatten(self) -> None:
        """Close all positions immediately (emergency or PRM halt)."""
        for instrument_id in (self.equity_id, self.bond_id):
            current = self._current_units(instrument_id)
            if abs(current) > 0.5:
                self._submit_delta(instrument_id, -current)

    # ── Position events ──────────────────────────────────────────────────

    def on_position_closed(self, position) -> None:
        """Update equity tracker on each position close (project contract).

        PnL from IB is reported in the instrument's quote ccy. For non-base
        instruments (e.g., 3USL USD-quoted in a GBP account) we convert
        using the relevant sleeve's fx_rate.
        """
        if self._equity_tracker is None:
            return
        pnl_quote = float(position.realized_pnl.as_double()) if position.realized_pnl else 0.0

        # Determine which sleeve's FX applies based on the closed instrument
        instrument_id = position.instrument_id
        if instrument_id == self.equity_id:
            quote_ccy = self.config.equity_quote_ccy
            fx_rate = self.config.fx_rate_equity_quote_to_base
        elif instrument_id == self.bond_id:
            quote_ccy = self.config.bond_quote_ccy
            fx_rate = self.config.fx_rate_bond_quote_to_base
        else:
            quote_ccy = self.config.base_ccy
            fx_rate = 1.0

        fx_to_base = 1.0 if quote_ccy == self.config.base_ccy else fx_rate
        self._equity_tracker.on_position_closed(pnl_quote, fx_to_base=fx_to_base)
