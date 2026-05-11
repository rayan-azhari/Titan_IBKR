"""samir_stack/strategy.py — Samir-Stack live NautilusTrader Strategy.

Regime-gated 40/60 leveraged-equity + bond stack. Two trading instruments
(leveraged-equity ETF + bond ETF) plus signal sources (SPY index proxy,
VIX, HYG, IEF for the regime classifier).

Research lineage: research/samir_stack/
WFO 5/5 folds positive, mean Calmar 0.655, sanctuary 1.05.

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
- FX handling: equity sleeve quoted in USD, account base may be GBP — use
  convert_notional_to_units with explicit fx_rate.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    # Strategy parameters (defaults from research)
    equity_weight: float = 0.40
    bond_weight: float = 0.60
    L_max: float = 3.0
    tier_thresholds: tuple[float, ...] = (0.30, 0.50, 0.75)
    hysteresis_buffer: float = 0.05
    re_entry_quiet_bars: int = 20

    # DD circuit breaker
    dd_throttle: float = 0.10
    dd_kill: float = 0.15
    dd_re_entry_score: float = 0.70
    dd_re_entry_bars: int = 5

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

        # Per-strategy equity tracker (project contract)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self.config.initial_equity,
        )
        portfolio_risk_manager.register_strategy(self._prm_id, self.config.initial_equity)
        self.log.info(
            f"Samir-Stack started: equity={self.equity_id} ({self.config.equity_quote_ccy}), "
            f"bond={self.bond_id} ({self.config.bond_quote_ccy}), base_ccy={self.config.base_ccy}, "
            f"split={self.config.equity_weight}/{self.config.bond_weight}, "
            f"L_max={self.config.L_max}"
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

        # Compute target weights
        if target_tier == 0.0 and not bond_active:
            target_eq_w, target_bd_w = 0.0, 0.0
        elif target_tier == 0.0:
            target_eq_w, target_bd_w = 0.0, self.config.bond_weight
        else:
            target_eq_w = self.config.equity_weight
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

        # Target notional in BASE currency.
        # Equity sleeve effective notional = target_eq_w × equity_value
        # (the leveraged ETF provides the L multiplier internally, so we hold
        # target_eq_w fraction of NAV in 3USL — which gives target_eq_w × L
        # effective S&P exposure).
        target_eq_notional = target_eq_w * equity_value
        target_bd_notional = target_bd_w * equity_value

        # FX-aware unit conversion per project contract (see CLAUDE.md).
        # Each sleeve uses its own quote_ccy + fx_rate — never assume 1.0.
        target_eq_units = self._compute_target_units(
            target_eq_notional,
            self._last_equity_price,
            quote_ccy=self.config.equity_quote_ccy,
            fx_rate=self.config.fx_rate_equity_quote_to_base,
        )
        target_bd_units = self._compute_target_units(
            target_bd_notional,
            self._last_bond_price,
            quote_ccy=self.config.bond_quote_ccy,
            fx_rate=self.config.fx_rate_bond_quote_to_base,
        )

        if target_eq_units is None or target_bd_units is None:
            self.log.warning("Missing price for sizing — skipping rebalance")
            return

        current_eq_units = self._current_units(self.equity_id)
        current_bd_units = self._current_units(self.bond_id)

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
            self._submit_delta(self.bond_id, bd_delta)

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
