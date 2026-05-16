"""GEM Dual Momentum -- live NautilusTrader strategy.

Thin orchestration layer around :class:`titan.strategies.gem.live_logic.GemLiveLogic`.

Subscribes to SPY / EFA / IEF (and optional VIX / HYG) daily bars,
synchronises across legs, feeds GemLiveLogic, and reconciles open
positions against the target weights on every synchronised bar.

Causality contract preserved: weights are decisions at close[t-1] applied
to the position FROM close[t] onwards. NT's bar callback at close[t]
triggers reconciliation, so orders submitted now hit the next session's
fills -- exactly the research-side discipline.

V3.6 discipline applied:
  * L17 -- L21: documented in pre-reg directive §4.5
  * Equity tracking via StrategyEquityTracker (April 21 risk rewrite)
  * Halt persistence via portfolio_risk_manager
  * Rehydration on restart (Rehydration Bug 2026-05-11 pattern)

Selected production cell: C12 (C8 + max_leverage=2.0). See
``directives/Pre-Reg GEM Dual Momentum 2026-05-14.md`` §4.2.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from research.gem.gem_strategy import GemConfig
from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import StrategyEquityTracker, report_equity_and_check
from titan.strategies.gem.config import GemStrategyConfig
from titan.strategies.gem.live_logic import GemLiveLogic
from titan.strategies.gem.sizing import (
    SizingDecision,
    size_with_etf_only,
    size_with_mes,
)
from titan.utils.notification import (
    notify_order_event,
    notify_position_closed,
    notify_signal,
)

# Legs we always trade. VIX/HYG are signal-only (no orders).
_TRADED_TICKERS = ("SPY", "EFA", "IEF")


def _parse_lookback_blend(s: str) -> tuple[int, ...]:
    """Parse ``"3,6,12"`` into ``(3, 6, 12)``."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


class GemStrategy(Strategy):
    """Live NautilusTrader strategy implementing the GEM production cell.

    Current production cell (since 2026-05-16): **J5 P_hl60_vt05**
    (vol_estimator_halflife=60, ann_vol_target=0.05). See
    ``docs/strategies/gem-dual-momentum.md`` for the full audit lineage
    (C12 → J3 → J4 → J5) and ``directives/Pre-Reg J5 GEM Hybrid Re-audit
    2026-05-16.md`` for the current pre-reg directive.
    """

    config: GemStrategyConfig  # type-narrow

    def __init__(self, config: GemStrategyConfig) -> None:
        super().__init__(config)
        self._cfg = config

        # Resolve instrument ids + bar types.
        self.spy_id = InstrumentId.from_str(config.spy_instrument_id)
        self.efa_id = InstrumentId.from_str(config.efa_instrument_id)
        self.ief_id = InstrumentId.from_str(config.ief_instrument_id)
        self.mes_id = (
            InstrumentId.from_str(config.mes_instrument_id) if config.mes_instrument_id else None
        )
        self.vix_id = (
            InstrumentId.from_str(config.vix_instrument_id) if config.vix_instrument_id else None
        )
        self.hyg_id = (
            InstrumentId.from_str(config.hyg_instrument_id) if config.hyg_instrument_id else None
        )

        self.bar_type_spy = BarType.from_str(config.spy_bar_type_d)
        self.bar_type_efa = BarType.from_str(config.efa_bar_type_d)
        self.bar_type_ief = BarType.from_str(config.ief_bar_type_d)
        self.bar_type_vix = (
            BarType.from_str(config.vix_bar_type_d) if config.vix_bar_type_d else None
        )
        self.bar_type_hyg = (
            BarType.from_str(config.hyg_bar_type_d) if config.hyg_bar_type_d else None
        )

        # Translate the TOML-config into the research-side GemConfig that
        # GemLiveLogic expects. This is the single source of truth for the
        # strategy mechanism.
        self._gem_cfg = GemConfig(
            lookback_blend=_parse_lookback_blend(config.lookback_blend_str),
            absolute_gate_lookback_months=config.absolute_gate_lookback_months,
            buffer_pct=config.buffer_pct,
            defensive_switch=config.defensive_switch,
            ann_vol_target=config.ann_vol_target,
            vol_lookback_days=config.vol_lookback_days,
            max_leverage=config.max_leverage,
            # J4 (2026-05-15) noise-robust redesign options. A1_ewma_hl40
            # production cell sets vol_estimator_kind="ewma", halflife=40.
            vol_estimator_kind=config.vol_estimator_kind,
            vol_estimator_halflife=config.vol_estimator_halflife,
            max_weight_delta_per_bar=config.max_weight_delta_per_bar,
            vol_target_kind=config.vol_target_kind,
            vol_target_quantile=config.vol_target_quantile,
            vol_target_quantile_window=config.vol_target_quantile_window,
            stress_gate_enabled=config.stress_gate_enabled,
            stress_realised_vol_threshold=config.stress_realised_vol_threshold,
            stress_realised_vol_window=config.stress_realised_vol_window,
            stress_vix_threshold=config.stress_vix_threshold,
            stress_credit_z_threshold=config.stress_credit_z_threshold,
            stress_credit_z_window=config.stress_credit_z_window,
            dd_breaker_enabled=config.dd_breaker_enabled,
            dd_breaker_haircut_threshold=config.dd_breaker_haircut_threshold,
            dd_breaker_haircut_scale=config.dd_breaker_haircut_scale,
            dd_breaker_flat_threshold=config.dd_breaker_flat_threshold,
            dd_breaker_flat_bars=config.dd_breaker_flat_bars,
            dd_breaker_recovery_threshold=config.dd_breaker_recovery_threshold,
        )

        self._logic: GemLiveLogic | None = None
        self._prm_id = ""
        self._equity_tracker: StrategyEquityTracker | None = None

        # Bar-synchronisation state. Track last-seen date per leg; only
        # reconcile when all REQUIRED legs have a bar for the same date.
        self._last_bar_date: dict[str, date | None] = {
            "SPY": None,
            "EFA": None,
            "IEF": None,
            "VIX": None,
            "HYG": None,
        }
        self._last_close: dict[str, float] = {"SPY": 0.0, "EFA": 0.0, "IEF": 0.0}
        self._last_reconciled_date: date | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._prm_id = "gem_voltarget_lev2"
        portfolio_risk_manager.register_strategy(self._prm_id, self._cfg.initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self._cfg.initial_equity,
            base_ccy=self._cfg.base_ccy,
        )
        self._logic = GemLiveLogic(cfg=self._gem_cfg)

        self._warmup()

        # Subscribe to traded legs.
        self.subscribe_bars(self.bar_type_spy)
        self.subscribe_bars(self.bar_type_efa)
        self.subscribe_bars(self.bar_type_ief)
        # Subscribe to optional regime indicators.
        if self.bar_type_vix is not None:
            self.subscribe_bars(self.bar_type_vix)
        if self.bar_type_hyg is not None:
            self.subscribe_bars(self.bar_type_hyg)

        self._rehydrate_positions_from_broker()

        self.log.info(
            f"GEM started | execution={self._cfg.execution_mode} | "
            f"blend={self._gem_cfg.lookback_blend} | "
            f"vol_target={self._gem_cfg.ann_vol_target} | "
            f"max_leverage={self._gem_cfg.max_leverage} | "
            f"warmup_bars={self._logic.n_bars}"
        )

    def on_stop(self) -> None:
        for inst_id in (self.spy_id, self.efa_id, self.ief_id):
            self.cancel_all_orders(inst_id)
        if self.mes_id is not None:
            self.cancel_all_orders(self.mes_id)
        # Note: positions are NOT flat-closed on stop -- the operator may
        # restart the strategy and rehydrate. Flatten manually via the
        # kill_switch.py script if a hard close is required.
        self.log.info("GEM stopped.")

    def _warmup(self) -> None:
        """Load warmup history from parquets via GemLiveLogic.add_bars_dataframe.

        Loads the last ``warmup_bars`` for each traded leg + optional regime
        extras. The bulk-load path ensures the live class's history matches
        what gem_target_weights would see.
        """
        project_root = Path(__file__).resolve().parents[3]
        n = self._cfg.warmup_bars

        # Build the closes DataFrame. The strategy logic uses logical role
        # keys ("SPY"/"EFA"/"IEF"/"VIX"/"HYG") internally, but the parquet
        # files are named by physical ticker (config.ticker_*). Under the UK
        # UCITS universe those differ (e.g. ticker_spy="CSPX"). Load by
        # physical name then rename columns to logical roles.
        physical = [self._cfg.ticker_spy, self._cfg.ticker_efa, self._cfg.ticker_ief]
        closes_df = self._load_parquet_columns(project_root, tickers=physical, tail=n)
        if closes_df is None:
            self.log.warning("Warmup skipped -- one or more required parquets missing.")
            return
        closes_df = closes_df.rename(
            columns={
                self._cfg.ticker_spy: "SPY",
                self._cfg.ticker_efa: "EFA",
                self._cfg.ticker_ief: "IEF",
            }
        )
        # Add optional regime columns.
        if self.bar_type_vix is not None:
            vix_col = self._load_parquet_columns(
                project_root, tickers=[self._cfg.ticker_vix], tail=n
            )
            if vix_col is not None:
                closes_df["VIX"] = vix_col.iloc[:, 0].reindex(closes_df.index)
        if self.bar_type_hyg is not None:
            hyg_col = self._load_parquet_columns(
                project_root, tickers=[self._cfg.ticker_hyg], tail=n
            )
            if hyg_col is not None:
                closes_df["HYG"] = hyg_col.iloc[:, 0].reindex(closes_df.index)

        assert self._logic is not None
        self._logic.add_bars_dataframe(closes_df.dropna(subset=["SPY", "EFA", "IEF"]))
        self.log.info(
            f"Warmup loaded {self._logic.n_bars} bars from parquets "
            f"(VIX={'yes' if 'VIX' in closes_df.columns else 'no'}, "
            f"HYG={'yes' if 'HYG' in closes_df.columns else 'no'})."
        )

    def _load_parquet_columns(
        self, project_root: Path, tickers: list[str], tail: int
    ) -> pd.DataFrame | None:
        """Load `_D.parquet` close columns for each ticker, aligned on date."""
        parts: dict[str, pd.Series] = {}
        for tkr in tickers:
            path = project_root / "data" / f"{tkr}_D.parquet"
            if not path.exists():
                self.log.warning(f"Warmup parquet missing: {path}")
                return None
            df = pd.read_parquet(path).sort_index()
            col = "close" if "close" in df.columns else "Close"
            s = df[col].tail(tail).copy()
            s.name = tkr
            # Normalise to date-only (L20 of V3.6).
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            parts[tkr] = s
        return pd.concat(parts, axis=1).dropna(how="all")

    def _rehydrate_positions_from_broker(self) -> None:
        """Adopt any open broker positions for our instruments.

        Pattern from bond_gold's _rehydrate_position_from_broker, generalised
        to multiple instruments. We don't try to RECONSTRUCT internal state
        (the strategy is stateless beyond the GemLiveLogic history -- weights
        are recomputed from prices); we just log the current broker positions
        so the operator knows what they are.
        """
        try:
            for inst_id in (self.spy_id, self.efa_id, self.ief_id):
                positions = self.cache.positions(instrument_id=inst_id)
                open_pos = [p for p in positions if not p.is_closed]
                if open_pos:
                    qty = sum(float(p.signed_qty) for p in open_pos)
                    tags = sorted({str(p.strategy_id) for p in open_pos})
                    self.log.info(f"REHYDRATED: {inst_id} open qty={qty} (tags={tags})")
        except Exception as e:
            self.log.warning(f"_rehydrate_positions_from_broker failed: {e}")

    # ── Bar handling ─────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        """Route bars to the correct leg + trigger reconcile when synced."""
        bar_date = unix_nanos_to_dt(bar.ts_event).date()
        bar_type_str = str(bar.bar_type)
        close = float(bar.close)

        if bar.bar_type == self.bar_type_spy:
            self._last_bar_date["SPY"] = bar_date
            self._last_close["SPY"] = close
        elif bar.bar_type == self.bar_type_efa:
            self._last_bar_date["EFA"] = bar_date
            self._last_close["EFA"] = close
        elif bar.bar_type == self.bar_type_ief:
            self._last_bar_date["IEF"] = bar_date
            self._last_close["IEF"] = close
        elif self.bar_type_vix is not None and bar.bar_type == self.bar_type_vix:
            self._last_bar_date["VIX"] = bar_date
        elif self.bar_type_hyg is not None and bar.bar_type == self.bar_type_hyg:
            self._last_bar_date["HYG"] = bar_date
        else:
            self.log.warning(f"Unknown bar_type: {bar_type_str}")
            return

        # Only reconcile when all 3 required legs report the same date.
        sd = self._last_bar_date
        if sd["SPY"] != bar_date or sd["EFA"] != bar_date or sd["IEF"] != bar_date:
            return
        # Idempotent: skip if already reconciled for this date.
        if self._last_reconciled_date == bar_date:
            return
        self._last_reconciled_date = bar_date

        self._on_synced_bar(bar, bar_date)

    def _on_synced_bar(self, bar: Bar, bar_date: date) -> None:
        """All 3 legs have a bar for `bar_date`. Update logic + reconcile."""
        assert self._logic is not None

        # Append the new bar to logic. Skip if already present (warmup overlap).
        ts = pd.Timestamp(bar_date)
        if ts in self._logic._closes_history.index:
            pass
        else:
            self._logic.add_bar(
                ts,
                self._last_close["SPY"],
                self._last_close["EFA"],
                self._last_close["IEF"],
            )

        # Portfolio risk check (per-strategy equity + explicit timestamp).
        _, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
        if halted:
            self.log.warning("Portfolio kill switch -- flattening all legs.")
            for inst_id in (self.spy_id, self.efa_id, self.ief_id):
                self.close_all_positions(inst_id)
            if self.mes_id is not None:
                self.close_all_positions(self.mes_id)
            return

        # Allocator tick (rebalance allocator weights).
        portfolio_allocator.tick(now=bar_date)

        # Pull target weights from logic + size them.
        target_w = self._logic.current_weights()
        equity = self._equity_tracker.current_equity() if self._equity_tracker else 0.0
        if equity <= 0:
            self.log.warning("Equity <= 0; skipping reconcile.")
            return

        alloc = portfolio_allocator.get_weight(self._prm_id)
        scaled_nav = equity * alloc * portfolio_risk_manager.scale_factor

        decisions = self._size(target_w, scaled_nav)
        self._reconcile_to_target(decisions, target_w, bar_date)

    def _size(self, target_w: dict[str, float], nav: float) -> list[SizingDecision]:
        """Translate weights -> contract/share counts per execution_mode."""
        if self._cfg.execution_mode == "mes":
            return size_with_mes(
                target_w,
                nav_usd=nav,
                es_price=self._last_close["SPY"],  # use SPY close as ES proxy
                efa_price=self._last_close["EFA"],
                ief_price=self._last_close["IEF"],
            )
        return size_with_etf_only(
            target_w,
            nav_usd=nav,
            spy_price=self._last_close["SPY"],
            efa_price=self._last_close["EFA"],
            ief_price=self._last_close["IEF"],
        )

    def _reconcile_to_target(
        self,
        decisions: list[SizingDecision],
        target_w: dict[str, float],
        bar_date: date,
    ) -> None:
        """Compare current positions to target and submit reconciling orders.

        Buffering: only submit if absolute weight delta > rebalance_threshold_weight.
        This avoids vol-target noise creating constant small orders.
        """
        wanted_by_sym = {d.symbol: d for d in decisions}

        # Map symbol -> instrument id.
        sym_to_id = {
            "SPY": self.spy_id,
            "EFA": self.efa_id,
            "IEF": self.ief_id,
        }
        if self.mes_id is not None:
            sym_to_id["MES"] = self.mes_id

        # For each symbol we COULD trade, compute the order delta.
        all_syms = set(sym_to_id.keys()) | set(wanted_by_sym.keys())
        for sym in sorted(all_syms):
            if sym not in sym_to_id:
                continue
            inst_id = sym_to_id[sym]
            wanted = wanted_by_sym.get(sym)
            wanted_qty = wanted.rounded_quantity if wanted else 0

            # Current position (signed quantity across all open positions).
            open_pos = [p for p in self.cache.positions(instrument_id=inst_id) if p.is_open]
            current_qty = int(sum(float(p.signed_qty) for p in open_pos))

            delta = wanted_qty - current_qty
            if delta == 0:
                continue

            # Buffering: skip small adjustments when expressed as a weight.
            weight_now = target_w.get(self._symbol_to_weight_key(sym), 0.0)
            weight_threshold = self._cfg.rebalance_threshold_weight
            # Approximate the current weight delta from qty delta.
            # For SPY/EFA/IEF: weight ≈ qty * price / nav. We allow any
            # delta when wanted_qty is 0 (closing position).
            if wanted_qty != 0 and current_qty != 0:
                relative_delta = abs(delta) / max(abs(wanted_qty), 1)
                if relative_delta < weight_threshold / max(abs(weight_now), 0.05):
                    continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            qty = Quantity.from_int(abs(delta))
            order = self.order_factory.market(
                instrument_id=inst_id,
                order_side=side,
                quantity=qty,
                time_in_force=TimeInForce.DAY,
            )
            try:
                notify_signal(
                    strategy=self._prm_id,
                    action=("BUY" if side == OrderSide.BUY else "SELL"),
                    instrument=str(inst_id),
                    qty=abs(delta),
                    price=self._last_close.get(self._symbol_to_weight_key(sym), 0.0) or None,
                    notional=(wanted.target_notional_usd if wanted else 0.0),
                    notional_ccy=self._cfg.base_ccy,
                    equity=(
                        self._equity_tracker.current_equity()
                        if self._equity_tracker is not None
                        else None
                    ),
                    equity_ccy=self._cfg.base_ccy,
                    reason={
                        "target_weight": weight_now,
                        "wanted_qty": wanted_qty,
                        "current_qty": current_qty,
                        "execution_mode": self._cfg.execution_mode,
                        "bar_date": str(bar_date),
                    },
                )
            except Exception as e:
                self.log.warning(f"notify_signal failed: {e}")
            self.submit_order(order)
            self.log.info(
                f"RECONCILE {sym}: {side.name} {abs(delta)} "
                f"(target={wanted_qty}, current={current_qty}, weight={weight_now:.3f})"
            )

    @staticmethod
    def _symbol_to_weight_key(sym: str) -> str:
        """MES weight corresponds to SPY weight key from the strategy."""
        return "SPY" if sym == "MES" else sym

    # ── Order / position lifecycle events ────────────────────────────────

    def on_position_closed(self, event) -> None:
        if self._equity_tracker is None:
            return
        try:
            pnl_usd = float(event.realized_pnl.as_double())
            self._equity_tracker.on_position_closed(pnl_usd, fx_to_base=1.0)
        except Exception as e:
            self.log.warning(f"tracker on_position_closed failed: {e}")
            return
        try:
            notify_position_closed(
                strategy=self._prm_id,
                instrument=str(event.instrument_id) if hasattr(event, "instrument_id") else "?",
                direction="LONG",
                realized_pnl=pnl_usd,
                realized_pnl_ccy=self._cfg.base_ccy,
                equity_after=self._equity_tracker.current_equity(),
                initial_equity=self._cfg.initial_equity,
                equity_ccy=self._cfg.base_ccy,
            )
        except Exception as e:
            self.log.warning(f"notify_position_closed failed: {e}")

    def on_order_accepted(self, event) -> None:
        try:
            order = self.cache.order(event.client_order_id)
            side = str(getattr(order, "side", "?")).split(".")[-1] if order else "?"
            qty = int(order.quantity) if order and order.quantity else 0
            notify_order_event(
                strategy=self._prm_id,
                event_type="accepted",
                instrument=str(order.instrument_id) if order else "?",
                side=side,
                qty=qty,
                venue_order_id=str(getattr(event, "venue_order_id", "") or ""),
                client_order_id=str(getattr(event, "client_order_id", "") or ""),
            )
        except Exception as e:
            self.log.warning(f"notify_order_event(accepted) failed: {e}")

    def on_order_filled(self, event) -> None:
        try:
            order = self.cache.order(event.client_order_id)
            side = str(getattr(order, "side", "?")).split(".")[-1] if order else "?"
            qty = int(getattr(event, "last_qty", 0) or 0) or (
                int(order.quantity) if order and order.quantity else 0
            )
            fill_px = float(getattr(event, "last_px", 0) or 0) or None
            notify_order_event(
                strategy=self._prm_id,
                event_type="filled",
                instrument=str(order.instrument_id) if order else "?",
                side=side,
                qty=qty,
                price=fill_px,
                venue_order_id=str(getattr(event, "venue_order_id", "") or ""),
                client_order_id=str(getattr(event, "client_order_id", "") or ""),
            )
        except Exception as e:
            self.log.warning(f"notify_order_event(filled) failed: {e}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
        try:
            order = self.cache.order(event.client_order_id)
            side = str(getattr(order, "side", "?")).split(".")[-1] if order else "?"
            qty = int(order.quantity) if order and order.quantity else 0
            notify_order_event(
                strategy=self._prm_id,
                event_type="rejected",
                instrument=str(order.instrument_id) if order else "?",
                side=side,
                qty=qty,
                client_order_id=str(getattr(event, "client_order_id", "") or ""),
                note=str(getattr(event, "reason", "") or ""),
            )
        except Exception as e:
            self.log.warning(f"notify_order_event(rejected) failed: {e}")
