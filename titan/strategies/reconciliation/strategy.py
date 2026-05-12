"""reconciliation/strategy.py — passive position-drift watchdog.

Tier 1.1 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).

Subscribes to a clock-tick bar (default AUD/JPY H1) and on every fire
runs a battery of cheap drift checks against the in-process NT cache:

  D1.  **Multiple open positions per instrument.** A single instrument
       holding more than one open position usually means an EXTERNAL
       rehydrated position is sitting alongside a strategy-tagged
       position — which is exactly the symptom of the May 11 bug
       (BondGoldStrategy doubled its inventory because the entry guard
       didn't recognise the EXTERNAL position).

  D2.  **Cache-vs-portfolio sum drift.** If the sum of signed quantities
       across all cache positions for an instrument differs from
       ``portfolio.net_position(instrument_id)``, NT's internal
       reconciliation has diverged from its own portfolio accounting.
       Should never happen; fire loud if it does.

  D3.  **Stale open orders.** Any order in ``ACCEPTED`` / ``PENDING_*``
       state for longer than ``stale_order_minutes`` (default 15) is
       worth flagging. Most legitimate orders fill within seconds; a
       stale order is usually a forgotten partial fill, a cancelled-but-
       acknowledged-at-exchange ghost, or a strategy that submitted an
       order then died before tracking it.

  D4.  **NLV drop.** Tracks broker NLV (per base currency) across
       reconciliation cycles and alerts on absolute drops greater than
       ``nlv_drop_threshold_pct`` (default 2%) since the previous
       sample. Catches sudden losses, phantom-position margin lockup,
       and broker accounting glitches. The first sample seeds the
       baseline (no alert).

  D5.  **Shadow-decision divergence (Tier 2.1).** For each
       bond_gold-class strategy in the portfolio, computes the
       *expected* action (entry/exit/hold) by running the shared
       decision primitives in ``titan.utils.bond_gold_decisions`` against
       the latest signal-instrument parquet, then compares to the
       observed cache state of the trade instrument. Catches:
         - Strategy was supposed to be long but cache is flat
           (missed entry, broker rejection, race condition)
         - Strategy was supposed to be flat but cache shows a long
           position (May 11 doubling pattern, late exit)
       Flagged only as ``[D5]`` warnings — does not gate the alert
       cooldown (so a persistent shadow divergence still pages once
       per ``alert_cooldown_minutes``).

Doesn't trade. Doesn't subscribe to anything the trading strategies
aren't already using. Cost on the message-bus is one cache scan per H1
bar.

State persistence: tracks last-seen alert hash in
``.tmp/reconciliation_last.json`` so a transient drift that resolves on
its own doesn't repeatedly spam the channel.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.trading.strategy import Strategy

from titan.utils.notification import notify_health

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = PROJECT_ROOT / ".tmp" / "reconciliation_last.json"


class ReconciliationConfig(StrategyConfig):
    """Configuration for the position-reconciliation watchdog."""

    # Bar type to subscribe to as a clock tick source. Default: AUD/JPY H1
    # (already subscribed-to by other strategies, so no extra IBKR cost).
    bar_type: str = "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL"
    # Threshold in absolute quantity units below which sum-drift is
    # ignored as floating-point noise.
    qty_epsilon: float = 1e-6
    # Open orders older than this are flagged as stale.
    stale_order_minutes: int = 15
    # Minimum minutes between identical alerts (suppress repeat spam if
    # a drift persists across several bars).
    alert_cooldown_minutes: int = 60
    # NLV-drop threshold (decimal, e.g. 0.02 = 2%). Drops greater than
    # this between successive reconciliation cycles trigger D4. Leave
    # generous (1-2%) to avoid false alarms from normal MTM swings on a
    # leveraged portfolio.
    nlv_drop_threshold_pct: float = 0.02
    # D5 (Tier 2.1) shadow-decision check. Set False to disable when
    # the bond_gold-class strategies aren't deployed (avoids spurious
    # warnings from missing parquets). Auto-skips per-strategy when the
    # signal-instrument parquet is missing or trade-instrument has no
    # cache positions, so leaving True is generally safe.
    shadow_check_enabled: bool = True


class ReconciliationStrategy(Strategy):
    """Detects position-state drift between strategy view, NT cache, and
    portfolio accounting; alerts on any mismatch."""

    def __init__(self, config: ReconciliationConfig) -> None:
        super().__init__(config)
        self.bar_type = BarType.from_str(config.bar_type)
        self._last_alert_hash: str | None = None
        self._last_alert_ts_ns: int = 0
        # NLV baseline per currency, e.g. {"GBP": 9882.45}. None until first sample.
        self._last_nlv: dict[str, float] = {}
        self._load_state()

    def _load_state(self) -> None:
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                self._last_alert_hash = data.get("hash")
                self._last_alert_ts_ns = int(data.get("ts_ns", 0))
                self._last_nlv = dict(data.get("last_nlv", {}))
        except Exception:
            self._last_alert_hash = None
            self._last_alert_ts_ns = 0
            self._last_nlv = {}

    def _persist_state(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(
                json.dumps(
                    {
                        "hash": self._last_alert_hash,
                        "ts_ns": self._last_alert_ts_ns,
                        "last_nlv": self._last_nlv,
                    }
                ),
                encoding="utf-8",
            )
        except Exception as e:
            self.log.warning(f"Reconciliation: persist state failed: {e}")

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)
        self.log.info(
            f"Reconciliation watchdog started | tick={self.bar_type} | "
            f"stale_order_minutes={self.config.stale_order_minutes} | "
            f"cooldown_minutes={self.config.alert_cooldown_minutes}"
        )

    def on_stop(self) -> None:
        self.log.info("Reconciliation watchdog stopped.")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type:
            return
        try:
            self._reconcile(bar.ts_event)
        except Exception as e:
            # Watchdog must NEVER crash the runtime. Log loudly, swallow.
            self.log.error(f"Reconciliation scan failed: {e}")

    def _sample_nlv(self) -> dict[str, float]:
        """Return current broker NLV per currency (e.g. {"GBP": 9882.45}).

        Uses ``self.portfolio.account(venue)`` for the bar-type's venue,
        falling back to ``self.cache.accounts()`` if the portfolio lookup
        fails. Returns an empty dict on any error so D4 simply skips.
        """
        try:
            account = None
            try:
                account = self.portfolio.account(self.bar_type.instrument_id.venue)
            except Exception:
                pass
            if account is None:
                try:
                    accounts = list(self.cache.accounts())
                    account = accounts[0] if accounts else None
                except Exception:
                    return {}
            if account is None:
                return {}
            try:
                balances = account.balances()
            except Exception:
                return {}
            return {str(ccy): float(bal.total.as_double()) for ccy, bal in balances.items()}
        except Exception as e:
            self.log.warning(f"Reconciliation: _sample_nlv failed: {e}")
            return {}

    def _shadow_decision_check(self) -> list[str]:
        """D5: For each live bond_gold-class strategy, compute the shadow
        expected action and compare to the cache's actual position state.

        Imports are local to the method so a missing optional import
        (e.g. shared decision module renamed) doesn't break the watchdog
        startup. Returns a list of finding strings (empty if all match).

        Mismatch semantics for v1:
          - Expected ENTRY but cache shows flat → "missed entry"
          - Expected EXIT but cache shows long  → "missed exit"
          - Expected HOLD-while-flat but cache shows long → "phantom long"
          - Expected HOLD-while-long but cache shows flat → "phantom flat"

        Notes:
          - We can't reconstruct the strategy's internal ``bars_held``
            from outside, so the v1 check assumes we're past min-hold.
            This means it can't tell the difference between
            "strategy is correctly waiting out hold_days" and
            "strategy missed an exit". Concretely: after an entry, if
            z drops below threshold WITHIN hold_days the live strategy
            holds (correct) but D5 may say "missed exit". Tunable in
            v2 by tracking last entry timestamp from fills.
        """
        findings: list[str] = []
        try:
            from pathlib import Path

            from titan.utils.bond_gold_decisions import (
                LIVE_CONFIGS,
                compute_z_score,
                load_signal_closes_from_parquet,
            )
        except Exception as e:
            self.log.debug(f"Reconciliation D5: shared decision module unavailable: {e}")
            return findings

        # Project root: titan/strategies/reconciliation/strategy.py → up 3
        data_dir = Path(__file__).resolve().parents[3] / "data"

        for cfg in LIVE_CONFIGS.values():
            try:
                closes = load_signal_closes_from_parquet(cfg.signal_ticker, data_dir)
                if closes is None:
                    # Parquet missing — skip this strategy silently.
                    continue
                z = compute_z_score(
                    closes,
                    lookback=cfg.lookback,
                    zscore_window=cfg.zscore_window,
                )
                if z is None:
                    continue

                # Look up cache state for the trade instrument. We don't
                # have a direct InstrumentId, only the bare symbol. NT's
                # cache.positions() iterates all positions; we filter by
                # the symbol component.
                trade_symbol = cfg.trade_symbol
                open_pos_for_symbol: list = []
                for p in self.cache.positions():
                    if not p.is_open:
                        continue
                    if str(p.instrument_id).split(".", 1)[0] == trade_symbol:
                        open_pos_for_symbol.append(p)
                net_qty = sum(float(p.signed_qty) for p in open_pos_for_symbol)
                cache_is_long = net_qty > 0

                # Expected directional state given current z (independent
                # of bars_held timing — see method docstring).
                if z > cfg.threshold and not cache_is_long:
                    findings.append(
                        f"[D5 shadow {cfg.name}] z={z:+.3f} > threshold "
                        f"{cfg.threshold} but cache is FLAT for {trade_symbol} — "
                        f"possible missed entry or broker rejection"
                    )
                elif z <= cfg.threshold and cache_is_long:
                    # Could be legitimate "still inside hold_days" — flag
                    # but mark as informational by including 'or-hold'.
                    findings.append(
                        f"[D5 shadow {cfg.name}] z={z:+.3f} <= threshold "
                        f"{cfg.threshold} but cache shows LONG {net_qty:+.0f} "
                        f"{trade_symbol} (may be inside hold_days={cfg.hold_days})"
                    )
                # else: agreement (long-and-bullish, or flat-and-bearish)
            except Exception as e:
                self.log.warning(f"Reconciliation D5 ({cfg.name}) failed: {e}")
        return findings

    # ── core reconciliation logic ─────────────────────────────────────

    def _reconcile(self, ts_event_ns: int) -> None:
        findings: list[str] = []
        bar_dt = unix_nanos_to_dt(ts_event_ns)

        open_positions = [p for p in self.cache.positions() if p.is_open]
        # Group by instrument
        by_instrument: dict = defaultdict(list)
        for p in open_positions:
            by_instrument[p.instrument_id].append(p)

        # D1. Multiple open positions per instrument
        for instrument_id, positions in by_instrument.items():
            if len(positions) > 1:
                pos_summary = ", ".join(
                    f"{str(p.strategy_id)}={float(p.signed_qty):+.0f}" for p in positions
                )
                findings.append(
                    f"[D1 multi-position] {instrument_id} has {len(positions)} open "
                    f"positions: {pos_summary}. Likely orphan + strategy double-up."
                )

        # D2. Cache sum vs portfolio.net_position
        for instrument_id, positions in by_instrument.items():
            cache_sum = sum(float(p.signed_qty) for p in positions)
            try:
                net = self.portfolio.net_position(instrument_id)
                portfolio_net = float(net) if net is not None else 0.0
            except Exception as e:
                self.log.warning(
                    f"Reconciliation: portfolio.net_position({instrument_id}) failed: {e}"
                )
                continue
            if abs(cache_sum - portfolio_net) > self.config.qty_epsilon:
                findings.append(
                    f"[D2 sum-drift] {instrument_id}: cache_sum={cache_sum:+.4f} "
                    f"vs portfolio.net_position={portfolio_net:+.4f}"
                )

        # D3. Stale open orders
        try:
            open_orders = [o for o in self.cache.orders() if o.is_open]
        except Exception as e:
            self.log.warning(f"Reconciliation: cache.orders() failed: {e}")
            open_orders = []
        stale_threshold_ns = ts_event_ns - self.config.stale_order_minutes * 60 * 1_000_000_000
        for order in open_orders:
            ts_init = getattr(order, "ts_init", None)
            if ts_init is None:
                continue
            if int(ts_init) < stale_threshold_ns:
                age_min = (ts_event_ns - int(ts_init)) / 60_000_000_000
                client_oid = str(getattr(order, "client_order_id", "?"))
                instrument = str(getattr(order, "instrument_id", "?"))
                findings.append(
                    f"[D3 stale-order] {instrument} order {client_oid} open for "
                    f"{age_min:.0f} minutes (status={order.status})"
                )

        # D4. NLV drop. Sample broker NLV per currency; alert on absolute
        # drops greater than nlv_drop_threshold_pct since previous sample.
        # First sample seeds the baseline (no alert).
        nlv_now = self._sample_nlv()
        nlv_changed = False
        for ccy, nlv_curr in nlv_now.items():
            nlv_prev = self._last_nlv.get(ccy)
            if nlv_prev is None or nlv_prev <= 0:
                continue
            drop_pct = (nlv_prev - nlv_curr) / nlv_prev
            if drop_pct > self.config.nlv_drop_threshold_pct:
                findings.append(
                    f"[D4 nlv-drop] {ccy}: NLV dropped {drop_pct * 100:.2f}% from "
                    f"{nlv_prev:,.2f} to {nlv_curr:,.2f} (threshold "
                    f"{self.config.nlv_drop_threshold_pct * 100:.1f}%)"
                )
        # Always update baseline so the next cycle compares against the
        # most recent sample (alert is "since last bar", not "since launch").
        if nlv_now != self._last_nlv:
            self._last_nlv = nlv_now
            nlv_changed = True

        # D5. Shadow-decision divergence (Tier 2.1). Compares cache state
        # of each bond_gold-class strategy's trade instrument against
        # the action the shared decision primitives would say is
        # current. Misses (expected long but flat) and over-reaches
        # (flat-expected but long) both fire warnings.
        if self.config.shadow_check_enabled:
            shadow_findings = self._shadow_decision_check()
            findings.extend(shadow_findings)

        if not findings:
            if nlv_changed:
                # Persist the updated baseline even when no alert fires.
                self._persist_state()
            self.log.debug(
                f"Reconciliation OK at {bar_dt.isoformat()}: "
                f"{len(open_positions)} open positions, {len(open_orders)} open orders, "
                f"NLV {nlv_now}"
            )
            return

        # Build alert hash to suppress repeats during cooldown
        alert_hash = "|".join(sorted(findings))
        cooldown_ns = self.config.alert_cooldown_minutes * 60 * 1_000_000_000
        if (
            alert_hash == self._last_alert_hash
            and (ts_event_ns - self._last_alert_ts_ns) < cooldown_ns
        ):
            self.log.warning(
                f"Reconciliation findings unchanged (cooldown "
                f"{self.config.alert_cooldown_minutes}m); skipping repeat alert. "
                f"Findings:\n  " + "\n  ".join(findings)
            )
            return

        self._last_alert_hash = alert_hash
        self._last_alert_ts_ns = ts_event_ns
        self._persist_state()

        msg = "Position-reconciliation watchdog detected drift:\n  " + "\n  ".join(findings)
        self.log.error(msg)
        try:
            notify_health(
                event="reconciliation_drift",
                severity="critical",
                detail=msg,
            )
        except Exception as e:
            self.log.warning(f"Reconciliation: notify_health failed: {e}")
