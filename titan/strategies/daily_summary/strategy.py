"""daily_summary/strategy.py — passive once-per-day portfolio rollup.

Posts a Slack / Telegram summary at a configured time-of-day in the
configured timezone, by:
  * subscribing to an existing bar type (default: AUD/JPY H1) as a clock
    tick source — H1 fires every hour around the clock for FX, so we
    always get a chance to fire shortly after the configured time
  * tracking the last calendar date a summary was sent (in the configured
    TZ) so we don't double-send within a single day
  * pulling the rollup data from the same in-process objects the trading
    strategies use: ``self.portfolio.account(...)``, ``self.cache.positions()``,
    and ``portfolio_risk_manager.get_summary()``

Doesn't trade. Doesn't subscribe to bars on instruments the trading
strategies aren't already using.

State persistence: the last-sent date is stored in
``.tmp/daily_summary_last.txt`` so a container restart doesn't double-send.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.trading.strategy import Strategy

from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.utils.notification import notify_daily_summary

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = PROJECT_ROOT / ".tmp" / "daily_summary_last.txt"


class DailySummaryConfig(StrategyConfig):
    """Configuration for the daily summary strategy."""

    # Bar type to subscribe to as a clock tick source. Default: AUD/JPY H1.
    bar_type: str = "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL"
    # Time-of-day to send the summary, in `tz`.
    summary_hour: int = 9
    summary_minute: int = 0
    # IANA timezone name (e.g. Europe/London, America/New_York, UTC).
    summary_tz: str = "Europe/London"
    # Account ID to read balances for. Empty -> first account on the venue.
    account_id: str = ""


class DailySummaryStrategy(Strategy):
    """Passive: subscribes to one bar type, sends a daily summary."""

    def __init__(self, config: DailySummaryConfig) -> None:
        super().__init__(config)
        self.bar_type = BarType.from_str(config.bar_type)
        try:
            self._tz = ZoneInfo(config.summary_tz)
        except Exception:
            self._tz = ZoneInfo("UTC")
        self._last_sent_date: str | None = None
        # Restore last-sent date from disk so a container restart doesn't
        # cause a duplicate summary on the same day.
        try:
            if STATE_FILE.exists():
                self._last_sent_date = STATE_FILE.read_text(encoding="utf-8").strip() or None
        except Exception:
            self._last_sent_date = None

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)
        self.log.info(
            f"DailySummary started | tick={self.bar_type} | "
            f"send_at={self.config.summary_hour:02d}:{self.config.summary_minute:02d} "
            f"{self.config.summary_tz} | last_sent={self._last_sent_date}"
        )

    def on_stop(self) -> None:
        self.log.info("DailySummary stopped.")

    def on_bar(self, bar: Bar) -> None:
        # Only act on our designated tick-source bar type
        if bar.bar_type != self.bar_type:
            return
        bar_ts_utc = unix_nanos_to_dt(bar.ts_event)
        local = bar_ts_utc.astimezone(self._tz)
        local_date_str = local.date().isoformat()

        # Already sent today?
        if self._last_sent_date == local_date_str:
            return

        # Past the configured send time?
        target_minutes = self.config.summary_hour * 60 + self.config.summary_minute
        local_minutes = local.hour * 60 + local.minute
        if local_minutes < target_minutes:
            return

        # Build and send
        try:
            body = self._build_body(local)
        except Exception as e:
            self.log.warning(f"daily_summary: build_body failed: {e}")
            return

        try:
            n = notify_daily_summary(body)
            if n > 0:
                self._last_sent_date = local_date_str
                self._persist_last_sent_date(local_date_str)
                self.log.info(f"DailySummary: sent to {n} backend(s) for {local_date_str}")
            else:
                self.log.warning(
                    "DailySummary: no backend configured (set SLACK_WEBHOOK_URL "
                    "or TELEGRAM_*); skipping"
                )
        except Exception as e:
            self.log.warning(f"daily_summary: send failed: {e}")

    def _persist_last_sent_date(self, date_str: str) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(date_str, encoding="utf-8")
        except Exception as e:
            self.log.warning(f"daily_summary: persist last-sent failed: {e}")

    # ── Body builders ─────────────────────────────────────────────────────────

    def _build_body(self, local_dt: datetime) -> str:
        lines: list[str] = [
            f"_{local_dt:%Y-%m-%d %H:%M %Z}_",
            "",
        ]
        lines.extend(self._account_lines())
        lines.extend(self._position_lines())
        lines.extend(self._strategy_lines())
        lines.extend(self._health_lines())
        return "\n".join(lines)

    def _account_lines(self) -> list[str]:
        out: list[str] = ["*Account*"]
        try:
            # Try resolving via venue first (typical pattern)
            account = None
            for v in (None,):  # placeholder loop — fallback path
                try:
                    account = self.portfolio.account(self.bar_type.instrument_id.venue)
                    if account is not None:
                        break
                except Exception:
                    pass
            if account is None:
                # Last-ditch: scan cache
                try:
                    accounts = list(self.cache.accounts())
                    account = accounts[0] if accounts else None
                except Exception:
                    account = None

            if account is None:
                out.append("  • (no account state available)")
                return out + [""]

            try:
                out.append(f"  • Account ID: `{account.id}`")
            except Exception:
                pass

            try:
                balances = account.balances()
            except Exception:
                balances = {}

            for ccy, bal in balances.items():
                try:
                    total = float(account.balance_total(ccy).as_double())
                    free = float(account.balance_free(ccy).as_double())
                    locked = float(account.balance_locked(ccy).as_double())
                    out.append(
                        f"  • {ccy}: NLV {total:,.2f}   free {free:,.2f}   locked {locked:,.2f}"
                    )
                except Exception:
                    pass
        except Exception as e:
            out.append(f"  • (account read failed: {e})")
        out.append("")
        return out

    def _position_lines(self) -> list[str]:
        out: list[str] = []
        try:
            positions = list(self.cache.positions())
            open_pos = [p for p in positions if not p.is_closed]
            if not open_pos:
                out.append("*Open positions*: none")
                out.append("")
                return out
            out.append(f"*Open positions* ({len(open_pos)})")
            for p in open_pos:
                try:
                    qty = float(p.signed_qty)
                    side_sign = "+" if qty > 0 else ""
                    avg = (
                        float(p.avg_px_open) if getattr(p, "avg_px_open", None) is not None else 0.0
                    )
                    out.append(f"  • `{p.instrument_id}`  {side_sign}{qty:g} @ avg {avg:,.4f}")
                except Exception:
                    out.append(f"  • {p.instrument_id} (details unavailable)")
        except Exception as e:
            out.append(f"*Open positions*: (read failed: {e})")
        out.append("")
        return out

    def _strategy_lines(self) -> list[str]:
        out: list[str] = ["*Strategies (PRM)*"]
        try:
            summary = portfolio_risk_manager.get_summary()
            total = summary.get("total_equity", 0.0)
            halt = summary.get("halt_all", False)
            scale = summary.get("scale_factor", 1.0)
            ann_vol = summary.get("realized_vol_ann_pct")
            out.append(
                f"  • Total equity: {total:,.2f}   "
                f"scale: {scale:.2f}   "
                f"halt: {'YES' if halt else 'no'}"
            )
            if ann_vol is not None:
                out.append(f"  • Realised ann vol: {ann_vol:.2f}%")
            for sid, s in summary.get("strategies", {}).items():
                eq = s.get("equity", 0.0)
                dd = s.get("drawdown_pct", 0.0)
                wt = s.get("weight_pct", 0.0)
                out.append(f"  • `{sid}`: equity {eq:,.2f}   DD {dd:+.2f}%   weight {wt:.1f}%")
        except Exception as e:
            out.append(f"  • (PRM read failed: {e})")
        out.append("")
        return out

    def _health_lines(self) -> list[str]:
        out: list[str] = ["*Health*"]
        try:
            halt_file = PROJECT_ROOT / ".tmp" / "portfolio_halt.json"
            out.append(f"  • Halt file present: {halt_file.exists()}")
        except Exception:
            pass
        out.append(f"  • Tick source: `{self.bar_type}`")
        return out
