"""daily_summary/strategy.py — passive once-per-day portfolio rollup.

Posts a Slack / Telegram summary at two configurable times-of-day
(morning + evening) in the configured timezone, by:
  * subscribing to an existing bar type (default: AUD/JPY H1) as a clock
    tick source — H1 fires every hour around the clock for FX, so we
    always get a chance to fire shortly after the configured time
  * tracking the last calendar date each slot was sent (in the configured
    TZ) so we don't double-send within a single day
  * pulling the rollup data from the same in-process objects the trading
    strategies use: ``self.portfolio.account(...)``, ``self.cache.positions()``,
    and ``portfolio_risk_manager.get_summary()``

Doesn't trade. Doesn't subscribe to bars on instruments the trading
strategies aren't already using.

State persistence: the last-sent dates are stored as JSON in
``.tmp/daily_summary_last.json`` so a container restart doesn't double-send.
"""

from __future__ import annotations

import json
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
STATE_FILE = PROJECT_ROOT / ".tmp" / "daily_summary_last.json"


class DailySummaryConfig(StrategyConfig):
    """Configuration for the daily summary strategy."""

    # Bar type to subscribe to as a clock tick source. Default: AUD/JPY H1.
    bar_type: str = "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL"
    # Morning send time, in `summary_tz`.
    summary_hour: int = 9
    summary_minute: int = 0
    # Evening send time, in `summary_tz`. Set equal to summary_hour to disable.
    evening_summary_hour: int = 22
    evening_summary_minute: int = 0
    # IANA timezone name (e.g. Europe/London, America/New_York, UTC).
    summary_tz: str = "Europe/London"
    # Account ID to read balances for. Empty -> first account on the venue.
    account_id: str = ""
    # Initial account equity for P&L-since-inception calculation.
    initial_equity: float = 10_000.0
    initial_equity_ccy: str = "GBP"


class DailySummaryStrategy(Strategy):
    """Passive: subscribes to one bar type, sends morning + evening summaries."""

    def __init__(self, config: DailySummaryConfig) -> None:
        super().__init__(config)
        self.bar_type = BarType.from_str(config.bar_type)
        try:
            self._tz = ZoneInfo(config.summary_tz)
        except Exception:
            self._tz = ZoneInfo("UTC")
        # {slot: last_sent_date_str}  slot in ("morning", "evening")
        self._last_sent: dict[str, str] = {}
        self._load_state()

    def _load_state(self) -> None:
        try:
            if STATE_FILE.exists():
                raw = STATE_FILE.read_text(encoding="utf-8").strip()
                try:
                    self._last_sent = json.loads(raw)
                except json.JSONDecodeError:
                    # Legacy plain-date format — treat as morning date
                    if raw:
                        self._last_sent = {"morning": raw}
        except Exception:
            self._last_sent = {}

    def _persist_state(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(self._last_sent), encoding="utf-8")
        except Exception as e:
            self.log.warning(f"DailySummary: persist state failed: {e}")

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)
        self.log.info(
            f"DailySummary started | tick={self.bar_type} | "
            f"morning={self.config.summary_hour:02d}:{self.config.summary_minute:02d} "
            f"evening={self.config.evening_summary_hour:02d}:{self.config.evening_summary_minute:02d} "
            f"{self.config.summary_tz} | last_sent={self._last_sent}"
        )

    def on_stop(self) -> None:
        self.log.info("DailySummary stopped.")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type:
            return
        bar_ts_utc = unix_nanos_to_dt(bar.ts_event)
        local = bar_ts_utc.astimezone(self._tz)
        local_date_str = local.date().isoformat()
        local_minutes = local.hour * 60 + local.minute

        for slot, hour, minute, label in (
            ("morning", self.config.summary_hour, self.config.summary_minute, "🌅 Morning"),
            ("evening", self.config.evening_summary_hour, self.config.evening_summary_minute, "🌙 Evening"),
        ):
            # Skip if morning == evening (evening disabled)
            if slot == "evening" and (
                self.config.evening_summary_hour == self.config.summary_hour
                and self.config.evening_summary_minute == self.config.summary_minute
            ):
                continue
            if self._last_sent.get(slot) == local_date_str:
                continue
            if local_minutes < hour * 60 + minute:
                continue
            self._send_summary(local, local_date_str, slot, label)

    def _send_summary(
        self, local_dt: datetime, date_str: str, slot: str, label: str
    ) -> None:
        try:
            body = self._build_body(local_dt, label)
        except Exception as e:
            self.log.warning(f"DailySummary: build_body failed ({slot}): {e}")
            return
        try:
            n = notify_daily_summary(body)
            if n > 0:
                self._last_sent[slot] = date_str
                self._persist_state()
                self.log.info(f"DailySummary: sent {slot} to {n} backend(s) for {date_str}")
            else:
                self.log.warning(
                    "DailySummary: no backend configured (set SLACK_WEBHOOK_URL "
                    "or TELEGRAM_*); skipping"
                )
        except Exception as e:
            self.log.warning(f"DailySummary: send failed ({slot}): {e}")

    # ── Body builders ─────────────────────────────────────────────────────────

    def _build_body(self, local_dt: datetime, label: str) -> str:
        lines: list[str] = [
            f"{label}   _{local_dt:%Y-%m-%d %H:%M %Z}_",
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
                    total = float(bal.total.as_double())
                    free = float(bal.free.as_double())
                    locked = float(bal.locked.as_double())
                    out.append(
                        f"  • {ccy}: NLV {total:,.2f}   free {free:,.2f}   locked {locked:,.2f}"
                    )
                    # P&L since inception for the configured base currency
                    if str(ccy) == self.config.initial_equity_ccy:
                        initial = self.config.initial_equity
                        pnl = total - initial
                        pnl_pct = pnl / initial * 100 if initial else 0.0
                        sign = "+" if pnl >= 0 else ""
                        out.append(
                            f"  • P&L since inception: {sign}{pnl:,.2f} {ccy} "
                            f"({sign}{pnl_pct:.2f}%)"
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
                    avg = float(p.avg_px_open) if getattr(p, "avg_px_open", None) is not None else 0.0
                    line = f"  • `{p.instrument_id}`  {side_sign}{qty:g} @ avg {avg:,.4f}"

                    # Realized P&L (always available on Position)
                    try:
                        rpnl = p.realized_pnl
                        if rpnl is not None:
                            rv = float(rpnl)
                            rsign = "+" if rv >= 0 else ""
                            line += f"   realised {rsign}{rv:,.2f} {rpnl.currency}"
                    except Exception:
                        pass

                    # Unrealized P&L (requires last price from portfolio)
                    try:
                        upnl = self.portfolio.unrealized_pnl(p.instrument_id)
                        if upnl is not None:
                            uv = float(upnl)
                            usign = "+" if uv >= 0 else ""
                            line += f"   unrealised {usign}{uv:,.2f} {upnl.currency}"
                    except Exception:
                        pass

                    out.append(line)
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
