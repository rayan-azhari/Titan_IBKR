"""cost_audit.py -- weekly comparison of modelled vs realised IBKR fill costs.

Tier 2.4 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).

The May 2026 cost-model audit found that strategies were modelling 5 bps
per fill when realised IBKR commissions averaged 12-117 bps for small
notionals (~$3k strategy equity at $4 minimum commission). This script
runs that comparison automatically every week (or on-demand) and alerts
when the average drift exceeds a configurable threshold.

Workflow::

    1. Connect to IBKR (fresh client_id 95) and request the last N days
       of execution reports via ``reqExecutions()``.
    2. For each fill, look up the strategy/instrument's modelled
       per-fill cost via ``realistic_cost_bps()`` (the same function the
       backtests use).
    3. Compute realised cost = actual IBKR commission (per-fill, in
       account base currency).
    4. Compare:  drift_pct = (realised - modelled) / modelled
       Aggregated by (strategy, instrument).
    5. If average absolute drift exceeds ``--alert-threshold`` (default
       50%), fire ``notify_health(severity="warning")``.

Output:
    - Console table with per-fill detail (when --verbose)
    - Aggregated summary always printed
    - .tmp/reports/cost_audit/audit_YYYY-MM-DD.csv saved for trail
    - Slack/Telegram notification if drift exceeds threshold

Usage (host or in-container)::

    uv run python scripts/cost_audit.py --days 7
    uv run python scripts/cost_audit.py --days 30 --alert-threshold 0.30
    uv run python scripts/cost_audit.py --days 7 --dry-run --verbose

Cron suggestion (Sunday 02:00 UTC):

    0 2 * * 0  cd /app && uv run python scripts/cost_audit.py --days 7

Exit codes:
    0  No alert (drift within threshold or no fills in window)
    1  Drift exceeded threshold (notification sent if not --dry-run)
    2  Operational error (gateway unreachable, ibapi failure)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.execution import ExecutionFilter
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.cross_asset.run_bond_equity_wfo_realistic import (  # noqa: E402
    SPREAD_BPS_PER_SIDE,
    realistic_cost_bps,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "cost_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IBKR_HOST = os.getenv("IBKR_HOST_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT_HOST", "4004"))
CLIENT_ID = 95  # one-off, distinct from strategy=7, kill=98, cspx=99, orphans=97, snapshot=96


@dataclass
class Fill:
    """One execution leg from IBKR."""

    exec_id: str
    time_utc: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    price: float
    avg_price: float
    commission_ccy: str = ""
    commission_amt: float = 0.0
    realized_pnl: float = 0.0
    yield_redemption_amt: float = 0.0  # unused but populated by IBKR


class AuditApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.fills: dict[str, Fill] = {}  # exec_id -> Fill
        self.exec_done = False

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self.ready = True

    def execDetails(self, reqId, contract: Contract, execution) -> None:  # noqa: N802
        try:
            ts = datetime.strptime(execution.time[:17], "%Y%m%d  %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except Exception:
            try:
                ts = datetime.strptime(execution.time[:17], "%Y%m%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                ts = datetime.now(timezone.utc)
        self.fills[execution.execId] = Fill(
            exec_id=execution.execId,
            time_utc=ts,
            symbol=contract.symbol,
            side=execution.side,
            qty=float(execution.shares),
            price=float(execution.price),
            avg_price=float(execution.avgPrice),
        )

    def execDetailsEnd(self, reqId) -> None:  # noqa: N802
        self.exec_done = True

    def commissionReport(self, report) -> None:  # noqa: N802
        # Commission reports come AFTER execDetails. Match by execId.
        f = self.fills.get(report.execId)
        if f is not None:
            f.commission_ccy = report.currency
            f.commission_amt = float(report.commission)
            f.realized_pnl = float(report.realizedPNL or 0.0)
            f.yield_redemption_amt = float(report.yield_ or 0.0)

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:  # noqa: N802
        if errorCode not in (2104, 2106, 2158, 2119, 2107):
            print(f"  [IB {errorCode}] {errorString}", file=sys.stderr)


def _fetch_fills(days: int, timeout_sec: int = 20) -> list[Fill]:
    """Pull the last ``days`` of executions from IBKR."""
    print(f"  Connecting to {IBKR_HOST}:{IBKR_PORT} (client_id={CLIENT_ID})...")
    app = AuditApp()
    app.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()

    deadline = time.time() + timeout_sec
    while not app.ready and time.time() < deadline:
        time.sleep(0.1)
    if not app.ready:
        app.disconnect()
        raise RuntimeError("IBKR connection handshake failed")

    # Build filter for last `days` days
    flt = ExecutionFilter()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    flt.time = since.strftime("%Y%m%d-%H:%M:%S")  # IBKR format
    print(f"  Fetching executions since {flt.time}...")
    app.reqExecutions(reqId=1, execFilter=flt)

    deadline = time.time() + timeout_sec
    while not app.exec_done and time.time() < deadline:
        time.sleep(0.2)

    # Wait a beat for commission reports to arrive
    time.sleep(2.0)
    app.disconnect()

    return list(app.fills.values())


def _compute_drift(
    fills: list[Fill],
    *,
    fx_to_account_base: dict[str, float] | None = None,
    notional_unit: float = 1.0,
    min_commission_usd: float = 4.0,
) -> list[dict]:
    """Compute modelled vs realised commission per fill.

    For commission-only audit we exclude slippage (would need
    decision-time mid quotes). The dominant drift in the May 2026
    audit was the $4 minimum-commission floor on small notionals,
    which this catches directly.

    Returns list of dicts with per-fill comparison.
    """
    fx_to_account_base = fx_to_account_base or {}
    rows: list[dict] = []
    for f in fills:
        notional = f.qty * f.avg_price * notional_unit
        if notional <= 0:
            continue
        spread_bps = SPREAD_BPS_PER_SIDE.get(f.symbol, 5.0)
        modelled_bps = realistic_cost_bps(
            notional_usd=notional,
            spread_bps_per_side=spread_bps,
            min_commission_usd=min_commission_usd,
        )
        modelled_cost = notional * modelled_bps / 10_000.0
        # Convert realised commission to account base if FX rate known
        realised_cost = abs(f.commission_amt)
        if f.commission_ccy and f.commission_ccy in fx_to_account_base:
            realised_cost *= fx_to_account_base[f.commission_ccy]
        # Drift: realised vs modelled, percentage of modelled
        drift_pct = (realised_cost - modelled_cost) / modelled_cost if modelled_cost > 0 else 0.0
        rows.append(
            {
                "exec_id": f.exec_id,
                "time_utc": f.time_utc.isoformat(),
                "symbol": f.symbol,
                "side": f.side,
                "qty": f.qty,
                "price": f.avg_price,
                "notional": round(notional, 2),
                "modelled_bps": round(modelled_bps, 2),
                "modelled_cost": round(modelled_cost, 4),
                "realised_commission_ccy": f.commission_ccy,
                "realised_commission_amt": round(f.commission_amt, 4),
                "realised_cost_base": round(realised_cost, 4),
                "drift_pct": round(drift_pct, 4),
            }
        )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    """Roll up to (symbol, side). Average drift weighted by notional."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["symbol"], r["side"])].append(r)

    summary: list[dict] = []
    for (sym, side), group in sorted(groups.items()):
        total_notional = sum(r["notional"] for r in group)
        total_modelled = sum(r["modelled_cost"] for r in group)
        total_realised = sum(r["realised_cost_base"] for r in group)
        weighted_drift = (
            (total_realised - total_modelled) / total_modelled if total_modelled > 0 else 0.0
        )
        summary.append(
            {
                "symbol": sym,
                "side": side,
                "n_fills": len(group),
                "total_notional": round(total_notional, 2),
                "total_modelled_cost": round(total_modelled, 4),
                "total_realised_cost": round(total_realised, 4),
                "weighted_drift_pct": round(weighted_drift, 4),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default 7)")
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.50,
        help="Trigger alert if max |weighted_drift_pct| > this (default 0.50 = 50%%)",
    )
    parser.add_argument("--min-commission-usd", type=float, default=4.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print but do not send notifications"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-fill detail rows")
    args = parser.parse_args()

    print("=" * 80)
    print(f"COST AUDIT — last {args.days} days")
    print("=" * 80)

    try:
        fills = _fetch_fills(args.days)
    except Exception as e:
        print(f"\nFAILED to fetch fills: {e}", file=sys.stderr)
        return 2

    print(f"\n  Fetched {len(fills)} fills")
    if not fills:
        print("  No fills in window. Nothing to audit.")
        return 0

    rows = _compute_drift(fills, min_commission_usd=args.min_commission_usd)
    summary = _aggregate(rows)

    print()
    print("PER-INSTRUMENT SUMMARY:")
    print(
        f"  {'symbol':<10} {'side':<5} {'n':>4} {'notional':>12} "
        f"{'modelled':>10} {'realised':>10} {'drift':>10}"
    )
    print("  " + "-" * 70)
    for s in summary:
        drift_pct = s["weighted_drift_pct"] * 100
        marker = " *" if abs(drift_pct) > args.alert_threshold * 100 else ""
        print(
            f"  {s['symbol']:<10} {s['side']:<5} {s['n_fills']:>4d} "
            f"{s['total_notional']:>12.2f} "
            f"{s['total_modelled_cost']:>10.4f} {s['total_realised_cost']:>10.4f} "
            f"{drift_pct:>+9.2f}%{marker}"
        )

    if args.verbose:
        print()
        print("PER-FILL DETAIL:")
        for r in rows:
            drift_pct = r["drift_pct"] * 100
            print(
                f"  {r['time_utc'][:19]} {r['symbol']:<8} {r['side']:<4} "
                f"{r['qty']:>6.0f}@{r['price']:>10.4f} "
                f"notional={r['notional']:>10.2f} "
                f"modelled_bps={r['modelled_bps']:>5.1f} drift={drift_pct:>+7.2f}%"
            )

    # Save audit trail
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_csv = REPORTS_DIR / f"audit_{today}.csv"
    import csv

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Saved per-fill trail: {out_csv}")

    out_summary = REPORTS_DIR / f"summary_{today}.csv"
    if summary:
        with out_summary.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"  Saved summary trail:   {out_summary}")

    # Decide whether to alert
    breaches = [s for s in summary if abs(s["weighted_drift_pct"]) > args.alert_threshold]
    if not breaches:
        print(
            f"\n  All instrument-side aggregates within {args.alert_threshold * 100:.0f}% "
            f"drift threshold. No alert."
        )
        return 0

    print(f"\n  ALERT: {len(breaches)} instrument-side aggregates breach threshold.")
    if args.dry_run:
        print("  (--dry-run: notification suppressed)")
        return 1

    try:
        from titan.utils.notification import notify_health

        lines = [
            f"Cost-model drift detected over last {args.days} days "
            f"(threshold {args.alert_threshold * 100:.0f}%):",
        ]
        for s in breaches:
            lines.append(
                f"  - {s['symbol']} {s['side']}: drift={s['weighted_drift_pct'] * 100:+.1f}%  "
                f"(realised {s['total_realised_cost']:.2f} vs modelled "
                f"{s['total_modelled_cost']:.2f}, {s['n_fills']} fills)"
            )
        notify_health(
            event="cost_model_drift",
            severity="warning",
            detail="\n".join(lines),
        )
        print("  Notification dispatched via notify_health.")
    except Exception as e:
        print(f"  notify_health failed: {e}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
