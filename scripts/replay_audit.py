"""replay_audit.py -- backtest-vs-live reconciliation for bond_equity strategies.

Tier 2.5 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).

For each trading day in the audit window:
  1. Compute the strategy's expected action by replaying its decision
     logic against the signal-instrument parquet (IHYG / IHYU history).
  2. Pull the actual IBKR fills for the trade-instrument (VUSD / EIMI /
     CSPX) on that day.
  3. Diff expected action (entry / exit / hold) against actual fills.
  4. Alert on any mismatch via ``notify_health(severity="warning")``.

This catches:
  - Strategy entered when backtest wouldn't have (operational bug,
    rehydration double-up, wrong signal calculation, race condition).
  - Strategy didn't enter when backtest would have (signal miss,
    execution failure, broker rejection).
  - Strategy entered/exited at the wrong time (timing drift, stale
    parquet, time-zone bug).

The replay uses a pure-Python copy of the bond_gold decision logic
(``BondGoldDecision``) that mirrors ``BondGoldStrategy._run_signal``.
Kept in lockstep with production by structural review — same approach
as ``tests/test_bond_gold_state_machine_properties.py`` (Tier 2.2) and
``tests/test_reconciliation_strategy.py`` (Tier 1.1).

State-anchored replay:
  The audit replays each day's decision using the *actual* prior-EOD
  position from broker fills, not a pure-backtest simulation. This
  isolates "given the same starting state, did live make the right
  call?" from "are we accumulating drift over time?" — the latter is
  T2.1 (shadow-strategy) territory.

Usage::

    uv run python scripts/replay_audit.py
    uv run python scripts/replay_audit.py --strategy bond_equity_ihyg_vusd --days 30
    uv run python scripts/replay_audit.py --days 14 --dry-run --verbose

Cron suggestion (Sunday 03:00 UTC, after cost_audit at 02:00):

    0 3 * * 0  cd /app && uv run python scripts/replay_audit.py --days 7

Exit codes:
    0  No mismatch (or no fills in window)
    1  Mismatch detected (notification dispatched if not --dry-run)
    2  Operational error (gateway / parquet load failure)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.execution import ExecutionFilter
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Decision primitives are shared with the reconciliation watchdog (T2.1)
# so the two operational checks can never silently disagree on what the
# strategy "should" have done.
from titan.utils.bond_gold_decisions import (  # noqa: E402
    LIVE_CONFIGS,
    BondGoldDecisionConfig,
    compute_z_score,
    expected_action,
)

# Re-exported names for backward compat with tests that import from this
# module directly. New callers should prefer ``titan.utils.bond_gold_decisions``.
StrategyConfig = BondGoldDecisionConfig
CONFIGS = LIVE_CONFIGS

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "replay_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IBKR_HOST = os.getenv("IBKR_HOST_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT_HOST", "4004"))
CLIENT_ID = 94  # distinct from strategy=7, kill=98, cspx=99, orphans=97, snap=96, audit=95


# ── IBKR fill fetch ───────────────────────────────────────────────────


@dataclass
class Fill:
    exec_id: str
    time_utc: datetime
    symbol: str
    side: str
    qty: float
    price: float


class FillsApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.fills: list[Fill] = []
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
        self.fills.append(
            Fill(
                exec_id=execution.execId,
                time_utc=ts,
                symbol=contract.symbol,
                side=execution.side,
                qty=float(execution.shares),
                price=float(execution.avgPrice),
            )
        )

    def execDetailsEnd(self, reqId) -> None:  # noqa: N802
        self.exec_done = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:  # noqa: N802
        if errorCode not in (2104, 2106, 2158, 2119, 2107):
            print(f"  [IB {errorCode}] {errorString}", file=sys.stderr)


def fetch_fills_for_symbol(symbol: str, days: int, timeout_sec: int = 20) -> list[Fill]:
    """Pull last ``days`` of executions for a single symbol."""
    print(f"  Connecting to {IBKR_HOST}:{IBKR_PORT} (client_id={CLIENT_ID})...")
    app = FillsApp()
    app.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()

    deadline = time.time() + timeout_sec
    while not app.ready and time.time() < deadline:
        time.sleep(0.1)
    if not app.ready:
        app.disconnect()
        raise RuntimeError("IBKR connection handshake failed")

    flt = ExecutionFilter()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    flt.time = since.strftime("%Y%m%d-%H:%M:%S")
    flt.symbol = symbol
    print(f"  Fetching {symbol} executions since {flt.time}...")
    app.reqExecutions(reqId=1, execFilter=flt)

    deadline = time.time() + timeout_sec
    while not app.exec_done and time.time() < deadline:
        time.sleep(0.2)
    time.sleep(1.0)
    app.disconnect()
    return app.fills


# ── Replay engine ─────────────────────────────────────────────────────


@dataclass
class ReplayState:
    """Position state tracked through the replay window."""

    is_long: bool = False
    bars_held: int = 0


def load_signal_closes(ticker: str) -> pd.DataFrame:
    """Load the signal-instrument parquet — same data the live strategy uses."""
    path = DATA_DIR / f"{ticker}_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Signal parquet missing: {path}")
    df = pd.read_parquet(path).sort_index()
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    elif df.index.tz is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    return df


def fills_to_daily_actions(fills: list[Fill]) -> dict[str, str]:
    """Convert IBKR fills to a {date_str: action} mapping.

    Multiple fills on the same day collapse to a single net-direction
    action (BUY > 0 net = "entry", SELL > 0 net = "exit"). Mixed
    directions in one day produce "mixed" (rare and worth flagging).
    """
    by_date: dict[str, dict[str, float]] = {}
    for f in fills:
        date_str = f.time_utc.date().isoformat()
        d = by_date.setdefault(date_str, {"BUY": 0.0, "SELL": 0.0})
        d[f.side] += f.qty
    out: dict[str, str] = {}
    for date_str, sides in by_date.items():
        if sides["BUY"] > 0 and sides["SELL"] == 0:
            out[date_str] = "entry"
        elif sides["SELL"] > 0 and sides["BUY"] == 0:
            out[date_str] = "exit"
        else:
            out[date_str] = "mixed"
    return out


def replay(cfg: StrategyConfig, days: int) -> dict:
    """Run replay for one strategy. Returns {expected, actual, mismatches}."""
    # Load signal data
    signal_df = load_signal_closes(cfg.signal_ticker)
    closes_full = signal_df["close"].astype(float).tolist()
    dates_full = signal_df.index.date

    # Audit window: last `days` of available signal data
    if len(closes_full) <= cfg.zscore_window + cfg.lookback:
        raise RuntimeError(f"Insufficient {cfg.signal_ticker} history")

    # Pull fills for the trade instrument
    fills = fetch_fills_for_symbol(cfg.trade_symbol, days=days)
    actual_actions = fills_to_daily_actions(fills)
    print(
        f"  Fetched {len(fills)} {cfg.trade_symbol} fills, "
        f"{len(actual_actions)} unique trading days"
    )

    # State-anchored replay over the audit window
    state = ReplayState()
    expected_per_day: dict[str, str] = {}
    z_per_day: dict[str, float] = {}
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)

    for i, dt_date in enumerate(dates_full):
        if dt_date < cutoff:
            continue
        # Slice closes up to and including this bar
        closes_so_far = closes_full[: i + 1]
        z = compute_z_score(
            closes_so_far,
            lookback=cfg.lookback,
            zscore_window=cfg.zscore_window,
        )
        if z is None:
            continue
        date_str = dt_date.isoformat()
        z_per_day[date_str] = z

        # Determine expected action with current state
        action = expected_action(
            z=z,
            is_long=state.is_long,
            bars_held=state.bars_held,
            threshold=cfg.threshold,
            hold_days=cfg.hold_days,
        )
        expected_per_day[date_str] = action

        # Anchor state to actual fills (state-anchored replay)
        live_action = actual_actions.get(date_str, "hold")
        if live_action == "entry":
            state.is_long = True
            state.bars_held = 0
        elif live_action == "exit":
            state.is_long = False
            state.bars_held = 0
        elif state.is_long:
            state.bars_held += 1

    # Diff
    all_dates = sorted(set(expected_per_day) | set(actual_actions))
    rows: list[dict] = []
    mismatches: list[dict] = []
    for d in all_dates:
        exp = expected_per_day.get(d, "<no-bar>")
        act = actual_actions.get(d, "hold")
        z = z_per_day.get(d)
        # Only consider it a mismatch if EXPECTED was an action that didn't happen,
        # or actual was an action backtest wouldn't have triggered.
        # "hold" vs "hold" is fine. "entry vs hold" is a miss. "hold vs entry" is
        # an unexpected trade.
        match = exp == act
        rows.append(
            {
                "date": d,
                "z_score": round(z, 3) if z is not None else None,
                "expected": exp,
                "actual": act,
                "match": match,
            }
        )
        if not match and (exp != "hold" or act != "hold"):
            mismatches.append({"date": d, "expected": exp, "actual": act, "z_score": z})

    return {
        "strategy": cfg.name,
        "trade_symbol": cfg.trade_symbol,
        "signal_ticker": cfg.signal_ticker,
        "rows": rows,
        "mismatches": mismatches,
    }


# ── Reporting ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", *CONFIGS.keys()],
        help="Strategy to audit (default: all bond_equity_*)",
    )
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days (default 14)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(f"REPLAY AUDIT — last {args.days} days")
    print("=" * 80)

    selected = list(CONFIGS.keys()) if args.strategy == "all" else [args.strategy]
    all_results: list[dict] = []
    overall_rc = 0
    for name in selected:
        cfg = CONFIGS[name]
        print(f"\n--- {cfg.name} ({cfg.trade_symbol} / signal={cfg.signal_ticker}) ---")
        try:
            result = replay(cfg, args.days)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            overall_rc = 2
            continue

        n_match = sum(1 for r in result["rows"] if r["match"])
        print(
            f"  {len(result['rows'])} bars analysed, "
            f"{n_match} match, {len(result['mismatches'])} actionable mismatch(es)"
        )

        if args.verbose:
            print(f"  {'date':<12}  {'z':>7}  {'expected':<10}  {'actual':<10}  match")
            print("  " + "-" * 60)
            for r in result["rows"]:
                z_str = f"{r['z_score']:>+7.3f}" if r["z_score"] is not None else "    n/a"
                marker = "" if r["match"] else "  *"
                print(
                    f"  {r['date']:<12}  {z_str}  {r['expected']:<10}  "
                    f"{r['actual']:<10}  {r['match']}{marker}"
                )

        if result["mismatches"]:
            print(f"  ACTIONABLE MISMATCHES ({len(result['mismatches'])}):")
            for m in result["mismatches"]:
                z_str = f"{m['z_score']:>+.3f}" if m["z_score"] is not None else "n/a"
                print(
                    f"    {m['date']}  z={z_str}  expected={m['expected']:<6}  actual={m['actual']}"
                )
            overall_rc = max(overall_rc, 1)

        all_results.append(result)

    # Save audit trail
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    import csv

    out_csv = REPORTS_DIR / f"replay_{today}.csv"
    flat_rows: list[dict] = []
    for res in all_results:
        for r in res["rows"]:
            flat_rows.append({"strategy": res["strategy"], **r})
    if flat_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            w.writerows(flat_rows)
        print(f"\nSaved replay trail: {out_csv}")

    # Notify if any mismatches across all strategies
    total_mismatches = sum(len(r["mismatches"]) for r in all_results)
    if total_mismatches == 0 or overall_rc == 0:
        print("\nNo actionable mismatches. Live decisions match backtest expectations.")
        return overall_rc

    if args.dry_run:
        print(f"\n  ({total_mismatches} mismatch(es) — --dry-run, notification suppressed)")
        return overall_rc

    try:
        from titan.utils.notification import notify_health

        lines = [f"Backtest-vs-live replay audit detected {total_mismatches} mismatch(es):"]
        for res in all_results:
            for m in res["mismatches"]:
                z_str = f"{m['z_score']:+.3f}" if m["z_score"] is not None else "n/a"
                lines.append(
                    f"  - {res['strategy']} ({res['trade_symbol']}) {m['date']}: "
                    f"z={z_str} expected={m['expected']} actual={m['actual']}"
                )
        notify_health(
            event="replay_audit_mismatch",
            severity="warning",
            detail="\n".join(lines),
        )
        print("\n  Notification dispatched via notify_health.")
    except Exception as e:
        print(f"  notify_health failed: {e}", file=sys.stderr)

    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
