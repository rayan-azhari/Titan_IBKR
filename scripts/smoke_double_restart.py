"""smoke_double_restart.py -- pre-deploy verification: restart twice, verify positions unchanged.

Tier 1.2 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).

The May 11 rehydration bug doubled positions on every container
restart. The first restart looked fine (no entry signal happened to
fire); the second one was when the bug manifested. A pre-deploy
"restart twice and verify positions unchanged" test catches this class
at L4 (deploy time) instead of L5/L6 (monitoring/post-mortem).

Workflow::

    1. Snapshot broker positions (direct ibapi, fresh client_id)
    2. docker compose restart titan-portfolio
    3. Wait for "RUNNING" signal in logs (or timeout)
    4. Snapshot positions again
    5. docker compose restart titan-portfolio
    6. Wait for "RUNNING" signal in logs (or timeout)
    7. Snapshot positions again
    8. Diff all three snapshots — fail if any per-instrument
       position count or net quantity changed

Usage::

    uv run python scripts/smoke_double_restart.py
    uv run python scripts/smoke_double_restart.py --service titan-portfolio --quiet-window-sec 60

Exit codes::

    0  All three snapshots match — safe to deploy.
    1  Mismatch detected — investigate before deploying.
    2  Operational error (gateway unreachable, container failed to
       start, etc.) — re-run after resolving.

Caveat: must be run during a "quiet" period (no bars firing during the
restart cycles). Bond_equity strategies trade on daily LSE close
(~16:30 UTC); mr_audjpy on AUD/JPY H1. Best window is weekend or
between 17:00 UTC and 23:00 UTC weekdays.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IBKR_HOST = os.getenv("IBKR_HOST_HOST", "127.0.0.1")  # for host-side connection
IBKR_PORT = int(os.getenv("IBKR_PORT_HOST", "4004"))
SNAPSHOT_CLIENT_ID = 96  # distinct from strategy=7, kill=98, cspx_close=99, orphans=97


class SnapshotApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.positions: list[tuple[str, Contract, float, float]] = []
        self.positions_done = False

    def nextValidId(self, orderId: int) -> None:
        self.ready = True

    def position(self, account: str, contract: Contract, pos: float, avgCost: float) -> None:
        if pos != 0:
            self.positions.append((account, contract, float(pos), float(avgCost)))

    def positionEnd(self) -> None:
        self.positions_done = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
        if errorCode not in (2104, 2106, 2158, 2119, 2107):
            print(f"  [IB {errorCode}] {errorString}", file=sys.stderr)


def _snapshot_positions(timeout_sec: int = 15) -> dict[str, dict]:
    """Connect to IBKR, snapshot all open positions, return as
    {symbol: {qty, avg_cost, currency}}."""
    print(f"  Connecting to {IBKR_HOST}:{IBKR_PORT} (client_id={SNAPSHOT_CLIENT_ID})...")
    app = SnapshotApp()
    app.connect(IBKR_HOST, IBKR_PORT, SNAPSHOT_CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()

    deadline = time.time() + timeout_sec
    while not app.ready and time.time() < deadline:
        time.sleep(0.1)
    if not app.ready:
        app.disconnect()
        raise RuntimeError("IBKR connection handshake failed")

    app.reqPositions()
    deadline = time.time() + timeout_sec
    while not app.positions_done and time.time() < deadline:
        time.sleep(0.1)
    app.cancelPositions()
    time.sleep(0.5)  # let pending events drain
    app.disconnect()

    snapshot: dict[str, dict] = {}
    # Collapse multi-position-per-instrument into per-symbol aggregates.
    # We track total qty AND position count so doubling is detected even
    # if total qty matches.
    qty_by_symbol: dict[str, float] = defaultdict(float)
    count_by_symbol: dict[str, int] = defaultdict(int)
    for _acct, c, pos, _avg in app.positions:
        sym = c.symbol
        qty_by_symbol[sym] += pos
        count_by_symbol[sym] += 1
    for sym, qty in qty_by_symbol.items():
        snapshot[sym] = {
            "qty": round(qty, 4),
            "position_count": count_by_symbol[sym],
        }
    return snapshot


def _docker_restart(service: str) -> None:
    print(f"  docker compose restart {service}...")
    res = subprocess.run(
        ["docker", "compose", "restart", service],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"docker compose restart failed: {res.stderr}")


def _wait_for_healthy(service: str, timeout_sec: int = 90) -> None:
    """Poll ``docker logs`` for the strategy startup banner."""
    print(f"  Waiting up to {timeout_sec}s for {service} to be healthy...")
    deadline = time.time() + timeout_sec
    last_size = 0
    while time.time() < deadline:
        res = subprocess.run(
            ["docker", "logs", "--tail", "200", service],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            time.sleep(2)
            continue
        log = res.stdout + res.stderr
        # Healthy signal: at least one strategy reports started + RUNNING
        if "RUNNING" in log and "started" in log.lower():
            print("    ✓ Healthy")
            return
        time.sleep(3)
        if len(log) != last_size:
            last_size = len(log)
    raise RuntimeError(f"{service} did not reach healthy state within {timeout_sec}s")


def _diff_snapshots(snapshots: list[dict[str, dict]]) -> list[str]:
    """Return human-readable findings for any mismatch across snapshots."""
    findings: list[str] = []
    all_symbols = set()
    for snap in snapshots:
        all_symbols.update(snap.keys())
    for sym in sorted(all_symbols):
        records = [snap.get(sym, {"qty": 0.0, "position_count": 0}) for snap in snapshots]
        qty_values = [r["qty"] for r in records]
        count_values = [r["position_count"] for r in records]
        if len(set(qty_values)) > 1:
            findings.append(f"  {sym}: qty changed across snapshots: {qty_values}")
        if len(set(count_values)) > 1:
            findings.append(
                f"  {sym}: position_count changed across snapshots: {count_values} "
                f"(SUSPICIOUS — may be the May 11 doubling bug class)"
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--service",
        default="titan-portfolio",
        help="docker compose service name (default: titan-portfolio)",
    )
    parser.add_argument(
        "--healthy-timeout",
        type=int,
        default=90,
        help="seconds to wait for service healthy after restart (default 90)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DOUBLE-RESTART SMOKE TEST")
    print("=" * 70)
    print(
        "Snapshots broker positions, restarts the container twice, and "
        "verifies position count + qty per instrument is unchanged across\n"
        "all three snapshots. Catches the May 11 phantom-doubling bug class\n"
        "at L4 (pre-deploy) instead of in production."
    )
    print()

    snapshots: list[dict[str, dict]] = []
    try:
        print("[1/5] Initial snapshot...")
        snapshots.append(_snapshot_positions())
        print(f"      → {len(snapshots[-1])} symbols, {snapshots[-1]}")

        print("[2/5] First restart...")
        _docker_restart(args.service)
        _wait_for_healthy(args.service, timeout_sec=args.healthy_timeout)

        print("[3/5] Snapshot after restart 1...")
        snapshots.append(_snapshot_positions())
        print(f"      → {len(snapshots[-1])} symbols, {snapshots[-1]}")

        print("[4/5] Second restart...")
        _docker_restart(args.service)
        _wait_for_healthy(args.service, timeout_sec=args.healthy_timeout)

        print("[5/5] Snapshot after restart 2...")
        snapshots.append(_snapshot_positions())
        print(f"      → {len(snapshots[-1])} symbols, {snapshots[-1]}")

    except Exception as e:
        print(f"\n✗ Operational error: {e}", file=sys.stderr)
        return 2

    print()
    print("=" * 70)
    print("DIFF ANALYSIS")
    print("=" * 70)
    findings = _diff_snapshots(snapshots)
    if not findings:
        print("✓ All three snapshots match. Safe to deploy.")
        # Save snapshots for audit trail
        out = PROJECT_ROOT / ".tmp" / "smoke_double_restart.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"snapshots": snapshots, "ts": time.time()}, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved audit trail: {out}")
        return 0

    print("✗ Position state changed across restart cycles:")
    for f in findings:
        print(f)
    print()
    print("  Investigate before deploying. Common causes:")
    print("    - A bar fired between snapshots (run during quiet window)")
    print("    - A strategy doubled positions on rehydrate (May 11 bug class)")
    print("    - A manual trade happened concurrently")
    return 1


if __name__ == "__main__":
    sys.exit(main())
