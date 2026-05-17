"""monitor_turtle_first_trade.py — host-side first-trade alert for turtle CAT.

Watches the titan-portfolio container logs for the first turtle entry/exit
event after V3.7 Phase 2 cutover (2026-05-17 23:49 UTC). Sends a one-shot
notify_health alert when detected so the operator gets a heads-up the
moment turtle moves from flat → long on CAT.

Why host-side: the live container is running production paper trading;
we don't touch its code during the validation period. Watching Docker logs
externally is non-intrusive — the alert can be killed without affecting
the live deployment.

Behaviour:
  - Tails `docker logs -f titan-portfolio` since the V3.7 cutover.
  - Greps for ENTRY BUY / EXIT / FILLED / REJECTED events on
    TITAN-PORTFOLIO.TurtleStrategy specifically.
  - First event detected → notify_health (info severity) + writes a flag
    file (`.tmp/turtle_first_trade.json`).
  - After first detection, continues to log subsequent events to a JSON
    line log so the operator can review the trade history.

Run:
  PYTHONIOENCODING=utf-8 uv run python scripts/monitor_turtle_first_trade.py
  (or systemd / Docker compose `restart: unless-stopped` to keep it alive)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FLAG_FILE = PROJECT_ROOT / ".tmp" / "turtle_first_trade.json"
TRADE_LOG = LOGS_DIR / "turtle_live_trades.jsonl"

# V3.7 cutover timestamp -- start tailing from here.
CUTOVER_TS = "2026-05-17T23:49:00Z"

# Patterns to match on turtle strategy events.
ENTRY_PATTERN = re.compile(
    r"TITAN-PORTFOLIO\.TurtleStrategy.*ENTRY (BUY|SELL).*qty=([\d.]+).*price=([\d.]+)",
    re.IGNORECASE,
)
EXIT_PATTERN = re.compile(
    r"TITAN-PORTFOLIO\.TurtleStrategy.*(EXIT|FLATTEN|CLOSED).*",
    re.IGNORECASE,
)
ORDER_FILLED_PATTERN = re.compile(
    r"TITAN-PORTFOLIO\.TurtleStrategy.*(?:OrderFilled|FILLED|REJECTED|REJECTED)",
    re.IGNORECASE,
)


def _notify(msg: str, severity: str = "info", detail: str = "") -> None:
    """Call notify_health (best-effort)."""
    try:
        from titan.utils.notification import notify_health
        notify_health(msg, severity=severity, detail=detail)
    except Exception as e:  # noqa: BLE001
        print(f"[notify_health unavailable: {e}] {msg}", file=sys.stderr)


def _record_event(event_type: str, raw_line: str) -> None:
    """Append a single event to the JSONL trade log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "raw_line": raw_line.strip(),
    }
    with TRADE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main() -> int:
    print(f"[monitor] watching titan-portfolio for first turtle trade since {CUTOVER_TS}")
    print(f"[monitor] flag file: {FLAG_FILE}")
    print(f"[monitor] trade log: {TRADE_LOG}")

    if FLAG_FILE.exists():
        print("[monitor] flag file already exists -- first trade already detected")
        print(f"[monitor] continuing to log subsequent events to {TRADE_LOG}")
        first_seen = True
    else:
        first_seen = False

    # Tail docker logs from the V3.7 cutover.
    cmd = [
        "docker", "logs", "-f",
        "--since", CUTOVER_TS,
        "titan-portfolio",
    ]
    print(f"[monitor] command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )

    if proc.stdout is None:
        print("[monitor] failed to open subprocess stdout", file=sys.stderr)
        return 1

    try:
        for line in proc.stdout:
            event_type = None
            if ENTRY_PATTERN.search(line):
                event_type = "ENTRY"
            elif EXIT_PATTERN.search(line):
                event_type = "EXIT"
            elif ORDER_FILLED_PATTERN.search(line):
                event_type = "FILL"

            if event_type is None:
                continue

            _record_event(event_type, line)
            print(f"[monitor] {event_type}: {line.rstrip()}")

            if not first_seen:
                first_seen = True
                # Write the flag file with metadata.
                FLAG_FILE.write_text(
                    json.dumps({
                        "first_seen_ts": datetime.now(timezone.utc).isoformat(),
                        "event_type": event_type,
                        "raw_line": line.strip(),
                    }, indent=2),
                    encoding="utf-8",
                )
                _notify(
                    f"turtle CAT first {event_type} event detected",
                    severity="info",
                    detail=line.strip(),
                )
                print("[monitor] FIRST TRADE DETECTED -- flag file written + notify_health sent")
    except KeyboardInterrupt:
        print("[monitor] interrupted -- exiting")
        proc.terminate()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
