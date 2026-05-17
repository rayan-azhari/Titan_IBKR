"""daily_portfolio_summary.py — host-side daily snapshot of live portfolio state.

Extracts the most recent AccountState + per-strategy equity events from the
titan-portfolio container logs and writes a structured summary to
`.tmp/portfolio_summary.json` for ops review.

Why host-side (not inside the container): during the V3.7 paper validation
window we avoid touching the live runner. The container's logs already
contain the data we need; an external parser is non-intrusive and easy to
revoke.

Output: `.tmp/portfolio_summary.json` (overwritten each run)
  {
    "ts": "...",
    "container_status": "...",
    "container_uptime": "...",
    "account": {
      "balance_total": ..., "locked": ..., "free": ...,
      "currency": "GBP", "account_id": "..."
    },
    "strategies": {
      "GemStrategy": {"last_seen": "...", "running": true, ...},
      "TurtleStrategy": {"last_seen": "...", "running": true, ...}
    },
    "errors_last_24h": [...]
  }

Run:
  PYTHONIOENCODING=utf-8 uv run python scripts/daily_portfolio_summary.py

Cron/systemd: schedule daily at 21:00 Europe/London (after US close + EU close).
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

OUTPUT_FILE = PROJECT_ROOT / ".tmp" / "portfolio_summary.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Patterns to extract from logs.
ACCOUNT_STATE_RE = re.compile(
    r"Updated AccountState\(account_id=(?P<account_id>[^,]+),.*"
    r"balances=\[AccountBalance\(total=(?P<total>[\d_\.]+) (?P<ccy>\w+),"
    r" locked=(?P<locked>[\d_\.]+) \w+, free=(?P<free>[\d_\.]+) \w+\)\]"
)
STRATEGY_RUNNING_RE = re.compile(
    r"(?P<strategy>TITAN-PORTFOLIO\.\w+Strategy): RUNNING"
)
ERROR_RE = re.compile(r"\[(ERROR|FATAL)\].*?(?P<msg>[^\[\]]+?)(?:\[|$)")


def _get_container_status() -> dict:
    """Get container uptime + state."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "titan-portfolio",
             "--format", "{{.State.Status}} | {{.State.StartedAt}} | {{.RestartCount}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"status": "missing", "raw": result.stderr.strip()}
        parts = result.stdout.strip().split(" | ")
        return {
            "status": parts[0],
            "started_at": parts[1],
            "restart_count": int(parts[2]),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def _get_recent_logs(lines: int = 500) -> list[str]:
    """Get recent container log lines."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), "titan-portfolio"],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.splitlines() + result.stderr.splitlines()
    except Exception as e:  # noqa: BLE001
        print(f"docker logs failed: {e}", file=sys.stderr)
        return []


def _parse_account_state(logs: list[str]) -> dict:
    """Find the most recent AccountState event."""
    for line in reversed(logs):
        m = ACCOUNT_STATE_RE.search(line)
        if m:
            return {
                "account_id": m.group("account_id"),
                "balance_total": float(m.group("total").replace("_", "")),
                "locked": float(m.group("locked").replace("_", "")),
                "free": float(m.group("free").replace("_", "")),
                "currency": m.group("ccy"),
                "log_line": line[:200],
            }
    return {}


def _parse_strategy_states(logs: list[str]) -> dict:
    """Find all RUNNING strategies."""
    found: dict[str, str] = {}
    for line in logs:
        m = STRATEGY_RUNNING_RE.search(line)
        if m:
            strat = m.group("strategy").split(".")[-1]
            found[strat] = line[:200]
    return {name: {"last_running_line": ln} for name, ln in found.items()}


def _parse_errors(logs: list[str], max_count: int = 10) -> list[str]:
    """Collect recent ERROR / FATAL lines."""
    errors = []
    for line in reversed(logs):
        if "[ERROR]" in line or "[FATAL]" in line:
            errors.append(line[:300])
            if len(errors) >= max_count:
                break
    return list(reversed(errors))


def main() -> int:
    container = _get_container_status()
    logs = _get_recent_logs(lines=1000)

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "container": container,
        "account": _parse_account_state(logs),
        "strategies": _parse_strategy_states(logs),
        "errors_recent": _parse_errors(logs),
        "log_lines_scanned": len(logs),
    }

    OUTPUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] wrote {OUTPUT_FILE}")
    print(f"  container: {container.get('status')}")
    if summary["account"]:
        acc = summary["account"]
        print(f"  account: {acc['account_id']} | "
              f"total={acc['balance_total']:.2f} {acc['currency']} | "
              f"locked={acc['locked']:.2f} | free={acc['free']:.2f}")
    print(f"  strategies running: {list(summary['strategies'].keys())}")
    print(f"  errors_recent: {len(summary['errors_recent'])}")

    return 0 if container.get("status") == "running" else 1


if __name__ == "__main__":
    sys.exit(main())
