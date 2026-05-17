"""cusum_daemon.py -- run scripts/monitor_live_drift.py daily.

Wraps the one-shot CUSUM drift monitor in a sleep-until-next-22:00-UTC
loop so it can be deployed as a long-running service (cron alternative).

Choose ONE of these deployment patterns:

1. Host cron (recommended -- simplest, no container resources):
       0 22 * * * cd /path/to/Titan-IBKR && PYTHONIOENCODING=utf-8 \\
           uv run python scripts/monitor_live_drift.py >> .tmp/live_drift/cron.log 2>&1

2. Compose service (if cron is unavailable on host -- e.g. Synology DSM,
   Windows-only deployment). Add to docker-compose.yml:
       titan-cusum-monitor:
         image: ${TITAN_IMAGE:-titan-portfolio:latest}
         container_name: titan-cusum-monitor
         restart: unless-stopped
         depends_on:
           - titan-portfolio
         command: ["python", "scripts/cusum_daemon.py"]
         volumes:
           - titan-tmp:/app/.tmp
           - /var/run/docker.sock:/var/run/docker.sock:ro  # required for `docker logs`
         networks:
           - titan-net

   Requires Docker socket access (security-sensitive). Prefer cron unless
   the deployment host lacks one.

3. Manual one-shot (validation only): run scripts/monitor_live_drift.py
   directly. The script is idempotent on date -- safe to invoke any time.

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/cusum_daemon.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITOR_FP = PROJECT_ROOT / "scripts" / "monitor_live_drift.py"
TARGET_HOUR_UTC = 22  # 22:00 UTC daily run (after US close, before next session)


def _seconds_until_next_run() -> float:
    """Seconds until next 22:00 UTC."""
    now = datetime.now(timezone.utc)
    target_today = datetime.combine(now.date(), dt_time(TARGET_HOUR_UTC, 0), timezone.utc)
    if now < target_today:
        return (target_today - now).total_seconds()
    target_tomorrow = target_today + timedelta(days=1)
    return (target_tomorrow - now).total_seconds()


def run_monitor() -> int:
    """Spawn the one-shot monitor and return its exit code."""
    print(f"[cusum-daemon] {datetime.now(timezone.utc).isoformat()} firing monitor_live_drift.py", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, str(MONITOR_FP)],
            cwd=str(PROJECT_ROOT),
            timeout=300,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("[cusum-daemon] monitor TIMED OUT after 300s", file=sys.stderr, flush=True)
        return 124


def main() -> None:
    print(f"[cusum-daemon] start. Target: {TARGET_HOUR_UTC:02d}:00 UTC daily.", flush=True)
    # Run once at start (helpful on container restart) then loop on schedule.
    run_monitor()
    while True:
        wait_s = _seconds_until_next_run()
        wakeup_iso = (datetime.now(timezone.utc) + timedelta(seconds=wait_s)).isoformat()
        print(f"[cusum-daemon] sleeping {wait_s/3600:.2f}h until {wakeup_iso}", flush=True)
        time.sleep(wait_s)
        run_monitor()


if __name__ == "__main__":
    main()
