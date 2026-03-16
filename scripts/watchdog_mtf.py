"""watchdog_mtf.py
-----------------
Autonomous watchdog for the MTF live strategy.

Runs run_live_mtf.py in a subprocess. If it exits for any reason
(crash, IB disconnect, maintenance window), waits briefly and restarts.

Usage:
    uv run python scripts/watchdog_mtf.py

Behaviour:
  - Restarts immediately on clean exit (code 0) or crash (non-zero exit).
  - Waits RESTART_DELAY_SECS between restarts to avoid hammering a dead gateway.
  - After the IB nightly maintenance window (~23:45 ET), Gateway needs ~2-3 min
    to come back up — RESTART_DELAY_SECS=180 covers this safely.
  - Logs each restart with timestamp to .tmp/logs/watchdog.log.
  - Stop the watchdog with Ctrl+C (SIGINT) or SIGTERM — it will NOT restart after
    receiving a signal.
"""

import logging
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RESTART_DELAY_SECS = 180  # 3 min — enough for IB Gateway to finish maintenance restart
STRATEGY_SCRIPT = PROJECT_ROOT / "scripts" / "run_live_mtf.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_file = LOGS_DIR / "watchdog.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog")

# ---------------------------------------------------------------------------
# Shutdown flag — set on SIGINT/SIGTERM so we don't restart after user stops
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    log.info(f"Watchdog received signal {signum}. Shutting down after current process exits.")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cmd = [sys.executable, str(STRATEGY_SCRIPT)]

    attempt = 0
    while not _shutdown:
        attempt += 1
        start = datetime.now(timezone.utc)
        log.info(f"[Attempt {attempt}] Starting MTF strategy at {start.isoformat()}")

        # Stream output directly to stdout/stderr (tee handled externally if needed)
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            break

        exit_code = proc.returncode
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        log.info(f"[Attempt {attempt}] Process exited with code {exit_code} after {elapsed:.0f}s")

        if _shutdown:
            log.info("Shutdown requested — not restarting.")
            break

        if elapsed < 30:
            # Died almost immediately — likely a config error, not a disconnect.
            # Longer delay to avoid restart storm.
            delay = 60
            log.warning(f"Process died in <30s — possible config error. Waiting {delay}s.")
        else:
            delay = RESTART_DELAY_SECS

        log.info(f"Restarting in {delay}s...")
        # Interruptible sleep
        for _ in range(delay):
            if _shutdown:
                break
            time.sleep(1)

    log.info("Watchdog stopped.")


if __name__ == "__main__":
    main()
