"""watchdog_portfolio.py
-----------------------
Autonomous watchdog for scripts/run_portfolio.py.

Runs the portfolio runner in a subprocess. If it exits for any reason
(crash, IB Gateway disconnect, weekly maintenance), waits briefly and
restarts. Designed for unattended Docker / systemd operation.

Usage:
    uv run python scripts/watchdog_portfolio.py --strategies v37_live
    uv run python scripts/watchdog_portfolio.py --strategies champion_portfolio

Behaviour:
  - Restarts on any child exit (clean or crash) until SIGTERM/SIGINT received.
  - Waits RESTART_DELAY_SECS (180s default) between restarts -- enough for
    the IBKR weekly auto-restart cycle (Sunday 03:00 ET if AUTO_RESTART_TIME
    is set on the gateway) to come back up.
  - Forwards SIGTERM to the child so `docker compose down` propagates cleanly
    -- the runner gets a chance to flatten / persist halt state before exit.
  - Logs each restart with a timestamp to stdout (Docker captures it) AND
    to .tmp/logs/watchdog_portfolio.log (survives via the named volume).

# TODO: this file is a near-duplicate of scripts/watchdog_mtf.py. When a
# third deployment lands, extract the loop into titan/ops/watchdog.py and
# leave thin CLI wrappers here. With N=2 the duplication is cheaper than
# the abstraction.
"""

from __future__ import annotations

import argparse
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

# Make `titan.utils.notification` importable when running this script directly
# (Docker entrypoint case: PYTHONPATH not yet set, no editable install).
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from titan.utils.notification import notify_health
except Exception:  # noqa: BLE001

    def notify_health(*_args, **_kwargs) -> int:  # type: ignore[no-redef]
        return 0


RESTART_DELAY_SECS = 180  # 3 min — covers IBKR maintenance / weekly restart
QUICK_DEATH_SECS = 30  # if process dies in <30s, treat as config error
QUICK_DEATH_DELAY_SECS = 60
STRATEGY_SCRIPT = PROJECT_ROOT / "scripts" / "run_portfolio.py"


# ---------------------------------------------------------------------------
# Logging — both stdout (for Docker) and a persistent file (for post-mortem)
# ---------------------------------------------------------------------------
log_file = LOGS_DIR / "watchdog_portfolio.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog_portfolio")


# ---------------------------------------------------------------------------
# Shutdown coordination
# ---------------------------------------------------------------------------
_shutdown = False
_current_proc: subprocess.Popen | None = None


def _handle_signal(signum: int, frame) -> None:  # noqa: ARG001
    global _shutdown
    _shutdown = True
    log.info(f"Watchdog received signal {signum}. Forwarding to child and stopping.")
    notify_health(
        f"Watchdog received signal {signum}; shutting down portfolio runner",
        severity="info",
    )
    if _current_proc is not None and _current_proc.poll() is None:
        try:
            _current_proc.terminate()
        except ProcessLookupError:
            pass


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watchdog for scripts/run_portfolio.py",
    )
    parser.add_argument(
        "--strategies",
        default="v37_live",
        help=(
            "Strategy bundle name passed to run_portfolio.py. "
            "V3.7 default: 'v37_live' (GEM J5 + turtle CAT). "
            "Other sets: 'champion_portfolio', 'samir_validation', "
            "'daily_only'. See STRATEGY_SETS in run_portfolio.py."
        ),
    )
    args = parser.parse_args()

    cmd = [sys.executable, str(STRATEGY_SCRIPT), "--strategies", args.strategies]
    log.info(f"Watchdog starting. Child command: {' '.join(cmd)}")
    log.info(
        f"Restart cooldown: {RESTART_DELAY_SECS}s normal, "
        f"{QUICK_DEATH_DELAY_SECS}s if child dies in <{QUICK_DEATH_SECS}s"
    )
    notify_health(
        f"Watchdog started — strategies={args.strategies}",
        severity="ok",
    )

    global _current_proc
    attempt = 0
    while not _shutdown:
        attempt += 1
        start = datetime.now(timezone.utc)
        log.info(f"[Attempt {attempt}] Starting portfolio runner at {start.isoformat()}")

        _current_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

        try:
            _current_proc.wait()
        except KeyboardInterrupt:
            # Belt-and-braces — _handle_signal already terminated the child
            _current_proc.terminate()
            _current_proc.wait()
            break

        exit_code = _current_proc.returncode
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        log.info(f"[Attempt {attempt}] Child exited code={exit_code} after {elapsed:.0f}s")

        if _shutdown:
            log.info("Shutdown requested -- not restarting.")
            break

        if elapsed < QUICK_DEATH_SECS:
            delay = QUICK_DEATH_DELAY_SECS
            log.warning(
                f"Child died in <{QUICK_DEATH_SECS}s -- likely config error or "
                f"missing dependency. Backing off {delay}s before retry."
            )
            notify_health(
                "Portfolio runner died quickly — likely config error or IB Gateway unreachable",
                severity="critical",
                detail=(
                    f"Attempt {attempt}: exit code {exit_code} after "
                    f"{elapsed:.0f}s. Restarting in {delay}s. Check "
                    f"`docker compose logs titan-portfolio` and "
                    f"`docker compose ps ib-gateway`."
                ),
            )
        else:
            delay = RESTART_DELAY_SECS
            notify_health(
                "Portfolio runner exited — auto-restarting",
                severity="warning",
                detail=(
                    f"Attempt {attempt}: exit code {exit_code} after "
                    f"{elapsed / 60:.1f} min. Likely IB Gateway disconnect "
                    f"or weekly maintenance. Restarting in {delay}s."
                ),
            )

        log.info(f"Restarting in {delay}s...")
        # Interruptible sleep so SIGTERM during cooldown takes effect immediately
        for _ in range(delay):
            if _shutdown:
                break
            time.sleep(1)

    log.info("Watchdog stopped.")
    notify_health("Watchdog stopped", severity="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
