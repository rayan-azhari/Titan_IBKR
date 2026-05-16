"""watchdog_gem.py -- Autonomous watchdog for the GEM Dual Momentum strategy.

Runs ``scripts/run_live_gem.py`` in a subprocess. If it exits for any reason
(crash, IB Gateway disconnect, weekly maintenance), waits briefly and
restarts. Designed for unattended Docker / systemd operation.

Selected production cell: C12 (C8 + max_leverage=2.0) per
``directives/Pre-Reg GEM Dual Momentum 2026-05-14.md`` §4.2.

Usage:
    uv run python scripts/watchdog_gem.py

Behaviour mirrors ``watchdog_portfolio.py``:
  - Restarts on any child exit (clean or crash) until SIGTERM/SIGINT.
  - 180s normal restart delay (covers IBKR weekly maintenance).
  - 60s delay if child dies in <30s (likely config error).
  - SIGTERM forwarding so ``docker compose down`` propagates cleanly.
  - Logs to stdout (Docker captures) AND ``.tmp/logs/watchdog_gem.log``.
"""

from __future__ import annotations

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

sys.path.insert(0, str(PROJECT_ROOT))
try:
    from titan.utils.notification import notify_health
except Exception:  # noqa: BLE001

    def notify_health(*_args, **_kwargs) -> int:  # type: ignore[no-redef]
        return 0


RESTART_DELAY_SECS = 180
QUICK_DEATH_SECS = 30
QUICK_DEATH_DELAY_SECS = 60
STRATEGY_SCRIPT = PROJECT_ROOT / "scripts" / "run_live_gem.py"


log_file = LOGS_DIR / "watchdog_gem.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog_gem")


_shutdown = False
_current_proc: subprocess.Popen | None = None


def _handle_signal(signum: int, frame) -> None:  # noqa: ARG001
    global _shutdown
    _shutdown = True
    log.info(f"Watchdog received signal {signum}. Forwarding to child and stopping.")
    notify_health(
        f"Watchdog received signal {signum}; shutting down GEM runner",
        severity="info",
    )
    if _current_proc is not None and _current_proc.poll() is None:
        try:
            _current_proc.terminate()
        except ProcessLookupError:
            pass


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main() -> int:
    cmd = [sys.executable, str(STRATEGY_SCRIPT)]
    log.info(f"Watchdog starting. Child command: {' '.join(cmd)}")
    log.info(
        f"Restart cooldown: {RESTART_DELAY_SECS}s normal, "
        f"{QUICK_DEATH_DELAY_SECS}s if child dies in <{QUICK_DEATH_SECS}s"
    )
    notify_health(
        "Watchdog started -- GEM Dual Momentum (C12 production)",
        severity="ok",
    )

    global _current_proc
    attempt = 0
    while not _shutdown:
        attempt += 1
        start = datetime.now(timezone.utc)
        log.info(f"[Attempt {attempt}] Starting GEM runner at {start.isoformat()}")

        _current_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

        try:
            _current_proc.wait()
        except KeyboardInterrupt:
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
                "GEM runner died quickly -- likely config error or IB Gateway unreachable",
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
                "GEM runner exited -- auto-restarting",
                severity="warning",
                detail=(
                    f"Attempt {attempt}: exit code {exit_code} after "
                    f"{elapsed / 60:.1f} min. Likely IB Gateway disconnect "
                    f"or weekly maintenance. Restarting in {delay}s."
                ),
            )

        log.info(f"Restarting in {delay}s...")
        for _ in range(delay):
            if _shutdown:
                break
            time.sleep(1)

    log.info("Watchdog stopped.")
    notify_health("Watchdog stopped", severity="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
