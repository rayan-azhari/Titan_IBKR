"""chaos_harness.py — scripted chaos scenarios for the live container stack.

Tier 2.3 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).

Tier 1.2 (``smoke_double_restart``) verifies a single happy-path restart
cycle. Tier 2.3 adds adversarial scenarios that the live system can hit
in production: restart storms, gateway flapping, and recovery-under-
operation. Each scenario sets up state, executes a controlled
disruption, waits for recovery, and asserts the system's invariants
still hold.

Scenarios (v1)
--------------

  S1. **Restart storm.** N (default 3) container restarts with random
      delays (5-30s) between each. Catches state-machine bugs that only
      manifest under fast successive restarts (rehydration race
      conditions, half-completed initialisation).

  S2. **Gateway flap.** Restart only ``titan-ib-gateway`` (leaving
      ``titan-portfolio`` running) and verify the strategy container
      reconnects cleanly without state drift. Catches connection-watchdog
      regressions and rehydration-on-reconnect bugs.

  S3. **Rapid back-to-back restarts.** Three restarts of
      ``titan-portfolio`` within 60s, no random delay. Stress-test the
      double-restart smoke-script's invariant under tighter timing.

For each scenario, after recovery:
  - Position snapshot taken via ibapi (fresh client_id)
  - Diffed against pre-chaos snapshot (any qty / count drift = fail)
  - Logs scanned for new ``[D1]``/``[D2]``/``[D5]`` alerts since chaos start
    (any new alert = fail)
  - Logs scanned for unhandled exception traces (any = fail)

Usage::

    uv run python scripts/chaos_harness.py                    # run all scenarios
    uv run python scripts/chaos_harness.py --scenario S2      # one scenario
    uv run python scripts/chaos_harness.py --skip S1          # skip restart-storm
    uv run python scripts/chaos_harness.py --dry-run          # plan only, no chaos

Cron suggestion (Sunday 04:00 UTC, after cost_audit at 02:00 + replay
at 03:00; markets closed window)::

    0 4 * * 0  cd /app && uv run python scripts/chaos_harness.py

Exit codes::

    0  All requested scenarios passed
    1  One or more scenarios failed (notification dispatched)
    2  Operational error (gateway unreachable, docker unavailable)

Caveats
-------
  * Must be run during a quiet window (no bars firing). Best: weekend or
    17:00-23:00 UTC weekdays.
  * Only operates on docker compose services. The host must have docker
    + the project's compose file accessible. Set CWD to project root.
  * The "is_long" check inside scenarios uses the snapshot-positions
    qty, not the strategy's internal state — covers broker-truth
    invariants only.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.smoke_double_restart import (  # noqa: E402
    _docker_restart,
    _snapshot_positions,
    _wait_for_healthy,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "chaos"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_SERVICE = "titan-portfolio"
GATEWAY_SERVICE = "titan-ib-gateway"

# Patterns we scan post-chaos in docker logs to detect new failures.
# Reconciliation D-alerts and unhandled exceptions are the primary
# signals. Excludes routine warnings (e.g. data-farm-OK heartbeat).
_FAILURE_LOG_PATTERNS = [
    "[D1 multi-position]",
    "[D2 sum-drift]",
    "[D5 shadow",  # D5 fires per-strategy, prefix-only match
    "Traceback (most recent call last):",
    "Connection failed",  # NT-internal
]


# ── Result + scenario primitives ─────────────────────────────────────


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    duration_sec: float
    findings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _diff_snapshots(before: dict[str, dict], after: dict[str, dict]) -> list[str]:
    """Return human-readable findings for snapshot mismatches.

    Lifted from smoke_double_restart's invariant check — same semantics.
    """
    findings: list[str] = []
    all_symbols = set(before.keys()) | set(after.keys())
    for sym in sorted(all_symbols):
        b = before.get(sym, {"qty": 0.0, "position_count": 0})
        a = after.get(sym, {"qty": 0.0, "position_count": 0})
        if b["qty"] != a["qty"]:
            findings.append(f"  {sym}: qty {b['qty']:+.2f} -> {a['qty']:+.2f}")
        if b["position_count"] != a["position_count"]:
            findings.append(
                f"  {sym}: position_count {b['position_count']} -> {a['position_count']} "
                f"(SUSPICIOUS — May 11 doubling-bug class)"
            )
    return findings


def _scan_logs_for_failures(service: str, since_ts: float) -> list[str]:
    """Return new occurrences of failure patterns in ``service`` logs since
    ``since_ts`` (unix seconds). Empty list = clean."""
    since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
    since_arg = since_dt.strftime("%Y-%m-%dT%H:%M:%S")
    res = subprocess.run(
        ["docker", "logs", "--since", since_arg, service],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return [f"docker logs failed for {service}: {res.stderr.strip()}"]
    full_log = res.stdout + res.stderr
    findings: list[str] = []
    for pattern in _FAILURE_LOG_PATTERNS:
        # Count occurrences for visibility (1 traceback line is fine; 50
        # is a clear escalation).
        count = full_log.count(pattern)
        if count > 0:
            # Emit one finding per pattern, with sample count + first 200
            # chars of context.
            idx = full_log.find(pattern)
            snippet = full_log[max(0, idx - 80) : idx + 200].replace("\n", " | ")
            findings.append(f"  '{pattern}' x{count}: ...{snippet}...")
    return findings


def _verify_recovery(
    service: str,
    before_snapshot: dict[str, dict],
    chaos_started_ts: float,
) -> list[str]:
    """Common post-chaos invariant check: snapshot diff + log scan."""
    findings: list[str] = []
    try:
        after_snapshot = _snapshot_positions()
    except Exception as e:
        return [f"  snapshot post-chaos failed: {e}"]
    findings.extend(_diff_snapshots(before_snapshot, after_snapshot))
    findings.extend(_scan_logs_for_failures(service, chaos_started_ts))
    return findings


# ── Scenarios ────────────────────────────────────────────────────────


def scenario_s1_restart_storm(
    n_restarts: int = 3,
    min_delay_sec: int = 5,
    max_delay_sec: int = 30,
    healthy_timeout_sec: int = 90,
) -> ScenarioResult:
    """S1: Restart titan-portfolio N times with random delays between.

    Catches state-machine bugs that only manifest under successive
    restart timing pressure (rehydration race, half-completed init).
    """
    name = f"S1 restart-storm (n={n_restarts}, delay {min_delay_sec}-{max_delay_sec}s)"
    started = time.time()
    print(f"\n[S1] {name}")
    print("-" * 70)

    rng = random.Random(42)  # deterministic for re-running
    try:
        before = _snapshot_positions()
        print(f"  Pre-chaos snapshot: {len(before)} symbols")
        for i in range(n_restarts):
            print(f"  Restart {i + 1}/{n_restarts}...")
            _docker_restart(PORTFOLIO_SERVICE)
            _wait_for_healthy(PORTFOLIO_SERVICE, timeout_sec=healthy_timeout_sec)
            if i < n_restarts - 1:
                delay = rng.randint(min_delay_sec, max_delay_sec)
                print(f"  Sleeping {delay}s before next restart...")
                time.sleep(delay)
    except Exception as e:
        return ScenarioResult(
            name=name,
            passed=False,
            duration_sec=time.time() - started,
            findings=[f"  Operational error: {e}"],
        )

    findings = _verify_recovery(PORTFOLIO_SERVICE, before, started)
    return ScenarioResult(
        name=name,
        passed=not findings,
        duration_sec=time.time() - started,
        findings=findings,
    )


def scenario_s2_gateway_flap(
    healthy_timeout_sec: int = 120,
) -> ScenarioResult:
    """S2: Restart only the IB gateway. Strategy container should
    reconnect cleanly without state drift."""
    name = "S2 gateway-flap"
    started = time.time()
    print(f"\n[S2] {name}")
    print("-" * 70)
    try:
        before = _snapshot_positions()
        print(f"  Pre-chaos snapshot: {len(before)} symbols")
        print(f"  Restarting {GATEWAY_SERVICE}...")
        _docker_restart(GATEWAY_SERVICE)
        # Gateway takes longer to come up (IBC + Xvfb + 2FA)
        print(f"  Waiting for {GATEWAY_SERVICE} to be healthy...")
        _wait_for_healthy(GATEWAY_SERVICE, timeout_sec=healthy_timeout_sec)
        # Give titan-portfolio time to detect + reconnect
        print("  Sleeping 60s to let strategy container reconnect...")
        time.sleep(60)
    except Exception as e:
        return ScenarioResult(
            name=name,
            passed=False,
            duration_sec=time.time() - started,
            findings=[f"  Operational error: {e}"],
        )

    findings = _verify_recovery(PORTFOLIO_SERVICE, before, started)
    return ScenarioResult(
        name=name,
        passed=not findings,
        duration_sec=time.time() - started,
        findings=findings,
    )


def scenario_s3_rapid_restarts(
    n_restarts: int = 3,
    healthy_timeout_sec: int = 90,
) -> ScenarioResult:
    """S3: Three back-to-back restarts with no inter-restart delay.

    Stress-test for any race between docker stop, gateway re-handshake,
    and rehydration. Tighter than S1's random-delay version.
    """
    name = f"S3 rapid-restarts (n={n_restarts}, no delay)"
    started = time.time()
    print(f"\n[S3] {name}")
    print("-" * 70)
    try:
        before = _snapshot_positions()
        print(f"  Pre-chaos snapshot: {len(before)} symbols")
        for i in range(n_restarts):
            print(f"  Restart {i + 1}/{n_restarts}...")
            _docker_restart(PORTFOLIO_SERVICE)
            _wait_for_healthy(PORTFOLIO_SERVICE, timeout_sec=healthy_timeout_sec)
    except Exception as e:
        return ScenarioResult(
            name=name,
            passed=False,
            duration_sec=time.time() - started,
            findings=[f"  Operational error: {e}"],
        )

    findings = _verify_recovery(PORTFOLIO_SERVICE, before, started)
    return ScenarioResult(
        name=name,
        passed=not findings,
        duration_sec=time.time() - started,
        findings=findings,
    )


SCENARIOS = {
    "S1": scenario_s1_restart_storm,
    "S2": scenario_s2_gateway_flap,
    "S3": scenario_s3_rapid_restarts,
}


# ── Main ─────────────────────────────────────────────────────────────


def _format_report(results: list[ScenarioResult]) -> str:
    lines = ["=" * 70, "CHAOS HARNESS RESULTS", "=" * 70]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{status}] {r.name}  ({r.duration_sec:.1f}s)")
        for f in r.findings:
            lines.append(f.rstrip())
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS.keys()), default=None)
    parser.add_argument("--skip", action="append", choices=list(SCENARIOS.keys()), default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.scenario or list(SCENARIOS.keys())
    if args.skip:
        selected = [s for s in selected if s not in args.skip]

    print("=" * 70)
    print(f"CHAOS HARNESS — {len(selected)} scenarios queued: {selected}")
    print("=" * 70)
    if args.dry_run:
        print("\n--dry-run: would execute the above. Exiting without chaos.")
        return 0

    results: list[ScenarioResult] = []
    for name in selected:
        try:
            results.append(SCENARIOS[name]())
        except Exception as e:
            results.append(
                ScenarioResult(
                    name=name,
                    passed=False,
                    duration_sec=0.0,
                    findings=[f"  Harness exception: {e}"],
                )
            )

    report = _format_report(results)
    print()
    print(report)

    out = REPORTS_DIR / f"chaos_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out}")

    failed = [r for r in results if not r.passed]
    if not failed:
        print("\n  ALL SCENARIOS PASSED.")
        return 0

    print(f"\n  {len(failed)} SCENARIO(S) FAILED.")
    try:
        from titan.utils.notification import notify_health

        lines = [f"Chaos harness failed {len(failed)}/{len(results)} scenario(s):"]
        for r in failed:
            lines.append(f"  - {r.name}")
            for f in r.findings:
                lines.append(f"      {f.strip()}")
        notify_health(
            event="chaos_harness_failure",
            severity="critical",
            detail="\n".join(lines),
        )
        print("  Notification dispatched via notify_health.")
    except Exception as e:
        print(f"  notify_health failed: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
