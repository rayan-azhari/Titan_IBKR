"""monitor_live_drift.py -- CUSUM drift monitor wired to live PnL stream.

V3.7 / L67: detects when live portfolio performance materially diverges
from research expectation. Uses the CUSUM detector from
`titan.research.framework.drift_cusum` against:

  - Live returns: daily portfolio NAV changes extracted from titan-portfolio
    container logs (AccountState events).
  - Research baseline: stitched OOS returns from the most recent joint
    portfolio evaluation (GEM 80% + turtle 20% @ 5y common window).

Why portfolio-level (not per-strategy): the live container already aggregates
PnL into a single AccountState event every ~3 minutes; we can extract daily
NAV changes from those. Per-strategy CUSUM needs in-container snapshotting
of per-strategy equity, which is out of scope during the V3.7 paper
validation period (we avoid touching the live runner).

State files:
  .tmp/live_drift/live_nav_history.jsonl   -- per-day NAV snapshot
  .tmp/live_drift/cusum_state.json          -- current upper/lower CUSUM values
  .tmp/live_drift/alerts.jsonl              -- record of breach events

Behaviour:
  1. Extract today's NAV from latest AccountState in container logs.
  2. Append to NAV history file (one entry per day).
  3. Compute daily log-return from NAV[-2] -> NAV[-1].
  4. Append to in-memory live_returns series; run CUSUM vs research baseline.
  5. If lower-CUSUM crosses threshold: write alert + notify_health (critical).

Run daily via cron / Docker compose alongside the daily portfolio summary.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework.drift_cusum import run_cusum_drift  # noqa: E402

STATE_DIR = PROJECT_ROOT / ".tmp" / "live_drift"
STATE_DIR.mkdir(parents=True, exist_ok=True)
NAV_HISTORY = STATE_DIR / "live_nav_history.jsonl"
CUSUM_STATE = STATE_DIR / "cusum_state.json"
ALERTS_LOG = STATE_DIR / "alerts.jsonl"

# Live deployment metadata.
CUTOVER_TS = pd.Timestamp("2026-05-17T23:49:00Z")

# CUSUM gate (Page 1954): h=5.0 corresponds to ~1% false-positive at 252 bars.
CUSUM_K_SLACK = 0.5
CUSUM_H_THRESHOLD = 5.0

ACCOUNT_STATE_RE = re.compile(
    r"Updated AccountState\(account_id=(?P<account_id>[^,]+),.*"
    r"balances=\[AccountBalance\(total=(?P<total>[\d_\.]+) (?P<ccy>\w+),"
)


def _research_baseline_returns() -> pd.Series:
    """Reload the joint portfolio research baseline (GEM 80% + turtle 20%).

    Returns the stitched-OOS daily return series used in the V3.7 portfolio
    evaluation. CUSUM compares live returns against the MEAN + STD of this.
    """
    from research.portfolio.joint_evaluation import (
        build_portfolio,
        gem_j5_returns,
        turtle_cat_returns_daily,
    )

    gem = gem_j5_returns().dropna()
    turtle = turtle_cat_returns_daily().dropna()
    return build_portfolio(
        {"gem_j5": gem, "turtle_cat": turtle},
        {"gem_j5": 0.80, "turtle_cat": 0.20},
    ).dropna()


def _latest_account_state() -> dict | None:
    """Get the most recent AccountState event from container logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", "200", "titan-portfolio"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        logs = result.stdout.splitlines()
    except Exception as e:  # noqa: BLE001
        print(f"docker logs failed: {e}", file=sys.stderr)
        return None
    for line in reversed(logs):
        m = ACCOUNT_STATE_RE.search(line)
        if m:
            return {
                "account_id": m.group("account_id"),
                "balance_total": float(m.group("total").replace("_", "")),
                "currency": m.group("ccy"),
                "raw_line": line[:200],
                "extracted_ts": datetime.now(timezone.utc).isoformat(),
            }
    return None


def _append_nav_history(snapshot: dict) -> None:
    """Append today's NAV snapshot. Idempotent on date (one entry per day)."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot["date"] = date_str

    # Read existing history to check if today already recorded.
    existing = []
    if NAV_HISTORY.exists():
        for line in NAV_HISTORY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Skip if today is already recorded.
    if any(e.get("date") == date_str for e in existing):
        print(f"[drift] today ({date_str}) already in NAV history -- updating last entry")
        existing = [e for e in existing if e.get("date") != date_str]

    existing.append(snapshot)

    # Rewrite the file.
    with NAV_HISTORY.open("w", encoding="utf-8") as fh:
        for e in existing:
            fh.write(json.dumps(e) + "\n")


def _live_returns_from_history() -> pd.Series:
    """Compute per-day log-returns from the NAV history."""
    if not NAV_HISTORY.exists():
        return pd.Series(dtype=float)
    rows = []
    for line in NAV_HISTORY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if len(rows) < 2:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    nav = df["balance_total"].astype(float)
    return np.log(nav / nav.shift(1)).dropna()


def main() -> int:
    print(f"[drift] V3.7 CUSUM drift monitor -- {datetime.now(timezone.utc).isoformat()}")

    # 1. Capture latest NAV from container.
    snapshot = _latest_account_state()
    if snapshot is None:
        print("[drift] failed to capture AccountState -- skipping")
        return 1
    _append_nav_history(snapshot)
    print(f"  NAV captured: {snapshot['balance_total']:.2f} {snapshot['currency']}")

    # 2. Compute live returns from history.
    live_returns = _live_returns_from_history()
    n_live = len(live_returns)
    print(f"  live return history: {n_live} bars")
    if n_live < 5:
        print("  insufficient live history for CUSUM (need >= 5 bars) -- recording NAV and exiting")
        CUSUM_STATE.write_text(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "status": "insufficient_history",
                    "n_live": n_live,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    # 3. Load research baseline.
    try:
        baseline = _research_baseline_returns()
    except Exception as e:  # noqa: BLE001
        print(f"  [error] failed to load research baseline: {e}")
        return 1
    print(
        f"  research baseline: {len(baseline)} bars, mean={baseline.mean():.6f}, std={baseline.std():.6f}"
    )

    # 4. Run CUSUM.
    res = run_cusum_drift(
        live_returns,
        baseline,
        k_slack=CUSUM_K_SLACK,
        h_threshold=CUSUM_H_THRESHOLD,
    )
    print(res.report())

    # 5. Persist state.
    state = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_live": n_live,
        "n_baseline": len(baseline),
        "live_mean": res.live_mean,
        "live_std": res.live_std,
        "oos_mean": res.oos_mean,
        "oos_std": res.oos_std,
        "upper_cusum": res.upper_cusum,
        "lower_cusum": res.lower_cusum,
        "upper_breach": res.upper_breach,
        "lower_breach": res.lower_breach,
        "breach_bar": str(res.breach_bar) if res.breach_bar else None,
    }
    CUSUM_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # 6. Alert on breach.
    if res.lower_breach:
        alert = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "severity": "critical",
            "type": "lower_cusum_breach",
            "lower_cusum": res.lower_cusum,
            "h_threshold": CUSUM_H_THRESHOLD,
            "n_live": n_live,
            "detail": "Live portfolio performance materially below research expectation",
        }
        with ALERTS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert) + "\n")
        try:
            from titan.utils.notification import notify_health

            notify_health(
                f"CUSUM LOWER BREACH: live drift below research baseline "
                f"(lower={res.lower_cusum:.2f}, threshold={CUSUM_H_THRESHOLD})",
                severity="critical",
                detail=f"n_live={n_live}, live_mean={res.live_mean:.6f}, "
                f"baseline_mean={res.oos_mean:.6f}. Re-audit recommended.",
            )
        except Exception:  # noqa: BLE001
            pass
        print("  [ALERT] LOWER CUSUM BREACH -- alert recorded + notify_health sent")
    elif res.upper_breach:
        print("  [info] upper CUSUM breach -- live exceeds research (informational)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
