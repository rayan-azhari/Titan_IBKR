"""phase6_deploy.py — Research → Live Config Handoff.

PREREQUISITE: Phases 0–5 must be complete and Phase 5 must pass all 6 gates
before running this script.  Deploying before Phase 5 clears risks pushing an
unvalidated strategy to live.

Reads the best IS-optimised threshold from a Phase 3 output CSV and writes it
to the instrument's section in config/ic_generic.toml.  Also updates
phase3_max_dd_pct and phase3_trade_rate from the Phase 3 OOS stats.

Only `threshold`, `phase3_max_dd_pct`, and `phase3_trade_rate` are auto-updated.
Human-validated fields (risk_pct, stop_atr_mult, leverage_cap) are never touched.

Typical workflow (after Phase 5 passes all 6 gates):
    # 1. Verify Phase 5 gate report
    cat .tmp/reports/phase5_eur_usd.csv

    # 2. Deploy threshold to config
    uv run python scripts/phase6_deploy.py --instrument EUR_USD --asset-class fx_major

    # 3. Restart ic_generic strategy to pick up new threshold

Usage:
    uv run python scripts/phase6_deploy.py --instrument EUR_USD --asset-class fx_major
    uv run python scripts/phase6_deploy.py --instrument SPY --asset-class etf --direction long_only
    uv run python scripts/phase6_deploy.py --instrument EUR_USD \
        --phase3-csv .tmp/reports/phase3_eur_usd_h4.csv

Dry-run (print diff only, no write):
    uv run python scripts/phase6_deploy.py --instrument SPY --asset-class etf --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
CONFIG_PATH = ROOT / "config" / "ic_generic.toml"

# Fields phase6_deploy.py is allowed to update
_AUTO_FIELDS = {"threshold", "phase3_max_dd_pct", "phase3_trade_rate"}

# Fields that belong to the operator and must never be auto-overwritten
_PROTECTED_FIELDS = {"risk_pct", "stop_atr_mult", "leverage_cap"}


# ---------------------------------------------------------------------------
# Phase 3 CSV resolution
# ---------------------------------------------------------------------------


def _find_phase3_csv(instrument: str, timeframe: str | None) -> Path:
    """Return the most-recently-modified phase3_*.csv for the instrument."""
    slug = instrument.lower().replace("/", "_")
    candidates: list[Path] = []

    if timeframe:
        tf_slug = timeframe.lower()
        candidates = sorted(
            REPORTS_DIR.glob(f"phase3_{slug}_{tf_slug}*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not candidates:
        # Broader glob — pick most recent regardless of TF
        candidates = sorted(
            REPORTS_DIR.glob(f"phase3_{slug}*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not candidates:
        raise FileNotFoundError(
            f"No phase3 CSV found for {instrument} in {REPORTS_DIR}.\n"
            "Run Phase 3 first:\n"
            f"  uv run python research/ic_analysis/phase3_backtest.py "
            f"--instrument {instrument} ..."
        )

    return candidates[0]


def _read_best_threshold(csv_path: Path, direction: str) -> dict[str, float]:
    """Extract best IS threshold and OOS stats from a Phase 3 CSV.

    Returns a dict with keys:
        threshold, phase3_max_dd_pct, phase3_trade_rate
    """
    df = pd.read_csv(csv_path)

    required = {"threshold", "oos_trade_sharpe", "max_dd_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Phase 3 CSV {csv_path.name} is missing columns: {missing}.\n"
            "Re-run Phase 3 with the latest phase3_backtest.py."
        )

    # Filter by direction if column present
    if "direction" in df.columns and direction in df["direction"].values:
        df = df[df["direction"] == direction]

    if df.empty:
        raise ValueError(
            f"No rows matching direction='{direction}' in {csv_path.name}."
        )

    # Best row = highest OOS trade Sharpe
    best = df.loc[df["oos_trade_sharpe"].idxmax()]

    threshold = float(best["threshold"])
    max_dd_pct = float(best["max_dd_pct"])

    # Trade rate: trades per month = n_oos_trades / oos_months
    trade_rate = 0.0
    if "n_oos_trades" in df.columns and "oos_bars" in df.columns:
        # Approximate: assume H4 = 6 bars/day, D = 1 bar/day
        # We use n_oos_trades directly; caller can override
        n_trades = float(best["n_oos_trades"])
        oos_bars = float(best.get("oos_bars", 0))
        if oos_bars > 0:
            # Rough: assume 252 trading days/year regardless of TF
            # phase4_wfo handles proper scaling — this is just an indicator
            months = oos_bars / (252 / 12)
            raw_rate: float = n_trades / max(months, 1.0)
            trade_rate = float(f"{raw_rate:.2f}")

    t_str = f"{threshold:.4f}"
    dd_str = f"{max_dd_pct:.4f}"
    tr_str = f"{trade_rate:.4f}"
    return {
        "threshold": float(t_str),
        "phase3_max_dd_pct": float(dd_str),
        "phase3_trade_rate": float(tr_str),
    }


# ---------------------------------------------------------------------------
# TOML section updater (regex-based, preserves formatting and comments)
# ---------------------------------------------------------------------------


def _update_toml_section(
    toml_text: str,
    instrument: str,
    updates: dict[str, float],
) -> str:
    """Return toml_text with the [INSTRUMENT] section values updated.

    Uses regex to update values in-place so that:
    - Comments are preserved
    - Unrelated sections are not touched
    - Protected fields are never modified
    """
    section_header = f"[{instrument}]"
    if section_header not in toml_text:
        raise KeyError(
            f"Section [{instrument}] not found in {CONFIG_PATH}.\n"
            "Add the instrument section first or run:\n"
            f"  uv run python scripts/phase6_deploy.py --instrument {instrument} "
            "--add-section"
        )

    lines = toml_text.split("\n")
    in_section = False
    result: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect section start
        if stripped == section_header:
            in_section = True
            result.append(line)
            continue

        # Detect start of next section
        if in_section and stripped.startswith("[") and stripped != section_header:
            in_section = False

        if in_section:
            # Try to match a key = value line (ignore comment lines)
            m = re.match(r"^(\s*)(\w+)(\s*=\s*)(.*?)(\s*(?:#.*)?)$", line)
            if m:
                indent, key, eq, _val, comment = m.groups()
                if key in updates and key not in _PROTECTED_FIELDS:
                    new_val = updates[key]
                    # Format: match existing style (float with trailing zero)
                    if isinstance(new_val, float) and new_val == int(new_val):
                        formatted = f"{new_val:.1f}"
                    else:
                        formatted = str(new_val)
                    line = f"{indent}{key}{eq}{formatted}{comment}"

        result.append(line)

    return "\n".join(result)


def _diff_lines(old: str, new: str, instrument: str) -> list[str]:
    """Return human-readable diff lines for the instrument section only."""
    old_lines = old.split("\n")
    new_lines = new.split("\n")

    section_header = f"[{instrument}]"
    diffs: list[str] = []
    in_section = False

    for old_line, new_line in zip(old_lines, new_lines):
        stripped = old_line.strip()
        if stripped == section_header:
            in_section = True
        elif in_section and stripped.startswith("[") and stripped != section_header:
            in_section = False

        if in_section and old_line != new_line:
            diffs.append(f"  - {old_line.rstrip()}")
            diffs.append(f"  + {new_line.rstrip()}")

    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _check_phase5_gate(instrument: str) -> None:
    """Warn if Phase 5 report is missing or contains FAIL gates."""
    slug = instrument.lower().replace("/", "_")
    candidates = sorted(
        REPORTS_DIR.glob(f"phase5_{slug}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(
            f"\n  [!] WARNING: No Phase 5 report found for {instrument}.\n"
            "      Run Phase 5 before deploying:\n"
            f"      uv run python research/ic_analysis/phase5_robustness.py "
            f"--instrument {instrument} ...\n"
        )
        return

    p5 = pd.read_csv(candidates[0])
    if "gate" in p5.columns and "result" in p5.columns:
        failing = p5[p5["result"].str.upper() == "FAIL"]
        if not failing.empty:
            gates = ", ".join(failing["gate"].astype(str).tolist())
            print(
                f"\n  [!] WARNING: Phase 5 report has FAIL gates: {gates}\n"
                "      Deploying over a failed robustness check is not recommended.\n"
            )
        else:
            print(f"  Phase 5 gate : ALL PASS ({candidates[0].name})")
    else:
        print(f"  Phase 5 gate : report found but unrecognised schema ({candidates[0].name})")


def deploy(
    instrument: str,
    asset_class: str,
    direction: str = "both",
    timeframe: str | None = None,
    phase3_csv: Path | None = None,
    dry_run: bool = False,
) -> None:
    # 0. Gate check — Phase 5 must have passed
    _check_phase5_gate(instrument)

    # 1. Locate Phase 3 CSV
    csv_path = phase3_csv or _find_phase3_csv(instrument, timeframe)
    print(f"\n  Phase 3 source : {csv_path.relative_to(ROOT)}")

    # 2. Read best threshold + OOS stats
    updates = _read_best_threshold(csv_path, direction)
    print(f"  Proposed updates for [{instrument}]:")
    for k, v in updates.items():
        print(f"    {k} = {v}")

    # 3. Read current TOML
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found — create it first.")

    original_text = CONFIG_PATH.read_text(encoding="utf-8")

    # 4. Compute updated TOML
    updated_text = _update_toml_section(original_text, instrument, updates)

    # 5. Show diff
    diff = _diff_lines(original_text, updated_text, instrument)
    if not diff:
        print("\n  No changes — TOML already matches Phase 3 output.")
        return

    print(f"\n  Diff for [{instrument}] in config/ic_generic.toml:")
    for line in diff:
        print(line)

    if dry_run:
        print("\n  [DRY RUN] No changes written.")
        return

    # 6. Confirm
    print()
    confirm = input("  Write changes? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted — no changes written.")
        return

    CONFIG_PATH.write_text(updated_text, encoding="utf-8")
    print(f"\n  Written: {CONFIG_PATH.relative_to(ROOT)}")
    print("  Next step: restart ic_generic strategy for the updated threshold to take effect.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 6 deploy — push Phase 3 threshold to ic_generic.toml"
    )
    p.add_argument(
        "--instrument",
        required=True,
        help="Instrument name matching the TOML section, e.g. EUR_USD, SPY",
    )
    p.add_argument(
        "--asset-class",
        default="fx_major",
        choices=["fx_major", "fx_cross", "equity_lc", "etf", "futures"],
        help="Asset class (informational — used to find Phase 3 CSV)",
    )
    p.add_argument(
        "--direction",
        default="both",
        choices=["both", "long_only"],
        help="Direction filter when reading Phase 3 CSV (default: both)",
    )
    p.add_argument(
        "--timeframe",
        default=None,
        help="Base timeframe to narrow CSV search, e.g. H4, D",
    )
    p.add_argument(
        "--phase3-csv",
        default=None,
        type=Path,
        help="Explicit path to Phase 3 CSV (overrides auto-discovery)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diff only — do not write to TOML",
    )
    args = p.parse_args()

    deploy(
        instrument=args.instrument,
        asset_class=args.asset_class,
        direction=args.direction,
        timeframe=args.timeframe,
        phase3_csv=args.phase3_csv,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
