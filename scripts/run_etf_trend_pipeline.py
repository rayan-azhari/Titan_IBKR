"""run_etf_trend_pipeline.py — End-to-end ETF Trend Optimization Pipeline.

Runs all 6 optimization stages sequentially for any instrument,
passing results between stages automatically via the state manager.

Stages:
  1. run_optimisation.py    — MA type + entry period sweep
  2. run_stage2_decel.py    — Deceleration signal selection
  3. run_stage3_exits.py    — Exit mode sweep (A/B/C/D)
  4. run_stage4_sizing.py   — Vol sizing mode + ATR stop sweep: writes config
  5. run_portfolio.py       — Full friction P&L simulation + B&H comparison
  6. run_robustness.py      — Monte Carlo + Rolling WFO

Usage:
    uv run python scripts/run_etf_trend_pipeline.py --instrument SPY
    uv run python scripts/run_etf_trend_pipeline.py --instrument SPY --from-stage 3
    uv run python scripts/run_etf_trend_pipeline.py --instrument SPY --skip-robustness
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _python_cmd() -> list[str]:
    """Return the best available Python command for this project."""
    import shutil

    if shutil.which("uv"):
        return ["uv", "run", "python"]
    for venv_python in [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]:
        if venv_python.exists():
            return [str(venv_python)]
    return [sys.executable]


def run_stage(label: str, cmd: list[str], abort_on_fail: bool = True) -> bool:
    """Run a pipeline stage and return True on success.

    Args:
        label: Human-readable stage label for display.
        cmd: Shell command list to execute.
        abort_on_fail: If True, exit the pipeline on failure.

    Returns:
        True if stage completed successfully.
    """
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  CMD: {' '.join(cmd)}\n")

    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)

    if result.returncode != 0:
        print(f"\n  [FAIL] {label} exited with code {result.returncode}")
        if abort_on_fail:
            print("  Pipeline aborted. Fix the issue and resume with --from-stage.")
            sys.exit(result.returncode)
        return False

    print(f"\n  [PASS] {label} completed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETF Trend Pipeline Orchestrator — runs all 6 stages for any instrument.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  run_optimisation.py    MA type + entry period sweep
  2  run_stage2_decel.py    Deceleration signal selection
  3  run_stage3_exits.py    Exit mode sweep (A/B/C/D)
  4  run_stage4_sizing.py   Vol sizing + ATR stop sweep: writes config TOML
  5  run_portfolio.py       Full friction P&L + buy-and-hold comparison
  6  run_robustness.py      Monte Carlo + Rolling WFO

Examples:
  uv run python scripts/run_etf_trend_pipeline.py --instrument SPY
  uv run python scripts/run_etf_trend_pipeline.py --instrument SPY --from-stage 3
  uv run python scripts/run_etf_trend_pipeline.py --instrument SPY --skip-robustness
        """,
    )
    parser.add_argument(
        "--instrument",
        default="SPY",
        help="Instrument symbol (default: SPY)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=range(1, 7),
        metavar="N",
        help="Resume pipeline from stage N (1-6). Earlier stages are skipped.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip Stage 6 (Monte Carlo + WFO). Faster for initial sweeps.",
    )
    args = parser.parse_args()

    instrument = args.instrument.upper()
    start = args.from_stage
    uv = _python_cmd()

    print("=" * 70)
    print("  ETF TREND OPTIMIZATION PIPELINE")
    print(f"  Instrument: {instrument}  |  Starting from stage {start}")
    print("=" * 70)

    gate_results: dict[str, bool] = {}

    # ── Stage 1: MA type + period sweep ───────────────────────────────────
    if start <= 1:
        ok = run_stage(
            "Stage 1: MA Type + Entry Period Sweep",
            uv + ["research/etf_trend/run_optimisation.py", "--instrument", instrument],
        )
        gate_results["Stage 1"] = ok
    else:
        print(f"\n  [SKIP] Stage 1 (--from-stage {start})")

    # ── Stage 2: Deceleration signal selection ─────────────────────────────
    if start <= 2:
        ok = run_stage(
            "Stage 2: Deceleration Signal Selection",
            uv
            + [
                "research/etf_trend/run_stage2_decel.py",
                "--instrument",
                instrument,
                "--load-state",
            ],
        )
        gate_results["Stage 2"] = ok
    else:
        print(f"\n  [SKIP] Stage 2 (--from-stage {start})")

    # ── Stage 3: Exit mode sweep ──────────────────────────────────────────
    if start <= 3:
        ok = run_stage(
            "Stage 3: Exit Mode Sweep (A/B/C/D)",
            uv
            + [
                "research/etf_trend/run_stage3_exits.py",
                "--instrument",
                instrument,
                "--load-state",
            ],
        )
        gate_results["Stage 3"] = ok
    else:
        print(f"\n  [SKIP] Stage 3 (--from-stage {start})")

    # ── Stage 4: Sizing + ATR stop sweep ──────────────────────────────────
    if start <= 4:
        ok = run_stage(
            "Stage 4: Vol Sizing + ATR Stop Sweep",
            uv
            + [
                "research/etf_trend/run_stage4_sizing.py",
                "--instrument",
                instrument,
                "--load-state",
            ],
        )
        gate_results["Stage 4"] = ok
    else:
        print(f"\n  [SKIP] Stage 4 (--from-stage {start})")

    # ── Stage 5: Full friction portfolio simulation ────────────────────────
    if start <= 5:
        ok = run_stage(
            "Stage 5: Portfolio Simulation (Full Friction + B&H Baseline)",
            uv + ["research/etf_trend/run_portfolio.py", "--instrument", instrument],
        )
        gate_results["Stage 5"] = ok
    else:
        print(f"\n  [SKIP] Stage 5 (--from-stage {start})")

    # ── Stage 6: Robustness (Monte Carlo + WFO) ────────────────────────────
    if start <= 6 and not args.skip_robustness:
        ok = run_stage(
            "Stage 6: Robustness Validation (Monte Carlo + WFO)",
            uv + ["research/etf_trend/run_robustness.py", "--instrument", instrument],
            abort_on_fail=False,
        )
        gate_results["Stage 6 (Robustness)"] = ok
    elif args.skip_robustness:
        print("\n  [SKIP] Stage 6 (--skip-robustness)")

    # ── Final summary ──────────────────────────────────────────────────────
    inst_lower = instrument.lower()
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — {instrument}")
    print("=" * 70)
    for name, passed in gate_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(gate_results.values())
    if all_pass:
        print("\n  All stages passed.")
        print(f"  Config saved to: config/etf_trend_{inst_lower}.toml")
        print(f"  Reports in:      .tmp/reports/etf_trend_{inst_lower}_*.html/csv")
        print("\n  Next steps:")
        print("  1. Review OOS Sharpe and B&H comparison in the HTML report")
        print("  2. Run 60-day paper trade: uv run python scripts/run_live_etf_trend.py")
    else:
        failed = [k for k, v in gate_results.items() if not v]
        print(f"\n  Failed stages: {', '.join(failed)}")
        print(f"  Resume with: --instrument {instrument} --from-stage <N>")

    print("=" * 70)


if __name__ == "__main__":
    main()
