"""run_mtf_pipeline.py — End-to-end MTF Optimization Pipeline Orchestrator.

Runs all optimization stages sequentially for any FX pair or instrument,
passing results between stages automatically via the state manager.

Stages:
  1. run_optimisation.py   — MA type + threshold sweep
  2. run_stage2.py         — Timeframe weight sweep
  3. run_pair_sweep.py     — Per-TF MA/RSI param sweep (greedy)
  4. run_stage4_atr.py     — ATR stop multiplier sweep
  5. run_portfolio.py      — Realistic portfolio simulation
  6. robustness_mtf.py     — Monte Carlo + Rolling WFO validation

Usage:
    uv run python scripts/run_mtf_pipeline.py --pair GBP_USD
    uv run python scripts/run_mtf_pipeline.py --pair EUR_USD --from-stage 3
    uv run python scripts/run_mtf_pipeline.py --pair USD_JPY --skip-robustness
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _python_cmd() -> list[str]:
    """Return the best available Python command for this project.

    Priority: uv run python > .venv/Scripts/python.exe > .venv/bin/python > sys.executable
    """
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
    """Run a pipeline stage and return True on success."""
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

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
        description="MTF Pipeline Orchestrator — runs all 6 stages for any pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  run_optimisation.py   MA type + threshold sweep
  2  run_stage2.py         Timeframe weight sweep
  3  run_pair_sweep.py     Per-TF greedy indicator sweep
  4  run_stage4_atr.py     ATR stop multiplier sweep
  5  run_portfolio.py      Realistic portfolio simulation
  6  robustness_mtf.py     Monte Carlo + Rolling WFO

Examples:
  uv run python scripts/run_mtf_pipeline.py --pair GBP_USD
  uv run python scripts/run_mtf_pipeline.py --pair GBP_USD --from-stage 3
  uv run python scripts/run_mtf_pipeline.py --pair EUR_USD --skip-robustness
        """,
    )
    parser.add_argument(
        "--pair",
        required=True,
        help="Instrument in BASE_QUOTE format, e.g. GBP_USD, EUR_USD, USD_JPY",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=range(1, 7),
        metavar="N",
        help="Resume pipeline from stage N (1-6). Stages before N are skipped.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip Stage 6 (Monte Carlo + WFO). Faster for initial sweeps.",
    )
    args = parser.parse_args()

    pair = args.pair.upper()
    pair_lower = pair.lower().replace("_", "")
    start = args.from_stage

    uv = _python_cmd()

    print("=" * 70)
    print("  MTF OPTIMIZATION PIPELINE")
    print(f"  Pair: {pair}  |  Starting from stage {start}")
    print("=" * 70)

    gate_results: dict[str, bool] = {}

    # ── Stage 1: MA type + threshold ──────────────────────────────────────────
    if start <= 1:
        ok = run_stage(
            "Stage 1: MA Type + Threshold Sweep",
            uv + ["research/mtf/run_optimisation.py", "--pair", pair],
        )
        gate_results["Stage 1"] = ok
    else:
        print(f"\n  [SKIP] Stage 1 (--from-stage {start})")

    # ── Stage 2: Timeframe weights ────────────────────────────────────────────
    if start <= 2:
        ok = run_stage(
            "Stage 2: Timeframe Weight Sweep",
            uv + ["research/mtf/run_stage2.py", "--pair", pair, "--load-state"],
        )
        gate_results["Stage 2"] = ok
    else:
        print(f"\n  [SKIP] Stage 2 (--from-stage {start})")

    # ── Stage 3: Per-TF greedy param sweep ────────────────────────────────────
    if start <= 3:
        ok = run_stage(
            "Stage 3: Per-TF Greedy Indicator Sweep",
            uv + ["research/mtf/run_pair_sweep.py", "--pair", pair, "--load-state"],
        )
        gate_results["Stage 3"] = ok
    else:
        print(f"\n  [SKIP] Stage 3 (--from-stage {start})")

    # ── Stage 4: ATR stop multiplier ─────────────────────────────────────────
    if start <= 4:
        ok = run_stage(
            "Stage 4: ATR Stop Multiplier Sweep",
            uv + ["research/mtf/run_stage4_atr.py", "--pair", pair],
        )
        gate_results["Stage 4"] = ok
    else:
        print(f"\n  [SKIP] Stage 4 (--from-stage {start})")

    # Determine config path for portfolio and robustness
    pair_cfg = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
    base_cfg = PROJECT_ROOT / "config" / "mtf.toml"
    config_path = (
        str(pair_cfg.relative_to(PROJECT_ROOT))
        if pair_cfg.exists()
        else str(base_cfg.relative_to(PROJECT_ROOT))
    )

    # ── Stage 5: Portfolio simulation ─────────────────────────────────────────
    if start <= 5:
        ok = run_stage(
            "Stage 5: Portfolio Simulation (Risk-Managed)",
            uv + ["research/mtf/run_portfolio.py", "--pair", pair, "--config", config_path],
        )
        gate_results["Stage 5"] = ok
    else:
        print(f"\n  [SKIP] Stage 5 (--from-stage {start})")

    # ── Stage 6: Robustness (Monte Carlo + WFO) ───────────────────────────────
    if start <= 6 and not args.skip_robustness:
        ok = run_stage(
            "Stage 6: Robustness Validation (Monte Carlo + WFO)",
            uv + ["scripts/robustness_mtf.py", "--pair", pair, "--config", config_path],
            abort_on_fail=False,  # Don't abort — just report
        )
        gate_results["Stage 6 (Robustness)"] = ok
    elif args.skip_robustness:
        print("\n  [SKIP] Stage 6 (--skip-robustness)")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — {pair}")
    print("=" * 70)
    for name, passed in gate_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(gate_results.values())
    if all_pass:
        print(f"\n  All stages passed. Config saved to: config/mtf_{pair_lower}.toml")
        print("  Next: review OOS metrics, then consider live deployment.")
    else:
        failed = [k for k, v in gate_results.items() if not v]
        print(f"\n  Failed stages: {', '.join(failed)}")
        print(f"  Resume with: --pair {pair} --from-stage <N>")

    print("=" * 70)


if __name__ == "__main__":
    main()
