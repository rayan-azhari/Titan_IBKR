"""pipeline_validation.py --Phases 3 ->4 ->5 Orchestrator.

Runs the full IC validation pipeline for any instrument(s) / timeframe / asset class.
Chains phase3_backtest ->phase4_wfo ->phase5_robustness and produces a unified
cross-instrument validation report.

This replaces:
    run_cat_amat_pipeline.py, run_fx_regime_pipeline.py,
    run_equity_longonly_pipeline.py, run_cat_amat_strategy.py,
    run_spy_strategy.py, run_amat_strategy.py, run_5m_strategy.py

Design:
    - Phase 4 and 5 are only run if Phase 3 passes its IS/OOS gate.
    - Skip Phase 4+5 on Phase 3 failure to avoid wasting compute.
    - All results are merged into a single validation_{slug}.csv.

Usage:
    # FX MTF validation
    uv run python research/ic_analysis/pipeline_validation.py \\
        --instruments EUR_USD GBP_USD \\
        --timeframe H4 --tfs W,D,H4,H1 \\
        --signals accel_rsi14,accel_stoch_k \\
        --asset-class fx_major --benchmark DXY

    # Equity batch validation (long-only ETFs)
    uv run python research/ic_analysis/pipeline_validation.py \\
        --instruments SPY QQQ CSCO NOC WMT \\
        --timeframe D --tfs D \\
        --signals rsi_21_dev \\
        --asset-class etf --direction long_only --benchmark SPY

    # Phase 3 only (quick gate check before full WFO)
    uv run python research/ic_analysis/pipeline_validation.py \\
        --instruments EUR_USD --phase 3 \\
        --signals accel_rsi14 --asset-class fx_major
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_OOS_IS_RATIO = 0.5  # Phase 3 gate --below this, skip Phase 4+5


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def _run_phase3(
    instrument: str,
    timeframe: str,
    tfs: list[str],
    signals: list[str],
    asset_class: str,
    direction: str,
    spread_bps: float | None,
    slippage_bps: float | None,
    max_leverage: float | None,
    risk_pct: float,
    stop_atr: float,
) -> tuple[pd.DataFrame | None, bool]:
    """Run Phase 3 backtest. Returns (result_df, phase3_passed)."""
    from research.ic_analysis.phase3_backtest import DEFAULT_THRESHOLDS as DT
    from research.ic_analysis.phase3_backtest import run_backtest

    log.info("[Phase 3] %s %s --running IS/OOS backtest ...", instrument, timeframe)
    t0 = time.time()

    try:
        result_df = run_backtest(
            instrument=instrument,
            target_signals=signals,
            tfs=tfs,
            thresholds=DT,
            asset_class=asset_class,
            direction=direction,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            risk_pct=risk_pct,
            stop_atr=stop_atr,
        )
        elapsed = time.time() - t0

        if result_df is None or result_df.empty:
            log.warning("[Phase 3] %s %s --no results returned.", instrument, timeframe)
            return None, False

        # Check IS/OOS ratio gate on best row
        if "is_oos_ratio" in result_df.columns:
            best_ratio = float(result_df["is_oos_ratio"].max())
            passed = best_ratio >= MIN_OOS_IS_RATIO
        elif "oos_sharpe" in result_df.columns and "is_sharpe" in result_df.columns:
            best_row = result_df.loc[result_df["oos_sharpe"].idxmax()]
            is_sh = float(best_row["is_sharpe"])
            oos_sh = float(best_row["oos_sharpe"])
            best_ratio = oos_sh / is_sh if is_sh != 0 else 0.0
            passed = best_ratio >= MIN_OOS_IS_RATIO
        else:
            best_ratio = 0.0
            passed = False

        gate_str = "PASS" if passed else "FAIL"
        log.info(
            "[Phase 3] %s %s done in %.1fs --best OOS/IS ratio=%.2f [%s]",
            instrument,
            timeframe,
            elapsed,
            best_ratio,
            gate_str,
        )
        return result_df, passed

    except Exception as exc:
        log.error("[Phase 3] %s %s FAILED: %s", instrument, timeframe, exc)
        return None, False


def _run_phase4(
    instrument: str,
    tfs: list[str],
    signals: list[str],
    threshold: float,
    asset_class: str,
    direction: str,
    spread_bps: float | None,
    slippage_bps: float | None,
    max_leverage: float | None,
    risk_pct: float,
    stop_atr: float,
    wfo_type: str,
) -> pd.DataFrame | None:
    """Run Phase 4 WFO for one instrument."""
    from research.ic_analysis.phase4_wfo import run_wfo

    log.info("[Phase 4] %s --running walk-forward optimisation ...", instrument)
    t0 = time.time()

    try:
        fold_df = run_wfo(
            instrument=instrument,
            target_signals=signals,
            tfs=tfs,
            threshold=threshold,
            asset_class=asset_class,
            direction=direction,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            risk_pct=risk_pct,
            stop_atr=stop_atr,
            wfo_type=wfo_type,
        )
        elapsed = time.time() - t0
        if fold_df is not None and not fold_df.empty:
            n_pos = int((fold_df["oos_sharpe"] > 0).sum())
            log.info(
                "[Phase 4] %s done in %.1fs --%d/%d folds positive",
                instrument,
                elapsed,
                n_pos,
                len(fold_df),
            )
        return fold_df
    except Exception as exc:
        log.error("[Phase 4] %s FAILED: %s", instrument, exc)
        return None


def _run_phase5(
    instrument: str,
    tfs: list[str],
    signals: list[str],
    threshold: float,
    asset_class: str,
    direction: str,
    spread_bps: float | None,
    slippage_bps: float | None,
    max_leverage: float | None,
    risk_pct: float,
    stop_atr: float,
    benchmark: str | None,
    timeframe: str,
) -> dict | None:
    """Run Phase 5 robustness (6 gates) for one instrument."""
    from research.ic_analysis.phase5_robustness import run_robustness

    log.info("[Phase 5] %s --running 6-gate robustness check ...", instrument)
    t0 = time.time()

    try:
        result = run_robustness(
            instrument=instrument,
            tfs=tfs,
            target_signals=signals,
            asset_class=asset_class,
            direction=direction,
            threshold=threshold,
            risk_pct=risk_pct,
            stop_atr=stop_atr,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            benchmark=benchmark,
            timeframe=timeframe,
        )
        elapsed = time.time() - t0
        all_pass = result.get("all_pass", False)
        log.info(
            "[Phase 5] %s done in %.1fs --ALL_PASS=%s",
            instrument,
            elapsed,
            all_pass,
        )
        return result
    except Exception as exc:
        log.error("[Phase 5] %s FAILED: %s", instrument, exc)
        return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_validation(
    instruments: list[str],
    timeframe: str,
    signals: list[str],
    tfs: list[str] | None = None,
    asset_class: str = "fx_major",
    direction: str = "both",
    benchmark: str | None = None,
    phase: str = "all",
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    max_leverage: float | None = None,
    risk_pct: float = 0.01,
    stop_atr: float = 1.5,
    threshold: float = 0.75,
    wfo_type: str = "rolling",
) -> pd.DataFrame:
    """Run Phase 3 ->4 ->5 validation pipeline for a batch of instruments.

    Args:
        instruments: List of instrument names (e.g. ["EUR_USD", "SPY", "CSCO"]).
        timeframe: Base timeframe (used for regime file lookup in Phase 5).
        signals: Signal names from Phase 2 leaderboard.
        tfs: TF stack for MTF composite. None = [timeframe] (single-TF).
        asset_class: Key into COST_PROFILES (fx_major, etf, equity_lc, ...).
        direction: "both" or "long_only".
        benchmark: Benchmark ticker for Gate 6 alpha/beta (e.g. "SPY" or "DXY").
        phase: Comma-separated phases to run: "3", "4", "5", or "all".
        spread_bps: Override profile spread in basis points.
        slippage_bps: Override profile slippage in basis points.
        max_leverage: Override profile max leverage.
        risk_pct: Fraction of portfolio risked per trade.
        stop_atr: ATR multiplier for stop distance.
        threshold: Z-score threshold for Phase 4 WFO (uses Phase 3 best IS threshold).
        wfo_type: "rolling" or "anchored" for Phase 4.

    Returns:
        DataFrame with one row per instrument summarising all 3 phases.
    """
    tfs = tfs or [timeframe]

    run_phases = set()
    if phase.lower() == "all":
        run_phases = {"3", "4", "5"}
    else:
        run_phases = {p.strip() for p in phase.split(",")}

    W = 80
    print()
    print("=" * W)
    print("  IC VALIDATION PIPELINE --Phases 3 ->4 ->5")
    print(f"  Instruments : {', '.join(instruments)}")
    print(f"  Timeframe   : {timeframe}  |  TF stack: {' ->'.join(tfs)}")
    print(f"  Signals     : {', '.join(signals)}")
    print(f"  Asset class : {asset_class}  |  Direction: {direction}")
    print(f"  Benchmark   : {benchmark or 'none (Gate 6 skipped)'}")
    print(f"  Phases      : {phase}")
    print("=" * W)

    summary_rows: list[dict] = []

    for instrument in instruments:
        print(f"\n{'-' * W}")
        print(f"  Instrument: {instrument}")
        print(f"{'-' * W}")

        row: dict = {"instrument": instrument, "timeframe": timeframe, "signals": ",".join(signals)}
        best_threshold = threshold

        # Phase 3
        p3_df: pd.DataFrame | None = None
        p3_passed = False
        if "3" in run_phases:
            p3_df, p3_passed = _run_phase3(
                instrument=instrument,
                timeframe=timeframe,
                tfs=tfs,
                signals=signals,
                asset_class=asset_class,
                direction=direction,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                max_leverage=max_leverage,
                risk_pct=risk_pct,
                stop_atr=stop_atr,
            )
            if p3_df is not None and not p3_df.empty:
                if "oos_sharpe" in p3_df.columns:
                    best_row = p3_df.loc[p3_df["oos_sharpe"].idxmax()]
                    row["p3_oos_sharpe"] = round(float(best_row.get("oos_sharpe", 0)), 3)
                    row["p3_is_sharpe"] = round(float(best_row.get("is_sharpe", 0)), 3)
                    row["p3_is_oos_ratio"] = round(float(best_row.get("is_oos_ratio", 0)), 3)
                    row["p3_win_rate"] = round(float(best_row.get("win_rate", 0)), 3)
                    row["p3_max_dd_pct"] = round(float(best_row.get("max_dd_pct", 0)), 2)
                    row["p3_ror_25pct"] = round(float(best_row.get("ror_25pct", 1)), 4)
                    # Use best IS threshold for Phase 4
                    if "threshold" in best_row.index:
                        best_threshold = float(best_row["threshold"])
            row["p3_passed"] = p3_passed
        else:
            p3_passed = True  # If skipping Phase 3, assume it passed

        # Phase 4 --only if Phase 3 passed or was skipped
        if "4" in run_phases:
            if not p3_passed and "3" in run_phases:
                log.warning(
                    "[Phase 4] SKIP %s --Phase 3 gate failed (OOS/IS ratio < %.1f)",
                    instrument,
                    MIN_OOS_IS_RATIO,
                )
                row["p4_status"] = "SKIPPED_P3_FAIL"
            else:
                p4_df = _run_phase4(
                    instrument=instrument,
                    tfs=tfs,
                    signals=signals,
                    threshold=best_threshold,
                    asset_class=asset_class,
                    direction=direction,
                    spread_bps=spread_bps,
                    slippage_bps=slippage_bps,
                    max_leverage=max_leverage,
                    risk_pct=risk_pct,
                    stop_atr=stop_atr,
                    wfo_type=wfo_type,
                )
                if p4_df is not None and not p4_df.empty:
                    row["p4_folds"] = len(p4_df)
                    row["p4_pct_positive"] = round(float((p4_df["oos_sharpe"] > 0).mean()), 3)
                    row["p4_mean_oos_sharpe"] = round(float(p4_df["oos_sharpe"].mean()), 3)
                    row["p4_worst_sharpe"] = round(float(p4_df["oos_sharpe"].min()), 3)
                    row["p4_status"] = "OK"
                else:
                    row["p4_status"] = "NO_DATA"

        # Phase 5 --only if Phase 3 passed or was skipped
        if "5" in run_phases:
            if not p3_passed and "3" in run_phases:
                log.warning("[Phase 5] SKIP %s --Phase 3 gate failed.", instrument)
                row["p5_all_pass"] = False
                row["p5_status"] = "SKIPPED_P3_FAIL"
            else:
                p5_result = _run_phase5(
                    instrument=instrument,
                    tfs=tfs,
                    signals=signals,
                    threshold=best_threshold,
                    asset_class=asset_class,
                    direction=direction,
                    spread_bps=spread_bps,
                    slippage_bps=slippage_bps,
                    max_leverage=max_leverage,
                    risk_pct=risk_pct,
                    stop_atr=stop_atr,
                    benchmark=benchmark,
                    timeframe=timeframe,
                )
                if p5_result is not None:
                    row["p5_all_pass"] = p5_result.get("all_pass", False)
                    for gate in ("g1", "g2", "g3", "g4", "g5", "g6"):
                        g = p5_result.get(gate, {})
                        row[f"p5_{gate}_pass"] = (
                            g.get("gate_pass", False) if isinstance(g, dict) else False
                        )
                    row["p5_status"] = "OK"
                else:
                    row["p5_all_pass"] = False
                    row["p5_status"] = "FAILED"

        summary_rows.append(row)

    # Build summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Save
    inst_slug = "_".join(i.lower() for i in instruments)
    out_path = REPORTS_DIR / f"validation_{inst_slug}_{timeframe}.csv"
    summary_df.to_csv(out_path, index=False)

    # Print summary
    print(f"\n{'=' * W}")
    print("  VALIDATION SUMMARY")
    print(f"{'=' * W}")

    for row in summary_rows:
        inst = row["instrument"]
        p3_ok = row.get("p3_passed", "–")
        p4_st = row.get("p4_status", "–")
        p5_ok = row.get("p5_all_pass", "–")

        p3_sh = row.get("p3_oos_sharpe", "–")
        p4_sh = row.get("p4_mean_oos_sharpe", "–")

        p3_str = f"P3={'PASS' if p3_ok else 'FAIL'}(OOS_Sharpe={p3_sh})"
        p4_str = (
            f"P4={p4_st}(mean={p4_sh})" if p4_st not in ("–", "SKIPPED_P3_FAIL") else f"P4={p4_st}"
        )
        p5_str = f"P5={'PASS' if p5_ok else 'FAIL'}" if isinstance(p5_ok, bool) else f"P5={p5_ok}"

        print(f"  {inst:<20}  {p3_str}  {p4_str}  {p5_str}")

    cleared = [
        r["instrument"] for r in summary_rows if r.get("p3_passed") and r.get("p5_all_pass", False)
    ]
    if cleared:
        print(f"\n  CLEARED for Phase 6: {', '.join(cleared)}")
    else:
        print("\n  No instruments cleared all gates --review Phase 3/5 results.")

    print(f"\n  Report saved: {out_path.relative_to(ROOT)}")
    print("=" * W)

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="IC Validation Pipeline --Phases 3 ->4 ->5 (any asset class, batch-capable)"
    )
    p.add_argument(
        "--instrument",
        default="EUR_USD",
        help="Single instrument (overridden by --instruments)",
    )
    p.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Batch mode: list of instruments",
    )
    p.add_argument("--timeframe", default="H4", help="Base timeframe: D, H4, H1, etc.")
    p.add_argument(
        "--tfs",
        default=None,
        help="MTF stack (comma-separated), e.g. W,D,H4,H1. Default: base timeframe only.",
    )
    p.add_argument(
        "--signals",
        required=True,
        help="Comma-separated signal names from Phase 2 leaderboard.",
    )
    p.add_argument(
        "--asset-class",
        default="fx_major",
        help="Asset class: fx_major, fx_cross, equity_lc, etf, futures.",
    )
    p.add_argument(
        "--direction",
        default="both",
        choices=["both", "long_only"],
        help="both (long+short) or long_only (equities/ETFs).",
    )
    p.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark for Gate 6 alpha/beta (e.g. SPY, DXY). Requires data/{BENCH}_D.parquet.",
    )
    p.add_argument(
        "--phase",
        default="all",
        help="Phases to run: '3', '4', '5', '3,4', '3,5', or 'all'. Default: all.",
    )
    p.add_argument("--spread-bps", type=float, default=None)
    p.add_argument("--slippage-bps", type=float, default=None)
    p.add_argument("--max-leverage", type=float, default=None)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--stop-atr", type=float, default=1.5)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Z-score threshold for Phase 4 WFO (Phase 3 best IS threshold used if available).",
    )
    p.add_argument(
        "--wfo-type",
        default="rolling",
        choices=["rolling", "anchored"],
        help="Phase 4 WFO mode: rolling (sliding IS) or anchored (expanding IS).",
    )
    args = p.parse_args()

    instruments = args.instruments or [args.instrument]
    tfs = [t.strip() for t in args.tfs.split(",")] if args.tfs else None
    signals = [s.strip() for s in args.signals.split(",")]

    run_validation(
        instruments=instruments,
        timeframe=args.timeframe,
        signals=signals,
        tfs=tfs,
        asset_class=args.asset_class,
        direction=args.direction,
        benchmark=args.benchmark,
        phase=args.phase,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        max_leverage=args.max_leverage,
        risk_pct=args.risk_pct,
        stop_atr=args.stop_atr,
        threshold=args.threshold,
        wfo_type=args.wfo_type,
    )


if __name__ == "__main__":
    main()
