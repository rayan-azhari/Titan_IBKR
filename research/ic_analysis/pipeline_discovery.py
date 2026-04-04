"""pipeline_discovery.py --Phases 0 ->1 ->2 Orchestrator.

Runs the full IC discovery pipeline for any instrument(s) / timeframe / asset class.
Chains phase0_regime ->phase1_sweep ->phase2_combination and produces a unified
cross-instrument leaderboard.

This replaces run_mtf_ic_pipeline.py and run_equity_phase2.py.

Usage:
    # FX --MTF stack discovery
    uv run python research/ic_analysis/pipeline_discovery.py \\
        --instruments EUR_USD GBP_USD --timeframe H4 --tfs W,D,H4,H1

    # Equity --daily single-TF discovery, batch
    uv run python research/ic_analysis/pipeline_discovery.py \\
        --instruments SPY QQQ CSCO NOC WMT ABNB --timeframe D

    # Phase 1 only (skip Phase 2 combination)
    uv run python research/ic_analysis/pipeline_discovery.py \\
        --instruments EUR_USD --timeframe H4 --phase 0,1

    # With fractional differencing
    uv run python research/ic_analysis/pipeline_discovery.py \\
        --instruments EUR_USD --timeframe H4 --frac-diff
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
REGIME_DIR = ROOT / ".tmp" / "regime"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REGIME_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
DEFAULT_TIMEFRAME = "H4"
DEFAULT_TFS = ["W", "D", "H4", "H1"]


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def _run_phase0(
    instrument: str,
    timeframe: str,
    frac_diff: bool = False,
    hmm_states: int = 2,
) -> pd.DataFrame | None:
    """Run Phase 0 regime labelling for one instrument."""
    from research.ic_analysis.phase0_regime import _load_ohlcv, compute_regime_labels

    out_path = REGIME_DIR / f"{instrument}_{timeframe}_regime.parquet"
    log.info("[Phase 0] %s %s --computing regime labels ...", instrument, timeframe)
    t0 = time.time()

    try:
        df = _load_ohlcv(instrument, timeframe)
    except FileNotFoundError as exc:
        log.warning("[Phase 0] SKIP %s %s: %s", instrument, timeframe, exc)
        return None

    try:
        regime_df = compute_regime_labels(df, hmm_n_states=hmm_states, frac_diff=frac_diff)
        regime_df.to_parquet(out_path)
        elapsed = time.time() - t0
        log.info(
            "[Phase 0] %s %s done in %.1fs --saved %s",
            instrument,
            timeframe,
            elapsed,
            out_path.relative_to(ROOT),
        )
        return regime_df
    except Exception as exc:
        log.error("[Phase 0] %s %s FAILED: %s", instrument, timeframe, exc)
        return None


def _run_phase1(
    instrument: str,
    timeframe: str,
    horizons: list[int],
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Run Phase 1 IC sweep for one instrument."""
    from research.ic_analysis.phase1_sweep import run_sweep

    log.info("[Phase 1] %s %s --running 52-signal IC sweep ...", instrument, timeframe)
    t0 = time.time()

    try:
        result_df = run_sweep(
            instrument=instrument,
            timeframe=timeframe,
            horizons=horizons,
            regime_df=regime_df,
        )
        elapsed = time.time() - t0
        if result_df is not None and not result_df.empty:
            n_strong = int((result_df["verdict"].str.contains("STRONG")).sum())
            n_usable = int((result_df["verdict"].str.contains("USABLE")).sum())
            log.info(
                "[Phase 1] %s %s done in %.1fs --%d STRONG, %d USABLE",
                instrument,
                timeframe,
                elapsed,
                n_strong,
                n_usable,
            )
        return result_df
    except Exception as exc:
        log.error("[Phase 1] %s %s FAILED: %s", instrument, timeframe, exc)
        return None


def _run_phase2(
    instrument: str,
    timeframe: str,
    horizon: int,
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Run Phase 2 signal combination for one instrument + horizon."""
    from research.ic_analysis.phase2_combination import run_combination

    log.info(
        "[Phase 2] %s %s h=%d --running combination methods ...",
        instrument,
        timeframe,
        horizon,
    )
    t0 = time.time()

    try:
        result_df = run_combination(
            instrument=instrument,
            timeframe=timeframe,
            horizon=horizon,
            regime_df=regime_df,
        )
        elapsed = time.time() - t0
        if result_df is not None and not result_df.empty:
            log.info(
                "[Phase 2] %s %s h=%d done in %.1fs --%d combinations",
                instrument,
                timeframe,
                horizon,
                elapsed,
                len(result_df),
            )
        return result_df
    except Exception as exc:
        log.error("[Phase 2] %s %s h=%d FAILED: %s", instrument, timeframe, horizon, exc)
        return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_discovery(
    instruments: list[str],
    timeframe: str = DEFAULT_TIMEFRAME,
    tfs: list[str] | None = None,
    horizons: list[int] | None = None,
    phase: str = "all",
    frac_diff: bool = False,
    hmm_states: int = 2,
    combo_horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Run Phase 0 ->1 ->2 discovery pipeline for a batch of instruments.

    Args:
        instruments: List of instrument names (e.g. ["EUR_USD", "SPY", "CSCO"]).
        timeframe: Base timeframe for Phase 0 regime labels and Phase 1 IC sweep.
        tfs: MTF stack for Phase 1 stacking. None = base timeframe only.
        horizons: Forecast horizons for Phase 1. Default: [1, 5, 10, 20, 60].
        phase: Comma-separated list of phases to run, e.g. "0,1" or "all".
        frac_diff: Enable fractional differencing in Phase 0.
        hmm_states: Number of HMM states in Phase 0.
        combo_horizons: Horizons for Phase 2 combination. Default: [horizons[0]].

    Returns:
        Unified Phase 1 leaderboard DataFrame with instrument column.
    """
    horizons = horizons or DEFAULT_HORIZONS
    tfs = tfs or [timeframe]
    combo_horizons = combo_horizons or [horizons[0]]

    run_phases = set()
    if phase.lower() == "all":
        run_phases = {"0", "1", "1.5", "2"}
    else:
        run_phases = {p.strip() for p in phase.split(",")}

    W = 80
    print()
    print("=" * W)
    print("  IC DISCOVERY PIPELINE -- Phases 0 -> 1 -> 2")
    print(f"  Instruments : {', '.join(instruments)}")
    print(f"  Timeframe   : {timeframe}")
    print(f"  TF stack    : {' -> '.join(tfs)}")
    print(f"  Horizons    : {horizons}")
    print(f"  Phases      : {phase}")
    print("=" * W)

    all_phase1_results: list[pd.DataFrame] = []

    for instrument in instruments:
        print(f"\n{'-' * W}")
        print(f"  Instrument: {instrument}")
        print(f"{'-' * W}")

        # Phase 0
        regime_df: pd.DataFrame | None = None
        if "0" in run_phases:
            regime_df = _run_phase0(instrument, timeframe, frac_diff, hmm_states)
        else:
            # Try to load existing regime file
            regime_path = REGIME_DIR / f"{instrument}_{timeframe}_regime.parquet"
            if regime_path.exists():
                try:
                    regime_df = pd.read_parquet(regime_path)
                    log.info(
                        "[Phase 0] Loaded existing regime labels for %s %s", instrument, timeframe
                    )
                except Exception:
                    pass

        # Phase 1
        if "1" in run_phases:
            p1_df = _run_phase1(instrument, timeframe, horizons, regime_df)
            if p1_df is not None and not p1_df.empty:
                p1_df.insert(0, "instrument", instrument)
                all_phase1_results.append(p1_df)
        else:
            # Try to load existing Phase 1 CSV for Phase 2 use
            p1_path = REPORTS_DIR / f"phase1_{instrument}_{timeframe}.csv"
            if p1_path.exists():
                p1_df = pd.read_csv(p1_path)
                p1_df.insert(0, "instrument", instrument)
                all_phase1_results.append(p1_df)

        # Phase 2
        if "2" in run_phases:
            for h in combo_horizons:
                _run_phase2(instrument, timeframe, h, regime_df)

    # Build unified leaderboard
    if not all_phase1_results:
        print("\n  [WARN] No Phase 1 results to summarise.")
        return pd.DataFrame()

    leaderboard = pd.concat(all_phase1_results, ignore_index=True)

    # Slug for output filename
    inst_slug = "_".join(i.lower() for i in instruments)
    out_path = REPORTS_DIR / f"discovery_{inst_slug}_{timeframe}.csv"
    leaderboard.to_csv(out_path, index=False)

    # Print top-10 across all instruments
    print(f"\n{'=' * W}")
    print("  DISCOVERY LEADERBOARD --top 10 signals (all instruments)")
    print(f"{'=' * W}")

    display_cols = ["instrument", "signal", "horizon", "ic", "icir", "verdict"]
    display_cols = [c for c in display_cols if c in leaderboard.columns]

    if "ic" in leaderboard.columns:
        top10 = leaderboard.reindex(columns=display_cols).nlargest(10, "ic")
        strong_total = int((leaderboard["verdict"].str.contains("STRONG")).sum())
        usable_total = int((leaderboard["verdict"].str.contains("USABLE")).sum())
        print(top10.to_string(index=False))
        print(f"\n  STRONG signals: {strong_total}  |  USABLE signals: {usable_total}")
    else:
        print(leaderboard.reindex(columns=display_cols).head(10).to_string(index=False))

    print(f"\n  Unified leaderboard saved: {out_path.relative_to(ROOT)}")
    print("=" * W)

    return leaderboard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="IC Discovery Pipeline --Phases 0 ->1 ->2 (any instrument, batch-capable)"
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
        help="Batch mode: list of instruments, e.g. EUR_USD GBP_USD SPY",
    )
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="Base timeframe: D, H4, H1, etc.")
    p.add_argument(
        "--tfs",
        default=None,
        help="MTF stack (comma-separated), e.g. W,D,H4,H1. Default: base timeframe only.",
    )
    p.add_argument(
        "--horizons",
        default=None,
        help="Forecast horizons (comma-separated), e.g. 1,5,10,20. Default: 1,5,10,20,60.",
    )
    p.add_argument(
        "--combo-horizons",
        default=None,
        help="Horizons for Phase 2 combination (comma-separated). Default: first horizon.",
    )
    p.add_argument(
        "--phase",
        default="all",
        help="Phases to run: '0', '1', '2', '0,1', '1,2', or 'all'. Default: all.",
    )
    p.add_argument(
        "--frac-diff", action="store_true", help="Enable fractional differencing in Phase 0."
    )
    p.add_argument("--hmm-states", type=int, default=2, help="Number of HMM states (default: 2).")
    args = p.parse_args()

    instruments = args.instruments or [args.instrument]

    tfs = [t.strip() for t in args.tfs.split(",")] if args.tfs else None
    horizons = [int(h.strip()) for h in args.horizons.split(",")] if args.horizons else None
    combo_horizons = (
        [int(h.strip()) for h in args.combo_horizons.split(",")] if args.combo_horizons else None
    )

    run_discovery(
        instruments=instruments,
        timeframe=args.timeframe,
        tfs=tfs,
        horizons=horizons,
        phase=args.phase,
        frac_diff=args.frac_diff,
        hmm_states=args.hmm_states,
        combo_horizons=combo_horizons,
    )


if __name__ == "__main__":
    main()
