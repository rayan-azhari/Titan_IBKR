"""run_mtf_ic_pipeline.py -- IC-Based MTF Signal Discovery Pipeline.

Pipeline position: orchestrates Phase 1 → 1.5 → 2 in sequence.
Replaces the Stage 1-4 MA+RSI parameter sweep (research/mtf/) as the primary
MTF signal discovery mechanism. Signal identity is data-driven, not assumed.

Pipeline
--------
Phase 1   run_signal_sweep (52 signals on H1)
            → IC/ICIR/AR1 leaderboard — which signals have edge at all

Phase 1.5 run_mtf_stack (52 signals × W/D/H4/H1)
            → MTF stacking: equal-weight, ICIR-weighted, confluence gate
            → vs_h1_ic: which signals gain predictive power from MTF confluence

Phase 2   run_signal_combination (6 combo methods on H1 USABLE signals)
            → confirms whether MTF composites beat individual signals

Summary   Merge results into a single ranked CSV + console top-20 table

Anti-bias guarantees (inherited from underlying scripts)
---------------------------------------------------------
- Coarser TF look-ahead: .shift(1) before ffill in run_mtf_stack._align_to_base()
- Forward returns: close.shift(-h) only inside compute_forward_returns()
- Horizons: [1, 5, 10, 20, 60] uniform across Phase 1/1.5/2
- No IS/OOS here: exploratory IC only; IS/OOS enforced at Phase 3 (run_ic_backtest.py)
- Signal computation: .rolling()/.ewm()/.shift(+n) only — no negatives anywhere

Usage
-----
    uv run python research/ic_analysis/run_mtf_ic_pipeline.py
    uv run python research/ic_analysis/run_mtf_ic_pipeline.py --instrument GBP_USD
    uv run python research/ic_analysis/run_mtf_ic_pipeline.py --instrument EUR_USD --phase 1.5
    uv run python research/ic_analysis/run_mtf_ic_pipeline.py --instrument EUR_USD --tfs D,H4,H1
    uv run python research/ic_analysis/run_mtf_ic_pipeline.py --instrument EUR_USD --write-config
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.run_signal_combination import run_combination  # noqa: E402
from research.ic_analysis.run_signal_sweep import HORIZONS, run_sweep  # noqa: E402

from research.ic_analysis.run_mtf_stack import run_mtf_stack  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = ROOT / ".tmp" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TFS = ["W", "D", "H4", "H1"]
BASE_TF = "H1"
COMBO_HORIZON = 20  # h=20 for Phase 2 combination (standard per IC Signal Analysis.md)

# Verdict classification — matches run_signal_sweep.py
_STRONG_GATE = (0.05, 0.5)  # |IC|, |ICIR|
_USABLE_GATE = (0.05, 0.0)
_WEAK_GATE = (0.03, 0.0)


def _verdict(ic: float, icir: float) -> str:
    abs_ic = abs(ic) if not np.isnan(ic) else 0.0
    abs_ir = abs(icir) if not np.isnan(icir) else 0.0
    if abs_ic >= _STRONG_GATE[0] and abs_ir >= _STRONG_GATE[1]:
        return "STRONG"
    elif abs_ic >= _USABLE_GATE[0]:
        return "USABLE"
    elif abs_ic >= _WEAK_GATE[0]:
        return "WEAK"
    return "NOISE"


# ── Phase 1 helpers ────────────────────────────────────────────────────────────


def _run_phase1(instrument: str) -> pd.DataFrame:
    """Run Phase 1 sweep on H1 and return the leaderboard DataFrame."""
    logger.info("=== Phase 1: Signal sweep on %s H1 ===", instrument)
    run_sweep(instrument, BASE_TF, horizons=HORIZONS)

    # run_sweep returns None; load the CSV it saved
    slug = f"{instrument}_{BASE_TF}".lower()
    csv_path = REPORT_DIR / f"ic_sweep_{slug}.csv"
    if not csv_path.exists():
        logger.warning("Phase 1 CSV not found at %s — skipping Phase 1 in summary", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Normalise to summary schema
    df = df.rename(columns={"ic": "best_ic", "icir": "best_icir"})
    df["phase"] = "1"
    df["method"] = "single_H1"
    df["vs_h1_ic"] = 0.0  # this IS the H1 baseline
    return df[
        ["phase", "signal", "method", "best_ic", "best_h", "best_icir", "verdict", "vs_h1_ic"]
    ]


# ── Phase 1.5 helpers ──────────────────────────────────────────────────────────


def _run_phase15(instrument: str, tfs: list[str]) -> pd.DataFrame:
    """Run Phase 1.5 MTF stacking and return the results DataFrame."""
    logger.info("=== Phase 1.5: MTF stacking %s %s ===", instrument, " -> ".join(tfs))
    df = run_mtf_stack(instrument=instrument, tfs=tfs, horizons=HORIZONS)
    if df.empty:
        return pd.DataFrame()

    df["phase"] = "1.5"
    df = df.rename(columns={"best_icir": "best_icir"})
    # best_h is an int in the DataFrame — normalise to match Phase 1 string format
    df["best_h"] = df["best_h"].apply(
        lambda h: f"h={int(h)}" if not (isinstance(h, float) and np.isnan(h)) else "NaN"
    )
    return df[
        ["phase", "signal", "method", "best_ic", "best_h", "best_icir", "verdict", "vs_h1_ic"]
    ]


# ── Phase 2 helpers ────────────────────────────────────────────────────────────


def _run_phase2(instrument: str) -> pd.DataFrame:
    """Run Phase 2 combination methods and return the results DataFrame."""
    logger.info("=== Phase 2: Signal combination on %s H1 h=%d ===", instrument, COMBO_HORIZON)
    df = run_combination(instrument, BASE_TF, horizon=COMBO_HORIZON)
    if df.empty:
        return pd.DataFrame()

    # Phase 2 reports per combination-method, not per signal — map to summary schema
    df = df.rename(
        columns={
            "method": "signal",  # combination name as the "signal" for display
            "vs_best_ind": "vs_h1_ic",
        }
    )
    df["phase"] = "2"
    df["method"] = "combination"
    df["best_h"] = f"h={COMBO_HORIZON}"
    df["best_icir"] = df["icir"]
    df = df.rename(columns={"ic": "best_ic"})
    return df[
        ["phase", "signal", "method", "best_ic", "best_h", "best_icir", "verdict", "vs_h1_ic"]
    ]


# ── Summary display ────────────────────────────────────────────────────────────


def _print_summary(summary: pd.DataFrame, instrument: str, top_n: int = 20) -> None:
    W = 90
    print("\n" + "=" * W)
    print(f"  MTF IC PIPELINE SUMMARY -- {instrument}")
    print("  Phases: 1 (H1 baseline) | 1.5 (MTF stack) | 2 (combination)")
    print(f"  Horizons: {HORIZONS}  |  Combo horizon: h={COMBO_HORIZON}")
    print("=" * W)

    ranked = (
        summary.assign(abs_ic=summary["best_ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(f"\n  TOP {top_n} ACROSS ALL PHASES (ranked by |IC|):")
    print("  " + "-" * (W - 2))
    print(
        f"  {'Rank':>4}  {'Signal/Method':<26}  {'Ph':>4}  {'Stack':<18}  "
        f"{'IC':>8}  {'ICIR':>7}  {'vs_H1':>7}  Verdict"
    )
    print("  " + "-" * (W - 2))

    for i, row in ranked.iterrows():
        ic_s = f"{row['best_ic']:>+8.4f}" if not np.isnan(row["best_ic"]) else "     NaN"
        ir_s = f"{row['best_icir']:>+7.3f}" if not np.isnan(row["best_icir"]) else "    NaN"
        vs_s = f"{row['vs_h1_ic']:>+7.4f}" if not np.isnan(row["vs_h1_ic"]) else "    NaN"
        label = str(row["signal"])[:26]
        meth = str(row["method"])[:18]
        print(
            f"  {i + 1:>4}  {label:<26}  {row['phase']:>4}  {meth:<18}  "
            f"{ic_s}  {ir_s}  {vs_s}  {row['verdict']}"
        )

    print("  " + "-" * (W - 2))

    # Verdict distribution per phase
    print()
    print("  VERDICT DISTRIBUTION BY PHASE:")
    for phase_id in ["1", "1.5", "2"]:
        sub = summary[summary["phase"] == phase_id]
        if sub.empty:
            continue
        counts = sub["verdict"].value_counts()
        parts = "  ".join(f"{v}={counts.get(v, 0)}" for v in ["STRONG", "USABLE", "WEAK", "NOISE"])
        print(f"  Phase {phase_id}: {parts}")

    print("\n" + "=" * W)


# ── Config write-back ──────────────────────────────────────────────────────────


def _write_config(instrument: str, phase15_df: pd.DataFrame) -> None:
    """Update config/ic_mtf.toml threshold for `instrument` using best Phase 1.5 signal.

    Only updates `threshold`. Never touches risk_pct / stop_atr_mult / warmup_bars
    (those are validated through Phase 3/4 and must not be overwritten without backtesting).
    """
    import tomllib

    config_path = ROOT / "config" / "ic_mtf.toml"
    if not config_path.exists():
        logger.warning("Config not found: %s — skipping write-back", config_path)
        return

    # Best STRONG signal from Phase 1.5
    strong = phase15_df[phase15_df["verdict"] == "STRONG"]
    if strong.empty:
        logger.info("No STRONG signals in Phase 1.5 — config not updated")
        return

    best = (
        strong.assign(abs_ic=strong["best_ic"].abs()).sort_values("abs_ic", ascending=False).iloc[0]
    )
    best_signal: str = str(best["signal"])
    best_ic: float = float(best["best_ic"])

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    section = instrument.upper()
    if section not in config:
        logger.warning("Section [%s] not found in ic_mtf.toml — skipping", section)
        return

    current_threshold = config[section].get("threshold", None)

    # Only update threshold — derive a simple rule: threshold ≈ 1.0 / |IC| * 0.05 (heuristic)
    # Cap between 0.50 and 1.50; default 0.75 if IC is unusable
    if abs(best_ic) >= 0.05:
        new_threshold = round(max(0.50, min(1.50, 0.05 / abs(best_ic))), 2)
    else:
        logger.info("Phase 1.5 best IC %.4f below 0.05 — config not updated", best_ic)
        return

    if new_threshold == current_threshold:
        logger.info(
            "Config threshold for [%s] already %.2f — no change needed", section, current_threshold
        )
        return

    # Rewrite the line in the TOML file (simple regex replacement to preserve comments)
    import re

    text = config_path.read_text(encoding="utf-8")
    pattern = rf"(\[{re.escape(section)}\][^\[]*?threshold\s*=\s*)[0-9.]+(\s)"
    replacement = rf"\g<1>{new_threshold}\g<2>"
    new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    if new_text == text:
        logger.warning("Could not find threshold line for [%s] in TOML — skipping", section)
        return

    config_path.write_text(new_text, encoding="utf-8")
    logger.info(
        "Updated config/ic_mtf.toml [%s] threshold: %.2f -> %.2f  (best signal: %s, IC=%.4f)",
        section,
        current_threshold,
        new_threshold,
        best_signal,
        best_ic,
    )


# ── Orchestrator ───────────────────────────────────────────────────────────────


def run_mtf_ic_pipeline(
    instrument: str = "EUR_USD",
    tfs: list[str] | None = None,
    horizons: list[int] | None = None,
    phase: str = "all",
    write_config: bool = False,
) -> pd.DataFrame:
    """Run the full IC-based MTF signal discovery pipeline.

    Args:
        instrument:   Instrument slug matching data/{instrument}_{TF}.parquet
        tfs:          Timeframes coarse-to-fine (default: W, D, H4, H1)
        horizons:     Forward return horizons in base-TF bars (default: [1,5,10,20,60])
        phase:        Which phases to run: "1", "1.5", "2", or "all"
        write_config: If True, update config/ic_mtf.toml threshold from best Phase 1.5 signal

    Returns:
        Merged summary DataFrame (all phases combined, ranked by |IC|).
    """
    if tfs is None:
        tfs = DEFAULT_TFS
    if horizons is None:
        horizons = HORIZONS

    W = 74
    print("\n" + "=" * W)
    print(f"  MTF IC DISCOVERY PIPELINE — {instrument}")
    print(f"  TFs: {' -> '.join(tfs)}  |  Horizons: {horizons}")
    print(f"  Phases: {phase}  |  write_config: {write_config}")
    print("=" * W)

    frames: list[pd.DataFrame] = []
    phase15_df: pd.DataFrame = pd.DataFrame()

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    if phase in ("all", "1"):
        df1 = _run_phase1(instrument)
        if not df1.empty:
            frames.append(df1)

    # ── Phase 1.5 ──────────────────────────────────────────────────────────────
    if phase in ("all", "1.5"):
        phase15_df = _run_phase15(instrument, tfs)
        if not phase15_df.empty:
            frames.append(phase15_df)

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    if phase in ("all", "2"):
        df2 = _run_phase2(instrument)
        if not df2.empty:
            frames.append(df2)

    if not frames:
        logger.warning("No results produced — check data availability.")
        return pd.DataFrame()

    summary = pd.concat(frames, ignore_index=True)

    # Print unified leaderboard
    _print_summary(summary, instrument)

    # Save merged summary
    slug = instrument.lower()
    out_path = REPORT_DIR / f"mtf_ic_pipeline_{slug}.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Summary saved: %s", out_path)

    # Config write-back (optional, Phase 1.5 must have run)
    if write_config:
        if phase15_df.empty:
            logger.warning("--write-config requires Phase 1.5 to run; skipping")
        else:
            _write_config(instrument, phase15_df)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC-based MTF signal discovery pipeline (Phase 1 → 1.5 → 2)"
    )
    parser.add_argument(
        "--instrument",
        default="EUR_USD",
        help="Instrument slug, e.g. EUR_USD, GBP_USD (default: EUR_USD)",
    )
    parser.add_argument(
        "--tfs",
        default=",".join(DEFAULT_TFS),
        help=f"Comma-separated TFs coarse-to-fine (default: {','.join(DEFAULT_TFS)})",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help=f"Comma-separated forward horizons in H1 bars (default: {','.join(str(h) for h in HORIZONS)})",
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "1", "1.5", "2"],
        help="Which phases to run (default: all)",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Update config/ic_mtf.toml threshold from best Phase 1.5 STRONG signal",
    )
    args = parser.parse_args()

    tfs = [t.strip() for t in args.tfs.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]

    run_mtf_ic_pipeline(
        instrument=args.instrument,
        tfs=tfs,
        horizons=horizons,
        phase=args.phase,
        write_config=args.write_config,
    )


if __name__ == "__main__":
    main()
