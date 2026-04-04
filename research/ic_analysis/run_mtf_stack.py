"""run_mtf_stack.py -- Multi-Timeframe Signal Stacking IC Analysis.

Pipeline position: Phase 1.5 (between run_signal_sweep.py and run_signal_combination.py).
Feeds the signal identity and sign orientation required by Phase 2 (run_signal_combination.py).

For each of the 52 signals, computes the signal on W, D, H4, and H1 bars
separately, aligns all to H1, then tests whether stacking (combining) signals
across timeframes produces a composite with higher IC/ICIR than H1 alone.

Hypothesis: when the same indicator agrees across W, D, H4, H1 simultaneously,
the composite is more reliable than the H1 signal in isolation. This is the
"MTF confluence" principle, validated empirically with IC statistics.

Stacking methods:
    1. Equal-weight     -- mean of all TF variants (sign-normalised)
    2. ICIR-weighted    -- weight each TF by its individual ICIR at H1
    3. Confluence gate  -- signal proportional to TF agreement fraction
                          (all 4 agree = 1.0; split 2-2 = 0.0; etc.)

Alignment:
    Coarser TF signals are forward-filled to the H1 bar index. A weekly
    signal is constant within each week (~168 H1 bars); it only changes
    when the new weekly bar closes. This is causal -- no look-ahead.

    Per the look-ahead safety rules in directives/IC Signal Analysis.md:
    coarser TF signals are .shift(1) on their native bars before ffill.
    This ensures the signal at any H1 bar reflects only the last CLOSED
    coarser bar, never the still-open current one.

Forward returns:
    Horizons [1, 5, 10, 20, 60] H1 bars -- identical to run_signal_sweep.py
    and run_signal_combination.py so that IC values are directly comparable
    across all Phase 1/1.5/2 scripts.

Usage:
    uv run python research/ic_analysis/run_mtf_stack.py
    uv run python research/ic_analysis/run_mtf_stack.py \\
        --instrument EUR_USD --tfs W,D,H4,H1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import research.ic_analysis.phase1_sweep as _sweep  # noqa: E402
from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
    build_all_signals,
)
from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
    compute_ic_table,
    compute_icir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default TFs in order: coarsest -> finest
DEFAULT_TFS = ["W", "D", "H4", "H1"]
BASE_TF = "H1"

# Horizons in H1 bars — roughly: 1h, 4h, 1day, 1wk, 2wks
HORIZONS = [1, 5, 10, 20, 60]  # matches phase1_sweep.py / phase2_combination.py


# ── Verdict ────────────────────────────────────────────────────────────────────


def _verdict(ic: float, icir: float) -> str:
    abs_ic = abs(ic) if not np.isnan(ic) else 0.0
    abs_ir = abs(icir) if not np.isnan(icir) else 0.0
    if abs_ic >= 0.05 and abs_ir >= 0.5:
        return "STRONG"
    elif abs_ic >= 0.05:
        return "USABLE"
    elif abs_ic >= 0.03:
        return "WEAK"
    return "NOISE"


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_all_tfs(instrument: str, tfs: list[str]) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for each TF. Skips missing files with a warning."""
    data = {}
    for tf in tfs:
        try:
            data[tf] = _load_ohlcv(instrument, tf)
        except FileNotFoundError:
            logger.warning("Data not found for %s %s -- skipping", instrument, tf)
    return data


# ── Signal computation and alignment ──────────────────────────────────────────


def _build_tf_signals(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Compute all 52 signals on a single TF's native bars."""
    logger.info("Computing 52 signals on %s...", tf)
    window_1y = _get_annual_bars(tf)
    return build_all_signals(df, window_1y)


def _align_to_base(
    signals: pd.DataFrame,
    base_index: pd.DatetimeIndex,
    is_coarser: bool = False,
) -> pd.DataFrame:
    """Forward-fill signals from native TF bars to the base (H1) index.

    This is causal: at any H1 bar, the coarser-TF signal reflects the last
    completed bar of that TF. No future bar information is used.

    For coarser TFs (H4, D, W), bars are timestamped at their OPEN. The signal
    is only available after the bar closes, so we shift by 1 native bar before
    forward-filling to prevent lookahead bias.
    """
    if is_coarser:
        signals = signals.shift(1)
    aligned = signals.reindex(base_index, method="ffill")
    return aligned


def build_aligned_signals(
    instrument: str,
    tfs: list[str],
    base_tf: str = BASE_TF,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Load, compute, and align 52 signals for all TFs to the base TF index.

    Returns:
        (tf_signals, base_index)
        tf_signals: dict TF -> aligned DataFrame (52 cols, H1 index)
        base_index: the H1 DatetimeIndex used for alignment
    """
    tfs_data = _load_all_tfs(instrument, tfs)
    if base_tf not in tfs_data:
        raise ValueError(f"Base TF '{base_tf}' data not available for {instrument}")

    base_index = tfs_data[base_tf].index

    tf_signals: dict[str, pd.DataFrame] = {}
    for tf, df in tfs_data.items():
        native_signals = _build_tf_signals(df, tf)
        aligned = _align_to_base(native_signals, base_index, is_coarser=(tf != base_tf))
        tf_signals[tf] = aligned
        logger.info(
            "%s: %d native bars -> %d aligned H1 bars",
            tf,
            len(native_signals),
            len(aligned),
        )

    return tf_signals, base_index


# ── Single-TF IC evaluation ────────────────────────────────────────────────────


def compute_single_tf_ics(
    tf_signals: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    horizons: list[int],
) -> dict[str, pd.DataFrame]:
    """Compute IC table for each TF's aligned signals (all evaluated on H1 fwd returns).

    Returns dict: TF -> IC DataFrame (52 signals × n horizons).
    """
    ic_by_tf: dict[str, pd.DataFrame] = {}
    for tf, signals in tf_signals.items():
        logger.info("Computing IC for %s signals (aligned to H1)...", tf)
        valid = signals.notna().any(axis=1) & fwd_returns.notna().any(axis=1)
        sig_v = signals[valid]
        fwd_v = fwd_returns[valid]
        ic_by_tf[tf] = compute_ic_table(sig_v, fwd_v)
    return ic_by_tf


def compute_single_tf_icirs(
    tf_signals: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    horizons: list[int],
) -> dict[str, pd.Series]:
    """Compute ICIR (at first horizon) for each TF. Returns dict: TF -> ICIR Series."""
    icir_by_tf: dict[str, pd.Series] = {}
    for tf, signals in tf_signals.items():
        valid = signals.notna().any(axis=1) & fwd_returns.notna().any(axis=1)
        sig_v = signals[valid]
        fwd_v = fwd_returns[valid]
        window_1y = _get_annual_bars(tf)
        icir_by_tf[tf] = compute_icir(sig_v, fwd_v, horizons=horizons, window=window_1y)
    return icir_by_tf


# ── MTF stacking ───────────────────────────────────────────────────────────────


def _score_series(
    composite: pd.Series,
    fwd_returns: pd.DataFrame,
    horizons: list[int],
) -> tuple[float, int, float, float, str]:
    """Compute IC/ICIR for a composite; return (best_ic, best_h, best_icir, icir_h1, verdict)."""
    fwd_cols = [f"fwd_{h}" for h in horizons]
    avail = [c for c in fwd_cols if c in fwd_returns.columns]
    if not avail:
        return np.nan, horizons[0], np.nan, np.nan, "NOISE"

    sig_df = composite.to_frame(name=composite.name or "stacked")
    ic_df = compute_ic_table(sig_df, fwd_returns[avail])

    # Best IC across horizons
    row = ic_df.iloc[0]
    abs_row = row.abs()
    if abs_row.isna().all():
        return np.nan, horizons[0], np.nan, np.nan, "NOISE"
    best_col = abs_row.idxmax()
    best_ic = float(row[best_col])
    best_h = int(best_col.replace("fwd_", ""))

    # ICIR at best horizon
    window_1y = _get_annual_bars(BASE_TF)
    icir_s = compute_icir(sig_df, fwd_returns[[best_col]], horizons=[best_h], window=window_1y)
    best_icir = float(icir_s.iloc[0]) if not icir_s.empty else np.nan

    # Also ICIR at first horizon for comparison
    h1_col = f"fwd_{horizons[0]}"
    if h1_col in fwd_returns.columns:
        icir_h1_s = compute_icir(
            sig_df, fwd_returns[[h1_col]], horizons=[horizons[0]], window=window_1y
        )
        icir_h1 = float(icir_h1_s.iloc[0]) if not icir_h1_s.empty else np.nan
    else:
        icir_h1 = np.nan

    return best_ic, best_h, best_icir, icir_h1, _verdict(best_ic, best_icir)


def build_stacked_composites(
    tf_signals: dict[str, pd.DataFrame],
    ic_by_tf: dict[str, pd.DataFrame],
    icir_by_tf: dict[str, pd.Series],
    signal_names: list[str],
    horizons: list[int],
    tfs: list[str],
    fwd_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build MTF stacked composites for each signal using 3 methods.

    For each signal:
      - Equal-weight:     mean of TF variants (sign-normalised at each TF's best IC)
      - ICIR-weighted:    weighted by |ICIR| per TF
      - Confluence gate:  mean of sign(TF variant) → proportion of TFs that agree

    Args:
        tf_signals:   dict TF -> aligned signal DataFrame (H1 index)
        ic_by_tf:     dict TF -> IC DataFrame (signals × horizons)
        icir_by_tf:   dict TF -> ICIR Series (one value per signal)
        signal_names: signals to stack (must be present in tf_signals[tfs[-1]])
        horizons:     forward return horizons in base-TF bars
        tfs:          timeframes coarse-to-fine; tfs[-1] is the base TF (H1)
        fwd_returns:  forward return DataFrame aligned to base-TF index

    Returns DataFrame with columns:
        signal, method, n_tfs, best_ic, best_h, best_icir, verdict, h1_best_ic, vs_h1_ic
    """
    base_tf = tfs[-1]
    all_results: list[dict] = []
    for sig in signal_names:
        rows = _build_stacked_for_signal(
            sig,
            tf_signals,
            ic_by_tf,
            icir_by_tf,
            fwd_returns,
            horizons,
            tfs,
            base_tf,
        )
        all_results.extend(rows)
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def _build_stacked_for_signal(
    signal_name: str,
    tf_signals: dict[str, pd.DataFrame],
    ic_by_tf: dict[str, pd.DataFrame],
    icir_by_tf: dict[str, pd.Series],
    fwd_returns: pd.DataFrame,
    horizons: list[int],
    tfs: list[str],
    h1_tf: str,
) -> list[dict]:
    """Build and score all 3 stacking methods for one signal. Returns list of result dicts."""
    # Collect aligned series per TF
    tf_series: dict[str, pd.Series] = {}
    for tf in tfs:
        if tf not in tf_signals or signal_name not in tf_signals[tf].columns:
            continue
        tf_series[tf] = tf_signals[tf][signal_name]

    if len(tf_series) < 2:
        return []

    # H1-only IC reference (at best horizon across all horizons)
    _h1_ic: pd.DataFrame = ic_by_tf[h1_tf] if h1_tf in ic_by_tf else pd.DataFrame()
    h1_ic_row = _h1_ic.loc[signal_name] if signal_name in _h1_ic.index else pd.Series(dtype=float)
    if h1_ic_row.empty:
        h1_best_ic = np.nan
    else:
        h1_best_ic = float(h1_ic_row.abs().max())

    # Sign orientation: for each TF, determine the sign of IC at first horizon
    fwd_col_ref = f"fwd_{horizons[0]}"

    def _ic_sign(tf: str) -> float:
        if tf not in ic_by_tf:
            return 1.0
        ic_df = ic_by_tf[tf]
        if signal_name not in ic_df.index or fwd_col_ref not in ic_df.columns:
            return 1.0
        ic_val = ic_df.loc[signal_name, fwd_col_ref]
        return float(np.sign(ic_val)) if not np.isnan(ic_val) and ic_val != 0.0 else 1.0

    def _icir_val(tf: str) -> float:
        if tf not in icir_by_tf:
            return 0.0
        return abs(float(icir_by_tf[tf].get(signal_name, 0.0) or 0.0))

    results = []

    # ── Method 1: Equal-weight ─────────────────────────────────────────────────
    oriented = [tf_series[tf] * _ic_sign(tf) for tf in tfs if tf in tf_series]
    if oriented:
        eq = pd.concat(oriented, axis=1).mean(axis=1)
        eq.name = f"{signal_name}_mtf_eq"
        # Drop NaN rows
        valid = eq.notna() & fwd_returns.notna().any(axis=1)
        best_ic, best_h, best_icir, _, verdict = _score_series(
            eq[valid], fwd_returns[valid], horizons
        )
        delta = (
            abs(best_ic) - h1_best_ic
            if not np.isnan(best_ic) and not np.isnan(h1_best_ic)
            else np.nan
        )
        results.append(
            {
                "signal": signal_name,
                "method": "equal_weight",
                "n_tfs": len(oriented),
                "best_ic": best_ic,
                "best_h": best_h,
                "best_icir": best_icir,
                "verdict": verdict,
                "h1_best_ic": h1_best_ic,
                "vs_h1_ic": delta,
            }
        )

    # ── Method 2: ICIR-weighted ────────────────────────────────────────────────
    icir_vals = {tf: _icir_val(tf) for tf in tfs if tf in tf_series}
    total_icir = sum(icir_vals.values())
    if total_icir > 0:
        weighted_parts = [
            tf_series[tf] * _ic_sign(tf) * (icir_vals[tf] / total_icir)
            for tf in tfs
            if tf in tf_series
        ]
        iw = pd.concat(weighted_parts, axis=1).sum(axis=1)
        iw.name = f"{signal_name}_mtf_iw"
        valid = iw.notna() & fwd_returns.notna().any(axis=1)
        best_ic, best_h, best_icir, _, verdict = _score_series(
            iw[valid], fwd_returns[valid], horizons
        )
        delta = (
            abs(best_ic) - h1_best_ic
            if not np.isnan(best_ic) and not np.isnan(h1_best_ic)
            else np.nan
        )
        results.append(
            {
                "signal": signal_name,
                "method": "icir_weighted",
                "n_tfs": len(icir_vals),
                "best_ic": best_ic,
                "best_h": best_h,
                "best_icir": best_icir,
                "verdict": verdict,
                "h1_best_ic": h1_best_ic,
                "vs_h1_ic": delta,
            }
        )

    # ── Method 3: Confluence gate (proportion of TFs that agree) ──────────────
    # +1 if TF agrees with positive orientation, -1 if disagrees, 0 if zero
    sign_parts = [np.sign(tf_series[tf] * _ic_sign(tf)) for tf in tfs if tf in tf_series]
    if sign_parts:
        conf = pd.concat(sign_parts, axis=1).mean(axis=1)  # ranges -1 to +1
        conf.name = f"{signal_name}_mtf_conf"
        valid = conf.notna() & fwd_returns.notna().any(axis=1)
        best_ic, best_h, best_icir, _, verdict = _score_series(
            conf[valid], fwd_returns[valid], horizons
        )
        delta = (
            abs(best_ic) - h1_best_ic
            if not np.isnan(best_ic) and not np.isnan(h1_best_ic)
            else np.nan
        )
        results.append(
            {
                "signal": signal_name,
                "method": "confluence",
                "n_tfs": len(sign_parts),
                "best_ic": best_ic,
                "best_h": best_h,
                "best_icir": best_icir,
                "verdict": verdict,
                "h1_best_ic": h1_best_ic,
                "vs_h1_ic": delta,
            }
        )

    return results


# ── Display ────────────────────────────────────────────────────────────────────


def _print_single_tf_table(
    ic_by_tf: dict[str, pd.DataFrame],
    tfs: list[str],
    horizons: list[int],
    top_n: int = 15,
) -> None:
    """Print cross-TF IC comparison for the top signals by best |IC| on H1."""
    h1_tf = tfs[-1]
    if h1_tf not in ic_by_tf:
        return

    ic_h1 = ic_by_tf[h1_tf]
    best_ic_h1 = ic_h1.abs().max(axis=1).sort_values(ascending=False)
    top_signals = best_ic_h1.head(top_n).index.tolist()

    W = 80
    print()
    print(f"  SINGLE-TF IC (aligned to H1) -- top {top_n} signals by H1 |IC|:")
    print("  " + "-" * (W - 2))

    # Header
    header = f"  {'Signal':<24}"
    for tf in tfs:
        header += f"  {tf:>8}"
    header += "  (IC at best horizon per TF)"
    print(header)
    print("  " + "-" * (W - 2))

    for sig in top_signals:
        row = f"  {sig:<24}"
        for tf in tfs:
            if tf not in ic_by_tf or sig not in ic_by_tf[tf].index:
                row += "       NaN"
                continue
            ic_row = ic_by_tf[tf].loc[sig]
            best_val = float(ic_row.loc[ic_row.abs().idxmax()])
            row += f"  {best_val:>+8.4f}"
        print(row)


def _print_stacking_leaderboard(
    stacking_results: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Print top stacked composites ranked by improvement over H1 alone."""
    W = 80
    if stacking_results.empty:
        print("  No stacking results.")
        return

    # Best result per (signal, method) — already one row per combination
    # Rank by vs_h1_ic descending
    ranked = (
        stacking_results.assign(abs_ic=stacking_results["best_ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print()
    print(f"  MTF STACKING LEADERBOARD (top {top_n}, ranked by stacked |IC|):")
    print("  " + "-" * (W - 2))
    print(
        f"  {'Rank':>4}  {'Signal':<22}  {'Method':<16}  "
        f"{'IC':>8}  {'BestH':>6}  {'ICIR':>7}  {'vs_H1':>7}  Verdict"
    )
    print("  " + "-" * (W - 2))

    for i, row in ranked.iterrows():
        ic_s = f"{row['best_ic']:>+8.4f}" if not np.isnan(row["best_ic"]) else "     NaN"
        ir_s = f"{row['best_icir']:>+7.3f}" if not np.isnan(row["best_icir"]) else "    NaN"
        vs_s = f"{row['vs_h1_ic']:>+7.4f}" if not np.isnan(row["vs_h1_ic"]) else "    NaN"
        best_h = int(row["best_h"]) if not np.isnan(row["best_h"]) else 0
        print(
            f"  {i + 1:>4}  {row['signal']:<22}  {row['method']:<16}  "
            f"{ic_s}  h={best_h:<4}  {ir_s}  {vs_s}  {row['verdict']}"
        )

    # Summary stats per method
    print("  " + "-" * (W - 2))
    print()
    print("  IMPROVEMENT vs H1-ALONE (median delta |IC| per method):")
    for method in stacking_results["method"].unique():
        sub = stacking_results[stacking_results["method"] == method]["vs_h1_ic"].dropna()
        if sub.empty:
            continue
        positive = (sub > 0).mean() * 100
        print(
            f"  {method:<20}  median delta: {sub.median():>+.4f}  "
            f"  signals improved: {positive:.0f}%"
        )


# ── Orchestrator ───────────────────────────────────────────────────────────────


def run_mtf_stack(
    instrument: str,
    tfs: list[str] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    if tfs is None:
        tfs = DEFAULT_TFS
    if horizons is None:
        horizons = HORIZONS

    base_tf = tfs[-1]  # finest TF is last (e.g., H1)

    W = 74
    print("\n" + "=" * W)
    print(f"  MTF SIGNAL STACKING -- {instrument}")
    print(f"  TFs: {' -> '.join(tfs)}  |  Base: {base_tf}  |  Horizons (bars): {horizons}")
    print("=" * W)

    # 1. Load and align all TFs to base
    tf_signals, base_index = build_aligned_signals(instrument, tfs, base_tf)
    group_map = dict(_sweep._SIGNAL_GROUP)  # populated by build_all_signals calls
    signal_names: list[str] = [str(s) for s in tf_signals[base_tf].columns if s in group_map]

    # 2. Forward returns on base TF (H1)
    base_close = _load_ohlcv(instrument, base_tf)["close"]
    fwd_returns = compute_forward_returns(base_close, horizons)

    # Align fwd_returns to base_index (may differ by a few bars at edges)
    fwd_returns = fwd_returns.reindex(base_index)

    n_bars = int(fwd_returns.notna().any(axis=1).sum())
    logger.info("Base index: %d H1 bars, forward returns valid: %d", len(base_index), n_bars)

    # 3. Single-TF IC for each TF (all aligned to H1)
    ic_by_tf = compute_single_tf_ics(tf_signals, fwd_returns, horizons)
    icir_by_tf = compute_single_tf_icirs(tf_signals, fwd_returns, horizons)

    # 4. Print single-TF comparison table
    _print_single_tf_table(ic_by_tf, tfs, horizons, top_n=15)

    # 5. Build stacked composites for all signals
    logger.info("Building stacked composites for %d signals x 3 methods...", len(signal_names))
    all_results: list[dict] = []
    for sig in signal_names:
        rows = _build_stacked_for_signal(
            sig,
            tf_signals,
            ic_by_tf,
            icir_by_tf,
            fwd_returns,
            horizons,
            tfs,
            base_tf,
        )
        all_results.extend(rows)

    if not all_results:
        logger.warning("No stacking results produced")
        return pd.DataFrame()

    stacking_df = pd.DataFrame(all_results)

    # 6. Print leaderboard
    _print_stacking_leaderboard(stacking_df, top_n=20)

    # 7. Save
    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{instrument}_mtf_{'_'.join(tfs)}".lower()
    out_path = report_dir / f"ic_mtf_stack_{slug}.csv"
    stacking_df.to_csv(out_path, index=False)
    logger.info("Saved: %s", out_path)

    # Also save single-TF IC comparison
    rows = []
    for sig in signal_names:
        row: dict = {"signal": sig, "group": group_map.get(sig, "?")}
        for tf in tfs:
            if tf not in ic_by_tf or sig not in ic_by_tf[tf].index:
                row[f"best_ic_{tf}"] = np.nan
                row[f"best_h_{tf}"] = np.nan
            else:
                ic_row = ic_by_tf[tf].loc[sig]
                best_col = ic_row.abs().idxmax()
                row[f"best_ic_{tf}"] = float(ic_row[best_col])
                row[f"best_h_{tf}"] = int(best_col.replace("fwd_", ""))
        rows.append(row)
    single_tf_df = pd.DataFrame(rows)
    single_path = report_dir / f"ic_single_tf_{slug}.csv"
    single_tf_df.to_csv(single_path, index=False)
    logger.info("Single-TF IC table saved: %s", single_path)

    print("\n" + "=" * W)

    return stacking_df


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF Signal Stacking IC Analysis")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument(
        "--tfs",
        default=",".join(DEFAULT_TFS),
        help="Comma-separated timeframes coarse-to-fine, e.g. W,D,H4,H1",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help="Comma-separated H1 forward horizons, e.g. 1,4,20,80,240",
    )
    args = parser.parse_args()
    tfs = [t.strip() for t in args.tfs.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]
    run_mtf_stack(instrument=args.instrument, tfs=tfs, horizons=horizons)


if __name__ == "__main__":
    main()
