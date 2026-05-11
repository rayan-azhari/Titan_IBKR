"""WFO across equity/bond allocations: 20/80 vs 30/70 vs 40/60.

Tests whether the in-sample finding from the allocation sweep
(more-bond is better) holds out-of-sample. Same 16-fold WFO
methodology as `run_wfo_drift_improvements.py` but parameterized over
the equity/bond split.

Engine: margin drift L=2 with I1+I2+I3 improvements (the champion).

Usage:
    uv run python research/samir_stack/run_wfo_allocation_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_allocation_sweep import run_with_allocation  # noqa: E402
from research.samir_stack.run_improved_with_margin import _engine_margin_drift  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _fold_stats(rets: np.ndarray) -> dict:
    if len(rets) < 20:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0}
    sh = sharpe(rets, periods_per_year=BARS_PER_YEAR["D"])
    eq = np.cumprod(1.0 + rets)
    n_years = len(rets) / 252.0
    cagr = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {"sharpe": round(sh, 3), "cagr": round(cagr, 4), "max_dd": round(maxdd, 4)}


def main() -> int:
    print("Loading data...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    efa = _load_close("EFA_D.parquet")

    common = (
        data["spy"]
        .index.intersection(efa.index)
        .intersection(data["tlt"].index)
        .intersection(data["hyg"].index)
        .intersection(data["ief"].index)
    )
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)
    efa = efa.reindex(common)
    print(
        f"Range: {common.min().date()} -> {common.max().date()} ({len(common)} bars)\n",
        flush=True,
    )

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    splits = [
        ("20/80", 0.20),
        ("30/70", 0.30),
        ("40/60 (current)", 0.40),
    ]

    # Run full-period backtests once per split, then slice into OOS folds.
    # No IS-fit parameters → slicing the equity curve is equivalent to
    # rebuilding per fold.
    print("Running full-period backtests...", flush=True)
    results: dict[str, pd.DataFrame] = {}
    for label, eq_w in splits:
        print(f"  {label}...", flush=True)
        results[label] = run_with_allocation(
            spy,
            efa,
            ief,
            hyg,
            tlt,
            samir_score,
            equity_weight=eq_w,
            bond_weight=1.0 - eq_w,
            L_max=2.0,
            equity_engine=_engine_margin_drift,
        )

    # Walk-forward
    is_days = 504
    oos_days = 252
    step = 252
    n = len(next(iter(results.values())))
    if n < is_days + oos_days:
        print(f"Insufficient data: {n} bars")
        return 1

    print(f"\nWFO: IS={is_days}d, OOS={oos_days}d, step={step}d", flush=True)

    fold_rows: list[dict] = []
    stitched: dict[str, list[np.ndarray]] = {label: [] for label, _ in splits}

    fold_idx = 0
    oos_start = is_days
    while oos_start + oos_days <= n:
        oos_end = oos_start + oos_days
        idx = next(iter(results.values())).index
        oos_dates = idx[oos_start:oos_end]
        rec: dict = {
            "fold": fold_idx,
            "oos_start": oos_dates[0].strftime("%Y-%m-%d"),
            "oos_end": oos_dates[-1].strftime("%Y-%m-%d"),
        }
        for label, _ in splits:
            slice_rets = results[label]["ret_strategy"].to_numpy()[oos_start:oos_end]
            stats = _fold_stats(slice_rets)
            rec[f"{label}_Sharpe"] = stats["sharpe"]
            rec[f"{label}_CAGR"] = stats["cagr"]
            rec[f"{label}_MaxDD"] = stats["max_dd"]
            stitched[label].append(slice_rets)
        fold_rows.append(rec)
        fold_idx += 1
        oos_start += step

    fold_df = pd.DataFrame(fold_rows)

    # Print fold-by-fold Sharpe table
    print("\nPer-fold OOS Sharpe:")
    print(
        f"{'fold':>4}  {'oos_start':<12}  {'oos_end':<12}  {'20/80':>9}  {'30/70':>9}  {'40/60':>9}"
    )
    print("-" * 70)
    for _, r in fold_df.iterrows():
        print(
            f"{int(r['fold']):>4}  {r['oos_start']:<12}  {r['oos_end']:<12}  "
            f"{r['20/80_Sharpe']:>+9.3f}  {r['30/70_Sharpe']:>+9.3f}  "
            f"{r['40/60 (current)_Sharpe']:>+9.3f}"
        )

    # Stitched aggregate stats
    print("\n" + "=" * 90)
    print("Stitched OOS summary (16-fold WFO, margin drift L=2 + I1+I2+I3):")
    print("=" * 90)

    summary_rows: list[dict] = []
    for label, _ in splits:
        all_rets = np.concatenate(stitched[label])
        sh = sharpe(all_rets, periods_per_year=BARS_PER_YEAR["D"])
        ci_lo, ci_hi = bootstrap_sharpe_ci(
            all_rets, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
        )
        n_years = len(all_rets) / 252.0
        eq = np.cumprod(1.0 + all_rets)
        cagr = float(eq[-1] ** (1.0 / n_years) - 1.0)
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        col = f"{label}_Sharpe"
        pos_pct = float((fold_df[col] > 0).mean())
        summary_rows.append(
            {
                "split": label,
                "n_oos_years": round(n_years, 2),
                "stitched_sharpe": round(sh, 3),
                "ci95_lo": round(ci_lo, 3),
                "ci95_hi": round(ci_hi, 3),
                "stitched_cagr": round(cagr, 4),
                "stitched_max_dd": round(dd, 4),
                "pct_pos_folds": round(pos_pct, 3),
                "passes_gate": ci_lo > 0,
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("split")
    print(summary.to_string())

    # Per-fold uplift summary (relative to 40/60 current)
    print("\nUplift vs current 40/60 (Sharpe diff per fold):")
    fold_df["uplift_2080"] = fold_df["20/80_Sharpe"] - fold_df["40/60 (current)_Sharpe"]
    fold_df["uplift_3070"] = fold_df["30/70_Sharpe"] - fold_df["40/60 (current)_Sharpe"]
    print(
        f"  20/80 vs 40/60: mean={fold_df['uplift_2080'].mean():+.3f}  "
        f"median={fold_df['uplift_2080'].median():+.3f}  "
        f"min={fold_df['uplift_2080'].min():+.3f}  "
        f"max={fold_df['uplift_2080'].max():+.3f}  "
        f"pos={float((fold_df['uplift_2080'] > 0).mean()):.0%}"
    )
    print(
        f"  30/70 vs 40/60: mean={fold_df['uplift_3070'].mean():+.3f}  "
        f"median={fold_df['uplift_3070'].median():+.3f}  "
        f"min={fold_df['uplift_3070'].min():+.3f}  "
        f"max={fold_df['uplift_3070'].max():+.3f}  "
        f"pos={float((fold_df['uplift_3070'] > 0).mean()):.0%}"
    )

    # Save
    out_csv = REPORTS_DIR / "wfo_allocation_comparison.csv"
    fold_df.to_csv(out_csv, index=False)
    summary.to_csv(REPORTS_DIR / "wfo_allocation_comparison_summary.csv")
    print(f"\nSaved fold detail: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
