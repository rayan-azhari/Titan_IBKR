"""WFO validation: margin drift L=2 + I1+I2+I3 improvements.

Rolling-window OOS validation of the strongest hybrid found in this
session: drift-margin L=2 on CSPX with all three Samir-Stack
improvements (rate-shock score, bond rotation, opt-in EFA overlay).

What this WFO is checking
-------------------------
None of the strategy's parameters are tuned in-sample (the rate-shock
threshold, bond-rotation lookback, EFA gap percent, regime-score
weights are all fixed by convention). So this is NOT a parameter-
tuning WFO. Instead it tests two things:

1. **Robustness across rolling windows**: does the OOS Sharpe stay
   positive across folds, or is the full-period number propped up by
   one or two lucky regimes (e.g. the 2009 bond-rotation HYG win,
   2018 rate-shock catch)?

2. **Stability of the improvement uplift**: is the gap between
   ``baseline`` and ``+ I1+I2+I3`` consistent fold-by-fold, or
   concentrated in specific regimes that may not repeat?

Methodology
-----------
- IS window: 504 days (warmup only — no fitting)
- OOS window: 252 days (~1 year)
- Step: 252 days (non-overlapping OOS folds)
- Variants compared per fold:
    * ``baseline``: drift L=2 margin, no improvements
    * ``+ I1+I2+I3``: drift L=2 margin with all three improvements
- Stitched stats use ``titan.research.metrics.sharpe`` with
  bootstrap CI for the deployment gate (CI lower bound > 0).

Usage:
    uv run python research/samir_stack/run_wfo_drift_improvements.py
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
from research.samir_stack.run_improved_with_margin import (  # noqa: E402
    _engine_margin_drift,
    run_improved_stack,
)
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
        f"Range: {common.min().date()} → {common.max().date()} ({len(common)} bars)\n", flush=True
    )

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    # Run BOTH variants over the FULL period once, then slice into OOS folds.
    # Re-running per fold would be more rigorous but the strategy has no
    # IS-fit parameters — slicing the equity curve is equivalent.
    print("Running full-period backtests (drift L=2)...", flush=True)
    df_baseline = run_improved_stack(
        spy,
        efa,
        ief,
        hyg,
        tlt,
        samir_score,
        L_max=2.0,
        equity_engine=_engine_margin_drift,
        use_rate_shock=False,
        use_bond_rotation=False,
        use_optin_efa=False,
    )
    df_improved = run_improved_stack(
        spy,
        efa,
        ief,
        hyg,
        tlt,
        samir_score,
        L_max=2.0,
        equity_engine=_engine_margin_drift,
        use_rate_shock=True,
        use_bond_rotation=True,
        use_optin_efa=True,
    )

    # Walk-forward fold definitions
    is_days = 504
    oos_days = 252
    step = 252
    n = len(df_improved)
    if n < is_days + oos_days:
        print(f"Insufficient data: {n} bars")
        return 1

    print(f"WFO: IS={is_days}d, OOS={oos_days}d, step={step}d\n", flush=True)
    print(
        f"{'fold':>4}  {'oos_start':<12}  {'oos_end':<12}  "
        f"{'BASE_Sharpe':>12}  {'IMP_Sharpe':>11}  {'IMP_CAGR':>9}  {'IMP_MaxDD':>10}  "
        f"{'uplift_Sharpe':>14}",
        flush=True,
    )
    print("-" * 110, flush=True)

    folds = []
    stitched_baseline: list[np.ndarray] = []
    stitched_improved: list[np.ndarray] = []

    base_rets = df_baseline["ret_strategy"].to_numpy()
    imp_rets = df_improved["ret_strategy"].to_numpy()
    idx = df_improved.index

    fold_idx = 0
    oos_start = is_days
    while oos_start + oos_days <= n:
        oos_end = oos_start + oos_days
        oos_dates = idx[oos_start:oos_end]
        base_slice = base_rets[oos_start:oos_end]
        imp_slice = imp_rets[oos_start:oos_end]
        base_stats = _fold_stats(base_slice)
        imp_stats = _fold_stats(imp_slice)
        folds.append(
            {
                "fold": fold_idx,
                "oos_start": oos_dates[0].strftime("%Y-%m-%d"),
                "oos_end": oos_dates[-1].strftime("%Y-%m-%d"),
                "base_sharpe": base_stats["sharpe"],
                "imp_sharpe": imp_stats["sharpe"],
                "imp_cagr": imp_stats["cagr"],
                "imp_max_dd": imp_stats["max_dd"],
                "uplift_sharpe": round(imp_stats["sharpe"] - base_stats["sharpe"], 3),
            }
        )
        stitched_baseline.append(base_slice)
        stitched_improved.append(imp_slice)
        print(
            f"{fold_idx:>4}  {oos_dates[0].strftime('%Y-%m-%d'):<12}  "
            f"{oos_dates[-1].strftime('%Y-%m-%d'):<12}  "
            f"{base_stats['sharpe']:>+12.3f}  {imp_stats['sharpe']:>+11.3f}  "
            f"{imp_stats['cagr']:>+9.2%}  {imp_stats['max_dd']:>+10.2%}  "
            f"{imp_stats['sharpe'] - base_stats['sharpe']:>+14.3f}",
            flush=True,
        )
        fold_idx += 1
        oos_start += step

    fold_df = pd.DataFrame(folds)

    # Stitched OOS stats with bootstrap CI
    all_base = np.concatenate(stitched_baseline)
    all_imp = np.concatenate(stitched_improved)

    base_sh = sharpe(all_base, periods_per_year=BARS_PER_YEAR["D"])
    base_lo, base_hi = bootstrap_sharpe_ci(
        all_base, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    imp_sh = sharpe(all_imp, periods_per_year=BARS_PER_YEAR["D"])
    imp_lo, imp_hi = bootstrap_sharpe_ci(
        all_imp, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )

    n_years = len(all_imp) / 252.0
    base_eq = np.cumprod(1.0 + all_base)
    imp_eq = np.cumprod(1.0 + all_imp)
    base_cagr = float(base_eq[-1] ** (1.0 / n_years) - 1.0)
    imp_cagr = float(imp_eq[-1] ** (1.0 / n_years) - 1.0)
    base_dd = float(
        ((base_eq - np.maximum.accumulate(base_eq)) / np.maximum.accumulate(base_eq)).min()
    )
    imp_dd = float(((imp_eq - np.maximum.accumulate(imp_eq)) / np.maximum.accumulate(imp_eq)).min())

    print("\n" + "=" * 110)
    print("WFO stitched-OOS summary (margin drift L=2):")
    print("=" * 110)
    summary = pd.DataFrame(
        [
            {
                "variant": "baseline (no improvements)",
                "n_oos_years": round(n_years, 2),
                "n_folds": len(folds),
                "stitched_sharpe": round(base_sh, 3),
                "ci95_lo": round(base_lo, 3),
                "ci95_hi": round(base_hi, 3),
                "stitched_cagr": round(base_cagr, 4),
                "stitched_max_dd": round(base_dd, 4),
                "pct_pos_folds": round(float((fold_df["base_sharpe"] > 0).mean()), 3),
                "passes_gate": base_lo > 0,
            },
            {
                "variant": "+ I1+I2+I3 improvements",
                "n_oos_years": round(n_years, 2),
                "n_folds": len(folds),
                "stitched_sharpe": round(imp_sh, 3),
                "ci95_lo": round(imp_lo, 3),
                "ci95_hi": round(imp_hi, 3),
                "stitched_cagr": round(imp_cagr, 4),
                "stitched_max_dd": round(imp_dd, 4),
                "pct_pos_folds": round(float((fold_df["imp_sharpe"] > 0).mean()), 3),
                "passes_gate": imp_lo > 0,
            },
        ]
    ).set_index("variant")
    print(summary.to_string())

    # Robustness check: distribution of fold uplifts
    print("\nFold-uplift distribution (improved Sharpe minus baseline Sharpe):")
    uplifts = fold_df["uplift_sharpe"].to_numpy()
    print(
        f"  mean={uplifts.mean():+.3f}  median={np.median(uplifts):+.3f}  "
        f"min={uplifts.min():+.3f}  max={uplifts.max():+.3f}  "
        f"pct_positive={float((uplifts > 0).mean()):.0%}"
    )

    out = REPORTS_DIR / "wfo_drift_improvements.csv"
    fold_df.to_csv(out, index=False)
    summary.to_csv(REPORTS_DIR / "wfo_drift_improvements_summary.csv")
    print(f"\nSaved fold detail: {out}")
    print(f"Saved summary:     {REPORTS_DIR / 'wfo_drift_improvements_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
