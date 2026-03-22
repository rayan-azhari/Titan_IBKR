"""run_spy_wfo.py -- Phase 4: Walk-Forward Optimisation for SPY Daily Strategy.

Rolling anchor-expanding IS / fixed OOS windows:
  - IS window : 756 bars (≈ 3 years daily)
  - OOS window: 252 bars (≈ 1 year daily)
  - Step       : 252 bars (annual step = OOS window length)

For each fold the IS threshold is re-selected independently (no look-ahead).
Composite sign-calibration and z-score stats are re-computed from IS data only.

Quality gates (long-only daily; calibrated below FX L+S targets):
  1. >= 60% of folds with positive OOS Sharpe
  2. >= 40% of folds with OOS Sharpe > 0.5
  3. Worst single fold Sharpe >= -2.0
  4. Stitched OOS equity-curve Sharpe >= 0.30
  5. IS/OOS Sharpe parity >= 0.30

Output:
  .tmp/reports/spy_wfo.csv         -- per-fold stats (used by run_spy_strategy.py Gate 4d)
  .tmp/reports/spy_wfo_equity.csv  -- stitched OOS daily equity curve

Usage:
  uv run python research/ic_analysis/run_spy_wfo.py
  uv run python research/ic_analysis/run_spy_wfo.py --no-regime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.run_spy_strategy import (  # noqa: E402
    INIT_CASH,
    REPORTS,
    THRESHOLDS,
    W,
    _run_long,
    build_composite_for_fold,
    load_data,
)

# ── WFO config ────────────────────────────────────────────────────────────────

IS_BARS  = 756   # 3 years × 252 trading days
OOS_BARS = 252   # 1 year
STEP     = 252   # annual step (matches OOS window length)

# Quality gates
GATE_PCT_POSITIVE = 0.60   # >= 60% folds with OOS Sharpe > 0
GATE_PCT_ABOVE05  = 0.40   # >= 40% folds with OOS Sharpe > 0.5
GATE_WORST_FOLD   = -2.0   # no fold worse than -2.0 Sharpe
GATE_STITCHED     = 0.30   # stitched OOS Sharpe >= 0.30
GATE_PARITY       = 0.30   # OOS / IS Sharpe >= 0.30


# ── WFO runner ────────────────────────────────────────────────────────────────

def run_wfo(use_regime: bool = True) -> None:
    mode_label = "REGIME-GATED" if use_regime else "UNCONDITIONAL"
    print("\n" + "=" * W)
    print(f"  SPY DAILY -- WALK-FORWARD OPTIMISATION  ({mode_label})")
    print(
        f"  IS: {IS_BARS} bars (~{IS_BARS//252}yr)  |  "
        f"OOS: {OOS_BARS} bars (~1yr)  |  Step: {STEP} bars"
    )
    print("=" * W)

    print("\nLoading data and building signals (once)...")
    sigs, close, df, size = load_data()
    n = len(sigs)
    print(f"  Total bars: {n:,}  ({sigs.index[0].date()} → {sigs.index[-1].date()})")

    # Define rolling folds
    folds: list[tuple[int, int, int, int]] = []
    start = 0
    while start + IS_BARS + OOS_BARS <= n:
        is0, is1     = start, start + IS_BARS
        oos0, oos1   = is1, is1 + OOS_BARS
        folds.append((is0, is1, oos0, oos1))
        start += STEP

    n_folds = len(folds)
    if n_folds < 3:
        print(f"  WARNING: only {n_folds} folds available — WFO is not meaningful.")
        return

    print(
        f"  Folds: {n_folds}  "
        f"(first OOS start: {sigs.index[folds[0][2]].date()}  "
        f"last OOS end: {sigs.index[folds[-1][3]-1].date()})"
    )

    fold_results:  list[dict]       = []
    stitched_rets: list[pd.Series]  = []

    print(
        f"\n  {'Fold':>5}  {'IS-Thr':>7}  {'IS-Sh':>7}  "
        f"{'OOS-Sh':>7}  {'OOS-Ann':>9}  {'OOS-DD':>8}  {'Trades':>7}"
    )
    print("  " + "-" * 62)

    for i, (is0, is1, oos0, oos1) in enumerate(folds):
        # Slice to fold range — signals were built on full dataset, just slice
        fold_sigs  = sigs.iloc[is0:oos1]
        fold_close = close.iloc[is0:oos1]
        fold_size  = size.iloc[is0:oos1]

        # IS mask within this fold's index
        fold_is_mask = pd.Series(False, index=fold_sigs.index)
        fold_is_mask.iloc[: is1 - is0] = True

        # Build composite calibrated to this fold's IS window
        composite_z, _ = build_composite_for_fold(
            fold_sigs, fold_close, fold_is_mask, use_regime=use_regime
        )

        close_is = fold_close.iloc[: is1 - is0]
        gz_is    = composite_z.iloc[: is1 - is0]
        sz_is    = fold_size.iloc[: is1 - is0]

        close_oos = fold_close.iloc[is1 - is0:]
        gz_oos    = composite_z.iloc[is1 - is0:]
        sz_oos    = fold_size.iloc[is1 - is0:]

        # Pick best threshold from IS (no look-ahead into OOS)
        best_thr   = THRESHOLDS[0]
        best_sh_is = -np.inf
        for thr in THRESHOLDS:
            r_is = _run_long(close_is, gz_is, thr, sz_is)
            if r_is["sharpe"] > best_sh_is:
                best_sh_is = r_is["sharpe"]
                best_thr   = thr

        # OOS evaluation at IS-selected threshold
        r_oos = _run_long(close_oos, gz_oos, best_thr, sz_oos)

        fold_results.append({
            "fold":      i + 1,
            "is_start":  sigs.index[is0].date(),
            "is_end":    sigs.index[is1 - 1].date(),
            "oos_start": sigs.index[oos0].date(),
            "oos_end":   sigs.index[oos1 - 1].date(),
            "threshold": best_thr,
            "is_sharpe": round(best_sh_is, 3),
            "oos_sharpe": round(r_oos["sharpe"], 3),
            "oos_annual": round(float(r_oos["annual"]), 4),
            "oos_dd":    round(float(r_oos["dd"]), 4),
            "oos_trades": r_oos["trades"],
        })

        # Stitch OOS bar returns for equity reconstruction
        stitched_rets.append(r_oos["pf"].returns())

        print(
            f"  {i+1:>5}  {best_thr:>7.2f}  {best_sh_is:>+7.3f}"
            f"  {r_oos['sharpe']:>+7.3f}  {r_oos['annual']:>+9.1%}"
            f"  {r_oos['dd']:>+8.1%}  {r_oos['trades']:>7}"
        )

    # ── Stitch OOS equity curve ───────────────────────────────────────────────
    stitched = pd.concat(stitched_rets).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    stitched_equity = (1 + stitched).cumprod() * INIT_CASH

    std_s = float(stitched.std())
    stitched_sharpe = (
        float(stitched.mean()) / std_s * np.sqrt(252)
        if std_s > 1e-10 else 0.0
    )

    # ── Gate computation ─────────────────────────────────────────────────────
    oos_sharpes  = [f["oos_sharpe"] for f in fold_results]
    is_sharpes   = [f["is_sharpe"]  for f in fold_results]
    avg_is_sh    = float(np.mean(is_sharpes))
    avg_oos_sh   = float(np.mean(oos_sharpes))
    pct_positive = sum(s > 0   for s in oos_sharpes) / n_folds
    pct_above05  = sum(s > 0.5 for s in oos_sharpes) / n_folds
    worst_fold   = float(min(oos_sharpes))
    parity       = avg_oos_sh / avg_is_sh if avg_is_sh > 1e-6 else 0.0

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  WFO SUMMARY")
    print(f"{'='*W}")
    print(f"  Folds evaluated          : {n_folds}")
    print(f"  Avg IS Sharpe            : {avg_is_sh:>+.3f}")
    print(f"  Avg OOS Sharpe           : {avg_oos_sh:>+.3f}")
    print(f"  Stitched OOS Sharpe      : {stitched_sharpe:>+.3f}")
    print(f"  % folds OOS Sharpe > 0   : {pct_positive:.0%}")
    print(f"  % folds OOS Sharpe > 0.5 : {pct_above05:.0%}")
    print(f"  Worst fold Sharpe        : {worst_fold:>+.3f}")
    print(f"  IS/OOS parity            : {parity:.3f}")

    g1 = pct_positive  >= GATE_PCT_POSITIVE
    g2 = pct_above05   >= GATE_PCT_ABOVE05
    g3 = worst_fold    >= GATE_WORST_FOLD
    g4 = stitched_sharpe >= GATE_STITCHED
    g5 = parity        >= GATE_PARITY

    print(f"\n  {'─' * 60}")
    print("  GATE RESULTS")
    print(f"  {'─' * 60}")

    def gline(name: str, value: float, gate_str: str, passed: bool) -> None:
        lbl = "PASS" if passed else "FAIL"
        print(f"  [{lbl}] {name}: {value:.3f} (gate: {gate_str})")

    gline(
        f"% folds OOS Sharpe > 0   (gate >= {GATE_PCT_POSITIVE:.0%})",
        pct_positive, f">= {GATE_PCT_POSITIVE:.0%}", g1,
    )
    gline(
        f"% folds OOS Sharpe > 0.5 (gate >= {GATE_PCT_ABOVE05:.0%})",
        pct_above05, f">= {GATE_PCT_ABOVE05:.0%}", g2,
    )
    gline(
        f"Worst fold Sharpe         (gate >= {GATE_WORST_FOLD})",
        worst_fold, f">= {GATE_WORST_FOLD}", g3,
    )
    gline(
        f"Stitched OOS Sharpe       (gate >= {GATE_STITCHED})",
        stitched_sharpe, f">= {GATE_STITCHED}", g4,
    )
    gline(
        f"IS/OOS parity             (gate >= {GATE_PARITY})",
        parity, f">= {GATE_PARITY}", g5,
    )

    all_pass = all([g1, g2, g3, g4, g5])
    verdict = (
        "ALL GATES PASS -- WFO validated"
        if all_pass
        else "GATES FAILED -- investigate over-fitting"
    )
    print(f"\n  Verdict: {verdict}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_folds  = pd.DataFrame(fold_results)
    wfo_path  = REPORTS / "spy_wfo.csv"
    df_folds.to_csv(wfo_path, index=False)
    print(f"\n  Fold results saved   : {wfo_path}")

    eq_path = REPORTS / "spy_wfo_equity.csv"
    stitched_equity.to_csv(eq_path)
    print(f"  Stitched equity saved: {eq_path}")

    # Plotly equity chart
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        eq_norm = stitched_equity / stitched_equity.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=eq_norm.index, y=eq_norm.values,
            name="WFO Stitched OOS", line=dict(color="#00E5FF", width=2.0),
        ))
        fig.update_layout(
            title=f"SPY WFO Stitched OOS Equity | {mode_label} | {n_folds} folds",
            height=480,
            template="plotly_dark",
            hovermode="x unified",
            yaxis_title="Value (rebased 100)",
        )
        html_out = REPORTS / "spy_wfo_equity.html"
        fig.write_html(str(html_out))
        print(f"  WFO chart saved      : {html_out}")
    except Exception as e:
        print(f"  [INFO] Chart skipped: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SPY WFO Phase 4")
    parser.add_argument("--no-regime", action="store_true", help="Disable ADX regime gate")
    args = parser.parse_args()
    run_wfo(use_regime=not args.no_regime)


if __name__ == "__main__":
    main()
