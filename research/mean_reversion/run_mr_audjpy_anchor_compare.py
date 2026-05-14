"""run_mr_audjpy_anchor_compare.py -- 2-cell audit comparison: anchor=24 vs anchor=6.

Pre-registered in:
    directives/MR AUDJPY Audit Re-Run 2026-05-14.md

Wraps the existing audit-corrected run_mr_wfo with two anchor periods,
emits a side-by-side comparison + applies the directive's §1.2 decision
rule. No new strategy semantics -- every aspect of the strategy
(tiers, regime gate, session window, sizing, costs) is identical
between cells; only `anchor_period` differs.

Usage::

    python research/mean_reversion/run_mr_audjpy_anchor_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.mean_reversion.run_confluence_regime_test import (  # noqa: E402
    build_confluence_disagreement_mask,
    compute_vwap_deviation,
    load_h1,
)
from research.mean_reversion.run_confluence_regime_wfo import run_mr_wfo  # noqa: E402


# Pre-committed parameters from directive §1.1 (also matches config/mr_audjpy.toml).
PAIR = "AUD_JPY"
TIERS_PCT = [0.95, 0.98, 0.99, 0.999]  # "conservative" grid -- matches live config
IS_BARS = 30_000
OOS_BARS = 7_500
ANCHORS = (24, 6)  # 24 == live config; 6 == IC-peak (from fine-grid)


def _decision(ci_lo_24: float, ci_lo_6: float) -> tuple[str, str]:
    """Apply directive §1.2 decision rule."""
    gap = ci_lo_6 - ci_lo_24
    if ci_lo_24 <= 0 and ci_lo_6 <= 0:
        return "RETIRE", (
            "Both anchors have CI_lo <= 0. Strategy's mechanical basis is too "
            "thin under audit-corrected math. Issue config-change PR to retire."
        )
    if ci_lo_24 > 0 and ci_lo_6 <= 0:
        return "STATUS_QUO_WATCHPOINT", (
            "anchor=6 is WORSE on the engine despite stronger raw IC. "
            "Strategy's layers (regime / tiers / sizing) interact with anchor "
            "in unexpected ways. Document; status quo with watchpoint."
        )
    if gap >= 0.20 and ci_lo_6 > 0:
        return "RECONFIGURE", (
            f"anchor=6 beats anchor=24 by CI_lo gap of {gap:.3f} (>= 0.20 threshold). "
            f"Reconfigure live to anchor=6: update config/mr_audjpy.toml, run live "
            f"parity test against the existing class, then deploy."
        )
    return "STATUS_QUO", (
        f"anchor=6 vs anchor=24 CI_lo gap of {gap:.3f} (< 0.20 threshold). "
        f"Anchor change is not material. Live config stands."
    )


def main() -> None:
    print("=" * 80)
    print(f"  MR AUDJPY ANCHOR COMPARISON -- {PAIR} H1")
    print(f"  Anchors: {ANCHORS[0]} (live) vs {ANCHORS[1]} (IC peak)")
    print(f"  IS: {IS_BARS} bars / OOS: {OOS_BARS} bars / tiers: {TIERS_PCT}")
    print("=" * 80)

    df = load_h1(PAIR)
    close = df["close"]
    print(f"\n  Loaded {len(close)} bars ({close.index[0].date()} -> {close.index[-1].date()})")

    # Build the regime mask ONCE — confluence disagreement on rsi_14_dev is
    # anchor-independent (RSI signal computed on close, not on deviation).
    print("\n  Building regime mask (conf_rsi_14_dev)...")
    regime_mask = build_confluence_disagreement_mask(df, "rsi_14_dev")
    n_total = len(df)
    pct_gated = float(regime_mask.sum()) / n_total * 100
    print(f"    regime mask True on {pct_gated:.1f}% of bars ({int(regime_mask.sum())} of {n_total})")

    # Run WFO at each anchor.
    results: dict[int, dict] = {}
    for anchor in ANCHORS:
        print(f"\n  --- anchor_period = {anchor} ---")
        deviation = compute_vwap_deviation(close, anchor_period=anchor)
        res = run_mr_wfo(
            close=close,
            deviation=deviation,
            regime_mask=regime_mask,
            tiers_pct=TIERS_PCT,
            is_bars=IS_BARS,
            oos_bars=OOS_BARS,
            spread_bps=2.0,
            slippage_bps=1.0,
        )
        results[anchor] = res
        print(
            f"    stitched OOS Sharpe = {res['stitched_sharpe']}, "
            f"CI = [{res['sharpe_ci_95_lo']}, {res['sharpe_ci_95_hi']}], "
            f"DD = {res['stitched_dd_pct']}%, "
            f"n_folds = {res['n_folds']}, pos% = {res['pct_positive']*100:.0f}%, "
            f"n_trades = {res['total_trades']}"
        )

    # Side-by-side summary.
    print()
    print("=" * 80)
    print("  COMPARISON")
    print("=" * 80)
    cols = ["Field", f"anchor=24 (live)", f"anchor=6 (IC peak)", "delta"]
    rows = []
    a24, a6 = results[24], results[6]
    for label, key in [
        ("Stitched OOS Sharpe", "stitched_sharpe"),
        ("CI_lo (95%)", "sharpe_ci_95_lo"),
        ("CI_hi (95%)", "sharpe_ci_95_hi"),
        ("Stitched DD %", "stitched_dd_pct"),
        ("n_folds", "n_folds"),
        ("% positive folds", "pct_positive"),
        ("total trades", "total_trades"),
    ]:
        v24 = a24.get(key, 0)
        v6 = a6.get(key, 0)
        try:
            delta = float(v6) - float(v24)
            delta_str = f"{delta:+.3f}"
        except (TypeError, ValueError):
            delta_str = ""
        rows.append([label, str(v24), str(v6), delta_str])

    width_label = max(len(r[0]) for r in rows) + 2
    width_val = 22
    print(f"  {'Field':<{width_label}} {'anchor=24 (live)':>{width_val}} {'anchor=6 (IC peak)':>{width_val}} {'delta':>12}")
    print("  " + "-" * (width_label + 2 * width_val + 14))
    for label, v24, v6, delta_str in rows:
        print(f"  {label:<{width_label}} {v24:>{width_val}} {v6:>{width_val}} {delta_str:>12}")

    # Apply decision rule.
    print()
    print("=" * 80)
    print("  DECISION (§1.2 rule, pre-committed)")
    print("=" * 80)
    decision, rationale = _decision(a24["sharpe_ci_95_lo"], a6["sharpe_ci_95_lo"])
    print(f"\n  Verdict: {decision}")
    print(f"\n  Rationale: {rationale}")
    print()

    # Save per-fold + summary parquets.
    out_dir = PROJECT_ROOT / ".tmp" / "reports" / "mr_audjpy_anchor_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_rows = []
    for anchor in ANCHORS:
        r = results[anchor]
        summary_rows.append({
            "anchor_period": anchor,
            "stitched_sharpe": r["stitched_sharpe"],
            "sharpe_ci_95_lo": r["sharpe_ci_95_lo"],
            "sharpe_ci_95_hi": r["sharpe_ci_95_hi"],
            "stitched_dd_pct": r["stitched_dd_pct"],
            "n_folds": r["n_folds"],
            "pct_positive": r["pct_positive"],
            "total_trades": r["total_trades"],
        })
        fold_path = out_dir / f"folds_anchor{anchor}_{stamp}.parquet"
        r["fold_df"].to_parquet(fold_path, index=False)
        print(f"  Wrote {fold_path.relative_to(PROJECT_ROOT)}")
    summary_path = out_dir / f"summary_{stamp}.parquet"
    pd.DataFrame(summary_rows).to_parquet(summary_path, index=False)
    print(f"  Wrote {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
