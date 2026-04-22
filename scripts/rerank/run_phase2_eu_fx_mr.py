"""Phase 2 of the EU strategy plan — H1 VWAP mean-reversion on
EUR crosses using the AUD/JPY confluence framework.

Targets: EUR/CHF, EUR/GBP, EUR/JPY.
  Pattern: MRAUDJPYStrategy's WFO (already a Bonferroni-cleared
    champion). Reuses research/mean_reversion/run_confluence_regime_wfo.run_mr_wfo.
  Sweeps:
    * vwap_anchor in {12, 24, 36}
    * filter in {donchian_pos_20 disagreement, rsi_14_dev disagreement,
                 atr_only, no_filter}
    * tier-grid in {standard 0.90/0.95/0.98/0.99, conservative 0.95/0.98/0.99/0.999}

Writes:
  .tmp/reports/phase2_eu_fx_mr_2026_04_22/results.csv
  .tmp/reports/phase2_eu_fx_mr_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "phase2_eu_fx_mr_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)


PAIRS = ["EUR_CHF", "EUR_GBP", "EUR_JPY"]
VWAP_ANCHORS = [12, 24, 36]
FILTERS = ["conf_donchian_pos_20", "conf_rsi_14_dev", "atr_only", "no_filter"]
TIER_GRIDS = {
    "standard": [0.90, 0.95, 0.98, 0.99],
    "conservative": [0.95, 0.98, 0.99, 0.999],
}

# AUD/JPY champion reference config: vwap_anchor=24, conf_donchian_pos_20,
# conservative. Setting the bar accordingly.
BONF_CI_LO = 0.40  # N=72, less-strict than US cross-asset
MIN_FOLDS = 2
MIN_POS = 0.50


def run_config(
    pair: str, vwap_anchor: int, filter_name: str, grid_name: str
) -> dict | None:
    from research.mean_reversion.run_confluence_regime_test import (
        build_atr_regime_mask,
        build_confluence_disagreement_mask,
        compute_vwap_deviation,
        load_h1,
    )
    from research.mean_reversion.run_confluence_regime_wfo import run_mr_wfo

    try:
        df = load_h1(pair)
    except FileNotFoundError:
        return None
    n_bars = len(df)
    if n_bars >= 40000:
        is_bars, oos_bars = 32000, 8000
    elif n_bars >= 15000:
        is_bars, oos_bars = int(n_bars * 0.75), int(n_bars * 0.10)
    else:
        return None

    close = df["close"]
    try:
        deviation = compute_vwap_deviation(close, anchor_period=vwap_anchor)
    except Exception:
        return None

    if filter_name == "conf_donchian_pos_20":
        mask = build_confluence_disagreement_mask(df, "donchian_pos_20")
    elif filter_name == "conf_rsi_14_dev":
        mask = build_confluence_disagreement_mask(df, "rsi_14_dev")
    elif filter_name == "atr_only":
        mask = build_atr_regime_mask(df)
    elif filter_name == "no_filter":
        mask = pd.Series(True, index=df.index)
    else:
        return None

    pcts = TIER_GRIDS[grid_name]
    try:
        r = run_mr_wfo(close, deviation, mask, pcts, is_bars=is_bars, oos_bars=oos_bars)
    except Exception as e:
        print(f"    ERROR: {pair} v{vwap_anchor} {filter_name}/{grid_name}: {e}")
        return None
    if r.get("n_folds", 0) < MIN_FOLDS:
        return None
    return {
        "pair": pair,
        "vwap_anchor": vwap_anchor,
        "filter": filter_name,
        "grid": grid_name,
        "sharpe": r.get("stitched_sharpe", 0.0),
        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
        "n_folds": r.get("n_folds", 0),
        "pct_positive": r.get("pct_positive", 0.0),
        "n_trades": r.get("total_trades", 0),
        "bars": n_bars,
    }


def passes_bonferroni(r: dict) -> bool:
    return (
        r["ci_lo"] >= BONF_CI_LO
        and r["n_folds"] >= MIN_FOLDS
        and r["pct_positive"] >= MIN_POS
        and r["max_dd_pct"] >= -40.0
    )


def main() -> None:
    print("=" * 70)
    print("  Phase 2: EU FX H1 VWAP mean-reversion")
    print("=" * 70)
    t0 = time.time()

    rows: list[dict] = []
    combos = [(p, v, f, g) for p in PAIRS for v in VWAP_ANCHORS for f in FILTERS for g in TIER_GRIDS]
    print(f"  Combos: {len(combos)}")

    tested = 0
    for pair, v, flt, grd in combos:
        tested += 1
        r = run_config(pair, v, flt, grd)
        if r is None:
            continue
        rows.append(r)
        if tested % 10 == 0:
            print(
                f"  [{tested}/{len(combos)}] {pair} v={v} {flt}/{grd}  "
                f"collected {len(rows)}, {time.time() - t0:.0f}s"
            )

    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Tested {tested}, collected {len(df)}, {time.time() - t0:.0f}s")

    bonf = df[df.apply(passes_bonferroni, axis=1)] if not df.empty else df
    print(f"  Bonferroni survivors (ci_lo >= {BONF_CI_LO}): {len(bonf)}")
    print(f"  Max CI_lo: {df['ci_lo'].max():+.3f}" if not df.empty else "  (no rows)")

    lines: list[str] = [
        f"# Phase 2 — EU FX H1 MR ({len(df)} combos)",
        "",
        "Tests EUR/CHF, EUR/GBP, EUR/JPY with the same VWAP-confluence "
        "mean-reversion framework as the AUD/JPY champion. Reuses "
        "`run_mr_wfo` with sweeps over vwap_anchor {12, 24, 36}, filter "
        "{donchian/rsi disagreement, atr-only, no_filter}, and tier grid "
        "{standard, conservative}.",
        "",
        f"**Reference**: AUD/JPY at vwap_anchor=24 / conf_donchian_pos_20 / conservative "
        f"produces OOS Sharpe +1.05 (CI_lo +0.21).",
        "",
        f"**Gate**: CI_lo >= {BONF_CI_LO}, folds >= {MIN_FOLDS}, pos >= {int(MIN_POS * 100)}%.",
        "",
    ]

    # Per-pair summary
    lines.append("## Per-pair summary\n")
    lines.append("| Pair | Configs | Max Sharpe | Max CI_lo | Bonf |")
    lines.append("|---|--:|---:|---:|--:|")
    for p in PAIRS:
        sub = df[df["pair"] == p] if not df.empty else df
        if sub.empty:
            lines.append(f"| {p} | 0 | - | - | 0 |")
            continue
        nbonf = sub.apply(passes_bonferroni, axis=1).sum()
        lines.append(
            f"| {p} | {len(sub)} | {sub['sharpe'].max():+.3f} | "
            f"{sub['ci_lo'].max():+.3f} | {nbonf} |"
        )
    lines.append("")

    if not bonf.empty:
        lines.append("## Bonferroni survivors\n")
        lines.append("| # | Pair | vwap | Filter | Grid | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |")
        lines.append("|--:|---|--:|---|---|---:|---:|---:|---:|--:|--:|")
        for i, r in bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True).iterrows():
            lines.append(
                f"| {i + 1} | {r['pair']} | {int(r['vwap_anchor'])} | "
                f"{r['filter']} | {r['grid']} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )
        lines.append("")

    lines.append("## Top 10 by CI_lo\n")
    if not df.empty:
        top = df.sort_values("ci_lo", ascending=False).head(10).reset_index(drop=True)
        lines.append("| # | Pair | vwap | Filter | Grid | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |")
        lines.append("|--:|---|--:|---|---|---:|---:|---:|---:|--:|--:|")
        for i, r in top.iterrows():
            lines.append(
                f"| {i + 1} | {r['pair']} | {int(r['vwap_anchor'])} | "
                f"{r['filter']} | {r['grid']} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
