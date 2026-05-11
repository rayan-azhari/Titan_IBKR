"""Phase 3 of the EU strategy plan — developed-market country rotation.

Universe: 8 US-listed country ETFs (EWG Germany, EWU UK, EWP Spain,
EWI Italy, EWQ France, EWY S.Korea, EWC Canada, EWJ Japan).

Grid:
  * lookback in {63, 126, 252} (3m, 6m, 12m)
  * top_k in {1, 2, 3}
  * bottom_k in {0 (long-only), 1, 2}  - not all combos tested
  * rebalance in {21, 42, 63} (monthly, bi-monthly, quarterly)

PIIGS vs Core spread: a specialisation of the same framework with
long=[EWP, EWI], short=[EWG, EWU] baseline (fixed, not rank-based).

Writes:
  .tmp/reports/phase3_country_rotation_2026_04_22/results.csv
  .tmp/reports/phase3_country_rotation_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "phase3_country_rotation_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["EWG", "EWU", "EWP", "EWI", "EWQ", "EWY", "EWC", "EWJ"]

LOOKBACKS = [63, 126, 252]
TOP_K = [1, 2, 3]
BOTTOM_K = [0, 1, 2]
REBALANCE = [21, 42, 63]

BONF_CI_LO = 0.45
MIN_FOLDS = 5
MIN_POS = 0.55


def load_universe() -> dict[str, pd.Series]:
    from research.cross_asset.run_bond_equity_wfo import load_daily

    out: dict[str, pd.Series] = {}
    for s in UNIVERSE:
        try:
            x = load_daily(s)
            x = x[~x.index.duplicated(keep="last")].sort_index()
            out[s] = x
        except FileNotFoundError:
            print(f"  missing {s}")
    return out


def passes_bonferroni(r: dict) -> bool:
    return (
        r["ci_lo"] >= BONF_CI_LO
        and r["n_folds"] >= MIN_FOLDS
        and r["pct_positive"] >= MIN_POS
        and r["max_dd_pct"] >= -40.0
    )


def main() -> None:
    from research.cross_sectional.country_momentum import run_country_wfo

    print("=" * 70)
    print("  Phase 3: Country rotation (8-country developed-market universe)")
    print("=" * 70)
    t0 = time.time()

    instruments = load_universe()
    print(f"\n  Loaded {len(instruments)} instruments")
    common_start = max(s.index[0] for s in instruments.values())
    common_end = min(s.index[-1] for s in instruments.values())
    print(f"  Common range: {common_start.date()} -> {common_end.date()}")

    rows: list[dict] = []
    combos = []
    for lb, tk, bk, rb in itertools.product(LOOKBACKS, TOP_K, BOTTOM_K, REBALANCE):
        # Require top_k + bottom_k <= universe size.
        if tk + bk > len(UNIVERSE) - 1:
            continue
        combos.append((lb, tk, bk, rb))
    print(f"  Configs: {len(combos)}")

    for i, (lb, tk, bk, rb) in enumerate(combos, 1):
        direction = "long-only" if bk == 0 else f"long-top-{tk} / short-bottom-{bk}"
        try:
            r = run_country_wfo(
                instruments,
                lookback=lb,
                top_k=tk,
                bottom_k=bk,
                rebalance_days=rb,
                is_days=504,
                oos_days=252,
                cost_bps=5.0,
            )
        except Exception as e:
            print(f"    ERROR lb={lb} tk={tk} bk={bk} rb={rb}: {e}")
            continue
        rows.append(
            {
                "lookback": lb,
                "top_k": tk,
                "bottom_k": bk,
                "rebalance": rb,
                "direction": direction,
                "sharpe": r["stitched_sharpe"],
                "ci_lo": r["sharpe_ci_95_lo"],
                "ci_hi": r["sharpe_ci_95_hi"],
                "max_dd_pct": r["stitched_dd_pct"],
                "n_folds": r["n_folds"],
                "pct_positive": r["pct_positive"],
                "n_trades": r["total_trades"],
            }
        )
        if i % 5 == 0:
            print(
                f"  [{i}/{len(combos)}] lb={lb} tk={tk} bk={bk} rb={rb}  "
                f"Sharpe {r['stitched_sharpe']:+.3f}  CI_lo {r['sharpe_ci_95_lo']:+.3f}  "
                f"{time.time() - t0:.0f}s"
            )

    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Total configs: {len(df)}, wall-clock: {time.time() - t0:.0f}s")

    bonf = df[df.apply(passes_bonferroni, axis=1)] if not df.empty else df
    print(f"  Bonferroni survivors (ci_lo >= {BONF_CI_LO}): {len(bonf)}")
    print(f"  Max CI_lo: {df['ci_lo'].max():+.3f}" if not df.empty else "  (no rows)")

    # ── Leaderboard ────────────────────────────────────────────────────
    lines: list[str] = [
        f"# Phase 3 — Country Rotation ({len(df)} configs)",
        "",
        f"Universe: {', '.join(UNIVERSE)} ({len(UNIVERSE)} developed-market country ETFs).",
        f"Common window: {common_start.date()} -> {common_end.date()}.",
        "",
        "Sweep: lookback {63d, 126d, 252d} x top_k {1, 2, 3} x "
        "bottom_k {0, 1, 2} x rebalance {21d, 42d, 63d}.",
        "",
        f"**Gate**: CI_lo >= {BONF_CI_LO}, folds >= {MIN_FOLDS}, pos >= {int(MIN_POS * 100)}%.",
        "",
    ]

    if not bonf.empty:
        lines.append("## Bonferroni survivors\n")
        lines.append(
            "| # | LB | Top-k | Bot-k | Rebal | Direction | Sharpe | CI_lo | "
            "CI_hi | DD | Folds | Pos% | Trades |"
        )
        lines.append("|--:|--:|--:|--:|--:|---|---:|---:|---:|---:|--:|--:|--:|")
        for i, r in bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True).iterrows():
            lines.append(
                f"| {i + 1} | {int(r['lookback'])} | {int(r['top_k'])} | "
                f"{int(r['bottom_k'])} | {int(r['rebalance'])} | {r['direction']} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% | "
                f"{int(r['n_trades'])} |"
            )
        lines.append("")
    else:
        lines.append("## No Bonferroni survivors\n")
        lines.append("")

    lines.append("## Top 10 by CI_lo\n")
    if not df.empty:
        top = df.sort_values("ci_lo", ascending=False).head(10).reset_index(drop=True)
        lines.append(
            "| # | LB | Top-k | Bot-k | Rebal | Direction | Sharpe | CI_lo | "
            "CI_hi | DD | Folds | Pos% | Trades |"
        )
        lines.append("|--:|--:|--:|--:|--:|---|---:|---:|---:|---:|--:|--:|--:|")
        for i, r in top.iterrows():
            lines.append(
                f"| {i + 1} | {int(r['lookback'])} | {int(r['top_k'])} | "
                f"{int(r['bottom_k'])} | {int(r['rebalance'])} | {r['direction']} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% | "
                f"{int(r['n_trades'])} |"
            )

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
