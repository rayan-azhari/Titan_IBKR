"""Phase 5 — Cross-continent equity lead-lag sweep.

Tests whether US equity momentum predicts EU equity (and vice versa).
The first direction is well-documented in the spillover literature
(Hamao-Masulis-Ng 1990); the reverse direction is rarely tested.

Universe:
  US: SPY, QQQ, IWB
  EU: GDAXI (^GDAXI), FTSE (^FTSE), EWG, EWU

Grid: 3 US x 4 EU x 2 directions = 24 directed pairs.
Per pair: 6 lookbacks x 4 holds x 4 thresholds = 96 combos.
Total: 2,304 combos.

Writes:
  .tmp/reports/phase5_cross_continent_2026_04_22/results.csv
  .tmp/reports/phase5_cross_continent_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "phase5_cross_continent_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)

US = ["SPY", "QQQ", "IWB"]
EU = ["GDAXI", "FTSE", "EWG", "EWU"]

LOOKBACKS = [5, 10, 15, 20, 40, 60]
HOLDS = [5, 10, 20, 40]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

BONF_CI_LO = 0.45
BONF_MIN_FOLDS = 25
BONF_MIN_POS = 0.60
BONF_MAX_DD = -40.0


def continent(sym: str) -> str:
    return "US" if sym in US else "EU"


def run_combo(
    signal_close: pd.Series,
    target_close: pd.Series,
    lookback: int,
    hold: int,
    threshold: float,
) -> dict | None:
    from research.cross_asset.run_bond_equity_wfo import run_bond_wfo

    try:
        r = run_bond_wfo(
            signal_close,
            target_close,
            lookback=lookback,
            hold_days=hold,
            threshold=threshold,
            is_days=504,
            oos_days=126,
            spread_bps=5.0,
        )
    except Exception:
        return None
    if r.get("n_folds", 0) < 5:
        return None
    return {
        "sharpe": r.get("stitched_sharpe", 0.0),
        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
        "n_folds": r.get("n_folds", 0),
        "pct_positive": r.get("pct_positive", 0.0),
        "n_trades": r.get("total_trades", 0),
    }


def passes_bonferroni(r: dict) -> bool:
    return (
        r["ci_lo"] >= BONF_CI_LO
        and r["n_folds"] >= BONF_MIN_FOLDS
        and r["pct_positive"] >= BONF_MIN_POS
        and r["max_dd_pct"] >= BONF_MAX_DD
    )


def main() -> None:
    from research.cross_asset.run_bond_equity_wfo import load_daily

    print("=" * 70)
    print("  Phase 5: Cross-continent equity lead-lag sweep")
    print("=" * 70)
    t0 = time.time()

    cache: dict[str, pd.Series] = {}
    for s in US + EU:
        x = load_daily(s)
        cache[s] = x[~x.index.duplicated(keep="last")].sort_index()

    # Build cross-continent pairs (both directions).
    pairs: list[tuple[str, str]] = []
    for a in US:
        for b in EU:
            pairs.append((a, b))  # US -> EU
            pairs.append((b, a))  # EU -> US
    print(f"  Pairs: {len(pairs)}")
    print(f"  Configs per pair: {len(LOOKBACKS) * len(HOLDS) * len(THRESHOLDS)}")
    print(f"  Total combos: {len(pairs) * len(LOOKBACKS) * len(HOLDS) * len(THRESHOLDS)}")

    rows: list[dict] = []
    tested = 0
    for sig, tgt in pairs:
        for lb in LOOKBACKS:
            for hold in HOLDS:
                for th in THRESHOLDS:
                    tested += 1
                    r = run_combo(cache[sig], cache[tgt], lb, hold, th)
                    if r is None:
                        continue
                    r.update(
                        {
                            "signal": sig,
                            "target": tgt,
                            "direction": f"{continent(sig)}->{continent(tgt)}",
                            "lookback": lb,
                            "hold": hold,
                            "threshold": th,
                        }
                    )
                    rows.append(r)
        print(
            f"  [{tested}/{len(pairs) * 96}] {sig}->{tgt} done  "
            f"collected {len(rows)}  {time.time() - t0:.0f}s"
        )

    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Tested {tested}, collected {len(df)}, {time.time() - t0:.0f}s")

    bonf = df[df.apply(passes_bonferroni, axis=1)] if not df.empty else df
    print(f"  Bonferroni survivors (ci_lo >= {BONF_CI_LO}): {len(bonf)}")
    print(f"  Max CI_lo: {df['ci_lo'].max():+.3f}")

    # ── Leaderboard ────────────────────────────────────────────────────
    lines: list[str] = [
        f"# Phase 5 — Cross-Continent Equity Lead-Lag ({len(df)} combos)",
        "",
        f"Tests US <-> EU equity momentum in both directions. "
        f"Universe: US = {US}, EU = {EU}.",
        "",
        f"**Gate**: CI_lo >= {BONF_CI_LO}, folds >= {BONF_MIN_FOLDS}, "
        f"pos >= {int(BONF_MIN_POS * 100)}%, DD >= {BONF_MAX_DD}%.",
        "",
    ]

    # Direction summary
    lines.append("## Direction summary\n")
    lines.append("| Direction | Configs | Max Sharpe | Max CI_lo | CI_lo > 0 | Bonf |")
    lines.append("|---|--:|---:|---:|--:|--:|")
    for direction in ("US->EU", "EU->US"):
        sub = df[df["direction"] == direction]
        if sub.empty:
            lines.append(f"| {direction} | 0 | - | - | 0 | 0 |")
            continue
        nb = sub.apply(passes_bonferroni, axis=1).sum()
        lines.append(
            f"| {direction} | {len(sub)} | {sub['sharpe'].max():+.3f} | "
            f"{sub['ci_lo'].max():+.3f} | {(sub['ci_lo'] > 0).sum()} | {nb} |"
        )
    lines.append("")

    # Per-pair summary
    lines.append("## Per-pair summary (top 5 pairs by max CI_lo)\n")
    lines.append("| Signal | Target | Direction | Max Sharpe | Max CI_lo | CI_lo > 0 |")
    lines.append("|---|---|---|---:|---:|--:|")
    pair_stats = (
        df.groupby(["signal", "target", "direction"])["ci_lo"].max().reset_index()
        .sort_values("ci_lo", ascending=False).head(10)
    )
    for _, r in pair_stats.iterrows():
        sub = df[(df["signal"] == r["signal"]) & (df["target"] == r["target"])]
        lines.append(
            f"| {r['signal']} | {r['target']} | {r['direction']} | "
            f"{sub['sharpe'].max():+.3f} | {r['ci_lo']:+.3f} | "
            f"{(sub['ci_lo'] > 0).sum()} |"
        )
    lines.append("")

    if not bonf.empty:
        lines.append("## Bonferroni survivors\n")
        lines.append("| # | Signal | Target | Direction | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |")
        lines.append("|--:|---|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
        for i, r in bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True).iterrows():
            lines.append(
                f"| {i + 1} | {r['signal']} | {r['target']} | {r['direction']} | "
                f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | {r['ci_hi']:+.3f} | "
                f"{r['max_dd_pct']:.1f}% | {int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )
    else:
        lines.append("## No Bonferroni survivors\n")
    lines.append("")

    lines.append("## Top 10 by CI_lo\n")
    if not df.empty:
        top = df.sort_values("ci_lo", ascending=False).head(10).reset_index(drop=True)
        lines.append("| # | Signal | Target | Direction | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |")
        lines.append("|--:|---|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
        for i, r in top.iterrows():
            lines.append(
                f"| {i + 1} | {r['signal']} | {r['target']} | {r['direction']} | "
                f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | {r['ci_hi']:+.3f} | "
                f"{r['max_dd_pct']:.1f}% | {int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
