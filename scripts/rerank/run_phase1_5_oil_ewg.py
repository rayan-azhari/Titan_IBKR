"""Phase 1.5 — Oil -> EWG follow-up.

Phase 1 found CL -> DAX at CI_lo +0.316, within 0.13 of the gate but
not deployable. The Europe v2 hedged-DAX result showed FX-translation
accounts for ~25% of the Sharpe drag on EU cross-asset trades. Test
whether using EWG (USD-denominated iShares Germany ETF) as target
lifts the oil channel above the Bonferroni gate.

Grid: 2 signals (CL=F, BZ=F) x 1 target (EWG) x 6 lookbacks x
4 holds x 4 thresholds = 192 combos. Wall-clock ~3 min.

Writes:
  .tmp/reports/phase1_5_oil_ewg_2026_04_22/results.csv
  .tmp/reports/phase1_5_oil_ewg_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "phase1_5_oil_ewg_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)

SIGNALS = ["CL=F", "BZ=F"]
TARGETS = ["EWG"]  # USD-denominated Germany ETF
LOOKBACKS = [5, 10, 15, 20, 40, 60]
HOLDS = [5, 10, 20, 40]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

BONF_CI_LO = 0.45
BONF_MIN_FOLDS = 25
BONF_MIN_POS = 0.60
BONF_MAX_DD = -40.0


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
    print("  Phase 1.5: Oil -> EWG (USD-denominated Germany)")
    print("=" * 70)
    t0 = time.time()

    sig_cache = {s: load_daily(s) for s in SIGNALS}
    tgt_cache = {t: load_daily(t) for t in TARGETS}

    rows: list[dict] = []
    tested = 0
    total = len(SIGNALS) * len(TARGETS) * len(LOOKBACKS) * len(HOLDS) * len(THRESHOLDS)
    print(f"  Combos: {total}")

    for sig in SIGNALS:
        for tgt in TARGETS:
            sc = sig_cache[sig]
            tc = tgt_cache[tgt]
            for lb in LOOKBACKS:
                for hold in HOLDS:
                    for th in THRESHOLDS:
                        tested += 1
                        r = run_combo(sc, tc, lb, hold, th)
                        if r is None:
                            continue
                        r.update(
                            {
                                "signal": sig,
                                "target": tgt,
                                "lookback": lb,
                                "hold": hold,
                                "threshold": th,
                            }
                        )
                        rows.append(r)
            print(f"  {sig}->{tgt}: collected {len(rows)} so far, {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Tested {tested}, collected {len(df)}, {time.time() - t0:.0f}s")

    bonf = df[df.apply(passes_bonferroni, axis=1)] if not df.empty else df
    print(f"  Bonferroni (ci_lo >= {BONF_CI_LO}): {len(bonf)}")
    print(f"  Max CI_lo: {df['ci_lo'].max():+.3f}")

    lines: list[str] = [
        f"# Phase 1.5 — Oil -> EWG ({len(df)} combos)",
        "",
        "Follow-up to Phase 1's CL -> DAX near-miss (CI_lo +0.316). "
        "Tests whether the USD-denominated EWG target lifts the signal "
        "above the Bonferroni gate (0.45), the way Europe v2 hedged-DAX "
        "did for UUP (+0.326 -> +0.425).",
        "",
    ]

    # Compare to Phase 1 CL->DAX
    lines.append("## Comparison to Phase 1 CL -> DAX\n")
    lines.append("| Config | Phase 1 (DAX) | Phase 1.5 (EWG) | Delta |")
    lines.append("|---|---:|---:|---:|")
    # Pick the best-matching config from Phase 1: CL lb=40 hold=20 th=0.25
    p1_cl_best = {"sharpe": 0.735, "ci_lo": 0.316}  # from Phase 1 leaderboard
    cl_ewg = df[(df.signal == "CL=F") & (df.lookback == 40) & (df.hold == 20) & (df.threshold == 0.25)]
    if not cl_ewg.empty:
        r = cl_ewg.iloc[0]
        lines.append(
            f"| CL lb=40 hold=20 th=0.25 | Sharpe +0.735 / CI_lo +0.316 | "
            f"Sharpe {r['sharpe']:+.3f} / CI_lo {r['ci_lo']:+.3f} | "
            f"Δ Sharpe {r['sharpe'] - p1_cl_best['sharpe']:+.3f}, "
            f"Δ CI_lo {r['ci_lo'] - p1_cl_best['ci_lo']:+.3f} |"
        )
    lines.append("")

    if not bonf.empty:
        lines.append("## Bonferroni survivors\n")
        lines.append(
            "| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |"
        )
        lines.append("|--:|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
        for i, r in bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True).iterrows():
            lines.append(
                f"| {i + 1} | {r['signal']} | {r['target']} | "
                f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
                f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
            )
    else:
        lines.append("## No Bonferroni survivors\n")
        lines.append("")

    lines.append("## Top 10 by CI_lo\n")
    top = df.sort_values("ci_lo", ascending=False).head(10).reset_index(drop=True)
    lines.append(
        "| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |"
    )
    lines.append("|--:|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|")
    for i, r in top.iterrows():
        lines.append(
            f"| {i + 1} | {r['signal']} | {r['target']} | "
            f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
            f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | "
            f"{r['ci_hi']:+.3f} | {r['max_dd_pct']:.1f}% | "
            f"{int(r['n_folds'])} | {int(r['pct_positive'] * 100)}% |"
        )

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
