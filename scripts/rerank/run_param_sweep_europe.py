"""Cross-asset momentum parameter sweep on European equity targets.

Extends the US-target sweep (run_param_sweep.py) with DAX (^GDAXI) and
FTSE (^FTSE). Tests whether the cross-asset edge transmits across
currency / market boundaries.

Grid: 6 signals x 2 EU targets x 6 lookbacks x 4 holds x 4 thresholds
  = 1,152 combos. Bonferroni gate (CI_lo >= 0.5) at N=1152 means any
  survivor's point estimate must clear the corrected threshold.

Only combinations with a plausible economic story are tested — we
exclude e.g. UUP -> HYG (already in US sweep) and same-instrument
pairs.

Output:
  .tmp/reports/param_sweep_europe_2026_04_22/results.csv
  .tmp/reports/param_sweep_europe_2026_04_22/leaderboard.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "param_sweep_europe_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)

# Signals: same 6 as the US sweep — bond/credit/dollar momentum.
SIGNALS = ["TLT", "IEF", "HYG", "TIP", "LQD", "UUP"]
# Targets: European indices only.
TARGETS = ["GDAXI", "FTSE"]
LOOKBACKS = [5, 10, 15, 20, 40, 60]
HOLDS = [5, 10, 20, 40]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

# Bonferroni gate adjusted for N=1152 instead of 3360 — slightly
# looser on CI_lo but still rigorous.
BONF_CI_LO = 0.45
BONF_MIN_FOLDS = 25
BONF_MIN_POS = 0.60
BONF_MAX_DD = -40.0

PERM_CI_LO = 0.0
PERM_MIN_FOLDS = 25
PERM_MIN_POS = 0.60


def economic_story(signal: str, target: str) -> str:
    """Return a one-line causal mechanism for a signal->EU-equity trade."""
    if signal == "UUP" and target == "GDAXI":
        return "USD up -> EUR down -> German exports (BMW, Siemens) earn more -> DAX up"
    if signal == "UUP" and target == "FTSE":
        return "USD up -> GBP down -> FTSE multinationals (mostly USD-earners) up"
    if signal == "HYG" and target in ("GDAXI", "FTSE"):
        return "global credit spread -> risk-off spillover to EU equity"
    if signal == "TLT" and target in ("GDAXI", "FTSE"):
        return "US long rates -> global discount rate -> EU equity"
    if signal == "IEF" and target in ("GDAXI", "FTSE"):
        return "US intermediate rates -> global rate beta -> EU equity"
    if signal == "TIP" and target in ("GDAXI", "FTSE"):
        return "US inflation expectations -> global risk premium -> EU equity"
    if signal == "LQD" and target in ("GDAXI", "FTSE"):
        return "US IG credit momentum -> global credit cycle -> EU equity"
    return ""


def run_combo(
    signal: str, target: str, lookback: int, hold: int, threshold: float
) -> dict | None:
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    try:
        sc = load_daily(signal)
        tc = load_daily(target)
    except FileNotFoundError:
        return None
    try:
        r = run_bond_wfo(
            sc,
            tc,
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
        "strategy": "cross_asset_eu",
        "signal": signal,
        "target": target,
        "lookback": lookback,
        "hold": hold,
        "threshold": threshold,
        "sharpe": r.get("stitched_sharpe", 0.0),
        "ci_lo": r.get("sharpe_ci_95_lo", 0.0),
        "ci_hi": r.get("sharpe_ci_95_hi", 0.0),
        "max_dd_pct": r.get("stitched_dd_pct", 0.0),
        "n_folds": r.get("n_folds", 0),
        "pct_positive": r.get("pct_positive", 0.0),
        "n_trades": r.get("total_trades", 0),
    }


def passes_permissive(r: dict) -> bool:
    return (
        r["ci_lo"] > PERM_CI_LO
        and r["n_folds"] >= PERM_MIN_FOLDS
        and r["pct_positive"] >= PERM_MIN_POS
    )


def passes_bonferroni(r: dict) -> bool:
    return (
        r["ci_lo"] >= BONF_CI_LO
        and r["n_folds"] >= BONF_MIN_FOLDS
        and r["pct_positive"] >= BONF_MIN_POS
        and r["max_dd_pct"] >= BONF_MAX_DD
    )


def main() -> None:
    rows: list[dict] = []
    t0 = time.time()

    combos = []
    for sig in SIGNALS:
        for tgt in TARGETS:
            if not economic_story(sig, tgt):
                continue
            for lb in LOOKBACKS:
                for hold in HOLDS:
                    for th in THRESHOLDS:
                        combos.append((sig, tgt, lb, hold, th))
    print(f"  Combos to test: {len(combos)}")

    tested = 0
    for sig, tgt, lb, hold, th in combos:
        tested += 1
        row = run_combo(sig, tgt, lb, hold, th)
        if row is None:
            continue
        rows.append(row)
        if tested % 100 == 0:
            print(
                f"  [{tested}] last: {sig}->{tgt} lb={lb} hold={hold} "
                f"th={th} -- collected {len(rows)}"
            )
            pd.DataFrame(rows).to_csv(REPORT / "results.csv", index=False)

    pd.DataFrame(rows).to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Tested {tested}, collected {len(rows)}, {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    n_total = len(df)
    perm = df[df.apply(passes_permissive, axis=1)] if n_total else df
    bonf = df[df.apply(passes_bonferroni, axis=1)] if n_total else df

    print(f"\n  N_total: {n_total}")
    print(f"  Permissive gate: {len(perm)}")
    print(f"  Bonferroni gate (ci_lo>={BONF_CI_LO}): {len(bonf)}")

    # Top 20 by ci_lo for visibility regardless of gate pass.
    top20 = df.sort_values("ci_lo", ascending=False).head(20).reset_index(drop=True)

    lines: list[str] = []
    lines.append(f"# European Cross-Asset Parameter Sweep ({n_total} combos)\n")
    lines.append(
        "Tests whether US bond/credit/dollar signals transmit to European "
        "equity targets (DAX, FTSE) on daily timeframe with the same WFO "
        "harness as the US sweep.\n"
    )
    lines.append("## Gates\n")
    lines.append(f"- Permissive (ci_lo > 0, folds >= {PERM_MIN_FOLDS}): {len(perm)}")
    lines.append(
        f"- Bonferroni (ci_lo >= {BONF_CI_LO}, folds >= {BONF_MIN_FOLDS}, "
        f"pos >= 60%, dd >= -40%): **{len(bonf)}**\n"
    )

    if len(bonf) > 0:
        lines.append("## Bonferroni survivors\n")
        lines.append(
            "| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | "
            "CI_hi | DD | Folds | Pos% |"
        )
        lines.append(
            "|--:|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|"
        )
        bonf_sorted = bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True)
        for i, r in bonf_sorted.iterrows():
            lines.append(
                f"| {i + 1} | {r['signal']} | {r['target']} | "
                f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
                f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | {r['ci_hi']:+.3f} | "
                f"{r['max_dd_pct']:.1f}% | {int(r['n_folds'])} | "
                f"{int(r['pct_positive'] * 100)}% |"
            )
    else:
        lines.append("## No Bonferroni survivors\n")
        lines.append(
            "No (signal, target, params) combo cleared the Bonferroni gate "
            "on European targets. This is **informative**: either the US "
            "cross-asset edge does not transmit cleanly across currency/market "
            "boundaries, or the DAX/FTSE regime is genuinely different from "
            "US equity in its response to US macro signals.\n"
        )

    lines.append("\n## Top 20 by CI_lo (regardless of gate)\n")
    lines.append(
        "| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | "
        "CI_hi | DD | Folds | Pos% | Story |"
    )
    lines.append(
        "|--:|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|---|"
    )
    for i, r in top20.iterrows():
        lines.append(
            f"| {i + 1} | {r['signal']} | {r['target']} | "
            f"{int(r['lookback'])} | {int(r['hold'])} | {r['threshold']:.2f} | "
            f"{r['sharpe']:+.3f} | {r['ci_lo']:+.3f} | {r['ci_hi']:+.3f} | "
            f"{r['max_dd_pct']:.1f}% | {int(r['n_folds'])} | "
            f"{int(r['pct_positive'] * 100)}% | "
            f"{economic_story(r['signal'], r['target'])} |"
        )

    (REPORT / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
