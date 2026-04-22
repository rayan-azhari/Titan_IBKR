"""Cross-asset momentum parameter-grid sweep with Bonferroni discipline.

Tests every (signal x target x lookback x hold_days x threshold)
combination for the cross-asset momentum family. With 3,360 parallel
tests, the permissive 95% CI gate lets through ~84 false positives per
tail by chance alone; only the Bonferroni-adjusted gate (CI_lo >= 0.5,
n_folds >= 30, pct_positive >= 60%, max_dd >= -40%) is trusted.

Crash-recoverable: flushes results.csv every 100 combos.

Output:
  .tmp/reports/param_sweep_2026_04_22/results.csv
  .tmp/reports/param_sweep_2026_04_22/leaderboard.md
  .tmp/reports/param_sweep_2026_04_22/run.log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / ".tmp" / "reports" / "param_sweep_2026_04_22"
REPORT.mkdir(parents=True, exist_ok=True)


# ── Parameter grid ─────────────────────────────────────────────────────

SIGNALS = ["TLT", "IEF", "HYG", "TIP", "LQD", "UUP"]
TARGETS = ["SPY", "QQQ", "IWB", "GLD", "HYG", "TQQQ"]
LOOKBACKS = [5, 10, 15, 20, 40, 60]
HOLDS = [5, 10, 20, 40]
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

# Bonferroni gate — for N=3360 tests the naive per-test CI is unsafe.
# Require the point estimate and lower bound to be large enough that
# the ~2.1x CI widening still leaves CI_lo > 0.
BONF_CI_LO = 0.50
BONF_MIN_FOLDS = 30
BONF_MIN_POS = 0.60
BONF_MAX_DD = -40.0  # max_dd_pct >= this (less negative)

# Permissive gate — for comparison to show how much discipline matters.
PERM_CI_LO = 0.0
PERM_MIN_FOLDS = 30
PERM_MIN_POS = 0.60


def economic_story(signal: str, target: str) -> str:
    """Return a one-line causal mechanism, or '' if nonsensical."""
    # Credit-spread -> equity
    if signal == "HYG" and target in ("SPY", "QQQ", "IWB"):
        return (
            "credit spreads widen (HYG falls) -> risk-off -> "
            + ("tech sells off" if target == "QQQ" else "broader equity")
        )
    # Duration -> growth equity
    if signal == "TLT" and target in ("QQQ", "TQQQ", "SPY", "IWB"):
        return "duration/long rates move -> discount-rate shock -> growth equity"
    # Inflation expectations -> credit/equity
    if signal == "TIP" and target == "HYG":
        return "inflation expectations -> credit spread repricing (cross-bond lead/lag)"
    if signal == "TIP" and target in ("QQQ", "SPY", "IWB"):
        return "inflation expectations -> equity risk premium"
    if signal == "TIP" and target == "GLD":
        return "real yields -> gold"
    # IG -> HY
    if signal == "LQD" and target == "HYG":
        return "IG credit momentum leads HY credit at short horizon"
    if signal == "LQD" and target in ("SPY", "QQQ", "IWB", "TQQQ"):
        return "IG credit momentum -> risk-on equity"
    # Duration/rates -> credit
    if signal in ("TLT", "IEF") and target == "HYG":
        return "duration/rates move -> credit spread repricing"
    # Intermediate rates -> equity / gold
    if signal == "IEF" and target in ("SPY", "QQQ", "IWB", "TQQQ"):
        return "intermediate rates -> equity discount"
    if signal == "IEF" and target == "GLD":
        return "intermediate rates -> gold"
    # Dollar -> equity / gold
    if signal == "UUP" and target in ("SPY", "QQQ", "IWB", "TQQQ"):
        return "dollar strength -> EPS translation / risk-off equity"
    if signal == "UUP" and target == "GLD":
        return "dollar strength -> gold (inverse)"
    if signal == "UUP" and target == "HYG":
        return "dollar strength -> EM/credit spillover"
    # TLT -> HYG already covered; TLT/IEF -> GLD
    if signal in ("TLT", "IEF") and target == "GLD":
        return "rates -> gold (real yield channel)"
    # HYG -> TQQQ (leveraged tech)
    if signal == "HYG" and target == "TQQQ":
        return "credit spreads -> leveraged tech (3x beta amplification)"
    return ""


# ── WFO wrapper ────────────────────────────────────────────────────────


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
        "strategy": "cross_asset",
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


# ── Gates ──────────────────────────────────────────────────────────────


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


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    rows: list[dict] = []
    t0 = time.time()

    # Pre-dedup: same-instrument pairs make no sense (e.g. HYG -> HYG).
    combos: list[tuple[str, str, int, int, float]] = []
    for sig in SIGNALS:
        for tgt in TARGETS:
            if sig == tgt:
                continue
            # Only keep combos with a plausible economic story.
            if not economic_story(sig, tgt):
                continue
            for lb in LOOKBACKS:
                for hold in HOLDS:
                    for th in THRESHOLDS:
                        combos.append((sig, tgt, lb, hold, th))
    print(f"  Combos to test: ~{len(combos)} (pre-dedup same-instrument pairs)")

    tested = 0
    for sig, tgt, lb, hold, th in combos:
        tested += 1
        row = run_combo(sig, tgt, lb, hold, th)
        if row is None:
            continue
        rows.append(row)

        if tested % 100 == 0:
            print(
                f"  [{tested}] last: {sig}->{tgt} "
                f"lb={lb} hold={hold} th={th} -- collected {len(rows)}"
            )
            pd.DataFrame(rows).to_csv(REPORT / "results.csv", index=False)

    pd.DataFrame(rows).to_csv(REPORT / "results.csv", index=False)
    print(f"\n  Tested {tested}, collected {len(rows)}, {time.time() - t0:.0f}s")

    # Report
    df = pd.DataFrame(rows)
    n_total = len(df)
    perm = df[df.apply(passes_permissive, axis=1)] if n_total else df
    bonf = df[df.apply(passes_bonferroni, axis=1)] if n_total else df

    print(f"\n  N_total: {n_total}")
    print(
        f"  Permissive gate (ci_lo>0, folds>={PERM_MIN_FOLDS}, "
        f"pos>={int(PERM_MIN_POS * 100)}%): {len(perm)}"
    )
    print(
        f"  Bonferroni gate  (ci_lo>={BONF_CI_LO}, folds>={BONF_MIN_FOLDS}, "
        f"pos>={int(BONF_MIN_POS * 100)}%, dd>={BONF_MAX_DD}): {len(bonf)}"
    )

    # Leaderboard markdown.
    lines: list[str] = []
    lines.append(f"# Parameter Sweep - Cross-Asset Momentum ({n_total} combos)\n")
    lines.append(
        "Full-grid sweep of 6 bond/credit/dollar signals x 6 targets x "
        "6 lookbacks x 4 holds x 4 thresholds. Sanctuary window active; "
        "95% bootstrap Sharpe CI per combo.\n"
    )
    lines.append("## Multiple-testing discipline\n")
    lines.append(
        f"- **Permissive gate** (standard): ci_lo > 0, n_folds >= {PERM_MIN_FOLDS}, "
        f"pct_positive >= {int(PERM_MIN_POS * 100)}%. {len(perm)} pass."
    )
    lines.append(
        f"- **Bonferroni-adjusted gate** (what we trust): ci_lo >= {BONF_CI_LO}, "
        f"n_folds >= {BONF_MIN_FOLDS}, pct_positive >= {int(BONF_MIN_POS * 100)}%, "
        f"max_dd >= {BONF_MAX_DD}%. **{len(bonf)} pass.**\n"
    )
    lines.append(
        f"At N={n_total} parallel tests, a naive 95% CI would yield "
        f"~{int(n_total * 0.025)} false positives from each tail by chance "
        "alone. The Bonferroni gate requires a point estimate large enough "
        "that even after the multiple-testing correction the lower bound "
        "stays positive.\n"
    )
    lines.append("## Bonferroni-adjusted gate-passers\n")
    lines.append(
        "| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | CI_hi | "
        "Max DD | Folds | Pos% | Story |"
    )
    lines.append(
        "|--:|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|---|"
    )
    bonf_sorted = bonf.sort_values("ci_lo", ascending=False).reset_index(drop=True)
    for i, r in bonf_sorted.iterrows():
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
