"""IBGM (Bund 7-10y UCITS) -> XDAX (DAX UCITS) parameter sweep.

EU analog of the live IHYU -> CSPX strategy. Same WFO machinery, swept
over a small principled grid with Bonferroni-corrected gate.

Grid (3x3x3 = 27 cells, kept tight to limit multiple-testing inflation):
  lookback    : [5, 10, 20]   bond-momentum window in days
  hold_days   : [5, 10, 20]   minimum hold after entry
  threshold   : [0.25, 0.50, 1.00]  z-score entry threshold

Gates:
  Unadjusted: Sharpe > 0.5 AND CI lo > 0
  Bonferroni: with N=27 tests, α=0.05 -> per-test α=0.0019 -> need ~99.8%
              bootstrap-CI lower bound > 0. We approximate via Sharpe > ~0.7
              and CI lo (at 95%) >= 0.3 — conservative but robust.

Sanctuary: 2025-01-01+ held out from gate evaluation (visible only as a
falsification check after a winner is picked).

Run: PYTHONUTF8=1 uv run python research/carver_trend/run_eu_bond_equity_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo  # noqa: E402

LOOKBACKS = [5, 10, 20]
HOLDS = [5, 10, 20]
THRESHOLDS = [0.25, 0.50, 1.00]

# Bonferroni gate (conservative, applied to UNADJUSTED 95% CI lo).
BONF_CI_LO_GATE = 0.30  # Approx Bonferroni-equivalent for N=27 tests
BONF_SHARPE_GATE = 0.70  # Tightened from 0.5 to compensate for selection
MIN_FOLDS = 20
MIN_POS = 0.55

SANCTUARY_START = pd.Timestamp("2025-01-01", tz="UTC")


def main() -> None:
    print("=" * 92)
    print("IBGM -> XDAX  bond-equity rotation parameter sweep")
    print("=" * 92)

    # Load full series (sanctuary will be trimmed inside each WFO call by the
    # IS+OOS rolling window). For the gate eval we exclude post-sanctuary
    # bars from input.
    ibgm = load_daily("IBGM")
    xdax = load_daily("XDAX")

    ibgm_eval = ibgm[ibgm.index < SANCTUARY_START]
    xdax_eval = xdax[xdax.index < SANCTUARY_START]
    print(
        f"  IBGM eval bars : {len(ibgm_eval):,} ({ibgm_eval.index[0].date()}..{ibgm_eval.index[-1].date()})"
    )
    print(f"  XDAX eval bars : {len(xdax_eval):,}")
    print(f"  Sanctuary start: {SANCTUARY_START.date()} (held out)")
    print(
        f"  Grid           : {len(LOOKBACKS) * len(HOLDS) * len(THRESHOLDS)} cells "
        f"({len(LOOKBACKS)} lb x {len(HOLDS)} hold x {len(THRESHOLDS)} th)"
    )
    print(
        f"  Bonferroni gate: Sharpe > {BONF_SHARPE_GATE}, CI lo > {BONF_CI_LO_GATE}, "
        f"folds >= {MIN_FOLDS}, pos% >= {MIN_POS:.0%}"
    )
    print()
    print(
        f"{'lb':>3} {'hold':>4} {'th':>5}  | {'Sharpe':>7} {'CI_lo':>7} {'CI_hi':>7}"
        f"  | {'DD%':>6} {'Pos%':>5} {'Folds':>5} {'Trd':>4}  {'Gate':>6}"
    )
    print("-" * 92)

    results: list[dict] = []
    for lb in LOOKBACKS:
        for hd in HOLDS:
            for th in THRESHOLDS:
                r = run_bond_wfo(
                    ibgm_eval,
                    xdax_eval,
                    lookback=lb,
                    hold_days=hd,
                    threshold=th,
                    is_days=504,
                    oos_days=126,
                )
                sh = r.get("stitched_sharpe", 0.0)
                ci_lo = r.get("sharpe_ci_95_lo", 0.0)
                ci_hi = r.get("sharpe_ci_95_hi", 0.0)
                dd = r.get("stitched_dd_pct", 0.0)
                pos = r.get("pct_positive", 0.0)
                folds = r.get("n_folds", 0)
                trades = r.get("total_trades", 0)
                ret = r.get("stitched_ret_pct", 0.0)

                gate_sh = sh >= BONF_SHARPE_GATE
                gate_ci = ci_lo >= BONF_CI_LO_GATE
                gate_folds = folds >= MIN_FOLDS
                gate_pos = pos >= MIN_POS
                passed = gate_sh and gate_ci and gate_folds and gate_pos
                marker = "PASS" if passed else " "

                results.append(
                    {
                        "lookback": lb,
                        "hold": hd,
                        "threshold": th,
                        "sharpe": sh,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "dd": dd,
                        "pos": pos,
                        "folds": folds,
                        "trades": trades,
                        "ret": ret,
                        "passed": passed,
                    }
                )
                print(
                    f"{lb:>3} {hd:>4} {th:>5.2f}  | "
                    f"{sh:>+7.3f} {ci_lo:>+7.3f} {ci_hi:>+7.3f}"
                    f"  | {dd:>+6.1f} {pos * 100:>5.0f}% {folds:>5} {trades:>4}  "
                    f"{marker:>6}"
                )

    print()
    print("=" * 92)
    print("SWEEP SUMMARY")
    print("=" * 92)

    df = pd.DataFrame(results)
    n_passed = df["passed"].sum()
    print(f"  Cells testing : {len(df)}")
    print(f"  Cells passing : {n_passed} ({n_passed / len(df) * 100:.0f}%)")
    print("  Best by CI_lo : ", end="")
    best_ci = df.loc[df["ci_lo"].idxmax()]
    print(
        f"lb={int(best_ci['lookback'])} hold={int(best_ci['hold'])} "
        f"th={best_ci['threshold']:.2f} -> Sharpe {best_ci['sharpe']:+.3f}, "
        f"CI lo {best_ci['ci_lo']:+.3f}"
    )
    print("  Best by Sharpe: ", end="")
    best_sh = df.loc[df["sharpe"].idxmax()]
    print(
        f"lb={int(best_sh['lookback'])} hold={int(best_sh['hold'])} "
        f"th={best_sh['threshold']:.2f} -> Sharpe {best_sh['sharpe']:+.3f}, "
        f"CI lo {best_sh['ci_lo']:+.3f}"
    )

    if n_passed > 0:
        # Sanctuary check on the best Bonferroni-passing cell
        best = df[df["passed"]].sort_values("ci_lo", ascending=False).iloc[0]
        print()
        print(
            f"  Sanctuary check on best PASS cell: lb={int(best['lookback'])}, "
            f"hold={int(best['hold'])}, th={best['threshold']:.2f}"
        )
        r_sanc = run_bond_wfo(
            ibgm,
            xdax,  # full data including sanctuary
            lookback=int(best["lookback"]),
            hold_days=int(best["hold"]),
            threshold=float(best["threshold"]),
            is_days=504,
            oos_days=126,
        )
        # Compute sanctuary-only stats
        rets_full = r_sanc.get("stitched_returns", pd.Series(dtype=float))
        if len(rets_full):
            sanc = rets_full[rets_full.index >= SANCTUARY_START]
            if len(sanc) > 5:
                from titan.research.metrics import bootstrap_sharpe_ci, max_drawdown, sharpe

                sanc_sh = sharpe(sanc, periods_per_year=252)
                sanc_lo, sanc_hi = bootstrap_sharpe_ci(sanc, periods_per_year=252, n_resamples=2000)
                sanc_dd = max_drawdown(sanc)
                sanc_total = float((1.0 + sanc).prod() - 1.0)
                print(
                    f"    bars: {len(sanc)} | Sharpe {sanc_sh:+.3f}  CI [{sanc_lo:+.3f}, {sanc_hi:+.3f}]"
                    f"  DD {sanc_dd * 100:+.1f}%  total {sanc_total * 100:+.1f}%"
                )
                if sanc_sh > 0:
                    print("    sanctuary: POSITIVE — strategy survives held-out window")
                else:
                    print("    sanctuary: NEGATIVE — strategy degraded on held-out window")
            else:
                print("    insufficient sanctuary bars")
    else:
        print()
        print("  No cells passed the Bonferroni-corrected gate.")
        print("  Best unadjusted result is shown above; treat as tier=unconfirmed.")
    print("=" * 92)


if __name__ == "__main__":
    main()
