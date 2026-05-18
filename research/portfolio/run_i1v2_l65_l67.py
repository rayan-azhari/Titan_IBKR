"""I1 v2 C6_smoothed -- L65 ruin + L67 portfolio inclusion gates.

Follow-on to `research/ewmac/run_i1v2_audit.py` (DEPLOY verdict 2026-05-17).
Per V3.7 framework: a DEPLOY verdict on the 5-axis decision matrix is
necessary but not sufficient. Before any live capital, the strategy must
clear:
    - L65 single-strategy ruin gate at proposed deployment weights
    - L65 joint ruin gate (with existing LIVE strategies)
    - L67 portfolio-vs-benchmark 10-metric matrix at the proposed allocation

Output: `.tmp/reports/i1v2_l65_l67/result_log.md`

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/run_i1v2_l65_l67.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ewmac.run_i1v2_audit import (  # noqa: E402
    EWMAC_CELLS,
    GATE_CELLS,
    _stitched_oos,
    load_panel,
    load_universe,
)
from research.portfolio.joint_evaluation import (  # noqa: E402
    benchmark_6040_returns,
    build_portfolio,
    evaluate_portfolio_vs_benchmark,
    gem_j5_returns,
    turtle_cat_returns_daily,
)
from titan.research.framework import (  # noqa: E402
    assess_joint_ruin,
    assess_strategy_ruin,
    defaults_for,  # noqa: E402
    slice_sanctuary,
)
from titan.research.framework.typology import StrategyClass  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "i1v2_l65_l67"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def i1v2_c6_returns() -> pd.Series:
    """Reproduce the I1 v2 C6_smoothed stitched OOS returns.

    Uses the same audit harness (`_stitched_oos`) to ensure parity with
    the deployment-verdict cell.
    """
    closes = load_universe()
    panel = load_panel()
    sanc = slice_sanctuary(closes, months=12)
    closes_v = sanc.visible
    d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(closes_v.index, d.wfo, bars_per_year=BARS_PER_YEAR["D"])
    ewmac_name, gate_cfg = GATE_CELLS["C6_smoothed"]
    stitched, _, _ = _stitched_oos(
        closes_v,
        panel,
        EWMAC_CELLS[ewmac_name],
        gate_cfg,
        folds,
    )
    stitched.index = pd.to_datetime(stitched.index).tz_localize(None).normalize()
    # Fold OOS slices can overlap at boundaries; collapse duplicates to one
    # observation per date (last fold's value wins). This matches how a live
    # paper run would record one PnL per day.
    stitched = stitched.groupby(stitched.index).last().sort_index()
    return stitched.rename("i1v2_c6")


def run_l65_single(rets: pd.Series, weights: list[float]) -> dict:
    """L65 single-strategy ruin at proposed weights."""
    out = {}
    for w in weights:
        ruin = assess_strategy_ruin(
            rets,
            deployment_weight=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        out[w] = {
            "p_kill": ruin.p_kill_trip,
            "p95_maxdd": ruin.p95_maxdd_at_size,
            "p50_maxdd": ruin.median_maxdd_at_size,
            "passes": ruin.passes(),
        }
    return out


def run_l65_joint(
    gem: pd.Series,
    turtle: pd.Series,
    i1v2: pd.Series,
    candidate_weights: list[dict[str, float]],
) -> dict:
    """L65 joint ruin for proposed multi-strategy mixes.

    Each entry in candidate_weights is a dict like {"gem": 0.75, "turtle": 0.15, "i1v2": 0.10}.
    """
    out = {}
    for w in candidate_weights:
        ruin = assess_joint_ruin(
            strategy_returns={"gem": gem, "turtle": turtle, "i1v2": i1v2},
            deployment_weights=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        out[tuple(sorted(w.items()))] = {
            "weights": dict(w),
            "p_kill": ruin.p_kill_trip,
            "p95_maxdd": ruin.p95_maxdd_at_size,
            "p50_maxdd": ruin.median_maxdd_at_size,
            "passes": ruin.passes(),
        }
    return out


def main() -> None:
    print("=" * 80)
    print("I1 v2 C6_smoothed -- L65 ruin + L67 portfolio inclusion")
    print("=" * 80)

    print("\n[1/4] Building strategy return series...")
    gem = gem_j5_returns().dropna()
    print(f"  gem_j5: {len(gem)} bars ({gem.index[0].date()} -> {gem.index[-1].date()})")
    turtle = turtle_cat_returns_daily().dropna()
    print(
        f"  turtle_cat: {len(turtle)} bars ({turtle.index[0].date()} -> {turtle.index[-1].date()})"
    )
    i1v2 = i1v2_c6_returns().dropna()
    print(f"  i1v2_c6: {len(i1v2)} bars ({i1v2.index[0].date()} -> {i1v2.index[-1].date()})")
    print(
        f"  i1v2_c6 stats: mean={i1v2.mean():+.6f}, std={i1v2.std():.6f}, "
        f"ann_sharpe={i1v2.mean() / i1v2.std() * ((252) ** 0.5):+.4f}"
    )

    print("\n[2/4] L65 single-strategy ruin for I1v2 C6...")
    candidate_w = [0.05, 0.10, 0.15]
    single = run_l65_single(i1v2, candidate_w)
    for w, r in single.items():
        verdict = "PASS" if r["passes"] else "FAIL"
        print(
            f"  w={w:.0%}: P_kill={r['p_kill'] * 100:.3f}%, "
            f"95th-pct DD={r['p95_maxdd'] * 100:.2f}%, 50th-pct DD={r['p50_maxdd'] * 100:.2f}%, "
            f"{verdict}"
        )

    print("\n[3/4] L65 joint ruin (GEM + Turtle + I1v2)...")
    # Candidate allocations: keep GEM dominant; shave from Turtle to fund I1v2.
    candidates = [
        {"gem": 0.80, "turtle": 0.20, "i1v2": 0.00},  # current LIVE for reference
        {"gem": 0.80, "turtle": 0.15, "i1v2": 0.05},
        {"gem": 0.75, "turtle": 0.15, "i1v2": 0.10},
        {"gem": 0.70, "turtle": 0.15, "i1v2": 0.15},
        {"gem": 0.75, "turtle": 0.10, "i1v2": 0.15},
    ]
    joint = run_l65_joint(gem, turtle, i1v2, candidates)
    for key, r in joint.items():
        w = r["weights"]
        verdict = "PASS" if r["passes"] else "FAIL"
        print(
            f"  {w}: P_kill={r['p_kill'] * 100:.3f}%, "
            f"95th-pct DD={r['p95_maxdd'] * 100:.2f}%, {verdict}"
        )

    print("\n[4/4] L67 10-metric portfolio inclusion test vs 60/40...")
    bench = benchmark_6040_returns().dropna()

    # Current LIVE portfolio (no I1v2).
    current = build_portfolio(
        {"gem": gem, "turtle": turtle},
        {"gem": 0.80, "turtle": 0.20},
    ).dropna()

    # Proposed portfolio with I1v2 included.
    # Pick the best joint-ruin-passing weight (typically 5-10% on a noise-best DEPLOY).
    proposed_weights_options = [
        {"gem": 0.80, "turtle": 0.15, "i1v2": 0.05},
        {"gem": 0.75, "turtle": 0.15, "i1v2": 0.10},
    ]
    print()
    eval_results: list[tuple[str, "object"]] = []
    eval_curr = evaluate_portfolio_vs_benchmark(
        current,
        bench,
        portfolio_label="CURRENT GEM80/Turtle20",
        benchmark_label="60/40 SPY/IEF",
    )
    print(eval_curr.report())
    eval_results.append(("CURRENT (no I1v2)", eval_curr))

    for w in proposed_weights_options:
        proposed = build_portfolio(
            {"gem": gem, "turtle": turtle, "i1v2": i1v2},
            w,
        ).dropna()
        label = f"PROPOSED GEM{int(w['gem'] * 100)}/Turtle{int(w['turtle'] * 100)}/I1v2{int(w['i1v2'] * 100)}"
        print()
        ev = evaluate_portfolio_vs_benchmark(
            proposed,
            bench,
            portfolio_label=label,
            benchmark_label="60/40 SPY/IEF",
        )
        print(ev.report())
        eval_results.append((label, ev))

    # Write result log.
    report_fp = REPORTS_DIR / "result_log.md"
    with report_fp.open("w", encoding="utf-8") as fh:
        fh.write("# I1 v2 C6_smoothed -- L65 + L67 Gates\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write(
            "**Pre-reg:** `directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate 2026-05-17.md`\n"
        )
        fh.write(
            "**Audit verdict:** C6_smoothed DEPLOY (Sharpe +0.52, CI_lo +0.049, noise=best)\n\n"
        )
        fh.write("## Strategy return series\n\n")
        fh.write(f"- gem_j5: {len(gem)} bars\n")
        fh.write(f"- turtle_cat: {len(turtle)} bars\n")
        fh.write(
            f"- i1v2_c6: {len(i1v2)} bars, ann Sharpe = {i1v2.mean() / i1v2.std() * ((252) ** 0.5):+.4f}\n\n"
        )
        fh.write("## L65 single-strategy ruin (I1v2 C6 standalone)\n\n")
        fh.write("| Weight | P_kill | 95th-pct DD | 50th-pct DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|:---:|\n")
        for w, r in single.items():
            fh.write(
                f"| {w:.0%} | {r['p_kill'] * 100:.3f}% | {r['p95_maxdd'] * 100:.2f}% | "
                f"{r['p50_maxdd'] * 100:.2f}% | {'PASS' if r['passes'] else 'FAIL'} |\n"
            )
        fh.write("\n## L65 joint ruin (GEM + Turtle + I1v2)\n\n")
        fh.write("| GEM | Turtle | I1v2 | P_kill | 95th-pct DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|---:|:---:|\n")
        for key, r in joint.items():
            w = r["weights"]
            fh.write(
                f"| {w['gem']:.0%} | {w['turtle']:.0%} | {w['i1v2']:.0%} | "
                f"{r['p_kill'] * 100:.3f}% | {r['p95_maxdd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )
        fh.write("\n## L67 10-metric portfolio inclusion test\n\n")
        for label, ev in eval_results:
            fh.write(f"### {label}\n\n")
            fh.write("```\n" + ev.report() + "\n```\n\n")
    print(f"\nResult log: {report_fp}")


if __name__ == "__main__":
    main()
