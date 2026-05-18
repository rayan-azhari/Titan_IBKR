"""V3.7 portfolio review: Kelly + ERC + Crisis + Joint Ruin + Joint Evaluation.

Runs the V3.7 framework end-to-end on the current LIVE + CONDITIONAL stack:
  - GEM J5 P_hl60_vt05 (LIVE)
  - turtle CAT C3_peak (CONDITIONAL, CAT-scoped)

Outputs:
  1. Per-strategy fractional Kelly sizing (L65/L67).
  2. ERC allocator weights.
  3. Per-crisis stress test results.
  4. Joint risk-of-ruin at multiple weight combinations.
  5. 10-metric portfolio-vs-benchmark evaluation.
  6. Reconciliation: what should the live allocator weights be?

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/v37_portfolio_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.portfolio.joint_evaluation import (  # noqa: E402
    benchmark_6040_returns,
    build_portfolio,
    evaluate_portfolio_vs_benchmark,
    gem_j5_returns,
    turtle_cat_returns_daily,
)
from titan.research.framework import (  # noqa: E402
    assess_joint_ruin,
    compute_erc_weights,
    compute_kelly_fraction,
    run_crisis_stress,
)


def main() -> None:
    print("=" * 100)
    print("V3.7 PORTFOLIO REVIEW: Kelly + ERC + Crisis Stress + Joint Ruin + 10-Metric Matrix")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────
    # Load strategy returns.
    # ─────────────────────────────────────────────────────────────────────
    print("\n[load] strategy returns:")
    gem = gem_j5_returns().dropna()
    turtle_cat = turtle_cat_returns_daily().dropna()
    bench_6040 = benchmark_6040_returns().dropna()
    print(f"  GEM J5:     {len(gem)} bars,  {gem.index[0].date()} -> {gem.index[-1].date()}")
    print(
        f"  turtle CAT: {len(turtle_cat)} bars, {turtle_cat.index[0].date()} -> {turtle_cat.index[-1].date()}"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Per-strategy Kelly fractions.
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEP 1 — Per-strategy fractional Kelly sizing (0.25× full Kelly)")
    print("=" * 100)
    kelly_gem = compute_kelly_fraction(gem, periods_per_year=252, fractional=0.25)
    kelly_turtle = compute_kelly_fraction(turtle_cat, periods_per_year=252, fractional=0.25)
    print("\n[GEM J5]")
    print(kelly_gem.report())
    print("\n[turtle CAT]")
    print(kelly_turtle.report())

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: ERC allocator on the strategy pair.
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEP 2 — Equal Risk Contribution (ERC) allocator")
    print("=" * 100)
    erc = compute_erc_weights(
        {"gem_j5": gem, "turtle_cat": turtle_cat},
        target_vol_ann=0.08,  # target 8% portfolio vol
        periods_per_year=252,
    )
    print()
    print(erc.report())

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Crisis stress on each strategy + the portfolio.
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEP 3 — Named-crisis stress test")
    print("=" * 100)

    print("\n[crisis] GEM J5 alone:")
    crisis_gem = run_crisis_stress(gem, max_dd_threshold=0.20)
    print(crisis_gem.report())

    print("\n[crisis] turtle CAT alone:")
    crisis_turtle = run_crisis_stress(turtle_cat, max_dd_threshold=0.20)
    print(crisis_turtle.report())

    portfolio_current = build_portfolio(
        {"gem_j5": gem, "turtle_cat": turtle_cat},
        {"gem_j5": 0.70, "turtle_cat": 0.20},
    )
    print("\n[crisis] Current portfolio (GEM 70% + turtle 20% + 10% cash):")
    crisis_port = run_crisis_stress(portfolio_current, max_dd_threshold=0.15)
    print(crisis_port.report())

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Joint ruin at multiple weight combos (L65).
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEP 4 — Joint risk-of-ruin sensitivity")
    print("=" * 100)
    print()
    print(f"{'gem_w':>6} {'turt_w':>7} {'P_kill':>8} {'p95_DD':>9} {'passes':>8}")
    for gem_w, turt_w in [(0.70, 0.20), (0.65, 0.20), (0.70, 0.15), (0.50, 0.30)]:
        res = assess_joint_ruin(
            {"gem_j5": gem, "turtle_cat": turtle_cat},
            deployment_weights={"gem_j5": gem_w, "turtle_cat": turt_w},
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        print(
            f"{gem_w:>6.0%} {turt_w:>7.0%} {res.p_kill_trip:>7.2%} {res.p95_maxdd_at_size:>+8.2%} {str(res.passes()):>8}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: 10-metric portfolio-vs-benchmark.
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEP 5 — Portfolio (live weights) vs 60/40 SPY/IEF: 10-metric matrix")
    print("=" * 100)
    eval_port = evaluate_portfolio_vs_benchmark(
        portfolio_current,
        bench_6040,
        portfolio_label="GEM(70%) + turtle(20%)",
        benchmark_label="60/40 SPY/IEF",
    )
    print()
    print(eval_port.report())

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Recommended deployment weights.
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEP 6 — Recommended deployment")
    print("=" * 100)
    deploy_w = erc.deployment_weights()
    print("\nERC deployment weights (target 8% portfolio vol):")
    for n, w in deploy_w.items():
        print(f"  {n}: {w:.4%}")
    sum_w = sum(deploy_w.values())
    print(f"  total: {sum_w:.4%}  cash: {1.0 - sum_w:.4%}")

    print(f"\nVs current pre-set weights: GEM 70% + turtle 20% (verdict: {eval_port.verdict})")

    # Compute Kelly-aware suggested weights.
    if kelly_gem.passes_gate() and kelly_turtle.passes_gate():
        print("\nFractional-Kelly suggested weights (0.25x full Kelly):")
        print(f"  gem_j5     0.25x Kelly: {kelly_gem.fractional_weight:.4%}")
        print(f"  turtle_cat 0.25x Kelly: {kelly_turtle.fractional_weight:.4%}")


if __name__ == "__main__":
    main()
