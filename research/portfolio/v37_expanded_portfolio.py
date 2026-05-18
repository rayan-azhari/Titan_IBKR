"""V3.7 expanded portfolio inclusion test (Option A, 2026-05-17).

Tests adding the two Wave-B CONDITIONAL candidates (fx_carry AUD/JPY +
ic_equity_daily top-3) to the existing live + conditional stack
(GEM J5 70% + turtle CAT 20%) on the L67 10-metric matrix and joint L65
ruin.

Decision question:
    Does adding fx_carry + ic_equity_daily at small weights IMPROVE the
    portfolio's 10-metric verdict vs 60/40 SPY/IEF?

    If yes: expand the live pipeline.
    If no: hold the new candidates at CONDITIONAL_CANDIDATE status.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/v37_expanded_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.exploration.audit_fx_carry import (  # noqa: E402
    _load_d as _load_d_fx,
)
from research.exploration.audit_fx_carry import (
    fx_carry_returns,
)
from research.exploration.audit_ic_equity_daily import (  # noqa: E402
    _load_d as _load_d_eq,
)
from research.exploration.audit_ic_equity_daily import (
    ic_equity_returns,
)
from research.portfolio.joint_evaluation import (  # noqa: E402
    benchmark_6040_returns,
    build_portfolio,
    evaluate_portfolio_vs_benchmark,
    gem_j5_returns,
    turtle_cat_returns_daily,
)
from titan.research.framework import assess_joint_ruin  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "v37_expanded_portfolio"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def fx_carry_audjpy_returns() -> pd.Series:
    """AUD/JPY live config returns from the audit_fx_carry module."""
    df = _load_d_fx("AUD_JPY")
    return fx_carry_returns(df, sma_period=50, vol_target=0.08).rename("fx_carry_audjpy")


def ic_equity_top3_returns(common_oos_cutoff: str = "2021-01-01") -> pd.Series:
    """Top-3 ic_equity strict-OOS returns (HWM + WMT + SYK), equal-weight.

    Uses a COMMON calendar OOS window (default: from 2021-01-01) so all
    3 tickers contribute aligned-date observations. IS = data before cutoff
    per ticker (used for ic_sign + mu/sigma fit); OOS reported is the
    post-cutoff window.
    """
    parts = {}
    thresholds = {"HWM": 0.25, "WMT": 0.50, "SYK": 0.50}
    cutoff_ts = pd.Timestamp(common_oos_cutoff)
    for tkr, thr in thresholds.items():
        df = _load_d_eq(tkr)
        # Find the IS split as the index position closest to cutoff_ts.
        try:
            is_split = int(df.index.get_indexer([cutoff_ts], method="nearest")[0])
        except KeyError:
            is_split = len(df) // 2
        ret_full = ic_equity_returns(df, threshold=thr, is_until_idx=is_split)
        oos = ret_full.iloc[is_split:]
        parts[tkr] = oos
    # Union of dates, equal-weight per available ticker.
    union_idx = None
    for r in parts.values():
        if union_idx is None:
            union_idx = r.index
        else:
            union_idx = union_idx.union(r.index)
    if union_idx is None or len(union_idx) == 0:
        return pd.Series(dtype=float, name="ic_equity_top3")
    portfolio = pd.Series(0.0, index=union_idx)
    for _tkr, r in parts.items():
        portfolio = portfolio + r.reindex(union_idx).fillna(0.0) / 3.0
    return portfolio.rename("ic_equity_top3")


def main() -> None:
    print("=" * 100)
    print("V3.7 EXPANDED PORTFOLIO INCLUSION TEST")
    print("Testing: GEM 70% + turtle 20% + fx_carry 5% + ic_equity 5% vs 60/40")
    print("=" * 100)

    # Load all 4 strategy returns.
    print("\n[load] strategy returns:")
    gem = gem_j5_returns().dropna()
    turtle = turtle_cat_returns_daily().dropna()
    fx_carry = fx_carry_audjpy_returns().dropna()
    ic_eq = ic_equity_top3_returns().dropna()
    bench_6040 = benchmark_6040_returns().dropna()

    print(f"  GEM J5:           {len(gem)} bars, {gem.index[0].date()} -> {gem.index[-1].date()}")
    print(
        f"  turtle CAT:       {len(turtle)} bars, {turtle.index[0].date()} -> {turtle.index[-1].date()}"
    )
    print(
        f"  fx_carry AUD/JPY: {len(fx_carry)} bars, {fx_carry.index[0].date()} -> {fx_carry.index[-1].date()}"
    )
    print(
        f"  ic_equity top-3:  {len(ic_eq)} bars, {ic_eq.index[0].date()} -> {ic_eq.index[-1].date()}"
    )

    # ─────────────────────────────────────────────────────────────────
    # Empirical correlation matrix across all 4 strategies (on overlap)
    # ─────────────────────────────────────────────────────────────────
    common = (
        gem.index.intersection(turtle.index).intersection(fx_carry.index).intersection(ic_eq.index)
    )
    print(f"\n[align] common index: {len(common)} bars ({common[0].date()} -> {common[-1].date()})")
    df = pd.DataFrame(
        {
            "gem_j5": gem.reindex(common).fillna(0.0),
            "turtle_cat": turtle.reindex(common).fillna(0.0),
            "fx_carry": fx_carry.reindex(common).fillna(0.0),
            "ic_equity": ic_eq.reindex(common).fillna(0.0),
        }
    )
    print("\nCorrelation matrix (overlap period):")
    print(df.corr().round(3).to_string())

    # ─────────────────────────────────────────────────────────────────
    # Scenario 1: Current portfolio (GEM 70% + turtle 20%)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("Scenario 1: CURRENT (GEM 70% + turtle 20%) — for comparison")
    print("─" * 100)
    p1 = build_portfolio(
        {"gem_j5": gem, "turtle_cat": turtle},
        {"gem_j5": 0.70, "turtle_cat": 0.20},
    )
    eval1 = evaluate_portfolio_vs_benchmark(
        p1,
        bench_6040,
        portfolio_label="GEM(70%) + turtle(20%)",
        benchmark_label="60/40 SPY/IEF",
    )
    print(eval1.report())

    # ─────────────────────────────────────────────────────────────────
    # Scenario 2: Expanded (GEM 70% + turtle 20% + fx_carry 5% + ic_eq 5%)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("Scenario 2: EXPANDED (GEM 70% + turtle 20% + fx_carry 5% + ic_equity 5%)")
    print("─" * 100)
    p2 = build_portfolio(
        {"gem_j5": gem, "turtle_cat": turtle, "fx_carry": fx_carry, "ic_equity": ic_eq},
        {"gem_j5": 0.70, "turtle_cat": 0.20, "fx_carry": 0.05, "ic_equity": 0.05},
    )
    eval2 = evaluate_portfolio_vs_benchmark(
        p2,
        bench_6040,
        portfolio_label="GEM(70%)+turtle(20%)+fx(5%)+ic_eq(5%)",
        benchmark_label="60/40 SPY/IEF",
    )
    print(eval2.report())

    # ─────────────────────────────────────────────────────────────────
    # Scenario 3: Expanded but with reduced GEM (60% GEM + 20% turtle + 10% fx + 10% ic_eq)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("Scenario 3: REBALANCED (GEM 60% + turtle 20% + fx_carry 10% + ic_equity 10%)")
    print("─" * 100)
    p3 = build_portfolio(
        {"gem_j5": gem, "turtle_cat": turtle, "fx_carry": fx_carry, "ic_equity": ic_eq},
        {"gem_j5": 0.60, "turtle_cat": 0.20, "fx_carry": 0.10, "ic_equity": 0.10},
    )
    eval3 = evaluate_portfolio_vs_benchmark(
        p3,
        bench_6040,
        portfolio_label="GEM(60%)+turtle(20%)+fx(10%)+ic_eq(10%)",
        benchmark_label="60/40 SPY/IEF",
    )
    print(eval3.report())

    # ─────────────────────────────────────────────────────────────────
    # Joint L65 ruin assessment for each scenario
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("Joint L65 ruin (P(NAV DD > 15% in 1y), 2000 paths, gate < 1%)")
    print("─" * 100)
    scenarios = [
        (
            "S1 GEM 70% + turtle 20%",
            {"gem_j5": gem, "turtle_cat": turtle},
            {"gem_j5": 0.70, "turtle_cat": 0.20},
        ),
        (
            "S2 + fx 5% + ic_eq 5%",
            {"gem_j5": gem, "turtle_cat": turtle, "fx_carry": fx_carry, "ic_equity": ic_eq},
            {"gem_j5": 0.70, "turtle_cat": 0.20, "fx_carry": 0.05, "ic_equity": 0.05},
        ),
        (
            "S3 GEM 60% + 20+10+10",
            {"gem_j5": gem, "turtle_cat": turtle, "fx_carry": fx_carry, "ic_equity": ic_eq},
            {"gem_j5": 0.60, "turtle_cat": 0.20, "fx_carry": 0.10, "ic_equity": 0.10},
        ),
    ]
    for label, returns_dict, weights in scenarios:
        res = assess_joint_ruin(
            returns_dict,
            deployment_weights=weights,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        print(f"\n  {label}:")
        print(f"    P_kill_trip = {res.p_kill_trip:.3%}  (gate < 1%)")
        print(f"    95th-pct MaxDD = {res.p95_maxdd_at_size:.3%}  (gate > -25%)")
        print(f"    L65 passes: {res.passes()}")

    # ─────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("VERDICT — does expansion IMPROVE the portfolio?")
    print("=" * 100)
    print(f"S1 (current):    {eval1.n_passes}/10 → {eval1.verdict}")
    print(f"S2 (5% + 5%):    {eval2.n_passes}/10 → {eval2.verdict}")
    print(f"S3 (rebalanced): {eval3.n_passes}/10 → {eval3.verdict}")
    delta_s2 = eval2.n_passes - eval1.n_passes
    delta_s3 = eval3.n_passes - eval1.n_passes
    print(f"\nΔ_passes(S2-S1) = {delta_s2:+d}")
    print(f"Δ_passes(S3-S1) = {delta_s3:+d}")
    if delta_s2 > 0 or delta_s3 > 0:
        winner = "S2" if delta_s2 >= delta_s3 else "S3"
        print(f"\nEXPANSION VALIDATED: {winner} improves portfolio matrix.")
    elif delta_s2 == 0 and delta_s3 == 0:
        print("\nEXPANSION NEUTRAL: same verdict count. Default to S1 (less complexity).")
    else:
        print("\nEXPANSION DETERIORATES: hold S1; defer new candidates.")


if __name__ == "__main__":
    main()
