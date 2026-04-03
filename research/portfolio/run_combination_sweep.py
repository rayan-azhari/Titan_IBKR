"""run_combination_sweep.py — Portfolio Combination Sweeper.

Enumerates all valid strategy subsets (size 2–N), tests each with multiple
weighting schemes, applies hard portfolio gates, and outputs a ranked leaderboard.

This is computationally intensive. Runtime scales with O(C(n,k)) combinations.
With 10 strategies and max_size=5: ~250 unique subsets × 3 schemes ≈ 750 evaluations.

Usage
-----
    # Default sweep (Tier-1 strategies, sizes 2-5, equal + risk_parity):
    uv run python research/portfolio/run_combination_sweep.py

    # Custom sweep with tighter gates:
    uv run python research/portfolio/run_combination_sweep.py \\
        --min-strategies 3 --max-strategies 6 \\
        --max-dd 0.15 --min-sharpe 1.5 --max-ror 0.0005

    # Fast mode (Balsara RoR only, smaller MC):
    uv run python research/portfolio/run_combination_sweep.py --fast

Output
------
    .tmp/reports/portfolio_sweep_leaderboard.csv   (ranked by combined_score)
    .tmp/reports/portfolio_sweep_leaderboard.html  (formatted table)

Combined score
--------------
    combined_score = Sharpe × DR / (1 + |MaxDD|)
    Rewards high risk-adjusted return AND diversification.
    Penalises drawdown.

Gates (hard filters — combinations failing any gate are excluded)
-----------------------------------------------------------------
    Max portfolio drawdown   < max_dd     (default 0.20 = 20%)
    Min portfolio Sharpe     > min_sharpe (default 1.00)
    Max risk of ruin         < max_ror    (default 0.001 = 0.1%)
    No single strategy weight > 60%      (from Ensemble Strategy Framework.md)
    Minimum common OOS days  >= 252      (1 full year of shared data)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.portfolio.loaders.oos_returns import (  # noqa: E402
    ICIR_SCORES,
    align_to_common_window,
    load_all_strategies,
)
from research.portfolio.metrics import (  # noqa: E402
    ANNUAL_BARS,
    apply_correlation_constraint,
    compute_portfolio_stats,
)

# ── Default tier-1 strategy universe ──────────────────────────────────────────

TIER1_STRATEGIES: list[str] = [
    "ic_mtf_eur_usd",
    "ic_mtf_gbp_usd",
    "ic_mtf_usd_jpy",
    "ic_mtf_aud_jpy",
    "ic_equity_csco",
    "ic_equity_noc",
    "etf_trend_spy",
]

# ── Constants matching directives ─────────────────────────────────────────────

MAX_SINGLE_WEIGHT: float = 0.60
CORR_THRESHOLD: float = 0.70

# ── Weighting helpers (same as run_portfolio_research.py) ─────────────────────


def _equal(labels: list[str], _rets: dict) -> dict[str, float]:
    n = len(labels)
    return {k: 1.0 / n for k in labels}


def _risk_parity(labels: list[str], rets: dict[str, pd.Series]) -> dict[str, float]:
    vols = {k: float(rets[k].std() * np.sqrt(ANNUAL_BARS)) for k in labels}
    inv = {k: 1.0 / v if v > 1e-9 else 0.0 for k, v in vols.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total > 1e-9 else _equal(labels, rets)


def _icir(labels: list[str], _rets: dict) -> dict[str, float]:
    scores = {k: ICIR_SCORES.get(k, 0.3) for k in labels}
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()} if total > 1e-9 else _equal(labels, _rets)


SCHEMES: dict[str, callable] = {  # type: ignore[type-arg]
    "equal": _equal,
    "risk_parity": _risk_parity,
    "icir_weighted": _icir,
}


def _enforce_caps(w: dict[str, float], rets: dict[str, pd.Series]) -> dict[str, float]:
    w = apply_correlation_constraint(w, rets, CORR_THRESHOLD)
    w = {k: min(v, MAX_SINGLE_WEIGHT) for k, v in w.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 1e-9 else w


# ── Per-subset evaluator (called from thread pool) ────────────────────────────


def _eval_subset(
    labels: list[str],
    all_returns: dict[str, pd.Series],
    min_common_days: int,
    n_sim_ror: int,
    fast: bool,
    min_sharpe: float,
    max_dd: float,
    max_ror: float,
) -> list[dict]:
    """Evaluate all weighting schemes for one strategy subset. Thread-safe."""
    subset_raw = {k: all_returns[k] for k in labels}
    aligned = align_to_common_window(subset_raw)

    n_days = len(next(iter(aligned.values()))) if aligned else 0
    if n_days < min_common_days:
        return []

    rows: list[dict] = []
    for scheme_name, scheme_fn in SCHEMES.items():
        raw_w = scheme_fn(labels, aligned)
        w = _enforce_caps(raw_w, aligned)

        if not w or sum(w.values()) < 1e-9:
            continue
        if max(w.values()) > MAX_SINGLE_WEIGHT + 1e-6:
            continue

        ret_df = pd.DataFrame(aligned)[labels]
        w_arr = np.array([w[k] for k in labels])
        port_rets = pd.Series(ret_df.values @ w_arr, index=ret_df.index)

        _n_sim = 0 if fast else n_sim_ror
        stats = compute_portfolio_stats(port_rets, w, aligned, n_sim_ror=_n_sim)

        if stats["sharpe"] < min_sharpe:
            continue
        if abs(stats["max_dd"]) > max_dd:
            continue
        if stats["ror_composite"] >= max_ror:
            continue

        combined_score = (
            stats["sharpe"] * stats["diversification_ratio"] / (1.0 + abs(stats["max_dd"]))
        )
        row: dict = {
            "rank": 0,
            "combined_score": round(combined_score, 4),
            "scheme": scheme_name,
            "n_strategies": len(labels),
            "strategies": "|".join(labels),
            "sharpe": stats["sharpe"],
            "max_dd": stats["max_dd"],
            "cagr": stats["cagr"],
            "calmar": stats["calmar"],
            "diversification_ratio": stats["diversification_ratio"],
            "avg_pairwise_corr": stats["avg_pairwise_corr"],
            "ror_montecarlo": stats["ror_montecarlo"],
            "ror_balsara": stats["ror_balsara"],
            "ror_composite": stats["ror_composite"],
            "n_days": stats["n_days"],
            "n_years": stats["n_years"],
        }
        for k in labels:
            row[f"w_{k}"] = round(w.get(k, 0.0), 4)
        rows.append(row)

    return rows


# ── Core sweep function ────────────────────────────────────────────────────────


def sweep(
    all_returns: dict[str, pd.Series],
    min_size: int = 2,
    max_size: int = 5,
    max_dd: float = 0.20,
    min_sharpe: float = 1.0,
    max_ror: float = 0.001,
    min_common_days: int = 252,
    n_sim_ror: int = 2_000,
    fast: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Enumerate all strategy subsets and return a ranked leaderboard DataFrame.

    Combinations are evaluated in parallel using a thread pool (NumPy releases
    the GIL, so threads provide genuine concurrency for the vectorised MC).

    Args:
        all_returns:      dict of OOS daily return series per strategy.
        min_size / max_size: subset size range.
        max_dd / min_sharpe / max_ror: hard gates.
        min_common_days:  minimum common OOS days.
        n_sim_ror:        Monte Carlo simulations per combination.
        fast:             skip Monte Carlo (Balsara only).
        max_workers:      thread pool size (default 4).
    """
    strategy_names = list(all_returns.keys())
    all_subsets = [
        list(subset)
        for size in range(min_size, max_size + 1)
        for subset in combinations(strategy_names, size)
    ]
    total_combos = len(all_subsets)

    print(
        f"\n  Sweeping {total_combos} combinations "
        f"(size {min_size}-{max_size}, {len(SCHEMES)} schemes, {max_workers} threads)..."
    )

    results: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _eval_subset,
                labels,
                all_returns,
                min_common_days,
                n_sim_ror,
                fast,
                min_sharpe,
                max_dd,
                max_ror,
            ): labels
            for labels in all_subsets
        }
        for fut in as_completed(futures):
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{total_combos} ...", end="\r")
            rows = fut.result()
            results.extend(rows)

    print(f"    {total_combos}/{total_combos} combinations evaluated.      ")

    if not results:
        print("  WARNING: No combinations passed all gates.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("combined_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio combination sweeper.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategy labels to sweep (default: Tier-1 universe).",
    )
    parser.add_argument(
        "--min-strategies",
        type=int,
        default=2,
        help="Minimum combination size (default 2).",
    )
    parser.add_argument(
        "--max-strategies",
        type=int,
        default=5,
        help="Maximum combination size (default 5).",
    )
    parser.add_argument(
        "--max-dd",
        type=float,
        default=0.20,
        help="Max portfolio drawdown gate (default 0.20).",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=1.0,
        help="Min portfolio Sharpe gate (default 1.0).",
    )
    parser.add_argument(
        "--max-ror",
        type=float,
        default=0.001,
        help="Max portfolio RoR gate (default 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=2_000,
        help="Monte Carlo simulations per combination (default 2000).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip Monte Carlo (Balsara only). ~10x faster, less accurate.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Print top N results to console (default 20).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for parallel combination evaluation (default 4).",
    )
    args = parser.parse_args()

    strategies = args.strategies if args.strategies is not None else TIER1_STRATEGIES

    W = 72
    print()
    print("=" * W)
    print("  PORTFOLIO COMBINATION SWEEP")
    print(f"  Universe   : {strategies}")
    print(f"  Size range : {args.min_strategies}–{args.max_strategies}")
    print(
        f"  Gates      : MaxDD<{args.max_dd:.0%}  "
        f"Sharpe>{args.min_sharpe:.1f}  "
        f"RoR<{args.max_ror:.2%}"
    )
    print("=" * W)

    # ── Load OOS returns ───────────────────────────────────────────────────
    print("\n  Loading OOS return series...")
    raw_returns = load_all_strategies(strategies, skip_errors=True)

    if len(raw_returns) < 2:
        print("  ERROR: Need at least 2 strategies. Exiting.")
        sys.exit(1)

    # Note: do NOT align here — alignment is done per-combination inside sweep()
    # so that each subset uses its own maximum-length common window.

    # ── Run sweep ──────────────────────────────────────────────────────────
    leaderboard = sweep(
        all_returns=raw_returns,
        min_size=args.min_strategies,
        max_size=min(args.max_strategies, len(raw_returns)),
        max_dd=args.max_dd,
        min_sharpe=args.min_sharpe,
        max_ror=args.max_ror,
        n_sim_ror=args.n_sim,
        fast=args.fast,
        max_workers=args.workers,
    )

    if leaderboard.empty:
        print("\n  No passing combinations found. Loosen gate parameters.")
        return

    # ── Print top-N ────────────────────────────────────────────────────────
    top = leaderboard.head(args.top_n)
    print(f"\n  TOP {min(args.top_n, len(leaderboard))} COMBINATIONS:")
    print(
        f"  {'#':>3}  {'Score':>7}  {'Sharpe':>7}  {'MaxDD':>7}  "
        f"{'CAGR':>7}  {'DR':>5}  {'RoR':>8}  "
        f"{'Scheme':<14}  Strategies"
    )
    print("  " + "-" * 90)
    for _, r in top.iterrows():
        print(
            f"  {int(r['rank']):>3}  "
            f"{r['combined_score']:>7.3f}  "
            f"{r['sharpe']:>+7.3f}  "
            f"{r['max_dd']:>+7.2%}  "
            f"{r['cagr']:>+7.2%}  "
            f"{r['diversification_ratio']:>5.3f}  "
            f"{r['ror_composite']:>8.4%}  "
            f"{r['scheme']:<14}  "
            f"{r['strategies']}"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = len(leaderboard)
    print(f"\n  {n_pass} combinations passed all gates.")
    if n_pass > 0:
        best = leaderboard.iloc[0]
        print(f"  Best combination: {best['strategies']}")
        print(f"    Scheme: {best['scheme']}")
        print(
            f"    Sharpe: {best['sharpe']:+.3f}  MaxDD: {best['max_dd']:+.2%}  "
            f"CAGR: {best['cagr']:+.2%}  RoR: {best['ror_composite']:.4%}"
        )
        print(f"    Diversification ratio: {best['diversification_ratio']:.3f}")

    # ── Save outputs ───────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"portfolio_sweep_leaderboard_{ts}.csv"
    leaderboard.to_csv(csv_path, index=False)

    html_path = REPORTS_DIR / f"portfolio_sweep_leaderboard_{ts}.html"
    leaderboard.to_html(html_path, index=False, float_format="{:.4f}".format)

    print("\n  Leaderboard saved:")
    print(f"    CSV  -> {csv_path}")
    print(f"    HTML -> {html_path}")


if __name__ == "__main__":
    main()
