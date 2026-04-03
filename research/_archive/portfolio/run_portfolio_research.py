"""run_portfolio_research.py — Multi-Strategy Portfolio Combination Explorer.

Interactive research tool. Loads OOS returns for a specified set of strategies,
tests multiple weighting schemes, and reports portfolio-level statistics including
zero-ruin verification metrics.

Usage
-----
    # Explore a specific combination with all weighting schemes:
    uv run python research/portfolio/run_portfolio_research.py \\
        --strategies ic_mtf_eur_usd ic_mtf_gbp_usd ic_equity_csco etf_trend_spy \\
        --weights equal risk_parity icir_weighted

    # Full Tier-1 recommended portfolio (equal + risk-parity):
    uv run python research/portfolio/run_portfolio_research.py \\
        --strategies ic_mtf_eur_usd ic_mtf_gbp_usd ic_mtf_usd_jpy ic_mtf_aud_jpy \\
                     ic_equity_csco ic_equity_noc etf_trend_spy \\
        --weights equal risk_parity

    # List available strategy labels:
    uv run python research/portfolio/run_portfolio_research.py --list

Output
------
    Console: per-scheme stats table + correlation heatmap
    CSV:     .tmp/reports/portfolio_research_{timestamp}.csv

Zero-ruin gate
--------------
    Flags combinations where ror_composite >= 0.001 (0.1%).
    Hard constraint: RoR < 0.001 required for live deployment.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.portfolio.loaders.oos_returns import (  # noqa: E402
    _REGISTRY,
    ICIR_SCORES,
    align_to_common_window,
    load_all_strategies,
)
from research.portfolio.metrics import (  # noqa: E402
    ANNUAL_BARS,
    apply_correlation_constraint,
    compute_portfolio_stats,
)

# ── Portfolio-level hard gates (from directives) ───────────────────────────────

MAX_SINGLE_WEIGHT: float = 0.60  # Ensemble Strategy Framework.md
CORR_THRESHOLD: float = 0.70  # Ensemble Strategy Framework.md
ROR_GATE: float = 0.001  # institutional threshold (0.1%)
MAX_DD_GATE: float = 0.20  # Phase 3 gate (from Backtesting & Validation.md)

# ── Weighting schemes ──────────────────────────────────────────────────────────


def _equal_weights(labels: list[str]) -> dict[str, float]:
    n = len(labels)
    return {k: 1.0 / n for k in labels}


def _risk_parity_weights(
    labels: list[str], component_returns: dict[str, pd.Series]
) -> dict[str, float]:
    """Inverse-volatility weighting (robust OOS, preferred over full risk parity)."""
    vols = {k: float(component_returns[k].std() * np.sqrt(ANNUAL_BARS)) for k in labels}
    inv_vols = {k: 1.0 / v if v > 1e-9 else 0.0 for k, v in vols.items()}
    total = sum(inv_vols.values())
    if total < 1e-9:
        return _equal_weights(labels)
    return {k: v / total for k, v in inv_vols.items()}


def _icir_weights(labels: list[str]) -> dict[str, float]:
    """Weights proportional to ICIR scores from FINDINGS.md."""
    scores = {k: ICIR_SCORES.get(k, 0.3) for k in labels}
    total = sum(scores.values())
    if total < 1e-9:
        return _equal_weights(labels)
    return {k: v / total for k, v in scores.items()}


def _kelly_weights(
    labels: list[str],
    component_returns: dict[str, pd.Series],
    kelly_fraction: float = 0.25,
) -> dict[str, float]:
    """Fractional Kelly (25% of full Kelly), capped at MAX_SINGLE_WEIGHT."""
    ret_df = pd.DataFrame({k: component_returns[k] for k in labels}).dropna()
    if len(ret_df) < 30:
        return _equal_weights(labels)
    mu = ret_df.mean().values
    cov = ret_df.cov().values
    try:
        cov_inv = np.linalg.inv(cov + np.eye(len(labels)) * 1e-8)
        full_kelly = cov_inv @ mu
        w = kelly_fraction * full_kelly
        # Clip negatives to 0 (no shorting strategies)
        w = np.clip(w, 0, MAX_SINGLE_WEIGHT)
        total = w.sum()
        if total < 1e-9:
            return _equal_weights(labels)
        w = w / total
    except np.linalg.LinAlgError:
        return _equal_weights(labels)
    return dict(zip(labels, w.tolist()))


SCHEME_BUILDERS = {
    "equal": lambda labels, rets: _equal_weights(labels),
    "risk_parity": lambda labels, rets: _risk_parity_weights(labels, rets),
    "icir_weighted": lambda labels, rets: _icir_weights(labels),
    "kelly": lambda labels, rets: _kelly_weights(labels, rets),
}


# ── Weight enforcement ─────────────────────────────────────────────────────────


def _enforce_weight_caps(
    weights: dict[str, float],
    component_returns: dict[str, pd.Series],
) -> dict[str, float]:
    """Apply correlation constraint + single-strategy cap, then re-normalise."""
    w = apply_correlation_constraint(weights, component_returns, CORR_THRESHOLD)
    # Hard cap
    w = {k: min(v, MAX_SINGLE_WEIGHT) for k, v in w.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 1e-9 else w


# ── Display helpers ────────────────────────────────────────────────────────────


def _print_separator(width: int = 72) -> None:
    print("-" * width)


def _print_stats_row(scheme: str, stats: dict, flags: list[str]) -> None:
    flag_str = "  " + " ".join(flags) if flags else ""
    print(
        f"  {scheme:<18s}  "
        f"Sharpe={stats['sharpe']:>+6.3f}  "
        f"MaxDD={stats['max_dd']:>+7.2%}  "
        f"CAGR={stats['cagr']:>+6.2%}  "
        f"DR={stats['diversification_ratio']:>5.3f}  "
        f"RoR={stats['ror_composite']:.4%}"
        f"{flag_str}"
    )


def _print_correlation_table(corr: pd.DataFrame) -> None:
    print("\n  Pairwise Correlation Matrix (OOS period):")
    labels = corr.columns.tolist()
    max_lbl = max(len(lbl) for lbl in labels)
    header = " " * (max_lbl + 2) + "  ".join(f"{lbl[:8]:>8}" for lbl in labels)
    print("  " + header)
    for row_label in labels:
        row_vals = "  ".join(f"{corr.loc[row_label, col]:>+8.3f}" for col in labels)
        print(f"  {row_label:{max_lbl}s}  {row_vals}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio combination explorer.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategy labels to combine (see --list for options).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["equal", "risk_parity"],
        choices=list(SCHEME_BUILDERS.keys()),
        help="Weighting schemes to evaluate.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available strategy labels and exit.",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=5_000,
        help="Monte Carlo simulations for RoR (default 5000; use 10000 for final validation).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip Monte Carlo (use Balsara only). Faster but less accurate.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable strategy labels:")
        for label in sorted(_REGISTRY.keys()):
            print(f"  {label}")
        return

    # ── Default to Tier-1 recommended portfolio ────────────────────────────
    if args.strategies is None:
        args.strategies = [
            "ic_mtf_eur_usd",
            "ic_mtf_gbp_usd",
            "ic_mtf_usd_jpy",
            "ic_mtf_aud_jpy",
            "ic_equity_csco",
            "ic_equity_noc",
            "etf_trend_spy",
        ]
        print("  [INFO] No --strategies specified. Using Tier-1 recommended portfolio.")

    W = 72
    print()
    print("=" * W)
    print("  PORTFOLIO COMBINATION EXPLORER")
    print(f"  Strategies : {args.strategies}")
    print(f"  Schemes    : {args.weights}")
    print("=" * W)

    # ── Load OOS returns ───────────────────────────────────────────────────
    print("\n  Loading OOS return series...")
    raw_returns = load_all_strategies(args.strategies, skip_errors=True)

    if len(raw_returns) < 2:
        print("  ERROR: Need at least 2 strategies. Exiting.")
        sys.exit(1)

    # ── Align to common window ─────────────────────────────────────────────
    aligned = align_to_common_window(raw_returns)
    labels = list(aligned.keys())
    n_days = len(next(iter(aligned.values())))
    if n_days < 50:
        print(f"  ERROR: Common OOS window too short ({n_days} days). Exiting.")
        sys.exit(1)

    start_dt = next(iter(aligned.values())).index[0].date()
    end_dt = next(iter(aligned.values())).index[-1].date()
    print(f"\n  Common OOS window: {start_dt} to {end_dt} ({n_days} days)")

    # ── Run each weighting scheme ──────────────────────────────────────────
    print(
        f"\n  {'Scheme':<18s}  {'Sharpe':>8}  {'MaxDD':>8}  "
        f"{'CAGR':>7}  {'DR':>6}  {'RoR':>8}  Flags"
    )
    _print_separator()

    all_results: list[dict] = []

    for scheme in args.weights:
        build_fn = SCHEME_BUILDERS[scheme]
        raw_w = build_fn(labels, aligned)
        w = _enforce_weight_caps(raw_w, aligned)

        # Portfolio returns
        ret_df = pd.DataFrame(aligned)
        w_arr = np.array([w.get(k, 0.0) for k in labels])
        port_rets = ret_df[labels].values @ w_arr
        port_series = pd.Series(port_rets, index=ret_df.index, name=f"portfolio_{scheme}")

        n_sim = 0 if args.fast else args.n_sim
        stats = compute_portfolio_stats(port_series, w, aligned, n_sim_ror=n_sim)

        flags: list[str] = []
        if abs(stats["max_dd"]) > MAX_DD_GATE:
            flags.append("[!MaxDD]")
        if stats["ror_composite"] >= ROR_GATE:
            flags.append("[!RoR]")
        if max(w.values()) > MAX_SINGLE_WEIGHT:
            flags.append("[!Conc]")

        _print_stats_row(scheme, stats, flags)

        row = {
            "scheme": scheme,
            "n_strategies": len(labels),
            "strategies": "|".join(labels),
            **{f"w_{k}": round(w.get(k, 0.0), 4) for k in labels},
            **{
                k: v
                for k, v in stats.items()
                if k not in ("correlation_matrix", "individual_sharpes")
            },
            "pass_ror": stats["ror_composite"] < ROR_GATE,
            "pass_maxdd": abs(stats["max_dd"]) <= MAX_DD_GATE,
        }
        all_results.append(row)

    _print_separator()

    # ── Correlation heatmap ────────────────────────────────────────────────
    # Use the last computed stats (correlation matrix is scheme-independent)
    if "correlation_matrix" in stats:
        _print_correlation_table(stats["correlation_matrix"])

    # ── Per-strategy Sharpe in common window ──────────────────────────────
    print("\n  Per-strategy Sharpe (common OOS window):")
    for lbl, s in aligned.items():
        sd = float(s.std())
        sh = float(s.mean() / sd * np.sqrt(252)) if sd > 1e-9 else 0.0
        print(f"    {lbl:<28s}  Sharpe={sh:>+6.3f}")

    # ── Zero-ruin summary ──────────────────────────────────────────────────
    print("\n  ZERO-RUIN GATE (RoR < 0.1% required for live deployment):")
    for r in all_results:
        status = "PASS" if r["pass_ror"] else "FAIL"
        ror = r.get("ror_composite", float("nan"))
        print(f"    [{status}] {r['scheme']:<18s}  RoR={ror:.4%}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(all_results)
    out_path = REPORTS_DIR / f"portfolio_research_{ts}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
