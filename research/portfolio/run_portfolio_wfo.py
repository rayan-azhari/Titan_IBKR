"""run_portfolio_wfo.py -- Portfolio-Level Walk-Forward Optimization.

Re-estimates allocation weights on rolling IS windows, evaluates on OOS folds,
and stitches the OOS folds into a combined portfolio equity curve.

This is NOT per-strategy WFO (which Phase 4 handles). This is portfolio-level:
the individual strategy return series are fixed inputs; only the allocation
weights are re-optimized per fold.

Usage:
    uv run python research/portfolio/run_portfolio_wfo.py
    uv run python research/portfolio/run_portfolio_wfo.py \\
        --strategies ic_equity_hwm etf_trend_spy gold_macro pairs_gld_efa fx_carry_aud_jpy \\
        --weighting risk_parity --is-days 504 --oos-days 126 --wfo-type rolling

Directive: Backtesting & Validation.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from math import sqrt
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

# ── Default strategy set ─────────────────────────────────────────────────────

DEFAULT_STRATEGIES = [
    "ic_equity_csco",
    "ic_equity_noc",
    "etf_trend_spy",
    "gold_macro",
    "pairs_gld_efa",
    "fx_carry_aud_jpy",
    "mtf_eur_usd",
]

# ── Quality gates (analogous to Phase 4 WFO) ─────────────────────────────────

MIN_PCT_FOLDS_POSITIVE = 0.70  # >= 70% of OOS folds must have Sharpe > 0
MIN_STITCHED_SHARPE = 1.0
MAX_STITCHED_DD = 0.20  # 20%
MAX_ROR = 0.001  # 0.1% risk of ruin


# ── Weighting schemes ────────────────────────────────────────────────────────


def _equal_weights(labels: list[str], _is_returns: dict[str, pd.Series]) -> dict[str, float]:
    """Equal 1/N allocation."""
    n = len(labels)
    return {k: 1.0 / n for k in labels}


def _risk_parity_weights(labels: list[str], is_returns: dict[str, pd.Series]) -> dict[str, float]:
    """Inverse-volatility weighting from IS-window returns."""
    vols = {}
    for k in labels:
        s = is_returns[k].dropna()
        v = float(s.std() * sqrt(ANNUAL_BARS)) if len(s) > 10 else 1.0
        vols[k] = max(v, 1e-6)

    inv_vols = {k: 1.0 / v for k, v in vols.items()}
    total = sum(inv_vols.values())
    return {k: v / total for k, v in inv_vols.items()}


def _icir_weights(labels: list[str], _is_returns: dict[str, pd.Series]) -> dict[str, float]:
    """ICIR-score proportional weighting (static, from FINDINGS.md)."""
    scores = {k: ICIR_SCORES.get(k, 0.30) for k in labels}
    total = sum(scores.values())
    if total < 1e-9:
        return _equal_weights(labels, _is_returns)
    return {k: v / total for k, v in scores.items()}


def _kelly_weights(labels: list[str], is_returns: dict[str, pd.Series]) -> dict[str, float]:
    """Fractional Kelly (25%) using IS-window mean/covariance."""
    ret_df = pd.DataFrame({k: is_returns[k] for k in labels}).dropna()
    if len(ret_df) < 30:
        return _equal_weights(labels, is_returns)

    mu = ret_df.mean().values * ANNUAL_BARS
    cov = ret_df.cov().values * ANNUAL_BARS

    # Ridge regularization for stability
    cov += np.eye(len(labels)) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return _equal_weights(labels, is_returns)

    raw = cov_inv @ mu
    raw = np.clip(raw, 0.0, None)  # long-only
    total = raw.sum()
    if total < 1e-9:
        return _equal_weights(labels, is_returns)

    # Quarter-Kelly for safety
    weights = raw / total * 0.25
    # Re-normalise to sum to 1
    weights = weights / weights.sum()
    return dict(zip(labels, weights))


WEIGHTING_SCHEMES = {
    "equal": _equal_weights,
    "risk_parity": _risk_parity_weights,
    "icir_weighted": _icir_weights,
    "kelly": _kelly_weights,
}


# ── Weight constraints ───────────────────────────────────────────────────────


def _apply_constraints(
    weights: dict[str, float],
    is_returns: dict[str, pd.Series],
    max_single_weight: float = 0.60,
    min_weight: float = 0.05,
    corr_threshold: float = 0.70,
) -> dict[str, float]:
    """Apply weight caps, floors, correlation penalty, and re-normalise."""
    w = dict(weights)

    # Floor
    for k in w:
        w[k] = max(w[k], min_weight)

    # Cap
    for k in w:
        w[k] = min(w[k], max_single_weight)

    # Correlation penalty (reuse metrics helper)
    w = apply_correlation_constraint(w, is_returns, corr_threshold)

    # Re-normalise
    total = sum(w.values())
    if total > 1e-9:
        w = {k: v / total for k, v in w.items()}

    return w


# ── Fold helpers ─────────────────────────────────────────────────────────────


def _quick_sharpe(rets: pd.Series) -> float:
    """Annualised Sharpe from daily returns."""
    s = rets.dropna()
    if len(s) < 5:
        return 0.0
    std = float(s.std())
    return float(s.mean() / std * sqrt(ANNUAL_BARS)) if std > 1e-9 else 0.0


def _max_dd(rets: pd.Series) -> float:
    """Maximum drawdown from daily returns."""
    equity = (1 + rets.fillna(0)).cumprod()
    hwm = equity.cummax()
    dd = (equity - hwm) / hwm
    return float(dd.min())


# ── Core WFO engine ──────────────────────────────────────────────────────────


def run_portfolio_wfo(
    strategy_returns: dict[str, pd.Series],
    weighting_scheme: str = "risk_parity",
    is_days: int = 504,
    oos_days: int = 126,
    wfo_type: str = "rolling",
    corr_threshold: float = 0.70,
    max_single_weight: float = 0.60,
) -> dict:
    """Run portfolio-level walk-forward optimization.

    Parameters
    ----------
    strategy_returns : aligned daily returns per strategy (common date range)
    weighting_scheme : one of "equal", "risk_parity", "icir_weighted", "kelly"
    is_days : trading days in the IS window
    oos_days : trading days per OOS fold
    wfo_type : "rolling" (fixed IS slides) or "anchored" (IS grows from day 0)
    corr_threshold : correlation penalty threshold
    max_single_weight : max allocation to any single strategy

    Returns
    -------
    dict with fold_results, stitched_returns, stitched_stats, weight_history, fold_df
    """
    labels = list(strategy_returns.keys())
    weight_fn = WEIGHTING_SCHEMES.get(weighting_scheme)
    if weight_fn is None:
        raise ValueError(
            f"Unknown weighting scheme: {weighting_scheme}. "
            f"Choose from: {list(WEIGHTING_SCHEMES.keys())}"
        )

    # Stack all returns into a DataFrame for easy slicing
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    n_bars = len(ret_df)

    if n_bars < is_days + oos_days:
        raise ValueError(
            f"Insufficient data: {n_bars} bars available, need at least "
            f"{is_days + oos_days} (IS={is_days} + OOS={oos_days}). "
            "Try reducing --is-days or --oos-days, or adding more strategies."
        )

    # Build fold boundaries
    fold_results = []
    weight_history = []
    stitched_oos = []

    fold_idx = 0
    oos_start = is_days

    while oos_start + oos_days <= n_bars:
        # IS window
        if wfo_type == "anchored":
            is_start = 0
        else:
            is_start = oos_start - is_days
        is_end = oos_start

        # OOS window
        oos_end = oos_start + oos_days

        # Slice returns
        is_slice = ret_df.iloc[is_start:is_end]
        oos_slice = ret_df.iloc[oos_start:oos_end]

        is_returns = {k: is_slice[k] for k in labels}

        # Compute weights on IS data
        weights = weight_fn(labels, is_returns)
        weights = _apply_constraints(
            weights,
            is_returns,
            max_single_weight=max_single_weight,
            corr_threshold=corr_threshold,
        )

        # Combine OOS returns: port = sum(w_i * strat_i)
        port_oos = pd.Series(0.0, index=oos_slice.index)
        for k in labels:
            port_oos += weights[k] * oos_slice[k]

        # IS portfolio returns (for parity comparison)
        port_is = pd.Series(0.0, index=is_slice.index)
        for k in labels:
            port_is += weights[k] * is_slice[k]

        is_sharpe = _quick_sharpe(port_is)
        oos_sharpe = _quick_sharpe(port_oos)
        parity = oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-6 else 0.0

        fold_results.append(
            {
                "fold": fold_idx,
                "is_start": is_slice.index[0].strftime("%Y-%m-%d"),
                "is_end": is_slice.index[-1].strftime("%Y-%m-%d"),
                "oos_start": oos_slice.index[0].strftime("%Y-%m-%d"),
                "oos_end": oos_slice.index[-1].strftime("%Y-%m-%d"),
                "is_bars": len(is_slice),
                "oos_bars": len(oos_slice),
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "parity": round(parity, 3),
                "oos_max_dd": round(_max_dd(port_oos) * 100, 2),
            }
        )
        weight_history.append({"fold": fold_idx, **{k: round(v, 4) for k, v in weights.items()}})
        stitched_oos.append(port_oos)

        fold_idx += 1
        oos_start += oos_days

    # Stitch all OOS folds
    stitched_returns = pd.concat(stitched_oos)

    # Compute aggregate stats on stitched series
    component_rets_oos = {}
    for k in labels:
        # Collect OOS slices for each component
        parts = [
            ret_df[k].iloc[f["fold"] * oos_days + is_days : (f["fold"] + 1) * oos_days + is_days]
            for f in fold_results
        ]
        component_rets_oos[k] = pd.concat(parts)

    # Use the last fold's weights for the stats computation
    final_weights = {k: weight_history[-1].get(k, 0.0) for k in labels} if weight_history else {}

    stitched_stats = compute_portfolio_stats(
        stitched_returns,
        final_weights,
        component_rets_oos,
        n_sim_ror=10_000,
    )

    fold_df = pd.DataFrame(fold_results)
    weight_df = pd.DataFrame(weight_history)

    return {
        "fold_results": fold_results,
        "stitched_returns": stitched_returns,
        "stitched_stats": stitched_stats,
        "weight_history": weight_history,
        "fold_df": fold_df,
        "weight_df": weight_df,
    }


# ── Display ──────────────────────────────────────────────────────────────────


def print_results(result: dict, weighting_scheme: str) -> None:
    """Print WFO results to console."""
    fold_df = result["fold_df"]
    stats = result["stitched_stats"]
    weight_df = result["weight_df"]

    print("\n" + "=" * 90)
    print(f"  PORTFOLIO WALK-FORWARD OPTIMIZATION | Weighting: {weighting_scheme.upper()}")
    print("=" * 90)

    # Fold table
    print(
        f"\n{'Fold':>4} | {'IS Period':>23} | {'OOS Period':>23} | "
        f"{'IS Sh':>6} {'OOS Sh':>6} {'Parity':>7} | {'OOS DD':>7}"
    )
    print("-" * 90)
    for _, row in fold_df.iterrows():
        print(
            f"{int(row['fold']):4d} | {row['is_start']} - {row['is_end']} | "
            f"{row['oos_start']} - {row['oos_end']} | "
            f"{row['is_sharpe']:+6.3f} {row['oos_sharpe']:+6.3f} {row['parity']:7.3f} | "
            f"{row['oos_max_dd']:+6.1f}%"
        )

    # Weight evolution
    labels = [c for c in weight_df.columns if c != "fold"]
    print("\n  Weight Evolution:")
    header = f"  {'Fold':>4}"
    for lbl in labels:
        print_lbl = lbl[:14]
        header += f" | {print_lbl:>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in weight_df.iterrows():
        line = f"  {int(row['fold']):4d}"
        for lbl in labels:
            line += f" | {row[lbl]:14.1%}"
        print(line)

    # Aggregate stats
    print(f"\n{'=' * 60}")
    print("  STITCHED OOS PORTFOLIO STATS")
    print(f"{'=' * 60}")
    print(f"  Sharpe:               {stats.get('sharpe', 0):.3f}")
    print(f"  CAGR:                 {stats.get('cagr', 0) * 100:.2f}%")
    print(f"  Calmar:               {stats.get('calmar', 0):.3f}")
    print(f"  Max Drawdown:         {stats.get('max_dd', 0) * 100:.2f}%")
    print(f"  Win Rate (daily):     {stats.get('win_rate_daily', 0) * 100:.1f}%")
    print(f"  Diversification Ratio:{stats.get('diversification_ratio', 0):.3f}")
    print(f"  Avg Pairwise Corr:    {stats.get('avg_pairwise_corr', 0):.3f}")
    print(f"  RoR (Monte Carlo):    {stats.get('ror_montecarlo', 0):.5f}")
    print(f"  RoR (Balsara):        {stats.get('ror_balsara', 0):.5f}")
    print(f"  RoR (Composite):      {stats.get('ror_composite', 0):.5f}")
    print(f"  OOS Days:             {stats.get('n_days', 0)}")
    print(f"  OOS Years:            {stats.get('n_years', 0):.2f}")

    # Per-strategy Sharpe
    ind = stats.get("individual_sharpes", {})
    if ind:
        print("\n  Per-Strategy OOS Sharpe:")
        for k, v in sorted(ind.items(), key=lambda x: -x[1]):
            print(f"    {k:<28s} {v:+.3f}")

    # Correlation matrix
    corr = stats.get("correlation_matrix")
    if corr is not None and len(corr) > 1:
        print("\n  Pairwise Correlation Matrix:")
        # Truncate labels for display
        short = {c: c[:12] for c in corr.columns}
        corr_display = corr.rename(columns=short, index=short)
        print(corr_display.round(2).to_string(float_format=lambda x: f"{x:+.2f}"))

    # Quality gates
    n_folds = len(fold_df)
    positive_folds = (fold_df["oos_sharpe"] > 0).sum()
    pct_positive = positive_folds / n_folds if n_folds > 0 else 0
    stitched_sharpe = stats.get("sharpe", 0)
    stitched_dd = abs(stats.get("max_dd", 0))
    ror = stats.get("ror_composite", 1.0)

    print(f"\n{'=' * 60}")
    print("  QUALITY GATES")
    print(f"{'=' * 60}")

    def _gate(name: str, value: float, threshold: float, op: str = ">=") -> str:
        if op == ">=":
            ok = value >= threshold
        elif op == "<=":
            ok = value <= threshold
        else:
            ok = value < threshold
        status = "PASS" if ok else "FAIL"
        return f"  [{status}] {name}: {value:.4f} (threshold {op} {threshold})"

    print(_gate("Positive OOS folds", pct_positive, MIN_PCT_FOLDS_POSITIVE))
    print(_gate("Stitched Sharpe", stitched_sharpe, MIN_STITCHED_SHARPE))
    print(_gate("Stitched MaxDD", stitched_dd, MAX_STITCHED_DD, "<="))
    print(_gate("Risk of Ruin", ror, MAX_ROR, "<"))


# ── Save outputs ─────────────────────────────────────────────────────────────


def save_outputs(result: dict, suffix: str = "") -> None:
    """Save WFO outputs to .tmp/reports/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{suffix}" if suffix else ""

    # Fold summary
    fold_path = REPORTS_DIR / f"portfolio_wfo_folds{tag}_{ts}.csv"
    result["fold_df"].to_csv(fold_path, index=False)
    print(f"\n  Folds saved to:   {fold_path}")

    # Weight history
    weight_path = REPORTS_DIR / f"portfolio_wfo_weights{tag}_{ts}.csv"
    result["weight_df"].to_csv(weight_path, index=False)
    print(f"  Weights saved to: {weight_path}")

    # Stitched equity
    equity_path = REPORTS_DIR / f"portfolio_wfo_equity{tag}_{ts}.csv"
    result["stitched_returns"].to_csv(equity_path, header=True)
    print(f"  Equity saved to:  {equity_path}")

    # Correlation matrix
    stats = result["stitched_stats"]
    corr = stats.get("correlation_matrix")
    if corr is not None:
        corr_path = REPORTS_DIR / f"portfolio_wfo_correlation{tag}_{ts}.csv"
        corr.to_csv(corr_path)
        print(f"  Corr matrix:      {corr_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio-Level Walk-Forward Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="Strategy labels to include (from oos_returns registry)",
    )
    parser.add_argument(
        "--weighting",
        default="risk_parity",
        choices=list(WEIGHTING_SCHEMES.keys()),
        help="Allocation weighting scheme",
    )
    parser.add_argument("--is-days", type=int, default=504, help="IS window (trading days)")
    parser.add_argument("--oos-days", type=int, default=126, help="OOS window (trading days)")
    parser.add_argument(
        "--wfo-type",
        default="rolling",
        choices=["rolling", "anchored"],
        help="WFO type: rolling (fixed IS window) or anchored (IS grows)",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.70)
    parser.add_argument("--max-weight", type=float, default=0.60)
    parser.add_argument("--no-save", action="store_true", help="Skip saving output files")
    args = parser.parse_args()

    print("=" * 60)
    print("  PORTFOLIO WALK-FORWARD OPTIMIZATION")
    print(f"  Strategies: {len(args.strategies)}")
    print(f"  Weighting:  {args.weighting}")
    print(f"  WFO Type:   {args.wfo_type}")
    print(f"  IS Window:  {args.is_days} days | OOS Window: {args.oos_days} days")
    print("=" * 60)

    # Load all strategy OOS returns
    print("\nLoading strategy OOS returns...")
    raw_returns = load_all_strategies(args.strategies, skip_errors=True)
    if len(raw_returns) < 2:
        print("\nERROR: Need at least 2 strategies to build a portfolio.")
        print(f"  Loaded: {list(raw_returns.keys())}")
        sys.exit(1)

    # Align to common date range
    aligned = align_to_common_window(raw_returns)
    if not aligned or all(len(s) == 0 for s in aligned.values()):
        print("\nERROR: No common date window found across loaded strategies.")
        print("  This usually means the OOS periods don't overlap.")
        print("  Try loading fewer strategies or downloading more data.")
        sys.exit(1)
    labels = list(aligned.keys())
    n_days = len(next(iter(aligned.values())))
    if n_days == 0:
        print("\nERROR: Common window is empty (0 overlapping days).")
        sys.exit(1)
    start = next(iter(aligned.values())).index[0].strftime("%Y-%m-%d")
    end = next(iter(aligned.values())).index[-1].strftime("%Y-%m-%d")

    print(f"\nAligned {len(labels)} strategies to common window:")
    print(f"  {start} -> {end} ({n_days} trading days)")
    print(f"  Strategies: {', '.join(labels)}")

    if n_days < args.is_days + args.oos_days:
        print(
            f"\nERROR: Insufficient common window ({n_days} days) for "
            f"IS={args.is_days} + OOS={args.oos_days}."
        )
        print("  Try: fewer strategies, smaller windows, or download more data.")
        sys.exit(1)

    # Run WFO
    print("\nRunning walk-forward optimization...")
    result = run_portfolio_wfo(
        strategy_returns=aligned,
        weighting_scheme=args.weighting,
        is_days=args.is_days,
        oos_days=args.oos_days,
        wfo_type=args.wfo_type,
        corr_threshold=args.corr_threshold,
        max_single_weight=args.max_weight,
    )

    # Display
    print_results(result, args.weighting)

    # Save
    if not args.no_save:
        save_outputs(result, args.weighting)


if __name__ == "__main__":
    main()
