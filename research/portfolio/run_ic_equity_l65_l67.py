"""ic_equity_daily -- L65 ruin + L67 portfolio inclusion (Wave B closure).

Follow-on to `research/exploration/audit_ic_equity_daily.py` (Wave B 2026-05-16).
Verdict was "CONDITIONAL_WATCHPOINT candidate" pending joint L65 + L67.

Wave B findings:

    All 7 tickers strict-OOS Sharpe > 0 (mean +0.62, median +0.61)
    Top-3 (by strict OOS Sharpe): HWM +1.12, WMT +0.95, SYK +0.72
    V1-style overstated by ~+2.56 (single look-ahead class)
    Recommendation: deploy top-3 at 5-10% portfolio weight total

This script closes the recommendation by:

    1. Reproducing the strict-OOS per-ticker return series (IS-fit ic_sign
       + IS-mu/sigma; OOS slice = back half of each ticker's series).
    2. Building an equal-weighted top-3 portfolio.
    3. L65 single-strategy ruin at proposed weights (5%, 10%, 15%).
    4. L65 joint ruin with GEM + turtle.
    5. L67 10-metric portfolio matrix vs 60/40 with ic_equity added.

The top-3 selection is itself a selection-bias risk (the 7-ticker panel
is small; picking the best 3 OOS can overfit). Mitigate by also reporting
the equal-weighted ALL-7 basket as a robustness check -- if all-7 also
clears the gates the top-3 verdict is not driven by tail-of-7
selection.

Output: `.tmp/reports/ic_equity_l65_l67/result_log.md`

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/run_ic_equity_l65_l67.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.exploration.audit_ic_equity_daily import (  # noqa: E402
    PERIODS_PER_YEAR,
    THRESHOLDS,
    TICKERS_LIVE,
    _load_d,
    ic_equity_returns,
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
)
from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ic_equity_l65_l67"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Wave B 2026-05-16 strict-OOS rank: HWM > WMT > SYK > SYK / NOC > CB > ABNB > GL
TOP3_TICKERS = ["HWM", "WMT", "SYK"]

# Candidate basket deployment weights (% of total portfolio).
WEIGHT_SWEEP = [0.05, 0.10, 0.15]


def strict_oos_returns_for_ticker(ticker: str) -> pd.Series:
    """Reproduce the Wave B strict-OOS return series for one ticker.

    Identical to the audit: IS-fit ic_sign and IS-mu/sigma on the first
    half of available bars, OOS = back half. No look-ahead.
    """
    df = _load_d(ticker)
    n = len(df)
    is_split = n // 2
    ret_full = ic_equity_returns(df, threshold=THRESHOLDS[ticker], is_until_idx=is_split)
    oos = ret_full.iloc[is_split:].rename(f"ic_{ticker.lower()}")
    oos.index = pd.to_datetime(oos.index).tz_localize(None).normalize()
    return oos.groupby(oos.index).last().sort_index()


def equal_weight_basket(returns_by_ticker: dict[str, pd.Series]) -> pd.Series:
    """Equal-weighted daily portfolio return across a dict of strategy
    return series. Index-aligned outer-join with zero-fill (a missing
    ticker on a date contributes 0, the weight stays on cash for that
    sleeve so the basket de-leverages -- the standard interpretation).
    """
    df = pd.concat(returns_by_ticker, axis=1).fillna(0.0)
    n = df.shape[1]
    return (df.sum(axis=1) / n).rename("basket")


def _stats(rets: pd.Series, label: str) -> dict:
    rets = rets.dropna()
    sr = float(sharpe(rets, periods_per_year=PERIODS_PER_YEAR))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        rets, periods_per_year=PERIODS_PER_YEAR, n_resamples=2000, seed=42
    )
    return {
        "label": label,
        "sharpe": sr,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": int(len(rets)),
    }


def main() -> None:
    print("=" * 80)
    print("ic_equity_daily -- L65 ruin + L67 portfolio inclusion (Wave B closure)")
    print("=" * 80)

    # ─── [1/5] Build per-ticker strict-OOS return series ──────────
    print("\n[1/5] Strict-OOS per-ticker return series (IS-fit + OOS-only Sharpe)")
    print(
        f"  {'ticker':>7s}  {'thr':>5s}  {'Sharpe':>10s}  {'CI_lo':>9s}  {'CI_hi':>9s}  {'n':>6s}"
    )
    rets_by_ticker: dict[str, pd.Series] = {}
    summary_stats: list[dict] = []
    for tkr in TICKERS_LIVE:
        try:
            rets = strict_oos_returns_for_ticker(tkr)
            rets_by_ticker[tkr] = rets
            s = _stats(rets, tkr)
            summary_stats.append(s)
            print(
                f"  {tkr:>7s}  {THRESHOLDS[tkr]:>4.2f}  "
                f"{s['sharpe']:>+9.4f}  {s['ci_lo']:>+8.3f}  "
                f"{s['ci_hi']:>+8.3f}  {s['n']:>6d}"
            )
        except FileNotFoundError as e:
            print(f"  {tkr}: missing parquet ({e})")

    # ─── [2/5] Build top-3 and all-7 basket return series ──────────
    print("\n[2/5] Basket return series (top-3 equal-weight, all-7 equal-weight)")
    top3 = {k: rets_by_ticker[k] for k in TOP3_TICKERS if k in rets_by_ticker}
    basket_top3 = equal_weight_basket(top3)
    s_top3 = _stats(basket_top3, "top3_basket")
    print(
        f"  TOP-3 EQ basket  (HWM+WMT+SYK):  Sharpe={s_top3['sharpe']:+.4f}  "
        f"CI=[{s_top3['ci_lo']:+.3f}, {s_top3['ci_hi']:+.3f}]  n={s_top3['n']}"
    )
    basket_all7 = equal_weight_basket(rets_by_ticker)
    s_all7 = _stats(basket_all7, "all7_basket")
    print(
        f"  ALL-7 EQ basket               :  Sharpe={s_all7['sharpe']:+.4f}  "
        f"CI=[{s_all7['ci_lo']:+.3f}, {s_all7['ci_hi']:+.3f}]  n={s_all7['n']}"
    )

    # ─── [3/5] L65 single-strategy ruin on each basket ──────────
    print("\n[3/5] L65 single-strategy ruin (basket x weight)")
    print(f"  {'basket':>7s}  {'w':>5s}  {'P_kill':>9s}  {'P95_DD':>9s}  {'P50_DD':>9s}  verdict")
    single_results: list[dict] = []
    for label, basket in (("top3", basket_top3), ("all7", basket_all7)):
        for w in WEIGHT_SWEEP:
            r = assess_strategy_ruin(
                basket,
                deployment_weight=w,
                portfolio_kill_threshold=0.15,
                horizon_bars=252,
                block_size=21,
                n_paths=2000,
                seed=42,
            )
            single_results.append(
                {
                    "basket": label,
                    "weight": w,
                    "p_kill": r.p_kill_trip,
                    "p95_dd": r.p95_maxdd_at_size,
                    "p50_dd": r.median_maxdd_at_size,
                    "passes": r.passes(),
                }
            )
            verdict = "PASS" if r.passes() else "FAIL"
            print(
                f"  {label:>7s}  {w:>4.0%}  "
                f"{r.p_kill_trip * 100:>7.3f}%  "
                f"{r.p95_maxdd_at_size * 100:>7.2f}%  "
                f"{r.median_maxdd_at_size * 100:>7.2f}%  {verdict}"
            )

    # ─── [4/5] L65 joint ruin (GEM + Turtle + ic_equity basket) ──────────
    print("\n[4/5] L65 joint ruin (GEM + Turtle + ic_equity top-3 basket)")
    gem = gem_j5_returns().dropna()
    gem.index = pd.to_datetime(gem.index).tz_localize(None).normalize()
    gem = gem.groupby(gem.index).last().sort_index()
    print(f"  gem_j5: {len(gem)} bars ({gem.index[0].date()} -> {gem.index[-1].date()})")

    turtle = turtle_cat_returns_daily().dropna()
    turtle.index = pd.to_datetime(turtle.index).tz_localize(None).normalize()
    turtle = turtle.groupby(turtle.index).last().sort_index()
    print(f"  turtle: {len(turtle)} bars ({turtle.index[0].date()} -> {turtle.index[-1].date()})")
    print(f"  ic_top3: {len(basket_top3)} bars")

    candidates = [
        {"gem": 0.80, "turtle": 0.20, "ic_top3": 0.00},  # current LIVE reference
        {"gem": 0.80, "turtle": 0.15, "ic_top3": 0.05},
        {"gem": 0.75, "turtle": 0.15, "ic_top3": 0.10},
        {"gem": 0.70, "turtle": 0.15, "ic_top3": 0.15},
    ]
    joint_results: list[dict] = []
    print(f"  {'weights':>40s}  {'P_kill':>9s}  {'P95_DD':>9s}  verdict")
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={"gem": gem, "turtle": turtle, "ic_top3": basket_top3},
            deployment_weights=w,
            portfolio_kill_threshold=0.15,
            horizon_bars=252,
            block_size=21,
            n_paths=2000,
            seed=42,
        )
        joint_results.append(
            {
                "weights": w,
                "p_kill": r.p_kill_trip,
                "p95_dd": r.p95_maxdd_at_size,
                "p50_dd": r.median_maxdd_at_size,
                "passes": r.passes(),
            }
        )
        w_str = f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/IC{int(w['ic_top3'] * 100)}"
        verdict = "PASS" if r.passes() else "FAIL"
        print(
            f"  {w_str:>40s}  {r.p_kill_trip * 100:>7.3f}%  "
            f"{r.p95_maxdd_at_size * 100:>7.2f}%  {verdict}"
        )

    # ─── [5/5] L67 10-metric portfolio inclusion ──────────
    print("\n[5/5] L67 10-metric portfolio inclusion (vs 60/40)")
    bench = benchmark_6040_returns().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    bench = bench.groupby(bench.index).last().sort_index()

    current = build_portfolio(
        {"gem": gem, "turtle": turtle},
        {"gem": 0.80, "turtle": 0.20},
    ).dropna()
    eval_current = evaluate_portfolio_vs_benchmark(
        current,
        bench,
        portfolio_label="CURRENT GEM80/Turtle20",
        benchmark_label="60/40 SPY/IEF",
    )
    print()
    print(eval_current.report())

    eval_proposals: list[tuple[str, "object"]] = []
    for label, basket in (("top3", basket_top3), ("all7", basket_all7)):
        for w_ic in [0.05, 0.10]:
            w_t = 0.20 - max(0, w_ic - 0.05)  # carve from turtle
            w_g = 1.0 - w_t - w_ic
            weights = {"gem": w_g, "turtle": w_t, f"ic_{label}": w_ic}
            proposed = build_portfolio(
                {"gem": gem, "turtle": turtle, f"ic_{label}": basket},
                weights,
            ).dropna()
            tag = f"PROPOSED G{int(w_g * 100)}/T{int(w_t * 100)}/IC{label.upper()}{int(w_ic * 100)}"
            print()
            ev = evaluate_portfolio_vs_benchmark(
                proposed, bench, portfolio_label=tag, benchmark_label="60/40 SPY/IEF"
            )
            print(ev.report())
            eval_proposals.append((tag, ev))

    # ─── Write result log ──────────
    report_fp = REPORTS_DIR / "result_log.md"
    with report_fp.open("w", encoding="utf-8") as fh:
        fh.write("# ic_equity_daily -- L65 + L67 Closure\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write(
            "**Parent audit:** `.tmp/reports/wave_b_completion/summary.md` "
            "(Wave B 2026-05-16, CONDITIONAL_WATCHPOINT candidate)\n"
        )
        fh.write("**Open items closed:** joint L65 ruin + L67 10-metric inclusion test.\n\n")

        fh.write("## [1/5] Per-ticker strict-OOS Sharpe (reproduced)\n\n")
        fh.write("| Ticker | Threshold | Strict OOS Sharpe | CI_lo | CI_hi | n bars |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for s in summary_stats:
            fh.write(
                f"| {s['label']} | {THRESHOLDS[s['label']]:.2f} | "
                f"{s['sharpe']:+.4f} | {s['ci_lo']:+.3f} | "
                f"{s['ci_hi']:+.3f} | {s['n']} |\n"
            )

        fh.write("\n## [2/5] Basket return series (equal-weighted)\n\n")
        fh.write(
            f"- **Top-3 (HWM+WMT+SYK)**: Sharpe={s_top3['sharpe']:+.4f}, "
            f"CI=[{s_top3['ci_lo']:+.3f}, {s_top3['ci_hi']:+.3f}], n={s_top3['n']}\n"
        )
        fh.write(
            f"- **All-7**: Sharpe={s_all7['sharpe']:+.4f}, "
            f"CI=[{s_all7['ci_lo']:+.3f}, {s_all7['ci_hi']:+.3f}], n={s_all7['n']}\n"
        )

        fh.write("\n## [3/5] L65 single-strategy ruin\n\n")
        fh.write("Kill threshold 15% portfolio DD; horizon 252 bars; 2000 MC paths.\n\n")
        fh.write("| Basket | Weight | P_kill | 95th-pct DD | 50th-pct DD | Verdict |\n")
        fh.write("|---|---:|---:|---:|---:|:---:|\n")
        for r in single_results:
            fh.write(
                f"| {r['basket']} | {r['weight']:.0%} | "
                f"{r['p_kill'] * 100:.3f}% | {r['p95_dd'] * 100:.2f}% | "
                f"{r['p50_dd'] * 100:.2f}% | {'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write("\n## [4/5] L65 joint ruin (GEM + Turtle + ic_top3)\n\n")
        fh.write("| GEM | Turtle | ic_top3 | P_kill | 95th-pct DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|---:|:---:|\n")
        for r in joint_results:
            w = r["weights"]
            fh.write(
                f"| {w['gem']:.0%} | {w['turtle']:.0%} | {w['ic_top3']:.0%} | "
                f"{r['p_kill'] * 100:.3f}% | {r['p95_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write("\n## [5/5] L67 10-metric portfolio inclusion\n\n")
        fh.write("### CURRENT (no ic_equity)\n\n")
        fh.write("```\n" + eval_current.report() + "\n```\n\n")
        for tag, ev in eval_proposals:
            fh.write(f"### {tag}\n\n")
            fh.write("```\n" + ev.report() + "\n```\n\n")

        fh.write("## Verdict\n\n")
        fh.write(
            "See tables. Closure decision is operator-driven: deploy if joint "
            "ruin PASSES at the chosen weight AND L67 matrix shows >= 8/10 "
            "metrics (or measurably improves over the current GEM80/Turtle20 "
            "7/10 baseline). Otherwise leave ic_equity_daily as audited-but-"
            "not-deployed.\n"
        )

    print(f"\nResult log: {report_fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
