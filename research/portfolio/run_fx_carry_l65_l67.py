"""fx_carry AUD/JPY -- L65 ruin + L67 portfolio inclusion + carry-premium sensitivity.

Follow-on to `research/exploration/audit_fx_carry.py` (Wave B audit
2026-05-16, verdict CONDITIONAL_WATCHPOINT scoped to long-yen-carry pairs).

Wave B left three explicit open items:

    1. Joint L65 ruin with GEM + turtle at proposed fx_carry weights.
    2. L67 10-metric portfolio matrix vs 60/40 with fx_carry added.
    3. Carry-premium accounting -- pure-price audit understates Sharpe
       by ~+0.6 because the swap differential (~3-4% AUD-JPY rate diff)
       is not in the per-bar return series.

This script closes all three. The carry premium is treated as a
sensitivity (sweep 0%/2%/3%/4% annualised swap yield) rather than a
single point estimate -- the historical AUD-JPY policy differential
varies materially across the 2010-2025 audit window and a single
value would be a fabrication.

Output: `.tmp/reports/fx_carry_l65_l67/result_log.md`

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/run_fx_carry_l65_l67.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.exploration.audit_fx_carry import (  # noqa: E402
    _load_d,
    fx_carry_returns,
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
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "fx_carry_l65_l67"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LIVE_PAIR = "AUD_JPY"
LIVE_SMA = 50
LIVE_VT = 0.08
# Carry-premium sensitivity (annualised swap yield, decimal). Spans the
# realised AUD-JPY policy-rate differential since 2010 (lows ~1% during
# coordinated easing, peaks ~4-5% during BoJ-divergence regimes).
CARRY_PREMIUM_SWEEP = [0.0, 0.02, 0.03, 0.04]
# Candidate fx_carry deployment weights (% of total portfolio).
CARRY_WEIGHT_SWEEP = [0.05, 0.10, 0.15]


def fx_carry_returns_with_premium(
    df: pd.DataFrame,
    *,
    sma_period: int = LIVE_SMA,
    vol_target: float = LIVE_VT,
    annual_swap_yield: float = 0.0,
    direction: int = 1,
) -> pd.Series:
    """fx_carry pure-price returns + daily-pro-rata swap accrual when held.

    The swap yield is applied only on bars where the position is on
    (signal != 0), scaled by the position size (same as the per-bar
    log-return is scaled by held). This mirrors how IBKR posts a daily
    swap on the OVERNIGHT-held notional; entry/exit days share the
    accrual proportionally to held position.

    annual_swap_yield is a constant assumption -- the historical
    differential varies but for sensitivity reporting one number per
    scenario is the right granularity.
    """
    base = fx_carry_returns(df, sma_period=sma_period, vol_target=vol_target, direction=direction)
    # Reconstruct the held position the same way the base function does
    # so we can scale the swap accrual identically.
    close = df["close"]
    sma = close.rolling(sma_period, min_periods=sma_period).mean()
    if direction > 0:
        signal = (close > sma).astype(float)
    else:
        signal = -(close < sma).astype(float)
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    var = log_ret.pow(2).ewm(span=20, adjust=False, min_periods=20).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (vol_target / realised_vol_ann.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)
    position = (signal * scale).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    daily_swap = annual_swap_yield / BARS_PER_YEAR["D"]
    swap_accrual = held * daily_swap * direction
    return (base + swap_accrual).rename("ret")


def _stats(rets: pd.Series) -> dict:
    rets = rets.dropna()
    sr = float(sharpe(rets, periods_per_year=BARS_PER_YEAR["D"]))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        rets, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    return {
        "sharpe": sr,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": int(len(rets)),
    }


def main() -> None:
    print("=" * 80)
    print("fx_carry AUD/JPY -- L65 ruin + L67 portfolio inclusion + premium sensitivity")
    print("=" * 80)

    # ─── [1/5] Build return series under each premium scenario ──────
    print("\n[1/5] AUD/JPY fx_carry returns under carry-premium sensitivity...")
    audjpy_df = _load_d(LIVE_PAIR)
    print(
        f"  {LIVE_PAIR}: {len(audjpy_df)} daily bars, "
        f"{audjpy_df.index[0].date()} -> {audjpy_df.index[-1].date()}"
    )

    scenarios: dict[float, pd.Series] = {}
    print(f"  {'premium':>10s} {'Sharpe':>10s} {'CI_lo':>10s} {'CI_hi':>10s}")
    for prem in CARRY_PREMIUM_SWEEP:
        rets = fx_carry_returns_with_premium(
            audjpy_df, annual_swap_yield=prem, direction=1
        ).dropna()
        rets.index = pd.to_datetime(rets.index).tz_localize(None).normalize()
        rets = rets.groupby(rets.index).last().sort_index()
        scenarios[prem] = rets
        s = _stats(rets)
        print(
            f"  {prem * 100:>+8.1f}%  {s['sharpe']:>+8.4f}  "
            f"{s['ci_lo']:>+8.3f}  {s['ci_hi']:>+8.3f}"
        )

    # ─── [2/5] Existing live-portfolio return series ────────────────
    print("\n[2/5] Loading existing live-portfolio return series...")
    gem = gem_j5_returns().dropna()
    gem.index = pd.to_datetime(gem.index).tz_localize(None).normalize()
    gem = gem.groupby(gem.index).last().sort_index()
    print(f"  gem_j5: {len(gem)} bars ({gem.index[0].date()} -> {gem.index[-1].date()})")

    turtle = turtle_cat_returns_daily().dropna()
    turtle.index = pd.to_datetime(turtle.index).tz_localize(None).normalize()
    turtle = turtle.groupby(turtle.index).last().sort_index()
    print(
        f"  turtle_cat: {len(turtle)} bars ({turtle.index[0].date()} -> {turtle.index[-1].date()})"
    )

    # ─── [3/5] L65 single-strategy ruin per (premium, weight) ─────────
    print("\n[3/5] L65 single-strategy ruin grid (premium x weight)...")
    single_results: list[dict] = []
    print(f"  {'prem':>6s} {'w':>5s} {'P_kill':>9s} {'P95_DD':>9s} {'P50_DD':>9s} {'verdict'}")
    for prem, rets in scenarios.items():
        for w in CARRY_WEIGHT_SWEEP:
            r = assess_strategy_ruin(
                rets,
                deployment_weight=w,
                portfolio_kill_threshold=0.15,
                horizon_bars=252,
                block_size=21,
                n_paths=2000,
                seed=42,
            )
            single_results.append(
                {
                    "premium": prem,
                    "weight": w,
                    "p_kill": r.p_kill_trip,
                    "p95_dd": r.p95_maxdd_at_size,
                    "p50_dd": r.median_maxdd_at_size,
                    "passes": r.passes(),
                }
            )
            verdict = "PASS" if r.passes() else "FAIL"
            print(
                f"  {prem * 100:>+4.1f}%  {w:>4.0%}  {r.p_kill_trip * 100:>7.3f}%  "
                f"{r.p95_maxdd_at_size * 100:>7.2f}%  "
                f"{r.median_maxdd_at_size * 100:>7.2f}%  {verdict}"
            )

    # ─── [4/5] L65 joint ruin (GEM + Turtle + fx_carry) ──────────────
    # Use the 3% premium scenario as the central case for joint analysis
    # (matches AUD-JPY historical median policy differential 2010-2025).
    print("\n[4/5] L65 joint ruin (GEM + Turtle + fx_carry @ 3% swap)...")
    central = scenarios[0.03]
    candidates = [
        {"gem": 0.80, "turtle": 0.20, "fx_carry": 0.00},  # current LIVE
        {"gem": 0.80, "turtle": 0.15, "fx_carry": 0.05},
        {"gem": 0.75, "turtle": 0.15, "fx_carry": 0.10},
        {"gem": 0.70, "turtle": 0.15, "fx_carry": 0.15},
    ]
    joint_results: list[dict] = []
    print(f"  {'weights':>40s} {'P_kill':>9s} {'P95_DD':>9s} {'verdict'}")
    for w in candidates:
        r = assess_joint_ruin(
            strategy_returns={"gem": gem, "turtle": turtle, "fx_carry": central},
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
        w_str = f"G{int(w['gem'] * 100)}/T{int(w['turtle'] * 100)}/F{int(w['fx_carry'] * 100)}"
        verdict = "PASS" if r.passes() else "FAIL"
        print(
            f"  {w_str:>40s}  {r.p_kill_trip * 100:>7.3f}%  "
            f"{r.p95_maxdd_at_size * 100:>7.2f}%  {verdict}"
        )

    # ─── [5/5] L67 10-metric portfolio inclusion under premium sensitivity ──
    print("\n[5/5] L67 10-metric portfolio inclusion (vs 60/40)...")
    bench = benchmark_6040_returns().dropna()
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    bench = bench.groupby(bench.index).last().sort_index()

    current_portfolio = build_portfolio(
        {"gem": gem, "turtle": turtle},
        {"gem": 0.80, "turtle": 0.20},
    ).dropna()
    eval_current = evaluate_portfolio_vs_benchmark(
        current_portfolio,
        bench,
        portfolio_label="CURRENT GEM80/Turtle20",
        benchmark_label="60/40 SPY/IEF",
    )
    print()
    print(eval_current.report())

    # Sensitivity over premium at the central proposed weight (10% fx_carry).
    sensitivity_evals: list[tuple[float, "object"]] = []
    proposed_w = {"gem": 0.75, "turtle": 0.15, "fx_carry": 0.10}
    for prem, rets in scenarios.items():
        proposed = build_portfolio(
            {"gem": gem, "turtle": turtle, "fx_carry": rets},
            proposed_w,
        ).dropna()
        label = f"PROPOSED G75/T15/F10 @ swap={prem * 100:+.1f}%"
        print()
        ev = evaluate_portfolio_vs_benchmark(
            proposed, bench, portfolio_label=label, benchmark_label="60/40 SPY/IEF"
        )
        print(ev.report())
        sensitivity_evals.append((prem, ev))

    # ─── Write result log ─────────────────────────────────────────────
    report_fp = REPORTS_DIR / "result_log.md"
    with report_fp.open("w", encoding="utf-8") as fh:
        fh.write("# fx_carry AUD/JPY -- L65 + L67 Closure\n\n")
        fh.write("**Run date:** 2026-05-19\n")
        fh.write(
            "**Pre-reg / parent audit:** "
            "`.tmp/reports/wave_b_completion/summary.md` "
            "(Wave B 2026-05-16, CONDITIONAL_WATCHPOINT long-yen-carry only)\n"
        )
        fh.write("**Open items closed here:**\n")
        fh.write("- Joint L65 ruin with GEM + turtle at proposed weights\n")
        fh.write("- L67 10-metric portfolio matrix vs 60/40 with fx_carry added\n")
        fh.write("- Carry-premium accounting sensitivity (0%/2%/3%/4% swap)\n\n")

        fh.write("## [1/5] AUD/JPY return series under carry-premium sensitivity\n\n")
        fh.write(
            "Pure-price audit (premium=0%) understates Sharpe by the daily-"
            "accrued swap yield. Sensitivity sweeps the historical AUD-JPY "
            "policy-rate differential range (2010-2025).\n\n"
        )
        fh.write("| Swap yield | Sharpe | CI_lo (95%) | CI_hi (95%) | n bars |\n")
        fh.write("|---:|---:|---:|---:|---:|\n")
        for prem, rets in scenarios.items():
            s = _stats(rets)
            fh.write(
                f"| {prem * 100:+.1f}% | {s['sharpe']:+.4f} | "
                f"{s['ci_lo']:+.3f} | {s['ci_hi']:+.3f} | {s['n']} |\n"
            )

        fh.write("\n## [3/5] L65 single-strategy ruin (premium x weight grid)\n\n")
        fh.write("Kill threshold 15% portfolio DD; horizon 252 bars; 2000 MC paths.\n\n")
        fh.write("| Swap yield | Weight | P_kill | 95th-pct DD | 50th-pct DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|---:|:---:|\n")
        for r in single_results:
            fh.write(
                f"| {r['premium'] * 100:+.1f}% | {r['weight']:.0%} | "
                f"{r['p_kill'] * 100:.3f}% | {r['p95_dd'] * 100:.2f}% | "
                f"{r['p50_dd'] * 100:.2f}% | {'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write("\n## [4/5] L65 joint ruin (GEM + Turtle + fx_carry @ 3% swap central)\n\n")
        fh.write("| GEM | Turtle | fx_carry | P_kill | 95th-pct DD | Verdict |\n")
        fh.write("|---:|---:|---:|---:|---:|:---:|\n")
        for r in joint_results:
            w = r["weights"]
            fh.write(
                f"| {w['gem']:.0%} | {w['turtle']:.0%} | {w['fx_carry']:.0%} | "
                f"{r['p_kill'] * 100:.3f}% | {r['p95_dd'] * 100:.2f}% | "
                f"{'PASS' if r['passes'] else 'FAIL'} |\n"
            )

        fh.write("\n## [5/5] L67 10-metric portfolio inclusion\n\n")
        fh.write("### CURRENT (no fx_carry)\n\n")
        fh.write("```\n" + eval_current.report() + "\n```\n\n")
        fh.write("### PROPOSED -- G75/T15/F10 sensitivity over swap yield\n\n")
        for prem, ev in sensitivity_evals:
            fh.write(f"#### swap = {prem * 100:+.1f}%\n\n")
            fh.write("```\n" + ev.report() + "\n```\n\n")

        fh.write("## Verdict\n\n")
        fh.write(
            "See the result log tables. The closure decision is operator-"
            "driven: deploy if joint ruin PASSES at the chosen weight AND "
            "the L67 matrix shows >=8/10 metrics improved vs 60/40 at the "
            "central swap assumption (3%). Otherwise leave fx_carry as "
            "audited-but-not-deployed.\n"
        )

    print(f"\nResult log: {report_fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
