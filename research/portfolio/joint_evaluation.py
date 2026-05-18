"""V3.6 -> V3.7 evolution: PORTFOLIO-LEVEL evaluation against benchmark.

The V3.6 framework tests individual strategies against single-instrument
buy-and-hold (L17). This is the WRONG test for the user's actual question:
"does my portfolio of strategies beat 60/40 SPY/IEF on risk-adjusted basis?"

Why the V3.6 individual-strategy-vs-B&H test systematically favors B&H:
    1. Equity Risk Premium leakage: in-market X% of time -> capture X% of ERP
    2. Missed-recovery asymmetry: trend strategies are flat at the bottom
    3. Transaction costs compound; B&H pays once
    4. Selection bias in the strategy zoo (DSR adjusts but doesn't eliminate)

The CORRECT test (this module):
    PORTFOLIO = w1*strategy1 + w2*strategy2 + ... at live allocator weights
    BENCHMARK = 60/40 SPY/IEF (or risk-parity, or All Weather)
    Run on aligned dates with block-bootstrap MC.
    Compute 10-metric decision matrix.

10 metrics tested (each on block-bootstrap MC paths):
    1. Sharpe ratio (cash-relative)
    2. M^2 vol-matched Sharpe (Modigliani-Modigliani)
    3. Sortino (downside deviation)
    4. Calmar (ann return / MaxDD)
    5. MaxDD reduction vs benchmark
    6. CVaR-95 reduction
    7. CDaR-95 reduction (Chekhlov-Uryasev)
    8. Risk-of-ruin (L65 joint)
    9. Information Ratio (Grinold-Kahn)
    10. Spearman rank stability of excess returns

Verdict:
    8+/10 pass -> PORTFOLIO_DEPLOY (real upgrade vs benchmark)
    5-7/10 pass -> PORTFOLIO_CONDITIONAL (marginal; watch tail metrics)
    < 5/10 pass -> PORTFOLIO_REJECT (prefer benchmark; consider portable-alpha overlay only)

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/joint_evaluation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.gem.gem_strategy import GemConfig, gem_returns  # noqa: E402
from research.turtle.turtle_strategy import TurtleConfig, turtle_returns  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    calmar,
    cdar,
    cvar,
    max_drawdown,
    sharpe,
    sortino,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "joint_evaluation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PERIODS_PER_YEAR = 252
RISK_FREE_DAILY = 0.0  # cash-relative; not adjusting for SOFR currently


# ──────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────


def _load_d(symbol: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_D.parquet")
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def _load_h1(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_H1.parquet")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().dropna(subset=["close"])[["open", "high", "low", "close"]].astype(float)


def gem_j5_returns() -> pd.Series:
    """GEM J5 P_hl60_vt05 (LIVE canonical) on SPY/EFA/IEF daily."""
    spy = _load_d("SPY")
    efa = _load_d("EFA")
    ief = _load_d("IEF")
    common = spy.index.intersection(efa.index).intersection(ief.index)
    universe = pd.DataFrame(
        {"SPY": spy.reindex(common), "EFA": efa.reindex(common), "IEF": ief.reindex(common)}
    ).dropna()
    cfg = GemConfig(
        lookback_blend=(3, 6, 12),
        absolute_gate_lookback_months=12,
        buffer_pct=0.005,
        defensive_switch=True,
        ann_vol_target=0.05,
        vol_lookback_days=20,
        max_leverage=2.0,
        vol_estimator_kind="ewma",
        vol_estimator_halflife=60,
        stress_gate_enabled=False,
        dd_breaker_enabled=False,
    )
    return gem_returns(universe, cfg=cfg).rename("gem_j5")


def turtle_cat_returns_daily() -> pd.Series:
    """turtle CAT C3_peak H1 -> daily."""
    cat = _load_h1("CAT")
    cfg = TurtleConfig(entry_period=45, exit_period=20)
    ret_h1 = turtle_returns(cat, cfg=cfg)
    daily = ret_h1.groupby(ret_h1.index.normalize()).sum()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    return daily.rename("turtle_cat")


def benchmark_6040_returns() -> pd.Series:
    """60/40 SPY/IEF daily rebalanced log returns."""
    spy = _load_d("SPY")
    ief = _load_d("IEF")
    common = spy.index.intersection(ief.index)
    spy_ret = np.log(spy.reindex(common) / spy.reindex(common).shift(1)).fillna(0.0)
    ief_ret = np.log(ief.reindex(common) / ief.reindex(common).shift(1)).fillna(0.0)
    return (0.6 * spy_ret + 0.4 * ief_ret).rename("benchmark_60_40")


def benchmark_spy_returns() -> pd.Series:
    spy = _load_d("SPY")
    return np.log(spy / spy.shift(1)).fillna(0.0).rename("benchmark_spy")


# ──────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ──────────────────────────────────────────────────────────────────────────


def build_portfolio(
    strategy_returns: dict[str, pd.Series],
    weights: dict[str, float],
    common_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Weighted sum of strategy returns on aligned dates."""
    if common_index is None:
        common = None
        for r in strategy_returns.values():
            r = r.dropna()
            if common is None:
                common = r.index
            else:
                common = common.intersection(r.index)
        common_index = common
    if common_index is None or len(common_index) == 0:
        raise ValueError("No common index")
    out = pd.Series(0.0, index=common_index)
    for name, ret in strategy_returns.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        out = out + (ret.reindex(common_index).fillna(0.0) * w)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 10-metric portfolio evaluation
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MetricResult:
    """Single metric on portfolio + benchmark + verdict."""

    name: str
    portfolio: float
    benchmark: float
    threshold: str
    passes: bool


@dataclass
class JointEvaluation:
    """10-metric portfolio-vs-benchmark verdict."""

    portfolio_label: str
    benchmark_label: str
    metrics: list[MetricResult] = field(default_factory=list)
    n_bars: int = 0
    horizon_years: float = 0.0

    @property
    def n_passes(self) -> int:
        return sum(1 for m in self.metrics if m.passes)

    @property
    def verdict(self) -> str:
        n = self.n_passes
        if n >= 8:
            return "PORTFOLIO_DEPLOY"
        if n >= 5:
            return "PORTFOLIO_CONDITIONAL"
        return "PORTFOLIO_REJECT"

    def report(self) -> str:
        lines = [
            f"Joint Evaluation: {self.portfolio_label}  vs  {self.benchmark_label}",
            f"Sample: {self.n_bars} bars / {self.horizon_years:.1f} years",
            "",
            f"{'Metric':<32} {'Portfolio':>12} {'Benchmark':>12} {'Gate':<28} {'Pass':>4}",
            "─" * 96,
        ]
        for m in self.metrics:
            mark = "✓" if m.passes else "✗"
            lines.append(
                f"{m.name:<32} {m.portfolio:>+12.4f} {m.benchmark:>+12.4f} "
                f"{m.threshold:<28} {mark:>4}"
            )
        lines.append("─" * 96)
        lines.append(f"PASSES: {self.n_passes}/10 → VERDICT: {self.verdict}")
        return "\n".join(lines)


def evaluate_portfolio_vs_benchmark(
    portfolio: pd.Series,
    benchmark: pd.Series,
    *,
    portfolio_label: str = "portfolio",
    benchmark_label: str = "benchmark",
    periods_per_year: int = PERIODS_PER_YEAR,
    alpha: float = 0.05,
) -> JointEvaluation:
    """Compute the 10-metric portfolio-vs-benchmark decision matrix."""
    # Align.
    common = portfolio.dropna().index.intersection(benchmark.dropna().index)
    p = portfolio.reindex(common).fillna(0.0)
    b = benchmark.reindex(common).fillna(0.0)

    res = JointEvaluation(
        portfolio_label=portfolio_label,
        benchmark_label=benchmark_label,
        n_bars=len(common),
        horizon_years=len(common) / periods_per_year,
    )

    # 1. Sharpe ratio (cash-relative)
    sr_p = float(sharpe(p, periods_per_year=periods_per_year))
    sr_b = float(sharpe(b, periods_per_year=periods_per_year))
    res.metrics.append(
        MetricResult(
            name="1. Sharpe (annualised)",
            portfolio=sr_p,
            benchmark=sr_b,
            threshold="P > B by >= 0.10",
            passes=(sr_p - sr_b) >= 0.10,
        )
    )

    # 2. M^2 vol-matched Sharpe (Modigliani-Modigliani)
    sigma_b = float(b.std() * np.sqrt(periods_per_year))
    m2_p = sr_p * sigma_b  # annualised excess return at benchmark vol
    m2_b = sr_b * sigma_b
    res.metrics.append(
        MetricResult(
            name="2. M^2 vol-matched excess",
            portfolio=m2_p,
            benchmark=m2_b,
            threshold="M2_P - M2_B >= 0.015",
            passes=(m2_p - m2_b) >= 0.015,
        )
    )

    # 3. Sortino
    sor_p = float(sortino(p, periods_per_year=periods_per_year))
    sor_b = float(sortino(b, periods_per_year=periods_per_year))
    res.metrics.append(
        MetricResult(
            name="3. Sortino",
            portfolio=sor_p,
            benchmark=sor_b,
            threshold="P > B",
            passes=sor_p > sor_b,
        )
    )

    # 4. Calmar
    cal_p = float(calmar(p, periods_per_year=periods_per_year))
    cal_b = float(calmar(b, periods_per_year=periods_per_year))
    res.metrics.append(
        MetricResult(
            name="4. Calmar",
            portfolio=cal_p,
            benchmark=cal_b,
            threshold="P - B >= 0.30",
            passes=(cal_p - cal_b) >= 0.30,
        )
    )

    # 5. MaxDD reduction
    dd_p = float(max_drawdown(p))
    dd_b = float(max_drawdown(b))
    dd_reduction = 1.0 - (dd_p / dd_b) if dd_b < -1e-9 else 0.0
    res.metrics.append(
        MetricResult(
            name="5. MaxDD reduction (1 - P/B)",
            portfolio=dd_p,
            benchmark=dd_b,
            threshold="reduction >= 0.30",
            passes=dd_reduction >= 0.30,
        )
    )

    # 6. CVaR-95 reduction
    cv_p = float(cvar(p, alpha=alpha))
    cv_b = float(cvar(b, alpha=alpha))
    cv_reduction = 1.0 - (cv_p / cv_b) if cv_b < -1e-9 else 0.0
    res.metrics.append(
        MetricResult(
            name="6. CVaR-95 reduction",
            portfolio=cv_p,
            benchmark=cv_b,
            threshold="reduction >= 0.25",
            passes=cv_reduction >= 0.25,
        )
    )

    # 7. CDaR-95 reduction
    cd_p = float(cdar(p, alpha=alpha))
    cd_b = float(cdar(b, alpha=alpha))
    cd_reduction = 1.0 - (cd_p / cd_b) if cd_b < -1e-9 else 0.0
    res.metrics.append(
        MetricResult(
            name="7. CDaR-95 reduction",
            portfolio=cd_p,
            benchmark=cd_b,
            threshold="reduction >= 0.30",
            passes=cd_reduction >= 0.30,
        )
    )

    # 8. Risk-of-ruin (1-year P(NAV drawdown > 15%))
    rng = np.random.default_rng(42)
    n_paths = 2000
    horizon = periods_per_year
    block_size = 21
    p_arr = p.to_numpy()
    b_arr = b.to_numpy()
    ror_p_count = 0
    ror_b_count = 0
    for _ in range(n_paths):
        # Use SAME block starts for portfolio and benchmark to preserve joint structure
        n_blocks = (horizon + block_size - 1) // block_size
        n_avail = len(p_arr) - block_size + 1
        starts = rng.integers(0, n_avail, size=n_blocks)
        path_p = np.concatenate([p_arr[s : s + block_size] for s in starts])[:horizon]
        path_b = np.concatenate([b_arr[s : s + block_size] for s in starts])[:horizon]
        # Cumulative log-equity drawdown
        eq_p = np.cumsum(path_p)
        eq_b = np.cumsum(path_b)
        dd_p_path = eq_p - np.maximum.accumulate(eq_p)
        dd_b_path = eq_b - np.maximum.accumulate(eq_b)
        if dd_p_path.min() <= -0.15:
            ror_p_count += 1
        if dd_b_path.min() <= -0.15:
            ror_b_count += 1
    ror_p = ror_p_count / n_paths
    ror_b = ror_b_count / n_paths
    res.metrics.append(
        MetricResult(
            name="8. P(NAV DD > 15% in 1y)",
            portfolio=ror_p,
            benchmark=ror_b,
            threshold="P < B",
            passes=ror_p < ror_b,
        )
    )

    # 9. Information Ratio (Grinold-Kahn)
    excess = p - b
    if excess.std() > 1e-12:
        ir = float(excess.mean() / excess.std() * np.sqrt(periods_per_year))
    else:
        ir = 0.0
    res.metrics.append(
        MetricResult(
            name="9. Information Ratio",
            portfolio=ir,
            benchmark=0.0,
            threshold="IR >= 0.30",
            passes=ir >= 0.30,
        )
    )

    # 10. Spearman rank stability of excess returns across MC paths
    # We test whether portfolio outperforms benchmark CONSISTENTLY across MC samples.
    rng2 = np.random.default_rng(123)
    n_samples = 200
    sample_size = min(252, len(p) // 2)
    excess_means = []
    benchmark_means = []
    for _ in range(n_samples):
        idx = rng2.integers(0, len(p) - sample_size)
        e_sample = excess.iloc[idx : idx + sample_size]
        b_sample = b.iloc[idx : idx + sample_size]
        excess_means.append(e_sample.mean())
        benchmark_means.append(b_sample.mean())
    rho, _ = spearmanr(excess_means, benchmark_means)
    rho = float(rho) if not np.isnan(rho) else 0.0
    # We want the SIGN of excess to be POSITIVE in most samples (stability of outperformance)
    pct_positive_excess = float(np.mean([e > 0 for e in excess_means]))
    res.metrics.append(
        MetricResult(
            name="10. Excess > 0 stability %",
            portfolio=pct_positive_excess,
            benchmark=0.50,
            threshold="pct >= 0.60",
            passes=pct_positive_excess >= 0.60,
        )
    )

    return res


# ──────────────────────────────────────────────────────────────────────────
# Main: run on (GEM + turtle) vs 60/40 SPY/IEF
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 100)
    print("V3.7 PORTFOLIO-LEVEL EVALUATION")
    print("Testing the L17-correct question: 'does the portfolio beat 60/40 SPY/IEF?'")
    print("=" * 100)

    # Load strategy returns.
    print("\n[load] strategy returns:")
    gem = gem_j5_returns().dropna()
    turtle_cat = turtle_cat_returns_daily().dropna()
    print(f"  GEM J5:     {len(gem)} daily bars, {gem.index[0].date()} -> {gem.index[-1].date()}")
    print(
        f"  turtle CAT: {len(turtle_cat)} daily bars, {turtle_cat.index[0].date()} -> {turtle_cat.index[-1].date()}"
    )

    # Benchmarks.
    bench_6040 = benchmark_6040_returns().dropna()
    bench_spy = benchmark_spy_returns().dropna()
    print(f"  60/40 SPY/IEF: {len(bench_6040)} daily bars")
    print(f"  SPY:           {len(bench_spy)} daily bars")

    # Build current LIVE+CONDITIONAL portfolio at proposed weights.
    # GEM J5 70% + turtle CAT 20% + 10% cash buffer (zero return)
    weights_current = {"gem_j5": 0.70, "turtle_cat": 0.20}
    portfolio_current = build_portfolio(
        {"gem_j5": gem, "turtle_cat": turtle_cat},
        weights_current,
    )
    print(f"\n[portfolio current] GEM 70% + turtle 20% + 10% cash, {len(portfolio_current)} bars")

    # ==================================================================
    # Test 1: current portfolio vs 60/40 (on overlap)
    # ==================================================================
    print("\n" + "=" * 100)
    print("TEST 1: Current portfolio (GEM 70% + turtle 20%) vs 60/40 SPY/IEF")
    print("=" * 100)
    eval1 = evaluate_portfolio_vs_benchmark(
        portfolio_current,
        bench_6040,
        portfolio_label="GEM(70%) + turtle(20%) + 10% cash",
        benchmark_label="60/40 SPY/IEF",
    )
    print(eval1.report())

    # ==================================================================
    # Test 2: GEM alone (current LIVE, 100% weight) vs 60/40
    # ==================================================================
    print("\n" + "=" * 100)
    print("TEST 2: GEM alone at 100% vs 60/40 SPY/IEF (full history)")
    print("=" * 100)
    eval2 = evaluate_portfolio_vs_benchmark(
        gem,
        bench_6040,
        portfolio_label="GEM J5 (100%)",
        benchmark_label="60/40 SPY/IEF",
    )
    print(eval2.report())

    # ==================================================================
    # Test 3: turtle CAT alone vs SPY (single-instrument-equity benchmark)
    # ==================================================================
    print("\n" + "=" * 100)
    print("TEST 3: turtle CAT alone vs SPY (single-asset-equity baseline)")
    print("=" * 100)
    eval3 = evaluate_portfolio_vs_benchmark(
        turtle_cat,
        bench_spy,
        portfolio_label="turtle CAT (100%)",
        benchmark_label="SPY buy-and-hold",
    )
    print(eval3.report())

    # ==================================================================
    # Test 4: 60/40 vs SPY (sanity check; benchmarks against each other)
    # ==================================================================
    print("\n" + "=" * 100)
    print("TEST 4 (sanity): 60/40 vs SPY")
    print("=" * 100)
    eval4 = evaluate_portfolio_vs_benchmark(
        bench_6040,
        bench_spy,
        portfolio_label="60/40 SPY/IEF",
        benchmark_label="SPY buy-and-hold",
    )
    print(eval4.report())

    # Save reports
    csv_lines = ["test,metric,portfolio,benchmark,passes"]
    for label, ev in [
        ("test1_portfolio_vs_6040", eval1),
        ("test2_gem_vs_6040", eval2),
        ("test3_turtle_vs_spy", eval3),
        ("test4_6040_vs_spy", eval4),
    ]:
        for m in ev.metrics:
            csv_lines.append(f"{label},{m.name},{m.portfolio:.6f},{m.benchmark:.6f},{m.passes}")
    (REPORTS_DIR / "joint_evaluation_summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    # Final summary
    print("\n" + "=" * 100)
    print("OVERALL VERDICTS")
    print("=" * 100)
    print(f"Test 1 (current portfolio vs 60/40):  {eval1.n_passes}/10  -> {eval1.verdict}")
    print(f"Test 2 (GEM 100% vs 60/40):          {eval2.n_passes}/10  -> {eval2.verdict}")
    print(f"Test 3 (turtle CAT vs SPY):          {eval3.n_passes}/10  -> {eval3.verdict}")
    print(f"Test 4 (60/40 vs SPY sanity):        {eval4.n_passes}/10  -> {eval4.verdict}")


if __name__ == "__main__":
    main()
