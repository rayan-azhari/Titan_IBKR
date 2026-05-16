"""Joint risk-of-ruin assessment: GEM J5 (LIVE) + turtle CAT (proposed CONDITIONAL).

Per L65 protocol: before deploying turtle CAT alongside the existing live
GEM J5, compute portfolio-level joint ruin probability accounting for
empirical cross-correlations between the two strategies.

Caveat: GEM J5 is DAILY cadence (SPY/EFA/IEF + IEF buffer); turtle CAT is
H1 cadence (US equity RTH). Joint MC requires aligning to a common
cadence. Approach: downsample turtle CAT to daily by summing per-day
H1 returns (correct since position is binary long-only and returns are
log returns), then bootstrap the daily portfolio.

Result reported under L65 default gate.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/portfolio/joint_ruin_gem_turtle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.gem.gem_strategy import GemConfig, gem_returns  # noqa: E402
from research.turtle.turtle_strategy import (  # noqa: E402
    TurtleConfig,
    turtle_returns,
)
from titan.research.framework.ruin import (  # noqa: E402
    assess_joint_ruin,
    assess_strategy_ruin,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "joint_ruin_gem_turtle"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_d(symbol: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_D.parquet")
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def _load_h1(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_H1.parquet")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().dropna(subset=["close"])[["open", "high", "low", "close"]].astype(float)


def gem_j5_canonical_returns() -> pd.Series:
    """Compute GEM P_hl60_vt05 (LIVE canonical) returns on SPY/EFA/IEF daily."""
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
    return gem_returns(universe, cfg=cfg)


def turtle_cat_c3peak_returns_daily() -> pd.Series:
    """Compute turtle CAT C3_peak (entry=45, exit=20) H1 returns and downsample to daily."""
    cat = _load_h1("CAT")
    cfg = TurtleConfig(entry_period=45, exit_period=20)
    ret_h1 = turtle_returns(cat, cfg=cfg)
    # Downsample: sum log returns per UTC date.
    daily = ret_h1.groupby(ret_h1.index.normalize()).sum()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    return daily


def main() -> None:
    print("=" * 72)
    print("Joint Ruin Assessment: GEM J5 (LIVE) + turtle CAT (proposed CONDITIONAL)")
    print("=" * 72)

    print("\n[load] computing GEM J5 P_hl60_vt05 returns...")
    gem_ret = gem_j5_canonical_returns().dropna()
    print(f"  GEM J5 daily returns: {len(gem_ret)} bars, "
          f"{gem_ret.index[0].date()} -> {gem_ret.index[-1].date()}")
    print(f"  Mean = {gem_ret.mean():.6f}, std = {gem_ret.std():.6f}")
    print(f"  Annualised vol = {gem_ret.std() * np.sqrt(252):.3%}")

    print("\n[load] computing turtle CAT C3_peak (entry=45, exit=20) H1->daily returns...")
    turtle_ret = turtle_cat_c3peak_returns_daily().dropna()
    print(f"  turtle CAT daily returns: {len(turtle_ret)} bars, "
          f"{turtle_ret.index[0].date()} -> {turtle_ret.index[-1].date()}")
    print(f"  Mean = {turtle_ret.mean():.6f}, std = {turtle_ret.std():.6f}")
    print(f"  Annualised vol = {turtle_ret.std() * np.sqrt(252):.3%}")

    # Empirical correlation on overlap.
    common = gem_ret.index.intersection(turtle_ret.index)
    print(f"\n[correlate] common bars: {len(common)}")
    if len(common) > 30:
        rho = float(gem_ret.reindex(common).corr(turtle_ret.reindex(common)))
        print(f"  empirical daily-return correlation = {rho:+.4f}")

    # Joint assessment scenarios.
    # Scenario A: existing live GEM J5 alone (baseline).
    # Scenario B: GEM J5 + turtle CAT at 30% (proposed CONDITIONAL).
    # Scenario C: GEM J5 + turtle CAT at 50% (over-aggressive sensitivity).
    print("\n" + "-" * 72)
    print("Scenario A: GEM J5 alone at current deployment weight (assume 100%)")
    print("-" * 72)
    res_a = assess_strategy_ruin(
        gem_ret,
        deployment_weight=1.0,
        portfolio_kill_threshold=0.15,
        horizon_bars=252,
        block_size=21,
        n_paths=2000,
        seed=42,
    )
    print(res_a.report())

    print("\n" + "-" * 72)
    print("Scenario B: GEM J5 (70%) + turtle CAT C3_peak (30%) JOINT")
    print("-" * 72)
    res_b = assess_joint_ruin(
        {"gem_j5": gem_ret, "turtle_cat": turtle_ret},
        deployment_weights={"gem_j5": 0.70, "turtle_cat": 0.30},
        portfolio_kill_threshold=0.15,
        horizon_bars=252,
        block_size=21,
        n_paths=2000,
        seed=42,
    )
    print(res_b.report())

    print("\n" + "-" * 72)
    print("Scenario C: GEM J5 (50%) + turtle CAT (50%) -- sensitivity check")
    print("-" * 72)
    res_c = assess_joint_ruin(
        {"gem_j5": gem_ret, "turtle_cat": turtle_ret},
        deployment_weights={"gem_j5": 0.50, "turtle_cat": 0.50},
        portfolio_kill_threshold=0.15,
        horizon_bars=252,
        block_size=21,
        n_paths=2000,
        seed=42,
    )
    print(res_c.report())

    # Verdict summary.
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"Scenario A (GEM alone @ 100%): passes={res_a.passes()}")
    print(f"Scenario B (GEM 70% + turtle 30%): passes={res_b.passes()}")
    print(f"Scenario C (GEM 50% + turtle 50%): passes={res_c.passes()}")

    if res_b.passes():
        print("\nDEPLOY APPROVED: GEM J5 (70%) + turtle CAT C3_peak (30%) clears L65 joint gate.")
    else:
        print("\nDEPLOY BLOCKED: reduce turtle weight or delay live cutover.")


if __name__ == "__main__":
    main()
