"""Risk-of-ruin analysis for the Samir-Stack champion (MES L=3, 10/90).

Uses the 16-fold WFO stitched OOS returns as the empirical distribution
and runs:

  1. Empirical stats: actual MaxDD, worst 1y/3y/5y, recovery time.
  2. Bootstrap (block + iid) of the daily-return distribution to project
     P(MaxDD > X) over horizons 1/5/10/20 years.
  3. VaR / CVaR at 95% and 99%.
  4. Path-dependent ruin probability: P(equity drops to F × starting
     value before T years) for F in {0.50, 0.75, 0.85, 0.90}.

Bootstrap uses 5-day blocks to preserve short-horizon serial dependence
(volatility clustering) without imposing a parametric vol model.

Usage:
    uv run python research/samir_stack/run_risk_of_ruin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_futures_sweep import (  # noqa: E402
    _engine_futures,
    _wfo_stitch,
    run_with_engine,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _max_drawdown(rets: np.ndarray) -> float:
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def _block_bootstrap_paths(
    rets: np.ndarray,
    *,
    horizon_days: int,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary block bootstrap. Returns array of shape (n_paths, horizon_days)."""
    n = len(rets)
    n_blocks = int(np.ceil(horizon_days / block_size))
    paths = np.empty((n_paths, horizon_days), dtype=float)
    for i in range(n_paths):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        chunks = [rets[s : s + block_size] for s in starts]
        path = np.concatenate(chunks)[:horizon_days]
        paths[i] = path
    return paths


def _ruin_probability(
    rets: np.ndarray,
    *,
    horizon_years: float,
    drawdown_threshold: float,
    n_paths: int = 5000,
    block_size: int = 5,
    seed: int = 42,
) -> dict:
    """P(max-DD over horizon exceeds threshold). Threshold positive, e.g. 0.25."""
    rng = np.random.default_rng(seed)
    horizon_days = int(round(horizon_years * 252))
    paths = _block_bootstrap_paths(
        rets, horizon_days=horizon_days, n_paths=n_paths, block_size=block_size, rng=rng
    )
    dds = np.array([_max_drawdown(p) for p in paths])
    breach = dds <= -drawdown_threshold
    return {
        "horizon_years": horizon_years,
        "dd_threshold": drawdown_threshold,
        "n_paths": n_paths,
        "p_breach": float(breach.mean()),
        "median_max_dd": float(np.median(dds)),
        "p95_max_dd": float(np.percentile(dds, 5)),  # 5th pct = worst 5%
        "p99_max_dd": float(np.percentile(dds, 1)),
        "worst_max_dd": float(dds.min()),
    }


def _ending_wealth_distribution(
    rets: np.ndarray,
    *,
    horizon_years: float,
    n_paths: int = 5000,
    block_size: int = 5,
    seed: int = 43,
) -> dict:
    rng = np.random.default_rng(seed)
    horizon_days = int(round(horizon_years * 252))
    paths = _block_bootstrap_paths(
        rets, horizon_days=horizon_days, n_paths=n_paths, block_size=block_size, rng=rng
    )
    eq = np.cumprod(1.0 + paths, axis=1)
    end = eq[:, -1]
    return {
        "horizon_years": horizon_years,
        "p1_ending_wealth": float(np.percentile(end, 1)),
        "p5_ending_wealth": float(np.percentile(end, 5)),
        "median_ending_wealth": float(np.median(end)),
        "p95_ending_wealth": float(np.percentile(end, 95)),
        "p_loss": float((end < 1.0).mean()),
        "p_loss_25pct": float((end < 0.75).mean()),
        "p_loss_50pct": float((end < 0.50).mean()),
    }


def main() -> int:
    print("Loading data and running L=3 10/90 strategy...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    efa = _load_close("EFA_D.parquet")
    common = (
        data["spy"]
        .index.intersection(efa.index)
        .intersection(data["tlt"].index)
        .intersection(data["hyg"].index)
        .intersection(data["ief"].index)
    )
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)
    efa_a = efa.reindex(common)
    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    df = run_with_engine(
        spy,
        efa_a,
        ief,
        hyg,
        tlt,
        samir_score,
        equity_engine=_engine_futures,
        L_max=3.0,
        equity_weight=0.10,
        bond_weight=0.90,
    )
    stitched, _ = _wfo_stitch(df)
    print(f"  OOS days: {len(stitched)} ({len(stitched) / 252:.1f} years)\n", flush=True)

    # ── 1. Empirical stats ──────────────────────────────────────────────
    eq = np.cumprod(1.0 + stitched)
    dd_path = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    n_years = len(stitched) / 252.0

    # Worst rolling N-day return
    def rolling_worst(rets: np.ndarray, window: int) -> float:
        if len(rets) < window:
            return 0.0
        cum = pd.Series(rets).rolling(window).apply(lambda x: float((1.0 + x).prod() - 1.0))
        return float(cum.min())

    print("=" * 90)
    print("EMPIRICAL STATS (16 OOS years, MES L=3 10/90)")
    print("=" * 90)
    print(f"  Annualised return:    {((eq[-1]) ** (1.0 / n_years) - 1.0) * 100:>7.2f}%")
    print(f"  Annualised vol:       {np.std(stitched) * np.sqrt(252) * 100:>7.2f}%")
    print(f"  Sharpe:               {(np.mean(stitched) / np.std(stitched) * np.sqrt(252)):>7.3f}")
    print(f"  Max drawdown:         {dd_path.min() * 100:>7.2f}%")
    print(f"  Worst single day:     {stitched.min() * 100:>7.2f}%")
    print(f"  Worst 5-day:          {rolling_worst(stitched, 5) * 100:>7.2f}%")
    print(f"  Worst 21-day (1mo):   {rolling_worst(stitched, 21) * 100:>7.2f}%")
    print(f"  Worst 63-day (3mo):   {rolling_worst(stitched, 63) * 100:>7.2f}%")
    print(f"  Worst 252-day (1yr):  {rolling_worst(stitched, 252) * 100:>7.2f}%")
    print(f"  P(daily ret < 0):     {(stitched < 0).mean() * 100:>7.2f}%")
    print()

    # VaR / CVaR (loss perspective; positive numbers)
    losses = -stitched
    var95 = np.percentile(losses, 95)
    var99 = np.percentile(losses, 99)
    cvar95 = losses[losses >= var95].mean()
    cvar99 = losses[losses >= var99].mean()
    print(f"  Daily VaR 95%:        {var95 * 100:>7.2f}%   (1-in-20 day expected loss)")
    print(f"  Daily VaR 99%:        {var99 * 100:>7.2f}%   (1-in-100 day expected loss)")
    print(f"  Daily CVaR 95%:       {cvar95 * 100:>7.2f}%   (avg loss in worst 5%)")
    print(f"  Daily CVaR 99%:       {cvar99 * 100:>7.2f}%   (avg loss in worst 1%)")

    # ── 2. Block-bootstrap drawdown projections ─────────────────────────
    print()
    print("=" * 90)
    print("BOOTSTRAP DRAWDOWN PROJECTIONS (5-day block bootstrap, 5000 paths)")
    print("=" * 90)
    horizons = [1.0, 5.0, 10.0, 20.0]
    thresholds = [0.05, 0.10, 0.15, 0.25, 0.50]

    dd_table_rows = []
    for h in horizons:
        row = {"horizon_yrs": h}
        for thr in thresholds:
            res = _ruin_probability(stitched, horizon_years=h, drawdown_threshold=thr)
            row[f"P(MaxDD>{int(thr * 100)}%)"] = round(res["p_breach"], 4)
        dd_table_rows.append(row)
    dd_table = pd.DataFrame(dd_table_rows).set_index("horizon_yrs")
    print(dd_table.to_string())

    # Tail percentiles of MaxDD over each horizon
    print()
    print("Max-drawdown distribution by horizon:")
    pct_rows = []
    for h in horizons:
        res = _ruin_probability(stitched, horizon_years=h, drawdown_threshold=0.05)
        pct_rows.append(
            {
                "horizon_yrs": h,
                "median_DD": round(res["median_max_dd"] * 100, 2),
                "5th_pct_DD": round(res["p95_max_dd"] * 100, 2),
                "1st_pct_DD": round(res["p99_max_dd"] * 100, 2),
                "worst_DD": round(res["worst_max_dd"] * 100, 2),
            }
        )
    print(pd.DataFrame(pct_rows).set_index("horizon_yrs").to_string())

    # ── 3. Ending-wealth distribution ───────────────────────────────────
    print()
    print("=" * 90)
    print("ENDING-WEALTH DISTRIBUTION (multiplier on starting capital)")
    print("=" * 90)
    ew_rows = []
    for h in [5.0, 10.0, 20.0]:
        res = _ending_wealth_distribution(stitched, horizon_years=h)
        ew_rows.append(
            {
                "horizon_yrs": h,
                "1st_pct": round(res["p1_ending_wealth"], 3),
                "5th_pct": round(res["p5_ending_wealth"], 3),
                "median": round(res["median_ending_wealth"], 3),
                "95th_pct": round(res["p95_ending_wealth"], 3),
                "P(end < start)": round(res["p_loss"], 4),
                "P(end < -25%)": round(res["p_loss_25pct"], 4),
                "P(end < -50%)": round(res["p_loss_50pct"], 4),
            }
        )
    print(pd.DataFrame(ew_rows).set_index("horizon_yrs").to_string())

    # Save
    pd.DataFrame(dd_table_rows).to_csv(REPORTS_DIR / "risk_of_ruin_drawdown.csv", index=False)
    pd.DataFrame(ew_rows).to_csv(REPORTS_DIR / "risk_of_ruin_wealth.csv", index=False)
    print(f"\nSaved: {REPORTS_DIR / 'risk_of_ruin_drawdown.csv'}")
    print(f"Saved: {REPORTS_DIR / 'risk_of_ruin_wealth.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
