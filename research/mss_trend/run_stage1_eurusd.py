"""Stage 1 — honest single-pair OOS test on EUR/USD 15M.

Builds 15M bars by resampling EUR_USD_M5.parquet, runs the full strategy
on the dataset using frozen parameters, splits into IS/OOS/sanctuary,
and reports Sharpe + bootstrap 95% CI on each.

Sanctuary (last 12 months) is reserved — not reported in this stage.
This run tests only IS (build period) and OOS (out-of-sample test).

Run with: uv run python research/mss_trend/run_stage1_eurusd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
)

from research.mss_trend.strategy import (  # noqa: E402
    generate_trades,
    trades_to_bar_returns,
)

DATA_DIR = PROJECT_ROOT / "data"
M15_PER_YEAR = 24 * 4 * 252  # 24,192


def load_m15_eur_usd() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "EUR_USD_M5.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Resample M5 -> 15M (right-closed/right-labeled like FX bar convention).
    # Use label='left' so the bar timestamp is the start of the 15M window;
    # this matches the convention everywhere else in the codebase.
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    m15 = df.resample("15min", label="left", closed="left").agg(agg).dropna()
    return m15


def load_daily_eur_usd() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "EUR_USD_D.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def report_block(name: str, rets: pd.Series, n_trades: int) -> dict:
    sh = sharpe(rets, periods_per_year=M15_PER_YEAR)
    lo, hi = bootstrap_sharpe_ci(rets, periods_per_year=M15_PER_YEAR, n_resamples=2000)
    dd = max_drawdown(rets)
    total_ret = float((1.0 + rets).prod() - 1.0)
    n_bars = len(rets)
    bars_per_year = M15_PER_YEAR
    years = n_bars / bars_per_year if bars_per_year > 0 else 0.0
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 and total_ret > -1.0 else 0.0
    print(f"  [{name}]")
    print(f"    bars               : {n_bars:>10,d}  (~{years:.2f}y)")
    print(f"    trades             : {n_trades:>10,d}")
    print(f"    sharpe (annualised): {sh:>10.3f}")
    print(f"    sharpe 95% CI      : [{lo:>+.3f}, {hi:>+.3f}]")
    print(f"    max drawdown       : {dd:>10.2%}")
    print(f"    total return       : {total_ret:>10.2%}")
    print(f"    cagr               : {cagr:>10.2%}")
    return {
        "name": name,
        "sharpe": sh,
        "ci_lo": lo,
        "ci_hi": hi,
        "max_dd": dd,
        "total_return": total_ret,
        "cagr": cagr,
        "n_trades": n_trades,
        "n_bars": n_bars,
    }


def main() -> None:
    print("=" * 72)
    print("Stage 1 — MSS Trend Strategy — EUR/USD 15M honest OOS test")
    print("=" * 72)

    print("\n[1/4] Loading data...")
    m15 = load_m15_eur_usd()
    daily = load_daily_eur_usd()
    print(f"      15M bars : {len(m15):,}  ({m15.index[0]} → {m15.index[-1]})")
    print(f"      Daily    : {len(daily):,}  ({daily.index[0]} → {daily.index[-1]})")

    # IS / OOS / Sanctuary split
    sanctuary_start = m15.index[-1] - pd.DateOffset(years=1)
    oos_start = pd.Timestamp("2021-01-01", tz="UTC")
    print(f"\n[2/4] Split:")
    print(f"      IS         : {m15.index[0]} → {oos_start}")
    print(f"      OOS        : {oos_start} → {sanctuary_start}")
    print(f"      Sanctuary  : {sanctuary_start} → {m15.index[-1]}  (UNTOUCHED)")

    print("\n[3/4] Generating trades on full dataset (one pass, no parameter "
          "tuning)...")
    trades = generate_trades(
        m15_df=m15,
        daily_df=daily,
        daily_n=2,
        m15_n=6,
        tp_r_multiple=2.0,
    )
    print(f"      total trades: {len(trades)}")
    if not trades:
        print("      NO TRADES — strategy did not trigger. Stopping.")
        return

    win_rate = sum(1 for t in trades if t.r_multiple > 0) / len(trades)
    avg_r = np.mean([t.r_multiple for t in trades])
    print(f"      win rate    : {win_rate:.1%}")
    print(f"      avg R       : {avg_r:+.3f}")

    bar_returns = trades_to_bar_returns(
        trades, bar_index=m15.index, risk_per_trade=0.01, cost_per_trade=5e-5
    )

    is_mask = m15.index < oos_start
    oos_mask = (m15.index >= oos_start) & (m15.index < sanctuary_start)
    is_trades = [t for t in trades if t.entry_ts < oos_start]
    oos_trades = [t for t in trades
                  if oos_start <= t.entry_ts < sanctuary_start]

    print("\n[4/4] Results:\n")
    report_block("IS  (2005 → 2021)", bar_returns[is_mask], len(is_trades))
    print()
    res_oos = report_block("OOS (2021 → 2025)", bar_returns[oos_mask],
                            len(oos_trades))

    print("\n" + "=" * 72)
    print("GATE EVALUATION")
    print("=" * 72)
    pass_sharpe = res_oos["sharpe"] > 0.5
    pass_ci = res_oos["ci_lo"] > 0
    if pass_sharpe and pass_ci:
        print("  PASS  → proceed to Stage 2 (cross-pair WFO at H1)")
    else:
        print("  FAIL  → tier=unconfirmed, do not advance")
        if not pass_sharpe:
            print(f"          OOS Sharpe {res_oos['sharpe']:.3f} <= 0.5 gate")
        if not pass_ci:
            print(f"          95% lower bound {res_oos['ci_lo']:.3f} <= 0")
    print("=" * 72)


if __name__ == "__main__":
    main()
