"""Stage 2 — single-pair WFO on EUR/USD daily.

Honest IS/OOS with sanctuary held out. Using Carver's published EWMAC
ladder + scaling factors verbatim, no in-sample tuning.

Run:  PYTHONUTF8=1 uv run python research/carver_trend/run_stage2_eurusd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.carver_trend.backtest import backtest_returns  # noqa: E402
from research.carver_trend.forecast import (  # noqa: E402
    DEFAULT_FDM,
    DEFAULT_LADDER,
    ladder_forecast,
)
from titan.research.metrics import bootstrap_sharpe_ci, max_drawdown, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
DAILY_PER_YEAR = 252


def load_daily(pair: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / f"{pair}_D.parquet")
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    return df["close"].sort_index().rename(pair)


def report_block(name: str, rets: pd.Series, position: pd.Series) -> dict:
    sh = sharpe(rets, periods_per_year=DAILY_PER_YEAR)
    lo, hi = bootstrap_sharpe_ci(rets, periods_per_year=DAILY_PER_YEAR, n_resamples=2000)
    dd = max_drawdown(rets)
    total = float((1.0 + rets).prod() - 1.0)
    yrs = len(rets) / DAILY_PER_YEAR
    cagr = (1.0 + total) ** (1.0 / yrs) - 1.0 if yrs > 0 and total > -1.0 else 0.0
    avg_pos = float(position.abs().mean())
    print(f"  [{name}]")
    print(f"    bars              : {len(rets):>10,d}  (~{yrs:.2f}y)")
    print(f"    sharpe (annualised): {sh:>+9.3f}")
    print(f"    sharpe 95% CI      : [{lo:>+.3f}, {hi:>+.3f}]")
    print(f"    max drawdown       : {dd:>9.2%}")
    print(f"    total return       : {total:>9.2%}")
    print(f"    cagr               : {cagr:>9.2%}")
    print(f"    avg |position|     : {avg_pos:>9.3f}")
    return {"sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": dd}


def main() -> None:
    print("=" * 78)
    print("Stage 2 — Carver trend ladder — EUR/USD daily")
    print("=" * 78)
    print(f"\n  Ladder: {DEFAULT_LADDER}")
    print(f"  FDM:    {DEFAULT_FDM}")
    print("  Vol target: 25% annualised | Cost: 1 bp per turn")

    close = load_daily("EUR_USD")
    print(f"\n  Bars: {len(close):,}  ({close.index[0].date()} → {close.index[-1].date()})")

    sanctuary_start = pd.Timestamp("2025-01-01", tz="UTC")
    oos_start = pd.Timestamp("2018-01-01", tz="UTC")
    print(f"  IS:        {close.index[0].date()} → {oos_start.date()}")
    print(f"  OOS:       {oos_start.date()} → {sanctuary_start.date()}")
    print(f"  Sanctuary: {sanctuary_start.date()} → present  (UNTOUCHED)")

    # Compute forecast on full history (causal — uses only past prices),
    # then slice into IS / OOS / sanctuary for reporting.
    forecast = ladder_forecast(close)
    bt = backtest_returns(forecast, close)
    rets = bt["net_returns"]
    pos = bt["position"]

    is_mask = rets.index < oos_start
    oos_mask = (rets.index >= oos_start) & (rets.index < sanctuary_start)
    sanc_mask = rets.index >= sanctuary_start

    print()
    report_block("IS  (..2018)", rets[is_mask], pos[is_mask])
    print()
    res_oos = report_block("OOS (2018-2024)", rets[oos_mask], pos[oos_mask])
    print()
    print("  [SANCTUARY (2025+) — held out, do NOT use for go/no-go]")
    report_block("SANCTUARY", rets[sanc_mask], pos[sanc_mask])

    print("\n" + "=" * 78)
    print("GATE EVALUATION (OOS only)")
    print("=" * 78)
    pass_sh = res_oos["sharpe"] > 0.5
    pass_ci = res_oos["ci_lo"] > 0
    if pass_sh and pass_ci:
        print("  PASS  → Stage 3 (cross-pair WFO on all 6 pairs)")
    else:
        print("  FAIL  → tier=unconfirmed")
        if not pass_sh:
            print(f"          OOS Sharpe {res_oos['sharpe']:.3f} <= 0.5")
        if not pass_ci:
            print(f"          95% CI lo {res_oos['ci_lo']:.3f} <= 0")
    print("=" * 78)


if __name__ == "__main__":
    main()
