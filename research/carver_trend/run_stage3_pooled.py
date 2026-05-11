"""Stage 3 — Pooled cross-pair WFO on all 6 FX pairs.

Carver's edge is *diversification across 35+ markets*. We can't replicate
that, but we can pool across the 6 FX pairs we have data for and see
whether the diversification reduces DDs enough to surface a positive
pooled Sharpe.

Pooled return per bar = sum(per-pair return) / N (equal risk weight).
Each pair contributes its own vol-targeted return, so the sum over all
pairs is the portfolio's daily return.

Common-window: aligned to 2016-03-17 (when GBP_USD/USD_CHF/USD_JPY data
begin). All 6 pairs evaluated on the same date range.

Run:  PYTHONUTF8=1 uv run python research/carver_trend/run_stage3_pooled.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "AUD_JPY", "USD_CHF", "USD_JPY"]


def load_daily(pair: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / f"{pair}_D.parquet")
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    return df["close"].sort_index().rename(pair)


def main() -> None:
    print("=" * 84)
    print("Stage 3 — Carver trend ladder POOLED across 6 FX pairs (daily)")
    print("=" * 84)
    print(f"\n  Ladder: {DEFAULT_LADDER}")
    print(f"  FDM: {DEFAULT_FDM} | Vol target 25% | 1 bp/turn cost")

    # Load each pair, compute its forecast + per-bar return series
    per_pair_returns: dict[str, pd.Series] = {}
    print(f"\n  {'Pair':<10}{'Bars':>10}{'Range':<35}{'IS Sh':>8}{'OOS Sh':>8}{'Sanc Sh':>10}")
    print("  " + "-" * 80)

    common_start = pd.Timestamp("2016-03-17", tz="UTC")
    oos_start = pd.Timestamp("2021-01-01", tz="UTC")
    sanctuary_start = pd.Timestamp("2025-01-01", tz="UTC")

    for pair in PAIRS:
        close = load_daily(pair)
        # Trim to common window so per-pair stats are comparable
        close = close[close.index >= common_start]
        forecast = ladder_forecast(close)
        bt = backtest_returns(forecast, close)
        rets = bt["net_returns"]
        per_pair_returns[pair] = rets

        is_mask = rets.index < oos_start
        oos_mask = (rets.index >= oos_start) & (rets.index < sanctuary_start)
        sanc_mask = rets.index >= sanctuary_start
        is_sh = sharpe(rets[is_mask], periods_per_year=DAILY_PER_YEAR)
        oos_sh = sharpe(rets[oos_mask], periods_per_year=DAILY_PER_YEAR)
        sanc_sh = sharpe(rets[sanc_mask], periods_per_year=DAILY_PER_YEAR)
        date_range = f"{close.index[0].date()} → {close.index[-1].date()}"
        print(
            f"  {pair:<10}{len(close):>10,d}{date_range:<35}"
            f"{is_sh:>+8.2f}{oos_sh:>+8.2f}{sanc_sh:>+10.2f}"
        )

    # Pool: equal-weight sum of per-pair daily returns (each pair carries
    # its own vol-targeted PnL, summing gives portfolio-level return).
    # Risk is reduced by diversification; expected return is unchanged.
    pooled = pd.concat(per_pair_returns.values(), axis=1).fillna(0.0)
    portfolio_rets = pooled.sum(axis=1) / len(PAIRS)

    print("\n" + "=" * 84)
    print(f"POOLED PORTFOLIO  (equal-weight average of {len(PAIRS)} pairs)")
    print("=" * 84)

    is_mask = portfolio_rets.index < oos_start
    oos_mask = (portfolio_rets.index >= oos_start) & (portfolio_rets.index < sanctuary_start)
    sanc_mask = portfolio_rets.index >= sanctuary_start

    def _stats(label: str, r: pd.Series) -> dict:
        sh = sharpe(r, periods_per_year=DAILY_PER_YEAR)
        lo, hi = bootstrap_sharpe_ci(r, periods_per_year=DAILY_PER_YEAR, n_resamples=2000)
        dd = max_drawdown(r)
        total = float((1.0 + r).prod() - 1.0)
        yrs = len(r) / DAILY_PER_YEAR
        cagr = (1.0 + total) ** (1.0 / yrs) - 1.0 if yrs > 0 and total > -1.0 else 0.0
        ann_vol = float(r.std() * np.sqrt(DAILY_PER_YEAR)) if r.std() > 0 else 0.0
        print(f"  [{label}]")
        print(f"    bars            : {len(r):>10,d}  (~{yrs:.2f}y)")
        print(f"    sharpe          : {sh:>+9.3f}")
        print(f"    sharpe 95% CI   : [{lo:>+.3f}, {hi:>+.3f}]")
        print(f"    realised vol    : {ann_vol:>9.2%}")
        print(f"    max drawdown    : {dd:>9.2%}")
        print(f"    total return    : {total:>9.2%}")
        print(f"    cagr            : {cagr:>9.2%}")
        return {"sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": dd}

    _stats("IS  (2016-2021)", portfolio_rets[is_mask])
    print()
    res_oos = _stats("OOS (2021-2025)", portfolio_rets[oos_mask])
    print()
    print("  [SANCTUARY 2025+ — held out, NOT a gate]")
    _stats("SANCTUARY", portfolio_rets[sanc_mask])

    print("\n" + "=" * 84)
    print("GATE EVALUATION (OOS only)")
    print("=" * 84)
    pass_sh = res_oos["sharpe"] > 0.5
    pass_ci = res_oos["ci_lo"] > 0
    if pass_sh and pass_ci:
        print("  PASS  → Stage 4 (add carry signal)")
    else:
        print("  FAIL  → tier=unconfirmed")
        if not pass_sh:
            print(f"          OOS Sharpe {res_oos['sharpe']:.3f} <= 0.5")
        if not pass_ci:
            print(f"          95% CI lo {res_oos['ci_lo']:.3f} <= 0")

    print()
    print("Honest note: Carver's framework needs 35+ markets across asset classes")
    print("(equities, rates, commodities, FX). We are testing only 6 FX pairs.")
    print("The diversification benefit is much smaller than Carver's portfolio gets.")
    print("=" * 84)


if __name__ == "__main__":
    main()
