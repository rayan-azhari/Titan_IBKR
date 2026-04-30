"""Stage 1 — Parabolic Short on full daily equity universe.

Runs the strategy across all _D.parquet equity files, pools per-day returns
across the universe (equal-weight: each day's return = mean of any trades
booked on that day, zero if no trades), and reports Sharpe / CI / DD on
IS / OOS, with a sanctuary window untouched.

Run with: PYTHONUTF8=1 uv run python research/parabolic_short/run_stage1.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.parabolic_short.strategy import (  # noqa: E402
    Trade,
    simulate_trades,
)
from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
    trade_sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
DAILY_PER_YEAR = 252


def list_equity_files() -> list[Path]:
    files = sorted(glob.glob(str(DATA_DIR / "*_D.parquet")))
    fx_prefixes = ("EUR_", "GBP_", "AUD_", "USD_", "NZD_", "CHF_", "JPY_")
    out: list[Path] = []
    for f in files:
        name = os.path.basename(f)
        if any(name.startswith(p) for p in fx_prefixes):
            continue
        # Skip ETF / index files heuristically — keep only single-ticker
        # equity files (already filtered by *_D.parquet suffix). We accept
        # ETFs in scope; the filter is on the setup itself.
        out.append(Path(f))
    return out


def load_one(path: Path) -> tuple[str, pd.DataFrame] | None:
    sym = path.name.replace("_D.parquet", "")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index(ts).drop(columns=["timestamp"]).sort_index()
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(set(df.columns)):
        return None
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < 50:
        return None
    return sym, df


def report(
    name: str, rets: pd.Series, trades: list,
) -> dict:
    sh = sharpe(rets, periods_per_year=DAILY_PER_YEAR)
    lo, hi = bootstrap_sharpe_ci(
        rets, periods_per_year=DAILY_PER_YEAR, n_resamples=2000
    )
    dd = max_drawdown(rets)
    total = float((1.0 + rets).prod() - 1.0)
    yrs = len(rets) / DAILY_PER_YEAR if DAILY_PER_YEAR else 0.0
    cagr = (1.0 + total) ** (1.0 / yrs) - 1.0 if yrs > 0 and total > -1.0 else 0.0
    n_trades = len(trades)
    n_winners = sum(1 for t in trades if t.r_multiple > 0)
    wr = (n_winners / n_trades) if n_trades > 0 else 0.0
    # Per-trade Sharpe annualised by trades-per-calendar-year
    trade_rs = np.array([t.r_multiple for t in trades]) if trades else np.array([])
    if yrs > 0 and len(trade_rs) >= 10:
        tps_per_year = len(trade_rs) / yrs
        t_sh = trade_sharpe(trade_rs, trades_per_year=tps_per_year)
    else:
        t_sh = 0.0
    print(f"  [{name}]")
    print(f"    calendar years : {yrs:>6.1f}y  ({len(rets):,} trading days)")
    print(f"    trades         : {n_trades:>6,d}   wins: {n_winners}  WR: {wr:.1%}")
    print(f"    sharpe (per-day, ann.)  : {sh:>+8.3f}")
    print(f"    sharpe 95% CI           : [{lo:>+.3f}, {hi:>+.3f}]")
    print(f"    sharpe (per-trade, ann.): {t_sh:>+8.3f}")
    print(f"    max drawdown            : {dd:>8.2%}")
    print(f"    total return            : {total:>8.2%}")
    print(f"    cagr                    : {cagr:>8.2%}")
    return {"sharpe": sh, "ci_lo": lo, "ci_hi": hi, "trade_sharpe": t_sh}


def main() -> None:
    print("=" * 78)
    print("Stage 1 — Parabolic Short — full equity universe")
    print("=" * 78)

    files = list_equity_files()
    print(f"\n  Universe candidates: {len(files)}")

    oos_start = pd.Timestamp("2018-01-01", tz="UTC")
    sanctuary_start = pd.Timestamp("2025-01-01", tz="UTC")
    print(f"  IS:        ... → {oos_start.date()}")
    print(f"  OOS:       {oos_start.date()} → {sanctuary_start.date()}")
    print(f"  Sanctuary: {sanctuary_start.date()} → present  (UNTOUCHED)\n")

    all_trades: list[Trade] = []
    n_processed = 0
    n_with_setups = 0
    for p in files:
        loaded = load_one(p)
        if loaded is None:
            continue
        sym, df = loaded
        n_processed += 1
        # Trim sanctuary
        df_in_scope = df[df.index < sanctuary_start]
        if len(df_in_scope) < 50:
            continue
        trades = simulate_trades(df_in_scope, symbol=sym)
        if trades:
            n_with_setups += 1
        all_trades.extend(trades)

    print(f"  Symbols processed       : {n_processed}")
    print(f"  Symbols with >=1 setup  : {n_with_setups}")
    print(f"  Total trades (full span): {len(all_trades)}")

    if not all_trades:
        print("  NO TRADES — strategy did not trigger.")
        return

    # Build per-symbol per-day return series, then pool equal-weighted across
    # symbols. Pooling: per-day equal-weight average of all trade-bearing
    # symbol returns. We approximate by booking each trade as a return on
    # exit_date scaled by 1/N_active_symbols_that_day. For Stage 1 simplicity,
    # we instead build a single union-day return series where each day's
    # return is the SUM of all trade returns booked that day (no rescaling).
    # This means total exposure is variable — that's a known limitation; if
    # the strategy has edge, both pooling methods will agree on Sharpe sign.

    # Build pooled daily return series keyed by *calendar date* (drop time-of-day).
    # Each calendar date contributes one mean-of-trade-returns observation.
    by_day: dict[pd.Timestamp, list[float]] = {}
    for t in all_trades:
        # Normalise to date (midnight UTC) so cross-symbol days align
        d = pd.Timestamp(t.exit_date).normalize()
        by_day.setdefault(d, []).append(t.r_multiple * 0.01)  # 1% risk

    # Master calendar = sorted union of trading dates across the universe (date-normalised)
    cal: set[pd.Timestamp] = set()
    for p in files:
        loaded = load_one(p)
        if loaded is None:
            continue
        _, df = loaded
        df = df[df.index < sanctuary_start]
        cal.update(pd.Timestamp(x).normalize() for x in df.index)
    master_idx = pd.DatetimeIndex(sorted(cal))

    pooled = pd.Series(0.0, index=master_idx)
    for d, vs in by_day.items():
        if d in pooled.index:
            pooled.loc[d] = float(np.mean(vs))

    # Splits
    is_mask = pooled.index < oos_start
    oos_mask = (pooled.index >= oos_start) & (pooled.index < sanctuary_start)

    is_trades = [t for t in all_trades if t.entry_date < oos_start]
    oos_trades = [t for t in all_trades
                  if oos_start <= t.entry_date < sanctuary_start]

    print("\n  Trade R-multiple summary (full):")
    rs = np.array([t.r_multiple for t in all_trades])
    print(f"    mean R   : {rs.mean():+.3f}")
    print(f"    median R : {np.median(rs):+.3f}")
    print(f"    win rate : {(rs > 0).mean():.1%}")
    print(f"    p95 R    : {np.quantile(rs, 0.95):+.3f}")
    print(f"    p05 R    : {np.quantile(rs, 0.05):+.3f}")
    er = {"tp": 0, "sl": 0, "time": 0}
    for t in all_trades:
        er[t.exit_reason] = er.get(t.exit_reason, 0) + 1
    print(f"    exit mix : tp={er['tp']}  sl={er['sl']}  time={er['time']}")

    print()
    report("IS", pooled[is_mask], is_trades)
    print()
    res_oos = report("OOS", pooled[oos_mask], oos_trades)

    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    pass_sh = res_oos["sharpe"] > 0.5
    pass_ci = res_oos["ci_lo"] > 0
    if pass_sh and pass_ci:
        print("  PASS  → consider Stage 2 (Pro-Trend Breakout) and portfolio fit")
    else:
        print("  FAIL  → tier=unconfirmed, do not advance")
        if not pass_sh:
            print(f"          OOS Sharpe {res_oos['sharpe']:.3f} <= 0.5")
        if not pass_ci:
            print(f"          95% lower bound {res_oos['ci_lo']:.3f} <= 0")
    print("=" * 78)


if __name__ == "__main__":
    main()
