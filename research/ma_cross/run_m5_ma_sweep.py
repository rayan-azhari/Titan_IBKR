"""run_m5_ma_sweep.py -- M5 Moving-Average Cross Parameter Sweep.

Tests a 3-level MA structure on M5 bars:
  - Trend filter (slow SMA): price must be above/below this to trade with trend
  - Medium EMA (slow): intermediate trend direction
  - Fast EMA (fast): timing trigger — cross above/below medium EMA = entry/exit

Entry logic (trend-following, long side):
    EMA_fast crosses above EMA_slow AND close > SMA_trend  -> Long
    EMA_fast crosses below EMA_slow OR  close < SMA_trend  -> Exit

With direction="both":
    EMA_fast crosses below EMA_slow AND close < SMA_trend  -> Short
    EMA_fast crosses above EMA_slow OR  close > SMA_trend  -> Cover

IS/OOS split: 70% IS (parameter selection) / 30% OOS (validation).

Gates (same as Phase 3):
    OOS Sharpe > 0
    OOS/IS Sharpe parity >= 0.50
    Trades/year >= 12  (enough data to estimate statistics)

Usage
-----
    # Sweep all available M5 instruments:
    uv run python research/ma_cross/run_m5_ma_sweep.py

    # Single instrument, both directions, verbose:
    uv run python research/ma_cross/run_m5_ma_sweep.py \\
        --instruments EUR_USD CAT --direction both

    # Quick check (fewer combinations):
    uv run python research/ma_cross/run_m5_ma_sweep.py --fast

Output
------
    Console: ranked table per instrument
    CSV:     .tmp/reports/ma_cross_m5_{timestamp}.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.phase3_backtest import (  # noqa: E402
    IS_RATIO,
    _load_ohlcv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 288 M5 bars = 288 x 5 min = 1,440 min = 24 h (one FX day)
# For US equities (6.5h session): 78 bars/day -> 288 ~= 3.7 days
DEFAULT_FAST_PERIODS: list[int] = [10, 15, 20, 25, 30]
DEFAULT_SLOW_PERIODS: list[int] = [40, 50, 60, 80, 100]
DEFAULT_TREND_PERIODS: list[int] = [144, 200, 288, 360]

# M5 bars per year: 252 trading days x 78 bars/day (US equities, 9:30-16:00 ET)
# FX: 252 x 288 (24h), but we normalise using US equity sessions for mixed instruments
BARS_PER_YEAR_M5: int = 252 * 78  # ~19,656 bars/year

INIT_CASH: float = 10_000.0
MIN_BARS: int = 1_000  # minimum bars needed to run a backtest
MIN_TRADES_PER_YEAR: float = 12.0

# Phase 3 gates
MIN_OOS_SHARPE: float = 0.0
MIN_PARITY: float = 0.50

# Cost profile for intraday equity on M5 (tighter than daily — fills at bid/ask)
M5_EQUITY_PROFILE: dict = {
    "spread_bps": 3.0,  # slightly wider intraday
    "slippage_bps": 2.0,
    "max_leverage": 2.0,  # conservative intraday leverage
}
M5_FX_PROFILE: dict = {
    "spread_bps": 1.0,  # FX spread is tight
    "slippage_bps": 0.5,
    "max_leverage": 5.0,
}

# Instruments with M5 data — auto-detected at runtime; this is the preferred order
M5_INSTRUMENTS: dict[str, str] = {
    # FX
    "EUR_USD": "fx",
    "EUR_CHF": "fx",
    "USD_CHF": "fx",
    # Equities
    "CAT": "equity",
    "CSCO": "equity",
    "UNH": "equity",
    "TMO": "equity",
    "INTC": "equity",
    "AMAT": "equity",
    "WMT": "equity",
    "TXN": "equity",
    "CRM": "equity",
}


# ---------------------------------------------------------------------------
# Signal helpers  (all vectorised, all shift(1) to prevent lookahead)
# ---------------------------------------------------------------------------


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()


def _compute_signals(
    close: pd.Series,
    fast: int,
    slow: int,
    trend: int,
    direction: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute entry/exit signals for one MA parameter set.

    Returns: long_entries, long_exits, short_entries, short_exits
    All signals are shifted 1 bar (fills on the bar AFTER the signal fires).
    """
    ema_fast = _ema(close, fast).shift(1)
    ema_slow = _ema(close, slow).shift(1)
    sma_trend = _sma(close, trend).shift(1)

    # Cross detection: current bar crosses vs previous bar
    fast_above_slow = ema_fast > ema_slow
    fast_above_slow_prev = fast_above_slow.shift(1).fillna(False)
    cross_up = fast_above_slow & ~fast_above_slow_prev
    cross_dn = ~fast_above_slow & fast_above_slow_prev

    above_trend = close.shift(1) > sma_trend
    below_trend = close.shift(1) < sma_trend

    long_entries = cross_up & above_trend
    long_exits = cross_dn | ~above_trend

    if direction == "both":
        short_entries = cross_dn & below_trend
        short_exits = cross_up | ~below_trend
    else:
        short_entries = pd.Series(False, index=close.index)
        short_exits = pd.Series(False, index=close.index)

    return long_entries, long_exits, short_entries, short_exits


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (shift(1) applied by caller)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# Core backtest (single parameter set)
# ---------------------------------------------------------------------------


def _backtest_params(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    trend: int,
    direction: str,
    asset_type: str,
) -> dict | None:
    """Run IS + OOS backtest for one (fast, slow, trend, direction) combination.

    Returns None if minimum data or trade-count gates are not met.
    """
    if len(df) < MIN_BARS:
        return None

    close = df["close"]
    n = len(df)
    is_n = int(n * IS_RATIO)

    # Cost profile
    profile = M5_FX_PROFILE if asset_type == "fx" else M5_EQUITY_PROFILE
    med_close = float(close.median()) or 1.0
    spread_price = (profile["spread_bps"] / 10_000) * med_close
    slip_price = (profile["slippage_bps"] / 10_000) * med_close

    # Dynamic fees/slippage as fraction of close (VBT convention)
    fees_frac = (spread_price / close).bfill().values
    slip_frac = (slip_price / close).bfill().values

    # ATR-based position sizing (1% risk per trade)
    atr_s = _atr(df).shift(1)
    safe_close = close.where(close > 0, np.nan)
    stop_pct = (2.0 * atr_s / safe_close).fillna(0.0)
    risk_pct = 0.01
    size_pct = (
        (risk_pct / stop_pct.replace(0.0, np.nan)).clip(upper=profile["max_leverage"]).fillna(0.0)
    )

    long_en, long_ex, short_en, short_ex = _compute_signals(close, fast, slow, trend, direction)

    freq = "5min"

    def _run_side(entries: pd.Series, exits: pd.Series, idx: pd.Index, side: str) -> vbt.Portfolio:
        _dir = "longonly" if side == "long" else "shortonly"
        return vbt.Portfolio.from_signals(
            close.loc[idx],
            entries=entries.loc[idx],
            exits=exits.loc[idx],
            size=size_pct.loc[idx].values,
            size_type="percent",
            init_cash=INIT_CASH / (2 if direction == "both" else 1),
            fees=fees_frac[df.index.get_indexer(idx)],
            slippage=slip_frac[df.index.get_indexer(idx)],
            freq=freq,
            direction=_dir,
        )

    is_idx = df.index[:is_n]
    oos_idx = df.index[is_n:]

    def _slice_stats(idx: pd.Index) -> dict:
        pf_long = _run_side(long_en, long_ex, idx, "long")
        rets = pf_long.returns()
        if direction == "both":
            pf_short = _run_side(short_en, short_ex, idx, "short")
            rets = (pf_long.returns() + pf_short.returns()) / 2.0
        std = float(rets.std())
        n_bars = len(idx)
        bars_per_year = BARS_PER_YEAR_M5
        sharpe = float(rets.mean() / std * np.sqrt(bars_per_year)) if std > 1e-10 else 0.0
        equity = (1.0 + rets).cumprod()
        peak = equity.cummax()
        max_dd = float(((equity - peak) / peak).min())
        total_ret = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
        n_years = n_bars / bars_per_year
        cagr = float((1 + total_ret) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0
        n_trades = int(pf_long.trades.count())
        if direction == "both":
            n_trades += int(pf_short.trades.count())
        trades_per_year = n_trades / n_years if n_years > 0 else 0.0
        return {
            "sharpe": sharpe,
            "max_dd": max_dd,
            "cagr": cagr,
            "n_trades": n_trades,
            "trades_per_year": trades_per_year,
            "daily_returns": rets,
        }

    is_s = _slice_stats(is_idx)
    oos_s = _slice_stats(oos_idx)

    is_sh = is_s["sharpe"]
    oos_sh = oos_s["sharpe"]
    parity = (oos_sh / is_sh) if is_sh != 0.0 else float("nan")

    # Apply gates
    pass_gate = (
        oos_sh > MIN_OOS_SHARPE
        and (not np.isnan(parity))
        and parity >= MIN_PARITY
        and oos_s["trades_per_year"] >= MIN_TRADES_PER_YEAR
    )

    return {
        "fast": fast,
        "slow": slow,
        "trend": trend,
        "direction": direction,
        "is_sharpe": round(is_sh, 3),
        "oos_sharpe": round(oos_sh, 3),
        "parity": round(parity, 3) if not np.isnan(parity) else float("nan"),
        "oos_max_dd": round(oos_s["max_dd"], 4),
        "oos_cagr": round(oos_s["cagr"], 4),
        "oos_trades_yr": round(oos_s["trades_per_year"], 1),
        "pass": pass_gate,
    }


# ---------------------------------------------------------------------------
# Per-instrument sweep
# ---------------------------------------------------------------------------


def sweep_instrument(
    instrument: str,
    asset_type: str,
    fast_periods: list[int],
    slow_periods: list[int],
    trend_periods: list[int],
    direction: str,
) -> pd.DataFrame:
    """Run full parameter sweep for one instrument. Returns DataFrame of results."""
    try:
        df = _load_ohlcv(instrument, "M5")
    except FileNotFoundError as exc:
        print(f"  [SKIP] {instrument}: {exc}")
        return pd.DataFrame()

    print(
        f"  {instrument} ({asset_type}): {len(df)} M5 bars  "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )

    rows: list[dict] = []
    combos = [
        (f, s, t)
        for f, s, t in product(fast_periods, slow_periods, trend_periods)
        if f < s < t  # logical ordering constraint
    ]
    print(f"    Testing {len(combos)} param combos x direction={direction} ...")

    for fast, slow, trend in combos:
        result = _backtest_params(df, fast, slow, trend, direction, asset_type)
        if result is not None:
            result["instrument"] = instrument
            rows.append(result)

    if not rows:
        print(f"    No valid results for {instrument}.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    passing = result_df[result_df["pass"]]
    print(f"    {len(passing)}/{len(result_df)} combinations pass all gates.")

    if not passing.empty:
        best = passing.iloc[0]
        print(
            f"    BEST: EMA({int(best.fast)}/{int(best.slow)}) + SMA({int(best.trend)}) "
            f"| IS={best.is_sharpe:+.2f}  OOS={best.oos_sharpe:+.2f}  "
            f"parity={best.parity:.2f}  DD={best.oos_max_dd:.1%}  "
            f"trades/yr={best.oos_trades_yr:.0f}"
        )
    else:
        best_any = result_df.iloc[0]
        print(f"    No pass (best OOS={best_any.oos_sharpe:+.2f}, parity={best_any.parity:.2f})")

    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 MA cross parameter sweep.")
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Instruments to sweep (default: all with M5 data).",
    )
    parser.add_argument(
        "--direction",
        choices=["long_only", "both"],
        default="long_only",
        help="Trade direction (default: long_only).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick mode: reduced parameter grid.",
    )
    args = parser.parse_args()

    if args.fast:
        fast_periods = [15, 20, 25]
        slow_periods = [50, 80]
        trend_periods = [200, 288]
    else:
        fast_periods = DEFAULT_FAST_PERIODS
        slow_periods = DEFAULT_SLOW_PERIODS
        trend_periods = DEFAULT_TREND_PERIODS

    # Determine instruments to sweep
    data_dir = ROOT / "data"
    if args.instruments:
        instruments = {k: M5_INSTRUMENTS.get(k, "equity") for k in args.instruments}
    else:
        instruments = {
            k: v for k, v in M5_INSTRUMENTS.items() if (data_dir / f"{k}_M5.parquet").exists()
        }

    W = 72
    print()
    print("=" * W)
    print("  M5 MA CROSS SWEEP")
    print(f"  Instruments : {list(instruments.keys())}")
    print(f"  Fast EMA    : {fast_periods}")
    print(f"  Slow EMA    : {slow_periods}")
    print(f"  Trend SMA   : {trend_periods}  (288 = 24h FX / ~3.7d equity)")
    print(f"  Direction   : {args.direction}")
    print(f"  Gates       : OOS Sharpe>0, parity>={MIN_PARITY}, trades/yr>={MIN_TRADES_PER_YEAR}")
    print("=" * W)

    all_results: list[pd.DataFrame] = []

    for instrument, asset_type in instruments.items():
        print(f"\n  --- {instrument} ---")
        df = sweep_instrument(
            instrument,
            asset_type,
            fast_periods,
            slow_periods,
            trend_periods,
            direction=args.direction,
        )
        if not df.empty:
            all_results.append(df)

    if not all_results:
        print("\n  No results produced. Check that M5 parquet files exist in data/.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    passing = combined[combined["pass"]].sort_values("oos_sharpe", ascending=False)

    print()
    print("=" * W)
    print(f"  SUMMARY: {len(passing)} passing combinations across {len(instruments)} instruments")
    print("=" * W)

    if not passing.empty:
        print(
            f"\n  {'Instrument':<12} {'Dir':<10} {'EMA fast':>8} {'EMA slow':>8} "
            f"{'SMA trend':>9} {'IS Sh':>7} {'OOS Sh':>7} {'parity':>7} "
            f"{'MaxDD':>7} {'Trd/yr':>7}"
        )
        print("  " + "-" * 80)
        for _, r in passing.head(20).iterrows():
            print(
                f"  {r['instrument']:<12} {r['direction']:<10} "
                f"{int(r['fast']):>8} {int(r['slow']):>8} {int(r['trend']):>9} "
                f"{r['is_sharpe']:>+7.2f} {r['oos_sharpe']:>+7.2f} "
                f"{r['parity']:>7.2f} {r['oos_max_dd']:>7.1%} "
                f"{r['oos_trades_yr']:>7.0f}"
            )
    else:
        print("\n  No parameter combinations passed all gates.")
        print("  Top results by OOS Sharpe (regardless of gates):")
        top = combined.sort_values("oos_sharpe", ascending=False).head(10)
        for _, r in top.iterrows():
            print(
                f"    {r['instrument']:<10} EMA({int(r['fast'])}/{int(r['slow'])}) "
                f"SMA({int(r['trend'])})  OOS={r['oos_sharpe']:+.2f}  "
                f"parity={r['parity']:.2f}  trades/yr={r['oos_trades_yr']:.0f}"
            )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"ma_cross_m5_{ts}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")

    if not passing.empty:
        print("\n  NEXT STEPS for passing instruments:")
        print("  1. Add signal to titan/strategies/ic_generic/strategy.py _SIGNAL_REGISTRY")
        print('  2. Add instrument section to config/ic_generic.toml with tfs=["M5"]')
        print("  3. Add loader entry to research/portfolio/loaders/oos_returns.py _REGISTRY")


if __name__ == "__main__":
    main()
