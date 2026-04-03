"""run_ma_sweep.py -- Multi-Timeframe Moving-Average Cross Sweep.

Comprehensive MA cross sweep on currencies, gold, and indices using H1 data
as the base timeframe (covers all target instruments). Also sweeps M5 data
where available.

Sweep dimensions:
  - MA type: SMA, EMA, WMA
  - Fast period: 5 .. 50   (on H1: 5h to ~2 days)
  - Slow period: 20 .. 200  (on H1: ~1 day to ~8 days)
  - Trend filter: 120 .. 500 (on H1: ~5 days to ~3 weeks; 168=1 week FX)
  - Direction: long_only, both

Period equivalences (H1 bars):
    24 bars  = 1 FX day (24h)
   120 bars  = 1 FX week (5 x 24)
   168 bars  = 1 FX week (7 x 24, incl weekend gap)
   504 bars  = 1 FX month (21 x 24)

Instruments:
    FX:    EUR_USD, GBP_USD, USD_JPY, AUD_JPY, AUD_USD, USD_CHF, EUR_CHF
    Gold:  GLD
    Index: SPY, QQQ
    (FTSE/DAX only have daily data -- not included in H1 sweep)

Usage
-----
    uv run python research/ma_cross/run_ma_sweep.py
    uv run python research/ma_cross/run_ma_sweep.py --instruments EUR_USD GLD SPY
    uv run python research/ma_cross/run_ma_sweep.py --fast
    uv run python research/ma_cross/run_ma_sweep.py --tf M5 --instruments EUR_USD
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

# H1: FX trades ~24 bars/day (continuous), equity ~7 bars/day (9:30-16:00)
BARS_PER_YEAR = {
    "H1": 252 * 24,  # ~6,048 (FX-like, conservative for mixed)
    "M5": 252 * 78,  # ~19,656 (equity session)
}
VBT_FREQ = {"H1": "h", "M5": "5min"}

INIT_CASH: float = 10_000.0
MIN_BARS: int = 2_000
MIN_TRADES_PER_YEAR: float = 12.0

# Phase 3 gates
MIN_OOS_SHARPE: float = 0.0
MIN_PARITY: float = 0.50

# Cost profiles by asset type
PROFILES = {
    "fx": {"spread_bps": 0.5, "slippage_bps": 0.5, "max_leverage": 10.0},
    "fx_cross": {"spread_bps": 1.0, "slippage_bps": 0.5, "max_leverage": 10.0},
    "gold": {"spread_bps": 2.0, "slippage_bps": 1.0, "max_leverage": 5.0},
    "index": {"spread_bps": 1.0, "slippage_bps": 1.0, "max_leverage": 3.0},
}

# Target instruments and their asset types
INSTRUMENTS_H1 = {
    "EUR_USD": "fx",
    "GBP_USD": "fx",
    "USD_JPY": "fx",
    "AUD_JPY": "fx_cross",
    "AUD_USD": "fx_cross",
    "USD_CHF": "fx",
    "GLD": "gold",
    "SPY": "index",
    "QQQ": "index",
}
INSTRUMENTS_M5 = {
    "EUR_USD": "fx",
    "EUR_CHF": "fx_cross",
    "USD_CHF": "fx",
}

# Sweep grids
# H1 full grid
H1_FAST = [5, 8, 10, 15, 20, 25, 30, 40, 50]
H1_SLOW = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
H1_TREND = [120, 168, 200, 250, 336, 500]  # 168=1wk FX, 336=2wk, 500~3wk

# H1 fast grid
H1_FAST_QUICK = [10, 20, 30]
H1_SLOW_QUICK = [50, 100, 150]
H1_TREND_QUICK = [168, 336]

# M5 full grid
M5_FAST = [10, 20, 30, 50]
M5_SLOW = [50, 80, 100, 150, 200]
M5_TREND = [288, 500, 576, 1000]  # 288=24h FX, 576=2 FX days

# M5 fast grid
M5_FAST_QUICK = [20, 30]
M5_SLOW_QUICK = [80, 150]
M5_TREND_QUICK = [288, 576]

MA_TYPES = ["EMA", "SMA", "WMA"]


# ---------------------------------------------------------------------------
# MA computation helpers
# ---------------------------------------------------------------------------


def _ma(s: pd.Series, period: int, ma_type: str) -> pd.Series:
    """Compute moving average of given type. No shift — caller must shift."""
    if ma_type == "EMA":
        return s.ewm(span=period, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, period + 1, dtype=float)
        return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    else:  # SMA
        return s.rolling(period).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def _compute_signals(
    close: pd.Series,
    fast: int,
    slow: int,
    trend: int,
    ma_type: str,
    direction: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """MA cross entry/exit signals. All shifted 1 bar to prevent lookahead."""
    ma_fast = _ma(close, fast, ma_type).shift(1)
    ma_slow = _ma(close, slow, ma_type).shift(1)
    ma_trend = _ma(close, trend, "SMA").shift(1)  # trend filter always SMA

    fast_above = ma_fast > ma_slow
    prev_fast_above = fast_above.shift(1).fillna(False)
    cross_up = fast_above & ~prev_fast_above
    cross_dn = ~fast_above & prev_fast_above

    price_above_trend = close.shift(1) > ma_trend
    price_below_trend = close.shift(1) < ma_trend

    long_entries = cross_up & price_above_trend
    long_exits = cross_dn | ~price_above_trend

    if direction == "both":
        short_entries = cross_dn & price_below_trend
        short_exits = cross_up | ~price_below_trend
    else:
        short_entries = pd.Series(False, index=close.index)
        short_exits = pd.Series(False, index=close.index)

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# Single backtest
# ---------------------------------------------------------------------------


def _backtest_one(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    trend: int,
    ma_type: str,
    direction: str,
    asset_type: str,
    tf: str,
) -> dict | None:
    """IS+OOS backtest for one parameter combo. Returns None if gates fail hard."""
    if len(df) < MIN_BARS:
        return None

    close = df["close"]
    n = len(df)
    is_n = int(n * IS_RATIO)
    bars_yr = BARS_PER_YEAR.get(tf, 6048)
    freq = VBT_FREQ.get(tf, "h")

    profile = PROFILES.get(asset_type, PROFILES["fx"])
    med_close = float(close.median()) or 1.0
    fees_arr = ((profile["spread_bps"] / 10_000) * med_close / close).bfill().values
    slip_arr = ((profile["slippage_bps"] / 10_000) * med_close / close).bfill().values

    # ATR sizing: 1% risk, 2x ATR stop
    atr_s = _atr(df).shift(1)
    stop_pct = (2.0 * atr_s / close.where(close > 0, np.nan)).fillna(0.0)
    size_pct = (
        (0.01 / stop_pct.replace(0.0, np.nan)).clip(upper=profile["max_leverage"]).fillna(0.0)
    )

    long_en, long_ex, short_en, short_ex = _compute_signals(
        close, fast, slow, trend, ma_type, direction
    )

    def _run_slice(idx):
        cash = INIT_CASH / (2 if direction == "both" else 1)
        iloc_mask = df.index.get_indexer(idx)

        pf_long = vbt.Portfolio.from_signals(
            close.loc[idx],
            entries=long_en.loc[idx],
            exits=long_ex.loc[idx],
            size=size_pct.loc[idx].values,
            size_type="percent",
            init_cash=cash,
            fees=fees_arr[iloc_mask],
            slippage=slip_arr[iloc_mask],
            freq=freq,
            direction="longonly",
        )
        rets = pf_long.returns()
        n_trades = int(pf_long.trades.count())

        if direction == "both":
            pf_short = vbt.Portfolio.from_signals(
                close.loc[idx],
                entries=short_en.loc[idx],
                exits=short_ex.loc[idx],
                size=size_pct.loc[idx].values,
                size_type="percent",
                init_cash=cash,
                fees=fees_arr[iloc_mask],
                slippage=slip_arr[iloc_mask],
                freq=freq,
                direction="shortonly",
            )
            rets = (rets + pf_short.returns()) / 2.0
            n_trades += int(pf_short.trades.count())

        std = float(rets.std())
        sharpe = float(rets.mean() / std * np.sqrt(bars_yr)) if std > 1e-10 else 0.0
        eq = (1.0 + rets).cumprod()
        max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        total_ret = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
        n_yrs = len(idx) / bars_yr
        cagr = float((1 + total_ret) ** (1.0 / n_yrs) - 1) if n_yrs > 0 else 0.0
        tpy = n_trades / n_yrs if n_yrs > 0 else 0.0
        return {
            "sharpe": sharpe,
            "max_dd": max_dd,
            "cagr": cagr,
            "n_trades": n_trades,
            "trades_yr": tpy,
        }

    is_idx = df.index[:is_n]
    oos_idx = df.index[is_n:]

    is_s = _run_slice(is_idx)
    oos_s = _run_slice(oos_idx)

    parity = (oos_s["sharpe"] / is_s["sharpe"]) if is_s["sharpe"] != 0.0 else float("nan")

    pass_gate = (
        oos_s["sharpe"] > MIN_OOS_SHARPE
        and (not np.isnan(parity))
        and parity >= MIN_PARITY
        and oos_s["trades_yr"] >= MIN_TRADES_PER_YEAR
    )

    return {
        "ma_type": ma_type,
        "fast": fast,
        "slow": slow,
        "trend": trend,
        "direction": direction,
        "is_sharpe": round(is_s["sharpe"], 3),
        "oos_sharpe": round(oos_s["sharpe"], 3),
        "parity": round(parity, 3) if not np.isnan(parity) else float("nan"),
        "oos_max_dd": round(oos_s["max_dd"], 4),
        "oos_cagr": round(oos_s["cagr"], 4),
        "oos_trades_yr": round(oos_s["trades_yr"], 1),
        "pass": pass_gate,
    }


# ---------------------------------------------------------------------------
# Per-instrument sweep
# ---------------------------------------------------------------------------


def sweep_instrument(
    instrument: str,
    asset_type: str,
    tf: str,
    fast_periods: list[int],
    slow_periods: list[int],
    trend_periods: list[int],
    ma_types: list[str],
    direction: str,
) -> pd.DataFrame:
    """Run full parameter sweep for one instrument."""
    try:
        df = _load_ohlcv(instrument, tf)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {instrument}_{tf}: {exc}")
        return pd.DataFrame()

    print(
        f"  {instrument} ({asset_type}, {tf}): {len(df):,} bars  "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )

    combos = [
        (f, s, t, mt)
        for f, s, t, mt in product(fast_periods, slow_periods, trend_periods, ma_types)
        if f < s < t
    ]
    n_combos = len(combos) * (2 if direction == "both_separate" else 1)
    if direction == "both_separate":
        dirs = ["long_only", "both"]
    else:
        dirs = [direction]
        n_combos = len(combos)

    print(
        f"    {len(combos)} param combos x {len(dirs)} dir x {len(ma_types)} MA types = {n_combos} total ..."
    )

    rows: list[dict] = []
    done = 0
    for f, s, t, mt in combos:
        for d in dirs:
            done += 1
            if done % 200 == 0:
                print(f"      {done}/{n_combos} ...", end="\r")
            result = _backtest_one(df, f, s, t, mt, d, asset_type, tf)
            if result is not None:
                result["instrument"] = instrument
                result["tf"] = tf
                rows.append(result)

    if not rows:
        print(f"    No valid results for {instrument}.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    passing = result_df[result_df["pass"]]
    print(f"    {len(passing)}/{len(result_df)} pass gates.                    ")

    if not passing.empty:
        b = passing.iloc[0]
        print(
            f"    BEST: {b.ma_type}({int(b.fast)}/{int(b.slow)}) trend=SMA({int(b.trend)}) "
            f"dir={b.direction} | IS={b.is_sharpe:+.2f}  OOS={b.oos_sharpe:+.2f}  "
            f"parity={b.parity:.2f}  DD={b.oos_max_dd:.1%}  "
            f"trades/yr={b.oos_trades_yr:.0f}"
        )
    else:
        b = result_df.iloc[0]
        print(f"    No pass (best OOS={b.oos_sharpe:+.2f}, parity={b.parity:.2f})")

    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-TF MA cross sweep (FX, gold, indices).")
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Override instrument list.",
    )
    parser.add_argument(
        "--tf",
        default="H1",
        choices=["H1", "M5"],
        help="Base timeframe (default H1).",
    )
    parser.add_argument(
        "--direction",
        default="both_separate",
        choices=["long_only", "both", "both_separate"],
        help="both_separate tests long_only AND both (default).",
    )
    parser.add_argument(
        "--ma-types",
        nargs="+",
        default=MA_TYPES,
        choices=MA_TYPES,
        help="MA types to test (default: all three).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick mode: reduced grid.",
    )
    args = parser.parse_args()

    tf = args.tf
    all_instruments = INSTRUMENTS_H1 if tf == "H1" else INSTRUMENTS_M5
    data_dir = ROOT / "data"

    if args.instruments:
        instruments = {k: all_instruments.get(k, "fx") for k in args.instruments}
    else:
        instruments = {
            k: v for k, v in all_instruments.items() if (data_dir / f"{k}_{tf}.parquet").exists()
        }

    if tf == "H1":
        fast_p = H1_FAST_QUICK if args.fast else H1_FAST
        slow_p = H1_SLOW_QUICK if args.fast else H1_SLOW
        trend_p = H1_TREND_QUICK if args.fast else H1_TREND
    else:
        fast_p = M5_FAST_QUICK if args.fast else M5_FAST
        slow_p = M5_SLOW_QUICK if args.fast else M5_SLOW
        trend_p = M5_TREND_QUICK if args.fast else M5_TREND

    W = 80
    print()
    print("=" * W)
    print("  MA CROSS SWEEP -- Currencies / Gold / Indices")
    print(f"  Timeframe   : {tf}")
    print(f"  Instruments : {list(instruments.keys())}")
    print(f"  MA types    : {args.ma_types}")
    print(f"  Fast        : {fast_p}")
    print(f"  Slow        : {slow_p}")
    print(f"  Trend (SMA) : {trend_p}")
    print(f"  Direction   : {args.direction}")
    n_per = sum(1 for f, s, t in product(fast_p, slow_p, trend_p) if f < s < t)
    n_total = n_per * len(args.ma_types) * len(instruments)
    if args.direction == "both_separate":
        n_total *= 2
    print(f"  Combos      : ~{n_total:,} total across all instruments")
    print("=" * W)

    all_results: list[pd.DataFrame] = []
    for inst, atype in instruments.items():
        print(f"\n  --- {inst} ({atype}) ---")
        df = sweep_instrument(
            inst,
            atype,
            tf,
            fast_p,
            slow_p,
            trend_p,
            args.ma_types,
            args.direction,
        )
        if not df.empty:
            all_results.append(df)

    if not all_results:
        print("\n  No results. Check that data files exist.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    passing = combined[combined["pass"]].sort_values("oos_sharpe", ascending=False)

    print()
    print("=" * W)
    print(
        f"  SUMMARY: {len(passing)} passing / {len(combined)} total across {len(instruments)} instruments"
    )
    print("=" * W)

    if not passing.empty:
        print(
            f"\n  {'Inst':<10} {'MA':>4} {'Fast':>5} {'Slow':>5} {'Trend':>6} "
            f"{'Dir':<10} {'IS':>7} {'OOS':>7} {'Par':>5} "
            f"{'DD':>7} {'CAGR':>7} {'T/yr':>5}"
        )
        print("  " + "-" * 85)
        for _, r in passing.head(30).iterrows():
            print(
                f"  {r['instrument']:<10} {r['ma_type']:>4} "
                f"{int(r['fast']):>5} {int(r['slow']):>5} {int(r['trend']):>6} "
                f"{r['direction']:<10} "
                f"{r['is_sharpe']:>+7.2f} {r['oos_sharpe']:>+7.2f} "
                f"{r['parity']:>5.2f} "
                f"{r['oos_max_dd']:>7.1%} {r['oos_cagr']:>+7.1%} "
                f"{r['oos_trades_yr']:>5.0f}"
            )

        # Summary by MA type
        print("\n  Best OOS Sharpe by MA type (across all instruments):")
        for mt in args.ma_types:
            mt_pass = passing[passing["ma_type"] == mt]
            if not mt_pass.empty:
                b = mt_pass.iloc[0]
                print(
                    f"    {mt:>4}: OOS={b['oos_sharpe']:+.3f}  "
                    f"({b['instrument']} {int(b['fast'])}/{int(b['slow'])}/{int(b['trend'])} "
                    f"dir={b['direction']})"
                )
            else:
                print(f"    {mt:>4}: no passing combos")

        # Summary by instrument
        print("\n  Best OOS Sharpe by instrument:")
        for inst in instruments:
            inst_pass = passing[passing["instrument"] == inst]
            if not inst_pass.empty:
                b = inst_pass.iloc[0]
                print(
                    f"    {inst:<10}: OOS={b['oos_sharpe']:+.3f}  "
                    f"{b['ma_type']}({int(b['fast'])}/{int(b['slow'])}) "
                    f"trend={int(b['trend'])} dir={b['direction']}"
                )
            else:
                print(f"    {inst:<10}: no passing combos")
    else:
        print("\n  No combinations passed all gates.")
        top = combined.sort_values("oos_sharpe", ascending=False).head(15)
        print("  Top results regardless of gates:")
        for _, r in top.iterrows():
            print(
                f"    {r['instrument']:<10} {r['ma_type']:>4}({int(r['fast'])}/{int(r['slow'])}) "
                f"trend={int(r['trend'])}  OOS={r['oos_sharpe']:+.2f}  "
                f"par={r['parity']:.2f}  trd/yr={r['oos_trades_yr']:.0f}"
            )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"ma_cross_{tf}_{ts}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
