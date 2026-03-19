"""run_ma_crossover_sweep.py -- MA Crossover Parameter Sweep.

Entry: fast SMA crosses above slow SMA (golden cross), executed next bar.
Exit:  fast SMA remains below slow SMA for >= confirm_days consecutive bars,
       executed next bar.

Sweeps (fast_ma, slow_ma, confirm_days) grid.
Reports IS/OOS Sharpe, OOS/IS ratio, OOS return, OOS max drawdown.

Compares best combo vs three benchmarks (OOS period):
  1. B&H QQQ
  2. Simple price > SMA200 (Stage 1 slow-only baseline)
  3. Full Malik QQQ pipeline (locked config reference constants)

Usage:
    uv run python research/etf_trend/run_ma_crossover_sweep.py --instrument QQQ
    uv run python research/etf_trend/run_ma_crossover_sweep.py --instrument SPY
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

FEES = 0.001        # 0.10% per side
SLIPPAGE = 0.0005   # 0.05% per side

# Parameter grid
SLOW_MAS = [50, 75, 100, 150, 200, 250]
FAST_MAS = [20, 30, 50, 75, 100, 125, 150]
CONFIRM_DAYS = [1, 2, 3, 5]

# Full Malik QQQ pipeline locked-config OOS reference (2018-2026)
# Source: run_portfolio.py --instrument QQQ output
MALIK_REF = {"sharpe": 1.15, "total_return": 2.910, "max_drawdown": -0.235}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(instrument: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/download_data_yfinance.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index().dropna(subset=["close"])


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------

def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def run_crossover_backtest(
    close: pd.Series,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    confirm_days: int,
) -> "vbt.Portfolio":
    """MA crossover backtest (binary sizing, long-only).

    Entry: fast crosses above slow (was below yesterday, above today).
           Executed on the next bar open (shift(1)).
    Exit:  fast has been below slow for >= confirm_days consecutive bars.
           Executed on the next bar open after confirmation.

    No look-ahead bias: all signals shifted +1 before passing to VBT.
    """
    fast_above = fast_ma > slow_ma

    # Crossover up: transition from False -> True on this bar
    crossover_up = fast_above & ~fast_above.shift(1).fillna(False)

    # Confirmed crossunder: rolling count of "fast below slow" bars
    fast_below = ~fast_above
    consec_below = fast_below.rolling(confirm_days).sum() >= confirm_days

    # Shift all signals +1 bar: detected on day T, executed on day T+1
    entries = crossover_up.shift(1).fillna(False)
    exits = consec_below.shift(1).fillna(False)

    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def run_price_above_sma(close: pd.Series, slow_ma: pd.Series) -> "vbt.Portfolio":
    """Stage 1 baseline: entry when close > slow MA, exit when close < slow MA."""
    above = close > slow_ma
    entries = above.shift(1).fillna(False)
    exits = (~above).shift(1).fillna(False)
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats_from_pf(pf: "vbt.Portfolio") -> dict:
    total_ret = float(pf.total_return())
    sharpe = float(pf.sharpe_ratio())
    max_dd = float(pf.max_drawdown())
    calmar = total_ret / abs(max_dd) if abs(max_dd) > 0.001 else 0.0
    return {
        "total_return": round(total_ret, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
    }


def bah_stats(close: pd.Series) -> dict:
    rets = close.pct_change().dropna()
    total_ret = float(close.iloc[-1] / close.iloc[0] - 1)
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
    max_dd = float((close / close.cummax() - 1).min())
    calmar = total_ret / abs(max_dd) if abs(max_dd) > 0.001 else 0.0
    return {
        "total_return": round(total_ret, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MA Crossover Parameter Sweep.")
    parser.add_argument("--instrument", default="QQQ", help="Symbol (default: QQQ)")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    valid_pairs = [
        (f, s) for s in SLOW_MAS for f in FAST_MAS if f < s
    ]
    total_combos = len(valid_pairs) * len(CONFIRM_DAYS)

    print("=" * 70)
    print("  MA Crossover Parameter Sweep")
    print("=" * 70)
    print(f"  Instrument:   {instrument}")
    print(f"  Slow MAs:     {SLOW_MAS}")
    print(f"  Fast MAs:     {FAST_MAS}")
    print(f"  Confirm days: {CONFIRM_DAYS}")
    print(f"  Grid size:    {total_combos} combinations")

    df = load_data(instrument)
    close = df["close"]

    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]

    print(f"\n  IS  period: {is_close.index[0].date()} -> {is_close.index[-1].date()}"
          f"  ({len(is_close)} bars)")
    print(f"  OOS period: {oos_close.index[0].date()} -> {oos_close.index[-1].date()}"
          f"  ({len(oos_close)} bars)")

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------
    bah = bah_stats(oos_close)
    sma200_full = compute_sma(close, 200)
    baseline_pf = run_price_above_sma(oos_close, sma200_full.iloc[split:])
    baseline = stats_from_pf(baseline_pf)

    print("\n  -- Benchmarks (OOS) ----------------------------------------")
    print(
        f"  B&H {instrument}:          Sharpe={bah['sharpe']:.3f}  "
        f"Return={bah['total_return']:.1%}  MaxDD={bah['max_drawdown']:.1%}"
    )
    print(
        f"  price>SMA200 (Stage1): Sharpe={baseline['sharpe']:.3f}  "
        f"Return={baseline['total_return']:.1%}  MaxDD={baseline['max_drawdown']:.1%}"
    )
    print(
        f"  Full Malik {instrument}:   Sharpe={MALIK_REF['sharpe']:.3f}  "
        f"Return={MALIK_REF['total_return']:.1%}  MaxDD={MALIK_REF['max_drawdown']:.1%}"
        f"  (locked config, binary+vol_target)"
    )

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    results: list[dict] = []
    print(f"\n  Running {total_combos} combinations ...\n")

    for slow_p in SLOW_MAS:
        slow_full = compute_sma(close, slow_p)
        is_slow = slow_full.iloc[:split]
        oos_slow = slow_full.iloc[split:]

        for fast_p in FAST_MAS:
            if fast_p >= slow_p:
                continue

            fast_full = compute_sma(close, fast_p)
            is_fast = fast_full.iloc[:split]
            oos_fast = fast_full.iloc[split:]

            for cd in CONFIRM_DAYS:
                is_pf = run_crossover_backtest(is_close, is_fast, is_slow, cd)
                oos_pf = run_crossover_backtest(oos_close, oos_fast, oos_slow, cd)
                is_s = stats_from_pf(is_pf)
                oos_s = stats_from_pf(oos_pf)
                ratio = (
                    oos_s["sharpe"] / is_s["sharpe"]
                    if is_s["sharpe"] > 0.01 else 0.0
                )
                label = f"fast={fast_p:3d} slow={slow_p:3d} cd={cd}"
                print(
                    f"  {label}  IS={is_s['sharpe']:6.3f}  "
                    f"OOS={oos_s['sharpe']:6.3f}  "
                    f"Ret={oos_s['total_return']:7.1%}  "
                    f"DD={oos_s['max_drawdown']:7.1%}  "
                    f"Ratio={ratio:.2f}"
                )
                results.append({
                    "fast_ma": fast_p,
                    "slow_ma": slow_p,
                    "confirm_days": cd,
                    "is_sharpe": is_s["sharpe"],
                    "oos_sharpe": oos_s["sharpe"],
                    "oos_is_ratio": round(ratio, 3),
                    "oos_total_return": oos_s["total_return"],
                    "oos_max_drawdown": oos_s["max_drawdown"],
                    "oos_calmar": oos_s["calmar"],
                })

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    scoreboard = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
    csv_path = REPORTS_DIR / f"ma_crossover_sweep_{inst_lower}.csv"
    scoreboard.to_csv(csv_path, index=False)
    print(f"\n  Scoreboard saved: {csv_path.relative_to(PROJECT_ROOT)}")

    best = scoreboard.iloc[0]
    best_fast = int(best["fast_ma"])
    best_slow = int(best["slow_ma"])
    best_cd = int(best["confirm_days"])

    print("\n" + "=" * 70)
    print("  WINNER (by OOS Sharpe)")
    print("=" * 70)
    print(f"  Fast MA:      SMA {best_fast}")
    print(f"  Slow MA:      SMA {best_slow}")
    print(f"  Confirm days: {best_cd}")
    print(f"  IS  Sharpe:   {best['is_sharpe']:.3f}")
    print(f"  OOS Sharpe:   {best['oos_sharpe']:.3f}")
    print(f"  OOS/IS ratio: {best['oos_is_ratio']:.3f}  "
          f"{'[OK]' if best['oos_is_ratio'] >= 0.5 else '[WARN: < 0.5, possible overfit]'}")
    print(f"  OOS Return:   {best['oos_total_return']:.1%}")
    print(f"  OOS MaxDD:    {best['oos_max_drawdown']:.1%}")
    print(f"  OOS Calmar:   {best['oos_calmar']:.3f}")

    print("\n  -- vs benchmarks (OOS Sharpe) ------------------------------")
    beats_bah = best["oos_sharpe"] > bah["sharpe"]
    beats_stage1 = best["oos_sharpe"] > baseline["sharpe"]
    beats_malik = best["oos_sharpe"] > MALIK_REF["sharpe"]
    print(f"  vs B&H {instrument}:          {bah['sharpe']:.3f}  ->  "
          f"{'BEATS' if beats_bah else 'LOSES'}")
    print(f"  vs price>SMA200:       {baseline['sharpe']:.3f}  ->  "
          f"{'BEATS' if beats_stage1 else 'LOSES'}")
    print(f"  vs Full Malik {instrument}:   {MALIK_REF['sharpe']:.3f}  ->  "
          f"{'BEATS' if beats_malik else 'LOSES'}")

    # Top 10 table
    print("\n  Top 10 by OOS Sharpe:")
    hdr = f"  {'fast':>4}  {'slow':>4}  {'cd':>2}  "
    hdr += f"{'IS Sh':>6}  {'OOS Sh':>6}  {'Ratio':>5}  {'Return':>7}  {'MaxDD':>7}"
    print(hdr)
    print("  " + "-" * 62)
    for _, row in scoreboard.head(10).iterrows():
        ratio_flag = "" if row["oos_is_ratio"] >= 0.5 else " (!)"
        print(
            f"  {int(row['fast_ma']):>4}  {int(row['slow_ma']):>4}  "
            f"{int(row['confirm_days']):>2}  "
            f"{row['is_sharpe']:>6.3f}  {row['oos_sharpe']:>6.3f}  "
            f"{row['oos_is_ratio']:>5.2f}{ratio_flag}  "
            f"{row['oos_total_return']:>7.1%}  {row['oos_max_drawdown']:>7.1%}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
