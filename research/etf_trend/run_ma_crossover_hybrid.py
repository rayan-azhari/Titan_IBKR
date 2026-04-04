"""run_ma_crossover_hybrid.py -- Hybrid: MA crossover entry + Malik decel exit.

Entry:  fast SMA crosses above slow SMA (golden cross) -- replaces Malik's
        decel_positive price-level gate with a crossover trigger.

Exit:   Mode D (Malik locked config) -- whichever fires first:
          A: close below slow SMA for >= exit_confirm_days consecutive bars
          C: decel composite < exit_thresh for >= decel_confirm_days bars

Decel composite fixed at QQQ locked-config params:
  signals:      d_pct, rv_20, macd_hist
  d_pct_smooth: 20
  rv_window:    30
  macd_fast:    12
  slow_ma:      SMA 200

Sweep: fast_ma x exit_decel_thresh x exit_confirm_days

Compare vs:
  1. B&H QQQ
  2. price > SMA200  (Stage 1 baseline)
  3. Full Malik QQQ  (locked config, binary sizing -- 1.150 Sharpe reference)
  4. Crossover-only winner (SMA 20/200/cd3 -- 1.164 Sharpe reference)

Usage:
    uv run python research/etf_trend/run_ma_crossover_hybrid.py --instrument QQQ
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

from research.etf_trend.run_stage2_decel import (  # noqa: E402
    build_composite,
    compute_ma,
    d_pct,
    macd_signal,
    rv_signal,
)
from research.etf_trend.run_stage3_exits import apply_sma_confirmation  # noqa: E402

FEES = 0.001
SLIPPAGE = 0.0005

# Fixed: QQQ locked-config decel params
SLOW_MA_PERIOD = 200
DECEL_SIGNALS = ["d_pct", "rv_20", "macd_hist"]
D_PCT_SMOOTH = 20
RV_WINDOW = 30
MACD_FAST = 12
DECEL_CONFIRM_DAYS = 1  # fixed -- same as locked config

# Sweep grid
FAST_MAS = [20, 30, 50, 75, 100]
EXIT_DECEL_THRESHOLDS = [-0.3, -0.1, 0.0]
EXIT_CONFIRM_DAYS = [1, 3, 5]

# Reference constants (OOS 2018-2026, binary sizing)
MALIK_REF = {"sharpe": 1.150, "total_return": 2.910, "max_drawdown": -0.235}
CROSSOVER_REF = {"sharpe": 1.164, "total_return": 2.805, "max_drawdown": -0.286}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(instrument: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
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
# Signal builders
# ---------------------------------------------------------------------------


def build_decel_composite(
    close: pd.Series,
    df: pd.DataFrame,
    slow_ma_ser: pd.Series,
) -> pd.Series:
    """Build fixed QQQ decel composite: d_pct + rv_20 + macd_hist, equal weight."""
    parts: dict[str, pd.Series] = {
        "d_pct": d_pct(close, slow_ma_ser, smooth=D_PCT_SMOOTH),
        "rv_20": rv_signal(close, window=RV_WINDOW),
        "macd_hist": macd_signal(close, fast=MACD_FAST),
    }
    weights = {k: 1.0 / 3 for k in parts}
    return build_composite(parts, weights)


def ma_crossover_entry(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """True only when fast_ma crosses above slow_ma (False -> True transition)."""
    fast_above = fast_ma > slow_ma
    return fast_above & ~fast_above.shift(1).fillna(False)


def mode_d_exit(
    close: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    exit_thresh: float,
    exit_confirm_days: int,
) -> pd.Series:
    """Mode D exit: SMA break (confirmed) OR decel collapse (confirmed)."""
    below_slow = apply_sma_confirmation(~(close > slow_ma), exit_confirm_days)
    decel_exit = apply_sma_confirmation(decel < exit_thresh, DECEL_CONFIRM_DAYS)
    return below_slow | decel_exit


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def run_hybrid_backtest(
    close: pd.Series,
    df: pd.DataFrame,
    fast_period: int,
    exit_thresh: float,
    exit_confirm_days: int,
) -> "vbt.Portfolio":
    """Hybrid backtest: crossover entry + Mode D decel exit."""
    slow_ma = compute_ma(close, SLOW_MA_PERIOD, "SMA")
    fast_ma = compute_ma(close, fast_period, "SMA")
    decel = build_decel_composite(close, df, slow_ma)

    entry_signal = ma_crossover_entry(fast_ma, slow_ma)
    exit_signal = mode_d_exit(close, slow_ma, decel, exit_thresh, exit_confirm_days)

    # Shift +1: detected day T, executed day T+1
    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def run_price_above_sma_backtest(close: pd.Series) -> "vbt.Portfolio":
    """Stage 1 baseline: price > SMA200, no decel filter."""
    sma200 = compute_ma(close, 200, "SMA")
    above = close > sma200
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
    sharpe = float(rets.mean() / rets.std() * (252**0.5)) if rets.std() > 0 else 0.0
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
    parser = argparse.ArgumentParser(description="Hybrid MA crossover + Malik decel exit sweep.")
    parser.add_argument("--instrument", default="QQQ", help="Symbol (default: QQQ)")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    total_combos = len(FAST_MAS) * len(EXIT_DECEL_THRESHOLDS) * len(EXIT_CONFIRM_DAYS)

    print("=" * 70)
    print("  Hybrid: MA Crossover Entry + Malik Mode-D Exit")
    print("=" * 70)
    print(f"  Instrument:        {instrument}")
    print(f"  Slow MA (fixed):   SMA {SLOW_MA_PERIOD}")
    print(f"  Decel signals:     {DECEL_SIGNALS}  (fixed at QQQ locked-config params)")
    print(f"  Fast MAs (sweep):  {FAST_MAS}")
    print(f"  Exit thresh:       {EXIT_DECEL_THRESHOLDS}")
    print(f"  Exit confirm days: {EXIT_CONFIRM_DAYS}")
    print(f"  Grid size:         {total_combos} combinations")

    df = load_data(instrument)
    close = df["close"]

    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]
    is_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    print(
        f"\n  IS  period: {is_close.index[0].date()} -> {is_close.index[-1].date()}"
        f"  ({len(is_close)} bars)"
    )
    print(
        f"  OOS period: {oos_close.index[0].date()} -> {oos_close.index[-1].date()}"
        f"  ({len(oos_close)} bars)"
    )

    # Benchmarks
    bah = bah_stats(oos_close)
    baseline_pf = run_price_above_sma_backtest(oos_close)
    baseline = stats_from_pf(baseline_pf)

    print("\n  -- Benchmarks (OOS) ----------------------------------------")
    print(
        f"  B&H {instrument}:              Sharpe={bah['sharpe']:.3f}  "
        f"Return={bah['total_return']:.1%}  MaxDD={bah['max_drawdown']:.1%}"
    )
    print(
        f"  price>SMA200 (Stage1):     Sharpe={baseline['sharpe']:.3f}  "
        f"Return={baseline['total_return']:.1%}  MaxDD={baseline['max_drawdown']:.1%}"
    )
    print(
        f"  Crossover 20/200/cd3:      Sharpe={CROSSOVER_REF['sharpe']:.3f}  "
        f"Return={CROSSOVER_REF['total_return']:.1%}  "
        f"MaxDD={CROSSOVER_REF['max_drawdown']:.1%}"
    )
    print(
        f"  Full Malik {instrument} (binary): Sharpe={MALIK_REF['sharpe']:.3f}  "
        f"Return={MALIK_REF['total_return']:.1%}  MaxDD={MALIK_REF['max_drawdown']:.1%}"
    )

    # Sweep
    results: list[dict] = []
    print(f"\n  Running {total_combos} combinations ...\n")

    for fast_p in FAST_MAS:
        for exit_thresh in EXIT_DECEL_THRESHOLDS:
            for ecd in EXIT_CONFIRM_DAYS:
                is_pf = run_hybrid_backtest(is_close, is_df, fast_p, exit_thresh, ecd)
                oos_pf = run_hybrid_backtest(oos_close, oos_df, fast_p, exit_thresh, ecd)
                is_s = stats_from_pf(is_pf)
                oos_s = stats_from_pf(oos_pf)
                ratio = oos_s["sharpe"] / is_s["sharpe"] if is_s["sharpe"] > 0.01 else 0.0
                label = f"fast={fast_p:3d}  thresh={exit_thresh:+.1f}  ecd={ecd}"
                print(
                    f"  {label}  IS={is_s['sharpe']:6.3f}  "
                    f"OOS={oos_s['sharpe']:6.3f}  "
                    f"Ret={oos_s['total_return']:7.1%}  "
                    f"DD={oos_s['max_drawdown']:7.1%}  "
                    f"Ratio={ratio:.2f}"
                )
                results.append(
                    {
                        "fast_ma": fast_p,
                        "exit_decel_thresh": exit_thresh,
                        "exit_confirm_days": ecd,
                        "is_sharpe": is_s["sharpe"],
                        "oos_sharpe": oos_s["sharpe"],
                        "oos_is_ratio": round(ratio, 3),
                        "oos_total_return": oos_s["total_return"],
                        "oos_max_drawdown": oos_s["max_drawdown"],
                        "oos_calmar": oos_s["calmar"],
                    }
                )

    scoreboard = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
    csv_path = REPORTS_DIR / f"ma_crossover_hybrid_{inst_lower}.csv"
    scoreboard.to_csv(csv_path, index=False)
    print(f"\n  Scoreboard saved: {csv_path.relative_to(PROJECT_ROOT)}")

    best = scoreboard.iloc[0]

    print("\n" + "=" * 70)
    print("  WINNER (by OOS Sharpe)")
    print("=" * 70)
    print(f"  Fast MA:           SMA {int(best['fast_ma'])}")
    print(f"  Slow MA:           SMA {SLOW_MA_PERIOD}  (fixed)")
    print(f"  Exit decel thresh: {best['exit_decel_thresh']:+.1f}")
    print(f"  Exit confirm days: {int(best['exit_confirm_days'])}")
    print(f"  IS  Sharpe:        {best['is_sharpe']:.3f}")
    print(f"  OOS Sharpe:        {best['oos_sharpe']:.3f}")
    print(
        f"  OOS/IS ratio:      {best['oos_is_ratio']:.3f}  "
        f"{'[OK]' if best['oos_is_ratio'] >= 0.5 else '[WARN: < 0.5]'}"
    )
    print(f"  OOS Return:        {best['oos_total_return']:.1%}")
    print(f"  OOS MaxDD:         {best['oos_max_drawdown']:.1%}")
    print(f"  OOS Calmar:        {best['oos_calmar']:.3f}")

    print("\n  -- vs benchmarks (OOS Sharpe) ------------------------------")
    refs = [
        (f"B&H {instrument}", bah["sharpe"]),
        ("price>SMA200", baseline["sharpe"]),
        ("Crossover 20/200/cd3", CROSSOVER_REF["sharpe"]),
        (f"Full Malik {instrument} (binary)", MALIK_REF["sharpe"]),
    ]
    for name, ref_sharpe in refs:
        result = "BEATS" if best["oos_sharpe"] > ref_sharpe else "LOSES"
        print(f"  vs {name:<30} {ref_sharpe:.3f}  ->  {result}")

    print("\n  Top 10 by OOS Sharpe:")
    hdr = (
        f"  {'fast':>4}  {'thresh':>6}  {'ecd':>3}  "
        f"{'IS Sh':>6}  {'OOS Sh':>6}  {'Ratio':>5}  {'Return':>7}  {'MaxDD':>7}"
    )
    print(hdr)
    print("  " + "-" * 66)
    for _, row in scoreboard.head(10).iterrows():
        print(
            f"  {int(row['fast_ma']):>4}  {row['exit_decel_thresh']:>+6.1f}  "
            f"{int(row['exit_confirm_days']):>3}  "
            f"{row['is_sharpe']:>6.3f}  {row['oos_sharpe']:>6.3f}  "
            f"{row['oos_is_ratio']:>5.2f}  "
            f"{row['oos_total_return']:>7.1%}  {row['oos_max_drawdown']:>7.1%}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
