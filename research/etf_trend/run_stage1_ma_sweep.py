"""run_stage1_ma_sweep.py -- Stage 1: Dual-MA Parameter Sweep.

Tests (fast_ma, slow_ma) regime gate combinations per Malik strategy design.

Two entry modes:
  slow_only   -- entry when close > slow_ma (single MA gate, current behaviour)
  dual_regime -- entry when close > fast_ma AND close > slow_ma (Malik regime gate)

Selects winner by OOS total return and saves (ma_type, slow_ma, fast_ma) to state.
fast_ma is None for slow_only winners.

Usage:
    uv run python research/etf_trend/run_stage1_ma_sweep.py --instrument QQQ
    uv run python research/etf_trend/run_stage1_ma_sweep.py --instrument SPY
    uv run python research/etf_trend/run_stage1_ma_sweep.py --instrument TQQQ --signal-instrument QQQ
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

from research.etf_trend.state_manager import save_stage1  # noqa: E402

FEES = 0.001
SLIPPAGE = 0.0005

# Malik strategy parameter grid
SLOW_MAS = [100, 125, 150, 175, 200, 250]
FAST_MAS = [30, 50, 75, 100]  # only used in dual_regime mode
ENTRY_MODES = ["slow_only", "dual_regime"]
MA_TYPE = "SMA"


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


def compute_ma(close: pd.Series, period: int, ma_type: str = "SMA") -> pd.Series:
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    return close.rolling(period).mean()


def run_backtest(
    close: pd.Series,
    slow_ma: pd.Series,
    fast_ma: pd.Series | None = None,
    exec_close: pd.Series | None = None,
) -> "vbt.Portfolio":
    """Backtest with either slow_only or dual_regime entry gate.

    slow_only:   entry when close > slow_ma
    dual_regime: entry when close > fast_ma AND close > slow_ma

    Exit in both cases: close < slow_ma.
    All signals shifted 1 bar to prevent look-ahead bias.

    Args:
        close: Signal close (used for MA comparison and entry/exit logic).
        slow_ma: Slow MA series computed on signal close.
        fast_ma: Fast MA for dual_regime mode.
        exec_close: Execution close for P&L (defaults to close when None).
    """
    exec_c = exec_close if exec_close is not None else close
    above_slow = close > slow_ma
    if fast_ma is not None:
        entry_signal = above_slow & (close > fast_ma)
    else:
        entry_signal = above_slow

    entries = entry_signal.shift(1).fillna(False)
    exits = (~above_slow).shift(1).fillna(False)

    return vbt.Portfolio.from_signals(
        exec_c,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 1: Dual-MA Parameter Sweep.")
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument(
        "--signal-instrument",
        default=None,
        help="Signal source instrument (e.g. QQQ for TQQQ). Defaults to --instrument.",
    )
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()
    signal_inst = (args.signal_instrument or instrument).upper()
    is_dual = signal_inst != instrument

    print("=" * 60)
    print("  ETF Trend -- Stage 1: Dual-MA Parameter Sweep")
    print("=" * 60)
    print(f"  Instrument: {instrument}")
    if is_dual:
        print(f"  Signal src: {signal_inst}  (signals on {signal_inst}, P&L on {instrument})")
    print(f"  Slow MAs:   {SLOW_MAS}")
    print(f"  Fast MAs:   {FAST_MAS}  (dual_regime only)")
    print(f"  Modes:      {ENTRY_MODES}")

    exec_df = load_data(instrument)
    if is_dual:
        sig_df = load_data(signal_inst)
        common = sig_df.index.intersection(exec_df.index)
        sig_df = sig_df.loc[common]
        exec_df = exec_df.loc[common]
        close = sig_df["close"]
        exec_close_full = exec_df["close"]
    else:
        close = exec_df["close"]
        exec_close_full = close

    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]
    is_exec = exec_close_full.iloc[:split]
    oos_exec = exec_close_full.iloc[split:]

    print(
        f"\n  IS  period: {is_close.index[0].date()} to {is_close.index[-1].date()}"
        f" ({len(is_close)} bars)"
    )
    print(
        f"  OOS period: {oos_close.index[0].date()} to {oos_close.index[-1].date()}"
        f" ({len(oos_close)} bars)"
    )

    results: list[dict] = []
    print("\n  Running combinations ...\n")

    for slow_p in SLOW_MAS:
        slow_full = compute_ma(close, slow_p, MA_TYPE)
        is_slow = slow_full.iloc[:split]
        oos_slow = slow_full.iloc[split:]

        # slow_only mode (fast_ma = None)
        is_pf = run_backtest(is_close, is_slow, fast_ma=None, exec_close=is_exec)
        oos_pf = run_backtest(oos_close, oos_slow, fast_ma=None, exec_close=oos_exec)
        is_sharpe = float(is_pf.sharpe_ratio())
        oos_sharpe = float(oos_pf.sharpe_ratio())
        oos_ret = float(oos_pf.total_return())
        ratio = oos_sharpe / is_sharpe if is_sharpe > 0.01 else 0.0
        label = f"slow_only  slow={slow_p:3d}"
        print(
            f"  {label}  IS={is_sharpe:.3f}  OOS={oos_sharpe:.3f}  "
            f"Ret={oos_ret:.1%}  Ratio={ratio:.2f}"
        )
        results.append(
            {
                "entry_mode": "slow_only",
                "slow_ma": slow_p,
                "fast_ma": None,
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "oos_is_ratio": round(ratio, 3),
                "oos_total_return": round(oos_ret, 3),
            }
        )

        # dual_regime mode (sweep fast_ma)
        for fast_p in FAST_MAS:
            if fast_p >= slow_p:
                continue  # fast must be shorter than slow
            fast_full = compute_ma(close, fast_p, MA_TYPE)
            is_fast = fast_full.iloc[:split]
            oos_fast = fast_full.iloc[split:]

            is_pf = run_backtest(is_close, is_slow, fast_ma=is_fast, exec_close=is_exec)
            oos_pf = run_backtest(oos_close, oos_slow, fast_ma=oos_fast, exec_close=oos_exec)
            is_sharpe = float(is_pf.sharpe_ratio())
            oos_sharpe = float(oos_pf.sharpe_ratio())
            oos_ret = float(oos_pf.total_return())
            ratio = oos_sharpe / is_sharpe if is_sharpe > 0.01 else 0.0
            label = f"dual_regime slow={slow_p:3d} fast={fast_p:3d}"
            print(
                f"  {label}  IS={is_sharpe:.3f}  OOS={oos_sharpe:.3f}  "
                f"Ret={oos_ret:.1%}  Ratio={ratio:.2f}"
            )
            results.append(
                {
                    "entry_mode": "dual_regime",
                    "slow_ma": slow_p,
                    "fast_ma": fast_p,
                    "is_sharpe": round(is_sharpe, 3),
                    "oos_sharpe": round(oos_sharpe, 3),
                    "oos_is_ratio": round(ratio, 3),
                    "oos_total_return": round(oos_ret, 3),
                }
            )

    scoreboard = pd.DataFrame(results).sort_values("oos_total_return", ascending=False)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage1_{inst_lower}_ma_sweep.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved: {scoreboard_path.relative_to(PROJECT_ROOT)}")

    best = scoreboard.iloc[0]
    best_slow = int(best["slow_ma"])
    best_fast_raw = best["fast_ma"]
    best_fast: int | None = int(best_fast_raw) if pd.notna(best_fast_raw) else None
    best_entry = str(best["entry_mode"])

    print("\n  -- Best configuration (by OOS total return) --------")
    print(f"  Entry mode:       {best_entry}")
    print(f"  Slow MA:          SMA {best_slow}")
    if best_fast:
        print(f"  Fast MA:          SMA {best_fast}")
    print(f"  IS  Sharpe:       {best['is_sharpe']}")
    print(f"  OOS Sharpe:       {best['oos_sharpe']}")
    print(f"  OOS/IS ratio:     {best['oos_is_ratio']}")
    print(f"  OOS Total Return: {best['oos_total_return']:.1%}")

    save_stage1(
        ma_type=MA_TYPE,
        slow_ma=best_slow,
        instrument=inst_lower,
        fast_ma=best_fast,
    )

    print("\n  [PASS] Stage 1 complete.")
    print(
        f"  Next: uv run python research/etf_trend/run_stage2_decel.py "
        f"--instrument {instrument} --load-state"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
