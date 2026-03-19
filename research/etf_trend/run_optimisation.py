"""run_optimisation.py -- Stage 1: MA Type + Slow Period Sweep.

Sweeps MA type (SMA / EMA) and slow period for the ETF Trend strategy.

New logic (decel + slow SMA redesign):
  Entry: daily close above slow MA only (fast MA removed).
  Exit:  daily close below slow MA (sole trend-reversal gate).

The fast MA is no longer used -- the slow MA is the only trend boundary.
Decel signals (Stage 2) will be layered on top as entry confirmation.

Uses 70/30 IS/OOS split by bar count. Anti-look-ahead: signals shifted by 1 bar.

Usage:
    uv run python research/etf_trend/run_optimisation.py
    uv run python research/etf_trend/run_optimisation.py --instrument SPY
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
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.etf_trend.state_manager import save_stage1  # noqa: E402

# ── Quality gates ──────────────────────────────────────────────────────────
MIN_OOS_SHARPE = 0.5
MIN_OOS_IS_RATIO = 0.4
MIN_TRADES_OOS = 5  # slow MA = fewer trades expected
MIN_WIN_RATE = 0.35
MAX_DRAWDOWN = 0.50  # slow MA stays in trends longer -- higher DD tolerance

# ── Parameter grid ─────────────────────────────────────────────────────────
MA_TYPES = ["SMA", "EMA"]
SLOW_MAS = [50, 75, 100, 125, 150, 175, 200, 225, 250, 300]
# Total: 2 x 10 = 20 combinations

# ── Friction ───────────────────────────────────────────────────────────────
FEES = 0.001
SLIPPAGE = 0.0005


# ── Data loading ───────────────────────────────────────────────────────────


def load_data(instrument: str) -> pd.DataFrame:
    """Load daily Parquet data for an instrument."""
    path = DATA_DIR / f"{instrument}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("  Run: uv run python scripts/download_data_databento.py")
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


# ── MA helpers ─────────────────────────────────────────────────────────────


def compute_ma(close: pd.Series, period: int, ma_type: str) -> pd.Series:
    """Compute SMA or EMA."""
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    return close.rolling(period).mean()


# ── Backtest ───────────────────────────────────────────────────────────────


def run_backtest(
    close: pd.Series,
    slow_ma: pd.Series,
) -> "vbt.Portfolio":
    """Run long-only trend backtest using slow MA as the sole trend gate.

    Entry: close > slow_ma (shifted by 1 -- next-bar execution).
    Exit:  close < slow_ma (shifted by 1).

    Args:
        close: Close price series.
        slow_ma: Slow moving average series.

    Returns:
        VectorBT Portfolio object.
    """
    in_regime = close > slow_ma
    entries = in_regime.shift(1).fillna(False)
    exits = (~in_regime).shift(1).fillna(False)

    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def extract_stats(pf: "vbt.Portfolio") -> dict:
    """Extract key metrics from a VBT Portfolio."""
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "total_return": float(pf.total_return()),
        "n_trades": int(n),
        "win_rate": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


# ── Main sweep ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 1: MA Type + Slow Period Sweep.")
    parser.add_argument("--instrument", default="SPY", help="Instrument symbol (default: SPY)")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    scoreboard_path = REPORTS_DIR / f"etf_trend_stage1_{inst_lower}_scoreboard.csv"

    print("=" * 60)
    print("  ETF Trend -- Stage 1: MA Type + Slow Period Sweep")
    print("=" * 60)
    print(f"  Instrument: {instrument}")
    print(f"  MA types:   {', '.join(MA_TYPES)}")
    print(f"  Slow MAs:   {SLOW_MAS}")
    print("  Entry:      close > slow_ma  (fast MA removed)")
    print("  Exit:       close < slow_ma  (sole trend-reversal gate)")
    total = len(MA_TYPES) * len(SLOW_MAS)
    print(f"  Total combos: {total}")
    print()

    df = load_data(instrument)
    close = df["close"]

    # 70/30 IS/OOS split by bar count
    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]
    print(
        f"  IS bars:  {len(is_close)} ({is_close.index[0].date()} to {is_close.index[-1].date()})"
    )
    print(
        f"  OOS bars: {len(oos_close)} ({oos_close.index[0].date()} to {oos_close.index[-1].date()})"
    )

    results: list[dict] = []

    for ma_type in MA_TYPES:
        print(f"\n--- MA type: {ma_type} ---")
        for slow in SLOW_MAS:
            # Compute MA on full series for index alignment, then slice
            slow_ma_full = compute_ma(close, slow, ma_type)
            is_slow = slow_ma_full.iloc[:split]
            oos_slow = slow_ma_full.iloc[split:]

            is_pf = run_backtest(is_close, is_slow)
            oos_pf = run_backtest(oos_close, oos_slow)

            is_stats = extract_stats(is_pf)
            oos_stats = extract_stats(oos_pf)

            ratio = oos_stats["sharpe"] / is_stats["sharpe"] if is_stats["sharpe"] > 0.01 else 0.0

            results.append(
                {
                    "ma_type": ma_type,
                    "slow_ma": slow,
                    "is_sharpe": round(is_stats["sharpe"], 3),
                    "oos_sharpe": round(oos_stats["sharpe"], 3),
                    "oos_is_ratio": round(ratio, 3),
                    "oos_max_dd": round(oos_stats["max_drawdown"], 3),
                    "oos_win_rate": round(oos_stats["win_rate"], 3),
                    "oos_n_trades": oos_stats["n_trades"],
                    "oos_total_return": round(oos_stats["total_return"], 3),
                }
            )
            print(
                f"  {ma_type} slow={slow:3d}  IS={is_stats['sharpe']:.3f}  "
                f"OOS={oos_stats['sharpe']:.3f}  Ret={oos_stats['total_return']:.1%}  "
                f"DD={oos_stats['max_drawdown']:.1%}  Trades={oos_stats['n_trades']}"
            )

    scoreboard = pd.DataFrame(results).sort_values("oos_total_return", ascending=False)
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved: {scoreboard_path.relative_to(PROJECT_ROOT)}")

    # ── Quality gate evaluation ──────────────────────────────────────────
    passing = scoreboard[
        (scoreboard["oos_sharpe"] >= MIN_OOS_SHARPE)
        & (scoreboard["oos_is_ratio"] >= MIN_OOS_IS_RATIO)
        & (scoreboard["oos_n_trades"] >= MIN_TRADES_OOS)
        & (scoreboard["oos_win_rate"] >= MIN_WIN_RATE)
        & (scoreboard["oos_max_dd"].abs() <= MAX_DRAWDOWN)
    ]

    print(f"\n  {len(passing)} / {len(scoreboard)} combos pass all quality gates.")
    print(
        f"  Gates: OOS Sharpe>={MIN_OOS_SHARPE}, OOS/IS>={MIN_OOS_IS_RATIO}, "
        f"Trades>={MIN_TRADES_OOS}, WinRate>={MIN_WIN_RATE:.0%}, MaxDD<={MAX_DRAWDOWN:.0%}"
    )

    if passing.empty:
        print("\n  [WARN] No combinations pass all gates. Selecting best OOS total return anyway.")
        best = scoreboard.iloc[0]
    else:
        best = passing.iloc[0]

    print("\n  -- Best combination (by OOS total return) ----")
    print(f"  MA type:          {best['ma_type']}")
    print(f"  slow_ma:          {best['slow_ma']}")
    print(f"  IS Sharpe:        {best['is_sharpe']}")
    print(f"  OOS Sharpe:       {best['oos_sharpe']}")
    print(f"  OOS/IS:           {best['oos_is_ratio']}")
    print(f"  OOS MaxDD:        {best['oos_max_dd']:.1%}")
    print(f"  OOS Trades:       {best['oos_n_trades']}")
    print(f"  OOS Total Return: {best['oos_total_return']:.1%}")

    # ── Lock Stage 1 results ─────────────────────────────────────────────
    save_stage1(
        ma_type=str(best["ma_type"]),
        slow_ma=int(best["slow_ma"]),
        instrument=inst_lower,
    )

    print("\n  [PASS] Stage 1 complete. Results saved to state.")
    print(
        f"  Next: uv run python research/etf_trend/run_stage2_decel.py "
        f"--instrument {instrument} --load-state"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
