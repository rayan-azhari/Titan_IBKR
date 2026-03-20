"""turtle_sweep.py -- Parameter sensitivity sweep for CAT H1 Turtle system.

Sweeps entry/exit channel periods around the classic Turtle anchors:
  System 1: entry in [10..35], exit in [5..18]
  System 2: entry in [35..75], exit in [10..30]

Reports OOS Sharpe, Max DD, Risk of Ruin, and annual returns for every
combination.  Highlights the top-5 OOS Sharpe configurations per system
band with full metric drill-down.

Usage:
    uv run python research/turtle/turtle_sweep.py
    uv run python research/turtle/turtle_sweep.py --instrument CAT --timeframe H1
    uv run python research/turtle/turtle_sweep.py --instrument AMAT --timeframe D
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import vectorbt as vbt  # noqa: F401 (side-effect: needed by turtle_backtest)
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv add vectorbt")
    sys.exit(1)

# Import helpers from the main backtest module (no logic duplication)
from research.turtle.turtle_backtest import (  # noqa: E402
    ATR_PERIOD,
    BARS_PER_YEAR,
    INIT_CASH,
    IS_RATIO,
    VBT_FREQ,
    _combine_portfolios,
    _risk_of_ruin,
    _run_vbt,
    _stats,
    _stats_from_returns,
    atr,
    build_size_pct,
    compute_signals,
    load_data,
)

# ---------------------------------------------------------------------------
# Sweep grid definition
# ---------------------------------------------------------------------------

# System 1 anchor: (20, 10) — search a ±15 neighbourhood
S1_ENTRY_RANGE = list(range(10, 36, 5))   # 10 15 20 25 30 35
S1_EXIT_RANGE  = list(range(5,  19, 3))   # 5  8  11 14 17

# System 2 anchor: (55, 20) — search a ±20 neighbourhood
S2_ENTRY_RANGE = list(range(35, 76, 5))   # 35 40 45 50 55 60 65 70 75
S2_EXIT_RANGE  = list(range(10, 31, 5))   # 10 15 20 25 30

TOP_N = 5   # number of top configs to drill down per band


# ---------------------------------------------------------------------------
# Single-config backtest (re-uses turtle_backtest primitives)
# ---------------------------------------------------------------------------

def _run_config(
    df: pd.DataFrame,
    entry_p: int,
    exit_p: int,
    timeframe: str,
    direction: str,
) -> dict:
    """Run IS+OOS for one (entry_p, exit_p) combination.

    Returns a flat dict with is_sharpe, oos_sharpe, oos_dd, oos_ret,
    oos_ror, oos_trades, oos_wr, oos_annual_rets, oos_max_dd_days.
    """
    if exit_p >= entry_p:
        return {}   # invalid: exit channel must be tighter than entry

    freq = VBT_FREQ.get(timeframe, "d")
    bars_per_year = BARS_PER_YEAR.get(timeframe, 252)

    long_en, long_ex, short_en, short_ex, stop_pct = compute_signals(
        df, entry_p, exit_p
    )
    atr_shifted = atr(df, p=ATR_PERIOD).shift(1)
    size_pct = build_size_pct(df["close"], atr_shifted)

    n = len(df)
    is_n = int(n * IS_RATIO)
    is_idx = df.index[:is_n]
    oos_idx = df.index[is_n:]

    def _slice(idx: pd.Index) -> dict:
        cash = INIT_CASH / 2 if direction == "both" else INIT_CASH
        pf_long = _run_vbt(
            df["close"].loc[idx],
            long_en.loc[idx], long_ex.loc[idx],
            stop_pct.loc[idx], size_pct.loc[idx],
            freq=freq, direction="longonly", cash=cash,
        )
        if direction == "both":
            pf_short = _run_vbt(
                df["close"].loc[idx],
                short_en.loc[idx], short_ex.loc[idx],
                stop_pct.loc[idx], size_pct.loc[idx],
                freq=freq, direction="shortonly", cash=cash,
            )
            combined_rets = _combine_portfolios(pf_long, pf_short)
            return _stats_from_returns(combined_rets, timeframe=timeframe)
        return _stats(pf_long, timeframe=timeframe)

    try:
        is_s  = _slice(is_idx)
        oos_s = _slice(oos_idx)
    except Exception:
        return {}

    is_sh  = is_s["sharpe"]
    oos_sh = oos_s["sharpe"]
    parity = oos_sh / is_sh if is_sh != 0.0 else float("nan")
    oor_years = len(oos_idx) / bars_per_year

    ror = _risk_of_ruin(oos_s["daily_returns"], ruin_pct=0.50, horizon_years=5)

    return {
        "entry_p":        entry_p,
        "exit_p":         exit_p,
        "is_sharpe":      is_sh,
        "oos_sharpe":     oos_sh,
        "parity":         parity,
        "gate":           "PASS" if (not np.isnan(parity) and parity >= 0.5) else "FAIL",
        "oos_ret":        oos_s["ret"],
        "oos_dd":         oos_s["dd"],
        "oos_avg_dd":     oos_s["avg_dd"],
        "oos_max_dd_days": oos_s["max_dd_days"],
        "oos_wr":         oos_s["wr"],
        "oos_trades":     oos_s["trades"],
        "oos_avg_win":    oos_s["avg_win"],
        "oos_avg_loss":   oos_s["avg_loss"],
        "oos_ror":        ror,
        "oos_years":      oor_years,
        "oos_annual_rets": oos_s["annual_rets"],
        "oos_daily_rets": oos_s["daily_returns"],
        "is_stats":       is_s,
    }


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------

def run_sweep(
    df: pd.DataFrame,
    entry_range: list[int],
    exit_range: list[int],
    timeframe: str,
    direction: str,
    label: str,
) -> pd.DataFrame:
    """Run sweep over all (entry, exit) combinations and return ranked DataFrame."""
    total = sum(
        1 for ep in entry_range for xp in exit_range if xp < ep
    )
    print(f"\n  {label}: sweeping {total} configs ...")
    rows = []
    done = 0
    for ep in entry_range:
        for xp in exit_range:
            res = _run_config(df, ep, xp, timeframe, direction)
            if res:
                rows.append(res)
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total} done ...", end="\r", flush=True)
    print(f"    {total}/{total} done.    ")

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        return df_res
    return df_res.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SEP  = "=" * 80
THIN = "-" * 80


def _fmt(v: float, pct: bool = False) -> str:
    if np.isnan(v):
        return f"{'N/A':>10}"
    if pct:
        return f"{v:>+9.1%}"
    return f"{v:>+9.3f}"


def print_sweep_table(label: str, df_sweep: pd.DataFrame) -> None:
    """Print ranked sweep results table (all rows, condensed)."""
    print(f"\n  {label} — all configs ranked by OOS Sharpe")
    print(f"  {'Entry':>6}  {'Exit':>5}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}"
          f"  {'OOS/IS':>7}  {'Gate':>5}  {'OOS MaxDD':>10}  {'OOS RoR':>8}")
    print(f"  {THIN}")
    for _, row in df_sweep.iterrows():
        parity_str = f"{row['parity']:>+6.2f}" if not np.isnan(row["parity"]) else "   N/A"
        ror_str    = f"{row['oos_ror']:>6.1%}" if not np.isnan(row["oos_ror"]) else "   N/A"
        print(
            f"  {int(row['entry_p']):>6}  {int(row['exit_p']):>5}"
            f"  {row['is_sharpe']:>+10.3f}  {row['oos_sharpe']:>+10.3f}"
            f"  {parity_str}  {'PASS' if row['gate'] == 'PASS' else 'FAIL':>5}"
            f"  {row['oos_dd']:>+9.1%}  {ror_str}"
        )


def print_top_configs(label: str, df_sweep: pd.DataFrame, n: int = TOP_N) -> None:
    """Print full metric drill-down for the top-N OOS Sharpe configs."""
    top = df_sweep.head(n)
    print(f"\n{SEP}")
    print(f"  TOP {n} CONFIGS  —  {label}")
    print(SEP)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        ep = int(row["entry_p"])
        xp = int(row["exit_p"])
        parity_str = f"{row['parity']:>+.2f}" if not np.isnan(row["parity"]) else "N/A"
        ror_str    = f"{row['oos_ror']:.1%}" if not np.isnan(row["oos_ror"]) else "N/A"
        is_s = row["is_stats"]

        print(f"\n  Rank #{rank}  —  ({ep}-bar entry / {xp}-bar exit)"
              f"  Gate: {row['gate']}")
        print(THIN)
        print(f"  {'Metric':<26} {'IS':>12}  {'OOS':>12}")
        print(THIN)
        print(f"  {'Sharpe':<26} {_fmt(is_s['sharpe']):>12}  {_fmt(row['oos_sharpe']):>12}")
        print(f"  {'Total Return':<26} {_fmt(is_s['ret'], pct=True):>12}"
              f"  {_fmt(row['oos_ret'], pct=True):>12}")
        print(f"  {'Max Drawdown':<26} {_fmt(is_s['dd'], pct=True):>12}"
              f"  {_fmt(row['oos_dd'], pct=True):>12}")
        print(f"  {'Avg Drawdown':<26} {_fmt(is_s['avg_dd'], pct=True):>12}"
              f"  {_fmt(row['oos_avg_dd'], pct=True):>12}")
        print(f"  {'Max DD Duration (bars)':<26} {is_s['max_dd_days']:>12}"
              f"  {int(row['oos_max_dd_days']):>12}")
        print(f"  {'Win Rate':<26} {_fmt(is_s['wr'], pct=True):>12}"
              f"  {_fmt(row['oos_wr'], pct=True):>12}")
        print(f"  {'Avg Win':<26} {_fmt(is_s['avg_win'], pct=True):>12}"
              f"  {_fmt(row['oos_avg_win'], pct=True):>12}")
        print(f"  {'Avg Loss':<26} {_fmt(is_s['avg_loss'], pct=True):>12}"
              f"  {_fmt(row['oos_avg_loss'], pct=True):>12}")
        print(f"  {'Trades':<26} {is_s['trades']:>12}  {int(row['oos_trades']):>12}")
        print(THIN)
        print(f"  OOS/IS Sharpe: {parity_str}   RoR (OOS, -50% DD, 5yr, 5k): {ror_str}")

        # Annual returns (OOS only)
        ann = row["oos_annual_rets"]
        if ann:
            print("\n  Annual Returns (OOS):")
            keys = sorted(ann.keys())
            pairs = [(keys[i], keys[i + 1] if i + 1 < len(keys) else None)
                     for i in range(0, len(keys), 2)]
            for k1, k2 in pairs:
                r2 = (f"  {k2:<8} {ann[k2]:>+9.1%}" if k2 else "")
                print(f"  {k1:<8} {ann[k1]:>+9.1%}{r2}")


def print_heatmap(label: str, df_sweep: pd.DataFrame, metric: str = "oos_sharpe") -> None:
    """Print ASCII heatmap of a metric over (entry_p, exit_p) grid."""
    if df_sweep.empty:
        return
    pivot = df_sweep.pivot_table(
        index="entry_p", columns="exit_p", values=metric, aggfunc="mean"
    )
    entries = sorted(pivot.index)
    exits   = sorted(pivot.columns)

    metric_label = metric.replace("oos_", "OOS ").replace("_", " ").title()
    print(f"\n  {metric_label} heatmap  (rows=entry, cols=exit):")
    col_hdr = "entry/exit"
    header = f"  {col_hdr:>12}" + "".join(f"  {xp:>5}" for xp in exits)
    print(header)
    print(f"  {THIN[:len(header)-2]}")
    for ep in entries:
        row_vals = "".join(
            f"  {pivot.loc[ep, xp]:>+5.2f}" if xp in pivot.columns else "     -- "
            for xp in exits
        )
        print(f"  {ep:>12}{row_vals}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turtle parameter sensitivity sweep"
    )
    parser.add_argument("--instrument", default="CAT")
    parser.add_argument("--timeframe",  default="H1", choices=["D", "H1", "M5"])
    parser.add_argument("--direction",  default="long", choices=["long", "both"])
    parser.add_argument("--start",      default=None,
                        help="Filter data from date, e.g. 2013-01-01")
    parser.add_argument("--data-dir",   default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    print(f"\n{SEP}")
    print(f"  TURTLE SWEEP  --  {args.instrument}  [{args.timeframe}]"
          f"  direction={args.direction}")
    print(f"  S1 entry {S1_ENTRY_RANGE}  exit {S1_EXIT_RANGE}")
    print(f"  S2 entry {S2_ENTRY_RANGE}  exit {S2_EXIT_RANGE}")
    print(SEP)

    print(f"\nLoading {args.timeframe} data for {args.instrument} ...")
    df = load_data(args.instrument, args.timeframe, data_dir=data_dir)

    if args.start:
        cutoff = pd.Timestamp(args.start, tz="UTC")
        df = df.loc[df.index >= cutoff]
        print(f"  Filtered to {args.start} onwards: {len(df)} bars")

    print(f"  {len(df)} bars  |  {df.index[0].date()} to {df.index[-1].date()}")

    # System 1 sweep
    s1_label = f"{args.instrument} {args.timeframe} System-1 band"
    df_s1 = run_sweep(df, S1_ENTRY_RANGE, S1_EXIT_RANGE,
                      args.timeframe, args.direction, s1_label)

    # System 2 sweep
    s2_label = f"{args.instrument} {args.timeframe} System-2 band"
    df_s2 = run_sweep(df, S2_ENTRY_RANGE, S2_EXIT_RANGE,
                      args.timeframe, args.direction, s2_label)

    # Full ranked tables
    print(f"\n{SEP}")
    if not df_s1.empty:
        print_sweep_table(s1_label, df_s1)
    if not df_s2.empty:
        print_sweep_table(s2_label, df_s2)

    # Heatmaps
    if not df_s1.empty:
        print_heatmap(s1_label, df_s1, "oos_sharpe")
    if not df_s2.empty:
        print_heatmap(s2_label, df_s2, "oos_sharpe")

    # Top-N drill-downs
    if not df_s1.empty:
        print_top_configs(s1_label, df_s1, TOP_N)
    if not df_s2.empty:
        print_top_configs(s2_label, df_s2, TOP_N)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
