"""turtle_universe.py -- Apply validated Turtle S2 (45/30) to all H1 instruments.

Loads config/turtle_h1.toml [system2] parameters and runs a fixed-param
IS/OOS backtest on every H1 parquet file in data/.  Outputs a ranked
leaderboard sorted by OOS Sharpe.

Usage:
    uv run python research/turtle/turtle_universe.py
    uv run python research/turtle/turtle_universe.py --system 1
    uv run python research/turtle/turtle_universe.py --equities-only
    uv run python research/turtle/turtle_universe.py --min-bars 8000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import vectorbt as vbt  # noqa: F401
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.phase1_sweep import _load_ohlcv  # noqa: E402
from research.turtle.turtle_backtest import (  # noqa: E402
    BARS_PER_YEAR,
    INIT_CASH,
    IS_RATIO,
    VBT_FREQ,
    _combine_portfolios,
    _risk_of_ruin,
    _run_vbt,
    _stats,
    _stats_from_returns,
    build_size_pct,
    compute_pyramid_entries,
    compute_signals,
)

# ---------------------------------------------------------------------------
# FX pairs — excluded from equities-only mode
# ---------------------------------------------------------------------------
FX_TICKERS = {
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "AUD_JPY",
    "USD_CHF", "NZD_USD", "EUR_JPY", "GBP_JPY",
}

# Non-US / non-standard tickers to always skip
SKIP_TICKERS: set[str] = set()   # ISF.L (FTSE ETF) and EXS1.DE (DAX ETF) now included

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

CONFIG_PATH = ROOT / "config" / "turtle_h1.toml"


def load_system_config(system_num: int) -> dict:
    """Load system parameters from config/turtle_h1.toml."""
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    key = f"system{system_num}"
    if key not in cfg:
        raise KeyError(f"[{key}] section not found in {CONFIG_PATH}")
    return cfg[key]


# ---------------------------------------------------------------------------
# Single-instrument backtest (fixed params from config)
# ---------------------------------------------------------------------------

def _run_instrument(
    ticker: str,
    entry_p: int,
    exit_p: int,
    atr_period: int,
    direction: str,
    timeframe: str = "H1",
    data_dir: Path | None = None,
    min_bars: int = 4000,
    pyramid: bool = False,
    trailing_stop: bool = False,
    max_units: int = 4,
    pyramid_atr_mult: float = 0.5,
) -> dict | None:
    """Load H1 data for ticker and run IS+OOS backtest.

    Returns None on load error or insufficient data.
    """
    try:
        df = _load_ohlcv(ticker, timeframe, data_dir=data_dir)
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}

    if len(df) < min_bars:
        return {
            "ticker": ticker,
            "error": f"only {len(df)} bars (need {min_bars})",
        }

    freq = VBT_FREQ.get(timeframe, "h")
    bars_per_year = BARS_PER_YEAR.get(timeframe, 252 * 14)

    try:
        long_en, long_ex, short_en, short_ex, stop_pct, hi_entry, atr_shifted = (
            compute_signals(df, entry_p, exit_p)
        )
        size_pct = build_size_pct(df["close"], atr_shifted)

        # Pyramid: expand entry signals to include scale-in levels
        if pyramid:
            long_en = compute_pyramid_entries(
                df["close"], hi_entry, atr_shifted,
                max_units=max_units, pyramid_atr_mult=pyramid_atr_mult,
            )

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
                sl_trail=trailing_stop, accumulate=pyramid,
            )
            if direction == "both":
                pf_short = _run_vbt(
                    df["close"].loc[idx],
                    short_en.loc[idx], short_ex.loc[idx],
                    stop_pct.loc[idx], size_pct.loc[idx],
                    freq=freq, direction="shortonly", cash=cash,
                    sl_trail=trailing_stop, accumulate=pyramid,
                )
                return _stats_from_returns(
                    _combine_portfolios(pf_long, pf_short), timeframe=timeframe
                )
            return _stats(pf_long, timeframe=timeframe)

        is_s  = _slice(is_idx)
        oos_s = _slice(oos_idx)

    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}

    is_sh  = is_s["sharpe"]
    oos_sh = oos_s["sharpe"]
    parity = oos_sh / is_sh if is_sh != 0.0 else float("nan")
    oos_years = len(oos_idx) / bars_per_year
    ror = _risk_of_ruin(oos_s["daily_returns"], ruin_pct=0.50, horizon_years=5)

    return {
        "ticker":          ticker,
        "bars":            n,
        "oos_years":       oos_years,
        "is_sharpe":       is_sh,
        "oos_sharpe":      oos_sh,
        "parity":          parity,
        "gate":            "PASS" if (not np.isnan(parity) and parity >= 0.5) else "FAIL",
        "oos_ret":         oos_s["ret"],
        "oos_dd":          oos_s["dd"],
        "oos_wr":          oos_s["wr"],
        "oos_trades":      oos_s["trades"],
        "oos_ror":         ror,
        "oos_annual_rets": oos_s["annual_rets"],
        "oos_avg_win":     oos_s["avg_win"],
        "oos_avg_loss":    oos_s["avg_loss"],
        "is_stats":        is_s,
        "oos_stats":       oos_s,
        "error":           None,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SEP  = "=" * 90
THIN = "-" * 90


def print_leaderboard(rows: list[dict], system_num: int, entry_p: int, exit_p: int) -> None:
    """Print ranked leaderboard table."""
    ok   = [r for r in rows if not r.get("error")]
    fail = [r for r in rows if r.get("error")]

    ok.sort(key=lambda r: r["oos_sharpe"], reverse=True)

    print(f"\n{SEP}")
    print(
        f"  TURTLE UNIVERSE  --  H1  S{system_num} ({entry_p}/{exit_p})"
        f"  |  {len(ok)} instruments  |  ranked by OOS Sharpe"
    )
    print(SEP)
    print(
        f"  {'#':>3}  {'Ticker':<10}  {'Bars':>7}  {'IS Sharpe':>10}"
        f"  {'OOS Sharpe':>10}  {'OOS/IS':>7}  {'Gate':>5}"
        f"  {'OOS Ret':>8}  {'OOS DD':>8}  {'RoR':>6}"
    )
    print(f"  {THIN}")

    for rank, r in enumerate(ok, 1):
        parity_str = f"{r['parity']:>+6.2f}" if not np.isnan(r["parity"]) else "   N/A"
        ror_str    = f"{r['oos_ror']:>5.1%}" if not np.isnan(r["oos_ror"]) else "  N/A"
        print(
            f"  {rank:>3}  {r['ticker']:<10}  {r['bars']:>7}"
            f"  {r['is_sharpe']:>+10.3f}  {r['oos_sharpe']:>+10.3f}"
            f"  {parity_str}  {'PASS' if r['gate'] == 'PASS' else 'FAIL':>5}"
            f"  {r['oos_ret']:>+7.1%}  {r['oos_dd']:>+7.1%}  {ror_str}"
        )

    # Summary
    n_pass = sum(1 for r in ok if r["gate"] == "PASS")
    print(f"\n  PASS: {n_pass}/{len(ok)}  ({100*n_pass/len(ok):.0f}%)")

    # Annual returns for top-10 PASS instruments
    top_pass = [r for r in ok if r["gate"] == "PASS"][:10]
    if top_pass:
        print("\n  Annual OOS Returns — Top 10 PASS instruments:")
        all_years: set[str] = set()
        for r in top_pass:
            all_years.update(r["oos_annual_rets"].keys())
        years = sorted(all_years)
        hdr = f"  {'Ticker':<10}" + "".join(f"  {y:>8}" for y in years)
        print(hdr)
        print(f"  {'-' * (len(hdr)-2)}")
        for r in top_pass:
            ann = r["oos_annual_rets"]
            vals = "".join(
                f"  {ann[y]:>+7.1%}" if y in ann else "       --"
                for y in years
            )
            print(f"  {r['ticker']:<10}{vals}")

    # Skipped / errored
    if fail:
        print(f"\n  Skipped ({len(fail)}):")
        for r in fail:
            print(f"    {r['ticker']:<12}  {r.get('error', 'unknown')}")

    print(f"\n{SEP}\n")


def _fv(v: float, fmt: str = "+.3f") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "    N/A"
    return f"{v:{fmt}}"


def _fvp(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "    N/A"
    return f"{v:>+7.1%}"


def print_pass_metrics(rows: list[dict], system_num: int, entry_p: int, exit_p: int) -> None:
    """Print full IS/OOS metric table for every PASS instrument, ranked by OOS Sharpe."""
    ok = [r for r in rows if not r.get("error") and r.get("gate") == "PASS"]
    ok.sort(key=lambda r: r["oos_sharpe"], reverse=True)

    if not ok:
        print("No PASS instruments to display.")
        return

    W = 28
    print(f"\n{SEP}")
    print(
        f"  FULL METRICS — PASS instruments  |  H1  S{system_num} ({entry_p}/{exit_p})"
        f"  |  {len(ok)} instruments"
    )

    for r in ok:
        is_s  = r["is_stats"]
        oos_s_raw = r.get("oos_stats", {})  # full dict stored separately if present
        # Fall back to top-level oos fields if oos_stats not stored
        ticker  = r["ticker"]
        parity  = r["parity"]
        gate    = r["gate"]
        parity_str = f"{parity:+.2f}" if not np.isnan(parity) else "N/A"

        print(f"\n{SEP}")
        print(f"  {ticker:<10}  |  OOS/IS parity: {parity_str}  Gate: {gate}")
        print(THIN)
        print(f"  {'Metric':<{W}} {'IS':>10}  {'OOS':>10}")
        print(THIN)

        def _oos(key: str, default: float = float("nan")) -> float:
            if oos_s_raw and key in oos_s_raw:
                return oos_s_raw[key]
            # fall back to top-level keys
            mapping = {
                "sharpe": "oos_sharpe", "ret": "oos_ret", "dd": "oos_dd",
                "wr": "oos_wr", "trades": "oos_trades", "avg_win": "oos_avg_win",
                "avg_loss": "oos_avg_loss",
            }
            return r.get(mapping.get(key, ""), default)

        # Returns
        print(f"  {'Total Return':<{W}} {_fvp(is_s['ret'])}  {_fvp(_oos('ret'))}")
        print(f"  {'Annualised Return':<{W}} {_fvp(is_s.get('ann_ret'))}  {_fvp(oos_s_raw.get('ann_ret') if oos_s_raw else float('nan'))}")
        print(THIN)

        # Risk-adjusted
        print(f"  {'Sharpe':<{W}} {_fv(is_s['sharpe']):>10}  {_fv(_oos('sharpe')):>10}")
        print(f"  {'Sortino':<{W}} {_fv(is_s.get('sortino')):>10}  {_fv(oos_s_raw.get('sortino') if oos_s_raw else float('nan')):>10}")
        print(f"  {'Calmar':<{W}} {_fv(is_s.get('calmar')):>10}  {_fv(oos_s_raw.get('calmar') if oos_s_raw else float('nan')):>10}")
        print(THIN)

        # Drawdown
        print(f"  {'Max Drawdown':<{W}} {_fvp(is_s['dd'])}  {_fvp(_oos('dd'))}")
        print(f"  {'Avg Drawdown':<{W}} {_fvp(is_s.get('avg_dd', float('nan')))}  {_fvp(oos_s_raw.get('avg_dd', float('nan')) if oos_s_raw else float('nan'))}")
        max_dd_is  = is_s.get('max_dd_days', 0)
        max_dd_oos = oos_s_raw.get('max_dd_days', 0) if oos_s_raw else 0
        print(f"  {'Max DD Duration (bars)':<{W}} {max_dd_is:>10}  {max_dd_oos:>10}")
        print(THIN)

        # Capital deployed
        td_is  = is_s.get('time_deployed', float('nan'))
        td_oos = oos_s_raw.get('time_deployed', float('nan')) if oos_s_raw else float('nan')
        td_is_s  = f"{td_is:>9.1%}"  if not (td_is  is None or np.isnan(td_is))  else "       N/A"
        td_oos_s = f"{td_oos:>9.1%}" if not (td_oos is None or np.isnan(td_oos)) else "       N/A"
        print(f"  {'Capital Deployed':<{W}} {td_is_s}  {td_oos_s}")
        print(THIN)

        # Trades
        n_long_is  = is_s.get('n_long',  is_s.get('trades', 0))
        n_short_is = is_s.get('n_short', 0)
        n_long_oos  = oos_s_raw.get('n_long',  _oos('trades', 0)) if oos_s_raw else _oos('trades', 0)
        n_short_oos = oos_s_raw.get('n_short', 0) if oos_s_raw else 0
        print(f"  {'Trades (total)':<{W}} {is_s['trades']:>10}  {int(_oos('trades', 0)):>10}")
        print(f"  {'  Long trades':<{W}} {int(n_long_is):>10}  {int(n_long_oos):>10}")
        print(f"  {'  Short trades':<{W}} {int(n_short_is):>10}  {int(n_short_oos):>10}")
        print(THIN)

        # Win/loss
        print(f"  {'Win Rate':<{W}} {_fvp(is_s.get('wr', 0))}  {_fvp(_oos('wr', 0))}")
        print(f"  {'Avg Win':<{W}} {_fvp(is_s.get('avg_win', 0))}  {_fvp(_oos('avg_win', 0))}")
        print(f"  {'Avg Loss':<{W}} {_fvp(is_s.get('avg_loss', 0))}  {_fvp(_oos('avg_loss', 0))}")
        payoff_is  = is_s.get('payoff', float('nan'))
        payoff_oos = oos_s_raw.get('payoff', float('nan')) if oos_s_raw else float('nan')
        print(f"  {'Payoff (|win/loss|)':<{W}} {_fv(payoff_is, '.2f'):>10}  {_fv(payoff_oos, '.2f'):>10}")
        print(THIN)

        # Duration
        avg_is  = is_s.get('avg_dur_bars',    float('nan'))
        avg_oos = oos_s_raw.get('avg_dur_bars',    float('nan')) if oos_s_raw else float('nan')
        med_is  = is_s.get('median_dur_bars', float('nan'))
        med_oos = oos_s_raw.get('median_dur_bars', float('nan')) if oos_s_raw else float('nan')

        def _dur(bars: float) -> str:
            if np.isnan(bars):
                return "       N/A"
            h = int(bars)
            d, rem = divmod(h, 14)
            return f"{d}d {rem}h" if d > 0 else f"{rem}h"

        print(f"  {'Avg Trade Duration':<{W}} {_dur(avg_is):>10}  {_dur(avg_oos):>10}")
        print(f"  {'Median Trade Duration':<{W}} {_dur(med_is):>10}  {_dur(med_oos):>10}")

        # Annual OOS returns
        ann = r.get("oos_annual_rets", {})
        if ann:
            print("\n  Annual OOS Returns:")
            keys = sorted(ann)
            pairs = [(keys[i], keys[i+1] if i+1 < len(keys) else None) for i in range(0, len(keys), 2)]
            for k1, k2 in pairs:
                r2 = f"  {k2:<8} {ann[k2]:>+8.1%}" if k2 else ""
                print(f"  {k1:<8} {ann[k1]:>+8.1%}{r2}")

        # RoR
        ror_str = f"{r['oos_ror']:.1%}" if not np.isnan(r["oos_ror"]) else "N/A"
        print(f"\n  Risk of Ruin (OOS, -50% DD, 5-yr, 5k sims): {ror_str}")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turtle H1 universe scan — apply config params to all instruments"
    )
    parser.add_argument(
        "--system", default=2, type=int, choices=[1, 2],
        help="Which config system to use: 1 (10/8) or 2 (45/30, default)",
    )
    parser.add_argument(
        "--equities-only", action="store_true",
        help="Skip FX pairs and non-standard tickers",
    )
    parser.add_argument(
        "--min-bars", default=4000, type=int,
        help="Skip instruments with fewer than this many H1 bars (default: 4000)",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--workers", default=4, type=int,
        help="Number of parallel threads (default: 4). VBT/Numba releases GIL.",
    )
    parser.add_argument(
        "--full-metrics", action="store_true",
        help="After the leaderboard, print full IS/OOS metric blocks for all PASS instruments.",
    )
    args = parser.parse_args()

    # Load validated params from config
    cfg = load_system_config(args.system)
    entry_p         = cfg["entry_period"]
    exit_p          = cfg["exit_period"]
    atr_period      = cfg["atr_period"]
    direction       = cfg["direction"]
    timeframe       = cfg.get("timeframe", "H1")
    pyramid         = cfg.get("max_units", 1) > 1
    trailing_stop   = cfg.get("use_trailing_stop", False)
    max_units       = cfg.get("max_units", 4)
    pyramid_atr_mult = cfg.get("pyramid_atr_mult", 0.5)

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data"

    # Discover all H1 parquet tickers
    h1_files = sorted(data_dir.glob("*_H1.parquet"))
    tickers = [f.stem.replace("_H1", "") for f in h1_files]

    if args.equities_only:
        tickers = [t for t in tickers if t not in FX_TICKERS and t not in SKIP_TICKERS]
    else:
        tickers = [t for t in tickers if t not in SKIP_TICKERS]

    pyr_label = f"pyramid x{max_units}" if pyramid else "no pyramid"
    sl_label  = "trailing SL" if trailing_stop else "fixed SL"
    print(f"\n{SEP}")
    print(
        f"  TURTLE UNIVERSE  --  H1  S{args.system} ({entry_p}/{exit_p})"
        f"  direction={direction}  {pyr_label}  {sl_label}"
    )
    print(f"  Config: {CONFIG_PATH.name} [{cfg['description']}]")
    print(f"  Instruments: {len(tickers)}  |  min_bars={args.min_bars}")
    print(SEP)

    rows: list[dict] = []
    completed = 0

    def _submit(ticker: str) -> dict | None:
        return _run_instrument(
            ticker, entry_p, exit_p, atr_period,
            direction=direction, timeframe=timeframe,
            data_dir=data_dir, min_bars=args.min_bars,
            pyramid=pyramid, trailing_stop=trailing_stop,
            max_units=max_units, pyramid_atr_mult=pyramid_atr_mult,
        )

    n_workers = min(args.workers, len(tickers))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_ticker = {pool.submit(_submit, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            result = future.result()
            if result is None:
                print(f"  [{completed:>3}/{len(tickers)}] {ticker:<12} SKIP (None)")
                continue
            if result.get("error"):
                print(f"  [{completed:>3}/{len(tickers)}] {ticker:<12} ERR  {result['error'][:55]}")
            else:
                oos_sh = result["oos_sharpe"]
                gate   = result["gate"]
                print(f"  [{completed:>3}/{len(tickers)}] {ticker:<12} OOS Sharpe {oos_sh:>+6.3f}  {gate}")
            rows.append(result)

    print_leaderboard(rows, args.system, entry_p, exit_p)
    if args.full_metrics:
        print_pass_metrics(rows, args.system, entry_p, exit_p)


if __name__ == "__main__":
    main()
