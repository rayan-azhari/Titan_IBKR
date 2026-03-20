"""turtle_backtest.py -- Turtle Trading System POC (Systems 1 & 2).

Classic Turtle rules adapted for equity bars (daily or 5-minute intraday).
  System 1: 20-bar Donchian breakout entry / 10-bar breakdown exit
  System 2: 55-bar Donchian breakout entry / 20-bar breakdown exit

On daily bars  (--timeframe D):   20 bars = 20 trading days (~1 month)
On 5-min bars  (--timeframe M5):  20 bars = 100 minutes (~1.5 hr intraday momentum)
                                  55 bars = 275 minutes (~4.5 hr = near full session)

Position sizing: unit = (equity * risk_pct) / (ATR_20 * 1.0)
Stop loss: 2 ATR below entry price (via VectorBT sl_stop).
Long-only. No pyramiding. IS/OOS split 70/30.

Data sources:
  Daily (D):   data/<TICKER>_D.parquet  (up to 11yr history)
  5-min (M5):  data/databento/<TICKER>_1yr_5m.csv  (~1yr, 90 stocks)

Recommended daily instruments (11yr): AMAT, CAT, GLD, SPY
Recommended 5-min instruments (1yr):  AMAT, CAT, NVDA, AAPL, XOM

Usage:
    uv run python research/turtle/turtle_backtest.py
    uv run python research/turtle/turtle_backtest.py --instrument CAT
    uv run python research/turtle/turtle_backtest.py --instrument AMAT --timeframe M5
    uv run python research/turtle/turtle_backtest.py --instrument NVDA --timeframe M5 --system 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.phase1_sweep import _load_ohlcv  # noqa: E402
from titan.strategies.ml.features import atr  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INIT_CASH: float = 100_000.0
IS_RATIO: float = 0.70
RISK_PCT: float = 0.01       # 1 % equity at risk per trade (classic Turtle unit)
STOP_MULT: float = 2.0       # stop placed 2 ATR below entry
MAX_LEVERAGE: float = 1.5    # hard cap on position size (Audit lowered from 4.0 to 1.5)
FEES: float = 0.001          # ~10 bps round-trip (equity_lc cost profile)
SLIPPAGE: float = 0.0002     # ~2 bps slippage

# (entry_channel_period, exit_channel_period)
SYSTEMS: dict[int, tuple[int, int]] = {
    1: (20, 10),
    2: (55, 20),
}

ATR_PERIOD: int = 20

# Bars per year by timeframe (used for Sharpe annualisation + RoR horizon)
BARS_PER_YEAR: dict[str, int] = {
    "D": 252,
    "H1": 252 * 14,   # ~14 hourly bars/day incl. extended hours
    "M5": 252 * 78,   # 78 five-min bars per regular session (9:30-16:00)
}

# VectorBT freq string by timeframe
VBT_FREQ: dict[str, str] = {
    "D": "d",
    "H1": "h",
    "M5": "5min",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_databento(
    instrument: str,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load 1-year 5-min OHLCV from databento CSV.

    Expected path: data/databento/<INSTRUMENT>_1yr_5m.csv
    Columns: ts_event, Open, High, Low, Close, Volume

    Args:
        instrument: Ticker symbol (e.g. "AMAT", "NVDA").
        data_dir:   Override repo data/ directory.

    Returns:
        DataFrame with lowercase OHLCV columns and UTC DatetimeIndex.
    """
    base = data_dir if data_dir else ROOT / "data"
    path = base / "databento" / f"{instrument}_1yr_5m.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Databento 5m file not found: {path}\n"
            f"Available tickers: ls {base / 'databento'}"
        )
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts_col = "ts_event" if "ts_event" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    df = df[["open", "high", "low", "close"]].dropna()
    print(
        f"  Databento 5m: {len(df)} bars  "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )
    return df


def load_data(
    instrument: str,
    timeframe: str,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Unified loader: parquet for daily, databento CSV for 5-min."""
    if timeframe == "M5":
        return _load_databento(instrument, data_dir)
    return _load_ohlcv(instrument, timeframe, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Signal & sizing helpers
# ---------------------------------------------------------------------------

def compute_signals(
    df: pd.DataFrame,
    entry_p: int,
    exit_p: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Turtle long and short entry/exit signals + fractional ATR stop.

    All channel levels are shifted by 1 bar to prevent lookahead bias.

    Long  entry: close breaks ABOVE prev N-bar high
    Long  exit:  close drops  BELOW prev M-bar low
    Short entry: close breaks BELOW prev N-bar low
    Short exit:  close rises  ABOVE prev M-bar high

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.
        entry_p: Entry channel lookback (e.g. 20 or 55).
        exit_p:  Exit channel lookback  (e.g. 10 or 20).

    Returns:
        long_entries, long_exits, short_entries, short_exits, stop_pct,
        hi_entry (shifted), atr_shifted
        (hi_entry and atr_shifted are exposed for pyramid level computation)
    """
    close = df["close"]

    hi_entry = df["high"].rolling(entry_p).max().shift(1)
    lo_entry = df["low"].rolling(entry_p).min().shift(1)
    hi_exit = df["high"].rolling(exit_p).max().shift(1)
    lo_exit = df["low"].rolling(exit_p).min().shift(1)

    long_entries: pd.Series = close > hi_entry
    long_exits: pd.Series = close < lo_exit
    short_entries: pd.Series = close < lo_entry
    short_exits: pd.Series = close > hi_exit

    atr_series = atr(df, p=ATR_PERIOD).shift(1)
    safe_close = close.where(close > 0, np.nan)
    stop_pct = (STOP_MULT * atr_series / safe_close).fillna(0.0).clip(upper=1.0)

    return long_entries, long_exits, short_entries, short_exits, stop_pct, hi_entry, atr_series


def build_size_pct(
    close: pd.Series,
    atr_series: pd.Series,
) -> pd.Series:
    """ATR-based position size as fraction of portfolio equity.

    unit_size = risk_pct / stop_fraction
    where stop_fraction = STOP_MULT * ATR / close

    Capped at MAX_LEVERAGE to prevent over-sizing on low-vol bars.
    """
    safe_close = close.where(close > 0, np.nan)
    stop_frac = (STOP_MULT * atr_series / safe_close).replace(0.0, np.nan)
    size_pct = (RISK_PCT / stop_frac).clip(upper=MAX_LEVERAGE).fillna(0.0)
    return size_pct


def compute_pyramid_entries(
    close: pd.Series,
    hi_entry: pd.Series,
    atr_series: pd.Series,
    max_units: int = 4,
    pyramid_atr_mult: float = 0.5,
) -> pd.Series:
    """Combine base Donchian entry with up to (max_units-1) pyramid levels.

    Each pyramid level k fires when close exceeds the initial breakout level
    by k * pyramid_atr_mult * ATR.  Used with accumulate='addonly' in VBT so
    each level adds one extra unit to an open position.

    Level 0 (base):  close > hi_entry                      (initial breakout)
    Level k:         close > hi_entry + k * mult * ATR_20

    All levels use the same pre-shifted hi_entry and atr_series (no new
    lookahead is introduced here).
    """
    combined = close > hi_entry  # level 0 already bool
    for k in range(1, max_units):
        level_k = hi_entry + k * pyramid_atr_mult * atr_series
        combined = combined | (close > level_k)
    return combined


# ---------------------------------------------------------------------------
# VectorBT runner
# ---------------------------------------------------------------------------

def _shift_sig(s: pd.Series) -> np.ndarray:
    """Shift boolean signal by 1 bar (fill at next bar, no lookahead)."""
    return np.r_[False, s.values[:-1]]


def _run_vbt(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    stop_pct: pd.Series,
    size_pct: pd.Series,
    freq: str = "d",
    direction: str = "longonly",
    cash: float = INIT_CASH,
    sl_trail: bool = False,
    accumulate: bool = False,
) -> vbt.Portfolio:
    """Run a single-direction VectorBT backtest.

    direction:   "longonly" or "shortonly"
    sl_trail:    If True, ATR stop trails the high-water mark (trailing stop).
    accumulate:  If True, uses accumulate='addonly' to allow pyramid entries.
    Signals shifted by 1 bar: signal at today's close fills at next bar.
    """
    kwargs: dict = dict(
        entries=_shift_sig(entries),
        exits=_shift_sig(exits),
        direction=direction,
        sl_stop=stop_pct.reindex(close.index).fillna(0.0).values,
        sl_trail=sl_trail,
        size=size_pct.reindex(close.index).fillna(0.0).values,
        size_type="percent",
        init_cash=cash,
        fees=FEES,
        slippage=SLIPPAGE,
        freq=freq,
    )
    if accumulate:
        kwargs["accumulate"] = "addonly"
    return vbt.Portfolio.from_signals(close, **kwargs)


def _combine_portfolios(
    pf_a: vbt.Portfolio,
    pf_b: vbt.Portfolio,
) -> pd.Series:
    """Combine two equal-capital portfolios into a single daily returns series."""
    val_a = pf_a.value()
    val_b = pf_b.value()
    combined = val_a.reindex(val_a.index.union(val_b.index)).ffill()
    combined += val_b.reindex(combined.index).ffill()
    return combined.pct_change().fillna(0.0)


def _stats_from_returns(
    rets: pd.Series,
    timeframe: str = "D",
    freq: str = "d",
    n_long: int = 0,
    n_short: int = 0,
) -> dict:
    """Compute summary stats directly from a daily returns series.

    Used for combined long+short portfolios where no single VBT object exists.
    """
    equity = (1 + rets).cumprod()
    peak = equity.expanding().max()
    dd_series = (equity - peak) / peak
    max_dd = float(dd_series.min())

    bars_per_year = BARS_PER_YEAR.get(timeframe, 252)
    sharpe = (
        float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
        if rets.std() > 0
        else 0.0
    )
    total_ret = float(equity.iloc[-1] - 1)
    n_bars = len(rets)
    ann_ret = float((1 + total_ret) ** (bars_per_year / n_bars) - 1) if n_bars > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd != 0.0 else float("nan")

    # Sortino
    down = rets[rets < 0].values
    d_std = np.std(down) * np.sqrt(bars_per_year) if len(down) > 1 else np.nan
    sortino = (float(rets.mean()) * bars_per_year) / d_std if d_std and d_std > 0 else float("nan")

    # Max DD duration in bars — vectorised run-length encoding
    in_dd = (dd_series < 0).values
    padded = np.concatenate([[False], in_dd, [False]]).astype(np.int8)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    max_dd_days = int((ends - starts).max()) if len(starts) else 0

    # Period returns
    period_rets: dict[str, float] = {}
    if timeframe == "M5":
        for (yr, mo), grp in rets.groupby([rets.index.year, rets.index.month]):
            period_rets[f"{int(yr)}-{int(mo):02d}"] = float(
                (1 + grp).prod() - 1
            )
    elif hasattr(rets.index, "year"):
        for yr, grp in rets.groupby(rets.index.year):
            period_rets[str(int(yr))] = float((1 + grp).prod() - 1)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "ann_ret": ann_ret,
        "dd": max_dd,
        "ret": total_ret,
        "trades": n_long + n_short,
        "n_long": n_long,
        "n_short": n_short,
        "wr": 0.0,
        "annual_rets": period_rets,
        "max_dd_days": max_dd_days,
        "avg_dd": float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "payoff": float("nan"),
        "avg_dur_bars": float("nan"),
        "median_dur_bars": float("nan"),
        "time_deployed": float("nan"),
        "daily_returns": rets,
    }


def _stats(pf: vbt.Portfolio, timeframe: str = "D") -> dict:
    """Extract full metrics from a VBT Portfolio (single direction)."""
    n_trades = int(pf.trades.count())
    bars_per_year = BARS_PER_YEAR.get(timeframe, 252)

    # Period returns
    period_rets: dict[str, float] = {}
    bar_r: pd.Series = pf.returns()
    try:
        if timeframe == "M5":
            for (yr, mo), grp in bar_r.groupby(
                [bar_r.index.year, bar_r.index.month]
            ):
                period_rets[f"{int(yr)}-{int(mo):02d}"] = float((1 + grp).prod() - 1)
        elif hasattr(bar_r.index, "year"):
            for yr, grp in bar_r.groupby(bar_r.index.year):
                period_rets[str(int(yr))] = float((1 + grp).prod() - 1)
    except Exception:
        pass

    # Drawdown stats
    max_dd_days: int = 0
    avg_dd: float = 0.0
    try:
        dds = pf.drawdowns
        durations = dds.duration.values
        if len(durations):
            max_dd_days = int(np.nanmax(durations))
            avg_dd = float(np.nanmean(dds.drawdown.values))
    except Exception:
        pass

    # Trade-level stats
    avg_win = avg_loss = payoff = 0.0
    avg_dur = median_dur = float("nan")
    wr = 0.0
    try:
        if n_trades > 0:
            tr = pf.trades.returns.values
            wins = tr[tr > 0]
            losses = tr[tr <= 0]
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0
            payoff = abs(avg_win / avg_loss) if avg_loss != 0.0 else float("nan")
            wr = float(pf.trades.win_rate())
            dur = pf.trades.duration.values.astype(float)
            avg_dur = float(np.mean(dur))
            median_dur = float(np.median(dur))
    except Exception:
        pass

    # % time capital deployed (fraction of bars with open position)
    time_deployed: float = float("nan")
    try:
        deployed = (pf.asset_value() > 0).mean()
        time_deployed = float(deployed)
    except Exception:
        pass

    # Risk-adjusted ratios
    ann_ret = 0.0
    calmar = sortino = float("nan")
    try:
        ann_ret = float(pf.annualized_return())
        max_dd_val = abs(float(pf.max_drawdown()))
        calmar = ann_ret / max_dd_val if max_dd_val > 0 else float("nan")
        rets_arr = bar_r.dropna().values
        down = rets_arr[rets_arr < 0]
        d_std = np.std(down) * np.sqrt(bars_per_year) if len(down) > 1 else np.nan
        sortino = (float(np.mean(rets_arr)) * bars_per_year) / d_std if d_std and d_std > 0 else float("nan")
    except Exception:
        pass

    return {
        "sharpe": float(pf.sharpe_ratio()),
        "sortino": sortino,
        "calmar": calmar,
        "ann_ret": ann_ret,
        "dd": float(pf.max_drawdown()),
        "ret": float(pf.total_return()),
        "trades": n_trades,
        "n_long": n_trades,   # single-direction pf: all trades are this direction
        "n_short": 0,
        "wr": wr,
        "payoff": payoff,
        "avg_dur_bars": avg_dur,
        "median_dur_bars": median_dur,
        "time_deployed": time_deployed,
        "annual_rets": period_rets,
        "max_dd_days": max_dd_days,
        "avg_dd": avg_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "daily_returns": bar_r,
    }


def _risk_of_ruin(
    daily_returns: pd.Series,
    ruin_pct: float = 0.50,
    horizon_years: int = 5,
    n_sims: int = 5000,
) -> float:
    """Bootstrap daily returns to estimate P(drawdown >= ruin_pct) over horizon.

    Fully vectorised: generates all (n_sims x horizon_bars) paths in one numpy
    call — ~20-50x faster than the equivalent Python for-loop.

    Args:
        daily_returns: Daily strategy returns series (fractions, e.g. 0.01 = 1%).
        ruin_pct:      Drawdown threshold to call "ruin" (default 0.50 = -50%).
        horizon_years: Simulation horizon in years (default 5).
        n_sims:        Number of bootstrap paths (default 5000).

    Returns:
        Probability in [0, 1].  NaN if insufficient data.
    """
    rets = daily_returns.dropna().values
    if len(rets) < 20:
        return float("nan")

    rng = np.random.default_rng(42)
    horizon_bars = horizon_years * 252
    # Shape: (n_sims, horizon_bars) — all paths in one allocation
    sampled = rng.choice(rets, size=(n_sims, horizon_bars), replace=True)
    equity = np.cumprod(1.0 + sampled, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    ruin = np.any((equity - peak) / peak <= -ruin_pct, axis=1)
    return float(ruin.mean())


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_turtle_system(
    df: pd.DataFrame,
    system_num: int,
    timeframe: str = "D",
    direction: str = "long",
    pyramid: bool = False,
    trailing_stop: bool = False,
    max_units: int = 4,
    pyramid_atr_mult: float = 0.5,
) -> dict:
    """Run IS + OOS backtest for one Turtle system.

    Args:
        direction:        "long" (long-only) or "both" (long + short).
        pyramid:          If True, add up to max_units pyramid entries at
                          +k * pyramid_atr_mult * ATR above Donchian high.
        trailing_stop:    If True, ATR stop trails the position high-water mark.
        max_units:        Maximum pyramid units (incl. initial entry). Default 4.
        pyramid_atr_mult: ATR multiple between pyramid levels. Default 0.5.
    """
    entry_p, exit_p = SYSTEMS[system_num]
    freq = VBT_FREQ.get(timeframe, "d")
    bars_per_year = BARS_PER_YEAR.get(timeframe, 252)

    long_en, long_ex, short_en, short_ex, stop_pct, hi_entry, atr_shifted = (
        compute_signals(df, entry_p, exit_p)
    )
    size_pct = build_size_pct(df["close"], atr_shifted)

    # Pyramid: combine base entry with upper levels
    if pyramid:
        long_en = compute_pyramid_entries(
            df["close"], hi_entry, atr_shifted,
            max_units=max_units, pyramid_atr_mult=pyramid_atr_mult,
        )

    # IS/OOS split (70/30 chronological)
    n = len(df)
    is_n = int(n * IS_RATIO)
    is_idx = df.index[:is_n]
    oos_idx = df.index[is_n:]

    def _run_slice(idx: pd.Index) -> dict:
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
            combined_rets = _combine_portfolios(pf_long, pf_short)
            return _stats_from_returns(
                combined_rets, timeframe=timeframe,
                n_long=int(pf_long.trades.count()),
                n_short=int(pf_short.trades.count()),
            )
        s = _stats(pf_long, timeframe=timeframe)
        # for long-only direction label the trades correctly
        s["n_long"] = s["trades"]
        s["n_short"] = 0
        return s

    is_stats = _run_slice(is_idx)
    oos_stats = _run_slice(oos_idx)

    # OOS/IS Sharpe ratio (parity gate: >= 0.5 to pass)
    is_sh = is_stats["sharpe"]
    oos_sh = oos_stats["sharpe"]
    if is_sh != 0.0:
        parity = oos_sh / is_sh
    else:
        parity = float("nan")

    oos_years = len(oos_idx) / bars_per_year

    return {
        "entry_p": entry_p,
        "exit_p": exit_p,
        "is_bars": len(is_idx),
        "oos_bars": len(oos_idx),
        "oos_years": oos_years,
        "is_stats": is_stats,
        "oos_stats": oos_stats,
        "parity": parity,
        "timeframe": timeframe,
        "direction": direction,
        "pyramid": pyramid,
        "trailing_stop": trailing_stop,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v:>+10.1%}"


def _fv(v: float, fmt: str = "+.3f") -> str:
    """Format float, returning 'N/A' for nan/inf."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "       N/A"
    return f"{v:{fmt}}"


def _fvp(v: float) -> str:
    """Format float as percentage."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "       N/A"
    return f"{v:>+10.1%}"


def _bars_to_hms(bars: float, timeframe: str) -> str:
    """Convert bar count to human-readable duration."""
    if np.isnan(bars):
        return "N/A"
    h = int(bars)
    if timeframe == "H1":
        d, r = divmod(h, 14)
        return f"{d}d {r}h" if d > 0 else f"{r}h"
    return f"{h} bars"


def print_results(instrument: str, results: dict[int, dict]) -> None:
    """Print comprehensive IS/OOS results: returns, risk, trade analytics, capital deployment."""
    sep = "=" * 76
    thin = "-" * 76

    first = next(iter(results.values()))
    direction = first.get("direction", "long")
    dir_label = "long+short" if direction == "both" else "long-only"
    pyr_label = f"pyramid x{first.get('max_units', 4)}" if first.get("pyramid") else "no pyramid"
    sl_label = "trailing SL" if first.get("trailing_stop") else "fixed SL"
    tf = first.get("timeframe", "D")
    print(f"\n{sep}")
    print(
        f"  TURTLE TRADING POC  --  {instrument}  "
        f"[{tf} | {dir_label} | {pyr_label} | {sl_label}]"
    )
    systems_str = ", ".join(
        f"S{k}({v['entry_p']}/{v['exit_p']})" for k, v in results.items()
    )
    print(f"  Systems: {systems_str}   IS/OOS: 70/30   Risk/trade: 1%   Cash: $100k")
    print(sep)

    for sys_num, r in results.items():
        entry_p = r["entry_p"]
        exit_p  = r["exit_p"]
        is_s    = r["is_stats"]
        oos_s   = r["oos_stats"]
        parity  = r["parity"]
        gate    = "PASS" if not np.isnan(parity) and parity >= 0.5 else "FAIL"
        parity_str = f"{parity:+.2f}" if not np.isnan(parity) else "N/A"

        W = 30  # label width
        print(f"\n  System {sys_num}  ({entry_p}-bar entry / {exit_p}-bar exit)")
        print(thin)
        print(f"  {'Metric':<{W}} {'IS':>10}  {'OOS':>10}")
        print(thin)

        # --- Returns ---
        print(f"  {'Period (bars)':<{W}} {r['is_bars']:>10}  {r['oos_bars']:>10}  ({r['oos_years']:.1f} yr)")
        print(f"  {'Total Return':<{W}} {_fvp(is_s['ret'])}  {_fvp(oos_s['ret'])}")
        print(f"  {'Annualised Return':<{W}} {_fvp(is_s.get('ann_ret'))}  {_fvp(oos_s.get('ann_ret'))}")

        # --- Risk-adjusted ---
        print(f"  {thin}")
        print(f"  {'Sharpe Ratio':<{W}} {_fv(is_s['sharpe']):>10}  {_fv(oos_s['sharpe']):>10}")
        print(f"  {'Sortino Ratio':<{W}} {_fv(is_s.get('sortino')):>10}  {_fv(oos_s.get('sortino')):>10}")
        print(f"  {'Calmar Ratio':<{W}} {_fv(is_s.get('calmar')):>10}  {_fv(oos_s.get('calmar')):>10}")

        # --- Drawdown ---
        print(f"  {thin}")
        print(f"  {'Max Drawdown':<{W}} {_fvp(is_s['dd'])}  {_fvp(oos_s['dd'])}")
        print(f"  {'Avg Drawdown':<{W}} {_fvp(is_s['avg_dd'])}  {_fvp(oos_s['avg_dd'])}")
        print(f"  {'Max DD Duration (bars)':<{W}} {is_s['max_dd_days']:>10}  {oos_s['max_dd_days']:>10}")

        # --- Capital deployment ---
        print(f"  {thin}")
        td_is  = is_s.get('time_deployed')
        td_oos = oos_s.get('time_deployed')
        td_is_s  = f"{td_is:>9.1%}"  if td_is  is not None and not np.isnan(td_is)  else "       N/A"
        td_oos_s = f"{td_oos:>9.1%}" if td_oos is not None and not np.isnan(td_oos) else "       N/A"
        print(f"  {'Capital Deployed':<{W}} {td_is_s}  {td_oos_s}")

        # --- Trade counts ---
        print(f"  {thin}")
        print(f"  {'Trades (total)':<{W}} {is_s['trades']:>10}  {oos_s['trades']:>10}")
        print(f"  {'  Long trades':<{W}} {is_s.get('n_long', 0):>10}  {oos_s.get('n_long', 0):>10}")
        print(f"  {'  Short trades':<{W}} {is_s.get('n_short', 0):>10}  {oos_s.get('n_short', 0):>10}")

        # --- Win/loss structure ---
        print(f"  {thin}")
        print(f"  {'Win Rate':<{W}} {_fvp(is_s['wr'])}  {_fvp(oos_s['wr'])}")
        print(f"  {'Avg Win':<{W}} {_fvp(is_s['avg_win'])}  {_fvp(oos_s['avg_win'])}")
        print(f"  {'Avg Loss':<{W}} {_fvp(is_s['avg_loss'])}  {_fvp(oos_s['avg_loss'])}")
        print(f"  {'Payoff Ratio (|win/loss|)':<{W}} {_fv(is_s.get('payoff'), '.2f'):>10}  {_fv(oos_s.get('payoff'), '.2f'):>10}")

        # --- Trade duration ---
        print(f"  {thin}")
        avg_is  = is_s.get('avg_dur_bars',  float('nan'))
        avg_oos = oos_s.get('avg_dur_bars', float('nan'))
        med_is  = is_s.get('median_dur_bars',  float('nan'))
        med_oos = oos_s.get('median_dur_bars', float('nan'))
        print(
            f"  {'Avg Trade Duration':<{W}} "
            f"{_bars_to_hms(avg_is, tf):>10}  {_bars_to_hms(avg_oos, tf):>10}"
        )
        print(
            f"  {'Median Trade Duration':<{W}} "
            f"{_bars_to_hms(med_is, tf):>10}  {_bars_to_hms(med_oos, tf):>10}"
        )

        print(f"  {thin}")
        print(f"  OOS/IS Sharpe ratio: {parity_str}   Gate (>= 0.50): {gate}")

        # Annual / monthly returns
        all_periods: dict[str, float] = {
            **is_s["annual_rets"],
            **oos_s["annual_rets"],
        }
        if all_periods:
            period_label = "Monthly Returns:" if tf == "M5" else "Annual Returns:"
            print(f"\n  {period_label}")
            col_w = 10 if tf == "M5" else 8
            hdr = f"{'Period':<{col_w}} {'Return':>10}"
            print(f"  {hdr}   {hdr}")
            keys = sorted(all_periods)
            pairs = [
                (keys[i], keys[i + 1]) if i + 1 < len(keys) else (keys[i], None)
                for i in range(0, len(keys), 2)
            ]
            for k1, k2 in pairs:
                r1 = f"{all_periods[k1]:>+9.1%}"
                r2 = (
                    f"  {k2:<{col_w}} {all_periods[k2]:>+9.1%}"
                    if k2 is not None else ""
                )
                print(f"  {k1:<{col_w}} {r1}{r2}")

        # Risk of ruin
        ror_horizon = 1 if tf == "M5" else 5
        ror_lbl = "1-yr" if tf == "M5" else "5-yr"
        ror = _risk_of_ruin(oos_s["daily_returns"], horizon_years=ror_horizon)
        ror_str = f"{ror:.1%}" if not np.isnan(ror) else "N/A"
        print(f"\n  Risk of Ruin (OOS, -50% DD, {ror_lbl} horizon, 5k sims): {ror_str}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turtle Trading System POC — Systems 1 & 2"
    )
    parser.add_argument(
        "--instrument",
        default="AMAT",
        help="Instrument ticker matching a data/<TICKER>_D.parquet file (default: AMAT)",
    )
    parser.add_argument(
        "--system",
        default="both",
        choices=["1", "2", "both"],
        help="System to run: 1, 2, or both (default: both)",
    )
    parser.add_argument(
        "--timeframe",
        default="D",
        choices=["D", "H1", "M5"],
        help="Bar timeframe: D=daily (default), H1=hourly, M5=5-min (databento)",
    )
    parser.add_argument(
        "--direction",
        default="long",
        choices=["long", "both"],
        help="Trade direction: long (default) or both (long+short)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Filter data from this date onward, e.g. 2013-01-01",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory path (default: <repo_root>/data/)",
    )
    parser.add_argument(
        "--pyramid",
        action="store_true",
        help="Enable pyramiding: add units at +0.5/1.0/1.5 ATR above breakout (max 4 units)",
    )
    parser.add_argument(
        "--trailing-stop",
        action="store_true",
        help="Use trailing ATR stop (trails high-water mark) instead of fixed stop",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    print(f"Loading {args.timeframe} data for {args.instrument} ...")
    df = load_data(args.instrument, args.timeframe, data_dir=data_dir)

    if args.start:
        cutoff = pd.Timestamp(args.start, tz="UTC")
        df = df.loc[df.index >= cutoff]
        print(f"  Filtered to {args.start} onwards: {len(df)} bars")

    if args.timeframe != "M5":
        print(
            f"  {len(df)} bars  |  {df.index[0].date()} to {df.index[-1].date()}"
        )

    systems_to_run: list[int]
    if args.system == "both":
        systems_to_run = [1, 2]
    else:
        systems_to_run = [int(args.system)]

    results: dict[int, dict] = {}
    for sys_num in systems_to_run:
        print(f"Running System {sys_num} ...")
        results[sys_num] = run_turtle_system(
            df, sys_num, timeframe=args.timeframe, direction=args.direction,
            pyramid=args.pyramid, trailing_stop=args.trailing_stop,
        )

    print_results(args.instrument, results)


if __name__ == "__main__":
    main()
