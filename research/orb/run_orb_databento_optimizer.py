import os
import sys
import itertools
import warnings
import time as _time
from datetime import timedelta, time as dt_time

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from dotenv import load_dotenv

# Add project root to path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from titan.indicators.gaussian_filter import _gaussian_channel_kernel

warnings.filterwarnings("ignore")
load_dotenv()

FEES = 0.0001
SLIPPAGE = 0.0005
ROUND_TRIP_COST = 2 * (FEES + SLIPPAGE)

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load cached data, converting CSV → parquet on first access for speed."""
    cache_dir = os.path.join("data", "databento")
    parquet_file = os.path.join(cache_dir, f"{ticker}_1yr_5m.parquet")
    csv_file = os.path.join(cache_dir, f"{ticker}_1yr_5m.csv")

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        return df

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        df.to_parquet(parquet_file)
        return df

    return pd.DataFrame()

# ── Indicators ────────────────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def normalize(value, min_val=-2.0, max_val=3.0):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

# ── Pre-computation ──────────────────────────────────────────────────────────

def precompute_daily_arrays(df_5m, orb_end_str, trading_start_str, cutoff_str):
    """Pre-compute daily numpy arrays and ORB levels for a (window, cutoff) combo.
    Called once per combo (4 total), not once per config (288)."""
    cutoff_time = dt_time(*map(int, cutoff_str.split(":")))
    days = []

    for date, group in df_5m.groupby(df_5m.index.date):
        if len(group) < 4:
            continue
        orb_bars = group.between_time("09:30", orb_end_str)
        if len(orb_bars) == 0:
            continue
        trading = group.between_time(trading_start_str, "15:55")
        if len(trading) == 0:
            continue

        times = trading.index.time
        cutoff_mask = np.array([t <= cutoff_time for t in times])
        if not cutoff_mask.any():
            continue
        cutoff_idx = int(np.max(np.where(cutoff_mask)))

        first_row = trading.iloc[0]
        sma = float(first_row["Daily_SMA50"]) if not pd.isna(first_row["Daily_SMA50"]) else 0.0
        rsi = float(first_row["Daily_RSI14"]) if not pd.isna(first_row["Daily_RSI14"]) else 50.0

        days.append({
            'close': trading["Close"].values.astype(np.float64),
            'high':  trading["High"].values.astype(np.float64),
            'low':   trading["Low"].values.astype(np.float64),
            'atr':   trading["ATR"].values.astype(np.float64),
            'gauss_mid': trading["Gauss_Mid"].values.astype(np.float64),
            'or_high': float(orb_bars["High"].max()),
            'or_low':  float(orb_bars["Low"].min()),
            'sma': sma,
            'rsi': rsi,
            'cutoff_idx': cutoff_idx,
            'first_bar_pos': df_5m.index.get_loc(trading.index[0]),
        })
    return days

# ── Fast Simulation ──────────────────────────────────────────────────────────

def simulate_config(daily_data, atr_mult, rr_ratio, use_sma, use_rsi, use_gauss, split_idx):
    """Simulate a single config across all days using numpy for exit detection."""
    is_rets = []
    oos_rets = []

    for day in daily_data:
        close = day['close']
        high  = day['high']
        low   = day['low']
        atr   = day['atr']
        g_mid = day['gauss_mid']
        or_h  = day['or_high']
        or_l  = day['or_low']
        sma   = day['sma']
        rsi   = day['rsi']
        co    = day['cutoff_idx']
        n     = len(close)
        is_oos = day['first_bar_pos'] >= split_idx

        bull_rsi = (rsi < 70) if use_rsi else True
        bear_rsi = (rsi > 30) if use_rsi else True

        i = 0
        while i <= co:
            c = close[i]
            a = atr[i] if not np.isnan(atr[i]) else (or_h - or_l)
            gm = g_mid[i]
            
            if a <= 0:
                i += 1
                continue

            bull_sma = (c > sma) if (use_sma and sma != 0.0) else True
            bear_sma = (c < sma) if (use_sma and sma != 0.0) else True
            
            bull_gauss = (c > gm) if (use_gauss and not np.isnan(gm)) else True
            bear_gauss = (c < gm) if (use_gauss and not np.isnan(gm)) else True

            direction = 0
            if c > or_h and bull_sma and bull_rsi and bull_gauss:
                direction = 1
            elif c < or_l and bear_sma and bear_rsi and bear_gauss:
                direction = -1
            else:
                i += 1
                continue

            # Compute SL / TP
            risk = a * atr_mult
            if direction == 1:
                sl, tp = c - risk, c + risk * rr_ratio
            else:
                sl, tp = c + risk, c - risk * rr_ratio

            # ── numpy exit detection ──
            exit_price = close[n - 1]  # default: EOD close
            exit_bar = n - 1

            if i + 1 < n:
                rem_h = high[i+1:]
                rem_l = low[i+1:]
                if direction == 1:
                    hit = (rem_l <= sl) | (rem_h >= tp)
                else:
                    hit = (rem_h >= sl) | (rem_l <= tp)

                if hit.any():
                    offset = int(np.argmax(hit))
                    exit_bar = i + 1 + offset
                    # Determine whether SL or TP was hit
                    if direction == 1:
                        exit_price = sl if low[exit_bar] <= sl else tp
                    else:
                        exit_price = sl if high[exit_bar] >= sl else tp

            # Trade return
            if direction == 1:
                ret = (exit_price - c) / c - ROUND_TRIP_COST
            else:
                ret = (c - exit_price) / c - ROUND_TRIP_COST

            (oos_rets if is_oos else is_rets).append(ret)
            i = exit_bar + 1

    return _score(is_rets, oos_rets)

def _score(is_rets, oos_rets):
    """Compute composite consistency score from trade-return lists."""
    is_arr  = np.array(is_rets)  if is_rets  else np.empty(0)
    oos_arr = np.array(oos_rets) if oos_rets else np.empty(0)
    is_n, oos_n = len(is_arr), len(oos_arr)

    def _sharpe(arr):
        if len(arr) < 2:
            return 0.0
        s = np.std(arr, ddof=1)
        return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0

    def _wr(arr):
        return float(np.sum(arr > 0) / len(arr) * 100) if len(arr) > 0 else 0.0

    def _ret(arr):
        return float((np.prod(1 + arr) - 1) * 100) if len(arr) > 0 else 0.0

    is_sharpe  = _sharpe(is_arr)
    oos_sharpe = _sharpe(oos_arr)
    is_wr      = _wr(is_arr)
    oos_wr     = _wr(oos_arr)
    is_ret     = _ret(is_arr)
    oos_ret    = _ret(oos_arr)

    # Early exit: skip configs with no IS edge
    if is_n < 10 or is_sharpe <= 0:
        return None

    retention = min(1.0, oos_sharpe / is_sharpe) if oos_sharpe > 0 else 0.0

    score = (
        0.40 * normalize(oos_sharpe)
      + 0.25 * normalize(is_sharpe)
      + 0.15 * (oos_wr / 100.0)
      + 0.10 * (min(oos_n, 30) / 30.0)
      + 0.10 * retention
    )

    return {
        'is_return': is_ret, 'is_sharpe': is_sharpe,
        'is_trades': is_n,   'is_win_rate': is_wr,
        'oos_return': oos_ret, 'oos_sharpe': oos_sharpe,
        'oos_trades': oos_n,   'oos_win_rate': oos_wr,
        'score': score,
    }

# ── Per-Ticker Grid Search ──────────────────────────────────────────────────

def run_grid_search_for_ticker(ticker: str) -> list[dict]:
    t0 = _time.perf_counter()
    print(f"[{ticker}] Loading data...")
    df_5m = load_ticker_data(ticker)
    if df_5m.empty:
        return []

    # Daily indicators via yfinance
    df_1d = yf.download(ticker,
                        start=df_5m.index.min() - timedelta(days=100),
                        end=df_5m.index.max(), interval="1d", progress=False)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.droplevel(1)
    df_1d.index = df_1d.index.tz_localize(None)

    df_5m["ATR"] = calculate_atr(df_5m, 14)
    
    # 5M Gaussian Channel (144 period, 4 poles, 2.0 sigma are defaults for 5m TF)
    high_arr = df_5m["High"].values
    low_arr = df_5m["Low"].values
    close_arr = df_5m["Close"].values
    _, _, g_mid = _gaussian_channel_kernel(high_arr, low_arr, close_arr, 144.0, 4, 2.0)
    df_5m["Gauss_Mid"] = g_mid
    
    df_1d["SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
    df_1d["RSI14"] = calculate_rsi(df_1d["Close"], 14).shift(1)

    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
    sma_map = df_1d.set_index("Date_Str")["SMA50"].to_dict()
    rsi_map = df_1d.set_index("Date_Str")["RSI14"].to_dict()
    df_5m["Daily_SMA50"] = df_5m["Date_Str"].map(sma_map)
    df_5m["Daily_RSI14"] = df_5m["Date_Str"].map(rsi_map)

    split_idx = int(len(df_5m) * 0.70)

    # Parameter grid
    atr_mults    = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    rr_ratios    = [1.5, 2.0, 3.0]
    sma_filters  = [True, False]
    rsi_filters  = [True, False]
    gauss_filters = [True, False]
    orb_windows  = [("09:40", "09:45"), ("09:45", "09:50")]
    entry_cutoffs = ["15:55", "11:00"]

    # Pre-compute daily arrays for each (window, cutoff) combo — 4 total
    precomputed = {}
    for (orb_end, trad_start), cutoff in itertools.product(orb_windows, entry_cutoffs):
        key = (orb_end, cutoff)
        precomputed[key] = precompute_daily_arrays(df_5m, orb_end, trad_start, cutoff)

    # Run 576 configs using pre-computed daily data
    results = []
    for atr_m, rr, use_sma, use_rsi, use_gauss, (orb_end, _), cutoff in itertools.product(
            atr_mults, rr_ratios, sma_filters, rsi_filters, gauss_filters, orb_windows, entry_cutoffs):

        daily_data = precomputed[(orb_end, cutoff)]
        metrics = simulate_config(daily_data, atr_m, rr, use_sma, use_rsi, use_gauss, split_idx)
        if metrics is None:
            continue

        metrics.update({
            'ticker': ticker, 'atr': atr_m, 'rr': rr,
            'sma': use_sma, 'rsi': use_rsi, 'gauss': use_gauss,
            'orb': orb_end, 'cutoff': cutoff,
        })
        results.append(metrics)

    elapsed = _time.perf_counter() - t0
    print(f"[{ticker}] Done — {len(results)} profitable configs in {elapsed:.1f}s")
    return results

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cache_dir = os.path.join("data", "databento")
    csv_files = sorted(
        f for f in os.listdir(cache_dir)
        if f.endswith("_1yr_5m.csv") or f.endswith("_1yr_5m.parquet")
    )
    # Deduplicate tickers (prefer parquet)
    tickers = sorted({f.split("_1yr_5m")[0] for f in csv_files})

    if not tickers:
        print("No cached Databento data found in data/databento/. Exiting.")
        exit(1)

    print(f"\n{'='*80}")
    print(f"  ORB + Gaussian Optimizer — {len(tickers)} tickers × 576 configs")
    print(f"  Total backtests: {576 * len(tickers):,}")
    print(f"{'='*80}\n")

    t_start = _time.perf_counter()
    all_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_grid_search_for_ticker)(t) for t in tickers
    )
    flat = [r for sub in all_results for r in sub]
    total_time = _time.perf_counter() - t_start

    if not flat:
        print("No profitable configurations found.")
        exit(0)

    df = pd.DataFrame(flat).sort_values("score", ascending=False)
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, "optimization_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nCompleted in {total_time:.0f}s — saved {len(df)} rows to {csv_path}\n")

    # ── Best per ticker ──────────────────────────────────────────────────────
    print("=" * 110)
    print("🏆 BEST CONFIGURATION PER TICKER (Ranked by Consistency Score)")
    print("=" * 110)
    hdr = f"{'Ticker':<7} | {'Score':<5} | {'IS Shrp':<7} | {'OOS Shrp':<8} | {'IS WR%':<6} | {'OOS WR%':<7} | {'IS Ret%':<7} | {'OOS Ret%':<8} | Config"
    print(hdr)
    print("-" * 110)

    best = df.sort_values(["ticker","score"], ascending=[True,False]).groupby("ticker").head(1)
    best = best.sort_values("score", ascending=False)
    for _, r in best.iterrows():
        cfg = f"ATR:{r['atr']} RR:{r['rr']} S:{str(r['sma'])[:1]} G:{str(r['gauss'])[:1]} R:{str(r['rsi'])[:1]} ORB:{r['orb'][3:]} Cut:{r['cutoff'][:2]}"
        print(f"{r['ticker']:<7} | {r['score']:<5.3f} | {r['is_sharpe']:<7.2f} | {r['oos_sharpe']:<8.2f} | {r['is_win_rate']:>5.1f}% | {r['oos_win_rate']:>5.1f}%  | {r['is_return']:>6.1f}% | {r['oos_return']:>7.1f}% | {cfg}")

    # ── Top 25 overall ───────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("🏆 TOP 25 OVERALL CONFIGURATIONS")
    print("=" * 110)
    print(hdr)
    print("-" * 110)
    for _, r in df.head(25).iterrows():
        cfg = f"ATR:{r['atr']} RR:{r['rr']} S:{str(r['sma'])[:1]} G:{str(r['gauss'])[:1]} R:{str(r['rsi'])[:1]} ORB:{r['orb'][3:]} Cut:{r['cutoff'][:2]}"
        print(f"{r['ticker']:<7} | {r['score']:<5.3f} | {r['is_sharpe']:<7.2f} | {r['oos_sharpe']:<8.2f} | {r['is_win_rate']:>5.1f}% | {r['oos_win_rate']:>5.1f}%  | {r['is_return']:>6.1f}% | {r['oos_return']:>7.1f}% | {cfg}")
    print("=" * 110)
