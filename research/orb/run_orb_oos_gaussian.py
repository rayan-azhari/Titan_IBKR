"""ORB IS/OOS Validation with Gaussian Channel filter — Databento 1-year 5m data.

Validates the top performers from the S&P 500 Gaussian ORB scan (59-day window)
on a full year of true 5m data with a 70/30 time-based IS/OOS split.

Entry filter: price must be above (long) or below (short) the Gaussian midline.
Acceptance: OOS/IS Sharpe ratio >= 0.5. Below that = overfit.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Allow absolute imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from titan.indicators.gaussian_filter import _gaussian_channel_kernel  # noqa: E402

load_dotenv()
API_KEY = os.getenv("DATABENTO_API_KEY")

if not API_KEY:
    print("[!] DATABENTO_API_KEY not found in .env — required for uncached tickers.")
    print("    Cached tickers will still run. Missing tickers will be skipped.")

TICKERS = [
    # Gaussian scan top performers (may need Databento fetch)
    "TYL",
    "IBM",
    "VLO",
    "VTRS",
    "STLD",
    "KKR",
    # Previously validated tickers (all cached)
    "ORCL",
    "INTU",
    "MO",
    "CAT",
    "AMAT",
    "BKNG",
    "HON",
    "ICE",
    "GE",
]

GAUSS_PERIOD = 144.0
GAUSS_POLES = 4
GAUSS_SIGMA = 2.0
ATR_MULTIPLIER = 1.5
RR_RATIO = 2.0
CACHE_DIR = "data/databento"


def download_databento_1yr(ticker: str) -> pd.DataFrame:
    """Fetch 1-year 5m data from Databento, using local cache if available."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/{ticker}_1yr_5m.csv"

    if os.path.exists(cache_file):
        print(f"  [cache] {ticker}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=False)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        return df

    if not API_KEY:
        print(f"  [skip]  {ticker} — no API key and no cache")
        return pd.DataFrame()

    import databento as db

    client = db.Historical(API_KEY)
    print(f"  [fetch] {ticker} from Databento...")

    end_date = pd.Timestamp.utcnow().floor("D")
    start_date = end_date - pd.Timedelta(days=365)
    monthly_dfs = []

    try:
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + pd.Timedelta(days=30), end_date)
            print(f"    -> {current_start.date()} to {current_end.date()}")
            data = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                stype_in="raw_symbol",
                symbols=[ticker],
                start=current_start.isoformat(),
                end=current_end.isoformat(),
                limit=None,
            )
            chunk = data.to_df()
            if not chunk.empty:
                if isinstance(chunk.index, pd.MultiIndex):
                    chunk = chunk.reset_index(level="symbol", drop=True)
                monthly_dfs.append(chunk)
            current_start = current_end

        if not monthly_dfs:
            return pd.DataFrame()

        df = pd.concat(monthly_dfs)
        df = df[~df.index.duplicated(keep="first")].tz_convert("America/New_York")
        df_5m = (
            df.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        df_5m.columns = ["Open", "High", "Low", "Close", "Volume"]
        df_5m = df_5m.between_time("09:30", "15:55").copy()
        df_5m.to_csv(cache_file)
        print(f"    Saved {len(df_5m)} bars to cache.")
        return df_5m

    except Exception as e:
        print(f"  [error] {ticker}: {e}")
        return pd.DataFrame()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()


def run_oos_validation(ticker: str) -> dict | None:
    df_5m = download_databento_1yr(ticker)
    if df_5m.empty or len(df_5m) < 500:
        return None

    # Daily 1d context from yfinance (SMA50 — no rate limit for 1d)
    df_1d = yf.download(
        ticker,
        start=df_5m.index.min() - timedelta(days=100),
        end=df_5m.index.max(),
        interval="1d",
        progress=False,
    )
    if df_1d.empty:
        return None
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.droplevel(1)
    df_1d.index = df_1d.index.tz_localize(None)
    df_1d["SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")

    df_5m["ATR"] = calculate_atr(df_5m, 14)
    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_5m["Daily_SMA50"] = df_5m["Date_Str"].map(df_1d.set_index("Date_Str")["SMA50"].to_dict())

    # Gaussian midline on full 5m series (needs full history for warmup)
    h = np.ascontiguousarray(df_5m["High"].values, dtype=np.float64)
    lo = np.ascontiguousarray(df_5m["Low"].values, dtype=np.float64)
    c = np.ascontiguousarray(df_5m["Close"].values, dtype=np.float64)
    _, _, gauss_mid_arr = _gaussian_channel_kernel(h, lo, c, GAUSS_PERIOD, GAUSS_POLES, GAUSS_SIGMA)
    df_5m["GaussMid"] = gauss_mid_arr

    # 70/30 IS/OOS time split
    split_idx = int(len(df_5m) * 0.70)
    in_sample = df_5m.iloc[:split_idx].copy()
    out_sample = df_5m.iloc[split_idx:].copy()

    def backtest(data: pd.DataFrame) -> vbt.Portfolio:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)

        for _date, group in data.groupby(data.index.date):
            if len(group) < 4:
                continue
            orb_bars = group.between_time("09:30", "09:40")
            if len(orb_bars) == 0:
                continue
            or_high = float(orb_bars["High"].max())
            or_low = float(orb_bars["Low"].min())

            trading_bars = group.between_time("09:45", "15:55")
            position = 0
            current_sl = 0.0
            current_tp = 0.0

            for ts, row in trading_bars.iterrows():
                close = float(row["Close"])
                high = float(row["High"])
                low = float(row["Low"])
                atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else (or_high - or_low)
                sma = float(row["Daily_SMA50"]) if not pd.isna(row["Daily_SMA50"]) else 0.0
                gauss_mid = float(row["GaussMid"]) if not pd.isna(row["GaussMid"]) else 0.0

                bull_trend = close > sma and close > gauss_mid
                bear_trend = close < sma and close < gauss_mid

                if position == 0:
                    if close > or_high and bull_trend:
                        entries.loc[ts] = True
                        position = 1
                        risk = atr * ATR_MULTIPLIER
                        current_sl = close - risk
                        current_tp = close + risk * RR_RATIO
                        continue
                    if close < or_low and bear_trend:
                        short_entries.loc[ts] = True
                        position = -1
                        risk = atr * ATR_MULTIPLIER
                        current_sl = close + risk
                        current_tp = close - risk * RR_RATIO
                        continue
                elif position == 1:
                    if low <= current_sl or high >= current_tp:
                        exits.loc[ts] = True
                        position = 0
                elif position == -1:
                    if high >= current_sl or low <= current_tp:
                        short_exits.loc[ts] = True
                        position = 0

            if position == 1:
                exits.loc[group.index[-1]] = True
            elif position == -1:
                short_exits.loc[group.index[-1]] = True

        return vbt.Portfolio.from_signals(
            data["Close"],
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100_000,
            fees=0.0001,
            slippage=0.0005,
            freq="5min",
        )

    is_pf = backtest(in_sample)
    oos_pf = backtest(out_sample)

    def safe(val: float) -> float:
        return float(val) if not (np.isnan(val) or np.isinf(val)) else 0.0

    is_sharpe = safe(is_pf.sharpe_ratio())
    oos_sharpe = safe(oos_pf.sharpe_ratio())
    ratio = oos_sharpe / is_sharpe if is_sharpe > 0.01 else 0.0

    return {
        "ticker": ticker,
        "is_return": safe(is_pf.total_return() * 100),
        "is_win_rate": safe(is_pf.trades.win_rate() * 100) if is_pf.trades.count() > 0 else 0.0,
        "is_sharpe": is_sharpe,
        "is_trades": is_pf.trades.count(),
        "oos_return": safe(oos_pf.total_return() * 100),
        "oos_win_rate": safe(oos_pf.trades.win_rate() * 100) if oos_pf.trades.count() > 0 else 0.0,
        "oos_sharpe": oos_sharpe,
        "oos_trades": oos_pf.trades.count(),
        "ratio": ratio,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("ORB PHASE 4 — Gaussian IS/OOS Validation (Databento 1-Year True 5m Data)")
    print("Gaussian filter: period=144, poles=4, sigma=2.0 | ATR stop: 1.5x | RR: 2:1")
    print("IS/OOS split: 70% / 30% time-based | Acceptance: OOS/IS Sharpe >= 0.5")
    print("=" * 90)

    results = []
    for ticker in TICKERS:
        print(f"\nValidating {ticker}...")
        try:
            res = run_oos_validation(ticker)
            if res:
                results.append(res)
        except Exception as e:
            print(f"  [error] {ticker}: {e}")

    if not results:
        print("\nNo results produced.")
        sys.exit(0)

    results.sort(key=lambda x: x["oos_sharpe"], reverse=True)

    print("\n" + "=" * 110)
    print("RESULTS (sorted by OOS Sharpe)")
    print("=" * 110)
    hdr = (
        f"{'Ticker':<6} | {'IS Ret':>8} | {'IS Win%':>8} | {'IS Sh':>6} | {'IS Tr':>5}"
        f" || {'OOS Ret':>8} | {'OOS Win%':>9} | {'OOS Sh':>7} | {'OOS Tr':>6}"
        f" | {'Ratio':>6} | Status"
    )
    print(hdr)
    print("-" * 110)

    for r in results:
        ratio = r["ratio"]
        if r["oos_return"] < 0:
            status = "FAIL"
        elif ratio >= 0.5:
            status = "VALID"
        else:
            status = "OVERFIT"

        row = (
            f"{r['ticker']:<6} | {r['is_return']:>7.2f}% | {r['is_win_rate']:>7.2f}% |"
            f" {r['is_sharpe']:>6.2f} | {r['is_trades']:>5}"
            f" || {r['oos_return']:>7.2f}% | {r['oos_win_rate']:>8.2f}% |"
            f" {r['oos_sharpe']:>7.2f} | {r['oos_trades']:>6}"
            f" | {ratio:>6.2f} | {status}"
        )
        print(row)

    print("=" * 110)

    # Save CSV
    os.makedirs(".tmp", exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = f".tmp/orb_oos_gaussian_{date_str}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    valid = [r for r in results if r["ratio"] >= 0.5 and r["oos_return"] > 0]
    print(f"\nValid tickers (OOS/IS >= 0.5, OOS > 0): {[r['ticker'] for r in valid]}")
