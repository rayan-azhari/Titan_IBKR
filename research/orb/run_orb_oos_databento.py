import os
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import databento as db
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()
API_KEY = os.getenv("DATABENTO_API_KEY")

if not API_KEY:
    print("\n[!] DATABENTO_API_KEY not found in .env file.")
    print("Databento is required to fetch 1-Year of 5-minute historical data for Out-of-Sample testing.")
    print("You can get $125 in free credits securely at https://databento.com/")
    print("Please add your key to the .env file and run this again.")
    exit(1)

client = db.Historical(API_KEY)

def download_databento_1yr(ticker: str) -> pd.DataFrame:
    # Check for cached data first to save API credits
    cache_dir = "data/databento"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{ticker}_1yr_5m.csv"
    
    if os.path.exists(cache_file):
        print(f"Loading cached 1-Year 5m data for {ticker} from {cache_file}...")
        df_5m = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Ensure timezone awareness is maintained from CSV
        if df_5m.index.tz is None:
            df_5m.index = df_5m.index.tz_localize("America/New_York")
        return df_5m

    print(f"Fetching 1-Year of 1m data for {ticker} from Databento in monthly chunks...")
    end_date = pd.Timestamp.utcnow().floor('D')
    start_date = end_date - pd.Timedelta(days=365)
    
    monthly_dfs = []
    
    try:
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + pd.Timedelta(days=30)
            if current_end > end_date:
                current_end = end_date
                
            print(f"  -> Downloading {current_start.date()} to {current_end.date()}")
            
            data = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                stype_in="raw_symbol",
                symbols=[ticker],
                start=current_start.isoformat(),
                end=current_end.isoformat(),
                limit=None
            )
            
            chunk_df = data.to_df()
            if not chunk_df.empty:
                if isinstance(chunk_df.index, pd.MultiIndex):
                    chunk_df = chunk_df.reset_index(level="symbol", drop=True)
                monthly_dfs.append(chunk_df)
                
            current_start = current_end
            
        if not monthly_dfs:
            return pd.DataFrame()
            
        df = pd.concat(monthly_dfs)
        df = df[~df.index.duplicated(keep='first')] # Drop overlaps
        df = df.tz_convert("America/New_York")
        
        df_5m = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        df_5m.columns = ["Open", "High", "Low", "Close", "Volume"]
        df_5m = df_5m.between_time("09:30", "15:55").copy()
        print(f"Successfully resampled {len(df_5m)} 5-minute bars for {ticker}.")
        
        df_5m.to_csv(cache_file)
        print(f"Saved {ticker} data to cache: {cache_file}")
        
        return df_5m
        
    except Exception as e:
        print(f"Failed to fetch Databento data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_out_of_sample_test(ticker: str, params: dict):
    df_5m = download_databento_1yr(ticker)
    if df_5m.empty: return None
    
    # Still use yfinance for the 1D trend and RSI since that is not rate-limited for 1-Year history
    df_1d = yf.download(ticker, start=df_5m.index.min() - timedelta(days=100), end=df_5m.index.max(), interval="1d", progress=False)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.droplevel(1)
    df_1d.index = df_1d.index.tz_localize(None)

    df_5m["ATR"] = calculate_atr(df_5m, period=14)
    df_1d["SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
    df_1d["RSI14"] = calculate_rsi(df_1d["Close"], period=14).shift(1)
    
    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
    
    df_5m["Daily_SMA50"] = df_5m["Date_Str"].map(df_1d.set_index("Date_Str")["SMA50"].to_dict())
    df_5m["Daily_RSI14"] = df_5m["Date_Str"].map(df_1d.set_index("Date_Str")["RSI14"].to_dict())

    # Split Data 70% In-Sample / 30% Out-of-Sample
    split_idx = int(len(df_5m) * 0.70)
    in_sample_df = df_5m.iloc[:split_idx].copy()
    out_sample_df = df_5m.iloc[split_idx:].copy()

    def backtest(data: pd.DataFrame):
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)

        groups = data.groupby(data.index.date)

        for date, group in groups:
            if len(group) < 4: continue
            
            # True 5-minute ORB (09:30 and 09:35 bars make up the 15-minute range)
            orb_bars = group.between_time("09:30", "09:40")
            if len(orb_bars) == 0: continue
            
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
                rsi = float(row["Daily_RSI14"]) if not pd.isna(row["Daily_RSI14"]) else 50.0

                bull_trend = (close > sma) if params['sma'] else True
                bear_trend = (close < sma) if params['sma'] else True
                bull_rsi = (rsi < 70) if params['rsi'] else True
                bear_rsi = (rsi > 30) if params['rsi'] else True

                if position == 0:
                    if close > or_high and bull_trend and bull_rsi:
                        entries.loc[ts] = True
                        position = 1
                        initial_risk = atr * params['atr']
                        current_sl = close - initial_risk
                        current_tp = close + (initial_risk * 2.0)
                        continue

                    if close < or_low and bear_trend and bear_rsi:
                        short_entries.loc[ts] = True
                        position = -1
                        initial_risk = atr * params['atr']
                        current_sl = close + initial_risk
                        current_tp = close - (initial_risk * 2.0)
                        continue

                elif position == 1:
                    if low <= current_sl or high >= current_tp:
                        exits.loc[ts] = True
                        position = 0
                elif position == -1:
                    if high >= current_sl or low <= current_tp:
                        short_exits.loc[ts] = True
                        position = 0

            # EOD Dump
            if position == 1: exits.loc[group.index[-1]] = True
            elif position == -1: short_exits.loc[group.index[-1]] = True

        pf = vbt.Portfolio.from_signals(
            data["Close"], entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            init_cash=100_000, fees=0.0001, slippage=0.0005, freq="5min"
        )
        return pf

    is_pf = backtest(in_sample_df)
    oos_pf = backtest(out_sample_df)
    
    return {
        "ticker": ticker,
        "is_return": is_pf.total_return() * 100,
        "is_win_rate": is_pf.trades.win_rate() * 100 if is_pf.trades.count() > 0 else 0,
        "oos_return": oos_pf.total_return() * 100,
        "oos_win_rate": oos_pf.trades.win_rate() * 100 if oos_pf.trades.count() > 0 else 0,
        "total_trades": is_pf.trades.count() + oos_pf.trades.count()
    }

if __name__ == "__main__":
    top_10 = [
        {"ticker": "IBM", "atr": 1.5, "sma": True, "rsi": False},
        {"ticker": "ORCL", "atr": 2.0, "sma": True, "rsi": False},
        {"ticker": "INTU", "atr": 2.0, "sma": False, "rsi": False},
        {"ticker": "MO", "atr": 2.0, "sma": True, "rsi": False},
        {"ticker": "CAT", "atr": 2.0, "sma": True, "rsi": True},
        {"ticker": "AMAT", "atr": 2.0, "sma": False, "rsi": True},
        {"ticker": "BKNG", "atr": 2.0, "sma": False, "rsi": False},
        {"ticker": "HON", "atr": 2.0, "sma": True, "rsi": True},
        {"ticker": "ICE", "atr": 2.0, "sma": True, "rsi": True},
        {"ticker": "GE", "atr": 2.0, "sma": False, "rsi": False},
    ]

    print("\nRunning 1-Year 70/30 In-Sample & Out-of-Sample Test on Top 10 Tickers using true 5m data from Databento...")
    results = []
    
    for config in top_10:
        res = run_out_of_sample_test(config['ticker'], config)
        if res:
            results.append(res)
            
    if results:
        print("\n================================================================================")
        print("📊 1-YEAR TRUE 5-MIN OOS BACKTEST VALIDATION RESULTS (70% IS / 30% OOS)")
        print("================================================================================")
        print(f"{'Ticker':<8} | {'IS Return':<12} | {'IS Win %':<10} ||| {'OOS Return':<12} | {'OOS Win %':<10}")
        print("-" * 80)
        
        for r in results:
            is_ret = f"{r['is_return']:.2f}%"
            is_win = f"{r['is_win_rate']:.2f}%"
            oos_ret = f"{r['oos_return']:.2f}%"
            oos_win = f"{r['oos_win_rate']:.2f}%"
            drift = "PASS" if r['oos_return'] > 0 else "FAIL"
            print(f"{r['ticker']:<8} | {is_ret:<12} | {is_win:<10} ||| {oos_ret:<12} | {oos_win:<10} [{drift}]")
        print("================================================================================")
