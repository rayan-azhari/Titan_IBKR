import math
import itertools
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

def download_1yr_5m_data(ticker: str) -> pd.DataFrame:
    # yfinance only allows 60 days of 5m data per request and Max 60 days from today.
    # To get 1 year of 5m data from free tier yahoo finance is not possible (limit 60 days max).
    # Since 5m Data is severely rate limited we will switch to 1-Hour ORB for the 1-Year Backtest 
    # to demonstrate the In-Sample / Out-of-Sample mathematical validation model.
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=360) # 1 Year
    
    # Download 1h data (Yahoo allows up to 730 days of 1h data)
    df = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
    
    if df.empty:
        return pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.index = df.index.tz_convert("America/New_York")
    df = df.between_time("09:30", "15:55").copy()
    
    return df

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
    # 1. Fetch 1 Year of 1-Hour Data
    df = download_1yr_5m_data(ticker)
    if df.empty:
        return None
        
    # Download 1d data for higher timeframe context
    df_1d = yf.download(ticker, start=df.index.min() - timedelta(days=100), end=df.index.max(), interval="1d", progress=False)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.droplevel(1)
    df_1d.index = df_1d.index.tz_localize(None)

    # Calculate Indicators
    df["ATR"] = calculate_atr(df, period=14)
    df_1d["SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
    df_1d["RSI14"] = calculate_rsi(df_1d["Close"], period=14).shift(1)
    
    df["Date_Str"] = df.index.strftime("%Y-%m-%d")
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
    
    df["Daily_SMA50"] = df["Date_Str"].map(df_1d.set_index("Date_Str")["SMA50"].to_dict())
    df["Daily_RSI14"] = df["Date_Str"].map(df_1d.set_index("Date_Str")["RSI14"].to_dict())

    # Split Data 70% In-Sample / 30% Out-of-Sample
    split_idx = int(len(df) * 0.70)
    in_sample_df = df.iloc[:split_idx].copy()
    out_sample_df = df.iloc[split_idx:].copy()

    def backtest(data: pd.DataFrame, is_oos: bool = False):
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)

        groups = data.groupby(data.index.date)

        for date, group in groups:
            if len(group) < 2: continue
            
            # Since using 1H data, the first bar (09:30-10:30) is our Opening Range
            orb_bar = group.iloc[0]
            or_high = float(orb_bar["High"])
            or_low = float(orb_bar["Low"])
            
            trading_bars = group.iloc[1:]
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
            init_cash=100_000, fees=0.0001, slippage=0.0005, freq="1h"
        )
        return pf

    # Run IS and OOS
    is_pf = backtest(in_sample_df, False)
    oos_pf = backtest(out_sample_df, True)
    
    return {
        "ticker": ticker,
        "is_return": is_pf.total_return() * 100,
        "is_win_rate": is_pf.trades.win_rate() * 100 if is_pf.trades.count() > 0 else 0,
        "oos_return": oos_pf.total_return() * 100,
        "oos_win_rate": oos_pf.trades.win_rate() * 100 if oos_pf.trades.count() > 0 else 0,
        "total_trades": is_pf.trades.count() + oos_pf.trades.count()
    }

if __name__ == "__main__":
    # Top 10 Configurations from previous Grid Search Optimization Output
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

    print("\nRunning 1-Year 70/30 In-Sample & Out-of-Sample Test on Top 10 Tickers...")
    print("Using 1-Hour proxy data due to strict Yahoo Finance 60-day limits on 5m data.\n")
    
    results = []
    for config in top_10:
        print(f"Testing {config['ticker']} (IS/OOS)...")
        res = run_out_of_sample_test(config['ticker'], config)
        if res:
            results.append(res)
            
    print("\n================================================================================")
    print("📊 1-YEAR OOS BACKTEST VALIDATION RESULTS (70% IN-SAMPLE / 30% OUT-OF-SAMPLE)")
    print("================================================================================")
    print(f"{'Ticker':<8} | {'IS Return':<12} | {'IS Win %':<10} ||| {'OOS Return':<12} | {'OOS Win %':<10}")
    print("-" * 80)
    
    for r in results:
        is_ret = f"{r['is_return']:.2f}%"
        is_win = f"{r['is_win_rate']:.2f}%"
        oos_ret = f"{r['oos_return']:.2f}%"
        oos_win = f"{r['oos_win_rate']:.2f}%"
        
        # Determine drift (Curve Fitzting check)
        drift = "PASS" if r['oos_return'] > 0 else "FAIL"
        
        print(f"{r['ticker']:<8} | {is_ret:<12} | {is_win:<10} ||| {oos_ret:<12} | {oos_win:<10} [{drift}]")
    print("================================================================================")
