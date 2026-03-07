import itertools
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

SP100_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "V",
    "JPM",
    "XOM",
    "UNH",
    "WMT",
    "MA",
    "JNJ",
    "PG",
    "HD",
    "ORCL",
    "COST",
    "ABBV",
    "CVX",
    "CRM",
    "BAC",
    "MRK",
    "KO",
    "NFLX",
    "AMD",
    "PEP",
    "LIN",
    "TMO",
    "ADBE",
    "WFC",
    "CSCO",
    "MCD",
    "ABT",
    "TMUS",
    "INTU",
    "IBM",
    "GE",
    "CAT",
    "QCOM",
    "TXN",
    "AMAT",
    "DHR",
    "VZ",
    "PFE",
    "PM",
    "ISRG",
    "NOW",
    "COP",
    "BA",
    "SPGI",
    "HON",
    "AMGN",
    "RTX",
    "UNP",
    "LOW",
    "INTC",
    "SYK",
    "GS",
    "NEE",
    "ELV",
    "BLK",
    "TJX",
    "PGR",
    "AXP",
    "MDT",
    "C",
    "LMT",
    "UBER",
    "VRTX",
    "CB",
    "REGN",
    "MMC",
    "ADI",
    "BSX",
    "CI",
    "CVS",
    "ZTS",
    "T",
    "FI",
    "SLB",
    "MDLZ",
    "MO",
    "BKNG",
    "GILD",
    "EOG",
    "BDX",
    "SO",
    "CME",
    "NOC",
    "CSX",
    "ITW",
    "DUK",
    "CL",
    "ICE",
    "FDX",
    "USB",
    "PNC",
]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def evaluate_ticker(ticker: str, days: int = 59) -> list[dict]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    # Download 5m data
    df_5m = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
    if df_5m.empty:
        return []

    # Download 1d data for higher timeframe context (offset for buffer)
    df_1d = yf.download(
        ticker, start=start_date - timedelta(days=100), end=end_date, interval="1d", progress=False
    )
    if df_1d.empty:
        return []

    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.droplevel(1)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.droplevel(1)

    df_5m.index = df_5m.index.tz_convert("America/New_York")
    df_1d.index = df_1d.index.tz_localize(None)

    # Calculate Indicators
    df_5m["ATR"] = calculate_atr(df_5m, period=14)
    df_1d["SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
    df_1d["RSI14"] = calculate_rsi(df_1d["Close"], period=14).shift(1)

    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")

    sma_map = df_1d.set_index("Date_Str")["SMA50"].to_dict()
    rsi_map = df_1d.set_index("Date_Str")["RSI14"].to_dict()

    df_5m["Daily_SMA50"] = df_5m["Date_Str"].map(sma_map)
    df_5m["Daily_RSI14"] = df_5m["Date_Str"].map(rsi_map)

    df_5m = df_5m.between_time("09:30", "15:55").copy()

    # Grid search configs
    atr_mults = [1.0, 1.5, 2.0]
    sma_filters = [True, False]  # True = Requires SMA alignment
    rsi_filters = [
        True,
        False,
    ]  # True = Requires RSI to NOT be overextended (<70 for long, >30 for short)

    results = []
    groups = df_5m.groupby(df_5m.index.date)

    for atr_m, use_sma, use_rsi in itertools.product(atr_mults, sma_filters, rsi_filters):
        entries = pd.Series(False, index=df_5m.index)
        exits = pd.Series(False, index=df_5m.index)
        short_entries = pd.Series(False, index=df_5m.index)
        short_exits = pd.Series(False, index=df_5m.index)

        for date, group in groups:
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
                rsi = float(row["Daily_RSI14"]) if not pd.isna(row["Daily_RSI14"]) else 50.0

                bull_trend = (close > sma) if use_sma else True
                bear_trend = (close < sma) if use_sma else True

                bull_rsi = (rsi < 70) if use_rsi else True
                bear_rsi = (rsi > 30) if use_rsi else True

                if position == 0:
                    if close > or_high and bull_trend and bull_rsi:
                        entries.loc[ts] = True
                        position = 1
                        initial_risk = atr * atr_m
                        current_sl = close - initial_risk
                        current_tp = close + (initial_risk * 2.0)  # Fixed 1:2 RR
                        continue

                    if close < or_low and bear_trend and bear_rsi:
                        short_entries.loc[ts] = True
                        position = -1
                        initial_risk = atr * atr_m
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
            if position == 1:
                exits.loc[group.index[-1]] = True
            elif position == -1:
                short_exits.loc[group.index[-1]] = True

        pf = vbt.Portfolio.from_signals(
            df_5m["Close"],
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100_000,
            fees=0.0001,
            slippage=0.0005,
            freq="5min",
        )

        trades = pf.trades.count()
        if trades > 0:
            returns = pf.total_return() * 100
            sharpe = pf.sharpe_ratio() if not np.isnan(pf.sharpe_ratio()) else 0
            if (
                returns > 0 and sharpe > 0
            ):  # Only store strictly profitable configurations to save memory
                results.append(
                    {
                        "ticker": ticker,
                        "atr": atr_m,
                        "sma": use_sma,
                        "rsi": use_rsi,
                        "return": returns,
                        "sharpe": sharpe,
                        "win_rate": pf.trades.win_rate() * 100,
                        "trades": trades,
                        "profit_factor": pf.trades.profit_factor()
                        if not np.isnan(pf.trades.profit_factor())
                        else 0,
                    }
                )
    return results


if __name__ == "__main__":
    print("Starting Multi-Threaded Grid Search Optimization on Top 100 Tickers...")
    print("Testing 12 configurations per ticker (1,200 total backtests). Please wait...\n")

    # Process the 100 tickers in parallel across CPU cores to drastically reduce execution time
    all_results = Parallel(n_jobs=-1)(delayed(evaluate_ticker)(t, 59) for t in SP100_TICKERS)

    # Flatten the list of lists
    flat_results = [item for sublist in all_results for item in sublist]

    if not flat_results:
        print("No profitable configurations found in the entire S&P 100.")
        exit(0)

    # Sort globally by best returns
    flat_results.sort(key=lambda x: x["return"], reverse=True)

    print(
        "=========================================================================================="
    )
    print("🏆 TOP 25 BEST ORB CONFIGURATIONS ACROSS S&P 100 (Last 59 Days)")
    print(
        "=========================================================================================="
    )
    print(
        f"{'Ticker':<8} | {'Return':<8} | {'Win Rate':<10} | {'Trades':<8} | {'ATR Stop':<10} | {'SMA Filter':<12} | {'RSI Filter'}"
    )
    print("-" * 90)

    for r in flat_results[:25]:
        ret_str = f"{r['return']:.2f}%"
        win_str = f"{r['win_rate']:.2f}%"
        sma_str = "Daily SMA50" if r["sma"] else "None"
        rsi_str = "Overbought/Sold" if r["rsi"] else "None"
        print(
            f"{r['ticker']:<8} | {ret_str:<8} | {win_str:<10} | {r['trades']:<8} | {r['atr']:<10.1f} | {sma_str:<12} | {rsi_str}"
        )
    print(
        "=========================================================================================="
    )
