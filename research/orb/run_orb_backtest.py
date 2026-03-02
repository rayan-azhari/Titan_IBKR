import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf


def run_orb_backtest(ticker: str = "SPY", days: int = 59):
    print(f"Fetching {days} days of 5-minute data for {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    df = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)

    if df.empty:
        print("No data downloaded.")
        return

    # Flatten multi-index if yfinance returns one
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.index = df.index.tz_convert("US/Eastern")
    df = df.between_time("09:30", "15:55")

    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    groups = df.groupby(df.index.date)

    for date, group in groups:
        if len(group) < 4:
            continue

        # Establish 15-minute range (09:30, 09:35, 09:40 bars)
        orb_bars = group.between_time("09:30", "09:40")
        if len(orb_bars) == 0:
            continue

        or_high = float(orb_bars["High"].max())
        or_low = float(orb_bars["Low"].min())

        # Trade the breakout after 09:45
        trading_bars = group.between_time("09:45", "15:55")
        position = 0
        current_sl = 0.0
        current_tp = 0.0

        for ts, row in trading_bars.iterrows():
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])

            if position == 0:
                # Long Breakout
                if close > or_high:
                    entries.loc[ts] = True
                    position = 1
                    current_sl = or_low
                    distance = close - current_sl
                    current_tp = close + (2 * distance)
                    continue

                # Short Breakout
                if close < or_low:
                    short_entries.loc[ts] = True
                    position = -1
                    current_sl = or_high
                    distance = current_sl - close
                    current_tp = close - (2 * distance)
                    continue

            elif position == 1:
                # Check Exits (Stop Loss or Take Profit)
                if low <= current_sl or high >= current_tp:
                    exits.loc[ts] = True
                    position = 0

            elif position == -1:
                # Check Exits
                if high >= current_sl or low <= current_tp:
                    short_exits.loc[ts] = True
                    position = 0

        # End of day flat exit
        if position == 1:
            exits.loc[group.index[-1]] = True
        elif position == -1:
            short_exits.loc[group.index[-1]] = True

    # Build VectorBT Portfolio
    pf = vbt.Portfolio.from_signals(
        df["Close"],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=100_000,
        fees=0.0001,  # Assume 1 bps fee
        slippage=0.0005,  # Assume 0.05% slippage
        freq="5min",
    )

    print("=" * 50)
    print(f"🚀 ORB Strategy Backtest: {ticker} (Last {days} Days)")
    print("=" * 50)
    print(f"Total Return [%]:   {pf.total_return() * 100:.2f}%")
    print(f"Sharpe Ratio:       {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown [%]:   {pf.max_drawdown() * 100:.2f}%")
    print(f"Win Rate [%]:       {pf.trades.win_rate() * 100:.2f}%")
    print(f"Total Trades:       {pf.trades.count()}")
    print(f"Profit Factor:      {pf.trades.profit_factor():.2f}")
    print("=" * 50)


if __name__ == "__main__":
    run_orb_backtest("SPY", 59)
