from datetime import datetime, timedelta

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

    # Fetch daily data for SMA50 context
    daily_start = start_date - timedelta(days=100)  # buffer for SMA50 calculation
    daily_df = yf.download(ticker, start=daily_start, end=end_date, interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = daily_df.columns.droplevel(1)

    daily_df["SMA50"] = daily_df["Close"].rolling(50).mean()
    # Shift daily SMA by 1 so we use yesterday's closed SMA50 for today's trading
    daily_df["SMA50"] = daily_df["SMA50"].shift(1)

    df.index = df.index.tz_convert("US/Eastern")
    df = df.between_time("09:30", "15:55")

    # Calculate Volume SMA for 5m bars
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()

    # Map daily SMA50 to 5m dataframe
    df["Date_Str"] = df.index.strftime("%Y-%m-%d")
    daily_df.index = daily_df.index.tz_localize(None)  # Remove tz to match string formatting
    daily_df["Date_Str"] = daily_df.index.strftime("%Y-%m-%d")

    sma_map = daily_df.set_index("Date_Str")["SMA50"].to_dict()
    df["Daily_SMA50"] = df["Date_Str"].map(sma_map)

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
        entry_price = 0.0
        highest_price = 0.0
        lowest_price = 0.0
        initial_risk = 0.0

        for ts, row in trading_bars.iterrows():
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            volume = float(row["Volume"])
            vol_sma = float(row["Vol_SMA20"])
            daily_sma = float(row["Daily_SMA50"])

            # Trend filter
            bull_trend = close > daily_sma if not pd.isna(daily_sma) else True
            bear_trend = close < daily_sma if not pd.isna(daily_sma) else True

            # Volume filter
            high_vol = volume > vol_sma if not pd.isna(vol_sma) else True

            if position == 0:
                # Long Breakout
                if close > or_high and bull_trend and high_vol:
                    entries.loc[ts] = True
                    position = 1
                    entry_price = close
                    current_sl = or_low
                    initial_risk = entry_price - current_sl
                    highest_price = close
                    continue

                # Short Breakout
                if close < or_low and bear_trend and high_vol:
                    short_entries.loc[ts] = True
                    position = -1
                    entry_price = close
                    current_sl = or_high
                    initial_risk = current_sl - entry_price
                    lowest_price = close
                    continue

            elif position == 1:
                highest_price = max(highest_price, high)

                # Dynamic Trailing Stop
                if initial_risk > 0:
                    r_multiple = (highest_price - entry_price) / initial_risk

                    if r_multiple >= 3.0:
                        current_sl = max(current_sl, entry_price + 2.0 * initial_risk)
                    elif r_multiple >= 2.0:
                        current_sl = max(current_sl, entry_price + 1.0 * initial_risk)
                    elif r_multiple >= 1.0:
                        current_sl = max(current_sl, entry_price)

                # Check Exits (Stop Loss)
                if low <= current_sl:
                    exits.loc[ts] = True
                    position = 0

            elif position == -1:
                lowest_price = min(lowest_price, low)

                # Dynamic Trailing Stop
                if initial_risk > 0:
                    r_multiple = (entry_price - lowest_price) / initial_risk

                    if r_multiple >= 3.0:
                        current_sl = min(current_sl, entry_price - 2.0 * initial_risk)
                    elif r_multiple >= 2.0:
                        current_sl = min(current_sl, entry_price - 1.0 * initial_risk)
                    elif r_multiple >= 1.0:
                        current_sl = min(current_sl, entry_price)

                # Check Exits
                if high >= current_sl:
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

    print("=" * 60)
    print(f"🚀 ORB Strategy Backtest (v2): {ticker} (Last {days} Days)")
    print("=" * 60)
    print("Filters Applied:")
    print("- Daily SMA50 Trend Filter")
    print("- Breakout Volume > 20-Period Avg")
    print("- Dynamic 1R Trailing Stop")
    print("=" * 60)
    print(f"Total Return [%]:   {pf.total_return() * 100:.2f}%")
    print(f"Sharpe Ratio:       {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown [%]:   {pf.max_drawdown() * 100:.2f}%")
    print(f"Win Rate [%]:       {pf.trades.win_rate() * 100:.2f}%")
    print(f"Total Trades:       {pf.trades.count()}")
    print(f"Profit Factor:      {pf.trades.profit_factor():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_orb_backtest("SPY", 59)
