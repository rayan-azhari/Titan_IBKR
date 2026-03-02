import math
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

def run_orb_optimization(ticker: str = "NVDA", days: int = 59):
    print(f"Fetching {days} days of data for {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    df_5m = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
    if df_5m.empty:
        print("No 5m data downloaded.")
        return

    # Download 1h data
    hourly_start = start_date - timedelta(days=20)
    df_1h = yf.download(ticker, start=hourly_start, end=end_date, interval="1h", progress=False)
    
    # Download 1d data
    daily_start = start_date - timedelta(days=100)
    df_1d = yf.download(ticker, start=daily_start, end=end_date, interval="1d", progress=False)

    for d in [df_5m, df_1h, df_1d]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)

    df_5m.index = df_5m.index.tz_convert("US/Eastern")
    df_1h.index = df_1h.index.tz_convert("US/Eastern")
    df_1d.index = df_1d.index.tz_localize(None)

    # Pre-calculate indicators
    df_1d.loc[:, "SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
    df_1d.loc[:, "SMA20"] = df_1d["Close"].rolling(20).mean().shift(1)
    df_1h.loc[:, "SMA50"] = df_1h["Close"].rolling(50).mean().shift(1)
    df_1h.loc[:, "SMA20"] = df_1h["Close"].rolling(20).mean().shift(1)
    
    df_5m.loc[:, "Vol_SMA20"] = df_5m["Volume"].rolling(20).mean()
    df_1h.loc[:, "Vol_SMA20"] = df_1h["Volume"].rolling(20).mean().shift(1)

    df_5m = df_5m.between_time("09:30", "15:55").copy()

    # Mapping to 5m
    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
    df_5m["Hour_Str"] = df_5m.index.strftime("%Y-%m-%d %H")
    df_1h["Hour_Str"] = df_1h.index.strftime("%Y-%m-%d %H")

    df_5m["Daily_SMA50"] = df_5m["Date_Str"].map(df_1d.set_index("Date_Str")["SMA50"].to_dict())
    df_5m["Daily_SMA20"] = df_5m["Date_Str"].map(df_1d.set_index("Date_Str")["SMA20"].to_dict())
    
    df_5m["Hourly_SMA50"] = df_5m["Hour_Str"].map(df_1h.set_index("Hour_Str")["SMA50"].to_dict())
    df_5m["Hourly_SMA20"] = df_5m["Hour_Str"].map(df_1h.set_index("Hour_Str")["SMA20"].to_dict())
    
    df_5m["Hourly_Vol_SMA20"] = df_5m["Hour_Str"].map(df_1h.set_index("Hour_Str")["Vol_SMA20"].to_dict())

    # Parameter grid
    sma_options = [
        ("Daily", "SMA50", "Daily_SMA50"),
        ("Daily", "SMA20", "Daily_SMA20"),
        ("Hourly", "SMA50", "Hourly_SMA50"),
        ("Hourly", "SMA20", "Hourly_SMA20"),
        ("None", "None", None)
    ]
    vol_options = [
        ("5m", "Vol_SMA20"),
        ("Hourly", "Hourly_Vol_SMA20"),
        ("None", None)
    ]
    trail_options = [
        "None",
        "Tight (1R->BE, 2R->1R)",
        "Loose (2R->BE)"
    ]

    best_sharpe = -999.0
    best_params = None
    best_pf = None

    groups = df_5m.groupby(df_5m.index.date)

    total_iters = len(sma_options) * len(vol_options) * len(trail_options)
    print(f"Running grid search over {total_iters} combinations...")

    for (sma_label, sma_per, sma_col), (vol_label, vol_col), trail_type in itertools.product(sma_options, vol_options, trail_options):
        
        entries = pd.Series(False, index=df_5m.index)
        exits = pd.Series(False, index=df_5m.index)
        short_entries = pd.Series(False, index=df_5m.index)
        short_exits = pd.Series(False, index=df_5m.index)

        for date, group in groups:
            if len(group) < 4: continue
            orb_bars = group.between_time("09:30", "09:40")
            if len(orb_bars) == 0: continue

            or_high = float(orb_bars["High"].max())
            or_low = float(orb_bars["Low"].min())

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

                # Trend filter
                bull_trend, bear_trend = True, True
                if sma_col and not pd.isna(row[sma_col]):
                    sma_val = float(row[sma_col])
                    bull_trend = close > sma_val
                    bear_trend = close < sma_val
                
                # Volume filter
                high_vol = True
                if vol_col and not pd.isna(row[vol_col]):
                    vol_val = float(row[vol_col])
                    high_vol = volume > vol_val

                if position == 0:
                    if close > or_high and bull_trend and high_vol:
                        entries.loc[ts] = True
                        position = 1
                        entry_price = close
                        current_sl = or_low
                        initial_risk = entry_price - current_sl
                        highest_price = close
                        continue

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
                    if initial_risk > 0:
                        r_multiple = (highest_price - entry_price) / initial_risk
                        if trail_type == "Tight (1R->BE, 2R->1R)":
                            if r_multiple >= 3.0: current_sl = max(current_sl, entry_price + 2.0 * initial_risk)
                            elif r_multiple >= 2.0: current_sl = max(current_sl, entry_price + 1.0 * initial_risk)
                            elif r_multiple >= 1.0: current_sl = max(current_sl, entry_price)
                        elif trail_type == "Loose (2R->BE)":
                            if r_multiple >= 3.0: current_sl = max(current_sl, entry_price + 1.0 * initial_risk)
                            elif r_multiple >= 2.0: current_sl = max(current_sl, entry_price)
                    
                    if low <= current_sl:
                        exits.loc[ts] = True
                        position = 0

                elif position == -1:
                    lowest_price = min(lowest_price, low)
                    if initial_risk > 0:
                        r_multiple = (entry_price - lowest_price) / initial_risk
                        if trail_type == "Tight (1R->BE, 2R->1R)":
                            if r_multiple >= 3.0: current_sl = min(current_sl, entry_price - 2.0 * initial_risk)
                            elif r_multiple >= 2.0: current_sl = min(current_sl, entry_price - 1.0 * initial_risk)
                            elif r_multiple >= 1.0: current_sl = min(current_sl, entry_price)
                        elif trail_type == "Loose (2R->BE)":
                            if r_multiple >= 3.0: current_sl = min(current_sl, entry_price - 1.0 * initial_risk)
                            elif r_multiple >= 2.0: current_sl = min(current_sl, entry_price)
                            
                    if high >= current_sl:
                        short_exits.loc[ts] = True
                        position = 0

            # EOD
            if position == 1: exits.loc[group.index[-1]] = True
            elif position == -1: short_exits.loc[group.index[-1]] = True

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

        sharpe = float(pf.sharpe_ratio())
        if not np.isnan(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (sma_label, sma_per, vol_label, trail_type)
            best_pf = pf

    print("\n" + "="*60)
    if best_params:
        print(f"🏆 BEST ORB OPTIMIZATION RESULTS: {ticker} (Last {days} Days)")
        print("="*60)
        print(f"Trend Filter:   {best_params[0]} {best_params[1]}")
        print(f"Volume Filter:  {best_params[2]} > SMA20")
        print(f"Trailing Stop:  {best_params[3]}")
        print("-" * 60)
        print(f"Total Return [%]:   {best_pf.total_return() * 100:.2f}%")
        print(f"Sharpe Ratio:       {best_pf.sharpe_ratio():.2f}")
        print(f"Max Drawdown [%]:   {best_pf.max_drawdown() * 100:.2f}%")
        print(f"Win Rate [%]:       {best_pf.trades.win_rate() * 100:.2f}%")
        print(f"Total Trades:       {best_pf.trades.count()}")
        print(f"Profit Factor:      {best_pf.trades.profit_factor():.2f}")
        print("="*60)
    else:
        print("No profitable combination found.")

if __name__ == "__main__":
    run_orb_optimization("NVDA", 59)
