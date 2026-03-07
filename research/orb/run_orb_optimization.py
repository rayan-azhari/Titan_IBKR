import itertools
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

# Suppress frequent yfinance warnings
warnings.filterwarnings("ignore")


def run_orb_optimization(ticker: str, days: int = 59) -> dict | None:
    print(f"Fetching {days} days of data for {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    df_5m = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
    if df_5m.empty:
        print(f"No 5m data downloaded for {ticker}.")
        return None

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

    # Parameter grid (We loosen the Volume filter by removing the harsh Hourly requirement)
    sma_options = [
        ("Daily", "SMA50", "Daily_SMA50"),
        ("Hourly", "SMA20", "Hourly_SMA20"),
        ("None", "None", None),
    ]
    vol_options = [("Loosened 5m > SMA20", "Vol_SMA20"), ("None", None)]
    trail_options = ["None", "Loose (2R->BE)"]

    best_sharpe = -999.0
    best_params = None
    best_pf = None

    groups = df_5m.groupby(df_5m.index.date)

    for (sma_label, sma_per, sma_col), (vol_label, vol_col), trail_type in itertools.product(
        sma_options, vol_options, trail_options
    ):
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
                        if trail_type == "Loose (2R->BE)":
                            if r_multiple >= 3.0:
                                current_sl = max(current_sl, entry_price + 1.0 * initial_risk)
                            elif r_multiple >= 2.0:
                                current_sl = max(current_sl, entry_price)

                    if low <= current_sl or (
                        entry_price > 0 and (close - entry_price) >= (2.0 * initial_risk)
                    ):  # Fixed 1:2 Reward fallback if no trail
                        exits.loc[ts] = True
                        position = 0

                elif position == -1:
                    lowest_price = min(lowest_price, low)
                    if initial_risk > 0:
                        r_multiple = (entry_price - lowest_price) / initial_risk
                        if trail_type == "Loose (2R->BE)":
                            if r_multiple >= 3.0:
                                current_sl = min(current_sl, entry_price - 1.0 * initial_risk)
                            elif r_multiple >= 2.0:
                                current_sl = min(current_sl, entry_price)

                    if high >= current_sl or (
                        entry_price > 0 and (entry_price - close) >= (2.0 * initial_risk)
                    ):
                        short_exits.loc[ts] = True
                        position = 0

            # EOD
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

        sharpe = float(pf.sharpe_ratio())
        if not np.isnan(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (sma_label, sma_per, vol_label, trail_type)
            best_pf = pf

    if best_params and best_pf:
        return {
            "ticker": ticker,
            "trend": f"{best_params[0]} {best_params[1]}",
            "volume": best_params[2],
            "trailing": best_params[3],
            "return": best_pf.total_return() * 100,
            "sharpe": best_pf.sharpe_ratio(),
            "win_rate": best_pf.trades.win_rate() * 100,
            "trades": best_pf.trades.count(),
            "profit_factor": best_pf.trades.profit_factor(),
        }
    return None


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    results = []

    for t in tickers:
        res = run_orb_optimization(t, 59)
        if res:
            results.append(res)

    print("\n" + "=" * 80)
    print("BEST ORB OPTIMIZATION RESULTS ACROSS TECH MAJORS (Last 59 Days)")
    print("=" * 80)
    print(
        f"{'Ticker':<8} | {'Return':<8} | {'Win Rate':<10} | {'Trades':<8} | {'Trend Filter':<20} | {'Volume Filter':<20} | {'Trailing Stop'}"
    )
    print("-" * 80)

    # Sort by return
    results.sort(key=lambda x: x["return"], reverse=True)

    for r in results:
        ret_str = f"{r['return']:.2f}%"
        win_str = f"{r['win_rate']:.2f}%"
        print(
            f"{r['ticker']:<8} | {ret_str:<8} | {win_str:<10} | {r['trades']:<8} | {r['trend']:<20} | {r['volume']:<20} | {r['trailing']}"
        )
    print("=" * 80)
