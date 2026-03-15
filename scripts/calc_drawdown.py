import os
import sys
from datetime import time as dt_time
from datetime import timedelta

import numpy as np
import pandas as pd

# Re-use the data loading and indicators from the optimizer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from research.orb.run_orb_databento_optimizer import (
    ROUND_TRIP_COST,
    _gaussian_channel_kernel,
    calculate_atr,
    calculate_rsi,
    load_ticker_data,
)


def get_best_configs():
    df = pd.read_csv("data/databento/optimization_results.csv")
    tickers = ["UNH", "WMT", "AMAT", "TXN", "CRM", "CSCO", "INTC"]
    best_cfgs = {}
    for t in tickers:
        tdf = df[df["ticker"] == t]
        if not tdf.empty:
            best_cfgs[t] = tdf.sort_values("score", ascending=False).iloc[0].to_dict()
    return best_cfgs


def precompute_daily(df_5m, df_1d, orb_end_str, cutoff_str):
    cutoff_time = dt_time(*map(int, cutoff_str.split(":")))
    days = []

    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
    sma_map = df_1d.set_index("Date_Str")["SMA50"].to_dict()
    rsi_map = df_1d.set_index("Date_Str")["RSI14"].to_dict()
    df_5m["Daily_SMA50"] = df_5m["Date_Str"].map(sma_map)
    df_5m["Daily_RSI14"] = df_5m["Date_Str"].map(rsi_map)

    for date, group in df_5m.groupby(df_5m.index.date):
        if len(group) < 4:
            continue
        orb_bars = group.between_time("09:30", orb_end_str)
        if len(orb_bars) == 0:
            continue
        trading = group.between_time("09:30", "15:55")
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

        days.append(
            {
                "date": date,
                "close": trading["Close"].values.astype(np.float64),
                "high": trading["High"].values.astype(np.float64),
                "low": trading["Low"].values.astype(np.float64),
                "atr": trading["ATR"].values.astype(np.float64),
                "gauss_mid": trading["Gauss_Mid"].values.astype(np.float64),
                "or_high": float(orb_bars["High"].max()),
                "or_low": float(orb_bars["Low"].min()),
                "sma": sma,
                "rsi": rsi,
                "cutoff_idx": cutoff_idx,
            }
        )
    return days


def simulate_equity_curve(daily_data, cfg):
    atr_mult = cfg["atr"]
    rr_ratio = cfg["rr"]
    use_sma = cfg["sma"]
    use_rsi = cfg["rsi"]
    use_gauss = cfg["gauss"]

    equity = 100000.0  # start with $100k
    equity_curve = [equity]

    for day in daily_data:
        close = day["close"]
        high = day["high"]
        low = day["low"]
        atr = day["atr"]
        g_mid = day["gauss_mid"]
        or_h = day["or_high"]
        or_l = day["or_low"]
        sma = day["sma"]
        rsi = day["rsi"]
        co = day["cutoff_idx"]
        n = len(close)

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

            risk = a * atr_mult
            if direction == 1:
                sl, tp = c - risk, c + risk * rr_ratio
            else:
                sl, tp = c + risk, c - risk * rr_ratio

            exit_price = close[n - 1]
            exit_bar = n - 1

            if i + 1 < n:
                rem_h = high[i + 1 :]
                rem_l = low[i + 1 :]
                if direction == 1:
                    hit = (rem_l <= sl) | (rem_h >= tp)
                else:
                    hit = (rem_h >= sl) | (rem_l <= tp)

                if hit.any():
                    offset = int(np.argmax(hit))
                    exit_bar = i + 1 + offset
                    if direction == 1:
                        exit_price = sl if low[exit_bar] <= sl else tp
                    else:
                        exit_price = sl if high[exit_bar] >= sl else tp

            if direction == 1:
                ret = (exit_price - c) / c - ROUND_TRIP_COST
            else:
                ret = (c - exit_price) / c - ROUND_TRIP_COST

            # 1% risk geometry
            trade_profit = equity * 0.01 * (ret / (risk / c))
            equity += trade_profit
            equity_curve.append(equity)
            i = exit_bar + 1

    return np.array(equity_curve)


def main():
    import yfinance as yf

    cfgs = get_best_configs()
    print("Ticker | Max Drawdown (%) | Final Equity ($100k start)")
    print("-" * 60)
    for ticker, cfg in cfgs.items():
        df_5m = load_ticker_data(ticker)
        if df_5m.empty:
            continue

        df_1d = yf.download(
            ticker,
            start=df_5m.index.min() - timedelta(days=100),
            end=df_5m.index.max(),
            interval="1d",
            progress=False,
        )
        if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = df_1d.columns.droplevel(1)
        df_1d.index = df_1d.index.tz_localize(None)

        df_5m["ATR"] = calculate_atr(df_5m, 14)
        high_arr = df_5m["High"].values
        low_arr = df_5m["Low"].values
        close_arr = df_5m["Close"].values
        _, _, g_mid = _gaussian_channel_kernel(high_arr, low_arr, close_arr, 144.0, 4, 2.0)
        df_5m["Gauss_Mid"] = g_mid

        df_1d["SMA50"] = df_1d["Close"].rolling(50).mean().shift(1)
        df_1d["RSI14"] = calculate_rsi(df_1d["Close"], 14).shift(1)

        daily_data = precompute_daily(df_5m, df_1d, cfg["orb"], cfg["cutoff"])
        eq_curve = simulate_equity_curve(daily_data, cfg)

        peak = np.maximum.accumulate(eq_curve)
        drawdown = (eq_curve - peak) / peak
        max_dd = np.min(drawdown) * 100

        print(f"{ticker:<6} | {max_dd:>16.2f}% | ${eq_curve[-1]:,.2f}")


if __name__ == "__main__":
    main()
