import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

# Suppress frequent yfinance warnings
warnings.filterwarnings("ignore")


def fetch_sp100_tickers() -> list[str]:
    # Hardcoded top 20 for speed in testing, can be expanded to full 100
    return [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
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
    ]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def run_advanced_orb(ticker: str, df_spy: pd.DataFrame, days: int = 59) -> dict | None:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    df_5m = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
    if df_5m.empty:
        return None

    df_1d = yf.download(
        ticker, start=start_date - timedelta(days=20), end=end_date, interval="1d", progress=False
    )
    if df_1d.empty:
        return None

    for d in [df_5m, df_1d]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)

    df_5m.index = df_5m.index.tz_convert("US/Eastern")
    df_1d.index = df_1d.index.tz_localize(None)

    # 1. Calculate 5m ATR
    df_5m["ATR"] = calculate_atr(df_5m, period=14)

    # 2. Calculate Daily Gaps
    df_1d["Prev_Close"] = df_1d["Close"].shift(1)
    df_1d["Gap_Pct"] = (df_1d["Open"] - df_1d["Prev_Close"]) / df_1d["Prev_Close"]

    # 3. Calculate Relative Strength vs SPY (Daily)
    df_1d["Daily_Ret"] = df_1d["Close"].pct_change()
    df_spy_1d = df_spy.copy()
    if isinstance(df_spy_1d.columns, pd.MultiIndex):
        df_spy_1d.columns = df_spy_1d.columns.droplevel(1)
    df_spy_1d["SPY_Ret"] = df_spy_1d["Close"].pct_change()

    # Align dates
    df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
    df_spy_1d["Date_Str"] = df_spy_1d.index.strftime("%Y-%m-%d")
    spy_ret_map = df_spy_1d.set_index("Date_Str")["SPY_Ret"].to_dict()
    df_1d["SPY_Ret"] = df_1d["Date_Str"].map(spy_ret_map)
    df_1d["RS"] = df_1d["Daily_Ret"] - df_1d["SPY_Ret"]  # Positive = Outperforming SPY

    # Map Daily context to 5m
    df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
    df_5m["Gap_Pct"] = df_5m["Date_Str"].map(df_1d.set_index("Date_Str")["Gap_Pct"].to_dict())
    df_5m["RS"] = df_5m["Date_Str"].map(
        df_1d.set_index("Date_Str")["RS"].shift(1).to_dict()
    )  # Use yesterday's RS

    df_5m = df_5m.between_time("09:30", "15:55").copy()

    # Fixed parameters for this screener
    atr_multiplier = 1.5
    reward_multiplier = 2.0
    time_decay_cutoff = "10:30"  # Ignore breakouts after 10:30 AM

    entries = pd.Series(False, index=df_5m.index)
    exits = pd.Series(False, index=df_5m.index)
    short_entries = pd.Series(False, index=df_5m.index)
    short_exits = pd.Series(False, index=df_5m.index)

    groups = df_5m.groupby(df_5m.index.date)

    for date, group in groups:
        if len(group) < 4:
            continue
        orb_bars = group.between_time("09:30", "09:40")
        if len(orb_bars) == 0:
            continue

        or_high = float(orb_bars["High"].max())
        or_low = float(orb_bars["Low"].min())

        # Get context for the day
        gap_pct = group["Gap_Pct"].iloc[0] if not pd.isna(group["Gap_Pct"].iloc[0]) else 0.0
        rs = group["RS"].iloc[0] if not pd.isna(group["RS"].iloc[0]) else 0.0

        trading_bars = group.between_time("09:45", "15:55")
        position = 0
        current_sl = 0.0
        current_tp = 0.0

        for ts, row in trading_bars.iterrows():
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else (or_high - or_low)
            time_str = ts.strftime("%H:%M")

            if position == 0:
                # Time Decay Filter: Only enter before 10:30
                if time_str > time_decay_cutoff:
                    continue

                # Gap & Go + RS Logic:
                # Only go long if gapped UP and outperforming SPY
                bull_context = gap_pct > 0.002 and rs > 0
                # Only go short if gapped DOWN and underperforming SPY
                bear_context = gap_pct < -0.002 and rs < 0

                if close > or_high and bull_context:
                    entries.loc[ts] = True
                    position = 1
                    initial_risk = atr * atr_multiplier
                    current_sl = close - initial_risk
                    current_tp = close + (initial_risk * reward_multiplier)
                    continue

                if close < or_low and bear_context:
                    short_entries.loc[ts] = True
                    position = -1
                    initial_risk = atr * atr_multiplier
                    current_sl = close + initial_risk
                    current_tp = close - (initial_risk * reward_multiplier)
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

    if pf.trades.count() > 0:
        return {
            "ticker": ticker,
            "return": pf.total_return() * 100,
            "sharpe": pf.sharpe_ratio() if not np.isnan(pf.sharpe_ratio()) else 0,
            "win_rate": pf.trades.win_rate() * 100,
            "trades": pf.trades.count(),
            "profit_factor": pf.trades.profit_factor()
            if not np.isnan(pf.trades.profit_factor())
            else 0,
        }
    return None


if __name__ == "__main__":
    days = 59
    tickers = fetch_sp100_tickers()

    print(f"Fetching {days} days of baseline SPY data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)
    df_spy = yf.download(
        "SPY", start=start_date - timedelta(days=20), end=end_date, interval="1d", progress=False
    )

    results = []
    print(f"\nScanning Top {len(tickers)} Mega-Caps with Advanced Institutional ORB...")
    print("Filters: 1.5 ATR Stop, 10:30 Time Decay, Gap & Go, Relative Strength vs SPY\n")

    for t in tickers:
        print(f"Testing {t}...")
        try:
            res = run_advanced_orb(t, df_spy, days)
            if res:
                results.append(res)
        except Exception as e:
            print(f"Error on {t}: {e}")

    print("\n" + "=" * 80)
    print("🏆 INSTITUTIONAL ORB SCREENER RESULTS (Ranked by Sharpe Ratio)")
    print("=" * 80)
    print(
        f"{'Ticker':<8} | {'Return':<8} | {'Win Rate':<10} | {'Trades':<8} | {'Profit Factor':<12} | {'Sharpe'}"
    )
    print("-" * 80)

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    for r in results:
        ret_str = f"{r['return']:.2f}%"
        win_str = f"{r['win_rate']:.2f}%"
        print(
            f"{r['ticker']:<8} | {ret_str:<8} | {win_str:<10} | {r['trades']:<8} | {r['profit_factor']:<12.2f} | {r['sharpe']:.2f}"
        )
    print("=" * 80)
