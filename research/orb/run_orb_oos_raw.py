"""Raw ORB IS/OOS Validation — no entry filters, pure breakout edge test.

Runs across all cached Databento tickers (90 S&P 500 stocks, 1-year 5m data).
No Gaussian, no SMA50 — just price breaking above/below the opening range.
Goal: determine if there is underlying ORB edge before layering filters back.

Entry: close breaks ORB high (long) or ORB low (short) after 09:45.
Exit: ATR stop (1.5x) or 2:1 take-profit, EOD flat.
IS/OOS: 70% IS / 30% OOS time-based split.
Acceptance: OOS Sharpe >= 0.5 AND OOS Return > 0.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore")

CACHE_DIR = "data/databento"
ATR_MULTIPLIER = 1.5
RR_RATIO = 2.0

TICKERS = [
    "AAPL",
    "ABBV",
    "ABT",
    "ADBE",
    "ADI",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "AXP",
    "BA",
    "BAC",
    "BDX",
    "BKNG",
    "BLK",
    "BSX",
    "C",
    "CAT",
    "CI",
    "CL",
    "CME",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DHR",
    "DUK",
    "ELV",
    "EOG",
    "FDX",
    "GE",
    "GILD",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "ICE",
    "INTC",
    "INTU",
    "ISRG",
    "JNJ",
    "JPM",
    "KKR",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDT",
    "META",
    "MMC",
    "MO",
    "MRK",
    "MSFT",
    "NEE",
    "NFLX",
    "NOW",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PGR",
    "PM",
    "PNC",
    "QCOM",
    "REGN",
    "RTX",
    "SLB",
    "SPGI",
    "STLD",
    "SYK",
    "T",
    "TJX",
    "TMO",
    "TMUS",
    "TXN",
    "TYL",
    "UNH",
    "UNP",
    "USB",
    "V",
    "VLO",
    "VRTX",
    "VTRS",
    "VZ",
    "WFC",
    "WMT",
    "XOM",
]


def load_cache(ticker: str) -> pd.DataFrame:
    cache_file = f"{CACHE_DIR}/{ticker}_1yr_5m.csv"
    if not os.path.exists(cache_file):
        return pd.DataFrame()
    df = pd.read_csv(cache_file, index_col=0, parse_dates=False)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(period).mean()


def run_raw_orb_oos(ticker: str) -> dict | None:
    df_5m = load_cache(ticker)
    if df_5m.empty or len(df_5m) < 500:
        return None

    df_5m["ATR"] = calculate_atr(df_5m, 14)

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
                risk = atr * ATR_MULTIPLIER

                if position == 0:
                    if close > or_high:
                        entries.loc[ts] = True
                        position = 1
                        current_sl = close - risk
                        current_tp = close + risk * RR_RATIO
                        continue
                    if close < or_low:
                        short_entries.loc[ts] = True
                        position = -1
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

    is_sh = safe(is_pf.sharpe_ratio())
    oos_sh = safe(oos_pf.sharpe_ratio())
    ratio = oos_sh / is_sh if is_sh > 0.01 else 0.0

    return {
        "ticker": ticker,
        "is_return": safe(is_pf.total_return() * 100),
        "is_win_rate": safe(is_pf.trades.win_rate() * 100) if is_pf.trades.count() > 0 else 0.0,
        "is_sharpe": is_sh,
        "is_trades": is_pf.trades.count(),
        "oos_return": safe(oos_pf.total_return() * 100),
        "oos_win_rate": safe(oos_pf.trades.win_rate() * 100) if oos_pf.trades.count() > 0 else 0.0,
        "oos_sharpe": oos_sh,
        "oos_trades": oos_pf.trades.count(),
        "ratio": ratio,
    }


if __name__ == "__main__":
    print("=" * 100)
    print(
        "ORB PHASE 4 — Raw ORB IS/OOS (No Filters) | Databento 1-Year True 5m | 90 S&P 500 Tickers"
    )
    print("Entry: price breaks ORB level | ATR stop: 1.5x | RR: 2:1 | No Gaussian, No SMA50")
    print("IS/OOS: 70%/30% time split | Acceptance: OOS Sharpe >= 0.5 AND OOS Return > 0")
    print("=" * 100)

    results = []
    for i, ticker in enumerate(TICKERS, 1):
        sys.stdout.write(f"\r  [{i:02d}/{len(TICKERS)}] {ticker:<6}...")
        sys.stdout.flush()
        try:
            res = run_raw_orb_oos(ticker)
            if res:
                results.append(res)
        except Exception as e:
            sys.stdout.write(f"\r  [error] {ticker}: {e}\n")

    print(f"\n\nProcessed {len(results)} tickers.")

    if not results:
        print("No results.")
        sys.exit(0)

    results.sort(key=lambda x: x["oos_sharpe"], reverse=True)

    print("\n" + "=" * 115)
    print("RESULTS (sorted by OOS Sharpe) — raw ORB, no filters")
    print("=" * 115)
    hdr = (
        f"{'Ticker':<6} | {'IS Ret':>8} | {'IS Win%':>8} | {'IS Sh':>6} | {'IS Tr':>5}"
        f" || {'OOS Ret':>8} | {'OOS Win%':>9} | {'OOS Sh':>7} | {'OOS Tr':>6}"
        f" | {'Ratio':>6} | Status"
    )
    print(hdr)
    print("-" * 115)

    valid_count = 0
    for r in results:
        ratio = r["ratio"]
        if r["oos_return"] < 0 or r["oos_sharpe"] <= 0:
            status = "FAIL"
        elif ratio >= 0.5:
            status = "VALID"
            valid_count += 1
        else:
            status = "WEAK"

        row = (
            f"{r['ticker']:<6} | {r['is_return']:>7.2f}% | {r['is_win_rate']:>7.2f}% |"
            f" {r['is_sharpe']:>6.2f} | {r['is_trades']:>5}"
            f" || {r['oos_return']:>7.2f}% | {r['oos_win_rate']:>8.2f}% |"
            f" {r['oos_sharpe']:>7.2f} | {r['oos_trades']:>6}"
            f" | {ratio:>6.2f} | {status}"
        )
        print(row)

    print("=" * 115)
    print(
        f"\nVALID: {valid_count} / {len(results)} tickers passed (OOS Sharpe >= 0.5, OOS Return > 0)"
    )

    os.makedirs(".tmp", exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = f".tmp/orb_oos_raw_{date_str}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    valid = [r for r in results if r["ratio"] >= 0.5 and r["oos_return"] > 0]
    if valid:
        print("\nTop VALID tickers (OOS/IS ratio >= 0.5):")
        for r in valid:
            print(
                f"  {r['ticker']:<6}  OOS Sharpe={r['oos_sharpe']:.2f}  "
                f"OOS Return={r['oos_return']:.2f}%  Win%={r['oos_win_rate']:.1f}%  "
                f"Ratio={r['ratio']:.2f}"
            )
