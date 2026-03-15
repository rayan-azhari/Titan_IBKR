"""Warrior Trading Stock Selection Filter + ORB Backtest.

Scans all US-listed stocks using Ross Cameron's 5 stock selection criteria,
then runs a VectorBT ORB backtest on survivors.

Warrior Trading Filters (filter 3 / news catalyst skipped — not automatable):
  1. Relative Volume >= 5x  (today's volume vs 30-day average)
  2. Up >= 10% on the day   (last close vs previous close)
  3. [SKIPPED] News catalyst
  4. Price $1.00 – $20.00
  5. Float < 10 million shares

Usage:
  uv run python research/orb/run_warrior_screener.py

Output:
  - Console: filter funnel + backtest results ranked by Sharpe
  - CSV:      .tmp/warrior_screener_YYYYMMDD.csv
"""

import io
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Warrior Trading Filter Thresholds ─────────────────────────────────────────
WARRIOR_PRICE_MIN: float = 1.0
WARRIOR_PRICE_MAX: float = 20.0
WARRIOR_UP_PCT: float = 0.10  # 10% gain on the day
WARRIOR_REL_VOL: float = 5.0  # 5x relative volume
WARRIOR_FLOAT_MAX: int = 10_000_000  # < 10 million shares float

# ── Script Config ──────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500  # tickers per yfinance batch download
BACKTEST_DAYS: int = 59  # lookback window for VectorBT backtest
ATR_MULTIPLIER: float = 1.5
REWARD_MULTIPLIER: float = 2.0
TIME_DECAY_CUTOFF: str = "10:30"  # no new entries after this time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TMP_DIR = PROJECT_ROOT / ".tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Fallback ticker list (S&P 100 subset) if NASDAQ FTP is unreachable
_FALLBACK_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "UNH",
    "WMT",
    "JPM",
    "V",
    "MA",
    "XOM",
    "JNJ",
    "PG",
    "HD",
    "ORCL",
    "AVGO",
    "LLY",
    "CAT",
    "AMAT",
    "TXN",
    "INTC",
    "CRM",
    "CSCO",
    "TMO",
    "BAC",
    "GS",
    "MS",
    "NFLX",
]


# ── Phase 1: Ticker Universe ───────────────────────────────────────────────────


def _ftp_fetch(filename: str) -> str | None:
    """Download a file from ftp.nasdaqtrader.com/symboldirectory/ via plain FTP."""
    import ftplib

    try:
        buf = io.BytesIO()
        with ftplib.FTP("ftp.nasdaqtrader.com", timeout=20) as ftp:
            ftp.login()
            ftp.retrbinary(f"RETR /symboldirectory/{filename}", buf.write)
        return buf.getvalue().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  FTP fetch of {filename} failed: {e}")
        return None


def fetch_all_us_tickers() -> list[str]:
    """Fetch all US common-stock tickers from NASDAQ public FTP directory.

    Returns combined NASDAQ + NYSE/AMEX ticker list, ETFs excluded.
    Falls back to a hardcoded S&P 100 subset if the FTP is unreachable.
    """
    tickers: set[str] = set()

    nasdaq_raw = _ftp_fetch("nasdaqlisted.txt")
    if nasdaq_raw:
        try:
            df = pd.read_csv(io.StringIO(nasdaq_raw), sep="|")
            df = df[df["ETF"] == "N"]
            raw = df["Symbol"].dropna().tolist()
            tickers.update(
                s
                for s in raw
                if isinstance(s, str) and s.replace("^", "").isalpha() and len(s) <= 5
            )
            print(f"  NASDAQ list: {len(tickers)} common stocks loaded")
        except Exception as e:
            print(f"  Warning: NASDAQ parse failed: {e}")

    other_raw = _ftp_fetch("otherlisted.txt")
    if other_raw:
        try:
            df = pd.read_csv(io.StringIO(other_raw), sep="|")
            df = df[df["ETF"] == "N"]
            raw = df["ACT Symbol"].dropna().tolist()
            before = len(tickers)
            tickers.update(
                s
                for s in raw
                if isinstance(s, str) and s.replace("-", "").isalpha() and len(s) <= 5
            )
            print(f"  NYSE/AMEX list: {len(tickers) - before} additional stocks loaded")
        except Exception as e:
            print(f"  Warning: Other listed parse failed: {e}")

    if not tickers:
        print("  Warning: FTP unreachable — using fallback S&P 100 subset")
        return _FALLBACK_TICKERS

    return sorted(tickers)


# ── Phase 2: Batch Download Daily Data ────────────────────────────────────────


def download_daily_chunks(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download 35 days of daily OHLCV for all tickers in chunks.

    Returns (close_df, volume_df) — columns are ticker symbols.
    """
    closes: list[pd.DataFrame] = []
    volumes: list[pd.DataFrame] = []
    total_chunks = (len(tickers) - 1) // CHUNK_SIZE + 1

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        print(f"  Batch download {chunk_num}/{total_chunks} ({len(chunk)} tickers)...")
        try:
            raw = yf.download(chunk, period="35d", interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                closes.append(raw["Close"])
                volumes.append(raw["Volume"])
            else:
                # Single ticker returned as flat DataFrame
                sym = chunk[0]
                closes.append(raw[["Close"]].rename(columns={"Close": sym}))
                volumes.append(raw[["Volume"]].rename(columns={"Volume": sym}))
        except Exception as e:
            print(f"  Warning: chunk {chunk_num} download error: {e}")

    if not closes:
        return pd.DataFrame(), pd.DataFrame()

    close_df = pd.concat(closes, axis=1)
    volume_df = pd.concat(volumes, axis=1)
    return close_df, volume_df


# ── Phase 3 & 4: Warrior Trading Filters ──────────────────────────────────────


def apply_price_volume_filters(
    tickers: list[str],
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> list[dict]:
    """Apply filters 1, 2, 4 (price, % change, relative volume).

    Uses the most recent completed trading day vs the prior 30-day average.
    Returns list of passing candidates with their filter metrics.
    """
    candidates = []

    for ticker in tickers:
        try:
            if ticker not in close_df.columns or ticker not in volume_df.columns:
                continue

            closes = close_df[ticker].dropna()
            volumes = volume_df[ticker].dropna()

            if len(closes) < 3 or len(volumes) < 3:
                continue

            last_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2])
            today_vol = float(volumes.iloc[-1])
            avg_vol = float(volumes.iloc[:-1].mean()) if len(volumes) > 1 else float(volumes.mean())

            if avg_vol == 0 or np.isnan(avg_vol):
                continue

            pct_change = (last_close - prev_close) / prev_close
            rel_vol = today_vol / avg_vol

            # Filter 4: Price $1–$20
            if not (WARRIOR_PRICE_MIN <= last_close <= WARRIOR_PRICE_MAX):
                continue

            # Filter 2: Up >= 10%
            if pct_change < WARRIOR_UP_PCT:
                continue

            # Filter 1: Relative Volume >= 5x
            if rel_vol < WARRIOR_REL_VOL:
                continue

            candidates.append(
                {
                    "ticker": ticker,
                    "price": last_close,
                    "pct_change": pct_change * 100,
                    "rel_vol": rel_vol,
                    "float_shares": None,
                }
            )
        except Exception:
            continue

    return candidates


def apply_float_filter(candidates: list[dict]) -> list[dict]:
    """Apply filter 5: float < 10 million shares.

    Fetches yfinance .info for each candidate (expensive — only runs on survivors
    of filters 1/2/4, which is a small set).
    """
    survivors = []
    for c in candidates:
        ticker = c["ticker"]
        try:
            info = yf.Ticker(ticker).info
            float_shares = info.get("floatShares") or info.get("sharesOutstanding")
            if float_shares is None:
                print(f"  {ticker}: float data unavailable, skipping")
                continue
            if int(float_shares) < WARRIOR_FLOAT_MAX:
                c["float_shares"] = int(float_shares)
                survivors.append(c)
        except Exception as e:
            print(f"  {ticker}: float fetch error: {e}")
    return survivors


# ── Phase 5: ORB VectorBT Backtest ────────────────────────────────────────────


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def run_orb_backtest(ticker: str, df_spy: pd.DataFrame) -> dict | None:
    """Run VectorBT ORB backtest on a single ticker.

    Mirrors run_orb_advanced_screener.py: Gap & Go + Relative Strength vs SPY
    context filters, 09:30–09:40 ORB window, 10:30 time decay cutoff,
    1.5× ATR stop, 2:1 RR target.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS - 1)

    try:
        df_5m = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
        if df_5m.empty:
            return None

        df_1d = yf.download(
            ticker,
            start=start_date - timedelta(days=20),
            end=end_date,
            interval="1d",
            progress=False,
        )
        if df_1d.empty:
            return None

        for d in [df_5m, df_1d]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)

        df_5m.index = df_5m.index.tz_convert("US/Eastern")
        df_1d.index = df_1d.index.tz_localize(None)

        df_5m["ATR"] = _calculate_atr(df_5m)
        df_1d["Gap_Pct"] = (df_1d["Open"] - df_1d["Close"].shift(1)) / df_1d["Close"].shift(1)
        df_1d["Daily_Ret"] = df_1d["Close"].pct_change()

        spy = df_spy.copy()
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.droplevel(1)
        spy["SPY_Ret"] = spy["Close"].pct_change()

        df_1d["Date_Str"] = df_1d.index.strftime("%Y-%m-%d")
        spy["Date_Str"] = spy.index.strftime("%Y-%m-%d")
        spy_ret_map = spy.set_index("Date_Str")["SPY_Ret"].to_dict()
        df_1d["SPY_Ret"] = df_1d["Date_Str"].map(spy_ret_map)
        df_1d["RS"] = df_1d["Daily_Ret"] - df_1d["SPY_Ret"]

        df_5m["Date_Str"] = df_5m.index.strftime("%Y-%m-%d")
        gap_map = df_1d.set_index("Date_Str")["Gap_Pct"].to_dict()
        rs_map = df_1d.set_index("Date_Str")["RS"].shift(1).to_dict()
        df_5m["Gap_Pct"] = df_5m["Date_Str"].map(gap_map)
        df_5m["RS"] = df_5m["Date_Str"].map(rs_map)

        df_5m = df_5m.between_time("09:30", "15:55").copy()

        entries = pd.Series(False, index=df_5m.index)
        exits = pd.Series(False, index=df_5m.index)
        short_entries = pd.Series(False, index=df_5m.index)
        short_exits = pd.Series(False, index=df_5m.index)

        for _date, group in df_5m.groupby(df_5m.index.date):
            if len(group) < 4:
                continue
            orb_bars = group.between_time("09:30", "09:40")
            if len(orb_bars) == 0:
                continue

            or_high = float(orb_bars["High"].max())
            or_low = float(orb_bars["Low"].min())
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

                if position == 0:
                    if ts.strftime("%H:%M") > TIME_DECAY_CUTOFF:
                        continue
                    bull_context = gap_pct > 0.002 and rs > 0
                    bear_context = gap_pct < -0.002 and rs < 0
                    risk = atr * ATR_MULTIPLIER

                    if close > or_high and bull_context:
                        entries.loc[ts] = True
                        position = 1
                        current_sl = close - risk
                        current_tp = close + risk * REWARD_MULTIPLIER
                    elif close < or_low and bear_context:
                        short_entries.loc[ts] = True
                        position = -1
                        current_sl = close + risk
                        current_tp = close - risk * REWARD_MULTIPLIER

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
            sharpe = pf.sharpe_ratio()
            pf_val = pf.trades.profit_factor()
            return {
                "ticker": ticker,
                "return_pct": pf.total_return() * 100,
                "sharpe": sharpe if not np.isnan(sharpe) else 0.0,
                "win_rate": pf.trades.win_rate() * 100,
                "trades": pf.trades.count(),
                "profit_factor": pf_val if not np.isnan(pf_val) else 0.0,
                "max_dd_pct": pf.max_drawdown() * 100,
            }

    except Exception as e:
        print(f"  Backtest error on {ticker}: {e}")

    return None


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("WARRIOR TRADING FILTER SCANNER + ORB BACKTEST")
    print("=" * 70)
    print("\nFilters applied:")
    print(f"  1. Relative Volume  >= {WARRIOR_REL_VOL}x")
    print(f"  2. Up on the day    >= {WARRIOR_UP_PCT * 100:.0f}%")
    print("  3. News catalyst       [SKIPPED — manual check]")
    print(f"  4. Price range         ${WARRIOR_PRICE_MIN:.2f} – ${WARRIOR_PRICE_MAX:.2f}")
    print(f"  5. Float               < {WARRIOR_FLOAT_MAX:,} shares")
    print()

    # ── Step 1: Universe ──────────────────────────────────────────────────────
    print("Step 1/5  Fetching US ticker universe...")
    all_tickers = fetch_all_us_tickers()
    print(f"  Total tickers: {len(all_tickers)}\n")

    # ── Step 2: Daily data ────────────────────────────────────────────────────
    print("Step 2/5  Downloading daily OHLCV data...")
    close_df, volume_df = download_daily_chunks(all_tickers)
    if close_df.empty:
        print("  ERROR: Could not download any daily data. Exiting.")
        return
    valid_tickers = [t for t in all_tickers if t in close_df.columns]
    print(f"  Data available for {len(valid_tickers)} tickers\n")

    # ── Step 3: Price / % gain / RelVol filters ───────────────────────────────
    print("Step 3/5  Applying price, % gain, and relative volume filters...")
    candidates = apply_price_volume_filters(valid_tickers, close_df, volume_df)
    price_pass = sum(
        1
        for t in valid_tickers
        if t in close_df.columns
        and not close_df[t].dropna().empty
        and WARRIOR_PRICE_MIN <= float(close_df[t].dropna().iloc[-1]) <= WARRIOR_PRICE_MAX
    )
    print(f"  After price ($1–$20):    {price_pass}")
    print(f"  After all 3 filters:     {len(candidates)}")
    if not candidates:
        print("\n  No stocks passed the quantitative filters today.")
        print("  This is expected — Warrior Trading filters target rare momentum events.")
        return
    print()

    # ── Step 4: Float filter ──────────────────────────────────────────────────
    print("Step 4/5  Fetching float data and applying < 10M filter...")
    survivors = apply_float_filter(candidates)
    print(f"  After float filter:      {len(survivors)} survivors")
    if not survivors:
        print("\n  No stocks survived the float filter.")
        print("  Warrior Trading criteria target very low-float small caps.")
        print("  Try running again on a high-volatility market day.\n")
        return
    print()
    print("  Survivors:")
    for s in survivors:
        print(
            f"    {s['ticker']:<8}  price={s['price']:.2f}  "
            f"+{s['pct_change']:.1f}%  relVol={s['rel_vol']:.1f}x  "
            f"float={s['float_shares']:,}"
        )
    print()

    # ── Step 5: ORB backtest on survivors ─────────────────────────────────────
    n_surv = len(survivors)
    print(f"Step 5/5  Running ORB backtest ({BACKTEST_DAYS}-day lookback) on {n_surv} survivors...")
    end_date = datetime.now()
    df_spy = yf.download(
        "SPY",
        start=end_date - timedelta(days=BACKTEST_DAYS + 20),
        end=end_date,
        interval="1d",
        progress=False,
    )

    backtest_results = []
    for s in survivors:
        ticker = s["ticker"]
        print(f"  Backtesting {ticker}...")
        result = run_orb_backtest(ticker, df_spy)
        if result:
            result["price"] = s["price"]
            result["pct_change"] = s["pct_change"]
            result["rel_vol"] = s["rel_vol"]
            result["float_shares"] = s["float_shares"]
            backtest_results.append(result)

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ORB BACKTEST RESULTS — Warrior Trading Filtered Universe (Ranked by Sharpe)")
    print("=" * 90)

    if not backtest_results:
        print("No backtest results (insufficient 5m data or no ORB signals generated).")
    else:
        backtest_results.sort(key=lambda x: x["sharpe"], reverse=True)
        header = (
            f"{'Ticker':<8} | {'Price':>6} | {'Day%':>6} | {'RelVol':>7} | "
            f"{'Float':>10} | {'Return%':>8} | {'WinRate':>8} | "
            f"{'Trades':>7} | {'PFactor':>8} | {'Sharpe':>7} | {'MaxDD%':>7}"
        )
        print(header)
        print("-" * 90)
        for r in backtest_results:
            print(
                f"{r['ticker']:<8} | {r['price']:>6.2f} | {r['pct_change']:>5.1f}% | "
                f"{r['rel_vol']:>6.1f}x | {r['float_shares']:>10,} | "
                f"{r['return_pct']:>7.2f}% | {r['win_rate']:>7.1f}% | "
                f"{r['trades']:>7} | {r['profit_factor']:>8.2f} | "
                f"{r['sharpe']:>7.2f} | {r['max_dd_pct']:>6.2f}%"
            )
        print("=" * 90)

        # Save CSV
        out_file = TMP_DIR / f"warrior_screener_{datetime.now().strftime('%Y%m%d')}.csv"
        pd.DataFrame(backtest_results).to_csv(out_file, index=False)
        print(f"\nResults saved to: {out_file}")

    print(
        "\nNote: Warrior Trading criteria target low-float small-cap momentum stocks."
        "\n      Surviving stocks should be manually reviewed for news catalysts"
        "\n      (filter 3) before considering them for live ORB trading."
    )


if __name__ == "__main__":
    main()
