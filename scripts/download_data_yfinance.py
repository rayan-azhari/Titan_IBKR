"""download_data_yfinance.py — Pull historical OHLCV from Yahoo Finance.

Downloads daily bars for ETF instruments using yfinance and writes
Parquet files to data/. Used as a longer-history supplement to Databento
(which only goes back to 2018-05-01 for ARCX.PILLAR).

SPY daily data available from 1993-01-29 (ETF inception).

Usage:
    uv run python scripts/download_data_yfinance.py
    uv run python scripts/download_data_yfinance.py --symbols SPY QQQ IWM
    uv run python scripts/download_data_yfinance.py --symbols SPY --start 2000-01-01

Output: data/{SYMBOL}_D.parquet  (DatetimeIndex UTC, columns: open/high/low/close/volume)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_SYMBOLS = ["SPY"]
DEFAULT_START = "1993-01-01"


def download_symbol(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance for one symbol.

    Args:
        symbol: Ticker (e.g. "SPY").
        start: ISO date string start of range.
        end: ISO date string end of range, or None for today.

    Returns:
        DataFrame with UTC DatetimeIndex and open/high/low/close/volume columns.
    """
    import yfinance as yf

    print(f"  Downloading {symbol} from Yahoo Finance ({start} to {end or 'today'}) ...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)

    if df.empty:
        print(f"  WARNING: No data returned for {symbol}.")
        return df

    # yfinance returns DatetimeIndex — normalize to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # Standardise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].copy()
    for col in df.columns:
        df[col] = df[col].astype(float)

    df = df.sort_index().dropna(how="all")
    print(f"  {symbol}: {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical daily OHLCV from Yahoo Finance -> data/*.parquet"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to download (default: SPY)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Start date ISO format (default: 1993-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date ISO format (default: today)",
    )
    args = parser.parse_args()

    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("ERROR: yfinance not installed. Run: uv add yfinance")
        sys.exit(1)

    print("=" * 60)
    print("  Yahoo Finance OHLCV Download")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Period:  {args.start} to {args.end or 'today'}")
    print("=" * 60)

    failed: list[str] = []
    for symbol in args.symbols:
        try:
            df = download_symbol(symbol, args.start, args.end)
            if df.empty:
                failed.append(symbol)
                continue
            out_path = DATA_DIR / f"{symbol}_D.parquet"
            df.to_parquet(out_path)
            print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"  ERROR downloading {symbol}: {exc}")
            failed.append(symbol)

    print("=" * 60)
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"  All {len(args.symbols)} symbol(s) downloaded successfully.")


if __name__ == "__main__":
    main()
