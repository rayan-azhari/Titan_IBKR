"""download_data_databento.py — Pull historical OHLCV from Databento.

Downloads daily bars for ETF instruments and writes Parquet files to data/.
Primarily used for the ETF Trend strategy (SPY and future multi-asset expansion).

Usage:
    uv run python scripts/download_data_databento.py
    uv run python scripts/download_data_databento.py --symbols SPY QQQ IWM
    uv run python scripts/download_data_databento.py --symbols SPY --start 2000-01-01

Requires:
    DATABENTO_API_KEY in .env  (see .env.example)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Dataset for US equity daily bars.
# ARCX.PILLAR = NYSE ARCA (where SPY trades). Available from 2018-05-01.
# DBEQ.BASIC  = consolidated US equities but only from 2023-03-28.
DATABENTO_DATASET = "ARCX.PILLAR"

# Default universe — start with SPY, extend as strategy expands
DEFAULT_SYMBOLS = ["SPY"]
DEFAULT_START = "2018-05-01"

# Databento ohlcv-1d field mapping → our standard column names
COLUMN_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def download_symbol(
    client, symbol: str, start: str, end: str | None, dataset: str = DATABENTO_DATASET
) -> pd.DataFrame:
    """Download daily OHLCV for one symbol and return a clean DataFrame.

    Args:
        client: Authenticated databento.Historical client.
        symbol: Ticker symbol (e.g. "SPY").
        start: ISO date string (e.g. "1993-01-01").
        end: ISO date string or None for today.
        dataset: Databento dataset identifier.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume.
    """

    print(f"  Downloading {symbol} daily bars from {start} to {end or 'today'} ...")
    data = client.timeseries.get_range(
        dataset=dataset,
        schema="ohlcv-1d",
        symbols=[symbol],
        start=start,
        end=end,
        stype_in="raw_symbol",
    )
    df = data.to_df()

    if df.empty:
        print(f"  WARNING: No data returned for {symbol}.")
        return df

    # Normalize index to UTC DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Databento returns 'ts_event' as index for ohlcv schemas
        if "ts_event" in df.columns:
            df = df.set_index("ts_event")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index.name = "timestamp"

    # Keep only OHLCV columns, convert to float
    available = [c for c in COLUMN_MAP if c in df.columns]
    df = df[available].rename(columns=COLUMN_MAP)
    for col in df.columns:
        df[col] = df[col].astype(float)

    df = df.sort_index().dropna(how="all")
    print(f"  {symbol}: {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV from Databento → data/*.parquet"
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
    parser.add_argument(
        "--dataset",
        default=DATABENTO_DATASET,
        help=f"Databento dataset (default: {DATABENTO_DATASET})",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key or api_key.startswith("db-xxx"):
        print("ERROR: DATABENTO_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    try:
        import databento as db
    except ImportError:
        print("ERROR: databento not installed. Run: uv add databento")
        sys.exit(1)

    client = db.Historical(key=api_key)

    print("=" * 60)
    print("  Databento OHLCV Download")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Symbols:  {', '.join(args.symbols)}")
    print(f"  Period:   {args.start} to {args.end or 'today'}")
    print("=" * 60)

    failed: list[str] = []
    for symbol in args.symbols:
        try:
            df = download_symbol(client, symbol, args.start, args.end, dataset=args.dataset)
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
