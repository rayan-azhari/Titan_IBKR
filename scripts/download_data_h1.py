"""download_data_h1.py -- Download 10-year hourly OHLCV from Databento for S&P 100.

Routes each symbol to its primary exchange dataset:
  - NASDAQ stocks  ->  XNAS.ITCH    (available from ~2015)
  - NYSE stocks    ->  XNYS.PILLAR  (available from ~2015)

Output: data/{SYMBOL}_H1.parquet  (DatetimeIndex UTC, open/high/low/close/volume)

Usage:
    uv run python scripts/download_data_h1.py
    uv run python scripts/download_data_h1.py --symbols AAPL MSFT GOOGL
    uv run python scripts/download_data_h1.py --start 2018-01-01
    uv run python scripts/download_data_h1.py --dry-run

Requires:
    DATABENTO_API_KEY in .env
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_START = "2018-05-01"  # earliest available on XNAS.ITCH / XNYS.PILLAR
SCHEMA = "ohlcv-1h"

# ── Exchange routing ───────────────────────────────────────────────────────────
# Primary-exchange Databento dataset for each S&P 100 component.
# XNAS.ITCH  = NASDAQ TotalView  (~2015-present)
# XNYS.PILLAR = NYSE Pillar      (~2015-present)

NASDAQ_ITCH: list[str] = [
    "AAPL",
    "ADBE",
    "ADI",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "BKNG",
    "CME",
    "COST",
    "CSCO",
    "GILD",
    "GOOGL",
    "INTU",
    "INTC",
    "ISRG",
    "META",
    "MSFT",
    "NFLX",
    "NOW",
    "NVDA",
    "QCOM",
    "REGN",
    "SBUX",
    "TMUS",
    "TXN",
    "VRTX",
    "ADP",
    "AVGO",
    "TSLA",
    "PYPL",
    "KLAC",
    "LRCX",
    "MU",
    "CDNS",
    "SNPS",
    "PANW",
    "ORLY",
    "FAST",
    "PAYX",
]

XNYS_PILLAR: list[str] = [
    "ABBV",
    "ABT",
    "ACN",
    "AFL",
    "APD",
    "AXP",
    "BA",
    "BAC",
    "BDX",
    "BLK",
    "BSX",
    "C",
    "CAT",
    "CB",
    "CI",
    "CL",
    "COP",
    "CRM",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "ELV",
    "EMR",
    "EOG",
    "ETN",
    "FDX",
    "GD",
    "GE",
    "GS",
    "HCA",
    "HD",
    "HON",
    "IBM",
    "ICE",
    "ITW",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MCO",
    "MDT",
    "MMC",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "NEE",
    "NOC",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PGR",
    "PM",
    "PLD",
    "PNC",
    "RTX",
    "SCHW",
    "SLB",
    "SO",
    "SPGI",
    "SYK",
    "T",
    "TJX",
    "TMO",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WFC",
    "WM",
    "WMT",
    "XOM",
    "ZTS",
    "EW",
    "HUM",
    "NEM",
    "OXY",
]

# Build routing lookup  {symbol -> dataset}
SYMBOL_DATASET: dict[str, str] = {
    **{s: "XNAS.ITCH" for s in NASDAQ_ITCH},
    **{s: "XNYS.PILLAR" for s in XNYS_PILLAR},
}

# Full default universe (deduplicated, preserving order)
_seen: set[str] = set()
DEFAULT_SYMBOLS: list[str] = []
for _s in NASDAQ_ITCH + XNYS_PILLAR:
    if _s not in _seen:
        _seen.add(_s)
        DEFAULT_SYMBOLS.append(_s)


# ── Column map ────────────────────────────────────────────────────────────────

COLUMN_MAP = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}


# ── Download single symbol ─────────────────────────────────────────────────────


def download_symbol(
    client,
    symbol: str,
    dataset: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    print(f"  [{dataset}] {symbol}: {start} -> {end or 'today'} ...")
    data = client.timeseries.get_range(
        dataset=dataset,
        schema=SCHEMA,
        symbols=[symbol],
        start=start,
        end=end,
        stype_in="raw_symbol",
    )
    df = data.to_df()

    if df.empty:
        print("    WARNING: no data returned.")
        return df

    # Normalise timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df = df.set_index("ts_event")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"

    available = [c for c in COLUMN_MAP if c in df.columns]
    df = df[available].rename(columns=COLUMN_MAP)
    for col in df.columns:
        df[col] = df[col].astype(float)

    df = df.sort_index().dropna(how="all")
    print(f"    {len(df):,} hourly bars  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 10-year S&P 100 hourly OHLCV from Databento"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=f"Symbols to download (default: all {len(DEFAULT_SYMBOLS)} S&P 100 symbols)",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START, help=f"Start date ISO (default: {DEFAULT_START})"
    )
    parser.add_argument("--end", default=str(date.today()), help="End date ISO (default: today)")
    parser.add_argument(
        "--dataset", default=None, help="Override dataset for all symbols (e.g. XNAS.ITCH)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print routing table without downloading"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip symbols that already have a parquet file in data/",
    )
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key or api_key.startswith("db-xxx"):
        print("ERROR: DATABENTO_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 68)
    print("  Databento Hourly (H1) Download — S&P 100")
    print(f"  Schema   : {SCHEMA}")
    print(f"  Period   : {args.start} -> {args.end or 'today'}")
    print(f"  Symbols  : {len(symbols)}")
    print("=" * 68)

    # Routing summary
    routing: dict[str, list[str]] = {}
    for sym in symbols:
        ds = args.dataset or SYMBOL_DATASET.get(sym)
        if ds is None:
            print(f"  WARNING: {sym} not in routing table — defaulting to XNYS.PILLAR")
            ds = "XNYS.PILLAR"
        routing.setdefault(ds, []).append(sym)

    for ds, syms in sorted(routing.items()):
        print(f"  {ds}: {', '.join(syms)}")
    print("=" * 68)

    if args.dry_run:
        print("  [DRY RUN] No downloads performed.")
        return

    try:
        import databento as db
    except ImportError:
        print("ERROR: databento not installed. Run: uv add databento")
        sys.exit(1)

    client = db.Historical(key=api_key)

    failed: list[str] = []
    skipped: list[str] = []

    for i, sym in enumerate(symbols, 1):
        out_path = DATA_DIR / f"{sym}_H1.parquet"

        if args.skip_existing and out_path.exists():
            print(f"  [{i}/{len(symbols)}] {sym}: SKIP (already exists)")
            skipped.append(sym)
            continue

        ds = args.dataset or SYMBOL_DATASET.get(sym, "XNYS.PILLAR")
        print(f"\n  [{i}/{len(symbols)}] {sym}")
        try:
            df = download_symbol(client, sym, ds, args.start, args.end)
            if df.empty:
                # Retry on the other exchange
                fallback = "XNAS.ITCH" if ds == "XNYS.PILLAR" else "XNYS.PILLAR"
                print(f"    Retrying on {fallback} ...")
                df = download_symbol(client, sym, fallback, args.start, args.end)

            if df.empty:
                print("    FAILED: no data from either exchange.")
                failed.append(sym)
                continue

            df.to_parquet(out_path)
            print(f"    Saved: {out_path.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            failed.append(sym)

    print("\n" + "=" * 68)
    print(f"  Downloaded : {len(symbols) - len(failed) - len(skipped)}")
    print(f"  Skipped    : {len(skipped)}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"  Failed list: {', '.join(failed)}")
    print("=" * 68)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
