"""download_sp500_universe.py -- Pull daily OHLCV for ~500 S&P 500
constituents from yfinance.

Output:
    data/SP500_universe/{TICKER}_D.parquet  -- one file per stock
    data/SP500_universe/_constituents_list.txt -- the ticker list used

Universe source: Wikipedia's "List of S&P 500 companies" via pandas.read_html.
Acknowledged survivorship bias (current constituents only) per pre-reg §5.

Usage::

    uv run python scripts/download_sp500_universe.py
    uv run python scripts/download_sp500_universe.py --start 2010-01-01
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "SP500_universe"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "2000-01-01"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# yfinance uses '-' for share classes (e.g. BRK.B -> BRK-B).
TICKER_NORMALISE = {".": "-"}


def fetch_constituent_list() -> list[str]:
    """Scrape the current S&P 500 ticker list from Wikipedia.

    Wikipedia returns HTTP 403 to the default urllib User-Agent, so we
    fetch the page ourselves with a browser-like UA and hand the HTML
    bytes to pandas.read_html.
    """
    print(f"Fetching constituent list from {WIKIPEDIA_URL} ...")
    req = urllib.request.Request(
        WIKIPEDIA_URL,
        headers={"User-Agent": "Mozilla/5.0 (titan-ibkr/A1 research)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    # The first table on the page is the constituents table.
    df = tables[0]
    if "Symbol" not in df.columns:
        # Fallback: search all tables for one with a "Symbol" column.
        for t in tables:
            if "Symbol" in t.columns:
                df = t
                break
    raw = df["Symbol"].astype(str).str.strip().tolist()
    # Normalise tickers to yfinance format.
    out = []
    for t in raw:
        for src, dst in TICKER_NORMALISE.items():
            t = t.replace(src, dst)
        out.append(t)
    print(f"  -> {len(out)} tickers")
    return out


def download_batch(yf, tickers: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    """Bulk download N tickers via yfinance; split per-ticker DataFrames."""
    print(f"  batch download {len(tickers)} tickers ...")
    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t].copy()
                # Drop empty (sometimes yfinance returns NaN rows for non-existent tickers).
                sub = sub.dropna(how="all")
                if not sub.empty:
                    out[t] = sub
    else:
        # Single-ticker case (rare in bulk mode but handled).
        if len(tickers) == 1:
            out[tickers[0]] = df.dropna(how="all")
    return out


def save_one(ticker: str, df: pd.DataFrame, root: Path) -> bool:
    """Normalise + save one ticker's DataFrame to parquet.

    Uses ``adj_close`` (yfinance total return — back-adjusted for splits and
    dividends) as the canonical "close" column. The raw price-only close
    is dropped because cross-sectional equity strategies care about TOTAL
    return rank — dividend-paying stocks are systematically under-ranked
    by price-only momentum, which biases the cross-section non-trivially
    (BHM 2011 fix: use CRSP total returns).
    """
    cols_lc = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lc)
    # Promote adj_close -> close (drops the price-only close).
    if "adj_close" in df.columns:
        df = df.drop(columns=["close"], errors="ignore")
        df = df.rename(columns={"adj_close": "close"})
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if not keep:
        return False
    df = df[keep].astype(float).sort_index().dropna(how="all")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "timestamp"
    if df.empty:
        return False
    out_path = root / f"{ticker}_D.parquet"
    df.to_parquet(out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="How many tickers per yfinance bulk download call (default 50).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debug: only download the first N tickers.",
    )
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed.")
        return 1

    tickers = fetch_constituent_list()
    if args.limit:
        tickers = tickers[: args.limit]
        print(f"  (debug: limited to first {args.limit})")

    # Persist the ticker list for reproducibility / audit.
    (DATA_DIR / "_constituents_list.txt").write_text("\n".join(tickers), encoding="utf-8")

    print(
        f"\nDownloading {len(tickers)} tickers in batches of {args.batch_size} "
        f"from {args.start} to {args.end or 'today'} ..."
    )

    succeeded: list[str] = []
    failed: list[str] = []

    for i in range(0, len(tickers), args.batch_size):
        batch = tickers[i : i + args.batch_size]
        print(
            f"\n[batch {i // args.batch_size + 1}/{(len(tickers) + args.batch_size - 1) // args.batch_size}]"
        )
        try:
            results = download_batch(yf, batch, args.start, args.end)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.extend(batch)
            continue
        for t in batch:
            if t in results and not results[t].empty:
                if save_one(t, results[t], DATA_DIR):
                    succeeded.append(t)
                else:
                    failed.append(t)
            else:
                failed.append(t)
        # Gentle pause between batches to be polite to Yahoo.
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"  Succeeded: {len(succeeded):>4}")
    print(f"  Failed:    {len(failed):>4}")
    if failed:
        print(f"  Failed tickers (first 20): {failed[:20]}")
    print("=" * 70)
    return 0 if not failed else (0 if len(succeeded) >= len(tickers) * 0.9 else 2)


if __name__ == "__main__":
    sys.exit(main())
