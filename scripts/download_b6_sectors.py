"""download_b6_sectors.py -- yfinance download of 9 SPDR Select Sector ETFs.

Pulls daily total-return-adjusted (auto_adjust=True) close for the B6
audit universe (see `directives/Pre-Reg B6 Sector-Momentum Crash-Hedge
2026-05-19.md` §1, §6 step 2). XLRE excluded (2015+ inception; too short
for full-window audit).

Saves to ``data/{TICKER}_D.parquet`` matching the existing convention.

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/download_b6_sectors.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SECTOR_TICKERS = ["XLF", "XLK", "XLE", "XLU", "XLY", "XLP", "XLI", "XLB", "XLV"]

REQ_PAUSE_S = 1.0  # courtesy pause between yfinance calls
START = "1998-12-22"  # SPDR Select Sector ETFs launched Dec 1998
END = None  # latest


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols_lc = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lc)
    # We use auto_adjust=True so the 'close' column is already adjusted.
    # Drop 'adj_close' if it exists.
    if "adj_close" in df.columns:
        df = df.drop(columns=["adj_close"])
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype(float).sort_index().dropna(how="all")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "timestamp"
    return df


def main() -> int:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. uv add yfinance && uv sync.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  B6 sector-ETF download: {len(SECTOR_TICKERS)} tickers")
    print(f"  Window: {START} -> {'latest' if END is None else END}")
    print(f"  Output: {DATA_DIR}/{{TICKER}}_D.parquet")
    print("=" * 72)

    n_ok = 0
    for tkr in SECTOR_TICKERS:
        print(f"  {tkr}", end=": ")
        try:
            df = yf.download(
                tkr,
                start=START,
                end=END,
                progress=False,
                auto_adjust=True,  # total-return adjusted
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {exc}")
            continue
        if df is None or df.empty:
            print("(empty)")
            continue
        df = _normalize_df(df)
        if df.empty:
            print("(empty after normalize)")
            continue
        out = DATA_DIR / f"{tkr}_D.parquet"
        df.to_parquet(out)
        print(f"{len(df)} bars  {df.index[0].date()} .. {df.index[-1].date()}")
        n_ok += 1
        time.sleep(REQ_PAUSE_S)

    print()
    print(f"  Downloaded: {n_ok}/{len(SECTOR_TICKERS)}")
    return 0 if n_ok == len(SECTOR_TICKERS) else 2


if __name__ == "__main__":
    sys.exit(main())
