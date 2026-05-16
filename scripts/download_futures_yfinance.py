"""download_futures_yfinance.py -- Pull M1 (front-month) continuous-contract
daily OHLCV for the BGR-2019 commodity universe from Yahoo Finance.

Yahoo's continuous-contract symbology uses the ``=F`` suffix:

    CL=F, GC=F, ZC=F, KC=F, ...

This is the fallback path when Databento GLBX.MDP3 is unresponsive
(see scripts/download_futures_databento.py for the primary path that
hangs on ``.c.1`` queries -- 2026-05-15 incident).

Yahoo provides M1 only; carry signal must use the BGR §3.2 rolling-yield
proxy (12-month return) instead of the strict M1/M2 basis. Document this
in the D2 pre-reg amendment.

Usage::

    uv run python scripts/download_futures_yfinance.py
    uv run python scripts/download_futures_yfinance.py --start 2010-01-01

Output:
    data/{ROOT}_M1_D.parquet -- columns open/high/low/close/volume, naive
    DatetimeIndex (UTC stripped + normalised to date).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_START = "2000-01-01"


@dataclass(frozen=True)
class FuturesRoot:
    root: str  # local save name (e.g. CL)
    yahoo: str  # Yahoo Finance ticker (e.g. CL=F)
    name: str
    sector: str


UNIVERSE: tuple[FuturesRoot, ...] = (
    # Energy
    FuturesRoot("CL", "CL=F", "WTI Crude Oil", "energy"),
    FuturesRoot("NG", "NG=F", "Natural Gas", "energy"),
    FuturesRoot("HO", "HO=F", "Heating Oil", "energy"),
    FuturesRoot("RB", "RB=F", "RBOB Gasoline", "energy"),
    FuturesRoot("BZ", "BZ=F", "Brent Crude Oil", "energy"),
    # Metals -- precious
    FuturesRoot("GC", "GC=F", "Gold", "metals_precious"),
    FuturesRoot("SI", "SI=F", "Silver", "metals_precious"),
    FuturesRoot("PL", "PL=F", "Platinum", "metals_precious"),
    FuturesRoot("PA", "PA=F", "Palladium", "metals_precious"),
    # Metals -- industrial
    FuturesRoot("HG", "HG=F", "Copper", "metals_industrial"),
    # Grains + oilseeds
    FuturesRoot("ZC", "ZC=F", "Corn", "grains"),
    FuturesRoot("ZW", "ZW=F", "Chicago Wheat", "grains"),
    FuturesRoot("ZS", "ZS=F", "Soybeans", "grains"),
    FuturesRoot("ZL", "ZL=F", "Soybean Oil", "grains"),
    FuturesRoot("ZM", "ZM=F", "Soybean Meal", "grains"),
    FuturesRoot("ZO", "ZO=F", "Oats", "grains"),
    # Livestock
    FuturesRoot("LE", "LE=F", "Live Cattle", "livestock"),
    FuturesRoot("GF", "GF=F", "Feeder Cattle", "livestock"),
    FuturesRoot("HE", "HE=F", "Lean Hogs", "livestock"),
    # Softs
    FuturesRoot("KC", "KC=F", "Coffee", "softs"),
    FuturesRoot("CC", "CC=F", "Cocoa", "softs"),
    FuturesRoot("SB", "SB=F", "Sugar No. 11", "softs"),
    FuturesRoot("CT", "CT=F", "Cotton No. 2", "softs"),
    FuturesRoot("OJ", "OJ=F", "Orange Juice", "softs"),
)


def download_one(yf, fr: FuturesRoot, start: str, end: str | None) -> pd.DataFrame:
    print(f"[{fr.root}] {fr.name}  yahoo={fr.yahoo}  sector={fr.sector}")
    df = yf.download(fr.yahoo, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        print("    (no rows)")
        return df
    # yfinance returns a MultiIndex on columns when one ticker is requested.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Rename to our schema.
    cols_lc = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lc)
    # Drop "adj_close" if present; we use raw OHLC for futures (no
    # corporate-action adjustments apply).
    if "adj_close" in df.columns:
        df = df.drop(columns=["adj_close"])
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype(float).sort_index().dropna(how="all")
    # Normalise index.
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "timestamp"
    print(f"    -> {len(df)} bars  {df.index[0].date()} .. {df.index[-1].date()}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--roots", default="", help="Comma-separated subset (default: full 24)")
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed.")
        return 1

    selected = (
        {r.strip().upper() for r in args.roots.split(",") if r.strip()} if args.roots else None
    )
    todo = [fr for fr in UNIVERSE if selected is None or fr.root in selected]

    print("=" * 70)
    print(
        f"  yfinance futures download: {len(todo)} commodities, "
        f"{args.start} -> {args.end or 'today'}"
    )
    print("=" * 70)

    succeeded: list[str] = []
    failed: list[str] = []
    for fr in todo:
        try:
            df = download_one(yf, fr, args.start, args.end)
            if df.empty:
                failed.append(fr.root)
                continue
            out = DATA_DIR / f"{fr.root}_M1_D.parquet"
            df.to_parquet(out)
            print(f"    saved: {out.relative_to(PROJECT_ROOT)}")
            succeeded.append(fr.root)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            failed.append(fr.root)

    print("\n" + "=" * 70)
    print(f"  Succeeded: {len(succeeded):>2}  {' '.join(succeeded)}")
    print(f"  Failed:    {len(failed):>2}  {' '.join(failed)}")
    print("=" * 70)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
