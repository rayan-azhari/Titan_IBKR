"""download_f2_commodities.py -- yfinance continuous-contract daily download
for the F2 audit (CFTC CoT positioning) universe.

Saves to ``data/{SYM}_F_D.parquet`` (distinct from existing
`{SYM}_M1_stitched_D.parquet` which is the IBKR-stitched 3y data).

Pre-reg: `directives/Pre-Reg F2 CFTC CoT Positioning 2026-05-20.md` §6 step 2.

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/download_f2_commodities.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Map our internal symbol -> Yahoo continuous-contract ticker.
# Per pre-reg §1: 15-commodity universe (intersection of yfinance + CFTC).
COMMODITY_TICKERS: dict[str, str] = {
    # Energy
    "CL": "CL=F",  # WTI crude
    "BZ": "BZ=F",  # Brent crude
    "NG": "NG=F",  # Natural gas
    # Metals
    "GC": "GC=F",  # Gold
    "SI": "SI=F",  # Silver
    "HG": "HG=F",  # Copper
    "PL": "PL=F",  # Platinum
    "PA": "PA=F",  # Palladium
    # Grains
    "ZC": "ZC=F",  # Corn
    "ZW": "ZW=F",  # Wheat
    "ZS": "ZS=F",  # Soybeans
    # Softs
    "CT": "CT=F",  # Cotton
    "KC": "KC=F",  # Coffee
    "SB": "SB=F",  # Sugar
    "CC": "CC=F",  # Cocoa
}

REQ_PAUSE_S = 1.0
START = "2005-01-01"  # comfortably before the 2006-06-13 CoT-data start
END = None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols_lc = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lc)
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
        print("ERROR: yfinance not installed.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  F2 commodity-futures download: {len(COMMODITY_TICKERS)} tickers")
    print(f"  Window: {START} -> {'latest' if END is None else END}")
    print(f"  Output: {DATA_DIR}/{{SYM}}_F_D.parquet")
    print("=" * 72)

    n_ok = 0
    for sym, yahoo in COMMODITY_TICKERS.items():
        print(f"  {sym} ({yahoo})", end=": ")
        try:
            df = yf.download(yahoo, start=START, end=END, progress=False, auto_adjust=True)
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
        out = DATA_DIR / f"{sym}_F_D.parquet"
        df.to_parquet(out)
        print(f"{len(df)} bars  {df.index[0].date()} .. {df.index[-1].date()}")
        n_ok += 1
        time.sleep(REQ_PAUSE_S)

    print()
    print(f"  Downloaded: {n_ok}/{len(COMMODITY_TICKERS)}")
    return 0 if n_ok == len(COMMODITY_TICKERS) else 2


if __name__ == "__main__":
    sys.exit(main())
