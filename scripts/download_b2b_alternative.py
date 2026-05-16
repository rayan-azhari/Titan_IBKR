"""download_b2b_alternative.py -- yfinance-based B2b universe expansion.

After IG's weekly quota (L47) blocked the original IG-data path for the
B2b Carver EWMAC universe expansion, this script pulls equivalent
cross-asset proxies from yfinance — which has no quota, no rate-limit
beyond reasonable use, and goes back 15-25 years on most symbols.

Critical data-construction caveats vs IG/IBKR per-contract:
    - **Bond ETFs are NOT bond futures.** TLT/IEF have credit duration
      structure but the *return* over time tracks the bond cash basket
      minus a tiny expense ratio. For trend-following the sign-of-signal
      is what matters, so ETFs work as proxies. Carver-style position
      sizing uses %vol so the contract-multiplier mismatch is absorbed.
    - **FX spot pairs are clean.** `EURUSD=X` is the daily spot rate; no
      rollover, no L40 risk.
    - **Equity index ETFs are clean.** SPY/QQQ etc track the underlying
      cash index; no futures roll.
    - **Commodity =F symbols are excluded** (L40 contamination). Commodity
      ETFs like GLD/USO have their own roll-drag issues (USO famously
      lost ~30%/yr in contango regimes) so we use them only for metals
      where the ETF is physical (GLD, SLV).

Run via::

    uv run python scripts/download_b2b_alternative.py
    uv run python scripts/download_b2b_alternative.py --sleeves bond,fx_major

Output: ``data/yf_b2b/{LABEL}_DAY.parquet`` — separate dir to avoid
colliding with IG (`data/ig_markets/`) or IBKR-stitched (`data/`) data.
The labels mirror B2b's UNIVERSE so `run_b2b_audit.py` can ingest both
sources via a flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "yf_b2b"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Inter-symbol pause -- yfinance rate-limits at ~2k requests/hour.
REQ_PAUSE_S = 1.0

# Mapping: local-label -> Yahoo ticker. Carefully curated to avoid the
# L40 commodity =F contamination AND to prefer ETF proxies that have no
# roll-yield artifacts.
UNIVERSE: dict[str, dict[str, str]] = {
    # Equity index ETFs / cash index — clean, deep history.
    "equity_index": {
        "SPX": "SPY",  # SPDR S&P 500 ETF, 1993+
        "NDX": "QQQ",  # Invesco QQQ, 1999+
        "DJI": "DIA",  # SPDR Dow, 1998+
        "RUT": "IWM",  # iShares Russell 2000, 2000+
        "FTSE": "^FTSE",  # FTSE 100 index, 1984+
        "DAX": "^GDAXI",  # DAX index, 1987+
        "NIKKEI": "^N225",  # Nikkei 225 index, 1965+
        "EUROSTOXX": "^STOXX50E",  # Euro Stoxx 50 index, 1990+
    },
    # FX spot pairs — clean, no roll. Yahoo ticker convention =X suffix.
    "fx_major": {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "NZDUSD": "NZDUSD=X",
        "DXY": "DX-Y.NYB",  # US Dollar Index
    },
    # Bond ETFs as proxies for bond futures. Carver's EWMAC uses %-vol
    # sizing so the contract-vs-ETF unit mismatch is absorbed by the
    # vol-normalisation. Sign-of-trend is the load-bearing quantity.
    "bond_etf": {
        "US10Y_PROXY": "IEF",  # iShares 7-10y Treasury, 2002+
        "US30Y_PROXY": "TLT",  # iShares 20+y Treasury, 2002+
        "US2Y_PROXY": "SHY",  # iShares 1-3y Treasury, 2002+
        "UK_GILT_PROXY": "IGLT.L",  # iShares UK Gilts UCITS (LSE), 2006+
        "EURO_GOV_PROXY": "IBGS.L",  # iShares Euro Govt Bond UCITS (LSE)
        "EM_BOND_PROXY": "EMB",  # iShares EM Sovereign Bond, 2007+
    },
    # Physical-precious commodity ETFs — no roll because they hold
    # physical bullion. GLD/SLV are clean. NOT including USO (oil ETF
    # with severe contango drag) or DBA/DBC (broad commodity baskets
    # that include front-month futures with the L40 issue).
    "physical_commodity_etf": {
        "GOLD_PROXY": "GLD",  # SPDR Gold Shares, 2004+
        "SILVER_PROXY": "SLV",  # iShares Silver Trust, 2006+
        "PLATINUM_PROXY": "PPLT",  # Aberdeen Platinum, 2010+
        "PALLADIUM_PROXY": "PALL",  # Aberdeen Palladium, 2010+
    },
    # Sector/regional equity ETFs for cross-sectional breadth.
    "regional_equity": {
        "EM_EQUITY": "EEM",  # iShares Emerging Markets, 2003+
        "EAFE_EQUITY": "EFA",  # iShares MSCI EAFE, 2001+
        "EUROPE_EQUITY": "VGK",  # Vanguard FTSE Europe, 2005+
        "JAPAN_EQUITY": "EWJ",  # iShares MSCI Japan, 1996+
        "CHINA_EQUITY": "FXI",  # iShares China Large-Cap, 2004+
    },
}


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


def download_one(yf, label: str, yahoo_ticker: str, start: str, end: str | None) -> int:
    """Returns bar count saved, 0 on failure."""
    print(f"  {label:<22} yahoo={yahoo_ticker}", end="  ")
    try:
        df = yf.download(yahoo_ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 0
    if df is None or df.empty:
        print("(no rows)")
        return 0
    df = _normalize_df(df)
    if df.empty:
        print("(empty after normalize)")
        return 0
    out = DATA_DIR / f"{label}_DAY.parquet"
    df.to_parquet(out)
    print(f"{len(df)} bars  {df.index[0].date()} .. {df.index[-1].date()}")
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sleeves",
        default="all",
        help=f"Comma-separated sleeves. Valid: {','.join(UNIVERSE.keys())}",
    )
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--instruments",
        default=None,
        help="Comma-separated label filter (e.g. SPX,EURUSD)",
    )
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed.")
        return 1

    if args.sleeves.lower() == "all":
        sleeves = list(UNIVERSE.keys())
    else:
        sleeves = [s.strip() for s in args.sleeves.split(",") if s.strip()]
        bad = [s for s in sleeves if s not in UNIVERSE]
        if bad:
            print(f"Unknown sleeves: {bad}. Valid: {list(UNIVERSE.keys())}")
            return 2

    instrument_filter: set[str] | None = (
        {s.strip() for s in args.instruments.split(",")} if args.instruments else None
    )

    todo: list[tuple[str, str, str]] = []
    for sleeve in sleeves:
        for label, yahoo_ticker in UNIVERSE[sleeve].items():
            if instrument_filter and label not in instrument_filter:
                continue
            todo.append((sleeve, label, yahoo_ticker))
    print("=" * 70)
    print(f"  B2b yfinance alternative download: {len(todo)} instruments")
    print(f"  Sleeves: {sleeves}")
    print(f"  Range: {args.start} -> {args.end or 'today'}")
    print("=" * 70)

    succeeded: list[tuple[str, str, int]] = []
    failed: list[tuple[str, str]] = []
    last_sleeve = None
    for sleeve, label, yahoo_ticker in todo:
        if sleeve != last_sleeve:
            print(f"\n=== sleeve {sleeve} ===")
            last_sleeve = sleeve
        n = download_one(yf, label, yahoo_ticker, args.start, args.end)
        if n > 0:
            succeeded.append((sleeve, label, n))
        else:
            failed.append((sleeve, label))
        time.sleep(REQ_PAUSE_S)

    print("\n" + "=" * 70)
    print(f"  Succeeded: {len(succeeded)}  Failed: {len(failed)}")
    print("=" * 70)
    for sleeve, label, n in succeeded:
        print(f"  [{sleeve:>22}] {label:<22} {n} bars")
    for sleeve, label in failed:
        print(f"  [FAIL {sleeve:>17}] {label}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
