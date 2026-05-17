"""download_b2_futures_databento.py -- Carver EWMAC futures basket.

B2 backlog (V3.7 fresh strategy audit): adds the four missing futures to
the existing basket so the audit covers a cross-asset trend-following
panel.

  - ZN  10y US Treasury Note  (CME GLBX.MDP3)
  - ZB  30y US Treasury Bond  (CME GLBX.MDP3)
  - 6E  Euro FX               (CME GLBX.MDP3)
  - 6J  Japanese Yen FX       (CME GLBX.MDP3)

All four are CME Globex continuous symbology, which begins ~2017-06.

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/download_b2_futures_databento.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# CME Globex continuous symbology starts here (per existing downloader).
START = "2017-06-01"
END = (date.today() - timedelta(days=1)).isoformat()
DATASET = "GLBX.MDP3"

ROOTS = [
    ("ZN", "10y US Treasury Note"),
    ("ZB", "30y US Treasury Bond"),
    ("6E", "Euro FX"),
    ("6J", "Japanese Yen FX"),
]


def download_continuous(client, root: str, contract_n: int) -> pd.DataFrame:  # noqa: ANN001
    sym = f"{root}.c.{contract_n}"
    print(f"  [{DATASET}] {sym} {START} -> {END} ...", end="", flush=True)
    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            schema="ohlcv-1d",
            symbols=[sym],
            start=START,
            end=END,
            stype_in="continuous",
        )
        df = data.to_df()
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc}")
        return pd.DataFrame()

    if df.empty:
        print("  (no rows)")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df = df.set_index("ts_event")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype(float).sort_index().dropna(how="all")
    print(f"  -> {len(df)} bars  {df.index[0].date()}..{df.index[-1].date()}")
    return df


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or api_key.startswith("db-xxx"):
        print("ERROR: DATABENTO_API_KEY missing", file=sys.stderr)
        return 1

    import databento as db
    client = db.Historical(key=api_key)

    print("=" * 70)
    print(f"  B2 Carver EWMAC futures download ({DATASET})")
    print(f"  Roots: {[r for r, _ in ROOTS]}")
    print(f"  Period: {START} -> {END}")
    print("=" * 70)

    succeeded: list[str] = []
    failed: list[str] = []
    for root, name in ROOTS:
        print(f"\n[{root}] {name}")
        df_m1 = download_continuous(client, root, 0)
        df_m2 = download_continuous(client, root, 1)
        if not df_m1.empty:
            df_m1.to_parquet(DATA_DIR / f"{root}_M1_D.parquet")
            print(f"    saved data/{root}_M1_D.parquet")
        if not df_m2.empty:
            df_m2.to_parquet(DATA_DIR / f"{root}_M2_D.parquet")
            print(f"    saved data/{root}_M2_D.parquet")
        if df_m1.empty and df_m2.empty:
            failed.append(root)
        else:
            succeeded.append(root)
        time.sleep(0.4)

    print("\n" + "=" * 70)
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print("=" * 70)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
