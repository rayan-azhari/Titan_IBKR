"""download_h1_fx.py -- Download hourly FX bars from IBKR and merge with existing data.

IBKR allows up to 1 year per historical data request for 1-hour bars.
This script chunks back from today in 1-year steps and merges the result
with any existing Parquet file so no data is duplicated.

Usage:
    uv run python scripts/download_h1_fx.py --pair AUD_USD --years 15
    uv run python scripts/download_h1_fx.py --pair AUD_JPY --years 15
    uv run python scripts/download_h1_fx.py --pair AUD_USD --years 10 --port 7497

Port 7497 = TWS Paper Trading.  Port 4002 = IB Gateway.
"""

import argparse
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BAR_SIZE = "1 hour"
CHUNK_DURATION = "1 Y"
CHUNK_STEP_DAYS = 365
TIMEOUT_SEC = 60


class IBKRApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data_store: list = []
        self.req_complete: bool = False
        self.error_received: bool = False
        self.error_msg: str = ""
        self.connected_event: Event = Event()

    def nextValidId(self, orderId: int):
        self.connected_event.set()

    def historicalData(self, reqId, bar):
        self.data_store.append(
            {
                "timestamp": bar.date,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

    def historicalDataEnd(self, reqId, start, end):
        self.req_complete = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2108, 2158]:
            self.error_received = True
            self.error_msg = f"[{errorCode}] {errorString}"
            if reqId > -1:
                self.req_complete = True


def build_contract(base: str, quote: str) -> Contract:
    c = Contract()
    c.symbol = base
    c.secType = "CASH"
    c.currency = quote
    c.exchange = "IDEALPRO"
    return c


def fetch_chunk(app: IBKRApp, req_id: int, contract: Contract, end_dt: str) -> list:
    app.data_store = []
    app.req_complete = False
    app.error_received = False

    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_dt,
        durationStr=CHUNK_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="MIDPOINT",
        useRTH=0,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    deadline = time.time() + TIMEOUT_SEC
    while not app.req_complete and time.time() < deadline:
        time.sleep(0.1)

    if not app.req_complete:
        print("    TIMEOUT waiting for data")

    if app.error_received:
        print(f"    IBKR error: {app.error_msg}")

    return list(app.data_store)


def parse_bars(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)

    # IBKR H1 format: "20250408 17:15:00 US/Eastern"  (may have tz suffix)
    def _parse(ts: str) -> datetime:
        ts = ts.strip()
        # Strip any timezone suffix after the time (e.g. " US/Eastern")
        # Format is always "YYYYMMDD HH:MM:SS [tz]" — take first 17 chars
        ts_core = ts[:17].strip()  # "YYYYMMDD HH:MM:SS"
        try:
            return datetime.strptime(ts_core, "%Y%m%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        # Some responses use double space separator
        ts_core2 = ts[:18].strip()
        try:
            return datetime.strptime(ts_core2, "%Y%m%d  %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        raise ValueError(f"Cannot parse IBKR timestamp: {ts!r}")

    df["timestamp"] = df["timestamp"].apply(_parse)
    df = df.set_index("timestamp").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download FX H1 bars from IBKR")
    parser.add_argument("--pair", default="AUD_USD", help="FX pair BASE_QUOTE (e.g. AUD_USD)")
    parser.add_argument("--years", type=int, default=15, help="Years of history to fetch")
    parser.add_argument("--host", default=os.getenv("IBKR_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="IBKR port (7497=TWS Paper, 4002=Gateway)",
    )
    parser.add_argument("--client-id", type=int, default=30)
    args = parser.parse_args()

    pair = args.pair.upper()
    parts = pair.split("_")
    if len(parts) != 2:
        print(f"ERROR: --pair must be BASE_QUOTE format, got: {pair}")
        sys.exit(1)

    base, quote = parts[0], parts[1]
    output_path = DATA_DIR / f"{pair}_H1.parquet"
    n_chunks = args.years  # 1 chunk = 1 year

    print(f"Downloading {pair} H1 ({n_chunks} x 1-year chunks from IBKR port {args.port})...")

    app = IBKRApp()
    app.connect(args.host, args.port, args.client_id)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    if not app.connected_event.wait(timeout=10):
        print("ERROR: Could not connect to IBKR. Check TWS/Gateway is running.")
        sys.exit(1)

    print(f"  Connected to IBKR on port {args.port}")

    contract = build_contract(base, quote)
    all_frames = []
    now_utc = datetime.now(timezone.utc)

    for chunk in range(n_chunks):
        end_ts = now_utc - timedelta(days=CHUNK_STEP_DAYS * chunk)
        end_str = end_ts.strftime("%Y%m%d-%H:%M:%S")  # IBKR UTC format: yyyymmdd-hh:mm:ss
        req_id = 200 + chunk

        print(f"  Chunk {chunk + 1}/{n_chunks} ending {end_str[:10]} ...", end=" ", flush=True)
        raw = fetch_chunk(app, req_id, contract, end_str)

        if raw:
            df_chunk = parse_bars(raw)
            if not df_chunk.empty:
                all_frames.append(df_chunk)
                first, last = df_chunk.index[0].date(), df_chunk.index[-1].date()
                print(f"{len(df_chunk)} bars  [{first} - {last}]")
            else:
                print("parse error")
        else:
            print("no data")

        time.sleep(1.5)  # IBKR rate limit

    app.disconnect()

    if not all_frames:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    # Merge with existing data
    new_df = pd.concat(all_frames).sort_index()
    new_df = new_df[~new_df.index.duplicated(keep="last")]

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if not isinstance(existing.index, pd.DatetimeIndex):
            print("  Existing file has non-datetime index — replacing entirely")
            merged = new_df
        else:
            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")
            else:
                existing.index = existing.index.tz_convert("UTC")
            merged = pd.concat([existing, new_df]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            print(f"  Merged: {len(existing)} existing + {len(new_df)} new = {len(merged)} total")
    else:
        merged = new_df
        print(f"  New file: {len(merged)} bars")

    merged.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
    print(f"  Coverage: {merged.index[0].date()} -> {merged.index[-1].date()}")
    print(f"  Total bars: {len(merged):,}")


if __name__ == "__main__":
    main()
