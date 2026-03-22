"""download_fx_m5.py — Download 5-minute FX bars from IBKR.

IBKR caps 5-min historical data at 6 months per request, so this script
fetches in 6-month chunks stepping back from today.

Usage:
  uv run python scripts/download_fx_m5.py --pair USD_CHF --chunks 4   # ~2 years back
  uv run python scripts/download_fx_m5.py --pair EUR_USD --chunks 6   # ~3 years back
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

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4002))
IBKR_CLIENT_ID = 12  # Must not conflict with ORB (20), MTF (11), check_balance (99)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BAR_SIZE = "5 mins"
CHUNK_DURATION = "6 M"   # IBKR max per request for 5-min bars
CHUNK_STEP_DAYS = 182    # ~6 months


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
        self.data_store.append({
            "timestamp": bar.date,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        })

    def historicalDataEnd(self, reqId, start, end):
        self.req_complete = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2108, 2158]:
            self.error_received = True
            self.error_msg = f"[{errorCode}] {errorString}"
            if reqId > -1:
                self.req_complete = True


def build_contract(symbol: str, currency: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "CASH"
    c.currency = currency
    c.exchange = "IDEALPRO"
    return c


def fetch_chunk(
    app: IBKRApp,
    req_id: int,
    symbol: str,
    currency: str,
    end_dt: str,
) -> list:
    app.data_store = []
    app.req_complete = False
    app.error_received = False
    app.error_msg = ""

    label = end_dt if end_dt else "now"
    print(f"  Fetching chunk end={label} ...")

    app.reqHistoricalData(
        reqId=req_id,
        contract=build_contract(symbol, currency),
        endDateTime=end_dt,
        durationStr=CHUNK_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="MIDPOINT",
        useRTH=0,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    while not app.req_complete:
        time.sleep(0.5)

    if app.error_received:
        print(f"    ERROR: {app.error_msg}")
        return []

    print(f"    Got {len(app.data_store)} bars")
    return list(app.data_store)


def _parse_timestamp(raw: str) -> pd.Timestamp:
    parts = raw.split(" ")
    dt_str = f"{parts[0]} {parts[1]}" if len(parts) > 1 else parts[0]
    return pd.Timestamp(dt_str, tz="UTC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="USD_CHF",
                        help="FX pair BASE_QUOTE (default: USD_CHF)")
    parser.add_argument("--chunks", type=int, default=4,
                        help="Number of 6-month chunks to fetch (default: 4 = ~2 years)")
    args = parser.parse_args()

    pair = args.pair.upper()
    parts = pair.split("_")
    if len(parts) != 2:
        print(f"ERROR: --pair must be BASE_QUOTE format, got: {pair}")
        sys.exit(1)
    symbol, currency = parts[0], parts[1]

    output_path = DATA_DIR / f"{pair}_M5.parquet"
    print(f"Downloading {pair} M5 ({args.chunks} x 6-month chunks)...")
    print(f"Output: {output_path}\n")

    app = IBKRApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    if not app.connected_event.wait(timeout=10):
        print("ERROR: Timed out waiting for IBKR connection.")
        print("  Check TWS/Gateway is running and API enabled on port", IBKR_PORT)
        app.disconnect()
        sys.exit(1)
    print("  Connected.\n")

    all_rows: list = []
    now_utc = datetime.now(timezone.utc)

    for chunk in range(args.chunks):
        end_ts = now_utc - timedelta(days=CHUNK_STEP_DAYS * chunk)
        end_dt = "" if chunk == 0 else end_ts.strftime("%Y%m%d-%H:%M:%S")
        rows = fetch_chunk(app, req_id=chunk + 1, symbol=symbol, currency=currency, end_dt=end_dt)
        all_rows.extend(rows)
        time.sleep(2)  # IBKR pacing

    app.disconnect()

    if not all_rows:
        print("\nERROR: No data received.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df["timestamp"] = df["timestamp"].apply(_parse_timestamp)
    df = (
        df.drop_duplicates(subset="timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    df = df[df["timestamp"] <= pd.Timestamp.now(tz="UTC")]

    # Merge with existing file if present
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        df = (
            pd.concat([existing, df])
              .drop_duplicates(subset="timestamp")
              .sort_values("timestamp")
              .reset_index(drop=True)
        )

    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} bars -> {output_path}")
    print(f"Range: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
