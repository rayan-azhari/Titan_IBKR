"""download_gld_h1.py -- Download extended GLD H1 data from IBKR.

Downloads hourly OHLCV for GLD (Gold ETF) going back as far as IBKR allows.
Fetches in 1-year chunks and merges with existing data.

Requires IBKR Gateway running.

Usage:
    uv run python scripts/download_gld_h1.py
    uv run python scripts/download_gld_h1.py --years 10
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
IBKR_PORT = int(os.getenv("IBKR_PORT", 7497))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 19))

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def main():
    parser = argparse.ArgumentParser(description="Download GLD H1 from IBKR")
    parser.add_argument("--years", type=int, default=10, help="Years of history")
    parser.add_argument("--symbol", default="GLD", help="ETF symbol")
    args = parser.parse_args()

    symbol = args.symbol
    years = args.years
    output_path = DATA_DIR / f"{symbol}_H1.parquet"

    print(f"Downloading {symbol} H1 ({years}Y) from IBKR...")
    print(f"  Host: {IBKR_HOST}:{IBKR_PORT} | Client ID: {IBKR_CLIENT_ID}")

    app = IBKRApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    if not app.connected_event.wait(timeout=10):
        print("ERROR: Could not connect to IBKR Gateway.")
        sys.exit(1)
    print("  Connected to IBKR Gateway.")

    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.currency = "USD"
    contract.exchange = "SMART"
    contract.primaryExchange = "ARCA"

    # Fetch in 1-year chunks going backward
    all_data = []
    now = datetime.now(timezone.utc)
    req_id = 1

    for y in range(years):
        end_dt = now - timedelta(days=365 * y)
        end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")

        print(f"\n  Chunk {y + 1}/{years}: ending {end_dt.date()}...")
        app.data_store = []
        app.req_complete = False
        app.error_received = False
        app.error_msg = ""

        app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=end_str,
            durationStr="1 Y",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )

        timeout = 120
        waited = 0
        while not app.req_complete and waited < timeout:
            time.sleep(1)
            waited += 1

        if app.error_received:
            print(f"    ERROR: {app.error_msg}")
            if "no data" in app.error_msg.lower() or "162" in app.error_msg:
                print("    No more data available. Stopping.")
                break
            continue

        if app.data_store:
            all_data.extend(app.data_store)
            print(f"    Got {len(app.data_store)} bars")
        else:
            print("    No data returned. Stopping.")
            break

        req_id += 1
        time.sleep(2)  # IBKR rate limit

    app.disconnect()

    if not all_data:
        print("\nERROR: No data downloaded.")
        sys.exit(1)

    df = pd.DataFrame(all_data)
    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
        .str.split(" ")
        .apply(lambda x: f"{x[0]} {x[1]}" if len(x) > 1 else x[0])
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Merge with existing
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if "timestamp" in existing.columns:
            if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        df = pd.concat([existing, df]).drop_duplicates(subset="timestamp").sort_values("timestamp")

    df = df[df["timestamp"] <= pd.Timestamp.now(timezone.utc)]
    df.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")
    t_min = df["timestamp"].min().date()
    t_max = df["timestamp"].max().date()
    print(f"Total: {len(df)} bars ({t_min} -> {t_max})")


if __name__ == "__main__":
    main()
