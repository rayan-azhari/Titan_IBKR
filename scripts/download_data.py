"""download_data.py — Pull historical OHLC data from Interactive Brokers.

Downloads candlestick data for the instruments and granularities
specified in config/instruments.toml. Stores output as Parquet
files in data/.
"""

import argparse
import sys
import threading
import time
import tomllib
from datetime import timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

# Config
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4002))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 10))

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class IBKRHistoricalDataApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data_store = []
        self.req_complete = False
        self.error_received = False
        self.error_msg = ""

    def historicalData(self, reqId, bar):
        # bar.date format depends on resolution.
        # Usually "YYYYMMDD HH:mm:ss" for intraday, "YYYYMMDD" for daily
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
        if errorCode not in [2104, 2106, 2158]:  # ignore simple info messages
            self.error_received = True
            self.error_msg = f"[{errorCode}] {errorString}"
            if reqId > -1:
                self.req_complete = True


def load_instruments_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "instruments.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_ib_contract(symbol: str) -> Contract:
    contract = Contract()
    if "_" in symbol:
        # Forex
        contract.symbol = symbol[:3]
        contract.secType = "CASH"
        contract.currency = symbol[4:]
        contract.exchange = "IDEALPRO"
    else:
        # Stock
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
    return contract


def map_granularity(gran: str) -> str:
    # IBKR format: "1 min", "5 mins", "1 hour", "1 day", "1 week"
    mapping = {
        "M1": "1 min",
        "M5": "5 mins",
        "M15": "15 mins",
        "H1": "1 hour",
        "H4": "4 hours",
        "D": "1 day",
        "W": "1 week",
    }
    return mapping.get(gran, "1 hour")


def map_duration(gran: str) -> str:
    # For a simple download script, how much data to request per batch
    # Can be adjusted based on needs. Max for intraday is usually "1 M" at a time, sometimes "1 W".
    if gran in ["D", "W"]:
        return "5 Y"
    elif gran in ["H1", "H4"]:
        return "1 Y"
    else:
        return "1 M"


def main() -> None:
    config = load_instruments_config()

    pairs = config.get("instruments", {}).get("pairs", [])
    granularities = config.get("instruments", {}).get("granularities", ["M5"])

    parser = argparse.ArgumentParser(description="Download IBKR data")
    parser.add_argument("-i", "--instrument", help="Filter by instrument (e.g. EUR_USD)")
    parser.add_argument("-g", "--granularity", help="Filter by granularity (e.g. M5, H1)")
    args = parser.parse_args()

    if args.instrument:
        if args.instrument not in pairs:
            print(f"❌ Error: Instrument {args.instrument} not in instruments.toml")
            sys.exit(1)
        pairs = [args.instrument]

    if args.granularity:
        if args.granularity in granularities:
            granularities = [args.granularity]
        else:
            print(f"❌ Error: Granularity {args.granularity} not in instruments.toml")
            sys.exit(1)

    print(
        f"📥 Downloading data for {len(pairs)} pairs × {len(granularities)} granularities from IBKR\n"  # noqa: E501
    )

    app = IBKRHistoricalDataApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for connection
    time.sleep(1)

    req_id = 1
    for pair in pairs:
        for gran in granularities:
            output_path = DATA_DIR / f"{pair}_{gran}.parquet"
            ib_contract = get_ib_contract(pair)
            bar_size = map_granularity(gran)
            duration = map_duration(gran)

            print(
                f"  ↓ Requesting {pair} {gran} (BarSize: '{bar_size}', Duration: '{duration}')..."
            )

            app.data_store = []
            app.req_complete = False
            app.error_received = False
            app.error_msg = ""

            # Request historical data
            # To get specific end times, use format "YYYYMMDD HH:mm:ss"
            # Leaving empty string gets the most recent data up to now.
            app.reqHistoricalData(
                reqId=req_id,
                contract=ib_contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="MIDPOINT" if "_" in pair else "TRADES",
                useRTH=0 if "_" in pair else 1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[],
            )

            # Wait for request to complete
            while not app.req_complete:
                time.sleep(0.5)

            if app.error_received:
                print(f"    ❌ Error fetching data: {app.error_msg}")
                req_id += 1
                continue

            if not app.data_store:
                print(f"    ⚠ No data returned for {pair} {gran}.")
                req_id += 1
                continue

            df = pd.DataFrame(app.data_store)

            # IBKR times can be "YYYYMMDD" or "YYYYMMDD HH:mm:ss" or "YYYYMMDD HH:mm:ss Timezone" depending on bar size  # noqa: E501
            df["timestamp"] = (
                df["timestamp"]
                .astype(str)
                .str.split(" ")
                .apply(lambda x: f"{x[0]} {x[1]}" if len(x) > 1 else x[0])
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                # Assume UTC or local depending on TWS setting. Usually returned in local timezone if not specified,  # noqa: E501
                # but let's localize standardizing to UTC for simplicity.
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

            # Merge with existing data if present
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
                    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

                df = (
                    pd.concat([existing, df])
                    .drop_duplicates(subset="timestamp")
                    .sort_values("timestamp")
                )

            now_utc = pd.Timestamp.now(timezone.utc)
            df = df[df["timestamp"] <= now_utc]

            df.to_parquet(output_path, index=False)
            print(
                f"    → Saved {len(app.data_store)} new rows. Total File Rows: {len(df)}. Last TS: {df['timestamp'].max()}"  # noqa: E501
            )

            req_id += 1
            # Rate limiting for IBKR historical data pacing
            time.sleep(1)

    print("\n✅ Data download complete.\n")
    app.disconnect()


if __name__ == "__main__":
    main()
