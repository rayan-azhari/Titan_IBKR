"""download_data_mtf.py — Pull FX historical OHLC data from Interactive Brokers.

Downloads H1, H4, D, and W candles for the MTF Confluence strategy.
Stores output as Parquet files in data/, e.g. for EUR/USD:
  data/EUR_USD_H1.parquet
  data/EUR_USD_H4.parquet
  data/EUR_USD_D.parquet
  data/EUR_USD_W.parquet

Usage:
  uv run python scripts/download_data_mtf.py             # defaults to EUR_USD
  uv run python scripts/download_data_mtf.py --pair GBP_USD

Called automatically by scripts/run_live_mtf.py at startup (EUR/USD only).
"""

import argparse
import sys
import threading
import time
from datetime import timezone
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
IBKR_CLIENT_ID = 11  # Fixed — must not conflict with ORB (20), check_balance (99), download_data (10)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Timeframes needed by the MTF strategy
TIMEFRAMES = [
    ("H1", "1 hour",  "1 Y"),
    ("H4", "4 hours", "1 Y"),
    ("D",  "1 day",   "5 Y"),
    ("W",  "1 week",  "5 Y"),
]


class IBKRHistoricalDataApp(EWrapper, EClient):
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
        if errorCode not in [2104, 2106, 2158]:  # ignore info-only messages
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


def fetch_timeframe(
    app: IBKRHistoricalDataApp,
    req_id: int,
    pair: str,
    symbol: str,
    currency: str,
    suffix: str,
    bar_size: str,
    duration: str,
) -> None:
    output_path = DATA_DIR / f"{pair}_{suffix}.parquet"
    print(f"  Downloading {pair} {suffix} (bar_size='{bar_size}', duration='{duration}')...")

    app.data_store = []
    app.req_complete = False
    app.error_received = False
    app.error_msg = ""

    app.reqHistoricalData(
        reqId=req_id,
        contract=build_contract(symbol, currency),
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="MIDPOINT",
        useRTH=0,          # forex trades 24h — include all sessions
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    while not app.req_complete:
        time.sleep(0.5)

    if app.error_received:
        print(f"    ERROR: {app.error_msg}")
        return

    if not app.data_store:
        print("    WARNING: No data returned.")
        return

    df = pd.DataFrame(app.data_store)

    # Normalise timestamp (IBKR may return "YYYYMMDD HH:mm:ss" or "YYYYMMDD HH:mm:ss TZ")
    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
        .str.split(" ")
        .apply(lambda x: f"{x[0]} {x[1]}" if len(x) > 1 else x[0])
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Merge with existing file to avoid re-downloading old data
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        df = (
            pd.concat([existing, df])
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
        )

    # Drop any future-timestamped rows (data artefact)
    now_utc = pd.Timestamp.now(timezone.utc)
    df = df[df["timestamp"] <= now_utc]

    df.to_parquet(output_path, index=False)
    print(f"    Saved. Total rows: {len(df)}. Last: {df['timestamp'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MTF candles from IBKR.")
    parser.add_argument(
        "--pair",
        default="EUR_USD",
        help="FX pair in BASE_QUOTE format, e.g. EUR_USD or GBP_USD (default: EUR_USD)",
    )
    args = parser.parse_args()

    pair = args.pair.upper()
    parts = pair.split("_")
    if len(parts) != 2:
        print(f"ERROR: --pair must be BASE_QUOTE format (e.g. EUR_USD), got: {pair}")
        sys.exit(1)
    symbol, currency = parts[0], parts[1]

    print(f"Downloading {pair} warmup data for MTF strategy...\n")

    app = IBKRHistoricalDataApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    if not app.connected_event.wait(timeout=10):
        print("ERROR: Timed out waiting for IBKR connection (nextValidId not received).")
        print("  Check that TWS/Gateway is running and API is enabled on port", IBKR_PORT)
        app.disconnect()
        sys.exit(1)
    print("  Connected.")

    for req_id, (suffix, bar_size, duration) in enumerate(TIMEFRAMES, start=1):
        fetch_timeframe(app, req_id, pair, symbol, currency, suffix, bar_size, duration)
        time.sleep(1)  # IBKR historical data pacing

    print(f"\n{pair} data download complete.\n")
    app.disconnect()


if __name__ == "__main__":
    main()
