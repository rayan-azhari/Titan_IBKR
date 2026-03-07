"""test_ibkr_historical.py
Test fetching historical warmup data directly from IBKR.
We need 100 Daily bars and 5 days of 5-minute bars for a single ticker (e.g., AMAT).
"""

import os
import sys
import threading
import time

import pandas as pd
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 7497))
IBKR_CLIENT_ID = 888


class TestHistApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.req_complete = False
        self.error_received = False
        self.error_msg = ""

    def historicalData(self, reqId, bar):
        self.data.append(
            {
                "time": bar.date,
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
        if errorCode not in [2104, 2106, 2158]:
            self.error_received = True
            self.error_msg = f"[{errorCode}] {errorString}"
            if reqId > -1:
                self.req_complete = True


def get_contract(symbol: str) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def fetch_data(app, ticker, duration, bar_size):
    print(f"\nRequesting {duration} of {bar_size} bars for {ticker}...")
    app.data = []
    app.req_complete = False
    app.error_received = False

    contract = get_contract(ticker)

    app.reqHistoricalData(
        reqId=int(time.time()),
        contract=contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    start = time.time()
    while not app.req_complete and time.time() - start < 15:
        time.sleep(0.1)

    if app.error_received:
        print(f"❌ Error: {app.error_msg}")
        return None

    if not app.data:
        print("⚠️ No data returned.")
        return None

    df = pd.DataFrame(app.data)
    print(f"✅ Success! Fetched {len(df)} rows.")
    print(f"First bar: {df.iloc[0]['time']} | Last bar: {df.iloc[-1]['time']}")
    return df


def main():
    print(f"Connecting to IBKR on {IBKR_HOST}:{IBKR_PORT}...")
    app = TestHistApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1)

    # 1. Test 100 days of Daily bars (for SMA50/RSI14)
    df_daily = fetch_data(app, "AMAT", "100 D", "1 day")

    # Wait to avoid pacing violations
    time.sleep(2)

    # 2. Test 5 days of 5-Minute bars (for ATR14 and Gaussian)
    df_5m = fetch_data(app, "AMAT", "5 D", "5 mins")

    app.disconnect()


if __name__ == "__main__":
    main()
