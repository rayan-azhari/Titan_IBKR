import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 7497))
IBKR_CLIENT_ID = 99  # Fixed ID — must not conflict with ORB (20) or download (10)
IBKR_ACCOUNT_ID = os.getenv("IBKR_ACCOUNT_ID", "DUXXXXX")


class BalanceApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = False  # set True in nextValidId when handshake is complete
        self.balance_retrieved = False
        self.net_liq = "Unknown"
        self.available_funds = "Unknown"

    def nextValidId(self, orderId: int):
        # IB fires this once the connection handshake is fully complete.
        # It is safe to make API calls only after this point.
        self.ready = True

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        # IB paper accounts send values with currency="USD"; live may use "BASE"
        if key == "NetLiquidation" and currency in ("USD", "BASE"):
            self.net_liq = val
        elif key == "AvailableFunds" and currency in ("USD", "BASE"):
            self.available_funds = val

    def accountDownloadEnd(self, accountName: str):
        self.balance_retrieved = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Suppress informational messages; print everything else
        if errorCode not in (2104, 2106, 2158, 2119):
            print(f"  [IB error {errorCode}] {errorString}")


def main():
    print(f"Connecting to IBKR on {IBKR_HOST}:{IBKR_PORT} for Account {IBKR_ACCOUNT_ID}...")
    app = BalanceApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for nextValidId (true connection ready signal), up to 5 seconds
    deadline = time.time() + 5
    while not app.ready and time.time() < deadline:
        time.sleep(0.05)

    if not app.ready:
        print("\n❌ Connection failed — TWS/Gateway did not respond within 5 seconds.")
        print("   Check that TWS is running and API is enabled (Edit > Global Config > API).")
        app.disconnect()
        sys.exit(1)

    app.reqAccountUpdates(True, IBKR_ACCOUNT_ID)

    # Wait for accountDownloadEnd, up to 10 seconds
    deadline = time.time() + 10
    while not app.balance_retrieved and time.time() < deadline:
        time.sleep(0.1)

    app.reqAccountUpdates(False, IBKR_ACCOUNT_ID)
    app.disconnect()

    print("\n✅ Connection Successful!")
    print("-" * 40)
    print(f"Account ID:      {IBKR_ACCOUNT_ID}")
    print(
        f"Net Liquidation: ${float(app.net_liq):,.2f}"
        if app.net_liq != "Unknown"
        else "Net Liquidation: Unknown"
    )
    print(
        f"Available Funds: ${float(app.available_funds):,.2f}"
        if app.available_funds != "Unknown"
        else "Available Funds: Unknown"
    )
    print("-" * 40)

    if app.net_liq == "Unknown":
        print(
            "\n⚠  Balance data not received. Possible causes:\n"
            "   1. Read-Only API is checked in TWS settings — uncheck it\n"
            "   2. Account ID in .env does not match the logged-in account\n"
            "   3. TWS is still initialising — wait 30 s and retry"
        )


if __name__ == "__main__":
    main()
