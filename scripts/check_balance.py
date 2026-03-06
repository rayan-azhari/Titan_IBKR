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
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 99))
IBKR_ACCOUNT_ID = os.getenv("IBKR_ACCOUNT_ID", "DUXXXXX")

class BalanceApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.balance_retrieved = False
        self.net_liq = "Unknown"
        self.available_funds = "Unknown"

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        if currency == "BASE":
            if key == "NetLiquidationByCurrency":
                self.net_liq = val
            elif key == "AvailableFunds":
                self.available_funds = val

    def accountDownloadEnd(self, accountName: str):
        self.balance_retrieved = True
        self.disconnect()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158]:
            pass

def main():
    print(f"Connecting to IBKR on {IBKR_HOST}:{IBKR_PORT} for Account {IBKR_ACCOUNT_ID}...")
    app = BalanceApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1) # wait for connection
    app.reqAccountUpdates(True, IBKR_ACCOUNT_ID)

    start_time = time.time()
    while not app.balance_retrieved and time.time() - start_time < 5:
        time.sleep(0.1)

    print("\n✅ Connection Successful!")
    print("-" * 40)
    print(f"Account ID:      {IBKR_ACCOUNT_ID}")
    print(f"Net Liquidation: ${float(app.net_liq):,.2f}" if app.net_liq != "Unknown" else "Net Liquidation: Unknown")
    print(f"Available Funds: ${float(app.available_funds):,.2f}" if app.available_funds != "Unknown" else "Available Funds: Unknown")
    print("-" * 40)

if __name__ == "__main__":
    main()
