"""kill_switch.py
----------------
Emergency: cancel all open orders and flatten all open positions via TWS API.

Usage:
    python scripts/kill_switch.py

Connects to TWS, shows open positions, asks for confirmation, then:
  1. Sends reqGlobalCancel to cancel every open order
  2. Submits a DAY market order to close each open position

Client ID: 98 — does not conflict with ORB (20), download (10), balance (99), test (25).
"""

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
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 7497))
IBKR_CLIENT_ID = 98  # Fixed — must not conflict with other scripts


class KillSwitchApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = False
        self.next_order_id = None
        self.positions: list = []  # (account, contract, size, avg_cost)
        self.positions_done = False

    def nextValidId(self, orderId: int):
        self.ready = True
        self.next_order_id = orderId

    def position(self, account: str, contract: Contract, pos: float, avgCost: float):
        if pos != 0:
            self.positions.append((account, contract, pos, avgCost))

    def positionEnd(self):
        self.positions_done = True

    def orderStatus(
        self, orderId, status, filled, remaining, avgFillPrice,
        permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice,
    ):
        if status not in ("PreSubmitted", "Submitted"):
            print(f"  Order {orderId}: {status}  filled={filled}  remaining={remaining}")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in (2104, 2106, 2158, 2119):
            print(f"  [IB {errorCode}] {errorString}")


def _stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def _market_order(action: str, quantity: float) -> Order:
    o = Order()
    o.action = action         # "BUY" or "SELL"
    o.orderType = "MKT"
    o.totalQuantity = quantity
    o.tif = "DAY"
    return o


def main():
    print(f"Connecting to TWS on {IBKR_HOST}:{IBKR_PORT}  (client_id={IBKR_CLIENT_ID})...")
    app = KillSwitchApp()
    app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    # Wait for connection handshake
    deadline = time.time() + 5
    while not app.ready and time.time() < deadline:
        time.sleep(0.05)

    if not app.ready:
        print("\n❌ Connection failed — is TWS running with API enabled?")
        app.disconnect()
        sys.exit(1)

    print("✅ Connected.\n")

    # Fetch all open positions
    print("Fetching open positions...")
    app.reqPositions()
    deadline = time.time() + 5
    while not app.positions_done and time.time() < deadline:
        time.sleep(0.1)
    app.cancelPositions()

    if not app.positions:
        print("No open positions found.")
        print("Sending reqGlobalCancel anyway to clear any pending orders...")
        app.reqGlobalCancel()
        time.sleep(1)
        print("✅ Done.")
        app.disconnect()
        return

    print(f"Found {len(app.positions)} open position(s):\n")
    for _, contract, pos, avg_cost in app.positions:
        side = "LONG " if pos > 0 else "SHORT"
        print(f"  {side}  {abs(pos):>6.0f}  {contract.symbol:<10}  avg {avg_cost:.2f}")

    print("\n" + "=" * 50)
    print("⚠  KILL SWITCH will:")
    print("   1. Cancel ALL open orders  (reqGlobalCancel)")
    print(f"   2. Close ALL {len(app.positions)} position(s) at market")
    print("=" * 50)
    confirm = input("\nType  KILL  to confirm, or anything else to abort: ").strip()

    if confirm != "KILL":
        print("\nAborted. No changes made.")
        app.disconnect()
        sys.exit(0)

    # Step 1: cancel all orders
    print("\n[1/2] Cancelling all orders...")
    app.reqGlobalCancel()
    time.sleep(1)
    print("  ✅ reqGlobalCancel sent.")

    # Step 2: market-close every position
    print("\n[2/2] Submitting market orders to close positions...")
    oid = app.next_order_id
    for _, contract, pos, _ in app.positions:
        action = "SELL" if pos > 0 else "BUY"
        qty = abs(pos)
        c = _stock_contract(contract.symbol)
        o = _market_order(action, qty)
        print(f"  {action} MKT {qty:.0f}  {contract.symbol}  (order_id={oid})")
        app.placeOrder(oid, c, o)
        oid += 1
        time.sleep(0.2)  # small gap to avoid rate-limiting

    print("\n✅ Kill switch executed. Monitor TWS for fill confirmations.")
    time.sleep(3)
    app.disconnect()


if __name__ == "__main__":
    main()
