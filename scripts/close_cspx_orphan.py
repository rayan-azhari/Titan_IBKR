"""close_cspx_orphan.py — surgical close of the +36 CSPX orphan.

The +36 CSPX position was an `EXTERNAL` (no strategy_id) position left
over after we hardened position rehydration to filter by strategy_id
(May 1 2026). Neither bond_equity_ihyu_cspx nor any other strategy
will adopt it, so it must be closed manually.

This is a one-off: connect to the gateway with a fresh client_id,
look up the CSPX.LSEETF position, send a MKT sell for the qty, exit.
Runs non-interactively (no confirmation prompt) — invoke explicitly.

Usage (from host):
    docker compose exec -T titan-portfolio uv run python scripts/close_cspx_orphan.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

IBKR_HOST = os.getenv("IBKR_HOST", "ib-gateway")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4004))
CLIENT_ID = 99  # one-off; doesn't conflict with strategy=7 or kill_switch=98
TARGET_SYMBOL = "CSPX"
TARGET_PRIMARY_EXCH = "LSEETF"
TARGET_CURRENCY = "USD"


class CloseApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.next_order_id: int | None = None
        self.positions: list[tuple[str, Contract, float, float]] = []
        self.positions_done = False
        self.order_terminal_status: str | None = None

    def nextValidId(self, orderId: int) -> None:
        self.ready = True
        self.next_order_id = orderId

    def position(self, account: str, contract: Contract, pos: float, avgCost: float) -> None:
        if pos != 0:
            self.positions.append((account, contract, pos, avgCost))

    def positionEnd(self) -> None:
        self.positions_done = True

    def orderStatus(
        self,
        orderId,
        status,
        filled,
        remaining,
        avgFillPrice,
        permId,
        parentId,
        lastFillPrice,
        clientId,
        whyHeld,
        mktCapPrice,
    ) -> None:
        print(
            f"  Order {orderId} status={status} filled={filled} "
            f"remaining={remaining} avgFill={avgFillPrice}"
        )
        if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
            self.order_terminal_status = status

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
        if errorCode not in (2104, 2106, 2158, 2119, 2107):
            print(f"  [IB {errorCode}] {errorString}")


def _market_order(action: str, qty: float) -> Order:
    o = Order()
    o.action = action
    o.orderType = "MKT"
    o.totalQuantity = qty
    o.tif = "DAY"
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    return o


def main() -> int:
    print(f"Connecting to {IBKR_HOST}:{IBKR_PORT}  (client_id={CLIENT_ID})...")
    app = CloseApp()
    app.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()

    deadline = time.time() + 8
    while not app.ready and time.time() < deadline:
        time.sleep(0.05)
    if not app.ready:
        print("Connection handshake failed.")
        app.disconnect()
        return 1
    print("Connected.\n")

    print("Fetching positions...")
    app.reqPositions()
    deadline = time.time() + 8
    while not app.positions_done and time.time() < deadline:
        time.sleep(0.1)
    app.cancelPositions()

    target = None
    for acct, c, pos, avg in app.positions:
        flag = ""
        if (
            c.symbol == TARGET_SYMBOL
            and (c.primaryExchange == TARGET_PRIMARY_EXCH or c.exchange == TARGET_PRIMARY_EXCH)
            and c.currency == TARGET_CURRENCY
        ):
            target = (acct, c, pos, avg)
            flag = "  <-- TARGET"
        print(f"  {c.symbol:<8} {c.primaryExchange:<10} {c.currency:<5}  pos={pos}  avg={avg}{flag}")

    if target is None:
        print(f"\nNo {TARGET_SYMBOL}.{TARGET_PRIMARY_EXCH} position found. Nothing to do.")
        app.disconnect()
        return 0

    _, contract, pos, _ = target
    qty = abs(pos)
    action = "SELL" if pos > 0 else "BUY"

    print(f"\nClosing {action} {qty:.0f} {contract.symbol}.{contract.primaryExchange}  ...")
    o = _market_order(action, qty)
    oid = app.next_order_id
    app.placeOrder(oid, contract, o)

    deadline = time.time() + 30
    while app.order_terminal_status is None and time.time() < deadline:
        time.sleep(0.2)

    if app.order_terminal_status == "Filled":
        print(f"\n✅ Closed: order {oid} Filled.")
        rc = 0
    else:
        print(
            f"\nOrder {oid} terminal status: {app.order_terminal_status} "
            f"(check TWS / Gateway log)."
        )
        rc = 1

    time.sleep(2)
    app.disconnect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
