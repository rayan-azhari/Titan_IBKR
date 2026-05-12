"""close_orphans.py -- surgical close of arbitrary LSEETF positions.

Generalisation of close_cspx_orphan.py for the May 12 2026 incident: a
NautilusTrader 1.221 Cython-enum bug in ``BondGoldStrategy._run_signal``
caused the strategy to layer fresh BUYs on top of EXTERNAL-rehydrated
VUSD/EIMI positions. The fix is in (see directives/Rehydration Bug
2026-05-11.md) but we still need to flatten the over-sized inventory so
the next strategy restart re-enters at the intended sizing.

Connects with a fresh client_id (so it can't conflict with the live
strategy's client_id=7), enumerates open positions, and submits a MKT
order to flatten each symbol passed on the command line.

Usage (from host, against a running stack):
    docker compose exec -T titan-portfolio \
        uv run python scripts/close_orphans.py VUSD EIMI

Or for a one-off close while the container is stopped, exec in via a
sidecar container on the same docker network.
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
CLIENT_ID = 97  # one-off; doesn't conflict with strategy=7, kill_switch=98, cspx_close=99
TARGET_PRIMARY_EXCH = "LSEETF"
TARGET_CURRENCY = "USD"


class CloseApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.next_order_id: int | None = None
        self.positions: list[tuple[str, Contract, float, float]] = []
        self.positions_done = False
        self.order_terminal_status: dict[int, str] = {}

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
            self.order_terminal_status[orderId] = status

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


def main(symbols: list[str]) -> int:
    if not symbols:
        print("Usage: close_orphans.py SYMBOL1 [SYMBOL2 ...]")
        return 2
    targets_wanted = {s.upper() for s in symbols}

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

    targets: list[tuple[str, Contract, float, float]] = []
    for acct, c, pos, avg in app.positions:
        match = (
            c.symbol in targets_wanted
            and (c.primaryExchange == TARGET_PRIMARY_EXCH or c.exchange == TARGET_PRIMARY_EXCH)
            and c.currency == TARGET_CURRENCY
        )
        flag = "  <-- TARGET" if match else ""
        print(
            f"  {c.symbol:<8} {c.primaryExchange:<10} {c.currency:<5}  pos={pos}  avg={avg}{flag}"
        )
        if match:
            targets.append((acct, c, pos, avg))

    if not targets:
        print(f"\nNone of {sorted(targets_wanted)} found. Nothing to do.")
        app.disconnect()
        return 0

    submitted: list[int] = []
    for _, contract, pos, _ in targets:
        qty = abs(pos)
        action = "SELL" if pos > 0 else "BUY"
        oid = app.next_order_id
        app.next_order_id += 1
        print(f"\nClosing {action} {qty:.0f} {contract.symbol}.{contract.primaryExchange}  (oid={oid})...")
        app.placeOrder(oid, contract, _market_order(action, qty))
        submitted.append(oid)
        time.sleep(0.5)

    deadline = time.time() + 60
    while time.time() < deadline:
        if all(oid in app.order_terminal_status for oid in submitted):
            break
        time.sleep(0.2)

    rc = 0
    for oid in submitted:
        status = app.order_terminal_status.get(oid)
        if status == "Filled":
            print(f"  ✓ {oid}: Filled")
        else:
            print(f"  ✗ {oid}: terminal status = {status} (check TWS / Gateway log)")
            rc = 1

    time.sleep(2)
    app.disconnect()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
