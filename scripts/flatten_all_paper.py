"""flatten_all_paper.py -- one-shot flatten of every open position on the
paper account.

Same connection model as scripts/close_cspx_orphan.py (re-uses the
contract returned by reqPositions, which is exchange/currency-correct
for UCITS / UK-listed instruments — unlike kill_switch.py which builds
fresh SMART/USD stock contracts and fails for non-US listings).

Usage::

    IBKR_PORT=4004 PYTHONIOENCODING=utf-8 \
        uv run python scripts/flatten_all_paper.py

Non-interactive (no KILL prompt). Invoke explicitly — flattens EVERY
open position on the connected account.
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

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4004))
CLIENT_ID = 97  # one-off; doesn't conflict with strategy=21, kill_switch=98, cspx_orphan=99


class FlattenApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.next_order_id: int | None = None
        self.positions: list[tuple[str, Contract, float, float]] = []
        self.positions_done = False
        self.order_states: dict[int, str] = {}

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
        print(f"  order {orderId} status={status} filled={filled} remaining={remaining}")
        self.order_states[orderId] = status

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
        if errorCode not in (2104, 2106, 2158, 2119, 2107):
            print(f"  [IB {errorCode}] {errorString}")


def _mkt_order(action: str, qty: float) -> Order:
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
    app = FlattenApp()
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

    # 1. reqGlobalCancel first to kill any working orders.
    print("[1/3] reqGlobalCancel to kill any working orders ...")
    app.reqGlobalCancel()
    time.sleep(2)

    # 2. Snapshot positions.
    print("[2/3] Fetching open positions ...")
    app.reqPositions()
    deadline = time.time() + 8
    while not app.positions_done and time.time() < deadline:
        time.sleep(0.1)
    app.cancelPositions()

    if not app.positions:
        print("No open positions. Done.")
        time.sleep(1)
        app.disconnect()
        return 0

    print(f"Found {len(app.positions)} open position(s):")
    for _, c, pos, avg in app.positions:
        side = "LONG " if pos > 0 else "SHORT"
        print(
            f"  {side} {abs(pos):>6.0f}  {c.symbol:<8} "
            f"primaryExch={c.primaryExchange or '-':<10} {c.currency:<4}  avg={avg:.2f}"
        )

    # 3. Submit MKT close on each using IB's returned contract (correct exchange/ccy).
    print("\n[3/3] Submitting MKT close orders ...")
    oid = app.next_order_id
    submitted = []
    for _, c, pos, _ in app.positions:
        action = "SELL" if pos > 0 else "BUY"
        qty = abs(pos)
        o = _mkt_order(action, qty)
        print(
            f"  {action} MKT {qty:.0f}  {c.symbol}.{c.primaryExchange or c.exchange}  (oid={oid})"
        )
        app.placeOrder(oid, c, o)
        submitted.append(oid)
        oid += 1
        time.sleep(0.3)

    # Wait for fills / terminal states.
    print("\nWaiting up to 60s for terminal order states ...")
    deadline = time.time() + 60
    while time.time() < deadline:
        terminal = sum(
            1
            for s in app.order_states.values()
            if s in ("Filled", "Cancelled", "ApiCancelled", "Inactive")
        )
        if terminal >= len(submitted):
            break
        time.sleep(0.5)

    filled = [oid for oid, s in app.order_states.items() if s == "Filled"]
    others = [
        (oid, s)
        for oid, s in app.order_states.items()
        if s not in ("Filled", "PreSubmitted", "Submitted")
    ]
    print("\n=== Summary ===")
    print(f"  Submitted: {len(submitted)}")
    print(f"  Filled   : {len(filled)}")
    if others:
        for oid, s in others:
            print(f"  Order {oid}: terminal status = {s}")

    time.sleep(2)
    app.disconnect()
    # Any non-filled order is treated as a deployment blocker.
    return 0 if len(filled) == len(submitted) else 2


if __name__ == "__main__":
    sys.exit(main())
