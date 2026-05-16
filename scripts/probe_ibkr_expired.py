"""probe_ibkr_expired.py -- Check if IBKR paper account exposes expired
futures contracts via:
    A) Contract.includeExpired = True on reqContractDetails(secType="FUT")
    B) Explicit query by lastTradeDateOrContractMonth (e.g. "202012")
    C) Historical bars on a known expired contract (e.g. CLZ20)

Critical for D2b/B4b audit — current chain only shows 6 forward CL contracts
covering 2025-05 → 2026-10. Need 2020-01 → 2026-05 for the pre-reg.

Usage::

    IBKR_PORT=4004 PYTHONIOENCODING=utf-8 uv run python scripts/probe_ibkr_expired.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4004))
CLIENT_ID = 93


class ProbeApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.contracts: list = []
        self.contract_details_done = False
        self.bars: list = []
        self.bars_done = False

    def nextValidId(self, orderId: int) -> None:
        self.ready = True

    def contractDetails(self, reqId, contractDetails) -> None:
        c = contractDetails.contract
        self.contracts.append(c)
        print(
            f"    conId={c.conId}  localSymbol={c.localSymbol}  "
            f"lastTradeDate={c.lastTradeDateOrContractMonth}  "
            f"exch={c.exchange}"
        )

    def contractDetailsEnd(self, reqId) -> None:
        self.contract_details_done = True

    def historicalData(self, reqId, bar) -> None:
        self.bars.append(bar)

    def historicalDataEnd(self, reqId, start: str, end: str) -> None:
        self.bars_done = True
        print(f"    [bars done: {len(self.bars)}, {start} -> {end}]")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
        if errorCode in (2104, 2106, 2158, 2119, 2107, 10167, 10168):
            return
        print(f"    [IB {errorCode}] reqId={reqId}: {errorString}")
        if errorCode in (162, 200, 322, 354):
            self.bars_done = True
            self.contract_details_done = True


def _wait(flag: str, app: ProbeApp, t: float = 12) -> bool:
    dl = time.time() + t
    while not getattr(app, flag) and time.time() < dl:
        time.sleep(0.05)
    return getattr(app, flag)


def main() -> int:
    app = ProbeApp()
    print(f"Connecting {IBKR_HOST}:{IBKR_PORT} (client_id={CLIENT_ID})")
    app.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()
    if not _wait("ready", app, 8):
        print("Connection failed.")
        return 1
    print("Connected.\n")

    # ── (A) Full CL chain WITH includeExpired=True ────────────────────────
    print("=== Probe A: CL FUT chain with includeExpired=True ===")
    c = Contract()
    c.secType = "FUT"
    c.symbol = "CL"
    c.exchange = "NYMEX"
    c.currency = "USD"
    c.includeExpired = True  # KEY FLAG
    app.contracts.clear()
    app.contract_details_done = False
    app.reqContractDetails(2001, c)
    _wait("contract_details_done", app, 25)
    n = len(app.contracts)
    print(f"  -> {n} contracts returned (was 129 without includeExpired)\n")
    if n > 0:
        dates = sorted(c2.lastTradeDateOrContractMonth or "99999999" for c2 in app.contracts)
        print(f"  earliest expiry: {dates[0]}")
        print(f"  latest expiry:   {dates[-1]}")

    time.sleep(1.5)

    # ── (B) Explicit per-month query for an expired contract ─────────────
    print("\n=== Probe B: explicit query for CL Dec 2020 (CLZ20) ===")
    c2 = Contract()
    c2.secType = "FUT"
    c2.symbol = "CL"
    c2.exchange = "NYMEX"
    c2.currency = "USD"
    c2.lastTradeDateOrContractMonth = "202012"
    c2.includeExpired = True
    app.contracts.clear()
    app.contract_details_done = False
    app.reqContractDetails(2002, c2)
    _wait("contract_details_done", app, 15)
    print(f"  -> {len(app.contracts)} contract(s) returned")

    if app.contracts:
        expired = app.contracts[0]
        time.sleep(1.5)
        # ── (C) Historical bars on the expired contract ────────────────
        print(f"\n=== Probe C: historical daily bars on {expired.localSymbol} ===")
        # Set endDateTime past the contract's expiry.
        end_dt = "20201220-23:59:59"
        app.bars.clear()
        app.bars_done = False
        # Must reuse the conId from the returned contract.
        app.reqHistoricalData(
            reqId=2003,
            contract=expired,
            endDateTime=end_dt,
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )
        _wait("bars_done", app, 25)
        if app.bars:
            print(f"  first bar: date={app.bars[0].date} close={app.bars[0].close}")
            print(f"  last bar:  date={app.bars[-1].date} close={app.bars[-1].close}")

    app.disconnect()
    print("\n=== Probe complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
