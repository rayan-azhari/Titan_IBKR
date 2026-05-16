"""probe_ibkr_futures.py -- One-shot probe to test if IBKR can give us
M1 + M2 continuous-contract data for commodity futures.

Tests three approaches:
    1. Continuous Future (CONTFUT) -- front-month only, automatically rolled
    2. Futures chain (build_futures_chain) -- pull all expiries
    3. Specific localSymbol (e.g. CLZ4.NYMEX) -- single contract by month

For D2 strict-carry to work, we need a way to get M2 (second-nearest)
continuous-contract data. CONTFUT gives M1 only. The futures-chain
approach lets us pull EVERY expiry and stitch M1/M2 ourselves with a
manual rolling schedule.

Usage::

    IBKR_PORT=4004 PYTHONIOENCODING=utf-8 uv run python scripts/probe_ibkr_futures.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4004))
CLIENT_ID = 92  # one-off; doesn't conflict with strategy=21, kill_switch=98, flatten=97


class ProbeApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.contracts: list[Contract] = []
        self.contract_details_done = False
        self.bars: list = []
        self.bars_done = False

    def nextValidId(self, orderId: int) -> None:
        self.ready = True

    def contractDetails(self, reqId: int, contractDetails) -> None:
        c = contractDetails.contract
        self.contracts.append(c)
        # Print last-trade-date for futures so we can sort by expiry.
        ltd = contractDetails.contract.lastTradeDateOrContractMonth or "n/a"
        print(
            f"  reqId={reqId}  conId={c.conId}  localSymbol={c.localSymbol}  "
            f"lastTradeDate={ltd}  exch={c.exchange}  ccy={c.currency}"
        )

    def contractDetailsEnd(self, reqId: int) -> None:
        self.contract_details_done = True

    def historicalData(self, reqId: int, bar) -> None:
        self.bars.append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self.bars_done = True
        print(f"  historicalData done: {len(self.bars)} bars, {start} -> {end}")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
        if errorCode not in (2104, 2106, 2158, 2119, 2107, 10167, 10168):
            print(f"  [IB {errorCode}] reqId={reqId}: {errorString}")


def _wait(flag_attr: str, app: ProbeApp, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while not getattr(app, flag_attr) and time.time() < deadline:
        time.sleep(0.1)
    return getattr(app, flag_attr)


def main() -> int:
    print(f"Connecting to IBKR {IBKR_HOST}:{IBKR_PORT} (client_id={CLIENT_ID})...")
    app = ProbeApp()
    app.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()
    if not _wait("ready", app, 6):
        print("Connection failed.")
        return 1
    print("Connected.\n")

    # ── (1) CONTFUT for CL (WTI Crude) ────────────────────────────────────
    print("=== Probe 1: CONTFUT CL @ NYMEX (front-month continuous) ===")
    c = Contract()
    c.secType = "CONTFUT"
    c.symbol = "CL"
    c.exchange = "NYMEX"
    c.currency = "USD"
    app.contracts.clear()
    app.contract_details_done = False
    app.reqContractDetails(1, c)
    _wait("contract_details_done", app, 8)
    print(f"  -> {len(app.contracts)} contract(s) returned\n")

    # ── (2) Full futures chain for CL ─────────────────────────────────────
    print("=== Probe 2: CL FUT chain @ NYMEX (all expiries) ===")
    c2 = Contract()
    c2.secType = "FUT"
    c2.symbol = "CL"
    c2.exchange = "NYMEX"
    c2.currency = "USD"
    app.contracts.clear()
    app.contract_details_done = False
    app.reqContractDetails(2, c2)
    _wait("contract_details_done", app, 12)
    n_chain = len(app.contracts)
    print(f"  -> {n_chain} contract(s) returned\n")

    if n_chain >= 2:
        # Sort by lastTradeDate (closest first) and pull a small bar sample
        # for the FIRST and SECOND to confirm both are historically available.
        sorted_contracts = sorted(
            app.contracts,
            key=lambda c: c.lastTradeDateOrContractMonth or "99999999",
        )
        for i, leg in enumerate(sorted_contracts[:2]):
            print(f"\n=== Probe 3.{i + 1}: historical daily bars on {leg.localSymbol} ===")
            app.bars.clear()
            app.bars_done = False
            # Pull 1 month of daily bars.
            end_dt = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
            app.reqHistoricalData(
                reqId=100 + i,
                contract=leg,
                endDateTime=end_dt,
                durationStr="30 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[],
            )
            _wait("bars_done", app, 15)
            if app.bars:
                first = app.bars[0]
                last = app.bars[-1]
                print(
                    f"  first: date={first.date} close={first.close}\n"
                    f"  last:  date={last.date} close={last.close}"
                )

    print("\n=== Probe complete. ===")
    app.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
