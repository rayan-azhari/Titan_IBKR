"""download_ibkr_futures.py -- Per-contract IBKR historical daily bars for the
24-commodity D2b / B4b roll-stitching audit.

For each commodity root, this:
    1. Pulls the full FUT chain via ``reqContractDetails(secType='FUT')``.
    2. Filters contracts whose ``lastTradeDate`` falls within
       ``[audit_start - 1y, audit_end + 6mo]``.
    3. For each filtered contract, requests ``durationStr='1 Y'``
       ``barSizeSetting='1 day'`` daily bars.
    4. Saves to ``data/ibkr_futures/{ROOT}/{LOCAL_SYMBOL}_D.parquet``.

Rate-limit: 1 request per 1.5 s (~40 req/min), well under IBKR's
50-req-per-10-min historical-data limit.

Usage::

    IBKR_PORT=4004 PYTHONIOENCODING=utf-8 \
        uv run python scripts/download_ibkr_futures.py --roots CL
    # full universe (slow, 4-6 h):
    IBKR_PORT=4004 PYTHONIOENCODING=utf-8 \
        uv run python scripts/download_ibkr_futures.py --roots all

References:
    directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md
    scripts/probe_ibkr_futures.py (chain probe, confirmed 129 CL contracts)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ibkr_futures"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4004))
CLIENT_ID = 95  # distinct from gem=21, kill=98, flatten=97, probe=92

REQ_RATE_SECONDS = 1.5  # 40 req/min
DEFAULT_AUDIT_START = "2020-01-01"
DEFAULT_AUDIT_END = "2026-05-15"
CHAIN_BUFFER_BEFORE_DAYS = 365
CHAIN_BUFFER_AFTER_DAYS = 180


@dataclass(frozen=True)
class FuturesRoot:
    root: str
    exchange: str
    currency: str
    name: str


UNIVERSE: tuple[FuturesRoot, ...] = (
    # Energy -- NYMEX
    FuturesRoot("CL", "NYMEX", "USD", "WTI Crude Oil"),
    FuturesRoot("NG", "NYMEX", "USD", "Natural Gas"),
    FuturesRoot("HO", "NYMEX", "USD", "Heating Oil"),
    FuturesRoot("RB", "NYMEX", "USD", "RBOB Gasoline"),
    FuturesRoot("BZ", "NYMEX", "USD", "Brent Crude Oil"),
    # Metals precious -- NYMEX/COMEX
    FuturesRoot("GC", "COMEX", "USD", "Gold"),
    FuturesRoot("SI", "COMEX", "USD", "Silver"),
    FuturesRoot("PL", "NYMEX", "USD", "Platinum"),
    FuturesRoot("PA", "NYMEX", "USD", "Palladium"),
    # Metals industrial -- COMEX
    FuturesRoot("HG", "COMEX", "USD", "Copper"),
    # Grains + oilseeds -- CBOT
    FuturesRoot("ZC", "CBOT", "USD", "Corn"),
    FuturesRoot("ZW", "CBOT", "USD", "Chicago Wheat"),
    FuturesRoot("ZS", "CBOT", "USD", "Soybeans"),
    FuturesRoot("ZL", "CBOT", "USD", "Soybean Oil"),
    FuturesRoot("ZM", "CBOT", "USD", "Soybean Meal"),
    FuturesRoot("ZO", "CBOT", "USD", "Oats"),
    # Livestock -- CME
    FuturesRoot("LE", "CME", "USD", "Live Cattle"),
    FuturesRoot("GF", "CME", "USD", "Feeder Cattle"),
    FuturesRoot("HE", "CME", "USD", "Lean Hogs"),
    # Softs -- NYBOT (ICE-US)
    FuturesRoot("KC", "NYBOT", "USD", "Coffee"),
    FuturesRoot("CC", "NYBOT", "USD", "Cocoa"),
    FuturesRoot("SB", "NYBOT", "USD", "Sugar No. 11"),
    FuturesRoot("CT", "NYBOT", "USD", "Cotton No. 2"),
    FuturesRoot("OJ", "NYBOT", "USD", "Orange Juice"),
)


class IBHistApp(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.ready = False
        self.contracts: list = []
        self.contract_details_done = False
        self.bars: list = []
        self.bars_done = False
        self.last_error: tuple[int, int, str] | None = None

    def nextValidId(self, orderId: int) -> None:
        self.ready = True

    def contractDetails(self, reqId: int, contractDetails) -> None:
        self.contracts.append(contractDetails.contract)
        # Stash lastTradeDate on the contract for filtering.
        self.contracts[
            -1
        ].lastTradeDateOrContractMonth = contractDetails.contract.lastTradeDateOrContractMonth

    def contractDetailsEnd(self, reqId: int) -> None:
        self.contract_details_done = True

    def historicalData(self, reqId: int, bar) -> None:
        self.bars.append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self.bars_done = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="") -> None:
        # Suppress benign farm-status notices.
        if errorCode in (2104, 2106, 2158, 2119, 2107, 10167, 10168):
            return
        # 162 = "no data" / 200 = "no security definition" / 322 = "duplicate".
        self.last_error = (reqId, errorCode, errorString)
        if errorCode in (162, 200, 322, 354):
            # Treat as terminal for this request.
            self.bars_done = True
            self.contract_details_done = True
        print(f"  [IB {errorCode}] reqId={reqId}: {errorString}")


def _wait(flag_attr: str, app: IBHistApp, timeout: float) -> bool:
    deadline = time.time() + timeout
    while not getattr(app, flag_attr) and time.time() < deadline:
        time.sleep(0.05)
    return getattr(app, flag_attr)


def _parse_yyyymmdd(s: str) -> datetime | None:
    """IBKR's lastTradeDateOrContractMonth is yyyymmdd or yyyymm; parse both."""
    if not s:
        return None
    try:
        if len(s) >= 8:
            return datetime.strptime(s[:8], "%Y%m%d")
        if len(s) == 6:
            return datetime.strptime(s + "01", "%Y%m%d")
    except ValueError:
        return None
    return None


def _filter_chain(contracts: list, audit_start: str, audit_end: str) -> list:
    win_start = datetime.fromisoformat(audit_start) - timedelta(days=CHAIN_BUFFER_BEFORE_DAYS)
    win_end = datetime.fromisoformat(audit_end) + timedelta(days=CHAIN_BUFFER_AFTER_DAYS)
    out: list = []
    for c in contracts:
        ltd = _parse_yyyymmdd(c.lastTradeDateOrContractMonth or "")
        if ltd is None:
            continue
        if win_start <= ltd <= win_end:
            out.append(c)
    out.sort(
        key=lambda c: _parse_yyyymmdd(c.lastTradeDateOrContractMonth or "") or datetime(9999, 1, 1)
    )
    return out


def _fetch_chain(app: IBHistApp, req_id: int, root: FuturesRoot) -> list:
    c = Contract()
    c.secType = "FUT"
    c.symbol = root.root
    c.exchange = root.exchange
    c.currency = root.currency
    c.includeExpired = True  # critical: expose expired contracts
    app.contracts.clear()
    app.contract_details_done = False
    app.last_error = None
    app.reqContractDetails(req_id, c)
    ok = _wait("contract_details_done", app, 25)
    if not ok or not app.contracts:
        # Retry with ICE for softs.
        if root.exchange == "NYBOT":
            c.exchange = "ICE"
            app.contracts.clear()
            app.contract_details_done = False
            app.reqContractDetails(req_id + 1, c)
            _wait("contract_details_done", app, 25)
    return list(app.contracts)


def _fetch_history(
    app: IBHistApp,
    req_id: int,
    contract: Contract,
    end_dt: str,
    duration: str,
) -> list:
    app.bars.clear()
    app.bars_done = False
    app.last_error = None
    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_dt,
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )
    _wait("bars_done", app, 30)
    return list(app.bars)


def _bars_to_df(bars: list) -> pd.DataFrame:
    rows = []
    for b in bars:
        # b.date is yyyymmdd for daily bars.
        try:
            ts = datetime.strptime(b.date[:8], "%Y%m%d")
        except (ValueError, AttributeError):
            continue
        rows.append(
            {
                "timestamp": ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index = df.index.normalize()
    df.index.name = "timestamp"
    return df


def _save_contract(
    root: str, local_symbol: str, conId: int, last_trade: str, df: pd.DataFrame
) -> Path:
    root_dir = DATA_DIR / root
    root_dir.mkdir(parents=True, exist_ok=True)
    # Filename uses localSymbol when present, else conId fallback.
    fname = (local_symbol or f"conId{conId}").replace("/", "_") + "_D.parquet"
    out = root_dir / fname
    # Attach metadata into a parquet sidecar via pandas attrs not persisted,
    # so encode in a simple CSV index file as well.
    df.to_parquet(out)
    meta_file = root_dir / "_chain_meta.csv"
    row = pd.DataFrame(
        [
            {
                "root": root,
                "local_symbol": local_symbol,
                "conId": conId,
                "lastTradeDate": last_trade,
                "parquet": fname,
                "n_bars": len(df),
                "first_bar": df.index[0].date().isoformat() if len(df) else "",
                "last_bar": df.index[-1].date().isoformat() if len(df) else "",
            }
        ]
    )
    if meta_file.exists():
        old = pd.read_csv(meta_file)
        # De-duplicate by conId.
        old = old[old["conId"] != conId]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(meta_file, index=False)
    return out


def _process_root(
    app: IBHistApp,
    root: FuturesRoot,
    audit_start: str,
    audit_end: str,
    force: bool,
    req_counter: list[int],
) -> tuple[int, int]:
    """Returns (n_contracts_attempted, n_contracts_saved)."""
    print(f"\n=== {root.root} {root.name} ({root.exchange}/{root.currency}) ===")
    req_counter[0] += 1
    chain = _fetch_chain(app, req_counter[0], root)
    time.sleep(REQ_RATE_SECONDS)
    if not chain:
        print(f"  [!] no contracts returned for {root.root}@{root.exchange}")
        return (0, 0)
    print(f"  chain returned {len(chain)} contracts")
    filtered = _filter_chain(chain, audit_start, audit_end)
    print(f"  filtered to {len(filtered)} contracts in audit window")

    n_saved = 0
    root_dir = DATA_DIR / root.root
    for i, c in enumerate(filtered, 1):
        last_trade = c.lastTradeDateOrContractMonth or ""
        local = c.localSymbol or f"{root.root}{last_trade}"
        out = root_dir / (local.replace("/", "_") + "_D.parquet")
        if out.exists() and not force:
            print(f"  [{i}/{len(filtered)}] {local} -- cached, skipping")
            continue
        # Set request endpoint slightly past lastTradeDate so the contract's
        # full life of bars is available.
        ltd = _parse_yyyymmdd(last_trade)
        if ltd is not None:
            end_dt = (ltd + timedelta(days=2)).strftime("%Y%m%d-23:59:59")
        else:
            end_dt = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        req_counter[0] += 1
        bars = _fetch_history(app, req_counter[0], c, end_dt, "1 Y")
        time.sleep(REQ_RATE_SECONDS)
        df = _bars_to_df(bars)
        if df.empty:
            print(f"  [{i}/{len(filtered)}] {local} (exp {last_trade}) -- no bars returned")
            continue
        _save_contract(root.root, local, c.conId, last_trade, df)
        n_saved += 1
        print(
            f"  [{i}/{len(filtered)}] {local} (exp {last_trade})"
            f" -- {len(df)} bars  {df.index[0].date()} .. {df.index[-1].date()}"
        )
    return (len(filtered), n_saved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        default="CL",
        help="Comma-separated roots, or 'all' for the full 24-commodity universe",
    )
    parser.add_argument("--audit-start", default=DEFAULT_AUDIT_START)
    parser.add_argument("--audit-end", default=DEFAULT_AUDIT_END)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download contracts even if a cached parquet exists.",
    )
    args = parser.parse_args()

    if args.roots.lower() == "all":
        todo = list(UNIVERSE)
    else:
        wanted = {r.strip().upper() for r in args.roots.split(",") if r.strip()}
        todo = [r for r in UNIVERSE if r.root in wanted]
    if not todo:
        print(f"No roots matched '{args.roots}'")
        return 2

    print("=" * 70)
    print(
        f"  IBKR per-contract download: {len(todo)} roots, {args.audit_start} -> {args.audit_end}"
    )
    print(f"  host={IBKR_HOST}  port={IBKR_PORT}  client_id={CLIENT_ID}")
    print(f"  rate-limit: 1 req per {REQ_RATE_SECONDS}s")
    print("=" * 70)

    app = IBHistApp()
    app.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()
    if not _wait("ready", app, 8):
        print("Connection failed.")
        return 1
    print("Connected.\n")

    req_counter = [1000]
    succeeded: list[tuple[str, int, int]] = []
    failed: list[str] = []
    try:
        for root in todo:
            try:
                attempted, saved = _process_root(
                    app, root, args.audit_start, args.audit_end, args.force, req_counter
                )
                if saved == 0 and attempted == 0:
                    failed.append(root.root)
                else:
                    succeeded.append((root.root, attempted, saved))
            except Exception as exc:
                print(f"  ERROR processing {root.root}: {exc}")
                failed.append(root.root)
    finally:
        app.disconnect()

    print("\n" + "=" * 70)
    print(f"  Roots succeeded: {len(succeeded)}")
    for r, att, sav in succeeded:
        print(f"    {r:>3} : {sav} saved / {att} in-window")
    if failed:
        print(f"  Roots failed:    {len(failed)}  {' '.join(failed)}")
    print("=" * 70)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
