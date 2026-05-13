"""download_data_phase_c_ibkr.py -- Phase C VIX-family intraday via IBKR Gateway.

Per directive directives/IC Signal Census Phase C 2026-05-13.md §1.7.

Best-effort attempt to pull CBOE VIX, VIX9D, VIX3M indices at D + H1 from
IBKR. Requires the **CBOE Indices Live** market-data subscription on the
account (free for paper accounts in some regions; paid live ~$1.50/mo).

Behaviour:
    * Tries IBKR ports in order: 4004 (Docker paper), 4002 (Gateway paper),
      7497 (TWS paper). Reports which one accepted the connection.
    * For each (instrument, timeframe) pair, walks history backwards in
      365-day chunks (IBKR's max for hourly bars on indices) until the
      broker returns no data.
    * If any request returns error 354 ("no market data subscription") or
      200 ("no security definition"), the script STOPS cleanly with a
      clear report -- no partial state, no half-written parquets.

Output naming follows Phase A/B convention so existing run_ic.load_ohlcv
reads the files without modification:
    data/VIX_D.parquet       (overwrites existing yfinance daily file)
    data/VIX_H1.parquet      (new)
    data/VIX9D_D.parquet     (overwrites existing yfinance daily file)
    data/VIX9D_H1.parquet    (new)
    data/VIX3M_D.parquet     (overwrites existing yfinance daily file)
    data/VIX3M_H1.parquet    (new)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class IndexSpec:
    local: str
    symbol: str
    exchange: str = "CBOE"
    currency: str = "USD"
    sec_type: str = "IND"


VIX_FAMILY: list[IndexSpec] = [
    IndexSpec("VIX",   "VIX"),
    IndexSpec("VIX9D", "VIX9D"),
    IndexSpec("VIX3M", "VIX3M"),
]

# IBKR historical-bar size strings.
BAR_SIZE = {"D": "1 day", "H1": "1 hour"}

# IBKR duration chunks. Max for hourly bars on indices is "1 Y".
DURATION_PER_TF = {"D": "20 Y", "H1": "1 Y"}

# How far back to walk overall. VIX9D only started 2011; VIX itself goes
# back further but we cap at 2010 to match the Databento window.
HISTORY_FROM = datetime(2010, 1, 1, tzinfo=timezone.utc)


class IBKRHistoricalFetcher(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.bars: dict[int, list] = {}
        self.req_complete: dict[int, str] = {}
        self.connection_event = threading.Event()
        self.fatal_error: str | None = None
        self.error_codes_for_req: dict[int, list[int]] = {}

    def nextValidId(self, orderId: int) -> None:
        self.connection_event.set()

    def error(  # type: ignore[override]
        self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""
    ) -> None:
        # Informational errors -- ignore.
        if errorCode in (2104, 2106, 2158, 2107, 2108, 2168, 2169):
            return
        # Per-request errors.
        if reqId >= 0:
            self.error_codes_for_req.setdefault(reqId, []).append(errorCode)
            print(f"  [IBKR err reqId={reqId} code={errorCode}] {errorString}")
            # Fatal: no security definition (200), no market data sub (354),
            # historical data request pacing violation (162 with bad subset).
            if errorCode in (200, 354):
                self.fatal_error = f"code={errorCode}: {errorString}"
                # Unblock any waiting requests.
                self.req_complete[reqId] = "fatal"
        else:
            print(f"  [IBKR connection err {errorCode}] {errorString}")
            if errorCode in (502, 504, 507):
                self.fatal_error = f"code={errorCode}: {errorString}"
                self.connection_event.set()  # don't hang

    def historicalData(self, reqId: int, bar) -> None:  # noqa: ANN001
        self.bars.setdefault(reqId, []).append({
            "date": bar.date,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume) if bar.volume >= 0 else 0.0,
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self.req_complete[reqId] = "ok"


def _make_contract(spec: IndexSpec) -> Contract:
    c = Contract()
    c.symbol = spec.symbol
    c.secType = spec.sec_type
    c.exchange = spec.exchange
    c.currency = spec.currency
    return c


def _connect_with_fallback(app: IBKRHistoricalFetcher) -> int | None:
    load_dotenv(PROJECT_ROOT / ".env")
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    candidates: list[int] = []
    env_port = os.environ.get("IBKR_PORT")
    if env_port:
        candidates.append(int(env_port))
    for p in (4004, 4002, 7497, 4001):
        if p not in candidates:
            candidates.append(p)

    for port in candidates:
        print(f"  Trying {host}:{port} (client_id=99) ...")
        try:
            app.connect(host, port, clientId=99)
        except Exception as exc:
            print(f"    connect raised: {exc}")
            continue
        thread = threading.Thread(target=app.run, daemon=True)
        thread.start()
        if app.connection_event.wait(timeout=5):
            if app.fatal_error:
                print(f"    rejected: {app.fatal_error}")
                app.disconnect()
                thread.join(timeout=2)
                # Reset for next attempt.
                app.connection_event.clear()
                app.fatal_error = None
                continue
            print(f"    Connected on port {port}.")
            return port
        app.disconnect()
        thread.join(timeout=2)
        app.connection_event.clear()
    return None


def _parse_ibkr_datetime(s: str) -> pd.Timestamp:
    """IBKR returns either 'YYYYMMDD' (daily) or 'YYYYMMDD HH:MM:SS TZ' (intraday)."""
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]), tz="UTC")
    # Drop trailing tz name if present; treat as US/Eastern then convert.
    parts = s.split()
    if len(parts) >= 2:
        date_part = parts[0]
        time_part = parts[1]
        # parts[2] may be 'US/Eastern' or 'America/New_York' etc.
        tz = parts[2] if len(parts) >= 3 else "US/Eastern"
        ts = pd.Timestamp(f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part}", tz=tz)
        return ts.tz_convert("UTC")
    raise ValueError(f"Unrecognised IBKR date string: {s!r}")


def _fetch_chunk(
    app: IBKRHistoricalFetcher,
    spec: IndexSpec,
    tf: str,
    end: datetime,
    duration: str,
    req_id: int,
    timeout_s: float = 90.0,
) -> list[dict]:
    contract = _make_contract(spec)
    end_str = end.strftime("%Y%m%d %H:%M:%S UTC")
    print(f"    {spec.local}_{tf} chunk req={req_id} end={end_str} duration={duration} ...")
    app.bars.pop(req_id, None)
    app.req_complete.pop(req_id, None)
    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_str,
        durationStr=duration,
        barSizeSetting=BAR_SIZE[tf],
        whatToShow="TRADES",
        useRTH=0,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if app.req_complete.get(req_id) in ("ok", "fatal"):
            break
        time.sleep(0.2)
    if app.fatal_error:
        raise RuntimeError(f"Fatal IBKR error: {app.fatal_error}")
    bars = app.bars.get(req_id, [])
    print(f"      got {len(bars)} bars.")
    return bars


def _fetch_full_history(
    app: IBKRHistoricalFetcher,
    spec: IndexSpec,
    tf: str,
    req_id_start: int,
) -> pd.DataFrame:
    duration = DURATION_PER_TF[tf]
    end = datetime.now(timezone.utc)
    all_bars: list[dict] = []
    req_id = req_id_start
    while end > HISTORY_FROM:
        bars = _fetch_chunk(app, spec, tf, end, duration, req_id)
        if not bars:
            break
        all_bars.extend(bars)
        # Walk backwards by the chunk length.
        try:
            earliest_str = bars[0]["date"]
            earliest_ts = _parse_ibkr_datetime(earliest_str)
            new_end = earliest_ts.to_pydatetime()
        except Exception:
            break
        if new_end >= end - timedelta(days=1):
            # No progress -- bail to avoid an infinite loop.
            break
        end = new_end
        req_id += 1
        time.sleep(0.5)  # gentle pacing
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars)
    df["timestamp"] = df["date"].apply(_parse_ibkr_datetime)
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


def main() -> None:
    app = IBKRHistoricalFetcher()
    port = _connect_with_fallback(app)
    if port is None or app.fatal_error:
        print("\nERROR: could not connect to any IBKR Gateway / TWS port.")
        print("       Confirm Docker Gateway is running and reachable.")
        sys.exit(1)

    req_id = 1
    failures: list[str] = []
    for spec in VIX_FAMILY:
        for tf in ("D", "H1"):
            try:
                df = _fetch_full_history(app, spec, tf, req_id_start=req_id)
                req_id += 100  # leave room for chunk req ids
                if df.empty:
                    failures.append(f"{spec.local}_{tf}: empty")
                    print(f"  WARNING: {spec.local}_{tf} returned no bars.")
                    continue
                out_path = DATA_DIR / f"{spec.local}_{tf}.parquet"
                df.to_parquet(out_path)
                print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}  "
                      f"({len(df):,} bars, {df.index[0].date()} to {df.index[-1].date()})")
            except RuntimeError as exc:
                print(f"\nFATAL: {spec.local}_{tf}: {exc}")
                print("Stopping all further downloads -- partial state may have been written.")
                failures.append(f"{spec.local}_{tf}: fatal")
                app.disconnect()
                if failures:
                    sys.exit(2)
                return

    app.disconnect()
    print()
    if failures:
        print(f"  {len(failures)} symbol-TF combinations failed:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  All VIX-family parquets written.")


if __name__ == "__main__":
    main()
