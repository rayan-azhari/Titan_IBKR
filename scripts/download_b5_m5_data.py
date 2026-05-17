"""download_b5_m5_data.py -- Pull SPY/QQQ/IWM M5 history from IBKR.

For B5 intraday momentum re-audit (V3.7). Targets 1+ year of M5 bars
per ticker (~19,656 bars/year at US equity RTH ~7h/day × 252 days).

IBKR paper account historical-bars limit per request:
  - 1m bars: up to 1 day per request
  - 5m bars: up to 30 days per request
  - Total depth: ~2 years for paper accounts (per L41)

Strategy: pull in 30-day chunks, save each as a chunk parquet, then
concatenate. Sleep 1.5s between requests to stay under IBKR's
50-req-per-10-min limit.

Run::

    IBKR_PORT=4004 PYTHONIOENCODING=utf-8 \
        uv run python scripts/download_b5_m5_data.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4004))
CLIENT_ID = 96  # distinct: gem=21, kill=98, flatten=97, probe=92, futures=95

# Targets: SPY/QQQ/IWM, US equity RTH
TICKERS = ["SPY", "QQQ", "IWM"]
# Pull 2 years (max paper depth, per L41)
LOOKBACK_DAYS = 730
CHUNK_DAYS = 28  # M5 limit per IBKR request (slightly under 30)
PACE_SECONDS = 1.5  # 1.5s between requests


def _us_equity_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.primaryExchange = "ARCA"
    c.currency = "USD"
    return c


class IBKRBarFetcher(EWrapper, EClient):
    """Single-request bar fetcher with completion event."""

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.bars: list[dict] = []
        self.done = threading.Event()
        self.error_msg: str | None = None
        self._next_id_evt = threading.Event()
        self._next_id = 1

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self._next_id = orderId
        self._next_id_evt.set()

    def historicalData(self, reqId: int, bar) -> None:  # noqa: N802, ARG002
        # Parse timestamp; IBKR returns "YYYYMMDD HH:MM:SS US/Eastern" for intraday
        ts_raw = bar.date.replace("US/Eastern", "").strip()
        try:
            ts = datetime.strptime(ts_raw, "%Y%m%d %H:%M:%S")
        except ValueError:
            try:
                ts = datetime.strptime(ts_raw, "%Y%m%d")
            except ValueError:
                ts = None
        self.bars.append({
            "timestamp": ts,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802, ARG002
        self.done.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: ARG002, ANN001
        if errorCode == 162:  # historical-data-service error
            self.error_msg = f"Code {errorCode}: {errorString}"
            self.done.set()
        elif errorCode in (2104, 2106, 2107, 2108, 2158):
            pass  # connection notifications
        elif errorCode == 502:
            print(f"  [error] {errorString}", file=sys.stderr)
            self.error_msg = errorString
            self.done.set()
        else:
            print(f"  [warn] code={errorCode}: {errorString}", file=sys.stderr)


def _connect() -> IBKRBarFetcher:
    fetcher = IBKRBarFetcher()
    fetcher.connect(IBKR_HOST, IBKR_PORT, CLIENT_ID)
    api_thread = threading.Thread(target=fetcher.run, daemon=True)
    api_thread.start()
    if not fetcher._next_id_evt.wait(timeout=20):
        print(f"  [error] connect timeout to {IBKR_HOST}:{IBKR_PORT}", file=sys.stderr)
        sys.exit(1)
    return fetcher


def _request_chunk(
    fetcher: IBKRBarFetcher,
    contract: Contract,
    end_dt: datetime,
    days: int,
) -> list[dict]:
    fetcher.bars = []
    fetcher.done.clear()
    fetcher.error_msg = None
    req_id = fetcher._next_id
    fetcher._next_id += 1
    # IBKR format: "YYYYMMDD-HH:MM:SS"  (UTC-naive end)
    end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")
    duration_str = f"{days} D"
    fetcher.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_str,
        durationStr=duration_str,
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=1,  # regular trading hours only
        formatDate=1,  # YYYYMMDD format
        keepUpToDate=False,
        chartOptions=[],
    )
    if not fetcher.done.wait(timeout=45):
        print("  [error] request timeout", file=sys.stderr)
        return []
    if fetcher.error_msg:
        print(f"  [error] {fetcher.error_msg}", file=sys.stderr)
        return []
    return fetcher.bars


def download_ticker(fetcher: IBKRBarFetcher, ticker: str) -> int:
    print(f"\n=== {ticker} ===")
    contract = _us_equity_contract(ticker)
    all_bars: list[dict] = []
    # Pull from "now" backwards in 28-day chunks
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    days_remaining = LOOKBACK_DAYS
    chunk_idx = 0
    while days_remaining > 0:
        chunk_idx += 1
        days = min(CHUNK_DAYS, days_remaining)
        print(f"  chunk {chunk_idx}: end={end_dt.strftime('%Y%m%d')}, duration={days}d", end="", flush=True)
        bars = _request_chunk(fetcher, contract, end_dt, days)
        n_bars = len(bars)
        print(f" -> {n_bars} bars")
        if n_bars == 0:
            print("  [warn] empty chunk, stopping")
            break
        all_bars.extend(bars)
        # Move end_dt back by `days`
        end_dt = end_dt - timedelta(days=days)
        days_remaining -= days
        time.sleep(PACE_SECONDS)

    if not all_bars:
        print(f"  [error] no bars retrieved for {ticker}")
        return 0

    df = pd.DataFrame(all_bars)
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    df = df.reset_index(drop=True)
    output_file = DATA_DIR / f"{ticker}_M5.parquet"
    df.to_parquet(output_file)
    print(f"  saved {len(df)} bars to {output_file}")
    print(f"  date range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    return len(df)


def main() -> int:
    print(f"Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT} (client_id={CLIENT_ID})...")
    fetcher = _connect()
    print("  connected")
    print(f"  targets: {TICKERS}, {LOOKBACK_DAYS}d lookback, {CHUNK_DAYS}d chunks")
    totals = {}
    for ticker in TICKERS:
        try:
            totals[ticker] = download_ticker(fetcher, ticker)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {ticker} failed: {e}")
            totals[ticker] = 0
    print("\n=== Summary ===")
    for tkr, n in totals.items():
        print(f"  {tkr}: {n} M5 bars")
    fetcher.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
