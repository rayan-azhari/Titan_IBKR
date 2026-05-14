"""download_data_phase_c.py — Phase C batch downloader (Databento).

Implements the V3.1 budget-discipline workflow for the Phase C IC Census
download itemised in:

    directives/IC Signal Census Phase C 2026-05-13.md

Two-step workflow, enforced by the script:

    1. ``--estimate``   Calls Databento ``metadata.get_cost`` for every
                        symbol-dataset-schema-range tuple. NO charges.
                        Prints an itemised cost table and the grand total.

    2. ``--confirm``    Actually invokes ``timeseries.get_range`` and writes
                        the parquets. Cannot be combined with ``--estimate``.

If neither flag is given, the script defaults to ``--estimate`` (safe-by-
default behaviour, no accidental spending). The script never charges
without an explicit ``--confirm`` flag from the operator.

Outputs (when --confirm):
    data/{INSTRUMENT}_D.parquet
    data/{INSTRUMENT}_H1.parquet

Each parquet has a UTC DatetimeIndex named ``timestamp`` and columns
``open, high, low, close, volume``. Same schema as Phase A/B parquets, so
the existing ``run_ic.load_ohlcv`` reads them without modification.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Phase C download manifest ─────────────────────────────────────────────
# Each entry: (local_name, dataset, symbol, stype, history_start)
# History starts are the earliest reasonable dates per Databento coverage
# notes. The actual download end is "today"; sanctuary slicing happens at
# scan time, not download time.
#
# stype_in:
#   "raw_symbol"  — fixed symbol (indices like VIX)
#   "continuous"  — continuous front-month futures (.c.0, .c.1)
#   "parent"      — parent symbol (rarely needed here)


@dataclass(frozen=True)
class Manifest:
    local: str        # parquet basename (e.g. "MES" → "MES_D.parquet" / "MES_H1.parquet")
    dataset: str
    symbol: str
    stype: str
    start: str
    note: str


PHASE_C_MANIFEST: list[Manifest] = [
    # As-built universe per directive §1.6 (Phase C 2026-05-13).
    # Eurex / ICE / CBOE entries from the original §1.1 were dropped after
    # the discovery audit (§1.5) -- Eurex EOBI has only 2 months of history
    # on Databento, ICE FTSE 100 symbology is undetermined, CBOE indices +
    # VIX futures are not in Databento's public catalogue.
    Manifest("MES",   "GLBX.MDP3", "MES.c.0", "continuous", "2019-05-06",
             "Micro E-mini S&P 500 front-month continuous (MES launched 2019-05)"),
    Manifest("MNQ",   "GLBX.MDP3", "MNQ.c.0", "continuous", "2019-05-06",
             "Micro E-mini Nasdaq-100 front-month continuous"),
    Manifest("ES",    "GLBX.MDP3", "ES.c.0",  "continuous", "2011-01-03",
             "E-mini S&P 500 front-month continuous (continuous definitions start 2011-01)"),
    Manifest("NQ",    "GLBX.MDP3", "NQ.c.0",  "continuous", "2011-01-03",
             "E-mini Nasdaq-100 front-month continuous (continuous definitions start 2011-01)"),
]


# ── Output writers ────────────────────────────────────────────────────────


def _normalise_to_parquet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Match the Phase A/B parquet schema: UTC index named 'timestamp',
    columns ['open', 'high', 'low', 'close', 'volume']."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df = df.set_index("ts_event")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    for col in df.columns:
        df[col] = df[col].astype(float)
    return df.sort_index().dropna(how="all")


# ── Cost estimate ─────────────────────────────────────────────────────────


def estimate_costs(client, end: str | None) -> list[dict]:
    """Call client.metadata.get_cost for each (manifest entry × schema).
    Returns a list of dicts ready to print as a table."""
    rows: list[dict] = []
    for m in PHASE_C_MANIFEST:
        for schema in ("ohlcv-1d", "ohlcv-1h"):
            try:
                cost = client.metadata.get_cost(
                    dataset=m.dataset,
                    schema=schema,
                    symbols=[m.symbol],
                    start=m.start,
                    end=end,
                    stype_in=m.stype,
                )
                # Databento's get_cost returns a float in USD.
                rows.append({
                    "local": m.local,
                    "schema": schema,
                    "dataset": m.dataset,
                    "symbol": m.symbol,
                    "start": m.start,
                    "cost_usd": float(cost),
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "local": m.local,
                    "schema": schema,
                    "dataset": m.dataset,
                    "symbol": m.symbol,
                    "start": m.start,
                    "cost_usd": None,
                    "error": str(exc),
                })
            time.sleep(0.05)  # gentle rate-limit
    return rows


def _print_cost_table(rows: list[dict]) -> float:
    """Print a per-symbol cost breakdown and return the grand total."""
    print()
    print("=" * 100)
    print("  DATABENTO COST ESTIMATE (no charge incurred)")
    print("=" * 100)
    header = f"  {'Local':<10} {'Schema':<11} {'Dataset':<14} {'Symbol':<10} {'Start':<12} {'Cost USD':>10}  Note"
    print(header)
    print("  " + "-" * 96)
    total = 0.0
    errors: list[dict] = []
    for r in rows:
        cost_str = f"{r['cost_usd']:>10.4f}" if r["cost_usd"] is not None else f"{'ERR':>10}"
        note = r["error"][:40] if r["error"] else ""
        print(f"  {r['local']:<10} {r['schema']:<11} {r['dataset']:<14} {r['symbol']:<10} {r['start']:<12} {cost_str}  {note}")
        if r["cost_usd"] is not None:
            total += r["cost_usd"]
        else:
            errors.append(r)
    print("  " + "-" * 96)
    print(f"  {'TOTAL':<59} {total:>10.4f} USD")
    print("=" * 100)
    if errors:
        print(f"\n  {len(errors)} symbol(s) failed cost estimate -- inspect 'error' column above.")
    return total


# ── Actual download ───────────────────────────────────────────────────────


def fetch_one(client, m: Manifest, schema: str, end: str | None) -> pd.DataFrame:
    print(f"  -> {m.local}_{('D' if schema == 'ohlcv-1d' else 'H1')} "
          f"({m.dataset} / {m.symbol} / {m.stype}) from {m.start} ...")
    data = client.timeseries.get_range(
        dataset=m.dataset,
        schema=schema,
        symbols=[m.symbol],
        start=m.start,
        end=end,
        stype_in=m.stype,
    )
    df = data.to_df()
    df = _normalise_to_parquet_frame(df)
    return df


def run_download(client, end: str | None) -> tuple[int, int]:
    """Download every manifest entry at both D and H1. Returns (ok, fail)."""
    ok = fail = 0
    for m in PHASE_C_MANIFEST:
        for schema, tf in (("ohlcv-1d", "D"), ("ohlcv-1h", "H1")):
            try:
                df = fetch_one(client, m, schema, end)
                if df.empty:
                    print(f"    WARNING: {m.local}_{tf} returned empty -- skipping write.")
                    fail += 1
                    continue
                out_path = DATA_DIR / f"{m.local}_{tf}.parquet"
                df.to_parquet(out_path)
                print(f"    Saved: {out_path.relative_to(PROJECT_ROOT)}  ({len(df):,} bars, "
                      f"{df.index[0].date()} to {df.index[-1].date()})")
                ok += 1
            except Exception as exc:
                print(f"    ERROR: {m.local}_{tf}: {exc}")
                fail += 1
            time.sleep(0.1)
    return ok, fail


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C IC Census Databento downloader (estimate-first)"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--estimate", action="store_true",
                     help="Call metadata.get_cost; print itemised costs; NO charges.")
    grp.add_argument("--confirm", action="store_true",
                     help="Actually invoke timeseries.get_range and write parquets.")
    parser.add_argument("--end", default=None,
                        help="End date ISO format (default: today, computed from UTC now). "
                             "Sanctuary handled at scan time, not here.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key or api_key.startswith("db-xxx"):
        print("ERROR: DATABENTO_API_KEY not set in .env -- bail out.")
        sys.exit(1)

    import databento as db
    client = db.Historical(key=api_key)

    # Default end to today (UTC) if not provided. Critical: Databento's
    # metadata.get_cost AND timeseries.get_range both interpret end=None
    # as "start + 1 day" (single-day query), NOT "end of available data".
    # Discovered the hard way on the first --confirm run -- see directive
    # §1.5 audit note.
    end = args.end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    # Default to estimate mode if neither flag set -- safe by default.
    if not args.confirm:
        rows = estimate_costs(client, end)
        total = _print_cost_table(rows)
        print()
        if not args.estimate:
            print("  (Default safe mode: --estimate. Pass --confirm to actually charge.)")
        else:
            print("  Estimate complete. To actually download, re-run with --confirm.")
        print(f"  Grand total estimate: ${total:.4f} USD")
        return

    # --confirm path: download for real.
    print("=" * 100)
    print("  DATABENTO DOWNLOAD -- ACTUAL CHARGES WILL BE INCURRED")
    print("=" * 100)
    print(f"  Manifest entries: {len(PHASE_C_MANIFEST)}")
    print(f"  Schemas per entry: 2 (ohlcv-1d, ohlcv-1h)")
    print(f"  Total parquet writes: {2 * len(PHASE_C_MANIFEST)}")
    print(f"  End date: {end}")
    print("=" * 100)
    ok, fail = run_download(client, end)
    print()
    print(f"  Download complete: {ok} parquet(s) written, {fail} failure(s).")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
