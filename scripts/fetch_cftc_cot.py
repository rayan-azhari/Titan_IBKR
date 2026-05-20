"""fetch_cftc_cot.py -- pull the CFTC Disaggregated Futures-Only CoT
report via the Socrata Open Data API.

Resource: ``72hh-3qpy`` -- the historical CoT Disaggregated report.
Endpoint: https://publicreporting.cftc.gov/resource/72hh-3qpy.json

The Disaggregated report (starts 2006-06-13) splits open interest into:
    - Producer / Merchant / Processor / User (commercial hedgers)
    - Swap Dealers
    - Managed Money (speculators -- the KRT 2020 signal source)
    - Other Reportables
    - Non-Reportables

We pull weekly Tuesday-close positions for the 15 commodities in the F2
pre-reg universe, save as a long-format parquet keyed by
``(report_date, commodity)``.

Pre-reg: `directives/Pre-Reg F2 CFTC CoT Positioning 2026-05-20.md` §6 step 2.

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/fetch_cftc_cot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Resource ``6dca-aqww`` is the LEGACY Futures-Only CoT report. Discovery
# notes from the 2026-05-20 build:
#   * Disaggregated resources (``kh3c-gbw2``, ``72hh-3qpy``, ``rxbv-e226``)
#     all stop at 2022-02-01 for the main NYMEX WTI / Brent / NG contracts
#     -- a CFTC infrastructure boundary -- and the corresponding "current"
#     resources (``ubmb-6exi``, ``gr4m-cvuh``) return HTTP 403 on the
#     public Socrata endpoint.
#   * Legacy ``6dca-aqww`` runs continuously 1986-01-15 to present (282k
#     rows at fetch time).
#   * Trade-off: Legacy uses 3 categories (Non-Commercial / Commercial /
#     Non-Reportable) vs Disaggregated's 5. Non-Commercial is a near-
#     equivalent superset of Managed Money for the KRT 2020 speculator
#     signal -- academic literature commonly substitutes when needed.
#   * Net win: 40 years of continuous history >> 16 years split across
#     two name variants. The audit pre-reg notes the substitution.
CFTC_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# Map our internal symbol -> CFTC commodity name substring(s). The CFTC's
# `market_and_exchange_names` field is verbose (e.g. "GOLD - COMMODITY
# EXCHANGE INC.") so we use substring match. Multiple variants for some
# commodities to handle exchange-name changes / contract variants.
# Filter strings are verified against the actual CFTC market_and_exchange_names
# field on resource ``kh3c-gbw2`` (2026-05-20 discovery scan). Multiple variants
# per commodity handle exchange-name changes over time:
#   - HG (Copper): CFTC renamed "COPPER-GRADE #1" -> "COPPER- #1" in 2022-02
#   - ZW (Wheat-SRW): CBOT relabelled "WHEAT" -> "WHEAT-SRW" in 2013-12
#   - CT/KC/SB/CC: NYBOT was absorbed into ICE U.S. in 2007-09 (name flip)
#   - NG: the main NYMEX nat-gas contract is named "HENRY HUB", NOT "NATURAL GAS"
#   - CL: WTI's CFTC name is "CRUDE OIL, LIGHT SWEET" (no "-WTI" suffix)
CFTC_COMMODITY_FILTERS: dict[str, list[str]] = {
    # Energy
    "CL": ["CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE"],
    "BZ": [
        "BRENT LAST DAY - NEW YORK MERCANTILE EXCHANGE",
        "BRENT CRUDE OIL LAST DAY - NEW YORK MERCANTILE EXCHANGE",
    ],
    "NG": [
        "HENRY HUB - NEW YORK MERCANTILE EXCHANGE",
        "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    ],
    # Metals
    "GC": ["GOLD - COMMODITY EXCHANGE INC."],
    "SI": ["SILVER - COMMODITY EXCHANGE INC."],
    "HG": [
        "COPPER- #1 - COMMODITY EXCHANGE INC.",  # 2022-02 onwards
        "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",  # 2006-06 to 2022-02
    ],
    "PL": ["PLATINUM - NEW YORK MERCANTILE EXCHANGE"],
    "PA": ["PALLADIUM - NEW YORK MERCANTILE EXCHANGE"],
    # Grains
    "ZC": ["CORN - CHICAGO BOARD OF TRADE"],
    "ZW": [
        "WHEAT-SRW - CHICAGO BOARD OF TRADE",  # 2013-12 onwards (SRW only)
        "WHEAT - CHICAGO BOARD OF TRADE",  # 2006-06 to 2013-12 (umbrella name)
    ],
    "ZS": ["SOYBEANS - CHICAGO BOARD OF TRADE"],
    # Softs (NYBOT absorbed by ICE U.S. on 2007-09-04)
    "CT": [
        "COTTON NO. 2 - ICE FUTURES U.S.",
        "COTTON NO. 2 - NEW YORK BOARD OF TRADE",
    ],
    "KC": ["COFFEE C - ICE FUTURES U.S.", "COFFEE C - NEW YORK BOARD OF TRADE"],
    "SB": [
        "SUGAR NO. 11 - ICE FUTURES U.S.",
        "SUGAR NO. 11 - NEW YORK BOARD OF TRADE",
    ],
    "CC": ["COCOA - ICE FUTURES U.S.", "COCOA - NEW YORK BOARD OF TRADE"],
}

# CFTC API row limit per request; we page to fetch all history.
PAGE_LIMIT = 50_000
START_DATE = "2006-06-13"  # Disaggregated report start
REQ_PAUSE_S = 0.5


def fetch_for_commodity(symbol: str, name_variants: list[str]) -> pd.DataFrame:
    """Fetch all weekly CoT rows for a given commodity (any name variant)."""
    # SoQL OR clause across name variants.
    name_filter = " OR ".join(f"market_and_exchange_names = '{v}'" for v in name_variants)
    where_clause = f"({name_filter}) AND report_date_as_yyyy_mm_dd >= '{START_DATE}'"

    all_rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "$where": where_clause,
            "$limit": PAGE_LIMIT,
            "$offset": offset,
            "$order": "report_date_as_yyyy_mm_dd ASC",
        }
        try:
            r = requests.get(CFTC_URL, params=params, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"    HTTP error: {exc}")
            return pd.DataFrame()
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
            return pd.DataFrame()
        page = r.json()
        if not page:
            break
        all_rows.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(REQ_PAUSE_S)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["report_date"] = (
        pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None).dt.normalize()
    )
    df["symbol"] = symbol

    # Keep only the fields needed for the KRT 2020 speculator-positioning signal.
    # Legacy report uses Non-Commercial / Commercial / Non-Reportable.
    # `noncomm_positions_long_all` is the speculator-side measure (Hedge
    # funds + CTAs + other large speculators) -- treated as a near-
    # equivalent superset of the Disaggregated report's Managed Money.
    fields = [
        "symbol",
        "report_date",
        "market_and_exchange_names",
        "open_interest_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
    ]
    df = df[[c for c in fields if c in df.columns]].copy()

    # Coerce numeric.
    for col in df.columns:
        if col in ("symbol", "report_date", "market_and_exchange_names"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("report_date").reset_index(drop=True)
    return df


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  F2 CFTC CoT Disaggregated fetch: {len(CFTC_COMMODITY_FILTERS)} commodities")
    print(f"  Window: {START_DATE} -> latest available")
    print(f"  Source: {CFTC_URL}")
    print("=" * 72)

    all_dfs: list[pd.DataFrame] = []
    for sym, variants in CFTC_COMMODITY_FILTERS.items():
        print(f"  {sym}", end=": ")
        df = fetch_for_commodity(sym, variants)
        if df.empty:
            print("NO DATA (check filter variants)")
            continue
        n = len(df)
        first = df["report_date"].min().date()
        last = df["report_date"].max().date()
        print(f"{n} reports  {first} .. {last}")
        all_dfs.append(df)
        time.sleep(REQ_PAUSE_S)

    if not all_dfs:
        print("\n  No data fetched -- aborting save.")
        return 2

    combined = pd.concat(all_dfs, ignore_index=True)
    out = DATA_DIR / "cftc_cot_disaggregated.parquet"
    combined.to_parquet(out)
    print(f"\n  Saved: {out.relative_to(PROJECT_ROOT)} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
