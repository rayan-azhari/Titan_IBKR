"""download_futures_databento.py -- Pull continuous-contract daily OHLCV
for the BGR-2019 commodity universe.

Downloads M1 (front month) AND M2 (second month) daily bars for the 24
commodity universe used in Bakshi, Gao & Rossi *Management Science* 2019,
which is required for D2 backlog item (Commodity Futures Carry).

Carry definition (typical academic form):
    carry_t = log(F_M1_t / F_M2_t)  -- annualised by months-to-expiry
or simpler:
    carry_t = F_M1_t / F_M2_t - 1   -- basis spread

Both series are needed to compute either.

Output:
    data/{ROOT}_M1_D.parquet  -- nearest-to-expiry continuous contract
    data/{ROOT}_M2_D.parquet  -- second-nearest continuous contract

Where ROOT is the CME/ICE product root (CL, GC, ZC, KC, ...).

Databento symbology:
    Continuous contracts use stype_in="continuous". The symbol pattern is
    "{ROOT}.c.{N}" where N=0 is the front month and N=1 is the next.

Usage::

    uv run python scripts/download_futures_databento.py
    uv run python scripts/download_futures_databento.py --groups CME,ICE
    uv run python scripts/download_futures_databento.py --roots CL,GC,ZC
    uv run python scripts/download_futures_databento.py --start 2015-01-01

Requires:
    DATABENTO_API_KEY in .env. Subscription must cover GLBX.MDP3 (CME)
    and IFEU.IMPACT + IFUS.IMPACT (ICE EU + US) for the full universe.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default history start. Databento GLBX.MDP3 begins 2010-06-06; ICE
# datasets begin 2020-12-04 (per Databento docs). The script per-root
# clamps to the dataset's earliest available date if the user passes a
# stricter start. Using 2010-06-06 here as the CME-safe global start.
DEFAULT_START = "2010-06-06"

# Dataset-specific earliest CONTINUOUS-CONTRACT availability. The raw
# contract data on GLBX.MDP3 goes back to 2010-06-06 but the continuous-
# symbology mapping (X.c.0 / X.n.0) starts later, ~2017-05-22. ICE
# continuous datasets start ~2020-12. We clamp the requested start per
# dataset so we don't hit "did not resolve" / 422.
DATASET_EARLIEST = {
    "GLBX.MDP3": "2017-06-01",  # CME Globex continuous symbology
    "IFEU.IMPACT": "2020-12-04",  # ICE Futures Europe (Brent)
    "IFUS.IMPACT": "2020-12-04",  # ICE Futures US (softs)
}


@dataclass(frozen=True)
class FuturesRoot:
    """One commodity in the BGR-2019 universe."""

    root: str  # e.g. "CL"
    name: str  # e.g. "WTI Crude Oil"
    dataset: str  # Databento dataset code for the venue
    sector: str  # for reporting only


# 24-commodity universe — Bakshi-Gao-Rossi MS 2019 (also used in AQR
# commodity carry papers). Mostly CME GLBX; the softs subset is on ICE.
UNIVERSE: tuple[FuturesRoot, ...] = (
    # ── Energy (4 CME + 1 ICE EU = 5) ─────────────────────────────────
    FuturesRoot("CL", "WTI Crude Oil", "GLBX.MDP3", "energy"),
    FuturesRoot("NG", "Natural Gas", "GLBX.MDP3", "energy"),
    FuturesRoot("HO", "NY Harbor ULSD (Heating Oil)", "GLBX.MDP3", "energy"),
    FuturesRoot("RB", "RBOB Gasoline", "GLBX.MDP3", "energy"),
    FuturesRoot("BZ", "Brent Crude Oil", "IFEU.IMPACT", "energy"),
    # ── Metals -- precious (4) + industrial (1) = 5 (all CME) ─────────
    FuturesRoot("GC", "Gold", "GLBX.MDP3", "metals_precious"),
    FuturesRoot("SI", "Silver", "GLBX.MDP3", "metals_precious"),
    FuturesRoot("PL", "Platinum", "GLBX.MDP3", "metals_precious"),
    FuturesRoot("PA", "Palladium", "GLBX.MDP3", "metals_precious"),
    FuturesRoot("HG", "Copper", "GLBX.MDP3", "metals_industrial"),
    # ── Grains + Oilseeds (6 CME) ─────────────────────────────────────
    FuturesRoot("ZC", "Corn", "GLBX.MDP3", "grains"),
    FuturesRoot("ZW", "Chicago Wheat", "GLBX.MDP3", "grains"),
    FuturesRoot("ZS", "Soybeans", "GLBX.MDP3", "grains"),
    FuturesRoot("ZL", "Soybean Oil", "GLBX.MDP3", "grains"),
    FuturesRoot("ZM", "Soybean Meal", "GLBX.MDP3", "grains"),
    FuturesRoot("ZO", "Oats", "GLBX.MDP3", "grains"),
    # ── Livestock (3 CME) ─────────────────────────────────────────────
    FuturesRoot("LE", "Live Cattle", "GLBX.MDP3", "livestock"),
    FuturesRoot("GF", "Feeder Cattle", "GLBX.MDP3", "livestock"),
    FuturesRoot("HE", "Lean Hogs", "GLBX.MDP3", "livestock"),
    # ── Softs (5 ICE US) ──────────────────────────────────────────────
    FuturesRoot("KC", "Coffee", "IFUS.IMPACT", "softs"),
    FuturesRoot("CC", "Cocoa", "IFUS.IMPACT", "softs"),
    FuturesRoot("SB", "Sugar No. 11", "IFUS.IMPACT", "softs"),
    FuturesRoot("CT", "Cotton No. 2", "IFUS.IMPACT", "softs"),
    FuturesRoot("OJ", "Orange Juice", "IFUS.IMPACT", "softs"),
)  # total = 24


def download_continuous(
    client,
    root: str,
    contract_n: int,
    *,
    start: str,
    end: str | None,
    dataset: str,
) -> pd.DataFrame:
    """Pull `{root}.c.{contract_n}` daily OHLCV from Databento.

    contract_n=0 -> front month; contract_n=1 -> next; etc.
    Returns empty DataFrame on failure (caller decides how to handle).
    """
    sym = f"{root}.c.{contract_n}"
    # Clamp start to dataset's earliest available date (Databento returns
    # 422 data_start_before_available_start otherwise).
    earliest = DATASET_EARLIEST.get(dataset)
    effective_start = max(start, earliest) if earliest else start
    if effective_start != start:
        print(
            f"  [{dataset}] {sym} from {effective_start} (clamped from {start}) to {end or 'today'} ..."
        )
    else:
        print(f"  [{dataset}] {sym} from {effective_start} to {end or 'today'} ...")
    try:
        data = client.timeseries.get_range(
            dataset=dataset,
            schema="ohlcv-1d",
            symbols=[sym],
            start=effective_start,
            end=end,
            stype_in="continuous",
        )
        df = data.to_df()
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return pd.DataFrame()

    if df.empty:
        print("    (no rows returned)")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df = df.set_index("ts_event")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype(float).sort_index().dropna(how="all")
    print(f"    -> {len(df)} bars  {df.index[0].date()} .. {df.index[-1].date()}")
    return df


def already_complete(root: str) -> bool:
    """True when both M1 + M2 parquets exist for this root."""
    return (DATA_DIR / f"{root}_M1_D.parquet").exists() and (
        DATA_DIR / f"{root}_M2_D.parquet"
    ).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        default="",
        help="Comma-separated subset of root symbols (default: full 24-commodity universe)",
    )
    parser.add_argument(
        "--groups",
        default="CME,ICE",
        help="Venues to include (CME=GLBX.MDP3, ICE=IFEU.IMPACT+IFUS.IMPACT)",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    # Databento ohlcv-1d requires an explicit end-date; passing None or
    # leaving the default returns only 1 boundary bar. Default to
    # yesterday (UTC) to avoid the same-day incomplete-bar edge case.
    default_end = (date.today() - timedelta(days=1)).isoformat()
    parser.add_argument("--end", default=default_end)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip roots whose M1+M2 parquets are already present (default on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-download even when parquets exist.",
    )
    parser.add_argument(
        "--m1-only",
        action="store_true",
        default=False,
        help=(
            "Skip M2 downloads (Databento .c.1 queries are very slow on "
            "GLBX.MDP3). Carry can be proxied via trailing 12-month return "
            "on M1 -- BGR 2019 §3.2 shows ρ~0.65 with strict M1/M2 carry."
        ),
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or api_key.startswith("db-xxx"):
        print("ERROR: DATABENTO_API_KEY not set / placeholder. Add it to .env.")
        return 1

    try:
        import databento as db
    except ImportError:
        print("ERROR: databento not installed. Run: uv add databento")
        return 1

    client = db.Historical(key=api_key)

    # Filter universe.
    selected_roots = (
        {r.strip().upper() for r in args.roots.split(",") if r.strip()} if args.roots else None
    )
    selected_groups = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    dataset_in_group = {
        "GLBX.MDP3": "CME",
        "IFEU.IMPACT": "ICE",
        "IFUS.IMPACT": "ICE",
    }

    todo: list[FuturesRoot] = []
    for fr in UNIVERSE:
        if selected_roots is not None and fr.root not in selected_roots:
            continue
        if dataset_in_group.get(fr.dataset, "") not in selected_groups:
            continue
        todo.append(fr)

    print("=" * 70)
    print("  Databento futures-carry universe download (BGR-2019, D2 audit)")
    print(f"  Roots:  {len(todo)} / {len(UNIVERSE)}  ({', '.join(r.root for r in todo)})")
    print(f"  Period: {args.start} to {args.end or 'today'}")
    print(f"  Skip already-downloaded: {args.skip_existing}")
    print("=" * 70)

    succeeded: list[str] = []
    failed_full: list[str] = []
    failed_partial: list[str] = []  # M1 ok but M2 missing
    skipped: list[str] = []

    for fr in todo:
        print(f"\n[{fr.root}] {fr.name}  sector={fr.sector}  dataset={fr.dataset}")

        if args.skip_existing and already_complete(fr.root):
            print("  Already complete -- skipping.")
            skipped.append(fr.root)
            continue

        # Front month (M1) and next month (M2).
        df_m1 = download_continuous(
            client, fr.root, 0, start=args.start, end=args.end, dataset=fr.dataset
        )
        if args.m1_only:
            df_m2 = pd.DataFrame()
        else:
            df_m2 = download_continuous(
                client, fr.root, 1, start=args.start, end=args.end, dataset=fr.dataset
            )

        if not df_m1.empty:
            out = DATA_DIR / f"{fr.root}_M1_D.parquet"
            df_m1.to_parquet(out)
            print(f"    saved: {out.relative_to(PROJECT_ROOT)}")
        if not df_m2.empty:
            out = DATA_DIR / f"{fr.root}_M2_D.parquet"
            df_m2.to_parquet(out)
            print(f"    saved: {out.relative_to(PROJECT_ROOT)}")

        if df_m1.empty and df_m2.empty:
            failed_full.append(fr.root)
            continue

        if df_m1.empty or df_m2.empty:
            failed_partial.append(fr.root)
        else:
            succeeded.append(fr.root)

        # Gentle rate-limit gap between roots.
        time.sleep(0.4)

    print("\n" + "=" * 70)
    print(f"  Succeeded (M1+M2): {len(succeeded):>2}  {' '.join(succeeded)}")
    print(f"  Partial:           {len(failed_partial):>2}  {' '.join(failed_partial)}")
    print(f"  Full failure:      {len(failed_full):>2}  {' '.join(failed_full)}")
    print(f"  Skipped:           {len(skipped):>2}  {' '.join(skipped)}")
    print("=" * 70)
    return 0 if not failed_full else 2


if __name__ == "__main__":
    sys.exit(main())
