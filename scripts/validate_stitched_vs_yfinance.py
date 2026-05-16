"""validate_stitched_vs_yfinance.py -- L38 falsification check for the
roll-stitched continuous series produced by ``research.futures_stitching``.

For each requested commodity root, computes:
    - daily log returns of IBKR-stitched M1
    - daily log returns of yfinance M1 (``CL=F``, ``GC=F``, ...)
    - Pearson correlation over the overlap window
    - per-bar absolute differences at known yfinance roll dates

Pre-Reg gate (§3): if rho < 0.80 for a major commodity, abort the audit and
debug the stitching engine. Expected rho > 0.95 for CL, GC, ZC, ZW, ZS, ZL.

Usage::

    uv run python scripts/validate_stitched_vs_yfinance.py --root CL
    uv run python scripts/validate_stitched_vs_yfinance.py --root all

References:
    directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md §3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Yahoo Finance ticker mapping (from scripts/download_futures_yfinance.py)
YFINANCE_TICKERS: dict[str, str] = {
    "CL": "CL=F",
    "NG": "NG=F",
    "HO": "HO=F",
    "RB": "RB=F",
    "BZ": "BZ=F",
    "GC": "GC=F",
    "SI": "SI=F",
    "PL": "PL=F",
    "PA": "PA=F",
    "HG": "HG=F",
    "ZC": "ZC=F",
    "ZW": "ZW=F",
    "ZS": "ZS=F",
    "ZL": "ZL=F",
    "ZM": "ZM=F",
    "ZO": "ZO=F",
    "LE": "LE=F",
    "GF": "GF=F",
    "HE": "HE=F",
    "KC": "KC=F",
    "CC": "CC=F",
    "SB": "SB=F",
    "CT": "CT=F",
    "OJ": "OJ=F",
}

MIN_RHO_ABORT = 0.80  # below this, abort
TARGET_RHO = 0.95  # expected for clean major commodities


def _load_stitched(root: str) -> pd.Series | None:
    fp = DATA_DIR / f"{root}_M1_stitched_D.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def _load_yfinance_m1(root: str) -> pd.Series | None:
    fp = DATA_DIR / f"{root}_M1_D.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "close" not in df.columns:
        return None
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def _validate_one(root: str) -> dict:
    stitched = _load_stitched(root)
    yfin = _load_yfinance_m1(root)
    if stitched is None:
        return {"root": root, "status": "NO_STITCHED", "rho": np.nan, "n": 0}
    if yfin is None:
        return {"root": root, "status": "NO_YFINANCE", "rho": np.nan, "n": 0}
    common = stitched.index.intersection(yfin.index)
    if len(common) < 60:
        return {
            "root": root,
            "status": "INSUFFICIENT_OVERLAP",
            "rho": np.nan,
            "n": int(len(common)),
        }
    ret_s = np.log(stitched.loc[common]).diff().dropna()
    ret_y = np.log(yfin.loc[common]).diff().dropna()
    common_ret = ret_s.index.intersection(ret_y.index)
    rho = float(np.corrcoef(ret_s.loc[common_ret], ret_y.loc[common_ret])[0, 1])
    status = "PASS" if rho >= TARGET_RHO else "MARGINAL" if rho >= MIN_RHO_ABORT else "ABORT"
    return {
        "root": root,
        "status": status,
        "rho": rho,
        "n": int(len(common_ret)),
        "stitched_first": str(stitched.index[0].date()),
        "stitched_last": str(stitched.index[-1].date()),
        "yfinance_first": str(yfin.index[0].date()),
        "yfinance_last": str(yfin.index[-1].date()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="CL",
        help="Single root, comma-separated list, or 'all' for full universe",
    )
    args = parser.parse_args()

    if args.root.lower() == "all":
        roots = list(YFINANCE_TICKERS.keys())
    else:
        roots = [r.strip().upper() for r in args.root.split(",") if r.strip()]

    print("=" * 70)
    print(f"  Stitched-vs-yfinance L38 validation: {len(roots)} root(s)")
    print(f"  ABORT threshold: rho<{MIN_RHO_ABORT:.2f}   PASS threshold: rho>={TARGET_RHO:.2f}")
    print("=" * 70)

    rows: list[dict] = []
    for r in roots:
        result = _validate_one(r)
        rows.append(result)
        rho_str = f"{result['rho']:.4f}" if not np.isnan(result["rho"]) else "n/a"
        print(f"  {r:>3}  rho={rho_str:>8}  n={result['n']:>5}  {result['status']}")

    df = pd.DataFrame(rows)
    out = DATA_DIR / "stitched_validation.csv"
    df.to_csv(out, index=False)
    print(f"\n  Detailed CSV: {out.relative_to(PROJECT_ROOT)}")

    n_abort = int((df["status"] == "ABORT").sum())
    n_pass = int((df["status"] == "PASS").sum())
    print(f"\n  PASS: {n_pass}   ABORT: {n_abort}   other: {len(rows) - n_pass - n_abort}")
    return 0 if n_abort == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
