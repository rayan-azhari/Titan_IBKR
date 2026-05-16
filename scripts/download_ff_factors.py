"""download_ff_factors.py -- Fetch Fama-French 3-factor daily returns from
Ken French's data library.

Output:
    data/FF3_daily.parquet  -- columns: mkt_rf, smb, hml, rf (all daily decimals)

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/
        Specifically: F-F_Research_Data_Factors_daily_CSV.zip

The factors are in *percent*; we convert to decimal so they're directly
addable to log returns.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)


def main() -> int:
    print(f"Downloading {URL} ...")
    req = urllib.request.Request(
        URL,
        headers={
            # Some hosts gate on a non-default UA.
            "User-Agent": "Mozilla/5.0 (titan-ibkr/A1 research)"
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        zbuf = io.BytesIO(resp.read())

    with zipfile.ZipFile(zbuf) as zf:
        names = zf.namelist()
        csv_name = next(n for n in names if n.lower().endswith(".csv"))
        print(f"  unzipped: {csv_name}")
        with zf.open(csv_name) as f:
            raw = f.read().decode("latin-1")

    # The CSV has a multi-line header / preamble; data rows are 8-digit dates.
    lines = raw.splitlines()
    data_lines: list[str] = []
    started = False
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            if started:
                # Annual-data section comes after the daily section, separated by blanks.
                break
            continue
        if not started:
            # Header line is the first one containing "Mkt-RF" (it's the column header).
            if "Mkt-RF" in stripped:
                data_lines.append(stripped)
                started = True
            continue
        if stripped[0].isdigit():
            data_lines.append(stripped)
        else:
            break

    print(f"  parsed {len(data_lines) - 1} data rows from CSV")
    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() for c in df.columns]
    # First column is unnamed; pandas calls it "Unnamed: 0".
    date_col = df.columns[0]
    df = df.rename(
        columns={
            date_col: "date",
            "Mkt-RF": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RF": "rf",
        }
    )
    # Parse YYYYMMDD ints into dates.
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()
    # Convert from percent to decimal.
    for col in ("mkt_rf", "smb", "hml", "rf"):
        df[col] = df[col].astype(float) / 100.0

    out = DATA_DIR / "FF3_daily.parquet"
    df.to_parquet(out)
    print(
        f"  saved: {out.relative_to(PROJECT_ROOT)}  "
        f"({len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
