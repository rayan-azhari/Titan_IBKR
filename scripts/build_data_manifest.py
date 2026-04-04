"""build_data_manifest.py -- Scan data/ and write data/manifest.json.

Creates a machine-readable catalog of all available market data:
    - Symbol, timeframe, source (inferred from filename)
    - Date range (first/last bar)
    - Bar count, file size
    - Last modified timestamp (proxy for freshness)

Usage:
    uv run python scripts/build_data_manifest.py

Called automatically by download scripts after data updates.
Output: data/manifest.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def scan_parquets() -> list[dict]:
    """Scan all parquet files in data/ and extract metadata."""
    entries = []

    for path in sorted(DATA_DIR.glob("*.parquet")):
        stem = path.stem  # e.g. "EUR_USD_H1"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue

        symbol = parts[0]
        timeframe = parts[1]

        try:
            df = pd.read_parquet(path)
            bar_count = len(df)
            if bar_count == 0:
                continue

            # Date range
            idx = df.index
            first_bar = str(idx[0])
            last_bar = str(idx[-1])

            # Columns
            cols = list(df.columns)

            # File metadata
            stat = path.stat()
            file_size_kb = round(stat.st_size / 1024, 1)
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            entries.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "file": path.name,
                    "bars": bar_count,
                    "first_bar": first_bar,
                    "last_bar": last_bar,
                    "columns": cols,
                    "size_kb": file_size_kb,
                    "modified_utc": modified,
                }
            )
        except Exception as e:
            print(f"  WARN: {path.name}: {e}", file=sys.stderr)

    return entries


def build_summary(entries: list[dict]) -> dict:
    """Build summary stats from entries."""
    symbols = sorted(set(e["symbol"] for e in entries))
    timeframes = sorted(set(e["timeframe"] for e in entries))
    total_bars = sum(e["bars"] for e in entries)
    total_size_mb = round(sum(e["size_kb"] for e in entries) / 1024, 1)

    return {
        "total_files": len(entries),
        "total_symbols": len(symbols),
        "total_bars": total_bars,
        "total_size_mb": total_size_mb,
        "timeframes": {tf: sum(1 for e in entries if e["timeframe"] == tf) for tf in timeframes},
        "symbols": symbols,
    }


def main():
    print("Scanning data/ for parquet files...")
    entries = scan_parquets()

    if not entries:
        print("No parquet files found in data/")
        sys.exit(1)

    summary = build_summary(entries)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": "data/",
        "summary": summary,
        "files": entries,
    }

    out_path = DATA_DIR / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nManifest written: {out_path}")
    print(f"  Files: {summary['total_files']}")
    print(f"  Symbols: {summary['total_symbols']}")
    print(f"  Total bars: {summary['total_bars']:,}")
    print(f"  Total size: {summary['total_size_mb']} MB")
    print(f"  Timeframes: {summary['timeframes']}")


if __name__ == "__main__":
    main()
