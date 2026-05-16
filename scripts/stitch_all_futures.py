"""stitch_all_futures.py -- Stitch M1 and M2 for every commodity root
under ``data/ibkr_futures/``.

For each root sub-directory: run ``stitch_root`` and write
``data/{ROOT}_M1_stitched_D.parquet`` + ``data/{ROOT}_M2_stitched_D.parquet``.

Skips roots with empty / single-contract directories.

Usage::

    uv run python scripts/stitch_all_futures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.futures_stitching import stitch_root  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "ibkr_futures"
OUT_DIR = PROJECT_ROOT / "data"


def main() -> int:
    if not DATA_DIR.exists():
        print(f"No directory {DATA_DIR}")
        return 1
    roots = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    if not roots:
        print(f"No commodity directories under {DATA_DIR}")
        return 2

    print("=" * 70)
    print(f"  Stitching {len(roots)} commodity roots from {DATA_DIR}")
    print("=" * 70)

    n_m1, n_m2 = 0, 0
    for root in roots:
        try:
            result = stitch_root(
                root, data_dir=DATA_DIR, out_dir=OUT_DIR, roll_buffer_days=5, save=True
            )
        except Exception as exc:
            print(f"  {root}: ERROR {exc}")
            continue
        m1, m2 = result["M1"], result["M2"]
        m1_n = len(m1) if not m1.empty else 0
        m2_n = len(m2) if not m2.empty else 0
        m1_range = f"{m1.index[0].date()} -> {m1.index[-1].date()}" if m1_n else "empty"
        m2_range = f"{m2.index[0].date()} -> {m2.index[-1].date()}" if m2_n else "empty"
        print(f"  {root}: M1={m1_n} ({m1_range})   M2={m2_n} ({m2_range})")
        if m1_n:
            n_m1 += 1
        if m2_n:
            n_m2 += 1

    print("=" * 70)
    print(f"  Stitched M1 series: {n_m1} / {len(roots)}")
    print(f"  Stitched M2 series: {n_m2} / {len(roots)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
