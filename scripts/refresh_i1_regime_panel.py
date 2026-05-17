"""refresh_i1_regime_panel.py -- daily refresh of the I1 regime panel.

Downloads the 6 underlying daily ETFs/indices needed for the panel
features (VIX, SPY, HYG, IEF, TLT, DXY) via yfinance, then re-runs
the panel construction. Designed to be called daily (cron or
scheduler) so the live ewmac_regime strategy reads a current panel.

The panel feature set is documented in
`research/exploration/build_i1_regime_panel.py`; this script just
keeps its inputs fresh.

Run::

    PYTHONIOENCODING=utf-8 uv run python scripts/refresh_i1_regime_panel.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# yfinance symbol mapping. Keys are local parquet filename prefixes;
# values are the yfinance ticker. VIX comes from ^VIX, DXY from DX-Y.NYB.
SYMBOLS = {
    "VIX": "^VIX",
    "SPY": "SPY",
    "HYG": "HYG",
    "IEF": "IEF",
    "TLT": "TLT",
    "DXY": "DX-Y.NYB",
}


def refresh_underlyings() -> int:
    """Download each underlying via scripts/download_data_yfinance.py."""
    pairs = [f"{local}={yf}" for local, yf in SYMBOLS.items()]
    print(f"[refresh] downloading {len(pairs)} underlyings via yfinance: {pairs}")
    cmd = [
        "uv", "run", "python", "scripts/download_data_yfinance.py",
        "--symbols", *pairs,
        "--interval", "D",
        "--start", "2010-01-01",
    ]
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=300)
    except subprocess.TimeoutExpired:
        print("[refresh] yfinance download TIMED OUT after 300s", file=sys.stderr)
        return 124
    return result.returncode


def rebuild_panel() -> int:
    """Re-run the panel construction script."""
    print("[refresh] rebuilding regime panel...")
    cmd = [
        "uv", "run", "python", "research/exploration/build_i1_regime_panel.py",
    ]
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=120)
    except subprocess.TimeoutExpired:
        print("[refresh] panel build TIMED OUT after 120s", file=sys.stderr)
        return 124
    return result.returncode


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    print("=" * 72)
    print(f"  I1 regime panel daily refresh @ {ts}")
    print("=" * 72)

    rc1 = refresh_underlyings()
    if rc1 != 0:
        print(f"[refresh] underlyings step FAILED (rc={rc1}); skipping rebuild", file=sys.stderr)
        return rc1

    rc2 = rebuild_panel()
    if rc2 != 0:
        print(f"[refresh] panel rebuild FAILED (rc={rc2})", file=sys.stderr)
        return rc2

    out = PROJECT_ROOT / "data" / "i1_regime_panel.parquet"
    if out.exists():
        st = out.stat()
        print(f"[refresh] OK -- {out} ({st.st_size} bytes, mtime={datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
