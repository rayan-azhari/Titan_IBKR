"""Data loading + alignment for the Samir-Stack pipeline.

All series are loaded as date-indexed (tz-naive, normalised to midnight)
``pd.Series`` of close prices. This avoids the cross-source timestamp
mismatch that exists in the data parquets (SPY is at 05:00 UTC, VIX is at
06:00 UTC, etc.) — for daily strategies we only care about the date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def _load_close(filename: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / filename)
    s = df["close"].copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index().rename(filename.replace(".parquet", ""))


def load_panel(
    *,
    start: str | None = "2003-04-01",
    end: str | None = "2026-04-02",
) -> dict[str, pd.Series]:
    """Load the full set of series the strategy needs, aligned by date.

    Returns a dict with keys: ``spy``, ``vix``, ``hyg``, ``ief``, ``tlt``,
    optionally ``gld``, ``dxy`` if available. All series are clipped to
    ``[start, end]``. Each Series is independent — alignment to a common
    index happens downstream in the indicator panel.
    """
    out: dict[str, pd.Series] = {}
    out["spy"] = _load_close("SPY_D.parquet")
    out["vix"] = _load_close("^VIX_D.parquet")
    out["hyg"] = _load_close("HYG_D.parquet")
    out["ief"] = _load_close("IEF_D.parquet")
    out["tlt"] = _load_close("TLT_D.parquet")

    # Optional series — loaded only if present
    for key, fname in [("gld", "GLD_D.parquet"), ("dxy", "DXY_D.parquet")]:
        try:
            out[key] = _load_close(fname)
        except FileNotFoundError:
            pass

    # Clip date range
    if start is not None or end is not None:
        for k, s in out.items():
            mask = pd.Series(True, index=s.index)
            if start is not None:
                mask &= s.index >= pd.Timestamp(start)
            if end is not None:
                mask &= s.index <= pd.Timestamp(end)
            out[k] = s.loc[mask]

    return out
