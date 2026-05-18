"""mr_fx — L58 signal-layer sweep on M5 EUR/USD VWAP-MR (Wave A.6).

Re-uses the mr_audjpy signal-layer module (rolling-VWAP MR, single-tier,
no regime filter, no session window). The L58 question:

  > **Does VWAP-deviation MR on M5 EUR/USD produce a meaningful signal-
    layer Sharpe edge under V3.6 cost + math?**

Different from mr_audjpy by:
- Bar timeframe: M5 (5-min) instead of H1
- Instrument: EUR/USD instead of AUD/JPY
- Periods/year: 72,576 (252×24×12) instead of 6,048

V1 live config uses session-anchored VWAP (london/ny) + 4-tier grid;
this signal-layer sweep uses ROLLING VWAP (the cleaner abstraction) and
single-tier. If the rolling-VWAP signal layer is negative, the
session-anchored filter machinery is unlikely to rescue it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.exploration.parameter_sweep import (  # noqa: E402
    detect_plateau,
    format_plateau_report,
    run_parameter_sweep,
)
from research.mean_reversion.mr_audjpy_strategy import (  # noqa: E402
    MrAudjpyConfig,
    mr_audjpy_returns,
)
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_mr_fx"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12


def load_m5_bars() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "EUR_USD_M5.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    return df[["close", "volume"]].dropna(subset=["close"])


def mr_fx_strategy_fn(bars_df: pd.DataFrame, *, vwap_anchor: int, entry_pct: float) -> pd.Series:
    """Sweep adapter — re-uses mr_audjpy module, just different bar tf + cost."""
    cfg = MrAudjpyConfig(
        vwap_anchor=vwap_anchor,
        entry_pct=entry_pct,
        cost_bps_per_turnover=1.5,  # EUR/USD spread + slip
    )
    return mr_audjpy_returns(bars_df, cfg=cfg)


def main() -> None:
    bars = load_m5_bars()
    print(f"[mrfx-sweep] EUR/USD M5: {bars.shape[0]} bars ({bars.index[0]} -> {bars.index[-1]})")

    cutoff = bars.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_bars = bars.loc[:cutoff]
    sanc_bars = bars.loc[cutoff:]
    print(
        f"[mrfx-sweep] IS slice: {is_bars.shape[0]} bars "
        f"({is_bars.index[0]} -> {is_bars.index[-1]})"
    )
    print(f"[mrfx-sweep] sanctuary held out: {sanc_bars.shape[0]} bars")

    # M5 grid: 12 M5 bars = 1 hour; 144 = 1 trading day; 288 = 1 24h day
    grid = {
        "vwap_anchor": [48, 144, 288, 576, 1440],  # 4h, 1d, 2d, 4d, 10d worth of M5
        "entry_pct": [0.90, 0.95, 0.98, 0.99],
    }
    n_cells = len(grid["vwap_anchor"]) * len(grid["entry_pct"])
    print(f"[mrfx-sweep] running sweep over {n_cells} cells (vwap_anchor x entry_pct)...")

    res = run_parameter_sweep(
        is_bars,
        strategy_fn=mr_fx_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["M5"],
        min_is_bars=288 * 252 * 1,  # 1y of M5
        meta={
            "strategy": "mr_fx (signal-layer M5 EUR/USD VWAP-MR — single-tier)",
            "universe": ["EUR/USD M5"],
            "sanctuary_months": SANCTUARY_MONTHS,
            "live_canonical_proxy": {"vwap_anchor": 144, "entry_pct": 0.95},
            "context": "V1-era Wave A.6 — L58 signal-layer-first audit",
        },
    )

    print("\n[mrfx-sweep] Sharpe surface (rows = vwap_anchor, cols = entry_pct):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[mrfx-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[mrfx-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[mrfx-sweep] NO plateau candidates passed.")

    # Live proxy is 1-day-equivalent VWAP (M5 anchor=144 = 12h, closest to live).
    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"vwap_anchor": 144, "entry_pct": 0.95}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        print(
            f"\n[mrfx-sweep] LIVE proxy (vwap_anchor=144 [≈1 trading day], "
            f"entry_pct=0.95): IS Sharpe = {res.sharpes[target]:.3f}"
        )

    finite_mask = np.isfinite(res.sharpes)
    if finite_mask.any():
        best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
        print(
            f"[mrfx-sweep] best cell: {res.cells[best_idx]} -> "
            f"IS Sharpe = {res.sharpes[best_idx]:.3f}"
        )

    print("\n[mrfx-sweep] Pattern: per L58 (mr_audjpy + mtf), signal-layer-first")
    print("[mrfx-sweep] sweep on VWAP-MR strategies typically reveals NEGATIVE")
    print("[mrfx-sweep] signal-layer Sharpe; live edge (if any) is filter-derived.")

    report = format_plateau_report(
        res, candidates, audit_label="mr_fx V1-era RE-AUDIT SWEEP (Wave A.6)"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )
    print(f"\n[mrfx-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
