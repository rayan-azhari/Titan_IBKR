"""mr_audjpy — Pardo-style sweep on H1 AUD/JPY data (Wave A.3).

**EXPLORATORY ONLY** (L52). The signal-layer question — does VWAP-deviation
MR on H1 AUD/JPY have a meaningful edge under V3.6 cost + math? — is the
gate for whether the live strategy's tier-grid + regime-filter machinery
is worth full V3.6 auditing.

Sweep axes:
    `vwap_anchor ∈ {12, 18, 24, 36, 48, 72}` — H1 bars; 24=1d, 48=2d
    `entry_pct ∈ {0.90, 0.95, 0.98, 0.99}` — entry-band aggressiveness

24 cells. Periods/year = 252*24 = 6048 (H1 bars).

Sanctuary: last 12 months held out.

If a clean plateau emerges with CI_lo plausibly > 0 → build the full
audit harness as Wave A.3 continuation.
If no plateau / negative Sharpe everywhere → document the signal-layer
failure; the live strategy's claim is suspect even before the tier-grid
question is asked.
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_mr_audjpy"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12


def load_h1_bars() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "AUD_JPY_H1.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    # Keep only what the strategy uses.
    return df[["close", "volume"]].dropna(subset=["close"])


def mr_audjpy_strategy_fn(
    bars_df: pd.DataFrame, *, vwap_anchor: int, entry_pct: float
) -> pd.Series:
    cfg = MrAudjpyConfig(vwap_anchor=vwap_anchor, entry_pct=entry_pct)
    return mr_audjpy_returns(bars_df, cfg=cfg)


def main() -> None:
    bars = load_h1_bars()
    print(f"[mr-sweep] AUD/JPY H1: {bars.shape[0]} bars ({bars.index[0]} -> {bars.index[-1]})")

    cutoff = bars.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_bars = bars.loc[:cutoff]
    sanc_bars = bars.loc[cutoff:]
    print(
        f"[mr-sweep] IS slice: {is_bars.shape[0]} bars ({is_bars.index[0]} -> {is_bars.index[-1]})"
    )
    print(f"[mr-sweep] sanctuary held out: {sanc_bars.shape[0]} bars")

    grid = {
        "vwap_anchor": [12, 18, 24, 36, 48, 72],
        "entry_pct": [0.90, 0.95, 0.98, 0.99],
    }
    n_cells = len(grid["vwap_anchor"]) * len(grid["entry_pct"])
    print(f"[mr-sweep] running sweep over {n_cells} cells (vwap_anchor x entry_pct)...")

    res = run_parameter_sweep(
        is_bars,
        strategy_fn=mr_audjpy_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["H1"],
        min_is_bars=24 * 252 * 2,  # need 2y of H1 minimum
        meta={
            "strategy": "mr_audjpy (simplified VWAP-MR — single-tier, no regime, no session)",
            "universe": ["AUD/JPY (H1)"],
            "sanctuary_months": SANCTUARY_MONTHS,
            "live_canonical_proxy": {"vwap_anchor": 24, "entry_pct": 0.95},
            "context": "V1-era live strategy re-audit Wave A.3 (signal-layer test)",
        },
    )

    print("\n[mr-sweep] Sharpe surface (rows = vwap_anchor, cols = entry_pct):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[mr-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[mr-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[mr-sweep] NO plateau candidates passed.")

    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"vwap_anchor": 24, "entry_pct": 0.95}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(
            f"\n[mr-sweep] LIVE proxy (vwap_anchor=24, entry_pct=0.95): IS Sharpe = {canonical_sr:.3f}"
        )

    finite_mask = np.isfinite(res.sharpes)
    if finite_mask.any():
        best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
        print(
            f"[mr-sweep] best cell: {res.cells[best_idx]} -> "
            f"IS Sharpe = {res.sharpes[best_idx]:.3f}"
        )

    # Note on V1 claims: live config (vwap_anchor=24) reported Sharpe +0.53
    # by the April-2026 corrected harness. Pre-fix V1 claimed Sharpe +4.64
    # (sqrt(252)-on-H1 bug). V3.6 expectation: <= +0.53 with positive CI_lo.
    print("\n[mr-sweep] V1 (corrected) claimed Sharpe +0.53 at vwap_anchor=24.")
    print("[mr-sweep] V3.6 deployment-relevant question: is THIS sweep's best")
    print("[mr-sweep] cell + CI_lo > 0 with the L17 rel-MC test pending?")

    report = format_plateau_report(
        res, candidates, audit_label="mr_audjpy V1-era RE-AUDIT SWEEP (Wave A.3)"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print(f"\n[mr-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
