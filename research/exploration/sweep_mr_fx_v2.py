"""mr_fx — Wave A.6 VERIFICATION sweep with corrected mechanics.

V2 replacement for `sweep_mr_fx.py`. Tests whether the Wave A.6
"SIGNAL-LAYER FAIL" verdict holds when:
    1. Cost reduced from 1.5 bps/turnover → 0.5 bps (liquid FX-hours realistic)
    2. Session-anchored VWAP (London 07:00 + NY 13:00 UTC) instead of rolling
    3. 4-tier grid entries [1, 2, 4, 8] at percentile bands [0.90, 0.95, 0.98, 0.99]
       instead of single-tier

Sweep axes:
    `cost_bps ∈ {0.25, 0.5, 1.0}` — realistic / nominal / conservative
    `reversion_target ∈ {0.30, 0.50, 0.70}` — exit aggressiveness

9 cells. Each cell uses live-config tier setup + session anchors.
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
from research.mean_reversion.mr_fx_session_strategy import (  # noqa: E402
    MrFxSessionConfig,
    mr_fx_session_assert_causal,
    mr_fx_session_returns,
)
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_mr_fx_v2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12


def load_m5_bars() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "EUR_USD_M5.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    return df[["close"]].dropna(subset=["close"])


def mr_fx_v2_strategy_fn(
    bars_df: pd.DataFrame, *, cost_bps: float, reversion_target: float
) -> pd.Series:
    cfg = MrFxSessionConfig(
        cost_bps_per_turnover=cost_bps,
        reversion_target_pct=reversion_target,
    )
    return mr_fx_session_returns(bars_df, cfg=cfg)


def main() -> None:
    bars = load_m5_bars()
    print(f"[mrfx-v2] EUR/USD M5: {bars.shape[0]} bars "
          f"({bars.index[0]} -> {bars.index[-1]})")

    # Causality smoke FIRST (matches the session-anchored mechanic).
    print("\n[mrfx-v2] L04 causality smoke...")
    try:
        mr_fx_session_assert_causal(bars)
        print("[mrfx-v2] Causality PASS.")
    except AssertionError as e:
        print(f"[mrfx-v2] Causality FAIL: {e}")
        return

    cutoff = bars.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_bars = bars.loc[:cutoff]
    print(f"[mrfx-v2] IS slice: {is_bars.shape[0]} bars "
          f"({is_bars.index[0]} -> {is_bars.index[-1]})")

    grid = {
        "cost_bps": [0.25, 0.5, 1.0],
        "reversion_target": [0.30, 0.50, 0.70],
    }
    n_cells = len(grid["cost_bps"]) * len(grid["reversion_target"])
    print(f"\n[mrfx-v2] running sweep over {n_cells} cells (cost_bps x reversion_target)...")

    res = run_parameter_sweep(
        is_bars,
        strategy_fn=mr_fx_v2_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["M5"],
        min_is_bars=288 * 252 * 1,
        meta={
            "strategy": "mr_fx V2 (session VWAP + 4-tier grid + corrected cost)",
            "universe": ["EUR/USD M5"],
            "sanctuary_months": SANCTUARY_MONTHS,
            "context": "Wave A.6 VERIFICATION — corrected vs original sweep_mr_fx.py",
        },
    )

    print("\n[mrfx-v2] Sharpe surface (rows = cost_bps, cols = reversion_target):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[mrfx-v2] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[mrfx-v2] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[mrfx-v2] NO plateau candidates passed.")

    # Live proxy: cost_bps=0.5, reversion_target=0.50
    target = next(
        (i for i, cell in enumerate(res.cells)
         if cell == {"cost_bps": 0.5, "reversion_target": 0.50}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        print(f"\n[mrfx-v2] LIVE-realistic (cost=0.5 bps, reversion=0.50): "
              f"IS Sharpe = {res.sharpes[target]:.3f}")

    finite_mask = np.isfinite(res.sharpes)
    if finite_mask.any():
        best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
        print(f"[mrfx-v2] best cell: {res.cells[best_idx]} -> "
              f"IS Sharpe = {res.sharpes[best_idx]:.3f}")

    print("\n[mrfx-v2] Original Wave A.6 (rolling VWAP, single-tier, 1.5 bps):")
    print("[mrfx-v2]   live proxy = -3.89, every cell -2.0 to -8.4 Sharpe.")
    print("[mrfx-v2] Verification question: does the RETIRE verdict hold")
    print("[mrfx-v2] under realistic cost + actual machinery?")

    report = format_plateau_report(res, candidates, audit_label="mr_fx Wave A.6 VERIFICATION SWEEP")
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )
    print(f"\n[mrfx-v2] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
