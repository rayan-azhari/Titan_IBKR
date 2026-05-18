"""mtf — L21 causality smoke + L52 hybrid sweep (Wave A.5).

Pipeline:
    1. Load EUR/USD bars at H1/H4/D/W timeframes.
    2. **L21 causality smoke** — corrupt future bars; assert past returns
       bit-exact unchanged. Per the V1-era roster's "causality FIRST"
       rule.
    3. If smoke passes → L52 sweep over the confluence threshold +
       relative weight axis. The V1 claim is OOS Sharpe +1.94; the
       sweep should locate that region (or refute it).

Sanctuary: last 12 months held out (matches the L52 standard for daily-
class strategies).
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
from research.mean_reversion.mtf_strategy import (  # noqa: E402
    MtfConfig,
    mtf_assert_causal,
    mtf_returns,
)
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_mtf"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12


def _load_bars(timeframe_filename: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / timeframe_filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    return df[["close"]].dropna()


def load_universe() -> dict[str, pd.DataFrame]:
    return {
        "H1": _load_bars("EUR_USD_H1.parquet"),
        "H4": _load_bars("EUR_USD_H4.parquet"),
        "D": _load_bars("EUR_USD_D.parquet"),
        "W": _load_bars("EUR_USD_W.parquet"),
    }


def mtf_strategy_fn(bars_h1: pd.DataFrame, *, threshold: float, d_weight: float) -> pd.Series:
    """Sweep adapter — H1 is the primary frame; H4/D/W are loaded globally."""
    # Build the bars_by_tf dict using a closure-captured _BARS_BY_TF (set in main).
    # We slice each TF to be ≤ the last H1 timestamp in `bars_h1` so the sweep
    # framework can pass in IS-only H1 bars and have the multi-TF view sliced
    # consistently.
    h1_last_ts = bars_h1.index[-1]
    bars_by_tf = {
        "H1": bars_h1[["close"]],
        "H4": _BARS_BY_TF["H4"].loc[:h1_last_ts],
        "D": _BARS_BY_TF["D"].loc[:h1_last_ts],
        "W": _BARS_BY_TF["W"].loc[:h1_last_ts],
    }
    # Construct cfg with the swept axes. Live default weights are
    # H1:0.10, H4:0.25, D:0.60, W:0.05; we sweep the D weight here as
    # the dominant timeframe-weight knob. Other weights re-normalize so
    # the total is 1.0.
    other_total = 1.0 - d_weight  # split among H1:H4:W in ratio 0.10:0.25:0.05 = 0.40
    h1_w = (0.10 / 0.40) * other_total
    h4_w = (0.25 / 0.40) * other_total
    w_w = (0.05 / 0.40) * other_total
    cfg = MtfConfig(
        weights={"H1": h1_w, "H4": h4_w, "D": d_weight, "W": w_w},
        confirmation_threshold=threshold,
    )
    return mtf_returns(bars_by_tf, cfg=cfg)


# Module-level cache so the sweep adapter can access higher-TF bars
# without round-tripping them through the parameter grid.
_BARS_BY_TF: dict[str, pd.DataFrame] = {}


def main() -> None:
    global _BARS_BY_TF
    _BARS_BY_TF = load_universe()
    print(
        f"[mtf-sweep] EUR/USD H1: {_BARS_BY_TF['H1'].shape[0]} bars "
        f"({_BARS_BY_TF['H1'].index[0]} -> {_BARS_BY_TF['H1'].index[-1]})"
    )
    print(f"[mtf-sweep] H4: {_BARS_BY_TF['H4'].shape[0]} bars")
    print(f"[mtf-sweep] D : {_BARS_BY_TF['D'].shape[0]} bars")
    print(f"[mtf-sweep] W : {_BARS_BY_TF['W'].shape[0]} bars")

    # ── L21 causality smoke FIRST ─────────────────────────────────────────
    print(
        "\n[mtf-sweep] L21 causality smoke test (corrupt future bars; "
        "assert past returns unchanged)..."
    )
    try:
        mtf_assert_causal(_BARS_BY_TF)
        print("[mtf-sweep] L21 PASS — multi-TF alignment is causally correct.")
    except AssertionError as e:
        print(f"[mtf-sweep] L21 *** FAIL *** : {e}")
        print("[mtf-sweep] Aborting sweep — fix L21 bug before proceeding.")
        return

    # ── Sanctuary slice ───────────────────────────────────────────────────
    h1_bars = _BARS_BY_TF["H1"]
    cutoff = h1_bars.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_h1 = h1_bars.loc[:cutoff]
    print(
        f"\n[mtf-sweep] IS H1 slice: {is_h1.shape[0]} bars ({is_h1.index[0]} -> {is_h1.index[-1]})"
    )
    print(f"[mtf-sweep] sanctuary (held out): {h1_bars.shape[0] - is_h1.shape[0]} H1 bars")

    # ── Sweep grid ────────────────────────────────────────────────────────
    grid = {
        "threshold": [0.05, 0.10, 0.15, 0.20],
        "d_weight": [0.30, 0.45, 0.60, 0.75],
    }
    n_cells = len(grid["threshold"]) * len(grid["d_weight"])
    print(f"\n[mtf-sweep] running sweep over {n_cells} cells (threshold x d_weight)...")

    res = run_parameter_sweep(
        is_h1,
        strategy_fn=mtf_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["H1"],
        min_is_bars=24 * 252 * 2,  # 2y of H1
        meta={
            "strategy": "mtf — multi-TF confluence H1/H4/D/W (V3.6-causal)",
            "universe": ["EUR/USD H1+H4+D+W"],
            "sanctuary_months": SANCTUARY_MONTHS,
            "live_canonical_proxy": {"threshold": 0.10, "d_weight": 0.60},
            "context": "V1-era Wave A.5 — L21 causality-corrected re-audit",
        },
    )

    print("\n[mtf-sweep] Sharpe surface (rows = threshold, cols = D weight):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[mtf-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[mtf-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[mtf-sweep] NO plateau candidates passed.")

    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"threshold": 0.10, "d_weight": 0.60}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        print(
            f"\n[mtf-sweep] LIVE proxy (threshold=0.10, D_weight=0.60): "
            f"IS Sharpe = {res.sharpes[target]:.3f}"
        )

    finite_mask = np.isfinite(res.sharpes)
    if finite_mask.any():
        best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
        print(
            f"[mtf-sweep] best cell: {res.cells[best_idx]} -> "
            f"IS Sharpe = {res.sharpes[best_idx]:.3f}"
        )

    print("\n[mtf-sweep] V1 (Round 4) claimed OOS Combined Sharpe +1.94 on WMA config.")
    print("[mtf-sweep] V3.6 expectation under causality-correct alignment:")
    print("[mtf-sweep]   if live proxy ~ +1.94 -> V1 audit was causally correct;")
    print("[mtf-sweep]   if live proxy << +1.94 -> V1 had the L21 look-ahead bug.")

    report = format_plateau_report(
        res, candidates, audit_label="mtf V1-era RE-AUDIT SWEEP (Wave A.5)"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )
    print(f"\n[mtf-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
