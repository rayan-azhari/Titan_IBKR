"""B4b TSMOM — retrospective Pardo-style sweep on IS-only data.

**EXPLORATORY ONLY** (L52). Outputs from this script are a PRIOR for a
re-pre-reg directive, NOT evidence of deployment eligibility.

Why this sweep:
    B4b was retired (L43 family) on plateau-pre-flight failure — its
    canonical (window=12 months, skip=1) sat on a knife-edge in IS. The
    recommended retroactive sweep tests whether the parameter space has
    a *flat* high-Sharpe plateau that the textbook canonical missed.

    If a plateau is found, the right next move is a fresh pre-reg
    directive with the plateau centre as canonical, then a NEW V3.6
    audit on the held-out sanctuary window (which the original B4b audit
    consumed). If no plateau exists, the retirement stands.

Sanctuary discipline (V3.6 + Pardo hybrid, L52):
    * Last 12 months of data held out (sanctuary). The sweep does NOT
      see them.
    * IS = everything before the sanctuary. ~24 months on the IBKR
      stitched data (depth bounded by L41).

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_b4b_tsmom.py
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
from research.tsmom.run_b4b_audit import UNIVERSE_ROOTS, load_universe  # noqa: E402
from research.tsmom.tsmom_strategy import TsmomConfig, tsmom_returns  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_b4b_tsmom"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# L52 sanctuary discipline: hold out the last 12 months. The original B4b
# audit consumed everything (sanctuary was applied via SanctuarySlice but
# is not the gate here — we're re-using B4b's exact data, so we MUST be
# more conservative: hold out at least 12 months that will become the gate
# for the new pre-reg's audit. After the pre-reg, run a fresh audit on
# the sanctuary; that audit is the deployment gate, not this sweep.
SANCTUARY_MONTHS = 12


def tsmom_strategy_fn(
    closes_df: pd.DataFrame, *, momentum_window_months: int, skip_months: int
) -> pd.Series:
    """Thin adapter from the sweep harness into ``tsmom_returns``.

    Uses ``signal_mode='sign'`` and other knobs at their pre-reg defaults —
    we're sweeping the lookback parameters, holding everything else equal.
    Costs ARE applied (the deployment gate is net Sharpe; sweeping gross
    would mis-rank cells that trade more often).
    """
    cfg = TsmomConfig(
        signal_mode="sign",
        momentum_window_months=momentum_window_months,
        skip_months=skip_months,
        weighting="inv_vol",
        target_vol_annual=0.10,
        vol_lookback_days=60,
        rebalance="monthly",
        apply_costs=True,
        cost_bps_per_turnover=1.0,
        cost_fixed_usd_per_fill=1.0,
        notional_usd_per_leg=30_000.0,
    )
    return tsmom_returns(closes_df, cfg=cfg).rename("ret")


def main() -> None:
    closes = load_universe()
    print(f"[b4b-sweep] universe: {closes.shape[1]} instruments x {closes.shape[0]} bars")
    print(f"[b4b-sweep] data range: {closes.index[0].date()} -> {closes.index[-1].date()}")

    # Sanctuary cutoff — hold out the last SANCTUARY_MONTHS.
    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(
        f"[b4b-sweep] IS slice: {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(
        f"[b4b-sweep] sanctuary (held out): {sanctuary_closes.shape[0]} bars "
        f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})"
    )

    # IS bar count check: TSMOM with momentum_window=24 + vol_lookback=60d
    # needs at minimum ~24mo + 60d of bars to produce a single signal.
    # With ~24mo IS the window=24 cell will be tight; document the issue.
    if is_closes.shape[0] < 252:
        raise SystemExit(
            f"IS slice has only {is_closes.shape[0]} bars; sweep needs >= 252. "
            f"Reduce SANCTUARY_MONTHS or wait for more data."
        )

    # Param grid. Single-window B4 form (not the B4c ensemble — that's a
    # separate hypothesis class). The (window, skip) plane is the canonical
    # MOP 2012 space.
    grid = {
        "momentum_window_months": [3, 6, 9, 12, 15, 18],  # drop 24 — needs > IS bars
        "skip_months": [0, 1, 2, 3],
    }

    print(
        f"[b4b-sweep] running sweep over {len(grid['momentum_window_months']) * len(grid['skip_months'])} cells..."
    )
    res = run_parameter_sweep(
        is_closes,
        strategy_fn=tsmom_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=252,
        meta={
            "strategy": "B4b TSMOM (single-window sign, monthly rebal, inv-vol, net of costs)",
            "universe": list(UNIVERSE_ROOTS),
            "n_instruments": closes.shape[1],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "demo": False,
            "context": "retroactive sweep on retired strategy (L52 hybrid workflow)",
        },
    )

    print("\n[b4b-sweep] Sharpe surface (rows = window_months, cols = skip_months):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[b4b-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[b4b-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% "
                f"n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[b4b-sweep] NO plateau candidates passed the spread + positivity gate.")
        print("[b4b-sweep] Interpretation: the (window, skip) parameter space has no flat")
        print("[b4b-sweep] high-Sharpe region on B4b's IS data — the L43 retirement stands.")

    # B4 canonical retrospective: (window=12, skip=1).
    target = next(
        (
            i
            for i, cell in enumerate(res.cells)
            if cell == {"momentum_window_months": 12, "skip_months": 1}
        ),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(f"\n[b4b-sweep] B4 canonical (window=12, skip=1): IS Sharpe = {canonical_sr:.3f}")
        # Find the best cell in the whole grid for comparison.
        finite_mask = np.isfinite(res.sharpes)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
            best_sr = res.sharpes[best_idx]
            best_cell = res.cells[best_idx]
            print(f"[b4b-sweep] best cell in grid: {best_cell} -> IS Sharpe = {best_sr:.3f}")
            if abs(canonical_sr) > 1e-6:
                gap_pct = (best_sr - canonical_sr) / abs(canonical_sr) * 100
                print(f"[b4b-sweep] best-vs-canonical gap: {gap_pct:+.1f}%")

    # Write artefacts.
    report = format_plateau_report(res, candidates, audit_label="B4b TSMOM RETROACTIVE SWEEP")
    report_fp = REPORTS_DIR / "plateau_report.md"
    report_fp.write_text(report, encoding="utf-8")

    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print(f"\n[b4b-sweep] wrote: {report_fp.relative_to(PROJECT_ROOT)}")
    print(f"[b4b-sweep] wrote: {(REPORTS_DIR / 'sharpe_surface.csv').relative_to(PROJECT_ROOT)}")
    print(f"[b4b-sweep] wrote: {(REPORTS_DIR / 'cells_long.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
