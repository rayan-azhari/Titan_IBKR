"""GEM Dual Momentum (A1_ewma_hl40) — L52 hybrid sweep.

**EXPLORATORY ONLY** (L52). This sweep does NOT change GEM's deployment
status — GEM is the only confirmed-live strategy (J4 audit 2026-05-15
verdict DEPLOY, all 5 axes best, CI_lo +0.387). The sweep tests whether
the J4-chosen canonical `(vol_estimator_halflife=40, ann_vol_target=0.10)`
sits on a FLAT PLATEAU in the (halflife × vol_target) plane, or whether
J4's 1D-sweep selection happened to land on a 2D knife-edge that a
2D plateau-detection would have re-centred.

Two possible outcomes:

1. **Plateau confirms J4 canonical** — `(40, 0.10)` is at the centre of
   a flat region. Result: J4 selection validated; no change to live config.
2. **Plateau identifies a different centre** — e.g., `(60, 0.10)` or
   `(40, 0.075)`. Result: candidate upgrade for the next GEM re-audit.
   Would trigger a fresh pre-reg + V3.6 audit, NOT immediate migration.

Sweep axes:
    `vol_estimator_halflife ∈ {10, 20, 40, 60, 100, 160}` — log-spaced;
        captures the J4 7-cell range and extends it.
    `ann_vol_target ∈ {0.05, 0.075, 0.10, 0.125, 0.15}` — 5 values
        around the live 0.10; tests sizing sensitivity.

30 cells. ~5 minutes total on the 23y SPY/EFA/IEF universe.

Sanctuary: last 12 months (matches GEM J4 audit convention; smaller than
the bond_gold/etf_trend 24m because GEM was already J4-audited on a 12m
sanctuary).

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_gem_hybrid.py
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
from research.gem.gem_strategy import GemConfig, gem_returns  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_gem_hybrid"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12

# Live A1_ewma_hl40 canonical (mirror config/gem_voltarget_lev2.toml minus the
# swept axes).
LIVE_CFG_FROZEN = {
    "lookback_blend": (3, 6, 12),
    "absolute_gate_lookback_months": 12,
    "buffer_pct": 0.005,
    "defensive_switch": True,
    "vol_lookback_days": 20,
    "max_leverage": 2.0,
    "vol_estimator_kind": "ewma",
    "stress_gate_enabled": False,
    "dd_breaker_enabled": False,
}

# Cost model from `config/gem_voltarget_lev2.toml` (etf mode):
COST_BPS = 6.0
COST_FIXED_USD = 1.0
NOTIONAL = 30_000.0


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    spy = _load_close("SPY")
    efa = _load_close("EFA")
    ief = _load_close("IEF")
    common = spy.index.intersection(efa.index).intersection(ief.index)
    return pd.DataFrame(
        {"SPY": spy.reindex(common), "EFA": efa.reindex(common), "IEF": ief.reindex(common)}
    ).dropna()


def gem_strategy_fn(
    closes_df: pd.DataFrame, *, vol_estimator_halflife: int, ann_vol_target: float
) -> pd.Series:
    """Sweep adapter — builds a GemConfig with frozen live params + swept axes."""
    cfg = GemConfig(
        lookback_blend=LIVE_CFG_FROZEN["lookback_blend"],
        absolute_gate_lookback_months=LIVE_CFG_FROZEN["absolute_gate_lookback_months"],
        buffer_pct=LIVE_CFG_FROZEN["buffer_pct"],
        defensive_switch=LIVE_CFG_FROZEN["defensive_switch"],
        ann_vol_target=ann_vol_target,
        vol_lookback_days=LIVE_CFG_FROZEN["vol_lookback_days"],
        max_leverage=LIVE_CFG_FROZEN["max_leverage"],
        vol_estimator_kind=LIVE_CFG_FROZEN["vol_estimator_kind"],
        vol_estimator_halflife=vol_estimator_halflife,
        stress_gate_enabled=LIVE_CFG_FROZEN["stress_gate_enabled"],
        dd_breaker_enabled=LIVE_CFG_FROZEN["dd_breaker_enabled"],
    )
    return gem_returns(
        closes_df,
        cfg=cfg,
        cost_bps_per_turnover=COST_BPS,
        cost_fixed_usd_per_fill=COST_FIXED_USD,
        notional_usd=NOTIONAL,
        execution_mode="etf",
        rebalance_threshold=0.05,
    ).rename("ret")


def main() -> None:
    closes = load_universe()
    print(
        f"[gem-sweep] universe: SPY+EFA+IEF, common range "
        f"{closes.index[0].date()} -> {closes.index[-1].date()}, {closes.shape[0]} bars"
    )

    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(
        f"[gem-sweep] IS slice: {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(
        f"[gem-sweep] sanctuary (held out): {sanctuary_closes.shape[0]} bars "
        f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})"
    )

    grid = {
        "vol_estimator_halflife": [10, 20, 40, 60, 100, 160],
        "ann_vol_target": [0.05, 0.075, 0.10, 0.125, 0.15],
    }
    n_cells = len(grid["vol_estimator_halflife"]) * len(grid["ann_vol_target"])
    print(f"[gem-sweep] running sweep over {n_cells} cells (EWMA halflife x ann vol_target)...")

    res = run_parameter_sweep(
        is_closes,
        strategy_fn=gem_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=380 + 252,  # 18mo warmup + 1y to evaluate
        meta={
            "strategy": "GEM Dual Momentum (A1_ewma_hl40 lineage, J4 noise-robust redesign)",
            "universe": ["SPY", "EFA", "IEF"],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "live_canonical": {"vol_estimator_halflife": 40, "ann_vol_target": 0.10},
            "context": "L52 hybrid sweep on confirmed-live strategy (GEM J4 DEPLOY verdict 2026-05-15)",
        },
    )

    print("\n[gem-sweep] Sharpe surface (rows = halflife, cols = vol_target):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[gem-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[gem-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[gem-sweep] NO plateau candidates passed the spread + positivity gate.")

    # Live canonical retrospective.
    target = next(
        (
            i
            for i, cell in enumerate(res.cells)
            if cell == {"vol_estimator_halflife": 40, "ann_vol_target": 0.10}
        ),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(
            f"\n[gem-sweep] LIVE A1_ewma_hl40 canonical "
            f"(halflife=40, vol_target=0.10): IS Sharpe = {canonical_sr:.3f}"
        )
        finite_mask = np.isfinite(res.sharpes)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
            best_sr = res.sharpes[best_idx]
            best_cell = res.cells[best_idx]
            print(f"[gem-sweep] best cell in grid: {best_cell} -> IS Sharpe = {best_sr:.3f}")
            gap = (
                (best_sr - canonical_sr) / abs(canonical_sr) * 100
                if abs(canonical_sr) > 1e-6
                else 0
            )
            print(f"[gem-sweep] best-vs-canonical gap: {gap:+.1f}%")
        # Check if canonical sits on a plateau.
        on_plateau = any(
            c.center == {"vol_estimator_halflife": 40, "ann_vol_target": 0.10} for c in candidates
        )
        print(f"[gem-sweep] LIVE canonical on plateau? {'YES' if on_plateau else 'no'}")

    report = format_plateau_report(
        res, candidates, audit_label="GEM J4 L52 HYBRID SWEEP (confirmed-live re-test)"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print(f"\n[gem-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")
    print(f"[gem-sweep] wrote: {(REPORTS_DIR / 'sharpe_surface.csv').relative_to(PROJECT_ROOT)}")
    print(f"[gem-sweep] wrote: {(REPORTS_DIR / 'cells_long.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
