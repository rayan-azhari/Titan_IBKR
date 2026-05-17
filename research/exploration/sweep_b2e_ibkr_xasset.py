"""B2e — IS-only EWMAC parameter sweep on the 11-symbol IBKR cross-asset universe.

V3.6 hybrid workflow (L52). EXPLORATORY ONLY -- this sweep informs the
canonical-speed choice for the upcoming B2e pre-reg directive. It does
NOT see the held-out sanctuary.

Universe (11 symbols, 4 asset classes):
    equity_index : ES, NQ           (data/<S>_D.parquet, since 2011)
    commodity    : CL, BZ, HG, SI, GC (data/<S>_M1_D.parquet, Databento, since 2017-06)
    bond         : ZN, ZB            (data/<S>_M1_D.parquet, Databento, since 2017-06)
    fx           : 6E, 6J            (data/<S>_M1_D.parquet, Databento, since 2017-06)

Common window: 2017-06-01 -> 2026-05-15 (~9 years).
IS slice:     common-start         -> sanctuary-cutoff (sanctuary = last 12 months).
Sanctuary:    2025-05-16 -> 2026-05-15 (HELD OUT — sweep never touches this).

Sweep:
    Single-speed Carver EWMAC over fast_hl in {2, 4, 8, 16, 32, 64, 128}
    with slow_hl = 4 * fast_hl (Carver canonical ratio). 7-cell 1D sweep.
    Costs ON (net Sharpe is the deployment-relevant metric).

Output:
    .tmp/reports/sweep_b2e_ibkr_xasset/{plateau_report.md, sharpe_surface.csv, cells_long.csv}

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_b2e_ibkr_xasset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ewmac.ewmac_strategy import EwmacConfig, ewmac_returns  # noqa: E402
from research.exploration.parameter_sweep import (  # noqa: E402
    detect_plateau,
    format_plateau_report,
)
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_b2e_ibkr_xasset"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12

# Universe + per-symbol parquet path.
UNIVERSE: dict[str, str] = {
    "ES": "ES_D.parquet",
    "NQ": "NQ_D.parquet",
    "CL": "CL_M1_D.parquet",
    "BZ": "BZ_M1_D.parquet",
    "HG": "HG_M1_D.parquet",
    "SI": "SI_M1_D.parquet",
    "GC": "GC_M1_D.parquet",
    "ZN": "ZN_M1_D.parquet",
    "ZB": "ZB_M1_D.parquet",
    "6E": "6E_M1_D.parquet",
    "6J": "6J_M1_D.parquet",
}


def _load_close(symbol: str, fname: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / fname)
    if "close" not in df.columns:
        raise ValueError(f"{fname} missing 'close' column: {list(df.columns)}")
    s = df["close"].astype(float)
    # Normalise index: drop tz, drop time-of-day.
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s.sort_index()
    # Drop duplicate dates (Databento+IBKR can both emit a same-day bar after merge).
    s = s[~s.index.duplicated(keep="last")]
    s.name = symbol
    return s


def load_universe() -> pd.DataFrame:
    parts = [_load_close(sym, fname) for sym, fname in UNIVERSE.items()]
    df = pd.concat(parts, axis=1).sort_index()
    # Trim to common date range (drop leading rows where any column is NaN).
    first_valid = df.dropna(how="any").index
    if len(first_valid) == 0:
        raise RuntimeError("No common date range across the 11-symbol universe.")
    df = df.loc[first_valid[0]:]
    # Forward-fill within-series gaps from exchange-holiday misalignment (cap 5d).
    df = df.ffill(limit=5)
    return df


def ewmac_strategy_fn(
    closes_df: pd.DataFrame, *, fast_hl: int, slow_hl: int
) -> pd.Series:
    """Single-speed Carver EWMAC, net of costs, monthly rebal -- per L52."""
    cfg = EwmacConfig(
        speeds=((fast_hl, slow_hl),),
        fdm=1.0,  # single-speed: FDM = 1.0 by definition
        forecast_cap=20.0,
        forecast_scalar_mode="carver",
        target_vol_annual=0.10,
        rebalance="monthly",
        apply_costs=True,
        cost_bps_per_turnover=1.0,
        cost_fixed_usd_per_fill=1.0,
        notional_usd_per_leg=30_000.0,
    )
    return ewmac_returns(closes_df, cfg=cfg).rename("ret")


def main() -> None:
    closes = load_universe()
    print(f"[b2e-sweep] universe: {closes.shape[1]} instruments x {closes.shape[0]} bars")
    print(f"[b2e-sweep] data range: {closes.index[0].date()} -> {closes.index[-1].date()}")
    print(f"[b2e-sweep] symbols: {list(closes.columns)}")

    # Sanctuary discipline (L52): hold out last 12 months. Sweep IS only.
    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(
        f"[b2e-sweep] IS:        {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(
        f"[b2e-sweep] sanctuary: {sanctuary_closes.shape[0]} bars "
        f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})  [HELD OUT]"
    )

    # Param grid: 1D over fast_hl with slow_hl = 4 * fast_hl (Carver canonical ratio).
    # 7 cells covering ~0.3d to ~91d effective averaging speed.
    fast_speeds = [2, 4, 8, 16, 32, 64, 128]
    grid_cells = [{"fast_hl": f, "slow_hl": 4 * f} for f in fast_speeds]

    # parameter_sweep iterates itertools.product over a dict's value lists,
    # which doesn't support our slow=4*fast constraint cleanly. We call
    # strategy_fn directly per cell and assemble a SweepResult by hand for
    # the plateau detector.
    import numpy as np

    from research.exploration.parameter_sweep import SweepResult
    from titan.research.metrics import sharpe

    sharpes = []
    vols = []
    n_obs = []
    for cell in grid_cells:
        ret = ewmac_strategy_fn(is_closes, **cell)
        ret = ret.dropna()
        if len(ret) == 0:
            sharpes.append(np.nan)
            vols.append(np.nan)
            n_obs.append(0)
            continue
        sh = float(sharpe(ret, periods_per_year=BARS_PER_YEAR["D"]))
        vol = float(ret.std(ddof=1) * np.sqrt(BARS_PER_YEAR["D"]))
        sharpes.append(sh)
        vols.append(vol)
        n_obs.append(int(len(ret)))

    # Build a 1D SweepResult-shaped object for the plateau detector.
    res = SweepResult(
        param_names=("fast_hl", "slow_hl"),
        param_grid={"fast_hl": fast_speeds, "slow_hl": [4 * f for f in fast_speeds]},
        cells=grid_cells,
        sharpes=np.array(sharpes),
        vols=np.array(vols),
        n_obs=np.array(n_obs),
        meta={
            "strategy": "B2e Carver EWMAC single-speed (slow=4*fast, monthly rebal, net of costs)",
            "universe": list(UNIVERSE.keys()),
            "n_instruments": closes.shape[1],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "context": "V3.6 hybrid workflow L52 pre-pre-reg sweep on new IBKR cross-asset 11-symbol universe",
        },
    )

    print("\n[b2e-sweep] IS Sharpe by single-speed (fast_hl -> slow_hl=4*fast):")
    print(f"  {'fast_hl':>8} {'slow_hl':>8} {'Sharpe':>9} {'vol':>9} {'n_obs':>7}")
    for cell, sh, vol, n in zip(grid_cells, sharpes, vols, n_obs):
        print(
            f"  {cell['fast_hl']:>8} {cell['slow_hl']:>8} "
            f"{sh:>+9.4f} {vol:>9.4f} {n:>7}"
        )

    print("\n[b2e-sweep] plateau detection (spread <= 30%, min_neighbours=2, top_k=5)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[b2e-sweep] {len(candidates)} plateau candidate(s):")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:+.4f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:+.4f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
        # Recommend the 3-speed ensemble centered on the top candidate.
        top = candidates[0]
        f = top.center["fast_hl"]
        recommended_speeds = [(f // 2, (f // 2) * 4), (f, f * 4), (f * 2, (f * 2) * 4)]
        recommended_speeds = [(a, b) for a, b in recommended_speeds if a >= 2]
        print(f"\n[b2e-sweep] PLATEAU CENTRE -> single-speed (fast={f}, slow={f * 4})")
        print(f"[b2e-sweep] RECOMMENDED 3-speed ensemble: {recommended_speeds}")
    else:
        print("[b2e-sweep] NO plateau candidates passed the spread + positivity gate.")
        print("[b2e-sweep] Interpretation: no flat high-Sharpe region on this universe;")
        print("[b2e-sweep] either the speed grid is too sparse OR EWMAC is genuinely")
        print("[b2e-sweep] not robust on the IBKR cross-asset 9y window.")

    # Compare to B2 canonical (16/64).
    canonical = next((i for i, c in enumerate(grid_cells) if c["fast_hl"] == 16), None)
    if canonical is not None:
        print(
            f"\n[b2e-sweep] B2 canonical (16/64) on IBKR x-asset IS: "
            f"Sharpe = {sharpes[canonical]:+.4f}"
        )

    # Write artefacts.
    report = format_plateau_report(res, candidates, audit_label="B2e IBKR cross-asset EWMAC L52 SWEEP")
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    # Cells long-form CSV.
    rows = []
    for cell, sh, vol, n in zip(grid_cells, sharpes, vols, n_obs):
        rows.append({**cell, "sharpe": sh, "vol": vol, "n_obs": n})
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "cells_long.csv", index=False)
    print(f"\n[b2e-sweep] wrote: {REPORTS_DIR / 'plateau_report.md'}")
    print(f"[b2e-sweep] wrote: {REPORTS_DIR / 'cells_long.csv'}")


if __name__ == "__main__":
    main()
