"""gold_macro (DAILY_TREND with cross-asset composite) -- Pardo-style sweep on IS data.

**EXPLORATORY ONLY** (L52). Outputs are a PRIOR for a re-pre-reg directive,
NOT a deployment gate.

Strategy: 3-component cross-asset composite on GLD:
    1. Real-rate proxy: -(d log(TIP/TLT) over `real_rate_window`)
    2. Dollar weakness: -d log(DXY) over `dollar_window`
    3. Momentum gate: GLD close > SMA(`slow_ma`)
Entry: composite_z (mean of rr_z + d_z) > 0 AND momentum gate ON -> LONG GLD.
Exit:  composite_z <= 0 OR momentum gate OFF.
Sizing: vol-targeted at 10% annualised (EWMA span 20), capped at 1.5x.

Live config (`config/gold_macro_gld.toml`): slow_ma=200, real_rate_window=20,
dollar_window=20, vol_target_pct=0.10.

Sweep axes (2 most-impactful):
    `slow_ma in {50, 100, 150, 200, 300}` x `real_rate_window in {10, 20, 40, 60}`

Dollar_window is fixed at 20 (live) to keep the surface 2D and visualisable.
A 3rd-axis sensitivity check could follow if the 2D plateau is borderline.

Sanctuary discipline (V3.6): hold out last 24 months. With ~16y of overlapping
GLD+TIP+TLT+DXY data (DXY is the binding constraint, starts 2010-01),
IS ~ 14y x 252 = ~3,500 bars, sanctuary ~500 bars.

L21 causality discipline:
    - Composite signal at bar t uses TIP/TLT/DXY closes at t (known by EOD t).
    - Z-score uses EXPANDING mean+std over signal history up to and including t.
    - Momentum gate uses SMA(t-slow_ma+1 .. t) on GLD close.
    - Position decided at t earns return from t -> t+1: ``position.shift(1) * gld_ret``.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_gold_macro.py
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
from titan.research.metrics import BARS_PER_YEAR  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_gold_macro"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24
DOLLAR_WINDOW = 20  # fixed
VOL_TARGET = 0.10
VOL_EWMA_SPAN = 20
MAX_LEVERAGE = 1.5
COST_BPS_PER_TURNOVER = 1.5  # daily ETF, conservative
ZSCORE_MIN_OBS = 60  # expanding z-score warmup


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    gld = _load_close("GLD")
    tip = _load_close("TIP")
    tlt = _load_close("TLT")
    dxy = _load_close("DXY")
    # Inner-join on dates so signals always line up.
    df = pd.DataFrame({"GLD": gld, "TIP": tip, "TLT": tlt, "DXY": dxy}).dropna()
    return df


def _expanding_zscore(series: pd.Series, *, min_obs: int = ZSCORE_MIN_OBS) -> pd.Series:
    """Causal expanding z-score: at t, uses values in [0..t] only.

    NumPy implementation for speed. Returns 0 for warmup positions.
    """
    arr = series.to_numpy()
    n = len(arr)
    out = np.zeros(n, dtype=float)
    # Expanding sum + sumsq for O(n) mean/std.
    csum = np.cumsum(arr)
    csumsq = np.cumsum(arr * arr)
    for i in range(min_obs, n):
        k = i + 1
        mu = csum[i] / k
        var = csumsq[i] / k - mu * mu
        if var <= 1e-12:
            out[i] = 0.0
        else:
            out[i] = (arr[i] - mu) / np.sqrt(var)
    return pd.Series(out, index=series.index)


def gold_macro_strategy_fn(
    closes_df: pd.DataFrame, *, slow_ma: int, real_rate_window: int
) -> pd.Series:
    """Per-bar net return of the 3-component composite on GLD.

    Causality (L18 shift discipline):
        - All component signals at t use data known by EOD t (no future).
        - Z-score is causal expanding (past-only).
        - Position effective at t earns return from t -> t+1.
    """
    gld = closes_df["GLD"]
    tip = closes_df["TIP"]
    tlt = closes_df["TLT"]
    dxy = closes_df["DXY"]

    # Component 1: real rate proxy. Signal = -d log(TIP/TLT) over window.
    log_rr = np.log(tip / tlt)
    rr_signal = -(log_rr - log_rr.shift(real_rate_window))

    # Component 2: dollar weakness. Signal = -d log(DXY) over fixed window.
    log_dxy = np.log(dxy)
    d_signal = -(log_dxy - log_dxy.shift(DOLLAR_WINDOW))

    # Z-score each component (causal expanding) and average.
    rr_z = _expanding_zscore(rr_signal.fillna(0.0))
    d_z = _expanding_zscore(d_signal.fillna(0.0))
    composite_z = (rr_z + d_z) / 2.0

    # Component 3: momentum gate (GLD > SMA).
    sma = gld.rolling(slow_ma, min_periods=slow_ma).mean()
    momentum = (gld > sma).astype(float)

    # Entry condition: composite_z > 0 AND momentum gate ON.
    signal = ((composite_z > 0) & (momentum > 0)).astype(float)

    # Vol-target sizing on GLD.
    gld_ret = np.log(gld / gld.shift(1)).fillna(0.0)
    var = gld_ret.pow(2).ewm(span=VOL_EWMA_SPAN, adjust=False, min_periods=VOL_EWMA_SPAN).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (VOL_TARGET / realised_vol_ann.replace(0, np.nan)).clip(upper=MAX_LEVERAGE).fillna(0.0)
    position = (signal * scale).fillna(0.0)

    # Per-bar return: position effective at t earns return from t -> t+1.
    held = position.shift(1).fillna(0.0)
    gross = held * gld_ret

    # Costs.
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    net = gross - cost
    return net.rename("ret")


def assert_causal_gold_macro(closes_df: pd.DataFrame) -> None:
    """L21 causality smoke: corrupting future bars must leave past returns unchanged."""
    closes_clean = closes_df.copy()
    baseline = gold_macro_strategy_fn(closes_clean, slow_ma=200, real_rate_window=20)

    # Corrupt the last 20 bars of every column.
    closes_corrupt = closes_df.copy()
    n = len(closes_corrupt)
    cutoff = n - 20
    for col in closes_corrupt.columns:
        closes_corrupt.iloc[cutoff:, closes_corrupt.columns.get_loc(col)] = (
            closes_corrupt.iloc[cutoff:, closes_corrupt.columns.get_loc(col)] * 100.0
        )
    perturbed = gold_macro_strategy_fn(closes_corrupt, slow_ma=200, real_rate_window=20)

    # Past returns (before cutoff - some buffer for shift+windows) must be IDENTICAL.
    # Use cutoff - 250 to be safe (accounts for slow_ma=200 window).
    safe_end = cutoff - 250
    if safe_end <= 0:
        raise RuntimeError("Not enough data for causality smoke")
    diff = (baseline.iloc[:safe_end] - perturbed.iloc[:safe_end]).abs().max()
    if diff > 1e-12:
        raise AssertionError(
            f"L21 CAUSALITY FAIL: past returns changed by max {diff} when future was corrupted. "
            f"Strategy has look-ahead bias."
        )
    print(f"[gm-sweep] L21 causality smoke PASS (max past-return diff = {diff:.2e})")


def main() -> None:
    closes = load_universe()
    print(
        f"[gm-sweep] universe: GLD+TIP+TLT+DXY, common range "
        f"{closes.index[0].date()} -> {closes.index[-1].date()}"
    )
    print(f"[gm-sweep] total bars: {closes.shape[0]}")

    # L21 causality smoke FIRST (cheap, runs once).
    assert_causal_gold_macro(closes)

    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(
        f"[gm-sweep] IS slice: {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(
        f"[gm-sweep] sanctuary (held out): {sanctuary_closes.shape[0]} bars "
        f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})"
    )

    grid = {
        "slow_ma": [50, 100, 150, 200, 300],
        "real_rate_window": [10, 20, 40, 60],
    }
    n_cells = len(grid["slow_ma"]) * len(grid["real_rate_window"])
    print(f"[gm-sweep] running sweep over {n_cells} cells (dollar_window fixed at {DOLLAR_WINDOW})...")

    res = run_parameter_sweep(
        is_closes,
        strategy_fn=gold_macro_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=300 + 252,  # slow_ma max + 1y after
        meta={
            "strategy": "gold_macro (3-component composite -> GLD long, vol-targeted)",
            "universe": ["GLD", "TIP", "TLT", "DXY"],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "fixed_knobs": {
                "dollar_window": DOLLAR_WINDOW,
                "vol_target_pct": VOL_TARGET,
                "ewma_span": VOL_EWMA_SPAN,
                "max_leverage": MAX_LEVERAGE,
                "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
                "zscore_min_obs": ZSCORE_MIN_OBS,
            },
            "live_canonical": {"slow_ma": 200, "real_rate_window": 20},
            "context": "V1-era live strategy re-audit (Wave B full audit, L52 hybrid workflow)",
        },
    )

    print("\n[gm-sweep] Sharpe surface (rows = slow_ma, cols = real_rate_window):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[gm-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[gm-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[gm-sweep] NO plateau candidates passed the spread + positivity gate.")

    # Live canonical retrospective.
    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"slow_ma": 200, "real_rate_window": 20}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(f"\n[gm-sweep] LIVE canonical (slow_ma=200, rr_window=20): IS Sharpe = {canonical_sr:.3f}")
        finite_mask = np.isfinite(res.sharpes)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
            best_sr = res.sharpes[best_idx]
            best_cell = res.cells[best_idx]
            print(f"[gm-sweep] best cell in grid: {best_cell} -> IS Sharpe = {best_sr:.3f}")
            if abs(canonical_sr) > 1e-6:
                gap = (best_sr - canonical_sr) / abs(canonical_sr) * 100
                print(f"[gm-sweep] best-vs-canonical gap: {gap:+.1f}%")

    report = format_plateau_report(res, candidates, audit_label="gold_macro V1-era RE-AUDIT SWEEP")
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(res.to_dataframe().to_csv(index=False), encoding="utf-8")
    print(f"\n[gm-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
