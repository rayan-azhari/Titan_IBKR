"""turtle (DAILY_TREND, H1 Donchian breakout) -- Pardo-style sweep on IS data.

**EXPLORATORY ONLY** (L52). Outputs are a PRIOR for a re-pre-reg directive,
NOT a deployment gate.

Strategy (live config matches CAT H1):
    - Entry: long when close > Donchian_high(entry_period).shift(1)
    - Exit: close < Donchian_low(exit_period).shift(1)
    - Binary signal (single unit; pyramiding handled in a later phase)
    - Vol-scaled sizing via ATR-implied risk (kept off for the sweep --
      sweep tests the raw signal economics, sizing is downstream)
    - 1 bp cost per turnover (US equity H1, conservative)

Live config (`config/turtle_h1.toml`): entry_period=45, exit_period=30,
atr_period=20, risk_pct=0.01, stop_atr_mult=2.0, max_units=4,
pyramid_atr_mult=0.5, use_trailing_stop=True, direction=long_only.

Sweep axes (2 most-impactful):
    `entry_period in {20, 30, 45, 60, 90}` x `exit_period in {10, 20, 30, 45}`

Stop/pyramiding/trailing are frozen at live values for the sweep -- these
are sizing/risk-management overlays, not signal-defining params. Sweeping
them would multiply cells without improving signal-edge identification.

Sanctuary discipline (V3.6): hold out last 12 months (~1764 H1 bars).
With CAT's 7.9y H1 history (2018-05 to 2026-03), IS = ~6.9y x ~1764 bars
per year = ~12,000 bars.

L21 causality discipline:
    - Donchian channels at bar t use bars [t-period+1 .. t-1] (shift(1)
      forces past-only).
    - Position decided at t earns return from t -> t+1: position.shift(1) * log_ret.

Annualisation note (L52/L53 hidden-bug warning):
    US equity H1 RTH cadence is ~7 bars/day x 252 = 1,764 bars/year, NOT
    BARS_PER_YEAR["H1"] = 6048 (which assumes 24/7 FX). The Wave B triage
    used the FX value -- that OVERSTATES Sharpe by sqrt(6048/1764) = 1.85x.
    The triage's +1.60 corresponds to a properly-annualised ~0.87. This
    sweep uses 1764 throughout.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_turtle.py
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

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_turtle"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# US equity H1 RTH cadence: ~7 bars/day x 252 days/year (NOT 24/7 FX).
PERIODS_PER_YEAR_H1_EQ = 1764
SANCTUARY_MONTHS = 12
COST_BPS_PER_TURNOVER = 1.0  # US equity H1, conservative
DEFAULT_TICKER = "CAT"  # live config instrument


def _load_h1(symbol: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{symbol}_H1.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    return df[["open", "high", "low", "close"]].astype(float)


def turtle_strategy_fn(bars: pd.DataFrame, *, entry_period: int, exit_period: int) -> pd.Series:
    """Per-bar net return of the Donchian breakout signal (long-only, binary).

    Causality (L18 + L21):
        - Donchian_high/low use [t-period+1 .. t-1] via shift(1) -- past-only.
        - Position decided at close[t] earns return from t -> t+1
          via position.shift(1) * log_return.
    """
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]

    # Donchian channels: shifted by 1 so the breakout test at bar t uses
    # the channel computed on [t-period+1 .. t-1] only.
    donch_high = high.rolling(entry_period, min_periods=entry_period).max().shift(1)
    donch_low = low.rolling(exit_period, min_periods=exit_period).min().shift(1)

    arr_close = close.to_numpy()
    arr_high = donch_high.to_numpy()
    arr_low = donch_low.to_numpy()
    pos = np.zeros(len(close), dtype=float)
    state = 0
    for i in range(len(close)):
        if np.isnan(arr_high[i]) or np.isnan(arr_low[i]):
            pos[i] = float(state)
            continue
        if state == 0 and arr_close[i] > arr_high[i]:
            state = 1
        elif state == 1 and arr_close[i] < arr_low[i]:
            state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=close.index)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret

    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    return (gross - cost).rename("ret")


def assert_causal_turtle(bars: pd.DataFrame) -> None:
    """L21 causality smoke: corrupting future bars must leave past returns unchanged."""
    baseline = turtle_strategy_fn(bars, entry_period=45, exit_period=30)

    corrupted = bars.copy()
    n = len(corrupted)
    cutoff = n - 50
    if cutoff <= 200:
        raise RuntimeError("Not enough data for causality smoke")
    for col in ("open", "high", "low", "close"):
        corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] = (
            corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] * 100.0
        )
    perturbed = turtle_strategy_fn(corrupted, entry_period=45, exit_period=30)

    safe_end = cutoff - 100
    diff = (baseline.iloc[:safe_end] - perturbed.iloc[:safe_end]).abs().max()
    if diff > 1e-12:
        raise AssertionError(
            f"L21 CAUSALITY FAIL: corrupting future bars changed past returns by {diff}"
        )
    print(f"[turtle-sweep] L21 causality smoke PASS (max past-return diff = {diff:.2e})")


def main(ticker: str = DEFAULT_TICKER) -> None:
    bars = _load_h1(ticker)
    print(f"[turtle-sweep] universe: {ticker} H1, range {bars.index[0]} -> {bars.index[-1]}")
    print(f"[turtle-sweep] total bars: {bars.shape[0]}")

    assert_causal_turtle(bars)

    cutoff = bars.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_bars = bars.loc[:cutoff]
    sanctuary_bars = bars.loc[cutoff:]
    print(
        f"[turtle-sweep] IS slice: {is_bars.shape[0]} bars "
        f"({is_bars.index[0]} -> {is_bars.index[-1]})"
    )
    print(
        f"[turtle-sweep] sanctuary: {sanctuary_bars.shape[0]} bars "
        f"({sanctuary_bars.index[0]} -> {sanctuary_bars.index[-1]})"
    )

    grid = {
        "entry_period": [20, 30, 45, 60, 90],
        "exit_period": [10, 20, 30, 45],
    }
    n_cells = len(grid["entry_period"]) * len(grid["exit_period"])
    print(
        f"[turtle-sweep] running sweep over {n_cells} cells, periods_per_year={PERIODS_PER_YEAR_H1_EQ}..."
    )

    res = run_parameter_sweep(
        is_bars,
        strategy_fn=turtle_strategy_fn,
        param_grid=grid,
        periods_per_year=PERIODS_PER_YEAR_H1_EQ,
        min_is_bars=200,
        meta={
            "strategy": f"turtle ({ticker} H1, Donchian breakout, binary long-only)",
            "universe": [ticker],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "fixed_knobs": {
                "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
                "periods_per_year": PERIODS_PER_YEAR_H1_EQ,
                "annualisation_note": "1764 = 7 bars/day x 252 days (US equity RTH H1, NOT 24/7 FX)",
            },
            "live_canonical": {"entry_period": 45, "exit_period": 30},
            "context": "V1-era live strategy re-audit (Wave B full audit, L52 hybrid workflow)",
        },
    )

    print("\n[turtle-sweep] Sharpe surface (rows = entry_period, cols = exit_period):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[turtle-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[turtle-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[turtle-sweep] NO plateau candidates passed the spread + positivity gate.")

    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"entry_period": 45, "exit_period": 30}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(
            f"\n[turtle-sweep] LIVE canonical (entry=45, exit=30): IS Sharpe = {canonical_sr:.3f}"
        )
        print(
            f"  (Note: Wave B triage Sharpe +1.60 used 24/7 FX annualisation; with US equity H1 RTH (1764), expect ~+{canonical_sr:.2f}.)"
        )
        finite_mask = np.isfinite(res.sharpes)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
            best_sr = res.sharpes[best_idx]
            best_cell = res.cells[best_idx]
            print(f"[turtle-sweep] best cell in grid: {best_cell} -> IS Sharpe = {best_sr:.3f}")
            if abs(canonical_sr) > 1e-6:
                gap = (best_sr - canonical_sr) / abs(canonical_sr) * 100
                print(f"[turtle-sweep] best-vs-canonical gap: {gap:+.1f}%")

    report = format_plateau_report(
        res, candidates, audit_label=f"turtle {ticker} V1-era RE-AUDIT SWEEP"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )
    print(
        f"\n[turtle-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
