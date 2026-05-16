"""etf_trend SPY (DAILY_TREND) — Pardo-style sweep on IS data.

**EXPLORATORY ONLY** (L52). Outputs are a PRIOR for a re-pre-reg directive,
NOT a deployment gate.

Strategy (simplified pure-research version mirroring the live SPY config
`config/etf_trend_spy.toml` with `decel_signals=[]`, `entry_mode=
"decel_positive"`, `exit_mode="A"`): Long SPY when ``close > SMA(slow_ma)``;
exit when ``close < SMA(slow_ma)`` for ``exit_confirm_days`` consecutive
bars. Vol-target sizing at 20% annualised vol, capped at 2x leverage.

Live config (`config/etf_trend_spy.toml`): `slow_ma=150,
exit_confirm_days=1, vol_target=0.20, max_leverage=2.0`.

Sweep axes (most-impactful to Sharpe given the live config):
    `slow_ma ∈ {50, 100, 150, 200, 250, 300}` ×
    `exit_confirm_days ∈ {1, 2, 3, 5, 10}`

Sanctuary discipline (V3.6): hold out last 24 months. With 23y of SPY data
(2003 → 2026), IS = ~21 years × 252 = ~5,300 bars.

The pure-research version DOES NOT model the ATR-based hard stop (which is
a risk-management feature, not a signal-edge feature). Sharpe sweep is over
signal cells; stop logic is invariant across the grid.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_etf_trend_spy.py
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_etf_trend_spy"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24
VOL_TARGET = 0.20
VOL_EWMA_SPAN = 20
MAX_LEVERAGE = 2.0
COST_BPS_PER_TURNOVER = 1.0  # 1bps per unit weight change (matches B2/B4/bond_gold)


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
    spy = _load_close("SPY")
    return pd.DataFrame({"SPY": spy})


def etf_trend_spy_returns(
    closes_df: pd.DataFrame, *, slow_ma: int, exit_confirm_days: int
) -> pd.Series:
    """Per-bar net return of the simplified SPY trend signal.

    Causality (L18 shift discipline):
        - SMA at close[t] uses [t-slow_ma+1..t]; known EOD t.
        - exit_confirm_days counter uses past-only.
        - Position at t earns return from t -> t+1 via .shift(1).
    """
    spy = closes_df["SPY"]
    sma = spy.rolling(slow_ma, min_periods=slow_ma).mean()
    above = (spy > sma).astype(float)

    # Build the signal with exit-confirm filter. State machine:
    #   in_pos=0; enter long the first bar above SMA;
    #   stay long until below SMA for exit_confirm_days consecutive bars.
    arr_above = above.to_numpy()
    sig = np.zeros(len(spy), dtype=float)
    pos = 0
    days_below = 0
    for i in range(len(spy)):
        if np.isnan(arr_above[i]):
            sig[i] = float(pos)
            continue
        if pos == 0 and arr_above[i] == 1.0:
            pos = 1
            days_below = 0
        elif pos == 1:
            if arr_above[i] == 0.0:
                days_below += 1
                if days_below >= exit_confirm_days:
                    pos = 0
                    days_below = 0
            else:
                days_below = 0
        sig[i] = float(pos)
    sig_series = pd.Series(sig, index=spy.index)

    # Vol-target sizing.
    log_ret = np.log(spy / spy.shift(1)).fillna(0.0)
    var = log_ret.pow(2).ewm(span=VOL_EWMA_SPAN, adjust=False, min_periods=VOL_EWMA_SPAN).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (VOL_TARGET / realised_vol_ann.replace(0, np.nan)).clip(upper=MAX_LEVERAGE).fillna(0.0)
    position = (sig_series * scale).fillna(0.0)

    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * log_ret

    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    return (gross - cost).rename("ret")


def main() -> None:
    closes = load_universe()
    print(f"[spy-sweep] SPY data: {closes.shape[0]} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")

    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(
        f"[spy-sweep] IS slice: {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(
        f"[spy-sweep] sanctuary (held out): {sanctuary_closes.shape[0]} bars "
        f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})"
    )

    grid = {
        "slow_ma": [50, 100, 150, 200, 250, 300],
        "exit_confirm_days": [1, 2, 3, 5, 10],
    }
    n_cells = len(grid["slow_ma"]) * len(grid["exit_confirm_days"])
    print(f"[spy-sweep] running sweep over {n_cells} cells...")

    res = run_parameter_sweep(
        is_closes,
        strategy_fn=etf_trend_spy_returns,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=300 + 252,  # slow_ma=300 warmup + at least 1y to evaluate
        meta={
            "strategy": "etf_trend SPY (simplified: long > SMA, exit < SMA for N consec days, vol-target 20%)",
            "universe": ["SPY"],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "fixed_knobs": {
                "vol_target": VOL_TARGET,
                "vol_ewma_span": VOL_EWMA_SPAN,
                "max_leverage": MAX_LEVERAGE,
                "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
            },
            "live_canonical": {"slow_ma": 150, "exit_confirm_days": 1},
            "context": "V1-era live strategy re-audit Wave A.2 (L52 hybrid workflow)",
        },
    )

    print("\n[spy-sweep] Sharpe surface (rows = slow_ma, cols = exit_confirm_days):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[spy-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[spy-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[spy-sweep] NO plateau candidates passed the spread + positivity gate.")

    # Live canonical retrospective.
    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"slow_ma": 150, "exit_confirm_days": 1}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(f"\n[spy-sweep] LIVE canonical (slow_ma=150, exit_confirm_days=1): IS Sharpe = {canonical_sr:.3f}")
        finite_mask = np.isfinite(res.sharpes)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
            best_sr = res.sharpes[best_idx]
            best_cell = res.cells[best_idx]
            print(f"[spy-sweep] best cell in grid: {best_cell} -> IS Sharpe = {best_sr:.3f}")
            if abs(canonical_sr) > 1e-6:
                gap = (best_sr - canonical_sr) / abs(canonical_sr) * 100
                print(f"[spy-sweep] best-vs-canonical gap: {gap:+.1f}%")

    # Buy-and-hold comparator — important for any trend strategy on equities.
    spy = is_closes["SPY"]
    bh_ret = np.log(spy / spy.shift(1)).fillna(0.0)
    from titan.research.metrics import sharpe as _sharpe

    bh_sr = float(_sharpe(bh_ret, periods_per_year=BARS_PER_YEAR["D"]))
    print(f"\n[spy-sweep] buy-and-hold SPY (no costs, no vol-target): IS Sharpe = {bh_sr:.3f}")
    print("[spy-sweep] NOTE: the strategy's value is risk-reduction vs B&H, not raw Sharpe.")

    report = format_plateau_report(
        res, candidates, audit_label="etf_trend SPY V1-era RE-AUDIT SWEEP"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print(f"\n[spy-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")
    print(f"[spy-sweep] wrote: {(REPORTS_DIR / 'sharpe_surface.csv').relative_to(PROJECT_ROOT)}")
    print(f"[spy-sweep] wrote: {(REPORTS_DIR / 'cells_long.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
