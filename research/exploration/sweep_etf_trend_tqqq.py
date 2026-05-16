"""etf_trend TQQQ (DAILY_TREND, leveraged) — Pardo-style sweep on IS data.

**EXPLORATORY ONLY** (L52). Outputs are a PRIOR for a re-pre-reg directive,
NOT a deployment gate.

Strategy (simplified pure-research version mirroring live
`config/etf_trend_tqqq.toml`): SIGNAL from QQQ trend (long when
`qqq_close > SMA(qqq, slow_ma)` for `exit_confirm_days`), POSITION in
TQQQ (3x leveraged). Binary sizing (1.0 when long, 0.0 when flat) —
TQQQ's 3x leverage IS the position sizing.

**Critical difference from SPY audit:** TQQQ is 3x leveraged and binary
sized — so the strategy's economic claim is "trend filter avoids the
volatility-decay catastrophes of holding leveraged ETFs through
downtrends." The L17 relative-MC test against B&H TQQQ is the right test:
under bootstrap, can the trend filter ACTUALLY reduce drawdown vs always
holding TQQQ?

Wave A.2-confirm purpose: TQQQ is the variant where the trend filter is
MOST PLAUSIBLE to add value (leveraged ETF + crash protection thesis).
If TQQQ also fails L17 rel-MC, L56 generalises and we bulk-RETIRE the
entire etf_trend class.

Live config (`config/etf_trend_tqqq.toml`): `slow_ma=175,
exit_confirm_days=1, sizing_mode="binary"`.

Sweep axes: same as SPY for direct comparability.
    `slow_ma ∈ {50, 100, 150, 200, 250, 300}` ×
    `exit_confirm_days ∈ {1, 2, 3, 5, 10}`

Sanctuary: last 24 months. TQQQ data 2010-2026 → IS ~14y.
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_etf_trend_tqqq"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24
COST_BPS_PER_TURNOVER = 1.0


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    qqq = _load_close("QQQ")
    tqqq = _load_close("TQQQ")
    common = qqq.index.intersection(tqqq.index)
    return pd.DataFrame({"QQQ": qqq.reindex(common), "TQQQ": tqqq.reindex(common)}).dropna()


def etf_trend_tqqq_returns(
    closes_df: pd.DataFrame, *, slow_ma: int, exit_confirm_days: int
) -> pd.Series:
    """Per-bar net return — signal from QQQ, position in TQQQ (binary, 3x).

    Causality (L18): SMA at t-1, position shift by 1.
    """
    qqq = closes_df["QQQ"]
    tqqq = closes_df["TQQQ"]
    sma = qqq.rolling(slow_ma, min_periods=slow_ma).mean()
    above = (qqq > sma).astype(float)

    # Exit-confirm state machine on QQQ signal.
    arr = above.to_numpy()
    sig = np.zeros(len(qqq), dtype=float)
    pos = 0
    days_below = 0
    for i in range(len(qqq)):
        if np.isnan(arr[i]):
            sig[i] = float(pos)
            continue
        if pos == 0 and arr[i] == 1.0:
            pos = 1
            days_below = 0
        elif pos == 1:
            if arr[i] == 0.0:
                days_below += 1
                if days_below >= exit_confirm_days:
                    pos = 0
                    days_below = 0
            else:
                days_below = 0
        sig[i] = float(pos)
    position = pd.Series(sig, index=qqq.index)

    # TQQQ returns (already 3x leveraged with intraday decay baked in).
    tqqq_ret = np.log(tqqq / tqqq.shift(1)).fillna(0.0)
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * tqqq_ret

    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    return (gross - cost).rename("ret")


def main() -> None:
    closes = load_universe()
    print(f"[tqqq-sweep] universe: QQQ + TQQQ, common range "
          f"{closes.index[0].date()} -> {closes.index[-1].date()}, {closes.shape[0]} bars")

    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(f"[tqqq-sweep] IS slice: {is_closes.shape[0]} bars "
          f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})")
    print(f"[tqqq-sweep] sanctuary (held out): {sanctuary_closes.shape[0]} bars "
          f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})")

    grid = {
        "slow_ma": [50, 100, 150, 200, 250, 300],
        "exit_confirm_days": [1, 2, 3, 5, 10],
    }
    n_cells = len(grid["slow_ma"]) * len(grid["exit_confirm_days"])
    print(f"[tqqq-sweep] running sweep over {n_cells} cells...")

    res = run_parameter_sweep(
        is_closes,
        strategy_fn=etf_trend_tqqq_returns,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=300 + 252,
        meta={
            "strategy": "etf_trend TQQQ (QQQ signal -> TQQQ binary 3x leveraged position)",
            "universe": ["QQQ (signal)", "TQQQ (traded)"],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "fixed_knobs": {
                "sizing_mode": "binary",
                "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
            },
            "live_canonical": {"slow_ma": 175, "exit_confirm_days": 1},
            "context": "V1-era live strategy re-audit Wave A.2-confirm (L52 + L56 generalisation test)",
        },
    )

    print("\n[tqqq-sweep] Sharpe surface (rows = slow_ma, cols = exit_confirm_days):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[tqqq-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[tqqq-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[tqqq-sweep] NO plateau candidates passed the spread + positivity gate.")

    # Live canonical retrospective. Live uses slow_ma=175 which isn't in the
    # grid; use slow_ma=200 as nearest neighbour.
    target_200 = next(
        (i for i, cell in enumerate(res.cells) if cell == {"slow_ma": 200, "exit_confirm_days": 1}),
        None,
    )
    if target_200 is not None and np.isfinite(res.sharpes[target_200]):
        print(f"\n[tqqq-sweep] LIVE canonical proxy (slow_ma=200, exit_confirm_days=1; "
              f"live uses 175): IS Sharpe = {res.sharpes[target_200]:.3f}")

    # Best cell.
    finite_mask = np.isfinite(res.sharpes)
    if finite_mask.any():
        best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
        best_sr = res.sharpes[best_idx]
        best_cell = res.cells[best_idx]
        print(f"[tqqq-sweep] best cell: {best_cell} -> IS Sharpe = {best_sr:.3f}")

    # Buy-and-hold TQQQ baseline (no signal — always long).
    tqqq = is_closes["TQQQ"]
    bh_ret = np.log(tqqq / tqqq.shift(1)).fillna(0.0)
    from titan.research.metrics import sharpe as _sharpe

    bh_sr = float(_sharpe(bh_ret, periods_per_year=BARS_PER_YEAR["D"]))
    print(f"\n[tqqq-sweep] buy-and-hold TQQQ (no signal): IS Sharpe = {bh_sr:.3f}")
    print("[tqqq-sweep] The strategy must beat B&H TQQQ on Sharpe AND reduce MaxDD")
    print("[tqqq-sweep] under bootstrap (L17) — TQQQ's 3x leverage means a -33% QQQ")
    print("[tqqq-sweep] drawdown ≈ -90% TQQQ catastrophe. Trend filter timing is the thesis.")

    report = format_plateau_report(
        res, candidates, audit_label="etf_trend TQQQ V1-era RE-AUDIT SWEEP (Wave A.2-confirm)"
    )
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print(f"\n[tqqq-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")
    print(f"[tqqq-sweep] wrote: {(REPORTS_DIR / 'sharpe_surface.csv').relative_to(PROJECT_ROOT)}")
    print(f"[tqqq-sweep] wrote: {(REPORTS_DIR / 'cells_long.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
