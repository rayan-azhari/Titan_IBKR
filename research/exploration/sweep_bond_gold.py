"""bond_gold (CROSS_ASSET_MOMENTUM) — Pardo-style sweep on IS data.

**EXPLORATORY ONLY** (L52). Outputs are a PRIOR for a re-pre-reg directive,
NOT a deployment gate.

Strategy: IEF (intermediate Treasury bond ETF) momentum → GLD (gold).
When the 60-day log-return of IEF z-scores above ``threshold``, go long
GLD vol-targeted. Hold ≥20 days, exit when z-score falls below threshold.

Live config (`config/bond_gold.toml`): `lookback=60, threshold=0.50,
hold_days=20, vol_target_pct=0.10, zscore_window=504`.

Sweep axes (most-impactful to Sharpe):
    `lookback ∈ {30, 45, 60, 90, 120}` × `threshold ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`

Sanctuary discipline (V3.6): hold out last 24 months. With 21y of GLD
data (2004→2026), IS = ~19 years × 252 = ~4,800 bars.

This sweep replicates the live signal logic in pure research form (returns
a per-bar net-of-cost return series). It does NOT include the
`max_leverage=1.5` cap because vol-targeting + the hold-day rule already
bounds gross. Adding the cap would not affect cell ranking.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_bond_gold.py
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_bond_gold"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24
HOLD_DAYS = 20
ZSCORE_WINDOW = 504
VOL_TARGET = 0.10
VOL_EWMA_SPAN = 20
COST_BPS_PER_TURNOVER = 1.0  # 1bps per unit weight change (matches B2/B4)


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
    ief = _load_close("IEF")
    gld = _load_close("GLD")
    # Common-date intersection so signal and target line up.
    common = ief.index.intersection(gld.index)
    return pd.DataFrame({"IEF": ief.reindex(common), "GLD": gld.reindex(common)}).dropna()


def bond_gold_strategy_fn(closes_df: pd.DataFrame, *, lookback: int, threshold: float) -> pd.Series:
    """Per-bar net return of the bond->gold signal.

    Causality (L18 shift discipline):
        - Bond momentum at close[t] uses ief.shift(0)/ief.shift(lookback)
          — known by EOD t.
        - Z-score at t uses rolling [t-zscore_window+1 .. t] — known by EOD t.
        - Position effective at t earns return from t -> t+1
          (``position.shift(1) * gld_log_return``).
    """
    ief = closes_df["IEF"]
    gld = closes_df["GLD"]

    # 1. Bond momentum: log-return over `lookback` days.
    bond_mom = np.log(ief / ief.shift(lookback))
    # 2. Rolling z-score over ZSCORE_WINDOW days (causal — past-only).
    zmean = bond_mom.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
    zstd = bond_mom.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std(ddof=1)
    zscore = (bond_mom - zmean) / zstd.replace(0, np.nan)
    # 3. Long-only entry: signal = 1 when zscore > threshold, else 0.
    raw_sig = (zscore > threshold).astype(float).fillna(0.0)
    # 4. Hold-day floor: once entered, stay in for >= HOLD_DAYS even if
    #    zscore drops. Iterative — small loop on signal transitions.
    sig = raw_sig.copy()
    held = 0
    pos = 0
    for i in range(len(sig)):
        if pos == 0 and raw_sig.iloc[i] == 1:
            pos = 1
            held = 0
        elif pos == 1:
            held += 1
            if held >= HOLD_DAYS and raw_sig.iloc[i] == 0:
                pos = 0
                held = 0
        sig.iloc[i] = float(pos)
    # 5. Vol-target sizing on GLD: target_vol / realised_vol.
    gld_ret = np.log(gld / gld.shift(1)).fillna(0.0)
    var = gld_ret.pow(2).ewm(span=VOL_EWMA_SPAN, adjust=False, min_periods=VOL_EWMA_SPAN).mean()
    realised_vol_ann = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (VOL_TARGET / realised_vol_ann.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)
    position = (sig * scale).fillna(0.0)

    # 6. Per-bar return: position EFFECTIVE at t earns gld_ret at t+1.
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * gld_ret

    # 7. Costs: bps drag per unit weight change.
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    net = gross - cost
    return net.rename("ret")


def main() -> None:
    closes = load_universe()
    print(
        f"[bg-sweep] universe: IEF + GLD, common range {closes.index[0].date()} -> {closes.index[-1].date()}"
    )
    print(f"[bg-sweep] total bars: {closes.shape[0]}")

    cutoff = closes.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    is_closes = closes.loc[:cutoff]
    sanctuary_closes = closes.loc[cutoff:]
    print(
        f"[bg-sweep] IS slice: {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(
        f"[bg-sweep] sanctuary (held out): {sanctuary_closes.shape[0]} bars "
        f"({sanctuary_closes.index[0].date()} -> {sanctuary_closes.index[-1].date()})"
    )

    grid = {
        "lookback": [30, 45, 60, 90, 120],
        "threshold": [0.00, 0.25, 0.50, 0.75, 1.00],
    }
    print(
        f"[bg-sweep] running sweep over {len(grid['lookback']) * len(grid['threshold'])} cells..."
    )

    res = run_parameter_sweep(
        is_closes,
        strategy_fn=bond_gold_strategy_fn,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=ZSCORE_WINDOW + 252,  # need at least 1y after the 504d z-window warms up
        meta={
            "strategy": "bond_gold (IEF momentum -> GLD, vol-targeted, monthly-ish via hold-floor)",
            "universe": ["IEF", "GLD"],
            "is_cutoff": str(cutoff.date()),
            "sanctuary_months": SANCTUARY_MONTHS,
            "fixed_knobs": {
                "hold_days": HOLD_DAYS,
                "zscore_window": ZSCORE_WINDOW,
                "vol_target_pct": VOL_TARGET,
                "ewma_span": VOL_EWMA_SPAN,
                "cost_bps_per_turnover": COST_BPS_PER_TURNOVER,
            },
            "live_canonical": {"lookback": 60, "threshold": 0.50},
            "context": "V1-era live strategy re-audit (Wave A.1, L52 hybrid workflow)",
        },
    )

    print("\n[bg-sweep] Sharpe surface (rows = lookback, cols = threshold):")
    surf = res.to_surface()
    print(surf.round(3).to_string(na_rep="    .  "))

    print("\n[bg-sweep] plateau detection (spread <= 30%, min_neighbours=2)...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[bg-sweep] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"hood_mean={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[bg-sweep] NO plateau candidates passed the spread + positivity gate.")

    # Live canonical retrospective.
    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"lookback": 60, "threshold": 0.50}),
        None,
    )
    if target is not None and np.isfinite(res.sharpes[target]):
        canonical_sr = res.sharpes[target]
        print(
            f"\n[bg-sweep] LIVE canonical (lookback=60, threshold=0.50): IS Sharpe = {canonical_sr:.3f}"
        )
        finite_mask = np.isfinite(res.sharpes)
        if finite_mask.any():
            best_idx = int(np.argmax(np.where(finite_mask, res.sharpes, -np.inf)))
            best_sr = res.sharpes[best_idx]
            best_cell = res.cells[best_idx]
            print(f"[bg-sweep] best cell in grid: {best_cell} -> IS Sharpe = {best_sr:.3f}")
            if abs(canonical_sr) > 1e-6:
                gap = (best_sr - canonical_sr) / abs(canonical_sr) * 100
                print(f"[bg-sweep] best-vs-canonical gap: {gap:+.1f}%")

    report = format_plateau_report(res, candidates, audit_label="bond_gold V1-era RE-AUDIT SWEEP")
    (REPORTS_DIR / "plateau_report.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "sharpe_surface.csv").write_text(surf.to_csv(), encoding="utf-8")
    (REPORTS_DIR / "cells_long.csv").write_text(
        res.to_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print(f"\n[bg-sweep] wrote: {(REPORTS_DIR / 'plateau_report.md').relative_to(PROJECT_ROOT)}")
    print(f"[bg-sweep] wrote: {(REPORTS_DIR / 'sharpe_surface.csv').relative_to(PROJECT_ROOT)}")
    print(f"[bg-sweep] wrote: {(REPORTS_DIR / 'cells_long.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
