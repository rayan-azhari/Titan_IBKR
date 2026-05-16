"""Demo: Pardo-style EWMAC speed sweep on B2 IS data.

**EXPLORATORY ONLY** (L52). This script does NOT make any deployment claim.
Its purpose is to demonstrate the hybrid workflow and to retrospectively
check whether B2/B2b's canonical (16/64, 32/128, 64/256) sat on a flat
high-Sharpe plateau in IS — or whether it was a knife-edge that the V3.6
audit then exposed on OOS.

Workflow:
    1. Load B2b's yfinance-ETF-proxy universe (no IG quota dependency).
    2. Slice IS-only (drop the last 5 years of "OOS" so we never peek).
    3. Sweep single-speed Carver-style EWMAC across (fast_hl, slow_hl) on
       IS data.
    4. Print the Sharpe surface + run plateau detection.
    5. Save the report to ``.tmp/reports/sweep_b2_ewmac_demo/``.

Run::

    uv run python research/exploration/demo_b2_ewmac_sweep.py
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

DATA_DIR = PROJECT_ROOT / "data" / "yf_b2b"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_b2_ewmac_demo"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Subset of B2b's universe — keep 8 instruments for a fast demo (full 31
# would also work; this is just a demo run, not a real audit).
DEMO_ROOTS: tuple[str, ...] = (
    "SPX",
    "NDX",
    "GOLD_PROXY",
    "SILVER_PROXY",
    "US10Y_PROXY",
    "EURUSD",
    "USDJPY",
    "FTSE",
)


def _load_close(root: str) -> pd.Series | None:
    fp = DATA_DIR / f"{root}_DAY.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "close" not in df.columns:
        return None
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def load_demo_universe() -> pd.DataFrame:
    series: list[pd.Series] = []
    missing: list[str] = []
    for r in DEMO_ROOTS:
        s = _load_close(r)
        if s is None or s.dropna().shape[0] < 1000:
            missing.append(r)
            continue
        series.append(s)
    if missing:
        print(f"[warn] missing/short symbols: {missing}")
    if not series:
        raise SystemExit(
            "No demo data found. Run `uv run python scripts/download_b2b_alternative.py` first."
        )
    df = pd.concat(series, axis=1).dropna(how="all")
    return df


def carver_singlespeed_returns(closes_df: pd.DataFrame, *, fast_hl: int, slow_hl: int) -> pd.Series:
    """Single-speed Carver-style EWMAC, equal-weight across assets, daily.

    Causal (L18 shift discipline). Monthly rebalance, no costs (demo).
    Returns NaN for cells where fast_hl >= slow_hl (handled by allow_cell).
    """
    if fast_hl >= slow_hl:
        return pd.Series(np.nan, index=closes_df.index, name="ret")
    log_ret = np.log(closes_df / closes_df.shift(1)).fillna(0.0)
    fast = closes_df.ewm(halflife=fast_hl, adjust=False).mean()
    slow = closes_df.ewm(halflife=slow_hl, adjust=False).mean()
    ewmac = fast - slow
    # Vol-normalise by daily price-change stdev (Carver convention).
    price_diff = closes_df.diff()
    vol_p = price_diff.rolling(20, min_periods=20).std(ddof=1).replace(0, np.nan)
    norm = (ewmac / vol_p).clip(lower=-20.0, upper=20.0)
    # Convert forecast to position: positions held constant within month.
    target_vol_per_asset = 0.10 / closes_df.shape[1]
    inst_vol = log_ret.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(BARS_PER_YEAR["D"])
    raw_pos = (norm / 10.0) * (target_vol_per_asset / inst_vol.replace(0, np.nan))
    # Monthly rebalance — hold position from the last bar of each month.
    month_end = closes_df.index.to_period("M")
    last_in_month = pd.Series(month_end).duplicated(keep="last").to_numpy()
    rebalance_mask = ~last_in_month
    pos = (
        raw_pos.where(pd.Series(rebalance_mask, index=closes_df.index), other=np.nan)
        .ffill()
        .fillna(0.0)
    )
    held = pos.shift(1).fillna(0.0)
    return (held * log_ret).sum(axis=1).rename("ret")


def main() -> None:
    closes = load_demo_universe()
    print(f"[demo] loaded {closes.shape[1]} instruments x {closes.shape[0]} bars")
    print(f"[demo] range: {closes.index[0].date()} -> {closes.index[-1].date()}")

    # IS-only slice: drop the last 5 years so the sweep can never peek.
    cutoff = closes.index[-1] - pd.DateOffset(years=5)
    is_closes = closes.loc[:cutoff]
    print(
        f"[demo] IS slice: {is_closes.shape[0]} bars "
        f"({is_closes.index[0].date()} -> {is_closes.index[-1].date()})"
    )
    print(f"[demo] OOS (held out): {closes.loc[cutoff:].shape[0]} bars — sweep never sees this")

    # Carver-style speed grid — fast in {4, 8, 16, 32, 64} x slow in {16, 32, 64, 128, 256, 512}.
    grid = {
        "fast_hl": [4, 8, 16, 32, 64],
        "slow_hl": [16, 32, 64, 128, 256, 512],
    }

    print("[demo] running sweep...")
    res = run_parameter_sweep(
        is_closes,
        strategy_fn=carver_singlespeed_returns,
        param_grid=grid,
        periods_per_year=BARS_PER_YEAR["D"],
        min_is_bars=252 * 5,
        allow_cell=lambda c: c["fast_hl"] < c["slow_hl"],
        meta={
            "strategy": "B2 EWMAC single-speed (Carver normalised)",
            "universe": list(closes.columns),
            "is_cutoff": str(cutoff.date()),
            "demo": True,
        },
    )

    print("\n[demo] Sharpe surface (rows = fast_hl, cols = slow_hl):")
    surface = res.to_surface()
    print(surface.round(3).to_string(na_rep="    .  "))

    print("\n[demo] plateau detection...")
    candidates = detect_plateau(res, spread_pct_max=0.30, min_neighbours=2, top_k=5)
    if candidates:
        print(f"[demo] {len(candidates)} plateau candidate(s) found:")
        for c in candidates:
            print(
                f"  #{c.rank}: center={c.center} sharpe={c.center_sharpe:.3f} "
                f"mean_hood={c.mean_neighbourhood_sharpe:.3f} "
                f"spread={c.spread_pct * 100:.1f}% "
                f"n_nb={len(c.neighbour_sharpes)}"
            )
    else:
        print("[demo] no plateau candidates passed the spread + positivity gate")

    # Write the report.
    report = format_plateau_report(res, candidates, audit_label="B2 EWMAC DEMO SWEEP")
    report_fp = REPORTS_DIR / "plateau_report.md"
    report_fp.write_text(report, encoding="utf-8")

    surface_fp = REPORTS_DIR / "sharpe_surface.csv"
    surface.to_csv(surface_fp)

    df_fp = REPORTS_DIR / "cells_long.csv"
    res.to_dataframe().to_csv(df_fp, index=False)

    print(f"\n[demo] wrote: {report_fp.relative_to(PROJECT_ROOT)}")
    print(f"[demo] wrote: {surface_fp.relative_to(PROJECT_ROOT)}")
    print(f"[demo] wrote: {df_fp.relative_to(PROJECT_ROOT)}")

    # B2 canonical retrospective: where did (16/64, 32/128, 64/256) sit?
    # Single-speed comparator: use the (32, 128) middle cell as the proxy.
    target = next(
        (i for i, cell in enumerate(res.cells) if cell == {"fast_hl": 32, "slow_hl": 128}),
        None,
    )
    if target is not None:
        print(
            f"\n[demo] B2 canonical proxy (fast_hl=32, slow_hl=128): "
            f"sharpe={res.sharpes[target]:.3f}"
        )
        print(
            "[demo] compare against the plateau centre above. If they differ, "
            "the canonical was a non-plateau choice (the L43 / L49 failure mode "
            "this hybrid workflow is designed to prevent)."
        )


if __name__ == "__main__":
    main()
