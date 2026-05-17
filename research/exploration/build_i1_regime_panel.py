"""I1 regime panel construction — V3.7 prep for HMM/regime-classifier audit.

Builds the multi-feature regime indicator panel for I1 (HMM regime gate
+ XGBoost meta-labeler). Uses existing daily data; no new acquisition
needed.

Features (all daily, indexed to common business-day grid):
  1. VIX level (z-scored on 252d rolling) — equity-vol regime
  2. Term-spread proxy: log(TLT / IEF) z-score — yield-curve steepness
  3. Credit spread: log(HYG / IEF) z-score — risk-appetite
  4. SPY realised vol (20d, annualised) z-score
  5. SPY trend (close > SMA200)
  6. DXY z-score (dollar strength)
  7. SPY drawdown velocity (DD-21d change)

Output: data/i1_regime_panel.parquet — 7-column DataFrame indexed on
daily dates. Used as input for the I1 audit.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/build_i1_regime_panel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "i1_regime_panel.parquet"


def _load_d(symbol: str) -> pd.Series:
    """Load daily close series with tz-naive normalized DateTimeIndex."""
    df = pd.read_parquet(DATA_DIR / f"{symbol}_D.parquet")
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def _zscore_rolling(s: pd.Series, window: int = 252) -> pd.Series:
    """Causal rolling z-score over `window` bars."""
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std()
    return ((s - mu) / sigma.replace(0, np.nan)).rename(s.name)


def main() -> None:
    print("=" * 72)
    print("I1 regime panel construction")
    print("=" * 72)

    # Load components
    print("\n[load]")
    components = {
        "VIX": _load_d("VIX"),
        "SPY": _load_d("SPY"),
        "HYG": _load_d("HYG"),
        "IEF": _load_d("IEF"),
        "TLT": _load_d("TLT"),
        "DXY": _load_d("DXY"),
    }
    for k, v in components.items():
        print(f"  {k}: {len(v)} bars, {v.index[0].date()} -> {v.index[-1].date()}")

    # Common index = intersection (will lose first ~1y to rolling z-windows)
    common = None
    for s in components.values():
        common = s.index if common is None else common.intersection(s.index)
    print(f"\n[align] common index: {len(common)} bars, "
          f"{common[0].date()} -> {common[-1].date()}")

    panel = pd.DataFrame(index=common)

    # Feature 1: VIX z-score (regime: hostile when high)
    print("\n[features]")
    panel["vix_z"] = _zscore_rolling(components["VIX"].reindex(common), 252)
    print(f"  vix_z: {panel['vix_z'].notna().sum()} non-NaN")

    # Feature 2: Term-spread (TLT/IEF -- log ratio z-score)
    # Higher TLT/IEF ratio = steeper curve (longer rates higher) = benign
    term_spread = np.log(components["TLT"].reindex(common) / components["IEF"].reindex(common))
    panel["term_spread_z"] = _zscore_rolling(term_spread, 252)
    print(f"  term_spread_z: {panel['term_spread_z'].notna().sum()} non-NaN")

    # Feature 3: Credit spread (HYG/IEF -- log ratio z-score)
    # Higher HYG/IEF = HY outperforming = risk-on = benign
    credit_spread = np.log(components["HYG"].reindex(common) / components["IEF"].reindex(common))
    panel["credit_spread_z"] = _zscore_rolling(credit_spread, 252)
    print(f"  credit_spread_z: {panel['credit_spread_z'].notna().sum()} non-NaN")

    # Feature 4: SPY realised vol (20d) z-score
    spy = components["SPY"].reindex(common)
    spy_log_ret = np.log(spy / spy.shift(1))
    realised_vol_20 = spy_log_ret.rolling(20, min_periods=15).std() * np.sqrt(252)
    panel["rv20_z"] = _zscore_rolling(realised_vol_20, 252)
    print(f"  rv20_z: {panel['rv20_z'].notna().sum()} non-NaN")

    # Feature 5: SPY trend (close > SMA200; binary 0/1)
    sma200 = spy.rolling(200, min_periods=100).mean()
    panel["spy_above_sma200"] = (spy > sma200).astype(float)
    print(f"  spy_above_sma200: {panel['spy_above_sma200'].notna().sum()} non-NaN")

    # Feature 6: DXY z-score
    panel["dxy_z"] = _zscore_rolling(components["DXY"].reindex(common), 252)
    print(f"  dxy_z: {panel['dxy_z'].notna().sum()} non-NaN")

    # Feature 7: SPY drawdown velocity (change in DD over 21d)
    cumret = spy_log_ret.cumsum()
    peak = cumret.cummax()
    dd = cumret - peak
    panel["dd_velocity_21"] = dd.diff(21)
    print(f"  dd_velocity_21: {panel['dd_velocity_21'].notna().sum()} non-NaN")

    # Drop early NaN rows (rolling windows need history)
    panel_clean = panel.dropna()
    print(f"\n[clean] full panel: {len(panel)} rows, "
          f"after dropna: {len(panel_clean)} rows "
          f"({panel_clean.index[0].date()} -> {panel_clean.index[-1].date()})")

    # Save
    panel_clean.to_parquet(OUTPUT_FILE)
    print(f"\n[write] {OUTPUT_FILE} ({len(panel_clean)} rows × {len(panel_clean.columns)} cols)")

    # Quick stats
    print("\n[stats] correlation matrix:")
    print(panel_clean.corr().round(3).to_string())


if __name__ == "__main__":
    main()
