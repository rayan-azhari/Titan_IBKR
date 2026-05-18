"""gld_confluence signal-layer L21 smoke + threshold sweep (Wave B).

Already DEPRECATED 2026-05-01 per strategy.py docstring (V1 Sharpe
+1.46 not reproducible on full GLD H1; best fresh result +0.35 / 34%
positive folds / -27% DD). This sweep formalises the V3.6 RETIRE
under the audit protocol with L21 smoke + threshold scan.

Strategy (per strategy.py):
    - Single H1 GLD stream; higher TF signals via VIRTUAL downsampling
      (H1 native, H4 = every 4th, D = every 24th, W = every 120th).
      No multi-parquet alignment, so no L21 multi-TF risk; but
      causality-of-virtual-scales is worth verifying.
    - Per scale: trend_mom = sign(ma_spread_5_20) * |rsi_14_dev| / 50.
    - AND-gate: only enter when ALL 4 scales agree on sign.
    - Composite = weighted sum (H1=0.10, H4=0.05, D=0.55, W=0.30).
    - Z-score normalised on 5000-bar expanding window.
    - Entry: |z| > threshold.

Sweep axis: threshold in {0.50, 0.75 (live), 1.00, 1.25}. Other
parameters frozen.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_gld_confluence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.metrics import bootstrap_sharpe_ci, max_drawdown, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_gld_confluence"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# US-listed GLD on H1 RTH cadence -- correct annualisation per L60.
PERIODS_PER_YEAR_H1_EQ = 1764
COST_BPS_PER_TURNOVER = 1.5  # GLD ETF H1
SCALES = {"H1": 1, "H4": 4, "D": 24, "W": 120}
WEIGHTS = {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30}
ZSCORE_WINDOW = 5000


def _load_h1() -> pd.DataFrame:
    fp = DATA_DIR / "GLD_H1.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    return df[["open", "high", "low", "close"]].astype(float)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _scale_signal(close_h1: pd.Series, scale: int) -> pd.Series:
    """Compute trend_mom signal at virtual scale by stride-sampling H1.

    Causality: at H1 timestamp t, the scale-K signal uses H1 bars
    [..., t-2K, t-K, t] (every Kth bar). MA_5 and MA_20 use the LAST
    5 and 20 of those stride-sampled bars. Forward-fill back to H1.
    """
    if scale == 1:
        sub = close_h1
    else:
        # Stride-sample: take every Kth bar.
        sub = close_h1.iloc[::scale]
    # Compute signal on the sub-series.
    rsi_dev = (_rsi(sub, 14) - 50.0).abs() / 50.0
    ma5 = sub.rolling(5, min_periods=5).mean()
    ma20 = sub.rolling(20, min_periods=20).mean()
    ma_sign = np.sign(ma5 - ma20)
    sig = (ma_sign * rsi_dev).fillna(0.0)
    # Reindex back to H1 index with ffill.
    return sig.reindex(close_h1.index, method="ffill").fillna(0.0)


def gld_confluence_returns(
    bars: pd.DataFrame, *, threshold: float = 0.75, apply_costs: bool = True
) -> pd.Series:
    """Per-bar net return of the AND-gated multi-scale confluence."""
    close = bars["close"]
    # Per-scale signals.
    scale_sigs = {tf: _scale_signal(close, k) for tf, k in SCALES.items()}
    # AND-gate: all 4 scales must have same sign (and non-zero).
    signs = pd.DataFrame({tf: np.sign(s) for tf, s in scale_sigs.items()})
    agree = (signs.eq(signs.iloc[:, 0], axis=0)).all(axis=1) & (signs.iloc[:, 0] != 0)
    # Composite signed score: weighted mean of magnitudes * agreed sign.
    weighted = pd.DataFrame({tf: WEIGHTS[tf] * scale_sigs[tf] for tf in SCALES})
    composite = weighted.sum(axis=1)
    composite = composite.where(agree, 0.0)
    # Z-score using expanding window capped at zscore_window.
    rolling_mu = composite.rolling(ZSCORE_WINDOW, min_periods=200).mean()
    rolling_sigma = composite.rolling(ZSCORE_WINDOW, min_periods=200).std()
    z = (composite - rolling_mu) / rolling_sigma.replace(0.0, np.nan)
    # Position from z crosses.
    arr_z = z.to_numpy()
    pos = np.zeros(len(z), dtype=float)
    state = 0
    for i in range(len(arr_z)):
        v = arr_z[i]
        if np.isnan(v):
            pos[i] = float(state)
            continue
        if state == 0:
            if v > threshold:
                state = 1
            elif v < -threshold:
                state = -1
        else:
            if state == 1 and v <= 0:
                state = 0
            elif state == -1 and v >= 0:
                state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=z.index)
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    if apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def assert_causal_gld_confluence(bars: pd.DataFrame) -> None:
    """L21 smoke: corrupting future bars must not change past returns."""
    base = gld_confluence_returns(bars, threshold=0.75)
    corrupted = bars.copy()
    n = len(corrupted)
    cutoff = n - 50
    for col in ("open", "high", "low", "close"):
        corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] = (
            corrupted.iloc[cutoff:, corrupted.columns.get_loc(col)] * 100.0
        )
    perturbed = gld_confluence_returns(corrupted, threshold=0.75)
    safe_end = cutoff - 200  # buffer for weekly virtual TF
    diff = (base.iloc[:safe_end] - perturbed.iloc[:safe_end]).abs().max()
    if diff > 1e-12:
        raise AssertionError(f"L21 fail: max diff {diff:.2e}")
    print(f"[gld_conf-sweep] L21 causality smoke PASS (max diff = {diff:.2e})")


def main() -> None:
    bars = _load_h1()
    print(f"GLD H1: {len(bars)} bars, range {bars.index[0]} -> {bars.index[-1]}")

    assert_causal_gld_confluence(bars)

    # Sanctuary split: last 12 months.
    cutoff = bars.index[-1] - pd.DateOffset(months=12)
    is_bars = bars.loc[:cutoff]
    sanctuary_bars = bars.loc[cutoff:]
    print(f"IS: {len(is_bars)} bars; sanctuary: {len(sanctuary_bars)} bars")

    print(f"\nthreshold sweep (periods_per_year={PERIODS_PER_YEAR_H1_EQ}, RTH H1)")
    print(f"{'threshold':>10s} {'Sharpe':>10s} {'CI_lo':>10s} {'CI_hi':>10s} {'MaxDD':>10s}")
    for thr in [0.50, 0.75, 1.00, 1.25]:
        ret_is = gld_confluence_returns(is_bars, threshold=thr)
        sr = float(sharpe(ret_is, periods_per_year=PERIODS_PER_YEAR_H1_EQ))
        ci_lo, ci_hi = bootstrap_sharpe_ci(ret_is, periods_per_year=PERIODS_PER_YEAR_H1_EQ, seed=42)
        mdd = float(max_drawdown(ret_is))
        print(f"{thr:>10.2f} {sr:+10.4f} {ci_lo:+10.3f} {ci_hi:+10.3f} {mdd * 100:+10.1f}%")

    # Sanctuary check at live threshold.
    print("\nSanctuary at live threshold 0.75:")
    ret_sanc = gld_confluence_returns(sanctuary_bars, threshold=0.75)
    sr_s = float(sharpe(ret_sanc, periods_per_year=PERIODS_PER_YEAR_H1_EQ))
    print(f"  Sharpe = {sr_s:+.4f}, n_bars = {len(ret_sanc)}")


if __name__ == "__main__":
    main()
