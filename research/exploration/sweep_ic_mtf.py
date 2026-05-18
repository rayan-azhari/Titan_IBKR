"""ic_mtf signal-layer L21 causality smoke + sweep (Wave B).

**EXPLORATORY** (L52/L58). V1 claim: OOS Sharpe 7.71-8.28 across 6 FX pairs.
For FX H1 strategies that's 5-10x physically plausible — same red flag as
mtf (Wave A.5, L21 BUG CONFIRMED). This script tests whether the IC MTF
edge survives strict-causality multi-TF alignment and IS-only calibration.

Strategy (per live config + strategy.py):
    - Per TF in {W, D, H4, H1}, two signals: accel_rsi14 + accel_stoch_k.
    - Each (signal, TF) pair gets an IC SIGN fit (Spearman corr vs forward
      H1 returns) and is multiplied by sign before averaging.
    - Composite = equal-weight mean of all (signal, TF) sign-oriented values
      aligned to H1 (ffill from higher TF).
    - Z-score normalised with frozen IS mu/sigma.
    - Entry: |z| > threshold (0.75 for EUR/USD per live config).
    - Bar Sharpe at periods_per_year = 6048 (FX 24/7 H1).

L21 risk vectors:
    1. IC sign fit: V1 likely used FULL backtest series (look-ahead).
       Fix: IS-only fit (first half).
    2. Higher-TF ffill: aligned[t] uses TF[t] which may have closed AFTER
       H1 bar t in real time. Fix: shift higher-TF series by 1 bar of its
       own TF before alignment (so H4 signal at H1 timestamp t uses H4
       bar that closed strictly before t).
    3. Z-score normalisation: V1 likely used FULL series mu/sigma.
       Fix: IS-only mu/sigma frozen.

Pre-reg context: Roster says causality FIRST per L21 risk.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_ic_mtf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.metrics import bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "sweep_ic_mtf"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# FX H1 24/7 cadence (correct for FX, unlike US equity case in L60).
PERIODS_PER_YEAR_H1_FX = 6048
TFS = ("W", "D", "H4", "H1")
SIGNALS = ("accel_rsi14", "accel_stoch_k")
PAIRS_TO_TEST = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "AUD_JPY", "USD_CHF"]
LIVE_THRESHOLDS = {  # per live config
    "EUR_USD": 0.75,
    "GBP_USD": 1.00,
    "USD_JPY": 0.75,
    "AUD_USD": 1.00,
    "AUD_JPY": 0.75,
    "USD_CHF": 1.00,
}
COST_BPS_PER_TURNOVER = 0.5  # FX spot, conservative


def _load_tf(pair: str, tf: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{pair}_{tf}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    df = pd.read_parquet(fp)
    # FX parquets store timestamp as a column with integer RangeIndex.
    # Promote timestamp column to DatetimeIndex; tz-naive UTC.
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


def _stoch_k(df: pd.DataFrame, k_period: int = 14) -> pd.Series:
    high_n = df["high"].rolling(k_period, min_periods=k_period).max()
    low_n = df["low"].rolling(k_period, min_periods=k_period).min()
    return 100.0 * (df["close"] - low_n) / (high_n - low_n).replace(0.0, np.nan)


def _accel_rsi14(close: pd.Series) -> pd.Series:
    return (_rsi(close, 14) - 50.0).diff(1)


def _accel_stoch_k(df: pd.DataFrame) -> pd.Series:
    k = _stoch_k(df, 14)
    return (k - 50.0).diff(1)


def _build_signal_at_h1(
    tf_data: dict[str, pd.DataFrame],
    h1_index: pd.DatetimeIndex,
    *,
    causal: bool,
) -> pd.DataFrame:
    """Build per (signal,TF) oriented signal aligned to H1 index.

    If causal=True: shift each higher-TF signal by 1 bar of its own TF before
    aligning to H1 (so H1[t] sees only completed higher-TF bars).
    If causal=False: V1-style alignment using contemporaneous higher-TF bars.
    """
    parts: dict[str, pd.Series] = {}
    for sig in SIGNALS:
        for tf in TFS:
            if tf not in tf_data:
                continue
            df = tf_data[tf]
            close = df["close"]
            if sig == "accel_rsi14":
                raw = _accel_rsi14(close)
            else:
                raw = _accel_stoch_k(df)
            # Causal shift: higher-TF signal at TF index t is only available
            # AFTER TF bar t closes. To use at H1 timestamps strictly after
            # the TF close, shift by 1 bar of the TF.
            if causal and tf != "H1":
                raw = raw.shift(1)
            elif causal and tf == "H1":
                # H1 itself: signal at H1[t] uses bar t close (known at EOD bar t).
                # Position formed at H1[t] earns t->t+1 return via downstream shift.
                pass
            # Align to H1 index by ffill.
            aligned = raw.reindex(h1_index, method="ffill")
            parts[f"{sig}_{tf}"] = aligned
    if not parts:
        return pd.DataFrame(index=h1_index)
    return pd.DataFrame(parts)


def _fit_ic_signs(
    parts_df: pd.DataFrame, h1_close: pd.Series, *, slice_until: int | None = None
) -> dict[str, float]:
    """Fit Spearman IC sign per (signal,TF) on a SLICE of the data.

    slice_until=None means fit on full series (V1-style look-ahead).
    slice_until=N means fit on first N bars (IS-only).
    """
    fwd_h1 = np.log(h1_close.shift(-1) / h1_close)
    signs: dict[str, float] = {}
    sub = parts_df if slice_until is None else parts_df.iloc[:slice_until]
    sub_fwd = fwd_h1 if slice_until is None else fwd_h1.iloc[:slice_until]
    for col in parts_df.columns:
        s = sub[col]
        both = pd.concat([s, sub_fwd], axis=1).dropna()
        if len(both) < 30:
            signs[col] = 1.0
            continue
        r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
        signs[col] = float(np.sign(r)) if not np.isnan(r) and r != 0.0 else 1.0
    return signs


def _compute_composite_z(
    parts_df: pd.DataFrame,
    signs: dict[str, float],
    *,
    mu_sigma_slice_until: int | None,
) -> pd.Series:
    """Compute composite z-score with optional IS-only mu/sigma."""
    oriented = parts_df * pd.Series(signs)
    composite = oriented.mean(axis=1)
    sub = composite if mu_sigma_slice_until is None else composite.iloc[:mu_sigma_slice_until]
    mu = float(sub.mean())
    sigma = float(sub.std()) or 1.0
    return (composite - mu) / sigma


def _signal_to_position(z: pd.Series, threshold: float) -> pd.Series:
    """Build long/short binary position from z-score crosses."""
    pos = np.zeros(len(z), dtype=float)
    state = 0
    arr = z.to_numpy()
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            pos[i] = float(state)
            continue
        if state == 0:
            if arr[i] > threshold:
                state = 1
            elif arr[i] < -threshold:
                state = -1
        else:
            # Exit on zero cross.
            if state == 1 and arr[i] <= 0:
                state = 0
            elif state == -1 and arr[i] >= 0:
                state = 0
        pos[i] = float(state)
    return pd.Series(pos, index=z.index)


def _strategy_returns(position: pd.Series, h1_close: pd.Series) -> pd.Series:
    log_ret = np.log(h1_close / h1_close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (COST_BPS_PER_TURNOVER / 10_000.0)
    return (gross - cost).rename("ret")


def run_pair(pair: str) -> dict[str, dict]:
    """Run V1-style (look-ahead-ok) vs strict-causal variants on one pair."""
    threshold = LIVE_THRESHOLDS.get(pair, 0.75)
    tf_data: dict[str, pd.DataFrame] = {}
    for tf in TFS:
        try:
            tf_data[tf] = _load_tf(pair, tf)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
    if "H1" not in tf_data:
        return {}
    h1_close = tf_data["H1"]["close"]
    h1_index = h1_close.index

    n_total = len(h1_index)
    is_split = n_total // 2  # first half = IS; sign + mu/sigma frozen here

    # Variant A: V1-style — full-series IC sign + full-series mu/sigma, ffill
    # WITHOUT shift (look-ahead-suspect). This SHOULD give Sharpe close to
    # V1's claimed 7-8.
    parts_v1 = _build_signal_at_h1(tf_data, h1_index, causal=False)
    signs_v1 = _fit_ic_signs(parts_v1, h1_close, slice_until=None)
    z_v1 = _compute_composite_z(parts_v1, signs_v1, mu_sigma_slice_until=None)
    pos_v1 = _signal_to_position(z_v1, threshold)
    ret_v1 = _strategy_returns(pos_v1, h1_close)
    sr_v1 = float(sharpe(ret_v1, periods_per_year=PERIODS_PER_YEAR_H1_FX))
    ci_lo_v1, ci_hi_v1 = bootstrap_sharpe_ci(
        ret_v1, periods_per_year=PERIODS_PER_YEAR_H1_FX, seed=42
    )

    # Variant B: causal but with V1-style full-series sign fit.
    parts_c = _build_signal_at_h1(tf_data, h1_index, causal=True)
    signs_c_full = _fit_ic_signs(parts_c, h1_close, slice_until=None)
    z_b = _compute_composite_z(parts_c, signs_c_full, mu_sigma_slice_until=None)
    pos_b = _signal_to_position(z_b, threshold)
    ret_b = _strategy_returns(pos_b, h1_close)
    sr_b = float(sharpe(ret_b, periods_per_year=PERIODS_PER_YEAR_H1_FX))
    ci_lo_b, ci_hi_b = bootstrap_sharpe_ci(ret_b, periods_per_year=PERIODS_PER_YEAR_H1_FX, seed=42)

    # Variant C: fully strict — causal alignment + IS-only sign + IS-only z.
    signs_c_is = _fit_ic_signs(parts_c, h1_close, slice_until=is_split)
    z_c = _compute_composite_z(parts_c, signs_c_is, mu_sigma_slice_until=is_split)
    pos_c = _signal_to_position(z_c, threshold)
    ret_c = _strategy_returns(pos_c, h1_close)
    # OOS slice only (second half).
    ret_c_oos = ret_c.iloc[is_split:]
    sr_c = float(sharpe(ret_c_oos, periods_per_year=PERIODS_PER_YEAR_H1_FX))
    ci_lo_c, ci_hi_c = bootstrap_sharpe_ci(
        ret_c_oos, periods_per_year=PERIODS_PER_YEAR_H1_FX, seed=42
    )

    return {
        "pair": pair,
        "threshold": threshold,
        "n_h1_bars": n_total,
        "is_split": is_split,
        "V1_style_look_ahead": {
            "sharpe": sr_v1,
            "ci_lo": ci_lo_v1,
            "ci_hi": ci_hi_v1,
        },
        "causal_full_sign_fit": {
            "sharpe": sr_b,
            "ci_lo": ci_lo_b,
            "ci_hi": ci_hi_b,
        },
        "strict_causal_IS_only_OOS": {
            "sharpe": sr_c,
            "ci_lo": ci_lo_c,
            "ci_hi": ci_hi_c,
            "n_oos": len(ret_c_oos),
        },
    }


def main() -> None:
    print("=" * 72)
    print("ic_mtf L21 causality smoke + signal-layer sweep")
    print("=" * 72)

    rows = []
    for pair in PAIRS_TO_TEST:
        try:
            print(f"\n[{pair}] running 3 variants...")
            res = run_pair(pair)
            if not res:
                print("  skipped (data missing)")
                continue
            print(
                f"  n_bars={res['n_h1_bars']}, is_split={res['is_split']}, "
                f"threshold={res['threshold']}"
            )
            print(
                f"  V1_style (look-ahead OK)  : Sharpe={res['V1_style_look_ahead']['sharpe']:+.4f}  "
                f"CI=[{res['V1_style_look_ahead']['ci_lo']:+.3f}, {res['V1_style_look_ahead']['ci_hi']:+.3f}]"
            )
            print(
                f"  causal align, full-sign  : Sharpe={res['causal_full_sign_fit']['sharpe']:+.4f}  "
                f"CI=[{res['causal_full_sign_fit']['ci_lo']:+.3f}, {res['causal_full_sign_fit']['ci_hi']:+.3f}]"
            )
            print(
                f"  STRICT causal+IS OOS     : Sharpe={res['strict_causal_IS_only_OOS']['sharpe']:+.4f}  "
                f"CI=[{res['strict_causal_IS_only_OOS']['ci_lo']:+.3f}, "
                f"{res['strict_causal_IS_only_OOS']['ci_hi']:+.3f}]  "
                f"(n_oos={res['strict_causal_IS_only_OOS']['n_oos']})"
            )
            rows.append(res)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [error] {e}")

    # CSV summary.
    lines = [
        "pair,threshold,n_bars,V1_style_sharpe,V1_style_ci_lo,causal_full_sharpe,causal_full_ci_lo,strict_OOS_sharpe,strict_OOS_ci_lo"
    ]
    for r in rows:
        lines.append(
            f"{r['pair']},{r['threshold']:.2f},{r['n_h1_bars']},"
            f"{r['V1_style_look_ahead']['sharpe']:.4f},{r['V1_style_look_ahead']['ci_lo']:.3f},"
            f"{r['causal_full_sign_fit']['sharpe']:.4f},{r['causal_full_sign_fit']['ci_lo']:.3f},"
            f"{r['strict_causal_IS_only_OOS']['sharpe']:.4f},{r['strict_causal_IS_only_OOS']['ci_lo']:.3f}"
        )
    csv_path = REPORTS_DIR / "triage_summary.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ic_mtf-sweep] wrote: {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
