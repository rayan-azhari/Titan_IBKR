"""Batch Time Series Analysis — all daily equity parquet files.

Runs the four systematic TS tests (Seasonal amplitude, Lag-1 ACF,
Hurst exponent, ADF stationarity) across every *_D.parquet in data/,
then writes a ranked summary to .tmp/ts_analysis_batch.csv.

Usage
-----
  uv run python research/ts_analysis_batch.py [--workers N] [--min-bars N]

  --workers   Parallel workers (default: 4)
  --min-bars  Skip symbols with fewer bars (default: 504 = 2 years)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf as _acf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Hurst R/S — numba JIT
# ---------------------------------------------------------------------------


@njit(cache=True)
def _hurst_rs_core(ts: np.ndarray, lags: np.ndarray) -> float:
    n = len(ts)
    n_lags = len(lags)
    log_lags = np.empty(n_lags)
    log_rs = np.empty(n_lags)
    count = 0
    for li in range(n_lags):
        lag = lags[li]
        n_chunks = n // lag
        if n_chunks == 0:
            continue
        rs_sum = 0.0
        rs_count = 0
        for ci in range(n_chunks):
            base = ci * lag
            m = 0.0
            for k in range(lag):
                m += ts[base + k]
            m /= lag
            dev = 0.0
            cum_min = 0.0
            cum_max = 0.0
            for k in range(lag):
                dev += ts[base + k] - m
                if dev < cum_min:
                    cum_min = dev
                if dev > cum_max:
                    cum_max = dev
            r = cum_max - cum_min
            var = 0.0
            for k in range(lag):
                d = ts[base + k] - m
                var += d * d
            s = (var / (lag - 1)) ** 0.5
            if s > 1e-10 and r > 0.0:
                rs_sum += r / s
                rs_count += 1
        if rs_count > 0:
            log_lags[count] = np.log(float(lag))
            log_rs[count] = np.log(rs_sum / rs_count)
            count += 1
    if count < 2:
        return np.nan
    xl = log_lags[:count]
    yl = log_rs[:count]
    n_pts = float(count)
    sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
    for i in range(count):
        sx += xl[i]; sy += yl[i]
        sxx += xl[i] * xl[i]; sxy += xl[i] * yl[i]
    denom = n_pts * sxx - sx * sx
    if denom == 0.0:
        return np.nan
    return (n_pts * sxy - sx * sy) / denom


def hurst_rs(ts: np.ndarray, min_lag: int = 10, max_lag: int | None = None) -> float:
    n = len(ts)
    if max_lag is None:
        max_lag = n // 2
    if max_lag < min_lag:
        return float("nan")
    lags = np.unique(np.geomspace(min_lag, max_lag, 20).astype(np.int64))
    return float(_hurst_rs_core(ts.astype(np.float64), lags))


# ---------------------------------------------------------------------------
# Per-symbol computation
# ---------------------------------------------------------------------------


def _analyse(path: Path, min_bars: int) -> dict | None:
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index, utc=True)
            else:
                return None
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            return None
        close = df["close"].dropna().sort_index()
        if len(close) < min_bars:
            return None

        log_ret = np.log(close / close.shift(1)).dropna()
        n_bars = len(close)

        # A. Seasonal amplitude
        try:
            decomp = seasonal_decompose(
                close, model="multiplicative", period=252, extrapolate_trend="freq"
            )
            seasonal_amp = float(decomp.seasonal.max() - decomp.seasonal.min())
            residual_std = float(decomp.resid.dropna().std())
        except Exception:
            seasonal_amp = float("nan")
            residual_std = float("nan")

        # B. ACF lag-1 + Ljung-Box
        acf_vals = _acf(log_ret, nlags=10, fft=True)
        lag1_acf = float(acf_vals[1])
        try:
            lb = acorr_ljungbox(log_ret, lags=[10], return_df=True)
            lb_p10 = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_p10 = float("nan")

        # C. Hurst exponent
        h = hurst_rs(log_ret.values)

        # D. ADF on close prices (Testing price stationarity)
        try:
            adf_stat, adf_p, *_ = adfuller(close, autolag="AIC")
            adf_ret_p = float(adf_p)
        except Exception:
            adf_ret_p = float("nan")

        # Character label
        if lag1_acf < -0.02:
            char = "mean-rev"
        elif lag1_acf > 0.02:
            char = "trending"
        else:
            char = "rand-walk"

        if h < 0.45:
            hurst_label = "mean-rev"
        elif h > 0.55:
            hurst_label = "trending"
        else:
            hurst_label = "rand-walk"

        return {
            "symbol": path.stem,
            "n_bars": n_bars,
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
            "lag1_acf": round(lag1_acf, 4),
            "acf_char": char,
            "lb_p10": round(lb_p10, 4),
            "hurst_h": round(h, 4),
            "hurst_char": hurst_label,
            "seasonal_amp": round(seasonal_amp, 4),
            "residual_std": round(residual_std, 4),
            "adf_ret_p": round(adf_ret_p, 4),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-bars", type=int, default=504, dest="min_bars")
    args = parser.parse_args()

    paths = sorted((ROOT / "data").glob("*_D.parquet"))
    print(f"Found {len(paths)} daily parquet files. Running with {args.workers} workers...")

    results = Parallel(n_jobs=args.workers, verbose=5)(
        delayed(_analyse)(p, args.min_bars) for p in paths
    )

    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)

    # Sort by Hurst H descending (most trending first), then lag1_acf
    df = df.sort_values("hurst_h", ascending=False).reset_index(drop=True)

    out_csv = TMP / "ts_analysis_batch.csv"
    df.to_csv(out_csv, index=False)

    # --- Console summary ---
    total = len(df)
    mean_rev_acf = (df["acf_char"] == "mean-rev").sum()
    trending_acf = (df["acf_char"] == "trending").sum()
    rw_acf = (df["acf_char"] == "rand-walk").sum()
    trending_h = (df["hurst_char"] == "trending").sum()
    mean_rev_h = (df["hurst_char"] == "mean-rev").sum()
    rw_h = (df["hurst_char"] == "rand-walk").sum()

    print(f"\n{'='*70}")
    print(f"BATCH RESULTS  ({total} symbols, min {args.min_bars} bars)")
    print(f"{'='*70}")
    print("\nLag-1 ACF character:")
    print(f"  Mean-reverting : {mean_rev_acf:4d}  ({100*mean_rev_acf/total:.1f}%)")
    print(f"  Trending       : {trending_acf:4d}  ({100*trending_acf/total:.1f}%)")
    print(f"  Random walk    : {rw_acf:4d}  ({100*rw_acf/total:.1f}%)")
    print("\nHurst exponent character:")
    print(f"  Trending (H>0.55)   : {trending_h:4d}  ({100*trending_h/total:.1f}%)")
    print(f"  Random walk         : {rw_h:4d}  ({100*rw_h/total:.1f}%)")
    print(f"  Mean-rev (H<0.45)   : {mean_rev_h:4d}  ({100*mean_rev_h/total:.1f}%)")
    print(f"\nHurst H stats:  mean={df['hurst_h'].mean():.3f}  "
          f"median={df['hurst_h'].median():.3f}  "
          f"std={df['hurst_h'].std():.3f}")
    print(f"Lag-1 ACF stats: mean={df['lag1_acf'].mean():.4f}  "
          f"median={df['lag1_acf'].median():.4f}  "
          f"std={df['lag1_acf'].std():.4f}")

    # Top 20 most trending by Hurst
    print("\n--- Top 20 most TRENDING (Hurst H) ---")
    top_h = df.nlargest(20, "hurst_h")[["symbol", "n_bars", "hurst_h", "lag1_acf", "seasonal_amp"]]
    print(top_h.to_string(index=False))

    # Top 20 strongest mean-reversion (most negative lag-1 ACF)
    print("\n--- Top 20 strongest MEAN-REVERSION (lag-1 ACF) ---")
    top_mr = df.nsmallest(20, "lag1_acf")[["symbol", "n_bars", "lag1_acf", "lb_p10", "hurst_h"]]
    print(top_mr.to_string(index=False))

    # Strongest seasonal amplitude
    print("\n--- Top 20 strongest SEASONAL signal ---")
    top_s = df.nlargest(20, "seasonal_amp")[["symbol", "n_bars", "seasonal_amp", "residual_std", "hurst_h"]]
    print(top_s.to_string(index=False))

    print(f"\nFull results -> {out_csv}")


if __name__ == "__main__":
    main()
