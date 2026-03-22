"""Daily — Systematic Time Series Analysis.

Applies four complementary techniques to characterise the statistical nature
of any daily close price series:

  A. Seasonal Decomposition  — trend / seasonality / residual breakdown
  B. ACF / PACF              — autocorrelation structure of log returns
  C. Hurst Exponent (R/S)    — trending vs mean-reverting vs random walk
  D. ADF + KPSS              — stationarity tests on levels and returns

Usage
-----
  uv run python research/spy_ts_analysis.py [--symbol SYMBOL_TF]

  --symbol   Parquet stem in data/ without extension, e.g. SPY_D (default) or GBP_USD_D

Output
------
  .tmp/<symbol>_ts_analysis.png   — 6-panel figure
  stdout                          — key statistics summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_daily(symbol: str) -> pd.Series:
    path = ROOT / "data" / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            raise ValueError("Cannot resolve timestamp index")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    close = df["close"].dropna()
    print(f"{symbol} loaded: {len(close)} bars  [{close.index[0].date()} to {close.index[-1].date()}]")
    return close


# ---------------------------------------------------------------------------
# C. Hurst Exponent via Rescaled Range (R/S) analysis — numba JIT
# ---------------------------------------------------------------------------


@njit(cache=True)
def _hurst_rs_core(ts: np.ndarray, lags: np.ndarray) -> float:
    """JIT-compiled R/S core. lags must be int64 array."""
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
            # mean
            m = 0.0
            for k in range(lag):
                m += ts[base + k]
            m /= lag
            # range of cumulative deviations
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
            # std (ddof=1)
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
    # OLS slope (polyfit degree 1) — manual to stay in numba
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


@njit(cache=True)
def _rolling_hurst_nb(series: np.ndarray, lags: np.ndarray, window: int, step: int) -> np.ndarray:
    """JIT-compiled rolling Hurst — called once per symbol after warmup."""
    n = len(series)
    values = np.full(n, np.nan)
    i = window
    while i < n:
        values[i] = _hurst_rs_core(series[i - window: i], lags)
        i += step
    return values


def hurst_rs(ts: np.ndarray, min_lag: int = 10, max_lag: int | None = None) -> float:
    n = len(ts)
    if max_lag is None:
        max_lag = n // 2
    if max_lag < min_lag:
        return float("nan")
    lags = np.unique(np.geomspace(min_lag, max_lag, 20).astype(np.int64))
    return float(_hurst_rs_core(ts.astype(np.float64), lags))


def rolling_hurst(series: np.ndarray, window: int = 252, step: int = 21) -> np.ndarray:
    lags = np.unique(np.geomspace(10, window // 2, 20).astype(np.int64))
    return _rolling_hurst_nb(series.astype(np.float64), lags, window, step)


# ---------------------------------------------------------------------------
# D. Stationarity tests
# ---------------------------------------------------------------------------


def _adf_summary(series: pd.Series, label: str) -> dict:
    result = adfuller(series, autolag="AIC")
    adf_stat, p_val, used_lags, nobs, crit_vals = result[:5]
    verdict = "STATIONARY (reject H0)" if p_val < 0.05 else "NON-STATIONARY (fail to reject H0)"
    print(f"\n  ADF [{label}]")
    print(f"    Stat={adf_stat:.4f}  p={p_val:.4f}  lags={used_lags}  n={nobs}")
    print(f"    Critical values: {crit_vals}")
    print(f"    Verdict: {verdict}")
    return {"label": label, "stat": adf_stat, "p": p_val, "verdict": verdict}


def _kpss_summary(series: pd.Series, label: str) -> dict:
    stat, p_val, lags, crit_vals = kpss(series, regression="c", nlags="auto")
    verdict = "STATIONARY (fail to reject H0)" if p_val > 0.05 else "NON-STATIONARY (reject H0)"
    print(f"\n  KPSS [{label}]")
    print(f"    Stat={stat:.4f}  p~{p_val:.4f}  lags={lags}")
    print(f"    Verdict: {verdict}")
    return {"label": label, "stat": stat, "p": p_val, "verdict": verdict}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY_D", help="Parquet stem in data/, e.g. SPY_D or GBP_USD_D")
    parser.add_argument(
        "--period", type=int, default=252,
        help="Seasonal decomposition period (default 252=annual daily; use 120 for H1 weekly cycle)",
    )
    args = parser.parse_args()
    symbol = args.symbol
    period = args.period

    close = _load_daily(symbol)
    log_ret = np.log(close / close.shift(1)).dropna()

    # -----------------------------------------------------------------------
    # A. Seasonal Decomposition
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"A. SEASONAL DECOMPOSITION  (multiplicative, period={period})")
    print("=" * 60)
    decomp = seasonal_decompose(close, model="multiplicative", period=period, extrapolate_trend="freq")
    seasonal_amplitude = (decomp.seasonal.max() - decomp.seasonal.min())
    print(f"  Seasonal amplitude (max-min): {seasonal_amplitude:.4f}")
    print(f"  Trend range: [{decomp.trend.min():.2f}, {decomp.trend.max():.2f}]")
    residual_std = decomp.resid.dropna().std()
    print(f"  Residual std: {residual_std:.4f}")

    # -----------------------------------------------------------------------
    # B. ACF / PACF on log returns
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("B. AUTOCORRELATION (log returns)")
    print("=" * 60)
    from statsmodels.tsa.stattools import acf as _acf
    acf_vals = _acf(log_ret, nlags=40, fft=True)
    lag1_acf = acf_vals[1]
    lb = acorr_ljungbox(log_ret, lags=[10, 20], return_df=True)
    print(f"  Lag-1 ACF: {lag1_acf:.4f}")
    nature = "TRENDING (momentum)" if lag1_acf > 0.02 else (
        "MEAN-REVERTING (choppiness)" if lag1_acf < -0.02 else "NEAR RANDOM WALK"
    )
    print(f"  Nature implied by lag-1: {nature}")
    print(f"  Ljung-Box Q (lag 10): stat={lb['lb_stat'].iloc[0]:.3f}  p={lb['lb_pvalue'].iloc[0]:.4f}")
    print(f"  Ljung-Box Q (lag 20): stat={lb['lb_stat'].iloc[1]:.3f}  p={lb['lb_pvalue'].iloc[1]:.4f}")

    # -----------------------------------------------------------------------
    # C. Hurst Exponent
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("C. HURST EXPONENT  (R/S analysis)")
    print("=" * 60)
    h_full = hurst_rs(log_ret.values)
    if h_full < 0.45:
        h_label = "MEAN-REVERTING (H < 0.45)"
    elif h_full > 0.55:
        h_label = "TRENDING / PERSISTENT (H > 0.55)"
    else:
        h_label = "RANDOM WALK (0.45 <= H <= 0.55)"
    print(f"  Full-series H: {h_full:.4f}  ->  {h_label}")

    rolling_h = rolling_hurst(log_ret.values, window=252, step=21)
    valid_h = rolling_h[~np.isnan(rolling_h)]
    if len(valid_h):
        print(f"  Rolling H (252-bar, step 21): mean={valid_h.mean():.3f}  "
              f"min={valid_h.min():.3f}  max={valid_h.max():.3f}")

    # -----------------------------------------------------------------------
    # D. ADF + KPSS stationarity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("D. STATIONARITY TESTS")
    print("=" * 60)
    adf_close = _adf_summary(close, "Close (levels)")
    adf_ret = _adf_summary(log_ret, "Log returns")
    kpss_close = _kpss_summary(close, "Close (levels)")
    kpss_ret = _kpss_summary(log_ret, "Log returns")

    # -----------------------------------------------------------------------
    # Figure: 6-panel (3×2)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{symbol} — Systematic Time Series Analysis", fontsize=15, fontweight="bold")

    # Panel 1: Seasonal trend
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(decomp.trend.index, decomp.trend.values, color="steelblue", linewidth=1)
    ax1.set_title("A. Seasonal Trend Component")
    ax1.set_ylabel("Price ($)")
    ax1.grid(alpha=0.3)

    # Panel 2: Seasonal component (one period)
    ax2 = fig.add_subplot(3, 2, 2)
    one_period = decomp.seasonal.iloc[:period]
    ax2.plot(range(len(one_period)), one_period.values, color="darkorange", linewidth=1.2)
    ax2.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_title(f"A. Seasonal Pattern (period={period}, amp={seasonal_amplitude:.4f})")
    ax2.set_ylabel("Multiplicative Factor")
    ax2.set_xlabel(f"Bar within period")
    ax2.grid(alpha=0.3)

    # Panel 3: ACF of log returns
    ax3 = fig.add_subplot(3, 2, 3)
    plot_acf(log_ret, lags=40, ax=ax3, color="steelblue", zero=False)
    ax3.set_title(f"B. ACF — Log Returns  (lag-1={lag1_acf:.4f})")
    ax3.set_xlabel("Lag (days)")
    ax3.grid(alpha=0.3)

    # Panel 4: PACF of log returns
    ax4 = fig.add_subplot(3, 2, 4)
    plot_pacf(log_ret, lags=40, ax=ax4, method="ywm", color="darkorange", zero=False)
    ax4.set_title("B. PACF — Log Returns")
    ax4.set_xlabel("Lag (days)")
    ax4.grid(alpha=0.3)

    # Panel 5: Rolling Hurst
    ax5 = fig.add_subplot(3, 2, 5)
    h_series = pd.Series(rolling_h, index=log_ret.index)
    h_series_valid = h_series.dropna()
    ax5.plot(h_series_valid.index, h_series_valid.values, color="purple", linewidth=0.9)
    ax5.axhline(0.5, color="black", linewidth=1.0, linestyle="--", label="H=0.5 (random walk)")
    ax5.axhline(0.55, color="green", linewidth=0.7, linestyle=":", label="H=0.55 (trending)")
    ax5.axhline(0.45, color="red", linewidth=0.7, linestyle=":", label="H=0.45 (mean-rev)")
    ax5.fill_between(h_series_valid.index, 0.45, 0.55, alpha=0.08, color="gray")
    ax5.set_title(f"C. Rolling Hurst Exponent (252-bar)  full-H={h_full:.3f}")
    ax5.set_ylabel("H")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # Panel 6: ADF/KPSS summary table
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis("off")
    table_data = [
        ["Test", "Series", "Stat", "p-value", "Verdict"],
        ["ADF", "Close", f"{adf_close['stat']:.3f}", f"{adf_close['p']:.4f}",
         "Non-Stat." if "NON" in adf_close["verdict"] else "Stationary"],
        ["ADF", "Returns", f"{adf_ret['stat']:.3f}", f"{adf_ret['p']:.4f}",
         "Non-Stat." if "NON" in adf_ret["verdict"] else "Stationary"],
        ["KPSS", "Close", f"{kpss_close['stat']:.3f}", f"{kpss_close['p']:.4f}",
         "Non-Stat." if "NON" in kpss_close["verdict"] else "Stationary"],
        ["KPSS", "Returns", f"{kpss_ret['stat']:.3f}", f"{kpss_ret['p']:.4f}",
         "Non-Stat." if "NON" in kpss_ret["verdict"] else "Stationary"],
        ["", "", "", "", ""],
        ["Lag-1 ACF", f"{lag1_acf:.4f}", "", "", nature[:20]],
        ["Full Hurst", f"{h_full:.4f}", "", "", h_label[:20]],
    ]
    tbl = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    ax6.set_title("D. Stationarity + Summary", pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = TMP / f"{symbol}_ts_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")


if __name__ == "__main__":
    main()
