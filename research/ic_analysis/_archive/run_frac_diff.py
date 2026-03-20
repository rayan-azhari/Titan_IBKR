"""run_frac_diff.py -- Fractional Differentiation IC Analysis.

Tests whether Fixed-Width Window Fractional Differentiation (FFD) of
technical signals improves IC/ICIR vs both raw (d=0) and first-difference
(d=1) versions.

Theory (Lopez de Prado, AFML 2018):
  Standard .diff(1) achieves stationarity but destroys long-term memory.
  Fractional differencing at d ≈ 0.3-0.5 achieves stationarity while
  retaining ~70-80% of the series' original memory (measured by correlation
  with raw). The hypothesis: retained memory → higher IC because the signal
  "knows" where price has been over weeks, not just one bar.

For each candidate series this script:
  1. Sweeps d from 0.0 to 1.0 in steps of 0.1
  2. At each d: tests ADF stationarity, computes IC at all horizons
  3. Finds minimum d that achieves ADF p < 0.05 (the optimal d)
  4. Compares IC at d=0 (raw), d=optimal, d=1 (.diff) side by side
  5. Reports memory retained at each d (corr between fd and raw series)

Candidates tested:
  - log(close)         — the raw price signal itself
  - rsi_14_dev         — Group E base: accel_rsi14 = rsi_14_dev.diff(1)
  - stoch_k_dev        — Group E base: accel_stoch_k = stoch_k_dev.diff(1)
  - realized_vol_20    — Group E base: accel_rvol20 = realized_vol_20.diff(1)
  - norm_atr_14        — Group E base: accel_atr = norm_atr_14.diff(1)
  - bb_width           — Group E base: accel_bb_width = bb_width.diff(1)
  - zscore_expanding   — strong unconditional IC on equities, already stationary
  - ma_spread_50_200   — top trend signal; test whether fd of price spread adds edge

Usage:
  uv run python research/ic_analysis/run_frac_diff.py
  uv run python research/ic_analysis/run_frac_diff.py --instrument CAT --timeframe D
  uv run python research/ic_analysis/run_frac_diff.py --instrument TXN --timeframe D --horizon 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr  # noqa: E402

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    print("ERROR: statsmodels not installed. Run: uv add statsmodels")
    sys.exit(1)

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402

REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

HORIZONS   = [1, 5, 10, 20, 60]
D_GRID     = np.round(np.arange(0.0, 1.05, 0.1), 2)
FFD_THRES  = 1e-4    # weight truncation threshold
ADF_PVAL   = 0.05    # stationarity significance level
MIN_BARS   = 100     # minimum valid bars per fd series
ICIR_WIN   = 60      # rolling ICIR window

W = 82


# ---------------------------------------------------------------------------
# FFD (Fixed-Width Window Fractional Differentiation)
# ---------------------------------------------------------------------------

def _get_weights_ffd(d: float, thres: float = FFD_THRES) -> np.ndarray:
    """Compute FFD weights [w_{L-1}, ..., w_1, w_0] (oldest → newest)."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])  # flip: oldest weight first


def frac_diff_ffd(series: pd.Series, d: float, thres: float = FFD_THRES) -> pd.Series:
    """Apply Fixed-Width Window Fractional Differencing.

    For d=0 returns raw series; for d=1 returns series.diff(1).
    For 0 < d < 1 returns the fractionally differenced series with
    NaN for the initial (width-1) bars.
    """
    if d == 0.0:
        return series.copy()
    if d == 1.0:
        return series.diff(1)

    w     = _get_weights_ffd(d, thres)
    width = len(w)
    vals  = series.values.astype(float)
    out   = np.full(len(vals), np.nan)

    for i in range(width - 1, len(vals)):
        window = vals[i - width + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = np.dot(w, window)

    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Stationarity + memory helpers
# ---------------------------------------------------------------------------

def _adf_pvalue(series: pd.Series) -> float:
    """ADF p-value. Returns 1.0 if series is too short or all-NaN."""
    s = series.dropna()
    if len(s) < 20:
        return 1.0
    try:
        result = adfuller(s, maxlag=1, regression="c", autolag=None)
        return float(result[1])
    except Exception:
        return 1.0


def _memory_retained(raw: pd.Series, fd: pd.Series) -> float:
    """Pearson correlation between fd series and raw — proxy for memory kept."""
    both = pd.concat([raw, fd], axis=1).dropna()
    if len(both) < 20:
        return np.nan
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def _find_min_d(series: pd.Series, d_grid: np.ndarray) -> tuple[float, float]:
    """Return (min_d, pvalue) for first d in grid where ADF rejects unit root."""
    for d in d_grid:
        fd = frac_diff_ffd(series, d)
        pv = _adf_pvalue(fd)
        if pv < ADF_PVAL:
            return float(d), float(pv)
    return float(d_grid[-1]), float(_adf_pvalue(frac_diff_ffd(series, d_grid[-1])))


# ---------------------------------------------------------------------------
# IC helpers
# ---------------------------------------------------------------------------

def _ic(signal: pd.Series, fwd: pd.Series) -> float:
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 20:
        return np.nan
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(r) if not np.isnan(r) else np.nan


def _icir(signal: pd.Series, fwd: pd.Series, window: int = ICIR_WIN) -> float:
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < window + 5:
        return np.nan
    s, f = both.iloc[:, 0], both.iloc[:, 1]
    rics = [
        spearmanr(s.iloc[i: i + window], f.iloc[i: i + window])[0]
        for i in range(len(both) - window)
    ]
    rics = [r for r in rics if not np.isnan(r)]
    if not rics:
        return np.nan
    mu  = np.mean(rics)
    std = np.std(rics)
    return float(mu / std) if std > 1e-10 else np.nan


def _best_ic(signal: pd.Series, fwd_dict: dict, horizons: list) -> tuple[float, float, int]:
    """Return (best_abs_ic, icir_at_best, best_h) across all horizons."""
    best_abs = 0.0
    best_ic  = np.nan
    best_h   = horizons[0]
    best_icir = np.nan
    for h in horizons:
        ic = _ic(signal, fwd_dict[h])
        if not np.isnan(ic) and abs(ic) > best_abs:
            best_abs  = abs(ic)
            best_ic   = ic
            best_h    = h
            best_icir = _icir(signal, fwd_dict[h])
    return best_ic, best_icir, best_h


def _verdict(ic: float, icir: float) -> str:
    a_ic = abs(ic)   if not np.isnan(ic)   else 0.0
    a_ir = abs(icir) if not np.isnan(icir) else 0.0
    if a_ic >= 0.05 and a_ir >= 0.5:
        return "STRONG"
    if a_ic >= 0.05:
        return "USABLE"
    if a_ic >= 0.03:
        return "WEAK"
    return "NOISE"


# ---------------------------------------------------------------------------
# D-sweep for one series
# ---------------------------------------------------------------------------

def _d_sweep(
    name: str,
    raw: pd.Series,
    fwd_dict: dict,
    horizons: list,
    target_h: int,
) -> pd.DataFrame:
    """Compute IC at target_h for each d in D_GRID. Also return ADF p-val and memory."""
    rows = []
    fwd = fwd_dict[target_h]
    for d in D_GRID:
        fd   = frac_diff_ffd(raw, d)
        ic   = _ic(fd, fwd)
        pv   = _adf_pvalue(fd)
        mem  = _memory_retained(raw, fd)
        rows.append({
            "signal":    name,
            "d":         d,
            "ic":        ic,
            "adf_pval":  pv,
            "stationary": pv < ADF_PVAL,
            "memory":    mem,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ASCII d-sweep chart
# ---------------------------------------------------------------------------

def _ascii_d_chart(sweep_df: pd.DataFrame, name: str, target_h: int) -> None:
    """Print IC vs d as a horizontal bar chart."""
    print(f"\n  d-sweep: {name} at h={target_h}")
    print(f"  {'d':>5}  {'IC':>8}  {'Stat?':>5}  {'Mem%':>6}  Chart")
    print("  " + "-" * 65)
    max_ic = sweep_df["ic"].abs().max()
    scale  = 30.0 / max_ic if max_ic > 0 else 1.0
    for _, row in sweep_df.iterrows():
        ic  = row["ic"]
        ic_str  = f"{ic:>+8.4f}" if not np.isnan(ic) else "     NaN"
        stat    = "  YES" if row["stationary"] else "   no"
        mem     = f"{row['memory']*100:>5.1f}%" if not np.isnan(row["memory"]) else "  N/A"
        bar_len = int(abs(ic) * scale) if not np.isnan(ic) else 0
        bar     = "+" * bar_len if ic >= 0 else "-" * bar_len
        print(f"  {row['d']:>5.1f}  {ic_str}  {stat}  {mem}  {bar}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_frac_diff(
    instrument: str,
    timeframe: str,
    horizons: list,
    target_h: int,
) -> pd.DataFrame:
    slug = f"{instrument}_{timeframe}".lower()

    df      = _load_ohlcv(instrument, timeframe)
    close   = df["close"]
    signals = build_all_signals(df)
    valid   = signals.notna().any(axis=1) & close.notna()
    signals = signals[valid]
    close   = close[valid]
    n       = len(signals)

    print(f"\n  Bars: {n:,}  |  Horizons: {horizons}  |  Target h: {target_h}")
    print(f"  FFD threshold: {FFD_THRES}  |  ADF p-value gate: {ADF_PVAL}")

    # Forward returns
    fwd_dict = {}
    for h in horizons:
        fwd_dict[h] = np.log(close.shift(-h) / close)

    # Candidate series to test
    candidates: dict[str, pd.Series] = {
        "log_close":        np.log(close),
        "rsi_14_dev":       signals["rsi_14_dev"],
        "stoch_k_dev":      signals["stoch_k_dev"],
        "realized_vol_20":  signals["realized_vol_20"],
        "norm_atr_14":      signals["norm_atr_14"],
        "bb_width":         signals["bb_width"],
        "zscore_expanding": signals["zscore_expanding"],
        "ma_spread_50_200": signals["ma_spread_50_200"],
    }

    # Group E baselines (d=1 equivalents already computed)
    baselines_d1: dict[str, str] = {
        "rsi_14_dev":      "accel_rsi14",
        "stoch_k_dev":     "accel_stoch_k",
        "realized_vol_20": "accel_rvol20",
        "norm_atr_14":     "accel_atr",
        "bb_width":        "accel_bb_width",
    }

    print(f"\n{'='*W}")
    print(f"  FRACTIONAL DIFFERENTIATION IC -- {instrument} {timeframe}  (h={target_h})")
    print(f"{'='*W}")
    print(f"  {'Signal':<22}  {'d_min':>5}  {'IC(d=0)':>9}  {'IC(d_opt)':>10}  "
          f"{'IC(d=1)':>9}  {'Mem%':>6}  {'Winner':>8}  {'Verdict'}")
    print("  " + "-" * (W - 2))

    all_sweeps = []
    summary_rows = []

    for name, raw_series in candidates.items():
        # Find minimum d for stationarity
        min_d, adf_pv = _find_min_d(raw_series, D_GRID)

        # IC at d=0, d=min_d, d=1
        fd_raw  = frac_diff_ffd(raw_series, 0.0)
        fd_opt  = frac_diff_ffd(raw_series, min_d)
        fd_d1   = frac_diff_ffd(raw_series, 1.0)

        ic_d0,  _,    h0  = _best_ic(fd_raw, fwd_dict, horizons)
        ic_opt, icir_opt, h_opt = _best_ic(fd_opt, fwd_dict, horizons)
        ic_d1,  icir_d1,  h1   = _best_ic(fd_d1,  fwd_dict, horizons)

        # Also get ICs specifically at target_h for fair comparison
        ic0_tgt  = _ic(fd_raw, fwd_dict[target_h])
        icopttgt = _ic(fd_opt, fwd_dict[target_h])
        ic1_tgt  = _ic(fd_d1,  fwd_dict[target_h])

        # If we have a Group E baseline, use it directly
        if name in baselines_d1 and baselines_d1[name] in signals.columns:
            accel_series = signals[baselines_d1[name]]
            ic1_tgt = _ic(accel_series, fwd_dict[target_h])

        mem_opt = _memory_retained(raw_series, fd_opt)

        # Determine winner
        ics_tgt = {
            f"d=0":     ic0_tgt  if not np.isnan(ic0_tgt)  else -999,
            f"d={min_d:.1f}": icopttgt if not np.isnan(icopttgt) else -999,
            "d=1":      ic1_tgt  if not np.isnan(ic1_tgt)  else -999,
        }
        winner = max(ics_tgt, key=lambda k: abs(ics_tgt[k]))

        verdict = _verdict(icopttgt, icir_opt) if not np.isnan(icopttgt) else "N/A"
        mem_s   = f"{mem_opt*100:.1f}%" if not np.isnan(mem_opt) else "  N/A"

        ic0_s   = f"{ic0_tgt:>+9.4f}"  if not np.isnan(ic0_tgt)  else "      NaN"
        icopt_s = f"{icopttgt:>+10.4f}" if not np.isnan(icopttgt) else "       NaN"
        ic1_s   = f"{ic1_tgt:>+9.4f}"  if not np.isnan(ic1_tgt)  else "      NaN"

        print(f"  {name:<22}  {min_d:>5.1f}  {ic0_s}  {icopt_s}  "
              f"{ic1_s}  {mem_s:>6}  {winner:>8}  {verdict}")

        summary_rows.append({
            "signal":   name,
            "min_d":    min_d,
            "adf_pval": adf_pv,
            "ic_d0":    ic0_tgt,
            "ic_opt":   icopttgt,
            "ic_d1":    ic1_tgt,
            "icir_opt": icir_opt,
            "memory":   mem_opt,
            "winner":   winner,
            "verdict":  verdict,
        })

        # D-sweep (for CSV and chart)
        sweep = _d_sweep(name, raw_series, fwd_dict, horizons, target_h)
        all_sweeps.append(sweep)

    # -- D-sweep charts for top candidates (those where optimal d beats d=1) ---
    print(f"\n{'='*W}")
    print("  D-SWEEP CHARTS  (candidates where fractional d outperforms d=1)")
    print(f"{'='*W}")

    for i, row in enumerate(summary_rows):
        # Show chart if optimal d is fractional AND beats both extremes
        if (0.0 < row["min_d"] < 1.0
                and not np.isnan(row["ic_opt"])
                and abs(row["ic_opt"]) > max(
                    abs(row["ic_d0"]) if not np.isnan(row["ic_d0"]) else 0,
                    abs(row["ic_d1"]) if not np.isnan(row["ic_d1"]) else 0,
                )):
            _ascii_d_chart(all_sweeps[i], row["signal"], target_h)

    # -- Summary verdict -------------------------------------------------------
    print(f"\n{'='*W}")
    print("  SUMMARY: FRACTIONAL D UPLIFT ANALYSIS")
    print(f"{'='*W}")

    frac_wins = [r for r in summary_rows
                 if 0.0 < r["min_d"] < 1.0
                 and not np.isnan(r["ic_opt"])
                 and "d=" in r["winner"] and r["winner"] not in ("d=0", "d=1")]

    if frac_wins:
        print(f"\n  {len(frac_wins)} signal(s) where fractional d BEATS d=0 and d=1:\n")
        for r in frac_wins:
            uplift_vs_d1 = abs(r["ic_opt"]) - abs(r["ic_d1"]) if not np.isnan(r["ic_d1"]) else np.nan
            uplift_vs_d0 = abs(r["ic_opt"]) - abs(r["ic_d0"]) if not np.isnan(r["ic_d0"]) else np.nan
            print(f"  {r['signal']:<22}  optimal d={r['min_d']:.1f}  "
                  f"IC={r['ic_opt']:>+.4f}  "
                  f"vs d=1 uplift={uplift_vs_d1:>+.4f}  "
                  f"vs d=0 uplift={uplift_vs_d0:>+.4f}  "
                  f"memory={r['memory']*100:.1f}%  verdict={r['verdict']}")
        print(f"\n  RECOMMENDATION: Add fd versions of the above signals as Group H in")
        print(f"  run_signal_sweep.py using frac_diff_ffd(series, d={frac_wins[0]['min_d']:.1f})")
    else:
        print("\n  No candidate achieved higher |IC| at a fractional d than at d=0 or d=1.")
        print("  Standard .diff(1) (Group E) remains optimal for these signals on this dataset.")

    # -- Save ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    sweep_df   = pd.concat(all_sweeps, ignore_index=True)

    out_summary = REPORTS / f"frac_diff_summary_{slug}_h{target_h}.csv"
    out_sweep   = REPORTS / f"frac_diff_sweep_{slug}_h{target_h}.csv"
    summary_df.to_csv(out_summary, index=False)
    sweep_df.to_csv(out_sweep, index=False)
    print(f"\n  Summary saved : {out_summary}")
    print(f"  D-sweep saved : {out_sweep}")

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fractional Differentiation IC Analysis"
    )
    parser.add_argument("--instrument", default="CAT")
    parser.add_argument("--timeframe",  default="D")
    parser.add_argument(
        "--horizons", default="1,5,10,20,60",
        help="Comma-separated forward return horizons (default: 1,5,10,20,60)"
    )
    parser.add_argument(
        "--horizon", type=int, default=20,
        help="Target horizon for d-sweep charts (default: 20)"
    )
    args = parser.parse_args()
    horizons  = [int(h) for h in args.horizons.split(",")]
    target_h  = args.horizon
    if target_h not in horizons:
        horizons.append(target_h)
        horizons.sort()

    print(f"\n{'='*W}")
    print(f"  FRACTIONAL DIFFERENTIATION RESEARCH")
    print(f"  Instrument: {args.instrument}  |  Timeframe: {args.timeframe}")
    print(f"  Testing d in {list(D_GRID)}  |  Target horizon: h={target_h}")
    print(f"{'='*W}")

    run_frac_diff(args.instrument, args.timeframe, horizons, target_h)


if __name__ == "__main__":
    main()
