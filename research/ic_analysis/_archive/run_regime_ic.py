"""run_regime_ic.py -- Regime-Conditional Information Coefficient Analysis.

Splits bars into market regimes and computes IC/ICIR within each bucket,
revealing signals that look like noise unconditionally but carry strong
predictive power in specific market states.

Two independent regime axes
---------------------------
  ADX axis   : Ranging (ADX<20) | Neutral (20-25) | Trending (ADX>25)
  Vol axis   : Low (<30th pct) | Mid (30-70th pct) | High (>70th pct)
               Vol proxy = realized_vol_20 (already in Group D signals)

For each regime the script:
  1. Filters bars to that regime only
  2. Computes IC and ICIR at the target horizon
  3. Computes IC uplift vs unconditional baseline
  4. Flags signals that flip sign across regimes (regime-sensitive)

Output
------
  Console : per-regime leaderboards + cross-regime comparison table
  CSV     : .tmp/reports/regime_ic_{slug}.csv  (signal x regime IC matrix)

Usage
-----
  uv run python research/ic_analysis/run_regime_ic.py
  uv run python research/ic_analysis/run_regime_ic.py --instrument TXN --timeframe D --horizon 20
  uv run python research/ic_analysis/run_regime_ic.py --instrument SPY --timeframe D --horizon 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402

from research.ic_analysis.run_ic import compute_forward_returns  # noqa: E402

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_REGIME_BARS = 50   # skip regime bucket if fewer bars
ICIR_WINDOW     = 60   # rolling window for ICIR (bars)


# ---------------------------------------------------------------------------
# IC helpers
# ---------------------------------------------------------------------------

def _ic(signal: pd.Series, fwd: pd.Series) -> float:
    """Spearman IC via rank-Pearson (faster than scipy.stats.spearmanr)."""
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 10:
        return np.nan
    r = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")
    return float(r) if not np.isnan(r) else np.nan


def _icir(signal: pd.Series, fwd: pd.Series, window: int = ICIR_WINDOW) -> float:
    """Rolling Spearman IC mean/std — uses rank-then-rolling-Pearson (vectorised)."""
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < window + 5:
        return np.nan
    # Spearman = Pearson on ranks; rolling().corr() is C-level, no Python loop
    s_rank = both.iloc[:, 0].rank()
    f_rank = both.iloc[:, 1].rank()
    rolling_ic = s_rank.rolling(window).corr(f_rank).dropna()
    if len(rolling_ic) < 2:
        return np.nan
    mu  = rolling_ic.mean()
    std = rolling_ic.std()
    return float(mu / std) if std > 1e-10 else np.nan


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


def _sweep_regime(
    signals: pd.DataFrame,
    fwd: pd.Series,
    mask: pd.Series,
    label: str,
) -> pd.DataFrame:
    """Compute IC + ICIR for every signal within the rows selected by mask."""
    sig_r = signals[mask]
    fwd_r = fwd[mask]
    n     = int(mask.sum())

    # Vectorised Spearman IC for all signals at once (rank then Pearson corrwith)
    sig_ranks = sig_r.rank()
    fwd_rank  = fwd_r.rank()
    ic_all    = sig_ranks.corrwith(fwd_rank)

    rows = []
    for col in signals.columns:
        ic   = float(ic_all[col]) if not np.isnan(ic_all[col]) else np.nan
        icir = _icir(sig_r[col], fwd_r)
        rows.append({
            "signal":  col,
            "regime":  label,
            "n_bars":  n,
            "ic":      ic,
            "icir":    icir,
            "verdict": _verdict(ic, icir) if not np.isnan(ic) else "N/A",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

def _adx_regimes(signals: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks per ADX regime. adx_14 must be in signals columns."""
    if "adx_14" not in signals.columns:
        return {}
    adx = signals["adx_14"]
    return {
        "ADX_Ranging":  adx < 20,
        "ADX_Neutral":  (adx >= 20) & (adx <= 25),
        "ADX_Trending": adx > 25,
    }


def _vol_regimes(signals: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks per realised-vol tercile. realized_vol_20 must exist."""
    if "realized_vol_20" not in signals.columns:
        return {}
    rv = signals["realized_vol_20"]
    p30 = rv.quantile(0.30)
    p70 = rv.quantile(0.70)
    return {
        "Vol_Low":  rv <= p30,
        "Vol_Mid":  (rv > p30) & (rv <= p70),
        "Vol_High": rv > p70,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

W = 74

def _print_regime_leaderboard(df: pd.DataFrame, regime: str, top_n: int = 15) -> None:
    sub = (
        df[df["regime"] == regime]
        .assign(abs_ic=lambda x: x["ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
    )
    n_bars = int(sub["n_bars"].iloc[0]) if not sub.empty else 0
    print(f"\n  {'-'*W}")
    print(f"  Regime: {regime:<30}  Bars: {n_bars:,}")
    print(f"  {'-'*W}")
    print(f"  {'Rank':>4}  {'Signal':<24}  {'IC':>8}  {'ICIR':>7}  Verdict")
    print(f"  {'-'*(W-2)}")
    for i, row in sub.head(top_n).iterrows():
        ic_s  = f"{row['ic']:>+8.4f}"  if not np.isnan(row["ic"])   else "     NaN"
        ir_s  = f"{row['icir']:>+7.3f}" if not np.isnan(row["icir"]) else "    NaN"
        print(f"  {i+1:>4}  {row['signal']:<24}  {ic_s}  {ir_s}  {row['verdict']}")


def _print_cross_regime_table(
    all_results: pd.DataFrame,
    baseline: pd.DataFrame,
    regime_cols: list[str],
    top_signals: list[str],
) -> None:
    """IC matrix: signal rows × regime columns, with uplift vs baseline."""
    base_map = baseline.set_index("signal")["ic"].to_dict()

    print(f"\n{'='*W}")
    print("  CROSS-REGIME IC MATRIX  (uplift = regime_IC - unconditional_IC)")
    print(f"{'='*W}")

    # header
    col_w = 10
    hdr = f"  {'Signal':<24}  {'Uncond':>{col_w}}"
    for r in regime_cols:
        short = r.replace("ADX_", "").replace("Vol_", "")[:col_w]
        hdr += f"  {short:>{col_w}}"
    hdr += "  Flip?"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    pivot = all_results.pivot_table(index="signal", columns="regime", values="ic")

    for sig in top_signals:
        base_ic = base_map.get(sig, np.nan)
        base_s  = f"{base_ic:>+{col_w}.4f}" if not np.isnan(base_ic) else " " * col_w

        regime_ics = []
        cells = []
        for r in regime_cols:
            val = pivot.loc[sig, r] if (sig in pivot.index and r in pivot.columns) else np.nan
            regime_ics.append(val)
            if np.isnan(val):
                cells.append(" " * col_w)
            else:
                uplift = val - base_ic if not np.isnan(base_ic) else 0.0
                sign   = "+" if uplift > 0 else "-"
                cells.append(f"{val:>+7.4f}{sign}{abs(uplift):.3f}"[:col_w])

        # Flip = signal changes sign across regime buckets
        valid_ics = [v for v in regime_ics if not np.isnan(v)]
        flips = (
            len({1 if v > 0 else -1 for v in valid_ics}) > 1
            if len(valid_ics) >= 2 else False
        )
        flip_s = " FLIP!" if flips else ""

        row_s = f"  {sig:<24}  {base_s}"
        for c in cells:
            row_s += f"  {c:>{col_w}}"
        row_s += flip_s
        print(row_s)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_regime_ic(
    instrument: str,
    timeframe: str,
    horizon: int,
    top_n: int = 15,
) -> pd.DataFrame:
    slug = f"{instrument}_{timeframe}".lower()

    df    = _load_ohlcv(instrument, timeframe)
    close = df["close"]

    print(f"\nBuilding 52 signals for {instrument} {timeframe}...")
    signals = build_all_signals(df)

    fwd_all = compute_forward_returns(close, [horizon])
    fwd     = fwd_all[f"fwd_{horizon}"]

    # Align: drop rows where signal or forward return is missing
    valid   = signals.notna().any(axis=1) & fwd.notna()
    signals = signals[valid]
    fwd     = fwd[valid]
    n_total = len(signals)

    print(f"Analysis window: {n_total:,} bars | horizon=h={horizon}")

    # -- Unconditional baseline (vectorised IC, per-signal ICIR) ---------------
    print("Computing unconditional IC (baseline)...")
    sig_ranks  = signals.rank()
    fwd_rank   = fwd.rank()
    ic_all     = sig_ranks.corrwith(fwd_rank)
    baseline_rows = []
    for col in signals.columns:
        ic   = float(ic_all[col]) if not np.isnan(ic_all[col]) else np.nan
        icir = _icir(signals[col], fwd)
        baseline_rows.append({"signal": col, "regime": "Unconditional",
                               "n_bars": n_total, "ic": ic, "icir": icir,
                               "verdict": _verdict(ic, icir) if not np.isnan(ic) else "N/A"})
    baseline = pd.DataFrame(baseline_rows)

    # -- Regime masks ----------------------------------------------------------
    all_masks: dict[str, pd.Series] = {}
    all_masks.update(_adx_regimes(signals))
    all_masks.update(_vol_regimes(signals))

    # -- Per-regime IC ---------------------------------------------------------
    regime_results: list[pd.DataFrame] = []
    for regime_label, mask in all_masks.items():
        n_reg = int(mask.sum())
        if n_reg < MIN_REGIME_BARS:
            print(f"  [SKIP] {regime_label}: only {n_reg} bars (min={MIN_REGIME_BARS})")
            continue
        print(f"  Computing IC for {regime_label} ({n_reg} bars, "
              f"{n_reg/n_total*100:.0f}% of data)...")
        regime_df = _sweep_regime(signals, fwd, mask, regime_label)
        regime_results.append(regime_df)

    if not regime_results:
        print("No regime had sufficient bars. Exiting.")
        return baseline

    all_results = pd.concat([baseline] + regime_results, ignore_index=True)

    # -- Print per-regime leaderboards -----------------------------------------
    regime_order = list(all_masks.keys())

    print(f"\n{'='*W}")
    print(f"  REGIME-CONDITIONAL IC -- {instrument} {timeframe}  (h={horizon})")
    print(f"  Unconditional bars: {n_total:,}  |  Min regime bars: {MIN_REGIME_BARS}")
    print(f"{'='*W}")

    for regime_label in regime_order:
        if regime_label in all_results["regime"].values:
            _print_regime_leaderboard(all_results, regime_label, top_n=top_n)

    # -- Cross-regime table for top-20 signals by |unconditional IC| -----------
    top_signals = (
        baseline.assign(abs_ic=lambda x: x["ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(20)["signal"]
        .tolist()
    )
    valid_regimes = [r for r in regime_order if r in all_results["regime"].values]
    _print_cross_regime_table(all_results, baseline, valid_regimes, top_signals)

    # -- Regime uplift summary -------------------------------------------------
    print(f"\n{'='*W}")
    print("  REGIME UPLIFT SUMMARY  (signals with largest IC change across regimes)")
    print(f"{'='*W}")

    pivot = all_results.pivot_table(index="signal", columns="regime", values="ic")
    baseline_map = baseline.set_index("signal")["ic"]

    uplift_rows = []
    for sig in pivot.index:
        base = baseline_map.get(sig, np.nan)
        if np.isnan(base):
            continue
        regime_vals = {r: pivot.loc[sig, r] for r in valid_regimes if r in pivot.columns}
        regime_vals = {k: v for k, v in regime_vals.items() if not np.isnan(v)}
        if not regime_vals:
            continue
        best_regime  = max(regime_vals, key=lambda r: abs(regime_vals[r]))
        best_ic      = regime_vals[best_regime]
        uplift       = abs(best_ic) - abs(base)
        valid_ics    = list(regime_vals.values())
        flips        = len({1 if v > 0 else -1 for v in valid_ics}) > 1 if len(valid_ics) >= 2 else False
        uplift_rows.append({
            "signal":       sig,
            "base_ic":      base,
            "best_regime":  best_regime,
            "best_ic":      best_ic,
            "uplift":       uplift,
            "flips":        flips,
        })

    uplift_df = (
        pd.DataFrame(uplift_rows)
        .sort_values("uplift", ascending=False)
        .reset_index(drop=True)
    )

    print(f"  {'Signal':<24}  {'Base IC':>8}  {'Best Regime':<18}  "
          f"{'Regime IC':>9}  {'Uplift':>7}  Flip?")
    print("  " + "-" * (W - 2))
    for _, row in uplift_df.head(15).iterrows():
        flip_s = "FLIP!" if row["flips"] else ""
        print(
            f"  {row['signal']:<24}  {row['base_ic']:>+8.4f}  "
            f"{row['best_regime']:<18}  {row['best_ic']:>+9.4f}  "
            f"{row['uplift']:>+7.4f}  {flip_s}"
        )

    # -- Save CSV --------------------------------------------------------------
    out_path = REPORTS_DIR / f"regime_ic_{slug}_h{horizon}.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-Conditional IC Analysis")
    parser.add_argument("--instrument", default="TXN")
    parser.add_argument("--timeframe",  default="D")
    parser.add_argument("--horizon",    type=int, default=20,
                        help="Forward return horizon in bars (default 20)")
    parser.add_argument("--top_n",      type=int, default=15,
                        help="Signals shown per regime leaderboard")
    args = parser.parse_args()
    run_regime_ic(args.instrument, args.timeframe, args.horizon, args.top_n)


if __name__ == "__main__":
    main()
