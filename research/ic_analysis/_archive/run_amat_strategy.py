"""run_amat_strategy.py -- AMAT Unconditional Trend Composite Strategy.

Architecture:
  No ADX regime gate -- adx_14 is rank #52 NOISE on AMAT (IC=+0.013, 2026-03-19 sweep).
  Eight STRONG trend signals combined as an IS sign-calibrated equal-weight composite.
  Signs and z-score stats are computed on IS data only (zero look-ahead).

IC sweep results (AMAT D, 2026-03-19 fresh sweep):
  ma_spread_20_100  IC=+0.244  STRONG  h=60
  roc_60            IC=+0.215  STRONG  h=60
  zscore_100        IC=+0.197  STRONG  h=60
  ma_spread_10_50   IC=+0.188  STRONG  h=60
  macd_norm         IC=+0.175  STRONG  h=60
  donchian_pos_55   IC=+0.148  STRONG  h=60
  price_vs_sma50    IC=+0.119  STRONG  h=60
  price_pct_rank_60 IC=+0.111  STRONG  h=60

Cost model (individual equity, per fill -- VBT charges entry AND exit):
  Spread:   10 bps per fill
  Slippage:  5 bps per fill
  Effective round-trip: ~30 bps total

Position sizing: 1% risk / (1.5x ATR14 stop), capped at 10x leverage.
IS/OOS split: 70/30.
WFO: 2yr IS (504 bars) / 6mo OOS (126 bars), semi-annual step.

Phases:
  Phase 3: IS/OOS backtest with threshold sweep + annual breakdown
  Phase 4: Walk-Forward Optimisation (threshold re-selected from IS each fold)
  Phase 5: Robustness (Monte Carlo, top-5 removal, 3x slippage, WFO consec. folds)

Usage:
  uv run python research/ic_analysis/run_amat_strategy.py
  uv run python research/ic_analysis/run_amat_strategy.py --sweep
  uv run python research/ic_analysis/run_amat_strategy.py --no-robust
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

INSTRUMENT = "AMAT"
TIMEFRAME  = "D"
IS_RATIO   = 0.70
FWD_H      = 60      # IC sign calibration horizon -- matches all 8 signal peaks

INIT_CASH  = 10_000.0
RISK_PCT   = 0.01
STOP_ATR   = 1.5
MAX_LEV    = 10.0
FREQ       = "d"

SPREAD     = 0.001    # 10 bps per fill (individual equity, wider than ETF)
SLIPPAGE   = 0.0005   # 5 bps per fill

# Eight STRONG h=60 trend signals from the 2026-03-19 IC sweep
SIGNALS = [
    "ma_spread_20_100",
    "roc_60",
    "zscore_100",
    "ma_spread_10_50",
    "macd_norm",
    "donchian_pos_55",
    "price_vs_sma50",
    "price_pct_rank_60",
]

THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# WFO config
WFO_IS   = 504   # 2yr IS  (252 * 2)
WFO_OOS  = 126   # 6mo OOS (252 * 0.5)
WFO_STEP = 126   # semi-annual step

# Robustness gates (individual equity -- stricter than SPY ETF gates)
MC_N                  = 1_000
MC_MIN_5PCT_SHARPE    = 0.50
MC_MIN_PROFITABLE_PCT = 0.80
TOP_N_REMOVE          = 5
STRESS_SLIPPAGE_MULT  = 3.0
STRESS_MIN_SHARPE     = 0.50
WFO_MAX_CONSEC_NEG    = 2

W       = 84
REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()


def _ic_sign(signal: pd.Series, fwd: pd.Series) -> float:
    """Spearman IC sign on IS data only."""
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0 else 1.0


def _zscore_is(series: pd.Series, is_mask: pd.Series) -> pd.Series:
    """Z-score using IS mean/std only; applied to full series (no look-ahead)."""
    is_vals = series[is_mask].dropna()
    mu, sigma = float(is_vals.mean()), float(is_vals.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _size_series(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    """1% risk / (1.5x ATR14 stop), capped at MAX_LEV."""
    atr = _atr14(df)
    stop_pct = (STOP_ATR * atr) / close.where(close > 0)
    return (RISK_PCT / stop_pct.where(stop_pct > 0)).clip(upper=MAX_LEV).fillna(0.0)


# ── Composite builder ─────────────────────────────────────────────────────────

def build_composite(
    sigs: pd.DataFrame,
    close: pd.Series,
    is_mask: pd.Series,
) -> pd.Series:
    """Equal-weight composite of SIGNALS, IS sign-calibrated at h=FWD_H.

    Calibrates IC sign against h=60 forward returns (matches signal peak horizon).
    Z-scores using IS mean/std only. No regime gate.
    """
    fwd_is = np.log(close.shift(-FWD_H) / close)[is_mask]
    parts = []
    for name in SIGNALS:
        if name not in sigs.columns:
            print(f"  [WARN] Signal '{name}' not found -- skipping")
            continue
        sign = _ic_sign(sigs[name][is_mask], fwd_is)
        parts.append(sigs[name] * sign)

    if not parts:
        raise RuntimeError("No signals found in signal dataframe.")

    raw = pd.concat(parts, axis=1).mean(axis=1)
    return _zscore_is(raw, is_mask)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load AMAT OHLCV, build all 52 signals, return (sigs, close, df, size)."""
    df    = _load_ohlcv(INSTRUMENT, TIMEFRAME)
    close = df["close"]
    sigs  = build_all_signals(df)

    valid = sigs.notna().any(axis=1) & close.notna()
    sigs  = sigs[valid]
    close = close[valid]
    df    = df.loc[valid]
    size  = _size_series(df, close)
    return sigs, close, df, size


# ── Backtest core ─────────────────────────────────────────────────────────────

def _run_long(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    size: pd.Series,
    fees: float = SPREAD,
    slippage: float = SLIPPAGE,
) -> dict:
    """Long-only VBT backtest. 1-bar execution delay. Returns stats + pf object."""
    sig = signal_z.shift(1).fillna(0.0)

    pf = vbt.Portfolio.from_signals(
        close,
        entries   = sig > threshold,
        exits     = sig <= 0.0,
        size      = size.values,
        size_type = "percent",
        init_cash = INIT_CASH,
        fees      = fees,
        slippage  = slippage,
        freq      = FREQ,
    )
    n_trades = pf.trades.count()
    return {
        "pf":     pf,
        "sharpe": float(pf.sharpe_ratio()),
        "annual": float(pf.annualized_return()),
        "ret":    float(pf.total_return()),
        "dd":     float(pf.max_drawdown()),
        "trades": int(n_trades),
        "wr":     float(pf.trades.win_rate()) if n_trades > 0 else 0.0,
        "value":  pf.value(),
    }


def _annual_table(value_series: pd.Series) -> pd.Series:
    rets = value_series.pct_change().dropna()
    return (
        pd.DataFrame({"ret": rets})
        .assign(year=lambda x: x.index.year)
        .groupby("year")["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
    )


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def run_phase3(
    sigs: pd.DataFrame,
    close: pd.Series,
    size: pd.Series,
    do_sweep: bool = False,
) -> dict:
    n    = len(sigs)
    is_n = int(n * IS_RATIO)

    is_mask = pd.Series(False, index=sigs.index)
    is_mask.iloc[:is_n] = True
    oos_mask = ~is_mask

    composite_z = build_composite(sigs, close, is_mask)

    close_is  = close[is_mask];  gz_is  = composite_z[is_mask];  sz_is  = size[is_mask]
    close_oos = close[oos_mask]; gz_oos = composite_z[oos_mask]; sz_oos = size[oos_mask]

    print(f"\n{'='*W}")
    print("  PHASE 3 -- IS/OOS BACKTEST (70/30 SPLIT)")
    print(f"{'='*W}")
    print(f"  Total: {n:,} bars  |  IS: {is_n:,}  |  OOS: {oos_mask.sum():,}")
    print(
        f"  IS : {sigs.index[0].date()} -> {sigs.index[is_n-1].date()}\n"
        f"  OOS: {sigs.index[is_n].date()} -> {sigs.index[-1].date()}"
    )

    bh_ret = float(close_oos.iloc[-1] / close_oos.iloc[0] - 1)
    bh_ann = float((1 + bh_ret) ** (252 / len(close_oos)) - 1)
    print(f"\n  Buy & Hold OOS: {bh_ret:+.1%} total  ({bh_ann:+.1%} ann.)")

    if do_sweep:
        print(f"\n  {'Thresh':>7}  {'Sharpe':>8}  {'Annual':>8}  "
              f"{'MaxDD':>8}  {'Trades':>7}  {'WR%':>6}")
        print("  " + "-" * 55)
        for thr in THRESHOLDS:
            r = _run_long(close_oos, gz_oos, thr, sz_oos)
            print(
                f"  {thr:>7.2f}  {r['sharpe']:>+8.3f}  {r['annual']:>+8.1%}  "
                f"{r['dd']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
            )

    # IS threshold selection (no look-ahead)
    best_thr = THRESHOLDS[0]
    best_sh  = -np.inf
    for thr in THRESHOLDS:
        r_is = _run_long(close_is, gz_is, thr, sz_is)
        if r_is["sharpe"] > best_sh:
            best_sh  = r_is["sharpe"]
            best_thr = thr

    r = _run_long(close_oos, gz_oos, best_thr, sz_oos)

    print(f"\n  OOS RESULTS  (IS-selected threshold = {best_thr:.2f}z)")
    print(f"  {'OOS Sharpe Ratio':<26}: {r['sharpe']:>+.3f}")
    print(f"  {'OOS Annualised Return':<26}: {r['annual']:>+.1%}")
    print(f"  {'OOS Total Return':<26}: {r['ret']:>+.1%}")
    print(f"  {'OOS Max Drawdown':<26}: {r['dd']:>+.1%}")
    print(f"  {'Trades':<26}: {r['trades']}")
    print(f"  {'Win Rate':<26}: {r['wr']*100:.1f}%")
    print(f"  {'B&H Annualised Return':<26}: {bh_ann:>+.1%}")

    verdict = "PASS" if r["sharpe"] > 1.0 and r["dd"] > -0.20 else "FAIL"
    print(f"\n  Phase 3 Verdict: [{verdict}]")

    # Annual breakdown
    annual    = _annual_table(r["value"])
    bh_annual = _annual_table(close_oos / close_oos.iloc[0] * INIT_CASH)
    print(f"\n  {'Year':<8}  {'Strategy':>12}  {'B&H':>10}  {'Alpha':>10}")
    print("  " + "-" * 46)
    for yr, val in annual.items():
        bh_yr = bh_annual.get(yr, np.nan)
        alpha = val - bh_yr if not np.isnan(bh_yr) else np.nan
        bh_s  = f"{bh_yr:>+10.1%}" if not np.isnan(bh_yr) else "       N/A"
        alp_s = f"{alpha:>+10.1%}" if not np.isnan(alpha) else "       N/A"
        print(f"  {yr:<8}  {val:>+12.1%}  {bh_s}  {alp_s}")

    return {
        "pf_oos":    r["pf"],
        "r_oos":     r,
        "bh_ann":    bh_ann,
        "best_thr":  best_thr,
        "composite_z": composite_z,
        "is_mask":   is_mask,
        "oos_mask":  oos_mask,
        "close_oos": close_oos,
        "gz_oos":    gz_oos,
        "sz_oos":    sz_oos,
        "is_n":      is_n,
        "n":         n,
    }


# ── Phase 4: WFO ─────────────────────────────────────────────────────────────

def run_phase4(
    sigs: pd.DataFrame,
    close: pd.Series,
    size: pd.Series,
) -> dict:
    n = len(sigs)
    print(f"\n{'='*W}")
    print("  PHASE 4 -- WALK-FORWARD OPTIMISATION")
    print(f"  IS: {WFO_IS} bars (2yr)  |  OOS: {WFO_OOS} bars (6mo)  |  Step: {WFO_STEP}")
    print(f"{'='*W}")

    # Define folds
    folds: list[tuple[int, int, int, int]] = []
    start = 0
    while start + WFO_IS + WFO_OOS <= n:
        is0, is1   = start, start + WFO_IS
        oos0, oos1 = is1, is1 + WFO_OOS
        folds.append((is0, is1, oos0, oos1))
        start += WFO_STEP

    n_folds = len(folds)
    if n_folds < 3:
        print(f"  WARNING: only {n_folds} folds -- WFO not meaningful.")
        return {"n_folds": n_folds, "all_pass": False}

    print(
        f"  Folds: {n_folds}  "
        f"({sigs.index[folds[0][2]].date()} -> {sigs.index[folds[-1][3]-1].date()})"
    )

    fold_results:  list[dict]      = []
    stitched_rets: list[pd.Series] = []

    print(
        f"\n  {'Fold':>5}  {'IS-Thr':>7}  {'IS-Sh':>7}  "
        f"{'OOS-Sh':>7}  {'OOS-Ann':>9}  {'OOS-DD':>8}  {'Trd':>4}"
    )
    print("  " + "-" * 62)

    for i, (is0, is1, oos0, oos1) in enumerate(folds):
        fold_sigs  = sigs.iloc[is0:oos1]
        fold_close = close.iloc[is0:oos1]
        fold_size  = size.iloc[is0:oos1]

        fold_is_mask = pd.Series(False, index=fold_sigs.index)
        fold_is_mask.iloc[: is1 - is0] = True

        # Build composite calibrated to this fold's IS
        try:
            comp_z = build_composite(fold_sigs, fold_close, fold_is_mask)
        except Exception as e:
            print(f"  Fold {i+1}: skipped ({e})")
            continue

        close_is = fold_close.iloc[: is1 - is0]
        gz_is    = comp_z.iloc[: is1 - is0]
        sz_is    = fold_size.iloc[: is1 - is0]

        close_oos = fold_close.iloc[is1 - is0:]
        gz_oos    = comp_z.iloc[is1 - is0:]
        sz_oos    = fold_size.iloc[is1 - is0:]

        # IS threshold selection
        best_thr   = THRESHOLDS[0]
        best_sh_is = -np.inf
        for thr in THRESHOLDS:
            r_is = _run_long(close_is, gz_is, thr, sz_is)
            if r_is["sharpe"] > best_sh_is:
                best_sh_is = r_is["sharpe"]
                best_thr   = thr

        r_oos = _run_long(close_oos, gz_oos, best_thr, sz_oos)
        stitched_rets.append(r_oos["pf"].returns())

        flag = " *" if r_oos["sharpe"] < 0 else "  "
        print(
            f"  {i+1:>5}  {best_thr:>7.2f}  {best_sh_is:>+7.3f}"
            f"  {r_oos['sharpe']:>+7.3f}{flag}  {r_oos['annual']:>+9.1%}"
            f"  {r_oos['dd']:>+8.1%}  {r_oos['trades']:>4}"
        )

        fold_results.append({
            "fold":       i + 1,
            "is_start":   sigs.index[is0].date(),
            "is_end":     sigs.index[is1 - 1].date(),
            "oos_start":  sigs.index[oos0].date(),
            "oos_end":    sigs.index[oos1 - 1].date(),
            "threshold":  best_thr,
            "is_sharpe":  round(best_sh_is, 3),
            "oos_sharpe": round(r_oos["sharpe"], 3),
            "oos_annual": round(float(r_oos["annual"]), 4),
            "oos_dd":     round(float(r_oos["dd"]), 4),
            "oos_trades": r_oos["trades"],
        })

    # Stitch
    stitched = pd.concat(stitched_rets).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    std_s    = float(stitched.std())
    stitched_sharpe = (
        float(stitched.mean()) / std_s * np.sqrt(252) if std_s > 1e-10 else 0.0
    )

    oos_sharpes  = [f["oos_sharpe"] for f in fold_results]
    is_sharpes   = [f["is_sharpe"]  for f in fold_results]
    pct_positive = sum(s > 0   for s in oos_sharpes) / n_folds
    pct_above1   = sum(s > 1.0 for s in oos_sharpes) / n_folds
    worst_fold   = float(min(oos_sharpes))
    avg_is_sh    = float(np.mean(is_sharpes))
    parity       = stitched_sharpe / avg_is_sh if avg_is_sh > 1e-6 else 0.0

    print(f"\n  WFO Gate Summary  ({n_folds} folds)")
    g1 = pct_positive >= 0.70
    g2 = pct_above1   >= 0.50
    g3 = worst_fold   >= -2.0
    g4 = stitched_sharpe >= 1.50
    g5 = parity >= 0.50

    def gline(lbl: str, val: str, passed: bool) -> None:
        print(f"  {lbl:<38}: {val}  [{'PASS' if passed else 'FAIL'}]")

    gline("> 0% folds",           f"{pct_positive:.0%}", g1)
    gline("> 1 Sharpe folds",     f"{pct_above1:.0%}",   g2)
    gline("Worst fold Sharpe",    f"{worst_fold:+.3f}",   g3)
    gline("Stitched OOS Sharpe",  f"{stitched_sharpe:+.3f}", g4)
    gline("IS/OOS parity",        f"{parity:.3f}",        g5)

    all_pass = all([g1, g2, g3, g4, g5])
    print(f"\n  Phase 4 Verdict: [{'PASS' if all_pass else 'FAIL'}]")

    # Save WFO CSV (used by Phase 5 Gate 4)
    wfo_path = REPORTS / "amat_wfo.csv"
    pd.DataFrame(fold_results).to_csv(wfo_path, index=False)
    print(f"  WFO results saved: {wfo_path}")

    return {
        "n_folds":         n_folds,
        "fold_results":    fold_results,
        "stitched_sharpe": stitched_sharpe,
        "pct_positive":    pct_positive,
        "pct_above1":      pct_above1,
        "worst_fold":      worst_fold,
        "parity":          parity,
        "all_pass":        all_pass,
    }


# ── Phase 5: Robustness ───────────────────────────────────────────────────────

def _trade_returns(pf) -> np.ndarray | None:
    if pf.trades.count() < 5:
        return None
    return pf.trades.records_readable["Return"].values


def _monte_carlo(pf, oos_bars: int) -> dict:
    """Shuffle per-trade % returns N times; Sharpe distribution.
    Gate: 5th-pct > MC_MIN_5PCT_SHARPE AND > MC_MIN_PROFITABLE_PCT sims profitable.
    """
    trade_ret = _trade_returns(pf)
    if trade_ret is None:
        return {"error": "< 5 trades", "gate_pass": False}

    oos_years       = oos_bars / 252.0
    trades_per_year = len(trade_ret) / max(oos_years, 0.1)
    rng = np.random.default_rng(42)
    sharpes: list[float] = []
    profitable = 0

    for _ in range(MC_N):
        s        = rng.permutation(trade_ret)
        mu, sig  = float(s.mean()), float(s.std())
        sh       = mu / sig * np.sqrt(trades_per_year) if sig > 1e-10 else 0.0
        sharpes.append(sh)
        if float(s.sum()) > 0:
            profitable += 1

    arr      = np.array(sharpes)
    pct5     = float(np.percentile(arr, 5))
    prof_pct = profitable / MC_N
    gate     = pct5 > MC_MIN_5PCT_SHARPE and prof_pct > MC_MIN_PROFITABLE_PCT

    return {
        "n_trades":       int(len(trade_ret)),
        "mean_sharpe":    round(float(arr.mean()), 3),
        "pct5_sharpe":    round(pct5, 3),
        "pct95_sharpe":   round(float(np.percentile(arr, 95)), 3),
        "profitable_pct": round(prof_pct, 3),
        "gate_pass":      gate,
    }


def _remove_top_n(pf) -> dict:
    """Remove TOP_N_REMOVE largest wins; check remaining arithmetic sum > 0."""
    trade_ret = _trade_returns(pf)
    if trade_ret is None:
        return {"error": "< 5 trades", "gate_pass": False}
    if len(trade_ret) < TOP_N_REMOVE + 1:
        return {"error": f"Only {len(trade_ret)} trades", "gate_pass": False}

    total   = float(trade_ret.sum())
    top_sum = float(np.sort(trade_ret)[-TOP_N_REMOVE:].sum())
    remain  = total - top_sum
    return {
        "total_trades":      int(len(trade_ret)),
        "total_ret_pct":     round(total * 100, 2),
        "top_n_ret_pct":     round(top_sum * 100, 2),
        "remaining_ret_pct": round(remain * 100, 2),
        "gate_pass":         remain > 0,
    }


def _stress_slippage(close_oos, gz_oos, sz_oos, threshold) -> dict:
    """Re-run OOS with STRESS_SLIPPAGE_MULT x slippage. Gate: Sharpe > STRESS_MIN_SHARPE."""
    r = _run_long(
        close_oos, gz_oos, threshold, sz_oos,
        fees=SPREAD,
        slippage=SLIPPAGE * STRESS_SLIPPAGE_MULT,
    )
    return {
        "sharpe":    round(r["sharpe"], 3),
        "slip_bps":  round(SLIPPAGE * STRESS_SLIPPAGE_MULT * 10_000, 1),
        "gate_pass": r["sharpe"] > STRESS_MIN_SHARPE,
    }


def _wfo_consec(wfo_path: Path) -> dict:
    """Read WFO CSV; check max consecutive negative OOS folds."""
    if not wfo_path.exists():
        return {"error": f"{wfo_path.name} not found", "gate_pass": False}
    df = pd.read_csv(wfo_path)
    if "oos_sharpe" not in df.columns:
        return {"error": "oos_sharpe column missing", "gate_pass": False}

    sharpes   = df["oos_sharpe"].tolist()
    n_folds   = len(sharpes)
    n_pos     = sum(s > 0 for s in sharpes)
    consec    = max_consec = 0
    for s in sharpes:
        if s <= 0:
            consec    += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "n_folds":        n_folds,
        "positive_folds": n_pos,
        "max_consec_neg": max_consec,
        "gate_pass":      max_consec <= WFO_MAX_CONSEC_NEG,
    }


def run_phase5(p3: dict) -> dict:
    """Run all 4 robustness gates."""
    pf_oos    = p3["pf_oos"]
    oos_bars  = p3["oos_mask"].sum()
    close_oos = p3["close_oos"]
    gz_oos    = p3["gz_oos"]
    sz_oos    = p3["sz_oos"]
    threshold = p3["best_thr"]

    print(f"\n{'='*W}")
    print("  PHASE 5 -- ROBUSTNESS VALIDATION")
    print(f"{'='*W}")

    # Gate 1: Monte Carlo
    print(f"\n  [Gate 1] Monte Carlo (N={MC_N:,} shuffles)...")
    mc = _monte_carlo(pf_oos, oos_bars)
    if "error" in mc:
        print(f"  SKIP: {mc['error']}")
    else:
        g1_lbl = "PASS" if mc["gate_pass"] else "FAIL"
        print(f"  Trades: {mc['n_trades']}  |  Mean Sharpe: {mc['mean_sharpe']:.3f}")
        print(
            f"  5th-pct: {mc['pct5_sharpe']:.3f} (gate > {MC_MIN_5PCT_SHARPE})  "
            f"95th-pct: {mc['pct95_sharpe']:.3f}  [{g1_lbl}]"
        )
        print(
            f"  Profitable sims: {mc['profitable_pct']:.1%}"
            f" (gate > {MC_MIN_PROFITABLE_PCT:.0%})"
        )

    # Gate 2: Remove top-N
    print(f"\n  [Gate 2] Remove top {TOP_N_REMOVE} winning trades...")
    top_n = _remove_top_n(pf_oos)
    if "error" in top_n:
        print(f"  SKIP: {top_n['error']}")
    else:
        g2_lbl = "PASS" if top_n["gate_pass"] else "FAIL"
        print(
            f"  Total: {top_n['total_ret_pct']:.1f}%  |  "
            f"Top-{TOP_N_REMOVE}: {top_n['top_n_ret_pct']:.1f}%  |  "
            f"Remaining: {top_n['remaining_ret_pct']:.1f}%  [{g2_lbl}]"
        )

    # Gate 3: 3x slippage
    slip_bps = round(SLIPPAGE * STRESS_SLIPPAGE_MULT * 10_000, 1)
    print(f"\n  [Gate 3] 3x slippage stress ({slip_bps} bps per fill)...")
    stress = _stress_slippage(close_oos, gz_oos, sz_oos, threshold)
    g3_lbl = "PASS" if stress["gate_pass"] else "FAIL"
    print(
        f"  OOS Sharpe @ 3x slip: {stress['sharpe']:.3f}"
        f" (gate > {STRESS_MIN_SHARPE})  [{g3_lbl}]"
    )

    # Gate 4: WFO consecutive folds
    print("\n  [Gate 4] WFO consecutive negative folds...")
    wfo = _wfo_consec(REPORTS / "amat_wfo.csv")
    if "error" in wfo:
        print(f"  SKIP: {wfo['error']}")
    else:
        g4_lbl = "PASS" if wfo["gate_pass"] else "FAIL"
        print(
            f"  Folds: {wfo['n_folds']}  |  Positive: {wfo['positive_folds']}  |  "
            f"Max consec. neg: {wfo['max_consec_neg']} (gate <= {WFO_MAX_CONSEC_NEG})  [{g4_lbl}]"
        )

    all_pass = all([mc["gate_pass"], top_n["gate_pass"], stress["gate_pass"], wfo["gate_pass"]])

    print(f"\n  {'─' * 60}")
    print("  GATE SUMMARY")
    print(f"  {'─' * 60}")
    gates = {
        f"MC 5th-pct > {MC_MIN_5PCT_SHARPE}, >{MC_MIN_PROFITABLE_PCT:.0%} profitable":
            mc["gate_pass"],
        f"Remove top {TOP_N_REMOVE}: remaining > 0":
            top_n["gate_pass"],
        f"3x slippage: Sharpe > {STRESS_MIN_SHARPE}":
            stress["gate_pass"],
        f"WFO max consec. neg <= {WFO_MAX_CONSEC_NEG}":
            wfo["gate_pass"],
    }
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    verdict = (
        "ALL GATES PASS -- cleared for live implementation"
        if all_pass else
        "GATES FAILED -- do not proceed to live"
    )
    print(f"\n  Phase 5 Verdict: {verdict}")
    return {"all_pass": all_pass}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(do_sweep: bool = False, do_robust: bool = True) -> None:
    print("\n" + "=" * W)
    print("  AMAT DAILY -- UNCONDITIONAL TREND COMPOSITE (8 STRONG SIGNALS)")
    print(
        f"  Signals: {', '.join(SIGNALS[:4])},\n"
        f"           {', '.join(SIGNALS[4:])}"
    )
    print(
        f"  Spread: {SPREAD*1e4:.0f}bps/fill  Slippage: {SLIPPAGE*1e4:.0f}bps/fill  "
        f"FwdH: {FWD_H}d  Risk: {RISK_PCT*100:.0f}%/trade  Stop: {STOP_ATR}xATR14"
    )
    print("=" * W)

    print("\nLoading AMAT data and building signals...")
    sigs, close, df, size = load_data()
    n = len(sigs)
    print(f"  {n:,} bars  ({sigs.index[0].date()} -> {sigs.index[-1].date()})")

    # Phase 3
    p3 = run_phase3(sigs, close, size, do_sweep=do_sweep)

    # Phase 4
    p4 = run_phase4(sigs, close, size)

    # Phase 5
    if do_robust:
        p5 = run_phase5(p3)
    else:
        p5 = {"all_pass": None}

    # Save summary CSV
    r = p3["r_oos"]
    pd.DataFrame([{
        "instrument":       INSTRUMENT,
        "signals":          len(SIGNALS),
        "fwd_h":            FWD_H,
        "threshold":        p3["best_thr"],
        "oos_sharpe":       r["sharpe"],
        "oos_annual":       r["annual"],
        "oos_ret":          r["ret"],
        "oos_dd":           r["dd"],
        "oos_trades":       r["trades"],
        "oos_wr":           r["wr"],
        "bh_annual":        p3["bh_ann"],
        "wfo_stitched_sh":  p4.get("stitched_sharpe"),
        "wfo_pct_pos":      p4.get("pct_positive"),
        "wfo_worst":        p4.get("worst_fold"),
        "robustness_pass":  p5["all_pass"],
    }]).to_csv(REPORTS / "amat_strategy.csv", index=False)
    print(f"\n  Summary saved: {REPORTS / 'amat_strategy.csv'}")

    # Equity chart
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        port_val = r["value"]
        oos_mask = p3["oos_mask"]
        close_oos = close[oos_mask]
        bh_val   = close_oos / close_oos.iloc[0] * INIT_CASH

        def norm(v: pd.Series) -> pd.Series:
            return v / v.iloc[0] * 100

        def dd_curve(v: pd.Series) -> pd.Series:
            pk = v.cummax()
            return (v - pk) / pk * 100

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.70, 0.30],
            subplot_titles=["Equity Curve (OOS, rebased 100)", "Drawdown %"],
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(
            x=port_val.index, y=norm(port_val),
            name="AMAT Strategy", line=dict(color="#00E5FF", width=2.0),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bh_val.index, y=norm(bh_val),
            name="AMAT Buy & Hold", line=dict(color="#FFD600", width=1.5, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=port_val.index, y=dd_curve(port_val),
            name="Strategy DD", fill="tozeroy",
            line=dict(color="#FF1744"), fillcolor="rgba(255,23,68,0.18)",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=bh_val.index, y=dd_curve(bh_val),
            name="B&H DD", line=dict(color="#FFD600", width=1.0, dash="dot"),
        ), row=2, col=1)

        fig.update_layout(
            title=(
                f"AMAT Daily | Unconditional Trend Composite | "
                f"Thr={p3['best_thr']:.2f}z | OOS {sigs.index[p3['is_n']].date()}->"
                f"{sigs.index[-1].date()}"
            ),
            height=720, template="plotly_dark", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Value (rebased 100)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %",          row=2, col=1)

        html_out = REPORTS / "amat_strategy.html"
        fig.write_html(str(html_out))
        print(f"  Chart saved:   {html_out}")
    except Exception as e:
        print(f"  [INFO] Chart skipped: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AMAT Unconditional Trend Strategy")
    parser.add_argument("--sweep",     action="store_true", help="OOS threshold sweep")
    parser.add_argument("--no-robust", action="store_true", help="Skip Phase 5 robustness")
    args = parser.parse_args()
    run(do_sweep=args.sweep, do_robust=not args.no_robust)


if __name__ == "__main__":
    main()
