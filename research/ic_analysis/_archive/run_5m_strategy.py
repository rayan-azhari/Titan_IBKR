"""run_5m_strategy.py -- Phase 3-5 pipeline for 5-minute intraday strategies.

Instrument-agnostic IC-weighted composite. Works for any 5m equity in data/databento/:
  - EOG (mean-reversion): all STRONG h=60 signals have negative IC at h=60
  - UNH (momentum):       all STRONG h=60 signals have positive IC at h=60

Both handled identically: IS sign-calibration flips each signal by sign(IS_IC),
so composite_z > 0 always means "bullish." No regime gate needed.

Architecture:
  - Entry  : next bar after composite_z > threshold
  - Exit   : after FWD_H=60 bars (5 hours) OR end-of-session (whichever first)
  - Costs  : 5 bps spread + 2 bps slippage per fill (realistic for liquid large-caps)
  - Long-only; ATR14-based sizing capped at MAX_LEV=2.0x

Phases:
  Phase 3  : IS/OOS backtest (70/30), IS-threshold sweep, annual breakdown
  Phase 4  : Walk-forward optimisation (IS=5040 bars ~3mo, OOS=1560 bars ~1mo)
  Phase 5  : Robustness: MC shuffle, remove-top-N, 3x slippage stress

Usage:
  uv run python research/ic_analysis/run_5m_strategy.py --instrument EOG
  uv run python research/ic_analysis/run_5m_strategy.py --instrument UNH --sweep
  uv run python research/ic_analysis/run_5m_strategy.py --instrument EOG --no-robust
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

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = ROOT / "data" / "databento"
TIMEFRAME = "1yr_5m"

FWD_H    = 60         # 60 bars = 5 hours holding period
MIN_IC   = 0.03       # minimum |IS IC| to include a signal in composite
IS_RATIO = 0.70

INIT_CASH = 10_000.0
RISK_PCT  = 0.01
STOP_ATR  = 1.5
MAX_LEV   = 2.0

SPREAD   = 0.0005     # 5 bps per fill (entry and exit charged separately by VBT)
SLIPPAGE = 0.0002     # 2 bps per fill

THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# WFO: 5m bars (~78 bars/day)
WFO_IS   = 5040   # ~65 trading days (~3 months)
WFO_OOS  = 1560   # ~20 trading days (~1 month)
WFO_STEP = 1560

BARS_PER_YEAR = 78 * 252   # 5m trading bars in a year (~19,656)

# Robustness gates
MC_N                  = 1_000
MC_MIN_5PCT_SHARPE    = 0.30
MC_MIN_PROFITABLE_PCT = 0.70
TOP_N_REMOVE          = 5
STRESS_MULT           = 3.0
STRESS_MIN_SHARPE     = 0.30
WFO_MAX_CONSEC_NEG    = 3
WFO_GATE_PCT_POS      = 0.60
WFO_GATE_STITCHED     = 0.30
WFO_GATE_PARITY       = 0.30

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


def _session_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series: True at the last 5m bar of each trading session."""
    ts      = pd.Series(index, index=index)
    next_ts = ts.shift(-1)
    gap_sec = (next_ts - ts).dt.total_seconds()
    # End-of-session: next bar is > 10 min away (overnight gap) or final bar
    return (gap_sec > 600) | gap_sec.isna()


def _sharpe_daily(bar_rets: pd.Series) -> float:
    """Annualised Sharpe via daily resample — avoids 5m calendar freq inflation.

    Includes all trading days (even flat days with zero return) to avoid
    upward bias from 'return given traded' conditioning.
    """
    daily       = bar_rets.resample("D").apply(lambda x: (1 + x).prod() - 1)
    bar_counts  = bar_rets.resample("D").count()
    trading_days = daily[bar_counts > 0]   # only days with 5m bars
    if len(trading_days) < 5 or float(trading_days.std()) < 1e-10:
        return 0.0
    return float(trading_days.mean() / trading_days.std() * np.sqrt(252))


def _annual_ret(pf, trading_days: int) -> float:
    ret = float(pf.total_return())
    return float((1 + ret) ** (252 / max(trading_days, 1)) - 1)


# ── Composite builder ─────────────────────────────────────────────────────────

def build_composite_for_fold(
    sigs: pd.DataFrame,
    close: pd.Series,
    is_mask: pd.Series,
    fwd_h: int = FWD_H,
    min_ic: float = MIN_IC,
) -> pd.Series:
    """IS-calibrated IC-weighted composite z-score. No look-ahead.

    Steps:
      1. Compute Spearman IC of each signal vs h=fwd_h log-return on IS bars.
      2. Drop signals with |IS IC| < min_ic.
      3. Multiply each signal by sign(IC) * |IC| (sign-flip + IC weighting).
      4. Sum weighted signals -> raw composite.
      5. Z-score using IS mean/std applied to full series.

    Result: composite_z > 0 means bullish for ANY instrument (direction auto-resolved).
    """
    fwd_ret = np.log(close.shift(-fwd_h) / close)
    fwd_is  = fwd_ret[is_mask].dropna()

    parts: list[pd.Series] = []
    for col in sigs.columns:
        sig_is  = sigs[col][is_mask].dropna()
        aligned = pd.concat([sig_is, fwd_is], axis=1).dropna()
        if len(aligned) < 30:
            continue
        r, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        if np.isnan(r) or abs(r) < min_ic:
            continue
        parts.append(sigs[col] * float(np.sign(r)) * abs(r))

    if not parts:
        return pd.Series(0.0, index=sigs.index)

    raw    = pd.concat(parts, axis=1).sum(axis=1)
    is_raw = raw[is_mask].dropna()
    mu, sigma = float(is_raw.mean()), float(is_raw.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=sigs.index)
    return (raw - mu) / sigma


# ── Data loader ────────────────────────────────────────────────────────────────

def load_data(
    instrument: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Load 5m CSV, build all 52 signals, compute ATR size and EOD mask.

    Returns: (sigs, close, df, size, is_eod)
    """
    df    = _load_ohlcv(instrument, TIMEFRAME, data_dir=DATA_DIR, fmt="csv")
    close = df["close"]
    sigs  = build_all_signals(df)

    valid = sigs.notna().any(axis=1) & close.notna()
    sigs  = sigs[valid]
    close = close[valid]
    df    = df.loc[valid]

    atr  = _atr14(df).bfill()
    stop = (STOP_ATR * atr) / close.where(close > 0)
    size = (RISK_PCT / stop.where(stop > 0)).clip(upper=MAX_LEV).fillna(MAX_LEV)

    is_eod = _session_end_mask(close.index)
    return sigs, close, df, size, is_eod


# ── Backtest core ─────────────────────────────────────────────────────────────

def _run(
    close: pd.Series,
    composite_z: pd.Series,
    threshold: float,
    size: pd.Series,
    is_eod: pd.Series,
    fees: float = SPREAD,
    slippage: float = SLIPPAGE,
) -> dict:
    """Long-only VBT backtest. Fixed FWD_H-bar hold + forced EOD flat."""
    entries = composite_z.shift(1).fillna(0.0) > threshold
    # Exit: timed (after FWD_H bars) OR end-of-session (force flat)
    exits   = entries.shift(FWD_H).fillna(False) | is_eod

    pf = vbt.Portfolio.from_signals(
        close,
        entries   = entries.values,
        exits     = exits.values,
        size      = size.values,
        size_type = "percent",
        init_cash = INIT_CASH,
        fees      = fees,
        slippage  = slippage,
        freq      = "5T",
    )
    bar_rets     = pf.returns()
    sharpe       = _sharpe_daily(bar_rets)
    bar_counts   = bar_rets.resample("D").count()
    trading_days = int((bar_counts > 0).sum())
    annual       = _annual_ret(pf, trading_days)
    n_trades     = int(pf.trades.count())

    return {
        "pf":     pf,
        "sharpe": sharpe,
        "annual": annual,
        "ret":    float(pf.total_return()),
        "dd":     float(pf.max_drawdown()),
        "trades": n_trades,
        "wr":     float(pf.trades.win_rate()) if n_trades > 0 else 0.0,
        "value":  pf.value(),
    }


# ── Annual breakdown ───────────────────────────────────────────────────────────

def _annual_table(value_series: pd.Series) -> pd.Series:
    rets = value_series.pct_change().dropna()
    return (
        pd.DataFrame({"ret": rets})
        .assign(year=lambda x: x.index.year)
        .groupby("year")["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
    )


# ── Robustness (Phases 5a-5c) ─────────────────────────────────────────────────

def _trade_rets(pf) -> np.ndarray | None:
    if pf.trades.count() < 10:
        return None
    return pf.trades.records_readable["Return"].values


def monte_carlo_shuffle(pf, oos_bars: int, n: int = MC_N) -> dict:
    """Shuffle per-trade returns N times; test Sharpe distribution robustness."""
    trade_ret = _trade_rets(pf)
    if trade_ret is None:
        return {"error": "Insufficient trades (< 10)", "gate_pass": False}

    oos_years       = oos_bars / BARS_PER_YEAR
    trades_per_year = len(trade_ret) / max(oos_years, 0.01)
    rng             = np.random.default_rng(42)
    sharpes: list[float] = []
    profitable = 0

    for _ in range(n):
        s      = rng.permutation(trade_ret)
        mu, sd = float(s.mean()), float(s.std())
        sh     = mu / sd * np.sqrt(trades_per_year) if sd > 1e-10 else 0.0
        sharpes.append(sh)
        if float(s.sum()) > 0:
            profitable += 1

    arr      = np.array(sharpes)
    pct5     = float(np.percentile(arr, 5))
    prof_pct = profitable / n
    gate     = pct5 > MC_MIN_5PCT_SHARPE and prof_pct > MC_MIN_PROFITABLE_PCT
    return {
        "n_trades":       int(len(trade_ret)),
        "mean_sharpe":    round(float(arr.mean()), 3),
        "pct5_sharpe":    round(pct5, 3),
        "pct95_sharpe":   round(float(np.percentile(arr, 95)), 3),
        "profitable_pct": round(prof_pct, 3),
        "gate_pass":      gate,
    }


def remove_top_n(pf, n: int = TOP_N_REMOVE) -> dict:
    """Drop N largest wins; check remaining arithmetic P&L > 0."""
    trade_ret = _trade_rets(pf)
    if trade_ret is None:
        return {"error": "Insufficient trades (< 10)", "gate_pass": False}
    if len(trade_ret) <= n:
        return {"error": f"Only {len(trade_ret)} trades, need > {n}", "gate_pass": False}
    total   = float(trade_ret.sum())
    top_sum = float(np.sort(trade_ret)[-n:].sum())
    remain  = total - top_sum
    return {
        "total_trades":      int(len(trade_ret)),
        "total_ret_pct":     round(total * 100, 2),
        "top_n_ret_pct":     round(top_sum * 100, 2),
        "remaining_ret_pct": round(remain * 100, 2),
        "gate_pass":         remain > 0,
    }


def stress_triple_slip(
    close: pd.Series,
    gz: pd.Series,
    size: pd.Series,
    threshold: float,
    is_eod: pd.Series,
) -> dict:
    """Re-run OOS at 3x slippage; gate on Sharpe > STRESS_MIN_SHARPE."""
    r = _run(close, gz, threshold, size, is_eod,
             fees=SPREAD, slippage=SLIPPAGE * STRESS_MULT)
    return {
        "sharpe":    round(r["sharpe"], 3),
        "gate_pass": r["sharpe"] > STRESS_MIN_SHARPE,
    }


def run_robustness(
    pf_oos,
    oos_bars: int,
    close_oos: pd.Series,
    gz_oos: pd.Series,
    size_oos: pd.Series,
    threshold: float,
    eod_oos: pd.Series,
) -> dict:
    print(f"\n{'='*W}")
    print("  ROBUSTNESS VALIDATION  (Phases 5a-5c)")
    print(f"{'='*W}")

    # 5a: Monte Carlo shuffle
    print(f"\n  [5a] Monte Carlo (N={MC_N:,} trade shuffles)...")
    mc = monte_carlo_shuffle(pf_oos, oos_bars)
    if "error" in mc:
        print(f"  SKIP: {mc['error']}")
    else:
        lbl = "PASS" if mc["gate_pass"] else "FAIL"
        print(f"  Trades: {mc['n_trades']}  |  Mean Sharpe: {mc['mean_sharpe']:.3f}")
        print(
            f"  5th-pct: {mc['pct5_sharpe']:.3f} (gate > {MC_MIN_5PCT_SHARPE})  "
            f"95th-pct: {mc['pct95_sharpe']:.3f}"
        )
        print(
            f"  Profitable: {mc['profitable_pct']:.1%}"
            f" (gate > {MC_MIN_PROFITABLE_PCT:.0%})  --> [{lbl}]"
        )

    # 5b: Remove top-N trades
    print(f"\n  [5b] Remove top {TOP_N_REMOVE} winning trades...")
    topn = remove_top_n(pf_oos)
    if "error" in topn:
        print(f"  SKIP: {topn['error']}")
    else:
        lbl = "PASS" if topn["gate_pass"] else "FAIL"
        print(
            f"  Total: {topn['total_ret_pct']:.1f}%  |  "
            f"Top-{TOP_N_REMOVE}: {topn['top_n_ret_pct']:.1f}%  |  "
            f"Remaining: {topn['remaining_ret_pct']:.1f}%  --> [{lbl}]"
        )

    # 5c: 3x slippage stress
    slip3_bps = round(SLIPPAGE * STRESS_MULT * 1e4, 1)
    print(f"\n  [5c] 3x slippage stress ({slip3_bps} bps slippage per fill)...")
    stress = stress_triple_slip(close_oos, gz_oos, size_oos, threshold, eod_oos)
    lbl = "PASS" if stress["gate_pass"] else "FAIL"
    print(f"  OOS Sharpe: {stress['sharpe']:.3f} (gate > {STRESS_MIN_SHARPE})  --> [{lbl}]")

    all_pass = mc["gate_pass"] and topn["gate_pass"] and stress["gate_pass"]
    print(f"\n  {'─'*60}")
    print("  GATE SUMMARY")
    print(f"  {'─'*60}")
    print(f"  [{'PASS' if mc['gate_pass'] else 'FAIL'}]  "
          f"MC 5th-pct Sharpe > {MC_MIN_5PCT_SHARPE},"
          f" > {MC_MIN_PROFITABLE_PCT:.0%} profitable")
    print(f"  [{'PASS' if topn['gate_pass'] else 'FAIL'}]  "
          f"Remove top {TOP_N_REMOVE}: remaining P&L > 0")
    print(f"  [{'PASS' if stress['gate_pass'] else 'FAIL'}]  "
          f"3x slippage Sharpe > {STRESS_MIN_SHARPE}")
    verdict = "ALL GATES PASS" if all_pass else "GATES FAILED -- do not proceed"
    print(f"\n  Verdict: {verdict}")

    return {
        "mc_pct5": mc.get("pct5_sharpe"),
        "mc_prof":  mc.get("profitable_pct"),
        "mc_pass":  mc["gate_pass"],
        "topn_remain": topn.get("remaining_ret_pct"),
        "topn_pass":   topn["gate_pass"],
        "stress_sh":   stress.get("sharpe"),
        "stress_pass": stress["gate_pass"],
        "all_pass":    all_pass,
    }


# ── Walk-Forward Optimisation ─────────────────────────────────────────────────

def run_wfo(
    sigs: pd.DataFrame,
    close: pd.Series,
    size: pd.Series,
    is_eod: pd.Series,
    instrument: str,
) -> None:
    n_months_is  = WFO_IS  // (78 * 21)
    n_months_oos = WFO_OOS // (78 * 21)
    print(f"\n{'='*W}")
    print(f"  {instrument} 5M -- WALK-FORWARD OPTIMISATION")
    print(f"  IS: {WFO_IS} bars (~{n_months_is}mo)  "
          f"OOS: {WFO_OOS} bars (~{n_months_oos}mo)  Step: {WFO_STEP} bars")
    print(f"{'='*W}")

    n = len(sigs)
    folds: list[tuple[int, int, int, int]] = []
    start = 0
    while start + WFO_IS + WFO_OOS <= n:
        folds.append((start, start + WFO_IS, start + WFO_IS, start + WFO_IS + WFO_OOS))
        start += WFO_STEP

    n_folds = len(folds)
    if n_folds < 3:
        print(f"  WARNING: only {n_folds} folds available -- WFO not meaningful.")
        return

    print(
        f"  Folds: {n_folds}  "
        f"(first OOS start: {sigs.index[folds[0][2]].date()}  "
        f"last OOS end: {sigs.index[folds[-1][3]-1].date()})"
    )

    fold_results:  list[dict]      = []
    stitched_rets: list[pd.Series] = []

    print(
        f"\n  {'Fold':>5}  {'Thr':>5}  {'IS-Sh':>7}  "
        f"{'OOS-Sh':>7}  {'OOS-Ann':>9}  {'OOS-DD':>8}  {'Trades':>7}"
    )
    print("  " + "-" * 60)

    for i, (is0, is1, oos0, oos1) in enumerate(folds):
        fold_sigs  = sigs.iloc[is0:oos1]
        fold_close = close.iloc[is0:oos1]
        fold_size  = size.iloc[is0:oos1]
        fold_eod   = is_eod.iloc[is0:oos1]

        is_m = pd.Series(False, index=fold_sigs.index)
        is_m.iloc[:is1 - is0] = True

        gz = build_composite_for_fold(fold_sigs, fold_close, is_m)

        gz_is  = gz.iloc[:is1 - is0]
        gz_oos = gz.iloc[is1 - is0:]

        # IS threshold selection (no look-ahead into OOS)
        best_thr, best_sh_is = THRESHOLDS[0], -np.inf
        for thr in THRESHOLDS:
            r_is = _run(fold_close.iloc[:is1-is0], gz_is, thr,
                        fold_size.iloc[:is1-is0], fold_eod.iloc[:is1-is0])
            if r_is["sharpe"] > best_sh_is:
                best_sh_is, best_thr = r_is["sharpe"], thr

        r_oos = _run(fold_close.iloc[is1-is0:], gz_oos, best_thr,
                     fold_size.iloc[is1-is0:], fold_eod.iloc[is1-is0:])

        fold_results.append({
            "fold":      i + 1,
            "is_start":  sigs.index[is0].date(),
            "is_end":    sigs.index[is1 - 1].date(),
            "oos_start": sigs.index[oos0].date(),
            "oos_end":   sigs.index[oos1 - 1].date(),
            "threshold": best_thr,
            "is_sharpe": round(best_sh_is, 3),
            "oos_sharpe": round(r_oos["sharpe"], 3),
            "oos_annual": round(float(r_oos["annual"]), 4),
            "oos_dd":    round(float(r_oos["dd"]), 4),
            "oos_trades": r_oos["trades"],
        })
        stitched_rets.append(r_oos["pf"].returns())

        print(
            f"  {i+1:>5}  {best_thr:>5.2f}  {best_sh_is:>+7.3f}"
            f"  {r_oos['sharpe']:>+7.3f}  {r_oos['annual']:>+9.1%}"
            f"  {r_oos['dd']:>+8.1%}  {r_oos['trades']:>7}"
        )

    # Stitch OOS equity curve
    stitched = pd.concat(stitched_rets).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    stitched_sharpe = _sharpe_daily(stitched)

    oos_sharpes = [f["oos_sharpe"] for f in fold_results]
    is_sharpes  = [f["is_sharpe"]  for f in fold_results]
    avg_is  = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    pct_pos = sum(s > 0   for s in oos_sharpes) / n_folds
    pct_05  = sum(s > 0.5 for s in oos_sharpes) / n_folds
    worst   = float(min(oos_sharpes))
    parity  = avg_oos / avg_is if avg_is > 1e-6 else 0.0
    consec = max_consec = 0
    for s in oos_sharpes:
        if s <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    print(f"\n{'='*W}")
    print("  WFO SUMMARY")
    print(f"{'='*W}")
    print(f"  Folds evaluated         : {n_folds}")
    print(f"  Avg IS Sharpe           : {avg_is:>+.3f}")
    print(f"  Avg OOS Sharpe          : {avg_oos:>+.3f}")
    print(f"  Stitched OOS Sharpe     : {stitched_sharpe:>+.3f}")
    print(f"  % folds OOS Sharpe > 0  : {pct_pos:.0%}")
    print(f"  % folds OOS Sharpe > 0.5: {pct_05:.0%}")
    print(f"  Worst fold Sharpe       : {worst:>+.3f}")
    print(f"  IS/OOS parity           : {parity:.3f}")
    print(f"  Max consec. neg folds   : {max_consec}")

    g1 = pct_pos >= WFO_GATE_PCT_POS
    g2 = stitched_sharpe >= WFO_GATE_STITCHED
    g3 = parity >= WFO_GATE_PARITY
    g4 = max_consec <= WFO_MAX_CONSEC_NEG

    print(f"\n  {'─'*60}")
    print("  WFO GATES")
    print(f"  {'─'*60}")
    print(f"  [{'PASS' if g1 else 'FAIL'}]  >= {WFO_GATE_PCT_POS:.0%} folds OOS Sharpe > 0  "
          f"(actual: {pct_pos:.0%})")
    print(f"  [{'PASS' if g2 else 'FAIL'}]  Stitched OOS Sharpe >= {WFO_GATE_STITCHED}  "
          f"(actual: {stitched_sharpe:+.3f})")
    print(f"  [{'PASS' if g3 else 'FAIL'}]  IS/OOS parity >= {WFO_GATE_PARITY}  "
          f"(actual: {parity:.3f})")
    print(f"  [{'PASS' if g4 else 'FAIL'}]  Max consec. neg <= {WFO_MAX_CONSEC_NEG}  "
          f"(actual: {max_consec})")

    all_wfo = all([g1, g2, g3, g4])
    verdict = "ALL WFO GATES PASS" if all_wfo else "WFO FAILED -- investigate overfitting"
    print(f"\n  Verdict: {verdict}")

    wfo_path = REPORTS / f"{instrument.lower()}_5m_wfo.csv"
    pd.DataFrame(fold_results).to_csv(wfo_path, index=False)
    print(f"\n  WFO folds saved: {wfo_path}")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run(instrument: str, do_sweep: bool = False, do_robust: bool = True) -> None:
    print("\n" + "=" * W)
    print(f"  {instrument} 5M -- IC-WEIGHTED COMPOSITE LONG-ONLY STRATEGY")
    print(
        f"  FWD_H: {FWD_H} bars (5hr hold)  |  "
        f"Spread: {SPREAD*1e4:.0f}bps/fill  |  "
        f"Slip: {SLIPPAGE*1e4:.0f}bps/fill  |  "
        f"MaxLev: {MAX_LEV}x"
    )
    print("=" * W)

    print(f"\nLoading {instrument} 5m data and computing 52 signals...")
    sigs, close, df, size, is_eod = load_data(instrument)
    n    = len(sigs)
    is_n = int(n * IS_RATIO)

    is_mask  = pd.Series(False, index=sigs.index)
    is_mask.iloc[:is_n] = True
    oos_mask = ~is_mask

    print(f"  Total bars : {n:,}")
    print(
        f"  IS  bars   : {is_n:,}  "
        f"({sigs.index[0].date()} -> {sigs.index[is_n-1].date()})"
    )
    print(
        f"  OOS bars   : {n - is_n:,}  "
        f"({sigs.index[is_n].date()} -> {sigs.index[-1].date()})"
    )

    print("\n  Building IS-calibrated composite signal (IC-weighted, all 52 signals)...")
    gz = build_composite_for_fold(sigs, close, is_mask)

    close_is  = close[is_mask];  gz_is  = gz[is_mask]
    size_is   = size[is_mask];   eod_is = is_eod[is_mask]
    close_oos = close[oos_mask]; gz_oos = gz[oos_mask]
    size_oos  = size[oos_mask];  eod_oos = is_eod[oos_mask]

    bh_oos_days = int((close_oos.resample("D").count() > 0).sum())
    bh_ret = float(close_oos.iloc[-1] / close_oos.iloc[0] - 1)
    bh_ann = float((1 + bh_ret) ** (252 / max(bh_oos_days, 1)) - 1)
    print(f"\n  Buy & Hold OOS: {bh_ret:+.1%} total  ({bh_ann:+.1%} ann.)")

    # OOS threshold sweep
    if do_sweep:
        print(f"\n{'='*W}")
        print("  THRESHOLD SWEEP (OOS)")
        print(
            f"  {'Thresh':>7}  {'Sharpe':>8}  {'Annual':>8}  "
            f"{'MaxDD':>8}  {'Trades':>7}  {'WR%':>6}"
        )
        print("  " + "-" * 56)
        for thr in THRESHOLDS:
            r = _run(close_oos, gz_oos, thr, size_oos, eod_oos)
            print(
                f"  {thr:>7.2f}  {r['sharpe']:>+8.3f}  {r['annual']:>+8.1%}  "
                f"{r['dd']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
            )

    # IS threshold selection (no look-ahead into OOS)
    best_thr, best_sh_is = THRESHOLDS[0], -np.inf
    for thr in THRESHOLDS:
        r_is = _run(close_is, gz_is, thr, size_is, eod_is)
        if r_is["sharpe"] > best_sh_is:
            best_sh_is, best_thr = r_is["sharpe"], thr

    r = _run(close_oos, gz_oos, best_thr, size_oos, eod_oos)

    print(f"\n{'='*W}")
    print(f"  OOS RESULTS  (IS threshold = {best_thr:.2f}z  |  IS Sharpe = {best_sh_is:+.3f})")
    print(f"{'='*W}")
    print(f"  {'OOS Sharpe Ratio':<26}: {r['sharpe']:>+.3f}")
    print(f"  {'OOS Annualised Return':<26}: {r['annual']:>+.1%}")
    print(f"  {'OOS Total Return':<26}: {r['ret']:>+.1%}")
    print(f"  {'OOS Max Drawdown':<26}: {r['dd']:>+.1%}")
    print(f"  {'Trades':<26}: {r['trades']}")
    print(f"  {'Win Rate':<26}: {r['wr']*100:.1f}%")
    print(f"  {'B&H Annualised':<26}: {bh_ann:>+.1%}")

    # Annual breakdown
    annual = _annual_table(r["value"])
    bh_annual = _annual_table(close_oos / close_oos.iloc[0] * INIT_CASH)
    print(f"\n  {'Year':<8}  {'Strategy':>14}  {'B&H':>10}  {'Alpha':>10}")
    print("  " + "-" * 48)
    for yr, val in annual.items():
        bh_yr  = bh_annual.get(yr, np.nan)
        alpha  = val - bh_yr if not np.isnan(bh_yr) else np.nan
        bh_str = f"{bh_yr:>+10.1%}" if not np.isnan(bh_yr) else "       N/A"
        al_str = f"{alpha:>+10.1%}" if not np.isnan(alpha) else "       N/A"
        print(f"  {yr:<8}  {val:>+14.1%}  {bh_str}  {al_str}")

    # Save Phase 3 results
    out = REPORTS / f"{instrument.lower()}_5m_strategy.csv"
    pd.DataFrame([{
        "instrument":  instrument,
        "threshold":   best_thr,
        "is_sharpe":   round(best_sh_is, 3),
        "oos_sharpe":  round(r["sharpe"], 3),
        "oos_annual":  round(r["annual"], 4),
        "oos_ret":     round(r["ret"], 4),
        "oos_dd":      round(r["dd"], 4),
        "trades":      r["trades"],
        "wr":          round(r["wr"], 3),
        "bh_annual":   round(bh_ann, 4),
    }]).to_csv(out, index=False)
    print(f"\n  Results saved: {out}")

    # Robustness
    if do_robust:
        run_robustness(
            pf_oos=r["pf"],
            oos_bars=len(close_oos),
            close_oos=close_oos,
            gz_oos=gz_oos,
            size_oos=size_oos,
            threshold=best_thr,
            eod_oos=eod_oos,
        )

    # WFO
    run_wfo(sigs, close, size, is_eod, instrument)

    # Equity chart (OOS only)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        port_val = r["value"]
        bh_val   = close_oos / close_oos.iloc[0] * INIT_CASH

        def norm(v: pd.Series) -> pd.Series:
            return v / v.iloc[0] * 100

        def dd_curve(v: pd.Series) -> pd.Series:
            pk = v.cummax()
            return (v - pk) / pk * 100

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.70, 0.30],
            subplot_titles=["Equity (OOS, rebased 100)", "Drawdown %"],
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(
            x=port_val.index, y=norm(port_val),
            name="Strategy", line=dict(color="#00E5FF", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bh_val.index, y=norm(bh_val),
            name="B&H", line=dict(color="#FFD600", width=1.5, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=port_val.index, y=dd_curve(port_val),
            fill="tozeroy", name="DD",
            line=dict(color="#FF1744"), fillcolor="rgba(255,23,68,0.18)",
        ), row=2, col=1)
        fig.update_layout(
            title=f"{instrument} 5m | IC-Weighted Composite | OOS",
            height=720, template="plotly_dark", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Value (rebased 100)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %",          row=2, col=1)

        html_out = REPORTS / f"{instrument.lower()}_5m_strategy.html"
        fig.write_html(str(html_out))
        print(f"\n  Chart saved: {html_out}")
    except Exception as e:
        print(f"\n  [INFO] Chart skipped: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="5m Intraday IC-Weighted Strategy")
    parser.add_argument(
        "--instrument", default="EOG",
        help="Instrument ticker (must exist in data/databento/{INST}_1yr_5m.csv)"
    )
    parser.add_argument("--sweep",     action="store_true", help="Print OOS threshold sweep")
    parser.add_argument("--no-robust", action="store_true", help="Skip robustness tests")
    args = parser.parse_args()
    run(
        instrument = args.instrument,
        do_sweep   = args.sweep,
        do_robust  = not args.no_robust,
    )


if __name__ == "__main__":
    main()
