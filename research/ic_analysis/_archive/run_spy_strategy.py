"""run_spy_strategy.py -- SPY Three-Signal Regime-Gated Strategy.

Combines three IC-validated signals on SPY Daily:
  1. macd_norm    -- 60-day trend (IC=+0.149, RcntIC=+0.164, MFE=+0.251)
  2. rsi_dev      -- 60-day mean-reversion/dip-buy (IC=+0.090, MFE=+0.155)
  3. momentum_5   -- 5-day short-term fade (IC=-0.097, ICIR=-1.27, Mono=-0.87)

Architecture:
  - ADX regime gate: Trending (ADX > 25) vs Ranging (ADX < 20)
  - Trending regime: macd_norm + rsi_dev composite (long-only, 60-day hold target)
  - Ranging regime: momentum_5 mean-reversion (long when 5-day momentum negative)
  - Each signal IS-calibrated for sign and z-score normalisation (70/30 IS/OOS)
  - Long-only: SPY is a long-only instrument (no shorting ETFs in retail)

Cost model (applied per fill — VBT charges entry AND exit):
  - Spread:   5 bps per fill
  - Slippage: 2 bps per fill
  → Effective round-trip cost: ~14 bps (2 × spread + 2 × slippage)

Position sizing:
  - 1% risk per trade with 1.5× ATR14 stop
  - Max leverage: 2.0× (realistic for ETF retail)

Phases run:
  Phase 3:  IS/OOS backtest (70/30 split), threshold sweep, annual breakdown
  Phase 4a: Monte Carlo (N=1,000 trade shuffles)
  Phase 4b: Remove top-10 winning trades
  Phase 4c: 3× slippage stress
  Phase 4d: WFO consecutive-folds check (reads spy_wfo.csv from run_spy_wfo.py)

Usage:
  uv run python research/ic_analysis/run_spy_strategy.py
  uv run python research/ic_analysis/run_spy_strategy.py --sweep
  uv run python research/ic_analysis/run_spy_strategy.py --no-regime
  uv run python research/ic_analysis/run_spy_strategy.py --no-robust
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

# ── Strategy config ──────────────────────────────────────────────────────────

INSTRUMENT = "SPY"
TIMEFRAME  = "D"
IS_RATIO   = 0.70

INIT_CASH  = 10_000.0
RISK_PCT   = 0.01
STOP_ATR   = 1.5
MAX_LEV    = 2.0
FREQ       = "d"

SPREAD   = 0.0005   # 5 bps per fill (charged on entry AND exit by VBT)
SLIPPAGE = 0.0002   # 2 bps per fill

ADX_RANGING  = 20.0
ADX_TRENDING = 25.0

TREND_SIGNALS    = ["macd_norm", "rsi_dev"]
REVERSION_SIGNAL = "momentum_5"

THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# ── Robustness gates (long-only daily; calibrated below FX L+S targets) ──────

MC_N                  = 1_000
MC_MIN_5PCT_SHARPE    = 0.30   # FX L+S uses 0.50; long-only daily is harder
MC_MIN_PROFITABLE_PCT = 0.70   # FX uses 0.80
TOP_N_REMOVE          = 10
STRESS_SLIPPAGE_MULT  = 3.0
STRESS_MIN_SHARPE     = 0.30
WFO_MAX_CONSEC_NEG    = 3      # daily folds have more noise than H1 FX folds

W       = 84
REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()


def _ic_sign(signal: pd.Series, fwd: pd.Series) -> float:
    """Spearman IC sign computed on IS data only."""
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


def _build_trend_composite(
    signals: pd.DataFrame, close: pd.Series, is_mask: pd.Series
) -> pd.Series:
    """Equal-weight composite of trend signals; IS sign-calibrated."""
    fwd_is = np.log(close.shift(-1) / close)[is_mask]
    parts = []
    for name in TREND_SIGNALS:
        if name not in signals.columns:
            continue
        sign = _ic_sign(signals[name][is_mask], fwd_is)
        parts.append(signals[name] * sign)
    if not parts:
        return pd.Series(0.0, index=signals.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def _build_reversion_composite(
    signals: pd.DataFrame, close: pd.Series, is_mask: pd.Series
) -> pd.Series:
    """momentum_5 sign-flipped for mean-reversion (buy dips after down 5 days)."""
    if REVERSION_SIGNAL not in signals.columns:
        return pd.Series(0.0, index=signals.index)
    fwd_h5_is = np.log(close.shift(-5) / close)[is_mask]
    sign = _ic_sign(signals[REVERSION_SIGNAL][is_mask], fwd_h5_is)
    return signals[REVERSION_SIGNAL] * sign


def _size_series(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    """ATR14-based position size: 1% risk / (1.5× ATR14 stop), capped at MAX_LEV."""
    atr = _atr14(df)
    stop_pct = (STOP_ATR * atr) / close.where(close > 0)
    return (RISK_PCT / stop_pct.where(stop_pct > 0)).clip(upper=MAX_LEV).fillna(0.0)


# ── Composite builder (reusable by WFO) ──────────────────────────────────────

def build_composite_for_fold(
    sigs: pd.DataFrame,
    close: pd.Series,
    is_mask: pd.Series,
    use_regime: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Build regime and composite_z for a given IS mask.

    All calibration (IC sign, z-score stats) uses only IS bars.
    Returns (composite_z, regime) with the same index as sigs.
    """
    adx    = sigs["adx_14"]
    regime = pd.Series("neutral", index=sigs.index)
    regime[adx < ADX_RANGING]  = "ranging"
    regime[adx > ADX_TRENDING] = "trending"

    trend_comp  = _build_trend_composite(sigs, close, is_mask)
    revert_comp = _build_reversion_composite(sigs, close, is_mask)

    is_trending = is_mask & (regime == "trending")
    is_ranging  = is_mask & (regime == "ranging")

    trend_z  = _zscore_is(trend_comp,  is_trending)
    revert_z = _zscore_is(revert_comp, is_ranging)

    if use_regime:
        composite_z = pd.Series(0.0, index=sigs.index)
        composite_z[regime == "trending"] = trend_z[regime == "trending"]
        composite_z[regime == "ranging"]  = revert_z[regime == "ranging"]
    else:
        fwd_is = np.log(close.shift(-1) / close)[is_mask]
        parts = []
        for name in [*TREND_SIGNALS, REVERSION_SIGNAL]:
            if name not in sigs.columns:
                continue
            sign = _ic_sign(sigs[name][is_mask], fwd_is)
            parts.append(sigs[name] * sign)
        raw = pd.concat(parts, axis=1).mean(axis=1)
        composite_z = _zscore_is(raw, is_mask)

    return composite_z, regime


# ── Data loader (shared with WFO) ─────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load OHLCV, build all signals, return (sigs, close, df, size).

    Filters rows where all signals are NaN or close is NaN.
    Position size is computed once on the full dataset.
    """
    df    = _load_ohlcv(INSTRUMENT, TIMEFRAME)
    close = df["close"]
    sigs  = build_all_signals(df)

    valid = sigs.notna().any(axis=1) & close.notna()
    sigs  = sigs[valid]
    close = close[valid]
    df    = df.loc[valid]

    size = _size_series(df, close)
    return sigs, close, df, size


# ── Phase 3: IS/OOS prepare ───────────────────────────────────────────────────

def prepare(use_regime: bool = True) -> dict:
    """70/30 IS/OOS split; build composite calibrated to IS only."""
    sigs, close, df, size = load_data()
    n    = len(sigs)
    is_n = int(n * IS_RATIO)

    is_mask = pd.Series(False, index=sigs.index)
    is_mask.iloc[:is_n] = True

    composite_z, regime = build_composite_for_fold(sigs, close, is_mask, use_regime)

    return {
        "sigs":        sigs,
        "close":       close,
        "df":          df,
        "composite_z": composite_z,
        "size":        size,
        "is_mask":     is_mask,
        "oos_mask":    ~is_mask,
        "regime":      regime,
        "is_n":        is_n,
        "n":           n,
        "index":       sigs.index,
    }


# ── Backtest core ─────────────────────────────────────────────────────────────

def _run_long(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    size: pd.Series,
    fees: float = SPREAD,
    slippage: float = SLIPPAGE,
) -> dict:
    """Long-only VBT backtest with 1-bar execution delay.

    Returns stats dict including raw `pf` object for downstream robustness tests.
    """
    sig = signal_z.shift(1).fillna(0.0)   # next-bar execution

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


# ── Annual breakdown ──────────────────────────────────────────────────────────

def _annual_table(value_series: pd.Series) -> pd.Series:
    rets = value_series.pct_change().dropna()
    return (
        pd.DataFrame({"ret": rets})
        .assign(year=lambda x: x.index.year)
        .groupby("year")["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
    )


# ── Robustness functions (Phases 4a–4d) ──────────────────────────────────────

def _trade_returns(pf) -> np.ndarray | None:
    """Per-trade % returns from VBT portfolio (normalised, not dollar P&L)."""
    if pf.trades.count() < 10:
        return None
    return pf.trades.records_readable["Return"].values


def monte_carlo_shuffle(pf, oos_bars: int, n: int = MC_N) -> dict:
    """Shuffle per-trade % returns N times; compute Sharpe distribution.

    Uses per-trade % returns (VBT Return column) — not cumulative P&L — so
    Sharpe is not distorted by compounding order.  Annualises by trades-per-year
    derived from the OOS window length.

    Gate: 5th-pct Sharpe > MC_MIN_5PCT_SHARPE AND > MC_MIN_PROFITABLE_PCT sims profitable.
    """
    trade_ret = _trade_returns(pf)
    if trade_ret is None:
        return {"error": "Insufficient trades (< 10)", "gate_pass": False}

    oos_years       = oos_bars / 252.0
    trades_per_year = len(trade_ret) / max(oos_years, 0.1)
    rng             = np.random.default_rng(42)
    sharpes: list[float] = []
    profitable = 0

    for _ in range(n):
        shuffled   = rng.permutation(trade_ret)
        mu, sigma  = float(shuffled.mean()), float(shuffled.std())
        sh         = mu / sigma * np.sqrt(trades_per_year) if sigma > 1e-10 else 0.0
        sharpes.append(sh)
        if float(shuffled.sum()) > 0:
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
    """Drop N largest winning trades; check arithmetic sum of remaining > 0.

    Uses per-trade % returns so the test is agnostic to compounding order.
    Gate: remaining P&L sum > 0.
    """
    trade_ret = _trade_returns(pf)
    if trade_ret is None:
        return {"error": "Insufficient trades", "gate_pass": False}
    if len(trade_ret) < n + 1:
        return {
            "error": f"Only {len(trade_ret)} trades — need > {n}",
            "gate_pass": False,
        }

    total    = float(trade_ret.sum())
    top_sum  = float(np.sort(trade_ret)[-n:].sum())
    remain   = total - top_sum

    return {
        "total_trades":      int(len(trade_ret)),
        "total_ret_pct":     round(total * 100, 2),
        "top_n_ret_pct":     round(top_sum * 100, 2),
        "remaining_ret_pct": round(remain * 100, 2),
        "gate_pass":         remain > 0,
    }


def stress_triple_slippage(
    close_oos: pd.Series,
    composite_z_oos: pd.Series,
    size_oos: pd.Series,
    threshold: float,
) -> dict:
    """Re-run OOS with 3× slippage; check Sharpe still above STRESS_MIN_SHARPE."""
    r = _run_long(
        close_oos,
        composite_z_oos,
        threshold,
        size_oos,
        fees=SPREAD,
        slippage=SLIPPAGE * STRESS_SLIPPAGE_MULT,
    )
    stress_slip_bps = round(SLIPPAGE * STRESS_SLIPPAGE_MULT * 10_000, 1)
    return {
        "sharpe":         round(r["sharpe"], 3),
        "stress_slip_bps": stress_slip_bps,
        "gate_pass":      r["sharpe"] > STRESS_MIN_SHARPE,
    }


def check_wfo_consecutive() -> dict:
    """Read spy_wfo.csv (from run_spy_wfo.py) and check max consecutive neg folds.

    Gate: max consecutive negative folds <= WFO_MAX_CONSEC_NEG.
    """
    wfo_path = REPORTS / "spy_wfo.csv"
    if not wfo_path.exists():
        return {
            "error": "spy_wfo.csv not found — run run_spy_wfo.py first.",
            "gate_pass": False,
        }

    df = pd.read_csv(wfo_path)
    if "oos_sharpe" not in df.columns:
        return {"error": "oos_sharpe column missing in spy_wfo.csv", "gate_pass": False}

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
        "positive_pct":   round(n_pos / n_folds, 3) if n_folds else 0.0,
        "max_consec_neg": max_consec,
        "gate_pass":      max_consec <= WFO_MAX_CONSEC_NEG,
    }


def run_robustness(
    pf_oos,
    oos_bars: int,
    close_oos: pd.Series,
    composite_z_oos: pd.Series,
    size_oos: pd.Series,
    threshold: float,
) -> dict:
    """Run all 4 robustness gates. Returns gate summary dict."""
    print(f"\n{'=' * W}")
    print("  ROBUSTNESS VALIDATION  (Phases 4a–4d)")
    print(f"{'=' * W}")

    # 4a: Monte Carlo
    print(f"\n  [4a] Monte Carlo (N={MC_N:,} trade shuffles)...")
    mc = monte_carlo_shuffle(pf_oos, oos_bars)
    if "error" in mc:
        print(f"  SKIP: {mc['error']}")
    else:
        mc_lbl = "PASS" if mc["gate_pass"] else "FAIL"
        print(f"  Trades: {mc['n_trades']}  |  Mean Sharpe: {mc['mean_sharpe']:.3f}")
        print(
            f"  5th-pct: {mc['pct5_sharpe']:.3f} (gate > {MC_MIN_5PCT_SHARPE})  "
            f"95th-pct: {mc['pct95_sharpe']:.3f}"
        )
        print(
            f"  Profitable sims: {mc['profitable_pct']:.1%}"
            f" (gate > {MC_MIN_PROFITABLE_PCT:.0%})  --> [{mc_lbl}]"
        )

    # 4b: Remove top-N
    print(f"\n  [4b] Remove top {TOP_N_REMOVE} winning trades...")
    top_n = remove_top_n(pf_oos)
    if "error" in top_n:
        print(f"  SKIP: {top_n['error']}")
    else:
        tn_lbl = "PASS" if top_n["gate_pass"] else "FAIL"
        print(
            f"  Total: {top_n['total_ret_pct']:.1f}%  |  "
            f"Top-{TOP_N_REMOVE}: {top_n['top_n_ret_pct']:.1f}%"
        )
        print(f"  Remaining: {top_n['remaining_ret_pct']:.1f}%  --> [{tn_lbl}]")

    # 4c: 3× slippage
    slip_bps = round(SLIPPAGE * STRESS_SLIPPAGE_MULT * 10_000, 1)
    print(f"\n  [4c] 3× slippage stress ({slip_bps} bps slippage per fill)...")
    stress = stress_triple_slippage(close_oos, composite_z_oos, size_oos, threshold)
    st_lbl = "PASS" if stress["gate_pass"] else "FAIL"
    print(
        f"  OOS Sharpe: {stress['sharpe']:.3f}"
        f" (gate > {STRESS_MIN_SHARPE})  --> [{st_lbl}]"
    )

    # 4d: WFO consecutive folds
    print("\n  [4d] WFO consecutive negative folds (spy_wfo.csv)...")
    wfo = check_wfo_consecutive()
    if "error" in wfo:
        print(f"  SKIP: {wfo['error']}")
    else:
        wf_lbl = "PASS" if wfo["gate_pass"] else "FAIL"
        print(
            f"  Folds: {wfo['n_folds']}  |  "
            f"Positive: {wfo['positive_folds']} ({wfo['positive_pct']:.0%})"
        )
        print(
            f"  Max consec. neg: {wfo['max_consec_neg']}"
            f" (gate <= {WFO_MAX_CONSEC_NEG})  --> [{wf_lbl}]"
        )

    all_pass = all([
        mc["gate_pass"], top_n["gate_pass"], stress["gate_pass"], wfo["gate_pass"]
    ])

    print(f"\n  {'─' * 60}")
    print("  GATE SUMMARY")
    print(f"  {'─' * 60}")
    gates = {
        (f"Monte Carlo 5th-pct > {MC_MIN_5PCT_SHARPE},"
         f" >{MC_MIN_PROFITABLE_PCT:.0%} profitable"):    mc["gate_pass"],
        f"Remove top {TOP_N_REMOVE}: remaining P&L > 0":  top_n["gate_pass"],
        f"3× slippage: Sharpe > {STRESS_MIN_SHARPE}":     stress["gate_pass"],
        f"WFO max consec. neg <= {WFO_MAX_CONSEC_NEG}":   wfo["gate_pass"],
    }
    for gate_name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {gate_name}")

    verdict = (
        "ALL GATES PASS -- cleared for live implementation"
        if all_pass
        else "GATES FAILED -- do not proceed to live"
    )
    print(f"\n  Verdict: {verdict}")

    return {
        "mc_pct5":        mc.get("pct5_sharpe"),
        "mc_prof":        mc.get("profitable_pct"),
        "mc_pass":        mc["gate_pass"],
        "topn_remain":    top_n.get("remaining_ret_pct"),
        "topn_pass":      top_n["gate_pass"],
        "stress_sharpe":  stress.get("sharpe"),
        "stress_pass":    stress["gate_pass"],
        "wfo_max_consec": wfo.get("max_consec_neg"),
        "wfo_pass":       wfo["gate_pass"],
        "all_pass":       all_pass,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    do_sweep: bool = False,
    use_regime: bool = True,
    do_robust: bool = True,
) -> None:
    mode_label = "REGIME-GATED" if use_regime else "UNCONDITIONAL (NO REGIME)"
    print("\n" + "=" * W)
    print(f"  SPY DAILY -- THREE-SIGNAL {mode_label} LONG-ONLY STRATEGY")
    print("  Signals: macd_norm(h=60) + rsi_dev(h=60) + momentum_5(h=5, faded)")
    print(
        f"  Spread: {SPREAD*1e4:.0f}bps/fill  Slippage: {SLIPPAGE*1e4:.0f}bps/fill  "
        f"Risk/trade: {RISK_PCT*100:.1f}%  Stop: {STOP_ATR}×ATR14  MaxLev: {MAX_LEV}×"
    )
    print("=" * W)

    print("\nPreparing SPY data and computing signals...")
    data = prepare(use_regime=use_regime)

    oos        = data["oos_mask"]
    is_mask    = data["is_mask"]
    close_full = data["close"]
    close_oos  = close_full[oos]
    gz_oos     = data["composite_z"][oos]
    size_oos   = data["size"][oos]
    regime_oos = data["regime"][oos]

    print(f"  Total bars : {data['n']:,}")
    print(
        f"  IS  bars   : {data['is_n']:,}  "
        f"({data['index'][0].date()} → {data['index'][data['is_n']-1].date()})"
    )
    print(
        f"  OOS bars   : {oos.sum():,}  "
        f"({data['index'][data['is_n']].date()} → {data['index'][-1].date()})"
    )
    print()
    for lbl in ["ranging", "neutral", "trending"]:
        pct = (regime_oos == lbl).mean() * 100
        print(f"  {lbl.capitalize():<10}: {pct:.0f}% of OOS bars")

    bh_ret = float(close_oos.iloc[-1] / close_oos.iloc[0] - 1)
    n_oos  = len(close_oos)
    bh_ann = float((1 + bh_ret) ** (252 / n_oos) - 1)
    print(f"\n  Buy & Hold OOS return : {bh_ret:+.1%}  ({bh_ann:+.1%} ann.)")

    # Threshold sweep (OOS)
    if do_sweep:
        print(f"\n{'='*W}")
        print("  THRESHOLD SWEEP (OOS)")
        print(
            f"  {'Thresh':>7}  {'Sharpe':>8}  {'Annual':>8}  "
            f"{'MaxDD':>8}  {'Trades':>7}  {'WR%':>6}"
        )
        print("  " + "-" * 55)
        for thr in THRESHOLDS:
            r = _run_long(close_oos, gz_oos, thr, size_oos)
            print(
                f"  {thr:>7.2f}  {r['sharpe']:>+8.3f}  {r['annual']:>+8.1%}  "
                f"{r['dd']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
            )

    # Pick best threshold from IS (no look-ahead)
    best_thr      = THRESHOLDS[0]
    best_sharpe_is = -np.inf
    close_is = close_full[is_mask]
    gz_is    = data["composite_z"][is_mask]
    size_is  = data["size"][is_mask]
    for thr in THRESHOLDS:
        r_is = _run_long(close_is, gz_is, thr, size_is)
        if r_is["sharpe"] > best_sharpe_is:
            best_sharpe_is = r_is["sharpe"]
            best_thr = thr

    # OOS final result
    r = _run_long(close_oos, gz_oos, best_thr, size_oos)

    print(f"\n{'='*W}")
    print(f"  OOS RESULTS  (best IS threshold = {best_thr:.2f}z)")
    print(f"{'='*W}")
    print(f"  {'OOS Sharpe Ratio':<26}: {r['sharpe']:>+.3f}")
    print(f"  {'OOS Annualised Return':<26}: {r['annual']:>+.1%}")
    print(f"  {'OOS Total Return':<26}: {r['ret']:>+.1%}")
    print(f"  {'OOS Max Drawdown':<26}: {r['dd']:>+.1%}")
    print(f"  {'Trades':<26}: {r['trades']}")
    print(f"  {'Win Rate':<26}: {r['wr']*100:.1f}%")
    print(f"  {'B&H Annualised Return':<26}: {bh_ann:>+.1%}")
    print(f"  {'Sharpe vs B&H baseline':<26}: {r['sharpe']:>+.3f} vs ~0.60 (SPY historical)")

    verdict = "✅ PASS" if r["sharpe"] > 0.8 and r["dd"] > -0.25 else "❌ FAIL"
    print(f"\n  Strategy Verdict: {verdict}")

    # Annual breakdown
    annual = _annual_table(r["value"])
    print(f"\n  {'Year':<8}  {'Strategy Return':>18}  {'vs B&H':>10}")
    print("  " + "-" * 42)
    bh_annual = _annual_table(close_oos / close_oos.iloc[0] * INIT_CASH)
    for yr, val in annual.items():
        bh_yr     = bh_annual.get(yr, np.nan)
        alpha     = val - bh_yr if not np.isnan(bh_yr) else np.nan
        alpha_str = f"{alpha:>+10.1%}" if not np.isnan(alpha) else "       N/A"
        print(f"  {yr:<8}  {val:>+18.1%}  {alpha_str}")

    # Save Phase 3 CSV
    out_path = REPORTS / "spy_strategy.csv"
    pd.DataFrame([{
        "instrument": INSTRUMENT,
        "mode":       "regime_gated" if use_regime else "unconditional",
        "threshold":  best_thr,
        "sharpe":     r["sharpe"],
        "annual":     r["annual"],
        "ret":        r["ret"],
        "dd":         r["dd"],
        "trades":     r["trades"],
        "wr":         r["wr"],
        "bh_annual":  bh_ann,
    }]).to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")

    # Robustness tests (Phases 4a–4d)
    if do_robust:
        run_robustness(
            pf_oos          = r["pf"],
            oos_bars        = len(close_oos),
            close_oos       = close_oos,
            composite_z_oos = gz_oos,
            size_oos        = size_oos,
            threshold       = best_thr,
        )

    # Equity chart
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
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.70, 0.30],
            subplot_titles=["Equity Curve (OOS, rebased 100)", "Drawdown %"],
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(
            x=port_val.index, y=norm(port_val),
            name="Strategy", line=dict(color="#00E5FF", width=2.0),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bh_val.index, y=norm(bh_val),
            name="SPY Buy & Hold", line=dict(color="#FFD600", width=1.5, dash="dot"),
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
                f"SPY Daily | macd_norm + rsi_dev + momentum_5 | {mode_label} | OOS"
            ),
            height=720,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Value (rebased 100)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %",          row=2, col=1)

        html_out = REPORTS / "spy_strategy.html"
        fig.write_html(str(html_out))
        print(f"\n  Chart saved  : {html_out}")
    except Exception as e:
        print(f"\n  [INFO] Chart skipped: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SPY Three-Signal IC Strategy")
    parser.add_argument("--sweep",     action="store_true", help="OOS threshold sweep")
    parser.add_argument("--no-regime", action="store_true", help="Disable ADX regime gate")
    parser.add_argument("--no-robust", action="store_true", help="Skip robustness tests")
    args = parser.parse_args()
    run(
        do_sweep   = args.sweep,
        use_regime = not args.no_regime,
        do_robust  = not args.no_robust,
    )


if __name__ == "__main__":
    main()
