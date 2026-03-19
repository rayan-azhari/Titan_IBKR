"""run_spy_strategy.py -- SPY Three-Signal Regime-Gated Strategy.

Combines three IC-validated signals on SPY Daily:
  1. macd_norm    -- 60-day trend (IC=+0.149, RcntIC=+0.164, MFE=+0.251)
  2. rsi_dev      -- 60-day mean-reversion/dip-buy (IC=+0.090, MFE=+0.155)
  3. momentum_5   -- 5-day short-term fade (IC=-0.097, ICIR=-1.27, Mono=-0.87)

Architecture:
  - ADX regime gate: Trending (ADX > 25) vs Ranging (ADX < 20)
  - Trending regime: macd_norm + rsi_dev composite (long-only, 60-day hold target)
  - Ranging regime: momentum_5 mean-reversion (long when 5d momentum negative)
  - Each signal IS-calibrated for sign and z-score normalisation (70/30 IS/OOS)
  - Long-only: SPY is a long-only instrument (no shorting ETFs in retail)

Cost model:
  - Spread:   5 bps (institutional ETF, very tight)
  - Slippage: 2 bps per fill

Position sizing:
  - 1% risk per trade with 1.5x ATR14 stop
  - Max leverage: 2.0x (realistic for ETF retail)

Usage:
  uv run python research/ic_analysis/run_spy_strategy.py
  uv run python research/ic_analysis/run_spy_strategy.py --sweep
  uv run python research/ic_analysis/run_spy_strategy.py --no-regime
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

# ── Config ──────────────────────────────────────────────────────────────────

INSTRUMENT = "SPY"
TIMEFRAME  = "D"
IS_RATIO   = 0.70

INIT_CASH  = 10_000.0
RISK_PCT   = 0.01
STOP_ATR   = 1.5
MAX_LEV    = 2.0          # SPY: low-leverage ETF
FREQ       = "d"

SPREAD     = 0.0005       # 5 bps round-trip (ETF)
SLIPPAGE   = 0.0002        # 2 bps per fill

ADX_RANGING  = 20.0
ADX_TRENDING = 25.0

# Signals validated by IC scanner with their natural horizons
TREND_SIGNALS    = ["macd_norm", "rsi_dev"]   # peak at h=60
REVERSION_SIGNAL = "momentum_5"               # peak at h=5, IC=-0.097, flip sign

THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

W = 84
REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()


def _ic_sign(signal: pd.Series, fwd: pd.Series) -> float:
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0 else 1.0


def _zscore_is(series: pd.Series, is_mask: pd.Series) -> pd.Series:
    """Z-score using only IS mean/std. Applied to full series."""
    is_vals = series[is_mask].dropna()
    mu, sigma = float(is_vals.mean()), float(is_vals.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _build_trend_composite(
    signals: pd.DataFrame, close: pd.Series, is_mask: pd.Series
) -> pd.Series:
    """Equal-weight composite of trend signals, IS sign-calibrated."""
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
    # IC is negative (momentum fades) → flip sign so composite > 0 = buy dip
    return signals[REVERSION_SIGNAL] * sign


# ── Prepare ───────────────────────────────────────────────────────────────────

def prepare(use_regime: bool = True) -> dict:
    df     = _load_ohlcv(INSTRUMENT, TIMEFRAME)
    close  = df["close"]
    sigs   = build_all_signals(df)

    valid = sigs.notna().any(axis=1) & close.notna()
    sigs  = sigs[valid]
    close = close[valid]
    df    = df.loc[valid]
    n     = len(sigs)
    is_n  = int(n * IS_RATIO)

    is_mask = pd.Series(False, index=sigs.index)
    is_mask.iloc[:is_n] = True

    adx    = sigs["adx_14"]
    regime = pd.Series("neutral", index=sigs.index)
    regime[adx < ADX_RANGING]  = "ranging"
    regime[adx > ADX_TRENDING] = "trending"

    # Build composites IS-calibrated
    trend_comp = _build_trend_composite(sigs, close, is_mask)
    revert_comp = _build_reversion_composite(sigs, close, is_mask)

    # IS z-score each composite in its own regime bucket
    is_trending = is_mask & (regime == "trending")
    is_ranging  = is_mask & (regime == "ranging")

    trend_z  = _zscore_is(trend_comp,  is_trending)
    revert_z = _zscore_is(revert_comp, is_ranging)

    if use_regime:
        # Regime-gated composite
        composite_z = pd.Series(0.0, index=sigs.index)
        composite_z[regime == "trending"] = trend_z[regime == "trending"]
        composite_z[regime == "ranging"]  = revert_z[regime == "ranging"]
    else:
        # Unconditional: blend all three signals equally (no regime gate)
        fwd_is = np.log(close.shift(-1) / close)[is_mask]
        parts = []
        for name in [*TREND_SIGNALS, REVERSION_SIGNAL]:
            if name not in sigs.columns:
                continue
            sign = _ic_sign(sigs[name][is_mask], fwd_is)
            parts.append(sigs[name] * sign)
        raw = pd.concat(parts, axis=1).mean(axis=1)
        composite_z = _zscore_is(raw, is_mask)

    size = _size_series(df, close)

    return {
        "close":       close,
        "composite_z": composite_z,
        "size":        size,
        "is_mask":     is_mask,
        "oos_mask":    ~is_mask,
        "regime":      regime,
        "is_n":        is_n,
        "n":           n,
        "index":       sigs.index,
        "trend_z":     trend_z,
        "revert_z":    revert_z,
    }


def _size_series(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    atr = _atr14(df)
    stop_pct = (STOP_ATR * atr) / close.where(close > 0)
    return (RISK_PCT / stop_pct.where(stop_pct > 0)).clip(upper=MAX_LEV).fillna(0.0)


# ── Backtest ──────────────────────────────────────────────────────────────────

def _run_long(
    close: pd.Series, signal_z: pd.Series, threshold: float, size: pd.Series
) -> dict:
    sig = signal_z.shift(1).fillna(0.0)  # 1 bar execution delay

    pf = vbt.Portfolio.from_signals(
        close,
        entries    = sig > threshold,
        exits      = sig <= 0.0,
        size       = size.values,
        size_type  = "percent",
        init_cash  = INIT_CASH,
        fees       = SPREAD,
        slippage   = SLIPPAGE,
        freq       = FREQ,
    )
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "annual": float(pf.annualized_return()),
        "ret":    float(pf.total_return()),
        "dd":     float(pf.max_drawdown()),
        "trades": int(n),
        "wr":     float(pf.trades.win_rate()) if n > 0 else 0.0,
        "value":  pf.value(),
    }


# ── Annual breakdown helper ───────────────────────────────────────────────────

def _annual_table(value_series: pd.Series) -> pd.Series:
    rets = value_series.pct_change().dropna()
    return (
        pd.DataFrame({"ret": rets})
        .assign(year=lambda x: x.index.year)
        .groupby("year")["ret"]
        .apply(lambda x: (1 + x).prod() - 1)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(do_sweep: bool = False, use_regime: bool = True) -> None:
    mode_label = "REGIME-GATED" if use_regime else "UNCONDITIONAL (NO REGIME)"
    print("\n" + "=" * W)
    print(f"  SPY DAILY -- THREE-SIGNAL {mode_label} LONG-ONLY STRATEGY")
    print(f"  Signals: macd_norm(h=60) + rsi_dev(h=60) + momentum_5(h=5, faded)")
    print(f"  Spread: {SPREAD*100:.2f}%  Slippage: {SLIPPAGE*100:.2f}%  "
          f"Risk/trade: {RISK_PCT*100:.1f}%  Stop: {STOP_ATR}xATR14")
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
    print(f"  IS  bars   : {data['is_n']:,}  "
          f"({data['index'][0].date()} → {data['index'][data['is_n']-1].date()})")
    print(f"  OOS bars   : {oos.sum():,}  "
          f"({data['index'][data['is_n']].date()} → {data['index'][-1].date()})")
    print()
    for lbl in ["ranging", "neutral", "trending"]:
        pct = (regime_oos == lbl).mean() * 100
        print(f"  {lbl.capitalize():<10}: {pct:.0f}% of OOS bars")

    # ── IS baseline (buy & hold comparison)
    bh_ret = float(close_oos.iloc[-1] / close_oos.iloc[0] - 1)
    n_oos  = len(close_oos)
    bh_ann = float((1 + bh_ret) ** (252 / n_oos) - 1)
    print(f"\n  Buy & Hold OOS return : {bh_ret:+.1%}  ({bh_ann:+.1%} ann.)")

    # ── Threshold sweep
    if do_sweep:
        print(f"\n{'='*W}")
        print("  THRESHOLD SWEEP (OOS)")
        print(f"  {'Thresh':>7}  {'Sharpe':>8}  {'Annual':>8}  "
              f"{'MaxDD':>8}  {'Trades':>7}  {'WR%':>6}")
        print("  " + "-" * 55)
        for thr in THRESHOLDS:
            r = _run_long(close_oos, gz_oos, thr, size_oos)
            print(
                f"  {thr:>7.2f}  {r['sharpe']:>+8.3f}  {r['annual']:>+8.1%}  "
                f"{r['dd']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
            )

    # ── Best threshold (1.0z default — from SPY IC analysis)
    # Pick the best threshold from IS data
    best_thr = 1.0
    best_sharpe = -np.inf
    close_is = close_full[is_mask]
    gz_is    = data["composite_z"][is_mask]
    size_is  = data["size"][is_mask]
    for thr in THRESHOLDS:
        r_is = _run_long(close_is, gz_is, thr, size_is)
        if r_is["sharpe"] > best_sharpe:
            best_sharpe = r_is["sharpe"]
            best_thr = thr

    # ── OOS final result
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
    print(f"  {'Sharpe vs B&H Alpha':<26}: {r['sharpe']:>+.3f} vs ~0.60 (SPY historical)")

    verdict = "✅ PASS" if r["sharpe"] > 0.8 and r["dd"] > -0.25 else "❌ FAIL"
    print(f"\n  Strategy Verdict: {verdict}")

    # ── Annual breakdown
    annual = _annual_table(r["value"])
    print(f"\n  {'Year':<8}  {'Strategy Return':>18}  {'vs B&H':>10}")
    print("  " + "-" * 42)
    close_oos_annual = _annual_table(
        close_oos / close_oos.iloc[0] * INIT_CASH
    )
    for yr, val in annual.items():
        bh_yr = close_oos_annual.get(yr, np.nan)
        alpha = val - bh_yr if not np.isnan(bh_yr) else np.nan
        alpha_str = f"{alpha:>+10.1%}" if not np.isnan(alpha) else "       N/A"
        print(f"  {yr:<8}  {val:>+18.1%}  {alpha_str}")

    # ── CSV
    out_path = REPORTS / "spy_strategy.csv"
    pd.DataFrame([{
        "instrument":   INSTRUMENT,
        "mode":         "regime_gated" if use_regime else "unconditional",
        "threshold":    best_thr,
        "sharpe":       r["sharpe"],
        "annual":       r["annual"],
        "ret":          r["ret"],
        "dd":           r["dd"],
        "trades":       r["trades"],
        "wr":           r["wr"],
        "bh_annual":    bh_ann,
    }]).to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")

    # ── Plotly equity curve
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        port_val = r["value"]
        bh_val   = close_oos / close_oos.iloc[0] * INIT_CASH

        def norm(v: pd.Series) -> pd.Series:
            return v / v.iloc[0] * 100

        def dd_series(v: pd.Series) -> pd.Series:
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
            name="Strategy", line=dict(color="#00E5FF", width=2.0)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bh_val.index, y=norm(bh_val),
            name="SPY Buy & Hold", line=dict(color="#FFD600", width=1.5, dash="dot")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=port_val.index, y=dd_series(port_val),
            name="Strategy DD", fill="tozeroy",
            line=dict(color="#FF1744"), fillcolor="rgba(255,23,68,0.18)"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=bh_val.index, y=dd_series(bh_val),
            name="B&H DD", line=dict(color="#FFD600", width=1.0, dash="dot"),
        ), row=2, col=1)

        fig.update_layout(
            title=(
                f"SPY Daily | macd_norm + rsi_dev + momentum_5 | "
                f"{mode_label} | OOS"
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
        print(f"  Chart saved  : {html_out}")
    except Exception as e:
        print(f"  [INFO] Chart skipped: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SPY Three-Signal IC Strategy")
    parser.add_argument("--sweep",     action="store_true", help="Run OOS threshold sweep")
    parser.add_argument("--no-regime", action="store_true", help="Disable ADX regime gate")
    args = parser.parse_args()
    run(do_sweep=args.sweep, use_regime=not args.no_regime)


if __name__ == "__main__":
    main()
