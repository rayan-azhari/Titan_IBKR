"""run_regime_backtest.py -- ADX Regime-Gated Strategy Backtest.

Tests whether regime-conditional IC translates to a profitable strategy
by gating signal selection based on the current ADX regime:

  ADX < 20  (Ranging)   -> Use oscillator composite (mean-reversion signals)
  ADX 20-25 (Neutral)   -> No new entries; exit open positions naturally
  ADX > 25  (Trending)  -> Use trend composite (momentum/trend signals)

Signal sets (derived from run_regime_ic.py findings):
  Ranging  : zscore_50, bb_zscore_50, cci_20, stoch_k_dev,
             donchian_pos_10, zscore_20, bb_zscore_20
  Trending : ma_spread_50_200, ma_spread_20_100, zscore_expanding,
             zscore_100, roc_60, price_pct_rank_60, ema_slope_20

Baseline comparison: ungated equal-weight composite of ALL signals.

Entry / exit:
  Long  : gated_z crosses above  +threshold (next open)
  Short : gated_z crosses below  -threshold
  Exit  : gated_z drops to 0 (regime neutralises OR z-cross)

Costs (stock defaults):
  Spread   : 0.10% per fill (5 bps each side)
  Slippage : 0.05% per fill
  No swap  : daily stocks held < overnight carry significance

Usage:
  uv run python research/ic_analysis/run_regime_backtest.py
  uv run python research/ic_analysis/run_regime_backtest.py --instrument TXN
  uv run python research/ic_analysis/run_regime_backtest.py --instrument SPY --spread 0.0005
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from scipy.stats import spearmanr  # noqa: E402

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IS_RATIO   = 0.70
INIT_CASH  = 10_000.0
RISK_PCT   = 0.01
STOP_ATR   = 1.5
MAX_LEV    = 10.0
FREQ       = "d"

# Stock cost defaults
DEFAULT_SPREAD   = 0.0010   # 10 bps round-trip (5 bps/side)
DEFAULT_SLIP     = 0.0005   # 5 bps slippage per fill

THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.50]

# ADX regime boundaries
ADX_RANGING  = 20.0
ADX_TRENDING = 25.0

# Signal sets derived from regime IC analysis
RANGING_SIGNALS = [
    "zscore_50", "bb_zscore_50", "cci_20", "stoch_k_dev",
    "donchian_pos_10", "zscore_20", "bb_zscore_20",
]
TRENDING_SIGNALS = [
    "ma_spread_50_200", "ma_spread_20_100", "zscore_expanding",
    "zscore_100", "roc_60", "price_pct_rank_60", "ema_slope_20",
]

W = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(p).mean()


def _ic_sign(signal: pd.Series, fwd: pd.Series) -> float:
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0 else 1.0


def _build_composite(
    signals: pd.DataFrame,
    close: pd.Series,
    signal_names: list[str],
    is_mask: pd.Series,
) -> pd.Series:
    """Equal-weight composite, IC sign-normalised on IS bars."""
    fwd_is = np.log(close.shift(-1) / close)[is_mask]
    parts = []
    for name in signal_names:
        if name not in signals.columns:
            print(f"  [WARN] Signal '{name}' not found -- skipping.")
            continue
        raw = signals[name]
        sign = _ic_sign(raw[is_mask], fwd_is)
        parts.append(raw * sign)
    if not parts:
        return pd.Series(0.0, index=signals.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def _zscore(series: pd.Series, is_mask: pd.Series) -> pd.Series:
    """Z-score normalise using IS mean/std only."""
    is_vals = series[is_mask].dropna()
    mu, sigma = float(is_vals.mean()), float(is_vals.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _size_array(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    atr = _atr14(df)
    stop_dist = STOP_ATR * atr
    safe_close = close.where(close > 0)
    stop_pct = stop_dist / safe_close
    size_pct = RISK_PCT / stop_pct.where(stop_pct > 0)
    return size_pct.clip(upper=MAX_LEV).fillna(0.0)


def _run_vbt(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    spread: float,
    slippage: float,
    size: pd.Series,
) -> dict:
    sig = signal_z.shift(1).fillna(0.0)
    size_arr = size.reindex(close.index).fillna(0.0).values
    med_close = float(close.median()) or 1.0
    fees = spread / med_close
    slip = slippage / med_close

    pf_long = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size_arr, size_type="percent",
        init_cash=INIT_CASH, fees=fees, slippage=slip, freq=FREQ,
    )
    pf_short = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        size=size_arr, size_type="percent",
        init_cash=INIT_CASH, fees=fees, slippage=slip, freq=FREQ,
    )

    def _s(pf):
        n = pf.trades.count()
        return {
            "sharpe": float(pf.sharpe_ratio()),
            "ret":    float(pf.total_return()),
            "dd":     float(pf.max_drawdown()),
            "trades": int(n),
            "wr":     float(pf.trades.win_rate()) if n > 0 else 0.0,
        }

    sl, ss = _s(pf_long), _s(pf_short)
    n_total = sl["trades"] + ss["trades"]
    wr_comb = (
        (sl["wr"] * sl["trades"] + ss["wr"] * ss["trades"]) / n_total
        if n_total > 0 else 0.0
    )
    return {
        "sharpe_long":  sl["sharpe"],
        "sharpe_short": ss["sharpe"],
        "sharpe_comb":  (sl["sharpe"] + ss["sharpe"]) / 2,
        "ret_long":     sl["ret"],
        "ret_short":    ss["ret"],
        "dd_long":      sl["dd"],
        "dd_short":     ss["dd"],
        "trades":       n_total,
        "wr":           wr_comb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(instrument: str, spread: float, slippage: float) -> None:
    slug = instrument.lower()

    print("\n" + "=" * W)
    print(f"  REGIME-GATED BACKTEST -- {instrument}")
    print(f"  Spread: {spread*100:.2f}%/fill  Slippage: {slippage*100:.2f}%/fill")
    print(f"  Risk/trade: {RISK_PCT*100:.1f}%  Stop: {STOP_ATR}xATR14  MaxLev: {MAX_LEV}x")
    print("=" * W)

    # 1. Load data
    df    = _load_ohlcv(instrument, "D")
    close = df["close"]

    # 2. Build signals
    print("\nBuilding 52 signals...")
    signals = build_all_signals(df)

    # Validate required signals exist
    missing_r = [s for s in RANGING_SIGNALS  if s not in signals.columns]
    missing_t = [s for s in TRENDING_SIGNALS if s not in signals.columns]
    if missing_r:
        print(f"  [WARN] Missing ranging signals: {missing_r}")
    if missing_t:
        print(f"  [WARN] Missing trending signals: {missing_t}")

    # 3. IS/OOS split
    valid = signals.notna().any(axis=1) & close.notna()
    signals = signals[valid]
    close   = close[valid]
    df      = df.loc[valid]

    n    = len(signals)
    is_n = int(n * IS_RATIO)
    is_mask  = pd.Series(False, index=signals.index)
    is_mask.iloc[:is_n] = True
    oos_mask = ~is_mask

    print(f"  Bars: {n:,}  |  IS: {is_n:,}  |  OOS: {n-is_n:,}")
    print(f"  IS : {signals.index[0].date()} -> {signals.index[is_n-1].date()}")
    print(f"  OOS: {signals.index[is_n].date()} -> {signals.index[-1].date()}")

    # 4. ADX regime labels (from signal already computed)
    adx = signals["adx_14"]
    regime = pd.Series("neutral", index=signals.index)
    regime[adx < ADX_RANGING]  = "ranging"
    regime[adx > ADX_TRENDING] = "trending"

    for lbl in ["ranging", "neutral", "trending"]:
        pct = (regime == lbl).mean() * 100
        print(f"  {lbl.capitalize():<10}: {pct:.0f}% of all bars")

    # 5. Build regime composites (IS-calibrated)
    print("\nBuilding composites...")
    ranging_comp  = _build_composite(signals, close, RANGING_SIGNALS,  is_mask)
    trending_comp = _build_composite(signals, close, TRENDING_SIGNALS, is_mask)

    # 6. Z-score each composite using IS bars of its own regime
    is_ranging  = is_mask & (regime == "ranging")
    is_trending = is_mask & (regime == "trending")
    print(f"  IS ranging bars : {is_ranging.sum():,}")
    print(f"  IS trending bars: {is_trending.sum():,}")

    ranging_z  = _zscore(ranging_comp,  is_ranging)
    trending_z = _zscore(trending_comp, is_trending)

    # 7. Gated signal: ranging_z in ranging, trending_z in trending, 0 in neutral
    gated_z = pd.Series(0.0, index=signals.index)
    gated_z[regime == "ranging"]  = ranging_z[regime == "ranging"]
    gated_z[regime == "trending"] = trending_z[regime == "trending"]

    # 8. Baseline: ungated composite of ALL signals
    all_sig_names = [c for c in signals.columns]
    baseline_comp = _build_composite(signals, close, all_sig_names, is_mask)
    baseline_z    = _zscore(baseline_comp, is_mask)

    # 9. ATR sizing
    size = _size_array(df, close)

    # 10. OOS only
    oos_close    = close[oos_mask]
    oos_gated    = gated_z[oos_mask]
    oos_baseline = baseline_z[oos_mask]
    oos_size     = size[oos_mask]

    # OOS regime breakdown
    oos_regime = regime[oos_mask]
    print("\nOOS regime breakdown:")
    for lbl in ["ranging", "neutral", "trending"]:
        pct = (oos_regime == lbl).mean() * 100
        n_r = (oos_regime == lbl).sum()
        print(f"  {lbl.capitalize():<10}: {n_r:,} bars ({pct:.0f}%)")

    # 11. Threshold sweep
    print(f"\n{'='*W}")
    print("  THRESHOLD SWEEP -- OOS RESULTS")
    print(f"  {'Thresh':>7}  {'Strategy':^50}  {'Baseline':^20}")
    print(f"  {'':>7}  {'Sharpe_L':>8}  {'Sharpe_S':>8}  {'Sharpe_C':>8}"
          f"  {'Trades':>7}  {'WR%':>5}  {'Shr_C_BL':>8}")
    print("  " + "-" * (W - 2))

    results = []
    for thr in THRESHOLDS:
        g = _run_vbt(oos_close, oos_gated,    thr, spread, slippage, oos_size)
        b = _run_vbt(oos_close, oos_baseline, thr, spread, slippage, oos_size)
        print(
            f"  {thr:>7.2f}  "
            f"{g['sharpe_long']:>+8.2f}  {g['sharpe_short']:>+8.2f}  "
            f"{g['sharpe_comb']:>+8.2f}  {g['trades']:>7,}  "
            f"{g['wr']*100:>5.1f}  {b['sharpe_comb']:>+8.2f}"
        )
        results.append({
            "threshold":     thr,
            "gated_sharpe":  g["sharpe_comb"],
            "base_sharpe":   b["sharpe_comb"],
            "uplift":        g["sharpe_comb"] - b["sharpe_comb"],
            "gated_trades":  g["trades"],
            "gated_wr":      g["wr"],
            "gated_ret_l":   g["ret_long"],
            "gated_ret_s":   g["ret_short"],
            "gated_dd_l":    g["dd_long"],
            "gated_dd_s":    g["dd_short"],
        })

    results_df = pd.DataFrame(results)

    # 12. Best threshold detail
    best = results_df.loc[results_df["gated_sharpe"].idxmax()]
    best_thr = float(best["threshold"])
    g_best = _run_vbt(oos_close, oos_gated,    best_thr, spread, slippage, oos_size)
    b_best = _run_vbt(oos_close, oos_baseline, best_thr, spread, slippage, oos_size)

    print(f"\n{'='*W}")
    print(f"  BEST THRESHOLD: {best_thr}z")
    print(f"{'='*W}")
    print(f"                     {'Gated':>12}  {'Baseline':>12}  {'Uplift':>10}")
    print(f"  {'Combined Sharpe':<22} {g_best['sharpe_comb']:>+12.3f}  "
          f"{b_best['sharpe_comb']:>+12.3f}  "
          f"{g_best['sharpe_comb']-b_best['sharpe_comb']:>+10.3f}")
    print(f"  {'Long Sharpe':<22} {g_best['sharpe_long']:>+12.3f}  "
          f"{b_best['sharpe_long']:>+12.3f}")
    print(f"  {'Short Sharpe':<22} {g_best['sharpe_short']:>+12.3f}  "
          f"{b_best['sharpe_short']:>+12.3f}")
    print(f"  {'Long Return':<22} {g_best['ret_long']:>+12.1%}  "
          f"{b_best['ret_long']:>+12.1%}")
    print(f"  {'Short Return':<22} {g_best['ret_short']:>+12.1%}  "
          f"{b_best['ret_short']:>+12.1%}")
    print(f"  {'Max DD (long)':<22} {g_best['dd_long']:>+12.1%}  "
          f"{b_best['dd_long']:>+12.1%}")
    print(f"  {'Trades':<22} {g_best['trades']:>12,}  {b_best['trades']:>12,}")
    print(f"  {'Win Rate':<22} {g_best['wr']:>+12.1%}  {b_best['wr']:>+12.1%}")

    # 13. Save
    out = ROOT / ".tmp" / "reports" / f"regime_bt_{slug}.csv"
    results_df.to_csv(out, index=False)
    print(f"\n  Results saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ADX Regime-Gated Strategy Backtest")
    parser.add_argument("--instrument", default="TXN")
    parser.add_argument("--spread",   type=float, default=DEFAULT_SPREAD,
                        help="Half-spread per fill as fraction of price (default 0.001)")
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIP,
                        help="Slippage per fill as fraction of price (default 0.0005)")
    args = parser.parse_args()
    run(args.instrument, args.spread, args.slippage)


if __name__ == "__main__":
    main()
