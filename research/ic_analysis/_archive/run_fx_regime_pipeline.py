"""run_fx_regime_pipeline.py -- Full Research Pipeline for FX Vol-Regime Strategy.

Runs the complete 3-phase validation:

  Phase 3: Threshold sweep on IS + IS/OOS backtest (70/30 split)
  Phase 4: Walk-Forward Optimisation (2yr IS / 6mo OOS, rolling)
  Phase 5: Robustness Validation (4 gates)

Strategy: Volatility regime-gated long/short composite
  Vol_High (>70th pct)  -> Mean-reversion composite (zscore_expanding, zscore_100, roc_60)
  Vol_Low  (<30th pct)  -> Trend composite (ma_spread_10_50, ema_slope_20, roc_20)
  Vol_Mid  (30-70th pct) -> Flat (no new entries)

Entry logic (after sign-normalisation):
  gated_z > +threshold  -> LONG
  gated_z < -threshold  -> SHORT
  else                  -> flat

Thresholds: Selected by IS Sharpe sweep across [0.25, 0.50, 0.75, 1.00, 1.50].

Costs: 5 bps spread + 2 bps slippage per fill. Long and short.

Usage:
  uv run python research/ic_analysis/run_fx_regime_pipeline.py
  uv run python research/ic_analysis/run_fx_regime_pipeline.py --instruments AUD_USD GBP_USD
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

from research.ic_analysis.run_signal_sweep import _load_ohlcv, build_all_signals  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_INSTRUMENTS = ["AUD_USD", "GBP_USD"]
TIMEFRAME           = "D"
THRESHOLD_GRID      = [0.25, 0.50, 0.75, 1.00, 1.50]

IS_RATIO = 0.70    # Phase 3 IS/OOS split
WFO_IS   = 504     # Phase 4: 2yr IS (252 x 2 daily bars)
WFO_OOS  = 126     # Phase 4: 6mo OOS (252 x 0.5)

INIT_CASH = 10_000.0
RISK_PCT  = 0.01
STOP_ATR  = 1.5
MAX_LEV   = 10.0
FREQ      = "d"
SPREAD    = 0.0005   # ~5 bps (FX typical spread, applied as SPREAD/med)
SLIPPAGE  = 0.0002   # ~2 bps per fill

# Vol-regime quantile boundaries (IS-only)
VOL_P30 = 0.30
VOL_P70 = 0.70

# Composite signal sets (from regime IC sweep findings)
REVERSION_SIGNALS = ["zscore_expanding", "zscore_100", "roc_60"]   # Vol_High
TREND_SIGNALS     = ["ma_spread_10_50", "ema_slope_20", "roc_20"]   # Vol_Low

MC_SIMS      = 1_000
TOP_N_REMOVE = 5
SLIP_MULT    = 3.0

W = 84
REPORTS = ROOT / ".tmp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Signal / composite helpers
# ---------------------------------------------------------------------------

def _atr14(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(p).mean()


def _ic_sign(signal: pd.Series, fwd: pd.Series) -> float:
    """Spearman IC sign via rank-Pearson (no scipy)."""
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")
    return float(np.sign(r)) if not np.isnan(r) and r != 0 else 1.0


def _build_composite(
    signals: pd.DataFrame,
    close: pd.Series,
    names: list,
    is_mask: pd.Series,
) -> pd.Series:
    """Equal-weight composite, sign-normalised using IS IC."""
    fwd_is = np.log(close.shift(-1) / close)[is_mask]
    parts = []
    for name in names:
        if name not in signals.columns:
            continue
        sign = _ic_sign(signals[name][is_mask], fwd_is)
        parts.append(signals[name] * sign)
    if not parts:
        return pd.Series(0.0, index=signals.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def _zscore_from_mask(series: pd.Series, is_mask: pd.Series) -> pd.Series:
    """Z-score normalised using IS mean/std."""
    is_vals = series[is_mask].dropna()
    mu, sigma = float(is_vals.mean()), float(is_vals.std())
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _size_series(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    atr = _atr14(df)
    stop_pct = (STOP_ATR * atr) / close.where(close > 0)
    return (RISK_PCT / stop_pct.where(stop_pct > 0)).clip(upper=MAX_LEV).fillna(0.0)


def _build_gated_z(
    signals: pd.DataFrame,
    close: pd.Series,
    is_mask: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Build vol-regime-gated z-score composite. Returns (gated_z, regime).

    Vol quantile thresholds are computed on IS data only to prevent look-ahead.
    Sign normalisation of each component is also computed on IS data only.
    """
    rv = signals["realized_vol_20"]

    # Quantile boundaries on IS only
    rv_is = rv[is_mask].dropna()
    p30 = float(rv_is.quantile(VOL_P30))
    p70 = float(rv_is.quantile(VOL_P70))

    regime = pd.Series("vol_mid", index=signals.index)
    regime[rv <= p30] = "vol_low"
    regime[rv > p70]  = "vol_high"

    # IS sub-masks for normalisation
    is_high = is_mask & (regime == "vol_high")
    is_low  = is_mask & (regime == "vol_low")

    # Build composites (sign-normalised on respective IS regime subset)
    rev_comp   = _build_composite(signals, close, REVERSION_SIGNALS, is_high)
    trend_comp = _build_composite(signals, close, TREND_SIGNALS,     is_low)

    # Z-score normalise within each regime
    rev_z   = _zscore_from_mask(rev_comp,   is_high)
    trend_z = _zscore_from_mask(trend_comp, is_low)

    gated_z = pd.Series(0.0, index=signals.index)
    gated_z[regime == "vol_high"] = rev_z[regime == "vol_high"]
    gated_z[regime == "vol_low"]  = trend_z[regime == "vol_low"]

    return gated_z, regime


def _run_longshort(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    size: pd.Series,
    slip_mult: float = 1.0,
) -> dict:
    """VBT long+short backtest using two separate leg portfolios.

    SizeType.Percent does not support signal reversals (long->short in one bar),
    so we run a long-only leg and a short-only leg independently and combine.
    Each leg gets half the initial cash so total exposure is preserved.
    """
    sig  = signal_z.shift(1).fillna(0.0)
    med  = float(close.median()) or 1.0
    fees = SPREAD / med
    slip = (SLIPPAGE * slip_mult) / med
    half = INIT_CASH / 2.0

    # Long leg: enter when signal > threshold, exit when signal <= 0
    pf_long = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size.values,
        size_type="percent",
        init_cash=half,
        fees=fees,
        slippage=slip,
        freq=FREQ,
    )

    # Short leg: enter short when signal < -threshold, exit when signal >= 0
    pf_short = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        size=size.values,
        size_type="percent",
        init_cash=half,
        fees=fees,
        slippage=slip,
        freq=FREQ,
    )

    # Combine legs into a single value series
    val_long  = pf_long.value()
    val_short = pf_short.value()
    common    = val_long.index.intersection(val_short.index)
    val_comb  = val_long.reindex(common) + val_short.reindex(common)

    rets   = val_comb.pct_change().dropna()
    n_bars = len(rets)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    total  = float(val_comb.iloc[-1] / INIT_CASH - 1) if len(val_comb) > 0 else 0.0
    n_yrs  = n_bars / 252.0
    annual = float((val_comb.iloc[-1] / INIT_CASH) ** (1 / n_yrs) - 1) if n_yrs > 0 else 0.0
    dd     = float(((val_comb - val_comb.cummax()) / val_comb.cummax()).min())

    # Aggregate trades from both legs
    n_long  = pf_long.trades.count()
    n_short = pf_short.trades.count()
    n_total = n_long + n_short

    if n_total > 0:
        wr_long  = float(pf_long.trades.win_rate())  if n_long  > 0 else 0.0
        wr_short = float(pf_short.trades.win_rate()) if n_short > 0 else 0.0
        wr = (wr_long * n_long + wr_short * n_short) / n_total
    else:
        wr = 0.0

    return {
        "sharpe":    sharpe,
        "ret":       total,
        "annual":    annual,
        "dd":        dd,
        "trades":    n_total,
        "wr":        wr,
        "value":     val_comb,
        "pf_long":   pf_long,
        "pf_short":  pf_short,
    }


# ---------------------------------------------------------------------------
# Phase 3: Threshold sweep + IS/OOS backtest
# ---------------------------------------------------------------------------

def phase3_sweep(
    close: pd.Series,
    gated_z: pd.Series,
    size: pd.Series,
    is_mask: pd.Series,
    oos_mask: pd.Series,
) -> tuple[float, dict, pd.DataFrame]:
    """Sweep thresholds on IS, select best, report OOS. Returns (best_thr, oos_stats, sweep_df)."""
    print("\n  Threshold sweep (IS Sharpe):")
    print(f"  {'Thr':>6}  {'IS Sh':>8}  {'IS Ret':>8}  {'IS Trd':>7}")
    print("  " + "-" * 40)

    best_thr   = THRESHOLD_GRID[0]
    best_sh    = -np.inf
    sweep_rows = []

    for thr in THRESHOLD_GRID:
        r = _run_longshort(close[is_mask], gated_z[is_mask], thr, size[is_mask])
        flag = " <--" if r["sharpe"] > best_sh else ""
        print(f"  {thr:>6.2f}  {r['sharpe']:>+8.3f}  {r['ret']:>+8.1%}  {r['trades']:>7}{flag}")
        sweep_rows.append({"threshold": thr, "is_sharpe": r["sharpe"],
                           "is_ret": r["ret"], "is_trades": r["trades"]})
        if r["sharpe"] > best_sh:
            best_sh  = r["sharpe"]
            best_thr = thr

    print(f"\n  Selected threshold: {best_thr:.2f}  (IS Sharpe={best_sh:+.3f})")
    oos_stats = _run_longshort(close[oos_mask], gated_z[oos_mask], best_thr, size[oos_mask])
    return best_thr, oos_stats, pd.DataFrame(sweep_rows)


# ---------------------------------------------------------------------------
# Phase 4: Walk-Forward Optimisation
# ---------------------------------------------------------------------------

def phase4_instrument(
    instrument: str,
    df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, list]:
    """Rolling WFO for one instrument. Returns (fold_df, oos_trade_returns)."""
    close   = df["close"]
    signals = build_all_signals(df)
    valid   = signals.notna().any(axis=1) & close.notna()
    signals = signals[valid]
    close   = close[valid]
    df_v    = df.loc[valid]

    n    = len(signals)
    rows = []
    all_trade_returns = []

    fold  = 0
    start = 0
    while start + WFO_IS + WFO_OOS <= n:
        is_idx  = range(start, start + WFO_IS)
        oos_idx = range(start + WFO_IS, start + WFO_IS + WFO_OOS)

        sig_slice   = signals.iloc[list(is_idx) + list(oos_idx)]
        close_slice = close.iloc[list(is_idx) + list(oos_idx)]
        is_slice    = pd.Series(
            [True] * WFO_IS + [False] * WFO_OOS, index=sig_slice.index
        )

        gz_slice, _ = _build_gated_z(sig_slice, close_slice, is_slice)
        size_slice  = _size_series(
            df_v.iloc[list(is_idx) + list(oos_idx)], close_slice
        )

        # OOS portion
        oos_close = close_slice.iloc[WFO_IS:]
        oos_gz    = gz_slice.iloc[WFO_IS:]
        oos_size  = size_slice.iloc[WFO_IS:]

        r = _run_longshort(oos_close, oos_gz, threshold, oos_size)

        if r["trades"] > 0:
            try:
                for leg_key in ("pf_long", "pf_short"):
                    pf_leg = r.get(leg_key)
                    if pf_leg is not None and pf_leg.trades.count() > 0:
                        all_trade_returns.extend(
                            pf_leg.trades.records_readable["Return"].tolist()
                        )
            except Exception:
                pass

        is_r = _run_longshort(
            close_slice.iloc[:WFO_IS],
            gz_slice.iloc[:WFO_IS],
            threshold,
            size_slice.iloc[:WFO_IS],
        )

        rows.append({
            "fold":       fold + 1,
            "is_start":   signals.index[start].date(),
            "is_end":     signals.index[start + WFO_IS - 1].date(),
            "oos_start":  signals.index[start + WFO_IS].date(),
            "oos_end":    signals.index[start + WFO_IS + WFO_OOS - 1].date(),
            "is_sharpe":  is_r["sharpe"],
            "oos_sharpe": r["sharpe"],
            "oos_ret":    r["ret"],
            "oos_trades": r["trades"],
            "oos_wr":     r["wr"],
        })
        fold  += 1
        start += WFO_OOS

    return pd.DataFrame(rows), all_trade_returns


# ---------------------------------------------------------------------------
# Phase 5: Robustness
# ---------------------------------------------------------------------------

def _mc_sharpe(trade_returns: list, n_sims: int, trades_per_year: float) -> dict:
    """Monte Carlo trade shuffle. Returns {p5, p95, pct_positive}."""
    if not trade_returns or trades_per_year <= 0:
        return {"p5": np.nan, "p95": np.nan, "pct_pos": np.nan}
    arr = np.array(trade_returns)
    sharpes = []
    for _ in range(n_sims):
        np.random.shuffle(arr)
        mu  = arr.mean()
        std = arr.std()
        sh  = (mu / std * np.sqrt(trades_per_year)) if std > 1e-10 else 0.0
        sharpes.append(sh)
    sharpes = np.array(sharpes)
    return {
        "p5":      float(np.percentile(sharpes, 5)),
        "p95":     float(np.percentile(sharpes, 95)),
        "pct_pos": float((sharpes > 0).mean()),
    }


def phase5(
    instrument: str,
    threshold: float,
    oos_data: dict,
    wfo_df: pd.DataFrame,
    wfo_trade_returns: list,
) -> dict:
    """Phase 5 robustness gates."""
    close = oos_data["close"]
    gz    = oos_data["gated_z"]
    size  = oos_data["size"]
    n_oos = len(close)

    # Gate 1: Monte Carlo
    n_trades_oos = oos_data["trades"]
    n_years      = n_oos / 252.0
    tpy          = n_trades_oos / n_years if n_years > 0 else 0.0
    mc = _mc_sharpe(wfo_trade_returns, MC_SIMS, tpy)

    # Gate 2: Remove top-N winning trades (combined from both legs)
    if oos_data["trades"] > 0:
        try:
            all_rets = []
            for leg_key in ("pf_long", "pf_short"):
                pf_leg = oos_data.get(leg_key)
                if pf_leg is not None and pf_leg.trades.count() > 0:
                    all_rets.extend(
                        pf_leg.trades.records_readable["Return"].tolist()
                    )
            trade_rets_arr = np.array(all_rets)
            top_idx   = np.argsort(trade_rets_arr)[-TOP_N_REMOVE:]
            remaining = np.delete(trade_rets_arr, top_idx)
            topn_sum  = float(remaining.sum())
        except Exception:
            topn_sum = 0.0
    else:
        topn_sum = 0.0

    # Gate 3: 3x slippage
    r_stress = _run_longshort(close, gz, threshold, size, slip_mult=SLIP_MULT)

    # Gate 4: Consecutive negative WFO folds
    neg_streak = 0
    if len(wfo_df) > 0:
        cur_streak = 0
        max_streak = 0
        for sh in wfo_df["oos_sharpe"]:
            if sh < 0:
                cur_streak += 1
                max_streak  = max(max_streak, cur_streak)
            else:
                cur_streak  = 0
        neg_streak = max_streak

    return {
        "mc_p5":      mc["p5"],
        "mc_p95":     mc["p95"],
        "mc_pct_pos": mc["pct_pos"],
        "topn_sum":   topn_sum,
        "stress_sh":  r_stress["sharpe"],
        "neg_streak": neg_streak,
        "g1_pass": mc["p5"] > 0.5 if not np.isnan(mc["p5"]) else False,
        "g2_pass": topn_sum > 0,
        "g3_pass": r_stress["sharpe"] > 0.5,
        "g4_pass": neg_streak <= 2,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")


def _gate(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _gk(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_wfo(instrument: str, wfo_df: pd.DataFrame, threshold: float) -> dict:
    _hdr(f"PHASE 4 -- WALK-FORWARD  ({instrument}  thr={threshold:.2f})")
    print(f"  IS window: {WFO_IS} bars (2yr)  |  OOS window: {WFO_OOS} bars (6mo)")
    print(f"  Folds: {len(wfo_df)}  |  "
          f"Stitched OOS coverage: ~{len(wfo_df)*WFO_OOS/252:.1f} years")
    print(f"\n  {'Fold':>5}  {'IS Start':>12}  {'OOS Start':>12}  "
          f"{'IS Sh':>8}  {'OOS Sh':>8}  {'OOS Ret':>8}  {'Trd':>5}  {'WR%':>6}")
    print("  " + "-" * (W - 2))
    for _, row in wfo_df.iterrows():
        flag = " *" if row["oos_sharpe"] < 0 else ""
        print(
            f"  {int(row['fold']):>5}  {str(row['is_start']):>12}  "
            f"{str(row['oos_start']):>12}  "
            f"{row['is_sharpe']:>+8.3f}  {row['oos_sharpe']:>+8.3f}"
            f"{flag}  {row['oos_ret']:>+8.1%}  {int(row['oos_trades']):>5}  "
            f"{row['oos_wr']*100:>6.1f}%"
        )

    n        = len(wfo_df)
    pct_pos  = (wfo_df["oos_sharpe"] > 0).mean()
    pct_gt1  = (wfo_df["oos_sharpe"] > 1).mean()
    worst    = float(wfo_df["oos_sharpe"].min())
    stitched = float(wfo_df["oos_sharpe"].mean())
    is_avg   = float(wfo_df["is_sharpe"].mean())
    parity   = stitched / is_avg if is_avg > 0 else 0.0

    print(f"\n  WFO Gate Summary  ({n} folds)")
    print(f"  {'> 0% folds':<28}: {pct_pos:.0%}  {_gate(pct_pos >= 0.70)}")
    print(f"  {'> 1 Sharpe folds':<28}: {pct_gt1:.0%}  {_gate(pct_gt1 >= 0.50)}")
    print(f"  {'Worst fold Sharpe':<28}: {worst:>+.3f}  {_gate(worst >= -2.0)}")
    print(f"  {'Mean OOS Sharpe':<28}: {stitched:>+.3f}  {_gate(stitched >= 1.0)}")
    print(f"  {'OOS/IS parity':<28}: {parity:>.3f}  {_gate(parity >= 0.50)}")

    return {
        "pct_pos": pct_pos, "pct_gt1": pct_gt1, "worst": worst,
        "stitched": stitched, "parity": parity,
    }


def _print_robustness(instrument: str, r5: dict) -> None:
    _hdr(f"PHASE 5 -- ROBUSTNESS  ({instrument})")
    print("  Gate 1 -- Monte Carlo (1,000 shuffles, thr=5th pct > 0.5)")
    print(f"    5th-pct Sharpe  : {r5['mc_p5']:>+.3f}  {_gate(r5['g1_pass'])}")
    print(f"    95th-pct Sharpe : {r5['mc_p95']:>+.3f}")
    print(f"    % profitable    : {r5['mc_pct_pos']*100:.1f}%")
    print(f"  Gate 2 -- Remove top {TOP_N_REMOVE} winning trades")
    print(f"    Remaining sum   : {r5['topn_sum']:>+.3f}  {_gate(r5['g2_pass'])}")
    print(f"  Gate 3 -- {SLIP_MULT}x slippage stress test (thr: OOS Sharpe > 0.5)")
    print(f"    Stress Sharpe   : {r5['stress_sh']:>+.3f}  {_gate(r5['g3_pass'])}")
    print("  Gate 4 -- Max consecutive negative WFO folds (thr: <= 2)")
    print(f"    Neg streak      : {r5['neg_streak']:>3}       {_gate(r5['g4_pass'])}")
    all_pass = r5["g1_pass"] and r5["g2_pass"] and r5["g3_pass"] and r5["g4_pass"]
    verdict  = "ALL PASS -- cleared for further validation" if all_pass else "ONE OR MORE GATES FAILED"
    print(f"\n  Overall Phase 5: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(instruments: list[str]) -> None:
    _hdr(f"FX VOL-REGIME PIPELINE  (L/S, {', '.join(instruments)})  PHASES 3-5")
    print(f"  Instruments  : {', '.join(instruments)}  [{TIMEFRAME}]")
    print(f"  Regime gate  : realized_vol_20 quantiles  "
          f"(Vol_Low<{VOL_P30:.0%} / Vol_High>{VOL_P70:.0%})")
    print(f"  Reversion    : {REVERSION_SIGNALS}")
    print(f"  Trend        : {TREND_SIGNALS}")
    print(f"  Thresh grid  : {THRESHOLD_GRID}")
    print(f"  Costs        : Spread {SPREAD*100:.2f}%  Slippage {SLIPPAGE*100:.2f}%")

    # -------------------------------------------------------------------------
    # Load, build signals, initial IS/OOS split
    # -------------------------------------------------------------------------
    inst_data: dict[str, dict] = {}

    for inst in instruments:
        print(f"\nLoading {inst}...")
        df      = _load_ohlcv(inst, TIMEFRAME)
        close   = df["close"]
        signals = build_all_signals(df)
        valid   = signals.notna().any(axis=1) & close.notna()
        signals = signals[valid]
        close   = close[valid]
        df      = df.loc[valid]

        n     = len(signals)
        is_n  = int(n * IS_RATIO)
        is_mask  = pd.Series(False, index=signals.index)
        is_mask.iloc[:is_n] = True
        oos_mask = ~is_mask

        gz, regime = _build_gated_z(signals, close, is_mask)
        size        = _size_series(df, close)

        # Regime bar counts
        n_low  = int((regime == "vol_low").sum())
        n_mid  = int((regime == "vol_mid").sum())
        n_high = int((regime == "vol_high").sum())

        print(f"  {n:,} bars | IS: {is_n:,} | OOS: {n-is_n:,}")
        print(f"  IS : {signals.index[0].date()} -> {signals.index[is_n-1].date()}")
        print(f"  OOS: {signals.index[is_n].date()} -> {signals.index[-1].date()}")
        print(f"  Regimes (all bars): Vol_Low={n_low:,} ({n_low/n:.0%})  "
              f"Vol_Mid={n_mid:,} ({n_mid/n:.0%})  "
              f"Vol_High={n_high:,} ({n_high/n:.0%})")

        inst_data[inst] = {
            "instrument": inst,
            "df":         df,
            "close":      close,
            "signals":    signals,
            "gated_z":    gz,
            "regime":     regime,
            "size":       size,
            "is_mask":    is_mask,
            "oos_mask":   oos_mask,
        }

    # -------------------------------------------------------------------------
    # Phase 3: Threshold sweep + IS/OOS
    # -------------------------------------------------------------------------
    _hdr("PHASE 3 -- THRESHOLD SWEEP + IS/OOS BACKTEST (70/30 SPLIT)")

    sweep_dfs = []
    for inst in instruments:
        d = inst_data[inst]
        print(f"\n  [{inst}]")

        best_thr, oos_r, sweep_df = phase3_sweep(
            d["close"], d["gated_z"], d["size"], d["is_mask"], d["oos_mask"]
        )
        sweep_df["instrument"] = inst
        sweep_dfs.append(sweep_df)
        inst_data[inst]["threshold"] = best_thr
        inst_data[inst]["p3"]        = oos_r

        print(f"\n  OOS  ({inst}  thr={best_thr:.2f}):")
        print(f"    Sharpe={oos_r['sharpe']:>+.3f}  Annual={oos_r['annual']:>+.1%}  "
              f"MaxDD={oos_r['dd']:>+.1%}  Trades={oos_r['trades']}  WR={oos_r['wr']:.0%}")

    # Phase 3 summary table
    print(f"\n  {'Inst':<10}  {'Thresh':>6}  {'Sharpe':>8}  {'Annual':>8}  "
          f"{'MaxDD':>8}  {'Ret':>8}  {'Trades':>7}  {'WR%':>6}")
    print("  " + "-" * (W - 2))
    p3_values = []
    for inst in instruments:
        d   = inst_data[inst]
        r   = d["p3"]
        thr = d["threshold"]
        print(
            f"  {inst:<10}  {thr:>6.2f}  {r['sharpe']:>+8.3f}  "
            f"{r['annual']:>+8.1%}  {r['dd']:>+8.1%}  "
            f"{r['ret']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
        )
        p3_values.append(r["value"])

    # Portfolio-level stats
    if len(p3_values) > 1:
        common   = p3_values[0].index.intersection(p3_values[1].index)
        port_val = sum(v.reindex(common) for v in p3_values)
        port_ret = port_val.pct_change().dropna()
        port_sh  = float(
            port_ret.mean() / port_ret.std() * np.sqrt(252)
        ) if port_ret.std() > 0 else 0.0
        port_ann = float(
            (port_val.iloc[-1] / port_val.iloc[0]) ** (252 / len(port_ret)) - 1
        )
        port_dd = float(((port_val - port_val.cummax()) / port_val.cummax()).min())
        print(f"  {'PORTFOLIO':<10}  {'mixed':>6}  {port_sh:>+8.3f}  "
              f"{port_ann:>+8.1%}  {port_dd:>+8.1%}")

        # Annual breakdown
        print("\n  Annual portfolio returns (OOS):")
        yr_rets = (
            pd.DataFrame({"r": port_ret})
            .assign(yr=lambda x: x.index.year)
            .groupby("yr")["r"]
            .apply(lambda x: (1 + x).prod() - 1)
        )
        for yr, val in yr_rets.items():
            val  = float(val)
            bar  = "#" * int(abs(val) * 200)
            sign = "+" if val >= 0 else "-"
            print(f"    {yr}  {val:>+7.1%}  {sign}{bar}")

    # Save sweep CSV
    sweep_out = REPORTS / "fx_regime_threshold_sweep.csv"
    pd.concat(sweep_dfs, ignore_index=True).to_csv(sweep_out, index=False)
    print(f"\n  Threshold sweep saved: {sweep_out}")

    # -------------------------------------------------------------------------
    # Phase 4: WFO
    # -------------------------------------------------------------------------
    wfo_results: dict[str, pd.DataFrame] = {}
    wfo_trades:  dict[str, list]         = {}

    for inst in instruments:
        d   = inst_data[inst]
        thr = d["threshold"]
        wfo_df, trade_rets = phase4_instrument(inst, d["df"], thr)
        wfo_results[inst] = wfo_df
        wfo_trades[inst]  = trade_rets
        gate_summary = _print_wfo(inst, wfo_df, thr)
        inst_data[inst]["wfo_gates"] = gate_summary

        out = REPORTS / f"fx_regime_wfo_{inst.lower()}.csv"
        wfo_df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")

    # -------------------------------------------------------------------------
    # Phase 5: Robustness
    # -------------------------------------------------------------------------
    for inst in instruments:
        d    = inst_data[inst]
        oos  = d["oos_mask"]
        thr  = d["threshold"]
        oos_data = {
            "close":    d["close"][oos],
            "gated_z":  d["gated_z"][oos],
            "size":     d["size"][oos],
            "trades":   d["p3"]["trades"],
            "pf_long":  d["p3"]["pf_long"],
            "pf_short": d["p3"]["pf_short"],
        }
        r5 = phase5(inst, thr, oos_data, wfo_results[inst], wfo_trades[inst])
        inst_data[inst]["p5"] = r5
        _print_robustness(inst, r5)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    _hdr("PIPELINE SUMMARY")
    print(f"  {'Inst':<10}  {'Thr':>5}  {'P3 Sh':>8}  {'WFO>0%':>8}  "
          f"{'WFO>1%':>8}  {'Worst':>8}  {'MC p5':>8}  {'3xSlip':>8}  "
          f"{'G1':>5}  {'G2':>5}  {'G3':>5}  {'G4':>5}")
    print("  " + "-" * (W - 2))
    for inst in instruments:
        d  = inst_data[inst]
        g  = d.get("wfo_gates", {})
        r5 = d.get("p5", {})
        print(
            f"  {inst:<10}  {d['threshold']:>5.2f}  {d['p3']['sharpe']:>+8.3f}  "
            f"{g.get('pct_pos', 0)*100:>8.0f}%  "
            f"{g.get('pct_gt1', 0)*100:>8.0f}%  "
            f"{g.get('worst', 0):>+8.3f}  "
            f"{r5.get('mc_p5', 0):>+8.3f}  "
            f"{r5.get('stress_sh', 0):>+8.3f}  "
            f"{_gk(r5.get('g1_pass', False)):>5}  "
            f"{_gk(r5.get('g2_pass', False)):>5}  "
            f"{_gk(r5.get('g3_pass', False)):>5}  "
            f"{_gk(r5.get('g4_pass', False)):>5}"
        )

    # Save summary CSV
    rows = []
    for inst in instruments:
        d  = inst_data[inst]
        g  = d.get("wfo_gates", {})
        r5 = d.get("p5", {})
        rows.append({
            "instrument":  inst,
            "timeframe":   TIMEFRAME,
            "threshold":   d["threshold"],
            "reversion":   ",".join(REVERSION_SIGNALS),
            "trend":       ",".join(TREND_SIGNALS),
            "p3_sharpe":   d["p3"]["sharpe"],
            "p3_annual":   d["p3"]["annual"],
            "p3_dd":       d["p3"]["dd"],
            "p3_trades":   d["p3"]["trades"],
            "wfo_pct_pos": g.get("pct_pos"),
            "wfo_pct_gt1": g.get("pct_gt1"),
            "wfo_worst":   g.get("worst"),
            "wfo_mean_sh": g.get("stitched"),
            "wfo_parity":  g.get("parity"),
            "mc_p5":       r5.get("mc_p5"),
            "stress_sh":   r5.get("stress_sh"),
            "neg_streak":  r5.get("neg_streak"),
            "all_gates":   all([r5.get(f"g{i}_pass") for i in range(1, 5)]),
        })
    out = REPORTS / "fx_regime_pipeline_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Summary saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FX Vol-Regime Pipeline (Phases 3-5)"
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=DEFAULT_INSTRUMENTS,
        help="FX instruments to test (default: AUD_USD GBP_USD)",
    )
    args = parser.parse_args()
    run(args.instruments)


if __name__ == "__main__":
    np.random.seed(42)
    main()
