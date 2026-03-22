"""run_cat_amat_pipeline.py -- Full Research Pipeline for CAT + AMAT.

Runs the complete 3-phase validation:

  Phase 3: IS/OOS backtest (70/30 split, 11yr daily data)
  Phase 4: Walk-Forward Optimisation (2yr IS / 6mo OOS, rolling)
  Phase 5: Robustness Validation (4 gates)

Strategy: ADX regime-gated long-only composite
  ADX > 25  -> Trend composite (momentum / moving-average signals)
  ADX < 20  -> Oscillator composite (mean-reversion signals)
  ADX 20-25 -> Flat

Thresholds (from Phase 3 long-only sweep):
  CAT  : 1.00z
  AMAT : 0.75z

Costs: 10 bps spread + 5 bps slippage per fill. Long-only, no swap.

Usage:
  uv run python research/ic_analysis/run_cat_amat_pipeline.py
"""

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
from scipy.stats import spearmanr  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INSTRUMENTS   = ["CAT", "AMAT"]
THRESHOLDS    = {"CAT": 1.00, "AMAT": 0.75}

IS_RATIO   = 0.70    # Phase 3 split
WFO_IS     = 504     # Phase 4: 2yr IS (252 × 2 daily bars)
WFO_OOS    = 126     # Phase 4: 6mo OOS (252 × 0.5)

INIT_CASH  = 10_000.0
RISK_PCT   = 0.01
STOP_ATR   = 1.5
MAX_LEV    = 10.0
FREQ       = "d"
SPREAD     = 0.0010   # 10 bps round-trip
SLIPPAGE   = 0.0005   # 5 bps per fill

ADX_RANGING  = 20.0
ADX_TRENDING = 25.0

RANGING_SIGNALS = [
    "zscore_50", "bb_zscore_50", "cci_20", "stoch_k_dev",
    "donchian_pos_10", "zscore_20", "bb_zscore_20",
]
TRENDING_SIGNALS = [
    "ma_spread_50_200", "ma_spread_20_100", "zscore_expanding",
    "zscore_100", "roc_60", "price_pct_rank_60", "ema_slope_20",
]

MC_SIMS        = 1_000
TOP_N_REMOVE   = 5
SLIP_MULT      = 3.0

W = 82
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
    both = pd.concat([signal, fwd], axis=1).dropna()
    if len(both) < 30:
        return 1.0
    r, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return float(np.sign(r)) if not np.isnan(r) and r != 0 else 1.0


def _build_composite(
    signals: pd.DataFrame,
    close: pd.Series,
    names: list,
    is_mask: pd.Series,
) -> pd.Series:
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


def _zscore(series: pd.Series, is_mask: pd.Series) -> pd.Series:
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
    """Build ADX-gated z-score composite. Returns (gated_z, regime)."""
    adx    = signals["adx_14"]
    regime = pd.Series("neutral", index=signals.index)
    regime[adx < ADX_RANGING]  = "ranging"
    regime[adx > ADX_TRENDING] = "trending"

    ranging_comp  = _build_composite(signals, close, RANGING_SIGNALS,  is_mask)
    trending_comp = _build_composite(signals, close, TRENDING_SIGNALS, is_mask)

    is_ranging  = is_mask & (regime == "ranging")
    is_trending = is_mask & (regime == "trending")

    ranging_z  = _zscore(ranging_comp,  is_ranging)
    trending_z = _zscore(trending_comp, is_trending)

    gated_z = pd.Series(0.0, index=signals.index)
    gated_z[regime == "ranging"]  = ranging_z[regime == "ranging"]
    gated_z[regime == "trending"] = trending_z[regime == "trending"]

    return gated_z, regime


def _run_long(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    size: pd.Series,
    slip_mult: float = 1.0,
) -> dict:
    """VBT long-only backtest. Returns stats + Portfolio object."""
    sig  = signal_z.shift(1).fillna(0.0)
    med  = float(close.median()) or 1.0
    fees = SPREAD / med
    slip = (SLIPPAGE * slip_mult) / med

    pf = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size.values,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=fees,
        slippage=slip,
        freq=FREQ,
    )
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "ret":    float(pf.total_return()),
        "annual": float(pf.annualized_return()),
        "dd":     float(pf.max_drawdown()),
        "trades": int(n),
        "wr":     float(pf.trades.win_rate()) if n > 0 else 0.0,
        "value":  pf.value(),
        "pf":     pf,
    }


# ---------------------------------------------------------------------------
# Phase 3: IS/OOS backtest
# ---------------------------------------------------------------------------

def phase3(inst_data: dict) -> dict:
    inst    = inst_data["instrument"]
    close   = inst_data["close"]
    gz      = inst_data["gated_z"]
    size    = inst_data["size"]
    oos     = inst_data["oos_mask"]
    thr     = THRESHOLDS[inst]

    r = _run_long(close[oos], gz[oos], thr, size[oos])
    return r


# ---------------------------------------------------------------------------
# Phase 4: Walk-Forward Optimisation
# ---------------------------------------------------------------------------

def phase4_instrument(instrument: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Rolling WFO for one instrument. Returns (fold_df, oos_return_lists)."""
    close = df["close"]
    signals = build_all_signals(df)
    valid   = signals.notna().any(axis=1) & close.notna()
    signals = signals[valid]
    close   = close[valid]
    df_v    = df.loc[valid]

    n    = len(signals)
    thr  = THRESHOLDS[instrument]
    rows = []
    all_trade_returns = []

    fold = 0
    start = 0
    while start + WFO_IS + WFO_OOS <= n:
        is_idx  = range(start, start + WFO_IS)
        oos_idx = range(start + WFO_IS, start + WFO_IS + WFO_OOS)

        is_mask = pd.Series(False, index=signals.index)
        is_mask.iloc[list(is_idx)] = True

        sig_slice   = signals.iloc[list(is_idx) + list(oos_idx)]
        close_slice = close.iloc[list(is_idx) + list(oos_idx)]

        # IS mask relative to slice
        is_slice = pd.Series(
            [True] * WFO_IS + [False] * WFO_OOS,
            index=sig_slice.index,
        )
        gz_slice, _ = _build_gated_z(sig_slice, close_slice, is_slice)
        size_slice  = _size_series(
            df_v.iloc[list(is_idx) + list(oos_idx)], close_slice
        )

        # OOS portion
        oos_close = close_slice.iloc[WFO_IS:]
        oos_gz    = gz_slice.iloc[WFO_IS:]
        oos_size  = size_slice.iloc[WFO_IS:]

        r = _run_long(oos_close, oos_gz, thr, oos_size)

        # Collect per-trade % returns (compounding-safe)
        if r["trades"] > 0:
            try:
                tr = r["pf"].trades.records_readable["Return"].tolist()
                all_trade_returns.extend(tr)
            except Exception:
                pass

        is_r = _run_long(
            close_slice.iloc[:WFO_IS],
            gz_slice.iloc[:WFO_IS],
            thr,
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
    oos_data: dict,
    wfo_df: pd.DataFrame,
    wfo_trade_returns: list,
) -> dict:
    """Phase 5 robustness gates."""
    close  = oos_data["close"]
    gz     = oos_data["gated_z"]
    size   = oos_data["size"]
    thr    = THRESHOLDS[instrument]
    n_oos  = len(close)

    # -- Gate 1: Monte Carlo --------------------------------------------------
    n_trades_oos = oos_data["trades"]
    n_years      = n_oos / 252.0
    tpy          = n_trades_oos / n_years if n_years > 0 else 0.0
    mc = _mc_sharpe(wfo_trade_returns, MC_SIMS, tpy)

    # -- Gate 2: Remove top-N winning trades ----------------------------------
    if oos_data["trades"] > 0:
        try:
            trade_rets = oos_data["pf"].trades.records_readable["Return"].tolist()
            trade_rets_arr = np.array(trade_rets)
            top_idx = np.argsort(trade_rets_arr)[-TOP_N_REMOVE:]
            remaining = np.delete(trade_rets_arr, top_idx)
            topn_sum  = float(remaining.sum())
        except Exception:
            topn_sum = 0.0
    else:
        topn_sum = 0.0

    # -- Gate 3: 3× slippage --------------------------------------------------
    r_stress = _run_long(close, gz, thr, size, slip_mult=SLIP_MULT)

    # -- Gate 4: Consecutive negative WFO folds --------------------------------
    if len(wfo_df) > 0:
        neg_streak  = 0
        max_streak  = 0
        cur_streak  = 0
        for sh in wfo_df["oos_sharpe"]:
            if sh < 0:
                cur_streak += 1
                max_streak  = max(max_streak, cur_streak)
            else:
                cur_streak  = 0
        neg_streak = max_streak
    else:
        neg_streak = 0

    return {
        "mc_p5":       mc["p5"],
        "mc_p95":      mc["p95"],
        "mc_pct_pos":  mc["pct_pos"],
        "topn_sum":    topn_sum,
        "stress_sh":   r_stress["sharpe"],
        "neg_streak":  neg_streak,
        # Gates
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


def _print_wfo(instrument: str, wfo_df: pd.DataFrame) -> None:
    _hdr(f"PHASE 4 -- WALK-FORWARD  ({instrument})")
    print(f"  IS window: {WFO_IS} bars (2yr)  |  OOS window: {WFO_OOS} bars (6mo)")
    print(f"  Folds: {len(wfo_df)}  |  Stitched OOS coverage: "
          f"~{len(wfo_df)*WFO_OOS/252:.1f} years")
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

    # Gates
    n          = len(wfo_df)
    pct_pos    = (wfo_df["oos_sharpe"] > 0).mean()
    pct_gt1    = (wfo_df["oos_sharpe"] > 1).mean()
    worst      = float(wfo_df["oos_sharpe"].min())
    stitched   = float(wfo_df["oos_sharpe"].mean())   # arithmetic mean of fold Sharpes
    is_avg     = float(wfo_df["is_sharpe"].mean())
    parity     = stitched / is_avg if is_avg > 0 else 0.0

    print(f"\n  WFO Gate Summary  ({n} folds)")
    print(f"  {'> 0% folds':<28}: {pct_pos:.0%}  "
          f"{'[PASS]' if pct_pos >= 0.70 else '[FAIL]'}")
    print(f"  {'> 1 Sharpe folds':<28}: {pct_gt1:.0%}  "
          f"{'[PASS]' if pct_gt1 >= 0.50 else '[FAIL]'}")
    print(f"  {'Worst fold Sharpe':<28}: {worst:>+.3f}  "
          f"{'[PASS]' if worst >= -2.0 else '[FAIL]'}")
    print(f"  {'Mean OOS Sharpe':<28}: {stitched:>+.3f}  "
          f"{'[PASS]' if stitched >= 1.0 else '[FAIL]'}")
    print(f"  {'OOS/IS parity':<28}: {parity:>.3f}  "
          f"{'[PASS]' if parity >= 0.50 else '[FAIL]'}")

    return {
        "pct_pos": pct_pos, "pct_gt1": pct_gt1, "worst": worst,
        "stitched": stitched, "parity": parity,
    }


def _gate(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _gk(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_robustness(instrument: str, r5: dict) -> None:
    _hdr(f"PHASE 5 -- ROBUSTNESS  ({instrument})")
    print("  Gate 1 — Monte Carlo (1,000 shuffles, thr=5th pct > 0.5)")
    print(f"    5th-pct Sharpe  : {r5['mc_p5']:>+.3f}  {_gate(r5['g1_pass'])}")
    print(f"    95th-pct Sharpe : {r5['mc_p95']:>+.3f}")
    print(f"    % profitable    : {r5['mc_pct_pos']*100:.1f}%")
    print(f"  Gate 2 — Remove top {TOP_N_REMOVE} winning trades")
    print(f"    Remaining sum   : {r5['topn_sum']:>+.3f}  {_gate(r5['g2_pass'])}")
    print(f"  Gate 3 — {SLIP_MULT}x slippage stress test (thr: OOS Sharpe > 0.5)")
    print(f"    Stress Sharpe   : {r5['stress_sh']:>+.3f}  {_gate(r5['g3_pass'])}")
    print("  Gate 4 — Max consecutive negative WFO folds (thr: <= 2)")
    print(f"    Neg streak      : {r5['neg_streak']:>3}       {_gate(r5['g4_pass'])}")
    all_pass = r5["g1_pass"] and r5["g2_pass"] and r5["g3_pass"] and r5["g4_pass"]
    print(f"\n  Overall Phase 5: {'ALL PASS -- cleared for live' if all_pass else 'ONE OR MORE GATES FAILED'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    _hdr("CAT + AMAT | REGIME-GATED LONG-ONLY | FULL PIPELINE (PHASES 3-5)")
    print(f"  Instruments: {', '.join(INSTRUMENTS)}")
    print(f"  Thresholds : {THRESHOLDS}")
    print(f"  Costs      : Spread {SPREAD*100:.2f}%  Slippage {SLIPPAGE*100:.2f}%  Long-only")
    print("  Data       : 11yr daily (2015-2026)")

    # -------------------------------------------------------------------------
    # Load and prepare all instruments
    # -------------------------------------------------------------------------
    inst_data: dict[str, dict] = {}
    for inst in INSTRUMENTS:
        print(f"\nLoading {inst}...")
        df      = _load_ohlcv(inst, "D")
        close   = df["close"]
        signals = build_all_signals(df)
        valid   = signals.notna().any(axis=1) & close.notna()
        signals = signals[valid]
        close   = close[valid]
        df      = df.loc[valid]

        n     = len(signals)
        is_n  = int(n * IS_RATIO)
        is_mask = pd.Series(False, index=signals.index)
        is_mask.iloc[:is_n] = True
        oos_mask = ~is_mask

        gz, regime = _build_gated_z(signals, close, is_mask)
        size        = _size_series(df, close)

        print(f"  {n:,} bars | IS: {is_n:,} | OOS: {n-is_n:,}")
        print(f"  IS : {signals.index[0].date()} -> {signals.index[is_n-1].date()}")
        print(f"  OOS: {signals.index[is_n].date()} -> {signals.index[-1].date()}")

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
    # Phase 3
    # -------------------------------------------------------------------------
    _hdr("PHASE 3 -- IS/OOS BACKTEST (70/30 SPLIT)")
    print(f"  {'Inst':<8}  {'Thresh':>6}  {'Sharpe':>8}  {'Annual':>8}  "
          f"{'MaxDD':>8}  {'Ret':>8}  {'Trades':>7}  {'WR%':>6}")
    print("  " + "-" * (W - 2))

    p3_values = []
    for inst in INSTRUMENTS:
        d   = inst_data[inst]
        oos = d["oos_mask"]
        thr = THRESHOLDS[inst]
        r   = _run_long(d["close"][oos], d["gated_z"][oos], thr, d["size"][oos])
        inst_data[inst]["p3"] = r
        print(
            f"  {inst:<8}  {thr:>6.2f}  {r['sharpe']:>+8.3f}  "
            f"{r['annual']:>+8.1%}  {r['dd']:>+8.1%}  "
            f"{r['ret']:>+8.1%}  {r['trades']:>7}  {r['wr']*100:>6.1f}%"
        )
        p3_values.append(r["value"])

    # Portfolio
    common = p3_values[0].index.intersection(p3_values[1].index)
    port_val = sum(v.reindex(common) for v in p3_values)
    port_ret  = port_val.pct_change().dropna()
    port_sh   = float(port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0.0
    port_ann  = float((port_val.iloc[-1] / port_val.iloc[0]) ** (252 / len(port_ret)) - 1)
    port_dd   = float(((port_val - port_val.cummax()) / port_val.cummax()).min())
    print(f"  {'PORTFOLIO':<8}  {'mixed':>6}  {port_sh:>+8.3f}  "
          f"{port_ann:>+8.1%}  {port_dd:>+8.1%}  "
          f"{'':>8}  {'':>7}  {'':>6}")

    # Annual breakdown
    print("\n  Annual portfolio returns (OOS):")
    yr_rets = (
        pd.DataFrame({"r": port_ret})
        .assign(yr=lambda x: x.index.year)
        .groupby("yr")["r"]
        .apply(lambda x: (1 + x).prod() - 1)
    )
    for yr, val in yr_rets.items():
        bar = "#" * int(abs(val) * 200)
        sign = "+" if val >= 0 else "-"
        print(f"    {yr}  {val:>+7.1%}  {sign}{bar}")

    # -------------------------------------------------------------------------
    # Phase 4: WFO
    # -------------------------------------------------------------------------
    wfo_results: dict[str, pd.DataFrame] = {}
    wfo_trades:  dict[str, list]         = {}

    for inst in INSTRUMENTS:
        d = inst_data[inst]
        wfo_df, trade_rets = phase4_instrument(inst, d["df"])
        wfo_results[inst] = wfo_df
        wfo_trades[inst]  = trade_rets
        gate_summary = _print_wfo(inst, wfo_df)
        inst_data[inst]["wfo_gates"] = gate_summary

        out = REPORTS / f"cat_amat_wfo_{inst.lower()}.csv"
        wfo_df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")

    # -------------------------------------------------------------------------
    # Phase 5: Robustness
    # -------------------------------------------------------------------------
    for inst in INSTRUMENTS:
        d    = inst_data[inst]
        oos  = d["oos_mask"]
        oos_data = {
            "close":    d["close"][oos],
            "gated_z":  d["gated_z"][oos],
            "size":     d["size"][oos],
            "trades":   d["p3"]["trades"],
            "pf":       d["p3"]["pf"],
        }
        r5 = phase5(inst, oos_data, wfo_results[inst], wfo_trades[inst])
        inst_data[inst]["p5"] = r5
        _print_robustness(inst, r5)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    _hdr("PIPELINE SUMMARY")
    print(f"  {'Inst':<8}  {'P3 Sh':>8}  {'WFO>0%':>8}  {'WFO>1%':>8}  "
          f"{'Worst':>8}  {'MC p5':>8}  {'3xSlip':>8}  {'G1':>5}  {'G2':>5}  "
          f"{'G3':>5}  {'G4':>5}")
    print("  " + "-" * (W - 2))
    for inst in INSTRUMENTS:
        d  = inst_data[inst]
        g  = d.get("wfo_gates", {})
        r5 = d.get("p5", {})
        print(
            f"  {inst:<8}  {d['p3']['sharpe']:>+8.3f}  "
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
    for inst in INSTRUMENTS:
        d  = inst_data[inst]
        g  = d.get("wfo_gates", {})
        r5 = d.get("p5", {})
        rows.append({
            "instrument":  inst,
            "threshold":   THRESHOLDS[inst],
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
    out = REPORTS / "cat_amat_pipeline_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Summary saved: {out}")


if __name__ == "__main__":
    np.random.seed(42)
    run()
