"""run_wfo.py -- Walk-Forward Validation (Phase 4).

Validates the MTF IC strategy using rolling out-of-sample windows instead of
a single IS/OOS split.  Gives a realistic picture of how the strategy would
have performed if re-calibrated periodically throughout its live history.

Walk-forward design (rolling, not expanding):
    IS window  : 2 years  (17,520 H1 bars)  -- calibrate IC signs + z-score
    OOS window : 6 months ( 4,380 H1 bars)  -- evaluate with fixed threshold
    Step       : 6 months (roll OOS forward one window at a time)

What is re-fitted per fold (on IS data only):
    - IC sign direction per signal x TF (so composite > 0 = bullish)
    - Z-score mean and std (applied to both IS and OOS)
    - Nothing else -- threshold is FIXED across all folds

Quality criteria (aggregate across all folds):
    - >= 70% of folds with OOS Sharpe > 0
    - >= 50% of folds with OOS Sharpe > 1
    - Worst-fold Sharpe >= -2
    - Stitched OOS Sharpe (all fold returns concatenated) >= 1.5
    - OOS/IS Sharpe ratio >= 0.5 on the aggregate

Costs (same as run_ic_backtest.py):
    Spread   : per-instrument default (pip-sized, normalised by close)
    Slippage : 0.5 pip per fill
    Swap     : overnight pips/night (per-instrument default)

Output:
    Console: per-fold table + aggregate quality report
    CSV:     .tmp/reports/wfo_{slug}.csv           (per-fold stats)
             .tmp/reports/wfo_equity_{slug}.csv    (stitched OOS returns)

Usage:
    uv run python research/ic_analysis/run_wfo.py
    uv run python research/ic_analysis/run_wfo.py --instrument GBP_USD --threshold 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.run_ic_backtest import (  # noqa: E402
    INIT_CASH,
    MAX_LEVERAGE,
    PIP_SIZE,
    SPREAD_DEFAULTS,
    SWAP_DEFAULTS,
    _compute_swap_drag,
    _load_tf,
    _stats,
    build_composite,
    build_size_array,
    zscore_normalise,
)
from research.ic_analysis.run_signal_sweep import build_all_signals  # noqa: E402

# -- Walk-forward parameters ---------------------------------------------------

IS_BARS = 2 * 365 * 24        # 2-year IS window in H1 bars
OOS_BARS = 6 * 30 * 24        # 6-month OOS window in H1 bars (approx)
DEFAULT_SIGNALS = ["accel_stoch_k", "accel_rsi14"]
DEFAULT_TFS = ["W", "D", "H4", "H1"]
DEFAULT_THRESHOLD = 0.75

# Quality thresholds
MIN_PCT_FOLDS_POSITIVE = 0.70
MIN_PCT_FOLDS_ABOVE_1 = 0.50
MIN_WORST_FOLD_SHARPE = -2.0
MIN_STITCHED_SHARPE = 1.5
MIN_AGGREGATE_PARITY = 0.50


# -- Helpers -------------------------------------------------------------------


def _build_and_align_all(
    instrument: str,
    tfs: list[str],
    base_tf: str = "H1",
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, pd.DataFrame]:
    base_df = _load_tf(instrument, base_tf)
    if base_df is None:
        raise FileNotFoundError(f"H1 data not found for {instrument}")
    base_index = base_df.index

    tf_signals: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        df = _load_tf(instrument, tf)
        if df is None:
            print(f"  [WARN] No data for {instrument} {tf} -- skipping.")
            continue
        native_sigs = build_all_signals(df)
        
        # PREVENT LOOKAHEAD BIAS
        if tf != base_tf:
            native_sigs = native_sigs.shift(1)
            
        aligned = native_sigs.reindex(base_index, method="ffill")
        tf_signals[tf] = aligned

    return tf_signals, base_index, base_df


def _fold_vbt(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    spread: float,
    slippage: float,
    size: pd.Series,
    swap_long_pips: float,
    swap_short_pips: float,
    pip_size: float,
) -> dict:
    """Run one fold with VBT. Returns stats + per-bar return series."""
    sig = signal_z.shift(1).fillna(0.0)
    size_arr = size.reindex(close.index).fillna(0.0).values
    med_close = float(close.median()) or 1.0
    vbt_fees = spread / med_close
    vbt_slip = slippage / med_close

    pf_long = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq="h",
    )
    pf_short = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq="h",
    )

    sl = _stats(pf_long)
    ss = _stats(pf_short)

    swap = _compute_swap_drag(
        pf_long.position_mask(),
        pf_short.position_mask(),
        size,
        close,
        swap_long_pips,
        swap_short_pips,
        pip_size,
    )

    combined_trades = sl["trades"] + ss["trades"]
    combined_wr = (
        (sl["wr"] * sl["trades"] + ss["wr"] * ss["trades"]) / combined_trades
        if combined_trades > 0 else 0.0
    )
    gross_ret = (sl["ret"] + ss["ret"]) / 2
    combined_rets = (pf_long.returns() + pf_short.returns()) / 2

    return {
        "long_sharpe": sl["sharpe"],
        "short_sharpe": ss["sharpe"],
        "combined_sharpe": (sl["sharpe"] + ss["sharpe"]) / 2,
        "long_ret": sl["ret"],
        "short_ret": ss["ret"],
        "gross_ret": gross_ret,
        "net_ret": (gross_ret * INIT_CASH + swap["total_drag_usd"]) / INIT_CASH,
        "long_dd": sl["dd"],
        "short_dd": ss["dd"],
        "combined_trades": combined_trades,
        "combined_wr": combined_wr,
        "swap_drag_usd": swap["total_drag_usd"],
        "long_nights": swap["long_nights"],
        "short_nights": swap["short_nights"],
        "returns_series": combined_rets,
    }


# -- Main WFO pipeline ---------------------------------------------------------


def run_wfo(
    instrument: str,
    target_signals: list[str],
    tfs: list[str],
    threshold: float,
    spread,
    risk_pct: float,
    stop_atr: float,
    slippage_pips: float,
    swap_long_pips,
    swap_short_pips,
    is_bars: int = IS_BARS,
    oos_bars: int = OOS_BARS,
) -> pd.DataFrame:
    slug = instrument.lower()
    base_tf = tfs[-1]

    pip_size = PIP_SIZE.get(instrument, 0.0001)
    eff_spread = spread if spread is not None else SPREAD_DEFAULTS.get(instrument, 0.00005)
    eff_swap_long = swap_long_pips if swap_long_pips is not None else (
        SWAP_DEFAULTS.get(instrument, {}).get("long", -0.5)
    )
    eff_swap_short = swap_short_pips if swap_short_pips is not None else (
        SWAP_DEFAULTS.get(instrument, {}).get("short", 0.3)
    )
    slippage = slippage_pips * pip_size

    W = 84
    print()
    print("=" * W)
    print(f"  WALK-FORWARD VALIDATION -- {instrument}")
    print(f"  Signals  : {' + '.join(target_signals)}")
    print(f"  TFs      : {' -> '.join(tfs)}")
    print(f"  IS       : {is_bars:,} bars (~{is_bars / (365 * 24):.1f} yr)  |  "
          f"OOS: {oos_bars:,} bars (~{oos_bars / (30 * 24):.1f} mo/fold)")
    print(f"  Threshold: {threshold:.2f}z (fixed)  |  "
          f"Spread: {eff_spread / pip_size:.2f} pip  |  "
          f"Slip: {slippage_pips:.2f} pip  |  Risk: {risk_pct * 100:.1f}%")
    print("=" * W)

    # 1. Load data once
    print("\n  Loading data...")
    tf_signals, base_index, base_df = _build_and_align_all(instrument, tfs, base_tf)
    base_close = base_df["close"]
    n = len(base_index)
    print(f"  H1 bars: {n:,}  ({base_index[0].date()} -> {base_index[-1].date()})")

    # 2. ATR sizing (once on full series)
    size_full = build_size_array(base_df, base_close, risk_pct, stop_atr, MAX_LEVERAGE)

    # 3. Define folds
    fold_starts = list(range(0, n - is_bars - oos_bars + 1, oos_bars))
    n_folds = len(fold_starts)
    if n_folds == 0:
        print(f"\n  [ERROR] Not enough data. Need >= {is_bars + oos_bars:,} bars, have {n:,}.")
        return pd.DataFrame()

    print(f"\n  Folds: {n_folds}  (IS={is_bars:,} bars rolling, step={oos_bars:,} bars)\n")

    hdr = (f"  {'#':>4}  {'OOS Start':>12}  {'OOS End':>12}  "
           f"{'IS Sharpe':>10}  {'OOS Sharpe':>11}  "
           f"{'Parity':>7}  {'WR%':>6}  {'MaxDD%':>7}  {'Trades':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    fold_results: list[dict] = []
    stitched_returns: list[pd.Series] = []

    for i, fold_start in enumerate(fold_starts):
        is_start = fold_start
        is_end = fold_start + is_bars
        oos_start = is_end
        oos_end = min(oos_start + oos_bars, n)

        if oos_end - oos_start < oos_bars // 2:
            break

        is_mask_fold = pd.Series(False, index=base_index)
        is_mask_fold.iloc[is_start:is_end] = True

        composite = build_composite(tf_signals, base_close, tfs, target_signals, is_mask_fold)
        composite_z = zscore_normalise(composite, is_mask_fold)

        is_res = _fold_vbt(
            base_close.iloc[is_start:is_end],
            composite_z.iloc[is_start:is_end],
            threshold, eff_spread, slippage,
            size_full.iloc[is_start:is_end],
            eff_swap_long, eff_swap_short, pip_size,
        )
        oos_res = _fold_vbt(
            base_close.iloc[oos_start:oos_end],
            composite_z.iloc[oos_start:oos_end],
            threshold, eff_spread, slippage,
            size_full.iloc[oos_start:oos_end],
            eff_swap_long, eff_swap_short, pip_size,
        )

        parity = (
            oos_res["combined_sharpe"] / is_res["combined_sharpe"]
            if is_res["combined_sharpe"] != 0 else 0.0
        )
        worst_dd = min(oos_res["long_dd"], oos_res["short_dd"]) * 100
        flag = " *" if oos_res["combined_sharpe"] > 0 else ""

        print(
            f"  {i + 1:>4}  "
            f"{str(base_index[oos_start].date()):>12}  "
            f"{str(base_index[oos_end - 1].date()):>12}  "
            f"{is_res['combined_sharpe']:>+10.3f}  "
            f"{oos_res['combined_sharpe']:>+11.3f}  "
            f"{parity:>7.3f}  "
            f"{oos_res['combined_wr'] * 100:>5.1f}%  "
            f"{worst_dd:>6.1f}%  "
            f"{oos_res['combined_trades']:>7.0f}{flag}"
        )

        fold_results.append({
            "fold": i + 1,
            "is_start": str(base_index[is_start].date()),
            "is_end": str(base_index[is_end - 1].date()),
            "oos_start": str(base_index[oos_start].date()),
            "oos_end": str(base_index[oos_end - 1].date()),
            "is_sharpe": is_res["combined_sharpe"],
            "oos_sharpe": oos_res["combined_sharpe"],
            "oos_long_sharpe": oos_res["long_sharpe"],
            "oos_short_sharpe": oos_res["short_sharpe"],
            "parity": parity,
            "oos_wr": oos_res["combined_wr"],
            "oos_long_dd": oos_res["long_dd"],
            "oos_short_dd": oos_res["short_dd"],
            "oos_trades": oos_res["combined_trades"],
            "oos_gross_ret": oos_res["gross_ret"],
            "oos_net_ret": oos_res["net_ret"],
            "oos_long_ret": oos_res["long_ret"],
            "oos_short_ret": oos_res["short_ret"],
            "swap_drag_usd": oos_res["swap_drag_usd"],
        })
        stitched_returns.append(oos_res["returns_series"])

    df = pd.DataFrame(fold_results)
    if df.empty:
        print("  [ERROR] No folds completed.")
        return df

    # 4. Aggregate
    pct_positive = float((df["oos_sharpe"] > 0).mean())
    pct_above_1 = float((df["oos_sharpe"] > 1.0).mean())
    worst_sharpe = float(df["oos_sharpe"].min())
    mean_is_sharpe = float(df["is_sharpe"].mean())
    mean_oos_sharpe = float(df["oos_sharpe"].mean())
    median_oos_sharpe = float(df["oos_sharpe"].median())
    mean_oos_wr = float(df["oos_wr"].mean())
    n_folds_done = len(df)

    stitched = pd.concat(stitched_returns).sort_index()
    stitched_n_years = len(stitched) / (365 * 24)
    stitched_ann_vol = float(stitched.std() * np.sqrt(365 * 24))
    stitched_ann_ret = float(stitched.mean() * 365 * 24)
    stitched_sharpe = stitched_ann_ret / stitched_ann_vol if stitched_ann_vol > 0 else 0.0
    aggregate_parity = mean_oos_sharpe / mean_is_sharpe if mean_is_sharpe != 0 else 0.0

    gate_pct_pos = pct_positive >= MIN_PCT_FOLDS_POSITIVE
    gate_pct_1 = pct_above_1 >= MIN_PCT_FOLDS_ABOVE_1
    gate_worst = worst_sharpe >= MIN_WORST_FOLD_SHARPE
    gate_stitch = stitched_sharpe >= MIN_STITCHED_SHARPE
    gate_parity = aggregate_parity >= MIN_AGGREGATE_PARITY
    all_pass = all([gate_pct_pos, gate_pct_1, gate_worst, gate_stitch, gate_parity])

    print()
    print("=" * W)
    print(f"  AGGREGATE  ({n_folds_done} folds | {stitched_n_years:.1f} yr stitched OOS)")
    print("=" * W)

    def _g(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  {'% folds OOS Sharpe > 0':<40}  {pct_positive:>8.1%}  >=70%  {_g(gate_pct_pos)}")
    print(f"  {'% folds OOS Sharpe > 1':<40}  {pct_above_1:>8.1%}  >=50%  {_g(gate_pct_1)}")
    print(f"  {'Worst fold OOS Sharpe':<40}  {worst_sharpe:>+9.3f}  >= -2  {_g(gate_worst)}")
    print(f"  {'Stitched OOS Sharpe (arith.)':<40}  {stitched_sharpe:>+9.3f}  >=1.5  {_g(gate_stitch)}")
    print(f"  {'Aggregate OOS/IS parity':<40}  {aggregate_parity:>+9.3f}  >=0.5  {_g(gate_parity)}")
    print("  " + "-" * 60)
    print(f"  {'Mean IS Sharpe':<40}  {mean_is_sharpe:>+9.3f}")
    print(f"  {'Mean OOS Sharpe':<40}  {mean_oos_sharpe:>+9.3f}")
    print(f"  {'Median OOS Sharpe':<40}  {median_oos_sharpe:>+9.3f}")
    print(f"  {'Mean OOS Win Rate':<40}  {mean_oos_wr:>8.1%}")
    print()

    if all_pass:
        print("  VERDICT: ALL GATES PASSED -- proceed to Phase 6 (titan/ implementation).")
    else:
        failed = []
        if not gate_pct_pos:
            failed.append(f"folds_positive ({pct_positive:.0%}<70%)")
        if not gate_pct_1:
            failed.append(f"folds_sharpe>1 ({pct_above_1:.0%}<50%)")
        if not gate_worst:
            failed.append(f"worst_fold ({worst_sharpe:.2f}<-2)")
        if not gate_stitch:
            failed.append(f"stitched_sharpe ({stitched_sharpe:.2f}<1.5)")
        if not gate_parity:
            failed.append(f"parity ({aggregate_parity:.2f}<0.5)")
        print(f"  VERDICT: FAILED -- {', '.join(failed)}.")

    # 5. Save
    out_fold = REPORTS_DIR / f"wfo_{slug}.csv"
    df.to_csv(out_fold, index=False)
    equity_df = stitched.to_frame(name="return")
    out_eq = REPORTS_DIR / f"wfo_equity_{slug}.csv"
    equity_df.to_csv(out_eq)
    print()
    print(f"  Fold stats  : {out_fold}")
    print(f"  Equity curve: {out_eq}")
    print("=" * W)

    return df


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="MTF IC Walk-Forward Validation")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    p.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--spread", type=float, default=None)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--stop-atr", type=float, default=1.5)
    p.add_argument("--slippage-pips", type=float, default=0.5)
    p.add_argument("--swap-long", type=float, default=None)
    p.add_argument("--swap-short", type=float, default=None)
    p.add_argument("--is-bars", type=int, default=IS_BARS)
    p.add_argument("--oos-bars", type=int, default=OOS_BARS)
    args = p.parse_args()

    run_wfo(
        instrument=args.instrument,
        target_signals=[s.strip() for s in args.signals.split(",")],
        tfs=[t.strip() for t in args.tfs.split(",")],
        threshold=args.threshold,
        spread=args.spread,
        risk_pct=args.risk_pct,
        stop_atr=args.stop_atr,
        slippage_pips=args.slippage_pips,
        swap_long_pips=args.swap_long,
        swap_short_pips=args.swap_short,
        is_bars=args.is_bars,
        oos_bars=args.oos_bars,
    )


if __name__ == "__main__":
    main()
