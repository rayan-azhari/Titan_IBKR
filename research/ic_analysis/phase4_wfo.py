"""Phase 4 — Walk-Forward Optimisation (WFO).

Asset-class-agnostic rolling or anchored walk-forward validation.

Validates the MTF composite strategy from phase3_backtest.py across non-overlapping
OOS folds. Supports rolling (fixed IS window slides) and anchored (IS grows from
bar 0) modes.

Usage:
    uv run python research/ic_analysis/phase4_wfo.py
    uv run python research/ic_analysis/phase4_wfo.py \\
        --instrument EUR_USD --asset-class fx_major --wfo-type rolling
    uv run python research/ic_analysis/phase4_wfo.py \\
        --instrument AAPL --asset-class equity_lc --direction long_only \\
        --wfo-type anchored
    uv run python research/ic_analysis/phase4_wfo.py \\
        --instruments EUR_USD GBP_USD USD_JPY --asset-class fx_major
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from math import sqrt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed. Run: uv add vectorbt")
    sys.exit(1)

from research.ic_analysis.phase1_sweep import build_all_signals  # noqa: E402, F401
from research.ic_analysis.phase3_backtest import (  # noqa: E402
    COST_PROFILES,
    DEFAULT_RISK_PCT,
    DEFAULT_SIGNALS,
    DEFAULT_STOP_ATR,
    DEFAULT_TFS,
    INIT_CASH,
    _build_and_align,
    _compute_swap_drag,
    build_composite,
    build_size_array,
    zscore_normalise,
)

# -- Constants -----------------------------------------------------------------

# 0 = auto-compute from timeframe inside run_wfo()
# Override with --is-bars / --oos-bars CLI flags or is_bars / oos_bars kwargs.
# Auto defaults: IS = 2 × bars_per_year, OOS = bars_per_year // 2
#   D  -> IS=504,   OOS=126
#   H4 -> IS=4380,  OOS=1095
#   H1 -> IS=17520, OOS=4380
#   W  -> IS=104,   OOS=26
IS_BARS_DEFAULT: int = 0
OOS_BARS_DEFAULT: int = 0
DEFAULT_THRESHOLD: float = 0.75
# G7: Default threshold grid for per-fold search (same as Phase 3).
DEFAULT_THRESHOLD_GRID: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Quality gates
MIN_PCT_FOLDS_POSITIVE: float = 0.70
MIN_PCT_FOLDS_ABOVE_1: float = 0.50
MIN_WORST_FOLD_SHARPE: float = -2.0
# R6 FIX: Aligned to directive (was 1.5, directive says 1.0).
MIN_STITCHED_SHARPE: float = 1.0
MIN_AGGREGATE_PARITY: float = 0.50

__all__ = ["run_wfo", "IS_BARS_DEFAULT", "OOS_BARS_DEFAULT", "DEFAULT_THRESHOLD"]


# -- Local stats helper --------------------------------------------------------


def _stats(pf) -> dict:
    """Extract performance stats from a vbt Portfolio object."""
    try:
        sharpe = float(pf.sharpe_ratio())
    except Exception:
        sharpe = 0.0
    try:
        trades = int(pf.trades.count())
    except Exception:
        trades = 0
    try:
        wr = float(pf.trades.win_rate()) if trades > 0 else 0.0
    except Exception:
        wr = 0.0
    try:
        ret = float(pf.total_return())
    except Exception:
        ret = 0.0
    try:
        dd = float(pf.max_drawdown())
    except Exception:
        dd = 0.0
    return {"sharpe": sharpe, "trades": trades, "wr": wr, "ret": ret, "dd": dd}


# -- Fold VBT ------------------------------------------------------------------


def _fold_vbt(
    close: pd.Series,
    signal_z: pd.Series,
    threshold: float,
    spread: float,
    slippage: float,
    size: pd.Series,
    stop_pct: pd.Series,
    swap_long_pips: float,
    swap_short_pips: float,
    pip_size: float,
    direction: str = "both",
    freq: str = "h",
) -> dict:
    """Run VBT backtest on a single fold slice.

    When direction == 'long_only': skip short portfolio; combined = long only.
    When direction == 'both': run long + short portfolios and combine.

    Returns a dict with sharpe, trades, wr, ret, dd, gross_ret, net_ret,
    swap_drag_usd, and returns_series.
    """
    sig = signal_z.shift(1).fillna(0.0)
    size_arr = size.reindex(close.index).fillna(0.0).values
    stop_arr = stop_pct.reindex(close.index).fillna(0.0).values

    # dynamic arrays to ensure high-priced assets aren't undercharged relative to historic medians
    vbt_fees = (spread / close).bfill().values
    vbt_slip = (slippage / close).bfill().values

    pf_long = vbt.Portfolio.from_signals(
        close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        sl_stop=stop_arr,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )

    sl = _stats(pf_long)

    swap = _compute_swap_drag(
        pf_long.position_mask(),
        pd.Series(False, index=close.index),
        size,
        close,
        swap_long_pips,
        0.0,
        pip_size,
    )

    if direction == "long_only":
        gross_ret = sl["ret"]
        net_pnl = gross_ret * INIT_CASH + swap["total_drag_usd"]
        returns_series = pf_long.returns()
        return {
            "sharpe": sl["sharpe"],
            "trades": sl["trades"],
            "wr": sl["wr"],
            "ret": sl["ret"],
            "dd": sl["dd"],
            "gross_ret": gross_ret,
            "net_ret": net_pnl / INIT_CASH,
            "swap_drag_usd": swap["total_drag_usd"],
            "returns_series": returns_series,
        }

    # direction == "both"
    pf_short = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(False, index=close.index),
        exits=pd.Series(False, index=close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        sl_stop=stop_arr,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq=freq,
    )

    ss = _stats(pf_short)

    swap_short = _compute_swap_drag(
        pd.Series(False, index=close.index),
        pf_short.position_mask(),
        size,
        close,
        0.0,
        swap_short_pips,
        pip_size,
    )
    total_swap_drag = swap["total_drag_usd"] + swap_short["total_drag_usd"]

    combined_trades = sl["trades"] + ss["trades"]
    if combined_trades > 0:
        combined_wr = (
            sl["wr"] * sl["trades"] + ss["wr"] * ss["trades"]
        ) / combined_trades
    else:
        combined_wr = 0.0

    gross_ret = (sl["ret"] + ss["ret"]) / 2
    net_pnl = gross_ret * INIT_CASH + total_swap_drag

    # C2 FIX: Compute combined Sharpe from blended return stream.
    long_ret_s = pf_long.returns()
    short_ret_s = pf_short.returns()
    returns_series = (long_ret_s + short_ret_s) / 2
    bl_mean = float(returns_series.mean())
    bl_std = float(returns_series.std())
    combined_sharpe = (
        bl_mean / bl_std * sqrt(252) if bl_std > 1e-10 else 0.0
    )

    return {
        "sharpe": combined_sharpe,
        "trades": combined_trades,
        "wr": combined_wr,
        "ret": (sl["ret"] + ss["ret"]) / 2,
        # A4 FIX: Use max() for worst drawdown, not min().
        "dd": max(sl["dd"], ss["dd"]),
        "gross_ret": gross_ret,
        "net_ret": net_pnl / INIT_CASH,
        "swap_drag_usd": total_swap_drag,
        "returns_series": returns_series,
    }


# -- Walk-Forward Optimisation -------------------------------------------------


def run_wfo(
    instrument: str,
    target_signals: list[str],
    tfs: list[str],
    threshold: float,
    asset_class: str = "fx_major",
    direction: str = "both",
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    max_leverage: float | None = None,
    risk_pct: float = DEFAULT_RISK_PCT,
    stop_atr: float = DEFAULT_STOP_ATR,
    is_bars: int = IS_BARS_DEFAULT,
    oos_bars: int = OOS_BARS_DEFAULT,
    wfo_type: str = "rolling",
    threshold_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Run walk-forward optimisation for one instrument.

    Parameters
    ----------
    instrument:      IBKR-style symbol, e.g. "EUR_USD" or "AAPL".
    target_signals:  Signal names to include in the composite.
    tfs:             Timeframes to stack (e.g. ["W","D","H4","H1"]).
    threshold:       Z-score threshold for entry.
    asset_class:     Key into COST_PROFILES.
    direction:       "both" or "long_only".
    spread_bps:      Override profile spread (basis points).
    slippage_bps:    Override profile slippage (basis points).
    max_leverage:    Override profile max leverage.
    risk_pct:        Fraction of equity risked per trade.
    stop_atr:        ATR multiplier for stop distance.
    is_bars:         Number of bars in IS window (rolling) or minimum IS (anchored).
    oos_bars:        Number of bars per OOS fold.
    wfo_type:        "rolling" (fixed IS slides) or "anchored" (IS grows from 0).

    Returns
    -------
    pd.DataFrame with one row per fold.
    """
    print(f"\n{'='*70}")
    print(f"  Phase 4 WFO  |  {instrument}  |  {wfo_type}  |  {direction}")
    print(f"{'='*70}")

    # -- Cost profile ----------------------------------------------------------
    profile = dict(COST_PROFILES[asset_class])
    if spread_bps is not None:
        profile["spread_bps"] = spread_bps
    if slippage_bps is not None:
        profile["slippage_bps"] = slippage_bps
    if max_leverage is not None:
        profile["max_leverage"] = max_leverage

    eff_spread_bps: float = profile["spread_bps"]
    eff_slip_bps: float = profile["slippage_bps"]
    eff_max_lev: float = profile["max_leverage"]
    swap_long_pips: float = profile.get("swap_long", 0.0)
    swap_short_pips: float = profile.get("swap_short", 0.0)
    pip_size: float = profile.get("pip_size", 0.01)

    # -- Data ------------------------------------------------------------------
    base_tf = tfs[-1]
    freq = "d" if base_tf in ("D", "daily") else "h"
    bars_per_year: int = 252 if freq == "d" else 365 * 24

    # Auto-compute window sizes from timeframe when not explicitly set
    if is_bars == 0:
        is_bars = 2 * bars_per_year
    if oos_bars == 0:
        oos_bars = bars_per_year // 2

    print(f"  Base TF: {base_tf}  |  freq={freq}  |  bars_per_year={bars_per_year}")

    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=base_tf)
    base_close = base_df["close"]
    n = len(base_index)

    print(f"  Total bars: {n}  |  IS={is_bars}  |  OOS={oos_bars}  |  threshold={threshold}")

    # -- Position sizing (full series; sliced per fold) -----------------------
    size_full, stop_pct_full = build_size_array(base_df, base_close, risk_pct, stop_atr, eff_max_lev)

    # -- Fold boundaries -------------------------------------------------------
    if wfo_type == "rolling":
        fold_starts = list(range(0, n - is_bars - oos_bars + 1, oos_bars))
    else:
        # anchored: oos_start runs from is_bars to n-oos_bars
        fold_starts = list(range(is_bars, n - oos_bars + 1, oos_bars))

    if not fold_starts:
        msg = (
            f"Not enough data for WFO: n={n}, is_bars={is_bars}, "
            f"oos_bars={oos_bars}"
        )
        print(f"  [ERROR] {msg}")
        return pd.DataFrame()

    print(f"  Folds: {len(fold_starts)}")
    print()

    # -- Header ----------------------------------------------------------------
    hdr = (
        f"{'Fold':>5} | {'IS Start':>10} {'IS End':>10} | "
        f"{'OOS Start':>10} {'OOS End':>10} | "
        f"{'IS Sh':>7} {'OOS Sh':>7} {'Parity':>7} | "
        f"{'Trades':>7} {'WR':>6} {'MaxDD':>7} {'NetRet':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: list[dict] = []
    oos_returns_list: list[pd.Series] = []

    for fold_i, fold_start in enumerate(fold_starts):
        if wfo_type == "rolling":
            is_start = fold_start
            is_end = fold_start + is_bars
            oos_start = is_end
            oos_end = oos_start + oos_bars
        else:
            # anchored: fold_start IS the OOS start (IS end)
            is_start = 0
            is_end = fold_start
            oos_start = fold_start
            oos_end = fold_start + oos_bars

        # Guard
        if oos_end > n:
            oos_end = n
        if oos_end <= oos_start or is_end <= is_start:
            continue

        # IS mask
        is_mask = pd.Series(False, index=base_index)
        is_mask.iloc[is_start:is_end] = True

        # Composite & z-score (calibrated on this fold's IS)
        try:
            composite = build_composite(
                tf_signals, base_close, tfs, target_signals, is_mask
            )
            signal_z = zscore_normalise(composite, is_mask)
        except ValueError as exc:
            print(f"  [WARN] Fold {fold_i}: composite failed — {exc}")
            continue

        # Fold data slices
        is_close = base_close.iloc[is_start:is_end]
        oos_close = base_close.iloc[oos_start:oos_end]
        is_sig = signal_z.iloc[is_start:is_end]
        oos_sig = signal_z.iloc[oos_start:oos_end]
        is_size = size_full.iloc[is_start:is_end]
        oos_size = size_full.iloc[oos_start:oos_end]
        is_stop = stop_pct_full.iloc[is_start:is_end]
        oos_stop = stop_pct_full.iloc[oos_start:oos_end]

        # Convert spread/slippage bps -> price units using fold median close
        med_close_is = float(is_close.median()) or 1.0
        med_close_oos = float(oos_close.median()) or 1.0
        spread_is = eff_spread_bps / 10_000 * med_close_is
        slip_is = eff_slip_bps / 10_000 * med_close_is
        spread_oos = eff_spread_bps / 10_000 * med_close_oos
        slip_oos = eff_slip_bps / 10_000 * med_close_oos

        # G7 FIX: Per-fold threshold search on IS Sharpe.
        if threshold_grid is not None and len(threshold_grid) > 1:
            best_is_sharpe = -np.inf
            fold_threshold = threshold  # fallback
            for th_candidate in threshold_grid:
                sweep_res = _fold_vbt(
                    is_close, is_sig, th_candidate,
                    spread_is, slip_is, is_size, is_stop,
                    swap_long_pips, swap_short_pips, pip_size,
                    direction=direction, freq=freq,
                )
                if sweep_res["sharpe"] > best_is_sharpe:
                    best_is_sharpe = sweep_res["sharpe"]
                    fold_threshold = th_candidate
        else:
            fold_threshold = threshold

        # IS backtest (at best threshold for this fold)
        is_res = _fold_vbt(
            is_close, is_sig, fold_threshold,
            spread_is, slip_is, is_size, is_stop,
            swap_long_pips, swap_short_pips, pip_size,
            direction=direction, freq=freq,
        )

        # OOS backtest (at same threshold selected on IS)
        oos_res = _fold_vbt(
            oos_close, oos_sig, fold_threshold,
            spread_oos, slip_oos, oos_size, oos_stop,
            swap_long_pips, swap_short_pips, pip_size,
            direction=direction, freq=freq,
        )

        is_sharpe = is_res["sharpe"]
        oos_sharpe = oos_res["sharpe"]
        parity = oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-6 else 0.0

        is_start_dt = str(base_index[is_start].date())
        is_end_dt = str(base_index[is_end - 1].date())
        oos_start_dt = str(base_index[oos_start].date())
        oos_end_dt = str(base_index[oos_end - 1].date())

        row = {
            "fold": fold_i,
            "is_start": is_start_dt,
            "is_end": is_end_dt,
            "oos_start": oos_start_dt,
            "oos_end": oos_end_dt,
            "is_bars": is_end - is_start,
            "oos_bars": oos_end - oos_start,
            "is_sharpe": round(is_sharpe, 3),
            "oos_sharpe": round(oos_sharpe, 3),
            "parity": round(parity, 3),
            "oos_wr": round(oos_res["wr"], 4),
            "oos_trades": oos_res["trades"],
            "oos_max_dd": round(oos_res["dd"], 4),
            "oos_gross_ret": round(oos_res["gross_ret"], 4),
            "oos_net_ret": round(oos_res["net_ret"], 4),
            "swap_drag_usd": round(oos_res["swap_drag_usd"], 2),
        }
        rows.append(row)

        # Accumulate OOS returns for stitched equity
        oos_ret_s = oos_res["returns_series"]
        if oos_ret_s is not None and len(oos_ret_s) > 0:
            oos_returns_list.append(oos_ret_s)

        # Print fold row
        fold_line = (
            f"{fold_i:>5} | {is_start_dt:>10} {is_end_dt:>10} | "
            f"{oos_start_dt:>10} {oos_end_dt:>10} | "
            f"{is_sharpe:>7.2f} {oos_sharpe:>7.2f} {parity:>7.2f} | "
            f"{oos_res['trades']:>7d} {oos_res['wr']:>6.1%} "
            f"{oos_res['dd']:>7.2%} {oos_res['net_ret']:>8.2%}"
        )
        print(fold_line)

    if not rows:
        print("  [ERROR] No folds completed.")
        return pd.DataFrame()

    df_folds = pd.DataFrame(rows)

    # -- Aggregate quality report ----------------------------------------------
    print()
    print("  AGGREGATE QUALITY REPORT")
    print("  " + "-" * 50)

    pct_positive = (df_folds["oos_sharpe"] > 0).mean()
    pct_above_1 = (df_folds["oos_sharpe"] > 1.0).mean()
    worst_sharpe = float(df_folds["oos_sharpe"].min())
    agg_parity = float(df_folds["parity"].mean())

    # Stitched OOS equity
    if oos_returns_list:
        stitched = pd.concat(oos_returns_list).sort_index()
        bpy = float(bars_per_year)
        ann_ret = float(stitched.mean()) * bpy
        ann_vol = float(stitched.std()) * float(np.sqrt(bpy))
        stitched_sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    else:
        stitched = pd.Series(dtype=float)
        stitched_sharpe = 0.0

    gate1_ok = pct_positive >= MIN_PCT_FOLDS_POSITIVE
    gate2_ok = pct_above_1 >= MIN_PCT_FOLDS_ABOVE_1
    gate3_ok = worst_sharpe >= MIN_WORST_FOLD_SHARPE
    gate4_ok = stitched_sharpe >= MIN_STITCHED_SHARPE
    gate5_ok = agg_parity >= MIN_AGGREGATE_PARITY

    def _glyph(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    g1_line = (
        f"  [{'PASS' if gate1_ok else 'FAIL'}] "
        f"%folds OOS>0 = {pct_positive:.1%}"
        f"  (need >= {MIN_PCT_FOLDS_POSITIVE:.0%})"
    )
    g2_line = (
        f"  [{_glyph(gate2_ok)}] "
        f"%folds OOS>1 = {pct_above_1:.1%}"
        f"  (need >= {MIN_PCT_FOLDS_ABOVE_1:.0%})"
    )
    g3_line = (
        f"  [{_glyph(gate3_ok)}] "
        f"Worst fold Sharpe = {worst_sharpe:.2f}"
        f"  (need >= {MIN_WORST_FOLD_SHARPE:.1f})"
    )
    g4_line = (
        f"  [{_glyph(gate4_ok)}] "
        f"Stitched Sharpe = {stitched_sharpe:.2f}"
        f"  (need >= {MIN_STITCHED_SHARPE:.1f})"
    )
    g5_line = (
        f"  [{_glyph(gate5_ok)}] "
        f"Aggregate parity = {agg_parity:.2f}"
        f"  (need >= {MIN_AGGREGATE_PARITY:.2f})"
    )

    print(g1_line)
    print(g2_line)
    print(g3_line)
    print(g4_line)
    print(g5_line)

    all_pass = gate1_ok and gate2_ok and gate3_ok and gate4_ok and gate5_ok
    verdict = "ACCEPTED" if all_pass else "REJECTED"
    print()
    n_pass = sum([gate1_ok, gate2_ok, gate3_ok, gate4_ok, gate5_ok])
    print(f"  VERDICT: {verdict} ({n_pass}/5 gates)")

    # -- Save outputs ----------------------------------------------------------
    slug = instrument.lower()
    if wfo_type != "rolling":
        slug = f"{slug}_{wfo_type}"

    fold_csv = REPORTS_DIR / f"phase4_{slug}.csv"
    df_folds.to_csv(fold_csv, index=False)
    print(f"\n  Saved fold stats : {fold_csv}")

    if len(stitched) > 0:
        equity_csv = REPORTS_DIR / f"phase4_equity_{slug}.csv"
        stitched.to_csv(equity_csv, header=True)
        print(f"  Saved OOS equity : {equity_csv}")

    return df_folds


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 4 — Walk-Forward Optimisation (WFO)"
    )
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Batch mode - overrides --instrument",
    )
    p.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    p.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument(
        "--asset-class",
        default="fx_major",
        choices=list(COST_PROFILES.keys()),
    )
    p.add_argument(
        "--direction",
        default="both",
        choices=["both", "long_only"],
    )
    p.add_argument("--spread-bps", type=float, default=None)
    p.add_argument("--slippage-bps", type=float, default=None)
    p.add_argument("--max-leverage", type=float, default=None)
    p.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    p.add_argument("--stop-atr", type=float, default=DEFAULT_STOP_ATR)
    p.add_argument("--is-bars", type=int, default=IS_BARS_DEFAULT)
    p.add_argument("--oos-bars", type=int, default=OOS_BARS_DEFAULT)
    p.add_argument(
        "--wfo-type",
        default="rolling",
        choices=["rolling", "anchored"],
    )
    args = p.parse_args()

    target_signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    instruments = args.instruments if args.instruments else [args.instrument]

    summary_rows: list[dict] = []

    for instr in instruments:
        df_folds = run_wfo(
            instrument=instr,
            target_signals=target_signals,
            tfs=tfs,
            threshold=args.threshold,
            asset_class=args.asset_class,
            direction=args.direction,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            max_leverage=args.max_leverage,
            risk_pct=args.risk_pct,
            stop_atr=args.stop_atr,
            is_bars=args.is_bars,
            oos_bars=args.oos_bars,
            wfo_type=args.wfo_type,
        )

        if df_folds.empty:
            summary_rows.append({"instrument": instr, "folds": 0, "status": "NO_DATA"})
            continue

        n_folds = len(df_folds)
        pct_pos = (df_folds["oos_sharpe"] > 0).mean()
        mean_oos_sharpe = df_folds["oos_sharpe"].mean()
        mean_parity = df_folds["parity"].mean()
        summary_rows.append({
            "instrument": instr,
            "folds": n_folds,
            "pct_oos_positive": round(pct_pos, 3),
            "mean_oos_sharpe": round(mean_oos_sharpe, 3),
            "mean_parity": round(mean_parity, 3),
            "status": "OK",
        })

    if len(instruments) > 1:
        print()
        print("=" * 70)
        print("  MULTI-INSTRUMENT SUMMARY")
        print("=" * 70)
        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
