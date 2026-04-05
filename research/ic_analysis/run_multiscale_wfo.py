"""run_multiscale_wfo.py -- Walk-Forward Validation for Multi-Scale IC Signals.

Takes the top multi-scale signals from run_multiscale_ic.py and validates them
via rolling WFO. Per fold: re-rank signals by IC on IS, build composite, test
on OOS. No future information leaks across folds.

Usage:
    uv run python research/ic_analysis/run_multiscale_wfo.py --pair GLD
    uv run python research/ic_analysis/run_multiscale_wfo.py --pair USD_JPY --top-n 5
    uv run python research/ic_analysis/run_multiscale_wfo.py --pair GLD --is-bars 10000 --oos-bars 2500

Directive: Backtesting & Validation.md
"""

import argparse
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    SCALE_MAP,
    _load_ohlcv,
    build_multiscale_signals,
)
from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
)
from research.ic_analysis.run_multiscale_ic import (  # noqa: E402
    WINDOW_1Y_H1,
    _fast_ic_table,
)

HORIZONS = [1, 5, 10, 20]

# Quality gates
MIN_PCT_FOLDS_POSITIVE = 0.60
MIN_STITCHED_SHARPE = 0.50
MIN_PARITY = 0.40


# ── Vectorized position simulation ──────────────────────────────────────────


def _vectorized_position(sig: np.ndarray, threshold: float, exit_buf: float) -> np.ndarray:
    n = len(sig)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        s = sig[i]
        if cur == 0.0:
            if s >= threshold:
                cur = 1.0
            elif s <= -threshold:
                cur = -1.0
        elif cur > 0:
            if s < -exit_buf:
                cur = 0.0
        elif cur < 0:
            if s > exit_buf:
                cur = 0.0
        pos[i] = cur
    return pos


# ── Per-fold backtest ────────────────────────────────────────────────────────


def _fold_backtest(
    close_vals: np.ndarray,
    score_z_vals: np.ndarray,
    threshold: float,
    exit_buffer: float,
    spread_bps: float,
    slippage_bps: float,
) -> dict:
    """Backtest a single fold. Returns Sharpe + basic metrics."""
    n = len(close_vals)
    # Shift signal by 1
    sig = np.empty(n)
    sig[0] = 0.0
    sig[1:] = score_z_vals[:-1]

    pos = _vectorized_position(sig, threshold, exit_buffer)

    bar_rets = np.empty(n)
    bar_rets[0] = 0.0
    bar_rets[1:] = (close_vals[1:] - close_vals[:-1]) / close_vals[:-1]

    transitions = np.empty(n)
    transitions[0] = abs(pos[0])
    transitions[1:] = np.abs(pos[1:] - pos[:-1])

    strat_rets = bar_rets * pos - transitions * (spread_bps + slippage_bps) / 10_000

    # Daily aggregation
    daily = pd.Series(strat_rets).groupby(np.arange(n) // 7).sum()  # ~7 H1 bars/day for equities
    daily = daily[daily != 0.0]

    if len(daily) < 10:
        return {"sharpe": 0.0, "ret_pct": 0.0, "dd_pct": 0.0, "trades": 0, "days": 0}

    ann_ret = float(daily.mean() * 252)
    ann_vol = float(daily.std() * sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

    equity = (1 + daily).cumprod()
    dd = float(((equity - equity.cummax()) / equity.cummax()).min())
    trades = int(np.sum(np.abs(np.diff(pos)) > 0))

    return {
        "sharpe": round(sharpe, 3),
        "ret_pct": round(ann_ret * 100, 2),
        "dd_pct": round(dd * 100, 2),
        "trades": trades,
        "days": len(daily),
    }


# ── WFO Engine ───────────────────────────────────────────────────────────────


def run_wfo(
    pair: str,
    top_n: int = 5,
    is_bars: int = 10_000,
    oos_bars: int = 2_500,
    threshold: float = 0.75,
    exit_buffer: float = 0.10,
    spread_bps: float = 2.0,
    slippage_bps: float = 1.0,
    scales: dict[str, int] | None = None,
) -> dict:
    """Rolling WFO with per-fold IC re-ranking."""
    if scales is None:
        scales = SCALE_MAP

    # Load data
    print(f"\n  Loading {pair} H1...")
    df = _load_ohlcv(pair, "H1")
    close = df["close"]
    close_vals = close.values.astype(np.float64)
    n_total = len(close)
    print(f"  {n_total} bars ({close.index[0].date()} -> {close.index[-1].date()})")

    # Compute all multi-scale signals once
    print(f"  Computing multi-scale signals ({len(scales)} scales)...")
    t0 = time.time()
    signals = build_multiscale_signals(df, WINDOW_1Y_H1, scales=scales)
    # Drop features with >70% NaN
    valid = signals.columns[signals.notna().sum() > len(signals) * 0.3]
    signals = signals[valid]
    print(f"  {len(signals.columns)} features in {time.time() - t0:.1f}s")

    # Forward returns for IC computation
    fwd = compute_forward_returns(close, HORIZONS, vol_adjust=True)

    if n_total < is_bars + oos_bars:
        print(f"  ERROR: Need {is_bars + oos_bars} bars, have {n_total}")
        sys.exit(1)

    # Rolling WFO
    fold_results = []
    stitched_rets = []
    fold_idx = 0
    oos_start = is_bars

    print(f"\n  Running WFO: IS={is_bars} OOS={oos_bars} threshold={threshold}...")

    while oos_start + oos_bars <= n_total:
        is_start = oos_start - is_bars
        is_end = oos_start
        oos_end = oos_start + oos_bars

        # ── IS: rank signals by IC ──
        is_signals = signals.iloc[is_start:is_end]
        is_fwd = fwd.iloc[is_start:is_end]

        ic_is = _fast_ic_table(is_signals, is_fwd)
        # Pick best horizon per signal, rank by |IC|
        best_ic = ic_is.abs().max(axis=1).sort_values(ascending=False)
        best_h = ic_is.abs().idxmax(axis=1)
        ic_sign = pd.Series({sig: np.sign(ic_is.loc[sig, best_h[sig]]) for sig in ic_is.index})

        # Top-N signals for this fold
        top_sigs = best_ic.head(top_n).index.tolist()

        # Build IS composite: sign-oriented equal-weight average
        is_oriented = pd.DataFrame(index=is_signals.index)
        for sig in top_sigs:
            is_oriented[sig] = is_signals[sig] * ic_sign[sig]
        is_composite = is_oriented.mean(axis=1)

        # Z-score calibration on IS
        is_mean = float(is_composite.mean())
        is_std = float(is_composite.std())
        if is_std < 1e-8:
            is_std = 1.0

        # ── OOS: apply frozen composite ──
        oos_signals = signals.iloc[oos_start:oos_end]
        oos_oriented = pd.DataFrame(index=oos_signals.index)
        for sig in top_sigs:
            oos_oriented[sig] = oos_signals[sig] * ic_sign[sig]
        oos_composite = oos_oriented.mean(axis=1)
        oos_z = ((oos_composite - is_mean) / is_std).values

        # IS z-score for parity
        is_z = ((is_composite - is_mean) / is_std).values

        # Backtest both
        is_result = _fold_backtest(
            close_vals[is_start:is_end],
            is_z,
            threshold,
            exit_buffer,
            spread_bps,
            slippage_bps,
        )
        oos_result = _fold_backtest(
            close_vals[oos_start:oos_end],
            oos_z,
            threshold,
            exit_buffer,
            spread_bps,
            slippage_bps,
        )

        parity = (
            oos_result["sharpe"] / is_result["sharpe"] if abs(is_result["sharpe"]) > 0.01 else 0.0
        )

        fold_results.append(
            {
                "fold": fold_idx,
                "is_start": close.index[is_start].strftime("%Y-%m-%d"),
                "oos_start": close.index[oos_start].strftime("%Y-%m-%d"),
                "oos_end": close.index[min(oos_end - 1, n_total - 1)].strftime("%Y-%m-%d"),
                "is_sharpe": is_result["sharpe"],
                "oos_sharpe": oos_result["sharpe"],
                "parity": round(parity, 3),
                "oos_ret_pct": oos_result["ret_pct"],
                "oos_dd_pct": oos_result["dd_pct"],
                "oos_trades": oos_result["trades"],
                "top_signals": ", ".join(s[:20] for s in top_sigs[:3]),
            }
        )

        # Collect OOS bar returns for stitching
        sig_shifted = np.empty(oos_end - oos_start)
        sig_shifted[0] = 0.0
        sig_shifted[1:] = oos_z[:-1]
        pos = _vectorized_position(sig_shifted, threshold, exit_buffer)
        br = np.empty(oos_end - oos_start)
        br[0] = 0.0
        c = close_vals[oos_start:oos_end]
        br[1:] = (c[1:] - c[:-1]) / c[:-1]
        trans = np.empty(len(pos))
        trans[0] = abs(pos[0])
        trans[1:] = np.abs(pos[1:] - pos[:-1])
        stitched_rets.append(br * pos - trans * (spread_bps + slippage_bps) / 10_000)

        fold_idx += 1
        oos_start += oos_bars

    # Stitch
    all_oos = np.concatenate(stitched_rets)
    stitched_daily = pd.Series(all_oos).groupby(np.arange(len(all_oos)) // 7).sum()
    stitched_daily = stitched_daily[stitched_daily != 0.0]

    if len(stitched_daily) >= 20:
        st_sharpe = float(stitched_daily.mean() / stitched_daily.std() * sqrt(252))
        st_eq = (1 + stitched_daily).cumprod()
        st_dd = float(((st_eq - st_eq.cummax()) / st_eq.cummax()).min())
        st_ret = float(stitched_daily.mean() * 252)
    else:
        st_sharpe = 0.0
        st_dd = 0.0
        st_ret = 0.0

    fold_df = pd.DataFrame(fold_results)

    return {
        "fold_df": fold_df,
        "stitched_sharpe": round(st_sharpe, 3),
        "stitched_dd_pct": round(st_dd * 100, 2),
        "stitched_ret_pct": round(st_ret * 100, 2),
        "stitched_days": len(stitched_daily),
        "n_folds": len(fold_results),
    }


# ── Display ──────────────────────────────────────────────────────────────────


def print_wfo_results(pair: str, result: dict, threshold: float) -> None:
    fold_df = result["fold_df"]

    print(f"\n{'=' * 90}")
    print(f"  MULTI-SCALE IC WFO -- {pair} H1 | threshold={threshold}")
    print(f"{'=' * 90}")

    print(
        f"\n{'Fold':>4} | {'OOS Period':>23} | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
        f" | {'Ret%':>6} {'DD%':>6} {'Trd':>4} | Top Signals"
    )
    print("-" * 90)
    for _, r in fold_df.iterrows():
        flag = "+" if r["oos_sharpe"] > 0 else " "
        print(
            f"{flag}{int(r['fold']):>3} | {r['oos_start']} - {r['oos_end']} | "
            f"{r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
            f" | {r['oos_ret_pct']:>+5.1f}% {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4}"
            f" | {r['top_signals']}"
        )

    print(f"\n{'=' * 60}")
    print("  STITCHED OOS RESULTS")
    print(f"{'=' * 60}")
    print(f"  Sharpe:    {result['stitched_sharpe']:+.3f}")
    print(f"  CAGR:      {result['stitched_ret_pct']:+.2f}%")
    print(f"  Max DD:    {result['stitched_dd_pct']:+.2f}%")
    print(f"  OOS Days:  {result['stitched_days']}")
    print(f"  Folds:     {result['n_folds']}")

    # Quality gates
    n_folds = result["n_folds"]
    positive = (fold_df["oos_sharpe"] > 0).sum()
    pct_pos = positive / n_folds if n_folds > 0 else 0

    print("\n  QUALITY GATES:")
    for name, val, gate, op in [
        ("Positive folds", pct_pos, MIN_PCT_FOLDS_POSITIVE, ">="),
        ("Stitched Sharpe", result["stitched_sharpe"], MIN_STITCHED_SHARPE, ">="),
    ]:
        ok = val >= gate if op == ">=" else val <= gate
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.3f} (gate {op} {gate})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Scale IC Walk-Forward Validation")
    parser.add_argument("--pair", default="GLD", help="Instrument")
    parser.add_argument("--top-n", type=int, default=5, help="Top N signals per fold")
    parser.add_argument("--is-bars", type=int, default=10_000, help="IS window (H1 bars)")
    parser.add_argument("--oos-bars", type=int, default=2_500, help="OOS window (H1 bars)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Z-score entry threshold")
    parser.add_argument("--exit-buffer", type=float, default=0.10, help="Exit buffer")
    parser.add_argument("--spread-bps", type=float, default=2.0, help="Spread bps")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage bps")
    parser.add_argument("--scales", default="H1,H4,D,W", help="Comma-separated scales")
    parser.add_argument("--sweep-threshold", action="store_true", help="Sweep thresholds")
    args = parser.parse_args()

    scales = {k: SCALE_MAP[k] for k in args.scales.split(",") if k in SCALE_MAP}

    print("=" * 60)
    print(f"  MULTI-SCALE IC WFO -- {args.pair} H1")
    print(f"  Top-N: {args.top_n} | IS: {args.is_bars} | OOS: {args.oos_bars}")
    print(f"  Scales: {scales}")
    print(f"  Costs: {args.spread_bps} + {args.slippage_bps} bps")
    print("=" * 60)

    if args.sweep_threshold:
        thresholds = [0.25, 0.50, 0.75, 1.0, 1.5]
        print(f"\n  Sweeping thresholds: {thresholds}")
        print(
            f"\n  {'Thr':>5} | {'Stitch Sh':>9} {'Stitch Ret':>10} {'Stitch DD':>9}"
            f" | {'Folds':>5} {'Pos%':>5}"
        )
        print("  " + "-" * 55)
        for th in thresholds:
            r = run_wfo(
                args.pair,
                top_n=args.top_n,
                is_bars=args.is_bars,
                oos_bars=args.oos_bars,
                threshold=th,
                exit_buffer=args.exit_buffer,
                spread_bps=args.spread_bps,
                slippage_bps=args.slippage_bps,
                scales=scales,
            )
            pct_pos = (r["fold_df"]["oos_sharpe"] > 0).mean()
            flag = (
                "+"
                if r["stitched_sharpe"] > MIN_STITCHED_SHARPE and pct_pos >= MIN_PCT_FOLDS_POSITIVE
                else " "
            )
            print(
                f" {flag}{th:>4.2f} | {r['stitched_sharpe']:>+9.3f} {r['stitched_ret_pct']:>+9.2f}%"
                f" {r['stitched_dd_pct']:>+8.2f}% | {r['n_folds']:>5} {pct_pos:>4.0%}"
            )
    else:
        result = run_wfo(
            args.pair,
            top_n=args.top_n,
            is_bars=args.is_bars,
            oos_bars=args.oos_bars,
            threshold=args.threshold,
            exit_buffer=args.exit_buffer,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            scales=scales,
        )
        print_wfo_results(args.pair, result, args.threshold)

        # Save
        save_path = REPORTS_DIR / f"multiscale_wfo_{args.pair.lower()}.csv"
        result["fold_df"].to_csv(save_path, index=False)
        print(f"\n  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
