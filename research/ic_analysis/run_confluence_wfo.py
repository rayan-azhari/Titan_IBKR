"""run_confluence_wfo.py -- WFO for AND-Gated Multi-Scale Confluence.

Takes a specific signal family, computes it at 4 scales on H1, applies
AND-gate confluence, and validates via rolling WFO.

Per fold: z-score calibrated on IS only, applied to OOS. Signal family is
fixed (no re-ranking per fold -- the signal was already selected by the
static confluence sweep).

Usage:
    uv run python research/ic_analysis/run_confluence_wfo.py --pair GLD --signal donchian_pos_10
    uv run python research/ic_analysis/run_confluence_wfo.py --pair GLD --signal ema_slope_10 --threshold 1.5
    uv run python research/ic_analysis/run_confluence_wfo.py --pair GLD --sweep-signals
"""

import argparse
import sys
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
    build_all_signals,
)

WINDOW_1Y_H1 = 252 * 22
TF_WEIGHTS = {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30}

# Top GLD confluence signals from the sweep
TOP_GLD_SIGNALS = [
    "donchian_pos_10",
    "ema_slope_10",
    "rsi_14_dev",
    "trend_mom",
    "rsi_7_dev",
    "stoch_d_dev",
]


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


def _build_and_gate_confluence(
    scale_signals: dict[str, pd.DataFrame],
    sig_name: str,
) -> pd.Series:
    """Build AND-gated confluence for a single signal family across scales."""
    per_scale = {}
    for label in SCALE_MAP:
        prefix = f"{label}_" if SCALE_MAP[label] > 1 else ""
        col = f"{prefix}{sig_name}"
        if col not in scale_signals[label].columns:
            return pd.Series(dtype=float)
        per_scale[label] = scale_signals[label][col]

    df = pd.concat(per_scale, axis=1).dropna()
    if len(df) < 100:
        return pd.Series(dtype=float)

    # Weighted sum
    weighted = pd.Series(0.0, index=df.index)
    for label in SCALE_MAP:
        weighted += df[label] * TF_WEIGHTS.get(label, 0.25)

    # AND gate
    signs = np.sign(df.values)
    agreement = (signs > 0).all(axis=1) | (signs < 0).all(axis=1)
    return weighted.where(agreement, 0.0)


def run_confluence_wfo(
    pair: str,
    sig_name: str,
    is_bars: int = 5000,
    oos_bars: int = 1250,
    threshold: float = 1.0,
    exit_buffer: float = 0.10,
    spread_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> dict:
    """Rolling WFO for a single AND-gated confluence signal."""
    df = _load_ohlcv(pair, "H1")
    close = df["close"]

    # Compute all scale signals once
    scale_signals = {}
    for label, mult in SCALE_MAP.items():
        prefix = f"{label}_" if mult > 1 else ""
        scale_signals[label] = build_all_signals(
            df, WINDOW_1Y_H1, period_scale=mult, name_prefix=prefix
        )

    # Build AND-gated confluence
    confluence = _build_and_gate_confluence(scale_signals, sig_name)
    if len(confluence) < is_bars + oos_bars:
        return {"error": f"Insufficient bars after warmup: {len(confluence)}"}

    # Align close to confluence index
    close_aligned = close.reindex(confluence.index)
    close_arr = close_aligned.values.astype(np.float64)
    conf_vals = confluence.values
    n_conf = len(confluence)

    # Rolling WFO
    folds = []
    stitched = []
    fold_idx = 0
    oos_start = is_bars

    while oos_start + oos_bars <= n_conf:
        is_start = oos_start - is_bars
        is_end = oos_start
        oos_end = oos_start + oos_bars

        # Z-score on IS
        is_vals = conf_vals[is_start:is_end]
        is_mean = np.nanmean(is_vals)
        is_std = np.nanstd(is_vals)
        if is_std < 1e-8:
            is_std = 1.0

        # IS backtest
        is_z = (is_vals - is_mean) / is_std
        is_sig = np.empty(is_bars)
        is_sig[0] = 0.0
        is_sig[1:] = is_z[:-1]
        is_pos = _vectorized_position(is_sig, threshold, exit_buffer)
        is_br = np.empty(is_bars)
        is_br[0] = 0.0
        c_is = close_arr[is_start:is_end]
        is_br[1:] = (c_is[1:] - c_is[:-1]) / c_is[:-1]
        is_trans = np.empty(is_bars)
        is_trans[0] = abs(is_pos[0])
        is_trans[1:] = np.abs(is_pos[1:] - is_pos[:-1])
        is_rets = is_br * is_pos - is_trans * (spread_bps + slippage_bps) / 10_000

        # OOS backtest
        oos_z = (conf_vals[oos_start:oos_end] - is_mean) / is_std
        oos_sig = np.empty(oos_bars)
        oos_sig[0] = 0.0
        oos_sig[1:] = oos_z[:-1]
        oos_pos = _vectorized_position(oos_sig, threshold, exit_buffer)
        oos_br = np.empty(oos_bars)
        oos_br[0] = 0.0
        c_oos = close_arr[oos_start:oos_end]
        oos_br[1:] = (c_oos[1:] - c_oos[:-1]) / c_oos[:-1]
        oos_trans = np.empty(oos_bars)
        oos_trans[0] = abs(oos_pos[0])
        oos_trans[1:] = np.abs(oos_pos[1:] - oos_pos[:-1])
        oos_rets = oos_br * oos_pos - oos_trans * (spread_bps + slippage_bps) / 10_000

        # Daily aggregate (~7 H1 bars/day for equities)
        def _sharpe(rets):
            daily = pd.Series(rets).groupby(np.arange(len(rets)) // 7).sum()
            daily = daily[daily != 0.0]
            if len(daily) < 10:
                return 0.0
            return float(daily.mean() / daily.std() * sqrt(252)) if daily.std() > 1e-9 else 0.0

        is_sh = _sharpe(is_rets)
        oos_sh = _sharpe(oos_rets)
        parity = oos_sh / is_sh if abs(is_sh) > 0.01 else 0.0
        oos_trades = int(np.sum(np.abs(np.diff(oos_pos)) > 0))

        oos_daily = pd.Series(oos_rets).groupby(np.arange(len(oos_rets)) // 7).sum()
        oos_daily = oos_daily[oos_daily != 0.0]
        if len(oos_daily) > 5:
            eq = (1 + oos_daily).cumprod()
            dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        else:
            dd = 0.0

        folds.append(
            {
                "fold": fold_idx,
                "oos_start": confluence.index[oos_start].strftime("%Y-%m-%d"),
                "oos_end": confluence.index[min(oos_end - 1, n_conf - 1)].strftime("%Y-%m-%d"),
                "is_sharpe": round(is_sh, 3),
                "oos_sharpe": round(oos_sh, 3),
                "parity": round(parity, 3),
                "oos_dd_pct": round(dd * 100, 2),
                "oos_trades": oos_trades,
            }
        )
        stitched.append(oos_rets)

        fold_idx += 1
        oos_start += oos_bars

    # Stitch
    all_oos = np.concatenate(stitched) if stitched else np.array([])
    st_daily = pd.Series(all_oos).groupby(np.arange(len(all_oos)) // 7).sum()
    st_daily = st_daily[st_daily != 0.0]

    if len(st_daily) >= 20:
        st_sh = (
            float(st_daily.mean() / st_daily.std() * sqrt(252)) if st_daily.std() > 1e-9 else 0.0
        )
        st_eq = (1 + st_daily).cumprod()
        st_dd = float(((st_eq - st_eq.cummax()) / st_eq.cummax()).min())
        st_ret = float(st_daily.mean() * 252)
    else:
        st_sh = 0.0
        st_dd = 0.0
        st_ret = 0.0

    fold_df = pd.DataFrame(folds)

    return {
        "signal": sig_name,
        "threshold": threshold,
        "fold_df": fold_df,
        "stitched_sharpe": round(st_sh, 3),
        "stitched_dd_pct": round(st_dd * 100, 2),
        "stitched_ret_pct": round(st_ret * 100, 2),
        "n_folds": len(folds),
        "pct_positive": round((fold_df["oos_sharpe"] > 0).mean(), 3) if len(fold_df) > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Confluence WFO Validation")
    parser.add_argument("--pair", default="GLD")
    parser.add_argument("--signal", default="donchian_pos_10")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--is-bars", type=int, default=5000)
    parser.add_argument("--oos-bars", type=int, default=1250)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sweep-signals", action="store_true", help="Test all top signals")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  CONFLUENCE WFO -- {args.pair} H1")
    print(f"  Costs: {args.spread_bps}+{args.slippage_bps} bps")
    print("=" * 70)

    if args.sweep_signals:
        signals = TOP_GLD_SIGNALS
        thresholds = [0.75, 1.0, 1.5]
        print(f"\n  Sweeping {len(signals)} signals × {len(thresholds)} thresholds")
        print(
            f"\n  {'Signal':<22} {'Thr':>5} | {'St.Sh':>6} {'Ret%':>6} {'DD%':>6}"
            f" | {'Folds':>5} {'Pos%':>5}"
        )
        print("  " + "-" * 60)

        all_results = []
        for sig in signals:
            for th in thresholds:
                r = run_confluence_wfo(
                    args.pair,
                    sig,
                    is_bars=args.is_bars,
                    oos_bars=args.oos_bars,
                    threshold=th,
                    spread_bps=args.spread_bps,
                    slippage_bps=args.slippage_bps,
                )
                if "error" in r:
                    print(f"  {sig:<22} {th:>5.2f} | SKIP: {r['error']}")
                    continue
                flag = "+" if r["stitched_sharpe"] > 0.5 and r["pct_positive"] >= 0.5 else " "
                print(
                    f" {flag}{sig:<21} {th:>5.2f} | {r['stitched_sharpe']:>+6.3f}"
                    f" {r['stitched_ret_pct']:>+5.1f}% {r['stitched_dd_pct']:>+5.1f}%"
                    f" | {r['n_folds']:>5} {r['pct_positive']:>4.0%}"
                )
                all_results.append(r)

        # Best result
        if all_results:
            best = max(all_results, key=lambda x: x["stitched_sharpe"])
            print(
                f"\n  BEST: {best['signal']} th={best['threshold']}"
                f" Sharpe={best['stitched_sharpe']:+.3f}"
                f" DD={best['stitched_dd_pct']:.1f}%"
                f" Pos={best['pct_positive']:.0%}"
            )
            ok = best["stitched_sharpe"] > 0.5 and best["pct_positive"] >= 0.5
            print(f"  Gate: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"\n  Signal: {args.signal} | Threshold: {args.threshold}")
        r = run_confluence_wfo(
            args.pair,
            args.signal,
            is_bars=args.is_bars,
            oos_bars=args.oos_bars,
            threshold=args.threshold,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        )
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            return

        fold_df = r["fold_df"]
        print(
            f"\n{'Fold':>4} | {'OOS Period':>23} | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
            f" | {'DD%':>6} {'Trd':>4}"
        )
        print("-" * 70)
        for _, f in fold_df.iterrows():
            flag = "+" if f["oos_sharpe"] > 0 else " "
            print(
                f"{flag}{int(f['fold']):>3} | {f['oos_start']} - {f['oos_end']}"
                f" | {f['is_sharpe']:>+6.3f} {f['oos_sharpe']:>+7.3f} {f['parity']:>6.3f}"
                f" | {f['oos_dd_pct']:>+5.1f}% {f['oos_trades']:>4}"
            )

        print(f"\n  Stitched Sharpe: {r['stitched_sharpe']:+.3f}")
        print(f"  Stitched Ret:    {r['stitched_ret_pct']:+.2f}%")
        print(f"  Stitched DD:     {r['stitched_dd_pct']:+.2f}%")
        print(f"  Folds: {r['n_folds']} | Positive: {r['pct_positive']:.0%}")

        ok = r["stitched_sharpe"] > 0.5 and r["pct_positive"] >= 0.5
        print(f"  Gate: {'PASS' if ok else 'FAIL'}")

        save_path = REPORTS_DIR / f"confluence_wfo_{args.pair.lower()}_{args.signal}.csv"
        fold_df.to_csv(save_path, index=False)
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
