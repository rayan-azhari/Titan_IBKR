"""run_bond_equity_wfo.py -- WFO for Bond->Equity/Gold Momentum.

Rolling WFO validation for the bond momentum signal.
Per fold: z-score calibrated on IS, applied to OOS.

Usage:
    uv run python research/cross_asset/run_bond_equity_wfo.py
    uv run python research/cross_asset/run_bond_equity_wfo.py --bond TLT --target GLD --lookback 20
"""

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_daily(sym: str) -> pd.Series:
    for prefix in ["", "^"]:
        path = DATA_DIR / f"{prefix}{sym}_D.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            s = df["close"].astype(float).sort_index()
            s.index = s.index.normalize()
            return s
    raise FileNotFoundError(f"No data for {sym}")


def run_bond_wfo(
    bond_close: pd.Series,
    target_close: pd.Series,
    lookback: int = 20,
    hold_days: int = 20,
    threshold: float = 0.50,
    is_days: int = 504,
    oos_days: int = 126,
    spread_bps: float = 5.0,
) -> dict:
    """Rolling WFO for bond momentum signal."""
    # Bond momentum
    bond_mom = np.log(bond_close / bond_close.shift(lookback)).dropna()

    # Align
    common = bond_mom.index.intersection(target_close.index)
    bond_mom = bond_mom.reindex(common)
    target = target_close.reindex(common)
    n = len(common)

    if n < is_days + oos_days:
        return {"error": f"Insufficient data: {n} bars"}

    mom_vals = bond_mom.values
    target_vals = target.values
    target_ret = np.empty(n)
    target_ret[0] = 0.0
    target_ret[1:] = (target_vals[1:] - target_vals[:-1]) / target_vals[:-1]

    folds = []
    stitched = []
    fold_idx = 0
    oos_start = is_days

    while oos_start + oos_days <= n:
        is_start = oos_start - is_days

        # Z-score on IS
        is_mom = mom_vals[is_start:oos_start]
        is_mean = np.nanmean(is_mom)
        is_std = np.nanstd(is_mom)
        if is_std < 1e-8:
            is_std = 1.0

        # IS backtest
        is_z = (is_mom - is_mean) / is_std
        is_pos = _position_with_hold(is_z, threshold, hold_days)
        is_ret = target_ret[is_start:oos_start]
        is_strat = is_ret * is_pos
        is_sh = _daily_sharpe(is_strat)

        # OOS backtest
        oos_end = oos_start + oos_days
        oos_mom = mom_vals[oos_start:oos_end]
        oos_z = (oos_mom - is_mean) / is_std  # frozen IS stats
        oos_pos = _position_with_hold(oos_z, threshold, hold_days)
        oos_ret = target_ret[oos_start:oos_end]
        oos_trans = np.zeros(oos_days)
        oos_trans[0] = abs(oos_pos[0])
        oos_trans[1:] = np.abs(oos_pos[1:] - oos_pos[:-1])
        oos_strat = oos_ret * oos_pos - oos_trans * spread_bps / 10_000

        oos_sh = _daily_sharpe(oos_strat)
        parity = oos_sh / is_sh if abs(is_sh) > 0.01 else 0.0
        oos_trades = int(np.sum(oos_trans > 0))

        # DD
        eq = np.cumprod(1 + oos_strat)
        hwm = np.maximum.accumulate(eq)
        dd = ((eq - hwm) / hwm).min() if len(eq) > 0 else 0.0

        folds.append(
            {
                "fold": fold_idx,
                "oos_start": common[oos_start].strftime("%Y-%m-%d"),
                "oos_end": common[min(oos_end - 1, n - 1)].strftime("%Y-%m-%d"),
                "is_sharpe": round(is_sh, 3),
                "oos_sharpe": round(oos_sh, 3),
                "parity": round(parity, 3),
                "oos_trades": oos_trades,
                "oos_dd_pct": round(dd * 100, 2),
            }
        )
        stitched.append(oos_strat)

        fold_idx += 1
        oos_start += oos_days

    # Stitch
    all_oos = np.concatenate(stitched) if stitched else np.array([])
    all_oos_nz = all_oos[all_oos != 0.0]
    if len(all_oos_nz) >= 20:
        st_sh = float(np.mean(all_oos_nz) / np.std(all_oos_nz) * sqrt(252))
        eq = np.cumprod(1 + all_oos_nz)
        hwm = np.maximum.accumulate(eq)
        st_dd = float(((eq - hwm) / hwm).min())
        st_ret = float(np.mean(all_oos_nz) * 252)
    else:
        st_sh = 0.0
        st_dd = 0.0
        st_ret = 0.0

    fold_df = pd.DataFrame(folds)

    return {
        "fold_df": fold_df,
        "stitched_sharpe": round(st_sh, 3),
        "stitched_dd_pct": round(st_dd * 100, 2),
        "stitched_ret_pct": round(st_ret * 100, 2),
        "n_folds": len(folds),
        "pct_positive": round((fold_df["oos_sharpe"] > 0).mean(), 3) if len(fold_df) > 0 else 0,
        "total_trades": int(fold_df["oos_trades"].sum()) if len(fold_df) > 0 else 0,
    }


def _position_with_hold(z: np.ndarray, threshold: float, hold_days: int) -> np.ndarray:
    """Position: long when z > threshold, hold for minimum hold_days."""
    n = len(z)
    pos = np.zeros(n)
    bars_held = 0
    for i in range(n):
        if pos[max(0, i - 1)] != 0:
            bars_held += 1
        if bars_held >= hold_days and z[i] <= threshold:
            pos[i] = 0.0
            bars_held = 0
        elif pos[max(0, i - 1)] == 0 and z[i] > threshold:
            pos[i] = 1.0
            bars_held = 0
        else:
            pos[i] = pos[max(0, i - 1)]
    return pos


def _daily_sharpe(rets: np.ndarray) -> float:
    nz = rets[rets != 0.0]
    if len(nz) < 20:
        return 0.0
    return float(np.mean(nz) / np.std(nz) * sqrt(252)) if np.std(nz) > 1e-9 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Bond->Equity/Gold Momentum WFO")
    parser.add_argument("--bond", default=None, help="Single bond to test")
    parser.add_argument("--target", default=None, help="Single target to test")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--is-days", type=int, default=504)
    parser.add_argument("--oos-days", type=int, default=126)
    args = parser.parse_args()

    print("=" * 70)
    print("  BOND->EQUITY/GOLD MOMENTUM WFO")
    print(f"  IS: {args.is_days}d | OOS: {args.oos_days}d")
    print("=" * 70)

    # Top combos from the sweep to validate
    if args.bond and args.target and args.lookback:
        configs = [(args.bond, args.target, args.lookback, 20, 0.50)]
    else:
        configs = [
            ("TLT", "QQQ", 10, 10, 0.50),
            ("TLT", "GLD", 10, 40, 0.25),
            ("TLT", "GLD", 20, 20, 0.50),
            ("IEF", "QQQ", 10, 10, 0.50),
            ("IEF", "GLD", 10, 10, 0.50),
            ("TLT", "SPY", 10, 10, 0.50),
            ("TLT", "GLD", 10, 10, 0.00),
            ("IEF", "GLD", 60, 20, 0.00),
        ]

    print(f"\n  Testing {len(configs)} configurations...")
    print(
        f"\n  {'Bond':>5} {'Tgt':>4} {'LB':>3} {'Hld':>3} {'Thr':>5}"
        f" | {'St.Sh':>6} {'Ret%':>6} {'DD%':>6}"
        f" | {'Folds':>5} {'Pos%':>5} {'Trd':>5}"
    )
    print("  " + "-" * 65)

    all_results = []
    for bond_sym, target_sym, lookback, hold, threshold in configs:
        bond = load_daily(bond_sym)
        target = load_daily(target_sym)

        r = run_bond_wfo(
            bond,
            target,
            lookback=lookback,
            hold_days=hold,
            threshold=threshold,
            is_days=args.is_days,
            oos_days=args.oos_days,
        )
        if "error" in r:
            print(
                f"  {bond_sym:>5} {target_sym:>4} {lookback:>3} {hold:>3} {threshold:>5.2f} | SKIP: {r['error']}"
            )
            continue

        flag = "+" if r["stitched_sharpe"] > 0.5 and r["pct_positive"] >= 0.5 else " "
        print(
            f" {flag}{bond_sym:>4} {target_sym:>4} {lookback:>3} {hold:>3} {threshold:>5.2f}"
            f" | {r['stitched_sharpe']:>+6.3f} {r['stitched_ret_pct']:>+5.1f}% {r['stitched_dd_pct']:>+5.1f}%"
            f" | {r['n_folds']:>5} {r['pct_positive']:>4.0%} {r['total_trades']:>5}"
        )
        all_results.append(
            {
                "bond": bond_sym,
                "target": target_sym,
                "lookback": lookback,
                "hold": hold,
                "threshold": threshold,
                **r,
            }
        )

        # Print fold details for passing combos
        if r["stitched_sharpe"] > 0.3:
            for _, f in r["fold_df"].iterrows():
                pflag = "+" if f["oos_sharpe"] > 0 else " "
                print(
                    f"   {pflag} fold {int(f['fold'])}: {f['oos_start']}-{f['oos_end']}"
                    f" IS={f['is_sharpe']:+.2f} OOS={f['oos_sharpe']:+.2f}"
                    f" Trd={int(f['oos_trades'])} DD={f['oos_dd_pct']:.1f}%"
                )

    if all_results:
        best = max(all_results, key=lambda x: x["stitched_sharpe"])
        print(
            f"\n  BEST: {best['bond']}->{best['target']} LB={best['lookback']}"
            f" Sharpe={best['stitched_sharpe']:+.3f} Pos={best['pct_positive']:.0%}"
        )
        ok = best["stitched_sharpe"] > 0.5 and best["pct_positive"] >= 0.5
        print(f"  Gate: {'PASS' if ok else 'FAIL'}")

    save_path = REPORTS_DIR / "cross_asset_bond_equity_wfo.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "fold_df"} for r in all_results]).to_csv(
        save_path, index=False
    )
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
