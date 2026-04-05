"""run_confluence_regime_wfo.py -- WFO for MR FX + Confluence Regime Filter.

Validates the confluence disagreement regime filter via rolling WFO.
Per fold: z-score/percentile calibrated on IS, applied to OOS.

Usage:
    uv run python research/mean_reversion/run_confluence_regime_wfo.py
    uv run python research/mean_reversion/run_confluence_regime_wfo.py --signal donchian_pos_20
"""

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.mean_reversion.run_confluence_regime_test import (  # noqa: E402
    build_atr_regime_mask,
    build_confluence_disagreement_mask,
    compute_percentile_levels,
    compute_vwap_deviation,
    load_h1,
)

TIER_GRIDS = {
    "aggressive": [0.80, 0.85, 0.90, 0.95],
    "standard": [0.90, 0.95, 0.98, 0.99],
    "conservative": [0.95, 0.98, 0.99, 0.999],
}
TIER_SIZES = [1, 2, 4, 8]


def run_mr_wfo(
    close: pd.Series,
    deviation: pd.Series,
    regime_mask: pd.Series,
    tiers_pct: list[float],
    is_bars: int = 30_000,
    oos_bars: int = 7_500,
    spread_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> dict:
    """Rolling WFO for MR strategy with a given regime filter.

    Per fold: percentile levels calibrated on IS, applied to OOS.
    """
    n = len(close)
    folds = []
    stitched_pnl = []
    fold_idx = 0
    oos_start = is_bars

    while oos_start + oos_bars <= n:
        is_start = oos_start - is_bars
        is_end = oos_start
        oos_end = oos_start + oos_bars

        # IS slice -- compute percentile levels
        is_close = close.iloc[is_start:is_end]
        is_dev = deviation.iloc[is_start:is_end]
        is_levels = compute_percentile_levels(is_dev, window=500, pcts=tiers_pct)

        # Get the last valid IS levels to use as fixed thresholds for OOS
        last_levels = is_levels.dropna().iloc[-1] if len(is_levels.dropna()) > 0 else None
        if last_levels is None:
            oos_start += oos_bars
            fold_idx += 1
            continue

        # OOS slice -- use frozen IS levels as fixed thresholds
        oos_close = close.iloc[oos_start:oos_end]
        oos_dev = deviation.iloc[oos_start:oos_end]
        oos_mask = regime_mask.iloc[oos_start:oos_end]

        # Build fixed levels for OOS (constant from IS calibration)
        oos_levels = pd.DataFrame(
            {col: last_levels[col] for col in is_levels.columns},
            index=oos_close.index,
        )

        # Run backtest on OOS with frozen levels
        oos_result = _fold_backtest(
            oos_close,
            oos_dev,
            oos_levels,
            oos_mask,
            TIER_SIZES,
            spread_bps,
            slippage_bps,
        )

        # Also run IS for parity
        is_mask = regime_mask.iloc[is_start:is_end]
        is_result = _fold_backtest(
            is_close,
            is_dev,
            is_levels,
            is_mask,
            TIER_SIZES,
            spread_bps,
            slippage_bps,
        )

        parity = (
            oos_result["sharpe"] / is_result["sharpe"] if abs(is_result["sharpe"]) > 0.01 else 0.0
        )

        folds.append(
            {
                "fold": fold_idx,
                "oos_start": oos_close.index[0].strftime("%Y-%m-%d"),
                "oos_end": oos_close.index[-1].strftime("%Y-%m-%d"),
                "is_sharpe": is_result["sharpe"],
                "oos_sharpe": oos_result["sharpe"],
                "parity": round(parity, 3),
                "oos_trades": oos_result["n_trades"],
                "oos_wr": oos_result["win_rate"],
                "oos_avg_hold": oos_result["avg_hold"],
                "oos_dd_pct": oos_result["dd_pct"],
            }
        )

        if oos_result["daily_pnl"] is not None:
            stitched_pnl.append(oos_result["daily_pnl"])

        fold_idx += 1
        oos_start += oos_bars

    # Stitch
    if stitched_pnl:
        all_oos = pd.concat(stitched_pnl)
        all_oos = all_oos[all_oos != 0.0]
        if len(all_oos) >= 20:
            st_sh = (
                float(all_oos.mean() / all_oos.std() * sqrt(252)) if all_oos.std() > 1e-9 else 0.0
            )
            st_eq = (1 + all_oos).cumprod()
            st_dd = float(((st_eq - st_eq.cummax()) / st_eq.cummax()).min())
        else:
            st_sh = 0.0
            st_dd = 0.0
    else:
        st_sh = 0.0
        st_dd = 0.0

    fold_df = pd.DataFrame(folds)
    pct_positive = (fold_df["oos_sharpe"] > 0).mean() if len(fold_df) > 0 else 0
    total_trades = fold_df["oos_trades"].sum() if len(fold_df) > 0 else 0

    return {
        "fold_df": fold_df,
        "stitched_sharpe": round(st_sh, 3),
        "stitched_dd_pct": round(st_dd * 100, 2),
        "n_folds": len(folds),
        "pct_positive": round(pct_positive, 3),
        "total_trades": int(total_trades),
    }


def _fold_backtest(
    close: pd.Series,
    deviation: pd.Series,
    levels: pd.DataFrame,
    regime_mask: pd.Series,
    tier_sizes: list[float],
    spread_bps: float,
    slippage_bps: float,
) -> dict:
    """Run MR backtest on a single fold slice."""
    n = len(close)
    close_arr = close.values
    dev_arr = deviation.values
    gate_arr = regime_mask.reindex(close.index, fill_value=False).values
    hour_arr = close.index.hour

    # Session filter: 07:00-12:00 UTC
    session_arr = (hour_arr >= 7) & (hour_arr < 12)
    combined_gate = gate_arr & session_arr

    position = np.zeros(n)
    entry_price = 0.0
    entry_bar = 0
    tiers_hit = set()
    trade_returns = []
    trade_durations = []
    bar_pnl = np.zeros(n)
    cost_per_unit = (spread_bps + slippage_bps) / 10_000

    for i in range(1, n):
        px = close_arr[i]
        dev = abs(dev_arr[i]) if not np.isnan(dev_arr[i]) else 0.0
        prev_pos = position[i - 1]

        # Exit checks
        if prev_pos != 0:
            if hour_arr[i] >= 21:
                ret = (px - entry_price) / entry_price * np.sign(prev_pos) - cost_per_unit
                trade_returns.append(ret)
                trade_durations.append(i - entry_bar)
                bar_pnl[i] = ret * abs(prev_pos)
                position[i] = 0
                entry_price = 0.0
                tiers_hit = set()
                continue

            lvl0 = levels.iloc[i, 0] if not np.isnan(levels.iloc[i, 0]) else 999
            if dev < lvl0 * 0.5:
                ret = (px - entry_price) / entry_price * np.sign(prev_pos) - cost_per_unit
                trade_returns.append(ret)
                trade_durations.append(i - entry_bar)
                bar_pnl[i] = ret * abs(prev_pos)
                position[i] = 0
                entry_price = 0.0
                tiers_hit = set()
                continue

        # Entry checks
        if combined_gate[i]:
            for tier_idx in range(len(levels.columns)):
                if tier_idx in tiers_hit:
                    continue
                lvl = levels.iloc[i, tier_idx]
                if np.isnan(lvl):
                    continue
                if dev > lvl:
                    size = tier_sizes[tier_idx] if tier_idx < len(tier_sizes) else 1
                    direction = -1.0 if dev_arr[i] > 0 else 1.0
                    if position[i] == 0:
                        entry_price = px
                        entry_bar = i
                    else:
                        old_size = abs(position[i])
                        entry_price = (entry_price * old_size + px * size) / (old_size + size)
                    position[i] += direction * size
                    tiers_hit.add(tier_idx)
                    bar_pnl[i] -= cost_per_unit * size

        if position[i] == 0 and prev_pos != 0 and bar_pnl[i] == 0:
            position[i] = prev_pos
            bar_pnl[i] = (px - close_arr[i - 1]) / close_arr[i - 1] * prev_pos
        elif position[i] == 0 and prev_pos == 0:
            position[i] = 0

    daily_pnl = pd.Series(bar_pnl, index=close.index).resample("D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0.0]

    def _sharpe(d):
        if len(d) < 10:
            return 0.0
        return float(d.mean() / d.std() * sqrt(252)) if d.std() > 1e-9 else 0.0

    def _dd(d):
        if len(d) < 5:
            return 0.0
        eq = (1 + d).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())

    n_trades = len(trade_returns)
    wr = sum(1 for r in trade_returns if r > 0) / n_trades * 100 if n_trades > 0 else 0
    avg_hold = np.mean(trade_durations) if trade_durations else 0

    return {
        "sharpe": round(_sharpe(daily_pnl), 3),
        "n_trades": n_trades,
        "win_rate": round(wr, 1),
        "avg_hold": round(avg_hold, 1),
        "dd_pct": round(_dd(daily_pnl) * 100, 2),
        "daily_pnl": daily_pnl,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MR FX Confluence Regime WFO")
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--signal", default=None, help="Single signal to test")
    parser.add_argument("--is-bars", type=int, default=30000, help="IS window (H1 bars)")
    parser.add_argument("--oos-bars", type=int, default=7500, help="OOS window (H1 bars)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  MR FX CONFLUENCE REGIME WFO -- {args.pair} H1")
    print(f"  IS: {args.is_bars} bars | OOS: {args.oos_bars} bars")
    print("=" * 70)

    df = load_h1(args.pair)
    close = df["close"]
    print(f"\n  Loaded {len(close)} bars ({close.index[0].date()} -> {close.index[-1].date()})")

    deviation = compute_vwap_deviation(close, anchor_period=24)

    # Filters to test
    if args.signal:
        signals_to_test = [args.signal]
    else:
        signals_to_test = ["donchian_pos_20", "rsi_14_dev"]

    # Build all masks
    print("\n  Building regime masks...")
    filters = {}
    for sig_name in signals_to_test:
        print(f"    {sig_name}...")
        mask = build_confluence_disagreement_mask(df, sig_name)
        filters[f"conf_{sig_name}"] = mask

    # Also test ATR and no-filter baselines
    filters["atr_only"] = build_atr_regime_mask(df, threshold_pct=0.30)
    filters["no_filter"] = pd.Series(True, index=df.index)

    # Sweep: filters × tier grids
    tier_grids_to_test = ["standard", "conservative"]

    print(f"\n  Running WFO ({len(filters)} filters × {len(tier_grids_to_test)} tier grids)...")
    print(
        f"\n  {'Filter':<30} {'Tiers':>12} | {'St.Sh':>6} {'DD%':>6}"
        f" | {'Folds':>5} {'Pos%':>5} {'Trd':>5}"
    )
    print("  " + "-" * 75)

    all_results = []
    for filter_name, mask in filters.items():
        for grid_name in tier_grids_to_test:
            tiers_pct = TIER_GRIDS[grid_name]
            r = run_mr_wfo(
                close,
                deviation,
                mask,
                tiers_pct,
                is_bars=args.is_bars,
                oos_bars=args.oos_bars,
            )
            tier_str = "/".join(str(int(p * 100)) for p in tiers_pct)
            flag = "+" if r["stitched_sharpe"] > 0.5 and r["pct_positive"] >= 0.5 else " "
            print(
                f" {flag}{filter_name:<29} {tier_str:>12}"
                f" | {r['stitched_sharpe']:>+6.3f} {r['stitched_dd_pct']:>+5.1f}%"
                f" | {r['n_folds']:>5} {r['pct_positive']:>4.0%} {r['total_trades']:>5}"
            )
            all_results.append(
                {
                    "filter": filter_name,
                    "tiers": grid_name,
                    "tiers_pct": tier_str,
                    **r,
                }
            )

            # Print fold details for passing combos
            if r["stitched_sharpe"] > 0.3 and len(r["fold_df"]) > 0:
                for _, f in r["fold_df"].iterrows():
                    pflag = "+" if f["oos_sharpe"] > 0 else " "
                    print(
                        f"   {pflag} fold {int(f['fold'])}: {f['oos_start']}-{f['oos_end']}"
                        f" IS={f['is_sharpe']:+.2f} OOS={f['oos_sharpe']:+.2f}"
                        f" Trd={int(f['oos_trades'])} WR={f['oos_wr']:.0f}%"
                        f" Hold={f['oos_avg_hold']:.0f}h"
                    )

    # Best result
    if all_results:
        best = max(all_results, key=lambda x: x["stitched_sharpe"])
        print(
            f"\n  BEST: {best['filter']} tiers={best['tiers_pct']}"
            f" Sharpe={best['stitched_sharpe']:+.3f}"
            f" Trades={best['total_trades']} Pos={best['pct_positive']:.0%}"
        )
        ok = best["stitched_sharpe"] > 0.5 and best["pct_positive"] >= 0.5
        print(f"  Gate: {'PASS' if ok else 'FAIL'}")

    # Save
    save_path = REPORTS_DIR / f"mr_confluence_wfo_{args.pair.lower()}.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "fold_df"} for r in all_results]).to_csv(
        save_path, index=False
    )
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
