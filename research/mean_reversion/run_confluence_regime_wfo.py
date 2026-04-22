"""run_confluence_regime_wfo.py -- WFO for MR FX + Confluence Regime Filter.

Validates the confluence disagreement regime filter via rolling WFO.
Per fold: z-score/percentile calibrated on IS, applied to OOS.

Usage:
    uv run python research/mean_reversion/run_confluence_regime_wfo.py
    uv run python research/mean_reversion/run_confluence_regime_wfo.py --signal donchian_pos_20
"""

import argparse
import sys
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
    stitched_trades: list[float] = []
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
        # Plumb trade-level returns up to the portfolio scaler.
        stitched_trades.extend(oos_result.get("trade_returns", []))

        fold_idx += 1
        oos_start += oos_bars

    # Stitch. Sharpe goes through the shared ``titan.research.metrics.sharpe``
    # — no filtering of zero-return days (that bias overstated Sharpe by
    # sqrt(1/active_ratio) in the old implementation).
    from titan.research.metrics import BARS_PER_YEAR as _BPY
    from titan.research.metrics import bootstrap_sharpe_ci as _boot_ci
    from titan.research.metrics import sharpe as _sh_metric

    st_ci_lo = 0.0
    st_ci_hi = 0.0
    if stitched_pnl:
        all_oos = pd.concat(stitched_pnl)
        if len(all_oos) >= 20:
            st_sh = _sh_metric(all_oos, periods_per_year=_BPY["D"])
            st_ci_lo, st_ci_hi = _boot_ci(
                all_oos, periods_per_year=_BPY["D"], n_resamples=1000, seed=42
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

    raw_returns = pd.concat(stitched_pnl) if stitched_pnl else pd.Series(dtype=float)

    return {
        "fold_df": fold_df,
        "stitched_sharpe": round(st_sh, 3),
        # 95% bootstrap CI. Deployment gate: strategies whose lower bound
        # <= 0 should be tagged tier=unconfirmed (April 2026 audit).
        "sharpe_ci_95_lo": round(st_ci_lo, 3),
        "sharpe_ci_95_hi": round(st_ci_hi, 3),
        "stitched_dd_pct": round(st_dd * 100, 2),
        "n_folds": len(folds),
        "pct_positive": round(pct_positive, 3),
        "total_trades": int(total_trades),
        "stitched_returns": raw_returns,
        # Trade-level returns across every OOS fold. Downstream sizing
        # (scale_to_risk) prefers these over daily P&L when available
        # because trade-level vol is the correct input for a "1% per
        # trade" risk budget.
        "stitched_trades": list(stitched_trades),
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
    """Run MR backtest on a single fold slice.

    Pyramiding semantics (April 2026 audit fix).

    The live ``MRAUDJPYStrategy`` accumulates tier entries across bars —
    e.g. tier 0 fires at bar 100, tier 1 at bar 102 — and holds the summed
    position until an exit triggers. The previous backtest reset
    ``position[i] = 0`` at the start of every bar from ``np.zeros(n)`` and
    only carried forward from the prior bar in a narrow branch, so a
    subsequent tier at a new bar *replaced* the prior position instead of
    adding to it. The backtest consequently simulated a different strategy
    than what was running live.

    The new loop keeps ``current_pos``, ``current_tiers_hit``, and a
    weighted ``entry_price`` across bars, so the pyramid accumulates and
    exits close the *full* accumulated position.
    """
    n = len(close)
    close_arr = close.values
    dev_arr = deviation.values
    gate_arr = regime_mask.reindex(close.index, fill_value=False).values
    hour_arr = close.index.hour

    # Session filter: 07:00-12:00 UTC
    session_arr = (hour_arr >= 7) & (hour_arr < 12)
    combined_gate = gate_arr & session_arr

    # Persistent trade state (mirrors live strategy).
    current_pos: float = 0.0  # signed total exposure
    entry_price: float = 0.0  # weighted-average entry
    entry_bar: int = 0
    current_tiers_hit: set[int] = set()

    trade_returns: list[float] = []
    trade_durations: list[int] = []
    bar_pnl = np.zeros(n)
    cost_per_unit = (spread_bps + slippage_bps) / 10_000
    # For reporting / correlation only — the P&L driver is bar_pnl.
    position_trace = np.zeros(n)

    def _close_position(i: int, px: float) -> None:
        nonlocal current_pos, entry_price, current_tiers_hit
        direction = np.sign(current_pos)
        size = abs(current_pos)
        if size <= 0 or entry_price <= 0:
            current_pos = 0.0
            entry_price = 0.0
            current_tiers_hit = set()
            return
        gross = (px - entry_price) / entry_price * direction
        # Exit cost is charged proportional to size exited.
        trade_ret = gross - cost_per_unit
        trade_returns.append(trade_ret)
        trade_durations.append(i - entry_bar)
        # Mark-to-market: bar_pnl on the exit bar reflects the move from the
        # prior bar plus the realised exit (cost). Keep simple: realise the
        # whole trade PnL on the exit bar for attribution.
        bar_pnl[i] += trade_ret * size
        current_pos = 0.0
        entry_price = 0.0
        current_tiers_hit = set()

    for i in range(1, n):
        px = close_arr[i]
        prev_px = close_arr[i - 1]
        dev = abs(dev_arr[i]) if not np.isnan(dev_arr[i]) else 0.0

        # 1) Carry-forward mark-to-market for any existing position. This
        #    happens *first* so the exit return below is computed against
        #    entry_price (not prev_px).
        if current_pos != 0.0:
            bar_pnl[i] += (px - prev_px) / prev_px * current_pos
            position_trace[i] = current_pos

        # 2) Exit checks.
        if current_pos != 0.0:
            if hour_arr[i] >= 21:
                _close_position(i, px)
                position_trace[i] = 0.0
                continue

            lvl0 = levels.iloc[i, 0] if not np.isnan(levels.iloc[i, 0]) else 999
            if dev < lvl0 * 0.5:
                _close_position(i, px)
                position_trace[i] = 0.0
                continue

        # 3) Entry checks — gate on session + regime mask.
        if combined_gate[i]:
            for tier_idx in range(len(levels.columns)):
                if tier_idx in current_tiers_hit:
                    continue
                lvl = levels.iloc[i, tier_idx]
                if np.isnan(lvl):
                    continue
                if dev <= lvl:
                    continue
                size = tier_sizes[tier_idx] if tier_idx < len(tier_sizes) else 1
                direction = -1.0 if dev_arr[i] > 0 else 1.0
                # Require new tier's direction to match current direction if
                # already in a position — a flip would be a re-entry.
                if current_pos != 0.0 and np.sign(current_pos) != direction:
                    # Directional disagreement: treat as exit, then skip
                    # remaining tier checks on this bar (next bar can re-enter).
                    _close_position(i, px)
                    position_trace[i] = 0.0
                    break
                if current_pos == 0.0:
                    entry_price = px
                    entry_bar = i
                else:
                    old_abs = abs(current_pos)
                    entry_price = (entry_price * old_abs + px * size) / (old_abs + size)
                current_pos += direction * size
                current_tiers_hit.add(tier_idx)
                # Entry cost charged on incremental size.
                bar_pnl[i] -= cost_per_unit * size
            position_trace[i] = current_pos

    daily_pnl = pd.Series(bar_pnl, index=close.index).resample("D").sum()
    # NOTE: filtering to non-zero days is now intentionally *not* done before
    # computing Sharpe. Zero-day inclusion is the correct annualisation (see
    # titan.research.metrics.sharpe for the why).

    def _dd(d):
        if len(d) < 5:
            return 0.0
        eq = (1 + d).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())

    # Import here to avoid heavy imports at module load.
    from titan.research.metrics import BARS_PER_YEAR
    from titan.research.metrics import sharpe as _sh_metric

    n_trades = len(trade_returns)
    wr = sum(1 for r in trade_returns if r > 0) / n_trades * 100 if n_trades > 0 else 0
    avg_hold = np.mean(trade_durations) if trade_durations else 0

    return {
        "sharpe": round(_sh_metric(daily_pnl, periods_per_year=BARS_PER_YEAR["D"]), 3),
        "n_trades": n_trades,
        "win_rate": round(wr, 1),
        "avg_hold": round(avg_hold, 1),
        "dd_pct": round(_dd(daily_pnl) * 100, 2),
        "daily_pnl": daily_pnl,
        "trade_returns": trade_returns,
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
