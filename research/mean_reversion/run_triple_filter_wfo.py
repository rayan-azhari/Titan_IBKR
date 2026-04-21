"""run_triple_filter_wfo.py -- WFO for MR FX with Triple Regime Filter.

Combines three regime detectors for the MR VWAP strategy:
  1. HMM P(ranging) >= threshold (trained per-fold on IS data)
  2. Hurst exponent < 0.50 (mean-reverting)
  3. Confluence disagreement (multi-scale signals disagree on direction)

Tests all combinations: each filter alone, pairs, and the triple gate.

Usage:
    uv run python research/mean_reversion/run_triple_filter_wfo.py
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

from research.mean_reversion.regime import (  # noqa: E402
    build_observations,
    ranging_state_index,
    rolling_hurst,
    rolling_regime_posterior,
    train_hmm,
)
from research.mean_reversion.run_confluence_regime_test import (  # noqa: E402
    build_confluence_disagreement_mask,
    compute_percentile_levels,
    compute_vwap_deviation,
    load_h1,
)

TIER_SIZES = [1, 2, 4, 8]
CONSERVATIVE_TIERS = [0.95, 0.98, 0.99, 0.999]


def _fold_backtest(
    close: pd.Series,
    deviation: pd.Series,
    levels: pd.DataFrame,
    regime_mask: pd.Series,
    spread_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> dict:
    """MR backtest on a single fold."""
    n = len(close)
    close_arr = close.values
    dev_arr = deviation.values
    gate_arr = regime_mask.reindex(close.index, fill_value=False).values
    hour_arr = close.index.hour

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

        if combined_gate[i]:
            for tier_idx in range(len(levels.columns)):
                if tier_idx in tiers_hit:
                    continue
                lvl = levels.iloc[i, tier_idx]
                if np.isnan(lvl):
                    continue
                if dev > lvl:
                    size = TIER_SIZES[tier_idx] if tier_idx < len(TIER_SIZES) else 1
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

    daily_pnl = pd.Series(bar_pnl, index=close.index).resample("D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0.0]

    def _sharpe(d):
        from titan.research.metrics import BARS_PER_YEAR as _BPY
        from titan.research.metrics import sharpe as _sh

        if len(d) < 10:
            return 0.0
        return float(_sh(d, periods_per_year=_BPY["D"]))

    n_trades = len(trade_returns)
    wr = sum(1 for r in trade_returns if r > 0) / n_trades * 100 if n_trades > 0 else 0
    avg_hold = np.mean(trade_durations) if trade_durations else 0

    return {
        "sharpe": round(_sharpe(daily_pnl), 3),
        "n_trades": n_trades,
        "win_rate": round(wr, 1),
        "avg_hold": round(avg_hold, 1),
        "daily_pnl": daily_pnl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MR FX Triple Regime Filter WFO")
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--is-bars", type=int, default=30000)
    parser.add_argument("--oos-bars", type=int, default=7500)
    parser.add_argument("--hmm-p-thresh", type=float, default=0.65)
    parser.add_argument("--hurst-thresh", type=float, default=0.50)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  MR FX TRIPLE REGIME FILTER WFO -- {args.pair} H1")
    print(f"  IS: {args.is_bars} | OOS: {args.oos_bars}")
    print(f"  HMM P(ranging) >= {args.hmm_p_thresh} | Hurst < {args.hurst_thresh}")
    print(f"  Tiers: {'/'.join(str(int(p * 100)) for p in CONSERVATIVE_TIERS)}")
    print("=" * 70)

    df = load_h1(args.pair)
    close = df["close"]
    n = len(close)
    print(f"\n  Loaded {n} bars ({close.index[0].date()} -> {close.index[-1].date()})")

    deviation = compute_vwap_deviation(close, anchor_period=24)

    # Pre-compute confluence masks (these don't need per-fold retraining)
    print("\n  Building confluence masks...")
    conf_donchian = build_confluence_disagreement_mask(df, "donchian_pos_20")
    conf_rsi = build_confluence_disagreement_mask(df, "rsi_14_dev")
    print(f"    donchian: {conf_donchian.sum() / n * 100:.0f}% allowed")
    print(f"    rsi_14:   {conf_rsi.sum() / n * 100:.0f}% allowed")

    # Pre-compute Hurst (expensive, do once)
    print("  Computing rolling Hurst exponent (this takes a moment)...")
    hurst = rolling_hurst(close, window=500)
    hurst_mask = hurst < args.hurst_thresh
    print(f"    Hurst < {args.hurst_thresh}: {hurst_mask.sum() / n * 100:.0f}% allowed")

    # Build HMM observations (full series, but model trained per-fold)
    print("  Building HMM observations...")
    obs = build_observations(close)

    # Define filter combinations
    filter_configs = {
        "no_filter": lambda hmm_m: pd.Series(True, index=close.index),
        "hmm_only": lambda hmm_m: hmm_m,
        "hurst_only": lambda hmm_m: hurst_mask,
        "conf_donch_only": lambda hmm_m: conf_donchian,
        "conf_rsi_only": lambda hmm_m: conf_rsi,
        "hmm+hurst": lambda hmm_m: hmm_m & hurst_mask,
        "hmm+conf_donch": lambda hmm_m: hmm_m & conf_donchian,
        "hurst+conf_donch": lambda hmm_m: hurst_mask & conf_donchian,
        "triple_donch": lambda hmm_m: hmm_m & hurst_mask & conf_donchian,
        "triple_rsi": lambda hmm_m: hmm_m & hurst_mask & conf_rsi,
    }

    # Rolling WFO
    print(f"\n  Running WFO ({len(filter_configs)} filters)...")
    print(
        f"\n  {'Filter':<22} | {'St.Sh':>6} | {'Folds':>5} {'Pos%':>5} {'Trd':>5}"
        f" | {'Avg WR':>6} {'Avg Hold':>8}"
    )
    print("  " + "-" * 70)

    all_results = []
    oos_start = args.is_bars

    # First pass: collect fold boundaries and train HMM per fold
    fold_boundaries = []
    while oos_start + args.oos_bars <= n:
        fold_boundaries.append((oos_start - args.is_bars, oos_start, oos_start + args.oos_bars))
        oos_start += args.oos_bars

    # Train HMM per fold (IS only)
    print(f"  Training HMM for {len(fold_boundaries)} folds...")
    hmm_masks_per_fold = []
    for fold_i, (is_start, is_end, oos_end) in enumerate(fold_boundaries):
        is_obs = obs[is_start:is_end]
        oos_index = close.index[is_end:oos_end]
        try:
            model = train_hmm(is_obs, n_states=3, random_state=42)
            ranging_idx = ranging_state_index(model)

            # Compute posterior on full fold range (IS+OOS) using IS-trained model
            fold_obs = obs[is_start:oos_end]
            posteriors = rolling_regime_posterior(model, fold_obs, ranging_idx, min_bars=200)

            # Extract OOS portion
            oos_posteriors = posteriors[is_end - is_start :]
            hmm_mask = pd.Series(oos_posteriors >= args.hmm_p_thresh, index=oos_index)
        except (ValueError, np.linalg.LinAlgError) as e:
            print(
                f"    Fold {fold_i}: HMM failed ({e.__class__.__name__}), using fallback (all True)"
            )
            hmm_mask = pd.Series(True, index=oos_index)
        hmm_masks_per_fold.append(hmm_mask)

    # Now run each filter combination
    for filter_name, build_mask_fn in filter_configs.items():
        fold_results = []
        stitched = []

        for fold_idx, (is_start, is_end, oos_end) in enumerate(fold_boundaries):
            oos_close = close.iloc[is_end:oos_end]
            oos_dev = deviation.iloc[is_end:oos_end]
            is_dev = deviation.iloc[is_start:is_end]

            # Percentile levels from IS
            is_levels = compute_percentile_levels(is_dev, window=500, pcts=CONSERVATIVE_TIERS)
            last_levels = is_levels.dropna().iloc[-1] if len(is_levels.dropna()) > 0 else None
            if last_levels is None:
                continue

            oos_levels = pd.DataFrame(
                {col: last_levels[col] for col in is_levels.columns},
                index=oos_close.index,
            )

            # Build regime mask for this fold
            hmm_fold_mask = hmm_masks_per_fold[fold_idx]
            oos_mask = build_mask_fn(hmm_fold_mask)
            if isinstance(oos_mask, pd.Series):
                oos_mask = oos_mask.reindex(oos_close.index, fill_value=False)
            else:
                oos_mask = pd.Series(True, index=oos_close.index)

            result = _fold_backtest(oos_close, oos_dev, oos_levels, oos_mask)
            fold_results.append(result)
            if result["daily_pnl"] is not None and len(result["daily_pnl"]) > 0:
                stitched.append(result["daily_pnl"])

        # Stitch
        if stitched:
            from titan.research.metrics import BARS_PER_YEAR as _BPY
            from titan.research.metrics import sharpe as _sh

            all_oos = pd.concat(stitched)
            all_oos = all_oos[all_oos != 0.0]
            if len(all_oos) >= 20:
                st_sh = float(_sh(all_oos, periods_per_year=_BPY["D"]))
            else:
                st_sh = 0.0
        else:
            st_sh = 0.0

        n_folds = len(fold_results)
        pct_pos = sum(1 for r in fold_results if r["sharpe"] > 0) / n_folds if n_folds > 0 else 0
        total_trades = sum(r["n_trades"] for r in fold_results)
        avg_wr = (
            np.mean([r["win_rate"] for r in fold_results if r["n_trades"] > 0])
            if any(r["n_trades"] > 0 for r in fold_results)
            else 0
        )
        avg_hold = (
            np.mean([r["avg_hold"] for r in fold_results if r["n_trades"] > 0])
            if any(r["n_trades"] > 0 for r in fold_results)
            else 0
        )

        flag = "+" if st_sh > 0.5 and pct_pos >= 0.5 else " "
        print(
            f" {flag}{filter_name:<21} | {st_sh:>+6.3f}"
            f" | {n_folds:>5} {pct_pos:>4.0%} {total_trades:>5}"
            f" | {avg_wr:>5.0f}% {avg_hold:>7.0f}h"
        )

        all_results.append(
            {
                "filter": filter_name,
                "stitched_sharpe": round(st_sh, 3),
                "n_folds": n_folds,
                "pct_positive": round(pct_pos, 3),
                "total_trades": total_trades,
                "avg_wr": round(avg_wr, 1),
                "avg_hold_h": round(avg_hold, 1),
            }
        )

    # Best
    best = max(all_results, key=lambda x: x["stitched_sharpe"])
    print(
        f"\n  BEST: {best['filter']} Sharpe={best['stitched_sharpe']:+.3f}"
        f" Trades={best['total_trades']} Pos={best['pct_positive']:.0%}"
    )
    ok = best["stitched_sharpe"] > 0.5 and best["pct_positive"] >= 0.5
    print(f"  Gate: {'PASS' if ok else 'FAIL'}")

    save_path = REPORTS_DIR / f"mr_triple_filter_wfo_{args.pair.lower()}.csv"
    pd.DataFrame(all_results).to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
