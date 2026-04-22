"""run_fibonacci_lift.py -- Phase 0B: Fibonacci Feature Lift Test.

Compares XGBoost WFO performance WITH vs WITHOUT Fibonacci retracement
features. Uses the same 52-signal + MA pipeline from run_52signal_classifier.py
as baseline, then adds 6 Fibonacci features and measures delta Sharpe.

Pass criterion: Fibonacci features must improve stitched OOS Sharpe by >= +0.10
on at least 2 of 4 test instruments.

Usage:
    uv run python research/retracement/run_fibonacci_lift.py
    uv run python research/retracement/run_fibonacci_lift.py --instrument GLD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
)
from research.ml.run_52signal_classifier import (  # noqa: E402
    COST_BPS,
    IS_RATIO_BARS,
    OOS_RATIO_BARS,
    SIGNAL_THRESHOLD,
    TARGET_INSTRUMENTS,
    XGB_PARAMS,
    _pred_to_position,
    build_features,
    compute_regime_pullback_labels,
    compute_signal_sharpe,
    walk_forward_splits,
)
from research.retracement.fibonacci_features import (  # noqa: E402
    compute_fibonacci_features,
)
from titan.strategies.ml.features import atr as feat_atr  # noqa: E402

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Instruments to test (diverse: gold, index, FX, miner)
TEST_INSTRUMENTS = ["GLD", "SPY", "EUR_USD", "QQQ"]

# Label sweep (same as main pipeline, trimmed for speed)
LABEL_SWEEP = [
    {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 10, "confirm_pct": 0.005},
    {"rsi_oversold": 50, "rsi_overbought": 50, "confirm_bars": 10, "confirm_pct": 0.003},
    {"rsi_oversold": 48, "rsi_overbought": 52, "confirm_bars": 10, "confirm_pct": 0.005},
]


def _run_wfo(
    features: pd.DataFrame,
    df: pd.DataFrame,
    tf: str,
    asset_type: str,
    tag: str,
) -> dict:
    """Run WFO loop and return stitched stats."""
    from xgboost import XGBClassifier

    cost = COST_BPS.get(asset_type, 1.0)
    bars_yr = _get_annual_bars(tf)
    bar_returns = df["close"].pct_change().fillna(0.0)

    # Pre-compute labels
    label_cache = []
    for lp in LABEL_SWEEP:
        labels, _ = compute_regime_pullback_labels(df, **lp)
        label_cache.append((lp, labels))

    mask = features.notna().all(axis=1)
    features_clean = features[mask].copy()
    returns_clean = bar_returns.reindex(features_clean.index).fillna(0.0)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(features_clean), is_bars, oos_bars)

    if not folds:
        return {"tag": tag, "sharpe": 0.0, "n_folds": 0, "positive_pct": 0.0}

    X_all = features_clean.values
    all_idx = features_clean.index

    all_oos_rets = []
    fold_sharpes = []

    for is_idx, oos_idx in folds:
        # Pick best label params for this fold
        is_mask = np.zeros(len(all_idx), dtype=bool)
        is_mask[is_idx] = True

        best_count = 0
        best_y = None
        best_entries = None

        for lp, labels in label_cache:
            lab = labels.reindex(all_idx).fillna(0).values
            entries = np.where(lab != 0)[0]
            is_entries = entries[is_mask[entries]]
            if len(is_entries) < 20:
                continue
            y_is = (lab[is_entries] == 1).astype(int)
            minority = min(y_is.mean(), 1 - y_is.mean())
            if minority < 0.15:
                continue
            if len(is_entries) > best_count:
                best_count = len(is_entries)
                best_entries = entries
                best_y = (lab == 1).astype(int)

        if best_entries is None:
            continue

        is_entries = best_entries[is_mask[best_entries]]
        X_train = np.nan_to_num(X_all[is_entries], nan=0.0, posinf=0.0, neginf=0.0)
        y_train = best_y[is_entries]

        spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        params = {**XGB_PARAMS, "scale_pos_weight": spw, "eval_metric": "logloss"}
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        X_oos = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        pred = model.predict_proba(X_oos)[:, 1]
        pos = _pred_to_position(pred, SIGNAL_THRESHOLD)

        oos_ret = returns_clean.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

    if not all_oos_rets:
        return {"tag": tag, "sharpe": 0.0, "n_folds": 0, "positive_pct": 0.0}

    stitched = pd.concat(all_oos_rets).sort_index()
    std = float(stitched.std())
    sharpe = float(stitched.mean() / std * np.sqrt(bars_yr)) if std > 1e-10 else 0.0
    pos_pct = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)

    return {
        "tag": tag,
        "sharpe": round(sharpe, 3),
        "n_folds": len(fold_sharpes),
        "positive_pct": round(pos_pct, 3),
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
    }


def run_lift_test(instrument: str) -> dict:
    """Run A/B test: baseline vs baseline+Fibonacci for one instrument."""
    if instrument in TARGET_INSTRUMENTS:
        tf, asset_type = TARGET_INSTRUMENTS[instrument]
    else:
        tf, asset_type = "D", "index"

    print(f"\n{'=' * 60}")
    print(f"  FIBONACCI LIFT TEST: {instrument} {tf}")
    print(f"{'=' * 60}")

    df = _load_ohlcv(instrument, tf)
    print(f"  Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # A: Baseline features (same as run_52signal_classifier)
    print("  Computing baseline features (~72) ...")
    df.attrs["instrument"] = instrument
    base_features = build_features(df, tf)

    # B: Baseline + Fibonacci features
    print("  Computing Fibonacci features (6) ...")
    atr_vals = feat_atr(df)
    fib_features = compute_fibonacci_features(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        atr=atr_vals,
        order=5,
    )
    # Shift 1 already applied inside compute_fibonacci_features

    augmented_features = pd.concat([base_features, fib_features], axis=1)

    # Run WFO for both
    print("\n  Running WFO: BASELINE ...")
    baseline = _run_wfo(base_features, df, tf, asset_type, "baseline")

    print("\n  Running WFO: BASELINE + FIBONACCI ...")
    augmented = _run_wfo(augmented_features, df, tf, asset_type, "baseline+fib")

    delta = augmented["sharpe"] - baseline["sharpe"]

    print(f"\n  {'─' * 50}")
    print(f"  {'Config':<25s} {'Sharpe':>8} {'Folds':>6} {'Pos%':>6}")
    print(f"  {'─' * 50}")
    print(
        f"  {'Baseline':<25s} {baseline['sharpe']:>+8.3f}"
        f" {baseline['n_folds']:>6} {baseline['positive_pct']:>5.0%}"
    )
    print(
        f"  {'Baseline + Fibonacci':<25s} {augmented['sharpe']:>+8.3f}"
        f" {augmented['n_folds']:>6} {augmented['positive_pct']:>5.0%}"
    )
    print(f"  {'─' * 50}")
    print(f"  Delta Sharpe: {delta:+.3f}  {'LIFT' if delta >= 0.10 else 'NO LIFT'}")

    return {
        "instrument": instrument,
        "tf": tf,
        "baseline_sharpe": baseline["sharpe"],
        "augmented_sharpe": augmented["sharpe"],
        "delta": round(delta, 3),
        "pass": delta >= 0.10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0B: Fibonacci Feature Lift Test")
    parser.add_argument(
        "--instrument", default=None, help="Single instrument (default: 4 test set)"
    )
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else TEST_INSTRUMENTS

    results = []
    for inst in instruments:
        try:
            r = run_lift_test(inst)
            results.append(r)
        except Exception as e:
            print(f"\n  [ERROR] {inst}: {e}")
            continue

    if not results:
        print("\n  No results.")
        return

    # Summary
    print(f"\n\n{'=' * 60}")
    print("  PHASE 0B SUMMARY: FIBONACCI FEATURE LIFT")
    print(f"{'=' * 60}")
    print(f"  {'Instrument':<12} {'TF':>3} {'Base':>7} {'+ Fib':>7} {'Delta':>7} {'Pass':>5}")
    print(f"  {'─' * 45}")
    for r in results:
        print(
            f"  {r['instrument']:<12} {r['tf']:>3}"
            f" {r['baseline_sharpe']:>+7.3f} {r['augmented_sharpe']:>+7.3f}"
            f" {r['delta']:>+7.3f} {'YES' if r['pass'] else 'NO':>5}"
        )

    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n  Instruments with lift: {n_pass}/{len(results)}")
    print(f"  VERDICT: {'PASS' if n_pass >= 2 else 'FAIL'} (need >= 2 with delta >= +0.10)")

    # Save
    df_results = pd.DataFrame(results)
    save_path = REPORTS_DIR / "fibonacci_lift_test.csv"
    df_results.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
