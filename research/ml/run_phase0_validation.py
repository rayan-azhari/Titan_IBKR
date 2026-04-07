"""run_phase0_validation.py -- Phase 0B-0E: Quick Validation Tests.

Runs A/B comparisons for each ML enhancement layer:
  0B: Fibonacci features (+6 features)
  0C: LSTM temporal features (+64 features)
  0D: Quantile regression features (+5 features)
  0E: Calibrated Kelly position sizing (replaces threshold-based sizing)

Each test uses the same WFO infrastructure as run_52signal_classifier.py.
Reports delta Sharpe for each layer and identifies which layers pass.

Pass criteria per layer:
  - Delta Sharpe >= +0.10 on majority of test instruments

Usage:
    uv run python research/ml/run_phase0_validation.py
    uv run python research/ml/run_phase0_validation.py --instrument GLD
    uv run python research/ml/run_phase0_validation.py --phase 0C
"""

from __future__ import annotations

import argparse
import sys
import time
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
from titan.strategies.ml.features import atr as feat_atr  # noqa: E402

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_INSTRUMENTS = ["GLD", "SPY", "EUR_USD", "QQQ"]

LABEL_SWEEP = [
    {"rsi_oversold": 45, "rsi_overbought": 55,
     "confirm_bars": 10, "confirm_pct": 0.005},
    {"rsi_oversold": 50, "rsi_overbought": 50,
     "confirm_bars": 10, "confirm_pct": 0.003},
    {"rsi_oversold": 48, "rsi_overbought": 52,
     "confirm_bars": 10, "confirm_pct": 0.005},
]

DELTA_THRESHOLD = 0.10  # minimum Sharpe improvement to count as a lift


def _prepare_data(instrument: str):
    """Load data, compute base features + labels. Shared across all tests."""
    if instrument in TARGET_INSTRUMENTS:
        tf, asset_type = TARGET_INSTRUMENTS[instrument]
    else:
        tf, asset_type = "D", "index"

    df = _load_ohlcv(instrument, tf)
    df.attrs["instrument"] = instrument
    base_features = build_features(df, tf)

    label_cache = []
    for lp in LABEL_SWEEP:
        labels, _ = compute_regime_pullback_labels(df, **lp)
        label_cache.append((lp, labels))

    bar_returns = df["close"].pct_change().fillna(0.0)
    mask = base_features.notna().all(axis=1)
    feat_clean = base_features[mask].copy()
    ret_clean = bar_returns.reindex(feat_clean.index).fillna(0.0)

    return df, tf, asset_type, feat_clean, ret_clean, label_cache


def _wfo_loop(
    features: pd.DataFrame,
    returns: pd.Series,
    label_cache: list,
    tf: str,
    asset_type: str,
    position_fn=None,
    per_fold_hook=None,
) -> dict:
    """Generic WFO loop. Returns stitched Sharpe stats.

    position_fn: optional callable(model, X_oos, fold_info) -> positions array.
                 If None, uses default threshold-based sizing.
    per_fold_hook: optional callable(fold_idx, is_idx, oos_idx, X_all, all_idx,
                   label_cache) that returns extra features or modifies X_all.
    """
    from xgboost import XGBClassifier

    cost = COST_BPS.get(asset_type, 1.0)
    bars_yr = _get_annual_bars(tf)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(features), is_bars, oos_bars)

    if not folds:
        return {"sharpe": 0.0, "n_folds": 0, "positive_pct": 0.0, "fold_sharpes": []}

    X_all = features.values
    all_idx = features.index

    all_oos_rets = []
    fold_sharpes = []

    for fi, (is_idx, oos_idx) in enumerate(folds):
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

        # Allow per-fold feature augmentation (for LSTM, QR)
        X_fold = X_all
        if per_fold_hook is not None:
            X_fold = per_fold_hook(fi, is_idx, oos_idx, X_all, all_idx, label_cache)
            if X_fold is None:
                continue

        is_entries = best_entries[is_mask[best_entries]]
        X_train = np.nan_to_num(X_fold[is_entries], nan=0.0, posinf=0.0, neginf=0.0)
        y_train = best_y[is_entries]

        spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        params = {**XGB_PARAMS, "scale_pos_weight": spw, "eval_metric": "logloss"}
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        X_oos = np.nan_to_num(X_fold[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)

        if position_fn is not None:
            fold_info = {
                "model": model, "is_idx": is_idx, "oos_idx": oos_idx,
                "X_all": X_fold, "returns": returns, "all_idx": all_idx,
            }
            pos = position_fn(model, X_oos, fold_info)
        else:
            pred = model.predict_proba(X_oos)[:, 1]
            pos = _pred_to_position(pred, SIGNAL_THRESHOLD)

        oos_ret = returns.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

    if not all_oos_rets:
        return {"sharpe": 0.0, "n_folds": 0, "positive_pct": 0.0, "fold_sharpes": []}

    stitched = pd.concat(all_oos_rets).sort_index()
    std = float(stitched.std())
    sharpe = float(stitched.mean() / std * np.sqrt(bars_yr)) if std > 1e-10 else 0.0
    pos_pct = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)

    return {
        "sharpe": round(sharpe, 3),
        "n_folds": len(fold_sharpes),
        "positive_pct": round(pos_pct, 3),
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
    }


# ─── Phase 0B: Fibonacci Features ────────────────────────────────────────────


def test_fibonacci(instrument: str) -> dict:
    """A/B: baseline vs baseline+fibonacci."""
    from research.retracement.fibonacci_features import compute_fibonacci_features

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)

    # Augmented features
    atr_vals = feat_atr(df)
    fib = compute_fibonacci_features(df["close"], df["high"], df["low"], atr_vals, order=5)
    fib.index = df.index
    aug = pd.concat([feat_clean, fib.reindex(feat_clean.index)], axis=1)

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    augmented = _wfo_loop(aug, ret_clean, label_cache, tf, asset_type)

    delta = augmented["sharpe"] - base["sharpe"]
    return {
        "phase": "0B", "layer": "Fibonacci",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": augmented["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
    }


# ─── Phase 0C: LSTM Temporal Features ────────────────────────────────────────


def test_lstm(instrument: str) -> dict:
    """A/B: baseline vs baseline+LSTM hidden states."""
    from research.ml.lstm_features import (
        LSTMFeatureExtractor,
        build_sequences,
        extract_lstm_features,
        train_lstm,
    )

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)

    LOOKBACK = 20
    HIDDEN_DIM = 64

    def lstm_fold_hook(fi, is_idx, oos_idx, X_all, all_idx, label_cache_):
        """Train LSTM on IS, extract features for IS+OOS."""
        # Get labels for this fold
        is_mask = np.zeros(len(all_idx), dtype=bool)
        is_mask[is_idx] = True

        best_y = None
        best_count = 0
        for lp, labels in label_cache_:
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
                best_y = (lab == 1).astype(int)

        if best_y is None:
            return None

        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        y_is = best_y[is_idx]

        # Train LSTM on IS
        try:
            model = train_lstm(
                X_is, y_is, lookback=LOOKBACK,
                hidden_dim=HIDDEN_DIM, epochs=20, patience=3,
            )
        except (ValueError, RuntimeError):
            return None

        # Extract features for all bars
        X_clean = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        lstm_feats = extract_lstm_features(model, X_clean, lookback=LOOKBACK)

        # Concatenate with base features
        X_aug = np.column_stack([X_all, lstm_feats.values])
        return X_aug

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)

    # For LSTM: use per_fold_hook to train LSTM per fold
    aug_feat = feat_clean.copy()
    augmented = _wfo_loop(
        aug_feat, ret_clean, label_cache, tf, asset_type,
        per_fold_hook=lstm_fold_hook,
    )

    delta = augmented["sharpe"] - base["sharpe"]
    return {
        "phase": "0C", "layer": "LSTM",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": augmented["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
    }


# ─── Phase 0D: Quantile Regression Features ──────────────────────────────────


def test_quantile(instrument: str) -> dict:
    """A/B: baseline vs baseline+quantile regression features."""
    from research.ml.quantile_features import QuantileFeatureExtractor

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)

    def qr_fold_hook(fi, is_idx, oos_idx, X_all, all_idx, label_cache_):
        """Fit quantile regressors on IS, predict for IS+OOS."""
        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        rets_is = ret_clean.iloc[is_idx].values

        qr = QuantileFeatureExtractor()
        try:
            qr.fit(X_is, rets_is)
        except Exception:
            return None

        # Predict for all bars
        X_clean = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        qr_feats = qr.transform(X_clean)

        X_aug = np.column_stack([X_all, qr_feats.values])
        return X_aug

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    augmented = _wfo_loop(
        feat_clean, ret_clean, label_cache, tf, asset_type,
        per_fold_hook=qr_fold_hook,
    )

    delta = augmented["sharpe"] - base["sharpe"]
    return {
        "phase": "0D", "layer": "Quantile",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": augmented["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
    }


# ─── Phase 0E: Calibrated Kelly Sizing ───────────────────────────────────────


def test_kelly(instrument: str) -> dict:
    """A/B: threshold sizing vs calibrated Kelly sizing."""
    from research.ml.calibration_kelly import CalibratedKellySizer

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)

    def kelly_position_fn(model, X_oos, fold_info):
        """Use calibrated Kelly instead of threshold sizing."""
        is_idx = fold_info["is_idx"]
        X_all = fold_info["X_all"]
        returns = fold_info["returns"]

        # Fit calibrator on IS data
        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        is_proba = model.predict_proba(X_is)[:, 1]
        is_rets = returns.iloc[is_idx].values

        sizer = CalibratedKellySizer(kelly_fraction=0.5, min_edge=0.02)
        try:
            sizer.fit(is_proba, is_rets)
        except Exception:
            # Fallback to threshold
            pred = model.predict_proba(X_oos)[:, 1]
            return _pred_to_position(pred, SIGNAL_THRESHOLD)

        # Apply Kelly to OOS
        oos_proba = model.predict_proba(X_oos)[:, 1]
        return sizer.predict(oos_proba)

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    kelly = _wfo_loop(
        feat_clean, ret_clean, label_cache, tf, asset_type,
        position_fn=kelly_position_fn,
    )

    delta = kelly["sharpe"] - base["sharpe"]
    return {
        "phase": "0E", "layer": "Kelly",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": kelly["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


PHASE_MAP = {
    "0B": ("Fibonacci", test_fibonacci),
    "0C": ("LSTM", test_lstm),
    "0D": ("Quantile", test_quantile),
    "0E": ("Kelly", test_kelly),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0B-0E: ML Layer Validation")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--phase", default=None, help="Run single phase (0B/0C/0D/0E)")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else TEST_INSTRUMENTS
    phases = [args.phase] if args.phase else list(PHASE_MAP.keys())

    all_results = []

    for phase in phases:
        if phase not in PHASE_MAP:
            print(f"  Unknown phase: {phase}")
            continue

        name, test_fn = PHASE_MAP[phase]
        print(f"\n{'#' * 60}")
        print(f"  PHASE {phase}: {name.upper()} VALIDATION")
        print(f"{'#' * 60}")

        for inst in instruments:
            t0 = time.time()
            try:
                result = test_fn(inst)
                elapsed = time.time() - t0
                result["elapsed_s"] = round(elapsed, 1)
                all_results.append(result)
                status = "LIFT" if result["pass"] else "flat"
                print(
                    f"  [{phase}] {inst:<10} base={result['base_sharpe']:+.3f}"
                    f"  aug={result['aug_sharpe']:+.3f}"
                    f"  delta={result['delta']:+.3f}  {status}"
                    f"  ({elapsed:.0f}s)"
                )
            except Exception as e:
                print(f"  [{phase}] {inst:<10} ERROR: {e}")

    if not all_results:
        print("\n  No results.")
        return

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  PHASE 0B-0E VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  {'Phase':>5} {'Layer':<12} {'Instrument':<10}"
        f" {'Base':>7} {'Aug':>7} {'Delta':>7} {'Pass':>5}"
    )
    print(f"  {'-' * 60}")
    for r in all_results:
        print(
            f"  {r['phase']:>5} {r['layer']:<12} {r['instrument']:<10}"
            f" {r['base_sharpe']:>+7.3f} {r['aug_sharpe']:>+7.3f}"
            f" {r['delta']:>+7.3f} {'YES' if r['pass'] else 'NO':>5}"
        )

    # Per-phase verdict
    print(f"\n  Per-phase verdicts:")
    for phase in phases:
        if phase not in PHASE_MAP:
            continue
        phase_results = [r for r in all_results if r["phase"] == phase]
        if not phase_results:
            continue
        n_pass = sum(1 for r in phase_results if r["pass"])
        n_total = len(phase_results)
        avg_delta = np.mean([r["delta"] for r in phase_results])
        verdict = "PASS" if n_pass >= max(1, n_total // 2) else "FAIL"
        name = PHASE_MAP[phase][0]
        print(
            f"    {phase} {name:<12}: {n_pass}/{n_total} instruments lifted,"
            f" avg delta={avg_delta:+.3f}  -> {verdict}"
        )

    # Save
    df_results = pd.DataFrame(all_results)
    save_path = REPORTS_DIR / "phase0_validation.csv"
    df_results.to_csv(save_path, index=False)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
