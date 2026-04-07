"""run_phase1_validation.py -- Phase 1A-1D: LSTM/ML Validation Tests.

Sequential A/B tests for each LSTM architecture:
  1A: LSTM end-to-end classifier (replaces XGBoost)
  1B: Ensemble stacking (XGBoost + LSTM, logistic meta-learner)
  1C: Probabilistic LSTM (Gaussian output, Kelly sizing)
  1D: Multi-horizon LSTM (multi-task regularizer)

Each test uses the same WFO infrastructure as run_52signal_classifier.py.
Decision tree: each phase gates the next (stop-early on failure).

Usage:
    uv run python research/ml/run_phase1_validation.py
    uv run python research/ml/run_phase1_validation.py --instrument GLD
    uv run python research/ml/run_phase1_validation.py --phase 1A
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

from research.ic_analysis.phase1_sweep import _get_annual_bars  # noqa: E402
from research.ml.run_52signal_classifier import (  # noqa: E402
    COST_BPS,
    IS_RATIO_BARS,
    OOS_RATIO_BARS,
    SIGNAL_THRESHOLD,
    TARGET_INSTRUMENTS,
    XGB_PARAMS,
    _pred_to_position,
    compute_signal_sharpe,
    walk_forward_splits,
)
from research.ml.run_phase0_validation import (  # noqa: E402
    LABEL_SWEEP,
    _prepare_data,
    _wfo_loop,
)

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_INSTRUMENTS = ["GLD", "SPY", "EUR_USD", "QQQ"]
DELTA_THRESHOLD = 0.10
LOOKBACK = 20
HIDDEN_DIM = 32


# --- Phase 1A: LSTM End-to-End Classifier ---


def test_lstm_classifier(instrument: str) -> dict:
    """A/B: XGBoost baseline vs LSTM end-to-end classifier."""
    from research.ml.lstm_classifier import (
        predict_lstm_classifier,
        train_lstm_classifier,
    )

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)
    cost = COST_BPS.get(asset_type, 1.0)
    bars_yr = _get_annual_bars(tf)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(feat_clean), is_bars, oos_bars)

    if not folds:
        return _empty_result("1A", "LSTM-E2E", instrument, tf)

    X_all = feat_clean.values
    all_idx = feat_clean.index
    all_oos_rets = []
    fold_sharpes = []

    for is_idx, oos_idx in folds:
        is_mask = np.zeros(len(all_idx), dtype=bool)
        is_mask[is_idx] = True

        best_y = _pick_best_labels(label_cache, all_idx, is_mask)
        if best_y is None:
            continue

        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        y_is = best_y[is_idx]

        model = train_lstm_classifier(
            X_is, y_is, lookback=LOOKBACK, hidden_dim=HIDDEN_DIM,
            epochs=50, patience=7,
        )
        if model is None:
            continue

        X_oos = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        proba = predict_lstm_classifier(model, X_oos, lookback=LOOKBACK)
        pos = _pred_to_position(proba, SIGNAL_THRESHOLD)

        oos_ret = ret_clean.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    lstm_stats = _stitch(all_oos_rets, fold_sharpes, bars_yr)
    delta = lstm_stats["sharpe"] - base["sharpe"]

    return {
        "phase": "1A", "layer": "LSTM-E2E",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": lstm_stats["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
        "n_folds": lstm_stats["n_folds"],
        "positive_pct": lstm_stats["positive_pct"],
    }


# --- Phase 1B: Ensemble Stacking ---


def test_stacking(instrument: str) -> dict:
    """A/B: XGBoost baseline vs XGBoost+LSTM stacking."""
    from research.ml.ensemble_stacking import StackedEnsemble

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)
    cost = COST_BPS.get(asset_type, 1.0)
    bars_yr = _get_annual_bars(tf)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(feat_clean), is_bars, oos_bars)

    if not folds:
        return _empty_result("1B", "Stacking", instrument, tf)

    X_all = feat_clean.values
    all_idx = feat_clean.index
    all_oos_rets = []
    fold_sharpes = []
    meta_coefs = []

    for is_idx, oos_idx in folds:
        is_mask = np.zeros(len(all_idx), dtype=bool)
        is_mask[is_idx] = True

        best_y, best_entries = _pick_best_labels_with_entries(
            label_cache, all_idx, is_mask
        )
        if best_y is None:
            continue

        # Entry mask for IS
        entry_mask_is = np.zeros(len(is_idx), dtype=bool)
        is_entries_in_fold = best_entries[is_mask[best_entries]]
        # Map global indices to local IS indices
        is_start = is_idx[0]
        for e in is_entries_in_fold:
            local = e - is_start
            if 0 <= local < len(is_idx):
                entry_mask_is[local] = True

        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        y_is = best_y[is_idx]

        ensemble = StackedEnsemble(
            lstm_hidden=HIDDEN_DIM, lstm_lookback=LOOKBACK,
            lstm_epochs=30, n_nested_folds=3,
        )
        try:
            ensemble.fit(X_is, y_is, entry_mask_is)
        except Exception:
            continue

        coefs = ensemble.get_meta_coefficients()
        if coefs:
            meta_coefs.append(coefs)

        X_oos = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        proba = ensemble.predict_proba(X_oos)
        pos = _pred_to_position(proba, SIGNAL_THRESHOLD)

        oos_ret = ret_clean.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    stack_stats = _stitch(all_oos_rets, fold_sharpes, bars_yr)
    delta = stack_stats["sharpe"] - base["sharpe"]

    result = {
        "phase": "1B", "layer": "Stacking",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": stack_stats["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
        "n_folds": stack_stats["n_folds"],
        "positive_pct": stack_stats["positive_pct"],
    }

    if meta_coefs:
        avg_xgb = np.mean([c["coef_xgb"] for c in meta_coefs])
        avg_lstm = np.mean([c["coef_lstm"] for c in meta_coefs])
        result["avg_coef_xgb"] = round(avg_xgb, 4)
        result["avg_coef_lstm"] = round(avg_lstm, 4)

    return result


# --- Phase 1C: Probabilistic LSTM ---


def test_probabilistic(instrument: str) -> dict:
    """A/B: XGBoost baseline vs probabilistic LSTM with Kelly sizing."""
    from research.ml.probabilistic_lstm import (
        compute_brier_score,
        predict_probabilistic,
        probabilistic_position,
        train_probabilistic_lstm,
    )

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)
    cost = COST_BPS.get(asset_type, 1.0)
    bars_yr = _get_annual_bars(tf)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(feat_clean), is_bars, oos_bars)

    if not folds:
        return _empty_result("1C", "Prob-LSTM", instrument, tf)

    X_all = feat_clean.values
    all_oos_rets = []
    fold_sharpes = []
    brier_scores = []
    sigma_stds = []

    for is_idx, oos_idx in folds:
        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        y_ret_is = ret_clean.iloc[is_idx].values

        model = train_probabilistic_lstm(
            X_is, y_ret_is, lookback=LOOKBACK, hidden_dim=HIDDEN_DIM,
            epochs=50, patience=7,
        )
        if model is None:
            continue

        X_oos = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        mu, sigma = predict_probabilistic(model, X_oos, lookback=LOOKBACK)
        pos = probabilistic_position(mu, sigma, kelly_fraction=0.25, min_prob=0.55)

        oos_ret = ret_clean.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

        # Diagnostics
        brier = compute_brier_score(mu, sigma, oos_ret.values)
        brier_scores.append(brier)
        sigma_stds.append(float(np.std(sigma[sigma > 0])) if np.any(sigma > 0) else 0.0)

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    prob_stats = _stitch(all_oos_rets, fold_sharpes, bars_yr)
    delta = prob_stats["sharpe"] - base["sharpe"]

    worst_fold = min(fold_sharpes) if fold_sharpes else -99.0
    avg_brier = float(np.mean(brier_scores)) if brier_scores else 1.0
    avg_sigma_std = float(np.mean(sigma_stds)) if sigma_stds else 0.0

    return {
        "phase": "1C", "layer": "Prob-LSTM",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": prob_stats["sharpe"],
        "delta": round(delta, 3),
        "pass": delta >= DELTA_THRESHOLD and worst_fold > -3.0 and avg_brier < 0.25,
        "n_folds": prob_stats["n_folds"],
        "positive_pct": prob_stats["positive_pct"],
        "brier": round(avg_brier, 4),
        "worst_fold": round(worst_fold, 3),
        "sigma_std": round(avg_sigma_std, 6),
    }


# --- Phase 1D: Multi-Horizon LSTM ---


def test_multi_horizon(instrument: str) -> dict:
    """A/B: single-horizon vs multi-horizon LSTM."""
    from research.ml.multi_horizon_lstm import (
        build_multi_horizon_targets,
        predict_multi_horizon,
        train_multi_horizon_lstm,
    )

    df, tf, asset_type, feat_clean, ret_clean, label_cache = _prepare_data(instrument)
    cost = COST_BPS.get(asset_type, 1.0)
    bars_yr = _get_annual_bars(tf)

    close_aligned = df["close"].reindex(feat_clean.index).values
    targets = build_multi_horizon_targets(close_aligned)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(feat_clean), is_bars, oos_bars)

    if not folds:
        return _empty_result("1D", "MultiHz", instrument, tf)

    X_all = feat_clean.values
    all_oos_rets = []
    fold_sharpes = []

    for is_idx, oos_idx in folds:
        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        targets_is = [t[is_idx] for t in targets]

        model = train_multi_horizon_lstm(
            X_is, targets_is, lookback=LOOKBACK, hidden_dim=HIDDEN_DIM,
            epochs=50, patience=7,
        )
        if model is None:
            continue

        X_oos = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        pred_ret = predict_multi_horizon(model, X_oos, lookback=LOOKBACK)

        # Convert predicted return to position: positive pred = long, negative = short
        pos = np.where(pred_ret > 0.001, 1.0, np.where(pred_ret < -0.001, -1.0, 0.0))

        oos_ret = ret_clean.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

    base = _wfo_loop(feat_clean, ret_clean, label_cache, tf, asset_type)
    mh_stats = _stitch(all_oos_rets, fold_sharpes, bars_yr)
    delta = mh_stats["sharpe"] - base["sharpe"]

    return {
        "phase": "1D", "layer": "MultiHz",
        "instrument": instrument, "tf": tf,
        "base_sharpe": base["sharpe"], "aug_sharpe": mh_stats["sharpe"],
        "delta": round(delta, 3), "pass": delta >= DELTA_THRESHOLD,
        "n_folds": mh_stats["n_folds"],
        "positive_pct": mh_stats["positive_pct"],
    }


# --- Helpers ---


def _pick_best_labels(label_cache, all_idx, is_mask):
    """Pick label params with most IS entries (balanced)."""
    best_y = None
    best_count = 0
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
            best_y = (lab == 1).astype(int)
    return best_y


def _pick_best_labels_with_entries(label_cache, all_idx, is_mask):
    """Same as above but also returns entry positions."""
    best_y = None
    best_entries = None
    best_count = 0
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
            best_y = (lab == 1).astype(int)
            best_entries = entries
    return best_y, best_entries


def _stitch(all_oos_rets, fold_sharpes, bars_yr):
    """Stitch OOS returns and compute summary stats."""
    if not all_oos_rets:
        return {"sharpe": 0.0, "n_folds": 0, "positive_pct": 0.0}
    stitched = pd.concat(all_oos_rets).sort_index()
    std = float(stitched.std())
    sharpe = float(stitched.mean() / std * np.sqrt(bars_yr)) if std > 1e-10 else 0.0
    pos_pct = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
    return {
        "sharpe": round(sharpe, 3),
        "n_folds": len(fold_sharpes),
        "positive_pct": round(pos_pct, 3),
    }


def _empty_result(phase, layer, instrument, tf):
    return {
        "phase": phase, "layer": layer,
        "instrument": instrument, "tf": tf,
        "base_sharpe": 0.0, "aug_sharpe": 0.0,
        "delta": 0.0, "pass": False,
        "n_folds": 0, "positive_pct": 0.0,
    }


# --- Main ---


PHASE_MAP = {
    "1A": ("LSTM-E2E", test_lstm_classifier),
    "1B": ("Stacking", test_stacking),
    "1C": ("Prob-LSTM", test_probabilistic),
    "1D": ("MultiHz", test_multi_horizon),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1A-1D: LSTM/ML Validation")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--phase", default=None, help="Run single phase (1A/1B/1C/1D)")
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

        phase_results = []
        for inst in instruments:
            t0 = time.time()
            try:
                result = test_fn(inst)
                elapsed = time.time() - t0
                result["elapsed_s"] = round(elapsed, 1)
                all_results.append(result)
                phase_results.append(result)
                status = "LIFT" if result["pass"] else "flat"
                extra = ""
                if "brier" in result:
                    extra = f"  brier={result['brier']:.3f}"
                if "avg_coef_lstm" in result:
                    extra = f"  coef_lstm={result['avg_coef_lstm']:.3f}"
                print(
                    f"  [{phase}] {inst:<10} base={result['base_sharpe']:+.3f}"
                    f"  aug={result['aug_sharpe']:+.3f}"
                    f"  delta={result['delta']:+.3f}  {status}"
                    f"{extra}  ({elapsed:.0f}s)"
                )
            except Exception as e:
                print(f"  [{phase}] {inst:<10} ERROR: {e}")

        # Phase gate check
        if phase_results:
            n_pass = sum(1 for r in phase_results if r["pass"])
            avg_delta = np.mean([r["delta"] for r in phase_results])
            print(f"\n  Phase {phase} gate: {n_pass}/{len(phase_results)} pass, avg delta={avg_delta:+.3f}")

            if phase == "1A" and n_pass == 0 and avg_delta <= 0:
                print("  STOP: LSTM shows no signal. Skipping 1B-1D.")
                break

    if not all_results:
        print("\n  No results.")
        return

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  PHASE 1A-1D VALIDATION SUMMARY")
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
    save_path = REPORTS_DIR / "phase1_validation.csv"
    df_results.to_csv(save_path, index=False)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
