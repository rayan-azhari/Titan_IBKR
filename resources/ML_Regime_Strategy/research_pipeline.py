"""research_pipeline.py

Full end-to-end research pipeline.

Calling run() produces:
  - Trained RegimeDetector and MetaLabeller saved to disk
  - vectorbt backtest with exits derived from the SAME barrier logic as labels
  - Training metadata (regime dist, prob stats) for health monitor init
  - Diagnostic report

The central constraint: every component in this pipeline uses the exact same
class instances and parameter values as the production Nautilus strategy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from .cross_validation import PurgedKFold, compute_sample_uniqueness
from .feature_engine import FeatureEngine
from .labelling import TripleBarrierLabeller
from .meta_labeller import MetaLabeller
from .regime_detection import RegimeDetector


def run(
    prices: pd.Series,
    output_dir: str = "models",
    # FeatureEngine
    vol_span: int = 20,
    frac_d: float = 0.4,
    frac_tau: float = 1e-4,
    # HMM
    n_hmm_states: int = 2,
    hmm_min_seq_len: int = 60,
    # Primary signal
    fast_ma: int = 10,
    slow_ma: int = 30,
    # Barriers
    vol_mult_upper: float = 2.0,
    vol_mult_lower: float = 1.0,
    max_holding_bars: int = 5,
    # CV
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    # Backtest
    fees: float = 0.001,
    slippage: float = 0.001,
) -> dict:
    """Run the full research pipeline and return trained artefacts + backtest results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: FEATURE ENGINEERING")
    print("=" * 60)

    fe = FeatureEngine(vol_span=vol_span, frac_d=frac_d, frac_tau=frac_tau)
    features = fe.batch(prices)

    print(f"  Frac-diff memory window: {fe.frac_window} bars (tau={frac_tau})")
    print(f"  Total warm-up bars: {fe.min_history}")
    print(f"  Feature observations: {len(features)}")

    prices_aligned = prices.reindex(features.index)

    print("\n" + "=" * 60)
    print("PHASE 2: REGIME DETECTION")
    print("=" * 60)

    detector = RegimeDetector(
        n_states=n_hmm_states,
        n_iter=200,
        min_seq_len=hmm_min_seq_len,
        vol_feature_idx=1,
    )
    detector.fit(features.values)

    regimes = pd.Series(
        detector.predict_sequence(features.values),
        index=features.index,
        name="regime",
    )
    regime_dist = np.bincount(regimes.values, minlength=n_hmm_states) / len(regimes)

    print("  Canonical state descriptions:")
    for label, desc in detector.state_descriptions.items():
        print(f"    {desc}  |  frequency: {regime_dist[label]:.1%}")

    print("\n" + "=" * 60)
    print("PHASE 3: PRIMARY SIGNAL + TRIPLE BARRIER LABELLING")
    print("=" * 60)

    fast_ma_series = prices_aligned.rolling(fast_ma).mean()
    slow_ma_series = prices_aligned.rolling(slow_ma).mean()
    primary_signal = (fast_ma_series > slow_ma_series) & (
        fast_ma_series.shift(1) <= slow_ma_series.shift(1)
    )
    print(f"  Primary signal entries: {primary_signal.sum()}")

    labeller = TripleBarrierLabeller(
        vol_multiplier_upper=vol_mult_upper,
        vol_multiplier_lower=vol_mult_lower,
        max_holding_bars=max_holding_bars,
    )
    labels = labeller.label(
        prices=prices_aligned,
        volatility=features["ewm_vol"],
        entry_signals=primary_signal,
        regimes=regimes,
    )

    label_counts = labels["label"].value_counts().sort_index()
    print(f"  Label distribution: {label_counts.to_dict()}")

    meta_y = (labels["label"] == 1).astype(int)
    print(f"  Meta-label positive rate: {meta_y.mean():.1%}")

    print("\n" + "=" * 60)
    print("PHASE 4: FEATURE ASSEMBLY FOR META-LABELLER")
    print("=" * 60)

    X_base = features.reindex(labels.index)[["log_return", "ewm_vol", "frac_diff"]].copy()

    # Compute regime posteriors at each entry point
    feature_arr = features.values
    posteriors_list = []
    for entry_time in labels.index:
        arr_idx = features.index.get_loc(entry_time)
        window_start = max(0, arr_idx - hmm_min_seq_len + 1)
        window = feature_arr[window_start : arr_idx + 1]
        if len(window) >= hmm_min_seq_len:
            _, post = detector.predict_current_state(window)
        else:
            post = regime_dist.copy()  # fall back to training marginal
        posteriors_list.append(post)

    post_df = pd.DataFrame(
        np.vstack(posteriors_list),
        index=labels.index,
        columns=[f"regime_prob_{i}" for i in range(n_hmm_states)],
    )
    X = pd.concat([X_base, post_df], axis=1)

    # Exit times for purging
    exit_times = pd.Series(prices_aligned.index[labels["exit_idx"].values], index=labels.index)

    print(f"  Final feature matrix: {X.shape}")

    # Sample uniqueness weights
    uniqueness = compute_sample_uniqueness(labels.index, exit_times, prices_aligned.index)
    print(f"  Mean sample uniqueness: {uniqueness.mean():.3f}")

    print("\n" + "=" * 60)
    print("PHASE 5: META-LABELLER TRAINING")
    print("=" * 60)

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    purge_counts = cv.n_purged_per_fold(X, exit_times)
    print(f"  Purged per fold: {purge_counts}")

    ml = MetaLabeller(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        threshold_target="f1",
        use_sample_weights=True,
    )
    ml.fit(X, meta_y, exit_times, bar_times=prices_aligned.index)

    print("  CV results:")
    for k, v in ml.cv_results.items():
        print(f"    {k}: {v}")

    print("\n  Top features:")
    print(ml.feature_importance().head(5).to_string())

    print("\n" + "=" * 60)
    print("PHASE 6: SIGNAL GENERATION + VECTORBT BACKTEST")
    print("=" * 60)

    probs = ml.predict_proba(X)
    meta_preds = ml.predict(X)
    filtered_labels = labels[meta_preds == 1]

    print(f"  Signals after meta-filter: {meta_preds.sum()} / {len(meta_preds)}")

    entry_series = pd.Series(False, index=prices_aligned.index)
    exit_series = pd.Series(False, index=prices_aligned.index)

    for entry_time, row in filtered_labels.iterrows():
        entry_series.loc[entry_time] = True
        exit_time = prices_aligned.index[int(row["exit_idx"])]
        exit_series.loc[exit_time] = True

    portfolio = vbt.Portfolio.from_signals(
        prices_aligned,
        entries=entry_series,
        exits=exit_series,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )

    stats = portfolio.stats()
    print("\n  Backtest summary:")
    print(stats.to_string())

    print("\n" + "=" * 60)
    print("PHASE 7: SAVING MODELS")
    print("=" * 60)

    hmm_path = str(output_path / "hmm_regime.pkl")
    xgb_path = str(output_path / "xgb_meta.pkl")
    detector.save(hmm_path)
    ml.save(xgb_path)
    print(f"  Saved HMM -> {hmm_path}")
    print(f"  Saved XGB -> {xgb_path}")

    # Compute OOF prob stats for health monitor initialisation
    oof_probs_series = pd.Series(probs)
    training_metadata = {
        "hmm_model_path": hmm_path,
        "xgb_model_path": xgb_path,
        "training_regime_dist": regime_dist.tolist(),
        "training_prob_mean": float(oof_probs_series.mean()),
        "training_prob_std": float(oof_probs_series.std()),
        "feature_engine_params": {
            "vol_span": vol_span,
            "frac_d": frac_d,
            "frac_tau": frac_tau,
        },
        "barrier_params": {
            "vol_multiplier_upper": vol_mult_upper,
            "vol_multiplier_lower": vol_mult_lower,
            "max_holding_bars": max_holding_bars,
        },
        "optimal_threshold": ml.optimal_threshold,
    }

    print("\n  Training metadata for Nautilus config:")
    for k, v in training_metadata.items():
        print(f"    {k}: {v}")

    return {
        "portfolio": portfolio,
        "stats": stats,
        "feature_engine": fe,
        "regime_detector": detector,
        "meta_labeller": ml,
        "labeller": labeller,
        "features": features,
        "labels": labels,
        "training_metadata": training_metadata,
    }
