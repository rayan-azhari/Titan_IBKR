"""
run_pipeline.py

ML Regime Strategy research pipeline with IS/OOS split.

Usage:
    python research/ml_regime/run_pipeline.py --instrument QQQ --timeframe D
    python research/ml_regime/run_pipeline.py --instrument QQQ --timeframe D --fast_ma 5 --slow_ma 20
    python research/ml_regime/run_pipeline.py --instrument EUR_USD --timeframe H1

Key lesson from MTF experiment: use the SAME exit logic in training labels and live
execution. Labels use MA crossover-back exit; OOS backtest uses the same exit.
This gives the MetaLabeller coherent training targets.

Stages:
  1. Load data/{INSTRUMENT}_{TIMEFRAME}.parquet
  2. 70/30 IS/OOS split
  3. FeatureEngine.batch(full_close) -> IS / OOS slices
  4. RegimeDetector.fit(IS features) -> predict_sequence(full)
  5. label_ma_trades() on IS: entry=crossover, exit=crossover-back, label=win/loss
  6. Build X: features + regime posteriors + ma_spread at IS entry times
  7. MetaLabeller.fit(X, meta_y, exit_times) with purged CV
  8. OOS: raw MA baseline vs ML-filtered MA (both using crossover-back exits)
  9. Save models to models/ml_regime/
 10. Report vs B&H benchmark
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ml_regime.feature_engine import FeatureEngine
from research.ml_regime.meta_labeller import MetaLabeller
from research.ml_regime.regime_detection import RegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Timeframe presets ─────────────────────────────────────────────────────────
TIMEFRAME_PRESETS = {
    "D": {
        "vol_span": 20,
        "hmm_min_seq": 60,
        "fast_ma": 5,       # faster MAs -> ~200 IS events (vs 77 with 10/30)
        "slow_ma": 20,
        "fees": 0.001,
        "slippage": 0.001,
        "vbt_freq": "1D",
    },
    "H4": {
        "vol_span": 30,
        "hmm_min_seq": 120,
        "fast_ma": 6,
        "slow_ma": 30,
        "fees": 0.0001,
        "slippage": 0.00005,
        "vbt_freq": "4h",
    },
    "H1": {
        "vol_span": 48,
        "hmm_min_seq": 168,
        "fast_ma": 24,
        "slow_ma": 120,
        "fees": 0.0001,
        "slippage": 0.00005,
        "vbt_freq": "1h",
    },
}

FRAC_D = 0.4
N_HMM_STATES = 2
N_CV_SPLITS = 5
EMBARGO_PCT = 0.01
IS_FRAC = 0.70


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_close(instrument: str, timeframe: str) -> pd.Series:
    """Load close series; handles both DatetimeIndex and timestamp-column formats."""
    path = ROOT / "data" / f"{instrument}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            raise ValueError(f"Cannot parse index for {path}.")
    col_lower = {c.lower(): c for c in df.columns}
    close_col = col_lower.get("close")
    if close_col is None:
        raise ValueError(f"No 'close' column in {path}.")
    close = df[close_col].dropna().sort_index()
    logger.info(
        f"Loaded {len(close)} bars for {instrument} {timeframe} "
        f"({close.index[0].date()} - {close.index[-1].date()})"
    )
    return close


def label_ma_trades(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    """
    Label MA crossover trades using MA-flip exit (same rule used in live trading).

    Entry: fast MA crosses above slow MA. Signal bar t -> execute bar t+1.
    Exit:  first bar after entry where fast MA crosses back below slow MA.
    Label: 1 if trade return > 0, else 0.

    Returns DataFrame indexed by entry_time: [label, exit_time, trade_return].
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    fast_above = fast_ma > slow_ma

    entry_cross = fast_above & ~fast_above.shift(1).fillna(False)
    entry_signal = entry_cross.shift(1).fillna(False)   # execute next bar
    exit_cond = ~fast_above                              # fast back below slow

    entry_locs = np.where(entry_signal.values)[0]
    records = []

    for ei in entry_locs:
        entry_t = close.index[ei]
        entry_p = float(close.iloc[ei])

        below_after = exit_cond.iloc[ei + 1:]
        hit = np.where(below_after.values)[0]
        xi = ei + 1 + int(hit[0]) if len(hit) > 0 else len(close) - 1

        exit_t = close.index[xi]
        ret = float(close.iloc[xi]) / entry_p - 1.0
        records.append(
            {"entry_time": entry_t, "exit_time": exit_t,
             "label": int(ret > 0), "trade_return": ret}
        )

    if not records:
        return pd.DataFrame(columns=["exit_time", "label", "trade_return"])
    return pd.DataFrame(records).set_index("entry_time")


def ma_exit_series(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Boolean exit signal: fast MA crosses below slow MA (shifted +1)."""
    fast_above = close.rolling(fast).mean() > close.rolling(slow).mean()
    crossed_below = ~fast_above & fast_above.shift(1).fillna(False)
    return crossed_below.shift(1).fillna(False)


def ma_entry_series(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Boolean entry signal: fast MA crosses above slow MA (shifted +1)."""
    fast_above = close.rolling(fast).mean() > close.rolling(slow).mean()
    crossed_above = fast_above & ~fast_above.shift(1).fillna(False)
    return crossed_above.shift(1).fillna(False)


def ma_spread_feature(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Normalised MA spread: (fast_ma - slow_ma) / slow_ma."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    return ((fast_ma - slow_ma) / slow_ma).rename("ma_spread")


def _canonical_posteriors(detector: RegimeDetector, arr: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = detector._model.predict_proba(detector._scaler.transform(arr))
    return raw[:, detector._canonical_to_raw]


def build_all_posteriors(
    detector: RegimeDetector, full_arr: np.ndarray, is_cutoff: int, hmm_min_seq: int
) -> np.ndarray:
    is_post = _canonical_posteriors(detector, full_arr[:is_cutoff])
    oos = full_arr[is_cutoff:]
    if len(oos) == 0:
        return is_post
    warmup = full_arr[max(0, is_cutoff - hmm_min_seq):is_cutoff]
    combined = _canonical_posteriors(detector, np.vstack([warmup, oos]))
    return np.vstack([is_post, combined[len(warmup):]])


def _run_pf(close, entries, exits, fees, slippage, freq) -> vbt.Portfolio:
    return vbt.Portfolio.from_signals(
        close, entries=entries, exits=exits,
        fees=fees, slippage=slippage, freq=freq,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_pipeline(
    instrument: str,
    timeframe: str = "D",
    is_frac: float = IS_FRAC,
    fast_ma: int | None = None,
    slow_ma: int | None = None,
) -> dict:

    preset = TIMEFRAME_PRESETS.get(timeframe.upper())
    if preset is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Choose: {list(TIMEFRAME_PRESETS)}")

    vol_span   = preset["vol_span"]
    hmm_min_seq = preset["hmm_min_seq"]
    fast       = fast_ma if fast_ma is not None else preset["fast_ma"]
    slow       = slow_ma if slow_ma is not None else preset["slow_ma"]
    fees       = preset["fees"]
    slippage   = preset["slippage"]
    vbt_freq   = preset["vbt_freq"]

    logger.info(f"Timeframe={timeframe} | fast_ma={fast}, slow_ma={slow}, hmm_min_seq={hmm_min_seq}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    close = load_close(instrument, timeframe)

    # ── 2. IS/OOS split ───────────────────────────────────────────────────────
    is_cutoff_price = int(len(close) * is_frac)
    is_close  = close.iloc[:is_cutoff_price]
    oos_close = close.iloc[is_cutoff_price:]
    logger.info(f"IS:  {len(is_close)} bars ({is_close.index[0].date()} - {is_close.index[-1].date()})")
    logger.info(f"OOS: {len(oos_close)} bars ({oos_close.index[0].date()} - {oos_close.index[-1].date()})")

    # ── 3. Features ───────────────────────────────────────────────────────────
    logger.info("Phase 1: Feature engineering...")
    engine = FeatureEngine(vol_span=vol_span, frac_d=FRAC_D)
    full_feat_df = engine.batch(close).dropna()

    is_cutoff_feat = int(full_feat_df.index.searchsorted(is_close.index[-1], side="right"))
    is_feat_df  = full_feat_df.iloc[:is_cutoff_feat]
    oos_feat_df = full_feat_df.iloc[is_cutoff_feat:]
    full_feat_arr = full_feat_df.values
    logger.info(f"Features: {len(full_feat_df)} bars. IS={len(is_feat_df)}, OOS={len(oos_feat_df)}")

    # ── 4. HMM regime detection ───────────────────────────────────────────────
    logger.info("Phase 2: HMM regime detection (fit IS only)...")
    detector = RegimeDetector(n_states=N_HMM_STATES, min_seq_len=hmm_min_seq)
    detector.fit(is_feat_df.values)
    for s, desc in detector.state_descriptions.items():
        logger.info(f"  {desc}")

    full_regimes = detector.predict_sequence(full_feat_arr)
    regime_dist  = np.bincount(full_regimes[:is_cutoff_feat], minlength=N_HMM_STATES) / is_cutoff_feat
    logger.info(f"  IS regime dist: {np.round(regime_dist, 3).tolist()}")

    all_posteriors = build_all_posteriors(detector, full_feat_arr, is_cutoff_feat, hmm_min_seq)
    posteriors_df  = pd.DataFrame(
        all_posteriors, index=full_feat_df.index,
        columns=[f"regime_prob_{i}" for i in range(N_HMM_STATES)],
    )
    spread_feat = ma_spread_feature(close, fast, slow).reindex(full_feat_df.index)

    # ── 5. IS trade labelling (MA-flip exits) ─────────────────────────────────
    logger.info("Phase 3: Labelling IS trades with MA-flip exits...")
    is_labels = label_ma_trades(is_close, fast, slow)
    # keep only entries where features are available
    is_labels = is_labels[is_labels.index.isin(is_feat_df.index)]

    if len(is_labels) == 0:
        raise RuntimeError(
            f"No IS trades generated. Check MA params (fast={fast}, slow={slow})."
        )

    n_win = int(is_labels["label"].sum())
    logger.info(
        f"  IS trades: {len(is_labels)} | wins={n_win} ({n_win/len(is_labels):.1%}) "
        f"| mean return={is_labels['trade_return'].mean():.4f}"
    )

    # ── 6. MetaLabeller features ──────────────────────────────────────────────
    logger.info("Phase 4: Building MetaLabeller features...")
    is_entry_times = is_labels.index
    X = pd.concat(
        [
            full_feat_df.reindex(is_entry_times),
            posteriors_df.reindex(is_entry_times),
            spread_feat.reindex(is_entry_times),
        ],
        axis=1,
    ).dropna()
    meta_y     = is_labels["label"].reindex(X.index).dropna().astype(int)
    exit_times = is_labels.loc[X.index, "exit_time"]

    logger.info(f"  X: {X.shape}, cols: {X.columns.tolist()}")
    logger.info(f"  Positive rate: {meta_y.mean():.2%}")

    # ── 7. Fit MetaLabeller ───────────────────────────────────────────────────
    logger.info("Phase 5: Training MetaLabeller with purged CV...")
    ml = MetaLabeller(n_splits=N_CV_SPLITS, embargo_pct=EMBARGO_PCT, threshold_target="f1")
    ml.fit(X, meta_y, exit_times, bar_times=is_feat_df.index)

    cv = ml.cv_results
    logger.info(f"  Mean CV AUC: {cv.get('mean_auc', float('nan')):.4f} +/- {cv.get('std_auc', float('nan')):.4f}")
    logger.info(f"  OOF AUC:   {cv.get('oof_auc')}")
    logger.info(f"  OOF F1:    {cv.get('oof_f1', 0.0):.4f}")
    logger.info(f"  Threshold: {cv.get('optimal_threshold'):.4f}")
    logger.info(f"  Top-5 features:\n{ml.feature_importance().head(5).to_string()}")

    # ── 8. OOS backtest: raw MA vs ML-filtered ────────────────────────────────
    logger.info("Phase 6: OOS backtest (raw MA vs ML-filtered)...")

    oos_entries_all = ma_entry_series(close, fast, slow).reindex(oos_close.index).fillna(False)
    oos_exits_all   = ma_exit_series(close, fast, slow).reindex(oos_close.index).fillna(False)

    # Variant 1: raw MA strategy
    raw_pf = _run_pf(oos_close, oos_entries_all, oos_exits_all, fees, slippage, vbt_freq)

    # Variant 2: ML filter applied to crossover entry bars
    oos_entry_times = oos_entries_all[oos_entries_all & oos_entries_all.index.isin(oos_feat_df.index)].index
    logger.info(f"  OOS crossover events: {len(oos_entry_times)}")

    ml_pf = None
    n_ml  = 0
    if len(oos_entry_times) > 0:
        X_oos = pd.concat(
            [
                full_feat_df.reindex(oos_entry_times),
                posteriors_df.reindex(oos_entry_times),
                spread_feat.reindex(oos_entry_times),
            ],
            axis=1,
        ).dropna()
        if len(X_oos) > 0:
            approved = X_oos.index[ml.predict(X_oos).astype(bool)]
            n_ml = len(approved)
            logger.info(f"  ML-approved: {n_ml} ({n_ml/max(len(oos_entry_times),1):.1%} pass rate)")
            if n_ml > 0:
                ml_entries = pd.Series(False, index=oos_close.index)
                ml_entries.loc[approved] = True
                ml_pf = _run_pf(oos_close, ml_entries, oos_exits_all, fees, slippage, vbt_freq)

    # IS Sharpe on raw MA (for OOS/IS ratio)
    is_entries_s = ma_entry_series(is_close, fast, slow)
    is_exits_s   = ma_exit_series(is_close, fast, slow)
    is_pf = _run_pf(is_close, is_entries_s, is_exits_s, fees, slippage, vbt_freq)
    is_sharpe = float(is_pf.sharpe_ratio())

    # B&H OOS
    bh_e = pd.Series(False, index=oos_close.index)
    bh_e.iloc[0] = True
    bh_pf     = _run_pf(oos_close, bh_e, pd.Series(False, index=oos_close.index), fees, slippage, vbt_freq)
    bh_sharpe = float(bh_pf.sharpe_ratio())
    bh_return = float(bh_pf.total_return())

    # ── 9. Save models ─────────────────────────────────────────────────────────
    model_dir = ROOT / "models" / "ml_regime"
    model_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{instrument.lower()}_{timeframe.lower()}"
    detector.save(str(model_dir / f"hmm_{slug}.pkl"))
    ml.save(str(model_dir / f"xgb_{slug}.pkl"))
    logger.info(f"Models saved: models/ml_regime/hmm_{slug}.pkl + xgb_{slug}.pkl")

    # ── 10. Report ─────────────────────────────────────────────────────────────
    def _stats(pf, label):
        n = int(pf.trades.count())
        return {
            "variant": label,
            "sharpe": round(float(pf.sharpe_ratio()), 3),
            "total_return": round(float(pf.total_return()), 4),
            "max_drawdown": round(float(pf.max_drawdown()), 4),
            "n_trades": n,
            "win_rate": round(float(pf.trades.win_rate()), 3) if n > 0 else None,
        }

    rows = [_stats(raw_pf, "Raw MA (no filter)")]
    if ml_pf is not None:
        rows.append(_stats(ml_pf, "MA + ML filter"))

    results_df = pd.DataFrame(rows)
    raw_sharpe = rows[0]["sharpe"]
    ml_sharpe  = rows[1]["sharpe"] if ml_pf is not None else None
    ratio      = (ml_sharpe / is_sharpe) if (ml_sharpe and is_sharpe > 0) else None

    print("\n" + "=" * 70)
    print(f"ML REGIME STRATEGY — {instrument} {timeframe} (fast={fast}, slow={slow})")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n  IS Sharpe (raw MA):        {is_sharpe:.3f}")
    print(f"  OOS/IS Ratio (ML/IS raw):  {ratio:.3f}" if ratio else "  OOS/IS Ratio:  N/A")
    print(f"  --- B&H {instrument} OOS ---")
    print(f"  B&H Sharpe: {bh_sharpe:.3f}  |  B&H Return: {bh_return:.2%}")
    print("  --- Model Quality ---")
    print(f"  OOF AUC:   {cv.get('oof_auc')}")
    print(f"  OOF F1:    {cv.get('oof_f1', 0.0):.4f}")
    print(f"  Threshold: {cv.get('optimal_threshold'):.4f}")
    print("=" * 70)

    if ratio is not None and ratio < 0.5:
        print("\n[WARNING] OOS/IS < 0.5 — possible overfit.")
    if cv.get("oof_auc") is not None and cv["oof_auc"] < 0.52:
        print("\n[NOTE] OOF AUC near 0.5 — ML adds marginal signal.")
    if ml_sharpe is not None and ml_sharpe < raw_sharpe:
        print(f"\n[NOTE] ML filter ({ml_sharpe:.3f}) below raw MA ({raw_sharpe:.3f}) — filter is not adding value.")
    if ml_sharpe is not None and ml_sharpe < bh_sharpe:
        print(f"\n[NOTE] ML Sharpe ({ml_sharpe:.3f}) below B&H ({bh_sharpe:.3f}).")

    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_dir / f"ml_regime_{slug}.csv", index=False)
    logger.info(f"Report saved: .tmp/reports/ml_regime_{slug}.csv")

    return {"variants": rows, "cv": cv, "is_sharpe": is_sharpe, "bh_sharpe": bh_sharpe}


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ML Regime Strategy Research Pipeline")
    parser.add_argument("--instrument", default="QQQ")
    parser.add_argument("--timeframe", default="D", choices=list(TIMEFRAME_PRESETS))
    parser.add_argument("--is_frac", type=float, default=IS_FRAC)
    parser.add_argument("--fast_ma", type=int, default=None, help="Override preset fast MA")
    parser.add_argument("--slow_ma", type=int, default=None, help="Override preset slow MA")
    args = parser.parse_args()
    run_pipeline(
        instrument=args.instrument,
        timeframe=args.timeframe,
        is_frac=args.is_frac,
        fast_ma=args.fast_ma,
        slow_ma=args.slow_ma,
    )


if __name__ == "__main__":
    main()
