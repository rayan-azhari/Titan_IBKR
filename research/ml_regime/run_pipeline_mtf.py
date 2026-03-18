"""
run_pipeline_mtf.py

ML Regime Strategy — MTF Confluence Edition.

Uses the proven MTF confluence signal (H1/H4/D/W) as the primary entry signal.
Labels trades using the SAME exit logic as the live MTF strategy (confluence flip
to negative), not triple barrier. This ensures the MetaLabeller is trained on the
exact game it is asked to play in production.

Architecture:
  MTF confluence crossover >= threshold
    -> MetaLabeller predicts win/loss before entry
    -> Exit: confluence < 0  (same as raw MTF)

OOS comparison (two variants):
  1. Raw MTF long/short (baseline — identical to run_backtest_mtf.py)
  2. MTF long + MetaLabeller filter (regime is a feature, not a hard gate)

Usage:
    python research/ml_regime/run_pipeline_mtf.py
    python research/ml_regime/run_pipeline_mtf.py --pair EUR_USD --is_frac 0.7

Config read from: config/mtf.toml  (read-only — never modified)
Models saved to:  models/ml_regime/hmm_{pair}_h4_mtf.pkl
                  models/ml_regime/xgb_{pair}_h4_mtf.pkl
"""

import argparse
import logging
import sys
import tomllib
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

# ── Fixed hyperparameters (H4 frequency) ─────────────────────────────────────
VOL_SPAN = 30           # ~5-day EWM vol on H4
FRAC_D = 0.4
N_HMM_STATES = 2
HMM_MIN_SEQ = 120       # ~20-day warm-up
N_CV_SPLITS = 5
EMBARGO_PCT = 0.01
FEES = 0.0001           # ~1 pip per side
SLIPPAGE = 0.00005
VBT_FREQ = "4h"
IS_FRAC = 0.70

TIMEFRAMES = ["H1", "H4", "D", "W"]


# ── MTF signal computation ────────────────────────────────────────────────────


def _load_tf_data(pair: str, granularity: str) -> pd.DataFrame | None:
    path = ROOT / "data" / f"{pair}_{granularity}.parquet"
    if not path.exists():
        logger.warning(f"Missing {path.name} — skipping {granularity}.")
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index()


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    """Directional signal for one timeframe in [-1, +1]."""
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = _compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def compute_mtf_confluence(
    pair: str, mtf_config: dict
) -> tuple[pd.Series, pd.DataFrame, dict[str, pd.Series]]:
    """
    Weighted MTF confluence score on the H4 primary index. Range [-1, +1].

    Returns:
        confluence  -- weighted sum across all timeframes
        h4_df       -- H4 OHLCV DataFrame (primary execution timeframe)
        tf_signals  -- dict of raw (unweighted) per-TF signals, keyed by TF label
                       Each signal is in [-1, +1]; used for H1 alignment filter.
    """
    weights = mtf_config.get("weights", {})
    h4_df = _load_tf_data(pair, "H4")
    if h4_df is None:
        raise FileNotFoundError(f"H4 data required for {pair}.")

    primary_index = h4_df.index
    weighted: list[pd.Series] = []
    tf_signals: dict[str, pd.Series] = {}
    total_weight = 0.0

    for tf in TIMEFRAMES:
        w = float(weights.get(tf, 0.0))
        if w == 0.0:
            continue
        tf_cfg = mtf_config.get(tf, {})
        df = _load_tf_data(pair, tf)
        if df is None:
            continue
        sig = _compute_tf_signal(
            df["close"],
            tf_cfg.get("fast_ma", 20),
            tf_cfg.get("slow_ma", 50),
            tf_cfg.get("rsi_period", 14),
        )
        aligned_sig = sig.reindex(primary_index, method="ffill")
        tf_signals[tf] = aligned_sig          # raw signal, unweighted
        weighted.append(aligned_sig * w)
        total_weight += w
        logger.info(f"  {tf} (w={w:.2f}): bullish {(sig > 0).mean():.1%} | {len(df)} bars")

    if not weighted:
        raise RuntimeError(f"No signals computed for {pair}.")
    confluence = pd.concat(weighted, axis=1).sum(axis=1)
    if 0.0 < total_weight < 1.0:
        confluence = confluence / total_weight
    return confluence, h4_df, tf_signals


# ── MTF trade labelling ───────────────────────────────────────────────────────


def label_mtf_trades(
    close: pd.Series,
    confluence: pd.Series,
    threshold: float,
    direction: str = "long",
    h1_signal: pd.Series | None = None,
    h1_min_score: float = -2.0,
    exit_buffer: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Label MTF trades using confluence-flip exit logic.

    Entry: confluence crosses threshold (bar t signal -> bar t+1 execution).
    Exit:  confluence crosses -exit_buffer (long) / +exit_buffer (short).
           exit_buffer=0.0 is the original behaviour (exit at zero crossing).
           exit_buffer>0.0 adds hysteresis: must drop further before exiting.

    H1 filter (optional):
        If h1_signal is provided, entry is skipped when h1_signal < h1_min_score.
        This prevents entering against a strong H1 counter-trend move.
        h1_min_score=-1.0 (fully bearish H1) disables the filter effectively.
        h1_min_score=0.0 requires H1 to be at least neutral.

    Also counts signals that fired while a trade was already open (blocked events).

    Returns:
        trades_df  -- DataFrame indexed by entry_time: [label, exit_time, trade_return]
        stats      -- dict with blocked_in_trade, h1_filtered counts
    """
    conf = confluence.reindex(close.index, method="ffill").fillna(0.0)
    h1_aligned: pd.Series | None = None
    if h1_signal is not None:
        h1_aligned = h1_signal.reindex(close.index, method="ffill").fillna(0.0)

    if direction == "long":
        above = conf >= threshold
        entry_cross = above & ~above.shift(1).fillna(False)
        adverse = conf < -exit_buffer
    else:
        below = conf <= -threshold
        entry_cross = below & ~below.shift(1).fillna(False)
        adverse = conf > exit_buffer

    entry_signal = entry_cross.shift(1).fillna(False)
    all_signal_locs = np.where(entry_signal.values)[0]

    records = []
    n_blocked_in_trade = 0
    n_h1_filtered = 0
    in_trade_until = -1  # bar index until which we are in a trade

    for ei in all_signal_locs:
        # Count signals that fire while a previous trade is still open
        if ei <= in_trade_until:
            n_blocked_in_trade += 1
            continue

        # H1 alignment filter
        if h1_aligned is not None and float(h1_aligned.iloc[ei]) < h1_min_score:
            n_h1_filtered += 1
            continue

        entry_t = close.index[ei]
        entry_p = float(close.iloc[ei])

        adverse_after = adverse.iloc[ei + 1:]
        hit = np.where(adverse_after.values)[0]
        xi = ei + 1 + int(hit[0]) if len(hit) > 0 else len(close) - 1
        in_trade_until = xi

        exit_t = close.index[xi]
        exit_p = float(close.iloc[xi])
        ret = (exit_p / entry_p - 1.0) if direction == "long" else (entry_p / exit_p - 1.0)

        records.append({"entry_time": entry_t, "exit_time": exit_t,
                        "label": int(ret > 0), "trade_return": ret})

    stats = {
        "total_signals": len(all_signal_locs),
        "blocked_in_trade": n_blocked_in_trade,
        "h1_filtered": n_h1_filtered,
        "entered": len(records),
    }

    if not records:
        return pd.DataFrame(columns=["exit_time", "label", "trade_return"]), stats
    return pd.DataFrame(records).set_index("entry_time"), stats


# ── Regime helpers ────────────────────────────────────────────────────────────


def _canonical_posteriors(detector: RegimeDetector, features_arr: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = detector._scaler.transform(features_arr)
        raw = detector._model.predict_proba(X)
    return raw[:, detector._canonical_to_raw]


def build_all_posteriors(
    detector: RegimeDetector, full_arr: np.ndarray, is_cutoff: int
) -> np.ndarray:
    is_post = _canonical_posteriors(detector, full_arr[:is_cutoff])
    oos = full_arr[is_cutoff:]
    if len(oos) == 0:
        return is_post
    warmup = full_arr[max(0, is_cutoff - HMM_MIN_SEQ):is_cutoff]
    combined = _canonical_posteriors(detector, np.vstack([warmup, oos]))
    return np.vstack([is_post, combined[len(warmup):]])


# ── VBT helpers ──────────────────────────────────────────────────────────────


def _run_pf(close, entries, exits, short=False) -> vbt.Portfolio:
    if short:
        return vbt.Portfolio.from_signals(
            close,
            entries=pd.Series(False, index=close.index),
            exits=pd.Series(False, index=close.index),
            short_entries=entries,
            short_exits=exits,
            fees=FEES, slippage=SLIPPAGE, freq=VBT_FREQ,
        )
    return vbt.Portfolio.from_signals(
        close, entries=entries, exits=exits,
        fees=FEES, slippage=SLIPPAGE, freq=VBT_FREQ,
    )


def _pf_stats(pf, label: str) -> dict:
    n = int(pf.trades.count())
    return {
        "variant": label,
        "sharpe": round(float(pf.sharpe_ratio()), 3),
        "total_return": round(float(pf.total_return()), 4),
        "max_drawdown": round(float(pf.max_drawdown()), 4),
        "n_trades": n,
        "win_rate": round(float(pf.trades.win_rate()), 3) if n > 0 else None,
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_pipeline_mtf(pair: str = "EUR_USD", is_frac: float = IS_FRAC) -> dict:

    # ── Config ────────────────────────────────────────────────────────────────
    with open(ROOT / "config" / "mtf.toml", "rb") as f:
        mtf_config = tomllib.load(f)
    threshold = mtf_config.get("confirmation_threshold", 0.30)
    logger.info(f"MTF config: threshold={threshold}, weights={mtf_config.get('weights')}")

    # ── 1. MTF confluence ─────────────────────────────────────────────────────
    logger.info(f"Phase 1: Computing MTF confluence for {pair} H4...")
    confluence, h4_df, tf_signals = compute_mtf_confluence(pair, mtf_config)
    close = h4_df["close"]
    h1_sig = tf_signals.get("H1")

    # ── 2. IS/OOS split ───────────────────────────────────────────────────────
    n = len(close)
    is_cutoff_price = int(n * is_frac)
    is_close = close.iloc[:is_cutoff_price]
    oos_close = close.iloc[is_cutoff_price:]
    is_conf = confluence.iloc[:is_cutoff_price]

    logger.info(
        f"IS: {len(is_close)} bars "
        f"({is_close.index[0].date()} - {is_close.index[-1].date()})"
    )
    logger.info(
        f"OOS: {len(oos_close)} bars "
        f"({oos_close.index[0].date()} - {oos_close.index[-1].date()})"
    )

    # ── 3. Features ───────────────────────────────────────────────────────────
    logger.info("Phase 2: Feature engineering (H4)...")
    engine = FeatureEngine(vol_span=VOL_SPAN, frac_d=FRAC_D)
    full_feat_df = engine.batch(close).dropna()

    is_boundary = is_close.index[-1]
    is_cutoff_feat = int(full_feat_df.index.searchsorted(is_boundary, side="right"))
    is_feat_df = full_feat_df.iloc[:is_cutoff_feat]
    oos_feat_df = full_feat_df.iloc[is_cutoff_feat:]
    full_feat_arr = full_feat_df.values
    logger.info(
        f"Features: {len(full_feat_df)} H4 bars. IS={len(is_feat_df)}, OOS={len(oos_feat_df)}"
    )

    # ── 4. HMM regime detection ───────────────────────────────────────────────
    logger.info("Phase 3: HMM regime detection (fit IS only)...")
    detector = RegimeDetector(n_states=N_HMM_STATES, min_seq_len=HMM_MIN_SEQ)
    detector.fit(is_feat_df.values)
    for state, desc in detector.state_descriptions.items():
        logger.info(f"  {desc}")

    full_regimes = detector.predict_sequence(full_feat_arr)
    regime_dist = (
        np.bincount(full_regimes[:is_cutoff_feat], minlength=N_HMM_STATES)
        / is_cutoff_feat
    )
    logger.info(f"  IS regime dist: {np.round(regime_dist, 3).tolist()}")

    all_posteriors = build_all_posteriors(detector, full_feat_arr, is_cutoff_feat)
    posteriors_df = pd.DataFrame(
        all_posteriors,
        index=full_feat_df.index,
        columns=[f"regime_prob_{i}" for i in range(N_HMM_STATES)],
    )
    conf_feat = confluence.reindex(full_feat_df.index, method="ffill").rename("confluence")

    # ── 5. IS trade labelling (MTF native exits) ──────────────────────────────
    logger.info("Phase 4: Labelling IS trades using MTF confluence-flip exits...")
    is_labels, is_stats = label_mtf_trades(is_close, is_conf, threshold, direction="long")

    if len(is_labels) == 0:
        raise RuntimeError("No IS trades generated. Check threshold and IS data.")

    n_win = is_labels["label"].sum()
    logger.info(
        f"  IS long trades: {len(is_labels)} | wins={n_win} ({n_win/len(is_labels):.1%}) "
        f"| mean return={is_labels['trade_return'].mean():.4f}"
    )
    logger.info(
        f"  IS signal stats: total={is_stats['total_signals']}, "
        f"blocked_in_trade={is_stats['blocked_in_trade']}, "
        f"entered={is_stats['entered']}"
    )

    # ── 6. MetaLabeller feature matrix ────────────────────────────────────────
    logger.info("Phase 5: Building MetaLabeller features at IS entry times...")
    is_entry_times = is_labels.index
    X = pd.concat(
        [
            full_feat_df.reindex(is_entry_times),
            posteriors_df.reindex(is_entry_times),
            conf_feat.reindex(is_entry_times),
        ],
        axis=1,
    ).dropna()
    meta_y = is_labels["label"].reindex(X.index).dropna().astype(int)

    # exit_times for PurgedKFold: actual confluence-flip exit datetimes
    exit_times = is_labels.loc[X.index, "exit_time"]

    logger.info(f"  MetaLabeller X: {X.shape}, columns: {X.columns.tolist()}")
    logger.info(f"  Positive rate in training set: {meta_y.mean():.2%}")

    # ── 7. Fit MetaLabeller ───────────────────────────────────────────────────
    logger.info("Phase 6: Training MetaLabeller with purged CV...")
    ml = MetaLabeller(n_splits=N_CV_SPLITS, embargo_pct=EMBARGO_PCT, threshold_target="f1")
    ml.fit(X, meta_y, exit_times, bar_times=is_feat_df.index)

    cv = ml.cv_results
    logger.info(
        f"  Mean CV AUC: {cv.get('mean_auc', float('nan')):.4f} "
        f"+/- {cv.get('std_auc', float('nan')):.4f}"
    )
    logger.info(f"  OOF AUC:           {cv.get('oof_auc')}")
    logger.info(f"  OOF F1:            {cv.get('oof_f1', 0.0):.4f}")
    logger.info(f"  Optimal threshold: {cv.get('optimal_threshold'):.4f}")
    logger.info(f"  Top-5 features:\n{ml.feature_importance().head(5).to_string()}")

    # ── 8. OOS backtest — two-way comparison ─────────────────────────────────
    logger.info("Phase 7: OOS backtest...")

    # Raw MTF exits: confluence sign flip
    oos_long_exit = (confluence < 0).reindex(oos_close.index).fillna(False)
    oos_short_exit = (confluence > 0).reindex(oos_close.index).fillna(False)

    # Raw MTF entries (level, shifted +1)
    oos_long_raw = (confluence >= threshold).shift(1).fillna(False).reindex(oos_close.index).fillna(False)
    oos_short_raw = (confluence <= -threshold).shift(1).fillna(False).reindex(oos_close.index).fillna(False)

    raw_long_pf = _run_pf(oos_close, oos_long_raw, oos_long_exit)
    raw_short_pf = _run_pf(oos_close, oos_short_raw, oos_short_exit, short=True)

    # MTF + ML: label_mtf_trades identifies the crossover entry bars; filter with ML
    oos_conf_aligned = confluence.reindex(oos_close.index, method="ffill")
    oos_labels_all, oos_stats = label_mtf_trades(
        oos_close, oos_conf_aligned, threshold, direction="long"
    )

    # H1 alignment filter variant
    oos_labels_h1, oos_stats_h1 = label_mtf_trades(
        oos_close, oos_conf_aligned, threshold, direction="long",
        h1_signal=h1_sig, h1_min_score=0.0,
    )

    # Exit hysteresis buffer variant
    oos_labels_buf, oos_stats_buf = label_mtf_trades(
        oos_close, oos_conf_aligned, threshold, direction="long",
        exit_buffer=0.10,
    )

    # Combined: H1 filter + exit buffer
    oos_labels_combo, oos_stats_combo = label_mtf_trades(
        oos_close, oos_conf_aligned, threshold, direction="long",
        h1_signal=h1_sig, h1_min_score=0.0, exit_buffer=0.10,
    )

    logger.info(
        f"  OOS signal stats (raw): total={oos_stats['total_signals']}, "
        f"blocked_in_trade={oos_stats['blocked_in_trade']}, "
        f"entered={oos_stats['entered']}"
    )
    logger.info(
        f"  OOS signal stats (H1 filter): h1_filtered={oos_stats_h1['h1_filtered']}, "
        f"entered={oos_stats_h1['entered']}"
    )

    # Build H1-filter VBT portfolio directly from labelled entry times
    def _labels_to_pf(oos_c: pd.Series, labs: pd.DataFrame, long_exit: pd.Series) -> vbt.Portfolio | None:
        if len(labs) == 0:
            return None
        entries_s = pd.Series(False, index=oos_c.index)
        entries_s.loc[labs.index[labs.index.isin(oos_c.index)]] = True
        return _run_pf(oos_c, entries_s, long_exit)

    h1_long_pf = _labels_to_pf(oos_close, oos_labels_h1, oos_long_exit)
    buf_long_pf = _labels_to_pf(oos_close, oos_labels_buf, oos_long_exit)
    combo_long_pf = _labels_to_pf(oos_close, oos_labels_combo, oos_long_exit)

    ml_long_pf = None
    n_oos_raw = len(oos_labels_all)
    n_oos_ml = 0

    if n_oos_raw > 0:
        oos_entry_times = oos_labels_all.index
        oos_entry_times_with_feat = oos_entry_times[oos_entry_times.isin(oos_feat_df.index)]

        X_oos = pd.concat(
            [
                full_feat_df.reindex(oos_entry_times_with_feat),
                posteriors_df.reindex(oos_entry_times_with_feat),
                conf_feat.reindex(oos_entry_times_with_feat),
            ],
            axis=1,
        ).dropna()

        if len(X_oos) > 0:
            ml_pass = pd.Series(ml.predict(X_oos).astype(bool), index=X_oos.index)
            approved = ml_pass[ml_pass].index
            n_oos_ml = len(approved)

            if n_oos_ml > 0:
                # Build entry series from approved crossover bars
                oos_long_ml_entries = pd.Series(False, index=oos_close.index)
                oos_long_ml_entries.loc[approved] = True

                # Exit: same confluence-flip logic as raw MTF
                ml_long_pf = _run_pf(oos_close, oos_long_ml_entries, oos_long_exit)

    logger.info(f"  OOS crossover events: raw={n_oos_raw}, ML-approved={n_oos_ml} "
                f"({n_oos_ml/max(n_oos_raw,1):.1%} pass rate)")

    # ── 9. Save models ─────────────────────────────────────────────────────────
    model_dir = ROOT / "models" / "ml_regime"
    model_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{pair.lower()}_h4_mtf"
    detector.save(str(model_dir / f"hmm_{slug}.pkl"))
    ml.save(str(model_dir / f"xgb_{slug}.pkl"))
    logger.info(f"Models saved: models/ml_regime/hmm_{slug}.pkl + xgb_{slug}.pkl")

    # ── 10. Report ─────────────────────────────────────────────────────────────
    rows = [
        _pf_stats(raw_long_pf, "MTF long  (raw)"),
        _pf_stats(raw_short_pf, "MTF short (raw)"),
    ]
    if h1_long_pf is not None:
        rows.append(_pf_stats(h1_long_pf, "MTF long  + H1 filter"))
    if buf_long_pf is not None:
        rows.append(_pf_stats(buf_long_pf, "MTF long  + exit buffer"))
    if combo_long_pf is not None:
        rows.append(_pf_stats(combo_long_pf, "MTF long  + H1 + buffer"))
    if ml_long_pf is not None:
        rows.append(_pf_stats(ml_long_pf, "MTF long  + ML filter"))

    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 75)
    print(f"ML REGIME STRATEGY -- {pair} H4 MTF -- OOS RESULTS")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)
    print(f"\n  IS trade win rate:  {n_win/len(is_labels):.2%} ({len(is_labels)} trades)")
    print(f"  OOF AUC:           {cv.get('oof_auc')}")
    print(f"  OOF F1:            {cv.get('oof_f1', 0.0):.4f}")
    print(f"  Optimal threshold: {cv.get('optimal_threshold'):.4f}")
    print("\n  Feature importances (top 5):")
    for feat, imp in ml.feature_importance().head(5).items():
        print(f"    {feat:<22} {imp:.4f}")

    print("\n  Signal path-dependency (OOS):")
    print(f"    Raw:        total={oos_stats['total_signals']}, "
          f"blocked_in_trade={oos_stats['blocked_in_trade']}, "
          f"entered={oos_stats['entered']}")
    print(f"    H1 filter:  h1_filtered={oos_stats_h1['h1_filtered']}, "
          f"entered={oos_stats_h1['entered']}")
    print(f"    Exit buf:   entered={oos_stats_buf['entered']}")
    print(f"    Combined:   h1_filtered={oos_stats_combo['h1_filtered']}, "
          f"entered={oos_stats_combo['entered']}")

    if cv.get("oof_auc") is not None and cv["oof_auc"] < 0.52:
        print("\n  [NOTE] OOF AUC near 0.5 -- ML adds marginal signal. "
              "Raw MTF may be the better choice.")

    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"ml_regime_{slug}.csv"
    results_df.to_csv(report_path, index=False)
    logger.info(f"Report saved: {report_path}")

    return {"variants": rows, "cv": cv}


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ML Regime + MTF Confluence Pipeline"
    )
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--is_frac", type=float, default=IS_FRAC)
    args = parser.parse_args()
    run_pipeline_mtf(pair=args.pair, is_frac=args.is_frac)


if __name__ == "__main__":
    main()
