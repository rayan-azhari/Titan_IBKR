"""run_metalabeling.py — Meta-Labeling on MTF Confluence signals (EUR/USD H1).

Pipeline:
  1. Compute MTF Confluence primary signals (H1/H4/D/W) using config/mtf.toml.
  2. Load TBM labels from build_tbm_labels.py (run that first).
  3. For each bar where primary signal != 0, generate:
       meta_label = 1 if TBM outcome matches primary direction
       meta_label = 0 otherwise (SL hit, time-out, or wrong direction)
  4. Train XGBoost binary meta-model with Purged K-Fold CV.
  5. Report per-fold: primary precision, meta AUC, EV improvement.
  6. Save meta-model to models/.

The meta-model does NOT replace the MTF Confluence strategy — it adds a
probability gate. At live inference, the MTF strategy only fires a trade
if meta_prob > threshold (default 0.60).
"""

import sys
import tomllib
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
META_THRESHOLD = 0.60  # Only trade if meta_prob > this
K_FOLDS = 5
EMBARGO_BARS = 24  # H1 bars to embargo at fold boundaries


# ── MTF Signal Computation ─────────────────────────────────────────────────


def _load_parquet(pair: str, gran: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI using Wilder's smoothing (EMA with alpha=1/period)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    """Per-TF directional signal in range [-1, +1].

    Each component contributes ±0.5:
      MA crossover:  fast > slow → +0.5, else -0.5
      RSI threshold: RSI > 50   → +0.5, else -0.5
    Total range: -1 (full bear) to +1 (full bull).
    """
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi_val = _rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi_val > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def compute_mtf_primary_signal(pair: str, mtf_cfg: dict) -> pd.Series:
    """Compute weighted MTF Confluence signal aligned to H1 index.

    Returns a Series with values:
      +1 = long signal (confluence > threshold)
      -1 = short signal (confluence < -threshold)
       0 = flat (below threshold)
    """
    weights = mtf_cfg["weights"]
    threshold = mtf_cfg["confirmation_threshold"]

    h1_df = _load_parquet(pair, "H1")
    h4_df = _load_parquet(pair, "H4")
    d_df = _load_parquet(pair, "D")
    w_df = _load_parquet(pair, "W")

    tfs = {"H1": h1_df, "H4": h4_df, "D": d_df, "W": w_df}
    h1_index = h1_df.index

    confluence = pd.Series(0.0, index=h1_index)
    for tf_name, df_tf in tfs.items():
        cfg = mtf_cfg[tf_name]
        sig = _tf_signal(df_tf["close"], cfg["fast_ma"], cfg["slow_ma"], cfg["rsi_period"])
        # Forward-fill higher TF signals onto the H1 index
        sig_aligned = sig.reindex(h1_index, method="ffill")
        confluence += sig_aligned * weights[tf_name]

    # Directional signal
    signal = pd.Series(0, index=h1_index, dtype=int)
    signal[confluence > threshold] = 1
    signal[confluence < -threshold] = -1

    raw_longs = (signal == 1).sum()
    raw_shorts = (signal == -1).sum()
    raw_flat = (signal == 0).sum()
    print(f"  MTF primary signals: {raw_longs:,} long, {raw_shorts:,} short, {raw_flat:,} flat")

    return signal, confluence


# ── Feature Engineering ────────────────────────────────────────────────────


def _bollinger_bandwidth(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    mid = close.rolling(period).mean()
    band = close.rolling(period).std() * std
    return (2 * band) / mid


def build_meta_features(
    h1_df: pd.DataFrame, confluence: pd.Series, primary: pd.Series
) -> pd.DataFrame:
    """Feature matrix for the meta-model.

    Includes technical features that predict *when* the MTF signal will succeed:
    volatility regime, momentum overextension, session, confluence strength.
    All computed on historical data only (no look-ahead).
    """
    close = h1_df["close"]
    feats = pd.DataFrame(index=h1_df.index)

    # Lagged returns (stationarity + momentum context)
    for lag in [1, 2, 3, 5, 10]:
        feats[f"ret_{lag}"] = close.pct_change(lag)

    # Volatility
    feats["atr_14"] = (
        pd.concat(
            [
                h1_df["high"] - h1_df["low"],
                (h1_df["high"] - h1_df["close"].shift()).abs(),
                (h1_df["low"] - h1_df["close"].shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )
    feats["atr_pct"] = feats["atr_14"] / close  # normalised ATR (avoids price level bias)
    feats["boll_bw"] = _bollinger_bandwidth(close, 20, 2.0)

    # Momentum
    feats["rsi_14"] = _rsi(close, 14)
    feats["rsi_21"] = _rsi(close, 21)
    feats["rsi_overextended"] = ((feats["rsi_14"] > 70) | (feats["rsi_14"] < 30)).astype(int)

    # Trend strength
    feats["adx_proxy"] = feats["atr_14"].rolling(14).mean() / feats["atr_14"].rolling(14).std()

    # Confluence signal features (the primary model's confidence)
    feats["confluence_score"] = confluence
    feats["confluence_abs"] = confluence.abs()  # how strong is the signal?
    feats["primary_signal"] = primary  # direction: +1 / -1 (only non-zero rows used)

    # Session (UTC hour)
    feats["hour_utc"] = h1_df.index.hour
    feats["is_london"] = ((feats["hour_utc"] >= 7) & (feats["hour_utc"] < 16)).astype(int)
    feats["is_new_york"] = ((feats["hour_utc"] >= 13) & (feats["hour_utc"] < 21)).astype(int)
    feats["is_overlap"] = ((feats["hour_utc"] >= 13) & (feats["hour_utc"] < 16)).astype(int)
    feats["day_of_week"] = h1_df.index.dayofweek

    return feats


# ── Purged K-Fold ──────────────────────────────────────────────────────────


def purged_kfold_splits(
    n: int, k: int = 5, embargo: int = 24
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward expanding splits with embargo.

    Unlike combinatorial purged K-Fold, this NEVER trains on future data.
    Each fold trains on all data BEFORE the test window (minus embargo),
    preventing look-ahead bias that occurs when future folds leak into training.

    For each fold i (starting from 1):
      - train = rows [0 : fold_start - embargo]
      - test  = rows [fold_start : fold_end]

    Args:
        n: Total number of observations.
        k: Number of folds.
        embargo: Bars to exclude between train and test sets.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    fold_size = n // k
    splits = []
    all_idx = np.arange(n)
    # Start from fold 1 — fold 0 has no prior training data
    for i in range(1, k):
        test_start = i * fold_size
        test_end = min(test_start + fold_size, n)
        train_end = max(0, test_start - embargo)

        train_idx = all_idx[:train_end]
        test_idx = all_idx[test_start:test_end]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ── VBT Sharpe helper (without VBT dependency) ────────────────────────────


def _signal_sharpe(meta_preds: np.ndarray, primary_vals: np.ndarray, returns: pd.Series) -> float:
    """Compute annualised Sharpe for meta-filtered trades.

    Strategy return per bar = primary_signal * forward_return if meta=1, else 0.
    """
    # Directional: primary +1 or -1 × forward return
    strat_ret = pd.Series(
        np.where(meta_preds == 1, primary_vals * returns.values, 0.0),
        index=returns.index,
    )
    if strat_ret.std() == 0:
        return 0.0
    # Annualise: H1 bars × 252 trading days × 24 bars/day
    return float(strat_ret.mean() / strat_ret.std() * np.sqrt(252 * 24))


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Run meta-labeling pipeline end-to-end."""
    print("=" * 70)
    print("Meta-Labeling Pipeline - MTF Confluence + TBM + Purged K-Fold CV")
    print("=" * 70)

    # 1. Load MTF config
    mtf_cfg_path = PROJECT_ROOT / "config" / "mtf.toml"
    with open(mtf_cfg_path, "rb") as f:
        mtf_cfg = tomllib.load(f)
    print(
        f"\nMTF config: weights={mtf_cfg['weights']}, threshold={mtf_cfg['confirmation_threshold']}"
    )

    # 2. Compute primary signals
    print("\n[1] Computing MTF Confluence primary signals...")
    primary, confluence = compute_mtf_primary_signal("EUR_USD", mtf_cfg)

    # 3. Load TBM labels
    tbm_path = FEATURES_DIR / "EUR_USD_H1_tbm_labels.parquet"
    if not tbm_path.exists():
        print(f"\nERROR: TBM labels not found at {tbm_path}")
        print("Run: .venv/Scripts/python.exe research/ml/build_tbm_labels.py")
        sys.exit(1)
    tbm = pd.read_parquet(tbm_path)
    print(f"\n[2] Loaded TBM labels: {len(tbm):,} rows")

    # 4. Align on H1 index
    h1_df = _load_parquet("EUR_USD", "H1")

    # Align all series to H1 index
    primary = primary.reindex(h1_df.index, fill_value=0)
    tbm_labels = tbm["tbm_label"].reindex(h1_df.index, fill_value=0)
    tbm_atr = tbm["atr"].reindex(h1_df.index)

    # 5. Build meta-labels
    # Only create meta observations where the primary model takes a trade
    print("\n[3] Building meta-labels...")
    trade_mask = primary != 0

    # meta_label = 1 if TBM label matches primary direction (win), else 0
    # TBM +1 = upper hit (long win), -1 = lower hit (long loss), 0 = time-out
    # If primary = +1 (long):  win when TBM = +1
    # If primary = -1 (short): win when TBM = -1
    meta_label = ((primary == 1) & (tbm_labels == 1)) | ((primary == -1) & (tbm_labels == -1))
    meta_label = meta_label.astype(int)

    # Restrict to bars where primary has a signal AND TBM label is non-zero
    # (zero TBM = warm-up or time-out — still valid, just inconclusive)
    active = trade_mask & tbm_atr.notna()

    print(f"  Total H1 bars:         {len(h1_df):,}")
    print(f"  Primary trade signals: {trade_mask.sum():,}  ({trade_mask.mean() * 100:.1f}%)")
    print(f"  Active (non-warmup):   {active.sum():,}")
    print(f"  Meta wins (label=1):   {meta_label[active].sum():,}")
    print(f"  Meta losses (label=0): {(1 - meta_label[active]).sum():,}")
    base_win_rate = meta_label[active].mean()
    print(f"  Baseline win rate:     {base_win_rate * 100:.1f}%")

    # 6. Build feature matrix (meta-model inputs)
    print("\n[4] Building feature matrix...")
    feats = build_meta_features(h1_df, confluence, primary)

    # Align active mask
    # Drop NaN features (warm-up rows)
    feats_active = feats[active].copy()
    labels_active = meta_label[active].copy()
    primary_active = primary[active].copy()
    h1_close_active = h1_df["close"][active].copy()

    # Drop rows with any NaN features
    valid_mask = feats_active.notna().all(axis=1)
    feats_active = feats_active[valid_mask]
    labels_active = labels_active[valid_mask]
    primary_active = primary_active[valid_mask]
    h1_close_active = h1_close_active[valid_mask]

    print(f"  Feature matrix: {feats_active.shape[0]:,} rows x {feats_active.shape[1]} features")
    print(f"  Class balance: {labels_active.mean() * 100:.1f}% wins")

    # 7. Purged K-Fold meta-model training
    print(f"\n[5] Purged K-Fold CV (k={K_FOLDS}, embargo={EMBARGO_BARS} bars)...")

    from xgboost import XGBClassifier

    X = feats_active.values
    y = labels_active.values
    # Compute forward returns on the FULL (consecutive) H1 series FIRST,
    # then filter to active bars. This avoids inflated returns when
    # pct_change() spans non-consecutive bars after filtering.
    h1_fwd_ret_full = h1_df["close"].pct_change(1).shift(-1)
    fwd_ret = h1_fwd_ret_full.reindex(feats_active.index).fillna(0).values

    splits = purged_kfold_splits(len(X), k=K_FOLDS, embargo=EMBARGO_BARS)

    fold_aucs = []
    fold_sharpes_raw = []
    fold_sharpes_meta = []
    fold_winrates_raw = []
    fold_winrates_meta = []
    fold_trade_counts = []

    all_meta_preds = np.zeros(len(X), dtype=int)
    all_meta_probs = np.zeros(len(X))

    print(
        f"\n  {'Fold':<5} {'Train':>8} {'Test':>8} {'AUC':>6} {'Win%(raw)':>10} {'Win%(meta)':>11} {'Shrp(raw)':>10} {'Shrp(meta)':>11} {'Trades':>7}"
    )
    print("  " + "-" * 82)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        meta_preds = (probs >= META_THRESHOLD).astype(int)

        # Store for final re-fit
        all_meta_probs[test_idx] = probs
        all_meta_preds[test_idx] = meta_preds

        auc = roc_auc_score(y_te, probs)
        fold_aucs.append(auc)

        # Primary (raw) win rate and Sharpe on test fold
        prim_test = primary_active.values[test_idx]
        ret_test = fwd_ret[test_idx]

        wr_raw = y_te.mean()
        fold_winrates_raw.append(wr_raw)

        # Meta-filtered win rate and Sharpe
        meta_mask = meta_preds == 1
        wr_meta = y_te[meta_mask].mean() if meta_mask.sum() > 0 else 0.0
        fold_winrates_meta.append(wr_meta)

        trade_count = meta_mask.sum()
        fold_trade_counts.append(trade_count)

        # Sharpe (raw = all primary trades, meta = filtered)
        sharpe_raw = _signal_sharpe(np.ones_like(meta_preds), prim_test, pd.Series(ret_test))
        sharpe_meta = _signal_sharpe(meta_preds, prim_test, pd.Series(ret_test))
        fold_sharpes_raw.append(sharpe_raw)
        fold_sharpes_meta.append(sharpe_meta)

        print(
            f"  {fold_i + 1:<5} {len(train_idx):>8,} {len(test_idx):>8,}"
            f" {auc:>6.3f} {wr_raw * 100:>9.1f}% {wr_meta * 100:>10.1f}%"
            f" {sharpe_raw:>10.2f} {sharpe_meta:>11.2f} {trade_count:>7,}"
        )

    print("  " + "-" * 82)
    avg_auc = np.mean(fold_aucs)
    avg_wr_raw = np.mean(fold_winrates_raw)
    avg_wr_meta = np.mean(fold_winrates_meta)
    avg_sh_raw = np.mean(fold_sharpes_raw)
    avg_sh_meta = np.mean(fold_sharpes_meta)
    avg_trades = np.mean(fold_trade_counts)

    print(
        f"  {'AVG':<5} {'':>8} {'':>8}"
        f" {avg_auc:>6.3f} {avg_wr_raw * 100:>9.1f}% {avg_wr_meta * 100:>10.1f}%"
        f" {avg_sh_raw:>10.2f} {avg_sh_meta:>11.2f} {avg_trades:>7.0f}"
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Meta-model AUC (avg over {K_FOLDS} folds): {avg_auc:.4f}")
    print(f"  Threshold:                           {META_THRESHOLD}")
    print(f"  Win rate - raw primary:              {avg_wr_raw * 100:.1f}%")
    print(
        f"  Win rate - meta-filtered:            {avg_wr_meta * 100:.1f}%  (d{(avg_wr_meta - avg_wr_raw) * 100:+.1f}pp)"
    )
    print(f"  Sharpe  - raw primary:               {avg_sh_raw:.2f}")
    print(
        f"  Sharpe  - meta-filtered:             {avg_sh_meta:.2f}  (d{avg_sh_meta - avg_sh_raw:+.2f})"
    )
    print(f"  Trades filtered in (avg/fold):       {avg_trades:.0f}")

    # 8. Final model: retrain on all data
    print("\n[6] Retraining final meta-model on full dataset...")
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    # Feature importance
    importances = pd.Series(
        final_model.feature_importances_, index=feats_active.columns
    ).sort_values(ascending=False)

    print("\n  Top 10 meta-model features:")
    for feat, imp in importances.head(10).items():
        bar = "#" * int(imp * 60)
        print(f"    {feat:<30s} {imp:.4f}  {bar}")

    # Save meta-model
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"meta_model_EURUSD_H1_{ts}.joblib"
    joblib.dump(
        {
            "model": final_model,
            "threshold": META_THRESHOLD,
            "feature_names": feats_active.columns.tolist(),
            "mtf_config": mtf_cfg,
            "trained_at": ts,
            "avg_auc": avg_auc,
            "avg_win_rate_meta": avg_wr_meta,
        },
        model_path,
    )
    print(f"\n  Meta-model saved: {model_path}")

    # Verdict
    print("\n" + "=" * 70)
    if avg_auc >= 0.55 and avg_wr_meta > avg_wr_raw:
        print("RESULT: Meta-model adds value. Proceed to NautilusTrader integration.")
        print(
            f"  AUC {avg_auc:.3f} >= 0.55 and win rate improved by {(avg_wr_meta - avg_wr_raw) * 100:+.1f}pp"
        )
    else:
        print("RESULT: Meta-model marginal. Review features or lower threshold.")
        print(f"  AUC {avg_auc:.3f} | Win rate delta {(avg_wr_meta - avg_wr_raw) * 100:+.1f}pp")
    print("=" * 70)


if __name__ == "__main__":
    main()
