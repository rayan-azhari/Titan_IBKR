"""Retrain the ml meta-classifier on the CURRENT 29-column build_features() set.

Path 1 from `.tmp/reports/ml_wave_c_audit/result_log.md` (L71 / Wave C
RETIRE_BLOCKED finding). The 2026-04-03 frozen artefact expected 12
columns that no longer exist; this script produces a new artefact whose
feature_names match the current code.

Pipeline:
    1. Load EUR/USD H1 + H4 + D parquets.
    2. Build TBM labels on H1 (PT=2x ATR, SL=1x ATR, MAX_HOLDING=24).
    3. Build features via titan.strategies.ml.features.build_features()
       with H4 + D as context_data -> 29 columns.
    4. Align features[t] to labels[t] (forward-looking TBM is causal).
    5. Filter to bars where label != 0 (decisive outcomes only). Convert
       to binary: label=+1 (TP first) -> 1, label=-1 (SL first) -> 0.
    6. Hold out last 12 months as sanctuary (V3.6 contract).
    7. Train XGBoost classifier with pre-committed hyperparams (no
       per-fold tuning per L13). Single fit on visible data; the
       deployment-eligibility audit runs separately via
       research/ml/run_ml_wave_c_audit.py.
    8. Save to models/meta_model_EURUSD_H1_<TS>.joblib with embedded
       feature_names + trained_at + audit metadata.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ml/retrain_ml_model.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ml.build_tbm_labels import _compute_atr, _tbm_kernel  # noqa: E402
from titan.strategies.ml.features import build_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# TBM params (match build_tbm_labels.py defaults).
PT_MULT = 2.0
SL_MULT = 1.0
MAX_HOLDING = 24
ATR_PERIOD = 14

# Sanctuary discipline (V3.6).
SANCTUARY_MONTHS = 12

# Threshold for predict_proba -> trade decision. The live strategy hard-
# codes 0.6; we keep the same.
META_THRESHOLD = 0.6

# Pre-committed XGB hyperparameters (no per-fold tuning per L13).
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
)


def _load_h1_h4_d() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load EUR/USD H1 with H4 + D as context."""
    def _load(p: str) -> pd.DataFrame:
        fp = DATA_DIR / f"EUR_USD_{p}.parquet"
        df = pd.read_parquet(fp)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index().dropna(subset=["close"])
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    h1 = _load("H1")
    h4 = _load("H4")
    d = _load("D")
    return h1, {"H4": h4, "D": d}


def _build_labels(h1: pd.DataFrame) -> pd.DataFrame:
    """Triple Barrier Method labels on H1."""
    atr_arr = _compute_atr(h1, ATR_PERIOD)
    labels = _tbm_kernel(
        h1["close"].values, h1["high"].values, h1["low"].values,
        atr_arr, PT_MULT, SL_MULT, MAX_HOLDING,
    )
    return pd.DataFrame(
        {"tbm_label": labels, "atr": atr_arr},
        index=h1.index,
    )


def main() -> None:
    print("=" * 80)
    print("ml retrain -- meta-classifier on current 29-col build_features()")
    print("=" * 80)

    print("\n[1] Loading EUR/USD H1 + H4 + D...")
    h1, ctx = _load_h1_h4_d()
    print(f"  H1: {len(h1)} bars  ({h1.index[0]} -> {h1.index[-1]})")
    print(f"  H4: {len(ctx['H4'])} bars")
    print(f"  D:  {len(ctx['D'])} bars")

    print(f"\n[2] Building TBM labels (PT={PT_MULT}x ATR, SL={SL_MULT}x ATR, "
          f"horizon={MAX_HOLDING}h)...")
    labels_df = _build_labels(h1)
    n_total = len(labels_df)
    n_up = int((labels_df["tbm_label"] == 1).sum())
    n_dn = int((labels_df["tbm_label"] == -1).sum())
    n_to = int((labels_df["tbm_label"] == 0).sum())
    print(f"  +1 TP first: {n_up:6d}  ({n_up/n_total*100:.1f}%)")
    print(f"  -1 SL first: {n_dn:6d}  ({n_dn/n_total*100:.1f}%)")
    print(f"   0 time-out: {n_to:6d}  ({n_to/n_total*100:.1f}%)")

    print("\n[3] Building feature matrix (current 29-col build_features())...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = build_features(h1, context_data=ctx, cfg=None)
    feature_names = sorted([c for c in feats.columns])
    print(f"  feature_names ({len(feature_names)}): {feature_names}")

    print("\n[4] Aligning features + labels; filtering decisive outcomes...")
    common_idx = feats.index.intersection(labels_df.index)
    X_all = feats.loc[common_idx, feature_names].copy()
    y_label = labels_df.loc[common_idx, "tbm_label"].copy()
    clean = X_all.notna().all(axis=1) & y_label.notna()
    X_all = X_all[clean]
    y_label = y_label[clean]
    # Keep only decisive labels (+1 or -1); skip vertical-barrier (label=0).
    decisive_mask = y_label != 0
    X = X_all[decisive_mask]
    # Binary: TP-first -> 1, SL-first -> 0.
    y = (y_label[decisive_mask] == 1).astype(int)
    print(f"  total clean rows: {len(X_all):,}")
    print(f"  decisive rows:    {len(X):,}  ({len(X)/len(X_all)*100:.1f}%)")
    print(f"  class balance:    1={int(y.sum())} ({y.mean()*100:.1f}%), "
          f"0={int((1-y).sum())} ({(1-y).mean()*100:.1f}%)")

    print(f"\n[5] Holding out sanctuary (last {SANCTUARY_MONTHS} months)...")
    cutoff = X.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    visible_mask = X.index <= cutoff
    sanctuary_mask = X.index > cutoff
    X_v, y_v = X[visible_mask], y[visible_mask]
    X_s, y_s = X[sanctuary_mask], y[sanctuary_mask]
    print(f"  visible:   {len(X_v):,} rows ({X_v.index[0]} -> {X_v.index[-1]})")
    print(f"  sanctuary: {len(X_s):,} rows ({X_s.index[0]} -> {X_s.index[-1]})")

    print("\n[6] Training XGBoost on visible (pre-committed hyperparams, "
          "no per-fold tuning)...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_v, y_v)
    # In-sample AUC (sanity).
    train_auc = float(roc_auc_score(y_v, model.predict_proba(X_v)[:, 1]))
    print(f"  IS AUC (in-sample, train): {train_auc:.4f}")
    # Sanctuary AUC (honest -- never seen during fit).
    sanc_auc = (
        float(roc_auc_score(y_s, model.predict_proba(X_s)[:, 1]))
        if len(X_s) > 0 and y_s.nunique() == 2 else float("nan")
    )
    print(f"  Sanctuary AUC (held-out): {sanc_auc:.4f}")
    sanc_proba = model.predict_proba(X_s)[:, 1] if len(X_s) > 0 else np.array([])
    sanc_decision = (sanc_proba > META_THRESHOLD)
    n_take = int(sanc_decision.sum())
    sanc_win_rate = float(y_s[sanc_decision].mean()) if n_take > 0 else float("nan")
    print(f"  Sanctuary @ threshold={META_THRESHOLD}: "
          f"{n_take}/{len(X_s)} trades ({n_take/max(len(X_s),1)*100:.1f}%), "
          f"win_rate_meta={sanc_win_rate:.4f}")

    print("\n[7] Serialising artefact...")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = MODELS_DIR / f"meta_model_EURUSD_H1_{ts}.joblib"
    artefact = {
        "model": model,
        "threshold": META_THRESHOLD,
        "feature_names": feature_names,
        "mtf_config": {},  # current build_features reads its own MTF config from mtf.toml
        "trained_at": ts,
        "training_window": {
            "start": str(X_v.index[0]),
            "end": str(X_v.index[-1]),
            "n_rows": len(X_v),
        },
        "sanctuary_window": {
            "start": str(X_s.index[0]) if len(X_s) > 0 else None,
            "end": str(X_s.index[-1]) if len(X_s) > 0 else None,
            "n_rows": len(X_s),
        },
        "tbm_params": {
            "pt_mult": PT_MULT,
            "sl_mult": SL_MULT,
            "max_holding": MAX_HOLDING,
            "atr_period": ATR_PERIOD,
        },
        "xgb_params": XGB_PARAMS,
        "train_auc": train_auc,
        "sanctuary_auc": sanc_auc,
        "sanctuary_win_rate_meta": sanc_win_rate,
        "audit_ref": (
            ".tmp/reports/ml_wave_c_audit/result_log.md "
            "(retrain after feature-pipeline drift, L71)"
        ),
    }
    joblib.dump(artefact, out_path)
    print(f"  wrote: {out_path.relative_to(PROJECT_ROOT)} "
          f"({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
