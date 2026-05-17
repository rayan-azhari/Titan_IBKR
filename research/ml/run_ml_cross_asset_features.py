"""ml cross-asset features test -- add 7-feature regime panel to every bar.

Follow-up to the L72 RETIRE verdict. The previous grid + 5-axis cascade
tested per-asset features only. This script adds the I1v2 regime panel
(`data/i1_regime_panel.parquet`, 7 macro features: vix_z, term_spread_z,
credit_spread_z, rv20_z, spy_above_sma200, dxy_z, dd_velocity_21) as
columns on EVERY asset row. Now when predicting EUR/USD at bar t, the
model also sees what bonds, equities, vol, and dollar were doing.

If this lifts mean sanctuary AUC materially AND the resulting strategy
Sharpe clears 5-axis on at least one asset, the L72 ml RETIRE was
premature -- the architecture needed cross-asset features. If not, L72
truly closes the line.

Comparison baseline (same cell, no cross-asset features):
    horizon=6h, XGBoost, mean sanctuary AUC = 0.5424,
    6/16 above 0.55, 0/6 deployment-eligible.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ml/run_ml_cross_asset_features.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ml.build_tbm_labels import _compute_atr, _tbm_kernel  # noqa: E402
from research.ml.run_ml_multi_asset_grid import (  # noqa: E402
    ALL_ASSETS,
    _load_d_context,
    _load_h1,
)
from titan.strategies.ml.features import build_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ml_cross_asset_features"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FP = DATA_DIR / "i1_regime_panel.parquet"
HORIZON = 6  # best cell from grid
SANCTUARY_MONTHS = 12

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="auc", tree_method="hist",
    random_state=42, n_jobs=4,
)


def _load_panel() -> pd.DataFrame:
    """Load regime panel and dedupe."""
    df = pd.read_parquet(PANEL_FP).sort_index().dropna(how="any")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_asset_dataset_with_cross_asset(
    symbol: str, horizon: int, panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex] | None:
    """Same as run_ml_multi_asset_grid.build_asset_dataset but joins the
    regime panel as additional columns on every row.
    """
    h1 = _load_h1(symbol)
    if len(h1) < 5000:
        return None
    d = _load_d_context(symbol)
    ctx = {"D": d} if not d.empty else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = build_features(h1, context_data=ctx, cfg=None)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Drop per-asset cols with >50% NaN.
    nan_rate = feats.isna().mean()
    keep_cols = nan_rate[nan_rate <= 0.5].index.tolist()
    feats = feats[keep_cols]

    # Cross-asset features: merge the daily regime panel as columns.
    # H1 bars use the most recent daily panel row (.ffill via reindex).
    h1_dates = pd.to_datetime(feats.index).normalize()
    panel_for_h1 = panel.reindex(h1_dates).ffill(limit=5)
    panel_for_h1.index = feats.index  # align back to H1 timestamps
    # Prefix to make sure there's no name collision (e.g. dxy_z, vix_z
    # already could appear in per-asset features... they don't, but be safe).
    panel_for_h1 = panel_for_h1.add_prefix("xa_")
    feats = feats.join(panel_for_h1, how="left")
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Build TBM labels.
    atr_arr = _compute_atr(h1, 14)
    labels = _tbm_kernel(
        h1["close"].values, h1["high"].values, h1["low"].values,
        atr_arr, 2.0, 1.0, horizon,
    )
    labels_s = pd.Series(labels, index=h1.index, name="tbm_label")

    common = feats.index.intersection(labels_s.index)
    X = feats.loc[common]
    y_raw = labels_s.loc[common]
    # Dedup index.
    if X.index.duplicated().any():
        X = X[~X.index.duplicated(keep="last")]
        y_raw = y_raw[~y_raw.index.duplicated(keep="last")]
    clean = X.notna().all(axis=1) & y_raw.notna()
    X = X[clean]
    y_raw = y_raw[clean]
    # Decisive only.
    decisive = y_raw != 0
    X = X[decisive]
    y = (y_raw[decisive] == 1).astype(int)
    if len(X) < 1000:
        return None
    cutoff = X.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    train_idx = X.index[X.index <= cutoff]
    sanc_idx = X.index[X.index > cutoff]
    if len(sanc_idx) < 200 or len(train_idx) < 1000:
        return None
    return X, y, train_idx, sanc_idx


def main() -> None:
    print("=" * 88)
    print("ml cross-asset features test (regime panel as columns on every bar)")
    print("=" * 88)
    panel = _load_panel()
    print(f"\nPanel: {panel.shape}, range {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Panel features: {list(panel.columns)}")

    # Load all assets with cross-asset features.
    per_asset_data = {}
    common_features = None
    for sym in ALL_ASSETS:
        d = build_asset_dataset_with_cross_asset(sym, HORIZON, panel)
        if d is None:
            print(f"  {sym}: DROPPED")
            continue
        X, y, tr, sa = d
        if common_features is None:
            common_features = sorted(X.columns.tolist())
        else:
            common_features = sorted(set(common_features) & set(X.columns))
        per_asset_data[sym] = (X, y, tr, sa)
        print(f"  {sym}: train={len(tr):,}, sanc={len(sa):,}, features={len(X.columns)}")

    if not per_asset_data:
        print("ERROR: no assets loaded")
        return

    feature_names = common_features
    n_xa = sum(1 for c in feature_names if c.startswith("xa_"))
    print(f"\nCommon features: {len(feature_names)} ({n_xa} cross-asset, "
          f"{len(feature_names) - n_xa} per-asset)")

    # Per-asset z-score + pool.
    pooled_X = []
    pooled_y = []
    scalers = {}
    for sym, (X, y, tr, _sa) in per_asset_data.items():
        Xf = X[feature_names]
        sc = StandardScaler()
        sc.fit(Xf.loc[tr])
        scalers[sym] = sc
        pooled_X.append(sc.transform(Xf.loc[tr]))
        pooled_y.append(y.loc[tr].to_numpy())
    X_pool = np.vstack(pooled_X)
    y_pool = np.concatenate(pooled_y)
    print(f"\nPool: {X_pool.shape}")

    print("\nTraining XGBoost (pre-committed hyperparams)...")
    model = XGBClassifier(**XGB_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_pool, y_pool)

    print("\nPer-asset sanctuary AUC:")
    print(f"  {'asset':>10} {'n_train':>10} {'n_sanc':>10} {'AUC':>10}")
    per_asset_auc = {}
    for sym, (X, y, _tr, sa) in per_asset_data.items():
        if len(sa) < 50 or y.loc[sa].nunique() < 2:
            print(f"  {sym:>10} -- insufficient sanctuary")
            continue
        Xn = scalers[sym].transform(X.loc[sa, feature_names])
        proba = model.predict_proba(Xn)[:, 1]
        auc = float(roc_auc_score(y.loc[sa].to_numpy(), proba))
        per_asset_auc[sym] = auc
        train_n = int((X.index <= sa[0]).sum())
        print(f"  {sym:>10} {train_n:>10,} {len(sa):>10,} {auc:>10.4f}")

    valid = list(per_asset_auc.values())
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    median_auc = float(np.median(valid)) if valid else float("nan")
    n_above_055 = int(sum(1 for v in valid if v > 0.55))
    n_above_060 = int(sum(1 for v in valid if v > 0.60))
    n_above_065 = int(sum(1 for v in valid if v > 0.65))
    pct_above_050 = float(sum(1 for v in valid if v > 0.50) / max(len(valid), 1))

    print(f"\nAggregate (n_assets={len(valid)}):")
    print(f"  mean AUC = {mean_auc:.4f}, median = {median_auc:.4f}")
    print(f"  n>0.55 = {n_above_055}, n>0.60 = {n_above_060}, n>0.65 = {n_above_065}")
    print(f"  pct > 0.50 = {pct_above_050:.0%}")

    # Comparison vs baseline (no cross-asset features).
    baseline_mean_auc = 0.5424
    baseline_n_above_055 = 6
    improvement = mean_auc - baseline_mean_auc

    print("\nCOMPARISON vs baseline (no cross-asset features):")
    print(f"  baseline mean AUC = {baseline_mean_auc:.4f}, n>0.55 = {baseline_n_above_055}")
    print(f"  with cross-asset = {mean_auc:.4f}, n>0.55 = {n_above_055}")
    print(f"  improvement: {improvement:+.4f} mean AUC, "
          f"{n_above_055 - baseline_n_above_055:+d} assets above 0.55")

    # Verdict.
    if mean_auc > 0.58 and n_above_055 >= 10:
        verdict = "MATERIAL_IMPROVEMENT -- proceed to 5-axis"
    elif mean_auc > baseline_mean_auc + 0.02 and n_above_055 > baseline_n_above_055:
        verdict = "MARGINAL_IMPROVEMENT -- consider per-asset 5-axis on top assets"
    else:
        verdict = "NO_MATERIAL_GAIN -- L72 RETIRE stands"
    print(f"\nVerdict: {verdict}")

    # Write result log.
    rpath = REPORTS_DIR / "result_log.md"
    with rpath.open("w", encoding="utf-8") as fh:
        fh.write("# ml cross-asset features test (regime panel as columns)\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write("**Hypothesis:** Adding the 7-feature regime panel as columns on "
                 "every asset row (so the model sees bond/equity/vol/dollar context "
                 "when predicting any asset) materially lifts the L72 RETIRE.\n\n")
        fh.write(f"**Cell:** horizon={HORIZON}h, XGBoost (best from grid 2033042).\n")
        fh.write(f"**Cross-asset features added (prefix `xa_`):** {len(panel.columns)} "
                 f"({list(panel.columns)})\n\n")
        fh.write(f"**Total features:** {len(feature_names)} ({n_xa} cross-asset + "
                 f"{len(feature_names) - n_xa} per-asset)\n\n")
        fh.write("## Per-asset sanctuary AUC\n\n")
        fh.write("| Asset | n_train | n_sanc | AUC | Above 0.55? |\n")
        fh.write("|---|---:|---:|---:|:---:|\n")
        for sym, auc in sorted(per_asset_auc.items(), key=lambda kv: -kv[1]):
            tr_n = int((per_asset_data[sym][0].index <= per_asset_data[sym][3][0]).sum())
            mark = "Y" if auc > 0.55 else "-"
            fh.write(f"| {sym} | {tr_n:,} | {len(per_asset_data[sym][3]):,} | "
                     f"{auc:.4f} | {mark} |\n")
        fh.write("\n## Aggregate\n\n")
        fh.write("| Metric | Baseline (no xa) | With xa | Diff |\n|---|---:|---:|---:|\n")
        fh.write(f"| Mean AUC | {baseline_mean_auc:.4f} | {mean_auc:.4f} | {improvement:+.4f} |\n")
        fh.write(f"| n above 0.55 | {baseline_n_above_055} | {n_above_055} | "
                 f"{n_above_055 - baseline_n_above_055:+d} |\n")
        fh.write(f"| n above 0.60 | -- | {n_above_060} | -- |\n")
        fh.write(f"| n above 0.65 | -- | {n_above_065} | -- |\n")
        fh.write(f"\n**Verdict:** {verdict}\n")
    print(f"\nResult log: {rpath}")


if __name__ == "__main__":
    main()
