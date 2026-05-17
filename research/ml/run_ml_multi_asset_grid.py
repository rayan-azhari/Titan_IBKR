"""ml multi-asset / multi-horizon / multi-model expansion of Wave C.

After the EUR/USD single-pair RETIRE (commit aa3a917), test whether the
no-edge finding is EUR/USD-specific or architecture-wide. L61 protocol:
before declaring a strategy line of research closed, run a multi-asset
panel test.

Grid:
    Universe (16 assets at H1):
        FX (9):     AUD_JPY, AUD_USD, EUR_CHF, EUR_GBP, EUR_JPY,
                    EUR_USD, GBP_USD, USD_CHF, USD_JPY
        Indices (2): ES, NQ
        ETFs (2):    SPY, QQQ
        Commod (2):  GLD, CL
        Bonds (1):   IEF
    Horizons (3): 6h, 12h, 24h  (TBM max_holding)
    Model classes (3): LogisticRegression, RandomForest, XGBoost
    Total trained models: 9 (horizon × model -- ONE pooled model per cell)

Per cell:
    1. For each asset:
       - Build 26-col features (current build_features() with D context;
         skip H4 because most non-FX assets lack H4 parquets).
       - Build TBM labels at this horizon (PT=2x ATR, SL=1x ATR).
       - Z-score features within IS (cross-asset pooling demands
         comparable feature scales).
       - Drop time-out (label=0) rows; binary +1 -> 1, -1 -> 0.
       - 12mo sanctuary holdout (per-asset window-based).
    2. POOL training rows across all assets.
    3. Train ONE model with pre-committed hyperparams (no L13 tuning).
    4. Per-asset: score the asset's sanctuary rows, compute AUC.
    5. Cell verdict:
       - mean_sanctuary_auc > 0.55 AND n_assets_above_055 >= 8 -> PROMOTE
       - mean_sanctuary_auc > 0.51 -> MARGINAL
       - else -> RETIRE

Outputs:
    .tmp/reports/ml_multi_asset_grid/result_log.md
    .tmp/reports/ml_multi_asset_grid/per_cell_table.csv

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ml/run_ml_multi_asset_grid.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ml.build_tbm_labels import _compute_atr, _tbm_kernel  # noqa: E402
from titan.strategies.ml.features import build_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ml_multi_asset_grid"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Universe.
ASSETS = {
    "fx": ["AUD_JPY", "AUD_USD", "EUR_CHF", "EUR_GBP", "EUR_JPY",
           "EUR_USD", "GBP_USD", "USD_CHF", "USD_JPY"],
    "index": ["ES", "NQ"],
    "etf": ["SPY", "QQQ"],
    "commodity": ["GLD", "CL"],
    "bond": ["IEF"],
}
ALL_ASSETS = [a for cls_list in ASSETS.values() for a in cls_list]

# TBM params.
PT_MULT = 2.0
SL_MULT = 1.0
ATR_PERIOD = 14
HORIZONS = [6, 12, 24]  # bars (each = 1h on H1)

# Sanctuary discipline.
SANCTUARY_MONTHS = 12

# Promotion thresholds.
SANC_AUC_PROMOTE = 0.55
SANC_AUC_MARGINAL = 0.51
N_ASSETS_PROMOTE = 8  # of 16

# Pre-committed model configs (L13: NO per-fold tuning).
RANDOM_SEED = 42

MODEL_CONFIGS = {
    "logreg": lambda: LogisticRegression(
        C=1.0, max_iter=500, random_state=RANDOM_SEED, n_jobs=4,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=RANDOM_SEED, n_jobs=4,
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=4,
    ),
}


def _load_h1(symbol: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{symbol}_H1.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    cols = ["open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]].astype(float)


def _load_d_context(symbol: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    cols = ["open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]].astype(float)


def build_asset_dataset(
    symbol: str, horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex] | None:
    """Build (X, y, train_idx, sanctuary_idx) for one asset.

    Returns None if data missing or insufficient.
    """
    h1 = _load_h1(symbol)
    if len(h1) < 5000:
        return None
    d = _load_d_context(symbol)
    ctx = {"D": d} if not d.empty else {}
    # Build features.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = build_features(h1, context_data=ctx, cfg=None)
    # Build TBM labels at this horizon.
    atr_arr = _compute_atr(h1, ATR_PERIOD)
    labels = _tbm_kernel(
        h1["close"].values, h1["high"].values, h1["low"].values,
        atr_arr, PT_MULT, SL_MULT, horizon,
    )
    labels_s = pd.Series(labels, index=h1.index, name="tbm_label")
    # Align + drop NaN/inf.
    common = feats.index.intersection(labels_s.index)
    X = feats.loc[common].replace([np.inf, -np.inf], np.nan)
    # Per-asset: drop columns with >50% NaN (e.g., vol_rsi when volume is
    # missing). The pool will use the intersection of surviving columns.
    nan_rate = X.isna().mean()
    keep_cols = nan_rate[nan_rate <= 0.5].index.tolist()
    X = X[keep_cols]
    y_raw = labels_s.loc[common]
    clean = X.notna().all(axis=1) & y_raw.notna()
    X = X[clean]
    y_raw = y_raw[clean]
    # Decisive only.
    decisive = y_raw != 0
    X = X[decisive]
    y = (y_raw[decisive] == 1).astype(int)
    if len(X) < 1000:
        return None
    # Sanctuary split.
    cutoff = X.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    train_idx = X.index[X.index <= cutoff]
    sanc_idx = X.index[X.index > cutoff]
    if len(sanc_idx) < 200 or len(train_idx) < 1000:
        return None
    return X, y, train_idx, sanc_idx


def run_cell(
    horizon: int, model_name: str,
) -> dict:
    """Run one (horizon, model_class) cell across all assets.

    Returns a dict with per-asset and aggregate metrics.
    """
    per_asset_data = {}
    common_features: list[str] | None = None
    for sym in ALL_ASSETS:
        d = build_asset_dataset(sym, horizon)
        if d is None:
            continue
        X, y, train_idx, sanc_idx = d
        # Use a consistent feature set across assets -- intersection.
        if common_features is None:
            common_features = sorted(X.columns.tolist())
        else:
            common_features = sorted(set(common_features) & set(X.columns))
        per_asset_data[sym] = (X, y, train_idx, sanc_idx)

    if not per_asset_data:
        return {"horizon": horizon, "model": model_name, "error": "no assets loaded"}

    # Restrict each asset's features to the common set.
    feature_names = common_features

    # Per-asset z-score within IS (so pooled training has comparable feature scales).
    per_asset_normed: dict[str, dict] = {}
    pooled_train_X: list[np.ndarray] = []
    pooled_train_y: list[np.ndarray] = []
    for sym, (X, y, train_idx, sanc_idx) in per_asset_data.items():
        Xf = X[feature_names].copy()
        scaler = StandardScaler()
        scaler.fit(Xf.loc[train_idx])
        Xn = pd.DataFrame(
            scaler.transform(Xf), index=Xf.index, columns=feature_names,
        )
        per_asset_normed[sym] = {
            "X_train": Xn.loc[train_idx].to_numpy(),
            "y_train": y.loc[train_idx].to_numpy(),
            "X_sanc": Xn.loc[sanc_idx].to_numpy(),
            "y_sanc": y.loc[sanc_idx].to_numpy(),
            "n_train": int(len(train_idx)),
            "n_sanc": int(len(sanc_idx)),
        }
        pooled_train_X.append(per_asset_normed[sym]["X_train"])
        pooled_train_y.append(per_asset_normed[sym]["y_train"])

    X_pool = np.vstack(pooled_train_X)
    y_pool = np.concatenate(pooled_train_y)

    # Train ONE model on pooled data.
    model = MODEL_CONFIGS[model_name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_pool, y_pool)

    # Per-asset sanctuary AUC.
    per_asset_metrics: dict[str, dict] = {}
    for sym, d in per_asset_normed.items():
        if len(d["y_sanc"]) < 50 or len(np.unique(d["y_sanc"])) < 2:
            per_asset_metrics[sym] = {
                "sanc_auc": float("nan"), "n_sanc": d["n_sanc"], "n_train": d["n_train"],
            }
            continue
        proba = model.predict_proba(d["X_sanc"])[:, 1]
        auc = float(roc_auc_score(d["y_sanc"], proba))
        per_asset_metrics[sym] = {
            "sanc_auc": auc, "n_sanc": d["n_sanc"], "n_train": d["n_train"],
        }

    # Aggregate.
    valid_aucs = [m["sanc_auc"] for m in per_asset_metrics.values()
                  if np.isfinite(m["sanc_auc"])]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    median_auc = float(np.median(valid_aucs)) if valid_aucs else float("nan")
    n_above_055 = int(sum(1 for v in valid_aucs if v > SANC_AUC_PROMOTE))
    n_above_051 = int(sum(1 for v in valid_aucs if v > SANC_AUC_MARGINAL))
    pct_above_05 = float(sum(1 for v in valid_aucs if v > 0.50) / max(len(valid_aucs), 1))

    # Verdict.
    if mean_auc > SANC_AUC_PROMOTE and n_above_055 >= N_ASSETS_PROMOTE:
        verdict = "PROMOTE"
    elif mean_auc > SANC_AUC_MARGINAL:
        verdict = "MARGINAL"
    else:
        verdict = "RETIRE"

    return {
        "horizon": horizon,
        "model": model_name,
        "n_assets_evaluated": len(per_asset_metrics),
        "n_pool_train_rows": int(len(X_pool)),
        "mean_sanc_auc": mean_auc,
        "median_sanc_auc": median_auc,
        "n_assets_above_055": n_above_055,
        "n_assets_above_051": n_above_051,
        "pct_assets_above_050": pct_above_05,
        "verdict": verdict,
        "per_asset": per_asset_metrics,
        "feature_count": len(feature_names),
    }


def main() -> None:
    print("=" * 88)
    print("ml multi-asset / multi-horizon / multi-model grid -- L61 expansion")
    print("=" * 88)
    print(f"\nUniverse: {len(ALL_ASSETS)} assets ({ASSETS})")
    print(f"Horizons: {HORIZONS}")
    print(f"Models:   {list(MODEL_CONFIGS.keys())}")
    print(f"Total cells: {len(HORIZONS) * len(MODEL_CONFIGS)}")

    all_results: list[dict] = []
    for horizon in HORIZONS:
        for model_name in MODEL_CONFIGS:
            print(f"\n--- horizon={horizon}h, model={model_name} ---")
            r = run_cell(horizon, model_name)
            all_results.append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            print(f"  assets={r['n_assets_evaluated']}, "
                  f"pool_train={r['n_pool_train_rows']:,}, "
                  f"features={r['feature_count']}")
            print(f"  mean_sanc_auc={r['mean_sanc_auc']:.4f}, "
                  f"median={r['median_sanc_auc']:.4f}")
            print(f"  n>0.55={r['n_assets_above_055']}, n>0.51={r['n_assets_above_051']}, "
                  f"pct>0.50={r['pct_assets_above_050']:.0%}")
            print(f"  VERDICT: {r['verdict']}")

    # Write outputs.
    rows = []
    for r in all_results:
        if "error" in r:
            continue
        rows.append({
            "horizon_h": r["horizon"],
            "model": r["model"],
            "n_assets": r["n_assets_evaluated"],
            "pool_train": r["n_pool_train_rows"],
            "features": r["feature_count"],
            "mean_sanc_auc": round(r["mean_sanc_auc"], 4),
            "median_sanc_auc": round(r["median_sanc_auc"], 4),
            "n_above_055": r["n_assets_above_055"],
            "n_above_051": r["n_assets_above_051"],
            "pct_above_050": round(r["pct_assets_above_050"], 4),
            "verdict": r["verdict"],
        })
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "per_cell_table.csv", index=False)

    # Markdown summary.
    rpath = REPORTS_DIR / "result_log.md"
    with rpath.open("w", encoding="utf-8") as fh:
        fh.write("# ml multi-asset grid -- L61 expansion of Wave C\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write(f"**Universe:** {len(ALL_ASSETS)} assets across "
                 f"{len(ASSETS)} classes\n")
        fh.write(f"**Horizons:** {HORIZONS}h\n")
        fh.write(f"**Models:** {list(MODEL_CONFIGS.keys())}\n")
        fh.write("**Per-cell training:** ONE pooled model across all "
                 "available assets after per-asset feature z-scoring.\n\n")
        fh.write("## Per-cell summary\n\n")
        fh.write("| Horizon | Model | n_assets | pool_train | mean_AUC | "
                 "median_AUC | n>0.55 | n>0.51 | %>0.50 | Verdict |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in all_results:
            if "error" in r:
                fh.write(f"| {r['horizon']}h | {r['model']} | ERROR -- {r['error']} |\n")
                continue
            fh.write(
                f"| {r['horizon']}h | {r['model']} | {r['n_assets_evaluated']} | "
                f"{r['n_pool_train_rows']:,} | {r['mean_sanc_auc']:.4f} | "
                f"{r['median_sanc_auc']:.4f} | {r['n_assets_above_055']} | "
                f"{r['n_assets_above_051']} | "
                f"{r['pct_assets_above_050']:.0%} | **{r['verdict']}** |\n"
            )
        fh.write("\n## Per-asset detail (best cell only)\n\n")
        best = max(
            [r for r in all_results if "error" not in r],
            key=lambda r: r["mean_sanc_auc"] if np.isfinite(r["mean_sanc_auc"]) else -1,
            default=None,
        )
        if best is not None:
            fh.write(f"**Best cell: horizon={best['horizon']}h, "
                     f"model={best['model']}, mean_AUC={best['mean_sanc_auc']:.4f}, "
                     f"verdict={best['verdict']}**\n\n")
            fh.write("| Asset | n_train | n_sanc | Sanctuary AUC |\n")
            fh.write("|---|---:|---:|---:|\n")
            for sym in sorted(best["per_asset"].keys()):
                m = best["per_asset"][sym]
                auc_str = (
                    f"{m['sanc_auc']:.4f}" if np.isfinite(m["sanc_auc"]) else "N/A"
                )
                fh.write(f"| {sym} | {m['n_train']:,} | {m['n_sanc']:,} | {auc_str} |\n")
        fh.write("\n## Verdict\n\n")
        if any(r.get("verdict") == "PROMOTE" for r in all_results):
            fh.write("**At least one cell PROMOTE.** Path forward: take the "
                     "PROMOTE cell, run full V3.6 5-axis on each asset's "
                     "stitched OOS returns, then portfolio-include via L67.\n")
        elif any(r.get("verdict") == "MARGINAL" for r in all_results):
            fh.write("**No cell PROMOTE, some MARGINAL.** Path forward: drill "
                     "into the marginal cell's per-asset AUC distribution; the "
                     "edge may be concentrated in a subset.\n")
        else:
            fh.write("**All cells RETIRE.** The architecture + features + "
                     "TBM label combination has no out-of-sample edge across "
                     "any tested horizon, model class, or asset universe. ml "
                     "line of research closed under V3.6 framework.\n")
    print(f"\nResult log: {rpath}")
    print(f"Per-cell CSV: {REPORTS_DIR / 'per_cell_table.csv'}")


if __name__ == "__main__":
    main()
