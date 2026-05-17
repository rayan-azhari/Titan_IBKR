"""ml top-6 per-asset V3.6 5-axis audit.

Follow-up to the L61 multi-asset grid (commit 2033042). The best cell
(6h XGB pooled across 16 assets) showed 6 assets with sanctuary AUC >
0.55. AUC alone is not a deployment signal; this script converts the
model's proba into a (-1, 0, +1) trading signal per asset, computes
strategy returns, and runs the standard V3.6 5-axis matrix:

    1. CI_lo bootstrap (bootstrap_sharpe_ci)
    2. DSR deflation -- n_trials = 6 (one per asset in this audit;
       conservative against asset-pick selection bias)
    3. MC block bootstrap (run_block_mc) -- synthetic close paths, score
       returns. The model and feature pipeline are reapplied to synthetic
       paths via the asset's own build_features().
    4. Sanctuary divergence (sanctuary_divergence_test)
    5. Noise robustness (run_noise_robustness)

Pre-committed model: 6h XGBoost pooled, trained on z-scored features
across all 16 assets in the grid (same hyperparams as
run_ml_multi_asset_grid.py: n_estimators=200, max_depth=4, lr=0.05,
subsample=0.8, colsample=0.8). Per-asset z-scoring uses each asset's
IS-only mean+std (frozen). The pooled model + per-asset scaler is
saved to .tmp/reports/ml_top6_5axis/ for reproducibility.

Top-6 assets (sorted by best-cell sanctuary AUC):
    IEF, EUR_GBP, EUR_USD, GLD, USD_CHF, EUR_CHF

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ml/run_ml_top6_5axis.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    build_asset_dataset,
)
from titan.research.framework import (  # noqa: E402
    DecisionInputs,
    NoiseConfig,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    run_block_mc,
    run_noise_robustness,
    sanctuary_divergence_test,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402
from titan.strategies.ml.features import build_features  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ml_top6_5axis"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Best cell from the grid (commit 2033042).
HORIZON = 6
THRESHOLD = 0.55  # symmetric: proba > thr -> long, proba < (1-thr) -> short
TOP6 = ["IEF", "EUR_GBP", "EUR_USD", "GLD", "USD_CHF", "EUR_CHF"]

# Cost model (intraday FX/ETF/bond H1).
COST_BPS = 0.5
PERIODS_PER_YEAR = BARS_PER_YEAR["H1"]
SANCTUARY_MONTHS = 12

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="auc", tree_method="hist",
    random_state=42, n_jobs=4,
)


@dataclass
class AssetResult:
    asset: str
    sanctuary_auc: float
    fraction_active: float
    sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    mc_p_maxdd_gt_threshold: float
    mc_threshold_pct: float
    sanctuary_sharpe: float
    sanctuary_percentile: float
    noise_base: float
    noise_passes_mean: bool
    noise_passes_worst: bool
    noise_axis: str
    verdict: str


def _train_pooled_model() -> tuple[XGBClassifier, dict[str, StandardScaler], list[str]]:
    """Re-train the 6h XGB pooled model from the grid, returning (model,
    per-asset-scaler-dict, feature-name-list).

    Reproduces commit 2033042's best cell so this 5-axis audit uses the
    same predictions as the grid's per-asset sanctuary AUC.
    """
    per_asset_data = {}
    common_features = None
    for sym in ALL_ASSETS:
        d = build_asset_dataset(sym, HORIZON)
        if d is None:
            continue
        X, y, train_idx, sanc_idx = d
        if common_features is None:
            common_features = sorted(X.columns.tolist())
        else:
            common_features = sorted(set(common_features) & set(X.columns))
        per_asset_data[sym] = (X, y, train_idx, sanc_idx)
    if not per_asset_data:
        raise RuntimeError("no assets loaded for pooled training")

    feature_names = common_features
    scalers: dict[str, StandardScaler] = {}
    pooled_X = []
    pooled_y = []
    for sym, (X, y, train_idx, _sanc_idx) in per_asset_data.items():
        Xf = X[feature_names]
        sc = StandardScaler()
        sc.fit(Xf.loc[train_idx])
        scalers[sym] = sc
        pooled_X.append(sc.transform(Xf.loc[train_idx]))
        pooled_y.append(y.loc[train_idx].to_numpy())
    X_pool = np.vstack(pooled_X)
    y_pool = np.concatenate(pooled_y)
    model = XGBClassifier(**XGB_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_pool, y_pool)
    return model, scalers, feature_names


def _compute_signals_for_asset(
    h1: pd.DataFrame, model: XGBClassifier, scaler: StandardScaler,
    feature_names: list[str], ctx: dict[str, pd.DataFrame] | None = None,
) -> pd.Series:
    """Returns a (-1, 0, +1) signal Series aligned to h1.index, V3.6-causal
    (shifted by 1 bar so position effective at close[t] earns t->t+1 return).
    """
    if ctx is None:
        ctx = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = build_features(h1, context_data=ctx, cfg=None)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Per-asset col-NaN drop (matches grid's >50% NaN gate).
    nan_rate = feats.isna().mean()
    keep = nan_rate[nan_rate <= 0.5].index.tolist()
    feats = feats[keep]
    # Restrict to model's feature set; any missing → can't predict.
    missing = [c for c in feature_names if c not in feats.columns]
    if missing:
        return pd.Series(0.0, index=h1.index, name="ml_signal")
    X = feats[feature_names]
    # Drop duplicate index entries so .loc[Xc.index] = ... aligns.
    if X.index.duplicated().any():
        X = X[~X.index.duplicated(keep="last")]
    clean_mask = X.notna().all(axis=1)
    Xc = X[clean_mask]
    if Xc.empty:
        return pd.Series(0.0, index=h1.index, name="ml_signal")
    Xn = scaler.transform(Xc)
    proba_vals = model.predict_proba(Xn)[:, 1]
    proba = pd.Series(0.5, index=X.index, dtype=float)
    proba.loc[Xc.index] = proba_vals
    signal = pd.Series(0.0, index=X.index, dtype=float)
    signal.loc[proba > THRESHOLD] = 1.0
    signal.loc[proba < (1.0 - THRESHOLD)] = -1.0
    # Causal shift: position effective at close[t] uses info up to close[t-1].
    signal = signal.shift(1).reindex(h1.index).fillna(0.0)
    return signal.rename("ml_signal")


def _strategy_returns(h1: pd.DataFrame, signal: pd.Series) -> pd.Series:
    log_ret = np.log(h1["close"] / h1["close"].shift(1)).fillna(0.0)
    gross = signal * log_ret
    dpos = signal.diff().abs().fillna(0.0)
    cost = dpos * (2 * COST_BPS / 10_000.0)
    return (gross - cost).rename("ret")


def audit_asset(
    asset: str, model: XGBClassifier, scaler: StandardScaler,
    feature_names: list[str], sweep_sharpes: list[float],
) -> AssetResult | None:
    print(f"\n--- {asset} ---")
    h1 = _load_h1(asset)
    if len(h1) < 5000:
        print(f"  insufficient bars ({len(h1)})")
        return None
    d = _load_d_context(asset)
    ctx = {"D": d} if not d.empty else {}
    signal = _compute_signals_for_asset(h1, model, scaler, feature_names, ctx=ctx)
    rets = _strategy_returns(h1, signal)

    # Sanctuary split (per-asset window).
    cutoff = h1.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    visible_mask = rets.index <= cutoff
    sanc_mask = rets.index > cutoff
    rets_v = rets[visible_mask]
    rets_s = rets[sanc_mask]
    n_active = int((signal != 0).sum())
    frac_active = float(n_active / max(len(signal), 1))
    print(f"  active bars: {n_active}/{len(signal)} ({frac_active:.2%})")
    print(f"  visible: {len(rets_v)}, sanctuary: {len(rets_s)}")

    # Sanctuary AUC for reference.
    # (Need raw proba; recompute on sanctuary slice.)
    sanctuary_auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats_full = build_features(h1, context_data=ctx, cfg=None)
        feats_full = feats_full.replace([np.inf, -np.inf], np.nan)
        Xf = feats_full[feature_names].dropna()
        Xfn = scaler.transform(Xf)
        proba_full = pd.Series(model.predict_proba(Xfn)[:, 1], index=Xf.index)
        # Labels for sanctuary.
        atr_arr = _compute_atr(h1, 14)
        labs = _tbm_kernel(
            h1["close"].values, h1["high"].values, h1["low"].values,
            atr_arr, 2.0, 1.0, HORIZON,
        )
        labs_s = pd.Series(labs, index=h1.index, name="lab")
        common_s = proba_full.index.intersection(labs_s.index)
        common_s = common_s[common_s > cutoff]
        proba_s = proba_full.loc[common_s]
        labs_s_arr = labs_s.loc[common_s]
        dec = labs_s_arr != 0
        if dec.sum() > 50:
            y_bin = (labs_s_arr[dec] == 1).astype(int).to_numpy()
            sanctuary_auc = float(roc_auc_score(y_bin, proba_s[dec].to_numpy()))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] sanctuary AUC compute failed: {e}")
    print(f"  sanctuary AUC: {sanctuary_auc:.4f}")

    # 5-axis matrix.
    cls = StrategyClass.INTRADAY_MICROSTRUCTURE
    d_def = defaults_for(cls)
    headline_sr = float(sharpe(rets_v, periods_per_year=PERIODS_PER_YEAR))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        rets_v, periods_per_year=PERIODS_PER_YEAR, n_resamples=1000, seed=42,
    )
    # DSR with N=6 trials (one per top-6 asset).
    sr_var = float(np.var(sweep_sharpes, ddof=1)) if len(sweep_sharpes) > 1 else 0.0
    dsr = deflated_sharpe(
        headline_sr, sr_var_across_trials=sr_var, returns=rets_v, n_trials=len(TOP6),
    )

    # MC block bootstrap. The strategy_fn rebuilds signal from synth close.
    primary_close = h1["close"][visible_mask]

    def _strategy_fn(synth_df: pd.DataFrame) -> pd.Series:
        synth_h1 = pd.DataFrame({
            "open": synth_df["close"], "high": synth_df["close"],
            "low": synth_df["close"], "close": synth_df["close"], "volume": 1.0,
        })
        sig = _compute_signals_for_asset(
            synth_h1, model, scaler, feature_names, ctx={},
        )
        return _strategy_returns(synth_h1, sig)

    mc = run_block_mc(
        primary_close=primary_close, cfg=d_def.mc,
        strategy_fn=_strategy_fn, periods_per_year=PERIODS_PER_YEAR,
        seed=42, extra_series=None, n_workers=DEFAULT_MC_WORKERS,
    )

    sanc_sr = float(sharpe(rets_s, periods_per_year=PERIODS_PER_YEAR)) if len(rets_s) >= 50 else 0.0
    div = sanctuary_divergence_test(
        historical_returns=rets_v,
        sanctuary_returns=rets_s,
        periods_per_year=PERIODS_PER_YEAR,
    )

    closes_v = primary_close.rename("close").to_frame()

    def _noise_fn(closes_df_local: pd.DataFrame) -> pd.Series:
        synth_h1 = pd.DataFrame({
            "open": closes_df_local["close"], "high": closes_df_local["close"],
            "low": closes_df_local["close"], "close": closes_df_local["close"],
            "volume": 1.0,
        })
        sig = _compute_signals_for_asset(
            synth_h1, model, scaler, feature_names, ctx={},
        )
        return _strategy_returns(synth_h1, sig)

    noise = run_noise_robustness(
        closes_v, _noise_fn, periods_per_year=PERIODS_PER_YEAR,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30),
    )

    inputs = DecisionInputs(
        ci_lo=ci_lo, dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
        pass_threshold_prob=mc.pass_threshold_prob,
        sanctuary_sharpe=sanc_sr,
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
    )
    decision = decide(inputs)

    r = AssetResult(
        asset=asset,
        sanctuary_auc=round(sanctuary_auc, 4) if np.isfinite(sanctuary_auc) else float("nan"),
        fraction_active=round(frac_active, 4),
        sharpe=round(headline_sr, 4),
        ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
        dsr_prob=round(dsr.dsr_prob, 4),
        mc_p_maxdd_gt_threshold=round(mc.p_maxdd_gt_threshold, 4),
        mc_threshold_pct=round(mc.threshold_pct, 4),
        sanctuary_sharpe=round(sanc_sr, 4),
        sanctuary_percentile=round(div.percentile, 4)
        if np.isfinite(div.percentile) else float("nan"),
        noise_base=round(noise.base_sharpe, 4),
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
        noise_axis=decision.noise_axis,
        verdict=decision.verdict.value,
    )
    print(f"  Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
    print(f"  DSR={r.dsr_prob:.4f}  MC P(>{r.mc_threshold_pct*100:.0f}%)={r.mc_p_maxdd_gt_threshold:.4f}")
    print(f"  Sanc Sharpe={r.sanctuary_sharpe:+.4f}")
    print(f"  Noise: base={r.noise_base:+.4f}, "
          f"mean_pass={r.noise_passes_mean}, worst={r.noise_passes_worst}, "
          f"axis={r.noise_axis}")
    print(f"  Verdict: {r.verdict}")
    return r


def main() -> None:
    print("=" * 80)
    print("ml top-6 per-asset V3.6 5-axis audit")
    print("=" * 80)
    print(f"Cell: horizon={HORIZON}h, XGBoost, pooled-feature-z-scored")
    print(f"Top-6: {TOP6}")

    print("\n[1] Training pooled 6h XGB model (reproduces grid commit 2033042)...")
    model, scalers, feature_names = _train_pooled_model()
    print(f"  features: {len(feature_names)}, scaler dict size: {len(scalers)}")

    print("\n[2] Headline OOS Sharpes (Pass 1 for DSR variance)...")
    pass1 = {}
    # First quick pass to get sweep sharpes for DSR.
    for asset in TOP6:
        if asset not in scalers:
            print(f"  {asset}: not in scalers (no data) -- skip")
            continue
        h1 = _load_h1(asset)
        d = _load_d_context(asset)
        ctx = {"D": d} if not d.empty else {}
        signal = _compute_signals_for_asset(
            h1, model, scalers[asset], feature_names, ctx=ctx,
        )
        rets = _strategy_returns(h1, signal)
        cutoff = h1.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
        rets_v = rets[rets.index <= cutoff]
        sr = float(sharpe(rets_v, periods_per_year=PERIODS_PER_YEAR))
        pass1[asset] = sr
        print(f"  {asset}: Sharpe={sr:+.4f}")

    sweep_sharpes = list(pass1.values())

    print(f"\n[3] Full 5-axis per asset -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[AssetResult] = []
    for asset in TOP6:
        if asset not in scalers:
            continue
        r = audit_asset(asset, model, scalers[asset], feature_names, sweep_sharpes)
        if r is not None:
            results.append(r)

    # Save pooled model + scalers for reproducibility.
    artefact_path = REPORTS_DIR / "ml_top6_pooled_artefact.joblib"
    joblib.dump({
        "model": model, "scalers": scalers, "feature_names": feature_names,
        "horizon": HORIZON, "threshold": THRESHOLD,
        "trained_at": "2026-05-17",
    }, artefact_path)
    print(f"\nPooled artefact: {artefact_path.relative_to(PROJECT_ROOT)}")

    # Write result log.
    rpath = REPORTS_DIR / "result_log.md"
    with rpath.open("w", encoding="utf-8") as fh:
        fh.write("# ml top-6 per-asset 5-axis audit\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write("**Cell:** horizon=6h, XGBoost, pooled-feature-z-scored "
                 "(reproduces grid commit 2033042 best cell)\n\n")
        fh.write("## §1. Per-asset 5-axis matrix\n\n")
        fh.write("| Asset | Sanc AUC | Active | Sharpe | CI95 lo | CI95 hi | "
                 "DSR | MC P | Sanc Sharpe | Noise base | Noise axis | Verdict |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|\n")
        for r in results:
            fh.write(
                f"| {r.asset} | {r.sanctuary_auc:.4f} | "
                f"{r.fraction_active:.2%} | {r.sharpe:+.4f} | "
                f"{r.ci_lo:+.3f} | {r.ci_hi:+.3f} | {r.dsr_prob:.4f} | "
                f"{r.mc_p_maxdd_gt_threshold:.4f} | "
                f"{r.sanctuary_sharpe:+.4f} | {r.noise_base:+.4f} | "
                f"{r.noise_axis} | {r.verdict} |\n"
            )
        fh.write("\n## §2. Deployment-eligibility per V3.7 rule\n\n")
        fh.write("DEPLOY-eligible cells (DEPLOY verdict OR CONDITIONAL_WATCHPOINT "
                 "with noise=best) AND CI_lo > 0:\n\n")
        eligible = [
            r for r in results
            if (r.verdict == "DEPLOY"
                or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best"))
            and r.ci_lo > 0
        ]
        if eligible:
            best = max(eligible, key=lambda r: r.ci_lo)
            fh.write(f"**{len(eligible)} eligible asset(s).** "
                     f"Top by CI_lo: **{best.asset}** "
                     f"(CI_lo={best.ci_lo:+.3f}, Sharpe={best.sharpe:+.4f}, "
                     f"verdict={best.verdict}, noise={best.noise_axis}).\n")
            for r in sorted(eligible, key=lambda r: -r.ci_lo):
                fh.write(f"- {r.asset}: CI_lo={r.ci_lo:+.3f}, Sharpe={r.sharpe:+.4f}, "
                         f"verdict={r.verdict}, noise={r.noise_axis}\n")
        else:
            fh.write("**0 eligible assets.** No top-6 asset passes the "
                     "strict 5-axis + CI_lo > 0 rule.\n")
        fh.write("\n## §3. Verdict + next steps\n\n")
        if eligible:
            fh.write("**ml line of research: PER-ASSET DEPLOY candidate(s) "
                     "identified.** L65 ruin assessment + L67 portfolio "
                     "inclusion test required before any live cutover.\n")
        else:
            fh.write("**ml line of research: RETIRE.** Top-6 sanctuary AUC "
                     "above 0.55 was a classification-layer signal that does "
                     "NOT translate to a deployment-eligible Sharpe under the "
                     "full V3.6 5-axis matrix. The line is now properly "
                     "closed -- single-pair, single-horizon, AND multi-asset "
                     "per-asset 5-axis tests all fail.\n")
    print(f"\nResult log: {rpath}")


if __name__ == "__main__":
    main()
