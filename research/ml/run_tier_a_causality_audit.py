"""run_tier_a_causality_audit.py -- ML Tier A same-bar look-ahead audit.

Pre-registered in:
    directives/ML Same-Bar Causality Audit 2026-05-14.md

Runs the EXISTING ML pipeline (build_features, label sweep, WFO, XGBoost)
on EUR/USD D and QQQ D, and at each WFO fold computes the OOS Sharpe
TWO ways:

    A) `positions[t] * bar_returns[t]`         (as-deployed, same-bar)
    B) `positions.shift(1)[t] * bar_returns[t]` (causal, audit-corrected)

Same predictions in both -- only the alignment differs. The Sharpe gap
quantifies the look-ahead bias in the live Tier A claims.

This script does NOT modify the production pipeline. It re-runs the
public functions in `run_52signal_classifier` and computes the
alternative Sharpe inline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.ml.run_52signal_classifier import (  # noqa: E402
    COST_BPS,
    IS_RATIO_BARS,
    OOS_RATIO_BARS,
    SIGNAL_THRESHOLD,
    XGB_PARAMS,
    _get_annual_bars,
    _load_ohlcv,
    _pred_to_position,
    build_features,
    compute_regime_pullback_labels,
    walk_forward_splits,
)
from titan.research.metrics import bootstrap_sharpe_ci, sharpe  # noqa: E402


# Cells from directive §1.2
CELLS = [
    ("EUR_USD", "D", "fx"),
    ("QQQ", "D", "etf"),
]


def _compute_sharpe_pair(
    position: np.ndarray, oos_returns: pd.Series, cost: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (strategy_rets_buggy, strategy_rets_causal). Both per-bar return series."""
    pos = pd.Series(position, index=oos_returns.index)
    # Transitions for cost: same in both (cost timing isn't the bug).
    transitions = (pos != pos.shift(1).fillna(0.0)).astype(float)
    cost_per_bar = transitions * cost / 10_000

    # A: as-deployed (same-bar)
    buggy = pos * oos_returns - cost_per_bar
    # B: causal (position lagged by 1; equivalently, return is forward-aligned)
    causal = pos.shift(1).fillna(0.0) * oos_returns - cost_per_bar

    return buggy.dropna(), causal.dropna()


def audit_one_instrument(instrument: str, tf: str, asset_type: str) -> dict:
    print(f"\n{'=' * 80}")
    print(f"  AUDIT: {instrument} {tf} ({asset_type})")
    print(f"{'=' * 80}")
    df = _load_ohlcv(instrument, tf)
    n_bars = len(df)
    bars_yr = _get_annual_bars(tf)
    cost = COST_BPS.get(asset_type, 1.0)
    print(f"  {n_bars} bars, {df.index[0].date()} -> {df.index[-1].date()}, cost={cost} bps, bars/yr={bars_yr}")

    df.attrs["instrument"] = instrument
    features = build_features(df, tf)

    # Use the most permissive label config from run_instrument's sweep
    # (row 7 of LABEL_SWEEP). Rationale: a strict label config produces
    # too few training swings per fold and skips most folds. For an
    # audit of position-return alignment we just need ENOUGH folds with
    # meaningful predictions; the exact label config doesn't affect the
    # look-ahead bias quantification.
    label_params = {
        "rsi_oversold": 50,
        "rsi_overbought": 50,
        "confirm_bars": 5,
        "confirm_pct": 0.002,
    }
    labels, _ = compute_regime_pullback_labels(df, **label_params)

    bar_returns = df["close"].pct_change().fillna(0.0)
    mask_feat = features.notna().all(axis=1)
    features_all = features[mask_feat].copy()
    returns_all = bar_returns.reindex(features_all.index).fillna(0.0)
    labels_all = labels.reindex(features_all.index).fillna(0).astype(np.int8)

    is_bars_n = IS_RATIO_BARS.get(tf, 504)
    oos_bars_n = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(features_all), is_bars_n, oos_bars_n)
    if not folds:
        print(f"  Not enough data for WFO ({len(features_all)} < {is_bars_n + oos_bars_n})")
        return {"instrument": instrument, "tf": tf, "n_folds": 0, "reason": "insufficient_data"}

    print(f"  WFO: {len(folds)} folds (IS={is_bars_n}, OOS={oos_bars_n})")

    X_all = features_all.values
    stitched_buggy: list[pd.Series] = []
    stitched_causal: list[pd.Series] = []
    fold_sharpes_buggy: list[float] = []
    fold_sharpes_causal: list[float] = []

    for k, (is_idx, oos_idx) in enumerate(folds):
        # IS: train only on bars where label != 0 (entry candidates).
        X_is = np.where(np.isinf(X_all[is_idx]), 0.0, X_all[is_idx])
        y_is_full = labels_all.iloc[is_idx].values
        mask_entry = y_is_full != 0
        # Original code's threshold is < 20 entries (run_52signal_classifier.py:751).
        if mask_entry.sum() < 20:
            print(f"    fold {k}: insufficient training swings ({int(mask_entry.sum())}) -- skip")
            continue
        X_is_entry = X_is[mask_entry]
        y_is_entry = (y_is_full[mask_entry] == 1).astype(int)
        if y_is_entry.sum() == 0 or y_is_entry.sum() == len(y_is_entry):
            print(f"    fold {k}: degenerate label balance -- skip")
            continue

        pos_count = int(y_is_entry.sum())
        neg_count = len(y_is_entry) - pos_count
        spw = neg_count / pos_count if pos_count > 0 else 1.0
        params = {**XGB_PARAMS, "scale_pos_weight": spw, "eval_metric": "logloss"}
        model = XGBClassifier(**params)
        model.fit(X_is_entry, y_is_entry)

        X_oos = np.where(np.isinf(X_all[oos_idx]), 0.0, X_all[oos_idx])
        pred_proba = model.predict_proba(X_oos)[:, 1]
        position = _pred_to_position(pred_proba, threshold=SIGNAL_THRESHOLD)
        oos_returns = returns_all.iloc[oos_idx]

        buggy_rets, causal_rets = _compute_sharpe_pair(position, oos_returns, cost)

        sh_buggy = sharpe(buggy_rets, periods_per_year=bars_yr)
        sh_causal = sharpe(causal_rets, periods_per_year=bars_yr)
        fold_sharpes_buggy.append(sh_buggy)
        fold_sharpes_causal.append(sh_causal)
        stitched_buggy.append(buggy_rets)
        stitched_causal.append(causal_rets)
        print(
            f"    fold {k}: sharpe_buggy={sh_buggy:+.3f}, sharpe_causal={sh_causal:+.3f}, "
            f"delta={sh_buggy - sh_causal:+.3f}, n_oos={len(oos_returns)}"
        )

    if not stitched_buggy:
        return {"instrument": instrument, "tf": tf, "n_folds": 0, "reason": "no_folds_completed"}

    all_buggy = pd.concat(stitched_buggy)
    all_causal = pd.concat(stitched_causal)
    sh_b = sharpe(all_buggy, periods_per_year=bars_yr)
    sh_c = sharpe(all_causal, periods_per_year=bars_yr)
    ci_b_lo, ci_b_hi = bootstrap_sharpe_ci(all_buggy, periods_per_year=bars_yr, n_resamples=500, seed=42)
    ci_c_lo, ci_c_hi = bootstrap_sharpe_ci(all_causal, periods_per_year=bars_yr, n_resamples=500, seed=42)

    inflation = sh_b - sh_c
    inflation_pct = (inflation / sh_b * 100.0) if abs(sh_b) > 1e-9 else 0.0

    return {
        "instrument": instrument,
        "tf": tf,
        "n_folds": len(stitched_buggy),
        "n_oos_bars": len(all_buggy),
        "stitched_sharpe_buggy": round(sh_b, 4),
        "stitched_ci_buggy_lo": round(ci_b_lo, 4),
        "stitched_ci_buggy_hi": round(ci_b_hi, 4),
        "stitched_sharpe_causal": round(sh_c, 4),
        "stitched_ci_causal_lo": round(ci_c_lo, 4),
        "stitched_ci_causal_hi": round(ci_c_hi, 4),
        "inflation": round(inflation, 4),
        "inflation_pct": round(inflation_pct, 1),
        "fold_sharpes_buggy": [round(x, 4) for x in fold_sharpes_buggy],
        "fold_sharpes_causal": [round(x, 4) for x in fold_sharpes_causal],
    }


def _verdict(row: dict) -> tuple[str, str]:
    """Apply directive §1.3 decision rule."""
    infl_pct = row.get("inflation_pct", 0.0)
    sh_b = row.get("stitched_sharpe_causal", 0.0)
    ci_lo_b = row.get("stitched_ci_causal_lo", -1.0)
    if infl_pct >= 50 and sh_b <= 0.3:
        return "RETIRE", (
            f"Inflation {infl_pct:.1f}% >= 50% and causal Sharpe {sh_b:.3f} <= 0.3. "
            f"Published Sharpe is dominated by look-ahead. Open config-change PR; halt live trading."
        )
    if (30 <= infl_pct < 50) or (0.3 < sh_b <= 0.6):
        return "MATERIAL_INFLATION", (
            f"Inflation {infl_pct:.1f}% or Sharpe-after-fix {sh_b:.3f} in marginal zone. "
            f"Halt live trading pending re-validation."
        )
    if infl_pct < 30 and sh_b > 0.6 and ci_lo_b > 0:
        return "CONFIRMED_DEPLOYABLE", (
            f"Look-ahead exists (inflation {infl_pct:.1f}%) but causal Sharpe {sh_b:.3f} "
            f"with CI_lo {ci_lo_b:.3f} > 0 supports deployment. Fix the alignment for future "
            f"reporting; leave live running."
        )
    if infl_pct < 30 and sh_b <= 0.3:
        return "RETIRE_MARGINAL", (
            f"Inflation small ({infl_pct:.1f}%) but causal Sharpe {sh_b:.3f} <= 0.3 -- "
            f"strategy was marginal even pre-look-ahead. Retire."
        )
    return "UNDETERMINED", "Did not match any pre-committed scenario; manual review."


def main() -> None:
    print("=" * 80)
    print("  ML TIER A SAME-BAR CAUSALITY AUDIT")
    print("=" * 80)
    rows = []
    for instrument, tf, asset_type in CELLS:
        try:
            r = audit_one_instrument(instrument, tf, asset_type)
            rows.append(r)
        except Exception as exc:
            print(f"  ERROR on {instrument} {tf}: {exc}")

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for r in rows:
        if r.get("n_folds", 0) == 0:
            continue
        v, rationale = _verdict(r)
        r["verdict"] = v
        r["rationale"] = rationale
        print(f"\n  {r['instrument']} {r['tf']}:")
        print(f"    BUGGY (as-deployed):  Sharpe={r['stitched_sharpe_buggy']}  CI=[{r['stitched_ci_buggy_lo']}, {r['stitched_ci_buggy_hi']}]")
        print(f"    CAUSAL (corrected):   Sharpe={r['stitched_sharpe_causal']}  CI=[{r['stitched_ci_causal_lo']}, {r['stitched_ci_causal_hi']}]")
        print(f"    Inflation: {r['inflation']} ({r['inflation_pct']:.1f}% of buggy)")
        print(f"    Verdict: {v}")
        print(f"    Rationale: {rationale}")

    # Save
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / ".tmp" / "reports" / "ml_causality_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_{stamp}.parquet"
    safe_rows = []
    for r in rows:
        safe = {k: v for k, v in r.items()
                if not isinstance(v, list)}
        safe["fold_count"] = len(r.get("fold_sharpes_buggy", []))
        safe_rows.append(safe)
    if safe_rows:
        pd.DataFrame(safe_rows).to_parquet(out_path, index=False)
        print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
