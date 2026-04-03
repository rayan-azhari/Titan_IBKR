"""run_52sig_param_sweep.py -- Parameter-optimized ML signal discovery.

Sweeps feature construction, labeler, and signal threshold parameters with
a 3-level caching strategy for speed:

  Level 1: Pre-compute 52 IC signals + regime features ONCE (shared by all).
  Level 2: Pre-compute MA features per unique (fast, slow, trend, lt, ma_type).
  Level 3: Pre-compute trailing-stop labels per unique (atr_p, stop_mult, max_hold).

XGBoost is fit once per (feature_set, label_set) pair.
Signal thresholds are swept analytically on predictions (no refit).

This reduces XGBoost fits from N_total_combos to N_feat_sets x N_label_sets,
typically 10-50x faster than the naive approach.

Usage
-----
    uv run python research/ml/run_52sig_param_sweep.py --instrument QQQ --fast
    uv run python research/ml/run_52sig_param_sweep.py --instrument EUR_USD
    uv run python research/ml/run_52sig_param_sweep.py  # all instruments
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
    build_all_signals,
)
from research.ml.run_52signal_classifier import (  # noqa: E402
    COST_BPS,
    IS_RATIO_BARS,
    OOS_RATIO_BARS,
    TARGET_INSTRUMENTS,
    _compute_regime_features,
    _trailing_stop_kernel,
    walk_forward_splits,
)
from titan.strategies.ml.features import atr as feat_atr  # noqa: E402
from titan.strategies.ml.features import ema, sma, wma  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep dimensions
# ---------------------------------------------------------------------------

# Feature params: (fast, slow, trend, longterm, ma_type)
FEAT_GRID_FULL = {
    "fast": [10, 20, 30],
    "slow": [50, 80, 120],
    "trend": [600, 1200, 2400],
    "longterm": [2400, 4800],
    "ma_type": ["EMA", "SMA", "WMA"],
}
FEAT_GRID_FAST = {
    "fast": [15, 20],
    "slow": [50, 100],
    "trend": [1200],
    "longterm": [4800],
    "ma_type": ["EMA", "SMA"],
}

# Label params: (atr_p, stop_mult, max_hold)
LABEL_GRID_FULL = {
    "atr_p": [10, 14, 20],
    "stop_mult": [1.5, 2.0, 3.0],
    "max_hold": [48, 120, 240],
}
LABEL_GRID_FAST = {
    "atr_p": [14, 20],
    "stop_mult": [2.0, 3.0],
    "max_hold": [120, 240],
}

# Threshold: swept analytically on predictions (no refit)
THRESH_FULL = [0.25, 0.5, 1.0]
THRESH_FAST = [0.5, 1.0]


# ---------------------------------------------------------------------------
# Parameterized feature builder (Level 2 cache)
# ---------------------------------------------------------------------------


def _ma_fn(s: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "EMA":
        return s.ewm(span=period, adjust=False).mean()
    elif ma_type == "WMA":
        w = np.arange(1, period + 1, dtype=float)
        return s.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    return s.rolling(period).mean()


def _build_ma_features(
    close: pd.Series,
    df: pd.DataFrame,
    fast_p: int,
    slow_p: int,
    trend_p: int,
    lt_p: int,
    ma_type: str,
) -> pd.DataFrame:
    """Build ~11 MA features for one parameter set."""
    ma = pd.DataFrame(index=close.index)
    fast_ma = _ma_fn(close, fast_p, ma_type)
    slow_ma = _ma_fn(close, slow_p, ma_type)
    trend_sma = sma(close, trend_p)
    lt_sma = sma(close, lt_p)
    lt_ema = ema(close, lt_p)
    lt_wma = wma(close, lt_p)
    trend_ema = ema(close, trend_p)
    trend_wma = wma(close, trend_p)

    def _sd(a, b):
        return a / b.where(b.abs() > 1e-10, np.nan)

    ma["ma_spread_fast"] = _sd(fast_ma - slow_ma, slow_ma)
    ma["ma_spread_intra"] = _sd(slow_ma - trend_sma, trend_sma)
    ma["ma_spread_daily"] = _sd(trend_sma - lt_sma, lt_sma)
    ma["price_vs_trend"] = _sd(close - trend_sma, trend_sma)
    ma["price_vs_longterm"] = _sd(close - lt_sma, lt_sma)
    ma["fast_above_slow"] = (fast_ma > slow_ma).astype(float)
    ma["trend_above_lt"] = (trend_sma > lt_sma).astype(float)
    ma["dist_from_lt"] = _sd(
        close - lt_sma,
        close.rolling(252).std().where(close.rolling(252).std() > 1e-10, np.nan),
    )
    ma["ema_spread_trend_lt"] = _sd(trend_ema - lt_ema, lt_ema)
    ma["wma_spread_trend_lt"] = _sd(trend_wma - lt_wma, lt_wma)
    atr_val = feat_atr(df)
    ma["trend_strength"] = (trend_sma - lt_sma).abs() / atr_val.where(atr_val > 1e-10, np.nan)
    return ma


# ---------------------------------------------------------------------------
# Label builder (Level 3 cache)
# ---------------------------------------------------------------------------


def _build_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_p: int,
    stop_mult: float,
    max_hold: int,
    n: int,
) -> np.ndarray:
    """Build trailing-stop net_R labels. Returns float64 array length n."""
    # Inline Wilder ATR for custom period
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for k in range(1, n):
        tr[k] = max(high[k] - low[k], abs(high[k] - close[k - 1]), abs(low[k] - close[k - 1]))
    atr_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= atr_p:
        atr_arr[atr_p - 1] = np.mean(tr[:atr_p])
        for k in range(atr_p, n):
            atr_arr[k] = (atr_arr[k - 1] * (atr_p - 1) + tr[k]) / atr_p

    long_r, short_r = _trailing_stop_kernel(close, high, low, atr_arr, stop_mult, max_hold)
    return long_r - short_r


# ---------------------------------------------------------------------------
# Evaluate: fit XGB once, sweep thresholds analytically
# ---------------------------------------------------------------------------


def _fit_and_sweep_thresholds(
    X_is: np.ndarray,
    y_is: np.ndarray,
    ret_is: np.ndarray,
    X_oos: np.ndarray,
    ret_oos: np.ndarray,
    thresholds: list[float],
    cost_bps: float,
    bars_yr: int,
) -> list[dict]:
    """Fit XGBRegressor on IS, predict on IS+OOS, sweep thresholds.

    Returns list of dicts (one per threshold) with is_sharpe, oos_sharpe, model.
    """
    from xgboost import XGBRegressor

    X_is_c = np.nan_to_num(X_is, nan=0.0, posinf=0.0, neginf=0.0)
    X_oos_c = np.nan_to_num(X_oos, nan=0.0, posinf=0.0, neginf=0.0)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_is_c, y_is)

    pred_is = model.predict(X_is_c)
    pred_oos = model.predict(X_oos_c)

    results = []
    for thresh in thresholds:
        # IS Sharpe
        sig_is = np.where(pred_is > thresh, 1.0, np.where(pred_is < -thresh, -1.0, 0.0))
        trans_is = np.abs(np.diff(sig_is, prepend=0.0))
        strat_is = sig_is * ret_is - trans_is * cost_bps / 10_000
        std_is = float(strat_is.std())
        sh_is = float(strat_is.mean() / std_is * np.sqrt(bars_yr)) if std_is > 1e-10 else 0.0

        # OOS Sharpe
        sig_oos = np.where(pred_oos > thresh, 1.0, np.where(pred_oos < -thresh, -1.0, 0.0))
        trans_oos = np.abs(np.diff(sig_oos, prepend=0.0))
        strat_oos = sig_oos * ret_oos - trans_oos * cost_bps / 10_000
        std_oos = float(strat_oos.std())
        sh_oos = float(strat_oos.mean() / std_oos * np.sqrt(bars_yr)) if std_oos > 1e-10 else 0.0

        n_long = int((sig_oos == 1.0).sum())
        n_short = int((sig_oos == -1.0).sum())
        pct_active = (n_long + n_short) / len(sig_oos) if len(sig_oos) > 0 else 0.0

        results.append(
            {
                "thresh": thresh,
                "is_sharpe": sh_is,
                "oos_sharpe": sh_oos,
                "n_long": n_long,
                "n_short": n_short,
                "pct_active": pct_active,
                "model": model,
                "oos_strat_rets": strat_oos,
                "importances": model.feature_importances_,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main per-instrument pipeline
# ---------------------------------------------------------------------------


def run_instrument_sweep(
    instrument: str,
    tf: str,
    asset_type: str,
    feat_grid: dict,
    label_grid: dict,
    thresholds: list[float],
) -> pd.DataFrame:
    """WFO with per-fold 3-level cached parameter sweep."""
    print(f"\n  Loading {instrument} {tf} ...")
    df = _load_ohlcv(instrument, tf)
    n = len(df)
    bars_yr = _get_annual_bars(tf)
    cost = COST_BPS.get(asset_type, 1.0)

    print(f"  {n:,} bars | {df.index[0].date()} to {df.index[-1].date()} | ~{n / bars_yr:.1f} yr")

    # Level 1: shared across all param sets (compute ONCE)
    print("  Level 1: 52 IC signals + regime features ...")
    window_1y = _get_annual_bars(tf)
    signals_52 = build_all_signals(df, window_1y)
    regime = _compute_regime_features(df)
    bar_returns = df["close"].pct_change().fillna(0.0).values
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_s = df["close"]

    # Level 2: pre-compute MA features for each unique feature param set
    feat_combos = [
        (f, s, t, lt, mt)
        for f, s, t, lt, mt in product(
            feat_grid["fast"],
            feat_grid["slow"],
            feat_grid["trend"],
            feat_grid["longterm"],
            feat_grid["ma_type"],
        )
        if f < s < t < lt
    ]
    print(f"  Level 2: {len(feat_combos)} MA feature sets ...")
    feat_cache: dict[tuple, np.ndarray] = {}
    feat_names: list[str] | None = None
    for fk in feat_combos:
        f, s, t, lt, mt = fk
        ma_feats = _build_ma_features(close_s, df, f, s, t, lt, mt)
        combined = pd.concat([signals_52, ma_feats, regime], axis=1).shift(1)
        feat_cache[fk] = combined.values
        if feat_names is None:
            feat_names = combined.columns.tolist()

    # Level 3: pre-compute labels for each unique label param set
    label_combos = list(
        product(
            label_grid["atr_p"],
            label_grid["stop_mult"],
            label_grid["max_hold"],
        )
    )
    print(f"  Level 3: {len(label_combos)} label sets ...")
    label_cache: dict[tuple, np.ndarray] = {}
    for lk in label_combos:
        ap, sm, mh = lk
        label_cache[lk] = _build_labels(close_arr, high_arr, low_arr, ap, sm, mh, n)

    n_models = len(feat_combos) * len(label_combos)
    n_total = n_models * len(thresholds)

    # WFO
    is_bars_n = IS_RATIO_BARS.get(tf, 504)
    oos_bars_n = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(n, is_bars_n, oos_bars_n)
    if not folds:
        print("  [SKIP] Not enough data for WFO")
        return pd.DataFrame()

    print(
        f"  WFO: {len(folds)} folds | {n_models} XGB fits/fold | "
        f"{n_total} total evals/fold | {len(thresholds)} thresholds (free)"
    )

    fold_results: list[dict] = []
    all_oos_returns: list[pd.Series] = []
    all_importances: list[np.ndarray] = []

    for fi, (is_idx, oos_idx) in enumerate(folds):
        best_is_sharpe = -np.inf
        best_result: dict | None = None
        best_fk = None
        best_lk = None
        n_tried = 0

        for fk in feat_combos:
            X_full = feat_cache[fk]
            for lk in label_combos:
                y_full = label_cache[lk]

                # Mask valid rows (no NaN in features or labels)
                mask = ~np.isnan(y_full) & np.all(np.isfinite(X_full), axis=1)

                is_mask = np.zeros(n, dtype=bool)
                is_mask[is_idx] = True
                oos_mask = np.zeros(n, dtype=bool)
                oos_mask[oos_idx] = True

                valid_is = mask & is_mask
                valid_oos = mask & oos_mask

                if valid_is.sum() < 100 or valid_oos.sum() < 50:
                    continue

                X_is = X_full[valid_is]
                y_is = y_full[valid_is]
                ret_is = bar_returns[valid_is]
                X_oos = X_full[valid_oos]
                ret_oos = bar_returns[valid_oos]

                # Fit once, sweep all thresholds
                thresh_results = _fit_and_sweep_thresholds(
                    X_is,
                    y_is,
                    ret_is,
                    X_oos,
                    ret_oos,
                    thresholds,
                    cost,
                    bars_yr,
                )

                for tr in thresh_results:
                    n_tried += 1
                    if tr["is_sharpe"] > best_is_sharpe:
                        best_is_sharpe = tr["is_sharpe"]
                        best_result = tr
                        best_result["oos_index"] = df.index[valid_oos]
                        best_fk = fk
                        best_lk = lk

        if best_result is None:
            print(f"    Fold {fi + 1}: no valid combo ({n_tried} tried).")
            continue

        oos_sh = best_result["oos_sharpe"]
        parity = (oos_sh / best_is_sharpe) if best_is_sharpe != 0.0 else float("nan")

        # Stitch OOS returns
        oos_idx_ts = best_result["oos_index"]
        oos_rets = pd.Series(best_result["oos_strat_rets"], index=oos_idx_ts)
        all_oos_returns.append(oos_rets)
        all_importances.append(best_result["importances"])

        f, s, t, lt, mt = best_fk
        ap, sm, mh = best_lk
        param_str = (
            f"{mt}({f}/{s}) trend={t} lt={lt} "
            f"atr={ap} stop={sm}x hold={mh} thr={best_result['thresh']}"
        )

        fold_results.append(
            {
                "fold": fi + 1,
                "oos_start": oos_idx_ts[0].date(),
                "oos_end": oos_idx_ts[-1].date(),
                "is_sharpe": round(best_is_sharpe, 3),
                "oos_sharpe": round(oos_sh, 3),
                "parity": round(parity, 3) if not np.isnan(parity) else float("nan"),
                "pct_active": round(best_result["pct_active"], 3),
                "n_long": best_result["n_long"],
                "n_short": best_result["n_short"],
                "best_params": param_str,
            }
        )

        print(
            f"    Fold {fi + 1}/{len(folds)} ({n_tried} evals): "
            f"IS={best_is_sharpe:+.2f}  OOS={oos_sh:+.2f}  par={parity:.2f}  "
            f"L={best_result['n_long']} S={best_result['n_short']}  "
            f"BEST: {param_str}"
        )

    if not fold_results:
        return pd.DataFrame()

    # Stitched OOS equity
    stitched = pd.concat(all_oos_returns).sort_index()
    std = float(stitched.std())
    stitched_sharpe = float(stitched.mean() / std * np.sqrt(bars_yr)) if std > 1e-10 else 0.0
    stitched_eq = (1.0 + stitched).cumprod()
    stitched_dd = float(((stitched_eq - stitched_eq.cummax()) / stitched_eq.cummax()).min())
    total_ret = float(stitched_eq.iloc[-1] - 1.0)

    results_df = pd.DataFrame(fold_results)
    n_folds = len(results_df)
    pct_pos = (results_df["oos_sharpe"] > 0).mean()
    avg_par = results_df["parity"].mean()
    worst = results_df["oos_sharpe"].min()

    print(f"\n  {'=' * 80}")
    print(f"  RESULTS: {instrument} {tf} (parameter-optimized)")
    print(f"  {'=' * 80}")
    print("\n  Per-fold WFO:")
    print(
        f"  {'Fold':>4} {'OOS Start':>12} {'OOS End':>12} "
        f"{'IS':>7} {'OOS':>7} {'Par':>5} {'%Act':>5}  Best Params"
    )
    print("  " + "-" * 90)
    for _, r in results_df.iterrows():
        print(
            f"  {int(r['fold']):>4} {str(r['oos_start']):>12} {str(r['oos_end']):>12} "
            f"{r['is_sharpe']:>+7.2f} {r['oos_sharpe']:>+7.2f} "
            f"{r['parity']:>5.2f} {r['pct_active']:>5.0%}  {r['best_params']}"
        )

    print(
        f"\n  Stitched OOS: Sharpe={stitched_sharpe:+.3f}  DD={stitched_dd:.1%}  Ret={total_ret:+.1%}"
    )
    print(f"  Folds positive: {pct_pos:.0%} ({int(pct_pos * n_folds)}/{n_folds})")
    print(f"  Avg parity: {avg_par:.2f}  |  Worst fold: {worst:+.2f}")

    # Parameter frequency
    print("\n  Most-selected parameters:")
    for params, count in results_df["best_params"].value_counts().head(5).items():
        print(f"    {count}x  {params}")

    # Feature importance (averaged)
    if all_importances and feat_names:
        avg_imp = np.mean(all_importances, axis=0)
        imp_df = pd.DataFrame({"feature": feat_names, "importance": avg_imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print("\n  Top 10 features:")
        for _, row in imp_df.head(10).iterrows():
            bar = "#" * int(row["importance"] * 200)
            print(f"    {row['feature']:<25s} {row['importance']:.4f}  {bar}")

    results_df["instrument"] = instrument
    results_df["tf"] = tf
    results_df["stitched_sharpe"] = stitched_sharpe
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter-optimized 52-signal ML sweep.")
    parser.add_argument("--instrument", default=None, help="Single instrument (default: all).")
    parser.add_argument("--tf", default=None, help="Override timeframe for all instruments.")
    parser.add_argument("--fast", action="store_true", help="Reduced grid (~4x faster).")
    args = parser.parse_args()

    if args.instrument:
        if args.instrument in TARGET_INSTRUMENTS:
            instruments = {args.instrument: TARGET_INSTRUMENTS[args.instrument]}
        else:
            instruments = {args.instrument: ("H1", "fx")}
        if args.tf:
            instruments = {k: (args.tf, v[1]) for k, v in instruments.items()}
    elif args.tf:
        instruments = {k: (args.tf, v[1]) for k, v in TARGET_INSTRUMENTS.items()}
    else:
        instruments = TARGET_INSTRUMENTS

    fast = args.fast
    fg = FEAT_GRID_FAST if fast else FEAT_GRID_FULL
    lg = LABEL_GRID_FAST if fast else LABEL_GRID_FULL
    th = THRESH_FAST if fast else THRESH_FULL

    n_feat = sum(
        1
        for c in product(fg["fast"], fg["slow"], fg["trend"], fg["longterm"], fg["ma_type"])
        if c[0] < c[1] < c[2] < c[3]
    )
    n_label = len(lg["atr_p"]) * len(lg["stop_mult"]) * len(lg["max_hold"])
    n_models = n_feat * n_label
    n_total = n_models * len(th)

    W = 80
    print()
    print("=" * W)
    print("  ML PARAM SWEEP -- 3-Level Cached Optimization")
    print(f"  Instruments  : {list(instruments.keys())}")
    print(f"  Feature sets : {n_feat} (MA params x types)")
    print(f"  Label sets   : {n_label} (ATR x stop x hold)")
    print(f"  Thresholds   : {th} (free — no refit)")
    print(f"  XGB fits/fold: {n_models}  |  Total evals/fold: {n_total}")
    print(f"  Mode         : {'FAST' if fast else 'FULL'}")
    print("=" * W)

    all_results: list[pd.DataFrame] = []
    for inst, (tf, asset_type) in instruments.items():
        try:
            df = run_instrument_sweep(inst, tf, asset_type, fg, lg, th)
            if not df.empty:
                all_results.append(df)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {inst}: {exc}")
        except Exception as exc:
            print(f"  [ERROR] {inst}: {exc}")
            import traceback

            traceback.print_exc()

    if not all_results:
        print("\n  No results.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    print()
    print("=" * W)
    print("  CROSS-INSTRUMENT SUMMARY")
    print("=" * W)
    print(
        f"\n  {'Instrument':<12} {'TF':>3} {'Folds':>5} {'Stitched':>9} "
        f"{'%Pos':>5} {'AvgPar':>7} {'Worst':>6}"
    )
    print("  " + "-" * 55)
    for inst in instruments:
        inst_df = combined[combined["instrument"] == inst]
        if inst_df.empty:
            print(f"  {inst:<12} -- skip")
            continue
        st = inst_df["stitched_sharpe"].iloc[0]
        pp = (inst_df["oos_sharpe"] > 0).mean()
        ap = inst_df["parity"].mean()
        wf = inst_df["oos_sharpe"].min()
        print(
            f"  {inst:<12} {inst_df['tf'].iloc[0]:>3} {len(inst_df):>5} {st:>+9.3f} {pp:>5.0%} {ap:>7.2f} {wf:>+6.2f}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"ml_52sig_paramsweep_{ts}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
