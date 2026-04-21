"""evaluate.py -- Multi-strategy evaluation harness for autonomous research.

DO NOT MODIFY THIS FILE. This is the ground truth evaluation.
The autonomous agent modifies experiment.py only; this file runs WFO
and computes the composite score for ANY strategy type.

Supported strategies:
    - xgboost, stacking, lstm_e2e  (ML classifiers)
    - mean_reversion               (VWAP MR with confluence regime)
    - trend_following              (ETF trend with decel signals)
    - cross_asset                  (Bond->equity/gold momentum)
    - gold_macro                   (3-component gold macro signal)
    - fx_carry                     (FX carry with VIX filter)
    - pairs_trading                (Spread mean reversion)
    - portfolio                    (Multi-strategy weighted combo)

Usage:
    uv run python research/auto/evaluate.py
    uv run python research/auto/evaluate.py --timeout 300

Output (stdout, parseable):
    SCORE: 1.234
    SHARPE: +1.450
    MAX_DD: -8.2%
    PARITY: 0.720
    POS_FOLDS: 80.0%
    TRADES: 156
    WORST_FOLD: -0.340
    N_FOLDS: 5
    INSTRUMENTS: QQQ,SPY
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
EXPERIMENT_FILE = Path(__file__).parent / "experiment.py"


# -- Composite Score (UNCHANGED from v1) --------------------------------------


def composite_score(
    sharpe: float,
    max_dd: float,
    parity: float,
    pct_positive: float,
    worst_fold: float,
    n_trades: int,
    n_folds: int,
) -> float:
    """Single scalar score. Higher is better.

    Hard gates reject clearly broken strategies. Soft penalties handle
    the grey area (bad folds, low parity) via continuous score reduction
    so non-ML strategies aren't unfairly rejected.
    """
    # Hard gates (absolute minimums)
    if n_folds < 2:
        return -99.0
    if n_trades < 5:
        return -99.0

    score = sharpe

    # Parity: reward consistency, penalize negative parity
    # Only reward parity when BOTH IS and OOS are profitable (positive Sharpe).
    # If IS < 0 and OOS < 0, parity would be positive but strategy is failing.
    if parity > 0 and sharpe > 0:
        score += 0.3 * min(parity, 1.5)
    elif parity < 0:
        score += 0.15 * max(parity, -2.0)  # mild penalty for negative parity

    # Drawdown penalty (continuous)
    score -= 0.5 * max(0, -max_dd - 0.15)

    # Fold consistency (continuous)
    score += 0.2 * (pct_positive - 0.5)

    # Worst fold: continuous penalty instead of hard gate
    # Below -3.0 gets progressively penalized, not instantly rejected
    if worst_fold < -3.0:
        score -= 0.3 * (-worst_fold - 3.0)

    return round(score, 4)


EMPTY_RESULT = {
    "sharpe": 0.0,
    "max_dd": 0.0,
    "parity": 0.0,
    "pct_positive": 0.0,
    "worst_fold": -99.0,
    "n_trades": 0,
    "n_folds": 0,
}


# -- Shared helpers ------------------------------------------------------------


def _load_daily(sym: str) -> pd.DataFrame:
    """Load daily OHLCV parquet. Returns DataFrame with DatetimeIndex."""
    for prefix in ["", "^"]:
        path = DATA_DIR / f"{prefix}{sym}_D.parquet"
        if path.exists():
            df = pd.read_parquet(path).sort_index()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
    raise FileNotFoundError(f"No daily data for {sym}")


def _sharpe_from_rets(rets, ann_factor=252):
    """Annualized Sharpe from a return series/array.

    Thin wrapper over ``titan.research.metrics.sharpe``. The old
    implementation filtered ``rets != 0.0`` before annualising, which
    overstated Sharpe by ``sqrt(1/active_ratio)`` for sparse strategies
    (April 2026 audit). The shared helper does NOT filter zero days; the
    caller is responsible for passing the correct ``ann_factor`` for the
    series frequency.
    """
    from titan.research.metrics import sharpe as _sh

    return _sh(rets, periods_per_year=int(ann_factor))


def _max_dd_from_rets(rets):
    """Max drawdown from a return series/array."""
    if hasattr(rets, "values"):
        rets = rets.values
    if len(rets) < 5:
        return 0.0
    eq = np.cumprod(1 + rets)
    hwm = np.maximum.accumulate(eq)
    dd = (eq - hwm) / np.where(hwm > 0, hwm, 1.0)
    return float(np.min(dd))


# -- Generic Rolling WFO (for strategies without built-in WFO) ----------------


def generic_rolling_wfo(
    build_returns_fn,
    df: pd.DataFrame,
    cfg: dict,
    is_days: int = 504,
    oos_days: int = 126,
) -> dict:
    """Rolling WFO for any signal-based strategy.

    build_returns_fn(df_slice, cfg) -> pd.Series of daily returns.
    We call it on IS and OOS slices separately.
    """
    n = len(df)
    if n < is_days + oos_days:
        return dict(EMPTY_RESULT)

    fold_sharpes = []
    all_oos_rets = []
    all_is_sharpes = []
    total_trades = 0
    oos_start = is_days

    while oos_start + oos_days <= n:
        is_start = max(0, oos_start - is_days)

        df_is = df.iloc[is_start:oos_start]
        df_oos = df.iloc[oos_start : oos_start + oos_days]

        try:
            is_rets = build_returns_fn(df_is, cfg)
            oos_rets = build_returns_fn(df_oos, cfg)
        except Exception:
            oos_start += oos_days
            continue

        is_sh = _sharpe_from_rets(is_rets)
        oos_sh = _sharpe_from_rets(oos_rets)

        fold_sharpes.append(oos_sh)
        all_is_sharpes.append(is_sh)
        if len(oos_rets) > 0:
            all_oos_rets.append(oos_rets.values if hasattr(oos_rets, "values") else oos_rets)

        # Count trades from signal transitions
        sig = cfg.get("_last_signal", None)
        if sig is not None and hasattr(sig, "__len__"):
            total_trades += int(np.sum(np.abs(np.diff(sig)) > 0))
        else:
            total_trades += max(1, len(oos_rets[oos_rets != 0]) // 20) if len(oos_rets) > 0 else 0

        oos_start += oos_days

    if not fold_sharpes:
        return dict(EMPTY_RESULT)

    stitched = np.concatenate(all_oos_rets) if all_oos_rets else np.array([])
    oos_sharpe = _sharpe_from_rets(stitched)
    max_dd = _max_dd_from_rets(stitched)

    avg_is = float(np.mean(all_is_sharpes)) if all_is_sharpes else 0.0
    parity = oos_sharpe / avg_is if abs(avg_is) > 0.01 else 0.0
    pct_pos = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
    worst = min(fold_sharpes)

    return {
        "sharpe": round(oos_sharpe, 4),
        "max_dd": round(max_dd, 4),
        "parity": round(parity, 4),
        "pct_positive": round(pct_pos, 4),
        "worst_fold": round(worst, 4),
        "n_trades": total_trades,
        "n_folds": len(fold_sharpes),
    }


# ==============================================================================
#  STRATEGY RUNNERS
# ==============================================================================


# -- 1. ML Classifier (xgboost, stacking, lstm_e2e) ---------------------------


def run_ml_wfo(instrument: str, cfg: dict, return_raw: bool = False) -> dict:
    """ML classifier WFO — existing v1 logic."""
    from research.ic_analysis.phase1_sweep import _get_annual_bars, _load_ohlcv
    from research.ml.run_52signal_classifier import (
        COST_BPS,
        IS_RATIO_BARS,
        OOS_RATIO_BARS,
        SIGNAL_THRESHOLD,
        TARGET_INSTRUMENTS,
        XGB_PARAMS,
        _pred_to_position,
        build_features,
        compute_regime_pullback_labels,
        compute_signal_sharpe,
        walk_forward_splits,
    )

    if instrument in TARGET_INSTRUMENTS:
        tf, asset_type = TARGET_INSTRUMENTS[instrument]
    else:
        tf = cfg.get("timeframe", "D")
        asset_type = "index"

    if "timeframe" in cfg:
        tf = cfg["timeframe"]

    cost = cfg.get("cost_bps", COST_BPS.get(asset_type, 2.0))
    bars_yr = _get_annual_bars(tf)

    df = _load_ohlcv(instrument, tf)
    df.attrs["instrument"] = instrument
    features = build_features(df, tf)

    label_params_list = cfg.get(
        "label_params",
        [
            {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 10, "confirm_pct": 0.005},
        ],
    )

    label_cache = []
    for lp in label_params_list:
        labels, _ = compute_regime_pullback_labels(df, **lp)
        label_cache.append((lp, labels))

    bar_returns = df["close"].pct_change().fillna(0.0)
    mask = features.notna().all(axis=1)
    feat_clean = features[mask].copy()
    ret_clean = bar_returns.reindex(feat_clean.index).fillna(0.0)

    is_bars = IS_RATIO_BARS.get(tf, 504)
    oos_bars = OOS_RATIO_BARS.get(tf, 126)

    if "is_years" in cfg:
        is_bars = int(cfg["is_years"] * 252 * (bars_yr / 252))
    if "oos_months" in cfg:
        oos_bars = int(cfg["oos_months"] * 21 * (bars_yr / 252))

    folds = walk_forward_splits(len(feat_clean), is_bars, oos_bars)
    if not folds:
        return dict(EMPTY_RESULT)

    X_all = feat_clean.values
    all_idx = feat_clean.index
    strategy_type = cfg.get("strategy", "xgboost")
    threshold = cfg.get("signal_threshold", SIGNAL_THRESHOLD)

    all_oos_rets = []
    all_is_rets = []
    fold_sharpes = []
    total_trades = 0

    for is_idx, oos_idx in folds:
        is_mask_arr = np.zeros(len(all_idx), dtype=bool)
        is_mask_arr[is_idx] = True

        best_y = None
        best_entries = None
        best_count = 0
        for lp, labels in label_cache:
            lab = labels.reindex(all_idx).fillna(0).values
            entries = np.where(lab != 0)[0]
            is_entries = entries[is_mask_arr[entries]]
            if len(is_entries) < 20:
                continue
            y_is = (lab[is_entries] == 1).astype(int)
            minority = min(y_is.mean(), 1 - y_is.mean())
            if minority < 0.15:
                continue
            if len(is_entries) > best_count:
                best_count = len(is_entries)
                best_entries = entries
                best_y = (lab == 1).astype(int)

        if best_y is None:
            continue

        X_is = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        X_oos = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        is_entries_fold = best_entries[is_mask_arr[best_entries]]

        if strategy_type == "xgboost":
            from xgboost import XGBClassifier

            xgb_p = cfg.get("xgb_params", XGB_PARAMS).copy()

            # Training samples: entry bars (setup occurred)
            local_entries = is_entries_fold - is_idx[0]
            y_entry = best_y[is_entries_fold]

            # Add background bars (no setup) to calibrate "flat" predictions
            # Sample ~equal number of non-entry IS bars as negative class
            all_is_local = np.arange(len(is_idx))
            non_entry = np.setdiff1d(all_is_local, local_entries)
            n_bg = min(len(non_entry), len(local_entries))
            if n_bg > 0:
                rng = np.random.RandomState(42)
                bg_idx = rng.choice(non_entry, size=n_bg, replace=False)
                train_idx = np.concatenate([local_entries, bg_idx])
                y_bg = np.zeros(n_bg, dtype=int)
                y_train = np.concatenate([y_entry, y_bg])
                sort_order = np.argsort(train_idx)
                train_idx = train_idx[sort_order]
                y_train = y_train[sort_order]
            else:
                train_idx = local_entries
                y_train = y_entry

            pos_c = y_train.sum()
            neg_c = len(y_train) - pos_c
            xgb_p["scale_pos_weight"] = neg_c / max(pos_c, 1)
            xgb_p["eval_metric"] = "logloss"
            model = XGBClassifier(**xgb_p)
            model.fit(X_is[train_idx], y_train)
            pred = model.predict_proba(X_oos)[:, 1]
            pos = _pred_to_position(pred, threshold)

        elif strategy_type in ("stacking", "tcn_stacking", "ae_stacking"):
            from research.ml.ensemble_stacking import StackedEnsemble

            entry_mask = np.zeros(len(is_idx), dtype=bool)
            is_start = is_idx[0]
            for e in is_entries_fold:
                local = e - is_start
                if 0 <= local < len(is_idx):
                    entry_mask[local] = True

            # Determine sequence model type
            seq_model = "tcn" if strategy_type == "tcn_stacking" else "lstm"

            # Autoencoder feature augmentation
            X_is_aug, X_oos_aug = X_is, X_oos
            if strategy_type == "ae_stacking":
                from research.ml.autoencoder_regime import (
                    extract_regime_features,
                    train_autoencoder,
                )

                ae = train_autoencoder(
                    X_is,
                    latent_dim=cfg.get("ae_latent_dim", 8),
                    epochs=cfg.get("ae_epochs", 100),
                )
                ae_is, kmeans = extract_regime_features(
                    ae,
                    X_is,
                    n_clusters=cfg.get("ae_clusters", 4),
                )
                ae_oos, _ = extract_regime_features(
                    ae,
                    X_oos,
                    n_clusters=cfg.get("ae_clusters", 4),
                    kmeans_model=kmeans,
                )
                X_is_aug = np.hstack([X_is, ae_is])
                X_oos_aug = np.hstack([X_oos, ae_oos])

            ensemble = StackedEnsemble(
                xgb_params=cfg.get("xgb_params", XGB_PARAMS),
                lstm_hidden=cfg.get("lstm_hidden", 32),
                lstm_lookback=cfg.get("lookback", 20),
                lstm_epochs=cfg.get("lstm_epochs", 30),
                n_nested_folds=cfg.get("n_nested_folds", 3),
                model_type=seq_model,
            )
            try:
                ensemble.fit(X_is_aug, best_y[is_idx], entry_mask)
                pred = ensemble.predict_proba(X_oos_aug)
            except Exception:
                continue
            pos = _pred_to_position(pred, threshold)

        elif strategy_type == "lstm_e2e":
            from research.ml.lstm_classifier import (
                predict_lstm_classifier,
                train_lstm_classifier,
            )

            y_is = best_y[is_idx]
            model = train_lstm_classifier(
                X_is,
                y_is,
                lookback=cfg.get("lookback", 20),
                hidden_dim=cfg.get("lstm_hidden", 32),
                epochs=cfg.get("lstm_epochs", 50),
                patience=7,
            )
            if model is None:
                continue
            pred = predict_lstm_classifier(model, X_oos, lookback=cfg.get("lookback", 20))
            pos = _pred_to_position(pred, threshold)
        else:
            continue

        oos_ret = ret_clean.iloc[oos_idx]
        stats = compute_signal_sharpe(pos, oos_ret, cost, bars_yr)
        fold_sharpes.append(stats["sharpe"])
        total_trades += stats.get("n_trades", 0)
        if "returns" in stats:
            all_oos_rets.append(stats["returns"])

        is_ret = ret_clean.iloc[is_idx]
        if strategy_type == "xgboost":
            is_pred = model.predict_proba(X_is)[:, 1]
        elif strategy_type in ("stacking", "tcn_stacking", "ae_stacking"):
            is_pred = ensemble.predict_proba(X_is_aug if strategy_type == "ae_stacking" else X_is)
        elif strategy_type == "lstm_e2e":
            is_pred = predict_lstm_classifier(model, X_is, cfg.get("lookback", 20))
        is_pos = _pred_to_position(is_pred, threshold)
        is_stats = compute_signal_sharpe(is_pos, is_ret, cost, bars_yr)
        if "returns" in is_stats:
            all_is_rets.append(is_stats["returns"])

    if not all_oos_rets:
        return dict(EMPTY_RESULT)

    stitched = pd.concat(all_oos_rets).sort_index()
    # Shared Sharpe helper — no nz-filter bias, explicit periods_per_year.
    oos_sharpe = _sharpe_from_rets(stitched, ann_factor=bars_yr)

    eq = (1 + stitched).cumprod()
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())

    if all_is_rets:
        is_stitched = pd.concat(all_is_rets).sort_index()
        is_sharpe = _sharpe_from_rets(is_stitched, ann_factor=bars_yr)
    else:
        is_sharpe = 0.0

    parity = (oos_sharpe / is_sharpe) if abs(is_sharpe) > 0.01 else 0.0
    pct_pos = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
    worst = min(fold_sharpes)

    result = {
        "sharpe": round(oos_sharpe, 4),
        "max_dd": round(max_dd, 4),
        "parity": round(parity, 4),
        "pct_positive": round(pct_pos, 4),
        "worst_fold": round(worst, 4),
        "n_trades": total_trades,
        "n_folds": len(fold_sharpes),
    }
    if return_raw:
        result["stitched_returns"] = stitched
    return result


# -- 2. Mean Reversion (VWAP + confluence regime) -----------------------------


def run_mean_reversion_wfo(instrument: str, cfg: dict, return_raw: bool = False) -> dict:
    """VWAP mean reversion with confluence regime filter."""
    from research.mean_reversion.run_confluence_regime_test import (
        build_atr_regime_mask,
        build_confluence_disagreement_mask,
        compute_vwap_deviation,
        load_h1,
    )
    from research.mean_reversion.run_confluence_regime_wfo import (
        TIER_GRIDS,
        run_mr_wfo,
    )

    tf = cfg.get("timeframe", "H1")
    if tf == "H1":
        df = load_h1(instrument)
    else:
        df = _load_daily(instrument)

    close = df["close"]
    vwap_anchor = cfg.get("vwap_anchor", 24)
    deviation = compute_vwap_deviation(close, anchor_period=vwap_anchor)

    # Build regime mask
    regime_filter = cfg.get("regime_filter", "conf_donchian_pos_20")
    if regime_filter == "no_filter":
        regime_mask = pd.Series(True, index=df.index)
    elif regime_filter == "atr_only":
        regime_mask = build_atr_regime_mask(df, threshold_pct=cfg.get("atr_gate_pct", 0.30))
    elif regime_filter.startswith("conf_"):
        sig_name = regime_filter[5:]  # strip "conf_"
        regime_mask = build_confluence_disagreement_mask(df, sig_name)
    else:
        regime_mask = build_atr_regime_mask(df)

    # Tier grid
    tier_grid_name = cfg.get("tier_grid", "standard")
    tiers_pct = TIER_GRIDS.get(tier_grid_name, TIER_GRIDS["standard"])

    is_bars = cfg.get("is_bars", 30000)
    oos_bars = cfg.get("oos_bars", 7500)

    r = run_mr_wfo(
        close,
        deviation,
        regime_mask,
        tiers_pct,
        is_bars=is_bars,
        oos_bars=oos_bars,
        spread_bps=cfg.get("spread_bps", 2.0),
        slippage_bps=cfg.get("slippage_bps", 1.0),
    )

    fold_df = r.get("fold_df", pd.DataFrame())
    if len(fold_df) == 0:
        return dict(EMPTY_RESULT)

    worst_fold = fold_df["oos_sharpe"].min() if "oos_sharpe" in fold_df.columns else -99.0
    # Parity from stitched OOS vs mean IS (avoids calibration artefact)
    oos_sh = r["stitched_sharpe"]
    mean_is = fold_df["is_sharpe"].mean() if "is_sharpe" in fold_df.columns else 0.0
    parity = oos_sh / mean_is if abs(mean_is) > 0.01 else 0.0

    result = {
        "sharpe": round(oos_sh, 4),
        "max_dd": round(r["stitched_dd_pct"] / 100, 4),
        "parity": round(parity, 4),
        "pct_positive": round(r["pct_positive"], 4),
        "worst_fold": round(worst_fold, 4),
        "n_trades": r["total_trades"],
        "n_folds": r["n_folds"],
    }
    if return_raw:
        result["stitched_returns"] = r.get("stitched_returns", pd.Series(dtype=float))
    return result


# -- 3. Trend Following (ETF trend with decel signals) ------------------------


def run_trend_following_wfo(instrument: str, cfg: dict) -> dict:
    """ETF trend-following strategy with rolling WFO."""
    df = _load_daily(instrument)
    close = df["close"]

    slow_ma_period = cfg.get("slow_ma", 200)
    ma_type = cfg.get("ma_type", "SMA")

    if ma_type.upper() == "EMA":
        slow_ma = close.ewm(span=slow_ma_period).mean()
    else:
        slow_ma = close.rolling(slow_ma_period).mean()

    # Build position signal
    above_ma = (close > slow_ma).astype(float)
    signal = above_ma.shift(1).fillna(0)  # no look-ahead

    cost_bps = cfg.get("cost_bps", 2.0)
    daily_ret = close.pct_change().fillna(0)
    # Pre-compute transitions globally to avoid cost erasure at fold boundaries
    transitions = signal.diff().abs().fillna(0)
    cost_series = transitions * cost_bps / 10_000

    def build_returns(df_slice, cfg_inner):
        idx = df_slice.index
        sig = signal.reindex(idx).fillna(0)
        ret = daily_ret.reindex(idx).fillna(0)
        cost = cost_series.reindex(idx).fillna(0)
        strat_ret = sig * ret - cost
        cfg_inner["_last_signal"] = sig.values
        return strat_ret

    is_days = cfg.get("is_days", 504)
    oos_days = cfg.get("oos_days", 126)

    # Drop warmup period (need slow_ma_period bars)
    df_valid = df.iloc[slow_ma_period + 1 :]

    return generic_rolling_wfo(build_returns, df_valid, cfg, is_days=is_days, oos_days=oos_days)


# -- 4. Cross-Asset Momentum (bond -> equity/gold) ----------------------------


def run_cross_asset_wfo(instrument: str, cfg: dict, return_raw: bool = False) -> dict:
    """Bond momentum -> target instrument WFO."""
    from research.cross_asset.run_bond_equity_wfo import load_daily, run_bond_wfo

    bond_sym = cfg.get("bond", "IEF")
    target_sym = instrument

    bond_close = load_daily(bond_sym)
    target_close = load_daily(target_sym)

    r = run_bond_wfo(
        bond_close,
        target_close,
        lookback=cfg.get("lookback", 20),
        hold_days=cfg.get("hold_days", 20),
        threshold=cfg.get("threshold", 0.50),
        is_days=cfg.get("is_days", 504),
        oos_days=cfg.get("oos_days", 126),
        spread_bps=cfg.get("spread_bps", 5.0),
    )

    if "error" in r:
        return dict(EMPTY_RESULT)

    fold_df = r.get("fold_df", pd.DataFrame())
    if len(fold_df) == 0:
        return dict(EMPTY_RESULT)

    worst_fold = fold_df["oos_sharpe"].min() if "oos_sharpe" in fold_df.columns else -99.0
    oos_sh = r["stitched_sharpe"]
    mean_is = fold_df["is_sharpe"].mean() if "is_sharpe" in fold_df.columns else 0.0
    parity = oos_sh / mean_is if abs(mean_is) > 0.01 else 0.0

    result = {
        "sharpe": round(oos_sh, 4),
        "max_dd": round(r["stitched_dd_pct"] / 100, 4),
        "parity": round(parity, 4),
        "pct_positive": round(r["pct_positive"], 4),
        "worst_fold": round(worst_fold, 4),
        "n_trades": r["total_trades"],
        "n_folds": r["n_folds"],
    }
    if return_raw:
        result["stitched_returns"] = r.get("stitched_returns", pd.Series(dtype=float))
    return result


# -- 5. Gold Macro (3-component signal) ---------------------------------------


def run_gold_macro_wfo(instrument: str, cfg: dict) -> dict:
    """Gold macro composite signal with rolling WFO."""
    from research.gold_macro.run_backtest import (
        build_composite_signal,
        load_data,
    )

    data = load_data()
    sig_df = build_composite_signal(
        data["GLD"],
        data["TIP"],
        data["TLT"],
        _load_daily("DXY") if "DXY" not in data else data["DXY"],
        real_rate_window=cfg.get("real_rate_window", 20),
        dollar_window=cfg.get("dollar_window", 20),
        slow_ma=cfg.get("slow_ma", 200),
    )

    close = sig_df["gld"]
    daily_ret = close.pct_change().fillna(0)
    signal = sig_df["signal"].fillna(0)
    cost_bps = cfg.get("cost_bps", 5.0)
    # Pre-compute transitions globally to avoid cost erasure at fold boundaries
    transitions_gm = signal.diff().abs().fillna(0)
    cost_gm = transitions_gm * cost_bps / 10_000

    def build_returns(df_slice, cfg_inner):
        idx = df_slice.index
        sig = signal.reindex(idx).fillna(0)
        ret = daily_ret.reindex(idx).fillna(0)
        cost = cost_gm.reindex(idx).fillna(0)
        strat_ret = sig * ret - cost
        cfg_inner["_last_signal"] = sig.values
        return strat_ret

    is_days = cfg.get("is_days", 504)
    oos_days = cfg.get("oos_days", 126)

    return generic_rolling_wfo(build_returns, sig_df, cfg, is_days=is_days, oos_days=oos_days)


# -- 6. FX Carry (carry + trend + VIX filter) ---------------------------------


def run_fx_carry_wfo(instrument: str, cfg: dict) -> dict:
    """FX carry strategy with rolling WFO."""
    from research.fx_carry.run_backtest import (
        build_signal,
        load_vix,
    )
    from research.fx_carry.run_backtest import (
        load_data as load_fx_data,
    )

    df = load_fx_data(instrument)
    vix = load_vix()

    sig_df = build_signal(
        df,
        carry_direction=cfg.get("carry_direction", 1),
        sma_period=cfg.get("sma_period", 50),
    )

    close = sig_df["close"]
    daily_ret = close.pct_change().fillna(0)
    signal = sig_df["signal"].fillna(0)
    cost_bps = cfg.get("spread_bps", 3.0) + cfg.get("slippage_bps", 1.0)

    # Vol targeting. Daily bars -> periods_per_year=252. Route through the
    # shared ``ewm_vol`` so the same math is used in every backtest and live.
    from titan.research.metrics import BARS_PER_YEAR as _BPY
    from titan.research.metrics import ewm_vol as _ewm_vol

    vol_target = cfg.get("vol_target_pct", 0.08)
    realized_vol = (
        _ewm_vol(
            daily_ret,
            lam=(20 - 1) / (20 + 1),  # span=20 -> RiskMetrics lambda
            periods_per_year=_BPY["D"],
        )
        .clip(lower=1e-4)
        .fillna(1.0)
    )
    vol_scale = (vol_target / realized_vol).clip(upper=1.5)

    # VIX halving
    if vix is not None:
        vix_thresh = cfg.get("vix_halve_threshold", 25.0)
        vix_aligned = vix.reindex(sig_df.index, method="ffill")
        vix_mask = vix_aligned > vix_thresh
        vol_scale = vol_scale.where(~vix_mask, vol_scale * 0.5)

    # Pre-compute globally to avoid boundary loss at fold edges
    vol_scale_lagged = vol_scale.shift(1).fillna(1.0)
    transitions_fx = signal.diff().abs().fillna(0)
    cost_fx = transitions_fx * cost_bps / 10_000

    def build_returns(df_slice, cfg_inner):
        idx = df_slice.index
        sig = signal.reindex(idx).fillna(0)
        ret = daily_ret.reindex(idx).fillna(0)
        vs = vol_scale_lagged.reindex(idx).fillna(1.0)
        cost = cost_fx.reindex(idx).fillna(0)
        strat_ret = sig * ret * vs - cost
        cfg_inner["_last_signal"] = sig.values
        return strat_ret

    is_days = cfg.get("is_days", 504)
    oos_days = cfg.get("oos_days", 126)

    return generic_rolling_wfo(build_returns, sig_df, cfg, is_days=is_days, oos_days=oos_days)


# -- 7. Pairs Trading (spread mean reversion) ---------------------------------


def run_pairs_trading_wfo(instrument: str, cfg: dict) -> dict:
    """Pairs trading with rolling WFO."""
    pair_b = cfg.get("pair_b", "TXN")

    df_a = _load_daily(instrument)
    df_b = _load_daily(pair_b)

    close_a = df_a["close"]
    close_b = df_b["close"]

    # Align
    common = close_a.index.intersection(close_b.index)
    close_a = close_a.reindex(common)
    close_b = close_b.reindex(common)

    entry_z = cfg.get("entry_z", 2.0)
    exit_z = cfg.get("exit_z", 0.5)
    max_z = cfg.get("max_z", 4.0)
    # refit_window reserved for future walk-forward beta refit plumbing.
    cfg.get("refit_window", 126)
    is_days = cfg.get("is_days", 504)
    oos_days = cfg.get("oos_days", 126)
    n = len(common)

    if n < is_days + oos_days:
        return dict(EMPTY_RESULT)

    fold_sharpes = []
    all_oos_rets = []
    all_is_sharpes = []
    total_trades = 0
    oos_start = is_days

    while oos_start + oos_days <= n:
        is_start = max(0, oos_start - is_days)

        a_is = close_a.iloc[is_start:oos_start]
        b_is = close_b.iloc[is_start:oos_start]
        a_oos = close_a.iloc[oos_start : oos_start + oos_days]
        b_oos = close_b.iloc[oos_start : oos_start + oos_days]

        # IS: estimate beta and spread stats
        from numpy.polynomial.polynomial import polyfit

        beta = float(polyfit(b_is.values, a_is.values, 1)[1])
        spread_is = a_is.values - beta * b_is.values
        mu = float(np.mean(spread_is))
        sigma = float(np.std(spread_is))
        if sigma < 1e-8:
            oos_start += oos_days
            continue

        # OOS: apply frozen IS stats
        spread_oos = a_oos.values - beta * b_oos.values
        z_oos = (spread_oos - mu) / sigma

        # Position logic
        pos = np.zeros(len(z_oos))
        for i in range(1, len(z_oos)):
            if pos[i - 1] == 0:
                if z_oos[i] > entry_z:
                    pos[i] = -1.0  # short spread
                elif z_oos[i] < -entry_z:
                    pos[i] = 1.0  # long spread
            elif pos[i - 1] > 0:  # long spread
                if z_oos[i] > -exit_z or abs(z_oos[i]) > max_z:
                    pos[i] = 0.0
                else:
                    pos[i] = pos[i - 1]
            elif pos[i - 1] < 0:  # short spread
                if z_oos[i] < exit_z or abs(z_oos[i]) > max_z:
                    pos[i] = 0.0
                else:
                    pos[i] = pos[i - 1]

        # Returns: use pct_change to preserve array length (no off-by-one)
        oos_close = close_a.iloc[oos_start : oos_start + oos_days]
        target_ret_oos = oos_close.pct_change().fillna(0).values
        oos_rets = pos * target_ret_oos

        is_close = close_a.iloc[is_start:oos_start]
        target_ret_is = is_close.pct_change().fillna(0).values
        is_sh = _sharpe_from_rets(target_ret_is)
        oos_sh = _sharpe_from_rets(oos_rets)
        fold_sharpes.append(oos_sh)
        all_is_sharpes.append(is_sh)
        all_oos_rets.append(oos_rets)
        total_trades += int(np.sum(np.abs(np.diff(pos)) > 0))

        oos_start += oos_days

    if not fold_sharpes:
        return dict(EMPTY_RESULT)

    stitched = np.concatenate(all_oos_rets)
    oos_sharpe = _sharpe_from_rets(stitched)
    max_dd = _max_dd_from_rets(stitched)
    avg_is = float(np.mean(all_is_sharpes)) if all_is_sharpes else 0.0
    parity = oos_sharpe / avg_is if abs(avg_is) > 0.01 else 0.0
    pct_pos = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
    worst = min(fold_sharpes)

    return {
        "sharpe": round(oos_sharpe, 4),
        "max_dd": round(max_dd, 4),
        "parity": round(parity, 4),
        "pct_positive": round(pct_pos, 4),
        "worst_fold": round(worst, 4),
        "n_trades": total_trades,
        "n_folds": len(fold_sharpes),
    }


# -- 8. Portfolio (multi-strategy weighted combo) ------------------------------


def run_portfolio_wfo(instrument: str, cfg: dict) -> dict:
    """Run multiple sub-strategies and combine returns by weight."""
    sub_cfgs = cfg.get("strategies", [])
    if not sub_cfgs:
        return dict(EMPTY_RESULT)

    sub_results = []
    for sub in sub_cfgs:
        strat = sub.get("strategy", "stacking")
        runner = STRATEGY_REGISTRY.get(strat)
        if runner is None or runner is run_portfolio_wfo:
            continue
        inst = sub.get("instruments", [instrument])
        if isinstance(inst, list):
            inst = inst[0]
        try:
            result = runner(inst, sub)
            weight = sub.get("weight", 1.0 / len(sub_cfgs))
            sub_results.append((weight, result))
        except Exception:
            continue

    if not sub_results:
        return dict(EMPTY_RESULT)

    # Weighted average of metrics
    total_weight = sum(w for w, _ in sub_results)
    if total_weight < 1e-8:
        return dict(EMPTY_RESULT)

    avg_sharpe = sum(w * r["sharpe"] for w, r in sub_results) / total_weight
    worst_dd = min(r["max_dd"] for _, r in sub_results)
    avg_parity = sum(w * r["parity"] for w, r in sub_results) / total_weight
    avg_pos = sum(w * r["pct_positive"] for w, r in sub_results) / total_weight
    worst_fold = min(r["worst_fold"] for _, r in sub_results)
    total_trades = sum(r["n_trades"] for _, r in sub_results)
    total_folds = sum(r["n_folds"] for _, r in sub_results)

    return {
        "sharpe": round(avg_sharpe, 4),
        "max_dd": round(worst_dd, 4),
        "parity": round(avg_parity, 4),
        "pct_positive": round(avg_pos, 4),
        "worst_fold": round(worst_fold, 4),
        "n_trades": total_trades,
        "n_folds": total_folds,
    }


# ==============================================================================
#  STRATEGY REGISTRY
# ==============================================================================

STRATEGY_REGISTRY = {
    "xgboost": run_ml_wfo,
    "stacking": run_ml_wfo,
    "tcn_stacking": run_ml_wfo,
    "ae_stacking": run_ml_wfo,
    "lstm_e2e": run_ml_wfo,
    "mean_reversion": run_mean_reversion_wfo,
    "trend_following": run_trend_following_wfo,
    "cross_asset": run_cross_asset_wfo,
    "gold_macro": run_gold_macro_wfo,
    "fx_carry": run_fx_carry_wfo,
    "pairs_trading": run_pairs_trading_wfo,
    "portfolio": run_portfolio_wfo,
}


# ==============================================================================
#  MAIN
# ==============================================================================


def load_experiment() -> dict:
    """Dynamically import experiment.py and call configure()."""
    spec = importlib.util.spec_from_file_location("experiment", EXPERIMENT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.configure()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate experiment.py")
    parser.add_argument("--timeout", type=int, default=300, help="Max seconds")
    args = parser.parse_args()

    t0 = time.time()

    try:
        cfg = load_experiment()
    except Exception as e:
        print("SCORE: -99.0")
        print(f"ERROR: Failed to load experiment.py: {e}")
        return

    strategy_type = cfg.get("strategy", "stacking")
    instruments = cfg.get("instruments", ["QQQ"])
    if isinstance(instruments, str):
        instruments = [instruments]

    # Get the runner
    runner = STRATEGY_REGISTRY.get(strategy_type)
    if runner is None:
        print("SCORE: -99.0")
        print(f"ERROR: Unknown strategy type: {strategy_type}")
        return

    all_results = []
    for inst in instruments:
        try:
            elapsed = time.time() - t0
            if elapsed > args.timeout:
                print("SCORE: -99.0")
                print(f"ERROR: Timeout after {elapsed:.0f}s")
                return

            result = runner(inst, cfg)
            result["instrument"] = inst
            all_results.append(result)
        except Exception as e:
            print(f"# {inst}: ERROR {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_results:
        print("SCORE: -99.0")
        print("ERROR: No results")
        return

    # Aggregate across instruments
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    worst_dd = min(r["max_dd"] for r in all_results)
    avg_parity = np.mean([r["parity"] for r in all_results])
    avg_pos = np.mean([r["pct_positive"] for r in all_results])
    worst_fold = min(r["worst_fold"] for r in all_results)
    total_trades = sum(r["n_trades"] for r in all_results)
    total_folds = sum(r["n_folds"] for r in all_results)

    score = composite_score(
        avg_sharpe,
        worst_dd,
        avg_parity,
        avg_pos,
        worst_fold,
        total_trades,
        total_folds,
    )

    elapsed = time.time() - t0

    # Output (parseable by agent)
    print(f"SCORE: {score:.4f}")
    print(f"SHARPE: {avg_sharpe:+.4f}")
    print(f"MAX_DD: {worst_dd:.1%}")
    print(f"PARITY: {avg_parity:.4f}")
    print(f"POS_FOLDS: {avg_pos:.1%}")
    print(f"TRADES: {total_trades}")
    print(f"WORST_FOLD: {worst_fold:.4f}")
    print(f"N_FOLDS: {total_folds}")
    print(f"INSTRUMENTS: {','.join(instruments)}")
    print(f"STRATEGY: {strategy_type}")
    print(f"ELAPSED: {elapsed:.1f}s")

    # Per-instrument breakdown
    for r in all_results:
        print(
            f"  {r['instrument']}: Sharpe={r['sharpe']:+.3f}"
            f" DD={r['max_dd']:.1%} Par={r['parity']:.2f}"
            f" Pos={r['pct_positive']:.0%} Folds={r['n_folds']}"
        )

    # Append to results.tsv
    try:
        import hashlib

        cfg_str = str(cfg)
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        desc = cfg.get("description", f"{strategy_type} on {','.join(instruments)}")

        header_needed = not RESULTS_FILE.exists()
        with open(RESULTS_FILE, "a") as f:
            if header_needed:
                f.write(
                    "timestamp\thash\tscore\tsharpe\tmax_dd\t"
                    "parity\tpos_folds\ttrades\tinstruments\t"
                    "description\n"
                )
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{cfg_hash}\t"
                f"{score:.4f}\t{avg_sharpe:+.4f}\t{worst_dd:.1%}\t"
                f"{avg_parity:.4f}\t{avg_pos:.0%}\t{total_trades}\t"
                f"{','.join(instruments)}\t{desc}\n"
            )
    except Exception:
        pass  # Don't fail on logging


if __name__ == "__main__":
    main()
