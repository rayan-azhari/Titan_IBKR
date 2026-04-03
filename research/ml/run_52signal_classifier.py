"""run_52signal_classifier.py -- ML Signal Discovery using 52 IC features.

Uses all 52 signals from the IC signal analysis (Groups A-G) plus 12 daily-equivalent
MA features as inputs to an XGBoost classifier. Target labels come from the Triple
Barrier Method (TBM): +1 = profitable long entry, -1 = stop-loss hit.

Walk-Forward validation (rolling IS=2yr, OOS=6mo) ensures no lookahead. Each fold
re-trains the model on IS data and evaluates on untouched OOS data.

Instruments: currencies, gold, indices only (no individual stocks).

Usage
-----
    # Single instrument (quick test):
    uv run python research/ml/run_52signal_classifier.py --instrument EUR_USD

    # All target instruments:
    uv run python research/ml/run_52signal_classifier.py

    # Custom TBM parameters:
    uv run python research/ml/run_52signal_classifier.py --pt-mult 3.0 --sl-mult 1.5

Output
------
    Console: per-fold WFO table, feature importance, stitched equity
    CSV:     .tmp/reports/ml_52sig_{instrument}_{tf}_{timestamp}.csv
    Joblib:  models/ml_52sig_{instrument}_{tf}.joblib (if stitched Sharpe >= 1.0)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from numba import njit  # noqa: E402

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
    build_all_signals,
)
from research.ml.build_tbm_labels import _compute_atr  # noqa: E402
from titan.strategies.ml.features import atr as feat_atr  # noqa: E402
from titan.strategies.ml.features import ema, sma  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# XGBoost hyperparameters (shallow trees to prevent overfit on noisy financial data)
XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "random_state": 42,
    "verbosity": 0,
}

# Trailing-stop labeler defaults
TRAIL_STOP_MULT: float = 2.0  # initial + trailing stop distance in ATR
TRAIL_MAX_HOLD_H1: int = 120  # 120 H1 bars = 5 days (ride the trend)
TRAIL_MAX_HOLD_D: int = 30  # 30 daily bars = ~6 weeks
TRAIL_MAX_HOLD_M5: int = 1440  # 1440 M5 bars = 5 days FX (288 bars/day x 5)

# Regression -> signal conversion: predict R-multiple, trade if |pred| > threshold
SIGNAL_THRESHOLD: float = 0.5  # minimum predicted R-multiple to take a trade

# WFO configuration (bars)
IS_RATIO_BARS = {
    "H1": 2 * 252 * 24,  # ~12,096 bars (2 years)
    "D": 2 * 252,  # ~504 bars (2 years)
    "M5": 252 * 288,  # ~72,576 bars (1 year FX — shorter IS for more folds)
}
OOS_RATIO_BARS = {
    "H1": 252 * 24 // 2,  # ~3,024 bars (6 months)
    "D": 126,  # ~126 bars (6 months)
    "M5": 252 * 288 // 4,  # ~18,144 bars (3 months FX)
}

# MA period scaling: daily-equivalent periods by timeframe
# 1 day = 24 H1 bars = 288 M5 bars (FX 24h)
MA_SCALE = {
    "H1": {"fast": 20, "slow": 50, "trend": 1200, "longterm": 4800},
    "M5": {"fast": 240, "slow": 600, "trend": 14400, "longterm": 57600},
    "D": {"fast": 2, "slow": 5, "trend": 50, "longterm": 200},
}

# Cost profiles (bps per fill, from phase3_backtest.py)
COST_BPS = {
    "fx": 1.0,  # 0.5 spread + 0.5 slippage
    "fx_cross": 1.5,
    "gold": 3.0,
    "index": 2.0,
}

# Target instruments
TARGET_INSTRUMENTS = {
    "EUR_USD": ("H1", "fx"),
    "GBP_USD": ("H1", "fx"),
    "USD_JPY": ("H1", "fx"),
    "AUD_JPY": ("H1", "fx_cross"),
    "AUD_USD": ("H1", "fx_cross"),
    "USD_CHF": ("H1", "fx"),
    "GLD": ("H1", "gold"),
    "SPY": ("H1", "index"),
    "QQQ": ("H1", "index"),
}

# Validation gates
MIN_STITCHED_SHARPE: float = 1.0
MIN_PCT_POSITIVE_FOLDS: float = 0.70
MIN_PARITY: float = 0.50
MIN_WORST_FOLD_SHARPE: float = -2.0
MAX_CONSEC_NEG: int = 2


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ADX regime, HMM state, and ATR percentile features.

    These are rolling/causal features — no lookahead. The HMM is fit on the
    first 70% of bars (IS) and predicted on all bars, matching phase0_regime.py.
    ADX regime and ATR percentile are purely rolling — always causal.

    Returns DataFrame with 5 columns:
      - adx_regime_ranging (1 if ADX<20, else 0)
      - adx_regime_trending (1 if ADX>25, else 0)
      - hmm_state (0 or 1, from 2-state Gaussian HMM on [log_ret, rvol20])
      - atr_pct_rank (rolling 252-bar percentile rank of ATR14, 0.0-1.0)
      - vol_regime_low (1 if atr_pct_rank < 0.30, else 0)
    """
    from titan.strategies.ml.features import adx as compute_adx

    regime = pd.DataFrame(index=df.index)
    close = df["close"]

    # 1. ADX regime (categorical -> binary dummies)
    adx_val = compute_adx(df, 14)
    regime["adx_regime_ranging"] = (adx_val < 20).astype(float)
    regime["adx_regime_trending"] = (adx_val > 25).astype(float)

    # 2. HMM state (2-state Gaussian on [log_ret, realized_vol_20])
    # Fit on IS (first 70%) to prevent lookahead; predict on all bars.
    log_ret = np.log(close).diff().fillna(0.0)
    rvol20 = log_ret.rolling(20).std().bfill()

    is_n = max(100, int(len(df) * 0.70))
    # Normalise using IS statistics only
    ret_mu, ret_std = float(log_ret.iloc[:is_n].mean()), float(log_ret.iloc[:is_n].std())
    vol_mu, vol_std = float(rvol20.iloc[:is_n].mean()), float(rvol20.iloc[:is_n].std())
    ret_z = ((log_ret - ret_mu) / ret_std) if ret_std > 1e-10 else log_ret * 0
    vol_z = ((rvol20 - vol_mu) / vol_std) if vol_std > 1e-10 else rvol20 * 0

    X_full = np.column_stack([ret_z.values, vol_z.values])
    X_is = X_full[:is_n]

    try:
        from hmmlearn.hmm import GaussianHMM

        hmm = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        hmm.fit(X_is)
        hmm_states = hmm.predict(X_full)
        regime["hmm_state"] = hmm_states.astype(float)
    except ImportError:
        print("  [WARN] hmmlearn not installed; hmm_state set to 0. Run: uv add hmmlearn")
        regime["hmm_state"] = 0.0

    # 3. ATR percentile rank (rolling 252-bar, causal)
    atr_val = feat_atr(df)
    regime["atr_pct_rank"] = atr_val.rolling(252, min_periods=50).rank(pct=True)
    regime["vol_regime_low"] = (regime["atr_pct_rank"] < 0.30).astype(float)

    return regime


def _load_cross_asset_daily() -> dict[str, pd.Series]:
    """Load daily close series for cross-asset features (VIX, GLD, SPY, QQQ, FTSE, DAX).

    Returns dict of instrument -> daily close pd.Series. Missing data returns empty.
    Each series is normalised to UTC midnight for alignment.
    """
    cross = {}
    for inst in ["^VIX", "GLD", "SPY", "QQQ", "^FTSE", "^GDAXI"]:
        try:
            d = _load_ohlcv(inst, "D")
            s = d["close"].copy()
            s.index = s.index.normalize()
            cross[inst] = s
        except FileNotFoundError:
            pass
    return cross


def build_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Build pruned feature set: 52 IC signals + 7 MA + 5 regime + 3 VIX + 1 cal + 4 momentum = ~72.

    Pruning rationale (from 4-instrument importance analysis):
    - Removed correlated MA duplicates: keep ma_spread_weekly + ema_spread_50_200,
      drop wma_spread_50_200 (r>0.95 with ema variant), ma_spread_intra (weak),
      ema_fast_slow (binary, less info than spread)
    - VIX: keep vix_sma20 (top 15 on all 4), vix_below_15 (#4 on DAX),
      vol_risk_premium. Drop vix_level (correlated with sma20), vix_pct_rank,
      vix_roc5/21 (noisy), vix_above_30 (rare event, few samples).
    - Cross-asset: removed entirely -- added noise, no instrument showed
      cross-asset returns in top 20 features.
    - Calendar: keep month only (#1-3 on all instruments). Drop day_of_week,
      is_month_start, is_quarter_end, is_january (weak or subsumed by month).
    - Momentum: keep ret_lag_126, ret_lag_252, mom_accel_21_252. Drop shorter
      lags (1,5,10,21,63) which are subsumed by the 52 IC signals (ROC 3/10/20/60).
    """
    window_1y = _get_annual_bars(tf)

    # 1. Core 52 signals from IC analysis
    signals_52 = build_all_signals(df, window_1y)

    # 2. Pruned MA features (7 features -- top performers, no redundancy)
    close = df["close"]
    ma_features = pd.DataFrame(index=df.index)

    scale = MA_SCALE.get(tf, MA_SCALE["H1"])
    p_slow = scale["slow"]
    p_trend = scale["trend"]
    p_lt = scale["longterm"]

    max_period = int(len(df) * 0.4)
    p_trend = min(p_trend, max_period)
    p_lt = min(p_lt, max_period)

    sma_trend = sma(close, p_trend)
    sma_lt = sma(close, p_lt)
    ema_trend = ema(close, p_trend)
    ema_lt = ema(close, p_lt)
    ema_slow = ema(close, p_slow)

    def _sd(a, b):
        return a / b.where(b.abs() > 1e-10, np.nan)

    ma_features["ma_spread_weekly"] = _sd(sma_trend - sma_lt, sma_lt)  # #1 everywhere
    ma_features["ema_spread_50_200"] = _sd(ema_trend - ema_lt, ema_lt)  # #1-2 everywhere
    ma_features["ma_spread_daily"] = _sd(ema_slow - sma_trend, sma_trend)  # top 10
    ma_features["price_vs_longterm"] = _sd(close - sma_lt, sma_lt)  # top 10
    ma_features["sma_50_200_cross"] = (sma_trend > sma_lt).astype(float)  # top 15
    atr_val = feat_atr(df)
    safe_atr = atr_val.where(atr_val > 1e-10, np.nan)
    ma_features["trend_strength"] = (sma_trend - sma_lt).abs() / safe_atr  # top 10
    vol_window = min(window_1y, max_period)
    ma_features["dist_from_200d"] = _sd(
        close - sma_lt,
        close.rolling(vol_window).std().where(close.rolling(vol_window).std() > 1e-10, np.nan),
    )

    # 3. Regime features (5 -- all proven useful)
    regime_features = _compute_regime_features(df)

    # 4. VIX features (3 -- pruned to top performers only)
    vix_features = pd.DataFrame(index=df.index)
    cross_data = _load_cross_asset_daily()

    if "^VIX" in cross_data:
        vix = cross_data["^VIX"]
        idx_norm = df.index.normalize()
        vix_aligned = pd.Series(vix.reindex(idx_norm).ffill().values, index=df.index)
        vix_features["vix_sma20"] = vix_aligned.rolling(20).mean()
        vix_features["vix_below_15"] = (vix_aligned < 15).astype(float)
        rvol20 = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        vix_features["vol_risk_premium"] = vix_aligned - rvol20

    # 5. Calendar (1 -- month was #1-3 on all instruments)
    cal_features = pd.DataFrame(index=df.index)
    cal_features["month"] = df.index.month / 12.0

    # 6. Extended momentum (4 -- longer horizons not in 52 IC signals)
    mom_features = pd.DataFrame(index=df.index)
    mom_features["ret_lag_126"] = close.pct_change(126)
    mom_features["ret_lag_252"] = close.pct_change(252)
    mom_features["mom_accel_21_252"] = close.pct_change(21) - close.pct_change(252)
    mom_features["mom_accel_5_63"] = close.pct_change(5) - close.pct_change(63)

    # Combine and shift 1 bar
    all_features = pd.concat(
        [signals_52, ma_features, regime_features, vix_features, cal_features, mom_features],
        axis=1,
    )
    all_features = all_features.shift(1)

    return all_features


# ---------------------------------------------------------------------------
# Trailing-Stop Labeler (replaces fixed-barrier TBM)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _trailing_stop_kernel(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    stop_mult: float,
    max_holding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-stop labeler for both long and short entries.

    For each bar t, simulates:
      LONG:  Enter at close[t]. Initial stop = close[t] - stop_mult*ATR.
             Trail stop up: stop = max(stop, highest_high - stop_mult*ATR).
             Exit when low hits stop or max_holding bars reached.
             P&L = (exit_price - entry) / ATR  (in R-multiples).

      SHORT: Enter at close[t]. Initial stop = close[t] + stop_mult*ATR.
             Trail stop down: stop = min(stop, lowest_low + stop_mult*ATR).
             Exit when high hits stop or max_holding bars reached.
             P&L = (entry - exit_price) / ATR  (in R-multiples).

    Returns:
        long_r:  float64 array — P&L in R-multiples for a long entry at each bar.
        short_r: float64 array — P&L in R-multiples for a short entry at each bar.
        NaN where ATR is invalid (warmup).
    """
    n = len(close)
    long_r = np.full(n, np.nan, dtype=np.float64)
    short_r = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        if np.isnan(atr[t]) or atr[t] <= 0.0:
            continue

        entry = close[t]
        risk = atr[t] * stop_mult  # 1R = this distance

        # --- Long side ---
        stop_long = entry - risk
        hwm = entry  # high-water mark for trailing
        exit_long = entry  # default if max_holding expires immediately
        for i in range(1, max_holding + 1):
            idx = t + i
            if idx >= n:
                # Use last available close
                exit_long = close[n - 1] if t + 1 < n else entry
                break
            # Update HWM with this bar's high
            if high[idx] > hwm:
                hwm = high[idx]
                # Trail stop up (never move down)
                new_stop = hwm - risk
                if new_stop > stop_long:
                    stop_long = new_stop
            # Check if stopped out (low touches stop)
            if low[idx] <= stop_long:
                exit_long = stop_long
                break
            exit_long = close[idx]  # still holding
        long_r[t] = (exit_long - entry) / (atr[t]) if atr[t] > 0 else 0.0

        # --- Short side ---
        stop_short = entry + risk
        lwm = entry  # low-water mark for trailing
        exit_short = entry
        for i in range(1, max_holding + 1):
            idx = t + i
            if idx >= n:
                exit_short = close[n - 1] if t + 1 < n else entry
                break
            # Update LWM with this bar's low
            if low[idx] < lwm:
                lwm = low[idx]
                new_stop = lwm + risk
                if new_stop < stop_short:
                    stop_short = new_stop
            # Check if stopped out (high touches stop)
            if high[idx] >= stop_short:
                exit_short = stop_short
                break
            exit_short = close[idx]
        short_r[t] = (entry - exit_short) / (atr[t]) if atr[t] > 0 else 0.0

    return long_r, short_r


def compute_trailing_labels(
    df: pd.DataFrame,
    stop_mult: float = 2.0,
    max_holding: int = 120,
) -> tuple[pd.Series, pd.Series]:
    """Compute trailing-stop labels for each bar (long and short R-multiples).

    Args:
        df: OHLCV DataFrame.
        stop_mult: ATR multiplier for initial/trailing stop distance.
        max_holding: Maximum bars to hold before forced exit.

    Returns:
        (long_r, short_r): two Series of float64 P&L in R-multiples.
        Positive = profitable entry, negative = stopped out at a loss.
    """
    atr_arr = _compute_atr(df, period=14)
    long_r, short_r = _trailing_stop_kernel(
        df["close"].values.astype(np.float64),
        df["high"].values.astype(np.float64),
        df["low"].values.astype(np.float64),
        atr_arr.astype(np.float64),
        stop_mult,
        max_holding,
    )
    return (
        pd.Series(long_r, index=df.index, name="long_r"),
        pd.Series(short_r, index=df.index, name="short_r"),
    )


# ---------------------------------------------------------------------------
# Regime + Pullback Labeler (v3)
# ---------------------------------------------------------------------------
#
# Step 1 — Trend regime (causal, no lookahead):
#   BULL: SMA(50) > SMA(200) AND MACD histogram > 0
#   BEAR: SMA(50) < SMA(200) AND MACD histogram < 0
#   NEUTRAL: otherwise
#
# Step 2 — Within regime, find pullback entries (forward-looking, labels only):
#   In BULL: label +1 when RSI(14) dipped below rsi_oversold AND
#            forward N-bar return > min_confirm_pct
#   In BEAR: label -1 when RSI(14) spiked above rsi_overbought AND
#            forward N-bar return < -min_confirm_pct
#   All other bars: label 0 (hold current position or flat)

# Default parameters
REGIME_SMA_FAST: int = 50
REGIME_SMA_SLOW: int = 200
REGIME_RSI_PERIOD: int = 14
REGIME_RSI_OVERSOLD: float = 45.0  # pullback in uptrend (wider to get more labels)
REGIME_RSI_OVERBOUGHT: float = 55.0  # rally in downtrend
REGIME_CONFIRM_BARS: int = 10  # forward bars to check confirmation
REGIME_CONFIRM_PCT: float = 0.005  # 0.5% min forward return (single horizon only)


def compute_regime_pullback_labels(
    df: pd.DataFrame,
    sma_fast: int = REGIME_SMA_FAST,
    sma_slow: int = REGIME_SMA_SLOW,
    rsi_period: int = REGIME_RSI_PERIOD,
    rsi_oversold: float = REGIME_RSI_OVERSOLD,
    rsi_overbought: float = REGIME_RSI_OVERBOUGHT,
    confirm_bars: int = REGIME_CONFIRM_BARS,
    confirm_pct: float = REGIME_CONFIRM_PCT,
) -> tuple[pd.Series, pd.Series]:
    """Compute Regime + Pullback labels for training.

    Returns:
        (labels, regime): two Series.
        labels: int8 -- +1 (confirmed long pullback), -1 (confirmed short rally), 0 (hold)
        regime: int8 -- +1 (bull), -1 (bear), 0 (neutral)
    """
    from titan.strategies.ml.features import rsi as compute_rsi

    close = df["close"]
    n = len(close)

    # Step 1: Trend regime (all causal / no lookahead)
    sma_f = sma(close, sma_fast)
    sma_s = sma(close, sma_slow)

    # MACD histogram (12/26/9 standard)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    bull = (sma_f > sma_s) & (macd_hist > 0)
    bear = (sma_f < sma_s) & (macd_hist < 0)
    regime = pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=df.index, dtype=np.int8)

    # Step 2: RSI pullback detection (causal)
    rsi_val = compute_rsi(close, rsi_period)

    bull_pullback = bull & (rsi_val < rsi_oversold)
    bear_rally = bear & (rsi_val > rsi_overbought)

    # Step 3: Forward return confirmation (for LABELS ONLY — single horizon)
    fwd_ret = close.pct_change(confirm_bars).shift(-confirm_bars)

    long_confirmed = bull_pullback & (fwd_ret > confirm_pct)
    short_confirmed = bear_rally & (fwd_ret < -confirm_pct)

    labels = pd.Series(
        np.where(long_confirmed, 1, np.where(short_confirmed, -1, 0)),
        index=df.index,
        dtype=np.int8,
        name="regime_pullback_label",
    )

    # Stats
    n_long = int((labels == 1).sum())
    n_short = int((labels == -1).sum())
    n_total = n_long + n_short
    n_bull = int(bull.sum())
    n_bear = int(bear.sum())
    n_neutral = n - n_bull - n_bear
    print(f"  Regime: bull={n_bull:,} bear={n_bear:,} neutral={n_neutral:,} bars")
    print(
        f"  Labels: {n_total} entries ({n_long} long pullbacks, {n_short} short rallies) "
        f"in {n:,} bars ({n_total / n * 100:.1f}%)"
    )

    return labels, regime


# ---------------------------------------------------------------------------
# Walk-Forward splits
# ---------------------------------------------------------------------------


def walk_forward_splits(
    n: int,
    is_bars: int,
    oos_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate rolling walk-forward train/test index arrays."""
    folds = []
    start = 0
    while start + is_bars + oos_bars <= n:
        is_idx = np.arange(start, start + is_bars)
        oos_idx = np.arange(start + is_bars, start + is_bars + oos_bars)
        folds.append((is_idx, oos_idx))
        start += oos_bars
    return folds


# ---------------------------------------------------------------------------
# Signal Sharpe computation
# ---------------------------------------------------------------------------


def compute_signal_sharpe(
    predictions: np.ndarray,
    bar_returns: pd.Series,
    cost_bps: float,
    bars_per_year: int,
) -> dict:
    """Compute annualised Sharpe of pred_signal x bar_returns, net of costs."""
    positions = pd.Series(predictions, index=bar_returns.index)
    transitions = (positions != positions.shift(1).fillna(0.0)).astype(float)
    cost_per_bar = transitions * cost_bps / 10_000

    strategy_rets = positions * bar_returns - cost_per_bar
    strategy_rets = strategy_rets.dropna()

    if len(strategy_rets) < 20:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0, "n_trades": 0}

    std = float(strategy_rets.std())
    sharpe = float(strategy_rets.mean() / std * np.sqrt(bars_per_year)) if std > 1e-10 else 0.0

    equity = (1.0 + strategy_rets).cumprod()
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())

    n_years = len(strategy_rets) / bars_per_year
    total_ret = float(equity.iloc[-1] - 1.0)
    cagr = float((1 + total_ret) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0

    n_trades = int(transitions.sum())

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "returns": strategy_rets,
    }


# ---------------------------------------------------------------------------
# Main pipeline for one instrument
# ---------------------------------------------------------------------------


def _pred_to_position(pred_proba: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Convert classifier probability to held position.

    pred_proba: probability of class 1 (swing low / long entry).
    If P(long) > threshold → go long. If P(long) < (1-threshold) → go short.
    Otherwise hold previous position (no flip).
    """
    n = len(pred_proba)
    position = np.zeros(n, dtype=np.float64)
    current = 0.0  # start flat
    for i in range(n):
        if pred_proba[i] > threshold:
            current = 1.0
        elif pred_proba[i] < (1.0 - threshold):
            current = -1.0
        position[i] = current
    return position


def run_instrument(
    instrument: str,
    tf: str,
    asset_type: str,
    **kwargs,
) -> pd.DataFrame:
    """Full ML pipeline using regime+pullback labels. Trains on confirmed
    pullback bars only, predicts on all bars, holds position until flip.
    """
    from xgboost import XGBClassifier

    print(f"\n  Loading {instrument} {tf} ...")
    df = _load_ohlcv(instrument, tf)
    n_bars = len(df)
    bars_yr = _get_annual_bars(tf)
    cost = COST_BPS.get(asset_type, 1.0)

    print(
        f"  {n_bars:,} bars | {df.index[0].date()} to {df.index[-1].date()} | ~{n_bars / bars_yr:.1f} yr"
    )

    # 1. Features
    print("  Computing ~72 features ...")
    df.attrs["instrument"] = instrument
    features = build_features(df, tf)

    # 2. Pre-compute labels for multiple parameter sets (sweep on IS per fold)
    LABEL_SWEEP = [
        {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 10, "confirm_pct": 0.005},
        {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 20, "confirm_pct": 0.005},
        {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 10, "confirm_pct": 0.01},
        {"rsi_oversold": 40, "rsi_overbought": 60, "confirm_bars": 10, "confirm_pct": 0.005},
        {"rsi_oversold": 40, "rsi_overbought": 60, "confirm_bars": 20, "confirm_pct": 0.01},
        {"rsi_oversold": 50, "rsi_overbought": 50, "confirm_bars": 10, "confirm_pct": 0.003},
        {"rsi_oversold": 50, "rsi_overbought": 50, "confirm_bars": 5, "confirm_pct": 0.002},
        {"rsi_oversold": 48, "rsi_overbought": 52, "confirm_bars": 10, "confirm_pct": 0.005},
    ]

    print(f"  Pre-computing labels for {len(LABEL_SWEEP)} parameter sets ...")
    label_cache: list[tuple[dict, pd.Series]] = []
    for lp in LABEL_SWEEP:
        labels, _ = compute_regime_pullback_labels(
            df,
            rsi_oversold=lp["rsi_oversold"],
            rsi_overbought=lp["rsi_overbought"],
            confirm_bars=lp["confirm_bars"],
            confirm_pct=lp["confirm_pct"],
        )
        label_cache.append((lp, labels))

    # 3. Bar returns for Sharpe computation
    bar_returns = df["close"].pct_change().fillna(0.0)

    # 4. Valid feature mask (shared across all label sets)
    mask_feat_valid = features.notna().all(axis=1)
    features_all = features[mask_feat_valid].copy()
    returns_all = bar_returns.reindex(features_all.index).fillna(0.0)

    # 5. Walk-Forward splits
    is_bars_n = IS_RATIO_BARS.get(tf, 504)
    oos_bars_n = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(features_all), is_bars_n, oos_bars_n)
    if not folds:
        print(f"  [SKIP] Not enough data for WFO ({len(features_all)} < {is_bars_n + oos_bars_n})")
        return pd.DataFrame()

    print(f"  WFO: {len(folds)} folds (IS={is_bars_n:,}, OOS={oos_bars_n:,}, rolling)")

    X_all = features_all.values
    all_idx_set = features_all.index
    feature_names = features_all.columns.tolist()

    fold_results: list[dict] = []
    all_oos_returns: list[pd.Series] = []
    all_importances: list[np.ndarray] = []

    for i, (is_idx, oos_idx) in enumerate(folds):
        # Sweep label params: pick the one with most IS entries (>= 20, balanced)
        best_label_params = None
        best_entry_count = 0
        best_entry_positions = None
        best_entry_y = None

        for lp, labels in label_cache:
            lab_aligned = labels.reindex(all_idx_set).fillna(0).values
            entry_positions = np.where(lab_aligned != 0)[0]
            # Filter to IS window
            is_set = set(is_idx)
            is_entries = np.array([p for p in entry_positions if p in is_set])
            if len(is_entries) < 20:
                continue
            # Check class balance (at least 30% minority class)
            y_is = (lab_aligned[is_entries] == 1).astype(int)
            minority_pct = min(y_is.mean(), 1 - y_is.mean())
            if minority_pct < 0.15:
                continue
            if len(is_entries) > best_entry_count:
                best_entry_count = len(is_entries)
                best_label_params = lp
                # Also get full entries for this label set (used in training below)
                best_entry_positions = entry_positions
                best_entry_y = (lab_aligned == 1).astype(int)

        if best_label_params is None:
            print(f"    Fold {i + 1}: no label params produced >= 20 entries, skipping.")
            continue

        # Train on IS entries with best label params
        is_set = set(is_idx)
        is_entries = np.array([p for p in best_entry_positions if p in is_set])
        X_is_entry = np.nan_to_num(X_all[is_entries], nan=0.0, posinf=0.0, neginf=0.0)
        y_train = best_entry_y[is_entries]
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        spw = neg_count / pos_count if pos_count > 0 else 1.0

        params = {**XGB_PARAMS, "scale_pos_weight": spw, "eval_metric": "logloss"}
        model = XGBClassifier(**params)
        model.fit(X_is_entry, y_train)

        # Predict on ALL OOS bars
        X_oos_all = np.nan_to_num(X_all[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)
        pred_proba = model.predict_proba(X_oos_all)[:, 1]  # P(long pullback)

        # Convert to position: hold until prediction flips
        position = _pred_to_position(pred_proba, threshold=SIGNAL_THRESHOLD)

        oos_returns = returns_all.iloc[oos_idx]
        stats = compute_signal_sharpe(position, oos_returns, cost, bars_yr)

        # IS evaluation (on all IS bars, not just swings)
        X_is_all = np.nan_to_num(X_all[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        pred_proba_is = model.predict_proba(X_is_all)[:, 1]
        position_is = _pred_to_position(pred_proba_is, threshold=SIGNAL_THRESHOLD)
        is_returns = returns_all.iloc[is_idx]
        is_stats = compute_signal_sharpe(position_is, is_returns, cost, bars_yr)

        parity = (
            (stats["sharpe"] / is_stats["sharpe"]) if is_stats["sharpe"] != 0.0 else float("nan")
        )

        # Trade count: number of position flips
        flips = np.sum(np.abs(np.diff(position)) > 0)
        n_long_bars = int((position == 1.0).sum())
        n_short_bars = int((position == -1.0).sum())
        hold_pct = (n_long_bars + n_short_bars) / len(position) if len(position) > 0 else 0.0

        all_importances.append(model.feature_importances_)

        lp_str = (
            f"RSI({best_label_params['rsi_oversold']:.0f}/{best_label_params['rsi_overbought']:.0f}) "
            f"cfm={best_label_params['confirm_bars']}bar/{best_label_params['confirm_pct']:.1%}"
        )

        oos_period = features_all.index[oos_idx]
        fold_results.append(
            {
                "fold": i + 1,
                "oos_start": oos_period[0].date(),
                "oos_end": oos_period[-1].date(),
                "is_sharpe": round(is_stats["sharpe"], 3),
                "oos_sharpe": round(stats["sharpe"], 3),
                "parity": round(parity, 3) if not np.isnan(parity) else float("nan"),
                "oos_cagr": round(stats["cagr"], 4),
                "oos_max_dd": round(stats["max_dd"], 4),
                "oos_trades": int(flips),
                "pct_active": round(hold_pct, 3),
                "n_long": n_long_bars,
                "n_short": n_short_bars,
                "label_params": lp_str,
                "n_train": best_entry_count,
            }
        )

        if "returns" in stats:
            all_oos_returns.append(stats["returns"])

        print(
            f"    Fold {i + 1}/{len(folds)} ({best_entry_count} train): "
            f"IS={is_stats['sharpe']:+.2f}  OOS={stats['sharpe']:+.2f}  "
            f"par={parity:.2f}  "
            f"flips={int(flips)}  hold={hold_pct:.0%} (L={n_long_bars} S={n_short_bars})  "
            f"DD={stats['max_dd']:.1%}  {lp_str}"
        )

    if not fold_results:
        return pd.DataFrame()

    # 7. Stitched OOS equity
    if all_oos_returns:
        stitched = pd.concat(all_oos_returns).sort_index()
        std = float(stitched.std())
        stitched_sharpe = float(stitched.mean() / std * np.sqrt(bars_yr)) if std > 1e-10 else 0.0
        stitched_eq = (1.0 + stitched).cumprod()
        stitched_dd = float(((stitched_eq - stitched_eq.cummax()) / stitched_eq.cummax()).min())
        total_ret = float(stitched_eq.iloc[-1] - 1.0)
    else:
        stitched_sharpe = 0.0
        stitched_dd = 0.0
        total_ret = 0.0

    # 8. Feature importance (averaged across folds)
    avg_importance = np.mean(all_importances, axis=0)
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": avg_importance,
        }
    ).sort_values("importance", ascending=False)

    # 9. Print results
    results_df = pd.DataFrame(fold_results)
    print(f"\n  {'=' * 70}")
    print(f"  RESULTS: {instrument} {tf}")
    print(f"  {'=' * 70}")

    print("\n  Per-fold WFO:")
    print(
        f"  {'Fold':>4} {'OOS Start':>12} {'OOS End':>12} "
        f"{'IS Sh':>7} {'OOS Sh':>7} {'Par':>5} "
        f"{'DD':>7} {'Flips':>5} {'Hold%':>5} {'L':>5} {'S':>5}"
    )
    print("  " + "-" * 80)
    for _, r in results_df.iterrows():
        print(
            f"  {int(r['fold']):>4} {str(r['oos_start']):>12} {str(r['oos_end']):>12} "
            f"{r['is_sharpe']:>+7.2f} {r['oos_sharpe']:>+7.2f} "
            f"{r['parity']:>5.2f} "
            f"{r['oos_max_dd']:>7.1%} {r['oos_trades']:>5} {r['pct_active']:>5.0%} "
            f"{r['n_long']:>5} {r['n_short']:>5}"
        )

    # Validation gates
    n_folds = len(results_df)
    pct_positive = (results_df["oos_sharpe"] > 0).mean()
    avg_parity = results_df["parity"].mean()
    worst_fold = results_df["oos_sharpe"].min()

    # Consecutive negative folds
    neg_streak = 0
    max_neg_streak = 0
    for sh in results_df["oos_sharpe"]:
        if sh <= 0:
            neg_streak += 1
            max_neg_streak = max(max_neg_streak, neg_streak)
        else:
            neg_streak = 0

    print(
        f"\n  Stitched OOS: Sharpe={stitched_sharpe:+.3f}  DD={stitched_dd:.1%}  Return={total_ret:+.1%}"
    )
    print(f"  Folds positive: {pct_positive:.0%} ({int(pct_positive * n_folds)}/{n_folds})")
    print(f"  Avg parity: {avg_parity:.2f}")
    print(f"  Worst fold: {worst_fold:+.2f}")
    print(f"  Max consec neg: {max_neg_streak}")

    # Gate verdicts
    gates = {
        "stitched_sharpe": stitched_sharpe >= MIN_STITCHED_SHARPE,
        "pct_positive": pct_positive >= MIN_PCT_POSITIVE_FOLDS,
        "avg_parity": avg_parity >= MIN_PARITY,
        "worst_fold": worst_fold >= MIN_WORST_FOLD_SHARPE,
        "consec_neg": max_neg_streak <= MAX_CONSEC_NEG,
    }
    all_pass = all(gates.values())

    print("\n  GATES:")
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {gate_name}")
    print(f"\n  VERDICT: {'PASS -- deploy candidate' if all_pass else 'FAIL -- needs improvement'}")

    # Top features
    print(f"\n  Top 15 features (averaged across {n_folds} folds):")
    for _, row in imp_df.head(15).iterrows():
        bar = "#" * int(row["importance"] * 200)
        print(f"    {row['feature']:<25s} {row['importance']:.4f}  {bar}")

    # Add metadata to results
    results_df["instrument"] = instrument
    results_df["tf"] = tf
    results_df["stitched_sharpe"] = stitched_sharpe
    results_df["all_gates_pass"] = all_pass

    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="52-signal ML classifier with swing-point labels (currencies + indices)."
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Single instrument to run (default: all targets).",
    )
    parser.add_argument(
        "--tf",
        default=None,
        help="Override timeframe (e.g. M5, H1, D). Default: per-instrument.",
    )
    args = parser.parse_args()

    if args.instrument:
        if args.instrument in TARGET_INSTRUMENTS:
            default_tf, default_asset = TARGET_INSTRUMENTS[args.instrument]
        else:
            default_tf, default_asset = "H1", "fx"
        tf_override = args.tf if args.tf else default_tf
        instruments = {args.instrument: (tf_override, default_asset)}
    else:
        instruments = TARGET_INSTRUMENTS

    W = 80
    print()
    print("=" * W)
    print("  ML SIGNAL DISCOVERY v3 -- Regime+Pullback Labels + XGBClassifier")
    print(f"  Instruments : {list(instruments.keys())}")
    print(f"  Regime      : SMA({REGIME_SMA_FAST}/{REGIME_SMA_SLOW}) + MACD(12/26/9)")
    print(
        f"  Pullback    : RSI({REGIME_RSI_PERIOD}) < {REGIME_RSI_OVERSOLD} (bull) / > {REGIME_RSI_OVERBOUGHT} (bear)"
    )
    print(f"  Confirm     : fwd {REGIME_CONFIRM_BARS}bar > {REGIME_CONFIRM_PCT:.1%}")
    print(f"  Flip thresh : P > {SIGNAL_THRESHOLD} = long, P < {1 - SIGNAL_THRESHOLD:.1f} = short")
    print("  Model       : XGBClassifier (depth=4, lr=0.03, 300 trees)")
    print("  Training    : confirmed pullback bars only (~3-8% of data)")
    print("  Validation  : Walk-Forward (rolling IS=2yr, OOS=6mo)")
    print("=" * W)

    all_results: list[pd.DataFrame] = []

    def _run_one(inst_tf_asset):
        inst, (tf, asset_type) = inst_tf_asset
        try:
            return run_instrument(inst, tf, asset_type)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {inst}: {exc}")
            return pd.DataFrame()
        except Exception as exc:
            print(f"  [ERROR] {inst}: {exc}")
            return pd.DataFrame()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_workers = min(4, len(instruments))
    print(f"\n  Running {len(instruments)} instruments ({n_workers} threads) ...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, (inst, meta)): inst for inst, meta in instruments.items()}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                all_results.append(df)

    if not all_results:
        print("\n  No results produced.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Summary
    print()
    print("=" * W)
    print("  CROSS-INSTRUMENT SUMMARY")
    print("=" * W)
    print(
        f"\n  {'Instrument':<12} {'TF':>3} {'Folds':>5} {'Stitched':>9} "
        f"{'%Pos':>5} {'AvgPar':>7} {'Worst':>6} {'Verdict':<8}"
    )
    print("  " + "-" * 60)

    for inst in instruments:
        inst_df = combined[combined["instrument"] == inst]
        if inst_df.empty:
            print(f"  {inst:<12} {'--':>3} {'--':>5} {'--':>9} {'--':>5} {'--':>7} {'--':>6} SKIP")
            continue
        st_sh = inst_df["stitched_sharpe"].iloc[0]
        pct_pos = (inst_df["oos_sharpe"] > 0).mean()
        avg_par = inst_df["parity"].mean()
        worst = inst_df["oos_sharpe"].min()
        verdict = "PASS" if inst_df["all_gates_pass"].iloc[0] else "FAIL"
        print(
            f"  {inst:<12} {inst_df['tf'].iloc[0]:>3} {len(inst_df):>5} "
            f"{st_sh:>+9.3f} {pct_pos:>5.0%} {avg_par:>7.2f} "
            f"{worst:>+6.2f} {verdict:<8}"
        )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"ml_52sig_{ts}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")

    # Save model for passing instruments
    passing = combined[combined["all_gates_pass"]]["instrument"].unique()
    if len(passing):
        print(f"\n  Passing instruments: {list(passing)}")
        print("  Models can be saved by re-running with --instrument <name>")
    else:
        print("\n  No instruments passed all gates.")


if __name__ == "__main__":
    main()
