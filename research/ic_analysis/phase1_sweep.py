"""phase1_sweep.py -- Phase 1: Comprehensive 52-Signal IC/ICIR Sweep.

Extends run_ic.py with all major indicator families, acceleration signals,
structural breakout signals, and semantic combinations. Outputs a ranked
leaderboard by |IC|/ICIR to identify signals with genuine predictive edge.

Optionally accepts a regime DataFrame (from phase0_regime.py) to compute
regime-conditional IC: unconditional, trending-only, and ranging-only.
FLIP signals (sign reversal across regimes) are highlighted prominently.

Usage:
    uv run python research/ic_analysis/phase1_sweep.py
    uv run python research/ic_analysis/phase1_sweep.py --instrument EUR_USD --timeframe D

Signal groups (52 total):
    A: Trend (10)           -- MA spreads, MACD norm, EMA slope
    B: Momentum (11)        -- RSI variants, stochastic, CCI, Williams %R, ROC
    C: Mean Reversion (6)   -- Bollinger z-score, rolling/expanding z-score
    D: Volatility State (7) -- ATR, realized vol, Garman-Klass, Parkinson, BW, ADX
    E: Acceleration (7)     -- .diff(1) of base signals (rate of change)
    F: Structural (6)       -- Donchian position, Keltner, price percentile rank
    G: Combinations (5)     -- Semantic blends of A/B/D/F signals

Interpretation:
    |IC| >= 0.05, ICIR >= 0.5   STRONG -- build strategy around this
    |IC| >= 0.05, ICIR <  0.5   USABLE -- IC present but inconsistent
    0.03 <= |IC| < 0.05         WEAK   -- try regime conditioning
    |IC| < 0.03                 NOISE  -- discard

Look-ahead safety:
    All signal factories use .rolling() / .ewm() / .shift(+n) only (causal).
    Forward returns use close.shift(-h) -- intentional (target, not feature).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.run_ic import (  # noqa: E402
    build_rolling_ic_map,
    compute_autocorrelation,
    compute_forward_returns,
    compute_ic_table,
    compute_icir,
    compute_icir_nw,
    quantile_spread,
)
from titan.strategies.ml.features import (  # noqa: E402
    adx,
    atr,
    bollinger_bw,
    ema,
    macd_hist,
    rsi,
    sma,
    stochastic,
    wma,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [1, 5, 10, 20, 60]

# Set by --strict CLI flag. When True, data quality issues raise errors instead of warnings.
_STRICT_VALIDATION: bool = False


def _get_annual_bars(tf: str) -> int:
    """Return the approximate number of bars in one calendar year."""
    t = tf.lower()
    if t == "m1":
        return int(252 * 6.5 * 60)
    if t == "m5":
        return int(252 * 6.5 * 12)
    if t == "m15":
        return int(252 * 6.5 * 4)
    if t == "m30":
        return int(252 * 6.5 * 2)
    if t.startswith("h"):
        h = float(t.replace("h", "") or 1.0)
        return int(252 * 24 / h) if "crypto" in t or "fx" in t else int(252 * 6.5 / h)
    if t == "w":
        return 52
    if t == "m":
        return 12
    return 252  # default (Daily)


# Populated by _tag() as signal factories run -- maps signal name -> group label
_SIGNAL_GROUP: dict[str, str] = {}


def _tag(name: str, group: str) -> str:
    """Register signal name -> group label and return the name."""
    _SIGNAL_GROUP[name] = group
    return name


def _bar(val: float, width: int = 20) -> str:
    """ASCII bar chart for a value normalised to [-1, 1]."""
    mid = width // 2
    filled = min(int(abs(val) * mid), mid)
    if val >= 0:
        return " " * mid + "#" * filled + " " * (mid - filled)
    return " " * (mid - filled) + "#" * filled + " " * mid


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_ohlcv(
    instrument: str,
    timeframe: str,
    data_dir: Path | None = None,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Load OHLCV data for the given instrument and timeframe."""
    if data_dir is None:
        data_dir = ROOT / "data"
    path = data_dir / f"{instrument}_{timeframe}.{fmt}"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    if fmt == "csv":
        df = pd.read_csv(path)
        ts_col = "ts_event" if "ts_event" in df.columns else "timestamp"
        df = df.set_index(ts_col)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index, utc=True)
            else:
                raise ValueError(f"Cannot resolve timestamp index: {path}")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    df = df[["open", "high", "low", "close"]].dropna()

    # R3 FIX: Data quality validation
    _validate_ohlcv(df, instrument, timeframe)

    logger.info(
        "Loaded %d bars | %s %s (%s - %s)",
        len(df),
        instrument,
        timeframe,
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def _validate_ohlcv(df: pd.DataFrame, instrument: str, timeframe: str) -> None:
    """Data quality validation.

    With _STRICT_VALIDATION=False (default): warns on issues, never raises.
    With _STRICT_VALIDATION=True (--strict flag): raises ValueError on any issue.
    Use --strict for production/pipeline runs where silent invalid IC results are
    unacceptable (e.g. a dataset with 20% stale bars).
    """
    tag = f"[{instrument} {timeframe}]"
    issues: list[str] = []

    # 1. Duplicate timestamps
    if not df.index.is_unique:
        n_dupes = df.index.duplicated().sum()
        issues.append(f"Duplicate timestamps detected: {n_dupes} rows")

    # 2. Stale prices (≥5 consecutive identical closes)
    stale_run = (df["close"].diff() == 0).astype(int)
    stale_streak = stale_run.groupby((stale_run != stale_run.shift()).cumsum()).cumsum()
    n_stale = int((stale_streak >= 5).sum())
    if n_stale > 0:
        issues.append(f"Stale prices (≥5 identical closes): {n_stale} bars")

    # 3. OHLC integrity
    high_ok = (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    low_ok = (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    if not high_ok or not low_ok:
        issues.append("OHLC integrity violation (high < max(O,C) or low > min(O,C))")

    # 4. Gap detection (>2x median bar spacing)
    if len(df) > 10:
        diffs = df.index.to_series().diff().dropna()
        median_gap = diffs.median()
        if median_gap.total_seconds() > 0:
            large_gaps = diffs[diffs > 2 * median_gap]
            if len(large_gaps) > 5:
                issues.append(
                    f"Large time gaps (>2x median): {len(large_gaps)} occurrences, "
                    f"largest={large_gaps.max()}"
                )

    for msg in issues:
        if _STRICT_VALIDATION:
            raise ValueError(f"{tag} Data quality error: {msg}")
        else:
            logger.warning("%s %s", tag, msg)


# ── Signal group factories (all causal -- no look-ahead) ──────────────────────


def _compute_group_a(close: pd.Series) -> pd.DataFrame:
    """Group A: Trend (10 signals)."""
    e5 = ema(close, 5)
    e10 = ema(close, 10)
    e12 = ema(close, 12)
    e20 = ema(close, 20)
    e26 = ema(close, 26)
    e50 = ema(close, 50)
    e100 = ema(close, 100)
    e200 = ema(close, 200)
    s20 = sma(close, 20)
    s50 = sma(close, 50)
    roll_std20 = close.rolling(20).std()
    w5 = wma(close, 5)
    w20 = wma(close, 20)

    out = pd.DataFrame(index=close.index)
    out[_tag("ma_spread_5_20", "Trend")] = (e5 - e20) / e20
    out[_tag("ma_spread_10_50", "Trend")] = (e10 - e50) / e50
    out[_tag("ma_spread_20_100", "Trend")] = (e20 - e100) / e100
    out[_tag("ma_spread_50_200", "Trend")] = (e50 - e200) / e200
    out[_tag("wma_spread_5_20", "Trend")] = (w5 - w20) / w20.replace(0, np.nan)
    out[_tag("price_vs_sma20", "Trend")] = (close - s20) / s20
    out[_tag("price_vs_sma50", "Trend")] = (close - s50) / s50
    out[_tag("macd_norm", "Trend")] = (e12 - e26) / roll_std20.replace(0, np.nan)
    out[_tag("ema_slope_10", "Trend")] = (e10 - e10.shift(5)) / e10.shift(5)
    out[_tag("ema_slope_20", "Trend")] = (e20 - e20.shift(10)) / e20.shift(10)
    return out


def _compute_group_b(df: pd.DataFrame) -> pd.DataFrame:
    """Group B: Momentum (11 signals). Needs OHLCV for stochastic / Williams %R."""
    close = df["close"]
    log_c = np.log(close)
    s20 = sma(close, 20)
    stoch_k_raw, stoch_d_raw = stochastic(df, k_period=14, d_period=3)
    hh14 = df["high"].rolling(14).max()
    ll14 = df["low"].rolling(14).min()
    # Williams %R: +50 = at range high (bullish), -50 = at range low (bearish)
    wr14 = (hh14 - close) / (hh14 - ll14).replace(0, np.nan) * -100
    mad20 = (close - s20).abs().rolling(20).mean()

    out = pd.DataFrame(index=close.index)
    out[_tag("rsi_7_dev", "Momen")] = rsi(close, 7) - 50.0
    out[_tag("rsi_14_dev", "Momen")] = rsi(close, 14) - 50.0
    out[_tag("rsi_21_dev", "Momen")] = rsi(close, 21) - 50.0
    out[_tag("stoch_k_dev", "Momen")] = stoch_k_raw - 50.0
    out[_tag("stoch_d_dev", "Momen")] = stoch_d_raw - 50.0
    out[_tag("cci_20", "Momen")] = (close - s20) / (0.015 * mad20.replace(0, np.nan))
    out[_tag("williams_r_dev", "Momen")] = wr14 + 50.0
    out[_tag("roc_3", "Momen")] = log_c - log_c.shift(3)
    out[_tag("roc_10", "Momen")] = log_c - log_c.shift(10)
    out[_tag("roc_20", "Momen")] = log_c - log_c.shift(20)
    out[_tag("roc_60", "Momen")] = log_c - log_c.shift(60)
    return out


def _compute_group_c(close: pd.Series, window_1y: int) -> pd.DataFrame:
    """Group C: Mean Reversion (6 signals)."""
    out = pd.DataFrame(index=close.index)
    for w in (20, 50):
        s = sma(close, w)
        std = close.rolling(w).std().replace(0, np.nan)
        out[_tag(f"bb_zscore_{w}", "MRev")] = (close - s) / (2.0 * std)
        out[_tag(f"zscore_{w}", "MRev")] = (close - s) / std
    s100 = sma(close, 100)
    std100 = close.rolling(100).std().replace(0, np.nan)
    out[_tag("zscore_100", "MRev")] = (close - s100) / std100
    roll_mean_1y = close.rolling(window_1y).mean()
    roll_std_1y = close.rolling(window_1y).std().replace(0, np.nan)
    out[_tag(f"zscore_{window_1y}", "MRev")] = (close - roll_mean_1y) / roll_std_1y
    return out


def _compute_group_d(df: pd.DataFrame) -> pd.DataFrame:
    """Group D: Volatility State (7 signals). Needs OHLCV."""
    close = df["close"]
    log_r = np.log(close).diff()

    # Garman-Klass volatility (rolling 20-bar)
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    log_co = np.log(close / df["open"].replace(0, np.nan))
    gk_bar = 0.5 * log_hl**2 - (2.0 * np.log(2) - 1.0) * log_co**2
    gk_roll = gk_bar.rolling(20).mean().clip(lower=0.0) ** 0.5

    # Parkinson volatility (rolling 20-bar)
    pk_bar = log_hl**2 / (4.0 * np.log(2))
    pk_roll = pk_bar.rolling(20).mean().clip(lower=0.0) ** 0.5

    out = pd.DataFrame(index=close.index)
    out[_tag("norm_atr_14", "Vol")] = atr(df, 14) / close
    out[_tag("realized_vol_5", "Vol")] = log_r.rolling(5).std() * np.sqrt(252)
    out[_tag("realized_vol_20", "Vol")] = log_r.rolling(20).std() * np.sqrt(252)
    out[_tag("garman_klass", "Vol")] = gk_roll
    out[_tag("parkinson_vol", "Vol")] = pk_roll
    out[_tag("bb_width", "Vol")] = bollinger_bw(close, 20)
    out[_tag("adx_14", "Vol")] = adx(df, 14)
    return out


def _compute_group_e(sigs: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Group E: Acceleration / Deceleration (7 signals). diff(1) of base signals."""
    out = pd.DataFrame(index=sigs.index)
    out[_tag("accel_roc10", "Accel")] = sigs["roc_10"].diff(1)
    out[_tag("accel_rsi14", "Accel")] = sigs["rsi_14_dev"].diff(1)
    out[_tag("accel_macd", "Accel")] = macd_hist(close, 12, 26, 9).diff(1)
    out[_tag("accel_atr", "Accel")] = sigs["norm_atr_14"].diff(1)
    out[_tag("accel_bb_width", "Accel")] = sigs["bb_width"].diff(1)
    out[_tag("accel_rvol20", "Accel")] = sigs["realized_vol_20"].diff(1)
    out[_tag("accel_stoch_k", "Accel")] = sigs["stoch_k_dev"].diff(1)
    return out


def _compute_group_f(df: pd.DataFrame) -> pd.DataFrame:
    """Group F: Structural / Breakout (6 signals). Needs OHLCV."""
    close = df["close"]
    e20 = ema(close, 20)
    atr10 = atr(df, 10).replace(0, np.nan)

    out = pd.DataFrame(index=close.index)
    for w in (10, 20, 55):
        lo = df["low"].rolling(w).min()
        hi = df["high"].rolling(w).max()
        rng = (hi - lo).replace(0, np.nan)
        out[_tag(f"donchian_pos_{w}", "Struct")] = (close - lo) / rng - 0.5
    out[_tag("keltner_pos", "Struct")] = (close - e20) / (2.0 * atr10)
    out[_tag("price_pct_rank_20", "Struct")] = close.rolling(20).rank(pct=True) - 0.5
    out[_tag("price_pct_rank_60", "Struct")] = close.rolling(60).rank(pct=True) - 0.5
    return out


def _compute_group_g(sigs: pd.DataFrame) -> pd.DataFrame:
    """Group G: Semantic Combinations (5 signals). Built from A/B/D/E/F."""
    atr_norm = sigs["norm_atr_14"]
    atr_ma60 = atr_norm.rolling(60).mean().replace(0, np.nan)

    out = pd.DataFrame(index=sigs.index)
    # Trend direction gates momentum magnitude
    out[_tag("trend_mom", "Combo")] = (
        np.sign(sigs["ma_spread_5_20"]) * sigs["rsi_14_dev"].abs() / 50.0
    )
    # Trend signal normalised by volatility regime
    out[_tag("trend_vol_adj", "Combo")] = sigs["ma_spread_5_20"] / (atr_norm + 1e-9)
    # Momentum × whether it is accelerating or decelerating
    out[_tag("mom_accel_combo", "Combo")] = sigs["rsi_14_dev"] * np.sign(sigs["accel_rsi14"])
    # Structural breakout gated by momentum
    out[_tag("donchian_rsi", "Combo")] = sigs["donchian_pos_20"] * (sigs["rsi_14_dev"] / 50.0)
    # Trend attenuated in elevated-vol regime
    out[_tag("vol_regime_trend", "Combo")] = sigs["ma_spread_5_20"] * (1.0 - atr_norm / atr_ma60)
    return out


def build_all_signals(
    df: pd.DataFrame,
    window_1y: int,
    period_scale: int = 1,
    name_prefix: str = "",
) -> pd.DataFrame:
    """Compute all 52 signals. Groups A->B->C->D, then E (uses A/B/D), F, G.

    When period_scale > 1, all internal indicator periods are multiplied by the
    scale factor. This allows computing "virtual higher-TF" signals on a single
    lower-TF data stream (e.g. daily-scale signals on H1 bars with scale=24).

    When name_prefix is set, all signal names are prefixed (e.g. "D_ma_spread_5_20").
    """
    close = df["close"]
    s = period_scale

    if s == 1 and name_prefix == "":
        # Original path -- no scaling, no prefix (backward compatible)
        a = _compute_group_a(close)
        b = _compute_group_b(df)
        c = _compute_group_c(close, window_1y)
        d = _compute_group_d(df)
        base = pd.concat([a, b, c, d], axis=1)
        e = _compute_group_e(base, close)
        f = _compute_group_f(df)
        all_so_far = pd.concat([base, e, f], axis=1)
        g = _compute_group_g(all_so_far)
        return pd.concat([all_so_far, g], axis=1)

    # Scaled path -- compute each group with scaled periods
    a = _compute_group_a_scaled(close, s, name_prefix)
    b = _compute_group_b_scaled(df, s, name_prefix)
    c = _compute_group_c_scaled(close, window_1y, s, name_prefix)
    d = _compute_group_d_scaled(df, s, name_prefix)
    base = pd.concat([a, b, c, d], axis=1)
    e = _compute_group_e_scaled(base, close, s, name_prefix)
    f = _compute_group_f_scaled(df, s, name_prefix)
    all_so_far = pd.concat([base, e, f], axis=1)
    g = _compute_group_g_scaled(all_so_far, name_prefix)
    return pd.concat([all_so_far, g], axis=1)


# ── Scaled group factories ────────────────────────────────────────────────────


def _compute_group_a_scaled(close: pd.Series, s: int, p: str) -> pd.DataFrame:
    """Group A: Trend (10 signals) -- all periods scaled by s."""
    e5 = ema(close, 5 * s)
    e10 = ema(close, 10 * s)
    e12 = ema(close, 12 * s)
    e20 = ema(close, 20 * s)
    e26 = ema(close, 26 * s)
    e50 = ema(close, 50 * s)
    e100 = ema(close, 100 * s)
    e200 = ema(close, 200 * s)
    s20 = sma(close, 20 * s)
    s50 = sma(close, 50 * s)
    roll_std20 = close.rolling(20 * s).std()
    w5 = wma(close, 5 * s)
    w20 = wma(close, 20 * s)

    out = pd.DataFrame(index=close.index)
    out[_tag(f"{p}ma_spread_5_20", "Trend")] = (e5 - e20) / e20
    out[_tag(f"{p}ma_spread_10_50", "Trend")] = (e10 - e50) / e50
    out[_tag(f"{p}ma_spread_20_100", "Trend")] = (e20 - e100) / e100
    out[_tag(f"{p}ma_spread_50_200", "Trend")] = (e50 - e200) / e200
    out[_tag(f"{p}wma_spread_5_20", "Trend")] = (w5 - w20) / w20.replace(0, np.nan)
    out[_tag(f"{p}price_vs_sma20", "Trend")] = (close - s20) / s20
    out[_tag(f"{p}price_vs_sma50", "Trend")] = (close - s50) / s50
    out[_tag(f"{p}macd_norm", "Trend")] = (e12 - e26) / roll_std20.replace(0, np.nan)
    out[_tag(f"{p}ema_slope_10", "Trend")] = (e10 - e10.shift(5 * s)) / e10.shift(5 * s)
    out[_tag(f"{p}ema_slope_20", "Trend")] = (e20 - e20.shift(10 * s)) / e20.shift(10 * s)
    return out


def _compute_group_b_scaled(df: pd.DataFrame, s: int, p: str) -> pd.DataFrame:
    """Group B: Momentum (11 signals) -- periods scaled by s."""
    close = df["close"]
    log_c = np.log(close)
    s20 = sma(close, 20 * s)
    stoch_k_raw, stoch_d_raw = stochastic(df, k_period=14 * s, d_period=3 * s)
    hh14 = df["high"].rolling(14 * s).max()
    ll14 = df["low"].rolling(14 * s).min()
    wr14 = (hh14 - close) / (hh14 - ll14).replace(0, np.nan) * -100
    mad20 = (close - s20).abs().rolling(20 * s).mean()

    out = pd.DataFrame(index=close.index)
    out[_tag(f"{p}rsi_7_dev", "Momen")] = rsi(close, 7 * s) - 50.0
    out[_tag(f"{p}rsi_14_dev", "Momen")] = rsi(close, 14 * s) - 50.0
    out[_tag(f"{p}rsi_21_dev", "Momen")] = rsi(close, 21 * s) - 50.0
    out[_tag(f"{p}stoch_k_dev", "Momen")] = stoch_k_raw - 50.0
    out[_tag(f"{p}stoch_d_dev", "Momen")] = stoch_d_raw - 50.0
    out[_tag(f"{p}cci_20", "Momen")] = (close - s20) / (0.015 * mad20.replace(0, np.nan))
    out[_tag(f"{p}williams_r_dev", "Momen")] = wr14 + 50.0
    out[_tag(f"{p}roc_3", "Momen")] = log_c - log_c.shift(3 * s)
    out[_tag(f"{p}roc_10", "Momen")] = log_c - log_c.shift(10 * s)
    out[_tag(f"{p}roc_20", "Momen")] = log_c - log_c.shift(20 * s)
    out[_tag(f"{p}roc_60", "Momen")] = log_c - log_c.shift(60 * s)
    return out


def _compute_group_c_scaled(close: pd.Series, window_1y: int, s: int, p: str) -> pd.DataFrame:
    """Group C: Mean Reversion (6 signals) -- windows scaled by s."""
    out = pd.DataFrame(index=close.index)
    for w in (20, 50):
        sw = w * s
        sm = sma(close, sw)
        std = close.rolling(sw).std().replace(0, np.nan)
        out[_tag(f"{p}bb_zscore_{w}", "MRev")] = (close - sm) / (2.0 * std)
        out[_tag(f"{p}zscore_{w}", "MRev")] = (close - sm) / std
    sw100 = 100 * s
    s100 = sma(close, sw100)
    std100 = close.rolling(sw100).std().replace(0, np.nan)
    out[_tag(f"{p}zscore_100", "MRev")] = (close - s100) / std100
    sw1y = window_1y * s
    roll_mean = close.rolling(sw1y).mean()
    roll_std = close.rolling(sw1y).std().replace(0, np.nan)
    out[_tag(f"{p}zscore_{window_1y}", "MRev")] = (close - roll_mean) / roll_std
    return out


def _compute_group_d_scaled(df: pd.DataFrame, s: int, p: str) -> pd.DataFrame:
    """Group D: Volatility State (7 signals) -- periods scaled by s."""
    close = df["close"]
    log_r = np.log(close).diff()
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    log_co = np.log(close / df["open"].replace(0, np.nan))
    gk_bar = 0.5 * log_hl**2 - (2.0 * np.log(2) - 1.0) * log_co**2
    gk_roll = gk_bar.rolling(20 * s).mean().clip(lower=0.0) ** 0.5
    pk_bar = log_hl**2 / (4.0 * np.log(2))
    pk_roll = pk_bar.rolling(20 * s).mean().clip(lower=0.0) ** 0.5

    out = pd.DataFrame(index=close.index)
    out[_tag(f"{p}norm_atr_14", "Vol")] = atr(df, 14 * s) / close
    out[_tag(f"{p}realized_vol_5", "Vol")] = log_r.rolling(5 * s).std() * np.sqrt(252)
    out[_tag(f"{p}realized_vol_20", "Vol")] = log_r.rolling(20 * s).std() * np.sqrt(252)
    out[_tag(f"{p}garman_klass", "Vol")] = gk_roll
    out[_tag(f"{p}parkinson_vol", "Vol")] = pk_roll
    out[_tag(f"{p}bb_width", "Vol")] = bollinger_bw(close, 20 * s)
    out[_tag(f"{p}adx_14", "Vol")] = adx(df, 14 * s)
    return out


def _compute_group_e_scaled(sigs: pd.DataFrame, close: pd.Series, s: int, p: str) -> pd.DataFrame:
    """Group E: Acceleration (7 signals). diff(1) stays at 1-bar for regime change detection."""
    out = pd.DataFrame(index=sigs.index)
    out[_tag(f"{p}accel_roc10", "Accel")] = sigs[f"{p}roc_10"].diff(1)
    out[_tag(f"{p}accel_rsi14", "Accel")] = sigs[f"{p}rsi_14_dev"].diff(1)
    out[_tag(f"{p}accel_macd", "Accel")] = macd_hist(close, 12 * s, 26 * s, 9 * s).diff(1)
    out[_tag(f"{p}accel_atr", "Accel")] = sigs[f"{p}norm_atr_14"].diff(1)
    out[_tag(f"{p}accel_bb_width", "Accel")] = sigs[f"{p}bb_width"].diff(1)
    out[_tag(f"{p}accel_rvol20", "Accel")] = sigs[f"{p}realized_vol_20"].diff(1)
    out[_tag(f"{p}accel_stoch_k", "Accel")] = sigs[f"{p}stoch_k_dev"].diff(1)
    return out


def _compute_group_f_scaled(df: pd.DataFrame, s: int, p: str) -> pd.DataFrame:
    """Group F: Structural / Breakout (6 signals) -- periods scaled by s."""
    close = df["close"]
    e20 = ema(close, 20 * s)
    atr10 = atr(df, 10 * s).replace(0, np.nan)

    out = pd.DataFrame(index=close.index)
    for w in (10, 20, 55):
        sw = w * s
        lo = df["low"].rolling(sw).min()
        hi = df["high"].rolling(sw).max()
        rng = (hi - lo).replace(0, np.nan)
        out[_tag(f"{p}donchian_pos_{w}", "Struct")] = (close - lo) / rng - 0.5
    out[_tag(f"{p}keltner_pos", "Struct")] = (close - e20) / (2.0 * atr10)
    out[_tag(f"{p}price_pct_rank_20", "Struct")] = close.rolling(20 * s).rank(pct=True) - 0.5
    out[_tag(f"{p}price_pct_rank_60", "Struct")] = close.rolling(60 * s).rank(pct=True) - 0.5
    return out


def _compute_group_g_scaled(sigs: pd.DataFrame, p: str) -> pd.DataFrame:
    """Group G: Semantic Combinations (5 signals). Built from scaled inputs."""
    atr_norm = sigs[f"{p}norm_atr_14"]
    atr_ma60 = atr_norm.rolling(60).mean().replace(0, np.nan)

    out = pd.DataFrame(index=sigs.index)
    out[_tag(f"{p}trend_mom", "Combo")] = (
        np.sign(sigs[f"{p}ma_spread_5_20"]) * sigs[f"{p}rsi_14_dev"].abs() / 50.0
    )
    out[_tag(f"{p}trend_vol_adj", "Combo")] = sigs[f"{p}ma_spread_5_20"] / (atr_norm + 1e-9)
    out[_tag(f"{p}mom_accel_combo", "Combo")] = sigs[f"{p}rsi_14_dev"] * np.sign(
        sigs[f"{p}accel_rsi14"]
    )
    out[_tag(f"{p}donchian_rsi", "Combo")] = sigs[f"{p}donchian_pos_20"] * (
        sigs[f"{p}rsi_14_dev"] / 50.0
    )
    out[_tag(f"{p}vol_regime_trend", "Combo")] = sigs[f"{p}ma_spread_5_20"] * (
        1.0 - atr_norm / atr_ma60
    )
    return out


# ── Multi-scale signal builder ────────────────────────────────────────────────

SCALE_MAP: dict[str, int] = {"H1": 1, "H4": 4, "D": 24, "W": 120}


def build_multiscale_signals(
    df: pd.DataFrame,
    window_1y: int,
    scales: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute 52 signals at each scale, concatenated into a single DataFrame.

    With default 4 scales (H1, H4, D, W), produces 208 features all on a
    single data stream -- no cross-TF alignment or ffill needed.
    """
    if scales is None:
        scales = SCALE_MAP
    parts = []
    for label, mult in scales.items():
        prefix = f"{label}_" if mult > 1 else ""
        logger.info(
            "Computing %d-scale signals (prefix='%s', period_mult=%d)...", mult, label, mult
        )
        signals = build_all_signals(df, window_1y, period_scale=mult, name_prefix=prefix)
        parts.append(signals)
        logger.info("  -> %d signals computed", len(signals.columns))
    result = pd.concat(parts, axis=1)
    logger.info("Total multi-scale features: %d", len(result.columns))
    return result


# ── Display helpers ────────────────────────────────────────────────────────────


def _verdict(ic: float, icir: float, ar1: float = 1.0, bh_sig: bool = True) -> str:
    """Classify a signal by its IC and ICIR into a qualitative tier.

    bh_sig: whether the signal survives BH FDR correction across all tests.
    Signals that do not survive BH are capped at WEAK even if IC >= 0.05,
    because the apparent edge may be a false discovery.
    """
    abs_ic = abs(ic) if not np.isnan(ic) else 0.0
    abs_ir = abs(icir) if not np.isnan(icir) else 0.0
    ar1_ok = (not np.isnan(ar1)) and ar1 > 0.3
    if abs_ic >= 0.05 and abs_ir >= 0.5 and ar1_ok and bh_sig:
        return "STRONG"
    elif abs_ic >= 0.05 and bh_sig:
        return "USABLE"
    elif abs_ic >= 0.03:
        return "WEAK"
    return "NOISE"


def _print_leaderboard(
    ic_df: pd.DataFrame,
    icir_s: pd.Series,
    icir_nw_s: pd.Series,
    ar1_s: pd.Series,
    instrument: str,
    timeframe: str,
    n_bars: int,
    regime_rows: dict[str, dict] | None = None,
    bh_sig_map: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Print ranked leaderboard. regime_rows: signal -> {ic_unconditional,
    ic_trending, ic_ranging, flip}. Shown as extra columns when provided.

    bh_sig_map: signal -> bool (True if signal survives BH FDR correction across
    all tests including regime-split). Signals not in map default to True (no
    penalty when regime data is unavailable).
    """
    best_h_col = ic_df.abs().idxmax(axis=1)
    best_ic = pd.Series(
        {sig: ic_df.loc[sig, col] for sig, col in best_h_col.items()},
        name="best_ic",
    )
    best_h_str = best_h_col.str.replace("fwd_", "h=", regex=False)

    rows = []
    for sig in ic_df.index:
        ic_val = best_ic[sig]
        ir_val = icir_s.get(sig, np.nan)
        ir_nw_val = icir_nw_s.get(sig, np.nan)
        ar1_val = ar1_s.get(sig, np.nan)
        # Use BH significance if available; default True when map not provided
        bh_sig = bh_sig_map.get(sig, True) if bh_sig_map is not None else True
        row: dict = {
            "signal": sig,
            "group": _SIGNAL_GROUP.get(sig, "?"),
            "best_h": best_h_str[sig],
            "ic": ic_val,
            "icir": ir_val,
            "icir_nw": ir_nw_val,
            "ar1": ar1_val,
            "bh_significant": bh_sig,
            "verdict": _verdict(
                ic_val,
                ir_nw_val if not pd.isna(ir_nw_val) else ir_val,
                ar1_val,
                bh_sig=bh_sig,
            ),
        }
        if regime_rows and sig in regime_rows:
            row.update(regime_rows[sig])
        rows.append(row)

    df_rank = (
        pd.DataFrame(rows)
        .assign(abs_ic=lambda x: x["ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
    )

    has_regime = regime_rows is not None
    W = 100 if has_regime else 82
    print("\n" + "=" * W)
    print(f"  IC SIGNAL SWEEP -- {instrument} {timeframe}  [Phase 1]")
    print(f"  Signals: {len(df_rank)}  |  Horizons: {HORIZONS}  |  Bars: {n_bars:,}")
    print("=" * W)
    print()
    print("  LEADERBOARD (ranked by |IC| at best horizon)")
    print("  " + "-" * (W - 2))
    hdr = (
        f"  {'Rank':>4}  {'Signal':<22}  {'Grp':>5}  "
        f"{'BestH':>6}  {'IC':>8}  {'ICIR':>7}  {'AR1':>6}  {'Verdict':<8}"
    )
    if has_regime:
        hdr += f"  {'Uncond':>8}  {'Trend':>8}  {'Ranging':>8}  {'FLIP?':<6}"
    print(hdr)
    print("  " + "-" * (W - 2))

    for i, row in df_rank.iterrows():
        ic_str = f"{row['ic']:>+8.4f}" if not np.isnan(row["ic"]) else "     NaN"
        ir_str = f"{row['icir']:>+7.3f}" if not np.isnan(row["icir"]) else "    NaN"
        ar_str = f"{row['ar1']:>+6.3f}" if not np.isnan(row["ar1"]) else "   NaN"
        line = (
            f"  {i + 1:>4}  {row['signal']:<22}  {row['group']:>5}  "
            f"{row['best_h']:>6}  {ic_str}  {ir_str}  {ar_str}  {row['verdict']:<8}"
        )
        if has_regime and "ic_unconditional" in row:
            unc = row.get("ic_unconditional", np.nan)
            trd = row.get("ic_trending", np.nan)
            rng = row.get("ic_ranging", np.nan)
            flip = row.get("flip", False)
            unc_s = f"{unc:>+8.4f}" if not np.isnan(unc) else "     NaN"
            trd_s = f"{trd:>+8.4f}" if not np.isnan(trd) else "     NaN"
            rng_s = f"{rng:>+8.4f}" if not np.isnan(rng) else "     NaN"
            flip_s = "**FLIP**" if flip else "      "
            line += f"  {unc_s}  {trd_s}  {rng_s}  {flip_s}"
        print(line)

    print("  " + "-" * (W - 2))
    counts = df_rank["verdict"].value_counts()
    print(f"  STRONG  (|IC|>=0.05, ICIR>=0.5) : {counts.get('STRONG', 0):>3} signals")
    print(f"  USABLE  (|IC|>=0.05, ICIR< 0.5) : {counts.get('USABLE', 0):>3} signals")
    print(f"  WEAK    (0.03<=|IC|<0.05)        : {counts.get('WEAK', 0):>3} signals")
    print(f"  NOISE   (|IC|<0.03)              : {counts.get('NOISE', 0):>3} signals")

    if has_regime and "flip" in df_rank.columns:
        n_flip = int(df_rank["flip"].sum())
        if n_flip:
            print(
                f"\n  *** {n_flip} FLIP signal(s) detected "
                "(sign reversal across trending/ranging regimes) ***"
            )
            flip_sigs = df_rank[df_rank["flip"]]["signal"].tolist()
            print(f"  FLIP signals: {', '.join(flip_sigs)}")
    print("=" * W)
    return df_rank


def _print_decile_plots(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    ranked: pd.DataFrame,
    top_n: int = 5,
    n_bins: int = 10,
) -> None:
    print(f"\n  TOP {top_n} DECILE PLOTS")
    print("  " + "-" * 60)
    for _, row in ranked.head(top_n).iterrows():
        sig = row["signal"]
        fwd_col = row["best_h"].replace("h=", "fwd_")
        print(f"\n  [{row['verdict']}] {sig}  (best {row['best_h']}, IC={row['ic']:+.4f})")
        if sig not in signals.columns or fwd_col not in fwd_returns.columns:
            print("  (data not available)")
            continue
        qs = quantile_spread(signals[sig], fwd_returns[fwd_col], n_bins=n_bins)
        if qs.empty:
            print("  (insufficient data for quantile spread)")
            continue
        max_ret = qs["mean_fwd_return"].abs().max()
        scale = 1.0 / max_ret if max_ret > 0 else 1.0
        print(f"  {'Bin':<5}  {'MeanFwdReturn':>14}  {'N':>5}  Chart")
        print("  " + "-" * 50)
        for idx, qrow in qs.iterrows():
            bar = _bar(float(qrow["mean_fwd_return"]) * scale)
            print(f"  {idx:<5}  {qrow['mean_fwd_return']:>+14.6f}  {int(qrow['n_obs']):>5}  {bar}")


# ── Regime-conditional IC helpers ─────────────────────────────────────────────


def _compute_regime_ic(
    all_signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, float], dict[str, float]]:
    """Compute per-signal IC split by adx_regime (trending vs ranging).

    Returns
    -------
    result : dict signal -> {ic_unconditional, ic_trending, ic_ranging, flip}
    pv_trending : dict signal -> p-value for best-horizon trending IC
    pv_ranging  : dict signal -> p-value for best-horizon ranging IC

    The p-value dicts are used to extend BH FDR correction to regime-split tests
    (otherwise only the 260 unconditional tests are corrected, missing 104 regime tests).
    """
    regime_aligned = regime_df["adx_regime"].reindex(all_signals.index)

    trending_mask = regime_aligned == "trending"
    ranging_mask = regime_aligned == "ranging"

    n_trending = int(trending_mask.sum())
    n_ranging = int(ranging_mask.sum())
    logger.info(
        "Regime split: %d trending bars, %d ranging bars (of %d total)",
        n_trending,
        n_ranging,
        len(all_signals),
    )

    result: dict[str, dict] = {}
    pv_trending: dict[str, float] = {}
    pv_ranging: dict[str, float] = {}

    from scipy import stats  # local import to keep top-level imports clean

    def _spearman_ic(s: pd.Series, f: pd.DataFrame) -> tuple[float, float]:
        """IC and p-value at best horizon for the given subset.

        Returns (best_rho, pvalue_at_best_rho). p-value is np.nan when
        no horizon has >= 30 paired observations.
        """
        best_rho = np.nan
        best_pv = np.nan
        for col in f.columns:
            paired = pd.concat([s, f[col]], axis=1).dropna()
            if len(paired) < 30:
                continue
            rho, pv = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
            if np.isnan(best_rho) or abs(rho) > abs(best_rho):
                best_rho = float(rho)
                best_pv = float(pv)
        return best_rho, best_pv

    for sig in all_signals.columns:
        sig_series = all_signals[sig]

        ic_unc, _ = _spearman_ic(sig_series, fwd_returns)

        if n_trending >= 30:
            ic_trend, pv_trend = _spearman_ic(sig_series[trending_mask], fwd_returns[trending_mask])
        else:
            ic_trend, pv_trend = np.nan, np.nan

        if n_ranging >= 30:
            ic_range, pv_range = _spearman_ic(sig_series[ranging_mask], fwd_returns[ranging_mask])
        else:
            ic_range, pv_range = np.nan, np.nan

        # FLIP: both regimes have |IC| >= 0.03 AND opposite signs
        flip = (
            not np.isnan(ic_trend)
            and not np.isnan(ic_range)
            and abs(ic_trend) >= 0.03
            and abs(ic_range) >= 0.03
            and np.sign(ic_trend) != np.sign(ic_range)
        )

        result[sig] = {
            "ic_unconditional": round(ic_unc, 5) if not np.isnan(ic_unc) else np.nan,
            "ic_trending": round(ic_trend, 5) if not np.isnan(ic_trend) else np.nan,
            "ic_ranging": round(ic_range, 5) if not np.isnan(ic_range) else np.nan,
            "flip": flip,
        }
        if not np.isnan(pv_trend):
            pv_trending[sig] = pv_trend
        if not np.isnan(pv_range):
            pv_ranging[sig] = pv_range

    return result, pv_trending, pv_ranging


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_sweep(
    instrument: str,
    timeframe: str,
    horizons: list[int] | None = None,
    n_bins: int = 10,
    data_dir: Path | None = None,
    fmt: str = "parquet",
    regime_df: pd.DataFrame | None = None,
    is_only: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Run the full Phase 1 sweep for one instrument/timeframe.

    Parameters
    ----------
    instrument:  Instrument name, e.g. "EUR_USD" or "SPY".
    timeframe:   Timeframe label, e.g. "H4", "D", "1yr_5m".
    horizons:    Forward return horizons in bars. Defaults to HORIZONS.
    n_bins:      Number of quantile bins for decile plots.
    data_dir:    Override for data directory (defaults to ROOT/data).
    fmt:         File format -- "parquet" or "csv".
    regime_df:   Optional DataFrame with adx_regime column (from phase0_regime.py).
                 If None, the function tries to auto-load from
                 .tmp/regime/{instrument}_{timeframe}_regime.parquet.
    strict:      If True, data quality warnings become fatal errors (see L2).

    Returns
    -------
    pd.DataFrame  Ranked leaderboard with IC, ICIR, verdict, and optional regime
                  columns if regime data was available.
    """
    global _STRICT_VALIDATION
    _STRICT_VALIDATION = strict
    if horizons is None:
        horizons = HORIZONS

    # Auto-load regime file if not supplied
    if regime_df is None:
        tf_slug = timeframe.lower().split("_")[-1] if "_" in timeframe else timeframe.lower()
        regime_path = ROOT / ".tmp" / "regime" / f"{instrument}_{tf_slug}_regime.parquet"
        if regime_path.exists():
            try:
                regime_df = pd.read_parquet(regime_path)
                if not isinstance(regime_df.index, pd.DatetimeIndex):
                    regime_df.index = pd.to_datetime(regime_df.index, utc=True)
                if "adx_regime" not in regime_df.columns:
                    logger.warning(
                        "Regime file missing 'adx_regime' column -- ignoring: %s",
                        regime_path,
                    )
                    regime_df = None
                else:
                    logger.info("Auto-loaded regime file: %s", regime_path)
            except Exception as exc:
                logger.warning("Could not load regime file %s: %s", regime_path, exc)
                regime_df = None

    df = _load_ohlcv(instrument, timeframe, data_dir=data_dir, fmt=fmt)
    close = df["close"]

    window_1y = _get_annual_bars(timeframe)
    logger.info("Timeframe '%s' -> using %d bars for 1-year rolling windows", timeframe, window_1y)

    logger.info("Computing 52 signals across 7 groups...")
    all_signals = build_all_signals(df, window_1y)

    fwd_returns = compute_forward_returns(close, horizons, vol_adjust=True)

    # Drop warmup rows where any group has no valid signal yet
    valid = all_signals.notna().any(axis=1) & fwd_returns.notna().any(axis=1)

    if is_only:
        is_cutoff = int(len(df) * 0.70)
        logger.warning(
            "IS-only mode: IC computed on first 70%% of data (%d bars) to prevent selection bias.",
            is_cutoff,
        )
        is_mask = np.zeros(len(valid), dtype=bool)
        is_mask[:is_cutoff] = True
        valid = valid & is_mask

    all_signals = all_signals[valid]
    fwd_returns = fwd_returns[valid]
    n_bars = len(all_signals)
    logger.info("Analysis window: %d bars after warmup", n_bars)

    logger.info("Computing IC table (52 signals x %d horizons)...", len(horizons))
    ic_df, pv_df = compute_ic_table(all_signals, fwd_returns, return_pvalues=True)

    # G1 FIX: Apply BH FDR correction across all signal×horizon p-values.
    from research.ic_analysis.run_ic import apply_bh_fdr

    adj_pv_df, reject_df = apply_bh_fdr(pv_df, alpha=0.05)

    # Compute best horizons first, then ICIR at each signal's best horizon
    best_h_col = ic_df.abs().idxmax(axis=1)
    best_horizons_dict = {sig: col for sig, col in best_h_col.items()}

    logger.info(
        "Computing rolling IC map (%d signals, window=%d bars, parallel)...",
        len(all_signals.columns),
        window_1y,
    )
    rolling_ic_map = build_rolling_ic_map(
        all_signals,
        fwd_returns,
        horizons,
        window=window_1y,
        best_horizons=best_horizons_dict,
    )

    logger.info("Computing ICIR (window=%d bars, at best horizon)...", window_1y)
    icir_s = compute_icir(
        all_signals,
        fwd_returns,
        horizons,
        window=window_1y,
        best_horizons=best_horizons_dict,
        rolling_ic_map=rolling_ic_map,
    )

    logger.info("Computing Newey-West ICIR (window=%d bars, at best horizon)...", window_1y)
    icir_nw_s = compute_icir_nw(
        all_signals,
        fwd_returns,
        horizons,
        window=window_1y,
        best_horizons=best_horizons_dict,
        rolling_ic_map=rolling_ic_map,
    )

    logger.info("Computing Autocorrelation (AR1)...")
    ar1_s = compute_autocorrelation(all_signals)

    # Regime-conditional IC (optional).
    # _compute_regime_ic() now also returns regime p-values so we can extend
    # BH correction from 260 unconditional tests to ~364 (+ 52 trending + 52 ranging).
    regime_rows: dict[str, dict] | None = None
    pv_trending_regime: dict[str, float] = {}
    pv_ranging_regime: dict[str, float] = {}
    if regime_df is not None:
        logger.info("Computing regime-conditional IC (trending / ranging)...")
        regime_rows, pv_trending_regime, pv_ranging_regime = _compute_regime_ic(
            all_signals, fwd_returns, regime_df
        )

    # H3 FIX: Extend BH FDR to include regime-split p-values.
    # Build a combined p-value DataFrame: index=signal, columns=test_label.
    # Unconditional tests: 52 signals × 5 horizons = 260 tests.
    # Regime tests: up to 52 trending + 52 ranging = 104 additional tests.
    # Total: up to 364 simultaneous tests corrected together.
    from research.ic_analysis.run_ic import apply_bh_fdr

    all_pv_df = pv_df.copy()  # 52 × 5 unconditional
    if pv_trending_regime:
        trend_s = pd.Series(pv_trending_regime, name="regime_trending")
        all_pv_df = all_pv_df.join(trend_s, how="left")
    if pv_ranging_regime:
        range_s = pd.Series(pv_ranging_regime, name="regime_ranging")
        all_pv_df = all_pv_df.join(range_s, how="left")

    n_total_tests = all_pv_df.notna().sum().sum()
    logger.info(
        "BH FDR correction applied to %d tests (%d unconditional + %d trending + %d ranging)",
        n_total_tests,
        pv_df.notna().sum().sum(),
        len(pv_trending_regime),
        len(pv_ranging_regime),
    )
    adj_pv_all, reject_all = apply_bh_fdr(all_pv_df, alpha=0.05)

    # A signal is BH-significant if it passes at its best unconditional horizon.
    # (Regime columns are used for FLIP detection BH correction, not primary verdict.)
    bh_sig_map: dict[str, bool] = {}
    for sig in ic_df.index:
        best_h = best_horizons_dict.get(sig)
        if best_h and sig in reject_all.index and best_h in reject_all.columns:
            bh_sig_map[sig] = bool(reject_all.loc[sig, best_h])
        else:
            bh_sig_map[sig] = False

    ranked = _print_leaderboard(
        ic_df,
        icir_s,
        icir_nw_s,
        ar1_s,
        instrument,
        timeframe,
        n_bars,
        regime_rows=regime_rows,
        bh_sig_map=bh_sig_map,
    )

    _print_decile_plots(all_signals, fwd_returns, ranked, top_n=5, n_bins=n_bins)

    # ── Save outputs ──────────────────────────────────────────────────────────
    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    tf_slug = timeframe.lower().split("_")[-1] if "_" in timeframe else timeframe.lower()
    inst_slug = instrument.lower()

    ic_path = report_dir / f"phase1_{inst_slug}_{tf_slug}.csv"
    icir_path = report_dir / f"phase1_icir_{inst_slug}_{tf_slug}.csv"

    # Build save columns -- always include base columns + BH flag
    save_cols = [
        "signal",
        "group",
        "best_h",
        "ic",
        "icir",
        "icir_nw",
        "ar1",
        "bh_significant",
        "verdict",
    ]
    if regime_rows is not None:
        save_cols += ["ic_unconditional", "ic_trending", "ic_ranging", "flip"]

    ranked[save_cols].to_csv(ic_path, index=False)
    icir_s.to_csv(icir_path, header=True)
    logger.info("Phase 1 leaderboard saved : %s", ic_path)
    logger.info("ICIR time series saved    : %s", icir_path)

    return ranked


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 -- 52-Signal IC/ICIR Sweep")
    parser.add_argument("--instrument", default="EUR_USD", help="Instrument name")
    parser.add_argument(
        "--timeframe",
        default="H4",
        help="Timeframe (H1/H4/D/1yr_5m/...)",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help="Comma-separated forward horizons, e.g. 1,5,10,20,60",
    )
    parser.add_argument("--n_bins", type=int, default=10, help="Decile bin count")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: data/). E.g. data/databento",
    )
    parser.add_argument(
        "--fmt",
        default="parquet",
        choices=["parquet", "csv"],
        help="File format: parquet (default) or csv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=("Run sweep on all instruments found in --data-dir matching the timeframe"),
    )
    parser.add_argument(
        "--is-only",
        action="store_true",
        help="IS-only mode: IC computed on first 70%% of data to prevent selection bias",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Strict mode: convert data quality warnings to fatal errors. "
            "Use for production/pipeline runs to prevent silently invalid IC results "
            "(e.g. a dataset with 20%% stale bars)."
        ),
    )
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    data_dir = Path(args.data_dir) if args.data_dir else None
    effective_dir = data_dir if data_dir else ROOT / "data"

    if args.all:
        pattern = f"*_{args.timeframe}.{args.fmt}"
        files = sorted(effective_dir.glob(pattern))
        if not files:
            print(f"No files found matching {effective_dir / pattern}")
            return
        instruments = [f.stem.replace(f"_{args.timeframe}", "") for f in files]
        print(f"Found {len(instruments)} instruments: {', '.join(instruments)}")
        failed = []
        for i, inst in enumerate(instruments, 1):
            print(f"\n[{i}/{len(instruments)}] {inst}")
            try:
                run_sweep(
                    instrument=inst,
                    timeframe=args.timeframe,
                    horizons=horizons,
                    n_bins=args.n_bins,
                    data_dir=effective_dir,
                    fmt=args.fmt,
                    is_only=args.is_only,
                    strict=args.strict,
                )
            except Exception as exc:
                logger.error("FAILED %s: %s", inst, exc)
                failed.append(inst)
        print(f"\nDone. {len(instruments) - len(failed)}/{len(instruments)} succeeded.")
        if failed:
            print(f"Failed: {', '.join(failed)}")
    else:
        run_sweep(
            instrument=args.instrument,
            timeframe=args.timeframe,
            horizons=horizons,
            n_bins=args.n_bins,
            data_dir=data_dir,
            fmt=args.fmt,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
