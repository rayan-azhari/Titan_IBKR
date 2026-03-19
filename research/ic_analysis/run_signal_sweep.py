"""run_signal_sweep.py -- Comprehensive 52-signal IC/ICIR sweep.

Extends run_ic.py with all major indicator families, acceleration signals,
structural breakout signals, and semantic combinations. Outputs a ranked
leaderboard by |IC|/ICIR to identify signals with genuine predictive edge.

Usage:
    uv run python research/ic_analysis/run_signal_sweep.py
    uv run python research/ic_analysis/run_signal_sweep.py --instrument EUR_USD --timeframe D

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
    compute_autocorrelation,
    compute_forward_returns,
    compute_ic_table,
    compute_icir,
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

HORIZONS = [1, 5, 10, 20, 60]
ICIR_WINDOW = 60

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


def _load_ohlcv(instrument: str, timeframe: str) -> pd.DataFrame:
    path = ROOT / "data" / f"{instrument}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
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
    logger.info(
        "Loaded %d bars | %s %s (%s - %s)",
        len(df), instrument, timeframe,
        df.index[0].date(), df.index[-1].date(),
    )
    return df


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
    out[_tag("ma_spread_5_20",   "Trend")] = (e5 - e20) / e20
    out[_tag("ma_spread_10_50",  "Trend")] = (e10 - e50) / e50
    out[_tag("ma_spread_20_100", "Trend")] = (e20 - e100) / e100
    out[_tag("ma_spread_50_200", "Trend")] = (e50 - e200) / e200
    out[_tag("wma_spread_5_20",  "Trend")] = (w5 - w20) / w20.replace(0, np.nan)
    out[_tag("price_vs_sma20",   "Trend")] = (close - s20) / s20
    out[_tag("price_vs_sma50",   "Trend")] = (close - s50) / s50
    out[_tag("macd_norm",        "Trend")] = (e12 - e26) / roll_std20.replace(0, np.nan)
    out[_tag("ema_slope_10",     "Trend")] = (e10 - e10.shift(5)) / e10.shift(5)
    out[_tag("ema_slope_20",     "Trend")] = (e20 - e20.shift(10)) / e20.shift(10)
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
    out[_tag("rsi_7_dev",      "Momen")] = rsi(close, 7) - 50.0
    out[_tag("rsi_14_dev",     "Momen")] = rsi(close, 14) - 50.0
    out[_tag("rsi_21_dev",     "Momen")] = rsi(close, 21) - 50.0
    out[_tag("stoch_k_dev",    "Momen")] = stoch_k_raw - 50.0
    out[_tag("stoch_d_dev",    "Momen")] = stoch_d_raw - 50.0
    out[_tag("cci_20",         "Momen")] = (
        (close - s20) / (0.015 * mad20.replace(0, np.nan))
    )
    out[_tag("williams_r_dev", "Momen")] = wr14 + 50.0
    out[_tag("roc_3",          "Momen")] = log_c - log_c.shift(3)
    out[_tag("roc_10",         "Momen")] = log_c - log_c.shift(10)
    out[_tag("roc_20",         "Momen")] = log_c - log_c.shift(20)
    out[_tag("roc_60",         "Momen")] = log_c - log_c.shift(60)
    return out


def _compute_group_c(close: pd.Series) -> pd.DataFrame:
    """Group C: Mean Reversion (6 signals)."""
    out = pd.DataFrame(index=close.index)
    for w in (20, 50):
        s = sma(close, w)
        std = close.rolling(w).std().replace(0, np.nan)
        out[_tag(f"bb_zscore_{w}", "MRev")] = (close - s) / (2.0 * std)
        out[_tag(f"zscore_{w}",    "MRev")] = (close - s) / std
    s100 = sma(close, 100)
    std100 = close.rolling(100).std().replace(0, np.nan)
    out[_tag("zscore_100",       "MRev")] = (close - s100) / std100
    exp_mean = close.expanding().mean()
    exp_std = close.expanding().std().replace(0, np.nan)
    out[_tag("zscore_expanding", "MRev")] = (close - exp_mean) / exp_std
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
    out[_tag("norm_atr_14",     "Vol")] = atr(df, 14) / close
    out[_tag("realized_vol_5",  "Vol")] = log_r.rolling(5).std() * np.sqrt(252)
    out[_tag("realized_vol_20", "Vol")] = log_r.rolling(20).std() * np.sqrt(252)
    out[_tag("garman_klass",    "Vol")] = gk_roll
    out[_tag("parkinson_vol",   "Vol")] = pk_roll
    out[_tag("bb_width",        "Vol")] = bollinger_bw(close, 20)
    out[_tag("adx_14",          "Vol")] = adx(df, 14)
    return out


def _compute_group_e(sigs: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Group E: Acceleration / Deceleration (7 signals). diff(1) of base signals."""
    out = pd.DataFrame(index=sigs.index)
    out[_tag("accel_roc10",    "Accel")] = sigs["roc_10"].diff(1)
    out[_tag("accel_rsi14",    "Accel")] = sigs["rsi_14_dev"].diff(1)
    out[_tag("accel_macd",     "Accel")] = macd_hist(close, 12, 26, 9).diff(1)
    out[_tag("accel_atr",      "Accel")] = sigs["norm_atr_14"].diff(1)
    out[_tag("accel_bb_width", "Accel")] = sigs["bb_width"].diff(1)
    out[_tag("accel_rvol20",   "Accel")] = sigs["realized_vol_20"].diff(1)
    out[_tag("accel_stoch_k",  "Accel")] = sigs["stoch_k_dev"].diff(1)
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
    out[_tag("keltner_pos",       "Struct")] = (close - e20) / (2.0 * atr10)
    out[_tag("price_pct_rank_20", "Struct")] = close.rolling(20).rank(pct=True) - 0.5
    out[_tag("price_pct_rank_60", "Struct")] = close.rolling(60).rank(pct=True) - 0.5
    return out


def _compute_group_g(sigs: pd.DataFrame) -> pd.DataFrame:
    """Group G: Semantic Combinations (5 signals). Built from A/B/D/E/F."""
    atr_norm = sigs["norm_atr_14"]
    atr_ma60 = atr_norm.rolling(60).mean().replace(0, np.nan)

    out = pd.DataFrame(index=sigs.index)
    # Trend direction gates momentum magnitude
    out[_tag("trend_mom",        "Combo")] = (
        np.sign(sigs["ma_spread_5_20"]) * sigs["rsi_14_dev"].abs() / 50.0
    )
    # Trend signal normalised by volatility regime
    out[_tag("trend_vol_adj",    "Combo")] = (
        sigs["ma_spread_5_20"] / (atr_norm + 1e-9)
    )
    # Momentum × whether it is accelerating or decelerating
    out[_tag("mom_accel_combo",  "Combo")] = (
        sigs["rsi_14_dev"] * np.sign(sigs["accel_rsi14"])
    )
    # Structural breakout gated by momentum
    out[_tag("donchian_rsi",     "Combo")] = (
        sigs["donchian_pos_20"] * (sigs["rsi_14_dev"] / 50.0)
    )
    # Trend attenuated in elevated-vol regime
    out[_tag("vol_regime_trend", "Combo")] = (
        sigs["ma_spread_5_20"] * (1.0 - atr_norm / atr_ma60)
    )
    return out


def build_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 52 signals. Groups A->B->C->D, then E (uses A/B/D), F, G."""
    close = df["close"]
    a = _compute_group_a(close)
    b = _compute_group_b(df)
    c = _compute_group_c(close)
    d = _compute_group_d(df)
    base = pd.concat([a, b, c, d], axis=1)
    e = _compute_group_e(base, close)
    f = _compute_group_f(df)
    all_so_far = pd.concat([base, e, f], axis=1)
    g = _compute_group_g(all_so_far)
    return pd.concat([all_so_far, g], axis=1)


# ── Display helpers ────────────────────────────────────────────────────────────


def _verdict(ic: float, icir: float, ar1: float = 1.0) -> str:
    abs_ic = abs(ic) if not np.isnan(ic) else 0.0
    abs_ir = abs(icir) if not np.isnan(icir) else 0.0
    ar1_ok = (not np.isnan(ar1)) and ar1 > 0.3
    if abs_ic >= 0.05 and abs_ir >= 0.5 and ar1_ok:
        return "STRONG"
    elif abs_ic >= 0.05:
        return "USABLE"
    elif abs_ic >= 0.03:
        return "WEAK"
    return "NOISE"


def _print_leaderboard(
    ic_df: pd.DataFrame,
    icir_s: pd.Series,
    ar1_s: pd.Series,
    instrument: str,
    timeframe: str,
    n_bars: int,
) -> pd.DataFrame:
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
        ar1_val = ar1_s.get(sig, np.nan)
        rows.append({
            "signal": sig,
            "group": _SIGNAL_GROUP.get(sig, "?"),
            "best_h": best_h_str[sig],
            "ic": ic_val,
            "icir": ir_val,
            "ar1": ar1_val,
            "verdict": _verdict(ic_val, ir_val, ar1_val),
        })

    df_rank = (
        pd.DataFrame(rows)
        .assign(abs_ic=lambda x: x["ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
    )

    W = 82
    print("\n" + "=" * W)
    print(f"  IC SIGNAL SWEEP -- {instrument} {timeframe}")
    print(f"  Signals: {len(df_rank)}  |  Horizons: {HORIZONS}  |  Bars: {n_bars:,}")
    print("=" * W)
    print()
    print("  LEADERBOARD (ranked by |IC| at best horizon)")
    print("  " + "-" * (W - 2))
    print(
        f"  {'Rank':>4}  {'Signal':<22}  {'Grp':>5}  "
        f"{'BestH':>6}  {'IC':>8}  {'ICIR':>7}  {'AR1':>6}  Verdict"
    )
    print("  " + "-" * (W - 2))
    for i, row in df_rank.iterrows():
        ic_str = f"{row['ic']:>+8.4f}" if not np.isnan(row["ic"]) else "     NaN"
        ir_str = f"{row['icir']:>+7.3f}" if not np.isnan(row["icir"]) else "    NaN"
        ar_str = f"{row['ar1']:>+6.3f}" if not np.isnan(row["ar1"]) else "   NaN"
        print(
            f"  {i + 1:>4}  {row['signal']:<22}  {row['group']:>5}  "
            f"{row['best_h']:>6}  {ic_str}  {ir_str}  {ar_str}  {row['verdict']}"
        )

    print("  " + "-" * (W - 2))
    counts = df_rank["verdict"].value_counts()
    print(f"  STRONG  (|IC|>=0.05, ICIR>=0.5) : {counts.get('STRONG', 0):>3} signals")
    print(f"  USABLE  (|IC|>=0.05, ICIR< 0.5) : {counts.get('USABLE', 0):>3} signals")
    print(f"  WEAK    (0.03<=|IC|<0.05)        : {counts.get('WEAK',   0):>3} signals")
    print(f"  NOISE   (|IC|<0.03)              : {counts.get('NOISE',  0):>3} signals")
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
        print(
            f"\n  [{row['verdict']}] {sig}  "
            f"(best {row['best_h']}, IC={row['ic']:+.4f})"
        )
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
            print(
                f"  {idx:<5}  {qrow['mean_fwd_return']:>+14.6f}  "
                f"{int(qrow['n_obs']):>5}  {bar}"
            )


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_sweep(
    instrument: str,
    timeframe: str,
    horizons: list[int] | None = None,
    n_bins: int = 10,
) -> None:
    if horizons is None:
        horizons = HORIZONS

    df = _load_ohlcv(instrument, timeframe)
    close = df["close"]

    logger.info("Computing 52 signals across 7 groups...")
    all_signals = build_all_signals(df)

    fwd_returns = compute_forward_returns(close, horizons, vol_adjust=True)

    # Drop warmup rows where any group has no valid signal yet
    valid = all_signals.notna().any(axis=1) & fwd_returns.notna().any(axis=1)
    all_signals = all_signals[valid]
    fwd_returns = fwd_returns[valid]
    n_bars = len(all_signals)
    logger.info("Analysis window: %d bars after warmup", n_bars)

    logger.info("Computing IC table (52 signals x %d horizons)...", len(horizons))
    ic_df = compute_ic_table(all_signals, fwd_returns)

    # Compute best horizons first, then ICIR at each signal's best horizon
    best_h_col = ic_df.abs().idxmax(axis=1)
    best_horizons_dict = {sig: col for sig, col in best_h_col.items()}

    logger.info("Computing ICIR (window=%d bars, at best horizon)...", ICIR_WINDOW)
    icir_s = compute_icir(all_signals, fwd_returns, horizons, window=ICIR_WINDOW,
                          best_horizons=best_horizons_dict)

    logger.info("Computing Autocorrelation (AR1)...")
    ar1_s = compute_autocorrelation(all_signals)

    ranked = _print_leaderboard(ic_df, icir_s, ar1_s, instrument, timeframe, n_bars)
    _print_decile_plots(all_signals, fwd_returns, ranked, top_n=5, n_bins=n_bins)

    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{instrument}_{timeframe}".lower()
    ic_path = report_dir / f"ic_sweep_{slug}.csv"
    icir_path = report_dir / f"icir_sweep_{slug}.csv"
    ranked.to_csv(ic_path, index=False)
    icir_s.to_csv(icir_path, header=True)
    logger.info("IC table saved: %s", ic_path)
    logger.info("ICIR saved: %s", icir_path)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="52-Signal IC/ICIR Sweep")
    parser.add_argument("--instrument", default="EUR_USD", help="Instrument name")
    parser.add_argument("--timeframe", default="H4", help="Timeframe (H1/H4/D/W)")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help="Comma-separated forward horizons, e.g. 1,5,10,20,60",
    )
    parser.add_argument("--n_bins", type=int, default=10, help="Decile bin count")
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    run_sweep(
        instrument=args.instrument,
        timeframe=args.timeframe,
        horizons=horizons,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
