"""
run_ic.py -- Information Coefficient (IC) Signal Scanner

Pre-backtest signal validation. Converts strategy ideas into continuous signals
and measures their correlation with forward returns (Spearman rank correlation).
Answers the question: does this signal have genuine predictive edge?

Usage:
    python research/ic_analysis/run_ic.py --instrument QQQ --timeframe D
    python research/ic_analysis/run_ic.py --instrument EUR_USD --timeframe H4
    python research/ic_analysis/run_ic.py --instrument QQQ --timeframe D --horizons 1,5,10,20

Signals tested:
    macd_norm   -- (EMA12 - EMA26) / rolling_std(20): MACD normalised by vol
    ma_spread   -- (EMA5 - EMA20) / EMA20: continuous MA crossover strength
    rsi_dev     -- RSI(14) - 50: RSI centred at 0
    bb_zscore   -- (close - SMA20) / (2 * std20): Bollinger z-score
    momentum_5  -- log(close / close.shift(5)): 5-bar momentum
    momentum_20 -- log(close / close.shift(20)): 20-bar momentum

Interpretation:
    |IC| < 0.03             -- noise, discard the signal
    0.03 <= |IC| < 0.05     -- weak signal, needs regime conditioning
    |IC| >= 0.05            -- usable signal
    ICIR (IC/std) > 0.5     -- signal is consistent enough to trade
    Monotonic quantile spread -- signal has linear predictive edge

Look-ahead safety:
    Signal computation uses only .rolling() / .ewm() with positive windows (causal).
    Forward returns use close.shift(-h) -- intentionally looks forward (this is
    the TARGET, not an input to the signal).
"""

import argparse
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
N_BINS = 10
ROLLING_IC_WINDOW = 252  # bars for rolling ICIR computation
MTF_CONFIG_PATH = ROOT / "config" / "mtf.toml"

# MTF timeframe definitions: (timeframe_label, fast_ma, slow_ma, rsi_period)
MTF_TIMEFRAMES = {
    "H1": ("H1", "fast_ma", "slow_ma", "rsi_period"),
    "H4": ("H4", "fast_ma", "slow_ma", "rsi_period"),
    "D": ("D", "fast_ma", "slow_ma", "rsi_period"),
    "W": ("W", "fast_ma", "slow_ma", "rsi_period"),
}


# -- Data loading ---------------------------------------------------------------


def load_ohlcv(instrument: str, timeframe: str) -> pd.DataFrame:
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
        f"Loaded {len(df)} bars | {instrument} {timeframe} "
        f"({df.index[0].date()} - {df.index[-1].date()})"
    )
    return df


# -- Signal factories (all causal -- no look-ahead) -----------------------------


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI using EWM (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_mtf_confluence(instrument: str, primary_tf: str) -> pd.Series | None:
    """
    Compute MTF confluence signal aligned to primary_tf bars.

    Each timeframe contributes: (fast_ma > slow_ma ? +0.5 : -0.5) + (rsi > 50 ? +0.5 : -0.5)
    Weighted sum per config/mtf.toml. Returns continuous signal in range [-1, +1].
    Returns None if config or required parquet files are missing.
    """
    if not MTF_CONFIG_PATH.exists():
        logger.warning("mtf.toml not found -- skipping MTF confluence signal")
        return None

    with open(MTF_CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    weights = cfg.get("weights", {})
    tf_labels = [tf for tf in ["H1", "H4", "D", "W"] if tf in weights]

    # Load primary timeframe close for the target index
    primary_path = ROOT / "data" / f"{instrument}_{primary_tf}.parquet"
    if not primary_path.exists():
        return None
    primary_close = load_ohlcv(instrument, primary_tf)["close"]

    contributions = []
    for tf in tf_labels:
        w = weights[tf]
        tf_cfg = cfg.get(tf, {})
        fast = tf_cfg.get("fast_ma", 10)
        slow = tf_cfg.get("slow_ma", 30)
        rsi_p = tf_cfg.get("rsi_period", 14)

        # Load native TF data
        path = ROOT / "data" / f"{instrument}_{tf}.parquet"
        if not path.exists():
            logger.warning(f"Missing {instrument}_{tf}.parquet -- skipping {tf} in MTF")
            continue

        try:
            close_tf = load_ohlcv(instrument, tf)["close"]
        except Exception:
            continue

        # Compute MA crossover and RSI on native TF bars
        fast_ma = close_tf.ewm(span=fast, adjust=False).mean()
        slow_ma = close_tf.ewm(span=slow, adjust=False).mean()
        ma_score = (fast_ma > slow_ma).astype(float) - 0.5   # +0.5 or -0.5
        rsi_score = (_rsi(close_tf, rsi_p) > 50).astype(float) - 0.5  # +0.5 or -0.5
        tf_contribution = (ma_score + rsi_score) * w  # range: [-w, +w]

        # PREVENT LOOKAHEAD BIAS: shift(1) if native TF is higher than primary TF
        if tf != primary_tf:
            tf_contribution = tf_contribution.shift(1)

        # Align to primary TF index using forward fill (causal)
        tf_contribution = (
            tf_contribution
            .reindex(primary_close.index, method="ffill")
        )
        contributions.append(tf_contribution)

    if not contributions:
        logger.warning("No MTF timeframes could be loaded")
        return None

    confluence = pd.concat(contributions, axis=1).sum(axis=1).rename("mtf_confluence")
    logger.info(f"MTF confluence computed: {len(confluence)} bars, "
                f"range [{confluence.min():.3f}, {confluence.max():.3f}]")
    return confluence


def compute_signals(close: pd.Series) -> pd.DataFrame:
    """
    Compute all continuous signals. All computations are causal:
    each value at bar t depends only on bars <= t.
    """
    log_close = np.log(close)

    # -- MACD normalised by rolling vol ----------------------------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    roll_std = close.rolling(20).std()
    macd_norm = (ema12 - ema26) / roll_std.replace(0, np.nan)

    # -- MA spread: (EMA5 - EMA20) / EMA20 ------------------------------------
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ma_spread = (ema5 - ema20) / ema20

    # -- RSI deviation from neutral ---------------------------------------------
    rsi_dev = _rsi(close, 14) - 50.0

    # -- Bollinger z-score ------------------------------------------------------
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_zscore = (close - sma20) / (2.0 * std20.replace(0, np.nan))

    # -- Raw momentum -----------------------------------------------------------
    momentum_5 = log_close - log_close.shift(5)
    momentum_20 = log_close - log_close.shift(20)

    signals = pd.DataFrame(
        {
            "macd_norm": macd_norm,
            "ma_spread": ma_spread,
            "rsi_dev": rsi_dev,
            "bb_zscore": bb_zscore,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
        },
        index=close.index,
    )
    return signals


# -- Forward returns ------------------------------------------------------------


def compute_forward_returns(
    close: pd.Series, horizons: list[int], vol_adjust: bool = True
) -> pd.DataFrame:
    """
    Compute log forward returns at each horizon.
    fwd_h[t] = log(close[t+h] / close[t])

    If vol_adjust=True, normalises the return by the prevailing bar volatility
    scaled by sqrt(h). This prevents macro volatility expansions from dominating
    the Spearman rank.

    close.shift(-h) is intentional -- this is the target (future price), not a
    feature. The signal at bar t is correlated against the return from t to t+h.
    """
    log_close = np.log(close)
    
    if vol_adjust:
        # 20-bar rolling standard deviation of log returns
        bar_vol = log_close.diff().rolling(20).std().replace(0, np.nan)
    else:
        bar_vol = 1.0

    fwd = {}
    for h in horizons:
        v = bar_vol * np.sqrt(h) if vol_adjust else 1.0
        fwd[f"fwd_{h}"] = (log_close.shift(-h) - log_close) / v
        
    return pd.DataFrame(fwd, index=close.index)


def compute_mfe_mae_targets(
    df: pd.DataFrame, horizons: list[int], vol_adjust: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
    MFE is the maximum high achieved over the NEXT h bars (forward-looking).
    MAE is the minimum low achieved over the NEXT h bars (forward-looking).
    Returns both as vol-adjusted log returns relative to current close.

    Uses reverse-roll-reverse to compute a forward-looking rolling window:
        reverse the series -> rolling max/min -> reverse back
    This correctly computes: max(high[t+1 : t+h]) for each bar t.
    """
    close = df["close"]
    log_close = np.log(close)
    log_high = np.log(df["high"])
    log_low = np.log(df["low"])

    if vol_adjust:
        bar_vol = log_close.diff().rolling(20).std().replace(0, np.nan)
    else:
        bar_vol = 1.0

    mfe = {}
    mae = {}

    for h in horizons:
        # Forward-looking rolling max/min via reverse-roll-reverse
        # Shift by -1 first so we exclude bar t itself (we want t+1 to t+h)
        fwd_high = log_high.shift(-1)
        fwd_low = log_low.shift(-1)
        roll_high = fwd_high[::-1].rolling(window=h, min_periods=1).max()[::-1]
        roll_low = fwd_low[::-1].rolling(window=h, min_periods=1).min()[::-1]

        v = bar_vol * np.sqrt(h) if vol_adjust else 1.0

        mfe[f"mfe_{h}"] = (roll_high - log_close) / v
        mae[f"mae_{h}"] = (roll_low - log_close) / v

    return pd.DataFrame(mfe, index=df.index), pd.DataFrame(mae, index=df.index)


# -- IC computation -------------------------------------------------------------


def compute_ic_table(
    signals: pd.DataFrame, fwd_returns: pd.DataFrame,
    return_pvalues: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spearman IC for every signal × horizon pair.
    Drops NaN pairs before correlation -- no imputation.

    Returns DataFrame: index=signal names, columns=horizon labels, values=Spearman ρ.
    If return_pvalues=True, also returns a parallel DataFrame of raw p-values.
    """
    ic = {}
    pv = {}
    for sig_col in signals.columns:
        row_ic = {}
        row_pv = {}
        for fwd_col in fwd_returns.columns:
            df = pd.concat([signals[sig_col], fwd_returns[fwd_col]], axis=1).dropna()
            if len(df) < 30:
                row_ic[fwd_col] = np.nan
                row_pv[fwd_col] = np.nan
            else:
                rho, p = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
                row_ic[fwd_col] = round(float(rho), 5)
                row_pv[fwd_col] = float(p)
        ic[sig_col] = row_ic
        pv[sig_col] = row_pv
    ic_df = pd.DataFrame(ic).T
    if return_pvalues:
        return ic_df, pd.DataFrame(pv).T
    return ic_df


def compute_icir(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons: list[int],
    window: int = ROLLING_IC_WINDOW,
    best_horizons: dict[str, str] | None = None,
) -> pd.Series:
    """
    Rolling ICIR = mean(rolling IC) / std(rolling IC).
    Computed at each signal's best horizon (from best_horizons dict).
    Falls back to the first horizon if best_horizons is not provided.
    Returns Series: index=signal names, values=ICIR.
    """
    icirs = {}
    for sig_col in signals.columns:
        # Determine which horizon to use for this signal
        if best_horizons and sig_col in best_horizons:
            fwd_col = best_horizons[sig_col]
        else:
            fwd_col = f"fwd_{horizons[0]}"

        # Skip if fwd_col not available (e.g. MFE/MAE prefix)
        if fwd_col not in fwd_returns.columns:
            fwd_col = f"fwd_{horizons[0]}"

        fwd = fwd_returns[fwd_col]
        df = pd.concat([signals[sig_col], fwd], axis=1).dropna()
        if len(df) < window:
            icirs[sig_col] = np.nan
            continue
        # rolling Spearman approximation: rank both series, rolling correlation of ranks
        sig_ranked = df.iloc[:, 0].rank()
        fwd_ranked = df.iloc[:, 1].rank()
        roll_corr = sig_ranked.rolling(window).corr(fwd_ranked)
        mean_ic = roll_corr.mean()
        std_ic = roll_corr.std()
        icirs[sig_col] = round(mean_ic / std_ic if std_ic > 0 else np.nan, 3)
    return pd.Series(icirs, name="icir")


def compute_alpha_decay(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.Series:
    """
    Compute Alpha Decay (Half-Life) for each signal.
    Dynamic max_h: uses 2× the largest requested horizon to ensure coverage.
    Returns the bar at which the IC decays to 50% of its peak.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    max_h = max(horizons) * 2
    # Cap at a reasonable maximum to avoid extremely slow computation
    max_h = min(max_h, 120)

    h_range = list(range(1, max_h + 1))
    fwd = compute_forward_returns(df["close"], h_range)
    ic_df = compute_ic_table(signals, fwd)

    half_lives = {}
    for sig in ic_df.index:
        curve = ic_df.loc[sig].abs()
        peak_ic = curve.max()
        if pd.isna(peak_ic) or peak_ic < 0.02:
            half_lives[sig] = np.nan
            continue

        peak_h = int(curve.idxmax().replace("fwd_", ""))
        threshold = peak_ic * 0.5

        hl = np.nan
        for h in range(peak_h, max_h + 1):
            if curve[f"fwd_{h}"] < threshold:
                hl = h
                break

        if pd.isna(hl):
            hl = max_h

        half_lives[sig] = hl

    return pd.Series(half_lives, name="half_life")


def compute_autocorrelation(signals: pd.DataFrame) -> pd.Series:
    """
    Compute AR(1) autocorrelation for each signal to measure turnover friction.
    Signals with AR1 near 0 or negative flip very frequently (high turnover)
    and will bleed in transaction costs.

    Returns Series: index=signal names, values=Spearman AR1.
    """
    ar1 = {}
    for col in signals.columns:
        s = signals[col]
        df = pd.concat([s, s.shift(1)], axis=1).dropna()
        if len(df) < 30:
            ar1[col] = np.nan
        else:
            rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
            ar1[col] = round(float(rho), 3)
    return pd.Series(ar1, name="ar1")


# -- Quantile spread ------------------------------------------------------------


def quantile_spread(
    signal: pd.Series, fwd: pd.Series, n_bins: int = N_BINS
) -> pd.DataFrame:
    """
    Sort signal into n_bins equal-frequency buckets. Report mean forward return
    per bucket. A monotonically increasing pattern confirms linear predictive edge.
    Includes a monotonicity score (Spearman rank corr of bin index vs mean return).
    """
    df = pd.concat([signal.rename("signal"), fwd.rename("fwd")], axis=1).dropna()
    try:
        df["bin"] = pd.qcut(df["signal"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    result = (
        df.groupby("bin", observed=True)["fwd"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_fwd_return", "count": "n_obs"})
    )
    result["mean_fwd_return"] = result["mean_fwd_return"].round(6)
    result.index = [f"Q{i+1}" for i in result.index]
    return result


def compute_monotonicity(signal: pd.Series, fwd: pd.Series, n_bins: int = N_BINS) -> float:
    """
    Compute monotonicity score: Spearman rank correlation of bin index vs mean
    forward return across quantile bins. +1.0 = perfectly monotonic positive,
    -1.0 = perfectly monotonic negative, 0 = no monotonic relationship.
    """
    qs = quantile_spread(signal, fwd, n_bins)
    if qs.empty or len(qs) < 3:
        return np.nan
    bin_idx = np.arange(len(qs))
    rho, _ = stats.spearmanr(bin_idx, qs["mean_fwd_return"].values)
    return round(float(rho), 3)


def compute_recency_weighted_ic(
    signals: pd.DataFrame, fwd_returns: pd.DataFrame,
    halflife_bars: int = 500,
) -> pd.DataFrame:
    """
    Compute exponentially-weighted IC giving more weight to recent bars.
    Uses an exponential decay kernel with the specified halflife.
    Returns DataFrame with same structure as compute_ic_table.
    """
    n = len(signals)
    decay = np.exp(-np.log(2) / halflife_bars * np.arange(n)[::-1])  # recent bars = higher weight

    ic = {}
    for sig_col in signals.columns:
        row = {}
        for fwd_col in fwd_returns.columns:
            df = pd.concat([signals[sig_col], fwd_returns[fwd_col]], axis=1).dropna()
            if len(df) < 30:
                row[fwd_col] = np.nan
            else:
                # Weighted Spearman: rank with recency weighting
                w = decay[-len(df):]
                sig_ranked = df.iloc[:, 0].rank()
                fwd_ranked = df.iloc[:, 1].rank()
                # Weighted Pearson of ranks approximates weighted Spearman
                wm_sig = np.average(sig_ranked, weights=w)
                wm_fwd = np.average(fwd_ranked, weights=w)
                cov = np.average((sig_ranked - wm_sig) * (fwd_ranked - wm_fwd), weights=w)
                std_sig = np.sqrt(np.average((sig_ranked - wm_sig) ** 2, weights=w))
                std_fwd = np.sqrt(np.average((fwd_ranked - wm_fwd) ** 2, weights=w))
                denom = std_sig * std_fwd
                rho = cov / denom if denom > 0 else np.nan
                row[fwd_col] = round(float(rho), 5)
        ic[sig_col] = row
    return pd.DataFrame(ic).T


# -- Display helpers ------------------------------------------------------------


def _bar(val: float, width: int = 20) -> str:
    """ASCII bar chart for a value in [-1, 1]."""
    mid = width // 2
    filled = int(abs(val) * mid)
    filled = min(filled, mid)
    if val >= 0:
        return " " * mid + "#" * filled + " " * (mid - filled)
    else:
        return " " * (mid - filled) + "#" * filled + " " * mid


def print_ic_table(ic_df: pd.DataFrame, horizons: list[int]) -> None:
    cols = [f"fwd_{h}" for h in horizons]
    header = f"{'Signal':<14}" + "".join(f"  h={h:>3}" for h in horizons)
    print(header)
    print("-" * len(header))
    for sig in ic_df.index:
        row = f"{sig:<14}"
        for col in cols:
            v = ic_df.loc[sig, col]
            if np.isnan(v):
                row += "     NaN"
            else:
                sign = "+" if v >= 0 else ""
                row += f"  {sign}{v:>6.4f}"
        print(row)


def print_quantile_spread(qs: pd.DataFrame, signal_name: str, horizon: int) -> None:
    if qs.empty:
        print(f"  (insufficient data for quantile spread on {signal_name})")
        return
    print(f"  {'Bin':<6}  {'MeanFwdReturn':>14}  {'N':>6}  Chart")
    print("  " + "-" * 55)
    for idx, row_data in qs.iterrows():
        bar = _bar(row_data["mean_fwd_return"] * 500)  # scale for visibility
        print(
            f"  {idx:<6}  {row_data['mean_fwd_return']:>+14.6f}  "
            f"{int(row_data['n_obs']):>6}  {bar}"
        )


# -- Main pipeline --------------------------------------------------------------


def run_ic(
    instrument: str,
    timeframe: str,
    horizons: list[int] | None = None,
    n_bins: int = N_BINS,
    include_mtf: bool = False,
) -> dict:
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    df = load_ohlcv(instrument, timeframe)
    close = df["close"]
    signals = compute_signals(close)

    if include_mtf:
        mtf_sig = compute_mtf_confluence(instrument, timeframe)
        if mtf_sig is not None:
            signals["mtf_confluence"] = mtf_sig.reindex(signals.index)

    fwd_returns = compute_forward_returns(close, horizons)
    mfe_targets, mae_targets = compute_mfe_mae_targets(df, horizons)

    # Align: drop rows where signals (exc. mtf which may have leading NaN) are ready
    base_cols = [c for c in signals.columns if c != "mtf_confluence"]
    valid_mask = (
        signals[base_cols].notna().all(axis=1) 
        & fwd_returns.notna().any(axis=1)
        & mfe_targets.notna().any(axis=1)
        & mae_targets.notna().any(axis=1)
    )
    signals = signals[valid_mask]
    fwd_returns = fwd_returns[valid_mask]
    mfe_targets = mfe_targets[valid_mask]
    mae_targets = mae_targets[valid_mask]

    logger.info(f"Analysis window: {len(signals)} bars after warmup drop")

    # IC tables (with p-values for statistical significance)
    ic_df, pval_df = compute_ic_table(signals, fwd_returns, return_pvalues=True)
    ic_mfe = compute_ic_table(signals, mfe_targets)
    ic_mae = compute_ic_table(signals, mae_targets)

    # Recency-weighted IC (halflife=500 bars)
    ic_recent = compute_recency_weighted_ic(signals, fwd_returns, halflife_bars=500)

    # Best horizon per signal (highest |IC|)
    best_horizons = {
        sig: ic_df.loc[sig].abs().idxmax()
        for sig in ic_df.index
    }

    # ICIR at each signal's best horizon (not just h=1)
    icir = compute_icir(signals, fwd_returns, horizons, best_horizons=best_horizons)

    # Alpha decay (half-life) -- dynamic max_h based on requested horizons
    alpha_decay = compute_alpha_decay(signals, df, horizons=horizons)

    # -- Print ------------------------------------------------------------------
    slug = f"{instrument}_{timeframe}"
    print("\n" + "=" * 72)
    print(f"  IC SIGNAL SCANNER -- {instrument} {timeframe}")
    print(f"  {len(signals)} bars  |  Horizons: {horizons}  |  Bins: {n_bins}")
    print("=" * 72)

    print("\n-- Spearman IC Table (rows=signal, cols=forward horizon) --------------")
    print_ic_table(ic_df, horizons)

    print("\n-- ICIR (rolling {}-bar, at best horizon per signal) -------------------".format(
        ROLLING_IC_WINDOW
    ))
    for sig in icir.index:
        v = icir[sig]
        bh = best_horizons.get(sig, f"fwd_{horizons[0]}").replace("fwd_", "h=")
        flag = "  *** STRONG" if abs(v) >= 0.5 else ("  * weak" if abs(v) >= 0.2 else "")
        print(f"  {sig:<14}  {bh:>5}  {v:>+7.3f}{flag}" if not np.isnan(v) else f"  {sig:<14}  {bh:>5}     NaN")

    print("\n-- Quantile Spread (deciles of signal vs mean forward return) ---------")
    for sig in signals.columns:
        best_h_col = best_horizons[sig]
        best_h = int(best_h_col.replace("fwd_", ""))
        ic_val = ic_df.loc[sig, best_h_col]
        print(f"\n  {sig}  (best h={best_h}, IC={ic_val:+.4f})")
        qs = quantile_spread(signals[sig], fwd_returns[best_h_col], n_bins)
        print_quantile_spread(qs, sig, best_h)

    # -- Signal summary ---------------------------------------------------------
    print("\n-- Signal Summary -------------------------------------------------------------------------------------")
    print(f"  {'Signal':<14}  {'BestH':>6}  {'BestIC':>8}  {'ICIR':>7}  {'p-val':>7}  {'MFE':>6}  {'MAE':>6}  {'HalfL':>5}  {'Mono':>5}  {'RcntIC':>7}  Verdict")
    print("  " + "-" * 117)
    for sig in ic_df.index:
        best_h_col = best_horizons[sig]
        best_h = int(best_h_col.replace("fwd_", ""))
        best_ic = ic_df.loc[sig, best_h_col]
        ir = icir[sig]
        abs_ic = abs(best_ic)

        mfe_val = ic_mfe.loc[sig, f"mfe_{best_h}"]
        mae_val = ic_mae.loc[sig, f"mae_{best_h}"]
        hl_val = alpha_decay.get(sig, np.nan)
        pv = pval_df.loc[sig, best_h_col]
        mono = compute_monotonicity(signals[sig], fwd_returns[best_h_col])
        rcnt = ic_recent.loc[sig, best_h_col] if best_h_col in ic_recent.columns else np.nan

        if abs_ic >= 0.05 and (not np.isnan(ir) and abs(ir) >= 0.5):
            verdict = "STRONG -- build strategy"
        elif abs_ic >= 0.05:
            verdict = "Moderate IC -- inconsistent"
        elif abs_ic >= 0.03:
            verdict = "Weak -- try regime conditioning"
        else:
            verdict = "Noise -- discard"

        ir_str = f"{ir:+.3f}" if not np.isnan(ir) else "  NaN"
        pv_str = f"{pv:.1e}" if not np.isnan(pv) else "    NaN"
        mfe_str = f"{mfe_val:+.3f}" if not np.isnan(mfe_val) else "   NaN"
        mae_str = f"{mae_val:+.3f}" if not np.isnan(mae_val) else "   NaN"
        hl_str = f"{int(hl_val)}" if not np.isnan(hl_val) else "  NaN"
        mono_str = f"{mono:+.2f}" if not np.isnan(mono) else "  NaN"
        rcnt_str = f"{rcnt:+.4f}" if not np.isnan(rcnt) else "    NaN"

        print(
            f"  {sig:<14}  {best_h:>6}  {best_ic:>+8.4f}  {ir_str:>7}  {pv_str:>7}  {mfe_str:>6}  {mae_str:>6}  {hl_str:>5}  {mono_str:>5}  {rcnt_str:>7}  {verdict}"
        )

    print("=" * 72)

    # -- Save report ------------------------------------------------------------
    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"ic_{slug.lower()}.csv"
    ic_df.to_csv(out_path)
    logger.info(f"IC table saved: {out_path}")

    return {"ic": ic_df, "icir": icir, "best_horizons": best_horizons}


# -- Entry point ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="IC Signal Scanner")
    parser.add_argument("--instrument", default="QQQ")
    parser.add_argument("--timeframe", default="D")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated list of forward horizons, e.g. 1,5,10,20",
    )
    parser.add_argument("--n_bins", type=int, default=N_BINS)
    parser.add_argument(
        "--include_mtf", action="store_true",
        help="Include MTF confluence signal (requires H1/H4/D/W parquets + config/mtf.toml)",
    )
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    run_ic(
        instrument=args.instrument,
        timeframe=args.timeframe,
        horizons=horizons,
        n_bins=args.n_bins,
        include_mtf=args.include_mtf,
    )


if __name__ == "__main__":
    main()
