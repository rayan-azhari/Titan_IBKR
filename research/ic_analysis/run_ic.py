"""
run_ic.py -- Information Coefficient (IC) Core Math Library

Pre-backtest signal validation library.
NOTE (C6/R1 Fix): This module has been deprecated as a standalone script.
The original 6-signal runner has been removed. Use `phase1_sweep.py`
as the canonical entry point for reading and evaluating 52 signals.

This file now serves strictly as a shared library for IC calculations.
"""

import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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
    MTF_CONFIG_PATH = ROOT / "config" / "mtf.toml"  # Moved inside function
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
        ma_score = (fast_ma > slow_ma).astype(float) - 0.5  # +0.5 or -0.5
        rsi_score = (_rsi(close_tf, rsi_p) > 50).astype(float) - 0.5  # +0.5 or -0.5
        tf_contribution = (ma_score + rsi_score) * w  # range: [-w, +w]

        # PREVENT LOOKAHEAD BIAS: shift(1) if native TF is higher than primary TF
        if tf != primary_tf:
            tf_contribution = tf_contribution.shift(1)

        # Align to primary TF index using forward fill (causal)
        tf_contribution = tf_contribution.reindex(primary_close.index, method="ffill")
        contributions.append(tf_contribution)

    if not contributions:
        logger.warning("No MTF timeframes could be loaded")
        return None

    confluence = pd.concat(contributions, axis=1).sum(axis=1).rename("mtf_confluence")
    logger.info(
        f"MTF confluence computed: {len(confluence)} bars, "
        f"range [{confluence.min():.3f}, {confluence.max():.3f}]"
    )
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
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
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


def apply_bh_fdr(
    pvalue_df: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G1 FIX: Benjamini-Hochberg False Discovery Rate correction.

    Takes the raw p-value DataFrame (signals × horizons) from compute_ic_table,
    flattens all p-values, applies BH adjustment, and reshapes back.

    Returns:
        (adjusted_pvalues_df, reject_mask_df)
    """
    from statsmodels.stats.multitest import multipletests

    flat_pvals = pvalue_df.values.flatten()
    # Replace NaN with 1.0 for BH (they won't reject).
    nan_mask = np.isnan(flat_pvals)
    flat_clean = np.where(nan_mask, 1.0, flat_pvals)

    reject, adj_pvals, _, _ = multipletests(flat_clean, alpha=alpha, method="fdr_bh")

    # Restore NaN positions.
    adj_pvals[nan_mask] = np.nan
    reject = reject.astype(float)
    reject[nan_mask] = np.nan

    adj_df = pd.DataFrame(
        adj_pvals.reshape(pvalue_df.shape),
        index=pvalue_df.index,
        columns=pvalue_df.columns,
    )
    reject_df = pd.DataFrame(
        reject.reshape(pvalue_df.shape),
        index=pvalue_df.index,
        columns=pvalue_df.columns,
    ).astype(bool)

    return adj_df, reject_df


@njit
def _tied_rank(x: np.ndarray) -> np.ndarray:
    n = len(x)
    ranks = np.empty(n)
    idx = np.argsort(x)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[idx[j]] == x[idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks

@njit
def _rolling_spearman_numba(sig_vals: np.ndarray, fwd_vals: np.ndarray, window: int) -> np.ndarray:
    """Numba-compiled exact rolling Spearman calculation."""
    n = len(sig_vals)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        ws = sig_vals[i - window : i]
        wf = fwd_vals[i - window : i]
        
        mean_ws, mean_wf = 0.0, 0.0
        for k in range(window):
            mean_ws += ws[k]
            mean_wf += wf[k]
        mean_ws /= window
        mean_wf /= window
        
        var_s, var_f = 0.0, 0.0
        for k in range(window):
            var_s += (ws[k] - mean_ws)**2
            var_f += (wf[k] - mean_wf)**2
            
        if var_s < 1e-14 or var_f < 1e-14:
            continue
            
        rank_s = _tied_rank(ws)
        rank_f = _tied_rank(wf)
        
        mean_rs, mean_rf = 0.0, 0.0
        for k in range(window):
            mean_rs += rank_s[k]
            mean_rf += rank_f[k]
        mean_rs /= window
        mean_rf /= window
        
        cov, var_rs, var_rf = 0.0, 0.0, 0.0
        for k in range(window):
            diff_s = rank_s[k] - mean_rs
            diff_f = rank_f[k] - mean_rf
            cov += diff_s * diff_f
            var_rs += diff_s * diff_s
            var_rf += diff_f * diff_f
            
        if var_rs < 1e-14 or var_rf < 1e-14:
            continue
            
        out[i-1] = cov / np.sqrt(var_rs * var_rf)
        
    return out

def _rolling_ic_series(
    sig: pd.Series,
    fwd: pd.Series,
    window: int,
    exact: bool = True,
) -> np.ndarray:
    """Rolling Spearman IC.

    Default (exact=True): re-ranks within every window — correct causal Spearman.
    O(n*window) but honest; use exact=False only for quick exploratory runs.

    exact=False uses rolling Pearson on globally pre-ranked series (O(n) Cython).
    WARNING: global ranking is not causal — ICIR values are upward-biased 5-15%
    because each bar's rank depends on values it has not yet seen.
    """
    df = pd.concat([sig, fwd], axis=1).dropna()
    if len(df) < window:
        return np.array([])

    if exact:
        sig_vals = df.iloc[:, 0].values
        fwd_vals = df.iloc[:, 1].values
        arr = _rolling_spearman_numba(sig_vals, fwd_vals, window)
        return arr[~np.isnan(arr)]

    # Fast path: rank globally, then use pandas rolling Pearson (Cython)
    sig_r = df.iloc[:, 0].rank()
    fwd_r = df.iloc[:, 1].rank()
    roll_ic = sig_r.rolling(window).corr(fwd_r)
    return roll_ic.dropna().values


def _resolve_fwd_col(
    sig_col: str,
    horizons: list[int],
    fwd_returns: pd.DataFrame,
    best_horizons: dict[str, str] | None,
) -> str:
    """Return the forward-return column to use for a given signal."""
    if best_horizons and sig_col in best_horizons:
        fwd_col = best_horizons[sig_col]
    else:
        fwd_col = f"fwd_{horizons[0]}"
    if fwd_col not in fwd_returns.columns:
        fwd_col = f"fwd_{horizons[0]}"
    return fwd_col


def _compute_one_rolling_ic(
    sig_col: str,
    sig_vals: np.ndarray,
    fwd_vals: np.ndarray,
    window: int,
) -> tuple[str, np.ndarray]:
    """Worker: compute rolling IC for one signal. Safe to call from subprocesses."""
    arr = _rolling_spearman_numba(sig_vals, fwd_vals, window)
    return sig_col, arr[~np.isnan(arr)]


def build_rolling_ic_map(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons: list[int],
    window: int,
    best_horizons: dict[str, str] | None = None,
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    """Compute rolling IC for every signal once, in parallel.

    Returns a dict {sig_col: rolling_ic_array}. Pass this to compute_icir()
    and compute_icir_nw() to avoid computing it twice.

    n_jobs=-1 uses all available CPU cores.
    """
    from joblib import Parallel, delayed

    tasks = []
    for sig_col in signals.columns:
        fwd_col = _resolve_fwd_col(sig_col, horizons, fwd_returns, best_horizons)
        df = pd.concat([signals[sig_col], fwd_returns[fwd_col]], axis=1).dropna()
        if len(df) < window:
            tasks.append((sig_col, np.array([]), np.array([]), window))
        else:
            tasks.append((sig_col, df.iloc[:, 0].values, df.iloc[:, 1].values, window))

    # Warm up Numba JIT on the first signal before launching subprocesses.
    if tasks:
        first = tasks[0]
        if len(first[1]) >= window:
            _rolling_spearman_numba(first[1][:window], first[2][:window], window)

    results: list[tuple[str, np.ndarray]] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_one_rolling_ic)(sig_col, sv, fv, w)
        for sig_col, sv, fv, w in tasks
    )
    return dict(results)


def compute_icir(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons: list[int],
    window: int = 252,
    best_horizons: dict[str, str] | None = None,
    rolling_ic_map: dict[str, np.ndarray] | None = None,
) -> pd.Series:
    """Rolling ICIR = mean(rolling IC) / std(rolling IC).

    If rolling_ic_map is provided (from build_rolling_ic_map), the expensive
    rolling Spearman step is skipped — reuse the pre-computed arrays.
    """
    icirs = {}
    for sig_col in signals.columns:
        if rolling_ic_map is not None:
            roll_ic = rolling_ic_map.get(sig_col, np.array([]))
        else:
            fwd_col = _resolve_fwd_col(sig_col, horizons, fwd_returns, best_horizons)
            roll_ic = _rolling_ic_series(signals[sig_col], fwd_returns[fwd_col], window)

        if len(roll_ic) < 10:
            icirs[sig_col] = np.nan
        else:
            icirs[sig_col] = (
                float(roll_ic.mean() / roll_ic.std()) if roll_ic.std() > 1e-12 else np.nan
            )

    return pd.Series(icirs, name="icir")


def compute_icir_nw(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons: list[int],
    window: int = 252,
    best_horizons: dict[str, str] | None = None,
    rolling_ic_map: dict[str, np.ndarray] | None = None,
) -> pd.Series:
    """ICIR with Newey-West standard error adjustment (h-1 lags).

    If rolling_ic_map is provided (from build_rolling_ic_map), the expensive
    rolling Spearman step is skipped — reuse the pre-computed arrays.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.Series(dtype=float, name="icir_nw")

    icirs_nw: dict[str, float] = {}
    for sig_col in signals.columns:
        fwd_col = _resolve_fwd_col(sig_col, horizons, fwd_returns, best_horizons)

        try:
            h = int(fwd_col.replace("fwd_", ""))
        except ValueError:
            h = 1
        nw_lags = max(h - 1, 0)

        if rolling_ic_map is not None:
            roll_ic_clean = rolling_ic_map.get(sig_col, np.array([]))
        else:
            roll_ic_clean = _rolling_ic_series(
                signals[sig_col], fwd_returns[fwd_col], window
            )

        if len(roll_ic_clean) < 30:
            icirs_nw[sig_col] = np.nan
            continue

        try:
            y = roll_ic_clean
            # Regress IC series on a constant; HAC SE of the intercept equals the
            # Newey-West SE of the mean.  ICIR_NW = mean(IC) / SE_NW = tvalues[0].
            X = np.ones((len(y), 1))
            ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(nw_lags, 1)})
            icirs_nw[sig_col] = float(ols.tvalues[0])
        except Exception:
            icirs_nw[sig_col] = np.nan

    return pd.Series(icirs_nw, name="icir_nw")


def compute_alpha_decay(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.Series:
    """Compute Alpha Decay (Half-Life) for each signal.

    Dynamic max_h: uses 2x the largest requested horizon to ensure coverage.
    Returns the bar at which the IC decays to 50% of its peak.
    """
    DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    max_h = max(horizons) * 2
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
    signal: pd.Series,
    fwd: pd.Series,
    n_bins: int = 10,  # N_BINS moved here
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
    result.index = [f"Q{i + 1}" for i in result.index]
    return result


def compute_monotonicity(
    signal: pd.Series, fwd: pd.Series, n_bins: int = 10
) -> float:  # N_BINS moved here
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
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
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
                w = decay[-len(df) :]
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
