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


def load_close(instrument: str, timeframe: str) -> pd.Series:
    path = ROOT / "data" / f"{instrument}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            raise ValueError(f"Cannot resolve index for {path}")
    col_lower = {c.lower(): c for c in df.columns}
    close_col = col_lower.get("close")
    if close_col is None:
        raise ValueError(f"No 'close' column in {path}")
    close = df[close_col].dropna().sort_index()
    logger.info(
        f"Loaded {len(close)} bars | {instrument} {timeframe} "
        f"({close.index[0].date()} - {close.index[-1].date()})"
    )
    return close


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
    primary_close = load_close(instrument, primary_tf)

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
            close_tf = load_close(instrument, tf)
        except Exception:
            continue

        # Compute MA crossover and RSI on native TF bars
        fast_ma = close_tf.ewm(span=fast, adjust=False).mean()
        slow_ma = close_tf.ewm(span=slow, adjust=False).mean()
        ma_score = (fast_ma > slow_ma).astype(float) - 0.5   # +0.5 or -0.5
        rsi_score = (_rsi(close_tf, rsi_p) > 50).astype(float) - 0.5  # +0.5 or -0.5
        tf_contribution = (ma_score + rsi_score) * w  # range: [-w, +w]

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


def compute_forward_returns(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """
    Compute log forward returns at each horizon.
    fwd_h[t] = log(close[t+h] / close[t])

    close.shift(-h) is intentional -- this is the target (future price), not a
    feature. The signal at bar t is correlated against the return from t to t+h.
    """
    log_close = np.log(close)
    fwd = {
        f"fwd_{h}": log_close.shift(-h) - log_close
        for h in horizons
    }
    return pd.DataFrame(fwd, index=close.index)


# -- IC computation -------------------------------------------------------------


def compute_ic_table(
    signals: pd.DataFrame, fwd_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Spearman IC for every signal × horizon pair.
    Drops NaN pairs before correlation -- no imputation.

    Returns DataFrame: index=signal names, columns=horizon labels, values=Spearman ρ
    """
    ic = {}
    for sig_col in signals.columns:
        row = {}
        for fwd_col in fwd_returns.columns:
            df = pd.concat([signals[sig_col], fwd_returns[fwd_col]], axis=1).dropna()
            if len(df) < 30:
                row[fwd_col] = np.nan
            else:
                rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
                row[fwd_col] = round(float(rho), 5)
        ic[sig_col] = row
    return pd.DataFrame(ic).T


def compute_icir(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizons: list[int],
    window: int = ROLLING_IC_WINDOW,
) -> pd.DataFrame:
    """
    Rolling ICIR = mean(rolling IC) / std(rolling IC).
    Computed at the first horizon only (simplification).
    Returns Series: index=signal names, values=ICIR.
    """
    fwd_col = f"fwd_{horizons[0]}"
    fwd = fwd_returns[fwd_col]
    icirs = {}
    for sig_col in signals.columns:
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


# -- Quantile spread ------------------------------------------------------------


def quantile_spread(
    signal: pd.Series, fwd: pd.Series, n_bins: int = N_BINS
) -> pd.DataFrame:
    """
    Sort signal into n_bins equal-frequency buckets. Report mean forward return
    per bucket. A monotonically increasing pattern confirms linear predictive edge.
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

    close = load_close(instrument, timeframe)
    signals = compute_signals(close)

    if include_mtf:
        mtf_sig = compute_mtf_confluence(instrument, timeframe)
        if mtf_sig is not None:
            signals["mtf_confluence"] = mtf_sig.reindex(signals.index)

    fwd_returns = compute_forward_returns(close, horizons)

    # Align: drop rows where signals (exc. mtf which may have leading NaN) are ready
    base_cols = [c for c in signals.columns if c != "mtf_confluence"]
    valid_mask = signals[base_cols].notna().all(axis=1) & fwd_returns.notna().any(axis=1)
    signals = signals[valid_mask]
    fwd_returns = fwd_returns[valid_mask]

    logger.info(f"Analysis window: {len(signals)} bars after warmup drop")

    # IC table
    ic_df = compute_ic_table(signals, fwd_returns)

    # ICIR (at shortest horizon)
    icir = compute_icir(signals, fwd_returns, horizons)

    # Best horizon per signal (highest |IC|)
    best_horizons = {
        sig: ic_df.loc[sig].abs().idxmax()
        for sig in ic_df.index
    }

    # -- Print ------------------------------------------------------------------
    slug = f"{instrument}_{timeframe}"
    print("\n" + "=" * 72)
    print(f"  IC SIGNAL SCANNER -- {instrument} {timeframe}")
    print(f"  {len(signals)} bars  |  Horizons: {horizons}  |  Bins: {n_bins}")
    print("=" * 72)

    print("\n-- Spearman IC Table (rows=signal, cols=forward horizon) --------------")
    print_ic_table(ic_df, horizons)

    print("\n-- ICIR (rolling {}-bar, at h={}) -----------------------------------".format(
        ROLLING_IC_WINDOW, horizons[0]
    ))
    for sig in icir.index:
        v = icir[sig]
        flag = "  *** STRONG" if abs(v) >= 0.5 else ("  * weak" if abs(v) >= 0.2 else "")
        print(f"  {sig:<14}  {v:>+7.3f}{flag}" if not np.isnan(v) else f"  {sig:<14}     NaN")

    print("\n-- Quantile Spread (deciles of signal vs mean forward return) ---------")
    for sig in signals.columns:
        best_h_col = best_horizons[sig]
        best_h = int(best_h_col.replace("fwd_", ""))
        ic_val = ic_df.loc[sig, best_h_col]
        print(f"\n  {sig}  (best h={best_h}, IC={ic_val:+.4f})")
        qs = quantile_spread(signals[sig], fwd_returns[best_h_col], n_bins)
        print_quantile_spread(qs, sig, best_h)

    # -- Signal summary ---------------------------------------------------------
    print("\n-- Signal Summary ----------------------------------------------------")
    print(f"  {'Signal':<14}  {'BestH':>6}  {'BestIC':>8}  {'ICIR':>7}  Verdict")
    print("  " + "-" * 58)
    for sig in ic_df.index:
        best_h_col = best_horizons[sig]
        best_h = int(best_h_col.replace("fwd_", ""))
        best_ic = ic_df.loc[sig, best_h_col]
        ir = icir[sig]
        abs_ic = abs(best_ic)
        if abs_ic >= 0.05 and (not np.isnan(ir) and abs(ir) >= 0.5):
            verdict = "STRONG -- build strategy"
        elif abs_ic >= 0.05:
            verdict = "Moderate IC -- inconsistent"
        elif abs_ic >= 0.03:
            verdict = "Weak -- try regime conditioning"
        else:
            verdict = "Noise -- discard"
        ir_str = f"{ir:+.3f}" if not np.isnan(ir) else "  NaN"
        print(
            f"  {sig:<14}  {best_h:>6}  {best_ic:>+8.4f}  {ir_str:>7}  {verdict}"
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
