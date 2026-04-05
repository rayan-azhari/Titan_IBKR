"""run_single_tf_backtest.py -- Single-TF MTF Confluence Backtest.

Replicates the multi-timeframe confluence strategy on a SINGLE H1 stream
by using scaled MA/RSI periods to approximate higher-TF behavior.

This eliminates the cross-timeframe alignment problem entirely:
- No ffill contamination (single data stream, single timestamp authority)
- No look-ahead from slower TFs (every bar has its own computed value)
- Proper causal signal: shift(1) on the single stream is unambiguous

Virtual TF mapping (EUR/USD, ~24 H1 bars/day, ~120 bars/week):
    H1: native periods (fast=10, slow=30, rsi=28)
    H4: 4x H1 periods (fast=40, slow=160, rsi=28)   [~4 H1 bars per H4]
    D:  24x H1 periods (fast=120, slow=480, rsi=240) [~24 H1 bars per D]
    W:  120x H1 periods (fast=1560, slow=3120, rsi=1200) [~120 H1 bars per W]

Usage:
    uv run python research/mtf/run_single_tf_backtest.py
    uv run python research/mtf/run_single_tf_backtest.py --pair EUR_USD --threshold 0.10
    uv run python research/mtf/run_single_tf_backtest.py --sweep  # threshold sweep

Directive: Backtesting & Validation.md
"""

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── H1 bars per higher TF ────────────────────────────────────────────────────
# EUR/USD: ~24 H1 bars/day, ~120/week. These are approximate multipliers.
# Using exact values from the config's native periods × the TF multiplier.

H1_PER_H4 = 4
H1_PER_D = 24
H1_PER_W = 120  # 5 days × 24 bars


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_h1(pair: str) -> pd.DataFrame:
    """Load H1 OHLCV parquet."""
    path = DATA_DIR / f"{pair}_H1.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index().dropna(subset=["close"])


# ── Signal Construction ──────────────────────────────────────────────────────


def _fast_wma(values: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average via numpy convolution -- ~100x faster than .apply(lambda)."""
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights = weights / weights.sum()
    # convolve in 'full' mode then trim to match input length
    result = np.convolve(values, weights[::-1], mode="full")[: len(values)]
    # First (period-1) values are incomplete -- set to NaN
    result[: period - 1] = np.nan
    return result


def _vectorized_position(sig: np.ndarray, threshold: float, exit_buffer: float) -> np.ndarray:
    """Vectorized position simulation using numpy arrays -- ~10x faster than pandas iloc."""
    n = len(sig)
    pos = np.zeros(n)
    current = 0.0
    for i in range(n):
        s = sig[i]
        if current == 0.0:
            if s >= threshold:
                current = 1.0
            elif s <= -threshold:
                current = -1.0
        elif current > 0:
            if s < -exit_buffer:
                current = 0.0
        elif current < 0:
            if s > exit_buffer:
                current = 0.0
        pos[i] = current
    return pos


def compute_virtual_tf_signal(
    close: pd.Series,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    ma_type: str = "WMA",
) -> pd.Series:
    """Compute MA-spread + RSI signal at a given scale on H1 data.

    Matches the live strategy signal formula:
        ma_signal = tanh(normalized_spread) * 0.5   range [-0.5, +0.5]
        rsi_signal = (RSI - 50) / 100               range [-0.5, +0.5]
        tf_signal = ma_signal + rsi_signal           range [-1.0, +1.0]
    """
    vals = close.values.astype(np.float64)

    # MA computation (all on numpy arrays for speed)
    if ma_type == "EMA":
        fast_ma = close.ewm(span=fast_period, adjust=False).mean().values
        slow_ma = close.ewm(span=slow_period, adjust=False).mean().values
    elif ma_type == "WMA":
        fast_ma = _fast_wma(vals, fast_period)
        slow_ma = _fast_wma(vals, slow_period)
    else:
        fast_ma = pd.Series(vals).rolling(fast_period).mean().values
        slow_ma = pd.Series(vals).rolling(slow_period).mean().values

    # Normalised MA spread
    slow_abs = np.maximum(np.abs(slow_ma), 1e-8)
    raw_spread = (fast_ma - slow_ma) / slow_abs
    # Rolling std for tanh normalisation
    norm_window = max(20, slow_period // 2)
    spread_std = pd.Series(raw_spread).rolling(norm_window).std().values
    spread_std = np.maximum(spread_std, 1e-8)
    ma_signal = np.tanh(raw_spread / spread_std) * 0.5

    # RSI via EWM (pandas is fast for this)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=rsi_period, adjust=False).mean().values
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=rsi_period, adjust=False).mean().values
    rs = gain / np.maximum(loss, 1e-8)
    rsi_signal = (100 - (100 / (1 + rs)) - 50) / 100

    result = ma_signal + rsi_signal
    return pd.Series(result, index=close.index)


def build_confluence(
    close: pd.Series,
    config: dict,
    ma_type: str = "WMA",
) -> pd.Series:
    """Build multi-scale confluence score on a single H1 stream.

    Each "virtual TF" uses MA/RSI periods scaled by the TF multiplier.
    The confluence score is the weighted sum across virtual TFs.
    """
    weights = config.get("weights", {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30})

    # Map TF label to H1 bar multiplier
    tf_multipliers = {"H1": 1, "H4": H1_PER_H4, "D": H1_PER_D, "W": H1_PER_W}

    signals = {}
    for tf, w in weights.items():
        tf_cfg = config.get(tf, {})
        native_fast = tf_cfg.get("fast_ma", 10)
        native_slow = tf_cfg.get("slow_ma", 30)
        native_rsi = tf_cfg.get("rsi_period", 14)

        mult = tf_multipliers.get(tf, 1)

        # Scale native periods to H1 bars
        h1_fast = native_fast * mult
        h1_slow = native_slow * mult
        h1_rsi = native_rsi * mult

        sig = compute_virtual_tf_signal(close, h1_fast, h1_slow, h1_rsi, ma_type)
        signals[tf] = sig * w

    # Weighted sum (weights already applied above)
    df = pd.DataFrame(signals)
    confluence = df.sum(axis=1)

    return confluence


# ── Backtest Engine ──────────────────────────────────────────────────────────


def backtest(
    close: pd.Series,
    confluence: pd.Series,
    threshold: float = 0.10,
    exit_buffer: float = 0.10,
    spread_bps: float = 2.5,
    slippage_bps: float = 1.0,
    is_ratio: float = 0.70,
    _pre_zscored: bool = False,
) -> dict:
    """Run IS/OOS backtest with the confluence signal.

    Signal: long when score >= threshold, short when score <= -threshold.
    Exit long when score < -exit_buffer, exit short when score > exit_buffer.
    Position shifted by 1 bar (trade on next bar's open).

    If _pre_zscored=True, skip z-score normalization (caller already did it).

    Returns dict with IS/OOS metrics and OOS return series.
    """
    # Align
    common = close.index.intersection(confluence.index)
    close = close.reindex(common)
    confluence = confluence.reindex(common)

    n = len(close)
    is_n = int(n * is_ratio)

    if _pre_zscored:
        score_z = confluence
    else:
        # Z-score normalise composite on IS only (no look-ahead)
        is_mean = confluence.iloc[:is_n].mean()
        is_std = confluence.iloc[:is_n].std()
        if is_std < 1e-8:
            is_std = 1.0
        score_z = (confluence - is_mean) / is_std

    # Position rule with exit buffer (matches live strategy)
    # shift(1): signal known at bar T, position taken at bar T+1
    sig = score_z.shift(1).fillna(0.0)

    pos_arr = _vectorized_position(sig.values, threshold, exit_buffer)
    position = pd.Series(pos_arr, index=sig.index)

    # Returns with transaction costs
    bar_rets = close.pct_change().fillna(0.0)
    transitions = (position != position.shift(1).fillna(0.0)).astype(float)
    cost = transitions * (spread_bps + slippage_bps) / 10_000
    strategy_rets = bar_rets * position - cost

    # Split
    is_rets = strategy_rets.iloc[:is_n]
    oos_rets = strategy_rets.iloc[is_n:]

    # Aggregate H1 to daily for Sharpe computation
    is_daily = is_rets.resample("D").sum()
    is_daily = is_daily[is_daily != 0.0]
    oos_daily = oos_rets.resample("D").sum()
    oos_daily = oos_daily[oos_daily != 0.0]

    def _metrics(daily: pd.Series, label: str) -> dict:
        if len(daily) < 20:
            return {"label": label, "sharpe": 0.0, "bars": len(daily)}
        ann_ret = daily.mean() * 252
        ann_vol = daily.std() * sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        equity = (1 + daily).cumprod()
        hwm = equity.cummax()
        dd = (equity - hwm) / hwm
        max_dd = dd.min()
        wins = (daily > 0).sum()
        total = (daily != 0).sum()
        wr = wins / total if total > 0 else 0.0
        trades = (position.resample("D").last().diff().abs() > 0).sum()
        return {
            "label": label,
            "sharpe": round(sharpe, 3),
            "annual_return_pct": round(ann_ret * 100, 2),
            "annual_vol_pct": round(ann_vol * 100, 2),
            "max_dd_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(wr * 100, 1),
            "trades": int(trades),
            "days": len(daily),
        }

    m_is = _metrics(is_daily, "IS")
    m_oos = _metrics(oos_daily, "OOS")
    parity = m_oos["sharpe"] / m_is["sharpe"] if abs(m_is["sharpe"]) > 0.01 else 0.0

    # Trade statistics
    n_long = (position == 1.0).sum()
    n_short = (position == -1.0).sum()
    n_flat = (position == 0.0).sum()
    total = len(position)

    return {
        "is": m_is,
        "oos": m_oos,
        "parity": round(parity, 3),
        "position_pct": {
            "long": round(n_long / total * 100, 1),
            "short": round(n_short / total * 100, 1),
            "flat": round(n_flat / total * 100, 1),
        },
        "oos_daily_returns": oos_daily,
        "threshold": threshold,
        "h1_bars": n,
        "is_bars": is_n,
        "oos_bars": n - is_n,
    }


# ── Threshold Sweep ──────────────────────────────────────────────────────────


def run_threshold_sweep(
    close: pd.Series,
    confluence: pd.Series,
    thresholds: list[float] | None = None,
    exit_buffer: float = 0.10,
    spread_bps: float = 2.5,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """Sweep confirmation thresholds and report IS/OOS metrics."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5]

    rows = []
    for th in thresholds:
        result = backtest(
            close,
            confluence,
            threshold=th,
            exit_buffer=exit_buffer,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        )
        rows.append(
            {
                "threshold": th,
                "is_sharpe": result["is"]["sharpe"],
                "oos_sharpe": result["oos"]["sharpe"],
                "parity": result["parity"],
                "oos_ret_pct": result["oos"]["annual_return_pct"],
                "oos_dd_pct": result["oos"]["max_dd_pct"],
                "oos_wr_pct": result["oos"]["win_rate_pct"],
                "oos_trades": result["oos"]["trades"],
                "long_pct": result["position_pct"]["long"],
                "short_pct": result["position_pct"]["short"],
            }
        )

    return pd.DataFrame(rows)


# ── Display ──────────────────────────────────────────────────────────────────


def print_config_table(config: dict) -> None:
    """Print the virtual TF -> H1 period mapping."""
    weights = config.get("weights", {})
    tf_mult = {"H1": 1, "H4": H1_PER_H4, "D": H1_PER_D, "W": H1_PER_W}

    print(
        f"\n  {'VTF':>4} | {'Weight':>6} | {'Native Fast':>11} {'Native Slow':>11} {'Native RSI':>10}"
        f" | {'H1 Fast':>8} {'H1 Slow':>8} {'H1 RSI':>7}"
    )
    print("  " + "-" * 85)
    for tf, w in weights.items():
        tc = config.get(tf, {})
        nf = tc.get("fast_ma", 10)
        ns = tc.get("slow_ma", 30)
        nr = tc.get("rsi_period", 14)
        m = tf_mult.get(tf, 1)
        print(
            f"  {tf:>4} | {w:>5.0%} | {nf:>11} {ns:>11} {nr:>10}"
            f" | {nf * m:>8} {ns * m:>8} {nr * m:>7}"
        )


def print_result(result: dict) -> None:
    """Print backtest results."""
    m_is = result["is"]
    m_oos = result["oos"]

    print(
        f"\n  {'':>4} | {'Sharpe':>7} | {'Ret%':>7} | {'Vol%':>7} | {'MaxDD%':>7} | {'WR%':>5} | {'Trades':>6} | {'Days':>5}"
    )
    print("  " + "-" * 70)
    print(
        f"  {'IS':>4} | {m_is['sharpe']:>+7.3f} | {m_is['annual_return_pct']:>+6.1f}% | {m_is['annual_vol_pct']:>6.1f}%"
        f" | {m_is['max_dd_pct']:>+6.1f}% | {m_is['win_rate_pct']:>4.0f}% | {m_is['trades']:>6} | {m_is['days']:>5}"
    )
    print(
        f"  {'OOS':>4} | {m_oos['sharpe']:>+7.3f} | {m_oos['annual_return_pct']:>+6.1f}% | {m_oos['annual_vol_pct']:>6.1f}%"
        f" | {m_oos['max_dd_pct']:>+6.1f}% | {m_oos['win_rate_pct']:>4.0f}% | {m_oos['trades']:>6} | {m_oos['days']:>5}"
    )

    parity = result["parity"]
    status = "PASS" if parity >= 0.5 else "FAIL"
    print(f"\n  OOS/IS parity: {parity:.3f} [{status}]")
    pp = result["position_pct"]
    print(f"  Position: Long {pp['long']:.0f}% | Short {pp['short']:.0f}% | Flat {pp['flat']:.0f}%")
    print(f"  H1 bars: {result['h1_bars']} (IS: {result['is_bars']}, OOS: {result['oos_bars']})")


# ── Fast Backtest (sweep-optimized) ──────────────────────────────────────────


def _fast_backtest(
    close_vals: np.ndarray,
    close_index: pd.DatetimeIndex,
    score_z_vals: np.ndarray,
    threshold: float,
    exit_buffer: float,
    spread_bps: float,
    slippage_bps: float,
    is_n: int,
) -> dict:
    """Minimal backtest for sweep: returns IS/OOS Sharpe, trades, basic metrics.

    All inputs are numpy arrays for maximum speed. No pandas resampling overhead.
    """
    n = len(close_vals)

    # Shift signal by 1 (trade on next bar)
    sig = np.empty(n)
    sig[0] = 0.0
    sig[1:] = score_z_vals[:-1]

    # Vectorized position
    pos = _vectorized_position(sig, threshold, exit_buffer)

    # Bar returns
    bar_rets = np.empty(n)
    bar_rets[0] = 0.0
    bar_rets[1:] = (close_vals[1:] - close_vals[:-1]) / close_vals[:-1]

    # Transaction costs
    transitions = np.empty(n)
    transitions[0] = abs(pos[0])
    transitions[1:] = np.abs(pos[1:] - pos[:-1])
    cost_per = (spread_bps + slippage_bps) / 10_000

    strategy_rets = bar_rets * pos - transitions * cost_per

    # Split IS/OOS
    is_rets = strategy_rets[:is_n]
    oos_rets = strategy_rets[is_n:]

    # Aggregate to daily using date grouping (faster than resample)
    oos_dates = close_index[is_n:]
    oos_day_keys = oos_dates.date
    # Use pandas for groupby (fast enough)
    oos_daily = pd.Series(oos_rets, index=oos_dates).groupby(oos_day_keys).sum()
    oos_daily = oos_daily[oos_daily != 0.0]

    is_dates = close_index[:is_n]
    is_day_keys = is_dates.date
    is_daily = pd.Series(is_rets, index=is_dates).groupby(is_day_keys).sum()
    is_daily = is_daily[is_daily != 0.0]

    def _sharpe(daily):
        if len(daily) < 20:
            return 0.0
        m = daily.mean()
        s = daily.std()
        return float(m / s * sqrt(252)) if s > 1e-9 else 0.0

    is_sharpe = _sharpe(is_daily)
    oos_sharpe = _sharpe(oos_daily)
    parity = oos_sharpe / is_sharpe if abs(is_sharpe) > 0.01 else 0.0

    # OOS metrics
    oos_len = len(oos_daily)
    if oos_len >= 20:
        ann_ret = float(oos_daily.mean() * 252)
        equity = (1 + oos_daily).cumprod()
        max_dd = float(((equity - equity.cummax()) / equity.cummax()).min())
        wr = float((oos_daily > 0).sum() / (oos_daily != 0).sum())
        trades = int(np.sum(np.abs(np.diff(pos[is_n:])) > 0))
    else:
        ann_ret = 0.0
        max_dd = 0.0
        wr = 0.0
        trades = 0

    return {
        "is_sharpe": round(is_sharpe, 3),
        "oos_sharpe": round(oos_sharpe, 3),
        "parity": round(parity, 3),
        "oos_ret_pct": round(ann_ret * 100, 2),
        "oos_dd_pct": round(max_dd * 100, 2),
        "oos_wr_pct": round(wr * 100, 1),
        "oos_trades": trades,
    }


# ── Full Parameter Sweep ─────────────────────────────────────────────────────

# Sweep grid: MA type × D fast/slow × W fast/slow × threshold
D_FAST_GRID = [3, 5, 8, 10, 15, 20]  # native D periods -> ×24 for H1
D_SLOW_GRID = [15, 20, 30, 40, 50, 60]
W_FAST_GRID = [8, 13, 20, 26]  # native W periods -> ×120 for H1
W_SLOW_GRID = [20, 26, 40, 52]
THRESHOLD_GRID = [0.10, 0.30, 0.50, 1.0]
MA_TYPES = ["SMA", "EMA", "WMA"]


def run_full_sweep(
    close: pd.Series,
    config: dict,
    spread_bps: float = 2.5,
    slippage_bps: float = 1.0,
    exit_buffer: float = 0.10,
) -> pd.DataFrame:
    """Full parameter sweep: MA type × D params × W params × threshold.

    H1 and H4 virtual TF parameters are fixed (15% combined weight).
    D (55%) and W (30%) are swept as they dominate the signal.
    """
    weights = config.get("weights", {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30})
    h1_cfg = config.get("H1", {"fast_ma": 10, "slow_ma": 30, "rsi_period": 28})
    h4_cfg = config.get("H4", {"fast_ma": 10, "slow_ma": 40, "rsi_period": 7})
    d_rsi = config.get("D", {}).get("rsi_period", 10)
    w_rsi = config.get("W", {}).get("rsi_period", 10)

    # IS/OOS setup
    n = len(close)
    is_n = int(n * 0.70)

    # Count total combos for progress
    d_combos = [(f, s) for f in D_FAST_GRID for s in D_SLOW_GRID if f < s]
    w_combos = [(f, s) for f in W_FAST_GRID for s in W_SLOW_GRID if f < s]
    total = len(MA_TYPES) * len(d_combos) * len(w_combos) * len(THRESHOLD_GRID)
    print(
        f"  Grid: {len(MA_TYPES)} MA × {len(d_combos)} D × {len(w_combos)} W × {len(THRESHOLD_GRID)} thresh = {total} combos"
    )

    # Pre-extract numpy arrays for speed
    close_vals = close.values.astype(np.float64)
    close_idx = close.index

    rows = []
    done = 0
    import time

    t0 = time.time()

    for ma_type in MA_TYPES:
        ma_t0 = time.time()
        # Pre-compute fixed H1 and H4 signals (don't change across D/W sweep)
        h1_sig = compute_virtual_tf_signal(
            close, h1_cfg["fast_ma"], h1_cfg["slow_ma"], h1_cfg["rsi_period"], ma_type
        ).values * weights.get("H1", 0.10)

        h4_sig = compute_virtual_tf_signal(
            close,
            h4_cfg["fast_ma"] * H1_PER_H4,
            h4_cfg["slow_ma"] * H1_PER_H4,
            h4_cfg["rsi_period"] * H1_PER_H4,
            ma_type,
        ).values * weights.get("H4", 0.05)

        h1_h4_base = np.nan_to_num(h1_sig, 0.0) + np.nan_to_num(h4_sig, 0.0)

        for d_fast, d_slow in d_combos:
            # Compute D virtual signal
            d_sig = compute_virtual_tf_signal(
                close, d_fast * H1_PER_D, d_slow * H1_PER_D, d_rsi * H1_PER_D, ma_type
            ).values * weights.get("D", 0.55)

            for w_fast, w_slow in w_combos:
                # Compute W virtual signal
                w_sig = compute_virtual_tf_signal(
                    close, w_fast * H1_PER_W, w_slow * H1_PER_W, w_rsi * H1_PER_W, ma_type
                ).values * weights.get("W", 0.30)

                # Combine (all numpy)
                composite = h1_h4_base + np.nan_to_num(d_sig, 0.0) + np.nan_to_num(w_sig, 0.0)

                # Z-score on IS only
                is_vals = composite[:is_n]
                is_mean = np.nanmean(is_vals)
                is_std = np.nanstd(is_vals)
                if is_std < 1e-8:
                    is_std = 1.0
                composite_z = (composite - is_mean) / is_std

                for th in THRESHOLD_GRID:
                    result = _fast_backtest(
                        close_vals,
                        close_idx,
                        composite_z,
                        th,
                        exit_buffer,
                        spread_bps,
                        slippage_bps,
                        is_n,
                    )
                    rows.append(
                        {
                            "ma_type": ma_type,
                            "d_fast": d_fast,
                            "d_slow": d_slow,
                            "w_fast": w_fast,
                            "w_slow": w_slow,
                            "threshold": th,
                            **result,
                        }
                    )

                    done += 1

            if done % 200 == 0 and done > 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"    {done}/{total} ({done / total * 100:.0f}%) "
                    f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s"
                )

        print(f"  {ma_type} done in {time.time() - ma_t0:.1f}s")

    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import tomllib

    parser = argparse.ArgumentParser(description="Single-TF MTF Confluence Backtest")
    parser.add_argument("--pair", default="EUR_USD", help="FX pair")
    parser.add_argument("--threshold", type=float, default=None, help="Confirmation threshold")
    parser.add_argument("--exit-buffer", type=float, default=0.10, help="Exit buffer")
    parser.add_argument("--spread-bps", type=float, default=2.5, help="Spread in bps")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in bps")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument(
        "--full-sweep", action="store_true", help="Full param sweep: MA type × D × W × threshold"
    )
    parser.add_argument("--ma-type", default=None, help="MA type override (SMA/EMA/WMA)")
    args = parser.parse_args()

    # Load config
    pair_lower = args.pair.lower().replace("/", "_")
    config_path = PROJECT_ROOT / "config" / f"mtf_{pair_lower.replace('_', '')}.toml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    else:
        print(f"WARNING: Config not found at {config_path}, using defaults.")
        config = {
            "ma_type": "WMA",
            "confirmation_threshold": 0.10,
            "exit_buffer": 0.10,
            "weights": {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30},
            "H1": {"fast_ma": 10, "slow_ma": 30, "rsi_period": 28},
            "H4": {"fast_ma": 10, "slow_ma": 40, "rsi_period": 7},
            "D": {"fast_ma": 5, "slow_ma": 20, "rsi_period": 10},
            "W": {"fast_ma": 13, "slow_ma": 26, "rsi_period": 10},
        }

    ma_type = args.ma_type or config.get("ma_type", "WMA")
    threshold = args.threshold or config.get("confirmation_threshold", 0.10)
    exit_buffer = args.exit_buffer or config.get("exit_buffer", 0.10)

    print("=" * 70)
    print(f"  SINGLE-TF MTF CONFLUENCE -- {args.pair} H1")
    print(f"  MA Type: {ma_type} | Threshold: {threshold} | Exit Buffer: {exit_buffer}")
    print(f"  Costs: {args.spread_bps} bps spread + {args.slippage_bps} bps slippage")
    print("=" * 70)

    # Load data
    df = load_h1(args.pair)
    close = df["close"]
    print(f"\n  Loaded {len(close)} H1 bars ({close.index[0].date()} -> {close.index[-1].date()})")

    # Show virtual TF mapping
    print_config_table(config)

    # Full parameter sweep mode
    if args.full_sweep:
        print("\n  Running FULL PARAMETER SWEEP...")
        print(f"  MA types: {MA_TYPES}")
        print(f"  D fast: {D_FAST_GRID} (native) | D slow: {D_SLOW_GRID}")
        print(f"  W fast: {W_FAST_GRID} (native) | W slow: {W_SLOW_GRID}")
        print(f"  Thresholds: {THRESHOLD_GRID}")

        sweep_df = run_full_sweep(
            close,
            config,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            exit_buffer=exit_buffer,
        )

        # Quality gates
        passed = sweep_df[
            (sweep_df["oos_sharpe"] > 0.0)
            & (sweep_df["parity"] >= 0.5)
            & (sweep_df["oos_trades"] >= 50)
        ]

        print(f"\n  RESULTS: {len(sweep_df)} combos tested")
        print(f"  Passed quality gates (OOS Sharpe>0, parity>=0.5, trades>=50): {len(passed)}")

        # Top 20
        top = sweep_df.head(20)
        print("\n  Top 20 by OOS Sharpe:")
        print(
            f"  {'MA':>4} {'D_f':>4} {'D_s':>4} {'W_f':>4} {'W_s':>4} {'Thr':>5}"
            f" | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
            f" | {'Ret%':>6} {'DD%':>6} {'WR%':>4} {'Trd':>4}"
        )
        print("  " + "-" * 78)
        for _, r in top.iterrows():
            flag = (
                "+" if r["oos_sharpe"] > 0 and r["parity"] >= 0.5 and r["oos_trades"] >= 50 else " "
            )
            print(
                f" {flag}{r['ma_type']:>3} {int(r['d_fast']):>4} {int(r['d_slow']):>4}"
                f" {int(r['w_fast']):>4} {int(r['w_slow']):>4} {r['threshold']:>5.2f}"
                f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
                f" | {r['oos_ret_pct']:>+5.1f}% {r['oos_dd_pct']:>+5.1f}% {r['oos_wr_pct']:>3.0f}% {int(r['oos_trades']):>4}"
            )

        # Save
        sweep_path = REPORT_DIR / f"mtf_single_tf_full_sweep_{pair_lower}.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\n  Full sweep saved to: {sweep_path}")

        if len(passed) > 0:
            best = passed.iloc[0]
            print("\n  BEST PASSING COMBO:")
            print(
                f"    MA={best['ma_type']} D=({int(best['d_fast'])},{int(best['d_slow'])})"
                f" W=({int(best['w_fast'])},{int(best['w_slow'])}) Threshold={best['threshold']}"
            )
            print(f"    OOS Sharpe: {best['oos_sharpe']:+.3f} | Parity: {best['parity']:.3f}")
        else:
            print("\n  NO COMBOS PASSED ALL QUALITY GATES.")
            print("  Conclusion: MTF confluence does not produce a tradeable edge on EUR/USD H1.")

        return

    # Build confluence (single config)
    print("\n  Computing confluence signal...")
    confluence = build_confluence(close, config, ma_type=ma_type)
    print(f"  Signal computed: {len(confluence)} bars (after warmup)")

    if args.sweep:
        print("\n  Running threshold sweep...")
        sweep_df = run_threshold_sweep(
            close,
            confluence,
            exit_buffer=exit_buffer,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        )
        print(
            f"\n  {'Thresh':>6} | {'IS Sh':>6} {'OOS Sh':>7} {'Parity':>7}"
            f" | {'OOS Ret%':>8} {'OOS DD%':>8} {'WR%':>5} {'Trades':>6}"
            f" | {'Long%':>5} {'Short%':>6}"
        )
        print("  " + "-" * 82)
        for _, r in sweep_df.iterrows():
            status = "+" if r["oos_sharpe"] > 0 and r["parity"] >= 0.5 else " "
            print(
                f" {status}{r['threshold']:>5.2f} | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>7.3f}"
                f" | {r['oos_ret_pct']:>+7.1f}% {r['oos_dd_pct']:>+7.1f}% {r['oos_wr_pct']:>4.0f}% {r['oos_trades']:>6}"
                f" | {r['long_pct']:>4.0f}% {r['short_pct']:>5.0f}%"
            )

        # Save sweep
        sweep_path = REPORT_DIR / f"mtf_single_tf_sweep_{pair_lower}.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\n  Sweep saved to: {sweep_path}")

        # Find best OOS
        best = sweep_df.loc[sweep_df["oos_sharpe"].idxmax()]
        print(f"\n  Best OOS threshold: {best['threshold']:.2f} (Sharpe {best['oos_sharpe']:+.3f})")

        # Run detailed backtest at best threshold
        result = backtest(
            close,
            confluence,
            threshold=best["threshold"],
            exit_buffer=exit_buffer,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        )
    else:
        result = backtest(
            close,
            confluence,
            threshold=threshold,
            exit_buffer=exit_buffer,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        )

    print_result(result)

    # Save OOS returns
    oos_path = REPORT_DIR / f"mtf_single_tf_{pair_lower}_oos_daily.parquet"
    result["oos_daily_returns"].to_frame().to_parquet(oos_path)
    print(f"\n  OOS daily returns saved to: {oos_path}")


if __name__ == "__main__":
    main()
