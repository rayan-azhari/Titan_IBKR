"""run_stage2_decel.py -- Stage 2: Deceleration Signal Selection.

Tests each deceleration signal independently against the Stage 1 baseline.
Selects signals that add >= 2% OOS total return vs the pure slow-MA baseline.

Deceleration signals serve two roles:
  1. Entry confirmation -- only enter if composite decel >= 0
  2. Position scalar (in dynamic sizing) -- position size proportional to decel strength

Signals tested:
  - d_pct:     (close - slow_ma) / slow_ma, EWM-smoothed -- price momentum
  - rv_20:     20-day realised vol (annualised) -- rising vol = warning
  - adx_14:    Average Directional Index -- falling ADX = trend weakening
  - macd_hist: MACD histogram -- shrinking histogram = deceleration

Baseline (Stage 1): entry = close > slow_ma only (no fast MA, no decel filter).

Auto-loads Stage 1 results via --load-state.

Usage:
    uv run python research/etf_trend/run_stage2_decel.py --instrument SPY --load-state
    uv run python research/etf_trend/run_stage2_decel.py --instrument TQQQ --signal-instrument QQQ --load-state
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from research.etf_trend.state_manager import get_stage1, save_stage2  # noqa: E402
from titan.strategies.ml.features import adx as compute_adx  # noqa: E402
from titan.strategies.ml.features import macd_hist as compute_macd_hist  # noqa: E402

FEES = 0.001
SLIPPAGE = 0.0005
DECEL_IMPROVEMENT_THRESHOLD = 0.02  # min OOS total return gain to include a signal

# Signal hyperparameter sweep grid (per Malik strategy)
D_PCT_SMOOTH_VALS = [5, 10, 20]  # EWM span for d_pct smoothing
RV_WINDOW_VALS = [10, 20, 30]  # rolling window (days) for realized vol
MACD_FAST_VALS = [8, 12, 16]  # MACD fast EMA period


# ── Data loading ───────────────────────────────────────────────────────────


def load_data(instrument: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/download_data_databento.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close", "high", "low"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index().dropna(subset=["close"])


# ── MA computation ─────────────────────────────────────────────────────────


def compute_ma(close: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    return close.rolling(period).mean()


# ── Deceleration signal functions ──────────────────────────────────────────


def d_pct(close: pd.Series, slow_ma: pd.Series, smooth: int = 10) -> pd.Series:
    """Distance from slow MA, normalised and EWM-smoothed.

    Positive = above MA (trend intact), Negative = below MA (deceleration).

    Args:
        close: Close price series.
        slow_ma: Slow moving average series.
        smooth: EWM smoothing span.

    Returns:
        Series in approximately [-2, +2], normalised to [-1, 1] via tanh.
    """
    raw = (close - slow_ma) / slow_ma * 100
    smoothed = raw.ewm(span=smooth, adjust=False).mean()
    return np.tanh(smoothed / 5)  # scale: 5% deviation ≈ tanh(1) ≈ 0.76


def rv_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Realised volatility signal -- inverted so rising vol = negative score.

    High volatility during a trend = deceleration warning.

    Args:
        close: Close price series.
        window: Rolling window for vol computation.

    Returns:
        Series in [-1, 1] -- positive = low vol (good), negative = high vol (warning).
    """
    from titan.research.metrics import BARS_PER_YEAR as _BPY

    log_ret = np.log(close / close.shift(1))
    rv_ann = log_ret.rolling(window).std() * np.sqrt(_BPY["D"])
    # Normalise: map 0-50% annual vol to [+1, -1]
    normalized = 1 - (rv_ann / 0.25).clip(0, 2)
    return normalized.clip(-1, 1)


def adx_signal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX-based signal -- high ADX = strong trend = positive score.

    ADX > 25 indicates trending, < 20 indicates choppy.

    Args:
        df: DataFrame with high, low, close columns.
        period: ADX lookback.

    Returns:
        Series in [-1, 1].
    """
    adx_vals = compute_adx(df, period=period)
    # Map: ADX 0: -1, ADX 25: 0, ADX 50+: +1
    return ((adx_vals - 25) / 25).clip(-1, 1)


def macd_signal(close: pd.Series, fast: int = 12) -> pd.Series:
    """MACD histogram signal -- positive and growing = bullish momentum.

    Histogram shrinking toward zero = deceleration warning.

    Args:
        close: Close price series.
        fast: Fast EMA period (default 12; Malik sweep: 8, 12, 16).

    Returns:
        Series in [-1, 1].
    """
    hist = compute_macd_hist(close, fast=fast)
    # Normalise via tanh: typical MACD hist range is ±1-5 for equities
    return np.tanh(hist)


def build_composite(signals: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Build weighted composite deceleration signal.

    Args:
        signals: Dict of signal name: Series (each in [-1, 1]).
        weights: Dict of signal name: weight (should sum to 1.0).

    Returns:
        Weighted sum series in approximately [-1, 1].
    """
    total_weight = sum(weights.get(k, 0.0) for k in signals)
    if total_weight == 0:
        return pd.Series(0.0, index=next(iter(signals.values())).index)
    composite = sum(signals[k] * weights.get(k, 0.0) for k in signals if k in weights)
    return composite / total_weight


# ── Signal hyperparameter sweep ────────────────────────────────────────────


def build_composite_from_params(
    close: pd.Series,
    slow_ma: pd.Series,
    df: pd.DataFrame,
    signals: list[str],
    d_pct_smooth: int,
    rv_window: int,
    macd_fast: int,
) -> pd.Series:
    """Build composite decel signal using parameterised signal functions."""
    parts: dict[str, pd.Series] = {}
    if "d_pct" in signals:
        parts["d_pct"] = d_pct(close, slow_ma, smooth=d_pct_smooth)
    if "rv_20" in signals:
        parts["rv_20"] = rv_signal(close, window=rv_window)
    if "adx_14" in signals:
        parts["adx_14"] = adx_signal(df)
    if "macd_hist" in signals:
        parts["macd_hist"] = macd_signal(close, fast=macd_fast)
    if not parts:
        return pd.Series(1.0, index=close.index)
    n = len(parts)
    weights = {k: 1.0 / n for k in parts}
    return build_composite(parts, weights)


def sweep_signal_params(
    close: pd.Series,
    slow_ma: pd.Series,
    df: pd.DataFrame,
    selected_signals: list[str],
    split: int,
    exec_close: pd.Series | None = None,
) -> dict:
    """Sweep hyperparameters of selected signals and return best param dict.

    Grid: D_PCT_SMOOTH_VALS x RV_WINDOW_VALS x MACD_FAST_VALS (27 combos max).
    Selects by OOS total return.

    Args:
        close: Full signal price series.
        slow_ma: Full slow MA series.
        df: Full OHLCV DataFrame (signal instrument).
        selected_signals: Signal names chosen in Stage 2 selection.
        split: Bar index separating IS from OOS.
        exec_close: Execution close for P&L (defaults to close when None).

    Returns:
        Dict with keys: d_pct_smooth, rv_window, macd_fast, oos_total_return.
    """
    _is_close, oos_close = close.iloc[:split], close.iloc[split:]
    _is_slow, oos_slow = slow_ma.iloc[:split], slow_ma.iloc[split:]
    _is_df, oos_df = df.iloc[:split], df.iloc[split:]
    oos_exec = exec_close.iloc[split:] if exec_close is not None else None

    best: dict = {}
    best_ret = float("-inf")

    print("\n  -- Hyperparameter sweep (signal internals) ----------")
    for d_smooth in D_PCT_SMOOTH_VALS:
        for rv_win in RV_WINDOW_VALS:
            for m_fast in MACD_FAST_VALS:
                oos_decel = build_composite_from_params(
                    oos_close,
                    oos_slow,
                    oos_df,
                    selected_signals,
                    d_smooth,
                    rv_win,
                    m_fast,
                ).fillna(0)
                oos_pf = run_backtest_with_decel(
                    oos_close,
                    oos_slow,
                    decel=oos_decel,
                    exec_close=oos_exec,
                )
                oos_ret = get_total_return(oos_pf)
                label = f"  d_smooth={d_smooth:2d}  rv_win={rv_win:2d}  macd_fast={m_fast:2d}"
                print(f"{label}  OOS Return={oos_ret:.1%}")
                if oos_ret > best_ret:
                    best_ret = oos_ret
                    best = {
                        "d_pct_smooth": d_smooth,
                        "rv_window": rv_win,
                        "macd_fast": m_fast,
                        "oos_total_return": round(oos_ret, 4),
                    }

    return best


# ── VectorBT backtest with decel entry filter ──────────────────────────────


def run_backtest_with_decel(
    close: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series | None = None,
    exec_close: pd.Series | None = None,
) -> "vbt.Portfolio":
    """Backtest with optional decel entry confirmation.

    Entry: close > slow_ma AND (decel >= 0 OR decel is None)
    Exit:  close < slow_ma (sole trend-reversal gate -- no fast MA)

    Args:
        close: Signal close series (used for regime gate and indicators).
        slow_ma: Slow MA series (sole trend boundary).
        decel: Composite decel series or None (baseline = no decel filter).
        exec_close: Execution close for P&L (defaults to close when None).

    Returns:
        VBT Portfolio object.
    """
    in_regime = close > slow_ma
    if decel is not None:
        entry_signal = in_regime & (decel >= 0)
    else:
        entry_signal = in_regime
    entries = entry_signal.shift(1).fillna(False)
    exits = (~in_regime).shift(1).fillna(False)
    exec_c = exec_close if exec_close is not None else close
    return vbt.Portfolio.from_signals(
        exec_c,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def get_sharpe(pf: "vbt.Portfolio") -> float:
    return float(pf.sharpe_ratio())


def get_total_return(pf: "vbt.Portfolio") -> float:
    return float(pf.total_return())


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETF Trend Stage 2: Deceleration Signal Selection."
    )
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument(
        "--load-state", action="store_true", help="Load Stage 1 results from state file"
    )
    parser.add_argument(
        "--signal-instrument",
        default=None,
        help="Signal source instrument (e.g. QQQ for TQQQ). Defaults to --instrument.",
    )
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()
    signal_inst = (args.signal_instrument or instrument).upper()
    is_dual = signal_inst != instrument

    print("=" * 60)
    print("  ETF Trend -- Stage 2: Deceleration Signal Selection")
    print("=" * 60)
    print(f"  Instrument: {instrument}")
    if is_dual:
        print(f"  Signal src: {signal_inst}  (signals on {signal_inst}, P&L on {instrument})")

    # ── Load Stage 1 ─────────────────────────────────────────────────────
    if args.load_state:
        s1 = get_stage1(inst_lower)
        if s1 is None:
            print("ERROR: Stage 1 state not found. Run run_stage1_ma_sweep.py first.")
            sys.exit(1)
        ma_type, slow_ma_p, _fast_ma_p = s1
    else:
        ma_type, slow_ma_p = "SMA", 200
        print(f"  [No state] Using defaults: {ma_type} slow={slow_ma_p}")

    print(f"  Stage 1 result: {ma_type} slow={slow_ma_p}")

    exec_df = load_data(instrument)
    if is_dual:
        sig_df = load_data(signal_inst)
        common = sig_df.index.intersection(exec_df.index)
        sig_df = sig_df.loc[common]
        exec_df = exec_df.loc[common]
        close = sig_df["close"]
        exec_close_full: pd.Series | None = exec_df["close"]
    else:
        sig_df = exec_df
        close = exec_df["close"]
        exec_close_full = None  # signals == P&L close

    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]
    is_exec = exec_close_full.iloc[:split] if exec_close_full is not None else None
    oos_exec = exec_close_full.iloc[split:] if exec_close_full is not None else None

    # Compute slow MA on full series then slice (no fast MA needed)
    slow_full = compute_ma(close, slow_ma_p, ma_type)
    is_slow = slow_full.iloc[:split]
    oos_slow = slow_full.iloc[split:]
    is_df = sig_df.iloc[:split]
    oos_df = sig_df.iloc[split:]

    # ── Baseline (no decel filter) ───────────────────────────────────────
    baseline_is_pf = run_backtest_with_decel(is_close, is_slow, exec_close=is_exec)
    baseline_oos_pf = run_backtest_with_decel(oos_close, oos_slow, exec_close=oos_exec)
    baseline_is = get_sharpe(baseline_is_pf)
    baseline_oos = get_sharpe(baseline_oos_pf)
    baseline_oos_ret = get_total_return(baseline_oos_pf)
    print(
        f"\n  Baseline (no decel): IS Sharpe={baseline_is:.3f}, "
        f"OOS Sharpe={baseline_oos:.3f}, OOS Return={baseline_oos_ret:.1%}"
    )

    # ── Compute all decel signals ────────────────────────────────────────
    all_signals: dict[str, tuple[pd.Series, pd.Series]] = {
        "d_pct": (
            d_pct(is_close, is_slow).fillna(0),
            d_pct(oos_close, oos_slow).fillna(0),
        ),
        "rv_20": (
            rv_signal(is_close, window=20).fillna(0),
            rv_signal(oos_close, window=20).fillna(0),
        ),
        "adx_14": (
            adx_signal(is_df).fillna(0),
            adx_signal(oos_df).fillna(0),
        ),
        "macd_hist": (
            macd_signal(is_close).fillna(0),
            macd_signal(oos_close).fillna(0),
        ),
    }

    # ── Test each signal independently ───────────────────────────────────
    results: list[dict] = []
    selected: list[str] = []

    print("\n  Testing each signal independently (objective: OOS total return):")
    for sig_name, (is_sig, oos_sig) in all_signals.items():
        is_pf = run_backtest_with_decel(is_close, is_slow, decel=is_sig, exec_close=is_exec)
        oos_pf = run_backtest_with_decel(oos_close, oos_slow, decel=oos_sig, exec_close=oos_exec)
        is_sharpe = get_sharpe(is_pf)
        oos_sharpe = get_sharpe(oos_pf)
        oos_ret = get_total_return(oos_pf)
        ret_improvement = oos_ret - baseline_oos_ret
        keeps = ret_improvement >= DECEL_IMPROVEMENT_THRESHOLD
        if keeps:
            selected.append(sig_name)
        results.append(
            {
                "signal": sig_name,
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "oos_total_return": round(oos_ret, 3),
                "vs_baseline_return": round(ret_improvement, 3),
                "selected": keeps,
            }
        )
        status = "KEEP" if keeps else "DROP"
        print(
            f"  [{status}] {sig_name:12s}  OOS Sharpe={oos_sharpe:.3f}  "
            f"OOS Return={oos_ret:.1%}  vs baseline={ret_improvement:+.1%}"
        )

    scoreboard = pd.DataFrame(results)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage2_{inst_lower}_decel_scoreboard.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved: {scoreboard_path.relative_to(PROJECT_ROOT)}")

    if not selected:
        print("\n  [WARN] No decel signals improved OOS Sharpe. Using baseline (no decel filter).")
        selected = []

    print(f"\n  Selected signals: {selected if selected else ['(none -- baseline)']}")

    # ── Equal weights for selected signals ───────────────────────────────
    n = len(selected)
    weights: dict[str, float] = {s: round(1.0 / n, 4) for s in selected} if n > 0 else {}

    # ── Hyperparameter sweep for selected signals ─────────────────────────
    best_params: dict = {}
    if selected:
        best_params = sweep_signal_params(
            close,
            slow_full,
            sig_df,
            selected,
            split,
            exec_close=exec_close_full,
        )
        print(
            f"\n  Best signal params: d_pct_smooth={best_params['d_pct_smooth']}  "
            f"rv_window={best_params['rv_window']}  macd_fast={best_params['macd_fast']}  "
            f"OOS Return={best_params['oos_total_return']:.1%}"
        )

    save_stage2(signals=selected, weights=weights, instrument=inst_lower, params=best_params)

    print("\n  [PASS] Stage 2 complete.")
    print(
        f"  Next: uv run python research/etf_trend/run_stage3_exits.py "
        f"--instrument {instrument} --load-state"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
