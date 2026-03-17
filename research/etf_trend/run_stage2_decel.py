"""run_stage2_decel.py — Stage 2: Deceleration Signal Selection.

Tests each deceleration signal independently against the Stage 1 baseline.
Selects signals that add ≥ 0.1 OOS Sharpe vs the pure MA regime baseline.

Deceleration signals serve two roles:
  1. Entry confirmation — only enter if composite decel ≥ 0
  2. Position scalar (in dynamic sizing) — position size ∝ decel strength

Signals tested:
  - d_pct:     (close - slow_ma) / slow_ma, EWM-smoothed — price momentum
  - rv_20:     20-day realised vol (annualised) — rising vol = warning
  - adx_14:    Average Directional Index — falling ADX = trend weakening
  - macd_hist: MACD histogram — shrinking histogram = deceleration

Auto-loads Stage 1 results via --load-state.

Usage:
    uv run python research/etf_trend/run_stage2_decel.py --instrument SPY --load-state
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
DECEL_IMPROVEMENT_THRESHOLD = 0.10  # min OOS Sharpe gain to include a signal


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
    """Realised volatility signal — inverted so rising vol = negative score.

    High volatility during a trend = deceleration warning.

    Args:
        close: Close price series.
        window: Rolling window for vol computation.

    Returns:
        Series in [-1, 1] — positive = low vol (good), negative = high vol (warning).
    """
    log_ret = np.log(close / close.shift(1))
    rv_ann = log_ret.rolling(window).std() * np.sqrt(252)
    # Normalise: map 0-50% annual vol to [+1, -1]
    normalized = 1 - (rv_ann / 0.25).clip(0, 2)
    return normalized.clip(-1, 1)


def adx_signal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX-based signal — high ADX = strong trend = positive score.

    ADX > 25 indicates trending, < 20 indicates choppy.

    Args:
        df: DataFrame with high, low, close columns.
        period: ADX lookback.

    Returns:
        Series in [-1, 1].
    """
    adx_vals = compute_adx(df, period=period)
    # Map: ADX 0 → -1, ADX 25 → 0, ADX 50+ → +1
    return ((adx_vals - 25) / 25).clip(-1, 1)


def macd_signal(close: pd.Series) -> pd.Series:
    """MACD histogram signal — positive and growing = bullish momentum.

    Histogram shrinking toward zero = deceleration warning.

    Args:
        close: Close price series.

    Returns:
        Series in [-1, 1].
    """
    hist = compute_macd_hist(close)
    # Normalise via tanh: typical MACD hist range is ±1-5 for equities
    return np.tanh(hist)


def build_composite(signals: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Build weighted composite deceleration signal.

    Args:
        signals: Dict of signal name → Series (each in [-1, 1]).
        weights: Dict of signal name → weight (should sum to 1.0).

    Returns:
        Weighted sum series in approximately [-1, 1].
    """
    total_weight = sum(weights.get(k, 0.0) for k in signals)
    if total_weight == 0:
        return pd.Series(0.0, index=next(iter(signals.values())).index)
    composite = sum(signals[k] * weights.get(k, 0.0) for k in signals if k in weights)
    return composite / total_weight


# ── VectorBT backtest with decel entry filter ──────────────────────────────


def run_backtest_with_decel(
    close: pd.Series,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series | None = None,
) -> "vbt.Portfolio":
    """Backtest with optional decel entry confirmation.

    Entry: in_regime AND (decel >= 0 OR decel is None)
    Exit:  not in_regime

    Args:
        close: Close price series.
        fast_ma: Fast MA series.
        slow_ma: Slow MA series.
        decel: Composite decel series or None (baseline).

    Returns:
        VBT Portfolio object.
    """
    in_regime = (close > fast_ma) & (close > slow_ma)
    if decel is not None:
        entry_signal = in_regime & (decel >= 0)
    else:
        entry_signal = in_regime
    entries = entry_signal.shift(1).fillna(False)
    exits = (~in_regime).shift(1).fillna(False)
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def get_sharpe(pf: "vbt.Portfolio") -> float:
    return float(pf.sharpe_ratio())


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETF Trend Stage 2: Deceleration Signal Selection."
    )
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument(
        "--load-state", action="store_true", help="Load Stage 1 results from state file"
    )
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    print("=" * 60)
    print("  ETF Trend — Stage 2: Deceleration Signal Selection")
    print("=" * 60)
    print(f"  Instrument: {instrument}")

    # ── Load Stage 1 ─────────────────────────────────────────────────────
    if args.load_state:
        s1 = get_stage1(inst_lower)
        if s1 is None:
            print("ERROR: Stage 1 state not found. Run run_optimisation.py first.")
            sys.exit(1)
        ma_type, fast_ma_p, slow_ma_p = s1
    else:
        ma_type, fast_ma_p, slow_ma_p = "SMA", 50, 200
        print(f"  [No state] Using defaults: {ma_type} {fast_ma_p}/{slow_ma_p}")

    print(f"  Stage 1 result: {ma_type} fast={fast_ma_p}, slow={slow_ma_p}")

    df = load_data(instrument)
    close = df["close"]

    split = int(len(close) * 0.70)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]

    # Compute MAs on full series then slice
    fast_full = compute_ma(close, fast_ma_p, ma_type)
    slow_full = compute_ma(close, slow_ma_p, ma_type)
    is_fast, is_slow = fast_full.iloc[:split], slow_full.iloc[:split]
    oos_fast, oos_slow = fast_full.iloc[split:], slow_full.iloc[split:]
    is_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    # ── Baseline (no decel filter) ───────────────────────────────────────
    baseline_is = get_sharpe(run_backtest_with_decel(is_close, is_fast, is_slow))
    baseline_oos = get_sharpe(run_backtest_with_decel(oos_close, oos_fast, oos_slow))
    print(f"\n  Baseline (no decel): IS={baseline_is:.3f}, OOS={baseline_oos:.3f}")

    # ── Compute all decel signals ────────────────────────────────────────
    is_slow_full = slow_full.iloc[:split]
    oos_slow_full = slow_full.iloc[split:]

    all_signals: dict[str, tuple[pd.Series, pd.Series]] = {
        "d_pct": (
            d_pct(is_close, is_slow_full).shift(1).fillna(0),
            d_pct(oos_close, oos_slow_full).shift(1).fillna(0),
        ),
        "rv_20": (
            rv_signal(is_close, window=20).shift(1).fillna(0),
            rv_signal(oos_close, window=20).shift(1).fillna(0),
        ),
        "adx_14": (
            adx_signal(is_df).shift(1).fillna(0),
            adx_signal(oos_df).shift(1).fillna(0),
        ),
        "macd_hist": (
            macd_signal(is_close).shift(1).fillna(0),
            macd_signal(oos_close).shift(1).fillna(0),
        ),
    }

    # ── Test each signal independently ───────────────────────────────────
    results: list[dict] = []
    selected: list[str] = []

    print("\n  Testing each signal independently:")
    for sig_name, (is_sig, oos_sig) in all_signals.items():
        is_sharpe = get_sharpe(run_backtest_with_decel(is_close, is_fast, is_slow, decel=is_sig))
        oos_sharpe = get_sharpe(
            run_backtest_with_decel(oos_close, oos_fast, oos_slow, decel=oos_sig)
        )
        improvement = oos_sharpe - baseline_oos
        keeps = improvement >= DECEL_IMPROVEMENT_THRESHOLD
        if keeps:
            selected.append(sig_name)
        results.append(
            {
                "signal": sig_name,
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(oos_sharpe, 3),
                "vs_baseline": round(improvement, 3),
                "selected": keeps,
            }
        )
        status = "KEEP" if keeps else "DROP"
        print(f"  [{status}] {sig_name:12s}  OOS={oos_sharpe:.3f}  vs baseline={improvement:+.3f}")

    scoreboard = pd.DataFrame(results)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage2_{inst_lower}_decel_scoreboard.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved → {scoreboard_path.relative_to(PROJECT_ROOT)}")

    if not selected:
        print("\n  [WARN] No decel signals improved OOS Sharpe. Using baseline (no decel filter).")
        selected = []

    print(f"\n  Selected signals: {selected if selected else ['(none — baseline)']}")

    # ── Equal weights for selected signals ───────────────────────────────
    n = len(selected)
    weights: dict[str, float] = {s: round(1.0 / n, 4) for s in selected} if n > 0 else {}

    save_stage2(signals=selected, weights=weights, instrument=inst_lower)

    print("\n  [PASS] Stage 2 complete.")
    print(
        f"  Next: uv run python research/etf_trend/run_stage3_exits.py "
        f"--instrument {instrument} --load-state"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
