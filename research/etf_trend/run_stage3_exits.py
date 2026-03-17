"""run_stage3_exits.py — Stage 3: Exit Mode Sweep.

Sweeps four exit strategies head-to-head to find the optimal exit condition.
Auto-loads Stage 1 + Stage 2 results via --load-state.

Exit modes:
  A — Close below slow_ma (regime reversal — classic trend following)
  B — Close below fast_ma (faster exit — reduces drawdown at cost of more whipsaws)
  C — Decel composite < exit_thresh (early exit before price breaks)
  D — A OR C, whichever fires first (combined — primary hypothesis)

Includes mandatory regime split analysis:
  - Bull trend (close > SMA250, ADX > 25)
  - Choppy (ADX < 20)
  - Bear trend (close < SMA250)

Usage:
    uv run python research/etf_trend/run_stage3_exits.py --instrument SPY --load-state
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

from research.etf_trend.run_stage2_decel import (  # noqa: E402
    adx_signal,
    build_composite,
    compute_ma,
    d_pct,
    macd_signal,
    rv_signal,
)
from research.etf_trend.state_manager import get_stage1, get_stage2, save_stage3  # noqa: E402
from titan.strategies.ml.features import adx as compute_adx  # noqa: E402

FEES = 0.001
SLIPPAGE = 0.0005

EXIT_MODES = ["A", "B", "C", "D"]
EXIT_DECEL_THRESHOLDS = [-0.5, -0.3, -0.1, 0.0]


def load_data(instrument: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_D.parquet"
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


def compute_decel_composite(
    close: pd.Series, df: pd.DataFrame, slow_ma_ser: pd.Series, signals: list[str]
) -> pd.Series:
    """Build composite decel signal from selected signal set.

    Args:
        close: Close price series.
        df: Full OHLCV DataFrame.
        slow_ma_ser: Slow moving average series.
        signals: List of selected signal names.

    Returns:
        Composite decel series in [-1, 1].
    """
    if not signals:
        return pd.Series(1.0, index=close.index)  # no filter — always pass

    parts: dict[str, pd.Series] = {}
    if "d_pct" in signals:
        parts["d_pct"] = d_pct(close, slow_ma_ser)
    if "rv_20" in signals:
        parts["rv_20"] = rv_signal(close, window=20)
    if "adx_14" in signals:
        parts["adx_14"] = adx_signal(df)
    if "macd_hist" in signals:
        parts["macd_hist"] = macd_signal(close)

    n = len(parts)
    weights = {k: 1.0 / n for k in parts}
    return build_composite(parts, weights)


def run_exit_backtest(
    close: pd.Series,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    decel: pd.Series,
    exit_mode: str,
    exit_decel_thresh: float = -0.3,
) -> "vbt.Portfolio":
    """Run long-only backtest with specified exit mode.

    Entry: in_regime AND decel >= 0 (shifted by 1).
    Exit: determined by exit_mode (shifted by 1).

    Args:
        close: Close price series.
        fast_ma: Fast MA series.
        slow_ma: Slow MA series.
        decel: Composite decel signal series.
        exit_mode: 'A', 'B', 'C', or 'D'.
        exit_decel_thresh: Threshold for decel exit (modes C and D).

    Returns:
        VBT Portfolio object.
    """
    in_regime = (close > fast_ma) & (close > slow_ma)
    entry_signal = in_regime & (decel >= 0)

    if exit_mode == "A":
        exit_signal = ~in_regime  # close below slow_ma
    elif exit_mode == "B":
        exit_signal = close < fast_ma
    elif exit_mode == "C":
        exit_signal = decel < exit_decel_thresh
    else:  # D — A OR C
        exit_signal = (~in_regime) | (decel < exit_decel_thresh)

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def regime_split(close: pd.Series, df: pd.DataFrame, n_sma: int = 250) -> dict[str, pd.Series]:
    """Classify each bar into a market regime.

    Args:
        close: Close price series.
        df: Full OHLCV DataFrame.
        n_sma: SMA period for regime classification.

    Returns:
        Dict with bool masks: 'bull', 'bear', 'choppy'.
    """
    sma250 = close.rolling(n_sma, min_periods=1).mean()
    adx_vals = compute_adx(df)
    bull = (close > sma250) & (adx_vals > 25)
    bear = close < sma250
    choppy = ~(bull | bear)
    return {"bull": bull, "bear": bear, "choppy": choppy}


def sharpe_in_regime(pf: "vbt.Portfolio", mask: pd.Series) -> float:
    """Compute Sharpe ratio using only returns in the given regime mask.

    Args:
        pf: VBT Portfolio object.
        mask: Boolean Series marking regime bars.

    Returns:
        Sharpe ratio for the masked period (annualised, 252 days).
    """
    ret = pf.returns()
    ret_masked = ret[mask.reindex(ret.index, fill_value=False)]
    if len(ret_masked) < 10:
        return float("nan")
    mean_r = ret_masked.mean()
    std_r = ret_masked.std()
    if std_r < 1e-8:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(252))


def extract_stats(pf: "vbt.Portfolio") -> dict:
    n = pf.trades.count()
    return {
        "sharpe": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "calmar": float(pf.calmar_ratio()),
        "total_return": float(pf.total_return()),
        "n_trades": int(n),
        "win_rate": float(pf.trades.win_rate()) if n > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 3: Exit Mode Sweep.")
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    parser.add_argument("--load-state", action="store_true", help="Load Stage 1+2 from state")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    print("=" * 60)
    print("  ETF Trend — Stage 3: Exit Mode Sweep")
    print("=" * 60)
    print(f"  Instrument: {instrument}")

    # ── Load Stage 1 + 2 ─────────────────────────────────────────────────
    if args.load_state:
        s1 = get_stage1(inst_lower)
        s2 = get_stage2(inst_lower)
        if s1 is None:
            print("ERROR: Stage 1 state not found.")
            sys.exit(1)
        ma_type, fast_ma_p, slow_ma_p = s1
        decel_signals, _ = s2 if s2 else ([], {})
    else:
        ma_type, fast_ma_p, slow_ma_p = "SMA", 50, 200
        decel_signals = []
        print("  [No state] Using defaults.")

    print(f"  MA: {ma_type} fast={fast_ma_p} slow={slow_ma_p}")
    print(f"  Decel signals: {decel_signals or '(none)'}")
    print(f"  Exit modes:    {EXIT_MODES}")
    print(f"  Decel thresholds: {EXIT_DECEL_THRESHOLDS}")

    df = load_data(instrument)
    close = df["close"]

    split = int(len(close) * 0.70)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]
    _is_df, oos_df = df.iloc[:split], df.iloc[split:]

    fast_full = compute_ma(close, fast_ma_p, ma_type)
    slow_full = compute_ma(close, slow_ma_p, ma_type)

    is_fast, is_slow = fast_full.iloc[:split], slow_full.iloc[:split]
    oos_fast, oos_slow = fast_full.iloc[split:], slow_full.iloc[split:]

    # ── Compute decel on full series then slice (avoids warmup edge) ─────
    decel_full = compute_decel_composite(close, df, slow_full, decel_signals)
    is_decel_raw = decel_full.iloc[:split].shift(1).fillna(0)
    oos_decel_raw = decel_full.iloc[split:].shift(1).fillna(0)

    # Regime masks (OOS only — that's what matters)
    regimes = regime_split(oos_close, oos_df)

    results: list[dict] = []
    total = len(EXIT_MODES) * len(EXIT_DECEL_THRESHOLDS)
    print(f"\n  Running {total} combinations ...\n")

    for exit_mode in EXIT_MODES:
        for thresh in EXIT_DECEL_THRESHOLDS:
            if exit_mode in ("A", "B") and thresh != EXIT_DECEL_THRESHOLDS[0]:
                continue  # thresh irrelevant for A and B

            is_pf = run_exit_backtest(is_close, is_fast, is_slow, is_decel_raw, exit_mode, thresh)
            oos_pf = run_exit_backtest(
                oos_close, oos_fast, oos_slow, oos_decel_raw, exit_mode, thresh
            )

            oos_stats = extract_stats(oos_pf)
            is_stats = extract_stats(is_pf)
            ratio = oos_stats["sharpe"] / is_stats["sharpe"] if is_stats["sharpe"] > 0.01 else 0.0

            bull_sharpe = sharpe_in_regime(oos_pf, regimes["bull"])
            bear_sharpe = sharpe_in_regime(oos_pf, regimes["bear"])
            choppy_sharpe = sharpe_in_regime(oos_pf, regimes["choppy"])

            label = f"Mode {exit_mode}" + (f" thresh={thresh}" if exit_mode in ("C", "D") else "")
            results.append(
                {
                    "exit_mode": exit_mode,
                    "exit_decel_thresh": thresh if exit_mode in ("C", "D") else "n/a",
                    "is_sharpe": round(is_stats["sharpe"], 3),
                    "oos_sharpe": round(oos_stats["sharpe"], 3),
                    "oos_is_ratio": round(ratio, 3),
                    "oos_calmar": round(oos_stats["calmar"], 3),
                    "oos_max_dd": round(oos_stats["max_drawdown"], 3),
                    "oos_n_trades": oos_stats["n_trades"],
                    "bull_sharpe": round(bull_sharpe, 3) if not np.isnan(bull_sharpe) else "n/a",
                    "bear_sharpe": round(bear_sharpe, 3) if not np.isnan(bear_sharpe) else "n/a",
                    "choppy_sharpe": round(choppy_sharpe, 3)
                    if not np.isnan(choppy_sharpe)
                    else "n/a",
                }
            )
            print(
                f"  {label:25s}  OOS Sharpe={oos_stats['sharpe']:.3f}  "
                f"Calmar={oos_stats['calmar']:.3f}  MaxDD={oos_stats['max_drawdown']:.1%}"
            )

    scoreboard = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
    scoreboard_path = REPORTS_DIR / f"etf_trend_stage3_{inst_lower}_exits.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    print(f"\n  Scoreboard saved → {scoreboard_path.relative_to(PROJECT_ROOT)}")

    best = scoreboard.iloc[0]
    best_mode = str(best["exit_mode"])
    best_thresh_raw = best["exit_decel_thresh"]
    best_thresh = float(best_thresh_raw) if best_thresh_raw != "n/a" else -0.3

    print("\n  ── Best exit configuration ───────────────────────")
    print(f"  Exit mode:   {best_mode}")
    if best_mode in ("C", "D"):
        print(f"  Decel thresh: {best_thresh}")
    print(f"  IS Sharpe:   {best['is_sharpe']}")
    print(f"  OOS Sharpe:  {best['oos_sharpe']}")
    print(f"  OOS Calmar:  {best['oos_calmar']}")

    print("\n  ── Regime split (best config) ────────────────────")
    for regime in ("bull", "bear", "choppy"):
        print(f"  {regime:10s}: Sharpe = {best[f'{regime}_sharpe']}")

    save_stage3(exit_mode=best_mode, exit_decel_thresh=best_thresh, instrument=inst_lower)

    print("\n  [PASS] Stage 3 complete.")
    print(
        f"  Next: uv run python research/etf_trend/run_stage4_sizing.py "
        f"--instrument {instrument} --load-state"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
