"""run_backtest.py -- FX Carry Trade Research & Validation.

Backtests a carry-premium strategy on AUD/JPY (or other FX pairs) with
SMA trend filter for crash protection.

Signal:
    1. SMA(50) as trend filter.
    2. Long when carry_direction=+1 AND price > SMA.
    3. Exit when price < SMA.
    4. Vol-targeted sizing: target_vol(8%) / realized_vol.
    5. Position halved when VIX > 25 (if VIX data available).

IS/OOS split: 70/30 time-based.

Usage:
    uv run python research/fx_carry/run_backtest.py
    uv run python research/fx_carry/run_backtest.py --instrument AUD_JPY --sma 50

Directive: Backtesting & Validation.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -- Data Loading --------------------------------------------------------------


def load_data(instrument: str) -> pd.DataFrame:
    """Load daily parquet for the FX instrument."""
    path = DATA_DIR / f"{instrument}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run download_data_yfinance.py first.")
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
    return df.sort_index()


def load_vix() -> pd.Series | None:
    """Load VIX daily close if available."""
    for name in ["^VIX_D.parquet", "VIX_D.parquet"]:
        path = DATA_DIR / name
        if path.exists():
            df = pd.read_parquet(path).sort_index()
            df.index = pd.to_datetime(df.index, utc=True)
            return df["close"].astype(float)
    return None


# -- Signal Construction -------------------------------------------------------


def build_signal(
    df: pd.DataFrame,
    carry_direction: int = 1,
    sma_period: int = 50,
) -> pd.DataFrame:
    """Build carry + trend signal.

    Returns DataFrame with columns: close, sma, above_sma, signal.
    Signal shifted by 1 bar to prevent look-ahead.
    """
    close = df["close"].copy()

    sma = close.rolling(sma_period).mean()

    above_sma = close > sma

    # Signal: carry_dir=+1 AND above_sma -> LONG (1), else FLAT (0)
    if carry_direction == 1:
        raw_signal = above_sma.astype(int)
    else:
        raw_signal = (~above_sma).astype(int) * (-1)

    # Shift to prevent look-ahead (trade on next bar open)
    signal = raw_signal.shift(1)

    result = pd.DataFrame(
        {"close": close, "sma": sma, "above_sma": above_sma, "signal": signal},
        index=df.index,
    )
    return result.dropna()


# -- Backtest Engine -----------------------------------------------------------


def backtest(
    df: pd.DataFrame,
    vol_target_pct: float = 0.08,
    ewma_span: int = 20,
    max_leverage: float = 1.5,
    spread_bps: float = 3.0,
    slippage_bps: float = 1.0,
    vix: pd.Series | None = None,
    vix_halve_threshold: float = 25.0,
) -> pd.DataFrame:
    """Vectorised backtest with vol-targeting and VIX halving.

    Returns DataFrame with daily returns, equity curve, and sizing.
    """
    close = df["close"]
    daily_ret = close.pct_change()

    # Realized vol (EWMA annualized)
    ewma_var = daily_ret.ewm(span=ewma_span, adjust=False).var()
    realized_vol = np.sqrt(ewma_var.clip(lower=1e-8) * 252)

    # Vol-targeted position fraction
    vol_scale = (vol_target_pct / realized_vol).clip(upper=max_leverage)

    # VIX halving
    if vix is not None:
        vix_aligned = vix.reindex(df.index, method="ffill")
        vix_mask = vix_aligned > vix_halve_threshold
        vol_scale = vol_scale.where(~vix_mask, vol_scale * 0.5)

    # Transaction costs on position transitions
    signal = df["signal"].fillna(0)
    transitions = (signal.diff().abs() > 0).astype(float)
    cost_per_transition = (spread_bps + slippage_bps) / 10_000

    # Position return: signal * daily_return * vol_scale - costs
    # Use lagged vol_scale to avoid look-ahead
    position_ret = signal * daily_ret * vol_scale.shift(1) - transitions * cost_per_transition

    equity = (1 + position_ret.fillna(0)).cumprod()

    results = pd.DataFrame(
        {
            "signal": signal,
            "daily_ret": daily_ret,
            "vol_scale": vol_scale,
            "position_ret": position_ret,
            "equity": equity,
        },
        index=df.index,
    )
    return results


# -- Performance Metrics -------------------------------------------------------


def compute_metrics(results: pd.DataFrame, label: str = "") -> dict:
    """Compute standard performance metrics."""
    rets = results["position_ret"].dropna()
    if len(rets) < 20:
        return {"label": label, "error": "insufficient data"}

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    equity = (1 + rets).cumprod()
    hwm = equity.cummax()
    dd = (equity - hwm) / hwm
    max_dd = dd.min()

    wins = (rets > 0).sum()
    losses = (rets < 0).sum()
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0

    sig = results["signal"]
    trades = (sig.diff().abs() > 0).sum()
    time_in_market = (sig != 0).mean()

    return {
        "label": label,
        "annual_return": round(ann_ret * 100, 2),
        "annual_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_daily": round(win_rate * 100, 1),
        "trades": int(trades),
        "time_in_market_pct": round(time_in_market * 100, 1),
        "bars": len(rets),
    }


# -- Public API for portfolio loader -------------------------------------------


def run_fx_carry_backtest(
    instrument: str = "AUD_JPY",
    carry_direction: int = 1,
    sma_period: int = 50,
    vol_target_pct: float = 0.08,
    ewma_span: int = 20,
    spread_bps: float = 3.0,
    slippage_bps: float = 1.0,
    is_ratio: float = 0.70,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run full backtest and return (full_results, oos_daily_returns).

    This is the entry point for the portfolio OOS loader.
    """
    df = load_data(instrument)
    vix = load_vix()

    sig_df = build_signal(df, carry_direction=carry_direction, sma_period=sma_period)

    results = backtest(
        sig_df,
        vol_target_pct=vol_target_pct,
        ewma_span=ewma_span,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        vix=vix,
    )

    # IS/OOS split
    split_idx = int(len(results) * is_ratio)
    oos_returns = results["position_ret"].iloc[split_idx:].dropna()
    oos_returns.index = oos_returns.index.normalize()
    oos_returns.name = f"fx_carry_{instrument.lower()}"

    return results, oos_returns


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FX Carry Trade Backtest")
    parser.add_argument("--instrument", default="AUD_JPY", help="FX pair (e.g. AUD_JPY)")
    parser.add_argument("--carry-dir", type=int, default=1, help="+1 long carry, -1 short")
    parser.add_argument("--sma", type=int, default=50, help="SMA period")
    parser.add_argument("--vol-target", type=float, default=0.08, help="Target vol (fraction)")
    parser.add_argument("--is-ratio", type=float, default=0.70, help="In-sample fraction")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FX CARRY TRADE -- {args.instrument}")
    print("=" * 60)

    df = load_data(args.instrument)
    vix = load_vix()
    print(f"\nData loaded: {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})")
    if vix is not None:
        print(f"VIX data loaded: {len(vix)} bars")
    else:
        print("VIX data not available -- skipping VIX halving")

    sig_df = build_signal(df, carry_direction=args.carry_dir, sma_period=args.sma)
    print(f"Signal built: {len(sig_df)} bars")
    long_pct = (sig_df["signal"] == 1).mean() * 100
    print(f"Signal distribution: Long={long_pct:.1f}% Flat={100 - long_pct:.1f}%")

    # IS/OOS split
    split_idx = int(len(sig_df) * args.is_ratio)
    sig_is = sig_df.iloc[:split_idx]
    sig_oos = sig_df.iloc[split_idx:]
    print(f"\nIS:  {len(sig_is)} bars ({sig_is.index[0].date()} -> {sig_is.index[-1].date()})")
    print(f"OOS: {len(sig_oos)} bars ({sig_oos.index[0].date()} -> {sig_oos.index[-1].date()})")

    # Backtest both periods
    res_is = backtest(sig_is, vol_target_pct=args.vol_target, vix=vix)
    res_oos = backtest(sig_oos, vol_target_pct=args.vol_target, vix=vix)

    m_is = compute_metrics(res_is, "IS")
    m_oos = compute_metrics(res_oos, "OOS")

    # Report
    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)
    for m in [m_is, m_oos]:
        if "error" in m:
            print(f"  {m['label']}: {m['error']}")
            continue
        print(
            f"  {m['label']:4s} | Sharpe {m['sharpe']:+.3f}"
            f" | Ret {m['annual_return']:+.1f}%"
            f" | Vol {m['annual_vol']:.1f}%"
            f" | DD {m['max_drawdown_pct']:.1f}%"
            f" | WR {m['win_rate_daily']:.0f}%"
            f" | Trades {m['trades']}"
        )

    # OOS/IS ratio
    if "sharpe" in m_is and "sharpe" in m_oos and m_is["sharpe"] != 0:
        ratio = m_oos["sharpe"] / m_is["sharpe"]
        status = "PASS" if ratio >= 0.5 else "FAIL"
        print(f"\n  OOS/IS ratio: {ratio:.2f} [{status}]")

    # Save OOS returns
    oos_rets = res_oos["position_ret"].dropna()
    cache_path = REPORT_DIR / f"fx_carry_{args.instrument.lower()}_oos_daily.parquet"
    oos_rets.to_frame().to_parquet(cache_path)
    print(f"\n  OOS returns saved to: {cache_path}")

    # Save equity curve
    equity_path = REPORT_DIR / f"fx_carry_{args.instrument.lower()}_equity.csv"
    full_results = backtest(sig_df, vol_target_pct=args.vol_target, vix=vix)
    full_results[["equity", "signal"]].to_csv(equity_path)
    print(f"  Equity curve saved to: {equity_path}")


if __name__ == "__main__":
    main()
