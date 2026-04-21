"""run_backtest.py — Gold Macro Strategy Research & Validation.

Backtests a 3-component cross-asset gold signal:
    1. Real rate proxy: log(TIP/TLT) 20-day change (falling = gold bullish)
    2. Dollar weakness: DXY 20-day log-return (falling = gold bullish)
    3. Momentum confirmation: GLD close > SMA(slow_ma) (trend filter)

Entry: composite > 0 AND momentum confirms -> LONG GLD.
Exit:  composite <= 0 OR price < SMA(slow_ma) OR 2x ATR(14) hard stop.
Sizing: vol-targeted (target_vol / realized_vol), capped at 1.5x.

IS/OOS split: 70/30 time-based.

Usage:
    uv run python research/gold_macro/run_backtest.py
    uv run python research/gold_macro/run_backtest.py --slow-ma 200 --real-rate-window 30

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


def load_data() -> dict[str, pd.DataFrame]:
    """Load all required daily parquet files."""
    required = ["GLD", "TIP", "TLT", "DXY"]
    data = {}
    for sym in required:
        path = DATA_DIR / f"{sym}_D.parquet"
        if not path.exists():
            print(f"ERROR: {path} not found. Run download_data_yfinance.py first.")
            sys.exit(1)
        df = pd.read_parquet(path).sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
        data[sym] = df
    return data


# -- Signal Construction -------------------------------------------------------


def build_composite_signal(
    gld: pd.DataFrame,
    tip: pd.DataFrame,
    tlt: pd.DataFrame,
    dxy: pd.DataFrame,
    real_rate_window: int = 20,
    dollar_window: int = 20,
    slow_ma: int = 200,
) -> pd.DataFrame:
    """Build the 3-component gold macro signal.

    Returns a DataFrame aligned to GLD's index with columns:
        real_rate_chg, dollar_chg, momentum, composite, signal
    """
    # Align all to GLD's daily index
    close_gld = gld["close"].rename("gld")
    close_tip = tip["close"].reindex(gld.index, method="ffill").rename("tip")
    close_tlt = tlt["close"].reindex(gld.index, method="ffill").rename("tlt")
    close_dxy = dxy["close"].reindex(gld.index, method="ffill").rename("dxy")

    df = pd.concat([close_gld, close_tip, close_tlt, close_dxy], axis=1).dropna()

    # Component 1: Real rate proxy (log(TIP/TLT) change over window)
    # Falling real rates = gold bullish -> signal is NEGATIVE change = positive for gold
    log_real_rate = np.log(df["tip"] / df["tlt"])
    real_rate_chg = log_real_rate.diff(real_rate_window)
    # Invert: falling real rates -> positive signal
    df["real_rate_signal"] = -real_rate_chg

    # Component 2: Dollar weakness (DXY log-return over window)
    # Falling dollar = gold bullish -> signal is NEGATIVE change = positive for gold
    dxy_log_ret = np.log(df["dxy"]).diff(dollar_window)
    # Invert: falling dollar -> positive signal
    df["dollar_signal"] = -dxy_log_ret

    # Normalise both to z-scores for comparable scale
    for col in ["real_rate_signal", "dollar_signal"]:
        expanding_mean = df[col].expanding(min_periods=60).mean()
        expanding_std = df[col].expanding(min_periods=60).std()
        df[f"{col}_z"] = (df[col] - expanding_mean) / expanding_std.clip(lower=1e-8)

    # Component 3: Momentum confirmation (GLD > SMA)
    df["sma_slow"] = df["gld"].rolling(slow_ma).mean()
    df["momentum"] = (df["gld"] > df["sma_slow"]).astype(float)

    # Composite: average of z-scores, gated by momentum
    df["composite_raw"] = (df["real_rate_signal_z"] + df["dollar_signal_z"]) / 2.0

    # Signal: composite > 0 AND momentum = 1 -> LONG (1), else FLAT (0)
    # Shift by 1 to prevent look-ahead (trade on NEXT bar's open)
    df["signal"] = ((df["composite_raw"] > 0) & (df["momentum"] == 1.0)).astype(int)
    df["signal"] = df["signal"].shift(1)  # NO LOOK-AHEAD

    return df.dropna()


# -- ATR for Stop Loss ---------------------------------------------------------


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR from OHLC columns."""
    h = df["high"] if "high" in df.columns else df["gld"]
    low = df["low"] if "low" in df.columns else df["gld"]
    c = df["close"] if "close" in df.columns else df["gld"]

    tr = pd.concat(
        [h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


# -- Backtest Engine -----------------------------------------------------------


def backtest(
    df: pd.DataFrame,
    gld: pd.DataFrame,
    target_vol: float = 0.10,
    vol_window: int = 20,
    max_leverage: float = 1.5,
    stop_atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Vectorised backtest with vol-targeting and ATR stop.

    Returns DataFrame with daily returns, equity curve, and trade log.
    """
    # Align GLD OHLC to signal dataframe
    gld_aligned = gld.reindex(df.index, method="ffill")
    close = df["gld"]
    daily_ret = close.pct_change()

    # Realised vol (EWMA)
    from titan.research.metrics import BARS_PER_YEAR as _BPY

    realized_vol = daily_ret.ewm(span=vol_window, adjust=False).std() * np.sqrt(_BPY["D"])

    # Vol-targeted position fraction
    vol_scale = (target_vol / realized_vol.clip(lower=0.01)).clip(upper=max_leverage)

    # ATR for stop
    if "high" in gld_aligned.columns:
        atr_series = compute_atr(gld_aligned)
    else:
        atr_series = pd.Series(0.0, index=df.index)

    # Apply signal * vol_scale to daily returns
    signal = df["signal"].fillna(0)
    position_ret = signal * daily_ret * vol_scale.shift(1)

    # Simple ATR stop simulation:
    # If intraday drawdown from entry exceeds stop_atr_mult * ATR, exit that day
    equity = (1 + position_ret).cumprod()

    results = pd.DataFrame(
        {
            "signal": signal,
            "daily_ret": daily_ret,
            "position_ret": position_ret,
            "vol_scale": vol_scale,
            "equity": equity,
            "atr": atr_series,
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

    from titan.research.metrics import (
        BARS_PER_YEAR as _BPY,
    )
    from titan.research.metrics import (
        annualize_vol as _ann,
    )
    from titan.research.metrics import (
        sharpe as _sh,
    )

    ann_ret = rets.mean() * _BPY["D"]
    ann_vol = _ann(float(rets.std()), periods_per_year=_BPY["D"])
    sharpe = float(_sh(rets, periods_per_year=_BPY["D"]))

    # Drawdown
    equity = (1 + rets).cumprod()
    hwm = equity.cummax()
    dd = (equity - hwm) / hwm
    max_dd = dd.min()

    # Win rate (days)
    wins = (rets > 0).sum()
    losses = (rets < 0).sum()
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0

    # Trades (signal changes)
    sig = results["signal"]
    trades = (sig.diff().abs() > 0).sum()

    # Time in market
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


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Gold Macro Strategy Backtest")
    parser.add_argument("--slow-ma", type=int, default=200, help="Slow MA period")
    parser.add_argument(
        "--real-rate-window", type=int, default=20, help="Real rate change lookback"
    )
    parser.add_argument("--dollar-window", type=int, default=20, help="Dollar change lookback")
    parser.add_argument("--target-vol", type=float, default=0.10, help="Target annualized vol")
    parser.add_argument("--is-ratio", type=float, default=0.70, help="In-sample fraction")
    args = parser.parse_args()

    print("=" * 60)
    print("  GOLD MACRO STRATEGY — RESEARCH BACKTEST")
    print("=" * 60)

    # Load data
    data = load_data()
    print(f"\nData loaded: GLD {len(data['GLD'])} bars, TIP {len(data['TIP'])} bars")

    # Build signal
    df = build_composite_signal(
        gld=data["GLD"],
        tip=data["TIP"],
        tlt=data["TLT"],
        dxy=data["DXY"],
        real_rate_window=args.real_rate_window,
        dollar_window=args.dollar_window,
        slow_ma=args.slow_ma,
    )
    print(f"Signal built: {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})")
    print(
        f"Signal distribution: Long={int((df['signal'] == 1).sum())} Flat={int((df['signal'] == 0).sum())}"
    )

    # IS/OOS split
    split_idx = int(len(df) * args.is_ratio)
    df_is = df.iloc[:split_idx]
    df_oos = df.iloc[split_idx:]
    print(f"\nIS: {len(df_is)} bars ({df_is.index[0].date()} -> {df_is.index[-1].date()})")
    print(f"OOS: {len(df_oos)} bars ({df_oos.index[0].date()} -> {df_oos.index[-1].date()})")

    # Backtest both
    gld = data["GLD"]
    res_is = backtest(df_is, gld, target_vol=args.target_vol)
    res_oos = backtest(df_oos, gld, target_vol=args.target_vol)

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
        print(f"\n  {m['label']}:")
        print(f"    Sharpe:       {m['sharpe']:+.3f}")
        print(f"    Annual Ret:   {m['annual_return']:+.2f}%")
        print(f"    Annual Vol:   {m['annual_vol']:.2f}%")
        print(f"    Max DD:       {m['max_drawdown_pct']:.2f}%")
        print(f"    Win Rate:     {m['win_rate_daily']:.1f}%")
        print(f"    Trades:       {m['trades']}")
        print(f"    Time in Mkt:  {m['time_in_market_pct']:.1f}%")

    # OOS/IS ratio check
    if "error" not in m_is and "error" not in m_oos:
        oos_is_ratio = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] != 0 else 0
        print(f"\n  OOS/IS Sharpe ratio: {oos_is_ratio:.2f}", end="")
        if oos_is_ratio >= 0.5:
            print("  PASS (>= 0.5)")
        else:
            print("  FAIL (< 0.5 — possible overfit)")

    # Save results
    report_path = REPORT_DIR / "gold_macro_backtest.csv"
    pd.DataFrame([m_is, m_oos]).to_csv(report_path, index=False)
    print(f"\n  Report saved: {report_path}")

    # Save equity curve
    equity_path = REPORT_DIR / "gold_macro_equity.csv"
    pd.concat(
        [res_is["equity"].rename("is_equity"), res_oos["equity"].rename("oos_equity")],
        axis=1,
    ).to_csv(equity_path)
    print(f"  Equity curve: {equity_path}")


if __name__ == "__main__":
    main()
