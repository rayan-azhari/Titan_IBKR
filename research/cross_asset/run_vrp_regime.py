"""run_vrp_regime.py -- Volatility Risk Premium as Regime Filter.

Tests whether the VRP (VIX - realized vol) can be used as a regime filter
to enhance existing strategies. When VRP is high (implied > realized),
markets are complacent -- favor long equities. When VRP is low or negative,
markets are stressed -- reduce exposure.

Also tests VRP as a standalone timing signal for SPY/QQQ/GLD.

Usage:
    uv run python research/cross_asset/run_vrp_regime.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_daily(sym: str) -> pd.Series:
    path = DATA_DIR / f"{sym}_D.parquet"
    if not path.exists():
        path = DATA_DIR / f"^{sym}_D.parquet"  # VIX uses caret prefix
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    s = df["close"].astype(float).sort_index()
    s.index = s.index.normalize()  # strip time component for cross-asset alignment
    return s


def compute_vrp(vix: pd.Series, spy_close: pd.Series, rv_window: int = 20) -> pd.Series:
    """VRP = VIX - realized vol of SPY (annualized).

    VRP > 0: implied > realized, insurance is expensive, markets complacent.
    VRP < 0: implied < realized, markets in stress.
    """
    log_ret = np.log(spy_close / spy_close.shift(1))
    from titan.research.metrics import BARS_PER_YEAR as _BPY

    realized_vol = log_ret.rolling(rv_window).std() * np.sqrt(_BPY["D"]) * 100  # %, matches VIX
    vrp = vix - realized_vol
    return vrp.dropna()


def backtest_vrp_timing(
    vrp: pd.Series,
    target_close: pd.Series,
    mode: str = "long_when_high",
    vrp_lookback: int = 60,
    spread_bps: float = 5.0,
    is_ratio: float = 0.70,
) -> dict:
    """Backtest VRP as timing signal.

    Modes:
      - long_when_high: long target when VRP > rolling median (complacent market)
      - short_when_low: short target when VRP < rolling 25th pct (stressed market)
      - both: long when high, short when low, flat in middle
    """
    common = vrp.dropna().index.intersection(target_close.dropna().index)
    if len(common) < 100:
        return {
            "is_sharpe": 0.0,
            "oos_sharpe": 0.0,
            "parity": 0.0,
            "oos_dd_pct": 0.0,
            "oos_trades": 0,
            "time_in_market_pct": 0.0,
        }
    vrp = vrp.reindex(common)
    target = target_close.reindex(common)

    n = len(common)
    is_n = int(n * is_ratio)

    # Rolling percentiles (computed on expanding window to avoid look-ahead)
    vrp_median = vrp.expanding(min_periods=60).median()
    vrp_p25 = vrp.expanding(min_periods=60).quantile(0.25)
    vrp_p75 = vrp.expanding(min_periods=60).quantile(0.75)

    # Shift signals by 1 day
    vrp_shifted = vrp.shift(1).values
    med_shifted = vrp_median.shift(1).values
    p25_shifted = vrp_p25.shift(1).values
    p75_shifted = vrp_p75.shift(1).values

    pos = np.zeros(n)
    for i in range(n):
        if np.isnan(vrp_shifted[i]) or np.isnan(med_shifted[i]):
            continue
        if mode == "long_when_high":
            pos[i] = 1.0 if vrp_shifted[i] > med_shifted[i] else 0.0
        elif mode == "short_when_low":
            pos[i] = -1.0 if vrp_shifted[i] < p25_shifted[i] else 0.0
        elif mode == "both":
            if vrp_shifted[i] > p75_shifted[i]:
                pos[i] = 1.0
            elif vrp_shifted[i] < p25_shifted[i]:
                pos[i] = -1.0
            else:
                pos[i] = 0.0

    daily_ret = target.pct_change().fillna(0.0).values
    transitions = np.zeros(n)
    transitions[0] = abs(pos[0])
    transitions[1:] = np.abs(pos[1:] - pos[:-1])
    strat_rets = daily_ret * pos - transitions * spread_bps / 10_000

    is_rets = pd.Series(strat_rets[:is_n])
    oos_rets = pd.Series(strat_rets[is_n:])

    def _sharpe(r):
        from titan.research.metrics import BARS_PER_YEAR as _BPY2
        from titan.research.metrics import sharpe as _sh

        r = r[r != 0.0]
        if len(r) < 20:
            return 0.0
        return float(_sh(r, periods_per_year=_BPY2["D"]))

    def _dd(r):
        r = r[r != 0.0]
        if len(r) < 5:
            return 0.0
        eq = (1 + r).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())

    oos_trades = int(np.sum(transitions[is_n:] > 0))
    tim = np.mean(np.abs(pos[is_n:])) * 100

    return {
        "is_sharpe": round(_sharpe(is_rets), 3),
        "oos_sharpe": round(_sharpe(oos_rets), 3),
        "parity": round(_sharpe(oos_rets) / _sharpe(is_rets), 3)
        if abs(_sharpe(is_rets)) > 0.01
        else 0.0,
        "oos_dd_pct": round(_dd(oos_rets) * 100, 2),
        "oos_trades": oos_trades,
        "time_in_market_pct": round(tim, 1),
    }


def main() -> None:
    print("=" * 70)
    print("  VOLATILITY RISK PREMIUM REGIME FILTER")
    print("=" * 70)

    vix = load_daily("VIX")
    spy = load_daily("SPY")
    qqq = load_daily("QQQ")
    gld = load_daily("GLD")

    print(f"\n  VIX: {len(vix)} bars | SPY: {len(spy)} bars")

    # Compute VRP for different realized vol windows
    rv_windows = [10, 20, 40, 60]
    targets = {"SPY": spy, "QQQ": qqq, "GLD": gld}
    modes = ["long_when_high", "short_when_low", "both"]

    rows = []
    for rv_win in rv_windows:
        vrp = compute_vrp(vix, spy, rv_window=rv_win)
        for target_name, target_close in targets.items():
            for mode in modes:
                result = backtest_vrp_timing(vrp, target_close, mode=mode)
                rows.append(
                    {
                        "rv_window": rv_win,
                        "target": target_name,
                        "mode": mode,
                        **result,
                    }
                )

    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)

    passed = df[(df["oos_sharpe"] > 0.5) & (df["parity"] >= 0.5) & (df["oos_trades"] >= 10)]

    print(f"\n  Results: {len(df)} combos, {len(passed)} passed gates")

    print("\n  All results:")
    print(
        f"  {'RV':>4} {'Tgt':>4} {'Mode':>16} | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
        f" | {'DD%':>6} {'Trd':>4} {'TIM%':>5}"
    )
    print("  " + "-" * 65)
    for _, r in df.iterrows():
        flag = (
            "+" if r["oos_sharpe"] > 0.5 and r["parity"] >= 0.5 and r["oos_trades"] >= 10 else " "
        )
        print(
            f" {flag}{int(r['rv_window']):>3} {r['target']:>4} {r['mode']:>16}"
            f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
            f" | {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4} {r['time_in_market_pct']:>4.0f}%"
        )

    # VRP statistics
    vrp_20 = compute_vrp(vix, spy, rv_window=20)
    print("\n  VRP Statistics (20-day RV):")
    print(f"    Mean:     {vrp_20.mean():+.2f} vol pts")
    print(f"    Median:   {vrp_20.median():+.2f} vol pts")
    print(f"    Positive: {(vrp_20 > 0).mean() * 100:.0f}% of time")
    print(f"    > 5 pts:  {(vrp_20 > 5).mean() * 100:.0f}% of time (complacent)")
    print(f"    < -5 pts: {(vrp_20 < -5).mean() * 100:.0f}% of time (stressed)")

    save_path = REPORTS_DIR / "cross_asset_vrp_regime.csv"
    df.to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
