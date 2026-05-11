"""run_asset_class_confluence.py -- Cross-Asset AND-Gate Confluence.

Tests whether agreement across 4 asset classes (bonds, gold, dollar, equities)
produces a high-conviction timing signal. This is an AND-gate at the ASSET
CLASS level -- similar to what worked for GLD at the timeframe level.

Signal: when 3+ of 4 asset classes agree on risk-on/risk-off direction,
take full position. When 2 agree, half position. When <2, stay flat.

Asset class signals:
  - Bonds (TLT): 20d momentum > 0 = risk-off (rates falling, flight to safety)
  - Gold (GLD): 20d momentum > 0 = risk-off (safe haven demand)
  - Dollar (DXY): 20d momentum > 0 = risk-off (dollar strength)
  - Equities (SPY): 20d momentum > 0 = risk-on

Risk-on: long equities. Risk-off: long gold + long bonds.

Usage:
    uv run python research/cross_asset/run_asset_class_confluence.py
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
    for prefix in ["", "^"]:
        path = DATA_DIR / f"{prefix}{sym}_D.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            s = df["close"].astype(float).sort_index()
            s.index = s.index.normalize()  # strip time for cross-asset alignment
            return s
    raise FileNotFoundError(f"No data for {sym}")


def compute_momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Log-return over lookback period."""
    return np.log(close / close.shift(lookback))


def build_asset_class_signals(
    tlt: pd.Series,
    gld: pd.Series,
    dxy: pd.Series,
    spy: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """Build directional signals for each asset class.

    Convention: +1 = risk-on, -1 = risk-off.
    - Bonds up (TLT mom > 0) = rates falling = risk-off for equities → -1
    - Gold up (GLD mom > 0) = safe haven demand = risk-off → -1
    - Dollar up (DXY mom > 0) = risk-off → -1
    - Equities up (SPY mom > 0) = risk-on → +1
    """
    signals = pd.DataFrame(index=spy.index)
    signals["bonds"] = -np.sign(compute_momentum(tlt, lookback))  # inverted
    signals["gold"] = -np.sign(compute_momentum(gld, lookback))  # inverted
    signals["dollar"] = -np.sign(compute_momentum(dxy, lookback))  # inverted
    signals["equities"] = np.sign(compute_momentum(spy, lookback))  # direct

    # Shift by 1 for no look-ahead
    return signals.shift(1).dropna()


def backtest_confluence(
    signals: pd.DataFrame,
    target_close: pd.Series,
    target_name: str,
    min_agree: int = 3,
    mode: str = "risk_on",
    spread_bps: float = 5.0,
    is_ratio: float = 0.70,
) -> dict:
    """Backtest asset-class confluence.

    mode:
      - risk_on: long target when risk-on consensus
      - risk_off: long target when risk-off consensus
      - adaptive: long equities in risk-on, long gold/bonds in risk-off
    """
    common = signals.index.intersection(target_close.index)
    if len(common) < 100:
        return {
            "is_sharpe": 0.0,
            "oos_sharpe": 0.0,
            "parity": 0.0,
            "oos_dd_pct": 0.0,
            "oos_trades": 0,
            "time_in_market_pct": 0.0,
        }
    sigs = signals.reindex(common)
    target = target_close.reindex(common)

    n = len(common)
    is_n = int(n * is_ratio)

    # Count risk-on votes (how many asset classes say +1)
    risk_on_count = (sigs > 0).sum(axis=1)
    risk_off_count = (sigs < 0).sum(axis=1)

    pos = np.zeros(n)
    for i in range(n):
        on = risk_on_count.iloc[i]
        off = risk_off_count.iloc[i]
        if mode == "risk_on":
            if on >= min_agree:
                pos[i] = 1.0
            elif on >= min_agree - 1:
                pos[i] = 0.5
        elif mode == "risk_off":
            if off >= min_agree:
                pos[i] = 1.0
            elif off >= min_agree - 1:
                pos[i] = 0.5
        elif mode == "adaptive":
            if on >= min_agree:
                pos[i] = 1.0  # full risk-on
            elif off >= min_agree:
                pos[i] = -1.0  # full risk-off (short equities or long gold)
            elif on >= min_agree - 1:
                pos[i] = 0.5
            elif off >= min_agree - 1:
                pos[i] = -0.5

    daily_ret = target.pct_change().fillna(0.0).values
    transitions = np.zeros(n)
    transitions[0] = abs(pos[0])
    transitions[1:] = np.abs(pos[1:] - pos[:-1])
    strat_rets = daily_ret * pos - transitions * spread_bps / 10_000

    is_rets = pd.Series(strat_rets[:is_n])
    oos_rets = pd.Series(strat_rets[is_n:])

    def _sharpe(r):
        from titan.research.metrics import BARS_PER_YEAR as _BPY
        from titan.research.metrics import sharpe as _sh

        r = r[r != 0.0]
        if len(r) < 20:
            return 0.0
        return float(_sh(r, periods_per_year=_BPY["D"]))

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
    print("  CROSS-ASSET AND-GATE CONFLUENCE")
    print("=" * 70)

    tlt = load_daily("TLT")
    gld = load_daily("GLD")
    dxy = load_daily("DXY")
    spy = load_daily("SPY")
    qqq = load_daily("QQQ")

    print(f"\n  TLT: {len(tlt)} | GLD: {len(gld)} | DXY: {len(dxy)} | SPY: {len(spy)}")

    lookbacks = [10, 20, 40, 60]
    targets = {"SPY": spy, "QQQ": qqq, "GLD": gld}
    modes = ["risk_on", "risk_off", "adaptive"]
    min_agrees = [2, 3, 4]

    rows = []
    for lookback in lookbacks:
        signals = build_asset_class_signals(tlt, gld, dxy, spy, lookback=lookback)

        for target_name, target_close in targets.items():
            for mode in modes:
                # For risk_off mode targeting equities, skip (would need to short)
                # For risk_off mode targeting gold, makes sense (long gold in risk-off)
                for min_agree in min_agrees:
                    result = backtest_confluence(
                        signals,
                        target_close,
                        target_name,
                        min_agree=min_agree,
                        mode=mode,
                    )
                    rows.append(
                        {
                            "lookback": lookback,
                            "target": target_name,
                            "mode": mode,
                            "min_agree": min_agree,
                            **result,
                        }
                    )

    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)

    passed = df[(df["oos_sharpe"] > 0.5) & (df["parity"] >= 0.5) & (df["oos_trades"] >= 10)]

    print(f"\n  Results: {len(df)} combos, {len(passed)} passed gates")

    # Top 20
    print("\n  Top 20 by OOS Sharpe:")
    print(
        f"  {'LB':>4} {'Tgt':>4} {'Mode':>8} {'Agr':>3}"
        f" | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
        f" | {'DD%':>6} {'Trd':>4} {'TIM%':>5}"
    )
    print("  " + "-" * 65)
    for _, r in df.head(20).iterrows():
        flag = (
            "+" if r["oos_sharpe"] > 0.5 and r["parity"] >= 0.5 and r["oos_trades"] >= 10 else " "
        )
        print(
            f" {flag}{int(r['lookback']):>3} {r['target']:>4} {r['mode']:>8} {int(r['min_agree']):>3}"
            f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
            f" | {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4} {r['time_in_market_pct']:>4.0f}%"
        )

    # Signal statistics
    sigs_20 = build_asset_class_signals(tlt, gld, dxy, spy, lookback=20)
    risk_on_count = (sigs_20 > 0).sum(axis=1)
    print("\n  Signal Statistics (20-day lookback):")
    for count in [0, 1, 2, 3, 4]:
        pct = (risk_on_count == count).mean() * 100
        print(f"    {count} risk-on votes: {pct:.0f}% of time")

    save_path = REPORTS_DIR / "cross_asset_confluence.csv"
    df.to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
