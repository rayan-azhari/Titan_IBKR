"""run_bond_equity_momentum.py -- Cross-Asset Bond→Equity Momentum.

Tests whether past bond returns predict future equity returns.
Research shows bond momentum positively predicts equity returns with a lag
(institutional constraint: bond investors move slowly into equities).

Signals tested:
  - TLT (long bonds) momentum → SPY/QQQ direction
  - IEF (intermediate bonds) momentum → SPY/QQQ direction
  - Combined TLT+IEF signal
  - Bond-equity divergence (bonds up + equities down = buy signal)

IS/OOS 70/30 split. Sweep lookback periods and holding periods.

Usage:
    uv run python research/cross_asset/run_bond_equity_momentum.py
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
    df = pd.read_parquet(DATA_DIR / f"{sym}_D.parquet")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    return df["close"].astype(float).sort_index()


def backtest_momentum_signal(
    signal: pd.Series,
    target_close: pd.Series,
    threshold: float = 0.0,
    hold_days: int = 20,
    spread_bps: float = 5.0,
    is_ratio: float = 0.70,
) -> dict:
    """Backtest: go long target when signal > threshold, flat otherwise.

    Uses hold_days minimum holding period to reduce turnover.
    """
    # Align
    common = signal.index.intersection(target_close.index)
    signal = signal.reindex(common)
    target = target_close.reindex(common)

    n = len(common)
    is_n = int(n * is_ratio)

    # Z-score signal on IS
    is_mean = signal.iloc[:is_n].mean()
    is_std = signal.iloc[:is_n].std()
    if is_std < 1e-8:
        is_std = 1.0
    sig_z = (signal - is_mean) / is_std

    # Position: shift(1) for no look-ahead, hold for minimum hold_days
    sig_shifted = sig_z.shift(1).fillna(0.0).values
    pos = np.zeros(n)
    bars_held = 0
    for i in range(n):
        if pos[max(0, i - 1)] != 0:
            bars_held += 1
        if bars_held >= hold_days and sig_shifted[i] <= threshold:
            pos[i] = 0.0
            bars_held = 0
        elif pos[max(0, i - 1)] == 0 and sig_shifted[i] > threshold:
            pos[i] = 1.0
            bars_held = 0
        else:
            pos[i] = pos[max(0, i - 1)]

    # Returns
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

    is_sh = _sharpe(is_rets)
    oos_sh = _sharpe(oos_rets)
    parity = oos_sh / is_sh if abs(is_sh) > 0.01 else 0.0
    trades = int(np.sum(transitions > 0))
    oos_trades = int(np.sum(transitions[is_n:] > 0))
    time_in_market = np.mean(pos[is_n:]) * 100

    return {
        "is_sharpe": round(is_sh, 3),
        "oos_sharpe": round(oos_sh, 3),
        "parity": round(parity, 3),
        "oos_dd_pct": round(_dd(oos_rets) * 100, 2),
        "total_trades": trades,
        "oos_trades": oos_trades,
        "time_in_market_pct": round(time_in_market, 1),
    }


def main() -> None:
    print("=" * 70)
    print("  CROSS-ASSET BOND->EQUITY MOMENTUM")
    print("=" * 70)

    # Load data
    tlt = load_daily("TLT")
    ief = load_daily("IEF")
    spy = load_daily("SPY")
    qqq = load_daily("QQQ")
    gld = load_daily("GLD")

    print(f"\n  TLT: {len(tlt)} bars | IEF: {len(ief)} bars")
    print(f"  SPY: {len(spy)} bars | QQQ: {len(qqq)} bars | GLD: {len(gld)} bars")

    # Sweep: bond instrument × lookback × target × holding period × threshold
    bond_instruments = {"TLT": tlt, "IEF": ief}
    targets = {"SPY": spy, "QQQ": qqq, "GLD": gld}
    lookbacks = [10, 20, 40, 60]
    hold_days_list = [10, 20, 40]
    thresholds = [0.0, 0.25, 0.50]

    rows = []
    total = (
        len(bond_instruments)
        * len(targets)
        * len(lookbacks)
        * len(hold_days_list)
        * len(thresholds)
    )
    print(f"\n  Sweeping {total} combinations...")

    for bond_name, bond_close in bond_instruments.items():
        for lookback in lookbacks:
            # Bond momentum signal: log-return over lookback period
            bond_mom = np.log(bond_close / bond_close.shift(lookback))

            # Also test combined TLT+IEF
            for target_name, target_close in targets.items():
                for hold_days in hold_days_list:
                    for th in thresholds:
                        result = backtest_momentum_signal(
                            bond_mom,
                            target_close,
                            threshold=th,
                            hold_days=hold_days,
                        )
                        rows.append(
                            {
                                "bond": bond_name,
                                "target": target_name,
                                "lookback": lookback,
                                "hold_days": hold_days,
                                "threshold": th,
                                **result,
                            }
                        )

    # Also test combined signal (TLT + IEF average momentum)
    for lookback in lookbacks:
        tlt_mom = np.log(tlt / tlt.shift(lookback))
        ief_mom = np.log(ief / ief.shift(lookback))
        combined = (tlt_mom + ief_mom) / 2
        combined = combined.dropna()
        for target_name, target_close in targets.items():
            for hold_days in hold_days_list:
                for th in thresholds:
                    result = backtest_momentum_signal(
                        combined,
                        target_close,
                        threshold=th,
                        hold_days=hold_days,
                    )
                    rows.append(
                        {
                            "bond": "TLT+IEF",
                            "target": target_name,
                            "lookback": lookback,
                            "hold_days": hold_days,
                            "threshold": th,
                            **result,
                        }
                    )

    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)

    # Quality gates
    passed = df[(df["oos_sharpe"] > 0.5) & (df["parity"] >= 0.5) & (df["oos_trades"] >= 10)]

    print(f"\n  Results: {len(df)} combos, {len(passed)} passed gates")

    # Top 20
    print("\n  Top 20 by OOS Sharpe:")
    print(
        f"  {'Bond':>7} {'Tgt':>4} {'LB':>4} {'Hold':>4} {'Thr':>5}"
        f" | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
        f" | {'DD%':>6} {'Trd':>4} {'TIM%':>5}"
    )
    print("  " + "-" * 72)
    for _, r in df.head(20).iterrows():
        flag = (
            "+" if r["oos_sharpe"] > 0.5 and r["parity"] >= 0.5 and r["oos_trades"] >= 10 else " "
        )
        print(
            f" {flag}{r['bond']:>6} {r['target']:>4} {int(r['lookback']):>4} {int(r['hold_days']):>4} {r['threshold']:>5.2f}"
            f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
            f" | {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4} {r['time_in_market_pct']:>4.0f}%"
        )

    # Compare bond→equity vs buy-and-hold
    print("\n  Buy-and-Hold Baselines (OOS period):")
    for name, close in targets.items():
        n = len(close)
        is_n = int(n * 0.70)
        oos = close.iloc[is_n:].pct_change().dropna()
        from titan.research.metrics import BARS_PER_YEAR as _BPY2
        from titan.research.metrics import sharpe as _sh2

        sh = float(_sh2(oos, periods_per_year=_BPY2["D"]))
        eq = (1 + oos).cumprod()
        dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        print(f"    {name}: Sharpe {sh:+.3f} | DD {dd * 100:+.1f}%")

    save_path = REPORTS_DIR / "cross_asset_bond_equity_momentum.csv"
    df.to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
