"""run_backtest.py -- Equity Pairs Trading Research & Validation.

Tests cointegration and backtests spread mean-reversion for equity pairs.
Candidates: INTC/TXN (semiconductors), QQQ/SPY (index arbitrage).

Signal:
    1. Engle-Granger cointegration test on IS data (p < 0.05).
    2. OLS hedge ratio: A = beta * B + epsilon.
    3. Spread z-score = (spread - expanding_mean) / expanding_std.
    4. Entry: |z| > entry_z (default 2.0).
    5. Exit: |z| < exit_z (default 0.5).
    6. Invalidation: |z| > max_z (default 4.0) -> force close.

Walk-forward: re-estimate beta every refit_window bars.

IS/OOS split: 70/30 time-based.

Usage:
    uv run python research/pairs_trading/run_backtest.py
    uv run python research/pairs_trading/run_backtest.py --pair-a INTC --pair-b TXN
    uv run python research/pairs_trading/run_backtest.py --pair-a QQQ --pair-b SPY

Directive: Backtesting & Validation.md
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

from research.mean_reversion.pairs_trading import (
    estimate_hedge_ratio,
    test_cointegration,
)


def load_pair(sym_a: str, sym_b: str) -> tuple[pd.Series, pd.Series]:
    """Load daily closes for a pair, aligned by date."""
    a = pd.read_parquet(DATA_DIR / f"{sym_a}_D.parquet").sort_index()["close"]
    b = pd.read_parquet(DATA_DIR / f"{sym_b}_D.parquet").sort_index()["close"]
    # Align on common dates
    combined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return combined["a"], combined["b"]


def backtest_pairs(
    series_a: pd.Series,
    series_b: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_z: float = 4.0,
    refit_window: int = 126,  # ~6 months
    risk_per_trade: float = 0.01,
) -> pd.DataFrame:
    """Walk-forward pairs trading backtest.

    Re-estimates hedge ratio every refit_window bars to handle beta drift.
    """
    n = len(series_a)
    results = []

    # State
    position = 0  # +1 long spread, -1 short spread, 0 flat
    beta = None
    spread_mean = None
    spread_std = None

    for i in range(refit_window, n):
        # Re-estimate beta periodically
        if (i - refit_window) % refit_window == 0 or beta is None:
            is_a = series_a.iloc[max(0, i - refit_window * 2) : i]
            is_b = series_b.iloc[max(0, i - refit_window * 2) : i]
            beta = estimate_hedge_ratio(is_a, is_b)

        # Compute spread
        spread = float(series_a.iloc[i]) - beta * float(series_b.iloc[i])

        # Expanding mean/std (from start of data to avoid look-ahead)
        all_spreads = (
            series_a.iloc[refit_window : i + 1] - beta * series_b.iloc[refit_window : i + 1]
        )
        spread_mean = float(all_spreads.mean())
        spread_std = float(all_spreads.std())
        if spread_std < 1e-8:
            spread_std = 1.0

        z = (spread - spread_mean) / spread_std

        # Previous day spread return (for P&L)
        if i > refit_window:
            prev_spread = float(series_a.iloc[i - 1]) - beta * float(series_b.iloc[i - 1])
            spread_ret = (spread - prev_spread) / abs(prev_spread) if abs(prev_spread) > 0 else 0.0
        else:
            spread_ret = 0.0

        # P&L
        pnl = position * spread_ret * risk_per_trade

        # Signal logic
        old_position = position

        # Invalidation: force close
        if position != 0 and abs(z) > max_z:
            position = 0

        # Exit: z reverts
        elif position == 1 and z < exit_z:
            position = 0
        elif position == -1 and z > -exit_z:
            position = 0

        # Entry: z diverges
        elif position == 0 and z > entry_z:
            position = -1  # Short the spread (spread is too wide)
        elif position == 0 and z < -entry_z:
            position = 1  # Long the spread (spread is too narrow)

        results.append(
            {
                "date": series_a.index[i],
                "spread": spread,
                "z": z,
                "beta": beta,
                "position": old_position,
                "spread_ret": spread_ret,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(results).set_index("date")


def compute_metrics(results: pd.DataFrame, label: str) -> dict:
    """Standard performance metrics from backtest results."""
    pnl = results["pnl"].dropna()
    if len(pnl) < 20:
        return {"label": label, "error": "insufficient data"}

    from titan.research.metrics import BARS_PER_YEAR as _BPY
    from titan.research.metrics import annualize_vol as _ann
    from titan.research.metrics import sharpe as _sh

    ann_ret = pnl.mean() * _BPY["D"]
    ann_vol = _ann(float(pnl.std()), periods_per_year=_BPY["D"])
    sharpe = float(_sh(pnl, periods_per_year=_BPY["D"]))

    equity = (1 + pnl).cumprod()
    hwm = equity.cummax()
    dd = (equity - hwm) / hwm
    max_dd = dd.min()

    # Trade count (position changes)
    pos = results["position"]
    trades = (pos.diff().abs() > 0).sum()

    # Win rate (per-trade)
    trade_groups = (pos.diff().fillna(0) != 0).cumsum()
    trade_pnls = pnl.groupby(trade_groups).sum()
    trade_pnls = trade_pnls[trade_pnls != 0]
    wins = (trade_pnls > 0).sum()
    total = len(trade_pnls)
    win_rate = wins / total if total > 0 else 0.0

    time_in_market = (pos != 0).mean()

    return {
        "label": label,
        "annual_return": round(ann_ret * 100, 2),
        "annual_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "trades": int(trades),
        "time_in_market_pct": round(time_in_market * 100, 1),
        "bars": len(pnl),
    }


def main():
    parser = argparse.ArgumentParser(description="Equity Pairs Trading Backtest")
    parser.add_argument("--pair-a", default="INTC", help="Symbol A")
    parser.add_argument("--pair-b", default="TXN", help="Symbol B")
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--max-z", type=float, default=4.0)
    parser.add_argument("--refit", type=int, default=126, help="Beta refit window (bars)")
    parser.add_argument("--is-ratio", type=float, default=0.70)
    args = parser.parse_args()

    sym_a, sym_b = args.pair_a, args.pair_b
    print("=" * 60)
    print(f"  EQUITY PAIRS TRADING -- {sym_a} / {sym_b}")
    print("=" * 60)

    # Load data
    series_a, series_b = load_pair(sym_a, sym_b)
    print(
        f"\nData: {len(series_a)} aligned bars ({series_a.index[0].date()} -> {series_a.index[-1].date()})"
    )

    # Cointegration test (full sample first)
    coint_result = test_cointegration(series_a, series_b)
    print("\nCointegration (full sample):")
    print(f"  t-stat={coint_result['t_stat']:.3f}  p-value={coint_result['p_value']:.4f}")
    print(f"  Cointegrated: {'YES' if coint_result['is_cointegrated'] else 'NO'}")

    beta = estimate_hedge_ratio(series_a, series_b)
    print(f"  Hedge ratio (beta): {beta:.4f}")

    # IS/OOS split
    split_idx = int(len(series_a) * args.is_ratio)
    a_is, b_is = series_a.iloc[:split_idx], series_b.iloc[:split_idx]
    a_oos, b_oos = series_a.iloc[split_idx:], series_b.iloc[split_idx:]

    # IS cointegration
    coint_is = test_cointegration(a_is, b_is)
    print(
        f"\nCointegration (IS only): p={coint_is['p_value']:.4f} -> {'YES' if coint_is['is_cointegrated'] else 'NO'}"
    )

    print(f"\nIS: {len(a_is)} bars ({a_is.index[0].date()} -> {a_is.index[-1].date()})")
    print(f"OOS: {len(a_oos)} bars ({a_oos.index[0].date()} -> {a_oos.index[-1].date()})")

    # Backtest
    res_is = backtest_pairs(
        a_is,
        b_is,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_z=args.max_z,
        refit_window=args.refit,
    )
    res_oos = backtest_pairs(
        a_oos,
        b_oos,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_z=args.max_z,
        refit_window=args.refit,
    )

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
        print(f"    Win Rate:     {m['win_rate']:.1f}%")
        print(f"    Trades:       {m['trades']}")
        print(f"    Time in Mkt:  {m['time_in_market_pct']:.1f}%")

    if "error" not in m_is and "error" not in m_oos and m_is["sharpe"] != 0:
        ratio = m_oos["sharpe"] / m_is["sharpe"] if m_is["sharpe"] != 0 else 0
        print(f"\n  OOS/IS Sharpe ratio: {ratio:.2f}", end="")
        if ratio >= 0.5:
            print("  PASS (>= 0.5)")
        else:
            print("  FAIL (< 0.5)")

    # Save
    report_path = REPORT_DIR / f"pairs_{sym_a}_{sym_b}_backtest.csv"
    pd.DataFrame([m_is, m_oos]).to_csv(report_path, index=False)
    print(f"\n  Report: {report_path}")

    # Also test the other pair
    if sym_a == "INTC":
        print("\n\n--- Also testing QQQ/SPY ---")
        sa2, sb2 = load_pair("QQQ", "SPY")
        coint2 = test_cointegration(sa2, sb2)
        print(
            f"  Cointegration: p={coint2['p_value']:.4f} -> {'YES' if coint2['is_cointegrated'] else 'NO'}"
        )
        split2 = int(len(sa2) * args.is_ratio)
        res2 = backtest_pairs(
            sa2.iloc[split2:],
            sb2.iloc[split2:],
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            max_z=args.max_z,
            refit_window=args.refit,
        )
        m2 = compute_metrics(res2, "QQQ/SPY OOS")
        if "error" not in m2:
            print(f"  QQQ/SPY OOS Sharpe: {m2['sharpe']:+.3f}")


if __name__ == "__main__":
    main()
