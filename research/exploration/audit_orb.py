"""orb V3.7 audit (Wave B, sparse-trade per-trade Sharpe).

Strategy: M5 Opening Range Breakout. First 30 min (09:30-10:00 ET) define
range. Enter on first breakout (high/low) between 10:00 and 11:00. Exit at
15:50 ET. Sparse: ~1 trade/day max per ticker.

V3.7 audit:
- L21 causality smoke
- L25/L06 per-trade Sharpe (NOT per-bar; sparse-trade convention)
- L61 multi-ticker built in (7 tickers in live config)
- L66 baseline: INTRADAY_BREAKOUT class -> cash (zero baseline)
- L65 ruin at proposed weight

Simplified model: no bracket orders, no per-ticker config; uses single
opening-range definition + flat-by-15:50 exit. Real live has tighter
risk overlays.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_orb.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework import assess_strategy_ruin  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "orb_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")
TICKERS = ["UNH", "AMAT", "TXN", "INTC", "CAT", "WMT", "TMO"]
COST_BPS_PER_TURNOVER = 1.0  # US equity M5, conservative


def _load_m5(ticker: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{ticker}_M5.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    # Times in IBKR US-equity M5 parquets are already RTH-local (09:30-15:55).
    # Use index.time / index.date directly without tz conversion.
    df_et = df.copy()
    df_et["et_time"] = df.index.time
    df_et["et_date"] = df.index.date
    return df_et


def orb_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-day trade list: opening range, breakout, flat at 15:50.

    Returns DataFrame with one row per day where a trade triggered:
    date, side, entry_price, exit_price, ret (net of cost).
    """
    import datetime as dt
    or_end = dt.time(10, 0)  # 09:30-10:00 = opening range
    entry_cutoff = dt.time(11, 0)
    exit_time = dt.time(15, 50)

    trades = []
    for trade_date, day_bars in df.groupby("et_date"):
        if len(day_bars) < 6:
            continue
        # Opening range = bars from 09:30 to 10:00 (exclusive)
        or_bars = day_bars[(day_bars["et_time"] >= dt.time(9, 30)) &
                          (day_bars["et_time"] < or_end)]
        if len(or_bars) < 3:
            continue
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        # Look for breakout between 10:00 and 11:00
        breakout_bars = day_bars[(day_bars["et_time"] >= or_end) &
                                (day_bars["et_time"] < entry_cutoff)]
        if breakout_bars.empty:
            continue
        side = 0
        entry_price = 0.0
        entry_idx = None
        for idx, bar in breakout_bars.iterrows():
            if side == 0:
                if bar["high"] > or_high:
                    side = 1
                    entry_price = or_high
                    entry_idx = idx
                    break
                elif bar["low"] < or_low:
                    side = -1
                    entry_price = or_low
                    entry_idx = idx
                    break
        if side == 0:
            continue
        # Exit at 15:50 ET
        exit_bars = day_bars[(day_bars["et_time"] >= exit_time) &
                            (day_bars.index > entry_idx)]
        if exit_bars.empty:
            continue
        exit_price = float(exit_bars["close"].iloc[0])
        if side == 1:
            gross_ret = (exit_price - entry_price) / entry_price
        else:
            gross_ret = (entry_price - exit_price) / entry_price
        # 2x turnover (entry + exit) at 1bp each
        cost = 2 * COST_BPS_PER_TURNOVER / 10_000.0
        net_ret = gross_ret - cost
        trades.append({
            "date": trade_date,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })
    return pd.DataFrame(trades)


def assert_causal_orb(df: pd.DataFrame) -> None:
    """L21 smoke for ORB: corrupt future bars, verify past trades unchanged."""
    base_trades = orb_trades(df)
    if len(base_trades) < 10:
        return  # not enough trades to test
    last_n = min(50, len(df) // 5)
    cutoff = len(df) - last_n
    df_corrupt = df.copy()
    df_corrupt.iloc[cutoff:, df_corrupt.columns.get_loc("close")] *= 100.0
    df_corrupt.iloc[cutoff:, df_corrupt.columns.get_loc("high")] *= 100.0
    df_corrupt.iloc[cutoff:, df_corrupt.columns.get_loc("low")] *= 100.0
    corrupt_trades = orb_trades(df_corrupt)
    # Past trades (entered before cutoff date) should be unchanged
    safe_cutoff_date = df.iloc[cutoff - 1]["et_date"]
    base_past = base_trades[base_trades["date"] < safe_cutoff_date]
    corrupt_past = corrupt_trades[corrupt_trades["date"] < safe_cutoff_date]
    if len(base_past) == 0 or len(corrupt_past) == 0:
        return
    common_dates = set(base_past["date"]) & set(corrupt_past["date"])
    for d in common_dates:
        b_row = base_past[base_past["date"] == d].iloc[0]
        c_row = corrupt_past[corrupt_past["date"] == d].iloc[0]
        assert abs(b_row["net_ret"] - c_row["net_ret"]) < 1e-12, f"L21 fail on {d}"
    print(f"[orb] L21 PASS ({len(common_dates)} common past trades)")


def per_trade_sharpe(trades: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Annualised per-trade Sharpe: assumes 1 trade per business day."""
    if len(trades) < 10:
        return 0.0
    rets = trades["net_ret"]
    if rets.std(ddof=1) < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year))


def main() -> None:
    print("=" * 88)
    print("orb V3.7 audit (per-trade Sharpe + L61 panel + L66 cash baseline + L65)")
    print("=" * 88)

    # Run on first ticker for L21 smoke
    df0 = _load_m5(TICKERS[0])
    print(f"\n[load] {TICKERS[0]}: {len(df0)} M5 bars, "
          f"{df0.index[0]} -> {df0.index[-1]}")
    assert_causal_orb(df0)

    # Per-ticker results
    print("\n--- L61 multi-ticker panel ---")
    print(f"{'ticker':>7} {'n_trades':>10} {'win_rate':>10} {'per_trade_SR':>14} {'mean_ret':>12} {'std_ret':>10}")
    panel_sharpes = {}
    daily_returns_combined = {}  # date -> avg net_ret across tickers
    for tkr in TICKERS:
        try:
            df = _load_m5(tkr)
            trades = orb_trades(df)
            n = len(trades)
            if n < 10:
                print(f"  {tkr:>7s}  {n:>10d}  (insufficient trades, <10)")
                continue
            win_rate = float((trades["net_ret"] > 0).mean())
            sr = per_trade_sharpe(trades)
            mean_r = float(trades["net_ret"].mean())
            std_r = float(trades["net_ret"].std(ddof=1))
            panel_sharpes[tkr] = sr
            print(f"  {tkr:>7s}  {n:>10d}  {win_rate:>9.2%}  {sr:>+13.4f}  {mean_r:>+11.4%} {std_r:>9.4%}")
            # Build daily-aggregated returns for portfolio
            for _, row in trades.iterrows():
                d = row["date"]
                daily_returns_combined[d] = daily_returns_combined.get(d, 0.0) + row["net_ret"]
        except FileNotFoundError:
            print(f"  {tkr}: missing data")

    if not panel_sharpes:
        print("\nVERDICT: RETIRE (no valid tickers)")
        return

    valid = list(panel_sharpes.values())
    panel_median = float(np.median(valid))
    panel_mean = float(np.mean(valid))
    pct_positive = float(np.mean([s > 0 for s in valid]))
    print(f"\n  panel median per-trade Sharpe = {panel_median:+.4f}")
    print(f"  panel mean per-trade Sharpe = {panel_mean:+.4f}")
    print(f"  pct of tickers with positive Sharpe: {pct_positive:.0%}")

    # Build daily portfolio return (equal-weight across tickers)
    n_tickers = len(panel_sharpes)
    daily_port_ret = pd.Series({
        d: r / n_tickers for d, r in daily_returns_combined.items()
    }).sort_index()
    daily_port_ret.index = pd.to_datetime(daily_port_ret.index)

    print(f"\n--- ORB portfolio (equal-weight {n_tickers} tickers) ---")
    if len(daily_port_ret) >= 100:
        port_sr = float(daily_port_ret.mean() / daily_port_ret.std(ddof=1) * np.sqrt(252))
        port_mean = float(daily_port_ret.mean() * 252)
        print(f"  Portfolio Sharpe = {port_sr:+.4f}")
        print(f"  Annualised return = {port_mean:+.4f}")

        # L65 ruin
        print("\n--- L65 ruin assessment (ORB portfolio) ---")
        for w in [0.05, 0.10, 0.15, 0.20]:
            ruin = assess_strategy_ruin(
                daily_port_ret, deployment_weight=w,
                portfolio_kill_threshold=0.15,
                horizon_bars=252, block_size=21, n_paths=2000, seed=42,
            )
            print(f"  weight={w:.0%}: P_kill={ruin.p_kill_trip:.3%}, "
                  f"95th-pct DD={ruin.p95_maxdd_at_size:.3%}, passes={ruin.passes()}")
    else:
        print(f"  Insufficient days ({len(daily_port_ret)}) for portfolio metrics")

    # Verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if panel_median <= 0 or pct_positive < 0.5:
        print(f"RETIRE: panel median Sharpe {panel_median:+.4f} or {pct_positive:.0%} positive "
              f"tickers fails generalisation gate. Signal layer doesn't generalise.")
    elif panel_median > 0 and pct_positive >= 0.5:
        print(f"CONDITIONAL_WATCHPOINT candidate: panel median {panel_median:+.4f} "
              f"and {pct_positive:.0%} positive tickers. ORB has weak-but-real edge across "
              f"the live universe. Joint L65 vs current portfolio still needed before deploy.")


if __name__ == "__main__":
    main()
