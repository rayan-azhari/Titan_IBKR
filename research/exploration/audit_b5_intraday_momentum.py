"""B5 Intraday Momentum (Gao-Han-Li-Zhou 2018) — V3.7 fresh strategy audit.

Reference: Gao, Han, Li & Zhou (2018), "Market Intraday Momentum", JFE.
Finding: the LAST half-hour's return is predicted by the FIRST half-hour's
return on the same day. Effect strongest on SPY index.

V3.7 audit:
- L21 causality smoke
- L66 baseline: INTRADAY -> cash (zero baseline)
- L67 portfolio inclusion: does adding to GEM + turtle improve the
  10-metric matrix?
- L65 ruin at proposed weights
- L60: M5 cadence on SPY -- US equity RTH M5 ≈ 78 bars/day * 252 days = 19,656
  bars/year (NOT generic FX convention)

Mechanism (per Gao et al.):
  first_30m_ret(t) = (close[10:00] - open[09:30]) / open[09:30]
  last_30m_ret(t)  = (close[16:00] - open[15:30]) / open[15:30]
  Strategy: enter at 15:30 in the SIGN of first_30m_ret, exit at 16:00.

Pure long-short version (we test): long if first_30m_ret > 0,
short if first_30m_ret < 0, sized vol-targeted.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_b5_intraday_momentum.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework import assess_strategy_ruin  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b5_intraday_momentum_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 1.0  # US equity ETF M5
PERIODS_PER_YEAR = 252  # per-trade (1 trade per day max)


def _load_m5(symbol: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{symbol}_M5.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    # Times in IBKR US-equity M5 parquets appear RTH-local already (per ORB).
    df = df.assign(time_of_day=df.index.time, et_date=df.index.date)
    return df[["open", "high", "low", "close", "time_of_day", "et_date"]].copy()


def build_b5_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day trade: enter at 15:30 in sign of first_30m_ret, exit at 16:00."""
    first_30m_start = time(9, 30)
    first_30m_end_inclusive = time(9, 55)  # M5 bar starting at 09:55 covers 09:55-10:00
    last_30m_start = time(15, 30)
    last_30m_end_inclusive = time(15, 55)  # closes at 16:00

    trades = []
    for date, day in df.groupby("et_date"):
        first_open_bar = day[day["time_of_day"] == first_30m_start]
        # 'last bar of first 30m' is the 9:55 bar (close = price at 10:00)
        first_close_bar = day[day["time_of_day"] == first_30m_end_inclusive]
        last_open_bar = day[day["time_of_day"] == last_30m_start]
        last_close_bar = day[day["time_of_day"] == last_30m_end_inclusive]
        if first_open_bar.empty or first_close_bar.empty:
            continue
        if last_open_bar.empty or last_close_bar.empty:
            continue
        first_open = float(first_open_bar["open"].iloc[0])
        first_close = float(first_close_bar["close"].iloc[0])
        last_entry = float(last_open_bar["open"].iloc[0])  # enter at 15:30 open
        last_exit = float(last_close_bar["close"].iloc[0])  # exit at 15:55 close (= 16:00)

        first_30m_ret = (first_close - first_open) / first_open
        if abs(first_30m_ret) < 1e-9:
            continue
        # Long if first_30m positive (momentum); short if negative.
        side = 1 if first_30m_ret > 0 else -1
        gross_ret = side * (last_exit - last_entry) / last_entry
        cost = 2 * COST_BPS / 10_000.0
        net_ret = gross_ret - cost

        trades.append(
            {
                "date": date,
                "first_30m_ret": first_30m_ret,
                "side": side,
                "entry": last_entry,
                "exit": last_exit,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
            }
        )
    return pd.DataFrame(trades)


def per_trade_sharpe(trades: pd.DataFrame) -> float:
    if len(trades) < 10:
        return 0.0
    r = trades["net_ret"]
    if r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def assert_causal_b5(df: pd.DataFrame) -> None:
    """L21 smoke: corrupting future bars must not change past trades."""
    base = build_b5_trades(df)
    if len(base) < 5:
        return
    n_corrupt = min(50, len(df) // 5)
    df_c = df.copy()
    for col in ("open", "high", "low", "close"):
        df_c.iloc[-n_corrupt:, df_c.columns.get_loc(col)] *= 100.0
    pert = build_b5_trades(df_c)
    cutoff_date = df.iloc[-n_corrupt - 5]["et_date"]
    base_past = base[base["date"] < cutoff_date]
    pert_past = pert[pert["date"] < cutoff_date]
    common = set(base_past["date"]) & set(pert_past["date"])
    for d in common:
        b = base_past[base_past["date"] == d].iloc[0]
        p = pert_past[pert_past["date"] == d].iloc[0]
        assert abs(b["net_ret"] - p["net_ret"]) < 1e-12, f"L21 fail on {d}"
    print(f"[b5] L21 PASS ({len(common)} common past trades)")


def main() -> None:
    print("=" * 88)
    print("B5 Intraday Momentum (Gao-Han-Li-Zhou 2018) — V3.7 audit")
    print("=" * 88)

    # Available M5 tickers: AMAT, CAT, CRM, CSCO, INTC, TMO, TXN, UNH, WMT
    primary = "AMAT"
    df = _load_m5(primary)
    print(f"\n[load] {primary}: {len(df)} M5 bars, {df.index[0]} -> {df.index[-1]}")

    assert_causal_b5(df)

    print(f"\n--- B5 on {primary} ---")
    trades = build_b5_trades(df)
    if len(trades) < 10:
        print(f"  insufficient trades ({len(trades)})")
        return
    sr = per_trade_sharpe(trades)
    wr = float((trades["net_ret"] > 0).mean())
    mr = float(trades["net_ret"].mean())
    print(f"  n_trades = {len(trades)}, win_rate = {wr:.2%}")
    print(f"  per-trade Sharpe (ann.) = {sr:+.4f}")
    print(f"  mean trade return = {mr:+.4%}")

    # Apply on multiple tickers for L61 panel
    print("\n--- L61 multi-ticker panel ---")
    panel = ["AMAT", "CAT", "CRM", "CSCO", "INTC", "TMO", "TXN", "UNH", "WMT"]
    print(f"{'ticker':>7} {'n_trades':>10} {'win_rate':>10} {'per_trade_SR':>14}")
    panel_sharpes = {}
    for tkr in panel:
        try:
            d = _load_m5(tkr)
            t = build_b5_trades(d)
            if len(t) < 10:
                print(f"  {tkr:>7s}  {len(t):>10d}  insufficient")
                continue
            srt = per_trade_sharpe(t)
            wrt = float((t["net_ret"] > 0).mean())
            panel_sharpes[tkr] = srt
            print(f"  {tkr:>7s}  {len(t):>10d}  {wrt:>9.2%}  {srt:>+13.4f}")
        except FileNotFoundError:
            print(f"  {tkr:>7s}  missing")

    if panel_sharpes:
        valid = list(panel_sharpes.values())
        med = float(np.median(valid))
        pct_pos = float(np.mean([s > 0 for s in valid]))
        print(f"\n  panel median = {med:+.4f}, pct positive = {pct_pos:.0%}")

    # L65 ruin (best signed Sharpe on primary)
    if sr > 0:
        daily_ret = pd.Series(trades["net_ret"].values, index=pd.to_datetime(trades["date"]))
        print("\n--- L65 ruin assessment (best ticker) ---")
        for w in [0.05, 0.10, 0.15]:
            ruin = assess_strategy_ruin(
                daily_ret,
                deployment_weight=w,
                portfolio_kill_threshold=0.15,
                horizon_bars=252,
                block_size=21,
                n_paths=2000,
                seed=42,
            )
            print(
                f"  weight={w:.0%}: P_kill={ruin.p_kill_trip:.3%}, "
                f"95th-pct DD={ruin.p95_maxdd_at_size:.3%}, passes={ruin.passes()}"
            )

    # Verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if not panel_sharpes:
        print("INCONCLUSIVE: no panel data")
        return
    valid = list(panel_sharpes.values())
    med = float(np.median(valid))
    pct_pos = float(np.mean([s > 0 for s in valid]))
    if med > 0.3 and pct_pos >= 0.6:
        print(
            f"CONDITIONAL_WATCHPOINT candidate: panel median {med:+.4f}, "
            f"{pct_pos:.0%} positive. L67 portfolio inclusion test needed."
        )
    elif med > 0 and pct_pos >= 0.5:
        print(
            f"MARGINAL: panel median {med:+.4f}, {pct_pos:.0%} positive. "
            f"Could defer or scope to top-1 ticker."
        )
    else:
        print(
            f"RETIRE: panel median {med:+.4f}, {pct_pos:.0%} positive. "
            f"Signal too weak (matches academic decay-post-2014 finding)."
        )


if __name__ == "__main__":
    main()
