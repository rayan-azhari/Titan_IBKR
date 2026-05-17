"""B5 SPY/QQQ/IWM intraday momentum re-audit (V3.7).

Replays the Gao-Han-Li-Zhou (2018) intraday-momentum test on the
INDEX ETFs (where the paper finds the effect strongest), with the
2-year IBKR M5 dataset just downloaded.

Goal: see whether the academic signal -- last-30m return predicted by
first-30m return on the same day -- survives in post-2024 SPY data.

Compares against the older AMAT/CAT panel (single-stock equity), which
returned a marginal verdict.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_b5_spy_intraday.py
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

# B5 entry-cost basket (ETFs are tight)
COST_BPS = 0.5
PERIODS_PER_YEAR = 252


def _load_m5(symbol: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{symbol}_M5.parquet"
    df = pd.read_parquet(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    df = df.assign(time_of_day=df.index.time, et_date=df.index.date)
    return df[["open", "high", "low", "close", "time_of_day", "et_date"]].copy()


def build_b5_trades(df: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Per-day trade: enter at 15:30 in SIGN of first_30m_ret, exit at 16:00."""
    first_open_t = time(9, 30)
    first_close_t = time(9, 55)   # M5 bar 09:55-10:00 -> close at 10:00
    last_open_t = time(15, 30)
    last_close_t = time(15, 55)   # closes at 16:00

    trades = []
    for date, day in df.groupby("et_date"):
        fo = day[day["time_of_day"] == first_open_t]
        fc = day[day["time_of_day"] == first_close_t]
        lo = day[day["time_of_day"] == last_open_t]
        lc = day[day["time_of_day"] == last_close_t]
        if fo.empty or fc.empty or lo.empty or lc.empty:
            continue
        first_open = float(fo["open"].iloc[0])
        first_close = float(fc["close"].iloc[0])
        last_entry = float(lo["open"].iloc[0])
        last_exit = float(lc["close"].iloc[0])

        first_30m_ret = (first_close - first_open) / first_open
        if abs(first_30m_ret) < 1e-9:
            continue
        side = 1 if first_30m_ret > 0 else -1
        gross_ret = side * (last_exit - last_entry) / last_entry
        cost = 2 * cost_bps / 10_000.0
        net_ret = gross_ret - cost

        trades.append({
            "date": date,
            "first_30m_ret": first_30m_ret,
            "side": side,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })
    return pd.DataFrame(trades)


def per_trade_sharpe(trades: pd.DataFrame) -> float:
    if len(trades) < 10:
        return 0.0
    r = trades["net_ret"]
    if r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def assert_causal_b5(df: pd.DataFrame, ticker: str) -> None:
    base = build_b5_trades(df)
    if len(base) < 5:
        return
    n_corrupt = min(200, len(df) // 5)
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
        assert abs(b["net_ret"] - p["net_ret"]) < 1e-12, f"L21 fail {ticker} on {d}"
    print(f"[b5] L21 PASS {ticker} ({len(common)} common past trades)")


def main() -> None:
    print("=" * 88)
    print("B5 SPY/QQQ/IWM Intraday Momentum -- V3.7 re-audit on 2y IBKR M5")
    print("=" * 88)

    panel = ["SPY", "QQQ", "IWM"]
    results = {}
    for tkr in panel:
        try:
            df = _load_m5(tkr)
        except FileNotFoundError:
            print(f"[{tkr}] missing parquet, skipping")
            continue
        print(f"\n[{tkr}] {len(df)} M5 bars  {df.index[0]} -> {df.index[-1]}")
        assert_causal_b5(df, tkr)
        trades = build_b5_trades(df)
        if len(trades) < 10:
            print(f"  insufficient trades ({len(trades)})")
            continue
        sr = per_trade_sharpe(trades)
        wr = float((trades["net_ret"] > 0).mean())
        mr = float(trades["net_ret"].mean())
        print(f"  n_trades={len(trades)}  win_rate={wr:.2%}  "
              f"per-trade Sharpe={sr:+.4f}  mean_ret={mr:+.4%}")
        results[tkr] = (sr, trades)

    if not results:
        print("\nNO RESULTS -- abort")
        return

    sharpes = {k: v[0] for k, v in results.items()}
    med = float(np.median(list(sharpes.values())))
    pct_pos = float(np.mean([s > 0 for s in sharpes.values()]))
    print("\n--- Panel summary ---")
    print(f"  median Sharpe = {med:+.4f}, pct positive = {pct_pos:.0%}")

    best_tkr = max(sharpes, key=sharpes.get)
    best_sr, best_trades = results[best_tkr]
    if best_sr > 0 and len(best_trades) >= 30:
        print(f"\n--- L65 ruin assessment (best ticker = {best_tkr}, Sharpe={best_sr:+.4f}) ---")
        daily_ret = pd.Series(best_trades["net_ret"].values,
                              index=pd.to_datetime(best_trades["date"]))
        for w in [0.05, 0.10, 0.15]:
            ruin = assess_strategy_ruin(
                daily_ret, deployment_weight=w,
                portfolio_kill_threshold=0.15, horizon_bars=252,
                block_size=21, n_paths=2000, seed=42,
            )
            print(f"  weight={w:.0%}: P_kill={ruin.p_kill_trip:.3%}, "
                  f"95th-pct DD={ruin.p95_maxdd_at_size:.3%}, passes={ruin.passes()}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if med > 0.5 and pct_pos >= 0.66:
        print(f"CONDITIONAL_WATCHPOINT candidate: median Sharpe {med:+.4f}, "
              f"{pct_pos:.0%} positive. L67 portfolio inclusion test needed.")
    elif med > 0 and pct_pos >= 0.5:
        print(f"MARGINAL: median {med:+.4f}, {pct_pos:.0%} positive. "
              f"Could scope to top-1 ticker only.")
    else:
        print(f"RETIRE: median {med:+.4f}, {pct_pos:.0%} positive. "
              f"Signal too weak post-2014 (matches academic decay).")


if __name__ == "__main__":
    main()
