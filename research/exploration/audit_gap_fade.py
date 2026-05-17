"""gap_fade V3.7 audit (Wave C, EUR/USD M5 session strategy).

Strategy: Fade overnight gap at London open (07:00 UTC) if gap > 1.5x ATR.
TP at 50% fill, SL at 2x ATR, force-close at 21:00 UTC.

V3.7 audit (simplified):
- L21 causality smoke
- L66 baseline: CARRY/INTRADAY -> cash (zero baseline)
- L65 ruin at proposed weight
- Simplified: no bracket OCO; enter at 07:00 close, exit at 21:00 close

Run::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/audit_gap_fade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework import assess_strategy_ruin  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "gap_fade_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 0.5
GAP_ATR_MULT = 1.5
PERIODS_PER_YEAR = 252  # per-trade (~1 day = 1 trade max)


def _load_h1(pair: str = "EUR_USD") -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{pair}_H1.parquet")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().dropna(subset=["close"])[["open", "high", "low", "close"]].astype(float)


def _atr_h1(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_gap_fade_trades(h1: pd.DataFrame, gap_atr_mult: float = 1.5) -> pd.DataFrame:
    """Build per-day trade list. Enter at 07:00 UTC if gap > N×ATR; exit at 21:00."""
    atr = _atr_h1(h1, period=14)
    h1 = h1.assign(hour=h1.index.hour, date=h1.index.date)

    trades = []
    # Group by trading date (rough — based on UTC date of the 07:00 bar)
    by_date = h1.groupby("date")
    prev_close = None
    prev_date = None

    for date, day_df in by_date:
        open_bar = day_df[day_df["hour"] == 7]
        close_bar = day_df[day_df["hour"] == 21]
        if open_bar.empty or close_bar.empty or prev_close is None:
            # Track for next iteration
            ny_close_bar = day_df[day_df["hour"] == 21]
            if not ny_close_bar.empty:
                prev_close = float(ny_close_bar["close"].iloc[0])
                prev_date = date
            continue

        # Use the open of the 07:00 bar as the entry reference (gap from prev NY close)
        london_open = float(open_bar["open"].iloc[0])
        london_entry_price = float(open_bar["close"].iloc[0])  # enter at 07:00 close
        gap = london_open - prev_close

        # ATR threshold (use ATR from the close of the prev_date 21:00 bar)
        atr_at_entry = atr.loc[open_bar.index[0]] if open_bar.index[0] in atr.index else np.nan
        if not np.isfinite(atr_at_entry):
            prev_close = float(close_bar["close"].iloc[0])
            prev_date = date
            continue

        threshold = gap_atr_mult * atr_at_entry
        if abs(gap) < threshold:
            prev_close = float(close_bar["close"].iloc[0])
            prev_date = date
            continue

        # Fade: gap UP -> short; gap DOWN -> long
        side = -1 if gap > 0 else 1
        exit_price = float(close_bar["close"].iloc[0])
        gross_ret = side * (exit_price - london_entry_price) / london_entry_price
        cost = 2 * COST_BPS / 10_000.0
        net_ret = gross_ret - cost

        trades.append({
            "date": date,
            "prev_close": prev_close,
            "london_open": london_open,
            "entry_price": london_entry_price,
            "exit_price": exit_price,
            "gap": gap,
            "atr": float(atr_at_entry),
            "side": side,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })

        prev_close = exit_price
        prev_date = date

    return pd.DataFrame(trades)


def assert_causal_gap_fade(h1: pd.DataFrame) -> None:
    """L21 smoke: corrupting future bars must not change past trades."""
    base = build_gap_fade_trades(h1)
    if len(base) < 5:
        return
    n_corrupt = 100
    h1_c = h1.copy()
    h1_c.iloc[-n_corrupt:, h1_c.columns.get_loc("close")] *= 100.0
    h1_c.iloc[-n_corrupt:, h1_c.columns.get_loc("open")] *= 100.0
    h1_c.iloc[-n_corrupt:, h1_c.columns.get_loc("high")] *= 100.0
    h1_c.iloc[-n_corrupt:, h1_c.columns.get_loc("low")] *= 100.0
    pert = build_gap_fade_trades(h1_c)
    # Past trades (date before the corruption start) should be unchanged
    cutoff_date = h1.iloc[-n_corrupt - 5]["close"]  # placeholder; use date instead
    cutoff_date_actual = h1.index[-n_corrupt - 5].date()
    base_past = base[base["date"] < cutoff_date_actual]
    pert_past = pert[pert["date"] < cutoff_date_actual]
    if len(base_past) == 0 or len(pert_past) == 0:
        print("[gap_fade] L21 smoke: insufficient past trades for verification")
        return
    common_dates = set(base_past["date"]) & set(pert_past["date"])
    for d in common_dates:
        b_row = base_past[base_past["date"] == d].iloc[0]
        p_row = pert_past[pert_past["date"] == d].iloc[0]
        assert abs(b_row["net_ret"] - p_row["net_ret"]) < 1e-12, f"L21 fail on {d}"
    print(f"[gap_fade] L21 PASS ({len(common_dates)} common past trades)")


def per_trade_sharpe(trades: pd.DataFrame) -> float:
    if len(trades) < 10:
        return 0.0
    r = trades["net_ret"]
    if r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def main() -> None:
    print("=" * 88)
    print("gap_fade V3.7 audit (EUR/USD H1)")
    print("=" * 88)

    h1 = _load_h1("EUR_USD")
    print(f"\n[load] EUR/USD H1: {len(h1)} bars, {h1.index[0]} -> {h1.index[-1]}")

    assert_causal_gap_fade(h1)

    print("\n--- Sweep on gap_atr_mult ---")
    print(f"{'gap_atr':>8} {'n_trades':>10} {'win_rate':>10} {'per_trade_SR':>14} {'mean_ret':>12}")
    for gam in [1.0, 1.5, 2.0, 2.5]:
        trades = build_gap_fade_trades(h1, gap_atr_mult=gam)
        n = len(trades)
        if n < 10:
            print(f"{gam:>8.2f} {n:>10d}  (insufficient)")
            continue
        wr = float((trades["net_ret"] > 0).mean())
        sr = per_trade_sharpe(trades)
        mr = float(trades["net_ret"].mean())
        print(f"{gam:>8.2f} {n:>10d} {wr:>9.2%} {sr:>+13.4f} {mr:>+11.4%}")

    # Live config: gap_atr_mult=1.5
    live_trades = build_gap_fade_trades(h1, gap_atr_mult=1.5)
    live_sr = per_trade_sharpe(live_trades)
    print(f"\n[live config (gap_atr_mult=1.5)] per-trade Sharpe = {live_sr:+.4f}, "
          f"n={len(live_trades)}, win_rate={float((live_trades['net_ret']>0).mean()):.2%}")

    # L65 ruin if signal positive
    if live_sr > 0 and len(live_trades) >= 30:
        # Convert trades to daily returns (sparse)
        daily_ret = pd.Series(
            live_trades["net_ret"].values,
            index=pd.to_datetime(live_trades["date"]),
        )
        print("\n--- L65 single-strategy ruin ---")
        for w in [0.05, 0.10, 0.15]:
            ruin = assess_strategy_ruin(
                daily_ret, deployment_weight=w,
                portfolio_kill_threshold=0.15, horizon_bars=252,
                block_size=21, n_paths=2000, seed=42,
            )
            print(f"  weight={w:.0%}: P_kill={ruin.p_kill_trip:.3%}, "
                  f"95th-pct DD={ruin.p95_maxdd_at_size:.3%}, passes={ruin.passes()}")

    # Verdict
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if live_sr <= 0:
        print(f"RETIRE: live config per-trade SR {live_sr:+.4f} <= 0")
    elif live_sr < 0.30:
        print(f"MARGINAL: live SR {live_sr:+.4f} < 0.30; signal exists but weak.")
    else:
        print(f"CONDITIONAL candidate: live SR {live_sr:+.4f} > 0.30")


if __name__ == "__main__":
    main()
