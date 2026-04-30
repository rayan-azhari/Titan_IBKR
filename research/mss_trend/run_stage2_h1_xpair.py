"""Stage 2 — Cross-pair H1 OOS test on all 5 FX pairs.

The creator's spec says 15M-entry is best and H1-entry is "poor returns".
We pivot to H1 because:
  (a) we have continuous H1 data 2016-2026 for all five pairs, and
  (b) if the structural idea (Daily trend + intraday MSS entry) has
      genuine edge, it should still show *some* signal at H1 — not the
      headline 957% but a non-zero positive Sharpe.

Methodology:
  * Single 80/20 IS/OOS split per pair (no in-sample tuning, frozen
    parameters: daily_n=2, m15_n=6, tp_r_multiple=2.0).
  * Last 12 months reserved as sanctuary (untouched).
  * Per-pair Sharpe + bootstrap CI.
  * Pooled Sharpe across pairs (equal-weighted return series).
  * Gate: pooled OOS Sharpe > 0.5 AND ci_lo > 0.

Run with: PYTHONUTF8=1 uv run python research/mss_trend/run_stage2_h1_xpair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.mss_trend.strategy import (  # noqa: E402
    generate_trades,
    trades_to_bar_returns,
)
from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
H1_PER_YEAR = 24 * 252  # 6,048

PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "USD_CHF", "USD_JPY"]


def load_pair(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    h1 = pd.read_parquet(DATA_DIR / f"{pair}_H1.parquet")
    if "timestamp" in h1.columns:
        h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
        h1 = h1.set_index("timestamp").sort_index()
    else:
        h1.index = pd.to_datetime(h1.index, utc=True)
        h1 = h1.sort_index()
    h1 = h1[["open", "high", "low", "close", "volume"]]

    d = pd.read_parquet(DATA_DIR / f"{pair}_D.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp").sort_index()
    d = d[["open", "high", "low", "close", "volume"]]
    return h1, d


def run_pair(pair: str, oos_start: pd.Timestamp, sanctuary_start: pd.Timestamp) -> dict:
    h1, daily = load_pair(pair)

    # Trim sanctuary
    h1_in_scope = h1[h1.index < sanctuary_start]
    daily_in_scope = daily[daily.index < sanctuary_start]

    trades = generate_trades(
        m15_df=h1_in_scope,
        daily_df=daily_in_scope,
        daily_n=2,
        m15_n=6,
        tp_r_multiple=2.0,
    )
    bar_rets = trades_to_bar_returns(
        trades,
        bar_index=h1_in_scope.index,
        risk_per_trade=0.01,
        cost_per_trade=5e-5,
    )

    is_mask = h1_in_scope.index < oos_start
    oos_mask = h1_in_scope.index >= oos_start
    is_trades = [t for t in trades if t.entry_ts < oos_start]
    oos_trades = [t for t in trades if t.entry_ts >= oos_start]

    def _block(rets, n_trades, label):
        sh = sharpe(rets, periods_per_year=H1_PER_YEAR)
        lo, hi = bootstrap_sharpe_ci(rets, periods_per_year=H1_PER_YEAR, n_resamples=2000)
        dd = max_drawdown(rets)
        wins = sum(1 for t in (oos_trades if "OOS" in label else is_trades) if t.r_multiple > 0)
        wr = wins / max(n_trades, 1)
        return {
            "label": label,
            "sharpe": sh,
            "ci_lo": lo,
            "ci_hi": hi,
            "max_dd": dd,
            "n_trades": n_trades,
            "win_rate": wr,
        }

    is_block = _block(bar_rets[is_mask], len(is_trades), "IS")
    oos_block = _block(bar_rets[oos_mask], len(oos_trades), "OOS")

    return {
        "pair": pair,
        "is": is_block,
        "oos": oos_block,
        "n_trades_total": len(trades),
        "oos_returns": bar_rets[oos_mask],
    }


def main() -> None:
    print("=" * 78)
    print("Stage 2 — MSS Trend Strategy — Cross-pair H1 OOS test")
    print("=" * 78)

    # Determine common date range first
    common_start = pd.Timestamp("2016-03-17", tz="UTC")
    common_end = pd.Timestamp("2026-03-13", tz="UTC")
    sanctuary_start = common_end - pd.DateOffset(years=1)
    span_yrs = (sanctuary_start - common_start).days / 365.25
    oos_start = common_start + pd.Timedelta(days=int(span_yrs * 365.25 * 0.7))
    print(f"\n  IS         : {common_start.date()}  →  {oos_start.date()}")
    print(f"  OOS        : {oos_start.date()}  →  {sanctuary_start.date()}")
    print(f"  Sanctuary  : {sanctuary_start.date()}  →  {common_end.date()}  (UNTOUCHED)")

    results = []
    print(
        f"\n{'Pair':<10}{'IS Sh':>8}{'IS CI':>16}{'IS DD':>10}{'IS#':>6}"
        f"  | {'OOS Sh':>8}{'OOS CI':>16}{'OOS DD':>10}{'OOS#':>6}{'OOS WR':>8}"
    )
    print("-" * 110)
    for pair in PAIRS:
        r = run_pair(pair, oos_start, sanctuary_start)
        results.append(r)
        is_b, oos_b = r["is"], r["oos"]
        print(
            f"{pair:<10}"
            f"{is_b['sharpe']:>+8.2f}"
            f"  [{is_b['ci_lo']:>+5.2f},{is_b['ci_hi']:>+5.2f}]"
            f"{is_b['max_dd']:>10.1%}{is_b['n_trades']:>6d}  | "
            f"{oos_b['sharpe']:>+8.2f}"
            f"  [{oos_b['ci_lo']:>+5.2f},{oos_b['ci_hi']:>+5.2f}]"
            f"{oos_b['max_dd']:>10.1%}{oos_b['n_trades']:>6d}"
            f"{oos_b['win_rate']:>7.1%}"
        )

    # Pooled OOS: average per-bar return series across pairs (equal-weight)
    pooled = pd.concat([r["oos_returns"] for r in results], axis=1).fillna(0.0)
    pooled_rets = pooled.mean(axis=1)
    pooled_sh = sharpe(pooled_rets, periods_per_year=H1_PER_YEAR)
    pooled_lo, pooled_hi = bootstrap_sharpe_ci(
        pooled_rets, periods_per_year=H1_PER_YEAR, n_resamples=2000
    )
    pooled_dd = max_drawdown(pooled_rets)
    pooled_total = float((1.0 + pooled_rets).prod() - 1.0)
    n_oos_trades = sum(r["oos"]["n_trades"] for r in results)
    n_oos_wins = sum(r["oos"]["n_trades"] * r["oos"]["win_rate"] for r in results)
    pooled_wr = n_oos_wins / max(n_oos_trades, 1)

    print("\n" + "=" * 78)
    print("POOLED (equal-weight across 5 pairs)")
    print("=" * 78)
    print(f"  OOS Sharpe        : {pooled_sh:>+.3f}")
    print(f"  OOS 95% CI        : [{pooled_lo:>+.3f}, {pooled_hi:>+.3f}]")
    print(f"  OOS max drawdown  : {pooled_dd:>.2%}")
    print(f"  OOS total return  : {pooled_total:>.2%}")
    print(f"  OOS total trades  : {n_oos_trades}")
    print(f"  OOS pooled WR     : {pooled_wr:.1%}")

    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    pass_sh = pooled_sh > 0.5
    pass_ci = pooled_lo > 0
    if pass_sh and pass_ci:
        print("  PASS  → consider Stage 3 (portfolio correlation analysis)")
    else:
        print("  FAIL  → tier=unconfirmed, do not advance to portfolio.")
        if not pass_sh:
            print(f"          Pooled OOS Sharpe {pooled_sh:.3f} <= 0.5 gate")
        if not pass_ci:
            print(f"          95% lower bound {pooled_lo:.3f} <= 0")
    print("=" * 78)


if __name__ == "__main__":
    main()
