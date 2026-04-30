"""Stage 1c — Parabolic Short v2 sanctuary directional check.

Runs strategy_v2 (3-day exit, no red-close trigger) on the SANCTUARY window
only (2025-01-01 → present). This window has not been touched in any prior
analysis; the v2 exit rule was frozen based on the Stage 1b diagnostic (which
saw IS+OOS data through 2024-12-31).

With ~13 expected trades, this is a low-power directional falsification check:
  - Strongly negative result → kill the strategy
  - Modestly positive / null → the diagnostic finding is not falsified;
    proceed to small-cap data sourcing for a real test
  - Strongly positive → encouraging but still underpowered; proceed to
    small-cap with extra confidence

Run with: PYTHONUTF8=1 uv run python research/parabolic_short/run_stage1c_sanctuary.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.metrics import (  # noqa: E402
    bootstrap_sharpe_ci,
    sharpe,
    trade_sharpe,
)

from research.parabolic_short.strategy_v2 import (  # noqa: E402
    TradeV2,
    simulate_trades_v2,
)

DATA_DIR = PROJECT_ROOT / "data"
DAILY_PER_YEAR = 252
SANCTUARY_START = pd.Timestamp("2025-01-01", tz="UTC")


def list_equity_files() -> list[Path]:
    files = sorted(glob.glob(str(DATA_DIR / "*_D.parquet")))
    fx_prefixes = ("EUR_", "GBP_", "AUD_", "USD_", "NZD_", "CHF_", "JPY_")
    return [Path(f) for f in files
            if not any(os.path.basename(f).startswith(p) for p in fx_prefixes)]


def load_one(path: Path):
    sym = path.name.replace("_D.parquet", "")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index(ts).drop(columns=["timestamp"]).sort_index()
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(set(df.columns)):
        return None
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < 50:
        return None
    return sym, df


def bootstrap_mean_ci(x: np.ndarray, n_resamples: int = 5000,
                      confidence: float = 0.95, seed: int = 42):
    if len(x) < 5:
        return (float(np.mean(x)) if len(x) > 0 else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    alpha = 1.0 - confidence
    return (float(x.mean()),
            float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def main() -> None:
    print("=" * 84)
    print("Stage 1c — Parabolic Short v2 (3d exit) — SANCTUARY DIRECTIONAL CHECK")
    print("=" * 84)
    print(f"\n  Sanctuary window: {SANCTUARY_START.date()} → present")
    print("  Frozen rules: 3-day exit, no red-close trigger, stop = parabolic high")
    print("  Power note: ~1.3y window, expect ~13 trades. CI will be wide.\n")

    files = list_equity_files()
    all_trades: list[TradeV2] = []
    pre_sanctuary_trades: list[TradeV2] = []  # for context

    for p in files:
        loaded = load_one(p)
        if loaded is None:
            continue
        _, df = loaded
        # Run v2 simulation on full data, then filter trades by entry date.
        # Setups close to sanctuary boundary that span across need consistent
        # entry-date filtering — we keep only trades whose ENTRY date is in
        # the sanctuary window, regardless of where the data ends.
        trades = simulate_trades_v2(df, symbol=df.index.name or "_")
        # Replace symbol attribute (dataclass is frozen so rebuild)
        sym = p.name.replace("_D.parquet", "")
        trades = [TradeV2(
            symbol=sym, setup_date=t.setup_date, entry_date=t.entry_date,
            entry_price=t.entry_price, stop_price=t.stop_price,
            exit_date=t.exit_date, exit_price=t.exit_price,
            exit_reason=t.exit_reason, r_multiple=t.r_multiple,
            return_pct=t.return_pct,
        ) for t in trades]

        for t in trades:
            if t.entry_date >= SANCTUARY_START:
                all_trades.append(t)
            else:
                pre_sanctuary_trades.append(t)

    print(f"  Pre-sanctuary trades  (context only): {len(pre_sanctuary_trades)}")
    print(f"  SANCTUARY trades                    : {len(all_trades)}")

    if not all_trades:
        print("\n  NO SANCTUARY TRADES — strategy did not trigger in 2025+.")
        print("  Result: inconclusive. The setup is too rare to fire in 1.3y.")
        return

    # Trade-level summary
    rs = np.array([t.r_multiple for t in all_trades])
    rets = np.array([t.return_pct for t in all_trades])
    wins = (rs > 0).sum()
    print(f"\n  Trade R-multiple stats (sanctuary):")
    print(f"    mean R         : {rs.mean():+.3f}")
    print(f"    median R       : {np.median(rs):+.3f}")
    print(f"    win rate       : {wins / len(rs):.1%}  ({wins}/{len(rs)})")
    print(f"    p95 R          : {np.quantile(rs, 0.95):+.3f}")
    print(f"    p05 R          : {np.quantile(rs, 0.05):+.3f}")
    er = {"sl": 0, "time": 0}
    for t in all_trades:
        er[t.exit_reason] = er.get(t.exit_reason, 0) + 1
    print(f"    exit mix       : sl={er['sl']}  time={er['time']}")

    # Bootstrap CI on mean R
    mean_r, lo_r, hi_r = bootstrap_mean_ci(rs, n_resamples=5000)
    print(f"\n  Mean R bootstrap 95% CI: [{lo_r:+.3f}, {hi_r:+.3f}]")

    # Build per-day return series for Sharpe calculation
    by_day: dict[pd.Timestamp, list[float]] = {}
    for t in all_trades:
        d = pd.Timestamp(t.exit_date).normalize()
        by_day.setdefault(d, []).append(t.r_multiple * 0.01)

    # Build calendar of trading dates in sanctuary window
    cal: set[pd.Timestamp] = set()
    for p in files:
        loaded = load_one(p)
        if loaded is None:
            continue
        _, df = loaded
        df_san = df[df.index >= SANCTUARY_START]
        cal.update(pd.Timestamp(x).normalize() for x in df_san.index)
    master_idx = pd.DatetimeIndex(sorted(cal))
    pooled = pd.Series(0.0, index=master_idx)
    for d, vs in by_day.items():
        if d in pooled.index:
            pooled.loc[d] = float(np.mean(vs))

    sh = sharpe(pooled, periods_per_year=DAILY_PER_YEAR)
    sh_lo, sh_hi = bootstrap_sharpe_ci(pooled, periods_per_year=DAILY_PER_YEAR,
                                        n_resamples=2000)
    yrs = len(pooled) / DAILY_PER_YEAR if DAILY_PER_YEAR else 0
    if yrs > 0 and len(rs) >= 5:
        t_sh = trade_sharpe(rs, trades_per_year=len(rs) / yrs)
    else:
        t_sh = 0.0
    total = float((1.0 + pooled).prod() - 1.0)

    print(f"\n  Sanctuary span        : {yrs:.2f}y ({len(pooled)} trading days)")
    print(f"  Sharpe (per-day, ann.): {sh:+.3f}  CI [{sh_lo:+.3f}, {sh_hi:+.3f}]")
    print(f"  Sharpe (per-trade)    : {t_sh:+.3f}")
    print(f"  Total return          : {total:+.2%}")

    # Directional check (NOT a strict gate due to power)
    print("\n" + "=" * 84)
    print("DIRECTIONAL VERDICT")
    print("=" * 84)

    if rs.mean() > 0 and lo_r > -0.3:
        print("  POSITIVE — sanctuary is consistent with the diagnostic finding.")
        print("  Mean R is positive and the lower CI bound isn't catastrophic.")
        print("  → Recommend Path 3: source small-cap data and re-test on")
        print("    Hanlin's actual universe.")
    elif rs.mean() > -0.1:
        print("  NULL — sanctuary is roughly flat; underpowered to confirm")
        print("  or refute. Equally consistent with 'real but small edge' or")
        print("  'no edge' given the trade count.")
        print("  → Path 3 is optional; expected value is low without small-cap data.")
    else:
        print("  NEGATIVE — sanctuary contradicts the diagnostic.")
        print("  Mean R is materially negative; the v2 rules don't generalize.")
        print("  → Kill the strategy.")
    print("=" * 84)


if __name__ == "__main__":
    main()
