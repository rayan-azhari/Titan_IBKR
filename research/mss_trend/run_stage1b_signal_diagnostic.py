"""Stage 1b — Entry signal diagnostic (Option A).

Strips away the 2R-TP / swing-SL exits. For each entry trigger, records the
direction-signed forward return at fixed horizons (1, 4, 12, 24, 48 H1 bars).

If the entry signal has any directional edge:
   mean(forward_return_signed) > 0 with CI lower bound > 0 at SOME horizon.

If forward returns are statistically indistinguishable from zero across all
horizons, the entry signal has no edge and no exit-rule tuning will rescue
it. (This is the cleanest possible test — no exits to confound results.)

Baseline comparison: random bar entries with random direction sampled from
the same data, matched in count.

Run with: PYTHONUTF8=1 uv run python research/mss_trend/run_stage1b_signal_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.mss_trend.swing import detect_swings, trend_state_series  # noqa: E402
from research.mss_trend.strategy import daily_trend_at_15m  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "USD_CHF", "USD_JPY"]
HORIZONS = [1, 4, 12, 24, 48]  # H1 bars: 1h, 4h, 12h, 1d, 2d


def load_pair(pair: str):
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


def collect_entries(
    h1: pd.DataFrame, daily: pd.DataFrame,
    daily_n: int = 2, m15_n: int = 6,
) -> list[tuple[int, int]]:
    """Return list of (entry_bar_idx, direction) for every MSS trigger.

    entry_bar_idx is the bar AT WHICH the trade would enter (i.e., next bar
    after the trigger close), so forward returns are well-defined.
    """
    daily_swings = detect_swings(daily["high"], daily["low"], n=daily_n)
    daily_trend = trend_state_series(daily_swings, n_bars=len(daily))
    daily_trend.index = daily.index
    h1_trend = daily_trend_at_15m(daily, daily_trend, h1.index, daily_n)

    h1_swings = detect_swings(h1["high"], h1["low"], n=m15_n)
    swings_by_confirm = sorted(h1_swings, key=lambda s: s.confirm_idx)

    closes = h1["close"].to_numpy()
    n_bars = len(h1)
    trend_arr = h1_trend.to_numpy()

    cursor = 0
    confirmed_highs = []
    confirmed_lows = []
    entries: list[tuple[int, int]] = []
    last_entry_bar = -1
    cooldown = 24  # require 1 day between entries to dedupe clustering

    for i in range(n_bars):
        while cursor < len(swings_by_confirm) and swings_by_confirm[cursor].confirm_idx <= i:
            sw = swings_by_confirm[cursor]
            if sw.is_high:
                confirmed_highs.append(sw)
            else:
                confirmed_lows.append(sw)
            cursor += 1
        if i - last_entry_bar < cooldown:
            continue
        if i + 1 >= n_bars:
            break

        td = trend_arr[i]
        c = closes[i]
        if td == 1 and confirmed_highs:
            recent_h = confirmed_highs[-1]
            if c > recent_h.price:
                entries.append((i + 1, +1))
                last_entry_bar = i
                confirmed_highs = []
        elif td == -1 and confirmed_lows:
            recent_l = confirmed_lows[-1]
            if c < recent_l.price:
                entries.append((i + 1, -1))
                last_entry_bar = i
                confirmed_lows = []
    return entries


def forward_returns(closes: np.ndarray, entry_idx: int, horizon: int,
                    direction: int) -> float | None:
    if entry_idx + horizon >= len(closes):
        return None
    p0 = closes[entry_idx]
    p1 = closes[entry_idx + horizon]
    if p0 <= 0 or p1 <= 0:
        return None
    raw = (p1 / p0 - 1.0)
    return direction * raw


def random_baseline(closes: np.ndarray, n_samples: int, horizon: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample n_samples random (idx, direction) pairs and compute signed
    forward returns. Same data; same horizon; random selection."""
    valid_max = len(closes) - horizon - 1
    if valid_max <= 0:
        return np.array([])
    idxs = rng.integers(low=0, high=valid_max, size=n_samples)
    dirs = rng.choice([-1, 1], size=n_samples)
    out = np.empty(n_samples)
    for i, (idx, d) in enumerate(zip(idxs, dirs)):
        out[i] = d * (closes[idx + horizon] / closes[idx] - 1.0)
    return out


def bootstrap_mean_ci(
    x: np.ndarray, n_resamples: int = 5000, confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    if len(x) < 10:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (float(x.mean()), lo, hi)


def main() -> None:
    print("=" * 84)
    print("Stage 1b — Entry signal diagnostic — fixed-horizon forward returns")
    print("=" * 84)
    print("\nPer-trigger signed forward return = direction * (close[t+h]/close[t] - 1)")
    print("If signal has alpha: mean > 0 with CI lo > 0 at some horizon.\n")

    rng = np.random.default_rng(42)

    all_entries_by_horizon: dict[int, list[float]] = {h: [] for h in HORIZONS}
    all_baseline_by_horizon: dict[int, list[float]] = {h: [] for h in HORIZONS}

    for pair in PAIRS:
        print(f"  Processing {pair}...", end=" ")
        h1, d = load_pair(pair)
        entries = collect_entries(h1, d, daily_n=2, m15_n=6)
        closes = h1["close"].to_numpy()
        n_entries = len(entries)
        print(f"{n_entries} entries")

        for h in HORIZONS:
            for (idx, direction) in entries:
                r = forward_returns(closes, idx, h, direction)
                if r is not None:
                    all_entries_by_horizon[h].append(r)
            baseline = random_baseline(closes, n_samples=n_entries * 5,
                                       horizon=h, rng=rng)
            all_baseline_by_horizon[h].extend(baseline.tolist())

    print("\n" + "=" * 84)
    print("POOLED RESULTS (all 5 pairs, all entries)")
    print("=" * 84)
    print(f"\n{'Horizon':>10}  {'N entry':>9}  "
          f"{'Mean entry (bps)':>18}  {'Entry 95% CI (bps)':>26}  "
          f"{'Mean baseline (bps)':>21}  {'p(better)':>11}")
    print("-" * 100)

    for h in HORIZONS:
        e = np.array(all_entries_by_horizon[h])
        b = np.array(all_baseline_by_horizon[h])
        if len(e) < 10 or len(b) < 10:
            continue
        e_mean, e_lo, e_hi = bootstrap_mean_ci(e)
        b_mean, b_lo, b_hi = bootstrap_mean_ci(b)
        # One-sided check: p(entry mean > baseline mean) via bootstrap
        rng2 = np.random.default_rng(123)
        n_b = 5000
        wins = 0
        for _ in range(n_b):
            er = rng2.choice(e, size=len(e), replace=True).mean()
            br = rng2.choice(b, size=min(len(b), 10000), replace=True).mean()
            if er > br:
                wins += 1
        p_better = wins / n_b
        bps = lambda x: x * 1e4
        print(f"{h:>9}h  {len(e):>9d}  "
              f"{bps(e_mean):>+17.2f}   "
              f"[{bps(e_lo):>+6.2f}, {bps(e_hi):>+6.2f}]   "
              f"{bps(b_mean):>+19.2f}   "
              f"{p_better:>10.1%}")

    print()
    print("=" * 84)
    print("INTERPRETATION")
    print("=" * 84)
    print("  - If 'Entry 95% CI lo' > 0  at any horizon  → entry signal has alpha.")
    print("  - If all CI's straddle zero AND p(better) < 70% across the board → no edge.")
    print("  - p(better) is a one-sided 'entry beats random' probability test.")


if __name__ == "__main__":
    main()
