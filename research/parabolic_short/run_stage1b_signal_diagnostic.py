"""Stage 1b — Signal diagnostic for the Parabolic Short setup.

For each day t that meets the SETUP filter (3 greens + gap + volume + extension),
record the forward return at multiple horizons (1d, 3d, 5d, 10d, 20d), regardless
of whether the trigger (red close) fired. Compare to a random-baseline drawn from
the same name and time window.

This isolates the SETUP edge from the TRIGGER edge:
  - If forward returns from setups are statistically negative → setup picks weak
    bars → strategy has structural edge that sample size is hiding.
  - If forward returns are zero / random → setup has no alpha; trigger and
    exits are just noise around a dead signal.

Run with: PYTHONUTF8=1 uv run python research/parabolic_short/run_stage1b_signal_diagnostic.py
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

from research.parabolic_short.strategy import (  # noqa: E402
    EXT_PCT,
    GAP_PCT,
    SMA_LEN,
    VOL_MULT,
)

DATA_DIR = PROJECT_ROOT / "data"
HORIZONS = [1, 3, 5, 10, 20]


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


def detect_setups_only(df: pd.DataFrame) -> pd.Series:
    """Setup WITHOUT the red-close trigger. Returns boolean Series."""
    o, c, v = df["open"], df["close"], df["volume"]
    sma10 = c.rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    avg_v20 = v.rolling(20, min_periods=20).mean()
    green = (c > o)
    three_green = green.shift(1) & green.shift(2) & green.shift(3)
    gap_up = (o > c.shift(1) * (1.0 + GAP_PCT))
    vol_blowoff = (v > avg_v20 * VOL_MULT)
    extended = (c.shift(1) > sma10.shift(1) * (1.0 + EXT_PCT))
    return (three_green & gap_up & vol_blowoff & extended).fillna(False)


def bootstrap_mean_ci(x: np.ndarray, n_resamples: int = 5000,
                      confidence: float = 0.95, seed: int = 42):
    if len(x) < 10:
        return (np.nan, np.nan, np.nan)
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
    print("Stage 1b — Parabolic Short SETUP signal diagnostic")
    print("=" * 84)
    print("\nMeasures forward returns FROM the setup day's close, regardless of trigger.")
    print("If setup has alpha (mean-reversion bias): forward returns < 0 with CI hi < 0.")
    print("Baseline: random bars from the same name's history.\n")

    files = list_equity_files()
    rng = np.random.default_rng(42)

    setup_rets_by_h: dict[int, list[float]] = {h: [] for h in HORIZONS}
    base_rets_by_h: dict[int, list[float]] = {h: [] for h in HORIZONS}
    n_total_setups = 0
    n_symbols_with_setups = 0

    for p in files:
        loaded = load_one(p)
        if loaded is None:
            continue
        _, df = loaded
        setups = detect_setups_only(df)
        if not setups.any():
            continue
        n_symbols_with_setups += 1
        n_total_setups += int(setups.sum())

        closes = df["close"].to_numpy()
        n_bars = len(df)

        setup_idxs = np.where(setups.to_numpy())[0]
        for h in HORIZONS:
            for i in setup_idxs:
                if i + h >= n_bars:
                    continue
                p0, pf = closes[i], closes[i + h]
                if p0 > 0 and pf > 0:
                    # signed for SHORT direction (negative forward = profit)
                    setup_rets_by_h[h].append(-(pf / p0 - 1.0))
            # Random baseline: 5 random bars per setup
            n_base = len(setup_idxs) * 5
            valid_max = n_bars - h - 1
            if valid_max <= 0:
                continue
            samp = rng.integers(0, valid_max, size=n_base)
            for j in samp:
                p0, pf = closes[j], closes[j + h]
                if p0 > 0 and pf > 0:
                    base_rets_by_h[h].append(-(pf / p0 - 1.0))

    print(f"  Symbols with >=1 setup : {n_symbols_with_setups}")
    print(f"  Total setup events     : {n_total_setups}")
    print()
    print(f"{'Horiz':>6}  {'N setups':>9}  {'Setup mean (bps)':>18}  "
          f"{'Setup 95% CI (bps)':>26}  {'Baseline mean (bps)':>21}  {'p(setup beats)':>15}")
    print("-" * 100)
    for h in HORIZONS:
        s = np.array(setup_rets_by_h[h])
        b = np.array(base_rets_by_h[h])
        if len(s) < 10 or len(b) < 10:
            continue
        s_mean, s_lo, s_hi = bootstrap_mean_ci(s)
        b_mean, _, _ = bootstrap_mean_ci(b)
        rng2 = np.random.default_rng(7)
        wins = 0
        n_b = 5000
        for _ in range(n_b):
            er = rng2.choice(s, size=len(s), replace=True).mean()
            br = rng2.choice(b, size=min(len(b), 10000), replace=True).mean()
            if er > br:
                wins += 1
        p_better = wins / n_b
        bps = lambda x: x * 1e4
        print(f"{h:>5}d  {len(s):>9d}  "
              f"{bps(s_mean):>+17.2f}   "
              f"[{bps(s_lo):>+6.2f}, {bps(s_hi):>+6.2f}]   "
              f"{bps(b_mean):>+19.2f}   "
              f"{p_better:>14.1%}")

    print()
    print("=" * 84)
    print("INTERPRETATION (short-direction, so positive = mean-reversion edge):")
    print("=" * 84)
    print("  - Setup mean > 0 with CI lo > 0 at any horizon → setup picks bars that")
    print("    on average decline (alpha for short side).")
    print("  - All CIs straddle zero AND p(setup beats) ~ 50% → no setup-level edge.")


if __name__ == "__main__":
    main()
