"""build_tbm_labels.py — Triple Barrier Method labeling for EUR/USD H1.

Generates path-dependent labels for each bar by looking forward up to
`max_holding` bars and determining which barrier is hit first:
  +1 = upper barrier (profit take) hit first → long win
  -1 = lower barrier (stop loss) hit first  → long loss
   0 = vertical barrier (time out)           → inconclusive

Uses high/low to detect barrier touches (not just close), which is more
realistic and consistent with how NautilusTrader bracket orders work.

Numba @njit kernel for performance on 134K+ row EUR/USD history.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / ".tmp" / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Barrier parameters ─────────────────────────────────────────────────────
PT_MULT = 2.0  # Profit-take = entry + ATR * PT_MULT
SL_MULT = 1.0  # Stop-loss   = entry - ATR * SL_MULT  (asymmetric: 2:1 RR)
MAX_HOLDING = 24  # Vertical barrier: 24 hours on H1 data
ATR_PERIOD = 14


# ── Numba kernel ───────────────────────────────────────────────────────────


@njit(cache=True)
def _tbm_kernel(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
) -> np.ndarray:
    """Triple Barrier Method — returns label array aligned with close.

    Args:
        close: Close price array.
        high: High price array.
        low: Low price array.
        atr: ATR array (NaN during warm-up).
        pt_mult: Profit-take ATR multiplier.
        sl_mult: Stop-loss ATR multiplier.
        max_holding: Maximum bars to hold (vertical barrier).

    Returns:
        int8 array: +1 upper hit, -1 lower hit, 0 time-out or NaN ATR.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)

    for t in range(n):
        if np.isnan(atr[t]) or atr[t] <= 0.0:
            continue  # warm-up bars stay 0

        pt = close[t] + atr[t] * pt_mult  # upper barrier (absolute price)
        sl = close[t] - atr[t] * sl_mult  # lower barrier (absolute price)

        for i in range(1, max_holding + 1):
            idx = t + i
            if idx >= n:
                break

            hit_upper = high[idx] >= pt
            hit_lower = low[idx] <= sl

            if hit_upper and hit_lower:
                # Both barriers touched in the same bar.
                # Award to whichever is closer to entry in price terms.
                if (pt - close[t]) <= (close[t] - sl):
                    labels[t] = 1
                else:
                    labels[t] = -1
                break
            elif hit_upper:
                labels[t] = 1
                break
            elif hit_lower:
                labels[t] = -1
                break
        # If no break → vertical barrier → labels[t] stays 0

    return labels


# ── ATR helper (pure numpy, no pandas dependency in kernel) ────────────────


def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ATR using Wilder's smoothing (same as NautilusTrader ATR indicator)."""
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n = len(close)

    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Wilder smoothing (equivalent to EMA with alpha=1/period)
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ── Main ───────────────────────────────────────────────────────────────────


def build_tbm_labels(
    pair: str = "EUR_USD",
    gran: str = "H1",
    pt_mult: float = PT_MULT,
    sl_mult: float = SL_MULT,
    max_holding: int = MAX_HOLDING,
) -> pd.DataFrame:
    """Build TBM labels for a given pair and granularity.

    Args:
        pair: Currency pair code (e.g., "EUR_USD").
        gran: Bar granularity (e.g., "H1").
        pt_mult: Profit-take multiplier.
        sl_mult: Stop-loss multiplier.
        max_holding: Vertical barrier in bars.

    Returns:
        DataFrame with timestamp index and 'tbm_label' + 'atr' columns.
    """
    data_path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    print(f"  Loaded {len(df):,} bars for {pair} {gran}")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")

    atr_arr = _compute_atr(df, ATR_PERIOD)
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values

    print(
        f"  Computing TBM labels (pt={pt_mult}x ATR, sl={sl_mult}x ATR, max_hold={max_holding}h)..."
    )
    labels = _tbm_kernel(close_arr, high_arr, low_arr, atr_arr, pt_mult, sl_mult, max_holding)

    result = pd.DataFrame(
        {"tbm_label": labels, "atr": atr_arr},
        index=df.index,
    )

    # Statistics
    n_total = len(result)
    n_warmup = int(np.isnan(atr_arr).sum())
    n_valid = n_total - n_warmup
    n_up = int((labels == 1).sum())
    n_down = int((labels == -1).sum())
    n_time = int((labels == 0).sum()) - n_warmup

    print(f"\n  Label distribution ({n_valid:,} valid bars, excl. {n_warmup} warm-up):")
    print(f"    +1 Upper (TP hit):  {n_up:6,}  ({n_up / n_valid * 100:.1f}%)")
    print(f"    -1 Lower (SL hit):  {n_down:6,}  ({n_down / n_valid * 100:.1f}%)")
    print(f"     0 Time-out:        {n_time:6,}  ({n_time / n_valid * 100:.1f}%)")

    return result


def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("Triple Barrier Method — Label Generation")
    print(f"Parameters: PT={PT_MULT}x ATR, SL={SL_MULT}x ATR, Max Hold={MAX_HOLDING}h")
    print("=" * 60)

    pair, gran = "EUR_USD", "H1"
    result = build_tbm_labels(pair, gran, PT_MULT, SL_MULT, MAX_HOLDING)

    out_path = OUTPUT_DIR / f"{pair}_{gran}_tbm_labels.parquet"
    result.to_parquet(out_path)
    print(f"\n  Saved TBM labels: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
