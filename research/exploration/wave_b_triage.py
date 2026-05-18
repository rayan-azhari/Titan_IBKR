"""Wave B P2 strategies — quick L58-style signal-layer triage.

Runs a minimal pure-research signal-layer reproduction for each of the
simplest 3 Wave B strategies (fx_carry, gold_macro, turtle) at their live
configs. Reports stitched Sharpe + CI_lo. Verdict criterion: if signal-
layer Sharpe is negative with CI_lo well below 0, recommend de-allocation
WITHOUT a full L52 hybrid audit (mirrors L58 economic logic).

The other 5 Wave B strategies (orb, gld_confluence, pairs, ic_equity_daily,
ic_mtf) are documented separately due to special mechanics that don't fit
the simple signal-layer abstraction (sparse trades, multi-instrument
cointegration, multi-TF causality risk).

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/exploration/wave_b_triage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    max_drawdown,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "wave_b_triage"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_parquet(name: str, *, date_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / name)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df.set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


# ─────────────────────────────────────────────────────────────────────
# fx_carry — daily AUD/JPY long when close > SMA(50), vol-targeted
# ─────────────────────────────────────────────────────────────────────


def fx_carry_signal(
    closes: pd.DataFrame, *, sma_period: int = 50, vol_target: float = 0.08
) -> pd.Series:
    """Pure-research fx_carry: long when close > SMA(sma_period); flat else.
    Vol-target sizing via 20-day EWMA realised vol. 1.5 bps cost."""
    close = closes["close"]
    sma = close.rolling(sma_period, min_periods=sma_period).mean()
    signal = (close > sma).astype(float)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    var = log_ret.pow(2).ewm(span=20, adjust=False, min_periods=20).mean()
    realised_vol = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (vol_target / realised_vol.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)
    position = signal * scale

    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (1.5 / 10_000.0)
    return (gross - cost).rename("ret")


# ─────────────────────────────────────────────────────────────────────
# gold_macro — daily GLD long when close > SMA(200) + macro composite
# ─────────────────────────────────────────────────────────────────────


def gold_macro_signal(
    closes: pd.DataFrame, *, slow_ma: int = 200, vol_target: float = 0.10
) -> pd.Series:
    """Pure-research gold_macro signal-layer: long GLD when close > SMA(200).
    The full live config also requires positive real_rate + dollar composite;
    we test the bare SMA filter — if that's not positive, the composite adds
    little. 1.5 bps cost."""
    close = closes["close"]
    sma = close.rolling(slow_ma, min_periods=slow_ma).mean()
    signal = (close > sma).astype(float)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    var = log_ret.pow(2).ewm(span=20, adjust=False, min_periods=20).mean()
    realised_vol = np.sqrt(var * BARS_PER_YEAR["D"])
    scale = (vol_target / realised_vol.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)
    position = signal * scale

    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (1.5 / 10_000.0)
    return (gross - cost).rename("ret")


# ─────────────────────────────────────────────────────────────────────
# turtle — H1 Donchian breakout on CAT
# ─────────────────────────────────────────────────────────────────────


def turtle_signal(
    closes: pd.DataFrame,
    *,
    entry_period: int = 45,
    exit_period: int = 30,
    atr_period: int = 20,
) -> pd.Series:
    """Pure-research turtle: enter long on N-bar Donchian high breakout;
    exit on M-bar Donchian low. ATR sizing → here normalised to 1 unit per
    breakout (binary signal). 1 bp cost (H1 frequency, equity)."""
    if "high" not in closes.columns or "low" not in closes.columns:
        # If only close available, use that for both.
        high = closes["close"]
        low = closes["close"]
    else:
        high = closes["high"]
        low = closes["low"]
    close = closes["close"]

    donch_high = high.rolling(entry_period, min_periods=entry_period).max().shift(1)
    donch_low = low.rolling(exit_period, min_periods=exit_period).min().shift(1)

    # State machine: enter long when close > donch_high (entry); exit when close < donch_low.
    arr_close = close.to_numpy()
    arr_high = donch_high.to_numpy()
    arr_low = donch_low.to_numpy()
    pos = np.zeros(len(close), dtype=float)
    state = 0
    for i in range(len(close)):
        if np.isnan(arr_high[i]) or np.isnan(arr_low[i]):
            pos[i] = float(state)
            continue
        if state == 0 and arr_close[i] > arr_high[i]:
            state = 1
        elif state == 1 and arr_close[i] < arr_low[i]:
            state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=close.index)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * log_ret
    dpos = position.diff().abs().fillna(0.0)
    cost = dpos * (1.0 / 10_000.0)  # 1 bp for liquid US equity H1
    return (gross - cost).rename("ret")


# ─────────────────────────────────────────────────────────────────────
# Main triage
# ─────────────────────────────────────────────────────────────────────


def _verdict(sr: float, ci_lo: float) -> str:
    if sr > 0.30 and ci_lo > 0:
        return "POSSIBLY VIABLE — full L52 hybrid audit recommended"
    elif sr > 0:
        return "MARGINAL — full L52 hybrid audit needed to confirm"
    elif sr > -0.30:
        return "MARGINAL FAIL — low edge; defer decision"
    else:
        return "RETIRE — clear signal-layer fail"


def run_triage() -> None:
    print("=" * 72)
    print("Wave B P2 Signal-Layer Triage (L58 pattern)")
    print("=" * 72)

    # fx_carry on AUD/JPY daily
    print("\n[fx_carry]")
    bars = _load_parquet("AUD_JPY_D.parquet")
    rets = fx_carry_signal(bars)
    sr = float(sharpe(rets, periods_per_year=BARS_PER_YEAR["D"]))
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets, periods_per_year=BARS_PER_YEAR["D"], seed=42)
    mdd = float(max_drawdown(rets))
    print("  AUD/JPY D, sma_period=50, vol_target=0.08, 14y")
    print(f"  Sharpe = {sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  MaxDD = {mdd * 100:+.1f}%")
    print(f"  Verdict: {_verdict(sr, ci_lo)}")
    fx_carry_summary = (sr, ci_lo, ci_hi, mdd)

    # gold_macro on GLD daily
    print("\n[gold_macro]")
    bars = _load_parquet("GLD_D.parquet", date_col="")  # No timestamp col; uses index
    # Fall back to manual indexing
    bars = pd.read_parquet(DATA_DIR / "GLD_D.parquet")
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars = bars.sort_index()
    rets = gold_macro_signal(bars)
    sr = float(sharpe(rets, periods_per_year=BARS_PER_YEAR["D"]))
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets, periods_per_year=BARS_PER_YEAR["D"], seed=42)
    mdd = float(max_drawdown(rets))
    print("  GLD D, slow_ma=200, vol_target=0.10, 21y")
    print(f"  Sharpe = {sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  MaxDD = {mdd * 100:+.1f}%")
    print(f"  Verdict: {_verdict(sr, ci_lo)}")
    gold_macro_summary = (sr, ci_lo, ci_hi, mdd)

    # turtle on CAT H1
    print("\n[turtle]")
    bars = pd.read_parquet(DATA_DIR / "CAT_H1.parquet")
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)
        bars = bars.set_index("timestamp").sort_index()
    else:
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        bars = bars.sort_index()
    rets = turtle_signal(bars)
    sr = float(sharpe(rets, periods_per_year=BARS_PER_YEAR["H1"]))
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets, periods_per_year=BARS_PER_YEAR["H1"], seed=42)
    mdd = float(max_drawdown(rets))
    print("  CAT H1, entry_period=45, exit_period=30, 8y")
    print(f"  Sharpe = {sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  MaxDD = {mdd * 100:+.1f}%")
    print(f"  Verdict: {_verdict(sr, ci_lo)}")
    turtle_summary = (sr, ci_lo, ci_hi, mdd)

    # Write summary CSV
    csv_path = REPORTS_DIR / "triage_summary.csv"
    lines = [
        "strategy,Sharpe,CI95_lo,CI95_hi,MaxDD,verdict",
        f"fx_carry_audjpy,{fx_carry_summary[0]:.4f},{fx_carry_summary[1]:.3f},{fx_carry_summary[2]:.3f},{fx_carry_summary[3] * 100:.1f},{_verdict(*fx_carry_summary[:2])}",
        f"gold_macro,{gold_macro_summary[0]:.4f},{gold_macro_summary[1]:.3f},{gold_macro_summary[2]:.3f},{gold_macro_summary[3] * 100:.1f},{_verdict(*gold_macro_summary[:2])}",
        f"turtle_cat_h1,{turtle_summary[0]:.4f},{turtle_summary[1]:.3f},{turtle_summary[2]:.3f},{turtle_summary[3] * 100:.1f},{_verdict(*turtle_summary[:2])}",
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wave-b-triage] wrote: {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_triage()
