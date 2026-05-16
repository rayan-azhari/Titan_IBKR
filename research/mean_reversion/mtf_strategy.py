"""mtf — pure-research multi-timeframe-confluence strategy module (Wave A.5).

SCOPE: signal-layer of the live `titan/strategies/mtf/strategy.py`. Drops
operational machinery (NautilusTrader event-driven order placement, ATR
sizing, position inertia) and focuses on the SIGNAL question:

  > **Does the H1/H4/D/W confluence score have a meaningful Sharpe edge
    when implemented with explicit L21-correct causality?**

V1 (Round 4 audit) claimed OOS Combined Sharpe **+1.94** on EUR/USD WMA
config. **L21 risk:** the live strategy is event-driven (live bars
arrive in order), but the V1 backtest may have aligned multi-timeframe
signals incorrectly — using a daily bar's close at H1 timestamps
BEFORE the daily bar closes. This module enforces causality explicitly
via `.shift(1)` on each timeframe's signal AND resampling-aware
alignment so the H4 signal at H1 bar `t` uses H4 bars closed strictly
BEFORE `t`.

Mechanism (per H1 bar t):
    1. For each timeframe (H1, H4, D, W):
       a. Compute fast/slow MA over the timeframe's native bars.
       b. ma_signal = tanh((fast - slow) / vol_normalised_spread) * 0.5
       c. rsi_signal = (RSI(close) - 50) / 100
       d. Per-TF signal = ma_signal + rsi_signal, in [-1.0, +1.0].
    2. Align all per-TF signals to the H1 grid via FORWARD-FILL with
       a forced 1-bar shift (so signal at H1 timestamp t reflects the
       most recent TF bar closed STRICTLY BEFORE t).
    3. Weighted confluence score = sum_TF (weight[TF] * shifted_signal[TF]).
    4. Position = sign(score) where |score| > threshold; else 0.
    5. Apply L18 shift discipline: position.shift(1) * h1_return.

Causality contract (L04 / A1 / L18 / L21):
    - All MAs/RSI use rolling backward window (causal).
    - Multi-TF resampling alignment SHIFTS each TF's signal by +1 native
      bar BEFORE aligning to H1, so daily signal at H1 bar 10am Tuesday
      uses Monday's close (NOT same-day close).
    - Position shift by 1 H1 bar before earning return.

Pass `mtf_assert_causal(...)` to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MtfConfig:
    """Pure-research mtf config. Matches the live mtf.toml structure.

    Live default (config/mtf.toml): weights ={H1:0.10, H4:0.25, D:0.60, W:0.05},
    confirmation_threshold=0.10, exit_buffer=0.10, ma_type=SMA per-TF
    (H1=10/30, H4=10/50, D=13/20).
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {"H1": 0.10, "H4": 0.25, "D": 0.60, "W": 0.05}
    )
    confirmation_threshold: float = 0.10
    exit_buffer: float = 0.10
    # Per-TF MA + RSI lengths.
    h1_fast: int = 10
    h1_slow: int = 30
    h1_rsi: int = 21
    h4_fast: int = 10
    h4_slow: int = 50
    h4_rsi: int = 21
    d_fast: int = 13
    d_slow: int = 20
    d_rsi: int = 14
    w_fast: int = 4
    w_slow: int = 8
    w_rsi: int = 8
    # Vol-normalisation lookback for MA spread.
    spread_norm_window: int = 20
    # Cost model (FX).
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.5


def _ma(close: pd.Series, n: int) -> pd.Series:
    """Simple MA. Pure-research uses SMA; live uses WMA. The SMA→WMA
    substitution does NOT alter the L21 causality question (both are
    rolling-backward); use SMA here for clarity."""
    return close.rolling(n, min_periods=n).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Standard RSI(close, n). Past-only."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _per_tf_signal(
    close: pd.Series,
    *,
    fast: int,
    slow: int,
    rsi_period: int,
    spread_norm_window: int,
) -> pd.Series:
    """Per-timeframe signal in [-1.0, +1.0]. Computed on the TF's NATIVE
    bar series; alignment to H1 happens upstream.
    """
    s_fast = _ma(close, fast)
    s_slow = _ma(close, slow)
    # tanh of vol-normalised spread.
    spread = (s_fast - s_slow) / s_slow.abs().clip(lower=1e-10)
    spread_std = spread.rolling(spread_norm_window, min_periods=spread_norm_window).std(ddof=1)
    spread_norm = spread / spread_std.replace(0, np.nan)
    ma_signal = np.tanh(spread_norm) * 0.5
    r = _rsi(close, rsi_period)
    rsi_signal = (r - 50.0) / 100.0
    return (ma_signal + rsi_signal).clip(lower=-1.0, upper=1.0)


def _align_to_h1(
    per_tf_signal: pd.Series, h1_index: pd.DatetimeIndex, *, force_one_bar_shift: bool = True
) -> pd.Series:
    """Align a per-TF signal (computed on its native bars) to the H1 index.

    Causality (L21): SHIFT the per-TF signal by 1 native bar BEFORE forward-
    filling to H1. This ensures the H1 timestamp t sees only signals computed
    on TF bars that closed STRICTLY BEFORE t.

    For an H4 bar that closes at 12:00, its signal becomes effective at the
    NEXT H4 bar (closing 16:00), so at H1 timestamps 13:00, 14:00, 15:00
    the relevant H4 signal is the 08:00 bar's signal (which closed at 08:00 +
    4h = 12:00 — i.e. the bar BEFORE the 12:00 close, which is 04:00-08:00).
    Confusing but the .shift(1) does it correctly.
    """
    if force_one_bar_shift:
        shifted = per_tf_signal.shift(1)
    else:
        shifted = per_tf_signal
    return shifted.reindex(h1_index, method="ffill")


def mtf_returns(
    bars_by_tf: dict[str, pd.DataFrame], *, cfg: MtfConfig | None = None
) -> pd.Series:
    """Per-bar (H1) net return of the mtf confluence strategy.

    `bars_by_tf` must contain DataFrames keyed by 'H1', 'H4', 'D', 'W' with
    a 'close' column and a DatetimeIndex.
    """
    if cfg is None:
        cfg = MtfConfig()
    for tf in ("H1", "H4", "D", "W"):
        if tf not in bars_by_tf:
            raise ValueError(f"Missing bars for timeframe {tf!r}")

    h1_close = bars_by_tf["H1"]["close"]
    h1_index = h1_close.index

    # Per-TF native signals.
    sig_h1 = _per_tf_signal(
        bars_by_tf["H1"]["close"],
        fast=cfg.h1_fast,
        slow=cfg.h1_slow,
        rsi_period=cfg.h1_rsi,
        spread_norm_window=cfg.spread_norm_window,
    )
    sig_h4 = _per_tf_signal(
        bars_by_tf["H4"]["close"],
        fast=cfg.h4_fast,
        slow=cfg.h4_slow,
        rsi_period=cfg.h4_rsi,
        spread_norm_window=cfg.spread_norm_window,
    )
    sig_d = _per_tf_signal(
        bars_by_tf["D"]["close"],
        fast=cfg.d_fast,
        slow=cfg.d_slow,
        rsi_period=cfg.d_rsi,
        spread_norm_window=cfg.spread_norm_window,
    )
    sig_w = _per_tf_signal(
        bars_by_tf["W"]["close"],
        fast=cfg.w_fast,
        slow=cfg.w_slow,
        rsi_period=cfg.w_rsi,
        spread_norm_window=cfg.spread_norm_window,
    )

    # Align to H1 with explicit +1-bar shift (L21 correctness).
    h1_on_h1 = _align_to_h1(sig_h1, h1_index)
    h4_on_h1 = _align_to_h1(sig_h4, h1_index)
    d_on_h1 = _align_to_h1(sig_d, h1_index)
    w_on_h1 = _align_to_h1(sig_w, h1_index)

    # Weighted confluence score.
    score = (
        cfg.weights.get("H1", 0.0) * h1_on_h1
        + cfg.weights.get("H4", 0.0) * h4_on_h1
        + cfg.weights.get("D", 0.0) * d_on_h1
        + cfg.weights.get("W", 0.0) * w_on_h1
    )

    # Binary position with exit_buffer hysteresis.
    arr_score = score.fillna(0.0).to_numpy()
    pos = np.zeros(len(arr_score), dtype=float)
    state = 0
    for i in range(len(arr_score)):
        s = arr_score[i]
        if state == 0:
            if s >= cfg.confirmation_threshold:
                state = 1
            elif s <= -cfg.confirmation_threshold:
                state = -1
        elif state == 1:
            if s < -cfg.exit_buffer:
                state = -1
            elif s < 0:
                state = 0
        elif state == -1:
            if s > cfg.exit_buffer:
                state = 1
            elif s > 0:
                state = 0
        pos[i] = float(state)
    position = pd.Series(pos, index=h1_index)

    log_ret = np.log(h1_close / h1_close.shift(1)).fillna(0.0)
    held_lagged = position.shift(1).fillna(0.0)
    gross = held_lagged * log_ret

    if cfg.apply_costs:
        dpos = position.diff().abs().fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def mtf_assert_causal(
    bars_by_tf: dict[str, pd.DataFrame], *, cfg: MtfConfig | None = None
) -> None:
    """L04 / L21 smoke test — corrupting future bars in ANY timeframe must not
    change past returns. This is the L21 multi-TF causality test."""
    if cfg is None:
        cfg = MtfConfig()
    base = mtf_returns(bars_by_tf, cfg=cfg)
    rng = np.random.default_rng(42)
    n_h1 = len(bars_by_tf["H1"])
    t_corrupt = int(rng.integers(2000, n_h1 - 200))
    t_corrupt_ts = bars_by_tf["H1"].index[t_corrupt]

    corrupted = {tf: df.copy() for tf, df in bars_by_tf.items()}
    for tf in ("H1", "H4", "D", "W"):
        mask = corrupted[tf].index >= t_corrupt_ts
        corrupted[tf].loc[mask] = np.nan

    corrupted_ret = mtf_returns(corrupted, cfg=cfg)

    base_past = base.iloc[: t_corrupt - 1].dropna()
    corrupted_past = corrupted_ret.iloc[: t_corrupt - 1].dropna()
    common = base_past.index.intersection(corrupted_past.index)
    if len(common) == 0:
        raise AssertionError("Causality test could not find common past index")
    diffs = (base_past.reindex(common) - corrupted_past.reindex(common)).abs()
    max_diff = float(diffs.max())
    if max_diff > 1e-12:
        n_changed = int((diffs > 1e-12).sum())
        raise AssertionError(
            f"L21 causality FAILED: corrupting future bars (t_corrupt={t_corrupt_ts}) "
            f"changed {n_changed} past returns (max |delta|={max_diff:.2e})"
        )


__all__ = [
    "MtfConfig",
    "mtf_returns",
    "mtf_assert_causal",
]
