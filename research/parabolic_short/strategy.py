"""Parabolic Short strategy: setup detection, trade simulation, returns.

Logic (frozen — no tuning):
  Setup (must all be true on day t):
    - close[t-3..t-1] > open[t-3..t-1]            (3 prior green days)
    - open[t] > close[t-1] * (1 + GAP_PCT)         (gap up on day t)
    - volume[t] > VOL_MULT * 20-day-avg-volume    (volume blowoff)
    - close[t-1] > 10dSMA[t-1] * (1 + EXT_PCT)    (price extended)

  Trigger (the parabolic must break):
    - close[t] < open[t]   (red day after the gap up — failed parabolic)
    -> short at open[t+1]

  Stop:  high[t]  (parabolic high, proxy for HOD)
  Exit:  first of:
    - close[k] <= 10dSMA[k]    (mean-reversion target hit) -> exit at close[k]
    - close[k] > stop          (stop out on close)         -> exit at close[k]
    - k - (t+1) >= MAX_HOLD    (time stop)                 -> exit at close[k]

  Costs: 5 bps per side = 10 bps round-trip on price.
  Sizing: 1% equity risk per trade (entry-stop distance).

The setup is intentionally conservative vs Hanlin's intraday spec — we
require the *daily-chart* parabolic-and-fail pattern to fully form, sacrificing
the intraday LOD-break entry for cleaner backtest fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Frozen parameters
GAP_PCT = 0.03      # 3% gap-up minimum
VOL_MULT = 1.5      # volume must exceed 1.5x 20d avg
EXT_PCT = 0.10      # 10% above 10dSMA
SMA_LEN = 10
MAX_HOLD = 10       # trading days
COST_BPS = 5.0      # per side
RISK_PCT = 0.01     # 1% per trade


@dataclass(frozen=True)
class Trade:
    symbol: str
    setup_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str   # 'tp' | 'sl' | 'time'
    r_multiple: float
    return_pct: float  # gross return on capital (sign for short)


def detect_setups(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series aligned to df.index, True on day t when a
    valid Parabolic Short setup with reversal trigger formed (i.e., t closed
    red after gap-up + green-streak + volume + extension).

    The signal at day t means "short at open of t+1 the next session."
    """
    o, _h, _l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    sma10 = c.rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    avg_v20 = v.rolling(20, min_periods=20).mean()

    green = (c > o)
    three_green = green.shift(1) & green.shift(2) & green.shift(3)
    gap_up = (o > c.shift(1) * (1.0 + GAP_PCT))
    vol_blowoff = (v > avg_v20 * VOL_MULT)
    extended = (c.shift(1) > sma10.shift(1) * (1.0 + EXT_PCT))
    red_today = (c < o)

    setup = three_green & gap_up & vol_blowoff & extended & red_today
    return setup.fillna(False)


def simulate_trades(df: pd.DataFrame, symbol: str) -> list[Trade]:
    """Walk the daily series, generate trades for every setup. Trades do not
    overlap (one open trade at a time per symbol — typical for swing trading)."""
    setups = detect_setups(df)
    o, h, c = df["open"], df["high"], df["close"]
    sma10 = c.rolling(SMA_LEN, min_periods=SMA_LEN).mean()

    idx = df.index
    open_arr = o.to_numpy()
    high_arr = h.to_numpy()
    close_arr = c.to_numpy()
    sma_arr = sma10.to_numpy()

    n = len(df)
    trades: list[Trade] = []
    blocked_until = -1
    setup_idxs = np.where(setups.to_numpy())[0]

    cost_factor = COST_BPS / 1e4  # per side, in price-fraction terms

    for t_idx in setup_idxs:
        if t_idx <= blocked_until:
            continue
        entry_idx = t_idx + 1
        if entry_idx >= n:
            continue
        entry_open = open_arr[entry_idx]
        stop = high_arr[t_idx]
        if not (np.isfinite(entry_open) and np.isfinite(stop)) or stop <= entry_open:
            continue
        risk = stop - entry_open  # short: stop is above entry
        if risk <= 0:
            continue

        # Walk forward to find exit
        exit_reason = "time"
        exit_idx = min(entry_idx + MAX_HOLD, n - 1)
        exit_price = close_arr[exit_idx]
        for k in range(entry_idx, min(entry_idx + MAX_HOLD + 1, n)):
            ck = close_arr[k]
            if not np.isfinite(ck):
                continue
            sma_k = sma_arr[k]
            # TP: close <= 10dSMA
            if np.isfinite(sma_k) and ck <= sma_k:
                exit_reason = "tp"
                exit_idx = k
                exit_price = ck
                break
            # SL: close above stop
            if ck > stop:
                exit_reason = "sl"
                exit_idx = k
                exit_price = ck
                break
        # gross short return: (entry - exit) / entry, minus 2x cost
        gross_ret = (entry_open - exit_price) / entry_open
        net_ret = gross_ret - 2.0 * cost_factor
        # R-multiple: how many "risk units" was the move
        r_mult = (entry_open - exit_price) / risk - 2.0 * cost_factor * entry_open / risk

        trades.append(Trade(
            symbol=symbol,
            setup_date=idx[t_idx],
            entry_date=idx[entry_idx],
            entry_price=float(entry_open),
            stop_price=float(stop),
            exit_date=idx[exit_idx],
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            r_multiple=float(r_mult),
            return_pct=float(net_ret),
        ))
        blocked_until = exit_idx

    return trades


def trades_to_bar_returns(
    trades: list[Trade],
    bar_index: pd.DatetimeIndex,
    risk_per_trade: float = RISK_PCT,
) -> pd.Series:
    """Book each trade's R-multiple * risk_per_trade at the exit date.

    Returns a per-day return series aligned to bar_index. Booking at exit
    (not entry) means days between entry and exit show zero, reflecting
    "in-trade waiting" — correct for per-bar Sharpe.
    """
    rets = pd.Series(0.0, index=bar_index)
    for t in trades:
        ret = risk_per_trade * t.r_multiple
        if t.exit_date in rets.index:
            rets.loc[t.exit_date] += ret
        else:
            pos = rets.index.searchsorted(t.exit_date, side="right") - 1
            if 0 <= pos < len(rets):
                rets.iloc[pos] += ret
    return rets
