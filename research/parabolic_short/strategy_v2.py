"""Parabolic Short v2 — frozen 3-day exit, no red-close trigger.

Refinement informed by Stage 1b diagnostic (peak mean-reversion edge at 3-day
horizon). To avoid in-sample peeking we validate on the sanctuary window only.

Differences from v1:
  * No red-close trigger: enter on every setup
  * Fixed 3-day exit (close of t+3) replaces 10dSMA target + 10-day time stop
  * Stop is still parabolic high of day t (close > stop = early exit)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.parabolic_short.strategy import (
    COST_BPS,
    EXT_PCT,
    GAP_PCT,
    SMA_LEN,
    VOL_MULT,
)

EXIT_HORIZON = 3  # trading days


@dataclass(frozen=True)
class TradeV2:
    symbol: str
    setup_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str   # 'time' | 'sl'
    r_multiple: float
    return_pct: float


def detect_setups_v2(df: pd.DataFrame) -> pd.Series:
    """Setup without red-close trigger."""
    o, c, v = df["open"], df["close"], df["volume"]
    sma10 = c.rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    avg_v20 = v.rolling(20, min_periods=20).mean()
    green = (c > o)
    three_green = green.shift(1) & green.shift(2) & green.shift(3)
    gap_up = (o > c.shift(1) * (1.0 + GAP_PCT))
    vol_blowoff = (v > avg_v20 * VOL_MULT)
    extended = (c.shift(1) > sma10.shift(1) * (1.0 + EXT_PCT))
    return (three_green & gap_up & vol_blowoff & extended).fillna(False)


def simulate_trades_v2(df: pd.DataFrame, symbol: str) -> list[TradeV2]:
    setups = detect_setups_v2(df)
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    n = len(df)
    cost = COST_BPS / 1e4

    trades: list[TradeV2] = []
    blocked_until = -1
    for t_idx in np.where(setups.to_numpy())[0]:
        if t_idx <= blocked_until:
            continue
        entry_idx = t_idx + 1
        if entry_idx >= n:
            continue
        entry_open = o[entry_idx]
        stop = h[t_idx]
        if not (np.isfinite(entry_open) and np.isfinite(stop)) or stop <= entry_open:
            continue
        risk = stop - entry_open
        if risk <= 0:
            continue

        # Walk t+1, t+2, t+3 — stop hit on close OR exit at t+3 close
        exit_reason = "time"
        exit_idx = min(entry_idx + EXIT_HORIZON - 1, n - 1)
        exit_price = c[exit_idx]
        for k in range(entry_idx, min(entry_idx + EXIT_HORIZON, n)):
            ck = c[k]
            if not np.isfinite(ck):
                continue
            if ck > stop:
                exit_reason = "sl"
                exit_idx = k
                exit_price = ck
                break

        gross_ret = (entry_open - exit_price) / entry_open
        net_ret = gross_ret - 2.0 * cost
        r_mult = (entry_open - exit_price) / risk - 2.0 * cost * entry_open / risk

        trades.append(TradeV2(
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
