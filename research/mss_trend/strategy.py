"""MSS trend strategy: signal generation, trade simulation, return series.

Logic (frozen, no tuning):
1. Daily trend state: 2 consecutive HH+HL (up) / LH+LL (down), N=2 swings.
2. 15M swings: N=6.
3. Entry trigger (long): 15M close > most recent confirmed 15M swing high,
   AND daily trend state at that 15M timestamp is +1.
4. Stop: at the most recent confirmed 15M swing low *prior* to the broken
   swing high (i.e., the low of the pullback that preceded the breakout).
   For shorts: most recent confirmed swing high prior to the broken swing low.
5. Take-profit: 2R (entry + 2 * |entry - stop|).
6. Trade is bar-by-bar simulated on 15M bars: enter at next bar open after
   the trigger close, then for each subsequent bar check intra-bar TP/SL hit
   (worst case: if both touched in the same bar, assume stop hit first =
   conservative).

All cross-timeframe joins use confirm-time, not bar-time. No ffill from
future. No same-bar look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.mss_trend.swing import Swing, detect_swings, trend_state_series


@dataclass(frozen=True)
class Trade:
    entry_idx: int
    entry_ts: pd.Timestamp
    entry_price: float
    direction: int  # +1 long, -1 short
    stop_price: float
    tp_price: float
    exit_idx: int
    exit_ts: pd.Timestamp
    exit_price: float
    exit_reason: str  # "tp", "sl", "eod"
    r_multiple: float


def daily_trend_at_15m(
    daily_df: pd.DataFrame,
    daily_trend: pd.Series,
    m15_index: pd.DatetimeIndex,
    daily_n: int,
) -> pd.Series:
    """Project the daily trend state onto the 15M bar index, causally.

    For 15M bar at timestamp `ts`, the daily trend state is the value
    associated with the most recent daily bar whose confirmation time is
    <= ts. Confirmation time of daily bar i is `daily_index[i + daily_n]`
    (i.e., the close of the bar that confirms i as a swing endpoint).

    Note: trend_state_series already reflects "state as of bar i, using
    only swings confirmed by i". So we want, for each 15M ts, the state
    at the latest daily bar whose timestamp <= ts. The trend itself then
    has a further `daily_n`-bar effective lag baked into it (no swing is
    confirmed before its idx + daily_n).
    """
    daily_ts = daily_df.index
    # For each m15 ts, find the latest daily idx whose daily_ts <= m15 ts.
    # Use searchsorted with side='right' - 1.
    pos = daily_ts.searchsorted(m15_index, side="right") - 1
    pos = np.clip(pos, 0, len(daily_ts) - 1)
    state = daily_trend.iloc[pos].to_numpy()
    # When pos < 0 (i.e., m15 ts before any daily bar) we should be 0
    invalid = m15_index < daily_ts[0]
    state = np.where(invalid, 0, state)
    return pd.Series(state, index=m15_index, dtype=np.int8)


def generate_trades(
    m15_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    daily_n: int = 2,
    m15_n: int = 6,
    tp_r_multiple: float = 2.0,
) -> list[Trade]:
    """Generate the full list of trades on the dataset (no train/test split here).

    Caller is responsible for slicing the resulting trade list / return series
    into IS / OOS / sanctuary.
    """
    # 1. Daily swings -> trend state on daily index
    daily_swings = detect_swings(daily_df["high"], daily_df["low"], n=daily_n)
    daily_trend = trend_state_series(daily_swings, n_bars=len(daily_df))
    daily_trend.index = daily_df.index

    # 2. Project daily trend onto 15M index causally
    m15_trend = daily_trend_at_15m(daily_df, daily_trend, m15_df.index, daily_n)

    # 3. 15M swings
    m15_swings = detect_swings(m15_df["high"], m15_df["low"], n=m15_n)
    # Sort by confirm_idx
    m15_swings_by_confirm = sorted(m15_swings, key=lambda s: s.confirm_idx)

    # 4. Walk 15M bars and detect MSS triggers
    closes = m15_df["close"].to_numpy()
    highs = m15_df["high"].to_numpy()
    lows = m15_df["low"].to_numpy()
    opens = m15_df["open"].to_numpy()
    n_bars = len(m15_df)
    trend_arr = m15_trend.to_numpy()

    cursor = 0
    confirmed_highs: list[Swing] = []
    confirmed_lows: list[Swing] = []

    trades: list[Trade] = []
    in_trade_until: int = -1  # no overlapping trades

    for i in range(n_bars):
        # Ingest swings confirmed by close of bar i
        while cursor < len(m15_swings_by_confirm) and m15_swings_by_confirm[cursor].confirm_idx <= i:
            sw = m15_swings_by_confirm[cursor]
            if sw.is_high:
                confirmed_highs.append(sw)
            else:
                confirmed_lows.append(sw)
            cursor += 1

        if i <= in_trade_until:
            continue
        if i + 1 >= n_bars:  # need a next bar to enter
            break

        td = trend_arr[i]
        c = closes[i]

        if td == 1 and confirmed_highs:
            recent_h = confirmed_highs[-1]
            # Need a confirmed low PRIOR to recent_h to use as stop
            prior_lows = [lo for lo in confirmed_lows if lo.idx < recent_h.idx]
            if not prior_lows or recent_h.idx >= i:
                continue
            # MSS: close above recent confirmed swing high
            if c > recent_h.price:
                stop = prior_lows[-1].price
                if stop >= c:  # malformed
                    continue
                entry_idx = i + 1
                entry_price = float(opens[entry_idx])
                risk = entry_price - stop
                if risk <= 0:
                    continue
                tp = entry_price + tp_r_multiple * risk
                tr = _simulate_trade(
                    m15_df=m15_df,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    direction=1,
                    stop=stop,
                    tp=tp,
                    highs=highs,
                    lows=lows,
                )
                trades.append(tr)
                in_trade_until = tr.exit_idx
                # Reset confirmed_highs to require a new break — we keep the
                # recent_h for future trades but mark we already used it by
                # requiring a fresh high above it next time. Simplest: drop
                # all confirmed highs so that a fresh structure must re-form.
                confirmed_highs = []

        elif td == -1 and confirmed_lows:
            recent_l = confirmed_lows[-1]
            prior_highs = [hi for hi in confirmed_highs if hi.idx < recent_l.idx]
            if not prior_highs or recent_l.idx >= i:
                continue
            if c < recent_l.price:
                stop = prior_highs[-1].price
                if stop <= c:
                    continue
                entry_idx = i + 1
                entry_price = float(opens[entry_idx])
                risk = stop - entry_price
                if risk <= 0:
                    continue
                tp = entry_price - tp_r_multiple * risk
                tr = _simulate_trade(
                    m15_df=m15_df,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    direction=-1,
                    stop=stop,
                    tp=tp,
                    highs=highs,
                    lows=lows,
                )
                trades.append(tr)
                in_trade_until = tr.exit_idx
                confirmed_lows = []

    return trades


def _simulate_trade(
    *,
    m15_df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    direction: int,
    stop: float,
    tp: float,
    highs: np.ndarray,
    lows: np.ndarray,
) -> Trade:
    """Walk bars from entry_idx forward. Conservative same-bar resolution:
    if both stop and tp touched in the same bar, assume stop hits first."""
    n = len(m15_df)
    for j in range(entry_idx, n):
        h, lo = highs[j], lows[j]
        if direction == 1:
            stop_hit = lo <= stop
            tp_hit = h >= tp
            if stop_hit and tp_hit:
                exit_price = stop
                reason = "sl"
            elif stop_hit:
                exit_price = stop
                reason = "sl"
            elif tp_hit:
                exit_price = tp
                reason = "tp"
            else:
                continue
            r = (exit_price - entry_price) / (entry_price - stop)
            return Trade(
                entry_idx=entry_idx,
                entry_ts=m15_df.index[entry_idx],
                entry_price=entry_price,
                direction=1,
                stop_price=stop,
                tp_price=tp,
                exit_idx=j,
                exit_ts=m15_df.index[j],
                exit_price=exit_price,
                exit_reason=reason,
                r_multiple=r,
            )
        else:
            stop_hit = h >= stop
            tp_hit = lo <= tp
            if stop_hit and tp_hit:
                exit_price = stop
                reason = "sl"
            elif stop_hit:
                exit_price = stop
                reason = "sl"
            elif tp_hit:
                exit_price = tp
                reason = "tp"
            else:
                continue
            r = (entry_price - exit_price) / (stop - entry_price)
            return Trade(
                entry_idx=entry_idx,
                entry_ts=m15_df.index[entry_idx],
                entry_price=entry_price,
                direction=-1,
                stop_price=stop,
                tp_price=tp,
                exit_idx=j,
                exit_ts=m15_df.index[j],
                exit_price=exit_price,
                exit_reason=reason,
                r_multiple=r,
            )
    # Forced exit at last bar
    exit_idx = n - 1
    exit_price = float(m15_df["close"].iloc[exit_idx])
    if direction == 1:
        r = (exit_price - entry_price) / (entry_price - stop)
    else:
        r = (entry_price - exit_price) / (stop - entry_price)
    return Trade(
        entry_idx=entry_idx,
        entry_ts=m15_df.index[entry_idx],
        entry_price=entry_price,
        direction=direction,
        stop_price=stop,
        tp_price=tp,
        exit_idx=exit_idx,
        exit_ts=m15_df.index[exit_idx],
        exit_price=exit_price,
        exit_reason="eod",
        r_multiple=r,
    )


def trades_to_bar_returns(
    trades: list[Trade],
    bar_index: pd.DatetimeIndex,
    risk_per_trade: float = 0.01,
    cost_per_trade: float = 5e-5,
) -> pd.Series:
    """Convert a list of trades into a per-bar return series aligned to bar_index.

    Each trade's R-multiple is converted to a return = `risk_per_trade *
    r_multiple` (so risk-per-trade=1% means a +2R trade returns +2%).
    The return is booked at the trade's exit bar (not entry) — this makes
    the per-bar Sharpe correct because zero-return bars between entry and
    exit reflect "in-trade waiting", which is real market exposure.

    Cost is deducted at entry+exit symmetrically. cost_per_trade=5e-5 = 0.5bp
    per side = 1bp round-trip.
    """
    rets = pd.Series(0.0, index=bar_index)
    for t in trades:
        # Net of round-trip cost expressed in equity terms
        gross = risk_per_trade * t.r_multiple
        # Cost is in price-fraction terms; for a long position the cost
        # is roughly cost_per_trade * 2 of position-notional. But our
        # equity return is already scaled by risk_per_trade; cost should
        # be applied as an absolute haircut on each trade.
        # Convert cost_per_trade to equity-return units: risk_per_trade
        # corresponds to (entry-stop)/entry of price move; cost is
        # cost_per_trade of price. So ratio = 2 * cost_per_trade *
        # entry_price / |entry-stop| in equity terms.
        denom = abs(t.entry_price - t.stop_price)
        if denom > 1e-12:
            cost_equity = 2.0 * cost_per_trade * t.entry_price / denom * risk_per_trade
        else:
            cost_equity = 0.0
        net = gross - cost_equity
        # Book at exit bar
        if t.exit_ts in rets.index:
            rets.loc[t.exit_ts] += net
        else:
            # Find nearest bar at or before exit_ts
            pos = rets.index.searchsorted(t.exit_ts, side="right") - 1
            if 0 <= pos < len(rets):
                rets.iloc[pos] += net
    return rets
