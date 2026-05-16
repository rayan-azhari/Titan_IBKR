"""G4 -- Overnight session decomposition strategy (SPY-only v1).

Pre-registered in ``directives/Pre-Reg G4 Overnight Session Decomposition 2026-05-15.md``.

Inputs:
    closes: DataFrame indexed by daily-bar timestamp, columns must include
            ``["open", "close"]``. Yfinance / Databento daily bars work.
            Adj-close is total-return (incl. dividends); we ignore the
            small bias from dividend-day adj on open vs close.

Outputs:
    per-bar strategy returns (pd.Series, cost-adjusted, indexed like closes).

Sessions:
    overnight_ret[t] = open[t] / close[t-1] - 1     # close-to-open
    intraday_ret[t]  = close[t] / open[t]    - 1    # open-to-close
    daily_ret[t]     = close[t] / close[t-1] - 1    # close-to-close (benchmark)

Position rules (Strategy enum):
    OVERNIGHT_LONG  : long during overnight, flat intraday (LPS canonical)
    OVERNIGHT_INTRADAY_SHORT: long overnight, short intraday (aggressive)
    INTRADAY_LONG   : flat overnight, long intraday (LPS counterfactual)
    BUY_HOLD        : long full day (benchmark)

Causality (V3.6 A1 / L04):
    Per-bar return at time t uses open[t], close[t], close[t-1] -- ALL
    observed by the END of bar t. There is no .shift(1) at the daily
    aggregation level because we DECOMPOSE the already-observed bar.

    The operational discipline (NOT enforced in this research function,
    but required for live deployment): the trader places a buy order to
    be filled at close[t-1] (Market-on-Close) and a sell at open[t]
    (Market-on-Open). At the daily-bar level, this is a single round-trip
    per overnight session.

Cost model (L23):
    Two-leg round-trip per overnight: 1 BUY at close[t-1], 1 SELL at
    open[t]. For Strategy.OVERNIGHT_LONG that's 2 fills per bar.
    Modelled as 2x ``cost_bps_per_turnover`` + 2x ``cost_fixed_usd_per_fill``
    when ``apply_costs`` is True.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Strategy(Enum):
    OVERNIGHT_LONG = "overnight_long"
    OVERNIGHT_INTRADAY_SHORT = "overnight_intraday_short"
    INTRADAY_LONG = "intraday_long"
    BUY_HOLD = "buy_hold"


@dataclass(frozen=True)
class OvernightConfig:
    strategy: Strategy = Strategy.OVERNIGHT_LONG
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.5  # SPY is the most liquid ETF
    cost_fixed_usd_per_fill: float = 1.0  # IBKR Pro tier floor
    notional_usd: float = 30_000.0
    # Date filter (post-publication decay test); None = no filter.
    start_date: str | None = None  # ISO-8601 "YYYY-MM-DD"


def _session_returns(closes: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose daily OHLC into (overnight, intraday, daily) per-bar return Series.

    Returns three pd.Series indexed like ``closes.index``. The first bar
    of each series is NaN because there is no ``close[t-1]`` (overnight)
    or ``open[t-1]`` -- caller fills with 0.0.
    """
    if "open" not in closes.columns or "close" not in closes.columns:
        raise ValueError("_session_returns: closes must have 'open' and 'close' columns")
    o = closes["open"].astype(float)
    c = closes["close"].astype(float)
    c_lag = c.shift(1)
    overnight = (o / c_lag) - 1.0
    intraday = (c / o) - 1.0
    daily = (c / c_lag) - 1.0
    return overnight.rename("overnight"), intraday.rename("intraday"), daily.rename("daily")


def overnight_returns(
    closes: pd.DataFrame,
    *,
    cfg: OvernightConfig | None = None,
) -> pd.Series:
    """Per-bar cost-adjusted strategy returns for a single overnight cell."""
    if cfg is None:
        cfg = OvernightConfig()
    if cfg.start_date is not None:
        # Apply the date filter (used by C6 decay-test cell).
        cutoff = pd.Timestamp(cfg.start_date)
        if closes.index.tz is not None:
            cutoff = cutoff.tz_localize(closes.index.tz)
        closes = closes.loc[closes.index >= cutoff]

    overnight, intraday, daily = _session_returns(closes)
    overnight = overnight.fillna(0.0)
    intraday = intraday.fillna(0.0)
    daily = daily.fillna(0.0)

    s = cfg.strategy
    if s == Strategy.OVERNIGHT_LONG:
        gross = overnight
        fills_per_bar = 2.0  # MOC buy + MOO sell
    elif s == Strategy.OVERNIGHT_INTRADAY_SHORT:
        # +1 overnight (long), -1 intraday (short). Two distinct trade legs
        # per bar: a MOO->MOC short on top of the overnight long round-trip.
        # Conservatively model as 4 fills per bar (2 round-trips).
        gross = overnight - intraday
        fills_per_bar = 4.0
    elif s == Strategy.INTRADAY_LONG:
        gross = intraday
        fills_per_bar = 2.0  # MOO buy + MOC sell
    elif s == Strategy.BUY_HOLD:
        gross = daily
        fills_per_bar = 0.0  # benchmark: assume already long, no rebalance
    else:
        raise ValueError(f"Unknown strategy {s!r}")

    if cfg.apply_costs and fills_per_bar > 0:
        bps = (cfg.cost_bps_per_turnover / 10_000.0) * fills_per_bar
        fixed = (cfg.cost_fixed_usd_per_fill / max(cfg.notional_usd, 1.0)) * fills_per_bar
        cost = bps + fixed
        net = gross - cost
    else:
        net = gross

    return net.rename(f"strat_{s.value}")


def overnight_assert_causal(
    closes: pd.DataFrame,
    *,
    cfg: OvernightConfig | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> None:
    """A10 causality smoke test.

    Corrupt future open/close at random t; assert per-bar strategy
    returns at every t' < t are bit-exact unchanged.
    """
    if cfg is None:
        cfg = OvernightConfig()
    n = len(closes)
    if n < 100:
        return
    base = overnight_returns(closes, cfg=cfg)
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        t_corrupt = int(rng.integers(20, n - 5))
        corrupt = closes.copy()
        # Corrupt FUTURE open AND close columns.
        corrupt.iloc[t_corrupt:, corrupt.columns.get_indexer(["open", "close"])] = (
            corrupt.iloc[t_corrupt:, corrupt.columns.get_indexer(["open", "close"])] * 1.5
        )
        corrupt_rets = overnight_returns(corrupt, cfg=cfg)
        past_base = base.iloc[:t_corrupt]
        past_corr = corrupt_rets.iloc[:t_corrupt]
        if not past_base.equals(past_corr):
            diff = past_base != past_corr
            raise AssertionError(
                f"overnight_assert_causal: future corruption at t={t_corrupt} "
                f"changed past returns at {int(diff.sum())} bars"
            )
