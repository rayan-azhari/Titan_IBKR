"""D2 -- Commodity futures carry strategy.

Pre-registered in ``directives/Pre-Reg D2 Commodity Futures Carry 2026-05-15.md``.
Class: ``StrategyClass.CARRY``.

Inputs:
    M1_df: DataFrame indexed by daily-bar timestamp (date-only), columns
           are commodity roots (CL, GC, ZC, ...). Each cell holds the
           CLOSE of the front-month continuous-contract series.
    M2_df: same shape, second-nearest contract close.

Output:
    per-bar strategy returns (pd.Series, cost-adjusted, indexed by the
    common-date intersection of M1 and M2). Convention: positive return
    means the long-short portfolio earned positive P&L on that bar.

Carry signal (per commodity, per bar t):
    raw_carry[i,t] = log(F_M1[i, t-1] / F_M2[i, t-1])

    Positive = backwardated (near contract more expensive than far),
    historically associated with positive expected forward return.

Portfolio construction (cross-sectional, at each rebalance date):
    1. Compute raw_carry for every commodity with valid M1+M2.
    2. Rank by carry. Long top breadth_pct, short bottom breadth_pct
       (canonical: quintile, breadth_pct=0.20). C6 long-only flag
       suppresses the short leg.
    3. Equal-weight within each leg (or inverse-vol if cfg.weighting=
       "inverse_vol"). Long leg notional = +0.5 NAV; short leg = -0.5 NAV;
       net = 0 (market-neutral).

Causality (V3.6 A1 / L04):
    Weights at bar t are derived from carry computed on close[t-1].
    Returns earned: w[t] * log(F_M1[t] / F_M1[t-1]).
    The `.shift(1)` discipline is enforced in build_portfolio_weights
    (the carry signal is already shifted; the rebalance dates use the
    PRIOR-period info exclusively).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CarryConfig:
    """One row of the D2 pre-reg grid."""

    rebalance: Literal["weekly", "monthly"] = "monthly"
    breadth_pct: float = 0.20  # quintile = 0.20; tercile ~ 0.33
    long_only: bool = False  # if True, suppress the short leg (C6)
    smooth_days: int = 1  # 1 = raw daily carry; 5 = rolling 5-day mean
    weighting: Literal["equal", "inverse_vol"] = "equal"
    inverse_vol_lookback: int = 60  # only used when weighting="inverse_vol"

    # Signal mode (post-2026-05-15 pre-reg amendment).
    # "m1_m2_basis": strict BGR carry = log(F_M1 / F_M2). Requires M2 data.
    # "rolling_yield": BGR §3.2 proxy = log(F_M1[t-1] / F_M1[t-1-yield_lookback]).
    #   Correlates ~0.65 with strict carry in BGR's data. Used when M2 is
    #   unavailable (Databento .c.1 is too slow on GLBX.MDP3).
    signal_mode: Literal["m1_m2_basis", "rolling_yield"] = "m1_m2_basis"
    yield_lookback: int = 252  # ~12 months; only used when signal_mode="rolling_yield"

    # Costs (CME futures liquid; L23).
    apply_costs: bool = True
    cost_bps_per_turnover: float = 1.0
    cost_fixed_usd_per_fill: float = 1.0
    notional_usd_per_leg: float = 30_000.0


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Date-only, tz-naive index — required for cross-commodity reindex (L20)."""
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def _align_universe(M1_df: pd.DataFrame, M2_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Common-date intersection across columns AND across the two frames.

    Returns (M1_aligned, M2_aligned) with identical date index and identical
    column set (only commodities present in BOTH frames after dropna).
    """
    M1 = _normalize_index(M1_df)
    M2 = _normalize_index(M2_df)
    # Keep only commodities that exist in BOTH frames.
    common_cols = sorted(set(M1.columns) & set(M2.columns))
    M1 = M1[common_cols]
    M2 = M2[common_cols]
    # Common dates only.
    common_idx = M1.index.intersection(M2.index)
    M1 = M1.loc[common_idx]
    M2 = M2.loc[common_idx]
    return M1, M2


def compute_carry_signal(
    M1_df: pd.DataFrame,
    M2_df: pd.DataFrame,
    *,
    smooth_days: int = 1,
    signal_mode: str = "m1_m2_basis",
    yield_lookback: int = 252,
) -> pd.DataFrame:
    """Causal cross-sectional carry signal.

    Two signal modes:
        m1_m2_basis (strict BGR): log(F_M1[t-1] / F_M2[t-1]).
            Positive when backwardated. Requires both M1 and M2 data.
        rolling_yield (BGR §3.2 proxy): log(F_M1[t-1] / F_M1[t-1-yield_lookback]).
            12-month trailing return on the M1 series. Correlates ~0.65
            with strict carry in BGR's data; used when M2 is unavailable.
            M2_df may be empty/None in this mode.

    Returns a DataFrame indexed like the common dates, columns = commodities.
    Cell [t, i] holds carry[i] computed from prices through t-1 (causal).
    """
    if signal_mode == "m1_m2_basis":
        M1, M2 = _align_universe(M1_df, M2_df)
        M1_lag = M1.shift(1)
        M2_lag = M2.shift(1)
        raw = np.log(M1_lag.clip(lower=1e-9) / M2_lag.clip(lower=1e-9))
    elif signal_mode == "rolling_yield":
        M1 = _normalize_index(M1_df)
        # Trailing-yield proxy: 12-month log return on M1 series.
        M1_lag = M1.shift(1).clip(lower=1e-9)
        M1_lag_back = M1_lag.shift(yield_lookback)
        raw = np.log(M1_lag / M1_lag_back)
    else:
        raise ValueError(
            f"compute_carry_signal: signal_mode must be 'm1_m2_basis' or "
            f"'rolling_yield', got {signal_mode!r}"
        )
    if smooth_days > 1:
        raw = raw.rolling(smooth_days, min_periods=smooth_days).mean()
    return raw


def _month_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask over idx marking the last trading day of each calendar month."""
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    period = idx.to_period("M")
    last_seen: dict = {}
    for i, p in enumerate(period):
        last_seen[p] = i
    mask = np.zeros(len(idx), dtype=bool)
    for i in last_seen.values():
        mask[i] = True
    return mask


def _week_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask marking the last trading day of each calendar week (Fri or
    last business day in the week)."""
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    period = idx.to_period("W")
    last_seen: dict = {}
    for i, p in enumerate(period):
        last_seen[p] = i
    mask = np.zeros(len(idx), dtype=bool)
    for i in last_seen.values():
        mask[i] = True
    return mask


def build_portfolio_weights(
    signal_df: pd.DataFrame,
    *,
    cfg: CarryConfig,
    realised_vol_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert the carry signal into per-commodity portfolio weights.

    Returns a DataFrame indexed like signal_df, columns same as signal_df.
    Cell [t, i] is the WEIGHT to hold commodity i during bar t.
    Sum of LONG-leg weights = +0.5; sum of SHORT-leg weights = -0.5
    (when long_only=False). When long_only=True, long sum = +1.0 and
    short side is zero.
    """
    if cfg.weighting == "inverse_vol" and realised_vol_df is None:
        raise ValueError("CarryConfig.weighting='inverse_vol' requires realised_vol_df")

    idx = signal_df.index
    cols = list(signal_df.columns)
    weights = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)

    if cfg.rebalance == "monthly":
        rebal_mask = _month_end_mask(idx)
    elif cfg.rebalance == "weekly":
        rebal_mask = _week_end_mask(idx)
    else:
        raise ValueError(f"Unknown rebalance={cfg.rebalance!r}")

    current = pd.Series(0.0, index=cols, dtype=float)
    for i in range(len(idx)):
        if rebal_mask[i]:
            signal_row = signal_df.iloc[i].dropna()
            if len(signal_row) >= 4:  # need at least a few commodities to rank
                n = len(signal_row)
                top_n = max(1, int(round(n * cfg.breadth_pct)))
                sorted_assets = signal_row.sort_values(ascending=False)
                long_names = sorted_assets.iloc[:top_n].index.tolist()
                short_names = sorted_assets.iloc[-top_n:].index.tolist()

                if cfg.weighting == "equal":
                    long_w = 0.5 / top_n if long_names else 0.0
                    short_w = -0.5 / top_n if short_names else 0.0
                    new_w = pd.Series(0.0, index=cols, dtype=float)
                    for n_ in long_names:
                        new_w[n_] = long_w
                    if not cfg.long_only:
                        for n_ in short_names:
                            new_w[n_] = short_w
                    else:
                        # Long-only variant: top quintile gets full 1.0 long.
                        for n_ in long_names:
                            new_w[n_] = 1.0 / top_n
                else:  # inverse_vol
                    rv = realised_vol_df.iloc[i] if realised_vol_df is not None else None
                    new_w = pd.Series(0.0, index=cols, dtype=float)
                    long_invvol = pd.Series(
                        [1.0 / max(rv[n_], 1e-6) for n_ in long_names], index=long_names
                    )
                    long_invvol = long_invvol / long_invvol.sum() * 0.5
                    for n_ in long_names:
                        new_w[n_] = long_invvol[n_]
                    if not cfg.long_only:
                        short_invvol = pd.Series(
                            [1.0 / max(rv[n_], 1e-6) for n_ in short_names], index=short_names
                        )
                        short_invvol = short_invvol / short_invvol.sum() * 0.5
                        for n_ in short_names:
                            new_w[n_] = -short_invvol[n_]
                    else:
                        new_w[:] = 0.0
                        long_invvol2 = pd.Series(
                            [1.0 / max(rv[n_], 1e-6) for n_ in long_names],
                            index=long_names,
                        )
                        long_invvol2 = long_invvol2 / long_invvol2.sum() * 1.0
                        for n_ in long_names:
                            new_w[n_] = long_invvol2[n_]

                current = new_w
        weights.iloc[i] = current.values

    return weights


def _bar_returns(M1_df: pd.DataFrame) -> pd.DataFrame:
    """Log returns of front-month contracts, bar-aligned."""
    return np.log(M1_df / M1_df.shift(1)).fillna(0.0)


def carry_returns(
    M1_df: pd.DataFrame,
    M2_df: pd.DataFrame,
    *,
    cfg: CarryConfig | None = None,
    M1_signal_df: pd.DataFrame | None = None,
    M2_signal_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Per-bar cost-adjusted carry-strategy returns.

    Parameters
    ----------
    M1_df, M2_df : DataFrames of front- and second-month CLOSE prices
        used for HOLDING-PERIOD RETURN computation. For continuous-
        contract backtests, these should be the BACK-ADJUSTED series
        (so daily log-returns reflect the actual P&L of holding the
        front contract through rolls, including the roll yield).
    M1_signal_df, M2_signal_df : optional DataFrames of RAW (non-back-
        adjusted) M1 and M2 closes used for the carry SIGNAL only.
        When the signal is ``log(M1/M2)`` (strict basis), ratios of
        independently back-adjusted series are biased by per-commodity
        cumulative roll factors and corrupt the cross-sectional ranking.
        Passing the raw forward curve here avoids that distortion. If
        ``None``, ``M1_df`` and ``M2_df`` are used for signal too
        (legacy behaviour, correct ONLY when caller has already supplied
        raw closes or is in ``rolling_yield`` mode where the bias is
        absent).
    """
    if cfg is None:
        cfg = CarryConfig()
    if M1_signal_df is None:
        M1_signal_df = M1_df
    if M2_signal_df is None:
        M2_signal_df = M2_df

    if cfg.signal_mode == "m1_m2_basis":
        M1, M2 = _align_universe(M1_df, M2_df)
        M1_sig, M2_sig = _align_universe(M1_signal_df, M2_signal_df)
        # Re-align signal frame to the returns frame's index (left join,
        # ffill so the signal lookup at any returns-bar t uses the most
        # recent raw basis observation).
        M1_sig = M1_sig.reindex(M1.index).ffill()
        M2_sig = M2_sig.reindex(M1.index).ffill()
    else:
        # rolling_yield mode -- M2 not required, and the signal uses M1
        # trailing returns. For back-adjusted M1 this is the CORRECT
        # economic holding-period return, so no separate signal frame
        # is needed.
        M1 = _normalize_index(M1_df)
        M2 = pd.DataFrame(index=M1.index, columns=M1.columns, dtype=float)
        M1_sig = M1
        M2_sig = M2
    signal = compute_carry_signal(
        M1_sig,
        M2_sig,
        smooth_days=cfg.smooth_days,
        signal_mode=cfg.signal_mode,
        yield_lookback=cfg.yield_lookback,
    )

    realised_vol = None
    if cfg.weighting == "inverse_vol":
        bar_log_ret = _bar_returns(M1)
        realised_vol = bar_log_ret.rolling(
            cfg.inverse_vol_lookback, min_periods=cfg.inverse_vol_lookback
        ).std(ddof=1)

    weights = build_portfolio_weights(signal, cfg=cfg, realised_vol_df=realised_vol)
    bar_log_ret = _bar_returns(M1)

    # Weights already encode the t-1 -> t holding period (they were
    # changed only on rebalance dates from prior-bar signal). No extra
    # shift needed.
    gross_per_leg = weights * bar_log_ret
    gross = gross_per_leg.sum(axis=1)

    if cfg.apply_costs:
        # Cost on rebalance bars only: |Δw_i| > 0 implies a fill on
        # commodity i. Fills per bar = sum_i 1{Δw_i != 0}. Per-fill bps
        # on the leg notional (|w_i|) + fixed USD per fill.
        dw = weights.diff().abs().fillna(0.0)
        fill_indicator = (dw > 1e-9).astype(float)
        n_fills_per_bar = fill_indicator.sum(axis=1)
        # bps drag per fill scaled by leg notional |w|
        bps_drag = (dw.sum(axis=1) * cfg.cost_bps_per_turnover) / 10_000.0
        fixed_drag = (
            n_fills_per_bar * cfg.cost_fixed_usd_per_fill / max(cfg.notional_usd_per_leg, 1.0)
        )
        net = gross - bps_drag - fixed_drag
    else:
        net = gross

    return net.rename("carry_strategy_returns")


def carry_assert_causal(
    M1_df: pd.DataFrame,
    M2_df: pd.DataFrame,
    *,
    cfg: CarryConfig | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> None:
    """A10 causality smoke test.

    Corrupt future M1 and M2 at random date t_corrupt; assert per-bar
    strategy returns at every date strictly before t_corrupt are bit-exact
    unchanged. Uses date-aligned corruption + comparison so that M1's
    raw index (which may be longer than the M1∩M2-aligned return series
    in m1_m2_basis mode) does not produce a positional mismatch.
    """
    if cfg is None:
        cfg = CarryConfig()
    n = len(M1_df)
    if n < 200:
        return
    base = carry_returns(M1_df, M2_df, cfg=cfg)
    if len(base) < 50:
        return
    rng = np.random.default_rng(seed)
    return_dates = base.index
    for _ in range(n_trials):
        # Pick a random date from the RETURN series (not M1's raw index).
        ret_pos = int(rng.integers(25, len(return_dates) - 5))
        t_corrupt_date = return_dates[ret_pos]
        m1c = M1_df.copy()
        m2c = M2_df.copy()
        # Date-aligned corruption: scale rows whose date >= t_corrupt_date.
        m1_mask = m1c.index >= t_corrupt_date
        m2_mask = m2c.index >= t_corrupt_date
        m1c.loc[m1_mask] = m1c.loc[m1_mask] * 1.5
        m2c.loc[m2_mask] = m2c.loc[m2_mask] * 1.5
        corrupt_rets = carry_returns(m1c, m2c, cfg=cfg)
        # Compare returns strictly before t_corrupt_date.
        past_base = base.loc[base.index < t_corrupt_date]
        past_corr = corrupt_rets.loc[corrupt_rets.index < t_corrupt_date]
        if not past_base.equals(past_corr):
            diff = past_base != past_corr
            raise AssertionError(
                f"carry_assert_causal: future corruption at "
                f"date={t_corrupt_date.date()} changed {int(diff.sum())} "
                f"past returns"
            )
