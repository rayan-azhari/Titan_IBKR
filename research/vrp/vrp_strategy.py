"""VRP Capture -- causal strategy function.

Pre-registered in ``directives/Pre-Reg E1 VRP Capture 2026-05-15.md`` §1-§3.
Class: ``StrategyClass.DAILY_MEAN_REVERSION``.

Inputs:
    closes: DataFrame indexed by daily bar timestamps with columns
            ``["VIXY"]`` (the implementation vehicle) and optionally
            ``["SPY"]`` (when ``defensive_long_spy_w > 0``). Yfinance
            adj close (total return). V3.6 A3: TR vs price-only must be
            EXPLICIT -- here it's TR.
    vix:    pd.Series of the spot VIX (CBOE 30-day implied vol index)
            indexed by date.
    vix9d:  pd.Series of the 9-day VIX index.
    vix3m:  pd.Series of the 3-month VIX index.

Output:
    per-bar strategy returns (pd.Series), cost-adjusted, indexed by
    closes.index. Convention: positive = long, negative = short. VIXY
    weight is typically negative (short-volatility position).

Causality contract (V3.6 A1 / L04):
    Position at bar t is the decision made on bar t-1's information.
    Decision uses VIX/VIX9D/VIX3M(t-1). New VIXY weight is held from
    close[t] to close[t+1] -- i.e. strategy_ret[t] = w[t-1] * vixy_ret[t].
    Implemented as ``weights.shift(1) * returns``.

Regime logic (V3.1 frozen at pre-reg commit):

    Contango regime (short VIXY at ``target_short_weight``):
        VIX(t-1)   / VIX9D(t-1) >= ratio_short_gate   (short-end contango)
        AND VIX3M(t-1) / VIX(t-1)  >= ratio_long_gate    (medium-end contango)

    Backwardation regime (flat VIXY; optional defensive long SPY):
        VIX3M(t-1) / VIX(t-1) < 0.98

    Mid-zone (flat VIXY): everything else.

Hysteresis (regime_buffer_pct):
    Once a regime is entered, the opposite gate must be crossed by
    >= regime_buffer_pct to flip. Prevents 50/50 churn near the
    boundary. (See L18: stateful buffer must compare against the
    LIVE incumbent's gate, not a stale snapshot.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VrpConfig:
    """One row of the E1 pre-reg grid."""

    ratio_short_gate: float = 1.00  # VIX/VIX9D contango threshold (short end)
    ratio_long_gate: float = 1.05  # VIX3M/VIX contango threshold (medium end)
    target_short_weight: float = -0.50  # VIXY weight in contango regime
    regime_buffer_pct: float = 0.02  # hysteresis on the ratio_long_gate
    defensive_long_spy_w: float = 0.00  # SPY weight in backwardation regime
    backwardation_ratio_long: float = 0.98  # ratio_long below which we go flat-VIXY


VRP_UNIVERSE: tuple[str, ...] = ("VIXY",)


def _aligned_series(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Normalise index to date-only (L20) and forward-fill any short gaps.

    Index normalisation prevents the timestamp-mismatch bug from GEM Step 7
    (L20). Forward-fill bounded at 3 bars handles weekends / single-day
    yfinance gaps; longer gaps remain NaN so the strategy fails-open to
    FLAT in those regions (it does not silently fabricate signal).
    """
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    target = pd.to_datetime(idx).tz_localize(None).normalize()
    return s.reindex(target).ffill(limit=3)


def vrp_target_weights(
    closes: pd.DataFrame,
    *,
    cfg: VrpConfig,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
) -> pd.DataFrame:
    """Compute the strategy's daily target-weight DataFrame.

    Returns a DataFrame indexed like closes with columns
    ``["VIXY", "SPY"]`` (SPY column all zeros if
    ``cfg.defensive_long_spy_w == 0``). Weights are the DESIRED next-bar
    holdings derived from end-of-bar-t-1 information; vrp_returns(...)
    applies the shift(1) to convert these to held-during-bar-t weights.

    Causality: all decisions use ``.shift(1)`` of the input signals so
    that ``weights.loc[t]`` is computed entirely from ``information <= t-1``.
    """
    idx = closes.index
    v = _aligned_series(vix, idx)
    v9 = _aligned_series(vix9d, idx)
    v3 = _aligned_series(vix3m, idx)

    # Causal: end-of-day t-1 ratios decide tomorrow's position.
    ratio_short = (v / v9).shift(1)
    ratio_long = (v3 / v).shift(1)

    # Vectorised gate evaluation. Hysteresis (regime_buffer_pct) applied
    # iteratively because the active regime depends on the prior state.
    n = len(idx)
    contango_raw = (ratio_short >= cfg.ratio_short_gate) & (ratio_long >= cfg.ratio_long_gate)
    backwardation_raw = ratio_long < cfg.backwardation_ratio_long
    contango_arr = contango_raw.values
    backwardation_arr = backwardation_raw.values
    rl = ratio_long.values

    state = np.zeros(n, dtype=np.int8)  # 0=flat, +1=contango/short, -1=backwardation/SPY
    cur = 0
    for i in range(n):
        if np.isnan(rl[i]):
            state[i] = cur
            continue
        if cur == 1:
            # Currently SHORT VIXY (contango). Stay unless ratio_long drops
            # below (ratio_long_gate - buffer) -- the symmetric exit gate.
            exit_thr = cfg.ratio_long_gate - cfg.regime_buffer_pct
            if rl[i] < exit_thr:
                cur = -1 if backwardation_arr[i] else 0
        elif cur == -1:
            # Currently in backwardation (flat VIXY / optional long SPY).
            # Stay unless ratio_long climbs back above
            # (backwardation_ratio_long + buffer).
            exit_thr = cfg.backwardation_ratio_long + cfg.regime_buffer_pct
            if rl[i] >= exit_thr:
                cur = 1 if contango_arr[i] else 0
        else:
            # Flat. Enter contango or backwardation if signals fire.
            if contango_arr[i]:
                cur = 1
            elif backwardation_arr[i]:
                cur = -1
        state[i] = cur

    w_vixy = np.where(state == 1, cfg.target_short_weight, 0.0)
    w_spy = np.where(state == -1, cfg.defensive_long_spy_w, 0.0)
    weights = pd.DataFrame(
        {"VIXY": w_vixy, "SPY": w_spy},
        index=idx,
    )
    return weights


def vrp_returns(
    closes: pd.DataFrame,
    *,
    cfg: VrpConfig,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
    cost_bps_per_turnover: float = 1.5,
    cost_fixed_usd_per_fill: float = 1.0,
    notional_usd: float = 30_000.0,
    rebalance_threshold: float = 0.05,
) -> pd.Series:
    """Per-bar cost-adjusted strategy returns.

    Apples-to-apples with `gem_returns`: same cost-model conventions
    (L23) -- bps/turnover plus a per-fill commission floor, with a live
    rebalance threshold suppressing sub-threshold daily tweaks.

    Returns
    -------
    pd.Series indexed like closes.index. NaN until the first valid
    signal (the lookback period). Convention: per-bar simple return.
    """
    weights = vrp_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)

    # Rebalance threshold (L23): only treat |Δw| > threshold as a real
    # fill -- live class will skip sub-threshold tweaks, so the backtest
    # mirrors that.
    held = pd.DataFrame(index=closes.index, columns=weights.columns, dtype=float)
    last_w = np.zeros(weights.shape[1])
    for i, ts in enumerate(weights.index):
        target = weights.iloc[i].values
        delta = np.abs(target - last_w)
        if delta.max() > rebalance_threshold or i == 0:
            last_w = target.copy()
        held.iloc[i] = last_w

    # Causal weights: shift(1) so today's return is earned with yesterday's
    # decided weight.
    held_lagged = held.shift(1).fillna(0.0)

    # Per-leg simple returns.
    leg_returns = pd.DataFrame(index=closes.index, columns=weights.columns, dtype=float)
    for col in weights.columns:
        if col in closes.columns:
            leg_returns[col] = closes[col].pct_change().fillna(0.0)
        else:
            leg_returns[col] = 0.0  # absent leg contributes nothing

    gross = (held_lagged * leg_returns).sum(axis=1)

    # Costs: bps/turnover on |Δw_held| + per-fill USD on any non-zero
    # turnover day.
    dw = held.diff().abs().fillna(0.0)
    turnover_per_bar = dw.sum(axis=1)
    bps_drag = turnover_per_bar * (cost_bps_per_turnover / 10_000.0)
    fill_mask = turnover_per_bar > 0
    fixed_drag = fill_mask.astype(float) * (cost_fixed_usd_per_fill / max(notional_usd, 1.0))
    cost = bps_drag + fixed_drag

    return (gross - cost).rename("vrp_returns")


def vrp_assert_causal(
    closes: pd.DataFrame,
    *,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
    cfg: VrpConfig | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> None:
    """Causality smoke test (A10 / L04).

    Corrupt future close values + future VIX-family values at n_trials
    random bars; assert weights at every t < t_corruption are bit-exact
    unchanged. Raises AssertionError on any divergence.
    """
    if cfg is None:
        cfg = VrpConfig()
    if "VIXY" not in closes.columns:
        raise ValueError("vrp_assert_causal: closes must include VIXY column")
    rng = np.random.default_rng(seed)
    n = len(closes)
    if n < 100:
        return  # not enough data to test

    base_w = vrp_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)

    for _ in range(n_trials):
        t_corrupt = int(rng.integers(50, n - 5))
        corrupt_closes = closes.copy()
        corrupt_closes.iloc[t_corrupt:] = corrupt_closes.iloc[t_corrupt:] * 1.5
        corrupt_vix = vix.copy()
        corrupt_vix9d = vix9d.copy()
        corrupt_vix3m = vix3m.copy()
        # Corrupt future VIX-family values too.
        if isinstance(corrupt_vix.index, pd.DatetimeIndex):
            t_ts = closes.index[t_corrupt]
            corrupt_vix.loc[corrupt_vix.index >= t_ts] *= 1.5
            corrupt_vix9d.loc[corrupt_vix9d.index >= t_ts] *= 1.5
            corrupt_vix3m.loc[corrupt_vix3m.index >= t_ts] *= 1.5

        corrupt_w = vrp_target_weights(
            corrupt_closes,
            cfg=cfg,
            vix=corrupt_vix,
            vix9d=corrupt_vix9d,
            vix3m=corrupt_vix3m,
        )
        past_base = base_w.iloc[:t_corrupt]
        past_corr = corrupt_w.iloc[:t_corrupt]
        if not past_base.equals(past_corr):
            diff_idx = (past_base != past_corr).any(axis=1)
            raise AssertionError(
                f"vrp_assert_causal: future corruption at t={t_corrupt} "
                f"changed past weights at {diff_idx.sum()} bars. "
                f"First divergence: {diff_idx[diff_idx].index[0]}"
            )
