"""GEM Dual Momentum -- causal strategy function.

Pre-registered in ``directives/Pre-Reg GEM Dual Momentum 2026-05-14.md`` §1-§3.
Class: ``StrategyClass.CROSS_ASSET_MOMENTUM``.

Inputs:
    closes: DataFrame indexed by daily bar timestamps, with columns
            ``["SPY", "EFA", "IEF"]`` -- adjusted (total-return) closes.
            Yfinance "adj close" is what we use; this is TR (incl. dividends).
            See V3.6 A3: TR vs price-only must be EXPLICIT.

Outputs:
    weights: DataFrame same index, columns ["SPY", "EFA", "IEF"],
             rows are 0/1 vectors summing to 1 (single-name long-only).

Causality contract (V3.6 A1 / L04):
    Position at bar t is the decision made on bar t-1's information.
    Decision at the end of month M is based on 12m total returns
    through close[M-end]. That position takes effect at the open of
    month M+1 -- i.e. is held from close[M-end] through close[(M+1)-end].
    Implemented as ``weights = raw_decisions.shift(1)`` so the strategy
    earns return[t] using weight decided at close[t-1].

Buffering (cell-dependent):
    To avoid 50/50 churn near the decision threshold, the new winner
    must beat the incumbent by ``buffer_pct`` (in 12m-return space)
    to trigger a switch. C1 (canonical) uses 0.5%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GemConfig:
    """GEM cell config (one row of the pre-reg grid).

    Attributes:
        lookback_months: Lookback for the absolute-momentum gate. 12 is canonical.
        buffer_pct: Minimum challenger advantage (12m return delta) to trigger a
                    switch. 0.005 = 0.5%. Reduces churn.
        defensive_switch: If True, allocate to IEF when all risk assets lose to it.
                          If False (cell C5), always hold max(SPY,EFA) regardless.
    """

    lookback_months: int = 12
    buffer_pct: float = 0.005
    defensive_switch: bool = True


# The three universe names. Stable canonical order across the module.
GEM_UNIVERSE: tuple[str, ...] = ("SPY", "EFA", "IEF")
RISK_ASSETS: tuple[str, ...] = ("SPY", "EFA")  # things we can rotate INTO


def _month_end_idx(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask over ``idx`` marking the LAST daily bar of each calendar month.

    We use ``idx.to_period("M")`` to bucket and pick the max bar within each month.
    Edge: if a month has only one bar, that bar IS the month-end.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"_month_end_idx requires DatetimeIndex, got {type(idx).__name__}")
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    period = idx.to_period("M")
    # For each unique month, the LAST occurrence is the month-end.
    is_end = np.zeros(len(idx), dtype=bool)
    last_seen = {}
    for i, p in enumerate(period):
        last_seen[p] = i
    for i in last_seen.values():
        is_end[i] = True
    return is_end


def gem_target_weights(
    closes: pd.DataFrame,
    cfg: GemConfig | None = None,
) -> pd.DataFrame:
    """Compute the CAUSAL GEM weight series.

    Returns a DataFrame of weights aligned to ``closes.index``. Row r is the
    weight to HOLD over the interval ``(close[r-1], close[r]]``. Thus a
    ``returns[r] * weights[r]`` product is bar-aligned and look-ahead-free.

    Behaviour:
        * Decision is made at each month-end based on trailing
          ``cfg.lookback_months`` total returns relative to IEF.
        * Between month-ends, the position is HELD (no intra-month changes).
        * The weights series is one-bar-shifted at the end so position
          at bar r reflects the decision made at bar r-1.
    """
    if cfg is None:
        cfg = GemConfig()
    if not isinstance(closes, pd.DataFrame):
        raise TypeError(f"gem_target_weights expects DataFrame, got {type(closes).__name__}")
    missing = set(GEM_UNIVERSE) - set(closes.columns)
    if missing:
        raise ValueError(f"closes is missing columns: {sorted(missing)}")
    if closes.empty:
        return pd.DataFrame(0.0, index=closes.index, columns=list(GEM_UNIVERSE))

    closes = closes[list(GEM_UNIVERSE)].astype(float).copy()
    closes = closes.dropna(how="any")  # drop rows where any leg missing
    if closes.empty:
        return pd.DataFrame(0.0, index=closes.index, columns=list(GEM_UNIVERSE))

    # Approx lookback in trading days. 21 trading days per month is the convention.
    lookback_bars = int(cfg.lookback_months * 21)
    if len(closes) <= lookback_bars + 1:
        # Not enough history for a single decision.
        return pd.DataFrame(0.0, index=closes.index, columns=list(GEM_UNIVERSE))

    # Trailing 12m total return per name. r_t = close[t] / close[t-lookback] - 1.
    trailing_ret = closes / closes.shift(lookback_bars) - 1.0

    month_end = _month_end_idx(closes.index)

    # Raw (unshifted) decision per bar -- only updated on month-ends; held in between.
    raw_weight = pd.DataFrame(0.0, index=closes.index, columns=list(GEM_UNIVERSE))
    current_pick: str | None = None
    current_ret: float = -np.inf

    for i in range(len(closes)):
        if not month_end[i]:
            # Hold previous decision.
            if current_pick is not None:
                raw_weight.iloc[i, raw_weight.columns.get_loc(current_pick)] = 1.0
            continue
        # Decision day. Use trailing return AT THIS BAR.
        r_spy = trailing_ret.iloc[i]["SPY"]
        r_efa = trailing_ret.iloc[i]["EFA"]
        r_ief = trailing_ret.iloc[i]["IEF"]

        if not np.isfinite([r_spy, r_efa, r_ief]).all():
            # First lookback months: keep no position.
            current_pick = None
            current_ret = -np.inf
            continue

        # Absolute-momentum gate: at least one risk asset must beat IEF (cash proxy).
        spy_beats_cash = r_spy > r_ief
        efa_beats_cash = r_efa > r_ief

        if not cfg.defensive_switch:
            # Cell C5: always hold max of risk assets.
            challenger = "SPY" if r_spy >= r_efa else "EFA"
            challenger_ret = max(r_spy, r_efa)
        else:
            if spy_beats_cash and efa_beats_cash:
                challenger = "SPY" if r_spy >= r_efa else "EFA"
                challenger_ret = max(r_spy, r_efa)
            elif spy_beats_cash:
                challenger = "SPY"
                challenger_ret = r_spy
            elif efa_beats_cash:
                challenger = "EFA"
                challenger_ret = r_efa
            else:
                # Defensive: long IEF.
                challenger = "IEF"
                challenger_ret = r_ief

        # Apply buffer: switch only if challenger beats incumbent by buffer_pct.
        if current_pick is None or challenger == current_pick:
            current_pick = challenger
            current_ret = challenger_ret
        else:
            if challenger_ret - current_ret >= cfg.buffer_pct:
                current_pick = challenger
                current_ret = challenger_ret
            # else hold incumbent (don't update current_ret -- next month re-evaluates)

        raw_weight.iloc[i, raw_weight.columns.get_loc(current_pick)] = 1.0

    # CAUSAL SHIFT: position at row r reflects decision made at row r-1.
    # This is the V3.6 L04 / A1 discipline. Without this we'd be using close[t] info
    # to earn return[t].
    weights = raw_weight.shift(1).fillna(0.0)
    return weights


def gem_returns(
    closes: pd.DataFrame,
    cfg: GemConfig | None = None,
) -> pd.Series:
    """Compute the per-day GEM strategy returns (geometric, simple-return).

    Returns a per-bar return Series. Implementation contract:

        ret_t = sum_n weight[t, n] * (close[t, n] / close[t-1, n] - 1)

    where weight is the CAUSAL shifted weight from ``gem_target_weights``.
    Costs are NOT applied here -- caller adds them per the chosen cost model.
    """
    weights = gem_target_weights(closes, cfg=cfg)
    closes_aligned = closes[list(GEM_UNIVERSE)].astype(float).reindex(weights.index)
    bar_returns = closes_aligned.pct_change()
    # Sanity: at any row, weights sum to either 0 (warmup / no decision) or 1.
    # (We don't enforce this here -- a downstream test asserts it.)
    strat_ret = (weights * bar_returns).sum(axis=1)
    return strat_ret


def gem_assert_causal(
    closes: pd.DataFrame,
    cfg: GemConfig | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> None:
    """Causality smoke test (V3.6 A10).

    Corrupt future closes at a random t. The weights at bars < t must be
    bit-exact unchanged.

    Raises AssertionError on the first failing trial.
    """
    if cfg is None:
        cfg = GemConfig()
    if len(closes) < 100:
        raise ValueError(f"gem_assert_causal: need >= 100 bars, got {len(closes)}")
    rng = np.random.default_rng(seed)
    baseline = gem_target_weights(closes, cfg=cfg)
    n = len(closes)
    for trial in range(n_trials):
        t_idx = int(rng.integers(n // 2, n))
        corrupted = closes.copy()
        corrupted.iloc[t_idx:] = corrupted.iloc[t_idx:] * 100.0  # arbitrary future shock
        corrupted_w = gem_target_weights(corrupted, cfg=cfg)
        cutoff = closes.index[t_idx]
        past_baseline = baseline[baseline.index < cutoff]
        past_corrupted = corrupted_w[corrupted_w.index < cutoff]
        pd.testing.assert_frame_equal(
            past_baseline,
            past_corrupted,
            check_exact=True,
            check_names=False,
        )
