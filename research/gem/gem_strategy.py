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

from titan.research.metrics import BARS_PER_YEAR


@dataclass(frozen=True)
class GemConfig:
    """GEM cell config (one row of the pre-reg grid).

    Attributes:
        lookback_months:
            Single-lookback mode. Used when ``lookback_blend`` is None.
            12 is the Antonacci canonical.
        lookback_blend:
            Multi-speed mode (Step 2 enhancement). A tuple of lookback windows
            in months, e.g. (3, 6, 12). The strategy ranks each asset by EACH
            lookback's return, then averages the ranks to choose the winner.
            Detects regime change faster than the canonical single 12m. When
            non-None, ``lookback_months`` is ignored.
        absolute_gate_lookback_months:
            Lookback used for the absolute-momentum gate (risk-asset 12m return
            vs IEF 12m return). Defaults to 12 even in blend mode, because the
            defensive switch is an "is the regime bullish at all" test, where
            12m is the canonical robust signal. Setting this to a shorter
            window would make the defensive switch more reactive but also more
            whippy.
        buffer_pct:
            Minimum challenger advantage (return delta) to trigger a switch.
            In blend mode the comparison is on rank-average return; in single
            mode it's on raw 12m return. 0.005 = 0.5%.
        defensive_switch:
            If True, allocate to IEF when all risk assets lose to it on the
            absolute_gate_lookback_months horizon. If False (cell C5), always
            hold max(SPY,EFA) regardless.
    """

    lookback_months: int = 12
    lookback_blend: tuple[int, ...] | None = None
    absolute_gate_lookback_months: int = 12
    buffer_pct: float = 0.005
    defensive_switch: bool = True

    # ── Step 3: vol-target overlay (continuous risk control) ──────────────
    # When ``ann_vol_target`` is set, position size is scaled so that the
    # strategy's TRAILING realised volatility matches the target. Scaling
    # is capped at ``max_leverage`` (default 1.0 = no leverage-up). The
    # undeployed fraction goes to IEF (the safe asset) to keep equity
    # capital deployed earning carry rather than sitting in cash.
    #
    # Mechanically reduces 2008/2020 drawdowns because realised vol spikes
    # BEFORE the 12m momentum gate inverts. Causal: vol(t) uses returns
    # through t-1.
    ann_vol_target: float | None = None  # e.g. 0.10 for 10% annual vol
    vol_lookback_days: int = 20
    max_leverage: float = 1.0  # 1.0 = no leverage-up; >1 allows scaling up


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

    # Determine the lookback windows in use. Blend takes precedence over single.
    if cfg.lookback_blend is not None:
        if len(cfg.lookback_blend) == 0:
            raise ValueError("lookback_blend must contain at least one month value")
        lookback_months_set = tuple(sorted(set(cfg.lookback_blend)))
    else:
        lookback_months_set = (cfg.lookback_months,)
    # Always need the absolute-gate lookback for IEF comparison.
    gate_lookback_months = cfg.absolute_gate_lookback_months
    all_lookback_months = tuple(sorted(set(lookback_months_set + (gate_lookback_months,))))

    # 21 trading days per month is the convention.
    longest_bars = max(all_lookback_months) * 21
    if len(closes) <= longest_bars + 1:
        # Not enough history for any decision.
        return pd.DataFrame(0.0, index=closes.index, columns=list(GEM_UNIVERSE))

    # Pre-compute trailing returns for every lookback we'll need.
    # trailing_rets[m] is a DataFrame indexed by closes.index with the
    # m-month trailing return per asset.
    trailing_rets: dict[int, pd.DataFrame] = {}
    for m in all_lookback_months:
        bars = int(m * 21)
        trailing_rets[m] = closes / closes.shift(bars) - 1.0

    # Per-asset "selection return": either a single lookback's return, or the
    # mean of raw returns across the blend lookbacks. Raw-return mean (rather
    # than rank-average) keeps the buffer comparison interpretable as a
    # return-vs-return delta. The blend smooths fast-spike artifacts because
    # one lookback's spike is averaged with two slower lookbacks' values.
    if cfg.lookback_blend is not None:
        selection_ret = sum(trailing_rets[m] for m in cfg.lookback_blend) / len(cfg.lookback_blend)
    else:
        selection_ret = trailing_rets[cfg.lookback_months]

    # The "absolute gate" return is fixed at gate_lookback_months for IEF.
    # We compare each risk-asset's gate-lookback return against IEF's
    # gate-lookback return to decide whether to go defensive.
    gate_ret = trailing_rets[gate_lookback_months]

    month_end = _month_end_idx(closes.index)

    # Raw (unshifted) decision per bar -- only updated on month-ends; held in between.
    raw_weight = pd.DataFrame(0.0, index=closes.index, columns=list(GEM_UNIVERSE))
    current_pick: str | None = None

    for i in range(len(closes)):
        if not month_end[i]:
            # Hold previous decision.
            if current_pick is not None:
                raw_weight.iloc[i, raw_weight.columns.get_loc(current_pick)] = 1.0
            continue
        # Decision day. Two return concepts at this bar:
        #   1) Absolute-gate returns (always at gate_lookback_months)
        #      -- used to decide whether to go defensive into IEF.
        #   2) Selection returns (blend or single, per cfg.lookback_blend)
        #      -- used to pick the winning risk asset AND for buffer compare.
        gate_r_spy = gate_ret.iloc[i]["SPY"]
        gate_r_efa = gate_ret.iloc[i]["EFA"]
        gate_r_ief = gate_ret.iloc[i]["IEF"]
        sel_r_spy = selection_ret.iloc[i]["SPY"]
        sel_r_efa = selection_ret.iloc[i]["EFA"]
        sel_r_ief = selection_ret.iloc[i]["IEF"]

        if not np.isfinite(
            [gate_r_spy, gate_r_efa, gate_r_ief, sel_r_spy, sel_r_efa, sel_r_ief]
        ).all():
            # First lookback months: keep no position.
            current_pick = None
            continue

        # Absolute-momentum gate uses the fixed-horizon gate return.
        spy_beats_cash = gate_r_spy > gate_r_ief
        efa_beats_cash = gate_r_efa > gate_r_ief

        if not cfg.defensive_switch:
            # Cell C5: always hold max of risk assets (use selection return).
            challenger = "SPY" if sel_r_spy >= sel_r_efa else "EFA"
            challenger_ret = max(sel_r_spy, sel_r_efa)
        else:
            if spy_beats_cash and efa_beats_cash:
                challenger = "SPY" if sel_r_spy >= sel_r_efa else "EFA"
                challenger_ret = max(sel_r_spy, sel_r_efa)
            elif spy_beats_cash:
                challenger = "SPY"
                challenger_ret = sel_r_spy
            elif efa_beats_cash:
                challenger = "EFA"
                challenger_ret = sel_r_efa
            else:
                # Defensive: long IEF.
                challenger = "IEF"
                challenger_ret = sel_r_ief

        # Apply buffer: switch only if challenger beats incumbent's CURRENT
        # selection return by buffer_pct. Compare against the LIVE return
        # (V3.6 L17-style discipline, never against a stale snapshot).
        if current_pick is None or challenger == current_pick:
            current_pick = challenger
        else:
            incumbent_current_ret = selection_ret.iloc[i][current_pick]
            if challenger_ret - incumbent_current_ret >= cfg.buffer_pct:
                current_pick = challenger
            # else hold incumbent

        raw_weight.iloc[i, raw_weight.columns.get_loc(current_pick)] = 1.0

    # CAUSAL SHIFT: position at row r reflects decision made at row r-1.
    # This is the V3.6 L04 / A1 discipline. Without this we'd be using close[t] info
    # to earn return[t].
    weights = raw_weight.shift(1).fillna(0.0)

    # ── Vol-target overlay (Step 3) -- vectorised ────────────────────────
    if cfg.ann_vol_target is not None and cfg.ann_vol_target > 0:
        weights = _apply_vol_target(weights, closes, cfg)

    return weights


def _apply_vol_target(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    cfg: GemConfig,
) -> pd.DataFrame:
    """Scale weights so trailing realised vol of the strategy hits the target.

    Vectorised: pandas rolling for vol, numpy elementwise for the scale.
    No Python loop over bars.

    Causality:
      * Bar-returns are pct_change on closes -> known at end of bar t.
      * Unscaled strategy returns at t use weights at t (already shifted by 1).
      * Rolling vol uses returns through bar t.
      * Scale factor at bar t is derived from vol through t, then SHIFTED BY
        ONE BAR before multiplying weights -- so the scaling acts on the
        position FROM the next bar onwards. This preserves the existing
        shift(1) discipline.
      * Result: returns[t] = weights[t] * scale[t-1] * bar_returns[t], where
        scale[t-1] is computable strictly from data <= t-1.
    """
    closes_aligned = closes[list(GEM_UNIVERSE)].astype(float).reindex(weights.index)
    bar_returns = closes_aligned.pct_change().fillna(0.0)
    # UNSCALED strategy returns (the binary version of the strategy).
    raw_strat_ret = (weights * bar_returns).sum(axis=1)
    # Realised vol over a rolling window (annualised). Sufficient samples
    # only after vol_lookback_days bars; before that, scale = 1.0 (don't
    # over-correct on a short sample).
    rolling_std = raw_strat_ret.rolling(
        cfg.vol_lookback_days, min_periods=cfg.vol_lookback_days
    ).std(ddof=1)
    realised_vol = rolling_std * np.sqrt(BARS_PER_YEAR["D"])

    # Scale factor; clip to [0, max_leverage]. Where realised vol is zero or
    # NaN, fall back to 1.0 (full position).
    scale = (cfg.ann_vol_target / realised_vol).clip(upper=cfg.max_leverage)
    scale = scale.fillna(1.0)
    # Apply the causal shift -- decision uses vol through t-1.
    scale = scale.shift(1).fillna(1.0)

    # The active position is whichever asset has weight=1 (binary). Multiply
    # its weight by the scale; route the un-deployed fraction (1 - scale) to
    # IEF (the safe asset).
    risk_assets = ["SPY", "EFA"]  # the things vol-target scales
    scaled = weights.copy()
    risk_mask = scaled[risk_assets].sum(axis=1) > 0  # rows where active asset is SPY/EFA

    # For rows where we're in a risk asset: scale its weight + add (1-scale) to IEF.
    if risk_mask.any():
        # Scale down the active risk-asset weight.
        for col in risk_assets:
            scaled.loc[risk_mask, col] = weights.loc[risk_mask, col] * scale.loc[risk_mask]
        # Cash portion routed to IEF (only when in a risk asset).
        leftover = (1.0 - scale).clip(lower=0.0, upper=1.0)
        scaled.loc[risk_mask, "IEF"] = scaled.loc[risk_mask, "IEF"] + leftover.loc[risk_mask]

    # If we're already in IEF (defensive switch), leave as-is -- we're
    # already in the safe asset. Same for warmup rows where weights sum to 0.
    return scaled


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
