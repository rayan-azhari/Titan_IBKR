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

    # ── Step 4: conditional stress gate (limit exposure ONLY in stress) ──
    # The Step 3 overlay scales exposure ALL the time -- including calm
    # markets, which costs total return. The Step 4 enhancement gates the
    # vol-target on a binary stress signal: only scale down (or stay at
    # max_leverage) when the regime is stressed.
    #
    # When ``stress_gate_enabled`` is False (default), behaviour is
    # identical to Step 3 (no gate). When True, the overlay reads the
    # stress signal at each bar:
    #   * stress[t] = True  -> apply vol-target scaling (with leverage cap)
    #   * stress[t] = False -> deploy at max_leverage (no scaling down)
    #
    # Stress signal sources (any True -> stress):
    #   * stress_realised_vol_threshold: realised vol of SPY (computed
    #     INSIDE the strategy from primary close, works in MC paths).
    #   * stress_vix_threshold: VIX level (requires vix_series passed in,
    #     OPTIONAL -- live-only).
    #   * stress_credit_z_threshold: HYG/IEF credit-spread Z-score widening
    #     (requires hyg_series + ief_series, OPTIONAL -- live-only).
    stress_gate_enabled: bool = False
    stress_realised_vol_threshold: float = 0.20  # 20% ann realised vol
    stress_realised_vol_window: int = 20  # rolling window for the signal
    stress_vix_threshold: float | None = None  # e.g. 25.0; None disables
    stress_credit_z_threshold: float | None = None  # e.g. 2.0; None disables
    stress_credit_z_window: int = 60  # rolling window for HYG/IEF z-score

    # ── Step 5: drawdown circuit breaker (V3.5 lesson) ─────────────────────
    # Independent safety net that triggers on the strategy's OWN equity
    # drawdown (not on input vol or external regime). Catches:
    #   * Slow grinding declines where vol doesn't lead
    #   * Overnight gaps that blow through vol-target before it adjusts
    #
    # State machine (causal, computed on shifted equity series):
    #   * dd <= dd_breaker_haircut_threshold   -> position * dd_breaker_haircut_scale
    #   * dd <= dd_breaker_flat_threshold      -> position = 0 (all -> IEF) for
    #                                             dd_breaker_flat_bars
    #   * after flat period: re-enter at the haircut scale until DD recovers
    #     above dd_breaker_recovery_threshold (then full position).
    #
    # Set ``dd_breaker_enabled = False`` to disable (default).
    dd_breaker_enabled: bool = False
    dd_breaker_haircut_threshold: float = -0.10  # -10% DD -> reduce position
    dd_breaker_haircut_scale: float = 0.50  # to 50% of current position
    dd_breaker_flat_threshold: float = -0.15  # -15% DD -> flatten to IEF
    dd_breaker_flat_bars: int = 21  # ~1 month "cooling off"
    dd_breaker_recovery_threshold: float = -0.05  # back to full when DD recovers above -5%


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


def compute_stress_signal(
    primary_close: pd.Series,
    cfg: GemConfig,
    *,
    vix: pd.Series | None = None,
    hyg: pd.Series | None = None,
    ief_for_credit: pd.Series | None = None,
) -> pd.Series:
    """Compute the causal binary stress signal driving the Step 4 gate.

    Returns a Series aligned to ``primary_close.index`` with True where
    the strategy should "go defensive" (apply vol-target scaling) and
    False where it should deploy at max_leverage.

    Stress sources (logical OR):
      * realised_vol(primary, window) > cfg.stress_realised_vol_threshold
        -- always available, computed from the primary close itself.
        Works in MC paths (just needs the synthetic close series).
      * vix > cfg.stress_vix_threshold (if vix series provided and
        cfg.stress_vix_threshold is not None).
      * HYG/IEF credit-spread Z-score < -cfg.stress_credit_z_threshold
        (spread WIDENING -- HYG underperforming IEF -- is credit stress).
        Requires hyg + ief_for_credit and cfg.stress_credit_z_threshold.

    All inputs are shifted by 1 bar so the signal at t uses data through
    t-1 (causal). False is the default when stress_gate_enabled is False
    (i.e. the stress gate is a no-op).
    """
    idx = primary_close.index
    if not cfg.stress_gate_enabled:
        return pd.Series(False, index=idx, dtype=bool)

    # Realised vol of primary (typically SPY).
    bar_returns = primary_close.pct_change()
    rolling_std = (
        bar_returns.shift(1)
        .rolling(
            cfg.stress_realised_vol_window,
            min_periods=cfg.stress_realised_vol_window,
        )
        .std(ddof=1)
    )
    realised_vol = rolling_std * np.sqrt(BARS_PER_YEAR["D"])
    stress = (realised_vol > cfg.stress_realised_vol_threshold).fillna(False)

    # Optional VIX threshold.
    if vix is not None and cfg.stress_vix_threshold is not None:
        vix_aligned = vix.reindex(idx).ffill().shift(1)
        stress = stress | (vix_aligned > cfg.stress_vix_threshold).fillna(False)

    # Optional HYG/IEF credit-spread Z-score (negative -> widening).
    if hyg is not None and ief_for_credit is not None and cfg.stress_credit_z_threshold is not None:
        hyg_aligned = hyg.reindex(idx).ffill()
        ief_aligned = ief_for_credit.reindex(idx).ffill()
        # log ratio change captures HYG vs IEF relative performance.
        log_ratio = (np.log(hyg_aligned) - np.log(ief_aligned)).shift(1)
        roll_mean = log_ratio.rolling(
            cfg.stress_credit_z_window,
            min_periods=cfg.stress_credit_z_window,
        ).mean()
        roll_std = log_ratio.rolling(
            cfg.stress_credit_z_window,
            min_periods=cfg.stress_credit_z_window,
        ).std(ddof=1)
        z = (log_ratio - roll_mean) / roll_std
        # Negative Z = spread widening = HYG falling vs IEF = credit stress.
        stress = stress | (z < -cfg.stress_credit_z_threshold).fillna(False)

    return stress.astype(bool)


def gem_target_weights(
    closes: pd.DataFrame,
    cfg: GemConfig | None = None,
    *,
    vix: pd.Series | None = None,
    hyg: pd.Series | None = None,
    ief_for_credit: pd.Series | None = None,
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

    # ── Vol-target overlay (Step 3) + conditional stress gate (Step 4) ──
    # The overlay always applies when ann_vol_target is set; the stress
    # gate (if cfg.stress_gate_enabled) only ACTIVATES scaling during
    # stress bars. In calm bars the position is deployed at max_leverage.
    if cfg.ann_vol_target is not None and cfg.ann_vol_target > 0:
        stress_signal = compute_stress_signal(
            closes["SPY"], cfg, vix=vix, hyg=hyg, ief_for_credit=ief_for_credit
        )
        weights = _apply_vol_target(weights, closes, cfg, stress_signal=stress_signal)
    elif cfg.max_leverage > 1.0:
        # Leverage without vol target: scale risk-asset weights by max_leverage
        # unconditionally (the simple "always-levered" baseline).
        weights = _apply_static_leverage(weights, cfg)

    # ── Drawdown circuit breaker (Step 5) -- independent safety net ──────
    if cfg.dd_breaker_enabled:
        weights = _apply_dd_breaker(weights, closes, cfg)

    return weights


def _apply_vol_target(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    cfg: GemConfig,
    *,
    stress_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """Scale weights so trailing realised vol of the strategy hits the target.

    Vectorised: pandas rolling for vol, numpy elementwise for the scale.
    No Python loop over bars.

    If ``stress_signal`` is provided AND ``cfg.stress_gate_enabled`` is
    True (Step 4 mode):
      * On stress bars (stress=True): apply vol-target scaling as below.
      * On calm bars (stress=False):  deploy at cfg.max_leverage (no
        scaling down -- preserves total return in calm regimes).

    Otherwise (Step 3 mode, no stress gate): apply vol-target scaling
    on every bar.

    Causality:
      * Bar-returns are pct_change on closes -> known at end of bar t.
      * Unscaled strategy returns at t use weights at t (already shifted by 1).
      * Rolling vol uses returns through bar t.
      * Scale factor at bar t is derived from vol through t, then SHIFTED BY
        ONE BAR before multiplying weights -- so the scaling acts on the
        position FROM the next bar onwards. This preserves the existing
        shift(1) discipline.
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

    # Vol-target scale -- can be < 1 (scale down) or > 1 (lever up,
    # bounded by max_leverage). Where realised vol is zero or NaN, fall
    # back to 1.0 (full position).
    voltarget_scale = (cfg.ann_vol_target / realised_vol).clip(upper=cfg.max_leverage)
    voltarget_scale = voltarget_scale.fillna(1.0)

    # Apply stress gate: on calm bars the scale is held at max_leverage
    # (full deployment / levered as per cfg). On stress bars the vol-target
    # scale takes over.
    if stress_signal is not None and cfg.stress_gate_enabled:
        stress_aligned = stress_signal.reindex(weights.index).fillna(False).astype(bool)
        calm_scale = pd.Series(cfg.max_leverage, index=weights.index, dtype=float)
        scale = voltarget_scale.where(stress_aligned, calm_scale)
    else:
        scale = voltarget_scale

    # Causal shift -- decision uses data through t-1.
    scale = scale.shift(1).fillna(1.0)

    # Apply the scale to the active risk asset. Three cases per bar:
    #   * scale < 1.0:  reduce risk-asset weight, route leftover -> IEF.
    #   * scale == 1.0: no change.
    #   * scale > 1.0:  increase risk-asset weight beyond 1 (leverage via
    #                   MES futures in live); IEF unchanged (no shorting).
    risk_assets = ["SPY", "EFA"]
    scaled = weights.copy()
    risk_mask = weights[risk_assets].sum(axis=1) > 0  # rows where active asset is SPY/EFA

    if risk_mask.any():
        for col in risk_assets:
            scaled.loc[risk_mask, col] = weights.loc[risk_mask, col] * scale.loc[risk_mask]
        # When scale < 1: route the un-deployed fraction to IEF.
        leftover = (1.0 - scale).clip(lower=0.0)  # only positive (i.e. scale < 1)
        scaled.loc[risk_mask, "IEF"] = scaled.loc[risk_mask, "IEF"] + leftover.loc[risk_mask]

    # If we're already in IEF (defensive switch), leave as-is.
    return scaled


def _apply_static_leverage(
    weights: pd.DataFrame,
    cfg: GemConfig,
) -> pd.DataFrame:
    """Unconditionally scale the active risk-asset weight by max_leverage.

    Used when ``cfg.ann_vol_target is None`` but ``cfg.max_leverage > 1``
    -- the "always-levered" baseline (no vol-target risk management).
    The defensive IEF position is NOT levered.
    """
    risk_assets = ["SPY", "EFA"]
    scaled = weights.copy()
    risk_mask = weights[risk_assets].sum(axis=1) > 0
    if risk_mask.any():
        for col in risk_assets:
            scaled.loc[risk_mask, col] = weights.loc[risk_mask, col] * cfg.max_leverage
    return scaled


def _apply_dd_breaker(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    cfg: GemConfig,
) -> pd.DataFrame:
    """Drawdown circuit breaker (Step 5 / V3.5 lesson).

    Computes the strategy's running equity from the PRE-BREAKER weights
    + bar returns, then derives an EXTRA scale factor in {0, haircut, 1}
    applied on top. Vectorised cumulative scan -- no Python per-bar loop.

    State machine (causal: dd[t] uses returns through t-1, so the scale
    at bar t is applied to weights[t] which already earn ret[t]):

      * cooling_bars > 0:                scale = 0.0  (in IEF; flat)
      * dd <= flat_threshold:            scale = 0.0  + start cooling
      * dd <= haircut_threshold:         scale = haircut_scale
      * dd > recovery_threshold:         scale = 1.0  (full position)
      * otherwise (haircut zone holds):  scale = haircut_scale

    The "leftover" capital when scale < 1 goes to IEF (consistent with
    how _apply_vol_target routes leftover).
    """
    risk_assets = ["SPY", "EFA"]
    closes_aligned = closes[list(GEM_UNIVERSE)].astype(float).reindex(weights.index)
    bar_returns = closes_aligned.pct_change().fillna(0.0)
    # Pre-breaker strategy returns -- the equity curve we'd have WITHOUT the breaker.
    pre_strat_ret = (weights * bar_returns).sum(axis=1)
    equity = (1.0 + pre_strat_ret).cumprod()
    hwm = equity.cummax()
    dd = equity / hwm - 1.0
    # Causal: decision at t uses dd through t-1.
    dd_shifted = dd.shift(1).fillna(0.0).to_numpy()

    n = len(weights)
    scale = np.ones(n, dtype=float)
    cooling_remaining = 0
    in_haircut = False  # latched once dd breaches haircut_threshold; cleared at recovery.
    for i in range(n):
        d = float(dd_shifted[i])
        if cooling_remaining > 0:
            scale[i] = 0.0
            cooling_remaining -= 1
            # After cooling completes, enter haircut state until recovery.
            if cooling_remaining == 0:
                in_haircut = True
            continue
        if d <= cfg.dd_breaker_flat_threshold:
            scale[i] = 0.0
            cooling_remaining = max(0, cfg.dd_breaker_flat_bars - 1)
            in_haircut = True
            continue
        if d <= cfg.dd_breaker_haircut_threshold:
            scale[i] = cfg.dd_breaker_haircut_scale
            in_haircut = True
            continue
        if d > cfg.dd_breaker_recovery_threshold and in_haircut:
            in_haircut = False
        if in_haircut:
            scale[i] = cfg.dd_breaker_haircut_scale
        else:
            scale[i] = 1.0

    scale_s = pd.Series(scale, index=weights.index)
    scaled = weights.copy()
    risk_mask = weights[risk_assets].sum(axis=1) > 0
    if risk_mask.any():
        for col in risk_assets:
            scaled.loc[risk_mask, col] = weights.loc[risk_mask, col] * scale_s.loc[risk_mask]
        # Route undeployed capital to IEF (safe asset).
        leftover = (1.0 - scale_s).clip(lower=0.0)
        scaled.loc[risk_mask, "IEF"] = scaled.loc[risk_mask, "IEF"] + leftover.loc[risk_mask]
    return scaled


def gem_returns(
    closes: pd.DataFrame,
    cfg: GemConfig | None = None,
    *,
    vix: pd.Series | None = None,
    hyg: pd.Series | None = None,
    ief_for_credit: pd.Series | None = None,
    cost_bps_per_turnover: float = 0.0,
) -> pd.Series:
    """Compute the per-day GEM strategy returns (geometric, simple-return).

    Implementation contract::

        ret_t = sum_n weight[t, n] * (close[t, n] / close[t-1, n] - 1)
              - cost_bps_per_turnover * turnover[t] / 1e4

    where weight is the CAUSAL shifted weight from ``gem_target_weights``
    and turnover[t] = sum(|weight[t] - weight[t-1]|).

    Transaction costs (``cost_bps_per_turnover``):
      * Default 0 = costless (legacy behaviour; backward-compatible).
      * ETF realistic default: 1.5 bps per unit of turnover. Derived from
        ``COST_US_ETF_LIQUID`` (spread 1.0 + slip 0.5 = 1.5 bps one-way).
        Commission ~$0.35/side on a $100 share is sub-bp; folded in.
      * MES futures: ~1.0 bps per unit of turnover.

    Turnover semantics: ``|weight_new - weight_old|`` per asset, summed.
    A clean 100% SPY -> 100% IEF rotation has turnover = 2.0 (1.0 sold
    SPY + 1.0 bought IEF). At 1.5 bps that's a 3 bp round-trip cost,
    matching the realistic US-ETF round-trip.

    Optional ``vix`` / ``hyg`` / ``ief_for_credit`` series feed the Step 4
    stress signal (when ``cfg.stress_gate_enabled``). Absent series silently
    skip their stress component.
    """
    weights = gem_target_weights(closes, cfg=cfg, vix=vix, hyg=hyg, ief_for_credit=ief_for_credit)
    closes_aligned = closes[list(GEM_UNIVERSE)].astype(float).reindex(weights.index)
    bar_returns = closes_aligned.pct_change()
    # Sanity: at any row, weights sum to either 0 (warmup) or some value
    # in [0, max_leverage]. Not enforced here.
    strat_ret = (weights * bar_returns).sum(axis=1)
    if cost_bps_per_turnover > 0:
        # Turnover[t] = sum over assets of |w[t] - w[t-1]|. This counts
        # both legs of a rotation (sell + buy) so total is 2 for a clean
        # swap; cost_bps_per_turnover is the ONE-WAY rate.
        turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
        cost = turnover * cost_bps_per_turnover / 1e4
        strat_ret = strat_ret - cost
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
