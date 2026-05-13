"""Samir-Stack V3 — pure equity, regime-gated MES futures with layered defenses.

Three independently-toggled defense layers on top of the VIX-HMM Layer-1+2 base:

  LAYER A (asymmetric hysteresis on the HMM gate)
    - score_enter_threshold and score_exit_threshold can differ.
    - Slow to enter (high confidence required), fast to exit (moderate
      evidence of regime shift is enough).
    - When equal (default 0.5/0.5), behaviour reduces to plain Layer-1+2.

  LAYER B (momentum confluence)
    - Additional indicators must agree before deployment:
        dd_velocity (21-day SPY return, normalised to [0,1] via clip)
        12-1 momentum (Asness convention, shift 21 to 252)
        trend (close > 200-day SMA)
    - All must clear their thresholds for deploy_intent. Any can trigger
      exit. AND on entry, OR on exit.
    - When disabled (use_momentum_confluence=False), no additional gates.

  LAYER C (DD circuit breaker)
    - Mechanical failsafe: regardless of HMM and momentum signals, if the
      strategy's own equity drawdown (from HWM) breaches ``dd_kill``,
      force cash. Stay in cash until ``regime_score >=
      dd_re_entry_score`` for ``dd_re_entry_bars`` consecutive bars; on
      exit, reset HWM to current equity (prevents permanent
      under-water-lock-out).
    - When disabled (dd_kill <= 0 or >= 1), no breaker.

Each layer is independent. Each addresses a different failure mode:
  A = speed of evidence accumulation (HMM lag on regime transitions)
  B = single-signal blind spots (VIX-only misses price-led shocks)
  C = the unknown unknowns (gate-failure failsafe)

Per the V3 design directive and the Layer-1+2 MC verdict (P(MaxDD>50%)
= 69% at L=2): the pure HMM gate is informative but at L=2 the gate's
LAG during regime transitions produces catastrophic drawdowns in
bootstrap-resampled paths. Layer C alone is hypothesised to drop
P(MaxDD>50%) from 69% to ~5% by capping mechanically; B adds further
robustness by catching shocks the HMM is slow on; A tightens the speed
of the HMM gate itself.

Defaults
--------
Defaults preserve the Layer-1+2 behaviour exactly (all defenses off).
The validation harness opts each one in to measure incremental impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.samir_stack.engines import FuturesEngine
from research.samir_stack.synthetic_3x import funding_series
from titan.research.metrics import BARS_PER_YEAR, annualize_vol, sharpe


@dataclass
class V3Config:
    """Configuration for the V3 binary-regime equity strategy with
    optional Layer-C/B/A defenses.

    Conservative defaults: all defenses OFF (reduces to Layer-1+2 exactly).
    Opt in by setting the relevant fields.
    """

    # ── Base Layer 1+2 ───────────────────────────────────────────────
    score_threshold: float = 0.5
    """Above this, deploy. Below or equal, hold cash. Pre-registered
    canonical choice = 0.5 (HMM filtering majority).
    DEPRECATED in favour of score_enter_threshold / score_exit_threshold;
    if those are None they fall back to this single value."""

    L_target: float = 2.0
    """Constant leverage when deployed. Pre-registered sweep: {2, 3, 4}."""

    re_entry_quiet_bars: int = 0
    """Bars after exiting to cash before re-entry permitted. Default 0
    for the minimum viable version. Set to 20 for v2-style quiet period."""

    transaction_cost_bps: float = 2.0
    """Round-trip cost per state change in basis points. MES round-trip
    commissions on IBKR ≈ 5bp at typical NAV; this is per-side."""

    # Futures engine pass-through parameters
    dividend_yield: float = 0.015
    margin_pct: float = 0.06
    rolls_per_year: int = 4
    roll_slippage_bps: float = 5.0

    # ── Layer A: asymmetric hysteresis on the HMM gate ──────────────
    # When both are None, behaviour falls back to ``score_threshold`` for
    # symmetric gating. To enable asymmetric hysteresis, set:
    #   score_enter_threshold = 0.7  (slow entry: requires high confidence)
    #   score_exit_threshold  = 0.5  (fast exit: moderate evidence is enough)
    score_enter_threshold: float | None = None
    score_exit_threshold: float | None = None

    # ── Layer B: momentum confluence (additional entry gates) ────────
    use_momentum_confluence: bool = False
    """When True, additional indicators must all agree benign for entry."""

    mom_12_1_threshold: float = 0.0
    """12-1 momentum (price[t-21] / price[t-252] - 1) must exceed this."""

    dd_velocity_threshold: float = -0.05
    """21-day SPY return must exceed this (cumulative, NOT annualised).
    -5% means "no worse than -5% over the last 21 days"."""

    require_above_200d_sma: bool = True
    """If True, close must be > 200-day SMA for entry."""

    # ── Layer C: DD circuit breaker ──────────────────────────────────
    # When dd_kill is in (0, 1), the breaker is active.
    dd_kill: float = 0.0
    """Strategy drawdown (positive number, e.g. 0.15 = 15%) at which to
    force cash. Set to 0.0 to disable."""

    dd_re_entry_score: float = 0.70
    """After kill, the regime score must reach this level."""

    dd_re_entry_bars: int = 5
    """...for this many consecutive bars before re-entry permitted."""


@dataclass
class _V3State:
    equity: float = 1.0
    hwm: float = 1.0
    dd_state: str = "normal"  # "normal" or "killed"
    dd_recovery_count: int = 0
    quiet_counter: int = 999
    prev_deployed: int = 0
    # B layer running buffers (for SMA / momentum lookups — pre-computed)
    pass


def _build_momentum_features(spy: pd.Series) -> pd.DataFrame:
    """Pre-compute the Layer-B momentum/trend features on the full series."""
    sma200 = spy.rolling(200, min_periods=200).mean()
    above_200 = (spy > sma200).astype(float)
    # 12-1 momentum: price[t-21] / price[t-252] - 1 (Asness convention).
    # Note: this uses past data only — the formula uses bars t-252 to t-21,
    # i.e. shifts the price back, so it's strictly causal.
    mom_12_1 = spy.shift(21) / spy.shift(252) - 1.0
    # 21-day SPY return (dd_velocity — captures fast drawdown).
    dd_velocity = spy.pct_change(21)
    return pd.DataFrame(
        {
            "above_200d_sma": above_200,
            "mom_12_1": mom_12_1,
            "dd_velocity": dd_velocity,
        }
    )


def _effective_enter_threshold(cfg: V3Config) -> float:
    return (
        cfg.score_enter_threshold if cfg.score_enter_threshold is not None else cfg.score_threshold
    )


def _effective_exit_threshold(cfg: V3Config) -> float:
    return cfg.score_exit_threshold if cfg.score_exit_threshold is not None else cfg.score_threshold


def run_v3_strategy(
    spy_close: pd.Series,
    regime_score: pd.Series,
    cfg: V3Config,
) -> pd.DataFrame:
    """Simulate the V3 strategy with optional Layer-C/B/A defenses.

    Returns DataFrame with columns:
      score, deployed (0/1), L_applied, equity, ret_strategy,
      ret_levered, ret_cash, transition, tx_cost, dd_state,
      dd_from_hwm, gate_a_intent, gate_b_intent, gate_c_active
    """
    common = spy_close.index.intersection(regime_score.index)
    spy = spy_close.reindex(common)
    score = regime_score.reindex(common)

    # Pre-compute the two engine return series.
    engine = FuturesEngine(
        dividend_yield=cfg.dividend_yield,
        margin_pct=cfg.margin_pct,
        rolls_per_year=cfg.rolls_per_year,
        roll_slippage_bps=cfg.roll_slippage_bps,
    )
    ret_levered_full = engine.daily_returns(spy, leverage=cfg.L_target).reindex(common).fillna(0.0)
    fund = funding_series(common).reindex(common).ffill().fillna(0.04)
    ret_cash_full = (fund / 252.0).astype(float)

    # Layer-B feature pre-compute (uses past closes only — see docstring).
    mom_feats = _build_momentum_features(spy)

    # Lagged inputs: signal at t-1 decides position at t (shift discipline).
    score_lag = score.shift(1).fillna(0.0)
    above_200_lag = mom_feats["above_200d_sma"].shift(1).fillna(0.0)
    mom_lag = mom_feats["mom_12_1"].shift(1).fillna(0.0)
    dd_vel_lag = mom_feats["dd_velocity"].shift(1).fillna(0.0)

    enter_thr = _effective_enter_threshold(cfg)
    exit_thr = _effective_exit_threshold(cfg)
    dd_breaker_active = 0.0 < cfg.dd_kill < 1.0

    # State series for output
    n = len(common)
    deployed_arr = np.zeros(n, dtype=int)
    equity_arr = np.empty(n, dtype=float)
    dd_state_arr = np.empty(n, dtype=object)
    dd_from_hwm_arr = np.empty(n, dtype=float)
    gate_a_intent_arr = np.zeros(n, dtype=int)
    gate_b_intent_arr = np.zeros(n, dtype=int)
    gate_c_active_arr = np.zeros(n, dtype=int)
    tx_cost_arr = np.zeros(n, dtype=float)

    s = _V3State()
    cost_per_transition = cfg.transaction_cost_bps / 10_000.0

    for i in range(n):
        # ── Layer A: asymmetric-hysteresis HMM gate ──
        score_now = float(score_lag.iat[i])
        if s.prev_deployed == 0:
            gate_a = 1 if score_now > enter_thr else 0
        else:
            gate_a = 1 if score_now > exit_thr else 0
        gate_a_intent_arr[i] = gate_a

        # ── Layer B: momentum confluence ──
        if cfg.use_momentum_confluence:
            sma_ok = (above_200_lag.iat[i] > 0.5) or not cfg.require_above_200d_sma
            mom_ok = float(mom_lag.iat[i]) > cfg.mom_12_1_threshold
            dd_ok = float(dd_vel_lag.iat[i]) > cfg.dd_velocity_threshold
            if s.prev_deployed == 0:
                # Entry: AND across all
                gate_b = 1 if (sma_ok and mom_ok and dd_ok) else 0
            else:
                # Exit: OR — any failure triggers exit
                gate_b = 1 if (sma_ok and mom_ok and dd_ok) else 0
                # Note: for binary deploy/cash this is mathematically the same
                # as AND; the asymmetry only matters if there's intermediate
                # state. Documented here for future tier-ladder extension.
        else:
            gate_b = 1
        gate_b_intent_arr[i] = gate_b

        # ── Combined deploy intent (A AND B) ──
        deploy_intent = 1 if (gate_a and gate_b) else 0

        # ── Re-entry quiet bars (from layer 1+2) ──
        if cfg.re_entry_quiet_bars > 0:
            if (
                deploy_intent == 1
                and s.prev_deployed == 0
                and s.quiet_counter < cfg.re_entry_quiet_bars
            ):
                deploy_intent = 0
                s.quiet_counter += 1

        # ── Layer C: DD circuit breaker override ──
        if dd_breaker_active:
            if s.dd_state == "killed":
                # Force cash. Check for recovery.
                deploy_intent = 0
                if score_now >= cfg.dd_re_entry_score:
                    s.dd_recovery_count += 1
                    if s.dd_recovery_count >= cfg.dd_re_entry_bars:
                        s.dd_state = "normal"
                        s.hwm = s.equity  # reset HWM
                        s.dd_recovery_count = 0
                else:
                    s.dd_recovery_count = 0
                gate_c_active_arr[i] = 1

        deployed = deploy_intent
        deployed_arr[i] = deployed
        dd_state_arr[i] = s.dd_state

        # ── Apply transaction cost on transitions ──
        transition = abs(deployed - s.prev_deployed)
        tx_cost_arr[i] = cost_per_transition * transition

        # ── Compute today's return using deployed at this bar ──
        if deployed:
            ret = float(ret_levered_full.iat[i]) - tx_cost_arr[i]
        else:
            ret = float(ret_cash_full.iat[i]) - tx_cost_arr[i]

        # ── Update state ──
        s.equity *= 1.0 + ret
        equity_arr[i] = s.equity
        if s.equity > s.hwm:
            s.hwm = s.equity
        dd_from_hwm = (s.equity - s.hwm) / s.hwm if s.hwm > 0 else 0.0
        dd_from_hwm_arr[i] = dd_from_hwm

        # ── Check for DD kill (only when in normal state) ──
        if dd_breaker_active and s.dd_state == "normal" and dd_from_hwm <= -cfg.dd_kill:
            s.dd_state = "killed"
            s.dd_recovery_count = 0
            # Note: kill fires AFTER the day's return is realised; we
            # exit at the next bar. This matches V2's semantics.

        # ── Update quiet counter ──
        if cfg.re_entry_quiet_bars > 0:
            if deployed == 0 and s.prev_deployed == 1:
                s.quiet_counter = 0  # just exited
            elif deployed == 0:
                s.quiet_counter += 1
            else:
                s.quiet_counter = 999

        s.prev_deployed = deployed

    # Assemble outputs.
    deployed_series = pd.Series(deployed_arr, index=common, name="deployed")
    ret_strategy = pd.Series(
        [
            float(ret_levered_full.iat[i]) - tx_cost_arr[i]
            if deployed_arr[i]
            else float(ret_cash_full.iat[i]) - tx_cost_arr[i]
            for i in range(n)
        ],
        index=common,
        name="ret_strategy",
    )
    return pd.DataFrame(
        {
            "score": score,
            "deployed": deployed_series,
            "L_applied": deployed_series.astype(float) * cfg.L_target,
            "ret_levered": ret_levered_full,
            "ret_cash": ret_cash_full,
            "transition": pd.Series(
                [abs(deployed_arr[i] - (deployed_arr[i - 1] if i > 0 else 0)) for i in range(n)],
                index=common,
            ),
            "tx_cost": pd.Series(tx_cost_arr, index=common),
            "ret_strategy": ret_strategy,
            "equity": pd.Series(equity_arr, index=common),
            "dd_state": pd.Series(dd_state_arr, index=common),
            "dd_from_hwm": pd.Series(dd_from_hwm_arr, index=common),
            "gate_a_intent": pd.Series(gate_a_intent_arr, index=common),
            "gate_b_intent": pd.Series(gate_b_intent_arr, index=common),
            "gate_c_active": pd.Series(gate_c_active_arr, index=common),
        },
        index=common,
    )


def summarize_v3_run(df: pd.DataFrame) -> dict:
    """Headline metrics for a V3 run."""
    rets = df["ret_strategy"].dropna()
    eq = df["equity"].dropna()
    if len(rets) < 2:
        return {"error": "insufficient bars"}
    n_y = len(rets) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / n_y) - 1.0)
    vol = annualize_vol(float(rets.std(ddof=1)), periods_per_year=BARS_PER_YEAR["D"])
    sh = sharpe(rets.to_numpy(), periods_per_year=BARS_PER_YEAR["D"])
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    frac_deployed = float(df["deployed"].mean())
    n_transitions = int(df["transition"].sum())
    n_kill_bars = int((df["dd_state"] == "killed").sum()) if "dd_state" in df.columns else 0
    return {
        "n_years": round(n_y, 2),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "sharpe": round(sh, 3),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 3),
        "frac_deployed": round(frac_deployed, 3),
        "n_transitions": n_transitions,
        "transitions_per_year": round(n_transitions / n_y, 2),
        "frac_in_kill_state": round(n_kill_bars / len(rets), 4),
    }
