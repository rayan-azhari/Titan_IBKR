"""Samir-Stack equity-sleeve state machine.

Discrete leverage tiers driven by regime score, with hysteresis to avoid
whipsaw. DD circuit breaker overlay for failsafe (catches the 2020 COVID
and 2022 grinding-bear cases the regime classifier misses or catches late).

State variables:
    tier         — 0=cash, 1=1x, 2=2x, 3=3x (or up to L_max)
    dd_state     — 'normal' | 'throttled' | 'killed'
    last_score   — for hysteresis
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.samir_stack.synthetic_3x import synthetic_leveraged_returns


@dataclass
class StrategyConfig:
    """Configuration for the Samir-Stack equity sleeve."""

    L_max: float = 3.0
    """Maximum leverage tier. Run sweeps over {1, 2, 3, 4}."""

    # Regime tier boundaries (lower edges)
    tier_thresholds: tuple[float, ...] = (0.30, 0.50, 0.75)
    """Score thresholds for transitioning UP into tiers 1, 2, 3.

    score >= 0.30 → at least 1x
    score >= 0.50 → at least 2x
    score >= 0.75 → 3x (= L_max)
    score <  0.30 → cash
    """

    hysteresis_buffer: float = 0.05
    """Extra score required to step UP a tier (whipsaw avoidance).

    Going DOWN uses the bare threshold; going UP requires threshold+buffer.
    Example: from 1x at score=0.49, must reach 0.55 (=0.50+0.05) to step
    up to 2x. Going down from 2x uses bare 0.50.
    """

    re_entry_quiet_bars: int = 20
    """After exiting to cash, require this many bars of score >= 0.50
    before re-entering. Avoids 2008-style false-dawn whipsaws.
    """

    # DD circuit breaker
    dd_throttle: float = 0.12
    """Drawdown level (negative side, expressed positive) at which leverage
    is capped at 1x regardless of regime score. 0.12 = 12% drawdown."""

    dd_kill: float = 0.18
    """Drawdown level at which strategy goes to full cash regardless."""

    dd_re_entry_score: float = 0.70
    """Required regime score to exit a DD-throttle or DD-kill event.
    On exit, the HWM is reset to current equity so the next drawdown is
    measured from the post-recovery level (avoids deadlock where cash
    can't recover the prior HWM)."""

    dd_re_entry_bars: int = 5
    """Number of consecutive bars at score >= dd_re_entry_score required
    to exit a DD-throttle or DD-kill state."""

    # Costs (per state change)
    transaction_cost_bps: float = 5.0
    """Round-trip transaction cost on each leverage *change*, applied as a
    fraction of position notional change. 5 bps = 0.05% per change.
    Approximates spread + slippage on the leveraged ETF."""

    leverage_ter_annual: float = 0.0075
    """Annual expense ratio for leveraged ETF. Applied via the synthetic
    Lx return generator."""


@dataclass
class StrategyState:
    """Mutable state during a backtest."""

    tier: float = 0.0
    """Current leverage tier (0=cash, 1=1x, ..., 3=3x)."""

    dd_state: str = "normal"
    """One of 'normal', 'throttled', 'killed'."""

    quiet_bars: int = 9999
    """Counter for re-entry after cash exit (regime-driven)."""

    dd_recovery_bars: int = 0
    """Counter of consecutive bars at score >= dd_re_entry_score while
    in throttled/killed state. Resets to 0 if score drops."""

    hwm: float = 1.0
    """High-water-mark of the equity curve. Reset to current equity when
    the strategy exits a DD-throttle or DD-kill state — this avoids the
    deadlock where cash (zero return) can't recover the prior HWM."""

    history: list[dict] = field(default_factory=list)


def _target_tier_from_score(score: float, cfg: StrategyConfig, current_tier: float) -> float:
    """Convert score → target tier with hysteresis on UP transitions.

    Caps at L_max. Cash if score below first threshold.
    """
    t1, t2, t3 = cfg.tier_thresholds
    buf = cfg.hysteresis_buffer

    # First decide what tier the score *would* support
    # Going UP requires threshold + buffer
    # Going DOWN uses bare threshold
    target = current_tier
    # Step DOWN tests
    if score < t1:
        target = 0.0
    elif score < t2 and current_tier > 1.0:
        target = 1.0
    elif score < t3 and current_tier > 2.0:
        target = 2.0
    # Step UP tests (require threshold + buffer)
    if current_tier == 0.0 and score >= t1 + buf:
        target = 1.0
    if current_tier <= 1.0 and score >= t2 + buf:
        target = 2.0
    if current_tier <= 2.0 and score >= t3 + buf:
        target = 3.0

    return min(target, cfg.L_max)


def run_strategy(
    spy_close: pd.Series,
    regime_score: pd.Series,
    cfg: StrategyConfig,
    *,
    fwd_funding: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate the Samir-Stack equity sleeve.

    Returns a DataFrame indexed by date with columns:
        score, target_tier, applied_tier, dd_state, dd, equity,
        ret_strategy, ret_underlying_lx, traded

    `score` is the regime score (input), `target_tier` is what the
    state machine wants given regime+hysteresis, `applied_tier` is what
    actually gets traded after the DD overlay. `traded` is True when
    the applied tier changes from the previous bar (used to apply
    transaction costs).
    """
    # Align inputs
    df = pd.concat([spy_close.rename("close"), regime_score.rename("score")], axis=1).dropna()
    if len(df) < 100:
        raise ValueError(f"insufficient overlap: {len(df)} bars")

    # Pre-compute Lx returns for each tier {1, 2, 3} so we can index by
    # applied_tier on each bar. tier=0 → 0 return.
    tier_returns: dict[float, pd.Series] = {0.0: pd.Series(0.0, index=df.index)}
    for L in (1.0, 2.0, 3.0, 4.0):
        if L > cfg.L_max:
            break
        # synthetic_leveraged_returns yields NaN on the first bar (pct_change)
        rets_L = (
            synthetic_leveraged_returns(
                df["close"],
                leverage=L,
                ter_annual=cfg.leverage_ter_annual if L > 1.0 else 0.0,
                funding_annual=fwd_funding,
            )
            .reindex(df.index)
            .fillna(0.0)
        )
        tier_returns[L] = rets_L

    state = StrategyState()
    rows = []
    equity = 1.0

    for i, ts in enumerate(df.index):
        score = df["score"].iat[i]

        # 1) Compute target tier from regime score (hysteresis)
        target = _target_tier_from_score(score, cfg, state.tier)

        # 2) Re-entry quiet-bar gate after a cash exit
        # If we're currently at 0 and target>0, require quiet_bars
        if state.tier == 0.0 and target > 0.0:
            if state.quiet_bars < cfg.re_entry_quiet_bars:
                target = 0.0

        # 3) DD circuit breaker overlay (secondary to regime classifier)
        # Recovery from throttled/killed is regime-driven: requires N
        # consecutive bars at score >= dd_re_entry_score. On exit, HWM
        # is reset to current equity (next DD measured from new floor).
        dd = (equity - state.hwm) / state.hwm  # ≤ 0
        if state.dd_state == "killed":
            target = 0.0
            if score >= cfg.dd_re_entry_score:
                state.dd_recovery_bars += 1
                if state.dd_recovery_bars >= cfg.dd_re_entry_bars:
                    state.dd_state = "normal"
                    state.hwm = equity  # reset HWM at recovery point
                    state.dd_recovery_bars = 0
                    # Allow target to be set by regime; don't force 0 next bar
                    target = _target_tier_from_score(score, cfg, state.tier)
            else:
                state.dd_recovery_bars = 0
        elif state.dd_state == "throttled":
            if abs(dd) >= cfg.dd_kill:
                state.dd_state = "killed"
                target = 0.0
                state.dd_recovery_bars = 0
            elif score >= cfg.dd_re_entry_score:
                state.dd_recovery_bars += 1
                if state.dd_recovery_bars >= cfg.dd_re_entry_bars:
                    state.dd_state = "normal"
                    state.hwm = equity
                    state.dd_recovery_bars = 0
                else:
                    target = min(target, 1.0)
            else:
                state.dd_recovery_bars = 0
                target = min(target, 1.0)
        else:  # normal
            if abs(dd) >= cfg.dd_kill:
                state.dd_state = "killed"
                target = 0.0
                state.dd_recovery_bars = 0
            elif abs(dd) >= cfg.dd_throttle:
                state.dd_state = "throttled"
                target = min(target, 1.0)
                state.dd_recovery_bars = 0

        # 4) Apply tier change with transaction cost
        applied = target
        traded = applied != state.tier
        tc = 0.0
        if traded:
            # Cost is proportional to absolute notional change
            notional_change = abs(applied - state.tier)
            tc = (cfg.transaction_cost_bps / 10000.0) * notional_change

        # 5) Compute return for this bar based on PREVIOUS bar's tier
        # (we trade at close, earn next bar — standard shift discipline)
        if i == 0:
            ret_bar = 0.0
        else:
            prev_tier = state.tier  # tier we held going into this bar
            ret_bar = tier_returns[prev_tier].iat[i] - tc

        equity *= 1.0 + ret_bar
        if equity > state.hwm:
            state.hwm = equity

        rows.append(
            {
                "date": ts,
                "score": score,
                "target_tier": target,
                "applied_tier": applied,
                "dd_state": state.dd_state,
                "dd": dd,
                "equity": equity,
                "ret_strategy": ret_bar,
                "transaction_cost": tc,
                "traded": traded,
            }
        )

        # 6) Update state
        # quiet_bars semantics: while in cash, count consecutive bars where
        # the regime score is benign-enough to consider re-entry (>= 0.50).
        # Reset only when score drops below 0.30 (deep hostile). When NOT in
        # cash, the counter is parked at a high value so a subsequent exit
        # doesn't artificially block immediate re-entry on whipsaw.
        if applied == 0.0:
            if score >= 0.50:
                state.quiet_bars += 1
            elif score < 0.30:
                state.quiet_bars = 0
        else:
            state.quiet_bars = 9999

        state.tier = applied

    out = pd.DataFrame(rows).set_index("date")
    return out


def annual_trade_count(strategy_df: pd.DataFrame) -> float:
    """Estimate annual trade count from `traded` column."""
    n_years = len(strategy_df) / 252.0
    return float(strategy_df["traded"].sum() / n_years)


def summarize(strategy_df: pd.DataFrame) -> dict:
    """Quick performance summary."""
    rets = strategy_df["ret_strategy"]
    eq = strategy_df["equity"]
    n_years = len(rets) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(252))
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    return {
        "n_years": round(n_years, 2),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "max_dd": round(maxdd, 4),
        "sharpe": round(float(sharpe), 3),
        "calmar": round(calmar, 3),
        "trades_per_year": round(annual_trade_count(strategy_df), 1),
        "frac_in_cash": round(float((strategy_df["applied_tier"] == 0.0).mean()), 3),
        "frac_in_full_lev": round(
            float((strategy_df["applied_tier"] >= strategy_df["applied_tier"].max() - 0.01).mean()),
            3,
        ),
    }
