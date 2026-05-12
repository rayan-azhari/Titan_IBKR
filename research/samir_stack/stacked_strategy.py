"""Stacked strategy: 40% leveraged equity + 60% bonds, regime-gated.

This is the user's original construct. Per-tier composition:

    tier 3 (most benign): equity_weight × 3x SPY + bond_weight × IEF
    tier 2:               equity_weight × 2x SPY + bond_weight × IEF
    tier 1:               equity_weight × 1x SPY + bond_weight × IEF
    tier 0 (hostile):     100% cash (both sleeves out)

Default split: 40% equity / 60% bonds. Rebalanced daily at the
sleeve-weight level. The leveraged ETF itself is daily-rebalanced (that's
where the vol drag comes from).

Optional: independent bond regime gate. When equity is benign but bonds
are hostile (e.g., 2022 rate shock), bond sleeve also goes to cash.

Phase 2 of the 2026-05-12 remediation introduced ``equity_engine`` and
``bond_sleeve`` kwargs to ``run_stacked_strategy``. They default to None,
in which case the function behaves exactly as before (synthetic 3x ETF
+ static IEF return via ``ief.pct_change()``). Passing an explicit engine
or sleeve overrides the corresponding default — the state machine itself
is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from research.samir_stack.capitulation import (
    CapitulationConfig,
    CapitulationState,
    detect_capitulation,
    detect_stabilisation,
)
from research.samir_stack.synthetic_3x import synthetic_leveraged_returns

if TYPE_CHECKING:
    from research.samir_stack.engines import BondSleeve, EquityEngine


@dataclass
class StackedConfig:
    """Configuration for the 40/60 stacked strategy."""

    # Sleeve weights
    equity_weight: float = 0.40
    bond_weight: float = 0.60

    # Equity sleeve leverage tiers — number of thresholds determines L_max ceiling
    L_max: float = 3.0
    tier_thresholds: tuple[float, ...] = (0.30, 0.50, 0.75)
    """Score thresholds for entering tier 1, 2, 3, ... in order.
    Length must equal int(L_max). For L_max=4, pass 4 thresholds e.g.
    (0.30, 0.50, 0.75, 0.90)."""
    hysteresis_buffer: float = 0.05
    re_entry_quiet_bars: int = 20

    # DD circuit breaker (applied to TOTAL portfolio, not just equity sleeve)
    dd_throttle: float = 0.10
    dd_kill: float = 0.15
    dd_re_entry_score: float = 0.70
    dd_re_entry_bars: int = 5

    # Costs
    transaction_cost_bps: float = 5.0
    leverage_ter_annual: float = 0.0075

    # Bond sleeve (optional independent regime gate)
    bond_gate_enabled: bool = False
    bond_ma_window: int = 200
    bond_velocity_threshold: float = -0.06  # -6% in 30 days = bond hostile

    # Capitulation overlay (optional — default off preserves v1 behaviour)
    capitulation: CapitulationConfig | None = None


def _equity_target_tier(score: float, cfg: StackedConfig, current_tier: float) -> float:
    """Convert score → equity tier with hysteresis (variable number of tiers).

    Each threshold ``t_k`` is the lower bound to enter tier ``k+1``. Going
    UP requires ``t_k + buffer``; going DOWN uses bare ``t_k``. Below
    ``t_0`` → cash (tier 0).
    """
    thresholds = cfg.tier_thresholds
    buf = cfg.hysteresis_buffer
    target = current_tier

    # Step DOWN: drop to highest tier whose threshold is <= score
    if score < thresholds[0]:
        target = 0.0
    else:
        for k in range(len(thresholds) - 1, 0, -1):
            if score < thresholds[k] and current_tier > k:
                target = float(k)
                break
    # Step UP: enter highest tier whose threshold + buffer is <= score
    if current_tier == 0.0 and score >= thresholds[0] + buf:
        target = 1.0
    for k in range(1, len(thresholds)):
        if current_tier <= k and score >= thresholds[k] + buf:
            target = float(k + 1)

    return min(target, cfg.L_max)


def _bond_benign(tlt_close: pd.Series, ts: pd.Timestamp, cfg: StackedConfig) -> bool:
    """Optional bond regime gate.

    Returns True if bonds appear benign. Uses 200d MA on TLT plus a
    rate-velocity check (rapid TLT drop = rate shock = hostile).
    """
    sub = tlt_close.loc[:ts]
    if len(sub) < cfg.bond_ma_window + 30:
        return True  # warmup
    ma = sub.iloc[-cfg.bond_ma_window :].mean()
    above_ma = sub.iloc[-1] > ma
    velocity_30d = sub.iloc[-1] / sub.iloc[-30] - 1.0
    return bool(above_ma and velocity_30d > cfg.bond_velocity_threshold)


def run_stacked_strategy(
    spy_close: pd.Series,
    ief_close: pd.Series,
    regime_score: pd.Series,
    cfg: StackedConfig,
    *,
    tlt_close: pd.Series | None = None,
    indicator_panel: pd.DataFrame | None = None,
    equity_engine: "EquityEngine | None" = None,
    bond_sleeve: "BondSleeve | None" = None,
) -> pd.DataFrame:
    """Simulate the stacked strategy.

    Parameters
    ----------
    indicator_panel : pd.DataFrame, optional
        The full regime indicator panel (vix, dd_velocity, credit, etc.
        each in [0, 1]). Required when ``cfg.capitulation`` is set so the
        overlay can detect capitulation events. If ``cfg.capitulation`` is
        None this argument is unused.
    equity_engine : EquityEngine, optional
        Pluggable equity-sleeve return generator. When None (default),
        the state machine uses ``synthetic_leveraged_returns`` with
        ``cfg.leverage_ter_annual`` for tiers > 1 — preserving the
        pre-Phase-2 behaviour bit-exactly. Pass an explicit engine to
        switch to MarginEngine / FuturesEngine etc.
    bond_sleeve : BondSleeve, optional
        Pluggable bond-sleeve return generator. When None (default), the
        state machine uses ``ief_close.pct_change()`` — preserving the
        pre-Phase-2 behaviour bit-exactly. Pass an explicit sleeve to
        switch to a rotating sleeve etc.

    Returns DataFrame with columns:
        score, equity_tier, equity_pos (notional %), bond_pos (notional %),
        dd_state, dd, equity, ret_strategy, ret_equity_sleeve,
        ret_bond_sleeve, traded, opportunistic
    """
    common = spy_close.index.intersection(ief_close.index).intersection(regime_score.index)
    if tlt_close is not None and cfg.bond_gate_enabled:
        common = common.intersection(tlt_close.index)
    spy = spy_close.loc[common]
    ief = ief_close.loc[common]
    score = regime_score.loc[common]
    tlt = tlt_close.loc[common] if (tlt_close is not None and cfg.bond_gate_enabled) else None

    # Capitulation overlay — pre-compute signals if enabled
    cap_cfg = cfg.capitulation
    cap_state = CapitulationState()
    if cap_cfg is not None and cap_cfg.enabled and indicator_panel is not None:
        panel_aligned = indicator_panel.reindex(common)
        cap_event_series = detect_capitulation(panel_aligned, cap_cfg).reindex(
            common, fill_value=False
        )
        stab_series = detect_stabilisation(panel_aligned, spy, cap_cfg).reindex(
            common, fill_value=False
        )
        # SPY drawdown from recent high
        spy_high = spy.rolling(cap_cfg.spy_dd_lookback, min_periods=1).max()
        spy_dd = (spy / spy_high - 1.0).abs()
        spy_dd_qualifies = (spy_dd >= cap_cfg.spy_dd_required).reindex(common, fill_value=False)
        opp_entry_series = (cap_event_series & stab_series & spy_dd_qualifies).fillna(False)
    else:
        opp_entry_series = pd.Series(False, index=common)

    # Pre-compute leveraged equity returns per tier.
    # Engine selection: if an explicit ``equity_engine`` is supplied, use it;
    # otherwise fall back to the legacy synthetic-leveraged-ETF path so
    # pre-Phase-2 callers reproduce bit-exactly.
    tier_eq_returns: dict[float, pd.Series] = {0.0: pd.Series(0.0, index=common)}
    for L_int in range(1, int(cfg.L_max) + 1):
        L = float(L_int)
        if equity_engine is None:
            rets_L = synthetic_leveraged_returns(
                spy,
                leverage=L,
                ter_annual=cfg.leverage_ter_annual if L > 1.0 else 0.0,
            )
        else:
            rets_L = equity_engine.daily_returns(spy, L)
        tier_eq_returns[L] = rets_L.reindex(common).fillna(0.0)

    # Bond sleeve: explicit sleeve overrides; otherwise legacy ief.pct_change().
    if bond_sleeve is None:
        bond_rets = ief.pct_change().reindex(common).fillna(0.0)
    else:
        bond_rets = bond_sleeve.daily_returns(common).reindex(common).fillna(0.0)

    # State
    equity_tier = 0.0
    dd_state = "normal"
    quiet_bars = 9999
    dd_recovery_bars = 0
    hwm = 1.0
    portfolio_equity = 1.0

    rows = []
    for i, ts in enumerate(common):
        s = score.iat[i]

        # 1) Equity tier from regime score (hysteresis)
        target_eq_tier = _equity_target_tier(s, cfg, equity_tier)

        # 2) Re-entry quiet bars (if previously in cash)
        if equity_tier == 0.0 and target_eq_tier > 0.0:
            if quiet_bars < cfg.re_entry_quiet_bars:
                target_eq_tier = 0.0

        # 2.5) Capitulation overlay (additive — only fires when regime
        # would otherwise keep us out and a capitulation+stabilisation
        # signal is currently active).
        if cap_cfg is not None and cap_cfg.enabled:
            if cap_state.active:
                cap_state.bars_since_entry += 1
                dd_from_entry = (
                    (portfolio_equity - cap_state.entry_equity) / cap_state.entry_equity
                    if cap_state.entry_equity > 0
                    else 0.0
                )
                # Failed-bounce stop: exit if equity drops > X% within Y bars
                if (
                    cap_state.bars_since_entry <= cap_cfg.failed_bounce_lookback
                    and dd_from_entry < -cap_cfg.failed_bounce_drawdown
                ):
                    cap_state.active = False
                    target_eq_tier = 0.0
                elif s >= cap_cfg.graduation_score:
                    # Graduate: regime confirms benign — let normal tier
                    # logic run on this bar onward.
                    cap_state.active = False
                    target_eq_tier = _equity_target_tier(s, cfg, equity_tier)
                else:
                    # Stay at opportunistic tier (overrides regime cash)
                    target_eq_tier = max(target_eq_tier, cap_cfg.opportunistic_tier)
            else:
                # Activate when in cash, regime would keep us out, and
                # all capitulation+stabilisation+SPY-DD conditions fire.
                # The SPY-DD gate (in opp_entry_series) ensures the
                # underlying actually crashed — independent of whether
                # our portfolio took the hit.
                if equity_tier == 0.0 and target_eq_tier == 0.0 and bool(opp_entry_series.iat[i]):
                    cap_state.active = True
                    cap_state.entry_equity = portfolio_equity
                    cap_state.bars_since_entry = 0
                    target_eq_tier = cap_cfg.opportunistic_tier

        # 3) DD circuit breaker (applies to total portfolio)
        dd = (portfolio_equity - hwm) / hwm
        bond_active = True
        if dd_state == "killed":
            target_eq_tier = 0.0
            bond_active = False
            if s >= cfg.dd_re_entry_score:
                dd_recovery_bars += 1
                if dd_recovery_bars >= cfg.dd_re_entry_bars:
                    dd_state = "normal"
                    hwm = portfolio_equity
                    dd_recovery_bars = 0
                    target_eq_tier = _equity_target_tier(s, cfg, equity_tier)
                    bond_active = True
            else:
                dd_recovery_bars = 0
        elif dd_state == "throttled":
            if abs(dd) >= cfg.dd_kill:
                dd_state = "killed"
                target_eq_tier = 0.0
                bond_active = False
                dd_recovery_bars = 0
            elif s >= cfg.dd_re_entry_score:
                dd_recovery_bars += 1
                if dd_recovery_bars >= cfg.dd_re_entry_bars:
                    dd_state = "normal"
                    hwm = portfolio_equity
                    dd_recovery_bars = 0
                else:
                    target_eq_tier = min(target_eq_tier, 1.0)
            else:
                dd_recovery_bars = 0
                target_eq_tier = min(target_eq_tier, 1.0)
        else:
            if abs(dd) >= cfg.dd_kill:
                dd_state = "killed"
                target_eq_tier = 0.0
                bond_active = False
                dd_recovery_bars = 0
                cap_state.active = False  # DD-kill overrides capitulation overlay
            elif abs(dd) >= cfg.dd_throttle:
                dd_state = "throttled"
                target_eq_tier = min(target_eq_tier, 1.0)
                dd_recovery_bars = 0

        # 4) Bond gate (optional)
        if cfg.bond_gate_enabled and tlt is not None and bond_active:
            if not _bond_benign(tlt, ts, cfg):
                bond_active = False

        # 5) Compute target positions
        if target_eq_tier == 0.0 and not bond_active:
            equity_pos = 0.0
            bond_pos = 0.0
        elif target_eq_tier == 0.0:
            equity_pos = 0.0
            bond_pos = cfg.bond_weight
        else:
            equity_pos = cfg.equity_weight
            bond_pos = cfg.bond_weight if bond_active else 0.0

        # 6) Apply transaction cost on changes
        prev_eq_pos = rows[-1]["equity_pos"] if rows else 0.0
        prev_bd_pos = rows[-1]["bond_pos"] if rows else 0.0
        eq_change = abs(equity_pos - prev_eq_pos) + abs(target_eq_tier - equity_tier) * 0.5
        bd_change = abs(bond_pos - prev_bd_pos)
        traded = eq_change > 0 or bd_change > 0
        tc = (cfg.transaction_cost_bps / 10000.0) * (eq_change + bd_change)

        # 7) Compute return for THIS bar based on PREVIOUS bar's positions
        if i == 0:
            ret_eq = 0.0
            ret_bd = 0.0
        else:
            prev_row = rows[-1]
            prev_eq_tier = prev_row["equity_tier"]
            prev_eq_w = prev_row["equity_pos"]
            prev_bd_w = prev_row["bond_pos"]
            ret_eq = prev_eq_w * tier_eq_returns[prev_eq_tier].iat[i]
            ret_bd = prev_bd_w * bond_rets.iat[i]

        ret_total = ret_eq + ret_bd - tc
        portfolio_equity *= 1.0 + ret_total
        if portfolio_equity > hwm:
            hwm = portfolio_equity

        rows.append(
            {
                "date": ts,
                "score": s,
                "equity_tier": target_eq_tier,
                "equity_pos": equity_pos,
                "bond_pos": bond_pos,
                "dd_state": dd_state,
                "dd": dd,
                "equity": portfolio_equity,
                "ret_strategy": ret_total,
                "ret_equity_sleeve": ret_eq,
                "ret_bond_sleeve": ret_bd,
                "transaction_cost": tc,
                "opportunistic": cap_state.active,
                "traded": traded,
            }
        )

        # 8) Update tier state
        if target_eq_tier == 0.0:
            if s >= 0.50:
                quiet_bars += 1
            elif s < 0.30:
                quiet_bars = 0
        else:
            quiet_bars = 9999
        equity_tier = target_eq_tier

    out = pd.DataFrame(rows).set_index("date")
    return out


def summarize_stacked(strategy_df: pd.DataFrame) -> dict:
    """Performance summary for the stacked strategy."""
    rets = strategy_df["ret_strategy"]
    eq = strategy_df["equity"]
    n_years = len(rets) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(252))
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    return {
        "n_years": round(n_years, 2),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "max_dd": round(maxdd, 4),
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "trades_per_year": round(float(strategy_df["traded"].sum() / n_years), 1),
        "frac_in_cash": round(float((strategy_df["equity_pos"] == 0.0).mean()), 3),
        "avg_eq_pos_when_in": round(
            float(strategy_df.loc[strategy_df["equity_pos"] > 0, "equity_pos"].mean()),
            3,
        ),
    }
