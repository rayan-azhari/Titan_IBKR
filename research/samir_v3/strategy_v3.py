"""Samir-Stack V3 Layer 2 — pure equity, binary regime-gated MES futures.

The minimum-viable strategy that honours Samir Varma's binary-classification
thesis. No bonds, no tier ladder, no capitulation. One signal (the V3
regime score), one decision rule (above threshold = deploy, below = cash),
one execution path (MES futures at constant leverage when deployed).

If THIS doesn't beat the baselines, nothing else built on top of it will.

Design choices
--------------

1. **Cash yield is earned, not zero.** When the strategy is in cash, the
   account earns the daily T-bill rate (matches IBKR's USD cash yield).
   Without this, the head-to-head vs "always-on MES" is unfair — the
   always-on engine earns T-bill on its cash equity via the futures basis
   mechanism, so the gated version must too when flat.

2. **Strict shift discipline.** Position at bar `t-1` (decided from score
   `t-1`, computed from data through bar `t-1`) earns return at bar `t`.
   The regime score itself is causally derived (filtered HMM). No look-
   ahead.

3. **Re-entry quiet period (optional).** Samir's "best-days cluster with
   worst-days" insight suggests staying out for N bars after exiting to
   avoid V-bounce whipsaw. Default: 0 (no quiet bars) for the minimum
   viable version; available as a tunable.

4. **No DD breaker, no capitulation.** Those are layer 4+ additions.
   Phase 5 documented their incremental value; we add them only after
   layer 1+2 has earned a deployment slot.
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
    """Configuration for the V3 binary-regime equity strategy.

    Conservative defaults: τ=0.5 majority-probability gate, L=2 per the
    Phase 5 finding that L=2 dominates higher leverage on Calmar CI lo.
    """

    score_threshold: float = 0.5
    """Above this, deploy. Below or equal, hold cash. Pre-registered
    canonical choice = 0.5 (HMM filtering majority)."""

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


def run_v3_strategy(
    spy_close: pd.Series,
    regime_score: pd.Series,
    cfg: V3Config,
) -> pd.DataFrame:
    """Simulate the V3 binary-gate strategy on SPY-equivalent underlying.

    Parameters
    ----------
    spy_close : pd.Series
        TR-adjusted SPY close prices. Drives the FuturesEngine return.
    regime_score : pd.Series
        Per-bar regime score in [0, 1], typically from
        ``vix_hmm.vix_hmm_regime_score``. Higher = benign.
    cfg : V3Config
        Strategy parameters.

    Returns
    -------
    pd.DataFrame indexed to the common dates with columns:
        score, deployed (0/1), L_applied, equity, ret_strategy,
        ret_levered, ret_cash, transition (bool), quiet_bars
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

    # Cash return: T-bill at the same funding-series rate the engine uses
    # for consistency. NOT zero — flat exposure still earns cash yield.
    fund = funding_series(common).reindex(common).ffill().fillna(0.04)
    ret_cash_full = (fund / 252.0).astype(float)

    # State machine: previous-bar score decides current-bar position.
    # (Standard shift discipline: signal at t-1 earns return at t.)
    score_lag = score.shift(1).fillna(0.0)
    deployed_intent = (score_lag > cfg.score_threshold).astype(int)

    # Re-entry quiet bars: if just exited (transition deployed → cash),
    # cannot re-enter until quiet_bars have elapsed below threshold.
    if cfg.re_entry_quiet_bars > 0:
        deployed = np.zeros(len(common), dtype=int)
        quiet_counter = 999  # large initial value → no quiet block at t=0
        for i in range(len(common)):
            intent = int(deployed_intent.iat[i])
            prev = deployed[i - 1] if i > 0 else 0
            if intent == 1 and prev == 0 and quiet_counter < cfg.re_entry_quiet_bars:
                deployed[i] = 0
                quiet_counter += 1
            else:
                deployed[i] = intent
                if intent == 0:
                    if prev == 1:
                        quiet_counter = 0  # just exited
                    else:
                        quiet_counter += 1
                else:
                    quiet_counter = 999  # deployed, reset
        deployed = pd.Series(deployed, index=common)
    else:
        deployed = deployed_intent.astype(int)

    # Apply transaction cost on state transitions.
    transition = deployed.diff().fillna(0).abs().astype(int)
    cost_bps = cfg.transaction_cost_bps
    tx_cost = (cost_bps / 10_000.0) * transition.astype(float)

    # Compose daily return.
    ret_strategy = (deployed * ret_levered_full + (1 - deployed) * ret_cash_full - tx_cost).astype(
        float
    )

    L_applied = deployed.astype(float) * cfg.L_target
    equity = (1.0 + ret_strategy.fillna(0.0)).cumprod()

    return pd.DataFrame(
        {
            "score": score,
            "deployed": deployed,
            "L_applied": L_applied,
            "ret_levered": ret_levered_full,
            "ret_cash": ret_cash_full,
            "transition": transition,
            "tx_cost": tx_cost,
            "ret_strategy": ret_strategy,
            "equity": equity,
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
    }
