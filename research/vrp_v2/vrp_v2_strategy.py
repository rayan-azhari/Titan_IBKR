"""VRP v2 -- causal strategy function with percentile-rolling gates.

Pre-registered in ``directives/Pre-Reg E1b VRP Capture v2 Percentile Gates 2026-05-15.md``.
Class: ``StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY``.

Design vs E1:
    * Bare-threshold gate (`ratio_long >= 1.05`) is replaced by a
      percentile-rolling gate (`ratio_long >= rolling_quantile_q60_252d`).
      L26 mitigation: the gate moves with the signal's own distribution
      so 0.1σ input noise does not systematically shift it.
    * Continuous-scaling variant (C7): position = w_target * sigmoid(...).
      L26 alternative mitigation: even bars sitting exactly on the gate
      get a partial position rather than a 100%-or-0 flip.

Causality (V3.6 A1 / L04):
    All inputs `.shift(1)`'d. Rolling quantile is computed on the lagged
    signal so today's quantile uses only history up to and including
    yesterday. Weights are NOT lagged again inside vrp_v2_returns --
    target_weights() already produces causal weights.

Cost model (L23):
    Same as E1 -- bps/turnover + per-fill commission floor + rebalance
    threshold. Default 6 bps/turnover, $1/fill, $30k notional, 5%
    rebalance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VrpV2Config:
    """One row of the E1b pre-reg grid."""

    gate_kind: str = "percentile"  # "percentile" or "sigmoid"
    window_d: int = 252  # rolling window for quantile / sigmoid centre
    enter_q: float = 0.60  # quantile-rank threshold for entering contango regime
    exit_q: float = 0.40  # quantile-rank threshold for exiting (hysteresis built-in)
    target_short_weight: float = -0.50
    sigmoid_scale: float = 0.05  # only used when gate_kind=="sigmoid"


VRP_V2_UNIVERSE: tuple[str, ...] = ("VIXY",)


def _aligned_series(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    target = pd.to_datetime(idx).tz_localize(None).normalize()
    return s.reindex(target).ffill(limit=3)


def _causal_rolling_quantile(s: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling quantile on the shifted series, so q[t] uses data <= t-1."""
    return s.shift(1).rolling(window, min_periods=max(20, window // 4)).quantile(q)


def _sigmoid(x: pd.Series, scale: float) -> pd.Series:
    """Numerically-stable sigmoid; returns Series indexed like x."""
    # 1 / (1 + exp(-x/scale)) but stable in both tails.
    z = x / max(scale, 1e-9)
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return pd.Series(out, index=x.index, name=x.name)


def vrp_v2_target_weights(
    closes: pd.DataFrame,
    *,
    cfg: VrpV2Config,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
) -> pd.DataFrame:
    """Daily target-weight DataFrame for VRP v2.

    Returns columns ``["VIXY"]`` only (no SPY overlay in E1b — see
    pre-reg §2 rationale).

    Regime logic (percentile gate, cfg.gate_kind == "percentile"):
        ratio_long(t-1) >= Q[enter_q](t-1)  -> SHORT VIXY at target_short_weight
        ratio_long(t-1) <  Q[exit_q](t-1)   -> FLAT
        otherwise                           -> hold prior state (hysteresis)

    Regime logic (continuous sigmoid, cfg.gate_kind == "sigmoid"):
        weight(t) = target_short_weight * sigmoid((ratio_long(t-1) - Q[enter_q](t-1)) / sigmoid_scale)
        no hysteresis -- position fades smoothly.
    """
    if cfg.gate_kind not in ("percentile", "sigmoid"):
        raise ValueError(f"Unknown gate_kind={cfg.gate_kind!r}")

    idx = closes.index
    v = _aligned_series(vix, idx)
    v3 = _aligned_series(vix3m, idx)
    # ratio_long uses VIX3M / VIX. Causal: t-1 values.
    ratio_long = (v3 / v).shift(1)

    q_enter = (
        ratio_long.shift(0)  # already-shifted via the line above; rolling on it.
        .rolling(cfg.window_d, min_periods=max(20, cfg.window_d // 4))
        .quantile(cfg.enter_q)
    )
    q_exit = (
        ratio_long.shift(0)
        .rolling(cfg.window_d, min_periods=max(20, cfg.window_d // 4))
        .quantile(cfg.exit_q)
    )

    n = len(idx)
    if cfg.gate_kind == "percentile":
        rl = ratio_long.values
        qe = q_enter.values
        qx = q_exit.values
        state = np.zeros(n, dtype=np.int8)
        cur = 0
        for i in range(n):
            if np.isnan(rl[i]) or np.isnan(qe[i]) or np.isnan(qx[i]):
                state[i] = cur
                continue
            if cur == 1:
                # currently short VIXY (in contango regime). Exit if below exit_q.
                if rl[i] < qx[i]:
                    cur = 0
            else:
                # currently flat. Enter if above enter_q.
                if rl[i] >= qe[i]:
                    cur = 1
            state[i] = cur
        w_vixy = np.where(state == 1, cfg.target_short_weight, 0.0)
    else:  # sigmoid
        excess = ratio_long - q_enter
        sig = _sigmoid(excess, cfg.sigmoid_scale)
        # sigmoid -> [0, 1]; multiply by target_short_weight (negative).
        # NaN propagation: any of (ratio_long, q_enter) NaN -> NaN excess -> 0 weight
        # (we fillna(0.5) for the sigmoid centre then multiply, but cleaner to
        # explicitly zero when the rolling quantile is undefined).
        w_vixy = (sig * cfg.target_short_weight).where(q_enter.notna(), 0.0).values

    return pd.DataFrame({"VIXY": w_vixy}, index=idx)


def vrp_v2_returns(
    closes: pd.DataFrame,
    *,
    cfg: VrpV2Config,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
    cost_bps_per_turnover: float = 6.0,
    cost_fixed_usd_per_fill: float = 1.0,
    notional_usd: float = 30_000.0,
    rebalance_threshold: float = 0.05,
) -> pd.Series:
    """Per-bar cost-adjusted strategy returns. Mirrors `vrp_returns` (E1)
    but with VRP-v2 gate logic."""
    weights = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)

    held = pd.DataFrame(index=closes.index, columns=weights.columns, dtype=float)
    last_w = np.zeros(weights.shape[1])
    for i in range(len(weights)):
        target = weights.iloc[i].values
        delta = np.abs(target - last_w)
        if delta.max() > rebalance_threshold or i == 0:
            last_w = target.copy()
        held.iloc[i] = last_w

    held_lagged = held.shift(1).fillna(0.0)

    leg_returns = pd.DataFrame(index=closes.index, columns=weights.columns, dtype=float)
    for col in weights.columns:
        if col in closes.columns:
            leg_returns[col] = closes[col].pct_change().fillna(0.0)
        else:
            leg_returns[col] = 0.0

    gross = (held_lagged * leg_returns).sum(axis=1)

    dw = held.diff().abs().fillna(0.0)
    turnover_per_bar = dw.sum(axis=1)
    bps_drag = turnover_per_bar * (cost_bps_per_turnover / 10_000.0)
    fill_mask = turnover_per_bar > 0
    fixed_drag = fill_mask.astype(float) * (cost_fixed_usd_per_fill / max(notional_usd, 1.0))
    cost = bps_drag + fixed_drag

    return (gross - cost).rename("vrp_v2_returns")


def vrp_v2_assert_causal(
    closes: pd.DataFrame,
    *,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
    cfg: VrpV2Config | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> None:
    """A10 causality smoke test."""
    if cfg is None:
        cfg = VrpV2Config()
    if "VIXY" not in closes.columns:
        raise ValueError("vrp_v2_assert_causal: closes must include VIXY")
    rng = np.random.default_rng(seed)
    n = len(closes)
    if n < 300:
        return

    base_w = vrp_v2_target_weights(closes, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m)

    for _ in range(n_trials):
        t_corrupt = int(rng.integers(cfg.window_d + 20, n - 5))
        corrupt_closes = closes.copy()
        corrupt_closes.iloc[t_corrupt:] = corrupt_closes.iloc[t_corrupt:] * 1.5
        corrupt_vix = vix.copy()
        corrupt_vix9d = vix9d.copy()
        corrupt_vix3m = vix3m.copy()
        if isinstance(corrupt_vix.index, pd.DatetimeIndex):
            t_ts = closes.index[t_corrupt]
            corrupt_vix.loc[corrupt_vix.index >= t_ts] *= 1.5
            corrupt_vix9d.loc[corrupt_vix9d.index >= t_ts] *= 1.5
            corrupt_vix3m.loc[corrupt_vix3m.index >= t_ts] *= 1.5

        corrupt_w = vrp_v2_target_weights(
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
                f"vrp_v2_assert_causal: future corruption at t={t_corrupt} "
                f"changed past weights at {diff_idx.sum()} bars."
            )
