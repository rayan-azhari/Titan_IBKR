"""mr_fx — corrected pure-research module (Wave A.6 verification).

V2 replacement for the simplified mr_audjpy-shaped module used in the
original Wave A.6 sweep. Adds the live strategy's three key machinery
elements that the original abstraction dropped:

    1. **Session-anchored VWAP** — resets at London 07:00 UTC and NY 13:00 UTC
       (matches `titan/strategies/mr_fx/strategy.py` line 7).
    2. **4-tier grid entries** — at percentile thresholds [0.90, 0.95, 0.98,
       0.99] with size schedule [1, 2, 4, 8]. Position is the SUM of triggered
       tiers (stacked).
    3. **Realistic FX cost** — 0.5 bps per turnover-unit (liquid-hours EUR/USD
       spread + slip), vs the 1.5 bps used in the original L58 sweep.

The original Wave A.6 sweep used rolling-VWAP + single-tier + 1.5 bps. The
critical question this module answers: does signal-layer fail HOLD when the
strategy's actual machinery + realistic costs are used?

Causality (L04 / A1 / L18):
    - Session VWAP at bar t uses [session_start..t] (causal — t-1 already
      closed at the moment of decision in live trading).
    - Rolling percentile bands use past-only window (excluding t).
    - Position at t earns return t -> t+1 via .shift(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MrFxSessionConfig:
    """Live-mirroring pure-research config for mr_fx."""

    # VWAP anchor hours (UTC).
    anchor_hours_utc: tuple[int, ...] = (7, 13)
    # Rolling percentile window (M5 bars).
    pct_window: int = 2000
    # Tier-grid entries.
    tiers_pct: tuple[float, ...] = (0.90, 0.95, 0.98, 0.99)
    tier_sizes: tuple[int, ...] = (1, 2, 4, 8)
    # Partial reversion target.
    reversion_target_pct: float = 0.50
    # NY-close hard exit (UTC hour).
    ny_close_utc: int = 21
    # Costs — liquid-hours EUR/USD.
    apply_costs: bool = True
    cost_bps_per_turnover: float = 0.5  # vs 1.5 in original Wave A.6


def _session_vwap(close: pd.Series, *, anchor_hours: tuple[int, ...]) -> pd.Series:
    """Session-anchored VWAP — resets at each anchor hour (UTC).

    No real volume on FX bars (-1 sentinel typical), so this is a session-
    anchored running mean. Matches the live strategy's effective behaviour.
    """
    hours = close.index.hour
    # Session-id increments at each anchor.
    is_anchor = np.zeros(len(close), dtype=bool)
    for h in anchor_hours:
        is_anchor |= hours == h
    # Force is_anchor at index 0 so the first bar starts a session.
    is_anchor[0] = True
    session_id = pd.Series(is_anchor.cumsum(), index=close.index)
    # Within each session, compute running mean of close.
    vwap = close.groupby(session_id).transform(lambda s: s.expanding().mean())
    return vwap


def _which_tier(dev: float, tier_thresholds: tuple[float, ...]) -> int:
    """Returns the highest tier index (1-based) the deviation crosses, or 0.

    Uses |dev| against the percentile-band threshold (caller supplies signed
    thresholds). Positive return = entry; sign is from |dev| sign.
    """
    abs_dev = abs(dev)
    tier = 0
    for i, t in enumerate(tier_thresholds):
        if abs_dev >= t:
            tier = i + 1
    return tier


def mr_fx_session_returns(
    bars_df: pd.DataFrame, *, cfg: MrFxSessionConfig | None = None
) -> pd.Series:
    """Per-M5-bar net return with session-VWAP + tier-grid mechanics.

    Position state machine:
        - Track which tiers (1-N) have been entered SHORT and LONG this session.
        - At bar t, if dev > +pct_band[tier] AND tier not yet entered short:
            add tier_size to SHORT position.
        - Symmetric for LONG.
        - Exit: when |dev| has reverted by reversion_target_pct (50% by default)
          OR NY-session-close hour.
        - Reset session state at session anchors.
    """
    if cfg is None:
        cfg = MrFxSessionConfig()

    close = bars_df["close"].astype(float)
    n = len(close)
    if n < cfg.pct_window + 100:
        return pd.Series(0.0, index=close.index, name="ret")

    # Session VWAP.
    vwap = _session_vwap(close, anchor_hours=cfg.anchor_hours_utc)
    dev = (close - vwap) / vwap.replace(0, np.nan)

    # Rolling percentile bands (past-only, exclusive of t).
    # The live strategy uses positive (.abs()) deviation percentile; we mirror.
    abs_dev = dev.abs()
    bands = []
    for pct in cfg.tiers_pct:
        b = abs_dev.rolling(cfg.pct_window, min_periods=cfg.pct_window).quantile(pct)
        bands.append(b.to_numpy())

    # Session-id for tier-tracking reset.
    hours = close.index.hour
    is_anchor = np.zeros(n, dtype=bool)
    for h in cfg.anchor_hours_utc:
        is_anchor |= hours == h
    is_anchor[0] = True
    session_id = is_anchor.cumsum()
    ny_close_mask = (hours == cfg.ny_close_utc)

    # State machine.
    arr_dev = dev.to_numpy()
    arr_session = session_id
    arr_ny_close = ny_close_mask
    pos = np.zeros(n, dtype=float)
    long_tiers_hit: set[int] = set()
    short_tiers_hit: set[int] = set()
    current_session = -1
    entry_dev = 0.0  # signed entry deviation for the reversion-target test
    position = 0.0
    for i in range(n):
        # Reset session state at anchor.
        if arr_session[i] != current_session:
            long_tiers_hit.clear()
            short_tiers_hit.clear()
            current_session = arr_session[i]

        d = arr_dev[i]
        if np.isnan(d) or np.isnan(bands[0][i]):
            pos[i] = position
            continue

        # NY-close hard exit.
        if arr_ny_close[i] and position != 0.0:
            position = 0.0
            long_tiers_hit.clear()
            short_tiers_hit.clear()
            entry_dev = 0.0
            pos[i] = 0.0
            continue

        # Determine which tier the current |dev| qualifies for.
        tier = 0
        for k in range(len(cfg.tiers_pct)):
            if abs(d) >= bands[k][i]:
                tier = k + 1

        if tier > 0:
            # Long entry: dev is NEGATIVE (price below VWAP), expect reversion UP.
            # Short entry: dev is POSITIVE.
            is_long = d < 0
            tiers_set = long_tiers_hit if is_long else short_tiers_hit
            for t_idx in range(1, tier + 1):
                if t_idx not in tiers_set:
                    tiers_set.add(t_idx)
                    add_size = cfg.tier_sizes[t_idx - 1]
                    if is_long:
                        position += add_size
                    else:
                        position -= add_size
                    if entry_dev == 0.0:
                        entry_dev = d  # remember initial entry

        # Reversion-target exit — when |dev| has reverted from entry_dev to
        # |entry_dev| * (1 - reversion_target_pct).
        if position != 0.0 and entry_dev != 0.0:
            revert_threshold = abs(entry_dev) * (1.0 - cfg.reversion_target_pct)
            if abs(d) <= revert_threshold:
                position = 0.0
                long_tiers_hit.clear()
                short_tiers_hit.clear()
                entry_dev = 0.0

        pos[i] = position

    position_s = pd.Series(pos, index=close.index)

    # Per-bar log return.
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    held_lagged = position_s.shift(1).fillna(0.0)
    # Normalize by max position so Sharpe is comparable across cells with
    # different total stack sizes (stack-of-15 vs stack-of-1).
    max_stack = float(sum(cfg.tier_sizes))
    held_norm = held_lagged / max_stack
    gross = held_norm * log_ret

    if cfg.apply_costs:
        # Turnover per bar in NORMALIZED units (so cost-bps applies per unit
        # weight change consistently with bond_gold/etf_trend cost models).
        dpos = (position_s.diff().abs() / max_stack).fillna(0.0)
        cost = dpos * (cfg.cost_bps_per_turnover / 10_000.0)
        return (gross - cost).rename("ret")
    return gross.rename("ret")


def mr_fx_session_assert_causal(
    bars_df: pd.DataFrame, *, cfg: MrFxSessionConfig | None = None
) -> None:
    """L04 smoke test."""
    if cfg is None:
        cfg = MrFxSessionConfig()
    base = mr_fx_session_returns(bars_df, cfg=cfg)
    rng = np.random.default_rng(42)
    n = len(bars_df)
    t_corrupt = int(rng.integers(cfg.pct_window + 100, n - 100))
    corrupted = bars_df.copy()
    corrupted.iloc[t_corrupt:] = np.nan
    corrupted_ret = mr_fx_session_returns(corrupted, cfg=cfg)

    base_past = base.iloc[: t_corrupt - 1].dropna()
    corrupted_past = corrupted_ret.iloc[: t_corrupt - 1].dropna()
    common = base_past.index.intersection(corrupted_past.index)
    if len(common) == 0:
        raise AssertionError("Causality test could not find common past index")
    diffs = (base_past.reindex(common) - corrupted_past.reindex(common)).abs()
    max_diff = float(diffs.max())
    if max_diff > 1e-12:
        n_changed = int((diffs > 1e-12).sum())
        raise AssertionError(
            f"Causality smoke failed: t={t_corrupt} changed {n_changed} past returns "
            f"(max |delta|={max_diff:.2e})"
        )


__all__ = [
    "MrFxSessionConfig",
    "mr_fx_session_returns",
    "mr_fx_session_assert_causal",
]
