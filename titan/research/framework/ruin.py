"""Risk-of-ruin assessment module (V3.6 L65).

Computes formal P(ruin) over a deployment horizon for a strategy at a
specified deployment weight. Block-bootstrap based, preserves empirical
return distribution and autocorrelation.

Designed to fill the gap between MC MaxDD gate (per-strategy tail-risk
proxy) and actual portfolio survival probability at deployed size.

Two main entry points:

- ``assess_strategy_ruin``: single-strategy ruin probability at a
  deployment weight, with respect to a portfolio kill-switch threshold.
- ``assess_joint_ruin``: portfolio-level ruin probability across multiple
  simultaneously-deployed strategies (preserves empirical cross-
  correlations via aligned-index block bootstrap).

Usage::

    from titan.research.framework.ruin import assess_strategy_ruin

    res = assess_strategy_ruin(
        strategy_returns=stitched_oos_returns,   # net of cost
        deployment_weight=0.30,
        portfolio_kill_threshold=0.15,           # 15% portfolio NAV DD
        horizon_bars=252,                        # 1 year of daily bars
        block_size=21,                           # ~1 month
        n_paths=1000,
        seed=42,
    )
    print(res.report())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuinAssessment:
    """Per-strategy or joint ruin-risk assessment.

    Attributes:
        deployment_weight: Fraction of full size the strategy is deployed at.
        portfolio_kill_threshold: Portfolio NAV drawdown that trips kill
            switch (e.g., 0.15 = 15%).
        horizon_bars: Forward horizon in bars (e.g., 252 = 1 year daily).
        p_kill_trip: Probability that scaled-strategy MaxDD crosses
            portfolio_kill_threshold within horizon.
        p_dd_50pct_strategy: Probability that the strategy itself
            crosses 50% MaxDD at full size (irrespective of deployment
            scaling) -- catastrophic-strategy probability.
        median_maxdd_at_size: Median MaxDD over MC paths at the deployed
            weight (negative number).
        p95_maxdd_at_size: 95th-percentile MaxDD at deployed weight
            (more negative than median).
        median_recovery_bars: Median bars to recover from median MaxDD,
            measured as bars until equity returns to the pre-DD peak.
            ``None`` if not recovered within horizon.
        n_paths: Number of MC paths simulated.
        block_size: Block size used in bootstrap.
    """

    deployment_weight: float
    portfolio_kill_threshold: float
    horizon_bars: int
    p_kill_trip: float
    p_dd_50pct_strategy: float
    median_maxdd_at_size: float
    p95_maxdd_at_size: float
    median_recovery_bars: int | None
    n_paths: int
    block_size: int

    def passes(
        self,
        *,
        max_p_kill_trip: float = 0.01,
        max_p_dd_50pct: float = 0.05,
        max_p95_dd: float = -0.25,
    ) -> bool:
        """Default L65 deployment gate.

        Pass criteria:
            - P(portfolio kill switch trips in horizon) < max_p_kill_trip
            - P(strategy DD > 50% at full size) < max_p_dd_50pct
            - 95th-percentile MaxDD at deployed weight better than
              max_p95_dd (i.e., closer to 0)
        """
        return (
            self.p_kill_trip < max_p_kill_trip
            and self.p_dd_50pct_strategy < max_p_dd_50pct
            and self.p95_maxdd_at_size >= max_p95_dd  # less-negative is better
        )

    def report(self) -> str:
        """Human-readable summary string."""
        rec_str = (
            f"{self.median_recovery_bars} bars"
            if self.median_recovery_bars is not None
            else f"NOT RECOVERED within {self.horizon_bars} bars"
        )
        return (
            f"RuinAssessment(weight={self.deployment_weight:.2%}, "
            f"kill_thresh={self.portfolio_kill_threshold:.2%}, "
            f"horizon={self.horizon_bars}b, n_paths={self.n_paths})\n"
            f"  P(kill switch trip) = {self.p_kill_trip:.3%}\n"
            f"  P(strategy DD > 50% full size) = {self.p_dd_50pct_strategy:.3%}\n"
            f"  median MaxDD at deployed weight = {self.median_maxdd_at_size:.3%}\n"
            f"  95th-pct MaxDD at deployed weight = {self.p95_maxdd_at_size:.3%}\n"
            f"  median recovery from median DD = {rec_str}\n"
            f"  passes default gate = {self.passes()}"
        )


def _block_bootstrap_path(
    returns_arr: np.ndarray, horizon_bars: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate one bootstrap path of length horizon_bars from returns_arr."""
    n_blocks = (horizon_bars + block_size - 1) // block_size
    n_available = len(returns_arr) - block_size + 1
    if n_available <= 0:
        return returns_arr[:horizon_bars]
    starts = rng.integers(0, n_available, size=n_blocks)
    path = np.concatenate([returns_arr[s : s + block_size] for s in starts])
    return path[:horizon_bars]


def _max_drawdown(returns_path: np.ndarray) -> float:
    """Maximum drawdown over a return path (negative number)."""
    cum = np.cumsum(returns_path)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def _recovery_bars(returns_path: np.ndarray, threshold_dd: float) -> int | None:
    """Find bars from start of MaxDD to recovery. None if not recovered."""
    cum = np.cumsum(returns_path)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    if dd.min() > threshold_dd:  # threshold_dd is negative
        return 0  # never had a worse DD
    # Find first bar where DD hits threshold or worse.
    dd_hit_idx = int(np.argmax(dd <= threshold_dd))
    # Peak value at that point.
    peak_at_dd = peak[dd_hit_idx]
    # Find first subsequent bar where cum recovers to peak.
    forward = cum[dd_hit_idx:]
    recovery_mask = forward >= peak_at_dd
    if not recovery_mask.any():
        return None
    return int(np.argmax(recovery_mask))


def assess_strategy_ruin(
    strategy_returns: pd.Series,
    *,
    deployment_weight: float,
    portfolio_kill_threshold: float = 0.15,
    horizon_bars: int = 252,
    block_size: int = 21,
    n_paths: int = 1000,
    seed: int = 42,
) -> RuinAssessment:
    """Compute single-strategy ruin probability at deployed weight.

    Parameters:
        strategy_returns: Per-bar net returns (OOS preferred). Must be
            sufficiently long to bootstrap; recommend at least 2 years.
        deployment_weight: Fraction of full Kelly the strategy is deployed
            at. E.g., 0.30 = 30% of full size.
        portfolio_kill_threshold: Portfolio NAV drawdown level that
            trips the kill switch (positive number, e.g., 0.15 = 15%).
        horizon_bars: Forward horizon to simulate (e.g., 252 daily bars).
        block_size: Block bootstrap size (preserves autocorrelation).
        n_paths: Number of MC paths.
        seed: RNG seed.

    Returns:
        RuinAssessment with key gauges.
    """
    rets = strategy_returns.dropna().to_numpy()
    if len(rets) < block_size * 4:
        raise ValueError(
            f"strategy_returns too short ({len(rets)} bars) for block_size={block_size}"
        )

    rng = np.random.default_rng(seed)
    threshold = -abs(portfolio_kill_threshold)

    kill_trips = 0
    dd_50_strategy = 0
    maxdds_at_weight: list[float] = []
    recoveries: list[int | None] = []

    for _ in range(n_paths):
        path = _block_bootstrap_path(rets, horizon_bars, block_size, rng)
        scaled = path * deployment_weight
        maxdd_scaled = _max_drawdown(scaled)
        maxdds_at_weight.append(maxdd_scaled)
        if maxdd_scaled <= threshold:
            kill_trips += 1
        # Strategy-only (full size) catastrophic DD
        maxdd_full = _max_drawdown(path)
        if maxdd_full <= -0.50:
            dd_50_strategy += 1
        # Median-recovery: at scaled weight
        # Use the realised MaxDD as threshold so we measure "recover from this DD"
        recoveries.append(_recovery_bars(scaled, threshold_dd=maxdd_scaled * 0.5))

    maxdds_arr = np.array(maxdds_at_weight)
    valid_recoveries = [r for r in recoveries if r is not None and r > 0]

    return RuinAssessment(
        deployment_weight=deployment_weight,
        portfolio_kill_threshold=portfolio_kill_threshold,
        horizon_bars=horizon_bars,
        p_kill_trip=kill_trips / n_paths,
        p_dd_50pct_strategy=dd_50_strategy / n_paths,
        median_maxdd_at_size=float(np.median(maxdds_arr)),
        p95_maxdd_at_size=float(np.percentile(maxdds_arr, 5)),  # 5th percentile = 95% worst
        median_recovery_bars=int(np.median(valid_recoveries)) if valid_recoveries else None,
        n_paths=n_paths,
        block_size=block_size,
    )


def assess_joint_ruin(
    strategy_returns: dict[str, pd.Series],
    *,
    deployment_weights: dict[str, float],
    portfolio_kill_threshold: float = 0.15,
    horizon_bars: int = 252,
    block_size: int = 21,
    n_paths: int = 1000,
    seed: int = 42,
) -> RuinAssessment:
    """Compute portfolio-level joint ruin probability.

    Aligns all strategy returns on the common index, applies deployment
    weights, sums for a portfolio return series, then block-bootstraps
    the PORTFOLIO series to preserve empirical cross-correlations.

    Parameters:
        strategy_returns: Mapping name -> per-bar returns.
        deployment_weights: Mapping name -> deployment weight (must sum
            to <= 1.0 for a valid portfolio).
        ... (rest same as assess_strategy_ruin).
    """
    common_idx = None
    for name, r in strategy_returns.items():
        r = r.dropna()
        if common_idx is None:
            common_idx = r.index
        else:
            common_idx = common_idx.intersection(r.index)
    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common index across strategies")

    weight_sum = sum(deployment_weights.values())
    if weight_sum > 1.0 + 1e-9:
        raise ValueError(
            f"Sum of deployment_weights={weight_sum:.3f} exceeds 1.0"
        )

    portfolio_returns = pd.Series(0.0, index=common_idx)
    for name, ret in strategy_returns.items():
        weight = deployment_weights.get(name, 0.0)
        if weight == 0.0:
            continue
        portfolio_returns = portfolio_returns + (ret.reindex(common_idx).fillna(0.0) * weight)

    # The "deployment_weight" of the portfolio is 1.0 (it's already weighted).
    return assess_strategy_ruin(
        portfolio_returns,
        deployment_weight=1.0,
        portfolio_kill_threshold=portfolio_kill_threshold,
        horizon_bars=horizon_bars,
        block_size=block_size,
        n_paths=n_paths,
        seed=seed,
    )


__all__ = [
    "RuinAssessment",
    "assess_joint_ruin",
    "assess_strategy_ruin",
]
