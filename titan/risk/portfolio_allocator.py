"""Portfolio Allocator — Inverse-Volatility Capital Allocation.

Computes per-strategy allocation weights using inverse-volatility weighting,
rebalanced monthly. Each strategy queries its weight and multiplies it into
position sizing.

Architecture
------------
Singleton: ``portfolio_allocator`` is imported by all live strategies.
It reads per-strategy equity histories from ``portfolio_risk_manager``.

    from titan.risk.portfolio_allocator import portfolio_allocator

    # In strategy on_bar(), after portfolio_risk_manager.update():
    alloc = portfolio_allocator.get_weight(self._prm_id)
    adjusted_size = raw_size * alloc * portfolio_risk_manager.scale_factor

Rebalancing
-----------
Weights are recomputed every ``rebalance_interval_days`` (default 21 ~ monthly).
Between rebalances the weights are frozen. This avoids high-frequency churn.

Method: Inverse-Volatility
--------------------------
    sigma_i = EWMA(lambda=0.94) annualized vol of strategy i's equity returns
    w_i = (1 / sigma_i) / SUM(1 / sigma_j)

Constraints (applied post-computation):
    - max_weight: 0.60 (no strategy > 60%)
    - min_weight: 0.05 (every strategy gets at least 5%)
    - correlation_penalty: when |r_ij| > 0.70, reduce combined weight by 10%

Configuration (config/risk.toml [allocation] section)
-----------------------------------------------------
    rebalance_interval_days   int    Rebalance period in trading days (default 21)
    ewma_lambda               float  EWMA decay for per-strategy vol (default 0.94)
    min_weight                float  Floor per strategy (default 0.05)
    max_weight                float  Ceiling per strategy (default 0.60)
    min_history_days          int    Min equity observations before allocating (default 30)
    correlation_penalty_threshold  float  Penalise pairs with |r| above this (default 0.70)
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_ALLOC_CONFIG: dict = {
    "rebalance_interval_days": 21,
    "ewma_lambda": 0.94,
    "min_weight": 0.05,
    "max_weight": 0.60,
    "min_history_days": 30,
    "correlation_penalty_threshold": 0.70,
}


class PortfolioAllocator:
    """Inverse-volatility capital allocator.

    Reads equity histories from PortfolioRiskManager._strategies and
    computes per-strategy allocation weights.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = {**_DEFAULT_ALLOC_CONFIG, **(config or {})}
        self._weights: dict[str, float] = {}  # strategy_id -> allocation weight
        self._update_counter: int = 0
        self._rebalance_due: bool = True

    def load_config(self, config: dict) -> None:
        """Update configuration at runtime."""
        self._config = {**_DEFAULT_ALLOC_CONFIG, **config}
        logger.info(
            "[Allocator] Config loaded: rebal=%dd min=%.0f%% max=%.0f%%",
            self._config["rebalance_interval_days"],
            self._config["min_weight"] * 100,
            self._config["max_weight"] * 100,
        )

    def tick(self) -> None:
        """Call once per bar (from any strategy). Triggers rebalance when due.

        Reads strategy equity histories from the portfolio_risk_manager
        singleton. Must be called AFTER portfolio_risk_manager.update().
        """
        interval = int(self._config["rebalance_interval_days"])
        self._update_counter += 1

        if self._update_counter >= interval or self._rebalance_due:
            self._rebalance()
            self._update_counter = 0
            self._rebalance_due = False

    def get_weight(self, strategy_id: str) -> float:
        """Get the current allocation weight for a strategy.

        Returns 1.0 / N (equal weight) if not yet computed or strategy unknown.
        """
        if not self._weights:
            return 1.0
        return self._weights.get(strategy_id, 1.0 / max(1, len(self._weights)))

    def get_all_weights(self) -> dict[str, float]:
        """Snapshot of all current allocation weights."""
        return dict(self._weights)

    def force_rebalance(self) -> None:
        """Force a rebalance on the next tick()."""
        self._rebalance_due = True

    # ── Internal ──────────────────────────────────────────────────────────

    def _rebalance(self) -> None:
        """Compute inverse-vol weights from equity histories."""
        from titan.risk.portfolio_risk_manager import portfolio_risk_manager

        strategies = portfolio_risk_manager._strategies
        if len(strategies) < 2:
            # Single strategy or none — equal weight
            for sid in strategies:
                self._weights[sid] = 1.0
            return

        min_hist = int(self._config["min_history_days"])
        lam = float(self._config["ewma_lambda"])
        min_w = float(self._config["min_weight"])
        max_w = float(self._config["max_weight"])
        corr_threshold = float(self._config["correlation_penalty_threshold"])

        # Compute per-strategy annualized vol from equity histories
        vols: dict[str, float] = {}
        return_series: dict[str, pd.Series] = {}

        for sid, state in strategies.items():
            hist = list(state.equity_history)
            if len(hist) < min_hist:
                continue

            eq = pd.Series(hist, dtype=float)
            rets = eq.pct_change().dropna()
            if len(rets) < 10:
                continue

            # EWMA variance
            ewma_var = rets.ewm(alpha=1.0 - lam, adjust=False).var().iloc[-1]
            ann_vol = math.sqrt(max(0.0, ewma_var) * 252)

            if ann_vol > 0:
                vols[sid] = ann_vol
                return_series[sid] = rets

        if not vols:
            return

        # Inverse-vol raw weights
        inv_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total_inv = sum(inv_vols.values())
        raw_weights = {sid: iv / total_inv for sid, iv in inv_vols.items()}

        # Correlation penalty: if |r_ij| > threshold, reduce both by 10%
        if len(return_series) >= 2:
            sids = list(return_series.keys())
            df = pd.DataFrame(return_series).dropna()
            if len(df) >= 20:
                corr = df.corr()
                for i, a in enumerate(sids):
                    for b in sids[i + 1 :]:
                        if a in corr.index and b in corr.columns:
                            r = abs(float(corr.loc[a, b]))
                            if r > corr_threshold:
                                raw_weights[a] *= 0.90
                                raw_weights[b] *= 0.90
                                logger.info(
                                    "[Allocator] Correlation penalty: %s <-> %s"
                                    " r=%.2f — both reduced 10%%",
                                    a,
                                    b,
                                    r,
                                )

        # Apply min/max constraints and re-normalise
        constrained = {}
        for sid, w in raw_weights.items():
            constrained[sid] = max(min_w, min(max_w, w))

        total_c = sum(constrained.values())
        if total_c > 0:
            self._weights = {sid: w / total_c for sid, w in constrained.items()}
        else:
            n = len(constrained)
            self._weights = {sid: 1.0 / n for sid in constrained}

        # Include strategies that didn't have enough history (equal share of remainder)
        all_sids = set(strategies.keys())
        allocated_sids = set(self._weights.keys())
        missing = all_sids - allocated_sids
        if missing:
            equal_share = min_w
            for sid in missing:
                self._weights[sid] = equal_share
            # Re-normalise
            total_w = sum(self._weights.values())
            if total_w > 0:
                self._weights = {s: w / total_w for s, w in self._weights.items()}

        logger.info(
            "[Allocator] Rebalanced: %s",
            {sid: f"{w:.1%}" for sid, w in sorted(self._weights.items())},
        )


# ── Module-level singleton ────────────────────────────────────────────────────


def _load_alloc_config() -> dict:
    """Load [allocation] section from config/risk.toml if it exists."""
    import tomllib
    from pathlib import Path

    risk_toml = Path(__file__).resolve().parents[2] / "config" / "risk.toml"
    if not risk_toml.exists():
        return {}
    try:
        with open(risk_toml, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("allocation", {})
    except Exception:
        return {}


portfolio_allocator: PortfolioAllocator = PortfolioAllocator(config=_load_alloc_config())
"""Module-level singleton imported by all live strategies.

Auto-loads config from config/risk.toml [allocation] section on import.

Usage:
    from titan.risk.portfolio_allocator import portfolio_allocator

    # In strategy on_bar(), after portfolio_risk_manager.update():
    portfolio_allocator.tick()
    alloc = portfolio_allocator.get_weight("ic_equity_UNH")
    adjusted_size = raw_size * alloc * portfolio_risk_manager.scale_factor
"""
