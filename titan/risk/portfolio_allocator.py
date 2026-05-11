"""Portfolio Allocator -- Inverse-Volatility Capital Allocation.

Computes per-strategy allocation weights using inverse-volatility weighting
on a **daily, timestamp-aligned** equity series. Rebalances by wall-clock
date (not by tick counter, which the previous implementation used and which
caused the "monthly" rebalance to fire every ~1 day when H1 strategies were
in the portfolio).

Reads equity histories via ``portfolio_risk_manager.get_equity_histories()``
(public accessor added in the April 2026 rewrite) instead of reaching into
the private ``_strategies`` dict.

Method
------
    sigma_i = sqrt(EWMA(lambda=0.94) var of daily returns of strategy i) * sqrt(252)
    w_i    = (1 / sigma_i) / SUM(1 / sigma_j)

Constraints (applied post-computation):
    - max_weight: 0.60 (no strategy > 60%)
    - min_weight: 0.05 (every strategy gets at least 5%)
    - correlation_penalty: when |r_ij| > 0.70 on the daily-aligned grid,
      reduce both weights by 10%
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from titan.research.metrics import BARS_PER_YEAR, annualize_vol

logger = logging.getLogger(__name__)

_DEFAULT_ALLOC_CONFIG: dict = {
    "rebalance_interval_days": 21,
    "ewma_lambda": 0.94,
    "min_weight": 0.05,
    "max_weight": 0.60,
    "min_history_days": 30,
    "correlation_penalty_threshold": 0.70,
}


class PortfolioAllocator:
    """Inverse-volatility capital allocator (timestamp-aware, wall-clock gated)."""

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = {**_DEFAULT_ALLOC_CONFIG, **(config or {})}
        self._weights: dict[str, float] = {}
        self._last_rebalance_date: date | None = None
        self._force_next: bool = False

    def load_config(self, config: dict) -> None:
        self._config = {**_DEFAULT_ALLOC_CONFIG, **config}
        logger.info(
            "[Allocator] Config loaded: rebal=%dd min=%.0f%% max=%.0f%%",
            self._config["rebalance_interval_days"],
            self._config["min_weight"] * 100,
            self._config["max_weight"] * 100,
        )

    def tick(self, now: date | None = None) -> None:
        """Wall-clock gated rebalance trigger.

        Safe to call from every bar of every strategy -- the actual
        rebalance only fires when the calendar distance from the last
        rebalance exceeds ``rebalance_interval_days`` business days.
        """
        today = now or date.today()
        interval = int(self._config["rebalance_interval_days"])

        due = (
            self._force_next
            or self._last_rebalance_date is None
            or ((today - self._last_rebalance_date).days >= interval)
        )
        if not due:
            return
        self._rebalance()
        self._last_rebalance_date = today
        self._force_next = False

    def get_weight(self, strategy_id: str) -> float:
        if not self._weights:
            return 1.0
        return self._weights.get(strategy_id, 1.0 / max(1, len(self._weights)))

    def get_all_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def force_rebalance(self) -> None:
        self._force_next = True

    # ── Internal ──────────────────────────────────────────────────────────

    def _rebalance(self) -> None:
        from titan.risk.portfolio_risk_manager import portfolio_risk_manager

        histories = portfolio_risk_manager.get_equity_histories()
        if len(histories) < 2:
            for sid in histories:
                self._weights[sid] = 1.0
            return

        min_hist = int(self._config["min_history_days"])
        lam = float(self._config["ewma_lambda"])
        min_w = float(self._config["min_weight"])
        max_w = float(self._config["max_weight"])
        corr_threshold = float(self._config["correlation_penalty_threshold"])

        # Build aligned daily-return DataFrame once and reuse for vol + corr.
        rets_map: dict[str, pd.Series] = {}
        for sid, eq in histories.items():
            if len(eq) < min_hist:
                continue
            r = eq.pct_change().dropna()
            if len(r) < 10:
                continue
            rets_map[sid] = r

        if len(rets_map) < 2:
            return

        df = pd.DataFrame(rets_map)
        df = df.fillna(0.0)  # non-trade days contribute zero return

        vols: dict[str, float] = {}
        for sid in df.columns:
            ewma_var = df[sid].ewm(alpha=1.0 - lam, adjust=False).var().iloc[-1]
            per_day_std = max(0.0, float(ewma_var)) ** 0.5
            # Per-strategy equity histories are resampled to business-day in
            # PortfolioRiskManager.get_equity_histories, so factor is 252.
            ann_vol = annualize_vol(per_day_std, periods_per_year=BARS_PER_YEAR["D"])
            if ann_vol > 0:
                vols[sid] = ann_vol

        if not vols:
            return

        inv_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total_inv = sum(inv_vols.values())
        raw_weights = {sid: iv / total_inv for sid, iv in inv_vols.items()}

        # Correlation penalty on the aligned grid.
        if len(df) >= 20:
            corr = df.corr()
            sids = list(df.columns)
            for i, a in enumerate(sids):
                for b in sids[i + 1 :]:
                    r = abs(float(corr.loc[a, b]))
                    if r > corr_threshold and a in raw_weights and b in raw_weights:
                        raw_weights[a] *= 0.90
                        raw_weights[b] *= 0.90
                        logger.info(
                            "[Allocator] Correlation penalty: %s <-> %s r=%.2f -- -10%% each",
                            a,
                            b,
                            r,
                        )

        # Constraints + renormalise.
        constrained = {sid: max(min_w, min(max_w, w)) for sid, w in raw_weights.items()}
        total_c = sum(constrained.values())
        if total_c > 0:
            self._weights = {sid: w / total_c for sid, w in constrained.items()}
        else:
            n = len(constrained)
            self._weights = {sid: 1.0 / n for sid in constrained}

        # Strategies with insufficient history get floor + renormalise.
        all_sids = set(histories.keys())
        missing = all_sids - set(self._weights.keys())
        if missing:
            for sid in missing:
                self._weights[sid] = min_w
            total_w = sum(self._weights.values())
            if total_w > 0:
                self._weights = {s: w / total_w for s, w in self._weights.items()}

        logger.info(
            "[Allocator] Rebalanced: %s",
            {sid: f"{w:.1%}" for sid, w in sorted(self._weights.items())},
        )


def _load_alloc_config() -> dict:
    import tomllib

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
