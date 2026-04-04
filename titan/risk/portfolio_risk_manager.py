"""Portfolio Risk Manager — Live Portfolio-Level Risk Control.

Aggregates equity snapshots across all active strategies, tracks portfolio-level
drawdown from a shared high-water mark, and broadcasts a halt signal when the
portfolio DD threshold is breached.

This module does NOT replace per-strategy monitoring already in
titan/strategies/ic_generic/strategy.py (which watches per-instrument DD,
rolling Sharpe, and trade frequency). It adds an orthogonal portfolio-level
layer that halts ALL strategies together when combined drawdown is excessive.

Architecture
------------
Singleton: import ``portfolio_risk_manager`` and call its methods.
Each live strategy:
    1. Calls ``portfolio_risk_manager.register_strategy()`` in ``on_start()``.
    2. Calls ``portfolio_risk_manager.update()`` after every bar / fill event.
    3. Reads ``portfolio_risk_manager.halt_all`` before sizing any new order.
    4. Multiplies computed size by ``portfolio_risk_manager.scale_factor``.

State resets on process restart (high-water mark starts at combined equity
at first call to ``update()`` after all strategies are registered).

Scale factor composition (Tier 2, April 2026)
----------------------------------------------
The final ``scale_factor`` is the minimum of three independent multipliers:

    scale_factor = min(dd_scale, vol_scale, regime_scale)

    dd_scale     : Drawdown heat — linear scale-down [0.25, 1.0] from 10% to 15% DD.
    vol_scale    : Vol-targeting — target_vol / EWMA(realized_vol), clipped [0.25, 2.0].
    regime_scale : Regime gate — min(vix_scale, atr_scale) from VIX tiers + ATR percentile.

Configuration (config/risk.toml [portfolio] section)
-----------------------------------------------------
    portfolio_max_dd_pct       float  Halt ALL at this portfolio DD % (default 15.0)
    portfolio_heat_scale_pct   float  Begin scaling at this portfolio DD % (default 10.0)
    correlation_window_days    int    Rolling lookback for live correlation (default 60)
    correlation_halt_threshold float  Log WARNING if strategy pair r > this (default 0.85)
    portfolio_max_single_pct   float  Max fraction any single strategy holds (default 60.0)
    vol_target_ann_pct         float  Annualized portfolio vol target (default 12.0)
    vol_ewma_lambda            float  EWMA decay for realized vol (default 0.94)
    vol_scale_min              float  Floor for vol-targeting scale (default 0.25)
    vol_scale_max              float  Ceiling for vol-targeting scale (default 2.0)
    vix_tier_1                 float  VIX threshold: below = full size (default 17.8)
    vix_tier_2                 float  VIX threshold: below = 75% (default 23.1)
    vix_tier_3                 float  VIX threshold: below = 50% (default 30.0)
    atr_pct_low                float  ATR percentile below = size up 1.25x (default 25.0)
    atr_pct_high               float  ATR percentile above = size down 0.50x (default 75.0)
    atr_pct_extreme            float  ATR percentile above = size down 0.25x (default 90.0)

Integration example (in each strategy's on_bar):
-------------------------------------------------
    from titan.risk.portfolio_risk_manager import portfolio_risk_manager

    # At end of on_bar, after computing current equity:
    portfolio_risk_manager.update(self._strategy_id, float(account_balance))
    if portfolio_risk_manager.halt_all:
        self._flatten_all_and_stop()

    # Optionally feed VIX (any strategy that has it):
    portfolio_risk_manager.update_vix(current_vix_level)

    # Before sizing a new order:
    adjusted_size = computed_size * portfolio_risk_manager.scale_factor
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ── Default configuration (overridden by config/risk.toml [portfolio]) ─────────

_DEFAULT_CONFIG: dict = {
    "portfolio_max_dd_pct": 15.0,
    "portfolio_heat_scale_pct": 10.0,
    "correlation_window_days": 60,
    "correlation_halt_threshold": 0.85,
    "portfolio_max_single_pct": 60.0,
    # Vol-targeting (Tier 2 #8)
    "vol_target_ann_pct": 12.0,
    "vol_ewma_lambda": 0.94,
    "vol_scale_min": 0.25,
    "vol_scale_max": 2.0,
    # Regime scaling (Tier 2 #9)
    "vix_tier_1": 17.8,
    "vix_tier_2": 23.1,
    "vix_tier_3": 30.0,
    "atr_pct_low": 25.0,
    "atr_pct_high": 75.0,
    "atr_pct_extreme": 90.0,
}


# ── Per-strategy state ─────────────────────────────────────────────────────────


@dataclass
class _StrategyState:
    strategy_id: str
    current_equity: float
    equity_hwm: float
    equity_history: deque[float] = field(default_factory=lambda: deque(maxlen=252))

    @property
    def drawdown_pct(self) -> float:
        if self.equity_hwm <= 0:
            return 0.0
        return (self.current_equity - self.equity_hwm) / self.equity_hwm


# ── Portfolio Risk Manager ─────────────────────────────────────────────────────


class PortfolioRiskManager:
    """Live portfolio-level risk manager.

    Thread safety: NautilusTrader runs strategies on a single event loop
    (single-threaded per actor). Python GIL protects dict/deque mutations
    from concurrent reads in the same process. No explicit locking needed.

    The halt state is sticky: once ``halt_all`` is True it stays True until
    ``reset_halt()`` is called explicitly by the operator.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self._strategies: dict[str, _StrategyState] = {}
        self._portfolio_hwm: float | None = None
        self._halt_all: bool = False
        self._scale_factor: float = 1.0
        self._corr_check_counter: int = 0

        # ── Vol-targeting state (Tier 2 #8) ──────────────────────────────
        self._ewma_var: float | None = None  # EWMA variance (squared vol)
        self._prev_total_equity: float | None = None

        # ── Regime scaling state (Tier 2 #9) ─────────────────────────────
        self._vix_level: float | None = None  # Latest VIX reading
        self._atr_percentiles: dict[str, float] = {}  # strategy_id -> ATR %ile

        # ── Component scales (for logging / monitoring) ──────────────────
        self._dd_scale: float = 1.0
        self._vol_scale: float = 1.0
        self._regime_scale: float = 1.0

    # ── Public API ─────────────────────────────────────────────────────────

    def load_config(self, config: dict) -> None:
        """Update configuration at runtime (e.g. after loading TOML)."""
        self._config = {**_DEFAULT_CONFIG, **config}
        logger.info(
            "[PortfolioRM] Config loaded: max_dd=%.1f%% heat=%.1f%% max_single=%.1f%%",
            self._config["portfolio_max_dd_pct"],
            self._config["portfolio_heat_scale_pct"],
            self._config["portfolio_max_single_pct"],
        )

    def register_strategy(
        self,
        strategy_id: str,
        initial_equity: float,
    ) -> None:
        """Register a strategy. Call once in each strategy's on_start().

        Args:
            strategy_id:     Unique string identifier (e.g. "ic_mtf_eur_usd").
            initial_equity:  Starting equity value for this strategy (float).
        """
        self._strategies[strategy_id] = _StrategyState(
            strategy_id=strategy_id,
            current_equity=initial_equity,
            equity_hwm=initial_equity,
        )
        logger.info(
            "[PortfolioRM] Registered '%s'  initial equity=%.2f  total strategies=%d",
            strategy_id,
            initial_equity,
            len(self._strategies),
        )

    def update(self, strategy_id: str, current_equity: float) -> None:
        """Update equity snapshot for one strategy and run portfolio checks.

        Call this at the end of every on_bar() or on_order_filled() event.

        Args:
            strategy_id:    Matches the id used in register_strategy().
            current_equity: Current account equity value (float).
        """
        if self._halt_all:
            return  # already halted; no further checks needed

        if strategy_id not in self._strategies:
            logger.warning(
                "[PortfolioRM] Unknown strategy '%s' — call register_strategy() first.",
                strategy_id,
            )
            return

        state = self._strategies[strategy_id]
        state.current_equity = current_equity
        state.equity_history.append(current_equity)

        # Update per-strategy high-water mark
        if current_equity > state.equity_hwm:
            state.equity_hwm = current_equity

        self._check_portfolio_health()

        # Correlation check every 24 updates (≈ once per day on H1 strategies)
        self._corr_check_counter += 1
        if self._corr_check_counter >= 24:
            self._corr_check_counter = 0
            self.check_correlation_regime()

    def update_vix(self, vix_level: float) -> None:
        """Feed the latest VIX level for regime-based scaling.

        Call this from any strategy that has access to VIX data (e.g. via
        a daily bar subscription to ^VIX or a VIX futures proxy). Only one
        strategy needs to call this -- the value is shared across all.

        Args:
            vix_level: Current VIX index level (e.g. 18.5).
        """
        self._vix_level = vix_level

    def update_atr_percentile(self, strategy_id: str, atr_pct: float) -> None:
        """Feed the ATR percentile rank for one strategy's instrument.

        Each strategy can compute its own ATR percentile (e.g. current ATR14
        as a percentile over the last 500 bars) and feed it here. The regime
        gate uses the *maximum* across all instruments (most stressed = most
        conservative).

        Args:
            strategy_id: Matches the id used in register_strategy().
            atr_pct:     ATR percentile [0-100] for this instrument.
        """
        self._atr_percentiles[strategy_id] = atr_pct

    @property
    def halt_all(self) -> bool:
        """True when the portfolio kill switch has been triggered.

        Strategies must check this before submitting any new orders.
        When True, flatten all open positions and stop trading.
        """
        return self._halt_all

    @property
    def scale_factor(self) -> float:
        """Composite position size multiplier.

        Computed as min(dd_scale, vol_scale, regime_scale).

        - dd_scale:     [0.25, 1.0] — drawdown heat.
        - vol_scale:    [0.25, 2.0] — vol-targeting overlay.
        - regime_scale: [0.25, 1.25] — VIX + ATR regime gate.

        Multiply computed position size by this before order submission.
        """
        return self._scale_factor

    def get_summary(self) -> dict:
        """Snapshot of current portfolio state for monitoring / logging."""
        total = self._total_equity()
        realized_vol = self._annualized_vol()
        return {
            "total_equity": round(total, 2),
            "portfolio_drawdown_pct": round(self._portfolio_drawdown() * 100, 3),
            "halt_all": self._halt_all,
            "scale_factor": round(self._scale_factor, 3),
            "dd_scale": round(self._dd_scale, 3),
            "vol_scale": round(self._vol_scale, 3),
            "regime_scale": round(self._regime_scale, 3),
            "realized_vol_ann_pct": round(realized_vol * 100, 2) if realized_vol else None,
            "vix_level": self._vix_level,
            "strategy_count": len(self._strategies),
            "strategies": {
                sid: {
                    "equity": round(s.current_equity, 2),
                    "drawdown_pct": round(s.drawdown_pct * 100, 3),
                    "weight_pct": round(s.current_equity / total * 100 if total > 0 else 0.0, 2),
                    "atr_pct": round(self._atr_percentiles.get(sid, 50.0), 1),
                }
                for sid, s in self._strategies.items()
            },
        }

    def check_correlation_regime(self) -> None:
        """Compute rolling correlation between strategy equity histories.

        Logs WARNING if any pair exceeds correlation_halt_threshold.
        Informational only — does not halt trading on correlation spikes.

        Called automatically every ~24 updates; also callable manually.
        """
        threshold = float(self._config["correlation_halt_threshold"])
        window = int(self._config["correlation_window_days"])
        histories: dict[str, list[float]] = {}

        for sid, state in self._strategies.items():
            if len(state.equity_history) >= 20:
                histories[sid] = list(state.equity_history)[-window:]

        if len(histories) < 2:
            return

        # Convert equity histories to return series for correlation
        ret_series: dict[str, pd.Series] = {}
        for sid, hist in histories.items():
            s = pd.Series(hist, dtype=float)
            r = s.pct_change().dropna()
            if len(r) >= 10:
                ret_series[sid] = r

        if len(ret_series) < 2:
            return

        df = pd.DataFrame(ret_series).dropna()
        if len(df) < 10:
            return

        corr = df.corr()
        labels = list(ret_series.keys())

        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                if a not in corr.index or b not in corr.columns:
                    continue
                r = float(corr.loc[a, b])
                if abs(r) > threshold:
                    logger.warning(
                        "[PortfolioRM] Correlation alert: '%s' ↔ '%s' r=%.3f "
                        "(threshold %.2f). Consider reducing combined allocation.",
                        a,
                        b,
                        r,
                        threshold,
                    )

    def reset_halt(self) -> None:
        """Manual operator action: clear the halt flag and reset HWM.

        Only call after diagnosing the cause of the halt and confirming it
        is safe to resume trading. Resets portfolio high-water mark to the
        current combined equity and all scale components.
        """
        old_equity = self._portfolio_hwm
        self._halt_all = False
        self._scale_factor = 1.0
        self._dd_scale = 1.0
        self._vol_scale = 1.0
        self._regime_scale = 1.0
        self._portfolio_hwm = self._total_equity()
        self._ewma_var = None
        self._prev_total_equity = None
        logger.warning(
            "[PortfolioRM] HALT RESET by operator. Old HWM=%.2f  New HWM=%.2f",
            old_equity or 0.0,
            self._portfolio_hwm,
        )

    # ── Internal logic ─────────────────────────────────────────────────────

    def _total_equity(self) -> float:
        return sum(s.current_equity for s in self._strategies.values())

    def _portfolio_drawdown(self) -> float:
        """Portfolio drawdown from portfolio high-water mark (negative = loss)."""
        total = self._total_equity()
        if self._portfolio_hwm is None:
            self._portfolio_hwm = total
            return 0.0
        if total > self._portfolio_hwm:
            self._portfolio_hwm = total
        if self._portfolio_hwm <= 0:
            return 0.0
        return (total - self._portfolio_hwm) / self._portfolio_hwm

    # ── Vol-targeting helpers (Tier 2 #8) ─────────────────────────────────

    def _update_ewma_vol(self, total_equity: float) -> None:
        """Update EWMA variance from portfolio equity return."""
        if self._prev_total_equity is None or self._prev_total_equity <= 0:
            self._prev_total_equity = total_equity
            return

        ret = (total_equity - self._prev_total_equity) / self._prev_total_equity
        self._prev_total_equity = total_equity

        lam = float(self._config["vol_ewma_lambda"])
        if self._ewma_var is None:
            # Seed with squared return (first observation)
            self._ewma_var = ret * ret
        else:
            self._ewma_var = lam * self._ewma_var + (1.0 - lam) * ret * ret

    def _annualized_vol(self) -> float | None:
        """Annualized portfolio vol from EWMA variance. Returns None if not ready."""
        if self._ewma_var is None or self._ewma_var <= 0:
            return None
        # Assume ~252 trading days per year for daily strategies.
        # For intraday, the EWMA adapts via update frequency automatically.
        return math.sqrt(self._ewma_var * 252)

    def _compute_vol_scale(self) -> float:
        """Vol-targeting scale: target_vol / realized_vol, clipped."""
        ann_vol = self._annualized_vol()
        if ann_vol is None or ann_vol <= 0:
            return 1.0  # Not enough data yet — no adjustment

        target = float(self._config["vol_target_ann_pct"]) / 100.0
        scale_min = float(self._config["vol_scale_min"])
        scale_max = float(self._config["vol_scale_max"])

        raw_scale = target / ann_vol
        return max(scale_min, min(scale_max, raw_scale))

    # ── Regime scaling helpers (Tier 2 #9) ────────────────────────────────

    def _compute_vix_scale(self) -> float:
        """VIX tier scaling. Returns 1.0 if VIX is not available."""
        if self._vix_level is None:
            return 1.0

        tier_1 = float(self._config["vix_tier_1"])
        tier_2 = float(self._config["vix_tier_2"])
        tier_3 = float(self._config["vix_tier_3"])

        if self._vix_level < tier_1:
            return 1.0
        elif self._vix_level < tier_2:
            return 0.75
        elif self._vix_level < tier_3:
            return 0.50
        else:
            return 0.25

    def _compute_atr_scale(self) -> float:
        """ATR percentile scaling. Uses max ATR %ile across all instruments."""
        if not self._atr_percentiles:
            return 1.0

        max_atr_pct = max(self._atr_percentiles.values())
        low = float(self._config["atr_pct_low"])
        high = float(self._config["atr_pct_high"])
        extreme = float(self._config["atr_pct_extreme"])

        if max_atr_pct < low:
            return 1.25  # Low-vol regime — can size up slightly
        elif max_atr_pct < high:
            return 1.0  # Normal
        elif max_atr_pct < extreme:
            return 0.50  # High vol — halve sizes
        else:
            return 0.25  # Extreme — quarter sizes

    def _compute_regime_scale(self) -> float:
        """Composite regime scale = min(vix_scale, atr_scale)."""
        return min(self._compute_vix_scale(), self._compute_atr_scale())

    # ── Core health check ─────────────────────────────────────────────────

    def _check_portfolio_health(self) -> None:
        """Core risk check — called on every update().

        Computes three independent scale factors and composes the final
        scale_factor as their minimum.
        """
        max_dd_threshold = float(self._config["portfolio_max_dd_pct"]) / 100.0
        heat_threshold = float(self._config["portfolio_heat_scale_pct"]) / 100.0
        max_single_pct = float(self._config["portfolio_max_single_pct"]) / 100.0

        port_dd = self._portfolio_drawdown()
        total = self._total_equity()

        # ── Gate 1: Portfolio drawdown kill switch ─────────────────────────
        if port_dd < -max_dd_threshold:
            self._halt_all = True
            self._scale_factor = 0.0
            self._dd_scale = 0.0
            logger.critical(
                "[PortfolioRM] KILL SWITCH TRIGGERED — portfolio DD %.2f%% "
                "exceeds %.1f%% limit. ALL strategies must flatten and halt.",
                port_dd * 100,
                max_dd_threshold * 100,
            )
            return

        # ── Gate 2: DD heat scale (linear from 1.0 to 0.25) ──────────────
        if port_dd < -heat_threshold:
            heat_fraction = abs(port_dd) / max_dd_threshold
            self._dd_scale = max(0.25, 1.0 - heat_fraction)
        else:
            self._dd_scale = 1.0

        # ── Gate 3: Vol-targeting scale ───────────────────────────────────
        self._update_ewma_vol(total)
        self._vol_scale = self._compute_vol_scale()

        # ── Gate 4: Regime scale (VIX + ATR) ──────────────────────────────
        self._regime_scale = self._compute_regime_scale()

        # ── Compose final scale factor ────────────────────────────────────
        self._scale_factor = min(self._dd_scale, self._vol_scale, self._regime_scale)

        # Log when any component is actively scaling down
        if self._scale_factor < 0.99:
            logger.warning(
                "[PortfolioRM] Scale %.0f%% (DD=%.0f%% Vol=%.0f%% Regime=%.0f%%) "
                "| port_DD=%.2f%% ann_vol=%s vix=%s",
                self._scale_factor * 100,
                self._dd_scale * 100,
                self._vol_scale * 100,
                self._regime_scale * 100,
                port_dd * 100,
                f"{self._annualized_vol() * 100:.1f}%" if self._annualized_vol() else "n/a",
                f"{self._vix_level:.1f}" if self._vix_level else "n/a",
            )

        # ── Gate 5: Single-strategy concentration warning ─────────────────
        if total > 0:
            for sid, state in self._strategies.items():
                weight = state.current_equity / total
                if weight > max_single_pct:
                    logger.warning(
                        "[PortfolioRM] Concentration: '%s' holds %.1f%% of portfolio "
                        "(limit %.1f%%).",
                        sid,
                        weight * 100,
                        max_single_pct * 100,
                    )


# ── Module-level singleton ─────────────────────────────────────────────────────


def _load_portfolio_config() -> dict:
    """Load [portfolio] section from config/risk.toml if it exists."""
    import tomllib
    from pathlib import Path

    risk_toml = Path(__file__).resolve().parents[2] / "config" / "risk.toml"
    if not risk_toml.exists():
        return {}
    try:
        with open(risk_toml, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("portfolio", {})
    except Exception:
        return {}


portfolio_risk_manager: PortfolioRiskManager = PortfolioRiskManager(config=_load_portfolio_config())
"""Module-level singleton imported by all live strategies.

Auto-loads config from config/risk.toml [portfolio] section on import.

Usage:
    from titan.risk.portfolio_risk_manager import portfolio_risk_manager

    # In strategy on_start():
    portfolio_risk_manager.register_strategy("ic_mtf_eur_usd", initial_equity=10_000.0)

    # In strategy on_bar():
    portfolio_risk_manager.update("ic_mtf_eur_usd", float(account.balance))
    if portfolio_risk_manager.halt_all:
        self._flatten_all_and_stop()
    size = computed_size * portfolio_risk_manager.scale_factor
"""
