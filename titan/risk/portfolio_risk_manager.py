"""Portfolio Risk Manager -- Live Portfolio-Level Risk Control.

Aggregates per-strategy equity, tracks portfolio-level drawdown, computes an
EWMA realised-vol overlay on a **timestamped, daily-aligned** basis, and
exposes a composite ``scale_factor`` plus a sticky ``halt_all`` kill switch.

Architecture (April 2026 rewrite)
---------------------------------
Each strategy owns a ``StrategyEquityTracker`` (see ``strategy_equity.py``)
that computes a true per-strategy equity curve (seed + realised + MTM). The
strategy calls ``portfolio_risk_manager.update(strategy_id, equity, ts)``
once per bar with that equity and an explicit UTC timestamp.

The risk manager stores each strategy's equity history as a **timestamped
``pd.Series``**, not a raw deque of floats. All variance / correlation math
happens after resampling every strategy onto a shared business-day grid --
no more mixing hourly and daily samples in a ``sqrt(252)`` annualisation.

Wall-clock gating
-----------------
Portfolio-vol recomputation, correlation checks, and allocator rebalances
are triggered by the *calendar date* changing, not by a counter of ticks.
An H1 strategy that fires 24 bars a day and a D1 strategy that fires once
therefore both cause "daily" work to happen once per day.

Halt persistence
----------------
The kill-switch state is persisted to ``.tmp/portfolio_halt.json``. On
startup the manager re-reads that file so a crashed + restarted process
does not silently un-halt. ``reset_halt`` is the only way to clear it and
writes the reset to the same file with an operator timestamp.

Scale factor composition
------------------------
    scale_factor = min(dd_scale, vol_scale, regime_scale)

    dd_scale     : Drawdown heat -- linear scale-down [0.25, 1.0].
    vol_scale    : Vol-targeting -- target_vol / realised_vol, clipped.
    regime_scale : min(vix_scale, atr_scale) from VIX tiers + ATR percentile.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from titan.research.metrics import BARS_PER_YEAR, annualize_vol

logger = logging.getLogger(__name__)

# ── Default configuration (overridden by config/risk.toml [portfolio]) ─────────

_DEFAULT_CONFIG: dict = {
    "portfolio_max_dd_pct": 15.0,
    "portfolio_heat_scale_pct": 10.0,
    "correlation_window_days": 60,
    "correlation_halt_threshold": 0.85,
    "portfolio_max_single_pct": 60.0,
    "vol_target_ann_pct": 12.0,
    "vol_ewma_lambda": 0.94,
    "vol_scale_min": 0.25,
    "vol_scale_max": 2.0,
    "vix_tier_1": 17.8,
    "vix_tier_2": 23.1,
    "vix_tier_3": 30.0,
    "atr_pct_low": 25.0,
    "atr_pct_high": 75.0,
    "atr_pct_extreme": 90.0,
    # Max rows stored per strategy (~4 years of business days).
    "history_max_days": 1000,
}

_HALT_STATE_PATH = Path(__file__).resolve().parents[2] / ".tmp" / "portfolio_halt.json"


# ── Per-strategy state ─────────────────────────────────────────────────────────


@dataclass
class _StrategyState:
    strategy_id: str
    initial_equity: float
    current_equity: float
    equity_hwm: float
    # Tick-level equity samples, indexed by UTC timestamp. Resampled to daily
    # on demand for vol / correlation work -- the raw samples remain here so
    # we can still compute per-tick drawdown.
    samples: "pd.Series" = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def drawdown_pct(self) -> float:
        if self.equity_hwm <= 0:
            return 0.0
        return (self.current_equity - self.equity_hwm) / self.equity_hwm

    def append(self, ts: pd.Timestamp, equity: float, max_rows: int) -> None:
        # Dedupe same-timestamp overwrites (re-emitted bars).
        self.samples.loc[ts] = equity
        if len(self.samples) > max_rows * 24:  # cap intraday sample count
            self.samples = self.samples.iloc[-(max_rows * 24) :]

    def daily_equity(self, max_days: int) -> pd.Series:
        """Resample to business-day last-observation; returned series is
        right-bounded by today. Cap to ``max_days`` rows.
        """
        if self.samples.empty:
            return pd.Series(dtype=float)
        s = self.samples.copy()
        s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").tz_localize(None)
        daily = s.resample("B").last().dropna()
        return daily.iloc[-max_days:]


# ── Portfolio Risk Manager ─────────────────────────────────────────────────────


class PortfolioRiskManager:
    """Live portfolio-level risk manager (timestamp-aware)."""

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self._strategies: dict[str, _StrategyState] = {}
        self._portfolio_hwm: float | None = None
        self._halt_all: bool = False
        self._halt_reason: str | None = None
        self._scale_factor: float = 1.0

        # Wall-clock gating for expensive work.
        self._last_daily_date: date | None = None
        self._last_corr_date: date | None = None

        # EWMA-vol state (computed from daily portfolio NAV, not ticks).
        self._ewma_var: float | None = None
        self._last_daily_nav: float | None = None

        # Regime inputs.
        self._vix_level: float | None = None
        self._atr_percentiles: dict[str, float] = {}

        # Component scales (for logging / monitoring).
        self._dd_scale: float = 1.0
        self._vol_scale: float = 1.0
        self._regime_scale: float = 1.0

        # Halt persistence -- read disk state on construction so crash+restart
        # cannot silently un-halt.
        self._load_halt_state()

    # ── Public API ─────────────────────────────────────────────────────────

    def load_config(self, config: dict) -> None:
        self._config = {**_DEFAULT_CONFIG, **config}
        logger.info(
            "[PortfolioRM] Config loaded: max_dd=%.1f%% heat=%.1f%% max_single=%.1f%%",
            self._config["portfolio_max_dd_pct"],
            self._config["portfolio_heat_scale_pct"],
            self._config["portfolio_max_single_pct"],
        )

    def register_strategy(self, strategy_id: str, initial_equity: float) -> None:
        self._strategies[strategy_id] = _StrategyState(
            strategy_id=strategy_id,
            initial_equity=float(initial_equity),
            current_equity=float(initial_equity),
            equity_hwm=float(initial_equity),
        )
        logger.info(
            "[PortfolioRM] Registered '%s' initial_equity=%.2f total=%d",
            strategy_id,
            initial_equity,
            len(self._strategies),
        )

    def update(
        self,
        strategy_id: str,
        current_equity: float,
        ts: pd.Timestamp | datetime | None = None,
    ) -> None:
        """Update a strategy's equity snapshot with explicit timestamp.

        ``ts`` must be UTC. If omitted, wall-clock ``datetime.now(UTC)`` is
        used -- pass an explicit bar timestamp where available.
        """
        if self._halt_all:
            return

        if strategy_id not in self._strategies:
            logger.warning(
                "[PortfolioRM] Unknown strategy '%s' -- register_strategy() first.",
                strategy_id,
            )
            return

        if ts is None:
            ts_utc = pd.Timestamp.now(tz="UTC")
        elif isinstance(ts, (int, float)):
            # Accept nanosecond epoch (NautilusTrader ``bar.ts_event``).
            ts_utc = pd.Timestamp(int(ts), unit="ns", tz="UTC")
        else:
            ts_utc = pd.Timestamp(ts)
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.tz_localize("UTC")
            else:
                ts_utc = ts_utc.tz_convert("UTC")

        state = self._strategies[strategy_id]
        state.current_equity = float(current_equity)
        state.append(ts_utc, float(current_equity), int(self._config["history_max_days"]))

        if current_equity > state.equity_hwm:
            state.equity_hwm = float(current_equity)

        self._check_portfolio_health(ts_utc)

    def update_vix(self, vix_level: float) -> None:
        self._vix_level = float(vix_level)

    def update_atr_percentile(self, strategy_id: str, atr_pct: float) -> None:
        self._atr_percentiles[strategy_id] = float(atr_pct)

    @property
    def halt_all(self) -> bool:
        return self._halt_all

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def get_equity_histories(self) -> dict[str, pd.Series]:
        """Public accessor -- returns each strategy's daily business-day equity.

        Used by ``PortfolioAllocator`` instead of reaching into the private
        ``_strategies`` dict.
        """
        max_days = int(self._config["history_max_days"])
        return {sid: st.daily_equity(max_days) for sid, st in self._strategies.items()}

    def get_summary(self) -> dict:
        total = self._total_equity()
        ann_vol = self._annualized_vol()
        return {
            "total_equity": round(total, 2),
            "portfolio_drawdown_pct": round(self._portfolio_drawdown() * 100, 3),
            "halt_all": self._halt_all,
            "halt_reason": self._halt_reason,
            "scale_factor": round(self._scale_factor, 3),
            "dd_scale": round(self._dd_scale, 3),
            "vol_scale": round(self._vol_scale, 3),
            "regime_scale": round(self._regime_scale, 3),
            "realized_vol_ann_pct": round(ann_vol * 100, 2) if ann_vol else None,
            "vix_level": self._vix_level,
            "strategy_count": len(self._strategies),
            "strategies": {
                sid: {
                    "equity": round(s.current_equity, 2),
                    "drawdown_pct": round(s.drawdown_pct * 100, 3),
                    "weight_pct": round(s.current_equity / total * 100 if total > 0 else 0.0, 2),
                    "atr_pct": round(self._atr_percentiles.get(sid, 50.0), 1),
                    "samples": len(s.samples),
                }
                for sid, s in self._strategies.items()
            },
        }

    def check_correlation_regime(self) -> None:
        """Compute rolling correlation on a shared business-day grid.

        Uses ``get_equity_histories`` (each strategy's daily last-equity),
        converts to pct-change returns, aligns on the common date index
        (outer-join + fill with zero-return), and correlates. Logs a warning
        when any pair exceeds ``correlation_halt_threshold``.
        """
        threshold = float(self._config["correlation_halt_threshold"])
        window = int(self._config["correlation_window_days"])

        series_map = self.get_equity_histories()
        ret_map: dict[str, pd.Series] = {}
        for sid, eq in series_map.items():
            if len(eq) < 20:
                continue
            r = eq.pct_change().dropna()
            if len(r) < 10:
                continue
            ret_map[sid] = r.iloc[-window:]

        if len(ret_map) < 2:
            return

        df = pd.DataFrame(ret_map)  # auto-aligns on timestamped index
        df = df.fillna(0.0)  # non-trade days contribute zero return
        if len(df) < 20:
            return

        corr = df.corr()
        labels = list(ret_map.keys())
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                if a not in corr.index or b not in corr.columns:
                    continue
                r = float(corr.loc[a, b])
                if abs(r) > threshold:
                    logger.warning(
                        "[PortfolioRM] Correlation alert: '%s' <-> '%s' r=%.3f "
                        "(threshold %.2f). Consider reducing combined allocation.",
                        a,
                        b,
                        r,
                        threshold,
                    )

    def reset_halt(self, operator: str = "unknown") -> None:
        """Manual operator action: clear the halt flag.

        The previous implementation re-anchored ``portfolio_hwm`` to the
        current (drawn-down) total equity, which silently reduced the absolute
        kill-switch distance. The new behaviour keeps the original HWM so a
        second drawdown of the same magnitude still trips the switch from the
        pre-halt peak. Operators who want to re-baseline must explicitly call
        ``reset_hwm`` afterwards.
        """
        old_hwm = self._portfolio_hwm
        self._halt_all = False
        self._halt_reason = None
        self._scale_factor = 1.0
        self._dd_scale = 1.0
        self._vol_scale = 1.0
        self._regime_scale = 1.0
        self._ewma_var = None
        self._last_daily_nav = None
        self._persist_halt_state(operator=operator, cleared=True)
        logger.warning(
            "[PortfolioRM] HALT RESET by operator=%s. HWM preserved at %.2f.",
            operator,
            old_hwm or 0.0,
        )

    def reset_hwm(self, operator: str = "unknown") -> None:
        """Re-anchor the portfolio high-water mark to current total equity."""
        new_hwm = self._total_equity()
        logger.warning(
            "[PortfolioRM] HWM RE-ANCHORED by operator=%s. old=%.2f new=%.2f",
            operator,
            self._portfolio_hwm or 0.0,
            new_hwm,
        )
        self._portfolio_hwm = new_hwm

    # ── Internal logic ─────────────────────────────────────────────────────

    def _total_equity(self) -> float:
        return sum(s.current_equity for s in self._strategies.values())

    def _portfolio_drawdown(self) -> float:
        total = self._total_equity()
        if self._portfolio_hwm is None:
            self._portfolio_hwm = total
            return 0.0
        if total > self._portfolio_hwm:
            self._portfolio_hwm = total
        if self._portfolio_hwm <= 0:
            return 0.0
        return (total - self._portfolio_hwm) / self._portfolio_hwm

    # ── Vol-targeting: daily NAV, EWMA variance ──────────────────────────

    def _recompute_daily_vol(self) -> None:
        """Recompute EWMA variance from the daily portfolio NAV series.

        Called **once per calendar day** via the wall-clock gate in
        ``_check_portfolio_health``. This decouples annualisation from
        per-strategy tick cadence -- the variance is always over daily NAV
        returns, so ``sqrt(var * 252)`` is now correct by construction.
        """
        histories = self.get_equity_histories()
        if not histories:
            return

        df = pd.DataFrame(histories)
        if df.empty:
            return
        df = df.ffill().fillna(0.0)
        nav = df.sum(axis=1)
        if len(nav) < 10:
            return

        rets = nav.pct_change().dropna()
        if len(rets) < 10:
            return

        lam = float(self._config["vol_ewma_lambda"])
        # Rebuild EWMA variance from scratch each day -- cheap, deterministic,
        # and avoids the stale ``_prev_total_equity`` race from the old code.
        var = float(rets.iloc[0] ** 2)
        for r in rets.iloc[1:]:
            var = lam * var + (1.0 - lam) * (r * r)
        self._ewma_var = var
        self._last_daily_nav = float(nav.iloc[-1])

    def _annualized_vol(self) -> float | None:
        if self._ewma_var is None or self._ewma_var <= 0:
            return None
        # Portfolio NAV is resampled to business-day before EWMA variance is
        # computed (see _recompute_daily_vol), so the series is daily and the
        # 252 factor is correct.
        per_day_std = self._ewma_var**0.5
        return annualize_vol(per_day_std, periods_per_year=BARS_PER_YEAR["D"])

    def _compute_vol_scale(self) -> float:
        ann_vol = self._annualized_vol()
        if ann_vol is None or ann_vol <= 0:
            return 1.0
        target = float(self._config["vol_target_ann_pct"]) / 100.0
        scale_min = float(self._config["vol_scale_min"])
        scale_max = float(self._config["vol_scale_max"])
        return max(scale_min, min(scale_max, target / ann_vol))

    # ── Regime helpers ────────────────────────────────────────────────────

    def _compute_vix_scale(self) -> float:
        if self._vix_level is None:
            return 1.0
        t1 = float(self._config["vix_tier_1"])
        t2 = float(self._config["vix_tier_2"])
        t3 = float(self._config["vix_tier_3"])
        if self._vix_level < t1:
            return 1.0
        if self._vix_level < t2:
            return 0.75
        if self._vix_level < t3:
            return 0.50
        return 0.25

    def _compute_atr_scale(self) -> float:
        if not self._atr_percentiles:
            return 1.0
        max_atr = max(self._atr_percentiles.values())
        low = float(self._config["atr_pct_low"])
        high = float(self._config["atr_pct_high"])
        extreme = float(self._config["atr_pct_extreme"])
        if max_atr < low:
            return 1.25
        if max_atr < high:
            return 1.0
        if max_atr < extreme:
            return 0.50
        return 0.25

    def _compute_regime_scale(self) -> float:
        return min(self._compute_vix_scale(), self._compute_atr_scale())

    # ── Core health check ─────────────────────────────────────────────────

    def _check_portfolio_health(self, now_ts: pd.Timestamp) -> None:
        max_dd = float(self._config["portfolio_max_dd_pct"]) / 100.0
        heat = float(self._config["portfolio_heat_scale_pct"]) / 100.0
        max_single = float(self._config["portfolio_max_single_pct"]) / 100.0

        port_dd = self._portfolio_drawdown()
        total = self._total_equity()

        # Kill switch is always evaluated on every tick.
        if port_dd < -max_dd:
            self._halt_all = True
            self._halt_reason = f"portfolio_dd={port_dd * 100:.2f}% exceeds {max_dd * 100:.1f}%"
            self._scale_factor = 0.0
            self._dd_scale = 0.0
            self._persist_halt_state(operator="auto-kill", cleared=False)
            logger.critical(
                "[PortfolioRM] KILL SWITCH -- portfolio DD %.2f%% > %.1f%%.",
                port_dd * 100,
                max_dd * 100,
            )
            return

        # DD heat always evaluated on every tick.
        if port_dd < -heat:
            heat_fraction = abs(port_dd) / max_dd
            self._dd_scale = max(0.25, 1.0 - heat_fraction)
        else:
            self._dd_scale = 1.0

        # Wall-clock daily gate: recompute vol + regime scales once per date.
        today = now_ts.date()
        if self._last_daily_date != today:
            self._last_daily_date = today
            self._recompute_daily_vol()

        self._vol_scale = self._compute_vol_scale()
        self._regime_scale = self._compute_regime_scale()
        self._scale_factor = min(self._dd_scale, self._vol_scale, self._regime_scale)

        # Correlation check once per date too (cheap, but verbose).
        if self._last_corr_date != today:
            self._last_corr_date = today
            try:
                self.check_correlation_regime()
            except Exception as e:
                logger.exception("[PortfolioRM] correlation check failed: %s", e)

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

        # Concentration warning.
        if total > 0:
            for sid, state in self._strategies.items():
                weight = state.current_equity / total
                if weight > max_single:
                    logger.warning(
                        "[PortfolioRM] Concentration: '%s' holds %.1f%% (limit %.1f%%).",
                        sid,
                        weight * 100,
                        max_single * 100,
                    )

    # ── Halt persistence ──────────────────────────────────────────────────

    def _load_halt_state(self) -> None:
        try:
            if _HALT_STATE_PATH.exists():
                data = json.loads(_HALT_STATE_PATH.read_text())
                if data.get("halted"):
                    self._halt_all = True
                    self._halt_reason = data.get("reason", "persisted-halt")
                    logger.critical(
                        "[PortfolioRM] Loaded persisted HALT state -- "
                        "reason=%s since=%s. Operator must call reset_halt() "
                        "to resume.",
                        self._halt_reason,
                        data.get("at"),
                    )
        except Exception as e:
            logger.exception("[PortfolioRM] halt-state load failed: %s", e)

    def _persist_halt_state(self, operator: str, cleared: bool) -> None:
        try:
            _HALT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "halted": self._halt_all,
                "reason": self._halt_reason,
                "operator": operator,
                "at": datetime.now(timezone.utc).isoformat(),
                "cleared": cleared,
            }
            _HALT_STATE_PATH.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.exception("[PortfolioRM] halt-state persist failed: %s", e)


# ── Module-level singleton ─────────────────────────────────────────────────────


def _load_portfolio_config() -> dict:
    import tomllib

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
