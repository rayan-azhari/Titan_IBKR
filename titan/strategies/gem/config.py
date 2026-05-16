"""TOML-loadable configuration for the GEM live strategy."""

from __future__ import annotations

from typing import Literal

from nautilus_trader.config import StrategyConfig


class GemStrategyConfig(StrategyConfig, frozen=False, kw_only=True):
    """Configuration for the GEM Dual Momentum live strategy.

    Loaded from ``config/gem_voltarget_lev2.toml`` via the standard pattern.

    Naming convention: every NT Strategy expects a ``StrategyConfig``
    subclass; field names are flat (no nested dicts).

    Execution mode:
      * ``etf``: trade SPY/EFA/IEF directly as ETFs. Caps weight at 1.0
        per leg (no leverage). Operator gets the 1x exposure profile.
      * ``mes``: trade MES futures for the SPY leg (provides leverage),
        ETFs for EFA/IEF. The production cell expects max_leverage=2.0
        which is achievable only with futures (note: at J5's vol_target=0.05
        the cap rarely binds — L57).
    """

    # ── Instruments and bar types ─────────────────────────────────────────
    spy_instrument_id: str  # e.g. "SPY.ARCA" -- the ETF used in ETF mode
    efa_instrument_id: str  # e.g. "EFA.ARCA"
    ief_instrument_id: str  # e.g. "IEF.ARCA"
    # e.g. "MES.CME-MES-202506" front-month MES futures contract id.
    mes_instrument_id: str | None = None
    spy_bar_type_d: str  # e.g. "SPY.ARCA-1-DAY-LAST-EXTERNAL"
    efa_bar_type_d: str
    ief_bar_type_d: str
    vix_instrument_id: str | None = None  # optional regime indicator (VIX index)
    hyg_instrument_id: str | None = None  # optional regime indicator (HYG ETF)
    vix_bar_type_d: str | None = None
    hyg_bar_type_d: str | None = None

    # Ticker labels (for parquet warmup + logging).
    ticker_spy: str = "SPY"
    ticker_efa: str = "EFA"
    ticker_ief: str = "IEF"
    ticker_vix: str = "VIX"
    ticker_hyg: str = "HYG"

    # ── Execution ─────────────────────────────────────────────────────────
    execution_mode: Literal["etf", "mes"] = "etf"
    # Rebalance trigger: if the desired-vs-current weight delta exceeds this
    # threshold (absolute weight units), submit a rebalance order. Caps
    # daily turnover when vol-target nudges are small. Carver-style buffering.
    rebalance_threshold_weight: float = 0.05

    # ── Strategy parameters (mirrors GemConfig in research/gem/gem_strategy.py) ──
    lookback_blend_str: str = "3,6,12"  # comma-separated, parsed to tuple
    absolute_gate_lookback_months: int = 12
    buffer_pct: float = 0.005
    defensive_switch: bool = True

    # Vol-target overlay (Step 3)
    ann_vol_target: float = 0.10
    vol_lookback_days: int = 20
    max_leverage: float = 2.0  # C12 setting

    # J4 (2026-05-15) noise-robust redesign — added the EWMA path.
    # J5 (2026-05-16) hybrid re-audit — promoted halflife=60 / vol_target=0.05.
    # Defaults preserve pre-J4 behaviour (vol_estimator_kind="rolling_std",
    # max_weight_delta_per_bar=None, vol_target_kind="fixed"). The CURRENT
    # production cell (J5 P_hl60_vt05) is configured via
    # config/gem_voltarget_lev2.toml: vol_estimator_kind="ewma",
    # vol_estimator_halflife=60, ann_vol_target=0.05. See
    # docs/strategies/gem-dual-momentum.md for the full audit lineage.
    vol_estimator_kind: Literal["rolling_std", "ewma"] = "rolling_std"
    vol_estimator_halflife: int = 40
    max_weight_delta_per_bar: float | None = None
    vol_target_kind: Literal["fixed", "rolling_quantile"] = "fixed"
    vol_target_quantile: float = 0.40
    vol_target_quantile_window: int = 252

    # Conditional stress gate (Step 4) -- OFF by default; C8/C12 don't use it
    stress_gate_enabled: bool = False
    stress_realised_vol_threshold: float = 0.20
    stress_realised_vol_window: int = 20
    stress_vix_threshold: float | None = None
    stress_credit_z_threshold: float | None = None
    stress_credit_z_window: int = 60

    # Drawdown circuit breaker (Step 5) -- OFF by default for C12
    dd_breaker_enabled: bool = False
    dd_breaker_haircut_threshold: float = -0.10
    dd_breaker_haircut_scale: float = 0.50
    dd_breaker_flat_threshold: float = -0.15
    dd_breaker_flat_bars: int = 21
    dd_breaker_recovery_threshold: float = -0.05

    # ── Risk + operations ─────────────────────────────────────────────────
    warmup_bars: int = 380  # enough for the longest (18m) lookback + buffer
    initial_equity: float = 30_000.0  # seed capital in base ccy
    base_ccy: str = "USD"
    cost_bps_per_turnover: float = 1.5  # informational; not applied to live orders
