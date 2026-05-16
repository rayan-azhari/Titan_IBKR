"""V3.6 A10 parity tests -- GEM research function vs live class.

Required by the V2.0 deployment discipline: before any live trading,
a parity test must demonstrate that the live class produces bit-exact
the same outputs as the research-side computation for a known input.

The live class :py:class:`titan.strategies.gem.live_logic.GemLiveLogic`
delegates to the research function ``gem_target_weights``, so parity
is satisfied BY CONSTRUCTION. The tests below assert this property
under representative configurations and demonstrate that the live
class produces the right INTEGER MES contract counts via the sizing
layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.gem.gem_strategy import GEM_UNIVERSE, GemConfig, gem_target_weights
from titan.strategies.gem.live_logic import GemLiveLogic
from titan.strategies.gem.sizing import size_with_etf_only, size_with_mes

# ── Fixtures ──────────────────────────────────────────────────────────────


def _synthetic_history(n_years: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 252 * n_years
    idx = pd.date_range("2020-01-02", periods=n, freq="B").normalize()
    spy = 100 * np.cumprod(1 + rng.normal(0.10 / 252, 0.012, n))
    efa = 100 * np.cumprod(1 + rng.normal(0.05 / 252, 0.011, n))
    ief = 100 * np.cumprod(1 + rng.normal(0.02 / 252, 0.003, n))
    return pd.DataFrame({"SPY": spy, "EFA": efa, "IEF": ief}, index=idx)


def _c12_config() -> GemConfig:
    """The selected production cell -- C12 from the audit."""
    return GemConfig(
        lookback_blend=(3, 6, 12),
        buffer_pct=0.005,
        defensive_switch=True,
        ann_vol_target=0.10,
        vol_lookback_days=20,
        max_leverage=2.0,
    )


# ── Parity ────────────────────────────────────────────────────────────────


def test_parity_bulk_warmup_matches_research_exactly():
    """Bulk-loading history via add_bars_dataframe must produce IDENTICAL
    weights to running gem_target_weights on the same DataFrame.
    """
    closes = _synthetic_history(n_years=3)
    cfg = _c12_config()

    research_weights = gem_target_weights(closes, cfg=cfg)

    logic = GemLiveLogic(cfg=cfg)
    logic.add_bars_dataframe(closes)
    live_weights = logic.weight_history()

    pd.testing.assert_frame_equal(
        research_weights,
        live_weights,
        check_exact=True,  # bit-exact equality is the V3.6 A10 standard
        check_names=False,
    )


def test_parity_bar_by_bar_matches_research():
    """Iteratively adding one bar at a time and reading current_weights
    must agree with the research-side weight at that bar index.
    """
    closes = _synthetic_history(n_years=2)
    cfg = _c12_config()

    research_weights = gem_target_weights(closes, cfg=cfg)

    # Iterate -- relatively slow for long series, so we sample every 25 bars
    # post-warmup. Each sampled bar must match research exactly.
    sample_indices = list(range(0, len(closes), 25))
    for i in sample_indices:
        # Reset logic to history through bar i.
        slice_df = closes.iloc[: i + 1]
        sub_logic = GemLiveLogic(cfg=cfg)
        sub_logic.add_bars_dataframe(slice_df)
        live = sub_logic.current_weights()
        expected = {sym: float(research_weights.iloc[i][sym]) for sym in GEM_UNIVERSE}
        for sym in GEM_UNIVERSE:
            assert live[sym] == pytest.approx(expected[sym], abs=1e-12), (
                f"Bar {i} symbol {sym}: live={live[sym]} research={expected[sym]}"
            )


def test_live_logic_rejects_out_of_order_bars():
    cfg = _c12_config()
    logic = GemLiveLogic(cfg=cfg)
    t1 = pd.Timestamp("2024-01-02")
    t2 = pd.Timestamp("2024-01-03")
    logic.add_bar(t2, 100.0, 100.0, 100.0)
    with pytest.raises(ValueError, match="out-of-order"):
        logic.add_bar(t1, 99.0, 99.0, 99.0)


def test_live_logic_rejects_duplicate_bars():
    cfg = _c12_config()
    logic = GemLiveLogic(cfg=cfg)
    t1 = pd.Timestamp("2024-01-02")
    logic.add_bar(t1, 100.0, 100.0, 100.0)
    with pytest.raises(ValueError, match="duplicate"):
        logic.add_bar(t1, 101.0, 100.0, 100.0)


def test_live_logic_normalises_timestamps():
    """Times-of-day should be normalised to 00:00 (L20 of V3.6)."""
    cfg = _c12_config()
    logic = GemLiveLogic(cfg=cfg)
    # Add a bar with non-zero time-of-day.
    t1 = pd.Timestamp("2024-01-02 14:30:00")
    logic.add_bar(t1, 100.0, 100.0, 100.0)
    # Internal history should have normalised the timestamp.
    assert logic._closes_history.index[0] == pd.Timestamp("2024-01-02")


# ── Sizing ────────────────────────────────────────────────────────────────


def test_size_with_mes_full_long_spy():
    """At weight=1.0 in SPY on a $100k NAV with ES at 5800, expect ~3 MES."""
    decisions = size_with_mes(
        target_weights={"SPY": 1.0, "EFA": 0.0, "IEF": 0.0},
        nav_usd=100_000.0,
        es_price=5800.0,
        efa_price=80.0,
        ief_price=95.0,
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.execution_vehicle == "MES"
    # $100k / ($5 × 5800) = 3.448 → rounded to 3.
    assert d.rounded_quantity == 3
    assert d.target_notional_usd == pytest.approx(100_000.0)


def test_size_with_mes_lev2_long_spy():
    """At weight=2.0 in SPY (C12 production cell), expect ~7 MES on $100k."""
    decisions = size_with_mes(
        target_weights={"SPY": 2.0, "EFA": 0.0, "IEF": 0.0},
        nav_usd=100_000.0,
        es_price=5800.0,
        efa_price=80.0,
        ief_price=95.0,
    )
    d = decisions[0]
    # $200k notional / $29k per MES = 6.896 → rounded to 7.
    assert d.rounded_quantity == 7
    assert d.target_notional_usd == pytest.approx(200_000.0)


def test_size_with_mes_defensive_ief_only():
    """When the defensive switch is active (all weight in IEF), MES sizing
    should output a single IEF decision with no MES.
    """
    decisions = size_with_mes(
        target_weights={"SPY": 0.0, "EFA": 0.0, "IEF": 1.0},
        nav_usd=100_000.0,
        es_price=5800.0,
        efa_price=80.0,
        ief_price=95.0,
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.symbol == "IEF"
    # $100k / $95 = 1052.6 shares → rounded to 1053.
    assert d.rounded_quantity == 1053


def test_size_with_mes_voltargeted_partial_split():
    """Vol-targeted weights split between SPY (scaled) and IEF (cash remainder)."""
    decisions = size_with_mes(
        target_weights={"SPY": 0.6, "EFA": 0.0, "IEF": 0.4},
        nav_usd=100_000.0,
        es_price=5800.0,
        efa_price=80.0,
        ief_price=95.0,
    )
    assert len(decisions) == 2
    by_sym = {d.symbol: d for d in decisions}
    # SPY: 60% of $100k = $60k notional / $29k per MES = 2.07 → 2 contracts.
    assert by_sym["MES"].rounded_quantity == 2
    # IEF: 40% of $100k = $40k / $95 = 421.05 → 421 shares.
    assert by_sym["IEF"].rounded_quantity == 421


def test_size_with_etf_only_caps_leverage_at_1():
    """ETF-only execution cannot exceed weight=1.0 per leg."""
    decisions = size_with_etf_only(
        target_weights={"SPY": 2.0, "EFA": 0.0, "IEF": 0.0},
        nav_usd=100_000.0,
        spy_price=580.0,
        efa_price=80.0,
        ief_price=95.0,
    )
    d = decisions[0]
    # Effective weight capped at 1.0. $100k / $580 = 172.4 → 172 shares.
    assert d.rounded_quantity == 172
    assert d.target_weight == 1.0  # capped


def test_sizing_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nav_usd"):
        size_with_mes(
            target_weights={"SPY": 1.0, "EFA": 0.0, "IEF": 0.0},
            nav_usd=0.0,
            es_price=5800.0,
            efa_price=80.0,
            ief_price=95.0,
        )
    with pytest.raises(ValueError, match="prices"):
        size_with_mes(
            target_weights={"SPY": 1.0, "EFA": 0.0, "IEF": 0.0},
            nav_usd=100_000.0,
            es_price=-1.0,
            efa_price=80.0,
            ief_price=95.0,
        )


# ── J4 production cell (A1_ewma_hl40) -- parity + plumbing ─────────────────


def _a1_ewma_config() -> GemConfig:
    """The current production cell after the J4 noise-robust redesign.

    Same as _c12_config() except vol_estimator_kind/halflife switched.
    """
    return GemConfig(
        lookback_blend=(3, 6, 12),
        buffer_pct=0.005,
        defensive_switch=True,
        ann_vol_target=0.10,
        vol_lookback_days=20,
        max_leverage=2.0,
        vol_estimator_kind="ewma",
        vol_estimator_halflife=40,
    )


def test_parity_a1_ewma_bulk_warmup_matches_research_exactly():
    """A1_ewma_hl40 (J4 production) live class must agree bit-exact with
    the research weights. Exercises the new vol_estimator_kind field
    path through GemConfig and gem_target_weights."""
    closes = _synthetic_history(n_years=3, seed=11)
    cfg = _a1_ewma_config()

    research_weights = gem_target_weights(closes, cfg=cfg)

    logic = GemLiveLogic(cfg=cfg)
    logic.add_bars_dataframe(closes)
    live_weights = logic.weight_history()

    pd.testing.assert_frame_equal(
        research_weights,
        live_weights,
        check_exact=True,
        check_names=False,
    )


def test_a1_ewma_produces_different_weights_than_c12_baseline():
    """The whole point of the J4 deployment: the EWMA path must produce
    measurably different weights than the rolling-std baseline.
    """
    closes = _synthetic_history(n_years=3, seed=12)
    w_c12 = gem_target_weights(closes, cfg=_c12_config())
    w_a1 = gem_target_weights(closes, cfg=_a1_ewma_config())
    diff = (w_c12 - w_a1).abs().sum().sum()
    assert diff > 0.1, (
        f"A1 EWMA produced same weights as C12 baseline -- the J4 field is not "
        f"propagating. Total |Δw| = {diff:.4f}"
    )


def test_strategy_config_loads_j4_fields_from_toml():
    """The TOML config schema must accept and propagate the J4 fields.

    Exercises the loading pipeline: TOML -> GemStrategyConfig -> the
    research GemConfig translation in titan/strategies/gem/strategy.py
    (which is mediated by GemStrategyConfig's defaults if the TOML omits
    them). This test ensures the new fields land on the live config
    object with the right values for the A1 production cell.
    """
    from titan.strategies.gem.config import GemStrategyConfig

    # Mimic what NautilusTrader's TOML loader does when handed the new
    # production TOML: it instantiates GemStrategyConfig from the parsed
    # dict. Required positional / non-default fields stubbed.
    cfg = GemStrategyConfig(
        spy_instrument_id="SPY.ARCA",
        efa_instrument_id="EFA.ARCA",
        ief_instrument_id="IEF.ARCA",
        spy_bar_type_d="SPY.ARCA-1-DAY-LAST-EXTERNAL",
        efa_bar_type_d="EFA.ARCA-1-DAY-LAST-EXTERNAL",
        ief_bar_type_d="IEF.ARCA-1-DAY-LAST-EXTERNAL",
        # J4 fields from the production TOML:
        vol_estimator_kind="ewma",
        vol_estimator_halflife=40,
    )
    assert cfg.vol_estimator_kind == "ewma"
    assert cfg.vol_estimator_halflife == 40
    # Other J4 fields should have their L31 defaults.
    assert cfg.max_weight_delta_per_bar is None
    assert cfg.vol_target_kind == "fixed"
    assert cfg.vol_target_quantile == 0.40
    assert cfg.vol_target_quantile_window == 252
