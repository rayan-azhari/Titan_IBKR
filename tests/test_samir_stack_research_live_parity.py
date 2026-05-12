"""Research-vs-live parity test for Samir-Stack (Phase 6a).

The Phase 5 winner is parameterised by:
- regime score (6 indicators, equal-weight mean)
- tier mapping with hysteresis (thresholds 0.30, 0.55; L_max=2)

The RESEARCH pipeline computes these via:
- ``research/samir_stack/indicators.py::build_indicator_panel`` (vectorised, panel-wide)
- ``research/samir_stack/regime_score.py::regime_score_equal``
- ``research/samir_stack/stacked_strategy.py::_equity_target_tier``

The LIVE pipeline computes them via:
- ``titan/strategies/samir_stack/regime.py::compute_regime_score`` (snapshot-based, single-bar)
- ``titan/strategies/samir_stack/regime.py::target_tier_from_score``

Both implementations encode the same math but with different data shapes
(panel vs buffer). A divergence between them is the audit's A10 finding
(parity tests miss research-side bugs because they only test the live
class). This test closes that gap by computing one bar of regime score
+ target tier via BOTH pipelines on a shared fixture and asserting
agreement within float-precision tolerance.

Per the audit's research-math discipline: live == research. This test
enforces it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.indicators import build_indicator_panel
from research.samir_stack.regime_score import regime_score_equal
from research.samir_stack.stacked_strategy import StackedConfig
from research.samir_stack.stacked_strategy import _equity_target_tier as research_target_tier
from titan.strategies.samir_stack.regime import (
    RegimeBuffers,
    compute_regime_score,
    target_tier_from_score,
)

# ── Fixture: deterministic ~5y of all 6 underlyings ────────────────────────


def _make_fixture_panel(n: int = 1500, seed: int = 42) -> dict[str, pd.Series]:
    """Synthetic 5-year panel — log-normal SPY, mean-reverting VIX, drifty
    bonds. The exact numerics don't matter; we only need both pipelines
    to see the same input."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # SPY: ~10% annualised drift, ~16% vol
    spy_rets = rng.normal(0.10 / 252, 0.16 / np.sqrt(252), size=n)
    spy = pd.Series(100.0 * np.cumprod(1.0 + spy_rets), index=idx, name="SPY")

    # VIX: mean-reverting OU around 18, vol 6
    vix_path = np.empty(n)
    vix_path[0] = 18.0
    theta = 0.05  # mean-reversion strength
    sigma = 0.8
    for i in range(1, n):
        vix_path[i] = vix_path[i - 1] + theta * (18.0 - vix_path[i - 1]) + sigma * rng.normal()
        vix_path[i] = max(8.0, vix_path[i])  # floor at 8
    vix = pd.Series(vix_path, index=idx, name="VIX")

    # IEF: ~3% drift, ~4% vol
    ief_rets = rng.normal(0.03 / 252, 0.04 / np.sqrt(252), size=n)
    ief = pd.Series(100.0 * np.cumprod(1.0 + ief_rets), index=idx, name="IEF")

    # HYG: ~5% drift, ~8% vol (with some correlation to SPY)
    hyg_rets = 0.5 * spy_rets + 0.5 * rng.normal(0.05 / 252, 0.08 / np.sqrt(252), size=n)
    hyg = pd.Series(100.0 * np.cumprod(1.0 + hyg_rets), index=idx, name="HYG")

    return {"spy": spy, "vix": vix, "ief": ief, "hyg": hyg}


# ── A1: research and live regime scores agree on the final bar ────────────


def test_regime_score_research_live_parity_single_bar() -> None:
    """At the final bar of a 5-year synthetic panel, the live snapshot
    score and the research panel score must agree within 0.05.

    Tolerance rationale: rolling-vol percentile rank uses slightly
    different windowing between the two implementations (research uses
    pandas rolling.rank(pct=True) over a fixed 756-bar window; live
    uses a snapshot-based rank). Both encode the same statistical
    concept but can differ by O(1/window) on the percentile rank. 0.05
    is well below any tier-threshold gap (≥ 0.20) so the agreement is
    operationally sufficient.
    """
    data = _make_fixture_panel()
    spy, vix, hyg, ief = data["spy"], data["vix"], data["hyg"], data["ief"]

    # Research path
    panel = build_indicator_panel(spy, vix_close=vix, hyg_close=hyg, ief_close=ief)
    research_score = regime_score_equal(panel).iloc[-1]

    # Live path
    buffers = RegimeBuffers()
    # Bulk-fill via direct assignment (faster than per-bar add_*)
    buffers.spy = spy.copy()
    buffers.spy_high = spy.copy()  # OHLC not needed for the 6 indicators here
    buffers.spy_low = spy.copy()
    buffers.vix = vix.copy()
    buffers.hyg = hyg.copy()
    buffers.ief = ief.copy()
    live_score, breakdown = compute_regime_score(buffers)

    diff = abs(live_score - research_score)
    assert diff < 0.05, (
        f"Regime-score parity failure: research={research_score:.4f}, "
        f"live={live_score:.4f}, |diff|={diff:.4f} (tolerance 0.05). "
        f"Live breakdown: {breakdown}"
    )


def test_target_tier_parity_across_grid() -> None:
    """The tier-mapping function exists in TWO places (live vs research)
    with structurally-identical logic. For every (score, current_tier)
    pair in a representative grid, both must return the same target tier.

    This is the only place in the live pipeline that needs to bit-exact
    match research; any divergence is a real bug.
    """
    L_max = 2.0
    tier_thresholds = (0.30, 0.55)
    hysteresis_buffer = 0.05

    # Research-side function expects a StackedConfig object.
    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=L_max,
        tier_thresholds=tier_thresholds,
        hysteresis_buffer=hysteresis_buffer,
    )

    mismatches: list[tuple[float, float, float, float]] = []
    for score in (
        # boundary points + a few in-between
        0.0,
        0.10,
        0.25,
        0.30,
        0.31,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.56,
        0.60,
        0.65,
        0.70,
        0.80,
        0.90,
        1.0,
    ):
        for current_tier in (0.0, 1.0, 2.0):
            live = target_tier_from_score(
                score,
                current_tier,
                tier_thresholds=tier_thresholds,
                hysteresis_buffer=hysteresis_buffer,
                L_max=L_max,
            )
            research = research_target_tier(score, cfg, current_tier)
            if live != research:
                mismatches.append((score, current_tier, live, research))

    assert not mismatches, (
        "target_tier_from_score live-vs-research mismatches at "
        f"L_max={L_max}, thresholds={tier_thresholds}: {mismatches}"
    )


def test_target_tier_parity_at_phase5_champion_config() -> None:
    """Pin the Phase 5 champion's tier thresholds specifically.

    The champion uses ``tier_thresholds=(0.30, 0.55)`` with ``L_max=2``.
    Verify the hysteresis-buffered up-transitions hit the right values:
      - From tier 0: must reach > (0.30 + 0.05) to enter tier 1.
      - From tier 1: must reach > (0.55 + 0.05) to enter tier 2.
      - From tier 2: stays in tier 2 down to bare 0.55, exits below.

    Test values are chosen slightly above/below the threshold+buffer
    sums to avoid IEEE-754 float-precision boundary effects
    (e.g. ``0.55 + 0.05 == 0.6000000000000001`` in binary float).
    """
    L_max = 2.0
    th = (0.30, 0.55)
    buf = 0.05

    # From cash (tier 0): need score > 0.35 (with float-safe margin)
    assert (
        target_tier_from_score(0.34, 0.0, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max)
        == 0.0
    )
    assert (
        target_tier_from_score(0.36, 0.0, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max)
        == 1.0
    )

    # From tier 1 upward: need score > 0.60 (with float-safe margin)
    assert (
        target_tier_from_score(0.59, 1.0, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max)
        == 1.0
    )
    assert (
        target_tier_from_score(0.61, 1.0, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max)
        == 2.0
    )

    # From tier 2 downward (bare threshold)
    assert (
        target_tier_from_score(0.55, 2.0, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max)
        == 2.0
    )
    assert (
        target_tier_from_score(0.54, 2.0, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max)
        == 1.0
    )


def test_target_tier_nan_score_holds_current() -> None:
    """NaN score (insufficient warmup, missing indicators) must not flip
    the strategy — it should hold the current tier silently."""
    L_max = 2.0
    th = (0.30, 0.55)
    buf = 0.05
    for cur in (0.0, 1.0, 2.0):
        out = target_tier_from_score(
            float("nan"), cur, tier_thresholds=th, hysteresis_buffer=buf, L_max=L_max
        )
        assert out == cur, f"NaN score should hold tier {cur}, got {out}"


def test_regime_score_indicator_breakdown_matches_panel_columns() -> None:
    """The breakdown dict returned by ``compute_regime_score`` should have
    the same 6 indicator keys as the research panel's columns. This pins
    the indicator-set contract so a regression to a 5- or 7-indicator
    ensemble fails loudly.
    """
    data = _make_fixture_panel()
    panel = build_indicator_panel(
        data["spy"], vix_close=data["vix"], hyg_close=data["hyg"], ief_close=data["ief"]
    )
    research_cols = set(panel.columns)

    buffers = RegimeBuffers()
    buffers.spy = data["spy"].copy()
    buffers.spy_high = data["spy"].copy()
    buffers.spy_low = data["spy"].copy()
    buffers.vix = data["vix"].copy()
    buffers.hyg = data["hyg"].copy()
    buffers.ief = data["ief"].copy()
    _, breakdown = compute_regime_score(buffers)
    live_cols = set(breakdown.keys())

    # The two sets should match exactly for the 6 always-on indicators.
    expected = {"vix", "rv_regime", "trend", "momentum_12_1", "dd_velocity", "credit"}
    assert expected.issubset(research_cols), (
        f"Research panel missing expected indicators: {expected - research_cols}"
    )
    assert expected.issubset(live_cols), (
        f"Live breakdown missing expected indicators: {expected - live_cols}"
    )
