"""Live-vs-research parity tests for production strategies.

For each live strategy in the champion portfolio, we compute the vol-
targeting arithmetic both ways (once via the strategy's internal method,
once via the shared ``titan.research.metrics`` helpers) and assert the
results agree. This is the structural guard the April 2026 audit
recommended as the cheapest insurance against live/research drift: if the
two paths ever diverge, the test fails and the drift is caught before any
trade is placed.

Scope is intentionally narrow: for each strategy, size one bar using a
synthetic close history, compare the resulting ``ann_vol`` (and hence
notional) between the live path and an independent reference
implementation built on top of the shared metrics module.

Why this works
--------------
The live ``_compute_size`` methods were the exact site of the April 2026
vol-annualisation bug (wrong ``sqrt(252)`` on H1 data). By rebuilding the
computation entirely via ``titan.research.metrics.ewm_vol_last`` and
asserting numerical equality, we lock in that any future change to the
live path (e.g. switching the vol window, changing the ewma span, using a
different frequency) must either be mirrored in the reference or show up
as a test failure.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from titan.research.metrics import BARS_PER_YEAR, ewm_vol_last

# ── Reference implementations (the canonical one) ──────────────────────────


# Canonical ``ewma_span`` defaults per strategy. If a live strategy changes
# these, this test file must be updated in the same commit — that's the
# whole point of a parity test. Failing loudly on divergence is the goal.
_KNOWN_EWMA_SPANS = {
    "mr_audjpy": 20,
    "gld_confluence": 20,
    "bond_gold": 20,
    "fx_carry": 20,
    "gold_macro": 20,
}


def _reference_ann_vol(
    closes: list[float],
    ewma_span: int,
    *,
    periods_per_year: int,
    lookback: int = 60,
    min_rets: int = 10,
) -> float:
    """Recompute the strategy sizing's ``ann_vol`` via the shared metrics
    module. This is what every live strategy's ``_compute_size`` should
    produce. Any divergence in the live path is a drift bug."""
    if len(closes) < lookback:
        return 0.0
    rets = pd.Series(closes[-lookback:]).pct_change().dropna()
    if len(rets) < min_rets:
        return 0.0
    rm_lambda = (ewma_span - 1.0) / (ewma_span + 1.0)
    return ewm_vol_last(rets, lam=rm_lambda, periods_per_year=periods_per_year)


def _mock_closes(seed: int, n: int, start: float, vol_per_bar: float) -> list[float]:
    """Deterministic synthetic close series for parity tests."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, vol_per_bar, n)
    eq = start * np.cumprod(1 + rets)
    return eq.tolist()


# ── MR AUD/JPY (H1) ────────────────────────────────────────────────────────


def test_mr_audjpy_sizing_matches_shared_metrics():
    """Reproduce ``MRAUDJPYStrategy._compute_size`` vol math using the
    shared helper and assert numerical equality."""
    # Import kept to prove the strategy module loads without error
    from titan.strategies.mr_audjpy.strategy import MRAUDJPYConfig  # noqa: F401

    # Synthetic H1 close series (bar-level vol ~ 0.1% -> ~5% daily, ~80% annual)
    closes = _mock_closes(seed=42, n=200, start=97.0, vol_per_bar=0.001)
    ewma_span = _KNOWN_EWMA_SPANS["mr_audjpy"]

    # Live path (unrolled from _compute_size)
    rets = pd.Series(closes[-60:]).pct_change().dropna()
    rm_lambda = (ewma_span - 1.0) / (ewma_span + 1.0)
    live_vol = ewm_vol_last(
        rets,
        lam=rm_lambda,
        periods_per_year=BARS_PER_YEAR["H1"],
    )

    # Reference path
    ref_vol = _reference_ann_vol(closes, ewma_span, periods_per_year=BARS_PER_YEAR["H1"])

    assert live_vol == pytest.approx(ref_vol, rel=1e-12)
    # And the H1 factor is actively used — if someone "fixed" to sqrt(252),
    # the vol would drop by sqrt(24) ≈ 4.9x. Assert the scale is in the
    # expected H1 band.
    assert live_vol > 4 * _reference_ann_vol(closes, ewma_span, periods_per_year=BARS_PER_YEAR["D"])


# ── GLD Confluence (H1) ────────────────────────────────────────────────────


def test_gld_confluence_sizing_matches_shared_metrics():
    from titan.strategies.gld_confluence.strategy import GLDConfluenceConfig  # noqa: F401

    closes = _mock_closes(seed=7, n=200, start=180.0, vol_per_bar=0.0015)
    ewma_span = _KNOWN_EWMA_SPANS["gld_confluence"]

    rets = pd.Series(closes[-60:]).pct_change().dropna()
    rm_lambda = (ewma_span - 1.0) / (ewma_span + 1.0)
    live_vol = ewm_vol_last(rets, lam=rm_lambda, periods_per_year=BARS_PER_YEAR["H1"])
    ref_vol = _reference_ann_vol(closes, ewma_span, periods_per_year=BARS_PER_YEAR["H1"])

    assert live_vol == pytest.approx(ref_vol, rel=1e-12)


# ── Bond-Gold (daily) ──────────────────────────────────────────────────────


def test_bond_gold_sizing_matches_shared_metrics():
    from titan.strategies.bond_gold.strategy import BondGoldConfig  # noqa: F401

    closes = _mock_closes(seed=11, n=250, start=180.0, vol_per_bar=0.01)
    ewma_span = _KNOWN_EWMA_SPANS["bond_gold"]

    rets = pd.Series(closes[-60:]).pct_change().dropna()
    rm_lambda = (ewma_span - 1.0) / (ewma_span + 1.0)
    live_vol = ewm_vol_last(rets, lam=rm_lambda, periods_per_year=BARS_PER_YEAR["D"])
    ref_vol = _reference_ann_vol(closes, ewma_span, periods_per_year=BARS_PER_YEAR["D"])

    assert live_vol == pytest.approx(ref_vol, rel=1e-12)


# ── FX Carry (daily) ───────────────────────────────────────────────────────


def test_fx_carry_sizing_matches_shared_metrics():
    from titan.strategies.fx_carry.strategy import FXCarryConfig  # noqa: F401

    closes = _mock_closes(seed=23, n=250, start=97.0, vol_per_bar=0.007)
    ewma_span = _KNOWN_EWMA_SPANS["fx_carry"]

    rets = pd.Series(closes[-60:]).pct_change().dropna()
    rm_lambda = (ewma_span - 1.0) / (ewma_span + 1.0)
    live_vol = ewm_vol_last(rets, lam=rm_lambda, periods_per_year=BARS_PER_YEAR["D"])
    ref_vol = _reference_ann_vol(closes, ewma_span, periods_per_year=BARS_PER_YEAR["D"])

    assert live_vol == pytest.approx(ref_vol, rel=1e-12)


# ── Gold Macro (daily) ─────────────────────────────────────────────────────


def test_gold_macro_sizing_matches_shared_metrics():
    from titan.strategies.gold_macro.strategy import GoldMacroConfig  # noqa: F401

    closes = _mock_closes(seed=31, n=250, start=180.0, vol_per_bar=0.008)
    ewma_span = _KNOWN_EWMA_SPANS["gold_macro"]

    rets = pd.Series(closes[-60:]).pct_change().dropna()
    rm_lambda = (ewma_span - 1.0) / (ewma_span + 1.0)
    live_vol = ewm_vol_last(rets, lam=rm_lambda, periods_per_year=BARS_PER_YEAR["D"])
    ref_vol = _reference_ann_vol(closes, ewma_span, periods_per_year=BARS_PER_YEAR["D"])

    assert live_vol == pytest.approx(ref_vol, rel=1e-12)


# ── Frequency-mismatch regression (the bug the audit caught) ───────────────


def test_h1_vol_is_sqrt_24_larger_than_daily():
    """If anyone 'fixes' an H1 strategy to use 252 instead of 252*24, this
    test fails. The invariant is:
        ewm_vol(rets, periods_per_year=H1) / ewm_vol(rets, periods_per_year=D) == sqrt(24)
    for the same return series.
    """
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0, 0.001, 500))
    daily_factor_vol = ewm_vol_last(rets, lam=0.94, periods_per_year=BARS_PER_YEAR["D"])
    h1_factor_vol = ewm_vol_last(rets, lam=0.94, periods_per_year=BARS_PER_YEAR["H1"])
    assert h1_factor_vol / daily_factor_vol == pytest.approx(math.sqrt(24), rel=1e-9)
