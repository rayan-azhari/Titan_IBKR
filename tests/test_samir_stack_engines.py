"""Phase 2 engine + sleeve tests.

Covers two distinct concerns:

1. The three ``EquityEngine`` implementations produce returns that match
   the underlying functions they wrap (no off-by-one, no signature
   surprises).

2. The Phase 2 gate from the 2026-05-12 remediation plan:
   ``run_stacked_strategy`` with explicit ``equity_engine=SyntheticETFEngine``
   and ``bond_sleeve=StaticBondSleeve(IEF)`` must reproduce the legacy
   (no-engine-no-sleeve) call **bit-exactly**. This proves the refactor
   is non-behavioural and any future engine/sleeve choice is purely a
   sleeve choice, not a stealth change to the state machine.

3. ``RotationBondSleeve([IGLT, IGLS])`` general-case lag semantics match
   the contract documented in ``engines.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.samir_stack.capitulation import CapitulationConfig
from research.samir_stack.engines import (
    FuturesEngine,
    MarginEngine,
    RotationBondSleeve,
    StaticBondSleeve,
    SyntheticETFEngine,
    assert_engine_matches_legacy,
)
from research.samir_stack.margin_model import (
    constant_leverage_margin_returns,
    futures_returns_tr,
)
from research.samir_stack.stacked_strategy import StackedConfig, run_stacked_strategy
from research.samir_stack.synthetic_3x import synthetic_leveraged_returns

# ── Fixtures: small synthetic series so tests run in <1 s ────────────────


def _make_underlying(n: int = 800, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # log-normal random walk with ~10% annualised drift, ~16% vol
    daily_drift = 0.10 / 252
    daily_vol = 0.16 / np.sqrt(252)
    rets = rng.normal(daily_drift, daily_vol, size=n)
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx, name="SPY")


def _make_bond(
    name: str, n: int = 800, mu: float = 0.03, sigma: float = 0.04, seed: int = 7
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(mu / 252, sigma / np.sqrt(252), size=n)
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx, name=name)


# ── EquityEngine implementations match their underlying functions ────────


def test_synthetic_etf_engine_matches_legacy_function() -> None:
    """SyntheticETFEngine(L=3) ≡ synthetic_leveraged_returns(spy, L=3)."""
    spy = _make_underlying()
    engine = SyntheticETFEngine(ter_annual=0.0075)
    assert_engine_matches_legacy(
        engine,
        lambda s, leverage: synthetic_leveraged_returns(s, leverage=leverage, ter_annual=0.0075),
        spy,
        leverage=3.0,
    )


def test_synthetic_etf_engine_zeros_ter_at_L1() -> None:
    """At L=1 the SyntheticETFEngine should not charge the leveraged-ETF
    TER (mirroring the existing convention used by run_stacked_strategy:
    ``ter_annual=cfg.leverage_ter_annual if L > 1.0 else 0.0``)."""
    spy = _make_underlying()
    engine = SyntheticETFEngine(ter_annual=0.0075)
    engine_l1 = engine.daily_returns(spy, leverage=1.0).dropna()
    legacy_l1 = synthetic_leveraged_returns(spy, leverage=1.0, ter_annual=0.0).dropna()
    common = engine_l1.index.intersection(legacy_l1.index)
    assert float(np.abs(engine_l1.loc[common] - legacy_l1.loc[common]).max()) < 1e-12


def test_margin_engine_matches_legacy_function() -> None:
    spy = _make_underlying()
    engine = MarginEngine()
    assert_engine_matches_legacy(
        engine,
        lambda s, leverage: constant_leverage_margin_returns(s, leverage=leverage),
        spy,
        leverage=2.0,
    )


def test_futures_engine_matches_legacy_function() -> None:
    spy = _make_underlying()
    engine = FuturesEngine()
    assert_engine_matches_legacy(
        engine,
        lambda s, leverage: futures_returns_tr(s, leverage=leverage),
        spy,
        leverage=3.0,
    )


# ── BondSleeve implementations ───────────────────────────────────────────


def test_static_bond_sleeve_returns_pct_change() -> None:
    ief = _make_bond("IEF", mu=0.03, sigma=0.04, seed=11)
    sleeve = StaticBondSleeve(name="IEF", close=ief)
    out = sleeve.daily_returns(ief.index)
    expected = ief.pct_change().fillna(0.0)
    common = out.index.intersection(expected.index)
    assert float(np.abs(out.loc[common] - expected.loc[common]).max()) < 1e-12


def test_rotation_sleeve_general_case_lags_winner_by_one_bar() -> None:
    """For the general N-instrument case (not the IEF/HYG shortcut), the
    same lag semantics must hold: winner_at_(t-1) decides return_at_t.

    Fixture: IGLT rises monotonically; IGLS flatlines for 60 days then
    jumps +5%. On flip day t*, the lagged winner is still IGLT — the
    sleeve must NOT pay IGLS's same-bar +5%.
    """
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    iglt = pd.Series(np.linspace(100.0, 110.0, n), index=idx, name="IGLT")
    igls_vals = np.full(n, 100.0)
    flip = 70
    igls_vals[flip] = 105.0  # +5% jump
    igls = pd.Series(igls_vals, index=idx, name="IGLS")

    sleeve = RotationBondSleeve(
        name="UK_gilts", candidates={"IGLT": iglt, "IGLS": igls}, lookback_days=60
    )
    out = sleeve.daily_returns(idx)

    iglt_ret_flip = float(iglt.pct_change().iloc[flip])
    igls_ret_flip = float(igls.pct_change().iloc[flip])
    out_flip = float(out.iloc[flip])

    assert out_flip < 0.04, (
        f"General-case rotation sleeve shows same-bar look-ahead: paid "
        f"{out_flip:.4f} on flip day, IGLS jumped {igls_ret_flip:.4f}. "
        f"Expected IGLT's lagged return ({iglt_ret_flip:.4f})."
    )


def test_rotation_sleeve_ief_hyg_uses_phase1_helper() -> None:
    """The IEF/HYG shortcut MUST delegate to the Phase 1-corrected
    bond_rotation_returns. We pin this by feeding the same fixture into
    both and asserting equality."""
    ief = _make_bond("IEF", mu=0.03, sigma=0.04, seed=11)
    hyg = _make_bond("HYG", mu=0.05, sigma=0.08, seed=13)
    sleeve = RotationBondSleeve(name="bonds", candidates={"IEF": ief, "HYG": hyg}, lookback_days=60)
    from research.samir_stack.run_samir_improvements import bond_rotation_returns

    a = sleeve.daily_returns(ief.index)
    b = bond_rotation_returns(ief, hyg, lookback_days=60).reindex(ief.index).fillna(0.0)
    common = a.index.intersection(b.index)
    assert float(np.abs(a.loc[common] - b.loc[common]).max()) < 1e-12


# ── Phase 2 gate: bit-exact baseline reproduction ────────────────────────


def test_run_stacked_strategy_engine_kwarg_is_bitexact_with_legacy() -> None:
    """The Phase 2 gate from the remediation plan §3 Phase 2:

    ``run_stacked_strategy(...)`` with no engine/sleeve kwargs (legacy
    path) MUST produce the same equity curve as
    ``run_stacked_strategy(..., equity_engine=SyntheticETFEngine(...),
    bond_sleeve=StaticBondSleeve(IEF))``.

    This proves the refactor introduced zero behavioural change: any
    future Phase-3+ result that uses a different engine is purely a
    sleeve choice, not a stealth state-machine change.
    """
    from research.samir_stack.indicators import build_indicator_panel
    from research.samir_stack.regime_score import regime_score_equal

    # Build a deterministic small panel so the test is fast.
    spy = _make_underlying(n=1200, seed=42)
    ief = _make_bond("IEF", n=1200, mu=0.03, sigma=0.04, seed=11)
    vix = _make_bond("VIX", n=1200, mu=0.0, sigma=0.20, seed=17).clip(lower=8.0)
    hyg = _make_bond("HYG", n=1200, mu=0.05, sigma=0.08, seed=23)
    panel = build_indicator_panel(spy, vix_close=vix, hyg_close=hyg, ief_close=ief)
    score = regime_score_equal(panel)

    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=3.0,
        tier_thresholds=(0.30, 0.50, 0.75),
    )

    legacy = run_stacked_strategy(spy, ief, score, cfg)
    new = run_stacked_strategy(
        spy,
        ief,
        score,
        cfg,
        equity_engine=SyntheticETFEngine(ter_annual=cfg.leverage_ter_annual),
        bond_sleeve=StaticBondSleeve(name="IEF", close=ief),
    )

    common = legacy.index.intersection(new.index)
    # Equity curve must match to floating-point precision
    diff_eq = float(np.abs(legacy.loc[common, "equity"] - new.loc[common, "equity"]).max())
    diff_ret = float(
        np.abs(legacy.loc[common, "ret_strategy"] - new.loc[common, "ret_strategy"]).max()
    )
    assert diff_eq < 1e-10, f"Engine kwarg breaks bit-exactness: max |Δequity| = {diff_eq:.3e}"
    assert diff_ret < 1e-12, f"Engine kwarg breaks bit-exactness: max |Δret| = {diff_ret:.3e}"


def test_run_stacked_strategy_bitexact_with_capitulation_enabled() -> None:
    """Same bit-exactness gate but with capitulation overlay ON — the
    overlay shares state with the main loop, so we test it doesn't
    accidentally couple to the synthetic-3x path."""
    from research.samir_stack.indicators import build_indicator_panel
    from research.samir_stack.regime_score import regime_score_equal

    spy = _make_underlying(n=1200, seed=42)
    ief = _make_bond("IEF", n=1200, mu=0.03, sigma=0.04, seed=11)
    vix = _make_bond("VIX", n=1200, mu=0.0, sigma=0.20, seed=17).clip(lower=8.0)
    hyg = _make_bond("HYG", n=1200, mu=0.05, sigma=0.08, seed=23)
    panel = build_indicator_panel(spy, vix_close=vix, hyg_close=hyg, ief_close=ief)
    score = regime_score_equal(panel)

    cap_cfg = CapitulationConfig(enabled=True)
    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=3.0,
        tier_thresholds=(0.30, 0.50, 0.75),
        capitulation=cap_cfg,
    )

    legacy = run_stacked_strategy(spy, ief, score, cfg, indicator_panel=panel)
    new = run_stacked_strategy(
        spy,
        ief,
        score,
        cfg,
        indicator_panel=panel,
        equity_engine=SyntheticETFEngine(ter_annual=cfg.leverage_ter_annual),
        bond_sleeve=StaticBondSleeve(name="IEF", close=ief),
    )

    common = legacy.index.intersection(new.index)
    diff_eq = float(np.abs(legacy.loc[common, "equity"] - new.loc[common, "equity"]).max())
    assert diff_eq < 1e-10, (
        f"Engine kwarg breaks bit-exactness under capitulation: max |Δequity| = {diff_eq:.3e}"
    )
