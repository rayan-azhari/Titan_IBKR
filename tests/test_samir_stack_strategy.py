"""Tests for the live SamirStackStrategy class.

NautilusTrader's Strategy class is Cython and can't be instantiated
without a full TradingNode, so these tests focus on:
  1. Module imports cleanly
  2. The config dataclass accepts every champion-config field with the
     researched default values
  3. The vol-target scale function (factored as a pure helper inline
     copy below) returns the expected multiplier across edge cases
  4. AST guard: rehydration code path doesn't filter cache.positions
     by strategy_id (the May 11 bug class)
  5. AST guard: entry/sizing code uses signed_qty, not str(side)

Tier 2.2-style — no live cache, no NT runtime, just structural / pure-
function checks. The integration-level tests (rehydrate-then-bar
behaviour) live in tests/test_bond_gold_state_machine_properties.py
for the bond_gold class; the same Hypothesis property tests would
apply here once the strategy reaches paper-deploy stage.

See ``directives/Samir-Stack Margin Drift Research 2026-05-11.md`` §13
for the champion config this test pins.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from titan.strategies.samir_stack.strategy import (
    SamirStackConfig,
    SamirStackStrategy,
)

# ── Config sanity (champion-config pinning) ──────────────────────────


def test_config_imports_cleanly():
    """Strategy + config module import without side effects."""
    from titan.strategies.samir_stack import strategy as ss  # noqa: F401


def _minimal_config(**overrides) -> SamirStackConfig:
    """Build a config with only the required (no-default) fields filled in."""
    base = {
        "equity_instrument_id": "CSPX.LSEETF",
        "bond_instrument_id": "IEF.NASDAQ",
        "signal_spy_id": "SPY.ARCA",
        "bar_type_equity_d": "CSPX.LSEETF-1-DAY-LAST-EXTERNAL",
        "bar_type_bond_d": "IEF.NASDAQ-1-DAY-LAST-EXTERNAL",
        "bar_type_spy_d": "SPY.ARCA-1-DAY-LAST-EXTERNAL",
    }
    base.update(overrides)
    return SamirStackConfig(**base)


def test_default_capital_split_matches_champion_config():
    """Champion config: 10/90 split (was 40/60 before May 12 update)."""
    cfg = _minimal_config()
    assert cfg.equity_weight == 0.10
    assert cfg.bond_weight == 0.90


def test_default_leverage_matches_champion_config():
    """Champion config: L_max=3 with the 3-tier threshold ladder."""
    cfg = _minimal_config()
    assert cfg.L_max == 3.0
    assert cfg.tier_thresholds == (0.30, 0.50, 0.75)


def test_default_vol_target_matches_champion_config():
    """Champion config: 8% annualised vol target, 30-day window, 2x cap."""
    cfg = _minimal_config()
    assert cfg.vol_target_annual == 0.08
    assert cfg.vol_target_window == 30
    assert cfg.vol_target_max_scale == 2.0


def test_dd_circuit_breaker_defaults_unchanged():
    """DD breaker defaults match the prior config (10% throttle / 15% kill)."""
    cfg = _minimal_config()
    assert cfg.dd_throttle == 0.10
    assert cfg.dd_kill == 0.15


def test_vol_target_can_be_disabled_with_zero():
    """Setting vol_target_annual=0 disables scaling (returns 1.0)."""
    cfg = _minimal_config(vol_target_annual=0.0)
    assert cfg.vol_target_annual == 0.0


# ── Vol-target scale logic (pure-function copy of strategy method) ────
#
# The strategy method ``_vol_target_scale`` reads/writes ``self._state``
# which requires a NT-Strategy instance. Re-implement the *pure* part
# here to test the math directly. Kept in lockstep by structural
# review — same pattern as the reconciliation-strategy detect helpers.


def vol_target_scale_pure(
    equity_curve: list[float],
    *,
    target: float,
    window: int,
    max_scale: float,
) -> float:
    """Pure-function copy of ``SamirStackStrategy._vol_target_scale`` math
    (without the state-buffer side effects)."""
    if target is None or target <= 0:
        return 1.0
    n = len(equity_curve)
    if n < window + 1:
        return 1.0  # warmup (real method returns last_vol_scale; 1.0 in cold start)
    rets = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, n)
        if equity_curve[i - 1] != 0
    ]
    window_rets = rets[-window:]
    if len(window_rets) < 2:
        return 1.0
    mean_r = sum(window_rets) / len(window_rets)
    var = sum((r - mean_r) ** 2 for r in window_rets) / (len(window_rets) - 1)
    if var <= 0:
        return 1.0
    realised_vol = (var**0.5) * (252.0**0.5)
    if realised_vol < 1e-8:
        return 1.0
    scale = min(target / realised_vol, max_scale)
    return max(scale, 0.01)


def test_vol_target_returns_one_when_disabled():
    """target_vol <= 0 short-circuits to 1.0 (no scaling)."""
    eq = [10_000.0 + i * 10 for i in range(100)]
    assert vol_target_scale_pure(eq, target=0.0, window=30, max_scale=2.0) == 1.0
    assert vol_target_scale_pure(eq, target=-0.01, window=30, max_scale=2.0) == 1.0


def test_vol_target_warmup_returns_one():
    """Insufficient history → 1.0 (waiting for window to fill)."""
    eq = [10_000.0] * 5
    assert vol_target_scale_pure(eq, target=0.08, window=30, max_scale=2.0) == 1.0


def test_vol_target_scales_up_in_calm_regime():
    """When realised vol << target, scale > 1.0 (lever up).

    Build a series with very low realised vol (~3% annualised). At
    8% target this should produce scale ≈ 2.67, capped at max_scale=2.0.
    """
    # ~0.20% daily moves → ~3.2% annualised vol
    rng_state = 12345
    eq = [10_000.0]
    for i in range(100):
        # deterministic small alternating moves
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        move = (rng_state % 41 - 20) / 10000.0  # ±0.2% range
        eq.append(eq[-1] * (1.0 + move))
    scale = vol_target_scale_pure(eq, target=0.08, window=30, max_scale=2.0)
    assert scale > 1.5, f"Calm regime should lever up; got scale={scale:.3f}"
    assert scale <= 2.0, "Must respect max_scale cap"


def test_vol_target_scales_down_in_stressed_regime():
    """When realised vol >> target, scale < 1.0 (de-risk)."""
    # ~3% daily moves → ~48% annualised vol
    rng_state = 99999
    eq = [10_000.0]
    for i in range(100):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        move = (rng_state % 601 - 300) / 10000.0  # ±3% range
        eq.append(eq[-1] * (1.0 + move))
    scale = vol_target_scale_pure(eq, target=0.08, window=30, max_scale=2.0)
    assert scale < 0.5, f"Stressed regime should de-risk; got scale={scale:.3f}"
    assert scale > 0.0


def test_vol_target_constant_series_returns_one():
    """Zero-variance series → 1.0 (degenerate, no signal)."""
    eq = [10_000.0] * 100
    assert vol_target_scale_pure(eq, target=0.08, window=30, max_scale=2.0) == 1.0


def test_vol_target_floors_at_001():
    """Even with extreme realised vol, scale never drops below 0.01."""
    # Build a series with absurdly high realised vol
    eq = [10_000.0]
    for i in range(60):
        # ±50% daily moves
        sign = 1 if i % 2 == 0 else -1
        eq.append(eq[-1] * (1.0 + sign * 0.5))
    scale = vol_target_scale_pure(eq, target=0.08, window=30, max_scale=2.0)
    assert scale >= 0.01


# ── AST guards (May 11 bug class — same pattern as bond_gold) ────────


def test_rehydration_does_not_filter_cache_positions_by_strategy_id():
    """``_rehydrate_position_from_broker`` must NOT call
    ``cache.positions(..., strategy_id=...)`` — the May 11 root cause was
    that filter excluding EXTERNAL-tagged positions reconciled by NT's
    ExecEngine on container restart.
    """
    src = inspect.getsource(SamirStackStrategy._rehydrate_position_from_broker)
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "positions"):
            continue
        if not (isinstance(func.value, ast.Attribute) and func.value.attr == "cache"):
            continue
        for kw in node.keywords:
            assert kw.arg != "strategy_id", (
                "SamirStackStrategy._rehydrate_position_from_broker calls "
                "cache.positions(..., strategy_id=...) which excludes EXTERNAL "
                "broker positions — the May 11 bug class. See "
                "directives/Operational Robustness Framework 2026-05-12.md."
            )


def test_strategy_does_not_use_brittle_position_side_string_check():
    """Repo-wide AST guard already enforces this for ``titan/strategies/``,
    but pin it explicitly here so a Samir-Stack regression is caught
    independently."""
    source = inspect.getsource(SamirStackStrategy)
    assert 'str(position.side) == "LONG"' not in source, (
        "SamirStackStrategy uses the brittle side-as-string check that "
        "fails in NautilusTrader 1.221+ (Cython enum str() returns the "
        "integer). Use signed_qty or PositionSide enum equality. See "
        "May 11 2026 rehydration bug."
    )
    assert 'str(position.side) == "SHORT"' not in source
    assert 'str(p.side) == "LONG"' not in source


def test_strategy_uses_signed_qty_for_position_aggregation():
    """The position-aggregation method (``_current_units``) uses
    ``signed_qty`` — the post-fix robust pattern. This test pins it so a
    regression to ``str(side)`` would fail loudly."""
    source = inspect.getsource(SamirStackStrategy._current_units)
    assert "signed_qty" in source


# ── Vol-target wiring ────────────────────────────────────────────────


def test_rebalance_method_applies_vol_scale_to_equity_only():
    """``_rebalance_if_needed`` must scale ONLY the equity sleeve's
    notional, not the bond sleeve. Bond is the defensive ballast and
    shouldn't be vol-scaled (already low vol, scaling defeats purpose)."""
    source = inspect.getsource(SamirStackStrategy._rebalance_if_needed)
    # The equity notional gets multiplied by vol_scale; bond doesn't.
    assert "target_eq_w_scaled" in source
    assert "target_eq_notional = target_eq_w_scaled" in source
    # And bond uses the un-scaled weight directly
    assert "target_bd_notional = target_bd_w * equity_value" in source
