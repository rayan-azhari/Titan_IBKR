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
    """Build a config with only the required (no-default) fields filled in.

    Uses MES futures-style symbols by default to match the current
    champion config (MES + IGLT), but any test can override via kwargs.
    """
    base = {
        "equity_instrument_id": "MESM26.CME",
        "bond_instrument_id": "IGLT.LSEETF",
        "signal_spy_id": "SPY.ARCA",
        "bar_type_equity_d": "MESM26.CME-1-DAY-LAST-EXTERNAL",
        "bar_type_bond_d": "IGLT.LSEETF-1-DAY-LAST-EXTERNAL",
        "bar_type_spy_d": "SPY.ARCA-1-DAY-LAST-EXTERNAL",
    }
    base.update(overrides)
    return SamirStackConfig(**base)


def test_default_capital_split_matches_champion_config():
    """Phase 5 GBP-clean champion (2026-05-13): 40/60 split.

    Replaces the 10/90 from the pre-audit champion (which the audit
    found delivered honest Sharpe 0.64, not the claimed 2.28, due to
    look-ahead and cost-model bugs).
    """
    cfg = _minimal_config()
    assert cfg.equity_weight == 0.40
    assert cfg.bond_weight == 0.60


def test_default_leverage_matches_champion_config():
    """Phase 5 GBP-clean champion: L_max=2 with a 2-tier threshold ladder.

    Phase 5 found L=2 dominates L=3 on Calmar CI lo because vol drag
    at L=3 (~14%/yr in the synthetic ETF engine at L=3 vol) erodes
    risk-adjusted returns more than the marginal upside from extra
    leverage.
    """
    cfg = _minimal_config()
    assert cfg.L_max == 2.0
    assert cfg.tier_thresholds == (0.30, 0.55)


def test_default_vol_target_disabled_in_champion_config():
    """Phase 5 GBP-clean champion: vol-targeting DISABLED.

    Phase 5 found the regime gate captures the same risk-management
    benefit without vol-targeting's operational complexity. The
    pre-Phase-5 default of 0.08 was based on a buggy backtest (audit
    2026-05-12); honest backtests show vol-target adds only ~+0.07
    Sharpe at significant operational cost.
    """
    cfg = _minimal_config()
    assert cfg.vol_target_annual == 0.0
    assert cfg.vol_target_window == 30
    assert cfg.vol_target_max_scale == 2.0


def test_dd_circuit_breaker_defaults_unchanged():
    """DD breaker defaults are stable across audit/remediation: 10%/15%."""
    cfg = _minimal_config()
    assert cfg.dd_throttle == 0.10
    assert cfg.dd_kill == 0.15


def test_vol_target_can_be_opted_in_with_positive_value():
    """Operator can opt back into vol-targeting by setting > 0.

    This is supported for experimentation only — the default-off behaviour
    matches the Phase 5 finding. Operators choosing to opt in own the
    operational complexity (vol scaling on a leveraged ETF position).
    """
    cfg = _minimal_config(vol_target_annual=0.06)
    assert cfg.vol_target_annual == 0.06


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


# ── Tier-sizing wiring (Phase 6a.5 fix) ──────────────────────────────


def test_default_equity_native_leverage_matches_champion_config():
    """Current champion is MES futures + IGLT (no leveraged ETF).
    ``equity_native_leverage`` must be 1.0 — the leverage comes from
    the number of futures contracts held, not from any built-in
    compounding of the instrument.
    """
    cfg = _minimal_config()
    assert cfg.equity_native_leverage == 1.0


def test_default_equity_is_future_matches_champion_config():
    """Current champion uses MES futures, NOT a leveraged ETF.

    The Phase 6 deployment switched from 3USL.LSEETF (3× daily-rebalanced
    UCITS) to MES futures to avoid the L²-compounded vol drag of
    daily-reset products. This default must reflect that choice.
    """
    cfg = _minimal_config()
    assert cfg.equity_is_future is True
    assert cfg.futures_multiplier == 5.0  # MES multiplier ($5/point)


def test_on_bar_scales_target_eq_w_by_tier_over_native_leverage():
    """AST guard: the on_bar method MUST scale target_eq_w by
    ``target_tier / equity_native_leverage``. A regression where the
    sizing is just ``equity_weight`` (no tier scaling) re-introduces
    the pre-audit always-max-tier behaviour and breaks the live-vs-
    research match documented in Strategy Guide §3.1.
    """
    source = inspect.getsource(SamirStackStrategy.on_bar)
    # The exact scaling expression must appear.
    assert "equity_native_leverage" in source, (
        "on_bar() must reference equity_native_leverage for tier-based "
        "sizing. Pre-audit code held a constant equity_weight regardless "
        "of tier; deploying that with a 3× ETF over-leveraged the live "
        "position by 1.5-3× vs the backtest model."
    )
    assert "tier_ratio" in source and "target_tier" in source, (
        "on_bar() must compute the tier_ratio = target_tier / "
        "equity_native_leverage and apply it as a multiplier on "
        "self.config.equity_weight."
    )


def test_tier_scaled_target_notional_at_mes_futures_champion_config():
    """Pure-function pin for the MES futures champion config.

    With equity_native_leverage=1.0 the formula
    ``equity_weight × (tier / native_leverage)`` simplifies to
    ``equity_weight × tier`` — exactly the levered notional target
    that the futures-sizing path floors into integer contracts.
    """
    cfg = _minimal_config()  # MES futures defaults
    assert cfg.equity_native_leverage == 1.0  # precondition
    # tier=1: 40% × 1 = 40% of NAV in target notional → 40% SPX
    expected_t1 = cfg.equity_weight * (1.0 / cfg.equity_native_leverage)
    assert abs(expected_t1 - 0.40) < 1e-12
    # tier=2: 40% × 2 = 80% of NAV in target notional → 80% SPX
    expected_t2 = cfg.equity_weight * (2.0 / cfg.equity_native_leverage)
    assert abs(expected_t2 - 0.80) < 1e-12


def test_tier_scaling_with_3x_etf_instrument_holds_less():
    """Inverse case: if the operator switches BACK to a 3× ETF (e.g.
    3USL.LSEETF) by setting ``equity_native_leverage=3.0``, the
    strategy must hold LESS of the instrument at each tier because
    each unit carries native 3× leverage.

    Phase 6 default is futures (native_leverage=1.0); this test pins
    the alternative path so a future operator's ETF switch behaves
    correctly.
    """
    cfg = _minimal_config(equity_native_leverage=3.0)
    # tier=1: 40% × 1/3 ≈ 13.3% in 3USL → 40% SPX
    expected_t1 = cfg.equity_weight * (1.0 / 3.0)
    assert abs(expected_t1 - 0.4 / 3.0) < 1e-12
    # tier=2: 40% × 2/3 ≈ 26.7% in 3USL → 80% SPX
    expected_t2 = cfg.equity_weight * (2.0 / 3.0)
    assert abs(expected_t2 - 0.8 / 3.0) < 1e-12


def test_end_to_end_futures_sizing_tier_to_contracts():
    """End-to-end check: at the champion config (ew=0.40, L_max=2,
    native_lev=1, equity_is_future=True), the tier-scaled target
    notional should produce sensible integer-contract counts at the
    deployment NAV.

    Pure-function test using ``futures_target_contracts_pure`` — verifies
    the full pipeline:
      tier → target_eq_w_scaled = ew × (tier / native_lev)
      → target_notional = target_eq_w_scaled × NAV (GBP)
      → contracts = floor(notional_usd / (multiplier × spx_price))

    At £30k NAV (operational floor), GBPUSD=1.33 (fx=0.7519), MES
    multiplier $5, SPX ~5500 → contract notional ~$27.5k. Expected:
      tier 0: 0 contracts
      tier 1: floor(£12k / 0.7519 = $16k / $27.5k) = 0 contracts
              (chunky — the strategy is operationally constrained at
              tier 1 at £30k NAV; this is documented in Strategy
              Guide §3.1)
      tier 2: floor(£24k / 0.7519 = $32k / $27.5k) = 1 contract
    """
    cfg = _minimal_config()
    nav_gbp = 30_000.0
    fx_quote_to_base = 0.7519  # 1 USD = 0.7519 GBP
    multiplier = cfg.futures_multiplier
    spx_price = 5500.0

    for tier, expected_contracts in [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]:
        target_eq_w_scaled = cfg.equity_weight * (tier / cfg.equity_native_leverage)
        target_notional_gbp = target_eq_w_scaled * nav_gbp
        contracts = futures_target_contracts_pure(
            target_notional_gbp,
            spx_price,
            multiplier=multiplier,
            fx_rate=fx_quote_to_base,
            quote_ccy="USD",
            base_ccy="GBP",
        )
        assert contracts == expected_contracts, (
            f"tier={tier}: expected {expected_contracts} contracts, got {contracts}. "
            f"target_eq_w_scaled={target_eq_w_scaled}, target_notional_gbp={target_notional_gbp}"
        )


def test_end_to_end_futures_sizing_at_larger_nav_resolves_chunkiness():
    """At £100k NAV (well above the £30k operational floor), tier 1
    should produce ≥1 contract and tier 2 should produce ≥2 contracts —
    chunkiness becomes ignorable.
    """
    cfg = _minimal_config()
    nav_gbp = 100_000.0
    fx_quote_to_base = 0.7519
    multiplier = cfg.futures_multiplier
    spx_price = 5500.0

    # tier 1: 0.40 × £100k = £40k = $53.2k → floor($53.2k / $27.5k) = 1
    target_eq_w_t1 = cfg.equity_weight * (1.0 / cfg.equity_native_leverage)
    contracts_t1 = futures_target_contracts_pure(
        target_eq_w_t1 * nav_gbp,
        spx_price,
        multiplier=multiplier,
        fx_rate=fx_quote_to_base,
        quote_ccy="USD",
        base_ccy="GBP",
    )
    assert contracts_t1 >= 1, f"tier 1 at £100k should fill ≥1 contract; got {contracts_t1}"

    # tier 2: 0.80 × £100k = £80k = $106.4k → floor($106.4k / $27.5k) = 3
    target_eq_w_t2 = cfg.equity_weight * (2.0 / cfg.equity_native_leverage)
    contracts_t2 = futures_target_contracts_pure(
        target_eq_w_t2 * nav_gbp,
        spx_price,
        multiplier=multiplier,
        fx_rate=fx_quote_to_base,
        quote_ccy="USD",
        base_ccy="GBP",
    )
    assert contracts_t2 >= 2, f"tier 2 at £100k should fill ≥2 contracts; got {contracts_t2}"


def test_max_target_notional_at_mes_futures_champion_is_well_below_natural_margin_cap():
    """With MES futures at the champion config (equity_weight=0.40,
    L_max=2), the max target notional is 80% of NAV — well below the
    natural ~16:1 cash collateral cap of MES (6% IM → ~16× notional
    per $1 cash). The strategy operates comfortably without margin
    borrowing on top of futures.
    """
    cfg = _minimal_config()
    max_target_notional_pct = cfg.equity_weight * cfg.L_max  # native_leverage=1
    assert max_target_notional_pct == 0.80
    # IM required at L_max for 6% IM: 0.80 × 0.06 = 4.8% of NAV.
    im_pct = max_target_notional_pct * 0.06
    assert im_pct < 0.20, "IM should sit at < 20% of NAV; lots of headroom"


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


# ── Phase 2: MES futures execution ───────────────────────────────────


def test_futures_enabled_by_default_post_phase6():
    """``equity_is_future`` defaults to True in the post-Phase-6 config.

    The strategy switched from 3USL.LSEETF (Phase 6a) to MES futures
    (Phase 6 post-fix) to avoid daily-rebalanced leveraged-ETF vol
    drag. Operators wanting to switch back to an ETF must explicitly
    set ``equity_is_future=False`` and update equity_native_leverage.
    """
    cfg = _minimal_config()
    assert cfg.equity_is_future is True
    assert cfg.futures_multiplier == 5.0  # MES default


def test_futures_config_accepts_es_multiplier_opt_in():
    """For larger NAVs the operator may switch to ES futures
    (50× multiplier vs MES's 5×)."""
    cfg = _minimal_config(futures_multiplier=50.0)
    assert cfg.equity_is_future is True  # still futures
    assert cfg.futures_multiplier == 50.0  # ES (full-size E-mini)


def test_etf_path_remains_supported_via_opt_out():
    """Operators wanting the ETF execution path (e.g. for small NAV
    where futures chunkiness is operationally painful) can set
    ``equity_is_future=False`` and update equity_native_leverage."""
    cfg = _minimal_config(
        equity_is_future=False,
        equity_native_leverage=3.0,
        equity_instrument_id="3USL.LSEETF",
        bar_type_equity_d="3USL.LSEETF-1-DAY-LAST-EXTERNAL",
    )
    assert cfg.equity_is_future is False
    assert cfg.equity_native_leverage == 3.0


def futures_target_contracts_pure(
    notional_base: float,
    price: float | None,
    *,
    multiplier: float = 5.0,
    fx_rate: float = 1.0,
    quote_ccy: str = "USD",
    base_ccy: str = "USD",
) -> float | None:
    """Pure copy of ``SamirStackStrategy._futures_target_contracts``
    sizing math (without the ``self.config.*`` indirection). Kept in
    lockstep with strategy by structural review."""
    if price is None or price <= 0:
        return None
    if notional_base <= 0:
        return 0.0
    if quote_ccy != base_ccy:
        if fx_rate <= 0:
            return None
        notional_quote = notional_base / fx_rate
    else:
        notional_quote = notional_base
    contract_notional = multiplier * price
    if contract_notional <= 0:
        return None
    from math import floor

    return float(floor(notional_quote / contract_notional))


def test_futures_sizing_floors_to_integer_contracts():
    """Sizing must always round DOWN — under-target by < 1 contract is
    preferable to over-leverage by rounding up."""
    # $30k notional / ($5 × $5800 = $29k per contract) = 1.034 → 1 contract
    assert futures_target_contracts_pure(30_000.0, 5800.0, multiplier=5.0) == 1.0
    # $58k → exactly 2 contracts (no rounding needed)
    assert futures_target_contracts_pure(58_000.0, 5800.0, multiplier=5.0) == 2.0
    # $57.999k → still 1 contract (below 2-contract threshold)
    assert futures_target_contracts_pure(57_999.0, 5800.0, multiplier=5.0) == 1.0


def test_futures_sizing_returns_zero_for_subcontract_notional():
    """If notional < cost of one contract, target = 0 (skip the trade)."""
    # $10k notional vs $29k per contract → 0
    assert futures_target_contracts_pure(10_000.0, 5800.0, multiplier=5.0) == 0.0


def test_futures_sizing_handles_missing_price():
    assert futures_target_contracts_pure(30_000.0, None) is None
    assert futures_target_contracts_pure(30_000.0, 0.0) is None
    assert futures_target_contracts_pure(30_000.0, -100.0) is None


def test_futures_sizing_handles_zero_notional():
    """Zero target notional → 0 contracts (no order)."""
    assert futures_target_contracts_pure(0.0, 5800.0) == 0.0
    assert futures_target_contracts_pure(-100.0, 5800.0) == 0.0


def test_futures_sizing_converts_base_to_quote_currency():
    """For a GBP-base account holding USD-quoted MES, sizing must
    convert base → quote first, then divide by USD contract notional."""
    # £30k base @ fx=0.75 (1 USD = 0.75 GBP) → $40k quote
    # $40k / ($5 × $5800 = $29k per contract) = 1.379 → 1 contract
    out = futures_target_contracts_pure(
        30_000.0, 5800.0, multiplier=5.0, fx_rate=0.75, quote_ccy="USD", base_ccy="GBP"
    )
    assert out == 1.0


def test_futures_sizing_rejects_invalid_fx_rate():
    """fx_rate <= 0 with mismatched currencies returns None (defensive)."""
    out = futures_target_contracts_pure(
        30_000.0, 5800.0, multiplier=5.0, fx_rate=0.0, quote_ccy="USD", base_ccy="GBP"
    )
    assert out is None


# ── Phase 3: bond rotation overlay ───────────────────────────────────


def test_phase3_rotation_disabled_by_default():
    """Rotation off by default — existing single-bond path preserved."""
    cfg = _minimal_config()
    assert cfg.bond_rotation_instruments == ()
    assert cfg.bond_rotation_bar_types == ()
    assert cfg.bond_rotation_lookback_days == 60


def test_phase3_rotation_config_accepts_paired_lists():
    cfg = _minimal_config(
        bond_rotation_instruments=("IEF.NASDAQ", "HYG.NYSE"),
        bond_rotation_bar_types=(
            "IEF.NASDAQ-1-DAY-LAST-EXTERNAL",
            "HYG.NYSE-1-DAY-LAST-EXTERNAL",
        ),
    )
    assert len(cfg.bond_rotation_instruments) == 2
    assert len(cfg.bond_rotation_bar_types) == 2


def select_bond_pure(
    bond_closes: dict[str, list[float]],
    *,
    lookback: int = 60,
) -> str | None:
    """Pure copy of ``SamirStackStrategy._select_bond_instrument`` selector
    logic (without the InstrumentId / config indirection). Returns the
    string id of the chosen bond, or None for cash."""
    from math import log

    scores: list[tuple[str, float]] = []
    for bid, closes in bond_closes.items():
        if len(closes) < lookback + 1:
            continue
        try:
            mom = log(closes[-1] / closes[-1 - lookback])
        except (ValueError, ZeroDivisionError):
            continue
        scores.append((bid, mom))
    if not scores:
        return None
    scores.sort(key=lambda kv: kv[1], reverse=True)
    best_bid, best_mom = scores[0]
    if best_mom <= 0:
        return None
    return best_bid


def test_rotation_picks_winner_among_positive_momentum():
    """When IEF +5%, HYG +2%, both positive → IEF wins."""
    closes = {
        "IEF.NASDAQ": [100.0] * 60 + [105.0],  # +5% over 60d
        "HYG.NYSE": [100.0] * 60 + [102.0],  # +2% over 60d
    }
    assert select_bond_pure(closes, lookback=60) == "IEF.NASDAQ"


def test_rotation_picks_higher_momentum_when_one_negative():
    """IEF +3%, HYG -5% → IEF wins (positive)."""
    closes = {
        "IEF.NASDAQ": [100.0] * 60 + [103.0],
        "HYG.NYSE": [100.0] * 60 + [95.0],
    }
    assert select_bond_pure(closes, lookback=60) == "IEF.NASDAQ"


def test_rotation_returns_cash_when_all_negative():
    """Both bonds in drawdown → cash (None)."""
    closes = {
        "IEF.NASDAQ": [100.0] * 60 + [95.0],
        "HYG.NYSE": [100.0] * 60 + [92.0],
    }
    assert select_bond_pure(closes, lookback=60) is None


def test_rotation_returns_cash_during_warmup():
    """Insufficient history for ANY candidate → cash."""
    closes = {
        "IEF.NASDAQ": [100.0] * 30,  # not enough for 60d lookback
        "HYG.NYSE": [100.0] * 30,
    }
    assert select_bond_pure(closes, lookback=60) is None


def test_rotation_allows_partial_warmup_with_at_least_one_candidate():
    """If IEF has enough history but HYG doesn't, we still rotate to IEF."""
    closes = {
        "IEF.NASDAQ": [100.0] * 60 + [103.0],
        "HYG.NYSE": [100.0] * 30,  # warmup
    }
    assert select_bond_pure(closes, lookback=60) == "IEF.NASDAQ"


def test_rotation_zero_momentum_treated_as_cash():
    """Strict > 0 condition: exactly-flat bond doesn't trigger entry."""
    closes = {
        "IEF.NASDAQ": [100.0] * 61,  # exactly 0% momentum
    }
    assert select_bond_pure(closes, lookback=60) is None


def test_rotation_handles_zero_or_negative_close_gracefully():
    """A degenerate price (0 or negative) doesn't crash the selector."""
    closes = {
        "IEF.NASDAQ": [100.0] + [0.0] * 60,  # division by zero in log
    }
    # Should silently skip this candidate, return None (no other candidates)
    assert select_bond_pure(closes, lookback=60) is None
