"""NautilusTrader API contract tests.

Pins the specific NautilusTrader behaviours the live strategies depend on,
so a future NT version bump (or Cython internals change) fails CI loudly
instead of silently breaking a live trading code path.

Trigger incident: May 11 2026 — under NautilusTrader 1.221, ``str(PositionSide.LONG)``
returns the integer-as-string ``"2"`` rather than ``"LONG"`` or
``"PositionSide.LONG"``. The ``BondGoldStrategy._run_signal`` entry guard
gated entries on ``str(position.side) == "LONG"``, which evaluated to
``"2" == "LONG"`` = always False. The strategy doubled positions on every
container restart for ~12 days before being caught.

These tests are intentionally narrow — each one pins one fact about NT
that a strategy relies on. A failure means: (a) NT changed behaviour and
strategy code may now be wrong, or (b) the strategy code was always
wrong and we just learned the contract. Either way, manual review is
required before bumping NT.

See ``directives/Operational Robustness Framework 2026-05-12.md`` for
the broader framework this fits into (Tier 1, item 1.5).
"""

from __future__ import annotations

import pytest

# Each import fenced separately — if NT renames or removes a symbol, the
# specific enum we depend on is identified.
from nautilus_trader.model.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)

# ── PositionSide ───────────────────────────────────────────────────────────


def test_position_side_enum_members_present():
    """The four PositionSide values strategies depend on must exist."""
    assert PositionSide.LONG is not None
    assert PositionSide.SHORT is not None
    assert PositionSide.FLAT is not None
    assert PositionSide.NO_POSITION_SIDE is not None


def test_position_side_name_is_bare_member_name():
    """``side.name`` is the canonical way to get the human-readable label.

    Strategies that need a label string (logs, notifications) MUST use
    ``side.name`` — never ``str(side)`` (which returns the integer in
    NT 1.221+) and never ``str(side).split(".")[-1]`` (which returns
    the integer too because the int has no dot).
    """
    assert PositionSide.LONG.name == "LONG"
    assert PositionSide.SHORT.name == "SHORT"
    assert PositionSide.FLAT.name == "FLAT"


def test_position_side_equality_uses_enum_object_not_string():
    """Direct enum equality is the canonical comparison pattern.

    ``position.side == PositionSide.LONG`` is correct.
    ``str(position.side) == "LONG"`` is the May 11 antipattern — it
    silently evaluates False in NT 1.221 because str(enum) returns the
    integer-as-string.
    """
    assert PositionSide.LONG == PositionSide.LONG
    assert PositionSide.LONG != PositionSide.SHORT
    # Critical: the antipattern that caused the May 11 incident.
    # Pin it explicitly so a future NT change that "fixes" str() can't
    # silently mask a regression.
    assert PositionSide.LONG != "LONG", (
        "PositionSide enum compares-False against the string 'LONG'. "
        "If this assertion ever fails (NT may have added __eq__ override) "
        "audit the codebase for any code that intentionally relies on "
        "the False-comparison behaviour. See May 11 2026 rehydration bug."
    )


# ── OrderSide ──────────────────────────────────────────────────────────────


def test_order_side_enum_members_present():
    assert OrderSide.BUY is not None
    assert OrderSide.SELL is not None


def test_order_side_name_is_bare_member_name():
    """``side.name`` is the canonical label getter for OrderSide too."""
    assert OrderSide.BUY.name == "BUY"
    assert OrderSide.SELL.name == "SELL"


# ── OrderType ──────────────────────────────────────────────────────────────


def test_order_type_enum_members_present():
    """The order types live strategies submit must remain available."""
    assert OrderType.MARKET is not None
    assert OrderType.LIMIT is not None
    assert OrderType.STOP_MARKET is not None
    assert OrderType.STOP_LIMIT is not None


def test_order_type_name_is_bare_member_name():
    assert OrderType.MARKET.name == "MARKET"
    assert OrderType.LIMIT.name == "LIMIT"


# ── TimeInForce ───────────────────────────────────────────────────────────


def test_time_in_force_enum_members_present():
    """Strategies and bracket orders rely on these TIF constants."""
    assert TimeInForce.DAY is not None
    assert TimeInForce.GTC is not None
    assert TimeInForce.IOC is not None
    assert TimeInForce.FOK is not None


# ── OrderStatus ───────────────────────────────────────────────────────────


def test_order_status_terminal_states_present():
    """``on_order_*`` callbacks fire on these statuses; strategies'
    notification + reconciliation logic dispatches on them."""
    for member in ("INITIALIZED", "SUBMITTED", "ACCEPTED", "REJECTED", "FILLED", "CANCELED"):
        assert hasattr(OrderStatus, member), f"OrderStatus.{member} no longer exists in NT"


# ── Position object (instance-level contracts via skipif harness) ──────────
#
# We can't easily instantiate a NautilusTrader Position without a full
# TradingNode. The contracts that matter for strategy code are exposed as
# attributes / methods on Position; the AST guard in
# ``tests/test_position_rehydration.py`` already locks in that strategies
# use ``signed_qty`` instead of ``str(side)``. Adding an instance-level
# property test here would require a fixture builder that's out of scope.
#
# If a future NT version changes the Position interface, the most likely
# breakage is that ``signed_qty`` becomes a different name. Catch that
# at import-time with the symbol-presence test below.


def test_position_class_exposes_required_attributes():
    """The Position class must expose the attributes strategy code reads.

    These are the attributes touched by the rehydration helpers and the
    entry/exit guards in bond_gold and mr_audjpy. If NT renames any of
    them, this test fails immediately at CI rather than at runtime.
    """
    from nautilus_trader.model.position import Position

    required = {
        "side",  # PositionSide enum — for enum-compare guards
        "signed_qty",  # numeric direction — preferred over str(side)
        "is_open",  # bool — used to filter open positions in cache
        "is_closed",  # bool — used in rehydration filter
        "instrument_id",  # InstrumentId — for cache lookup keys
        "strategy_id",  # StrategyId or "EXTERNAL" — rehydration tag
    }
    missing = [attr for attr in required if not hasattr(Position, attr)]
    if missing:
        pytest.fail(
            f"NautilusTrader Position class is missing attributes that live "
            f"strategy code depends on: {sorted(missing)}. A version bump may "
            f"have renamed/removed these. Audit strategy state-management code "
            f"before continuing."
        )
