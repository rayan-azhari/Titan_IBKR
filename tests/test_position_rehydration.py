"""Regression test for the position-rehydration fix (May 2026).

The bug: ``BondGoldStrategy`` and ``MRAUDJPYStrategy`` filtered cache
positions by ``strategy_id=self.id`` during startup rehydration. Broker
positions reconciled by NautilusTrader's ExecEngine come back tagged with
``position_id=*-EXTERNAL`` (no strategy_id, because the prior session's
strategy_id was lost on shutdown). The filter excluded those, leading to
phantom-position layering on every container restart.

AST guard: ``_rehydrate_position_from_broker`` MUST call
``cache.positions`` without a ``strategy_id`` kwarg, so the rehydration
path is open to EXTERNAL-tagged positions.

(Behavioural mocking of NautilusTrader Strategy state is not practical:
``id``, ``cache``, and ``log`` are all read-only Cython properties on the
parent ``Actor`` / ``Component`` classes, so the strategy can't be
instantiated without a full TradingNode. The AST guard alone is the same
kind of structural invariant the project already uses in
``tests/test_research_math_guardrails.py``.)

See ``directives/Rehydration Bug 2026-05-11.md`` for the full post-mortem.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from titan.strategies.bond_gold.strategy import BondGoldStrategy
from titan.strategies.mr_audjpy.strategy import MRAUDJPYStrategy


def _calls_cache_positions_without_strategy_id_filter(source: str) -> bool:
    """Return True iff every ``cache.positions(...)`` call inside the source
    omits the ``strategy_id`` kwarg. The fix requires the rehydration path
    to NOT filter by strategy_id so EXTERNAL-tagged broker positions are
    visible to it.
    """
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "positions"):
            continue
        if not (isinstance(func.value, ast.Attribute) and func.value.attr == "cache"):
            continue
        for kw in node.keywords:
            if kw.arg == "strategy_id":
                return False
    return True


@pytest.mark.parametrize(
    "strategy_cls",
    [BondGoldStrategy, MRAUDJPYStrategy],
    ids=["bond_gold", "mr_audjpy"],
)
def test_rehydration_does_not_filter_by_strategy_id(strategy_cls):
    """Locks in the May 2026 fix: rehydration must see EXTERNAL positions."""
    method_source = inspect.getsource(strategy_cls._rehydrate_position_from_broker)
    assert _calls_cache_positions_without_strategy_id_filter(method_source), (
        f"{strategy_cls.__name__}._rehydrate_position_from_broker calls "
        f"cache.positions(..., strategy_id=...) which excludes EXTERNAL "
        f"broker positions and re-introduces the May 2026 phantom-position "
        f"bug. See directives/Rehydration Bug 2026-05-11.md."
    )


def test_bond_gold_does_not_use_brittle_position_side_string_check():
    """Locks in the May 12 2026 fix: bond_gold must not gate entries on
    ``str(position.side) == "LONG"``.

    The bug: NautilusTrader 1.221's PositionSide Cython enum renders as
    ``"PositionSide.LONG"`` rather than ``"LONG"`` for EXTERNAL-rehydrated
    positions, so the brittle string comparison silently evaluated to False
    and bond_gold layered a fresh BUY on top of the rehydrated inventory.
    Use ``signed_qty`` (numeric) or the enum value directly instead.
    """
    source = inspect.getsource(BondGoldStrategy)
    assert 'str(position.side) == "LONG"' not in source, (
        "BondGoldStrategy uses the brittle string comparison "
        '`str(position.side) == "LONG"` which fails for EXTERNAL-rehydrated '
        "positions in NautilusTrader 1.221 (Cython enum str() returns "
        "'PositionSide.LONG' not 'LONG'). Use signed_qty or the enum value "
        "directly. See directives/Rehydration Bug 2026-05-11.md."
    )
