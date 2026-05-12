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

    The bug: NautilusTrader 1.221's PositionSide Cython enum's ``__str__``
    returns the integer value (e.g. ``"2"`` for LONG), not the bare label
    or qualified path. The brittle string comparison silently evaluated to
    False for ALL positions, and bond_gold layered a fresh BUY on top of
    the rehydrated inventory. Use ``signed_qty`` (numeric) or the enum
    value directly instead.
    """
    source = inspect.getsource(BondGoldStrategy)
    assert 'str(position.side) == "LONG"' not in source, (
        "BondGoldStrategy uses the brittle string comparison "
        '`str(position.side) == "LONG"` which fails for ALL positions in '
        "NautilusTrader 1.221 (Cython enum str() returns the integer). "
        "Use signed_qty or the enum value directly. See "
        "directives/Rehydration Bug 2026-05-11.md."
    )


def test_no_strategy_uses_brittle_position_side_string_check():
    """Repo-wide guard: NO live strategy may use ``str(position.side) == "..."``.

    The May 11 audit found 7 additional strategies (fx_carry, gld_confluence,
    ic_equity_daily, ic_generic, turtle, mtf, ic_mtf) using the same brittle
    pattern as the bond_gold rehydration bug. Even though most are not
    currently deployed, the bug class must be locked out repo-wide so
    future deployments don't reactivate it.

    Robust replacements:
      - For a long/short branch:           ``position.signed_qty > 0``
      - For an enum equality check:        ``position.side == PositionSide.LONG``
      - For a human-readable label:        ``position.side.name``  (e.g. "LONG")
    """
    import re
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    strategies_dir = project_root / "titan" / "strategies"
    # Match str(<name>.side|status|type) == "..." OR != "..."
    pattern = re.compile(
        r"\bstr\s*\(\s*\w+\s*\.\s*(side|status|order_status|order_type|order_side|tif)\s*\)"
        r'\s*[!=]=\s*["\']'
    )
    offenders: list[str] = []
    for py in strategies_dir.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = str(py.relative_to(project_root)).replace("\\", "/")
                offenders.append(f"  {rel}:{i}: {line.strip()}")
    assert not offenders, (
        'Brittle `str(<obj>.side|status|type) == "..."` comparison detected. '
        "Under NautilusTrader 1.221 the Cython-enum __str__ returns the integer "
        "value, so this comparison is always False (see May 11 2026 incident). "
        "Use `obj.signed_qty > 0`, `obj.side == PositionSide.LONG`, or "
        '`obj.side.name == "LONG"` instead. Offenders:\n' + "\n".join(offenders)
    )
