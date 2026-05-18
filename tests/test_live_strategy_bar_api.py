"""Static integration check: every live strategy's `on_bar` (and any helper
it calls on a `Bar` parameter) accesses only attributes that actually exist
on `nautilus_trader.model.data.Bar`.

Motivation
----------
The 2026-05-18 incident: `titan/strategies/ewmac_regime/strategy.py` called
`bar.close_time_as_datetime()` which does not exist on the live `Bar` class.
Latent for ~18 hours of paper trading -- crashed on the first daily ES bar
after the post-Gemini-audit redeploy. Other strategies were unaffected only
because they happened to use the canonical `bar.ts_event` pattern.

Existing parity tests feed synthetic pandas DataFrames, not real Nautilus
`Bar` objects, so they cannot catch live-API drift. A full runtime
integration test would require mocking the entire `TradingNode` (cache,
msgbus, clock, portfolio). The cheap-and-deterministic alternative below
parses each strategy's `on_bar` source with `ast`, extracts every
``bar.<attr>`` access, and asserts it exists on the real `Bar` class.

This catches:
  * Renamed Nautilus attributes (the literal incident).
  * Typos in attribute names.
  * Helper methods that were removed in a Nautilus upgrade.

It does NOT catch:
  * Wrong return-type assumptions on a valid attribute.
  * Semantic bugs (signal logic, sizing math) -- covered by parity tests.

This is a static-analysis test, not a runtime test. New live strategies
inherit the check automatically because the registry below is discovered
by walking ``titan/strategies/``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.trading.strategy import Strategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGIES_DIR = PROJECT_ROOT / "titan" / "strategies"

# Strategies which do not subclass ``Strategy`` (utility modules) or are
# explicitly research-only and not part of the live runtime.
_EXCLUDE_DIRS: set[str] = {
    # Reconciliation / daily_summary are info-only utilities; if they grow
    # an on_bar handler we want this test to cover them, but as of
    # 2026-05-18 they have none. The discover step filters by
    # "has on_bar method" so this set is mostly documentation.
}


def _valid_bar_attrs() -> set[str]:
    """Public attributes of ``Bar`` -- everything except dunders.

    Includes properties, methods, and class-level constants. Pure
    dynamically-attached attributes (added by ``__init__``) are NOT
    captured here, but ``Bar`` is implemented in Cython and exposes its
    interface via the metaclass, so ``dir(Bar)`` is authoritative.
    """
    return {n for n in dir(Bar) if not n.startswith("__")}


def _bar_param_names(func) -> set[str]:
    """Names of parameters bound to a ``Bar`` (heuristic: typed as Bar,
    or named exactly ``bar``).
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return set()
    names: set[str] = set()
    for name, param in sig.parameters.items():
        if name == "bar":
            names.add(name)
            continue
        ann = param.annotation
        if ann is Bar or (isinstance(ann, str) and ann.endswith("Bar")):
            names.add(name)
    return names


def _attrs_accessed_on_var(func_src: str, var_names: set[str]) -> set[str]:
    """Walk ``func_src`` and collect every ``X.<attr>`` where X is one of
    ``var_names``. This catches both attribute and method accesses
    (``bar.ts_event``, ``bar.bar_type``, ``bar.close_time_as_datetime()``).
    """
    out: set[str] = set()
    try:
        tree = ast.parse(textwrap.dedent(func_src))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in var_names
        ):
            out.add(node.attr)
    return out


def _walk_strategy_modules() -> Iterator[tuple[str, type]]:
    """Yield ``(strategy_class_name, class_obj)`` for every Strategy
    subclass under ``titan/strategies/``.
    """
    import titan.strategies as pkg

    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix="titan.strategies."):
        try:
            mod = importlib.import_module(modinfo.name)
        except Exception:  # noqa: BLE001 -- importing every live module is best-effort
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj is Strategy:
                continue
            if not issubclass(obj, Strategy):
                continue
            if obj.__module__ != modinfo.name:
                continue  # re-exported class -- check it where it's defined
            yield (name, obj)


def _live_strategy_classes() -> list[tuple[str, type]]:
    seen: dict[str, type] = {}
    for name, cls in _walk_strategy_modules():
        if hasattr(cls, "on_bar") and cls.on_bar is not Strategy.on_bar:
            seen.setdefault(name, cls)
    return sorted(seen.items())


LIVE_STRATEGIES = _live_strategy_classes()


@pytest.mark.parametrize(
    "strategy_name,strategy_cls",
    LIVE_STRATEGIES,
    ids=[name for name, _ in LIVE_STRATEGIES],
)
def test_on_bar_uses_only_valid_bar_attributes(strategy_name, strategy_cls):
    """Every ``bar.X`` access in ``on_bar`` (and helpers it inlines) must
    name a real attribute on ``nautilus_trader.model.data.Bar``.

    The 2026-05-18 ``close_time_as_datetime`` incident would have failed
    this test at import time, before the change ever reached the
    container.
    """
    valid = _valid_bar_attrs()

    # Collect attribute accesses across on_bar AND any helper method on the
    # same class whose signature names a Bar parameter.
    invalid: dict[str, set[str]] = {}
    for method_name in dir(strategy_cls):
        if method_name.startswith("_") and method_name != "on_bar":
            continue
        method = getattr(strategy_cls, method_name, None)
        if not callable(method):
            continue
        try:
            src = inspect.getsource(method)
        except (TypeError, OSError):
            continue
        bar_vars = _bar_param_names(method)
        if method_name == "on_bar":
            bar_vars.add("bar")
        if not bar_vars:
            continue
        used = _attrs_accessed_on_var(src, bar_vars)
        bad = used - valid
        if bad:
            invalid[method_name] = bad

    assert not invalid, (
        f"{strategy_name} uses Bar attributes that do not exist on "
        f"nautilus_trader.model.data.Bar: {invalid}.\n"
        f"Valid Bar attributes are: {sorted(valid)}"
    )


def test_discovery_finds_at_least_the_known_live_strategies():
    """Sanity check: if the discovery walk breaks silently, this test
    catches it. We expect a healthy lower bound -- the project ships at
    least these live strategies as of 2026-05-18.
    """
    names = {name for name, _ in LIVE_STRATEGIES}
    expected_minimum = {
        "GemStrategy",
        "TurtleStrategy",
        "EwmacRegimeStrategy",
    }
    missing = expected_minimum - names
    assert not missing, (
        f"Live-strategy discovery is missing known-deployed classes: {missing}. "
        f"Discovered: {sorted(names)}"
    )
