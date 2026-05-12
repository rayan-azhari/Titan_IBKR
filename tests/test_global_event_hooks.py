"""Sanity tests for the global event hooks.

We can't easily stand up a full TradingNode in CI (would need the IBKR
adapter and a gateway), so these tests focus on the hook module's
import-level invariants and the handler dispatch logic, which are the
parts most likely to break on a NautilusTrader version bump.

Tier 1.4 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


def test_module_imports_without_side_effects():
    """The hook module must import cleanly even when notification config
    is absent. Importing must not fire any network calls or background
    threads — it's loaded by run_portfolio at startup."""
    from titan.utils import global_event_hooks  # noqa: F401


def test_register_requires_built_node():
    """Calling ``register_global_event_hooks`` on a node that hasn't been
    built() yet must raise a clear RuntimeError, not a cryptic
    AttributeError."""
    from titan.utils.global_event_hooks import register_global_event_hooks

    fake_node = SimpleNamespace()  # no ``kernel`` attribute

    with pytest.raises(RuntimeError, match="after node.build"):
        register_global_event_hooks(fake_node)


def test_handler_dispatches_rejected_to_notifier():
    """Handler must call notify_order_event with event_type='rejected'
    when an OrderRejected message is received."""
    from nautilus_trader.model.events import OrderRejected

    from titan.utils import global_event_hooks

    fake_event = mock.Mock(spec=OrderRejected)
    fake_event.strategy_id = "test_strategy"
    fake_event.instrument_id = "EUR/USD.IDEALPRO"
    fake_event.client_order_id = "O-test-1"
    fake_event.venue_order_id = "12345"
    fake_event.reason = "Test rejection<br>from IBKR"

    # Patch isinstance so the spec-based mock is recognised as OrderRejected
    with mock.patch.object(global_event_hooks, "notify_order_event") as mock_notify:
        with mock.patch.object(
            global_event_hooks, "isinstance", lambda obj, cls: cls is OrderRejected
        ):
            global_event_hooks._on_order_event(fake_event)

    assert mock_notify.called
    call_kwargs = mock_notify.call_args.kwargs
    assert call_kwargs["event_type"] == "rejected"
    assert call_kwargs["strategy"] == "test_strategy"
    assert call_kwargs["instrument"] == "EUR/USD.IDEALPRO"
    # Reason should be HTML-flattened
    assert "<br>" not in call_kwargs["note"]
    assert "Test rejection from IBKR" in call_kwargs["note"]


def test_handler_swallows_exceptions():
    """If notify_order_event throws (e.g. Slack URL malformed at runtime),
    the handler must NOT propagate the exception into the runtime loop."""
    from nautilus_trader.model.events import OrderRejected

    from titan.utils import global_event_hooks

    fake_event = mock.Mock(spec=OrderRejected)
    fake_event.strategy_id = "test"
    fake_event.instrument_id = "TEST"
    fake_event.client_order_id = "x"
    fake_event.venue_order_id = "y"
    fake_event.reason = "boom"

    with mock.patch.object(
        global_event_hooks, "notify_order_event", side_effect=RuntimeError("slack down")
    ):
        with mock.patch.object(
            global_event_hooks, "isinstance", lambda obj, cls: cls is OrderRejected
        ):
            # Must not raise
            global_event_hooks._on_order_event(fake_event)


def test_handler_ignores_unrelated_event_types():
    """Other events on the same topic must be silently ignored (no notify)."""
    from titan.utils import global_event_hooks

    fake_event = SimpleNamespace(name="some_other_event")  # not OrderRejected/Canceled

    with mock.patch.object(global_event_hooks, "notify_order_event") as mock_notify:
        global_event_hooks._on_order_event(fake_event)
    assert not mock_notify.called
