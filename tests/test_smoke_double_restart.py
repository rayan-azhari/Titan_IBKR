"""Tests for the double-restart smoke test diff logic.

The full script orchestrates docker subprocess + ibapi I/O, neither of
which is testable in CI. But the per-snapshot diff logic — which
determines pass/fail — is pure and testable.

Tier 1.2 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).
"""

from __future__ import annotations

from scripts.smoke_double_restart import _diff_snapshots


def test_diff_three_identical_snapshots_no_findings():
    """Happy path: positions unchanged across all three snapshots."""
    snap = {"VUSD": {"qty": 30.0, "position_count": 1}}
    findings = _diff_snapshots([snap, snap, snap])
    assert findings == []


def test_diff_position_count_doubled_detected():
    """May 11 bug class: position count doubled after restart."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after = {"VUSD": {"qty": 30.0, "position_count": 2}}  # qty same, count doubled
    findings = _diff_snapshots([before, after, after])
    assert any("position_count" in f and "VUSD" in f for f in findings)
    assert any("SUSPICIOUS" in f for f in findings)


def test_diff_qty_changed_detected():
    """Net qty changed — could be a legitimate fill or a bug."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after = {"VUSD": {"qty": 60.0, "position_count": 1}}  # qty doubled
    findings = _diff_snapshots([before, before, after])
    assert any("qty changed" in f and "VUSD" in f for f in findings)


def test_diff_new_symbol_appears_detected():
    """A symbol that wasn't there originally should be flagged."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after = {
        "VUSD": {"qty": 30.0, "position_count": 1},
        "EIMI": {"qty": 21.0, "position_count": 1},  # new
    }
    findings = _diff_snapshots([before, before, after])
    # EIMI absent in 2 snapshots → qty values [0, 0, 21] → mismatch
    assert any("EIMI" in f for f in findings)


def test_diff_multiple_instruments_independent():
    """Mismatches in different instruments are reported independently."""
    before = {
        "VUSD": {"qty": 30.0, "position_count": 1},
        "EIMI": {"qty": 21.0, "position_count": 1},
    }
    after = {
        "VUSD": {"qty": 30.0, "position_count": 2},  # doubled
        "EIMI": {"qty": 21.0, "position_count": 1},  # unchanged
    }
    findings = _diff_snapshots([before, after, after])
    assert any("VUSD" in f for f in findings)
    assert not any("EIMI" in f for f in findings)


def test_diff_empty_snapshots_no_findings():
    """No positions, no findings."""
    findings = _diff_snapshots([{}, {}, {}])
    assert findings == []
