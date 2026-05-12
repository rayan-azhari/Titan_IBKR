"""Tests for the chaos harness pure logic.

The actual chaos scenarios require docker + a running container stack
and are not testable in CI. The diff/log-scan/report-formatting logic
is pure and is what we exercise here.

Tier 2.3 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``).
"""

from __future__ import annotations

from scripts.chaos_harness import (
    ScenarioResult,
    _diff_snapshots,
    _format_report,
)

# ── _diff_snapshots ──────────────────────────────────────────────────


def test_diff_snapshots_no_change_no_findings():
    snap = {"VUSD": {"qty": 30.0, "position_count": 1}}
    assert _diff_snapshots(snap, snap) == []


def test_diff_snapshots_qty_drift_flagged():
    """Net qty changed → finding (could be legitimate fill or a bug)."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after = {"VUSD": {"qty": 60.0, "position_count": 1}}
    findings = _diff_snapshots(before, after)
    assert any("VUSD" in f and "qty" in f for f in findings)


def test_diff_snapshots_position_count_drift_flagged_as_suspicious():
    """The May 11 doubling-bug class: one position became two."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after = {"VUSD": {"qty": 30.0, "position_count": 2}}
    findings = _diff_snapshots(before, after)
    assert any("position_count" in f and "SUSPICIOUS" in f for f in findings)


def test_diff_snapshots_new_symbol_appears_flagged():
    """A symbol that wasn't there originally → flagged."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after = {
        "VUSD": {"qty": 30.0, "position_count": 1},
        "EIMI": {"qty": 21.0, "position_count": 1},  # new
    }
    findings = _diff_snapshots(before, after)
    assert any("EIMI" in f for f in findings)


def test_diff_snapshots_empty_inputs_no_findings():
    assert _diff_snapshots({}, {}) == []


def test_diff_snapshots_handles_missing_symbol_in_after():
    """A symbol present before but missing after = qty change to 0."""
    before = {"VUSD": {"qty": 30.0, "position_count": 1}}
    after: dict = {}
    findings = _diff_snapshots(before, after)
    assert any("VUSD" in f for f in findings)


# ── _format_report ───────────────────────────────────────────────────


def test_format_report_all_pass():
    """Happy path: all scenarios pass → clean PASS lines."""
    results = [
        ScenarioResult(name="S1 restart-storm", passed=True, duration_sec=42.0, findings=[]),
        ScenarioResult(name="S2 gateway-flap", passed=True, duration_sec=120.0, findings=[]),
    ]
    report = _format_report(results)
    assert "[PASS] S1 restart-storm" in report
    assert "[PASS] S2 gateway-flap" in report
    assert "FAIL" not in report


def test_format_report_failures_include_findings():
    """Failed scenarios show their findings under the FAIL line."""
    results = [
        ScenarioResult(
            name="S1 restart-storm",
            passed=False,
            duration_sec=10.0,
            findings=["  VUSD: position_count 1 -> 2 (SUSPICIOUS)"],
        )
    ]
    report = _format_report(results)
    assert "[FAIL] S1 restart-storm" in report
    assert "SUSPICIOUS" in report


def test_format_report_mixed_pass_fail():
    results = [
        ScenarioResult(name="S1", passed=True, duration_sec=5.0, findings=[]),
        ScenarioResult(name="S2", passed=False, duration_sec=10.0, findings=["  bad thing"]),
        ScenarioResult(name="S3", passed=True, duration_sec=15.0, findings=[]),
    ]
    report = _format_report(results)
    assert "[PASS] S1" in report
    assert "[FAIL] S2" in report
    assert "[PASS] S3" in report
    assert "bad thing" in report


# ── ScenarioResult ───────────────────────────────────────────────────


def test_scenario_result_default_findings_is_empty_list():
    """Each ScenarioResult instance gets its own list (not a class-shared
    default), so failures across scenarios don't leak."""
    r1 = ScenarioResult(name="A", passed=True, duration_sec=1.0)
    r2 = ScenarioResult(name="B", passed=True, duration_sec=2.0)
    r1.findings.append("x")
    assert r2.findings == [], "ScenarioResult.findings must be a per-instance list"
