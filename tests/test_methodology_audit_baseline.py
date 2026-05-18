"""CI ratchet: methodology-audit findings can only DECREASE.

The scanner at ``scripts/audit_codebase_methodology.py`` flags known
anti-patterns (look-ahead bias, hardcoded annualisation, ad-hoc Sharpe
defs, etc). It is intentionally a sieve -- false positives are expected
and the operator reviews them. But unreviewed regressions historically
accumulate: someone adds a new ``np.sqrt(252)`` to a research script,
the scanner reports it, nobody acts, the count creeps up.

This test promotes the scanner from "aspirational" to "CI gate". Each
pattern has a frozen count in ``tests/baselines/methodology_audit_baseline.json``;
adding a NEW violation fails the build. To lower a baseline after
genuinely fixing violations, run the scanner and edit the baseline file
to match the new (lower) count. Increasing a baseline is allowed only
with a documented justification (in the JSON file's leading "_doc"
key) plus reviewer approval.

The 2026-05-18 Gemini audit caught one ``np.sqrt(252)`` in the LIVE
EwmacRegime strategy that had slipped past the existing
``test_research_math_guardrails.py`` allowlist. The ratchet would have
flagged the regression at merge time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "tests" / "baselines" / "methodology_audit_baseline.json"

# Make ``scripts/`` importable without polluting sys.path elsewhere.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import audit_codebase_methodology as scanner  # noqa: E402


def _load_baseline() -> dict[str, int]:
    raw = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in raw["counts"].items()}


def test_methodology_audit_baseline_not_exceeded():
    """For every pattern with a baseline, the current count must be
    less than or equal to the baseline. Catches REGRESSIONS at merge
    time. Lowering a baseline is a separate operator action.
    """
    baseline = _load_baseline()
    findings = scanner.scan_all()
    current = scanner.counts_by_pattern(findings)

    regressions: dict[str, tuple[int, int]] = {}
    for pattern_id, baseline_count in baseline.items():
        cur = current.get(pattern_id, 0)
        if cur > baseline_count:
            regressions[pattern_id] = (cur, baseline_count)

    assert not regressions, (
        "Methodology-audit regression detected. New violations of one or "
        "more patterns vs the frozen baseline at "
        "tests/baselines/methodology_audit_baseline.json:\n"
        + "\n".join(
            f"  {pid}: current={cur} > baseline={base}"
            for pid, (cur, base) in regressions.items()
        )
        + "\n\nEither fix the new violations OR (with reviewer approval and "
        "a documented reason) raise the baseline by editing the JSON file."
    )


def test_methodology_audit_baseline_in_sync_with_known_patterns():
    """The baseline file should not contain stale entries for patterns
    the scanner no longer emits. If a pattern is removed from the
    scanner, drop its baseline entry too.
    """
    baseline = _load_baseline()
    known_pattern_ids = {p[0] for p in scanner.PATTERNS}
    stale = set(baseline.keys()) - known_pattern_ids
    assert not stale, (
        f"Baseline contains entries for patterns the scanner no longer "
        f"emits: {stale}. Remove them from "
        f"tests/baselines/methodology_audit_baseline.json."
    )


def test_methodology_audit_baseline_opportunity_to_tighten():
    """Soft-fail informational: if the current count is STRICTLY LESS
    than the baseline for some pattern, that's a chance to ratchet the
    baseline tighter. Logged as a warning, not a failure, because some
    transient noise (e.g., a deleted file) shouldn't break the build --
    but you should lower the baseline at your next convenience.
    """
    baseline = _load_baseline()
    findings = scanner.scan_all()
    current = scanner.counts_by_pattern(findings)
    opportunities = {
        pid: (current.get(pid, 0), base)
        for pid, base in baseline.items()
        if current.get(pid, 0) < base
    }
    if opportunities:
        print(
            "\n[ratchet opportunity] Some pattern counts are below baseline. "
            "Consider lowering the baseline in "
            "tests/baselines/methodology_audit_baseline.json:"
        )
        for pid, (cur, base) in opportunities.items():
            print(f"  {pid}: current={cur} < baseline={base}")
    # No assertion -- soft signal only.
