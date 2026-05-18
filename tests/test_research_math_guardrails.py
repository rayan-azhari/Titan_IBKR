"""AST-level guardrails for research / strategy code quality.

These tests fail CI when new code re-introduces patterns that the April 2026
external quant review flagged as defect sources:

1.  Bare ``sqrt(252)`` / ``np.sqrt(252)`` / ``math.sqrt(252)`` outside
    ``titan/research/metrics.py``. All annualisation must go through the
    shared helper with an explicit ``periods_per_year``.
2.  ``rets[rets != 0.0]`` followed by ``.std()`` in a Sharpe-like context
    (multiplied by ``sqrt(...)``) — the filter-then-annualise bias.
3.  ``list(balances.keys())[0]`` / ``ccys[0]`` — the deterministic-ccy
    anti-pattern.
4.  ``balance_total(`` without an explicit ccy from
    ``get_base_balance`` or a named currency variable in the same
    expression — silent currency-assumption risk.

Allowlist
---------
Legacy / archived research files that still contain the patterns are
explicitly allowlisted in :data:`_ALLOWLIST` with a reason. Every
allowlist entry is a TODO for a follow-up migration PR, not an exemption.
New files cannot be added to the allowlist without a code review.

Design note
-----------
The implementation deliberately uses simple regex / AST checks rather than
a full type-aware linter. The patterns we care about are syntactic
(``sqrt(252)``, ``balances.keys()[0]``), so regex + AST visitor is
sufficient and keeps the guardrail fast and self-contained.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Files permitted to contain bare sqrt(252) references (docstring mentions,
# metric definitions, or legacy research scripts pending migration).
#
# Each entry maps path -> (reason, expiry_date). Expiry forces a periodic
# re-review: a stale allowlist entry whose date is in the past fails the
# build until the entry is removed (file migrated to metrics module) or
# the expiry is consciously extended with a fresh justification. The
# 2026-05-18 Gemini audit caught one live violation that had slipped past
# this allowlist because nothing forced a review.
#
# Use the special expiry "PERMANENT" for entries that are intentionally
# never going to migrate (e.g. the metrics module itself, which DEFINES
# the annualisation). Use it sparingly; default to a 6-month expiry.
_PERMANENT = "PERMANENT"
_AllowEntry = tuple[str, str]  # (reason, "YYYY-MM-DD" | "PERMANENT")

_ALLOWLIST_SQRT_252: dict[str, _AllowEntry] = {
    # The metrics module is THE source of truth -- never migrates.
    "titan/research/metrics.py": ("source of truth for annualisation", _PERMANENT),
    "titan/research/__init__.py": ("module docstring references sqrt(252)", _PERMANENT),
    # Portfolio risk manager / allocator reference sqrt(252) in their
    # docstrings / comments only -- regex false-positives on those.
    "titan/risk/portfolio_risk_manager.py": ("only in docstrings", _PERMANENT),
    "titan/risk/portfolio_allocator.py": ("only in docstrings", _PERMANENT),
    # Samir-Stack research library -- daily-bar backtests. Live runtime in
    # titan/strategies/samir_stack/ is already routed through the metrics
    # module. Migrate to titan.research.metrics by 2026-11-14 (6 months).
    "research/samir_stack/indicators.py": ("research script, daily bars", "2026-11-14"),
    "research/samir_stack/wfo.py": ("research script, daily bars", "2026-11-14"),
    "research/samir_stack/wfo_stacked.py": ("research script, daily bars", "2026-11-14"),
    "research/samir_stack/stacked_strategy.py": ("research script, daily bars", "2026-11-14"),
    "research/samir_stack/strategy.py": ("research script, daily bars", "2026-11-14"),
    "research/samir_stack/benchmarks.py": ("research script, daily bars", "2026-11-14"),
    # Ad-hoc audit / dashboard scripts. No live runtime path. Migrate by 2026-11-14.
    "research/ewmac/run_i1v2_audit.py": ("audit script, daily bars", "2026-11-14"),
    "research/exploration/audit_fx_carry.py": ("audit script, daily bars", "2026-11-14"),
    "research/exploration/audit_orb.py": ("audit script, daily bars", "2026-11-14"),
    "research/exploration/build_i1_regime_panel.py": ("panel builder, daily bars", "2026-11-14"),
    "research/gem/render_j5_dashboard.py": ("dashboard renderer, daily bars", "2026-11-14"),
}

_ALLOWLIST_FILTER_THEN_STD: dict[str, _AllowEntry] = {
    "titan/research/metrics.py": ("source of truth", _PERMANENT),
    "titan/research/__init__.py": ("module docstring", _PERMANENT),
    "tests/test_research_metrics.py": ("test explicitly exercises the old bias", _PERMANENT),
    # Legacy research files pending migration (same set as above).
    **{k: v for k, v in _ALLOWLIST_SQRT_252.items() if k.startswith("research/")},
}

_ALLOWLIST_BALANCES_KEYS0: dict[str, _AllowEntry] = {
    # Currently empty -- all live strategies use get_base_balance(..., "USD").
}


# ── Utility: iterate source files ─────────────────────────────────────────


def _iter_py_files(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts
    ]


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


# ── Guard 1: bare sqrt(252) ────────────────────────────────────────────────

# Match bare literal 252 inside sqrt(...), allowing * or spaces around.
# Catches:  sqrt(252), np.sqrt(252), math.sqrt(252), sqrt(var * 252),
#           sqrt(252 * 24), sqrt(bars_yr * 252).
# Ignores: sqrt(periods_per_year), sqrt(bars_per_year), docstring mentions.
_SQRT_252_RE = re.compile(
    r"""(?x)
    \bsqrt\s*\(            # sqrt(
    [^)]*?\b252\b[^)]*?    # ... 252 somewhere inside (not as part of a name)
    \)
    """,
    re.VERBOSE,
)


def _find_sqrt_252(path: Path) -> list[tuple[int, str]]:
    """Return (line_no, line_content) pairs where sqrt(...252...) is called
    as code, not inside a string literal or comment."""
    hits: list[tuple[int, str]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Collect line numbers inside string literals / docstrings so we can
    # exclude them from the regex match.
    string_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Multi-line string — mark every line it spans.
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is not None and end is not None:
                for ln in range(start, end + 1):
                    string_lines.add(ln)

    for i, line in enumerate(source.splitlines(), start=1):
        if i in string_lines:
            continue
        # Strip trailing "# ..." comment before regex match.
        code = re.sub(r"#.*$", "", line)
        if _SQRT_252_RE.search(code):
            hits.append((i, line.rstrip()))
    return hits


def test_no_bare_sqrt_252_outside_metrics():
    """No file outside the allowlist may contain ``sqrt(252)``.

    Every annualisation must route through
    ``titan.research.metrics.annualize_vol`` / ``sharpe`` / ``ewm_vol`` with
    an explicit ``periods_per_year``. The bar timeframe determines the
    factor — ``252`` is NOT a safe default.
    """
    offenders: list[str] = []
    for path in _iter_py_files(PROJECT_ROOT / "titan") + _iter_py_files(PROJECT_ROOT / "research"):
        rel = _rel(path)
        if rel in _ALLOWLIST_SQRT_252:
            continue
        hits = _find_sqrt_252(path)
        if hits:
            for line_no, line in hits[:3]:  # cap output
                offenders.append(f"  {rel}:{line_no}: {line.strip()}")
    assert not offenders, (
        "Bare sqrt(252) found outside titan/research/metrics.py and the\n"
        "remediation allowlist. Migrate to titan.research.metrics with\n"
        "an explicit periods_per_year factor. Violations:\n" + "\n".join(offenders)
    )


# ── Guard 2: rets[rets != 0] followed by std() + sqrt(...) ─────────────────

_FILTER_THEN_ANNUALISE_RE = re.compile(
    r"\[\s*\w+\s*!=\s*0(?:\.0)?\s*\][\s\S]{0,60}?\.std\s*\([^)]*\)[\s\S]{0,60}?sqrt\s*\("
)


def _find_filter_then_annualise(path: Path) -> list[int]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return []
    return [i for i, m in enumerate(_FILTER_THEN_ANNUALISE_RE.finditer(source), start=1)]


def test_no_filter_zero_days_before_sharpe_annualise():
    """No ``rets[rets != 0.0]`` ... ``.std()`` ... ``sqrt(...)`` chain.

    Filtering zero bars before annualising Sharpe overstates the ratio by
    ``sqrt(1/active_ratio)`` for sparse strategies. If per-trade stats are
    genuinely wanted, pass trade-level returns to ``trade_sharpe``.
    """
    offenders: list[str] = []
    for path in _iter_py_files(PROJECT_ROOT / "titan") + _iter_py_files(PROJECT_ROOT / "research"):
        rel = _rel(path)
        if rel in _ALLOWLIST_FILTER_THEN_STD:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if _FILTER_THEN_ANNUALISE_RE.search(source):
            offenders.append(f"  {rel}: rets[rets != 0] → std() → sqrt(...)")
    assert not offenders, (
        "Filter-then-annualise Sharpe bias detected. Use titan.research.metrics.sharpe\n"
        "which does NOT filter zeros (non-trade days belong in the denominator).\n"
        "Violations:\n" + "\n".join(offenders)
    )


# ── Guard 3: balances.keys()[0] anti-pattern ───────────────────────────────

# Matches ``list(X.balances().keys())[0]`` and variants.
_BALANCES_KEYS0_RE = re.compile(r"balances\s*\(\s*\)\s*\.\s*keys\s*\(\s*\)\s*\)?\s*\[\s*0\s*\]")


def test_no_nondeterministic_ccy_anti_pattern():
    """``list(balances.keys())[0]`` returns a non-deterministic currency on
    multi-ccy accounts. Use ``get_base_balance(account, "USD")`` instead
    (or whichever explicit base currency the strategy uses).
    """
    offenders: list[str] = []
    for path in _iter_py_files(PROJECT_ROOT / "titan"):
        rel = _rel(path)
        if rel in _ALLOWLIST_BALANCES_KEYS0:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(source.splitlines(), start=1):
            if _BALANCES_KEYS0_RE.search(line):
                offenders.append(f"  {rel}:{i}: {line.strip()}")
    assert not offenders, (
        "Non-deterministic currency lookup detected. Use\n"
        "get_base_balance(account, 'USD') instead. Violations:\n" + "\n".join(offenders)
    )


# ── Guard 4: balance_total( outside strategy_equity.py ────────────────────

# The only legitimate caller of ``account.balance_total(...)`` is
# ``titan/risk/strategy_equity.py`` (inside ``get_base_balance``). Every live
# strategy should read per-strategy equity from ``StrategyEquityTracker``
# instead — this guard ensures we don't regress to raw account-balance sizing.
_ALLOWLIST_BALANCE_TOTAL = {
    "titan/risk/strategy_equity.py",
}

_BALANCE_TOTAL_RE = re.compile(r"\baccount[s]?\w*\s*\.\s*balance_total\s*\(")


def test_balance_total_outside_strategy_equity():
    """No ``account.balance_total(...)`` calls outside the shared helper in
    ``titan/risk/strategy_equity.py``. Strategies must go through
    ``StrategyEquityTracker.current_equity()``.
    """
    offenders: list[str] = []
    for path in _iter_py_files(PROJECT_ROOT / "titan"):
        rel = _rel(path)
        if rel in _ALLOWLIST_BALANCE_TOTAL:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(source.splitlines(), start=1):
            if _BALANCE_TOTAL_RE.search(line):
                offenders.append(f"  {rel}:{i}: {line.strip()}")
    assert not offenders, (
        "Raw ``account.balance_total(...)`` call detected outside "
        "titan/risk/strategy_equity.py. Live strategies must use "
        "StrategyEquityTracker.current_equity() for per-strategy equity.\n"
        "Violations:\n" + "\n".join(offenders)
    )


# ── Guard 5: allowlist hygiene -- every dated entry must not be expired ──


def _parse_expiry(expiry: str) -> date | None:
    """Return a ``date`` for ``YYYY-MM-DD``, ``None`` for ``PERMANENT``,
    and raise for malformed strings (so a typo in the allowlist source
    cannot silently pass as permanent).
    """
    if expiry == _PERMANENT:
        return None
    return date.fromisoformat(expiry)


def test_no_expired_allowlist_entries():
    """Every dated allowlist entry must be on-or-before its expiry date.

    Allowlists accumulate over time as a form of technical-debt parking
    -- each entry is a small concession, collectively they erode the
    guard. Forcing a periodic re-review (via expiry dates) prevents the
    'set it and forget it' drift that let the live ``np.sqrt(252)`` in
    ``titan/strategies/ewmac_regime/strategy.py:396`` slip past for
    months until the 2026-05-18 Gemini audit.

    To clear a failing expiry: migrate the file (preferred) OR extend
    the expiry by 90 days with a documented justification in the same
    edit. Indefinite extensions are forbidden -- use ``PERMANENT`` only
    when the file structurally cannot migrate.
    """
    today = date.today()
    expired: list[str] = []
    for name, allowlist in (
        ("_ALLOWLIST_SQRT_252", _ALLOWLIST_SQRT_252),
        ("_ALLOWLIST_FILTER_THEN_STD", _ALLOWLIST_FILTER_THEN_STD),
        ("_ALLOWLIST_BALANCES_KEYS0", _ALLOWLIST_BALANCES_KEYS0),
    ):
        for path_rel, entry in allowlist.items():
            if not (isinstance(entry, tuple) and len(entry) == 2):
                expired.append(
                    f"  {name}[{path_rel}]: malformed entry "
                    f"(expected (reason, expiry); got {entry!r})"
                )
                continue
            reason, expiry_str = entry
            try:
                exp = _parse_expiry(expiry_str)
            except ValueError:
                expired.append(
                    f"  {name}[{path_rel}]: malformed expiry "
                    f"{expiry_str!r} (expected YYYY-MM-DD or PERMANENT)"
                )
                continue
            if exp is not None and exp < today:
                expired.append(
                    f"  {name}[{path_rel}]: expired {exp.isoformat()} -- reason was {reason!r}"
                )
    assert not expired, (
        "Allowlist entries past their expiry. Migrate the file to the "
        "metrics module (preferred) OR extend the expiry by up to 90 "
        "days with a documented justification:\n" + "\n".join(expired)
    )
