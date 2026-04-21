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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Files permitted to contain bare sqrt(252) references (docstring mentions,
# metric definitions, or legacy research scripts pending migration).
#
# Each entry names the file path relative to PROJECT_ROOT and a short reason.
# Keep this list short and SHRINKING. New files require explicit review.
_ALLOWLIST_SQRT_252: dict[str, str] = {
    # The metrics module is THE source of truth — all sqrt(periods_per_year)
    # calls live here.
    "titan/research/metrics.py": "source of truth for annualisation",
    "titan/research/__init__.py": "module docstring references sqrt(252)",
    # Portfolio risk manager / allocator reference sqrt(252) in their
    # docstrings / comments only — actual calls route through the metrics
    # module. Regex would false-positive on those comments.
    "titan/risk/portfolio_risk_manager.py": "only in docstrings",
    "titan/risk/portfolio_allocator.py": "only in docstrings",
    # Archived research — kept for historical audit trail only; never
    # imported from the live code path.
    "research/_archive/portfolio/run_portfolio_research.py": "archived",
    "research/_archive/portfolio/run_ftse_dax_expansion.py": "archived",
    "research/_archive/portfolio/loaders/oos_returns.py": "archived",
    # phase_portfolio.py has the docstring mention but no bare sqrt(252)
    # call; it's already routed through the metrics module.
    "research/auto/phase_portfolio.py": "docstring mention only",
}

_ALLOWLIST_FILTER_THEN_STD: dict[str, str] = {
    "titan/research/metrics.py": "source of truth",
    "titan/research/__init__.py": "module docstring",
    "tests/test_research_metrics.py": "test explicitly exercises the old bias",
    # Legacy research files pending migration (same set as above).
    **{k: v for k, v in _ALLOWLIST_SQRT_252.items() if k.startswith("research/")},
}

_ALLOWLIST_BALANCES_KEYS0: dict[str, str] = {
    # Currently empty — all live strategies use get_base_balance(..., "USD").
    # turtle/strategy.py previously used the anti-pattern; now fixed.
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
