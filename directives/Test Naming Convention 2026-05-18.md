# Test naming convention: event-shaped, not function-shaped

**Status**: active convention, retroactive from 2026-05-18
**Origin**: Gemini external audit (2026-05-18) caught 4 PortfolioRiskManager + Allocator bugs that internal regression tests had missed. All four were in code the prior author had written tests for and approved. Common pattern: tests were named after the function they exercised (`test_recompute_daily_vol_works`) and asserted "doesn't crash" / "math is correct" -- they did not encode the operational SCENARIO that breaks the function.

## The rule

When writing a test for live-trading code, name the test after the **event** or **scenario** the code must handle, not after the **function** under test.

| Function-shaped (bad) | Event-shaped (good) |
|---|---|
| `test_recompute_daily_vol_works` | `test_recompute_daily_vol_ignores_capital_addition` |
| `test_allocator_rebalance` | `test_allocator_floor_never_violated_with_missing_strategy` |
| `test_risk_contributions_returns_dict` | `test_risk_contributions_expose_unmeasured_bucket` |
| `test_strategy_on_bar` | `test_on_bar_handles_first_daily_bar_after_restart` |

## Why it works

Function-shaped names force you to ask "does this function compile and return something?" Event-shaped names force you to ask "what could happen in production that this function must handle?"

The answer to the second question is what catches bugs. The first question gets you green tests on broken code.

## Practical checklist (write the test name first)

When opening a new test file for live-trading code, write the names BEFORE the test bodies. For each name, answer:

1. **What real-world event triggers this code path?** (new strategy registered, container restart, first weekend bar, kill-switch trips, broker disconnect, ...)
2. **What is the invariant that must hold?** (vol_scale not collapsed, weight floor not violated, no unmeasured-risk pretending to be measured, ...)
3. **What is the minimal scenario that exercises the path?**

A test that doesn't answer (1) is a function-shaped test. Rewrite it.

## Examples in this repo

The 2026-05-18 fix commit added three event-shaped tests:

- [test_recompute_daily_vol_ignores_capital_addition](../tests/test_portfolio_risk_april2026_fixes.py) -- the event is "register a strategy mid-life with seed equity".
- [test_allocator_floor_never_violated_with_missing_strategy](../tests/test_portfolio_risk_april2026_fixes.py) -- the event is "one strategy has fewer than min_history_days of data".
- [test_risk_contributions_expose_unmeasured_bucket](../tests/test_portfolio_risk_april2026_fixes.py) -- the event is "some strategies have <20d history".

[tests/test_live_strategy_bar_api.py](../tests/test_live_strategy_bar_api.py) is also event-shaped at the suite level -- the event is "Nautilus upgrades the Bar API and renames an attribute".

## Anti-patterns to avoid

- **`test_X_works`** -- the answer to "does X work" is usually "yes, on my machine, today".
- **`test_X_returns_correct_value`** -- requires the test author to know all correct values, which is the very thing under test.
- **`test_X_basic` / `test_X_edge_case`** -- vague; nobody knows what was actually exercised.
- **`test_init` / `test_constructor`** -- almost never useful; what is the SCENARIO that breaks construction?

## Relationship to parity tests

Parity tests (`test_*_parity.py`) are event-shaped by construction -- the event is "live runtime sees the same data as research backtest". Keep that naming; it works.

## Enforcement

This is a convention, not a CI gate. PR review should flag function-shaped names and request a rename. The repo's `directives/V3.6 Lessons Catalogue.md` would record any post-mortem where a function-shaped test allowed a bug to ship.
