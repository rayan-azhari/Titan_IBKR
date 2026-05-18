# External quant audit cadence

**Status**: active process directive from 2026-05-18
**Origin**: Gemini external audit (2026-05-18) caught 4 portfolio-risk + allocator math bugs that 14 internal regression tests had missed. The internal tests were written by the same authors who wrote the code -- a convergent blind spot. An outside perspective with no project context found in one pass what familiarity had hidden for months.

## The rule

Schedule an **external quant audit every quarter**, AND before any major risk / allocator architecture change. "External" means: a reviewer (human or LLM with fresh context) who has not been involved in the design of the code under review.

## Why quarterly

| Trigger | Why it's not enough alone |
|---|---|
| Code review per PR | Reviewers see the diff, not the system. They check "does this change look right?" not "is this whole module right?" |
| Pre-deployment review | Catches deploy bugs, not foundational math bugs. The PortfolioRiskManager bugs had been in production for weeks before Gemini found them. |
| Internal post-mortems | Only fire after a loss. The Gemini-caught bugs had not yet caused a real loss but would have on the next strategy add. |

Quarterly is short enough that drift doesn't compound, long enough that the audit gets full attention rather than rubber-stamping.

## What to audit

Each quarter, rotate focus across the four high-stakes surfaces. One per quarter on a rolling basis means each is fully reviewed annually:

1. **Q1 / Risk math**: `titan/risk/portfolio_risk_manager.py`, `titan/risk/portfolio_allocator.py`, `titan/risk/strategy_equity.py`, `titan/research/metrics.py`. Vol calculation, drawdown gates, FX conversion, EWMA, allocator weights.
2. **Q2 / Strategy execution**: every `titan/strategies/*/strategy.py` -- sizing, position state, kill-switch, REHYDRATE semantics. The 2026-05-18 `close_time_as_datetime` bug would live here.
3. **Q3 / Research framework**: `titan/research/framework/` -- typology, WFO, sanctuary, MC, DSR, decision matrix. Any change to the deployment gates needs external review.
4. **Q4 / Data + infra**: data acquisition scripts, Docker / container orchestration, IBKR adapter wiring, halt persistence, watchdog logic.

The next scheduled audit is **2026-08-18 (Q3 / Research framework)**. Owner: human operator; reviewer: LLM with fresh context (Gemini Pro or equivalent), provided with the directory under review + the V3.6 lessons catalogue but NOT the strategy result logs (avoid anchoring on past verdicts).

## Inputs to the reviewer

The 2026-05-18 audit succeeded because Gemini was given a focused scope:

- The risk-math directory under review.
- `directives/Methodology Audit & Unified Framework 2026-05-14.md` for the standard.
- `directives/V3.6 Lessons Catalogue.md` for prior failure modes.

It was NOT given: result logs, Sharpe numbers, deployment status. Those would bias the reviewer toward "this strategy is already live, so it must be fine."

## Output expected

A markdown report (like `Titan-IBKR Portfolio & Methodology Audit Report.md`) with:

1. Findings ranked by severity (high / medium / low).
2. Location (file + line).
3. Concrete fix or test for each.
4. Plan for the operator to verify each fix before merge.

Findings that turn into bug fixes get a regression test named per [Test Naming Convention 2026-05-18.md](Test%20Naming%20Convention%202026-05-18.md).

## Cost

A focused external audit takes ~2 hours of operator time to prepare the inputs + ~30 minutes of LLM time to review + ~2-8 hours to implement fixes. Total per quarter: 1 working day.

A single missed live-trading math bug costs an order of magnitude more.

## Tracking

Each completed audit appends a line to this directive:

| Date | Surface | Bugs found | Reviewer | Report |
|---|---|---|---|---|
| 2026-05-18 | Risk math (Q2 ad-hoc) | 3 high/med + 1 latent | Gemini | `Titan-IBKR Portfolio & Methodology Audit Report.md` |
