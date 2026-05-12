# Operational Robustness Framework

**Created:** 2026-05-12
**Trigger incident:** [Rehydration Bug 2026-05-11.md](Rehydration%20Bug%202026-05-11.md) — `BondGoldStrategy` doubled positions on every container restart for ~12 days because `str(position.side) == "LONG"` failed for EXTERNAL-rehydrated positions in NautilusTrader 1.221's Cython enum repr.

## 1. Problem statement

A live algorithmic-trading system at this scale faces two distinct risk pools:

- **Statistical risk** — what the price path can do to the strategy. The Samir-Stack risk-of-ruin analysis ([run_risk_of_ruin.py](../research/samir_stack/run_risk_of_ruin.py)) shows this is essentially zero (P(MaxDD > 25%) over 20 years ≈ 0% in 5,000 bootstrap paths).
- **Operational risk** — what bugs, deployments, and infrastructure failures can do to the strategy. The May 11 incident demonstrated this is the *dominant* risk: a single Cython enum-repr change quietly inflated positions to ~8× target on the live paper account before being caught.

The goal of this framework is to **systematically push bug detection earlier in the lifecycle**, where it's orders of magnitude cheaper to fix.

## 2. Defense-in-depth framework

Every bug gets caught at one of six layers. The cost of catching a bug grows ~10× per layer:

| Layer | When it fires | Cost to find | Cost to fix |
|---|---|---|---|
| **L1** Static analysis (lint, types, AST guards) | code-write time | seconds | minutes |
| **L2** Unit tests | PR time | minutes | hours |
| **L3** Integration tests in CI | merge time | hours | hours |
| **L4** Pre-deploy smoke | deploy time | hours | hours |
| **L5** Live monitoring (alerts, dashboards) | minutes-to-hours after fire | half a day + recovery | hours-days |
| **L6** Post-mortem after incident | hours-to-days after fire | a day + remediation + lost capital | days |

**The May 11 bug travelled all the way to L6.** The remediation we shipped (AST guard, regression test) pulls *future re-introductions* up to L1. But the bug itself was caught only by chance during a routine `docker logs` review. That's the fragility this framework targets.

## 3. Bug categories the system actually faces

Roughly in order of frequency × blast radius:

1. **State-management bugs** — rehydration, position tracking, in-memory cache vs broker truth. *(Today's class.)*
2. **Order-management bugs** — wrong size/side/instrument, stale orders not cancelled, unintentional doubling.
3. **Connection bugs** — gateway drops, missed fill events, race conditions on reconnect.
4. **Sizing/equity bugs** — wrong NLV, FX conversion, leverage applied twice.
5. **Dependency-upgrade bugs** — NautilusTrader Cython internals change between versions. *(Also today's class.)*
6. **Configuration drift** — TOML typos, env-var defaults, port mismatches.
7. **Data-quality bugs** — stale parquet warmup, missing bars, splits/dividends not adjusted.
8. **Resource exhaustion** — disk full, memory leak, log rotation failure.
9. **Auth/secrets** — token expiry, gateway 2FA timeout.

The May 11 incident hit categories 1 *and* 5 simultaneously and they reinforced each other (NT version change broke the state-rehydration semantics).

## 4. Tiered remediation plan

### Tier 1 — High value, low effort, do first

These are highest-leverage. Implementing items 1, 2, 5 individually would each have caught the May 11 bug. Together they form a belt-and-braces guard.

| # | Item | Catches at | Effort |
|---|---|---|---|
| 1.1 | **Position-reconciliation watchdog** — diff `broker.get_positions()` vs `cache.positions()` every 5 min, alert on mismatch | L5 | medium |
| 1.2 | **Double-restart smoke** — pre-deploy automation: stop, restart, stop, restart, verify net positions unchanged | L4 | small |
| 1.3 | **NLV divergence alarm** — track expected NLV vs actual, alert on >2% drift | L5 | medium |
| 1.4 | **Order-rejection page** — every `OrderRejected` event becomes a Slack alert, not just a log line | L5 | small |
| 1.5 | **NT API contract tests** — pin specific NautilusTrader behaviours we depend on; fail loudly on version bump | L2 / L3 | small |

### Tier 2 — Medium effort, do this month

| # | Item | Catches at | Effort |
|---|---|---|---|
| 2.1 | **Shadow-strategy mode** — parallel instance that only logs decisions; diff vs live every bar | L5 | medium |
| 2.2 | **Property-based state-machine tests** — Hypothesis generates random sequences of (rehydrate, bar, fill, restart, partial-fill, reject, reconnect); assert invariants | L2 | medium |
| 2.3 | **Chaos test harness** — scripted: kill mid-bar, network blip, broker disconnect during fill, replay events out of order. Run nightly on paper | L3 | medium-large |
| 2.4 | **Cost-model audit job** — weekly cron compares modelled commission/slippage vs realised fills; alert on >50% drift | L5 | small |
| 2.5 | **Backtest-vs-live reconciliation** — replay each day's bars through backtest from actual prior-EOD positions; assert action matches | L5 | medium |

### Tier 3 — High effort, do this quarter

| # | Item | Catches at | Effort |
|---|---|---|---|
| 3.1 | **Deterministic broker simulator** — test harness for replaying any sequence of broker events into a strategy | L2 / L3 | large |
| 3.2 | **Mutation testing** — `mutmut`/`cosmic-ray`: introduce small code changes, verify tests catch them | L2 | medium |
| 3.3 | **Pre-flight invariant suite per strategy** — every new strategy must pass: starts flat from cold cache; rehydrates correctly; survives restart mid-position; respects PRM kill | L3 | medium |

### Tier 4 — Process / cultural

| # | Item | Catches at | Effort |
|---|---|---|---|
| 4.1 | **Incident-driven AST guards** — every production incident → AST/unit test that makes the bug class impossible to re-introduce. Already in use (rehydration tests, research-math guardrails) | L1 / L2 | minutes per incident |
| 4.2 | **Pre-PR "What Could Go Wrong" checklist** — three required questions in any PR touching trading code: (a) what state does this change? (b) what happens on restart with that state? (c) what happens if the broker reports something unexpected? | L1 | minutes per PR |
| 4.3 | **Runbook per failure mode** — markdown per known failure (PDT lockout, gateway daily restart, IBKR 2104, position phantom, etc.) with detection signature + immediate response | L6 | hours per failure mode |

## 5. May 11 incident — defense layers walkthrough

Mapping the rehydration bug through the framework:

| Layer | Did it catch? | Why / why not |
|---|---|---|
| L1 | ❌ | `str(position.side) == "LONG"` is syntactically valid; ruff has no rule for "brittle enum string compare." |
| L2 | ❌ | `tests/test_position_rehydration.py` covered the *rehydration filter* (kwarg discipline) but not the *entry-guard* code path. |
| L3 | ❌ | No CI integration test exercised "rehydrate → restart → assert no doubling." Behavioural mocking of NT Strategy is impractical (Cython properties are read-only). |
| L4 | ❌ | First post-deploy restart "looked fine" because the entry signal happened to be inactive. Bug only manifested when a bullish IHYG bar arrived. |
| L5 | ❌ | No alert on broker-vs-strategy position divergence. No alert on "strategy entered while already long." |
| L6 | ✅ | Caught during routine `docker logs titan-portfolio` review when investigating the cancelled VUSD/EIMI orders. |

**Already remediated** (this PR cycle):
- L1 / L2: AST guard test added — `str(position.side) == "LONG"` antipattern is now CI-blocked.
- The rehydration kwarg-discipline guard already in place since the May 11 fix.

**Tier 1 items 1–5 will move detection from L6 → L4 / L5 for the next bug of this class.**

## 6. Implementation status

| Tier | Item | Status | PR |
|---|---|---|---|
| T1.5 | NT API contract tests | pending | — |
| T1.4 | Order-rejection page (audit + global hook) | pending | — |
| T1.1 | Position-reconciliation watchdog | pending | — |
| T1.3 | NLV divergence alarm | pending | — |
| T1.2 | Double-restart smoke script | pending | — |

Update this table as items ship.

## 7. Principles

A few invariants this framework rests on:

- **Every production incident yields a guard.** No "we just need to be more careful." The incident is evidence the discipline must be encoded in code, not memory.
- **Cheap layers first.** A 30-line AST test is worth more than a 300-line integration test if it covers the same bug class, because it runs on every commit.
- **Defence in depth.** No single layer is reliable. Today's bug bypassed L1-L5; tomorrow's will bypass L1-L3 but trip L4. Build all layers.
- **Make the right thing easy.** If "restart and verify positions unchanged" requires a manual checklist, it won't get done. Automate the verification, gate the deploy.
- **Bias toward small, ship-able PRs.** Each Tier 1 item is independently valuable and reviewable in isolation.

## References

- [Rehydration Bug 2026-05-11.md](Rehydration%20Bug%202026-05-11.md) — the post-mortem that motivated this framework
- [tests/test_position_rehydration.py](../tests/test_position_rehydration.py) — the AST guards already in place
- [tests/test_research_math_guardrails.py](../tests/test_research_math_guardrails.py) — pattern for AST-level invariants
- PR #5 — the bond_gold rehydration entry-guard fix
