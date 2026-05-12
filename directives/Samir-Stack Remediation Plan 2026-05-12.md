# Samir-Stack Remediation Plan

**Created:** 2026-05-12
**Status:** Approved — operator sign-off on 2026-05-12 with three decisions (see §0).
**Source audit:** External independent audit completed 2026-05-12.
**Target end state:** A single deployable 40/60 Samir-Stack variant with corrected
math, optional bond rotation, optional futures engine, validated under a proper
WFO + DSR + sanctuary gate, and documented consistently across the Strategy
Guide, live registry, and paper-validation runbook.

---

## 0. Locked decisions (2026-05-12)

The plan below reflects these three operator decisions:

1. **Drop I3 (opt-in EFA overlay)** — same look-ahead class as I2, marginal
   uplift post-fix, removes an extra instrument and FX leg from the
   strategy. Old I3 function stays in the codebase with a `# DEPRECATED`
   marker for history.
2. **GBP rotation candidate set** — Phase 5 sweep tests
   `RotationBondSleeve([IGLT, IGLS])` (UK gilts long + short duration +
   cash fallback). USD `[IEF, HYG]` rotation is **not evaluated** in this
   cycle — it would add FX vol to the previously FX-clean bond ballast
   for the UK paper account.
3. **Selection rule: Calmar CI lower bound with parsimony tie-break.**
   Among cells that survive all hard gates (Sharpe CI lo > 0, DSR > 0
   after N=240, 2022 cum return ≥ -10%, sanctuary Sharpe ≥ 0), pick the
   one with the highest bootstrap Calmar lower bound. Ties broken by
   lower L_max, then lower equity_weight.

---

## 1. Audit findings this plan must answer to

| ID | Finding | Severity | Plan step |
|---|---|---|---|
| A1 | `bond_rotation_returns` uses today's close for both signal and return | 🔴 | Phase 1 |
| A2 | I3 opt-in EFA mask uses today's close for both signal and return | 🔴 | Phase 1 (drop) |
| A3 | `futures_returns` double-counts dividends (input is TR-adjusted, +4.75pp/yr bias at L=3) | 🔴 | Phase 1 |
| A4 | WFO has no per-fold parameter selection — stability check, not OOS | 🟠 | Phase 5 |
| A5 | Multiple-testing burden uncorrected across thousands of sweep cells | 🟠 | Phase 5 (DSR gate) |
| A6 | Bootstrap MC of strategy returns destroys conditional structure | 🟠 | Phase 5 (underlying-resampled MC) |
| A7 | Vol-target backtest doesn't model real-world cost of scaling leverage | 🟠 | Phase 3-4 cost decomposition |
| A8 | `validate_against_3usl()` was never actually run | 🟡 | Phase 2 |
| A9 | Strategy Guide describes 40/60; live registry deploys 10/90 — divergent | 🟠 | Phase 6 |
| A10 | Parity tests miss research-side bugs | 🟡 | Phase 1 + Phase 6 |

Reproduced numbers from the audit (these are the targets to confirm post-fix):

| Strategy variant | Claimed | As-coded reproduction | Bias-corrected |
|---|---|---|---|
| 10/90 + vol_target=0.08 + I1+I2+I3 + MES futures | 20.03% CAGR / 2.28 Sharpe / -6.62% MaxDD | 18.78% / 2.15 / -9.79% | **4.97% / 0.63 / -23.06%** |
| 40/60 + capitulation + synthetic 3x ETF + static IEF | 9.13% / 0.83 / -27.11% | **9.13% / 0.82 / -27.11%** ✓ | unchanged ✓ |

The 40/60 baseline reproduces honestly; the 10/90 champion does not.

---

## 2. Design decisions (Architect)

### 2.1 The 40/60 architecture stays. The sleeves get componentised.

Refactor `run_stacked_strategy` to take an `EquityEngine` and a `BondSleeve`
instead of hard-coded `spy_close` and `ief_close`. Both engines produce a
daily return series on $1 of sleeve equity. The state machine, regime gate,
DD breaker, capitulation overlay, and re-entry quiet period all remain
**unchanged** — they're sleeve-agnostic.

```
equity_sleeve: EquityEngine    # synthetic_3x | margin_constant_L | futures_constant_L (fixed)
bond_sleeve:   BondSleeve      # static(IEF) | static(IGLT) | rotation([IGLT, IGLS])
```

### 2.2 Rotation lives at the sleeve level and carries its own lag.

`BondSleeve.rotation_returns(t)` returns `r_t = winner_decided_at_t-1 ×
asset_return_t`. The shift is baked into the sleeve. The state machine
never has to know.

### 2.3 Futures engine takes TR-adjusted input — explicitly typed.

Add `futures_returns_tr(underlying_tr_close, leverage, ...)` with corrected
math:

```
daily_ret_per_$equity = L * (spy_TR_ret - div_yield/252)   # strip dividend from TR
                       + funding/252                        # T-bill on cash
                       - L * roll/252                       # roll slippage
```

Old `futures_returns` keeps existing math + `DeprecationWarning`.
At L=1 the new function should give ~SPY TR + (funding - div - roll), not
the +1.46pp/yr bias.

### 2.4 I3 (opt-in EFA) is dropped from the 40/60 path.

Decision per §0(1).

### 2.5 Optimisation objective is named upfront.

```
maximise:    bootstrap_lower_bound(Calmar)
subject to:  bootstrap_Sharpe_CI_lo > 0
             P(MaxDD>50%) < 1% on underlying-resampled MC
             Net of cost
             >= 100 effective trades over WFO OOS
             Multiple-testing-adjusted (Deflated Sharpe Ratio) above zero
parsimony:   lower L_max preferred, lower equity_weight preferred (tie-break)
```

### 2.6 Bond-sleeve currency / FX is preserved.

GBP-clean candidate set only this cycle. USD rotation deferred to a
separate USD-base-account evaluation.

### 2.7 Capitulation overlay parameters frozen.

Not re-tuned in this remediation. Current defaults stay. Re-tuning is a
separate experiment after Phase 7.

---

## 3. Phases (with gates)

### Phase 1 — Fix the bugs, prove the fixes (1-2 days)

**Closes A1, A2, A3, A10.** Branch: `fix/samir-stack-research-bugs`.

| # | Action |
|---|---|
| 1.1 | Fix `bond_rotation_returns`: `winner = winner.shift(1)` before applying returns |
| 1.2 | Add `futures_returns_tr` with corrected TR-input math |
| 1.3 | Deprecate old `futures_returns` with warning; update all callers |
| 1.4 | Drop I3 wiring from new pipelines (keep file with `# DEPRECATED` comment) |
| 1.5 | Add `tests/test_samir_stack_research_lookahead.py` with bond-rotation and futures-L1 tests |
| 1.6 | Regenerate `.tmp/reports/samir_stack/*` with corrected math |

**Phase 1 gate:**
- New lookahead tests pass.
- 10/90 vol-target=0.08 + futures_fixed re-run gives ~5% CAGR / 0.6 Sharpe (matches audit's bias-corrected number).
- 40/60 + cap baseline re-run still gives ~9% CAGR / 0.82 Sharpe (proves honest path unbroken).
- Full pytest suite green.

### Phase 2 — Component refactor + 3USL validation (1 day)

**Closes A8, sets up later phases.**

| # | Action |
|---|---|
| 2.1 | Introduce `EquityEngine` ABC + 3 concrete engines |
| 2.2 | Introduce `BondSleeve` ABC + `StaticBondSleeve` + `RotationBondSleeve` |
| 2.3 | Refactor `run_stacked_strategy` to use engine + sleeve |
| 2.4 | Actually run `validate_against_3usl()` with downloaded 3USL data |

**Phase 2 gate:** refactor reproduces existing 40/60 + cap baseline
bit-exactly. 3USL validation report saved with synth-vs-actual CAGR /
MaxDD / daily-return correlation.

### Phase 3 — Bond rotation on 40/60, isolated test (1 day)

**Closes A1 properly.**

Head-to-head: `SyntheticETFEngine(L=3)` × `{StaticBondSleeve(IEF),
StaticBondSleeve(IGLT), RotationBondSleeve([IGLT, IGLS])}` × `equity_weight=0.40`.

Anchored WFO (504 IS / 252 OOS / 252 step), bootstrap-Sharpe CI lo, sanctuary
holdout. Report rotation churn / flip win-rate / 2022 cum return.

**Phase 3 gate:**
- Rotation variant has `ci95_lo > 0`.
- 2022 cum return better than static-IEF baseline (rotation hypothesis verified).
- Churn ≤ 4 flips/year.
- Sanctuary OOS Sharpe ≥ 0.

If rotation fails the gate, **the bond sleeve stays static** in Phase 5.

**Phase 3 OUTCOME (2026-05-13):** REJECTED. G1, G2, G4 passed but G3
(churn) failed at 22.94 flips/year — more than 5× the 4/year ceiling.
Lookback sensitivity {30d, 60d, 90d, 120d, 180d, 252d} confirms the
failure is structural (IGLT/IGLS are too correlated for momentum
rotation to be operationally tractable). Bond sleeve stays static in
Phase 5; cell count reduced 180 → 120. Full analysis in
`.tmp/reports/samir_stack/phase3_bond_rotation_report.md`.

### Phase 4 — Futures engine on 40/60, isolated test (1 day)

**Closes A3 properly.**

Head-to-head: `{SyntheticETFEngine, MarginEngine, FuturesEngine}` × `L=3` ×
`StaticBondSleeve(IEF)` × `equity_weight=0.40`.

Cost decomposition per engine (vol drag, funding, TER, roll, margin spread).
Integer-contract sizing modelled honestly for futures at small NAV.

**Phase 4 gate:**
- Futures engine reproduces SPY TR at L=1 within +0.5pp/yr (Phase 1 fix held).
- Futures vs synthetic_3x ETF CAGR diff < 3pp/yr at L=3 (not free money).
- Chunky sizing produces *lower* CAGR than fractional at NAV ≤ £30k.

### Phase 5 — Joint split + L_max + engine sweep (2-3 days)

**Closes A4, A5, A6, A7. Only runs after Phases 1-4 are green.**

| Factor | Levels |
|---|---|
| `equity_weight` | 0.20, 0.30, 0.40, 0.50, 0.60 |
| `L_max` | 2, 3, 4 |
| `equity_engine` | synthetic_3x, futures_fixed |
| `bond_sleeve` | static_IEF, static_IGLT |
| `capitulation` | on, off |

Total: **5 × 3 × 2 × 2 × 2 = 120 cells** (reduced from 180 after
Phase 3 rejected the IGLT/IGLS rotation on the churn gate — see
`.tmp/reports/samir_stack/phase3_bond_rotation_report.md`).

Each cell evaluated with:
- Anchored WFO (504 IS / 252 OOS / 252 step), bootstrap-Sharpe CI (n=2000)
- Underlying-resampled MC (block bootstrap raw SPY/IEF/HYG/VIX, n=500 in sweep, 2000 for survivor)
- Crisis decomposition (GFC, COVID, 2022)
- Net of costs per Phase 4 decomposition
- Deflated Sharpe Ratio adjusted for N=180

**Selection rule** (committed *before* the sweep runs):

```
1. eliminate: sharpe_ci95_lo <= 0
2. eliminate: dsr_prob < 0.95
3. eliminate: 2022_cum_return < -10%
4. eliminate: sanctuary_sharpe < 0 OR sanctuary_max_dd > 15%
5. rank survivors by: calmar_ci95_lo (descending)
6. tie-break: lower L_max, then lower equity_weight
```

**Phase 5 gate:** at least one cell survives all five eliminators. If
zero survive, ship the existing 40/60 baseline + cap with the bug fixes
and stop optimising — that is itself a valid finding ("no joint config
has confirmed multiple-testing-adjusted edge").

### Phase 6 — Reconcile docs and live config (0.5 day)

**Closes A9.**

| # | Action |
|---|---|
| 6.1 | Rewrite `directives/Samir-Stack Strategy Guide.md` to describe the Phase 5 survivor (delete or archive 10/90 vol-target description) |
| 6.2 | Update `config/samir_stack.toml` to match survivor exactly |
| 6.3 | Update `titan/strategies/samir_stack/strategy.py` defaults + docstring |
| 6.4 | Update `tests/test_samir_stack_strategy.py` champion-pinning assertions |
| 6.5 | Add research-vs-live parity test (one-bar fixture asserts same decision) |

### Phase 7 — Paper validation re-baseline (4 weeks calendar)

Existing [Samir-Stack Paper Validation 2026-05-12.md](Samir-Stack%20Paper%20Validation%202026-05-12.md)
runbook stands, but expected-CAGR / expected-Sharpe / D5-alert thresholds
reset to the Phase 5 survivor's metrics. Do not promote on the old 20%-CAGR
anchor — that anchor is invalid.

---

## 4. Risk callouts

> [!WARNING]
> Phase 5 sweep results **must not** go into production without re-running
> the Phase 6 live-parity test. The Phase 2 refactor changes the function
> signature of `run_stacked_strategy`. The live strategy class will need
> to be updated to call the new abstractions.

> [!WARNING]
> The decision to use GBP-clean rotation (§0(2)) means the USD-rotation
> variant is **not evaluated in this cycle**. If a USD-base account is
> ever considered for Samir-Stack, that evaluation is a separate
> experiment — do not retroactively assume USD rotation has the same
> properties as GBP rotation.

> [!CAUTION]
> Futures sizing at small NAV is structurally lossy. The chunky-sizing
> reality check in Phase 4 is non-negotiable. If the futures variant
> wins Phase 5 but the operator's actual NAV is below ~£100k, deploy
> the synthetic_3x ETF variant even if it loses by a thin margin.

> [!IMPORTANT]
> The Phase 5 selection rule is **pre-committed**. Once the 180-cell
> results land, do not relax the gates to admit a favourite cell. That
> is the exact failure mode that produced the original 10/90 champion.

---

## 5. Not in scope for this cycle

- Re-tuning capitulation overlay parameters (separate experiment, post-Phase 7).
- Re-validating the 7-indicator regime score or the HMM (audit confirmed both are causal and honest).
- I1 rate-shock indicator (kept as optional ensemble member; not made required).
- Automating MES quarterly rolls (manual rolls until Phase 7 promotion).
- Migrating to a USD base account (separate operational decision).

---

## 6. Calendar estimate

| Phase | Engineer-days | Calendar | Blocking |
|---|---|---|---|
| 1. Fix bugs + lookahead tests | 1-2 | 2 days | yes |
| 2. Component refactor + 3USL validation | 1 | 1 day | yes |
| 3. Bond rotation isolated test | 1 | 1 day | yes |
| 4. Futures engine isolated test | 1 | 1 day | yes |
| 5. Joint 180-cell sweep | 2-3 | 3 days | yes |
| 6. Reconcile docs + parity test | 0.5 | 0.5 day | yes |
| 7. Paper revalidation | 0 (calendar only) | 4 weeks | yes |

**Total to deployable: ~2-3 weeks engineering + 4 weeks paper = ~6-7 weeks.**

---

## 7. Phase 1 kickoff commands

```bash
git checkout -b fix/samir-stack-research-bugs

# (implement Phase 1.1-1.5 — see §3 Phase 1 table)

uv run pytest tests/test_samir_stack_research_lookahead.py -v
uv run ruff check . --fix && uv run ruff format .
uv run pytest tests/ -q

uv run python research/samir_stack/run_pipeline.py
uv run python research/samir_stack/run_overlay_sweep.py
uv run python research/samir_stack/run_futures_sweep.py
uv run python research/samir_stack/run_risk_of_ruin.py

git diff -- .tmp/reports/samir_stack/    # confirm corrected numbers
```

---

## 8. References

- Original audit (this remediation responds to): conversation transcript 2026-05-12.
- Existing Strategy Guide: [Samir-Stack Strategy Guide.md](Samir-Stack%20Strategy%20Guide.md)
- Paper-validation runbook: [Samir-Stack Paper Validation 2026-05-12.md](Samir-Stack%20Paper%20Validation%202026-05-12.md)
- Research-math guardrails: `references/research-math-guardrails.md`
- Portfolio-risk architecture: `references/portfolio-risk-architecture.md`
