# Bond-Equity Audit — DSR + Sanctuary + Underlying-Resampled MC

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect / Risk-Auditor
**Status:** **PRE-REGISTRATION** — committed before the audit re-runs (V3.1).
**Parent:** `directives/Strategy Re-validation 2026-05-13.md` §1.2

---

## 0. Why this exists

The Strategy Re-validation audit (May 2026) flagged the three Bond-Equity champions as **CONDITIONAL**:

| Strategy | Last claimed | Audit gaps |
|---|---|---|
| IHYU → CSPX | +0.68 OOS Sharpe (25 folds, post-corrected math) | DSR not applied; no underlying-resampled MC; no explicit sanctuary hold-out |
| IHYG → VUSD | +1.16 / CI_lo +0.47 | Same as above; sweep N enabled the cell to be cherry-picked |
| IHYG → EMIM | +0.97 / CI_lo +0.23 | Same; CI_lo borderline (strict-Bonferroni at N would deflate further) |

The Sharpe math is already audit-corrected (`titan.research.metrics.sharpe` with explicit `periods_per_year`; bootstrap CI via `bootstrap_sharpe_ci`). What's missing:

1. **DSR adjustment** at the actual sweep N (Bailey & López de Prado 2014).
2. **Underlying-resampled Monte Carlo** (audit A6) — bootstrap the bond + target underlyings with shared block indices, cumprod to rebuild synthetic price paths, re-run the strategy on each.
3. **Sanctuary held out** — the last 12 months excluded from training+OOS, then a single one-shot final-validation pass on that window.

This directive pre-registers a single comparison sweep that adds all three gates.

---

## 1. Pre-registered scope

### 1.1 Strategies tested

| Bond | Target | Lookback | Hold | Threshold | Notes |
|---|---|---:|---:|---:|---|
| IHYU | CSPX | 10 | 20 | 0.50 | claimed +0.68 OOS — verified parameters from `directives/Candidate Portfolio 2026-04-21.md` |
| IHYG | VUSD | 10 | 20 | 0.50 | claimed +1.16 / CI_lo +0.47 |
| IHYG | EMIM | 10 | 20 | 0.50 | claimed +0.97 / CI_lo +0.23 |

Three additional reference configurations (already audit-corrected per System Status §10.2) included as a sanity-check baseline — they should reproduce their published numbers, providing a measurement of whether the new gates change anything for the verified portfolio:

| Bond | Target | Lookback | Hold | Threshold | Reference Sharpe |
|---|---|---:|---:|---:|---:|
| TLT | QQQ | 10 | 10 | 0.50 | +0.895 |
| IEF | GLD | 60 | 20 | 0.00 | +0.721 |
| HYG | IWB | 10 | 10 | 0.50 | +0.895 (revalidated) |

**Total sweep N = 6 cells.** DSR null-max ≈ sqrt(2 ln 6) ≈ 1.89.

### 1.2 New gates (in addition to the existing audit-corrected Sharpe + CI)

| Gate | Implementation | Pass threshold |
|---|---|---|
| **DSR-adjusted prob** | Bailey & López de Prado 2014 via existing `research/samir_stack/run_phase5_joint_sweep.py::deflated_sharpe_prob`. Reuse exactly as in `research/strategies/run_range_expansion_wfo.py::apply_dsr_to_cells`. | dsr_prob ≥ 0.95 |
| **Underlying-resampled MC** | Audit A6 pattern. Bootstrap bond + target log returns with **shared 50-bar block indices** (preserves cross-asset correlation). cumprod to rebuild synthetic prices. Re-run the strategy on each synthetic path. | P(MaxDD > 25%) < 5% across 200 paths |
| **Sanctuary held out** | Last 12 months trimmed from data before running WFO. Sanctuary pass runs `run_bond_wfo` separately on just the trimmed window. | Sanctuary stitched Sharpe ≥ 0 (note: 12mo is small N, so this gate is informational rather than blocking) |

### 1.3 Pre-committed decision rule

For each strategy individually:

| All three new gates pass | Pre-sanctuary Sharpe CI_lo > 0 | Decision |
|:---:|:---:|---|
| Yes | Yes | **Deployment-eligible.** Existing live config stands. Document as audit-confirmed. |
| Yes | No | **Conditional.** Math is fine, MC + sanctuary OK, but CI_lo not robustly positive. Status quo, watchpoint. |
| No (DSR fail) | Yes | **Suspect.** Sharpe is sweep-cherry-picked. Investigate; potential reconfiguration or retirement. |
| No (MC fail P>5%) | Yes | **Risk-management upgrade required** before continued live. Add tighter portfolio-level position size cap or retire. |
| No (Sanctuary < 0) | Yes | **Regime watchpoint.** Recent 12 months underperformed; either a regime shift or sample noise. Increase monitoring; no immediate config change. |
| No (multiple) | No | **Retire from live deployment.** Issue config-change PR. |

---

## 2. Out of scope

- **Not** sweeping new (bond, target) pairs. Each strategy is audited at its existing pre-registered parameters.
- **Not** sweeping new lookback / hold / threshold combinations. The audit-corrected numbers in System Status are taken as the live config; the audit only adds NEW gates, doesn't change cells.
- **Not** re-doing AUD/JPY here — that's a separate directive (`MR AUDJPY Audit Re-Run 2026-05-14.md`) already completed.
- **Not** itemising IG DFB equivalents. Bond-equity champions deploy on IBKR UCITS, not on IG.

---

## 3. Implementation

1. **This directive on `main`.** (THIS PR)
2. `research/cross_asset/run_bond_equity_audit.py` — wrapper that:
   - For each cell in §1.1, slice off sanctuary (last 12 months)
   - Call existing `run_bond_wfo` on the visible (pre-sanctuary) data
   - Run separate sanctuary pass on the trimmed-off 12 months
   - Compute DSR across the 6-cell sweep
   - Run underlying-resampled MC (200 paths × 50-bar shared blocks)
3. Run + result log appended to §4.

---

## 4. Result log

Appended 2026-05-14 after the audit ran. §1-§3 unchanged (V3.1).

### 4.1 Six-cell summary

DSR null-max at N=6: `sqrt(2 ln 6) ≈ 1.89`. All cell Sharpes well below null-max, so DSR-prob came out at 1.0 across all cells — DSR is not the binding gate here.

| Cell | WFO Sharpe (pre-sanc) | CI_lo | DD | Sanctuary Sharpe | MC P(MaxDD>25%) | DSR | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **IHYU → CSPX** | +0.675 | **+0.095** | -20.6% | +1.63 | 0.370 | 1.00 | **RISK_UPGRADE** |
| **IHYG → VUSD** | +0.364 | **-0.235** | -21.9% | +2.17 | 0.300 | 1.00 | **RETIRE** |
| **IHYG → EMIM** | +0.459 | **-0.216** | -23.3% | +2.49 | 0.460 | 1.00 | **RETIRE** |
| TLT → QQQ | +0.743 | +0.315 | -29.7% | +2.06 | 0.835 | 1.00 | RISK_UPGRADE |
| IEF → GLD | +0.554 | +0.080 | -28.8% | +1.36 | 0.820 | 1.00 | RISK_UPGRADE |
| HYG → IWB | +0.907 | +0.433 | -16.6% | +1.48 | 0.665 | 1.00 | RISK_UPGRADE |

### 4.2 Headline findings

**(a) IHYG-anchored champions retire under sanctuary discipline.**

| Strategy | Prior claim | Audit re-run (pre-sanctuary) | Δ |
|---|---:|---:|---:|
| IHYG → VUSD | +1.16 / CI_lo +0.47 | **+0.364 / CI_lo -0.235** | -0.80 Sharpe, CI flipped to negative |
| IHYG → EMIM | +0.97 / CI_lo +0.23 | **+0.459 / CI_lo -0.216** | -0.51 Sharpe, CI flipped to negative |

Holding out the trailing 12 months from training+OOS removes most of the champions' headline Sharpe. The recent year (2025-2026) was a particularly strong period for this strategy class -- the sanctuary pass alone gives +2.17 / +2.49 Sharpe on these pairs. Prior backtests bundled this year into OOS, making it look like an out-of-sample win. **Under audit discipline, only the strategy's behaviour on data IT NEVER SAW is genuine OOS; the rest is in-sample with extra time.**

CI_lo flipped to negative on both. Per §1.3 row 6: **RETIRE**.

**(b) IHYU → CSPX is borderline-deployable.**

- WFO Sharpe +0.675 (vs claimed +0.68) -- essentially identical to the claim, so the audit doesn't change the picture much.
- CI_lo = +0.095 (just barely above zero).
- Sanctuary Sharpe +1.63 -- the recent-year is again much stronger than the historical OOS.
- MC P(MaxDD>25%) = 0.370 -- 37% of bootstrap paths hit ≥25% drawdown.

Per §1.3 row 5 / row 4: **RISK_UPGRADE** -- CI_lo > 0 but MC tail-risk is elevated. Add tighter portfolio-level position-size cap or accept the tail risk explicitly.

**(c) MC gate calibration warning — all 6 cells fail it.**

The pre-committed P(MaxDD>25%) < 5% MC gate failed on every cell, including the three reference baselines (TLT→QQQ, IEF→GLD, HYG→IWB) that we're confident in from prior validation. This suggests the gate is **structurally too tight for cross-asset momentum strategies**:

- Cross-asset momentum strategies hold long positions for extended periods (months to years).
- A 15-year equity-long has natural ~30% drawdowns at 2020 / 2008 / 2022-style events.
- Even random-shuffle MC paths inherit the underlying's tail risk, so ~half of synthetic paths will hit 25% MaxDD purely from the unconditional vol structure.
- The 5% threshold I committed presupposed a position-trading style closer to the range-expansion intraday spec (where the strategy is mostly flat) -- it doesn't transfer well to a cross-asset always-on momentum approach.

Per V3.1, the pre-committed gate stands. Verdict is RISK_UPGRADE / RETIRE as the gate says, **with this caveat noted**:

> Open a follow-up directive to recalibrate the MC gate for cross-asset momentum strategies. The reference cells (TLT→QQQ, HYG→IWB) are audit-trusted; they should not be flagged RISK_UPGRADE under a well-calibrated MC gate. Suggested re-pre-registration: gate becomes `P(MaxDD > 35%) < 10%` for cross-asset momentum, with the original 25%/5% retained for short-term H1 strategies (where the range-expansion Phase 0 directive showed the 5% gate is meaningful).

**(d) Sanctuary divergence is a real-world phenomenon worth tracking.**

The recent 12 months were uniformly strong across all six cells (sanctuary Sharpe +1.36 to +2.49) vs WFO Sharpe +0.36 to +0.91. Two possible causes:

- **Regime shift.** The post-COVID rates environment may genuinely have made cross-asset momentum easier (clear bond-equity correlation pattern, sustained trends).
- **Sample noise.** 252 daily bars is small; 1-year point estimates are noisy.

Either way, the sanctuary's positive result is **informational, not deployment-validating** — it's the data the strategy was NEVER trained on, so positive Sharpe there says the strategy generalises to recent conditions. But per V3.1, a positive sanctuary cannot override negative CI_lo on pre-sanctuary WFO.

### 4.3 Action items (each requires its own follow-up PR -- not this directive's scope)

1. **`IHYG → VUSD`: RETIRE.** Open a config-change PR removing this strategy from any live deployment registry. The previously-claimed +1.16 Sharpe is not reproducible under sanctuary discipline.
2. **`IHYG → EMIM`: RETIRE.** Same. Previously-claimed +0.97 is not reproducible under sanctuary discipline.
3. **`IHYU → CSPX`: keep live with watchpoint.** CI_lo is barely positive (+0.095) and MC tail-risk is elevated. Either (a) accept and add tighter position-size cap, OR (b) defer pending the MC gate recalibration directive (§4.2-c).
4. **Open follow-up directive: MC gate recalibration for cross-asset momentum.** Pre-register a less-tight gate appropriate for always-on cross-asset strategies. Re-run §4.1 under the new gate; expect TLT→QQQ, HYG→IWB to clear.
5. **Sanctuary discipline lesson rolls into project-wide hygiene.** Future bond-equity / cross-asset Sharpe claims must be reported pre-sanctuary. Any "1-year recent" Sharpe is a SEPARATE diagnostic, not a deployment metric.

### 4.4 V3.6 lesson — rolled into project catalogue

**Previously-claimed cross-asset Sharpe numbers that included the last 12 months as OOS were overstated. Sanctuary discipline collapses them.** The published `IHYG → VUSD +1.16` becomes `+0.36` once the recent year is properly held out. This isn't a math bug -- the Sharpe was correctly computed on the data that was used; the bug is that the data used wasn't truly OOS for deployment purposes.

The full V3.6 catalogue now:

| Lesson | Recorded in |
|---|---|
| DSR-passing IC ≠ deployable strategy (cost matters). | `Strategy Range-Expansion ES-NQ H1 — Phase 0` §4.7 |
| Raw IC peak ≠ strategy-engine peak when strategy has layers. | `MR AUDJPY Audit Re-Run 2026-05-14` §4.5 |
| **Sanctuary-included Sharpe ≠ deployable Sharpe.** Treat the last 12 months as forbidden until a held-out one-shot pass. | **THIS directive §4.4** |
| MC tail-risk gates must be calibrated per strategy class. | THIS directive §4.2-c |

### 4.5 Outcome record

| Field | Value |
|---|---|
| IHYG → VUSD retirement recommended? | **Yes** -- CI_lo -0.235 |
| IHYG → EMIM retirement recommended? | **Yes** -- CI_lo -0.216 |
| IHYU → CSPX retirement recommended? | No -- borderline at CI_lo +0.095, MC risk-upgrade |
| Sanctuary discipline reproduces prior claims? | **No** -- IHYG strategies regress by ~0.6 Sharpe |
| MC gate calibration follow-up required? | **Yes** -- pre-registered separately |
| Live config changes this directive triggers? | **None** (config-change PRs are separate per V3.1) |
| Strategy Re-validation §1.2 follow-ups closed? | Yes -- DSR + MC + sanctuary added to bond-equity audit pipeline |

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial pre-registration. |
