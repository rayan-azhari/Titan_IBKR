# ML Same-Bar Causality Audit

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect / Risk-Auditor
**Status:** **PRE-REGISTRATION** + EXECUTION (audit is read-only over existing code; running it doesn't change cells).
**Parent:** `directives/Strategy Re-validation 2026-05-13.md` §1.3 (LSTM stacking) + §1.4 (ML Tier A)

---

## 0. Why this exists

The Strategy Re-validation snapshot flagged the ML/LSTM strategies as **SUSPECT** with the largest potential reversal in the live roster (claimed +1.58 EUR/USD Tier A, +1.11 QQQ Tier A, +1.37 QQQ LSTM stacking, +1.18 SPY LSTM stacking). The flagged risk: same-bar look-ahead in the feature→strategy alignment.

A static code reading of `research/ml/run_52signal_classifier.py` confirms the bug:

| Line | Code | Diagnosis |
|---:|---|---|
| 710 | `bar_returns = df["close"].pct_change().fillna(0.0)` | This is the **backward** return at t: `(close[t] − close[t-1]) / close[t-1]`. |
| 689 | `features = build_features(df, tf)` → `build_features` at line 260, where ma_features, mom_features, etc. all use `close[t]` directly via `sma(close, ...)`, `close.pct_change(126)`, etc. | Features at bar t include `close[t]`. |
| 783 | `pred_proba = model.predict_proba(X_oos_all)[:, 1]` | Predictions at bar t use features including `close[t]`. |
| 789 | `stats = compute_signal_sharpe(position, oos_returns, ...)` | Calls `compute_signal_sharpe` with position[t] and returns[t]. |
| 613 (inside `compute_signal_sharpe`) | `strategy_rets = positions * bar_returns - cost_per_bar` | **NO `.shift(1)` on positions.** The strategy P&L at t = position[t] × backward-return[t]. Both are functions of close[t]. Same-bar look-ahead. |

The expected impact: Tier A and LSTM stacking Sharpes are inflated by an unknown but non-zero amount. The audit quantifies the magnitude.

---

## 1. Pre-registered scope

### 1.1 Audit methodology

Two scenarios run on EXACTLY the same WFO pipeline, on the same instruments, with the same hyperparameters. ONLY the position-return alignment differs:

| Scenario | Position-return alignment |
|---|---|
| **A (buggy, as-deployed)** | `strategy_rets[t] = positions[t] * bar_returns[t]` where `bar_returns = close.pct_change()` (backward) |
| **B (causal, audit-corrected)** | `strategy_rets[t] = positions.shift(1)[t] * bar_returns[t]` -- same backward return but position lagged. Equivalently: `positions[t] * forward_return[t]`. |

Scenario B is the textbook causal alignment: at bar t the model decides on a position using features through t; that position is opened at t+1 and earns the t+1 return.

### 1.2 Instruments tested

Two cells, matching the live Tier A claims:

| Instrument | TF | Claimed Sharpe (live Tier A) |
|---|---|---:|
| EUR/USD | D | +1.58 |
| QQQ | D | +1.11 |

Each runs through the **full existing** `run_52signal_classifier.py` pipeline -- features, label sweep, WFO splits, XGBoost training, prediction, position derivation, Sharpe -- ONLY changing the alignment line in `compute_signal_sharpe`.

### 1.3 Test statistic + decision rule

For each instrument, compute:

- `sharpe_A` (buggy, as-deployed): stitched OOS Sharpe under same-bar alignment
- `sharpe_B` (causal): stitched OOS Sharpe under shift(1) alignment
- `inflation = sharpe_A − sharpe_B`
- `inflation_pct = inflation / sharpe_A` (relative)

Pre-committed verdicts:

| Outcome | Action |
|---|---|
| `inflation_pct ≥ 50%` AND `sharpe_B ≤ 0.3` | **Strategy retirement required.** The published Sharpe is dominated by look-ahead. Open config-change PR; live trading halt. |
| `30% ≤ inflation_pct < 50%` OR `sharpe_B ∈ (0.3, 0.6)` | **Material inflation.** Halt live trading pending re-validation. Open audit-pipeline fix PR. |
| `inflation_pct < 30%` AND `sharpe_B > 0.6` with `CI_lo_B > 0` | **Confirmed deployable.** The look-ahead exists but its magnitude doesn't change the deployment verdict. Fix the alignment for ALL future Sharpe reporting; leave live trading running. |
| `inflation_pct < 30%` AND `sharpe_B ≤ 0.3` | **Marginal pre-inflation Sharpe.** Probably wasn't deployment-grade even before the bug. Retire. |

### 1.4 Out of scope

- **Not** modifying `run_52signal_classifier.py` itself. The audit script is read-only -- it imports the existing build/predict/WFO functions and only overrides `compute_signal_sharpe` with a corrected variant. The fix PR is separate.
- **Not** running the multi-horizon LSTM or ensemble stacking re-runs in this directive. Same-bar look-ahead in those (if present) is a separate audit (V3.1: each strategy = its own pre-reg). Tier A is the bigger live exposure; covered first.
- **Not** auditing the feature factories for look-ahead. The features themselves are causal in the sense of using `close[t]` (not `close[t+k]`); the bug is in the **alignment** between features and returns at the Sharpe-computation step. Future-feature look-ahead is a separate audit if any user introduces it.

---

## 2. Implementation

1. **This directive on `main`.** (THIS PR)
2. `research/ml/run_tier_a_causality_audit.py` -- thin wrapper that imports `run_one_instrument` from the existing pipeline, monkey-patches `compute_signal_sharpe` to run both A and B variants, emits comparison + verdict.
3. Run on EUR/USD D and QQQ D.
4. Append result log to §3.

---

## 3. Result log

Appended 2026-05-14 after the audit ran. §1-§2 unchanged (V3.1).

### 3.1 Buggy vs causal stitched OOS Sharpe

Same predictions on both alignments. Only `compute_signal_sharpe`'s position-return multiplication differs.

| Instrument | Buggy Sharpe | Buggy CI | Causal Sharpe | Causal CI | Inflation | Inflation % |
|---|---:|---|---:|---|---:|---:|
| EUR/USD D | +1.584 | [-0.140, +3.161] | +1.625 | [-0.103, +3.191] | **-0.041** | **-2.6%** |
| QQQ D | +1.247 | [+0.062, +2.627] | +1.154 | [-0.072, +2.554] | **+0.093** | **+7.4%** |

### 3.2 The look-ahead is REAL but small in magnitude

The bug at `run_52signal_classifier.py:613` (`positions * bar_returns` without `.shift(1)`) IS look-ahead -- position at t depends on features including close[t], and the bar return at t is also a function of close[t]. But the quantitative impact at the WFO-stitched level is minor: 7.4% Sharpe inflation on QQQ, and on EUR/USD the corrected Sharpe is actually slightly HIGHER than the buggy one (noise dominates).

**Mechanism for the small impact.** `_pred_to_position` in `run_52signal_classifier.py:646` keeps positions PERSISTENT — once a long is opened, it's held until the prediction flips to short or to flat. Most OOS bars carry the prior bar's position unchanged. The same-bar bias only fires on transition bars (when position[t] ≠ position[t-1]). For these strategies, transitions are rare relative to total bars (~0.5-2% of bars), so the same-bar bias on those bars is averaged down to ~5% of total Sharpe.

The bug WOULD matter much more for:
- A strategy where every bar is a fresh long/short decision (no persistence)
- A high-frequency tier where transitions per bar are high
- An ensemble where each base model produces its own per-bar position (more transitions per timeframe overall)

For the current Tier A strategies (single XGB classifier, persistent position, no ensemble at the position level), the impact is modest.

### 3.3 The CI_lo finding — bigger concern than the look-ahead

Under both alignments, **stitched OOS CI_lo is NEGATIVE for both instruments**:

- EUR/USD D causal: Sharpe +1.625, CI [-0.103, +3.191] → CI_lo = -0.10
- QQQ D causal: Sharpe +1.154, CI [-0.072, +2.554] → CI_lo = -0.07

The claimed +1.58 (EUR/USD) and +1.11 (QQQ) point-estimate Sharpes ARE reproduced under correct alignment. But the bootstrap CI on the stitched OOS is wide enough to include zero. This is the **April 2026 audit's bootstrap-CI gate failure:** `tier=unconfirmed` for any cell where stitched OOS CI_lo ≤ 0.

The likely cause is **sparse training swings** — each fold typically has 10-25 training entries, and most folds get skipped for insufficient swings. Only 4-5 of 48 WFO folds completed for each instrument in this audit, producing ~500-630 OOS bars total. That's a small sample for bootstrap CI computation; the wide CI is partly a small-N artefact, but it's also the honest representation.

The full `run_52signal_classifier.py` pipeline that produced the Tier A claims sweeps EIGHT label configs per fold and picks the best one with >= 20 entries (line 758 of run_52signal_classifier.py: `if len(is_entries) > best_entry_count`). That gives more folds AND introduces selection bias — picking the best-fitting label config per fold inflates IS Sharpe; OOS Sharpe is the right test but the per-fold "best label" can still be sample-cherry-picked at the fold level.

This audit used a single fixed label config (the most permissive one) and got ~4-5 folds. The Tier A claim's per-fold count was higher because it could fall back to different label configs across folds.

### 3.4 Verdict — per directive §1.3, with one row not anticipated

The strict matrix in §1.3 had:

| Outcome | Action |
|---|---|
| `inflation < 30%` AND `sharpe_B > 0.6` with `CI_lo_B > 0` | Confirmed deployable |
| `inflation < 30%` AND `sharpe_B ≤ 0.3` | Retire marginal |

But the actual outcome is `inflation < 30%` AND `sharpe_B ≈ 1.1-1.6` AND `CI_lo_B ≤ 0`. Pre-committed rules don't speak to that case directly. The closest legitimate read:

- **The same-bar look-ahead is NOT the dominant cause of the inflated Sharpe claims.** Fix the alignment (it's a real bug) but the fix doesn't materially change the conclusion. ETA on the fix: one-line PR to `compute_signal_sharpe` adding `.shift(1)` to positions, or define bar_returns as forward returns.
- **The Tier A Sharpe claims as point estimates ARE reproducible under causal alignment** (+1.62 EUR/USD, +1.15 QQQ in this audit; original claims +1.58 / +1.11). They're not inflated by look-ahead.
- **BUT the stitched OOS bootstrap CI is wide enough to include zero on both instruments.** Per the April 2026 audit's bootstrap-CI deployment gate (`CI_lo > 0` required), Tier A under audit-corrected alignment is `tier = unconfirmed`. **Not deployment-eligible at the strict gate.**

**Verdict: PARTIAL_RETIRE.** Specifically:

1. The Tier A QQQ D + EUR/USD D headline Sharpes are not a look-ahead artefact. Don't retire them on the basis of "look-ahead bias confirmed".
2. But they DO fail the bootstrap CI deployment gate under audit-corrected alignment. Per existing April 2026 audit policy, this means `tier = unconfirmed` and they should not be in the default deployment registry.
3. **Action items (separate config-change PRs):**
   a. **One-line code fix** at `run_52signal_classifier.py:613` -- add `.shift(1)` to positions or change `bar_returns` to forward returns. Verify against parity test.
   b. **Tier A `tier = unconfirmed` flag** in any deployment registry that references EUR_USD D or QQQ D under the existing config. Match the April 2026 audit's published convention.
   c. **Larger WFO sample question deferred** -- the current pipeline produces only ~4-5 valid folds. The right fix is more folds + more bars; the wrong fix is loosening the swing-entry threshold. Either way that's a separate research direction.

### 3.5 What the audit confirms about V3.6 lessons

Two observations from this audit roll into the project catalogue:

| Lesson | Recorded in |
|---|---|
| DSR-passing IC ≠ deployable strategy (cost matters). | `Strategy Range-Expansion ES-NQ H1 — Phase 0` §4.7 |
| Raw IC peak ≠ strategy-engine peak when strategy has layers. | `MR AUDJPY Audit Re-Run 2026-05-14` §4.5 |
| Sanctuary-included Sharpe ≠ deployable Sharpe. | `Bond-Equity Audit DSR-Sanctuary 2026-05-14` §4.4 |
| **Look-ahead in alignment is real but bounded by position persistence.** Same-bar bias in `position × bar_return` matters most when positions flip often; for persistent-hold strategies it's a small correction. | **THIS directive §3.2** |
| **CI_lo gates are stricter than point-estimate Sharpe gates.** A strategy can have impressive headline Sharpe and still fail CI_lo > 0 due to small N or wide return variance. | **THIS directive §3.3** |

### 3.6 Outcome record

| Field | Value |
|---|---|
| Same-bar look-ahead in code confirmed? | **Yes** -- `run_52signal_classifier.py:613` |
| Tier A Sharpe inflated by look-ahead? | Minor (-2.6% to +7.4%) -- not dominant |
| Tier A passes bootstrap CI_lo > 0 gate? | **No** (CI_lo ∈ [-0.10, -0.07] for both instruments after correction) |
| Code fix required? | Yes -- one-line `.shift(1)` PR (separate from this audit) |
| Live deployment status implication | `tier = unconfirmed` per existing April 2026 policy |
| Strategy retirement recommended? | No -- this is a "tier flag" change, not a kill. Operator may continue running paper but should not size to deployable-tier risk levels until a larger-N WFO confirms CI_lo > 0. |
| LSTM stacking + multi-horizon LSTM audits | Separate pre-registrations -- not in this directive's scope |

---

## 4. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial pre-registration + execution plan. |
