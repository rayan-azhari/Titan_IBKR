# Strategy Re-validation — Track 1 deliverable

**Version:** 1.0 | **Date:** 2026-05-13 | **Author:** Architect / Risk-Auditor
**Status:** Snapshot. Not a deployment-action directive — a checkpoint on what is and isn't reproducible under the May-2026 audit guardrails (audit findings A1-A11 + V3.1-V3.6).

---

## 0. What this document is and isn't

Migrate.md proposed a two-track process: Track 1 = **mechanical re-derivation** of every "validated" strategy under the corrected pipeline; Track 2 = **honest Samir-Stack post-mortem**. This document is the Track 1 output: a table of every live or research-validated champion's *audit-reproducibility status*, plus the concrete next action each needs.

It is **not** a deployment plan and **not** a kill list. No `config/*.toml` parameters are changed by this directive; no strategy is retired in code. The table records what the audit-discipline gates say about each strategy. Decisions about reconfiguring or retiring follow separately, per strategy, in a follow-up review.

The audit dimensions checked per strategy:

| Dim | Description | Audit ref |
|---|---|---|
| A1/A2 | Same-bar look-ahead: `ret * pos` where `pos[t]` uses `close[t]` | A1, A2 |
| A3 | Cost-model input semantics declared (TR vs price-only) | A3 |
| A4 | WFO honesty (per-fold parameter selection OR pre-registration) | A4 |
| A5 | DSR adjustment at sweep N > 5 | A5 |
| A6 | Tail-risk MC bootstraps underlying returns, not strategy returns | A6 |
| A7 | Position-scaling overlays change the cost-aware engine | A7 |
| A8 | "Validated against X" has a checked-in artifact | A8 |
| A9 | Strategy Guide matches deployed config | A9 |
| A10 | Parity test uses independent reference + full chain + causality | A10 |
| A11 | Tier-scaled sizing for leveraged-ETF / native-leverage instruments | A11 |
| V3.1 | Selection rule pre-registered before sweep ran | V3.1 |
| V3.2 | Plateau-seeking selection (neighbours + IC-range gate) | V3.2 |
| V3.6 | Negative results documented | V3.6 |
| ShM | Sharpe via `titan.research.metrics.sharpe` with explicit `periods_per_year` | (April quant audit) |
| Boot | Bootstrap CI emitted with CI_lo > 0 gate | (April quant audit) |
| Sanc | 12-month sanctuary held out | (V3.1 / Migrate.md §3.6) |
| PT | Live parity test exists | A10 |

Legend: ✓ = present and verified, ✗ = absent, ⚠ = present but suspect, n/a = not applicable.

---

## 1. Audit table

### 1.1 Mr AUD/JPY (`mr_audjpy`, `MRAUDJPYStrategy`)

**Claim:** +0.97 / CI_lo +0.47 OOS Sharpe at `vwap_anchor=24`, H1.

**Audit row:**

| A1/A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | V3.1 | V3.2 | ShM | Boot | Sanc | PT |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | n/a | ✓ | ✗ | ✗ | n/a | ✓ | ✓ | ✓ | n/a | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |

**Re-derived under audit pipeline?** Partial — the backtest math itself is corrected (shared metrics, bootstrap CI, shift discipline). The **signal layer** has now been re-validated under the IC Census ([directives/IC AUDJPY Vwap Fine-Grid 2026-05-13.md](./IC%20AUDJPY%20Vwap%20Fine-Grid%202026-05-13.md)) and **failed plateau stability** — IC at h=1 is monotonically decaying in anchor across `{3, 4, 6, 8, 12}`, with no interior peak. The live `vwap_anchor=24` sits in the tail of the decay where IC = -0.009 / t = -2.71, below the parent census's DSR floor.

**Deployment eligibility under audit-discipline:** the backtest Sharpe is real but the signal is not plateau-stable. The +0.97 Sharpe is consistent with a borderline edge held up by position-sizing discipline rather than a robust signal.

**Next action:** Two-part follow-up review (not actioned by this directive):

1. Re-run the full vectorbt-engine backtest at anchor=6 (the IC-peak candidate) and compare CI ranges to anchor=24. If anchor=6 produces a meaningfully tighter CI, propose a config change PR. If not, signal is retired as IC-justified.
2. Microstructure-aware reformulation (deviation in EWM-spread units) as a separate signal class. Out of scope of this directive.

---

### 1.2 Bond-Equity (`BondGoldStrategy` re-use, IHYU/CSPX, HYG/IWB)

**Claim:** +0.68 OOS Sharpe revalidated (25 folds), IHYG→VUSD +1.16 / CI_lo +0.47, IHYG→EMIM +0.97 / CI_lo +0.23.

**Pipeline:** `research/cross_asset/run_bond_equity_wfo.py`, `run_bond_equity_wfo_realistic.py`.

**Audit row:**

| A1/A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | V3.1 | V3.2 | ShM | Boot | Sanc | PT |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | n/a | ✓ | ⚠ | ✗ | n/a | ✓ | ✓ | ✓ | n/a | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |

**Verified in this audit:**
- Shared `titan.research.metrics.sharpe` with `periods_per_year=BARS_PER_YEAR["D"]` — explicit annualisation (`run_bond_equity_wfo.py:188-200`).
- Inline code comment documents the prior bug: *"implementation `mean(nz)/std(nz) * sqrt(252)` overstated Sharpe by..."*. ✓ corrected.
- `bootstrap_sharpe_ci(all_oos, ..., n_resamples=1000, seed=42)` emits CI. ✓.

**Missing under full audit discipline:**
- **DSR adjustment** for selection across (signal × parameter × instrument-pair) cells. The IHYG→VUSD / IHYG→EMIM claims came from a multi-cell sweep without explicit DSR. The April-2026 strict-Bonferroni note (System Status §10.2) flags IHYG→EMIM as borderline.
- **No underlying-resampled MC.** Tail-risk MC bootstraps strategy returns rather than the HYG/IEF/equity underlyings with shared block indices (audit A6, 5-15× tail-risk understatement).
- **No explicit sanctuary window** in the runner. Last-12-months hold-out was not part of the pipeline that produced the headline Sharpes.

**Re-derived through IC Census today?** Yes, partial. The cross-asset IC scan on Phase A ran HYG/IEF spread (and DXY proxy) against equity / fixed-income / commodity targets at D timeframe (42 headline rows). Strongest cross-asset ICs:

| Cell | IC | t_NW | Notes |
|---|---|---|---|
| DBC h=21, dxy_z | -0.158 | -3.10 | USD↑ → commodities↓; largest IC magnitude in entire Phase A census |
| DBC h=5, hyg_ief_z | +0.084 | +3.12 | risk-on bond spread → commodities up |
| IHYG h=1, dxy_z | -0.051 | -3.04 | USD-translation effect on USD-denominated UCITS |
| CSPX h=1, hyg_ief_z | -0.035 | -2.07 | the **bond-equity mechanism itself**: HY widens → CSPX falls |

**None clear the parent census's |t_NW| > 4.5 floor.** The CSPX h=1 hyg_ief_z signal that motivates the bond-equity champions has the right sign and is BH-significant on its own raw p-value (p < 0.05), but its t-stat of 2.07 sits well below the DSR-corrected floor.

**Deployment eligibility under audit-discipline:** The math of the existing backtests is correct (corrected `sqrt(252)` issue, bootstrap CI, shift discipline). What's missing is (a) DSR for the sweep N, (b) underlying-resampled MC, (c) sanctuary held out. The IC at the signal level is mechanistically real but t-stat-thin given the sample size (~3.7k daily bars).

**Next action:**
1. Extend `run_bond_equity_wfo.py` with DSR (reuse `deflated_sharpe_prob` from samir_stack).
2. Replace strategy-return bootstrap with underlying-resampled MC (audit A6 pattern).
3. Carve out the 12-month sanctuary and hold IHYU→CSPX, IHYG→VUSD, IHYG→EMIM out of any retraining on that window.
4. Re-emit the Sharpe + DSR-prob + sanctuary-IC numbers. If headline Sharpe survives DSR + sanctuary, keep deploying. If it doesn't, decommission per strategy.

---

### 1.3 LSTM stacking (`research/ml/multi_horizon_lstm.py`, `ensemble_stacking.py`)

**Claim:** +1.37 (QQQ), +1.18 (SPY) OOS Sharpe via LSTM stacker.

**Audit row:**

| A1/A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | V3.1 | V3.2 | ShM | Boot | Sanc | PT |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ⚠ | n/a | ✓ | ✗ | ✗ | n/a | ⚠ | n/a | n/a | n/a | ✗ | ✗ | ⚠ | ✗ | ✗ | ✗ |

**Verified:**
- WFO is per-fold (training within fold, OOS evaluation on next slice). ✓ A4.

**Suspect / missing:**
- **A1/A2 same-bar look-ahead risk.** In `run_52signal_classifier.py:609-613`, `positions = pd.Series(predictions, index=bar_returns.index); strategy_rets = positions * bar_returns - cost_per_bar`. This is causal **only if** `predictions[t]` is computed using features through `t-1` (and `bar_returns[t]` is the `t → t+1` return). The pipeline must verify this. The risk is non-trivial because the feature engineering in `lstm_features.py` and `quantile_features.py` is high-dimensional and a single feature using `.shift(0)` of close would flip causality.
- **A5 DSR absent.** Tier A came from selection across 28 instruments × multiple LSTM architectures × hyperparameter grids. The April-2026 LSTM Phase 0B-0E feature-stuffing failures are a precedent for the multi-test exposure being large enough to inflate the "best" cell.
- **A6 no underlying-resampled MC.**
- **Bootstrap CI ⚠** — `_stitch` in `run_phase1_validation.py:430` computes `stitched.mean() / std * np.sqrt(bars_yr)` directly. Not filter-then-annualised (no `rets != 0`), so the formula is correct. But it does NOT emit a CI; the headline Sharpe has no bootstrap interval.
- **Sanctuary ✗** — no last-12-months hold-out in the validation runners.
- **A10 parity test absent.** No `tests/test_lstm_stacking_live_parity.py` (LSTM stacker is not currently deployed live, but a parity test would still verify the research-side math matches a hypothetical live implementation).

**Deployment eligibility under audit-discipline:** **suspect**. The claimed Sharpes are higher than anything Phase A's IC census produced under proper discipline, which on its own is a red flag given Migrate.md's repeated finding that audit-grade gates collapse formerly-promising numbers to 0.5-1.0 Sharpe range.

**Next action:**
1. **A1/A2 causality audit** of `run_52signal_classifier.py`, `run_phase1_validation.py`, `multi_horizon_lstm.py`. For each, locate the feature computation, verify every feature is `.shift(1)`-ed or uses only `close.shift(k)` with `k ≥ 1`. If any feature touches `close[t]` and the strategy then earns `bar_returns[t]`, the Sharpe is overstated by an unknown but probably-large factor.
2. **DSR adjustment.** Compute N = (instruments × architectures × hyperparams × folds) actually swept, apply `deflated_sharpe_prob` at that N. The +1.37 Tier-A Sharpe needs to clear DSR-prob ≥ 0.95.
3. **Bootstrap CI.** Add `bootstrap_sharpe_ci` to `_stitch` output.
4. **Sanctuary.** Carve out the last 12 months and never train on it. Sanctuary IC + Sharpe is a separate pass.
5. **Until 1-4 are done, treat the LSTM-stacker headline numbers as not deployment-eligible.**

---

### 1.4 ML Tier A (`research/ml/run_52signal_classifier.py`)

**Claim:** +1.58 (EUR/USD D), +1.11 (QQQ D) OOS Sharpe.

**Audit row:** identical to LSTM stacking row above. Same pipeline, same gaps. The `compute_signal_sharpe` function at `run_52signal_classifier.py:602` is the canonical example of the same-bar look-ahead risk pattern.

**Next action:** same as LSTM stacking, applied to this strategy's specific feature pipeline (52-signal classifier features in `lstm_features.py` + `quantile_features.py`).

---

### 1.5 ORB 7-instrument (`research/orb/run_orb_full_optimizer.py`, `run_orb_oos_*.py`)

**Claim:** "validated" — no specific Sharpe number in System Status.

**Audit row:**

| A1/A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | V3.1 | V3.2 | ShM | Boot | Sanc | PT |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | n/a | ✓ | ✗ | ✗ | n/a | ⚠ | ✓ | ✓ | n/a | ✗ | ✗ | ⚠ | ✗ | ✗ | ✓ |

**Verified:**
- Uses `vbt.Portfolio` for backtesting — vectorbt's Sharpe computation is internally correct (uses bar-frequency returns, no zero-filter).
- IS / OOS split mechanically enforced via `is_oos: bool = False` flag (`run_orb_oos_validation.py:87`).
- Parity test exists in `tests/` (per the strategy roster).

**Suspect / missing:**
- **A5 DSR absent.** Selection across 7 instruments was done by ranking, not by DSR. The "7 of 482 screened" finding (System Status §2.x) on `run_warrior_screener.py` is the classic high-N selection bias setup. With 482 instruments and selecting the top 7, the expected null-hypothesis max Sharpe is sqrt(2 ln 482) ≈ 3.5 — meaning even random series would produce a top-7 set with apparent Sharpes around that magnitude.
- **A6 no underlying-resampled MC** for the 5-minute bar bootstrap.
- **Bootstrap CI ✗** — `vbt.Portfolio.stats` reports point-estimate Sharpe but the runners do not call `bootstrap_sharpe_ci`.
- **Sanctuary ✗.**
- **ShM ⚠** — vbt computes Sharpe internally; if `freq='5min'` is set correctly, the annualisation factor is right. If it's set to `'1D'` on M5 data, annualisation is biased up by sqrt(78). Worth a code grep before trusting headline numbers.
- **A8 "validated" claim has no specific artifact pointing to the methodology audit.** The +X Sharpe vs reproducible-output test pair is undocumented in System Status.

**Deployment eligibility:** Until DSR is applied to the 7-of-482 selection, the headline outcome is suspect for the same reason Migrate.md flagged.

**Next action:**
1. Inspect `vbt.Portfolio` freq setting in every ORB runner; verify M5 → annualised correctly.
2. Apply DSR at N = 482 (the screener pool). If the top-7 instruments don't clear DSR-prob ≥ 0.95, the strategy's apparent edge is selection bias.
3. Add bootstrap CI on the per-instrument and combined-portfolio OOS series.
4. Define and hold out a 12-month sanctuary.
5. Re-emit the headline numbers with DSR-prob + CI_lo + sanctuary-Sharpe.

---

## 2. Cross-cutting findings

Three patterns emerge from this audit:

1. **The Sharpe-math layer is mostly fixed.** Every runner I examined uses `titan.research.metrics.sharpe` with explicit `periods_per_year`, or computes the equivalent `mean / std * sqrt(periods_per_year)` directly without the `rets != 0` filter. The April-2026 filter-then-annualise bug is gone from the codebase. ✓
2. **The selection-bias layer is universally missing.** Zero of the five strategies above apply DSR to their sweep N. Zero apply underlying-resampled MC. Zero hold out an explicit sanctuary window. These are not bugs in the headline-Sharpe formula but **systemic gaps in how we account for multi-test exposure**. Every "validated" Sharpe in the System Status roster was selected from some sweep; without DSR at the actual N, every one of them is point-estimate-inflated by an unknown but non-zero amount.
3. **Bootstrap CI coverage is patchy.** Only `bond_equity` and the recently-added IC census emit `bootstrap_sharpe_ci` consistently. ML/LSTM/ORB report point Sharpes with no CI, so we cannot say whether their CI_lo is positive.

The combined Track 1 verdict, treating each strategy as a row in a deployment-eligibility table:

| Strategy | Sharpe math | Selection bias | Sanctuary | Verdict |
|---|---|---|---|---|
| mr_audjpy | ✓ | ✗ (signal failed Census plateau) | ✗ | Backtest math OK; signal-layer not robust per IC Census. Conditional. |
| Bond-equity (IHYU→CSPX, IHYG→VUSD/EMIM) | ✓ | ✗ | ✗ | Math OK; DSR + sanctuary + MC needed. Conditional. |
| LSTM stacking (QQQ, SPY) | ⚠ | ✗ | ✗ | Same-bar causality unverified; DSR absent; CI absent. **Suspect.** |
| ML Tier A (EUR/USD, QQQ) | ⚠ | ✗ | ✗ | Same as LSTM stacking. **Suspect.** |
| ORB 7-instrument | ⚠ (vbt freq) | ✗ (482→7 selection bias) | ✗ | Vbt freq verification needed; DSR at N=482 critical. **Suspect.** |

> [!IMPORTANT]
> "Conditional" means: the headline Sharpe is computed correctly given the data the strategy was fitted on, but the audit hasn't yet shown the number survives DSR + sanctuary + MC. "Suspect" means: at least one of the foundational audit dimensions is uncertain (typically same-bar look-ahead in feature engineering, or large unaudited selection-N).

---

## 3. What this document does NOT propose

- **No live config changes.** This is research output, not a deployment action.
- **No strategy retirement.** Even "suspect" strategies stay deployed pending the follow-up actions in §1.3 / §1.4 / §1.5.
- **No re-running of the IC Census or extending its scope.** Phase A scan is complete. Phase B (futures, VIX, breadth) is the right next-scan, not a rerun.
- **No new pre-registrations in this document.** Each follow-up action (DSR adjustment per strategy, sanctuary hold-out per strategy) implies its own small pre-registration when actioned.

---

## 4. Forward sequencing

Recommended order, smallest blast radius first:

1. **Bond-equity DSR + sanctuary** (lowest risk, math is already correct, just adding gates). One PR per the `run_bond_equity_wfo.py` extension; runs in <5 min.
2. **ML same-bar causality audit** (highest information yield). One PR that adds a `tests/test_ml_causality.py` asserting every feature factory in `lstm_features.py` and `quantile_features.py` uses only `.shift(k)` for `k ≥ 1`. If the test fails, the headline Sharpes need re-computation — that is the largest potential reversal in the System Status §2 roster.
3. **ORB DSR at N=482** (focused, mechanical). Apply `deflated_sharpe_prob` to the 482-screener output. Likely outcome: most of the 7 deployed instruments will fail, leaving a smaller robust set.
4. **Phase B data download + extended IC Census** (independent of 1-3, can run in parallel).

Each follow-up gets its own one-page directive with pre-registered gates. None of them retroactively reopens this snapshot.

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-13 | Initial Track 1 snapshot. Audit table across 5 strategy families. |
