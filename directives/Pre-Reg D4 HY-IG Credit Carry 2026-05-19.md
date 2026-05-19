# Pre-Registration — D4: HY/IG Credit Carry (Israel-Palhares-Richardson 2018)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-19
**Branch:** v2-main → research/d4-credit-carry
**Type:** Strategy audit (class `CARRY` — credit-class variant)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE any HY/IG spread, carry P&L, or per-cell statistic has been computed on our HYG / LQD daily parquets. The Israel-Palhares-Richardson finding + HY/IG ETF prices are public knowledge; no cell-level Sharpe, no event-conditional return, no HYG-LQD ratio statistic has been inspected for OUR specific sample slice.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` row D4 — HY/IG credit carry (2d, IBKR-implementable, `CARRY/PAIRS`).

**Source.** Israel, R., Palhares, D., & Richardson, S. A. (2018). *"Common factors in corporate bond returns."* Journal of Investment Management (JoIM), 16(2), 17–46. (AQR Working Paper variant: SSRN 2576784.) Extends Fama-French factor logic to corporate credit; finds that the credit-carry factor — going long HY (high-yield) corporate bonds and short IG (investment-grade) — earns significant risk-adjusted returns over 1988-2014, with the bulk of the return coming from the carry differential rather than alpha against equity factors.

**Key claim.** A monthly-rebalanced long-HY / short-IG portfolio (or a daily-rebalanced ETF proxy via HYG long + LQD short) earns positive risk-adjusted returns over the full period. The carry differential (HY yield ~5-9% vs IG yield ~3-5% historically) provides the structural edge. Fat-tail risk concentrates in credit-stress events (2008 GFC, 2011 Eurozone, 2020 COVID).

**Hypothesis (falsifiable).** A long-HYG / short-LQD position, vol-targeted and (in some cells) gated by a trend filter on the HYG/LQD ratio, earns positive risk-adjusted returns over 2008-2026 visible window with:
- per-bar annualised Sharpe `CI_lo > 0`
- L52 plateau spread `≤ 50%` (H1) across the 3×3 cell grid
- DSR `dsr.dsr_prob ≥ 0.95`
- L65 single-strategy ruin PASS at proposed deployment weight ≤ 10%
- L65 joint ruin with current LIVE strategies PASS
- L67 10-metric portfolio matrix either maintains 7/10 with ≥ +0.05 Sharpe lift OR clears 8/10

**Specific RETIRE triggers per §3.7:**
- Plateau spread > 50% → RETIRE (L52)
- DSR p < 0.95 → RETIRE (selection bias dominates)
- canonical CI_lo ≤ 0 → RETIRE (no edge after costs)
- L65 single FAIL at 5% weight → RETIRE
- L65 joint FAIL at the smallest proposed mix → RETIRE
- L67 portfolio Sharpe lift < 0 AND any currently-passing metric breaks → CONDITIONAL_WATCHPOINT (paper-only diversifier verdict, no live cutover)

**L74 carry-premium accounting.** Unlike fx_carry's pure-price-vs-swap-premium issue, the HYG/LQD ETFs are total-return ETFs: their `close` reflects reinvested dividends (yield → price). So the carry premium IS already in the per-bar log return when we use `Adj Close`. **Pre-flight check (in §6 implementation step 2): verify our parquets use total-return-adjusted closes.** If they use unadjusted closes, the audit understates carry by ~3-6% per year and a L74-style sensitivity sweep is required. If adjusted (expected for yfinance default), no sensitivity needed.

**Mechanism.**

1. **Universe.** Two ETFs:
   - HYG (iShares iBoxx $ High Yield Corporate Bond ETF, ARCA, launched 2007-04-04)
   - LQD (iShares iBoxx $ Investment Grade Corporate Bond ETF, ARCA, launched 2002-07-22)
2. **Bars.** Daily close from `data/HYG_D.parquet` and `data/LQD_D.parquet`.
3. **Signal & position rule (per cell):**
   - Compute log returns `r_HY_t = log(HYG_t / HYG_{t-1})` and `r_IG_t = log(LQD_t / LQD_{t-1})`.
   - Compute the realised vol of the net carry `(r_HY - r_IG)` over a rolling EWMA span (default 20 days).
   - Position size `scale_t = vol_target / realised_vol_t`, clipped at upper=1.5.
   - If `sma_filter_on`: position only engages when `HYG_t / LQD_t > SMA_period(HYG/LQD)_{t-1}` (trend filter on the ratio).
   - Net per-bar return: `held_t = scale_{t-1} × signal_{t-1}` (causal `.shift(1)`).
     `gross_t = held_t × (r_HY_t - r_IG_t)`
     `cost_t = |Δposition_t| × cost_bps_per_leg × 2 / 10000` (two legs)
     `net_t = gross_t - cost_t`
4. **Cost model.** `cost_bps_per_leg = 1.0` (conservative for HYG/LQD at retail IBKR — wider bid-ask than SPY but still institutional liquid). With 2 legs per turnover: 2 bps per round-trip half-cycle.
5. **Causality.** Position effective from `t+1` based on signal+vol computed using data through `t`. L21 smoke: corrupting future HYG/LQD prices must not change past trade outputs.

**Why this is novel for our stack.** No credit-class carry exists in `titan/strategies/`. The closest pattern is `fx_carry` (AUD/JPY long-yen-carry, paper-only CONDITIONAL). D4 introduces:
- First credit-class strategy.
- Genuinely non-equity sleeve (HYG correlates ~0.7 with SPY but the carry differential is a different signal).
- Tests whether the V3.7 hybrid generalises from FX carry (where the swap rate isn't in spot) to credit carry (where the yield IS in the total-return ETF price).

---

## §2. Cell sweep (V3.7 hybrid mandatory per L75)

3×3 grid pre-committed BEFORE any HY/IG carry P&L is inspected:

| dim | values |
|---|---|
| `sma_period` for HYG/LQD ratio trend filter | {0 (off), 50, 100} |
| `vol_target` annualised | {0.06, 0.08, 0.10} |

Total: **9 cells.** Canonical (Israel-Palhares-Richardson default + Carver carry-strategy style): `(sma_period=100, vol_target=0.08)`. The unconditional carry (`sma_period=0`) is the strict IPR pattern; the SMA-filtered variants test whether trend-conditioning improves drawdown without killing the carry.

**Plateau set:** canonical + its 4 immediate neighbours (one step in each dim). Plateau spread = (max − min) / |mean|. L52 gates:
- H1: spread ≤ 50%
- Strict: spread ≤ 30%

**Per-cell metrics:**
- Annualised Sharpe (explicit `periods_per_year=252`, per L60).
- Bootstrap CI 95% (n_resamples=2000, seed=42).
- Max drawdown.
- Per-bar time-in-market fraction (informational — verifies L74 doesn't apply since total-return ETFs already include carry).

**DSR.** Bailey & Lopez de Prado 2014 deflation across the 9 cells using `sr_var_from_sweep`. Canonical cell's `dsr.dsr_prob ≥ 0.95` required.

---

## §3. Gates & decision rule

The audit is the V3.7 hybrid + L65 + L67 closure in one script, per L75.

### §3.1 Pre-flight L52 plateau

Plateau spread on canonical + 4 neighbours. H1 ≤ 50%; strict ≤ 30%.

### §3.2 5-axis decision matrix

| Axis | Gate | Source |
|---|---|---|
| 1. CI_lo | annualised Sharpe CI_lo > 0 | bootstrap_sharpe_ci |
| 2. DSR | dsr_prob ≥ 0.95 over 9 cells | deflated_sharpe |
| 3. MC abs DD | P(MaxDD > 35%) < 10% | run_block_mc on HYG with LQD as extra_series |
| 4. Sanctuary divergence | visible vs sanctuary Sharpe within ±0.3 | sanctuary_divergence_test |
| 5. Noise robustness | mean degradation < 30% at σ=0.5; worst-case > 0 | run_noise_robustness |

### §3.3 Baseline (L66) and relative MC (L17)

**L66 baseline class** for `CARRY` credit variant: **cash (zero)**. The strategy is a long-short net-zero exposure that earns the carry differential; the "cash" baseline is the right benchmark for the strategy's own return distribution.

**L17 relative MC vs 60/40 SPY/IEF:** APPLIED. Bootstrap blocks size 63, 200 paths. The strategy is in-market continuously (or 40-60% if SMA-filtered), so the standard rel-MC interpretation applies (UNLIKE F3's sparse-trade quirk). Gate: median dd_reduction ≤ 0.80 AND p_strategy_better ≥ 0.50.

### §3.4 L65 single-strategy ruin

Weights: {0.02, 0.05, 0.10}. Gate per `assess_strategy_ruin`: P_kill_trip < 1% AND p95 DD at size > -25%.

### §3.5 L65 joint ruin

With current LIVE strategies (gem_j5 + turtle_cat_c3peak + ic_top3 basket). Candidate mixes:
- (G68/T17/IC15/D4=0): current LIVE reference
- (G66/T17/IC15/D4=2)
- (G64/T16/IC15/D4=5)
- (G62/T13/IC15/D4=10)

### §3.6 L67 10-metric portfolio inclusion

Vs 60/40 SPY/IEF. CURRENT (GEM68/T17/IC15) baseline. PROPOSED at §3.4-passing weights.

**Deploy gate**: ≥ 8/10 OR Sharpe lift ≥ +0.05 vs CURRENT with no metric regression. Matches ic_top3 deployment threshold (which lifted from 0.93 to 1.07 at IC15).

### §3.7 Decision tree (pre-committed)

```
plateau spread > 50%                          → RETIRE (L52)
DSR p < 0.95                                  → RETIRE (selection bias)
canonical CI_lo ≤ 0                           → RETIRE (no edge)
MC abs-DD gate FAIL                           → RETIRE
sanctuary divergence > 0.30                   → RETIRE (look-ahead suspected)
noise mean degradation > 30% at σ=0.5         → CONDITIONAL_WATCHPOINT (J3 axis)
L17 rel-MC FAIL                               → CONDITIONAL_WATCHPOINT (DD-reduction axis, mirrors samir_stack Phase 6c pattern)
L65 single FAIL at 5% weight                  → RETIRE
L65 joint FAIL                                → RETIRE
L67 < 7/10 OR Sharpe lift < 0                 → bench (CONDITIONAL_WATCHPOINT, paper-only)
L67 ≥ 7/10 AND Sharpe lift ≥ +0.05 (no regression) → DEPLOY-eligible
otherwise                                     → CONDITIONAL_WATCHPOINT
```

### §3.8 Sanctuary

12-month hold-out at the END of the visible window. Visible = 2008-01-01 (or HYG start) → 2025-04-30; sanctuary = 2025-05-01 → 2026-04-30. Sweep + plateau + DSR + L65 + L67 run on VISIBLE only; sanctuary divergence test then validates the canonical cell.

---

## §4. Result log

(To be appended after audit run. Empty at pre-reg commit.)

---

## §5. Known caveats (acknowledged at pre-reg)

1. **2008 GFC + 2020 COVID = fat-tail regime risk.** Both events are in the visible window. The strategy is structurally short credit-stress events. Expect MaxDD to concentrate around these periods. The MC abs-DD gate (§3.2 axis 3) is the primary defense; if the audit aborts on MC, it's not a methodology failure — it's the strategy showing its tail risk under shuffle.

2. **HY-IG correlation regime change.** Pre-2008, HY-IG spread was relatively stable. Post-2008, central-bank intervention has compressed spreads and changed the correlation structure. The audit window straddles both regimes; cell sensitivity to regime split is a real risk.

3. **Total-return ETF assumption.** HYG and LQD are total-return ETFs in principle, but our parquets might use unadjusted close. Pre-flight check in §6 step 2 verifies this. If unadjusted, L74-style sensitivity is required.

4. **No SHORT-leg cost asymmetry.** The strategy is long HYG / short LQD. We model symmetric cost on both legs (1 bp each = 2 bps round-trip per turnover). In practice, shorting LQD has a borrow cost (~0.5-1.0% annualised) which would degrade Sharpe by ~0.06-0.12 SR units. The audit DOES NOT MODEL borrow cost explicitly. Worst-case: a borderline-positive Sharpe in the audit may flip negative after borrow cost. Pre-committed sensitivity: if canonical Sharpe is < +0.30 after audit, flag the borrow-cost risk in the result log.

5. **L74 (carry-premium math) might not apply.** Unlike fx_carry's spot vs swap differential, HYG/LQD ETF prices include the carry via dividends. Verified in §6 step 2.

6. **HY-IG correlation ~0.85 with SPY/equity.** Adding this sleeve doesn't add as much diversification as fx_carry or i1v2. The L67 matrix is the right test — if it doesn't improve, the strategy is informational only.

7. **No new StrategyClass commit YET.** §1 mechanism uses `CARRY` defaults (same as fx_carry). If D4 promotes, no new class needed; just a new `D4` entry in `STRATEGY_REGISTRY`.

---

## §6. Implementation plan

1. **Pre-reg commit (THIS commit).** Freeze §1–§3.
2. **Data sanity check** (in audit script preamble): confirm HYG / LQD parquets use total-return (adjusted) close. Test: compute 1-year dividend yield from declared HYG dividends (~5-6%/year historical) vs realised price return — if the price return materially understates total return, the parquet uses unadjusted close and we need L74 sensitivity.
3. **Audit script** `research/exploration/audit_d4_credit_carry.py`: runs §2 sweep + §3 gates end-to-end. Outputs `.tmp/reports/d4_credit_carry/result_log.md`.
4. **Joint L65 + L67** (per L75 mandatory hybrid): integrated.
5. **Append §4 result log** with verdict.
6. **Update artifacts**: Retirement Registry (if RETIRE), v37_live registry (if DEPLOY), V3.6 catalogue meta, README V2.0 portfolio status table.

---

## §7. Discipline crosscheck (V3.6 / V3.7 lessons)

- **L06 (sparse-trade Sharpe):** N/A — D4 is continuous (or ~40-60% in-market under SMA filter), not sparse.
- **L11 (class defaults):** uses `CARRY` defaults (same as fx_carry).
- **L21 (causality smoke):** mandatory.
- **L52 (plateau pre-flight):** mandatory per L75.
- **L60 (annualisation):** `periods_per_year=252` explicit on every Sharpe / vol. NO hardcoded `sqrt(252)`.
- **L66 (baseline class declaration):** cash for standalone; 60/40 for L67 portfolio test.
- **L67 (portfolio inclusion):** 7/10 + lift ≥ 0.05 OR 8/10.
- **L72 (AUC ≠ Sharpe):** N/A — D4 is not a classifier.
- **L73 (cutover diff):** if DEPLOY, the L73 checklist applies for v37_live cutover.
- **L74 (carry-premium):** verified inapplicable IFF parquets are adjusted close (else L74 sensitivity required).
- **L75 (hybrid pre-flight mandatory):** §2 sweep + §3 plateau + DSR run BEFORE any deployment-eligible claim.
