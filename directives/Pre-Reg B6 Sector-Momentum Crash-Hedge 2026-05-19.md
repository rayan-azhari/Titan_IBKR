# Pre-Registration — B6: Sector-ETF Momentum with Crash-Hedge Overlay

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-19
**Branch:** v2-main → research/b6-sector-momentum
**Type:** Strategy audit (class `CROSS_ASSET_MOMENTUM` — sector-equity cross-sectional variant + META_LABELING overlay)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> **Inspiration disclosure (not replication).** The Daniel-Moskowitz (JFE 2016) "Momentum Crashes" paper documents the dynamic-scaling overlay on the CRSP universe (~3000 US stocks, 1927-2013). We do NOT have CRSP/Compustat access; A1 residual momentum already retired on yfinance current-S&P-500 universe due to survivorship bias (L36/L37). This audit tests the **design principle** of dynamic-scaling on a survivorship-clean ETF universe (sector ETFs are passive index funds — no constituent survivorship issue at the ETF level), NOT a replication of D-M's headline numbers. Pre-reg labels this explicitly. The hypothesis is about the OVERLAY's marginal contribution, not the underlying momentum signal's absolute magnitude.

> V3.1 pre-registration: §1–§3 frozen BEFORE any sector-ETF return statistic is computed on our (not-yet-downloaded) parquets. The yfinance sector-ETF tickers are public knowledge; no momentum rank, no crash-period return, no Sharpe number has been computed for OUR specific sample. Data acquisition is §6 step 2, AFTER the pre-reg freeze.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` row B6 — Momentum-crash hedge (Daniel-Moskowitz dynamic scaling), 2d, class META_LABELING.

**Source.** Daniel, K., & Moskowitz, T. J. (2016). *"Momentum crashes."* Journal of Financial Economics, 122(2), 221-247. Updated working paper: Daniel, Klos, Rottke (2017).

**Key claim.** Standard cross-sectional momentum strategies (long winners, short losers) suffer rare but catastrophic crashes when bear markets reverse. The crashes are conditional on PANIC STATE (recent bear market + elevated volatility): post-2008-March, post-1932-July. A dynamic scaling overlay that reduces momentum exposure in panic states can produce ~50% lower MaxDD with comparable Sharpe.

**Adapted hypothesis (falsifiable).** A long-short cross-sectional momentum strategy on US sector ETFs (`XL{F,K,E,U,Y,P,I,B,V}`, 1998-2026 visible window) earns positive risk-adjusted returns over the audit period. A panic-state overlay (recent market drawdown + elevated realised vol → scale down momentum exposure) measurably reduces drawdown vs the unscaled benchmark WITHOUT killing the Sharpe.

Concretely: comparing the canonical scaled cell to its `crash_scale=1.0` (overlay off) neighbour:
- Scaled cell MaxDD < 80% of unscaled cell MaxDD (overlay reduces tail).
- Scaled cell Sharpe ≥ 90% of unscaled cell Sharpe (overlay preserves edge).

**Specific RETIRE triggers per §3.7:**
- L52 plateau spread > 50% → RETIRE.
- DSR p < 0.95 → RETIRE.
- Canonical CI_lo ≤ 0 → RETIRE.
- L65 single FAIL at 5% weight → RETIRE.
- L65 joint FAIL at smallest proposed mix → RETIRE.

**CONDITIONAL_WATCHPOINT triggers:**
- Overlay reduces MaxDD by < 20% (overlay ineffective).
- Overlay degrades Sharpe by > 20% (overlay over-conservative).
- L67 lift < 0.05 (paper-only diversifier).

**Decay risk acknowledged (per §5).** Cross-sectional sector-equity momentum has been documented since Jegadeesh-Titman 1993. Hong-Stein (1999) and Pesaran-Timmermann (2004) reported attenuation post-publication. Our window (1998-2026) covers both pre- and post-decay regimes. The V3.7 hybrid is the right test — if the signal doesn't replicate at retail constraints on sector ETFs, the strategy retires the same way A1, F3, and D4 did.

**Mechanism.**

1. **Universe.** 9 SPDR Select Sector ETFs (XLRE excluded — launched 2015, too short for full-window audit):
   - XLF (Financials), XLK (Technology), XLE (Energy), XLU (Utilities), XLY (Consumer Discretionary), XLP (Consumer Staples), XLI (Industrials), XLB (Materials), XLV (Health Care)
2. **Bars.** Daily total-return-adjusted close (yfinance `auto_adjust=False` then prefer `Adj Close`; or `auto_adjust=True` to use adjusted price directly). Verified in §6 step 2 pre-flight (same protocol as D4 audit's L74 check).
3. **Signal (per ETF, per rebalance date t):**
   - 12-month skip-1 cumulative return: `mom_i,t = log(close_i,{t-21}) - log(close_i,{t-252})` (252-day momentum, 21-day skip to avoid 1-month reversal).
4. **Portfolio construction (cross-sectional):**
   - Rank ETFs by `mom_i,t` at end of each month.
   - **Long-leg**: top `N` ETFs, equal-weighted, weight per ETF = `+0.5 / N`.
   - **Short-leg**: bottom `N` ETFs, equal-weighted, weight per ETF = `-0.5 / N`.
   - Net portfolio gross exposure = 1.0 (long 0.5 + short 0.5, market-neutral if betas align).
5. **Crash-hedge overlay (causal):**
   - `bear_signal_t = 1` if SPY's trailing 24-month log return < `-0.10` (i.e., market down >10% over 2 years).
   - `vol_signal_t = 1` if SPY's trailing 21-day realised vol > 2× its 252-day median.
   - `panic_state_t = bear_signal_{t-1} AND vol_signal_{t-1}` (causal `.shift(1)`).
   - In panic state, scale momentum exposure by `crash_scale ∈ {0.0, 0.25, 0.50}` (cell parameter); otherwise full 1.0× exposure.
6. **Rebalance frequency.** Monthly (last business day).
7. **Costs.** 1 bp per leg per turnover. Net cost per rebalance ≈ (turnover_per_month) × 2 × 1bp.
8. **Causality.** All signals use data through `t-1`; positions effective from `t` onwards. L21 smoke: corrupting future ETF prices must not change past portfolio returns.

**Why this is novel for our stack.** No cross-sectional sector-rotation strategy exists in `titan/strategies/`. The closest patterns are:
- `gem_j5_canonical` (dual-momentum on SPY/EFA/IEF — 3-asset rotation, NOT cross-sectional with N>3)
- `etf_trend_*` (single-instrument MA crossovers, bulk-retired 2026-05-16 per L56)

B6 introduces:
- First TRUE cross-sectional ranked-momentum strategy (rank 9 ETFs, top/bottom N).
- First META_LABELING-class overlay (panic-state gate on a base momentum signal).
- Tests the design principle of dynamic-scaling crash hedge in a setting where we have data (not CRSP).

---

## §2. Cell sweep (V3.7 hybrid mandatory per L75)

3×3 grid pre-committed BEFORE any sector-ETF momentum statistic is inspected:

| dim | values |
|---|---|
| `top_n` (long-leg / short-leg count) | {2, 3, 4} |
| `crash_scale` (panic-state momentum multiplier) | {0.0 (overlay full off), 0.25 (overlay aggressive), 0.50 (overlay moderate)} |

Total: **9 cells.** Canonical: `(top_n=3, crash_scale=0.0)` — D-M's most aggressive crash hedge (zero exposure in panic state) with mid-range portfolio breadth.

**Plateau set:** canonical + its 4 immediate neighbours. Plateau spread gates:
- H1: spread ≤ 50%
- Strict: spread ≤ 30%

**Per-cell metrics:**
- Annualised Sharpe (`periods_per_year=252`, per L60).
- Bootstrap CI 95% (n_resamples=2000, seed=42).
- Max drawdown.
- Per-bar time-in-market fraction (the overlay should reduce this in panic states).

**DSR.** Bailey & Lopez de Prado 2014 deflation across 9 cells. Canonical's `dsr.dsr_prob ≥ 0.95`.

---

## §3. Gates & decision rule

The audit is the V3.7 hybrid + L65 + L67 closure in one script, per L75.

### §3.1 L52 plateau pre-flight

Canonical + 4 immediate neighbours. H1 ≤ 50%; strict ≤ 30%. Failure → RETIRE.

### §3.2 5-axis decision matrix

| Axis | Gate | Source |
|---|---|---|
| 1. CI_lo | annualised Sharpe CI_lo > 0 | bootstrap_sharpe_ci |
| 2. DSR | dsr_prob ≥ 0.95 over 9 cells | deflated_sharpe |
| 3. MC abs DD | P(MaxDD > 35%) < 10% | run_block_mc on SPY (representative; bootstrap blocks share across ETFs) |
| 4. Sanctuary divergence | visible vs sanctuary Sharpe within ±0.3 | sanctuary_divergence_test |
| 5. Noise robustness | mean degradation < 30% at σ=0.5; worst-case > 0 | run_noise_robustness |

### §3.3 Baseline (L66) and relative MC (L17)

**L66 baseline class** for `CROSS_ASSET_MOMENTUM` (sector-equity sub-class): **cash (zero)**. The strategy is long-short net-zero exposure; cash is the right baseline for the strategy's own distribution.

**L17 relative MC vs 60/40 SPY/IEF:** APPLIED. Bootstrap blocks size 63, 200 paths. Strategy is in-market continuously (~80-100% depending on panic-state frequency), so standard interpretation. Gate: median dd_reduction ≤ 0.80 AND p_strategy_better ≥ 0.50.

### §3.4 L65 single-strategy ruin

Weights: {0.02, 0.05, 0.10}. Standard gates per `assess_strategy_ruin`.

### §3.5 L65 joint ruin

With current LIVE strategies (gem_j5 + turtle_cat + ic_top3 basket). Candidates: {(G68/T17/IC15/B6=0), (G66/T17/IC15/B6=2), (G64/T16/IC15/B6=5), (G62/T13/IC15/B6=10)}.

### §3.6 L67 10-metric portfolio inclusion

Vs 60/40 SPY/IEF. CURRENT (GEM68/T17/IC15) baseline at 7/10. PROPOSED with B6 at §3.4-passing weights.

**Deploy gate**: ≥ 8/10 OR Sharpe lift ≥ +0.05 with no metric regression.

### §3.7 Decision tree (pre-committed)

```
plateau spread > 50%                              → RETIRE
DSR p < 0.95                                      → RETIRE
canonical CI_lo ≤ 0                               → RETIRE
MC abs-DD gate FAIL                               → RETIRE
sanctuary divergence > 0.30                       → RETIRE (look-ahead suspected)
noise mean degradation > 30% at σ=0.5             → CONDITIONAL_WATCHPOINT (J3 axis)
L17 rel-MC FAIL                                   → CONDITIONAL_WATCHPOINT (DD-reduction axis)
L65 single FAIL at 5% weight                      → RETIRE
L65 joint FAIL                                    → RETIRE
Overlay test: scaled cell MaxDD ≥ 80% unscaled    → CONDITIONAL (overlay ineffective; question the META_LABELING value)
Overlay test: scaled cell Sharpe < 90% unscaled   → CONDITIONAL (overlay over-conservative)
L67 < 7/10 OR Sharpe lift < 0                     → bench (paper-only)
L67 ≥ 7/10 AND Sharpe lift ≥ +0.05 (no regression) AND overlay tests PASS → DEPLOY-eligible
otherwise                                         → CONDITIONAL_WATCHPOINT
```

### §3.8 Sanctuary

12-month hold-out at the END of the visible window. Visible = sector-ETF launch (~1998 for the 9 we use) → 2025-04-30; sanctuary = 2025-05-01 → 2026-04-30.

---

## §4. Result log

(To be appended after audit run. Empty at pre-reg commit.)

---

## §5. Known caveats (acknowledged at pre-reg)

1. **Decay risk (the 5th instance pattern).** A1 residual momentum, ic_mtf, F3 FOMC, and D4 credit carry have all failed the V3.7 hybrid as "academic edge doesn't replicate post-2008 on retail data". B6 is at high risk of the same outcome: cross-sectional momentum decay is well-documented in the post-publication literature (Hong-Stein 1999, Pesaran-Timmermann 2004). If B6 retires, the meta-pattern is now firmly the 5th of its kind and the cumulative L36/L37/F3/D4/B6 quintet may warrant a synthesizing lesson (proposed L76).

2. **Sector momentum ≠ stock momentum.** D-M's results are on individual-stock cross-section. Sector ETFs have a smaller universe (9 vs 3000), less cross-sectional dispersion, more macro-driven moves. The DESIGN PRINCIPLE (dynamic scaling reduces tail) should still hold IF the underlying momentum has tail risk at all. If sector momentum doesn't crash (i.e., its baseline MaxDD is already small), the overlay has nothing to fix — the strategy fails on signal absence, not on overlay design.

3. **9 ETFs = small universe.** Top-3 of 9 = top-third; D-M's deciles are 10% × 3000 = 300 stocks per leg. Statistical noise per ETF leg is higher. Bootstrap CIs will be wider than D-M's.

4. **Survivorship-clean by construction.** Sector ETFs are passive index funds. The S&P sector definitions changed (e.g., XLC carved from XLK in 2018; XLRE carved from XLF in 2015). We exclude XLRE for window length; XLC's separation is captured in post-2018 XLK adjusted prices via the carve-out (treated as a corporate action — yfinance handles this for the original sector tickers).

5. **2008 + 2020 in window.** Both major crashes are in our visible data, which is the right test environment for a crash-hedge design.

6. **Total-return ETF check.** Like D4, sector ETF prices need to be total-return-adjusted for dividends. Verified in §6 step 2 pre-flight using the same heuristic (annualised price return > 2%).

7. **No new StrategyClass YET.** §1 mechanism uses `CROSS_ASSET_MOMENTUM` defaults provisionally. If B6 deploys, it joins GEM as the second strategy in that class.

---

## §6. Implementation plan

1. **Pre-reg commit (THIS commit).** Freeze §1–§3.
2. **Sector-ETF download** (`scripts/download_b6_sectors.py`): pull XLF/XLK/XLE/XLU/XLY/XLP/XLI/XLB/XLV daily from yfinance with `auto_adjust=True` (total-return). Save to `data/{TICKER}_D.parquet`. Pre-flight L74 check (HYG-style heuristic) confirms total-return-adjusted close.
3. **Audit script** (`research/exploration/audit_b6_sector_momentum.py`): runs §2 sweep + §3 gates end-to-end. Outputs `.tmp/reports/b6_sector_momentum/result_log.md`.
4. **Overlay-specific tests**: compare canonical (with overlay) vs unscaled neighbour (crash_scale=1.0 — added as an ADDITIONAL extreme cell, NOT in the §2 sweep grid since the sweep is over scaled values 0.0/0.25/0.50; this 10th "overlay-off" cell is reported separately for the overlay-effectiveness test in §3.7).
5. **Joint L65 + L67** per L75.
6. **Append §4 result log** with verdict.
7. **Update artifacts**: Retirement Registry (if RETIRE), v37_live (if DEPLOY), catalogue meta, README.

---

## §7. Discipline crosscheck (V3.6 / V3.7 lessons)

- **L06 (sparse-trade Sharpe):** N/A — B6 is continuous monthly rebalance (~12 turnovers/year, all 9 ETFs partially active each month).
- **L11 (class defaults):** uses `CROSS_ASSET_MOMENTUM` defaults (same as GEM).
- **L21 (causality smoke):** mandatory.
- **L36 / L37 (academic edge non-replication on yfinance):** acknowledged in §5 caveat 1. The audit is designed to falsify this for sector momentum + overlay.
- **L52 (plateau pre-flight):** mandatory per L75.
- **L60 (annualisation):** `periods_per_year=252` explicit.
- **L61 (multi-instrument panel):** the strategy IS the panel — cross-sectional on 9 sectors is L61-native.
- **L66 (baseline class):** cash (long-short net-zero); 60/40 for L67.
- **L67 (portfolio inclusion):** ≥ 7/10 + lift ≥ 0.05 OR ≥ 8/10.
- **L73 (cutover diff):** applies if DEPLOY.
- **L74 (carry premium):** N/A — momentum, not carry. But L74 pre-flight check (TR-adjusted close) is applied per §5 caveat 6.
- **L75 (hybrid pre-flight mandatory):** §2 sweep + §3 plateau + DSR run BEFORE any deployment claim.
