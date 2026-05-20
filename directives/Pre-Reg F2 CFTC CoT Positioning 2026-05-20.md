# Pre-Registration — F2: CFTC CoT Positioning Extremes (Kang-Rouwenhorst-Tang 2020)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-20
**Branch:** v2-main → research/f2-cot-positioning
**Type:** Strategy audit (class `DAILY_MEAN_REVERSION` — commodity cross-section variant)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE any CFTC CoT / commodity-return joint statistic is computed on our sample. The CFTC CoT API + yfinance commodity `=F` continuous-contract prices are public; no per-cell Sharpe, no per-commodity positioning quantile, no hit-rate has been inspected for OUR specific slice. Data acquisition is §6 step 2, AFTER pre-reg freeze.

> **L76 framing**: F2's source paper is Kang-Rouwenhorst-Tang *JF* 2020 — a post-2018 publication with a 1990-2018 sample. Per L76, post-2018-published edges are LESS exposed to the "pre-2014-sample edge doesn't replicate on post-2008 retail data" pattern than the previous 5 retires (A1, ic_mtf, F3, D4, B6). But the falsification framing still applies: the V3.7 hybrid early-exit on plateau/DSR/CI_lo is the primary defense, and a RETIRE verdict via that path is a clean outcome.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` row F2 — CFTC CoT positioning extremes (Kang-Rouwenhorst-Tang *JF* 2020), 3d effort, free CFTC data.

**Source.** Kang, W., Rouwenhorst, K. G., & Tang, K. (2020). *"A Tale of Two Premiums: The Role of Hedgers and Speculators in Commodity Futures Markets."* Journal of Finance, 75(1), 377–417.

**Key claim (1990-2018 sample).** In commodity futures markets, the CFTC's weekly Commitment of Traders (CoT) reports decompose open interest by trader category (managed money / "speculators", producer-merchant / "hedgers", swap dealers, etc.). Extreme positioning by speculators tends to revert: when speculators are heavily long (top 10-20% historical extreme), forward returns are negative; when heavily short, forward returns are positive. The effect manifests at weekly-to-monthly horizons. KRT 2020 frame the strategy as harvesting a "hedging premium" (commercial hedgers pay speculators to take the other side of their net positions).

**Hypothesis (falsifiable).** A weekly-rebalanced cross-sectional commodity portfolio that goes LONG the bottom-N commodities by speculator-positioning z-score (most-short by speculators) and SHORT the top-N (most-long by speculators) earns positive risk-adjusted returns over the audit window (2006-2026 — Disaggregated CoT report start date through latest), with:

- per-bar annualised Sharpe CI_lo > 0
- L52 plateau spread ≤ 50% across 3×3 cell grid
- DSR p ≥ 0.95
- L65 single + joint ruin PASS at ≤ 10% portfolio weight
- L67 portfolio matrix maintains 7/10 with Sharpe lift ≥ +0.05 over current GEM68/T17/IC15 baseline (Sharpe +1.07)

**Specific RETIRE triggers per §3.7:**

- Plateau spread > 50% → RETIRE (L52)
- DSR p < 0.95 → RETIRE (selection bias)
- Canonical CI_lo ≤ 0 → RETIRE (no edge after costs)
- L65 single FAIL at 5% weight → RETIRE
- L65 joint FAIL at smallest proposed mix → RETIRE

**CONDITIONAL_WATCHPOINT triggers:**

- L67 < 7/10 OR Sharpe lift < +0.05 → paper-only diversifier verdict
- Noise mean degradation > 30% at σ=0.5 → CONDITIONAL (J3 axis)

**Decay risk (per L76).** KRT 2020 published 2 years ago with a 1990-2018 sample. The post-2018 sample is in our visible window but small (~7 years). Decay mechanisms to watch for:
- Algorithmic / quant funds may have arbitraged the positioning signal post-publication.
- The 2020-2022 COVID + post-COVID commodity supercycle is a regime that's atypical of the 1990-2018 KRT sample.

**Mechanism.**

1. **Universe.** Up to 15 commodity futures (intersection of yfinance `=F` continuous-contract daily prices AND CFTC Disaggregated CoT report coverage):
   - Energy: CL (WTI crude), BZ (Brent), NG (natural gas)
   - Metals: GC (gold), SI (silver), HG (copper), PL (platinum), PA (palladium)
   - Grains: ZC (corn), ZW (wheat), ZS (soybeans)
   - Softs: CT (cotton), KC (coffee), SB (sugar), CC (cocoa)
2. **CoT data.** CFTC Disaggregated Futures-Only report (resource `72hh-3qpy` via the CFTC Open Data Socrata API). Weekly, released Friday for Tuesday positions. Key fields:
   - `m_money_positions_long_all` / `m_money_positions_short_all` (Managed Money = speculators)
   - `open_interest_all`
3. **Signal (per commodity, per Friday t):**
   - Speculator net positioning: `spec_net_t = (m_money_long − m_money_short) / open_interest_all`
   - Normalisation: `z_t = (spec_net_t − μ_lookback) / σ_lookback` with rolling lookback window in weeks
4. **Portfolio construction (cross-sectional, per week):**
   - Rank commodities by `z_t`.
   - LONG bottom-N (most-short by speculators); SHORT top-N (most-long).
   - Equal-weighted within each leg; long-leg notional +0.5 NAV, short-leg −0.5 NAV; net = 0.
5. **Position-effective lag.** CoT report released Friday for Tuesday positions. Position taken from Monday open following the Friday release, held through next Monday. **Causality lag = 1 trading week** (signal at week t uses Tuesday positions reported Friday; held from following Monday).
6. **Costs.** Continuous-contract commodity ETF cost ≈ 2 bps per leg per turnover (round-trip ~4 bps per commodity per rebalance). Weekly rebalance → ~52 turnovers/year per commodity. With N=3 long + N=3 short = 6 commodities, ~6 × 52 × 4 = 1248 bps/yr per ensemble = 12.5% annualised cost. **This is high.** Per-cell sweep must account for this; the canonical cell must clear `Sharpe > 0` AFTER this drag.
7. **Causality.** Signal at week t uses CoT report data through t-3 days (Tuesday positions). Position effective from Monday t+0 onwards. L21 smoke: corrupting future CoT / future prices must not change past returns.

**Why this is novel for our stack.** No commodity strategy currently exists in `titan/strategies/` (D2 commodity carry retired 2026-05-15; B4 TSMOM retired). F2 introduces:

- First sentiment / positioning signal.
- Different signal class than current LIVE (cross-sectional momentum × ranks vs cross-sectional sentiment × ranks).
- Tests whether the V3.7 hybrid generalises to weekly-cadence signals (all current LIVE are daily).

---

## §2. Cell sweep (V3.7 hybrid mandatory per L75)

3×3 grid pre-committed BEFORE any CoT-conditional return is computed:

| dim | values |
|---|---|
| `z_lookback` (weeks of rolling history for z-score) | {52, 104, 156} (1y, 2y, 3y) |
| `top_n` (long-leg + short-leg count) | {2, 3, 4} |

Total: **9 cells.** Canonical: `(z_lookback=104, top_n=3)` — KRT 2020's stated baseline closest to a 2-year positioning history × top-tercile.

**Plateau set:** canonical + 4 immediate neighbours. L52 gates:
- H1: spread ≤ 50%
- Strict: spread ≤ 30%

**Per-cell metrics:**
- Annualised Sharpe (`periods_per_year=52` for weekly rebalance; **NOT 252** — per L60).
- Bootstrap CI 95% (n_resamples=2000, seed=42).
- Max drawdown.
- Per-week turnover (informational — verifies cost model accuracy).

**DSR.** Bailey & Lopez de Prado 2014 deflation across the 9 cells.

---

## §3. Gates & decision rule

### §3.1 L52 plateau pre-flight

Spread ≤ 50% H1; strict ≤ 30%. Failure → RETIRE.

### §3.2 5-axis decision matrix

| Axis | Gate | Source |
|---|---|---|
| 1. CI_lo | annualised Sharpe CI_lo > 0 | bootstrap_sharpe_ci |
| 2. DSR | dsr_prob ≥ 0.95 over 9 cells | deflated_sharpe |
| 3. MC abs DD | P(MaxDD > 35%) < 10% | run_block_mc (block=4 weeks; n_paths=200) |
| 4. Sanctuary divergence | visible vs sanctuary Sharpe within ±0.3 | sanctuary_divergence_test |
| 5. Noise robustness | mean degradation < 30% at σ=0.5; worst-case > 0 | run_noise_robustness |

### §3.3 Baseline (L66) and relative MC (L17)

**L66 baseline class** for `DAILY_MEAN_REVERSION` cross-sectional variant: **cash (zero)**. Long-short market-neutral; cash is the right comparator.

**L17 relative MC vs 60/40 SPY/IEF:** APPLIED but de-emphasised. The strategy is uncorrelated with equity/bond by design (commodity-cross-section, not equity), so the rel-MC dd_reduction interpretation has the same quirk as F3 (sparse-active vs continuous). Pre-committed: report rel-MC but don't gate the verdict on it.

### §3.4 L65 single-strategy ruin

Weights: {0.02, 0.05, 0.10}. Gate per `assess_strategy_ruin`: P_kill_trip < 1% AND p95 DD at size > -25%.

### §3.5 L65 joint ruin

With current LIVE (gem_j5 + turtle_cat + ic_top3 basket). Candidates:
- (G68/T17/IC15/F2=0): current reference
- (G66/T17/IC15/F2=2)
- (G64/T16/IC15/F2=5)
- (G62/T13/IC15/F2=10)

### §3.6 L67 10-metric portfolio inclusion

Vs 60/40 SPY/IEF. CURRENT (G68/T17/IC15) baseline at Sharpe +1.07.

**Deploy gate**: ≥ 8/10 OR Sharpe lift ≥ +0.05 with no metric regression.

### §3.7 Decision tree (pre-committed)

```
plateau spread > 50%                          → RETIRE (L52)
DSR p < 0.95                                  → RETIRE (selection bias)
canonical CI_lo ≤ 0                           → RETIRE (no edge)
MC abs-DD gate FAIL                           → RETIRE
sanctuary divergence > 0.30                   → RETIRE (look-ahead suspected)
noise mean degradation > 30% at σ=0.5         → CONDITIONAL (J3 axis)
L65 single FAIL at 5% weight                  → RETIRE
L65 joint FAIL                                → RETIRE
L67 < 7/10 OR Sharpe lift < +0.05             → bench (paper-only)
L67 ≥ 7/10 AND lift ≥ +0.05 (no regression)   → DEPLOY-eligible
otherwise                                     → CONDITIONAL_WATCHPOINT
```

### §3.8 Sanctuary

12-month hold-out at the end of the visible window. Visible = CoT-data-start (2006-06-13) → 2025-05-20; sanctuary = 2025-05-20 → 2026-05-20.

---

## §4. Result log

(To be appended after audit run. Empty at pre-reg commit.)

---

## §5. Known caveats (acknowledged at pre-reg)

1. **L76 still applies.** Although KRT 2020 is post-2018 publication, the SAMPLE is 1990-2018. Some decay risk; the V3.7 hybrid is the gate.

2. **Cost is high.** Weekly rebalance on 6+ commodities × 2 legs × 4 bps round-trip ≈ 12.5%/yr drag. The cell sweep must net-of-cost; any positive Sharpe in the audit IS already net-of-cost.

3. **yfinance `=F` continuous-contract roll handling.** The `=F` series is not transparently roll-adjusted. Roll-noise will introduce per-commodity discontinuities every contract month. We accept this as a known cost of using free retail data; the alternative (IBKR-stitched M1 with 3y history) is too short.

4. **CoT report timing.** Released Friday afternoon for Tuesday-close positions. Our position-effective lag of "Monday following Friday release" is the canonical realistic implementation. We do NOT use the Tuesday positions retrospectively (would be look-ahead).

5. **Disaggregated report starts 2006-06-13.** Legacy CoT format goes back to 1986 but uses different category groupings (no separate Managed Money). For consistency we use Disaggregated only — about 20 years of visible history.

6. **Universe size uncertainty.** Target 15 commodities (intersection of yfinance `=F` availability AND CFTC coverage). Data-acquisition step §6 step 2 will confirm exact count; sweep grid stays 9 cells regardless.

7. **Equity-correlation low by design.** Commodities are uncorrelated with the GEM/Turtle/IC equity-heavy portfolio. Adding F2 should ADD diversification regardless of its standalone Sharpe (per L67 interpretation: low-correlation low-Sharpe sleeves can lift portfolio Sharpe). If standalone Sharpe is mildly positive (e.g. 0.2-0.4), L67 may still favour deployment.

---

## §6. Implementation plan

1. **Pre-reg commit (THIS commit).** Freeze §1–§3.
2. **Data acquisition** (`scripts/download_f2_commodities.py` + `scripts/fetch_cftc_cot.py`):
   - yfinance `=F` continuous-contract daily for the 15 target commodities.
   - CFTC Open Data API Socrata pull for Disaggregated Futures-Only report covering the same commodities, weekly history from 2006-06-13 to latest.
   - Cache CoT data to `data/cftc_cot_disaggregated.parquet` and per-commodity prices to `data/{SYM}_F_D.parquet`.
3. **Audit script** (`research/exploration/audit_f2_cot.py`):
   - L21 causality smoke
   - 3×3 sweep + L52 plateau
   - DSR over 9 cells
   - 5-axis decision matrix
   - L65 single + joint ruin
   - L67 portfolio inclusion
   - Result log to `.tmp/reports/f2_cot/result_log.md`
4. **Append §4 result log** with verdict.
5. **Update artifacts**: Retirement Registry (if RETIRE), `v37_live` registry (if DEPLOY) per L73 checklist, V3.6 catalogue meta, README.

---

## §7. Discipline crosscheck (V3.6 / V3.7 lessons)

- **L06 (sparse-trade Sharpe):** N/A — F2 is weekly-continuous, ~52 obs/year.
- **L11 (class defaults):** uses `DAILY_MEAN_REVERSION` defaults (provisionally).
- **L21 (causality smoke):** mandatory.
- **L52 (plateau pre-flight):** mandatory per L75.
- **L60 (annualisation):** `periods_per_year=52` for weekly rebalance. NOT 252.
- **L61 (multi-instrument panel):** the strategy IS the panel — L61-native.
- **L66 (baseline class):** cash (long-short net-zero); 60/40 for L67.
- **L67 (portfolio inclusion):** ≥ 7/10 + lift ≥ 0.05 OR ≥ 8/10.
- **L73 (cutover diff):** applies if DEPLOY.
- **L74 (carry-premium math):** N/A — sentiment-based signal, not carry.
- **L75 (hybrid pre-flight mandatory):** §2 sweep + §3 plateau + DSR run BEFORE any deployment claim.
- **L76 (falsification framing):** F2 is FRAMED as a falsification test of the KRT 2020 design principle on retail-implementable post-2008 data. PASS = meaningful evidence the mechanism survives. FAIL = expected baseline per the 5-strategy decay cascade.
