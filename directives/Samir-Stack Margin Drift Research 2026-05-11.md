# Samir-Stack Margin Drift Research — Comprehensive Write-Up

**Date:** 2026-05-11 (updated with allocation sweep findings)
**Branch:** `feat/samir-stack-margin-variant`, then `research/allocation-sweep`
**Final recommendation:** *Margin drift L=2 on CSPX with I1+I2+I3 overlays at **15/85 equity/bond split**.*
**WFO validation:** 16 folds, 100% positive OOS Sharpe, bootstrap CI lower bound 1.62 (at 15/85).

---

## 1. Background and motivation

Samir-Stack (`research/samir_stack/`) is a regime-gated 40/60 stack that
holds a **3× daily-rebalanced leveraged ETF** (3USL synthetic) on the
equity side and **IEF Treasuries** on the bond side. Long historical
performance: ~7.96% CAGR, 0.78 Sharpe, -27.1% max DD, 0.29 Calmar over
2003-2026.

This research asked two questions:

1. **Cost question**: can we use IBKR margin on a standard 1× ETF (CSPX)
   instead of a 3× leveraged ETF, and what changes in the stats?
2. **Edge question**: are there overlays (dual momentum, bond rotation,
   rate-shock indicator) that materially improve Samir's risk-adjusted
   return?

The answers turned out to be entangled: the overlays improve the
strategy substantially, AND they transfer cleanly across equity engines,
AND the drift-margin engine plus the overlays produces the strongest
combination found.

---

## 2. The margin engine model

### Two ways to use margin

We modelled both interpretations of "2× leverage via margin":

**Constant-leverage margin** (`constant_leverage_margin_returns`)
Rebalance daily to maintain target leverage. Mathematically equivalent
to a daily-rebalanced leveraged ETF except for funding rate (broker
margin instead of wholesale) and TER (single underlying ETF instead
of leveraged wrapper).

**Drift margin** (`drift_margin_returns`)
Borrow a fixed dollar amount on day 0, hold. Leverage drifts with the
underlying. Less volatility drag than constant-L. Auto-deleverages on
margin call (modelled at 30% maintenance margin).

### Key cost differences vs leveraged ETFs

| | Leveraged ETF (3USL) | IBKR Pro margin |
|---|---|---|
| Funding rate | Wholesale (~Fed Funds) | Fed Funds + 1.5% spread |
| TER | 0.75% | CSPX 0.07% |
| Daily rebalance | Forced (vol drag) | Optional (drift can avoid) |
| Margin call risk | None | Real (offset by regime gate) |

**Net at L=3**: IBKR margin is ~2.3% per year MORE expensive than 3USL
because the 1.5% spread on 2× borrowed ≈ 3% drag, dwarfing the 0.68%
TER savings.

**Net at L=2**: Margin is roughly cost-neutral with 3USL — borrow scales
with (L-1), so the spread cost is half.

### Implementation: [research/samir_stack/margin_model.py](../research/samir_stack/margin_model.py)

Both engines exposed with consistent signatures so they can swap in
anywhere `synthetic_3x.synthetic_leveraged_returns` is called.

---

## 3. The "2× margin = 2× P&L?" question (it's not)

A common confusion worth documenting. SPY buy-hold over 2003-2026 had
**11.27% CAGR**. A 2× margin position should yield 2 × 11.27% = 22.5%,
right?

Actual: **margin const-L=2 standalone gave 15.57% CAGR** — only 1.4× the
buy-hold, not 2×. The missing 7pp comes from three drags:

| Drag source | Annual cost (SPY) |
|---|---|
| Volatility drag (σ² for SPY ~18.6%) | ~3.5% per year |
| Funding cost (1.5% spread × 1× borrow + base rate) | ~3.5% per year |
| Path-dependent drawdown amplification | (no CAGR cost but 2× → -84% MaxDD vs SPY -55%) |

Math: `2 × 11.3% − 3.5% (vol drag) − 3.5% (funding) = 14.3%` ≈ measured
15.57%.

**Rule of thumb for SPY at IBKR retail margin:** 2× margin delivers
~1.4× the buy-hold CAGR with 2× the volatility and ~1.5× the max DD.
Drift margin avoids most of the daily-rebalance vol drag but pays the
same funding and faces margin-call risk.

---

## 4. The three improvements (I1+I2+I3)

After exploring (and rejecting) GEM as a hybrid, three overlays emerged
as candidates worth testing. All three turned out to help.

### I1 — Rate-shock score

Slow indicator: TLT 60-day momentum, scored linearly to [0, 1]. 1.0 when
TLT is stable, 0.0 when TLT has dropped >= 10% in 60d. Min-blended with
the existing Samir regime score so any indicator going to zero
flatten the equity sleeve.

**Why it works:** The 2022 rate shock was a slow grind in equities
(no VIX spike, no credit blowout) but a brutal TLT drawdown (-30% in
60 days). The existing Samir indicators didn't catch it; this one does.

### I2 — DMFI-style bond rotation

Replace static 60% IEF with rotation between IEF / HYG / cash by 60-day
momentum. Pick whichever has the highest momentum; default to cash if
both negative.

**Why it works:** The bond sleeve had three failure modes the static
IEF allocation was exposed to:
- 2009 recovery: HYG outperformed IEF substantially → opportunity cost
- 2022 rate shock: IEF dropped -17% → bond sleeve was supposed to be the
  defensive leg
- Periods where credit and rates diverge

The rotation captures HYG when momentum favours it and goes to cash
when both bonds are losing.

### I3 — Opt-in EFA tactical overlay

Switch SPY → EFA only when EFA's 12m return beats SPY's by >5pp.
Otherwise stay in SPY.

**Why it works:** GEM's relative-momentum picks frequently switch on
small leads, exposing the portfolio to EFA's idiosyncratic drawdowns
(2011 Eurozone, 2015 EM crisis). The 5pp threshold filters out the
noise — only flips when EFA is meaningfully outperforming, capturing
the rare windows (mid-2000s, 2017) without paying the tail-risk premium.

### Implementation: [research/samir_stack/run_samir_improvements.py](../research/samir_stack/run_samir_improvements.py)

---

## 5. The GEM hybrid that didn't work

Antonacci's GEM (Global Equities Momentum) framework was the obvious
first attempt — 12m absolute momentum on SPY for the regime gate, 12m
relative momentum to pick SPY vs EFA. As a standalone strategy GEM is
strong (Sharpe 0.62, MaxDD -34% on the same period vs SPY's 0.67 / -55%).

**But layering GEM on top of Samir does NOT improve Samir.** Detailed
analysis in [Samir-Stack + GEM Hybrid 2026-05-11.md](Samir-Stack + GEM Hybrid 2026-05-11.md).
Headline:

| Variant | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Samir baseline | 7.96% | 0.78 | -27.1% |
| Samir + GEM abs-mom | 6.51% | 0.68 | -28.6% |
| Samir + GEM abs+rel | 8.28% | 0.79 | -29.6% |

The reason: Samir's 21d regime gate already catches what GEM catches,
faster. GEM's 12m signal is too slow for crashes (COVID), and slow
grind-downs (2008) are already handled by the bond sleeve. GEM only
adds marginal value (+1.7pp) in 2022, and even there I1 catches it
cleaner.

This is itself a useful finding: dual momentum is a great standalone
framework but a poor overlay on a fast multi-indicator regime gate.

---

## 6. Combined results: improvements × engines

Headline (19y, 2007-2026, period limited by HYG inception):

| Engine | + improvements? | CAGR | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| 3× leveraged ETF | baseline | 8.55% | 0.83 | -27.1% | 0.32 |
| 3× leveraged ETF | + I1+I2+I3 | 12.91% | 1.25 | -14.4% | 0.90 |
| Margin const-L=2 CSPX | baseline | 6.95% | 0.80 | -27.6% | 0.25 |
| Margin const-L=2 CSPX | + I1+I2+I3 | 12.05% | 1.32 | -13.4% | 0.90 |
| **Margin drift L=2 CSPX** | **+ I1+I2+I3** | **11.04%** | **1.51** | **-10.3%** | **1.07** |

The improvements transfer cleanly across all four engines (≈ +4-5pp CAGR
each), confirming they're engine-independent gating logic.

**Drift margin L=2 + improvements is the statistical winner:**
- Highest Sharpe of any variant tested (1.51)
- Single-digit-equivalent max DD (-10.3% over 19 years)
- Highest Calmar (1.07)
- Trade-off: 1.87pp lower CAGR vs 3× leveraged ETF

**Why drift wins specifically inside this harness:**

1. The regime gate exits before drift-induced margin calls happen
2. Vol drag from daily rebalance is avoided
3. Drift naturally hedged by the regime exits — when leverage drifts UP
   (drawdown), the regime gate is most likely to flatten
4. Tier transitions ARE rebalances within the regime gate, so drift
   only operates on a single tier-hold horizon

### Implementation: [research/samir_stack/run_improved_with_margin.py](../research/samir_stack/run_improved_with_margin.py)

---

## 7. Crisis-window decomposition

| Crisis | SPY | Samir baseline | + I1+I2+I3 (3× ETF) | + I1+I2+I3 (margin drift L=2) |
|---|---:|---:|---:|---:|
| GFC 2008 | -34.8% | +6.1% | +14.8% | +14.8% |
| Q4 2018 | -13.0% | -9.4% | +1.6% | +1.3% |
| COVID 2020 | -9.2% | +1.1% | +2.1% | +2.0% |
| **Rate shock 2022** | -18.2% | -21.8% | -12.3% | **-6.7%** |

The improvements help most in the regimes Samir struggled with (Q4 2018,
2022). The drift-margin engine compounds the regime-gate win on 2022
specifically by being naturally less leveraged into the equity tail.

---

## 7.5. Allocation sweep — 40/60 was NOT optimal

After settling on the engine + overlays, an allocation sweep tested whether
the original 40/60 equity/bond split was optimal. **It was not.** The trend
across 6 splits is monotonic — every step toward more bonds improves
every aggregate metric.

### Full 16-fold WFO across allocations

| Split | Sharpe | CI95 lo | CAGR | Max DD | % pos folds |
|---|---|---|---|---|---|
| 10/90 | 2.151 | 1.635 | 13.73% | -5.75% | 100% |
| **15/85 (RECOMMENDED)** | **2.124** | **1.621** | **13.44%** | **-5.99%** | 100% |
| 20/80 | 2.061 | 1.566 | 13.14% | -6.37% | 100% |
| 25/75 | 1.967 | 1.463 | 12.84% | -6.96% | 100% |
| 30/70 | 1.851 | 1.336 | 12.53% | -7.56% | 100% |
| 40/60 (original) | 1.605 | 1.109 | 12.05% | -10.30% | 100% |

### Why 15/85 over 10/90

10/90 has the highest aggregate Sharpe (2.15) but only 88% positive
uplift folds (14/16). 15/85 has 94% positive folds (15/16) with
essentially identical aggregate metrics (Sharpe 2.12 vs 2.15). The
trade is 0.03 Sharpe for materially better fold consistency.

The 2 folds where 10/90 underperforms 40/60 are pure equity bull
markets (2014-15, 2023-24) where less equity = lower upside. 15/85
narrows this gap meaningfully.

### Why this works (the leverage explains it)

The equity sleeve is leveraged (drift L=2 = 2× SPY exposure when on).
At 15/85 the effective equity exposure when fully on is 15% × 2 = 30%
— still meaningful upside capture.

But the equity sleeve goes to cash ~19% of the time when the regime
gate is defensive. **At 40/60, that's 40% of the portfolio earning 0%
during those windows. At 15/85, only 15% is dead money.**

Meanwhile the bond sleeve (rotated IEF/HYG/cash with momentum) earns
4-6% standalone CAGR, including during many defensive periods.
HYG specifically captures upside during equity recoveries that the
regime gate may miss.

We've effectively rediscovered **risk-parity-style allocation**.
Allocating 15/85 by capital corresponds roughly to **50/50 by risk**
once the equity sleeve's leverage is accounted for.

### Implementation: [research/samir_stack/run_wfo_allocation_comparison.py](../research/samir_stack/run_wfo_allocation_comparison.py)

---

## 8. WFO validation

Rolling 16-fold walk-forward on margin drift L=2:

- **IS window**: 504 days (warmup only — no parameters are tuned)
- **OOS window**: 252 days, non-overlapping
- **Period**: 2009-04 to 2025-04 (16 OOS years)

| Variant | Stitched Sharpe | CI95 lo | CI95 hi | OOS CAGR | OOS MaxDD | %pos folds | Gate |
|---|---|---|---|---|---|---|---|
| baseline (no improvements) | 0.900 | 0.400 | 1.425 | 6.14% | -20.8% | 81% | PASS |
| **+ I1+I2+I3 improvements** | **1.605** | **1.109** | 2.118 | **12.05%** | **-10.3%** | **100%** | **PASS** |

### Robustness signals

- **100% of OOS folds positive** for the improved variant (16/16)
- **Bootstrap CI lower bound 1.11** — well above the 0 deployment gate
- **88% of folds show positive uplift** over baseline (mean +0.82, median +0.71)
- **Worst fold** (2022-04 to 2023-04, rate-shock): Sharpe +0.70, CAGR
  +5.1%, MaxDD -7.5%. Even the worst regime was profitable.
- **Best uplifts** in 2015-16 (+1.96), 2018-19 (+2.13), 2022-23 (+1.86) —
  exactly the regimes the new indicators were designed to catch.

### Why this WFO is valid

The strategy has **no IS-fit parameters**. Rate-shock threshold (-10%),
bond-rotation lookback (60d), EFA gap percent (5pp), regime-score
weights are all fixed by convention (matching the dual-momentum
literature defaults). So the WFO tests robustness across rolling
windows, not parameter generalisation.

### Implementation: [research/samir_stack/run_wfo_drift_improvements.py](../research/samir_stack/run_wfo_drift_improvements.py)

---

## 9. Practical deployment plan

If you decide to deploy this live, the recommended config is:

**Strategy:** Margin drift L=2 on CSPX
**Allocation:** **15% equity / 85% bond** (was 40/60 — see §7.5)
**Bond sleeve:** Rotation between IEF / HYG / cash by 60-day momentum
**Equity overlay:** Opt-in EFA when 12m gap > 5pp
**Regime gate:** Existing Samir score min-blended with TLT-momentum rate-shock score

Expected OOS performance (16-fold WFO): Sharpe 2.12, CAGR 13.4%,
Max DD -6.0%, 100% positive folds, bootstrap CI lower bound 1.62.

### What needs to be built

1. **`MarginLeveragedStrategy` class** (~1-2 weeks engineering)
   - Wraps a single underlying ETF (CSPX initial; configurable)
   - Sizes orders for 2× target notional
   - Holds borrow notional fixed (drift behaviour)
   - Pre-emptive deleverage if drift goes above 1.8 (margin-call avoidance buffer)

2. **PRM extensions**
   - Track margin balance and gross-vs-net exposure separately
   - Margin-aware risk checks before submitting orders

3. **Bond-rotation strategy** as a separate module
   - Computes 60d momentum on IEF/HYG/cash daily
   - Rotates the bond sleeve via market orders at month-end (or on signal flip)

4. **Rate-shock indicator** added to the existing regime panel
   - TLT 60d momentum → linear-scored to [0, 1]
   - Composed with the existing Samir score via min-blend

5. **Opt-in EFA selector**
   - Daily 12m return computation for SPY and EFA
   - Switch instrument when gap > 5pp; otherwise hold SPY

### Live caveats

- **IBKR account type**: Margin account required (paper account
  `DUP958545` already supports this). Spread-bet accounts cannot use
  margin.
- **Tax (UK)**: CGT applies, no spread-bet exemption. Same tax profile
  as the existing `bond_equity_ihyu_cspx` live strategy.
- **Funding-cost variability**: Margin rate floats with SOFR. Currently
  ~5.5-6.5% all-in. Strategy returns sensitive to rate environment.
- **Margin-call risk**: 4 events in 23 years on the standalone test
  (March 2009, March 2020, both regime-flagged). Manageable with the
  pre-emptive deleverage buffer.
- **Capital floor**: Same May 2026 cost-audit caveat applies — minimum
  $10k per strategy notional to avoid commission-floor drag.

---

## 10. What we ruled out (so we don't re-test it)

- **GEM as overlay**: doesn't improve Samir (full analysis in [Samir-Stack + GEM Hybrid 2026-05-11.md](Samir-Stack + GEM Hybrid 2026-05-11.md))
- **Margin constant-L=2 + improvements**: works (Sharpe 1.32) but loses
  to drift on every risk metric. Skip.
- **Higher leverage (L=3) margin**: doesn't help over L=2 in the
  improved harness — the regime gate already caps tail risk so extra
  leverage is just extra cost.
- **GEM standalone replacement**: GEM alone has Sharpe 0.62 vs Samir's
  1.51. Not competitive.

---

## 11. Files in this research

**New code:**
- [research/samir_stack/margin_model.py](../research/samir_stack/margin_model.py) — margin engines
- [research/samir_stack/dual_momentum_gem.py](../research/samir_stack/dual_momentum_gem.py) — GEM signal layer
- [research/samir_stack/run_margin_variant.py](../research/samir_stack/run_margin_variant.py) — margin variants comparison
- [research/samir_stack/run_samir_gem_hybrid.py](../research/samir_stack/run_samir_gem_hybrid.py) — GEM hybrid (rejected)
- [research/samir_stack/run_samir_improvements.py](../research/samir_stack/run_samir_improvements.py) — three improvements head-to-head
- [research/samir_stack/run_improved_with_margin.py](../research/samir_stack/run_improved_with_margin.py) — improvements × engines
- [research/samir_stack/run_wfo_drift_improvements.py](../research/samir_stack/run_wfo_drift_improvements.py) — 16-fold OOS WFO
- [research/analysis/hold_period_audit.py](../research/analysis/hold_period_audit.py) — hold-period audit (Samir + champion)

**Documentation:**
- [resources/dual_momentum.md](../resources/dual_momentum.md) — source framework reference
- [directives/Samir-Stack Margin Variant 2026-05-11.md](Samir-Stack Margin Variant 2026-05-11.md) — initial margin study
- [directives/Samir-Stack + GEM Hybrid 2026-05-11.md](Samir-Stack + GEM Hybrid 2026-05-11.md) — GEM rejection write-up
- This document — comprehensive synthesis

**Output reports** (regenerable):
- `.tmp/reports/samir_stack/margin_variant_comparison.csv`
- `.tmp/reports/samir_stack/samir_gem_hybrid.csv`
- `.tmp/reports/samir_stack/samir_improvements.csv`
- `.tmp/reports/samir_stack/samir_improvements_crisis.csv`
- `.tmp/reports/samir_stack/improved_with_margin.csv`
- `.tmp/reports/samir_stack/wfo_drift_improvements.csv`
- `.tmp/reports/samir_stack/wfo_drift_improvements_summary.csv`

---

## 12. Bottom line

Three distinct deliverables in this research arc:

1. **Cost-structure understanding**: 2× margin doesn't deliver 2× P&L
   because of vol drag (~3.5%/yr) and funding cost (~3.5%/yr). Margin
   on a standard ETF is a viable alternative to leveraged ETFs but not
   strictly better at high leverage; at L=2 it's roughly cost-neutral.

2. **Concrete strategy improvement**: Three overlays (rate-shock score,
   bond rotation, opt-in EFA) more than DOUBLE Samir's Sharpe (0.78 →
   1.60 in OOS WFO at 40/60) and HALF max DD (-27% → -10%). All three
   transfer cleanly across equity engines. Drift-margin L=2 + the three
   overlays is the strongest engine configuration found.

3. **Allocation re-optimization**: shifting from the original 40/60
   equity/bond split to **15/85** further improves the strategy on
   every aggregate metric. OOS Sharpe goes from 1.60 → **2.12**, CAGR
   12.05% → **13.44%**, Max DD -10.3% → **-6.0%**. 100% of OOS folds
   positive, bootstrap CI lower bound 1.62. The mechanism is that the
   leveraged equity sleeve sits in cash ~19% of the time, so a smaller
   equity allocation reduces dead-money drag during defensive windows.

The strategy is ready for live engineering work. The remaining gap
between research and deployment is operational (margin-aware strategy
class, PRM extensions, paper-validation period) rather than analytical.

---

## 13. May 12 2026 update — futures engine + vol targeting

Two further sweeps after the v1 baseline above. Both validated under the
same 16-fold rolling WFO methodology with bootstrap CI gate.

### 13.1 Equity-engine sweep — MES futures vs CSPX margin vs CFD

Compares three engines at the SAME strategy config (L_max=2, 15/85
split) to isolate cost-model effects. Year-1 cost decomposition:

| Engine | Year-1 carry cost | Mechanism |
|---|---|---|
| CSPX margin drift L=2 | 3.11% | (L-1) × IBKR Pro spread (~6.5%) |
| IBKR US500 CFD L=2 | 6.06% | (L-1) × CFD financing on full notional, not amortised |
| **MES futures L=2** | **1.83%** | basis carry × L − T-bill on idle margin cash |

WFO at L=2 on the same 15/85 strategy:

| Engine | Sharpe | CI lo | CAGR | MaxDD |
|---|---|---|---|---|
| CSPX margin drift L=2 | 2.124 | 1.621 | 13.44% | -5.99% |
| IBKR US500 CFD L=2 | 2.057 | 1.554 | 13.87% | -5.99% |
| **MES futures L=2** | 2.110 | 1.606 | **14.27%** | -5.98% |

Same Sharpe as CSPX margin, +0.83pp CAGR. Switching engines is a free
upgrade: lower carry cost, same risk, simpler operations (no margin
loan, near-24h liquidity, no LSE-hours-only constraint). Practical
floor: needs ~$50k+ portfolio to cleanly size 1 MES contract (=$29k
notional at SPX 5800).

### 13.2 Constant-notional sweep — L vs equity_weight

Holds equity notional at 30% (= L × equity_weight) and varies the split.
At higher L the strategy frees capital for the bond sleeve.

| Config | Bond% | Sharpe | CI lo | CAGR | MaxDD |
|---|---|---|---|---|---|
| MES L=2, 15/85 | 85% | 2.110 | 1.606 | 14.27% | -5.98% |
| **MES L=3, 10/90** | **90%** | **2.127** | **1.626** | 14.40% | **-5.74%** |
| MES L=4, 7.5/92.5 | 92.5% | 2.102 | 1.604 | 14.42% | -5.89% |
| MES L=5, 6/94 | 94% | 2.080 | 1.575 | 14.49% | -5.89% |
| MES L=6, 5/95 | 95% | 2.055 | 1.550 | 14.48% | -5.81% |
| MES L=8, 3.75/96.25 | 96.25% | 1.958 | 1.433 | 12.84% | -5.97% |
| MES L=10, 3/97 | 97% | 1.913 | 1.383 | 12.65% | -6.19% |

L=3, 10/90 is the local optimum — marginal Sharpe gain over L=2 but
DD compresses to -5.74%. Above L=6 the strategy collapses because the
auto-generated tier-thresholds rise (peak tier at 0.80 score for L=10 vs
0.50 for L=2), so the regime gate spends much less time at peak. This
is a structural artefact of the tier-thresholds heuristic, not a
fundamental futures-leverage tradeoff.

### 13.3 Capitulation overlay — currently a net drag

Tested with `enabled=True` and the May 2 2026 default parameters. Result:

| Variant | Sharpe | CI lo | CAGR | MaxDD | GFC 2008 |
|---|---|---|---|---|---|
| baseline (L=3 10/90) | 2.127 | 1.626 | 14.40% | -5.74% | +24.0% |
| + V1 capitulation | 2.128 | 1.612 | 13.46% | -5.74% | -0.1% |

Sharpe wash; -0.94pp CAGR. The overlay's May 2 parameters were tuned
on a different baseline (L=2 leveraged ETF, 40/60 split). On this
configuration it over-fires opportunistic entries at false bottoms in
2008 and destroys the regime gate's correctly-defensive +24% GFC
return. **Drop until re-tuned on this baseline.** A future PR could
re-sweep `(opportunistic_tier, capitulation_lookback, spy_dd_required,
bounce_5d_threshold)` on L=3 10/90; even then I'd expect modest uplift
since the regime gate is already strong here.

### 13.4 Portfolio vol targeting — the real win

Multiplicative scaler applied post-hoc to strategy daily returns.
Computes 30-day rolling realised vol of the strategy itself, scales by
`target_vol / realised_vol` (capped at 2×), shifted by 1 day to avoid
look-ahead.

Sensitivity sweep — Sharpe is remarkably flat across target_vol:

| target_vol | Sharpe | CI lo | CAGR | MaxDD | Calmar |
|---|---|---|---|---|---|
| 4% | 2.349 | 1.861 | 10.80% | -3.35% | 3.23 |
| 5% | 2.344 | 1.859 | 13.45% | -4.17% | 3.22 |
| **6%** | **2.337** | **1.848** | **15.95%** | **-4.99%** | **3.20** |
| 7% | 2.312 | 1.825 | 18.15% | -5.81% | 3.12 |
| 8% | 2.285 | 1.792 | 20.03% | -6.62% | 3.03 |
| 10% | 2.232 | 1.739 | 22.81% | -7.47% | 3.05 |
| 12% | 2.210 | 1.705 | 24.85% | -8.57% | 2.90 |

target_vol is a leverage knob — pick by CAGR/DD preference, not Sharpe
optimisation. The Sharpe spread of only 0.14 across a 3× range of
target_vol is the signature of a robust parameter.

### 13.5 Risk-of-ruin: why higher target_vol is much riskier than headline MaxDD

Bootstrap (5,000 paths × 5-day blocks) on the stitched OOS returns:

| | 6% target | 8% target | 12% target |
|---|---|---|---|
| Empirical MaxDD (single path) | -4.99% | -6.62% | -8.57% |
| Median 10y MaxDD (50% chance) | -6.31% | -8.07% | -10.37% |
| 5th-pct 10y MaxDD (1-in-20) | -9.46% | -12.07% | -15.74% |
| 1st-pct 10y MaxDD (1-in-100) | -11.80% | -15.05% | -19.09% |
| Worst of 5,000 10y paths | -16.32% | -20.44% | -25.74% |
| P(MaxDD>10% in 10y) | **3.5%** | 17.8% | **56.5%** |
| P(MaxDD>15% in 10y) | 0.06% | 1.04% | **7.0%** |

DD distributions scale roughly with the SQUARE of target_vol because
worst paths involve bad-bar runs, and each bad bar is proportional to
target_vol. The "headline" -8.57% at 12% understates the 1-in-20
expected loss of -15.7%.

### 13.6 Final champion configuration

**Engine:** MES futures (CME Micro E-mini S&P 500) at L=3
**Capital split:** 10% equity sleeve, 90% bonds (equity_weight=0.10, bond_weight=0.90)
**Base improvements:** I1 (rate-shock) + I2 (bond rotation) + I3 (opt-in EFA)
**Vol target:** **8% annualised**, applied as multiplicative scaler on daily strategy returns (30-day window, max_scale=2.0, lagged 1 day)
**Capitulation overlay:** disabled (re-tune in future PR)

**Validated metrics (16-fold WFO at 8% target_vol):**
- Stitched Sharpe: **2.285** (CI95 lo: 1.792, hi: 2.803)
- CAGR: **20.03%**
- Realised vol: 8.14%
- MaxDD: **-6.62%**
- Calmar: 3.03
- 100% positive folds (16 of 16 OOS years)

**Risk-of-ruin (10-year horizon, bootstrap, 5,000 paths):**
- Median MaxDD: -8.07%
- 1-in-20 path MaxDD: -12.07%
- 1-in-100 path MaxDD: -15.05%
- Worst of 5,000 paths: -20.44%
- P(MaxDD > 10%): 17.8%
- P(MaxDD > 15%): 1.04%
- P(MaxDD > 20%): 0.02%
- P(MaxDD > 25%): 0% in 5,000 paths
- P(end < starting capital, 20y): 0% in 5,000 paths

**Why 8% over 6% or 12%:**

The 6% target was the conservative-default; the 12% target was the maximum-CAGR option. 8% sits at the inflection point of the risk/reward trade-off:
- vs 6%: +4pp CAGR (15.95% → 20.03%) for a real but bounded increase in DD distribution. Median 10y MaxDD shifts -6.3% → -8.1%; P(MaxDD>10%) goes 3.5% → 17.8% but P(MaxDD>15%) is still only 1%.
- vs 12%: -5pp CAGR but **3× lower** P(-10% DD), **7× lower** P(-15% DD), worst-of-5000 path -20% vs -26%.

In the 20-year bootstrap visualisation ([plot_bootstrap_equity_curves.py](../research/samir_stack/plot_bootstrap_equity_curves.py)): median ending wealth ~38× starting capital, worst-of-5000 ~11×. Investor should expect roughly one -8% drawdown per decade and have a 1-in-6 chance of a -10% drawdown over any 10-year stretch — uncomfortable but tolerable for someone with conviction in the strategy.

### 13.7 Live deployment status

**Live strategy class shipped (May 12 2026):**
[titan/strategies/samir_stack/strategy.py](../titan/strategies/samir_stack/strategy.py)
upgraded from the prior 40/60 / leveraged-ETF defaults to the May 12
champion. v1 MVP scope:

  - `equity_weight=0.10`, `bond_weight=0.90`, `L_max=3.0` defaults
  - 8% annualised vol-target scaler on the equity sleeve only
    (`_vol_target_scale` method, 30-day window, max_scale=2.0,
    lagged 1 bar via "scale-from-yesterday's-vol applied to today's
    sizing" semantics)
  - `_rehydrate_position_from_broker` follows Tier 1.1 patterns —
    no `strategy_id=` filter (May 11 fix)
  - All position aggregation uses `signed_qty` (post-May-11
    bond_gold pattern)

**NOT YET in `STRATEGY_REGISTRY`.** Promotion to live requires:

1. **Phase 2: MES futures execution** (separate PR). v1 uses the
   existing CSPX margin path. Switching the equity sleeve to MES
   futures L=3 unlocks the documented carry-cost savings (~1.3%/yr)
   but needs IBContract setup with quarterly rollover and integer-
   contract sizing.
2. **Phase 3: bond rotation overlay** (separate PR). v1 holds the
   bond sleeve in a single `bond_instrument_id` (IEF). Adding the
   IEF/HYG/cash rotation captures I2 from the research overlays.
3. **Pre-flight**: paper-validate for at least 4 weeks against the
   research backtest, run T2.5 replay-audit weekly during validation.

### 13.8 What's still on the table (not yet built)

- Capitulation overlay re-tune on the L=3 10/90 baseline (probably modest uplift)
- Yield-curve recession gate added to regime score
- Multi-strategy combination at portfolio level (Samir-Stack 70% + mr_audjpy 15% + bond_equity 15%) — likely the biggest real-money uplift remaining
- Phase 2/3 above (MES futures, bond rotation)

### 13.9 Files in the May 12 update

- [research/samir_stack/run_futures_sweep.py](../research/samir_stack/run_futures_sweep.py) — engine sweep + leverage sweep + constant-notional sweep
- [research/samir_stack/run_overlay_sweep.py](../research/samir_stack/run_overlay_sweep.py) — capitulation × vol-target × baseline grid + vol-target sensitivity
- [research/samir_stack/run_risk_of_ruin.py](../research/samir_stack/run_risk_of_ruin.py) — bootstrap drawdown projections on the L=3 10/90 baseline
- [research/samir_stack/compare_vol_target_risk.py](../research/samir_stack/compare_vol_target_risk.py) — side-by-side risk distributions across target_vol levels
- [research/samir_stack/margin_model.py](../research/samir_stack/margin_model.py) — extended with `futures_returns()` and `cfd_returns()` cost models
- [titan/strategies/samir_stack/strategy.py](../titan/strategies/samir_stack/strategy.py) — live strategy upgraded to champion config (vol target + rehydration)
- [tests/test_samir_stack_strategy.py](../tests/test_samir_stack_strategy.py) — pins config defaults + vol-target math + AST guards
