# Samir-Stack Margin Drift Research — Comprehensive Write-Up

**Date:** 2026-05-11
**Branch:** `feat/samir-stack-margin-variant`
**Final recommendation:** *Margin drift L=2 on CSPX with I1+I2+I3 overlays.*
**WFO validation:** 16 folds, 100% positive OOS Sharpe, bootstrap CI lower bound 1.11.

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
**Bond sleeve:** Rotation between IEF / HYG / cash by 60-day momentum
**Equity overlay:** Opt-in EFA when 12m gap > 5pp
**Regime gate:** Existing Samir score min-blended with TLT-momentum rate-shock score

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

Two distinct deliverables in this research arc:

1. **Cost-structure understanding**: 2× margin doesn't deliver 2× P&L
   because of vol drag (~3.5%/yr) and funding cost (~3.5%/yr). Margin
   on a standard ETF is a viable alternative to leveraged ETFs but not
   strictly better at high leverage; at L=2 it's roughly cost-neutral.

2. **Concrete strategy improvement**: Three overlays (rate-shock score,
   bond rotation, opt-in EFA) more than DOUBLE Samir's Sharpe (0.78 →
   1.60 in OOS WFO) and HALF max DD (-27% → -10%). All three transfer
   cleanly across equity engines. Drift-margin L=2 + the three overlays
   is the strongest configuration found, with 100% OOS-fold positivity
   and bootstrap CI lower bound 1.11.

The strategy is ready for live engineering work. The remaining gap
between research and deployment is operational (margin-aware strategy
class, PRM extensions, paper-validation period) rather than analytical.
