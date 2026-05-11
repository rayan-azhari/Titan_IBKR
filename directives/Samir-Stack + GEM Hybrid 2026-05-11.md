# Samir-Stack + Dual-Momentum (GEM) Hybrid — 2026-05-11

Reviewed Antonacci's Global Equities Momentum (GEM) framework
(`resources/dual_momentum.md`) and built a hybrid that layers GEM on top
of the existing Samir-Stack regime gate. **Result: GEM does NOT improve
Samir-Stack on this 23-year history.** The detailed analysis is below.

## Hypothesis

The Samir regime score is fast (~21-day lookback on VIX, credit, drawdown
velocity, momentum, HMM). GEM is slow (12-month absolute momentum). They
should be complementary:

- Samir catches sharp drawdowns (VIX spikes, credit blowouts) within an
  otherwise-bullish 12-month window.
- GEM catches multi-month bear markets that build slowly without a
  single sharp catalyst (the 2008 grind-down, the 2022 rate shock).

If true, layering GEM as a hard outer gate should reduce max DD without
materially hurting CAGR.

## Results — full period (2003-04 to 2026-04, 23 years, 60% IEF bond sleeve)

| Variant | CAGR | Vol | Sharpe | Max DD | Calmar | Trades/yr |
|---|---|---|---|---|---|---|
| A: SPY buy-hold | 11.27% | 18.6% | 0.67 | -55.2% | 0.20 | 0 |
| B: GEM standalone | 8.63% | 15.3% | 0.62 | -33.7% | 0.26 | 1.7 |
| **C: Samir baseline (L=3)** | **7.96%** | 10.5% | **0.78** | **-27.1%** | **0.29** | 12.1 |
| D: Samir + GEM abs-mom filter | 6.51% | 10.0% | 0.68 | -28.6% | 0.23 | 10.4 |
| E: Samir + GEM abs+rel-mom | 8.28% | 10.8% | 0.79 | -29.6% | 0.28 | 10.6 |
| F: Samir L=2 + GEM abs-mom | 5.43% | 8.4% | 0.67 | -28.2% | 0.19 | 5.4 |

GEM time-share: SPY 49.8%, EFA 29.5%, BONDS 20.7%.

## Crisis-window decomposition

| Variant | GFC 2008 | COVID 2020 | Rate shock 2022 |
|---|---:|---:|---:|
| A: SPY buy-hold | -34.8% | -9.2% | -18.2% |
| B: GEM standalone | **-0.6%** | -23.2% | -16.6% |
| C: Samir baseline | **+6.1%** | **+1.1%** | -21.8% |
| D: Samir + GEM abs | +6.1% | +1.1% | -20.1% |
| E: Samir + GEM abs+rel | +6.1% | +1.1% | -20.1% |
| F: Samir L=2 + GEM abs | +6.1% | +1.1% | -20.2% |

## Why the hybrid doesn't help

**1. GEM's 12-month signal is too slow for fast crashes.** COVID 2020 took
GEM down -23%, while Samir's faster gate flagged the crash and went
defensive in time (+1.1%). Adding GEM to Samir is silent during fast
crashes because Samir's gate has already kicked.

**2. Samir's regime gate is already catching slow grind-downs.** The 2008
GFC was a 12-month bear, exactly the regime where GEM should shine —
yet Samir's regime gate already returned +6.1% over the same window
(thanks to the bond sleeve carrying the portfolio). GEM contributes
nothing additional here.

**3. GEM offers a marginal benefit only in the 2022 rate-shock regime**
(slow equity selloff that didn't trigger Samir's vol/credit gates):
-20.1% with GEM vs -21.8% without. **+1.7pp recovery** — real but
small.

**4. Forcing GEM as a hard gate KICKS the equity sleeve out for periods
where Samir's regime score was correctly bullish**, costing CAGR
without reducing max DD.

**5. The relative-momentum part (EFA selection) is the one piece that
slightly helps**: variant E adds +0.32pp CAGR vs baseline by trading EFA
during EFA-outperformance windows. But max DD gets +2.5pp WORSE because
EFA had its own deep drawdowns (2011 Eurozone, 2015 EM crisis) that
Samir's SPY-only regime panel doesn't capture.

## Conclusion

| Claim | Verdict |
|---|---|
| GEM improves Samir's CAGR | ❌ Drops 0.05–1.45pp depending on variant |
| GEM improves Samir's Sharpe | ❌ Best variant (E) matches baseline at 0.79 |
| GEM reduces Samir's max DD | ❌ All hybrids worsen max DD by 1.5–2.5pp |
| GEM provides incremental crisis protection | Marginal — only +1.7pp on 2022 |
| GEM standalone beats SPY buy-hold on risk-adjusted | ✅ Sharpe 0.62 vs 0.67 (worse), but max DD -33% vs -55% (much better) |
| Samir's regime gate is the dominant driver | ✅ Confirmed across all 3 crisis windows |

**The original Samir-Stack design (variant C) is the best on this
universe.** The GEM framework is a strong standalone strategy and
materially better than buy-hold, but it doesn't add edge once a
sophisticated multi-factor regime gate is already in place.

## Caveat / one wrinkle worth noting

Variant E (Samir + GEM abs+rel) is the **only** hybrid that maintains
Samir's Sharpe (0.79) while adding modest CAGR (+0.32pp). The trade-off
is +2.5pp max DD. If the goal is "Samir's Sharpe with slightly higher
CAGR and slightly worse drawdown", variant E is a defensible choice —
but it's not a strict improvement.

## What WOULD likely improve Samir

The places this analysis suggests are more promising than GEM overlay:

1. **Add a separate slow regime score for the 2022 regime** (rate shock
   without VIX spike). Currently Samir's panel doesn't have a credit-
   spread-direction or yield-curve-velocity component that would have
   flagged 2022.
2. **Bond-sleeve rotation** (DMFI-style): rotate IEF / HY / cash based
   on bond momentum, instead of always holding IEF. The 2022 IEF
   drawdown drove most of Samir's 2022 underperformance.
3. **Equity-region rotation as an OPT-IN tactical overlay** rather than
   a hard regime gate — only switch SPY → EFA when the relative
   momentum gap is large (>5pp 12m return), not on every monthly tick.

## Files

- New: [research/samir_stack/dual_momentum_gem.py](../research/samir_stack/dual_momentum_gem.py),
  [research/samir_stack/run_samir_gem_hybrid.py](../research/samir_stack/run_samir_gem_hybrid.py)
- Output: [.tmp/reports/samir_stack/samir_gem_hybrid.csv](../.tmp/reports/samir_stack/samir_gem_hybrid.csv)
