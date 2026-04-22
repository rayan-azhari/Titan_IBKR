# European Cross-Asset Parameter Sweep — 2026-04-22

Tests whether the cross-asset momentum edge (US bond/credit/dollar
signals → equity) transmits to European index targets (DAX, FTSE)
on daily timeframe, using the same WFO harness as the US sweep.

**Driver**: [scripts/rerank/run_param_sweep_europe.py](../scripts/rerank/run_param_sweep_europe.py)
**Raw**: [.tmp/reports/param_sweep_europe_2026_04_22/results.csv](../.tmp/reports/param_sweep_europe_2026_04_22/results.csv)
**Leaderboard**: [.tmp/reports/param_sweep_europe_2026_04_22/leaderboard.md](../.tmp/reports/param_sweep_europe_2026_04_22/leaderboard.md)

**Scope**: 6 signals (TLT, IEF, HYG, TIP, LQD, UUP) × 2 targets (DAX,
FTSE) × 6 lookbacks × 4 holds × 4 thresholds = **1,152 combos**.
Wall-clock 76 s. Data coverage: FTSE from 1984 (10,665 bars), DAX
from 1987 (9,665 bars).

---

## Headline: **zero Bonferroni survivors on either target**

| Gate | Criterion | Passers |
|---|---|--:|
| Permissive | CI_lo > 0, folds ≥ 25, pos ≥ 60 % | 145 |
| **Bonferroni** | CI_lo ≥ 0.45, folds ≥ 25, pos ≥ 60 %, DD ≥ -40 % | **0** |

The Bonferroni gate was loosened slightly (CI_lo ≥ 0.45 vs 0.50 in the
US sweep) to reflect the smaller N (1,152 vs 3,360 tests). **Nothing
cleared it.**

---

## DAX vs FTSE — the data actually disagrees on which is worse

| Target | Rows | Max Sharpe | Max CI_lo | Positive-Sharpe combos | CI_lo > 0 combos |
|---|--:|---:|---:|--:|--:|
| DAX | 576 | +0.880 | +0.326 | 565 / 576 (98 %) | 149 / 576 (26 %) |
| FTSE | 576 | +0.653 | **+0.127** | 509 / 576 (88 %) | **12 / 576 (2 %)** |

**DAX has a marginal edge; FTSE has nothing.** Every US macro signal
transmits faintly to DAX — point estimates are consistently positive
(98 % of combos) but too small to pass the deployment gate. FTSE shows
essentially zero signal: only 12 of 576 combos even have CI_lo > 0,
and the best CI_lo is +0.127 — not close to deployable.

### FTSE top 5 (for the record)

| Signal | Target | LB | Hold | Th | Sharpe | CI_lo | DD |
|---|---|--:|--:|---:|---:|---:|---:|
| UUP | FTSE | 60 | 10 | 1.00 | +0.653 | +0.127 | -13.8 % |
| UUP | FTSE | 5 | 10 | 0.50 | +0.611 | +0.124 | -28.4 % |
| IEF | FTSE | 20 | 40 | 0.50 | +0.467 | +0.060 | -34.9 % |

None of these are deployable. The point estimates are half to
two-thirds of what the equivalent configs produce on US targets.

### DAX top 5 (the interesting ones)

| Signal | Target | LB | Hold | Th | Sharpe | CI_lo | DD | Story |
|---|---|--:|--:|---:|---:|---:|---:|---|
| UUP | GDAXI | 60 | 40 | 1.00 | +0.880 | +0.326 | -20.6 % | USD up → EUR down → German exporters (BMW, Siemens) earn more |
| HYG | GDAXI | 20 | 20 | 0.75 | +0.780 | +0.260 | -14.6 % | global credit spread → risk-off spillover |
| IEF | GDAXI | 20 | 20 | 0.50 | +0.753 | +0.258 | -38.8 % | US rates → global rate beta |
| HYG | GDAXI | 20 | 10 | 1.00 | +0.740 | +0.254 | -12.6 % | global credit spread |
| HYG | GDAXI | 20 | 10 | 0.25 | +0.786 | +0.248 | -17.1 % | global credit spread |

Two plausible channels show up in the DAX top-10: **UUP → DAX**
(dollar strength → European export boost) and **HYG → DAX** (global
credit cycle). Both stories are economically sensible. Neither is
statistically strong enough for the Bonferroni gate.

---

## Why the edge doesn't transmit (hypotheses)

1. **Noise dilution across currency boundaries**. A US credit spread
   signal predicts US equity at 10-bar horizons. The same signal's
   predictive power for European equity has to survive EUR/USD
   translation noise and a genuinely different macro regime (ECB vs
   Fed cycles). You lose Sharpe every step.
2. **FTSE composition is 70 % USD-earners**. The FTSE-100 has a huge
   share of globally-listed firms (Shell, BP, HSBC, Diageo, AstraZeneca)
   whose economics are more sensitive to the GBP/USD rate than to
   UK-specific macro. The "European equity" label is misleading.
3. **DAX has a weak-but-real dollar channel**. UUP → DAX shows up
   consistently (rows 1, 6, 10, 11, 13 of the top 20) because it's a
   genuine FX-driven export story. The signal would strengthen if
   tested on an FX-hedged DAX series, or if we tested EWG/EWU (USD-
   denominated proxy ETFs) directly rather than the local-currency
   index.
4. **US sample is the "natural" population**. The cross-asset
   momentum literature that motivates these signals (Asness, Ilmanen,
   Kroencke) is US-centric. Transmitting it out-of-sample to Europe is
   a genuine generalisation test and it mostly fails.

---

## Recommendation

### For the v4 live portfolio: **no change**

The v4 portfolio (HYG → IWB, TIP → HYG, TLT → QQQ, MR AUD/JPY, IEF → GLD,
ML IWB) stays US-only. Adding DAX or FTSE exposure via cross-asset
momentum **would degrade the portfolio** — you would add a component
with CI_lo under zero on the same statistical test we use to gate every
other strategy.

### Worth exploring later (bounded follow-ups)

1. **EWG / EWU targets**. Download data for iShares Germany (EWG) and
   iShares UK (EWU), which are USD-denominated tradable proxies for
   DAX / FTSE. The FX translation is already baked into their returns,
   so a HYG → EWG trade would avoid the local-currency issue. If the
   Sharpe is much higher than HYG → GDAXI, we have evidence point 3 is
   correct. Cost: ~5 min sweep.
2. **EU-native signals**. Test German bund futures (ZN, ZF) momentum
   against DAX, or Bund-BTP spread as a European credit-cycle signal.
   This is a different experiment — test the native EU factor, not a
   US factor export.
3. **Currency-hedged index**. Use DAX × (1/EURUSD) as the target to
   isolate the equity component from the FX. If this works where DAX
   fails, we've localised the edge to the equity level, not the FX.

### For the roadmap

Update the System Status document: the gap "DAX/FTSE systematic
deployment" is now **closed** with a negative result. Move both to
`Deprecated Strategies.md` under "cross-asset momentum: does not
transmit to European targets". Document the negative so this
experiment doesn't get re-run in six months.

---

## What this does NOT invalidate

- The DAX IC signal findings (`rsi_21_dev` 60-day, `ma_spread_10_50`) are a different strategy family (direct momentum/mean-reversion on the index itself, not cross-asset). Those may still work — they weren't tested here.
- Autoresearch experiment 83 (`^GDAXI oos3 Sharpe +1.07`) was a single-strategy run on raw DAX with only 9 trades; too sparse to re-evaluate from this sweep.
- A bund-based cross-asset strategy would be a new experiment; we have not run it.
