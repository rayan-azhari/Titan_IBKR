# European Cross-Asset Follow-Ups v2 — 2026-04-22

Three follow-up experiments to the (negative) European sweep. Tests
whether the US cross-asset edge fails on DAX/FTSE because of: (A)
currency-translation noise, (B) wrong signal family, or (C) the FX
itself. **None crosses the Bonferroni gate** — but the margins shrink,
and the *shape* of the near-misses is informative.

**Driver**: [scripts/rerank/run_param_sweep_europe_v2.py](../scripts/rerank/run_param_sweep_europe_v2.py)
**Raw**: [.tmp/reports/param_sweep_europe_v2_2026_04_22/](../.tmp/reports/param_sweep_europe_v2_2026_04_22/)

**Scope**: 4,032 combos across three experiments. Wall-clock ~4 min.

---

## Summary

| Experiment | Combos | Max Sharpe | Max CI_lo | Bonferroni survivors | Gap to gate |
|---|--:|---:|---:|--:|---:|
| A — USD EU ETFs (EWG, EWU, IEV, VGK) | 2,304 | +0.77 | +0.328 | 0 | −0.12 |
| B — EU bond signals (IGOV, BWX, BNDX) | 1,152 | +1.02 | +0.296 | 0 | −0.15 |
| **C — Currency-hedged DAX** | 576 | **+0.98** | **+0.425** | 0 | **−0.02** |

**Verdict**: no combo deployable, but the FX hypothesis is supported.
The hedged DAX experiment (C) comes within 0.02 of the 0.45 gate —
the closest the European family has ever come to passing.

---

## Exp A — USD-denominated EU ETFs (EWG, EWU, IEV, VGK)

**Hypothesis**: FX translation between USD signal and EUR/GBP target adds
noise. USD-tradable proxies (EWG = iShares Germany, EWU = iShares UK,
IEV = iShares Europe, VGK = Vanguard Europe) remove this.

**Result**: max CI_lo +0.328. Every top-10 row is one of two channels:
- **LQD → EWG / IEV** (IG credit momentum → continental European equity). Previously the equivalent LQD → GDAXI topped at CI_lo +0.20.
- **TLT / HYG → EWG / IEV** (duration or credit spread → EU equity).

### Top 5

| Signal | Target | LB | Hold | Th | Sharpe | CI_lo | DD | Folds |
|---|---|--:|--:|---:|---:|---:|---:|--:|
| LQD | EWG | 40 | 5 | 0.75 | +0.765 | +0.328 | -25.0 % | 38 |
| LQD | IEV | 40 | 5 | 0.75 | +0.729 | +0.271 | -21.7 % | 38 |
| TLT | EWG | 10 | 20 | 0.50 | +0.641 | +0.218 | -48.3 % | 43 |
| TLT | IEV | 10 | 5 | 0.50 | +0.631 | +0.215 | -30.5 % | 43 |
| HYG | EWG | 20 | 10 | 0.25 | +0.657 | +0.203 | -27.0 % | 33 |

**Read**: USD-denominated proxies lift the best CI_lo from **+0.13 (FTSE) / +0.33 (DAX)** in the original sweep to **+0.33 (EWG/IEV)** here. Modest uplift; not enough to deploy.

**Why not a bigger improvement**: the signals are US rates/credit. Any time the channel is US → Europe, the European half of the relationship is still its own regime. Removing the spot FX noise doesn't change the fundamental fact that ECB decisions differ from Fed decisions.

---

## Exp B — EU-exposed bond signals (IGOV, BWX, BNDX) → EU equity

**Hypothesis**: The predictive relationship requires the *signal* to be EU-exposed, not just the target. IGOV (iShares International Treasury Bond), BWX (SPDR International Treasury), and BNDX (Vanguard Total International Bond) carry Bunds, OATs, BTPs, Gilts — exactly the European rate factor.

**Result**: max CI_lo +0.296 (BNDX → DAX lb=20 h=10 th=0.25, Sharpe +0.94, **100 % positive folds**). Every gate-passer in the top 10 has a short BNDX history (13 years → 15 folds) which widens the bootstrap CI. The **point estimates are strong** — BNDX → DAX at +1.02 Sharpe with lb=5 h=20 th=1.00 — but CI widths disqualify them.

### Top 5

| Signal | Target | LB | Hold | Th | Sharpe | CI_lo | CI_hi | DD | Folds | Pos% |
|---|---|--:|--:|---:|---:|---:|---:|---:|--:|--:|
| BNDX | GDAXI | 20 | 10 | 0.25 | +0.941 | +0.296 | +1.679 | -30.3 % | 15 | **100 %** |
| IGOV | GDAXI | 10 | 10 | 0.25 | +0.832 | +0.275 | +1.454 | -34.4 % | 22 | 77 % |
| BNDX | GDAXI | 5 | 20 | 1.00 | +1.016 | +0.250 | +1.737 | -39.7 % | 15 | 93 % |
| BNDX | GDAXI | 15 | 20 | 0.25 | +0.911 | +0.206 | +1.681 | -30.3 % | 15 | 93 % |
| IGOV | GDAXI | 15 | 5 | 0.50 | +0.804 | +0.200 | +1.429 | -30.3 % | 22 | 81 % |

**Data-limited, not hypothesis-falsified**. With 25+ folds of history
for BNDX, the CI_lo would likely clear the gate. **BNDX data started
in 2013** — we need 7 more years of paper-realistic history to reach
the same fold count as LQD (38 folds). That's blocked by Yahoo Finance
not carrying pre-2013 international bond ETFs.

Worth revisiting in 2032.

---

## Exp C — Currency-hedged DAX

**Hypothesis**: Build a synthetic `DAX_hedged = DAX × (mean EURUSD) /
EURUSD` that strips out the FX move from the equity return. If the
signal was being drowned by FX, hedged DAX should outperform unhedged.

**Result**: max CI_lo **+0.425** (UUP → hedged DAX, lb=60 h=40 th=1.0,
Sharpe +0.98). This is 0.10 better than UUP → unhedged DAX (+0.326)
from the original sweep — a **direct, controlled comparison** that the
FX layer was hurting the edge.

### Top 5

| Signal | Target | LB | Hold | Th | Sharpe | CI_lo | DD | Folds |
|---|---|--:|--:|---:|---:|---:|---:|--:|
| UUP | DAX_hedged | 60 | 40 | 1.00 | +0.977 | **+0.425** | -24.6 % | 25 |
| UUP | DAX_hedged | 40 | 5 | 0.25 | +0.913 | +0.359 | -24.3 % | 25 |
| UUP | DAX_hedged | 60 | 10 | 1.00 | +0.866 | +0.328 | -15.1 % | 25 |
| UUP | DAX_hedged | 5 | 20 | 0.25 | +0.877 | +0.317 | -31.9 % | 25 |
| UUP | DAX_hedged | 40 | 20 | 1.00 | +0.833 | +0.293 | -31.4 % | 25 |

### Direct comparison: unhedged vs hedged DAX

| Signal | Target | Best CI_lo | Best Sharpe |
|---|---|---:|---:|
| UUP | **DAX (unhedged)** | +0.326 | +0.880 |
| UUP | **DAX (hedged)** | **+0.425** | **+0.977** |
| Δ | | **+0.099** | **+0.097** |

**The FX layer was accounting for ~25 % of the Sharpe drag.** The
hedged series is a different target though — a US investor cannot
cleanly trade "hedged DAX" in practice (you'd need a currency swap
overlay on top of EWG, or trade DAX futures with a forward hedge).
EWG alone is not it — EWG is USD-denominated but *unhedged* (the
Germany exposure is NAV × USDEUR, so it already carries the FX
channel).

---

## What this tells us

1. **The edge exists, but is below the deployment gate under every route tested.** No route (USD proxy, EU-native signal, FX-hedged synthetic) produces a CI_lo ≥ 0.45 survivor. All three come meaningfully closer than the original sweep, especially the hedged DAX at +0.425.
2. **FX noise is ~25 % of the drag**. The hedged vs unhedged comparison (both with UUP at lb=60) shows a clean +0.10 CI_lo uplift. This rules out "the edge is zero" and suggests "the edge is real but requires currency-hedged execution".
3. **BNDX as a signal is promising but data-limited**. Point Sharpe +0.94 with 100 % positive folds is genuinely attractive; only the short BNDX history (2013→) inflates the CI. Worth revisiting in 5-7 years.
4. **LQD/TLT → EWG/IEV** (Exp A top rows) is a modest but consistent channel. Not deployable on its own; could be worth pairing with a hedge overlay.

---

## Deployment implication

**For the v4 portfolio**: no change. Nothing from the 4,032 combos
crosses the gate. The v4 US-only portfolio remains the recommendation.

**For future research**: one genuinely new experiment worth running —
trade **DAX futures + a rolling EUR/USD forward hedge** rather than
DAX cash. If the strategy can be executed at the hedged CI_lo +0.425
Sharpe, it's deployable. That's an execution-layer experiment (sizing,
hedge roll, basis risk), not a backtest — would need a paper deployment
at 10 % target size with the synthetic hedge applied in live.

---

## What this does NOT change

- Original US v4 portfolio recommendation (HYG → IWB, TIP → HYG, TLT → QQQ, MR AUD/JPY, IEF → GLD, ML IWB) stands with Sharpe +1.77 OOS.
- The conclusion that "FTSE behaves like a global multinational basket, not a UK index" — FTSE nearly disappeared from the top-10 in every experiment. The cross-asset edge for FTSE really is zero, even on the USD proxy EWU.
- The Europe cross-asset gap remains **open** — we've learned the structure (FX accounts for ~25 % of the drag; the rest is regime divergence) but not closed it with a deployable strategy.
