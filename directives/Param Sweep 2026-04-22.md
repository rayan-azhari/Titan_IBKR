# Cross-Asset Parameter Sweep — 2026-04-22

Full-grid parameter sweep with Bonferroni-corrected multiple-testing
discipline. Tests every (signal × target × lookback × hold × threshold)
combination in the cross-asset momentum family.

**Driver**: [scripts/rerank/run_param_sweep.py](../scripts/rerank/run_param_sweep.py)
**Raw results**: [.tmp/reports/param_sweep_2026_04_22/results.csv](../.tmp/reports/param_sweep_2026_04_22/results.csv)
**Raw leaderboard**: [.tmp/reports/param_sweep_2026_04_22/leaderboard.md](../.tmp/reports/param_sweep_2026_04_22/leaderboard.md)

**Scope**: 6 signals (TLT, IEF, HYG, TIP, LQD, UUP) × 6 targets (SPY, QQQ,
IWB, GLD, HYG, TQQQ) × 6 lookbacks × 4 holds × 4 thresholds = 3,360 combos.
Sanctuary window active; 95 % bootstrap Sharpe CI per combo. Wall-clock
273 s.

---

## Multiple-testing discipline

| Gate | Criterion | Passers | Interpretation |
|---|---|--:|---|
| Permissive | CI_lo > 0, folds ≥ 30, pos_folds ≥ 60 % | **1,448** | At N=3,360, ~84 false positives expected per tail by chance |
| **Bonferroni** | CI_lo ≥ 0.5, folds ≥ 30, pos_folds ≥ 60 %, DD ≥ -40 % | **20** | Genuine discoveries — point estimate high enough that the ~2.1× CI widening from Bonferroni still leaves the lower bound positive |

The gap (1,448 → 20) is exactly the multiple-testing inflation the April
2026 audit warned about. Before the audit we had ~50 "champion" strategies,
many now failing. The sweep confirms the discipline we need: **anything
on the permissive list but not the Bonferroni list is not deployable**.

---

## The 20 survivors

Ranked by CI_lo:

| # | Signal | Target | LB | Hold | Th | Sharpe | CI_lo | Max DD | Folds | Pos% |
|--:|---|---|--:|--:|---:|---:|---:|---:|--:|--:|
| 1 | HYG | QQQ | 20 | 5 | **0.25** | +1.058 | **+0.606** | -15.7 % | 33 | 79 % |
| 2 | HYG | IWB | 10 | 20 | 0.25 | +1.095 | +0.595 | -19.9 % | 33 | 91 % |
| 3 | TIP | HYG | 60 | 40 | 0.25 | +1.054 | +0.589 | **-10.1 %** | 33 | 79 % |
| 4 | HYG | SPY | 10 | 20 | 0.25 | +1.096 | +0.588 | -19.8 % | 33 | 91 % |
| 5 | TIP | HYG | 60 | 10 | 0.25 | +1.058 | +0.581 | -10.1 % | 33 | 79 % |
| 6 | HYG | QQQ | 60 | 10 | 0.25 | +1.072 | +0.575 | -20.6 % | 33 | 67 % |
| 7 | TIP | HYG | 60 | 5 | 0.25 | +1.035 | +0.559 | -10.5 % | 33 | 79 % |
| 8 | HYG | QQQ | 10 | 40 | 0.25 | +1.033 | +0.549 | -25.2 % | 33 | 88 % |
| 9 | TIP | HYG | 60 | 20 | 0.25 | +1.023 | +0.549 | -10.1 % | 33 | 70 % |
| 10 | HYG | QQQ | 10 | 5 | 0.25 | +1.061 | +0.548 | -14.7 % | 33 | 91 % |
| 11 | TLT | QQQ | 10 | 20 | **0.50** | +0.978 | +0.543 | -37.0 % | 43 | 93 % |
| 12 | HYG | IWB | 60 | 10 | 0.25 | +1.036 | +0.530 | -14.4 % | 33 | 70 % |
| 13 | HYG | QQQ | 10 | 20 | 0.25 | +1.041 | +0.530 | -27.0 % | 33 | 88 % |
| 14 | HYG | IWB | 10 | 5 | 0.25 | +1.018 | +0.529 | -11.2 % | 33 | 88 % |
| 15 | LQD | HYG | 40 | 10 | 0.25 | +0.975 | +0.516 | -10.4 % | 33 | 79 % |
| 16 | HYG | QQQ | 10 | 10 | 0.50 | +1.030 | +0.514 | -21.5 % | 33 | 73 % |
| 17 | HYG | SPY | 10 | 5 | 0.25 | +1.004 | +0.513 | -11.0 % | 33 | 85 % |
| 18 | HYG | QQQ | 20 | 5 | 0.75 | +0.997 | +0.509 | -13.2 % | 33 | 73 % |
| 19 | HYG | SPY | 60 | 10 | 0.25 | +1.022 | +0.504 | -14.1 % | 33 | 70 % |
| 20 | HYG | QQQ | 20 | 10 | 0.75 | +0.967 | +0.500 | -12.7 % | 33 | 70 % |

---

## Big finding: **threshold = 0.25 dominates**

Of the 20 Bonferroni survivors, **16 use threshold = 0.25** (our prior
optimizer and the original catalogue used 0.50 as the default). This
is a significant parameter refinement hiding in plain sight:

### HYG→IWB comparison

| Config | Sharpe | CI_lo | CI_hi | Max DD |
|---|---:|---:|---:|---:|
| HYG→IWB lb=10 hold=20 **th=0.50** (prior optimizer) | +0.895 | +0.396 | +1.334 | -24.0 % |
| HYG→IWB lb=10 hold=20 **th=0.25** (sweep winner) | **+1.095** | **+0.595** | +1.524 | -19.9 % |

**+22 % Sharpe, +0.20 CI_lo, 4 pp lower DD — from a single threshold change.**

### TIP→HYG comparison

| Config | Sharpe | CI_lo | Max DD |
|---|---:|---:|---:|
| TIP→HYG lb=60 hold=20 **th=0.50** (prior autonomous loop winner) | +0.913 | +0.432 | -10.1 % |
| TIP→HYG lb=60 hold=40 **th=0.25** (sweep winner) | **+1.054** | **+0.589** | -10.1 % |

**+15 % Sharpe, +0.16 CI_lo, DD unchanged.**

### Why would threshold=0.25 help?

A lower z-score entry threshold means we trade more often on smaller
signals, which typically adds noise. It helps here because: (a) HYG/TIP
signals are relatively persistent, so a weak signal predicts next-bar
return better than the threshold suggests; (b) the hold-days gate
(5-40 bars) already filters out pure noise by requiring the signal to
hold; (c) more trades = smaller per-trade vol, which the Sharpe likes.

This was hiding because every prior sweep started at threshold=0.50 as
the default, and we never parameter-swept that dimension until now.

---

## Pattern: the cross-asset family collapses to two channels

Taxonomy of the 20 survivors:

| Channel | Members | Notes |
|---|--:|---|
| **HYG → QQQ / IWB / SPY** | 14 | Same HY-credit-spread → equity story, different targets. All three targets are ~95% correlated equities. One holding captures the channel. |
| **TIP → HYG** | 4 | Cross-bond: inflation expectations → credit spread. Distinct from HYG→equity (they measure different things). Separate holding. |
| **LQD → HYG** | 1 | Within-credit lead/lag (IG → HY). Close cousin to TIP → HYG. |
| **TLT → QQQ** | 1 | Duration → growth equity. Different story from HYG/TIP. |

**Four channel heads, not twenty strategies.** The 14 HYG→equity variants
are not 14 independent edges; they're one edge (HYG is a credit spread
indicator) measured four different ways (QQQ / IWB / SPY at various
holds). Holding all 14 is double-counting. Same logic as the v3 portfolio
analysis (three X→HYG signals share a credit factor).

---

## Refined portfolio candidates (v4)

Based on this sweep, the deployable cross-asset set narrows to **four
channel-head strategies** with their sweep-optimal parameters:

| # | Channel | Best config | Sharpe | CI_lo | Max DD |
|--:|---|---|---:|---:|---:|
| 1 | HYG → equity | HYG → IWB lb=10 hold=20 **th=0.25** | +1.095 | +0.595 | -19.9 % |
| 2 | TIP → credit | TIP → HYG lb=60 hold=40 **th=0.25** | +1.054 | +0.589 | -10.1 % |
| 3 | TLT → growth | TLT → QQQ lb=10 hold=20 th=0.50 | +0.978 | +0.543 | -37.0 % |
| 4 | LQD → credit | LQD → HYG lb=40 hold=10 **th=0.25** | +0.975 | +0.516 | -10.4 % |

**Caveat**: channel 4 (LQD → HYG) is 0.67 correlated with channel 2
(TIP → HYG) per the v3 optimizer analysis. Keep only one in the live
portfolio — TIP → HYG wins on every metric except parameter robustness
(LQD-HYG lb=40 is isolated in the sweep; TIP-HYG lb=60 has 4 of the
20 rows all agreeing).

**Live v4 recommendation**: channels 1, 2, 3 only. Equal-weight them
inside the cross-asset bucket, then combine with ML IWB stacking and
MR AUD/JPY per the prior v3 analysis.

---

## What the sweep did NOT find

Even with 3,360 combinations, the Bonferroni gate admitted zero new
signal families. All 20 survivors were refinements of already-known
cross-asset edges. Interpretation: **the cross-asset edge landscape is
small and well-covered.** No hidden gems found. This is actually
informative — it means additional sweeps in this dimension are unlikely
to surface anything new.

What we did NOT test (possible future work):
- **Indicator parameter sweeps on single-instrument strategies** (e.g.,
  MR on AUD/JPY at non-default RSI periods, MA windows — the existing
  sweep uses the catalogue defaults).
- **Regime-conditioned variants** (e.g., "HYG→QQQ only when VIX > 20").
  Would require new strategy code, not just parameter tuning.
- **Non-linear signal combinations** (AND-gates, ratios). Likewise.
- **Sector-ETF targets** (XLK, XLE, XLF, etc.) — data missing from
  the local parquet directory.

---

## Actionable changes for the live portfolio

1. **Upgrade HYG → IWB threshold** from 0.50 → 0.25 in the live config
   (`scripts/run_portfolio.py` `bond_equity_ihyu_cspx` block or whichever
   currently holds the HYG → IWB parameter).
2. **Upgrade TIP → HYG threshold** from 0.50 → 0.25 (if/when TIP→HYG is
   promoted from v3 candidate to live). Also bump hold_days from 20 → 40.
3. **Keep TLT → QQQ at its current threshold=0.50** — it's the only
   survivor that prefers 0.50, so don't override.
4. **Do NOT add LQD → HYG** to the live portfolio even though it passes
   the gate — TIP → HYG captures the same credit-factor with better
   parameter stability.

Once deployed on paper, the 4 live strategies (MR AUD/JPY, HYG→IWB,
TIP→HYG, TLT→QQQ) plus ML IWB would form the post-audit champion
portfolio. Expected OOS Sharpe from equal-weight on the common window:
we don't have a realized number for this exact mix because TIP→HYG
th=0.25 and HYG→IWB th=0.25 are both new configurations — would need
to re-run `scripts/rerank/optimize_v3_portfolio.py` with the refined
parameters.
