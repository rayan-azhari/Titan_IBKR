# Phase 1 — EU-Native Free Cross-Asset Sweep — 2026-04-22

Phase 1 of the [EU Strategy Plan 2026-04-22.md](./EU%20Strategy%20Plan%202026-04-22.md).
5 "free" channels tested with already-downloaded data via the standard
`run_bond_wfo` harness. 480 total combos in 28 s wall-clock.

**Driver**: [scripts/rerank/run_phase1_eu_native.py](../scripts/rerank/run_phase1_eu_native.py)
**Raw**: [.tmp/reports/phase1_eu_native_2026_04_22/results.csv](../.tmp/reports/phase1_eu_native_2026_04_22/results.csv)
**Leaderboard**: [.tmp/reports/phase1_eu_native_2026_04_22/leaderboard.md](../.tmp/reports/phase1_eu_native_2026_04_22/leaderboard.md)

---

## Headline: 0 Bonferroni survivors. **But the per-channel signal is very informative.**

| Channel | Hypothesis | Max Sharpe | Max CI_lo | CI_lo > 0 combos | Verdict |
|---|---|---:|---:|--:|---|
| **CL → DAX** | Oil shock → DAX (net energy importer) | +0.735 | **+0.316** | 31 / 96 | **Real but weak** |
| **DAX/SPY ratio → DAX** | Relative-momentum continuation | +0.680 | +0.302 | 39 / 96 | **Real but weak** |
| IGOV-IEF → EURUSD | International-vs-US rate premium → FX | +0.378 | −0.186 | 0 / 96 | Weak negative |
| GLD → EURUSD | Gold rally → USD weakness → EUR up | +0.036 | −0.461 | 0 / 96 | **Strong negative** |
| UUP → EURUSD | Dollar strength → EUR/USD momentum | +0.029 | −0.506 | 0 / 96 | **Strong negative** |

---

## What this tells us — cleanly separates the 5 hypotheses into 3 buckets

### Bucket 1 — Real but not deployable (CL → DAX, DAX/SPY → DAX)

These two channels had **roughly 1/3 of combos** with CI_lo > 0 and point Sharpes in the +0.6-0.7 range. Oil → DAX at CI_lo **+0.316** is the closest we've come to clearing a gate on any EU channel.

**Best configs**:

| Channel | LB | Hold | Th | Sharpe | CI_lo | Max DD | Folds | Pos% |
|---|--:|--:|---:|---:|---:|---:|--:|--:|
| CL → GDAXI | 40 | 20 | 0.25 | +0.735 | +0.316 | -28.4 % | 35 | 74 % |
| DAX/SPY → GDAXI | 60 | 20 | 0.50 | +0.680 | +0.302 | -39.7 % | 47 | 66 % |
| CL → GDAXI | 40 | 20 | 1.00 | +0.704 | +0.257 | -14.2 % | 35 | 65 % |
| CL → GDAXI | 15 | 40 | 0.50 | +0.707 | +0.228 | -36.6 % | 35 | 85 % |

Consistency pattern for oil → DAX: **lb=40 hold=20** wins at multiple thresholds (rows 1, 3, 6, 7, 8, 10 of the top 10). This is a parameter-stable result — not cherry-picked — and the story is clean (40-day oil trend predicts 20-day DAX move via the energy-cost channel).

**Why it doesn't pass the gate**: 35 folds × −0.184 bootstrap widening reaches the +0.316 lower bound but can't break +0.45. The point estimate +0.74 is real; the sample just isn't rich enough to give it a tight CI. This is the same problem as BNDX → DAX in Europe v2 — point Sharpe is genuine, CI width blocks deployment.

### Bucket 2 — Weak negative (IGOV-IEF → EURUSD)

The rate-differential-predicts-FX channel shows zero positive-CI combos but the point Sharpe max is +0.38. Some signal exists but direction is inconsistent across configs.

### Bucket 3 — **Strong negatives** (UUP → EURUSD, GLD → EURUSD)

Both have **max CI_lo < -0.4** with max Sharpe < +0.05 — these are genuinely predictively zero.

The practical implication: **EUR/USD is not a forecastable series at daily frequency from dollar or gold momentum**. These are the cleanest negative results of any experiment in the April 2026 research series. The EUR/USD edge, if it exists, lives at intraday frequencies (where the H1 MR strategies live) — NOT at daily cross-asset level.

---

## Decisions for Phase 2

**Given Phase 1 findings, Phase 2 remains worth running**:

- The Phase 1 negatives for UUP/GLD → EURUSD are **daily-frequency findings**. Phase 2 tests H1 mean-reversion on EUR crosses (EUR/CHF, EUR/GBP, EUR/JPY) which is a different strategy family entirely — intraday MR with VWAP confluence, not daily cross-asset.
- Phase 1's best survivor (CL → DAX) suggests there's **some** EU-local edge to be found if we give it more data. Phase 4 (Eurex futures, Bund yields) becomes more interesting because the "weak but real" pattern says the structural edge exists — it just needs native-frequency, native-instrument signals.

**Phase 3 (country rotation)** is still high-probability worth running: zero of the 5 channels tested here covered the cross-sectional momentum factor, which is the best-documented EU edge in the literature.

---

## Near-miss worth one follow-up experiment

**Oil → DAX at lb=40 hold=20** is consistent enough to warrant a bounded follow-up: **extend the signal set with Brent (BZ=F)** and test oil → EWG (USD-denominated Germany) instead of raw local-currency DAX. If EWG lifts CI_lo meaningfully (same pattern as the Europe v2 hedged-DAX result), we could get to Bonferroni via a USD-tradable target.

Quick cost: ~3 min wall-clock, 192 extra combos.

**Recommendation**: run this as an extended Phase 1.5, then decide whether to proceed to Phase 2 (FX downloads) or Phase 3 (country rotation) based on outcome.

---

## What this does NOT change

- v4 portfolio stays US-only (no new deployable EU strategies from Phase 1).
- The hedged-DAX hypothesis from Europe v2 still stands as the most promising near-miss (CI_lo +0.425, 0.025 below gate).
- Phases 2, 3, 4 from the EU plan are unchanged — Phase 1 was designed as the cheap binary signal, and the signal is "some weak edge exists, the gate remains unbreached".
