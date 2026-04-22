# Phase 3 — Country Rotation — 2026-04-22

Phase 3 of the [EU Strategy Plan 2026-04-22.md](./EU%20Strategy%20Plan%202026-04-22.md).
Cross-sectional developed-market country momentum, 8-ETF universe
(EWG Germany, EWU UK, EWP Spain, EWI Italy, EWQ France, EWY S.Korea,
EWC Canada, EWJ Japan). 81 configs tested.

**Driver**: [scripts/rerank/run_phase3_country_rotation.py](../scripts/rerank/run_phase3_country_rotation.py)
**Framework**: [research/cross_sectional/country_momentum.py](../research/cross_sectional/country_momentum.py) (new, 160 LOC)
**Raw**: [.tmp/reports/phase3_country_rotation_2026_04_22/results.csv](../.tmp/reports/phase3_country_rotation_2026_04_22/results.csv)

---

## Result: 0 Bonferroni survivors, max CI_lo **+0.054**

The country momentum factor (Asness-Moskowitz, classical academic
finding) **does not clear the deployment gate** on this universe.

### Top 10 by CI_lo

| # | LB | Top-k | Bot-k | Rebal | Direction | Sharpe | CI_lo | DD | Folds | Pos% | Trades |
|--:|--:|--:|--:|--:|---|---:|---:|---:|--:|--:|--:|
| 1 | 63 | 3 | 0 | 63 | **long-only** | +0.460 | +0.054 | -61.6 % | 23 | 69 % | 89 |
| 2 | 63 | 1 | 0 | 63 | long-only | +0.445 | +0.041 | -56.9 % | 23 | 73 % | 76 |
| 3 | 63 | 2 | 0 | 63 | long-only | +0.418 | +0.019 | -62.7 % | 23 | 69 % | 88 |
| 4 | 126 | 3 | 0 | 42 | long-only | +0.412 | -0.001 | -61.4 % | 23 | 65 % | 109 |
| 5 | 126 | 3 | 0 | 21 | long-only | +0.414 | -0.004 | -63.0 % | 23 | 65 % | 179 |

**Every top-10 row is long-only** (`bottom_k = 0`). The short leg of country momentum loses money on this universe — which mirrors the broader literature finding that international short-momentum has been dead since ~2010 (the "slow-moving-capital" explanation: cross-border shorts are expensive/hard, so the asymmetry between long and short has collapsed).

### Direction-split Sharpe distribution

| Direction | Configs | Median Sharpe | Max Sharpe | Median CI_lo |
|---|--:|---:|---:|---:|
| long-only (bk=0) | 27 | **+0.35** | +0.46 | −0.05 |
| long-short (bk=1 or 2) | 54 | −0.18 | +0.03 | −0.56 |

A 0.53-Sharpe spread between long-only and long-short. This is the
single clearest signal of the whole EU experiment series: **the long
leg of country momentum is real; the short leg isn't**.

---

## Why the long-only result still doesn't pass the gate

- **Max DD −62 %** for the best config. 12-month lookback equity ETFs
  are structurally high-beta to the global risk cycle; a top-3 long
  portfolio of them still suffers a 2008-style drawdown.
- **23 folds** is enough for a bootstrap CI; the CI lower bound at
  +0.054 means under the null the point estimate +0.46 is just barely
  distinguishable from zero. Not a sample-size issue — the point
  estimate is genuinely too low.
- Compare to TLT→QQQ in the v4 portfolio: Sharpe +1.33, CI_lo +0.52
  at a similar fold count. The country momentum long-only edge is
  **3x weaker** than the cross-asset bond-momentum family.

---

## What this does NOT close

- **Sector-level rotation** — not tested here. An intra-Europe sector
  rotation (VGK sector ETFs like EUFN) might behave differently.
- **Intra-country carry tilts** — country momentum combined with a
  currency-carry overlay is academically-validated to improve Sharpe
  by ~0.3. Untested on this universe.
- **Shorter-duration momentum** — we tested 3m/6m/12m; the literature
  suggests 1-month reversal + 12-month momentum hybrid can be stronger.

---

## Decision

**No addition to v4 portfolio.** The long-only top-3 config (+0.46
Sharpe, -62 % DD, CI_lo +0.054) is not deployable, and the
long-short version is negative.

The `research/cross_sectional/country_momentum.py` framework is
reusable — any future cross-sectional rotation experiment (sector,
intra-country, carry overlay) can plug into it with just a different
universe dict.

---

## What this does NOT change

- v4 portfolio (HYG→IWB, TIP→HYG, TLT→QQQ, MR AUD/JPY, IEF→GLD, ML IWB) stays intact.
- Phase 2 (EU FX MR on EUR crosses) is a different strategy family and runs next.
- Phase 4 (EU-native bonds via IBKR Eurex) remains a live option — no country-momentum result touches that hypothesis.
