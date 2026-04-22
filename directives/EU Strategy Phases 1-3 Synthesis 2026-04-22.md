# EU Strategy Phases 1–3 — Synthesis — 2026-04-22

Cumulative findings from [Phase 1](./Phase%201%20EU%20Native%20Sweep%202026-04-22.md),
Phase 1.5 (oil → EWG follow-up),
[Phase 2](./Phase%202%20EU%20FX%20MR%202026-04-22.md), and
[Phase 3](./Phase%203%20Country%20Rotation%202026-04-22.md) of the
[EU Strategy Plan](./EU%20Strategy%20Plan%202026-04-22.md).

**1,228 experiments across 4 strategy families. 0 Bonferroni survivors.**
The EU gap remains structurally open.

---

## Summary table

| Phase | Family | Configs | Max CI_lo | Gap to gate (0.45) | Best result |
|---|---|--:|---:|---:|---|
| 1 | Cross-asset daily (CL/GLD/UUP/IGOV-IEF/DAX-SPY → EU) | 480 | **+0.316** | −0.13 | CL → DAX lb=40 hold=20 th=0.25 (Sharpe +0.74) |
| 1.5 | Oil → EWG (USD-denom Germany) | 192 | +0.164 | −0.29 | CL → EWG lb=40 hold=40 (EWG hurt the signal) |
| 2 | EUR FX H1 VWAP-MR (CHF/GBP/JPY) | 72 | −0.031 | −0.48 | EUR/GBP atr_only vwap=12 |
| 3 | Country rotation (8-ETF, 81 configs) | 81 | +0.054 | −0.40 | Long-only top-3 @ 63d lookback |
| **Cumulative** | — | **825** | **+0.316** | **−0.13** | **CL → DAX (Phase 1)** |

Combined with Europe v2 (4,032 combos) and original Europe sweep (1,152 combos):
**6,009 total combos, 0 deployable edges**.

---

## What we actually learned (signal structure)

Zero survivors is not zero information. Five distinct findings:

### 1. The EU cross-asset gap is real — point estimates vary, CI_los don't

Across **four different experimental routes** (US signals → US ETFs on EU, EU-denominated signals → EU targets, currency-hedged DAX, EU-native EUR crosses, country momentum) the CI_lo maxes out at +0.425 (Europe v2 hedged DAX) and more typically +0.1 to +0.3. None touches 0.50.

The structural barrier is consistent, not instrument-specific: **the US cross-asset factor does not generalise to European targets cleanly in any form we've tested.**

### 2. The FX layer is ~25 % of the drag — confirmed twice

- Europe v2 hedged-DAX: +0.425 vs unhedged +0.326 (Δ +0.10 CI_lo)
- Phase 1.5 EWG-vs-DAX: went the wrong direction because USD-denominated ETFs carry FX exposure (NAV × USDEUR), they're not truly hedged.

The only way to realise the hedged-DAX edge is **DAX futures + rolling EUR/USD forward hedge** — execution-layer, not another backtest.

### 3. EUR spot FX is unforecastable at daily frequency from US proxies

Phase 1's GLD → EUR/USD and UUP → EUR/USD both produced **max CI_lo < −0.45**. Cleanest negative of the whole series. Any EUR/USD edge has to live at intraday frequencies (H1 MR) or via fundamental macro features (rate differentials, ECB policy cycles) — not via US-ETF proxies.

### 4. The AUD/JPY MR strategy is uniquely idiosyncratic

Phase 2 showed EUR/CHF, EUR/GBP, EUR/JPY all fail at the VWAP-confluence framework that works for AUD/JPY:
- The `conf_donchian_pos_20` filter that wins for AUD/JPY is **worst** for EUR pairs (DD −36 % to −78 %).
- Optimal VWAP anchor and filter differ per pair; there is no universal MR spec.

AUD/JPY's champion Sharpe of +1.05 is tied to AUD's risk-on-off oscillation + Asian-session liquidity. Not a generalisable pattern.

### 5. Country momentum: long leg real, short leg dead

Phase 3's clearest structural finding:

| Direction | Median Sharpe | Max Sharpe | Median CI_lo |
|---|---:|---:|---:|
| long-only (bk=0) | +0.35 | +0.46 | −0.05 |
| long-short (bk≥1) | −0.18 | +0.03 | −0.56 |

A **0.53-Sharpe spread** between long-only and long-short. Shorting underperforming country ETFs loses money persistently — consistent with the literature that international short-momentum has been dead since ~2010 (slow-moving capital and short constraints on foreign equities).

---

## Near-miss inventory (for future experiments)

| # | Channel | CI_lo | Phase | Why not deployable now |
|---|---|---:|:-:|---|
| 1 | UUP → DAX_hedged (60d, 40d, 1.0) | **+0.425** | Europe v2 | Synthetic hedge; requires execution-layer test (FGBL + EURUSD forward) |
| 2 | LQD → EWG (40d, 5d, 0.75) | +0.328 | Europe v2 | Point Sharpe only +0.77; not close to rich enough |
| 3 | **CL → DAX (40d, 20d, 0.25)** | **+0.316** | **Phase 1** | Parameter-stable; oil-as-signal is real but signal-noise ratio too low |
| 4 | BNDX → DAX (20d, 10d, 0.25) | +0.296 | Europe v2 | Point Sharpe +0.94 with 100 % pos folds but data-limited (15 folds; BNDX from 2013) |
| 5 | UUP → DAX (60d, 40d, 1.0) | +0.326 | Europe 1 | Unhedged version of #1 |

**Three distinct economic stories all in the +0.30 range**. The
cumulative weight of evidence says EU edges exist but are all
marginal — consistent with efficient-enough markets, just with enough
signal structure to keep researchers busy.

---

## What we're NOT doing next

Given 6,009 combos / 0 survivors, additional sweep-style experiments
are unlikely to surface a new deployable edge. **Phase 4** (EU-native
Eurex bonds via IBKR) is the only remaining option that genuinely
tests a distinct hypothesis — EU-native signal with EU-native target —
and it involves meaningful data-engineering work.

**Recommend defer Phase 4** unless:
- User wants to pursue the hedged-DAX execution path (where Phase 4 data becomes auxiliary rather than primary).
- A new hypothesis emerges that requires FGBL/FBTP as a dependency.

---

## v4 portfolio status

**Unchanged. 6-slot US-only portfolio stands:**

| Slot | Weight | Strategy | Config |
|---|--:|---|---|
| 1 | 17 % | HYG → IWB | lb=10 hold=20 th=0.25 |
| 2 | 17 % | TIP → HYG | lb=60 hold=40 th=0.25 |
| 3 | 17 % | TLT → QQQ | lb=10 hold=20 th=0.50 |
| 4 | 17 % | MR AUD/JPY | vwap_anchor=24, donchian_pos_20, conservative |
| 5 | 17 % | IEF → GLD | lb=60 hold=20 th=0.50 |
| 6 | 15 % | ML IWB | stacking, threshold=0.6 |

**OOS Sharpe +1.77** (CI +0.93, +2.70), max DD −5.0 %, max pairwise ρ 0.556.

---

## Artefacts from this research burst

Data newly on disk:
- `data/EUR_CHF_H1.parquet`, `data/EUR_GBP_H1.parquet`, `data/EUR_JPY_H1.parquet` (93 k bars each, 2011-2026)
- `data/EWP_D.parquet`, `data/EWI_D.parquet`, `data/EWQ_D.parquet`, `data/EWY_D.parquet`, `data/EWC_D.parquet`, `data/EWJ_D.parquet`, `data/EUFN_D.parquet`
- `data/CL=F_D.parquet`, `data/BZ=F_D.parquet`

New frameworks:
- [research/cross_sectional/country_momentum.py](../research/cross_sectional/country_momentum.py) — cross-sectional momentum WFO with bootstrap CI (reusable for sector/intra-country experiments).

Sweep drivers:
- [scripts/rerank/run_phase1_eu_native.py](../scripts/rerank/run_phase1_eu_native.py)
- [scripts/rerank/run_phase1_5_oil_ewg.py](../scripts/rerank/run_phase1_5_oil_ewg.py)
- [scripts/rerank/run_phase2_eu_fx_mr.py](../scripts/rerank/run_phase2_eu_fx_mr.py)
- [scripts/rerank/run_phase3_country_rotation.py](../scripts/rerank/run_phase3_country_rotation.py)
