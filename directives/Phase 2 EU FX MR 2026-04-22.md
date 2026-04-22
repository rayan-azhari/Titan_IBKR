# Phase 2 — EU FX H1 Mean-Reversion — 2026-04-22

Phase 2 of the [EU Strategy Plan 2026-04-22.md](./EU%20Strategy%20Plan%202026-04-22.md).
Tests the VWAP-confluence MR framework (AUD/JPY champion pattern) on
three EUR crosses after downloading 15-year H1 history from IBKR.

**Driver**: [scripts/rerank/run_phase2_eu_fx_mr.py](../scripts/rerank/run_phase2_eu_fx_mr.py)
**Raw**: [.tmp/reports/phase2_eu_fx_mr_2026_04_22/results.csv](../.tmp/reports/phase2_eu_fx_mr_2026_04_22/results.csv)

**Data**: 93,130 H1 bars each for EUR/CHF, EUR/GBP, EUR/JPY, coverage 2011-04-26 → 2026-04-22 via IBKR IDEALPRO MIDPOINT.

**Scope**: 3 pairs × 3 vwap_anchors × 4 filters × 2 tier grids = **72 configs** in 268 s.

---

## Result: 0 Bonferroni survivors, max CI_lo **−0.031**

No EUR cross produces an MR edge comparable to AUD/JPY.

### Per-pair summary

| Pair | Max Sharpe | Max CI_lo | Best config | AUD/JPY reference |
|---|---:|---:|---|---|
| **EUR/GBP** | **+0.503** | **−0.031** | vwap=12 / atr_only / conservative | AUD/JPY +1.05 CI_lo +0.21 |
| EUR/JPY | +0.180 | −0.351 | vwap=24 / conf_donchian / standard | same |
| EUR/CHF | +0.158 | −0.328 | vwap=36 / conf_donchian / standard | same |

**EUR/GBP is the least-bad**. The other two pairs show point Sharpes ~0.16 — essentially zero signal at all. And even EUR/GBP's best is 60 % below the AUD/JPY Sharpe.

### Why does the AUD/JPY pattern not transfer?

Three observations from the top 10:
1. **`conf_donchian_pos_20` — the AUD/JPY winning filter — is the worst for EUR crosses**. EUR/CHF and EUR/JPY with that filter produce DDs of −36 % to −78 %. The filter works for AUD/JPY because it catches disagreement on AUD's risk-on/off cycles; EUR doesn't have the same range-bound regime.
2. **`atr_only` is the best filter for EUR/GBP**. A regime-based filter (trade only in high-ATR periods) beats the confluence filter. This is a different market microstructure: EUR/GBP has range/trend phase clarity via volatility, not via multi-scale disagreement.
3. **Shorter VWAP anchor (12-bar = half-day) wins** on EUR/GBP. AUD/JPY wants the full-day anchor (24). EUR/GBP's mean-reversion horizon is half that of AUD/JPY — intraday deviations revert within a single session.

Taken together: the AUD/JPY strategy is idiosyncratic to AUD's Asian-session-liquidity + risk-on-off pattern. **Nothing like it exists for EUR crosses** in this time window.

---

## What this tells us about EUR FX

The cleanest way to read this result: **EUR crosses are dominated by
trend, not mean-reversion**, at H1. AUD/JPY oscillates; EUR/GBP
trends through regimes (Brexit, 2022 energy crisis). VWAP-MR is
the wrong strategy family.

**Worth trying** in follow-up research (not this phase):
- **Trend-following** on the same EUR crosses (Donchian breakout or Keltner-channel trend, NOT MR).
- **Different mean-reversion spec** — z-score-based rather than VWAP-based, with longer rebalance periods.
- **Session-conditional MR** — only trade EUR/GBP during the overlap of London+NY sessions (high liquidity, range-bound) and skip Asian-session hours.

---

## Decision

**No addition to v4 portfolio.** EUR/GBP at CI_lo −0.031 is not
deployable, and further tuning is unlikely to produce a gate-passer
(we've already tested 24 config combinations per pair).

The 15-year H1 history for all three pairs is now on disk
(`data/EUR_CHF_H1.parquet`, `data/EUR_GBP_H1.parquet`,
`data/EUR_JPY_H1.parquet`) and available for follow-up experiments.

---

## What this does NOT change

- v4 portfolio (HYG→IWB, TIP→HYG, TLT→QQQ, MR AUD/JPY, IEF→GLD, ML IWB) remains the champion.
- MR AUD/JPY's champion status is not contested — the opposite, it's **reinforced**: no other carry-pair MR variant comes close, so the AUD/JPY edge is uniquely tied to that instrument's market structure.
- Phase 4 (EU-native Eurex bonds) is unchanged — this is a bonds/rates hypothesis, independent of EUR spot FX.
