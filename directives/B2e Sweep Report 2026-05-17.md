# B2e Audit Report — Sweep + L52-override audit (full closure)

**Date:** 2026-05-17
**Persona:** Researcher
**Context:** Following the data acquisition wave that added ZN/ZB/6E/6J via Databento (completing an 11-symbol IBKR-quality cross-asset basket), this report documents (a) the L52-mandated IS-only sweep and (b) the full V3.6 audit run at operator request despite the sweep's no-plateau finding.

**Final closure verdict:** No cell deployment-eligible. B2 family closed. Failure mode = **noise fragility**, not regime artifact (different from B2 sample-size or B2b yfinance contamination).

## Setup

- **Universe (11 symbols, 4 asset classes):** ES, NQ (equity); CL, BZ, HG, SI, GC (commodity); ZN, ZB (bond); 6E, 6J (FX).
- **Common window:** 2017-06-01 → 2026-05-15 (9 years).
- **IS slice:** 2017-06-01 → 2025-05-15 (2,476 bars).
- **Sanctuary held out:** 2025-05-15 → 2026-05-15 (314 bars) — untouched.
- **Sweep:** single-speed Carver EWMAC, fast_hl ∈ {2, 4, 8, 16, 32, 64, 128} with slow_hl = 4×fast_hl. Monthly rebalance, net of 1bp + $1/leg costs.
- **Plateau-detect gate:** spread ≤ 30% (L27 default), min_neighbours = 2.

## Result

| fast_hl | slow_hl | IS Sharpe | vol (ann) |
|---|---|---|---|
| 2 | 8 | +0.0303 | 0.0853 |
| 4 | 16 | +0.3821 | 0.0847 |
| **8** | **32** | **+0.5046** | 0.0814 |
| 16 | 64 | +0.3987 | 0.0776 |
| 32 | 128 | +0.3194 | 0.0734 |
| 64 | 256 | +0.0650 | 0.0681 |
| 128 | 512 | -0.1284 | 0.0628 |

Sharpe surface is **single-peaked at fast_hl = 8** with steep dropoff on both sides. Soft "plateau" {4, 8, 16, 32} has mean 0.40, range 0.18, **spread ≈ 45%** — fails the L27 ≤30% gate.

**Plateau detection:** NO candidate satisfies the (spread ≤ 30%, min_neighbours = 2) gate. The peak (8, 32) is a single-cell maximum, not a plateau centre.

## Comparison to prior B2 audits

| Audit | Universe | n inst | Window | C1 (16/64,32/128,64/256) Sharpe | Plateau spread | Verdict |
|---|---|---|---|---|---|---|
| **B2** (2026-05-15) | IBKR commodities | 24 | 3y | +2.02 | 15.3% | CI_lo<0, tier=unconfirmed |
| **B2b** (2026-05-16) | yfinance broad | 31 | 21y | -0.28 | 37.1% | RETIRED (L48) |
| **B2e** (this sweep) | IBKR cross-asset | 11 | 9y | +0.40 (single-speed) | 45% | **L52 PRE-PRE-FLIGHT FAIL** |

The IBKR cross-asset 9y result independently confirms B2b's L48 RETIRE on different data (Databento continuous, no yfinance/L40). This rules out the alternative "yfinance contamination caused the B2b failure" hypothesis.

## Interpretation

1. **B2's +2.02 was a narrow-regime commodity-trend artifact (L48 confirmed).** When the universe broadens past commodities, the EWMAC signal does not retain enough Sharpe to clear the plateau gate.
2. **The diversification benefit goes the wrong way.** Cross-asset diversification across equity + commodity + bond + FX adds noise rather than signal averaging. EWMAC's per-instrument vol-normalisation cannot rescue a basket where 6 of 11 instruments have weak own-asset trend.
3. **Carver's published results assume ~30-50 instruments with deep history.** With 11 instruments and 9 years, even the well-curated universe doesn't reach Carver's threshold for ensemble robustness.

## L52 override: full V3.6 audit run anyway

At operator request, the full audit (`research/ewmac/run_b2e_audit.py`) was executed on the held-out sanctuary + 30-fold WFO, applying the standard 5-axis decision matrix to the same C1-C8 cells as B2 + B2b. Plateau still failed (50.85% spread on OOS, even higher than the IS sweep). Headline per-cell:

| Cell | OOS Sharpe | CI95 lo | CI95 hi | DSR | MC P(>35%) | Sanc Sharpe | Noise base | Noise axis | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| C1_canonical | +0.4863 | -0.008 | +1.002 | 1.0000 | 0.0600 | **+1.0934** | +0.3231 | mid | TIER_UNCONFIRMED |
| C2_short_speeds | +0.5933 | **+0.114** | +1.049 | 1.0000 | 0.0450 | +1.0803 | +0.5097 | mid | CONDITIONAL_WATCHPOINT |
| C3_long_speeds | +0.2969 | -0.222 | +0.786 | 1.0000 | 0.0450 | +1.1412 | +0.1349 | mid | TIER_UNCONFIRMED |
| C4_full_six | +0.5451 | **+0.079** | +1.033 | 1.0000 | 0.0150 | +1.1829 | +0.4097 | mid | CONDITIONAL_WATCHPOINT |
| C5_two_speed | +0.4610 | -0.014 | +0.979 | 1.0000 | 0.0350 | +1.1275 | +0.3025 | worst | TIER_UNCONFIRMED |
| C6_singleton_canonical | +0.4823 | -0.019 | +1.008 | 1.0000 | 0.0750 | +0.9811 | +0.3168 | mid | TIER_UNCONFIRMED |
| C7_full_six_unclipped | +0.3168 | -0.189 | +0.805 | 1.0000 | 0.4550 | +1.3168 | +0.2234 | **best** | TIER_UNCONFIRMED |
| C8_gross_no_costs | +0.5394 | **+0.044** | +1.060 | 1.0000 | 0.0450 | +1.1508 | +0.3749 | best | **DEPLOY** |

**Three observations the L52 sweep missed.**

1. **Signal is positive on every cell** (OOS Sharpe +0.30 to +0.59). The plateau "failure" is not a sign reversal — it's gradient steepness in a single-peaked surface where the peak is fast-side, not B2's canonical centre.
2. **Sanctuary outperforms OOS** (+0.98 to +1.32 vs +0.30 to +0.59). The last 12 months are not a regime break; they're a regime favourable for short-speed EWMAC. This is the OPPOSITE of B2b's universal-trend-decline thesis.
3. **C8 (gross) DEPLOYs with CI_lo +0.044.** The underlying edge is real; the costs eat ~0.05 Sharpe (C8 − C1 = +0.05). C2 and C4 also have CI_lo > 0 net of costs.

**Why no cell promotes under the strict V3.7 rule:** the deployment-eligible filter requires either (a) DEPLOY verdict (C8 only, but it's a baseline) or (b) CONDITIONAL_WATCHPOINT with `noise_axis == "best"` (C2 and C4 have noise_axis = "mid" — `mean_pass=True, worst_pass=False`). The failure mode is **noise fragility**: worst-case 50% noise injection knocks degradation above the 30% threshold for the canonical cells.

## Decision (revised after audit)

**No cell promoted to live deployment.** B2 family closed.

But the failure mode is qualitatively different from prior closures:

| Audit | Closure mechanism |
|---|---|
| B2 (IBKR-3y commodities) | CI_lo bottleneck (sample size, L46) |
| B2b (yfinance broad 21y) | Sign reversal (L48 regime artifact, possibly L40-aggravated) |
| **B2e (IBKR x-asset 9y)** | **Noise fragility — signal positive but worst-case noise robustness fails** |

The B2e result independently **falsifies** the L48 "EWMAC universally decayed" thesis on clean Databento data. The signal is there; it just doesn't survive the V3.7 worst-case noise gate at retail-cost levels. Two paths forward for any future revival:

- **Cost-quality improvement.** C8 vs C1 gap = +0.05 Sharpe (small). A lower-cost broker or larger notional would close it; the underlying edge clears CI_lo at gross. Not deployment-actionable today.
- **Per-asset regime gating.** Noise fragility on a heterogeneous 4-asset-class universe suggests a regime filter could selectively trade only the cells where EWMAC works. **This is exactly the I1 hypothesis** — HMM per-asset regime detection. The B2e result is the strongest motivation yet for I1.

**B2 family officially closed.** Variants tried:
- B2 (commodity-only, 3y): blocked by CI_lo
- B2b (yfinance broad, 21y): RETIRED L48
- B2c (broad-trend gate): closed per directive (separate work)
- B2d (vol-regime gate): closed per directive (separate work)
- **B2e (IBKR x-asset, 9y): RETIRED at L52 sweep — no plateau**

## Recommended next steps

1. **I1 — HMM regime + EWMAC gate.** The remaining trend-research avenue. Per-asset regime detection might filter EWMAC signals down to high-conviction windows. Data is ready (`data/i1_regime_panel.parquet`). Effort: 3 days.
2. **Update Retirement Registry** with the cross-universe B2e RETIRE finding.
3. **Update V3.6 Lessons Catalogue L48** with the confirmation that L48 generalises across data sources.
4. **Optional: a portfolio-of-many-assets-WEIGHTED-by-asset-class-correlation Carver basket.** Out of scope for V3.7. Would require 30+ instruments and per-asset-class FDM, materially different from the V3.6 framework.

## Artefacts

- Sweep code: `research/exploration/sweep_b2e_ibkr_xasset.py`
- Sweep report: `.tmp/reports/sweep_b2e_ibkr_xasset/plateau_report.md`
- Cells CSV: `.tmp/reports/sweep_b2e_ibkr_xasset/cells_long.csv`

This document closes the B2 family. The Strategy Backlog is being updated to reflect the closure.
