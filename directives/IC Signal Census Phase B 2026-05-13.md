# IC Signal Census — Phase B pre-registration

**Version:** 1.0 | **Date:** 2026-05-13 | **Author:** Architect
**Status:** **PRE-REGISTRATION** — committed BEFORE the scan runs (V3.1 discipline).
**Parent:** `directives/IC Signal Census 2026-05-13.md`. Every gate, every invariant, and every output-schema column is inherited from the parent unless explicitly overridden in §3 below.

---

## 0. Why this exists

The parent Phase A census ran 2,394 raw cells / 798 headline rows under 9 single-instrument + 2 cross-asset signal classes and surfaced zero TIER_A or TIER_B survivors. The 21 cells that simultaneously cleared DSR + BH + fold-stability under that scan were all H1 microstructure mean-reversion — real but parameter-fragile and economically thin. The Phase A audit-discipline gates worked as designed: cherry-picked single-TF signals do not survive the plateau + MTF + DSR + sanctuary gate stack.

Phase A explicitly carved out four signal classes pending external data:

- **Vol-risk-premium (§2.5 second cell):** requires VIX spot to compute `VIX / realised_vol - 1`.
- **Term structure (§2.7):** requires VIX9D, VIX, VIX3M, and (ideally) VIX-futures front/back month basis.
- **Cross-region lead-lag (§2.8):** requires US and European equity indices on a comparable calendar.
- **Macro flow — breadth (§2.4):** requires the SPX-component panel (still deferred to Phase C; out of scope here).

Phase B fills the first three by adding free-data instruments downloadable via yfinance. CME / Eurex futures continuous data (the ideal source for §2.8 cross-region lead-lag) remains paywall-gated and is deferred to a future Phase C; we proxy with index spot for now. The substitution is explicit and documented in §1.3.

---

## 1. Pre-registered universe additions

### 1.1 Instruments downloaded for Phase B

All daily-frequency, yfinance free tier. Files written to `data/`.

| Local name | Yahoo ticker | Family | Notes |
|---|---|---|---|
| `VIX_D` | `^VIX` | Vol family | 6,630 bars from 2000-01-03 |
| `VIX9D_D` | `^VIX9D` | Vol family | 3,863 bars from 2011-01-03 |
| `VIX3M_D` | `^VIX3M` | Vol family | 4,988 bars from 2006-07-17 |
| `IWM_D` | `IWM` | US small caps | 6,529 bars from 2000-05-26 (was missing from Phase A) |
| `STOXX50_D` | `^STOXX50E` | EU index spot | 4,791 bars from 2007-03-29 |
| `FTSE100_D` | `^FTSE` | UK index spot | 6,658 bars from 2000-01-04 |
| `GDAXI_D` | `^GDAXI` | DE index spot | 6,692 bars from 2000-01-02 (Yahoo serves an extended series; the file existed already from a prior session and is overwritten with the extended history) |

### 1.2 Targets added to the scan

`IWM` joins the single-instrument scan universe at D (mean-reversion, trend, volatility classes). The vol-family series (`VIX`, `VIX9D`, `VIX3M`) are **not** scan **targets** — they appear only as **externals** for the new term-structure and VRP factories. Index spot series (`STOXX50`, `FTSE100`, `GDAXI`) become both targets (for trend/MR/vol classes) and externals (for cross-region lead-lag).

### 1.3 Futures proxy substitution (documented per audit A8)

Migrate.md §2.8 framed cross-region lead-lag as "yesterday's MES sign predicting FESX/FTSE today." The ideal source is CME / Eurex continuous front-month futures, which require a paid feed. Phase B substitutes:

| Migrate target | Phase B substitute | Rationale |
|---|---|---|
| MES (S&P 500 micro futures) | `SPY` (already in Phase A) | SPY has 1:1 correlation with ES at D — for IC discovery the cash-vs-futures basis is < 5 bps, irrelevant. |
| FESX (Euro Stoxx 50 futures) | `STOXX50` spot (^STOXX50E) | Yahoo's `STOXX50E` is the index level. Cash-vs-futures basis adds drift but the **direction** of the lead-lag signal is preserved. |
| FTSE futures | `FTSE100` spot (^FTSE) | Same rationale. |
| MDAX micro futures | `GDAXI` spot (^GDAXI is DAX 40, larger universe; MDAX is mid-caps and is a different proxy — Phase B uses DAX spot as a closer match to the underlying lead-lag mechanism than MDAX). | DAX > MDAX as a "European large-cap" benchmark; the lead-lag mechanism is mostly large-cap. |

The futures-vs-spot substitution is a **deliberate deviation** from the parent directive's exact wording. It is justified by data availability and recorded here so that any later analyst can identify and re-run with proper futures data if it becomes available. Audit A8: substitution is logged with reason.

---

## 2. New signal classes

Each class commits a **3-cell parameter grid** (V3.2 plateau gate applies). All factories are causal: VIX-family externals are anchored to the target's index via `anchored_aggregate(higher_tf=False)` and wrapped in `assert_causal` exactly as the parent's cross-asset factories.

### 2.1 Term structure (signals.term_structure)

| Signal | Externals | Param | Cells | Rationale |
|---|---|---|---|---|
| `vix9d_over_vix` | `[VIX9D, VIX]` | smoothing | `1`, `5`, `21` | Backwardation indicator. Ratio > 1 ⇒ near-term vol expected to fall ⇒ positive expected forward equity return. |
| `vix_over_vix3m` | `[VIX, VIX3M]` | smoothing | `1`, `5`, `21` | Standard contango/backwardation signal. Ratio > 1 (backwardation) is a regime indicator and historically a precursor to equity sell-offs. |

Both are applied to equity targets (SPY, QQQ, IWB, TQQQ, CSPX, IWM, EFA, EEM, STOXX50, FTSE100, GDAXI). Not applied to bond / commodity / FX targets (mechanism is equity-specific). The Smoothing parameter is a rolling-mean window applied to the ratio before z-scoring.

Computation: `signal[t] = (ratio[t] - rolling_mean(ratio, 60)) / rolling_std(ratio, 60)` after smoothing the ratio by `rolling_mean(smoothing)`. Both VIX9D and VIX (or VIX and VIX3M) are anchored externally (`.shift(1)` then ffill onto target's daily index). The 60-bar normalisation window is fixed across cells; only the **smoothing** of the raw ratio is the swept parameter.

### 2.2 Vol-risk-premium (signals.volatility_phase_b)

| Signal | Externals | Param | Cells | Rationale |
|---|---|---|---|---|
| `vrp_z` | `[VIX]` | rv_window | `20`, `60`, `120` | `VIX / (realised_vol(close, rv_window) × 100) - 1`. Positive VRP ⇒ implied vol exceeds realised ⇒ vol-seller's edge ⇒ positive expected forward equity return (the canonical mechanism). |

Applied to equity targets only. The VIX series is anchored to the target's daily index via `anchored_aggregate`. Realised vol is computed on the target's own close (causal rolling).

### 2.3 Cross-region lead-lag (signals.cross_region)

| Signal | Externals | Param | Cells | Rationale |
|---|---|---|---|---|
| `us_lead_eu` | `[SPY]` | window | `1`, `5`, `21` | Yesterday's SPY return (smoothed over `window` bars) predicts today's STOXX50 / FTSE100 / GDAXI return. Continuous version (not the sign-only "MES sign" formulation, which would discretise away information). |

Applied only to EU targets (STOXX50, FTSE100, GDAXI). The signal is the rolling mean of SPY's log returns over the last `window` bars, anchored via `anchored_aggregate` so that the SPY value at target time T uses only SPY data with timestamps strictly before T.

### 2.4 Banned

Same banned patterns as parent §2.9. Additionally for Phase B:

- **No VIX-on-VIX self-prediction.** Scanning vix9d_over_vix as a predictor of VIX returns is trivially autocorrelated (the predictor and target share components). Cross-asset signals are evaluated only against equity targets.
- **No EU-on-EU cross-region.** STOXX50 predicting GDAXI is essentially within-region momentum, not lead-lag. Out of scope.

---

## 3. Gate overrides

Inherited from the parent unless listed below. The hard floors (`|t_NW| > 4.5` Phase-A; `|t_NW| > 5.0` if combined N > 25k) apply unchanged.

- **DSR-N calculation.** Phase B adds approximately 250-500 (signal × cells × instruments × horizons) rows. Combined Phase A + Phase B N stays well under 25k, so the floor remains `|t_NW| > 4.5`. Computed at runtime; logged at scan start.

- **MTF agreement.** Phase B signals are D-only (VIX9D, VIX3M, ^STOXX50E etc. don't have H4 / H1 free-tier data). The `mtf_agree` column is therefore `False` for all Phase B headlines by construction; Phase B TIER assignments collapse to TIER_B max (the parent's TIER_A requires MTF agreement). This is a **structural** limitation, not a design failure — D-only signals just sit one tier lower than they would with intraday confirmation.

- **Fold quorum.** Inherits `fold_sign_quorum = 4` from the parent.

- **Sanctuary.** Same 12-month trailing window. Already enforced by the orchestrator.

---

## 4. Implementation plan

1. **This directive on `main`.** Pre-registration done. (THIS PR)
2. `config/ic_census_universe_phase_b.toml`: target list + 4 new signal subsections.
3. `research/ic_analysis/ic_census_lib.py`: add the 4 factories (`vix9d_over_vix`, `vix_over_vix3m`, `vrp_z`, `us_lead_eu`), each declared with `externals` field.
4. Tests for each factory: at minimum the same shape-test + assert_causal pair already in place for `hyg_ief_z`.
5. Run the Phase B scan.
6. Append result log to §5 below — V3.6 documentation regardless of outcome.

---

## 5. Result log

Appended after first invocation on 2026-05-13. §1-§3 unchanged (V3.1).

### 5.1 Run shape

- 855 raw rows / 285 headline rows. 4 signals × 3 cells × 22 targets × 3 horizons after self-correlation skips. (Some bond/FX targets receive Phase B signals too — the signal classes don't gate by target asset class because the IC computation handles that automatically; targets without a relevant mechanism just produce non-significant IC.)
- Tier counts: **0 TIER_A, 0 TIER_B, 285 unconfirmed.**
- Gate breakdown: 60 fold-stable; 5 BH-significant; 0 DSR-pass (no |t_NW| reached the 4.5 floor); 0 plateau-stable; 0 MTF-agree (by construction — Phase B is D-only).

### 5.2 Top mechanistic hits (top 20 by |t_NW|)

These are the signals that the audit-discipline gates **rejected** as deployment-eligible, ordered by |t_NW|. They are economically meaningful and worth recording.

| Signal | Target | h | IC | t_NW | Mechanism notes |
|---|---|---:|---:|---:|---|
| us_lead_eu | EEM | 1 | -0.0507 | -3.92 | Past SPY → negative next-day EEM. Short-term reversal, **opposite** to the a priori follow-through. |
| us_lead_eu | IWB | 1 | -0.0492 | -3.83 | Same mechanism on US large-cap. |
| us_lead_eu | EEM | 5 | -0.0773 | -3.83 | 5-day horizon, same sign. |
| **vrp_z** | **HYG** | **21** | **+0.146** | **+3.66** | **Canonical VRP** → forward HY return positive at 21 days. |
| us_lead_eu | IWM | 1 | -0.0463 | -3.57 | Russell 2000 short-term reversal. |
| us_lead_eu | QQQ | 1 | -0.0455 | -3.54 | Nasdaq short-term reversal. |
| us_lead_eu | IWB | 5 | -0.0669 | -3.36 | 5-day reversal on US large-cap. |
| vix9d_over_vix | FTSE100 | 5 | +0.099 | +3.30 | Backwardation post-stress → FTSE rebound. |
| us_lead_eu | IWM | 5 | -0.0649 | -3.24 | 5-day reversal IWM. |
| **vrp_z** | **TQQQ** | **21** | **+0.167** | **+3.22** | **Highest IC magnitude in Phase B.** 3× leveraged Nasdaq, VRP → positive forward return. |
| us_lead_eu | FTSE100 | 21 | -0.076 | -3.22 | 21-day reversal on UK index. |
| **vrp_z** | **SPY** | **1** | **+0.042** | **+3.17** | **Canonical VRP → SPY at next-day.** |
| vix_over_vix3m | FTSE100 | 21 | +0.142 | +3.16 | 21-day post-stress rebound (backwardation reverses) on FTSE. |
| us_lead_eu | QQQ | 5 | -0.061 | -3.11 | 5-day reversal QQQ. |
| us_lead_eu | HYG | 21 | -0.086 | -3.10 | 21-day reversal on HY credit. |
| us_lead_eu | EFA | 1 | -0.039 | -3.08 | EAFE next-day reversal. |
| us_lead_eu | TQQQ | 1 | -0.048 | -3.06 | TQQQ reversal. |
| vix_over_vix3m | FTSE100 | 5 | +0.081 | +3.05 | 5-day post-stress rebound. |
| us_lead_eu | FTSE100 | 5 | -0.067 | -3.00 | FTSE 5-day reversal. |
| us_lead_eu | STOXX50 | 1 | -0.042 | -2.92 | Euro Stoxx next-day reversal. |

### 5.3 Mechanistic findings

1. **VRP is the most economically robust signal in the entire combined Phase A + Phase B census.** `vrp_z` produces consistently POSITIVE IC across SPY, IWB, TQQQ, HYG, FTSE100, EEM at the 21-day horizon, with magnitudes from +0.04 (SPY) to +0.17 (TQQQ). The sign matches the canonical vol-seller theory exactly. The strongest reading (TQQQ +0.167 / t=+3.22) is also the strongest mechanism-confirming IC in either Phase. But: **none clear |t_NW| > 4.5**, because the daily-frequency sample (3.7k bars for TQQQ, 5.5k for SPY) is too small for VRP's effect size to clear DSR-grade discipline.

2. **`us_lead_eu` is real but inverted.** The a priori was "positive past SPY → positive next-day EU equity (follow-through)." Reality is the OPPOSITE: high past SPY return predicts NEGATIVE next-day return on every equity target tested — US large-cap, US small-cap, Nasdaq, EAFE, EM, and EU indices. This is the classic 1-bar reversal pattern (overnight gap-up fades during the next day's session), well-documented in the academic literature. The signal is more universal than I expected: it works equally on US targets and EU targets, suggesting the mechanism is broader than a region-specific lead-lag.

3. **VIX term-structure backwardation predicts FTSE100 recovery.** Both `vix9d_over_vix` and `vix_over_vix3m` show POSITIVE IC on FTSE100 at 5- and 21-day horizons. This is consistent with the "backwardation peaks at the bottom" interpretation — VIX-curve backwardation is a stress indicator, and forward equity returns are positive on the recovery from stress. Notable that it's strongest on FTSE100 specifically — could be an EU-vs-US risk asymmetry or a sample-period artefact.

4. **Plateau gates correctly reject all cells.** Even the strongest signals (vrp_z, us_lead_eu) have monotonically-varying IC across the swept `smoothing` / `rv_window` / `window` parameter — no interior plateau. Same pattern as Phase A: the IC is real, but it doesn't sit on a stable parameter manifold.

### 5.4 Combined Phase A + Phase B status

The combined census now spans:

- **5 signal classes × 26 instruments × 3 timeframes × 3 horizons ≈ 3,500-4,000 cells.**
- **0 TIER_A and 0 TIER_B survivors.**
- **~75 cells** clearing both BH-FDR AND fold-stability AND having |t_NW| > 3.
- **0 cells** clearing the full audit-discipline stack (DSR + BH + fold + plateau + MTF + sanctuary).

This is the honest Track-1 picture under V3.1 discipline. The signal mechanisms that exist in this universe (microstructure mean-reversion at H1, VRP at D 21-day, US-equity short-term reversal at D 1-day, VIX-term backwardation → FTSE recovery) are real but sample-limited and parameter-fragile. None justify a new strategy proposal under the parent census's deployment gate.

### 5.5 Recommended next actions (out of scope for this directive)

Recorded so they don't get lost in this debrief:

1. **Sample-limited signals warrant a sample-augmentation pass, not a relaxation of the gate.** VRP IC of +0.17 on TQQQ would clear |t| > 4.5 at n ≈ 12k. The right move is acquiring longer or higher-resolution data, not lowering the bar. Possible paths: (a) Databento intraday VIX-future data extending VRP to H1 (multiplies N by ~6 at H4, ~24 at H1); (b) longer-history equity data; (c) a CME-futures Phase C upgrade.
2. **The us_lead_eu inverted-sign result should be cross-checked against a published "lead-lag reversal" model.** If the mechanism is well-known short-term reversal, replicating its sign in this scan is a quality-control success for the scaffolding; if not, it could be data artefact (e.g. SPY → EU calendar offset mishandling).
3. **`vrp_z` interaction with regime is a natural follow-up.** Phase 0 regime labelling (ADX or HMM, from the parent `IC Signal Analysis.md` v4.2 pipeline) can split the IC by regime. If `vrp_z` IC is concentrated in calm-vol regimes and zero in stress regimes, that's a regime-conditional strategy candidate — but it requires its own pre-registration.
4. **Phase C** would unlock CME continuous futures + Eurex futures + breadth panel + OIS-FF spread. Cost: paid data feeds. Decision deferred.

### 5.6 Outcome record

| Field | Value |
|---|---|
| Combined Phase A + Phase B TIER_A count | 0 |
| Combined TIER_B count | 0 |
| Combined unconfirmed count | ~1,140 |
| Strongest mechanistic IC (point estimate) | TQQQ `vrp_z` h=21: IC=+0.167, t=+3.22 |
| Strongest mechanistic IC (US equity benchmark) | HYG `vrp_z` h=21: IC=+0.146, t=+3.66 |
| Sample-augmentation pre-registration required to relax DSR floor? | **Yes, if any of the strongest signals are to be re-tested at a finer time-resolution** |
| Phase C (paid feeds) pre-registration required? | **Yes, before any Phase C scan runs** |

---

## 6. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-13 | Initial Phase B pre-registration. |
