# Pre-Registration — B2: Carver EWMAC Trend Ensemble

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** New strategy audit — multi-speed EWMAC trend-following on the IBKR roll-stitched commodities universe.
**Predecessors:**

- Pre-Reg B4 TSMOM 2026-05-15 (RETIRED on yfinance, L40 contamination).
- Pre-Reg D2b B4b 2026-05-15 (B4b: H1 SUPPORTED for L40, RETIRED at L27 plateau with knife-edge).
- Pre-Reg B4c 2026-05-15 (window-ensemble TSMOM: H1 PARTIALLY SUPPORTED, plateau still 47.93%, L45 partial-mitigation lesson).
- Rob Carver, *Systematic Trading* (2015) + *Leveraged Trading* (2019) — EWMAC framework.
- `resources/Rob Carver's Systematic Trend Following Framework.md` (committed reference).

**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE the audit examines new data. **The hypothesis being tested is that a many-speed EWMAC ensemble dissolves the residual plateau brittleness that B4c (3-window TSMOM ensemble) could not.**

---

## §1. Motivation & mechanism

**Why EWMAC ensembles after B4c.** B4c's 3-window TSMOM ensemble achieved 33.9% plateau-spread reduction but still failed L27 (47.93% spread). The remaining fragility concentrated at short-window neighbours (P_shift_short = (6,9,12) at +0.88 vs P_drop_short = (12,15) at +1.69). Carver's diagnosis (widely-cited in *Systematic Trading*): single-window or few-window momentum is parameter-sensitive because each window's signal is binary-ish. A many-speed EWMAC ensemble averages signals across 4-8 different EWMA-pair speeds, producing a continuous "forecast" that is robust to any single speed's misalignment with the current regime. **L43+L45 together motivate the EWMAC formulation.**

**Mechanism (per asset i, per bar t).**

1. **Raw EWMAC per speed s.** For each speed pair `(fast_s, slow_s)` with `fast_s = slow_s / 4`:
   ```
   ewmac_s(t) = EWMA(close, halflife=fast_s) - EWMA(close, halflife=slow_s)
   ```
   This is the difference between a fast and slow exponentially-weighted moving average of price. Positive = uptrend, negative = downtrend.

2. **Vol-normalised forecast per speed.** Divide by 20-day stdev of price daily change to make the forecast scale-invariant:
   ```
   forecast_raw_s(t) = ewmac_s(t) / stdev(close.diff(), 20)
   ```

3. **Scale to target forecast magnitude.** Multiply by an empirical scalar so the long-run absolute forecast equals 10 (Carver's convention). Forecast scalar table (Carver 2015, p.286):
   ```
   speed 16/64:   scalar = 2.65
   speed 32/128:  scalar = 1.83
   speed 64/256:  scalar = 1.19
   ```
   (For other speeds, use the empirical absolute mean of vol-normalised forecast over the IS window and rescale to 10. For pre-reg V3.1 honesty, we use Carver's published scalars where available and compute IS-frozen scalars for non-standard speeds.)

4. **Forecast cap.** Clip each speed's forecast to [-20, +20] (Carver's "double cap"). Prevents one runaway speed from dominating.

5. **Forecast diversification multiplier (FDM).** When averaging N speeds, multiply by `FDM_N` to compensate for the diversification effect (Carver: FDM_4 = 1.43, FDM_6 = 1.51). Pre-registered constants per speed-count.

6. **Combined forecast.** `combined(t) = clip(FDM_N × mean(forecast_s(t) for s in SPEEDS), -20, +20)`.

7. **Position per asset.** `position_i(t) = combined_i(t) × target_vol_dollars / (instrument_vol_dollars × forecast_scaling)`. Forecast = +10 → position = full target-vol allocation; +20 → 2× allocation; 0 → flat; -20 → 2× short.

8. **Portfolio.** Sum positions across the 24 commodities (each diversified, each vol-scaled). Apply portfolio-level vol target (10% annualised).

9. **Costs.** Same model as B4: 1 bp per turnover + $1 fixed per fill at $30k notional per leg.

**Causality.** All EWMAs use only past data through `t-1` then shift by 1 bar; the position effective at `t` earns the return `r(t+1)`. Same pattern as B4b/B4c.

## §2. Universe + audit configurations

**Universe.** Same 24 commodities as B4b/B4c. Stitched M1 (back-adjusted) from `data/{ROOT}_M1_stitched_D.parquet`.

**Date range.** Same as B4b/B4c: 2023-05-12 → 2026-05-15 (~3 years, L41 ceiling).

**WFO override (L25).** Same as B4b/B4c: 1.5y IS / 0.5y OOS / 5 folds → ~2 effective folds.

**Pre-registered cells (V3.1).**

| Cell                  | Speeds (fast/slow halflife days) | FDM      | Notes                                                |
|-----------------------|----------------------------------|---------:|------------------------------------------------------|
| **C1_canonical**      | (16/64, 32/128, 64/256)          | 1.35     | Carver 3-speed default                               |
| C2_short_speeds       | (4/16, 8/32, 16/64)              | 1.35     | All-short ensemble — should be the fragile one       |
| C3_long_speeds        | (32/128, 64/256, 128/512)        | 1.35     | Long-only-end ensemble                               |
| C4_full_six           | (4/16, 8/32, 16/64, 32/128, 64/256, 128/512) | 1.51 | 6-speed canonical Carver                       |
| C5_two_speed          | (16/64, 64/256)                  | 1.20     | Minimal 2-speed                                      |
| C6_singleton_canonical| (32/128,)                        | 1.0      | Reference — single-speed baseline (no FDM)           |
| C7_full_six_unclipped | (4/16, 8/32, 16/64, 32/128, 64/256, 128/512) | 1.51 | C4 with forecast cap disabled (gross variant)  |
| C8_gross_no_costs     | (16/64, 32/128, 64/256)          | 1.35     | C1 with `apply_costs=False`                          |

C6 is the reference single-speed baseline. C7 / C8 are gross-economics references.

**Plateau pre-flight neighbours of C1 (3-speed default).**

| Neighbour            | Speed change                                |
|----------------------|---------------------------------------------|
| P_shift_short        | (8/32, 16/64, 32/128) — shift down one octave |
| P_shift_long         | (32/128, 64/256, 128/512) — shift up one octave |
| P_drop_fast          | (32/128, 64/256) — drop fastest             |
| P_drop_slow          | (16/64, 32/128) — drop slowest              |

**Falsification hypotheses (pre-committed).**

- **B2 H1 (mitigation completion).** "EWMAC ensemble (C1) achieves plateau spread ≤ 30% (L27 PASS)." Falsifiable: if spread > 30%, ensemble is no better than B4c at fixing the knife-edge.
- **B2 H2 (canonical Sharpe ≥ +1.20).** "C1 canonical Sharpe ≥ 75% of B4b's +1.6309 = +1.22." Falsifiable: a much higher Sharpe would suggest overfit; much lower would suggest the EWMAC formulation lost signal.
- **B2 H3 (forecast cap is load-bearing).** "C7 unclipped Sharpe is materially LOWER than C4 clipped (Carver's claim that the cap reduces noise)." Falsifiable: if C7 ≥ C4, the cap is redundant on this universe.

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds:** `CROSS_ASSET_MOMENTUM` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (DEPLOY, or CONDITIONAL_WATCHPOINT with noise=best), excluding C6/C7/C8 (baselines / gross references), pick the one with highest CI_lo.

**Plateau pre-flight (L27).** C1 + 4 neighbours above. If spread ≤ 30%, audit proceeds. If 30% < spread < B4c's 47.93% (L45 partial mitigation but better than B4c), document and abort. If spread ≥ 47.93%, EWMAC has NOT improved over B4c — log and pivot to a different strategy class.

**L43 + L45 carry-over.** If EWMAC C1 produces plateau ≤ 30%, the "many-speed ensemble" pattern becomes the default mitigation in V3.6 for any trend-following follow-up. If still > 30%, L43/L45 are reinforced — single- and few-window-or-few-speed trend on this short window cannot escape plateau brittleness.

**Sanctuary:** same as B4b/B4c — last 6 months held out.

## §4. Result log

### §4.1 Baseline reproduction

C6 singleton (32/128 EWMAC): Sharpe = **+2.0572**. Single-speed Carver EWMAC at the 32/128 halflife is robustly positive on the IBKR stitched M1 universe. Not directly comparable to B4b's TSMOM singleton (different signal construction), but same broad regime sanity.

### §4.2 Plateau pre-flight — L27 GATE PASSED

| Neighbour       | Speed set                          | Sharpe   |
|-----------------|------------------------------------|---------:|
| **C1_canonical**| (16/64, 32/128, 64/256) FDM=1.35   | +2.0218  |
| P_shift_short   | (8/32, 16/64, 32/128)              | +1.8152  |
| P_shift_long    | (32/128, 64/256, 128/512)          | +2.1373  |
| P_drop_fast     | (32/128, 64/256) FDM=1.20          | +2.1426  |
| P_drop_slow     | (16/64, 32/128) FDM=1.20           | +1.9350  |

**Relative plateau spread: 15.28%** — comfortably below L27's 30% gate.

Mitigation vs prior audits:
- vs B4b (single-window TSMOM, 72.48%): **+78.9% spread reduction**
- vs B4c (3-window TSMOM ensemble, 47.93%): **+68.1% spread reduction**

The many-speed EWMAC formulation fully dissolves the plateau knife-edge that defeated single-window TSMOM (B4b) AND 3-window TSMOM ensembles (B4c). L43's mitigation thesis is fully validated by the EWMAC construction — but NOT by the TSMOM-ensemble construction (B4c).

### §4.3 Per-cell 5-axis matrix

| Cell                       | Sharpe   | CI95 lo  | CI95 hi  | DSR    | MC P  | Sanc Sharpe | Noise base | Noise axis | Verdict                |
|----------------------------|---------:|---------:|---------:|-------:|------:|------------:|-----------:|:----------:|------------------------|
| C1_canonical               | +2.0218  | -0.044   | +3.974   | 1.0000 | 0.000 | +0.3258     | +0.7834    | best       | CONDITIONAL_WATCHPOINT |
| C2_short_speeds            | +1.5820  | -0.466   | +3.421   | 1.0000 | 0.000 | +1.3603     | +0.5891    | mid        | TIER_UNCONFIRMED       |
| C3_long_speeds             | +2.1373  | +0.054   | +4.198   | 1.0000 | 0.000 | -0.4006     | +0.9325    | mid        | TIER_UNCONFIRMED       |
| C4_full_six                | +1.9069  | -0.152   | +3.875   | 1.0000 | 0.000 | +0.7583     | +0.7620    | best       | CONDITIONAL_WATCHPOINT |
| C5_two_speed               | +2.0223  | -0.088   | +3.979   | 1.0000 | 0.000 | +0.4007     | +0.7973    | mid        | TIER_UNCONFIRMED       |
| C6_singleton_canonical     | +2.0572  | +0.005   | +4.102   | 1.0000 | 0.000 | +0.2836     | +0.8648    | mid        | CONDITIONAL_WATCHPOINT |
| C7_full_six_unclipped      | +2.0350  | +0.040   | +3.990   | 1.0000 | 0.000 | +0.7055     | +0.9805    | mid        | CONDITIONAL_WATCHPOINT |
| C8_gross_no_costs          | +2.2530  | +0.214   | +4.247   | 1.0000 | 0.000 | +0.5496     | +1.0105    | best       | **DEPLOY**             |

### §4.4 H1 verdict — SUPPORTED

> "EWMAC ensemble (C1) achieves plateau spread ≤ 30%."

Plateau spread **15.28%** vs 30% gate. H1 unambiguously SUPPORTED.

### §4.5 H2 verdict — SUPPORTED

> "C1 canonical Sharpe ≥ 75% of B4b's +1.6309 = +1.22."

C1 Sharpe = +2.0218 = **124% of B4b**. Well above the 75% threshold. The EWMAC formulation does NOT lose Sharpe relative to single-window TSMOM — it gains it, presumably because the multi-speed averaging captures regime-switching better than a fixed 12-month window.

### §4.6 H3 verdict — REJECTED

> "Forecast cap is load-bearing (C4 clipped Sharpe > C7 unclipped)."

- C4 (6-speed, capped at ±20): Sharpe = **+1.9069**
- C7 (6-speed, uncapped): Sharpe = **+2.0350**
- Cap effect: **−0.1281** (cap HURTS Sharpe on this universe)

The forecast cap is REDUNDANT or marginally harmful on the IBKR 3y commodities sample. Possible explanations: (a) max forecasts on this data never exceed natural levels, so the cap rarely binds; (b) on the few days it does bind, it cuts off legitimate strong-signal positions. Carver's cap is conservative; on a sample where saturation is rare, it removes signal without adding robustness. **Methodology note**: in a longer/broader audit (more crash days, more vol spikes) the cap likely IS load-bearing. The H3 rejection is sample-specific, not universal.

### §4.7 Promotion verdict + sample-size caveat

Per pre-reg §3 selection rule (DEPLOY-eligible, exclude C6/C7/C8 baselines/gross, highest CI_lo):

- **Eligible cells:** C1 (CI_lo=-0.044, noise=best) and C4 (CI_lo=-0.152, noise=best). Both CONDITIONAL_WATCHPOINT with noise axis = best.
- **Highest CI_lo:** **C1_canonical** at CI_lo = **−0.044**.

**However: C1's CI_lo is marginally NEGATIVE.** Per V3.6 research-math rule: "A strategy whose 95% lower bound is ≤ 0 is `tier=unconfirmed` and cannot be added to default deployment registries." This is the binding constraint. The decision-matrix verdict (CONDITIONAL_WATCHPOINT) conflicts with the bootstrap-CI verdict (tier=unconfirmed for negative CI_lo).

**Resolution.** The tighter constraint wins. **C1 is NOT promoted to live deployment** despite passing plateau, H1, H2, MC, DSR, sanctuary, and noise axes. The bottleneck is the 95% bootstrap CI lower bound, driven by the limited 2-fold WFO sample on L41's 3-year IBKR window.

### §4.8 What this means for the backlog

**Research alive, deployment dead-on-sample-size.** Three implications:

1. **C8_gross_no_costs PROMOTES** under the 5-axis matrix (DEPLOY verdict, CI_lo = +0.214). This tells us the EDGE IS REAL and the bottleneck is cost drag (Δ Sharpe between C8 gross and C1 net = +0.23). For a live retail-cost account on IBKR, the strategy is borderline.

2. **The "wait for more data" path** is methodologically clean. With another ~12 months of data and 3-4 WFO folds instead of 2, CI_lo widens to positive territory and C1 likely passes the strict CI gate. Track-in-shadow on paper for 12 months, then re-audit.

3. **The "different universe" path** — apply the same EWMAC framework to a broader (commodities + bonds + equities + FX) Carver universe. Diversification across asset classes typically tightens CI substantially (and Carver's published EWMAC results are on ~30-50 instruments, not 24).

**New lesson: L46 — sample-size-limited audits should report decision-matrix verdict AND the binding constraint, separately.** Detail in V3.6 catalogue.

**Backlog next step.** With the EWMAC ensemble passing plateau + H1 + H2 but blocked by sample-size CI, the natural next moves are:

- **(a) Track-in-shadow.** Port C1 EWMAC to `titan/strategies/ewmac/` as a NON-LIVE shadow strategy (computes signals + paper-PnL every bar but doesn't trade). Re-audit in 6 + 12 months as the CI shrinks.
- **(b) Universe expansion.** Pre-reg B2b — same EWMAC config, expanded universe (commodities + bond futures from IBKR + currency futures). Larger N tightens CI.
- **(c) Pivot to I1 (HMM + XGBoost meta-labeler).** A meta-labeler trained on EWMAC's signals might lift CI_lo by selectively trading only the high-confidence forecasts.

Recommendation: **(a) + (b) in parallel.** (a) buys time; (b) gives a clean re-audit path with broader instrument coverage.

---

## §5. Failure modes to watch

- **L04 / A1 (causality).** EWMA must use `.shift(1)` before earning return. Smoke test as B4b/B4c.
- **L25 (class override).** WFO override is L41 contingency.
- **L27 (plateau).** Whole test is for L27 robustness.
- **L37 (TSMOM persistence).** EWMAC IS TSMOM in disguise (fast - slow MA is mathematically equivalent to a band-pass-filtered momentum signal). Reconfirm via singleton-baseline C6.
- **L40 (yfinance contamination).** Already fixed by using stitched M1.
- **L43 (knife-edge plateau).** What we're trying to dissolve.
- **L44 (back-adjust bias).** N/A — EWMAC on single back-adjusted M1 series is correct.
- **L45 (partial mitigation reporting).** Apply: report B2 spread vs B4c's 47.93% vs B4b's 72.48%, plus mitigation pct.
- **A6 (MC bootstrap).** Class default.
- **A5 (DSR).** 8 cells, DSR at N=8 trials.

**Specific risks unique to EWMAC.**

- **Forecast scalar IS-fitting.** Carver's published scalars assume vol distributions roughly stable; on our 3y window these might be slightly off. Mitigation: for non-published speeds, compute IS-frozen scalars (V3.1 — frozen on IS window of first fold, applied to all folds).
- **FDM constants.** Carver's FDM table assumes diversified speed correlation. With short data (3y, 2 folds), the FDM might over- or under-correct. The pre-registered FDM values are explicit; do NOT post-hoc tune them.

## §6. Implementation plan

1. **Build `research/ewmac/`** module with `ewmac_strategy.py`:
   - `EwmacConfig` dataclass (speeds: tuple of (fast, slow) halflife days; forecast_cap: float; fdm: float; target_vol_annual: float; ...).
   - `compute_ewmac_forecast(closes_df, speeds, ...)` returning per-bar combined forecast per asset.
   - `ewmac_returns(closes_df, *, cfg)` for end-to-end strategy returns.
   - `ewmac_assert_causal(closes_df, cfg)` smoke test (same pattern as TSMOM).
2. **Unit tests** in `tests/test_ewmac.py`:
   - Forecast positive for uptrend, negative for downtrend.
   - Forecast cap clips correctly.
   - FDM × mean is applied.
   - Causality smoke test.
   - Single-speed (C6 baseline) is reproducible.
3. **Write `research/ewmac/run_b2_audit.py`** mirroring `run_b4c_audit.py` structure (baseline reproduction → plateau pre-flight → full 5-axis → result log).
4. **Run audit.** Document.
5. **If C1 promotes:** add to recommended-next ports list (`titan/strategies/ewmac/`). If still plateau-fails, the commodity-trend research on this window is exhausted; pivot to I1 (HMM + XGBoost meta-labeler) or shift to a longer historical window via paid futures data.

---

## §7. B2b yfinance follow-up (2026-05-16) — REGIME-ARTIFACT FALSIFICATION

**Motivation.** B2 §4.7 noted CI_lo = -0.044 as the binding constraint (L46) and recommended either (a) shadow-deploy or (b) universe-expansion. Option (b) was attempted via IG live API (L47 blocked it) and then via yfinance ETF-proxy fallback (`scripts/download_b2b_alternative.py`).

**Universe (31 instruments, yfinance, 2005-01-01 → 2026-05-16, 5574 bars):**

- 8 equity-index proxies (SPY/QQQ/DIA/IWM/^FTSE/^GDAXI/^N225/^STOXX50E)
- 8 FX spot pairs (`*=X`)
- 6 bond ETF proxies (IEF/TLT/SHY/IGLT.L/IBGS.L/EMB)
- 4 physical commodity ETFs (GLD/SLV/PPLT/PALL — no L40)
- 5 regional equity ETFs (EEM/EFA/VGK/EWJ/FXI)

WFO: class-default `CROSS_ASSET_MOMENTUM` (2y IS / 0.5y OOS) → **60 folds** (vs B2's 2).

**Plateau pre-flight (B2b):**

| Cell                 | Sharpe   |
|----------------------|---------:|
| **C1_canonical**     | -0.2817  |
| P_shift_short        | -0.2505  |
| P_shift_long         | -0.3434  |
| P_drop_fast          | -0.3151  |
| P_drop_slow          | -0.2626  |

**Relative plateau spread: 37.10%** (L27 30% gate FAIL).

**§7.1 Comparison vs B2 IBKR-3y:**

| Metric | B2 IBKR-3y | B2b yf-21y |
|---|---:|---:|
| Universe size | 24 commodities | 31 cross-asset |
| Bars | 760 | 5574 |
| WFO folds | 2 | 60 |
| C1 Sharpe | **+2.0218** | **-0.2817** |
| Plateau spread | 15.28% (PASS) | 37.10% (FAIL) |
| All 5 plateau cells | All positive | All negative |

**§7.2 Verdict — REGIME-ARTIFACT confirmed (L48).**

The 10×-larger sample REVERSED the canonical Sharpe sign. The +2.02 was a 2023-2025 commodity-trend-regime artifact, not a generalizable Carver-style edge. Cross-asset universal-trend EWMAC is NEGATIVE-Sharpe over the post-2008 regime — consistent with the widely-documented decline of pure trend strategies after the 2008-2014 commodity-trend bonanza.

**§7.3 Deployment status update:**

- **B2 (IBKR-3y, narrow universe):** demoted from SHADOW → **NARROW_REGIME_RESEARCH**. The strategy may still produce positive returns on the 24-commodity universe specifically over the 2023-2025 regime, but the operator must own that scope honestly. NOT a universal trend strategy.
- **B2b (yf-21y, broad universe):** RETIRED. Universal-trend hypothesis falsified.
- **shadow_ewmac.py** continues to track the 3 IG-deep instruments (CL/BZ/SPX) for the narrow-regime hypothesis, but the broader DEPLOY path is closed.

**§7.4 New lesson L48 added to V3.6 Catalogue.** "A high Sharpe on a narrow universe + narrow window must be re-validated on a broader universe + window BEFORE deployment promotion; sample-size CI tightening is necessary but NOT sufficient." The framework's plateau + CI gates correctly REFUSED to promote B2; L48 explains the deeper reason.

**§7.5 Recommended backlog re-prioritisation:**

- DROP: further B2 EWMAC variants on commodity-only universes (research consumed).
- ADD: I1 — HMM regime + XGBoost meta-labeler. The regime-dependence finding makes regime detection a natural next research stream.
- ADD: B5 — Intraday momentum (Gao-Han-Li-Zhou) on equity-index ETFs. Different signal class, less subject to the cross-asset regime-decline that killed B2b.
- KEEP: shadow_ewmac running on (CL/BZ/SPX) as narrow-regime watchpoint — if the 2023-2025 commodity-trend continues forward, the operator gets paid for the narrow scope while acknowledging it's not universal.
