# IC Signal Census — Pre-Registered Scan Plan

**Version:** 1.0 | **Date:** 2026-05-13 | **Author:** Architect
**Status:** **PRE-REGISTRATION** — committed to `main` BEFORE the scan runs (V3.1 discipline).

---

## 0. Why this exists

The May-12 Samir-Stack audit revised the prior champion from 2.28 Sharpe / -6.6% DD to 0.63 Sharpe / -23% DD once same-bar look-ahead, TR/price conflation, and DSR-uncorrected sweep selection were removed. The V3 layered-defence reruns then showed that no individual HMM / momentum / dd-velocity gate carried IC above noise. The honest conclusion: stop building strategies, build a signal catalogue first, IC-rank it, then construct strategies from the survivors. Only the survivors get the cost-aware engine, parameter sweep, and live deployment work that has so far been spent on signals we cannot prove are real.

This directive **pre-registers** the universe, the parameter grid, the gates, the sanctuary, and the look-ahead invariants for the census. Once this file is on `main`, the gates can only be **relaxed** in a subsequent PR that explains why the original was unimplementable. Tightening or cell-favouring after results land is forbidden (V3.1).

This is additive to `directives/IC Signal Analysis.md` (v4.2). That document specifies the single-instrument Phase 0-2 pipeline (regime IC, ICIR-NW, BH-FDR, MFE/MAE, half-life). This document specifies the **multi-instrument multi-timeframe multi-horizon meta-scan** that consumes Phase 1 outputs, applies DSR for the much-larger N, enforces plateau-stability across a pre-registered parameter sub-grid, requires cross-timeframe IC agreement, and produces the canonical `ic_census.parquet` database that future strategy proposals must query before they are accepted.

---

## 1. Pre-registered universe

### 1.1 Instruments (Phase A — already downloaded)

Phase A is the census we can run today. Phase B instruments require a `scripts/download_data.py` pass and will be appended once data lands; their gates are identical (see §3).

| Class | Instruments | Notes |
|---|---|---|
| US equity index proxies | SPY, QQQ, IWB, TQQQ, CSPX | TQQQ is 3× — beware leveraged-ETF compounding bias (see §5.3) |
| Sector / regional ETFs | EFA, EEM, GLD, DBC | EFA/EEM D-only |
| Fixed income ETFs | TLT, IEF, HYG, LQD, TIP, IHYG | IHYG UCITS, D-only |
| FX majors | EUR/USD, AUD/JPY, AUD/USD, USD/JPY, GBP/USD | All have H1/H4/D |

### 1.2 Instruments (Phase B — download required before census)

| Class | Instruments | Reason |
|---|---|---|
| US equity futures | MES, ES, NQ, MNQ | Required for Samir-Stack equity sleeve re-derivation under MES |
| EU / UK index futures | FESX, FTSE-fut, MDAX-micro | Cross-region lead-lag class needs European indices |
| Vol family | VIX (spot), VIX9D, VIX3M, VIX futures front+M2 | Term-structure signal class |
| Misc | IWM, VUSD H1+H4, single-stock leaders SPX top-10 | Cross-sectional rank tests |

The Phase A census runs first. Phase B is appended once data download lands; results are merged into the same `ic_census.parquet` with `phase` column.

### 1.3 Timeframes

D, H4, H1. No M5 in the census — bar count is fine but cost-to-signal ratio is empirically poor for the signal classes we are testing. M5 may be added in a separate directive if intraday microstructure signals (gap/range/vol-of-vol) emerge as survivors at H1.

### 1.4 Forward horizons (per timeframe)

| Timeframe | Horizons (bars) | Approximate calendar |
|---|---|---|
| D | 1, 5, 21 | 1 day, 1 week, 1 month |
| H4 | 1, 6, 30 | 4h, 1 day, 1 week |
| H1 | 1, 8, 40 | 1h, 1 day, 1 week |

All forward returns are vol-adjusted log returns per `run_ic.compute_forward_returns(close, horizons, vol_adjust=True)`. The signal at bar `t` is correlated against the standardised return from `t` to `t+h`. The lag discipline (`.shift(-h)` is the **target**, not a feature) is enforced inside that function; any new feature factory must contain **zero** `shift(-n)` calls.

---

## 2. Signal classes

Eight classes. Each class commits a **3-cell parameter grid** (§2.10). The full Phase-A pre-registered enumeration is roughly:

```
~18 signals (across classes, see §2.1-2.8)
×  3 parameter cells per signal (geometric spacing)
×  ~20 Phase-A instruments
×  3 timeframes (D, H4, H1) — many instrument×TF cells empty when data missing
×  3 forward horizons
≈ ~9,700 (signal, params, instrument, timeframe, horizon) cells, Phase A
```

Empty cells (instrument has no data at that TF) are skipped. Phase B adds approximately another 8-10k cells. The combined sweep N drives the DSR floor in §3.4.

### 2.1 Trend (single-instrument)

| Signal | Parameter being swept | Pre-registered cells |
|---|---|---|
| Momentum (12-1 style) | lookback / skip | `(252, 21)`, `(126, 21)`, `(63, 5)` (D) ; auto-scaled at H4/H1 by `× bars_per_day` |
| EWMAC pair | (fast, slow) | `(8, 32)`, `(16, 64)`, `(32, 128)` |
| MA distance (close/SMA - 1) | window | `20`, `50`, `200` |

### 2.2 Mean reversion (single-instrument)

| Signal | Parameter | Cells |
|---|---|---|
| RSI deviation | period | `7`, `14`, `28` |
| VWAP overshoot (close/VWAP - 1, anchored) | anchor bars | `6`, `24`, `96` |
| Bollinger %B | window | `10`, `20`, `50` |

### 2.3 Cross-asset (z-score of an external series)

| Signal | Parameter | Cells |
|---|---|---|
| HYG / IEF spread z (priced on target return) | lookback | `21`, `60`, `120` |
| USD index (proxy: DXY or EUR/USD inverted) z | lookback | `21`, `60`, `120` |

These run on every Phase-A target instrument (the signal is computed on HYG/IEF or DXY, then aligned to the target's bars via the **Anchored MTF Aggregation Rule** in §4). When the target is HYG or IEF itself, the same-series cell is skipped to avoid trivial autocorrelation.

### 2.4 Macro / flow (D only)

| Signal | Parameter | Cells |
|---|---|---|
| Breadth proxy: % of SPX components above 200D SMA | n_components | `100`, `300`, `500` |
| TIPS real-yield proxy: TIP/TLT ratio | smoothing | `5`, `21`, `60` |
| OIS-FF spread (Phase B if data missing) | smoothing | `5`, `21`, `60` |

The breadth signal requires the SPX-component panel (Phase B). The TIP/TLT ratio is Phase A.

### 2.5 Volatility regime

| Signal | Parameter | Cells |
|---|---|---|
| Realised vol z (vol-of-returns) | rolling window | `20`, `60`, `120` |
| Vol-risk-premium proxy (VIX / RV20 - 1) | RV window | `20`, `60`, `120` |

Both require VIX — Phase B for the VRP variant. The realised-vol z is Phase A (computed on the target itself).

### 2.6 Microstructure (H1 / H4)

| Signal | Parameter | Cells |
|---|---|---|
| Overnight gap z | lookback days | `20`, `60`, `120` |
| Intraday range / ATR ratio | ATR period | `7`, `14`, `28` |

### 2.7 Term structure (Phase B — gated on VIX futures data)

| Signal | Parameter | Cells |
|---|---|---|
| VIX9D / VIX | smoothing | `1`, `5`, `21` |
| VIX / VIX3M | smoothing | `1`, `5`, `21` |
| Front futures basis (M1/M2 - 1) | smoothing | `1`, `5`, `21` |

### 2.8 Cross-region lead-lag

| Signal | Parameter | Cells |
|---|---|---|
| Yesterday's MES sign predicting FESX | window for sign vote | `1`, `5`, `21` |
| Yesterday's MES sign predicting FTSE | window | `1`, `5`, `21` |

Phase B (requires futures data).

### 2.9 Banned signal patterns

The census **does not** include:

- Any signal derived purely from price autocorrelation that the lag-1 already captures (e.g. `RSI(close).shift(1)` predicting next-bar close).
- Any feature that uses `.shift(-n)` with `n > 0` outside `compute_forward_returns`.
- Any "ML stacker" or composite — composites are constructed only in §6 from IC-survivors.
- Any signal whose computation contains `.expanding()` over close (look-ahead variance estimate; audit fix A6).

### 2.10 Parameter grid policy

Three cells per signal, **geometric spacing**, **pre-registered above**. Selection rule:

- A signal passes IC validation only if (a) its best cell clears the DSR-adjusted t-floor (§3.4), AND (b) **both grid neighbours** also clear a relaxed `|t| > 3.0` floor, AND (c) IC magnitude varies by `< 30%` across the three cells.
- A signal that survives only at one isolated parameter value is **overfit** and is **rejected** regardless of headline t-stat. This is the plateau-stability gate (V3.2).
- Edge cells (no two-sided neighbour, e.g. parameter `1` in a `1/5/21` grid) are ineligible to be selected as the headline cell. Tie-break on plateaus by parsimony — simpler parameters win (smaller lookback, fewer free parameters).

---

## 3. Pre-registered gates

A signal `(class, instrument, timeframe, parameter cell, horizon)` enters the **survivors** table only if ALL of the following hold. Each gate has a `null-result` action: failed gate is recorded with the failure reason, not silently dropped.

### 3.1 Spearman IC and NW-adjusted t-stat

- IC = Spearman rank correlation per `run_ic.compute_ic_table` (causal, no imputation, drops NaN pairs).
- NW-adjusted t-stat per `run_ic.compute_icir_nw` (HAC SE with `h-1` lags).
- Hard gate: `|t_NW| > 4.5` at the headline cell (see §3.4 derivation).

### 3.2 BH-FDR survival

Apply Benjamini-Hochberg across the entire Phase-A pooled `(signal × instrument × TF × cell × horizon)` p-value table at `α = 0.05`. Phase-B is BH-pooled with Phase A on the combined table; do not BH twice. A cell that doesn't survive BH is recorded `bh_significant = False` and is ineligible regardless of `|t|`.

### 3.3 Sign-stability across walk-forward folds

Five non-overlapping folds spanning the sample (excluding sanctuary window). A signal must have IC of the same sign in **≥ 4 of 5 folds** at the headline cell. Failure → `fold_stable = False`.

The 4-of-5 quorum is the right operationalisation of Migrate.md's "reject if any fold IC sign-flips" — one tolerated near-zero fold is realistic (NaN-IC or |IC| < 0.005 fold should not torpedo an otherwise-stable signal), but two opposing folds is a regime artefact, not robustness. A naive "≥ 3 of 5" gate is **tautological** since 5 non-NaN folds always have ≥ 3 sharing a sign by pigeonhole.

### 3.4 Deflated Sharpe / Deflated IC at total N

The DSR adjustment from Bailey & López de Prado 2014, applied to IC's t-statistic. The expected null-hypothesis max |t| over N independent tests is approximately `sqrt(2 ln N)`. For Phase A alone, N ≈ 9,700 → null-expected-max ≈ 4.4. For Phase A + Phase B combined N ≈ 18,000 → null-expected-max ≈ 4.4-4.5. The hard floor is therefore `|t_NW| > 4.5` (≈ 0.1 standard deviations above the expected null max), tightened to **`|t_NW| > 5.0` if Phase B brings total N above 25k**.

Implementation reference: `research/samir_stack/run_phase5_joint_sweep.py::deflated_sharpe_prob`. The same function is reused in `research/ic_analysis/run_ic_census.py` (to be built) with IC's t-stat in place of Sharpe's.

Apply in addition to BH-FDR; the two correct for different things (BH for false discovery rate across many simultaneous tests, DSR for the inflated point estimate from selecting the best cell).

### 3.5 Multi-timeframe agreement

A signal must clear the headline gate (§3.1, §3.2, §3.4) at **at least 2 of 3 timeframes** for the same instrument with the same IC sign. A signal that fires only at H1 (or only at D) is a candidate but is recorded `mtf_agree = False` and demoted to TIER-B.

This is the legitimate version of MTF confluence: not stacking positions at multiple resolutions (which produced the EUR/USD MTF +1.94 Sharpe look-ahead bug), but cross-timeframe IC validation. The MTF EUR/USD failure mode is avoided by the Anchored MTF Aggregation Rule in §4.

### 3.6 Sanctuary window held out

The autoresearch agent never sees the most recent **12 months** of data during census, fold construction, BH-FDR, or DSR. The sanctuary window is computed at scan start as `[max(index) - 365 days, max(index)]` and dropped before any signal sees the panel.

The sanctuary is reserved for **one** final-validation pass per release: once Phase A survivors are catalogued, the top-K survivors are re-IC'd on the sanctuary window and the result is reported alongside the headline cell. Any survivor whose sanctuary IC is the wrong sign is downgraded to `tier = sanctuary_fail` and is **not** advanced to strategy construction (§6).

### 3.7 IC decay sanity check

Per `run_ic.compute_alpha_decay`. A signal's IC must monotonically decay (or at minimum, not strengthen) when going from horizon 1 → max horizon. If IC at the longest horizon is more than `1.5×` the IC at the headline horizon, the signal is flagged `decay_anomaly = True` and demoted — this pattern usually indicates a leaked label or an autocorrelation artefact in the standardised return.

---

## 4. Anchored MTF Aggregation Rule

Forced by the EUR/USD MTF +1.94 Sharpe look-ahead bug. The rule is non-negotiable for every cross-timeframe signal in §2.3, §2.4, §2.8 and any future MTF feature.

**Rule.** A higher-TF signal at timestamp `T` may only see lower-TF data with timestamps **strictly less than `T`**. The aggregation window closes one bar before `T`; it never includes the bar `T` itself or any bar inside the higher-TF interval that `T` represents.

**Symmetric rule for lower-TF using higher-TF gate.** An H1 strategy at `13:30 on 2026-05-13` may use the D1 bar for `2026-05-12` (closed) but **not** for `2026-05-13` (still forming). The pattern is `daily_signal.shift(1).reindex(h1_index, method="ffill")` — daily must be shifted by **one daily bar before** reindexing.

**Canonical code pattern** (commit to `run_ic_census.py`):

```python
# H1 → D1 aggregation: D1 signal at T uses H1 bars in [T-1d, T), never in [T, T+1d)
h1_daily = h1_signal.resample("1D", label="left", closed="left").last().shift(1)
# label="left" + closed="left" + .shift(1) is belt + braces:
#   closed="left"  → bin [T, T+1d) instead of (T, T+1d]
#   label="left"   → bin labelled with start time
#   .shift(1)      → defensive against tz/DST boundary surprises
d1_view_of_h1 = h1_daily.reindex(d1_index, method=None)  # no ffill — accept NaNs
```

**Banned operations in the signal pipeline:**

- `.ffill()` of a lower-TF signal onto a higher-TF index. This is the actual mechanism by which MTF EUR/USD looked profitable. If you need frequency alignment, use `reindex(method=None)` and accept the NaNs; let `.shift(1)` handle alignment.
- `.resample(...).agg(...)` without `closed="left", label="left"` for forward-aggregation. The default `closed="right", label="right"` will pull bar `T+1`'s content into the bin labelled `T`.
- `groupby(date).transform("last")` for per-day aggregation — this is forward-looking within the day.

### 4.1 Causality smoke test (mandatory)

Every aggregation function in `run_ic_census.py` must be wrapped by this test before its output is consumed by IC computation. The test fails the whole run if any aggregation fails — no silent skip.

```python
def assert_causal(agg_fn, src: pd.Series, dst_index: pd.DatetimeIndex,
                  n_trials: int = 5, seed: int = 42):
    """A10 causality test from V3 audit. Corrupt future bars and assert past
    aggregator output is bit-exact unchanged. Raises AssertionError on leak."""
    rng = np.random.default_rng(seed)
    baseline = agg_fn(src.copy(), dst_index).copy()
    n = len(src)
    for _ in range(n_trials):
        t_idx = rng.integers(n // 2, n)  # corrupt second half
        corrupted = src.copy()
        corrupted.iloc[t_idx:] = corrupted.iloc[t_idx:] * 100.0
        corrupted_out = agg_fn(corrupted, dst_index)
        # All output values whose timestamp < src.index[t_idx] must match baseline.
        cutoff = src.index[t_idx]
        past_baseline = baseline[baseline.index < cutoff].dropna()
        past_corrupted = corrupted_out[corrupted_out.index < cutoff].dropna()
        pd.testing.assert_series_equal(past_baseline, past_corrupted, check_exact=True)
```

This is the V3 audit's A10 test ported to MTF aggregation. Any aggregation that doesn't survive 5 random corruption trials is rejected — and the test runs **inside** the census, not as a separate step that can be skipped.

---

## 5. Cost-model and return-engine invariants

The IC census does **not** apply transaction costs (that's Phase 3 / `phase3_backtest.py`). But several invariants from the May-12 audit still apply because the IC is computed against the return engine's output.

### 5.1 Vol-adjusted log-return convention

Per `run_ic.compute_forward_returns`. `raw[t] = log(close[t+h] / close[t])`, `vol[t] = rolling_std(log_returns, 20) * sqrt(h)`, `fwd[t] = raw[t] / vol[t]`. The vol denominator is computed from the **same close** series — no TR/price-only ambiguity at IC stage. The audit's A3 lesson (TR vs price-only must be explicit) applies in Phase 3, not here. But:

### 5.2 Close-series source must be declared

Every parquet under `data/` is total-return-adjusted close (yfinance default for ETFs/equities, NautilusTrader native for FX). The census **does not** mix TR and price-only series. New instruments from Phase B that come from a different source must be documented in this file before they enter the census.

### 5.3 Leveraged-ETF compounding bias

TQQQ is in the universe and is 3× leveraged. The audit's A11 lesson is that constant equity-weight × NAV sizing over-deploys by `equity_native_leverage`× for leveraged ETFs. At IC stage we are not sizing, but the **return** of TQQQ is not 3× the return of QQQ — it suffers compounding decay during volatile sideways periods. A signal that has IC on QQQ and is then tested on TQQQ may appear stronger or weaker for reasons that are purely a leverage artefact. Both are retained in the census, but TQQQ results are reported with a `leveraged_etf = True` flag and excluded from MTF-agreement quorum decisions on the QQQ family.

---

## 6. From census output to strategy construction

The census produces `ic_census.parquet` with columns:

| Column | Type | Meaning |
|---|---|---|
| `signal` | str | Signal name (e.g. `rsi_dev`, `hyg_ief_z`) |
| `signal_class` | str | One of §2.1-§2.8 |
| `params` | str | JSON-encoded parameter dict, e.g. `{"period": 14}` |
| `instrument` | str | E.g. `SPY`, `AUD_JPY` |
| `timeframe` | str | `D`, `H4`, `H1` |
| `horizon` | int | Forward bars |
| `n_bars` | int | Non-NaN sample size |
| `ic_spearman` | float | Spearman IC at headline horizon |
| `t_stat_nw` | float | Newey-West adjusted t-stat |
| `bh_pvalue_adj` | float | BH-adjusted p-value across full pool |
| `bh_significant` | bool | Survives BH at α=0.05 |
| `fold_ic` | list[float] | IC per fold (5 folds) |
| `fold_stable` | bool | Sign agrees in ≥ 4 of 5 (one tolerated near-zero fold with `|IC| < 0.005`) |
| `dsr_pvalue` | float | Deflated-Sharpe-style p-value at total N |
| `plateau_stable` | bool | Two grid neighbours also clear t > 3 AND IC range < 30% |
| `mtf_agree` | bool | Headline gate clears at ≥ 2 of 3 TFs same sign |
| `sanctuary_ic` | float | IC on the held-out 12-month sanctuary (filled in final-validation pass only) |
| `ic_decay_ratio` | float | IC(max horizon) / IC(headline horizon) |
| `decay_anomaly` | bool | True if ratio > 1.5 |
| `phase` | str | `A` or `B` |
| `tier` | str | `TIER_A` (all gates pass), `TIER_B` (mtf_agree=False but other gates pass), `unconfirmed` (any other gate fails), `sanctuary_fail` |
| `leveraged_etf` | bool | True for TQQQ-class instruments |
| `notes` | str | Failure reason if any |

### 6.1 Strategy construction rule

Strategies are proposed only from TIER_A and TIER_B survivors. Three legitimate construction patterns:

1. **Single-IC simple strategies.** Top-K signals by `t_stat_nw × fold_stability_ratio`, each as a stand-alone bet. Sharpe target is modest (+0.3 to +0.5) but the bootstrap CI must be tight (CI_lo > 0).
2. **Multi-IC confluence (AND-gate).** Pick pairs/triples of TIER_A signals whose pairwise IC-rank-correlation is `< 0.3`. AND-gate the binary entry decision. This is the legitimate version of what the Samir-Stack tried to do.
3. **Regime-conditional gating.** If the census shows signal X has IC only when regime indicator Y is "on" (use `phase1_sweep.py` regime split to detect this), build a gated version. Each layer is now IC-justified, not just assumed. This is the proper "Layer 1 + Layer 2".

### 6.2 What is explicitly NOT in scope here

- **Probabilistic sizing / meta-labelling.** Calibrated `P(win|signal)` models (logistic / GBM / conformal) sit downstream of an IC-validated signal. They go in a follow-up directive once we have ≥ 5 TIER_A survivors.
- **ML stacking.** A stacker on `[signal_1, ..., signal_K]` → forward return is only meaningful if every input is IC-validated independently first. Follow-up directive.
- **ML forecasting of the signal itself.** Forecasting RSI is forecasting a deterministic function of price. **Banned** for this census's strategy construction.

### 6.3 Why the lower Sharpe bar is correct

The validated AUD/JPY MR (CI_lo +0.47) and IHYG → VUSD (CI_lo +0.47) sit around +1.0 Sharpe under the corrected math. A new IC survivor with t_NW = 5 and Sharpe of +0.4 with CI_lo > 0 is a **better** deployment candidate than a freshly-discovered +2.0 Sharpe that came from a multi-cell sweep without DSR. The point of the census is to refuse Sharpe inflation, not to reproduce it.

---

## 7. Implementation milestones (week 1)

1. **This directive on `main`.** Pre-registration. No data examination yet. (THIS PR)
2. **Methodology dry-run.** Re-IC AUD/JPY at `vwap_anchor ∈ {6, 24, 96}`, H1, horizons `(1, 8, 40)`, under all §3 gates, with sanctuary held out. Result: either confirms or fails the prior +0.97 / CI_lo +0.47 result. Either is information.
3. **`research/ic_analysis/run_ic_census.py`.** New driver that:
   - Reads the universe from `config/ic_census_universe.toml` (§1) — never hardcoded.
   - Calls existing `run_ic.compute_signals`, `compute_forward_returns`, `compute_ic_table`, `compute_icir_nw`, `apply_bh_fdr`.
   - Adds the missing pieces: DSR p-value (`deflated_sharpe_prob` reuse), 5-fold sign-stability, plateau-neighbour check, MTF-agreement check, sanctuary-window slicing.
   - Wraps every cross-TF aggregation with `assert_causal` (§4.1).
   - Writes `ic_census.parquet` plus a per-instrument CSV in `.tmp/reports/ic_census/`.
4. **Phase-B data download.** `scripts/download_data.py` extended for VIX family, MES/ES/NQ futures, FESX/FTSE/MDAX futures. Phase-B census run after data lands; results merged into the same parquet with `phase=B`.

Steps 1, 2, 3 are independent in the sense that 2 and 3 can proceed in parallel after this directive lands. Step 4 is gated on data download.

---

## 8. Quality gates (run before declaring the census complete)

Mirrors the §11 pre-flight checklist in `SKILL.md`, specialised to this scan:

- [ ] This directive on `main` before any code runs. Commit SHA recorded in the census output header.
- [ ] No `shift(-n)` in any feature factory (`grep -rn "shift(-" research/ic_analysis/` returns only `compute_forward_returns` matches).
- [ ] No `.ffill()` in MTF aggregation paths. Allowed only on the `daily_signal.shift(1).reindex(h1_index, method="ffill")` symmetric-rule pattern with the `.shift(1)` already applied.
- [ ] `assert_causal` runs 5 trials on every aggregation function. Output logs `PASS` per function.
- [ ] Sanctuary window dropped before any signal sees the panel. Logged at scan start: `Sanctuary [start, end]; bars excluded: N`.
- [ ] BH-FDR pool size and α logged.
- [ ] DSR null-expected-max-t and total N logged. Hard t-floor printed.
- [ ] Census output schema matches §6 exactly; CI test asserts column set.
- [ ] Final-validation sanctuary pass run as a **separate** invocation after TIER assignment, **never** in the same run as the gates. The sanctuary-pass column is populated by a follow-up step.
- [ ] Methodology dry-run on AUD/JPY produces the same headline cell or a documented disagreement before the bulk census is trusted.

---

## 9. Negative-result reporting (V3.6)

If the census returns **zero TIER_A survivors**, that is the result. The output report says so, with the failure-mode breakdown (which gate eliminated how many candidates). This is research output, not a wasted afternoon. The next iteration (broader signal classes, longer horizons, conditional features) gets its own pre-registration directive — it does not retroactively rerun the failed scan with relaxed gates.

If the census returns only TIER_B (no MTF agreement), the implication is that the available signal classes are timeframe-specific and the universe needs broadening before MTF-confluence strategies are tractable. Document and stop strategy construction at that point.

---

## 10. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-13 | Initial pre-registration. Phase A universe + gates committed. |
