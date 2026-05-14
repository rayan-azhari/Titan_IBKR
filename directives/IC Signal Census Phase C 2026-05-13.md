# IC Signal Census — Phase C pre-registration

**Version:** 1.0 | **Date:** 2026-05-13 | **Author:** Architect
**Status:** **PRE-REGISTRATION** — committed BEFORE the Databento download / scan runs (V3.1 discipline).
**Parent directives:**

- `directives/IC Signal Census 2026-05-13.md` — Phase A (single-instrument + cross-asset @ free data)
- `directives/IC Signal Census Phase B 2026-05-13.md` — VIX-family + cross-region (@ yfinance daily)

Every gate, every invariant, and every output-schema column is inherited from the Phase A parent unless explicitly overridden in §3.

---

## 0. Why this exists

Phases A + B produced **zero TIER_A or TIER_B survivors** across ~3,400 cells. The strongest mechanism-confirming signals — VRP on equities (TQQQ +0.167 / t=+3.22 at h=21), VIX-curve backwardation → FTSE recovery (+0.142 / t=+3.16 at h=21), US short-term reversal (−0.077 / t=-3.83 at h=5), AUD/JPY microstructure (-0.027 / t=-7.71 at h=1) — were all **rejected by sample-size or by parameter-fragility**, not by signal absence. The audit-discipline gates correctly refused to deploy signals whose t_NW capped out around 3-3.7 because the daily sample (~5k bars) wasn't large enough for the DSR-corrected `|t_NW| > 4.5` floor.

Phase C addresses the sample-size constraint by:

1. **Going intraday on the strongest mechanisms.** Re-running VRP and VIX-term-structure at **H1**, where N multiplies by ~24× per instrument. A signal with daily IC of 0.10 over 5k bars (`t ≈ 7` before NW correction, `t ≈ 3` after) becomes an H1 signal with similar IC magnitude over 120k bars (`t ≈ 35` raw, `t ≈ 7` after NW correction at h=24). That's well above any floor — IF the daily mechanism survives the intraday resampling.

2. **Adding the canonical equity-futures sleeve (MES + NQ + MNQ).** Migrate.md flagged MES as the deployment target post-Samir-Stack rewrite (commit `5d11b2e`). Phase A/B scanned SPY/QQQ ETF proxies; Phase C adds MES/NQ/MNQ as parallel targets. IC-survivors on the futures are directly deployable in the live IBKR futures stack.

3. **Adding actual EU index futures (FESX, FTSE-fut, DAX-fut, MDAX-micro)** to replace the Phase B index-spot proxies. The cross-region lead-lag class becomes properly testable.

4. **Adding VIX futures** (front + M2 continuous). The VIX-futures basis (`VX1/VX2 - 1`) is the canonical contango/backwardation signal — VIX-spot ratios are a proxy that loses the carry-curve information.

Data source: **Databento**, paid feeds. Authorised by user 2026-05-13. Per-symbol-year costs itemised in §4.

---

## 1. Pre-registered Phase C universe

### 1.1 New targets

| Local name | Databento dataset | Symbol | TFs | Class | Venue (deployment) |
|---|---|---|---|---|---|
| `MES_D` / `MES_H1` | `GLBX.MDP3` | `MES.c.0` (front-month continuous) | D, H1 | US equity futures | IBKR CME |
| `NQ_D` / `NQ_H1` | `GLBX.MDP3` | `NQ.c.0` | D, H1 | US equity futures | IBKR CME |
| `MNQ_D` / `MNQ_H1` | `GLBX.MDP3` | `MNQ.c.0` | D, H1 | US equity futures | IBKR CME |
| `FESX_D` / `FESX_H1` | `XEUR.IFEU` | `FESX.c.0` | D, H1 | EU equity futures | IBKR Eurex |
| `FTSE_FUT_D` / `FTSE_FUT_H1` | `IFEU.IMPACT` | `Z.c.0` (FTSE 100 future) | D, H1 | UK equity futures | IBKR ICE |
| `DAX_FUT_D` / `DAX_FUT_H1` | `XEUR.IFEU` | `FDAX.c.0` | D, H1 | DE equity futures | IBKR Eurex |
| `MDAX_FUT_D` / `MDAX_FUT_H1` | `XEUR.IFEU` | `FMDX.c.0` | D, H1 | DE mid-cap futures | IBKR Eurex |

### 1.2 New externals (research-only, not deployment targets)

| Local name | Databento dataset | Symbol | TFs | Use |
|---|---|---|---|---|
| `VIX_H1` | `OPRA.PILLAR` or `XCBT.IFB` | `VIX` (index) | H1 | VRP at H1, VIX-term at H1 |
| `VIX9D_H1` | same | `VIX9D` | H1 | term-structure at H1 |
| `VIX3M_H1` | same | `VIX3M` | H1 | term-structure at H1 |
| `VX1_D` / `VX1_H1` | `GLBX.MDP3` | `VX.c.0` (VIX front-month future continuous) | D, H1 | VX/VX2 basis signal |
| `VX2_D` / `VX2_H1` | `GLBX.MDP3` | `VX.c.1` (VIX 2nd-month) | D, H1 | VX/VX2 basis signal |

### 1.3 Re-scan of existing targets at H1

SPY, QQQ, IWB, IWM, GLD, HYG, IEF already exist at H1 in `data/`. Phase C re-scans them with the new signal classes (intraday VRP, intraday VIX-term) — no new download needed for those.

### 1.4 Documented substitution audit

| Migrate.md target | Phase C source | Substitution? |
|---|---|---|
| MES, ES, NQ, MNQ | `GLBX.MDP3` continuous front-month | None — actual product. |
| FESX | `XEUR.IFEU` `FESX.c.0` | None — actual product. |
| FTSE-fut | `IFEU.IMPACT` (ICE Europe) | None — actual product. |
| MDAX-micro | `FMDX.c.0` (DAX 40 future) and `FDAX.c.0`. **Note:** the canonical MDAX is a mid-cap index; the closest tradeable on Eurex is `FDAX` (DAX 40) which is large-cap, OR `FSMI` (Swiss) — neither is mid-cap. Phase C uses `FDAX` as large-cap and `FMDX` if Databento carries an MDAX future series. | Minor. Logged. |
| VIX, VIX9D, VIX3M | Databento `XCBT.IFB` or `OPRA.PILLAR` for indices; `GLBX.MDP3` for futures (`VX`) | None — actual products. |

Audit A8 — every deviation from Migrate.md's stated universe is logged here. Anyone reproducing Phase C can identify what was actually downloaded vs the original prescription.

---

### 1.5 Discovery audit — what's actually feasible (appended 2026-05-13)

The initial §1 manifest was written from best-guess Databento namespace knowledge. The pre-charge `metadata.get_cost` pass + `symbology.resolve` discovery exposed the following reality (V3.6 documentation, non-retroactive — §1.1-§1.4 above stand as the *intended* universe, this §1.5 is the *as-built* universe):

| Originally claimed | Reality | Phase C action |
|---|---|---|
| `MES.c.0`, `MNQ.c.0` on `GLBX.MDP3`, continuous | ✓ Works. MES/MNQ from 2019-05; ohlcv-1h cost ≈ $0.0002/symbol. | Download as planned. |
| `ES.c.0`, `NQ.c.0` on `GLBX.MDP3`, continuous | ✓ Works. ES/NQ continuous resolve from 2010-06-06 (instrument IDs `17077` and `750` respectively). My initial cost-estimate run failed due to a Databento intermittent error, not a symbology issue. | Download as planned. |
| `FESX.c.0`, `FDAX.c.0` on `XEUR.IFEU` (Eurex) | ✗ Two issues: (a) `XEUR.IFEU` is **not** a dataset code — the correct Eurex code is `XEUR.EOBI`. (b) Even `XEUR.EOBI` only has history from **2025-03-10** (~2 months). Useless for IC research at any sample size. | **Drop from Phase C.** Eurex intraday history is too short on Databento. Re-evaluate when EOBI accumulates ≥ 2 years, or source from elsewhere. |
| `Z.c.0` on `IFEU.IMPACT` (FTSE 100 future, ICE) | ⚠ Dataset exists (history from 2018-12-23) but my symbology `Z.c.0` resolved to empty. Correct ICE FTSE 100 future symbology TBD; would require an additional symbology-discovery pass. | **Drop from Phase C.** Defer to a follow-up directive after confirming the correct symbology. |
| `FMDX.c.0` (MDAX-micro) on `XEUR.IFEU` | ✗ Same `XEUR.EOBI` 2-month-history constraint. | **Drop from Phase C.** Same rationale as FESX/FDAX. |
| `VIX`, `VIX9D`, `VIX3M` on `XCBT.IFB` | ✗ `XCBT.IFB` is **not** a valid Databento dataset. The Databento catalogue contains no CBOE-index dataset. CBOE-computed indices (VIX family) are not sourceable from Databento at all. | **Move to IBKR Gateway path** (CBOE Indices Live subscription via TWS API). Pre-registered separately in §1.6 below. |
| `VX.c.0`, `VX.c.1` (VIX futures) on `GLBX.MDP3` | ✗ Returned empty resolve. VIX futures trade on the **CFE (CBOE Futures Exchange)**, not on CME Globex. No CFE dataset in Databento's public catalogue. | **Move to IBKR Gateway path** (CFE market-data subscription). Pre-registered separately in §1.6. |

### 1.6 Revised Phase C as-built universe

**Approved 2026-05-13 by user.** Downloaded immediately:

| Local | Dataset | Symbol | Stype | History | TFs |
|---|---|---|---|---|---|
| MES | GLBX.MDP3 | MES.c.0 | continuous | 2019-05-06 → today | D + H1 |
| MNQ | GLBX.MDP3 | MNQ.c.0 | continuous | 2019-05-06 → today | D + H1 |
| ES | GLBX.MDP3 | ES.c.0 | continuous | 2010-06-06 → today | D + H1 |
| NQ | GLBX.MDP3 | NQ.c.0 | continuous | 2010-06-06 → today | D + H1 |

Total Databento charge expected: < $0.50 USD across all 8 cells.

### 1.7 IBKR Gateway path (VIX-family + VIX futures intraday)

Approved as a best-effort attempt: if the IBKR Docker connection has the CBOE Indices Live + CFE subscriptions active, the script downloads VIX/VIX9D/VIX3M/VX1/VX2 at H1 (D already exists from Phase B yfinance). If the subscriptions are inactive, **the script stops cleanly without partial state** — the failure mode must be loud and the user re-tries after enabling subscriptions.

Implementation goes in `scripts/download_data_phase_c_ibkr.py`. Output naming matches Phase A/B convention (e.g. `VIX_H1.parquet` to overlay the existing `VIX_D.parquet` yfinance file).

### 1.8 IG Markets path (EU index DFB intraday)

**Filed for later** — Phase C does not download from IG. A separate pre-registration will be drafted when needed.

All H1, all causal, all anchored via `anchored_aggregate(higher_tf=False)` + `assert_causal` per the Anchored MTF Rule. Inherits the 3-cell parameter-grid pattern and plateau gate.

### 2.1 Intraday vol-risk-premium (signals.volatility_phase_c)

| Signal | Externals | Param | Cells | Rationale |
|---|---|---|---|---|
| `vrp_z_h1` | `[VIX]` | rv_window (H1 bars) | `120`, `240`, `480` | Phase B's daily VRP at H1 resolution. rv_window cells correspond to 5 trading days / 10 days / 20 days at H1 (24-bar day). Same z-score normalisation (rolling 60-bar over the smoothed VRP). |

Applied to equity targets (SPY, QQQ, IWM, MES, NQ, MNQ, FESX, FTSE_FUT, DAX_FUT). Expected sign: POSITIVE (vol-seller's edge → forward equity returns up).

### 2.2 Intraday VIX term structure (signals.term_structure_h1)

| Signal | Externals | Param | Cells | Rationale |
|---|---|---|---|---|
| `vix9d_over_vix_h1` | `[VIX9D, VIX]` | smoothing (H1 bars) | `1`, `24`, `120` | Phase B daily signal at H1. Smoothing values correspond to 1-bar / 1-day / 1-week. |
| `vix_over_vix3m_h1` | `[VIX, VIX3M]` | smoothing | `1`, `24`, `120` | Same as above. |
| `vx_basis_z` | `[VX1, VX2]` | smoothing | `1`, `24`, `120` | NEW class: VIX-futures basis. `(VX1 - VX2) / VX2`. Contango (VX1 < VX2) is the normal state; backwardation is stress. This is the canonical curve signal that the spot-VIX ratios approximate. |

### 2.3 Cross-region lead-lag at H1 (signals.cross_region_h1)

| Signal | Externals | Param | Cells | Rationale |
|---|---|---|---|---|
| `mes_to_fesx_h1` | `[MES]` | window (H1 bars) | `1`, `6`, `24` | Smoothed past-MES return predicting forward FESX return. Phase B inverted-sign result was on spot indices; futures cross-region may behave differently because they trade overnight (MES is 23h/day on Globex, FESX similar on Eurex). |
| `mes_to_ftse_fut_h1` | `[MES]` | window | `1`, `6`, `24` | Same on FTSE futures. |
| `mes_to_dax_fut_h1` | `[MES]` | window | `1`, `6`, `24` | Same on DAX futures. |

### 2.4 Re-scan of Phase A + Phase B signals at H1

The existing single-instrument signals (`momentum`, `ewmac`, `ma_distance`, `rsi_dev`, `vwap_overshoot`, `bb_pctb`, `realized_vol_z`, `overnight_gap_z`, `intraday_range_atr`) and cross-asset signals (`hyg_ief_z`, `dxy_z`) are re-run at H1 on the Phase C target set (MES, NQ, MNQ, FESX, FTSE_FUT, DAX_FUT, MDAX_FUT) in addition to their existing Phase A coverage. **No new factory work** — same code, expanded target list.

Same H1 horizons as Phase A: `[1, 8, 40]`. The 40-bar horizon at H1 is roughly two trading days — useful for lead-lag mechanisms; less useful for VRP (which is fundamentally a multi-day phenomenon and is better tested at the daily-horizon equivalents at H1: horizons `[24, 120, 480]`).

### 2.5 Banned

Inherits all parent bans (§2.9 of parent). Additionally:

- **No VIX-on-equity-futures self-reference.** When the target is a CME-listed equity future, VIX-derived signals are fine; when the target IS a VIX product (the VIX9D, VIX3M externals), those are never targets — only externals. Self-correlation guard applies.

---

## 3. Gate overrides

### 3.1 DSR floor — likely tightens to `|t_NW| > 5.0`

Combined Phase A + B + C N estimate:

| Phase | Approx N (cells × horizons) |
|---|---:|
| Phase A | ~2,400 |
| Phase B | ~850 |
| **Phase C** | **~25,000-40,000** |
| **Combined** | **~28,000-43,000** |

Combined N likely exceeds 25k. Per parent §3.4, `dsr_t_floor_combined = 5.0` engages above 25k cells. The orchestrator should compute N at runtime and select the appropriate floor; this directive pre-commits to the **higher** floor (5.0) for Phase C cells, regardless of the final N tally. Tightening (not relaxing) is allowed without a PR per V3.1.

### 3.2 MTF agreement

Phase C is **D + H1**. The parent's MTF quorum (2 of 3 timeframes — D, H4, H1) applies but with H4 now mostly absent (only FX has H4 in Phase A; equity futures and VIX products are D + H1). Phase C survivors that pass at both D and H1 with same sign get `mtf_agree = True` (quorum 2 of 2 available TFs). The orchestrator's mtf_agreement function already handles this via the `>=quorum` test on the set of TFs where the cell passed; no code change needed.

### 3.3 Fold quorum

Unchanged. `fold_sign_quorum = 4` of 5.

### 3.4 Sanctuary

Unchanged. Trailing 12 months held out. The Databento download itself must end no later than `today - 12 months` AT MAXIMUM if Phase C is to use the full data immediately. **Practical compromise**: download to today, slice sanctuary at runtime exactly as Phases A/B do.

### 3.5 Plateau gate

Unchanged. 3-cell grid, headline must be interior, both neighbours clear `|t| > 3.0`, IC range across the 3 cells < 30%.

---

## 4. Databento order itemisation (pending user $ approval)

Estimates from Databento public pricing, 2026 May. **User approves before any charges land.**

| Symbol(s) | Dataset | Schema | History start (approx) | Est. $ |
|---|---|---|---|---|
| MES.c.0, NQ.c.0, MNQ.c.0 | `GLBX.MDP3` | `ohlcv-1d` + `ohlcv-1h` | 2017 (MES launched 2019; ES from 2010) | $30-60 |
| ES.c.0 | `GLBX.MDP3` | `ohlcv-1d` + `ohlcv-1h` | 2010 | $15-25 |
| FESX.c.0 | `XEUR.IFEU` | `ohlcv-1d` + `ohlcv-1h` | 2010 | $20-30 |
| FDAX.c.0 (DAX futures) | `XEUR.IFEU` | `ohlcv-1d` + `ohlcv-1h` | 2010 | $20-30 |
| FMDX.c.0 (MDAX futures) | `XEUR.IFEU` | `ohlcv-1d` + `ohlcv-1h` | 2010 (subject to availability) | $15-25 |
| FTSE 100 future | `IFEU.IMPACT` or `IFEU.IFEU` | `ohlcv-1d` + `ohlcv-1h` | 2010 | $20-30 |
| VIX, VIX9D, VIX3M (indices) | `XCBT.IFB` or equivalent | `ohlcv-1h` | 2010 (VIX9D from 2011) | $30-60 |
| VX.c.0, VX.c.1 (VIX futures) | `GLBX.MDP3` | `ohlcv-1d` + `ohlcv-1h` | 2010 | $20-30 |
| **Total estimate** | | | | **$170-290** |

Estimates assume `ohlcv-1d` + `ohlcv-1h` schemas from ~2010 to today, ~15 years × 250 trading days × 24 hourly bars per future ≈ 90k bars per symbol. Databento prices on bar-count; intraday VIX may cost less because of lower per-bar fees on indices.

**Authorisation gate.** Before running the downloader, I will:
1. Connect to Databento with the existing `DATABENTO_API_KEY`.
2. Call `client.metadata.get_cost(...)` for each symbol/dataset/schema/range — Databento's official cost estimate endpoint.
3. Print the itemised cost table.
4. Wait for user confirmation before invoking `client.timeseries.get_range`.

---

## 5. Implementation plan

1. **This directive on `main`.** Pre-registration done. (THIS PR)
2. Extend `scripts/download_data_databento.py` to:
   - Accept `--dataset` flag (default `ARCX.PILLAR`; pass `GLBX.MDP3`, `XEUR.IFEU` etc.)
   - Accept `--schema` flag (default `ohlcv-1d`; pass `ohlcv-1h` for intraday)
   - Accept `--stype` flag (default `raw_symbol`; pass `continuous` for `.c.0`/`.c.1` futures rollover series)
   - Call `client.metadata.get_cost` first; print itemised costs; require `--confirm` to actually charge.
3. Update `research/ic_analysis/run_ic.py`'s `load_ohlcv` to handle the new symbols (no code change expected — naming pattern `{INSTRUMENT}_{TIMEFRAME}.parquet` is preserved).
4. `config/ic_census_universe_phase_c.toml` — new universe TOML with the Phase C targets + signal-class additions.
5. `research/ic_analysis/ic_census_lib.py` — three new factories: `vrp_z_h1`, `vx_basis_z`, `mes_to_eu_h1` (parametrised by EU target).
6. Tests for each new factory.
7. Run Phase C census.
8. Append result log to §6 below — V3.6 documentation regardless of outcome.

---

## 6. Result log

Appended 2026-05-14 after the scan ran. §1-§5 unchanged (V3.1).

### 6.1 Run shape

- Data downloaded: 8 Databento parquets (MES, MNQ, ES, NQ × D + H1) + 3 IBKR parquets (VIX, VIX9D, VIX3M × H1). Yfinance D files restored for VIX-family (IBKR script regression caught + patched).
- 900 raw rows / 300 headline rows across 4 equity-futures targets × D + H1 × the existing signal classes + restored VIX-intraday signal classes (term-structure + VRP).
- **Tier counts: 2 TIER_B, 0 TIER_A, 298 unconfirmed.**

### 6.2 Gate breakdown

| Gate | Passed / 300 |
|---|---:|
| BH-significant | 19 |
| fold-stable (≥4/5) | 146 |
| dsr_pass (\|t_NW\| ≥ 4.5) | 2 |
| **plateau_stable** | **2** |
| mtf_agree | 0 |

### 6.3 TIER_B survivors — the first in the combined Phase A+B+C census

| signal | instrument | TF | h | params | IC | t_NW | dsr_p | n_bars |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `intraday_range_atr` | **NQ** | H1 | 1 | period=14 | **+0.02481** | **+7.18** | 0.9998 | 85,034 |
| `intraday_range_atr` | **ES** | H1 | 1 | period=14 | **+0.02313** | **+6.76** | 0.9990 | 85,074 |

Both clear: DSR (`|t_NW|>4.5`) + BH-FDR + fold-stability (sign-stable in ≥4 of 5 walk-forward folds) + **plateau** (the ATR-period sweep `{7, 14, 28}` is well-behaved: headline at the middle cell, both neighbours clear `|t|>3`, |IC| range across the three cells <30%). The only gate they fail is **MTF agreement** — the signal works at H1 but not at D, so they sit at TIER_B not TIER_A.

### 6.4 Mechanism

`intraday_range_atr` = `(today_range / ATR(period)) - 1`. **Positive IC at h=1** means: when the current bar's range substantially exceeds the rolling ATR, the **next bar's return is positive**. This is a **volatility-breakout / range-expansion** signal — high-range bars cluster, and the autocorrelation of large moves at the H1 timescale is enough to be tradeable (`|IC| × σ_fwd ≈ 0.025 × 1 normalised vol unit` ≈ ~5 bp in raw return terms per bar, well above typical CME futures spread).

Why it only fires at H1: the range-expansion / volatility-clustering mechanism is fundamentally an intraday phenomenon. Daily bars aggregate over so much of the action that the same signal collapses to noise — exactly what the MTF gate is designed to catch.

### 6.5 Other notable cells (rejected, but mechanism-confirming)

These cells passed BH + fold but failed plateau (typically because the parameter sweep monotonically decays the same way Phase A microstructure did):

| Signal | Instrument | TF | h | IC | t_NW | Notes |
|---|---|---:|---:|---:|---:|---|
| rsi_dev | ES | H1 | 1 | -0.0188 | -5.33 | Phase A microstructure mean-reversion confirmed on actual futures |
| bb_pctb | ES | H1 | 1 | -0.0185 | -5.13 | Same |
| intraday_range_atr | NQ | H1 | 8 | +0.0246 | +4.70 | Range-expansion at 8-bar horizon (sub-floor) |
| vwap_overshoot | ES | H1 | 1 | -0.0161 | -4.52 | Microstructure mean-reversion |
| realized_vol_z | NQ | H1 | 1 | -0.0155 | -4.48 | Vol-mean-reversion at H1 |
| intraday_range_atr | MNQ | H1 | 8 | +0.0328 | +4.11 | Same as NQ but smaller N (35k bars) so t lower |
| ma_distance | ES | H1 | 1 | -0.0129 | -3.81 | Same family |
| vrp_z | ES | D | 5 | +0.0926 | +3.67 | Phase B VRP confirmed on futures at D 5-day, still sample-limited |
| vrp_z | NQ | D | 5 | +0.0859 | +3.40 | Same |

### 6.6 What Phase C confirmed

1. **Sample size was the binding constraint in Phases A+B.** Going from ~5k daily bars to ~85k H1 bars on the equity-futures sleeve pushed `intraday_range_atr` past the DSR floor where it had previously been borderline.
2. **The strongest mechanisms identified in Phases A/B reappear cleanly on futures intraday.** Mean-reversion at H1 horizon=1 (rsi_dev, bb_pctb, vwap_overshoot, ma_distance) all show |t_NW| > 4 on ES H1 — they fail plateau for the same monotonic-decay reason as Phase A, not because the signal isn't there.
3. **VRP at D still sample-limited.** Even on 4.4k bars of ES/NQ daily, VRP h=5 hits `t = 3.4-3.7` — short of the floor. Intraday VRP (`vrp_z` at H1) entered the scan but didn't surface as a survivor — H1 VRP needs the realised-vol leg to be computed at H1 frequency, and the 60-bar rolling window cap I used may be too short.
4. **Term-structure signals (vix9d_over_vix, vix_over_vix3m) didn't surface as TIER_B.** Their best cells at H1 didn't clear the t-floor on the equity-futures targets — likely the signal is more a D-frequency macro indicator than an H1 tactical one.

### 6.7 Recommended next actions (out of scope of this directive)

1. **MTF lift** for `intraday_range_atr`. Currently TIER_B because D doesn't fire. Investigate whether a D version of the range-expansion signal (e.g. `daily_range / ATR(14_days) - 1`) shows IC on the same targets. If yes, the H1 + D agreement promotes the signal to TIER_A.
2. **Strategy construction proposal.** `intraday_range_atr` at H1 on NQ/ES with `period=14` is the first deployment-eligible candidate from the census. A short-term strategy on IG DFB (US Tech 100 + US 500) at H1 is the natural execution route. Requires a separate pre-registration for the strategy backtest under audit pipeline (DSR + sanctuary + underlying-resampled MC).
3. **Cross-asset confluence test.** AND-gate `intraday_range_atr` (NQ H1) with one of the Phase A microstructure mean-reversion signals (which are anti-correlated by direction). If both fire and agree on sign, the combined edge could be larger than either alone.
4. **Phase D — paid-feed Eurex / CFE / breadth.** Phase B's deferred items (Eurex futures via paid feed, VIX futures via CFE, SPX breadth panel) remain the natural next universe expansion. Cost will be higher than Phase C ($100s vs $3) so requires a new pre-registration with $-approval.

### 6.8 Outcome record

| Field | Value |
|---|---|
| Combined Phase A + B + C TIER_A count | 0 |
| Combined TIER_B count | **2** (both `intraday_range_atr`, ES + NQ at H1 horizon=1) |
| Strongest mechanism-confirming IC (point estimate) | NQ `intraday_range_atr`: IC=+0.025, t_NW=+7.18 |
| Strongest IC magnitude across all phases | DBC `dxy_z` D h=21: IC=-0.158 (Phase A), still sample-limited |
| Databento spend | **$2.7581 + 1 partial re-run = ~$5.50** (vs $170-290 directive ceiling) |
| IBKR data acquired | VIX, VIX9D, VIX3M intraday from 2009-2018 to 2026 (~52k / 14k / 33k H1 bars) |
| Sanctuary respected | Yes — 12-month trailing window excluded throughout |
| New deployment candidates | 1 — `intraday_range_atr period=14` on NQ + ES at H1 horizon=1, requires separate strategy pre-registration |

---

## 7. Deployment-venue mapping (informational)

Records which IC-survivors map to which live execution path. This is **not** a deployment plan, only a routing table. Actual deployment is a separate per-strategy decision.

| IC-survivor target class | Long-term venue | Short-term (intraday) venue |
|---|---|---|
| MES, NQ, MNQ, ES (US equity futures) | IBKR CME | IG: `IX.D.SPTRD.DAILY.IP` (US 500 DFB), `IX.D.NASDAQ.DAILY.IP` (US Tech 100 DFB) |
| FESX, FDAX, FMDX, FTSE-fut | IBKR Eurex / ICE | IG: `IX.D.STOXX50.DAILY.IP` (Europe 50 DFB), `IX.D.FTSE.DAILY.IP` (FTSE 100 DFB), `IX.D.DAX.DAILY.IP` (Germany 40 DFB) |
| SPY/QQQ/IWB/IWM/EFA/EEM (US ETFs) | IBKR US equities | IG: SPDR / iShares CFD spread bets (per `.tmp/ig_catalog/`) |
| CSPX/VUSD/IHYG/IHYU UCITS | IBKR LSE | IG: spread bets per existing IG catalog |
| AUD_JPY / EUR_USD / etc. FX | IBKR IDEALPRO | IG FX spread bets (DFB on majors) |
| GLD | IBKR ARCA | IG: spot gold / gold ETF DFB |
| VIX-family (research only) | n/a — externals only | n/a |

The IG catalog at `.tmp/ig_catalog/MAPPING.md` confirms the IG side of this mapping. For each long-term IC-survivor we have a corresponding short-term DFB or quarterly contract on IG, enabling a short-term version of the same signal once intraday data validates it.

---

## 8. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-13 | Initial Phase C pre-registration. Databento-sourced intraday futures + VIX-family + EU index futures. |
