# IC MTF Lift — Range-Expansion D-frequency probe

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect
**Status:** **PRE-REGISTRATION** — committed BEFORE the D-frequency scan runs (V3.1).
**Parent:** `directives/IC Signal Census Phase C 2026-05-13.md` §6.7 item 1.

---

## 0. Why this exists

Phase C surfaced two TIER_B survivors -- `intraday_range_atr period=14` on **NQ H1** (IC=+0.0248, t_NW=+7.18) and **ES H1** (IC=+0.0231, t_NW=+6.76). Both clear DSR + BH + plateau + fold-stability gates. **The only failed gate is MTF agreement** -- the H1 signal wasn't tested at D for these targets, so the cross-timeframe quorum is 1 of 1 (H1) instead of 2 of 2 (H1 + D).

This directive pre-registers a narrow D-frequency probe of the same signal on the same targets. If the D-cell ALSO clears DSR + BH + plateau + fold-stability with the same sign as H1, the combined `intraday_range_atr` signal gets promoted from TIER_B to TIER_A on those instruments.

The mechanism makes the D test mechanistically plausible: volatility clustering (large-range bars predict more large-range bars) shows up at multiple timescales. If the H1 result is a true mechanism rather than a microstructure artefact, the daily version should show a positive IC too -- though potentially weaker, because aggregating into daily bars smooths the autocorrelation.

---

## 1. Pre-registered scope

This is a focused probe. NO other signals, instruments, timeframes, or horizons are tested.

| Instrument | Timeframe | Signal | Param grid | Horizons |
|---|---|---|---|---|
| MES, MNQ, ES, NQ | **D** | `intraday_range_atr` (existing factory) | `period ∈ {7, 14, 28}` | 1, 5, 21 |
| MES, MNQ, ES, NQ | H1 | `intraday_range_atr` | `period ∈ {7, 14, 28}` | 1, 8, 40 |

The H1 cells are the **same headline cells** Phase C already produced (`period=14, h=1`). Re-running them is necessary because the orchestrator's MTF-agreement computation requires both D and H1 to be in the same headline-output parquet. No new H1 result is expected; the H1 IC + t_NW must match Phase C's numbers within rounding to within at most a 1% drift (sanity).

### 1.1 Gates

Inherited from Phase C unchanged:

- `|t_NW| > 4.5` (DSR-corrected null-max floor at combined N)
- BH-FDR at α=0.05 across the full pool of this scan's p-values
- Fold-stable: signal sign agrees in ≥4 of 5 walk-forward folds (`fold_sign_quorum=4`)
- Plateau-stable: headline at interior cell, both neighbours clear `|t|>3`, |IC| range across the 3 cells < 30%
- MTF agreement: signal clears DSR + BH at ≥2 of the 2 available TFs (D + H1) with the **same sign**
- Sanctuary: trailing 12 months held out

### 1.2 Expected outcome

Three scenarios mapped to actions:

| Outcome | Action |
|---|---|
| D and H1 both clear DSR + BH + plateau + fold with **same sign** | → MTF agree True → **TIER_A promotion** on ES + NQ. Proceed to strategy pre-registration. |
| D cell clears gates with **opposite sign** | → Statistical anomaly. Investigate before any action. The Phase C result and this D result conflict — most likely a sample-period regime artefact. |
| D cell fails one or more gates (especially plateau or t-floor) | → Phase C's TIER_B status holds. Range-expansion is an H1-only edge — fine, the directive's §6.7-2 strategy proposal route is still open, just at TIER_B not TIER_A. |
| MES / MNQ also pass with same sign | → Bonus confidence. Same instrument family, smaller sample → likely lower t but still useful as a robustness check. |

---

## 2. Out of scope

- **Not** running this on Phase A microstructure signals (rsi_dev, bb_pctb, etc.) — their D-frequency results from Phase A already exist and failed plateau there. No new information.
- **Not** running on non-equity-futures targets — the Phase C TIER_B was specifically on US equity futures. Lifting to D on SPY/QQQ would be a different test (yfinance D vs Databento futures D — basis & adjustments differ).
- **Not** changing the parameter grid — same `{7, 14, 28}` as Phase C / Phase A. V3.2 plateau-stability inherits as-is.

---

## 3. Implementation

1. **This directive on `main`.** (THIS PR)
2. `config/ic_census_universe_mtf_lift.toml` — focused universe TOML.
3. Re-run the existing `run_ic_census.py` against the new universe.
4. Append result log to §4.

---

## 4. Result log

Appended 2026-05-14 after the scan ran. §1-§3 unchanged (V3.1).

### 4.1 H1 reproducibility check (sanity)

The Phase C TIER_B cells re-emerged from this scan with identical numbers (cells deterministic given same data + same gate config):

| signal | instrument | TF | h | params | IC | t_NW | reproducibility vs Phase C |
|---|---|---:|---:|---|---:|---:|:---:|
| intraday_range_atr | NQ | H1 | 1 | period=14 | +0.02481 | +7.1832 | Match |
| intraday_range_atr | ES | H1 | 1 | period=14 | +0.02313 | +6.7643 | Match |

Scaffolding is deterministic. The TIER_B headline stands.

### 4.2 D-frequency probe — outcome scenario 3 (the most likely)

| instrument | TF | h | IC | t_NW | clears DSR? | clears plateau? |
|---|---:|---:|---:|---:|:---:|:---:|
| ES | D | 1 | -0.00273 | -0.18 | No | No |
| ES | D | 5 | +0.01039 | +0.65 | No | No |
| ES | D | 21 | -0.00720 | -0.35 | No | No |
| NQ | D | 1 | -0.01986 | -1.30 | No | No |
| NQ | D | 5 | +0.01014 | +0.66 | No | No |
| NQ | D | 21 | -0.03487 | -1.76 | No | No |
| MES | D | 1 | -0.01930 | -0.83 | No | No |
| MNQ | D | 1 | -0.02100 | -0.89 | No | No |

**Every D-cell fails every gate.** The strongest |t_NW| is 1.76 (NQ D h=21) and it has the OPPOSITE sign to the H1 cells. The signal is essentially zero at daily frequency.

### 4.3 Interpretation

The range-expansion / volatility-clustering mechanism is **fundamentally an intraday phenomenon**. The autocorrelation of large-range bars dissipates as the bar-size grows. Aggregating to daily smooths the signal away completely. Two consequences:

1. **TIER_B status holds — no TIER_A promotion.** Per §1.2 outcome scenario 3: "TIER_B holds, strategy proposal route still open at TIER_B not TIER_A."
2. **The mechanism is microstructural, not macro.** This means the natural execution venue is **intraday on IG DFB** (US Tech 100, US 500), not a daily rebalancing on IBKR equity futures. Note: trading the signal still requires CME futures + an H1 timer in some venue; the IG DFB path is mechanically simpler for short-term execution but loses tax efficiency. Both routes are open.

### 4.4 Outcome record

| Field | Value |
|---|---|
| TIER_A promotion achieved? | **No** |
| TIER_B status preserved? | **Yes** |
| D-frequency range_atr deployment-eligible? | **No** — fails every gate |
| H1 reproducibility of Phase C numbers? | **Match** |
| Mechanism classification | Microstructural / intraday-only |
| Implied execution venue | Intraday (IG DFB short-term, OR CME futures with H1 timer) — not daily rebal |

### 4.5 Negative-result hygiene (V3.6)

This is documented null result, not a wasted run. The information value is twofold:

- **Confirms H1 is the right timeframe** for range_atr deployment. If a future researcher considers re-applying this signal at D, they'll find this directive showing the test has already been run and failed -- saving the rerun.
- **Pre-empts the lazy "let's also try D" question** during the strategy backtest pre-registration. The strategy spec can confidently scope to H1 only.

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial pre-registration. |
