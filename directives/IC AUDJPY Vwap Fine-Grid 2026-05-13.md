# IC Follow-Up — AUD/JPY × H1 × vwap_overshoot fine grid

**Version:** 1.0 | **Date:** 2026-05-13 | **Author:** Architect
**Status:** **PRE-REGISTRATION** — committed before any data examination on the new grid (V3.1).
**Parent:** `directives/IC Signal Census 2026-05-13.md` (all gates inherited unless overridden below).

---

## 0. Why this exists (V3.6 negative-result documentation)

The parent IC Signal Census ran the `vwap_overshoot` signal on AUD/JPY × H1 with the pre-registered 3-cell grid `{anchor: 6, 24, 96}`. The result:

| anchor | IC (h=1) | t_NW (h=1) | fold-stable (5/5) | BH ✓ | Plateau? |
|---|---|---|---|---|---|
| **6** | -0.0235 | **-6.89** | yes | ✓ | edge cell — not eligible |
| 24 | -0.0091 | -2.71 | no (3/5) | ✓ | n/a — fails t-floor |
| 96 | -0.0023 | -0.68 | no (1/5) | ✗ | n/a — fails BH |

**Verdict under the parent pre-registration:** `unconfirmed` — the best cell sits at the edge of the grid, so plateau-stability cannot be verified. The signal alpha is concentrated at **short anchors**; the grid was too coarse to bracket the peak with two-sided neighbours.

This is **research output, not a failure** (V3.6). The mechanism is clear: vwap_overshoot at short lookback captures genuine 1-bar mean-reversion in AUD/JPY H1, and the IC magnitude decays roughly 10× from anchor=6 to anchor=96. A finer grid centred on the short-anchor regime should either (a) confirm a plateau around anchor=4-8 and produce a TIER-A/B candidate, or (b) fail to find a plateau and conclusively retire the signal.

**The live mr_audjpy strategy uses `vwap_anchor=24`.** Under the parent census this cell shows real but borderline IC (-0.009, t=-2.71) — below the DSR floor for a multi-cell pre-registered scan. The corrected backtest Sharpe (+0.97 / CI_lo +0.47) is consistent with a residual edge at anchor=24, but the IC analysis suggests the bulk of the alpha lives at shorter anchors and the strategy is **operating off-peak**. This follow-up scan answers whether the peak is plateau-stable; whether the strategy should be reconfigured to a different anchor is a separate decision contingent on this scan's outcome.

---

## 1. Pre-registered scope (this directive only)

This is a narrow follow-up. Every other (instrument, signal, timeframe, horizon) combination outside the rows below is **out of scope** of this directive and must not be reported here.

| Instrument | Timeframe | Signal | Horizons |
|---|---|---|---|
| AUD/JPY | H1 | `vwap_overshoot_fine` | 1, 8, 40 |

### 1.1 New parameter grid (pre-committed)

| Cell index | anchor (bars) |
|---:|---:|
| 0 | 3 |
| 1 | 4 |
| 2 | **6** |
| 3 | 8 |
| 4 | 12 |

Geometric-ish spacing concentrated around the suspected peak (anchor=6 in the parent scan). 5 cells gives 3 interior positions (1, 2, 3) eligible to be the headline (edge cells 0 and 4 remain ineligible per V3.2).

### 1.2 Gates (inherited from parent, with one tightening)

All gates from `directives/IC Signal Census 2026-05-13.md` §3 apply unchanged **except**:

- **DSR floor.** N for this follow-up is `5 cells × 3 horizons = 15` (much smaller than the parent's ~9.7k). The null-expected max |t| at N=15 is `sqrt(2·ln 15) ≈ 2.33`. Keeping the parent's hard floor at `|t_NW| > 4.5` is appropriate here too because the floor is calibrated against the **combined** family of scans we are running this week, not just this one. Lowering it would let small-N follow-ups slip past on noise. Floor unchanged: `|t_NW| > 4.5`.

- **Sanctuary.** Same 12-month window as the parent. The follow-up sees exactly the same training-visible bars as the parent did — no information added from the held-out year.

- **Plateau gate.** Headline must be interior (index 1, 2, or 3 of the 5-cell grid). Both adjacent neighbours (headline ± 1) must clear `|t| > 3.0`. |IC| range across `(left, headline, right)` must be < 30% of |headline IC|. **Note:** the gate is local — the rule does not require the cells two steps away (e.g. anchor=3 when headline is anchor=6) to also pass. That follows V3.2 as written; if it turns out the 1-step plateau passes but the 2-step neighbourhood collapses, the result will be reported with that caveat in §4.

### 1.3 Verdict mapping

| Outcome | Action |
|---|---|
| **Plateau passes at any interior cell, IC same sign as parent (negative)** | New TIER_B candidate (no MTF — H1 only). Propose a re-derivation of `mr_audjpy` with the new anchor; full backtest under audit pipeline before any config change. |
| **Plateau passes, but at a different sign than parent** | Statistical anomaly — investigate before any other action. Most likely cause is regime change inside the visible window. |
| **No interior cell clears t-floor and plateau gate** | `vwap_overshoot` is conclusively retired as an IC-justified signal on AUD/JPY H1. The +0.97 Sharpe of the corrected `mr_audjpy` backtest is then attributed to position-sizing discipline + the borderline edge at anchor=24, not to a robust signal. Recommend re-evaluating whether the strategy continues to deploy. |

---

## 2. Out of scope

To stay disciplined and avoid scope creep that would silently expand the deflation-N (which already lives at the family-of-scans level):

- **Not** running fine grids on other vwap-related signals (e.g. EUR/USD, AUD/USD). Those would be separate pre-registrations once the parent census provides initial coverage.
- **Not** running this fine grid at H4 or D for AUD/JPY. The parent scan already showed H4 and D have only borderline IC at the original grid; adding more cells at those TFs is unlikely to change the picture and would inflate N without adding signal.
- **Not** changing the signal definition. `vwap_overshoot` here is the same `close / rolling_mean(close, anchor) - 1.0` formula as the parent. A volume-weighted variant on instruments that have real volume would be a different signal and a different pre-registration.

---

## 3. Implementation

Universe: `config/ic_census_audjpy_vwap_fine.toml` (separate file, references this directive in its header). Signal registry alias: `vwap_overshoot_fine` → same closure as `vwap_overshoot` in `research/ic_analysis/ic_census_lib.signal_factories()`. Identical math; the alias exists so the audit trail keeps the two scans separate.

Runner invocation:

```bash
python research/ic_analysis/run_ic_census.py \
    --universe config/ic_census_audjpy_vwap_fine.toml
```

---

## 4. Result log

Appended once after the scan ran on 2026-05-13. §1 unchanged (V3.1).

### 4.1 Raw cells

#### Horizon h = 1 (next H1 bar)

| anchor | n_bars | IC | t_NW | raw p-value | fold_stable | fold_quorum | BH ✓ |
|---:|---:|---:|---:|---:|:---:|---:|:---:|
| 3 | 87,328 | -0.02649 | -7.7123 | 1.2e-14 | yes | 5/5 | ✓ |
| 4 | 87,328 | -0.02631 | -7.6753 | 1.7e-14 | yes | 5/5 | ✓ |
| 6 | 87,328 | -0.02346 | -6.8907 | 5.6e-12 | yes | 5/5 | ✓ |
| 8 | 87,328 | -0.02065 | -6.0922 | 1.1e-09 | yes | 5/5 | ✓ |
| 12 | 87,328 | -0.01555 | -4.6271 | 3.7e-06 | yes | 4/5 | ✓ |

#### Horizon h = 8 (next 8 H1 bars ≈ 8 hours)

| anchor | IC | t_NW | BH ✓ |
|---:|---:|---:|:---:|
| 3 | -0.01140 | -2.8192 | ✓ |
| 4 | -0.01017 | -2.1927 | ✗ |
| 6 | -0.00691 | -1.2621 | ✗ |
| 8 | -0.00416 | -0.6945 | ✗ |
| 12 | +0.00143 | +0.2194 | ✗ |

#### Horizon h = 40 (next 40 H1 bars ≈ 1.7 days)

All |t| < 1.1. No predictive signal.

### 4.2 Plateau gate

| Horizon | Headline cell | Plateau pass | Reason |
|---|---|:---:|---|
| 1 | anchor=3 | **No** | Headline at edge cell 0 (no two-sided neighbours). |
| 8 | anchor=3 | **No** | Edge cell + h=8 IC too weak (only anchor=3 clears BH). |
| 40 | anchor=12 | **No** | Edge cell + |t| < 4.5 at all cells. |

### 4.3 Interpretation

Every cell at h=1 clears the t-floor (`|t_NW| > 4.5`), passes BH-FDR on its own raw p-value, and is sign-stable in 4-5 of 5 folds. **The signal is real.** But:

1. **IC magnitude is monotonically decaying in anchor.** The data point at every interior cell is consistent with `IC(anchor) ≈ const − k·log(anchor)`. There is no plateau, no interior peak, no anchor at which doubling or halving the lookback leaves IC roughly unchanged.

2. **IC range across the 5 cells is 0.0109, which is 41% of |max IC| = 0.0265.** This fails the V3.2 plateau gate (`< 30%`) even if the headline weren't at an edge cell. There is no interior 3-cell window that satisfies plateau-stability.

3. **The natural mechanism is bid-ask bounce / 1-bar return autocorrelation.** At very short anchors `close / rolling_mean(close, N) - 1` is dominated by the most recent 1-2 bars; the signal effectively measures "how far is close from yesterday's close", and the next-bar return then partially reverses that. In FX H1 the typical half-spread (~0.5-1 pip on AUD/JPY) is large relative to the IC's economic content — `|IC| × σ_fwd_return ≈ 0.026 × ~0.6 normalised vol units` is a small absolute return magnitude, and the cost-aware engine will mostly chew through it.

4. **The mr_audjpy strategy at vwap_anchor=24 sits in the tail of the decay**, where IC has dropped to -0.009 (t=-2.71) — borderline real, below the DSR floor for a pre-registered scan. The corrected backtest's +0.97 Sharpe / CI_lo +0.47 is consistent with that residual edge being partially salvaged by position-sizing discipline, regime gating in the strategy class, and the strategy's own selectivity (it doesn't trade every bar).

### 4.4 Verdict per §1.3

The middle row of §1.3 applies (with an edit reflecting what actually failed): **plateau fails, but not because cells fail t-floor — they all clear it. They fail because IC monotonically decays without an interior peak.** This is mechanistically more informative than a generic "no edge" outcome: the signal is **real but not robust**.

### 4.5 Recommended actions (out of scope for this directive — recorded for follow-up)

A separate decision step, not bundled into this directive. Recorded here so it doesn't slip:

1. **`mr_audjpy` deployment review.** Given the IC analysis shows the signal is monotonically anchor-decaying and the live config sits at the tail of that decay, the +0.97 Sharpe / CI_lo +0.47 audit-corrected number is now interpreted as a **borderline edge held up by execution discipline, not a robust signal**. Open a separate review item to: (a) re-run the full audit pipeline Sharpe under the corrected math on the H1 horizon=1 anchor=6 cell (the IC-peak candidate); (b) compare to the live anchor=24 configuration in a parity test that includes spread + slippage; (c) decide whether to reconfigure or retire the strategy.

2. **Do not retroactively expand this grid.** V3.1. Future shorter-anchor exploration (anchor ∈ {1, 2}) would require a new pre-registration AND a justification that the result is not just measuring 1-bar autocorrelation noise.

3. **Microstructure-aware reformulation as a fresh signal class.** The pattern looks like a microstructure/spread bounce signature. A proper reformulation would account for the half-spread directly (e.g. measure the deviation in units of EWM-spread, then test whether the residual is predictive). That is a new signal class, requires its own pre-registration, and is parked for now.

### 4.6 Outcome record

| Field | Value |
|---|---|
| Tier assigned to all cells | `unconfirmed` |
| Reason | Plateau gate failure: monotonic IC decay, no interior peak, |IC| range > 30% |
| Signal retired on AUD/JPY H1 as IC-justified? | **Yes** under this pre-registration |
| Strategy `mr_audjpy` flagged for review? | **Yes** (recommended action 1 above) |
| Re-pre-registration permitted? | Only as a structurally different signal class (recommended action 3) |
