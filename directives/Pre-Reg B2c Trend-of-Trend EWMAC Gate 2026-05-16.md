# Pre-Registration — B2c: Trend-of-Trend Regime Filter on Carver EWMAC

**Author:** rayanazhari (planner) + Claude orchestrator (Architect/Researcher)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** L48 follow-up to B2/B2b. Tests whether a broad-market trend-regime filter rescues the cross-asset universal-trend EWMAC negative Sharpe.
**Predecessors:**

- `directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md` (B2 IBKR-3y commodity: +2.02 Sharpe BUT regime-artifact per L48)
- `directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md` §7 (B2b yfinance-21y cross-asset: -0.28 Sharpe, FALSIFIED universal-trend hypothesis)
- Carver, *Systematic Trading* Ch. 9: "Trend Filter" pattern for raising the bar on trade entry.

**Status:** §1–§3 frozen at commit (BEFORE any audit data is examined under the new filter); §4 result log appended post-audit.

> **V3.1 pre-registration honesty note:** the filter mechanism + cells + selection rule are committed in this directive BEFORE running anything. No filter design alteration after seeing the audit numbers. If the pre-reg cells all fail, we accept the verdict (filter doesn't rescue B2) and add a new pre-reg for the next mitigation; we do NOT iterate on filter parameters post-hoc.

---

## §1. Motivation & mechanism

**L48 in plain language.** B2 EWMAC's +2.02 Sharpe on the 24-commodity IBKR-3y window was a 2023-2025 commodity-trend-regime artifact. On a 21y cross-asset 31-instrument universe (B2b yfinance), the same strategy returned **-0.28 Sharpe**, with all 5 plateau cells negative. The universal-trend hypothesis is **falsified**. The narrow-regime hypothesis remains alive: B2 may work specifically when commodity (or broad-asset) trends are in a "trend-friendly" regime, and it loses money outside that regime.

**B2c hypothesis:** A binary on/off gate based on a slow EWMAC of a broad equal-weight universe index will:

- Keep B2 active during identified "trend regime" sub-periods (recovering the +2.02-style Sharpe contribution from those periods).
- Force the strategy flat outside the trend regime (avoiding the -0.5 to -0.8 Sharpe drag from non-trend regimes).
- Net effect: rescue Sharpe to materially-positive territory across the full 21y sample.

**Mechanism (per audit-window date t):**

1. **Broad-universe index construction (causal, scale-invariant).** Compute the cumulative log-return mean across all N instruments since some baseline t0:
   `U(t) = exp(mean over i of log(close_i(t) / close_i(t0)))`. This is unit-free (FX vs equity scale doesn't matter), equal-weighted across the universe, causal (uses only data through t).
2. **Broad-trend EWMAC.** For a pre-registered speed `(fast_hl, slow_hl)`:
   `broad_ewmac(t) = EWMA(log(U), halflife=fast_hl) - EWMA(log(U), halflife=slow_hl)`.
3. **Regime gate** (binary, with optional deadband):
   - `gate(t) = 1` if `broad_ewmac(t) > +deadband`
   - `gate(t) = 0` if `|broad_ewmac(t)| <= deadband` (flat — no trades)
   - `gate(t) = 1` if `broad_ewmac(t) < -deadband` AND `mode == "absolute_trend"` (trend in either direction qualifies)
   - `gate(t) = -1` if `broad_ewmac(t) < -deadband` AND `mode == "directional"` (negative trend → flip per-asset signal)
4. **Per-asset gated forecast.** `gated_forecast_i(t) = gate(t) * raw_forecast_i(t)` (where `raw_forecast_i` is B2's existing combined EWMAC).
5. **Position sizing & costs** unchanged from B2.

Causality: all EWMAs use `.shift(1)` discipline; gate at date t uses data through t-1; positions held effective at t earn r(t+1).

## §2. Universe + audit configurations

**Universe.** Same 31-instrument yfinance ETF-proxy universe as B2b (`data/yf_b2b/*_DAY.parquet`):

- 8 equity-index ETFs/indices
- 8 FX spot pairs
- 6 bond-ETF proxies
- 4 physical-commodity ETFs
- 5 regional-equity ETFs

**Date range.** 2005-01-03 → 2026-05-16 (same as B2b: ~5574 bars, ~21 years on majors).

**WFO.** `CROSS_ASSET_MOMENTUM` defaults (2y IS / 0.5y OOS / 8 folds) → ~60 effective folds.

**Pre-registered cells (V3.1).** All cells use C1_canonical's underlying B2 EWMAC config `(speeds=(16/64, 32/128, 64/256), fdm=1.35)`. The new knob is `broad_trend_filter`.

| Cell                    | Filter speed     | Mode               | Deadband | Notes                                                  |
|-------------------------|------------------|--------------------|----------|--------------------------------------------------------|
| **C1_canonical**        | (64, 256)        | absolute_trend     | 0.0      | The L48-mitigation default — broad trend in either direction enables trading |
| C2_directional          | (64, 256)        | directional        | 0.0      | Sign-aligned: negative broad-trend flips per-asset signal sign         |
| C3_faster_filter        | (32, 128)        | absolute_trend     | 0.0      | Shorter broad-trend lookback                            |
| C4_slower_filter        | (128, 512)       | absolute_trend     | 0.0      | Longer broad-trend lookback                             |
| C5_deadband             | (64, 256)        | absolute_trend     | 5.0      | Require `|broad_ewmac|>5` (vol-normalised) before activating              |
| C6_baseline_b2b         | n/a              | n/a (filter off)   | n/a      | Reproduce B2b's -0.28 Sharpe                            |
| C7_gross_no_costs       | (64, 256)        | absolute_trend     | 0.0      | C1 with `apply_costs=False`                             |
| C8_filter_only          | (64, 256)        | absolute_trend     | 0.0      | Use ONLY the broad-trend signal (no per-asset EWMAC) — sanity check |

**Plateau pre-flight neighbours of C1 = (64, 256) absolute, deadband=0:**

| Neighbour            | What changes                              |
|----------------------|-------------------------------------------|
| P_shift_short_filter | (32, 128) absolute, deadband=0            |
| P_shift_long_filter  | (128, 512) absolute, deadband=0           |
| P_add_deadband       | (64, 256) absolute, deadband=2.5          |
| P_per_asset_filter   | per-asset trend filter (not broad-index)  |

**Falsification hypotheses (pre-committed V3.1):**

- **B2c H1 (plateau rescue).** "The trend-of-trend filter restores plateau passing (spread ≤ 30%)." Falsifiable: B2b's spread was 37.10% → filtered spread <= 30% would PASS this hypothesis.
- **B2c H2 (Sharpe rescue).** "Filter brings C1 Sharpe to materially positive territory (≥ +0.30 — broadly comparable to retail-quality trend systems)." Falsifiable: if C1 Sharpe < +0.30, the filter doesn't rescue.
- **B2c H3 (cost-of-filter check).** "The filter pays for itself — C1 (filter on) Sharpe ≥ C6 (filter off, B2b baseline)." Falsifiable: if filter Sharpe ≤ B2b's -0.28, it's worse than nothing.
- **B2c H4 (regime is universe-wide, not per-asset).** "Broad-index filter (C1) outperforms per-asset filter (P_per_asset_filter)." Falsifiable: if per-asset filter is better, the regime is per-instrument, not universe-wide.

## §3. Decision rule (pre-committed V3.1)

**Per-axis thresholds:** `CROSS_ASSET_MOMENTUM` defaults. Bootstrap-CI gate (CI_lo > 0) is mandatory.

**Cell selection:** among DEPLOY-eligible cells (5-axis matrix DEPLOY or CONDITIONAL_WATCHPOINT-with-noise=best, AND CI_lo > 0), excluding C6/C7/C8 (baselines + cost reference), pick highest CI_lo.

**Plateau pre-flight (L27):** C1 + 4 neighbours above; spread <= 30%. If fail, audit aborts and L48 is reinforced (broad-trend filter doesn't rescue cross-asset universal-trend EWMAC).

**Sanctuary:** last 12 months held out (we have enough data for the longer hold-out now).

**Methodological discipline:** No filter-parameter tuning after seeing results. If C1 fails plateau, we don't switch to a different filter speed and try again — that's curve-fitting. We accept the verdict and the next pre-reg (B2d?) addresses a fundamentally different mitigation (e.g., HMM-based regime per I1).

## §4. Result log (appended post-audit)

*To be filled.*

### §4.1 Baseline reproduction (C6)

*Should reproduce B2b's C1 = -0.28 within rounding.*

### §4.2 Plateau pre-flight

*To be filled.*

### §4.3 Per-cell verdicts (full 5-axis)

*To be filled if plateau passes.*

### §4.4 H1-H4 falsification verdicts

*To be filled.*

### §4.5 Recommendation + new lessons

*To be filled.*

---

## §5. Failure modes to watch

- **L04 / A1 (causality):** broad-index EWMA must use `.shift(1)` before earning return. Smoke test added.
- **L24 / J3 (noise gate):** apply 5-axis matrix as usual.
- **L27 (plateau):** the primary gate. Filter speed neighbours stress the regime-dependence.
- **L40 (yfinance roll):** not applicable — we use ETF/spot proxies, no roll.
- **L46 (sample-size CI):** we have 60 folds now; if CI_lo is still negative, the strategy is signal-bound, not sample-bound.
- **L48 (the thing we're testing):** the broad regime filter is exactly the mitigation L48 advocates.
- **A4 (WFO honesty):** filter parameters are pre-registered per cell; no per-fold tuning.
- **A6 (MC bootstrap):** shared-block on the multi-asset universe; the broad-index EWMA is re-computed within strategy_fn on each bootstrap path so the filter learns the new regime structure correctly.
- **Specific risk: filter introduces auto-correlation in returns.** The gate switches on/off in long stretches; this can inflate Sharpe artificially if returns are positively serially correlated. The bootstrap CI mitigates this but watch for unusually-narrow CI in successful cells.

## §6. Implementation plan

1. **Extend `EwmacConfig`** in `research/ewmac/ewmac_strategy.py` with optional `broad_trend_filter: BroadTrendFilterConfig | None = None`. The new dataclass has `fast_hl`, `slow_hl`, `mode` ('absolute_trend' or 'directional'), `deadband`.
2. **Add `_compute_broad_trend_gate()`** helper: builds the equal-weight cum-log-return universe index, computes its EWMAC, returns a Series of gate values [-1, 0, +1].
3. **Modify `compute_ewmac_forecast`** to multiply per-asset forecasts by the gate when filter config is set.
4. **Add unit tests** to `tests/test_ewmac.py`:
   - Filter disabled by default (B2 parity).
   - Gate is +1 when broad universe is uptrending.
   - Gate is 0 when within deadband.
   - Causality: filter at t uses data <=t-1.
5. **Write `research/ewmac/run_b2c_audit.py`** mirroring `run_b2b_audit.py` structure (same yf_b2b data, new cells with filter config).
6. **Run audit, document.** If C1 promotes → port to `titan/strategies/ewmac/` with the filter wired through `config.py`. If C1 still plateau-fails → close out, escalate to I1 HMM-based regime detection.
