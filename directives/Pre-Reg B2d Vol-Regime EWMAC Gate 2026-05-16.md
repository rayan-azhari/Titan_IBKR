# Pre-Registration — B2d: Realised-Vol Regime Filter on Carver EWMAC

**Author:** rayanazhari (planner) + Claude orchestrator (Researcher)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** Second mitigation attempt after B2c trend-of-trend filter (L49 reinforced L48). Tests whether a broad-realised-vol percentile gate succeeds where the broad-trend gate failed.
**Predecessors:**
- B2 IBKR-3y commodity (Sharpe +2.02 — regime-artifact per L48)
- B2b yfinance-21y cross-asset (Sharpe -0.28 — falsified universal-trend)
- B2c trend-of-trend filter (no rescue; L49 — regime is per-asset)

**Status:** §1–§3 frozen at commit BEFORE the audit runs. §4 result log appended post-run.

> V3.1 honesty: filter parameters pre-committed. No post-hoc tuning.

---

## §1. Motivation & mechanism

**L49's key finding:** B2c's broad-trend gate didn't help because (1) `absolute_trend` mode + zero deadband produces a near-constant gate (the broad-trend is non-zero most of the time), and (2) when it did differ (`directional` mode), the Sharpe-sign-flip diagnosed regime is per-asset, not universe-wide. The broad-index gating granularity is wrong.

**B2d hypothesis:** A realised-vol gate may avoid L49's degeneracy problem because:

1. Realised vol moves over a wider dynamic range than trend signal (which is bounded by Carver's [-20, +20]). Percentile-based gating gives more spread across cells.
2. Vol regimes (quiescent / moderate / crisis) are a different decomposition than trend regimes. The cross-asset correlation breakdown during crises is well-documented (Goyenko et al., Bali et al.).
3. Even if regime is per-asset for trend direction, vol regime is universe-wide enough (crises tend to be everywhere-correlated) that a broad-index vol gate has a defensible economic basis.

If B2d fails the same way B2c failed (degenerate plateau or no Sharpe rescue), L49 generalizes from trend-gates to ALL broad-index regime gates, strengthening the I1 HMM per-asset escalation case.

**Mechanism (per audit-window date t):**

1. Equal-weight broad-universe cum-log-return → broad daily-return series.
2. Rolling annualised realised vol over `vol_lookback_days` (default 60).
3. Rolling percentile of broad-vol over `percentile_window_days` (default 252).
4. Gate = 1 when `pct_lo ≤ current_pct ≤ pct_hi`, else 0.
5. Per-asset combined forecast multiplied by gate.

Causality: vol uses past returns; percentile is rolling backward. Same `.shift(1)` discipline as B2/B2b/B2c.

## §2. Universe + audit configurations

**Universe + window:** identical to B2b/B2c (31-instrument yf_b2b, 2005-01-03 → 2026-05-16, ~5574 bars, 60 WFO folds at class defaults).

**Pre-registered cells (V3.1):**

| Cell                  | vol_lookback | pct_window | pct_lo | pct_hi | Notes                                  |
|-----------------------|-------------:|-----------:|-------:|-------:|----------------------------------------|
| **C1_canonical**      | 60           | 252        | 20     | 80     | Moderate-vol band; default              |
| C2_tighter            | 60           | 252        | 25     | 75     | Tighter middle band                     |
| C3_wider              | 60           | 252        | 10     | 90     | Wider band — closer to "always on"      |
| C4_low_vol_only       | 60           | 252        | 0      | 50     | Low-vol regime only (avoid crises)      |
| C5_high_vol_only      | 60           | 252        | 50     | 100    | High-vol regime only (counter-intuitive) |
| C6_baseline_b2b       | n/a          | n/a        | n/a    | n/a    | Filter off (B2b reproduction)           |
| C7_long_window        | 60           | 504        | 20     | 80     | 2y percentile window                    |
| C8_gross_no_costs     | 60           | 252        | 20     | 80     | C1 with `apply_costs=False`             |

**Plateau pre-flight neighbours of C1 = (vol_lb=60, pct_win=252, [20, 80]):**

| Neighbour              | What changes                  |
|------------------------|-------------------------------|
| P_pct_15_85            | [15, 85]                      |
| P_pct_25_75            | [25, 75]                      |
| P_vol_lb_30            | vol_lookback=30 (faster vol)  |
| P_pct_window_504       | percentile_window=504 (longer)|

**Falsification hypotheses (V3.1):**

- **B2d H1 (plateau):** spread ≤ 30%. If broad-vol gate has reasonable distribution variation across cells, plateau should be passable. If it degenerates like B2c → L49 generalizes to vol-gates.
- **B2d H2 (Sharpe ≥ +0.30):** materially-positive Sharpe.
- **B2d H3 (filter helps):** C1 Sharpe > C6 baseline (-0.28).
- **B2d H4 (vs B2c):** vol-gate (C1 here) > trend-gate (B2c C1 = -0.28). If yes, vol regime is a more useful regime decomposition than trend regime for EWMAC.

## §3. Decision rule (V3.1)

Per `CROSS_ASSET_MOMENTUM` defaults. Bootstrap-CI gate (CI_lo > 0) mandatory. Cell selection: DEPLOY-eligible (excluding C6/C8/baselines), highest CI_lo.

Plateau pre-flight L27 ≤ 30%. Sanctuary 12 months. No filter-param tuning post-hoc.

## §4. Result log (appended post-audit)

*To be filled.*

---

## §5. Failure modes

- L04/A1 causality: smoke test covers.
- L27 plateau: primary gate.
- L46 sample-size: 60 folds; if CI_lo still negative, signal-bound.
- L48 (regime-artifact): the thing we're testing a SECOND mitigation against.
- L49 (broad-index granularity wrong): the thing this audit confirms or refutes for vol-gates.
- A6 MC bootstrap: shared-block on multi-asset universe; vol-gate re-computed within strategy_fn on each path.

## §6. Implementation plan

1. **Extend `EwmacConfig`** with optional `vol_regime_filter: VolRegimeFilterConfig`. (DONE)
2. **Add `_compute_vol_regime_gate()`** helper. (DONE)
3. **Wire into `compute_ewmac_forecast`** as AND-combined with broad_trend_filter when both set. (DONE)
4. **Unit tests** in `tests/test_ewmac.py` for filter disabled by default, gate fires sometimes, crisis vol excluded, causality. (DONE)
5. **Write `research/ewmac/run_b2d_audit.py`** mirroring run_b2c structure.
6. **Run + document.** If C1 promotes → port to live. If still fails → escalate to I1 HMM (the per-asset escalation pre-registered by L49).
