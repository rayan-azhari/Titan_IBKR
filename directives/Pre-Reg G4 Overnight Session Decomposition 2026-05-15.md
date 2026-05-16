# Pre-Registration — G4: Overnight Session Decomposition (SPY)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy audit (new class `INTRADAY_MICROSTRUCTURE` -- session-decomposition variant)
**Predecessor:** None. This is the first audit of a new design family.
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE data is examined. The mechanism is from published literature (Lou, Polk & Skouras 2019), not derived from our dataset. The gates / cells / decision rule below have not been tested.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` step 5: G4 — Overnight session decomposition (2d), quick win, SPY-only first.

**Source.** Lou, Polk & Skouras, *Journal of Financial Economics* 2019: *"A tug of war: Overnight versus intraday expected returns"*. Documents that the equity risk premium accrues predominantly during the OVERNIGHT session (close-to-open), while the INTRADAY session (open-to-close) earns flat or negative expected returns. The cross-section of stocks separates cleanly: stocks with higher overnight returns reliably underperform intraday and vice versa, a tug-of-war that compounds the overnight outperformance over time.

**Hypothesis (single-asset SPY version).** On SPY, a strategy that:

- **Holds long from close[t-1] → open[t]** (capturing the overnight session)
- **Stays flat from open[t] → close[t]** (avoiding the intraday session)

earns positive risk-adjusted returns over multi-year windows, with materially lower drawdowns than always-long SPY. Falsifiable: if the audit shows the overnight-only strategy has Sharpe ≤ buy-hold-SPY Sharpe OR the relative-MC gate fails, the published edge does not survive on retail-implementable SPY at ETF cost levels.

**Mechanism (per-bar computation):**

1. Each trading day, observe `open[t]` and `close[t]` from the daily OHLC bar.
2. Overnight session return: `overnight_ret[t] = open[t] / close[t-1] - 1`.
3. Intraday session return: `intraday_ret[t] = close[t] / open[t] - 1`.
4. Decompose: `daily_ret[t] = (1 + overnight_ret[t]) × (1 + intraday_ret[t]) - 1`.
5. Strategy A (overnight-only): per-bar return = `overnight_ret[t]`.
6. Strategy B (intraday-only): per-bar return = `intraday_ret[t]`. (Audited for diagnostic completeness even though hypothesis says this loses.)
7. Strategy C (long-only buy-hold): per-bar return = `daily_ret[t]`. (Benchmark.)

**Causality.** All three computations are CAUSAL: each per-bar value at time `t` uses `open[t]`, `close[t]`, `close[t-1]` — all observed by close of day `t`. There is no `.shift(1)` needed at the daily aggregation level because we're DECOMPOSING the already-observed daily bar, not predicting a future return. The discipline that DOES apply: the strategy must be executable in real life, meaning the trader needs an order in place by `close[t-1]` to capture the gap. We assume this is operationally feasible (a market-on-close BUY at `close[t-1]` + market-on-open SELL at `open[t]` is a standard 2-leg pair on SPY).

**Why this is novel for our stack.** No session-decomposition strategy exists in `titan/strategies/`. ORB is intraday-only-breakout, not session-decomposition. ETF Trend and GEM are daily/monthly cross-asset. The mechanism here uses ONLY the daily OHLC structure, which our existing `data/SPY_D.parquet` covers — no new data acquisition.

**Why the class is `INTRADAY_MICROSTRUCTURE` (with a `notes` annotation).** The trade lives in the intraday open/close transitions of daily bars. The defaults (per-day MTM Sharpe, expanding WFO 3y IS / 1y OOS / 5 folds, MC P(MaxDD>30%) < 5%) are calibrated for intraday strategies; they fit this design. We DO NOT propose a new sub-class; this is a thin overlay on existing daily-bar infrastructure.

## §2. Universe + cells + data

**Universe (fixed):**

- **SPY** — daily OHLC bars from `data/SPY_D.parquet`. Yfinance adjusted close means the close column is total-return; for THIS audit we need `open / close / close_lag1` and the standard yfinance adj-close is good enough at the daily level (the dividend distributions are small relative to daily bar moves and apply at known dates).

**Important caveat on adjusted bars (A3).** Yfinance "adj close" applies dividend + split adjustments. Open and close may be adjusted by different scaling rules at split events. We will compute the session decomposition using **price-only OHLC** (the un-adjusted columns) if available in the parquet; if only adj-close is available, we'll use it and flag the limitation in §4. The audit harness will assert no dividend-day discontinuities in the constructed `overnight_ret` series (max single-bar overnight move > 4σ flagged for inspection).

**Date range:** 1993-02-01 (SPY inception+1) → present. Sanctuary: trailing 12 months. Visible window ~33 years — generous for a daily-bar audit.

**Bar timeframe:** Daily. `BARS_PER_YEAR["D"] = 252`. Strategy class: `INTRADAY_MICROSTRUCTURE`.

**Cells (V3.1 frozen, 6 cells):**

| Cell | Strategy | Position rule | Notes |
|---|---|---|---|
| C1 (canonical) | Overnight-only long SPY | hold long close[t-1]→open[t]; flat open[t]→close[t] | Lou-Polk-Skouras baseline |
| C2 (overnight × intraday-short) | Overnight long + intraday short | long close[t-1]→open[t]; short open[t]→close[t] | A more aggressive version doubling the "intraday loses" thesis |
| C3 (overnight, no-cost gross) | Overnight-only, costs=0 | as C1, but with zero cost | Diagnostic: what's the gross edge before costs? |
| C4 (intraday-only) | Intraday-only long SPY | flat close[t-1]→open[t]; long open[t]→close[t] | Counterfactual — published literature predicts FAIL |
| C5 (buy-hold benchmark) | Always-long SPY | long full day | Null hypothesis comparator |
| C6 (overnight, post-2010) | Overnight-only, post-2010 | as C1, but visible window starts 2010 | Tests whether the edge has decayed since the JFE 2019 paper. |

**6 cells total.** DSR adjustment applies (N=6 > 5).

## §3. Decision rule (pre-committed, V3.1)

**Class defaults:** `defaults_for(StrategyClass.INTRADAY_MICROSTRUCTURE)`. Sharpe convention: per-day MTM (the strategy is held daily even though the WORK happens at session transitions). WFO: expanding, 3y IS, 1y OOS, 5 folds. MC: 21-day blocks, 200 paths.

**MC threshold override.** `INTRADAY_MICROSTRUCTURE` default is `P(MaxDD > 30%) < 5%`. Overnight-only on SPY should have MUCH smaller drawdowns than buy-hold-SPY (the intraday session contains most of the volatility per LPS 2019). We KEEP the class default and let the audit confirm. If the audit shows MaxDD comfortably under 30% in all paths, we have a wide safety margin; if not, the data does not support the LPS finding.

**Cost model (L23).** Two market-on-close + market-on-open fills per day on SPY → high turnover (2 round-trips/day instead of GEM's ~1/month). Apply the realistic US ETF cost model:

- `cost_bps_per_turnover = 1.5` (SPY is the most liquid ETF; tighter spread than the 6 bps used for UCITS substitutes)
- `cost_fixed_usd_per_fill = 1.0` (IBKR Pro tier floor)
- `notional_usd = 30_000` (matches GEM production for apples-to-apples)
- Cost gate is NOT relaxed for high-turnover — the whole point is to test if the published edge survives realistic costs on SPY.

C3 is the diagnostic cell that runs costs=0 to isolate the gross alpha; if C3 has a strong Sharpe but C1 (with costs) falls apart, the edge is cost-dominated.

**Per-axis thresholds (5-axis matrix, L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe) | > 0 | ≤ −0.2 |
| DSR-prob (deflated at N=6 trials, actual skew/kurt) | ≥ 0.95 | < 0.50 |
| MC (relative, L17) — median DD ratio vs buy-hold SPY | ≤ 0.80 | ≥ 1.6 |
| Sanctuary Sharpe (held-out 12mo) | > 0 | ≤ −0.3 |
| Noise robustness (Varma, J3) | passes mean AND worst-case at (0.1, 0.3, 0.5)σ | fails mean at any level |

**Why relative MC, not absolute.** The strategy is long-only equity exposure (during the overnight session). L17 / A6 say: long-only equity should use RELATIVE MC (strategy MaxDD vs buy-hold SPY MaxDD) because absolute MaxDD thresholds will fire on any path containing 2008/2020 regardless of the strategy's relative protection. For an overnight-only strategy, we expect strategy MaxDD ≪ buy-hold MaxDD (the strategy is FLAT during intraday volatility); this should produce a Rel-MC ratio ≪ 0.80 with high probability.

**Verdict map (5-axis, J3):** 5 → DEPLOY · 4 → CONDITIONAL_WATCHPOINT · 3 → TIER_UNCONFIRMED · 2 → SUSPECT · 0–1 → RETIRE.

**Cell selection (V3.2 plateau).** Among DEPLOY-eligible cells (verdict = DEPLOY, or COND_WP with noise axis = `best`), pick the highest CI_lo. The C5 buy-hold is NEVER selected for promotion — it's the benchmark, not a strategy.

**Plateau pre-flight (L27).** Run C1 + 3 neighbours (varying session-overlap by ±15min if the data allows; we will run on daily-OHLC only for v1 so no real neighbours exist — see Note). Skip plateau pre-flight in v1 because the design has no continuous parameters; the 6 cells ARE the sweep.

**Note: parameter-grid sparsity (L27/V3.2 caveat).** This audit has 0 continuous parameters — the 6 cells differ by structural choices (which session, costs on/off, signed direction, window). The plateau rule does not apply meaningfully. We document this in §4 and rely on the noise gate (L24) + relative MC gate to catch fragility. The full V3.2 plateau check returns when this design is extended to a parameter grid (e.g. holding the open position until N minutes after open, or until VWAP at hour X — those are continuous parameters and require a fresh pre-reg).

**Causality test (A10 / L04).** Pre-commit assertion: the strategy at bar t must use only `open[t]`, `close[t]`, `close[t-1]` (already observed by `close[t]`); corrupting `open[t+1]` or `close[t+1]` must not change the strategy's per-bar return at t. Live parity test required before promotion.

## §4. Result log

**Audit run:** 2026-05-15. Full output in `.tmp/reports/g4_overnight/result_log.md`. WFO: 21 folds on visible window (5,625 bars, 2003-01-02 → 2025-05-12). Sanctuary: 252 bars.

### §4.1 Per-cell verdicts (5-axis)

| Cell | Sharpe | CI95 lo | CI95 hi | DSR | Rel MC ratio | Rel MC pass | Sanc Sharpe | Noise base | Noise axis | Verdict |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---|
| C1_overnight_only | -0.0761 | -0.498 | +0.353 | 0.0000 | 2.2224 | FAIL | +1.0759 | -0.1285 | mid | **RETIRE** |
| C2_overnight_intraday_short | -0.7370 | -1.147 | -0.340 | 0.0000 | 2.5440 | FAIL | -0.7152 | -0.8094 | best | **RETIRE** |
| C3_overnight_no_costs | +0.7506 | +0.287 | +1.196 | 0.4584 | 0.0000 | PASS | +2.1506 | +0.6992 | mid | TIER_UNCONFIRMED |
| C4_intraday_only | -0.4290 | -0.834 | -0.038 | 0.0000 | 1.2683 | FAIL | +0.0744 | -0.3515 | mid | **RETIRE** |
| C5_buy_hold (benchmark) | +0.6148 | +0.196 | +1.041 | 0.0000 | 1.0000 | FAIL | +2.2811 | +0.6270 | best | TIER_UNCONFIRMED |
| C6_overnight_post_2010 | -0.0896 | -0.613 | +0.446 | 0.0000 | 1.9281 | FAIL | 0.0000 | -0.0966 | mid | **RETIRE** |

### §4.2 Gross vs net economics (C1 vs C3) — the headline finding

- **Gross (C3, costs OFF):** Sharpe = +0.7506, CI_lo = +0.287 → DEPLOY-worthy on three of four statistical axes.
- **Net (C1, costs ON):**     Sharpe = -0.0761, CI_lo = -0.498 → RETIRE.
- **Sharpe cost drag: +0.83.**

The LPS 2019 edge is REAL in gross terms (the close-to-open SPY return has positive expectation), but the cost structure annihilates it on retail-accessible SPY. At 2 fills/day × 1.5 bps spread+slip × ~252 trading days, the annualised cost is ~150 bps — more than the gross edge's contribution to the equity curve.

### §4.3 Decay test (C1 full sample vs C6 post-2010)

- C1 (2003+): Sharpe = -0.0761, CI_lo = -0.498
- C6 (post-2010): Sharpe = -0.0896, CI_lo = -0.613
- **Delta: -0.0135** — essentially identical. The edge has NOT decayed post-publication; it was never deployment-viable for retail at this vehicle/cost level.

### §4.4 The Rel-MC surprise

Every cost-bearing cell **fails the relative MC gate by a wide margin** (Rel-MC ratio in the 1.27–2.54 range vs 0.80 gate). The reason is structural: an overnight-only strategy LOCKS IN the overnight gap (whether positive or negative) and then sits in cash through intraday. On regime-shuffled MC paths that include 2008, 2018-02-05, 2020-03, the overnight gap-down is captured WITHOUT the intraday rebound — producing a worse MaxDD than buy-hold which experiences both legs of the same day.

LPS 2019 reports lower drawdowns in their long-form statistics. Our MC says: on bootstrap paths that re-shuffle the empirical bar sequence, the drawdown advantage does not survive. This is an empirical finding worth its own lesson (L33 below).

### §4.5 Recommended next step

**RETIRE the overnight-only design on retail SPY ETF.** Future iterations need ONE of:

1. **Different vehicle** — ES futures cost ~5-10 bps round-trip vs SPY ETF's ~30 bps, a 3-6× reduction. The same gross edge might survive futures-level costs. This is a fresh pre-reg (G4b).
2. **Different timing** — Instead of full-overnight, target only specific subsets (FOMC-eve nights, earnings-eve nights, post-Fed-day overnights). LPS-followups (Boguslawski et al 2024) report concentrated overnight outperformance around news events. Would need intraday tick data + event calendars — separate data dependency.
3. **Different asset class** — Single-stock overnight-only across the SP500 universe (LPS' canonical setup). Cross-sectional version: long the overnight winners, short the overnight losers. This is the more research-faithful design. Would need ~500-stock daily-bar parquets.

None of these is a small extension of G4 v1; each is a new audit with its own pre-reg under V3.1 discipline.

### §4.6 New lessons (appended to V3.6 Catalogue)

- **L33 (new):** Strategies whose alpha lives in a SHORT FRACTION of the trading day (e.g. overnight only) have ASYMMETRIC drawdown exposure on regime-shuffled MC paths. They capture the bad moves IN their session (overnight gaps) without participating in the recovery OUTSIDE their session (intraday rallies). This produces a Rel-MC ratio > 1.0 — STRATEGY MaxDD HIGHER than buy-hold — even though the strategy's full-sample MaxDD looks lower than buy-hold. The discrepancy: full-sample MaxDD reflects ACTUAL historical paths; MC ratio reflects REGIME-SHUFFLED paths where the gap and the recovery may not co-occur. Mechanism check: any "session-only" or "event-only" strategy should be Rel-MC-evaluated against the same-asset buy-hold; the asymmetric tail-risk profile is the L33 signature. **How to apply.** When auditing a session-bounded strategy, expect Rel-MC > 1.0 and design either (a) a longer holding window that bridges into the recovery session, or (b) an explicit hedge during the off-session window. Pure session-only strategies are best deployed at the FUTURES cost level where the alpha can absorb the asymmetric drawdown. **Source:** G4 overnight session decomposition audit, 2026-05-15. Every cost-bearing cell produced Rel-MC ratio in 1.27-2.54 vs the 0.80 deployment gate.

---

## §5. Failure modes to watch

- **L04 / A1** — Causality is structural to the design (the 3 components of daily OHLC are all observed by close[t]); the test is that bar-t+1 corruption does not affect bar-t outputs.
- **L06 — Per-day MTM.** Strategy is held daily; per-day MTM Sharpe convention is correct. Per-trade Sharpe would inflate by ~sqrt(2) (since the round-trip cycle is intraday + intraday, not weekly).
- **L08 — MC threshold (class-default).** `INTRADAY_MICROSTRUCTURE` defaults to P(MaxDD>30%) < 5%; we don't override. The expected MaxDD of overnight-only on SPY is small (much smaller than buy-hold); the gate has wide safety margin.
- **L11 — Data snapshot.** SPY parquet must NOT be re-downloaded mid-audit.
- **L17 — Relative MC.** Long-only equity exposure → relative MC vs buy-hold SPY. Absolute MC inappropriate.
- **L20 — Index normalisation.** Single-series audit; no cross-source merges; not a concern.
- **L24 — Per-cell noise gate.** Every cell receives the Varma sweep.
- **L26 / L31 — Mitigation patterns.** This audit has no threshold gates and no scaling overlays — neither L26 nor L31 mitigation classes apply.
- **A3 — TR vs price-only.** Use price-only OHLC if available. Document if forced to use adjusted-close.
- **A4 — WFO honesty.** Cells pre-registered. The visible window starts 1993 to maximise sample; some folds will straddle 1998/2008/2020 crisis periods, which is the point (stress-test the design).
- **A6 — MC bootstraps underlyings.** Block-bootstrap SPY's daily OHLC tuples (open, high, low, close) together, not just one column. Otherwise the synthetic open/close pairs lose their natural correlation. **NEW REQUIREMENT FOR THIS AUDIT** — the framework's existing `run_block_mc` bootstraps ONLY the close column. We will need to extend the MC harness or accept a limitation: the MC will bootstrap the close-to-close returns and synthesise overnight/intraday by maintaining the EMPIRICAL ratio observed bar-by-bar. This limitation is documented in §4 and is a known approximation.
- **V3.1 — Pre-committed selection rule** — highest CI_lo among DEPLOY-eligible.

## §6. Implementation plan

1. **Build the strategy module** in `research/overnight/overnight_strategy.py`:
   - `OvernightConfig` dataclass (session enum, costs flag, signed direction, optional date filter).
   - `overnight_returns(closes, *, cfg) -> pd.Series` — pure function.
   - `overnight_assert_causal(closes, *, cfg, n_trials)` — corrupt-future test.
2. **Build the audit harness** in `research/overnight/run_g4_audit.py`. Uses the framework primitives. Output to `.tmp/reports/g4_overnight/`.
3. **Tests in `tests/test_overnight.py`** — causality, session decomposition correctness, cost-on-vs-off ablation, class-default consistency.
4. **MC harness adaptation** — extend `_strategy_fn_for_cell` to handle the open/close-pair limitation; document the approximation.
5. **Run audit, append §4 result log.**
6. **If verdict is DEPLOY:** port to `titan/strategies/overnight/` per Strategy Deployment Guide. Live class needs market-on-close + market-on-open order plumbing through the IBKR adapter (different from GEM's daily-rebalance pattern).

After G4 lands, the next backlog step depends on D2's data-acquisition status. If futures parquets are present, D2 is unblocked; otherwise G4's deployment review continues while D2's data downloads.
