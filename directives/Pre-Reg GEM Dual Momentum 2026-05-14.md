# Pre-Registration — GEM Dual Momentum

**Author:** rayanazhari (planner) + Claude orchestrator
**Date committed:** 2026-05-14
**Branch:** v2-main
**Status:** §1-§3 frozen at commit; §4 result log appended after audit

> This is a V3.1 pre-registration. §1-§3 stay frozen for the lifetime of the audit. No retroactive cell-favouring (V3.2). Gates can only be RELAXED in a separate PR explaining why the original was unimplementable.

---

## §1. Hypothesis & mechanism

**Hypothesis.** Antonacci's Global Equities Momentum (GEM): combining absolute momentum (time-series, vs cash) with relative momentum (cross-sectional, US vs international equity) outperforms a static 60/40 portfolio on Sharpe and MaxDD, with the absolute-momentum gate providing a defensive switch into bonds during bear markets.

**Mechanism.**
1. Monthly rebalance on last trading day of each month.
2. Compute 12-month total return for each of: SPY (US large-cap), EFA (developed international), IEF (intermediate Treasuries, used as the risk-free proxy in absence of BIL parquet).
3. Decision rule:
   - If `R_SPY(12m) > R_IEF(12m)` AND `R_EFA(12m) > R_IEF(12m)`: long max(SPY, EFA) by 12m return.
   - Else if `R_SPY(12m) > R_IEF(12m)` only: long SPY.
   - Else if `R_EFA(12m) > R_IEF(12m)` only: long EFA.
   - Else: long IEF (defensive switch).
4. **Buffered transitions.** Once a winner is chosen, require the next-month challenger to beat the incumbent's 12m return by ≥ 0.5% (absolute) to trigger a switch. Reduces churn during near-ties.
5. **Causality.** Decision uses returns through close of bar t-1 (the last day of the prior month). New allocation effective at open of bar t+1 — i.e. position enters at the next month's first close (we hold from close of t to close of t+1). One-bar shift discipline strict.

**Why this is novel for our stack.** No top-down macro asset allocator exists in `titan/strategies/`. `etf_trend` is per-instrument deceleration; `bond_gold` is asset-pair momentum. GEM is cross-section + absolute-momentum gate in one model. Well-documented edge (Antonacci 2014 and subsequent replications). Implementable end-of-day with three parquets we already have.

## §2. Universe + cells + data

**Universe (fixed for this audit):**
- SPY (US large-cap ETF)
- EFA (MSCI EAFE)
- IEF (7-10y Treasury — risk-free proxy; T-bills (BIL) substituted because BIL parquet not downloaded)

**Data sources:** parquets in `data/SPY_D.parquet`, `data/EFA_D.parquet`, `data/IEF_D.parquet`. Yfinance-sourced, adjusted close (total return).

**Date range:** 2003-01-02 → present (limited by EFA inception). Sanctuary slice: trailing 12 months from latest data.

**Bar timeframe:** Daily close. Bar-per-year convention: 252 (`BARS_PER_YEAR["D"]`).

**Strategy class:** `CROSS_ASSET_MOMENTUM`.

**Cells.** Pre-committed grid:

| Cell | lookback_months | buffer_pct | exit_signal |
|---|---|---|---|
| C1 (canonical) | 12 | 0.5% | absolute_mom |
| C2 (no-buffer) | 12 | 0.0% | absolute_mom |
| C3 (short-lookback) | 6 | 0.5% | absolute_mom |
| C4 (long-lookback) | 18 | 0.5% | absolute_mom |
| C5 (no-defensive) | 12 | 0.5% | none (always long max(SPY,EFA)) |

5 cells total. **DSR adjustment applied since N=5 borderline**; we'll apply it anyway for discipline (A5).

## §3. Decision rule (pre-committed, V3.1)

**Class defaults (from `defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)`):**
- Sharpe convention: per-day MTM
- WFO mode: rolling
- MC gate: P(MaxDD > 35%) < 10%

**Per-axis thresholds:**
- CI_lo > 0 (95% bootstrap on stitched OOS Sharpe, 1000 resamples)
- DSR-prob ≥ 0.95 (after deflation at N=5 trials with actual returns' skew/kurt)
- MC P(MaxDD > 35%) ≤ 10% (block bootstrap on underlyings, ~21-day blocks, 1000 paths)
- Sanctuary Sharpe > 0

**Verdict map (4-axis decision matrix, total function, no UNDETERMINED — L09):**
- 4 axes at "best" → `DEPLOY`
- 3 → `CONDITIONAL_WATCHPOINT`
- 2 → `TIER_UNCONFIRMED`
- 1 → `SUSPECT`
- 0 → `RETIRE`

**Cell selection.** Plateau rule (V3.2): the cell whose ±1-step grid neighbour also passes the same gates, AND whose headline Sharpe varies by <30% across the neighbourhood, wins. C1 is the pre-committed canonical cell; the others test parameter robustness around it. Tie-break by parsimony (fewest moving parts).

**Causality test (A10 / L04):** required pre-commit assertion — corrupt future returns of all three instruments at t = N/2, assert decisions at t < N/2 are bit-exact unchanged.

## §4. Result log (appended post-audit)

*Empty at commit. Will be populated after running the audit harness.*

---

## §5. Failure modes to watch (V3.6 lessons applied to this strategy)

- **L04 / A1 — Same-bar look-ahead.** Decision rule must use 12m returns through `close[t-1]`, NOT `close[t]`. Verify in code.
- **L06 — Sparse-trade Sharpe.** GEM rebalances ~12x/year. With daily bars, this is a sparse strategy in trade-count terms but DAILY in equity-curve terms. Use per-day MTM Sharpe (class default for `CROSS_ASSET_MOMENTUM`), NOT per-trade.
- **L08 — MC gate calibration.** `CROSS_ASSET_MOMENTUM` defaults to P(MaxDD>35%) < 10%, recalibrated from the broken uniform 25%/5%. Don't override.
- **L11 — Data overwrites.** SPY_D / EFA_D / IEF_D parquets must NOT be re-downloaded mid-audit. Snapshot before run.
- **A3 — TR vs price-only.** Yfinance "adj close" is total-return (including dividends). The audit's 12m return = total return, not price-only. Document explicitly in the code.
- **A4 — WFO honesty.** Pre-registered parameters (this directive committed BEFORE any data examined). Rolling WFO folds; per-fold parameter selection inside the framework primitive.
- **V3.4 — MC re-run before removing "ballast".** Does GEM beat 60/40 BUT lose vs SPY-only? Run that decomposition. Don't simplify ("just go all SPY!") without testing the underlying-resampled MC of that variant.

## §6. Implementation plan

1. Build the strategy function in `research/gem/gem_strategy.py` — pure function, returns per-bar position weights given a DataFrame of close prices.
2. Build the audit harness in `research/gem/run_gem_audit.py` — calls framework primitives, prints + writes result log.
3. Add unit tests in `tests/test_gem.py`:
   - Causality assertion (corrupt-the-future test)
   - Buffer logic (no-switch when delta < 0.5%)
   - Defensive switch (when all instruments < cash, allocates to IEF)
   - Class-default consistency (`defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)` returns the row this audit relies on)
4. Run the audit, append §4 result log.
5. If verdict is DEPLOY or CONDITIONAL_WATCHPOINT, port to `titan/strategies/gem/`.
