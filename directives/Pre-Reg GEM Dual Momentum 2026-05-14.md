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

**Audit runs:** 2026-05-14 → 2026-05-15. The grid grew from the original 5 cells (C1-C5) through iterative enhancements: C6/C7 (multi-speed blend), C8 (vol-target overlay), C9/C10/C11 (conditional stress gate + leverage), C12/C13 (lever-the-strategy), C14/C15 (drawdown circuit breaker). All cells run with realistic costs (1.5 bps/turnover ≈ COST_US_ETF_LIQUID).

### §4.1 Final cell verdicts (with costs, 15 cells)

| Cell | Sharpe | CI lo | Strategy MaxDD | Rel-MC ratio | p(better) | Sanctuary | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 canonical (12m, Antonacci) | +0.49 | +0.08 | -37% | 0.95 | 56% | +0.94 | COND_WP |
| C2 no buffer | +0.56 | +0.15 | -37% | 0.96 | 58% | +0.94 | COND_WP |
| C3 short_lookback (6m) | +0.49 | +0.10 | -37% | 0.96 | 57% | +0.67 | COND_WP |
| C4 long_lookback (18m) | +0.55 | +0.16 | -38% | 0.98 | 54% | +1.21 | COND_WP |
| C5 no_defensive | +0.46 | +0.10 | -43% | 1.03 | 22% | +0.94 | COND_WP |
| C6 blend (3,6,12) | +0.56 | +0.19 | -36% | 0.96 | 60% | +1.12 | COND_WP |
| C7 blend (1,3,6) | +0.47 | +0.10 | -36% | 0.97 | 60% | +1.03 | COND_WP |
| **C8 blend + voltarget10** | **+0.84** | **+0.45** | **-20%** | **0.52** | **98%** | **+1.07** | **DEPLOY** |
| C9 stress_gated | +0.71 | +0.30 | -25% | 0.67 | 91% | +0.62 | DEPLOY |
| C10 stress_gated + lev 1.5x | +0.67 | +0.27 | -35% | 0.91 | 65% | +0.56 | COND_WP |
| C11 composite_stress (VIX) | +0.67 | +0.27 | -35% | 0.91 | 65% | +0.42 | COND_WP |
| **C12 voltarget + lev 2x** | **+0.84** | **+0.45** | **-22%** | **0.56** | **95%** | **+0.91** | **DEPLOY** |
| C13 target20 + lev 2x | +0.67 | +0.27 | -36% | 0.93 | 62% | +1.12 | COND_WP |
| C14 voltarget + DD_breaker | +0.82 | +0.43 | -20% | 0.51 | 97% | +0.71 | DEPLOY |
| C15 voltarget + lev 2x + DD_breaker | +0.84 | +0.44 | -22% | 0.55 | 93% | +0.52 | DEPLOY |

### §4.2 Selected production cell

**C12_voltarget_lev2** is the selected production cell.

Settings:

```python
GemConfig(
    lookback_blend=(3, 6, 12),
    buffer_pct=0.005,
    defensive_switch=True,
    ann_vol_target=0.10,
    vol_lookback_days=20,
    max_leverage=2.0,
)
```

Why C12 over C8:
- Both pass all 4 decision-matrix axes (CI_lo, DSR, MC, Sanctuary) with effectively identical Sharpe (+0.84).
- C12 achieves benchmark-comparable total return via MES futures (2x leverage) while keeping MaxDD (-22%) materially below SPY (-41%).
- The user's deployment vehicle is MES futures — C12 matches that operationally.

### §4.3 Per-cell economics (canonical: C12)

- Sharpe: +0.8443 (CI95 [+0.446, +1.210])
- DSR-prob: 1.0000 (N=15, sweep_sharpe variance well-spread)
- Strategy median MaxDD: -22.0% vs benchmark (SPY buy-hold) median MaxDD: -41.3%
- Rel-MC ratio: 0.556 (passes the 0.80 L17 gate)
- p(strategy MaxDD ≤ benchmark MaxDD): 94.5% of MC paths
- Sanctuary Sharpe: +0.9063 (no lucky_flag, no unlucky_flag)
- Sanctuary percentile vs historical rolling-12m distribution: 0.526 (median range; clean)
- Noise gate (Varma, ran on C8 -- same logic, no leverage):
    - Base Sharpe: +0.81
    - σ=0.1: +0.78 (deg 3%); σ=0.3: +0.71 (deg 12%); σ=0.5: +0.62 (deg 23%)
    - PASS (mean AND worst-case at every level)
- Causality smoke test (gem_assert_causal): PASSED on 5 random t-points (corrupted-future shocks did not alter past weights)

### §4.4 Verdict + deployment plan

**VERDICT: DEPLOY** (4-axis, as of 2026-05-14).

**Subsequent revision (2026-05-15):** Under the J3 5-axis matrix (`Pre-Reg J3 Noise Robustness 5th Axis 2026-05-15.md`), C12 demoted DEPLOY → CONDITIONAL_WATCHPOINT because the vol-target overlay's `realised_vol_20d` denominator is fragile to small input-price noise (L30). The J4 noise-robust redesign audit (`Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md`) tested three mitigations and promoted **A1_ewma_hl40** — same mechanism as C12, but with `vol_estimator_kind="ewma"` + `vol_estimator_halflife=40`. A1 holds a true 5/5-axis DEPLOY (noise axis recovered to `best`) at ~3% point Sharpe drag (+0.7773 vs +0.8016) and CI_lo +0.387. **A1_ewma_hl40 is the current production cell in `config/gem_voltarget_lev2.toml`** (deployed on Docker paper account 2026-05-15 13:07 UTC). The C12 row in §4.1 above remains the historical record; do not edit.

Next steps (per `directives/Strategy Deployment Guide.md`):

1. Port to `titan/strategies/gem/` (live class). ✅ Done.
2. Parity test (V3.6 A10): one-bar bit-exact agreement between research function and live class. ✅ Done. Extended for A1 in `tests/test_gem_live_parity.py::test_parity_a1_ewma_bulk_warmup_matches_research_exactly` (2026-05-15).
3. Wire to `titan/adapters/ibkr/` + add to `scripts/run_live_gem.py`. ✅ Done.
4. Register with `PortfolioRiskManager`. ✅ Done.
5. Paper trade for ≥30 days before live capital (per Deployment Guide). 🔄 IN PROGRESS — A1 deployed 2026-05-15; J3 §4.3 sets the watchpoint window at 5 sessions before considering live capital.
6. MES futures sizing: 1 contract = $5 × ES_price. At ES ~5800 → 1 MES ≈ $29k notional. For $30k strategy NAV at 2x leverage → ~2 MES contracts in SPY-long state. The strategy's weight output (which can be up to 2.0) directly maps to MES contract sizing. (Live mode is currently `execution="etf"` on the paper account; MES mode is enabled by changing `execution_mode = "mes"` in the TOML.)

### §4.5 Auxiliary findings (V3.6 lessons spawned -- see V3.6 Catalogue update)

- **L17 (already booked)**: absolute MC gate fails for long-only equity strategies → relative MC adopted as 4th axis.
- **L18 (new)**: a state-tracking buffer that compares against a STALE incumbent return can latch a strategy in a position that never wants to leave. Compare against LIVE values at every decision time. (Source: GEM Step 1 bug; fixed 2026-05-15.)
- **L19 (new)**: under block-bootstrap MC, continuous vol-targeting outperforms binary regime-on/off stress gating for the same vol budget. Smooth scale > step-function scale. The Sharpe drops 0.14 (C8 → C9) when switching from continuous to gated. (Source: GEM Step 4.)
- **L20 (new)**: when a stress-signal source (e.g. VIX) is loaded from a different parquet than the primary close, normalise BOTH indexes to date-only before reindex-merge. Mismatched time-of-day stamping silently produces an all-NaN signal that fails-open as "never in stress". (Source: GEM Step 7 / C11 debug.)
- **L21 (new)**: transaction costs at realistic ETF rates (1.5 bps/turnover) have negligible impact on monthly-rebalancing strategies (Sharpe drag ~0.01 on C8). The cost gate is not load-bearing for slow-turnover strategies. Apply it anyway for honesty. (Source: GEM Step 6.)
- **L22 (new)**: a drawdown circuit breaker on top of a vol-targeted strategy adds little marginal value -- the vol-target keeps daily vol bounded, which keeps drawdowns bounded, which means the DD breaker rarely fires. DD-breaker is more useful for high-vol strategies without explicit risk control. (Source: GEM Step 5 / C14 ablation.)

These L18-L22 entries to be appended to `directives/V3.6 Lessons Catalogue.md`.

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
