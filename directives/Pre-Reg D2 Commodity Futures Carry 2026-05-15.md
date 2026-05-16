# Pre-Registration — D2: Commodity Futures Carry

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy audit (class `CARRY`)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE the data is examined for this design. The 22-commodity universe and data shape (2017-06 → present from Databento) are KNOWN, but no cell-level Sharpe, no cross-sectional carry distribution, no per-commodity stats have been inspected. The cells / gates / decision rule below are theoretical, not data-tuned.

> **AMENDMENT 2026-05-15 PM (pre-audit).** During the audit-launch step, Databento GLBX.MDP3 continuous `.c.1` queries proved structurally too slow (~5 s/bar; ~4 h per commodity for the full window — pathological server-side behaviour on the second-month continuous symbology). **The pivot:** switch data source to Yahoo Finance (`yfinance` library), which provides reliable continuous M1 data via `{ROOT}=F` symbols. Yahoo does NOT provide a clean M2 continuous series. **Strategy adjustment:** use the BGR 2019 §3.2 documented proxy — "rolling yield" = 12-month trailing log return on the M1 series — which correlates ~0.65 with strict M1/M2 carry in BGR's data. This converts D2 from a strict-carry strategy into a cross-sectional time-series-momentum strategy with the same theoretical motivation. **Two consequences:** (a) the universe expands to the FULL 24 commodities since yfinance covers all 24 (vs Databento's 18 with M2 success); (b) the visible window expands to 2000-2026 (~25 years) vs the planned 7.5. The cells / gates / decision rule of §3 remain frozen; only §2's data-source line is amended. A new lesson (L34) will document the Databento `.c.1` slowness for future audits.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` step 4: D2 — Commodity futures carry (5d), strong peer-reviewed edge, blocked on 24-commodity data acquisition. Data acquired 2026-05-15 via Databento (22 of 24 commodities; see §2).

**Source.** Bakshi, Gao & Rossi, *"Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns"*, *Management Science* 2019. Builds on Erb-Harvey 2006, Gorton-Hayashi-Rouwenhorst 2013. Documents that the *carry* signal — the slope of the futures curve at the front end — predicts the cross-section of commodity returns: backwardated contracts (M1 > M2 in price) earn positive excess returns over time; contangoed contracts (M1 < M2) earn negative excess returns. The premium has been stable across decades and is robust to controls for momentum, value, and risk factors.

**Hypothesis.** A monthly-rebalanced cross-sectional long-short portfolio that goes LONG the top quintile of commodities by carry and SHORT the bottom quintile earns positive risk-adjusted returns over the 2017-06 → 2026-05 window. The portfolio is approximately market-neutral with respect to broad commodity beta (equal $-weight long vs short legs), so its alpha is the cross-sectional carry premium itself. Falsifiable: if the audit shows the long-short portfolio's CI_lo ≤ 0 or the Rel-MC fails OR Sharpe is dominated by either leg alone, the published edge does not survive in our 2017+ subsample at realistic CME-futures costs.

**Mechanism.**

1. **Carry signal (daily).** For each commodity i and bar t:
   - `raw_carry[i,t] = log(F_M1[i,t-1] / F_M2[i,t-1])`
   - Positive raw_carry → backwardated (near contract more expensive)
   - Negative raw_carry → contangoed (far contract more expensive)
   - Causality: signal at t uses prices observed by close of t-1. No look-ahead.
2. **Rebalance schedule.** End of each calendar month (last trading day). Between rebalances, weights are held constant.
3. **Portfolio rule (canonical).** At each rebalance date:
   - Compute `raw_carry[i]` for every commodity in the universe with valid M1 + M2 prices for the day.
   - Rank commodities by carry. Long the top 20% (quintile), short the bottom 20%.
   - Equal-weighted within each leg. Long leg notional = +0.5 × NAV; short leg notional = -0.5 × NAV. Net = 0 (market-neutral).
4. **Returns.** Per-bar strategy return = `Σ_i w[i,t-1] × log(F_M1[i,t] / F_M1[i,t-1])`. The strategy trades the front-month leg only; the M2 series is used only for the signal.
5. **Causality.** Weights at bar t are derived from carry computed on close of t-1; held throughout day t; new return earned `log(F_M1[t] / F_M1[t-1])` on each held leg. The `.shift(1)` discipline is enforced by computing carry on `F.shift(1)` before ranking.

**Why this is novel for our stack.** No `CARRY` class strategy exists in `titan/strategies/` (the existing `fx_carry` runs on AUD/JPY only). D2 introduces:

- The first multi-asset futures basket audited under the V2.0 framework.
- The first long-short market-neutral strategy under the 5-axis matrix.
- The first audit on Databento-sourced continuous-contract data (so far we've used yfinance ETFs).

## §2. Universe + cells + data

**Universe — 22 of the 24 BGR commodities.** Already-downloaded `data/{ROOT}_M1_D.parquet` and `data/{ROOT}_M2_D.parquet` for:

| Group | Roots (count) |
|---|---|
| Energy (4) | CL, NG, HO, RB |
| Metals — precious (4) | GC, SI, PL, PA |
| Metals — industrial (1) | HG |
| Grains + oilseeds (6) | ZC, ZW, ZS, ZL, ZM, ZO |
| Livestock (3) | LE, GF, HE |
| Softs (4 — M1 only, no M2) | KC, CC, CT, OJ |

**Not included:** BZ (Brent), SB (Sugar) — neither M1 nor M2 fully downloaded. The 4 ICE softs (KC, CC, CT, OJ) have M1 only, so they CANNOT contribute to the carry signal (which needs both M1 and M2). The audit's effective carry-eligible universe is therefore **18 commodities** (all CME GLBX). The 4 ICE M1-only contracts could in principle be used as buy-and-hold benchmarks, but for D2's cross-sectional carry portfolio they are excluded.

**Date range.** 2017-06-01 (Databento GLBX continuous symbology start) → 2026-05-12. ~8.5 calendar years = ~2,150 trading days. Sanctuary: trailing 12 months → ~7.5 years visible.

**Bar timeframe.** Daily close. `BARS_PER_YEAR["D"] = 252`. Strategy class: `CARRY`.

**Class-default override (L25 discipline).** Class default WFO is 5y IS / 1y OOS / 5 folds — calibrated for FX carry with 20+ year history. Our 7.5-year visible window does not support 5y IS without single-fold collapse. **OVERRIDE pre-committed here:**

- WFO mode: expanding (unchanged)
- IS minimum: 3 years (was 5)
- OOS: 1 year (unchanged)
- Fold count: auto-determined by framework (~4 folds expected on 7.5 years)
- All other CARRY defaults (per-day MTM Sharpe, MC P(MaxDD>30%) < 10%, 21-day MC blocks, 200 MC paths) unchanged.

Rationale: 3-year IS gives ~4 folds of OOS coverage (years 4, 5, 6, 7); 5-year IS gives ~3 folds (years 6, 7, 8) or possibly only 2 depending on boundary. The 3-year override is the smallest reduction that gets us into 4-fold territory without overfitting individual cells to a single fold.

**Cells (V3.1 frozen, 7 cells):**

| Cell | Rebalance | Portfolio | Lookback | Weighting | Costs | Notes |
|---|---|---|---|---|---|---|
| C1 (canonical) | monthly EOM | top/bottom quintile L/S | 1-day carry | equal-weight | ON | BGR canonical |
| C2 (weekly) | weekly Fri | top/bottom quintile L/S | 1-day carry | equal-weight | ON | tests rebalance frequency |
| C3 (tercile) | monthly EOM | top/bottom tercile L/S | 1-day carry | equal-weight | ON | broader legs (~6 each side) |
| C4 (vol-weighted) | monthly EOM | top/bottom quintile L/S | 1-day carry | inverse-vol | ON | within-leg vol parity |
| C5 (smoothed carry) | monthly EOM | top/bottom quintile L/S | 5-day avg carry | equal-weight | ON | reduces signal noise |
| C6 (long-only) | monthly EOM | top quintile long, flat short | 1-day carry | equal-weight | ON | tests if alpha is in long leg only |
| C7 (gross, no costs) | monthly EOM | top/bottom quintile L/S | 1-day carry | equal-weight | OFF | diagnostic: cost ablation |

**7 cells total.** DSR adjustment applies (N=7 > 5).

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds (5-axis matrix, L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe) | > 0 | ≤ −0.2 |
| DSR-prob (deflated at N=7 trials, actual skew/kurt) | ≥ 0.95 | < 0.50 |
| MC (absolute — class default) — P(MaxDD > 30%) | ≤ 0.10 | ≥ 0.20 |
| Sanctuary Sharpe | > 0 | ≤ −0.3 |
| Noise robustness (Varma, J3) | passes mean AND worst-case at (0.1, 0.3, 0.5)σ | fails mean at any level |

**MC axis is ABSOLUTE, not relative.** This is a long-short market-neutral strategy, NOT a long-only equity strategy. L17's relative-MC rule applies to long-only equity ONLY; here the benchmark would be... what? A long-only commodity index would have a fundamentally different exposure profile. Use the absolute MC threshold from CARRY class defaults (P(MaxDD>30%) < 10%).

**Cost model (L23).** CME futures liquid:

- `cost_bps_per_turnover = 1.0` (CME futures spread+slip, BGR-tier liquid contracts)
- `cost_fixed_usd_per_fill = 1.0` (IBKR Pro tier)
- `notional_usd = 30_000` per leg (per-commodity allocation in a 22-commodity book)
- Monthly rebalance with ~22 names → ~22 fills per rebalance × 12 rebalances/year ≈ 264 fills/year ≈ 88 bps/year in fixed costs + 24 bps/year in spread+slip (assuming ~10% per-commodity turnover per month) ≈ ~110 bps/year total drag.

C7 runs with `apply_costs=False` to isolate the gross alpha.

**Cell selection (V3.1 + V3.2).** Among DEPLOY-eligible cells (verdict = DEPLOY, OR CONDITIONAL_WATCHPOINT with noise axis = `best`), pick the one with the highest CI_lo. C7 (no costs) is NEVER selected — it's the diagnostic baseline.

**Plateau pre-flight (L27).** Run C1 + 4 grid neighbours (±1 step on rebalance frequency, ±1 step on portfolio breadth). If neighbour Sharpes spread > 30% relative, ABORT the audit and report — the design is parameter-fragile.

Pre-committed neighbours of C1 for the plateau check:

- P_rebalance_2week (between monthly and weekly)
- P_portfolio_decile (narrower legs than quintile)
- P_portfolio_third (broader legs than quintile)
- P_carry_3day_smooth (between 1-day and 5-day)

**Causality test (A10 / L04).** Pre-commit assertion: corrupt F_M1[t] and F_M2[t] at random t; assert per-bar strategy returns at t' < t are bit-exact unchanged. Live parity test required before any promotion.

**Specific failure modes to watch:**

- **L20 — Index normalisation.** 22 commodity parquets each with their own date index. All must be reindexed to the COMMON-date intersection BEFORE the cross-sectional rank is computed. Otherwise, an out-of-sample roll on one contract would silently exclude that commodity from the long-short on that day.
- **L25 — Class override transparency.** WFO 3y IS / 1y OOS is overridden from class default 5y IS. Documented above; the result log must report the override.
- **A6 — MC bootstrapping a basket.** Block-bootstrap the COMMON-date returns matrix (22 commodities × N bars) with SHARED block indices across columns — so the cross-sectional structure is preserved. Otherwise we'd shuffle individual columns independently and destroy the carry-signal-to-return correlation. The framework's `run_block_mc` works on a single primary close; we'll pass M1 as primary and the other 21 commodities as `extra_series`, which preserves shared-block bootstrap.
- **L33 (G4 lesson) — Asymmetric tail risk on session-bounded strategies.** D2 is NOT session-bounded; the strategy holds continuously between rebalances. L33 does not apply.
- **Survivorship bias.** The 22-commodity universe is fixed throughout — no commodities added or removed mid-sample. This is appropriate for a 7.5-year window where the listed-contract set is essentially stable; no need for a survivorship-bias correction.

## §4. Result log

**Audit run:** 2026-05-15 PM. **ABORTED at plateau pre-flight (L27).** Per pre-reg §3, the full per-cell pipeline was not run because the canonical cell's 4 grid neighbours produced a relative Sharpe spread of **109.81%** (gate: 30%). Full output in `.tmp/reports/d2_futures_carry/result_log.md`.

Universe: 24 commodities (full BGR set), 6,648 daily bars 2000-01-03 → 2026-05-15. Signal mode: `rolling_yield` (BGR §3.2 proxy — 12-month trailing log return), per the §1 amendment after the Databento `.c.1` blocker.

### §4.1 Plateau pre-flight — the design-fragility verdict

| Cell | Stitched OOS Sharpe |
|---|---:|
| C1_canonical (monthly, quintile, 252d lookback) | -0.5949 |
| P_rebalance_2week (weekly rebalance) | -0.6207 |
| P_portfolio_decile (10% breadth) | -0.2958 |
| P_portfolio_third (33% breadth) | -0.5566 |
| P_carry_3day_smooth (3-day signal smooth) | -0.5998 |

**Relative spread: 109.81% / 30% gate → FAIL.** Range: -0.62 → -0.30. Every cell loses money. The breadth axis (decile vs quintile vs tercile) is the only non-uniform direction; smoothing and rebalance-frequency don't change the result.

### §4.2 Why this happened

The rolling-yield proxy carry signal is mathematically identical to a cross-sectional 12-month time-series momentum (long winners, short losers). In our 2000-2026 commodity sample, this signal has **NEGATIVE expected return** — commodities exhibit medium-horizon mean reversion: past 12-month winners underperform past losers over the subsequent month. This is the opposite of the BGR carry pattern (which is supposed to be a positive premium).

Two non-exclusive explanations:

1. **The proxy is a poor substitute for strict carry in this sample.** BGR's §3.2 reports ρ=0.65 between rolling yield and M1/M2 basis. But ρ=0.65 leaves 58% of the variance unexplained. The unexplained component (cross-sectional dispersion in convenience-yield / inventory factors) is what gives strict carry its positive premium. Without M2 data, we are not testing BGR's carry — we are testing cross-sectional momentum on commodities.
2. **Cross-sectional commodity momentum DID exist pre-2010 but has decayed/inverted post-2010.** The combined sample (25 years) blends a momentum-pays regime with a mean-reversion-pays regime, producing the negative-Sharpe outcome. Splitting the sample would test this but is RESERVED for a fresh pre-reg under V3.1 (no data-snooping the date cut).

### §4.3 Recommended next step

**RETIRE the rolling-yield proxy variant of D2.** The audit is closed at the plateau-pre-flight stage; no per-cell MC/DSR/noise gates were run.

To test BGR strict carry properly, the next iteration needs:

1. **A working source of M2 (second-month continuous) data.** Options: (a) wait for Databento `.c.1` server-side fix, (b) pay for a different vendor (CSI Data, Norgate), (c) build our own M2 series from raw individual contracts with manual rolling logic.
2. A fresh pre-reg (D2b) with the strict M1/M2 basis signal explicitly required.

Until M2 is available, **D2 is RETIRED from the backlog as an open audit item.** The L25 class-default override (3y IS) and the `rolling_yield` signal-mode infrastructure remain in the codebase as documentation; they don't ship as a deployed strategy.

### §4.4 New lessons (V3.6 Catalogue)

- **L34 (new):** Databento `GLBX.MDP3` continuous-contract symbology `.c.1` (and presumably `.c.N` for N>0) is server-side pathologically slow — ~5 s/bar versus `.c.0`'s ~0.005 s/bar (a 1000× ratio for the same daily OHLCV schema). For a full 2017-2026 window (~2783 bars) this means ~4 hours per commodity for the second-month series, infeasible for a 22-commodity download. The `.c.0` queries work normally. Working assumption is that Databento's continuous-symbology pre-computation pipeline only fully materialises `.c.0` (the front month, by far the most commonly queried); back-month continuous mappings get computed on-demand at query time. **How to apply.** (a) For multi-commodity audits needing back-month data: budget a dedicated overnight run for `.c.1` if Databento is the source, or (b) fall back to per-contract raw_symbol queries with manual rolling logic, or (c) use a different vendor (CSI Data, Norgate, or scrape Yahoo Finance for proxy series). **Source.** D2 audit data-acquisition incident 2026-05-15: CL.c.0 returned 2783 bars in ~3 s; CL.c.1 short-window query returned 5 bars in 24.9 s; CL.c.1 full-window query never returned. Confirmed pattern across GC, ZC, and other commodity roots.

- **L35 (new):** When falling back from strict-carry M1/M2 basis to the BGR §3.2 rolling-yield proxy, the strategy stops being a carry strategy and becomes a cross-sectional momentum strategy. The two have ρ~0.65 in BGR's data but FUNDAMENTALLY DIFFERENT directional behavior in samples where mean reversion dominates momentum. **Empirical:** D2 plateau pre-flight on 24-commodity rolling-yield 2000-2026 produced ALL cells Sharpe -0.30 to -0.62. The negative sign signals that the strategy is harvesting commodity mean-reversion (against the rolling-yield signal) rather than the strict-carry premium. **How to apply.** Never substitute a "proxy" signal for the strategy's structural signal without explicitly testing whether the proxy's directional behavior matches the original in the target sample. A correlation of 0.65 is NOT a substitute for a directional check — 35% unexplained variance can flip the sign of expected return in cross-sectional sorts. Document the proxy as a SEPARATE STRATEGY HYPOTHESIS, not a fallback for the original. **Source.** D2 audit 2026-05-15.

---

## §5. Failure modes to watch (V3.6 lessons applied)

- **L04 / A1** — All signals via `.shift(1)` on F_M1, F_M2 series. Position × return product uses `pos.shift(1) * ret`. Causality test (A10) runs at audit start.
- **L06** — Per-day MTM Sharpe (class default). Strategy holds continuously between monthly rebalances; per-trade Sharpe would inflate by ~sqrt(monthly_turnover) — DO NOT use.
- **L08** — MC threshold from CARRY class defaults (P(MaxDD>30%) < 10%). No override on the MC axis.
- **L17** — Absolute MC, NOT relative. Long-short market-neutral; no buy-hold equity benchmark applies.
- **L20** — Common-date reindex MANDATORY before cross-sectional rank.
- **L24** — Per-cell Varma noise gate.
- **L25** — Class-default WFO override (3y IS, not 5y) is pre-registered here with rationale.
- **L27** — Plateau pre-flight runs BEFORE the full per-cell sweep.
- **A3** — Databento futures continuous contracts return *raw* (non-adjusted) prices. F_M1/F_M2 are price-only; the log-ratio is robust to the level. Document.
- **A4** — Cells pre-registered here; no per-fold parameter selection.
- **A5 / V3.1** — DSR at N=7 trials with actual skew/kurt.
- **A6** — Shared-block bootstrap on the multi-column return matrix; the framework's `run_block_mc` extra_series mechanism preserves cross-sectional structure.

## §6. Implementation plan

1. **Strategy module** at `research/futures_carry/carry_strategy.py`:
   - `CarryConfig` dataclass.
   - `compute_carry_signal(M1_df, M2_df, *, smooth_days=1) -> pd.DataFrame` — common-date reindexed carry per commodity.
   - `build_portfolio_weights(signal_df, *, cfg) -> pd.DataFrame` — apply portfolio rule (quintile, tercile, long-only, vol-weighted).
   - `carry_returns(M1_df, M2_df, *, cfg) -> pd.Series` — top-level public function.
   - `carry_assert_causal(M1_df, M2_df, *, cfg, n_trials)` — A10 smoke test.
2. **Audit harness** at `research/futures_carry/run_d2_audit.py`. Plateau pre-flight + per-cell 5-axis decision. Output to `.tmp/reports/d2_futures_carry/`.
3. **Tests** at `tests/test_futures_carry.py`:
   - Class defaults consistency.
   - Carry signal correctness (synthetic: known M1/M2 → expected raw_carry).
   - Portfolio construction (quintile, tercile, long-only).
   - Causality (A10).
   - Common-date reindex correctness.
4. **Run audit**, append §4 result log + any new lessons.
5. **If verdict ≥ CONDITIONAL_WATCHPOINT with noise=best:** port to `titan/strategies/futures_carry/` per Strategy Deployment Guide. Live class needs CME futures contract resolution + monthly rebalance scheduling — non-trivial; separate PR.

After D2 lands, the next backlog step is **J1/J2 — HRP → NCO allocator** (3+2d infrastructure) — load-bearing once we have ≥ 3 DEPLOYed strategies (currently 1: GEM A1).
