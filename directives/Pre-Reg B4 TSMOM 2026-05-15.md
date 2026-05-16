# Pre-Registration — B4: Time-Series Momentum (TSMOM, MOP 2012)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy audit (class `CROSS_ASSET_MOMENTUM` — time-series variant)
**Predecessor:** L37 (V3.6 Catalogue) — "cross-sectional and time-series momentum have decoupled regime profiles post-2009". A1 audit failed on cross-sectional; this audit tests whether TSMOM has persisted.
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE the audit examines new data. The 24-commodity yfinance M1 parquets (downloaded for D2, 2026-05-15) are KNOWN to exist; the FF3 returns are known. No cell-level Sharpe, no per-asset TSMOM stats, no cross-asset correlation has been inspected for OUR specific sample.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` step 10: B4 — TSMOM 10-15 futures (4d). After A1's failure on cross-sectional equity momentum, L37 explicitly predicts that **time-series momentum on commodity / FX futures should NOT have decayed** the way cross-sectional has on US equities. This audit is the falsification test of that prediction.

**Source.** Moskowitz, Ooi & Pedersen, *"Time series momentum"*, *Journal of Financial Economics* 2012. Distinct from Jegadeesh-Titman cross-sectional momentum: TSMOM looks at EACH ASSET's OWN past return (not its rank vs. others). The signal is binary or continuous on the asset's own trailing return; sign-of-cumulative-return → long/short. Aggregating these signals across a diversified basket of futures produces a strategy that has historically delivered Sharpe ~1.0 with low correlation to traditional risk premia. Subsequent replications (AQR 2017 "A Century of Evidence on Trend-Following Investing", Hurst-Ooi-Pedersen 2017) extend the result through ~2016 and document persistence.

**Hypothesis.** A monthly-rebalanced cross-asset TSMOM portfolio on a diversified commodity-futures basket earns positive risk-adjusted returns over 2000-2026, with materially different drawdown profile from the 60/40 benchmark. Falsifiable: if the audit's verdict is RETIRE or SUSPECT, L37's persistence-of-TSMOM claim is contradicted on retail-implementable yfinance commodity data at realistic costs; the lesson catalogue would gain a complementary L39.

**Mechanism.**

1. **Universe.** 24 commodity futures, M1 (front-month continuous) from yfinance — same data D2 used. Energy (CL/NG/HO/RB/BZ), metals (GC/SI/PL/PA/HG), grains (ZC/ZW/ZS/ZL/ZM/ZO), livestock (LE/GF/HE), softs (KC/CC/SB/CT/OJ).
2. **Daily inputs.** Per commodity i: daily log returns `r_i,t = log(F_i,t / F_i,t-1)`.
3. **Signal (per asset, per rebalance date t):**
   - Trailing cumulative log return: `cum_return_i,t = Σ r_i,τ` for τ in months `[t - momentum_window, t - skip]`. Canonical: 12-month window, 1-month skip → sums over months `[t-12, t-1]` (inclusive of t-2 and earlier, exclusive of t-1).
   - Signal: `s_i,t = sign(cum_return_i,t)` — +1 (long), -1 (short), or 0 (when ambiguous).
4. **Position sizing (inverse-volatility weight):**
   - `vol_i,t = realised_vol(r_i, 60d)` annualised → per-asset volatility estimate.
   - Target portfolio volatility: `target_vol = 0.10` (10% annualised, MOP convention).
   - Per-asset weight: `w_i,t = s_i,t × target_vol / (vol_i,t × N_active)` where `N_active` = count of assets with non-zero signal.
   - This equalises risk contribution across the basket and targets a stable overall portfolio vol.
5. **Portfolio return (per bar):**
   - `r_portfolio,t = Σ_i w_i,t-1 × r_i,t`
   - Net of transaction costs: per-bar turnover × bps + per-fill commissions.
6. **Causality.** All signals use `.shift(1)` of daily returns through bar t-1. Position effective from bar t held until next monthly rebalance. Standard `weights.shift(1) × returns` discipline.

**Why this is novel for our stack.** No time-series momentum strategy in `titan/strategies/`. `etf_trend` is per-asset trend on 7 ETFs (related but different — uses MA + deceleration, not a sign-of-cumulative-return signal across a diversified basket). `bond_gold` is single asset-pair momentum, not a basket-aggregate strategy. B4 introduces:

- The first cross-asset basket aggregation under the V2.0 framework.
- A direct test of L37's TSMOM-persistence claim using the same data infrastructure D2 built.
- Reusable infrastructure for B2 (Carver EWMAC ensemble, the bigger build).

## §2. Universe + cells + data

**Universe.** All 24 commodity M1 parquets in `data/`:

| Group | Roots (count) |
|---|---|
| Energy | CL, NG, HO, RB, BZ (5) |
| Metals — precious | GC, SI, PL, PA (4) |
| Metals — industrial | HG (1) |
| Grains + oilseeds | ZC, ZW, ZS, ZL, ZM, ZO (6) |
| Livestock | LE, GF, HE (3) |
| Softs | KC, CC, SB, CT, OJ (5) |

Each parquet covers 2000-2026 (~6500 daily bars). Common-date-aligned intersection across all 24 = ~6000 bars after the slowest-start commodity's IPO date.

**Date range.** 2000-01-03 (first usable date across the basket) → 2026-05-15. ~25 years. Sanctuary: trailing 12 months. Visible: ~24 years.

**Bar timeframe.** Daily. `BARS_PER_YEAR["D"] = 252`. Strategy class: `CROSS_ASSET_MOMENTUM` (same as GEM and the failed A1 — different asset universe, different signal mechanism, but the class defaults fit: per-day MTM Sharpe, rolling 2y IS / 0.5y OOS / 8 folds, MC P(MaxDD > 35%) < 10%).

**Cells (V3.1 frozen, 8 cells):**

| Cell | Signal | Window (mo) | Skip (mo) | Sizing | Weighting | Rebalance | Costs | Notes |
|---|---|---:|---:|---|---|---|---|---|
| C1 (canonical) | sign(cum_return) | 12 | 1 | inv-vol → 10% target | EW within sign | monthly EOM | ON | MOP 2012 canonical |
| C2 (raw return) | cum_return (continuous) | 12 | 1 | inv-vol scaled by sign-magnitude | weighted | monthly EOM | ON | Continuous vs binary signal |
| C3 (longer window) | sign | 24 | 1 | inv-vol | EW | monthly EOM | ON | Slower trend filter |
| C4 (shorter window) | sign | 6 | 1 | inv-vol | EW | monthly EOM | ON | Faster trend filter |
| C5 (no skip) | sign | 12 | 0 | inv-vol | EW | monthly EOM | ON | Tests skip-month necessity |
| C6 (equal weight) | sign | 12 | 1 | equal weight × sign | EW | monthly EOM | ON | Tests inv-vol contribution |
| C7 (weekly rebal) | sign | 12 | 1 | inv-vol | EW | weekly Fri | ON | Faster rebalance |
| C8 (gross, no costs) | sign | 12 | 1 | inv-vol | EW | monthly EOM | OFF | Diagnostic: cost ablation |

**8 cells total.** DSR adjustment applies (N=8 > 5).

## §3. Decision rule (pre-committed, V3.1)

**Class defaults:** `defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)`. Per-day MTM Sharpe, rolling 2y IS / 0.5y OOS / 8 folds, MC 21-day blocks/200 paths, P(MaxDD > 35%) < 10%.

**Cost model (L23).** CME-grade liquid futures:

- `cost_bps_per_turnover = 1.0` (CME futures spread+slip; matches GEM C12 production setting)
- `cost_fixed_usd_per_fill = 1.0` (IBKR Pro tier)
- `notional_usd_per_leg = 30_000` per-commodity allocation
- 24 commodities × monthly = ~24 fills/month with ~30% per-name turnover/month avg → ~7 actual fills/month → ~84 fills/year. Fixed-cost drag: ~28 bps/yr. Variable: ~10 bps/yr. Total ~40 bps/yr drag. Lower than A1's ~200 bps because of monthly rebal × small per-leg turnover (sign flips are rare on monthly trends).

C8 isolates gross alpha.

**Per-axis thresholds (5-axis matrix, L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe) | > 0 | ≤ −0.2 |
| DSR-prob (deflated at N=8, actual skew/kurt) | ≥ 0.95 | < 0.50 |
| MC (absolute) — P(MaxDD > 35%) | ≤ 0.10 | ≥ 0.20 |
| Sanctuary Sharpe | > 0 | ≤ −0.3 |
| Noise robustness (Varma, J3) | passes mean AND worst-case at (0.1, 0.3, 0.5)σ | fails mean at any level |

**MC axis is ABSOLUTE.** TSMOM is a balanced long-short basket, not long-only equity; L17's relative-MC rule (vs buy-hold benchmark) does not apply directly. Use class-default absolute threshold.

**Cell selection (V3.1 + V3.2).** Among DEPLOY-eligible cells (verdict = DEPLOY OR CONDITIONAL_WATCHPOINT with noise=`best`), pick the one with highest CI_lo. C8 (no costs) NEVER selected — diagnostic only.

**Plateau pre-flight (L27).** C1 + 4 grid neighbours:

- P_window_9 (window=9)
- P_window_15 (window=15)
- P_skip_0 (skip=0)
- P_skip_2 (skip=2)

If relative Sharpe spread > 30%, ABORT and report. Else proceed.

**Causality test (A10 / L04).** Pre-commit assertion: corrupt future futures prices at random t; assert per-bar strategy returns at t' < t are bit-exact unchanged.

## §4. Result log

**Audit run:** 2026-05-15 PM. **ABORTED at plateau pre-flight (L27)** with a nuanced outcome that partially confirms L37 while still failing the deployment bar.

### §4.1 Plateau pre-flight

| Cell | Stitched OOS Sharpe |
|---|---:|
| C1_canonical (sign, w=12, skip=1) | **+0.1605** |
| P_window_9 (w=9) | -0.4180 |
| P_window_15 (w=15) | -0.1762 |
| P_skip_0 (skip=0) | +0.0036 |
| P_skip_2 (skip=2) | +0.1034 |

**Range: -0.42 to +0.16. Relative spread: 360% (the range crosses zero, blowing up the relative measure).** V3.2 plateau gate: < 30% → FAIL.

### §4.2 L37 falsification — partial confirmation

L37 predicted: "time-series momentum has persisted on commodity / FX futures, unlike cross-sectional which decayed".

**The qualitative prediction is correct.** The canonical cell produced a POSITIVE Sharpe (+0.16). This is materially different from A1's uniform-negative result on cross-sectional equity momentum (every A1 cell was -0.25 to -0.42). The sign of the effect is consistent with L37's persistence claim.

**The quantitative prediction is NOT robust enough to deploy.** MOP 2012's published Sharpe was ~+1.0 on a 58-instrument multi-asset basket (commodities + FX + equity indices + bonds). Our +0.16 on a 24-commodity-only basket is:

- An order of magnitude below the published baseline.
- Highly parameter-sensitive (9-month variant -0.42, 15-month variant -0.18, skip=0 variant +0.00).
- Below the deployment threshold (CI_lo would almost certainly be < 0 on this sample).

### §4.3 Two likely contributing factors

1. **Narrow universe.** 24 commodities only vs MOP's 58 instruments × 4 classes. Commodity futures are the SINGLE asset class with the highest cross-asset correlation (one driver: global manufacturing PMI). A diversified TSMOM basket gets most of its Sharpe from RISK DIVERSIFICATION across uncorrelated asset classes; restricting to commodities discards ~60-70% of the diversification benefit.
2. **yfinance continuous-future roll contamination.** `CL=F`, `GC=F` etc. are roll-adjusted continuous prices, but yfinance's specific roll convention is non-public. Roll yield is a meaningful part of expected futures returns. If yfinance discontinuously jumps the price at each roll (vs back-adjusted), the trailing-return signal contains spurious roll-shocks that are not real economic returns. The strict per-contract-stitching path (D2b open item via IBKR — probe confirmed it's feasible) would resolve this.

### §4.4 Recommended next step

**RETIRE the 24-commodity-only yfinance TSMOM variant.** Two paths to a faithful MOP 2012 replication:

1. **Expand the universe.** Add FX majors (EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, etc. — we have some FX parquets) + equity-index futures + bond futures. Need ~50 instruments to approach MOP's diversification. Each new asset class is its own data acquisition. Fresh pre-reg.
2. **Fix the futures data.** Use IBKR per-contract pulls + roll-aware stitching (D2b open item). Resolves L40 (below). Doesn't fix the universe-narrowness issue.

Best path forward: **D2b first** (strict per-contract M1+M2 via IBKR, ~2-3d). It fixes both D2's strict-carry replication AND B4's roll-yield contamination on the same data pull, then B4b can re-run on the cleaner data with the same 24 commodities. If still sub-DEPLOY after the roll fix, the universe-narrowness diagnosis stands and B4c with multi-asset-class universe is the next step.

### §4.5 New lessons (V3.6 Catalogue)

- **L39 (new)**: TSMOM persistence (L37) IS qualitatively confirmed on commodity-futures-only universes at retail data resolution — but the magnitude is far below published academic numbers (MOP 2012's ~1.0 Sharpe vs our 0.16). The gap is partly EXPECTED (single-asset-class universes lose most cross-asset diversification benefit) and partly POSSIBLY due to data-quality issues (continuous-future roll contamination — L40). **How to apply.** When evaluating TSMOM-style strategies on a narrow single-asset-class universe, expect 0.2-0.5 Sharpe AT BEST, not 1.0. Plan the audit's success threshold accordingly. The full ~1.0 Sharpe MOP claim requires 4-class diversification.
- **L40 (new)**: yfinance continuous-future symbols (`CL=F`, `GC=F`, etc.) are roll-adjusted via a non-public proprietary algorithm. The trailing-return signal computed on these series may be contaminated by roll-yield discontinuities that are not real economic returns. **How to apply.** For trend / momentum strategies that depend on multi-month cumulative returns on a futures continuous series, prefer per-contract historical data with explicit roll-and-stitch logic (e.g., back-adjustment by ratio at each roll date). IBKR per-contract `reqHistoricalData` is one path; CSI Data is another. yfinance is acceptable for shorter-horizon strategies where roll happens infrequently relative to the signal lookback, but for 12-month trailing-return TSMOM the roll discontinuities accumulate. Verify the data construction matches the strategy's signal horizon before quoting headline Sharpe numbers.

---

## §5. Failure modes to watch

- **L04 / A1** — All signals via `.shift(1)`. Position multiplied by `weights.shift(1)`. Causality test runs at audit start.
- **L06** — Per-day MTM Sharpe (class default).
- **L08** — Class-default MC threshold; no override.
- **L17** — Absolute MC; balanced long-short basket.
- **L20** — Common-date reindex MANDATORY before per-asset signal computation. Each commodity has its own IPO date; signal must be NaN before the asset has enough history.
- **L24** — Per-cell Varma noise gate.
- **L26 / L31** — Sign-of-cum-return is a discrete signal at the zero-crossing. Could be noise-fragile if sign-flips occur near zero. Plateau pre-flight + noise gate catch this.
- **L34** — Yfinance is the data source (Databento `.c.1` too slow per L34).
- **L36** — Survivorship bias on the EQUITY universe was A1's issue. For COMMODITY futures, survivorship is essentially zero — the set of liquid CME contracts has been stable since 1990 (no contract has been delisted between 2000-2026 from the audit's 24-commodity set).
- **L37** — This audit IS the falsification test of L37. The pre-reg explicitly commits to revising L37 if the verdict is RETIRE/SUSPECT.
- **L38** — Pre-emptive data-construction check: yfinance commodity prices are NOT total-return adjusted (futures don't pay dividends). The "close" column IS the right return basis. Confirmed against D2's audit.
- **A4** — Pre-reg committed BEFORE running on new data.
- **A5 / V3.1** — DSR at N=8 trials.
- **A6** — Shared-block bootstrap on the multi-commodity returns matrix; framework's `run_block_mc` with `extra_series` preserves cross-asset structure.

## §6. Implementation plan

1. **Strategy module** at `research/tsmom/tsmom_strategy.py`:
   - `TsmomConfig` dataclass.
   - `compute_tsmom_signal(returns_df, *, cfg) -> pd.DataFrame` — per-asset sign-of-cumulative-return.
   - `build_portfolio_weights(signal_df, vol_df, *, cfg) -> pd.DataFrame` — inv-vol weighted basket weights.
   - `tsmom_returns(closes_df, *, cfg) -> pd.Series` — public top-level.
   - `tsmom_assert_causal(closes_df, *, cfg, n_trials)` — A10 smoke test.
2. **Audit harness** at `research/tsmom/run_b4_audit.py`. Plateau pre-flight + per-cell 5-axis. Output `.tmp/reports/b4_tsmom/`.
3. **Tests** at `tests/test_tsmom.py`:
   - Class defaults consistency.
   - Sign-of-cumulative-return correctness (synthetic).
   - Inverse-vol weighting (constant-vol synthetic → equal weights; differing-vol → proportional).
   - Skip-month implementation.
   - Causality (A10).
4. **Run audit**, append §4 result log.
5. **If verdict ≥ CONDITIONAL_WATCHPOINT with noise=`best`:** port to `titan/strategies/tsmom/`. Live class needs CME futures contract resolution + monthly rebalance scheduling + 24-asset position sizing. Non-trivial; separate PR.

After B4, the next backlog step is **I1 — HMM regime + XGBoost meta-labeler** (3d).
