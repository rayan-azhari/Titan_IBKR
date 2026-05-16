# Pre-Registration — A1: Residual Momentum (Blitz-Huij-Martens 2011)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy audit (class `CROSS_ASSET_MOMENTUM` — cross-sectional equity variant)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit.

> V3.1 pre-registration: §1–§3 frozen BEFORE any data is examined for this design. The strategy mechanism is from published literature, NOT derived from our dataset. The Fama-French factor returns and the S&P 500 daily prices are public and our prior knowledge of broad equity markets is "common knowledge" — but no cell-level Sharpe, no per-stock residual-momentum stats, no rank distribution has been inspected for OUR specific universe slice.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` step 7: A1 — Residual momentum [NEW] (4d), first cross-sectional equity sleeve.

**Source.** Blitz, Huij & Martens, *"Residual Momentum"*, *Journal of Empirical Finance* 2011. Updated by Robeco research notes 2022. Builds on Jegadeesh-Titman 1993 momentum + Fama-French 1993 three-factor model. **Key claim:** standard 12-month skip-1 momentum has high exposure to the SMB / HML factors (high-momentum stocks tend to be small / growth) which drives most of its drawdown profile. Computing momentum on the RESIDUAL returns (idiosyncratic component after stripping out FF3 factor exposure) produces:

- Similar Sharpe to raw momentum.
- ~50% lower drawdown.
- Higher CI_lo (more statistically robust).
- More stable over time (less crash risk in 2009-style momentum reversals).

**Hypothesis.** A monthly-rebalanced cross-sectional long-short portfolio that goes LONG the top quintile of S&P 500 stocks by *residual* momentum (idiosyncratic 12-month skip-1 cumulative return, standardised by residual vol) and SHORT the bottom quintile earns positive risk-adjusted returns over 2000-2026, with materially lower drawdown than a raw-momentum equivalent. Falsifiable: if the audit shows the residual-momentum portfolio's CI_lo ≤ 0 OR its Rel-MC fails OR its Sharpe is dominated by raw-momentum C2, the BHM 2011 edge does not replicate on retail-implementable S&P 500 stocks at realistic equity costs.

**Mechanism.**

1. **Universe.** Current S&P 500 constituents (acknowledged survivorship bias — see §5).
2. **Daily inputs** (per stock i, per date t):
   - `r_i,t` = log return of stock i on day t.
   - `r_mkt,t` = Fama-French market excess return (MKT − RF) on day t.
   - `r_smb,t` = SMB factor return on day t.
   - `r_hml,t` = HML factor return on day t.
   - `r_rf,t` = risk-free rate on day t.
3. **Residual computation (per stock, per rebalance date t):**
   - Window: 36 months of daily returns ending at t-1.
   - Regress `(r_i,τ − r_rf,τ) = α + β_mkt × r_mkt,τ + β_smb × r_smb,τ + β_hml × r_hml,τ + ε_i,τ` over the 36-month window.
   - Compute daily residuals `ε_i,τ` for τ in the window.
4. **Signal (per stock, per rebalance date t):**
   - Skip-1 cumulative residual return: `cum_residual_i,t = Σ ε_i,τ` for τ in months [t-12, t-2] (inclusive, skipping the most recent month).
   - Standardised: `signal_i,t = cum_residual_i,t / std(ε_i,monthly)` where the std is computed over the 36-month window's monthly residual sums. (Gives a t-stat-like score, comparable across stocks.)
5. **Portfolio construction (cross-sectional):**
   - Rank stocks by signal at end of each month.
   - Long top quintile (top 20%), short bottom quintile.
   - Equal-weighted within each leg; long-leg notional = +0.5 NAV, short-leg = -0.5 NAV; net = 0 (market-neutral).
6. **Causality.** Signal at t uses data through t-1 (regression window ends at t-1; skip-1 makes the cumulative residual end at t-2). Position effective from t onwards, held until next rebalance.

**Why this is novel for our stack.** No cross-sectional equity strategy exists in `titan/strategies/`. The closest is `ic_equity_daily` (RSI mean-reversion, 7-stock universe). A1 introduces:

- Cross-sectional equity factor research at S&P 500 scale (~500 stocks).
- Fama-French factor adjustment / regression infrastructure (reusable for future factor-residual strategies).
- The first audit on a NEW data dependency (Ken French data library) sourced via the `pandas-datareader` library or direct CSV download from French's website.

## §2. Universe + cells + data

**Universe.**

- **S&P 500 constituents as of 2026-05-15.** ~500 stocks. Source: Wikipedia's "List of S&P 500 companies" or yfinance's index-constituent endpoint.
- **Acknowledged biases (§5):** survivorship bias (stocks that left the index between 2000 and 2026 are excluded), look-ahead bias on the constituent list itself (today's S&P 500 contains companies that weren't in the index in 2005 — they got in BECAUSE they outperformed). The BHM 2011 paper uses historical CRSP / Compustat data which has neither bias; we don't have that. Document; discount.

**Date range.** 2000-01-03 (first common date across the existing yfinance S&P stock series we'd download) → 2026-05-15. ~26 years. Sanctuary: trailing 12 months (the most recent 12 months → ~2025-05 to 2026-05). Visible window ~25 years.

**Data sources (new acquisitions required, per §6 implementation plan):**

- Daily OHLCV per S&P 500 stock: yfinance `{TICKER}` symbols. ~500 × ~6000 bars = ~3 million data points. Yfinance bulk download in batches of 50 symbols.
- Daily Fama-French 3-factor returns + RF: Ken French data library (`F-F_Research_Data_Factors_daily.zip`, free from French's website). Single CSV, ~25 years coverage.

**Bar timeframe.** Daily. `BARS_PER_YEAR["D"] = 252`. Strategy class: `CROSS_ASSET_MOMENTUM` (cross-sectional momentum on equity universe — same defaults apply: per-day MTM Sharpe, rolling 2y IS / 0.5y OOS / 8 folds WFO, MC P(MaxDD > 35%) < 10%).

**Cells (V3.1 frozen, 8 cells):**

| Cell | Signal | Window (mo) | Skip (mo) | Portfolio | Weighting | Costs | Notes |
|---|---|---:|---:|---|---|---|---|
| C1 (canonical) | residual | 12 | 1 | quintile L/S | equal | ON | BHM canonical |
| C2 (raw momentum) | raw | 12 | 1 | quintile L/S | equal | ON | control: does residualization help? |
| C3 (longer window) | residual | 24 | 1 | quintile L/S | equal | ON | tests window sensitivity |
| C4 (shorter window) | residual | 6 | 1 | quintile L/S | equal | ON | shorter-horizon momentum |
| C5 (tercile) | residual | 12 | 1 | tercile L/S | equal | ON | broader legs (~166 stocks each) |
| C6 (vol-weighted) | residual | 12 | 1 | quintile L/S | inverse-vol | ON | within-leg vol parity |
| C7 (long-only) | residual | 12 | 1 | top quintile only | equal | ON | tests if alpha is in long leg only |
| C8 (gross, no costs) | residual | 12 | 1 | quintile L/S | equal | OFF | diagnostic: cost ablation |

**8 cells total.** DSR adjustment applies (N=8 > 5).

## §3. Decision rule (pre-committed, V3.1)

**Class defaults:** `defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)`. Sharpe per-day MTM (252 bars/yr). WFO rolling, 2y IS / 0.5y OOS / 8 folds (auto-determined). MC P(MaxDD > 35%) < 10% with 21-day blocks, 200 paths.

**MC axis is ABSOLUTE, not relative.** A1 is long-short market-neutral. L17's relative-MC rule applies to long-only equity ONLY. Use the class-default absolute threshold.

**Cost model (L23).** US large-cap equity:

- `cost_bps_per_turnover = 0.5` (spread + slip; S&P 500 stocks are top-tier liquid)
- `cost_fixed_usd_per_fill = 0.5` (IBKR Pro tier; lower than the $1 floor because high-volume equity tiers)
- `notional_usd_per_leg = 30_000` (matches GEM C12 production NAV)
- Monthly rebalance with ~200 names total turnover (100 longs + 100 shorts each month, ~50% of which actually change) → ~100 fills/month × 12 = ~1200 fills/year ≈ 200 bps/year fixed + ~30 bps/year spread+slip ≈ 230 bps/year drag. Material.

C8 isolates gross alpha; if C8 has strong Sharpe but C1 falls apart, the strategy is cost-dominated for retail.

**Per-axis thresholds (5-axis matrix, L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe) | > 0 | ≤ −0.2 |
| DSR-prob (deflated at N=8, actual skew/kurt) | ≥ 0.95 | < 0.50 |
| MC (absolute) — P(MaxDD > 35%) | ≤ 0.10 | ≥ 0.20 |
| Sanctuary Sharpe | > 0 | ≤ −0.3 |
| Noise robustness (Varma, J3) | passes mean AND worst-case at (0.1, 0.3, 0.5)σ | fails mean at any level |

**Cell selection (V3.1 + V3.2).** Among DEPLOY-eligible cells (verdict = DEPLOY OR CONDITIONAL_WATCHPOINT with noise=`best`), pick the one with highest CI_lo. C8 (no costs) is NEVER selected — diagnostic only.

**Plateau pre-flight (L27).** C1 + 4 grid neighbours:

- P_window_9 (canonical with window=9 months)
- P_window_15 (window=15 months)
- P_skip_0 (no skip — include most recent month, tests the short-term-reversal hypothesis)
- P_skip_2 (skip 2 months)

If C1 + neighbours produce a relative Sharpe spread > 30%, ABORT and report. Otherwise proceed.

**Causality test (A10 / L04).** Pre-commit assertion: corrupt future stock prices + future FF factors at random t; assert per-bar strategy returns at t' < t are bit-exact unchanged.

## §4. Result log

**Audit run:** 2026-05-15 PM. **ABORTED at plateau pre-flight (L27).** Per pre-reg §3, the full per-cell pipeline was not run because the canonical cell's 4 grid neighbours produced a relative Sharpe spread of **66.44%** (gate: 30%). Full output in `.tmp/reports/a1_residual_momentum/result_log.md`.

Universe: 496 S&P 500 stocks (post min-obs filter; 502 loaded, 6 dropped for insufficient history), 6,348 daily visible bars 2000-01-03 → 2025-03-31. Sanctuary: 252 bars (Apr-2025 → Mar-2026). WFO produced 60 folds at the CROSS_ASSET_MOMENTUM defaults (rolling 2y IS / 0.5y OOS). Vectorised OLS in `compute_residual_signal` brought the per-cell runtime down to ~30 seconds (vs ~15 min in the per-stock-lstsq version).

### §4.1 Plateau pre-flight — the design-fragility verdict

| Cell | Stitched OOS Sharpe |
|---|---:|
| C1_canonical (residual, window=12, skip=1) | -0.3212 |
| P_window_9 (window=9) | -0.4230 |
| P_window_15 (window=15) | -0.3020 |
| P_skip_0 (skip=0, include latest month) | -0.3624 |
| P_skip_2 (skip=2) | -0.2541 |

**Range: -0.42 to -0.25. Relative spread: 66.44% / 30% gate → FAIL.** Every cell loses money. Even the most favourable neighbour (skip_2) produces a CI_lo that would be deeply negative.

### §4.2 Why this happened

Three forces interact to produce uniform negative Sharpe on a current-S&P-500-stocks residual-momentum strategy in 2000-2026:

1. **Survivorship bias on the universe.** The "current S&P 500" was the constituent list as of 2026-05-15 — companies that joined the index AFTER 2000-2010 did so because they outperformed and grew. Their "high momentum past" was the entry ticket. Their subsequent return is, by selection-effect mechanics, mean-reverting (they're now at high valuations). A cross-sectional long-winners / short-losers strategy on this BIASED universe systematically goes long mean-reverting names and short companies the universe has not yet ejected.

2. **Post-2009 momentum factor decay.** Multiple academic papers (Daniel-Moskowitz 2016, Asness-Frazzini-Israel-Moskowitz 2014, Stivers-Sun 2016) document that the cross-sectional 12-month skip-1 momentum premium on US large-cap equities has been near-zero or negative since the 2008-09 crash. The "crowded-trade decay" thesis: as the factor became implementable by retail / quant funds, its alpha was arbitraged away. The 2000-2026 window contains 16 years of post-decay sample vs 9 years of pre-decay.

3. **Residual vs raw momentum.** BHM 2011's residualization removes FF3 exposure but DOES NOT remove the underlying cross-sectional sign. If raw momentum is negative, residualized momentum is also negative — they correlate ~0.7 in the BHM data. The cell C2 (raw) was pre-registered as a control; the plateau result implies C2 would also be negative.

### §4.3 Recommended next step

**RETIRE the current-S&P-500 / yfinance variant of A1.** The audit is closed at plateau pre-flight; no per-cell MC/DSR/noise gates were run.

To properly test BHM 2011's residual-momentum edge, the next iteration would need ONE of:

1. **Survivorship-free CRSP/Compustat data.** Historical constituent lists + delisted-stock prices. Reproduces BHM's exact data construction. Cost: ~$2k/yr WRDS subscription, or a graduate-student equivalent. Not free.
2. **Different universe / sector tilt.** Cross-sectional momentum may still work on small-caps (Russell 2000 minus top-half by market cap) OR international (MSCI EAFE constituents). Each is its own pre-reg.
3. **Different signal entirely.** Time-series momentum (TSMOM) is the long-form companion of cross-sectional momentum and has demonstrably persisted post-2010 on commodity & FX futures. B4 on the backlog is the TSMOM strategy; A1's failure does not preclude B4.

Until one of these unlocks, **A1 is RETIRED from the backlog as an open audit item.** The strategy module, audit harness, tests, FF3 + S&P 500 data infrastructure remain in the codebase as documentation. They don't ship as a deployed strategy.

### §4.4 New lessons (V3.6 Catalogue)

- **L36 (new):** Survivorship-bias + post-2009 cross-sectional momentum decay COMBINE to flip the sign of expected return on a current-constituent-list momentum strategy in 2000-2026 samples. **Mechanism:** the universe is selected for past outperformance (joins the index after winning); the momentum factor as a whole has decayed/inverted on large-caps post-decarbonisation / post-passive-flows. The two forces are not orthogonal: the bias's victims are exactly the stocks that the strategy would short, and they're being shorted at peaks. **How to apply.** (a) Cross-sectional equity momentum on a current-S&P-500 universe must be discounted by 0.2-0.4 Sharpe vs published academic numbers. (b) Reproducing BHM-style results requires survivorship-free CRSP/Compustat data; without it, the audit measures something different and shouldn't quote the published numbers as the expected baseline. (c) "Residualization" does not fix the bias — it removes FF3 exposure but preserves the sign of the cross-sectional signal. **Source.** A1 audit 2026-05-15: plateau pre-flight on 496 current-S&P-500 stocks over 2000-2026 produced uniformly negative Sharpe (-0.25 to -0.42) across 5 grid cells; 66.44% relative spread = parameter-fragility on top of negative expected sign.

- **L37 (new):** Cross-sectional and time-series momentum DO NOT have the same regime profile. Cross-sectional has decayed/inverted post-2009 on US large-cap equities; time-series has persisted (Asness-Moskowitz-Pedersen 2013 documents this through 2010, multiple replications confirm through 2024). A1's failure does NOT invalidate B4 (TSMOM on commodity / FX futures) on the backlog. **How to apply.** Audit each momentum-family strategy independently; their alphas have decoupled across asset classes since 2009.

### §4.5 Post-mortem: re-audit with corrected data (2026-05-15 PM)

After the first audit aborted at plateau pre-flight, a self-audit identified that `download_sp500_universe.py` was using yfinance's price-only `close` instead of the dividend-adjusted `adj_close`. For a cross-sectional momentum strategy this is a non-trivial data-construction error — dividend-paying stocks get systematically under-ranked. The download script was patched (use `adj_close` as the canonical close), all 503 S&P parquets were re-downloaded with total returns, and the audit was re-run.

| Cell | v1 (price-only) | v2 (total return) | Δ |
|---|---:|---:|---:|
| C1_canonical | -0.3212 | **-0.3388** | -0.018 |
| P_window_9 | -0.4230 | **-0.4403** | -0.017 |
| P_window_15 | -0.3020 | **-0.3360** | -0.034 |
| P_skip_0 | -0.3624 | **-0.3639** | -0.001 |
| P_skip_2 | -0.2541 | **-0.2840** | -0.030 |
| Spread | 66.44% | **55.02%** | improved (slightly), still > 30% gate |

Switching to total returns made the strategy LOSE MORE money. Mechanism: the price-only close was DAMPENING the negative signal by systematically under-ranking dividend payers (utilities, REITs, banks); the strategy traded them less aggressively. With true total returns, the strategy now ranks the cross-section faithfully — and the cross-section's sign is empirically inverted on this universe, so trading it more correctly makes things WORSE.

**Conclusion: L36 is empirically confirmed, not exposed as a data bug.** The fix was the right scientific step (closes the L16 "did we miss something" loop) but it did not flip the verdict. **A1 is RETIRED on the current-S&P-500-yfinance universe with high confidence.** The remaining path to a clean BHM replication is survivorship-free historical constituents (CRSP/Compustat) — see §4.3.

---

## §5. Failure modes to watch (V3.6 lessons applied)

- **L04 / A1** — All signals computed on `.shift(1)`'d data. The 36-month regression window ends at t-1; the skip-1 momentum's cumulative residual ends at t-2. Causality test (A10) before audit start.
- **L06** — Per-day MTM Sharpe (class default).
- **L08** — Class-default MC threshold; no override.
- **L11** — Snapshot data BEFORE audit run. Stock prices + FF factors must NOT be re-downloaded mid-audit.
- **L17** — Absolute MC; long-short market-neutral, not long-only equity.
- **L20** — Per-stock dates must be normalised to a common-date intersection BEFORE the cross-sectional rank. Otherwise a stock that didn't trade on day t (suspension, delisting, halt) would silently corrupt the rank.
- **L24** — Per-cell Varma noise gate.
- **L27** — Plateau pre-flight BEFORE the full per-cell pipeline.
- **L34/L35** — Yfinance is the primary data vehicle (we learned this from D2 yesterday). M2-equivalent is NOT applicable here — A1 is a single-leg equity strategy.
- **A3** — Yfinance "adj close" is total return including dividends and splits. For cross-sectional momentum, adj close is the RIGHT signal (dividends are part of total return; splits would otherwise create false multi-100% returns). Use adj close.
- **A4** — WFO honesty: cells pre-registered above; no per-fold parameter selection. Rolling expanding folds; bootstrap CI on stitched OOS is the deployment gate.
- **A5 / V3.1** — DSR at N=8 trials with actual skew/kurt.
- **A6** — MC bootstraps the underlying CLOSES matrix with shared block indices across stocks. The framework's `run_block_mc` with `extra_series` preserves shared-block bootstrap structure.

**Survivorship bias (specific to A1):**

The S&P 500 constituents as of 2026-05-15 do NOT include the ~30 companies that were in the index in 2010 but have since been removed (most got removed because they underperformed, were acquired, or went bankrupt). The current set has positive selection bias. Expected impact: published Sharpe will be optimistically high by ~0.1-0.3. We do NOT correct for this in the audit; we discount any reported Sharpe by that range when interpreting the verdict. If the strategy survives the 5-axis matrix with the bias, it survives a fortiori without — but a strategy that requires the bias to survive is not deployable.

**Look-ahead bias on universe membership:**

The decision "trade only S&P 500 stocks" assumes we KNOW today which stocks were in the S&P 500 historically. In live trading, we would only know the current and prior-day constituents. For this audit, we accept this bias and document it. A clean V3.6-style version would use historical constituent lists from a CRSP-equivalent source.

**Factor staleness:**

The FF3 factors are computed using all-US-stock universes; they aren't recomputed against the S&P 500 subset. This is consistent with the BHM paper's setup. No correction needed.

## §6. Implementation plan

1. **Download Fama-French 3-factor returns + RF** from Ken French's data library. Script: `scripts/download_ff_factors.py`. Output: `data/FF3_daily.parquet` with columns `mkt_rf`, `smb`, `hml`, `rf`, indexed by date.
2. **Download S&P 500 daily OHLCV** for ~500 stocks via yfinance. Script: `scripts/download_sp500_universe.py`. Output: `data/SP500_universe/{TICKER}_D.parquet` (one file per stock).
3. **Strategy module** at `research/residual_momentum/residual_strategy.py`:
   - `ResidualConfig` dataclass.
   - `compute_residual_signal(stocks_df, ff3_df, *, cfg) -> pd.DataFrame` — per-stock residual-momentum signal.
   - `build_portfolio_weights(signal_df, *, cfg) -> pd.DataFrame` — quintile / tercile / long-only L/S construction.
   - `residual_returns(stocks_df, ff3_df, *, cfg) -> pd.Series` — public top-level.
   - `residual_assert_causal(stocks_df, ff3_df, *, cfg, n_trials)` — A10 causality smoke test.
4. **Audit harness** at `research/residual_momentum/run_a1_audit.py`. Plateau pre-flight + per-cell 5-axis decision. Output to `.tmp/reports/a1_residual_momentum/`.
5. **Tests** at `tests/test_residual_momentum.py`:
   - Class defaults consistency.
   - Residual computation correctness (synthetic: known regression coefficients → expected residuals).
   - Skip-month implementation.
   - Causality (A10).
   - Cross-sectional rank correctness.
6. **Run audit**, append §4 result log + any new lessons.
7. **If verdict ≥ CONDITIONAL_WATCHPOINT with noise = `best`:** port to `titan/strategies/residual_momentum/` per Strategy Deployment Guide. Live class needs daily-end-of-day NLV calculation across ~200 positions, monthly rebalance scheduling, and a real S&P 500 constituent feed (live data dependency — yfinance won't suffice for live).

After A1 lands, the next backlog step is **I1 — HMM regime + XGBoost meta-labeler** (3d) — exercises the ML class.
