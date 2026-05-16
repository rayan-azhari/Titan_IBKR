# Pre-Registration — E1: VRP Capture via SPY + VIXY

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-15
**Branch:** v2-main
**Type:** Strategy audit (DAILY_MEAN_REVERSION class)
**Status:** §1–§3 frozen at commit; §4 result log appended after audit

> This is a V3.1 pre-registration committed BEFORE any data is examined. §1–§3 stay frozen for the lifetime of the audit. Gates can only be RELAXED in a separate PR explaining why the original was unimplementable.

---

## §1. Hypothesis & mechanism

**Backlog reference.** `directives/Strategy Backlog 2026-05-14.md` step 3 of the recommended execution order: **E1 — VRP capture via SPY/VIXY** (2d). Different class from GEM and J3 — first audit under the 5-axis decision matrix (L24).

**Hypothesis.** The Volatility Risk Premium (VRP) — the persistent gap between implied volatility (VIX) and subsequent realised volatility (RV) — is a compensated risk premium. Selling vol via short-VIX-futures exposure (here, short VIXY ETF) earns a positive expected return in *contango* regimes, where VIX9D < VIX < VIX3M (i.e. the near-term term structure is upward-sloping). The position must be hedged or risk-classified to avoid being annihilated by the rare-but-massive *backwardation* events (2008, 2018-02, 2020-03, 2022).

**Mechanism.**

1. **Daily decision** at the close (US session).
2. **Term-structure regime signal** computed on bar `t-1`:
   - `ratio_short = VIX(t-1) / VIX9D(t-1)` — short-end slope. >1 = contango at the short end.
   - `ratio_long  = VIX3M(t-1) / VIX(t-1)` — medium-end slope. >1 = full contango (calm).
3. **Position rule:**
   - **Contango regime** (`ratio_short ≥ 1.0` AND `ratio_long ≥ 1.05`): SHORT VIXY at weight = `target_short_weight` (default −0.50 of NAV — capped because VIXY draws down hard on spikes).
   - **Backwardation regime** (`ratio_long < 0.98`): FLAT VIXY. Optional `defensive_long_spy_w` long-SPY exposure (default 0) to keep capital working in the regime where VRP doesn't compensate.
   - **Mid-zone** (everything else): FLAT VIXY.
4. **Buffered transitions.** Once a regime is entered, require the ratio to cross the opposite gate by at least `regime_buffer_pct` (default 0.02) before flipping. Reduces churn during near-tie days.
5. **Causality.** All signals use VIX / VIX9D / VIX3M closes through bar `t-1`. New VIXY position effective at the close of bar `t` (held one full bar: `close[t] → close[t+1]`). Strict `.shift(1)` discipline on signals.

**Why this is novel for our stack.** No volatility-ecosystem strategy exists yet in `titan/strategies/`. The backlog explicitly lists `E1 — VRP capture via SPY/VIXY` (`DAILY_MEAN_REVERSION` class). The mechanism is well-documented (Eraker & Wu, *JFE* 2017; Cheng SSRN 2020; Hartz-Mittnik-Paolella 2006; Konstantinidi-Skiadopoulos-Tzagkaraki 2008). It is also the **first non-momentum strategy** we audit under the V2.0 framework — important to cover a different class.

## §2. Universe + cells + data

**Universe (fixed for this audit):**

- **VIXY** (ProShares VIX Short-Term Futures ETF; inception 2011-01-04). Implementation vehicle — we SHORT it during contango.
- **VIX** (CBOE 30-day implied vol index; signal input — already in `data/VIX_D.parquet`).
- **VIX9D** (CBOE 9-day implied vol index; signal input — `data/VIX9D_D.parquet`).
- **VIX3M** (CBOE 3-month implied vol index; signal input — `data/VIX3M_D.parquet`).
- **SPY** (US large-cap ETF; optional `defensive_long_spy_w` overlay — already in `data/SPY_D.parquet`).

**Data sources.** Yfinance parquets via `scripts/download_data_yfinance.py`. VIXY needs to be downloaded (not yet in `data/`); the audit harness must call the downloader before running, OR fail-fast with a clear error message if `data/VIXY_D.parquet` is absent.

**Date range:** 2011-01-04 (VIXY inception) → present. Sanctuary slice: trailing 12 months from latest data. Visible window ≈ 14 years × 252 ≈ 3500 bars — adequate for daily-bar WFO under `DAILY_MEAN_REVERSION` defaults.

**Bar timeframe:** Daily close. Bar-per-year convention: 252 (`BARS_PER_YEAR["D"]`).

**Strategy class:** `DAILY_MEAN_REVERSION` (the trade is mean-reversion of the VIX premium toward its long-run average).

**Cells.** Pre-committed grid (V3.1 — frozen at this commit):

| Cell | ratio_short_gate | ratio_long_gate | target_short_weight | regime_buffer_pct | defensive_long_spy_w |
|---|---:|---:|---:|---:|---:|
| C1 (canonical) | 1.00 | 1.05 | −0.50 | 0.02 | 0.00 |
| C2 (looser_long) | 1.00 | 1.02 | −0.50 | 0.02 | 0.00 |
| C3 (tighter_long) | 1.00 | 1.10 | −0.50 | 0.02 | 0.00 |
| C4 (smaller_short) | 1.00 | 1.05 | −0.25 | 0.02 | 0.00 |
| C5 (no_buffer) | 1.00 | 1.05 | −0.50 | 0.00 | 0.00 |
| C6 (spy_overlay) | 1.00 | 1.05 | −0.50 | 0.02 | 0.50 |
| C7 (with_short_signal) | 1.02 | 1.05 | −0.50 | 0.02 | 0.00 |

**7 cells total.** DSR adjustment applies (N=7 > 5 — L23 / A5).

## §3. Decision rule (pre-committed, V3.1)

**Class defaults (from `defaults_for(StrategyClass.DAILY_MEAN_REVERSION)`):**

- Sharpe convention: **per-day MTM** (the strategy holds positions for multiple bars; per-trade Sharpe inappropriate — L06).
- WFO: rolling, IS = 4y, OOS = 1y, ≥ 8 folds.
- MC gate: P(MaxDD > class_threshold) ≤ class pass_prob (read from `defaults_for(...)`).

**Per-axis thresholds (5-axis decision matrix — L24):**

| Axis | Best | Worst |
|---|---|---|
| CI_lo (95% bootstrap on stitched OOS Sharpe, 1000 resamples) | > 0 | ≤ −0.2 |
| DSR-prob (deflated at N=7 trials with actual skew/kurt) | ≥ 0.95 | < 0.50 |
| MC P(MaxDD > class_threshold) on underlying-resampled paths (block bootstrap, ~21-day blocks, 1000 paths) | ≤ class pass_prob | ≥ 2 × class pass_prob |
| Sanctuary Sharpe (on held-out 12mo) | > 0 | ≤ −0.3 |
| **Noise robustness (Varma, J3)** | **passes mean AND worst-case at every level in (0.1, 0.3, 0.5)σ** | **fails mean gate at any level** |

> NOTE on MC. VIXY's price path is endogenous to its VIX-future basket and to the regime — a naive block-bootstrap of VIXY's own returns under-tests the strategy because synthetic paths can't generate a 2018-02 / 2020-03 backwardation spike from contango-regime blocks alone. We will additionally run a **stress MC**: insert synthetic VIX spikes (drawn from the empirical distribution of historical 5-day VIX moves > +50%) at random points in the synthetic path. The framework's `run_block_mc` is used as-is for the headline; the stress MC is reported in §4 alongside as a robustness check, not a separate axis. (If the stress MC reveals MaxDD > 50% in ≥ 5% of paths, the cell is demoted to TIER_UNCONFIRMED regardless of the 5-axis verdict — a manual override codified here.)

**Verdict map (5-axis, J3):** 5 axes at best → DEPLOY, 4 → CONDITIONAL_WATCHPOINT, 3 → TIER_UNCONFIRMED, 2 → SUSPECT, 0–1 → RETIRE.

**Cell selection (V3.2 plateau).** The cell whose ±1-step grid neighbour also passes the same gates AND whose headline Sharpe varies by < 30% across the neighbourhood wins. C1 is the pre-committed canonical anchor; C2/C3 vary `ratio_long_gate` (the most consequential threshold); C4 varies `target_short_weight`; C5 ablates the buffer; C6 adds the SPY overlay; C7 ablates the short-end signal. Tie-break by parsimony (fewer non-default parameters).

**Causality test (A10 / L04).** Pre-commit assertion: corrupt VIX / VIX9D / VIX3M futures of all four series at random t, assert decisions at t' < t are bit-exact unchanged. Live parity test required before any DEPLOY/CWP cell is promoted to `titan/strategies/`.

## §4. Result log

**Audit run:** 2026-05-15. Full output in `.tmp/reports/vrp/result_log.md`.

### §4.1 Per-cell verdicts (5-axis, J3 / L24)

| Cell | Sharpe | CI95 lo | CI95 hi | DSR-prob | MC P(>25%) | Sanc Sharpe | Sanc pct | Noise base | Noise axis | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| C1_canonical | +0.3301 | -0.240 | +0.924 | 1.0000 | 1.0000 | +0.8279 | 0.625 | +0.4545 | worst | SUSPECT |
| C2_looser_long | +0.4583 | -0.115 | +1.105 | 1.0000 | 1.0000 | +1.0699 | 0.600 | +0.5515 | worst | SUSPECT |
| C3_tighter_long | +0.4507 | -0.113 | +1.057 | 1.0000 | 1.0000 | +0.6553 | 0.600 | +0.6664 | worst | SUSPECT |
| C4_smaller_short | +0.3284 | -0.242 | +0.922 | 1.0000 | 0.9250 | +0.8263 | 0.625 | +0.4528 | worst | SUSPECT |
| C5_no_buffer | +0.2134 | -0.321 | +0.774 | 0.9997 | 1.0000 | +0.6655 | 0.625 | +0.3815 | worst | SUSPECT |
| C6_spy_overlay | +0.3818 | -0.155 | +0.994 | 1.0000 | 1.0000 | +0.9141 | 0.625 | +0.5099 | worst | SUSPECT |
| C7_with_short_signal | +0.2469 | -0.317 | +0.861 | 1.0000 | 1.0000 | +0.3735 | 0.625 | +0.3735 | worst | SUSPECT |

**Result: all 7 cells → SUSPECT** (2 of 5 axes at "best": DSR + Sanctuary).

### §4.2 Verdict explanation

Three axes failed across the grid:

- **CI_lo**: every cell's 95% bootstrap CI straddles zero. After costs, there is insufficient statistical evidence that any of the 7 cells has a non-zero edge over the 14-year visible window. Point Sharpes are in the +0.21 to +0.46 range — modest, and within bootstrap noise of zero. The "VRP exists" hypothesis cannot be rejected, but neither can "VRP after costs ≈ 0".
- **MC**: every cell shows P(MaxDD > 25%) ≈ 1.0 under VIXY-block-bootstrap MC paths. Short-vol exposure is structurally drawdown-prone; the `DAILY_MEAN_REVERSION` class-default gate (P>25% < 10%) is calibrated for oscillator-based equity mean-reversion, not for short-vol-of-vol carry. The class default is too tight for this strategy class. V3.1 forbids relaxing the gate retroactively — instead, this is documented as a class-calibration mismatch (see L25 in V3.6 Catalogue).
- **Noise (Varma, J3 / L24)**: every cell fails the 0.30 max_degradation threshold. The regime-gate logic is threshold-based (cross ratio_long_gate = 1.05 → flip), so even 0.1σ input price noise can flip the regime decision on bars near the gate. Classic parameter-spike-not-plateau (V3.2). The noise axis worked exactly as J3 was designed to: it catches strategies whose realised Sharpe depends on the specific input price path, not on a durable signal.

### §4.3 Plateau check (V3.2)

Per-cell Sharpes range +0.21 (C5) to +0.46 (C2). Relative spread = (0.46 − 0.21) / 0.46 = 54%, FAILING the 30% spread gate. The grid does NOT exhibit plateau behaviour — Sharpe is highly sensitive to gate widths, which compounds the noise-axis failure: the strategy lives at a parameter spike, not a plateau.

### §4.4 Selected production cell

**None.** Verdict for the entire grid is SUSPECT. No cell promoted to `titan/strategies/`. Per the J3 pre-reg deployment rule, no CONDITIONAL_WATCHPOINT cells exist either.

### §4.5 New lessons (appended to V3.6)

- **L25 (new)**: MC gate thresholds must be calibrated by strategy class but ALSO by exposure type within the class. `DAILY_MEAN_REVERSION` defaults (P(MaxDD>25%) < 10%) are calibrated for long/short oscillator mean-reversion in equity space. They are systematically too tight for short-volatility carry strategies, where structural drawdowns of 30-50% are part of the trade. Either: (a) sub-class `DAILY_MEAN_REVERSION` into "equity oscillator" and "vol carry" with separate defaults, OR (b) document explicit per-strategy overrides in pre-reg directives. Pre-reg E1 used (a) implicitly by not overriding; a future VRP-capture variant should pre-register a strategy-specific MC threshold (e.g. P(MaxDD>50%) < 10% — Eraker-Wu and Cheng both report MaxDDs in this range as the cost of doing VRP business).
- **L26 (new)**: Threshold-based regime gates are STRUCTURALLY noise-fragile. Any "cross threshold → flip state" logic will fail the Varma noise gate unless the threshold is robustly inside a parameter plateau. Mitigations: (i) use percentile-based gates (e.g. "ratio_long in the top 40% of its 252-day rolling distribution") rather than fixed-value gates; (ii) blend signals across multiple gate widths and average the regime votes; (iii) apply a CONTINUOUS scaling function (e.g. `min(1, max(0, (ratio_long - 1.0) / 0.10))`) instead of a step function. (Source: VRP E1 audit, all 7 cells failed noise — variation across cells was 0.07-0.10 base Sharpe but every cell failed degradation gate.)

### §4.6 Negative-result discipline (V3.6 / L16)

Per L16: negative results are research output, not a wasted afternoon. **What's ruled out:**

1. Naive threshold-based VIX-term-structure regime gating on VIXY does not produce a deployable strategy under the V2.0 framework (5-axis matrix). The strategy is sample-fragile (CI_lo straddles zero) and noise-fragile (regime gates flip under tiny perturbations).
2. The `defensive_long_spy_w` overlay (C6) marginally improved Sharpe (+0.38 vs +0.33 C1) but did not move the needle on the other failing axes. SPY-overlay is not the missing piece.
3. The `regime_buffer_pct` is not load-bearing for the noise failure — even C5 (no buffer) and C1 (with buffer) both fail noise. Buffer was designed to fight regime churn near the gate, but Varma noise injection is a different fragility class.

**Promising next-iteration directions (NOT for this directive — would need a fresh pre-reg):**

- **Percentile-based regime signal:** replace `ratio_long >= 1.05` with `ratio_long >= 60th-pctile_rolling_252d(ratio_long)`. Tests both the L26 mitigation and gives the strategy a self-adaptive gate.
- **Continuous scaling overlay:** position size = `min(target, max(0, sigmoid((ratio_long - 1.0) / 0.05)))`. Eliminates the step-function fragility.
- **Class-recalibrated MC gate:** propose a new `DAILY_MEAN_REVERSION_VOL_CARRY` sub-class in `titan.research.framework.typology` with P(MaxDD>50%) < 10% defaults.
- **Implementation vehicle change:** VIXY is dominated by 1-month VIX futures; using SVXY (1× short, then -0.5× since 2018) or direct VIX futures might give a cleaner cost/exposure profile.

Each of these is a separate audit and a separate pre-reg. The current E1 directive is closed.

---

## §5. Failure modes to watch (V3.6 lessons applied)

- **L04 / A1 — Same-bar look-ahead.** All signals use `VIX.shift(1)`, `VIX9D.shift(1)`, `VIX3M.shift(1)`. Position multiplied by `pos.shift(1)`. AST-level guards enforce; per-cell causality test in `tests/test_vrp.py`.
- **L06 — Per-day MTM, not per-trade.** Strategy holds positions for multiple bars (regime changes daily-to-weekly cadence). Use class-default `DAILY_MEAN_REVERSION` Sharpe convention.
- **L08 — MC gate calibration.** Class-default thresholds for `DAILY_MEAN_REVERSION` MUST be used. Don't override.
- **L11 — Data overwrites.** Snapshot VIXY/VIX/VIX9D/VIX3M/SPY parquets before run; commit the snapshot timestamp in the result log.
- **L17 — Relative MC.** Long-VIX-short is not a long-only equity strategy — the absolute MC gate is the right fit here (not relative). Document explicitly.
- **L18 — Stateful buffer comparison.** The `regime_buffer_pct` must compare against the LIVE incumbent regime's gate, not a stale snapshot from regime entry. (GEM Step 1 bug recurrence risk.)
- **L20 — Cross-source index normalisation.** VIX / VIX9D / VIX3M parquets come from different yfinance series with potentially different time-of-day stamping. Normalise all indexes to date-only before merge (GEM Step 7 / C11 bug recurrence risk).
- **L24 — Noise robustness is 5th axis.** Per-cell `run_noise_robustness` fed to `DecisionInputs`. A cell that passes 4 axes but fails noise is CONDITIONAL_WATCHPOINT, not DEPLOY.
- **A3 — Cost model honesty.** VIXY is a US ETF — use `COST_US_ETF_LIQUID` (≈ 1.5 bps/turnover) plus the IBKR commission floor per L23. Document the rates in the result log.
- **A4 — WFO honesty.** This directive committed BEFORE any data examined. Rolling WFO folds; per-fold parameter selection is not applicable (cells are pre-registered). The bootstrap CI on stitched OOS is the deployment gate.
- **A5 / V3.1 — DSR for N=7.** Apply deflation at the actual N=7 trial count with empirical skew/kurt.
- **V3.2 — Plateau, not peak.** Cell selection rewards a robust neighbourhood, not a single-cell maximum.
- **A6 — MC bootstraps underlyings, not strategy returns.** Use `run_block_mc(primary_close=VIXY, extra_series={...})` per existing framework pattern.
- **A7 — Position-scaling overlays.** The `target_short_weight` is a static notional, NOT a vol-target overlay. If a vol-target version is added in a later cell, it must be backtested through the cost-aware engine, not by post-hoc multiplying strategy returns.
- **Specific to this strategy:** VIXY's structural decay in contango is what generates the edge BUT also caps the maximum harvest rate — the strategy cannot earn more than the absolute roll yield per unit time. Expected Sharpe range from the literature: 0.5 – 1.2 (pre-cost). A reported Sharpe materially above 1.5 should be treated with suspicion (Specifically: cost-omission, or a lucky regime sample). Stress MC is the cross-check.

## §6. Implementation plan

1. **Download VIXY data:** `uv run python scripts/download_data_yfinance.py --symbols VIXY --start 2011-01-04` → `data/VIXY_D.parquet`. Commit the download timestamp in the result log.
2. **Build the strategy function** in `research/vrp/vrp_strategy.py` — pure function `vrp_returns(closes, *, cfg, vix, vix9d, vix3m, **cost_kwargs) -> pd.Series`. Mirror the structure of `research/gem/gem_strategy.py`.
3. **Build the audit harness** in `research/vrp/run_vrp_audit.py` — uses `titan.research.framework.*` primitives end-to-end; mirrors `research/gem/run_gem_audit.py` including per-cell noise gate (L24).
4. **Tests** in `tests/test_vrp.py`:
   - Causality assertion (corrupt-the-future at random t).
   - Contango logic (ratio_long ≥ 1.05 with buffer entered → short VIXY at `target_short_weight`).
   - Backwardation logic (ratio_long < 0.98 → flat VIXY).
   - Regime buffer (no flip when delta < buffer).
   - Class-default consistency (`defaults_for(StrategyClass.DAILY_MEAN_REVERSION)` returns the row the audit relies on).
5. **Run the audit, append §4 result log.**
6. **If verdict is DEPLOY or CONDITIONAL_WATCHPOINT (with non-noise axis as the failing one):** port to `titan/strategies/vrp/` per `directives/Strategy Deployment Guide.md`. Live parity test required (A10). MES-equivalent or direct VIXY trading via IBKR — vehicle decision deferred to deployment PR.

After E1 lands, the next backlog step is **D2 — Commodity futures carry [NEW]** (5d, needs 24-commodity data acquisition first).
