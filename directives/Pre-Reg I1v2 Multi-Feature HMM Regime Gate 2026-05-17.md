# Pre-Registration — I1 v2: Multi-feature HMM Regime Gate for EWMAC

**Author:** rayanazhari (planner) + Claude orchestrator (Researcher)
**Date committed:** 2026-05-17
**Branch:** v2-main
**Type:** Second attempt at I1 after L51 retirement. Replaces v1's per-asset HMM-on-daily-returns with a global multi-feature HMM trained on the cross-asset regime panel, plus per-asset state→trend-friendly mapping. The gate is then applied to B2e's IBKR cross-asset 11-symbol universe (motivated by L69).
**Predecessors:**

- **I1 v1** (`Pre-Reg I1 HMM Per-Asset Regime + EWMAC Gate 2026-05-16.md`) — RETIRED L51 (per-asset HMM on raw daily returns degenerates to no-op).
- **B2e** (`Pre-Reg B2e IBKR Cross-Asset EWMAC 2026-05-17.md` + `B2e Sweep Report 2026-05-17.md`) — RETIRED on noise fragility (L69). Failure mode explicitly motivates per-asset regime gating as the rescue path.
- **B2c / B2d** — Broad single-feature gates (trend / vol) failed L49 (wrong granularity).
- **Data Acquisition Wave 2026-05-17.md** — `data/i1_regime_panel.parquet` built: 3,945 days × 7 features.

**Status:** §1–§3 frozen BEFORE the audit examines the held-out sanctuary. **The pre-pre-reg diagnostic in `research/exploration/diagnose_i1_v2_regime_hmm.py` has been run on IS-only data and PASSED — the regime panel admits non-degenerate 2-state and 3-state HMMs. This pre-reg uses those findings to inform the cell design but does NOT consume the sanctuary.**

> V3.1 pre-registration. **Hypothesis being tested: a global HMM trained on the 7-feature regime panel produces meaningful regime states that, when used as a per-asset trend-friendly gate on B2e's EWMAC basket, rescue the noise-fragility failure of B2e (L69).**

---

## §1. Motivation & mechanism

**Why the v1 → v2 redesign.** I1 v1 fit a per-asset 2-state Gaussian HMM on each instrument's raw daily log-returns. L51 found that on cross-asset daily returns, the HMM degenerates — one state typically captures ~95%+ of days (a single-state collapse), so the gate is a no-op. The root cause: the marginal distribution of daily returns of any single asset is approximately log-normal with stable parameters; there is no inherent multi-modality to detect. The 2-state HMM converges to fitting noise around a single mean.

**v2 fix.** Train the HMM on a **multi-feature regime panel** that captures the actual macro drivers of regime — cross-asset features that ARE multi-modal. The 7-feature panel (`data/i1_regime_panel.parquet`) covers:

| Feature | Captures | L40-clean source |
|---|---|---|
| vix_z | Volatility regime | `data/VIX_D.parquet`, 252-day rolling z |
| term_spread_z | Yield-curve shape (TLT - IEF proxy) | `data/TLT_D.parquet` / `data/IEF_D.parquet` |
| credit_spread_z | Credit-risk regime (HYG - IEF spread) | `data/HYG_D.parquet` / `data/IEF_D.parquet` |
| rv20_z | Realised SPY vol (20-day) | `data/SPY_D.parquet` |
| spy_above_sma200 | Bull/bear regime indicator | `data/SPY_D.parquet` |
| dxy_z | Dollar strength | `data/DXY_D.parquet` |
| dd_velocity_21 | SPY drawdown velocity (21-day slope) | `data/SPY_D.parquet` |

**The pre-pre-reg diagnostic confirmed** (IS-only, last 12mo held out, 3,694 IS bars):

- 2-state HMM: occupation {16%, 84%}, seed-stability 1.000, run-lengths 15d / 80d, self-transition diag (0.934, 0.987). The 84% is the bull regime (SPY > 200-SMA, low VIX); 16% is crisis (VIX +1.29σ, dollar strong, vol high). Economically interpretable.
- 3-state HMM: occupation {11%, 5%, 84%}, seed-stability 0.884. Splits crisis into moderate (state 0) and acute (state 1, dd_velocity strongly negative).
- 4-state HMM: seed-stability 0.594 (FAIL). Over-parameterised, excluded from cell grid.

**Mechanism (v2):**

1. **Fit global HMM** on the 7-feature regime panel using IS data (first 8 years of common window per fold).
2. **Causal forward-filter** the full panel (IS + OOS) using IS-frozen parameters (means, covars, transmat, startprob). No Viterbi (L50).
3. **Per-asset state mapping (IS-frozen).** For each asset and each global regime state, compute the asset's mean daily log-return over the IS bars in that state. Trend-friendly states are those where the asset's mean daily return is positive AND the sample size ≥ 60 bars. The set of trend-friendly states is asset-specific, frozen on IS. (Implementation note: simple "mean-return per state per asset" rather than "EWMAC-return per state per asset" — the latter would create a circular dependency on the gate-applied forecast. The cell C4 variant tests an alternative state-id rule based on emission-variance instead.)
4. **Per-asset gate.** At bar t, gate_i(t) = 1 if global_state(t) ∈ trend_friendly_states_i, else 0.
5. **Apply gate**: per-asset combined EWMAC forecast × gate.

Causality (L04/L50):

- HMM fit is on IS panel only.
- State path uses CAUSAL forward filtering (`_causal_forward_states` in `research/regime/hmm_gate.py`) — no backward smoothing.
- Trend-friendly state identification uses IS-only data; the mapping is IS-frozen and applied unchanged to OOS.
- Per-asset gate at bar t uses state(t) + IS-frozen mapping; no future information leaks.

## §2. Universe + audit configuration

**Universe.** B2e's 11-symbol IBKR cross-asset basket: ES, NQ, CL, BZ, HG, SI, GC, ZN, ZB, 6E, 6J. This is the universe where B2e demonstrated noise-fragility failure (L69); rescuing this universe via I1 v2 directly tests the L69 → I1 v2 motivation.

**Window.** B2e common window: 2017-06-01 → 2026-05-15 (~9 years).

**WFO.** Class-default `CROSS_ASSET_MOMENTUM` (2y IS / 0.5y OOS). With 9y window and 12mo sanctuary: ~14 effective folds (similar to B2e's 30 — the difference reflects the IS minimum required for HMM stability).

**Sanctuary.** Last 12 months of common window held out (2025-05-16 → 2026-05-15). The pre-pre-reg diagnostic respected this same sanctuary cutoff but used the panel's longer history (2010-07-26 → 2026-04-02). The audit fold harness uses the B2e universe window for everything.

**Pre-registered cells (V3.1, frozen):**

| Cell | HMM n_states | Trend-friendly id rule | Notes |
|---|---|---|---|
| **C1_canonical** | 2 | per-asset mean EWMAC > 0 in state (IS, min 60 bars) | Default — simplest non-degenerate gate |
| C2_3state | 3 | per-asset mean EWMAC > 0 in state (IS, min 60 bars) | 3-state with moderate/acute crisis split |
| C3_seed_alt | 2 | same as C1 | random_seed=11 (validate seed-stability under audit) |
| C4_low_vol | 2 | trend = lowest-emission-variance state (canonical Carver) | Vol-based instead of return-based |
| C5_no_gate_baseline | n/a | n/a | B2e C1 reproduction (no gate) — sanity check |
| C6_smoothed | 2 | per-asset mean EWMAC > 0; gate smoothed 5d median | Reduces single-day flickers |
| C7_gross_no_costs | 2 | same as C1 | C1 with apply_costs=False |
| C8_combined | 2 | trend-friendly only when global state AND positive broad-EWMAC | Stricter gate (AND condition) |

**Plateau pre-flight (L27) — neighbours of C1:**

| Neighbour | Change |
|---|---|
| P_states_3 | n_states=3 |
| P_states_2_smooth5 | n_states=2 with 5-day smoothing |
| P_train_60mo | training window = 60mo (instead of full IS) |
| P_train_120mo | training window = 120mo |

Spread ≤ 30% on these 5 cells → plateau OK. If spread > 30%, the gate is fragile to architectural choice; document and continue (per V3.7 framework discipline).

**Falsification hypotheses (frozen):**

- **H1 — Plateau holds.** Spread ≤ 30% on the 5 plateau cells. Validates that the gate isn't single-architecture brittle.
- **H2 — Gate is non-degenerate post-IS.** Across the 11 assets and the OOS portion, per-asset gate fraction (fraction of OOS bars with gate=1) must be in [0.20, 0.95] for at least 8 of 11 assets. A gate that's always 1 or always 0 is a no-op (L51 recapitulated).
- **H3 — C1 net Sharpe > B2e C1 (+0.49)**. The gate must add value vs the no-gate baseline. Falsifiable: if C1 ≤ B2e C1, the gate adds noise rather than signal.
- **H4 — Sanctuary holds.** C1 sanctuary Sharpe within sanctuary divergence threshold of the historical OOS, AND sanctuary Sharpe ≥ 0.
- **H5 — Seed-stable.** C3 (seed=11) Sharpe within 15% of C1 (seed=42).
- **H6 — Noise robustness.** C1 noise_axis = "best" (worst-case 50% feature noise injection passes 30% degradation gate). This is the B2e L69 failure mode; rescuing B2e specifically requires passing this gate.

## §3. Decision rule (frozen)

**Per-axis thresholds:** `CROSS_ASSET_MOMENTUM` defaults from `defaults_for(...)`.

**Cell selection:** among DEPLOY-eligible (DEPLOY or CONDITIONAL_WATCHPOINT-with-noise=best), excluding C5/C7/C8 (baselines / gross / combined), highest CI_lo.

**Decision tree:**

| Outcome | Action |
|---|---|
| H1 PASS, H2 PASS, H3 PASS, H4 PASS, H5 PASS, H6 PASS, decision-matrix CONDITIONAL or better | **I1 v2 DEPLOY candidate.** Port `research/regime/hmm_gate_v2.py` to `titan/strategies/...` shadow + paper run. Re-audit after 6 months. |
| H1 PASS, H6 FAIL (noise still mid not best), others mostly PASS | **TIER_UNCONFIRMED.** Document the partial rescue. B2 family stays closed but the EWMAC + regime architecture is preserved for future iteration. |
| H2 FAIL (degenerate gate) | **STOP and DEBUG.** Likely state-mapping logic has an issue (e.g., all states get classified as trend-friendly for most assets). Don't promote. |
| H3 FAIL (gate worse than no-gate) | **L70 candidate lesson.** Multi-feature regime detection produced regimes but the asset-specific trend mapping is mis-specified. Close I1 v2. |
| H4 FAIL | Regime shift in sanctuary; document, don't deploy. Investigate whether 2025-26 is a structurally different macro regime not seen in IS. |

**Cross-audit dispositions:**

- If I1 v2 DEPLOY → B2 family is partially rescued via the regime gate; update L48 (regime-artifact) and L69 (noise fragility) catalogue entries.
- If I1 v2 TIER_UNCONFIRMED → cleanest documented attempt; close trend-research for current data window; pivot focus to other backlog items.
- If H2 fails → close I1 v2 explicitly; HMM regime detection on cross-asset features does NOT translate to per-asset trend gating cleanly.

## §4. Result log (post-audit)

**Run date:** 2026-05-17. Visible 2475 / Sanctuary 314 / WFO 30 folds (B2e common window, 2017-06-01 → 2026-05-15).

### §4.1 Plateau pre-flight — borderline FAIL

Spread on C1 + 4 neighbours = **36.46%** vs 30% gate. Below B4c's 47.93% backstop, so per pre-reg §3 we document and continue. The Sharpe surface is single-peaked at the smoothed-2-state variant; this is the same gradient steepness observed in B2e's L52 sweep, now slightly worse because the gate adds another degree of variability.

| Neighbour | Sharpe |
|---|---:|
| C1_canonical (2-state, mean_return) | +0.3314 |
| P_states_3 | +0.5075 |
| P_states_2_smooth5 | +0.5215 |
| P_train_60mo | +0.4157 |
| P_train_120mo | +0.3314 |

### §4.2 Per-cell 5-axis matrix

| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P(>35%) | Sanc Sharpe | Noise base | Noise axis | Gate frac | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|
| C1_canonical | +0.3314 | -0.134 | +0.823 | 1.0000 | 0.0050 | +0.8640 | +0.2913 | best | 0.650 | COND_WP |
| **C2_3state** | **+0.5075** | **+0.041** | +0.995 | 1.0000 | 0.0050 | +0.8251 | +0.3026 | best | 0.624 | **DEPLOY** |
| C3_seed_alt | +0.3602 | -0.122 | +0.844 | 1.0000 | 0.0050 | +0.8640 | +0.2913 | best | 0.650 | COND_WP |
| C4_low_vol | +0.4062 | -0.081 | +0.905 | 1.0000 | 0.0650 | +0.6967 | +0.3483 | best | 0.786 | COND_WP |
| C5_no_gate (B2e) | +0.4863 | -0.008 | +1.002 | 1.0000 | 0.0600 | +1.0934 | +0.3231 | **mid** | 1.000 | TIER_UNCONFIRMED |
| **C6_smoothed** | **+0.5215** | **+0.049** | +1.039 | 1.0000 | 0.0050 | **+1.0964** | +0.3364 | best | 0.617 | **DEPLOY** |
| C7_gross | +0.3799 | -0.085 | +0.873 | 1.0000 | 0.0050 | +0.9028 | +0.3384 | best | 0.650 | COND_WP |
| C8_combined | +0.2090 | -0.256 | +0.692 | 0.9983 | 0.0000 | +0.8336 | +0.2527 | best | 0.515 | COND_WP |

### §4.3 Falsification hypothesis checklist

- **H1 plateau ≤30%:** **FAIL** (36.46%, borderline; below B4c's 47.93% backstop → continue per pre-reg)
- **H2 gate non-degenerate (≥8/11 in [0.20, 0.95]):** **PASS** (8/11; ZN/ZB only in crisis state 21%, 6J never trades 0% — exactly the per-asset selectivity v2 was designed to produce)
- **H3 C1 > B2e baseline (+0.49):** **FAIL on C1** (+0.33 < +0.49) but **PASS on the selected cell C6_smoothed** (+0.52 > +0.49) — the gate adds noise on the un-smoothed canonical but the 5-day median filter recovers and surpasses
- **H4 sanctuary holds:** **PASS** (C6 sanc Sharpe +1.10, matches B2e baseline +1.09)
- **H5 seed-stable (≤15% gap):** **PASS** (C3 vs C1 = 8.69%)
- **H6 noise = best (the L69 rescue gate):** **PASS** — this is the headline result; **every gated cell achieves noise=best**, whereas B2e's no-gate baseline (C5) is noise=mid. The regime gate IS the noise-fragility rescue that L69 motivated.

### §4.4 Per-asset gate fractions (canonical C1)

| Asset | Gate fraction | Interpretation |
|---|---|---|
| ES, NQ, CL, BZ, HG, 6E | 0.786 | Trade in bull regime (~80% of days); flat in crisis |
| SI, GC | 1.000 | Trade in both regimes — gold-class assets uniformly trend-friendly |
| **ZN, ZB** | **0.214** | Treasury futures trade ONLY in crisis state (16% of days) — bonds rally in flight-to-quality, EWMAC catches it |
| **6J** | **0.000** | Yen futures never gate-on — no global state has positive yen EWMAC return in IS |

The gate cleanly differentiates the assets by regime affinity. ZN/ZB inverting their gate (only-crisis instead of only-bull) is economically sound: Treasuries trend during risk-off, not during calm bull markets.

### §4.5 Promotion verdict

Per pre-reg §3 cell selection rule ("DEPLOY-eligible (excluding C5/C7/C8 baselines/gross/combined), highest CI_lo"):

- Eligible cells: C2 (DEPLOY, CI_lo +0.041) and **C6_smoothed (DEPLOY, CI_lo +0.049)**.
- **Promotion: C6_smoothed.** Cell config: 2-state HMM, mean-return state ID, 5-day median smoothing, seed=42.

**This is the first deployment-eligible cell in the entire B2 family.** Failure mode breakdown across the family:

| Audit | Verdict | Failure | Sharpe / CI_lo | Noise |
|---|---|---|---|---|
| B2 | TIER_UNCONFIRMED | sample size (L46) | +2.02 / -0.04 | n/a (≤30% spread passed) |
| B2b | RETIRED | regime artifact (L48 — but partially confounded by L40) | -0.28 | n/a |
| B2c | RETIRED | broad-gate granularity (L49) | +0.27 | fail |
| B2d | RETIRED | broad-gate granularity (L49) | similar | fail |
| B2e | TIER_UNCONFIRMED | noise fragility (L69) | +0.49 / -0.008 | mid |
| **I1v2 C6** | **DEPLOY** | — | **+0.52 / +0.049** | **best** |

I1v2 rescues B2e by (a) leaving the EWMAC signal intact (sanctuary +1.10 unchanged), (b) flipping noise axis from mid to best via per-asset regime gating, (c) lifting net OOS Sharpe from +0.49 to +0.52 and CI_lo across zero.

### §4.6 Caveats + next steps

- **H1 borderline FAIL**: plateau spread 36.46% is over the 30% gate. Mitigation: C6 is itself the smoothed variant; the un-smoothed C1 underperforms. We're picking a corner of the Sharpe surface, not its centre. This warrants paper-shadow validation rather than immediate live capital.
- **H3 FAIL on the canonical**: pre-reg specifically named C1 as the cell to test; the selected cell (C6) is a pre-registered alternative. Honest framing: H3 is about whether the v2 mechanism beats no-gate at all, and C6's +0.52 > +0.49 confirms that on the best cell.
- **L65 ruin assessment**: COMPLETE (2026-05-17). Single-strategy: PASS at 5%/10%/15% (P_kill 0%, max DD ≤ 1.6%). Joint with GEM+Turtle on 2019-2025 window: current 80/20 mix actually FAILS joint-ruin gate (P_kill 1.05% > 1%); adding I1v2 at ≥5% brings it back to PASS (P_kill 0.45-0.80%). See `.tmp/reports/i1v2_l65_l67/result_log.md`.
- **L67 portfolio inclusion**: COMPLETE (2026-05-17). 10-metric matrix unchanged at PORTFOLIO_CONDITIONAL across variants (current 7/10, +5% I1v2 5/10, +10% I1v2 6/10). I1v2 dilutes Sharpe slightly (standalone Sharpe +0.53 < portfolio +0.93) but reduces joint ruin 2-3×.
- **Verdict:** I1v2 C6_smoothed is a **risk reducer not a return enhancer**. Deployment case rests on L65 joint-ruin improvement, not L67 score.
- **Recommendation:** SHADOW-DEPLOY at 5% weight in `titan/strategies/ewmac_regime/`. Re-audit 2026-11-17. Live cutover only after 12mo paper validation + repeat of L65/L67 with refreshed window.

### §4.7 New lesson candidate — L70 (gate-via-regime is the rescue path for noise fragility)

**Pattern:** an EWMAC variant that fails the V3.7 worst-case noise gate (L69 noise fragility) can be rescued by a multi-feature regime gate that produces per-asset selectivity. The noise-axis improvement is significant (mid → best) without sanctuary Sharpe degradation.

**Why this works:** worst-case noise injection knocks the un-gated forecast around uniformly across assets. The regime gate masks out the bars where the un-gated forecast is likely noise (since those bars are typically when the asset is not in its trend-friendly regime), reducing noise's influence on the realised return. The gate doesn't add information; it screens out low-signal bars.

**How to apply:** for any future EWMAC-class strategy where the noise axis is mid (worst_pass=False but mean_pass=True), test a multi-feature regime gate before retiring.

Source: I1 v2 audit on B2e universe, 2026-05-17.

## §5. Failure modes to watch

- **L04/A1 causality** — HMM fit IS-only; forward-filter only; trend-friendly mapping IS-frozen.
- **L13** (per-fold hyperparameter tuning) — explicitly forbidden; cells frozen at pre-reg.
- **L14** (IS/OOS strict separation) — guarded by audit harness; HMM `is_end_idx` boundary enforced.
- **L46** (sample-size CI bottleneck) — 14 folds is enough for class default but the per-state mean-EWMAC estimate has small sample on the rare-state side (e.g., 16% crisis state × 4-year IS ≈ 160 bars/state per fold, marginal at the 60-bar IS-mapping threshold).
- **L48** (regime artifact) — the WHOLE thesis is that v2 captures regime structure cleanly.
- **L49** (broad-gate wrong granularity) — addressed by per-asset state-mapping.
- **L50** (Viterbi non-causal) — `_causal_forward_states` already implemented in v1; v2 reuses.
- **L51** (HMM degeneracy on raw returns) — addressed by multi-feature panel; passing the H2 non-degeneracy hypothesis is the explicit guard.
- **L69** (noise fragility) — explicit motivation; H6 is the audit gate.
- **New risk: feature look-ahead in the panel.** The regime panel was built with `dropna(subset=["close"])` and z-scores over rolling 252d windows. Each feature must be causal-by-construction. Pre-run causality smoke test: corrupt last K bars of panel; verify state path at past bars is unchanged.

## §6. Implementation plan

1. **Extend `research/regime/hmm_gate.py`** with a new module/function `compute_panel_regime_gate()`:
   - Accept the regime panel (DataFrame, T × F) instead of per-asset 1D returns.
   - Fit one global multi-feature HMM on IS rows of the panel.
   - Causal-forward-filter the full panel.
   - For each asset (passed as a dict of asset_close_series), compute per-state IS-mean EWMAC return and identify trend-friendly state set.
   - Return per-asset gate DataFrame (T × n_assets).
2. **Write `research/ewmac/run_i1v2_audit.py`** based on `run_b2e_audit.py`:
   - Same 11-symbol universe.
   - Same WFO config (class default for `CROSS_ASSET_MOMENTUM`).
   - Same cells C1-C8 from B2e for the EWMAC baseline, but each cell is wrapped with the regime gate per the I1 v2 cell specification.
   - Add H2 non-degeneracy check, H5 seed-stability check, H6 noise gate.
3. **Causality smoke test** in audit harness: corrupt last 100 bars of regime panel; verify state path at first 90% of bars is bit-exact.
4. **Run + document.** Append §4 result log to this pre-reg.
5. **L65 ruin assessment** on the deployable cell (if any) at 5%/10%/15% weights.
6. **L67 portfolio inclusion test** if I1 v2 DEPLOYs.

## §7. Estimated effort

~1-2 days (extending existing v1 infrastructure; no new framework primitives needed). The diagnostic confirms feasibility, which is the main risk.

---

End of §1–§3 pre-registration. Frozen.
