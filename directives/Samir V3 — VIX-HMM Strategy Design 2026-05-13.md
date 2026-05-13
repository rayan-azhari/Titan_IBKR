# Samir-Stack V3 — VIX-HMM Risk-Regime Strategy

**Created:** 2026-05-13
**Status:** Design document. Research-phase. No live deployment until validated.
**Operator brief:** "Rethink Samir's strategy using MES futures, go back to basics
of risk regime classification, layer on top (e.g. HMM on VIX not price), build
correctly with the right math. Bonds come later if needed."

---

## 0. Why v3 (and not "fix v2 again")

V1 (2026-05-02) and V2 (2026-05-13) progressively layered improvements onto a
6-indicator equal-weight regime score plus capitulation overlay plus tier
ladder plus DD breaker plus optional vol target. The audit (2026-05-12) found
material bugs at every layer and the remediation closed them — but the
*resulting* strategy is essentially V1 with a different sleeve.

V3 asks a different question: **if we strip back to first principles, what is
the simplest possible Samir-aligned strategy that delivers documented edge?**

The hypothesis: a single well-specified risk-regime classifier (HMM on VIX)
plus pure equity (MES futures) plus regime-gated leverage may be enough. If
not, we ADD layers in pre-registered order — bonds, then capitulation, then
momentum — each one earning its keep through bootstrap-CI evidence.

---

## 1. Conceptual foundation (Samir Varma's framework)

The three pillars from Samir's *Don't Predict, Classify Risk*:

1. **Risk regimes are the alpha, not direction.** Alpha decays under
   competition; catastrophic systemic risk does not, because it stems from
   forced liquidations rather than predictable price patterns.
2. **Binary classification dominates Texas hedges.** Decide *deploy* or *not*.
   Avoid the middle ground where you're half-in and exposed to both upside
   loss and downside damage.
3. **Best-days cluster with worst-days.** Stepping aside during high-vol
   regimes loses some upside but avoids both tails. Variance compounds
   against you; the geometric mean wins by avoiding the high-dispersion
   regime, not by timing direction inside it.

V3 honours this strictly. The strategy classifies a binary regime, deploys
maximum-cheap-leverage when benign (MES futures), holds cash when hostile.
No "kinda-deployed" partial sizing in v1; that's a v2 layer if needed.

---

## 2. Why HMM on VIX specifically (not on price)

The audit's `hmm_risk` module fits on price-derived dispersion features
(realised vol, |return|, NTR, ATR, 5-day drawdown). Those are *backward*-
looking measures of realised price volatility.

**VIX is fundamentally different.** It is the option-market's *implied*,
*forward-looking*, 30-day SPX volatility. Three properties make VIX the right
target for risk-regime classification:

1. **Forward-looking.** Captures the market's *expectation* of next-30-day
   risk, not yesterday's realised. Aligned with Samir's "what's the regime
   right now" question, not "what was it last month".
2. **Direction-free.** VIX is not contaminated by price level. A 2% SPY day
   can be a calm-regime day or a panic day depending on whether it surprised
   the option market; VIX captures the surprise.
3. **Bimodal-ish empirical distribution.** Historical VIX clusters around
   12-25 (calm regime) with periodic excursions to 30-80+ (stress regime),
   making it natural for a 2-state HMM to converge meaningfully.

**Drawbacks acknowledged.** VIX has its own biases: term-structure premia,
roll effects on VIX futures (but we use the spot VIX index, not futures), and
liquidity-thin regimes during 2020-Q1 etc. We address these with feature
engineering choices below.

---

## 3. Architecture (layered)

```
                  ┌─────────────────────────────────────────────────┐
                  │  LAYER 1 — VIX-HMM regime classifier (this PR)  │
                  │  features: log(VIX), causal rolling 2-state fit │
                  │  output:   regime_score ∈ [0, 1]                │
                  └─────────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────────┐
                  │  LAYER 2 — Position rule (this PR, simple)      │
                  │  if score > τ: deploy MES at L_target           │
                  │  else:         cash                              │
                  │  L_target ∈ {2, 3, 4} swept                     │
                  └─────────────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────────┐
                  │  LAYER 3 — Validation harness (this PR)         │
                  │  - Anchored WFO + bootstrap Sharpe CI           │
                  │  - 12-month sanctuary holdout                   │
                  │  - Underlying-resampled MC for tail risk        │
                  │  - Compare to baselines (see §5)                │
                  └─────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌───────────────────────────────────────────────────────────┐
            │  IF layer-1+2 passes, add layers in pre-registered order: │
            │    A. Momentum confirmation overlay (12-1 mom + VIX)      │
            │    B. Tier ladder (multi-state HMM or threshold ladder)   │
            │    C. Bond sleeve (only if portfolio Calmar improves)     │
            │    D. Capitulation re-entry overlay                       │
            │    E. DD circuit breaker                                  │
            │  Each layer earns its keep through CI-bounded edge.       │
            └───────────────────────────────────────────────────────────┘
```

Each layer is its own PR with its own gate. The remediation plan's research-
math discipline applies throughout (Deflated Sharpe, anchored OOS, no
look-ahead, sanctuary respect).

---

## 4. Layer 1 specification — the VIX-HMM classifier

### 4.1 Features

**Primary feature:** ``log(VIX)``. Log-transform stabilises the right-tail
heaviness of VIX (which clusters tight around 15 and spikes to 80+).

**Optional secondary features** (for ablation, not baseline):
- ``Δ log(VIX) 1-day`` — captures shock dynamics.
- ``log(VIX) 21-day mean`` — trend.

**Baseline runs only ``log(VIX)``.** If the baseline doesn't work, additional
features won't rescue it.

### 4.2 Model

- **2-state Gaussian HMM** via `hmmlearn`.
- States labelled by mean VIX (state 0 = low-VIX = benign;
  state 1 = high-VIX = hostile).
- Diagonal covariance (since we have one feature initially).
- Random seed 42 for reproducibility.

### 4.3 Causal rolling fit (NO look-ahead)

This is the most operationally-critical piece. The HMM must be trained only
on data available at each evaluation bar.

- **Warmup:** 504 trading days (~2 years).
- **Refit cadence:** every 252 days (annual).
- At each refit point `t`:
  1. Fit HMM on `[t - history, t]` (expanding window from warmup-start).
  2. Predict states for `[t, t + 252]` via Viterbi or filtering (we use
     filtering — the conditional posterior given data through that bar).
  3. The state labelling (which state is "low-VIX") is determined by mean
     log(VIX) on the IS slice — fixed once per refit.
- Continuity across refits: the new model's state labels may flip; we
  re-anchor by mean VIX per refit window.

The output `regime_score(t)` is the **filtering probability** of being in the
low-VIX state at bar `t`, given data through `t`. Real-time-compatible.

### 4.4 Gate / threshold

The strategy deploys when `regime_score(t) > τ`. Initial choice: `τ = 0.5`
(majority probability). We will sweep `τ ∈ {0.4, 0.5, 0.6, 0.7}` as a
sensitivity check, **NOT** to optimise — to confirm the result is not
threshold-dependent.

If `τ = 0.5` doesn't work, we don't search for a better one — the strategy
fails layer 1 and we go back to feature engineering.

### 4.5 Position rule (layer 2 minimal version)

```
if regime_score(t) > 0.5:
    target_notional = L_target × NAV     # full deployment in MES futures
else:
    target_notional = 0                  # all cash
```

`L_target ∈ {2, 3, 4}` swept once (this IS a parameter choice but pre-
registered; we report all three and use bootstrap CI to discriminate).

No partial sizing, no tier ladder, no bonds, no capitulation. Pure binary
deploy-or-cash via VIX-HMM.

---

## 5. Baselines (Layer 1 must dominate at least three)

A regime-gated strategy is interesting only if it beats simpler alternatives.
Layer 1 must clear:

| Baseline | What it tests | Pass requirement |
|---|---|---|
| **SPY buy-and-hold** | Does the strategy add anything over the index? | Strategy MaxDD better by ≥ 20pp |
| **Always-on MES at L=2** | Does the regime gate help, or is leverage alone enough? | Strategy Sharpe ≥ always-on Sharpe AND Calmar materially better |
| **VIX z-score threshold gate** | Does the HMM beat a 1-line rule? (z-score > +1 → cash) | Sharpe within +0.05 OR Calmar materially better |
| **Samir v2 6-indicator gate** | Does HMM-on-VIX beat the current ensemble? | Calmar CI lo ≥ v2's |

Failing any of these is a real result — it tells us the HMM isn't pulling its
weight and we should keep v2.

---

## 6. Math discipline (research-math guardrails)

All applicable:

- Bars labelled `D` (daily); annualisation factor `BARS_PER_YEAR["D"]`.
- Sharpe via `titan.research.metrics.sharpe`, with explicit `periods_per_year`.
- Signal/return shift: position at `t-1` earns return at `t`.
- Rolling normalisation: `titan.research.metrics.rolling_zscore` for any z-
  score features (we don't have any in baseline — log VIX is used directly).
- Bootstrap Sharpe CI via `bootstrap_sharpe_ci` (n_resamples=2000).
- WFO uses anchored expanding window (IS 504, OOS 252, step 252).
- Sanctuary: last 12 months untouched until final validation.
- Underlying-resampled MC: bootstrap of daily returns (not prices), shared
  block indices to preserve cross-asset correlation, cumprod to rebuild paths.

Per the operator instruction "the right math": every claim in the layer-1
report must trace to a specific function call with stated `periods_per_year`,
stated lag convention, and a bootstrap CI on the headline metric.

---

## 7. Failure paths

If layer 1 fails (HMM doesn't classify regimes informatively, or strategy
doesn't beat baselines), the diagnostic order is:

1. Confirm HMM converges meaningfully (Viterbi vs simple VIX threshold should
   roughly agree on extreme regimes).
2. Check forward-vol discrimination: forward 21-day SPY vol should be
   materially higher in HMM "hostile" state vs "benign" state. If not, HMM
   isn't classifying real regimes.
3. Check feature space: maybe log(VIX) alone is too noisy; try
   ``log(VIX) z-scored over 252d`` to normalise across rate regimes.

Only after these diagnostics, consider abandoning the HMM-on-VIX hypothesis
and trying alternatives (term structure, VVIX, realised-vol HMM, etc.).

---

## 8. What's deliberately NOT in v3 layer 1

| Excluded | Why | Available later |
|---|---|---|
| Bond sleeve (IGLT/IEF) | "Bonds come later if needed" per operator brief | Layer 4+ |
| Capitulation overlay | Adds complexity without first proving the regime gate works | Layer 4+ |
| DD circuit breaker | Same — failsafe, not primary edge | Layer 4+ |
| Vol-targeting | Phase 5 found redundant with regime gate; doesn't earn keep | Optional layer 5+ |
| Momentum (12-1) | Could be a confirmation overlay; layer 3+ | Layer 3 |
| Tier ladder | Binary deploy/cash is the Samir-correct baseline | Layer 3 |
| Bond rotation | Phase 3 rejected (churn > gate) | Probably never |
| I3 EFA overlay | Audit-rejected (look-ahead) | Dropped permanently |

---

## 9. Concrete deliverables for this PR (Layer 1)

1. `research/samir_v3/__init__.py`
2. `research/samir_v3/vix_hmm.py` — causal rolling HMM on VIX, regime_score series
3. `research/samir_v3/strategy_v3.py` — pure equity binary-gate strategy using MES futures
4. `research/samir_v3/run_v3_validation.py` — head-to-head vs baselines (anchored WFO + bootstrap CI + underlying-resampled MC + sanctuary)
5. `tests/test_samir_v3_vix_hmm.py` — causality test (HMM trained at `t` cannot use bars after `t`); state-labelling test (state 0 has lower mean VIX)
6. `.tmp/reports/samir_v3/layer1_validation_report.md` — results and verdict

The PR title and body explicitly distinguish "v3 layer 1 — research prototype, NOT for live deployment". The current paper-deployed strategy (v2 / MES futures + IGLT) stays untouched until v3 has earned the right to replace it.

---

## 10. Decision rule for "should v3 replace v2?"

V3 layer 1 (or any subsequent layer) becomes the new champion only if:

1. Stitched OOS Sharpe CI lo > 0
2. Beats v2's Calmar CI lo (currently 0.137 for the futures+IGLT cell)
3. Sanctuary Sharpe ≥ 0
4. MC P(MaxDD>50%) < 1%
5. Statistical-significance gap over the four baselines in §5

Less than that: v2 stays deployed, v3 stays as research curiosity until
layers 3-5 raise the bar.

---

## 11. Open questions for the operator (defer if not blocking)

- **Cash yield assumption.** When the strategy is in cash, does it earn
  IBKR cash yield (currently ~4% USD)? This materially affects backtest CAGR
  vs always-on MES baseline. Current default: yes, earn the daily piecewise
  funding rate on cash.
- **Cost model.** Use `FuturesEngine` (Phase 4 validated, basis + roll +
  T-bill). Same numbers as v2's equity sleeve.
- **VIX data.** `^VIX_D.parquet` already exists; coverage 1990-2026. Use full
  history less the 12-month sanctuary.

If these are wrong, flag now and re-derive. Otherwise we proceed.
