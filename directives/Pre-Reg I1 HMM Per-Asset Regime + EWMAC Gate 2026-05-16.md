# Pre-Registration — I1: HMM Per-Asset Regime Detection + EWMAC Gate

**Author:** rayanazhari (planner) + Claude orchestrator (Researcher)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** Per-asset regime detection escalation after L49 (broad-index regime gates fail). Tests whether a per-asset HMM regime model rescues B2 EWMAC where universe-wide gates couldn't.
**Predecessors:**

- B2 IBKR-3y commodity (Sharpe +2.02 — regime-artifact per L48)
- B2b yfinance-21y cross-asset (Sharpe -0.28 — universal-trend falsified)
- B2c trend-of-trend gate (no rescue; L49 — regime is per-asset, broad-trend wrong)
- B2d realised-vol gate (no rescue; L49 generalized — broad-vol also wrong granularity)

**Status:** §1–§3 frozen at commit BEFORE the audit runs. §4 result log appended post-run.

> **V3.1 honesty + L13/L14 carry-over from prior ML work**: HMM is high-risk for overfit. Pre-registered cells, frozen on IS folds, no per-fold parameter tuning. Cross-fold sanity checks mandatory.

---

## §1. Motivation & mechanism

**Why HMM (and why per-asset).** B2c+B2d empirically demonstrated that broad-index regime decompositions (trend AND vol) do NOT rescue B2 on cross-asset 21y data (L49 generalized). The B2c `directional` mode flipped per-asset signal signs based on broad regime and produced +0.27 Sharpe — but failed noise/CI gates. The crucial diagnostic was that the directional flip helped at all: it implies per-asset trend signs are systematically WRONG during certain broad regimes, but the broad regime gating granularity is too coarse — per-asset regime structure varies across the universe.

**HMM hypothesis.** A 2-state Hidden Markov Model trained on each instrument's daily log-returns will identify "trend-friendly" (state A) and "trend-unfriendly" (state B) regimes PER ASSET. Gating per-asset EWMAC by per-asset HMM state should:

1. Activate the EWMAC signal only when the asset is in its trend-friendly state
2. Avoid the per-asset noise of the broad-index gates from L49
3. Capture asynchronous regime shifts across the universe (e.g., FX in trend while bonds in mean-reversion)

**Mechanism (per asset i, per bar t):**

1. Fit a 2-state Gaussian HMM on rolling-window log-returns of asset i (IS-frozen — fitted once on the IS portion of each WFO fold, then APPLIED to OOS without re-fitting).
2. Compute Viterbi-decoded state path on full series (IS + OOS).
3. Identify which state is "trend-friendly" based on IN-SAMPLE return-autocorrelation (state with higher positive autocorr → trend-friendly). Frozen on IS, NOT re-derived on OOS.
4. Gate = 1 when asset is in trend-friendly state, 0 otherwise.
5. Per-asset combined EWMAC forecast multiplied by per-asset gate.

Causality: HMM training uses only IS data (frozen); OOS predictions are Viterbi-decoded using IS-frozen transition matrices and emission parameters. State labels are IS-frozen. Per-asset, no cross-asset information leaks.

## §2. Universe + audit configurations

**Universe:** same 31-instrument yf_b2b (B2b/B2c/B2d compat).
**Window:** 2005-01-03 → 2026-05-16.
**WFO:** `CROSS_ASSET_MOMENTUM` defaults (2y IS / 0.5y OOS / 8 folds → ~60 effective folds).

**Pre-registered cells (V3.1):**

| Cell                  | HMM training window | State identification              | Notes                          |
|-----------------------|---------------------|-----------------------------------|--------------------------------|
| **C1_canonical**      | full IS window      | autocorr-based (higher autocorr = trend)  | Default                      |
| C2_short_train        | last 1y of IS        | autocorr-based                    | Shorter training window         |
| C3_long_train         | 5y rolling           | autocorr-based                    | 5y context                      |
| C4_vol_based_id       | full IS              | lower-vol state = trend-friendly  | Vol-based state identification  |
| C5_no_gate_baseline   | n/a                  | n/a                               | B2b reproduction (filter off)   |
| C6_combined           | full IS              | autocorr + B2c broad-trend AND    | HMM + broad-trend gate combined |
| C7_gross_no_costs     | full IS              | autocorr-based                    | C1 with apply_costs=False       |
| C8_3_state            | full IS              | 3-state HMM, middle = trend       | Three-regime variant            |

**Plateau pre-flight neighbours of C1:**

| Neighbour            | Change                                |
|----------------------|---------------------------------------|
| P_train_2y           | training window=2y (vs full IS)       |
| P_train_3y           | training window=3y                    |
| P_seed_alt           | random seed=11 (HMM init sensitivity) |
| P_smoothing_5d       | post-Viterbi 5-day smoothing of state |

**Falsification hypotheses (V3.1):**

- **I1 H1 (plateau):** spread ≤ 30%. Per-asset gate should be less degenerate than B2c.
- **I1 H2 (Sharpe ≥ +0.30):** materially positive.
- **I1 H3 (per-asset > broad):** C1 Sharpe > max(B2c C1 = -0.28, B2d C1 = -0.22). If per-asset granularity is correct, should outperform broad gates.
- **I1 H4 (HMM not overfit):** OOS Sharpe ≥ 50% of IS Sharpe (small gap = honest model; large gap = HMM overfit). Computed per cell.
- **I1 H5 (seed-stability):** P_seed_alt Sharpe within 10% of C1 Sharpe (HMM init shouldn't drive result; if it does, model is unstable and conclusions are seed-dependent).

## §3. Decision rule (V3.1)

Per `CROSS_ASSET_MOMENTUM` defaults. Bootstrap-CI gate mandatory. Cell selection: DEPLOY-eligible (excl C5/C7/baselines), highest CI_lo.

**HMM-specific gates (L13/L14 carry-over from prior ML work):**

- **No per-fold parameter tuning** — HMM hyperparameters (state count, training window, init seed) are pre-committed per cell.
- **IS/OOS strict separation** — HMM fit on IS only; Viterbi decode on full series uses IS-frozen parameters; state-label identification (which state = trend) is IS-frozen.
- **Seed stability check** mandatory (H5). If HMM init sensitivity is high, audit is informational only.
- **State-distribution sanity check** post-run: report what fraction of OOS bars each state is active per asset. If a single state dominates >90% on most assets, the HMM degenerated to a no-op.

Sanctuary 12 months. Plateau pre-flight L27 ≤ 30%.

## §4. Result log (post-audit)

*To be filled.*

---

## §5. Failure modes

- L04/A1 causality: HMM fit on IS frozen; Viterbi decode forward-only with frozen params.
- L13 (per-fold parameter discipline): no HMM hyperparameter tuning per fold.
- L14 (IS/OOS separation): no information from OOS bars leaks into HMM fit or state labeling.
- L24/J3 noise: standard noise gate, 5-axis matrix.
- L27 plateau: standard gate.
- L46 sample-size: 60 folds available.
- L48 (regime-artifact): the underlying problem.
- L49 (broad-index wrong granularity): the thing this audit's per-asset granularity addresses.
- **A new risk: HMM overfit on the per-asset training window.** Mitigated by H4 (OOS/IS Sharpe ratio gate) + seed stability check (H5).

## §6. Implementation plan

1. **Add `hmmlearn` to `pyproject.toml`** if not present (likely needs install).
2. **Build `research/regime/hmm_gate.py`** — `HMMRegimeConfig` dataclass + `fit_per_asset_hmm()` + `compute_per_asset_gate()`. IS-frozen API: separate fit + apply phases.
3. **Extend `EwmacConfig`** with optional `per_asset_regime_gate: PerAssetRegimeGateConfig`.
4. **Wire into `compute_ewmac_forecast`** — per-asset gate multiplication when set.
5. **Tests** in `tests/test_hmm_gate.py`:
   - Synthetic regime-switching universe → HMM identifies states correctly.
   - Causality: OOS Viterbi uses only IS-frozen params.
   - Seed sensitivity within tolerance.
   - State degeneracy check (no single state dominates >95% on synthetic regime-switching data).
6. **Write `research/ewmac/run_i1_audit.py`** mirroring B2c/B2d structure.
7. **Run + document.** If C1 promotes → port. If still fails → close out the "rescue B2" line of research; document as deepest investigation of the regime-artifact L48 problem.

## §7. Estimated effort

~3 days (Carver's HMM-on-trend work, plus L13/L14 discipline, plus seed-stability validation). Higher methodological risk than B2c/B2d due to HMM overfit potential — pre-reg gates are the safeguard.

---

**This pre-reg is committed but the audit is NOT YET RUN.** Awaiting user authorization to proceed (HMM is a meaningful effort + cost vs the B2c/B2d simpler attempts).
