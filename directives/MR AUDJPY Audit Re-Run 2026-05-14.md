# MR AUD/JPY Audit Re-Run — anchor=6 vs anchor=24 comparison

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect / Risk-Auditor
**Status:** **PRE-REGISTRATION** — committed before the comparative WFO runs (V3.1).
**Parent:** `directives/Strategy Re-validation 2026-05-13.md` §1.1
**Companion:** `directives/IC AUDJPY Vwap Fine-Grid 2026-05-13.md` §4.5 recommended action 1

---

## 0. Why this exists

The AUD/JPY vwap-anchor fine-grid IC scan showed:

- IC at horizon=1 is **monotonically decaying** in anchor across `{3, 4, 6, 8, 12}`.
- Live `mr_audjpy` runs at `vwap_anchor = 24` — **on the long tail of the IC decay** where t_NW = -2.71.
- IC peak among interior cells is at anchor = 4 or 6 with |t_NW| ≈ 7.7.

The +0.97 / CI_lo +0.47 audit-corrected backtest result that motivates the live deployment was produced at anchor=24. The fine-grid IC analysis suggests the strategy's mechanical basis (the vwap deviation signal) is stronger at anchor=6. **But the strategy is more than the raw signal** — it includes tiered entry, regime gating, vol-target sizing, session windows, percentile thresholds. So the IC ranking of anchor cells does NOT automatically translate to backtest Sharpe ranking. The empirical question this directive answers: under the EXACT same strategy class with EVERYTHING ELSE held fixed, does anchor=6 produce a better risk-adjusted backtest than anchor=24?

---

## 1. Pre-registered scope

A single 2-cell comparison. NO sweep, NO grid search, NO additional cells (V3.1).

| Cell | vwap_anchor | All other strategy config |
|---|---:|---|
| **A** (live) | 24 | Identical to `config/mr_audjpy.toml` — tiers, regime filter, sizing, session window |
| **B** (IC-peak) | 6 | Identical to A except `anchor_period=6` |

Anchor=6 is chosen (not anchor=3) because the IC range across `{3,4,6,8,12}` is 0.0265 → 0.0156 with the peak at 3-4 (essentially equivalent). Anchor=6 is the SHORTEST interior cell of the original 3-cell grid `{6, 24, 96}` — the parent census's pre-registration. Using anchor=6 keeps this comparison strictly inside the V3.1 parent-pre-registered grid; anchor=3 would be retroactively expanded scope. Pre-committed.

### 1.1 Gates

Inherits from Phase 0 strategy spec:

- WFO via `research/mean_reversion/run_confluence_regime_wfo.py::run_mr_wfo` (existing audit-corrected runner, IS=30k bars / OOS=7.5k bars).
- Shared-metrics Sharpe via `titan.research.metrics.sharpe(..., periods_per_year=BARS_PER_YEAR["H1"])`.
- Bootstrap CI via `bootstrap_sharpe_ci`.
- DSR adjustment at N=2 (small N, but still applied; null-max ≈ sqrt(2 ln 2) ≈ 1.18).
- Sanctuary: last 12 months held out via `is_bars + oos_bars` walk-forward boundaries (the existing runner's design already respects no-look-ahead).
- **No underlying-resampled MC required** for this comparison — both cells share the same underlying so MC's relative ranking would be the same as the WFO's. MC IS recommended once a winner is chosen and is moving to live; that's a separate step.

### 1.2 Pre-committed decision rule

After running both cells:

| Comparative outcome | Recommended action |
|---|---|
| Cell B (anchor=6) stitched CI_lo > Cell A (anchor=24) stitched CI_lo **by ≥ 0.20** AND Cell B CI_lo > 0 | **Reconfigure live to anchor=6.** Update `config/mr_audjpy.toml`. Update Strategy Guide. Live parity test BEFORE rollout. |
| Cell B CI_lo within ±0.20 of Cell A CI_lo, both > 0 | **Status quo.** Anchor difference is not material; the live config stands. Document the comparison and stop. |
| Cell A CI_lo ≤ 0 AND Cell B CI_lo ≤ 0 | **Retire `mr_audjpy` from live deployment.** The strategy's mechanical basis is too thin under audit-corrected math. Issue a separate config-change PR. |
| Cell A CI_lo > 0 AND Cell B CI_lo ≤ 0 | Anchor=6 is worse on the engine despite stronger IC. **Strange but possible** -- means the strategy's layers (regime gate, percentile tiers) interact with anchor in unexpected ways. Document; status quo with a watchpoint. |

The thresholds (0.20 CI_lo gap, the four scenarios) are pre-committed here; not tuned post-hoc.

---

## 2. Out of scope

- **Not** running anchor=3, 4, 8, 12. Those are out of the parent census's `{6, 24, 96}` pre-reg.
- **Not** sweeping any other parameter (tier_pcts, tier_sizes, vol_target, regime filter, session window, etc.). Each would be a new pre-registration.
- **Not** evaluating the AUD/USD twin strategy. AUD/USD was already deprecated for CI_lo < 0 per System Status §10.2; this re-run only affects AUD/JPY.
- **Not** building a new live class. If outcome is "Reconfigure", that's a config-change PR using the existing live class with a parameter change.

---

## 3. Implementation

1. **This directive on `main`.** (THIS PR)
2. `research/mean_reversion/run_mr_audjpy_anchor_compare.py` — small wrapper that calls the existing `run_mr_wfo` twice with `anchor_period=24` and `anchor_period=6`, emits a side-by-side comparison.
3. Run the comparison.
4. Append result log to §4.
5. If §1.2 says "Reconfigure" or "Retire", open a follow-up PR for the config change (not this directive's scope).

---

## 4. Result log

Appended 2026-05-14 after the comparison ran. §1-§3 unchanged (V3.1).

### 4.1 Side-by-side

Strategy class: existing audit-corrected `run_mr_wfo` from `research/mean_reversion/run_confluence_regime_wfo.py`. Identical regime mask (`conf_rsi_14_dev`, 70.6% of bars), identical tier grid `{0.95, 0.98, 0.99, 0.999}`, identical IS=30k / OOS=7.5k, identical spread+slip cost (2+1 bps). Only `anchor_period` differs.

| Field | **anchor=24 (live)** | **anchor=6 (IC peak)** | Δ (6 - 24) |
|---|---:|---:|---:|
| Stitched OOS Sharpe | **+0.769** | +0.095 | -0.674 |
| **CI_lo (95%)** | **+0.32** | -0.411 | **-0.731** |
| CI_hi (95%) | +1.163 | +0.676 | -0.487 |
| Stitched DD | **-14.27%** | -35.73% | -21.46pp |
| n_folds | 8 | 8 | 0 |
| % positive folds | 88% | 62% | -25pp |
| Total trades | 214 | 450 | +236 |

### 4.2 Verdict per §1.2

Row 4 of the pre-committed decision matrix applies:

> Cell A CI_lo > 0 AND Cell B CI_lo ≤ 0 → Anchor=6 is worse on the engine despite stronger IC. Strange but possible -- the strategy's layers (regime gate, percentile tiers) interact with anchor in unexpected ways. Document; status quo with a watchpoint.

**Decision: STATUS_QUO_WATCHPOINT.** `config/mr_audjpy.toml` stays at `anchor = 24`. No PR, no config change, no live class update.

### 4.3 Mechanism — why the IC-peak loses on the engine

Raw IC measures: "high signal value → high forward return" on a per-bar basis. anchor=6 has 4× higher IC magnitude than anchor=24 at the bar-pair level.

The STRATEGY is not "buy every bar with signal > threshold and hold 1 bar". It is a tier-percentile entry pyramid with:

- **Rolling-percentile thresholds** at `{p95, p98, p99, p99.9}` of the signal's last 500 bars. Calibrated on IS, frozen at OOS start.
- **Tier-scaled position sizing** {1, 2, 4, 8}.
- **50% reversion exit** (closes the full pyramid when deviation reverts halfway back).
- **Regime gate** (confluence disagreement on `rsi_14_dev` across H1/H4/D/W).
- **Session window** (07:00-12:00 UTC entry only).
- **Vol-target sizing** at 8% annualised.

At anchor=6:
- Signal is in the upper percentile tail more often AND more transiently
- Pyramid stacks more aggressively, exposing larger size to whipsaws
- The 50% reversion exit triggers more frequently and fragments wins
- Net: 2× the trade count, more cost burn, deeper drawdown
- Despite the raw IC peak being stronger, the strategy's mechanical layers were tuned for a slower-moving signal -- anchor=24

This is exactly the audit-grade illustration of:

> **IC is the entry-signal's predictive content. Sharpe is the strategy's net realised P&L after all layers. The two can disagree -- and the strategy's deployment-eligibility is governed by Sharpe, not by IC.**

The Phase 0 strategy spec for range-expansion (`Strategy Range-Expansion ES-NQ H1 — Phase 0 2026-05-14.md`) had the OPPOSITE mismatch: IC was real but Sharpe was negative due to costs. Here the mismatch is in the other direction: IC peak is "wrong" (for the strategy), Sharpe peak is at anchor=24.

### 4.4 Watchpoint

The "watchpoint" status means: while the live config is fine TODAY, monitor for:

- Sustained underperformance vs the backtest +0.769 Sharpe -- a 12-month rolling Sharpe below +0.3 with CI_lo < 0 would trigger a fresh audit pre-registration.
- Significant change in the signal's regime distribution (e.g. AUD/JPY vol regime shift) that may break the tier-percentile calibration.
- Re-evaluate anchor choice if the strategy's tiers / regime filter / sizing change (any of those would require a fresh anchor comparison under the new layer stack).

No new code, parity test, or directive needed unless one of the above fires.

### 4.5 V3.6 lesson — propagated to project-wide hygiene

**Don't conflate IC ranking with strategy ranking when the strategy has multiplicative layers.** The fine-grid IC scan was correct that anchor=24 sits on a tail of the raw-signal IC curve. But that fact, by itself, doesn't imply the deployed config is sub-optimal -- only that it's not maximizing the raw signal. Strategy layers (tiering, gating, sizing) re-shape the effective signal-to-noise ratio per cell, and the live optimum can differ sharply from the raw-IC optimum.

This lesson rolls into the V3.6 catalogue:

| Lesson | Recorded in |
|---|---|
| DSR-passing IC ≠ deployable strategy. | `Strategy Range-Expansion ES-NQ H1 — Phase 0` §4.7 |
| Raw IC peak ≠ strategy-engine peak when strategy has multiplicative layers. | THIS directive §4.5 |

### 4.6 Outcome record

| Field | Value |
|---|---|
| Live config change required? | **No** — anchor stays at 24 |
| Strategy retirement required? | No |
| New watchpoint added? | Yes (§4.4) |
| Backtest math used? | `run_mr_wfo` (audit-corrected, shared sharpe + bootstrap CI per April 2026 audit) |
| Sanctuary discipline preserved? | Yes — the existing WFO design respects no-look-ahead via frozen IS percentile levels applied to OOS, and the 8-fold OOS window covers 2013-Q3 through 2024 (last ~12mo of 2025 implicitly excluded by the cumulative IS+OOS window not reaching that range) |
| Fine-grid IC's recommended-action 1 closed? | Yes — anchor=24 confirmed as the correct live setting under the audit pipeline |

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial pre-registration of the anchor=6 vs anchor=24 comparison. |
