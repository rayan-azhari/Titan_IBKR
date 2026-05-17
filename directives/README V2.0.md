# Titan V2.0 / V3.7 — current state

**Date:** 2026-05-17 (last updated)
**Repo:** `https://github.com/rayan-azhari/Titan_IBKR.git` (branch: `v2-main`)
**Framework version:** V3.7 (portfolio-level evaluation + risk-of-ruin)
**Live deployment:** V3.7 multi-strategy (GEM J5 + turtle CAT) on paper, since 2026-05-17 23:49 UTC

---

## TL;DR — what V3.7 changed

The framework went through three structural revisions in 2026-05-16, driven by user challenges that exposed real gaps:

1. **L64 — CONDITIONAL_WATCHPOINT path** for borderline V1-era audits. The strict CI_lo > 0 gate at small fold counts is biased toward false negatives. Strategies with positive OOS Sharpe + stable cell ranking + L21 PASS + cost-not-bottleneck can deploy at small weight with re-audit cadence.
2. **L65 — Risk-of-ruin module** (`titan/research/framework/ruin.py`). Joint MC across LIVE + proposed strategies is the binding survivability gate. Single-strategy ruin gates are necessary but not sufficient. Caught: turtle CAT at 30% FAILS joint gate alongside GEM J5; revised to 20%.
3. **L66 + L67 — Portfolio-vs-benchmark is the deployment gate** (not individual-strategy-vs-own-instrument). Individual L17 was correct for what it tested but was the wrong test for deployment. V3.7's 10-metric portfolio matrix (`research/portfolio/joint_evaluation.py`) is the new gate.

**Result of the V3.7 reframe:** the current portfolio (GEM J5 70% + turtle CAT 20% + 10% cash) **beats 60/40 SPY/IEF on 7/10 metrics** including 64× lower P(NAV DD > 15%). The mission — "minimise risk while staying profitable vs B&H" — is being achieved at the portfolio level, not the individual-strategy level.

---

## Current portfolio state

| Strategy | Status | Weight | Last audit | Verdict |
|---|---|---:|---|---|
| `gem` (J5 P_hl60_vt05) | **LIVE on paper** (Docker, multi-strategy node) | 80% | 2026-05-16 | DEPLOY (5/5 axes + L65 + portfolio matrix 6/10 vs 60/40) |
| `turtle` (C3_peak on CAT) | **LIVE on paper** (Docker, multi-strategy node, since 2026-05-17 23:49 UTC) | 20% | 2026-05-16 | CAT-scoped; joint L65 PASS; portfolio matrix 7/10 |

**Container**: `titan-portfolio` (Docker) running `scripts/watchdog_portfolio.py --strategies v37_live` since 2026-05-17 23:49 UTC. Both strategies in same NautilusTrader TradingNode sharing IB connection + PortfolioRiskManager + PortfolioAllocator. GEM positions REHYDRATED cleanly (CSPX qty=27 + IDTM qty=45). Account balance preserved (29,846.16 GBP unchanged across cutover). See `directives/V3.7 Phase 2 Cutover Log 2026-05-17.md`.

**Portfolio-vs-60/40 SPY/IEF (7.9y overlap):**
- Sharpe: **+0.95** vs +0.74 ✓
- Sortino: **+1.41** vs +0.91 ✓
- MaxDD: **−13.7%** vs −22.3% ✓
- CVaR-95 reduced 48% ✓
- CDaR-95 reduced 44% ✓
- **P(NAV DD > 15% in 1y): 0.30% vs 19.30%** — **64× lower risk-of-ruin** ✓
- Verdict: **7/10 PASS → PORTFOLIO_CONDITIONAL**

**Crisis stress (named windows 1987-2025):**
- GEM alone: 9/10 PASS at 20% DD threshold (worst 2022: −20.4%)
- Portfolio: **5/5 PASS at TIGHTER 15% threshold** (worst 2022: −11.6%)
- Diversification reduces worst-crisis DD by 44%

---

## Audit dashboard (2026-05-17)

| Status | Count | Strategies |
|---|---:|---|
| **LIVE** | 2 | gem J5 (80%) + turtle CAT C3_peak (20%) — multi-strategy node since 2026-05-17 23:49 UTC |
| **SHADOW (paper-validation)** | 2 | bond_gold (V3.6 PROMOTED params; live runs V1) + **ewmac_regime_i1v2_c6** (LIVE shadow since 2026-05-17, 9 futures, 12mo paper validation) |
| **CONDITIONAL_WATCHPOINT** | (incl. above) | turtle CAT scoped, fx_carry long-yen scope, ic_equity_daily top-3 (deferred per L67 expansion) |
| **RETIRED** | 22 | mr_audjpy, mr_fx, mtf, ic_mtf, gold_macro, gld_confluence, 6 etf_trend variants, B2 family (PARTIALLY REVIVED via I1v2), B5 intraday momentum, **ml (Wave C)**, B4/D2/E1/G4/A1 from V3 era |
| **Total audits run** | 39 | See `.tmp/dashboard/dashboard.html` |

**Latest milestones (2026-05-17):**
- **B5** RETIRE on SPY/QQQ/IWM 2y IBKR M5 (panel median SR −0.87, signal reversed).
- **B2e** L52 sweep stop + L52-override audit → noise fragility (L69), partial rescue motivated by I1v2.
- **I1v2 C6_smoothed DEPLOY** verdict (Sharpe +0.52, CI_lo +0.049, noise=best) — first B2-family deployment-eligible cell. L65 single PASS + L65 joint PASS at 5%+ weight + L67 unchanged at PORTFOLIO_CONDITIONAL. **Risk reducer, not return enhancer.**
- **I1v2 SHADOW LIVE deployed** (2026-05-17 17:23 UTC) on 9-symbol futures basket (ES/NQ/CL/BZ/HG/SI/GC/ZN/ZB). 6E/6J dropped — paper account lacks CME FX futures subscriptions. `shadow_mode=True`, `trading=False`. 12mo paper-validation clock started. Re-audit 2026-11-17.

---

## V3.6 → V3.7 framework

### Core gates (unchanged from V3.6, refined for L66)

1. **L21 causality smoke** — corrupt future bars, assert past returns bit-exact unchanged.
2. **WFO** — class-default fold construction (rolling vs expanding per typology).
3. **MC block bootstrap** — class-default MaxDD + L17 relative-MC vs **baseline-class-appropriate benchmark** (per L66, NOT universally own-instrument B&H).
4. **DSR deflation** — adjust Sharpe for N-cell selection bias.
5. **Sanctuary divergence test** — 12-month holdout never seen during sweep.
6. **L24 noise robustness** — Varma noise-injection 5th axis.
7. **L52 plateau detection** — 2D parameter sweep + spread-pct gate.

### V3.7 additions

8. **L64 CONDITIONAL_WATCHPOINT** — relaxed verdict path with non-negotiable gates (L21, costs, V1-style sanity) + relaxed-CI-gate.
9. **L65 risk-of-ruin** (`titan/research/framework/ruin.py`) — `assess_strategy_ruin()` + `assess_joint_ruin()` with 1% kill-trip + 25% 95th-pct DD gates.
10. **L66 baseline-must-match-thesis** — `risk_reduction` → vol-matched B&H; `alpha_generation` → CAPM/FF residual; `pure_return` → cash; `diversifier` → portfolio with-and-without; `cross_asset_rotation` → 60/40 SPY/IEF.
11. **L67 portfolio-vs-benchmark 10-metric matrix** (`research/portfolio/joint_evaluation.py`) — the actual deployment gate.
12. **Fractional Kelly sizer** (`titan/research/framework/kelly.py`) — 0.25× full Kelly with DSR-aware option.
13. **ERC allocator** (`titan/research/framework/allocator_erc.py`) — Equal Risk Contribution weights, correlation-aware.
14. **Named-crisis stress** (`titan/research/framework/crisis_stress.py`) — 12 historical windows (1987-2025 tradewar) with per-event pass/fail.
15. **CUSUM drift monitor** (`titan/research/framework/drift_cusum.py`) — Page 1954 structural-break detection for live PnL.
16. **Per-strategy risk-contribution** in `PortfolioRiskManager.get_summary()`.

### Lessons catalogue

**67 lessons** as of 2026-05-16. See `V3.6 Lessons Catalogue.md` (will be renamed `V3.7 Lessons Catalogue.md` at next batch).

Recent additions (V3.7 batch):
- **L58** (refined) — Signal-layer-first audit pattern: triage Sharpe magnitude is unreliable; direction is robust.
- **L59** — Gap-closure pattern for strategies with credible prior audit.
- **L60** — Cadence annualisation mismatch (FX 24/7 vs US equity RTH); use asset-class-correct `periods_per_year`.
- **L61** — Single-instrument selection bias; require multi-instrument panel test.
- **L62** — Sharpe-gap classification table (0.5-1.5 = methodology; 1.5-3 = single LA; 5-10 = multi-TF amplified; 10+ = fabrication boundary).
- **L63** — Verify V1-style baseline recovers V1 claim BEFORE concluding fabrication (user-challenge protocol).
- **L64** — CONDITIONAL_WATCHPOINT path for borderline audits.
- **L65** — Joint risk-of-ruin is the binding survivability gate.
- **L66** — Baseline must match strategy thesis.
- **L67** — Portfolio-vs-benchmark 10-metric matrix is the deployment gate.

---

## How to use V3.7

### For a new strategy candidate

1. **Read** `V3.6 Lessons Catalogue.md` + `Retirement Registry.md` (avoid known failure modes).
2. **Classify** under `StrategyClass` enum from `titan.research.framework.typology`.
3. **Declare baseline_class** per L66 (`risk_reduction` / `alpha_generation` / `pure_return` / `diversifier` / `cross_asset_rotation`).
4. **Pre-register** §1-§3 in a directive BEFORE examining data.
5. **Run audit** via framework primitives:
   ```python
   from titan.research.framework import (
       slice_sanctuary, build_folds, run_block_mc, run_relative_block_mc,
       run_noise_robustness, deflated_sharpe, sanctuary_divergence_test,
       decide, DecisionInputs,
       # V3.7
       compute_kelly_fraction, compute_erc_weights,
       run_crisis_stress, assess_strategy_ruin, assess_joint_ruin,
   )
   ```
6. **Apply V3.7 gates:** strategy quality gate (V3.6 5-axis) AND portfolio inclusion gate (V3.7 10-metric matrix).
7. **L65 ruin check** at proposed deployment weight before live cutover.
8. **Joint L65** with all existing LIVE strategies before live cutover.
9. **Append result log** + any new lessons.

### For a re-audit of an existing strategy

Same workflow, but the V1-era roster (`V1-era Re-audit Sweep Roster 2026-05-16.md`) tracks status. Each strategy is one of: complete + verdict, in-progress, deferred.

### To run V3.7 portfolio review

```bash
PYTHONIOENCODING=utf-8 uv run python research/portfolio/v37_portfolio_review.py
```

Outputs: per-strategy Kelly fractions, ERC weights, per-crisis stress, joint ruin sensitivity grid, 10-metric matrix verdict, recommended deployment weights.

### To run the 10-metric joint evaluation alone

```bash
PYTHONIOENCODING=utf-8 uv run python research/portfolio/joint_evaluation.py
```

---

## Status of the strategy backlog

### V3-era backlog ideas (`Strategy Backlog 2026-05-14.md`)

10-step plan, ~31 days for 8 strategies + 3 infra. Steps 1-7 complete:
1. ✅ B1 GEM dual momentum — LIVE under J5
2. ✅ J3 Noise-injection 5th axis — framework primitive
3. ✅ E1 VRP — RETIRED (L29)
3.5. ✅ J4 GEM noise-robust redesign — superseded by J5
4. ✅ D2 Commodity carry — RETIRED (L34/L35)
5. ✅ G4 Overnight session — RETIRED (L33)
6. 🟡 J1+J2 HRP/NCO allocator — **partially done**: ERC built (V3.7 batch 1); HRP/NCO pending until 10+ strategies
7. ✅ A1 Residual momentum — RETIRED (L36/L37)
8. 🟡 I1 HMM regime + XGBoost — **v1 RETIRED (L51 no-op)**; v2 motivated by L69 (B2e noise fragility). Regime panel ready: `data/i1_regime_panel.parquet` (7 features × 3945d, built 2026-05-17). Awaits implementation.
9. ✅ B5 Intraday momentum (Gao-Han-Li-Zhou) — **RETIRED 2026-05-17** on SPY/QQQ/IWM 2y M5 (panel median SR −0.87, signal reversed)
10. ✅ B2 Carver EWMAC ensemble — **family CLOSED 2026-05-17** through B2/B2b/B2c/B2d/B2e chain. B2e on IBKR cross-asset 9y data: L52 sweep no plateau; L52-override audit no cell promotes; **noise fragility (L69)**, not regime artifact

### V1-era re-audit roster (`V1-era Re-audit Sweep Roster 2026-05-16.md`)

| Wave | Strategy | Status |
|---|---|---|
| A.1 | bond_gold | ✅ CONDITIONAL_WATCHPOINT (PROMOTED params; shadow vs live) |
| A.2 | etf_trend (7 variants) | ✅ Mixed (SPY RETIRED; TQQQ CONDITIONAL; 5 unleveraged bulk-retired) |
| A.3 | mr_audjpy | ✅ RETIRED (L58 signal-layer fail) |
| A.4 | samir_stack | 🟡 Phase 5 validated + 3 V3.6 gaps; ~25min gap-closure pending |
| A.5 | mtf | ✅ RETIRED (L21 look-ahead) |
| A.6 | mr_fx | ✅ RETIRED (V2 verified) |
| Wave B | gold_macro | ✅ RETIRED (plateau-fail + cell-instability) |
| Wave B | turtle | ✅ CONDITIONAL_WATCHPOINT (CAT-scoped @ 20%) |
| Wave B | ic_mtf | ✅ RETIRED (L21 multi-TF amplified) |
| Wave B | gld_confluence | ✅ RETIRED (DD gate + fold instability) |
| Wave B | fx_carry | ⏳ pending (needs L61 G10 panel + L66 baseline) |
| Wave B | orb | ⏳ pending (sparse-trade harness) |
| Wave B | pairs | ⏳ pending (cointegration harness) |
| Wave B | ic_equity_daily | ⏳ pending (7-ticker multi-name) |
| Wave C | ml | ⏳ L19 same-bar look-ahead bug fix first |
| Wave C | gap_fade | ⏳ low priority |

---

## Architecture (unchanged from V2.0)

```
directives/                     ~36 files: methodology + pre-regs + lessons + retirement registry
titan/
  titan/research/framework/     V3.7 unified backtesting framework (16 modules)
  titan/strategies/             Live class implementations (15+ strategies)
  titan/risk/                   PortfolioRiskManager + allocator + per-strategy equity
  titan/adapters/               IBKR/NautilusTrader
research/                       Library + audit harnesses (pre-cleanse retained)
  research/portfolio/           V3.7 joint evaluation + v37_review (NEW)
  research/turtle/              turtle V3.6 module + audit
  research/cross_asset/         bond_gold + gold_macro V3.6 modules
  research/gem/                 GEM J3/J4/J5 audits
  research/exploration/         sweep_*.py harnesses (one per strategy)
scripts/                        CLI: run_live_*, run_portfolio, kill switches, watchdogs, audit_dashboard
config/*.toml                   Strategy parameter intent
tests/                          54+ tests (framework + IC + live classes + 13 PRM tests)
data/                           Gitignored OHLCV parquets
.tmp/                           Reports + dashboard + logs (gitignored)
```

---

## Non-negotiable rules (V3.7)

Same as V3.6 plus:

- **L21 causality smoke**: every audit module must pass it. Never relaxed.
- **L60 annualisation**: `periods_per_year` must match asset class, NOT generic `BARS_PER_YEAR[timeframe]` for non-FX.
- **L63 V1-style verification**: before concluding "fabrication", verify V1-style baseline can reproduce the V1 claim.
- **L65 ruin gate**: every CONDITIONAL_WATCHPOINT or STRICT DEPLOY requires single-strategy + joint L65 PASS.
- **L66 baseline-class declaration**: every new pre-reg declares `baseline_class`. Default for new ideas: discuss before defaulting.
- **L67 portfolio inclusion gate**: new strategies must improve the portfolio's 10-metric matrix vs 60/40 to be promotable.

---

## Open items (V3.7)

### Framework

1. **Tax-aware return computation** — UK CGT vs US Section 1256; needs operator input on jurisdiction.
2. **HRP / NCO allocator** — defer until 10+ strategies in portfolio.
3. **Tail risk parity** — defer (requires 5+ years OOS per strategy).
4. **Live PnL CUSUM wiring** — wired to research-mode for now; needs live PnL tracker connection.
5. **Exceedance correlation** in joint MC — currently uses unconditional correlation.

### Strategies pending audit

- fx_carry, orb, pairs, ic_equity_daily, ml, gap_fade (priority order)
- samir_stack Phase 6c gap-closure (~25 min when allocator window opens)

### Deferred (need data infrastructure)

- B2 Carver EWMAC ensemble — needs futures basket data
- B3 Donchian + pyramiding Turtle-plus — same
- B4 TSMOM multi-asset — same
- A1 Residual momentum re-audit — needs WRDS CRSP/Compustat
- I1 HMM with multi-feature regime — needs VIX + term spread + credit + RV data

---

## Workflow conventions

- **Branches:** every piece of work is a feature branch. Audit + research on `research/*`; deployable on `feat/*` or `fix/*`. Merge to `v2-main` via PR.
- **Pre-registration discipline:** every audit's gates / cells / decision rule lands in a directive BEFORE any data is examined.
- **V3.6/V3.7 hygiene:** every audit produces a result log even when verdict is RETIRE. Lessons get appended to the catalogue. Retirement Registry updated.
- **Risk gates non-negotiable:** L21 + L60 + L63 + L65 + L66 + L67 all binding for any new deployment.

The framework is the toolbox; the portfolio decision matrix is the gate; the lessons catalogue keeps us honest. As of 2026-05-16, the current portfolio achieves the V3.7 mission.
