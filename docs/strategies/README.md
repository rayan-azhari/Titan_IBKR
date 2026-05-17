# Strategy Documentation

Operator + reference guides for every strategy that has reached at least
a paper-trading deployment under the Titan-IBKR V2.0 / V3.7 framework.
Each guide is the **single source of truth** for how its strategy is
configured, deployed, and supervised. Research artefacts (pre-reg
directives, audit logs, dashboards) live elsewhere — see the linked
sources in each guide.

## Portfolio status (2026-05-17, V3.7 framework)

| Metric (portfolio GEM 70% + turtle 20%) | Value | 60/40 SPY/IEF | Verdict |
|---|---:|---:|:---:|
| Sharpe (annualised) | **+0.95** | +0.74 | ✓ |
| Sortino | **+1.41** | +0.91 | ✓ |
| MaxDD | **−13.7%** | −22.3% | ✓ |
| **P(NAV DD > 15% in 1y)** | **0.30%** | 19.30% | ✓ **64× lower** |
| CVaR-95 reduction | 48% | — | ✓ |
| CDaR-95 reduction | 44% | — | ✓ |
| Worst named-crisis MaxDD (2022) | −11.6% | — | (vs GEM alone −20.4%) |
| 10-metric matrix verdict | **7/10 PASS** | — | PORTFOLIO_CONDITIONAL |

The portfolio HAS LOWER risk-of-ruin than passive 60/40 buy-and-hold
AND higher Sharpe AND lower MaxDD. See
[.tmp/reports/joint_evaluation/framework_evolution_v37.md](../../.tmp/reports/joint_evaluation/framework_evolution_v37.md)
for the V3.6 → V3.7 reframe.

## Convention

One file per strategy (or per strategy family). Filename = canonical
slug, e.g. `gem-dual-momentum.md`. Each guide must contain at least:

1. **Header** — version, last-updated date, status, source paths.
2. **Executive summary** — 1 paragraph + headline audit numbers.
3. **What it trades** — instruments per universe (US / UK / etc).
4. **How it trades** — the decision logic in plain language.
5. **Parameters** — the production cell's frozen config.
6. **Cost model** — calibrated against real fills.
7. **Live deployment** — docker, env vars, ports, watchdog.
8. **Operations** — warmup, logs, kill switch, restart.
9. **Audit summary** — the 5-axis verdict numbers.
10. **Known caveats** — data history limits, substitution losses, etc.

## V3.6-confirmed strategies

| Slug | Status | Class | Production cell | V3.6/V3.7 verdict | Guide |
|---|---|---|---|---|---|
| `gem-dual-momentum` | **LIVE on paper** (multi-strategy node since 2026-05-17 23:49 UTC) | CROSS_ASSET_MOMENTUM | **J5 `P_hl60_vt05`** (since 2026-05-16) | DEPLOY (5/5 axes + L65 + portfolio matrix 6/10 vs 60/40) | [gem-dual-momentum.md](gem-dual-momentum.md) |
| `turtle-donchian` | **LIVE on paper (CAT-scoped)** (multi-strategy node since 2026-05-17 23:49 UTC) | DAILY_TREND | **C3_peak `(entry=45, exit=20)` on CAT** | L64 CONDITIONAL_WATCHPOINT + L65 joint PASS + portfolio matrix 7/10 vs 60/40 | (TODO: write guide) |
| `ewmac_regime_i1v2_c6` | **SHADOW LIVE on paper** (multi-strategy node since 2026-05-17 17:23 UTC) | DAILY_TREND + regime gate | **C6_smoothed** (2-state HMM panel gate + 5d median smooth + 3-speed EWMAC) on 9 futures (ES/NQ/CL/BZ/HG/SI/GC/ZN/ZB) | DEPLOY (Sharpe +0.52, CI_lo +0.049, noise=best) + L65 single+joint PASS + L67 PORTFOLIO_CONDITIONAL unchanged. **Risk reducer.** 12mo paper-validation; re-audit 2026-11-17. | (TODO) |
| `bond-gold` | LIVE on paper (V1-era config); V3.6 Phase 1 shadow config exists | CROSS_ASSET_MOMENTUM | V3.6 PROMOTED `(lookback=120, threshold=0.50)` — sidecar config ready | CONDITIONAL_WATCHPOINT (4/5 axes) | [bond-gold.md](bond-gold.md) |
| `etf-trend` | Family of 7 variants with **mixed verdicts** | DAILY_TREND | TQQQ `(150, 5)` PROMOTED CONDITIONAL; SPY RETIRED; 5 unleveraged variants HIGH-conf bulk-retire (DBC/GLD spot-checks confirmed 2026-05-16) | Mixed (see family doc) | [etf-trend.md](etf-trend.md) |

## V1-era strategies pending V3.6 re-audit

The remaining V1-era live strategies in [titan/strategies/](../../titan/strategies/) still run per their existing `config/*.toml`. **None has been re-audited under the V3.6 framework yet.** V1-era Sharpe/CI claims are suspect (per [directives/README V2.0.md](../../directives/README%20V2.0.md)).

See [directives/V1-era Re-audit Sweep Roster 2026-05-16.md](../../directives/V1-era%20Re-audit%20Sweep%20Roster%202026-05-16.md) for the audit schedule.

| Strategy | Class | Priority | Audit wave | Status |
|---|---|---|---|---|
| `mr_audjpy` | INTRADAY_MICROSTRUCTURE (H1 MR) | P1 | Wave A.3 | **SIGNAL-LAYER FAIL 2026-05-16 (L58)** — V1 +0.53 claim not reproducible at signal layer (IS -0.30); de-allocation recommended |
| `samir_stack` | CROSS_ASSET_MOMENTUM + overlay | P1 | Wave A.4 | **PHASE 5 VALIDATED + 3 V3.6 GAPS 2026-05-16 (L59)** — keep live; ~25-min Phase 6c gap-closure scheduled |
| `mtf` | INTRADAY_MICROSTRUCTURE | P1 | Wave A.5 | **RETIRE 2026-05-16 (L21 bug confirmed)** — V1 +1.94 claim is look-ahead-derived; V3.6-correct sweep gives -0.08 |
| `mr_fx` | INTRADAY_MICROSTRUCTURE (M5 VWAP) | P1 | Wave A.6 | **RETIRE 2026-05-16 (verified)** — every cell negative on 15y M5 EUR/USD even with corrected mechanics; L58 magnitude-vs-direction caveat refined |
| `gold_macro` | DAILY_TREND | P2 | Wave B | **RETIRED 2026-05-16 (full audit)** — L52 H1 plateau-fail OOS 71% spread + L46 CI_lo bottleneck (every cell CI95_lo < 0). Composite ADDS value over bare-SMA but absolute CI gate fails. Never Docker-deployed; no allocator action |
| `turtle` | DAILY_TREND | P2 | Wave B | **LIVE on paper since 2026-05-17 23:49 UTC** (multi-strategy node alongside GEM). C3_peak (entry=45, exit=20) on CAT only @ 20% weight. L64 CONDITIONAL_WATCHPOINT + L65 joint PASS + portfolio matrix 7/10 vs 60/40. Re-audit 2026-11-16 |
| `fx_carry` | CARRY | P2 | Wave B | **CONDITIONAL_WATCHPOINT (long-yen-carry scope) 2026-05-16** — L21 PASS, live AUD/JPY SR +0.29. L61 panel: only AUD/JPY + USD/JPY positive (panel median −0.007); pattern is coherent. Carry-premium adds ~+0.6 SR pure-price audit ignores. Deploy ~10% AUD/JPY only, joint L65 + L67 needed before live |
| `ic_mtf` | INTRADAY_MICROSTRUCTURE | P2-HIGH | Wave B | **RETIRED 2026-05-16 (full audit)** — L21 look-ahead bug CONFIRMED (same pattern as mtf, ~4-5x more severe). V1 +7-8 Sharpe reproducible at V1-style methodology; vanishes under causal higher-TF alignment. New lessons L62 (refined) + L63 added |
| `gld_confluence` | INTRADAY_MICROSTRUCTURE | P2-low | Wave B | **RETIRED 2026-05-16 (formalises April-2026 deprecation)** — MaxDD ~−35% at every threshold (hard DD gate) + prior 34% positive WFO folds (L43 cell-instability). Sharpe +0.22 too small to override. Already removed from STRATEGY_REGISTRY |
| `orb` | INTRADAY_BREAKOUT | P2-medium | Wave B | **TENTATIVE RETIRE 2026-05-16 — DATA INSUFFICIENT** (5 weeks M5 = 17-19 trades/ticker; simplified-no-filter panel SR median −2.50, 29% positive). Re-audit when 1+ year M5 available |
| `ic_equity_daily` | DAILY_MEAN_REVERSION | P2-medium | Wave B | **CONDITIONAL_WATCHPOINT candidate 2026-05-16** — L21 PASS; all 7 tickers strict-OOS SR > 0 (median +0.61, mean +0.62). L62 V1-gap +2.56 = single-LA class. Deploy top-3 (HWM/WMT/SYK) at ~5-10%, joint L65 + L67 needed |
| `pairs` | PAIRS | P2-low | Wave B | **RETIRED 2026-05-16** — L21 smoke FAILED (subtle non-causality in beta refit / z-window interaction). Live SR +0.11, CI_lo −0.29, MaxDD −30%. L21 PASS gate non-negotiable |
| `ml` | ML_CLASSIFIER | P3 | Wave C | **L19 same-bar look-ahead bug** must be fixed before audit |
| `gap_fade` | INTRADAY_MICROSTRUCTURE | P3 | Wave C | pending V3.6 audit (low priority) |

These strategies' V1-era pre-reg directives still exist in [directives/](../../directives/), but the V3.6 protocol (5-axis matrix + L17 relative-MC + Varma noise robustness + L52 hybrid sweep) was not applied. **Treat their advertised Sharpes as informational only** until each gets a fresh V3.6 audit + guide in this folder.

## Retired strategies (V3.6/V3.7 RETIRED verdict)

The following strategies were audited and RETIRED. **42 audits / 22 retired / 2 LIVE / 2 SHADOW** as of 2026-05-17. See [.tmp/dashboard/dashboard.html](../../.tmp/dashboard/dashboard.html) for the full registry and [directives/Retirement Registry.md](../../directives/Retirement%20Registry.md) for one-line "next time" lessons per retired strategy.

V3-era (data-infrastructure or methodology):

- B2 Carver EWMAC family — RETIRED via B2 → B2b → B2c → B2d → I1 (HMM v1) → **B2e** chain (L46/L48/L49/L51/**L69**). B2e (2026-05-17, IBKR cross-asset 11sym 9y, clean Databento futures) refined the closure: failure mode is **noise fragility**, not regime artifact — B2b L48 universal-decline thesis partially falsified. Signal positive every cell; sanctuary +1.0 to +1.3; what fails is worst-case noise robustness at retail costs.
- B5 intraday momentum (Gao-Han-Li-Zhou) — **RETIRED 2026-05-17** on SPY/QQQ/IWM 2y IBKR M5. Panel median Sharpe **−0.87**, 0% positive, signal REVERSED in 2024-26 (matches academic post-2014 decay).
- ml (EUR/USD H1 XGB TBM meta-classifier) — **RETIRED 2026-05-17** after 4-stage exhaustive cascade: (1) single-pair / 24h: sanc AUC 0.488; (2) L61 multi-asset grid: best 6h XGB 6/16 above 0.55; (3) per-asset 5-axis on top-6: 0/6 deploy-eligible (CI_lo < 0, 96-99% bars active, costs dominate); (4) cross-asset features (regime panel as columns): mean AUC 0.5424 → 0.5576 (+0.015), equities lift 4-6pp but 0/16 above 0.60 and 5-axis unchanged. **L71** (frozen ML artefact is build-of-feature-pipeline) + **L72** (classification AUC > 0.55 is NOT a deployment signal; always close the loop with strategy-return 5-axis) added.
- B4 TSMOM (3 variants) — RETIRED on L43 plateau-fragility.
- D2 commodity carry — RETIRED on data-quality + signal absence (L34/L35).
- E1/E1b VRP — RETIRED (L29).
- G4 overnight session decomposition — RETIRED (cost drag + L33).
- A1 residual momentum — RETIRED on survivorship bias (L36/L37).
- I1 HMM regime gate (v1, raw-returns features) — RETIRED on L51. **A v2 with richer regime features is now motivated by L69** (B2e noise fragility opens the per-asset regime-gating rescue path).

V1-era live strategies retired under V3.6/V3.7:

- **mr_audjpy** — RETIRED (Wave A.3, L58 signal-layer fail).
- **mr_fx EUR/USD** — RETIRED (Wave A.6, V2 verified).
- **mtf EUR/USD** — RETIRED (Wave A.5, L21 multi-TF look-ahead).
- **etf_trend SPY** — RETIRED (Wave A.2, L56).
- **etf_trend QQQ, IWB, EFA, DBC, GLD** — RETIRED bulk (Wave A.2 + spot-checks, L56).
- **ic_mtf 6 FX pairs** — RETIRED (Wave B, L21 multi-TF amplified 8-15 SR).
- **gold_macro** — RETIRED (Wave B, L52 plateau-fail + cell-instability).
- **gld_confluence** — RETIRED (Wave B, DD gate + 34% positive folds).

Retired strategies are NOT documented in this folder — their lessons are captured in [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md) and one-line "next time" lessons in [directives/Retirement Registry.md](../../directives/Retirement%20Registry.md).

## Strategies documented elsewhere (TODO: migrate)

The following V1-era strategy guides still live in `directives/`. They should be moved into this folder ONLY AFTER the underlying strategy has been re-audited under V3.6 (a V1-era guide here would mislead operators into trusting V1 numbers):

- **Samir-Stack** — [directives/Samir-Stack Strategy Guide.md](../../directives/Samir-Stack%20Strategy%20Guide.md)
- **ORB** — [directives/ORB Strategy User Guide.md](../../directives/ORB%20Strategy%20User%20Guide.md)
- **MTF** — [directives/MTF Strategy User Guide.md](../../directives/MTF%20Strategy%20User%20Guide.md)
- **Samir V3 (VIX-HMM)** — [directives/Samir V3 — VIX-HMM Strategy Design 2026-05-13.md](../../directives/Samir%20V3%20%E2%80%94%20VIX-HMM%20Strategy%20Design%202026-05-13.md)

## V3.7 Framework Reference

The V3.6 framework was extended to V3.7 on 2026-05-16 with portfolio-level evaluation. Key new modules:

- **Portfolio joint evaluation** (10-metric matrix vs benchmark): [research/portfolio/joint_evaluation.py](../../research/portfolio/joint_evaluation.py)
- **V3.7 end-to-end review runner** (Kelly + ERC + Crisis + Joint Ruin + 10-metric): [research/portfolio/v37_portfolio_review.py](../../research/portfolio/v37_portfolio_review.py)
- **Risk-of-ruin module** (single + joint MC): [titan/research/framework/ruin.py](../../titan/research/framework/ruin.py)
- **Fractional Kelly sizer** (0.25× full Kelly, DSR-aware): [titan/research/framework/kelly.py](../../titan/research/framework/kelly.py)
- **ERC allocator** (Equal Risk Contribution): [titan/research/framework/allocator_erc.py](../../titan/research/framework/allocator_erc.py)
- **Named-crisis stress** (12 windows 1987-2025): [titan/research/framework/crisis_stress.py](../../titan/research/framework/crisis_stress.py)
- **CUSUM drift monitor** (live PnL structural breaks): [titan/research/framework/drift_cusum.py](../../titan/research/framework/drift_cusum.py)
- **Risk-contribution reporting** added to [titan/risk/portfolio_risk_manager.py](../../titan/risk/portfolio_risk_manager.py)`.get_summary()`

## Related reading

- **Project status:** [directives/README V2.0.md](../../directives/README%20V2.0.md) (updated 2026-05-16 with V3.7 state)
- **Methodology + decision matrix:** [directives/Methodology Audit & Unified Framework 2026-05-14.md](../../directives/Methodology%20Audit%20%26%20Unified%20Framework%202026-05-14.md)
- **Lessons catalogue (L01–L69):** [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md)
- **Retirement Registry** (one-line lessons per retired strategy): [directives/Retirement Registry.md](../../directives/Retirement%20Registry.md)
- **V1-era re-audit roster:** [directives/V1-era Re-audit Sweep Roster 2026-05-16.md](../../directives/V1-era%20Re-audit%20Sweep%20Roster%202026-05-16.md)
- **L52 hybrid framework (sweep + plateau + audit):** [memory/reference_hybrid_workflow.md](../../) (auto-loaded into Claude memory)
- **Strategy backlog (38 candidate ideas):** [directives/Strategy Backlog 2026-05-14.md](../../directives/Strategy%20Backlog%202026-05-14.md)
- **Docker paper-trading runbook:** [directives/Docker Paper Trading Guide.md](../../directives/Docker%20Paper%20Trading%20Guide.md)
- **Audit dashboard:** [.tmp/dashboard/dashboard.html](../../.tmp/dashboard/dashboard.html) (31 audits, regenerated by [scripts/build_audit_dashboard.py](../../scripts/build_audit_dashboard.py))
- **V3.7 reframe memo** (answers "is B&H really lower risk?"): [.tmp/reports/joint_evaluation/framework_evolution_v37.md](../../.tmp/reports/joint_evaluation/framework_evolution_v37.md)
