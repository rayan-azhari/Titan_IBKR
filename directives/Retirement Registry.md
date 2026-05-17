# V3.6/V3.7 Retirement Registry

**Purpose:** One-line "what we'd test differently next time" for each retired strategy. Institutional-grade post-mortem registry. Recoverable lessons that survive the strategy itself.

**Last updated:** 2026-05-16
**Total retires:** 18 (as of 2026-05-16 dashboard count)

## Wave A (P1) — V1-era live strategies

| Strategy | Retired | Verdict driver | One-line "next time" lesson |
|---|---|---|---|
| `mr_audjpy` | 2026-05-16 (A.3) | L58 signal-layer −0.30 on 14y H1 | **Don't promote a strategy whose signal-layer is negative just because the filter machinery (tier-grid, regime gate, NY close) makes the audit look positive. Always test signal-layer first.** |
| `mr_fx` (EUR/USD) | 2026-05-16 (A.6) | L58 signal-layer −1.13 (V2 verified) | **Triage cost-realism + mechanic-realism: rolling VWAP + single-tier + 1.5bps is a different test from session-VWAP + 4-tier + 0.5bps. Verify magnitude with live-matched mechanics; direction is the robust signal.** |
| `mtf` | 2026-05-16 (A.5) | L21 multi-TF look-ahead | **Multi-TF strategies need explicit causality smoke: corrupt future bars across ALL TFs and assert past returns unchanged. V1's same-day-daily-close-at-H1 bug inflated Sharpe by 2 SR units.** |
| `etf_trend_spy` | 2026-05-16 (A.2) | L56 long-only-MA-crossover rel-MC fail | **MA-crossover long-only on liquid equity ETFs fundamentally can't reduce drawdown vs B&H (ratio ~1.0). Don't audit them individually; classify as a family and bulk-retire unless leverage-amplified (TQQQ case).** |
| `etf_trend_qqq` | 2026-05-16 (A.2) | L56 (bulk-retire generalisation) | (same as SPY) |
| `etf_trend_iwb` | 2026-05-16 (A.2) | L56 | (same as SPY) |
| `etf_trend_efa` | 2026-05-16 (A.2) | L56 | (same as SPY) |
| `etf_trend_dbc` | 2026-05-16 (A.2 spot-check) | L56 generalises to commodity-basket ETFs | **Even commodity ETFs (DBC) fail L56 — the issue isn't asset class, it's the MA-crossover mechanic on long-only-equity-like underlyings.** |
| `etf_trend_gld` | 2026-05-16 (A.2 spot-check) | L56 | **Gold trend on GLD itself differs from bond_gold's IEF→GLD signal — the former is L56-vulnerable, the latter is genuine cross-asset edge. Always classify by signal mechanic, not asset.** |

## Wave B (P2) — Wave B triage and full audits

| Strategy | Retired | Verdict driver | One-line "next time" lesson |
|---|---|---|---|
| `gold_macro` | 2026-05-16 (Wave B full) | L52 H1 plateau-fail OOS spread 71%, L46 CI bottleneck, cell-ranking instability | **IS plateau is necessary but not sufficient — check IS→OOS cell-ranking stability. If IS-best ≠ OOS-best AND surface reshuffles, the plateau was IS-specific.** |
| `turtle` | (revised to CONDITIONAL_WATCHPOINT 2026-05-16) | L60 annualisation + L61 single-instrument bias | **(NOT retired — revised to CONDITIONAL on CAT only via L64.) Apply forward: every intraday audit must justify periods_per_year per asset class; every single-instrument live config needs an H6 multi-instrument panel test.** |
| `ic_mtf` | 2026-05-16 (Wave B) | L21 multi-TF look-ahead, 8-15 SR amplification | **Multi-signal × multi-TF composites can amplify L21 look-ahead 4-5x over single-signal multi-TF. Always verify V1-style baseline recovers V1 claim BEFORE concluding fabrication (L63).** |
| `gld_confluence` | 2026-05-16 (Wave B) | Class-default DD gate hard fail (-35% MaxDD) + 34% positive folds | **L43 cell-instability isn't just about cell ranking on Sharpe — also applies to FOLD pass rate. <50% positive folds means aggregate Sharpe is dominated by few good folds; not deployable.** |

## Wave C (P3) — speculative / deferred audits

| Strategy | Retired | Verdict driver | One-line "next time" lesson |
|---|---|---|---|
| `ml` (single-pair EUR/USD H1 24h XGB) | **RETIRE 2026-05-17** (single-pair / single-horizon path) | Retrained on current features, sanctuary AUC 0.488 on EUR/USD H1 24h alone -- worse than random. **Note: L61 multi-asset grid (see below) revised the architecture-wide verdict to MARGINAL_BY_ASSET. The EUR/USD 24h cell remains a retire, but the architecture is not closed.** | **Single-pair RETIRE necessary but not sufficient; always run multi-asset panel before closing strategy class. EUR/USD 24h is a per-cell retire, not a class retire.** |
| ~~`ml` (architecture-wide)~~ | **RETIRE 2026-05-17 (final, after 4-stage cascade)** | (1) Single-pair 24h sanc AUC 0.488. (2) L61 multi-asset grid: best 6h XGB 6/16 > 0.55. (3) Per-asset 5-axis on top-6: 0/6 deployment-eligible. (4) **Cross-asset features (regime panel as columns on every row): mean AUC 0.5424 → 0.5576, 6→9 above 0.55, but 0 above 0.60. Equities benefit most (ES 0.55→0.60, NQ 0.55→0.59 — economically interpretable: bond/vol/dollar context helps equity prediction). However +0.015 AUC bump cannot overturn 5-axis CI_lo failure.** | **L72 still applies. NEW takeaway: cross-asset features DO help equity prediction (~5pp AUC bump on ES/NQ/SPY/QQQ) but the V3.6 5-axis matrix remains the binding constraint regardless of feature engineering. Once 4 architecturally-distinct approaches converge on "AUC ≠ deployable Sharpe", STOP — the framework is correct.** |

## V3-era systematic retires (B2-B4, E1, G4, A1, I1)

| Strategy | Retired | Verdict driver | One-line "next time" lesson |
|---|---|---|---|
| `B2_carver_ewmac` | 2026-04 (V3 era) | L48-L51 chain (B2→B2b→B2c→B2d→I1) | **Carver EWMAC needs futures basket; on cross-asset 21y data it doesn't reproduce. Don't audit single-asset EWMAC on equity ETFs — wait for IBKR futures-data pipeline.** |
| `B4_tsmom_24commodity` | 2026-04 (V3 era) | L43 plateau-fragility on yfinance data | **TSMOM works on multi-asset basket per Moskowitz et al. but yfinance commodity data has continuous-contract gaps. Need per-contract-roll-aware data from IBKR or Databento.** |
| `D2_commodity_carry` | 2026-05 (V3 era) | L34 (Databento .c.1 slowness) + L35 (rolling-yield signal absent) | **Commodity carry needs M2 contract data (not M1). Databento's .c.1 queries are 1000× slower than .c.0; use IBKR M2 or Refinitiv. Strict-carry BGR is still open with proper data.** |
| `E1_vrp_capture` | 2026-04 (V3 era) | L25/L26/L27/L28/L29 chain | **VRP via VIXY at retail daily resolution is not viable. Either use VX futures directly or accept the strategy class doesn't fit retail constraints.** |
| `G4_overnight_session` | 2026-05 (V3 era) | +0.83 SR cost drag + L33 (session strategies have asymmetric tail risk under regime-shuffled MC) | **Session-bounded strategies need session-shuffled MC, not full-bar MC, to avoid asymmetric tail-risk artifacts. Real edge confirmed (+0.75 gross) but cost destroys it.** |
| `A1_residual_momentum` | 2026-05 (V3 era) | L36/L37 (survivorship bias on current-S&P-500 universe) | **Residual momentum NEEDS survivorship-free CRSP/Compustat ($2k/yr WRDS). Don't audit it on current-constituent yfinance samples — the sign flips.** |
| `I1_hmm_regime` | 2026-04 (V3 era) | L51 (HMM-on-daily-returns degenerates to no-op) | **HMM on single-asset daily returns has no useful structure; the model finds 1 regime or noise. Need multi-feature HMM (VIX + term spread + credit + realised vol) per Statistical Jump Model paper.** |

## Aggregate patterns

**Most common verdict drivers (count across 18 retires):**
1. L21 multi-TF look-ahead bug: 2 (mtf, ic_mtf)
2. L56 long-only-MA-crossover rel-MC fail: 5 (SPY, QQQ, IWB, EFA, DBC, GLD)
3. L58 signal-layer-negative: 2 (mr_audjpy, mr_fx)
4. L52 plateau-fail + L46 CI: 2 (gold_macro, gld_confluence)
5. L48-L51 / L33 / L36 / L51 (data + methodology): 6 (B2, B4, D2, E1, G4, A1, I1)

**Themes:**
- **Look-ahead bugs are the dominant cause for V1-era retires** (L21 confirmed in 2; L25/L46/L58 signature in 4 more).
- **MA-crossover long-only ETFs are a category retire** — bulk-retire safely (L56).
- **Data infrastructure gaps** are the most common deferral cause (futures basket missing for B2/B4; survivorship-free equity for A1; M2 commodity contract for D2).
- **None of the retires were "wrong" methodology calls** — the V3.6 framework correctly identified broken strategies. The one that got revised (turtle → CONDITIONAL) is the exception that L64 was specifically added to handle.

## What "next time" looks like

For each retire, two improvements would unlock re-audit:

1. **Data infrastructure** — once we have:
   - IBKR-per-contract-roll-aware futures data → re-audit B2 (Carver), B3 (Donchian-plus), B4 (TSMOM), D2 (BGR carry)
   - WRDS CRSP/Compustat → re-audit A1 (residual momentum)
   - Multi-feature regime classifier data (VIX + term + credit + RV) → re-audit I1 (HMM)

2. **Framework upgrades** — most relevant retires would have been caught faster with:
   - L60 (cadence annualisation) for any future intraday audits
   - L61 (single-instrument selection bias) for fx_carry, pairs, gld_confluence, A1 (already-applied: turtle 100th percentile finding)
   - L63 (V1-style verification protocol) for any future V1-era audits
   - L64 (CONDITIONAL_WATCHPOINT path) for borderline cases
   - L65 (joint ruin) before any new deployment
   - L66 (baseline must match thesis) for any cross-asset or alpha strategy
   - L67 (portfolio-vs-benchmark as the deployment gate) for ALL future audits

## Strategy revival candidates

If the framework gains the missing data infrastructure:

| Strategy | Re-audit priority | Required data |
|---|---|---|
| ~~B2 Carver EWMAC~~ | **PARTIALLY REVIVED 2026-05-17 via I1v2.** B2e closed (noise fragility, L69). I1v2 C6_smoothed: DEPLOY (Sharpe +0.52, CI_lo +0.049, noise=best, sanctuary +1.10). **L65 single PASS** at 5/10/15%; **L65 joint**: current 80/20 mix marginally fails on 2019-2025 window (P_kill 1.05%), I1v2 inclusion at 5%+ restores compliance (P_kill 0.45-0.80%). **L67**: PORTFOLIO_CONDITIONAL unchanged (Sharpe diluted, ruin reduced). **I1v2 is a risk reducer, not return enhancer.** Recommended: shadow port to `titan/strategies/ewmac_regime/` at 5% weight, 12mo paper validation, re-audit 2026-11-17. L70 added. | n/a |
| A1 Residual momentum | High (academic literature strong) | WRDS CRSP/Compustat |
| D2 Commodity carry | Medium | IBKR M2 contracts OR Refinitiv |
| B4 TSMOM multi-asset | Medium | same as B2 |

These are NOT pending re-audits at the moment — they're entries in the strategy backlog (`Strategy Backlog 2026-05-14.md`) that retire-and-defer until infrastructure unlocks.

## Status

Registry complete as of 2026-05-16. Update when:
- A new strategy is retired (add row to relevant Wave section)
- A retired strategy is revived (move to "Strategy revival candidates" with new audit verdict)
- A new aggregate pattern emerges (update Themes section)
