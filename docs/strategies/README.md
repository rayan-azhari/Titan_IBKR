# Strategy Documentation

Operator + reference guides for every strategy that has reached at least
a paper-trading deployment under the Titan-IBKR V2.0 framework. Each
guide is the **single source of truth** for how its strategy is
configured, deployed, and supervised. Research artefacts (pre-reg
directives, audit logs, dashboards) live elsewhere — see the linked
sources in each guide.

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

| Slug | Status | Class | Production cell | V3.6 verdict | Guide |
|---|---|---|---|---|---|
| `gem-dual-momentum` | **LIVE on paper** | CROSS_ASSET_MOMENTUM | **J5 `P_hl60_vt05`** (since 2026-05-16) | DEPLOY (5/5 axes) | [gem-dual-momentum.md](gem-dual-momentum.md) |
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
| `turtle` | DAILY_TREND | P2 | Wave B | **TRIAGE: POSSIBLY VIABLE 2026-05-16** — signal-layer Sharpe +1.60 (CI_lo +0.35) on 8y CAT-H1; needs multi-ticker robustness + full L52 audit (~2h) |
| `fx_carry` | CARRY | P2 | Wave B | **TRIAGE: MARGINAL 2026-05-16** — signal-layer Sharpe +0.26, CI95 straddles 0; needs macro-overlay machinery |
| `ic_mtf` | INTRADAY_MICROSTRUCTURE | P2-HIGH | Wave B | pending — **L21 risk pattern like `mtf`**; causality-smoke FIRST |
| `gld_confluence` | INTRADAY_MICROSTRUCTURE | P2-low | Wave B | pending — likely RETIRE per L58 + L56 pattern |
| `orb` | INTRADAY_BREAKOUT | P2-medium | Wave B | pending — sparse trades, per-trade Sharpe protocol |
| `ic_equity_daily` | DAILY_MEAN_REVERSION | P2-medium | Wave B | pending — multi-ticker + tier-grid complexity |
| `pairs` | PAIRS | P2-low | Wave B | pending — needs dedicated pair-strategy audit harness |
| `ml` | ML_CLASSIFIER | P3 | Wave C | **L19 same-bar look-ahead bug** must be fixed before audit |
| `gap_fade` | INTRADAY_MICROSTRUCTURE | P3 | Wave C | pending V3.6 audit (low priority) |

These strategies' V1-era pre-reg directives still exist in [directives/](../../directives/), but the V3.6 protocol (5-axis matrix + L17 relative-MC + Varma noise robustness + L52 hybrid sweep) was not applied. **Treat their advertised Sharpes as informational only** until each gets a fresh V3.6 audit + guide in this folder.

## Retired strategies (V3.6 RETIRED verdict)

The following strategies were audited under V3.6 and RETIRED. See [.tmp/dashboard/dashboard.html](../../.tmp/dashboard/dashboard.html) for the full registry (21 audits, 13 retired as of 2026-05-16):

- B2 Carver EWMAC universal-trend (on cross-asset 21y data) — RETIRED via B2 → B2b → B2c → B2d → I1 chain (L48-L51).
- B4 TSMOM (3 variants) — RETIRED on L43 plateau-fragility.
- D2 carry — RETIRED on data-quality + signal absence.
- E1/E1b VRP — RETIRED.
- G4 overnight — RETIRED.
- A1 residual momentum — RETIRED on plateau-pre-flight.
- I1 HMM regime gate — RETIRED on L51 (HMM-on-daily-returns degenerates to no-op).
- **etf_trend SPY** — RETIRED on L56 (2026-05-16).

Retired strategies are NOT documented in this folder — their lessons are captured in [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md).

## Strategies documented elsewhere (TODO: migrate)

The following V1-era strategy guides still live in `directives/`. They should be moved into this folder ONLY AFTER the underlying strategy has been re-audited under V3.6 (a V1-era guide here would mislead operators into trusting V1 numbers):

- **Samir-Stack** — [directives/Samir-Stack Strategy Guide.md](../../directives/Samir-Stack%20Strategy%20Guide.md)
- **ORB** — [directives/ORB Strategy User Guide.md](../../directives/ORB%20Strategy%20User%20Guide.md)
- **MTF** — [directives/MTF Strategy User Guide.md](../../directives/MTF%20Strategy%20User%20Guide.md)
- **Samir V3 (VIX-HMM)** — [directives/Samir V3 — VIX-HMM Strategy Design 2026-05-13.md](../../directives/Samir%20V3%20%E2%80%94%20VIX-HMM%20Strategy%20Design%202026-05-13.md)

## Related reading

- **Methodology + decision matrix:** [directives/Methodology Audit & Unified Framework 2026-05-14.md](../../directives/Methodology%20Audit%20%26%20Unified%20Framework%202026-05-14.md)
- **Lessons catalogue (L01–L57):** [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md)
- **V1-era re-audit roster:** [directives/V1-era Re-audit Sweep Roster 2026-05-16.md](../../directives/V1-era%20Re-audit%20Sweep%20Roster%202026-05-16.md)
- **L52 hybrid framework (sweep + plateau + audit):** [memory/reference_hybrid_workflow.md](../../) (auto-loaded into Claude memory)
- **Strategy backlog (38 candidate ideas):** [directives/Strategy Backlog 2026-05-14.md](../../directives/Strategy%20Backlog%202026-05-14.md)
- **Docker paper-trading runbook:** [directives/Docker Paper Trading Guide.md](../../directives/Docker%20Paper%20Trading%20Guide.md)
- **Audit dashboard:** [.tmp/dashboard/dashboard.html](../../.tmp/dashboard/dashboard.html) (21 audits, regenerated by [scripts/build_audit_dashboard.py](../../scripts/build_audit_dashboard.py))
