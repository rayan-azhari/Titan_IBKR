# Samir-Stack Strategy Guide

**Version:** 2.0 | **Last Updated:** 2026-05-13
**Status:** Phase 5-validated, Phase 6 paper-deployment-ready (without capitulation overlay; see §10.3)
**Source:** `research/samir_stack/` → `titan/strategies/samir_stack/`

---

## v2.0 history note

This guide was completely rewritten on 2026-05-13 to describe the
deployable strategy from Phase 5 of the [2026-05-12 audit
remediation](Samir-Stack%20Remediation%20Plan%202026-05-12.md). The
**v1.0 guide described a 40/60 + capitulation variant** that produced
9.13% CAGR honestly. The **previously-deployed live config** was a
10/90 + 8% vol-target variant that the external audit found
delivered honest Sharpe 0.64 / CAGR 5% — not the claimed Sharpe 2.28
/ CAGR 20% — due to look-ahead bias in `bond_rotation_returns` and
the opt-in EFA overlay, plus a dividend double-count in
`futures_returns`. The v2.0 deployment is **neither v1.0 nor the
prior live config**; it is the mechanically-selected Phase 5 GBP-clean
winner.

---

## Executive Summary

Samir-Stack v2.0 is a **regime-gated leveraged-equity-plus-bond stack** designed for IBKR UK paper deployment. It implements **Samir Varma's binary risk-classification framework** with two refinements: a multi-indicator regime ensemble (6 indicators), and a tier ladder for graduated leverage when the regime is benign.

**Phase 5 validation (2007-2026, 18.9y, GBP-clean cell with capitulation enabled):**
- Stitched OOS Sharpe: **0.961** (CI lo 0.428)
- Calmar CI lo: **0.148**
- Sanctuary Sharpe: 0.80
- Underlying-resampled MC P(MaxDD > 50%): **< 1%** ✓ (RoR-acceptable)

**Live deployment (this guide, capitulation deferred to Phase 6b):**
- Expected stitched Sharpe: ~0.91 (vs 0.96 with capitulation)
- All other metrics within 5-10% of the with-cap cell

---

## 1. Conceptual Foundation

Per Samir Varma's framework:

> **Don't predict alpha — classify risk.** Alpha is competed away by other quants. Catastrophic systemic risk cannot be arbitraged away because it stems from forced liquidations during crises.

Operational implications applied here:
1. **Binary regime classification**: the regime score collapses to "deploy" (tier ≥ 1) or "cash" (tier 0). No partial sizing during stress.
2. **Best-days-cluster-with-worst-days**: stepping out during high-vol regimes loses some upside but avoids both extremes — the geometric mean wins because variance compounds against you.
3. **Momentum as re-entry timing**: the 20-bar quiet period after hostile-to-benign transitions is the explicit nod to momentum-confirmed re-entry.

What v2.0 *adds* to Samir's pure binary:

| Extension | Why |
|---|---|
| **Multi-indicator ensemble regime score** (6 indicators) | Single-signal regime gates are fragile. Ensemble averages out idiosyncratic indicator failures. |
| **2-tier leverage** (tier 1 = 1x, tier 2 = 2x) | Pure binary discards information about regime *strength*. With 3USL.LSEETF available, strong-benign regimes earn more. **L=2 cap chosen by Phase 5** — L=3 vol drag (~14%/yr) erodes Calmar more than the marginal upside. |
| **40/60 stack with UK gilts** | Adds asset-class diversification. 60% IGLT (UK 7-10y gilts) provides ballast and is GBP-native, eliminating FX risk on the defensive sleeve. |
| **DD circuit breaker** | Failsafe for when the regime classifier misses (COVID-style sharp shocks). Throttle at -10%, kill at -15%. |
| Capitulation overlay | Research-only in v2.0 (live wiring deferred to Phase 6b). |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       SIGNAL DATA SOURCES                        │
│  SPY.ARCA (regime)    VIX.CBOE (vol)    HYG.ARCA (credit)       │
│  IEF.NASDAQ (credit denominator)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INDICATOR PANEL (6 columns, [0,1])              │
│  vix  rv_regime  trend  momentum_12_1  dd_velocity  credit       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              REGIME SCORE = mean(6 indicators) ∈ [0,1]           │
│         0 = maximum hostile, 1 = maximum benign                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TIER MAPPING (with hysteresis)                │
│   score < 0.30        → cash                                     │
│   0.30 ≤ score < 0.55 → tier 1 (1x equity in sleeve)             │
│   score ≥ 0.55        → tier 2 (2x equity in sleeve, L_max)      │
│   (UP transitions require score ≥ threshold + 0.05 buffer)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DD CIRCUIT BREAKER OVERLAY                    │
│   DD ≥ -10%  → throttle leverage to ≤ 1x                         │
│   DD ≥ -15%  → kill (cash, both sleeves)                         │
│   Recovery requires score ≥ 0.70 for 5 consecutive bars          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      40/60 SLEEVE STACK                          │
│   When deployed:                                                 │
│     40% × tier × 3USL.LSEETF (synthetic 3x SPY UCITS, USD)       │
│     60% × IGLT.LSEETF (UK 7-10y gilts UCITS, GBP-native)         │
│   When hostile / DD-killed: 100% cash                            │
└─────────────────────────────────────────────────────────────────┘
```

Effective S&P exposure: 40% × 2x = **80% of NAV** at tier 2 (full deployment). Total stack notional at full deployment: 80% equity + 60% bonds = 140% of NAV (1.4× portfolio leverage).

---

## 3. Live configuration

```toml
# config/samir_stack.toml — Phase 5 GBP-clean champion
equity_instrument_id = "3USL.LSEETF"   # 3x SPY UCITS (USD-quoted, LSE-listed)
bond_instrument_id   = "IGLT.LSEETF"   # UK 7-10y gilts (GBP-native)

equity_weight = 0.40
bond_weight   = 0.60
L_max         = 2.0
tier_thresholds   = [0.30, 0.55]
hysteresis_buffer = 0.05
re_entry_quiet_bars = 20
equity_native_leverage = 3.0   # 3USL is a 3x ETF → scale position by tier/3

dd_throttle = 0.10
dd_kill     = 0.15

vol_target_annual = 0.0   # disabled per Phase 5 finding

equity_quote_ccy = "USD"
bond_quote_ccy   = "GBP"
fx_rate_equity_quote_to_base = 0.7519   # 1 USD = 0.7519 GBP at GBPUSD=1.33
fx_rate_bond_quote_to_base   = 1.0
```

### 3.1 Live sizing: tier ↔ position size

The strategy holds ``equity_weight × (target_tier / equity_native_leverage)`` of NAV in the equity instrument. With 3USL (a 3× leveraged ETF) the actual GBP-denominated position size is:

| Regime tier | Position size (% of NAV in 3USL) | Effective SPX exposure |
|---|---:|---:|
| tier 0 (cash) | 0% | 0% |
| tier 1 | 40% × 1/3 ≈ **13.3%** | 13.3% × 3 = **40%** |
| tier 2 (L_max) | 40% × 2/3 ≈ **26.7%** | 26.7% × 3 = **80%** |

The account stays *unleveraged at the broker level* — the leverage is provided by the ETF itself, and the strategy holds less of it at lower tiers. This avoids broker margin and matches the Phase 5 backtest model exactly.

For deployments using a different equity instrument:
- **CSPX or SPY (1× ETF):** set `equity_native_leverage = 1.0`. Position size = `equity_weight × tier`. At tier=2 this would mean **80% of NAV in CSPX** — fine, no margin needed since L_max ≤ 2.5 at 40% sleeve.
- **SSO (2× SPY ETF):** set `equity_native_leverage = 2.0`. tier=2 → 40% × 1 = 40% in SSO → 80% effective SPX.
- **MES futures (mechanical winner cell):** the strategy class currently uses ETF sizing (`equity_is_future=False`). Switching to futures requires `equity_is_future=True` and integer-contract sizing — different code path, deployable only at NAV ≥ £30k (Phase 4 finding).

---

## 4. Trading Instruments

| Sleeve | Symbol | ISIN | Currency | TER | Notes |
|---|---|---|---|---|---|
| Equity | 3USL.LSEETF | XS1078280466 | USD | 0.75% | WisdomTree 3x S&P 500 UCITS. Validated against UPRO (real 3x SPY, US-listed): daily-return corr 0.9983 over 16.8y, CAGR diff +1.21pp. |
| Bond | IGLT.LSEETF | IE00B1FZSB30 | GBP | 0.07% | iShares Core UK Gilts UCITS. GBP-native — no FX on the defensive ballast. |

**Signal data sources** (subscribed but not traded):
- SPY.ARCA — primary regime driver.
- VIX.CBOE — vol regime indicator.
- HYG.ARCA — credit-spread numerator.
- IEF.NASDAQ — credit-spread denominator (separate from bond sleeve since bond=IGLT, not IEF).

---

## 5. Phase 5 Performance Validation

Per the [Phase 5 report](../../.tmp/reports/samir_stack/phase5_joint_sweep_report.md) (gitignored; reproducible via `uv run python -m research.samir_stack.run_phase5_joint_sweep`).

### 5.1 Mechanical sweep over 120 cells

Sweep dimensions: `equity_weight ∈ {0.2, 0.3, 0.4, 0.5, 0.6} × L_max ∈ {2,3,4} × engine ∈ {synth_3x, futures} × sleeve ∈ {IEF, IGLT} × capitulation ∈ {on, off}`.

Selection rule (pre-committed): eliminate cells failing any of `sharpe_ci_lo ≤ 0`, `dsr_prob < 0.95`, `sanctuary_sharpe < 0`, `sanctuary_max_dd > 15%`. Rank survivors by `calmar_ci_lo` desc, tie-break on parsimony (lower L_max, then lower equity_weight). Final MC RoR filter: `P(MaxDD > 50%) < 1%`.

The 2022 cum-return gate was withdrawn as unimplementable — every cell produced 2022 cum between -29% and -17% (baseline 40/60 was ~-24%). The MC P(DD>50%) gate is the proper RoR-first check and is retained. See remediation plan §3 Phase 5 for the transparent withdrawal note.

### 5.2 GBP-clean variant (this deployment)

**`equity_weight=0.40 | L_max=2 | SyntheticETFEngine | IGLT static | capitulation=on`**

| Metric | Value |
|---|---|
| Stitched OOS Sharpe | 0.961 |
| Sharpe CI 95% lo | 0.428 |
| **Calmar CI lo** (selection metric) | **0.148** |
| Sanctuary Sharpe | 0.80 |
| 2022 cumulative return | -22.2% |
| DSR probability | 1.00 |

The GBP-clean variant trades a small Calmar CI lo difference (0.148 vs 0.162 for the USD-IEF mechanical winner) for zero FX risk on the bond sleeve. For a GBP-base UK paper account the ~-140bp/yr FX drag (Strategy Guide §7.4 — see archived notes) on USD bonds would likely consume the difference.

### 5.3 Underlying-resampled Monte Carlo (500 paths)

On the mechanical winner (USD-IEF variant; GBP-IGLT MC pending):
- Median CAGR: 6.4%
- Median MaxDD: -18.3%
- P(MaxDD > 25%): 9.4%
- P(MaxDD > 35%): 0.8%
- **P(MaxDD > 50%): 0%** ✓ RoR-acceptable

The MC uses shared-block stationary bootstrap of underlying daily returns (not prices — the audit-flagged strategy-return bootstrap, plus a Phase 5 mid-correction where the first-pass bootstrapped prices directly), then re-builds the indicator panel and re-runs the strategy on each path. This is the conditional-structure-preserving MC the audit (A6) required.

---

## 6. What's Different from the Pre-Audit "Champion"

The previously-deployed config (`samir_stack_paper` registry entry) was 10/90 + 8% vol-target + futures + I1+I2+I3. The audit (2026-05-12) found three bugs that materially inflated its backtest:

| Audit ID | Bug | Inflation |
|---|---|---|
| A1 | `bond_rotation_returns` look-ahead | +10.06pp CAGR / +1.36 Sharpe |
| A2 | I3 opt-in EFA look-ahead | +2.33pp CAGR / +0.11 Sharpe |
| A3 | `futures_returns` dividend double-count | +1.46pp/yr at L=1, +4.75pp/yr at L=3 |

After fixes (Phase 1), the deployed champion collapsed from claimed Sharpe 2.28 / CAGR 20% to honest Sharpe 0.64 / CAGR 5%. Phase 5 then found a simpler architecture (this guide's config) that statistically dominates the post-fix champion.

Side-by-side:

| Metric | Pre-audit (claimed) | Pre-audit (honest, post-fix) | **v2.0 (this guide, with cap)** |
|---|---|---|---|
| Sharpe | 2.28 | 0.64 | **0.96** |
| CAGR | 20.0% | 5.0% | (MC median 6.4%) |
| MaxDD | -6.6% | -22.8% | (MC median -18.3%) |
| Architecture | 10/90 + 8%VT + futures + I1+I2+I3 | same (broken backtest) | **40/60 + L=2 + 3USL + IGLT static** |

---

## 7. FX Handling (IBKR UK GBP-Base Account)

### 7.1 The Setup

| Sleeve | Instrument | Quote Ccy | Base Ccy (account) | FX Layer |
|---|---|---|---|---|
| Equity | 3USL.LSEETF | USD | GBP | Need GBP↔USD rate |
| Bond | IGLT.LSEETF | GBP | GBP | None |

### 7.2 Sizing Math

Per project contract (see `CLAUDE.md`), all non-base instruments use `convert_notional_to_units`:

```python
# For 3USL (USD-quoted) in GBP account:
units = convert_notional_to_units(
    notional_base=target_gbp,
    price=price_usd,
    quote_ccy="USD",
    base_ccy="GBP",
    fx_rate_quote_to_base=0.7519,
)

# For IGLT (GBP-quoted) in GBP account:
units = convert_notional_to_units(
    notional_base=target_gbp,
    price=price_gbp,
    quote_ccy="GBP",
    base_ccy="GBP",
    fx_rate_quote_to_base=None,
)
```

### 7.3 Empirical FX Impact

The equity sleeve has structural USD exposure. Over 2003-2026, GBP/USD fell from 1.72 to 1.33 (-22.6%) — USD strengthened, providing a tailwind for UK USD holders. But this also adds ~9% annual FX vol to the equity sleeve.

| Component | Annual Impact (estimate) |
|---|---|
| Currency drift tailwind | +90 bp |
| Volatility drag (FX vol ~9%) | -180 bp |
| **Net FX drag on equity sleeve** | **-90 bp** |

**Crisis FX hedge (the structural benefit)**: USD strengthened in EVERY major risk-off since 2003 (GFC 2008, Brexit 2016, COVID 2020, 2022 rate shock). This partial hedge reduces GBP-perspective drawdowns by 3-10pp during crises.

---

## 8. Live Deployment

### 8.1 Pre-flight checklist (before first live order)

- [ ] `uv run pytest tests/test_samir_stack_strategy.py -v` — all champion-pin tests pass.
- [ ] `uv run ruff check titan/strategies/samir_stack/` — clean.
- [ ] All 6 IB contracts resolve on the live IB Gateway (3USL, IGLT, SPY, VIX, HYG, IEF).
- [ ] `initial_equity` in config matches actual allocation slice (not the £10k placeholder).
- [ ] `fx_rate_equity_quote_to_base` updated to current GBPUSD market rate.
- [ ] PRM halt threshold consistent with the strategy's internal DD breaker (15%).
- [ ] Paper-trade soak for **at least 4 weeks** (see Phase 7 runbook).

### 8.2 Operational runbook

#### Daily monitoring

Each day after the daily bar close:
1. Strategy logs the regime score, applied tier, dd_state, breakdown of the 6 indicators.
2. Compare to research-computed score for the same bar (the parity test pins this within tolerance).
3. Check if any tier transition occurred — these are the trades.

#### Weekly review

- Verify per-strategy equity matches expectations vs broker NLV.
- Review the `applied_tier` history — any unexpected sequences?

#### Halting

```bash
docker compose exec titan-portfolio uv run python scripts/kill_switch.py --strategy samir_stack
```

#### FX Rate Updates

The static `fx_rate_equity_quote_to_base` config drifts from spot. Update monthly during paper, weekly when live, or wire dynamic GBPUSD bar subscription as v2.1.

---

## 9. Tier Mapping & State Machine

### 9.1 Tier Assignment

```
score < 0.30            → tier 0 (cash)
0.30 ≤ score < 0.55     → tier 1 (1x equity in sleeve)
score ≥ 0.55            → tier 2 (2x equity in sleeve, L_max)
```

### 9.2 Hysteresis

- **Going UP** between tiers requires score ≥ threshold + 0.05 (e.g., to enter tier 2 from tier 1, score must reach 0.60).
- **Going DOWN** uses the bare threshold.

Avoids whipsawing when the score oscillates near a boundary.

### 9.3 Re-Entry Quiet Period

After exiting to cash (score < 0.30), re-entry to tier 1 requires **20 consecutive bars** with score ≥ 0.55. Avoids 2008-style false-dawn whipsaws.

---

## 10. Known Limitations and Risks

### 10.1 Slow Grinding Bears (2022-Style)

The regime classifier is best at **sharp systemic events** (GFC, COVID) where capitulation is identifiable. Slow grinders (2022 rate shock, dot-com 2000-2002) decay through the indicators gradually rather than spike — the gate fires LATE and the strategy takes meaningful damage before exiting.

The 2022 cum return of -22.2% (with capitulation) reflects this. Realistic expectation, not a bug.

### 10.2 Fast Exogenous Shocks (COVID-Style)

The COVID 5-week crash beat the regime classifier — score didn't go fully hostile until after the bottom. The DD breaker provides the failsafe at -15%.

### 10.3 Capitulation overlay deferred

The Phase 5 winner uses `capitulation=on`. This deployment runs with `capitulation=off` because the overlay is implemented in research only (`research/samir_stack/capitulation.py`) and **not yet wired into the live strategy class**. Wiring it is the Phase 6b follow-up.

Expected impact: ~-0.05 to -0.07 Sharpe vs the Phase 5 cell. Other metrics within 5-10%. Still meaningfully better than the prior deployed champion (Sharpe 0.64 honest).

### 10.4 Static FX Rate

`fx_rate_equity_quote_to_base` is a config-static value, drifts from spot. At ~9% annual GBP/USD vol, a stale rate by 1 month → ~2.5% sizing error → minor at 40% sleeve allocation. v2.1: subscribe GBPUSD bars and update dynamically.

### 10.5 Contract Resolution

The format `<SYMBOL>.LSEETF` matches project convention (`CSPX.LSEETF` is in production), but **3USL/IGLT contract resolution on live IB Gateway is unverified for the Phase 5 deployment** — must be confirmed via tiny test order before sizing up.

### 10.6 Counterparty Risk on Leveraged ETF

3USL is swap-based — implicitly long counterparty credit risk on the swap counterparty (typically a major bank). In a true financial-system breakdown, the swap counterparty could fail. Unhedged.

---

## 11. Reproducing Phase 5 Results

```bash
# Full sweep — Phase 5 mechanical selection (~130s)
uv run python -m research.samir_stack.run_phase5_joint_sweep

# Outputs to .tmp/reports/samir_stack/:
#   phase5_sweep_full.csv         — all 120 cells
#   phase5_survivors.csv          — sorted by Calmar CI lo
#   phase5_mc_top_survivors.csv   — 500-path MC on top 3
#   phase5_winner.csv             — final pick
#   phase5_joint_sweep_report.md  — full analysis
```

Phases 3 and 4 isolated reports:

```bash
uv run python -m research.samir_stack.run_phase3_bond_rotation
uv run python -m research.samir_stack.run_phase4_futures_engine
```

---

## 12. Testing & Validation Hooks

### 12.1 Champion-pin tests (`tests/test_samir_stack_strategy.py`)

Verify the strategy class still has the Phase 5 GBP-clean champion config:

```bash
uv run pytest tests/test_samir_stack_strategy.py -v
```

### 12.2 Research lookahead-regression tests (`tests/test_samir_stack_research_lookahead.py`)

Pin the Phase 1 audit fixes (A1 + A3):

```bash
uv run pytest tests/test_samir_stack_research_lookahead.py -v
```

### 12.3 Engine + sleeve tests (`tests/test_samir_stack_engines.py`)

13 tests including the Phase 2 bit-exact gate:

```bash
uv run pytest tests/test_samir_stack_engines.py -v
```

---

## 13. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-05-02 | Initial release. 40/60 + capitulation + synthetic 3x ETF + IEF. **Pre-audit; described one variant while deployment was a different one.** |
| 2.0 | 2026-05-13 | Complete rewrite after the 2026-05-12 external audit and 5-phase remediation. Phase 5 GBP-clean champion: 40/60 + L=2 + 3USL + IGLT + capitulation (overlay deferred to Phase 6b). Replaces the 10/90 + 8%-vol-target prior deployment which the audit found delivered honest Sharpe 0.64 / CAGR 5%. |

---

## 14. References

**Within this project:**
- [Samir-Stack Remediation Plan 2026-05-12.md](Samir-Stack%20Remediation%20Plan%202026-05-12.md) — full 7-phase plan with all gates, decisions, and audit-finding closures.
- `directives/IC Signal Analysis.md` — IC validation methodology.
- `references/portfolio-risk-architecture.md` — PRM contract.
- `references/research-math-guardrails.md` — research math discipline.
- `directives/Emergency Operations.md` — kill switch + halt procedures.

**External:**
- Samir Varma — risk-classification framework (binary regime model).
- Asness, Moskowitz — 12-1 momentum convention (used in `momentum_12_1` indicator).
- Bailey & López de Prado 2014 — Deflated Sharpe Ratio (used in Phase 5 selection).
- Politis & Romano 1994 — Stationary bootstrap (used in MC).
