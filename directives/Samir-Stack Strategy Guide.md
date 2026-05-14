# Samir-Stack Strategy Guide

**Version:** 2.1 | **Last Updated:** 2026-05-13
**Status:** Phase 5-validated, Phase 6 paper-deployment-ready (without capitulation overlay; see §10.3)
**Source:** `research/samir_stack/` → `titan/strategies/samir_stack/`

---

## v2.1 history note

**Equity sleeve switched from 3USL.LSEETF (3× daily-rebalanced UCITS ETF)
to MES futures** to avoid the L²-compounded volatility drag of
daily-reset leveraged products. All other Phase 5 findings unchanged
(capitulation overlay deferred to Phase 6b, L_max=2 still optimal,
IGLT bond sleeve kept GBP-native, vol-targeting disabled).

Phase 5 sweep ranking for `futures + IGLT + capitulation` cells:
- `ew=0.40 L=2` → **Calmar CI lo 0.137** (this deployment)
- `ew=0.40 L=3` → Calmar CI lo 0.107
- `ew=0.30 L=4` → Calmar CI lo 0.079

L=2 dominates higher leverage due to vol drag at the strategy level.
The operator can move toward L=3 if more aggressive sizing is desired.

## v2.0 history note

This guide was completely rewritten on 2026-05-13 to describe the
deployable strategy from Phase 5 of the 2026-05-12 audit remediation
(remediation plan deleted in the V2.0 cleanse — lessons live in
`V3.6 Lessons Catalogue.md` A1-A11 + V3.1-V3.6). The
**v1.0 guide described a 40/60 + capitulation variant** that produced
9.13% CAGR honestly. The **previously-deployed live config** was a
10/90 + 8% vol-target variant that the external audit found
delivered honest Sharpe 0.64 / CAGR 5% — not the claimed Sharpe 2.28
/ CAGR 20% — due to look-ahead bias in `bond_rotation_returns` and
the opt-in EFA overlay, plus a dividend double-count in
`futures_returns`. The v2.x deployment is **neither v1.0 nor the
prior live config**; it is the mechanically-selected Phase 5 sweep
winner.

---

## Executive Summary

Samir-Stack v2.0 is a **regime-gated leveraged-equity-plus-bond stack** designed for IBKR UK paper deployment. It implements **Samir Varma's binary risk-classification framework** with two refinements: a multi-indicator regime ensemble (6 indicators), and a tier ladder for graduated leverage when the regime is benign.

**Phase 5 validation (2010-2026, 14y, futures + IGLT + capitulation enabled):**
- Stitched OOS Sharpe: **0.940** (CI lo 0.407)
- Calmar CI lo: **0.137**
- Sanctuary Sharpe: 1.02
- 2022 cumulative return: -22.1%
- Underlying-resampled MC P(MaxDD > 50%): **< 1%** ✓ (RoR-acceptable)

> [!NOTE]
> **V2.0 status (2026-05-14):** The numbers above were the Phase 5 sweep result
> at the time of writing. Post-cleanse, this strategy still needs a fresh
> framework-audited pre-reg + result log under `titan.research.framework`
> (classify as `CROSS_ASSET_MOMENTUM` + overlay) before its CI / DSR / MC verdict
> can be re-trusted as deployment evidence. Current verdict: `tier=unconfirmed`.

**Live deployment (this guide, capitulation deferred to Phase 6b):**
- Expected stitched Sharpe: ~0.88 (vs 0.94 with capitulation)
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

### 3.1 Live sizing: tier ↔ position size (MES futures path)

The strategy targets a notional exposure of ``equity_weight × target_tier × NAV`` (with ``equity_native_leverage = 1`` for futures). The sizing path floors this to integer MES contracts:

```
contracts = floor(target_notional_usd / (futures_multiplier × spx_price))
```

At the Phase 5 GBP-clean champion config (ew=0.40, L_max=2) with MES at the current ~5500 SPX index price ($27.5k notional per contract):

| Regime tier | Target notional (USD, % of NAV) | MES contracts at £30k NAV | Effective SPX exposure |
|---|---:|---:|---:|
| tier 0 (cash) | 0% | 0 | 0% |
| tier 1 | 40% × 1 = **40%** | floor(£12k × 1/0.78 USD / $27.5k) = **0** (under-fill at small NAV) | 0% |
| tier 2 (L_max) | 40% × 2 = **80%** | floor(£24k × 1/0.78 USD / $27.5k) ≈ **1** | ~$27.5k / £30k × 0.78 ≈ **71%** SPX |

At larger NAV (£100k+) the chunkiness shrinks and the live position tracks the target notional within a few percent — see Phase 4 Part C for the empirical drag-vs-NAV curve.

**Operational floor: NAV ≥ £30k** is the recommended minimum for the futures variant.

For deployments using ETFs instead (lower NAV, no futures-rollover overhead):

- **3USL.LSEETF (3× UCITS):** set `equity_is_future=False`, `equity_native_leverage=3.0`. Position size becomes `equity_weight × (tier / 3)` of NAV in 3USL.
- **CSPX or SPY (1× ETF):** set `equity_is_future=False`, `equity_native_leverage=1.0`. Position size = `equity_weight × tier`. At tier=2 this would mean 80% of NAV in CSPX — no margin needed since 0.80 < 1.0.
- **SSO (2× SPY ETF):** set `equity_is_future=False`, `equity_native_leverage=2.0`. tier=2 → 40% × 1 = 40% in SSO → 80% effective SPX.

### 3.2 Cost-model comparison: MES futures vs. leveraged-ETF wrapper

**Why the change from 3USL to MES.** A daily-rebalanced leveraged ETF (3USL) incurs L²-compounded volatility drag every day the underlying moves. Over a 23-year backtest at L=3 this drag is the *dominant* component of carry cost (~14% of the 15% total). MES futures don't rebalance daily — they roll quarterly — so the same target exposure is achieved with a fundamentally cheaper cost structure.

Measured on the full SPY TR series (2003-04 → 2026-04, 22.97y) per Phase 4 Part A:

| L | Synthetic 3x ETF CAGR | MES Futures CAGR | Futures advantage |
|---:|---:|---:|---:|
| 1 | 11.27% | 11.42% | +0.15pp |
| 2 | 16.53% | 17.73% | **+1.20pp/yr** |
| 3 | 18.67% | 20.05% | **+1.39pp/yr** |

The gap widens with leverage because the ETF wrapper's vol-drag scales with L².

**Component-by-component decomposition** (at L=2, the deployment leverage cap):

| Cost component | Synthetic 3× ETF (e.g. 3USL) | MES futures (this deployment) | Net to MES |
|---|---|---|---:|
| **Daily-rebalance vol drag** | ½ × L(L-1) × σ² ≈ ½×2×18²/100 ≈ **+3.2%/yr at L=2** | **none** (no daily rebal) | **−3.2pp** |
| **Wholesale funding on borrow** | (L-1) × ~Fed Funds ≈ **+2%/yr** | implicit in basis = funding − div ≈ **+0.5%/yr × L = +1%/yr** | **−1.0pp** |
| **TER (wrapper fee)** | **0.75%/yr** (3USL) | **0%** (no wrapper) | **−0.75pp** |
| **Quarterly roll slippage** | n/a | **~0.2%/yr × L = +0.4%/yr at L=2** | +0.4pp |
| **T-bill earned on cash equity** | **0** (you hold the ETF, not cash) | ~funding × (NAV − posted_IM) → **−~3.5%/yr CREDIT at recent rates** | **−3.5pp** |
| **Commissions** | ETF: ~5bp per rebalance, ~10bp/yr | MES: ~$1.40 round trip × ~30 trades/yr at £30k NAV → ~14bp/yr | comparable |
| **Tracking error / counterparty** | small (~10-30bp) | n/a | small |
| **Theoretical net cost gap @ L=2** | | | **≈ -7-8pp/yr** advantage for MES |

The empirical 1.2pp/yr gap is *less* than the theoretical 7-8pp because:
- The vol-drag estimate ½×L(L-1)×σ² is an approximation; the realised drag on SPY TR over 23y is lower because of mean-reversion in returns.
- The T-bill credit on cash is partially offset by the basis premium on the futures position (in normalised steady state).
- Commission and slippage are higher in practice than the theoretical models suggest.

**What the live deployment actually pays** (best estimate, at the Phase 5 champion config £30k NAV ew=0.40 L_max=2 with avg deployed leverage ~1.4 across regimes):

| Cost | Estimate | Notes |
|---|---:|---|
| Basis decay on equity sleeve | ~0.4-0.6%/yr | funding ≈ div in recent regimes |
| Quarterly MES roll slippage | ~0.10%/yr | ~14bp on the 40% sleeve × ~1.4 avg L |
| IBKR commissions (MES + IGLT) | ~0.20%/yr | $1.40 r/t × 30 fills + IGLT commissions |
| T-bill on free cash | **−2-3%/yr CREDIT** | IBKR pays ~4% on USD cash > $10k |
| **Net carry cost (futures path)** | **roughly NEUTRAL to ~+0.5%/yr** drag | |

The futures variant is essentially free-carry under the current high-rate regime. The strategy's expected return ≈ regime-gated SPY TR exposure × 1.4 + IGLT carry, minus a thin operational drag.

> [!NOTE]
> The cost components above are NOT separately implemented in the live strategy class — the broker handles funding, dividends, basis, and commissions transparently through the futures-contract market price and the IBKR cash-yield mechanism. The strategy class only submits orders; the cost realisation flows through P&L automatically. The numbers above describe what the operator should *expect* to see, not what code computes.

### 3.3 Why daily-reset leveraged ETFs were rejected

Operator decision (2026-05-13): MES futures preferred over 3USL.LSEETF because:

1. **Vol drag is structural and unavoidable in daily-reset wrappers.** At L=3 it costs ~14%/yr in vol-drag alone. The same effective exposure via futures costs ~0.5%/yr.
2. **Path dependence.** A daily-reset 3× ETF held for 1 year does NOT give 3× the underlying's 1-year return — it gives 3× the *daily compounded* return, which diverges substantially in choppy markets. Futures held to expiry (or rolled) track the spot index much more cleanly.
3. **Operational simplicity** at the contract level. One MES contract = one P&L line. A leveraged ETF position incurs implicit daily rebalances on the issuer's side, creating exposure to the issuer's counterparty (swap-based wrappers) and tracking error.
4. **Funding regime sensitivity.** A 3× ETF pays wholesale Fed Funds on the borrow. Futures pay a basis that nets to funding − dividend, which at current rates is much smaller. The MES advantage widens further if rates rise.

The cost of moving to futures: quarterly contract rollover (manual operator update of `equity_instrument_id`) and a £30k NAV operational floor (Phase 4 Part C).

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

The static `fx_rate_equity_quote_to_base` config drifts from spot. Update monthly during paper, weekly when live, or wire dynamic GBPUSD bar subscription as v2.2.

### 8.3 Quarterly MES contract rollover

MES futures expire on the third Friday of March/June/September/December (month codes H/M/U/Z). The operator must update the strategy's `equity_instrument_id` config and restart the strategy ~8 business days before expiry to roll into the next contract.

**Standard quarterly roll schedule** (third Friday of expiry month, roll the prior week):

| Current contract | Expires | Roll target | Roll on or before |
|---|---|---|---|
| MESM26 | 2026-06-19 | MESU26 (Sep) | 2026-06-09 |
| MESU26 | 2026-09-18 | MESZ26 (Dec) | 2026-09-08 |
| MESZ26 | 2026-12-18 | MESH27 (Mar) | 2026-12-08 |

**Roll procedure:**

1. ~8 business days before expiry, edit `config/samir_stack.toml`:
   ```toml
   equity_instrument_id = "MESU26.CME"   # next contract
   bar_type_equity_d    = "MESU26.CME-1-DAY-LAST-EXTERNAL"
   ```
2. Verify the new contract is tradable on the IB Gateway:
   ```bash
   docker compose exec titan-portfolio uv run python -c "
   from titan.adapters.ibkr import resolve_contract
   print(resolve_contract('MESU26.CME'))
   "
   ```
3. Restart the strategy:
   ```bash
   docker compose restart titan-portfolio
   ```
4. Confirm the existing MES position is rolled (the strategy will see the new bar type, treat the prior position as needing a flatten + open at the new contract, and submit the resulting orders on the next bar).
5. **Cost of one roll: ~5bp of notional** (Phase 4 cost-model assumption). At ~£30k NAV × 80% × 1 roll, that's ~£12 per quarterly roll = ~£48/yr.

> [!CAUTION]
> Forgetting to roll before expiry causes a forced cash settlement and re-entry — **2-3 days of zero exposure plus an extra round-trip**. Set a calendar reminder for each quarter's roll-by date. If this overhead becomes painful, future work could implement automated rollover detection.

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
- `directives/V3.6 Lessons Catalogue.md` — distilled lessons from the May 2026 Samir-Stack audit + V3 re-derivation (A1-A11, V3.1-V3.6) live here now. The original `Samir-Stack Remediation Plan 2026-05-12.md` was removed in the V2.0 cleanse — its lessons survive in this catalogue.
- `directives/Samir V3 — VIX-HMM Strategy Design 2026-05-13.md` — V3 design directive (kept).
- `directives/Methodology Audit & Unified Framework 2026-05-14.md` — the unified framework that supersedes the per-audit remediation plans.
- `directives/IC Signal Analysis.md` — IC validation methodology.
- `references/portfolio-risk-architecture.md` — PRM contract.
- `references/research-math-guardrails.md` — research math discipline.
- `directives/Deployment & Operations.md` — kill switch + halt procedures.

**External:**
- Samir Varma — risk-classification framework (binary regime model).
- Asness, Moskowitz — 12-1 momentum convention (used in `momentum_12_1` indicator).
- Bailey & López de Prado 2014 — Deflated Sharpe Ratio (used in Phase 5 selection).
- Politis & Romano 1994 — Stationary bootstrap (used in MC).
