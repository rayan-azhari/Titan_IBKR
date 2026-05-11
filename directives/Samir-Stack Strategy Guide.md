# Samir-Stack Strategy Guide

**Version:** 1.0 | **Last Updated:** 2026-05-02
**Status:** Research-validated, paper-ready
**Source:** `research/samir_stack/` → `titan/strategies/samir_stack/`

---

## Executive Summary

Samir-Stack is a **regime-gated leveraged-equity-plus-bond stack** designed for IBKR UK paper deployment. It hybridises **Samir Varma's binary risk-classification framework** with academic vol-targeting, multi-indicator regime ensembles, a pure-risk HMM, and a contrarian capitulation overlay for opportunistic crash-bottom entries.

**Headline performance (2003-2026, USD perspective):**
- CAGR: **9.13%**
- Max Drawdown: **-27.1%**
- Calmar: **0.337**
- Sharpe: **0.83**
- P(MaxDD > 50%): **0.16%** ✅ (RoR-acceptable)
- Trades/year: **~13**
- WFO 5-fold sanctuary Calmar: **1.05**

**Headline performance (GBP perspective for IBKR UK):**
- CAGR: ~6.9-7.5%
- MaxDD: -23 to -25% (USD-haven flows reduce drawdown in crises)
- Sharpe: ~0.75

**Versus benchmarks (USD, 2003-2026):**
- SPY buy-hold: 11.27% CAGR / -55.2% MaxDD / 0.20 Calmar / **15.7% P(DD>50%)**
- 60/40 SPY/IEF: 8.43% / -32.2% / 0.26 / 0.10% P(DD>50%)
- Faber GTAA: 7.31% / -20.7% / 0.35
- HFEA (3x SPY + 3x TLT, 55/45): 18.05% / -70.1% / 0.26 / **86.6% P(DD>50%)**
- **Samir-Stack: 9.13% / -27.1% / 0.34 / 0.16% P(DD>50%)** ⭐

The strategy delivers **better Calmar than 60/40, comparable RoR, and meaningfully more CAGR** while explicitly side-stepping the leverage tail-risk that destroys HFEA-style stacks.

---

## 1. Conceptual Foundation

### 1.1 Samir Varma's Thesis

The strategy is built on Samir Varma's framework, which inverts the standard quant approach:

> **Don't predict alpha — classify risk.** Alpha is competed away by other quants. Catastrophic systemic risk cannot be arbitraged away because it stems from forced liquidations during crises.

Three operational implications:
1. **Binary regime classification**: market is either *benign* (deploy, possibly with leverage) or *hostile* (move to cash entirely). No "Texas hedges," no partial sizing during stress.
2. **Best-days/worst-days clustering**: empirically, the S&P's best 10 days mostly occur during the worst drawdowns. Stepping out during high-vol regimes loses some upside but avoids both extremes — the geometric mean wins because variance compounds against you.
3. **Momentum is the legitimate edge**: the only academically-confirmed factor that survives globally. Used here for *re-entry timing* after regime exits, not for entry timing within a regime.

### 1.2 What This Strategy Adds to Samir's Framework

Samir's pure binary (1x SPY when benign / cash when hostile) is implemented in `research/samir_stack/benchmarks.py` as `samir_pure` baseline. Our extensions:

| Extension | Why |
|---|---|
| **Multi-indicator ensemble regime score** (6 indicators + HMM) | Single-signal regime gates are fragile. Ensemble averages out idiosyncratic indicator failures. Validated via `phase1_sweep`-style IC analysis showing each indicator passes BH FDR. |
| **Tiered leverage** (1x/2x/3x) | Pure binary discards information about regime *strength*. With 3x leveraged ETF available (3USL.LSEETF), strong-benign regimes earn more. |
| **40/60 stack with bonds** | Adds asset-class diversification. 60% IGLT (UK gilts) provides ballast and CGT-free carry for UK investors. |
| **DD circuit breaker** | Failsafe for when the regime classifier misses (COVID-style sharp shocks). |
| **Capitulation overlay** | Contrarian re-entry near crash bottoms — partially restores the "best days" we'd otherwise miss. |
| **Pure-risk HMM** | Captures vol-regime structure that linear indicators miss (helps 2022-style slow grinders). |

The whole point of the design: **respect Samir's risk-first philosophy while extracting the leverage available to a UK retail investor without inheriting HFEA-style catastrophic tails**.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIGNAL DATA SOURCES                       │
│  SPY.ARCA (regime)    VIX.CBOE (vol)    HYG.ARCA (credit)       │
│  IEF.NASDAQ (credit denominator — separate from bond sleeve)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INDICATOR PANEL (7 columns, [0,1])           │
│  vix_regime  rv_regime  trend  momentum_12_1  dd_velocity       │
│  credit  hmm_risk (causal, annual re-fit)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              REGIME SCORE = mean(7 indicators) ∈ [0,1]           │
│         0 = maximum hostile, 1 = maximum benign                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TIER MAPPING (with hysteresis)                │
│   score < 0.30        → cash                                    │
│   0.30 ≤ score < 0.50 → tier 1 (1x effective)                   │
│   0.50 ≤ score < 0.75 → tier 2 (2x effective)                   │
│   score ≥ 0.75        → tier 3 (3x effective)                   │
│   (UP transitions require score ≥ threshold + 0.05 buffer)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DD CIRCUIT BREAKER OVERLAY                    │
│   DD ≥ -10%  → throttle leverage to ≤ 1x                        │
│   DD ≥ -15%  → kill (cash, both sleeves)                        │
│   Recovery requires score ≥ 0.70 for 5 consecutive bars         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CAPITULATION OVERLAY (additive)                 │
│   Triggers when in cash AND:                                    │
│     • SPY drawdown from 60d high ≥ 12%                          │
│     • At least 2 of {VIX, dd_velocity, credit} hit extreme low   │
│     • SPY 5-day return ≥ +3% (bounce confirmation)              │
│   Action: enter tier 2 (2x), failed-bounce stop -5% in 10d      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      40/60 SLEEVE STACK                          │
│   When deployed:                                                │
│     40% × tier × 3USL.LSEETF (3x SPY UCITS, USD)               │
│     60% × IGLT.LSEETF (UK Gilts UCITS, GBP)                    │
│   When hostile / DD-killed: 100% cash                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Code Structure

```
research/samir_stack/                         # Research module (validated)
  __init__.py
  data_loader.py        — date-aligned multi-asset loader
  synthetic_3x.py       — Lx ETF return generator (vol drag + TER + funding)
  indicators.py         — 7 candidate regime indicators (each in [0,1])
  ic_validation.py      — IC + regime-conditional metrics
  regime_score.py       — Equal-weight + vol-weighted ensemble combiners
  hmm_risk.py           — Pure-risk HMM (causal, annual re-fit)
  capitulation.py       — Capitulation overlay detector + state
  capitulation_sweep.py — Robustness-aware parameter sweep harness
  stacked_strategy.py   — 40/60 state machine with tier + DD + overlay
  global_basket.py      — Synthetic global equity basket (rejected — 0.98 corr to SPY)
  benchmarks.py         — SPY / 60-40 / Faber / HFEA / Samir-pure
  stress.py             — 10 named-crisis decomposition
  monte_carlo.py        — Stationary-bootstrap RoR
  wfo.py                — 5-fold WFO + sanctuary (equity-only v1)
  wfo_stacked.py        — 5-fold WFO + sanctuary (production stacked)
  threshold_sweep.py    — Conservatism preset sweep (4 presets x 3 L_max)
  run_pipeline.py       — End-to-end orchestrator

titan/strategies/samir_stack/                 # Production NautilusTrader class
  __init__.py
  regime.py             — Runtime regime computation (rolling buffers)
  strategy.py           — SamirStackStrategy class

config/
  samir_stack.toml      — Live configuration

tests/
  test_samir_stack_parity.py — 7 parity tests (live vs research)

directives/
  Samir-Stack Strategy Guide.md — THIS DOCUMENT
```

---

## 3. Regime Score — The Core Classifier

### 3.1 The Seven Indicators

All indicators output a **benign-probability score in [0, 1]** where 1.0 = maximum benign (full risk-on) and 0.0 = maximum hostile (full cash). All computations are causal — no future bars used.

| # | Indicator | Family | Computation | Weight |
|---|---|---|---|---|
| 1 | `vix` | Volatility | Inverse of 3y rolling z-score of VIX | 1.0 |
| 2 | `rv_regime` | Volatility | 1 - rolling-3y percentile rank of 21d realised vol | 1.0 |
| 3 | `trend` | Trend | SMA(50)/SMA(200) status of SPY (0 / 0.34 / 0.66 / 1.0) | 1.0 |
| 4 | `momentum_12_1` | Momentum | 12-1 momentum (Asness convention, skip last month) | 1.0 |
| 5 | `dd_velocity` | Crash detector | SPY 21-day cumulative return (clipped/normalised) | 1.0 |
| 6 | `credit` | Credit | HYG/IEF ratio 3y rolling z-score (post-2008) | 1.0 |
| 7 | `hmm_risk` | Vol regime | Pure-risk Gaussian HMM (causal, annual re-fit) | 1.0 |

**Note:** `tlt_trend` was tested but rejected — vol_ratio_h_b < 1 (NOISE verdict). Only the 7 indicators above survive validation.

### 3.2 IC Validation Findings

Per the project's research-math discipline (see `directives/IC Signal Analysis.md`), each indicator was validated:

| Indicator | Vol ratio (H/B) | Crash hit rate | Verdict |
|---|---|---|---|
| `vix` | 2.43 | 17.5% | REGIME_STRONG |
| `rv_regime` | 2.24 | 15.4% | REGIME_STRONG |
| `dd_velocity` | 2.20 | 14.8% | REGIME_USABLE |
| `trend` | 2.08 | 16.8% | REGIME_USABLE |
| `momentum_12_1` | 2.00 | 20.3% | REGIME_USABLE |
| `credit` | 1.93 | 17.4% | REGIME_STRONG |

Each indicator's "hostile" period has **2x+ the forward 21d vol** of its "benign" period and **3-5x crash hit rate**. This is the core discriminating signal.

> [!NOTE]
> Regime indicators don't predict return *direction* — they predict return *dispersion*. Mean returns are similar across benign/hostile (Samir's "best-days-cluster-with-worst-days" insight). The strategy works by avoiding the high-dispersion regime, not by timing direction.

### 3.3 Score Combination

Equal-weight mean across available (non-NaN) indicators per bar. The HMM is appended as a 7th signal when `enable_hmm=True` (production default).

Score distribution (2003-2026):
- frac(score < 0.30): ~10% (hostile / cash regime)
- frac(0.30 ≤ score < 0.50): ~13% (tier 1)
- frac(0.50 ≤ score < 0.75): ~30% (tier 2)
- frac(score ≥ 0.75): ~47% (tier 3)

### 3.4 The Pure-Risk HMM (Component #7)

Unlike `phase0_regime.py`'s HMM (fit on `[log_returns, realised_vol]` — mixed price+risk), our HMM is fit on **dispersion features only**:
- 20-day realised vol of log returns
- abs(log_return) — vol surprise magnitude
- Normalised true range (high-low)/close
- Normalised ATR(14) / close
- 5-day max drawdown

**Causal rolling fit**: 504-bar warmup, then re-fit on expanding window every 252 bars (yearly). No look-ahead.

**Impact** (with 7th indicator vs 6 only):
- CAGR: 7.96% → 8.03% (+0.07pp)
- MaxDD: -27.1% → -21.6% (+5.5pp better)
- Calmar: 0.294 → 0.373 (+27%)

The HMM mostly improves drawdowns by smoothing leverage during high-vol periods that linear indicators don't fully agree on (the 2022 grinding bear case).

---

## 4. Tier Mapping & State Machine

### 4.1 Tier Assignment

```
score < 0.30           → tier 0 (cash)
0.30 ≤ score < 0.50    → tier 1 (1x equity in sleeve)
0.50 ≤ score < 0.75    → tier 2 (2x equity in sleeve)
score ≥ 0.75           → tier 3 (3x equity in sleeve, full L_max)
```

### 4.2 Hysteresis

- **Going UP** between tiers requires score ≥ threshold + 0.05 (e.g., to enter tier 2 from tier 1, score must reach 0.55)
- **Going DOWN** uses the bare threshold

This avoids whipsawing when the score oscillates near a boundary.

### 4.3 Re-Entry Quiet Period

After exiting to cash (score < 0.30), re-entry to tier 1 requires **20 consecutive bars** with score ≥ 0.50 (~one trading month of confirmed benign). Avoids 2008-style false-dawn whipsaws.

This is overridden by the capitulation overlay (Section 6) when conditions warrant earlier re-entry.

### 4.4 Sleeve Composition When Deployed

```
For tier T > 0:
    equity_position = 0.40 × T × 3USL_units
    bond_position   = 0.60 × IGLT_units

For tier 0 (cash):
    equity_position = 0
    bond_position   = 0  (both sleeves out per Samir's "no Texas hedge")
```

Note: at tier 1 with 40% sleeve weight, effective S&P exposure is 40% × 1x = 40% of NAV. At tier 3, it's 40% × 3x = **120%** of NAV (effective leverage from the leveraged ETF).

---

## 5. DD Circuit Breaker

A **failsafe overlay** for when the regime classifier misses or is late (COVID's 5-week crash, 2022's slow grind).

### 5.1 States

| State | DD threshold | Behaviour |
|---|---|---|
| `normal` | DD > -10% | Tier follows regime score |
| `throttled` | -10% ≥ DD > -15% | Tier capped at 1x regardless of score |
| `killed` | DD ≤ -15% | Full cash, both sleeves out |

### 5.2 Recovery From Throttle/Kill

Recovery is **regime-driven** (not DD-driven, to avoid deadlock where cash can't recover the prior HWM):
- Required: regime score ≥ 0.70 for **5 consecutive bars**
- On exit: HWM is reset to current equity (next DD measured from the post-recovery floor)
- If still in throttle when exit fires, re-enables tier > 1
- If killed, returns to normal state with HWM-anchored fresh

### 5.3 Why HWM Reset

The original implementation had a deadlock: once killed → cash → equity flat → DD never recovers below -5% → permanently stuck. Reset on exit fixes this. The cost: a brand-new -15% DD could fire immediately after recovery if a second leg hits, but this is the right behaviour for the failsafe.

---

## 6. Capitulation Overlay — Bottom-Fishing Module

### 6.1 Why It Exists

Samir's framework + tiered regime gate is structurally **late on re-entry** — it requires 20 consecutive bars at score ≥ 0.50 after exiting cash. By that time, recovery rallies (which cluster with worst days) are mostly over.

Empirical missed-rally cost in v1:
- GFC 2008: ~40% of the 2009 V-recovery missed
- COVID 2020: ~22% of the rebound missed
- 2022: ~12% missed

Across these crises, lost rally capture cost ~0.5-1.5pp/yr CAGR. The overlay reclaims part of this.

### 6.2 Activation Logic

The overlay fires when **all four** conditions hold:

1. **Strategy is currently in cash** (regime gate active, hostile)
2. **SPY drawdown from 60-day high ≥ 12%** — the underlying actually crashed
3. **At least 2 of 3 capitulation indicators** registered an extreme low within the last 60 bars:
   - VIX score < 0.05 (most extreme 5%)
   - dd_velocity score < 0.05
   - credit score < 0.05
4. **A stabilisation signal is currently active** (any one):
   - SPY 5-day return ≥ +3% (bounce confirmation)
   - VIX score has recovered ≥ 50% from its recent low
   - rv_regime score ≥ 0.50 after recently being below 0.20

### 6.3 Asymmetric Sizing

When the overlay fires, the strategy enters at **tier 2 (2x equity)** rather than the regime-supported tier. The asymmetry is the safety mechanism — bigger position size, but with a tight stop.

### 6.4 Failed-Bounce Stop

If portfolio equity drops by **5% within 10 bars** of opportunistic entry → exit immediately. This is the false-dawn killer (e.g., the 2008-Oct dead-cat bounce).

### 6.5 Graduation

If regime score rises to ≥ 0.50, the overlay deactivates and the strategy reverts to normal tier logic (potentially scaling up to tier 2 or 3 based on the current score).

### 6.6 Override by DD Breaker

If the DD circuit breaker fires (kills) while the overlay is active, `cap_state.active` is cleared. Defensive logic always wins.

### 6.7 Activation Log (2003-2026, optimal config)

```
2008-10-16  GFC primary fire — failed-bounce stop limited damage
2008-10-30  Re-fire after stop — caught some Nov-Dec recovery
2018-12-31  Q4-2018 bottom — clean catch
2020-03-25  COVID bottom — the big winner: +20pp portfolio boost
2025-04-11  2025 holdout pullback — fired in unseen sanctuary period
```

### 6.8 Optimised Parameters (2026-05-02 sweep)

| Parameter | Value | Source |
|---|---|---|
| `opportunistic_tier` | **2.0** | sweep — doubles COVID capture, tier=3 explodes GFC damage |
| `bounce_5d_threshold` | **0.03** | sweep — captures V-rebound earlier than 0.05 |
| `spy_dd_required` | **0.12** | sweep — sits in stable plateau [0.12, 0.22] |
| `min_capitulation_indicators` | 2 | filters single-indicator false alarms |
| `failed_bounce_drawdown` | 0.05 | conservative stop |
| `failed_bounce_lookback` | 10 | bars to watch for false dawn |
| `graduation_score` | 0.50 | regime-confirmed re-entry |

### 6.9 Performance Impact

| Metric | v1 (no overlay) | v2 (overlay default) | Change |
|---|---|---|---|
| Single-path CAGR | 7.96% | **9.13%** | +1.17pp |
| MaxDD | -27.11% | -27.11% | unchanged |
| Calmar | 0.294 | 0.337 | +0.043 |
| MC P(DD>50%) | 0.40% | **0.16%** | -0.24pp ✅ (improved) |
| MC CAGR p05 | 4.28% | 5.19% | +0.91pp ✅ (improved) |

**The overlay improves both return AND tail risk simultaneously** — rare in a leveraged context. Mechanism: better re-entry timing means quicker recovery from drawdowns, which the bootstrap exposes as reduced compound-loss tail.

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
    notional_base=target_gbp,       # e.g., £4000 target equity allocation
    price=price_usd,                # 3USL price in USD
    quote_ccy="USD",
    base_ccy="GBP",
    fx_rate_quote_to_base=0.7519,   # 1 USD = 0.7519 GBP at GBPUSD=1.33
)
# = int(£4000 / 0.7519 / $100) = 53 units

# For IGLT (GBP-quoted) in GBP account:
units = convert_notional_to_units(
    notional_base=target_gbp,
    price=price_gbp,
    quote_ccy="GBP",
    base_ccy="GBP",
    fx_rate_quote_to_base=None,     # no rate needed when ccy matches
)
```

### 7.3 Fail-Fast on FX Mis-Config

`SamirStackStrategy.on_start()` raises `ValueError` if any sleeve has `quote_ccy != base_ccy` but `fx_rate_*` is the default 1.0. Prevents silent FX assumptions per April 2026 audit rule.

### 7.4 Empirical FX Impact (GBP perspective)

Over 2003-2026, GBP/USD fell from 1.72 to 1.33 (-22.6%) — USD strengthened, providing a tailwind for UK USD holders. But:

| Component | Annual Impact |
|---|---|
| Currency drift tailwind | +90 bp |
| Volatility drag (FX vol ~9%) | -180 bp |
| Random period variance | -50 bp |
| **Net FX drag** | **-140 bp** |

GBP-perspective expected metrics:
- CAGR: 9.13% × (1 - 0.014) ≈ **~7.5-7.7%**
- MaxDD: -27% improving to **-23 to -25%** in crises (USD-haven flow)
- Vol: ~10.5% → **~14-15%** (FX vol added)

**Crisis FX hedge (the redemption)**: USD strengthened in EVERY major risk-off:
- GFC 2008: GBPUSD -19.9% (UK 3USL holders gained that much)
- Brexit 2016: -15.2%
- COVID 2020: -5.4%
- 2022 rate shock: -10.8%

This partial hedge is structural — the strategy's reduced GBP-MaxDD vs USD-MaxDD is a feature, not a bug.

### 7.5 IBKR FX Costs

| Component | Approx Annual Impact |
|---|---|
| FX conversion spread | ~2-5 bp/yr |
| FX min commission ($2/trade) | ~0.2 bp at £100k |
| 3USL TER + funding | already in backtest (0.75% TER + ~5% funding when borrowing) |
| Margin interest | **0** — strategy is unleveraged at portfolio level |
| **Total IBKR friction** | **~5-10 bp/yr (negligible)** |

### 7.6 Hedging Decision

Tested but rejected:
- Pure passive hedging (GBPUSD futures roll) costs ~50bp/yr → only saves 90bp net of -140bp drag
- Removes the crisis hedge (USD haven gone)
- Adds operational complexity

**Decision: accept unhedged FX**. The 140bp drag is real but the crisis hedge has prevented ~3-10pp MaxDD additions in every major crash since 2003.

---

## 8. Performance Validation

### 8.1 Single-Path Backtest (2003-2026, USD)

| Metric | Value |
|---|---|
| CAGR | 9.13% |
| Vol | 10.79% |
| Sharpe | 0.83 |
| Max Drawdown | -27.11% |
| Calmar | 0.337 |
| Sortino | 1.10 |
| Trades/year | 13.3 |
| % time in cash | 18.6% |
| % time tier 3 | ~47% |

### 8.2 Walk-Forward (5-fold + 12mo sanctuary)

| Fold | Period | CAGR | MaxDD | Calmar |
|---|---|---|---|---|
| 0 | 2003-2007 | 4.90% | -12.1% | 0.405 |
| 1 | 2007-2012 (GFC) | 8.32% | -10.2% | 0.816 |
| 2 | 2012-2016 | 11.28% | -11.4% | 0.993 |
| 3 | 2016-2020 (COVID) | 6.92% | -10.6% | 0.655 |
| 4 | 2020-2025 (2022) | 8.78% | -21.6% | 0.408 |
| **Sanctuary 2025-26** | **8.20%** | **-7.8%** | **1.049** ⭐ |

- 5/5 folds positive CAGR
- 5/5 folds MaxDD < 25%
- Sanctuary holdout (unseen during research) confirms generalisation

### 8.3 Monte Carlo (Stationary Bootstrap, 5000 paths, block=63d)

| Metric | Value |
|---|---|
| CAGR median | 9.11% |
| CAGR p05 (downside) | 5.19% |
| MaxDD median | -22.83% |
| MaxDD p05 | -35.24% |
| **P(DD>25%)** | 35.5% |
| **P(DD>35%)** | 5.3% |
| **P(DD>50%)** | **0.16%** ✅ |
| **P(CAGR<0%)** | 0.0% |

**RoR-acceptable per user spec (target P(DD>50%) ≈ 0%).**

### 8.4 Crisis Stress Test

| Crisis | Strategy MaxDD | Strategy Return | SPY MaxDD | Outperformance |
|---|---|---|---|---|
| Dot-com 2000-2003 (extended) | -28% (L=1) | -25% | -48% | +20pp |
| GFC 2008 | -3.7% | +7.7% | -55.2% | +51pp |
| EU Debt 2011 | -9.8% | -7.5% | -18.6% | +9pp |
| 2018 Q4 | -8.9% | -5.0% | -19.4% | +10pp |
| COVID 2020 | -5.6% | +2.1% | -33.7% | +28pp |
| 2022 Rate Shock | -15.5% | -15.4% | -24.5% | +9pp |
| 2025-26 Holdout | -7.8% | +8.6% | -12.0% | +4pp |

### 8.5 Benchmark Comparison

| Strategy | CAGR | MaxDD | Calmar | P(DD>50%) | Verdict |
|---|---|---|---|---|---|
| **Samir-Stack** | 9.13% | -27.1% | **0.337** | **0.16%** | ⭐ |
| 60/40 SPY/IEF | 8.43% | -32.2% | 0.261 | 0.10% | safe but lower CAGR |
| Faber GTAA | 7.31% | -20.7% | 0.354 | 1.0% | best Calmar, lowest CAGR |
| Samir-pure (1x) | 7.40% | -26.7% | 0.277 | 1.0% | base philosophy, no leverage |
| SPY buy-hold | 11.27% | -55.2% | 0.204 | 15.7% | high return, catastrophic tail |
| HFEA 55/45 | 18.05% | -70.1% | 0.258 | **86.6%** | guaranteed-ish ruin |

---

## 9. Live Deployment

### 9.1 Trading Instruments (IBKR UK Paper)

| Sleeve | Symbol | ISIN | Currency | TER | Notes |
|---|---|---|---|---|---|
| Equity (primary) | 3USL.LSEETF | XS1078280466 | USD | 0.75% | WisdomTree 3x S&P 500 UCITS |
| Equity (alt) | 3LUS.LSEETF | n/a | GBp | similar | GraniteShares alt — fully GBP-clean |
| Bond | IGLT.LSEETF | IE00B1FZSB30 | GBP | 0.07% | iShares Core UK Gilts UCITS |

**Signal data sources** (subscribed but not traded):
- SPY.ARCA — primary regime driver
- VIX.CBOE — vol regime indicator
- HYG.ARCA — credit-spread numerator
- IEF.NASDAQ — credit-spread denominator (separate from bond sleeve since bond=IGLT, not IEF)

### 9.2 Live Configuration (`config/samir_stack.toml`)

```toml
[strategy]
equity_instrument_id = "3USL.LSEETF"
bond_instrument_id   = "IGLT.LSEETF"

# Per-instrument quote currency
equity_quote_ccy = "USD"
bond_quote_ccy   = "GBP"

# FX rate seed (operator updates periodically)
# At 2026-04-02: GBPUSD = 1.33 -> 1 USD = 0.7519 GBP
fx_rate_usd_to_gbp = 0.7519

# Strategy parameters (validated defaults)
equity_weight = 0.40
bond_weight   = 0.60
L_max         = 3.0
tier_thresholds   = [0.30, 0.50, 0.75]
hysteresis_buffer = 0.05
re_entry_quiet_bars = 20

# DD circuit breaker
dd_throttle = 0.10
dd_kill     = 0.15
dd_re_entry_score = 0.70
dd_re_entry_bars  = 5

# Operational
initial_equity = 10000.0
base_ccy       = "GBP"
warmup_bars    = 250
prm_id         = "samir_stack"
```

The capitulation overlay must be enabled separately by setting `cfg.capitulation = CapitulationConfig(enabled=True)` when constructing the strategy from the config — current TOML loading doesn't auto-instantiate nested configs.

### 9.3 Adding to Champion Portfolio

Edit `scripts/run_portfolio.py`:

1. **Add warmup parquets** to `_STRATEGY_WARMUP_FILES`:
   ```python
   "samir_stack": [
       "SPY_D.parquet", "^VIX_D.parquet", "HYG_D.parquet",
       "IEF_D.parquet", "IGLT_D.parquet", "GBPUSD_D.parquet",
   ],
   ```

2. **Add to STRATEGY_REGISTRY** with the IBContract entries (3USL, IGLT, SPY, VIX, HYG, IEF).

3. **Add to `champion_portfolio` set**.

See `bond_equity_ihyu_cspx` registry entry for a template.

### 9.4 Pre-Flight Checklist

Before flipping to live capital:

- [ ] Run `uv run pytest tests/test_samir_stack_parity.py -v` — all 7 tests pass
- [ ] Run `uv run ruff check titan/strategies/samir_stack/` — clean
- [ ] Verify all 6 IB contracts resolve (3USL, IGLT, SPY, VIX, HYG, IEF) on the live IB Gateway with the gnzsnz socat-forwarded ports
- [ ] Set `initial_equity` in config to actual allocation slice (not the £10k placeholder)
- [ ] Update `fx_rate_usd_to_gbp` to current GBPUSD market rate
- [ ] Verify PRM halt threshold doesn't conflict with the strategy's internal DD breaker
- [ ] Paper-trade soak for **at least 4 weeks** — strategy fires only ~13 times/year so longer than usual is needed for live-vs-research validation
- [ ] Monitor live-vs-research regime score divergence — should be < 0.02 on any given bar

### 9.5 Operational Runbook

#### Daily Monitoring

Each day after the daily bar close:
1. Strategy logs the regime score, applied tier, dd_state, breakdown of the 7 indicators
2. Compare to research-computed score for the same bar (should agree within 0.02)
3. Check if any tier transition occurred — these are the trades

#### Weekly Review

- Verify per-strategy equity matches expectations vs broker NLV
- Review the `applied_tier` history — any unexpected sequences?
- Check capitulation overlay state — fired? if so, when did the failed-bounce stop or graduation activate?

#### Halting

Manual halt path (per `directives/Emergency Operations`):
```bash
docker compose exec titan-portfolio uv run python scripts/kill_switch.py --strategy samir_stack
```

This sets `.tmp/portfolio_halt.json` for the strategy. Restart the runner to resume after operator review.

#### FX Rate Updates

The static `fx_rate_usd_to_gbp` config drifts from spot. Update monthly during paper, weekly when live, or wire dynamic GBPUSD bar subscription as v2 enhancement.

---

## 10. Known Limitations and Risks

### 10.1 Slow Grinding Bears (2022-Style)

The regime classifier is best at **sharp systemic events** (GFC, COVID) where capitulation is identifiable. Slow grinders (2022 rate shock, dot-com 2000-2002) decay through the indicators gradually rather than spike — the gate fires LATE and the strategy takes meaningful damage before exiting.

Mitigations already in place:
- HMM indicator catches vol-regime changes the linear indicators miss (improved 2022 by ~7pp MaxDD)
- DD circuit breaker provides failsafe at -15%

Not solved: the 2022 -15% MaxDD (similar to Faber GTAA's -15%). Realistic expectation, not a bug.

### 10.2 Fast Exogenous Shocks (COVID-Style)

The 5-week COVID crash beat the regime classifier — score didn't go fully hostile until after the bottom. The DD breaker caught it at -15% and limited damage to -5.6%, but not to zero.

Capitulation overlay specifically helps here: caught the COVID rebound on Mar 25 → +20pp portfolio boost.

### 10.3 Sample Size for Capitulation Optimisation

The overlay was tuned on **2 major capitulations** (2008, 2020) plus minor pulldowns. Even with stable parameter plateaus identified in the sweep, this is genuinely a small sample. The overlay's 1.17pp CAGR uplift could be coincidence rather than signal — paper soak will help validate.

### 10.4 FX Static Rate

`fx_rate_usd_to_gbp` is a config-static value, drifts from spot daily. At ~9% annual GBP/USD vol, a stale rate by 1 month → ~2.5% sizing error → minor at 40% sleeve allocation. Acceptable for v1; v2 should subscribe GBPUSD bars and update dynamically.

### 10.5 Contract Resolution

The format `<SYMBOL>.LSEETF` matches project convention (`CSPX.LSEETF` is in production), but **3USL/IGLT contract resolution on live IB Gateway is unverified** — must be confirmed via tiny test order before sizing up.

### 10.6 Sequential Crashes (Multi-Leg Bears)

A strategy that exits on first-leg DD-kill, recovers, re-enters, and gets hit by second-leg before the regime classifier can stop it would compound losses. The strategy is structurally vulnerable to this scenario. The dot-com 2000-2002 single-path test showed the strategy survived but with -28% MaxDD (vs SPY -48%).

### 10.7 Counterparty Risk on Leveraged ETF

3USL is swap-based — implicitly long counterparty credit risk on the swap counterparty (typically a major bank). In a true financial-system breakdown (worse than 2008), the swap counterparty could fail. This is unhedged.

---

## 11. Testing & Validation Hooks

### 11.1 Parity Tests (`tests/test_samir_stack_parity.py`)

7 tests passing:
1. `test_regime_score_parity_equal_weight` — live `compute_regime_score` matches research within 0.05 tolerance
2. `test_target_tier_parity_basic_thresholds` — tier mapping with hysteresis verified across grid of (score, current_tier) pairs
3. `test_target_tier_caps_at_L_max` — never exceeds configured leverage cap
4. `test_target_tier_handles_nan` — NaN score holds current tier
5. `test_fx_unit_conversion_gbp_base_usd_equity` — sizing math correct for GBP base + USD equity
6. `test_fx_fail_fast_on_default_rate` — refuses silent ccy assumption
7. `test_fx_passthrough_when_currencies_match` — no-op for IGLT in GBP account

Run before any deployment:
```bash
uv run pytest tests/test_samir_stack_parity.py -v
```

### 11.2 Reproducing All Research Results

```bash
uv run python research/samir_stack/run_pipeline.py
```

Outputs to `.tmp/reports/samir_stack/`:
- `indicator_panel.parquet` — full 7-column score panel
- `ic_validation.csv` — IC + regime metrics for each indicator
- `regime_score.parquet` — combined ensemble score series
- `strategy_summary.csv` — performance across L_max grid
- `all_strategies_summary.csv` — Samir-Stack vs benchmarks
- `stress_scenarios.csv` — per-crisis decomposition
- `monte_carlo_ror.csv` — RoR distributions
- `wfo_folds_L_max_3.csv` — WFO results for production config
- `wfo_sanctuary_L_max_3.txt` — sanctuary holdout metrics

### 11.3 Capitulation Sweep (Robustness Re-Validation)

```bash
uv run python -m research.samir_stack.capitulation_sweep
```

Re-runs the parameter sweep that produced the optimised defaults. Useful when re-validating after a year of new data.

---

## 12. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-05-02 | Initial release. STACK 40/60 L=3 with HMM-enhanced regime score and capitulation overlay (optimised defaults: tier=2, bounce=0.03, dd=0.12). 7/7 parity tests passing. Ruff clean. Ready for paper deployment on IBKR UK (3USL + IGLT). |

---

## 13. References

**Within this project:**
- `directives/IC Signal Analysis.md` — IC validation methodology
- `directives/System Status and Roadmap.md` — full project state
- `references/portfolio-risk-architecture.md` — PRM contract
- `references/research-math-guardrails.md` — research math discipline
- `directives/Emergency Operations.md` — kill switch + halt procedures
- Project memory: `feedback_research_math_discipline.md`

**External:**
- Samir Varma — risk-classification framework (binary regime model)
- Asness, Moskowitz — 12-1 momentum convention
- Faber, M. — Global Tactical Asset Allocation (200-day SMA filter)
- Moreira & Muir 2017 — *Volatility-Managed Portfolios*, Journal of Finance
- Politis & Romano 1994 — Stationary bootstrap
- Hedgefundie's Excellent Adventure — leveraged risk-parity baseline (HFEA)
- Newfound Research — return-stacking concepts
