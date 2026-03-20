# Backtesting & Validation Pipeline

**Version:** 1.0 | **Last Updated:** 2026-03-20

---

## Overview

This directive specifies Phases 3–6 of the quantitative research pipeline. It is the natural continuation of `IC Signal Analysis.md` (Phases 0–2). Phase 2 delivers a validated composite signal with known IC, ICIR, and natural horizon. This document describes how to convert that signal into a tested, stress-validated, deployable strategy.

**This document is asset-class and strategy agnostic.** The same process applies whether the input is a momentum composite on EUR/USD H1, a mean-reversion RSI on daily equities, or an ensemble of ML features on futures. Only the cost model parameters change.

> [!IMPORTANT]
> For specific strategy results, validated symbols, and per-instrument findings,
> see `research/ic_analysis/FINDINGS.md`. This document describes the **process**,
> not the outcomes.

---

## Pipeline Position

```
Phase 0  — Regime Identification     ← IC Signal Analysis.md
Phase 1  — Signal Discovery          ← IC Signal Analysis.md
Phase 2  — Signal Combination        ← IC Signal Analysis.md (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3  — IS/OOS Backtest           ← YOU ARE HERE
Phase 4  — Walk-Forward Optimisation ← This document
Phase 5  — Robustness Validation     ← This document (6 gates)
Phase 6  — Live Deployment           ← This document
```

### What Phase 2 Delivers to Phase 3

- **Composite signal:** a single time series (or set of sub-signals) to trade
- **Sign orientation:** each signal oriented so positive = bullish
- **IS calibration stats:** mean/std for z-score normalisation (frozen on IS data)
- **Natural horizon:** the holding period implied by Phase 1 IC peak

---

## Directional Constraints

> [!IMPORTANT]
> **Equities / ETFs:** Long-only. Short side excluded from backtest statistics.
> Negative-IC signals are sign-flipped to produce buy-the-dip entries.
>
> **FX / Futures:** Both directions. Positive composite = long, negative = short.

This constraint is enforced in Phase 3 entry logic and must be respected in Phases 4–6.

---

## Sharpe Ratio Definitions

> [!WARNING]
> **Two Sharpe metrics exist in this pipeline. They are NOT comparable.**

### Trade Sharpe (Phases 3–5 Gate Metric)

```
Trade_Sharpe = mean(per_trade_returns) / std(per_trade_returns) × sqrt(trades_per_year)
```

Rewards high win rates and consistent per-trade edge. Used for ranking signals and passing quality gates. **Inflates when trade frequency is low** — a strategy with 5 trades/year can show Trade Sharpe > 3 from win rate alone.

### Daily Sharpe (Comparable to Buy-and-Hold)

```
Daily_Sharpe = mean(daily_equity_returns) / std(daily_equity_returns) × sqrt(252)
```

Includes flat days (days with no position). This is the only Sharpe directly comparable to a benchmark. **Phase 3 must report both.**

### Per-Trade % Returns (Critical)

> [!CAUTION]
> **Never use absolute trade P&L when the portfolio uses `size_type="percent"` (compounding).**
> A $10K win at equity=$50K is a 20% return; the same $10K at equity=$500K is 2%.
> Always use `pf.trades.records_readable["Return"]` — VBT's normalised per-trade return.
> This makes Monte Carlo, top-N removal, and all Phase 5 gates valid.

---

## Phase 3 — IS/OOS Backtest

### IS/OOS Split

```
IS  = bars[0 : int(0.70 × n)]    ← threshold selection + calibration ONLY
OOS = bars[int(0.70 × n) : n]    ← ALL reported performance metrics
```

**Rules:**
- No shuffling. OOS is always the chronologically most recent 30%.
- IS mean/std are frozen and applied to OOS as-is. OOS z-score may drift — this is expected.
- Only IS Sharpe is used for threshold selection. OOS is strictly blind.

### Composite Signal Construction

```
1. Compute raw signals on native bar resolution
2. For multi-timeframe: forward-fill higher TFs to base timeline (causal only)
   CRITICAL: shift higher-TF signals by 1 bar BEFORE reindex to prevent current-bar leak
3. Sign-normalise each signal using IC sign from IS window
4. Average → raw composite
5. Z-score using IS mean and IS std only
6. Apply z-score to both IS and OOS unchanged
```

> [!WARNING]
> **Look-Ahead Bias in MTF Alignment:** When aligning higher-TF signals to a base TF
> (e.g., Daily → H1), the current higher-TF bar's close is not yet known. Apply
> `native_signals.shift(1)` before `reindex(base_index, method="ffill")`. Without this
> shift, OOS Sharpe inflates dramatically (confirmed: biased +7.7 vs debiased −2.5 on EUR/USD).

### Entry / Exit Logic

```
Long  entry : composite_z crosses above  +threshold  (next-bar open execution)
Long  exit  : composite_z drops below    0
Short entry : composite_z crosses below  -threshold  (FX/futures only)
Short exit  : composite_z rises above    0
```

The `signal.shift(1)` in VectorBT ensures fills on the bar *after* the signal fires.

### Full Cost Model

All three cost components are applied simultaneously. **There is no gross backtest** — every Sharpe ratio includes all friction.

#### 1. Spread (Per-Fill)

The bid/ask half-spread paid at each fill, normalised by median close:

```python
vbt_fees = spread_price_units / median_close
```

| Asset Class | Typical Spread | Notes |
|---|---|---|
| FX Majors | 0.5–1.0 pip | EUR/USD ≈ 0.5 pip, crosses ≈ 1.0 pip |
| US Large-Cap Equity | 1–3 cents | $0.01–0.03 per share |
| ETFs (SPY, QQQ) | 1 cent | Extremely liquid |
| Small-Cap Equity | 5–20 cents | Wider spreads, higher impact |

#### 2. Slippage (Per-Fill, Market Impact)

Execution slippage at each fill. Represents the price movement between signal and fill:

| Asset Class | Default | Stress (3×) |
|---|---|---|
| FX | 0.5 pip | 1.5 pip |
| Large-Cap Equity | 2 bps | 6 bps |
| ETF | 1–2 bps | 3–6 bps |

Total friction per round trip = (spread + slippage) × 2 fills.

#### 3. Carry / Swap Cost (Overnight Positions)

**FX:** Swap is the interest-rate differential paid/received for holding overnight at 21:00 UTC. Pair-specific and direction-specific. Computed post-hoc and subtracted from gross P&L.

**Equities:** No swap; instead commission per share (if applicable) and SEC/TAF regulatory fees.

#### Cost Sensitivity Sweep (New)

Phase 3 must include a **cost sensitivity analysis**:

```
Run backtest at: 0.5×, 1.0×, 1.5×, 2.0×, 3.0× base friction
Report: Sharpe at each multiplier
Identify: break-even friction (max cost where Sharpe > 0)
```

This reveals how fragile the edge is to execution quality. A strategy whose Sharpe drops below 0 at 1.5× friction is not deployable.

### Position Sizing (ATR-Based, Currency-Agnostic)

```python
atr       = ATR(14)
stop_dist = stop_atr_mult × atr           # price distance to stop
stop_pct  = stop_dist / close             # stop as fraction of price
size_pct  = risk_pct / stop_pct           # portfolio fraction
size_pct  = min(size_pct, max_leverage)   # cap leverage
```

| Parameter | FX Default | Equity Default |
|---|---|---|
| `risk_pct` | 1.0% (0.01) | 0.5–1.0% |
| `stop_atr_mult` | 1.5 | 1.5 |
| `max_leverage` | 20–30× | 2–5× |

By expressing stop as a fraction of price, the formula works identically for EUR/USD (~1.08) and USD/JPY (~145) without pip-value conversion.

### Threshold Search (IS Only)

Default grid: `[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]`

Best threshold = `argmax(IS_Sharpe)`. Applied unchanged to OOS.

### Risk of Ruin Gate (Balsara)

Computed from OOS trade results:

```
edge      = (win_rate × avg_win) − (loss_rate × avg_loss)
cap_units = 1 / risk_per_trade
P(ruin)   = ((1 − edge) / (1 + edge)) ^ cap_units
```

| Level | Target |
|---|---|
| Retail | P(ruin at 25% DD) < 5% |
| Professional | P(ruin at 25% DD) < 1% |
| Institutional | P(ruin at 50% DD) < 0.1% |

### Holding Period Distribution (New)

Phase 3 must output a **histogram of actual trade durations** (in bars). Compare the median holding period to the signal's natural horizon from Phase 1:

- If Phase 1 IC peaks at `h=20` but median hold = 3 bars → exit logic is too aggressive
- If Phase 1 IC peaks at `h=5` but median hold = 40 bars → exit logic is too loose
- Ideal: median hold ≈ 0.5–1.5× natural horizon

### Phase 3 Quality Gates

| Gate | Threshold | Notes |
|---|---|---|
| OOS Trade Sharpe | > 1.0 | Minimum viable edge after friction |
| OOS Daily Sharpe | > 0.5 | Conservative, comparable to B&H |
| IS/OOS Parity | > 0.5 | OOS must be at least half as good as IS |
| OOS Trades | ≥ 30 | Minimum for statistical significance |
| Win Rate | ≥ 40% | Below this, payoff ratio must be exceptional |
| Max Drawdown | ≤ 25% | Hard ceiling; strategy-specific |
| Risk of Ruin | P(ruin 25% DD) < 5% | Survival before performance |
| Break-Even Friction | ≥ 2.0× base | Edge must survive doubled costs |

---

## Phase 4 — Walk-Forward Optimisation (WFO)

### Purpose

Validate temporal stability by repeatedly re-fitting the strategy on a rolling IS window and evaluating on the subsequent OOS window. This replicates live deployment — the threshold is re-selected as the IS window advances.

### WFO Structure

```
    ┌──── IS Window ────┐ ┌── OOS Window ──┐
    |                   | |                 |
    t0                  t1                  t2

              ┌──── IS Window ────┐ ┌── OOS Window ──┐
              t1                  t2                  t3
    ...
```

### Configuration

| Parameter | FX Default | Equity Default | Notes |
|---|---|---|---|
| IS Window | 2 years | 504 bars (~2yr) | Must contain ≥ 2 regime cycles |
| OOS Window | 6 months | 126 bars (~6mo) | Non-overlapping |
| Step | = OOS Window | = OOS Window | Non-overlapping OOS periods |
| Min Folds | 5 | 5 | Fewer is statistically unreliable |

### Anchored vs Rolling WFO

| Type | IS Window | When to Use |
|---|---|---|
| **Rolling** | Fixed length, slides forward | Default. Tests regime adaptability. |
| **Anchored** | Starts at bar 0, grows forward | When you have short history (<5yr). Maximises IS data. |

### Per-Fold Procedure

1. Select IS data: `bars[i : i + IS_bars]`
2. Re-calibrate composite z-score (IS mean/std)
3. Search threshold on IS Sharpe (same grid as Phase 3)
4. Apply best IS threshold to OOS: `bars[i + IS_bars : i + IS_bars + OOS_bars]`
5. Record: IS Sharpe, OOS Sharpe, threshold used, win rate, trade count

### Stitched OOS Equity Curve

All OOS periods are concatenated into one continuous equity curve. This is the most honest performance representation — it shows how the strategy would have performed with periodic re-calibration, as it would in live deployment.

### Phase 4 Quality Gates

| Gate | Threshold | Rationale |
|---|---|---|
| % folds with OOS Sharpe > 0 | ≥ 70% | Profitable in most periods |
| % folds with OOS Sharpe > 1 | ≥ 50% | Meaningful edge in at least half |
| Worst single fold OOS Sharpe | ≥ −2.0 | No catastrophic regime breaks |
| Stitched OOS Sharpe | ≥ 1.0 | Full OOS period must be investable |
| OOS/IS Parity (aggregate) | ≥ 0.5 | OOS at least half as good as IS |

---

## Phase 5 — Robustness Validation

Six adversarial stress tests. **All six gates must pass** before Phase 6.

### Gate 1: Monte Carlo Trade Shuffle

**Tests:** Whether the observed Sharpe reflects genuine edge or lucky sequencing.

```
1. Extract OOS per-trade % returns (NOT absolute P&L)
2. Shuffle trade order N=1,000 times
3. For each shuffle, annualise Sharpe:
     trades_per_year = n_trades / (oos_bars / bars_per_year)
     sharpe = (mean / std) × sqrt(trades_per_year)
4. Report 5th-percentile Sharpe and % profitable simulations
```

| Gate | Threshold |
|---|---|
| 5th-percentile Sharpe | > 0.5 |
| % profitable simulations | > 80% |

**Interpretation:** If 5th-pct Sharpe < 0.5, the edge is sequence-dependent — consistent with lucky clustering, not genuine alpha.

### Gate 2: Top-N Trade Removal (Concentration Risk)

**Tests:** Whether returns are concentrated in a small number of outlier trades.

```
1. Combine all per-trade % returns
2. Remove the 10 largest by absolute magnitude
3. Sum remaining returns
4. Gate: remaining sum > 0
```

A strategy where top-10 trades generate >50% of returns is fragile — it relied on rare events.

### Gate 3: Cost Stress Test (3× Slippage)

**Tests:** Sensitivity to execution quality.

```
Re-run full OOS backtest with slippage = 3× default
Gate: OOS Sharpe > 0.5
```

If the strategy breaks when slippage triples, it is not robust enough for live markets where slippage spikes during volatility events.

### Gate 4: WFO Consecutive Negative Folds

**Tests:** Sustained losing streaks in the walk-forward.

```
Scan Phase 4 OOS Sharpe column for longest run of consecutive negative values
Gate: max consecutive negative ≤ 2
```

One bad fold (~6 months) can happen. Two consecutive (~12 months) is borderline. Three+ means the signal has genuinely lost its edge.

### Gate 5: Regime Robustness (New — ADX + HMM)

**Tests:** Whether the edge exists across different market regimes, not just one.

Using Phase 0 regime labels, compute OOS Sharpe during:

| Regime Axis | Buckets |
|---|---|
| **ADX** | Trending (ADX > 25), Ranging (ADX < 20), Neutral (20-25) |
| **HMM** | State 0 (low-vol/mean-reverting), State 1 (high-vol/trending) |
| **Volatility** | rv20 terciles: Low (<30th pct), Mid (30-70th), High (>70th) |

```
Gate: OOS Sharpe > 0 in at least 2/3 ADX regimes
Gate: OOS Sharpe > 0 in at least 1/2 HMM states
```

**Why this matters:** A trend-following strategy that only works when ADX > 25 will suffer catastrophic losses during the next prolonged ranging period. Regime robustness ensures the strategy either (a) works across regimes or (b) has an effective regime gate that prevents losses in adverse regimes.

> [!IMPORTANT]
> A strategy that fails the regime gate but has a validated regime filter (e.g., ADX > 25 gate
> that prevents trading in ranging regimes) can still pass — provided the gate is active and
> the remaining regimes show positive Sharpe. The test is: **does the strategy protect capital
> in every regime it encounters?**

### Gate 6: Alpha vs Beta Decomposition (New)

**Tests:** Whether the strategy generates genuine alpha or is just leveraged beta.

```python
import statsmodels.api as sm

# Regress daily strategy returns on benchmark
# FX: use DXY (US Dollar Index)
# Equities: use SPY
benchmark_returns = benchmark.pct_change()
strategy_returns  = strategy_equity.pct_change()

X = sm.add_constant(benchmark_returns.dropna())
model = sm.OLS(strategy_returns.dropna(), X).fit()

beta  = model.params[1]
alpha = model.params[0] × 252        # annualised
r_sq  = model.rsquared
```

| Metric | Gate | Interpretation |
|---|---|---|
| Annualised Alpha | > 0 | Strategy adds returns beyond benchmark exposure |
| Beta | < 1.0 (equities) | Not just leveraged B&H |
| R² | < 0.5 | Strategy returns are not merely benchmark-driven |

**Why this matters:** A strategy with Sharpe = 1.5 but beta = 1.3 to SPY is just leveraged buy-and-hold. It will generate the same Sharpe as 1.3× SPY with no new information. True alpha requires low beta and positive intercept.

### Phase 5 Summary Table

| Gate | Test | Threshold |
|---|---|---|
| G1 | Monte Carlo 5th-pct Sharpe | > 0.5 |
| G1 | Monte Carlo % profitable | > 80% |
| G2 | Top-10 removal remaining sum | > 0 |
| G3 | 3× slippage OOS Sharpe | > 0.5 |
| G4 | Max consecutive negative WFO folds | ≤ 2 |
| G5 | Regime Sharpe > 0 in ≥ 2/3 ADX + ≥ 1/2 HMM | Required |
| G6 | Annualised alpha > 0, beta < 1.0 | Required |

---

## Phase 6 — Live Deployment

### Strategy Implementation Pattern

The live strategy must faithfully replicate the research composite construction:

1. **Warmup calibration:** Load the last N bars per timeframe from parquet, compute signals, derive IC signs via Spearman correlation, freeze composite mean/std for z-scoring — exactly as the IS period does in research.

2. **On each bar:** Recompute signals for that TF, store latest oriented value.

3. **Higher TF updates:** Signals update only when those bars close. Between closings, last-known value is forward-filled (matching research `method="ffill"`).

4. **Composite z-score:** `mean(oriented_signals)` normalised by calibration stats. Evaluated on every base-TF bar.

5. **Entry/exit:** Threshold crossing for entries, zero crossing for exits.

6. **Sizing:** Identical formula to Phase 3 — ATR-based, capped at leverage limit.

### TOML Configuration

```toml
[instrument]
threshold     = 0.75      # from Phase 3 best IS threshold
risk_pct      = 0.01      # 1% equity risk per trade
stop_atr_mult = 1.5       # ATR multiplier for sizing
leverage_cap  = 20.0      # max leverage (FX: 20-30, equity: 2-5)
warmup_bars   = 1000      # bars for IS calibration
direction     = "both"    # "long_only" for equities, "both" for FX
```

**Do not change thresholds without re-running Phases 3–5.** The threshold was selected on IS data and validated on OOS — changing it retroactively is look-ahead bias.

### Live Monitoring

| Metric | Frequency | Alert Threshold |
|---|---|---|
| Rolling 20-trade Sharpe | Daily | < 0 for 3 consecutive checks |
| Max drawdown from peak | Real-time | > Phase 3 OOS max DD × 1.5 |
| Trade frequency | Weekly | < 50% or > 200% of Phase 3 rate |
| Signal autocorrelation | Monthly | AR1 drift > 0.2 from Phase 1 value |
| RcntIC vs Phase 1 IC | Monthly | RcntIC < 0.5 × Phase 1 IC |

### Kill Switch Triggers (Mandatory)

| Trigger | Action |
|---|---|
| Max DD exceeds 1.5× OOS max DD | Flatten all positions, halt strategy |
| 3 consecutive months negative P&L | Pause; re-run Phase 1 to check IC decay |
| RcntIC falls below 0.5× full-sample IC | Signal edge is dying; halt and investigate |
| Risk of Ruin > 5% (rolling 50-trade window) | Reduce position size to 0.25% until recovery |

---

## Design Principles

### 1. No Look-Ahead Bias

- Forward returns: **only** via `.shift(-h)` in `compute_forward_returns()` — the single permitted negative shift.
- Higher TF reindex: always `method="ffill"` with prior-bar shift.
- IS/OOS split: always time-ordered. Never shuffle.
- WFO: IS calibration applied to next OOS window only.
- Z-score: IS mean/std frozen, applied to OOS as-is.

### 2. Full Friction on Every Run

The cost model (spread + slippage + carry) is never disabled. There is no "gross" backtest. Every Sharpe ratio includes all three friction components.

### 3. OOS Is the Only Number That Matters

IS Sharpe is used internally for threshold selection only. Every external-facing metric is computed on OOS data.

### 4. Per-Trade % Returns for Monte Carlo

Never use absolute trade P&L when the portfolio compounds. Use normalised per-trade returns for all Phase 5 gates.

### 5. ATR-Based Sizing Is Currency-Agnostic

By expressing stop distance as a fraction of price, the sizing formula works identically across all instruments.

### 6. Multiple Sharpe Definitions Are Non-Negotiable

Phase 3 must report **both** Trade Sharpe and Daily Sharpe. Comparing Trade Sharpe to a B&H Daily Sharpe is invalid. All Phase 3–5 gates use Trade Sharpe; all benchmark comparisons use Daily Sharpe.

---

## Scripts Reference

| Script | Phase | Purpose |
|---|---|---|
| `run_ic_backtest.py` | 3 | IS/OOS full-friction backtest (FX MTF) |
| `run_regime_backtest.py` | 3 | ADX-gated backtest per instrument |
| `run_cat_amat_strategy.py` | 3 | Equities regime-gated long-only |
| `run_spy_strategy.py` | 3 | SPY three-signal strategy |
| `run_equity_longonly_pipeline.py` | 3–5 | Cross-asset equity pipeline |
| `run_wfo.py` | 4 | Walk-forward optimisation |
| `run_ic_robustness.py` | 5 | Robustness gates (MC, top-N, 3× slip, WFO) |
| `titan/strategies/ic_mtf/strategy.py` | 6 | NautilusTrader live strategy (FX) |
| `titan/strategies/ic_equity_daily/strategy.py` | 6 | NautilusTrader live strategy (equity) |

---

## When to Re-Run the Pipeline

Re-run Phases 3–5 if:

1. **Model is stale > 6 months** with degrading live performance
2. **New signals added** — composite has changed from Phase 2
3. **New instrument** beyond existing validated set
4. **Cost model updated** — spread/slippage assumptions changed
5. **RcntIC decay detected** — Phase 1 edge may have died

> [!CAUTION]
> Always use **fresh OOS data** (extend the data window forward, maintain 70/30 split).
> Never re-optimise on the period previously used as OOS — that is look-ahead bias.

---

## Version History

| Version | Date | Changes |
|---|---|---|
| **1.0** | 2026-03-20 | Initial unified directive. Consolidated from IC MTF Backtesting Guide, MTF Optimization Protocol, and IC Equity Daily Strategy into a single asset-agnostic process spec. Added 6 improvements: Sharpe definition clarity, cost sensitivity sweep, regime robustness gate (ADX+HMM), alpha/beta decomposition, holding period distribution, per-trade % return warning. Risk of Ruin gate added to Phase 3. Kill switch triggers added to Phase 6. |
