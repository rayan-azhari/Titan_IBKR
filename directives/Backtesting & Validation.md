# Backtesting & Validation Pipeline

**Version:** 1.2 | **Last Updated:** 2026-03-20

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

Includes flat days (days with no position). This is the only Sharpe directly comparable to a benchmark. **Phase 3 reports both side-by-side and gates on both:** OOS Trade Sharpe > 0 AND OOS Daily Sharpe > 0.

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

**Time-varying swap rates (C8):** When historical swap data is available in `data/rates/{instrument}_swap.parquet`, `titan/costs/swap_curve.py` loads per-date swap values and applies them dynamically. Otherwise, static values from `COST_PROFILES` are used as fallback.

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
edge      = (win_rate × avg_win_R) − (loss_rate × avg_loss_R)
cap_units = 0.25 / risk_per_trade    # 25% DD expressed in risk units
P(ruin)   = ((1 − edge) / (1 + edge)) ^ cap_units
```

> [!NOTE]
> **Audit fix C3:** Win/loss values are normalised to R-multiples (`avg_win / risk_pct`) so that
> the edge is in the same units as `cap_units`. Without this, raw percentage returns make
> `edge ≈ 0` and P(ruin) always ≈ 1.0.

| Level | Target |
|---|---|
| Retail | P(ruin at 25% DD) < 5% |
| Professional | P(ruin at 25% DD) < 1% |
| Institutional | P(ruin at 50% DD) < 0.1% |

### Holding Period Distribution (New)

Phase 3 outputs a **histogram of actual trade durations** (in bars). Compare the median holding period to the signal's natural horizon from Phase 1:

- If Phase 1 IC peaks at `h=20` but median hold = 3 bars → exit logic is too aggressive
- If Phase 1 IC peaks at `h=5` but median hold = 40 bars → exit logic is too loose
- Ideal: median hold ≈ 0.5–1.5× natural horizon

> [!NOTE]
> **Audit fix A3:** Holding histogram durations are now adjusted for bar frequency
> (e.g., `hours_per_bar = 24` for daily, `4` for H4, `1` for H1) instead of hardcoding H1.

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
| abs(Beta) | < threshold (asset-class dependent) | Not just leveraged B&H |
| R² | < 0.5 | Strategy returns are not merely benchmark-driven |

**Asset-Class Beta Thresholds (A5):**

| Asset Class | Max abs(Beta) | Rationale |
|---|---|---|
| FX Major/Minor | 0.5 | FX strategies should have minimal equity beta |
| Equity | 1.0 | Long-only equity beta up to 1.0 is acceptable |
| Crypto | 1.5 | Higher beta tolerance for crypto |
| Commodity | 0.8 | Moderate independence expected |

**Why this matters:** A strategy with Sharpe = 1.5 but beta = 1.3 to SPY is just leveraged buy-and-hold. It will generate the same Sharpe as 1.3× SPY with no new information. True alpha requires low beta and positive intercept.

### Phase 5 Summary Table

| Gate | Test | Threshold |
|---|---|---|
| G1 | Monte Carlo 5th-pct Sharpe | > 0.5 |
| G1 | Monte Carlo % profitable | > 80% |
| G2 | Top-10 removal remaining sum | > 0 |
| G3 | 3× slippage OOS Sharpe | > 0.5 |
| G4 | Max consecutive negative WFO folds | ≤ 2 |
| G5 | Regime Sharpe > 0 in ≥ 2/3 ADX + ≥ 1/2 HMM + ≥ 2/3 vol terciles | Required |
| G6 | Annualised alpha > 0, abs(beta) < asset-class threshold | Required |

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

### Portfolio Heat & Margin Constraints (Mandatory)

When deploying multiple instances of the strategy or operating in an ensemble, strict correlative and exposure limits must be enforced to prevent account blow-ups due to unbounded "Portfolio Heat". 

**Equities (Gross Exposure Limit):**
- **Limiter:** Gross Exposure (Total Notional value of all open positions / Account Equity).
- **Rule:** Max Gross Exposure must be strictly capped (e.g., 100% or 150% depending on account margin type).
- **Correlation:** Sector correlation limits must be applied (e.g., never hold > 3 highly correlated tech stocks simultaneously).

**FX & Leveraged Derivatives (Margin Usage Limit):**
- **Limiter:** Margin Usage (Total Margin tied up in open positions / Account Equity).
- **Rule:** Never tie up more than 15-20% of total account equity in margin simultaneously. 
- **Correlation:** Strict limits on base/quote exposure (e.g., maximum 2 pairs simultaneously long the USD). Because entries are highly leveraged (20x-50x), unconstrained Gross Notional Exposure to a single currency can easily exceed 2000%+ of account equity, leading to immediate margin calls on minor overnight price gaps.

### TOML Configuration

Live parameters live in `config/ic_generic.toml`, one section per instrument:

```toml
[EUR_USD]
asset_class       = "fx_major"     # cost profile: fx_major | fx_cross | equity_lc | etf | futures
direction         = "both"         # "long_only" for equities/ETFs, "both" for FX/futures
signals           = ["accel_rsi14", "accel_stoch_k"]   # from Phase 2 leaderboard
tfs               = ["W", "D", "H4", "H1"]             # TF stack; base TF = last element
threshold         = 0.75           # z-score entry trigger — set by phase6_deploy.py, not by hand
risk_pct          = 0.01           # fraction of equity risked per trade (human-validated)
stop_atr_mult     = 1.5            # ATR14 multiplier for stop distance (human-validated)
leverage_cap      = 20.0           # max position leverage (human-validated)
warmup_bars       = 1000           # bars per TF loaded at startup for IS calibration
phase3_max_dd_pct = 8.2            # Phase 3 OOS max DD % — enables live DD halt gate at 1.5×
phase3_trade_rate = 4.5            # expected trades/month — enables frequency monitoring gate
```

**Fields auto-updated by `phase6_deploy.py`:** `threshold`, `phase3_max_dd_pct`, `phase3_trade_rate`.

**Fields that are human-validated and never auto-updated:** `risk_pct`, `stop_atr_mult`, `leverage_cap`.

> [!CAUTION]
> Never edit `threshold` by hand. It was selected on IS data and validated on OOS.
> Changing it retroactively is look-ahead bias. Use `scripts/phase6_deploy.py` instead.

### Config Handoff (phase6_deploy.py)

After Phase 5 passes all 6 gates, push the Phase 3 threshold to config:

```bash
# 1. Verify Phase 5 passed
cat .tmp/reports/phase5_eur_usd.csv

# 2. Dry-run: preview what will change
uv run python scripts/phase6_deploy.py \
  --instrument EUR_USD --asset-class fx_major --dry-run

# 3. Write on confirmation
uv run python scripts/phase6_deploy.py \
  --instrument EUR_USD --asset-class fx_major

# 4. For equities
uv run python scripts/phase6_deploy.py \
  --instrument SPY --asset-class etf --direction long_only
```

The script:
- Checks `.tmp/reports/phase5_{instrument}*.csv` — warns if missing or has FAIL gates
- Finds the most recent `phase3_{instrument}*.csv` automatically
- Proposes a diff and requires explicit `y` confirmation before writing
- Never touches `risk_pct`, `stop_atr_mult`, or `leverage_cap`

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

### Phase Scripts

| Script | Phase | Purpose |
|---|---|---|
| `research/ic_analysis/phase3_backtest.py` | 3 | IS/OOS full-friction backtest — asset-agnostic, `--direction` flag, cost sensitivity sweep (0.5–3×), RoR gate |
| `research/ic_analysis/phase4_wfo.py` | 4 | Walk-forward optimisation — `--wfo-type rolling\|anchored`, per-fold recalibration, stitched equity curve |
| `research/ic_analysis/phase5_robustness.py` | 5 | Robustness validation — all 6 gates (G1 Monte Carlo, G2 top-N, G3 3× slip, G4 WFO folds, G5 regime, G6 alpha/beta) |

### Orchestrators

| Script | Phases | Purpose |
|---|---|---|
| `research/ic_analysis/pipeline_validation.py` | 3–5 | Chains Phase 3 → 4 → 5 for any instrument batch; skips Phase 4–5 if Phase 3 OOS/IS ratio < 0.5 |

### Live Deployment

| Script / File | Phase | Purpose |
|---|---|---|
| `titan/strategies/ic_generic/strategy.py` | 6 | NautilusTrader live strategy — any asset class, TOML-driven, replaces `ic_mtf` and `ic_equity_daily` |
| `config/ic_generic.toml` | 6 | Per-instrument live parameters (threshold, signals, TFs, leverage, direction) |
| `scripts/phase6_deploy.py` | 6 | Research → config handoff; reads Phase 3 CSV, checks Phase 5 gate report, writes threshold to `ic_generic.toml` |

> **Archived scripts** (superseded, kept for reference): `research/ic_analysis/_archive/` contains all
> instrument-specific pipelines (`run_ic_backtest.py`, `run_wfo.py`, `run_ic_robustness.py`,
> `run_spy_strategy.py`, `run_equity_longonly_pipeline.py`, and 15 others).

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
| **1.2** | 2026-03-20 | Audit fixes: Asset-class beta thresholds in Gate 6 (A5), R-unit Balsara normalisation (C3), holding histogram timeframe awareness (A3), time-varying swap rates (C8 via `swap_curve.py`), combined Sharpe from blended returns (C2), OOS years fix for daily bars (C5), per-fold threshold search (G7), volatility tercile gate in G5 (G3), dual Trade/Daily Sharpe gating (G6). Added `decimal.Decimal` compliance note (R5). |
| **1.1** | 2026-03-20 | Updated Scripts Reference to reflect pipeline restructure: `phase3_backtest.py`, `phase4_wfo.py`, `phase5_robustness.py`, `pipeline_validation.py` replace all instrument-specific scripts (archived to `_archive/`). Phase 6 TOML expanded with `signals`, `tfs`, `asset_class`, `phase3_max_dd_pct`, `phase3_trade_rate`. Added Config Handoff subsection documenting `phase6_deploy.py` workflow. `ic_generic/strategy.py` replaces `ic_mtf` and `ic_equity_daily`. |
| **1.0** | 2026-03-20 | Initial unified directive. Consolidated from IC MTF Backtesting Guide, MTF Optimization Protocol, and IC Equity Daily Strategy into a single asset-agnostic process spec. Added 6 improvements: Sharpe definition clarity, cost sensitivity sweep, regime robustness gate (ADX+HMM), alpha/beta decomposition, holding period distribution, per-trade % return warning. Risk of Ruin gate added to Phase 3. Kill switch triggers added to Phase 6. |
