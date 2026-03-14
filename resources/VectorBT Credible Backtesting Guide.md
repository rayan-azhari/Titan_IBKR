# Credible Backtesting in VectorBT: A Practitioner's Guide

**Derived from:** MTF Confluence Strategy — Rounds 1, 2 & 3 robustness validation (2025)
**Applies to:** Any VectorBT `from_signals` strategy on H1 FX or equity data

---

## The Core Problem

A backtest is a simulation of the past. Without discipline, it becomes a simulation of what you
*wish* had happened. Four categories of error corrupt every naive backtest:

1. **Execution bias** — fills at prices that were never achievable
2. **Friction omission** — ignoring costs that erode returns in live trading
3. **Brittleness** — results that hold only at exactly the tested parameters
4. **Path dependency** — returns that depend on a lucky sequence of trades

This guide documents every fix applied to move the MTF strategy from a "paper holy grail"
(OOS Sharpe 6.40, zero stop-loss) to a credible validated result (OOS Sharpe 2.73, with
slippage, carry, next-bar fills, and six robustness gates).

---

## Part 1: Eliminating Execution Bias

### 1.1 Look-Ahead Bias (Same-Bar Fill)

**The flaw:**
VectorBT's default is to fill entries and exits at the close price of the *signal bar*.
In live trading, a signal generated at bar-close cannot be filled until the *next* bar opens.
Using the close of the signal bar assumes you know the price before the bar ends — this is
look-ahead bias.

**Why it matters:**
On H1 EUR/USD, the difference between close[t] and open[t+1] averages 0.3–0.8 pips.
Over hundreds of trades this adds up to a material Sharpe inflation.

**The fix:**
```python
# In run_vbt_portfolio():
if open_prices is not None:
    kwargs["price"] = open_prices.shift(-1).reindex(close.index).ffill()
```

`open_prices.shift(-1)` at position t gives you `open[t+1]` — the price available at the
start of the next bar, which is when a market order would realistically execute.
The last bar gets `.ffill()` which is negligible (one bar).

**Verification:**
Run IS/OOS with and without this fix. Sharpe should decline slightly. If it improves,
the signal was look-ahead dependent.

---

### 1.2 Stop-Loss Consistency

**The flaw:**
The original screenshot showed a 67% win rate *without* a stop-loss in the simulation,
while the actual trading plan used a 1x ATR hard stop. These are different probability
distributions. The stop strips out the "breathing room" that inflated the win rate.

**The fix:**
Always include the stop-loss in the backtest using the same parameters as the live strategy:

```python
# ATR-based fractional stop (fraction of entry price)
tr = pd.concat([
    h1_df["high"] - h1_df["low"],
    (h1_df["high"] - h1_df["close"].shift(1)).abs(),
    (h1_df["low"]  - h1_df["close"].shift(1)).abs(),
], axis=1).max(axis=1)
atr14 = tr.rolling(14).mean()
sl_stop = (atr14 / close).ffill()   # fraction: ~0.008 for EUR/USD H1

# Pass to VBT:
kwargs["sl_stop"] = sl_stop.reindex(close.index).ffill().fillna(sl_stop.median())
kwargs["sl_trail"] = False           # fixed stop from entry price, not trailing
```

**Rule:** The stop-loss in the backtest must be identical to the stop-loss you will use live.
Any discrepancy invalidates the P&L distribution.

---

## Part 2: Friction Calibration

### 2.1 Fees (Spread + Commission)

```python
FEES = 0.00015   # 1.5 pips per side (EUR/USD typical spread at IBKR)
```

For EUR/USD H1 on IBKR:
- Raw spread: ~1.0–1.5 pip during London/NY session
- IBKR commission: ~$2/side per 100k lot ≈ 0.2 pip equivalent
- Round-trip: ~3–4 pips total

Pass `fees=FEES` to every `vbt.Portfolio.from_signals()` call. Never omit this.

---

### 2.2 Slippage

**The flaw:**
Signal-triggered market orders in FX do not fill at the exact price you see. The difference
between the theoretical fill price and the actual fill is slippage — caused by queue position,
latency, and spread widening at volatile moments.

**Calibration:**
- Normal H1 conditions: 0.5–1.0 pip per side
- High volatility (news, GFC, COVID): 2–5 pips per side
- **Conservative default: 1.0 pip per side**

```python
SLIPPAGE = 0.00010   # 1.0 pip per side (EUR/USD, H1 market orders)

# In run_vbt_portfolio():
if slippage > 0:
    kwargs["slippage"] = slippage
```

**Important:** `slippage` and `fees` are separate VBT parameters. Slippage is applied
to the fill price. Fees are applied to the notional value. Do not conflate them.

**Slippage stress test — run this for every strategy:**
```python
SLIPPAGE_LEVELS = [0.00005, 0.00010, 0.00015, 0.00020, 0.00030]  # 0.5 to 3.0 pips

for slip in SLIPPAGE_LEVELS:
    pf = run_vbt_portfolio(..., slippage=slip)
    # Record Sharpe, CAGR, MaxDD
```

**Pass criteria:** Sharpe > 1.0 at all 5 levels.
If it breaks below 1.0 at 1.5 pip, the strategy is fragile to execution friction.

**MTF result:** Sharpe degraded from 2.83 → 2.34 across 0.5 to 3.0 pip — never below 1.0. ROBUST.

---

### 2.3 Carry Cost (Tom-Next / Overnight Swap)

**The flaw — symmetric model (WRONG):**
A naive model assumes long pays carry and short *receives* carry:
```python
# WRONG: shorts never receive carry on a retail FX broker
trades["carry_cost"] = trades["dir"] * carry_per_hour * trades["hours_held"] * INIT_CASH
# dir=-1 for shorts → negative cost = credit. This is incorrect.
```

**The flaw — symmetric model produces near-zero net carry:**
When the long/short trade count is balanced (~262 vs 244 in the MTF OOS period),
the credits roughly offset the costs, giving a misleadingly small ~$12 net carry cost.
This masks the true friction.

**The correct model — asymmetric broker rates:**
IBKR applies a markup on the Tom-Next differential for *both* legs. When the rate
differential is small (EUR/USD near parity), this markup can exceed the raw rate diff,
meaning **both long and short positions pay a net carry cost**.

```python
# CORRECT: both sides pay, rates are asymmetric
CARRY_LONG_ANNUAL  = 0.020   # ~2.0%/yr: long EUR/USD pays USD−EUR diff + IBKR markup
CARRY_SHORT_ANNUAL = 0.008   # ~0.8%/yr: short EUR/USD also pays (markup > rate diff)

long_cost_hr  = CARRY_LONG_ANNUAL  / (252 * 24)
short_cost_hr = CARRY_SHORT_ANNUAL / (252 * 24)

trades["carry_cost"] = np.where(
    is_long,
    trades["hours_held"] * long_cost_hr  * INIT_CASH,
    trades["hours_held"] * short_cost_hr * INIT_CASH,
)
total_carry = float(trades["carry_cost"].sum())   # always > 0
```

**Calibration guidance:**
- Check your broker's actual Tom-Next rates in the IBKR platform (varies by currency pair,
  direction, and the prevailing central bank rate differential)
- EUR/USD long: typically USD SOFR − ECB rate + IBKR markup
- EUR/USD short: typically ECB rate − USD SOFR + IBKR markup (markup dominates near parity)

**MTF result:** Corrected carry cost = $12,444 over 6.4yr OOS (+12.44% equity).
Carry-adjusted return: +202.25% → +189.81%. Meaningful but not strategy-breaking.

---

## Part 3: Position Sizing and the Compounding Trap

### 3.1 The VectorBT Default Size Trap

When you call `vbt.Portfolio.from_signals()` without a `size` parameter, VBT defaults to
**all-cash sizing** — it deploys the full portfolio value on every trade. This compounds
aggressively. A strategy that makes 40 small wins in a row will appear to have extraordinary
CAGR because each win reinvests the full enlarged equity.

This inflates both CAGR and Sharpe, and compresses the MaxDD percentage relative to the
profit percentage — producing the suspicious 5:1 R/D ratios that trigger alarm bells.

### 3.2 The Fixed-Notional Diagnostic

To isolate whether the edge comes from the signal or from compounding mechanics, run the
strategy with fixed notional sizes:

```python
# VBT SizeType.Percent does NOT support signal reversals.
# Use size_type="value" with a fixed dollar amount instead.
SIZING_LEVELS = [1.0, 0.10, 0.01]   # 100%, 10%, 1% of INIT_CASH

for sz in SIZING_LEVELS:
    pf = run_vbt_portfolio(
        ...,
        size_pct=sz,   # deploys sz * INIT_CASH per trade (fixed notional)
    )

# In run_vbt_portfolio():
if size_pct is not None:
    kwargs["size"] = size_pct * INIT_CASH
    kwargs["size_type"] = "value"
```

**Interpretation:**
- If Sharpe stays stable across 100%, 10%, and 1% sizing → the edge is per-trade signal quality.
  The 5:1 R/D is genuine, not a compounding artefact.
- If Sharpe collapses at 10% → the profitability was driven by compounding, not signal quality.
  The strategy lacks a real edge per trade.

**MTF result:**

| Size    | Sharpe | CAGR%  | MaxDD% |
|---------|--------|--------|--------|
| 100%    | 2.714  | +17.81 | -5.42  |
| 10%     | 2.699  |  +2.35 | -0.58  |
| 1%      | 2.686  |  +0.24 | -0.06  |

Sharpe is essentially flat. **The edge is in the signal**, not the compounding.

---

## Part 4: IS/OOS Split and Validation Protocol

### 4.1 The Split

```python
IS_SPLIT = 0.70   # 70% in-sample for training/optimisation, 30% OOS for honest evaluation

# Split on active-signal rows, not calendar rows.
# This avoids the IS period being artificially inflated by flat (no-signal) time.
active_idx = primary[primary != 0].index
split_at   = active_idx[int(len(active_idx) * IS_SPLIT)]
```

**Never split on calendar rows if signals are sparse.** A flat period of 6 months at the
start will push the split point earlier in calendar time, giving OOS less volatile history
and less representative data.

### 4.2 IS/OOS Consistency Check

The OOS/IS Sharpe ratio is the primary gate for overfitting:

```python
ratio = oos_sharpe / is_sharpe
# Reject if ratio < 0.5: OOS Sharpe is less than half the IS Sharpe
# This indicates parameter fitting exploited historical noise
```

**Pass criteria:** OOS/IS ratio ≥ 0.5

**MTF result:** Raw Primary OOS/IS = 0.98 ✓ (IS 2.79, OOS 2.73 — near-identical, no overfitting)

### 4.3 What Sharpe Ratios Mean in Context

| Sharpe | Interpretation |
|--------|----------------|
| < 0.5  | Not worth trading |
| 0.5–1.0 | Marginal — will lose to friction in bad years |
| 1.0–2.0 | Credible, institutional quality |
| 2.0–3.0 | Excellent — verify look-ahead bias carefully |
| > 3.0  | Suspicious in live FX — recheck fills, split, and stop consistency |
| > 4.0  | Almost certainly contains a data or methodology error |

Renaissance Technologies Medallion Fund: ~2.0–4.0 Sharpe (after massive friction).
An OOS Sharpe of 6.40 with no stop-loss is a red flag, not an achievement.

---

## Part 5: Monte Carlo Robustness

### 5.1 What It Tests

The Monte Carlo trade shuffle reveals **path dependency**. If a strategy's equity curve relies
on a specific lucky sequence of winning and losing trades, shuffling that sequence will reveal it.
A robust strategy should show similar final equity and Sharpe regardless of trade order.

### 5.2 Implementation

```python
def monte_carlo_shuffle(pf, n_sims: int = 1000) -> None:
    pnl = pf.trades.records_readable["PnL"].values   # extract per-trade P&L
    n_trades = len(pnl)

    # Derive actual trade frequency for correct Sharpe annualization
    oos_days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
    oos_years = float(oos_days) / 365.25
    trades_per_year = n_trades / oos_years            # e.g. 78 trades/yr

    rng = np.random.default_rng(42)
    finals, sharpes, min_equities = [], [], []

    for _ in range(n_sims):
        shuffled = rng.permutation(pnl)
        equity = np.empty(n_trades + 1)
        equity[0] = INIT_CASH
        for i, p in enumerate(shuffled):
            equity[i + 1] = equity[i] + p

        finals.append(float(equity[-1]))
        min_equities.append(float(equity.min()))

        ret = equity[1:] / equity[:-1] - 1.0
        std = float(ret.std())
        # CRITICAL: annualize at trade frequency, NOT bar frequency
        sharpe = float(ret.mean()) / std * np.sqrt(trades_per_year) if std > 0 else 0.0
        sharpes.append(sharpe)
```

### 5.3 The Annualization Bug (Common Mistake)

**Wrong:**
```python
sharpe = ret.mean() / std * np.sqrt(252 * 24)   # H1 bars per year
```
This treats each equity step as if it were one H1 bar. But the equity array has one point
per *trade*, not per bar. With ~78 trades/yr and 6048 bars/yr, this inflates the annualization
factor by √(6048/78) ≈ 8.8×, producing Sharpe values of 16–18 instead of the real ~2.0.

**Correct:**
```python
trades_per_year = n_trades / oos_years           # actual trade frequency
sharpe = ret.mean() / std * np.sqrt(trades_per_year)
```

**MTF result before fix:** 5th-pct Monte Carlo Sharpe = ~16 (nonsensical)
**MTF result after fix:** 5th-pct Monte Carlo Sharpe = 1.80 (credible, consistent with OOS 2.73)

### 5.4 Pass Criteria

```
5th-pct Sharpe > 0.5    AND    % of sims ending profitable > 80%
```

If < 80% of shuffled orderings are profitable, the strategy depends on a lucky sequence.
If 5th-pct Sharpe < 0.5, the worst plausible ordering breaks the strategy.

**MTF result:** 100% profitable, 5th-pct Sharpe 1.80. ROBUST.

---

## Part 6: Rolling Walk-Forward Validation

### 6.1 What It Tests

Monte Carlo proves path independence *within* a period. Rolling WFO tests whether the strategy
works *across different market regimes* — trending, ranging, high-volatility, low-volatility.

The key distinction from parameter WFO: **no re-fitting per window**. The parameters from
`config/mtf.toml` are fixed. This tests signal quality, not the ability to overfit each window.

### 6.2 Implementation

```python
TRAIN_YEARS = 2     # anchor window (signal must be robust to this period before testing)
TEST_MONTHS = 6     # non-overlapping test windows

start = close.index[0]
anchor_start = start + pd.DateOffset(years=TRAIN_YEARS)

t = anchor_start
while t + pd.DateOffset(months=TEST_MONTHS) <= end:
    test_start = t
    test_end   = t + pd.DateOffset(months=TEST_MONTHS)

    mask = (close.index >= test_start) & (close.index < test_end)
    # Run backtest on this window with FIXED parameters
    pf = run_vbt_portfolio(close[mask], ..., sl_stop=sl[mask])
    # Record: Sharpe, CAGR, MaxDD, WinRate, n_trades

    t += pd.DateOffset(months=TEST_MONTHS)   # step forward by one test window
```

### 6.3 Pass Criteria

```
% windows with positive Sharpe > 70%    AND    max consecutive negative windows ≤ 2
```

Three or more consecutive negative Sharpe windows suggests a prolonged regime where the
strategy's core logic breaks down. Two consecutive is acceptable (all strategies have bad
quarters); three consecutive means the signal is genuinely failing.

**MTF result:** 35/38 windows positive (92%), max consecutive negative = 1. ROBUST.

### 6.4 Interpreting WFO Window Results

Pay attention to *which* windows are negative:
- Negative in one isolated period → regime blip, acceptable
- Negative in a cluster at the end → signal may be deteriorating in the current regime
- All negative windows in high-volatility periods → strategy is regime-conditional

**MTF negative windows:** #22 (H2-2017, low-vol EUR squeeze), #25 (H1-2019, ranging market),
#38 (H2-2025, recent period). No structural deterioration pattern.

---

## Part 7: ATR Stop Multiplier Sensitivity

### 7.1 What It Tests

If a strategy is only profitable at exactly 1.0x ATR stop-loss and collapses at 1.1x, the
result is overfit to that exact parameter. A real edge should be robust across a reasonable
range of stop distances.

### 7.2 Implementation

```python
multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

for mult in multipliers:
    sl = sl_stop_1x * mult    # sl_stop_1x = ATR14 / close
    pf = run_vbt_portfolio(..., sl_stop=sl)
    # Record Sharpe, CAGR, MaxDD, WinRate
```

### 7.3 Pass Criteria

Sharpe > 1.0 in at least 5/8 multiplier scenarios.

**MTF result:** 8/8 scenarios above 1.0. Sharpe *improves* with wider stops (2.73 → 2.94),
suggesting the tight 1x ATR stop is cutting off winners prematurely. ROBUST — and actionable.

---

## Part 8: The Complete Validation Checklist

Run these gates in order. A FAIL at any gate requires investigation before proceeding.

```
Gate 1: No look-ahead bias
  □ Fills at open[t+1], not close[t]
  □ Features use .shift(1) (no current-bar data in signal)
  □ IS/OOS split before any training — no OOS data touches model

Gate 2: Friction is realistic
  □ sl_stop matches the live trading stop exactly
  □ fees = spread + commission (not zero)
  □ slippage ≥ 1.0 pip/side for H1 FX market orders
  □ carry cost included (asymmetric broker model for FX)

Gate 3: IS/OOS split is honest
  □ Split on active-signal rows, not calendar rows
  □ OOS/IS Sharpe ratio ≥ 0.5  (not overfitted)
  □ OOS period contains at least one bear market or high-vol regime

Gate 4: Parameter robustness
  □ ATR sweep: Sharpe > 1.0 in ≥ 5/8 multiplier scenarios
  □ Slippage stress: Sharpe > 1.0 at 1.5 pip/side minimum

Gate 5: Monte Carlo (path independence)
  □ N ≥ 1,000 simulations
  □ 5th-pct Sharpe > 0.5
  □ > 80% of simulations ending profitable
  □ Annualization: use sqrt(trades_per_year), NOT sqrt(bars_per_year)

Gate 6: Rolling walk-forward (regime robustness)
  □ > 70% of 6-month windows positive Sharpe
  □ Max consecutive negative windows ≤ 2
  □ No structural deterioration in recent windows

Gate 7: Sizing sanity check
  □ Fixed-notional comparison (10% of INIT_CASH per trade)
  □ Sharpe at 10% sizing remains > 1.0 (edge is signal, not compounding)
```

---

## Part 9: VectorBT Implementation Template

```python
import numpy as np
import pandas as pd
import vectorbt as vbt

# ── Constants ────────────────────────────────────────────────────────────────
INIT_CASH  = 100_000.0
FEES       = 0.00015    # 1.5 pip/side spread + commission
SLIPPAGE   = 0.00010    # 1.0 pip/side market order fill slippage
FREQ       = "1h"
IS_SPLIT   = 0.70

CARRY_LONG_ANNUAL  = 0.020   # Long EUR/USD: ~2.0%/yr (broker-asymmetric)
CARRY_SHORT_ANNUAL = 0.008   # Short EUR/USD: ~0.8%/yr (markup > rate diff)


def run_vbt_portfolio(
    close, long_entries, short_entries, long_exits, short_exits,
    sl_stop=None, open_prices=None, slippage=SLIPPAGE, size_pct=None,
):
    kwargs: dict = {
        "close":         close,
        "entries":       long_entries,
        "exits":         long_exits,
        "short_entries": short_entries,
        "short_exits":   short_exits,
        "init_cash":     INIT_CASH,
        "fees":          FEES,
        "freq":          FREQ,
        "accumulate":    False,
    }

    # Gate 1: eliminate look-ahead bias
    if open_prices is not None:
        kwargs["price"] = open_prices.shift(-1).reindex(close.index).ffill()

    # Gate 2: add execution slippage
    if slippage > 0:
        kwargs["slippage"] = slippage

    # Gate 2: add stop-loss (must match live strategy)
    if sl_stop is not None:
        kwargs["sl_stop"] = sl_stop.reindex(close.index).ffill().fillna(sl_stop.median())
        kwargs["sl_trail"] = False   # fixed stop from entry

    # Gate 7: optional fixed-notional sizing
    if size_pct is not None:
        # Note: SizeType.Percent doesn't support signal reversals in VBT
        kwargs["size"]      = size_pct * INIT_CASH
        kwargs["size_type"] = "value"

    return vbt.Portfolio.from_signals(**kwargs)


def compute_carry_cost(pf) -> float:
    """Return total Tom-Next carry cost over the portfolio's lifetime (USD)."""
    if pf.trades.count() == 0:
        return 0.0
    trades = pf.trades.records_readable.copy()
    dir_col = "Direction" if "Direction" in trades.columns else "Side"
    is_long = trades[dir_col].astype(str).str.upper() == "LONG"
    trades["hours"] = (
        (trades["Exit Timestamp"] - trades["Entry Timestamp"]).dt.total_seconds() / 3600
    )
    long_hr  = CARRY_LONG_ANNUAL  / (252 * 24)
    short_hr = CARRY_SHORT_ANNUAL / (252 * 24)
    trades["carry"] = np.where(
        is_long,
        trades["hours"] * long_hr  * INIT_CASH,
        trades["hours"] * short_hr * INIT_CASH,
    )
    return float(trades["carry"].sum())


def atr_stop(h1_df: pd.DataFrame, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ATR-based fractional stop (fraction of price, e.g. 0.008)."""
    tr = pd.concat([
        h1_df["high"] - h1_df["low"],
        (h1_df["high"] - close.shift(1)).abs(),
        (h1_df["low"]  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return (atr / close).ffill()


def is_oos_split(primary: pd.Series, split: float = IS_SPLIT):
    """Split index on active-signal rows (not calendar rows)."""
    active = primary[primary != 0].index
    cut = active[int(len(active) * split)]
    return cut   # everything before cut = IS, everything from cut = OOS


def monte_carlo_sharpe(pf, n_sims: int = 1000) -> dict:
    """Shuffle trade order N times. Returns 5th-pct and median Sharpe."""
    pnl = pf.trades.records_readable["PnL"].values
    n   = len(pnl)
    oos_years = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days / 365.25
    trades_per_year = n / oos_years if oos_years > 0 else 252.0

    rng = np.random.default_rng(42)
    sharpes: list[float] = []
    finals:  list[float] = []

    for _ in range(n_sims):
        shuffled = rng.permutation(pnl)
        equity   = np.empty(n + 1)
        equity[0] = INIT_CASH
        for i, p in enumerate(shuffled):
            equity[i + 1] = equity[i] + p
        finals.append(float(equity[-1]))
        ret = equity[1:] / equity[:-1] - 1.0
        std = float(ret.std())
        sharpes.append(float(ret.mean()) / std * np.sqrt(trades_per_year) if std > 0 else 0.0)

    return {
        "pct5_sharpe":    float(np.percentile(sharpes, 5)),
        "median_sharpe":  float(np.median(sharpes)),
        "pct_profitable": float((np.array(finals) > INIT_CASH).mean() * 100),
        "pass":           np.percentile(sharpes, 5) > 0.5
                          and (np.array(finals) > INIT_CASH).mean() > 0.8,
    }
```

---

## Part 11: Secondary ML Filter Validation

If you add a machine-learning layer on top of a rule-based signal (e.g., an XGBoost classifier
to filter which signals to trade), it must pass its own validation protocol before deployment.
The MTF Confluence strategy used an XGBoost meta-model trained to predict trade win probability.
It destroyed OOS performance despite passing the IS/OOS ratio gate (0.57 > 0.5).

### 11.1 The Two Failure Modes of ML Filters

**Mode 1 — Entry overfitting:** The model learned which signals historically won in IS but
those patterns don't generalise to OOS. The filter skips valid OOS signals, reducing trade
count and degrading Sharpe. This is classic overfitting.

**Mode 2 — Exit churn (often missed):** If the ML probability also triggers mid-position
exits (e.g., "exit when prob drops below 0.60 while primary signal is still active"),
the oscillating probability creates rapid exit→re-entry cycles:

```
Primary: LONG for bars t ... t+50
meta_prob: 0.62, 0.58, 0.63, 0.57, 0.61, ...
Result:   exit t+1, re-enter t+2, exit t+3, re-enter t+4 ...
```

Each re-entry pays full round-trip costs and immediately faces stop-loss risk.
A position the raw signal would have held for 50 bars gets chopped into 10 micro-trades,
most of which hit the stop before the trade matures. This can **double** the trade count
while halving the win rate.

### 11.2 The Diagnostic: Three-Variant Test

Run three variants simultaneously and compare trade count, win rate, and Sharpe:

```python
# Variant 1: Raw signal (control)
long_entries_raw  = primary.eq(1)
long_exits_raw    = primary.ne(1)

# Variant 2: Meta-Entry-Only (entries filtered, exits follow raw signal)
# Isolates whether the ENTRY filter adds value, independent of exit churn
long_entries_meta_e = primary.eq(1) & meta_gate   # gated entry
long_exits_meta_e   = long_exits_raw               # raw exits only — no prob-gate exits

# Variant 3: Meta-Full (entries AND exits gated by probability)
long_entries_meta = primary.eq(1) & meta_gate
long_exits_meta   = long_exits_raw | (primary.eq(1) & ~meta_gate)  # probability-gate exits
```

**Reading the results:**

| Variant 2 trade count vs Variant 1 | Win rate vs Variant 1 | Interpretation |
|---|---|---|
| Lower trades, higher win rate | Sharpe ≥ Raw | Entry filter is working — exit churn was the problem |
| Lower trades, similar/lower win rate | Sharpe < Raw | Entry filter adds no value — discard ML layer |
| *Higher* trades | Any | Exit churn is active — both modes failing simultaneously |

### 11.3 MTF Meta-Model Result

| Variant | Trades | Win Rate | Sharpe |
|---|---|---|---|
| Raw (control) | 506 | 40.3% | 2.732 |
| Meta-Entry-Only | 247 | 38.9% | 1.839 |
| Meta-Full | 928 | 35.8% | 0.831 |

Three simultaneous failures:
- **Trade count increased** from 506 → 928 (exit churn active)
- **Win rate declined** from 40.3% → 35.8% (entry filter is not selecting better trades)
- **Sharpe destroyed by 1.90 points** (−0.893 from bad entries, −1.007 from exit churn)

The IS/OOS ratio of 0.57 technically passed the 0.5 gate, but the absolute OOS Sharpe
(0.831 vs raw 2.732) made the verdict unambiguous: discard the ML layer entirely.

### 11.4 Decision Rule

```
If Variant 2 (Entry-Only) OOS Sharpe ≥ Raw OOS Sharpe:
    → Keep entry gate, discard probability-gate exits
If Variant 2 OOS Sharpe is within 10% of Raw OOS Sharpe:
    → Marginal — complexity not worth it, consider discarding
If Variant 2 OOS Sharpe < 90% of Raw OOS Sharpe:
    → Discard ML filter entirely — the base signal is the alpha
```

### 11.5 Why ML Filters Fail on Signal-Following Strategies

The fundamental mismatch: the ML model is trained on TBM labels (did price hit TP or SL
within N bars?) but the VBT strategy exits on signal reversal (primary flips from long
to short), which may be weeks later. The model optimises for short-term outcome; the
strategy exits on a different event entirely. The features that predict short-term TBM
outcomes don't predict which longer-duration signal-following trades will win.

A better-matched ML approach would train on the same exit mechanism the strategy uses:
label trades by their actual P&L from entry to *signal reversal*, not from entry to
*barrier touch*.

---

## Part 10: Reporting Standards

Every strategy result should be reported in this fixed order:

```
Strategy:    [name]
Period:      [IS date range] / [OOS date range]
Instrument:  [pair/symbol, timeframe]
Friction:    fees=[x] pips/side, slippage=[x] pips/side, carry=[x]%/yr long / [x]%/yr short

IS Sharpe:            [x]
OOS Sharpe:           [x]
OOS/IS ratio:         [x]   [PASS/FAIL ≥ 0.5]
OOS CAGR:             +[x]%
OOS Max Drawdown:     -[x]%
OOS Win Rate:         [x]%
OOS Trades:           [n]

Carry-adjusted CAGR:  +[x]%

ATR Sweep:            [n]/8 scenarios Sharpe > 1.0  [ROBUST/BRITTLE]
Slippage Stress:      [n]/5 levels Sharpe > 1.0     [ROBUST/FRAGILE]
Monte Carlo (N=1000): 5th-pct Sharpe=[x], profitable=[x]%  [ROBUST/FRAGILE]
WFO (2yr/6mo):        [n]/[total] windows positive  [ROBUST/NEEDS REVIEW]
Sizing test (10%):    Sharpe=[x]  [signal edge confirmed / compounding dependent]
```

A strategy that passes all five robustness gates with realistic friction has earned the right
to be considered for live deployment. One that passes only on paper has not.

---

## Appendix: Common Red Flags at a Glance

| Symptom | Likely cause |
|---------|--------------|
| Sharpe > 4.0 OOS | Look-ahead bias or stop-loss inconsistency |
| OOS/IS ratio < 0.3 | Overfit to IS data |
| Win rate 65%+ with signal-only exits | Stop-loss not in backtest |
| MaxDD < 2% with 10x leverage | Position sizing error or missing stop |
| Monte Carlo Sharpe 10-20× OOS Sharpe | Annualization uses bar frequency on trade-level equity |
| Carry cost ≈ $0 (balanced long/short) | Symmetric carry model (wrong for brokers) |
| Sharpe collapses at 10% sizing | Edge was compounding, not signal |
| 0 negative WFO windows | Overfitting: parameters too tightly tuned to each sub-period |
| ML filter increases trade count vs raw | Probability-gate exits creating exit churn |
| Meta IS/OOS ratio "passes" 0.5 but absolute Sharpe collapses | Run 3-variant diagnostic: entry-only vs full-meta vs raw |
