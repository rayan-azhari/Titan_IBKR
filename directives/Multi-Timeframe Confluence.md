# Directive: Multi-Timeframe Confluence Strategy

> **Status: VALIDATED & DEPLOYMENT-READY**
> Last updated: Round 4 backtesting (Mar 2026). EUR/USD and GBP/USD both validated.
> This directive supersedes all prior versions (Round 3 / SMA / mtf.toml).

---

## Validated Configuration

**Primary config file:** `config/mtf_eurusd.toml`
**GBP/USD config:** `config/mtf_gbpusd.toml`
**Runner:** `scripts/run_live_mtf.py`
**Strategy class:** `titan/strategies/mtf/strategy.py`

### EUR/USD Timeframe Weights

| Timeframe | Weight | Role |
|---|---|---|
| D (Daily) | 0.55 | Primary trend bias — dominant driver |
| W (Weekly) | 0.30 | Long-term macro regime filter |
| H1 | 0.10 | Entry timing |
| H4 | 0.05 | Minimal — swing noise at EUR/USD scale |

**Why high Weekly weight for EUR/USD?**
EUR/USD is macro-driven. Weekly trends persist for months at a time (ECB policy cycles, risk-on/off
regimes). The 0.30 weekly weight acts as a regime filter — it suppresses false entries during
choppy counter-trend periods and amplifies conviction during clean macro trends.

### EUR/USD Indicator Parameters

| TF | MA Type | Fast | Slow | RSI Period |
|---|---|---|---|---|
| H1 | WMA | 10 | 50 | 14 |
| H4 | WMA | 10 | 40 | 14 |
| D  | WMA | 10 | 20 | 7 |
| W  | WMA | 8  | 21 | 14 |

**Confirmation threshold:** ±0.10
**ATR stop multiplier:** 4.0× (insurance; primary exit is signal reversal)

### GBP/USD Configuration

| TF | MA Type | Fast | Slow | RSI Period | Weight |
|---|---|---|---|---|---|
| D  | SMA | 5  | 20  | 10 | 0.55 |
| H4 | SMA | 10 | 30  | 14 | 0.30 |
| H1 | SMA | 20 | 100 | 21 | 0.10 |
| W  | SMA | 10 | 21  | 7  | 0.05 |

**Confirmation threshold:** ±0.10 | **ATR stop:** 4.0×

---

## Strategy Logic

### Confluence Formula

```
Score = w_D × Signal_D + w_H4 × Signal_H4 + w_H1 × Signal_H1 + w_W × Signal_W
```

Each timeframe signal ∈ {−1.0, −0.5, 0.0, +0.5, +1.0}:
- **MA component:** fast WMA > slow WMA → +0.5, else −0.5
- **RSI component:** RSI > 50 → +0.5, else −0.5
- **Sum:** −1.0 to +1.0 per timeframe

**Entry (next bar):** Long if Score ≥ +0.10 | Short if Score ≤ −0.10

> [!IMPORTANT]
> Signals are shifted by 1 bar before entry evaluation (`.shift(1)`). Signal computed at bar close
> executes at the **next bar's open**. This is enforced in both backtesting and the live strategy.

### Exit Logic (Two-Layer)

1. **Primary — signal reversal:** Score returns to neutral zone (between −0.10 and +0.10) or
   flips direction. ATR stop is cancelled first, then position closed at market.

2. **Secondary — 4.0× ATR hard stop:** `STOP_MARKET` order placed immediately after entry fill at
   `fill_price ± 4.0 × ATR(14, H1)`. GTC; cancelled automatically if primary exit fires first.

### Why Signal-Only Exit (Not Tight ATR Stop)

ATR stop sweep (OOS, EUR/USD, full friction — lower combined Sharpe = worse):

| ATR Mult | Result |
|---|---|
| 1.0–2.0 | Combined Sharpe negative — stops cut positions prematurely |
| 2.5–3.0 | Near breakeven — heavy drag on long-running trend trades |
| **4.0** | **Best Sharpe — insurance only; most exits via signal reversal** |
| Signal-only | Highest Sharpe overall — but no catastrophic loss cap |

**The market temporarily moves against MTF positions (~9–10% IS drawdown) before the trend
confirms. Tight stops are the enemy of trend-following strategies. Signal-only exit is the
theoretically optimal choice; 4.0× stop is the live compromise for catastrophic protection.**

### Meta-Filter: Discarded

XGBoost meta-model (win probability filter) was tested and rejected. Raw signal outperformed
all meta variants — the ML overlay caused churn (exit + re-entry) and skipped genuine winners.

**Do not re-introduce an ML overlay without a fresh OOS test.**

---

## Risk Management

### Position Sizing

```
stop_dist = atr_stop_mult × ATR(14, H1)
units = (equity × 0.01) / stop_dist         # 1% equity risk per trade
units = min(units, equity × 5.0 / price)    # 5× leverage cap
```

The sizing is calibrated to the 4.0× ATR stop distance so that if the stop fires, loss = 1% of equity.

### Swap Costs (from `config/spread.toml [swap]`)

Annual financing drag (symmetric — modelled as cost regardless of direction):

| Pair | Annual Drag |
|---|---|
| EUR/USD | ~1.0%/yr |
| GBP/USD | ~1.5%/yr |
| AUD/USD | ~2.0%/yr |
| USD/CAD | ~1.2%/yr |

Swap drag is modelled post-hoc in `run_portfolio.py` as `abs(position_value) × annual_rate / bars_per_year`.

---

## Round 4 Validated Performance

### EUR/USD (10yr IS/OOS, full friction + next-bar execution + swap costs)

| Metric | Long | Short | Combined |
|---|---|---|---|
| OOS Sharpe | 2.252 | 1.958 | **1.943** |
| IS Sharpe | ~2.5 | ~2.2 | — |
| OOS/IS ratio | ✓ ≥ 0.5 | ✓ ≥ 0.5 | — |
| Swap-adj CAGR | — | — | ~8%/yr |
| Max Drawdown | — | — | ~10% |
| Gates passed | 7/7 ✓ | 7/7 ✓ | — |

### GBP/USD (10yr IS/OOS, full friction)

| Metric | Long | Short | Combined |
|---|---|---|---|
| OOS Sharpe | 1.947 | 1.471 | **1.331** |
| IS Sharpe | ✓ ≥ 0.5 | ✓ ≥ 0.5 | — |
| Gates passed | 7/7 ✓ | 7/7 ✓ | — |

**EUR/USD substantially outperforms GBP/USD** (Combined Sharpe 1.943 vs 1.331).
EUR/USD is the primary deployment pair. GBP/USD can be added as a diversifying second position.

### Robustness Gates (EUR/USD, all passed)

| Gate | Result |
|---|---|
| IS/OOS ratio ≥ 0.5 | ✓ |
| OOS Sharpe ≥ 1.0 (both legs) | ✓ |
| Trades ≥ 30 in OOS | ✓ |
| Win Rate ≥ 40% | ✓ |
| Max DD ≤ 25% | ✓ |
| Monte Carlo (N=1,000): 5th-pct Sharpe > 0.5 AND >80% profitable | ✓ |
| WFO (2yr/6mo): >70% positive AND max consec. negative ≤ 2 | ✓ |

---

## Live Execution

### Prerequisites

- `IBKR_ACCOUNT_ID` set in `.env`
- `IBKR_PORT=4002` (paper) or `4001` (live)
- Parquet warmup files present in `data/`: `EUR_USD_H1.parquet`, `EUR_USD_H4.parquet`, `EUR_USD_D.parquet`, `EUR_USD_W.parquet`

### Run Command

```bash
uv run python scripts/run_live_mtf.py 2>&1 | tee .tmp/logs/mtf_stdout_$(date +%Y%m%d_%H%M%S).log
```

### Expected Startup

```
MTF Strategy Started. Warming up...
Loading H1 warmup from data/EUR_USD_H1.parquet
Loading H4 warmup from data/EUR_USD_H4.parquet
Loading D  warmup from data/EUR_USD_D.parquet
Loading W  warmup from data/EUR_USD_W.parquet
Warmup complete. ATR stop mult: 4.0x. Ready for signals.
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard shows `??` | Warmup parquet missing | Run `uv run python scripts/download_data_mtf.py` |
| No trades for days | Sideways market; threshold ±0.10 conservative | Expected — wait for trending regime (~1.5 trades/week avg) |
| Stop placed but not cancelled | `_cancel_stops()` may have failed | Check logs for `cancel_order`; cancel orphaned stop in TWS manually |
| `No account in cache` | Gateway/TWS not connected | Check port, API enabled, Read-Only unchecked |
| Wrong weights at runtime | Old `config/mtf.toml` being loaded | Confirm `config_path="config/mtf_eurusd.toml"` in `run_live_mtf.py` |
