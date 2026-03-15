# Directive: Multi-Timeframe Confluence Strategy

> **Status: VALIDATED & DEPLOYMENT-READY**
> Last updated: Round 3 backtesting (Mar 2026). This directive supersedes all prior versions.

---

## Validated Configuration

**Config file:** `config/mtf.toml`
**Runner:** `scripts/run_live_mtf.py`
**Strategy class:** `titan/strategies/mtf/strategy.py`

### Timeframe Weights

| Timeframe | Weight | Role |
|---|---|---|
| D (Daily) | 0.60 | Primary trend bias — dominant driver |
| H4 | 0.25 | Swing confirmation |
| H1 | 0.10 | Entry timing |
| W (Weekly) | 0.05 | Long-term context |

### Indicator Parameters (all timeframes use SMA)

| TF | MA Type | Fast | Slow | RSI Period |
|---|---|---|---|---|
| H1 | SMA | 10 | 30 | 21 |
| H4 | SMA | 10 | 50 | 21 |
| D | SMA | 13 | 20 | 14 |
| W | SMA | 13 | 21 | 10 |

**Confirmation threshold:** ±0.10

---

## Strategy Logic

### Confluence Formula

```
Score = 0.60 × Signal_D + 0.25 × Signal_H4 + 0.10 × Signal_H1 + 0.05 × Signal_W
```

Each timeframe signal ∈ {−1.0, −0.5, 0.0, +0.5, +1.0}:
- MA component: fast > slow → +0.5, else −0.5
- RSI component: RSI > 50 → +0.5, else −0.5
- Sum: −1.0 to +1.0

**Entry:** Long if Score ≥ +0.10 | Short if Score ≤ −0.10

### Exit Logic (Two-Layer)

1. **Primary — signal reversal:** Score returns to neutral zone (between −0.10 and +0.10) or flips direction. ATR stop is cancelled and position closed at market.
2. **Hard stop — 2.5× ATR:** STOP_MARKET order placed immediately after entry fill at `fill_price ± 2.5 × ATR(14, H1)`. This is a GTC order; cancelled automatically if primary exit fires first.

### Why 2.5× ATR (not tighter)

Round 3 ATR sensitivity sweep (OOS, full friction):

| ATR Mult | Sharpe | CAGR% | MaxDD% | Win Rate |
|---|---|---|---|---|
| 0.50 | 2.563 | 25.14 | −5.75 | 31.3% |
| 1.00 | 2.732 | 27.14 | −5.59 | 40.3% |
| 1.50 | 2.831 | 28.29 | −5.25 | 45.6% |
| **2.50** | **2.936** | **29.44** | **−5.12** | **50.7%** |
| 3.00 | 2.895 | 29.05 | −5.25 | 52.2% |

Sharpe peaks at 2.5×. All 8 multipliers > 1.0 — the strategy is robust across the full range. Tight stops cut winners prematurely; 2.5× gives the trade room to breathe while still capping catastrophic loss.

### Meta-Filter: Discarded

An XGBoost meta-model (trained to predict win probability) was tested and rejected. Results:

| Variant | Trades | Win Rate | OOS Sharpe |
|---|---|---|---|
| Raw signal (live config) | 506 | 40.3% | **2.732** |
| Meta-Entry-Only | 247 | 38.9% | 1.839 |
| Meta-Full | 928 | 35.8% | 0.831 |

The meta-model failed on both dimensions simultaneously:
- Entry filter skipped good trades (−0.893 Sharpe vs raw)
- Probability-gate exits caused churn: exited then re-entered the same position repeatedly (−1.007 Sharpe)

**Verdict: trade the raw MTF signal only. No ML overlay.**

---

## Risk Management

### Position Sizing

```
stop_dist = 2.5 × ATR(14, H1)
units = (equity × 0.01) / stop_dist
units = min(units, equity × 5.0 / price)   # 5× leverage cap
```

This sizes each trade so that the 2.5× ATR stop represents exactly 1% of equity loss if hit.

### Risk Limits (from `config/risk.toml`)

- Max open trades: 1 (single EUR/USD position)
- Max daily loss: 2.0%
- Max leverage: 5.0×

### Carry Cost (asymmetric broker model)

IBKR applies a markup on Tom-Next differential for both legs. Near parity (EUR/USD), **both** long and short positions pay a net carry cost:

- Long EUR/USD: ~2.0%/yr (USD SOFR − ECB rate + IBKR markup)
- Short EUR/USD: ~0.8%/yr (markup exceeds near-parity rate differential)

Round 3 carry cost: $12,444 over 6.4yr OOS (+12.44% equity).
Carry-adjusted CAGR: +27.14% → **+26%/yr** — meaningful but not strategy-breaking.

---

## Round 3 Validated Performance (EUR/USD H1, 2005–2026)

**Friction assumptions:** 1.5 pip fees/side, 1.0 pip slippage/side, next-bar open fills, 2.5× ATR fixed stop, asymmetric carry.

| Metric | Value |
|---|---|
| OOS Sharpe | **2.936** |
| IS Sharpe | 2.790 |
| OOS/IS ratio | 0.98 ✓ (≥ 0.5 gate) |
| OOS CAGR | +29.44% |
| Carry-adjusted CAGR | ~+26%/yr |
| OOS Max Drawdown | −5.12% |
| OOS Win Rate | 50.7% |
| OOS Trades | ~371 (6.4yr period) |

### Robustness Gates (all passed)

| Gate | Result |
|---|---|
| IS/OOS ratio ≥ 0.5 | 0.98 ✓ |
| Slippage stress (5 levels, 0.5–3.0 pip) | Sharpe never below 1.0 ✓ |
| ATR sensitivity (8 multipliers) | 8/8 above Sharpe 1.0 ✓ |
| Monte Carlo (N=1,000, annualized at 78 trades/yr) | 5th-pct Sharpe 1.80, 100% profitable ✓ |
| Rolling WFO (2yr anchor / 6mo windows) | 35/38 windows positive (92%), max consecutive negative = 1 ✓ |
| Fixed-notional sizing (1%, 10%, 100%) | Sharpe flat — edge is signal, not compounding ✓ |

---

## Live Execution

### Prerequisites

- `IBKR_ACCOUNT_ID` set in `.env`
- `IBKR_ENVIRONMENT=practice` (start here, switch to live only after a clean paper session)
- Parquet warmup files present: `data/EUR_USD_H1.parquet`, `data/EUR_USD_H4.parquet`, `data/EUR_USD_D.parquet`, `data/EUR_USD_W.parquet`

### Run Command

```bash
uv run python scripts/run_live_mtf.py
```

### Expected Startup Sequence

```
Data download finished.
MTF Strategy Started. Warming up...
Loading H1 warmup from data/EUR_USD_H1.parquet
Loading H4 warmup from data/EUR_USD_H4.parquet
Loading D  warmup from data/EUR_USD_D.parquet
Loading W  warmup from data/EUR_USD_W.parquet
Warmup complete. ATR stop mult: 2.5x. Ready for signals.
```

Dashboard prints on every bar — if MA/RSI show `??`, warmup data is missing. Re-run `scripts/download_data_mtf.py`.

### Log Monitoring

```bash
# PowerShell
Get-Content .tmp/logs/mtf_live_*.log -Wait -Tail 50
```

Key log strings:
- `ATR stop placed @ 1.08234 (2.5x ATR = 0.00250)` — stop active
- `ATR stop triggered. Fill @ ...` — hard stop fired; strategy re-evaluates signal next bar
- `Signal Flip: Long -> Short.` — primary exit + reversal
- `Signal Neutral. Closing position.` — primary exit, going flat

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard shows `??` | Warmup parquet missing | Run `uv run python scripts/download_data_mtf.py` |
| No trades in long sessions | Threshold 0.10 conservative; sideways market → no signal | Expected behaviour — wait for trend |
| Stop placed but never cancelled | Signal exit didn't run `_cancel_stops` | Check logs for `cancel_order` lines; verify position closed |
| `No account in cache` | IBKR exec client not connected | Check Gateway/TWS is running on correct port |
| Wrong timeframe weights | `config/mtf.toml` manually edited | Restore D=0.60, H4=0.25, H1=0.10, W=0.05 |
