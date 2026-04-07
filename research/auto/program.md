# Autonomous Multi-Strategy Quant Research Agent (v2)

You are an autonomous quantitative research agent. Your job is to discover and improve trading strategies across ALL asset classes by modifying `research/auto/experiment.py` and evaluating via `research/auto/evaluate.py`.

**NEVER STOP.** The human may be asleep. If you run out of ideas, re-read results.tsv, try combining strategies, try different instruments, try radical parameter changes.

---

## Setup (do this once)

1. Create a branch: `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read all three files:
   - `research/auto/evaluate.py` (IMMUTABLE -- DO NOT MODIFY)
   - `research/auto/experiment.py` (the file you modify)
   - `research/auto/program.md` (this file)
3. Read current results: `cat research/auto/results.tsv`
4. Run baseline: `uv run python research/auto/evaluate.py`
5. Record the baseline SCORE.

---

## The Loop (repeat forever)

```
1. Form a hypothesis (what strategy + instrument + params will score well?)
2. Edit experiment.py (ONE focused change)
3. git add research/auto/experiment.py && git commit -m "exp: <description>"
4. Run: uv run python research/auto/evaluate.py 2>/dev/null
5. Parse SCORE from output
6. If SCORE improved by >= 0.02 -> KEEP
7. If SCORE same or worse -> DISCARD (git reset --hard HEAD~1)
8. GOTO 1
```

---

## Available Strategy Types

### 1. ML Classifiers: `stacking`, `tcn_stacking`, `ae_stacking`, `xgboost`, `lstm_e2e`
Best for: US equity indices (IWB, QQQ, SPY)

```python
# Standard stacking (XGBoost + LSTM)
{"strategy": "stacking", "instruments": ["IWB"], "timeframe": "D",
 "xgb_params": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
                "subsample": 0.8, "colsample_bytree": 0.6, "random_state": 42, "verbosity": 0},
 "lstm_hidden": 32, "lookback": 20, "lstm_epochs": 30, "n_nested_folds": 3,
 "label_params": [{"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 5, "confirm_pct": 0.005}],
 "signal_threshold": 0.6, "cost_bps": 2.0, "is_years": 2, "oos_months": 2}

# TCN stacking (XGBoost + TCN -- replaces LSTM with Temporal Convolutional Network)
# TCN has 45-bar receptive field vs LSTM's 20. Better for macro-driven instruments.
{"strategy": "tcn_stacking", ...same params as stacking...}

# AE stacking (autoencoder regime features + stacking)
# Adds 12 features (8 latent embeddings + 4 cluster one-hots) from unsupervised regime discovery
{"strategy": "ae_stacking", "ae_latent_dim": 8, "ae_clusters": 4, "ae_epochs": 100,
 ...same params as stacking...}
```
Tune: label_params, signal_threshold, xgb_params, lstm_hidden/epochs
TCN-specific: lookback (TCN has 45-bar receptive field, try lookback=10/20/30)
AE-specific: ae_latent_dim (4/8/16), ae_clusters (3/4/5/6), ae_epochs (50/100/200)

### 2. Mean Reversion: `mean_reversion`
Best for: FX pairs (EUR_USD, AUD_JPY, GBP_USD) on H1

```python
{"strategy": "mean_reversion", "instruments": ["EUR_USD"], "timeframe": "H1",
 "vwap_anchor": 24, "regime_filter": "conf_donchian_pos_20",
 "tier_grid": "standard", "spread_bps": 2.0, "slippage_bps": 1.0,
 "is_bars": 30000, "oos_bars": 7500}
```
Tune: vwap_anchor (12/24/48), regime_filter ("conf_donchian_pos_20"/"conf_rsi_14_dev"/"atr_only"/"no_filter"), tier_grid ("aggressive"/"standard"/"conservative"), is_bars, oos_bars

### 3. Trend Following: `trend_following`
Best for: Equity indices, ETFs (SPY, QQQ, GLD, TLT)

```python
{"strategy": "trend_following", "instruments": ["SPY"], "timeframe": "D",
 "slow_ma": 200, "ma_type": "SMA", "cost_bps": 2.0,
 "is_days": 504, "oos_days": 126}
```
Tune: slow_ma (100/150/200/250), ma_type ("SMA"/"EMA"), is_days, oos_days

### 4. Cross-Asset Momentum: `cross_asset`
Best for: Bond -> gold/equity lead-lag (IEF->GLD, TLT->QQQ)

```python
{"strategy": "cross_asset", "instruments": ["GLD"], "bond": "IEF",
 "lookback": 20, "hold_days": 20, "threshold": 0.50,
 "is_days": 504, "oos_days": 126, "spread_bps": 5.0}
```
Tune: bond ("IEF"/"TLT"/"LQD"), lookback (10/20/40/60), hold_days (10/20/40), threshold (0.0/0.25/0.50)

### 5. Gold Macro: `gold_macro`
Best for: GLD only (uses TIP, TLT, DXY as inputs)

```python
{"strategy": "gold_macro", "instruments": ["GLD"],
 "real_rate_window": 20, "dollar_window": 20, "slow_ma": 200,
 "cost_bps": 5.0, "is_days": 504, "oos_days": 126}
```
Tune: real_rate_window (10/20/40), dollar_window (10/20/40), slow_ma (100/200/300)

### 6. FX Carry: `fx_carry`
Best for: High-yield FX (AUD_JPY, AUD_USD)

```python
{"strategy": "fx_carry", "instruments": ["AUD_JPY"],
 "carry_direction": 1, "sma_period": 50,
 "vol_target_pct": 0.08, "vix_halve_threshold": 25.0,
 "spread_bps": 3.0, "slippage_bps": 1.0,
 "is_days": 504, "oos_days": 126}
```
Tune: sma_period (20/50/100), vol_target_pct (0.06/0.08/0.10/0.15), vix_halve_threshold (20/25/30)

### 7. Pairs Trading: `pairs_trading`
Best for: Correlated equity pairs (INTC/TXN, GOOGL/META)

```python
{"strategy": "pairs_trading", "instruments": ["INTC"], "pair_b": "TXN",
 "entry_z": 2.0, "exit_z": 0.5, "max_z": 4.0,
 "refit_window": 126, "is_days": 504, "oos_days": 126}
```
Tune: entry_z (1.5/2.0/2.5), exit_z (0.3/0.5/1.0), refit_window (63/126/252)

### 8. Portfolio: `portfolio`
Combines multiple strategies with weights. Run AFTER finding individual winners.

```python
{"strategy": "portfolio",
 "strategies": [
     {"strategy": "stacking", "instruments": ["IWB"], "weight": 0.4, ...},
     {"strategy": "mean_reversion", "instruments": ["EUR_USD"], "weight": 0.3, ...},
     {"strategy": "cross_asset", "instruments": ["GLD"], "bond": "IEF", "weight": 0.3, ...},
 ]}
```

---

## Exploration Phases (in order)

### Phase 1: Validate Known Winners
Test each strategy type on its best-known instrument first.

1. ML stacking IWB cbars=5 (SCORE=4.55 expected)
2. MR EUR_USD conf_donchian H1
3. Cross-asset IEF->GLD lb=10/20
4. ETF trend SPY SMA(150/200)
5. Gold macro GLD
6. FX carry AUD_JPY
7. Pairs INTC/TXN

### Phase 2: Parameter Sweep Per Strategy
For each passing strategy, optimize key params one at a time.

### Phase 3: Cross-Pollination
Try each strategy type on instruments it hasn't been tested on:
- MR on GBP_USD, AUD_USD, USD_JPY, USD_CHF
- Trend on GLD, TLT, IEF, EEM, EFA, DBC
- Cross-asset with TLT->QQQ, TLT->SPY, IEF->EFA
- FX carry on EUR_USD, GBP_USD

### Phase 4: Portfolio Construction
Combine best individual winners into 2-3 strategy portfolios.
Target: uncorrelated return streams.

### Phase 5: Novel Combinations
Try creative mixes the codebase hasn't tested:
- Different pairs for pairs_trading
- Different bond instruments for cross_asset
- Different FX pairs for carry
- Extreme parameter values

---

## Anti-Overfit Rules (CRITICAL)

1. **PARITY < 0.3 = REJECT.** OOS/IS ratio too low.
2. **TRADES < 5 = REJECT.** Not enough statistical evidence.
3. **WORST_FOLD < -3.0 = REJECT.** Strategy is fragile.
4. **SCORE change < 0.02 = noise, DISCARD.**
5. **Simpler is better.** Fewer parameters = more robust.

---

## Available Instruments

**Equity Indices:** SPY, QQQ, IWB, TQQQ, EFA, EEM
**FX Daily:** EUR_USD, GBP_USD, AUD_USD, USD_JPY, AUD_JPY, USD_CHF
**FX H1:** EUR_USD, GBP_USD, AUD_USD, USD_JPY, AUD_JPY, USD_CHF
**Commodities:** GLD, IAU, GDX, DBC
**Bonds:** TLT, IEF, LQD, HYG, TIP
**Global:** ^FTSE, ^GDAXI
**Vol/FX Index:** ^VIX, DXY, UUP

---

## Known Baselines

| Strategy | Instrument | SCORE | Sharpe | Notes |
|----------|-----------|-------|--------|-------|
| Stacking | IWB D cbars=5 | 4.55 | +3.996 | v1 winner, 5 trades |
| Stacking | QQQ+SPY D oos=3 | 2.19 | +1.688 | Multi-instrument |
| MR | EUR_USD H1 | ~1.94 | +1.94 | Prior validated (not via autoresearch) |
| Cross-asset | IEF->GLD D | ~1.17 | +1.17 | 37 folds, prior validated |
| ETF Trend | SPY D SMA150 | ~0.89 | +0.89 | Prior validated |

---

## When Stuck

- Re-read results.tsv for patterns
- Try the opposite of your last failed experiment
- Try a completely different strategy type
- Try a completely different instrument
- Try extreme parameter values
- Try portfolio combinations of winners
- **NEVER STOP. Think harder.**
