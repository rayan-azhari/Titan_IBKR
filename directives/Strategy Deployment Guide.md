# Strategy Deployment Guide

> Last updated: 2026-04-05
> Covers: deploying validated research strategies to production

---

## 1. Deployment Pipeline

Research strategies follow this path to production:

```
research/ (WFO validated)
    -> titan/strategies/{name}/strategy.py  (NautilusTrader strategy class)
    -> config/{name}.toml                   (parameters)
    -> scripts/run_live_{name}.py           (runner script)
    -> Paper trade (min 30 days)
    -> Live
```

### Quality gates before deployment

- [ ] WFO Sharpe > 0.5 with >= 50% positive folds
- [ ] IS/OOS parity >= 0.5 on at least one parameter set
- [ ] Minimum 50 OOS trades (or 10 for slow daily strategies)
- [ ] Documented in `directives/System Status and Roadmap.md`
- [ ] `uv run ruff check . --fix && uv run ruff format . && uv run pytest tests/ -v`

---

## 2. Strategies Ready for Deployment

### 2.1 GLD AND-Gated Multi-Scale Confluence

| Property | Value |
|---|---|
| **Research** | `research/ic_analysis/run_confluence_wfo.py` |
| **Signal** | `trend_mom` = sign(ma_spread_5_20) x \|rsi_14_dev\| / 50 |
| **Scales** | H1 (native), H4 (x4), D (x24), W (x120) on single H1 stream |
| **Entry** | AND-gate: all 4 scales agree on direction, z-score > threshold |
| **Exit** | Score flips or exit buffer hit |
| **WFO Sharpe** | +1.46 (5 folds, 80% positive) |
| **CAGR** | +21.6% |
| **Max DD** | -12.0% |
| **Instrument** | GLD (ARCA) |
| **Timeframe** | H1 bars |
| **Config** | `config/gld_confluence.toml` (to create) |
| **Runner** | `scripts/run_live_gld_confluence.py` (to create) |
| **Client ID** | 14 |

**To deploy:**
1. Create `titan/strategies/gld_confluence/strategy.py` -- port AND-gate logic from `research/ic_analysis/run_multiscale_confluence.py`
2. Use `build_multiscale_signals()` from `phase1_sweep.py` for signal computation
3. Warmup from `data/GLD_H1.parquet`
4. Wire to PortfolioRiskManager

---

### 2.2 AUD/JPY MR + Confluence Regime Filter

| Property | Value |
|---|---|
| **Research** | `research/mean_reversion/run_confluence_regime_wfo.py` |
| **Signal** | VWAP deviation grid (95/98/99/99.9 percentile tiers) |
| **Regime gate** | rsi_14_dev confluence disagreement (scales must disagree = ranging) |
| **Entry** | Tiered grid [1,2,4,8] when deviation exceeds percentile threshold |
| **Exit** | 50% reversion TP, or 21:00 UTC hard close |
| **WFO Sharpe** | +2.08 (4 folds, 75% positive) |
| **Win Rate** | 60-71% |
| **Avg Hold** | ~8 hours (intraday) |
| **Instrument** | AUD/JPY (IDEALPRO) |
| **Timeframe** | H1 bars |
| **Config** | `config/mr_audjpy.toml` (to create) |
| **Runner** | `scripts/run_live_mr_audjpy.py` (to create) |
| **Client ID** | 15 |

**To deploy:**
1. Adapt `titan/strategies/mr_fx/strategy.py` pattern for AUD/JPY
2. Add confluence disagreement regime gate (compute 4-scale rsi_14_dev, require disagreement)
3. Warmup from `data/AUD_JPY_H1.parquet`

---

### 2.3 Bond->Gold Momentum (IEF->GLD)

| Property | Value |
|---|---|
| **Research** | `research/cross_asset/run_bond_equity_wfo.py` |
| **Signal** | IEF 60-day log-return (bond momentum) |
| **Entry** | Long GLD when IEF momentum z-score > 0.50 (bonds rising) |
| **Exit** | IEF momentum z-score drops below threshold after min hold period |
| **WFO Sharpe** | +1.17 (37 folds, 68% positive) |
| **Hold period** | 20+ days (monthly rebalance pace) |
| **Instrument** | GLD (ARCA), with IEF as signal source |
| **Timeframe** | Daily bars |
| **Config** | `config/bond_gold_momentum.toml` (to create) |
| **Runner** | `scripts/run_live_bond_gold.py` (to create) |
| **Client ID** | 16 |

**To deploy:**
1. Create `titan/strategies/bond_gold/strategy.py` -- simple: load IEF daily for signal, trade GLD
2. Subscribe to daily bars for both GLD and IEF
3. Compute IEF momentum on warmup data, z-score calibrated on rolling 504-day window
4. Long GLD when z > threshold, hold for minimum 20 days

---

### 2.4 MR FX EUR/USD (existing, missing runner only)

| Property | Value |
|---|---|
| **Strategy** | `titan/strategies/mr_fx/strategy.py` (already exists) |
| **Config** | `config/mr_fx_eurusd.toml` (already exists) |
| **Instrument** | EUR/USD (IDEALPRO) |
| **Timeframe** | M5 bars |
| **Runner** | `scripts/run_live_mr_fx.py` (MISSING -- to create) |
| **Client ID** | 17 |

---

## 3. Client ID Assignment (updated)

| Runner | Default Client ID |
|--------|:-:|
| watchdog_mtf.py | 1 |
| run_live_orb.py | 2 |
| run_live_etf_trend.py | 3 |
| run_live_ic_mtf.py | 4 |
| run_live_ml.py | 1 |
| run_live_gold_macro.py | 10 |
| run_live_pairs.py | 11 |
| run_live_fx_carry.py | 12 |
| run_live_gap_fade.py | 13 |
| **run_live_gld_confluence.py** | **14** |
| **run_live_mr_audjpy.py** | **15** |
| **run_live_bond_gold.py** | **16** |
| **run_live_mr_fx.py** | **17** |
| kill_switch.py | 98 |

---

## 4. Deployment Checklist

For each new strategy:

- [ ] Strategy class in `titan/strategies/{name}/strategy.py`
- [ ] `__init__.py` in the strategy directory
- [ ] Config TOML in `config/`
- [ ] Runner script in `scripts/`
- [ ] Registered with PortfolioRiskManager
- [ ] Warmup data file exists in `data/`
- [ ] `uv run ruff check . --fix && uv run ruff format .`
- [ ] `uv run pytest tests/ -v` passes
- [ ] Added to Paper Trading Guide
- [ ] Added to System Status and Roadmap
- [ ] Paper traded for minimum 30 days before live
