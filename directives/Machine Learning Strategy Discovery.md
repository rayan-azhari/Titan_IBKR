# Directive: Machine Learning Strategy Discovery

> Last updated: 2026-04-03

## Status

**Full universe scan complete (28 instruments x 3 timeframes).** Two Tier A signals identified:
- **EUR_USD Daily**: OOS Sharpe +1.58, 100% positive folds (3 folds)
- **QQQ Daily**: OOS Sharpe +1.11, 57% positive folds (7 folds)
- **PSKY Daily** (miners): OOS Sharpe +0.55, 69% positive, **13 folds** (most robust)

Daily is the only viable timeframe. H1 and H4 produce no usable signals.
Gold/silver (GLD, IAU, GC=F, SI=F) produce zero signal with this approach.

> [!IMPORTANT]
> An ML meta-overlay was tested on top of the MTF Confluence strategy and **rejected**.
> The meta-model destroyed OOS Sharpe from 2.73 to 0.83. Do not add an ML filter to MTF.

**13 bugs fixed in April 2026 audit** (3 critical, 4 high, 6 medium). Previous results were
overstated due to backward returns in Sharpe, HMM lookahead, and VIX same-day lookahead.
See `research/ml/ML_STRATEGY_DOCUMENTATION.md` for full details.

---

## Architecture

79 features (52 IC signals + 7 MA + 5 regime + 3 VIX + 1 calendar + 4 momentum + 7 stochastic)
fed into XGBClassifier with regime+pullback labeling (v3). Walk-forward validation
(rolling IS=2yr, OOS=6mo).

## Execution

```bash
# Full universe scan (28 instruments, daily):
uv run python research/ml/run_52signal_classifier.py --tf D

# Single instrument:
uv run python research/ml/run_52signal_classifier.py --instrument QQQ --tf D

# Comprehensive evaluation with all stats + charts:
uv run python research/ml/run_ml_full_eval.py

# Parameter sweep:
uv run python research/ml/run_52sig_param_sweep.py --tf D --fast
```

## Full Documentation

See `research/ml/ML_STRATEGY_DOCUMENTATION.md` for:
- Feature groups and importance rankings
- Labeling methodology (v3 regime+pullback)
- Cross-asset signal map (Tier A/B/C/D)
- Comprehensive evaluation stats (Sharpe, Sortino, Calmar, max DD, risk of ruin)
- Backtest assumptions and known problems
- Portfolio construction candidates
- What's missing for a real portfolio

## Next Steps

1. Deploy PSKY as first ML live strategy (most robust: 13 folds, 69% positive)
2. Add gold cross-asset features (TIP, DXY, breakevens) for commodity signals
3. Build portfolio risk layer for realistic multi-asset sizing
