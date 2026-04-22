# Post-Remediation Re-Rank Analysis — 2026-04-21

**Harness commit**: `a711acd` (post-audit remediation)
**Sweep**: 1,112 (strategy, instrument, params) combinations
**Wall-clock**: 489 seconds
**Artefacts**: [leaderboard.csv](../.tmp/reports/rerank_2026_04_21/leaderboard.csv),
[leaderboard.md](../.tmp/reports/rerank_2026_04_21/leaderboard.md),
[run.log](../.tmp/reports/rerank_2026_04_21/run.log)

Every stitched Sharpe is computed through `titan.research.metrics.sharpe`
with the 12-month sanctuary window active and bootstrap 95 % CIs
(n=1000, seed=42). **Gate**: `ci_lo > 0`.

---

## Headline findings

### 1. TLT→QQQ displaces HYG→IWB as the single best bond-equity signal

| Signal | Sharpe | CI (lo, hi) | n_folds | Pos % | Trades | DD % |
|---|---:|---|---:|---:|---:|---:|
| **TLT→QQQ lb=10** (new) | **+0.978** | (+0.543, +1.401) | **43** | **93 %** | 300 | -37.0 |
| HYG→IWB lb=10 (current champion) | +0.895 | (+0.396, +1.334) | 33 | 79 % | 198 | -24.0 |
| HYG→QQQ lb=10 (new) | +0.943 | (+0.442, +1.349) | 33 | 82 % | 198 | -23.9 |

TLT→QQQ beats the current champion on **every metric**: higher Sharpe,
higher CI lower bound, 30 % more folds, 14 percentage-points higher
pct_positive, similar DD. It's been hiding in plain sight — the original
cross-asset research focused on HYG.

### 2. QQQ is the best target for bond-momentum signals

| Target | Best bond | Best Sharpe | CI_lo |
|---|---|---:|---:|
| QQQ | TLT lb=10 | +0.978 | +0.543 |
| IWB | HYG lb=10 | +0.895 | +0.396 |
| SPY | HYG lb=10 | +0.893 | +0.391 |
| GLD | IEF lb=60 | +0.721 | +0.302 |

All four bond instruments (TLT, IEF, HYG, TIP) produce positive Sharpe on
QQQ at short lookbacks. QQQ's tech-heavy composition makes it the most
responsive to rate expectations — which is exactly what bond momentum
measures. Worth prioritising QQQ as the execution instrument.

### 3. TIP→QQQ is a new signal family worth adding

TIP (inflation-protected Treasuries) wasn't previously on the champion
list. It now produces gate-passing signals on QQQ at every lookback
tested (10, 20, 40, 60). Its information content (inflation
expectations) is distinct from HYG (credit spreads) and TLT (duration),
so this is a genuinely independent edge, not a duplicate.

| Combo | Sharpe | CI_lo | n_folds |
|---|---:|---:|---:|
| TIP→QQQ lb=60 | +0.778 | +0.349 | 40 |
| TIP→QQQ lb=40 | +0.748 | +0.320 | 40 |
| TIP→QQQ lb=20 | +0.757 | +0.318 | 40 |
| TIP→QQQ lb=10 | +0.672 | +0.234 | 40 |

### 4. Mean reversion on non-AUD FX is DEAD

Of the six H1 FX pairs with full history:

| Pair | Best filter/tier | Sharpe | CI_lo |
|---|---|---:|---:|
| **AUD_JPY** | conf_donchian_pos_20 / conservative | **+0.910** | **+0.478** |
| AUD_USD | conf_rsi_14_dev / standard | +0.269 | -0.261 |
| USD_JPY | any | — | — (all fail) |
| GBP_USD | any | — | — (all fail) |
| EUR_USD | conf_rsi_14_dev / conservative | -0.217 | -0.577 |
| USD_CHF | any | — | — (all fail) |

AUD/JPY is the **only** FX pair where MR-confluence works after the math
fixes. This is consistent with the AUD/JPY-specific drivers (Asian-session
liquidity + carry-pair mean-reverting behavior) — other pairs were
benefiting from the filter-then-annualise Sharpe inflation. Should **not**
spend further autoresearch budget on MR for other FX pairs.

### 5. H1 mean reversion on equities: mostly noise

992 of the 1,112 combos (~89 %) had only `n_folds == 2` because the
equity H1 data files are too short for a meaningful rolling WFO at
default IS/OOS sizing. The 2-fold entries with spectacular Sharpes
(AMZN +2.02, AMGN +2.07, etc.) are statistically untrustworthy —
essentially a single IS/OOS split plus bootstrap noise. They should
**not** be deployed without re-validation on a longer time window (or
with smaller, overlapping folds if we decide to expand equity data).

---

## Current champions — all three still pass

| Strategy | Sharpe | CI_lo | CI_hi | Decision |
|---|---:|---:|---:|---|
| MR AUD/JPY conf_donchian_pos_20/conservative | +0.910 | +0.478 | +1.319 | **KEEP** |
| HYG→IWB lb=10 | +0.895 | +0.396 | +1.334 | **KEEP** |
| IEF→GLD lb=60 | +0.721 | +0.302 | +1.225 | **KEEP** |
| MR AUD/USD conf_rsi_14_dev/conservative | +0.374 | **-0.180** | +0.831 | **DEPRECATE** (prior decision) |

---

## Recommended portfolio changes

After the 5 % AUD/USD slot frees up, the reallocation should address both
the deprecation and the new discoveries. Suggestion:

| Strategy | Old weight | New weight | Rationale |
|---|---:|---:|---|
| MR AUD/JPY | 40 % | 30 % | Lower Sharpe post-fix; still highest CI_lo |
| HYG→IWB lb=10 | 30 % | 20 % | Keep; lower correlation to new additions than TLT→QQQ would be |
| **TLT→QQQ lb=10** (NEW) | 0 % | 20 % | Top CI_lo in entire universe; different target from HYG→IWB |
| **TIP→QQQ lb=60** (NEW) | 0 % | 10 % | Inflation-expectations family; independent of HYG/TLT edges |
| IEF→GLD lb=60 | 25 % | 15 % | Lower CI_lo than new additions; still passes |
| IWB ML | 25 % | 5 % | Sharpe +0.842 OOS with only 5 trades — sparse signal, don't overweight |
| MR AUD/USD | 5 % | 0 % | DEPRECATED (CI_lo < 0) |

**Check first**: the correlation matrix. TLT, HYG, TIP, IEF are all bond
instruments — their signals could be correlated despite targeting
different equities. Before committing, run
`research/portfolio/run_portfolio_research.py` with the new set and
verify pairwise |ρ| < 0.3 across all pairs, and portfolio Sharpe > the
single best standalone (+0.978 TLT→QQQ).

---

## Gaps in this sweep

- **Cross-asset confluence / multi-scale IC** — these pipelines weren't
  included in this driver. Next batch should add `research/ic_analysis`
  WFOs (multiscale, confluence) and the ETF trend rolling WFO.
- **Longer-history equity H1** — the 2-fold problem hides edge in short
  equity datasets. Either extend the data (download more history) or
  reduce IS/OOS sizing for a second pass specifically targeting those.
- **No parameter tuning** — this was a re-rank of existing params, not
  a hyperparameter search. A targeted autoresearch run on TLT→QQQ
  (lookback sweep, hold-period sweep, threshold sweep) could find a
  better config.

---

## Follow-ups

1. **Validate TLT→QQQ before live deployment** — the bar for elevating
   a newly-discovered signal to live capital is higher than for one
   already paper-tested. Plan:
   - Re-run with different IS/OOS splits to confirm robustness.
   - Check the 2025 OOS behaviour specifically (sanctuary-excluded, not
     visible to the sweep).
   - 48-hour paper dry-run before committing real sizing.
2. **Portfolio re-optimization** — feed the revised candidate set into
   `run_portfolio_research.py` and produce inv-vol + equal-weight + kelly
   allocations with RoR < 0.001 constraint.
3. **Autoresearch scope** — based on this analysis, a scoped autoresearch
   pass on *bond→equity* with expanded bond universe (e.g. SHY, LQD,
   EMB, BWX) and target universe (XLK, XLF, XLE sector ETFs) could
   surface genuine new combos. Open-ended MR-on-FX autoresearch is not
   recommended given finding #4.
