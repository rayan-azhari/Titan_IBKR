# Titan V2.0 — fresh start

**Date:** 2026-05-14
**Repo:** `https://github.com/rayan-azhari/titan-V2.0.git`
**Branch convention:** `main` is the cleansed V2.0 baseline; every new piece of work goes on a feature branch.

---

## Why V2.0 exists

The V1 codebase (`rayan-azhari/Titan_IBKR`) accumulated ~6 months of research, ~13 live strategies, and a series of audits that exposed methodology gaps the published Sharpe numbers depend on. Specifically:

- Sharpe annualisation conventions varied across audits, producing extreme magnitudes on sparse-trade strategies and silently inflated numbers on filtered-bar strategies.
- Walk-forward designs were inconsistent (5 contiguous folds, 5 expanding folds, 8 expanding-IS folds, 48 rolling folds — all simultaneously in flight).
- Sanctuary slicing was ad-hoc; sanctuary divergence (recent 12 months >> historical OOS) appeared in three independent audits without investigation.
- Monte Carlo gates were uniformly calibrated at P(MaxDD > 25%) < 5%, which failed every cross-asset cell including audit-trusted references.
- Decision matrices were incomplete (one audit hit UNDETERMINED).
- Same-bar look-ahead was confirmed in one ML file but untested in six others.
- DSR formula assumed normal returns (skew=0, kurt=3) — real returns aren't normal.

Acting on the resulting verdicts (`tier=unconfirmed` flags, retirement recommendations) before fixing the methodology would have been the opposite of disciplined. V2.0 is the disciplined response: discard the suspect results, keep the ideas, rebuild under a unified framework.

---

## What's on V2.0 right now

### KEPT — code + infrastructure

| Layer | What it contains |
|---|---|
| `titan/` | Live class implementations of every deployed strategy (mr_audjpy, bond_gold, etf_trend, fx_carry, gld_confluence, gold_macro, ic_equity_daily, ic_mtf, ml, mtf, orb, pairs, samir_stack, turtle, etc.). |
| `titan/research/framework/` | **The new unified backtesting framework.** Single source of truth for typology + WFO + sanctuary + MC + DSR + decision matrix. Every new audit + every re-audit goes through this. |
| `titan/research/metrics.py` | Audit-corrected shared metrics (`sharpe`, `bootstrap_sharpe_ci`, `BARS_PER_YEAR`, `rolling_zscore`, `is_frozen_zscore`, `max_drawdown`, etc.). |
| `titan/risk/` | Portfolio risk architecture: per-strategy equity tracker, FX unit conversion, halt persistence, kill switch. |
| `titan/adapters/` | IBKR / NautilusTrader adapter code. |
| `titan/portfolio/` | Portfolio risk manager, allocator, drawdown breaker. |
| `tests/` (~30 files, 53 tests passing) | Documents expected behaviour of every live class + the framework. |
| `scripts/` | Deployment + ops infrastructure: kill switch, watchdogs, validators, data downloaders, run_live_*, refresh_market_data.sh, build_image.sh. |
| `config/*.toml` | Strategy parameter intent. Values may need re-tuning under the new framework but the schema + parameter axes are the strategies' "what". |
| `data/` | Gitignored, regen via `scripts/download_data_*.py`. |

### KEPT — directives (ideas, methodology, infrastructure)

**Methodology framework + lessons:**
- `Methodology Audit & Unified Framework 2026-05-14.md` — the framework spec
- `V3.6 Lessons Catalogue.md` — 16 distilled lessons across waves 1-4
- `IC Signal Analysis.md` — IC framework v4.2

**API / Docker / deployment learnings:**
- `IBKR & NautilusTrader API Reference.md`
- `IBKR UK Paper Account Setup.md`
- `Docker Paper Trading Guide.md`
- `Paper Trading Guide.md`
- `Deployment & Operations.md`
- `Strategy Deployment Guide.md`, `Strategy Deployment Protocol.md`
- `Operational Robustness Framework 2026-05-12.md`
- `Titan Library Reference.md`, `Titan-IBKR Adapter Guide.md`, `Workspace Structure.md`
- `Market Data Refresh Strategy.md`
- `Broker Migration Assessment.md` (OANDA / Pepperstone / IG comparison)
- `Rehydration Bug 2026-05-11.md` (postmortem)

**Strategy mechanism guides (how each strategy works conceptually):**
- `ORB Trading Strategy.md`, `ORB Strategy User Guide.md`
- `MTF Strategy User Guide.md`, `Multi-Timeframe Confluence.md`
- `ETF Trend Strategy.md`
- `Turtle Trading Strategy Analysis.md`
- `Ensemble Strategy Framework.md`
- `Samir-Stack Strategy Guide.md`
- `Samir V3 — VIX-HMM Strategy Design 2026-05-13.md`
- `Deprecated Strategies.md` (historical reference)

**Methodology reference (the "how to do research"):**
- `Backtesting & Validation.md`
- `Machine Learning Strategy Discovery.md`
- `Time Series & Correlation Analysis.md`
- `Cost Model Audit 2026-05-11.md`

**Ops runbooks:**
- `operations/` subdirectory

### DISCARDED in the cleanse

- All April-22 EU strategy phase result logs
- All April-21 candidate portfolio / remediation / rerank result logs
- All May-11/12 Samir-Stack variant + paper validation directives (kept only the Strategy Guide + V3 design)
- All May-13 / May-14 IC Census directives (Phase A/B/C/D + AUDJPY + Strategy Re-validation)
- All May-14 Track 1 audit verdicts (Bond-Equity, ML Causality, MR AUDJPY, ORB DSR, Range-Expansion Phase 0)
- `System Status and Roadmap.md` (outdated)
- All `research/*/run_*.py` and `research/*/phase*.py` orchestrators (~122 files) — the suspect runners
- `research/_archive/`, `research/auto/`, `research/strategies/` subtrees
- 6 IC Census universe config files
- 2 Phase C download scripts
- 4 audit runner scripts created during the cleanse session

---

## How to use V2.0

### To audit a new strategy idea

1. **Read the V3.6 Lessons Catalogue.** Every lesson is a known failure mode.
2. **Classify the strategy** under one of the 9 `StrategyClass` values in `titan.research.framework.typology`. The class determines defaults for Sharpe convention, WFO design, MC config, decision-matrix thresholds.
3. **Write a pre-registration directive** (V3.1) listing the universe, gates, decision rule. Commit BEFORE any data is examined.
4. **Run via the framework primitives:**
   - `slice_sanctuary(df, months=12)` for the 12-month holdout
   - `build_folds(idx, cfg, bars_per_year=...)` for the WFO folds
   - `sharpe(returns, periods_per_year=...)` + `bootstrap_sharpe_ci(...)` for the headline metric
   - `deflated_sharpe(...)` for DSR
   - `run_block_mc(...)` for underlying-resampled Monte Carlo
   - `sanctuary_divergence_test(...)` to check whether the sanctuary's Sharpe is unusually high vs the historical rolling-window distribution
   - `decide(DecisionInputs(...))` for the 4-axis verdict
5. **Append the result log** to the pre-reg directive (§4).
6. **Append any new lesson** to `V3.6 Lessons Catalogue.md`.

### To re-audit an existing live strategy

Same workflow as above. The live class lives in `titan/strategies/*/strategy.py`; the audit calls into it through a thin backtest harness. **DO NOT** approximate the live class with a simplified simulator (Lesson L10).

### To run the codebase methodology scanner

```bash
python scripts/audit_codebase_methodology.py --write-parquet
```

Reports anti-patterns (hardcoded `sqrt(252)`, same-bar `position * return`, `.ffill` after `.reindex`, global z-scores, etc.) across `research/` + `titan/` + `scripts/`. False positives are expected — the script is a sieve.

### To verify the framework

```bash
python -m pytest tests/test_framework_synthetic.py tests/test_ic_census_lib.py -v
```

53 tests covering: typology completeness, sanctuary slicing + divergence test, WFO fold construction, DSR with skew/kurt, decision matrix totality, end-to-end known-edge vs known-no-edge synthetic ground truth.

---

## Status of the live strategies

**Updated 2026-05-15** — GEM is the first live strategy under a true 5-axis DEPLOY verdict. The remaining ~13 strategies in `titan/strategies/` still run on V1 verdicts pending framework re-audits.

### Production deployments under the V2.0 framework

| Live strategy | Production cell | Verdict | Pre-reg | Deployed |
|---|---|---|---|---|
| `gem` (Dual Momentum) | **J5 P_hl60_vt05** | **DEPLOY** (5/5 axes) | `Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md` | 2026-05-16 17:10 UTC (paper) |

GEM trajectory: original C12 was DEPLOY (4-axis, 2026-05-14) → demoted CONDITIONAL_WATCHPOINT under J3's 5-axis matrix (noise axis = `mid`) → J4 `A1_ewma_hl40` promoted from the noise-robust redesign sweep (EWMA half-life=40, paper live 2026-05-15 13:07 UTC) → **J5 `P_hl60_vt05` promoted via L52 hybrid framework 2026-05-16** (vol_estimator_halflife=60, ann_vol_target=0.05). The J5 audit applied the full V3.6 protocol incl. L17 relative-MC vs 60/40 SPY/IEF benchmark — J4 fails L17 (12% DD reduction, gate is 20%) while J5 passes (38%). J5 stitched-OOS Sharpe +1.00 vs J4 +0.74 (+34%), CI_lo +0.51 vs +0.24 (2.1× tighter), MaxDD -10% vs -15%. See `directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md` + V3.6 Catalogue **L52, L57**.

**L52 hybrid framework now also applied to:**

- `bond_gold` (Wave A.1, 2026-05-16) — **PROMOTED CONDITIONAL_WATCHPOINT** for `(lookback=120, threshold=0.50)`. Live V1-era config remains; migration pending allocator approval. See `docs/strategies/bond-gold.md` (TODO).
- `etf_trend` family — **SPY RETIRED** (Wave A.2; L56 — long-only equity MA-crossover fails L17 rel-MC). **TQQQ PROMOTED CONDITIONAL_WATCHPOINT** for `(slow_ma=150, exit_confirm_days=5)` (Wave A.2-confirm; L56 refined — leveraged variant survives via noise-mitigation knob). **5 unleveraged variants (QQQ/IWB/EFA/DBC/GLD) flagged for bulk-retire** per `directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md`.

### Strategies still on V1 verdicts (pending re-audit)

The remaining V1-era strategies in `titan/strategies/` (mr_audjpy, mr_fx, mtf, orb, gld_confluence, gold_macro, fx_carry, pairs, ic_equity_daily, ic_mtf, ml, samir_stack, turtle) still run per the existing `config/*.toml`. None has been re-audited under the framework. Each needs a fresh framework-audited pre-reg + result log before its current Sharpe/CI claims can be trusted. See `directives/V1-era Re-audit Sweep Roster 2026-05-16.md` for the Wave A.3+ schedule.

The Methodology Audit & Unified Framework directive's §3 lists the migration plan. The L52 hybrid framework (sweep → pre-reg → audit) is now the standard approach for any V1-era re-audit and for any new strategy entering production: 2D sweep (not 1D — L57) identifies plateau candidates, pre-reg freezes them, the full 5-axis audit on held-out sanctuary is the deployment gate.

---

## Workflow conventions

- **Branches:** every piece of work is a feature branch. Audit + research goes on `research/*` branches; deployable changes on `feat/*` or `fix/*`. Merge to `main` via PR.
- **Pre-registration discipline:** every audit's gates / cells / decision rule lands in a directive BEFORE any data is examined. After the run, only the §4 result log is appended; §1-§3 stay frozen.
- **V3.6 hygiene:** every audit produces a result log even when the verdict is NULL or RETIRE. Lessons get appended to the catalogue.
- **No retroactive cell-favouring.** If gates were too tight / too loose, the next audit gets a fresh pre-reg; you don't relax the gate on the failing audit.

---

## What's NOT here yet (and is genuinely open)

1. **Re-audit of every live strategy under the framework.** ~14 strategies in `titan/strategies/`, none yet re-audited.
2. **Cost-model validation** against actual fills (vs the framework's class-default cost models).
3. **Phase D paid-feed acquisition** (Eurex / CFE / breadth panel) — pending vendor selection.
4. **ML same-bar causality sweep** on 6 untested feature files (`multi_horizon_lstm.py`, `ensemble_stacking.py`, `lstm_classifier.py`, `autoencoder_regime.py`, `run_metalabeling.py`, `calibration_kelly.py`). The same-bar fix is already documented in the framework; testing the 6 files is a separate task.

The framework is the toolbox; the actual research builds back up from here, one strategy at a time, with pre-reg discipline and the lessons catalogue keeping us honest.
