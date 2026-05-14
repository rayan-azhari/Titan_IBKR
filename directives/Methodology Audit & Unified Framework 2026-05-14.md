# Methodology Audit & Unified Framework

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect / Risk-Auditor
**Status:** **PRE-REGISTRATION + ACTIVE WORK.** This directive both catalogues the methodology gaps in the existing audits / backtests / research pipelines AND specifies the unified framework that replaces them. Live config changes from prior audits are FROZEN pending this directive's outcome.

---

## 0. Why this exists — and what "trust the audit" should mean

The IC Census + Track 1 audit cycle (May 2026) generated 7 directives, 8 commits, and ~25k LOC of pre-registered research, including verdicts that would retire two live strategies (IHYG → VUSD, IHYG → EMIM) and flag four others as `tier = unconfirmed` (Tier A EUR/USD + QQQ, 9 ORB instruments). All of those verdicts emerged from pipelines that were themselves audit-corrected per April 2026 standards.

But the audits themselves have accumulated material limitations:

- **Pre-committed thresholds** sometimes don't match the actual outcome distribution (MC P(MaxDD>25%)<5% failed all 6 bond-equity cells including audit-trusted references).
- **Simplified simulators** approximate the live class poorly (ORB DSR audit's per-cell verdicts cannot be trusted).
- **Decision matrices** don't cover all empirical outcomes (ML causality audit hit "UNDETERMINED").
- **Sharpe annualisation conventions** produce extreme magnitudes on sparse-trade strategies (ORB Sharpe -16 vs +5 across instruments under the same simulator).
- **Inconsistent WFO designs** across audits (5 contiguous folds for IC Census; 5 expanding-window for range-expansion; 8 for mr_audjpy; 48 for ML Tier A; variable for bond-equity).
- **Sanctuary discipline** isn't standardised — slicing by calendar months interacts with data start dates differently per instrument.
- **DSR formulas** assume normal returns (skew=0, kurt=3); real returns aren't normal.
- **Monte Carlo** block sizes and threshold gates are hardcoded without sensitivity analysis.
- **Same-bar look-ahead** confirmed in `run_52signal_classifier.py:613` but **not tested** in `multi_horizon_lstm.py`, `ensemble_stacking.py`, `lstm_classifier.py`, `autoencoder_regime.py`, `run_metalabeling.py`, `calibration_kelly.py`. Pattern could exist elsewhere.
- **Strategy coverage gaps** — LSTM stacking, GLD strategies, MR FX, ETF Trend, MTF, FX Carry, Pairs, IC Equity Daily, Turtle, Samir-Stack remain un-audited.
- **31 unit tests cover library primitives only** — no end-to-end ground-truth test of WFO orchestrators or audit scripts.

Acting on the Track 1 verdicts before fixing these is the opposite of disciplined. **All live config changes from the Track 1 audit cycle are FROZEN** until this directive's outcome.

This is V3.1 + V3.6 hygiene at the meta-level: pre-register the audit of the audit, fix everything, then re-derive verdicts.

---

## 1. Gap catalogue

Severity codes: 🔴 **blocking** (verdicts cannot stand), 🟠 **major** (verdicts likely to flip), 🟡 **minor** (verdicts hold but methodology needs cleanup), ⚪ **cosmetic** (style / documentation).

### A. Sharpe + annualisation

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| A1 | Hard-coded `sqrt(252)` outside shared metrics | 🟡 | Mostly fixed post-April-2026 audit; one remaining instance in `run_52signal_classifier.py:620` |
| A2 | Filter-then-annualise (`rets[rets != 0]`) | 🟡 | Fixed in `titan.research.metrics`; one remaining in `run_orb_dsr_audit.py` (this session — caught) |
| A3 | Wrong `periods_per_year` for bar frequency | 🟠 | Risk on intraday strategies; need a per-strategy-class default |
| A4 | **Per-bar Sharpe on sparse-trade strategies produces extreme magnitudes** | 🟠 | ORB DSR audit reported Sharpe -16 to +5 on 15-25 trades over 1872 bars. Annualised at sqrt(19656) amplifies any per-trade bias by 140×. **Per-trade Sharpe with `trades_per_year` is more honest for sparse strategies.** |
| A5 | Daily aggregation choice (MTM vs trade-level) inconsistent across audits | 🟠 | Bond-equity uses MTM daily; ORB uses per-bar M5; range-expansion uses per-bar H1. Cross-comparison breaks. |
| A6 | No standardised "strategy-class typology" tells which Sharpe convention applies | 🔴 | This is the root of A3-A5. Without classification, defaults are picked ad-hoc. |

### B. Look-ahead / causality

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| B1 | Same-bar `pos * ret` without `.shift(1)` | 🟠 | Confirmed in `run_52signal_classifier.py:613`. Impact bounded by position persistence (small in tier A). |
| B2 | Features use `close[t]` when prediction earns `t-1 → t` return | 🟠 | Same root cause as B1 -- alignment, not feature itself |
| B3 | Forward returns mistakenly entered as features | 🟢 | Audited; only target uses `close.shift(-h)`. ✓ |
| B4 | `.expanding()` over close (look-ahead variance) | 🟡 | Audit fix A6 from May-12 audit. One legacy use in some research scripts. |
| B5 | `.ffill` of higher-TF onto lower-TF without prior `.shift(1)` | 🟠 | The EUR/USD MTF +1.94 Sharpe bug pattern. **Tested in IC Census via assert_causal but not in production strategies.** |
| B6 | `groupby(date).transform("last")` for per-day aggregation | 🟡 | Subtle look-ahead. Not systematically grepped. |
| B7 | Same-fold IS leakage via global normalisation in WFO | 🟠 | `is_frozen_zscore` exists in `titan.research.metrics`; not used uniformly. |
| B8 | **Same-bar look-ahead in ML feature engineering NOT tested** in 6+ files: `multi_horizon_lstm.py`, `ensemble_stacking.py`, `lstm_classifier.py`, `autoencoder_regime.py`, `run_metalabeling.py`, `calibration_kelly.py` | 🔴 | Unknown impact. Tier A claim of +1.37 LSTM stacking on QQQ is suspect until tested. |
| B9 | No programmatic causality test on actual production feature pipelines (only synthetic anchored_aggregate test in IC Census) | 🔴 | Methodology gap -- the assert_causal pattern should be applied to every production feature factory. |

### C. Walk-forward / sanctuary

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| C1 | Inconsistent WFO designs across audits | 🟠 | IC Census 5 contiguous; range-expansion 5 expanding; mr_audjpy 8 expanding-IS; ML Tier A 48 rolling; bond-equity variable. Cross-comparison is currently unsound. |
| C2 | Sanctuary slicing by calendar months interacts with data start dates inconsistently | 🟡 | Sanctuary windows of different effective bar counts per instrument |
| C3 | **Sanctuary divergence (recent 12mo >> historical OOS) unexplained** across range-expansion, bond-equity, mr_audjpy | 🔴 | Three independent audits showed recent-12mo Sharpe 2-4× historical OOS. Possible: regime change, sample noise, OR a structural artefact of how WFO is fit. Not investigated. |
| C4 | WFO without IS-frozen normalisation in some places | 🟡 | bond-equity uses frozen IS levels; some ML pipelines may use global z-score. |
| C5 | Folds too small for bootstrap CI in some audits | 🟠 | ML Tier A: ~500 bars over 4-5 folds; bond-equity: ~400 bars over 8 folds; CI is wide enough to span zero for "promising" cells. |
| C6 | No standardised WFO config (IS_min, OOS, stride, fold count) per strategy class | 🔴 | Each audit improvises. |

### D. Cost model

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| D1 | Spread/slippage hardcoded per strategy (2 bps + 1 bps everywhere in audits this session) | 🟠 | ORB intraday spread varies 5-15 bps; cross-asset UCITS spread can be 5-30 bps. Cost model isn't realistic for all strategy classes. |
| D2 | No bid-ask realism check (intraday spread varies by time of day, vol regime) | 🟡 | Affects intraday strategies most |
| D3 | Commission not applied uniformly | 🟡 | IBKR has min commission $1.04 ES, $0.85 NQ -- not in all backtests |
| D4 | FX unit conversion silent assumption | 🟠 | `convert_notional_to_units` exists in titan/risk but not used in research backtests |
| D5 | Cost model not validated against actual fills | 🟡 | No live-backtest reconciliation step |

### E. DSR / multiple-testing

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| E1 | DSR formula assumes normal returns (skew=0, kurt=3) | 🟠 | Cross-asset / intraday returns aren't normal. DSR-prob may be biased. |
| E2 | `sr_var_across_trials` source ambiguous (survivors only?) | 🟠 | ORB audit acknowledged this -- used survivors-only, which UNDERSTATES true variance, producing OPTIMISTIC DSR-prob. |
| E3 | N count from survivors vs full screening pool ambiguous | 🟠 | Different N produces different e_max_SR. Choice not standardised. |
| E4 | DSR formula version: Bailey 2012 vs Bailey & López de Prado 2014 | 🟡 | Subtle differences in moment correction. Project uses BLP 2014. ✓ |

### F. Monte Carlo

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| F1 | MC block size hardcoded (50-bar) without sensitivity analysis | 🟠 | Different block sizes give materially different MaxDD distributions |
| F2 | **MC threshold gate P(MaxDD>25%)<5% wrong for cross-asset momentum** | 🔴 | Bond-equity audit failed ALL 6 cells including audit-trusted references. Gate is structurally too tight for always-on strategies. |
| F3 | MC bootstrap method: full reshuffle vs block vs stationary | 🟡 | Different methods preserve different statistical properties |
| F4 | Bond-target shared-block correlation preservation not validated | 🟡 | Bond-equity audit used shared-block; not verified that the synthetic correlation matches the empirical |
| F5 | MC for path-dependent strategies not validated | 🟡 | Range-expansion strategy bootstrap re-runs the strategy on synthetic prices; tied to bar-size autocorrelation. Reasonable but not stress-tested. |
| F6 | No standard "MC config per strategy class" | 🔴 | Same root cause as A6 / C6. Without classification, MC is ad-hoc. |

### G. Decision matrices

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| G1 | Pre-committed matrices don't cover all empirical outcomes | 🟠 | ML causality audit hit UNDETERMINED; my matrix had no row for "small inflation AND high sharpe AND negative CI_lo" |
| G2 | UNDETERMINED verdicts force manual review and risk drift | 🟡 | V3.1 spirit broken if undetermined → manual override is common |
| G3 | Verdict thresholds (e.g. CI_lo > 0) inherited from April 2026 audit but not re-validated | 🟡 | Maybe CI_lo > 0.2 is the right gate, not 0.0 |
| G4 | No standardised verdict template | 🔴 | Each directive invents its own. |

### H. Tests

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| H1 | Library primitives covered (31 tests); orchestrators uncovered | 🟠 | A bug in run_ic_census.py could produce wrong results silently |
| H2 | **No end-to-end synthetic ground-truth test** | 🔴 | Known-edge synthetic strategy should pass gates; known-no-edge should fail. Without this we can't verify the gate suite. |
| H3 | No causality test on production feature pipelines | 🔴 | Same as B9 |
| H4 | No test asserting cost-model realism | 🟡 | Affects D-class gaps |
| H5 | No test for the orchestrator-level integration (assert_causal in census, but not in audit scripts) | 🟠 | Audit scripts could have look-ahead in their own scaffolding without detection |

### I. Code hygiene + ops

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| I1 | Data file overwrites silent (IBKR overwrote yfinance VIX_D) | 🟠 | Caught only by manual file-size check |
| I2 | API defaults assumed (Databento `end=None` silently → 1-day) | 🟡 | Caught mid-session by file-size mismatch |
| I3 | Unicode characters in scripts cause cp1252 crashes on Windows | ⚪ | Hit twice this session |
| I4 | Background process death silent (Databento re-run died after MNQ) | 🟡 | Caught by file-size monitor |
| I5 | Audit scripts don't have a "reproducibility seed" enforced everywhere | 🟡 | Most do; not standardised |

### J. Strategy coverage

| ID | Strategy | Audited? | Severity |
|---|---|:---:|:---:|
| J1 | mr_audjpy | ✓ STATUS_QUO | OK |
| J2 | Bond-equity (IHYU→CSPX, IHYG→VUSD, IHYG→EMIM) | ✓ Partial — MC gate broken | 🟠 |
| J3 | ML Tier A (EUR/USD, QQQ) | ✓ Partial — same-bar small, CI_lo<0 | 🟠 |
| J4 | ORB 9 instruments | ✓ Partial — simulator doesn't match live | 🟠 |
| J5 | LSTM stacking (QQQ, SPY) | ✗ | 🔴 |
| J6 | gld_confluence | ✗ | 🟠 |
| J7 | gold_macro_gld | ✗ | 🟠 |
| J8 | bond_gold | ✗ | 🟠 |
| J9 | mr_fx (EUR/USD) | ✗ | 🟠 |
| J10 | etf_trend (SPY, QQQ, IWB, TQQQ, EFA, GLD, DBC) | ✗ | 🟠 |
| J11 | mtf | ✗ — known +1.94 Sharpe was invalidated | 🔴 |
| J12 | fx_carry (AUD/JPY) | ✗ | 🟠 |
| J13 | pairs (GLD/EFA) | ✗ | 🟠 |
| J14 | ic_equity_daily (7 US equities) | ✗ | 🟠 |
| J15 | turtle | ✗ | 🟡 |
| J16 | samir_stack | Partial — May-12 audit done | OK |

**Net coverage: 4 of 16 strategy families audited; 12 remain.**

### K. Pre-registration discipline

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| K1 | V3.1 enforcement is honor-system (no programmatic check) | 🟡 | Relies on operator discipline; works in practice but could regress |
| K2 | Decision rules sometimes too narrow | 🟠 | Same root cause as G1 |
| K3 | Result logs not always machine-readable | 🟡 | Hard to aggregate audit verdicts across directives |
| K4 | Sanctuary windows not always recorded in result parquets | 🟡 | Reproducibility risk |

### L. Framework absence

| ID | Gap | Severity | Affects |
|---|---|:---:|---|
| L1 | **No single source of truth for "what a backtest must include"** | 🔴 | Root cause of A6, C6, F6, G4. Each audit improvises. |
| L2 | **No strategy-class typology** (always-on momentum / sparse-trade intraday / cross-asset / ML / pairs / etc.) with class-specific defaults | 🔴 | Without typology, gate thresholds are picked ad-hoc. |
| L3 | **No "audit-grade" wrapper** that takes a strategy class + data + config and emits the standard audit output (Sharpe + CI + DSR + MC + sanctuary + verdict) | 🔴 | Every audit reimplements the same scaffolding. |
| L4 | **Living document mechanism** absent — V3.6 lessons accumulate in ad-hoc directives | 🟡 | The 7-lesson catalogue exists in commit messages, not in a centralised reference |

---

## 2. Unified framework specification

The framework lives in `titan/research/framework/`. Every NEW strategy + every RE-AUDITED strategy uses these primitives. The existing audit scripts are NOT mass-refactored — they stay frozen as historical records — but their verdicts are RE-COMPUTED under the framework, and any verdict that flips is documented and triggers a follow-up.

### 2.1 Strategy-class typology (fixes L2, A6, C6, F6)

```
class StrategyClass(Enum):
    INTRADAY_MICROSTRUCTURE   # H1 or higher freq, sparse trades, mean-reversion
    INTRADAY_BREAKOUT         # M5/M15, ORB-style, very sparse trades
    DAILY_TREND               # D, persistent long-only
    DAILY_MEAN_REVERSION      # D, oscillator-based
    CROSS_ASSET_MOMENTUM      # D, always-on long positions
    PAIRS                     # D, market-neutral
    ML_CLASSIFIER             # any TF, predicts label, holds until flip
    META_LABELING             # primary signal + ML filter
    CARRY                     # FX, slow-moving
```

Each class has:
- Default `periods_per_year`
- Default Sharpe convention (per-bar / per-trade / per-day-MTM)
- Default WFO design (IS_min_years, OOS_years, fold count, stride)
- Default MC config (block_size, n_paths, P(MaxDD>X) threshold)
- Default cost model (spread bps, slippage bps, commission, min_commission)
- Default sanctuary window (typically 12 months but class-specific overrides allowed)
- Default decision-matrix template

### 2.2 Sharpe convention per class (fixes A4, A5)

| Class | Sharpe input | periods_per_year |
|---|---|---|
| INTRADAY_MICROSTRUCTURE (H1) | Per-bar MTM return | `252 * 24` |
| INTRADAY_BREAKOUT (M5) | **Per-TRADE return** (not per-bar — bars are too sparse) | `n_trades / n_years` annualised at trade-level |
| DAILY_TREND | Per-day MTM return | `252` |
| DAILY_MEAN_REVERSION | Per-day MTM return | `252` |
| CROSS_ASSET_MOMENTUM | Per-day MTM return | `252` |
| PAIRS | Per-day MTM return | `252` |
| ML_CLASSIFIER | Per-bar MTM return at the model's TF | bar-frequency `BARS_PER_YEAR[tf]` |
| META_LABELING | Per-trade return | `n_trades / n_years` |
| CARRY | Per-day MTM return | `252` |

**Both Sharpes (per-bar AND per-trade) are reported in the audit output for every class.** The class-specific PRIMARY metric is the one used for gates; the secondary is for diagnostic purposes.

### 2.3 WFO design per class (fixes C1, C6)

| Class | IS min (years) | OOS (years) | Fold count | Stride | Sanctuary |
|---|---:|---:|---:|---:|---|
| INTRADAY_* | 1 (large H1/M5 sample) | 1 | 5 | non-overlapping | 12 months |
| DAILY_TREND | 3 | 1 | 5 | non-overlapping | 12 months |
| DAILY_MEAN_REVERSION | 3 | 1 | 5 | non-overlapping | 12 months |
| CROSS_ASSET_MOMENTUM | 2 | 0.5 | 8 | rolling 6-month | 12 months |
| PAIRS | 3 | 1 | 5 | non-overlapping | 12 months |
| ML_CLASSIFIER | 2 | 0.5 | 8+ | rolling 6-month | 12 months |
| META_LABELING | 2 | 0.5 | 8 | rolling | 12 months |
| CARRY | 5 | 1 | 5 | non-overlapping | 12 months |

Folds use **anchored expanding IS** for daily / cross-asset / pairs / carry, **rolling IS** for ML / intraday. Sanctuary is always **calendar-time-strict** (the last 12 calendar months by `df.index[-1]`).

### 2.4 MC config per class (fixes F1, F2, F6)

| Class | Block size (bars) | n_paths | Bootstrap method | P(MaxDD > X) threshold |
|---|---:|---:|---|---|
| INTRADAY_MICROSTRUCTURE | 50 | 200 | Block bootstrap (shared blocks if multi-leg) | P(MaxDD > 25%) < 5% |
| INTRADAY_BREAKOUT | 20 | 200 | Block bootstrap | P(MaxDD > 15%) < 10% (lower because sparse-trade) |
| DAILY_TREND | 21 (1mo) | 200 | Block bootstrap | P(MaxDD > 35%) < 10% (always-on) |
| DAILY_MEAN_REVERSION | 21 | 200 | Block bootstrap | P(MaxDD > 25%) < 10% |
| **CROSS_ASSET_MOMENTUM** | **63 (3mo)** | **200** | **Shared-block (preserves cross-asset correlation)** | **P(MaxDD > 35%) < 10%** (recalibrated from broken 25%/5% per Bond-Equity Audit §4.2-c) |
| PAIRS | 21 | 200 | Shared-block | P(MaxDD > 20%) < 5% (market-neutral) |
| ML_CLASSIFIER | strategy-bar-TF dependent | 200 | Block bootstrap | P(MaxDD > 25%) < 10% |

These thresholds are pre-committed HERE and become the framework's defaults. Strategy-specific overrides require their own pre-registration directive.

### 2.5 DSR application (fixes E1, E2, E3)

- **N**: total parameter cells tested in the sweep. NOT the survivor count. If the strategy went through a screener (e.g. ORB N=482), use the screener pool size.
- **`sr_var_across_trials`**: when possible, computed across ALL tested cells (not just survivors). For survivor-only data, the value is an **OPTIMISTIC lower bound**; this is documented in the audit output.
- **Skew + Kurt**: per-cell skew + kurt of the **per-bar return series** (not the per-trade return series — that's a different distribution). Computed via `scipy.stats.skew` and `kurtosis` with `fisher=False` (Pearson kurtosis, kurt=3 for normal).
- **Output**: `dsr_prob` per cell; the audit reports both `dsr_prob` and the explicit `e_max_SR` so the reader can see how close the cell sits to the null max.

### 2.6 Decision-matrix template (fixes G1, G4)

Every audit's decision matrix uses this 4-axis template, expanded per strategy class:

```
Axis 1: Point-estimate Sharpe (CI_lo > 0 / between 0 and -0.2 / < -0.2)
Axis 2: DSR-prob (>= 0.95 / between 0.5 and 0.95 / < 0.5)
Axis 3: MC P(MaxDD > X) (< threshold / between threshold and 2× threshold / >= 2× threshold)
Axis 4: Sanctuary Sharpe (> 0 / between 0 and -0.3 / < -0.3)
```

**Verdict mapping:**

| All 4 best | 3-of-4 best (worst is 1 axis only) | 2-of-4 best | 1-of-4 best | None best |
|---|---|---|---|---|
| DEPLOY | CONDITIONAL_WATCHPOINT | TIER_UNCONFIRMED | SUSPECT | RETIRE |

The template eliminates UNDETERMINED outcomes by construction — every empirical result maps to one cell of the 81-cell (3⁴) outcome space, and every cell maps to a verdict.

### 2.7 Cost model standard (fixes D1-D5)

Per asset class:

```
CME_FUTURES_LIQUID = {spread_bps: 1.0, slip_bps: 1.0, commission_usd_per_side: 1.0}
US_EQUITY_LARGE_CAP = {spread_bps: 0.5, slip_bps: 0.5, commission_usd_per_side: 0.5}
US_ETF_LIQUID = {spread_bps: 1.0, slip_bps: 0.5, commission_usd_per_side: 0.35}
UCITS_ETF = {spread_bps: 8.0, slip_bps: 2.0, commission_usd_per_side: 2.0}
FX_MAJOR = {spread_bps: 0.5, slip_bps: 0.3, commission_usd_per_side: 0.0}
IG_DFB_INDEX = {spread_bps: 2.5, slip_bps: 1.0, commission_usd_per_side: 0.0}
```

These are conservative defaults; per-instrument overrides allowed via `cost_model_override` in the strategy's pre-reg.

### 2.8 Sanctuary discipline (fixes C2, C3)

- Sanctuary = **last 12 calendar months by `df.index[-1]`**. Strictly time-based; not bar-based.
- Sliced BEFORE any IS/OOS fold construction. Folds operate on `df[df.index < sanctuary_start]`.
- One-shot sanctuary pass runs the strategy on `df[df.index >= sanctuary_start]` ONLY, using:
  - For strategies with frozen calibration: the LAST fold's frozen parameters
  - For strategies with continuous calibration: a calibration window of the prior 12 months (so 24-month-back to 12-month-back is the calibration; 12-month-back to today is the sanctuary)
- Sanctuary divergence test (NEW — fixes C3): compute the **distribution of 12-month-window Sharpes** over the WFO portion. If the sanctuary's 1-year Sharpe is in the top 5% of historical 12-month windows, flag `sanctuary_lucky = True` — possibly regime-specific, not deployment-validating.

### 2.9 Living document (fixes L4)

`directives/V3.6 Lessons Catalogue.md` (NEW — created in this PR) is the single authoritative list of V3.6 lessons. Every audit's result log appends its lessons to this catalogue. Lessons can be REPHRASED (refined) but not REMOVED.

### 2.10 Audit-grade wrapper (fixes L3)

`titan/research/framework/audit.py` provides:

```python
result = run_audit(
    strategy: AuditableStrategy,
    data: DataConfig,
    strategy_class: StrategyClass,
    pre_reg_directive_path: Path,
    overrides: AuditOverrides | None = None,
)
```

`AuditableStrategy` is a protocol:
```python
class AuditableStrategy(Protocol):
    def simulate(self, ohlcv: pd.DataFrame, config: dict) -> SimulationResult: ...
    # Returns per-bar P&L, per-trade list, transitions, etc.
```

`run_audit` does:
1. Loads data + applies sanctuary slice
2. Builds WFO folds per strategy class
3. Runs strategy on each IS+OOS fold
4. Computes per-bar AND per-trade Sharpe (both reported)
5. Computes bootstrap CI on both
6. Computes DSR over the cell sweep
7. Runs MC (block bootstrap with class-default config)
8. Runs sanctuary pass (one-shot)
9. Computes sanctuary divergence test (§2.8)
10. Applies decision-matrix template
11. Returns `AuditResult` dataclass + writes a parquet + a markdown result log to the directive's path

Every existing audit re-runs through this wrapper as part of the Migration (§3).

---

## 3. Migration plan

### 3.1 Phase 1 — build the framework + tests (THIS COMMIT)

| Deliverable | Status |
|---|---|
| `titan/research/framework/__init__.py` | TBD |
| `titan/research/framework/typology.py` (StrategyClass + defaults) | TBD |
| `titan/research/framework/wfo.py` (standardised fold construction) | TBD |
| `titan/research/framework/sanctuary.py` (slicing + divergence test) | TBD |
| `titan/research/framework/mc.py` (block bootstrap with class defaults) | TBD |
| `titan/research/framework/dsr.py` (skew+kurt-aware) | TBD |
| `titan/research/framework/decision.py` (4-axis template) | TBD |
| `titan/research/framework/audit.py` (run_audit wrapper) | TBD |
| `tests/test_framework_synthetic.py` (known-edge / known-no-edge) | TBD |
| `scripts/audit_codebase_methodology.py` (programmatic gap detector) | TBD |
| `directives/V3.6 Lessons Catalogue.md` | TBD |

### 3.2 Phase 2 — re-run prior audits under the framework (separate PRs, post-Phase 1)

| Strategy | Re-run priority | Reason |
|---|:---:|---|
| Bond-equity (IHYG → VUSD, EMIM) | 1 | RETIRE recommendation currently rests on broken MC gate. Re-run with §2.4 corrected gate; verdict may flip. |
| Range-expansion Phase 0 | 2 | Sanctuary divergence unexplained; framework's divergence test (§2.8) may identify cause. |
| ML Tier A | 3 | UNDETERMINED verdict; framework's 4-axis template eliminates that outcome. |
| ORB DSR | 4 | Simulator caveat means per-cell verdicts can't stand; framework requires the live class, not a simulator. |
| mr_audjpy anchor compare | 5 | Verdict was robust (STATUS_QUO) but framework re-run validates the gate uniformity. |

### 3.3 Phase 3 — audit the 12 un-audited strategies

| Strategy | Strategy class | Priority |
|---|---|:---:|
| LSTM stacking (QQQ, SPY) | ML_CLASSIFIER | 1 — Tier A scale, largest unknown |
| mtf (EUR/USD) | INTRADAY_MICROSTRUCTURE | 2 — known +1.94 Sharpe was invalidated; current status unclear |
| ic_equity_daily | DAILY_MEAN_REVERSION | 3 — live deployed |
| etf_trend (7 ETFs) | DAILY_TREND | 4 — live deployed, simple |
| mr_fx (EUR/USD) | INTRADAY_MICROSTRUCTURE | 5 |
| gld_confluence | INTRADAY_MICROSTRUCTURE | 6 |
| gold_macro_gld | DAILY_TREND | 7 |
| bond_gold | CROSS_ASSET_MOMENTUM | 8 |
| fx_carry (AUD/JPY) | CARRY | 9 |
| pairs (GLD/EFA) | PAIRS | 10 |
| turtle | DAILY_TREND | 11 |
| samir_stack | already-audited May 2026 | 12 — verify with framework |

### 3.4 Phase 4 — ML same-bar causality sweep

Apply the `assert_causal`-style corruption test to every production feature factory in `research/ml/`:
- `lstm_features.py`, `quantile_features.py`, `multi_horizon_lstm.py`, `ensemble_stacking.py`, `lstm_classifier.py`, `autoencoder_regime.py`, `run_metalabeling.py`, `calibration_kelly.py`

For each, build a small test: corrupt future bars of the input close, recompute features, assert past feature values are bit-exact unchanged.

### 3.5 Phase 5 — frozen config changes from Track 1

Until Phase 1-4 complete, FREEZE the following pending config-change PRs:

- IHYG → VUSD retirement (gate broken; verdict may flip)
- IHYG → EMIM retirement (same)
- `tier = unconfirmed` flag on EUR/USD D + QQQ D (Tier A)
- `tier = unconfirmed` flag on 9 ORB instruments
- `run_52signal_classifier.py:613` one-line `.shift(1)` fix (this one is unambiguous and can ship before Phase 2; queued)

---

## 4. What this directive does NOT do

- **Does not** retire any strategy. Track 1 retirement recommendations are FROZEN.
- **Does not** change any live config.
- **Does not** modify the existing audit scripts. They stay as historical records.
- **Does not** delete any prior verdict — they're catalogued in §3.2 with the reason for re-running.
- **Does not** introduce new strategies. Phase 3 catalogues the 12 un-audited families; that's audit work, not new research.

---

## 5. Execution status

- Phase 1 framework + tests + audit script + catalogue (THIS COMMIT)
- Phase 2-5 sequenced as separate PRs

---

## 6. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial methodology audit + framework spec. Phase 1 execution. |
