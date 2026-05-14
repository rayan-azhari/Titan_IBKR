# Titan-IBKR-Algo (V2.0)

> A quantitative trading system for Interactive Brokers — NautilusTrader execution, portfolio-level risk management, and a unified backtesting framework with typology / WFO / sanctuary / DSR / MC / decision-matrix discipline.

---

## V2.0 status (2026-05-14)

The repo went through a methodology cleanse on 2026-05-14. The audit found systemic gaps that the V1 published Sharpe numbers depended on — inconsistent annualisation, ad-hoc WFO designs, uniformly-mis-calibrated Monte Carlo gates, decision matrices that could return `UNDETERMINED`, and a confirmed same-bar look-ahead bug in one ML file (six others untested). Acting on the resulting verdicts before fixing the methodology would have been the opposite of disciplined.

**V2.0 is the disciplined response:** discard the suspect results, keep the ideas, rebuild under a single framework.

- **Result-log directives + research orchestrators discarded.** All April/May 2026 Sharpe claims are no longer evidence.
- **Live strategies are still running on V1 verdicts.** Every strategy in `titan/strategies/` is currently `tier = unconfirmed` until re-audited under the framework.
- **The unified framework lives at `titan/research/framework/`** — single source of truth for typology (9 strategy classes) + WFO + sanctuary + DSR + MC + 4-axis decision matrix. Every new audit + every re-audit goes through it.

**Start here:** [`directives/README V2.0.md`](directives/README%20V2.0.md) — the master reference for what was kept, what was discarded, and what's open.

---

## Live Strategies (running, all `tier = unconfirmed`)

The strategies in `titan/strategies/` continue to run in their existing config. **None has been re-audited under the V2.0 framework yet.** V1-era Sharpe numbers are suspect; treat each as unconfirmed until a framework-grade audit lands.

| Live class | Strategy class (`StrategyClass`) | Runner |
|---|---|---|
| `mr_audjpy` | `INTRADAY_MICROSTRUCTURE` | `scripts/run_live_mr_audjpy.py` |
| `mr_fx` | `INTRADAY_MICROSTRUCTURE` | `scripts/run_live_mr_fx.py` |
| `mtf` | `INTRADAY_MICROSTRUCTURE` | `scripts/run_live_mtf.py` |
| `orb` | `INTRADAY_BREAKOUT` | `scripts/run_live_orb.py` |
| `etf_trend` | `DAILY_TREND` | `scripts/run_live_etf_trend.py` |
| `bond_gold` | `CROSS_ASSET_MOMENTUM` | `scripts/run_live_bond_gold.py` |
| `gld_confluence` | `INTRADAY_MICROSTRUCTURE` | `scripts/run_live_gld_confluence.py` |
| `gold_macro` | `DAILY_TREND` | `scripts/run_live_gold_macro.py` |
| `fx_carry` | `CARRY` | `scripts/run_live_fx_carry.py` |
| `pairs` | `PAIRS` | `scripts/run_live_pairs.py` |
| `ic_mtf` | `INTRADAY_MICROSTRUCTURE` | `scripts/run_live_ic_mtf.py` |
| `samir_stack` | `CROSS_ASSET_MOMENTUM` + overlay | (re-audit pending) |
| `turtle` | `DAILY_TREND` | (re-audit pending) |

All strategies are wired to the shared `PortfolioRiskManager` (per-strategy equity tracker + FX-aware sizing + drawdown circuit breaker + halt persistence). The April 21, 2026 risk-layer rewrite is still authoritative — see `references/portfolio-risk-architecture.md` in the orchestrator skill.

**ML strategies are deferred.** V1 ML artifacts depended on suspect feature-pipeline causality. The `models/` directory is intentionally absent in V2.0; ML re-introduces after the same-bar causality audit on 6 untested feature files (`multi_horizon_lstm.py`, `ensemble_stacking.py`, `lstm_classifier.py`, `autoencoder_regime.py`, `run_metalabeling.py`, `calibration_kelly.py`).

---

## Architecture

```
directives/                32 SOPs — strategy mechanisms, methodology, ops
  README V2.0.md           cleanse rationale + how to use the framework — READ FIRST
  Methodology Audit & Unified Framework 2026-05-14.md
  V3.6 Lessons Catalogue.md
titan/                     Core library: strategies, adapters, indicators, risk
  research/framework/      The unified backtesting framework (typology + WFO + MC + DSR + decision)
  research/metrics.py      Audit-corrected shared metrics (sharpe, bootstrap_sharpe_ci, BARS_PER_YEAR, ...)
  risk/                    PortfolioRiskManager + Allocator + StrategyEquityTracker
research/                  Library/concept modules (signal defs, regime detectors, state managers)
                           NO orchestrator runners — build fresh via the framework
scripts/                   CLI entry points: kill switch, watchdogs, validators, data download, run_live_*
config/                    TOML parameters (V1-era values may need re-tuning under framework)
data/                      Historical Parquet files (gitignored; regen via scripts/download_data_*.py)
tests/                     ~30 files, 296 tests covering framework + IC primitives + live classes
.tmp/                      Logs + transient reports (gitignored)
```

Code flows one direction: `research/` discovers → `config/` captures → `titan/` implements → `scripts/` executes. Never import from `scripts/` inside `titan/`. Never put entry points inside `titan/`.

---

## Quick Start

### 1. Install dependencies
```bash
uv sync --extra dev
```

### 2. Configure credentials
```bash
cp .env.example .env   # then edit with your IBKR account details
```

Key `.env` variables:
```ini
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # 4002 = paper gateway, 4001 = live gateway
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_ID=DUxxxxxxx
```

### 3. Verify connection
```bash
uv run python scripts/verify_connection.py
```

### 4. Download data
```bash
uv run python scripts/download_data.py
```

### 5. Run a strategy (paper)

Use the **portfolio watchdog** for production (handles IB nightly restarts + reconciles positions):
```bash
uv run python scripts/watchdog_portfolio.py
```

Or run individual strategies directly:
```bash
uv run python scripts/run_live_mtf.py            # MTF EUR/USD
uv run python scripts/run_live_orb.py            # ORB equities
uv run python scripts/run_live_etf_trend.py      # ETF Trend
uv run python scripts/run_live_mr_audjpy.py      # AUD/JPY mean-reversion
```

See `directives/Deployment & Operations.md` for full deployment (VPS, Docker, systemd, IB Gateway headless).

---

## Emergency Stop

```bash
uv run python scripts/kill_switch.py
```

Cancels all orders and closes all positions immediately via IBKR API. Does not require the strategy process to be running. Halt state persists in `.tmp/portfolio_halt.json` — operator must explicitly `reset_halt(operator=...)` to resume.

---

## Research Workflow — the V2.0 framework

Every audit goes through `titan.research.framework`. Don't write ad-hoc WFO loops.

```python
from titan.research.framework import (
    StrategyClass, defaults_for,
    slice_sanctuary, sanctuary_divergence_test,
    build_folds,
    deflated_sharpe, sr_var_from_sweep,
    run_block_mc,
    DecisionInputs, decide,
)
from titan.research.metrics import sharpe, bootstrap_sharpe_ci

# 1. Classify
cls = StrategyClass.DAILY_TREND
d = defaults_for(cls)   # Sharpe convention, WFO config, MC gate, decision thresholds

# 2. Pre-register a directive listing universe + grid + selection rule (V3.1)
# 3. Slice sanctuary (hold-out last 12 months)
# 4. Build WFO folds on the visible window
# 5. Run strategy on each fold → stitched OOS returns
# 6. Per-cell: sharpe + bootstrap_sharpe_ci + deflated_sharpe + run_block_mc
# 7. decide(DecisionInputs(...)) → DEPLOY / CONDITIONAL_WATCHPOINT / TIER_UNCONFIRMED / SUSPECT / RETIRE
# 8. Append result log to §4 of the pre-reg directive
# 9. Append any new lesson to V3.6 Lessons Catalogue.md
```

Canonical recipe: see [`directives/Methodology Audit & Unified Framework 2026-05-14.md`](directives/Methodology%20Audit%20%26%20Unified%20Framework%202026-05-14.md). The framework's 53 synthetic tests are in [`tests/test_framework_synthetic.py`](tests/test_framework_synthetic.py).

### Codebase methodology scanner

```bash
uv run python scripts/audit_codebase_methodology.py --write-parquet
```

Scans `research/` + `titan/` + `scripts/` for 8 known anti-patterns (hardcoded `sqrt(252)`, same-bar `position * return`, `.ffill` after `.reindex`, global z-scores, …). Sieve, not proof — operator reviews. Run before declaring a re-audit done.

---

## Pre-Push Checklist

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest tests/ -v
```

All three must pass. CI on `main` enforces all three plus AST-level guardrails ([`tests/test_research_math_guardrails.py`](tests/test_research_math_guardrails.py)) that fail if a PR re-introduces bare `sqrt(252)`, filter-then-annualise Sharpe, or `list(balances.keys())[0]`.

---

## Key Rules

- **`uv` only** — no bare `pip` installs
- **`decimal.Decimal`** for all financial types (or NautilusTrader `Price` / `Quantity`)
- **`random_state=42`** on every ML training call
- **No look-ahead bias** — features `.shift(1)`'d, targets future-derived, every audit verifies via `assert_causal`
- **Factory methods** for NautilusTrader objects (`Price.from_str()`, not constructors)
- **Per-strategy equity** — strategies must use `StrategyEquityTracker`, never `account.balance_total(list(balances.keys())[0])`
- **FX unit conversion** — `convert_notional_to_units(notional_base, price, quote_ccy, base_ccy, fx_rate)` for non-USD-quoted instruments. Never silent `notional / price`.
- **Annualisation factor explicit** — every `sharpe(returns, periods_per_year=...)` call passes the factor matching the bar timeframe. `252` is not a safe default.

Full rules: [`directives/Workspace Structure.md`](directives/Workspace%20Structure.md) + the V3.6 catalogue.

---

## Documentation

All operational knowledge lives in `directives/` (32 SOPs). Key files:

| Topic | Directive |
|---|---|
| **System overview** | [`README V2.0.md`](directives/README%20V2.0.md) |
| **Framework spec** | [`Methodology Audit & Unified Framework 2026-05-14.md`](directives/Methodology%20Audit%20%26%20Unified%20Framework%202026-05-14.md) |
| **Lessons (16, L01-L16)** | [`V3.6 Lessons Catalogue.md`](directives/V3.6%20Lessons%20Catalogue.md) |
| Deployment | `Deployment & Operations.md` |
| Docker | `Docker Paper Trading Guide.md` |
| ORB strategy | `ORB Trading Strategy.md` + `ORB Strategy User Guide.md` |
| MTF strategy | `Multi-Timeframe Confluence.md` + `MTF Strategy User Guide.md` |
| ETF Trend | `ETF Trend Strategy.md` |
| Samir-Stack | `Samir-Stack Strategy Guide.md` + `Samir V3 — VIX-HMM Strategy Design 2026-05-13.md` |
| ML pipeline (deferred) | `Machine Learning Strategy Discovery.md` |
| IC signals | `IC Signal Analysis.md` |
| Backtesting concept | `Backtesting & Validation.md` (framework primer is the code path) |
| Cost models | `Cost Model Audit 2026-05-11.md` |
| Operational robustness | `Operational Robustness Framework 2026-05-12.md` |
| Broker migration | `Broker Migration Assessment.md` |
| API reference | `Titan Library Reference.md` + `IBKR & NautilusTrader API Reference.md` |
| Adapter guide | `Titan-IBKR Adapter Guide.md` |
| Workspace layout | `Workspace Structure.md` |
