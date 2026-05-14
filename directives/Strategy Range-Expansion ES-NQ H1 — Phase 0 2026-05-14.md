# Strategy Phase 0 — Range-Expansion on ES + NQ H1

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect
**Status:** **PRE-REGISTRATION** for the strategy backtest pipeline.
Committed BEFORE any backtest, parameter sweep, or live-deployment work
runs (V3.1).
**Parent:** `directives/IC Signal Census Phase C 2026-05-13.md` §6.7 item 2.

---

## 0. Why this exists

The combined Phase A + B + C IC Census produced exactly two TIER_B
survivors: `intraday_range_atr period=14` at **NQ H1** (IC=+0.0248,
t_NW=+7.18, dsr_p=0.9998) and **ES H1** (IC=+0.0231, t_NW=+6.76,
dsr_p=0.9990). Both clear every audit-discipline gate except MTF
agreement -- the H1 signal does not have a D-frequency analogue (the
MTF Lift probe at `directives/IC MTF Lift Range-Expansion 2026-05-14.md`
proved D-frequency `intraday_range_atr` has essentially zero IC).

This directive **does not run a backtest.** It pre-registers the
strategy spec, gates, and protocol BEFORE the backtest runs. Once on
`main`, the cells, parameters, and selection rules can only be
RELAXED in a subsequent PR explaining why the original was
unimplementable (V3.1). Tightening or cell-favouring after results
land is forbidden.

The companion confluence directive (`IC Confluence Range-Reversion
2026-05-14.md`) tested whether the signal concentrates in oversold
regimes — verdict: NO, gating destroys the signal. The strategy here
uses the **unconditional** range_atr.

---

## 1. Strategy specification (pre-registered)

### 1.1 Signal & entry

```
signal[t] = (range[t] / ATR(close, period=14)) - 1
            where range[t] = high[t] - low[t]
            ATR is Wilder's average true range (causal, period=14)

entry_long  IF signal[t] > θ_entry
entry_short IF signal[t] < -θ_entry  (range_atr is rarely deeply negative;
                                       short side likely sparse)

The signal is computed on bar [t] CLOSE. The position is opened at
[t+1] open with a market order. Closes at [t+1+H] close where H is
the holding horizon. Same shift discipline as Phase C IC computation
(audit A1/A2).
```

### 1.2 Pre-committed parameter grid

Three cells × two instruments × two horizons. Plateau gate applies in
the joint space; the headline cell is `(θ_entry=0.5, H=1)` if `θ_entry`
is swept on each side independently.

| Cell | θ_entry | H (bars) | Notes |
|---|---:|---:|---|
| 0 | 0.25 | 1 | low threshold, more trades |
| 1 | **0.50** | **1** | **headline candidate** |
| 2 | 1.00 | 1 | high threshold, fewer but stronger trades |
| 3 | 0.50 | 4 | hold longer (4 H1 bars ≈ half-day) |
| 4 | 0.50 | 8 | hold a full trading day |

Pre-committed: only these 5 cells. Plateau check on the (θ_entry sweep
holding H=1) axis: cell 1 is interior, neighbours are cells 0 and 2.

### 1.3 Cost model

Inherits the audit-corrected pattern from `research/cross_asset/run_bond_equity_wfo.py`:
explicit `periods_per_year` via `titan.research.metrics.BARS_PER_YEAR["H1"]`,
explicit cost-per-fill, no filter-then-annualise (audit fix).

| Cost component | NQ futures | ES futures | IG US Tech 100 DFB | IG US 500 DFB |
|---|---|---|---|---|
| Spread (per side) | 0.25 NQ pts × $5/pt = $1.25 / contract | 0.25 ES pts × $12.50/pt = $3.125 / contract | 1.0 IG pt | 0.4 IG pt |
| Slippage (per fill) | 0.25 NQ pts | 0.25 ES pts | 1.0 IG pt | 0.4 IG pt |
| Commission (per side, IBKR) | $0.85 | $1.04 | $0 (DFB) | $0 (DFB) |
| Per round-trip cost | ~$4 NQ / ~$8 ES (IBKR) | | ~2 IG pt | ~0.8 IG pt |

The IG DFB column matters because that's the short-term execution
venue. Both routes are evaluated in the backtest; the deployment
decision picks the better Sharpe-after-costs.

### 1.4 Position sizing

Per-strategy equity tracker (titan/risk/strategy_equity.py), 1% equity
risk per trade, ATR-based stop equivalent to 2× ATR(period=14) at the
entry bar. Position sizing follows the rest of the live stack pattern
documented in `references/portfolio-risk-architecture.md`.

For the backtest, position size = `floor((1% × equity) / (2 × ATR_$))`
contracts, rounded down to integer. ES one-tick is $12.50; NQ one-tick
is $5.

### 1.5 Walk-forward design

15 years of ES history (2011-2026), 6 years of NQ (likely identical
window). Walk-forward:

- IS / OOS folds: 5 non-overlapping, anchored at start, expanding window.
  IS_min = 3 years of H1 bars. OOS = next 1 year.
- Sanctuary: last 12 months untouched until final-validation pass.
- Fold metrics: per-fold Sharpe + bootstrap CI.
- Stitched OOS series: aggregated across folds → headline Sharpe,
  bootstrap CI, max DD, profit factor.

### 1.6 Gates the backtest must clear before live deployment

Inherits from the parent's pre-flight checklist plus the May 2026 audit
rules:

| Gate | Source | Requirement |
|---|---|---|
| Sharpe math | titan.research.metrics | Explicit `periods_per_year=BARS_PER_YEAR["H1"]`. No filter-then-annualise. |
| Same-bar look-ahead | Audit A1/A2 | Entry at `t+1` open, signal computed at `t` close. Asserted by a parity test. |
| Bootstrap CI | April-2026 audit | CI_lo > 0 on the stitched OOS Sharpe series. |
| DSR | Audit A5 | Apply DSR at N=5 cells. With N=5, null-max ≈ 1.6 -- minor adjustment but required. |
| Underlying-resampled MC | Audit A6 | Bootstrap NQ + ES H1 returns with shared block indices (50-bar blocks), cumprod to rebuild synthetic price paths, run strategy on each. P(MaxDD > 25%) < 5%. **Mandatory before live.** |
| Plateau-stable | V3.2 | Headline cell + (θ_entry ± 1 grid step) all clear the Sharpe gate. |
| Live parity test | A10 | `tests/test_range_expansion_live_parity.py` -- one H1 bar compared between research class and live class. Bit-exact equality. Causality test: corrupt future bars, assert past output unchanged. |
| Strategy Guide | A9 | Updated Strategy Guide section with deployed config. No drift between Guide and `config/*.toml`. |
| Sanctuary | V3.6 | 12-month trailing window held out from IS/OOS. Final-validation pass on sanctuary computed separately. |

### 1.7 Deployment-venue routing decision

Two routes are backtested in parallel. The deployment decision picks
whichever route produces the higher post-cost CI_lo, with tax-efficiency
as tie-breaker:

| Route | Pros | Cons |
|---|---|---|
| **IBKR CME futures** (NQ + ES) | Lower per-trade cost, can short cleanly, multi-leg supported, tax-favoured (60/40 §1256) | Bigger min position size ($1.25-3.125 / pt × tick), futures rollover handling |
| **IG DFB** (US Tech 100 + US 500) | Smaller min size (fractional spread bets), tax-free in UK, short-term-only execution natural fit | Wider spread, no overnight position carry without rollover cost, basis vs cash index, capital-gains-tax inefficient outside UK |

The strategy is fundamentally short-term (h=1 to h=8 H1 bars = 1 hour
to 8 hours holding), so IG DFB's overnight-cost disadvantage is minor.
The Sharpe-after-costs comparison decides.

### 1.8 No-go conditions

The strategy is NOT deployed live if any of:

- Stitched OOS Sharpe CI_lo ≤ 0
- DSR-adjusted prob < 0.95
- Underlying-resampled MC P(MaxDD > 25%) ≥ 5%
- Sanctuary-pass Sharpe < 0 (i.e. the strategy fails in the held-out year)
- Plateau-stability fails (the headline cell's IS Sharpe is more than 2× the IS Sharpe at θ_entry±0.25 or H±2)
- Live parity test fails

---

## 2. Out of scope

- Variants of the entry rule (signal threshold, position-sizing curve,
  stop-loss configuration). Each variant is a new pre-registration.
- Combining range_expansion with other Phase A/B/C survivors (none
  exist; Phase A surfaced no TIER candidates). The IC confluence test
  has already rejected RSI gating.
- Auto-rebalancing across NQ ↔ ES. Each instrument runs as a separate
  strategy instance with its own equity tracker; the PRM allocator
  handles correlation-aware sizing across the portfolio.
- Cross-portfolio integration (vs other live strategies in
  `directives/Strategy Deployment Guide.md`). That's a Phase 6 task,
  not Phase 0.

---

## 3. Implementation milestones

| Milestone | What it produces | Pre-registration needed beyond this directive? |
|---|---|---|
| 1. Backtest runner | `research/strategies/run_range_expansion_wfo.py` -- per-cell IS/OOS Sharpe with CI, DSR, MC | None — this directive specifies it |
| 2. Underlying-resampled MC | parquet of (cell × MC path) MaxDD + Sharpe | None |
| 3. Sanctuary-pass | One-shot final-validation Sharpe on the 12-month sanctuary | None |
| 4. Live class | `titan/strategies/range_expansion/` Nautilus class | Implementation, not new spec |
| 5. Parity test | `tests/test_range_expansion_live_parity.py` | None |
| 6. Strategy Guide update | new section in `directives/Strategy Deployment Guide.md` | None |
| 7. Paper trial 30 days | Sharpe + drawdown vs backtest | Requires a separate deployment-decision PR |

This directive is milestone 0 — the spec lands FIRST so the backtest
runs against a frozen target rather than a moving one.

---

## 4. Result log

Milestone 1-3 (backtest runner + MC + sanctuary) ran 2026-05-14. §1-§3
unchanged (V3.1).

### 4.1 WFO stitched OOS Sharpe + bootstrap CI (audit-corrected)

| Instrument | Cell (θ_entry, H) | OOS Sharpe | CI_lo | CI_hi | MaxDD | n_folds | n_trades | DSR prob |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | 0.25, 1 | -4.18 | -4.99 | -3.33 | -0.90 | 5 | 16,742 | 0.0 |
| **NQ** | **0.50, 1 (headline)** | **-3.63** | **-4.43** | **-2.70** | **-0.83** | **5** | **11,330** | **0.0** |
| NQ | 1.00, 1 | -1.13 | -1.96 | -0.19 | -0.36 | 5 | 2,928 | 0.0 |
| NQ | 0.50, 4 | -1.71 | -2.51 | -0.91 | -0.78 | 5 | 17,971 | 0.0 |
| NQ | 0.50, 8 | -1.00 | -1.82 | -0.13 | -0.67 | 5 | 21,691 | 0.0 |
| ES | 0.25, 1 | -1.91 | -2.87 | -1.01 | -0.85 | 5 | 16,470 | 0.0 |
| **ES** | **0.50, 1 (headline)** | **-1.86** | **-2.71** | **-1.01** | **-0.78** | **5** | **10,888** | **0.0** |
| ES | 1.00, 1 | -0.69 | -1.48 | +0.29 | -0.37 | 5 | 2,588 | 0.0 |
| ES | 0.50, 4 | -1.09 | -1.95 | -0.11 | -0.79 | 5 | 17,155 | 0.0 |
| ES | 0.50, 8 | -0.76 | -1.65 | +0.06 | -0.76 | 5 | 20,663 | 0.0 |

### 4.2 Underlying-resampled Monte Carlo (audit A6)

50 bootstrap paths with 50-bar shared block indices. The synthetic
H/L per bar carries the observed (high-low)/close ratio resampled
together with the close-return — preserves bar-size autocorrelation
which is what the strategy actually trades.

| Instrument | Cell | Median Sharpe | Median MaxDD | P5 Sharpe | P95 Sharpe | **P(MaxDD > 25%)** |
|---|---|---:|---:|---:|---:|---:|
| NQ | 0.25, 1 | -2.61 | -0.91 | -6.18 | -1.09 | **1.00** |
| NQ | 0.50, 1 | -2.68 | -0.87 | -5.74 | -1.15 | **1.00** |
| NQ | 1.00, 1 | -1.19 | -0.53 | -2.65 | -0.25 | **0.98** |
| NQ | 0.50, 4 | -1.39 | -0.82 | -2.34 | -0.36 | **1.00** |
| NQ | 0.50, 8 | -0.85 | -0.78 | -1.62 | -0.28 | **1.00** |
| ES | 0.25, 1 | -1.78 | -0.98 | -3.65 | -0.65 | **1.00** |
| ES | 0.50, 1 | -1.70 | -0.95 | -3.33 | -0.67 | **1.00** |
| ES | 1.00, 1 | -0.62 | -0.51 | -1.09 | -0.14 | **1.00** |
| ES | 0.50, 4 | -0.86 | -0.94 | -1.67 | -0.15 | **1.00** |
| ES | 0.50, 8 | -0.75 | -0.94 | -1.20 | -0.21 | **1.00** |

The MC's P(MaxDD > 25%) = 100% on every cell confirms the cost-dominated
loss is a structural feature, not a sample artefact.

### 4.3 Sanctuary pass (last 12 months, NOT used in WFO)

| Instrument | Cell | Sharpe | MaxDD | n_trades | win rate | Net P&L (USD) |
|---|---|---:|---:|---:|---:|---:|
| NQ | 0.25, 1 | -0.00 | -0.16 | 1,655 | 0.485 | -8 |
| **NQ** | **0.50, 1** | **+1.24** | **-0.14** | **1,087** | **0.494** | **+$13,603** |
| **NQ** | **1.00, 1** | **+1.66** | **-0.07** | **286** | **0.549** | **+$14,744** |
| NQ | 0.50, 4 | +0.32 | -0.21 | 701 | 0.477 | +$5,153 |
| NQ | 0.50, 8 | -0.03 | -0.17 | 489 | 0.483 | -$514 |
| ES | 0.25, 1 | -0.80 | -0.29 | 1,659 | 0.450 | -$18,601 |
| ES | 0.50, 1 | -0.04 | -0.20 | 1,079 | 0.470 | -$807 |
| **ES** | **1.00, 1** | **+0.80** | **-0.11** | **250** | **0.456** | **+$11,517** |
| ES | 0.50, 4 | +0.05 | -0.28 | 700 | 0.466 | +$1,607 |
| ES | 0.50, 8 | -0.22 | -0.33 | 487 | 0.481 | -$7,690 |

**Sanctuary diverges sharply from pre-sanctuary** — the higher-threshold cells (θ=0.5, 1.0 with H=1) show positive Sharpe in the last 12 months despite being deeply negative in the 14-year WFO. Per V3.1, **a positive sanctuary cannot override a failing WFO** — and this divergence is itself a warning signal that whatever produced the sanctuary's lift is not stable across regimes.

### 4.4 Verdict per §1.8 no-go conditions

| §1.8 condition | Status | Pass? |
|---|---|:---:|
| Stitched OOS Sharpe CI_lo ≤ 0 | All cells CI_lo ∈ [-4.99, -0.11], all < 0 | **FAIL** |
| DSR-adjusted prob < 0.95 | All cells dsr_prob = 0.0 | **FAIL** |
| Underlying-resampled MC P(MaxDD > 25%) ≥ 5% | All cells 98-100% | **FAIL** |
| Sanctuary-pass Sharpe < 0 | Cell-dependent: NQ θ=0.5 H=1 is +1.24; ES θ=0.5 H=1 is -0.04 | Mixed |
| Plateau-stability: IS Sharpe at headline > 2× neighbours | All Sharpes negative; ratio undefined / not the bottleneck | n/a |
| Live parity test | Not run (failed gates above stop us before this) | n/a |

**Three independent no-go conditions fail.** The strategy is **not deployment-eligible** under the pre-registered gates.

### 4.5 Mechanism analysis

The IC at H1 horizon=1 was real (Phase C: NQ +0.025, ES +0.023, t_NW > 6.5). But the per-trade economic content scales with the IC magnitude × per-bar volatility × position size, while the cost is fixed per contract:

```
Expected per-trade alpha ≈ IC * std(forward_return) * notional * size
  ≈ 0.025 * 0.002 * $25k * 1 contract  ≈ $1.25 on ES
Round-trip cost ≈ 2 * (spread+slip + commission) * size
  ≈ 2 * ($6.25 + $1.04) * 1  ≈ $14.58 on ES
```

Cost exceeds alpha by ~10×. The IC discovery was statistically significant but **economically below the friction floor**. This is exactly the case the audit-grade cost-aware backtest is designed to catch — DSR alone (Phase C's gate) confirmed the signal exists; the cost-aware engine confirms it doesn't translate to deployable alpha at realistic CME futures friction.

### 4.6 Action

1. **No live deployment.** Strategy is closed at Phase 0.
2. **No milestone 4-7.** Live class + parity test are not built (they presume gate-passing backtest).
3. **No retroactive cost-tuning.** V3.1 forbids re-running with a more aggressive cost model to make the backtest profitable.
4. **IG DFB execution route is not auto-evaluated.** The directive committed to evaluating both IBKR CME and IG DFB. The IG DFB cost structure differs (point value scaled to GBP stake, spread varies by index). A **separate** pre-registration is required to test IG DFB at H1 — under V3.1, this isn't a continuation of the current strategy spec, it's a new strategy because the cost model is structurally different. Filed for later; not started now.
5. **The sanctuary's positive result is research output, not a deployment signal.** Could indicate a recent regime change in volatility clustering on NQ/ES — worth a separate research directive (signal stability over time, not strategy deployment) if pursued.

### 4.7 V3.6 negative-result hygiene

This is the cleanest "IC is real but not tradeable at realistic friction" outcome in the project to date. Two lasting lessons:

1. **DSR-passing IC doesn't equal deployable strategy.** Phase C's audit-discipline gates filtered signals statistically; the strategy pre-reg's cost-aware backtest filtered them economically. Both gates are needed.
2. **The cost/alpha ratio is the binding constraint on H1 microstructure mechanisms.** Range-expansion is real, but its per-bar economic content (~0.0025% in raw return) is on the order of the CME futures half-spread. Most short-term microstructure signals will face the same ratio. Future H1 strategy proposals must compute `alpha_per_trade / cost_per_trade` BEFORE running a full backtest -- it's a one-line check.

### 4.8 Outcome record

| Field | Value |
|---|---|
| Backtest run? | Yes — milestone 1-3 |
| Live class built? | **No** — gates failed |
| Parity test written? | **No** — strategy closed at Phase 0 |
| Phase 0 verdict | **REJECTED** under audit-discipline gates |
| TIER_B IC discovery retained as research output? | Yes (Phase C census parquet unchanged) |
| Re-pre-registration permitted with different cost model? | Only as a structurally new strategy (e.g. IG DFB execution route) with its own pre-reg |

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial strategy Phase 0 pre-registration. |
