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

To be appended in stages as milestones complete. Backtest result first
(after milestone 1-3); paper-trial result later (after milestone 7).

> _Pending backtest run — appended below after each milestone lands._

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial strategy Phase 0 pre-registration. |
