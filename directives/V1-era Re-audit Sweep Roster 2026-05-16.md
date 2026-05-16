# V1-era strategy re-audit — sweep-first roster

**Status:** 2026-05-16. Working document. Updated as each strategy
completes its discovery sweep.

**Context (L52 + README V2.0):** Every strategy in `titan/strategies/`
was promoted under V1 verdicts whose Sharpe numbers came from now-deleted
directives. They remain LIVE in code but their deployment claims are
*unconfirmed* under V3.6. Per the V3.6 hybrid workflow (L52), each
re-audit BEGINS with a Pardo-style parameter sweep on IS-only data to
identify a flat high-Sharpe plateau, then commits a pre-reg with the
plateau centre as canonical, then runs the full V3.6 audit on the held-out
sanctuary as the deployment gate.

## Sweep-first workflow

For each strategy below:

1. **Identify the parameter axes that materially affect Sharpe.** Usually
   2 axes (lookback × threshold, or fast × slow). Limit to 2 for surface
   visualisation; can extend later.
2. **Pick the data source.** Use the same data the live strategy uses
   (data/<…>) so the sweep result transfers directly to deployment.
3. **Define the sanctuary window** — minimum 12 months held out per
   V3.6. For long-history strategies (DAILY_TREND, CROSS_ASSET_MOMENTUM
   on ETFs), can hold out 24 months.
4. **Build a sweep script** in `research/exploration/` named
   `sweep_<strategy>.py` (template: `sweep_b4b_tsmom.py`).
5. **Run + plateau-detect.** If a plateau exists, the plateau centre is
   the candidate canonical for the new pre-reg directive.
6. **Pre-reg + audit + decision matrix.** Standard V3.6 path.

## Priority tiering

- **P1** — strategy is live OR has high notional allocation. Re-audit
  before next allocator rebalance.
- **P2** — strategy is in registry but not currently funded.
- **P3** — research-only / disabled.

## Roster

| # | Strategy | Class | Priority | Tunable axes (suggested sweep) | Data | Notes |
|---|---|---|---|---|---|---|
| 1 | `mr_audjpy` | INTRADAY_MICROSTRUCTURE (H1 MR) | **P1** | `(zscore_lookback, threshold)` | data/AUDJPY_H1 | Live deployed. H1 frequency, monthly+ history. |
| 2 | `mr_fx` | INTRADAY_MICROSTRUCTURE (M5 VWAP) | **P1** | `(vwap_window, deviation_threshold)` | data/EURUSD_M5 | Live deployed. Beware L18 shift discipline. |
| 3 | `mtf` | INTRADAY_MICROSTRUCTURE | **P1** | `(fast_ma, slow_ma)` + threshold | data/EURUSD_M5 + H1 | **L21 look-ahead risk** — sweep MUST run with verified causality (`assert_causal`). |
| 4 | `orb` | INTRADAY_BREAKOUT | **P2** | `(range_minutes, breakout_atr)` | data/UNH,AMAT,TXN,INTC,CAT,WMT,TMO_M5 + D | Sparse trades — use `trade_sharpe` not bar Sharpe (L25). |
| 5 | `etf_trend` (7 variants) | DAILY_TREND | **P1** | `(slow_ma, fast_reentry_ma)` | data/SPY,QQQ,IWB,GLD,DBC,EFA,TQQQ_D | 7 separate configs. Sweep each independently. |
| 6 | `bond_gold` | CROSS_ASSET_MOMENTUM | **P1** | `(lookback, threshold)` | data/IEF_D, GLD_D | Originally Sharpe +1.17 V1; canonical (60, 0.50). |
| 7 | `gld_confluence` | INTRADAY_MICROSTRUCTURE | P2 | `(confluence_window, threshold)` | data/GLD_M5 | Gold-specific MR. |
| 8 | `gold_macro` | DAILY_TREND | P2 | `(slow_ma, regime_threshold)` | data/GLD_D | Macro-trend variant. |
| 9 | `fx_carry` | CARRY | P2 | `(carry_lookback, vol_lookback)` | data/AUDJPY_D | Single instrument. |
| 10 | `pairs` | PAIRS | P2 | `(zscore_window, entry_threshold, exit_threshold)` | data/GLD_D, EFA_D | 3-axis — pick 2 for surface, sweep 3rd separately. |
| 11 | `ic_equity_daily` | DAILY_MEAN_REVERSION | P2 | `(rsi_lookback, oversold_threshold)` | data/{7 US tickers}_D | RSI-based. |
| 12 | `ic_mtf` | INTRADAY_MICROSTRUCTURE | P2 | `(fast_window, slow_window)` | data/various | IC-derived MTF. |
| 13 | `ml` | ML_CLASSIFIER | P3 | N/A — sweep doesn't apply directly | data/{52-signal grid} | **L19 same-bar look-ahead bug** caught not fixed; FIX FIRST, then re-audit (no parameter sweep — feature pipeline is the gate). |
| 14 | `samir_stack` | CROSS_ASSET_MOMENTUM + overlay | **P1** | `(stack_lookback, overlay_threshold)` | data/{samir basket} | May-2026 audited; framework re-audit pending. |
| 15 | `turtle` | DAILY_TREND | P2 | `(donchian_window, atr_stop_mult)` | data/{turtle universe} | Donchian + ATR. |
| 16 | `gap_fade` | INTRADAY_MICROSTRUCTURE | P3 | `(gap_threshold, fade_window)` | data/EURUSD_H1 | Marked CONDITIONAL in original verdict. |

**Total: 16 strategies (excluding gem which is the only confirmed live).**

## Execution plan

**Wave A (P1, this week-or-next):**

1. `bond_gold` — **COMPLETE 2026-05-16. Verdict: PROMOTED — CONDITIONAL_WATCHPOINT** for `(lookback=120, threshold=0.50)` (recommended) or `(120, 0.25)` (§3-rule strict pick). Live config `(60, 0.50)` remains in production; 6-month shadow comparison + fresh sanctuary re-audit before migration. See `.tmp/reports/bond_gold_reaudit/findings.md`. Sanctuary regime-favourable (L17/L55 caveat).
2. `etf_trend SPY` — **COMPLETE 2026-05-16. Verdict: RETIRED.** L52 hybrid found a real OOS plateau (5% spread) and positive Sharpe (+0.72 vs B&H +0.68), but FAILED L17 relative MC (median DD reduction 1.01) AND Varma noise robustness (worst axis). See `.tmp/reports/etf_trend_spy_reaudit/findings.md` + L56.
2b. `etf_trend TQQQ` — **COMPLETE 2026-05-16 (Wave A.2-confirm). Verdict: PROMOTED CONDITIONAL_WATCHPOINT** for `(slow_ma=150, exit_confirm_days=5)`. L56 rel-MC failure GENERALISES BUT bulk-retire REFUTED: leveraged variant has +54% Sharpe edge over B&H (vs SPY's +6%) and exit_confirm_days=5 rescues noise axis. Migrate live `(175, 1)` at next rebalance after 6mo shadow. See `.tmp/reports/etf_trend_tqqq_reaudit/findings.md`. **Refined recommendation: bulk-retire 5 unleveraged variants (QQQ, IWB, EFA, DBC, GLD); audit any other leveraged variants (SOXL/SPXL if registered) individually.**
3. `mr_audjpy` — **COMPLETE 2026-05-16 (Wave A.3). Verdict: SIGNAL-LAYER FAIL.** The live config `(vwap_anchor=24, entry_pct=0.95)` proxy has IS Sharpe **-0.30** on 14y H1 data — the V1 (corrected) claim of +0.53 is NOT reproducible at the signal layer. The +0.53 V1 number comes from the filter machinery (tier-grid + regime filter + session window + NY close) layered on a negative base signal — classic L43/L46 overfit-risk. Recommend de-allocation at next allocator window unless operator wants to invest 1-2 days in a multi-dimensional L52 sweep including the filters as axes. See `.tmp/reports/sweep_mr_audjpy/findings.md` + new **L58 reservation** (signal-layer-first audit pattern).
4. `samir_stack` — already partially audited; sweep validates the pre-reg
   canonical against IS plateau.
5. `mtf` — **causality check FIRST** (L21); only then sweep.
6. `mr_fx` — M5 data; expect ~5-10x slower per sweep cell.

**Wave B (P2):**

7. `orb`, `gld_confluence`, `gold_macro`, `fx_carry`, `pairs`,
   `ic_equity_daily`, `ic_mtf`, `turtle`.

**Wave C (P3 / blocked):**

9. `ml` — fix the L19 same-bar look-ahead bug before any sweep is
   meaningful. The signal-class is "feature pipeline integrity," not
   "parameter choice."
10. `gap_fade` — low priority; only re-audit if a Wave A/B strategy
    is RETIRED freeing allocation.

## Out-of-band: GEM J5 hybrid re-audit (2026-05-16)

The only V3.6-CONFIRMED-live strategy (GEM A1_ewma_hl40) was also run through the L52 hybrid framework as a confirmatory exercise:

- **Sweep** (`research/exploration/sweep_gem_hybrid.py`) — 2D over `(halflife × vol_target)`. Live `(40, 0.10)` is OFF the plateau; plateau at `vol_target=0.05` column, IS Sharpe ~0.82 vs live 0.75.
- **Pre-reg** (`directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md`) — 8 cells.
- **Audit** (`research/gem/run_gem_j5_reaudit.py`) — **PROMOTED `P_hl60_vt05` to DEPLOY**. OOS Sharpe +1.00 (vs J4 live +0.74), CI_lo +0.51 (vs +0.24), L17 rel-MC PASSES (38% DD reduction vs 60/40 SPY/IEF vs J4 live's 12%).
- **New lesson L57** — vol-target overlay max_leverage cap asymmetry; hybrid 2D sweep required.
- **Migration plan** (Phase 0-3) in `.tmp/reports/gem_j5_reaudit/findings.md` — pending allocator approval (50% capital reduction at vol_target=0.05 vs 0.10).

This is the first successful application of L52 to a confirmed-live strategy. Pattern: 1D-sweep canonicals can miss 2D plateau structure; hybrid framework can RESCUE a confirmed-live strategy by identifying superior canonicals.

## What this roster is NOT

- **It is not a pre-reg directive.** Each sweep produces a CANDIDATE
  canonical; the pre-reg + audit are still required.
- **It does not commit to a specific Sharpe gate.** The deployment gate
  remains the V3.6 5-axis decision matrix on held-out sanctuary data,
  not the sweep result.
- **It is not a binding allocation roadmap.** Champion-portfolio
  weights are unaffected until each re-audit completes.

## Template

For each strategy:

- Copy `research/exploration/sweep_b4b_tsmom.py` to
  `research/exploration/sweep_<strategy>.py`.
- Replace the data loader, the strategy adapter (`tsmom_strategy_fn` →
  `<strategy>_fn`), the parameter grid, and the metadata.
- Run `PYTHONIOENCODING=utf-8 uv run python research/exploration/sweep_<strategy>.py`.
- Read `plateau_report.md`. If candidates exist, the top-ranked center
  is the new pre-reg canonical.
- Write a brief `findings.md` (template: `.tmp/reports/sweep_b4b_tsmom/findings.md`).
