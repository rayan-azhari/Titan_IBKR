# Pre-Registration — turtle V1-era Re-audit (V3.6 + L52 hybrid)

**Author:** rayanazhari (planner) + Claude orchestrator (Architect)
**Date committed:** 2026-05-16
**Branch:** v2-main
**Type:** V1-era live strategy re-audit. Wave B full audit (second P2 strategy to clear Wave B triage gate after gold_macro). See `directives/V1-era Re-audit Sweep Roster 2026-05-16.md` + `.tmp/reports/wave_b_triage/findings.md`.
**Strategy class:** `DAILY_TREND` (H1 Donchian breakout with multi-day position holds; trend economics dominate the bar-cadence question).
**Predecessors:**

- V1-era `titan/strategies/turtle/` deployment (config: `config/turtle_h1.toml`, CAT-only). Live mechanism includes pyramiding (max_units=4, +0.5 ATR per unit), ATR(20) hard stop at 2.0 ATR, trailing stop, risk_pct=1%. The signal layer (the SWEPT axes) is the Donchian (entry / exit) pair.
- Wave B triage 2026-05-16: per-bar Sharpe +1.60 (CI_lo +0.35) on 8y CAT-H1 at the live (45, 30) config. **CRITICAL CAVEAT**: triage used `BARS_PER_YEAR["H1"] = 6048` (24/7 FX assumption), which OVERSTATES US equity RTH H1 Sharpe by `sqrt(6048/1764) = 1.85x`. With correct US-equity RTH annualisation (1,764 bars/year), the live config IS Sharpe is **+0.54**, not +1.60. **L52/L53 hidden-bug class: annualisation mismatch in sparse-trade cadence audits.**
- L52 sweep (`research/exploration/sweep_turtle.py`, 2026-05-16): IS plateau on 6.9y CAT-H1 IS data found at **(entry_period=20, exit_period=30)** with IS Sharpe +0.422, hood_mean +0.409, spread 14.1% (tightest plateau yet seen in any Wave audit). Live (45, 30) IS Sharpe +0.542 sits in the entry=45 hot row but doesn't pass strict plateau gate. Surface IS-maximum at **(entry=45, exit=20) IS Sharpe +0.730** — peak, not plateau.
- L21 causality smoke on pure-research implementation: **PASS** (max past-return diff = 0.00 when 50 future bars corrupted across OHLC).

**Status:** §1–§3 frozen at this commit. §4 result log appended after the audit harness runs.

> **V3.1 pre-registration.** §1–§3 are frozen BEFORE the audit examines the 12-month sanctuary window. **Hypothesis being tested: the V3.6-correct turtle canonical `(entry_period=20, exit_period=30)`, chosen from a Pardo-style sweep on the 6.9y CAT-H1 IS, achieves CI_lo > 0 on the held-out sanctuary divergence test AND survives the 5-axis decision matrix AND demonstrates non-trivial generalisation across other liquid US equities.**
>
> **L52 canonical-selection note.** The canonical `(entry=20, exit=30)` is the strict-L52 plateau center (hood_mean +0.409, spread 14.1%). The hot cell `(entry=45, exit=20)` has the surface-maximum IS Sharpe +0.730 but does NOT pass the strict 30% spread gate (hood spread ~44.5%, passes only the looser 50% H1 gate). Pre-committing to the plateau choice over the peak choice is the L43 discipline. **However, C3_peak is included in the eligible-for-promotion pool** so the OOS data adjudicates whether plateau discipline OR peak economics wins — the selection rule "highest CI_lo" handles this naturally.
>
> **Why the live config (45, 30) is mid-table.** Live sits in the entry=45 "hot row" but with the slower exit (30) instead of the optimal exit (20). The IS-sweep shows entry=45 with exit=20 maximises Sharpe — V1's choice of exit=30 was suboptimal even within its own row.

---

## §1. Motivation & mechanism

**Why re-audit turtle now.** Three converging signals:

1. The strategy is **currently in `titan/strategies/turtle/`** with a config (CAT H1, entry=45, exit=30, pyramiding ON, trailing stop ON) whose Sharpe claim comes from a V1 audit whose harness has documented failure modes. Per V2.0 README, all such strategies need a V3.6 re-audit.
2. **Wave B triage (2026-05-16) found turtle POSSIBLY VIABLE** (per-bar Sharpe +1.60 with CI_lo +0.35 at live config), strongest signal-layer result of any V1-era audit so far. **BUT** the +1.60 was annualised with the FX 24/7 convention (6048 bars/year); proper US equity RTH H1 cadence (1,764 bars/year) deflates this by `sqrt(6048/1764) = 1.85x` to ~+0.87. The IS sweep with correct annualisation confirms: live (45, 30) IS Sharpe = +0.54.
3. **L52 sweep found a tight plateau at (20, 30) IS Sharpe +0.42, spread 14.1%** — the tightest plateau in any Wave A/B audit. The plateau is at a DIFFERENT cell than live, and the surface maximum is at yet a third cell (45, 20). Live (45, 30) is sub-optimal even by V1 hindsight.

**Mechanism (per bar t, long-only Donchian breakout on H1 close).**

1. **Entry signal**: `state == 0` flips to `1` when `close[t] > Donchian_high(entry_period)[t-1]`.
2. **Exit signal**: `state == 1` flips to `0` when `close[t] < Donchian_low(exit_period)[t-1]`.
3. **Position**: binary (single-unit, no pyramiding in the sweep/audit signal layer; pyramiding is a sizing overlay tested as a future Phase 2 axis).
4. **Cost**: 1 bp per turnover (US equity H1, conservative; live mechanic uses MOC fills which can be slightly worse intra-day but ~1 bp is the right order for liquid large-cap names).
5. **Per-bar net return**: `net(t) = position(t-1) * log(close[t]/close[t-1]) - cost`.

**Note on live vs research mechanics.** The live strategy adds pyramiding (up to 4 units at +0.5 ATR each), an ATR(20) hard stop at 2.0 ATR below entry, and a trailing stop that ratchets. These are sizing/risk overlays on the signal. The audit tests the SIGNAL economics; the overlays are evaluated separately if the signal passes the 5-axis matrix. (If the signal layer fails, no overlay can rescue it — L46 + L43 + L58 precedent.)

**Causality (L04 / A1 / L21).** Donchian channels use `.shift(1)` so the breakout test at t uses past-only OHLC. Position decided at close[t] earns return from t→t+1 via `position.shift(1) * log_ret`. L21 smoke test PASSED at sweep stage.

## §2. Universe + audit configurations

**Primary instrument.** CAT (Caterpillar Inc.), live config target. Daily-large-cap US equity, liquid, no special microstructure.

**Multi-instrument robustness panel.** 10 additional liquid US large-cap equities with H1 RTH data of similar depth, run at the canonical params after the primary audit completes. Tickers (selected for liquidity + sector diversity + H1 history ≥ CAT's 15,794 bars): **AAPL, MSFT, NVDA, AMZN, GOOGL, JPM (if data sufficient), XOM, CVX, JNJ, PG**. Reported as per-name Sharpe distribution + percentile of CAT within the cross-section. (Sample-of-1 fragility is the primary turtle risk; this is the test.)

**Date range.** CAT H1: **2018-05-01 → 2026-03-18 (15,794 bars, ~7.9 years)**.

**Sanctuary.** **Last 12 months** held out: 2025-03-18 → 2026-03-18 (2,000 bars). Applied via `slice_sanctuary()` BEFORE WFO fold construction.

**Visible window for WFO.** 13,795 bars (2018-05-01 → 2025-03-18), ~6.9 years.

**Annualisation.** `periods_per_year = 1,764` (US equity RTH H1: 7 bars/day × 252 days). **NOT** `BARS_PER_YEAR["H1"] = 6048` which is the FX 24/7 convention. This is the critical calibration that distinguishes the triage's +1.60 from the audit's correct level.

**WFO.** `DAILY_TREND` class defaults `is_min=3y, oos=1y, fold_count=5, expanding, no overlap`. With 6.9y visible, expect 4-5 expanding folds. No per-fold tuning.

**MC.** `DAILY_TREND` class defaults `block_size_bars=21, n_paths=200, max_dd=0.35, pass_prob=0.10`. For H1 cadence, block_size=21 ~= 3 trading days. Tested at 21 bars (default) and 147 bars (≈3 weeks) as a sensitivity check in §4.

**Pre-registered cells (V3.1, frozen).** Sweep informed canonical + 3 plateau cells + 3 references.

| Cell | `entry_period` | `exit_period` | Notes |
|---|---:|---:|---|
| **C1_canonical** | **20** | **30** | L52 plateau center (hood_mean +0.409, spread 14.1%). The new V3.6 canonical. |
| P_short_exit | 20 | 20 | Plateau neighbour, IS +0.441. |
| P_long_exit | 20 | 45 | Plateau neighbour (also plateau #2 in sweep), IS +0.390. |
| P_north | 20 | 10 | Plateau neighbour (also plateau #3), IS +0.334. |
| **C2_live_canonical** | 45 | 30 | V1-era live config, IS +0.542. Parity baseline + V1-vs-V3.6 comparator. |
| **C3_peak** | 45 | 20 | Surface IS-maximum +0.730 (entry=45 hot row, short exit). Off-plateau peak; tests whether peak economics survives WFO. **Included in promotion-eligible pool.** |
| C4_gross_no_costs | 20 | 30 | Canonical with `apply_costs=False`. Gross-economics reference. |

7 cells total. C2 and C4 are baselines / references EXCLUDED from §3 selection. C3_peak IS eligible.

**Falsification hypotheses (pre-committed, V3.1).**

- **H1 (plateau holds OOS, spread ≤ 50%).** Relative spread across (C1, P_short_exit, P_long_exit, P_north) on stitched OOS remains ≤ 50%. **Falsifiable.**
- **H2 (canonical sanctuary Sharpe ≥ +0.30).** **Falsifiable.**
- **H3 (new canonical beats live).** C1 sanctuary Sharpe > C2 (live) sanctuary Sharpe. **Falsifiable.**
- **H4 (CI_lo > 0 on stitched OOS).** C1's bootstrap 95% lower bound across the full WFO stitched OOS is > 0. **Falsifiable** — L46 binding-constraint gate.
- **H5 (peak vs plateau).** C3_peak (45, 20) sanctuary Sharpe > C1_canonical (20, 30) sanctuary Sharpe AND C3_peak CI_lo > 0. **Falsifiable** — if SUPPORTED, the peak was genuine and L43 strict plateau discipline cost alpha; if REJECTED, plateau discipline was right and the peak was IS-overfit. (Either outcome generates a valuable L52/L43 lesson.)
- **H6 (multi-instrument robustness).** Canonical (C1) median Sharpe across the 10-instrument robustness panel is ≥ 0 AND CAT's percentile within the cross-section is ≤ 75%. **Falsifiable** — if CAT is the 90th-percentile outlier, the +0.54 IS Sharpe is single-instrument-overfit; if median panel Sharpe ≤ 0, the signal doesn't generalise.
- **H7 (cost drag is not the bottleneck).** C4 gross Sharpe − C1 net Sharpe ≤ 0.25. **Falsifiable.**

## §3. Decision rule (pre-committed, V3.1)

**Per-axis thresholds:** `DAILY_TREND` defaults.

**Cell selection rule:** among DEPLOY-eligible cells (DEPLOY, or CONDITIONAL_WATCHPOINT with noise=best), EXCLUDING C2 (live baseline) and C4 (gross reference), pick the cell with the highest CI_lo. **C3_peak is eligible** — if its OOS economics support promotion, it wins despite being off-plateau on IS.

**Pre-flight gate (L27 + L52).**

1. **L52 plateau pre-flight (run FIRST).** Compute Pass-1 stitched OOS Sharpe for C1 + 3 plateau cells (P_short_exit, P_long_exit, P_north). Relative spread ≤ 50% (H1 gate). If > 50%, abort Pass 2 → RETIRED.
2. **L53 early gate.** If NO cell can plausibly clear CI_lo > 0, skip Pass 2 → RETIRED.
3. **Pass 2 + 5-axis decision matrix.** Only run on cells that pass L52 + L53 gates.

**Tighter-constraint rule (L46).** If a cell passes the 5-axis matrix but has CI_lo ≤ 0, it does NOT promote.

**L17 relative-MC application.** Long-only on US equity → buy-and-hold benchmark. Pass criterion: `median_dd_reduction ≤ 0.80` AND `p_strategy_better ≥ 0.50`. CAT is single-name (idiosyncratic), so L17 may be more or less binding than for ETF tests; documented as a primary 5-axis input regardless.

**Multi-instrument robustness.** Tested AFTER promotion gate. If C1 passes 5-axis on CAT, run the 10-instrument panel and report H6 verdict. If H6 is REJECTED, downgrade promotion verdict from DEPLOY → CONDITIONAL_WATCHPOINT (CAT-specific) and recommend "deploy ONLY on CAT, audit each additional instrument independently before allocating capital".

## §4. Result log

To be appended AFTER the audit harness runs. Sections:

- §4.1 — Baseline reproduction (C2 vs triage; document annualisation correction)
- §4.2 — Plateau pre-flight (L27 + L52 on stitched OOS)
- §4.3 — L53 early-gate report
- §4.4 — Per-cell 5-axis decision matrix
- §4.5 — L17 relative MC (CAT buy-and-hold benchmark)
- §4.6 — Multi-instrument robustness panel (10 tickers at canonical params)
- §4.7 — H1 / H2 / H3 / H4 / H5 / H6 / H7 verdicts
- §4.8 — Promotion verdict + binding constraint
- §4.9 — Live-config impact + backlog
- §4.10 — New lessons (annualisation-mismatch L60?)

## §5. Failure modes to watch

- **L04 / A1 (causality).** Verified at sweep stage.
- **L13 / L14 (IS/OOS separation).** Donchian uses rolling windows that grow with time but use only past data; no per-fold re-fitting needed.
- **L17 (relative MC for long-only equity).** Primary 5-axis input.
- **L18 (shift discipline).** Position must shift by 1 bar; parity contract with sweep.
- **L21 (causality smoke).** PASSED at sweep stage.
- **L25/L06 (sparse-trade Sharpe).** Per-bar Sharpe is reported as primary (matches sweep); per-trade Sharpe is secondary. **For a sparse-trade strategy on H1, the per-bar Sharpe with correct annualisation is closer to the per-trade-annualised Sharpe than naive intuition suggests** because non-position bars contribute zero variance under the binary signal — this is why the triage's `pos.shift(1) * log_ret` formulation is informative even at the per-bar level.
- **L27 (plateau).** Tested as H1 on OOS.
- **L43 (knife-edge plateau).** Tested as H5 — does peak OR plateau win OOS?
- **L46 (sample-size CI bottleneck).** May bind — only 4-5 expanding folds with 6.9y visible.
- **L52 (hybrid workflow).** Third applied case after bond_gold and gold_macro.
- **L58 (triage magnitude-vs-direction caveat).** Already engaged — triage Sharpe magnitude (+1.60) was inflated 1.85x by FX annualisation. Direction (positive) preserved at correct +0.54.
- **L60 (NEW, candidate)?** "Cadence annualisation mismatch": when a strategy spans markets with different trading conventions (24/7 FX vs RTH equity), the `periods_per_year` constant in shared metrics modules must match the asset class, not the bar timeframe. The L58 triage applied `BARS_PER_YEAR["H1"] = 6048` (FX) to US equity H1 and overstated Sharpe by 1.85x. **If this is the first time the framework catches it, propose L60 in §4.10.**

## §6. What "complete" looks like

- §4 fully appended with all 7 cells' 5-axis matrix + L17 rel-MC + H6 robustness panel.
- One of four outcomes:
  - **DEPLOY** (C1 passes all gates + H6 supports generalisation): live config can migrate to `(entry=20, exit=30)`. Migration requires updated `config/turtle_h1.toml` + parity test + 6-month shadow.
  - **CONDITIONAL_WATCHPOINT** (C1 passes 5-axis but CI_lo ≤ 0, OR H6 fails): shadow-deploy canonical for 6-12 months OR scope to CAT-only with explicit per-instrument-audit gate.
  - **C3_peak promotion** (if H5 SUPPORTED + CI_lo > 0 on (45, 20)): the off-plateau peak wins. Live (45, 30) migrates to (45, 20). Documents L43 trade-off in lesson.
  - **RETIRED** (plateau-fail OR L46 bottleneck OR H6 fail): turtle's V1-era config remains in code but flag as "V1-claim unconfirmed". De-allocation depending on allocator priorities (note: turtle is NOT in champion_portfolio runner, so no live impact regardless).

In all four cases: update [V1-era Re-audit Sweep Roster 2026-05-16.md](V1-era%20Re-audit%20Sweep%20Roster%202026-05-16.md) to mark turtle's Wave B full-audit status as COMPLETE.
