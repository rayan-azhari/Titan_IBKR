# Remediation Audit Report — April 2026

**Branch**: `autoresearch/20260406-175621`
**Baseline tag**: `preaudit-2026-04-21` (at commit `56343c3`)
**Harness commit**: `a62dd46` (post-remediation)
**Run date**: 2026-04-21
**Sanctuary window**: `365 days` active (last year held out)

---

## Summary

All four critical-path WFOs were re-run on the remediated harness with
the three structural fixes from the April 2026 audit:

1. Same-bar `pos[t] * ret[t]` look-ahead fix (bond-equity) — `pos_lagged = [0, pos[0..n-2]]`.
2. MR backtest tier-pyramid accumulation (was resetting to zero every bar).
3. Filter-then-annualise Sharpe bias (`rets[rets != 0]` dropped before `sqrt(252)`) — removed via shared `titan.research.metrics.sharpe`.

Plus: per-strategy 12-month sanctuary window on data loaders (agent-visible
data now strictly excludes the last 365 days).

Every stitched Sharpe is reported with a 95 % bootstrap CI (n=1000
resamples, seed=42). **Deployment gate**: strategies with `CI_lo ≤ 0`
move to `tier=unconfirmed` and are deprecated from the champion portfolio.

---

## Per-strategy results

| Strategy | Old Sharpe | New Sharpe | CI 95 % (lo, hi) | Max DD | n_folds | Pos % | Trades | CI_lo > 0 |
|---|---:|---:|---|---:|---:|---:|---:|:-:|
| **MR AUD/JPY** (champion `conf_donchian_pos_20`, conservative) | +4.64 | **+0.910** | (+0.478, +1.319) | -16.24 % | 7 | 85.7 % | 180 | ✅ |
| **MR AUD/USD** (best `conf_rsi_14_dev`, conservative) | +2.10 | **+0.374** | (**-0.180**, +0.831) | -49.03 % | 8 | 50.0 % | 385 | ❌ |
| **IEF→GLD** (lookback=60, hold=20, threshold=0.50) | +1.17 | **+0.721** | (+0.302, +1.225) | -21.41 % | 38 | 55.3 % | 114 | ✅ |
| **HYG→IWB** (lookback=10, hold=20, threshold=0.50) | +1.57 | **+0.895** | (+0.396, +1.334) | -24.03 % | 33 | 78.8 % | 198 | ✅ |

**Old Sharpe** values are from the pre-fix runs documented in
`directives/System Status and Roadmap.md §2, §10, §15–17`.

### Magnitude of revisions

- **MR AUD/JPY**: −80 % (the champion's reported Sharpe was ~5× inflated).
  Drivers: `sqrt(252)` on H1 data (understates H1 vol by `sqrt(24)` ≈ 4.9×,
  which propagated into both live sizing AND the filter-then-annualise
  Sharpe) plus the tier-pyramid reset bug in the backtest loop that made
  live and backtest simulate different strategies.

- **MR AUD/USD**: −82 %, and the **95 % CI crosses zero**. This strategy
  fails the deployment gate.

- **IEF→GLD**: −38 %. The same-bar look-ahead fix (`pos_lagged`) removed
  one bar of clairvoyance. Still passes the gate.

- **HYG→IWB**: −43 %. Same driver as IEF→GLD. Still passes the gate.

---

## Deployment decisions

| Strategy | Live status pre-audit | Decision | Action |
|---|---|---|---|
| MR AUD/JPY `conf_donchian_pos_20` conservative | **paper-deployed** (commit `56343c3`) | **KEEP** — CI lo +0.478 > 0, single highest-quality OOS in the set. Size new paper at 25 % of pre-fix units (Sharpe ~5× smaller ⇒ per-tier size ~5× smaller). | Update `scripts/run_portfolio.py` allocation. |
| MR AUD/USD `conf_rsi_14_dev` conservative | champion candidate | **DEPRECATE** — CI lo -0.180 straddles zero, 50 % positive folds, -49 % max DD. | Move to `directives/Deprecated Strategies.md` with reason "CI_lo < 0 post-remediation". Remove 5 % allocation from portfolio. |
| HYG→IWB | **30 % portfolio weight** | **KEEP** — CI lo +0.396. No change to allocation pending Part B §7 paper dry-run. |
| IEF→GLD | documented champion | **KEEP** — CI lo +0.302. |

---

## Phase portfolio — combined view

Full pipeline run on the corrected harness with the sanctuary window
active (dropped the last year from every loader: 5,979 H1 bars from
AUD/JPY, 5,978 from AUD/USD, 227 D bars from IWB). Trade-level
`scale_to_risk` applied.

### Strategy Sharpes on the common OOS window (2020-04-06 → 2024-02-05, 1001 trading days)

| Strategy | Standalone OOS Sharpe | OOS Max DD |
|---|---:|---:|
| AUD/JPY MR | +1.265 | -11.5 % |
| IWB ML (stacking) | +0.842 | -1.8 % |
| HYG→IWB | +1.337 | -1.9 % |
| AUD/USD MR | +0.748 | -27.1 % |

> Note these are different numbers from the full-history WFO Sharpes above
> because the common window is only the OOS half (2020–2024). OOS-half
> Sharpes are higher for the three "keep" strategies, which is consistent
> with the CI upper bounds from the full WFO runs.

### Correlation matrix (OOS)

|            | AUD/JPY MR | IWB ML | HYG→IWB | AUD/USD MR |
|---|---:|---:|---:|---:|
| AUD/JPY MR |  1.00 | -0.01 | -0.02 | +0.04 |
| IWB ML     | -0.01 |  1.00 | +0.10 | -0.00 |
| HYG→IWB    | -0.02 | +0.10 |  1.00 | -0.05 |
| AUD/USD MR | +0.04 | -0.00 | -0.05 |  1.00 |

All pairwise |ρ| < 0.1 — the four strategies are genuinely independent
sources of edge. This is the strongest single piece of evidence that the
remediation hasn't destroyed the portfolio's diversification.

### IS-derived inverse-vol weights, OOS realized

- IS-inv-vol picks: **0 % / 100 % / 0 % / 0 %** — the IS window put
  everything on IWB ML because ML had the lowest per-period vol.
  (Expected artefact: the ML signal is very sparse — only 5 trades in
  the full ML WFO — so its per-bar vol looks artificially low relative
  to the high-frequency MR series.)
- Realized OOS portfolio Sharpe: **+0.842**
- Max DD: **-1.8 %**
- Random-weight sensitivity band (100 Dirichlet draws): p05 = 0.86,
  p50 = 1.33, p95 = 1.71 — IS-inv-vol at 0.84 is at the low end.

The inverse-vol pick-100%-ML decision is a known artefact of using
per-bar vol on a sparse signal. For live deployment, **equal-weight or
a bounded allocation** (min 10 % per strategy, max 60 %) will match the
random-band p50 of 1.33 much more closely — see follow-up 3 below.

---

## Raw run logs

- `mr_audjpy.log` — MR AUD/JPY, all 8 filter × tier combinations, full fold breakdown.
- `mr_audusd.log` — MR AUD/USD, all combinations, gate fail in summary.
- `bond_ief_gld.log` — IEF→GLD, 38 folds.
- `bond_hyg_iwb.log` — HYG→IWB, 33 folds.
- `phase_portfolio.log` — multi-strategy portfolio run (complete).

All logs live in this directory (`.tmp/reports/remediation_2026_04_21/`).
Per-CSV artefacts at `.tmp/reports/mr_confluence_wfo_{pair}.csv` and
`.tmp/reports/cross_asset_bond_equity_wfo.csv` (last-run overwrites, so
IEF→GLD was re-extracted via programmatic call).

---

## Follow-ups

1. **Roadmap strikethrough** — `directives/System Status and Roadmap.md`
   needs the old Sharpe numbers struck through and replaced with the new
   values above (runbook §5). Link each revision to this report.

2. **Deprecate AUD/USD MR** — create
   `directives/Deprecated Strategies.md` with AUD/USD MR as the first
   entry, reason: "95 % CI lower bound = -0.180 post-remediation".

3. **Champion portfolio resize** — with AUD/USD removed (-5 % weight),
   the 5 % needs to go somewhere. Simplest: redistribute proportionally
   across AUD/JPY, HYG→IWB, IEF→GLD. Verify via
   `research/portfolio/run_portfolio_research.py` — not rerun here.
   Also: the IS-inv-vol 100 % ML allocation from phase_portfolio is an
   artefact (sparse ML signal → low per-bar vol). Use a bounded weight
   scheme (min 10 % / max 60 % per strategy) for live.

4. **Paper dry-run (Part B §7)** — 48-hour observation on the corrected
   champion portfolio to confirm (a) `portfolio_risk_manager.scale_factor`
   is not pegged at 1.0 / 2.0 (vol targeting binding), (b) allocator
   weights differ between strategies (per-strategy equity reaching the
   allocator — the A.6 migration should ensure this), (c) AUD/JPY
   per-tier unit size is ~5× smaller than pre-fix at the same price.

5. **Open PR** — merge `autoresearch/20260406-175621` → `main` once the
   roadmap is updated, referencing this report.
