# Deprecated Strategies

Strategies that were deployed or championed pre-audit and have been
retired after the April 2026 remediation. Each entry records:

* the original deployment context,
* the failure criterion on the corrected harness,
* the revised performance numbers,
* any migration notes (what replaced it in the portfolio, if anything).

Strategies listed here should NOT be re-enabled in the live portfolio,
autoresearch catalogues, or portfolio optimizer candidate sets without
an explicit re-validation on the corrected harness.

---

## MR AUD/USD (conf_rsi_14_dev, conservative) — deprecated 2026-04-21

**Previously**: 5 % weight in the champion portfolio, commit `56343c3`
(pre-audit), deployed to paper trading.

**Failure reason**: Post-audit WFO on the corrected harness gives

| Metric | Value |
|---|---|
| Stitched Sharpe | +0.374 |
| 95 % CI | (**-0.180**, +0.831) |
| n_folds | 8 |
| % positive folds | 50 % |
| Max DD | -49.0 % |
| n_trades | 385 |

The **CI lower bound is below zero**, meaning on resamples of the
actual OOS returns there is a non-trivial probability that the true
Sharpe is zero or negative. The deployment gate from the remediation
runbook (CI_lo > 0) is not met.

**Root cause of pre-audit Sharpe +2.10 claim**: the combination of
(a) `sqrt(252)` applied to H1 data (under-states true H1 vol by
`sqrt(24)`, so Sharpe is over-stated by ~4.9× before any annualisation
factor inside the stats helper), and (b) the filter-then-annualise
bias (`rets[rets != 0.0]` dropped before `/ std`) over-stating Sharpe
by an additional `sqrt(1/active_ratio)`.

On the corrected harness (`titan.research.metrics.sharpe` with
explicit `periods_per_year=BARS_PER_YEAR["H1"]` and no zero-day
filtering), the real Sharpe is +0.374 — a factor-of-5.6 revision.

**Migration**:
- Removed from `scripts/run_portfolio.py` champion_portfolio bundle.
- Freed 5 % weight redistributed to the new v3 portfolio components
  (primarily cross-bond signals and MR AUD/JPY at corrected
  `vwap_anchor=24`).
- Instrument still present in `data/AUD_USD_H1.parquet`; the MR
  strategy class in `titan/strategies/mr_audjpy/strategy.py` is
  asset-agnostic so the code path is intact for future re-validation
  if the market regime changes.

**Artefacts**:
- Re-validation: `directives/Remediation Audit 2026-04-21.md` §
  Per-strategy results.
- Full WFO data: `.tmp/reports/mr_confluence_wfo_aud_usd.csv`.
- Re-rank sweep: `directives/Rerank Analysis 2026-04-21.md` (entry
  in the "MR on non-AUD FX is dead" section).

---

## Pre-audit Sharpe numbers — disavowed

Several directives contain Sharpe numbers that were computed under the
buggy math. These should not be cited as evidence of strategy edge;
they are historical records only.

| Strategy | Pre-fix Sharpe | Post-fix Sharpe | Revised CI_lo |
|---|---:|---:|---:|
| MR AUD/JPY (vwap_anchor=46, conservative) | +4.64 | +0.910 | +0.478 |
| MR AUD/JPY (vwap_anchor=24, conservative) | — | +0.97 | +0.47 (new) |
| MR AUD/USD (above) | +2.10 | +0.374 | -0.180 |
| IEF→GLD lb=60 | +1.17 | +0.721 | +0.302 |
| HYG→IWB lb=10 | +1.57 | +0.895 | +0.396 |

The AUD/JPY pre-fix number (+4.64) relied on the same math bugs plus
the pre-fix `vwap_anchor=46` that is now known to be a 4th-of-6 choice
on the corrected harness; at the corrected `vwap_anchor=24` the
honest Sharpe is +0.97.

---

## Change log

- **2026-04-21** — initial file. AUD/USD MR deprecated; pre-fix Sharpe
  numbers across the champion portfolio disavowed. Created as part of
  the April 2026 external quant audit remediation (branch
  `autoresearch/20260406-175621`).
