# Remediation Runbook — April 2026 External Quant Audit

## Context

The external quant review identified four bug families that inflated the reported performance of every champion-portfolio strategy and several Tier A/B signals:

1. **Frequency-mismatched annualisation** — `sqrt(252)` applied to H1/M5 data in live sizing and research Sharpe (both in `_compute_size` methods and in WFO helpers).
2. **Same-bar signal / return look-ahead** — `run_bond_equity_wfo.py` multiplied `pos[t]` (using `close[t]`) by `ret[t-1 -> t]`, giving the strategy one bar of clairvoyance.
3. **Tier pyramid accumulation drift** — MR backtest `position = np.zeros(n)` reset every bar, so later-bar tier entries replaced (rather than added to) prior tiers, making live and backtest simulate different strategies.
4. **Live / research divergence** — `bond_gold` expanding z-score vs. research IS-frozen; cross-asset bar arrival race on `bond_gold` / `pairs`; `turtle` using `list(balances.keys())[0]`; `mr_audjpy` silently defaulting `fx_rate_quote_to_base=1.0`.

All four families are **fixed in code** (see the remediation plan at `C:\Users\rayan\.claude\plans\i-need-you-to-encapsulated-wall.md` and the commit history on this branch). This runbook covers **Part B: re-validating the published performance numbers** against the corrected harness so deployment decisions are based on real Sharpes.

> [!CAUTION]
> Do NOT commit further capital to the champion portfolio until Part B §3 has been executed and the 95% Sharpe CI lower bound for every deployed strategy is above zero.

---

## Decisions confirmed with the user

| Decision | Choice |
|---|---|
| Scope | Fix all 13 strategies + every research WFO pipeline + re-validate every published Sharpe. |
| CI strictness | Hard-fail guardrails in CI (AST tests). Allowlist entries require an in-file justification comment. |
| Roadmap handling | Strikethrough original numbers + revised numbers alongside. Preserve the audit trail. |
| Statistical rigor | 12-month sanctuary window for autoresearch + 95% bootstrap CIs on every WFO Sharpe. |

---

## Part B — Re-validation checklist

Run this sequence on the `research-remediation` branch (or equivalent).

### 1. Freeze the pre-fix baseline
```bash
# Tag the pre-fix commit as the audit baseline — do this BEFORE cherry-picking
# any remediation commits onto main.
git tag preaudit-2026-04-21 <commit-hash>
git push origin preaudit-2026-04-21
```

### 2. Verify the code fixes land cleanly

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest tests/ -v
```

Expected: `80 passed, 1 skipped` (the `balance_total` scan is deferred to A.6).

Key tests that must pass:
- `tests/test_research_metrics.py` — 21 tests, shared metrics module contract
- `tests/test_research_rigor.py` — 8 tests, bootstrap CI + no-lookahead permutation
- `tests/test_research_math_guardrails.py` — 3 tests + 1 skipped, AST guardrails
- `tests/test_strategy_live_parity.py` — 6 tests, live-vs-research vol math parity
- `tests/test_portfolio_risk_april2026_fixes.py` — 13 tests, PRM regressions

### 3. Re-run every WFO on the corrected harness

```bash
# Champion portfolio source-of-truth WFOs:
uv run python research/mean_reversion/run_confluence_regime_wfo.py --pair AUD_JPY --is-bars 32000 --oos-bars 8000
uv run python research/mean_reversion/run_confluence_regime_wfo.py --pair AUD_USD --is-bars 30000 --oos-bars 7500
uv run python research/cross_asset/run_bond_equity_wfo.py --bond IEF --target GLD --lookback 60 --is-days 504 --oos-days 126
uv run python research/cross_asset/run_bond_equity_wfo.py --bond HYG --target IWB --lookback 10 --is-days 504 --oos-days 126
uv run python research/auto/phase_portfolio.py

# Record Sharpe, 95% CI, max DD, n_folds, % positive. Compare old-vs-new in
# .tmp/reports/remediation_audit_2026_04_XX.md.
```

Expected direction of change (not magnitudes — re-run to get those):
- **MR AUD/JPY** and **MR AUD/USD**: Sharpe will drop because the filter-zero-days bias is gone; pyramid fix may push in either direction.
- **IEF→GLD** and **HYG→IWB**: Sharpe will drop materially because the same-bar look-ahead is gone.
- **Phase-portfolio combined Sharpe**: will drop proportionally to the component changes.

### 4. Apply the deployment gate

For every strategy, check `sharpe_ci_95_lo` against zero:

| CI lower bound | Action |
|---|---|
| `> 0` | `tier=confirmed` — keep in champion registry. |
| `<= 0` | `tier=unconfirmed` — move to `directives/Deprecated Strategies.md` with a fail reason. |

### 5. Update the Roadmap with strikethrough

In `directives/System Status and Roadmap.md`, for every performance number affected:
- Strikethrough the old value with `~~+X.XX~~`.
- Add the new value immediately after: `**+Y.YY** (remediation 2026-04)`.
- Add a link to the remediation report line.
- For deprecated strategies, replace the row with a reference to the deprecated doc.

Affected sections: §2 (live strategies), §3 (ML signal map), §10 (multi-scale confluence), §15–17 (autoresearch / portfolio eval).

### 6. Sanctuary window

**Status: implemented.** `research/auto/evaluate.py` now exposes
`_enforce_sanctuary(df, source=...)`, applied inside `_load_daily` and
next to every `_load_ohlcv` / `load_h1` call in the WFO runners. Bars
within the last `SANCTUARY_DAYS` (default 365) are dropped before any
signal / fold builder sees them.

Opt-outs (both exist only for the human release-gate validation — the
autoresearch agent must use neither):

1. CLI flag `--include-sanctuary` on the invoking process.
2. Env var `TITAN_INCLUDE_SANCTUARY=1`.

Tests in `tests/test_sanctuary_window.py` lock in that the guard trims
by default, respects both opt-outs, and plays nicely with empty, tz-naive,
and non-datetime indexed frames.

### 7. 48-hour paper dry-run

With corrected code deployed on paper:
- Confirm `portfolio_risk_manager.scale_factor` is not pegged at 1.0 or 2.0 at rest (means vol targeting is binding).
- Confirm allocator weights differ between strategies (means per-strategy equity is reaching the allocator).
- Confirm AUD/JPY per-tier unit size is ~5× smaller than the pre-fix paper runs at comparable prices (vol annualisation fix landed).
- Monitor for any `[PortfolioRM] Concentration` warnings and halt persistence leaks.

### 8. Commit messages format

```
fix(remediation): <one-line summary>

Part of April 2026 external quant audit remediation.
Plan: C:\Users\rayan\.claude\plans\i-need-you-to-encapsulated-wall.md
Reference: SKILL.md §11 Research Math Pre-Flight
```

---

## Residual work (follow-up PRs)

### A.6: Migrate remaining 11 strategies to StrategyEquityTracker

Currently, only `mr_audjpy`, `mr_audusd`, and `bond_equity_ihyu_cspx` use the full tracker. The other 11 fall back to `get_base_balance(account, "USD")`, which re-introduces the original audit bug when `--strategies all` is used. Each needs:

```python
self._equity_tracker = StrategyEquityTracker(
    prm_id=self._prm_id,
    initial_equity=self.config.initial_equity,
    base_ccy=self.config.base_ccy,
)
# In on_bar:
_, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
# In on_position_closed:
self._equity_tracker.on_position_closed(pnl_quote, fx_to_base=fx_rate)
```

Enable the currently-skipped `test_balance_total_has_explicit_ccy` in `tests/test_research_math_guardrails.py` once this is done.

### Legacy research file migration

40+ research files still contain bare `sqrt(252)` (allowlisted in `tests/test_research_math_guardrails.py::_ALLOWLIST_SQRT_252`). Migrate one by one:

1. Replace `mean / std * sqrt(252)` with `titan.research.metrics.sharpe(returns, periods_per_year=BARS_PER_YEAR["D"])`.
2. Replace `std * sqrt(252)` with `titan.research.metrics.annualize_vol(std, periods_per_year=BARS_PER_YEAR["D"])`.
3. Replace `rolling().std() * sqrt(252)` with `titan.research.metrics.ewm_vol(rets, periods_per_year=BARS_PER_YEAR["D"])`.
4. Remove the allowlist entry and confirm the guardrail test still passes.

Prioritise files whose output feeds `research/portfolio/loaders/oos_returns.py`.

### Trade-level `scale_to_risk` plumbing

`research/auto/phase_portfolio.py::scale_to_risk` still uses an approximation — per-active-day vol instead of per-trade. Plumb trade-level returns through each runner (already a TODO in the code), then replace `bars_per_year` with `trades_per_year` for exact "1% per trade" semantics.

---

## Pointers

- **Remediation plan** (full detail): `C:\Users\rayan\.claude\plans\i-need-you-to-encapsulated-wall.md`
- **Titan-orchestrator SKILL.md** — §7 Research Math non-negotiables, §11 Research Math Pre-Flight, §10 Architect Mindset question 4.
- **Guardrail reference**: `C:\Users\rayan\.claude\skills\titan-orchestrator\references\research-math-guardrails.md` — worked examples of every failure mode.
- **Shared metrics module**: `titan/research/metrics.py` — single source of truth for Sharpe, vol, z-score, annualisation, bootstrap CI.
- **Memory entry**: `memory/feedback_research_math_discipline.md` — the discipline rules future sessions must follow.
