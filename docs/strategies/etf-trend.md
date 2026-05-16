# etf_trend — ETF Trend Family (7 variants, V3.6 mixed verdicts)

**Version:** 1.0 | **Last updated:** 2026-05-16
**Status:** **7 live variants** with mixed V3.6 verdicts. **SPY RETIRED** (Wave A.2, L56). **TQQQ PROMOTED CONDITIONAL** for `(slow_ma=150, exit_confirm_days=5)` (Wave A.2-confirm). **5 unleveraged variants (QQQ/IWB/EFA/DBC/GLD) flagged for bulk-retire** pending allocator action.
**Source:**
- Research mechanics: [research/etf_trend/etf_trend_spy_strategy.py](../../research/etf_trend/etf_trend_spy_strategy.py), [etf_trend_tqqq_strategy.py](../../research/etf_trend/etf_trend_tqqq_strategy.py)
- Audit harnesses: [run_etf_trend_spy_reaudit.py](../../research/etf_trend/run_etf_trend_spy_reaudit.py), [run_etf_trend_tqqq_reaudit.py](../../research/etf_trend/run_etf_trend_tqqq_reaudit.py)
- Sweeps: [sweep_etf_trend_spy.py](../../research/exploration/sweep_etf_trend_spy.py), [sweep_etf_trend_tqqq.py](../../research/exploration/sweep_etf_trend_tqqq.py)
- Live class: [titan/strategies/etf_trend/strategy.py](../../titan/strategies/etf_trend/strategy.py)
- Pre-regs: [Pre-Reg etf_trend SPY Re-audit 2026-05-16.md](../../directives/Pre-Reg%20etf_trend%20SPY%20Re-audit%202026-05-16.md), [Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md](../../directives/Bulk-Retire%20etf_trend%20Unleveraged%20Variants%202026-05-16.md)
- TOML configs: [config/etf_trend_*.toml](../../config/) (7 files)
- Findings: [SPY audit](../../.tmp/reports/etf_trend_spy_reaudit/findings.md), [TQQQ audit](../../.tmp/reports/etf_trend_tqqq_reaudit/findings.md)

---

## Executive Summary

The `etf_trend` family is 7 V1-era variants of a single strategy template: **long ETF when close > slow SMA, exit on confirmation of close < slow SMA, vol-target sized**. The Wave A.2 audits (2026-05-16) found the family splits cleanly on **L56**:

- **Unleveraged variants** (SPY, QQQ, IWB, EFA, DBC, GLD) **fail L17 relative-MC** (drawdown protection vs B&H is regime-specific, not statistically robust under bootstrap). Recommended: **bulk-retire** without per-variant audits. Risk: ≤2% portfolio Sharpe if any one of the 5 actually deserved CONDITIONAL — recoverable.
- **Leveraged variant** (TQQQ) **also fails L17 rel-MC** but the +54% Sharpe edge over B&H is enough to absorb the noise-mitigation cost (longer `exit_confirm_days`). Wave A.2-confirm promoted `(slow_ma=150, exit_confirm_days=5)` to CONDITIONAL_WATCHPOINT.

**Audit headline:**

| Variant | Live config | V3.6 verdict | OOS Sharpe | vs B&H | Status |
|---|---|---|---:|---:|---|
| **SPY** | (150, 1, vt=0.20) | **RETIRED** (Wave A.2) | +0.60 | +6% over B&H | Audited; flagged for de-allocation |
| **TQQQ** | (175, 1, binary) | **CONDITIONAL** (Wave A.2-confirm) | +0.73 (live), +1.00 (promoted) | +54% over B&H | Live; migration to `(150, 5)` pending |
| **QQQ** | (200, 5, vt=0.25) | **RETIRE-RECOMMENDED** (bulk, L56) | n/a | n/a | Awaiting allocator action |
| **IWB** | (150, 3, binary) | **RETIRE-RECOMMENDED** (bulk, L56) | n/a | n/a | Awaiting allocator action |
| **EFA** | (200, 5, binary) | **RETIRE-RECOMMENDED** (bulk, L56) | n/a | n/a | Awaiting allocator action |
| **DBC** | (75, 1, binary) | **CONFIRM-AUDIT-RECOMMENDED** (commodity asset class differs) | n/a | n/a | Awaiting spot-check audit |
| **GLD** | (250, 5, binary) | **CONFIRM-AUDIT-RECOMMENDED** (gold-specific) | n/a | n/a | Awaiting spot-check audit |

---

## 1. What they trade

| Variant | Instrument | Asset class | Class default sizing | Live in titan/strategies/ |
|---|---|---|---|---|
| `etf_trend_spy` | SPY (ARCA) | US large-cap equity | vol_target=0.20, max_leverage=2.0 | yes |
| `etf_trend_qqq` | QQQ (NASDAQ) | US tech-heavy equity | vol_target=0.25, max_leverage=1.5 | yes |
| `etf_trend_iwb` | IWB (ARCA) | Russell 1000 (broad US equity) | binary 0/1 | yes |
| `etf_trend_efa` | EFA (ARCA) | Developed-markets ex-US | binary 0/1 | yes |
| `etf_trend_dbc` | DBC (ARCA) | Broad commodity basket | binary 0/1 | yes |
| `etf_trend_gld` | GLD (ARCA) | Physical gold | binary 0/1 | yes |
| `etf_trend_tqqq` | TQQQ (NASDAQ), signal from QQQ | 3× leveraged NASDAQ | binary 0/1 | yes |

All variants share the same `ETFTrendStrategy` class in [titan/strategies/etf_trend/strategy.py](../../titan/strategies/etf_trend/strategy.py); they differ only in `config/etf_trend_<TICKER>.toml`.

---

## 2. How they trade (shared template)

Decision pipeline, evaluated once at daily-bar close, executed at MOC (Market-On-Close) the **next** session:

1. **Slow MA boundary.** `sma(t) = SMA(close, slow_ma)`. Regime is "long-eligible" when `close > sma`.
2. **Exit confirmation.** Once long, increment a counter when `close < sma`. Exit only when counter reaches `exit_confirm_days` consecutive bars — suppresses single-bar whipsaws.
3. **Sizing** (variant-dependent):
   - `sizing_mode = "vol_target"` (SPY, QQQ): scale position to target annualised vol; cap at `max_leverage`. Excess capital sits in cash (no auto-rotation to a safety asset).
   - `sizing_mode = "binary"` (IWB, EFA, DBC, GLD, TQQQ): position is 0 or 1.0. TQQQ's 3× leverage IS the position sizing.
4. **MOC submission filter.** Order is suppressed unless per-asset position delta exceeds the 5% rebalance threshold (suppresses ~70% of vol-target tweaks that would hit the IBKR commission floor).
5. **Hard stop** (optional): `atr_stop_mult × ATR` trailing stop, STOP_MARKET GTC alongside the position. Risk-management overlay; not part of the signal-edge sweep.

All variants are **causal** (signal at `t` uses data through `t-1`; position effective at next-session MOC = `t+1` open).

---

## 3. V3.6 audit findings (Wave A.2 + A.2-confirm, 2026-05-16)

### SPY (Wave A.2 — RETIRED)

- **Sweep IS plateau:** `slow_ma=300` row (5% spread). Live `(150, 1)` is OFF the plateau (IS Sharpe 0.62 vs plateau best 0.72).
- **Audit OOS:** all 8 cells pass plateau (5% spread, best so far) AND have positive Sharpe with CI_lo > 0. But **L17 relative-MC FAILS** for every cell (median DD reduction 0.99-1.06 vs 0.80 gate — strategy ≈ B&H under bootstrap). **Varma noise axis = worst** for every cell (MA-crossover signal flips on small perturbations near the SMA boundary).
- **Verdict: RETIRED.** The strategy adds <10% Sharpe over B&H AND fails the drawdown-protection claim under bootstrap. Live `(150, 1)` config also fails — recommend de-allocation or conversion to vol-targeted B&H SPY.
- **New lesson L56** documented; refined by TQQQ below.

### TQQQ (Wave A.2-confirm — CONDITIONAL_WATCHPOINT)

- **Sweep IS plateau:** noisy surface (no clean plateau like SPY had). Best individual cell `(slow_ma=150, exit_confirm_days=1)` at IS Sharpe 0.75 vs B&H 0.57 (+32%).
- **Audit OOS:** rel-MC FAILS for every cell (median DD reduction 0.98-0.99) — **L56 generalises** to leveraged ETFs. But **noise axis = best at exit_confirm_days ≥ 5** (the longer confirmation absorbs the noise sensitivity). Combined with the +54% Sharpe edge over B&H, this is enough for ONE cell to earn CONDITIONAL_WATCHPOINT.
- **Verdict: PROMOTED CONDITIONAL** for `(slow_ma=150, exit_confirm_days=5)`. OOS Sharpe +0.67, CI_lo +0.10, noise=best.
- Live `(175, 1)` config FAILS noise axis (mid at best). **Migration to `(150, 5)` recommended after 6-month shadow.**
- **L56 refined:** rel-MC failure generalises, BUT exit_confirm_days is a noise-mitigation lever for high-Sharpe-edge underlyings. Leveraged ETFs need individual audits; unleveraged can be bulk-retired.

### QQQ, IWB, EFA, DBC, GLD (Bulk-retire recommendation, no individual audits)

Per [directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md](../../directives/Bulk-Retire%20etf_trend%20Unleveraged%20Variants%202026-05-16.md):

- **High-confidence retire (L56 directly applies):** QQQ, IWB, EFA — long-only equity ETFs, same mechanism as SPY. Will fail L17 rel-MC + Varma noise for the same reason. Audit would consume ~5h compute and produce predictable RETIRED verdicts.
- **Medium-confidence retire (verify before final action):** DBC (commodity basket — different crash structure), GLD (gold — see bond_gold; the IEF→GLD signal already shows positive value). **Recommended: fast spot-check audit before de-allocation** (~30 min per audit via the existing harness with symbol substitution).

**Allocator action plan:**

1. **Phase 1 (immediate):** De-allocate QQQ, IWB, EFA. Free capital → bond_gold CONDITIONAL or cash buffer.
2. **Phase 2 (2 weeks):** Spot-check DBC + GLD. Confirm RETIRED or upgrade to individual audit.
3. **Phase 3 (1 month):** Migrate TQQQ to V3.6 canonical `(150, 5)` after 6-month shadow.

---

## 4. Parameters (per-variant live configs)

| Variant | slow_ma | exit_confirm_days | sizing_mode | vol_target | max_leverage | atr_stop_mult | exit_mode |
|---|---:|---:|---|---:|---:|---:|:---:|
| SPY | 150 | 1 | vol_target | 0.20 | 2.0 | 5.0 | A (close < SMA) |
| QQQ | 200 | 5 | vol_target | 0.25 | 1.5 | (default) | (default) |
| IWB | 150 | 3 | binary | n/a | n/a | (default) | (default) |
| EFA | 200 | 5 | binary | n/a | n/a | (default) | (default) |
| DBC | 75 | 1 | binary | n/a | n/a | (default) | (default) |
| GLD | 250 | 5 | binary | n/a | n/a | (default) | (default) |
| TQQQ | 175 (live) → **150 (V3.6)** | 1 (live) → **5 (V3.6)** | binary | n/a | n/a | 3.0 | D (close < SMA OR decel) |

All variants share `entry_mode="decel_positive"` with `decel_signals=[]` (the decel composite is disabled in the live configs).

---

## 5. Live deployment

All 7 variants run under the shared `ETFTrendStrategy` class in [titan/strategies/etf_trend/strategy.py](../../titan/strategies/etf_trend/strategy.py). Each has its own TOML config in [config/etf_trend_*.toml](../../config/) and is started via the champion-portfolio runner ([scripts/run_portfolio.py](../../scripts/run_portfolio.py)) with the appropriate instrument id.

### Containers + ports

Same shared stack as GEM/bond_gold:

| Container | Image | Role |
|---|---|---|
| `titan-ib-gateway` | `gnzsnz/ib-gateway:stable` | IBKR Gateway (port 4004 paper) |
| `titan-portfolio` | `titan-portfolio:latest` | Runs the champion-portfolio runner |

### Migration to V3.6 verdicts

| Variant | Action | Procedure |
|---|---|---|
| SPY | De-allocate | Set allocation to 0 in champion-portfolio registry; restart `titan-portfolio` |
| QQQ/IWB/EFA | De-allocate | Same |
| DBC/GLD | Spot-check audit, then de-allocate if confirmed | Run `research/etf_trend/run_etf_trend_<TICKER>_reaudit.py` (copy-adapt the SPY harness) |
| TQQQ | Migrate to V3.6 canonical | Overwrite `config/etf_trend_tqqq.toml` with `slow_ma=150, exit_confirm_days=5`; restart container |

**Sequencing recommendation:** do not retire ALL variants in a single rebalance. Stagger over 2-3 weeks to monitor portfolio-level vol behaviour as the etf_trend exposure unwinds.

---

## 6. Operations (shared)

### Warmup
Each strategy loads `data/<TICKER>_D.parquet` for `warmup_bars=300` daily bars on `on_start`. Missing parquets trigger a `WARN` and the strategy waits for live bars.

### Logs
- `ETF Trend Strategy Started -- <TICKER>.ARCA | MA=SMA(slow=N) | entry=... | exit=... | sizing=... | atr_mult=N`
- `Daily bar <DATE>: close=X regime=R decel=D` — per-bar diagnostic
- `RECONCILE <TICKER>: BUY|SELL n (target=t, current=c)` — MOC submission

### Kill switch
Standard `docker compose down` or `kill_switch` runbook.

---

## 7. Known caveats

1. **L56 is the family-defining lesson.** MA-crossover trend filters on long-only equity systematically fail L17 relative-MC under bootstrap. **Do NOT cite "drawdown protection vs B&H" as a deployment claim for any unleveraged variant.**
2. **L56 is refined for leveraged variants.** TQQQ survives because the 3× leverage amplifies the Sharpe edge above the noise-mitigation cost. **This refinement does NOT extend to unleveraged variants** — the Sharpe edge there is too small to absorb the cost of slower entry/exit.
3. **Sanctuary luckiness varies.** SPY audit's sanctuary period (2024-05 → 2026-05) was NOT lucky-flagged. TQQQ audit's sanctuary (2024-03 → 2026-03) was also not lucky-flagged (note: this differs from bond_gold and GEM where sanctuary was lucky-flagged). The sanctuary regime for tech-heavy long-leveraged was apparently within historical norms.
4. **DBC and GLD spot-checks may surprise.** Commodity and gold have different drawdown structures than equity. The bulk-retire memo flags both as medium-confidence; spot-check audits before final action.
5. **Stop-loss overlay is NOT modelled in the audit.** The `atr_stop_mult` hard stop is risk-management infrastructure; it doesn't affect the signal-edge audit. Live deployment should keep the stop until/unless a follow-up audit explicitly tests with stop in place.

---

## 8. References

- [directives/Pre-Reg etf_trend SPY Re-audit 2026-05-16.md](../../directives/Pre-Reg%20etf_trend%20SPY%20Re-audit%202026-05-16.md) — SPY V3.6 pre-reg.
- [directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md](../../directives/Bulk-Retire%20etf_trend%20Unleveraged%20Variants%202026-05-16.md) — bulk-retire memo for 5 variants.
- [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md) — L56 (MA-crossover trend filter on long-only equity), L17 (relative MC), L52 (hybrid framework).
- [.tmp/reports/etf_trend_spy_reaudit/findings.md](../../.tmp/reports/etf_trend_spy_reaudit/findings.md) — SPY audit.
- [.tmp/reports/etf_trend_tqqq_reaudit/findings.md](../../.tmp/reports/etf_trend_tqqq_reaudit/findings.md) — TQQQ audit + L56 refinement.
- [docs/strategies/gem-dual-momentum.md](gem-dual-momentum.md) — sibling strategy (cross-asset momentum, different mechanism, DEPLOY verdict).
- [docs/strategies/bond-gold.md](bond-gold.md) — sibling strategy (cross-asset bond-momentum → gold, CONDITIONAL_WATCHPOINT verdict).
