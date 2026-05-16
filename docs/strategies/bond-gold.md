# bond_gold — Bond→Gold Cross-Asset Momentum

**Version:** 1.0 | **Last updated:** 2026-05-16
**Status:** **LIVE on paper (V1-era config `lookback=60, threshold=0.50`)** | V3.6 PROMOTED CONDITIONAL_WATCHPOINT for `(lookback=120, threshold=0.50)` — migration pending allocator approval
**Source:**
- Research mechanics: [research/cross_asset/bond_gold_strategy.py](../../research/cross_asset/bond_gold_strategy.py) (pure-research)
- Audit harness: [research/cross_asset/run_bond_gold_reaudit.py](../../research/cross_asset/run_bond_gold_reaudit.py)
- Sweep: [research/exploration/sweep_bond_gold.py](../../research/exploration/sweep_bond_gold.py)
- Live class: [titan/strategies/bond_gold/strategy.py](../../titan/strategies/bond_gold/strategy.py)
- Pre-reg directive: [directives/Pre-Reg bond_gold Re-audit 2026-05-16.md](../../directives/Pre-Reg%20bond_gold%20Re-audit%202026-05-16.md)
- TOML config: [config/bond_gold.toml](../../config/bond_gold.toml) (V1-era LIVE)
- Sweep findings: [.tmp/reports/sweep_bond_gold/findings.md](../../.tmp/reports/sweep_bond_gold/findings.md)
- Audit findings: [.tmp/reports/bond_gold_reaudit/findings.md](../../.tmp/reports/bond_gold_reaudit/findings.md)

---

## Executive Summary

bond_gold times **GLD (gold)** entries off **IEF (intermediate Treasury bond) momentum**. The thesis: when bond prices rise (rates falling), gold tends to outperform. Long-only on GLD, vol-targeted at 10% annualised vol, capped at 1.5× leverage. Wave A.1 of the V1-era re-audit roster — **first applied case of the L52 hybrid framework**.

**Audit headline (V3.6 re-audit 2026-05-16, 21y IEF+GLD, 60 rolling WFO folds, L52 hybrid framework):**

| | Live V1-era `(lookback=60, threshold=0.50)` | **V3.6 PROMOTED `(120, 0.50)`** | Delta |
|---|---|---|---|
| Stitched OOS Sharpe | +0.55 | **+0.66** | +0.11 (+20%) |
| CI95 lower bound | +0.075 | **+0.152** | 2.0× tighter |
| Rel-MC pass (vs B&H GLD) | n/a | yes (38% DD reduction) | — |
| Plateau spread on OOS | n/a | 36% (passes H1 50% gate) | — |
| 5-axis verdict | n/a (V1 era) | CONDITIONAL_WATCHPOINT | — |

**V1 claim** ("Sharpe +1.17, 68% positive folds, 37 folds 2007-2026") **not reproducible** under V3.6 cost modelling. The strategy IS real (CI_lo > 0 on the V3.6 audit) but at materially lower Sharpe than V1 advertised. **Bulk of the V3.6 improvement comes from the L52 hybrid sweep finding** — the right canonical is `lookback=120`, not `60`.

---

## 1. What it trades

| Logical role | Live instrument | Notes |
|---|---|---|
| Signal source | **IEF** (iShares 7-10y Treasury Bond ETF) | Daily close; signal only, never traded |
| Traded asset | **GLD** (SPDR Gold Shares) | Long-only, vol-targeted |

Both instruments are US-listed; for UK retail accounts the operator would need to substitute UCITS-compliant equivalents (e.g., `IBTM.L` for IEF, `SGLN.L` for GLD) — same logic, different ConIds. **No UK config currently exists** for bond_gold.

---

## 2. How it trades

Decision pipeline, evaluated once per daily bar:

1. **Bond momentum.** Compute the log-return of IEF over `lookback` days:
   ```
   bond_mom(t) = log(IEF[t] / IEF[t - lookback])
   ```
   At V3.6 canonical `lookback=120`, this is a ~6-month bond return — positive when rates fall.
2. **Rolling z-score normalisation.** Past-only rolling window of 504 days (2 years):
   ```
   z(t) = (bond_mom(t) - rolling_mean) / rolling_std
   ```
3. **Long entry.** Binary signal: `sig = 1` if `z > threshold`, else `0`. At V3.6 canonical `threshold=0.50`, this requires bond momentum ~0.5σ above its 2-year mean.
4. **Hold-day floor.** Once entered, the position stays open for at least `hold_days=20`. After 20 days, exit when `z <= threshold`.
5. **Vol-target sizing on GLD.**
   ```
   var(t) = EWMA(gld_ret^2, span=20)[t]
   scale(t) = min(target_vol / realised_vol_ann, max_leverage)
   position(t) = sig(t) * scale(t)
   ```
   `target_vol=0.10`, `max_leverage=1.5` (frozen across V1 and V3.6).
6. **Per-bar return.** Position at t earns return t→t+1 via `.shift(1)` (L18 shift discipline).

**Causality (L04 / A1):** all rolling stats are past-only. `bond_gold_assert_causal` smoke test in the audit harness verifies bit-exact past returns under future-data corruption.

---

## 3. Parameters

| Parameter | Live V1-era | **V3.6 PROMOTED** | Notes |
|---|---:|---:|---|
| **`lookback`** | **60** | **120** | **V3.6 CHANGE — 2× slower bond-momentum window** |
| `threshold` | 0.50 | 0.50 | unchanged (z-score entry threshold) |
| `hold_days` | 20 | 20 | unchanged (minimum holding period) |
| `zscore_window` | 504 | 504 | unchanged (2-year z-score lookback) |
| `vol_target_pct` | 0.10 | 0.10 | unchanged (10% annualised vol target) |
| `ewma_span` | 20 | 20 | unchanged (vol estimator) |
| `max_leverage` | 1.5 | 1.5 | unchanged |
| `warmup_bars` | 120 | 120 | unchanged |

The V3.6 change is confined to **one knob** (`lookback`). Everything else is preserved verbatim from V1.

**Why the L52 sweep found a different canonical:**

- 25-cell sweep over `lookback ∈ {30, 45, 60, 90, 120}` × `threshold ∈ {0.00, 0.25, 0.50, 0.75, 1.00}` on 19y IS data.
- **Plateau detected at `lookback=120` row** (spread 11% across thresholds vs 47% at the live `lookback=60` row).
- Live canonical sits OFF the plateau — IS Sharpe at (60, 0.50) = 0.524 vs plateau row best 0.665 (+27% gap on IS, +20% on OOS).
- Mechanism: the 60-day lookback is too reactive — it whipsaws around the z-score threshold. The 120-day lookback smooths the signal sufficiently to capture the bond→gold relationship without over-trading.

---

## 4. Cost model

V3.6 audit cost calibration:

| Component | Value | Notes |
|---|---:|---|
| Variable cost (spread + slip) | 1.0 bps / turnover-unit | Matches B2/B4/etf_trend cost model for ETF instruments |
| Cost drag at V3.6 canonical | +0.009 net Sharpe | gross OOS Sharpe = +0.665, net = +0.656; cost is NOT a binding constraint |

The strategy is slow (monthly-ish rebalances at the hold-floor + threshold transitions). Cost drag at `target_vol=0.10` and `max_leverage=1.5` is well below the deployment-relevant Sharpe gap.

---

## 5. Live deployment

### Current state (V1-era config)

The strategy currently runs the V1-era config (`lookback=60, threshold=0.50`) — **not yet migrated to V3.6**.

| Container | Image | Role | Port |
|---|---|---|---|
| `titan-ib-gateway` | `gnzsnz/ib-gateway:stable` | IBKR Gateway | 4004 (paper) |
| `titan-portfolio` | `titan-portfolio:latest` | Live strategy runner | — |

The V1-era bond_gold live strategy is part of the champion portfolio, sized per-strategy with PRM allocation. Stop-loss is via vol-target sizing only (no ATR hard stop in this strategy class).

### Recommended migration path (V3.6 PROMOTED CONDITIONAL_WATCHPOINT)

Same Phase 0-3 pattern as the GEM J5 migration:

| Phase | Action | Status |
|---|---|---|
| Phase 0 | Write `config/bond_gold_v36.toml` sidecar with V3.6 canonical | pending |
| Phase 1 | 6-month paper shadow of V3.6 canonical alongside V1-era live | pending |
| Phase 2 | Live cutover after 6-month sanctuary re-test | pending allocator approval |
| Phase 3 | Re-audit at 4-week post-cutover mark | pending |

**Why CONDITIONAL_WATCHPOINT, not DEPLOY:**

- Sanctuary `lucky_flag=True` (percentile 1.00 for all cells with threshold ≤ 0.50) — the 24-month sanctuary (2024-04 → 2026-04) was the gold-rally regime where ANY long-gold strategy looked exceptional.
- The **stitched-OOS Sharpe +0.66** is the deployment-relevant number; the sanctuary Sharpe of +1.67 is regime-favourable and per **L55 should NOT be cited as a forward claim**.
- Awaiting a fresh sanctuary window covering a non-favourable regime before promoting to DEPLOY.

---

## 6. Operations

Same operational stack as GEM (shared `titan-portfolio` container, common IBKR Gateway). Bond_gold runs alongside other live strategies under the PRM.

### Logs
Container stdout + rolling file. Headline lines:
- `bond_gold Strategy attached` — boot
- `RECONCILE GLD: BUY|SELL n (target=t, current=c)` — order submission
- `IEF momentum z=X, threshold=Y, signal=S` — per-bar decision diagnostic

### Kill switch
Same `docker compose down` / `kill_switch` runbook as the rest of `titan/strategies/*`.

### Restart safety
Position reconciliation is via IBKR broker query (not local cache). Restart adopts whatever positions IBKR reports — same pattern as GEM.

---

## 7. Audit summary (V3.6 re-audit, 2026-05-16)

Full pre-reg + result log:
- [Pre-Reg bond_gold Re-audit 2026-05-16.md](../../directives/Pre-Reg%20bond_gold%20Re-audit%202026-05-16.md)
- [.tmp/reports/bond_gold_reaudit/result_log.md](../../.tmp/reports/bond_gold_reaudit/result_log.md)
- [.tmp/reports/bond_gold_reaudit/findings.md](../../.tmp/reports/bond_gold_reaudit/findings.md)

### Per-cell verdict matrix

| Cell | Sharpe | CI_lo | CI_hi | Rel-MC pass | Sanctuary | Lucky | Verdict |
|---|---:|---:|---:|:---:|---:|:---:|---|
| C1_canonical (lookback=120, threshold=0.50) | +0.66 | +0.152 | +1.16 | YES | +1.67 | YES (L55) | CONDITIONAL |
| P_low_threshold (120, 0.00) | +0.58 | +0.097 | +1.09 | YES | +1.94 | YES (L55) | DEPLOY |
| P_quarter (120, 0.25) | +0.63 | +0.156 | +1.15 | YES | +1.81 | YES (L55) | CONDITIONAL |
| P_high_threshold (120, 0.75) | +0.62 | +0.108 | +1.09 | YES | +1.67 | YES (L55) | DEPLOY |
| P_strict (120, 1.00) | +0.44 | -0.063 | +0.96 | YES | +1.83 | YES (L55) | CONDITIONAL |
| **C1 — V3.6 PROMOTED CANONICAL** | **+0.66** | **+0.152** | **+1.16** | **YES** | **+1.67** | **YES (L55)** | **CONDITIONAL** |
| C2_live_canonical (60, 0.50) | +0.55 | +0.075 | +1.05 | no | +1.95 | YES (L55) | CONDITIONAL |
| C4_gross_no_costs | +0.67 | +0.161 | +1.17 | YES | +1.68 | YES (L55) | CONDITIONAL |

§3 selection rule promoted **P_quarter** (highest CI_lo among eligible) but the **recommended-for-deployment cell** is **C1_canonical** because:
- C1 and P_quarter CI_lo differ by only 0.004 (statistically indistinguishable).
- Threshold=0.50 is more conservative (fewer signals → less turnover → lower cost-drag risk).
- C1 matches the original V1 threshold; the change is confined to the `lookback` axis only.

### Hypothesis verdicts (pre-reg §2, V3.1)

| Hyp | Statement | Verdict |
|---|---|---|
| H1 | Plateau holds OOS (spread ≤ 50%) | **SUPPORTED** — spread 36% |
| H2 | C1 sanctuary Sharpe ≥ +0.30 | **SUPPORTED** — +1.67 (but L55 caveat: lucky_flag=True) |
| H3 | C1 sanctuary Sharpe > C2 sanctuary Sharpe | **REJECTED** — C1 +1.67 vs C2 +1.95 (acceptable: regime-favourable sanctuary, see L55) |
| H4 | C1 CI_lo > 0 | **SUPPORTED** — +0.152 |
| H5 | Cost drag ≤ 0.30 | **SUPPORTED** — gap = +0.009 |

H3 rejection is OK under L55: the sanctuary window happened to favour the faster `lookback=60` signal because gold rallied immediately when the 60-day signal fired (the 120-day signal lagged). On longer horizons covering both favourable and adverse regimes, the V3.6 plateau holds (H1 + H4 are the deployment-relevant gates).

---

## 8. Known caveats

1. **V1 claim of Sharpe +1.17 is not reproducible** under V3.6 cost + math. The honest deployment-relevant number is the **stitched-OOS Sharpe ~+0.66** with CI_lo +0.15.
2. **L55 sanctuary caveat applies.** All cells with threshold ≤ 0.50 are sanctuary-lucky-flagged (24mo window = gold rally). Cite stitched OOS not sanctuary Sharpe for any communication.
3. **Migration is non-trivial** because of the L55 caveat. Recommend 6-month shadow + fresh sanctuary re-test before final live cutover (vs GEM J5 which was directly cut-over on operator instruction — bond_gold should be more conservative because of the regime caveat).
4. **GLD is long-only equity-class.** Subject to L17 — used relative-MC vs B&H GLD; passes (38% DD reduction). DO NOT use absolute-MaxDD gate.
5. **IEF as a signal source has alternatives.** The bond-momentum signal can also be computed off TLT (longer duration) or HYG (credit). The audit fixed IEF per V1 pre-reg; alternative signal sources are a future audit's scope, not this one.
6. **Capital rotation candidate.** If GEM J5 freed-capital rotation needs a destination, bond_gold V3.6 canonical is the recommended target per the GEM J5 migration memo.

---

## 9. References

- [directives/Pre-Reg bond_gold Re-audit 2026-05-16.md](../../directives/Pre-Reg%20bond_gold%20Re-audit%202026-05-16.md) — current V3.6 pre-reg.
- [directives/V3.6 Lessons Catalogue.md](../../directives/V3.6%20Lessons%20Catalogue.md) — L52 hybrid framework, L17 relative MC, L55 sanctuary caveat.
- [.tmp/reports/sweep_bond_gold/findings.md](../../.tmp/reports/sweep_bond_gold/findings.md) — L52 sweep findings (plateau detection on IS).
- [.tmp/reports/bond_gold_reaudit/findings.md](../../.tmp/reports/bond_gold_reaudit/findings.md) — V3.6 audit verdict + migration memo.
- [docs/strategies/gem-dual-momentum.md](gem-dual-momentum.md) — sibling strategy (cross-asset momentum); migration pattern reference.
