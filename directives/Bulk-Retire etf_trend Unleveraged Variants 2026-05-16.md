# Bulk-Retire Memo — etf_trend Unleveraged Variants (5 strategies)

**Date:** 2026-05-16
**Author:** rayanazhari (planner) + Claude orchestrator (Risk Auditor + Architect)
**Type:** Allocator-action memo. Justifies de-allocation of 5 live strategies without per-variant V3.6 audits.
**Status:** Pending allocator review.
**Lessons cited:** L17 (relative MC for long-only equity), L52 (hybrid sweep), L56 (refined — leveraged vs unleveraged).

---

## TL;DR — recommendation

**Bulk-retire 5 unleveraged `etf_trend` variants** at the next allocator-rebalance window:

| Variant | Live config | Action | Confidence | Replacement |
|---|---|---|---|---|
| `etf_trend_qqq` | slow_ma=200, ec=5, vol_target 0.25 | DE-ALLOCATE | high (L56 direct) | B&H QQQ (vol-targeted) |
| `etf_trend_iwb` | slow_ma=150, ec=3, binary | DE-ALLOCATE | high (L56 direct) | B&H IWB (vol-targeted) |
| `etf_trend_efa` | slow_ma=200, ec=5, binary | DE-ALLOCATE | high (L56 direct) | B&H EFA (vol-targeted) |
| `etf_trend_dbc` | slow_ma=75, ec=1, binary | DE-ALLOCATE | **HIGH (spot-check CONFIRMED 2026-05-16)** | B&H DBC OR rotate to GEM |
| `etf_trend_gld` | slow_ma=250, ec=5, binary | DE-ALLOCATE | **HIGH (spot-check CONFIRMED 2026-05-16)** | B&H GLD OR rotate to GEM |

**Phase 2 spot-checks completed 2026-05-16 — both confirm L56 generalises:**

- **DBC:** 4-cell audit on 20y data (5071 bars). Plateau spread 82% (FAILS L27). All cells TIER_UNCONFIRMED or SUSPECT. CI_lo NEGATIVE for every cell (-0.20 to -0.38). Noise axis = worst on every cell. **L56 confirmed. Bulk-retire confidence: HIGH.** See `.tmp/reports/etf_trend_dbc_spotcheck/result_log.md`.
- **GLD:** 4-cell audit on 21y data (5376 bars). Plateau spread 43% (FAILS L27). All cells SUSPECT or TIER_UNCONFIRMED. CI_lo NEGATIVE for every cell (-0.12 to -0.23). Rel-MC FAILS for every cell (median DD reduction ~0.93 vs B&H GLD). **L56 confirmed. Bulk-retire confidence: HIGH.** See `.tmp/reports/etf_trend_gld_spotcheck/result_log.md`.

Both spot-checks took ~5 minutes each via the generic [`research/etf_trend/run_etf_trend_binary_spotcheck.py`](../research/etf_trend/run_etf_trend_binary_spotcheck.py) harness (a one-time investment that now generalises to any future binary-sized etf_trend variant). With both medium-confidence variants confirmed, **all 5 unleveraged variants are now HIGH-confidence retire** — no further audits required.

**Capital freed by de-allocation rotates to:**
1. `bond_gold` (V3.6 promoted CONDITIONAL_WATCHPOINT — Wave A.1)
2. `etf_trend_tqqq` if migrated to V3.6 canonical `(150, 5)` (Wave A.2-confirm)
3. Cash buffer + GEM A1_ewma_hl40 (only confirmed-live)

## Why bulk action vs per-variant audits

The Wave A.2 + A.2-confirm audits (SPY 2026-05-16 RETIRED; TQQQ 2026-05-16 CONDITIONAL) established **L56** with a clean leveraged-vs-unleveraged split. Running 5 more identical audits on QQQ/IWB/EFA/DBC/GLD would consume ~5-10 hours of compute and produce predictable outcomes:

1. **L17 relative-MC failure is mechanistic, not asset-specific.** Random-block bootstrap shuffles crash bars; MA-crossover trend filters can't preempt randomly-located crashes any better than B&H. This applies to ANY long-only ETF.
2. **The unleveraged Sharpe edge is too small to absorb noise-mitigation.** SPY edge over B&H was +6%; with vol-target sizing, that's the upper bound. The other unleveraged equity ETFs will have similar or smaller edges. Exit_confirm_days=5 helps noise but costs ~5-15% of Sharpe → not enough margin to land CONDITIONAL like TQQQ did (where the +54% edge absorbed the cost).
3. **The 5-axis matrix verdict is predictable.** CI_lo OK, DSR OK, sanctuary OK, **rel-MC FAIL, noise FAIL or borderline** → TIER_UNCONFIRMED → RETIRED per §3 selection rule. Predictable enough to skip the audits per L62 (cost-benefit of confirmation audits — see catalogue).

## Confidence gradient

**HIGH confidence retire (L56 directly applies):**

- **`etf_trend_qqq`** — long-only equity ETF, same mechanism as SPY. QQQ Sharpe is ~10% higher than SPY historically; trend filter edge similar; L56 applies directly.
- **`etf_trend_iwb`** — Russell 1000 index ETF. Highly correlated with SPY (ρ ≈ 0.95). L56 applies directly.
- **`etf_trend_efa`** — Developed-markets ex-US. Long-only equity. Lower Sharpe than US equity historically; trend filter edge similar; L56 applies.

**MEDIUM confidence retire (asset-class differs; mechanism likely applies):**

- **`etf_trend_dbc`** — broad commodity basket (CL, NG, GC, BAL, etc.). Commodity drawdowns are more idiosyncratic (supply shocks) than equity crashes (deleveraging cascades). The bootstrap rel-MC test still applies — random shuffling destroys whatever predictability the trend filter has — but the Sharpe edge could be slightly different. **Recommend fast spot-check audit before final de-allocation.**
- **`etf_trend_gld`** — gold ETF. Gold has different drawdown structure (no 2008-style crash) and the bond_gold audit already established that gold has a real signal via IEF momentum. Trend filter on GLD specifically MAY add value differently than on equity. **Recommend fast spot-check audit before final de-allocation.**

## Allocator action plan

**Phase 1 (immediate, next allocator rebalance):**

1. De-allocate `etf_trend_qqq`, `etf_trend_iwb`, `etf_trend_efa` to 0 weight.
2. Flag as RETIRED in `titan/strategies/etf_trend/__init__.py` registry (with `# RETIRED 2026-05-16 — L56 (unleveraged eq-trend bulk-retire)` comment).
3. Free capital rotates to: bond_gold CONDITIONAL allocation + cash buffer.

**Phase 2 (within 2 weeks):**

4. Run fast spot-check audits on `etf_trend_dbc` and `etf_trend_gld` (use existing harness with symbol substitution — ~30min per audit including MC).
5. If spot-checks confirm RETIRED, complete bulk action.
6. If either gold/commodity audit produces a CONDITIONAL/DEPLOY, keep that variant and migrate to its canonical.

**Phase 3 (within 1 month):**

7. Migrate `etf_trend_tqqq` to V3.6 canonical `(slow_ma=150, exit_confirm_days=5)` per Wave A.2-confirm. This is the ONLY etf_trend variant that survives V3.6 deployment gates.

## Risk assessment

**Risk of bulk-retire being WRONG (false-positive retire):**

- Worst case: one of the 5 strategies actually adds value the audit would have caught. This would mean lost Sharpe ≈ +0.05-0.15 on a portion of the portfolio. Magnitude: ≤2% portfolio Sharpe drag — recoverable in the next allocator rebalance with a fresh audit.
- Mitigation: the Phase 2 spot-checks on DBC and GLD reduce false-positive risk for the medium-confidence variants. High-confidence retires (QQQ/IWB/EFA) are unlikely to be wrong given the SPY mechanism.

**Risk of KEEPING the strategies live (current state):**

- The V3.6 5-axis matrix is the deployment gate. Keeping a TIER_UNCONFIRMED strategy live violates the post-V2.0 cleanse principle: "no strategy ships with V3.6 deployment-gate failures."
- Operational risk: each live strategy adds maintenance burden, parity test surface, and IB connection overhead. Five RETIRED-but-deployed strategies is technical debt.
- Current allocation magnitude: per-strategy allocations are small (each ≤ 5% per V3.6 risk caps), so capital-at-risk impact is limited — but capital is misallocated relative to confidence levels.

## Allocator approval checklist

- [ ] Read this memo end-to-end.
- [ ] Confirm Phase 1 strategies (QQQ, IWB, EFA) are eligible for immediate de-allocation.
- [ ] Confirm Phase 2 spot-check audits for DBC and GLD are authorised.
- [ ] Confirm Phase 3 TQQQ migration to V3.6 canonical `(150, 5)` is authorised after the 6-month shadow.
- [ ] Sign off on the rotation: freed capital → bond_gold CONDITIONAL + cash buffer.

Sign-off here: ____________________   Date: __________

## Catalogue impact

- **L56 (refined)** already updated post-TQQQ.
- **No new lesson needed for the bulk-retire decision** — it's an *application* of L56 + risk-management principles, not a new methodology.
- If the Phase 2 spot-checks reveal DBC or GLD behaves differently, a new lesson may emerge about commodity/gold-specific trend filter behaviour. **Hold L57 reservation for this contingency.**

## Status

- This memo is the **deliverable** of Wave A.2-confirm. The L52 hybrid workflow has now run through 3 full audits (bond_gold PROMOTED, SPY RETIRED, TQQQ PROMOTED CONDITIONAL) and produced a bulk-action recommendation that previously would have required 7 individual audits.
- After allocator approval, the Phase 2 spot-checks for DBC and GLD are the next audit tasks. The GEM hybrid re-audit (sweep + plateau on EWMA halflife axis) is queued behind these.

---

**See also:**

- `.tmp/reports/etf_trend_spy_reaudit/findings.md` — SPY audit
- `.tmp/reports/etf_trend_tqqq_reaudit/findings.md` — TQQQ audit
- `directives/V3.6 Lessons Catalogue.md` — L56 (refined)
- `directives/V1-era Re-audit Sweep Roster 2026-05-16.md` — full re-audit roster
