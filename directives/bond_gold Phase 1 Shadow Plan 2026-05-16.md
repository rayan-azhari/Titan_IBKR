# bond_gold Phase 1 Shadow Plan + Migration Runbook

**Author:** rayanazhari (planner) + Claude orchestrator (Operator)
**Date committed:** 2026-05-16
**Type:** Operations memo. Defines the 6-month paper-shadow comparison and the final-cutover criteria for the V3.6 bond_gold canonical.
**Predecessors:** `directives/Pre-Reg bond_gold Re-audit 2026-05-16.md`, `.tmp/reports/bond_gold_reaudit/findings.md`, `docs/strategies/bond-gold.md`.

---

## Context

The Wave A.1 audit (2026-05-16) PROMOTED `(lookback=120, threshold=0.50)` to CONDITIONAL_WATCHPOINT. **The live config remains `(60, 0.50)` per the findings memo recommendation** — bond_gold needs a more conservative migration than GEM J5 because:

1. **Sanctuary is lucky-flagged (L55).** All cells with threshold ≤ 0.50 have sanctuary percentile=1.00 (gold rally 2024-04 → 2026-04). The OOS Sharpe of +0.66 is the deployment number; the sanctuary +1.67 is not.
2. **The CI_lo improvement is modest** (+0.075 → +0.152) compared to GEM J5's (+0.24 → +0.51). Less margin for surprise.
3. **The 6-month forward window should cover a non-gold-rally period** to confirm the V3.6 plateau holds outside the favourable regime.

**Phase 0 (sidecar config) is DONE** — `config/bond_gold_v36.toml` exists alongside the live `config/bond_gold.toml`.

---

## Phase 1 — 6-month paper-shadow comparison (2026-05-16 → 2026-11-13)

### Setup

Run a SECOND `bond_gold` paper-trading instance alongside the V1-era live one. Both share the IBKR Gateway (port 4004) but use separate client IDs to keep their orders + positions distinct.

| Instance | Config | IBKR client ID | Initial equity | Trades on |
|---|---|---|---|---|
| Live V1-era | `config/bond_gold.toml` | (existing) | 30 000 USD | Real paper account |
| **Shadow V3.6** | `config/bond_gold_v36.toml` | new (suggest 24, avoid GEM=21, kill-switch=98) | 30 000 USD | Real paper account |

The shadow instance places real paper-trading orders. Both instances run simultaneously; the same paper account holds both sets of positions. Capital comes from the `TITAN_PORTFOLIO_USD_EQUITY` envelope; PRM allocates to each.

### Deployment procedure

```bash
# 1. Add client ID env var
echo "IBKR_CLIENT_ID_BOND_GOLD_V36=24" >> .env.docker

# 2. Add the shadow strategy to the champion-portfolio runner registry
#    (see scripts/run_portfolio.py for the registration pattern)
#    Use config_path="config/bond_gold_v36.toml"

# 3. Restart the container so the shadow strategy boots
docker compose restart titan-portfolio

# 4. Verify both instances booted
docker compose logs --since 2m titan-portfolio | grep "bond_gold.*Strategy attached"
# Expected: 2 lines, one per instance.
```

### Daily monitoring (operator)

| Check | Method | Action if abnormal |
|---|---|---|
| Both instances up | `docker compose logs --since 24h \| grep "bond_gold.*Strategy"` | If shadow exits with error, debug + restart |
| No conflicts | Verify the IBKR account shows GLD positions sized for BOTH instances (not one being overwritten) | If positions disappear, check client-ID setup |
| Realised PnL comparison | Track each instance's NLV contribution | Live config vs shadow config should diverge gradually as their (lookback=60 vs 120) signals disagree |

### Weekly checks

| Check | Method | What we're looking for |
|---|---|---|
| Signal-disagreement frequency | Log lines for both `bond_gold` instances showing `z, threshold, signal` | At ~50% of weeks the signals will disagree; that's expected given the lookback difference |
| Forward Sharpe diff | Compute realised Sharpe over the rolling 8-week window for both | Shadow should match OOS Sharpe +0.66 within +/- 0.20; if shadow Sharpe < 0 for 4+ consecutive weeks, investigate |

---

## Phase 2 — Fresh sanctuary re-test (at 6-month mark, 2026-11-13)

### Re-audit procedure

Run `research/cross_asset/run_bond_gold_reaudit.py` AS-IS (no code changes; the harness re-loads fresh data). The sanctuary window will now be 30 months (2024-04-02 → 2026-11-13), covering both:
- The 2024-25 gold rally (current sanctuary, lucky-flagged)
- The post-rally period (2026-05 → 2026-11, 6 months of fresh data)

If the post-rally 6mo period happens to be unfavourable for gold (e.g., bond yields rising, gold prices flat or down), this provides the regime-diversity test we need.

### Decision rule (pre-committed)

After the 6-month re-audit, evaluate:

| Criterion | Pass for Phase 2 cutover | Action if fail |
|---|---|---|
| V3.6 canonical (lookback=120, threshold=0.50) verdict | DEPLOY or CONDITIONAL with CI_lo > 0 on the extended sanctuary | If sanctuary `lucky_flag=True` STILL, extend Phase 1 another 3 months (defer Phase 2 to 2027-02) |
| Shadow realised Sharpe (live 6-month forward) | ≥ +0.20 | If realised < 0, abort migration; flag as RETIRED |
| Live V1-era realised Sharpe | informational only | (V1 config remains, just not migrated) |

### If all criteria pass → Phase 2 cutover

```bash
# 1. Stop the container
docker compose stop titan-portfolio

# 2. Replace live config with V3.6 canonical (sidecar) IN PLACE
cp config/bond_gold_v36.toml config/bond_gold.toml

# 3. Disable the separate shadow instance (revert the registry change)

# 4. Restart with the new live config
docker compose start titan-portfolio

# 5. Verify J5-style position adoption
docker compose logs --since 2m titan-portfolio | grep "bond_gold.*Strategy\|RECONCILE GLD"
# Expected: positions adopted, no double-fills.
```

### Phase 3 (after Phase 2 cutover, by 2027-03-13)

- Re-audit at 4-week post-cutover mark to confirm V3.6 verdict holds with NEW data (this is the analog of GEM's J6 audit).
- Update `docs/strategies/bond-gold.md` to v2.0 noting "V3.6 LIVE since YYYY-MM-DD".

---

## Risks + caveats

1. **Two instances may compete for capital.** If PRM treats both as separate strategies, each gets its own capital allocation; freed capital from de-allocating other strategies (e.g., etf_trend variants) should flow to the V3.6 shadow.
2. **Shadow consumes additional paper-account positions.** Account NLV will reflect both instances' GLD positions; risk metrics are doubled vs single-strategy mode. This is fine for paper trading.
3. **L55 caveat may still bind at 2026-11.** If gold continues to rally for 6 more months, the new sanctuary will ALSO be lucky-flagged. In that case Phase 2 is deferred further until a non-favourable regime appears.
4. **Rollback path stays clean** because the V3.6 config is in a separate file. Until Phase 2 fires, `config/bond_gold.toml` is untouched.

---

## Status

- **Phase 0 (sidecar config):** DONE 2026-05-16 — `config/bond_gold_v36.toml`.
- **Phase 1 (6-month shadow):** **pending deployment**. Operator action required: register the shadow instance in the champion-portfolio runner + restart container.
- **Phase 2 (Fresh sanctuary re-test + cutover):** target 2026-11-13.
- **Phase 3 (4-week confirmation re-audit):** target 2026-12-11.
