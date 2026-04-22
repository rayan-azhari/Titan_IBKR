# Candidate Portfolio Optimization — 2026-04-21

**Harness commit**: `7295b8c` (post-rerank)
**Common OOS window**: 2016-06-06 → 2025-06-12 (2,354 business days)
**Driver**: [scripts/rerank/optimize_candidate_portfolio.py](../scripts/rerank/optimize_candidate_portfolio.py)
**Raw output**: [.tmp/reports/rerank_2026_04_21/candidate_portfolio.md](../.tmp/reports/rerank_2026_04_21/candidate_portfolio.md)

> [!NOTE]
> **Superseded by the autonomous loop v3 analysis** — the subsequent run
> in [Autonomous Loop 2026-04-21.md](./Autonomous%20Loop%202026-04-21.md)
> and its portfolio optimizer
> ([scripts/rerank/optimize_v3_portfolio.py](../scripts/rerank/optimize_v3_portfolio.py))
> discovered a cross-bond signal family (TIP→HYG, LQD→HYG, TLT→HYG)
> with lower drawdowns than the bond→equity set in this doc. The v3
> correlation check then revealed those three cross-bond signals share
> a **common credit factor with 0.63-0.67 pairwise correlation** on
> the common 2016-2025 OOS window — holding all three would double-count
> the credit exposure. TIP→HYG is the single best representative
> (highest Sharpe +0.913, highest CI_lo +0.432, lowest DD -10.1%). Use
> the v3 recommendation for live deployment decisions.

Pulls `stitched_returns` from each candidate's WFO run on the corrected
harness, aligns on the common date range, and evaluates correlation +
portfolio metrics under two allocation schemes.

---

## Recommended portfolio — 5 strategies, equal-weight

| Strategy | Weight | Role | Standalone OOS Sharpe | CI lo |
|---|---:|---|---:|---:|
| MR AUD/JPY conf_donchian_pos_20/conservative | 20 % | FX MR factor | +1.045 | +0.208 |
| TLT→QQQ lb=10 | 20 % | Duration factor (NEW) | +1.325 | +0.524 |
| TIP→QQQ lb=60 | 20 % | Inflation-expectations factor (NEW) | +1.045 | +0.197 |
| HYG→IWB lb=10 | 20 % | Credit factor (existing champion) | +1.135 | +0.309 |
| IEF→GLD lb=60 | 20 % | Gold-bond-momentum factor (existing champion) | +0.697 | -0.169 |

### Portfolio-level OOS performance

| Metric | Equal-weight | IS inv-vol (capped 60 %) |
|---|---:|---:|
| **Sharpe** | **+1.893** | **+1.916** |
| 95 % CI | (+1.106, +2.743) | (+1.117, +2.798) |
| Max DD | -7.4 % | -6.5 % |
| Annualized return | +15.0 % | +14.3 % |
| Random Dirichlet p05/p50/p95 | 1.34 / 1.66 / 1.86 | — |

Both schemes put portfolio Sharpe above the single-best standalone
(TLT→QQQ at +1.325), confirming diversification is earning real Sharpe
— not just splitting one signal into five buckets.

---

## Why 5 strategies, not 6

The original 6-candidate set included **HYG→QQQ lb=10** (from the
re-rank top-5). Its correlation with HYG→IWB is **0.928** — the same
signal on two near-identical targets. Keeping both is redundant
exposure to one factor.

### Full 6-set vs deduplicated 5-set

| Metric | 6-strategy (with HYG→QQQ) | 5-strategy (no HYG→QQQ) | Delta |
|---|---:|---:|---:|
| Max pairwise \|ρ\| | 0.928 | **0.607** | -0.32 |
| Equal-weight Sharpe | +1.768 | **+1.893** | **+0.12** |
| Equal-weight Max DD | -8.5 % | **-7.4 %** | +1.1 pp |
| IS inv-vol Sharpe | +1.816 | **+1.916** | **+0.10** |

Dropping the duplicate produces a strictly better portfolio on every
metric.

### Correlation matrix (5-strategy set)

|   | mr_audjpy | tlt_qqq | tip_qqq | hyg_iwb | ief_gld |
|---|---:|---:|---:|---:|---:|
| mr_audjpy   | +1.000 | -0.020 | -0.047 | -0.048 | -0.049 |
| tlt_qqq     | -0.020 | +1.000 | +0.607 | +0.562 | +0.023 |
| tip_qqq     | -0.047 | +0.607 | +1.000 | +0.449 | +0.005 |
| hyg_iwb     | -0.048 | +0.562 | +0.449 | +1.000 | -0.019 |
| ief_gld     | -0.049 | +0.023 | +0.005 | -0.019 | +1.000 |

**Structure**: two genuinely uncorrelated blocks (AUD/JPY MR and
IEF→GLD), plus a bond-momentum trio (TLT, TIP, HYG) with moderate
internal correlation (0.45–0.61). This is the expected structure — bond
signals share a common risk-on/off factor. The 0.45–0.61 range is low
enough to still earn diversification premium (vs the 0.93 duplicate
that was hurting the 6-set).

---

## Comparison to pre-remediation portfolio

The old portfolio (pre-audit) was AUD/JPY (40 %) + IWB ML (25 %) +
HYG→IWB (30 %) + AUD/USD (5 %), with reported combined Sharpe around
**+5.14** (inflated by the math bugs).

After remediation:

| Metric | Old (inflated) | New (corrected) | Interpretation |
|---|---:|---:|---|
| Portfolio Sharpe | +5.14 | +1.91 | Honest number, gate-passed |
| Max DD | not reported | -6.5 % | |
| n_strategies | 4 | 5 | |
| Includes deprecated AUD/USD | Yes (5 %) | No | |
| Includes new TLT→QQQ | No | Yes (20 %) | |
| Includes new TIP→QQQ | No | Yes (20 %) | |

The +1.91 corrected Sharpe is the real trading edge. Any comparison to
the +5.14 is meaningless — those were different calculations of
different things.

---

## Risk caveats

1. **2-fold artefact not fully eliminated** — TIP→QQQ has 40 folds but
   its standalone CI_lo is only +0.197 (borderline). Most of its
   portfolio contribution comes from low correlation, not strong
   standalone alpha. If its OOS edge erodes further, the portfolio
   Sharpe falls closer to the single best (~+1.3) not the +1.9 shown.

2. **Bond-momentum cluster risk** — TLT / TIP / HYG all respond to the
   same macro risk-off shock. In a tail event they will correlate
   higher than their benign-period correlations suggest. The 0.6
   average is a fair-weather estimate.

3. **AUD/JPY carry regime dependency** — this strategy survived
   remediation but relies on Asian-session liquidity patterns plus
   carry-pair mean-reversion. A structural shift in JPY policy (BoJ
   tightening) could break the regime. Monitor live.

4. **IEF→GLD is borderline** — standalone CI_lo is -0.17 (fails the
   gate on its own). Keeps its 20 % weight in this portfolio only
   because it's fully uncorrelated with everything else; if another
   uncorrelated diversifier appears (e.g. gold-specific signals), IEF→GLD
   should be replaced.

---

## Deployment path

1. **Do NOT deploy immediately** — the new TLT→QQQ and TIP→QQQ
   strategies are research-grade. Each needs a 48-hour paper dry-run
   at 10 % of target size to confirm the live sizing math agrees with
   the WFO.

2. **Order of rollout** (least risky first):
   - Week 1: Deploy TLT→QQQ as the first new addition; verify allocator
     weight and scale_factor behave correctly with 5-strategy PRM.
   - Week 2: Add TIP→QQQ.
   - Week 3: Resize old champions to target weights (20/20/20/20/20).
   - Week 4: Remove AUD/USD MR.

3. **Stop conditions** — abort rollout if any of:
   - Live TLT→QQQ 20-day rolling Sharpe goes negative before the
     population WFO mean (strategy not replicating in live).
   - Portfolio scale_factor pegs at 2.0 for >3 consecutive days
     (vol-targeting saturating).
   - Pairwise correlation of live returns exceeds the OOS-computed
     0.61 by more than 0.2 (bond-momentum cluster tightening).

---

## Artefacts

- [scripts/rerank/optimize_candidate_portfolio.py](../scripts/rerank/optimize_candidate_portfolio.py)
  — driver (re-runnable).
- `.tmp/reports/rerank_2026_04_21/candidate_portfolio.md` — raw markdown output.
- `.tmp/reports/rerank_2026_04_21/candidate_portfolio.log` — full stdout log.
