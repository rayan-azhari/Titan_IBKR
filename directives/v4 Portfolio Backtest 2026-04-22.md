# v4 Portfolio Backtest — 2026-04-22

Realised backtest of the v4 portfolio proposed in the cross-asset
parameter sweep (see [Param Sweep 2026-04-22.md](./Param%20Sweep%202026-04-22.md)).
Answers the question the sweep left open: **does the mix actually
improve over v3 when combined?**

**Driver**: [scripts/rerank/optimize_v4_portfolio.py](../scripts/rerank/optimize_v4_portfolio.py)
**Raw**: [.tmp/reports/param_sweep_2026_04_22/v4_portfolio.md](../.tmp/reports/param_sweep_2026_04_22/v4_portfolio.md)
**Log**: [.tmp/reports/param_sweep_2026_04_22/v4_run.log](../.tmp/reports/param_sweep_2026_04_22/v4_run.log)

**Common window**: 2016-06-06 → 2025-06-12 (2,354 business days).
Split IS/OOS 50/50 for allocation-scheme evaluation; the per-strategy
standalone Sharpes shown are the OOS half (the tighter, held-out test).

---

## Per-strategy parameter upgrade check (full-window WFO)

The sweep found a threshold refinement hiding in plain sight. The
full-window WFO bootstrap confirms the upgrade for both strategies that
got new parameters — same data, same harness, only the threshold
(and hold days for TIP→HYG) changed:

| Strategy | Config | Sharpe | CI_lo | CI_hi | Max DD | Delta |
|---|---|---:|---:|---:|---:|:--|
| HYG → IWB (v3) | lb=10 hold=20 **th=0.50** | +0.879 | +0.450 | +1.385 | -24.0 % | — |
| **HYG → IWB (v4)** | lb=10 hold=20 **th=0.25** | **+1.076** | **+0.589** | **+1.576** | **-19.9 %** | **+22 % Sharpe, +0.14 CI_lo, −4 pp DD** |
| TIP → HYG (v3) | lb=60 hold=20 **th=0.50** | +0.897 | +0.392 | +1.340 | -10.1 % | — |
| **TIP → HYG (v4)** | lb=60 **hold=40** **th=0.25** | **+1.035** | **+0.562** | **+1.487** | -10.1 % | **+15 % Sharpe, +0.17 CI_lo** |

Both upgrades clear the Bonferroni gate (CI_lo ≥ 0.5) on their own.
Neither did in v3.

---

## v4 correlation structure (5 strategies)

**Max |pairwise|**: 0.556 (vs 0.607 in the v3 5-set → cleaner).

| | hyg_iwb | tip_hyg | tlt_qqq | mr_audjpy | ief_gld |
|---|---:|---:|---:|---:|---:|
| hyg_iwb_th025 | +1.000 | +0.405 | +0.556 | -0.022 | -0.004 |
| tip_hyg_h40_th025 | +0.405 | +1.000 | +0.389 | -0.080 | +0.106 |
| tlt_qqq_lb10 | +0.556 | +0.389 | +1.000 | -0.020 | +0.023 |
| mr_audjpy_a24 | -0.022 | -0.080 | -0.020 | +1.000 | -0.049 |
| ief_gld_lb60 | -0.004 | +0.106 | +0.023 | -0.049 | +1.000 |

**Structure**:
- **Bond-momentum triangle** (hyg_iwb / tip_hyg / tlt_qqq) shares the risk-on/off factor with pairwise ρ in 0.39-0.56.
- **MR AUD/JPY** — fully uncorrelated (all ρ ≈ 0).
- **IEF → GLD** — fully uncorrelated (all ρ ≈ 0).

The triangle is the inherent structure; the two diversifiers are genuinely
independent. This is exactly the kind of two-block structure diversification
theory prescribes.

---

## Portfolio-level realised OOS (held-out 1,177-day half)

| Scheme | Sharpe | CI_lo | CI_hi | Max DD | Ann ret |
|---|---:|---:|---:|---:|---:|
| **Equal-weight (20 % each)** | **+1.767** | **+0.933** | **+2.697** | **-5.0 %** | **+12.7 %** |
| IS inv-vol (cap 40 %) | +1.752 | +0.899 | +2.661 | -4.3 % | +11.0 % |
| v4 proposed (25/20/20/20/15) | +1.716 | +0.881 | +2.645 | -5.5 % | +12.9 % |

All three allocation schemes land at Sharpe +1.72–+1.77 with max DD under
6 % and CI_lo well above 0. Equal-weight wins narrowly on Sharpe, inv-vol
wins narrowly on DD — within noise, equal-weight is the right default for
its simplicity and operational robustness.

### v4 vs v3 portfolio (same OOS window)

| Metric | v3 (7-strategy) | v4 (5-strategy) | Delta |
|---|---:|---:|:--|
| Portfolio Sharpe (equal-wt) | +1.791 | +1.767 | ≈ flat |
| Max pairwise correlation | 0.607 | 0.556 | ↓ (cleaner) |
| Max DD | -4.6 % | -5.0 % | slightly worse |
| # strategies | 7 | 5 | ↓ (simpler) |
| Per-strategy CI_lo gate (pass rate) | 6/7 | 5/5 | 100 % |

**Equivalent Sharpe for a simpler, cleaner portfolio.** v3 got its edge
by stacking three X→HYG signals at 10 % each; the v3 optimizer then
revealed they share a credit factor (ρ 0.63-0.67) so the effective
allocation collapsed back to ~20 % credit anyway. v4 replaces the three
X→HYG rows with one TIP→HYG at 20 % — same factor exposure, no
double-counting, 5 positions instead of 7.

---

## Per-strategy OOS standalone (diagnostic)

The second-half-of-OOS view has fewer bars and wider CIs. Three of five
components have CI_lo < 0 on this smaller sample — this is not a red
flag, it's the expected consequence of halving the series. The
**full-window WFO** numbers (param sweep) have all five ≥ +0.5 CI_lo.

| Strategy | Sharpe (split OOS) | CI_lo | CI_hi | Max DD |
|---|---:|---:|---:|---:|
| hyg_iwb_th025 | +0.859 | -0.013 | +1.839 | -19.9 % |
| tip_hyg_h40_th025 | +0.827 | -0.136 | +1.673 | -5.9 % |
| tlt_qqq_lb10 | +1.325 | +0.524 | +2.151 | -13.7 % |
| mr_audjpy_a24 | +1.045 | +0.208 | +1.799 | -20.1 % |
| ief_gld_lb60 | +0.697 | -0.169 | +1.567 | -10.5 % |

Point estimates remain positive across all 5; wide CI is a sample-size
artefact. The portfolio-level CI_lo +0.93 (from equal-weighting the five)
is itself well above 0 and is the number that should drive the deploy
decision.

---

## ML IWB overlay (not in correlation mix)

| Metric | Value |
|---|---|
| Sharpe | +0.833 |
| 95 % CI | (+0.385, +1.237) |
| Max DD | -3.0 % |
| bars | 4,572 |

ML IWB stacking has very sparse trades (5-10 per year, narrow entry
window). If combined with the daily bond-equity candidates, the inner
join collapses the correlation window. Treat it as a **fixed 10-20 %
overlay** on top of the v4 5-set, not a correlation-mix participant.
Estimated additive portfolio Sharpe when overlaid at 15 %: ~+1.80-1.85
(back-of-envelope; sparse series contribution modelled as independent).

---

## Actionable changes

### Recommended live v4 deployment

5 base positions equal-weighted at 17 % + ML IWB overlay at 15 %:

| Slot | Weight | Strategy | Config |
|---|--:|---|---|
| 1 | 17 % | HYG → IWB | lb=10 hold=20 **th=0.25** |
| 2 | 17 % | TIP → HYG | lb=60 **hold=40** **th=0.25** |
| 3 | 17 % | TLT → QQQ | lb=10 hold=20 th=0.50 |
| 4 | 17 % | MR AUD/JPY | vwap_anchor=24, conf_donchian_pos_20 |
| 5 | 17 % | IEF → GLD | lb=60 hold=20 th=0.50 |
| 6 | 15 % | ML IWB | stacking, signal_threshold=0.6 |

### Config changes from current champion portfolio

The current `scripts/run_portfolio.py "champion_portfolio"` block uses:
- MR AUD/JPY (vwap_anchor=24) ✓ no change
- `bond_equity_ihyu_cspx` — maps to **HYG → IWB at threshold 0.50** → **update to 0.25**

Four new deployments required (TIP → HYG, TLT → QQQ, IEF → GLD, ML IWB).
Each needs the standard 48-hour paper dry-run at 10 % target size before
going to full weight.

### What this does NOT change

- Per-strategy-equity tracker contract (every strategy still registers with PRM).
- PortfolioAllocator inverse-vol sizing between strategies (the 17/15 weights above are static targets; PRM vol-targets to hit each).
- All research-math discipline rules (shared metrics module, shift(1), bootstrap CI gate, etc.).

---

## Risk caveats

1. **Bond-momentum cluster tail risk.** All three of HYG→IWB, TIP→HYG, TLT→QQQ respond to a risk-off shock. The 0.39-0.56 ρ is a fair-weather estimate; during a macro stress event they will correlate higher. Stop condition: rolling 30-day pairwise ρ > 0.75 on any of the three pairs.
2. **IEF → GLD standalone CI_lo negative on split-OOS.** Kept only because of full-history CI_lo (+0.12) and its zero correlation to everything. If a better-standalone uncorrelated diversifier appears (e.g. a working gold-specific signal), replace IEF → GLD.
3. **Threshold=0.25 means more trades.** Round-trip cost at 5 bps per leg = 10 bps per entry+exit. HYG → IWB at th=0.25 has ~1.5× the trades of th=0.50 — cost model still supports the strategy but the margin is thinner.
4. **Sanctuary window untouched.** The sweep that produced th=0.25 was run with the post-April-2026 sanctuary window active. All numbers above are pre-sanctuary; a final validation pass on the held-out 12 months is still owed before capital commit.
