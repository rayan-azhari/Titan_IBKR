# Cost Model Audit — 2026-05-11

Live trading vs modeled costs comparison. Baseline: 8 fills on the IBKR
paper account between 2026-05-09 and 2026-05-11 across VUSD and EIMI
(both `bond_equity_*` strategies). `mr_audjpy` had not yet traded.

## Summary of findings

| # | Finding | Severity |
|---|---|---|
| 1 | IBKR paper account charges flat **$4 USD per fill**. At the current live $3,427 USD per-strategy notional this is **117 bps per fill** — 23× the 5 bps modeled. | High |
| 2 | Spread cost is paid implicitly via bid/ask and **never appears in the commission report**. The model's 5 bps was meant to cover BOTH spread + commission together. Real all-in: VUSD ≈ 13 bps, EIMI ≈ 45 bps per round-trip. | High |
| 3 | `bond_equity` WFO charges `spread_bps=5.0` per direction-change. Round-trip is 10 bps modeled vs 25–30 bps real on VUSD at $17k notional. | Medium |
| 4 | Same-day phantom round-trip observed (May 11): VUSD bought 09:00 → sold 14:30 → re-bought 14:30. Root cause = rehydration bug (see separate note). $8 in pure-waste commissions per restart. | High |
| 5 | No FX conversion modeled. Account base is GBP, instruments USD. IBKR autoFX = ~0.2 bps. | Low |
| 6 | mr_audjpy uses EUR/USD-tier defaults from [titan/models/spread.py](../titan/models/spread.py) (1.2–3.5 pips). Real AUD/JPY in-session is 0.5–1.5 pips, off-session 5–10 pips. mr_audjpy's 07:00–12:00 UTC filter keeps it in tight spreads → model is conservative for in-session AUD/JPY but no live data yet to confirm. | Low |
| 7 | No financing / borrowing / stamp duty observed. Long-only + UCITS ETFs → match between model and reality. | None |

## Realistic-cost WFO results

Re-ran all 3 deployed `bond_equity` strategies under
`max($4 / notional × 10000, per_share_bps) + spread_bps_per_side`.
Per-instrument spreads: CSPX/VUSD = 2.5 bps/side, EIMI = 6.0 bps/side.

| Strategy | Documented (5 bps flat) | Live $3,427 | $10k | $25k | $50k |
|---|---|---|---|---|---|
| IHYU→CSPX | Sharpe +0.68 (CI +0.03) | +0.85 (CI +0.23) | +0.93 (+0.31) | +0.95 (+0.34) | +0.96 (+0.34) |
| IHYG→VUSD | Sharpe +1.16 (CI +0.47) | +0.84 (CI +0.15) | +1.12 (+0.43) | +1.21 (+0.52) | +1.24 (+0.55) |
| IHYG→EIMI | Sharpe +0.97 (CI +0.23) | +0.71 (CI +0.11) | +0.96 (+0.36) | +1.03 (+0.43) | +1.06 (+0.46) |

**All 3 strategies pass the deployment gate (CI lower bound > 0) at every notional level, including the live $3,427.**

Cost as a % of equity per year:
- $3.4k: ~5–6% per year eaten by costs
- $10k: ~2% per year
- $50k: <0.5% per year

## Action items (deferred — flagged for follow-up)

| Priority | Action | File |
|---|---|---|
| High | Replace `spread_bps = 5.0` flat default in `run_bond_equity_wfo.py` with `realistic_cost_bps()` so all future WFOs are calibrated by default. | [research/cross_asset/run_bond_equity_wfo.py:48](../research/cross_asset/run_bond_equity_wfo.py#L48) |
| High | Set min strategy notional to **$10k** (or drop active count from 4 → 2 to redistribute the existing $10k GBP NLV). The $3,427 auto-allocation sits below the cost-Sharpe-elasticity asymptote. | [scripts/run_portfolio.py](../scripts/run_portfolio.py) |
| Medium | Re-run the WFO with `--min-commission 1.0` to quantify the IBKR Pro tier gain. Floor cost drops from 117 bps to 29 bps at $3,427 notional. | [research/cross_asset/run_bond_equity_wfo_realistic.py](../research/cross_asset/run_bond_equity_wfo_realistic.py) |
| Medium | Add per-pair AUD/JPY entry to [titan/models/spread.py](../titan/models/spread.py) instead of falling through to EUR/USD defaults. Suggested: 0.5/1.5/3.0 pips for london/ny/tokyo, 8.0 for off-hours. | [titan/models/spread.py:46](../titan/models/spread.py#L46) |
| Low | Add `config/spread.toml` so the spread table is operator-tunable without code changes. Currently the file is missing and the code falls back to defaults. | new |

## Reference

- [research/cross_asset/run_bond_equity_wfo_realistic.py](../research/cross_asset/run_bond_equity_wfo_realistic.py) — reusable realistic-cost WFO
- [.tmp/reports/bond_equity_realistic_costs.csv](../.tmp/reports/bond_equity_realistic_costs.csv) — full sweep table
