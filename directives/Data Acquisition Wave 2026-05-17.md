# Data Acquisition Wave 2026-05-17 — B2 / I1 / B5

**Date:** 2026-05-17
**Persona:** Operator
**Status:** Complete

## Objective

Close the data gaps blocking three backlog audits (B2 Carver EWMAC, I1 HMM regime, B5 SPY/QQQ/IWM intraday momentum). Survey sources (IBKR, IG, Databento, yfinance) and acquire whatever each strategy needs at the cheapest available depth.

## Inventory of existing sources

| Source | Coverage | Cost | Used here |
|---|---|---|---|
| **IBKR** (Docker paper, port 4004) | M1/M5 intraday, ≈2-year cap on paper, 50req/10min | Live paper subscription | ✅ B5 |
| **Databento** (Historical SDK, GLBX.MDP3 + ICE) | Continuous-contract futures, M5 + daily | Per-byte API; CME continuous symbology starts 2017-06 | ✅ B2 |
| **IG** (`.env.ig`) | FX + indices intraday & daily | Demo | not needed this wave |
| **yfinance** | ETF / index daily (no intraday > 60d) | Free | ✅ I1 (already cached) |

## Gap analysis (before this wave)

| Backlog | Need | Pre-wave state |
|---|---|---|
| **B2** Carver EWMAC | 8 futures including ZN/ZB/6E/6J | Had ES/NQ/CL/BZ/HG/SI/GC only — missing 4 |
| **I1** HMM regime panel | VIX, term-spread, credit-spread, RV20, SPY-vs-200SMA, DXY, DD-velocity | All inputs present in `data/` |
| **B5** Intraday momentum (SPY focus) | SPY/QQQ/IWM M5, ≥1y | Had SPY_H1 + QQQ_H1 only, no M5 ETF data |

## What we did

### I1 — built from existing parquets (no acquisition needed)

`research/exploration/build_i1_regime_panel.py` constructed a **7-feature regime panel** (3,945 trading days, 2010-07-26 → 2026-04-02) from already-cached daily files:

| Feature | Source | Spec |
|---|---|---|
| `vix_z` | VIX_D.parquet | 252-day rolling z-score of close |
| `term_spread_z` | TLT/IEF | z-score of (TLT − IEF) yield proxy |
| `credit_spread_z` | HYG/IEF | z-score of (HYG − IEF) return spread |
| `rv20_z` | SPY_D | z-score of 20-day realised vol |
| `spy_above_sma200` | SPY_D | binary regime indicator |
| `dxy_z` | DXY_D | 252-day rolling z-score |
| `dd_velocity_21` | SPY_D | drawdown velocity over 21d (slope) |

Output: `data/i1_regime_panel.parquet` — 264 KB.

### B5 — IBKR M5 over 2y, 3 ETFs

`scripts/download_b5_m5_data.py` (CLIENT_ID=96, 28-day chunks, 1.5s pacing) pulled **2 years of M5 RTH data** for SPY/QQQ/IWM via IBKR paper account.

**Result:** 39,408 bars per ticker, 2024-05-07 → 2026-05-15.

Initial run failed with `NoneType not callable` — `self.error: str | None = None` shadowed the inherited `EWrapper.error()` method, so IBKR couldn't dispatch error events. Fixed by renaming the attribute to `self.error_msg`.

### B2 — Databento continuous futures, 4 new roots

`scripts/download_b2_futures_databento.py` pulled **front+next month daily OHLCV** for the missing Carver basket members:

| Root | Name | M1 bars | M2 bars |
|---|---|---|---|
| ZN | 10y US Treasury Note | 2207 | 2207 |
| ZB | 30y US Treasury Bond | 2207 | 2207 |
| 6E | Euro FX | 2704 | 2702 |
| 6J | Japanese Yen FX | 2665 | 2659 |

All cover **2017-06-01 → 2026-05-15** (CME GLBX continuous symbology earliest date). Databento flagged a handful of dataset-quality degraded days (2017-11-13, 2018-10-21, 2019-01-15) — informational, not blocking.

## Audit follow-on completed in this wave

### B5 — VERDICT: **RETIRE**

`research/exploration/audit_b5_spy_intraday.py` ran the Gao-Han-Li-Zhou first-30m → last-30m intraday-momentum test on the new SPY/QQQ/IWM data:

| Ticker | n_trades | win_rate | per-trade Sharpe |
|---|---|---|---|
| SPY | 502 | 44.4% | **−1.07** |
| QQQ | 500 | 45.6% | **−0.87** |
| IWM | 500 | 48.6% | **−0.62** |

Panel median **−0.87**, 0% positive. L21 PASS on all three (causality holds). Signal is materially **reversed** in 2024-26 — consistent with academic post-2014 decay finding plus possible HFT crowding effect. Retire.

## Files added / modified

| File | Status | Purpose |
|---|---|---|
| `research/exploration/build_i1_regime_panel.py` | NEW | I1 7-feature panel constructor |
| `data/i1_regime_panel.parquet` | NEW | 3,945 × 7 |
| `scripts/download_b5_m5_data.py` | NEW (+ NoneType fix) | IBKR M5 ETF puller |
| `data/{SPY,QQQ,IWM}_M5.parquet` | NEW | 2y M5, 39,408 bars each |
| `scripts/download_b2_futures_databento.py` | NEW | Databento ZN/ZB/6E/6J puller |
| `data/{ZN,ZB,6E,6J}_{M1,M2}_D.parquet` | NEW | 2017-06 → 2026-05 continuous |
| `research/exploration/audit_b5_spy_intraday.py` | NEW | SPY-focused B5 re-audit |

## Next backlog items unblocked

- **B2 Carver EWMAC ensemble** — all 8 futures now present (ES/NQ/CL/BZ/HG/SI/GC/ZN/ZB/6E/6J). Ready to audit under framework primitives.
- **I1 HMM regime + EWMAC gate** — panel built; HMM training next.

B5 closed (RETIRE).
