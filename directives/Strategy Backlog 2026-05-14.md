# Strategy Backlog — V2.0 menu

**Date:** 2026-05-14
**Source:** synthesis of `resources/` (Carver, Antonacci, Bastien, Varma, López de Prado, ML_Regime_Strategy/, Pyramiding, Mean Reversion best practice, Mixed) + literature scan (Gao et al., Daniel-Moskowitz, Bakshi-Gao-Rossi, Frazzini-Pedersen, Blitz, Bali, Lou-Polk-Skouras, Lucca-Moench, Kang-Rouwenhorst-Tang, Avellaneda-Lee, Krauss-Stübinger, Gu-Kelly-Xiu, López de Prado NCO, Bogousslavsky-Muravyev, etc.)

## Purpose

A menu of novel strategy ideas not yet implemented in `titan/strategies/`. Every entry has a class mapping into the V2.0 typology + a paper reference where it exists. **No entry below is a verdict — it's an idea queued for framework-grade audit.** Every strategy follows the standard recipe: pre-register a directive § 1–3 before any data is examined, run the audit, append the § 4 result log, append any new lesson to V3.6 Lessons Catalogue.

## Conventions

- "[NEW]" = added from the 2026-05-14 literature scan (post-Carver/Antonacci/Bastien sources).
- "Effort" is rough person-days assuming framework primitives already exist.
- Strategies needing data we don't currently have are flagged "**DATA**" in the Caveats column.

## A. Cross-sectional equity alpha (currently uncovered)

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| A1 | **[NEW] Residual momentum** | CROSS_ASSET_MOMENTUM | Blitz, Huij & Martens, *JEmpF* 2011; Robeco 2022 update | 4d | Needs FF3 factor returns (Ken French library) |
| A2 | **[NEW] Betting-Against-Beta** (with Novy-Marx micro-cap drop) | CROSS_ASSET_MOMENTUM | Frazzini & Pedersen, *JFE* 2014; Novy-Marx & Velikov, *JFE* 2022 | 5d | Decayed since 2014, controversial |
| A3 | **[NEW] Lottery / MAX reversal** | DAILY_MEAN_REVERSION | Bali, Cakici & Whitelaw, *JFE* 2011 | 3d | Short-locate frictions on top decile |
| A4 | **[NEW] 52-week-high anomaly** | CROSS_ASSET_MOMENTUM | George & Hwang, *JF* 2004; Hong-Jordan-Liu SSRN 3625480 2022 | 2d | — |
| A5 | Sector rotation MR with Hanna aggregator | DAILY_MEAN_REVERSION | `Mean reversion best practice.md` | 3d | — |

## B. Trend-following / momentum

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| B1 | **GEM dual-momentum macro sleeve** | CROSS_ASSET_MOMENTUM | Antonacci | 1d | **DOABLE NOW** — SPY/EFA/IEF parquets present |
| B2 | Carver multi-speed EWMAC futures ensemble | DAILY_TREND | Carver, *Systematic Trend Following* | 5d | ✅ DONE 2026-05-17 (PARTIALLY REVIVED via I1v2). B2/B2b/B2c/B2d/B2e all closed (L48/L49/L51/L69). I1v2 multi-feature HMM regime gate on B2e universe produced **C6_smoothed DEPLOY** (Sharpe +0.52, CI_lo +0.049, noise=best). Recommended shadow port `titan/strategies/ewmac_regime/`. |
| B3 | Donchian + pyramiding Turtle-plus | DAILY_TREND | Turtle.md + Strategic Pyramiding doc | 3d | **DATA**: same as B2 |
| B4 | TSMOM 10-15 futures | DAILY_TREND | Moskowitz-Ooi-Pedersen 2012 | 4d | ✅ DONE (RETIRED on 24-commodity yfinance; sign of L37 confirmed but magnitude +0.16 << MOP's +1.0). Plateau FAIL. See L39, L40 — fixable with multi-asset universe + IBKR per-contract roll-aware data. |
| B5 | **[NEW] Intraday momentum** (first-30m → last-30m) | INTRADAY_BREAKOUT | Gao, Han, Li & Zhou, *JFE* 2018 | 2d | ✅ DONE 2026-05-17 (RETIRED: SPY/QQQ/IWM panel median Sharpe -0.87, 0% positive on 2y IBKR M5, signal REVERSED — matches academic post-2014 decay) |
| B6 | **[NEW] Momentum-crash hedge** (Daniel-Moskowitz dynamic scaling) | META_LABELING | Daniel & Moskowitz, *JFE* 2016 | 2d | Specification-sensitive |

## C. Mean reversion / pairs

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| C1 | USD/CAD ↔ WTI cointegration | PAIRS | Malik_strategy.md | 2d | **DATA**: needs USD/CAD spot |
| C2 | **[NEW] PCA-residual stat-arb** | PAIRS (basket) | Avellaneda & Lee, *Quant Finance* 2010; Yeo-Papanicolaou 2017 | 5d | Needs top-500 US stocks daily |
| C3 | **[NEW] Copula-based pairs** | PAIRS | Krauss & Stübinger, *Applied Economics* 2017 | 3d | — |
| C4 | **[NEW] Kalman-filter dynamic-β pairs** | PAIRS | Triantafyllopoulos-Montana 2011 | 3d | — |

## D. Carry (currently only AUD/JPY)

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| D1 | G10 FX carry basket with VIX risk-off kill | CARRY | Malik + standard FX | 4d | **DATA**: G10 spot + rates |
| D2 | **[NEW] Commodity futures carry** | CARRY | Bakshi, Gao & Rossi, *Management Science* 2019 | 5d | **DATA**: 24 commodity futures front + back month for basis |
| D3 | **[NEW] Equity-index carry** | CARRY | Koijen-Moskowitz-Pedersen-Vrugt, *JFE* 2018 | 6d | **DATA**: dividend futures |
| D4 | **[NEW] Credit carry — HY/IG spread** | CARRY / PAIRS | Israel-Palhares-Richardson, *JoIM* 2018 (AQR) | 2d | HYG/LQD ETFs — IBKR-implementable |
| D5 | **[NEW] VIX term-structure carry** | CARRY (vol) | Eraker & Wu, *JFE* 2017 | 4d | **DATA**: VX1/VX2 futures or VXX/VXZ ETFs |

## E. Volatility ecosystem

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| E1 | VRP capture via SPY/VIXY | DAILY_MEAN_REVERSION | IC + standard literature | 2d | — |
| E2 | SKEW arbitrage | DAILY_MEAN_REVERSION | — | 3d | low priority |
| E3 | **[NEW] Dealer-gamma (GEX) regime gate** | META_LABELING / INTRADAY_MICROSTRUCTURE | Barbon & Buraschi SSRN 3725454 2021 | 4d | **DATA**: CBOE OI feed (~$200/mo) |
| E4 | **[NEW] Dispersion / correlation premium** | new class DISPERSION_CARRY | Driessen, Maenhout & Vilkov, *JF* 2009 | — | **SKIP**: needs single-name option chain |

## F. Cross-asset macro / regime

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| F1 | Risk-off bond bid (VIX + HYG/IEF) | CROSS_ASSET_MOMENTUM | — | 2d | — |
| F2 | **[NEW] CFTC CoT positioning extremes** | DAILY_MEAN_REVERSION | Kang, Rouwenhorst & Tang, *JF* 2020 | 3d | Free CFTC weekly data |
| F3 | **[NEW] FOMC pre-announcement drift** | new class CALENDAR_ANOMALY (or DAILY_MEAN_REVERSION + gate) | Lucca & Moench, *JF* 2015 | 1d | Small sleeve, ~8 events/year |
| F4 | **[NEW] ETF-flow contrarian** | DAILY_MEAN_REVERSION (sector) | Brown, Davies & Ringgenberg, *RoF* 2021 | 3d | Free daily flow data |

## G. Microstructure / intraday (non-ORB)

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| G1 | FX weekend gap fade | INTRADAY_MICROSTRUCTURE | — | 3d | **DATA**: M5 weekend FX |
| G2 | Small-cap ORB | INTRADAY_BREAKOUT | — | 3d | **DATA**: small-cap intraday |
| G3 | **[NEW] Closing-auction imbalance** | INTRADAY_MICROSTRUCTURE | Bogousslavsky & Muravyev, *JFM* 2023 | 4d | **DATA**: NYSE/Nasdaq imbalance feed |
| G4 | **[NEW] Overnight session decomposition** (long-overnight / flat-intraday) | new class SESSION_DECOMPOSITION (or INTRADAY_MICROSTRUCTURE) | Lou, Polk & Skouras, *JFE* 2019 | 2d | SPY daily open+close — already have |
| G5 | **[NEW] Order-flow-imbalance via L1 ticks** | INTRADAY_MICROSTRUCTURE | Cont-Kukanov-Stoikov 2014 | 4d | Thin margins, commission-sensitive |

## H. Crypto / digital assets

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| H1 | **[NEW] Perp-funding-rate carry** | CARRY (crypto) | Practitioner Paradigm 2022; Alexander et al *JFM* 2023 | 7d | **DATA + EXEC**: needs Binance/Bybit API (not IBKR) |
| H2 | **[NEW] Stablecoin-depeg MR** | DAILY_MEAN_REVERSION | Lyons & Viswanath-Natraj *JIMF* 2023 | 5d | **DATA + EXEC**: same |
| H3 | **[NEW] On-chain MVRV-Z BTC regime overlay** | META_LABELING | Liu & Tsyvinski, *RFS* 2021 + Glassnode methodology | 3d | **DATA**: Glassnode subscription |

## I. ML / statistical

| # | Strategy | Class | Source | Effort | Caveats |
|---|---|---|---|---|---|
| I1 | HMM regime + XGBoost meta-labeler on SPY | ML_CLASSIFIER | resources/ML_Regime_Strategy/ + López de Prado | 3d | Resources folder has full python implementation |
| I2 | **[NEW] Deep-learning cross-section** | ML_CLASSIFIER | Gu, Kelly & Xiu, *RFS* 2020 | 10d | Heavy data infrastructure |
| I3 | **[NEW] Conformal prediction for risk-aware sizing** | META_LABELING | Bates et al *Annals of Stats* 2023 | 2d | Drop-in over existing XGBoost |
| I4 | **[NEW] Temporal Fusion Transformer** | ML_CLASSIFIER (quantile) | Lim et al *IJF* 2021; Wood-Roberts-Zohren *JFDS* 2022 | 8d | Heavy training; transformer infra |

## J. Portfolio construction / risk overlays (infrastructure)

| # | Overlay | Type | Source | Effort |
|---|---|---|---|---|
| J1 | HRP allocator | infrastructure | López de Prado | 3d |
| J2 | **[NEW] Nested Clustered Optimisation (NCO)** | infrastructure | López de Prado SSRN 3469961 2019 | 2d (after J1) |
| J3 | Noise-injection robustness gate | framework axis | Varma | 2d — **NEXT** |
| J4 | DD-conditional de-allocation | infrastructure | Bastien | 2d |
| J5 | Carver FDM (forecast diversification multiplier) | infrastructure | Carver | 1d |
| J6 | Sample-uniqueness weighted purged k-fold | framework | López de Prado | 2d |
| J7 | **[NEW] Betting-Against-Correlation tilts** | sleeve | Asness, Frazzini, Gormsen & Pedersen *JFE* 2020 | 4d |
| J8 | **[NEW] Return Stacking infrastructure** | infrastructure | Newfound + ReSolve 2021 | 4d |

---

## Recommended order of execution

10-step plan, ~31 days total for 8 strategies + 3 infrastructure upgrades.

**Status (2026-05-15):** Steps 1–3 complete; J4 redesign added as 3.5 between E1 and D2. Step 4 (D2) is up next.

1. ✅ **B1 — GEM dual momentum** (1d) — DONE. Original C12 verdict DEPLOY (4-axis). Promoted to production 2026-05-14.
2. ✅ **J3 — Noise-injection robustness gate** (2d) — DONE. 5th decision axis live. C12 demoted DEPLOY → CONDITIONAL_WATCHPOINT. See `Pre-Reg J3 Noise Robustness 5th Axis 2026-05-15.md` + L24 in V3.6 Catalogue.
3. ✅ **E1 — VRP capture** (2d) — DONE (RETIRED). E1 SUSPECT under 5-axis; E1b's percentile-gate redesign aborted at plateau pre-flight. VIX-term-structure VRP on VIXY at retail daily resolution is not viable (L29). See E1 + E1b pre-regs + L25/L26/L27/L28/L29.
3.5. ✅ **J4 — GEM noise-robust redesign** (1d, unplanned) — DONE. Promoted **A1_ewma_hl40** to production 2026-05-15 — recovers C12's DEPLOY verdict under 5-axis. EWMA vol estimator (half-life 40) recovers noise axis to `best` at ~3% Sharpe drag. Live on Docker paper account since 13:07 UTC 2026-05-15. See `Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md` + L30/L31/L32.
4. ✅ **D2 — Commodity futures carry [NEW]** (5d) — DONE (proxy variant RETIRED; strict-carry deferred). Databento `.c.1` queries proved structurally too slow (L34 — ~5 s/bar vs `.c.0`'s 0.005 s/bar; ~4h per commodity for second-month series). Pivoted to yfinance M1-only basket + BGR §3.2 rolling-yield proxy (12-month trailing return). Audit aborted at plateau pre-flight (L27): all 5 cells produced Sharpe -0.30 to -0.62 across 24 commodities × 25 years (2000-2026); commodities exhibit medium-horizon mean reversion AGAINST the rolling-yield signal. RETIRED as a hypothesis; strict-carry BGR remains open pending M2 data acquisition. See `Pre-Reg D2 Commodity Futures Carry 2026-05-15.md` + L34/L35.
5. ✅ **G4 — Overnight session decomposition [NEW]** (2d) — DONE (RETIRE). Gross edge confirmed (+0.75 Sharpe in C3 no-costs) but +0.83 Sharpe cost drag kills it at retail SPY ETF cost levels; Rel-MC ratio = 2.22× buy-hold (L33 — session-bounded strategies have asymmetric tail risk on regime-shuffled MC). See `Pre-Reg G4 Overnight Session Decomposition 2026-05-15.md` + L33.
6. **J1+J2 — HRP → NCO allocator** (3+2d) — infrastructure for multi-strategy portfolios. Becomes load-bearing only when ≥ 3 strategies hold DEPLOY simultaneously.
7. ✅ **A1 — Residual momentum [NEW]** (4d) — DONE (RETIRED on current-S&P-500 universe). Audit aborted at plateau pre-flight (L27): all 5 cells produced negative Sharpe (-0.25 to -0.42), 66.44% relative spread. Survivorship bias + post-2009 cross-sectional momentum decay flip the sign on retail-accessible yfinance current-constituent samples (L36). Survivorship-free CRSP/Compustat data ($2k/yr WRDS) would unblock a proper replication. See `Pre-Reg A1 Residual Momentum 2026-05-15.md` + L36/L37. NOTE: backlog label "A1" — unrelated to J4's GEM cell name "A1_ewma_hl40".
8. **I1 — HMM regime + XGBoost meta-labeler** (3d) — exercises ML class. `research/ml/ML_STRATEGY_DOCUMENTATION.md` is the reference; same-bar causality must be re-verified post-V2.0 framework.
9. **B5 — Intraday momentum (Gao-Han-Li-Zhou) [NEW]** (2d).
10. **B2 — Carver EWMAC ensemble** (5d) — biggest build. **DATA**: same 24-futures requirement as D2.

## Discipline (V3.6 lessons applied)

For every audit:
- **L01-L09** — classify under `StrategyClass`, use class defaults, NEVER deviate without a pre-reg justification
- **L04 / A1** — every position × return product is `pos.shift(1) * ret`; AST-level guardrails enforce
- **L06** — sparse-trade strategies use per-trade Sharpe, not per-bar
- **L08** — MC gates are class-specific (DEFAULTS[cls].mc), not uniform 25%/5%
- **L13** — fold sign-stability quorum is 4 of 5, not 3 (pigeonhole)
- **A4** — WFO honesty: per-fold parameter selection OR pre-registration BEFORE data examined
- **A5** — DSR for any sweep with N>5 cells
- **A6** — MC bootstraps underlying returns + cumprod, NOT strategy returns directly
- **A8** — "validated against X" must point to an artifact + 30s reproducible test
- **A10** — parity tests use independent reference + causality test (corrupt future data, assert past output bit-exact unchanged)
- **V3.1** — pre-committed selection rule to git BEFORE the sweep runs
- **V3.2** — plateau selection, not peak selection: ±1-step grid neighbours must also pass gates
