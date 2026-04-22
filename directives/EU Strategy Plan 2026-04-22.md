# European Strategy Research Plan — 2026-04-22

Detailed plan for running the 19 EU strategy ideas (currency / bonds /
equity / cross-asset) brainstormed after the negative cross-asset
follow-ups. Organised by **data dependency** (what we need, where it
comes from) and **framework reuse** (which existing code path runs it).

> [!IMPORTANT]
> Every experiment below must pass the same quality gate as the v4
> portfolio: `CI_lo ≥ 0.50` on a bootstrap 95 % Sharpe CI, `folds ≥ 25`,
> `pct_positive ≥ 60 %`, `max_dd ≥ -40 %`. Sanctuary window active.
> Any "near-miss" survivor goes to `tier=unconfirmed`, not
> `champion_portfolio`.

---

## Part 0 — Data obtainability matrix

Three sources:
- **yfinance** — free, US-listed ETFs + major futures + daily FX. Already have [scripts/download_data_yfinance.py](../scripts/download_data_yfinance.py).
- **IBKR** via the `EClient`/`EWrapper` pattern from [scripts/download_h1_fx.py](../scripts/download_h1_fx.py). Used for FX H1 (MIDPOINT) and cleaner order-book data. For Eurex futures (FGBL/FBTP/FDAX) we extend the contract builder.
- **Databento** — paid, US equities/futures only ([scripts/download_data_databento.py](../scripts/download_data_databento.py)). Not useful for EU Eurex; **skip for this plan**.

| Data | Needed by | Source | Already have? | Download command |
|---|---|---|:-:|---|
| **Core FX H1** (EUR/USD, AUD/JPY) | champion portfolio | IBKR | ✅ | n/a |
| **EUR/CHF H1** | Strategy 1 | IBKR | ❌ | `uv run python scripts/download_h1_fx.py --pair EUR_CHF --years 15` |
| **EUR/GBP H1** | Strategy 2 | IBKR | ❌ | `uv run python scripts/download_h1_fx.py --pair EUR_GBP --years 15` |
| **EUR/JPY H1** | Strategies 5 | IBKR | ❌ | `uv run python scripts/download_h1_fx.py --pair EUR_JPY --years 15` |
| **Country ETFs** (EWP, EWI, EWQ, EWY, EWC, EWJ) | Strategy 12, 15 | yfinance | ❌ | `uv run python scripts/download_data_yfinance.py --symbols EWP EWI EWQ EWY EWC EWJ` |
| **Europe financials** (EUFN) | Strategy 14 | yfinance | ❌ | `uv run python scripts/download_data_yfinance.py --symbols EUFN` |
| **Oil futures** (CL=F, BZ=F) | Strategy 16 | yfinance | ❌ | `uv run python scripts/download_data_yfinance.py --symbols CL=F BZ=F` |
| **DXY proxy** (UUP) | Strategy 4 | yfinance | ✅ | already local |
| **International bonds** (IGOV, BWX, BNDX) | Strategies 7–10, 19 | yfinance | ✅ | already local |
| **Bund futures H1** (FGBL on Eurex) | Strategy 8, 10 | IBKR (new contract builder needed) | ❌ | see §0.1 below |
| **BTP futures H1** (FBTP on Eurex) | Strategy 6 | IBKR | ❌ | see §0.1 |
| **EU equity indices** (^GDAXI, ^FTSE) | Strategies 11, 13 | yfinance | ✅ | already local |
| **Bund yields** (2y, 10y) | Strategy 9 | ECB Statistical Data Warehouse (SDW) via `pandasdmx` or FRED | ❌ | see §0.2 |
| **Options** (EUR/USD for ECB day vol) | Strategy 18 | IBKR | ❌ — **dropped from plan** (too complex for near-term ROI) |

### §0.1 — IBKR Eurex futures contract extension

Add to [scripts/download_h1_fx.py](../scripts/download_h1_fx.py) (or a
new `scripts/download_eurex_h1.py`) a second contract builder:

```python
def build_future_contract(symbol: str, exchange: str = "EUREX") -> Contract:
    c = Contract()
    c.symbol = symbol        # "FGBL" for Bund, "FBTP" for BTP, "FDAX" for DAX
    c.secType = "FUT"
    c.exchange = exchange
    c.currency = "EUR"
    # IBKR auto-resolves continuous front-month if lastTradeDateOrContractMonth is empty
    return c
```

Eurex futures via IBKR require **Level 1 data subscription** (~$4/mo).
Verify before running:
```
uv run python scripts/check_balance.py          # confirms IBKR connection works
uv run python -c "from ibapi.contract import Contract; ..."  # contract test
```

### §0.2 — Bund yield time series

Easiest path: install `pandasdmx` and pull from ECB SDW:

```python
# research/alpha_loop/load_bund_yields.py (new)
import pandasdmx
ecb = pandasdmx.Request("ECB")
bund10y = ecb.data("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y")
bund2y  = ecb.data("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y")
# write to data/bund_2y_D.parquet, data/bund_10y_D.parquet
```

Alternative: FRED series `IRLTLT01DEM156N` (Germany long-term yield)
and `IRSTCI01DEM156N` (short-term), via `pandas_datareader`. Less
clean but no new API key.

---

## Part 1 — 19 strategies, grouped by data dependency

### Cluster A — reuses existing H1 MR framework (data: IBKR H1 FX)

| # | Strategy | Class | Target | Signal | Filter | Gate check |
|--:|---|---|---|---|---|---|
| 1 | MR EUR/CHF | `MRAUDJPYStrategy` | EUR/CHF H1 | VWAP deviation (anchor=24) | `conf_donchian_pos_20` disagreement | CI_lo ≥ 0.5 |
| 2 | MR EUR/GBP | `MRAUDJPYStrategy` | EUR/GBP H1 | VWAP deviation (anchor=24) | same | CI_lo ≥ 0.5 |
| 5 | MR EUR/JPY (carry-pair) | `MRAUDJPYStrategy` | EUR/JPY H1 | VWAP deviation (anchor=24) | same | CI_lo ≥ 0.5 |

**Shared infrastructure**: reuse `research/mean_reversion/run_confluence_regime_wfo.py` — already parameterised by instrument. Just pass the new parquet path.

**Wall-clock**: ~45 s each × 3 = 2 min total, after data download (~5 min per pair × 3 = 15 min of IBKR sequential pulls).

### Cluster B — cross-asset D using existing `run_bond_wfo` (data: already have)

| # | Strategy | Signal | Target | Framework |
|--:|---|---|---|---|
| 4 | DXY → EUR/USD MR (daily resample) | UUP | EUR_USD | `run_bond_wfo` (inverted to MR by negating z) |
| 13 | DAX vs SPY relative momentum | SPY-normed DAX | ^GDAXI or EWG | custom: build relative-return series, run WFO |
| 16 | Oil → DAX | CL=F | ^GDAXI / EWG / DAX_hedged | `run_bond_wfo` (CL as signal instrument) |
| 17 | Gold → EUR/USD | GLD | EUR_USD (daily resample) | `run_bond_wfo` |
| 19 | Bund-Treasury spread → EUR/USD | IGOV − IEF (synthetic) | EUR_USD | custom: build spread, then `run_bond_wfo` |

**Wall-clock**: ~5 min each via parameter sweep = 25 min total.

### Cluster C — country rotation (new framework, data: yfinance country ETFs)

| # | Strategy | Universe | Sort-horizon | Trade |
|--:|---|---|---|---|
| 12 | 8-country rotation | EWG, EWU, EWP, EWI, EWQ, EWY, EWC, EWJ | 3m, 6m, 12m cross-section | Long top-2 short bottom-2, equal weight |
| 15 | PIIGS vs Core spread | (EWP + EWI) − (EWG + VGK) | 20d / 60d mom | Long-short spread |

**New code**: small `research/cross_sectional/country_momentum.py` (~80 LOC). Ranks cross-sectionally, applies WFO. Model after Asness-Moskowitz; reuse `bootstrap_sharpe_ci` for CI.

**Wall-clock**: implementation ~30 min, sweep ~5 min.

### Cluster D — IC signals on DAX/FTSE directly (data: already have)

| # | Strategy | Signal | Target | Framework |
|--:|---|---|---|---|
| 11 | DAX `rsi_21_dev` 60d | built into IC pipeline | ^GDAXI | `research/ic_analysis/` |
| 11 | FTSE `rsi_21_dev` 60d (already STRONG) | built into IC pipeline | ^FTSE | same |
| 14 | VGK sector rotation (if EUFN etc pass data gate) | `ma_spread_10_50` on EUFN | — | IC Equity pipeline extended |

**Existing framework**: IC pipeline already exists. Just add the index targets to the config. ~15 min code + 10 min sweep.

### Cluster E — EU-native bonds (data: IBKR Eurex or ETF proxy)

| # | Strategy | Data | Status |
|--:|---|---|---|
| 3 | EUR rate-diff carry | Per-country rates (ECB/FRED) | Needs rate data build-out |
| 6 | Bund-BTP spread MR | FGBL + FBTP H1 (IBKR) OR BWX as crude proxy | Needs IBKR Eurex |
| 8 | Bund futures → DAX | FGBL (IBKR) OR use 10y Bund ETF (GLNE, 2009+) | Data-limited but doable |
| 9 | Bund yield curve slope → DAX | ECB SDW | Needs new loader |
| 10 | USD credit → hedged DAX execution test | FGBL + EUR/USD forward | **Execution test, not backtest** |

### Cluster F — deferred / dropped

| # | Strategy | Reason |
|--:|---|---|
| 7 | IGOV → DAX with longer history | Blocked — BNDX data starts 2013; revisit in ~2030 |
| 18 | ECB announcement day EUR/USD vol | Needs options chain + ECB calendar — 3-week engineering, deferred |

---

## Part 2 — Execution plan (4 phases, ~1 week total)

### Phase 1 — "Free" strategies (0 extra data needed, ~3 h)

Strategies 4, 13, 16, 17, 19 from Cluster B + strategy 11 from Cluster D.

| Step | Action | Owner | Est |
|--:|---|---|--:|
| 1.1 | Extend `run_param_sweep.py` to take `--signal_family` arg (e.g. `--oil_dax`) | Engineer | 20 min |
| 1.2 | Create [scripts/rerank/run_param_sweep_eu_native.py](../scripts/rerank/run_param_sweep_eu_native.py) with 6 channels: DXY→EURUSD, CL→DAX, GLD→EURUSD, IGOV-IEF→EURUSD, DAX-SPY relative, IC on DAX/FTSE | Researcher | 45 min |
| 1.3 | Run sweep (~1,500 combos × 6 channels = ~9,000 combos, ~10 min wall-clock) | Operator | 15 min |
| 1.4 | Analyse + write `directives/EU Native Sweep 2026-04-XX.md` | Researcher | 45 min |

**Pass/fail criterion**: does any channel produce a Bonferroni survivor?

### Phase 2 — FX downloads + MR tests (IBKR data required, ~4 h)

Strategies 1, 2, 5 from Cluster A.

| Step | Action | Owner | Est |
|--:|---|---|--:|
| 2.1 | Start IB Gateway / TWS on port 7497 (paper) | Operator | 5 min |
| 2.2 | Run IBKR H1 pulls for EUR_CHF, EUR_GBP, EUR_JPY × 15 years each (sequential; ~20 min per pair) | Operator | 1 h |
| 2.3 | Verify parquet: sanity-check bar counts, continuity, no duplicate index labels | Researcher | 15 min |
| 2.4 | Create `scripts/rerank/run_mr_eu_fx_sweep.py` reusing the confluence regime WFO, sweeping each pair × {vwap_anchor 24, filter donchian_pos_20/no_filter, tier grid {standard, conservative}} | Engineer | 30 min |
| 2.5 | Run sweep (3 pairs × ~20 configs = ~60 runs, ~3 min) | Operator | 5 min |
| 2.6 | Analyse + write directive | Researcher | 45 min |

**Pass/fail criterion**: any pair ≥ AUD/JPY's OOS Sharpe (+1.05)?

### Phase 3 — Country rotation (yfinance download + new framework, ~5 h)

Strategies 12, 15 from Cluster C.

| Step | Action | Owner | Est |
|--:|---|---|--:|
| 3.1 | Download country ETFs: `uv run python scripts/download_data_yfinance.py --symbols EWP EWI EWQ EWY EWC EWJ EUFN` | Operator | 10 min |
| 3.2 | Implement `research/cross_sectional/country_momentum.py` (Asness-Moskowitz: rank, long top-k, short bottom-k, monthly rebalance) | Engineer | 1.5 h |
| 3.3 | WFO driver `research/cross_sectional/run_country_wfo.py` with 504/126 IS/OOS split, sweep {k=1,2,3} × {3m, 6m, 12m lookback} × {long-only, long-short} | Engineer | 45 min |
| 3.4 | Run sweep (~30 configs × ~10 s each) | Operator | 5 min |
| 3.5 | PIIGS vs Core spread as a special case (compute spread, run WFO as a single-asset trend strategy) | Engineer | 30 min |
| 3.6 | Analyse + write directive | Researcher | 45 min |

**Pass/fail criterion**: country momentum at Bonferroni gate would be a genuine diversifier (zero correlation with v4).

### Phase 4 — EU-native bonds (data build-out, ~1 week)

Strategies 3, 6, 8, 9 from Cluster E. **Bigger project — tackle only if Phases 1–3 produce survivors and we want to extend the EU basket.**

| Step | Action | Est |
|--:|---|--:|
| 4.1 | Add Eurex futures contract builder to `download_h1_fx.py` (generalise) | 1 h |
| 4.2 | Test-pull FGBL (Bund) 1 year | 30 min |
| 4.3 | Extend WFO to handle futures (continuous roll, contract changes) | 3 h |
| 4.4 | Build Bund yield loader from ECB SDW | 2 h |
| 4.5 | Run sweeps: Bund→DAX, Bund-BTP→DAX, slope→DAX | 30 min |
| 4.6 | Write directive | 1 h |

---

## Part 3 — Portfolio integration path

If any strategy from Phases 1–4 passes the gate:

1. **Correlation check with v4**: run a 6-strategy (new + v4 five) correlation matrix via `scripts/rerank/optimize_v4_portfolio.py` extended with the new candidate.
2. **If max |ρ| < 0.5 with everything in v4**: candidate becomes v4.1.
3. **If |ρ| ≥ 0.5 with any v4 member**: reject unless it out-performs that member on CI_lo.
4. **Equal-weight 6 or 7 strategies**: recompute portfolio Sharpe and DD.
5. **Live deployment path**: 48 h paper dry-run at 10 % sizing, then ramp.

---

## Part 4 — Expected outcomes (what the distribution should look like)

**Prior probabilities**, based on the EU follow-ups evidence:

| Cluster | Prob ≥ 1 Bonferroni survivor | Rationale |
|---|---:|---|
| A (FX MR) | 45 % | EUR/CHF has a real SNB-anchor story; EUR/JPY is the second-best carry pair globally |
| B (cross-asset daily) | 20 % | Already tested cross-asset; the free follow-ups are mostly variations of the same factor |
| C (country momentum) | 55 % | Well-documented academic factor; directly applicable to our data |
| D (IC on DAX/FTSE) | 35 % | FTSE IC signals already STRONG but untested as end-to-end WFO |
| E (EU bonds) | 40 % | Native signal for native target — the structural fit we've been missing |

**Most likely mode**: 1–2 survivors across Phases 1–3 (total combos ~15,000; expected Bonferroni survivors under the null ≈ 9, but we have real signal intuitions for about 5 of the 19). If the actual count is zero, it's a strong negative result for the entire EU hypothesis. If ≥ 3 survivors appear, validate each one against the sanctuary window **before** promoting to v4.1.

---

## Part 5 — Risk caveats specific to EU research

1. **Data-licensing**: IBKR Eurex data requires a Level-1 futures subscription (~$4/mo). Check your active subscriptions before Phase 4.
2. **FX-hedged execution**: any strategy that works on a currency-hedged synthetic (like the +0.425 CI_lo hedged-DAX result from v2) needs an execution-layer paper test, not just a backtest. Add to live deployment checklist.
3. **Regime instability**: Europe's macro regime shifted materially in 2022 (energy crisis, ECB pivot, euro bottom). Any strategy with IS ending pre-2022 and OOS spanning 2022+ may show artificial OOS degradation.
4. **ETF proxy drift**: EWG/EWU/IEV have tracking error vs their native indices (~10–20 bps/yr) that accumulates in long-horizon rotation strategies. Account for this in the Sharpe gate by adding 1 bp/day cost if the strategy holds for 60+ days.
5. **Survivorship**: the 19 strategies are my ideas not the academy's; we've already biased our brainstorm toward channels with plausible stories. The Bonferroni correction on the *final combined sweep* should use the grand total combos count, not per-cluster.

---

## Decision point

**Greenlight for Phase 1 only** is the recommended next step — it's
"free" (no downloads, no new code beyond a sweep wrapper) and resolves
5 of 19 strategies in one 3-hour session. If Phase 1 produces any
survivor, proceed to Phase 2. If Phase 1 is fully negative, we have
strong evidence the EU gap is closed at the cross-asset level and can
decide whether to invest in Phases 2–4 on their own merits (FX MR,
country rotation) rather than as EU-specific diversifiers.

Wait for user confirmation before starting any phase.
