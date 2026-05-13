# IG Markets Migration Research — bond_equity strategies

**Last updated:** 2026-05-12 (live-account scout)
**Status:** RESEARCH — nothing deployed. No code in `titan/` modified. No live orders placed.
**Scope:** Migrate the three live `bond_equity_*` strategies (signal IHYU/IHYG → trade CSPX / VUSD / EIMI) from IBKR onto a UK retail IG Markets spread betting account.
**Companion artefacts:**
- Scout script: [scripts/ig_scout_catalog.py](../scripts/ig_scout_catalog.py)
- Offline analyser: [scripts/ig_map_underlyings.py](../scripts/ig_map_underlyings.py)
- Demo output: `.tmp/ig_catalog/20260512T114959Z/` + `20260512T115918Z/`
- **Live output (authoritative):** `.tmp/ig_catalog/20260512T122019Z/` (searches) + `20260512T122129Z/` (epic details)

---

## TL;DR

> [!WARNING]
> **Instrument-level: clean 1:1 mapping. Cost-level: at current rates only ONE strategy barely passes — and a financing-rate bug fix made the picture worse than originally reported.**
>
> All five strategy underlyings map 1:1 to live IG spread-bet epics. But the corrected Phase R1 cost-aware WFO (re-run 2026-05-12 with `/252` financing divisor fix and leverage capped at FCA retail 5x) shows:
>
> | Strategy | IG @ 4 %/yr | IG @ 6.8 %/yr (current SONIA + 2.5 % admin) | IG @ 8 %/yr |
> |---|---|---|---|
> | IHYU → CSPX | ❌ FAIL (CI lo -0.01) | ❌ FAIL (CI lo -0.13) | ❌ FAIL (CI lo -0.18) |
> | IHYG → VUSD | ✅ PASS (CI lo +0.18) | ✅ PASS (CI lo +0.04, **marginal**) | ❌ FAIL (CI lo -0.02) |
> | IHYG → EIMI | ✅ PASS (CI lo +0.01, **marginal**) | ❌ FAIL (CI lo -0.11) | ❌ FAIL (CI lo -0.16) |
>
> **At today's real IG financing rate (~6.8 %/yr = SONIA 4.3 % + IG admin 2.5 %), only IHYG → VUSD passes, and only just (CI lower bound +0.04).** A 50-100 bps move up in SONIA breaks it. The other two strategies fail.

> [!IMPORTANT]
> **Three migration options open to you:**
>
> **Option A — Migrate IHYG → VUSD only, capped at 1-2x leverage.** Accept that the IG SB tax shelter eats most of the alpha. Expected: Sharpe ~+0.6, ret ~6-12 %/yr, max DD ~20-35 %. Two strategies (`ihyu_cspx`, `ihyg_eimi`) stay on IBKR or get dropped.
>
> **Option B — Skip the DFB and use quarterly futures spread bets (`KA.D.*.JUN/SEP/DEC.IP`).** Quarterly contracts avoid daily financing entirely; cost shifts to the embedded "fair value adjustment" baked into the contract price. May change the migration economics meaningfully — needs separate WFO using futures-spread data.
>
> **Option C — Modify the strategy to reduce time-in-market.** Current strategies are in position ~50 % of trading days. Adding a regime filter (e.g., only trade when realised volatility ≥ X, or only after a credit-spread shock signal) could halve the financing exposure. Becomes a research thread on its own.
>
> **Default if none chosen:** keep all three on IBKR; defer migration.

| Strategy | IBKR signal | IG signal epic | IBKR position | IG position epic |
|---|---|---|---|---|
| `bond_equity_ihyu_cspx` | IHYU.LSEETF | `KA.D.IHYULN.DAILY.IP` | CSPX.LSEETF | `KA.D.CSPXLN.DAILY.IP` |
| `bond_equity_ihyg_vusd` | IHYG.LSEETF | `KA.D.IHYGLN.DAILY.IP` | VUSD.LSEETF | `KA.D.VUSDLN.DAILY.IP` |
| `bond_equity_ihyg_eimi` | IHYG.LSEETF | (same IHYG epic) | EIMI.LSEETF | `KA.D.EIMILN.DAILY.IP` |

All five epics are: `type=SHARES`, `expiry=DFB` (perpetual daily funded bet), **streaming via Lightstreamer = True**, tier-1 margin 20 %, min stake £0.01/point, native currency match (USD for the four USD-line ETFs, EUR for IHYG, GBP fallback).

**Architectural consequence:** the migration is no longer hybrid. Signal data and execution both come from IG. The strategy class (`titan/strategies/bond_gold/strategy.py`) does not change algorithmically; what changes is the **adapter**, the **instrument mapping**, and the **sizing math** (£/point not USD/share).

> [!WARNING]
> **The cost picture is the binding gate, not instrument availability.** Live snapshot spreads are: CSPX 0.024 %, VUSD 0.12 %, EIMI 0.32 %, IHYU 0.24 %, IHYG 0.24 % bid/ask. Plus overnight funding ~5 %/yr on the long side for DFBs. A strategy with 5-10 day hold and 50 round-trips/year accumulates roughly **24-30 % annual drag** on the IHYG/IHYU/EIMI legs, much less on CSPX/VUSD. The original IBKR backtest cost model assumed 1-3 bps commission + tight spreads, an order of magnitude tighter. **The Phase R1 WFO re-validation with IG's actual cost model is the critical go/no-go gate** — the existing Sharpe +1.64 will not survive intact.

> [!IMPORTANT]
> **Open question carried forward:** decide whether `bond_equity_ihyg_vusd` (now `KA.D.VUSDLN`) and `bond_equity_ihyu_cspx` (now `KA.D.CSPXLN`) — both Sterling-tax-free S&P 500 UCITS proxies — provide enough product diversification when held side-by-side, or whether to drop one. The recorded 0.52 inter-strategy correlation from [project_ihyg_cspx_discovery.md](../memory/project_ihyg_cspx_discovery.md) was on the *IBKR* products; the IG share-bet equivalents track the same S&P 500 but with different liquidity / spread profiles. Likely correlation rises closer to 0.7-0.9 in practice. Re-derive empirically once we have IG paper-trade data.

---

## 1. Spread betting — what actually changes vs. holding ETFs at IBKR

### 1.1 Mechanics
- **Bet size is in £/point**, not shares. One "point" = the smallest price move IG quotes (for US 500 that's 0.1 index points based on the scout response showing `min step: 1.0POINTS` and `min/max deal: 0.04POINTS` minimum stake).
- **No share ownership**. You are taking a contract-for-difference dressed in UK gambling-tax wrapping. You receive no dividends; instead, IG credits/debits a "dividend adjustment" to your account on ex-date (a couple of bps drift vs. holding the underlying outright).
- **Tax (UK)**: spread betting profits are exempt from CGT and income tax in the UK (HMRC's official position). Losses are not deductible. The user's plan in [Broker Migration Assessment.md](Broker%20Migration%20Assessment.md) is the primary driver.
- **Overnight funding ("daily funded bet" / DFB)**: for cash markets, IG debits an overnight financing charge each night the position is open. From IG's published methodology this is `(LIBOR-style reference rate ± IG admin spread) × position notional × days / 365`. For long S&P 500 cash bets currently the charge is roughly **5–6 %/yr** of position notional — material for a strategy with average hold-time of 5–10 days. **Cost-model update required before paper deployment.**
- **Margin**: from scout data — US 500 cash is `marginFactor=5%`, MSCI EM ETF is `20%`. The strategy's `max_leverage=2.0` in [scripts/run_portfolio.py:524](../scripts/run_portfolio.py#L524) is hard-capped well below this, so margin is non-binding, but operator account-margin monitoring is still needed.

### 1.2 Quote conventions — verified against live

All five target epics share the same shape (`SHARES` type, DFB, Lightstreamer streaming, tiered margin):

| Epic | Name | Native ccy | Tier-1 margin | Min stake | Min stop dist | Live spread (snapshot) | "Unborrowable" |
|---|---|---|---|---|---|---|---|
| `KA.D.CSPXLN.DAILY.IP` | iShares Core S&P 500 UCITS ETF - CSPX | USD | 20 % | 0.01 POINTS | 32 POINTS | **0.024 %** (79339/79358) | no |
| `KA.D.VUSDLN.DAILY.IP` | Vanguard S&P 500 UCITS ETF - USD | USD | 20 % | 0.01 POINTS | 15 POINTS | **0.12 %**  (13987/14004) | yes |
| `KA.D.EIMILN.DAILY.IP` | iShares Core MSCI EM IMI UCITS ETF | USD | 20 % | 0.01 POINTS | 13 POINTS | **0.32 %**  (5386/5403)   | yes |
| `KA.D.IHYULN.DAILY.IP` | iShares $ HY Corp Bond UCITS ETF - USD | USD | 20 % | 0.01 POINTS | 21 POINTS | **0.24 %**  (9535/9558)   | yes |
| `KA.D.IHYGLN.DAILY.IP` | iShares € HY Corp Bond UCITS ETF - IHYG | EUR | 20 % | 0.01 POINTS | 21 POINTS | **0.24 %**  (9119/9141)   | yes |

Common fields: `streamingPricesAvailable: True`, `slippageFactor: 100% (POSITION)`, `marketOrderPreference: AVAILABLE_DEFAULT_OFF`, `trailingStops: AVAILABLE`, `currencies: [native, GBP]`, `country: GB`, `chartCode` = the bare ticker (CSPX, VUSD, EIMI, IHYU, IHYG).

> [!CAUTION]
> **"Unborrowable" applies to four of the five epics.** This means IG will not extend a short on these instruments (no securities-borrow available against the SB book). For `bond_equity_*` this is fine — they're long-only on the position leg. **But if any future strategy variant goes short on VUSD/EIMI/IHYU/IHYG, that variant cannot run on IG SB.** Update strategy specs accordingly.

### 1.3 Margin band escalation (verified from `/markets/{epic}` v3 `marginDepositBands`)

IG escalates margin punitively as position size grows. The four bands per epic:

| Epic | Tier 1 (≤ band-1 max) | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| `KA.D.CSPXLN.DAILY.IP` | 20 % up to $42 stake | 45 % up to $120 | 75 % above $120 |   |
| `KA.D.VUSDLN.DAILY.IP` | 20 % up to $497 | 40 % up to $1,420 | 75 % above $1,420 |   |
| `KA.D.EIMILN.DAILY.IP` | 20 % up to $287 | 25 % up to $820 | 55 % up to $820 | 75 % above |
| `KA.D.IHYULN.DAILY.IP` | 20 % up to $115 | 45 % up to $460 | 75 % above $460 |   |
| `KA.D.IHYGLN.DAILY.IP` | 20 % up to €145 | 45 % up to €580 | 75 % above €580 |   |

The "stake" units here are $/point or €/point — IG's tier thresholds are expressed in the bet currency. **CSPX has the most aggressive escalation** ($42/point ≈ a £40/point bet at GBP/USD≈1.05 — easy to exceed at moderate account sizes). At a target 2x leverage on a £10k account, you'd want to stay in tier-1, which constrains max stake per position. The sizing helper must respect this — `convert_notional_to_bet_per_point` should cap output to the tier-1 max.

### 1.3 Sizing — the breakage point for naïve migration

Current sizing in [titan/strategies/bond_gold/strategy.py](../titan/strategies/bond_gold/strategy.py) (reused for all three `bond_equity_*` strategies) computes a USD notional via vol-targeting, converts to **shares** via `convert_notional_to_units(notional_usd / price_usd_per_share)`, and submits to IBKR.

On IG you do not have shares. The mapping becomes:
```
target_notional_gbp     = vol_target * sqrt(equity_gbp / vol_estimate_gbp)
bet_size_per_point_gbp  = target_notional_gbp / (price_in_points * point_value_gbp)
```
For US 500: `point_value_gbp = 1 £/point × £/£` (account is GBP; bet quoted in GBP). For the MSCI EM share bet, "point" = 1 penny of the share price.

This requires a **new sizing helper** alongside `convert_notional_to_units` — call it `convert_notional_to_bet_per_point` — that lives in `titan/risk/spread_bet_sizing.py` and follows the same fail-fast discipline (raise if `quote_ccy != base_ccy` without an explicit FX rate). Code does not exist yet.

### 1.4 Currency
- Account is **GBP base** (confirmed: `session_meta.currencyIsoCode=GBP`).
- The bond_equity champion is configured for **USD-quoted strategies** (`base_currency=USD` implicit, `fx_rate_quote_to_base` plumbing in [titan/risk/strategy_equity.py](../titan/risk/strategy_equity.py)).
- Two options:
  - **(A) Rebase the strategies to GBP.** P&L is GBP-native, easier to reason about for a UK tax-free account, but the existing backtests/`StrategyEquityTracker` are USD-based and need re-running.
  - **(B) Keep USD-quoted, convert at the adapter boundary.** Strategy thinks in USD; adapter converts at trade time using the same `fx_rate_quote_to_base` mechanism the FX strategies use. Simpler diff to existing code. Recommended.

Recommendation: **(B)**, with `fx_rate_quote_to_base = GBP-per-USD spot` plumbed through the IG adapter. We have prior art for this in `mr_audjpy` (JPY→USD).

---

## 2. IG REST + streaming API surface (verified against demo)

### 2.1 Auth
- **v2 `/session`** — POST `{identifier, password}` with header `X-IG-API-KEY` + `VERSION: 2`. Returns body with `currentAccountId`, `lightstreamerEndpoint`, plus headers `CST` and `X-SECURITY-TOKEN` (~24 h validity).
- **v3 `/session`** — same body but returns `oauthToken` (access + refresh). Cleaner for long-running processes since refresh is on a Bearer token, but adds plumbing. Recommend v2 for first cut; v3 as a follow-up.
- **`PUT /session`** — switch active account (when API key default ≠ desired). Used by the scout when `IG_ACCOUNT_ID` doesn't match the credentials' default account.
- **`DELETE /session`** — invalidate tokens. Mandatory on shutdown to avoid leaked sessions counting against concurrent-session caps.

### 2.2 Read-only catalogue (verified working)
- `GET /marketnavigation[/{id}]` — hierarchical tree, recursive. Heavy; depth 2 walks ~150 nodes.
- `GET /markets?searchTerm=...` — free-text search; returns up to 50 `{epic, instrumentName, instrumentType, expiry, bid, offer, marketStatus}` rows. Verified across 23 distinct queries.
- `GET /markets/{epic}` — full detail: `instrument`, `dealingRules`, `snapshot`. **This endpoint has a daily allowance** — we saw `error.public-api.exceeded-api-key-allowance` after ~150 calls. Production must cache aggressively (mapping is mostly static; refresh weekly).

### 2.3 Trading endpoints (NOT exercised by scout — listed here for adapter design only)
| Endpoint | Purpose | Notes |
|---|---|---|
| `POST /positions/otc` v2 | Open spread-bet position (MARKET, LIMIT, QUOTE) | Body: `{epic, expiry, direction, size, orderType, level?, guaranteedStop, stopLevel?, limitLevel?, trailingStop?, currencyCode, forceOpen}` |
| `PUT /positions/otc/{dealId}` | Modify open position (stop / limit only — not size) | |
| `DELETE /positions/otc/{dealId}` v1 | Close position fully or partially | Body: `{dealId, direction (opposite), size, orderType}` |
| `GET /positions` v2 | List open positions | |
| `GET /confirms/{dealReference}` v1 | Get confirmation of an async deal | **Critical**: `POST /positions/otc` returns a `dealReference` immediately; the actual fill comes async — must poll `confirms` or subscribe to `TRADE` stream on Lightstreamer. |
| `POST /workingorders/otc` | Limit / stop entries | |
| `GET /history/transactions` v2 | Historical fills | Use for reconciliation. |
| `GET /history/activity` v3 | Account activity (orders, deletions) | |

### 2.4 Streaming (Lightstreamer)
- Endpoint provided in session response: `https://demo-apd.marketdatasystems.com`.
- Protocol is **Lightstreamer TLCP** — proprietary text-line protocol, not WebSocket. Python client: `lightstreamer-client` (official, sync) or `ls-python-client` (community async wrappers).
- Subscription items follow `<MODE>:<symbol>` form:
  - `MARKET:IX.D.SPTRD.DAILY.IP` → tick-level bid/offer/UTM/MID_OPEN/HIGH/LOW
  - `CHART:IX.D.SPTRD.DAILY.IP:1MINUTE` → 1-min OHLC bars (also SECOND, 5MINUTE, HOUR)
  - `TRADE:<accountId>` → fill confirmations (mirrors what `GET /confirms` provides synchronously)
  - `ACCOUNT:<accountId>` → balance / margin / available updates
- All `bond_equity_*` strategies run on **DAILY bars** → no live tick subscription needed for the signal. We need streaming only for:
  - Trade confirmations (`TRADE:<accountId>`)
  - Account state (`ACCOUNT:<accountId>`)
  - Optionally intraday quotes for slippage monitoring on entry

> [!IMPORTANT]
> Lightstreamer is the single biggest implementation risk in this adapter. Unlike OANDA's clean REST-streaming endpoint (your `Titan-Oanda-Enhanced` adapter uses HTTP chunked-transfer), IG forces a stateful Lightstreamer session. Your existing OANDA adapter's `oandapyV20.streaming` pattern does **not** port directly. The Python `lightstreamer-client` package handles reconnection/heartbeat for you but is sync-only — for NautilusTrader integration you need to drive it from a dedicated thread and bridge events back into the async runtime via an `asyncio.Queue`.

### 2.5 Historical bars
- `GET /prices/{epic}` v3 — `{resolution, from, to, max}`. Resolutions: SECOND, MINUTE, MINUTE_2/3/5/10/15/30, HOUR, HOUR_2/3/4, DAY, WEEK, MONTH.
- **Allowance**: 10,000 *historical price points* per week per app key (separate quota from the request rate). Daily-bar warmup for two epics × 5 years ≈ 2,500 points — well within budget.
- Returns: OHLC bid/offer plus `lastTradedVolume`. Bid/offer split means we get the actual spread at each bar — useful for cost modelling.

### 2.6 Rate limits (from IG developer docs, cross-checked with our scout 429/403 behaviour)
| Type | Limit |
|---|---|
| Non-trading REST requests (per app) | 60 / minute |
| Non-trading REST requests (per account) | 30 / minute |
| Trading REST requests | 100 / minute per account |
| Historical price points | 10,000 / week per app |
| `/markets/{epic}` detail | Hits an app-level *daily allowance* (we triggered it at ~150 calls) |

The scout's 1.2s inter-request sleep stays under per-minute limits; the daily allowance forces us to cache catalogue calls in a static mapping table refreshed once a week.

---

## 3. NautilusTrader IG adapter — design

### 3.1 Reuse from your Titan-Oanda-Enhanced
Your OANDA adapter at https://github.com/rayan-azhari/Titan-Oanda-Enhanced demonstrates the right Nautilus shape:
```
adapters/oanda/
    config.py          — OandaDataClientConfig, OandaExecClientConfig
    factories.py       — OandaLiveDataClientFactory, OandaLiveExecClientFactory
    providers.py       — InstrumentProvider (loads OANDA Instrument list, maps to Nautilus types)
    data.py            — LiveMarketDataClient — streams prices via REST-stream
    execution.py       — LiveExecutionClient — order submission, position tracking
    http_client.py     — auth, retries, paging
```

The IG adapter mirrors this 1:1, but with **two structural differences**:

1. **Streaming uses Lightstreamer not REST-stream.** A separate `lightstreamer_client.py` module wraps the official `lightstreamer-client` Python package, runs it in a daemon thread, and bridges events into Nautilus via an `asyncio.Queue` consumed by `data.py`. The OANDA adapter's `OandaStreamClient` (REST chunked) is not reusable as-is.

2. **Async deal confirmation.** OANDA returns fills synchronously on the order response. IG returns a `dealReference` and the fill follows on the `TRADE` stream (or via polling `GET /confirms/{dealReference}`). The exec client must (a) submit, (b) record the dealReference, (c) match the incoming TRADE message back to the originating Nautilus `OrderId`. This is a small but irreducible piece of state-machine complexity. The OANDA adapter has no analogue — its order submission is one-shot.

### 3.2 Proposed module layout
```
titan/adapters/ig/
    __init__.py
    config.py
        IGDataClientConfig(NautilusKernelConfig)
        IGExecClientConfig(NautilusKernelConfig)
            base_url, api_key, username, password, account_id, env (demo|live)
    auth.py
        IGAuthClient — v2 login/refresh/logout, CST/XST header management
    http_client.py
        IGHttpClient — retry/backoff, rate-limit aware (per-minute + per-day budgets)
    lightstreamer_client.py
        IGLightstreamerClient — owns the LS thread, exposes async iterator API
    providers.py
        IGInstrumentProvider — caches /markets/{epic} once per epic, maps to Nautilus
            Instrument (use Cfd or Equity subtype as appropriate)
    data.py
        IGLiveDataClient — implements LiveMarketDataClient
            - subscribe(BarType) → polls /prices for warmup, then CHART subscription
            - subscribe(InstrumentId) for tick → MARKET subscription
    execution.py
        IGLiveExecutionClient
            - submit_order → POST /positions/otc
            - cancel_order → DELETE /workingorders/otc/{dealId}
            - modify_order → PUT /positions/otc/{dealId}
            - position state reconciled from TRADE stream + GET /positions on startup
    factories.py
        IGLiveDataClientFactory, IGLiveExecClientFactory
    schemas.py
        TypedDicts mirroring IG response payloads — kept thin, only used internally
```

### 3.3 Sizing & FX plumbing (new code in titan/risk/)
```
titan/risk/spread_bet_sizing.py
    convert_notional_to_bet_per_point(
        notional_base: Decimal,       # e.g. $20,000 USD if strategy thinks USD
        price_points: Decimal,        # the IG quote in "points" (US 500 ≈ 7400 points)
        point_value_quote: Decimal,   # currency-per-point unit (US 500: 1 USD/point per £1/point bet)
        quote_ccy: str,
        base_ccy: str,
        fx_rate_quote_to_base: Decimal,  # raises if missing and currencies differ
        min_bet_size: Decimal,        # from /markets/{epic} dealingRules.minDealSize
        bet_step: Decimal,            # from /markets/{epic} dealingRules.minStepDistance
    ) -> Decimal:                     # rounded to step, validated against min
        ...
```
This mirrors the existing `convert_notional_to_units` and inherits the same fail-fast discipline (no silent FX = 1.0 fallback).

### 3.4 PortfolioRiskManager wiring
No PRM changes required. The integration contract documented at the top of [scripts/run_portfolio.py](../scripts/run_portfolio.py) (the `report_equity_and_check` / `StrategyEquityTracker` pattern) is broker-agnostic — strategies report PnL in their own currency, PRM converts to base, allocator works on aggregated risk. The adapter swap is invisible to PRM.

### 3.5 Live ↔ research parity
Per [research-math-guardrails](research-math-guardrails) project rule, every live strategy needs a `test_*_live_parity.py` asserting one bar of signal/vol/size match between backtest and live. The IG adapter introduces a new code path for sizing (`convert_notional_to_bet_per_point`) — we add `tests/test_bond_equity_ig_parity.py` that:
1. Loads CSPX.LSEETF historical data and computes one day's signal + size in USD (existing path).
2. Maps that to a £/point bet using the same target volatility (via the new sizing function) and asserts the resulting GBP risk per unit volatility matches the USD risk per unit volatility within 5 bps.

---

## 4. Per-strategy migration matrix

| Strategy | Signal (IG epic) | Position (IG epic) | Migration status |
|---|---|---|---|
| `bond_equity_ihyu_cspx` | `KA.D.IHYULN.DAILY.IP` (IHYU USD) | `KA.D.CSPXLN.DAILY.IP` (CSPX USD) | **Direct 1:1 mapping**. Same iShares funds the strategy was originally researched on. WFO re-validation only needs an updated cost model (spread + financing), not an instrument-substitution sensitivity check. |
| `bond_equity_ihyg_vusd` | `KA.D.IHYGLN.DAILY.IP` (IHYG EUR) | `KA.D.VUSDLN.DAILY.IP` (VUSD USD) | **Direct 1:1 mapping**. Same Vanguard fund the strategy was originally researched on. Cross-currency signal→position is intentional (EUR HY drives USD equity) — no FX-rate change vs original research. |
| `bond_equity_ihyg_eimi` | `KA.D.IHYGLN.DAILY.IP` (IHYG EUR, shared with above) | `KA.D.EIMILN.DAILY.IP` (EIMI USD) | **Direct 1:1 mapping**. Same iShares MSCI EM IMI fund. Critical: pick the **USD line** epic (`KA.D.EIMILN`) — the **GBP line** (`KA.D.EMIMLN.DAILY.IP`, price ~3990) is a different share class with embedded GBP/USD overlay, and was historically the source of the WFO confusion documented in [project_ihyg_emim_discovery.md](../memory/project_ihyg_emim_discovery.md). |

> [!IMPORTANT]
> The `mr_audjpy` and `mr_audusd` strategies in the wider portfolio are FX — IG offers AUD/JPY and AUD/USD as spot spread bets (not exercised by this scout but standard IG offering). Their migration is largely orthogonal: same signal, different sizing math, no HY signal data dependency. Out of scope here but on the same critical path if you want the *whole* champion portfolio on IG.

---

## 5. Phased migration roadmap

Each phase has a single explicit exit gate. Do not skip ahead.

### Phase R0 — Catalogue verification (COMPLETE)
- [x] Demo API auth working
- [x] Demo scout instrument list captured (turned out to be a restricted subset)
- [x] **Live scout completed** (account `APS9A`, GBP base) — all 5 underlyings confirmed available 1:1
- [x] HYGU disambiguated — it's the USD-hedged share class of IHYG, not a substitute for IHYU. Not used.

### Phase R1 — Cost-aware WFO re-validation on IG vehicles (COMPLETE)
- [x] Ran WFO on all 3 strategies under both IBKR and IG cost regimes (financing sensitivity 4 / 6 / 8 %/yr)
- [x] Script: [research/cross_asset/run_bond_equity_ig_vs_ibkr.py](../research/cross_asset/run_bond_equity_ig_vs_ibkr.py)
- [x] Output: `.tmp/reports/bond_equity_ig_vs_ibkr.{csv,md}`

**Phase R1 verdict (corrected, deployment gate = 95 % bootstrap CI lo > 0; leverage capped at FCA retail 5x):**

| Strategy | Zero-cost Sharpe (CI lo) | IBKR $25k Sharpe (CI lo) | IG SB 4 %/yr (CI lo) | IG SB 6.8 %/yr — current (CI lo) | IG SB 8 %/yr (CI lo) | **Migrate?** |
|---|---|---|---|---|---|---|
| IHYU → CSPX | +0.73 (+0.18) | +0.66 (+0.10) ✅ | +0.54 (−0.01) ❌ | +0.43 (−0.13) ❌ | +0.38 (−0.18) ❌ | **NO** at any realistic IG rate |
| IHYG → VUSD | +1.18 (+0.60) | +1.03 (+0.45) ✅ | +0.76 (+0.18) ✅ | +0.62 (+0.04) ✅ marginal | +0.56 (−0.02) ❌ | **MARGINAL** — passes today, fails on +50 bps SONIA |
| IHYG → EIMI | +1.27 (+0.66) | +1.03 (+0.43) ✅ | +0.61 (+0.01) ✅ marginal | +0.50 (−0.11) ❌ | +0.45 (−0.16) ❌ | **NO** at current rates |

> [!WARNING]
> **Correction history.** The original Phase R1 run (earlier in this document's history) had a financing-divisor bug: `rate / 365.25` was applied per trading bar, but trading-day bars average 1.45 calendar days each, so the actual financing drag is 45 % higher than originally reported. Fixed in `daily_drag` (commit-pending: divide by 252 trading bars/yr instead). The corrected numbers above are the binding ones; the previously reported "IHYG → VUSD passes at all rates" was based on the buggy math.

> [!CAUTION]
> **Leverage is verified Sharpe-invariant.** Sensitivity sweep 1x–5x (FCA retail cap) confirmed: stitched Sharpe and CI bounds are identical to 2 decimal places across all leverage levels for both IBKR and IG. Returns and drawdowns scale linearly. Implication: leverage cannot rescue a strategy whose CI lower bound is negative at 1x — it only amplifies a positive (or negative) alpha proportionally.

> [!IMPORTANT]
> **Even the strategy that passes (IHYG → VUSD) is dangerous at higher leverage**. 1x DD is already −19 %. At 5x retail max it becomes −71 %. The DD-budget table in `.tmp/reports/bond_equity_ig_vs_ibkr.md` shows that to stay under a 25 % DD budget on IHYG → VUSD at current 6.8 % financing, max usable leverage is **~1.3x**, generating ~7.5 %/yr return.

**Strategy modification ideas (if proceeding despite the verdict):**

The corrected WFO suggests financing drag is the binding constraint. Two structural changes could improve the picture:

1. **Reduce time-in-market.** Current strategies hold a position ~50 % of trading days. A regime filter (e.g. only trade when realised credit-spread vol > X, or skip first 5 days of every signal as a confirmation lag) could cut financing exposure proportionally. Trades fewer but with potentially higher edge per trade.
2. **Switch from DFB to quarterly futures spread bets.** IG offers `KA.D.*.JUN/SEP/DEC.IP` for all five epics. Quarterly contracts have **no daily financing** — the rate is baked into the contract spread via "fair value adjustment" at roll. Cost profile changes from daily-grind to quarterly-roll. Needs a separate WFO using futures-spread time series.
3. **Add a drawdown stop overlay.** Halt strategy when stitched-equity drawdown exceeds N % from rolling HWM; resume after recovery. Cuts the tail of bad scenarios at the cost of missing some recoveries. Standard portfolio-risk-manager halt machinery already exists in `titan/risk/`.

None of these is part of the migration scope — they're separate research threads. Pick which to pursue based on whether tax-free SB is worth a 4-6 month detour vs. keeping IBKR.

### Phase R2 — Build adapter skeleton (titan/adapters/ig/)
- Implement auth + http_client + provider + read-only data client (warmup parquets via `/prices`)
- No execution path yet
- Reuse OANDA adapter's `factories.py` shape
- [ ] **GATE:** Unit tests pass; can pull bars from IG and feed them into a NautilusTrader paper backtest harness (loopback exec).

### Phase R3 — Execution path (paper account only)
- Add `execution.py` and `lightstreamer_client.py`
- Implement TRADE stream → fill reconciliation
- Add `convert_notional_to_bet_per_point` + parity test
- Wire IG demo credentials into `scripts/run_portfolio.py` behind a `--broker ig` flag
- [ ] **GATE:** A 1-week IG demo paper run with a single strategy (`bond_equity_ihyu_cspx_ig`) trades cleanly, all fills reconciled, no orphan positions, halt.json works end-to-end.

### Phase R4 — Multi-strategy paper
- Add the other two strategies (`bond_equity_ihyg_vusd_ig`, `bond_equity_ihyg_eimi_ig`)
- 4-week paper run with full champion portfolio (incl. existing IBKR strategies unchanged)
- [ ] **GATE:** P&L on IG paper matches the IBKR shadow portfolio (CSPX/VUSD/EIMI legs) within 5% drift attributable to cost differences. If drift > 10%, investigate before going live.

### Phase R5 — Live migration
- Generate live IG API key, populate `.env.ig.live`
- Cut over one strategy at a time over 3 weeks (smallest size first)
- Existing IBKR positions wound down opposite to IG opening (delta-neutral hand-off, not a same-day swap)
- [ ] **GATE:** Phase R4 paper passed; user explicit go/no-go.

---

## 6. Open questions for the user

> [!IMPORTANT]
> **Q1 (Phase R0 verification) — RESOLVED.** Live scout confirmed all 5 underlyings available 1:1. Demo was a restricted subset.
>
> **Q2 — RESOLVED in part.** No need to collapse — `cspx` (KA.D.CSPXLN) and `vusd` (KA.D.VUSDLN) remain distinct epics with distinct liquidity profiles. They both track S&P 500 so the **inter-strategy correlation may rise** from the recorded 0.52 to ~0.7-0.9. Empirically verifiable after paper trading. **Sub-question remaining:** are you OK with that higher correlation, or do you prefer dropping one of the two (and which)?
>
> **Q3 (currency).** Account is GBP-base; strategies are USD-quoted. Confirm preference: keep USD-quoted with FX plumbing in the adapter (option B in §1.4) or rebase strategies to GBP-quoted? **Recommendation: keep USD-quoted**, since the underlying ETFs are dollar-denominated and the existing research stack is USD. The adapter does the GBP→USD conversion at trade boundary, mirroring how `mr_audjpy` converts JPY→USD today.
>
> **Q4 (architecture) — RESOLVED.** No hybrid needed. IG provides both signal and execution data.
>
> **Q5 (Lightstreamer client).** Choice between:
> - Official `lightstreamer-client` PyPI package (sync, ~1.5 MB native deps, stable) — recommended; bridge to async via dedicated thread + `asyncio.Queue`
> - Community `lightstreamer-py` / `pylightstreamer` async wrappers — fewer deps but less battle-tested
>
> Need your go-ahead before adding to `pyproject.toml`.
>
> **Q6 (NEW, cost-sensitivity).** Given the 0.24-0.32 % spread on IHYG / IHYU / EIMI and ~5 %/yr financing, the original Sharpe +1.64 of `ihyu_cspx` will compress significantly. **Are you committed to migrating only if the Phase R1 WFO clears the deployment gate** (95 % bootstrap CI lo > 0 post-cost), even if that means only 1 of 3 strategies survives? Or is the tax-free benefit alone worth deploying at lower confidence?

---

## 7. Risks captured but not blocking

- **Daily allowance on `/markets/{epic}`** — production must cache the instrument map (epic → dealing rules). Cache file: `data/ig_instruments.json`, refreshed weekly by a cron-style script.
- **Demo ≠ live exec semantics** — IG demo fills almost always at touch price, no rejections. Live can reject on volatility (`error.confirm.deal.rejected.no-price`). Adapter must handle reject pathway distinctly from "delayed confirm".
- **Lightstreamer reconnect storm** — official client backs off automatically but during IG's daily 22:00 UTC maintenance window streams disconnect en masse. Strategy must tolerate ~5-minute streaming gap without flatten.
- **AUTO_RESTART_TIME consideration** — current `docker-compose.yml:22` sets IB gateway restart `Sun 03:00 America/New_York`. IG has no analogous container — sessions just persist until token expiry. Watchdog needs to detect session expiry (HTTP 401 with `error.security.client-token-invalid`) and re-auth.
- **Settlement / dividend adjustments** — for the US 500 cash SB, IG applies a dividend adjustment on the ex-date of underlying SPX components. This is small (few bps/yr) but is a P&L line that the existing `bond_equity` strategies don't model. Need to add to the cost model in Phase R1.

---

## 8. What was NOT changed by this research

To keep the surface narrow and reviewable:
- No code under `titan/` was touched. Strategy modules, PRM, allocator, equity tracker — all unchanged.
- No config under `config/` was added or modified.
- No registry changes in `scripts/run_portfolio.py`.
- Docker compose unchanged.
- The only new files are the read-only scout and analyser scripts (both in `scripts/`), this directive, and an updated `.gitignore` rule for `.env.ig*`.

The repository is in the same state as before this research with respect to anything that could affect live trading.

---

## Appendix A — Scout output reference

- Raw catalogue dump: `.tmp/ig_catalog/20260512T114959Z/` (initial scout, 13 search queries, 195 epic details attempted, ~120 successful before daily allowance)
- Supplementary scout: `.tmp/ig_catalog/20260512T115918Z/` (10 search queries focused on US-listed ETFs)
- Combined mapping digest: `.tmp/ig_catalog/MAPPING.md`
- Each `search_*.json` is the raw response from `GET /markets?searchTerm=`
- Each `epic_*.json` is the raw response from `GET /markets/{epic}` (some are 106-byte error stubs from the daily-allowance limit)

## Appendix B — Confirmed IG epics (live-verified 2026-05-12)

### Strategy-required epics (use these in config)

| Epic | Name | Role | Native ccy | Tier-1 margin | Min stake | Live spread |
|---|---|---|---|---|---|---|
| `KA.D.CSPXLN.DAILY.IP` | iShares Core S&P 500 UCITS ETF - CSPX | `ihyu_cspx` POSITION | USD | 20 % | £0.01/point | 0.024 % |
| `KA.D.VUSDLN.DAILY.IP` | Vanguard S&P 500 UCITS ETF - USD | `ihyg_vusd` POSITION | USD | 20 % | £0.01/point | 0.12 % |
| `KA.D.EIMILN.DAILY.IP` | iShares Core MSCI EM IMI UCITS ETF (USD line) | `ihyg_eimi` POSITION | USD | 20 % | £0.01/point | 0.32 % |
| `KA.D.IHYULN.DAILY.IP` | iShares $ HY Corp Bond UCITS ETF - USD | `ihyu_cspx` SIGNAL | USD | 20 % | £0.01/point | 0.24 % |
| `KA.D.IHYGLN.DAILY.IP` | iShares Euro HY Corp Bond UCITS ETF - IHYG | `ihyg_vusd` + `ihyg_eimi` SIGNAL | EUR | 20 % | £0.01/point | 0.24 % |

All five: `type=SHARES`, `expiry=DFB`, `streaming=True`, `slippageFactor=100% (POSITION)`, `country=GB`.

### Alternative listings (do NOT use these unless you know why)

| Epic | Name | Why NOT to use |
|---|---|---|
| `KA.D.EMIMLN.DAILY.IP` | iShares Core MSCI EM IMI UCITS ETF (**GBP line**, price ~3990) | GBP-overlayed share class — embeds GBP/USD into returns; conflicts with USD-quoted strategy research. Use `KA.D.EIMILN` (USD line) instead. |
| `KA.D.VUSALN.DAILY.IP` | Vanguard S&P 500 UCITS ETF - GBP | GBP-overlayed share class of VUSA. Use `KA.D.VUSDLN` (USD line) for VUSD strategy. |
| `EG.D.CSPXNA.DAILY.IP` / `EG.D.VUSANA` / `EG.D.EMIMNA` | EUR-listed lines (Xetra/SIX) of the same UCITS funds | Different exchange listings, EUR-quoted. Use LSE USD lines for all USD-quoted strategies. |
| `KA.D.HYGULN.DAILY.IP` | iShares EUR High Yield Corp Bond UCITS ETF - HYGU | USD-hedged share class of the IHYG fund (price ~730 vs IHYG ~9120, ratio ≈ 12×). NOT a substitute for IHYU. We already use IHYG directly. |

## Appendix C — IG markets that look right but ARE NOT useful

For future-proofing this document — these came up in scout results and could be confusingly named:
- `KA.D.SEMSLN.*` — iShares MSCI EM **Small Cap** UCITS — different exposure profile to EIMI/IMI; skip
- `SI.D.SPYDAILY.*` — "SPDR S&P 500 ETF Trust - **Daily (100x Leverage)**" — extreme leverage product; do not use
- `SI.D.IBHBUS.*`, `SI.D.IBHCUS.*` — iShares iBonds **2022/2023 Term** HY — term-maturity products that have already matured
- `SC.D.HYTRUS.*`, `SI.D.DHYRUS.*` — closed-end fund **rights issues**, not the underlying funds
- `SI.D.WYDEUS.*` — ProShares CDS **Short** NA HY (inverse / short product)
- `UB.D.HYNDUS.*` — WisdomTree **Negative Duration** HY (interest-rate hedged, not vanilla)
