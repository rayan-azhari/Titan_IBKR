# IBKR UK Paper Account — Champion Portfolio Compatibility

> Verdict: **all three instruments are available on IBKR UK paper. No
> backtests need re-running. The work is account configuration, not
> code or research.**
>
> Cross-references: `directives/Docker Paper Trading Guide.md` (how to run
> the stack), `directives/Deployment & Operations.md` (canonical pre-flight
> checklist).

---

## 1. Why the question matters

UK retail clients are subject to **PRIIPs / MiFID II disclosure rules**: a
broker may not solicit retail trades on a fund that does not publish a Key
Information Document (KID) in an EU/UK-recognised format. In practice this
blocks UK retail from buying virtually all US-domiciled ETFs (SPY, IWB, HYG,
IEF, GLD, etc.), even on IBKR. Forex and individual stocks are unaffected.

The champion portfolio's instrument choice was **already made with this in
mind** (see comment at [scripts/run_portfolio.py:202-204](scripts/run_portfolio.py)):

> `IHYU.LSEETF -> CSPX.LSEETF cross-asset (UCITS substitute for HYG -> IWB).`
> `Original HYG/IWB blocked by EU/UK PRIIPs (no KID for US-domiciled ETFs).`

So the swap from HYG/IWB → IHYU/CSPX has already happened. This document
verifies the choice still holds for paper trading and surfaces the
account-side configuration steps that are easy to miss.

---

## 2. Per-instrument verdict

### 2.1 `AUD/JPY.IDEALPRO` (mr_audjpy strategy)

| Field | Value |
|---|---|
| Contract | `secType="CASH", symbol="AUD", exchange="IDEALPRO", currency="JPY"` |
| Asset class | Spot forex |
| IBKR UK paper available? | **YES** — IDEALPRO is a global IBKR venue |
| Regulatory blocker? | None — forex is not a PRIIPs concern |
| Trading permission needed | "Forex" in Client Portal → Trading → Trading Permissions |
| Currency to enable | JPY (otherwise IBKR will auto-FX-convert and ruin sizing) |
| Realtime data on paper? | Yes — IDEALPRO data is included in IBKR Pro accounts (which all UK accounts default to) |
| Bar subscription used | `AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL` (H1 mid) |

**Verdict: works as-is on IBKR UK paper.** Just enable the Forex
permission and confirm JPY is in the account's traded currencies list.

### 2.2 `CSPX.LSEETF` (bond_equity_ihyu_cspx — traded leg)

| Field | Value |
|---|---|
| Contract | `secType="STK", symbol="CSPX", exchange="SMART", primaryExchange="LSEETF", currency="USD"` |
| Full name | iShares Core S&P 500 UCITS ETF (Acc) |
| ISIN | `IE00B5BMR087` |
| Domicile | Ireland (UCITS) |
| Listing | London Stock Exchange, ETF segment (`LSEETF`), USD-denominated |
| TER | 0.07% |
| AUM | Largest S&P 500 UCITS ETF |
| KID for UK retail? | Yes |
| IBKR UK paper available? | **YES** |
| Trading permission needed | "Stocks" with UK market access (or "All Europe") |
| Bar subscription | `CSPX.LSEETF-1-DAY-LAST-EXTERNAL` (daily last) |
| Realtime data sub needed | "UK Securities Exchange (LSE)" — ~£3-4/month for retail level 1, OR accept free delayed bars (acceptable for daily strategy) |

**Verdict: works as-is. UCITS-compliant, KID-documented, deliberately
chosen as the UK-tradable substitute for IWB/SPY.**

### 2.3 `IHYU.LSEETF` (bond_equity_ihyu_cspx — signal leg)

| Field | Value |
|---|---|
| Contract | `secType="STK", symbol="IHYU", exchange="SMART", primaryExchange="LSEETF", currency="USD"` |
| Full name | iShares $ High Yield Corp Bond UCITS ETF (Dist) |
| ISIN | `IE00B4PY7Y77` |
| Domicile | Ireland (UCITS) |
| Listing | London Stock Exchange, ETF segment (`LSEETF`), USD-denominated |
| TER | 0.50% |
| KID for UK retail? | Yes |
| IBKR UK paper available? | **YES** |
| Trading permission needed | Same as CSPX — UK Stocks |
| Bar subscription | `IHYU.LSEETF-1-DAY-LAST-EXTERNAL` (daily last) |
| Realtime data sub needed | Shared LSE feed with CSPX — one subscription covers both |

> [!IMPORTANT]
> IHYU is the **signal-only** leg of the strategy — the system reads its
> daily bars to compute a momentum z-score but never sends an order on
> IHYU. So even if IHYU were marginally restricted in some scenario, the
> strategy would still place orders only on CSPX. (It is not restricted
> — just worth knowing the dependency.)

**Verdict: works as-is.**

---

## 3. Account-configuration checklist

Run through these *before* starting the Docker stack against your IBKR UK
paper account. None require a fresh paper account if your existing one
is on the IBKR UK entity (`IBUK` / `IBIE`); they're permission toggles.

### 3.1 Confirm your paper account is on the UK entity

Log into the **paper** Client Portal at
<https://www.interactivebrokers.com/portal/?loginType=2>. Top of the
landing page shows the entity (e.g. "Interactive Brokers (U.K.) Limited"
or "Interactive Brokers Ireland Limited"). If your live account is US
(`IBLLC`), your paper account is also US — you would need to open a new
UK live account to get a UK paper account.

> [!CAUTION]
> A US-entity paper account would let you "trade" CSPX/IHYU in
> simulation, but the corresponding live account could not place those
> orders due to PRIIPs. Paper that doesn't reflect live constraints is
> worse than no paper test. **Verify entity match before proceeding.**

### 3.2 Enable trading permissions

In the paper Client Portal: **Settings → Account Settings → Trading
Permissions**. Add:

- **Stocks** for at least:
  - **United Kingdom** (covers `LSE` and `LSEETF`)
  - Optionally **All Europe** for future flexibility
- **Forex** (covers `IDEALPRO`)

Permission updates typically apply within a few hours, occasionally up
to 24 hours. Paper permissions are typically copied from live, so if
live already has these, paper inherits them.

### 3.3 Enable JPY as a tradable currency

For the AUD/JPY pair, the account must have **JPY** explicitly enabled
under **Settings → Account Settings → Configure Account → Trading
Permissions → Currencies**, otherwise IBKR will auto-convert orders
through the base currency and the live position size won't match what
the strategy intended.

### 3.4 (Optional) Subscribe to LSE realtime data

For paper testing the daily bond-equity strategy, **delayed data is
fine** — the strategy fires once per day on the close, so delays of
15-20 minutes are immaterial. If you later want realtime quotes for
monitoring or for going live, subscribe to:

- **UK Securities Exchange (LSE)** — ~£3.27/month (waived if monthly
  commissions exceed a threshold)

The forex strategy needs no extra subscription — IDEALPRO data is
included in IBKR Pro by default.

### 3.5 Verify market-data API acknowledgement

In Client Portal: **Settings → User Settings → Market Data API
Acknowledgement**. This must be accepted once or all API market-data
requests return errors. Easy to miss; the symptom is the strategy
logging "market data not subscribed" with no other clue.

---

## 4. What is NOT changing

- **No code changes** to `titan/strategies/`.
- **No config changes** to `config/*.toml`.
- **No backtest re-runs as part of this account setup.** Same instruments →
  same historical returns. Note however that under the V2.0 cleanse all V1-era
  Sharpe figures were declared untrustworthy — each strategy still needs a
  framework-audited pre-reg + result log before its Sharpe/CI claims are
  deployable. See `directives/README V2.0.md`.
- **No data downloads.** The existing `data/CSPX_D.parquet` and
  `data/IHYU_D.parquet` warmup files are still authoritative; live
  bars stream in over the IBKR socket.

---

## 5. Open considerations (flagged, not blocking)

### 5.1 Account base currency

The strategy's `BondGoldStrategy` config sets `base_ccy="USD"` and
holds CSPX/IHYU which are USD-denominated. If your **paper account base
currency is GBP**, IBKR will auto-convert P&L in/out of GBP, which:

1. introduces an FX P&L that's not part of the strategy's edge, and
2. complicates the per-strategy equity tracker which assumes USD.

Option A: change paper account base currency to USD (one-click in
Client Portal, but requires holding USD cash to fund — IBKR auto-funds
paper accounts but the currency is fixed at account creation).

Option B: leave base as GBP and accept that paper P&L will be a noisy
signal of strategy P&L. Acceptable for a reliability test of the *stack*
(the question we're answering with paper); not acceptable for a final
strategy validation.

**Recommendation:** Option A if you can choose at account creation.
Otherwise live with Option B for plumbing tests and re-validate strategy
edge once on live USD account.

### 5.2 IDEALPRO minimum order size

IBKR enforces a minimum order size on forex pairs (typically $25k
notional or equivalent) to discourage retail tick-scalping. The
`mr_audjpy` strategy's vol-targeted sizing typically clears this on a
$10k paper account, but at very low vol the size could fall under the
minimum and the order would reject. The strategy should already log
this as `order rejected: below minimum size` if it happens — watch for
it during the first session.

### 5.3 Going live in the UK requires same-region live account

The Docker stack works identically against IBKR UK live as paper. But
to flip `TRADING_MODE=live` you need an actual IBKR UK live account
with appropriately funded USD (or accept GBP→USD auto-conversion).
This is an account-opening process, not a code change.

---

## 6. Summary in one paragraph

The champion portfolio's three instruments — `AUD/JPY.IDEALPRO`,
`CSPX.LSEETF`, `IHYU.LSEETF` — are all tradable on IBKR UK paper as-is.
The CSPX/IHYU choice was deliberately made to substitute UK-blocked
US-listed ETFs (HYG, IWB) with their UCITS-compliant LSE-listed
equivalents; this work is already done. The only remaining tasks are
account-side: confirm the paper account is on the IBKR UK entity (not
US), enable Stocks-UK + Forex trading permissions, enable JPY as a
currency, and accept the Market Data API acknowledgement. No backtest
re-runs are needed because no instruments are changing.
