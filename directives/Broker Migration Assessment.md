# Broker Migration Assessment — IBKR → Spread Betting

**Last updated:** 2026-05-10
**Status:** Assessment only — no migration in progress
**Plan file:** `C:\Users\rayan\.claude\plans\review-the-current-workspace-gentle-curry.md`

## Why this exists

Evaluation of moving from Interactive Brokers (taxable CGT/income on profits) to a UK spread betting broker (tax-free profits in UK/Ireland). This document is the project-side reference; full assessment lives in the plan file linked above.

## TL;DR

| Broker | Best for | Main blocker |
|---|---|---|
| **OANDA** | Fast migration of FX strategies | No ETFs/stocks in spread betting account |
| **Pepperstone** | Stocks + ETFs available | cTrader API is gRPC/protobuf, no NautilusTrader adapter |
| **IG Markets** | Best instrument fit (CSPX/IGLT confirmed) | No NautilusTrader adapter, but `trading-ig` Python lib makes adapter build feasible |

**Companion repo:** [Titan-Oanda-Enhanced](https://github.com/rayan-azhari/Titan-Oanda-Enhanced) — pre-built NautilusTrader OANDA adapter using `oandapyV20`.

## Strategy survival matrix

| Strategy | Current | OANDA | Pepperstone | IG |
|---|---|---|---|---|
| `mr_audjpy` (live champion) | IBKR FX | ✅ direct | ✅ direct | ✅ direct |
| MTF EUR/USD | IBKR FX | ✅ | ✅ | ✅ |
| Gap Fade EUR/USD | IBKR FX | ✅ | ✅ | ✅ |
| IC MTF (6 FX pairs) | IBKR FX | ✅ | ✅ | ✅ |
| MR FX multi-pair | IBKR FX | ✅ | ✅ | ✅ |
| ORB (7 US stocks) | IBKR equities | ❌ DROP | ✅ tax-free | ✅ tax-free |
| IC Equity Daily | IBKR equities | ❌ DROP | ✅ tax-free | ✅ tax-free |
| ETF Trend (SPY/QQQ/GLD/IWB) | IBKR ETFs | index substitute | ✅ likely | ✅ likely |
| ML Classifier (QQQ/SPY) | IBKR ETFs | index substitute | ✅ likely | ✅ likely |
| **`bond_equity_ihyu_cspx`** (live champion) | IBKR LSE UCITS | ❌ | ❓ | **✅ CSPX confirmed; IHYU likely** |
| **Samir-Stack (3USL+IGLT)** | IBKR LSE UCITS | ❌ | ❓ | **✅ IGLT confirmed; 3USL likely** |
| Bond→Gold (IEF→GLD) | IBKR ETFs | T-Note substitute | ✅ likely | ✅ likely |
| FX Carry AUD/JPY | IBKR FX | ⚠️ economics change | ⚠️ economics change | ⚠️ economics change |

**FX carry warning:** In spread betting the broker absorbs positive carry into the spread; you do NOT receive rollover credits. The carry trade alpha source partially evaporates. Re-research required before deployment.

## Tech stack implications

### What stays (any broker)
- NautilusTrader, Python 3.11, uv, ruff
- Research stack (VectorBT, XGBoost, HMM, statsmodels)
- Docker deployment, TOML config system
- Portfolio risk manager (`titan/risk/portfolio_risk_manager.py`)
- All strategy logic in `titan/strategies/`

### What changes
| Component | Current (IBKR) | OANDA | Pepperstone | IG |
|---|---|---|---|---|
| Broker SDK | `python-ibapi-stable` | `oandapyV20` | `ctrader-open-api-py` | `trading-ig` |
| API protocol | TWS socket | REST/JSON | gRPC/protobuf | REST + Lightstreamer WebSocket |
| Auth | TWS + 2FA via gateway | API key | OAuth2 | API key + creds |
| Adapter location | `titan/adapters/ibkr/` | `titan/adapters/oanda/` (exists in companion repo) | `titan/adapters/pepperstone/` (build) | `titan/adapters/ig/` (build) |
| Docker | 2 containers (ib-gateway + portfolio) | 1 container | 1 container | 1 container |
| Instrument IDs | `EUR/USD.IDEALPRO` | `EUR_USD` | epic codes | epic codes |
| Build effort | (current) | 1–2 weeks | 4–8 weeks | 4–6 weeks |

## FCA retail leverage caps (apply to all UK spread betting)

- FX majors: 30:1
- FX minors / gold / major indices: 20:1
- Minor indices: 10:1
- Individual equities: 5:1
- Other commodities: 10:1
- Crypto: 2:1

**Current strategies that exceed retail caps:**
- `IC Generic` 20x leverage → illegal at retail level
- ORB 4x equity leverage → exceeds 5:1 individual equity cap
- Samir-Stack 3x via 3USL → can be replicated at 3:1 on US SPX 500 spread bet within 20:1 cap

Either obtain professional client status (volume/experience requirements) or redesign sizing for retail caps.

## Recommended path

**Option D — Hybrid (recommended):**
1. Migrate FX strategies to OANDA immediately using existing Titan-Oanda-Enhanced adapter (1–2 weeks to first paper trade, mr_audjpy as the test)
2. Build IG Markets NautilusTrader adapter in parallel on top of `trading-ig` Python lib (4–6 weeks)
3. Once IG adapter is ready, migrate equity/ETF/UCITS strategies (`bond_equity_ihyu_cspx`, `Samir-Stack`, ORB, ETF Trend, ML, IC Equity) to IG
4. Decommission IBKR connection only after both sleeves are stable on paper

## Verification checklist before committing to IG

- [ ] Log into IG spread betting demo and search for IHYU.L — confirm available as spread bet
- [ ] Search for 3USL.L — confirm available as spread bet
- [ ] Search for IEF, TLT, HYG (US-listed bond ETFs) — confirm spread bet availability
- [ ] Note IG's spread on CSPX vs IBKR's commission cost — wider spreads erode edge on daily-signal strategies
- [ ] Note minimum bet size per instrument — confirms position sizing won't conflict with risk-percent sizing
- [ ] Confirm IG REST API rate limits (100 trade requests/minute) accommodate full portfolio at peak

## Critical caveats

1. **Spread betting account ≠ spot FX/CFD account.** Even on OANDA the same v20 API works for both, but the account itself is created separately. Verify the account type in dashboard before assuming tax treatment.
2. **Wider spreads.** Spread betting accounts have wider spreads than spot FX/CFDs. Backtest costs need to be re-modelled for spread betting before deploying.
3. **No MOC orders on spread betting brokers.** ETF Trend's `Market-on-Close` execution pattern doesn't translate. Strategy needs to be modified to last-bar-close limit orders or similar.
4. **Backtest re-validation required.** Substituting IEF → US 10Y T-Note (or SPY → US SPX 500 index) requires re-running WFO. Backtest results from ETF data do NOT automatically transfer to index/bond spread bets.
5. **Third-party broker reviews are unreliable** on the spread bet vs CFD wrapper distinction. Always verify against the actual broker's product detail pages or your own account dashboard.

## Related documentation
- `directives/IBKR & NautilusTrader API Reference.md` — current adapter approach
- `directives/Docker Paper Trading Guide.md` — current 2-container Docker setup (would simplify to 1 container)
- `directives/Samir-Stack Strategy Guide.md` — strategy that depends on LSE UCITS ETFs (IG only)
- `directives/README V2.0.md` — current cleansed-state portfolio status
