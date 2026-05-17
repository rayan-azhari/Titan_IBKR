# Capital Flow Architecture — Account → Trade (2026-05-17)

How a dollar in the IBKR account becomes a contract or share in a live position.
Every stage is grounded to a code pointer; nothing here is aspirational.

```
IBKR account NLV
  │
  └─► Seed:   weight × portfolio_size           (run_portfolio.py startup)
        │
        └─► StrategyEquityTracker.initial_equity (per strategy, base ccy = USD)
              │
              └─► current_equity = initial + realized_pnl + MTM      (every bar)
                    │
                    └─► PortfolioRiskManager.update(prm_id, equity, ts)
                          │
                          ├─► dd_scale       (HWM-based, 0.25–1.0, hard halt at 15%)
                          ├─► vol_scale      (target 12% ann / realised, clipped 0.25–2.0)
                          └─► regime_scale   (min(VIX_tier, ATR_tier), 0.25–1.0)
                                │
                                └─► scale_factor = min(dd, vol, regime)
                                      │
                                      └─► allocator_weight           (inverse-vol, ρ-penalty)
                                            │
                                            └─► scaled_NAV = equity × weight × scale_factor
                                                  │
                                                  └─► strategy sizing      (per-class rule)
                                                        │
                                                        └─► convert_notional_to_units
                                                              │
                                                              └─► order delta vs current position
                                                                    │
                                                                    └─► IBKR
```

There are **seven** stages. Each is enforced by code referenced inline.

---

## Stage 1 — Seed capital assignment

Source: `scripts/run_portfolio.py` startup; per-strategy entry in `STRATEGY_REGISTRY`
contains an explicit `weight` and the runner multiplies by `--portfolio-size` to get
that strategy's `initial_equity` in USD.

Key contract: **a strategy's initial_equity is not the account NLV**. Two strategies
running in the same `TradingNode` see different seed values. This is what makes the
inverse-vol allocator meaningful — see V3.6 lesson on the `balance_total(ccys[0])`
defect (April 2026 rewrite of `titan/risk/`).

The seed flows into:

```python
self._equity_tracker = StrategyEquityTracker(
    prm_id=self._prm_id,
    initial_equity=config.initial_equity,
)
portfolio_risk_manager.register_strategy(self._prm_id, config.initial_equity)
```

[`titan/risk/strategy_equity.py:54-70`](titan/risk/strategy_equity.py#L54-L70).

---

## Stage 2 — Per-bar equity update

Every strategy calls the standard contract on every bar:

```python
_, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
if halted:
    self._flatten(); return
portfolio_allocator.tick(now=bar_ts.date())
```

[`titan/risk/strategy_equity.py:192-232`](titan/risk/strategy_equity.py#L192-L232).

`current_equity()` returns `initial_equity + realized_pnl_base + mtm_base`
([`strategy_equity.py:91-93`](titan/risk/strategy_equity.py#L91-L93)). It is the
strategy's **private** book value in USD — not the IBKR account NLV.

The PRM then appends the sample to a per-strategy `pd.Series` keyed by `bar.ts_event`
([`portfolio_risk_manager.py:102-117`](titan/risk/portfolio_risk_manager.py#L102-L117))
and resamples to business-day for the vol / correlation pipeline.

---

## Stage 3 — Portfolio scale factor (PRM)

`PortfolioRiskManager._check_portfolio_health` recomputes
`scale_factor = min(dd_scale, vol_scale, regime_scale)` on every PRM update
([`portfolio_risk_manager.py:510-547`](titan/risk/portfolio_risk_manager.py#L510-L547)).

### dd_scale (heat brake + kill switch)

| Portfolio DD | Action |
|---|---|
| `> -10%` | `dd_scale = 1.0` |
| `-10%` to `-15%` | `dd_scale ∈ [0.25, 1.0]`, linear in DD heat |
| `< -15%` | **HALT**, scale = 0, write `.tmp/portfolio_halt.json`, never auto-resets |

Operator must `reset_halt(operator=...)` to resume.
[`portfolio_risk_manager.py:519-537`](titan/risk/portfolio_risk_manager.py#L519-L537).

### vol_scale (portfolio volatility target)

`vol_scale = target_vol / realised_vol`, clipped to `[0.25, 2.0]`.

- Target: 12% annualised (config default).
- Realised: EWMA(λ=0.94) of *portfolio-level* daily returns, recomputed once per
  calendar day ([`portfolio_risk_manager.py:465-472`](titan/risk/portfolio_risk_manager.py#L465-L472)).

### regime_scale (VIX & ATR tiers)

`regime_scale = min(vix_scale, atr_scale)`:

| VIX | scale |  | ATR pct | scale |
|---|---|---|---|---|
| `< 17.8` | 1.00 |  | `< 25` | 1.25 |
| `17.8–23.1` | 0.75 |  | `25–75` | 1.00 |
| `23.1–30.0` | 0.50 |  | `75–90` | 0.50 |
| `> 30.0` | 0.25 |  | `> 90` | 0.25 |

[`portfolio_risk_manager.py:476-506`](titan/risk/portfolio_risk_manager.py#L476-L506).

VIX / ATR are fed in externally via `prm.set_vix(level)` / `prm.set_atr_percentile(sym, pct)`;
absent inputs default to `scale = 1.0`.

---

## Stage 4 — Inverse-vol allocator weight

`PortfolioAllocator.get_weight(prm_id)` returns this strategy's slice of the portfolio
([`portfolio_allocator.py:47-204`](titan/risk/portfolio_allocator.py#L47-L204)).

```
σ_i = √( EWMA(λ=0.94) of daily-return variance ) × √252
w_i = (1 / σ_i) / Σ_j (1 / σ_j)
```

Then:

1. **Clip** each weight to `[0.05, 0.60]`.
2. **Correlation penalty**: for any pair `(i, j)` with `|ρ_ij| > 0.70` on aligned daily
   returns, both weights × 0.9.
3. **Renormalise** to sum to 1.

[`portfolio_allocator.py:130-171`](titan/risk/portfolio_allocator.py#L130-L171).

### Rebalance schedule

`tick(now=...)` is called every bar but is **wall-clock gated**:
the rebalance only fires when `(today − last_rebalance_date).days ≥ 21`
business days ([`portfolio_allocator.py:65-84`](titan/risk/portfolio_allocator.py#L65-L84)).

Between rebalances, `get_weight()` returns the cached weight.

### Minimum history

Strategies with `< 30` daily return samples are excluded from the inverse-vol
calculation and receive a temporary equal weight until they have enough history.

---

## Stage 5 — Strategy-level sizing (per-class)

Each strategy receives:

```python
scaled_NAV = current_equity × allocator.get_weight(prm_id) × prm.scale_factor
```

Then translates `scaled_NAV` into a target notional / quantity using its
strategy-class sizing rule. The rules are **not** unified — they reflect different
research conventions and are individually parity-tested against research math
(audit A10).

| Strategy | Class | Sizing formula | Code |
|---|---|---|---|
| **GEM J5** | DAILY_TREND (rotation) | `qty = target_w[asset] × scaled_NAV / price` (per-leg) | [`gem/strategy.py:350-379`](titan/strategies/gem/strategy.py#L350-L379) |
| **turtle CAT** | DAILY_TREND (breakout) | `qty = equity × 0.01 / (2 × ATR)` — risk-target, NOT scaled by allocator/PRM | [`turtle/strategy.py:28-29`](titan/strategies/turtle/strategy.py#L28-L29) |
| **samir_stack** | CROSS_ASSET_MOM + overlay | Equity: `scaled_NAV × eq_w × tier / native_lev`; Bond: `scaled_NAV × bond_w` | [`samir_stack/strategy.py:468-470`](titan/strategies/samir_stack/strategy.py#L468-L470) |
| **bond_gold** | CROSS_ASSET_MOM | `notional = scaled_NAV × vol_target / realised_vol(20d EWMA)` (10% ann target) | [`bond_gold/strategy.py:36-243`](titan/strategies/bond_gold/strategy.py#L36-L243) |
| **etf_trend** | DAILY_TREND | `notional = scaled_NAV × decel_composite` (continuous) or binary; MOC fill | [`etf_trend/strategy.py:274-341`](titan/strategies/etf_trend/strategy.py#L274-L341) |

> [!IMPORTANT]
> Turtle is the only live strategy that sizes by **stop-distance**, not by allocator
> weight. Its equity bucket is still tracked in the PRM (so DD / correlation still
> apply), but its quantity formula bypasses `allocator.get_weight()` and
> `prm.scale_factor`. This is intentional — turtle's pre-reg (L61) defines the
> 1%-per-unit ATR sizing as the canonical math.

---

## Stage 6 — Notional → units (FX conversion)

The base currency of the portfolio is USD. Instruments may price in USD, in the
account ccy (e.g. AUD for AUD/USD), or in a third currency (e.g. JPY for AUD/JPY).

The single conversion helper:

```python
units = convert_notional_to_units(
    notional_base,                       # USD
    price,                               # quote ccy per unit
    quote_ccy=...,                       # e.g. "JPY"
    base_ccy="USD",
    fx_rate_quote_to_base=...,           # REQUIRED when quote != base
)
```

[`titan/risk/strategy_equity.py:125-171`](titan/risk/strategy_equity.py#L125-L171).

Rules:

- `quote_ccy == base_ccy`  → `int(notional_base / price)`.
- `quote_ccy != base_ccy` and `fx_rate_quote_to_base is None`  → **raises ValueError**.
  Silent `notional / price` divisions are forbidden (V3.6 audit fix).
- Otherwise: `notional_quote = notional_base / fx_rate_quote_to_base`, then
  `int(notional_quote / price)`.

---

## Stage 7 — Reconciliation to existing position

Strategies never submit raw "buy N" orders directly off a signal. They compute a
target quantity and submit `delta = target_qty − current_qty` only.

GEM example ([`gem/strategy.py:381-429`](titan/strategies/gem/strategy.py#L381-L429)):

```python
delta = wanted_qty - current_qty
if delta == 0: continue
if relative_delta < rebalance_threshold_weight / weight_now: continue  # buffer
submit(MarketOrder(delta))
```

The **rebalance buffer** (`rebalance_threshold_weight`) suppresses sub-threshold
orders that the vol-target / decel-scaling would otherwise emit every bar.

Exits go through the same path: when a signal sets `target_w[leg] = 0`, the next
reconciliation produces `delta = -current_qty`, flattening the leg.

Hard exits — portfolio halt, drawdown kill, strategy-local stop hit — call
`self.close_all_positions(inst_id)` directly without going through the
reconciliation buffer.

---

## Halt persistence

When `dd_scale = 0` fires, the PRM writes `.tmp/portfolio_halt.json`. Process restart
re-reads it ([`portfolio_risk_manager.py:519-529`](titan/risk/portfolio_risk_manager.py#L519-L529)).
**No strategy can take a new position while halted**, even after a fresh process
spawn. Operator must explicitly:

- `reset_halt(operator="<name>")` — resumes trading, preserves pre-halt HWM.
- `reset_hwm(operator="<name>")` — re-anchors HWM to current equity (use after
  intentional capital change).

---

## Worked example — V3.7 live portfolio (`v37_live` set)

Account: USD 100,000. Two strategies: `gem_j5_canonical` (weight=0.7),
`turtle_cat_c3peak` (weight=0.3).

| Stage | GEM J5 | turtle CAT |
|---|---|---|
| Seed (`initial_equity`) | 70,000 | 30,000 |
| Per-bar equity (after some P&L) | 71,500 | 29,200 |
| `dd_scale` | 1.0 (port DD = -1.3%) | 1.0 |
| `vol_scale` | 1.0 (realised ≈ target) | 1.0 |
| `regime_scale` | 0.75 (VIX = 19.4) | 0.75 |
| PRM `scale_factor` | 0.75 | 0.75 |
| Allocator weight | 0.62 (GEM higher vol → lower weight, post-ρ penalty) | 0.38 |
| `scaled_NAV` | `71,500 × 0.62 × 0.75 = 33,247` | — (bypassed) |
| Sizing | per-leg: SPY 50% → 16,624 USD → 38 sh @ 437 | risk-target: `29,200 × 0.01 / (2 × ATR)` |
| Units after FX | 38 SPY shares | 7 CAT shares |
| Order | `delta = 38 - 35 = +3` (rebalance buffer permitting) | `delta = +7` (new unit, pyramid #1) |

---

## What is **not** in this system today

These are deliberately absent and any proposal to add them must be a separate
directive:

- **No gross-risk-at-stop ceiling.** There is no global "total open risk-at-stop
  ≤ X% of NAV" gate. Each strategy enforces its own internal stops; the PRM only
  sees mark-to-market equity, not stop-loss exposure.
- **No max-concurrent-positions cap.** A strategy can hold any number of legs
  simultaneously; the portfolio can run any number of strategies. Concentration
  is implicit via `max_weight = 60%` and the correlation penalty, not explicit.
- **No per-trade fixed-fractional sizing** (except turtle, by class definition).
  GEM / samir / bond_gold / etf_trend are all position-target, not risk-target.
- **No coupling between strategies' order submission.** Two strategies firing on
  the same bar do not see each other's intended order flow. Cross-strategy net
  exposure is observable in the IBKR account but not in the sizing math.

These four gaps are exactly the design space for the "2% per trade × max 10
concurrent" proposal (see chat history 2026-05-17). They are documented here so
that any future overlay knows what it is overlaying.

---

## File index

| Concern | Path |
|---|---|
| Per-strategy equity ledger | [`titan/risk/strategy_equity.py`](titan/risk/strategy_equity.py) |
| Portfolio aggregation + scale factors + halt | [`titan/risk/portfolio_risk_manager.py`](titan/risk/portfolio_risk_manager.py) |
| Inverse-vol allocator | [`titan/risk/portfolio_allocator.py`](titan/risk/portfolio_allocator.py) |
| Risk config defaults | `config/risk.toml` (`[portfolio]` section) |
| Multi-strategy runner | [`scripts/run_portfolio.py`](scripts/run_portfolio.py) |
| Halt-state file | `.tmp/portfolio_halt.json` |
| Architecture review (V3.7) | [`directives/V3.7 Multi-Strategy Live Architecture 2026-05-17.md`](directives/V3.7%20Multi-Strategy%20Live%20Architecture%202026-05-17.md) |
| Portfolio-risk rewrite (April 2026) | skill `references/portfolio-risk-architecture.md` |
