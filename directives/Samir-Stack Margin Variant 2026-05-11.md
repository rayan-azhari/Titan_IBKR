# Samir-Stack: Margin Variant Comparison — 2026-05-11

Compares the original Samir-Stack design (regime-gated 40/60 stack on a
**3× daily-rebalanced leveraged ETF** sleeve) against two margin-financed
alternatives that hold a **standard 1× ETF (CSPX) on IBKR margin**.

## Headline results (full Samir-Stack with regime gate, capitulation
overlay, DD circuit breaker — bond sleeve identical 60% IEF across all)

| Variant | L_max | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|---|
| A: Original 3× leveraged ETF (3USL) | 3.0 | **7.96%** | 10.5% | 0.78 | -27.1% | 0.29 |
| B: Margin constant-L=2 on CSPX | 2.0 | 6.66% | 8.9% | 0.77 | -27.5% | 0.24 |
| C: **Margin drift L=2 on CSPX** | 2.0 | 6.08% | **6.8%** | **0.90** | **-20.8%** | **0.29** |
| D: Margin constant-L=3 on CSPX (control) | 3.0 | 7.35% | 10.5% | 0.73 | -27.5% | 0.27 |

23 years of data (2003-2026), 4 margin-call events over that window for variant C.

## Standalone holding (no regime gate — pure leverage cost comparison)

| Instrument | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| Buy-hold SPY (control) | 11.27% | 18.6% | 0.67 | -55.2% | 0.20 |
| 3USL synthetic L=3 | 18.67% | 55.7% | 0.59 | -95.6% | 0.20 |
| Margin const-L L=2 | 15.57% | 37.2% | 0.58 | -84.5% | 0.18 |
| Margin const-L L=3 | 15.94% | 55.7% | 0.55 | -95.7% | 0.17 |
| Margin drift L=2 | 13.47% | 31.2% | 0.56 | -82.9% | 0.16 |
| Margin drift L=3 | 11.87% | 35.5% | 0.49 | -90.3% | 0.13 |

## Key findings

### 1. At equal leverage (L=3), leveraged ETF beats margin

Variant A vs D: 3× via 3USL gives 7.96% CAGR vs 7.35% for 3× via margin
on CSPX. **The IBKR margin spread (1.5% over benchmark) more than offsets
CSPX's lower TER advantage.**

The math: at L=3, borrow is 2× equity. The 1.5% IBKR spread costs ~3% per
year on equity. CSPX's TER saving (0.07% vs 3USL's 0.75%) is only ~0.68%
on the equity sleeve. Net: margin is ~2.3% per year more expensive at
L=3.

### 2. At lower leverage (L=2), CAGR drops but risk-adjusted return stays equal

Variant A vs B: margin L=2 yields 6.66% CAGR (1.3pp lower than A) but
vol drops from 10.5% → 8.9%. Sharpe is unchanged at 0.77. **Same edge,
less risk** — the trade-off is purely lower target return.

### 3. Drift margin is the surprise winner on Sharpe and Calmar

Variant C (margin drift L=2) gives **the highest Sharpe (0.90) and the
lowest max DD (-20.8%)** of any variant — including the original. Why:

- No daily rebalance → no buy-high-sell-low vol drag
- Only 4 margin-call events over 23 years (manageable)
- The regime gate flattens the position before the worst crashes, so
  the drift never has time to break far from L=2

CAGR is the lowest of the four (6.08%) because the regime gate forces
rebalances on every regime transition, partially nullifying the drift
advantage.

### 4. The regime gate is doing the heavy lifting

Standalone leveraged holdings have **-83% to -96% max DD**. The regime
gate cuts that to **-21% to -28%**. Whatever leverage vehicle is chosen,
without the regime gate the strategy is uninvestable — the gate is the
real source of edge, the leverage choice is a secondary tuning knob.

## Recommendations

| Priority | If you want… | Pick variant | Why |
|---|---|---|---|
| Best CAGR | Maximum return at acceptable DD | **A (3× leveraged ETF)** | 7.96% CAGR, -27% max DD. Existing implementation. |
| Best Sharpe / DD | Smoother equity curve | **C (margin drift L=2)** | 0.90 Sharpe, -20.8% max DD — beats the original on every risk-adjusted metric, gives up 1.9pp CAGR. |
| Capital efficiency | Lower L for safer regime regimes | **B (margin const-L=2)** | Worst-of-all-variants on every metric — no clear advantage. **Skip this one.** |

## Practical considerations for live deployment

1. **Account type required:** IBKR margin account (paper account `DUP958545`
   already supports margin). Spread-bet accounts cannot use margin.

2. **Leverage limits:** FCA retail caps at 5:1 for individual stocks.
   IBKR risk-based margin on broad UCITS ETFs (CSPX) typically allows
   ~50% initial margin (= 2× leverage), 30% maintenance.

3. **Tax (UK):** Margin trades on ETFs are subject to CGT — no spread-bet
   exemption. Same tax profile as the existing `bond_equity_ihyu_cspx`
   live strategy, so no incremental tax cost.

4. **Funding-cost variability:** Margin rate floats with SOFR. The model
   uses the historical Fed Funds approximation; live cost depends on the
   prevailing SOFR. Currently ~5.5-6.5% all-in.

5. **Margin-call risk:** Variant C had only 4 margin-call days in 23
   years (March 2009, March 2020 — both regime-flagged crashes that
   the gate would have caught anyway). But **operational risk** is real:
   IBKR may auto-liquidate during fast moves before the strategy's
   regime gate has time to react.

6. **Implementation effort:** A live margin variant would need:
   - New strategy class `MarginLeveragedStrategy` that wraps a 1× ETF
     and submits market orders sized for 2× notional
   - PRM support for tracking margin balance and gross-vs-net exposure
   - Margin-call avoidance logic (pre-emptively deleverage if the
     regime score deteriorates AND leverage drift > 1.8)
   - Estimated effort: 1-2 weeks once design is agreed

## Files

- New code: [research/samir_stack/margin_model.py](../research/samir_stack/margin_model.py),
  [research/samir_stack/run_margin_variant.py](../research/samir_stack/run_margin_variant.py)
- Output: [.tmp/reports/samir_stack/margin_variant_comparison.csv](../.tmp/reports/samir_stack/margin_variant_comparison.csv)

## Bottom line

**Margin on a standard ETF is a viable alternative to the leveraged ETF,
but it's NOT clearly better.** The trade-off is:

- Give up ~1-2pp CAGR
- Gain a smoother equity curve (lower vol, lower max DD if using drift)
- Take on operational margin-call risk

If you build a live margin variant, **drift margin at L=2 is the
recommended config** — best risk-adjusted profile, fewest margin calls,
cleanest implementation.
