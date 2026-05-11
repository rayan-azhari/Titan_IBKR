# Phase 5 — Cross-Continent Equity Lead-Lag — 2026-04-22

**Headline: 82 apparent Bonferroni survivors with max CI_lo +0.899
(GDAXI → QQQ). ALL are a timezone-overlap artefact.** The ^GDAXI and
^FTSE "close" prices are stamped at 11:30 AM NY (local market close,
normalised to date) but the return series for US targets is close-to-close
at 4 PM NY. The 2-hour US trading overlap between 9:30 AM and 11:30 AM NY
leaks into the "European close" signal, producing what looks like a
cross-continent prediction but is actually US-morning-momentum wrapped
in a foreign ticker.

> [!WARNING]
> This is a research-math failure, not a trading edge. Do not add any
> GDAXI/FTSE → US equity strategy to the v4 portfolio. The correct
> interpretation uses USD-listed country ETFs (EWG, EWU), which close at
> 4 PM NY same as SPY. EWG/EWU → US equity CI_lo caps at +0.36 — below
> the Bonferroni gate.

**Driver**: [scripts/rerank/run_phase5_cross_continent.py](../scripts/rerank/run_phase5_cross_continent.py)
**Raw**: [.tmp/reports/phase5_cross_continent_2026_04_22/results.csv](../.tmp/reports/phase5_cross_continent_2026_04_22/results.csv)

**Scope**: 2,304 combos (3 US × 4 EU × 2 directions × 6 lb × 4 hold × 4 th) in 179 s.

---

## The smoking gun — identical exposure, different close time

If the signal were a real EU → US effect, switching from `^GDAXI` (local
index close at 11:30 AM NY) to `EWG` (NYSE-listed Germany ETF, close at
4 PM NY) should produce similar Sharpes — both carry the same German
equity exposure. **Instead**:

| Signal | Underlying | Close time (NY) | Max CI_lo → US equity targets |
|---|---|---|---:|
| **^GDAXI** | German equity (local) | **11:30 AM NY** | **+0.899** |
| **EWG** | German equity (NYSE-listed) | **4:00 PM NY** | **+0.326** |
| Δ | same exposure | +4h 30min | **−0.573** |
| **^FTSE** | UK equity (local) | 11:30 AM NY | +0.503 |
| **EWU** | UK equity (NYSE-listed) | 4:00 PM NY | +0.356 |
| Δ | same exposure | +4h 30min | −0.147 |

The magnitude of the Germany gap (−0.57) is bigger than the UK gap (−0.15)
because German market hours overlap more tightly with US morning (Frankfurt
closes when SPY is up ~2 hours of trading), while London closes slightly
later and has less pure overlap.

### Why the timezone overlap creates a fake signal

Sequence of events on any given trading day:
1. **09:30 AM NY**: SPY opens. GDAXI is still trading.
2. **09:30 AM – 11:30 AM NY**: SPY trades for 2 hours. News drives both SPY AND the still-open GDAXI. They're **positively correlated** during this overlap period (global risk-on/off).
3. **11:30 AM NY**: GDAXI closes. Its "daily close" timestamp is stamped at this moment. **This close embeds the last 2 hours of SPY movement.**
4. **11:30 AM – 4:00 PM NY**: SPY continues for 4.5 more hours.
5. **4:00 PM NY**: SPY closes.

The WFO harness uses `signal.shift(1)` — so the position at bar `t` is
computed from GDAXI close(t-1). That close is at 11:30 AM NY on day t-1.
The return earned is SPY(close t-1) → SPY(close t), from 4 PM NY t-1 →
4 PM NY t.

**Causally**, the GDAXI close at 11:30 AM NY t-1 is known before SPY
closes at 4 PM NY t-1, so the position is causally valid. **But** the
signal's predictive power doesn't come from "EU leading US" — it comes
from the fact that GDAXI close(t-1) is ~= SPY(11:30 AM NY t-1) via
intraday correlation, and **SPY morning momentum (9:30 AM - 11:30 AM)
is a well-documented predictor of SPY afternoon (11:30 AM - 4 PM) within
the same day**, plus overnight to next day.

So the backtest is backing out SPY intraday momentum through the GDAXI
proxy. **It's real (the predictive power exists), but it's not a
cross-continent edge — it's US-intraday momentum.** And you can't
realise it at the backtest's implied Sharpe because to do so you would
need to enter SPY at 11:30 AM NY (when the GDAXI close is first
available), not at 4 PM NY close as the backtest pretends.

---

## Direction summary

| Direction | Configs | Max Sharpe | Max CI_lo | Bonf survivors |
|---|--:|---:|---:|--:|
| **US → EU** | 1,152 | +0.779 | +0.320 | **0** (correctly zero) |
| **EU → US** | 1,152 | +1.359 | +0.899 | 82 (all artefacts) |

The US → EU direction correctly shows zero survivors: a US signal at
close(t-1) = 4 PM NY knows nothing about what happened from 4 PM to
11:30 AM NY next day (intraday European opening), so there's no parallel
leak.

---

## The **genuine** cross-continent finding (EWG/EWU → US equity)

Stripping the timezone artefact, the real EU → US signal shows up
in EWG/EWU rows. Max CI_lo per true cross-continent pair:

| Signal | Target | Max CI_lo | Pass Bonferroni? |
|---|---|---:|:-:|
| EWU | QQQ | +0.356 | ❌ |
| EWG | QQQ | +0.326 | ❌ |
| EWU | IWB | +0.326 | ❌ |
| EWG | IWB | +0.233 | ❌ |
| EWU | SPY | +0.220 | ❌ |
| EWG | SPY | +0.170 | ❌ |

**None passes the CI_lo ≥ 0.45 gate.** The genuine EU → US cross-
continent edge is in the +0.2 to +0.4 "real but weak" range we've
seen across every prior EU experiment. Not deployable.

---

## What this tells us methodologically

This is the **cleanest methodology finding** of the entire April 2026
research series. A single subtle assumption (that `^GDAXI` close and
`SPY` close can be aligned by date without adjusting for wall-clock
time) produced **dozens of apparent champions** with Sharpes 3× higher
than anything real.

### Research-math guardrail to add

When testing cross-instrument signals where the signal and target
trade on different exchanges with different close times:

1. **State the close time (wall-clock UTC) for both signal and target** before running the WFO. If they differ, flag it.
2. **Prefer signals that close at the same wall-clock time as the target** (e.g. US-listed country ETFs for US-target signals — EWG, EWU, EWJ, EWC, EWY).
3. **If local-indices must be used**, shift the signal by an extra day so it's known before the PRIOR day's target close. In our case: position at bar t should use GDAXI close(t-2), not close(t-1), to ensure the GDAXI close is clearly "yesterday's news" with no same-day US session overlap.
4. **Validate any cross-continent signal by running the same backtest with the USD-listed proxy as signal.** If the proxy gives similar Sharpe, the edge is real. If the proxy gives materially lower Sharpe, the "edge" is a timezone overlap.

This should go into the titan-orchestrator skill at
`references/research-math-guardrails.md` as a new failure-mode entry:
**"Cross-timezone signal alignment"**.

---

## v4 portfolio status

**Unchanged.** No deployment change. The 82 "survivors" are not real
edges. The genuine cross-continent finding (EWG/EWU → US equity at
~+0.3 CI_lo) is below the gate.

| v4 slot | Weight | Strategy |
|---|--:|---|
| 1 | 17 % | HYG → IWB (th=0.25) |
| 2 | 17 % | TIP → HYG (h=40, th=0.25) |
| 3 | 17 % | TLT → QQQ |
| 4 | 17 % | MR AUD/JPY |
| 5 | 17 % | IEF → GLD |
| 6 | 15 % | ML IWB |

---

## Broader lesson for the EU research track

Taken together, the April 2026 EU research produced:
- 5 Phase 1-3 experiments across 1,228 combos: 0 survivors
- Europe v1 + v2 across 5,184 combos: 0 survivors
- **Phase 5: 2,304 combos, 82 apparent survivors — ALL spurious**

The timezone-overlap finding is arguably more valuable than any
individual backtest result, because it's a **methodology correction
that protects future research**. Without hostile auditing, Phase 5
would have produced a GDAXI→QQQ strategy with Sharpe +1.36 that
would have failed in live trading (the real execution cost of
entering SPY at 11:30 AM NY sharply cuts the backtest Sharpe).

**Total EU research: 8,716 combos, 0 deployable edges, 1 methodology
guardrail added.** This is the correct outcome given the hypothesis:
European markets have been efficiently arbitraged against US signals
at daily frequency.
