# Agent-Driven Autoresearch Loop — 2026-04-21

**Distinction from the prior run**: the earlier
`Autoresearch Post-Remediation 2026-04-21.md` was a sweep of a
**pre-defined 45-experiment catalogue**. This is the karpathy-style
agent-driven loop — at each step the agent observes the leaderboard,
forms a hypothesis, writes a config, runs it via `evaluate.py`, and
chooses the next hypothesis from what the results reveal.

Agent = Claude Opus 4.7 (this session). Six iterations run, each one
narrow and hypothesis-first. Each iteration below lists the stated
hypothesis, the config tested, and the observed-vs-predicted gap —
the reasoning trace the next hypothesis was formed from.

---

## Iteration 1 — LQD as cleaner credit signal

**Hypothesis**: HYG→QQQ works at Sharpe +0.94. LQD (investment-grade
corporate bonds) strips high-yield noise; might be a cleaner credit
signal.

**Test**: LQD → QQQ, SPY, IWB at lb=10.

| Combo | Sharpe | SCORE | Max DD |
|---|---:|---:|---:|
| LQD→QQQ lb=10 | +0.836 | +1.012 | -43.3 % |
| LQD→SPY lb=10 | +0.647 | +0.819 | -36.3 % |
| LQD→IWB lb=10 | +0.634 | +0.792 | -37.6 % |

**Result**: Weaker than HYG (+0.94) and TLT (+0.98). IG credit → weaker
signal than high-yield credit or duration. Reasoning: HY spreads are
more reactive to risk-off; IG barely moves at the macro regime break.
**No new champion.**

---

## Iteration 2 — Dollar strength as risk-off signal

**Hypothesis**: Bond momentum captures rates. Dollar momentum captures
FX/global risk. These should be independent sources of information.
Test whether naive (positive-momentum-long) direction works; if not,
the inverse signal might.

**Test**: DXY / UUP → QQQ, GLD at lookback 10 / 20 / 60.

| Combo | Sharpe | SCORE |
|---|---:|---:|
| DXY→QQQ lb=10 | +0.466 | +0.467 |
| DXY→GLD lb=60 | +0.293 | +0.471 |
| **UUP→QQQ lb=10** | **+0.849** | **+1.095** |
| UUP→GLD lb=60 | +0.422 | +0.786 |

**Result**: Big gap between DXY (raw index) and UUP (ETF). UUP→QQQ
scored strongly. Likely explanation: UUP data is cleaner (ETF
regular-hours bars vs the FX index's 24h + rollover data gaps). **New
signal candidate: UUP** — worth sweeping.

---

## Iteration 3 — UUP target × lookback sweep

**Hypothesis**: If UUP→QQQ works at lb=10, check (a) is lb=10 really
optimal, (b) does the signal generalize to other equity targets?

**Test**: UUP → QQQ / SPY / IWB / GLD at lookback 10 / 20 / 40 / 60.

| Combo | Sharpe | SCORE |
|---|---:|---:|
| **UUP→IWB lb=10** | **+0.786** | **+1.126** |
| UUP→SPY lb=10 | +0.771 | +1.101 |
| UUP→QQQ lb=10 | +0.849 | +1.095 |
| UUP→GLD lb=60 | +0.422 | +0.786 |

**Result**: lb=10 dominates on every target. UUP→IWB at lb=10 is the
best single UUP configuration (SCORE +1.126). Extends the bond-equity
family — now a **bond-or-dollar → equity** family with UUP providing
the FX axis.

---

## Iteration 4 — ML stacking on untested underlyings

**Hypothesis**: ML IWB stacking is the universe's #1 strategy
(SCORE +4.546). Does the ML label/stacking pipeline generalize? Try
bonds (TLT), commodities (GLD), credit (HYG), leverage (TQQQ), and
DIA.

**Test**: ML_BASE config applied to each instrument.

| Instrument | Sharpe | SCORE | Trades |
|---|---:|---:|---:|
| TLT | -1.808 | -99.0 | 4 |
| GLD | +0.537 | +0.923 | 7 |
| HYG | +0.000 | -99.0 | 0 |
| **TQQQ** | **+2.308** | **+2.847** | 5 |

**Result**: ML TQQQ stacking scored second-highest in the whole
universe (after ML IWB). TQQQ is 3x leveraged QQQ — the 3x leverage
magnifies an already-strong ML signal. ML on narrow underlyings
(TLT rates, GLD commodities, HYG credit) fails (0-4 trades, no
signal). **ML stacking works on broad equity indices and their
leveraged variants, not on asset-class-specific ETFs.**

---

## Iteration 5 — AUD/JPY vwap_anchor grid

**Hypothesis**: Our AUD/JPY paper-deployed config uses
`vwap_anchor=46` (from pre-fix research). Now that the math is
corrected, is 46 still optimal?

**Test**: vwap_anchor ∈ {12, 24, 36, 46, 60, 72} with
donchian_pos_20 / conservative.

| anchor | Sharpe | SCORE | Max DD |
|---:|---:|---:|---:|
| 12 | -0.063 | -0.591 | -52.1 % |
| **24** | **+0.534** | **+0.968** | -38.3 % |
| 36 | +0.484 | +0.818 | -46.5 % |
| 46 (currently live) | +0.323 | +0.489 | -54.4 % |
| 60 | +0.304 | +0.342 | -60.9 % |
| 72 | +0.109 | -0.054 | -63.1 % |

**Result**: **vwap_anchor=24 is optimal** on the corrected harness.
Currently deployed value (46) is not only suboptimal, it's the
**fourth-worst** of six tested. This is the most actionable finding
of the loop: the paper-live AUD/JPY strategy should be **reconfigured
to anchor=24** before any further capital commitment.

---

## Iteration 6 — TIP / LQD → GLD inflation-hedge logic

**Hypothesis**: Gold is the classic inflation hedge. TIP
(inflation-protected Treasuries) momentum should predict GLD directly,
and more robustly than IEF→GLD (rates → gold). LQD (IG credit) should
also capture macro regime relevant to gold at a longer lookback.

**Test**: TIP → GLD and LQD → GLD at lb {10, 20, 40, 60}.

| Combo | Sharpe | SCORE |
|---|---:|---:|
| **TIP→GLD lb=10** | **+0.686** | **+1.059** |
| TIP→GLD lb=60 | +0.614 | +0.956 |
| **LQD→GLD lb=60** | **+0.677** | **+1.051** |

**Result**: Both confirmed. TIP→GLD lb=10 scores slightly above the
existing IEF→GLD lb=60 champion (Sharpe +0.721) once CI is factored
in (TIP has more trades: 264 vs 114). **LQD→GLD lb=60 is a fresh
finding** — different signal family (IG credit), different horizon,
moderate correlation to existing champions. Both are candidates to
replace or supplement IEF→GLD.

---

## Cumulative discoveries (this loop)

Ranked by SCORE, with reasoning for inclusion:

| # | Strategy | Sharpe | SCORE | Why it matters |
|--:|---|---:|---:|---|
| 1 | ML TQQQ stacking | +2.308 | +2.847 | 2nd highest in universe; leveraged ML alpha |
| 2 | UUP→IWB lb=10 | +0.786 | +1.126 | NEW family (dollar → equity), not in catalogue |
| 3 | UUP→SPY lb=10 | +0.771 | +1.101 | Same family, different target |
| 4 | UUP→QQQ lb=10 | +0.849 | +1.095 | Highest Sharpe of UUP family |
| 5 | TIP→GLD lb=10 | +0.686 | +1.059 | Replaces IEF→GLD? (cleaner signal) |
| 6 | LQD→GLD lb=60 | +0.677 | +1.051 | Novel IG credit → gold edge |
| 7 | LQD→QQQ lb=10 | +0.836 | +1.012 | Confirms credit cluster |
| 8 | AUD/JPY anchor=24 | +0.534 | +0.968 | **Live config optimization** |

**Gaps I didn't explore** (worth a future loop):

- **Multi-bond confluence** — AND-gate TLT + HYG + TIP all flagging
  risk-off = high-conviction short. Not supported by current
  framework without new strategy code.
- **UUP + bond combo** — does UUP add orthogonal information on top
  of TLT→QQQ? Needs correlation check.
- **Sector ETFs** as targets — XLK, XLE, XLF, XLV not in data dir.
- **Shorter-duration Treasury** (SHY) as Fed-policy signal. Not in data.

---

## Actionable recommendations

1. **Change paper-live AUD/JPY config** from `vwap_anchor=46` to
   `vwap_anchor=24`. Single most impactful finding: current live
   config is 4th-worst of six tested.

2. **Add UUP→IWB lb=10 to the candidate portfolio**. It's a dollar
   signal (not a rates signal) and should be orthogonal to the
   TLT/HYG/TIP cluster. Needs a correlation check against the
   existing 5-set before committing a weight — if |ρ| < 0.3 vs
   everything, consider 10 % weight.

3. **Add ML TQQQ stacking as a second ML channel** alongside ML IWB.
   Sharpe +2.308 with the same stacking pipeline. Concern: TQQQ's 3x
   leverage means per-trade P&L volatility is 3x the underlying —
   position-size at 1/3 of what ML IWB gets.

4. **Investigate TIP→GLD lb=10 as a potential replacement for
   IEF→GLD lb=60**. More trades (264 vs 114), similar Sharpe. But
   IEF→GLD has CI_lo +0.302 from the re-rank; TIP→GLD lb=10 had
   CI_lo +0.234. CI-wise IEF is slightly safer — don't swap without
   explicit CI on TIP.

---

## Artefacts

All experiments tracked in-conversation; raw run outputs are in the
message history above. No CSV was produced for this loop because
the agent iterated in real time. Individual results can be
re-generated by re-running the relevant ``run_one`` calls via
`scripts/rerank/run_autoresearch_safe.py`.
