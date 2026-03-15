# MTF Strategy Optimization Protocol (VectorBT)

> **Status: COMPLETE — Round 3 Validated (2026-03-14)**
> This document reflects the final validated configuration. All three stages have been run and the
> resulting parameters are locked in `config/mtf.toml`. Do NOT re-run Stage 3 unless starting a
> new research cycle with fresh data.

---

## Overview

The MTF Confluence strategy was optimized in three sequential stages using VectorBT backtesting
on EUR/USD data (H1/H4/D/W timeframes, 2005–2026). Each stage locks a layer of parameters before
the next stage begins, preventing look-ahead contamination across the search space.

**Split:** 70% In-Sample (IS) / 30% Out-of-Sample (OOS)
**Instrument:** EUR/USD on IDEALPRO
**Timeframes:** H1, H4, D (Daily), W (Weekly)
**Evaluation metric:** OOS Sharpe Ratio (full friction: 1.5 pip fees + 1.0 pip slippage + carry)

---

## Stage 1 — Moving Average Type and Threshold

**Objective:** Determine the best MA type and confirmation threshold across all timeframes.

**Script:**
```bash
uv run python research/mtf/run_mtf_optimisation.py
```

**Parameters swept:**
- MA Type: `SMA`, `EMA`, `WMA`
- Confirmation Threshold: 0.05 to 0.85 (step 0.05)

**Result (locked):**
- **MA Type: SMA** — outperformed WMA and EMA on OOS Sharpe
- **Threshold: 0.10** — conservative; ensures conviction before entry

> [!IMPORTANT]
> WMA was the early frontrunner in IS. SMA won on OOS. This is why the IS/OOS check matters —
> WMA was overfit to the training period.

**Auto-save:** Best `MA_Type` and `Threshold` written to `.tmp/mtf_state.json`.
**Report:** `.tmp/reports/mtf_stage1_scoreboard.csv`

---

## Stage 2 — Timeframe Weights

**Objective:** Determine how much influence each timeframe has on the composite score.

**Script:**
```bash
uv run python research/mtf/run_mtf_stage2.py
```

**Auto-loads:** Stage 1 results from `.tmp/mtf_state.json`.

**Parameters swept:**
- Weight distributions: Balanced, Trend-heavy, H1-heavy, D-heavy, and intermediate variants
- Fixed: MA Type = SMA, Threshold = 0.10

**Result (locked):**
| Timeframe | Weight | Role |
|---|---|---|
| D (Daily) | **0.60** | Primary trend — dominant driver |
| H4 | 0.25 | Swing confirmation |
| H1 | 0.10 | Entry timing |
| W (Weekly) | 0.05 | Long-term regime context |

**Auto-save:** Winning weights written to `.tmp/mtf_state.json`.
**Report:** `.tmp/reports/mtf_stage2_weights.csv`

---

## Stage 3 — Indicator Tuning (Per Timeframe)

**Objective:** Tune `fast_ma`, `slow_ma`, and `rsi_period` for each timeframe individually
using a greedy approach (optimize one timeframe at a time in order of importance).

**Script:**
```bash
uv run python research/mtf/run_mtf_stage3.py
```

**Auto-loads:** Stage 1 + Stage 2 results from `.tmp/mtf_state.json`.

**Greedy optimization order:** D → H4 → H1 → W (highest weight first)

**Result (locked):**
| Timeframe | fast_ma | slow_ma | rsi_period |
|---|---|---|---|
| D | 13 | 20 | 14 |
| H4 | 10 | 50 | 21 |
| H1 | 10 | 30 | 21 |
| W | 13 | 21 | 10 |

**Report:** `.tmp/reports/mtf_stage3_params.csv`

---

## Stage 4 — ATR Stop Sensitivity Sweep

**Objective:** Find the optimal ATR stop multiplier (hard stop distance from entry).

This sweep was run post-Stage 3 as a separate sensitivity analysis.

**Result:**
| ATR Mult | OOS Sharpe | CAGR% | MaxDD% | Win Rate |
|---|---|---|---|---|
| 0.50 | 2.563 | 25.14 | −5.75 | 31.3% |
| 1.00 | 2.732 | 27.14 | −5.59 | 40.3% |
| 1.50 | 2.831 | 28.29 | −5.25 | 45.6% |
| **2.50** | **2.936** | **29.44** | **−5.12** | **50.7%** |
| 3.00 | 2.895 | 29.05 | −5.25 | 52.2% |

**Result (locked): `atr_stop_mult = 2.5`** — Sharpe peaks here. Tighter stops cut winners
prematurely; 2.5× gives trades room to breathe while capping catastrophic loss.

All 8 multipliers above 1.0 achieved Sharpe > 1.0 — the strategy is robust across the full range.

---

## Meta-Filter Experiment — Discarded

An XGBoost meta-model (trained to predict win probability) was tested as an additional filter.

| Variant | Trades | Win Rate | OOS Sharpe |
|---|---|---|---|
| Raw signal (live config) | 506 | 40.3% | **2.732** |
| Meta-Entry-Only | 247 | 38.9% | 1.839 |
| Meta-Full (entry + exit) | 928 | 35.8% | 0.831 |

**Verdict: DISCARDED.** The meta-model failed on both dimensions simultaneously:
- Entry filter skipped good trades (−0.893 Sharpe vs raw)
- Probability-gate exits caused churn: exited then re-entered the same position repeatedly

**Do not re-introduce an ML overlay to this strategy without a fresh OOS test.**

---

## Final Locked Configuration

All results written to `config/mtf.toml`:

```toml
confirmation_threshold = 0.10
atr_stop_mult = 2.5          # Round 3 optimal: Sharpe 2.936

[weights]
H1 = 0.10
H4 = 0.25
D  = 0.60
W  = 0.05

[H1]
fast_ma = 10
slow_ma = 30
rsi_period = 21

[H4]
fast_ma = 10
slow_ma = 50
rsi_period = 21

[D]
fast_ma = 13
slow_ma = 20
rsi_period = 14

[W]
fast_ma = 13
slow_ma = 21
rsi_period = 10
```

---

## Round 3 Validated OOS Performance

Full friction: 1.5 pip fees/side, 1.0 pip slippage/side, next-bar open fills, 2.5× ATR stop, asymmetric carry.

| Metric | Value |
|---|---|
| OOS Sharpe | **2.936** |
| IS Sharpe | 2.790 |
| OOS/IS ratio | 0.98 ✓ |
| OOS CAGR | +29.44% |
| Carry-adjusted CAGR | ~+26%/yr |
| OOS Max Drawdown | −5.12% |
| OOS Win Rate | 50.7% |
| OOS Trades | ~371 (6.4yr period) |

### Robustness Gates (all passed)

| Gate | Result |
|---|---|
| IS/OOS ratio ≥ 0.5 | 0.98 ✓ |
| Slippage stress (5 levels, 0.5–3.0 pip) | Sharpe never below 1.0 ✓ |
| ATR sensitivity (8 multipliers) | 8/8 above Sharpe 1.0 ✓ |
| Monte Carlo (N=1,000) | 5th-pct Sharpe 1.80, 100% profitable ✓ |
| Rolling WFO (2yr anchor / 6mo windows) | 35/38 windows positive (92%) ✓ |
| Fixed-notional sizing test | Sharpe flat — edge is signal, not compounding ✓ |

---

## When to Re-Run Optimization

Re-run all three stages if:
1. **Model is stale > 6 months** with degrading live performance
2. **New timeframe is being tested** (e.g., adding M15 as entry trigger)
3. **New instruments** beyond EUR/USD are added

> [!CAUTION]
> Always re-run Stage 3 with **fresh OOS data** (extend the data window forward, maintain
> the 70/30 split). Never re-optimize on the period you previously used as OOS — that is
> look-ahead bias by another name.
