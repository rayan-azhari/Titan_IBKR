# IC Confluence — Range-Expansion × Mean-Reversion gating

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect
**Status:** **PRE-REGISTRATION** — committed BEFORE the confluence scan runs (V3.1).
**Parent:** `directives/IC Signal Census Phase C 2026-05-13.md` §6.7 item 3.

---

## 0. Why this exists

Phase C surfaced two TIER_B survivors:

* `intraday_range_atr` on NQ/ES H1: **positive IC** at h=1 (range-expansion → next bar up).
* `rsi_dev`, `bb_pctb`, `vwap_overshoot` on ES H1: **negative IC** at h=1 (overbought → next bar down).

Both are real (clear DSR+BH+fold) on ES H1, on overlapping bars, on the SAME forward returns. Their signals are mechanically distinct:

| Signal | High value means | Direction at h=1 |
|---|---|---|
| `intraday_range_atr` | high range vs ATR (vol expansion) | + (range-expansion continues / mean-reverting overshoot) |
| `rsi_dev` | overbought (RSI > 50) | − (mean-reversion) |

The two can fire in **agreement** or **contradiction** on any given bar:

| Range_atr sign | RSI sign | Both signals say |
|:-:|:-:|---|
| high (positive signal) | low (oversold) | **BUY** (range-expansion ⇒ up + mean-reversion from oversold ⇒ up) |
| high | high (overbought) | conflict (range-expansion ⇒ up, mean-reversion ⇒ down) |
| low | low | conflict (range-expansion ⇒ down, mean-reversion ⇒ up) |
| low | high | **SELL** (low range ⇒ small move OR down, overbought ⇒ down) |

The agreement quadrants should produce **stronger IC** than the contradiction quadrants. This directive tests that hypothesis with a regime-gated factory: `range_atr_gated_by_rsi`, which returns `range_atr` only when RSI is in the agreeing regime, and zero otherwise. The IC of this gated signal at h=1 is the test statistic.

This is **not** an AND-gate strategy proposal -- that comes downstream in the strategy pre-registration. This is the IC-level question: does the regime-conditioning add information?

---

## 1. Pre-registered scope

### 1.1 Signal

`range_atr_when_oversold` is the gated signal:

```python
def range_atr_when_oversold(close, period_atr, period_rsi, high, low):
    """Range-expansion magnitude when RSI is below 50 (oversold)."""
    iat = (today_range / ATR(period_atr)) - 1
    rsi_dev = wilder_rsi(close, period_rsi) - 50
    return iat.where(rsi_dev < 0, 0.0)
```

The factory is added to `research/ic_analysis/ic_census_lib.py::signal_factories()` with `needs=["close", "high", "low"]`.

### 1.2 Targets, timeframes, horizons

| Target | TF | Horizons |
|---|---|---|
| NQ | H1 | 1, 8, 40 |
| ES | H1 | 1, 8, 40 |
| MES | H1 | 1, 8, 40 |
| MNQ | H1 | 1, 8, 40 |

Only H1 -- the parent's TIER_B is H1-only; the D-version was tested in the MTF Lift directive and failed every gate. No D scans here.

### 1.3 Parameter grid (V3.1 pre-committed)

Three cells, sweeping `period_atr`, holding `period_rsi=14` fixed:

| Cell | period_atr | period_rsi |
|---|---:|---:|
| 0 | 7 | 14 |
| 1 | 14 | 14 |
| 2 | 28 | 14 |

The `period_atr=14` cell is the **headline candidate** -- it matches the Phase C TIER_B headline. Plateau gate as usual (interior cell, neighbours clear `|t|>3`, IC range <30%).

### 1.4 Gates

Inherited from Phase C unchanged. `|t_NW| > 4.5`, BH at α=0.05 across the full pool of this scan, fold-stable ≥4 of 5, plateau-stable, sanctuary 12 months.

MTF agreement is **not required** here (H1 only by construction). Survivors of this scan would be tier_mtf_n_a if we used the existing schema strictly -- but for diagnostic purposes a confluence-survivor on H1 is informative on its own. We'll log MTF as N/A in the result table.

### 1.5 Test statistic + decision rule

**Compare the gated IC to the unconditional Phase C IC on the same target/horizon/cell.**

| Outcome | Action |
|---|---|
| Gated IC magnitude **> 1.5×** unconditional IC at the headline cell | Strong evidence the mechanism concentrates in oversold regimes. Worth incorporating as a regime gate in the eventual strategy. |
| Gated IC magnitude **0.5×–1.5×** unconditional | Inconclusive. The gating doesn't significantly help or hurt. Use the unconditional signal. |
| Gated IC magnitude **< 0.5×** unconditional | Gating destroys the signal. Either the mechanism isn't regime-concentrated, OR my prior about the agreement quadrant was wrong. Document and stop. |

The 1.5× and 0.5× thresholds are pre-committed here; they're not tuned post-hoc.

---

## 2. Out of scope

- Other gating variants (overbought, MA-distance-based, etc.) — each would need its own pre-registration. V3.1 forbids running multiple regime gates and picking the best one.
- AND-gate strategy construction. This directive only computes IC. Strategy backtest gets its own pre-reg.
- The bb_pctb or vwap_overshoot gating variants. RSI is chosen because (a) it's the most standard, (b) it's parameter-symmetric with the range_atr period, and (c) the Phase C top-of-leaderboard mean-reversion signal.

---

## 3. Implementation

1. **This directive on `main`.** (THIS PR)
2. Add `range_atr_when_oversold` factory to `ic_census_lib.py`.
3. `config/ic_census_universe_confluence.toml` — focused universe TOML.
4. Run the existing `run_ic_census.py` against the new universe.
5. Append result log to §4. Compare gated IC to Phase C unconditional IC per §1.5.

---

## 4. Result log

Appended 2026-05-14 after the scan ran. §1-§3 unchanged (V3.1).

### 4.1 Headline comparison — gated vs unconditional

For each (target, horizon) at the headline cell `period_atr=14, period_rsi=14`:

| Target | h | Unconditional IC | Unconditional t_NW | **Gated IC** | **Gated t_NW** | Ratio | Decision (§1.5) |
|---|---:|---:|---:|---:|---:|---:|---|
| NQ | 1 | +0.02481 | +7.18 | **+0.00900** | +2.58 | **0.36×** | < 0.5× — gating destroys |
| ES | 1 | +0.02313 | +6.76 | **+0.00625** | +1.79 | **0.27×** | < 0.5× — gating destroys |
| MNQ | 1 | +0.02180 | +4.07 | +0.00701 | +1.30 | 0.32× | < 0.5× |
| MES | 1 | +0.02007 | +3.80 | +0.00528 | +0.98 | 0.26× | < 0.5× |

Across all four equity-futures targets at h=1, **the gated IC is 26-36% of the unconditional IC**. The gated signal does not pass any audit-discipline gate (DSR, BH, plateau, fold-stable) on any target.

### 4.2 Mechanism

The range-expansion mechanism is **regime-agnostic with respect to RSI sign**. Gating to RSI<50 throws away roughly half the predictive bars and proportionally weakens the signal — consistent with the gating capturing ~half the population of bars on which the signal works, not concentrating to a stronger half.

Two implications:

1. **The strategy proposal should use the unconditional `intraday_range_atr`** at `period=14` H1. Regime-conditional gating by RSI offers no IC improvement.
2. **The original prior was wrong.** I had hypothesised that the "high range + oversold" quadrant would be the strongest. Instead, range-expansion works in both regimes — the mechanism is microstructural autocorrelation of bar size, not a mean-reversion-into-trend continuation.

V3.1: this directive's verdict is "Gating Destroys" (§1.5 row 3). V3.1 forbids me from now retroactively testing `range_atr_when_overbought` as a fishing expedition to find a regime where the signal IS concentrated. Any such test would require a separate pre-registration directive with its own rationale, before the data is examined again.

### 4.3 Outcome record

| Field | Value |
|---|---|
| Confluence-gated IC stronger than unconditional? | **No** — universally weaker (0.26-0.36× across 4 targets) |
| Regime-conditional strategy variant viable? | **No** — gating destroys the signal |
| Should the eventual strategy include RSI gating? | **No** — use unconditional `intraday_range_atr` |
| Negative result documented per V3.6? | Yes (this §) |
| Re-test permitted with a different gating signal (overbought, vwap, bb_pctb)? | Only with a fresh pre-registration directive that gives mechanistic reason BEFORE the data is examined. No retroactive variant-fishing. |

### 4.4 What this confirms about the range-expansion mechanism

The range-expansion signal IS predictive (Phase C confirmed) but its **predictive bars are spread across RSI regimes**, not concentrated in any one. That's consistent with the simplest mechanistic story: bar-size autocorrelation. A high-range bar tends to be followed by another high-range bar regardless of where momentum or mean-reversion indicators sit. The signal IS the mechanism; gating it is throwing away samples.

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial pre-registration. |
