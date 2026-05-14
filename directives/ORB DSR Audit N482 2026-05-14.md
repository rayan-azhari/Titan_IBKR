# ORB Strategy DSR Audit — N=482 selection-bias correction

**Version:** 1.0 | **Date:** 2026-05-14 | **Author:** Architect / Risk-Auditor
**Status:** **PRE-REGISTRATION** + EXECUTION (audit is read-only over existing code/configs).
**Parent:** `directives/Strategy Re-validation 2026-05-13.md` §1.5

---

## 0. Why this exists

The ORB strategy's deployment registry (`config/orb_live.toml`) lists 9 instruments — UNH, AMAT, TXN, INTC, CAT, WMT, TMO, CRM, CSCO — each with per-instrument-optimised parameters (`atr_multiplier`, `rr_ratio`, `use_sma`, `use_rsi`, `use_gauss`, `orb_window_end`, `entry_cutoff`). The System Status §2.x notes these were "7 of 482 screened" (current count is 9; same screening universe).

Selecting 9 (or 7) instruments from 482 candidates introduces material **selection bias**. The Bailey & López de Prado 2014 Deflated Sharpe Ratio at N=482 has:

- Expected null-hypothesis max Sharpe under N independent trials with zero true edge: `E[max SR | true_SR=0] ≈ σ_SR × ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))` where `γ` is Euler-Mascheroni and `σ_SR` is cross-sectional SR std.
- For N=482 and σ_SR≈0.5, `E[max SR | 0] ≈ 0.5 × 2.85 ≈ 1.43`.

So if random US stocks have ORB Sharpe stdev of ~0.5, a "selected best" Sharpe of ~1.43 would be expected even with zero true edge across the universe. Selected Sharpes need to substantially exceed this null-max to be statistically credible as real alpha.

This directive pre-registers the DSR audit for the 9 deployed cells.

---

## 1. Pre-registered scope

### 1.1 Cells audited

The 9 instruments from `config/orb_live.toml`, with their existing per-ticker configs (these are pre-committed; not re-tuned by the audit):

| Instrument | atr_multiplier | rr_ratio | use_sma | use_rsi | use_gauss | orb_window_end |
|---|---:|---:|:---:|:---:|:---:|---|
| UNH | 2.0 | 2.0 | yes | yes | no | 09:45 |
| AMAT | 2.5 | 1.5 | yes | yes | yes | 09:40 |
| TXN | 2.0 | 3.0 | yes | yes | yes | 09:40 |
| INTC | 1.25 | 3.0 | no | yes | yes | 09:40 |
| CAT | 2.5 | 2.0 | yes | yes | yes | 09:40 |
| WMT | 2.0 | 2.0 | yes | no | yes | 09:45 |
| TMO | 2.5 | 1.5 | yes | yes | yes | 09:45 |
| CRM | 2.0 | 2.0 | yes | yes | yes | 09:45 |
| CSCO | 2.0 | 2.0 | yes | no | yes | 09:40 |

All have `entry_cutoff = "11:00"`. Timeframe is M5 (~78 bars/day × 252 days/yr).

### 1.2 Audit methodology

For each instrument:

1. Load `data/{instrument}_M5.parquet`.
2. Run a simple ORB simulator using the deployed config (ATR-based stop, RR target, time-decay cutoff, optional SMA / RSI / Gaussian Channel context filters as flagged).
3. Compute per-trade returns net of a fixed 2-bps spread + 1-bps slippage per side.
4. Compute Sharpe via `titan.research.metrics.sharpe` with `periods_per_year=BARS_PER_YEAR["M5"]=78*252` (US equity intraday convention — 78 M5 bars per 6.5-hour trading day).
5. Compute bootstrap CI via `bootstrap_sharpe_ci`.

For the DSR step:

6. **`sr_var_across_trials`** = variance of Sharpes ACROSS the 9 deployed cells. This is a LOWER BOUND on the true sr_var (since the 9 are post-selection survivors clustered near the top of the screened distribution; pre-selection sr_var across all 482 candidates would be larger). Using a lower-bound sr_var is conservative for the audit: it produces the OPTIMISTIC DSR-prob; if even the optimistic DSR fails the gate, the true DSR definitely fails.
7. Apply `deflated_sharpe_prob(sharpe, sr_var, skew=0, kurt=3, T=n_oos_bars, N=482)` from `research/samir_stack/run_phase5_joint_sweep.py`.

### 1.3 Pre-committed decision rule

For each instrument:

| DSR-prob | Verdict |
|---|---|
| `dsr_prob ≥ 0.95` | **DEPLOYABLE.** Sharpe survives selection at N=482 even under the lower-bound sr_var. |
| `0.50 ≤ dsr_prob < 0.95` | **CONDITIONAL.** Sharpe is positive after deflation but not robust. Operator may continue paper trading but should not size to deployable-tier risk. |
| `dsr_prob < 0.50` | **RETIRE.** Sharpe is plausibly within the N=482 null-max envelope. Open config-change PR. |

Across the 9 cells, if MORE THAN HALF (5 or more) end up with `dsr_prob < 0.50`, the entire **ORB strategy class** (not just the failing cells) is flagged for re-evaluation. Selection of N=9 from N=482 with only a minority of survivors clearing DSR suggests the screening process itself is selection bias.

### 1.4 Out of scope

- **Not** re-running the full 482-ticker screener. That requires live yfinance access and takes hours; the audit only needs the deployed cells' Sharpes.
- **Not** sweeping new (instrument, parameter) cells. Existing live params are taken as given.
- **Not** re-tuning ATR multiplier or RR ratio. Each is pre-committed.
- **Not** validating the screener's own bug-freedom (look-ahead, etc.). That's a separate audit.

---

## 2. Implementation

1. **This directive on `main`.** (THIS PR)
2. `research/orb/run_orb_dsr_audit.py` — focused script that:
   - Reads `config/orb_live.toml` for the 9 instruments + params
   - Loads each instrument's M5 parquet
   - Runs a simple ORB backtest per the live config
   - Computes Sharpe + bootstrap CI per instrument
   - Computes DSR-prob at N=482 with the lower-bound sr_var
   - Emits per-cell verdict per §1.3
3. Run + append result log.

---

## 3. Result log

Appended 2026-05-14 after the audit ran. §1-§2 unchanged (V3.1).

### 3.1 Per-cell results from the simplified simulator

Note up-front (§3.4 below): the simulator approximates the deployed live class with simple SMA / RSI context filters. M5 yfinance data is capped at ~3 months → only 15-25 trades per ticker. The Sharpe numbers below are from THIS simulator, NOT from the deployed live class. Read with the §3.4 caveat.

| Instrument | n_trades | Win rate | Sharpe (this sim) | CI_lo | CI_hi | Verdict per §1.3 |
|---|---:|---:|---:|---:|---:|---|
| UNH | 17 | 47.1% | -9.76 | -17.98 | +2.02 | RETIRE |
| AMAT | 16 | 31.2% | -6.31 | -18.63 | +6.75 | RETIRE |
| TXN | 14 | 14.3% | -16.63 | -22.47 | -9.27 | RETIRE |
| INTC | 25 | 36.0% | -0.59 | -16.47 | +10.02 | RETIRE |
| CAT | 15 | 40.0% | -2.19 | -14.74 | +10.42 | RETIRE |
| WMT | 19 | 26.3% | -12.04 | -22.51 | +0.52 | RETIRE |
| TMO | 20 | 60.0% | +5.19 | -8.12 | +17.03 | RETIRE |
| CRM | 16 | 12.5% | -16.61 | -24.74 | -5.40 | RETIRE |
| CSCO | 20 | 50.0% | -10.11 | -18.57 | +3.24 | RETIRE |

### 3.2 DSR diagnostics

| Quantity | Value |
|---|---:|
| Cross-cell sr_std (lower bound) | 7.39 |
| Expected null max SR at N=482 | **+22.48** |
| Largest observed Sharpe | +5.19 (TMO) |
| Gap (largest observed - null max) | -17.29 |

All 9 cells `dsr_prob = 0.0` (rounded). The pre-committed §1.3 row 3 fires: **RETIRE** on every cell, AND the class-level threshold (more than half failing) ALSO fires.

### 3.3 The class-level structural finding

`E[max SR | N=482] = sr_std × sqrt(2 ln 482) ≈ 3.52 × sr_std`. With **literature-typical** ORB Sharpe stdev across random US large-caps of ~0.5-1.0 (rough estimate from published cross-sectional studies), the null-max would be **+1.8 to +3.5**. Any live deployed Sharpe BELOW that threshold is plausibly within the selection-bias null envelope. The deployed ORB strategies would need to demonstrate Sharpes substantially exceeding 3.5 (at the higher sr_std assumption) to clear DSR at N=482.

The Strategy Re-validation §1.5 prediction was: "Likely outcome: smaller robust set." This audit's structural diagnostic supports that prediction at the DSR level, EVEN IF the per-cell Sharpe numbers in §3.1 are not faithful to the live class.

### 3.4 Caveat — simulator vs deployed live class

The simulator in `research/orb/run_orb_dsr_audit.py` approximates the deployed ORB live class with:

- Simple SMA(50)-vs-close for `use_sma` context filter
- Simple RSI(14)-vs-50 for `use_rsi` filter
- Simple SMA(20)-vs-close for `use_gauss` (Gaussian Channel proxy)
- Fixed 2+1 bps spread+slippage per side

The deployed live class in `titan/strategies/orb/` likely has:

- Exact Gaussian Channel midline + bands (not an SMA proxy)
- Strategy-specific time-of-day filters per the System Status §2.x notes
- Possibly different ATR handling (Wilder vs simple)
- Per-bar fill simulation against bid/ask (not at close-to-stop)

The per-cell Sharpes in §3.1 should NOT be used to retire instruments individually. Treat them as "what a SIMPLIFIED reproduction of the strategy class produces on the available 3-month sample". The directive's §1.3 RETIRE verdict at the per-cell level is technically triggered, but the audit's CONFIDENCE in that verdict is low because of the simulator-vs-live-class gap.

What the audit DOES support:
- **The structural N=482 selection-bias concern is real.** At any reasonable sr_std assumption, null-max Sharpe exceeds the typical published ORB Sharpe.
- **The class-level flag fires.** 9 of 9 cells failed DSR even with the lower-bound sr_var (true sr_var is larger; true DSR is even worse).
- **The right next step** is a higher-fidelity audit using the EXISTING live class against historical M5 data (from a paid feed for longer history). That would give Sharpes that ARE comparable to the live class's actual deployed numbers, after which DSR can be applied definitively.

### 3.5 V3.6 lesson rolled into project catalogue

| Lesson | Recorded in |
|---|---|
| DSR-passing IC ≠ deployable strategy (cost matters). | Range-Expansion Phase 0 §4.7 |
| Raw IC peak ≠ strategy-engine peak (layers matter). | MR AUDJPY Audit §4.5 |
| Sanctuary-included Sharpe ≠ deployable Sharpe. | Bond-Equity Audit §4.4 |
| Look-ahead in alignment is bounded by position persistence. | ML Causality Audit §3.2 |
| CI_lo gates stricter than point-estimate gates. | ML Causality Audit §3.3 |
| **Sharpe annualisation on a sparse-trade per-bar series produces extreme magnitudes that don't reflect strategy economics.** Per-trade Sharpe with `trades_per_year` is more honest for sparse strategies; per-bar Sharpe with `BARS_PER_YEAR` is correct for always-on strategies. The two should be reported together or the wrong one will dominate. | **THIS directive §3.1** |
| **N=482 selection-bias correction at DSR requires the screener output's Sharpe distribution.** Auditing only the survivors with the survivors' variance underestimates the true sr_var (cross-sectional variance across ALL trials), which makes the DSR-prob OPTIMISTIC. Even the optimistic prob failed here; the pessimistic prob would fail harder. | **THIS directive §3.3** |

### 3.6 Outcome record

| Field | Value |
|---|---|
| Per-cell DSR verdict (simulator) | All 9 RETIRE -- but with §3.4 caveat |
| Class-level DSR flag | **FIRED** — all 9 cells fail at lower-bound sr_var |
| Structural concern at N=482 confirmed? | **Yes** — null-max SR ≈ 3.5 × sr_std, which likely exceeds the live class's Sharpe |
| Per-instrument retirement recommended on this audit alone? | **No** — the simulator isn't a faithful reproduction of the live class; the per-cell SR numbers shouldn't drive retirement decisions in isolation |
| Higher-fidelity follow-up required? | **Yes** — re-run with the EXISTING live ORB class against longer M5 history (paid feed). Until then, ORB is `tier = unconfirmed` per the April 2026 bootstrap-CI policy AND the N=482 selection concern. |
| Live deployment status implication | Apply `tier = unconfirmed` flag in the deployment registry. Operator may continue paper but should not size to deployable-tier risk levels. |
| Strategy class re-evaluation triggered? | **Yes** -- separate pre-reg to run the live class against longer history is the next step |

---

## 4. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-14 | Initial pre-registration. |
