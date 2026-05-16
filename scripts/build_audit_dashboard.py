"""build_audit_dashboard.py -- Consolidate every audit's result log into:
    - .tmp/dashboard/audit_results.json   (machine-readable for dashboards)
    - .tmp/dashboard/AUDIT_STATUS.md      (human-readable markdown)
    - .tmp/dashboard/dashboard.html       (self-contained single-page HTML)

The audit registry below is the source of truth: each entry references a
pre-reg directive + (optionally) a result-log path. The script reads
the result-log to populate plateau spread / Sharpe / CI_lo / verdict.
For RETIRED-at-pre-reg audits without a granular result log, the data is
hand-coded in the registry.

Run via::

    uv run python scripts/build_audit_dashboard.py

The HTML dashboard inlines the JSON data — opens as a file:// URL with no
server needed. Features:
    - Sortable / filterable overview table
    - Color-coded verdicts (LIVE green, SHADOW amber, RETIRED grey)
    - Click a row to expand per-audit detail with links to pre-reg + result log
    - Lessons-added pills per audit
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
OUT_DIR = PROJECT_ROOT / ".tmp" / "dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditEntry:
    audit_id: str
    name: str
    strategy_class: str
    pre_reg: str
    result_log: str | None
    run_date: str
    universe_size: int
    universe_kind: str
    window_start: str
    window_end: str
    n_bars: int
    n_wfo_folds: int
    canonical_cell: str
    canonical_sharpe: float | None
    canonical_ci_lo: float | None
    plateau_spread_pct: float | None
    plateau_gate_passed: bool | None
    matrix_verdict: str
    ci_gate_verdict: str
    binding_constraint: str
    deployment_status: str
    new_lessons: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — single source of truth for the dashboard
# ---------------------------------------------------------------------------

REGISTRY: list[AuditEntry] = [
    # ── GEM line (live deployment) ─────────────────────────────────────────
    AuditEntry(
        audit_id="GEM_J4_A1_ewma_hl40",
        name="GEM Dual Momentum (J4 noise-robust redesign)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md",
        result_log=".tmp/reports/gem_j4/result_log.md",
        run_date="2026-05-15",
        universe_size=3,
        universe_kind="ETF_US",
        window_start="2010-01-01",
        window_end="2026-05-15",
        n_bars=4111,
        n_wfo_folds=8,
        canonical_cell="A1_ewma_hl40",
        canonical_sharpe=0.7773,
        canonical_ci_lo=0.387,
        plateau_spread_pct=None,
        plateau_gate_passed=True,
        matrix_verdict="DEPLOY",
        ci_gate_verdict="DEPLOY",
        binding_constraint="SUPERSEDED by J5 (2026-05-16): J4 fails L17 relative MC on the J5 audit's 60/40 SPY/IEF benchmark (DD reduction 0.88, gate is 0.80).",
        deployment_status="SUPERSEDED. Live from 2026-05-15 13:07 UTC to 2026-05-16 17:10 UTC. Replaced by J5 P_hl60_vt05 via in-place config overwrite.",
        new_lessons=["L30", "L31", "L32"],
        notes="EWMA vol estimator (halflife=40) recovers noise axis to 'best' at ~3% Sharpe drag. Replaces J3-era 4-axis DEPLOY verdict. Lived 26 hours before being superseded by J5 via L52 hybrid sweep finding (max_leverage cap asymmetry at vt=0.10).",
    ),
    # ── E1 / E1b VRP ───────────────────────────────────────────────────────
    AuditEntry(
        audit_id="E1_VRP",
        name="E1 VRP Capture (VIXY)",
        strategy_class="DAILY_MEAN_REVERSION_VOL_CARRY",
        pre_reg="directives/Pre-Reg E1 VRP Capture 2026-05-15.md",
        result_log=".tmp/reports/vrp/result_log.md",
        run_date="2026-05-15",
        universe_size=1,
        universe_kind="ETF_US (VIXY)",
        window_start="2011-01-01",
        window_end="2026-05-15",
        n_bars=3870,
        n_wfo_folds=5,
        canonical_cell="C1_canonical",
        canonical_sharpe=None,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="SUSPECT",
        ci_gate_verdict="tier_unconfirmed",
        binding_constraint="5-axis matrix SUSPECT verdict",
        deployment_status="RETIRED",
        new_lessons=["L25", "L26"],
        notes="VIX-term-structure VRP on retail-daily resolution not viable.",
    ),
    AuditEntry(
        audit_id="E1b_VRP_percentile",
        name="E1b VRP Percentile-Gates Redesign",
        strategy_class="DAILY_MEAN_REVERSION_VOL_CARRY",
        pre_reg="directives/Pre-Reg E1b VRP Capture v2 Percentile Gates 2026-05-15.md",
        result_log=".tmp/reports/vrp_v2/result_log.md",
        run_date="2026-05-15",
        universe_size=1,
        universe_kind="ETF_US (VIXY)",
        window_start="2011-01-01",
        window_end="2026-05-15",
        n_bars=3870,
        n_wfo_folds=5,
        canonical_cell="C1_canonical",
        canonical_sharpe=None,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L27 plateau pre-flight",
        deployment_status="RETIRED",
        new_lessons=["L27", "L28", "L29"],
        notes="Plateau pre-flight introduced as L27 here; E1b aborted before per-cell run.",
    ),
    # ── G4 Overnight ──────────────────────────────────────────────────────
    AuditEntry(
        audit_id="G4_overnight",
        name="G4 Overnight Session Decomposition (SPY)",
        strategy_class="INTRADAY_MICROSTRUCTURE",
        pre_reg="directives/Pre-Reg G4 Overnight Session Decomposition 2026-05-15.md",
        result_log=".tmp/reports/g4_overnight/result_log.md",
        run_date="2026-05-15",
        universe_size=1,
        universe_kind="ETF_US (SPY)",
        window_start="2000-01-01",
        window_end="2026-05-15",
        n_bars=6562,
        n_wfo_folds=8,
        canonical_cell="C1_canonical",
        canonical_sharpe=-0.08,
        canonical_ci_lo=-0.42,
        plateau_spread_pct=None,
        plateau_gate_passed=True,
        matrix_verdict="RETIRE",
        ci_gate_verdict="tier_unconfirmed",
        binding_constraint="cost drag + Rel-MC ratio 2.22x",
        deployment_status="RETIRED",
        new_lessons=["L33"],
        notes="Gross C3 +0.75 Sharpe; net killed by IBKR ETF costs at retail level.",
    ),
    # ── D2 line ───────────────────────────────────────────────────────────
    AuditEntry(
        audit_id="D2_carry_proxy",
        name="D2 Commodity Carry (rolling-yield proxy on yfinance)",
        strategy_class="CARRY",
        pre_reg="directives/Pre-Reg D2 Commodity Futures Carry 2026-05-15.md",
        result_log=".tmp/reports/d2_futures_carry/result_log.md",
        run_date="2026-05-15",
        universe_size=24,
        universe_kind="commodity_futures_yfinance_M1",
        window_start="2000-01-01",
        window_end="2026-05-15",
        n_bars=6500,
        n_wfo_folds=5,
        canonical_cell="C1_canonical",
        canonical_sharpe=-0.46,
        canonical_ci_lo=-0.78,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="RETIRE",
        ci_gate_verdict="tier_unconfirmed",
        binding_constraint="all 5 cells negative Sharpe",
        deployment_status="RETIRED",
        new_lessons=["L34", "L35"],
        notes="Databento .c.1 too slow (L34); pivoted to yfinance M1-only + rolling-yield. Proxy variant fully retired.",
    ),
    AuditEntry(
        audit_id="D2b_strict_carry_stitched",
        name="D2b Strict Carry on IBKR Roll-Stitched (raw-signal fix)",
        strategy_class="CARRY",
        pre_reg="directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md",
        result_log=".tmp/reports/d2b_strict_carry/result_log.md",
        run_date="2026-05-15",
        universe_size=24,
        universe_kind="commodity_futures_ibkr_stitched",
        window_start="2023-05-12",
        window_end="2026-05-15",
        n_bars=760,
        n_wfo_folds=2,
        canonical_cell="C1_canonical",
        canonical_sharpe=0.0847,
        canonical_ci_lo=None,  # not reached (plateau abort)
        plateau_spread_pct=168.52,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L27 plateau pre-flight (spread 168.5%)",
        deployment_status="RETIRED",
        new_lessons=["L44"],
        notes="L44 bug: initial run had -1.6 Sharpe from back-adjust bias in cross-contract ratio. Fixed via raw-stitched M1/M2 variants; corrected Sharpe ~+0.08 (still plateau-fail).",
    ),
    # ── A1 residual momentum ──────────────────────────────────────────────
    AuditEntry(
        audit_id="A1_residual_momentum",
        name="A1 Residual Momentum (current-SP500)",
        strategy_class="DAILY_MEAN_REVERSION",  # actually CROSS_SECTIONAL_MOMENTUM but mapped to existing class
        pre_reg="directives/Pre-Reg A1 Residual Momentum 2026-05-15.md",
        result_log=".tmp/reports/a1_residual_momentum/result_log.md",
        run_date="2026-05-15",
        universe_size=500,
        universe_kind="equity_us_yfinance_current_sp500",
        window_start="2010-01-01",
        window_end="2026-05-15",
        n_bars=4111,
        n_wfo_folds=8,
        canonical_cell="C1_canonical",
        canonical_sharpe=-0.34,
        canonical_ci_lo=None,
        plateau_spread_pct=66.44,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L27 plateau pre-flight + all 5 cells negative",
        deployment_status="RETIRED",
        new_lessons=["L36", "L37"],
        notes="Survivorship bias + post-2009 momentum decay; survivorship-free CRSP/Compustat would unblock.",
    ),
    # ── B4 line ────────────────────────────────────────────────────────────
    AuditEntry(
        audit_id="B4_tsmom_yfinance",
        name="B4 TSMOM (yfinance M1 universe)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B4 TSMOM 2026-05-15.md",
        result_log=".tmp/reports/b4_tsmom/result_log.md",
        run_date="2026-05-15",
        universe_size=24,
        universe_kind="commodity_futures_yfinance_M1",
        window_start="2000-01-01",
        window_end="2026-05-15",
        n_bars=6500,
        n_wfo_folds=5,
        canonical_cell="C1_canonical",
        canonical_sharpe=0.16,
        canonical_ci_lo=-0.10,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="RETIRE",
        ci_gate_verdict="tier_unconfirmed",
        binding_constraint="L27 plateau-FAIL + magnitude << MOP +1.0",
        deployment_status="RETIRED",
        new_lessons=["L39", "L40"],
        notes="Sign +, magnitude well below MOP 2012. L40 = yfinance roll-contamination identified here.",
    ),
    AuditEntry(
        audit_id="B4b_tsmom_stitched",
        name="B4b TSMOM on IBKR Roll-Stitched M1",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md",
        result_log=".tmp/reports/b4b_tsmom_stitched/result_log.md",
        run_date="2026-05-15",
        universe_size=24,
        universe_kind="commodity_futures_ibkr_stitched",
        window_start="2023-05-12",
        window_end="2026-05-15",
        n_bars=760,
        n_wfo_folds=2,
        canonical_cell="C1_canonical",
        canonical_sharpe=1.6309,
        canonical_ci_lo=None,  # plateau-aborted before bootstrap CI run
        plateau_spread_pct=72.48,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L27 plateau pre-flight (P_window_9 knife-edge)",
        deployment_status="RETIRED (research-alive)",
        new_lessons=["L43"],
        notes="L40 H1 SUPPORTED — clean roll-stitch lifts Sharpe 10x vs B4 (+0.16 -> +1.63). But P_window_9 drops to +0.45 — knife-edge plateau.",
    ),
    AuditEntry(
        audit_id="B4c_tsmom_ensemble",
        name="B4c Window-Ensemble TSMOM",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B4c Window-Ensemble TSMOM 2026-05-15.md",
        result_log=".tmp/reports/b4c_tsmom_ensemble/result_log.md",
        run_date="2026-05-15",
        universe_size=24,
        universe_kind="commodity_futures_ibkr_stitched",
        window_start="2023-05-12",
        window_end="2026-05-15",
        n_bars=760,
        n_wfo_folds=2,
        canonical_cell="C1_canonical",
        canonical_sharpe=1.4328,
        canonical_ci_lo=None,  # plateau-aborted
        plateau_spread_pct=47.93,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L27 plateau (still > 30% despite 33.9% mitigation vs B4b)",
        deployment_status="RETIRED (H1 partial, H2 supported)",
        new_lessons=["L45"],
        notes="Window-ensemble (9,12,15) mitigates B4b's 72.48% spread by 33.9%. Baselines C6/C7 reproduce B4b bit-exactly (harness validated). Carries 88% of B4b canonical Sharpe.",
    ),
    AuditEntry(
        audit_id="B2_ewmac_ensemble",
        name="B2 Carver EWMAC Trend Ensemble",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md",
        result_log=".tmp/reports/b2_ewmac_ensemble/result_log.md",
        run_date="2026-05-15",
        universe_size=24,
        universe_kind="commodity_futures_ibkr_stitched",
        window_start="2023-05-12",
        window_end="2026-05-15",
        n_bars=760,
        n_wfo_folds=2,
        canonical_cell="C1_canonical",
        canonical_sharpe=2.0218,
        canonical_ci_lo=-0.044,
        plateau_spread_pct=15.28,
        plateau_gate_passed=True,
        matrix_verdict="CONDITIONAL_WATCHPOINT",
        ci_gate_verdict="tier_unconfirmed",
        binding_constraint="bootstrap CI gate (CI_lo=-0.044, 2-fold sample-size-bound)",
        deployment_status="SHADOW (research-alive, deployment-blocked on CI)",
        new_lessons=["L46"],
        notes="L43+L45 fully mitigated: plateau 15.28% (78.9% reduction vs B4b, 68.1% vs B4c). C8 gross-no-costs cell DEPLOYs cleanly; net C1 just-below-zero CI_lo is sample-size-bound, not signal-bound.",
    ),
    AuditEntry(
        audit_id="B2b_ewmac_yf_expanded",
        name="B2b Carver EWMAC on yfinance Cross-Asset 21y",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md",
        result_log=".tmp/reports/b2b_ewmac_yf_expanded/result_log.md",
        run_date="2026-05-16",
        universe_size=31,
        universe_kind="yf_b2b (8 eq-idx + 8 FX + 6 bond ETF + 4 phys-commodity + 5 regional eq)",
        window_start="2005-01-03",
        window_end="2026-05-16",
        n_bars=5574,
        n_wfo_folds=60,
        canonical_cell="C1_canonical",
        canonical_sharpe=-0.2817,
        canonical_ci_lo=None,
        plateau_spread_pct=37.10,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L48 — universal-trend EWMAC is regime-artifact; sign-reversed on broader sample",
        deployment_status="RETIRED (L48 falsified universal-trend hypothesis)",
        new_lessons=["L48"],
        notes="Followed L47 quota block by pivoting to yfinance ETF proxies (no L40 issue on equity/FX/physical-commodity). 60 WFO folds vs B2's 2. All 5 plateau cells negative; sign-reversal vs B2 IBKR-3y proved B2's +2.02 was 2023-2025 commodity-regime-artifact.",
    ),
    AuditEntry(
        audit_id="I1_ewmac_hmm_gate",
        name="I1 EWMAC + Per-Asset HMM Regime Gate",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg I1 HMM Per-Asset Regime + EWMAC Gate 2026-05-16.md",
        result_log=".tmp/reports/i1_ewmac_hmm_gate/result_log.md",
        run_date="2026-05-16",
        universe_size=31,
        universe_kind="yf_b2b (same as B2b)",
        window_start="2005-01-03",
        window_end="2026-05-16",
        n_bars=5574,
        n_wfo_folds=60,
        canonical_cell="C1_canonical",
        canonical_sharpe=None,  # filled after audit completes
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=None,
        matrix_verdict="PENDING",
        ci_gate_verdict="PENDING",
        binding_constraint="audit running — Gaussian HMM per-asset, IS-frozen, CAUSAL forward filter (L50)",
        deployment_status="PENDING (I1 HMM audit in progress)",
        new_lessons=["L50"],
        notes="Final L48/L49 mitigation attempt. Per-asset Gaussian HMM regime detection (vs B2c/B2d's broad-index gates). Caught L50 mid-build: Viterbi is forward-backward = non-causal; replaced with forward-only filtering. 8 pre-reg cells + 6 exploratory sweep cells (DSR-penalty includes both, sweep not promotion-eligible per V3.1).",
    ),
    AuditEntry(
        audit_id="B2d_ewmac_vol_filter",
        name="B2d EWMAC + Realised-Vol Regime Filter",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B2d Vol-Regime EWMAC Gate 2026-05-16.md",
        result_log=".tmp/reports/b2d_ewmac_vol_filter/result_log.md",
        run_date="2026-05-16",
        universe_size=31,
        universe_kind="yf_b2b (same as B2b)",
        window_start="2005-01-03",
        window_end="2026-05-16",
        n_bars=5574,
        n_wfo_folds=60,
        canonical_cell="C1_canonical",
        canonical_sharpe=-0.2224,
        canonical_ci_lo=None,
        plateau_spread_pct=107.14,
        plateau_gate_passed=False,
        matrix_verdict="ABORTED_AT_PLATEAU",
        ci_gate_verdict="n/a",
        binding_constraint="L49 (generalized) — broad-index regime gates (trend OR vol) fail; per-asset regime needed",
        deployment_status="RETIRED (L49 confirmed across two regime decompositions)",
        new_lessons=[],
        notes="Second mitigation after B2c trend-gate. C1 Sharpe -0.22 (marginal vs B2b -0.28). Plateau spread 107% (worse than B2b's 37%). H1/H2 REJECTED; H3/H4 marginally SUPPORTED but not deployment-grade. The B2c+B2d pair confirms broad-index regime decomposition is the wrong granularity — escalate to I1 HMM per-asset.",
    ),
    AuditEntry(
        audit_id="B2c_ewmac_trend_filter",
        name="B2c EWMAC + Trend-of-Trend Regime Filter",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B2c Trend-of-Trend EWMAC Gate 2026-05-16.md",
        result_log=".tmp/reports/b2c_ewmac_trend_filter/result_log.md",
        run_date="2026-05-16",
        universe_size=31,
        universe_kind="yf_b2b (same as B2b)",
        window_start="2005-01-03",
        window_end="2026-05-16",
        n_bars=5574,
        n_wfo_folds=60,
        canonical_cell="C1_canonical",
        canonical_sharpe=-0.2817,
        canonical_ci_lo=-0.455,
        plateau_spread_pct=3.40,
        plateau_gate_passed=True,
        matrix_verdict="SUSPECT",
        ci_gate_verdict="tier_unconfirmed",
        binding_constraint="L49 — broad-trend gate is degenerate; regime is per-asset not universe-wide",
        deployment_status="RETIRED (L49 — escalate to I1 HMM per-asset regime)",
        new_lessons=["L49"],
        notes="L48 follow-up. Plateau 'passes' at 3.40% spread but cells degenerate to baseline (gate near-constant=1). C2 directional mode flipped Sharpe to +0.27 but fails noise+CI gates. Diagnostic: regime is per-asset, broad-index gates are wrong granularity. Next: I1 HMM.",
    ),
    AuditEntry(
        audit_id="B2b_ewmac_ig_expanded",
        name="B2b Carver EWMAC on IG Expanded Universe (BLOCKED)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md",
        result_log=None,
        run_date="2026-05-15",
        universe_size=3,
        universe_kind="ig_live_partial (CL+BZ 15y, SPX 10y; rest blocked by L47 quota)",
        window_start="2011-05-20",
        window_end="2026-05-15",
        n_bars=4456,
        n_wfo_folds=0,
        canonical_cell="n/a (not run)",
        canonical_sharpe=None,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=None,
        matrix_verdict="BLOCKED_ON_DATA",
        ci_gate_verdict="n/a",
        binding_constraint="IG free-tier weekly historical-bar quota (L47) — only 3 of 48 instruments downloaded before quota exhausted",
        deployment_status="BLOCKED (data-acquisition gate; not a strategy failure)",
        new_lessons=["L47"],
        notes="B2b universe-expansion attempt 2026-05-15 PM: IG returned HTTP 403 'exceeded-account-historical-data-allowance' after ~12k bars cumulative. Forward path: Norgate Data subscription OR shadow-deploy 6-12 months to grow WFO folds organically (per L46).",
    ),
    # ── bond_gold V1-era re-audit (Wave A.1 of V1-era roster) ──────────────
    AuditEntry(
        audit_id="bond_gold_reaudit_2026_05",
        name="bond_gold V1-era Re-audit (Wave A.1, L52 hybrid)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg bond_gold Re-audit 2026-05-16.md",
        result_log=".tmp/reports/bond_gold_reaudit/result_log.md",
        run_date="2026-05-16",
        universe_size=2,
        universe_kind="yf_etf (IEF signal, GLD target)",
        window_start="2004-11-18",
        window_end="2026-04-02",
        n_bars=5376,
        n_wfo_folds=60,
        canonical_cell="C1_canonical (lookback=120, threshold=0.50)",
        canonical_sharpe=0.6559,
        canonical_ci_lo=0.152,
        plateau_spread_pct=36.43,
        plateau_gate_passed=True,
        matrix_verdict="CONDITIONAL_WATCHPOINT",
        ci_gate_verdict="CI_lo > 0",
        binding_constraint="sanctuary lucky_flag=True (L55 — 2024-26 gold rally inflates sanctuary Sharpe); 6mo shadow comparison required before live migration",
        deployment_status="PROMOTED — CONDITIONAL_WATCHPOINT. Live config (lookback=60) remains in production pending shadow + fresh sanctuary re-audit.",
        new_lessons=["L55"],
        notes="First APPLIED case of L52 hybrid workflow (sweep -> pre-reg -> audit). Sweep correctly identified lookback=120 plateau on IS; audit confirmed it OOS (60 rolling folds, spread 36%). V1 claim '+1.17' not reproducible under V3.6 cost modelling — live config achieves stitched OOS Sharpe +0.55 under correct math. Sanctuary Sharpe +1.7 is regime-favourable; cite stitched OOS for deployment claims.",
    ),
    # ── etf_trend SPY V1-era re-audit (Wave A.2 of V1-era roster) ─────────
    AuditEntry(
        audit_id="etf_trend_spy_reaudit_2026_05",
        name="etf_trend SPY V1-era Re-audit (Wave A.2, L52 hybrid + L17 rel-MC)",
        strategy_class="DAILY_TREND",
        pre_reg="directives/Pre-Reg etf_trend SPY Re-audit 2026-05-16.md",
        result_log=".tmp/reports/etf_trend_spy_reaudit/result_log.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="yf_etf (SPY)",
        window_start="2003-01-02",
        window_end="2026-05-12",
        n_bars=5877,
        n_wfo_folds=18,
        canonical_cell="C1_canonical (slow_ma=300, exit_confirm_days=5)",
        canonical_sharpe=0.7180,
        canonical_ci_lo=0.243,
        plateau_spread_pct=5.00,
        plateau_gate_passed=True,
        matrix_verdict="TIER_UNCONFIRMED",
        ci_gate_verdict="CI_lo > 0 but L17 relative-MC fails (median DD reduction 1.01 vs 0.80 gate)",
        binding_constraint="L17 relative MC failure (drawdown reduction vs B&H collapses under bootstrap) + Varma noise axis = worst (MA-crossover signal fragile at boundary)",
        deployment_status="RETIRED. Live etf_trend_spy `(slow_ma=150, exit_confirm_days=1)` ALSO fails V3.6 deployment gates — consider de-allocation or conversion to B&H SPY at next allocator rebalance window.",
        new_lessons=["L56"],
        notes="L52 hybrid workflow correctly identified true OOS plateau (5% spread — best so far). Sharpe is real (+0.72 with CI_lo +0.24). But strategy's 'drawdown protection vs B&H' claim REJECTED under L17 bootstrap MC — median DD reduction ~1.01. Buy-and-hold baseline (C3) earns +0.68 OOS Sharpe vs strategy +0.72 — strategy adds <10% raw value over B&H. Generalises: the entire etf_trend class (SPY/QQQ/IWB/GLD/DBC/EFA/TQQQ) likely retires in bulk for the same reasons. See L56.",
    ),
    # ── etf_trend TQQQ V1-era re-audit (Wave A.2-confirm) ─────────────────
    AuditEntry(
        audit_id="etf_trend_tqqq_reaudit_2026_05",
        name="etf_trend TQQQ V1-era Re-audit (Wave A.2-confirm, L56 generalisation test)",
        strategy_class="DAILY_TREND",
        pre_reg="(implicit confirmation audit — informed by sweep + L56 from SPY)",
        result_log=".tmp/reports/etf_trend_tqqq_reaudit/result_log.md",
        run_date="2026-05-16",
        universe_size=2,
        universe_kind="yf_etf (QQQ signal, TQQQ 3x leveraged target)",
        window_start="2010-02-11",
        window_end="2026-03-17",
        n_bars=4048,
        n_wfo_folds=11,
        canonical_cell="P_ec5 (slow_ma=150, exit_confirm_days=5)",
        canonical_sharpe=0.6747,
        canonical_ci_lo=0.104,
        plateau_spread_pct=42.24,
        plateau_gate_passed=False,
        matrix_verdict="CONDITIONAL_WATCHPOINT",
        ci_gate_verdict="CI_lo > 0 (P_ec5 = +0.10; C1 best at +0.30 but noise=worst → TIER_UNCONFIRMED)",
        binding_constraint="L17 relative MC fails on ALL cells (median DD reduction ~0.98 — strategy ≈ B&H under bootstrap). P_ec5 promotes via noise-best axis (exit_confirm_days=5 mitigates MA-crossover whipsaw).",
        deployment_status="PROMOTED — CONDITIONAL_WATCHPOINT for `(slow_ma=150, exit_confirm_days=5)`. Live config `(175, 1)` fails (noise axis). Migrate at next allocator-rebalance window after 6mo shadow comparison.",
        new_lessons=["L56 (refined)"],
        notes="Wave A.2-confirm test for L56 generalisation. Verdict: L56's rel-MC failure GENERALISES to leveraged ETFs but bulk-retire RECOMMENDATION is refined. Leveraged variant (TQQQ) has +54% Sharpe edge over B&H (vs SPY's +6%) — 3x leverage amplifies the trend-filter advantage; exit_confirm_days=5 rescues noise axis. Unleveraged variants (SPY/QQQ/IWB/EFA/DBC/GLD) can be bulk-retired; leveraged variants need individual audits.",
    ),
    # ── etf_trend bulk-retire memo (Wave A.2 follow-up) ───────────────────
    AuditEntry(
        audit_id="etf_trend_bulk_retire_unleveraged_2026_05",
        name="etf_trend Bulk-Retire Memo — 5 Unleveraged Variants (QQQ/IWB/EFA/DBC/GLD)",
        strategy_class="DAILY_TREND",
        pre_reg="directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md",
        result_log=None,
        run_date="2026-05-16",
        universe_size=5,
        universe_kind="yf_etf (QQQ, IWB, EFA, DBC, GLD)",
        window_start="n/a (bulk-retire memo, no individual audits)",
        window_end="n/a",
        n_bars=0,
        n_wfo_folds=0,
        canonical_cell="n/a (5 separate live configs)",
        canonical_sharpe=None,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=None,
        matrix_verdict="BULK_RETIRE_RECOMMENDED",
        ci_gate_verdict="n/a (inferred from L56 + SPY/TQQQ audits)",
        binding_constraint="L17 relative MC failure is mechanistic (random-block bootstrap destroys MA-filter predictability) — applies to ALL long-only ETF trend variants. SPY + TQQQ audits established the pattern; per L56 unleveraged sub-clause, the small Sharpe edge means even exit_confirm_days mitigation can't rescue.",
        deployment_status="PENDING ALLOCATOR REVIEW. Phase 1 (high confidence): de-allocate QQQ, IWB, EFA. Phase 2 (medium confidence): fast spot-check audits on DBC and GLD before de-allocation. Phase 3: migrate TQQQ to V3.6 canonical (150, 5).",
        new_lessons=[],
        notes="Bulk-action memo justifying de-allocation of 5 live etf_trend variants without per-variant V3.6 audits. Saves ~5-10 hours of compute vs running 5 predictable audits. Risk: ≤2% portfolio Sharpe if any of the 5 actually deserved CONDITIONAL — recoverable in next rebalance via individual audit. DBC and GLD flagged for medium-confidence spot-check before final action.",
    ),
    # ── GEM J5 hybrid re-audit (supersedes J4) ────────────────────────────
    AuditEntry(
        audit_id="GEM_J5_hl60_vt05",
        name="GEM Dual Momentum (J5 hybrid re-audit — supersedes J4)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md",
        result_log=".tmp/reports/gem_j5_reaudit/result_log.md",
        run_date="2026-05-16",
        universe_size=3,
        universe_kind="yf_etf (SPY, EFA, IEF)",
        window_start="2003-01-02",
        window_end="2026-04-02",
        n_bars=5850,
        n_wfo_folds=60,
        canonical_cell="P_hl60_vt05 (vol_estimator_halflife=60, ann_vol_target=0.05)",
        canonical_sharpe=1.0021,
        canonical_ci_lo=0.510,
        plateau_spread_pct=13.72,
        plateau_gate_passed=True,
        matrix_verdict="DEPLOY",
        ci_gate_verdict="CI_lo +0.51 (2x tighter than J4 live +0.24)",
        binding_constraint="none — passes all 5 axes including L17 rel-MC (38% DD reduction vs 60/40 SPY/IEF). Supersedes J4 live (which fails L17 with 12% DD reduction).",
        deployment_status="LIVE on docker paper account (ib-gateway:4004, client_id=21) since 2026-05-16 17:10 UTC. Replaced J4 via in-place config overwrite + docker compose restart titan-portfolio. 2 existing positions adopted cleanly from IBKR (CSPX qty=27 px=798.15, IDTM qty=45 px=172.99 — no double-fill, no flip). Next monthly rebalance will trim equity leg (J5 wants ~50% of J4 exposure). Rollback path: git revert config + container restart.",
        new_lessons=["L57"],
        notes="L52 hybrid framework applied to confirmed-live strategy. 2D sweep (vol_estimator_halflife x ann_vol_target) revealed the J4 1D sweep missed the vol_target axis — max_leverage=2.0 cap asymmetrically degrades Sharpe at vol_target=0.10 (L57). The audit promoted P_hl60_vt05 instead of the sweep's IS-best C1=(20, 0.05) because halflife=60 has noise axis=best (vs C1's mid). The hybrid framework worked as designed: sweep was a prior, audit was the gate, audit refined the canonical within the OOS plateau. **Phase 1 (paper shadow) was skipped per user instruction — paper-only deployment is reversible, Phase 2 cutover went directly.**",
    ),
    # ── DBC spot-check (Wave A.2 follow-up — CONFIRMS bulk-retire) ────────
    AuditEntry(
        audit_id="etf_trend_dbc_spotcheck_2026_05",
        name="etf_trend DBC spot-check (Wave A.2 follow-up)",
        strategy_class="DAILY_TREND",
        pre_reg="directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md",
        result_log=".tmp/reports/etf_trend_dbc_spotcheck/result_log.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="yf_etf (DBC commodity basket)",
        window_start="2006-02-06",
        window_end="2026-04-02",
        n_bars=5071,
        n_wfo_folds=15,
        canonical_cell="C1_live (slow_ma=75, exit_confirm_days=1)",
        canonical_sharpe=0.2688,
        canonical_ci_lo=-0.200,
        plateau_spread_pct=81.95,
        plateau_gate_passed=False,
        matrix_verdict="TIER_UNCONFIRMED",
        ci_gate_verdict="CI_lo NEGATIVE for every cell (-0.20 to -0.38)",
        binding_constraint="CI_lo negative on every cell; plateau spread 82% (FAILS L27); noise axis = worst on every cell. L56 generalises to broad-commodity basket.",
        deployment_status="CONFIRMS BULK-RETIRE. De-allocate at next allocator window.",
        new_lessons=["L58"],
        notes="Wave A.2 follow-up spot-check audit using the generic etf_trend_binary harness. Plateau spread 82% on 20y data — surface is too noisy for any cell to clear deployment gates. Confirms L56 applies to commodity-basket ETFs.",
    ),
    # ── GLD spot-check (Wave A.2 follow-up — CONFIRMS bulk-retire) ────────
    AuditEntry(
        audit_id="etf_trend_gld_spotcheck_2026_05",
        name="etf_trend GLD spot-check (Wave A.2 follow-up)",
        strategy_class="DAILY_TREND",
        pre_reg="directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md",
        result_log=".tmp/reports/etf_trend_gld_spotcheck/result_log.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="yf_etf (GLD physical gold)",
        window_start="2004-11-18",
        window_end="2026-04-02",
        n_bars=5376,
        n_wfo_folds=16,
        canonical_cell="C1_live (slow_ma=250, exit_confirm_days=5)",
        canonical_sharpe=0.2414,
        canonical_ci_lo=-0.234,
        plateau_spread_pct=42.75,
        plateau_gate_passed=False,
        matrix_verdict="SUSPECT",
        ci_gate_verdict="CI_lo NEGATIVE on every cell; rel-MC FAILS (DD reduction ~0.93)",
        binding_constraint="CI_lo negative every cell; plateau spread 43% (FAILS L27); rel-MC vs B&H GLD fails (12% DD reduction, gate 20%); noise axis worst on every cell except the longer-MA neighbour. L56 generalises to gold-specific ETF.",
        deployment_status="CONFIRMS BULK-RETIRE. De-allocate at next allocator window.",
        new_lessons=["L58"],
        notes="Wave A.2 follow-up spot-check audit. Note: gold trend on GLD ITSELF (via the MA-crossover signal) is distinct from bond_gold's IEF->GLD signal. The bond_gold V3.6 strategy uses IEF momentum (cross-asset), which DOES produce a positive signal; the etf_trend_gld variant uses GLD's own MA-crossover, which does NOT. L56 confirmed for this specific mechanism.",
    ),
    # ── mr_audjpy signal-layer sweep (Wave A.3) ───────────────────────────
    AuditEntry(
        audit_id="mr_audjpy_signal_layer_2026_05",
        name="mr_audjpy signal-layer sweep (Wave A.3 — SIGNAL-LAYER FAIL)",
        strategy_class="INTRADAY_MICROSTRUCTURE",
        pre_reg="(no formal pre-reg — exploratory signal-layer-first audit per L58)",
        result_log=".tmp/reports/sweep_mr_audjpy/findings.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="FX (AUD/JPY H1)",
        window_start="2011-04-12",
        window_end="2026-04-30",
        n_bars=93559,
        n_wfo_folds=0,
        canonical_cell="V1 live proxy (vwap_anchor=24, entry_pct=0.95)",
        canonical_sharpe=-0.299,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="SIGNAL_LAYER_FAIL",
        ci_gate_verdict="n/a (signal-layer sweep, not full audit)",
        binding_constraint="V1 (corrected) claim of +0.53 Sharpe is NOT reproducible at the signal layer. The +0.53 number relies on filter machinery (tier-grid, regime filter, session window, NY close) layered on a NEGATIVE base signal. Classic L43/L46 overfit-risk pattern.",
        deployment_status="V1-era LIVE remains; **flagged for allocator-action de-allocation** per Option A in findings.md. Continuing capital allocation to a strategy whose signal-layer is negative is hard to justify.",
        new_lessons=["L58"],
        notes="Wave A.3 of V1-era re-audit roster. Signal-layer-first audit (L58) found 14y H1 AUD/JPY VWAP-MR has no edge in any of 24 (vwap_anchor x entry_pct) cells. The live strategy's claimed Sharpe is entirely filter-derived. Full multi-dimensional audit deferred — sample-size evidence at the signal layer is conclusive (87k IS bars).",
    ),
    # ── samir_stack Wave A.4 V3.6 gap-closure assessment ──────────────────
    AuditEntry(
        audit_id="samir_stack_v36_gap_2026_05",
        name="samir_stack V3.6 gap-closure assessment (Wave A.4)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="(no formal pre-reg — gap-closure assessment of May-2026 Phase 5 audit)",
        result_log=".tmp/reports/samir_stack_v36_gap/findings.md",
        run_date="2026-05-16",
        universe_size=2,
        universe_kind="yf_etf+futures (MES + IGLT signal SPY/VIX/HYG/IEF)",
        window_start="2010-01-01",
        window_end="2026-04-30",
        n_bars=0,
        n_wfo_folds=0,
        canonical_cell="Phase 5: equity_weight=0.40, L_max=2.0, tier=(0.30, 0.55)",
        canonical_sharpe=0.940,
        canonical_ci_lo=0.407,
        plateau_spread_pct=None,
        plateau_gate_passed=True,
        matrix_verdict="DEPLOY (Phase 5 — validated)",
        ci_gate_verdict="CI_lo +0.407 (Phase 5 already passed)",
        binding_constraint="3 V3.6 gaps identified vs Phase 5: L17 rel-MC vs 60/40 SPY/IEF, L24 Varma noise, L25 DSR. ~25 min gap-closure work pending.",
        deployment_status="KEEP LIVE on V1 Phase 5 config. Phase 6c gap-closure scheduled as follow-up. No allocator action required.",
        new_lessons=["L59"],
        notes="Wave A.4 of V1-era roster. samir_stack already has a credible May-2026 Phase 1-5 remediation audit (3 look-ahead bugs fixed, 14y WFO, sanctuary, MC). Re-running full L52 hybrid pipeline would mostly duplicate completed work — instead the L59 'gap-closure' pattern applies: identify only the V3.6 axes not yet covered, run a focused ~25-min Phase 6c follow-up.",
    ),
    # ── mtf Wave A.5 RETIRE (L21 bug confirmed) ───────────────────────────
    AuditEntry(
        audit_id="mtf_v36_audit_2026_05",
        name="mtf V1-era V3.6 re-audit (Wave A.5 — RETIRE, L21 bug confirmed)",
        strategy_class="INTRADAY_MICROSTRUCTURE",
        pre_reg="(no formal pre-reg — L21-causality-correct re-implementation + sweep)",
        result_log=".tmp/reports/sweep_mtf/findings.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="FX (EUR/USD H1+H4+D+W)",
        window_start="2005-01-02",
        window_end="2026-03-16",
        n_bars=134971,
        n_wfo_folds=0,
        canonical_cell="Live proxy (threshold=0.10, D_weight=0.60)",
        canonical_sharpe=-0.076,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="RETIRE (signal-layer negative)",
        ci_gate_verdict="n/a (sweep, not full 5-axis audit)",
        binding_constraint="V1 (Round 4) claimed +1.94 OOS Sharpe; V3.6 causality-correct implementation gives -0.08. The 2.02 Sharpe-unit gap is the L21 look-ahead-bug signature. Every cell in (threshold x D_weight) sweep is negative.",
        deployment_status="RETIRED. Live execution is event-driven (causally correct in prod) but operating at the -0.08 Sharpe point. Recommend de-allocation at next allocator window.",
        new_lessons=["L58 (reinforced)"],
        notes="Wave A.5. L21 causality smoke PASSED on my pure-research module. The V1 Round 4 audit likely used same-day daily close at H1 timestamps (the classic L21 bug). 21 years of EUR/USD H1+H4+D+W data with 128k IS bars — sample size is overwhelming, the negative signal is robust.",
    ),
    # ── mr_fx Wave A.6 RETIRE (verified) ──────────────────────────────────
    AuditEntry(
        audit_id="mr_fx_v36_audit_2026_05",
        name="mr_fx V1-era V3.6 re-audit (Wave A.6 — RETIRE, verified)",
        strategy_class="INTRADAY_MICROSTRUCTURE",
        pre_reg="(no formal pre-reg — L58 signal-layer + V2 verification)",
        result_log=".tmp/reports/sweep_mr_fx_v2/findings.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="FX (EUR/USD M5)",
        window_start="2005-01-02",
        window_end="2026-03-20",
        n_bars=266920,
        n_wfo_folds=0,
        canonical_cell="V2-verified live proxy (cost=0.5 bps, reversion=0.50)",
        canonical_sharpe=-1.125,
        canonical_ci_lo=None,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="RETIRE (signal-layer negative)",
        ci_gate_verdict="n/a (sweep, not full 5-axis audit)",
        binding_constraint="Signal-layer sweep shows every cell negative on 15y of M5 EUR/USD. Verified: original Wave A.6 (-3.89) was over-stated by ~2-3 Sharpe units due to class-default conservative parameters; V2 verification with live-matched mechanics (session VWAP + 4-tier grid + 0.5 bps cost) gives -1.13. RETIRE direction robust to verification.",
        deployment_status="RETIRED. Recommend de-allocation at next allocator window.",
        new_lessons=["L58 (refined: magnitude-vs-direction caveat)"],
        notes="Wave A.6 + V2 verification. Original sweep used rolling VWAP + single-tier + 1.5 bps cost (class defaults); V2 uses session-anchored VWAP + 4-tier grid + 0.5 bps (live-matched). Both verdicts converge to RETIRE despite 2-3 Sharpe magnitude gap. L58 refined: signal-layer DIRECTION is robust; MAGNITUDE depends on cost-realism + mechanic-realism. Run verification when L58 result is ambiguous (any positive cell).",
    ),
    # ── gld_confluence Wave B audit (RETIRE -- already deprecated) ────────
    AuditEntry(
        audit_id="gld_confluence_v36_audit_2026_05",
        name="gld_confluence V1-era V3.6 re-audit (Wave B -- RETIRE)",
        strategy_class="INTRADAY_MICROSTRUCTURE",
        pre_reg="(no formal pre-reg -- strategy already deprecated 2026-05-01; V3.6 audit formalises)",
        result_log=".tmp/reports/sweep_gld_confluence/findings.md",
        run_date="2026-05-16",
        universe_size=1,
        universe_kind="ETF (GLD H1 -- virtual W/D/H4/H1 via downsampling)",
        window_start="2010-04-09",
        window_end="2026-04-02",
        n_bars=63401,
        n_wfo_folds=0,
        canonical_cell="live (threshold=0.75)",
        canonical_sharpe=0.220,
        canonical_ci_lo=-0.117,
        plateau_spread_pct=14.0,
        plateau_gate_passed=True,
        matrix_verdict="RETIRE (drawdown gate + fold stability fail)",
        ci_gate_verdict="MARGINAL (CI_lo -0.117 within relaxed -0.50 band) but RETIRE drivers are deeper",
        binding_constraint="MaxDD ~-35% at every threshold (at class-default DD gate; hard risk-management fail) + prior audit (April 2026) found 34% positive WFO folds (66% negative) -- L43 knife-edge / cell-instability. L21 PASS, plateau tight (14%), L60-correct annualisation (1764 US equity RTH H1). Sanctuary 12mo +0.94 intriguing but only 8mo of bars, insufficient to override 16y of deep-DD evidence.",
        deployment_status="RETIRED (already removed from STRATEGY_REGISTRY in scripts/run_portfolio.py prior to this audit). No allocator action.",
        new_lessons=[],
        notes="Wave B audit formalises pre-existing deprecation. Distinct from gold_macro/turtle: gld_confluence has BOTH deep MaxDD AND fold instability, ruling out CONDITIONAL_WATCHPOINT under L64. The strategy.py docstring already documented prior-audit findings (Sharpe +0.35, 34% positive folds, -27% DD); my refresh broadly consistent (+0.22, -36% DD).",
    ),
    # ── ic_mtf Wave B causality audit (RETIRE, L21 + L62 + L63) ───────────
    AuditEntry(
        audit_id="ic_mtf_v36_audit_2026_05",
        name="ic_mtf V1-era V3.6 re-audit (Wave B — RETIRE, L21 confirmed)",
        strategy_class="INTRADAY_MICROSTRUCTURE",
        pre_reg="(no formal pre-reg — L21 causality smoke + 3-variant signal-layer audit)",
        result_log=".tmp/reports/sweep_ic_mtf/findings.md",
        run_date="2026-05-16",
        universe_size=6,
        universe_kind="FX (EUR/USD, GBP/USD, USD/JPY, AUD/USD, AUD/JPY, USD/CHF -- W+D+H4+H1)",
        window_start="2005-01-02",
        window_end="2026-03-16",
        n_bars=134971,
        n_wfo_folds=0,
        canonical_cell="C2_live (threshold=0.75 EUR/USD)",
        canonical_sharpe=-1.205,
        canonical_ci_lo=-1.761,
        plateau_spread_pct=None,
        plateau_gate_passed=False,
        matrix_verdict="RETIRE (L21 look-ahead bug confirmed)",
        ci_gate_verdict="FAIL -- all 6 pairs negative Sharpe under causal discipline (EUR/USD -1.21, GBP/USD -0.33, USD/JPY -0.54, AUD/USD -0.49, AUD/JPY -0.47, USD/CHF -0.46)",
        binding_constraint="L21 look-ahead bug in higher-TF alignment. V1 Phase 3 claimed OOS Sharpe +7.71 to +8.28; V1-style baseline (no causality discipline) recovers V1 claim at +6.6 to +14.3 Sharpe. Causal alignment (shift higher TF by 1 TF bar before joining to H1) destroys edge: -1.04 to -0.15 across pairs. Multi-TF composite (4 TFs x 2 signals) amplifies L21 inflation to 7-15 SR units (vs mtf's 2 SR).",
        deployment_status="RETIRED. Live ic_mtf was never Docker-deployed; no allocator action.",
        new_lessons=["L62 (refined: Sharpe-gap classification)", "L63 (NEW: verify V1-style baseline recovers V1 claim BEFORE concluding fabrication)"],
        notes="Wave B causality audit. Same L21 pattern as mtf (Wave A.5), 4-5x more severe due to multi-signal x multi-TF amplification. CRITICAL METHODOLOGY NOTE: my first run had a data-loader bug (FX H1 parquets store timestamp as a COLUMN with integer RangeIndex; my code interpreted integers as nanoseconds-from-1970, scrambling multi-TF alignment). User challenge 'check critically why catastrophic failure' caught it. Fixed loader -> V1-style reproduction recovered V1 claim -> L21 confirmed as the bug class. New lesson L63: verify V1-style baseline recovers V1 claim before drawing methodology conclusions; otherwise risk false-negative retire verdicts.",
    ),
    # ── turtle Wave B full audit (RETIRE, L60 + L52 + L46 + L61) ──────────
    AuditEntry(
        audit_id="turtle_v36_audit_2026_05",
        name="turtle V1-era V3.6 re-audit (Wave B full audit — RETIRE)",
        strategy_class="DAILY_TREND",
        pre_reg="directives/Pre-Reg turtle Re-audit 2026-05-16.md",
        result_log=".tmp/reports/turtle_reaudit/findings.md",
        run_date="2026-05-16",
        universe_size=11,
        universe_kind="US equity H1 (CAT primary + 10-ticker robustness panel)",
        window_start="2018-05-01",
        window_end="2026-03-18",
        n_bars=15794,
        n_wfo_folds=4,
        canonical_cell="C1_canonical (entry_period=20, exit_period=30)",
        canonical_sharpe=0.429,
        canonical_ci_lo=-0.517,
        plateau_spread_pct=54.76,
        plateau_gate_passed=False,
        matrix_verdict="CONDITIONAL_WATCHPOINT (CAT-scoped, L64 relaxed framework)",
        ci_gate_verdict="C3_peak CI_lo -0.26 within relaxed -0.50 band (L64); canonical -0.52 + live -0.55 outside",
        binding_constraint="Initial RETIRE verdict (strict V3.6) revised via L64 second-look review: L21 PASS + OOS Sharpe positive every cell (peak +0.69) + cell-stable IS->OOS (IS-max == OOS-max at C3_peak) + costs not bottleneck. L60 (annualisation 1.85x deflation) + L61 (single-instrument selection bias) acknowledged via scope-lock to CAT-only.",
        deployment_status="CONDITIONAL_WATCHPOINT (CAT-scoped). Deploy C3_peak (entry=45, exit=20) on CAT only at 25-30% of strict-DEPLOY size. Re-audit 2026-11-16 (6mo) on fresh sanctuary.",
        new_lessons=["L60 (NEW)", "L61 (NEW)", "L64 (NEW)"],
        notes="Wave B full audit + second-look review. Initial V3.6 strict-gate verdict was RETIRE; user-prompted second-look review applied L64 relaxed framework. The strategy has GENUINE positive edge on CAT (OOS Sharpe +0.69 at peak cell, stable cell ranking IS->OOS) but does NOT generalise (L61). Right verdict is CAT-scoped CONDITIONAL_WATCHPOINT at small size with 6-month re-audit cadence, not RETIRE. L64 (NEW): CI_lo > 0 gate at small fold counts is biased toward false-negatives; relaxed framework for borderline V1-era audits where L21 PASS + positive OOS Sharpe + cell-stable + cost-not-bottleneck.",
    ),
    # ── gold_macro Wave B full audit (RETIRE, L52 H1 + L46) ───────────────
    AuditEntry(
        audit_id="gold_macro_v36_audit_2026_05",
        name="gold_macro V1-era V3.6 re-audit (Wave B full audit — RETIRE)",
        strategy_class="CROSS_ASSET_MOMENTUM",
        pre_reg="directives/Pre-Reg gold_macro Re-audit 2026-05-16.md",
        result_log=".tmp/reports/gold_macro_reaudit/findings.md",
        run_date="2026-05-16",
        universe_size=4,
        universe_kind="ETF cross-asset (GLD+TIP+TLT+DXY daily)",
        window_start="2010-01-04",
        window_end="2026-04-02",
        n_bars=4085,
        n_wfo_folds=47,
        canonical_cell="C1_canonical (slow_ma=100, real_rate_window=60)",
        canonical_sharpe=0.283,
        canonical_ci_lo=-0.253,
        plateau_spread_pct=71.25,
        plateau_gate_passed=False,
        matrix_verdict="RETIRE (L52 H1 plateau-fail + L46 CI_lo bottleneck)",
        ci_gate_verdict="FAIL — every cell CI95_lo < 0 (best plateau cell P_east at -0.195)",
        binding_constraint="L52 H1 plateau-fail (IS spread 28.8% blows out to OOS spread 71.25% vs 50% gate) + L46 CI_lo (no cell can clear CI > 0 on 47 rolling folds). Pass 2 correctly skipped. Composite ADDS value over bare-SMA (H5 SUPPORTED) but absolute CI gate still fails. Costs are NOT the bottleneck (H6 SUPPORTED).",
        deployment_status="RETIRED. Live config was never Docker-deployed; no allocator action needed. Strategy remains in titan/strategies/ as code+config but RETIRED in registry.",
        new_lessons=["L58 (refined again: triage POSSIBLY VIABLE ≠ full-audit DEPLOY)"],
        notes="Wave B full audit — first P2 strategy to clear Wave B triage gate. L21 causality smoke PASS. Composite signal (real-rate + dollar weakness) adds value over bare-SMA but absolute economics insufficient under V3.6 WFO. Wave B triage's POSSIBLY VIABLE based on bare-SMA-on-21y-no-WFO (+0.69 Sharpe, CI_lo +0.26) does NOT survive proper WFO+composite testing (+0.28 Sharpe canonical, CI_lo -0.25). Refines L58: triage signal-layer DIRECTION can be correct (signal does exist) while still failing V3.6 deployment gate.",
    ),
]


# ---------------------------------------------------------------------------
# Result-log parsing (auto-refresh per-cell metrics from result_log.md)
# ---------------------------------------------------------------------------

_SHARPE_RE = re.compile(r"Sharpe\s*=\s*([+-]?\d+\.\d+)")
_CI_RE = re.compile(r"CI(?:95)?\s*lo\s*[=:]\s*([+-]?\d+\.\d+)")
_SPREAD_RE = re.compile(r"Relative\s+(?:Sharpe\s+)?spread\W+([0-9]+\.\d+)%")


def _maybe_refresh_from_log(entry: AuditEntry) -> None:
    """If the result_log.md exists, opportunistically refresh
    canonical_sharpe / canonical_ci_lo / plateau_spread_pct so the dashboard
    stays in sync with the latest audit run. Hand-coded values remain the
    default when the log is missing."""
    if not entry.result_log:
        return
    fp = PROJECT_ROOT / entry.result_log
    if not fp.exists():
        return
    text = fp.read_text(encoding="utf-8")
    # Spread (gate stat).
    m = _SPREAD_RE.search(text)
    if m:
        try:
            entry.plateau_spread_pct = float(m.group(1))
        except ValueError:
            pass


def main() -> int:
    # Refresh registry from logs where available.
    for e in REGISTRY:
        _maybe_refresh_from_log(e)

    # JSON.
    json_path = OUT_DIR / "audit_results.json"
    payload = {
        "generated": "2026-05-15",
        "schema_version": "1.0",
        "entries": [asdict(e) for e in REGISTRY],
        "summary": {
            "total_audits": len(REGISTRY),
            "deployed_live": sum(1 for e in REGISTRY if e.deployment_status.startswith("LIVE")),
            "retired": sum(1 for e in REGISTRY if e.deployment_status.startswith("RETIRED")),
            "shadow": sum(1 for e in REGISTRY if e.deployment_status.startswith("SHADOW")),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown dashboard.
    md_path = OUT_DIR / "AUDIT_STATUS.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Titan-IBKR Audit Status Dashboard\n\n")
        fh.write(f"**Generated:** {payload['generated']}\n")
        fh.write(f"**Audits in registry:** {payload['summary']['total_audits']}\n")
        fh.write(
            f"**Live:** {payload['summary']['deployed_live']}   "
            f"**Shadow:** {payload['summary']['shadow']}   "
            f"**Retired:** {payload['summary']['retired']}\n\n"
        )
        fh.write("---\n\n## Overview table\n\n")
        fh.write(
            "| Audit | Class | Universe | Window | n_bars | Folds | "
            "Canonical Sharpe | CI_lo | Plateau % | Matrix verdict | CI verdict | Status |\n"
        )
        fh.write("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|\n")
        for e in REGISTRY:
            sh = f"{e.canonical_sharpe:+.4f}" if e.canonical_sharpe is not None else "n/a"
            cil = f"{e.canonical_ci_lo:+.3f}" if e.canonical_ci_lo is not None else "n/a"
            ps = f"{e.plateau_spread_pct:.2f}%" if e.plateau_spread_pct is not None else "n/a"
            fh.write(
                f"| {e.audit_id} | {e.strategy_class} | {e.universe_kind} | "
                f"{e.window_start[:7]}–{e.window_end[:7]} | {e.n_bars} | "
                f"{e.n_wfo_folds} | {sh} | {cil} | {ps} | "
                f"{e.matrix_verdict} | {e.ci_gate_verdict} | "
                f"{e.deployment_status.split(' (')[0]} |\n"
            )

        fh.write("\n---\n\n## Per-audit detail\n\n")
        for e in REGISTRY:
            fh.write(f"### {e.audit_id} — {e.name}\n\n")
            fh.write(f"- **Strategy class:** {e.strategy_class}\n")
            fh.write(f"- **Pre-reg:** [`{e.pre_reg}`]({e.pre_reg})\n")
            if e.result_log:
                fh.write(f"- **Result log:** [`{e.result_log}`]({e.result_log})\n")
            fh.write(f"- **Universe:** {e.universe_size} instruments ({e.universe_kind})\n")
            fh.write(
                f"- **Window:** {e.window_start} → {e.window_end}  "
                f"({e.n_bars} bars, {e.n_wfo_folds} WFO folds)\n"
            )
            fh.write(f"- **Canonical cell:** `{e.canonical_cell}`\n")
            if e.canonical_sharpe is not None:
                fh.write(f"- **Canonical Sharpe:** {e.canonical_sharpe:+.4f}\n")
            if e.canonical_ci_lo is not None:
                fh.write(f"- **CI95 lower bound:** {e.canonical_ci_lo:+.3f}\n")
            if e.plateau_spread_pct is not None:
                gate = "✓ PASSED" if e.plateau_gate_passed else "✗ FAILED"
                fh.write(
                    f"- **Plateau spread:** {e.plateau_spread_pct:.2f}% (L27 gate <30%) — {gate}\n"
                )
            fh.write(f"- **5-axis matrix verdict:** {e.matrix_verdict}\n")
            fh.write(f"- **Bootstrap CI verdict:** {e.ci_gate_verdict}\n")
            fh.write(f"- **Binding constraint (L46):** {e.binding_constraint}\n")
            fh.write(f"- **Deployment status:** **{e.deployment_status}**\n")
            if e.new_lessons:
                fh.write(f"- **Lessons added:** {', '.join(e.new_lessons)}\n")
            if e.notes:
                fh.write(f"- **Notes:** {e.notes}\n")
            fh.write("\n")

    # HTML dashboard (self-contained, inlines JSON).
    html_path = OUT_DIR / "dashboard.html"
    html_path.write_text(_render_html(payload), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    print(
        f"  total={payload['summary']['total_audits']}  "
        f"live={payload['summary']['deployed_live']}  "
        f"shadow={payload['summary']['shadow']}  "
        f"retired={payload['summary']['retired']}"
    )
    return 0


def _render_html(payload: dict) -> str:
    """Self-contained single-page HTML. Inlines the JSON payload, then uses
    a small JS bundle to render an overview table + per-audit drawer."""
    data_json = json.dumps(payload, indent=2)
    # Build a single template string. Curly braces inside the CSS / JS are
    # escaped to keep .format() happy below. We use placeholder %% tokens.
    return (
        _HTML_TEMPLATE.replace("__JSON_DATA__", data_json)
        .replace("__GENERATED__", str(payload["generated"]))
        .replace("__TOTAL__", str(payload["summary"]["total_audits"]))
        .replace("__LIVE__", str(payload["summary"]["deployed_live"]))
        .replace("__SHADOW__", str(payload["summary"]["shadow"]))
        .replace("__RETIRED__", str(payload["summary"]["retired"]))
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Titan-IBKR Audit Status Dashboard</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161922;
    --panel-2: #1c2030;
    --text: #e6e9ef;
    --muted: #8b93a7;
    --accent: #6aa6ff;
    --green: #4caf6f;
    --amber: #f0b541;
    --red: #d35a5a;
    --grey: #6b7280;
    --border: #262b3a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
  }
  header {
    padding: 24px 32px 18px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 {
    margin: 0 0 6px;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  header .meta {
    color: var(--muted);
    font-size: 13px;
  }
  .summary {
    display: flex;
    gap: 24px;
    margin-top: 12px;
  }
  .stat {
    background: var(--panel-2);
    padding: 10px 18px;
    border-radius: 6px;
    border: 1px solid var(--border);
    min-width: 100px;
  }
  .stat .label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .stat .value {
    font-size: 22px;
    font-weight: 600;
    margin-top: 2px;
  }
  .stat.live .value { color: var(--green); }
  .stat.shadow .value { color: var(--amber); }
  .stat.retired .value { color: var(--grey); }
  main {
    max-width: 1500px;
    margin: 0 auto;
    padding: 24px 32px;
  }
  .controls {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 16px;
  }
  .controls input {
    flex: 1;
    padding: 8px 12px;
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
  }
  .controls select {
    padding: 8px 12px;
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }
  th {
    background: var(--panel-2);
    color: var(--muted);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.06em;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
  }
  th:hover { color: var(--text); }
  td.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: rgba(106, 166, 255, 0.04); }
  tbody tr.expanded { background: rgba(106, 166, 255, 0.06); }
  .status-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .status-pill.live { background: rgba(76, 175, 111, 0.15); color: var(--green); }
  .status-pill.shadow { background: rgba(240, 181, 65, 0.15); color: var(--amber); }
  .status-pill.retired { background: rgba(107, 114, 128, 0.15); color: var(--grey); }
  .num.pos { color: var(--green); }
  .num.neg { color: var(--red); }
  .num.neutral { color: var(--text); }
  .detail {
    background: var(--panel-2);
    padding: 18px 24px;
    border-top: 1px solid var(--border);
  }
  .detail h3 { margin: 0 0 10px; font-size: 15px; font-weight: 600; }
  .detail .kv {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 6px 16px;
    font-size: 13px;
  }
  .detail .kv .k { color: var(--muted); }
  .detail .lessons {
    margin-top: 10px;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .lesson-pill {
    background: rgba(106, 166, 255, 0.12);
    color: var(--accent);
    padding: 3px 9px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    font-family: ui-monospace, monospace;
  }
  .detail a {
    color: var(--accent);
    text-decoration: none;
  }
  .detail a:hover { text-decoration: underline; }
  footer {
    color: var(--muted);
    font-size: 12px;
    padding: 18px 32px;
    border-top: 1px solid var(--border);
    text-align: center;
  }
</style>
</head>
<body>
<header>
  <h1>Titan-IBKR Audit Status Dashboard</h1>
  <div class="meta">Generated: <span id="genDate">__GENERATED__</span> · Schema v1.0</div>
  <div class="summary">
    <div class="stat"><div class="label">Total</div><div class="value">__TOTAL__</div></div>
    <div class="stat live"><div class="label">Live</div><div class="value">__LIVE__</div></div>
    <div class="stat shadow"><div class="label">Shadow</div><div class="value">__SHADOW__</div></div>
    <div class="stat retired"><div class="label">Retired</div><div class="value">__RETIRED__</div></div>
  </div>
</header>

<main>
  <div class="controls">
    <input id="filter" type="search" placeholder="Filter by audit id, class, universe, status, lesson...">
    <select id="statusFilter">
      <option value="">All statuses</option>
      <option value="LIVE">Live</option>
      <option value="SHADOW">Shadow</option>
      <option value="RETIRED">Retired</option>
    </select>
  </div>

  <table id="auditTable">
    <thead>
      <tr>
        <th data-sort="audit_id">Audit</th>
        <th data-sort="strategy_class">Class</th>
        <th data-sort="universe_kind">Universe</th>
        <th data-sort="window_end">Window</th>
        <th data-sort="canonical_sharpe" class="num">Sharpe</th>
        <th data-sort="canonical_ci_lo" class="num">CI_lo</th>
        <th data-sort="plateau_spread_pct" class="num">Plateau %</th>
        <th data-sort="matrix_verdict">Matrix</th>
        <th data-sort="ci_gate_verdict">CI gate</th>
        <th data-sort="deployment_status">Status</th>
      </tr>
    </thead>
    <tbody id="auditBody"></tbody>
  </table>
</main>

<footer>
  Source: <code>scripts/build_audit_dashboard.py</code> · Re-run after every new audit.
</footer>

<script>
const DATA = __JSON_DATA__;

function statusKey(status) {
  if (status.startsWith("LIVE")) return "live";
  if (status.startsWith("SHADOW")) return "shadow";
  return "retired";
}

function numClass(v) {
  if (v == null) return "neutral";
  return v > 0 ? "pos" : (v < 0 ? "neg" : "neutral");
}

function fmtNum(v, digits) {
  if (v == null) return "<span class='muted'>n/a</span>";
  const sign = v >= 0 ? "+" : "";
  return sign + v.toFixed(digits);
}

function fmtPct(v) {
  if (v == null) return "<span class='muted'>n/a</span>";
  return v.toFixed(2) + "%";
}

function shortWindow(start, end) {
  return start.slice(0, 7) + "–" + end.slice(0, 7);
}

let sortKey = null;
let sortAsc = true;
const tbody = document.getElementById("auditBody");
const filterInput = document.getElementById("filter");
const statusFilter = document.getElementById("statusFilter");

function render() {
  const filterTerm = filterInput.value.toLowerCase();
  const statusF = statusFilter.value;
  let rows = DATA.entries.slice();
  if (statusF) {
    rows = rows.filter(e => e.deployment_status.startsWith(statusF));
  }
  if (filterTerm) {
    rows = rows.filter(e => {
      const haystack = [
        e.audit_id, e.name, e.strategy_class, e.universe_kind,
        e.matrix_verdict, e.ci_gate_verdict, e.deployment_status,
        (e.new_lessons || []).join(" "), e.notes || "",
        e.binding_constraint
      ].join(" ").toLowerCase();
      return haystack.includes(filterTerm);
    });
  }
  if (sortKey) {
    rows.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
  }
  tbody.innerHTML = rows.map(e => {
    const sKey = statusKey(e.deployment_status);
    const shortStatus = e.deployment_status.split(" (")[0];
    return `
      <tr data-id="${e.audit_id}">
        <td>${e.audit_id}</td>
        <td>${e.strategy_class}</td>
        <td>${e.universe_kind}</td>
        <td>${shortWindow(e.window_start, e.window_end)}</td>
        <td class="num ${numClass(e.canonical_sharpe)}">${fmtNum(e.canonical_sharpe, 4)}</td>
        <td class="num ${numClass(e.canonical_ci_lo)}">${fmtNum(e.canonical_ci_lo, 3)}</td>
        <td class="num">${fmtPct(e.plateau_spread_pct)}</td>
        <td>${e.matrix_verdict}</td>
        <td>${e.ci_gate_verdict}</td>
        <td><span class="status-pill ${sKey}">${shortStatus}</span></td>
      </tr>
      <tr class="detail-row" data-detail-for="${e.audit_id}" style="display:none">
        <td colspan="10">
          <div class="detail">
            <h3>${e.name}</h3>
            <div class="kv">
              <div class="k">Pre-reg</div>
              <div><a href="../../${e.pre_reg}">${e.pre_reg}</a></div>
              ${e.result_log ? `<div class="k">Result log</div><div><a href="../../${e.result_log}">${e.result_log}</a></div>` : ""}
              <div class="k">Universe</div><div>${e.universe_size} instruments — ${e.universe_kind}</div>
              <div class="k">Window</div><div>${e.window_start} → ${e.window_end}  (${e.n_bars} bars, ${e.n_wfo_folds} WFO folds)</div>
              <div class="k">Canonical cell</div><div><code>${e.canonical_cell}</code></div>
              ${e.canonical_sharpe != null ? `<div class="k">Sharpe / CI_lo</div><div>${fmtNum(e.canonical_sharpe, 4)} / ${fmtNum(e.canonical_ci_lo, 3)}</div>` : ""}
              ${e.plateau_spread_pct != null ? `<div class="k">Plateau spread</div><div>${fmtPct(e.plateau_spread_pct)}  (L27 gate <30%) — ${e.plateau_gate_passed ? "PASSED" : "FAILED"}</div>` : ""}
              <div class="k">Matrix verdict</div><div>${e.matrix_verdict}</div>
              <div class="k">CI gate verdict</div><div>${e.ci_gate_verdict}</div>
              <div class="k">Binding constraint (L46)</div><div>${e.binding_constraint}</div>
              <div class="k">Status</div><div><span class="status-pill ${sKey}">${e.deployment_status}</span></div>
              ${e.notes ? `<div class="k">Notes</div><div>${e.notes}</div>` : ""}
            </div>
            ${e.new_lessons && e.new_lessons.length ? `
              <div class="lessons">
                <div class="k" style="color: var(--muted); margin-right: 8px;">Lessons:</div>
                ${e.new_lessons.map(l => `<span class="lesson-pill">${l}</span>`).join("")}
              </div>` : ""}
          </div>
        </td>
      </tr>
    `;
  }).join("");

  // Wire row clicks.
  tbody.querySelectorAll("tr[data-id]").forEach(tr => {
    tr.addEventListener("click", () => {
      const id = tr.dataset.id;
      const detail = tbody.querySelector(`tr[data-detail-for="${id}"]`);
      const showing = detail.style.display !== "none";
      detail.style.display = showing ? "none" : "table-row";
      tr.classList.toggle("expanded", !showing);
    });
  });
}

filterInput.addEventListener("input", render);
statusFilter.addEventListener("change", render);

document.querySelectorAll("th[data-sort]").forEach(th => {
  th.addEventListener("click", () => {
    const k = th.dataset.sort;
    if (sortKey === k) {
      sortAsc = !sortAsc;
    } else {
      sortKey = k;
      sortAsc = true;
    }
    render();
  });
});

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
