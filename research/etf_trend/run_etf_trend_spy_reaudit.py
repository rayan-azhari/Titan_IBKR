"""etf_trend SPY V1-era re-audit harness.

Specified by ``directives/Pre-Reg etf_trend SPY Re-audit 2026-05-16.md``.

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/etf_trend/run_etf_trend_spy_reaudit.py

Pipeline (V3.6 + L52 hybrid + L17 relative MC):

1. Load SPY daily closes.
2. ``slice_sanctuary(months=24)``.
3. Causality smoke.
4. Build WFO folds (DAILY_TREND defaults, auto_fold_count).
5. **Pass 1** — per-cell stitched OOS Sharpe + bootstrap CI.
6. **L52 plateau pre-flight** — strict 30% gate on stitched OOS spread.
7. **L53 early gate** — skip Pass 2 if no cell can plausibly clear CI_lo > 0.
8. **Pass 2** — relative-MC vs B&H (L17), DSR, sanctuary divergence,
   noise robustness, decide.
9. Write §4 result log.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.etf_trend.etf_trend_spy_strategy import (  # noqa: E402
    EtfTrendSpyConfig,
    buy_and_hold_spy_returns,
    etf_trend_spy_assert_causal,
    etf_trend_spy_returns,
)
from titan.research.framework import (  # noqa: E402
    DecisionInputs,
    NoiseConfig,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    pass1_can_clear_any_cell,
    run_noise_robustness,
    run_relative_block_mc,
    sanctuary_divergence_test,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "etf_trend_spy_reaudit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24

CELLS: dict[str, EtfTrendSpyConfig] = {
    "C1_canonical": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5),
    "P_ec1": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=1),
    "P_ec2": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=2),
    "P_ec3": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=3),
    "P_ec10": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=10),
    "C2_live_canonical": EtfTrendSpyConfig(slow_ma=150, exit_confirm_days=1),
    # C3_buy_and_hold uses the dedicated buy_and_hold_spy_returns fn — params
    # carry through for vol-target/leverage parity.
    "C3_buy_and_hold": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5),
    "C4_gross_no_costs": EtfTrendSpyConfig(slow_ma=300, exit_confirm_days=5, apply_costs=False),
}
CANONICAL_CELL = "C1_canonical"
PLATEAU_CELLS = ("C1_canonical", "P_ec1", "P_ec2", "P_ec3", "P_ec10")
EXCLUDED_FROM_PROMOTION = ("C2_live_canonical", "C3_buy_and_hold", "C4_gross_no_costs")


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    return pd.DataFrame({"SPY": _load_close("SPY")})


@dataclass
class CellRow:
    name: str
    cfg: EtfTrendSpyConfig
    is_buy_and_hold: bool
    n_oos_bars: int
    n_folds: int
    sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    rel_mc_median_dd_reduction: float
    rel_mc_p_strategy_better: float
    rel_mc_passes: bool
    sanctuary_sharpe: float
    sanctuary_percentile: float
    sanctuary_lucky: bool
    noise_base: float
    noise_passes_mean: bool
    noise_passes_worst: bool
    noise_axis: str
    verdict: str


def _stitched_oos_returns(full_returns: pd.Series, folds: list) -> pd.Series:
    parts = [full_returns.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    return stitched[~stitched.index.duplicated(keep="last")].sort_index()


def _returns_for_cell(closes: pd.DataFrame, name: str, cfg: EtfTrendSpyConfig) -> pd.Series:
    if name == "C3_buy_and_hold":
        return buy_and_hold_spy_returns(closes, cfg=cfg)
    return etf_trend_spy_returns(closes, cfg=cfg)


def main() -> None:
    print("=" * 72)
    print("etf_trend SPY V3.6 RE-AUDIT")
    print("=" * 72)

    closes = load_universe()
    print(f"[load] SPY: {closes.shape[0]} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")

    sanc = slice_sanctuary(closes, months=SANCTUARY_MONTHS)
    visible = sanc.visible
    print(f"[sanctuary] visible: {visible.shape[0]} bars "
          f"({visible.index[0].date()} -> {visible.index[-1].date()})")
    print(f"[sanctuary] held out: {sanc.sanctuary.shape[0]} bars "
          f"({sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()})")

    etf_trend_spy_assert_causal(closes, cfg=CELLS[CANONICAL_CELL])
    print("[causality] assert_causal: PASS")

    class_def = defaults_for(StrategyClass.DAILY_TREND)
    wfo_cfg = class_def.wfo
    mc_cfg = class_def.mc
    folds = build_folds(visible.index, wfo_cfg, bars_per_year=252)
    print(f"[wfo] {len(folds)} folds (mode={wfo_cfg.is_mode}, "
          f"is_min={wfo_cfg.is_min_years}y, oos={wfo_cfg.oos_years}y)")
    if not folds:
        print("[wfo] No folds — abort.")
        return

    # ── Pass 1 ────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("Pass 1 — per-cell stitched OOS Sharpe + CI")
    print("-" * 72)
    pass1_returns: dict[str, pd.Series] = {}
    pass1_sharpes: dict[str, float] = {}
    pass1_cis: dict[str, tuple[float, float]] = {}
    for name, cfg in CELLS.items():
        full = _returns_for_cell(visible, name, cfg)
        stitched = _stitched_oos_returns(full, folds)
        sr = float(sharpe(stitched, periods_per_year=252))
        ci_lo, ci_hi = bootstrap_sharpe_ci(stitched, periods_per_year=252, seed=42)
        pass1_returns[name] = stitched
        pass1_sharpes[name] = sr
        pass1_cis[name] = (ci_lo, ci_hi)
        print(f"  {name:>20s}  sharpe={sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"n_oos={len(stitched.dropna())}")

    # ── L52 plateau pre-flight ────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("L52 plateau pre-flight (stitched OOS, strict L27 gate)")
    print("-" * 72)
    plateau_sharpes = [pass1_sharpes[c] for c in PLATEAU_CELLS]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    for c, s in zip(PLATEAU_CELLS, plateau_sharpes, strict=True):
        print(f"  {c:>20s}  sharpe={s:+.4f}")
    print(f"  spread = {plateau_spread * 100:.2f}%  (H1 gate: 30%)")
    plateau_pass = plateau_spread <= 0.30
    if not plateau_pass:
        print("  L52 H1 FAILED — IS plateau did NOT hold OOS. Aborting Pass 2.")
        _write_report_minimal(pass1_sharpes, pass1_cis, plateau_spread,
                              verdict_global="RETIRED (L52 H1 plateau failed)")
        return

    # ── L53 early gate ────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("L53 early gate")
    print("-" * 72)
    any_can_clear, gate_per_cell = pass1_can_clear_any_cell(
        pass1_returns, periods_per_year=252, block_size=mc_cfg.block_size_bars,
    )
    for name, gr in gate_per_cell.items():
        marker = "RUN P2" if gr.can_clear else "skip"
        print(f"  {name:>20s}  approx_ci_lo={gr.approx_ci_lo:+.4f}  -> {marker}")
    if not any_can_clear:
        print("  L53 — no cell can plausibly clear CI_lo > 0. Skipping Pass 2.")
        _write_report_minimal(pass1_sharpes, pass1_cis, plateau_spread,
                              verdict_global="RETIRED (L53 no cell can clear)")
        return

    # ── Pass 2 — relative MC + DSR + sanctuary + noise + decide ───────────
    print("\n" + "-" * 72)
    print("Pass 2 — relative MC (vs B&H), DSR, sanctuary, noise, decide")
    print("-" * 72)

    sanctuary_sharpe_by_cell: dict[str, float] = {}
    sanctuary_pct_by_cell: dict[str, float] = {}
    sanctuary_lucky_by_cell: dict[str, bool] = {}
    for name, cfg in CELLS.items():
        full = _returns_for_cell(closes, name, cfg)
        sanc_ret = full.loc[sanc.sanctuary_start :]
        sanc_sr = float(sharpe(sanc_ret, periods_per_year=252))
        sanctuary_sharpe_by_cell[name] = sanc_sr
        div = sanctuary_divergence_test(
            historical_returns=pass1_returns[name].dropna(),
            sanctuary_returns=sanc_ret.dropna(),
            periods_per_year=252,
        )
        sanctuary_pct_by_cell[name] = float(div.percentile) if np.isfinite(div.percentile) else float("nan")
        sanctuary_lucky_by_cell[name] = bool(div.lucky_flag)

    # B&H benchmark for relative MC.
    def benchmark_for_mc(df: pd.DataFrame) -> pd.Series:
        # df.columns = ('close',) since no extras.
        return buy_and_hold_spy_returns(df.rename(columns={"close": "SPY"}))

    cell_rows: list[CellRow] = []
    for name, cfg in CELLS.items():
        sr_pass1 = pass1_sharpes[name]
        ci_lo, ci_hi = pass1_cis[name]

        def strategy_for_mc(df: pd.DataFrame, _name: str = name, _cfg: EtfTrendSpyConfig = cfg) -> pd.Series:
            renamed = df.rename(columns={"close": "SPY"})
            if _name == "C3_buy_and_hold":
                return buy_and_hold_spy_returns(renamed, cfg=_cfg)
            return etf_trend_spy_returns(renamed, cfg=_cfg)

        rel_mc = run_relative_block_mc(
            visible["SPY"],
            mc_cfg,
            strategy_for_mc,
            benchmark_for_mc,
            periods_per_year=252,
            seed=42,
            n_workers=DEFAULT_MC_WORKERS,
            # L17 default gates: median_dd_reduction <= 0.80, p_better >= 0.50.
            median_ratio_gate=0.80,
            p_strategy_better_gate=0.50,
        )

        # DSR.
        sweep_sharpes_for_dsr = [pass1_sharpes[c] for c in PLATEAU_CELLS]
        sr_var = sr_var_from_sweep(sweep_sharpes_for_dsr)
        dsr = deflated_sharpe(
            sr_hat=sr_pass1,
            sr_var_across_trials=sr_var,
            returns=pass1_returns[name],
            n_trials=len(PLATEAU_CELLS),
        )

        # Noise robustness.
        def strategy_for_noise(closes_subset: pd.DataFrame, _name: str = name,
                               _cfg: EtfTrendSpyConfig = cfg) -> pd.Series:
            return _returns_for_cell(closes_subset, _name, _cfg)

        noise_res = run_noise_robustness(
            visible, strategy_for_noise, periods_per_year=252, cfg=NoiseConfig(),
        )

        # Decide.
        # For the MC axis, we use p_strategy_better (L17 metric) re-mapped
        # to the existing DecisionInputs schema: pass_threshold_prob remains
        # the class default; p_maxdd_gt_threshold is set such that the axis
        # passes iff rel_mc passes (use 0.0 if pass, 1.0 if fail).
        synthetic_mc_p = 0.0 if rel_mc.passes else 1.0
        decision = decide(
            DecisionInputs(
                ci_lo=ci_lo,
                dsr_prob=dsr.dsr_prob,
                p_maxdd_gt_threshold=synthetic_mc_p,
                pass_threshold_prob=mc_cfg.max_dd_pass_prob,
                sanctuary_sharpe=sanctuary_sharpe_by_cell[name],
                noise_passes_mean=noise_res.passes,
                noise_passes_worst=noise_res.worst_case_passes,
            )
        )

        cell_rows.append(
            CellRow(
                name=name,
                cfg=cfg,
                is_buy_and_hold=(name == "C3_buy_and_hold"),
                n_oos_bars=len(pass1_returns[name].dropna()),
                n_folds=len(folds),
                sharpe=sr_pass1,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                dsr_prob=dsr.dsr_prob,
                rel_mc_median_dd_reduction=rel_mc.median_dd_reduction,
                rel_mc_p_strategy_better=rel_mc.p_strategy_better,
                rel_mc_passes=rel_mc.passes,
                sanctuary_sharpe=sanctuary_sharpe_by_cell[name],
                sanctuary_percentile=sanctuary_pct_by_cell[name],
                sanctuary_lucky=sanctuary_lucky_by_cell[name],
                noise_base=noise_res.base_sharpe,
                noise_passes_mean=noise_res.passes,
                noise_passes_worst=noise_res.worst_case_passes,
                noise_axis=decision.noise_axis,
                verdict=decision.verdict.value,
            )
        )
        print(f"  {name:>20s}  verdict={decision.verdict.value}  "
              f"CI_lo={ci_lo:+.3f}  sanc_sr={sanctuary_sharpe_by_cell[name]:+.3f}  "
              f"rel_dd={rel_mc.median_dd_reduction:.3f}  "
              f"rel_pass={rel_mc.passes}  noise={decision.noise_axis}")

    # ── Selection per §3 rule ─────────────────────────────────────────────
    eligible = [
        r for r in cell_rows
        if r.name not in EXCLUDED_FROM_PROMOTION
        and r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT")
    ]
    selected = max(eligible, key=lambda r: r.ci_lo) if eligible else None
    if selected is not None and selected.ci_lo <= 0:
        verdict_global = (
            f"NOT PROMOTED — {selected.name} passes 5-axis ({selected.verdict}) "
            f"but CI_lo={selected.ci_lo:+.3f} <= 0 (L46 binds)"
        )
    elif selected is None:
        verdict_global = "RETIRED — no cell DEPLOY or CONDITIONAL_WATCHPOINT"
    else:
        verdict_global = f"PROMOTED — {selected.name} verdict={selected.verdict}, CI_lo={selected.ci_lo:+.3f}"

    print("\n" + "=" * 72)
    print(f"FINAL VERDICT: {verdict_global}")
    print("=" * 72)

    _write_report(plateau_spread, cell_rows, verdict_global)


def _write_report_minimal(pass1_sharpes, pass1_cis, plateau_spread, verdict_global) -> None:
    fp = REPORTS_DIR / "result_log.md"
    lines = [
        "# etf_trend SPY V3.6 Re-Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        f"**Verdict:** {verdict_global}",
        "",
        "## §4.1 Pass-1 stitched OOS Sharpes",
        "",
        "| Cell | Sharpe | CI95_lo | CI95_hi |",
        "|---|---:|---:|---:|",
    ]
    for name, sr in pass1_sharpes.items():
        ci_lo, ci_hi = pass1_cis[name]
        lines.append(f"| {name} | {sr:+.4f} | {ci_lo:+.3f} | {ci_hi:+.3f} |")
    lines.append("")
    lines.append(f"## §4.2 L52 plateau pre-flight — spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append("Pass 2 SKIPPED.")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


def _write_report(plateau_spread, cell_rows, verdict_global) -> None:
    fp = REPORTS_DIR / "result_log.md"
    canonical = next(r for r in cell_rows if r.name == "C1_canonical")
    live = next(r for r in cell_rows if r.name == "C2_live_canonical")
    bh = next(r for r in cell_rows if r.name == "C3_buy_and_hold")
    gross = next(r for r in cell_rows if r.name == "C4_gross_no_costs")

    lines = [
        "# etf_trend SPY V3.6 Re-Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        "**Strategy class:** DAILY_TREND",
        "**Pre-reg:** `directives/Pre-Reg etf_trend SPY Re-audit 2026-05-16.md`",
        "**Sweep informant:** `.tmp/reports/sweep_etf_trend_spy/findings.md`",
        f"**Final verdict:** **{verdict_global}**",
        "",
        "## §4.1 Pass 1 — stitched OOS Sharpe + bootstrap CI",
        "",
        "| Cell | Sharpe | CI95_lo | CI95_hi | n_oos |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | {r.n_oos_bars} |"
        )
    lines.append("")
    lines.append(f"## §4.2 L52 plateau pre-flight — spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append(f"- L27 strict gate (≤ 30%): **{'PASS' if plateau_spread <= 0.30 else 'FAIL'}**")
    lines.append("")
    lines.append("## §4.3 Per-cell 5-axis matrix (with L17 relative MC)")
    lines.append("")
    lines.append(
        "| Cell | Sharpe | CI_lo | CI_hi | DSR | Rel-MC DD reduction | P(strat better) | Rel-MC pass | Sanc Sharpe | Sanc %ile | Lucky | Noise axis | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|---|")
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
            f"{r.dsr_prob:.3f} | {r.rel_mc_median_dd_reduction:.3f} | "
            f"{r.rel_mc_p_strategy_better:.3f} | {'YES' if r.rel_mc_passes else 'no'} | "
            f"{r.sanctuary_sharpe:+.3f} | {r.sanctuary_percentile:.2f} | "
            f"{'YES' if r.sanctuary_lucky else 'no'} | {r.noise_axis} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## §4.4 Falsification hypothesis verdicts")
    lines.append("")
    h1_msg = (
        f"H1 (plateau holds OOS, spread ≤ 30%): "
        f"**{'SUPPORTED' if plateau_spread <= 0.30 else 'REJECTED'}** "
        f"— spread = {plateau_spread * 100:.2f}%"
    )
    h2_msg = (
        f"H2 (canonical OOS Sharpe ≥ +0.30): "
        f"**{'SUPPORTED' if canonical.sharpe >= 0.30 else 'REJECTED'}** "
        f"— OOS Sharpe = {canonical.sharpe:+.3f}"
    )
    h3_msg = (
        f"H3 (C1 OOS Sharpe > C2 OOS Sharpe): "
        f"**{'SUPPORTED' if canonical.sharpe > live.sharpe else 'REJECTED'}** "
        f"— C1 {canonical.sharpe:+.3f} vs C2 {live.sharpe:+.3f}"
    )
    h4_msg = (
        f"H4 (C1 CI_lo > 0): **{'SUPPORTED' if canonical.ci_lo > 0 else 'REJECTED'}** "
        f"— CI_lo = {canonical.ci_lo:+.3f}"
    )
    h5_msg = (
        f"H5 (MaxDD reduction ≤ 0.80 vs B&H): "
        f"**{'SUPPORTED' if canonical.rel_mc_passes else 'REJECTED'}** "
        f"— median DD reduction = {canonical.rel_mc_median_dd_reduction:.3f}, "
        f"p_strategy_better = {canonical.rel_mc_p_strategy_better:.3f}"
    )
    h6_gap = gross.sharpe - canonical.sharpe
    h6_msg = (
        f"H6 (cost drag ≤ 0.10): **{'SUPPORTED' if h6_gap <= 0.10 else 'REJECTED'}** "
        f"— gross − net = {h6_gap:+.3f}"
    )
    for m in (h1_msg, h2_msg, h3_msg, h4_msg, h5_msg, h6_msg):
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## §4.5 Promotion verdict")
    lines.append("")
    lines.append(f"**{verdict_global}**")
    lines.append("")
    if canonical.sanctuary_lucky:
        lines.append(
            "**L55 caveat applies** — sanctuary `lucky_flag=True` "
            f"(percentile {canonical.sanctuary_percentile:.2f}). Cite stitched OOS Sharpe "
            f"(+{canonical.sharpe:.3f}) NOT sanctuary Sharpe (+{canonical.sanctuary_sharpe:.3f}) "
            "as deployment-relevant number."
        )
        lines.append("")
    lines.append("## §4.6 Buy-and-hold baseline")
    lines.append("")
    lines.append(
        f"C3 (B&H SPY vol-targeted): stitched OOS Sharpe = {bh.sharpe:+.3f}, "
        f"sanctuary Sharpe = {bh.sanctuary_sharpe:+.3f}. The strategy's economic claim is "
        f"MaxDD reduction vs B&H — H5 verdict above is the deployment-relevant test."
    )
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
