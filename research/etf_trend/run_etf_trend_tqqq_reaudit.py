"""etf_trend TQQQ V1-era re-audit harness (Wave A.2-confirm).

Purpose: confirm or refute that L56 generalises from etf_trend_spy to the
leveraged TQQQ variant. If TQQQ — where the trend filter is most PLAUSIBLE
to add value (3x leverage means catastrophic drawdowns under B&H) — also
fails L17 relative MC, the entire etf_trend strategy class can be bulk-
retired without auditing the remaining 5 variants.

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/etf_trend/run_etf_trend_tqqq_reaudit.py
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

from research.etf_trend.etf_trend_tqqq_strategy import (  # noqa: E402
    EtfTrendTqqqConfig,
    buy_and_hold_tqqq_returns,
    etf_trend_tqqq_assert_causal,
    etf_trend_tqqq_returns,
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "etf_trend_tqqq_reaudit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24

# §2 cells: focus on the best-Sharpe region (slow_ma=150) per sweep findings.
# Live config uses slow_ma=175; closest grid points are 150 and 200.
CELLS: dict[str, EtfTrendTqqqConfig] = {
    "C1_canonical": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=1),  # sweep best
    "P_ec2": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=2),
    "P_ec3": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=3),
    "P_ec5": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=5),
    "P_ec10": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=10),
    "C2_live_proxy": EtfTrendTqqqConfig(slow_ma=200, exit_confirm_days=1),  # nearest to live (175,1)
    "C3_buy_and_hold": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=1),
    "C4_gross_no_costs": EtfTrendTqqqConfig(slow_ma=150, exit_confirm_days=1, apply_costs=False),
}
CANONICAL_CELL = "C1_canonical"
PLATEAU_CELLS = ("C1_canonical", "P_ec2", "P_ec3", "P_ec5", "P_ec10")
EXCLUDED_FROM_PROMOTION = ("C2_live_proxy", "C3_buy_and_hold", "C4_gross_no_costs")


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    qqq = _load_close("QQQ")
    tqqq = _load_close("TQQQ")
    common = qqq.index.intersection(tqqq.index)
    return pd.DataFrame({"QQQ": qqq.reindex(common), "TQQQ": tqqq.reindex(common)}).dropna()


@dataclass
class CellRow:
    name: str
    cfg: EtfTrendTqqqConfig
    is_bh: bool
    n_oos_bars: int
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
    noise_axis: str
    verdict: str


def _stitched_oos_returns(full_returns: pd.Series, folds: list) -> pd.Series:
    parts = [full_returns.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    return stitched[~stitched.index.duplicated(keep="last")].sort_index()


def _returns_for_cell(closes: pd.DataFrame, name: str, cfg: EtfTrendTqqqConfig) -> pd.Series:
    if name == "C3_buy_and_hold":
        return buy_and_hold_tqqq_returns(closes, cfg=cfg)
    return etf_trend_tqqq_returns(closes, cfg=cfg)


def main() -> None:
    print("=" * 72)
    print("etf_trend TQQQ V3.6 RE-AUDIT (Wave A.2-confirm)")
    print("=" * 72)

    closes = load_universe()
    print(f"[load] QQQ+TQQQ: {closes.shape[0]} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")

    sanc = slice_sanctuary(closes, months=SANCTUARY_MONTHS)
    visible = sanc.visible
    print(f"[sanctuary] visible: {visible.shape[0]} bars")
    print(f"[sanctuary] held out: {sanc.sanctuary.shape[0]} bars "
          f"({sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()})")

    etf_trend_tqqq_assert_causal(closes, cfg=CELLS[CANONICAL_CELL])
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
    print("L52 plateau pre-flight")
    print("-" * 72)
    plateau_sharpes = [pass1_sharpes[c] for c in PLATEAU_CELLS]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    for c, s in zip(PLATEAU_CELLS, plateau_sharpes, strict=True):
        print(f"  {c:>20s}  sharpe={s:+.4f}")
    print(f"  spread = {plateau_spread * 100:.2f}%  (H1 gate: 30%; "
          "Wave A.2-confirm uses strict L27 since this is a leveraged-ETF audit)")
    plateau_pass = plateau_spread <= 0.30

    # NOTE: even if plateau fails, we run Pass 2 anyway for the L17 rel-MC
    # data (the confirmation question is whether L17 generalises — not whether
    # the plateau holds).

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

    # ── Pass 2 ────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("Pass 2 — relative MC (vs B&H TQQQ), DSR, sanctuary, noise, decide")
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

    # Benchmark: B&H TQQQ on the synthetic path.
    def benchmark_for_mc(df: pd.DataFrame) -> pd.Series:
        # df.columns = ('close', 'TQQQ') for shared_block MC with extras.
        # primary close = QQQ (signal source); extras has TQQQ.
        renamed = pd.DataFrame({"QQQ": df["close"], "TQQQ": df["TQQQ"]})
        return buy_and_hold_tqqq_returns(renamed)

    cell_rows: list[CellRow] = []
    for name, cfg in CELLS.items():
        sr_pass1 = pass1_sharpes[name]
        ci_lo, ci_hi = pass1_cis[name]

        def strategy_for_mc(df: pd.DataFrame, _name: str = name,
                            _cfg: EtfTrendTqqqConfig = cfg) -> pd.Series:
            renamed = pd.DataFrame({"QQQ": df["close"], "TQQQ": df["TQQQ"]})
            if _name == "C3_buy_and_hold":
                return buy_and_hold_tqqq_returns(renamed, cfg=_cfg)
            return etf_trend_tqqq_returns(renamed, cfg=_cfg)

        rel_mc = run_relative_block_mc(
            visible["QQQ"],
            mc_cfg,
            strategy_for_mc,
            benchmark_for_mc,
            periods_per_year=252,
            seed=42,
            extra_series={"TQQQ": visible["TQQQ"]},
            n_workers=DEFAULT_MC_WORKERS,
            median_ratio_gate=0.80,
            p_strategy_better_gate=0.50,
        )

        sweep_sharpes_for_dsr = [pass1_sharpes[c] for c in PLATEAU_CELLS]
        sr_var = sr_var_from_sweep(sweep_sharpes_for_dsr)
        dsr = deflated_sharpe(
            sr_hat=sr_pass1,
            sr_var_across_trials=sr_var,
            returns=pass1_returns[name],
            n_trials=len(PLATEAU_CELLS),
        )

        def strategy_for_noise(closes_subset: pd.DataFrame, _name: str = name,
                               _cfg: EtfTrendTqqqConfig = cfg) -> pd.Series:
            return _returns_for_cell(closes_subset, _name, _cfg)

        noise_res = run_noise_robustness(
            visible, strategy_for_noise, periods_per_year=252, cfg=NoiseConfig(),
        )

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
                name=name, cfg=cfg,
                is_bh=(name == "C3_buy_and_hold"),
                n_oos_bars=len(pass1_returns[name].dropna()),
                sharpe=sr_pass1, ci_lo=ci_lo, ci_hi=ci_hi,
                dsr_prob=dsr.dsr_prob,
                rel_mc_median_dd_reduction=rel_mc.median_dd_reduction,
                rel_mc_p_strategy_better=rel_mc.p_strategy_better,
                rel_mc_passes=rel_mc.passes,
                sanctuary_sharpe=sanctuary_sharpe_by_cell[name],
                sanctuary_percentile=sanctuary_pct_by_cell[name],
                sanctuary_lucky=sanctuary_lucky_by_cell[name],
                noise_base=noise_res.base_sharpe,
                noise_axis=decision.noise_axis,
                verdict=decision.verdict.value,
            )
        )
        print(f"  {name:>20s}  verdict={decision.verdict.value}  CI_lo={ci_lo:+.3f}  "
              f"rel_dd={rel_mc.median_dd_reduction:.3f}  rel_pass={rel_mc.passes}  "
              f"noise={decision.noise_axis}")

    # Selection.
    eligible = [
        r for r in cell_rows
        if r.name not in EXCLUDED_FROM_PROMOTION
        and r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT")
    ]
    selected = max(eligible, key=lambda r: r.ci_lo) if eligible else None
    if selected is not None and selected.ci_lo <= 0:
        verdict_global = (
            f"NOT PROMOTED — {selected.name} passes 5-axis ({selected.verdict}) "
            f"but CI_lo={selected.ci_lo:+.3f} <= 0"
        )
    elif selected is None:
        verdict_global = "RETIRED — no cell DEPLOY or CONDITIONAL_WATCHPOINT"
    else:
        verdict_global = f"PROMOTED — {selected.name} verdict={selected.verdict}"

    print("\n" + "=" * 72)
    print(f"FINAL VERDICT: {verdict_global}")
    print("=" * 72)

    # L56 generalisation test:
    canonical = next(r for r in cell_rows if r.name == CANONICAL_CELL)
    l56_confirmed = (not canonical.rel_mc_passes) and (canonical.noise_axis == "worst")
    print()
    if l56_confirmed:
        print("*** L56 CONFIRMED on TQQQ ***")
        print("    Rel-MC fails AND noise axis = worst on the canonical cell.")
        print("    The etf_trend strategy class can be BULK RETIRED.")
    else:
        print("*** L56 PARTIALLY REFUTED on TQQQ ***")
        print(f"    rel_mc.passes={canonical.rel_mc_passes}, noise_axis={canonical.noise_axis}")
        print("    TQQQ behaves DIFFERENTLY than SPY; cannot bulk-retire the class.")
        print("    Recommend auditing each remaining variant individually.")

    _write_report(plateau_spread, plateau_pass, cell_rows, verdict_global, l56_confirmed)


def _write_report(plateau_spread, plateau_pass, cell_rows, verdict_global, l56_confirmed) -> None:
    fp = REPORTS_DIR / "result_log.md"
    canonical = next(r for r in cell_rows if r.name == "C1_canonical")
    live = next(r for r in cell_rows if r.name == "C2_live_proxy")
    bh = next(r for r in cell_rows if r.name == "C3_buy_and_hold")
    gross = next(r for r in cell_rows if r.name == "C4_gross_no_costs")

    lines = [
        "# etf_trend TQQQ V3.6 Re-Audit Result Log (Wave A.2-confirm)",
        "",
        "**Run date:** 2026-05-16",
        "**Strategy class:** DAILY_TREND",
        "**Purpose:** confirm or refute L56 generalisation from SPY to TQQQ.",
        "**Sweep informant:** `.tmp/reports/sweep_etf_trend_tqqq/plateau_report.md`",
        f"**Final verdict:** **{verdict_global}**",
        f"**L56 generalisation:** **{'CONFIRMED — bulk-retire etf_trend class' if l56_confirmed else 'PARTIALLY REFUTED — TQQQ behaves differently'}**",
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
    lines.append(f"- L27 strict gate (≤ 30%): **{'PASS' if plateau_pass else 'FAIL'}**")
    lines.append("")
    lines.append("## §4.3 Per-cell 5-axis matrix (with L17 relative MC vs B&H TQQQ)")
    lines.append("")
    lines.append(
        "| Cell | Sharpe | CI_lo | DSR | Rel-MC DD red | P(strat better) | Rel-MC pass | Sanc Sharpe | Sanc %ile | Lucky | Noise axis | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|---|")
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.dsr_prob:.3f} | "
            f"{r.rel_mc_median_dd_reduction:.3f} | {r.rel_mc_p_strategy_better:.3f} | "
            f"{'YES' if r.rel_mc_passes else 'no'} | {r.sanctuary_sharpe:+.3f} | "
            f"{r.sanctuary_percentile:.2f} | {'YES' if r.sanctuary_lucky else 'no'} | "
            f"{r.noise_axis} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## §4.4 L56 generalisation verdict")
    lines.append("")
    if l56_confirmed:
        lines.append("**L56 CONFIRMED on TQQQ.** The leveraged-ETF variant fails L17 relative")
        lines.append("MC just like SPY did — the trend filter does NOT statistically reduce")
        lines.append("drawdown vs B&H TQQQ under bootstrap. **Bulk-retire the etf_trend class.**")
    else:
        lines.append("**L56 PARTIALLY REFUTED on TQQQ.**")
        lines.append(f"- Canonical rel_mc.passes = {canonical.rel_mc_passes}")
        lines.append(f"- Canonical noise axis = {canonical.noise_axis}")
        lines.append("TQQQ may genuinely add value over B&H under bootstrap (likely because the")
        lines.append("3x leverage decay makes B&H so catastrophic). Audit remaining variants individually.")
    lines.append("")
    lines.append("## §4.5 Comparison to SPY audit")
    lines.append("")
    lines.append(
        f"- SPY canonical (300, 5): OOS Sharpe +0.72 vs B&H +0.68 (+6%); rel-MC FAIL (DD reduction 1.01)\n"
        f"- TQQQ canonical (150, 1): OOS Sharpe {canonical.sharpe:+.3f} vs B&H {bh.sharpe:+.3f} "
        f"({(canonical.sharpe - bh.sharpe) / max(abs(bh.sharpe), 1e-9) * 100:+.0f}%); "
        f"rel-MC {'PASS' if canonical.rel_mc_passes else 'FAIL'} (DD reduction {canonical.rel_mc_median_dd_reduction:.3f})\n"
        f"- Live TQQQ proxy (200, 1): OOS Sharpe {live.sharpe:+.3f}, "
        f"rel-MC {'PASS' if live.rel_mc_passes else 'FAIL'}"
    )
    lines.append("")
    lines.append("## §4.6 Cost drag")
    lines.append("")
    lines.append(f"C4 (gross, no costs): OOS Sharpe {gross.sharpe:+.3f}; gap to net = {gross.sharpe - canonical.sharpe:+.3f}")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
