"""GEM J5 Hybrid Re-audit harness.

Specified by ``directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md``.

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/gem/run_gem_j5_reaudit.py

Pipeline (V3.6 + L52 + L17):

1. Load SPY + EFA + IEF daily TR closes.
2. ``slice_sanctuary(months=12)``.
3. Causality smoke (``gem_assert_causal``).
4. Build WFO folds (CROSS_ASSET_MOMENTUM defaults, auto_fold_count).
5. **Pass 1** — per-cell stitched OOS Sharpe + bootstrap CI.
6. **L52 plateau pre-flight** (strict 30% gate).
7. **L53 early gate**.
8. **Pass 2** — L17 relative MC vs 60/40 SPY+IEF B&H, DSR, sanctuary,
   noise, decide.
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

from research.gem.gem_strategy import GemConfig, gem_assert_causal, gem_returns  # noqa: E402
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "gem_j5_reaudit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12

# Frozen knobs (mirror config/gem_voltarget_lev2.toml minus swept axes).
COST_BPS = 6.0
COST_FIXED_USD = 1.0
NOTIONAL = 30_000.0


def _make_cfg(*, halflife: int, vol_target: float) -> GemConfig:
    return GemConfig(
        lookback_blend=(3, 6, 12),
        absolute_gate_lookback_months=12,
        buffer_pct=0.005,
        defensive_switch=True,
        ann_vol_target=vol_target,
        vol_lookback_days=20,
        max_leverage=2.0,
        vol_estimator_kind="ewma",
        vol_estimator_halflife=halflife,
        stress_gate_enabled=False,
        dd_breaker_enabled=False,
    )


CELLS: dict[str, GemConfig] = {
    "C1_canonical": _make_cfg(halflife=20, vol_target=0.05),
    "P_hl10_vt05": _make_cfg(halflife=10, vol_target=0.05),
    "P_hl40_vt05": _make_cfg(halflife=40, vol_target=0.05),
    "P_hl60_vt05": _make_cfg(halflife=60, vol_target=0.05),
    "P_hl20_vt075": _make_cfg(halflife=20, vol_target=0.075),
    "C2_constrained": _make_cfg(halflife=20, vol_target=0.10),
    "C3_J4_live": _make_cfg(halflife=40, vol_target=0.10),
    "C4_gross_no_costs": _make_cfg(halflife=20, vol_target=0.05),
}
CANONICAL_CELL = "C1_canonical"
PLATEAU_CELLS = ("C1_canonical", "P_hl10_vt05", "P_hl40_vt05", "P_hl60_vt05", "P_hl20_vt075")
EXCLUDED_FROM_PROMOTION = ("C3_J4_live", "C4_gross_no_costs")


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    spy = _load_close("SPY")
    efa = _load_close("EFA")
    ief = _load_close("IEF")
    common = spy.index.intersection(efa.index).intersection(ief.index)
    return pd.DataFrame(
        {"SPY": spy.reindex(common), "EFA": efa.reindex(common), "IEF": ief.reindex(common)}
    ).dropna()


@dataclass
class CellRow:
    name: str
    cfg: GemConfig
    is_gross: bool
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


def _returns_for_cell(closes: pd.DataFrame, name: str, cfg: GemConfig) -> pd.Series:
    apply_costs = name != "C4_gross_no_costs"
    return gem_returns(
        closes,
        cfg=cfg,
        cost_bps_per_turnover=COST_BPS if apply_costs else 0.0,
        cost_fixed_usd_per_fill=COST_FIXED_USD if apply_costs else 0.0,
        notional_usd=NOTIONAL,
        execution_mode="etf",
        rebalance_threshold=0.05,
    ).rename("ret")


def buy_and_hold_60_40(closes: pd.DataFrame) -> pd.Series:
    """60/40 SPY/IEF buy-and-hold — L17 benchmark for GEM.

    The "do nothing alternative" for a cross-asset momentum strategy on a
    SPY/EFA/IEF universe is a static 60/40 SPY/IEF allocation.
    Causality: position at t-1 earns return at t.
    """
    if "SPY" in closes.columns and "IEF" in closes.columns:
        spy = closes["SPY"]
        ief = closes["IEF"]
    elif "close" in closes.columns:
        # MC synthetic: primary = SPY, IEF in extras.
        spy = closes["close"]
        ief = closes["IEF"]
    else:
        raise ValueError(
            f"buy_and_hold_60_40 needs SPY+IEF or close+IEF; got {list(closes.columns)}"
        )
    spy_ret = np.log(spy / spy.shift(1)).fillna(0.0)
    ief_ret = np.log(ief / ief.shift(1)).fillna(0.0)
    # Static 60/40 (rebalanced daily — close enough for MC purposes).
    weight_spy = pd.Series(0.6, index=spy_ret.index)
    weight_ief = pd.Series(0.4, index=ief_ret.index)
    weight_spy.iloc[0] = 0.0
    weight_ief.iloc[0] = 0.0
    return (
        weight_spy.shift(1).fillna(0.0) * spy_ret + weight_ief.shift(1).fillna(0.0) * ief_ret
    ).rename("ret")


def main() -> None:
    print("=" * 72)
    print("GEM J5 HYBRID RE-AUDIT")
    print("=" * 72)

    closes = load_universe()
    print(
        f"[load] SPY+EFA+IEF: {closes.shape[0]} bars "
        f"({closes.index[0].date()} -> {closes.index[-1].date()})"
    )

    sanc = slice_sanctuary(closes, months=SANCTUARY_MONTHS)
    visible = sanc.visible
    print(f"[sanctuary] visible: {visible.shape[0]} bars")
    print(
        f"[sanctuary] held out: {sanc.sanctuary.shape[0]} bars "
        f"({sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()})"
    )

    gem_assert_causal(closes, cfg=CELLS[CANONICAL_CELL])
    print("[causality] gem_assert_causal: PASS")

    class_def = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    wfo_cfg = class_def.wfo
    mc_cfg = class_def.mc
    folds = build_folds(visible.index, wfo_cfg, bars_per_year=252)
    print(f"[wfo] {len(folds)} folds")
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
        print(
            f"  {name:>20s}  sharpe={sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
            f"n_oos={len(stitched.dropna())}"
        )

    # ── L52 plateau pre-flight ────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("L52 plateau pre-flight (strict L27 30% gate)")
    print("-" * 72)
    plateau_sharpes = [pass1_sharpes[c] for c in PLATEAU_CELLS]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    for c, s in zip(PLATEAU_CELLS, plateau_sharpes, strict=True):
        print(f"  {c:>20s}  sharpe={s:+.4f}")
    print(f"  spread = {plateau_spread * 100:.2f}%")
    plateau_pass = plateau_spread <= 0.30

    # ── L53 early gate ────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("L53 early gate")
    print("-" * 72)
    any_can_clear, gate_per_cell = pass1_can_clear_any_cell(
        pass1_returns,
        periods_per_year=252,
        block_size=mc_cfg.block_size_bars,
    )
    for name, gr in gate_per_cell.items():
        marker = "RUN P2" if gr.can_clear else "skip"
        print(f"  {name:>20s}  approx_ci_lo={gr.approx_ci_lo:+.4f}  -> {marker}")

    # ── Pass 2 ────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("Pass 2 — relative MC vs 60/40, DSR, sanctuary, noise, decide")
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
        sanctuary_pct_by_cell[name] = (
            float(div.percentile) if np.isfinite(div.percentile) else float("nan")
        )
        sanctuary_lucky_by_cell[name] = bool(div.lucky_flag)

    def benchmark_for_mc(df: pd.DataFrame) -> pd.Series:
        # df.columns = ('close', 'EFA', 'IEF') — primary is SPY (renamed to close).
        renamed = pd.DataFrame({"SPY": df["close"], "EFA": df["EFA"], "IEF": df["IEF"]})
        return buy_and_hold_60_40(renamed)

    cell_rows: list[CellRow] = []
    for name, cfg in CELLS.items():
        sr_pass1 = pass1_sharpes[name]
        ci_lo, ci_hi = pass1_cis[name]

        def strategy_for_mc(
            df: pd.DataFrame, _name: str = name, _cfg: GemConfig = cfg
        ) -> pd.Series:
            renamed = pd.DataFrame({"SPY": df["close"], "EFA": df["EFA"], "IEF": df["IEF"]})
            return _returns_for_cell(renamed, _name, _cfg)

        rel_mc = run_relative_block_mc(
            visible["SPY"],
            mc_cfg,
            strategy_for_mc,
            benchmark_for_mc,
            periods_per_year=252,
            seed=42,
            extra_series={"EFA": visible["EFA"], "IEF": visible["IEF"]},
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

        def strategy_for_noise(
            closes_subset: pd.DataFrame, _name: str = name, _cfg: GemConfig = cfg
        ) -> pd.Series:
            return _returns_for_cell(closes_subset, _name, _cfg)

        noise_res = run_noise_robustness(
            visible,
            strategy_for_noise,
            periods_per_year=252,
            cfg=NoiseConfig(),
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
                name=name,
                cfg=cfg,
                is_gross=(name == "C4_gross_no_costs"),
                n_oos_bars=len(pass1_returns[name].dropna()),
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
                noise_axis=decision.noise_axis,
                verdict=decision.verdict.value,
            )
        )
        print(
            f"  {name:>20s}  verdict={decision.verdict.value}  CI_lo={ci_lo:+.3f}  "
            f"rel_dd={rel_mc.median_dd_reduction:.3f}  rel_pass={rel_mc.passes}  "
            f"noise={decision.noise_axis}"
        )

    # Selection.
    eligible = [
        r
        for r in cell_rows
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
        verdict_global = "NOT PROMOTED — no cell DEPLOY or CONDITIONAL_WATCHPOINT (J4 live remains)"
    else:
        verdict_global = (
            f"PROMOTED — {selected.name} verdict={selected.verdict}, CI_lo={selected.ci_lo:+.3f}"
        )

    print("\n" + "=" * 72)
    print(f"FINAL VERDICT: {verdict_global}")
    print("=" * 72)

    # H3 comparison: did C1 beat C3 live baseline?
    c1 = next(r for r in cell_rows if r.name == "C1_canonical")
    c3 = next(r for r in cell_rows if r.name == "C3_J4_live")
    print("\nH3 comparison (C1 plateau vs C3 J4 live):")
    print(f"  C1 OOS Sharpe = {c1.sharpe:+.4f}, CI_lo = {c1.ci_lo:+.3f}")
    print(f"  C3 OOS Sharpe = {c3.sharpe:+.4f}, CI_lo = {c3.ci_lo:+.3f}")
    print(f"  Δ = {c1.sharpe - c3.sharpe:+.4f}")

    _write_report(plateau_spread, plateau_pass, cell_rows, verdict_global)


def _write_report(plateau_spread, plateau_pass, cell_rows, verdict_global) -> None:
    fp = REPORTS_DIR / "result_log.md"
    c1 = next(r for r in cell_rows if r.name == "C1_canonical")
    c2 = next(r for r in cell_rows if r.name == "C2_constrained")
    c3 = next(r for r in cell_rows if r.name == "C3_J4_live")
    c4 = next(r for r in cell_rows if r.name == "C4_gross_no_costs")

    lines = [
        "# GEM J5 Hybrid Re-Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        "**Strategy class:** CROSS_ASSET_MOMENTUM",
        "**Pre-reg:** `directives/Pre-Reg J5 GEM Hybrid Re-audit 2026-05-16.md`",
        "**Sweep informant:** `.tmp/reports/sweep_gem_hybrid/findings.md`",
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
    lines.append(f"- L27 strict gate (≤ 30%): **{'PASS' if plateau_pass else 'FAIL'}**")
    lines.append("")
    lines.append("## §4.3 Per-cell 5-axis matrix (with L17 relative MC vs 60/40 SPY/IEF)")
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
    lines.append("## §4.4 Falsification hypothesis verdicts")
    lines.append("")
    h1_msg = (
        f"H1 (plateau holds OOS, spread ≤ 30%): "
        f"**{'SUPPORTED' if plateau_pass else 'REJECTED'}** — spread = {plateau_spread * 100:.2f}%"
    )
    h2_msg = (
        f"H2 (canonical OOS Sharpe ≥ +0.50): "
        f"**{'SUPPORTED' if c1.sharpe >= 0.50 else 'REJECTED'}** — OOS Sharpe = {c1.sharpe:+.3f}"
    )
    h3_msg = (
        f"H3 (C1 OOS Sharpe > C3 OOS Sharpe): "
        f"**{'SUPPORTED' if c1.sharpe > c3.sharpe else 'REJECTED'}** "
        f"— C1 {c1.sharpe:+.3f} vs C3 (J4 live) {c3.sharpe:+.3f}"
    )
    h4_msg = (
        f"H4 (C1 CI_lo > 0): **{'SUPPORTED' if c1.ci_lo > 0 else 'REJECTED'}** "
        f"— CI_lo = {c1.ci_lo:+.3f}"
    )
    h5_msg = (
        f"H5 (rel-MC ≤ 0.80 vs 60/40): "
        f"**{'SUPPORTED' if c1.rel_mc_passes else 'REJECTED'}** "
        f"— DD reduction = {c1.rel_mc_median_dd_reduction:.3f}, p_better = {c1.rel_mc_p_strategy_better:.3f}"
    )
    h6_msg = (
        f"H6 (noise axis at least mid): "
        f"**{'SUPPORTED' if c1.noise_axis in ('mid', 'best') else 'REJECTED'}** "
        f"— C1 noise axis = {c1.noise_axis}"
    )
    for m in (h1_msg, h2_msg, h3_msg, h4_msg, h5_msg, h6_msg):
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## §4.5 Promotion verdict")
    lines.append("")
    lines.append(f"**{verdict_global}**")
    lines.append("")
    lines.append("## §4.6 Migration option comparison")
    lines.append("")
    lines.append(
        f"- C1 (20, 0.05) — plateau-centre: Sharpe {c1.sharpe:+.3f}, CI_lo {c1.ci_lo:+.3f}, verdict {c1.verdict}\n"
        f"- C2 (20, 0.10) — constrained best (preserves vol_target=0.10): Sharpe {c2.sharpe:+.3f}, CI_lo {c2.ci_lo:+.3f}, verdict {c2.verdict}\n"
        f"- C3 (40, 0.10) — J4 live baseline: Sharpe {c3.sharpe:+.3f}, CI_lo {c3.ci_lo:+.3f}, verdict {c3.verdict}\n"
        f"- C4 (20, 0.05 gross): Sharpe {c4.sharpe:+.3f}; cost drag vs C1 = {c4.sharpe - c1.sharpe:+.3f}"
    )
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
