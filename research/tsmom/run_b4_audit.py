"""B4 -- TSMOM audit harness.

Specified by ``directives/Pre-Reg B4 TSMOM 2026-05-15.md``.

Run via::

    uv run python research/tsmom/run_b4_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.tsmom.tsmom_strategy import (  # noqa: E402
    TsmomConfig,
    tsmom_assert_causal,
    tsmom_returns,
)
from titan.research.framework import (  # noqa: E402
    DecisionInputs,
    NoiseConfig,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    run_block_mc,
    run_noise_robustness,
    sanctuary_divergence_test,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b4_tsmom"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered cells (V3.1).
CELLS: dict[str, TsmomConfig] = {
    "C1_canonical": TsmomConfig(signal_mode="sign", momentum_window_months=12, skip_months=1),
    "C2_raw": TsmomConfig(signal_mode="raw", momentum_window_months=12, skip_months=1),
    "C3_window_24": TsmomConfig(signal_mode="sign", momentum_window_months=24, skip_months=1),
    "C4_window_6": TsmomConfig(signal_mode="sign", momentum_window_months=6, skip_months=1),
    "C5_no_skip": TsmomConfig(signal_mode="sign", momentum_window_months=12, skip_months=0),
    "C6_equal_weight": TsmomConfig(signal_mode="sign", weighting="equal"),
    "C7_weekly_rebal": TsmomConfig(signal_mode="sign", rebalance="weekly"),
    "C8_gross_no_costs": TsmomConfig(signal_mode="sign", apply_costs=False),
}
CANONICAL_CELL = "C1_canonical"

# Plateau pre-flight neighbours of C1.
PLATEAU_NEIGHBOURS: dict[str, TsmomConfig] = {
    "P_window_9": replace(CELLS[CANONICAL_CELL], momentum_window_months=9),
    "P_window_15": replace(CELLS[CANONICAL_CELL], momentum_window_months=15),
    "P_skip_0": replace(CELLS[CANONICAL_CELL], skip_months=0),
    "P_skip_2": replace(CELLS[CANONICAL_CELL], skip_months=2),
}

# 24-commodity universe (M1, downloaded for D2).
UNIVERSE_ROOTS: tuple[str, ...] = (
    "CL",
    "NG",
    "HO",
    "RB",
    "BZ",  # energy (5)
    "GC",
    "SI",
    "PL",
    "PA",  # metals precious (4)
    "HG",  # metals industrial (1)
    "ZC",
    "ZW",
    "ZS",
    "ZL",
    "ZM",
    "ZO",  # grains (6)
    "LE",
    "GF",
    "HE",  # livestock (3)
    "KC",
    "CC",
    "SB",
    "CT",
    "OJ",  # softs (5)
)


def _load_close(root: str) -> pd.Series:
    path = DATA_DIR / f"{root}_M1_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def load_universe() -> pd.DataFrame:
    """Load all 24 commodity M1 closes into a single DataFrame."""
    parts = []
    for root in UNIVERSE_ROOTS:
        try:
            parts.append(_load_close(root))
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue
    if not parts:
        raise RuntimeError("No commodity parquets found")
    df = pd.concat(parts, axis=1).dropna(how="all").sort_index()
    return df


@dataclass
class CellResult:
    cell: str
    n_oos_bars: int
    n_folds: int
    sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    mc_p_maxdd_gt_threshold: float
    mc_threshold_pct: float
    sanctuary_sharpe: float
    sanctuary_percentile: float
    noise_base: float
    noise_passes_mean: bool
    noise_passes_worst: bool
    noise_axis: str
    verdict: str
    rationale: str


def _stitched_oos_sharpe(closes, cfg, folds, bars_per_year):
    rets = tsmom_returns(closes, cfg=cfg)
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def _strategy_fn_for_cell(cfg: TsmomConfig, closes_visible: pd.DataFrame):
    """Closure for MC / noise gate. Treat first commodity as primary."""
    primary_root = closes_visible.columns[0]
    other_roots = list(closes_visible.columns[1:])

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        u = pd.DataFrame(index=df.index)
        u[primary_root] = df["close"]
        for r in other_roots:
            if r in df.columns:
                u[r] = df[r]
        return tsmom_returns(u, cfg=cfg)

    return strategy_fn


def run_cell(
    cell_name: str,
    cfg: TsmomConfig,
    closes_visible: pd.DataFrame,
    closes_sanctuary: pd.DataFrame,
    *,
    n_trials_sweep: int,
    bars_per_year: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
) -> CellResult:
    cls = StrategyClass.CROSS_ASSET_MOMENTUM
    d = defaults_for(cls)
    folds = build_folds(closes_visible.index, d.wfo, bars_per_year=bars_per_year)
    if not folds:
        raise RuntimeError(f"{cell_name}: no folds")

    sh, stitched = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year)
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=bars_per_year, n_resamples=1000, seed=42
    )

    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh, sr_var_across_trials=sr_var, returns=stitched, n_trials=n_trials_sweep
    )

    primary = closes_visible.iloc[:, 0]
    extras: dict[str, pd.Series] = {r: closes_visible[r] for r in closes_visible.columns[1:]}
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, closes_visible),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    full = pd.concat([closes_visible, closes_sanctuary])
    sanc_ret = tsmom_returns(full, cfg=cfg).iloc[len(closes_visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        return tsmom_returns(df, cfg=cfg)

    noise = run_noise_robustness(
        closes_visible,
        _noise_fn,
        periods_per_year=bars_per_year,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30),
    )

    inputs = DecisionInputs(
        ci_lo=ci_lo,
        dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
        pass_threshold_prob=mc.pass_threshold_prob,
        sanctuary_sharpe=sanc_sh,
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
    )
    decision = decide(inputs)

    return CellResult(
        cell=cell_name,
        n_oos_bars=len(stitched),
        n_folds=len(folds),
        sharpe=round(sh, 4),
        ci_lo=round(ci_lo, 4),
        ci_hi=round(ci_hi, 4),
        dsr_prob=round(dsr.dsr_prob, 4),
        mc_p_maxdd_gt_threshold=round(mc.p_maxdd_gt_threshold, 4),
        mc_threshold_pct=round(mc.threshold_pct, 4),
        sanctuary_sharpe=round(sanc_sh, 4),
        sanctuary_percentile=round(div.percentile, 4)
        if np.isfinite(div.percentile)
        else float("nan"),
        noise_base=round(noise.base_sharpe, 4),
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
        noise_axis=decision.noise_axis,
        verdict=decision.verdict.value,
        rationale=decision.rationale,
    )


def main():
    print("=" * 72)
    print("B4 -- Time-Series Momentum Audit (MOP 2012)")
    print("Pre-reg: directives/Pre-Reg B4 TSMOM 2026-05-15.md")
    print("=" * 72)

    closes = load_universe()
    print(
        f"\nUniverse: {len(closes.columns)} commodities -- {list(closes.columns)}\n"
        f"Date range: {closes.index[0].date()} -> {closes.index[-1].date()}  "
        f"({len(closes)} bars)"
    )

    tsmom_assert_causal(closes, n_trials=3, seed=42)
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=12)
    closes_visible = sanc.visible
    closes_sanctuary = sanc.sanctuary
    print(f"Visible: {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}")

    bars_per_year = BARS_PER_YEAR["D"]
    cls_d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(closes_visible.index, cls_d.wfo, bars_per_year=bars_per_year)
    print(f"WFO: {len(folds)} folds")

    print("\n[Plateau pre-flight] canonical + 4 neighbours...")
    plateau_sharpes: dict[str, float] = {}
    sh_c1, _ = _stitched_oos_sharpe(closes_visible, CELLS[CANONICAL_CELL], folds, bars_per_year)
    plateau_sharpes[CANONICAL_CELL] = sh_c1
    for name, cfg in PLATEAU_NEIGHBOURS.items():
        sh, _ = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year)
        plateau_sharpes[name] = sh
    for n, s in plateau_sharpes.items():
        print(f"  {n}: Sharpe={s:+.4f}")
    vals = list(plateau_sharpes.values())
    mx, mn = max(vals), min(vals)
    rel_spread = abs(mx - mn) / max(abs(mx), 1e-9)
    print(f"  Relative spread: {rel_spread:.2%}  (V3.2 gate: < 30%)")
    if rel_spread > 0.30:
        print("\n  [!] PLATEAU GATE FAILED. Aborting per pre-reg §3 / L27.")
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# B4 Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"Relative Sharpe spread: {rel_spread:.2%} > 30%\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
        return None
    print("  [OK] plateau gate passed.")

    print(f"\n[Pass 1/2] Headline OOS Sharpes ({len(CELLS)} cells)...")
    sweep_sharpes: list[float] = []
    for name, cfg in CELLS.items():
        sh_v, _ = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year)
        sweep_sharpes.append(sh_v)
        print(f"  {name}: Sharpe={sh_v:+.4f}")

    print(f"\n[Pass 2/2] Full per-cell audit -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[CellResult] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            cfg,
            closes_visible,
            closes_sanctuary,
            n_trials_sweep=len(CELLS),
            bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes,
            mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(r)
        print(f"    Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        print(
            f"    DSR={r.dsr_prob:.4f}  "
            f"MC P(>{r.mc_threshold_pct * 100:.0f}%)={r.mc_p_maxdd_gt_threshold:.4f}"
        )
        print(f"    Sanc Sharpe={r.sanctuary_sharpe:+.4f}")
        print(
            f"    Noise: base={r.noise_base:+.4f} "
            f"mean_pass={r.noise_passes_mean} worst_pass={r.noise_passes_worst} "
            f"axis={r.noise_axis}"
        )
        print(f"    Verdict (5-axis): {r.verdict}")
        print(f"    Rationale: {r.rationale}")

    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# B4 Audit Result Log -- TSMOM (MOP 2012)\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(f"**Universe:** {len(closes.columns)} commodities -- {list(closes.columns)}\n")
        fh.write(f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()}\n")
        fh.write(f"**Visible bars:** {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}\n\n")

        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(f"**Relative spread:** {rel_spread:.2%} -- PASSED (gate 30%)\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell verdicts (5-axis, J3 / L24)\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P(>35%) | "
            "Sanc Sharpe | Noise base | Noise axis | Verdict |\n"
        )
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|\n")
        for r in results:
            fh.write(
                f"| {r.cell} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
                f"{r.dsr_prob:.4f} | {r.mc_p_maxdd_gt_threshold:.4f} | "
                f"{r.sanctuary_sharpe:+.4f} | {r.noise_base:+.4f} | "
                f"{r.noise_axis} | {r.verdict} |\n"
            )

        c1 = next(r for r in results if r.cell == "C1_canonical")
        c8 = next(r for r in results if r.cell == "C8_gross_no_costs")
        fh.write("\n## §4.3 Gross vs net (C1 vs C8)\n\n")
        fh.write(
            f"- Gross (C8, costs OFF): Sharpe={c8.sharpe:+.4f}, CI_lo={c8.ci_lo:+.3f}\n"
            f"- Net   (C1, costs ON):  Sharpe={c1.sharpe:+.4f}, CI_lo={c1.ci_lo:+.3f}\n"
            f"- Cost drag:             {c8.sharpe - c1.sharpe:+.4f}\n"
        )

        # L37 falsification verdict.
        fh.write("\n## §4.4 L37 falsification\n\n")
        if c1.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT") and c1.ci_lo > 0:
            fh.write(
                f"L37 CONFIRMED: C1 canonical produces Sharpe={c1.sharpe:+.4f} "
                f"with CI_lo={c1.ci_lo:+.3f} > 0 and verdict={c1.verdict}. "
                f"Time-series momentum has persisted on commodity futures, "
                f"unlike cross-sectional momentum on equities (A1 RETIRED).\n"
            )
        else:
            fh.write(
                f"L37 CHALLENGED: C1 canonical produces Sharpe={c1.sharpe:+.4f} "
                f"with CI_lo={c1.ci_lo:+.3f} and verdict={c1.verdict}. "
                f"Time-series momentum on retail-implementable yfinance commodity "
                f"data does NOT meet the deployment bar. A complementary L39 "
                f"may be needed to qualify L37's persistence claim.\n"
            )

        deploy_eligible = [
            r
            for r in results
            if r.cell != "C8_gross_no_costs"
            and (
                r.verdict == "DEPLOY"
                or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best")
            )
        ]
        fh.write("\n## §4.5 Recommended next step\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"Promote **{best.cell}** (CI_lo={best.ci_lo:+.3f}, "
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}). "
                f"Build live class in titan/strategies/tsmom/ + plumb monthly "
                f"rebalance on the 24-commodity CME futures basket.\n"
            )
        else:
            fh.write(
                "No cell deployment-eligible under the 5-axis matrix. "
                "Document negative result + L39 lesson.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
