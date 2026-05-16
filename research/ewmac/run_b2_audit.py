"""B2 -- Carver EWMAC trend ensemble audit on IBKR roll-stitched M1.

Specified by ``directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md``.
Sibling to ``run_b4b_audit.py`` (single-window TSMOM) and
``run_b4c_audit.py`` (3-window TSMOM ensemble; H1 PARTIALLY SUPPORTED).

Run via::

    uv run python research/ewmac/run_b2_audit.py
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

from research.ewmac.ewmac_strategy import (  # noqa: E402
    EwmacConfig,
    ewmac_assert_causal,
    ewmac_returns,
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
from titan.research.framework.typology import WfoConfig  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b2_ewmac_ensemble"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WFO_OVERRIDE = WfoConfig(
    is_min_years=1.5,
    oos_years=0.5,
    fold_count=5,
    is_mode="expanding",
    stride_overlap_allowed=False,
)

# Pre-registered cells (V3.1) per B2 pre-reg §2.
CELLS: dict[str, EwmacConfig] = {
    "C1_canonical": EwmacConfig(
        speeds=((16, 64), (32, 128), (64, 256)),
        fdm=1.35,
    ),
    "C2_short_speeds": EwmacConfig(
        speeds=((4, 16), (8, 32), (16, 64)),
        fdm=1.35,
    ),
    "C3_long_speeds": EwmacConfig(
        speeds=((32, 128), (64, 256), (128, 512)),
        fdm=1.35,
    ),
    "C4_full_six": EwmacConfig(
        speeds=((4, 16), (8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        fdm=1.51,
    ),
    "C5_two_speed": EwmacConfig(
        speeds=((16, 64), (64, 256)),
        fdm=1.20,
    ),
    "C6_singleton_canonical": EwmacConfig(
        speeds=((32, 128),),
        fdm=1.0,
    ),
    "C7_full_six_unclipped": EwmacConfig(
        speeds=((4, 16), (8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        fdm=1.51,
        forecast_cap=1e9,  # effectively no cap
    ),
    "C8_gross_no_costs": EwmacConfig(
        speeds=((16, 64), (32, 128), (64, 256)),
        fdm=1.35,
        apply_costs=False,
    ),
}
CANONICAL_CELL = "C1_canonical"

# Plateau pre-flight neighbours of C1 = (16/64, 32/128, 64/256) FDM=1.35.
PLATEAU_NEIGHBOURS: dict[str, EwmacConfig] = {
    "P_shift_short": replace(CELLS[CANONICAL_CELL], speeds=((8, 32), (16, 64), (32, 128))),
    "P_shift_long": replace(CELLS[CANONICAL_CELL], speeds=((32, 128), (64, 256), (128, 512))),
    "P_drop_fast": replace(CELLS[CANONICAL_CELL], speeds=((32, 128), (64, 256)), fdm=1.20),
    "P_drop_slow": replace(CELLS[CANONICAL_CELL], speeds=((16, 64), (32, 128)), fdm=1.20),
}

UNIVERSE_ROOTS: tuple[str, ...] = (
    "CL",
    "NG",
    "HO",
    "RB",
    "BZ",
    "GC",
    "SI",
    "PL",
    "PA",
    "HG",
    "ZC",
    "ZW",
    "ZS",
    "ZL",
    "ZM",
    "ZO",
    "LE",
    "GF",
    "HE",
    "KC",
    "CC",
    "SB",
    "CT",
    "OJ",
)


def _load_stitched_m1(root: str) -> pd.Series:
    path = DATA_DIR / f"{root}_M1_stitched_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).normalize()
    return s


def load_universe() -> pd.DataFrame:
    parts = []
    for root in UNIVERSE_ROOTS:
        try:
            parts.append(_load_stitched_m1(root))
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue
    if not parts:
        raise RuntimeError("No stitched M1 parquets found")
    return pd.concat(parts, axis=1).dropna(how="all").sort_index()


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
    rets = ewmac_returns(closes, cfg=cfg)
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def _strategy_fn_for_cell(cfg, closes_visible):
    primary_root = closes_visible.columns[0]
    other_roots = list(closes_visible.columns[1:])

    def strategy_fn(df):
        u = pd.DataFrame(index=df.index)
        u[primary_root] = df["close"]
        for r in other_roots:
            if r in df.columns:
                u[r] = df[r]
        return ewmac_returns(u, cfg=cfg)

    return strategy_fn


def run_cell(
    cell_name,
    cfg,
    closes_visible,
    closes_sanctuary,
    *,
    n_trials_sweep,
    bars_per_year,
    sweep_sharpes,
    mc_n_workers,
):
    cls = StrategyClass.CROSS_ASSET_MOMENTUM
    d = defaults_for(cls)
    folds = build_folds(closes_visible.index, WFO_OVERRIDE, bars_per_year=bars_per_year)
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
    sanc_ret = ewmac_returns(full, cfg=cfg).iloc[len(closes_visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    def _noise_fn(df):
        return ewmac_returns(df, cfg=cfg)

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
    print("B2 -- Carver EWMAC Trend Ensemble Audit")
    print("Pre-reg: directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md")
    print("=" * 72)

    closes = load_universe()
    print(
        f"\nUniverse: {len(closes.columns)} commodities  "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)"
    )

    ewmac_assert_causal(closes, cfg=CELLS[CANONICAL_CELL], n_trials=3, seed=42)
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=6)
    closes_visible = sanc.visible
    closes_sanctuary = sanc.sanctuary
    print(f"Visible: {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}")

    bars_per_year = BARS_PER_YEAR["D"]
    folds = build_folds(closes_visible.index, WFO_OVERRIDE, bars_per_year=bars_per_year)
    print(f"WFO: {len(folds)} folds")
    if len(folds) < 3:
        print(f"\n  [!] WARNING: only {len(folds)} folds. Audit power limited.")

    # --- Baseline reproduction sanity check (C6 = singleton 32/128) ------
    print("\n[Baseline reproduction] C6 singleton (32/128)...")
    sh_c6, _ = _stitched_oos_sharpe(
        closes_visible, CELLS["C6_singleton_canonical"], folds, bars_per_year
    )
    print(f"  C6_singleton_canonical: Sharpe={sh_c6:+.4f}  (reference single-speed EWMAC)")

    # --- Plateau pre-flight on C1 EWMAC ensemble -------------------------
    print("\n[Plateau pre-flight] C1 + 4 EWMAC-speed neighbours...")
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
    print("  Comparison: B4b=72.48%, B4c=47.93%")

    b4b_spread = 0.7248
    b4c_spread = 0.4793
    mit_vs_b4b = (b4b_spread - rel_spread) / b4b_spread * 100
    mit_vs_b4c = (b4c_spread - rel_spread) / b4c_spread * 100
    print(f"  Mitigation vs B4b: {mit_vs_b4b:+.1f}%")
    print(f"  Mitigation vs B4c: {mit_vs_b4c:+.1f}%")

    plateau_passed = rel_spread <= 0.30

    if not plateau_passed:
        print("\n  [!] PLATEAU GATE FAILED. Aborting per pre-reg §3 / L27.")
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# B2 Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"**Relative Sharpe spread:** {rel_spread:.2%} > 30%\n")
            fh.write("**B4b reference:** 72.48%   **B4c reference:** 47.93%\n")
            fh.write(f"**Mitigation vs B4b:** {mit_vs_b4b:+.1f}%\n")
            fh.write(f"**Mitigation vs B4c:** {mit_vs_b4c:+.1f}%\n\n")
            fh.write(f"**C6 singleton (32/128):** Sharpe={sh_c6:+.4f}\n\n")
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
            f"    DSR={r.dsr_prob:.4f}  MC P(>{r.mc_threshold_pct * 100:.0f}%)="
            f"{r.mc_p_maxdd_gt_threshold:.4f}"
        )
        print(f"    Sanc Sharpe={r.sanctuary_sharpe:+.4f}")
        print(
            f"    Noise: base={r.noise_base:+.4f} mean_pass={r.noise_passes_mean} "
            f"worst_pass={r.noise_passes_worst} axis={r.noise_axis}"
        )
        print(f"    Verdict (5-axis): {r.verdict}")

    # Result log.
    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# B2 Audit Result Log -- Carver EWMAC Trend Ensemble\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(f"**Universe:** {len(closes.columns)} commodities\n")
        fh.write(f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()}\n")
        fh.write(f"**Visible bars:** {len(closes_visible)} / Sanctuary: {len(closes_sanctuary)}\n")
        fh.write(f"**WFO override (L41):** {len(folds)} folds\n\n")

        fh.write("## §4.1 Baseline reproduction\n\n")
        fh.write(
            f"- C6_singleton_canonical (32/128 EWMAC): Sharpe={sh_c6:+.4f}\n"
            "  (Reference single-speed baseline; B4b singleton 12-month "
            "TSMOM was +1.6309, not directly comparable but same broad class.)\n"
        )

        fh.write("\n## §4.2 Plateau pre-flight + neighbour grid\n\n")
        fh.write(f"**Relative spread:** {rel_spread:.2%}  (B4b: 72.48%, B4c: 47.93%) -- PASSED\n\n")
        fh.write(f"**Mitigation vs B4b:** {mit_vs_b4b:+.1f}%\n")
        fh.write(f"**Mitigation vs B4c:** {mit_vs_b4c:+.1f}%\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.3 Per-cell 5-axis matrix\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P | "
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
        c4 = next(r for r in results if r.cell == "C4_full_six")
        c7 = next(r for r in results if r.cell == "C7_full_six_unclipped")

        fh.write("\n## §4.4 H1 verdict (plateau-spread completion)\n\n")
        fh.write(
            f"B4b: 72.48%. B4c: 47.93%. B2: {rel_spread:.2%}. "
            f"Mitigation vs B4b: {mit_vs_b4b:+.1f}%, vs B4c: {mit_vs_b4c:+.1f}%.\n"
        )
        fh.write("\n**H1 SUPPORTED** — L27 gate passed.\n")

        fh.write("\n## §4.5 H2 verdict (Sharpe vs B4b baseline)\n\n")
        b4b_canonical = 1.6309
        h2_threshold = 0.75 * b4b_canonical
        fh.write(
            f"B4b canonical: {b4b_canonical:+.4f}. H2 threshold (75%): "
            f"{h2_threshold:+.4f}.\n"
            f"B2 C1_canonical: {c1.sharpe:+.4f} "
            f"({c1.sharpe / b4b_canonical * 100:.0f}% of B4b).\n"
        )
        if c1.sharpe >= h2_threshold:
            fh.write("\n**H2 SUPPORTED.**\n")
        else:
            fh.write("\n**H2 REJECTED.**\n")

        fh.write("\n## §4.6 H3 verdict (forecast cap is load-bearing)\n\n")
        fh.write(
            f"C4 (6-speed, capped): Sharpe={c4.sharpe:+.4f}\n"
            f"C7 (6-speed, uncapped): Sharpe={c7.sharpe:+.4f}\n"
            f"Cap effect: {c4.sharpe - c7.sharpe:+.4f}\n"
        )
        if c4.sharpe > c7.sharpe + 0.1:
            fh.write(
                "\n**H3 SUPPORTED** — forecast cap meaningfully reduces noise on this universe.\n"
            )
        elif c4.sharpe > c7.sharpe:
            fh.write("\n**H3 WEAKLY SUPPORTED** — cap helps marginally.\n")
        else:
            fh.write("\n**H3 REJECTED** — cap is redundant or harmful on this universe.\n")

        deploy_eligible = [
            r
            for r in results
            if r.cell
            not in ("C6_singleton_canonical", "C7_full_six_unclipped", "C8_gross_no_costs")
            and (
                r.verdict == "DEPLOY"
                or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best")
            )
        ]
        fh.write("\n## §4.7 Recommended next step\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"Promote **{best.cell}** (CI_lo={best.ci_lo:+.3f}, "
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}). "
                f"Port to titan/strategies/ewmac/ with speeds {best.cell}.\n"
            )
        else:
            fh.write(
                "Plateau passed but no cell deployment-eligible under 5-axis. "
                "Document; consider regime-overlay or longer history.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
