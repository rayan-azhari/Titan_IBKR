"""D2b -- Commodity futures STRICT carry audit (signal_mode='m1_m2_basis')
on IBKR roll-stitched M1 + M2 data.

Specified by ``directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures
2026-05-15.md``. Sibling to ``run_d2_audit.py`` (rolling-yield proxy).

Run via::

    uv run python research/futures_carry/run_d2b_audit.py

Inputs:
    data/{ROOT}_M1_stitched_D.parquet   (back-adjusted continuous front)
    data/{ROOT}_M2_stitched_D.parquet   (back-adjusted continuous 2nd)

Audit window:
    Bounded by stitched data availability (typically 2023-05 -> 2026-05,
    ~3 years, per L41 IBKR paper-account depth).
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

from research.futures_carry.carry_strategy import (  # noqa: E402
    CarryConfig,
    carry_assert_causal,
    carry_returns,
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "d2b_strict_carry"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-reg D2b §2: shortened WFO due to ~3y stitched window (L41).
WFO_OVERRIDE = WfoConfig(
    is_min_years=1.5,
    oos_years=0.5,
    fold_count=5,
    is_mode="expanding",
    stride_overlap_allowed=False,
)

# All cells run with signal_mode="m1_m2_basis" (strict BGR carry).
_BASE_SIGNAL = dict(signal_mode="m1_m2_basis")

CELLS: dict[str, CarryConfig] = {
    "C1_canonical": CarryConfig(
        rebalance="monthly",
        breadth_pct=0.20,
        long_only=False,
        smooth_days=1,
        weighting="equal",
        apply_costs=True,
        **_BASE_SIGNAL,
    ),
    "C2_weekly": CarryConfig(
        rebalance="weekly",
        breadth_pct=0.20,
        long_only=False,
        smooth_days=1,
        weighting="equal",
        apply_costs=True,
        **_BASE_SIGNAL,
    ),
    "C3_tercile": CarryConfig(
        rebalance="monthly",
        breadth_pct=0.333,
        long_only=False,
        smooth_days=1,
        weighting="equal",
        apply_costs=True,
        **_BASE_SIGNAL,
    ),
    "C4_vol_weighted": CarryConfig(
        rebalance="monthly",
        breadth_pct=0.20,
        long_only=False,
        smooth_days=1,
        weighting="inverse_vol",
        apply_costs=True,
        **_BASE_SIGNAL,
    ),
    "C5_smoothed": CarryConfig(
        rebalance="monthly",
        breadth_pct=0.20,
        long_only=False,
        smooth_days=5,
        weighting="equal",
        apply_costs=True,
        **_BASE_SIGNAL,
    ),
    "C6_long_only": CarryConfig(
        rebalance="monthly",
        breadth_pct=0.20,
        long_only=True,
        smooth_days=1,
        weighting="equal",
        apply_costs=True,
        **_BASE_SIGNAL,
    ),
    "C7_gross_no_costs": CarryConfig(
        rebalance="monthly",
        breadth_pct=0.20,
        long_only=False,
        smooth_days=1,
        weighting="equal",
        apply_costs=False,
        **_BASE_SIGNAL,
    ),
}
CANONICAL_CELL = "C1_canonical"

PLATEAU_NEIGHBOURS: dict[str, CarryConfig] = {
    "P_rebalance_2week": replace(CELLS[CANONICAL_CELL], rebalance="weekly"),
    "P_portfolio_decile": replace(CELLS[CANONICAL_CELL], breadth_pct=0.10),
    "P_portfolio_third": replace(CELLS[CANONICAL_CELL], breadth_pct=0.333),
    "P_carry_3day_smooth": replace(CELLS[CANONICAL_CELL], smooth_days=3),
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


def _load_stitched(root: str, contract: str) -> pd.Series:
    """Load RAW M1 / M2 close series (NOT back-adjusted).

    For the strict-carry signal log(M1[t-1]/M2[t-1]) we need the actual
    forward-curve basis on date t, NOT the ratio of independently
    back-adjusted continuous series. Each back-adjusted series carries
    its own cumulative ratio that distorts the cross-contract basis.
    The _raw variant stores the unadjusted close of whichever contract
    is at the requested offset on each date — the correct input for
    cross-sectional carry ranking.
    """
    path = DATA_DIR / f"{root}_{contract}_raw_stitched_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).normalize()
    return s


def _load_adjusted_m1(root: str) -> pd.Series:
    """Back-adjusted M1 for bar-return computation (correct holding-period P&L)."""
    path = DATA_DIR / f"{root}_M1_stitched_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).normalize()
    return s


def load_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw M1 + raw M2 (for signal) and adjusted M1 (for returns).

    Returns (M1_raw, M2_raw, M1_adjusted).
    """
    m1_raw_cols, m2_raw_cols, m1_adj_cols = [], [], []
    for root in UNIVERSE_ROOTS:
        try:
            m1r = _load_stitched(root, "M1")
            m2r = _load_stitched(root, "M2")
            m1a = _load_adjusted_m1(root)
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue
        m1_raw_cols.append(m1r)
        m2_raw_cols.append(m2r)
        m1_adj_cols.append(m1a)
    if not m1_raw_cols:
        raise RuntimeError("No stitched parquets found -- run stitch_root() first")
    M1_raw = pd.concat(m1_raw_cols, axis=1).dropna(how="all").sort_index()
    M2_raw = pd.concat(m2_raw_cols, axis=1).dropna(how="all").sort_index()
    M1_adj = pd.concat(m1_adj_cols, axis=1).dropna(how="all").sort_index()
    return M1_raw, M2_raw, M1_adj


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


def _stitched_oos_sharpe(M1_adj, M2_adj, M1_raw, M2_raw, cfg, folds, bars_per_year):
    """Signal from raw M1/M2 closes (true basis); returns from back-adjusted
    M1 (correct holding-period P&L)."""
    rets = carry_returns(M1_adj, M2_adj, cfg=cfg, M1_signal_df=M1_raw, M2_signal_df=M2_raw)
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def _strategy_fn_for_cell(cfg: CarryConfig, M1_raw_v, M2_raw_v, M2_adj_v):
    """Closure for MC + noise gate.

    The MC primitive bootstraps the primary close + extras (back-adjusted
    series, since those reflect holding-period P&L). For the signal it
    re-uses the OUT-OF-MC raw M1 / M2 frames — the carry basis is computed
    from the raw forward curve, NOT from MC-perturbed back-adjusted series.
    """
    primary_root = M2_adj_v.columns[0]
    other_m1_roots = list(M2_adj_v.columns[1:])
    m2_roots = list(M2_adj_v.columns)

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        m1_df = pd.DataFrame(index=df.index)
        m1_df[primary_root] = df["close"]
        for r in other_m1_roots:
            if r in df.columns:
                m1_df[r] = df[r]
        m2_df = pd.DataFrame(index=df.index)
        for r in m2_roots:
            key = f"{r}_M2"
            if key in df.columns:
                m2_df[r] = df[key]
        if m2_df.empty:
            m2_df = M2_adj_v.reindex(df.index).ffill()
        return carry_returns(
            m1_df,
            m2_df,
            cfg=cfg,
            M1_signal_df=M1_raw_v.reindex(df.index).ffill(),
            M2_signal_df=M2_raw_v.reindex(df.index).ffill(),
        )

    return strategy_fn


def plateau_pre_flight(M1_adj, M2_adj, M1_raw, M2_raw, folds, bars_per_year):
    sharpes: dict[str, float] = {}
    sh, _ = _stitched_oos_sharpe(
        M1_adj, M2_adj, M1_raw, M2_raw, CELLS[CANONICAL_CELL], folds, bars_per_year
    )
    sharpes[CANONICAL_CELL] = sh
    for n, cfg in PLATEAU_NEIGHBOURS.items():
        sh, _ = _stitched_oos_sharpe(M1_adj, M2_adj, M1_raw, M2_raw, cfg, folds, bars_per_year)
        sharpes[n] = sh
    vals = list(sharpes.values())
    mx, mn = max(vals), min(vals)
    rel_spread = abs(mx - mn) / max(abs(mx), 1e-9)
    return rel_spread, sharpes


def run_cell(
    cell_name,
    cfg,
    M1_adj_v,
    M2_adj_v,
    M1_raw_v,
    M2_raw_v,
    M1_adj_full,
    M2_adj_full,
    M1_raw_full,
    M2_raw_full,
    *,
    n_trials_sweep,
    bars_per_year,
    sweep_sharpes,
    mc_n_workers,
):
    cls = StrategyClass.CARRY
    d = defaults_for(cls)
    folds = build_folds(M1_adj_v.index, WFO_OVERRIDE, bars_per_year=bars_per_year)
    if not folds:
        raise RuntimeError(f"{cell_name}: no folds")

    sh_value, stitched = _stitched_oos_sharpe(
        M1_adj_v, M2_adj_v, M1_raw_v, M2_raw_v, cfg, folds, bars_per_year
    )
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=bars_per_year, n_resamples=1000, seed=42
    )
    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh_value, sr_var_across_trials=sr_var, returns=stitched, n_trials=n_trials_sweep
    )

    primary = M1_adj_v.iloc[:, 0]
    extras: dict[str, pd.Series] = {col: M1_adj_v[col] for col in M1_adj_v.columns[1:]}
    for col in M2_adj_v.columns:
        extras[f"{col}_M2"] = M2_adj_v[col]
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, M1_raw_v, M2_raw_v, M2_adj_v),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    sanc_ret = carry_returns(
        M1_adj_full,
        M2_adj_full,
        cfg=cfg,
        M1_signal_df=M1_raw_full,
        M2_signal_df=M2_raw_full,
    ).iloc[len(M1_adj_v) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    def _noise_fn(df):
        return carry_returns(
            df,
            M2_adj_v.reindex(df.index).ffill(),
            cfg=cfg,
            M1_signal_df=M1_raw_v.reindex(df.index).ffill(),
            M2_signal_df=M2_raw_v.reindex(df.index).ffill(),
        )

    noise = run_noise_robustness(
        M1_adj_v,
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
        sharpe=round(sh_value, 4),
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
    print("D2b -- Strict M1/M2 Carry on IBKR Roll-Stitched Data")
    print("Pre-reg: directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md")
    print("=" * 72)

    M1_raw_full, M2_raw_full, M1_adj_full = load_universe()
    # Align M2_adj_full to the universe shape used elsewhere. The
    # back-adjusted M2 frame can be derived directly from the stitched
    # M2 parquets via the same loader pattern.
    m2_adj_cols = []
    for root in UNIVERSE_ROOTS:
        path = DATA_DIR / f"{root}_M2_stitched_D.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path)["close"].astype(float)
        s.name = root
        s.index = pd.to_datetime(s.index).normalize()
        m2_adj_cols.append(s)
    M2_adj_full = pd.concat(m2_adj_cols, axis=1).dropna(how="all").sort_index()

    print(
        f"\nUniverse (stitched): {len(M1_raw_full.columns)} commodities -- "
        f"{list(M1_raw_full.columns)}\n"
        f"M1_raw range: {M1_raw_full.index[0].date()} -> {M1_raw_full.index[-1].date()}  "
        f"({len(M1_raw_full)} bars)\n"
        f"M2_raw range: {M2_raw_full.index[0].date()} -> {M2_raw_full.index[-1].date()}  "
        f"({len(M2_raw_full)} bars)\n"
        f"Signal mode: m1_m2_basis (strict BGR carry, raw forward curve)"
    )

    carry_assert_causal(
        M1_adj_full,
        M2_adj_full,
        cfg=CarryConfig(signal_mode="m1_m2_basis"),
        n_trials=3,
        seed=42,
    )
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(M1_adj_full, months=6)  # smaller sanctuary on short window
    M1_adj_v = sanc.visible
    M1_adj_sanc = sanc.sanctuary
    M2_adj_v = M2_adj_full.reindex(M1_adj_v.index).ffill()
    M1_raw_v = M1_raw_full.reindex(M1_adj_v.index).ffill()
    M2_raw_v = M2_raw_full.reindex(M1_adj_v.index).ffill()
    print(f"Visible: {len(M1_adj_v)}  Sanctuary: {len(M1_adj_sanc)}")

    bars_per_year = BARS_PER_YEAR["D"]
    folds = build_folds(M1_adj_v.index, WFO_OVERRIDE, bars_per_year=bars_per_year)
    print(f"WFO (override 1.5y IS / 0.5y OOS): {len(folds)} folds")
    if len(folds) < 3:
        print(f"\n  [!] WARNING: only {len(folds)} folds. Audit statistical power limited.")

    print("\n[Plateau pre-flight] canonical + 4 neighbours...")
    rel_spread, plateau_sharpes = plateau_pre_flight(
        M1_adj_v, M2_adj_v, M1_raw_v, M2_raw_v, folds, bars_per_year
    )
    for n, s in plateau_sharpes.items():
        print(f"  {n}: Sharpe={s:+.4f}")
    print(f"  Relative spread: {rel_spread:.2%}  (V3.2 gate: < 30%)")
    if rel_spread > 0.30:
        print("\n  [!] PLATEAU GATE FAILED. Aborting per pre-reg §3 / L27.")
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# D2b Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"Relative Sharpe spread: {rel_spread:.2%} > 30%\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
        return None
    print("  [OK] plateau gate passed.")

    print(f"\n[Pass 1/2] Headline OOS Sharpes ({len(CELLS)} cells)...")
    sweep_sharpes: list[float] = []
    for name, cfg in CELLS.items():
        sh_v, _ = _stitched_oos_sharpe(
            M1_adj_v, M2_adj_v, M1_raw_v, M2_raw_v, cfg, folds, bars_per_year
        )
        sweep_sharpes.append(sh_v)
        print(f"  {name}: Sharpe={sh_v:+.4f}")

    print(f"\n[Pass 2/2] Full per-cell audit -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[CellResult] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            cfg,
            M1_adj_v,
            M2_adj_v,
            M1_raw_v,
            M2_raw_v,
            M1_adj_full,
            M2_adj_full,
            M1_raw_full,
            M2_raw_full,
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

    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# D2b Audit Result Log -- Strict Carry on IBKR Roll-Stitched Data\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(
            f"**Universe:** {len(M1_adj_full.columns)} commodities -- {list(M1_adj_full.columns)}\n"
        )
        fh.write(
            f"**Data range:** {M1_adj_full.index[0].date()} -> {M1_adj_full.index[-1].date()}\n"
        )
        fh.write(f"**Visible bars:** {len(M1_adj_v)}  Sanctuary: {len(M1_adj_sanc)}\n")
        fh.write(f"**WFO override (L41, 1.5y IS / 0.5y OOS):** {len(folds)} folds\n\n")

        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(f"**Relative spread:** {rel_spread:.2%} -- PASSED\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell verdicts (5-axis, J3 / L24)\n\n")
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
        c7 = next(r for r in results if r.cell == "C7_gross_no_costs")
        fh.write("\n## §4.3 Gross vs net economics (C1 vs C7)\n\n")
        fh.write(
            f"- Gross (C7): Sharpe={c7.sharpe:+.4f}, CI_lo={c7.ci_lo:+.3f}\n"
            f"- Net   (C1): Sharpe={c1.sharpe:+.4f}, CI_lo={c1.ci_lo:+.3f}\n"
            f"- Cost drag: {c7.sharpe - c1.sharpe:+.4f}\n"
        )

        # D2b H1 falsification: strict-carry vs rolling-yield proxy.
        fh.write("\n## §4.4 D2b H1 falsification (strict vs proxy)\n\n")
        fh.write(
            "D2 (rolling-yield proxy) was RETIRED on all 7 cells. If D2b strict-carry\n"
            "produces ANY DEPLOY-eligible cell, H1 is supported (the proxy's failure\n"
            "was not load-bearing on the strategy thesis). If D2b also all-negative,\n"
            "H1 is REJECTED (the carry signal itself is the problem).\n\n"
        )
        any_deploy = any(
            r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT") and r.ci_lo > 0 for r in results
        )
        if any_deploy:
            fh.write("**H1 SUPPORTED** — at least one strict-carry cell clears CI_lo>0 + tier.\n")
        else:
            fh.write(
                "**H1 REJECTED** — strict-carry also all-negative. The signal is the problem.\n"
            )

        deploy_eligible = [
            r
            for r in results
            if r.cell != "C7_gross_no_costs"
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
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}).\n"
            )
        else:
            fh.write(
                "No cell deployment-eligible. Document negative result + new lessons. "
                "Strict-carry on the 24-commodity BGR universe does not survive 5-axis "
                "matrix under IBKR roll-stitched data over ~3 years.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
