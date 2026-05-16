"""D2 -- Commodity futures carry audit harness.

Specified by ``directives/Pre-Reg D2 Commodity Futures Carry 2026-05-15.md``.

Run via::

    uv run python research/futures_carry/run_d2_audit.py
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "d2_futures_carry"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# WFO override per pre-reg §2 (L25): 3y IS instead of class-default 5y.
# Visible window ~7.5y -> ~4 folds at 3y IS / 1y OOS.
WFO_OVERRIDE = WfoConfig(
    is_min_years=3.0,
    oos_years=1.0,
    fold_count=5,
    is_mode="expanding",
    stride_overlap_allowed=False,
)

# Pre-registered cells (V3.1). All cells run with signal_mode="rolling_yield"
# (BGR §3.2 proxy) due to Databento M2 data unavailability -- see Pre-Reg
# D2 §2 amendment 2026-05-15 PM.
_BASE_SIGNAL = dict(signal_mode="rolling_yield", yield_lookback=252)

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

# Plateau pre-flight neighbours of C1 (pre-reg §3).
PLATEAU_NEIGHBOURS: dict[str, CarryConfig] = {
    "P_rebalance_2week": replace(CELLS[CANONICAL_CELL], rebalance="weekly"),
    "P_portfolio_decile": replace(CELLS[CANONICAL_CELL], breadth_pct=0.10),
    "P_portfolio_third": replace(CELLS[CANONICAL_CELL], breadth_pct=0.333),
    "P_carry_3day_smooth": replace(CELLS[CANONICAL_CELL], smooth_days=3),
}


# Pre-reg §2 + 2026-05-15 PM amendment: yfinance M1 data covers the full
# 24-commodity BGR universe (2000+ history). Rolling-yield signal mode is
# active because M2 is not available from yfinance.
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
)  # total = 24


def _load_close(root: str, contract: str) -> pd.Series:
    """Load close-price series for a continuous contract leg."""
    path = DATA_DIR / f"{root}_{contract}_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def load_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (M1_df, M2_df) DataFrames spanning the carry-eligible universe.

    For the M1-only-data path (post-2026-05-15 amendment), M2 is loaded
    opportunistically when present, otherwise empty. The strategy uses
    cfg.signal_mode='rolling_yield' which doesn't require M2.
    """
    m1_cols = []
    m2_cols = []
    for root in UNIVERSE_ROOTS:
        try:
            m1_cols.append(_load_close(root, "M1"))
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue
        try:
            m2_cols.append(_load_close(root, "M2"))
        except FileNotFoundError:
            pass  # M2-absent: rolling_yield mode handles this
    M1 = pd.concat(m1_cols, axis=1).dropna(how="all") if m1_cols else pd.DataFrame()
    M2 = pd.concat(m2_cols, axis=1).dropna(how="all") if m2_cols else pd.DataFrame()
    return M1, M2


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


def _strategy_fn_for_cell(cfg: CarryConfig, M1_visible: pd.DataFrame, M2_visible: pd.DataFrame):
    """Build a strategy_fn closure for MC + noise gate.

    The framework's MC primitive bootstraps ``df["close"]`` (the primary
    series) + ``extra_series`` columns. For D2's 18-commodity basket, we
    treat the FIRST commodity (CL) as primary and pass the other 17
    commodities AND their M2 counterparts via extras. The strategy_fn
    re-assembles the M1 / M2 DataFrames from the input columns.
    """
    primary_root = M1_visible.columns[0]  # CL by construction
    other_m1_roots = list(M1_visible.columns[1:])
    m2_roots = list(M2_visible.columns)

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        # Re-assemble M1 frame from input.
        m1_df = pd.DataFrame(index=df.index)
        m1_df[primary_root] = df["close"]
        for r in other_m1_roots:
            if r in df.columns:
                m1_df[r] = df[r]
        # Re-assemble M2 frame from M2-suffixed columns OR fall back to the
        # original visible M2 (in the noise-gate path, full OHLC is preserved).
        m2_df = pd.DataFrame(index=df.index)
        for r in m2_roots:
            key = f"{r}_M2"
            if key in df.columns:
                m2_df[r] = df[key]
            elif r in df.columns and r not in m1_df.columns:
                m2_df[r] = df[r]
        if m2_df.empty:
            # Noise-gate path: the input df contains the original M1
            # frame structure; pair with the visible M2 for signal.
            m2_df = M2_visible.reindex(df.index).ffill()
        return carry_returns(m1_df, m2_df, cfg=cfg)

    return strategy_fn


def _stitched_oos_sharpe(
    M1: pd.DataFrame, M2: pd.DataFrame, cfg: CarryConfig, folds, bars_per_year
):
    rets = carry_returns(M1, M2, cfg=cfg)
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def plateau_pre_flight(
    M1_visible: pd.DataFrame,
    M2_visible: pd.DataFrame,
    folds,
    bars_per_year: int,
) -> tuple[float, dict[str, float]]:
    sharpes: dict[str, float] = {}
    sh_c1, _ = _stitched_oos_sharpe(
        M1_visible, M2_visible, CELLS[CANONICAL_CELL], folds, bars_per_year
    )
    sharpes[CANONICAL_CELL] = sh_c1
    for name, cfg in PLATEAU_NEIGHBOURS.items():
        sh, _ = _stitched_oos_sharpe(M1_visible, M2_visible, cfg, folds, bars_per_year)
        sharpes[name] = sh
    vals = list(sharpes.values())
    mx, mn = max(vals), min(vals)
    rel_spread = abs(mx - mn) / max(abs(mx), 1e-9)
    return rel_spread, sharpes


def run_cell(
    cell_name: str,
    cfg: CarryConfig,
    M1_visible: pd.DataFrame,
    M2_visible: pd.DataFrame,
    M1_full: pd.DataFrame,
    M2_full: pd.DataFrame,
    *,
    n_trials_sweep: int,
    bars_per_year: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
) -> CellResult:
    cls = StrategyClass.CARRY
    d = defaults_for(cls)
    folds = build_folds(M1_visible.index, WFO_OVERRIDE, bars_per_year=bars_per_year)
    if not folds:
        raise RuntimeError(f"{cell_name}: no folds")

    sh_value, stitched = _stitched_oos_sharpe(M1_visible, M2_visible, cfg, folds, bars_per_year)
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=bars_per_year, n_resamples=1000, seed=42
    )

    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh_value, sr_var_across_trials=sr_var, returns=stitched, n_trials=n_trials_sweep
    )

    # MC: bootstrap M1 series with shared blocks across columns. CL is the
    # primary; the other 17 M1 columns + 18 M2 columns ride as extras.
    primary = M1_visible.iloc[:, 0]  # CL
    extras: dict[str, pd.Series] = {}
    for col in M1_visible.columns[1:]:
        extras[col] = M1_visible[col]
    for col in M2_visible.columns:
        extras[f"{col}_M2"] = M2_visible[col]
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, M1_visible, M2_visible),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    # Sanctuary.
    sanc_ret = carry_returns(M1_full, M2_full, cfg=cfg).iloc[len(M1_visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    # Noise gate.
    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        # df has all M1 cols (incl. CL as "close" if primary_close was
        # named CL; here we'll just feed the full visible M1 since the
        # noise primitive replaces every column with perturbed versions).
        return carry_returns(df, M2_visible.reindex(df.index).ffill(), cfg=cfg)

    noise = run_noise_robustness(
        M1_visible,
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
    print("D2 -- Commodity Futures Carry Audit (BGR-2019, 18-commodity universe)")
    print("Pre-reg: directives/Pre-Reg D2 Commodity Futures Carry 2026-05-15.md")
    print("=" * 72)

    M1_full, M2_full = load_universe()
    # Rolling-yield mode does NOT need M2. Restrict to M1 cols only.
    # (Pre-reg amendment 2026-05-15 PM.)
    M1_full = M1_full.copy()
    M1_full.index = pd.to_datetime(M1_full.index).tz_localize(None).normalize()
    M1_full = M1_full.dropna(how="all").sort_index()
    if not M2_full.empty:
        # Align M2 to M1's universe to keep the audit's I/O symmetric.
        M2_full = M2_full.reindex(columns=M1_full.columns).reindex(M1_full.index)
    else:
        M2_full = pd.DataFrame(index=M1_full.index, columns=M1_full.columns, dtype=float)
    print(
        f"\nUniverse: {len(M1_full.columns)} commodities -- {list(M1_full.columns)}\n"
        f"Date range: {M1_full.index[0].date()} -> {M1_full.index[-1].date()}  "
        f"({len(M1_full)} bars)\n"
        f"Signal mode: rolling_yield (BGR §3.2 proxy)"
    )

    carry_assert_causal(M1_full, M2_full, n_trials=3, seed=42)
    print("Causality smoke test: PASSED")

    # Slice sanctuary off the FULL M1; mirror onto M2.
    sanc = slice_sanctuary(M1_full, months=12)
    M1_visible = sanc.visible
    M1_sanctuary = sanc.sanctuary
    M2_visible = M2_full.reindex(M1_visible.index).ffill()
    print(f"Visible: {len(M1_visible)}  Sanctuary: {len(M1_sanctuary)}")

    bars_per_year = BARS_PER_YEAR["D"]
    folds = build_folds(M1_visible.index, WFO_OVERRIDE, bars_per_year=bars_per_year)
    print(f"WFO (override: 3y IS): {len(folds)} folds")

    print("\n[Plateau pre-flight] canonical + 4 neighbours...")
    rel_spread, plateau_sharpes = plateau_pre_flight(M1_visible, M2_visible, folds, bars_per_year)
    for n, s in plateau_sharpes.items():
        print(f"  {n}: Sharpe={s:+.4f}")
    print(f"  Relative spread: {rel_spread:.2%}  (V3.2 gate: < 30%)")
    if rel_spread > 0.30:
        print(
            f"\n  [!] PLATEAU GATE FAILED: spread={rel_spread:.2%} > 30%. "
            f"Aborting per pre-reg §3 / L27."
        )
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# D2 Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"Relative Sharpe spread: {rel_spread:.2%} > 30%\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
        return None
    print("  [OK] plateau gate passed.")

    print(f"\n[Pass 1/2] Headline OOS Sharpes ({len(CELLS)} cells)...")
    sweep_sharpes: list[float] = []
    for name, cfg in CELLS.items():
        sh_value, _ = _stitched_oos_sharpe(M1_visible, M2_visible, cfg, folds, bars_per_year)
        sweep_sharpes.append(sh_value)
        print(f"  {name}: Sharpe={sh_value:+.4f}")

    print(f"\n[Pass 2/2] Full per-cell audit -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[CellResult] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            cfg,
            M1_visible,
            M2_visible,
            M1_full,
            M2_full,
            n_trials_sweep=len(CELLS),
            bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes,
            mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(r)
        print(f"    Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        print(
            f"    DSR={r.dsr_prob:.4f}  MC P(>{r.mc_threshold_pct * 100:.0f}%)={r.mc_p_maxdd_gt_threshold:.4f}"
        )
        print(f"    Sanc Sharpe={r.sanctuary_sharpe:+.4f}")
        print(
            f"    Noise: base={r.noise_base:+.4f} "
            f"mean_pass={r.noise_passes_mean} worst_pass={r.noise_passes_worst} axis={r.noise_axis}"
        )
        print(f"    Verdict (5-axis): {r.verdict}")
        print(f"    Rationale: {r.rationale}")

    # Result log.
    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# D2 Audit Result Log -- Commodity Futures Carry\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(f"**Universe:** {len(M1_full.columns)} commodities -- {list(M1_full.columns)}\n")
        fh.write(f"**Data range:** {M1_full.index[0].date()} -> {M1_full.index[-1].date()}\n")
        fh.write(f"**Visible bars:** {len(M1_visible)}  Sanctuary: {len(M1_sanctuary)}\n")
        fh.write(f"**WFO override (L25, 3y IS):** {len(folds)} folds\n\n")

        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(f"**Relative spread:** {rel_spread:.2%} -- PASSED (gate 30%)\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell verdicts (5-axis, J3 / L24)\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P(>30%) | "
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
            f"- Gross (C7, costs OFF): Sharpe = {c7.sharpe:+.4f}, CI_lo = {c7.ci_lo:+.3f}\n"
            f"- Net   (C1, costs ON):  Sharpe = {c1.sharpe:+.4f}, CI_lo = {c1.ci_lo:+.3f}\n"
            f"- Sharpe cost drag: {c7.sharpe - c1.sharpe:+.4f}\n"
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
        fh.write("\n## §4.4 Recommended next step\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"Promote **{best.cell}** (CI_lo={best.ci_lo:+.3f}, "
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}).\n"
            )
        else:
            fh.write(
                "No cell deployment-eligible under the 5-axis matrix. "
                "Document the negative result; see §4.5 for new lessons.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
