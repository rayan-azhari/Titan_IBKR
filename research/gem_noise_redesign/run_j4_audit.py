"""J4 -- GEM noise-robust redesign audit harness.

Specified by ``directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md``.

Re-uses the existing GEM audit primitives (cell looping, MC, noise gate,
5-axis decision) but with a different cell grid: C0_baseline (C12 reproduction)
+ 6 mitigation cells (A1, A2, B1, B2, C1, C2).

Run via::

    uv run python research/gem_noise_redesign/run_j4_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.gem.gem_strategy import (  # noqa: E402
    GEM_UNIVERSE,
    GemConfig,
    gem_assert_causal,
    gem_returns,
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
    run_relative_block_mc,
    sanctuary_divergence_test,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "gem_j4"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cost calibration — identical to GEM C12 production audit (L23).
COST_BPS_PER_TURNOVER = 6.0
COST_FIXED_USD_PER_FILL = 1.0
COST_BPS_PER_TURNOVER_MES = 1.0
COST_FIXED_USD_PER_FILL_MES = 1.19
COST_NOTIONAL_USD = 30_000.0
COST_REBALANCE_THRESHOLD = 0.05


def _cost_kwargs(cfg: GemConfig) -> dict:
    max_lev = getattr(cfg, "max_leverage", 1.0) or 1.0
    use_mes = max_lev > 1.0
    return dict(
        cost_bps_per_turnover=COST_BPS_PER_TURNOVER,
        cost_fixed_usd_per_fill=COST_FIXED_USD_PER_FILL,
        cost_bps_per_turnover_mes=COST_BPS_PER_TURNOVER_MES if use_mes else 0.0,
        cost_fixed_usd_per_fill_mes=COST_FIXED_USD_PER_FILL_MES if use_mes else 0.0,
        notional_usd=COST_NOTIONAL_USD,
        execution_mode="mes" if use_mes else "etf",
        rebalance_threshold=COST_REBALANCE_THRESHOLD,
    )


# ─── J4 cells (V3.1 frozen at directive commit) ────────────────────────────
# All cells share C12's structure: lookback_blend=(3,6,12), buffer_pct=0.005,
# defensive_switch=True, ann_vol_target=0.10 (except mitigation C which
# overrides via vol_target_kind), max_leverage=2.0. Each cell differs by
# exactly one mitigation knob from C0_baseline.

_BASE = dict(
    lookback_blend=(3, 6, 12),
    buffer_pct=0.005,
    defensive_switch=True,
    ann_vol_target=0.10,
    vol_lookback_days=20,
    max_leverage=2.0,
)

CELLS: dict[str, GemConfig] = {
    "C0_baseline": GemConfig(**_BASE),
    # Mitigation A: smoother realised-vol estimator.
    "A1_ewma_hl40": GemConfig(**_BASE, vol_estimator_kind="ewma", vol_estimator_halflife=40),
    "A2_window_60": GemConfig(**{**_BASE, "vol_lookback_days": 60}),
    # Mitigation B: per-bar position-change cap.
    "B1_cap_5pct": GemConfig(**_BASE, max_weight_delta_per_bar=0.05),
    "B2_cap_10pct": GemConfig(**_BASE, max_weight_delta_per_bar=0.10),
    # Mitigation C: rolling-percentile vol target (overrides ann_vol_target).
    "C1_qtile_q40": GemConfig(
        **{k: v for k, v in _BASE.items() if k != "ann_vol_target"},
        ann_vol_target=0.10,  # ignored when vol_target_kind != "fixed"
        vol_target_kind="rolling_quantile",
        vol_target_quantile=0.40,
        vol_target_quantile_window=252,
    ),
    "C2_qtile_q50": GemConfig(
        **{k: v for k, v in _BASE.items() if k != "ann_vol_target"},
        ann_vol_target=0.10,  # ignored when vol_target_kind != "fixed"
        vol_target_kind="rolling_quantile",
        vol_target_quantile=0.50,
        vol_target_quantile_window=252,
    ),
}

BASELINE_CELL = "C0_baseline"


def _load_close(sym: str) -> pd.Series:
    path = DATA_DIR / f"{sym}_D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    df = pd.read_parquet(path)
    s = df["close"] if "close" in df.columns else df["Close"]
    s.name = sym
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def load_closes() -> pd.DataFrame:
    parts = [_load_close(sym) for sym in GEM_UNIVERSE]
    df = pd.concat(parts, axis=1).dropna(how="any")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _strategy_fn_for_cell(cfg: GemConfig):
    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        u = pd.DataFrame(index=df.index)
        u["SPY"] = df["close"]
        for sym in ("EFA", "IEF"):
            if sym in df.columns:
                u[sym] = df[sym]
        if not all(s in u.columns for s in GEM_UNIVERSE):
            return pd.Series(0.0, index=df.index)
        return gem_returns(u, cfg=cfg, **_cost_kwargs(cfg))

    return strategy_fn


def _stitched_oos_sharpe(
    visible: pd.DataFrame, cfg: GemConfig, folds, bars_per_year: int
) -> tuple[float, pd.Series]:
    rets = gem_returns(visible, cfg=cfg, ief_for_credit=visible["IEF"], **_cost_kwargs(cfg))
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def run_cell(
    cell_name: str,
    cfg: GemConfig,
    visible: pd.DataFrame,
    sanctuary: pd.DataFrame,
    *,
    n_trials_sweep: int,
    bars_per_year: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
):
    cls = StrategyClass.CROSS_ASSET_MOMENTUM
    d = defaults_for(cls)
    folds = build_folds(visible.index, d.wfo, bars_per_year=bars_per_year)

    sh_value, stitched = _stitched_oos_sharpe(visible, cfg, folds, bars_per_year)
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=bars_per_year, n_resamples=1000, seed=42
    )

    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh_value, sr_var_across_trials=sr_var, returns=stitched, n_trials=n_trials_sweep
    )

    primary = visible["SPY"]
    extras = {"EFA": visible["EFA"], "IEF": visible["IEF"]}
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    def benchmark_fn(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change().fillna(0.0)

    rel_mc = run_relative_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg),
        benchmark_fn=benchmark_fn,
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        median_ratio_gate=0.80,
        p_strategy_better_gate=0.50,
        n_workers=mc_n_workers,
    )

    full = pd.concat([visible, sanctuary])
    full_rets = gem_returns(full, cfg=cfg, ief_for_credit=full["IEF"], **_cost_kwargs(cfg))
    sanc_ret = full_rets.iloc[len(visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    # Per-cell noise gate (J3 / L24).
    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        return gem_returns(df, cfg=cfg, ief_for_credit=df.get("IEF"), **_cost_kwargs(cfg))

    noise = run_noise_robustness(
        visible,
        _noise_fn,
        periods_per_year=bars_per_year,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30),
    )

    inputs = DecisionInputs(
        ci_lo=ci_lo,
        dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=rel_mc.median_dd_reduction,
        pass_threshold_prob=rel_mc.median_ratio_gate,
        sanctuary_sharpe=sanc_sh,
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
    )
    decision = decide(inputs)

    return dict(
        cell=cell_name,
        n_oos_bars=len(stitched),
        n_folds=len(folds),
        sharpe=round(sh_value, 4),
        ci_lo=round(ci_lo, 4),
        ci_hi=round(ci_hi, 4),
        dsr_prob=round(dsr.dsr_prob, 4),
        mc_p_maxdd_gt_threshold=round(mc.p_maxdd_gt_threshold, 4),
        mc_threshold_pct=round(mc.threshold_pct, 4),
        rel_mc_median_ratio=round(rel_mc.median_dd_reduction, 4),
        rel_mc_p_strategy_better=round(rel_mc.p_strategy_better, 4),
        rel_mc_passes=rel_mc.passes,
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
    print("=" * 80)
    print("GEM J4 -- Noise-Robust Redesign Audit")
    print("Pre-reg: directives/Pre-Reg J4 GEM Noise-Robust Redesign 2026-05-15.md")
    print("Testing mitigations A (EWMA vol) / B (dw cap) / C (qtile target) vs C0 (=C12) baseline")
    print("=" * 80)

    closes = load_closes()
    print(f"\nData: {closes.shape[0]} bars {closes.index[0].date()} -> {closes.index[-1].date()}")

    gem_assert_causal(closes, n_trials=3, seed=42)
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=12)
    visible = sanc.visible
    sanctuary = sanc.sanctuary
    print(f"Visible: {len(visible)} bars  Sanctuary: {len(sanctuary)} bars")

    bars_per_year = BARS_PER_YEAR["D"]
    cls_defaults = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(visible.index, cls_defaults.wfo, bars_per_year=bars_per_year)
    print(f"WFO: {len(folds)} folds")

    # Pass 1: headline sharpes (drives DSR variance and plateau diagnostic).
    print(f"\n[Pass 1/2] Headline OOS Sharpes for {len(CELLS)} cells...")
    sweep_sharpes: list[float] = []
    cell_sharpes: dict[str, float] = {}
    for name, cfg in CELLS.items():
        sh_value, _ = _stitched_oos_sharpe(visible, cfg, folds, bars_per_year)
        sweep_sharpes.append(sh_value)
        cell_sharpes[name] = sh_value
        print(f"  {name}: Sharpe={sh_value:+.4f}")

    max_sh = max(sweep_sharpes)
    min_sh = min(sweep_sharpes)
    plateau_spread = abs(max_sh - min_sh) / max(abs(max_sh), 1e-9)
    print(f"\nPlateau spread (informational, not gate): {plateau_spread:.2%}")

    print(
        f"\n[Pass 2/2] Full per-cell audit (DSR + MC + sanctuary + noise + decide) "
        f"-- MC parallel x{DEFAULT_MC_WORKERS}..."
    )
    results: list[dict] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            cfg,
            visible,
            sanctuary,
            n_trials_sweep=len(CELLS),
            bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes,
            mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(r)
        print(f"    Sharpe={r['sharpe']:+.4f}  CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        print(
            f"    DSR={r['dsr_prob']:.4f}  "
            f"Rel-MC ratio={r['rel_mc_median_ratio']:.4f} (pass={r['rel_mc_passes']})"
        )
        print(f"    Sanc Sharpe={r['sanctuary_sharpe']:+.4f}")
        print(
            f"    Noise: base={r['noise_base']:+.4f} "
            f"mean_pass={r['noise_passes_mean']} worst_pass={r['noise_passes_worst']} "
            f"axis={r['noise_axis']}"
        )
        print(f"    Verdict (5-axis): {r['verdict']}")
        print(f"    Rationale: {r['rationale']}")

    # Result log.
    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# GEM J4 Audit Result Log -- Noise-Robust Redesign\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()}\n")
        fh.write(f"**Visible bars:** {len(visible)}  Sanctuary: {len(sanctuary)}\n")
        fh.write(f"**Plateau spread:** {plateau_spread:.2%}\n\n")

        fh.write("## §4.1 Sanity check (C0_baseline vs J3's C12)\n\n")
        baseline = next(r for r in results if r["cell"] == BASELINE_CELL)
        fh.write(f"C0_baseline Sharpe: {baseline['sharpe']:+.4f}  (J3 C12 reported: +0.8016)\n")
        delta = abs(baseline["sharpe"] - 0.8016)
        fh.write(
            f"|delta|: {delta:.4f}  (tolerance: 0.05)  "
            f"{'PASS' if delta <= 0.05 else 'FAIL -- harness divergence; J4 results invalid'}\n\n"
        )
        fh.write(
            f"C0_baseline noise axis: {baseline['noise_axis']}  "
            f"(J3 C12 reported: mid -- expected match)\n\n"
        )

        fh.write("## §4.2 Per-cell verdicts (5-axis, J3 / L24)\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | Rel MC ratio | Rel MC pass | "
            "Sanc Sharpe | Noise base | Noise axis | Verdict |\n"
        )
        fh.write("|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---|\n")
        for r in results:
            fh.write(
                f"| {r['cell']} | {r['sharpe']:+.4f} | {r['ci_lo']:+.3f} | "
                f"{r['ci_hi']:+.3f} | {r['dsr_prob']:.4f} | "
                f"{r['rel_mc_median_ratio']:.4f} | "
                f"{'PASS' if r['rel_mc_passes'] else 'FAIL'} | "
                f"{r['sanctuary_sharpe']:+.4f} | {r['noise_base']:+.4f} | "
                f"{r['noise_axis']} | {r['verdict']} |\n"
            )

        fh.write("\n## §4.3 Mitigation attribution\n\n")
        winners = [r for r in results if r["cell"] != BASELINE_CELL and r["noise_axis"] == "best"]
        if not winners:
            fh.write(
                "**No mitigation recovered the noise axis to `best`.** Every cell remains "
                "at noise=mid or worse. The vol-target overlay is intrinsically noise-fragile "
                "at this asset/frequency. Recommended: permanently accept C12's CONDITIONAL_WATCHPOINT "
                "verdict, OR remove the vol-target overlay (regression to non-levered C6_blend "
                "performance).\n"
            )
        else:
            best = max(winners, key=lambda r: r["ci_lo"])
            fh.write(f"**Winning cell:** {best['cell']}\n")
            fh.write(
                f"- Sharpe: {best['sharpe']:+.4f}  CI_lo: {best['ci_lo']:+.3f}\n"
                f"- Noise axis: {best['noise_axis']} (recovered from C0's mid)\n"
                f"- Verdict: {best['verdict']}\n"
            )

        fh.write("\n## §4.4 Recommended production change\n\n")
        # J3 pre-reg §3 selection rule: DEPLOY-eligible iff
        #   (a) verdict == DEPLOY, OR
        #   (b) verdict == CONDITIONAL_WATCHPOINT AND failing axis is NOT the noise axis.
        # A CONDITIONAL_WATCHPOINT cell has exactly one non-`best` axis.
        # If that axis is noise (i.e. noise_axis in {"mid", "worst"}), the
        # cell is blocked from production promotion. Hence "the failing
        # axis is NOT the noise axis" reduces to "noise_axis == 'best'"
        # for any COND_WP cell.
        deploy_eligible = [
            r
            for r in results
            if r["cell"] != BASELINE_CELL
            and (
                r["verdict"] == "DEPLOY"
                or (r["verdict"] == "CONDITIONAL_WATCHPOINT" and r["noise_axis"] == "best")
            )
        ]
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r["ci_lo"])
            fh.write(
                f"Promote `{best['cell']}` to production. Update "
                f"`config/gem_voltarget_lev2.toml` with the cell's parameters (see CELLS dict).\n"
            )
        else:
            fh.write(
                "No cell qualifies for promotion. C12 stays live at "
                "CONDITIONAL_WATCHPOINT pending alternative redesign (e.g. combining "
                "mitigations, or removing the overlay).\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
