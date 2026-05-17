"""B2e -- Carver EWMAC audit on the 11-symbol IBKR cross-asset universe.

Universe (4 asset classes):
    equity_index : ES, NQ                       (data/<S>_D.parquet)
    commodity    : CL, BZ, HG, SI, GC           (Databento M1_D)
    bond         : ZN, ZB                       (Databento M1_D)
    fx           : 6E, 6J                       (Databento M1_D)

Common window: 2017-06-01 -> 2026-05-15 (~9y).
Sanctuary held out: last 12 months.

NOTE -- L52 override.
    The V3.6 hybrid IS-only sweep (research/exploration/sweep_b2e_ibkr_xasset.py
    + .tmp/reports/sweep_b2e_ibkr_xasset/) found NO plateau on this universe
    (single-peaked at fast_hl=8 with spread 45% > L52's 30% gate). Strict L52
    discipline would have stopped at the sweep with no pre-reg + no audit.
    This script runs the audit ANYWAY at the operator's explicit request,
    using the SAME C1-C8 cells as B2 + B2b for apples-to-apples cross-audit.
    The result here is informational; the L52 stop documented in
    'directives/B2e Sweep Report 2026-05-17.md' remains the authoritative
    closure of the B2 family.

Predecessors:
    - B2 (IBKR commodities 3y): Sharpe +2.02 but CI_lo<0 -> tier=unconfirmed
    - B2b (yfinance broad 21y): Sharpe -0.28, plateau 37% -> RETIRED (L48)
    - B2e sweep (this universe IS-only): no plateau, peak +0.50 single-speed

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ewmac/run_b2e_audit.py
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
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b2e_ewmac_ibkr_xasset"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Universe (path per symbol).
UNIVERSE: dict[str, str] = {
    "ES": "ES_D.parquet",
    "NQ": "NQ_D.parquet",
    "CL": "CL_M1_D.parquet",
    "BZ": "BZ_M1_D.parquet",
    "HG": "HG_M1_D.parquet",
    "SI": "SI_M1_D.parquet",
    "GC": "GC_M1_D.parquet",
    "ZN": "ZN_M1_D.parquet",
    "ZB": "ZB_M1_D.parquet",
    "6E": "6E_M1_D.parquet",
    "6J": "6J_M1_D.parquet",
}

# Same C1-C8 as B2/B2b for direct cross-audit comparison.
CELLS: dict[str, EwmacConfig] = {
    "C1_canonical": EwmacConfig(
        speeds=((16, 64), (32, 128), (64, 256)), fdm=1.35,
    ),
    "C2_short_speeds": EwmacConfig(
        speeds=((4, 16), (8, 32), (16, 64)), fdm=1.35,
    ),
    "C3_long_speeds": EwmacConfig(
        speeds=((32, 128), (64, 256), (128, 512)), fdm=1.35,
    ),
    "C4_full_six": EwmacConfig(
        speeds=((4, 16), (8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        fdm=1.51,
    ),
    "C5_two_speed": EwmacConfig(
        speeds=((16, 64), (64, 256)), fdm=1.20,
    ),
    "C6_singleton_canonical": EwmacConfig(
        speeds=((32, 128),), fdm=1.0,
    ),
    "C7_full_six_unclipped": EwmacConfig(
        speeds=((4, 16), (8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        fdm=1.51, forecast_cap=1e9,
    ),
    "C8_gross_no_costs": EwmacConfig(
        speeds=((16, 64), (32, 128), (64, 256)), fdm=1.35, apply_costs=False,
    ),
}
CANONICAL_CELL = "C1_canonical"

PLATEAU_NEIGHBOURS: dict[str, EwmacConfig] = {
    "P_shift_short": replace(CELLS[CANONICAL_CELL], speeds=((8, 32), (16, 64), (32, 128))),
    "P_shift_long": replace(CELLS[CANONICAL_CELL], speeds=((32, 128), (64, 256), (128, 512))),
    "P_drop_fast": replace(CELLS[CANONICAL_CELL], speeds=((32, 128), (64, 256)), fdm=1.20),
    "P_drop_slow": replace(CELLS[CANONICAL_CELL], speeds=((16, 64), (32, 128)), fdm=1.20),
}


def _load_close(symbol: str, fname: str) -> pd.Series:
    df = pd.read_parquet(DATA_DIR / fname)
    if "close" not in df.columns:
        raise ValueError(f"{fname} missing 'close' column: {list(df.columns)}")
    s = df["close"].astype(float)
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # Mask non-positive closes (e.g., CL 2020-04-20 -$37.63 settle anomaly).
    # Forward-fill from prior positive close so log returns stay finite.
    s = s.where(s > 0).ffill()
    s.name = symbol
    return s


def load_universe() -> pd.DataFrame:
    parts = [_load_close(sym, fname) for sym, fname in UNIVERSE.items()]
    df = pd.concat(parts, axis=1).sort_index()
    first_valid = df.dropna(how="any").index
    if len(first_valid) == 0:
        raise RuntimeError("No common date range across the 11-symbol universe.")
    df = df.loc[first_valid[0]:]
    df = df.ffill(limit=5)
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


def run_cell(cell_name, cfg, closes_visible, closes_sanctuary, folds, *,
             n_trials_sweep, bars_per_year, sweep_sharpes, mc_n_workers):
    cls = StrategyClass.CROSS_ASSET_MOMENTUM
    d = defaults_for(cls)
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
        primary_close=primary, cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, closes_visible),
        periods_per_year=bars_per_year, seed=42,
        extra_series=extras, n_workers=mc_n_workers,
    )
    full = pd.concat([closes_visible, closes_sanctuary])
    sanc_ret = ewmac_returns(full, cfg=cfg).iloc[len(closes_visible):]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched, sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    def _noise_fn(df):
        return ewmac_returns(df, cfg=cfg)

    noise = run_noise_robustness(
        closes_visible, _noise_fn, periods_per_year=bars_per_year,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30),
    )
    inputs = DecisionInputs(
        ci_lo=ci_lo, dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
        pass_threshold_prob=mc.pass_threshold_prob,
        sanctuary_sharpe=sanc_sh,
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
    )
    decision = decide(inputs)
    return CellResult(
        cell=cell_name, n_oos_bars=len(stitched), n_folds=len(folds),
        sharpe=round(sh, 4), ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
        dsr_prob=round(dsr.dsr_prob, 4),
        mc_p_maxdd_gt_threshold=round(mc.p_maxdd_gt_threshold, 4),
        mc_threshold_pct=round(mc.threshold_pct, 4),
        sanctuary_sharpe=round(sanc_sh, 4),
        sanctuary_percentile=round(div.percentile, 4) if np.isfinite(div.percentile) else float("nan"),
        noise_base=round(noise.base_sharpe, 4),
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
        noise_axis=decision.noise_axis,
        verdict=decision.verdict.value, rationale=decision.rationale,
    )


def main():
    print("=" * 72)
    print("B2e -- Carver EWMAC audit on IBKR cross-asset 11-symbol universe")
    print("L52 OVERRIDE: sweep found no plateau; audit run at operator request")
    print("=" * 72)

    closes = load_universe()
    print(
        f"\nUniverse: {len(closes.columns)} instruments  "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)"
    )
    print(f"Symbols: {list(closes.columns)}")

    ewmac_assert_causal(closes, cfg=CELLS[CANONICAL_CELL], n_trials=3, seed=42)
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=12)
    closes_visible = sanc.visible
    closes_sanctuary = sanc.sanctuary
    print(f"Visible: {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}")

    bars_per_year = BARS_PER_YEAR["D"]
    cls_d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    wfo_cfg = cls_d.wfo
    print(
        f"WFO (CROSS_ASSET_MOMENTUM defaults: "
        f"{wfo_cfg.is_min_years}y IS / {wfo_cfg.oos_years}y OOS)"
    )
    folds = build_folds(closes_visible.index, wfo_cfg, bars_per_year=bars_per_year)
    print(f"  -> {len(folds)} folds")

    print("\n[Plateau pre-flight] C1 + 4 EWMAC-speed neighbours (OOS Sharpes)...")
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
    print(f"  Relative spread: {rel_spread:.2%}  (B2 IBKR-3y: 15.28%; B2b yf-21y: 37.10%; L52 gate: <=30%)")
    plateau_passed = rel_spread <= 0.30
    print(f"  Plateau gate: {'PASSED' if plateau_passed else 'FAILED'}")
    if not plateau_passed:
        print("  [continuing despite gate fail -- operator override]")

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
            name, cfg, closes_visible, closes_sanctuary, folds,
            n_trials_sweep=len(CELLS), bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes, mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(r)
        print(f"    Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        print(f"    DSR={r.dsr_prob:.4f}  MC P(>{r.mc_threshold_pct * 100:.0f}%)={r.mc_p_maxdd_gt_threshold:.4f}")
        print(f"    Sanc Sharpe={r.sanctuary_sharpe:+.4f}")
        print(f"    Noise: base={r.noise_base:+.4f} mean_pass={r.noise_passes_mean} worst_pass={r.noise_passes_worst} axis={r.noise_axis}")
        print(f"    Verdict (5-axis): {r.verdict}")

    # Result log.
    report_fp = REPORTS_DIR / "result_log.md"
    with report_fp.open("w", encoding="utf-8") as fh:
        fh.write("# B2e Audit Result Log -- Carver EWMAC on IBKR Cross-Asset 11sym Universe\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write("**Data:** ES/NQ (existing) + CL/BZ/HG/SI/GC/ZN/ZB/6E/6J (Databento, this wave)\n")
        fh.write(
            f"**Universe:** {len(closes.columns)} instruments, "
            f"{closes.index[0].date()} -> {closes.index[-1].date()} "
            f"({len(closes)} bars).\n"
        )
        fh.write(
            f"**Visible:** {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}  "
            f"WFO folds: {len(folds)}\n\n"
        )
        fh.write("**L52 OVERRIDE:** IS-only sweep found no plateau (spread 45% > 30% gate); ")
        fh.write("audit run anyway per operator request, using B2/B2b canonical C1-C8 cells.\n\n")

        fh.write("## §4.1 Plateau pre-flight (OOS)\n\n")
        fh.write(f"**Relative spread:** {rel_spread:.2%}  -- ")
        fh.write(f"{'PASSED' if plateau_passed else 'FAILED'} (gate: <= 30%)\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell 5-axis matrix\n\n")
        fh.write("| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P | Sanc Sharpe | Noise base | Noise axis | Verdict |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|\n")
        for r in results:
            fh.write(
                f"| {r.cell} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
                f"{r.dsr_prob:.4f} | {r.mc_p_maxdd_gt_threshold:.4f} | "
                f"{r.sanctuary_sharpe:+.4f} | {r.noise_base:+.4f} | "
                f"{r.noise_axis} | {r.verdict} |\n"
            )

        c1 = next(r for r in results if r.cell == "C1_canonical")
        fh.write("\n## §4.3 B2 -> B2b -> B2e comparison\n\n")
        fh.write("| Metric | B2 IBKR-3y | B2b yf-21y | B2e IBKR-x-asset-9y |\n|---|---:|---:|---:|\n")
        fh.write(f"| Universe size | 24 | 31 | {len(closes.columns)} |\n")
        fh.write(f"| Bars | 760 | 5574 | {len(closes)} |\n")
        fh.write(f"| WFO folds | 2 | 60 | {len(folds)} |\n")
        fh.write(f"| C1 Sharpe | +2.0218 | -0.2817 | {c1.sharpe:+.4f} |\n")
        fh.write(f"| C1 CI_lo | -0.044 | n/a | {c1.ci_lo:+.3f} |\n")
        fh.write(f"| Plateau spread | 15.28% | 37.10% | {rel_spread:.2%} |\n")
        fh.write(f"| Matrix verdict | COND_WP | RETIRED | {c1.verdict} |\n")

        deploy_eligible = [
            r for r in results
            if r.cell not in ("C6_singleton_canonical", "C7_full_six_unclipped", "C8_gross_no_costs")
            and (r.verdict == "DEPLOY"
                 or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best"))
            and r.ci_lo > 0
        ]
        fh.write("\n## §4.4 Promotion verdict\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"**PROMOTE {best.cell}** -- CI_lo={best.ci_lo:+.3f}, "
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}.\n"
            )
        else:
            fh.write(
                "No cell deployment-eligible under strict 5-axis + CI_lo>0 rule. "
                "Confirms the L52 sweep finding: B2e EWMAC does not deploy on this universe.\n"
            )

    print(f"\nResult log: {report_fp}")
    return results


if __name__ == "__main__":
    main()
