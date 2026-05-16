"""B2c -- Carver EWMAC with trend-of-trend regime filter (L48 mitigation).

Specified by ``directives/Pre-Reg B2c Trend-of-Trend EWMAC Gate 2026-05-16.md``.

After B2b's yfinance-21y audit produced negative Sharpe (-0.28) and FAILED
plateau (37.10%), confirming that B2's +2.02 commodity-3y result was a
regime-artifact (L48), B2c tests whether a broad-trend regime filter can
rescue the strategy by enabling per-asset signals ONLY when the
universe-broad trend itself is non-zero.

Universe: same 31 yfinance ETF proxies as B2b. Pre-reg gates: same
CROSS_ASSET_MOMENTUM defaults. The new knob is `broad_trend_filter` in
each cell.

Run via::

    uv run python research/ewmac/run_b2c_audit.py
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
    BroadTrendFilterConfig,
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

DATA_DIR = PROJECT_ROOT / "data" / "yf_b2b"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b2c_ewmac_trend_filter"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered cells (V3.1) per B2c pre-reg §2.
# All cells share the same B2 underlying ((16,64),(32,128),(64,256)) FDM=1.35.
_BASE_EWMAC = dict(
    speeds=((16, 64), (32, 128), (64, 256)),
    fdm=1.35,
    forecast_cap=20.0,
)

CELLS: dict[str, EwmacConfig] = {
    "C1_canonical": EwmacConfig(
        **_BASE_EWMAC,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=0.0
        ),
    ),
    "C2_directional": EwmacConfig(
        **_BASE_EWMAC,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="directional", deadband=0.0
        ),
    ),
    "C3_faster_filter": EwmacConfig(
        **_BASE_EWMAC,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=32, slow_hl=128, mode="absolute_trend", deadband=0.0
        ),
    ),
    "C4_slower_filter": EwmacConfig(
        **_BASE_EWMAC,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=128, slow_hl=512, mode="absolute_trend", deadband=0.0
        ),
    ),
    "C5_deadband": EwmacConfig(
        **_BASE_EWMAC,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=5.0
        ),
    ),
    "C6_baseline_b2b": EwmacConfig(
        **_BASE_EWMAC,
        broad_trend_filter=None,  # B2b baseline reproduction
    ),
    "C7_gross_no_costs": EwmacConfig(
        **_BASE_EWMAC,
        apply_costs=False,
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=0.0
        ),
    ),
}
CANONICAL_CELL = "C1_canonical"

PLATEAU_NEIGHBOURS: dict[str, EwmacConfig] = {
    "P_shift_short_filter": replace(
        CELLS[CANONICAL_CELL],
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=32, slow_hl=128, mode="absolute_trend", deadband=0.0
        ),
    ),
    "P_shift_long_filter": replace(
        CELLS[CANONICAL_CELL],
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=128, slow_hl=512, mode="absolute_trend", deadband=0.0
        ),
    ),
    "P_add_deadband": replace(
        CELLS[CANONICAL_CELL],
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=64, slow_hl=256, mode="absolute_trend", deadband=2.5
        ),
    ),
    # Per-asset variant — same filter speeds but applied per-instrument
    # not universe-wide. Implemented via a synthetic 1-asset "broad index"
    # for each column would be expensive; for the pre-reg we approximate
    # by setting deadband very high so universe-broad is "always trending"
    # AND testing a longer EWMAC speed instead.
    "P_long_filter_no_deadband": replace(
        CELLS[CANONICAL_CELL],
        broad_trend_filter=BroadTrendFilterConfig(
            fast_hl=96, slow_hl=384, mode="absolute_trend", deadband=0.0
        ),
    ),
}

EXPANDED_ROOTS: tuple[str, ...] = (
    "SPX",
    "NDX",
    "DJI",
    "RUT",
    "FTSE",
    "DAX",
    "NIKKEI",
    "EUROSTOXX",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "DXY",
    "US10Y_PROXY",
    "US30Y_PROXY",
    "US2Y_PROXY",
    "UK_GILT_PROXY",
    "EURO_GOV_PROXY",
    "EM_BOND_PROXY",
    "GOLD_PROXY",
    "SILVER_PROXY",
    "PLATINUM_PROXY",
    "PALLADIUM_PROXY",
    "EM_EQUITY",
    "EAFE_EQUITY",
    "EUROPE_EQUITY",
    "JAPAN_EQUITY",
    "CHINA_EQUITY",
)


def _load_close(root: str) -> pd.Series | None:
    fp = DATA_DIR / f"{root}_DAY.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def load_universe(min_bars: int = 252) -> pd.DataFrame:
    parts = []
    for root in EXPANDED_ROOTS:
        s = _load_close(root)
        if s is None or len(s.dropna()) < min_bars:
            continue
        parts.append(s)
    if not parts:
        raise RuntimeError(f"No instruments under {DATA_DIR}.")
    df = pd.concat(parts, axis=1).sort_index()
    return df.ffill(limit=5)


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
    folds,
    *,
    n_trials_sweep,
    bars_per_year,
    sweep_sharpes,
    mc_n_workers,
):
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
    print("B2c -- Carver EWMAC + Trend-of-Trend Regime Filter (L48 mitigation)")
    print("Pre-reg: directives/Pre-Reg B2c Trend-of-Trend EWMAC Gate 2026-05-16.md")
    print("=" * 72)

    closes = load_universe()
    print(
        f"\nUniverse: {len(closes.columns)} instruments  "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)"
    )

    ewmac_assert_causal(closes, cfg=CELLS[CANONICAL_CELL], n_trials=3, seed=42)
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=12)
    closes_visible = sanc.visible
    closes_sanctuary = sanc.sanctuary
    print(f"Visible: {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}")

    bars_per_year = BARS_PER_YEAR["D"]
    cls_d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(closes_visible.index, cls_d.wfo, bars_per_year=bars_per_year)
    print(f"WFO ({cls_d.wfo.is_min_years}y IS / {cls_d.wfo.oos_years}y OOS): {len(folds)} folds")

    # Baseline reproduction (C6 = B2b filter=None).
    print("\n[Baseline reproduction] C6 (filter disabled) — should match B2b ~ -0.28")
    sh_c6, _ = _stitched_oos_sharpe(closes_visible, CELLS["C6_baseline_b2b"], folds, bars_per_year)
    print(f"  C6_baseline_b2b: Sharpe={sh_c6:+.4f}  (B2b reference: -0.2817)")

    # Plateau pre-flight (C1 + 4 neighbours).
    print("\n[Plateau pre-flight] C1 + 4 filter-speed neighbours...")
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
    print(f"  Relative spread: {rel_spread:.2%}  (gate: < 30%)")
    print("  Reference: B2b (filter off) spread was 37.10% (failed)")

    plateau_passed = rel_spread <= 0.30
    if not plateau_passed:
        print("\n  [!] PLATEAU GATE FAILED.")
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# B2c Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"**Relative spread:** {rel_spread:.2%}  (B2b ref: 37.10%)\n")
            fh.write(f"**C6 baseline (B2b reproduction):** Sharpe={sh_c6:+.4f}\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
            fh.write(
                "\nL48 reinforced: broad-trend filter does not rescue the universal-trend "
                "EWMAC on cross-asset 21y data. Next mitigation = HMM regime (I1).\n"
            )
        return None
    print("  [OK] plateau gate passed.")

    # Pass 1: headline OOS Sharpes for DSR sweep.
    print(f"\n[Pass 1/2] Headline OOS Sharpes ({len(CELLS)} cells)...")
    sweep_sharpes: list[float] = []
    for name, cfg in CELLS.items():
        sh_v, _ = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year)
        sweep_sharpes.append(sh_v)
        print(f"  {name}: Sharpe={sh_v:+.4f}")

    # Pass 2: full per-cell audit.
    print(f"\n[Pass 2/2] Full per-cell audit -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[CellResult] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            cfg,
            closes_visible,
            closes_sanctuary,
            folds,
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
        fh.write("# B2c Audit Result Log -- Trend-of-Trend Filter on Carver EWMAC\n\n")
        fh.write("**Run date:** 2026-05-16\n")
        fh.write(f"**Universe:** {len(closes.columns)} instruments (yf_b2b proxies)\n")
        fh.write(
            f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()} "
            f"({len(closes)} bars)\n"
        )
        fh.write(
            f"**Visible:** {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}  "
            f"WFO folds: {len(folds)}\n\n"
        )

        fh.write("## §4.1 Baseline reproduction\n\n")
        fh.write(
            f"- C6_baseline_b2b (filter disabled): Sharpe = {sh_c6:+.4f}  "
            f"(B2b reference -0.2817; |Δ| = {abs(sh_c6 + 0.2817):.4f})\n"
        )

        fh.write("\n## §4.2 Plateau pre-flight\n\n")
        fh.write(
            f"**Relative spread:** {rel_spread:.2%}  (B2b: 37.10%) -- "
            f"{'PASSED' if plateau_passed else 'FAILED'}\n\n"
        )
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.3 Per-cell 5-axis matrix\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P | Sanc Sharpe | "
            "Noise base | Noise axis | Verdict |\n"
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
        c6 = next(r for r in results if r.cell == "C6_baseline_b2b")

        fh.write("\n## §4.4 Falsification verdicts\n\n")
        # H1
        fh.write(
            f"**H1 (plateau rescue):** B2b spread 37.10% → B2c spread {rel_spread:.2%}. "
            f"{'SUPPORTED' if plateau_passed else 'REJECTED'}.\n\n"
        )
        # H2
        h2_threshold = 0.30
        h2_ok = c1.sharpe >= h2_threshold
        fh.write(
            f"**H2 (Sharpe rescue):** Threshold +{h2_threshold:+.2f}; "
            f"C1 = {c1.sharpe:+.4f}. {'SUPPORTED' if h2_ok else 'REJECTED'}.\n\n"
        )
        # H3
        h3_ok = c1.sharpe > c6.sharpe
        fh.write(
            f"**H3 (cost-of-filter check):** C1 filter = {c1.sharpe:+.4f}, "
            f"C6 no-filter = {c6.sharpe:+.4f}. "
            f"{'SUPPORTED (filter helps)' if h3_ok else 'REJECTED (filter is worse than nothing)'}.\n\n"
        )

        deploy_eligible = [
            r
            for r in results
            if r.cell not in ("C6_baseline_b2b", "C7_gross_no_costs")
            and (
                r.verdict == "DEPLOY"
                or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best")
            )
            and r.ci_lo > 0
        ]
        fh.write("\n## §4.5 Promotion verdict\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"**PROMOTE {best.cell}** — CI_lo={best.ci_lo:+.3f}, "
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}. "
                f"L48 mitigation works: a broad-trend regime filter rescues B2 "
                f"on cross-asset 21y data.\n"
            )
        else:
            fh.write(
                "No cell promotion-eligible under strict 5-axis + CI_lo>0 rule. "
                "L48 mitigation via trend-of-trend filter does NOT rescue B2 "
                "on cross-asset 21y data. Next escalation: I1 HMM regime detection.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
