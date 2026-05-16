"""A1 -- Residual momentum audit harness.

Specified by ``directives/Pre-Reg A1 Residual Momentum 2026-05-15.md``.

Run via::

    uv run python research/residual_momentum/run_a1_audit.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.residual_momentum.residual_strategy import (  # noqa: E402
    ResidualConfig,
    residual_assert_causal,
    residual_returns,
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
STOCK_DIR = DATA_DIR / "SP500_universe"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "a1_residual_momentum"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered cells (V3.1).
CELLS: dict[str, ResidualConfig] = {
    "C1_canonical": ResidualConfig(
        signal_mode="residual", momentum_window_months=12, skip_months=1
    ),
    "C2_raw_momentum": ResidualConfig(signal_mode="raw", momentum_window_months=12, skip_months=1),
    "C3_window_24": ResidualConfig(
        signal_mode="residual", momentum_window_months=24, skip_months=1
    ),
    "C4_window_6": ResidualConfig(signal_mode="residual", momentum_window_months=6, skip_months=1),
    "C5_tercile": ResidualConfig(signal_mode="residual", breadth_pct=0.333),
    "C6_vol_weighted": ResidualConfig(signal_mode="residual", weighting="inverse_vol"),
    "C7_long_only": ResidualConfig(signal_mode="residual", long_only=True),
    "C8_gross_no_costs": ResidualConfig(signal_mode="residual", apply_costs=False),
}
CANONICAL_CELL = "C1_canonical"

# Plateau pre-flight neighbours of C1.
PLATEAU_NEIGHBOURS: dict[str, ResidualConfig] = {
    "P_window_9": replace(CELLS[CANONICAL_CELL], momentum_window_months=9),
    "P_window_15": replace(CELLS[CANONICAL_CELL], momentum_window_months=15),
    "P_skip_0": replace(CELLS[CANONICAL_CELL], skip_months=0),
    "P_skip_2": replace(CELLS[CANONICAL_CELL], skip_months=2),
}


def load_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (stocks_close, ff3_df) ready for the strategy.

    Returns:
        stocks_close: DataFrame index=daily date, columns=ticker, cell=close.
        ff3_df: DataFrame index=daily date, columns=mkt_rf/smb/hml/rf.
    """
    print(f"Scanning {STOCK_DIR} for stock parquets...")
    parquets = sorted(STOCK_DIR.glob("*_D.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No stock parquets in {STOCK_DIR}")
    print(f"  -> {len(parquets)} parquet files found")

    closes: dict[str, pd.Series] = {}
    for p in parquets:
        ticker = p.stem.replace("_D", "")
        try:
            df = pd.read_parquet(p)
            if "close" not in df.columns:
                continue
            s = df["close"].astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s.dropna()
            if len(s) < 252:
                continue
            closes[ticker] = s
        except Exception:
            continue
    print(f"  -> {len(closes)} stocks loaded after filtering")

    stocks_close = pd.DataFrame(closes).sort_index()

    # Load FF3 factors.
    ff_path = DATA_DIR / "FF3_daily.parquet"
    if not ff_path.exists():
        raise FileNotFoundError(f"Missing {ff_path}")
    ff3 = pd.read_parquet(ff_path)
    ff3.index = pd.to_datetime(ff3.index).tz_localize(None).normalize()

    # Restrict to the overlap.
    common_idx = stocks_close.index.intersection(ff3.index)
    stocks_close = stocks_close.loc[common_idx]
    ff3 = ff3.loc[common_idx]
    return stocks_close, ff3


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


def _stitched_oos_sharpe(stocks, ff3, cfg, folds, bars_per_year):
    rets = residual_returns(stocks, ff3, cfg=cfg)
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def _strategy_fn_for_cell(cfg: ResidualConfig, stocks_visible, ff3_visible):
    """MC closure. The MC primitive bootstraps the 'close' column; we
    don't have a meaningful single primary here (it's a basket). We
    treat the FIRST stock as primary and the other 499 as extras. The
    FF3 factors are passed through unchanged (NOT bootstrapped — they're
    market-level data that exists regardless of any individual stock's
    bootstrap path).
    """
    primary_ticker = stocks_visible.columns[0]
    other_tickers = list(stocks_visible.columns[1:])

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        # Reassemble the stocks frame.
        s_df = pd.DataFrame(index=df.index)
        s_df[primary_ticker] = df["close"]
        for t in other_tickers:
            if t in df.columns:
                s_df[t] = df[t]
        # FF3 factors are the visible originals reindexed to df.index.
        ff3_local = ff3_visible.reindex(df.index).ffill()
        return residual_returns(s_df, ff3_local, cfg=cfg)

    return strategy_fn


def run_cell(
    cell_name: str,
    cfg: ResidualConfig,
    stocks_visible: pd.DataFrame,
    stocks_sanctuary: pd.DataFrame,
    ff3_visible: pd.DataFrame,
    ff3_sanctuary: pd.DataFrame,
    *,
    n_trials_sweep: int,
    bars_per_year: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
) -> CellResult:
    cls = StrategyClass.CROSS_ASSET_MOMENTUM
    d = defaults_for(cls)
    folds = build_folds(stocks_visible.index, d.wfo, bars_per_year=bars_per_year)
    if not folds:
        raise RuntimeError(f"{cell_name}: no folds")

    sh, stitched = _stitched_oos_sharpe(stocks_visible, ff3_visible, cfg, folds, bars_per_year)
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=bars_per_year, n_resamples=1000, seed=42
    )

    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh, sr_var_across_trials=sr_var, returns=stitched, n_trials=n_trials_sweep
    )

    # MC: bootstrap the multi-stock returns matrix with shared blocks.
    primary = stocks_visible.iloc[:, 0]
    extras: dict[str, pd.Series] = {t: stocks_visible[t] for t in stocks_visible.columns[1:]}
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, stocks_visible, ff3_visible),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    # Sanctuary.
    full_stocks = pd.concat([stocks_visible, stocks_sanctuary])
    full_ff3 = pd.concat([ff3_visible, ff3_sanctuary])
    sanc_ret = residual_returns(full_stocks, full_ff3, cfg=cfg).iloc[len(stocks_visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    # Noise gate.
    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        ff3_local = ff3_visible.reindex(df.index).ffill()
        return residual_returns(df, ff3_local, cfg=cfg)

    noise = run_noise_robustness(
        stocks_visible,
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
    print("A1 -- Residual Momentum Audit (Blitz-Huij-Martens 2011)")
    print("Pre-reg: directives/Pre-Reg A1 Residual Momentum 2026-05-15.md")
    print("=" * 72)

    t0 = time.time()
    stocks_close, ff3 = load_universe()
    print(
        f"\nLoaded: {len(stocks_close.columns)} stocks x {len(stocks_close)} bars "
        f"({stocks_close.index[0].date()} -> {stocks_close.index[-1].date()})  "
        f"FF3 rows: {len(ff3)}"
    )
    print(f"Load time: {time.time() - t0:.1f}s")

    # Drop stocks with too few observations to compute the 36-month regression.
    min_obs = ResidualConfig().regression_months * 21 + 252
    valid_mask = stocks_close.notna().sum(axis=0) >= min_obs
    stocks_close = stocks_close.loc[:, valid_mask]
    print(f"After min-obs filter ({min_obs} bars): {len(stocks_close.columns)} stocks")

    print("\nCausality smoke test...")
    t0 = time.time()
    residual_assert_causal(
        stocks_close.iloc[:, :30],  # subset of 30 stocks to keep test fast
        ff3,
        cfg=ResidualConfig(regression_months=24),
        n_trials=1,
    )
    print(f"  PASSED ({time.time() - t0:.1f}s)")

    sanc = slice_sanctuary(stocks_close, months=12)
    stocks_visible = sanc.visible
    stocks_sanctuary = sanc.sanctuary
    ff3_visible = ff3.reindex(stocks_visible.index)
    ff3_sanctuary = ff3.reindex(stocks_sanctuary.index)
    print(
        f"\nSanctuary: {sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()}\n"
        f"  Visible: {len(stocks_visible)} bars  Sanctuary: {len(stocks_sanctuary)} bars"
    )

    bars_per_year = BARS_PER_YEAR["D"]
    cls_d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(stocks_visible.index, cls_d.wfo, bars_per_year=bars_per_year)
    print(f"WFO: {len(folds)} folds")

    print("\n[Plateau pre-flight] canonical + 4 neighbours...")
    plateau_sharpes: dict[str, float] = {}
    sh_c1, _ = _stitched_oos_sharpe(
        stocks_visible, ff3_visible, CELLS[CANONICAL_CELL], folds, bars_per_year
    )
    plateau_sharpes[CANONICAL_CELL] = sh_c1
    for name, cfg in PLATEAU_NEIGHBOURS.items():
        sh_n, _ = _stitched_oos_sharpe(stocks_visible, ff3_visible, cfg, folds, bars_per_year)
        plateau_sharpes[name] = sh_n
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
            fh.write("# A1 Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"Relative Sharpe spread: {rel_spread:.2%} > 30%\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
        return None
    print("  [OK] plateau gate passed.")

    print(f"\n[Pass 1/2] Headline OOS Sharpes ({len(CELLS)} cells)...")
    sweep_sharpes: list[float] = []
    for name, cfg in CELLS.items():
        t1 = time.time()
        sh_v, _ = _stitched_oos_sharpe(stocks_visible, ff3_visible, cfg, folds, bars_per_year)
        sweep_sharpes.append(sh_v)
        print(f"  {name}: Sharpe={sh_v:+.4f}  ({time.time() - t1:.0f}s)")

    print(f"\n[Pass 2/2] Full per-cell audit -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[CellResult] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        t1 = time.time()
        r = run_cell(
            name,
            cfg,
            stocks_visible,
            stocks_sanctuary,
            ff3_visible,
            ff3_sanctuary,
            n_trials_sweep=len(CELLS),
            bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes,
            mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(r)
        print(
            f"    Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
            f"DSR={r.dsr_prob:.4f}  MC P(>{r.mc_threshold_pct * 100:.0f}%)={r.mc_p_maxdd_gt_threshold:.4f}"
        )
        print(
            f"    Sanc Sharpe={r.sanctuary_sharpe:+.4f}  "
            f"Noise: base={r.noise_base:+.4f} mean={r.noise_passes_mean} "
            f"worst={r.noise_passes_worst} axis={r.noise_axis}"
        )
        print(f"    Verdict (5-axis): {r.verdict}  -- {r.rationale}  ({time.time() - t1:.0f}s)")

    # Result log.
    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# A1 Audit Result Log -- Residual Momentum (BHM 2011)\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(
            f"**Universe:** {len(stocks_visible.columns)} S&P 500 stocks (post-min-obs filter)\n"
        )
        fh.write(
            f"**Data range:** {stocks_visible.index[0].date()} -> "
            f"{stocks_visible.index[-1].date()}  ({len(stocks_visible)} visible bars)\n"
        )
        fh.write(f"**Sanctuary:** {len(stocks_sanctuary)} bars\n\n")

        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(f"Relative spread: {rel_spread:.2%}  --  PASSED (gate 30%)\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell verdicts (5-axis, J3 / L24)\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P(>X%) | "
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

        # Residual vs raw comparison.
        c1 = next(r for r in results if r.cell == "C1_canonical")
        c2 = next(r for r in results if r.cell == "C2_raw_momentum")
        fh.write("\n## §4.3 Residual vs raw momentum (C1 vs C2)\n\n")
        fh.write(
            f"- C1 residual: Sharpe={c1.sharpe:+.4f}, CI_lo={c1.ci_lo:+.3f}\n"
            f"- C2 raw:      Sharpe={c2.sharpe:+.4f}, CI_lo={c2.ci_lo:+.3f}\n"
            f"- Delta:       {c1.sharpe - c2.sharpe:+.4f}\n"
        )

        # Gross vs net.
        c8 = next(r for r in results if r.cell == "C8_gross_no_costs")
        fh.write("\n## §4.4 Gross vs net (C1 vs C8)\n\n")
        fh.write(
            f"- Gross (C8, costs OFF): Sharpe={c8.sharpe:+.4f}, CI_lo={c8.ci_lo:+.3f}\n"
            f"- Net   (C1, costs ON):  Sharpe={c1.sharpe:+.4f}, CI_lo={c1.ci_lo:+.3f}\n"
            f"- Cost drag:             {c8.sharpe - c1.sharpe:+.4f}\n"
        )

        # Recommendation.
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
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}).\n"
            )
        else:
            fh.write(
                "No cell deployment-eligible under the 5-axis matrix. "
                "Document negative result + lessons; the BHM residual-momentum "
                "edge does not survive on retail-implementable S&P 500 stocks at "
                "realistic costs in this audit's specific universe slice.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
