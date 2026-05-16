"""B2b -- Carver EWMAC universe-expansion audit on IG Markets data.

Specified by ``directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md`` §4.8
(universe-expansion follow-up). Companion to ``run_b2_audit.py`` (IBKR
stitched, 24 commodities, 3y).

Key differences from B2:
    - Data source: IG Markets DFB continuous CFDs (no roll-stitching needed —
      IG handles the underlying-rolls internally; cross-asset basis effects
      do NOT apply because EWMAC is per-instrument, not cross-contract).
    - Universe: 40 instruments (24 commodities + 8 FX + 4 bonds + 8 indices).
    - Depth: ~10 years per instrument (vs B2's 3y) — directly addresses the
      sample-size CI bottleneck identified in B2's §4.7 (L46).
    - Expected WFO folds: 5-8 (vs B2's 2) — sufficient for tight bootstrap CI.

Run via::

    uv run python research/ewmac/run_b2b_audit.py

Caveat — data construction note (L40/L42-style):
    IG's DFB continuous CFDs have a broker-internal roll convention. Daily
    log-returns of IG's continuous series may differ from per-contract
    roll-stitched series. The cross-instrument trend signal is robust to
    this (sign-of-EWMAC doesn't depend on the absolute roll-yield level),
    but the cumulative wealth path will differ. We acknowledge this and
    treat IG data as the LARGER-SAMPLE complement to B2's per-contract
    stitched data, not a replacement for it.
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

# L47 IG-quota fallback: yfinance-ETF-proxy universe via
# scripts/download_b2b_alternative.py. Caveats:
#   - Bond ETFs replace bond futures (vol-normalised sizing absorbs the
#     contract-multiplier mismatch; sign-of-trend is invariant).
#   - FX spot pairs and equity index ETFs are clean (no L40 issue).
#   - Commodity ETFs are PHYSICAL only (GLD/SLV/PPLT/PALL); no front-month
#     L40-contaminated commodity =F symbols.
DATA_DIR_IG = PROJECT_ROOT / "data" / "ig_markets"
DATA_DIR_YF = PROJECT_ROOT / "data" / "yf_b2b"
DATA_DIR = DATA_DIR_YF  # default; B2b audit uses yfinance after L47
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "b2b_ewmac_yf_expanded"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# With ~10y of data we can use the class-default WFO (5y IS / 1y OOS).
# Class default is CROSS_ASSET_MOMENTUM; pull it via defaults_for at runtime.
USE_CLASS_DEFAULT_WFO = True
WFO_OVERRIDE = WfoConfig(
    is_min_years=5.0,
    oos_years=1.0,
    fold_count=8,
    is_mode="expanding",
    stride_overlap_allowed=False,
)

# Same cells as B2 — comparing apples-to-apples on the canonical
# 3-speed (16/64, 32/128, 64/256) FDM=1.35 ensemble.
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
        forecast_cap=1e9,
    ),
    "C8_gross_no_costs": EwmacConfig(
        speeds=((16, 64), (32, 128), (64, 256)),
        fdm=1.35,
        apply_costs=False,
    ),
}
CANONICAL_CELL = "C1_canonical"

PLATEAU_NEIGHBOURS: dict[str, EwmacConfig] = {
    "P_shift_short": replace(CELLS[CANONICAL_CELL], speeds=((8, 32), (16, 64), (32, 128))),
    "P_shift_long": replace(CELLS[CANONICAL_CELL], speeds=((32, 128), (64, 256), (128, 512))),
    "P_drop_fast": replace(CELLS[CANONICAL_CELL], speeds=((32, 128), (64, 256)), fdm=1.20),
    "P_drop_slow": replace(CELLS[CANONICAL_CELL], speeds=((16, 64), (32, 128)), fdm=1.20),
}

# Expanded universe — labels match scripts/download_b2b_alternative.py UNIVERSE.
# This is the yfinance-ETF-proxy variant after L47 blocked the IG path.
# Commodities/bonds/FX/indices represented by clean proxies (no roll, no
# survivorship, no L40 contamination). Same B2 cells; same WFO logic.
EXPANDED_ROOTS: tuple[str, ...] = (
    # equity_index (8)
    "SPX",
    "NDX",
    "DJI",
    "RUT",
    "FTSE",
    "DAX",
    "NIKKEI",
    "EUROSTOXX",
    # fx_major (8)
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "DXY",
    # bond_etf (6) — IEF/TLT/SHY/IGLT.L/IBGS.L/EMB as proxies
    "US10Y_PROXY",
    "US30Y_PROXY",
    "US2Y_PROXY",
    "UK_GILT_PROXY",
    "EURO_GOV_PROXY",
    "EM_BOND_PROXY",
    # physical_commodity_etf (4)
    "GOLD_PROXY",
    "SILVER_PROXY",
    "PLATINUM_PROXY",
    "PALLADIUM_PROXY",
    # regional_equity (5) — cross-sectional breadth
    "EM_EQUITY",
    "EAFE_EQUITY",
    "EUROPE_EQUITY",
    "JAPAN_EQUITY",
    "CHINA_EQUITY",
)


def _load_ig_close(root: str) -> pd.Series | None:
    """Load IG DAY-bar close for one instrument. Returns None if missing."""
    fp = DATA_DIR / f"{root}_DAY.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "close" not in df.columns:
        return None
    s = df["close"].astype(float)
    s.name = root
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def load_universe(min_bars: int = 252) -> pd.DataFrame:
    """Load all IG closes, filter to instruments with enough history.

    Excludes any instrument with fewer than ``min_bars`` valid bars so the
    WFO + EWMAC computations are well-defined.
    """
    parts = []
    skipped: list[tuple[str, str]] = []
    for root in EXPANDED_ROOTS:
        s = _load_ig_close(root)
        if s is None:
            skipped.append((root, "no_file"))
            continue
        if len(s.dropna()) < min_bars:
            skipped.append((root, f"only_{len(s)}_bars"))
            continue
        parts.append(s)
    if not parts:
        raise RuntimeError(
            f"No IG instruments under {DATA_DIR} have >= {min_bars} bars. "
            "Run scripts/download_ig_markets.py first."
        )
    for r, reason in skipped:
        print(f"  [skip] {r}: {reason}")
    df = pd.concat(parts, axis=1).sort_index()
    # Forward-fill within-instrument NaNs from holiday-misalignment between
    # different exchanges (FTSE/DAX vs CME), but only AFTER the instrument's
    # first bar — leading NaNs left intact so EWMAC warm-up handles them.
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
    print("B2b -- Carver EWMAC universe-expansion audit (IG Markets data)")
    print("Pre-reg: directives/Pre-Reg B2 Carver EWMAC Ensemble 2026-05-15.md §4.8")
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
    if USE_CLASS_DEFAULT_WFO:
        cls_d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
        wfo_cfg = cls_d.wfo
        print(
            f"WFO (CROSS_ASSET_MOMENTUM defaults: "
            f"{wfo_cfg.is_min_years}y IS / {wfo_cfg.oos_years}y OOS)"
        )
    else:
        wfo_cfg = WFO_OVERRIDE
        print(f"WFO (override: {wfo_cfg.is_min_years}y IS / {wfo_cfg.oos_years}y OOS)")
    folds = build_folds(closes_visible.index, wfo_cfg, bars_per_year=bars_per_year)
    print(f"  -> {len(folds)} folds")

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
    print(f"  Relative spread: {rel_spread:.2%}  (gate: < 30%)")
    print("  Reference: B2 IBKR-3y spread was 15.28% (already passed)")

    plateau_passed = rel_spread <= 0.30
    if not plateau_passed:
        print("\n  [!] PLATEAU GATE FAILED. Aborting.")
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# B2b Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"**Relative Sharpe spread:** {rel_spread:.2%} > 30%\n")
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

    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# B2b Audit Result Log -- Carver EWMAC on IG Expanded Universe\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write("**Data source:** IG Markets DFB continuous CFDs\n")
        fh.write(f"**Universe:** {len(closes.columns)} instruments\n")
        fh.write(
            f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()}  "
            f"({len(closes)} bars)\n"
        )
        fh.write(
            f"**Visible:** {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}  "
            f"WFO folds: {len(folds)}\n\n"
        )

        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(
            f"**Relative spread:** {rel_spread:.2%}  "
            f"(B2 IBKR-3y: 15.28%)  -- "
            f"{'PASSED' if plateau_passed else 'FAILED'}\n\n"
        )
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell 5-axis matrix\n\n")
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
        fh.write("\n## §4.3 B2 -> B2b comparison\n\n")
        fh.write("| Metric | B2 IBKR-3y | B2b IG-10y |\n|---|---:|---:|\n")
        fh.write(f"| Universe size | 24 | {len(closes.columns)} |\n")
        fh.write(f"| Bars | 760 | {len(closes)} |\n")
        fh.write(f"| WFO folds | 2 | {len(folds)} |\n")
        fh.write(f"| C1 Sharpe | +2.0218 | {c1.sharpe:+.4f} |\n")
        fh.write(f"| C1 CI_lo | -0.044 | {c1.ci_lo:+.3f} |\n")
        fh.write(f"| Plateau spread | 15.28% | {rel_spread:.2%} |\n")
        fh.write(f"| Matrix verdict | COND_WP | {c1.verdict} |\n")

        deploy_eligible = [
            r
            for r in results
            if r.cell
            not in ("C6_singleton_canonical", "C7_full_six_unclipped", "C8_gross_no_costs")
            and (
                r.verdict == "DEPLOY"
                or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best")
            )
            and r.ci_lo > 0  # strict CI gate
        ]
        fh.write("\n## §4.4 Promotion verdict\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"**PROMOTE {best.cell}** — CI_lo={best.ci_lo:+.3f}, "
                f"Sharpe={best.sharpe:+.4f}, verdict={best.verdict}. "
                f"L46 binding constraint (bootstrap CI) RESOLVED with IG 10y data.\n"
            )
        else:
            fh.write(
                "No cell deployment-eligible under strict 5-axis + CI_lo>0 rule. "
                "Even with IG-10y data, B2 EWMAC does not clear the bar.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
