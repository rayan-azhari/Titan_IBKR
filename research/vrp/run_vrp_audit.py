"""VRP Capture -- framework-grade audit harness (E1).

Specified by ``directives/Pre-Reg E1 VRP Capture 2026-05-15.md`` §3.

Run via::

    uv run python research/vrp/run_vrp_audit.py

The harness:
    1. Snapshots VIXY / VIX / VIX9D / VIX3M / SPY parquets (L11).
    2. Slices the sanctuary (last 12 months).
    3. For each pre-committed cell, runs a rolling WFO on the visible
       window, stitches per-fold OOS returns, runs the Varma noise-
       injection gate, and applies the 5-axis decision matrix (J3 / L24).
    4. Writes a result log appendix that can be pasted into the §4
       section of the pre-reg directive.

V3.6 discipline applied throughout:
    * Strict ``shift(1)`` on weights (causality enforced in
      ``vrp_strategy.vrp_target_weights``).
    * Per-day MTM Sharpe convention (class default; L06).
    * DAILY_MEAN_REVERSION-specific MC gate.
    * Pre-reg directive committed BEFORE data was examined (V3.1).
    * DSR applied at N=7 trials with actual skew/kurt.
    * Noise gate per cell -> 5th axis (L24).
    * Sanctuary divergence test reported alongside sanctuary Sharpe (L15).
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

from research.vrp.vrp_strategy import (  # noqa: E402
    VrpConfig,
    vrp_assert_causal,
    vrp_returns,
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
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "vrp"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cost calibration (US ETF; L23). VIXY is liquid (~$10M ADV) -- spread+slip
# ~3 bps/side; commission floor $1 IBKR Pro.
COST_BPS_PER_TURNOVER = 6.0
COST_FIXED_USD_PER_FILL = 1.0
COST_NOTIONAL_USD = 30_000.0
COST_REBALANCE_THRESHOLD = 0.05


def _cost_kwargs() -> dict:
    return dict(
        cost_bps_per_turnover=COST_BPS_PER_TURNOVER,
        cost_fixed_usd_per_fill=COST_FIXED_USD_PER_FILL,
        notional_usd=COST_NOTIONAL_USD,
        rebalance_threshold=COST_REBALANCE_THRESHOLD,
    )


# Pre-registered cells (V3.1 frozen at directive commit).
CELLS: dict[str, VrpConfig] = {
    "C1_canonical": VrpConfig(),
    "C2_looser_long": VrpConfig(ratio_long_gate=1.02),
    "C3_tighter_long": VrpConfig(ratio_long_gate=1.10),
    "C4_smaller_short": VrpConfig(target_short_weight=-0.25),
    "C5_no_buffer": VrpConfig(regime_buffer_pct=0.0),
    "C6_spy_overlay": VrpConfig(defensive_long_spy_w=0.50),
    "C7_with_short_signal": VrpConfig(ratio_short_gate=1.02),
}
CANONICAL_CELL = "C1_canonical"


def _load_close(sym: str, required: bool = True) -> pd.Series | None:
    path = DATA_DIR / f"{sym}_D.parquet"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required data file: {path}.")
        return None
    df = pd.read_parquet(path)
    if "close" in df.columns:
        s = df["close"]
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        raise ValueError(f"{path}: no 'close' or 'Close' column.")
    s.name = sym
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def load_universe() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load VIXY + SPY closes, VIX / VIX9D / VIX3M signal series.

    Returns (closes_df, vix, vix9d, vix3m). closes_df is the strategy's
    implementation-vehicle data; vix/vix9d/vix3m are the signal inputs.

    All indexes normalised to date-only (L20).
    """
    vixy = _load_close("VIXY", required=True)
    spy = _load_close("SPY", required=True)
    closes = pd.concat([vixy, spy], axis=1).dropna(how="any")
    vix = _load_close("VIX", required=True)
    vix9d = _load_close("VIX9D", required=True)
    vix3m = _load_close("VIX3M", required=True)
    # Align on dates present in all five series.
    common_idx = (
        closes.index.intersection(vix.index).intersection(vix9d.index).intersection(vix3m.index)
    )
    closes = closes.reindex(common_idx)
    vix = vix.reindex(common_idx)
    vix9d = vix9d.reindex(common_idx)
    vix3m = vix3m.reindex(common_idx)
    return closes, vix, vix9d, vix3m


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
    mc_pass_prob: float
    sanctuary_sharpe: float
    sanctuary_percentile: float
    sanctuary_lucky_flag: bool
    sanctuary_unlucky_flag: bool
    noise_base_sharpe: float
    noise_passes_mean: bool
    noise_passes_worst: bool
    noise_axis: str
    verdict: str
    verdict_rationale: str


def _strategy_fn_for_cell(
    cfg: VrpConfig,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
):
    """Build a closure compatible with run_block_mc / run_noise_robustness.

    Both framework primitives pass a DataFrame containing 'close' (the
    primary series — here VIXY) plus extras dict keys. The strategy
    needs VIXY + SPY in the closes frame; the VIX-family series are
    passed in as closures (NOT bootstrapped — bootstrapping VIX with VIXY
    would destroy the cointegrated structure that drives the signal).
    """

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        universe_df = pd.DataFrame(index=df.index)
        universe_df["VIXY"] = df["close"]  # primary in MC framework
        if "SPY" in df.columns:
            universe_df["SPY"] = df["SPY"]
        else:
            universe_df["SPY"] = df["close"]  # fallback; weight stays 0
        return vrp_returns(
            universe_df,
            cfg=cfg,
            vix=vix,
            vix9d=vix9d,
            vix3m=vix3m,
            **_cost_kwargs(),
        )

    return strategy_fn


def run_cell(
    cell_name: str,
    cfg: VrpConfig,
    visible: pd.DataFrame,
    sanctuary: pd.DataFrame,
    *,
    vix: pd.Series,
    vix9d: pd.Series,
    vix3m: pd.Series,
    n_trials_sweep: int,
    bars_per_year: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
    noise_cfg: NoiseConfig | None = None,
) -> tuple[CellResult, pd.Series, pd.Series]:
    cls = StrategyClass.DAILY_MEAN_REVERSION
    d = defaults_for(cls)

    folds = build_folds(visible.index, d.wfo, bars_per_year=bars_per_year)
    if not folds:
        raise RuntimeError(f"{cell_name}: no folds constructed (visible too short)")

    # Strategy has no IS-trained parameters (cells pre-registered). Compute
    # returns ONCE on the full visible window then slice per fold.
    visible_strategy_returns = vrp_returns(
        visible, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs()
    )
    stitched_parts = [visible_strategy_returns.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(stitched_parts).fillna(0.0)

    sh = float(sharpe(stitched, periods_per_year=bars_per_year))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=bars_per_year, n_resamples=1000, seed=42
    )

    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh, sr_var_across_trials=sr_var, returns=stitched, n_trials=n_trials_sweep
    )

    # MC: bootstrap VIXY (the implementation vehicle) with SPY as extra.
    # Per the pre-reg, the VIX-family signals are NOT bootstrapped --
    # they're conditioning data passed via the closure. Strategy's own MC
    # primitive is the absolute gate; relative MC is not appropriate
    # because this is NOT a long-only equity strategy (L17 inverts here).
    primary = visible["VIXY"]
    extras = {"SPY": visible["SPY"]}
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, vix, vix9d, vix3m),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    # Sanctuary: re-run on full series, take only the sanctuary slice.
    full = pd.concat([visible, sanctuary])
    full_rets = vrp_returns(full, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs())
    sanc_ret = full_rets.iloc[len(visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    # Varma noise-injection robustness gate (J3 5th axis).
    _noise_cfg = noise_cfg or NoiseConfig(
        noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30
    )

    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        return vrp_returns(df, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs())

    noise_result = run_noise_robustness(
        visible, _noise_fn, periods_per_year=bars_per_year, cfg=_noise_cfg
    )

    inputs = DecisionInputs(
        ci_lo=ci_lo,
        dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
        pass_threshold_prob=mc.pass_threshold_prob,
        sanctuary_sharpe=sanc_sh,
        noise_passes_mean=noise_result.passes,
        noise_passes_worst=noise_result.worst_case_passes,
    )
    decision = decide(inputs)

    cell_result = CellResult(
        cell=cell_name,
        n_oos_bars=len(stitched),
        n_folds=len(folds),
        sharpe=round(sh, 4),
        ci_lo=round(ci_lo, 4),
        ci_hi=round(ci_hi, 4),
        dsr_prob=round(dsr.dsr_prob, 4),
        mc_p_maxdd_gt_threshold=round(mc.p_maxdd_gt_threshold, 4),
        mc_threshold_pct=round(mc.threshold_pct, 4),
        mc_pass_prob=round(mc.pass_threshold_prob, 4),
        sanctuary_sharpe=round(sanc_sh, 4),
        sanctuary_percentile=round(div.percentile, 4)
        if np.isfinite(div.percentile)
        else float("nan"),
        sanctuary_lucky_flag=div.lucky_flag,
        sanctuary_unlucky_flag=div.unlucky_flag,
        noise_base_sharpe=round(noise_result.base_sharpe, 4),
        noise_passes_mean=noise_result.passes,
        noise_passes_worst=noise_result.worst_case_passes,
        noise_axis=decision.noise_axis,
        verdict=decision.verdict.value,
        verdict_rationale=decision.rationale,
    )
    return (cell_result, stitched, sanc_ret)


def main():
    print("=" * 80)
    print("VRP CAPTURE -- Framework Audit (E1)")
    print("Pre-reg: directives/Pre-Reg E1 VRP Capture 2026-05-15.md")
    print("=" * 80)

    closes, vix, vix9d, vix3m = load_universe()
    print(
        f"\nData loaded: {closes.shape[0]} bars "
        f"{closes.index[0].date()} -> {closes.index[-1].date()}"
    )
    print(f"Universe: {list(closes.columns)}  Signals: VIX, VIX9D, VIX3M")

    # Causality smoke test (A10).
    try:
        vrp_assert_causal(closes, vix=vix, vix9d=vix9d, vix3m=vix3m, n_trials=5, seed=42)
        print("Causality smoke test: PASSED")
    except AssertionError as e:
        print(f"Causality smoke test: FAILED - {e}")
        return

    sanc = slice_sanctuary(closes, months=12)
    visible = sanc.visible
    sanctuary = sanc.sanctuary
    print(f"\nSanctuary: {sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()}")
    print(f"  Visible: {len(visible)} bars  |  Sanctuary: {len(sanctuary)} bars")

    bars_per_year = BARS_PER_YEAR["D"]

    # Pass 1: headline Sharpes for DSR variance.
    print(f"\n[Pass 1/2] Headline Sharpes for {len(CELLS)} cells...")
    cls_defaults = defaults_for(StrategyClass.DAILY_MEAN_REVERSION)
    folds = build_folds(visible.index, cls_defaults.wfo, bars_per_year=bars_per_year)
    sweep_sharpes: list[float] = []
    for cell_name, cfg in CELLS.items():
        rets = vrp_returns(visible, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs())
        stitched_parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
        stitched = pd.concat(stitched_parts).fillna(0.0)
        sh = float(sharpe(stitched, periods_per_year=bars_per_year))
        sweep_sharpes.append(sh)
        print(f"  {cell_name}: Sharpe={sh:+.4f}  oos_bars={len(stitched)}")

    # Pass 2: full audit per cell.
    from titan.research.framework.mc import DEFAULT_MC_WORKERS

    print(
        f"\n[Pass 2/2] Full per-cell audit (DSR + MC + sanctuary + noise + decide) "
        f"-- MC paths parallel x{DEFAULT_MC_WORKERS}..."
    )
    results: list[CellResult] = []
    for cell_name, cfg in CELLS.items():
        print(f"\n  > {cell_name}...")
        result, _, _ = run_cell(
            cell_name,
            cfg,
            visible,
            sanctuary,
            vix=vix,
            vix9d=vix9d,
            vix3m=vix3m,
            n_trials_sweep=len(CELLS),
            bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes,
            mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(result)
        print(f"    Sharpe={result.sharpe:+.4f}  CI=[{result.ci_lo:+.3f}, {result.ci_hi:+.3f}]")
        print(
            f"    DSR={result.dsr_prob:.4f}  "
            f"MC P(MaxDD>{result.mc_threshold_pct * 100:.0f}%)={result.mc_p_maxdd_gt_threshold:.4f}"
        )
        print(f"    Sanc Sharpe={result.sanctuary_sharpe:+.4f}  pct={result.sanctuary_percentile}")
        print(
            f"    Noise (Varma): base={result.noise_base_sharpe:+.4f} "
            f"mean_pass={result.noise_passes_mean} worst_pass={result.noise_passes_worst} "
            f"axis={result.noise_axis}"
        )
        print(f"    Verdict (5-axis): {result.verdict}")
        print(f"    Rationale: {result.verdict_rationale}")

    # Write result log.
    report_path = REPORTS_DIR / "result_log.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# VRP Capture Audit Result Log (E1)\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()}\n")
        fh.write(f"**Visible bars:** {len(visible)}  |  **Sanctuary bars:** {len(sanctuary)}\n")
        fh.write(f"**Universe:** {list(closes.columns)}\n\n")

        fh.write("## §4.1 Per-cell verdicts (5-axis, J3 / L24)\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR-prob | MC P(>25%) | "
            "Sanc Sharpe | Sanc pct | Noise base | Noise axis | Verdict |\n"
        )
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|\n")
        for r in results:
            fh.write(
                f"| {r.cell} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
                f"{r.dsr_prob:.4f} | {r.mc_p_maxdd_gt_threshold:.4f} | "
                f"{r.sanctuary_sharpe:+.4f} | {r.sanctuary_percentile} | "
                f"{r.noise_base_sharpe:+.4f} | {r.noise_axis} | {r.verdict} |\n"
            )
        fh.write("\n_Used unsigned variable `r` for compactness; rows preserve insertion order._\n")

    print(f"\nResult log written to: {report_path}")
    return results


if __name__ == "__main__":
    main()
