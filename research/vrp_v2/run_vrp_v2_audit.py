"""VRP v2 -- framework-grade audit harness (E1b).

Specified by ``directives/Pre-Reg E1b VRP Capture v2 Percentile Gates 2026-05-15.md``.
Class: ``DAILY_MEAN_REVERSION_VOL_CARRY`` (new sub-class added in this PR).

Run via::

    uv run python research/vrp_v2/run_vrp_v2_audit.py

Pipeline mirrors `research/vrp/run_vrp_audit.py` (E1) with two changes:
    1. defaults_for(DAILY_MEAN_REVERSION_VOL_CARRY) instead of
       defaults_for(DAILY_MEAN_REVERSION) -> MC threshold P(>50%) < 10%
       instead of P(>25%) < 10% (L25 mitigation).
    2. Per-cell strategy uses percentile-rolling gates (or sigmoid C7)
       instead of bare-threshold gates (L26 mitigation).

Plateau pre-flight (pre-reg §3): before main audit, the canonical cell's
±1 step on (window_d, enter_q, exit_q) is checked. If Sharpe spread > 30%,
abort with a clear error.
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

from research.vrp_v2.vrp_v2_strategy import (  # noqa: E402
    VrpV2Config,
    vrp_v2_assert_causal,
    vrp_v2_returns,
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "vrp_v2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cost calibration (US ETF; L23) — identical to E1 for apples-to-apples.
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


# Pre-registered cells (V3.1, frozen at directive commit).
CELLS: dict[str, VrpV2Config] = {
    "C1_canonical": VrpV2Config(gate_kind="percentile", window_d=252, enter_q=0.60, exit_q=0.40),
    "C2_wider_band": VrpV2Config(gate_kind="percentile", window_d=252, enter_q=0.70, exit_q=0.30),
    "C3_narrower_band": VrpV2Config(
        gate_kind="percentile", window_d=252, enter_q=0.55, exit_q=0.45
    ),
    "C4_shorter_window": VrpV2Config(
        gate_kind="percentile", window_d=126, enter_q=0.60, exit_q=0.40
    ),
    "C5_longer_window": VrpV2Config(
        gate_kind="percentile", window_d=504, enter_q=0.60, exit_q=0.40
    ),
    "C6_smaller_short": VrpV2Config(
        gate_kind="percentile", window_d=252, enter_q=0.60, exit_q=0.40, target_short_weight=-0.25
    ),
    "C7_continuous_sigmoid": VrpV2Config(
        gate_kind="sigmoid", window_d=252, enter_q=0.60, exit_q=0.40, sigmoid_scale=0.05
    ),
}
CANONICAL_CELL = "C1_canonical"

# Plateau pre-flight neighbours of C1: ±1 step in each of window_d, enter_q, exit_q.
# Pre-committed at directive commit.
PLATEAU_NEIGHBOURS: dict[str, VrpV2Config] = {
    "P_window_short": VrpV2Config(window_d=189, enter_q=0.60, exit_q=0.40),  # midway to C4
    "P_window_long": VrpV2Config(window_d=378, enter_q=0.60, exit_q=0.40),  # midway to C5
    "P_enter_lower": VrpV2Config(window_d=252, enter_q=0.55, exit_q=0.40),
    "P_enter_higher": VrpV2Config(window_d=252, enter_q=0.65, exit_q=0.40),
    "P_exit_lower": VrpV2Config(window_d=252, enter_q=0.60, exit_q=0.35),
    "P_exit_higher": VrpV2Config(window_d=252, enter_q=0.60, exit_q=0.45),
}


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
    vixy = _load_close("VIXY", required=True)
    closes = pd.DataFrame({"VIXY": vixy}).dropna(how="any")
    vix = _load_close("VIX", required=True)
    vix9d = _load_close("VIX9D", required=True)
    vix3m = _load_close("VIX3M", required=True)
    common_idx = (
        closes.index.intersection(vix.index).intersection(vix9d.index).intersection(vix3m.index)
    )
    closes = closes.reindex(common_idx)
    return closes, vix.reindex(common_idx), vix9d.reindex(common_idx), vix3m.reindex(common_idx)


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


def _strategy_fn_for_cell(cfg: VrpV2Config, vix, vix9d, vix3m):
    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        universe_df = pd.DataFrame(index=df.index)
        universe_df["VIXY"] = df["close"]
        return vrp_v2_returns(
            universe_df, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs()
        )

    return strategy_fn


def _stitched_oos_sharpe(
    visible: pd.DataFrame,
    cfg: VrpV2Config,
    folds,
    vix,
    vix9d,
    vix3m,
    bars_per_year: int,
) -> float:
    rets = vrp_v2_returns(visible, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs())
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year))


def plateau_pre_flight(
    visible: pd.DataFrame, folds, vix, vix9d, vix3m, bars_per_year: int
) -> tuple[float, dict[str, float]]:
    """Run canonical + 6 neighbours; return (relative_spread, per-cell-sharpes).

    Per the pre-reg §3: if relative_spread > 0.30, the audit aborts —
    the design has a parameter-spike risk before we even start the
    full MC + noise sweep.
    """
    canonical_sh = _stitched_oos_sharpe(
        visible, CELLS[CANONICAL_CELL], folds, vix, vix9d, vix3m, bars_per_year
    )
    neighbour_sharpes: dict[str, float] = {CANONICAL_CELL: canonical_sh}
    for name, cfg in PLATEAU_NEIGHBOURS.items():
        sh = _stitched_oos_sharpe(visible, cfg, folds, vix, vix9d, vix3m, bars_per_year)
        neighbour_sharpes[name] = sh

    all_sh = list(neighbour_sharpes.values())
    max_sh = max(all_sh)
    min_sh = min(all_sh)
    rel_spread = abs(max_sh - min_sh) / max(abs(max_sh), 1e-9)
    return rel_spread, neighbour_sharpes


def run_cell(
    cell_name: str,
    cfg: VrpV2Config,
    visible: pd.DataFrame,
    sanctuary: pd.DataFrame,
    *,
    vix,
    vix9d,
    vix3m,
    n_trials_sweep: int,
    bars_per_year: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
    noise_cfg: NoiseConfig | None = None,
) -> CellResult:
    cls = StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY
    d = defaults_for(cls)

    folds = build_folds(visible.index, d.wfo, bars_per_year=bars_per_year)
    if not folds:
        raise RuntimeError(f"{cell_name}: no folds (visible too short)")

    visible_strategy_returns = vrp_v2_returns(
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

    # MC: bootstrap VIXY paths only (extras are empty — no SPY overlay in v2).
    primary = visible["VIXY"]
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn_for_cell(cfg, vix, vix9d, vix3m),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=None,
        n_workers=mc_n_workers,
    )

    # Sanctuary.
    full = pd.concat([visible, sanctuary])
    full_rets = vrp_v2_returns(full, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs())
    sanc_ret = full_rets.iloc[len(visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched, sanctuary_returns=sanc_ret, periods_per_year=bars_per_year
    )

    # Noise gate (J3 5th axis).
    _noise_cfg = noise_cfg or NoiseConfig(
        noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30
    )

    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        return vrp_v2_returns(df, cfg=cfg, vix=vix, vix9d=vix9d, vix3m=vix3m, **_cost_kwargs())

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


def main():
    print("=" * 80)
    print("VRP v2 -- Framework Audit (E1b)")
    print("Pre-reg: directives/Pre-Reg E1b VRP Capture v2 Percentile Gates 2026-05-15.md")
    print("Class: DAILY_MEAN_REVERSION_VOL_CARRY (new sub-class; L25)")
    print("=" * 80)

    closes, vix, vix9d, vix3m = load_universe()
    print(
        f"\nData loaded: {closes.shape[0]} bars "
        f"{closes.index[0].date()} -> {closes.index[-1].date()}"
    )

    vrp_v2_assert_causal(closes, vix=vix, vix9d=vix9d, vix3m=vix3m, n_trials=5, seed=42)
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=12)
    visible = sanc.visible
    sanctuary = sanc.sanctuary
    print(f"\nSanctuary: {sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()}")
    print(f"  Visible: {len(visible)} bars  Sanctuary: {len(sanctuary)} bars")

    bars_per_year = BARS_PER_YEAR["D"]
    cls_defaults = defaults_for(StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY)
    print(
        f"\nClass MC default: P(MaxDD > {cls_defaults.mc.max_dd_threshold_pct * 100:.0f}%) "
        f"< {cls_defaults.mc.max_dd_pass_prob * 100:.0f}%"
    )

    folds = build_folds(visible.index, cls_defaults.wfo, bars_per_year=bars_per_year)
    print(f"WFO: {len(folds)} folds")

    # ── Plateau pre-flight (V3.2 / pre-reg §3) ───────────────────────────
    print("\n[Plateau pre-flight] Canonical + 6 grid neighbours...")
    rel_spread, plateau_sharpes = plateau_pre_flight(
        visible, folds, vix, vix9d, vix3m, bars_per_year
    )
    for name, s in plateau_sharpes.items():
        print(f"  {name}: Sharpe={s:+.4f}")
    print(f"  Relative spread: {rel_spread:.2%}  (V3.2 gate: < 30%)")
    if rel_spread > 0.30:
        print(
            f"\n  [!] PLATEAU GATE FAILED: spread={rel_spread:.2%} > 30%. "
            f"Design lives at a parameter spike, not a plateau. ABORTING audit per pre-reg §3."
        )
        # Still write a partial result log so the negative result is captured.
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# VRP v2 Audit Result Log (E1b) -- ABORTED at plateau pre-flight\n\n")
            fh.write(f"**Relative Sharpe spread:** {rel_spread:.2%} (gate: 30%)\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
            fh.write(
                "\nThe canonical cell's ±1-step neighbourhood does not form a plateau. "
                "Per the pre-reg, the full audit is not run.\n"
            )
        return None

    print("  [OK] Plateau gate passed — proceeding to full audit.")

    # ── Pass 1: headline Sharpes ─────────────────────────────────────────
    print(f"\n[Pass 1/2] Headline Sharpes for {len(CELLS)} cells...")
    sweep_sharpes: list[float] = []
    for name, cfg in CELLS.items():
        sh = _stitched_oos_sharpe(visible, cfg, folds, vix, vix9d, vix3m, bars_per_year)
        sweep_sharpes.append(sh)
        print(f"  {name}: Sharpe={sh:+.4f}")

    # ── Pass 2: full per-cell audit ───────────────────────────────────────
    from titan.research.framework.mc import DEFAULT_MC_WORKERS

    print(
        f"\n[Pass 2/2] Full per-cell audit (DSR + MC + sanctuary + noise + decide) "
        f"-- MC parallel x{DEFAULT_MC_WORKERS}..."
    )
    results: list[CellResult] = []
    for name, cfg in CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
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
        results.append(r)
        print(f"    Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        print(
            f"    DSR={r.dsr_prob:.4f}  "
            f"MC P(>{r.mc_threshold_pct * 100:.0f}%)={r.mc_p_maxdd_gt_threshold:.4f}"
        )
        print(f"    Sanc Sharpe={r.sanctuary_sharpe:+.4f}  pct={r.sanctuary_percentile}")
        print(
            f"    Noise: base={r.noise_base_sharpe:+.4f} "
            f"mean_pass={r.noise_passes_mean} worst_pass={r.noise_passes_worst} axis={r.noise_axis}"
        )
        print(f"    Verdict (5-axis): {r.verdict}")
        print(f"    Rationale: {r.verdict_rationale}")

    # ── Result log ───────────────────────────────────────────────────────
    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# VRP v2 Audit Result Log (E1b)\n\n")
        fh.write("**Run date:** 2026-05-15\n")
        fh.write(f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()}\n")
        fh.write(f"**Visible bars:** {len(visible)}  Sanctuary bars: {len(sanctuary)}\n")
        fh.write("**Class:** DAILY_MEAN_REVERSION_VOL_CARRY\n")
        fh.write(
            f"**MC default:** P(MaxDD > {cls_defaults.mc.max_dd_threshold_pct * 100:.0f}%) "
            f"< {cls_defaults.mc.max_dd_pass_prob * 100:.0f}%\n\n"
        )

        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(f"**Relative spread:** {rel_spread:.2%} (gate: 30%) -- PASSED\n\n")
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.2 Per-cell verdicts (5-axis, J3 / L24)\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR-prob | MC P(>50%) | "
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

    print(f"\nResult log written: {report}")
    return results


if __name__ == "__main__":
    main()
