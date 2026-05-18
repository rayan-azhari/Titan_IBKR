"""I1 v2 -- Multi-feature HMM regime gate on B2e's 11-symbol universe.

Pre-registered in `directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate
2026-05-17.md`.

Architecture:
    - Reuse B2e's 11-symbol IBKR cross-asset universe + 9y common window.
    - Reuse B2 C1-C8 EWMAC cells.
    - Apply I1 v2 panel regime gate (multi-feature HMM + per-asset
      trend-friendly state mapping) BEFORE position sizing.

The gate is applied at audit-harness level (not via EwmacConfig) because
the v2 gate needs the external regime panel which doesn't fit cleanly
into the per-asset EwmacConfig API used by v1.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ewmac/run_i1v2_audit.py
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
    build_positions,
    compute_ewmac_forecast,
)
from research.ewmac.run_b2e_audit import UNIVERSE, _load_close  # noqa: E402
from research.regime.hmm_gate_v2 import (  # noqa: E402
    PanelHMMGateConfig,
    compute_panel_regime_gate,
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
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
PANEL_FP = DATA_DIR / "i1_regime_panel.parquet"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "i1v2_hmm_panel_gate"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# EWMAC C1-C8 baseline (matches B2e). The gate wraps each cell.
_BASE = EwmacConfig(
    speeds=((16, 64), (32, 128), (64, 256)),
    fdm=1.35,
)
EWMAC_CELLS: dict[str, EwmacConfig] = {
    "C1_canonical": _BASE,
    "C5_no_gate_baseline": _BASE,
    "C7_gross_no_costs": replace(_BASE, apply_costs=False),
}

# I1 v2 gate cells. Keys map to (EwmacConfig name, PanelHMMGateConfig).
# None gate = baseline (C5 EWMAC without gate).
GATE_CELLS: dict[str, tuple[str, PanelHMMGateConfig | None]] = {
    "C1_canonical": ("C1_canonical", PanelHMMGateConfig(n_states=2, state_id="mean_return")),
    "C2_3state": ("C1_canonical", PanelHMMGateConfig(n_states=3, state_id="mean_return")),
    "C3_seed_alt": (
        "C1_canonical",
        PanelHMMGateConfig(n_states=2, state_id="mean_return", random_seed=11),
    ),
    "C4_low_vol": ("C1_canonical", PanelHMMGateConfig(n_states=2, state_id="low_vol")),
    "C5_no_gate_baseline": ("C5_no_gate_baseline", None),
    "C6_smoothed": (
        "C1_canonical",
        PanelHMMGateConfig(n_states=2, state_id="mean_return", smoothing_days=5),
    ),
    "C7_gross_no_costs": (
        "C7_gross_no_costs",
        PanelHMMGateConfig(n_states=2, state_id="mean_return"),
    ),
    "C8_combined": (
        "C1_canonical",
        PanelHMMGateConfig(n_states=2, state_id="mean_return", require_broad_trend=True),
    ),
}
CANONICAL_CELL = "C1_canonical"

# Plateau-pre-flight neighbours.
PLATEAU_NEIGHBOURS: dict[str, tuple[str, PanelHMMGateConfig]] = {
    "P_states_3": ("C1_canonical", PanelHMMGateConfig(n_states=3, state_id="mean_return")),
    "P_states_2_smooth5": ("C1_canonical", PanelHMMGateConfig(n_states=2, smoothing_days=5)),
    "P_train_60mo": ("C1_canonical", PanelHMMGateConfig(n_states=2, train_window_bars=60 * 21)),
    "P_train_120mo": ("C1_canonical", PanelHMMGateConfig(n_states=2, train_window_bars=120 * 21)),
}


def load_universe() -> pd.DataFrame:
    parts = [_load_close(sym, fname) for sym, fname in UNIVERSE.items()]
    df = pd.concat(parts, axis=1).sort_index()
    first_valid = df.dropna(how="any").index
    if len(first_valid) == 0:
        raise RuntimeError("No common date range across the 11-symbol universe.")
    df = df.loc[first_valid[0] :].ffill(limit=5)
    return df


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_FP)
    return df.sort_index().dropna(how="any")


def _instrument_vol(closes_df: pd.DataFrame, *, lookback_days: int = 60) -> pd.DataFrame:
    log_ret = np.log(closes_df / closes_df.shift(1))
    return log_ret.rolling(lookback_days, min_periods=lookback_days).std(ddof=1) * np.sqrt(252)


def gated_ewmac_returns(
    closes: pd.DataFrame,
    panel: pd.DataFrame,
    ewmac_cfg: EwmacConfig,
    gate_cfg: PanelHMMGateConfig | None,
    is_end_idx: int,
) -> pd.Series:
    """Compute per-bar portfolio returns with the I1v2 panel gate applied
    at the forecast level. is_end_idx defines the IS slice for the HMM.
    """
    forecast = compute_ewmac_forecast(closes, cfg=ewmac_cfg)
    if gate_cfg is not None:
        gate = compute_panel_regime_gate(
            closes,
            panel,
            cfg=gate_cfg,
            is_end_idx=is_end_idx,
        )
        forecast = forecast * gate
    inst_vol = _instrument_vol(closes, lookback_days=ewmac_cfg.instrument_vol_lookback_days)
    positions = build_positions(forecast, inst_vol, cfg=ewmac_cfg)
    log_ret = np.log(closes / closes.shift(1)).fillna(0.0)
    held_lagged = positions.shift(1).fillna(0.0)
    gross = (held_lagged * log_ret).sum(axis=1)
    if ewmac_cfg.apply_costs:
        dpos = positions.diff().abs().fillna(0.0)
        n_fills = (dpos > 1e-9).sum(axis=1).astype(float)
        bps_drag = (dpos.sum(axis=1) * ewmac_cfg.cost_bps_per_turnover) / 10_000.0
        fixed_drag = (
            n_fills * ewmac_cfg.cost_fixed_usd_per_fill / max(ewmac_cfg.notional_usd_per_leg, 1.0)
        )
        net = gross - bps_drag - fixed_drag
    else:
        net = gross
    return net.rename("ret")


@dataclass
class CellResult:
    cell: str
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
    gate_fraction_mean: float  # H2 non-degeneracy diagnostic
    gate_fraction_per_asset: dict[str, float]


def _stitched_oos(
    closes: pd.DataFrame,
    panel: pd.DataFrame,
    ewmac_cfg: EwmacConfig,
    gate_cfg: PanelHMMGateConfig | None,
    folds: list,
) -> tuple[pd.Series, float, pd.DataFrame]:
    """Run gated EWMAC fold-by-fold (each fold gets its own IS-frozen HMM
    fit using that fold's IS slice) and stitch OOS slices.
    """
    parts = []
    # Cache the IS-bounded gate path per fold for H2 diagnostic.
    last_gate_df = pd.DataFrame()
    for f in folds:
        is_end_idx_full = f.oos_start
        ret = gated_ewmac_returns(closes, panel, ewmac_cfg, gate_cfg, is_end_idx_full)
        parts.append(ret.iloc[f.oos_start : f.oos_end_excl])
        if gate_cfg is not None:
            last_gate_df = compute_panel_regime_gate(
                closes,
                panel,
                cfg=gate_cfg,
                is_end_idx=is_end_idx_full,
            ).iloc[f.oos_start : f.oos_end_excl]
    stitched = pd.concat(parts).fillna(0.0)
    sh = float(sharpe(stitched, periods_per_year=BARS_PER_YEAR["D"]))
    return stitched, sh, last_gate_df


def run_cell(
    cell_name: str,
    ewmac_cfg: EwmacConfig,
    gate_cfg: PanelHMMGateConfig | None,
    closes_v: pd.DataFrame,
    closes_s: pd.DataFrame,
    panel: pd.DataFrame,
    folds: list,
    *,
    n_trials_sweep: int,
    sweep_sharpes: list[float],
    mc_n_workers: int,
) -> CellResult:
    bars_per_year = BARS_PER_YEAR["D"]
    d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    stitched, sh, gate_df = _stitched_oos(closes_v, panel, ewmac_cfg, gate_cfg, folds)
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched,
        periods_per_year=bars_per_year,
        n_resamples=1000,
        seed=42,
    )
    sr_var = sr_var_from_sweep(sweep_sharpes)
    dsr = deflated_sharpe(
        sh,
        sr_var_across_trials=sr_var,
        returns=stitched,
        n_trials=n_trials_sweep,
    )

    def _strategy_fn(df: pd.DataFrame) -> pd.Series:
        # MC bootstrap synthesises asset close paths; gate the synthesised
        # path using the SAME panel (panel is exogenous and not bootstrapped).
        # is_end_idx = len(df) i.e. all-IS for MC paths (no OOS slice).
        u = pd.DataFrame(index=df.index)
        u[closes_v.columns[0]] = df["close"]
        for r in closes_v.columns[1:]:
            if r in df.columns:
                u[r] = df[r]
        return gated_ewmac_returns(u, panel, ewmac_cfg, gate_cfg, is_end_idx=len(u))

    primary = closes_v.iloc[:, 0]
    extras = {r: closes_v[r] for r in closes_v.columns[1:]}
    mc = run_block_mc(
        primary_close=primary,
        cfg=d.mc,
        strategy_fn=_strategy_fn,
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )

    # Sanctuary: fit HMM/gate on visible, apply to visible+sanctuary; pick sanctuary tail.
    full_closes = pd.concat([closes_v, closes_s])
    sanc_ret = gated_ewmac_returns(
        full_closes,
        panel,
        ewmac_cfg,
        gate_cfg,
        is_end_idx=len(closes_v),
    ).iloc[len(closes_v) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    def _noise_fn(df: pd.DataFrame) -> pd.Series:
        return gated_ewmac_returns(df, panel, ewmac_cfg, gate_cfg, is_end_idx=len(df))

    noise = run_noise_robustness(
        closes_v,
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

    # H2 diagnostic: per-asset gate-on fraction across stitched OOS.
    if gate_cfg is None:
        gate_frac_per_asset = {col: 1.0 for col in closes_v.columns}
        gate_frac_mean = 1.0
    else:
        # Re-compute gate over full visible window using max-IS boundary.
        full_gate = compute_panel_regime_gate(
            closes_v,
            panel,
            cfg=gate_cfg,
            is_end_idx=len(closes_v),
        )
        # OOS portion only (skip the last fold's IS-only bars at start of vis).
        oos_start = folds[0].oos_start
        oos_gate = full_gate.iloc[oos_start:]
        gate_frac_per_asset = {col: float(oos_gate[col].mean()) for col in oos_gate.columns}
        gate_frac_mean = float(np.mean(list(gate_frac_per_asset.values())))

    return CellResult(
        cell=cell_name,
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
        gate_fraction_mean=round(gate_frac_mean, 4),
        gate_fraction_per_asset={k: round(v, 3) for k, v in gate_frac_per_asset.items()},
    )


def main():
    print("=" * 80)
    print("I1 v2 -- Multi-feature HMM regime gate on B2e 11-symbol universe")
    print("Pre-reg: directives/Pre-Reg I1v2 Multi-Feature HMM Regime Gate 2026-05-17.md")
    print("=" * 80)
    closes = load_universe()
    panel = load_panel()
    print(
        f"\nUniverse: {len(closes.columns)} symbols, "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)"
    )
    print(
        f"Panel: {len(panel.columns)} features, "
        f"{panel.index[0].date()} -> {panel.index[-1].date()} ({len(panel)} bars)"
    )

    sanc = slice_sanctuary(closes, months=12)
    closes_v = sanc.visible
    closes_s = sanc.sanctuary
    print(f"Visible: {len(closes_v)}  Sanctuary: {len(closes_s)}")

    bars_per_year = BARS_PER_YEAR["D"]
    d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(closes_v.index, d.wfo, bars_per_year=bars_per_year)
    print(f"WFO: {d.wfo.is_min_years}y IS / {d.wfo.oos_years}y OOS -> {len(folds)} folds")

    # Plateau pre-flight on canonical + 4 neighbours.
    print("\n[Plateau pre-flight] C1 + 4 neighbours (OOS Sharpes)...")
    plateau_sharpes: dict[str, float] = {}
    _, sh_c1, _ = _stitched_oos(
        closes_v,
        panel,
        EWMAC_CELLS["C1_canonical"],
        GATE_CELLS["C1_canonical"][1],
        folds,
    )
    plateau_sharpes["C1_canonical"] = sh_c1
    for name, (ewmac_name, gate_cfg) in PLATEAU_NEIGHBOURS.items():
        _, sh, _ = _stitched_oos(
            closes_v,
            panel,
            EWMAC_CELLS[ewmac_name],
            gate_cfg,
            folds,
        )
        plateau_sharpes[name] = sh
    for n, s in plateau_sharpes.items():
        print(f"  {n}: Sharpe={s:+.4f}")
    vals = list(plateau_sharpes.values())
    mx, mn = max(vals), min(vals)
    rel_spread = abs(mx - mn) / max(abs(mx), 1e-9)
    plateau_passed = rel_spread <= 0.30
    print(
        f"  Relative spread: {rel_spread:.2%}  ({'PASSED' if plateau_passed else 'FAILED'} <=30%)"
    )

    # Pass 1: headline OOS Sharpes for DSR variance.
    print("\n[Pass 1/2] Headline OOS Sharpes for DSR sweep variance...")
    sweep_sharpes: list[float] = []
    for name, (ewmac_name, gate_cfg) in GATE_CELLS.items():
        _, sh, _ = _stitched_oos(
            closes_v,
            panel,
            EWMAC_CELLS[ewmac_name],
            gate_cfg,
            folds,
        )
        sweep_sharpes.append(sh)
        print(f"  {name}: Sharpe={sh:+.4f}")

    # Pass 2: full per-cell.
    print(f"\n[Pass 2/2] Full per-cell audit -- MC parallel x{DEFAULT_MC_WORKERS}...")
    results: list[CellResult] = []
    for name, (ewmac_name, gate_cfg) in GATE_CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            EWMAC_CELLS[ewmac_name],
            gate_cfg,
            closes_v,
            closes_s,
            panel,
            folds,
            n_trials_sweep=len(GATE_CELLS),
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
            f"    Noise: base={r.noise_base:+.4f} mean={r.noise_passes_mean} worst={r.noise_passes_worst} axis={r.noise_axis}"
        )
        print(f"    Gate frac (mean across assets): {r.gate_fraction_mean:.3f}")
        print(f"    Verdict (5-axis): {r.verdict}")

    # H2 non-degeneracy check on canonical.
    c1 = next(r for r in results if r.cell == "C1_canonical")
    n_assets_in_band = sum(1 for v in c1.gate_fraction_per_asset.values() if 0.20 <= v <= 0.95)
    h2_passed = n_assets_in_band >= 8
    print("\n[H2 non-degeneracy] C1 gate fractions per asset:")
    for asset, frac in c1.gate_fraction_per_asset.items():
        print(f"  {asset:>5s}: {frac:.3f}")
    print(
        f"  {n_assets_in_band}/11 assets with gate fraction in [0.20, 0.95] -- H2 {'PASS' if h2_passed else 'FAIL'}"
    )

    # H3 vs B2e C1 (+0.49)
    h3_passed = c1.sharpe > 0.49
    print(
        f"\n[H3] C1 Sharpe {c1.sharpe:+.4f} vs B2e C1 (+0.4863) -- {'PASS' if h3_passed else 'FAIL'}"
    )

    # H5 seed-stability: C3 vs C1
    c3 = next(r for r in results if r.cell == "C3_seed_alt")
    h5_diff = abs(c3.sharpe - c1.sharpe) / max(abs(c1.sharpe), 1e-9)
    h5_passed = h5_diff <= 0.15
    print(f"[H5] C3 vs C1 Sharpe gap: {h5_diff:.2%} -- {'PASS' if h5_passed else 'FAIL'} (<=15%)")

    # H6 noise gate on C1
    h6_passed = c1.noise_axis == "best"
    print(
        f"[H6] C1 noise axis: {c1.noise_axis} -- {'PASS' if h6_passed else 'FAIL'} (must be best)"
    )

    # Write result log.
    report_fp = REPORTS_DIR / "result_log.md"
    with report_fp.open("w", encoding="utf-8") as fh:
        fh.write("# I1 v2 Audit Result Log\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write("**Universe:** 11-symbol IBKR cross-asset (B2e universe)\n")
        fh.write(
            f"**Visible:** {len(closes_v)}  Sanctuary: {len(closes_s)}  Folds: {len(folds)}\n\n"
        )
        fh.write("## §4.1 Plateau pre-flight\n\n")
        fh.write(
            f"Relative spread: {rel_spread:.2%}  {'PASSED' if plateau_passed else 'FAILED'}\n\n"
        )
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
        fh.write("\n## §4.2 Per-cell 5-axis matrix\n\n")
        fh.write(
            "| Cell | Sharpe | CI95 lo | CI95 hi | DSR | MC P | Sanc Sharpe | Noise base | Noise axis | Gate frac | Verdict |\n"
        )
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|\n")
        for r in results:
            fh.write(
                f"| {r.cell} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
                f"{r.dsr_prob:.4f} | {r.mc_p_maxdd_gt_threshold:.4f} | "
                f"{r.sanctuary_sharpe:+.4f} | {r.noise_base:+.4f} | "
                f"{r.noise_axis} | {r.gate_fraction_mean:.3f} | {r.verdict} |\n"
            )
        fh.write("\n## §4.3 Falsification hypothesis checklist\n\n")
        fh.write(
            f"- H1 plateau spread <=30%: **{'PASS' if plateau_passed else 'FAIL'}** ({rel_spread:.2%})\n"
        )
        fh.write(
            f"- H2 gate non-degenerate (>=8/11 assets in [0.20, 0.95]): **{'PASS' if h2_passed else 'FAIL'}** ({n_assets_in_band}/11)\n"
        )
        fh.write(
            f"- H3 C1 Sharpe > B2e (+0.49): **{'PASS' if h3_passed else 'FAIL'}** ({c1.sharpe:+.4f})\n"
        )
        fh.write(
            f"- H5 seed-stable (gap <=15%): **{'PASS' if h5_passed else 'FAIL'}** ({h5_diff:.2%})\n"
        )
        fh.write(
            f"- H6 noise axis = best: **{'PASS' if h6_passed else 'FAIL'}** ({c1.noise_axis})\n"
        )
        fh.write("\n## §4.4 Per-asset gate fractions (C1)\n\n")
        for a, f in c1.gate_fraction_per_asset.items():
            fh.write(f"- {a}: {f:.3f}\n")
        fh.write("\n## §4.5 B2e vs I1 v2 comparison\n\n")
        fh.write("| Metric | B2e C1 | I1v2 C1 |\n|---|---:|---:|\n")
        fh.write(f"| Sharpe | +0.4863 | {c1.sharpe:+.4f} |\n")
        fh.write(f"| CI95 lo | -0.008 | {c1.ci_lo:+.3f} |\n")
        fh.write(f"| Sanc Sharpe | +1.0934 | {c1.sanctuary_sharpe:+.4f} |\n")
        fh.write(f"| Noise axis | mid | {c1.noise_axis} |\n")
        fh.write(f"| Verdict | TIER_UNCONFIRMED | {c1.verdict} |\n")
    print(f"\nResult log: {report_fp}")
    return results


if __name__ == "__main__":
    main()
