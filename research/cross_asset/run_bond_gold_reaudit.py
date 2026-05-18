"""bond_gold V1-era re-audit harness.

Specified by ``directives/Pre-Reg bond_gold Re-audit 2026-05-16.md``.

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/cross_asset/run_bond_gold_reaudit.py

Pipeline (V3.6 + L52 hybrid):

1. Load IEF + GLD daily closes (common index).
2. ``slice_sanctuary(months=24)`` — visible vs sanctuary.
3. Causality smoke (``bond_gold_assert_causal``).
4. Build WFO folds on visible (CROSS_ASSET_MOMENTUM defaults).
5. **Pass 1** — for each cell, stitched-OOS Sharpe + bootstrap CI.
6. **L52 plateau pre-flight** — relative spread across (C1, P_*) cells.
7. **L53 early gate** — if no cell can plausibly clear CI_lo > 0, skip Pass 2.
8. **Pass 2** — MC + DSR + sanctuary + noise + ``decide()`` per cell.
9. Write §4 result log to ``.tmp/reports/bond_gold_reaudit/result_log.md``.
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

from research.cross_asset.bond_gold_strategy import (  # noqa: E402
    BondGoldConfig,
    bond_gold_assert_causal,
    bond_gold_returns,
)
from titan.research.framework import (  # noqa: E402
    DecisionInputs,
    NoiseConfig,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    pass1_can_clear_any_cell,
    run_block_mc,
    run_noise_robustness,
    sanctuary_divergence_test,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "bond_gold_reaudit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 24

# §2 frozen cell grid.
CELLS: dict[str, BondGoldConfig] = {
    "C1_canonical": BondGoldConfig(lookback=120, threshold=0.50),
    "P_low_threshold": BondGoldConfig(lookback=120, threshold=0.00),
    "P_quarter": BondGoldConfig(lookback=120, threshold=0.25),
    "P_high_threshold": BondGoldConfig(lookback=120, threshold=0.75),
    "P_strict": BondGoldConfig(lookback=120, threshold=1.00),
    "C2_live_canonical": BondGoldConfig(lookback=60, threshold=0.50),
    "C3_no_threshold": BondGoldConfig(lookback=120, threshold=0.00),
    "C4_gross_no_costs": BondGoldConfig(lookback=120, threshold=0.50, apply_costs=False),
}
CANONICAL_CELL = "C1_canonical"
PLATEAU_CELLS = ("C1_canonical", "P_low_threshold", "P_quarter", "P_high_threshold", "P_strict")
EXCLUDED_FROM_PROMOTION = ("C2_live_canonical", "C3_no_threshold", "C4_gross_no_costs")


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe() -> pd.DataFrame:
    ief = _load_close("IEF")
    gld = _load_close("GLD")
    common = ief.index.intersection(gld.index)
    return pd.DataFrame({"IEF": ief.reindex(common), "GLD": gld.reindex(common)}).dropna()


@dataclass
class CellRow:
    name: str
    cfg: BondGoldConfig
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


def _stitched_oos_returns(full_returns: pd.Series, folds: list) -> pd.Series:
    """Concatenate OOS slices across folds, deduplicating overlap by 'last-wins'."""
    parts: list[pd.Series] = []
    for f in folds:
        oos_slice = full_returns.iloc[f.oos_start : f.oos_end_excl]
        parts.append(oos_slice)
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    # rolling/overlapping folds may double-count bars; keep the LATEST estimate.
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    return stitched.sort_index()


def main() -> None:
    print("=" * 72)
    print("bond_gold V3.6 RE-AUDIT")
    print("=" * 72)

    closes = load_universe()
    print(
        f"[load] universe: IEF + GLD, {closes.shape[0]} bars "
        f"({closes.index[0].date()} -> {closes.index[-1].date()})"
    )

    # L52 sanctuary discipline.
    sanc = slice_sanctuary(closes, months=SANCTUARY_MONTHS)
    visible = sanc.visible
    print(
        f"[sanctuary] visible: {visible.shape[0]} bars "
        f"({visible.index[0].date()} -> {visible.index[-1].date()})"
    )
    print(
        f"[sanctuary] held out: {sanc.sanctuary.shape[0]} bars "
        f"({sanc.sanctuary_start.date()} -> {sanc.sanctuary_end.date()})"
    )

    # A1 causality smoke on the canonical config.
    bond_gold_assert_causal(closes, cfg=CELLS[CANONICAL_CELL])
    print("[causality] assert_causal: PASS")

    # WFO config — CROSS_ASSET_MOMENTUM default.
    class_def = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    wfo_cfg = class_def.wfo
    mc_cfg = class_def.mc
    folds = build_folds(visible.index, wfo_cfg, bars_per_year=252)
    print(
        f"[wfo] {len(folds)} folds (mode={wfo_cfg.is_mode}, "
        f"is_min={wfo_cfg.is_min_years}y, oos={wfo_cfg.oos_years}y)"
    )
    if not folds:
        print("[wfo] No folds — visible window too short. Abort.")
        return

    # ── Pass 1 — per-cell stitched OOS Sharpe + CI ────────────────────────
    print("\n" + "-" * 72)
    print("Pass 1 — per-cell stitched OOS Sharpe + CI")
    print("-" * 72)
    pass1_returns: dict[str, pd.Series] = {}
    pass1_sharpes: dict[str, float] = {}
    pass1_cis: dict[str, tuple[float, float]] = {}
    for name, cfg in CELLS.items():
        full = bond_gold_returns(visible, cfg=cfg)
        stitched = _stitched_oos_returns(full, folds)
        sr = float(sharpe(stitched, periods_per_year=252))
        ci_lo, ci_hi = bootstrap_sharpe_ci(stitched, periods_per_year=252, seed=42)
        pass1_returns[name] = stitched
        pass1_sharpes[name] = sr
        pass1_cis[name] = (ci_lo, ci_hi)
        print(
            f"  {name:>20s}  sharpe={sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
            f"n_oos={len(stitched.dropna())}"
        )

    # ── L52 plateau pre-flight (on stitched OOS) ──────────────────────────
    print("\n" + "-" * 72)
    print("L52 plateau pre-flight (on stitched OOS)")
    print("-" * 72)
    plateau_sharpes = [pass1_sharpes[c] for c in PLATEAU_CELLS]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    for c, s in zip(PLATEAU_CELLS, plateau_sharpes, strict=True):
        print(f"  {c:>20s}  sharpe={s:+.4f}")
    print(f"  spread = {plateau_spread * 100:.2f}%  (H1 gate: 50%; L27 strict: 30%)")
    plateau_h1_pass = plateau_spread <= 0.50
    plateau_l27_pass = plateau_spread <= 0.30
    if not plateau_h1_pass:
        print("  L52 H1 FAILED — IS plateau did NOT hold OOS. Aborting Pass 2.")
        _write_minimal_report(
            pass1_sharpes,
            pass1_cis,
            plateau_spread,
            verdict_global="RETIRED (L52 H1 plateau failed)",
        )
        return

    # ── L53 early gate ────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("L53 early gate (Pass-1 cannot clear CI_lo > 0 check)")
    print("-" * 72)
    any_can_clear, gate_per_cell = pass1_can_clear_any_cell(
        pass1_returns,
        periods_per_year=252,
        block_size=mc_cfg.block_size_bars,
    )
    for name, gr in gate_per_cell.items():
        marker = "RUN P2" if gr.can_clear else "skip"
        print(f"  {name:>20s}  approx_ci_lo={gr.approx_ci_lo:+.4f}  -> {marker}")
    if not any_can_clear:
        print("  L53 — no cell can plausibly clear CI_lo > 0. Skipping Pass 2.")
        _write_minimal_report(
            pass1_sharpes,
            pass1_cis,
            plateau_spread,
            verdict_global="RETIRED (L53 no cell can clear CI_lo)",
        )
        return

    # ── Pass 2 — MC + DSR + sanctuary + noise + decide ────────────────────
    print("\n" + "-" * 72)
    print("Pass 2 — MC + DSR + sanctuary + noise + decide")
    print("-" * 72)

    # Pre-build the sanctuary returns per cell (same logic as Pass 1, but
    # on the held-out 24-month window).
    sanctuary_returns_by_cell: dict[str, pd.Series] = {}
    sanctuary_sharpes_by_cell: dict[str, float] = {}
    sanctuary_pct_by_cell: dict[str, float] = {}
    for name, cfg in CELLS.items():
        # Compute on full series (visible + sanctuary) so the warm-up window
        # (504d z-score) has data from visible to draw on.
        full_for_sanc = bond_gold_returns(closes, cfg=cfg)
        sanc_ret = full_for_sanc.loc[sanc.sanctuary_start :]
        sanctuary_returns_by_cell[name] = sanc_ret
        sanc_sr = float(sharpe(sanc_ret, periods_per_year=252))
        sanctuary_sharpes_by_cell[name] = sanc_sr
        # Divergence test
        div = sanctuary_divergence_test(
            historical_returns=pass1_returns[name].dropna(),
            sanctuary_returns=sanc_ret.dropna(),
            periods_per_year=252,
        )
        sanctuary_pct_by_cell[name] = (
            float(div.percentile) if np.isfinite(div.percentile) else float("nan")
        )

    # MC needs a per-cell strategy_fn that operates on the bootstrap synthetic
    # DataFrame. Since the bootstrap is "shared_block" with IEF as extra,
    # the synthetic df has columns ("close", "IEF") (close is the primary =
    # GLD). Adapt the bond_gold computation to that schema.
    cell_rows: list[CellRow] = []
    for name, cfg in CELLS.items():
        sr_pass1 = pass1_sharpes[name]
        ci_lo, ci_hi = pass1_cis[name]

        def strategy_for_mc(df: pd.DataFrame, _cfg: BondGoldConfig = cfg) -> pd.Series:
            # df.columns = ('close', 'IEF') for shared_block MC with extras.
            synth = pd.DataFrame({"GLD": df["close"], "IEF": df["IEF"]})
            return bond_gold_returns(synth, cfg=_cfg)

        mc_res = run_block_mc(
            visible["GLD"],
            mc_cfg,
            strategy_for_mc,
            periods_per_year=252,
            seed=42,
            extra_series={"IEF": visible["IEF"]},
            n_workers=DEFAULT_MC_WORKERS,
        )

        # DSR — combine across N cells using sweep variance.
        # Use the 5 plateau cells as the "sweep" for DSR adjustment.
        sweep_sharpes_for_dsr = [pass1_sharpes[c] for c in PLATEAU_CELLS]
        sr_var = sr_var_from_sweep(sweep_sharpes_for_dsr)
        dsr = deflated_sharpe(
            sr_hat=sr_pass1,
            sr_var_across_trials=sr_var,
            returns=pass1_returns[name],
            n_trials=len(PLATEAU_CELLS),
        )

        # Noise robustness (Varma).
        def strategy_for_noise(
            closes_subset: pd.DataFrame, _cfg: BondGoldConfig = cfg
        ) -> pd.Series:
            return bond_gold_returns(closes_subset, cfg=_cfg)

        noise_res = run_noise_robustness(
            visible,
            strategy_for_noise,
            periods_per_year=252,
            cfg=NoiseConfig(),
        )

        # Decide.
        decision = decide(
            DecisionInputs(
                ci_lo=ci_lo,
                dsr_prob=dsr.dsr_prob,
                p_maxdd_gt_threshold=mc_res.p_maxdd_gt_threshold,
                pass_threshold_prob=mc_res.pass_threshold_prob,
                sanctuary_sharpe=sanctuary_sharpes_by_cell[name],
                noise_passes_mean=noise_res.passes,
                noise_passes_worst=noise_res.worst_case_passes,
            )
        )

        cell_rows.append(
            CellRow(
                name=name,
                cfg=cfg,
                n_oos_bars=len(pass1_returns[name].dropna()),
                n_folds=len(folds),
                sharpe=sr_pass1,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                dsr_prob=dsr.dsr_prob,
                mc_p_maxdd_gt_threshold=mc_res.p_maxdd_gt_threshold,
                mc_threshold_pct=mc_res.threshold_pct,
                sanctuary_sharpe=sanctuary_sharpes_by_cell[name],
                sanctuary_percentile=sanctuary_pct_by_cell[name],
                noise_base=noise_res.base_sharpe,
                noise_passes_mean=noise_res.passes,
                noise_passes_worst=noise_res.worst_case_passes,
                noise_axis=decision.noise_axis,
                verdict=decision.verdict.value,
                rationale=decision.rationale,
            )
        )
        print(
            f"  {name:>20s}  verdict={decision.verdict.value}  "
            f"CI_lo={ci_lo:+.3f}  sanc_sr={sanctuary_sharpes_by_cell[name]:+.3f}  "
            f"MC_p={mc_res.p_maxdd_gt_threshold:.3f}  noise={decision.noise_axis}"
        )

    # ── Selection per §3 rule ─────────────────────────────────────────────
    eligible = [
        r
        for r in cell_rows
        if r.name not in EXCLUDED_FROM_PROMOTION
        and r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT")
    ]
    selected = max(eligible, key=lambda r: r.ci_lo) if eligible else None

    # Apply L46 tighter-constraint rule.
    if selected is not None and selected.ci_lo <= 0:
        verdict_global = (
            f"NOT PROMOTED — {selected.name} passes 5-axis ({selected.verdict}) "
            f"but CI_lo={selected.ci_lo:+.3f} <= 0 (L46 tighter constraint binds)"
        )
    elif selected is None:
        verdict_global = "RETIRED — no cell DEPLOY or CONDITIONAL_WATCHPOINT"
    else:
        verdict_global = (
            f"PROMOTED — {selected.name} verdict={selected.verdict}, CI_lo={selected.ci_lo:+.3f}"
        )

    print("\n" + "=" * 72)
    print(f"FINAL VERDICT: {verdict_global}")
    print("=" * 72)

    _write_full_report(
        plateau_spread,
        plateau_l27_pass,
        cell_rows,
        verdict_global,
    )


def _write_minimal_report(
    pass1_sharpes,
    pass1_cis,
    plateau_spread,
    verdict_global,
) -> None:
    fp = REPORTS_DIR / "result_log.md"
    lines = [
        "# bond_gold V3.6 Re-Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        f"**Verdict:** {verdict_global}",
        "",
        "## §4.1 Pass-1 stitched OOS Sharpes",
        "",
        "| Cell | Sharpe | CI95_lo | CI95_hi |",
        "|---|---:|---:|---:|",
    ]
    for name, sr in pass1_sharpes.items():
        ci_lo, ci_hi = pass1_cis[name]
        lines.append(f"| {name} | {sr:+.4f} | {ci_lo:+.3f} | {ci_hi:+.3f} |")
    lines.append("")
    lines.append(f"## §4.2 L52 plateau pre-flight — spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append("Pass 2 SKIPPED.")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


def _write_full_report(
    plateau_spread,
    plateau_l27_pass,
    cell_rows,
    verdict_global,
) -> None:
    fp = REPORTS_DIR / "result_log.md"
    lines = [
        "# bond_gold V3.6 Re-Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        "**Strategy class:** CROSS_ASSET_MOMENTUM",
        "**Pre-reg:** `directives/Pre-Reg bond_gold Re-audit 2026-05-16.md`",
        "**Sweep informant:** `.tmp/reports/sweep_bond_gold/findings.md`",
        f"**Final verdict:** **{verdict_global}**",
        "",
        "## §4.1 Pass 1 — stitched OOS Sharpe + bootstrap CI",
        "",
        "| Cell | Sharpe | CI95_lo | CI95_hi | n_oos |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | {r.n_oos_bars} |"
        )
    lines.append("")
    lines.append(f"## §4.2 L52 plateau pre-flight — spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append(f"- H1 gate (≤ 50%): **{'PASS' if plateau_spread <= 0.50 else 'FAIL'}**")
    lines.append(f"- L27 strict gate (≤ 30%): **{'PASS' if plateau_l27_pass else 'FAIL'}**")
    lines.append("")
    lines.append("## §4.3 Per-cell 5-axis matrix")
    lines.append("")
    lines.append(
        "| Cell | Sharpe | CI_lo | CI_hi | DSR | MC P(DD>thr) | Sanc Sharpe | Sanc %ile | Noise base | Noise axis | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|")
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
            f"{r.dsr_prob:.3f} | {r.mc_p_maxdd_gt_threshold:.3f} | "
            f"{r.sanctuary_sharpe:+.3f} | {r.sanctuary_percentile:.2f} | "
            f"{r.noise_base:+.3f} | {r.noise_axis} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## §4.4 Falsification hypothesis verdicts")
    lines.append("")
    canonical = next(r for r in cell_rows if r.name == "C1_canonical")
    live = next(r for r in cell_rows if r.name == "C2_live_canonical")
    gross = next(r for r in cell_rows if r.name == "C4_gross_no_costs")
    h1_msg = (
        f"H1 (plateau holds OOS, spread ≤ 50%): "
        f"**{'SUPPORTED' if plateau_spread <= 0.50 else 'REJECTED'}** — spread = {plateau_spread * 100:.2f}%"
    )
    h2_msg = (
        f"H2 (canonical sanctuary Sharpe ≥ +0.30): "
        f"**{'SUPPORTED' if canonical.sanctuary_sharpe >= 0.30 else 'REJECTED'}** "
        f"— sanctuary Sharpe = {canonical.sanctuary_sharpe:+.3f}"
    )
    h3_msg = (
        f"H3 (C1 sanctuary Sharpe > C2 sanctuary Sharpe): "
        f"**{'SUPPORTED' if canonical.sanctuary_sharpe > live.sanctuary_sharpe else 'REJECTED'}** "
        f"— C1 {canonical.sanctuary_sharpe:+.3f} vs C2 {live.sanctuary_sharpe:+.3f}"
    )
    h4_msg = (
        f"H4 (C1 CI_lo > 0): **{'SUPPORTED' if canonical.ci_lo > 0 else 'REJECTED'}** "
        f"— CI_lo = {canonical.ci_lo:+.3f}"
    )
    h5_gap = gross.sharpe - canonical.sharpe
    h5_msg = (
        f"H5 (cost drag ≤ 0.30): **{'SUPPORTED' if h5_gap <= 0.30 else 'REJECTED'}** "
        f"— gross − net = {h5_gap:+.3f}"
    )
    for m in (h1_msg, h2_msg, h3_msg, h4_msg, h5_msg):
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## §4.5 Promotion verdict")
    lines.append("")
    lines.append(f"**{verdict_global}**")
    lines.append("")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
