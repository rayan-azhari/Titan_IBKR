"""turtle V1-era re-audit harness.

Specified by ``directives/Pre-Reg turtle Re-audit 2026-05-16.md``.

Run via::

    PYTHONIOENCODING=utf-8 uv run python research/turtle/run_turtle_reaudit.py

Pipeline (V3.6 + L52 hybrid):

1. Load CAT H1 OHLC bars.
2. ``slice_sanctuary(months=12)`` -> visible vs sanctuary.
3. L21 causality smoke.
4. Build WFO folds on visible (DAILY_TREND defaults, bars_per_year=1764).
5. Pass 1: per-cell stitched-OOS Sharpe + bootstrap CI.
6. L52 plateau pre-flight.
7. L53 early gate.
8. Pass 2: MC + L17 rel-MC + DSR + sanctuary + noise + decide per cell.
9. H6: multi-instrument robustness panel (10 liquid US equities).
10. Write S4 result log.
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

from research.turtle.turtle_strategy import (  # noqa: E402
    PERIODS_PER_YEAR_H1_EQ,
    TurtleConfig,
    turtle_assert_causal,
    turtle_returns,
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
    run_relative_block_mc,
    sanctuary_divergence_test,
    slice_sanctuary,
    sr_var_from_sweep,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.framework.wfo import build_folds  # noqa: E402
from titan.research.metrics import bootstrap_sharpe_ci, sharpe  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "turtle_reaudit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SANCTUARY_MONTHS = 12
PRIMARY_TICKER = "CAT"
ROBUSTNESS_PANEL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "XOM", "CVX", "JNJ", "PG", "MU",
]

# S2 frozen cell grid (per pre-reg).
CELLS: dict[str, TurtleConfig] = {
    "C1_canonical": TurtleConfig(entry_period=20, exit_period=30),
    "P_short_exit": TurtleConfig(entry_period=20, exit_period=20),
    "P_long_exit": TurtleConfig(entry_period=20, exit_period=45),
    "P_north": TurtleConfig(entry_period=20, exit_period=10),
    "C2_live_canonical": TurtleConfig(entry_period=45, exit_period=30),
    "C3_peak": TurtleConfig(entry_period=45, exit_period=20),
    "C4_gross_no_costs": TurtleConfig(entry_period=20, exit_period=30, apply_costs=False),
}
CANONICAL_CELL = "C1_canonical"
PLATEAU_CELLS = ("C1_canonical", "P_short_exit", "P_long_exit", "P_north")
EXCLUDED_FROM_PROMOTION = ("C2_live_canonical", "C4_gross_no_costs")  # C3_peak is eligible


def _load_h1(symbol: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{symbol}_H1.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    return df[["open", "high", "low", "close"]].astype(float)


@dataclass
class CellRow:
    name: str
    cfg: TurtleConfig
    n_oos_bars: int
    n_folds: int
    sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    mc_p_maxdd_gt_threshold: float
    mc_threshold_pct: float
    rel_mc_median_dd_reduction: float
    rel_mc_p_strategy_better: float
    rel_mc_passes: bool
    sanctuary_sharpe: float
    sanctuary_percentile: float
    noise_base: float
    noise_passes_mean: bool
    noise_passes_worst: bool
    noise_axis: str
    verdict: str


def _stitched_oos_returns(full_returns: pd.Series, folds: list) -> pd.Series:
    parts: list[pd.Series] = []
    for f in folds:
        oos_slice = full_returns.iloc[f.oos_start : f.oos_end_excl]
        parts.append(oos_slice)
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    return stitched.sort_index()


def _run_robustness_panel(cfg: TurtleConfig) -> dict[str, float]:
    """Run canonical cfg on each robustness ticker, return per-ticker Sharpe."""
    out: dict[str, float] = {}
    for sym in ROBUSTNESS_PANEL:
        try:
            bars = _load_h1(sym)
            if len(bars) < 5000:
                out[sym] = float("nan")
                continue
            ret = turtle_returns(bars, cfg=cfg)
            sr = float(sharpe(ret, periods_per_year=PERIODS_PER_YEAR_H1_EQ))
            out[sym] = sr
        except (FileNotFoundError, ValueError):
            out[sym] = float("nan")
    return out


def main() -> None:
    print("=" * 72)
    print("turtle V3.6 RE-AUDIT (CAT primary + 10-ticker robustness panel)")
    print("=" * 72)

    bars = _load_h1(PRIMARY_TICKER)
    print(
        f"[load] {PRIMARY_TICKER} H1, {bars.shape[0]} bars "
        f"({bars.index[0]} -> {bars.index[-1]})"
    )

    sanc = slice_sanctuary(bars, months=SANCTUARY_MONTHS)
    visible = sanc.visible
    print(
        f"[sanctuary] visible: {visible.shape[0]} bars "
        f"({visible.index[0]} -> {visible.index[-1]})"
    )
    print(
        f"[sanctuary] held out: {sanc.sanctuary.shape[0]} bars "
        f"({sanc.sanctuary_start} -> {sanc.sanctuary_end})"
    )

    turtle_assert_causal(bars, cfg=CELLS[CANONICAL_CELL])
    print("[causality] L21 assert_causal: PASS")

    class_def = defaults_for(StrategyClass.DAILY_TREND)
    wfo_cfg = class_def.wfo
    mc_cfg = class_def.mc
    folds = build_folds(visible.index, wfo_cfg, bars_per_year=PERIODS_PER_YEAR_H1_EQ)
    print(
        f"[wfo] {len(folds)} folds (mode={wfo_cfg.is_mode}, "
        f"is_min={wfo_cfg.is_min_years}y, oos={wfo_cfg.oos_years}y, "
        f"bars_per_year={PERIODS_PER_YEAR_H1_EQ})"
    )
    if not folds:
        print("[wfo] no folds -- visible window too short. Abort.")
        return

    # Pass 1 ---------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Pass 1 -- per-cell stitched OOS Sharpe + CI")
    print("-" * 72)
    pass1_returns: dict[str, pd.Series] = {}
    pass1_sharpes: dict[str, float] = {}
    pass1_cis: dict[str, tuple[float, float]] = {}
    for name, cfg in CELLS.items():
        full = turtle_returns(visible, cfg=cfg)
        stitched = _stitched_oos_returns(full, folds)
        sr = float(sharpe(stitched, periods_per_year=PERIODS_PER_YEAR_H1_EQ))
        ci_lo, ci_hi = bootstrap_sharpe_ci(stitched, periods_per_year=PERIODS_PER_YEAR_H1_EQ, seed=42)
        pass1_returns[name] = stitched
        pass1_sharpes[name] = sr
        pass1_cis[name] = (ci_lo, ci_hi)
        print(
            f"  {name:>20s}  sharpe={sr:+.4f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
            f"n_oos={len(stitched.dropna())}"
        )

    # L52 plateau pre-flight ------------------------------------------------
    print("\n" + "-" * 72)
    print("L52 plateau pre-flight (on stitched OOS)")
    print("-" * 72)
    plateau_sharpes = [pass1_sharpes[c] for c in PLATEAU_CELLS]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (
        (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    )
    for c, s in zip(PLATEAU_CELLS, plateau_sharpes, strict=True):
        print(f"  {c:>20s}  sharpe={s:+.4f}")
    print(f"  spread = {plateau_spread * 100:.2f}%  (H1 gate: 50%; L27 strict: 30%)")
    plateau_h1_pass = plateau_spread <= 0.50
    plateau_l27_pass = plateau_spread <= 0.30
    if not plateau_h1_pass:
        print("  L52 H1 FAILED -- IS plateau did NOT hold OOS. Aborting Pass 2.")
        _write_minimal_report(
            pass1_sharpes, pass1_cis, plateau_spread,
            verdict_global="RETIRED (L52 H1 plateau failed)",
        )
        return

    # L53 early gate --------------------------------------------------------
    print("\n" + "-" * 72)
    print("L53 early gate")
    print("-" * 72)
    any_can_clear, gate_per_cell = pass1_can_clear_any_cell(
        pass1_returns,
        periods_per_year=PERIODS_PER_YEAR_H1_EQ,
        block_size=mc_cfg.block_size_bars,
    )
    for name, gr in gate_per_cell.items():
        marker = "RUN P2" if gr.can_clear else "skip"
        print(f"  {name:>20s}  approx_ci_lo={gr.approx_ci_lo:+.4f}  -> {marker}")
    if not any_can_clear:
        print("  L53 -- no cell can plausibly clear CI_lo > 0. Skipping Pass 2.")
        _write_minimal_report(
            pass1_sharpes, pass1_cis, plateau_spread,
            verdict_global="RETIRED (L53 no cell can clear CI_lo)",
        )
        return

    # Pass 2 ----------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Pass 2 -- MC + L17 rel-MC + DSR + sanctuary + noise + decide")
    print("-" * 72)

    sanctuary_returns_by_cell: dict[str, pd.Series] = {}
    sanctuary_sharpes_by_cell: dict[str, float] = {}
    sanctuary_pct_by_cell: dict[str, float] = {}
    for name, cfg in CELLS.items():
        full_for_sanc = turtle_returns(bars, cfg=cfg)
        sanc_ret = full_for_sanc.loc[sanc.sanctuary_start :]
        sanctuary_returns_by_cell[name] = sanc_ret
        sanctuary_sharpes_by_cell[name] = float(sharpe(sanc_ret, periods_per_year=PERIODS_PER_YEAR_H1_EQ))
        div = sanctuary_divergence_test(
            historical_returns=pass1_returns[name].dropna(),
            sanctuary_returns=sanc_ret.dropna(),
            periods_per_year=PERIODS_PER_YEAR_H1_EQ,
        )
        sanctuary_pct_by_cell[name] = (
            float(div.percentile) if np.isfinite(div.percentile) else float("nan")
        )

    cell_rows: list[CellRow] = []
    for name, cfg in CELLS.items():
        sr_pass1 = pass1_sharpes[name]
        ci_lo, ci_hi = pass1_cis[name]

        def strategy_for_mc(df: pd.DataFrame, _cfg: TurtleConfig = cfg) -> pd.Series:
            # Synth df has 'close' from primary; we don't have OHLC in synth bootstrap
            # so we approximate high/low = close. This is a conservative simplification
            # (under-estimates Donchian channel width), so any pass under this MC is
            # robust; any fail could be due to the simplification, not the strategy.
            synth = pd.DataFrame(
                {
                    "open": df["close"],
                    "high": df["close"],
                    "low": df["close"],
                    "close": df["close"],
                }
            )
            return turtle_returns(synth, cfg=_cfg)

        mc_res = run_block_mc(
            visible["close"],
            mc_cfg,
            strategy_for_mc,
            periods_per_year=PERIODS_PER_YEAR_H1_EQ,
            seed=42,
            n_workers=DEFAULT_MC_WORKERS,
        )

        def benchmark_bh(df: pd.DataFrame) -> pd.Series:
            return np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

        rel_mc_res = run_relative_block_mc(
            visible["close"],
            mc_cfg,
            strategy_for_mc,
            benchmark_bh,
            periods_per_year=PERIODS_PER_YEAR_H1_EQ,
            seed=42,
            n_workers=DEFAULT_MC_WORKERS,
            median_ratio_gate=0.80,
            p_strategy_better_gate=0.50,
        )

        sweep_sharpes_for_dsr = [pass1_sharpes[c] for c in PLATEAU_CELLS]
        sr_var = sr_var_from_sweep(sweep_sharpes_for_dsr)
        dsr = deflated_sharpe(
            sr_hat=sr_pass1,
            sr_var_across_trials=sr_var,
            returns=pass1_returns[name],
            n_trials=len(PLATEAU_CELLS),
        )

        def strategy_for_noise(bars_subset: pd.DataFrame, _cfg: TurtleConfig = cfg) -> pd.Series:
            return turtle_returns(bars_subset, cfg=_cfg)

        noise_res = run_noise_robustness(
            visible,
            strategy_for_noise,
            periods_per_year=PERIODS_PER_YEAR_H1_EQ,
            cfg=NoiseConfig(),
        )

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
                rel_mc_median_dd_reduction=rel_mc_res.median_dd_reduction,
                rel_mc_p_strategy_better=rel_mc_res.p_strategy_better,
                rel_mc_passes=rel_mc_res.passes,
                sanctuary_sharpe=sanctuary_sharpes_by_cell[name],
                sanctuary_percentile=sanctuary_pct_by_cell[name],
                noise_base=noise_res.base_sharpe,
                noise_passes_mean=noise_res.passes,
                noise_passes_worst=noise_res.worst_case_passes,
                noise_axis=decision.noise_axis,
                verdict=decision.verdict.value,
            )
        )
        print(
            f"  {name:>20s}  verdict={decision.verdict.value}  "
            f"CI_lo={ci_lo:+.3f}  sanc_sr={sanctuary_sharpes_by_cell[name]:+.3f}  "
            f"L17_rel_dd={rel_mc_res.median_dd_reduction:.2f} "
            f"({'PASS' if rel_mc_res.passes else 'FAIL'})  "
            f"noise={decision.noise_axis}"
        )

    # H6: multi-instrument robustness ---------------------------------------
    print("\n" + "-" * 72)
    print("H6: Multi-instrument robustness panel (canonical params on 10 tickers)")
    print("-" * 72)
    panel_sharpes = _run_robustness_panel(CELLS[CANONICAL_CELL])
    valid_sharpes = [s for s in panel_sharpes.values() if np.isfinite(s)]
    panel_median = float(np.median(valid_sharpes)) if valid_sharpes else float("nan")
    cat_sharpe = pass1_sharpes[CANONICAL_CELL]
    # Build full distribution: CAT (Pass-1) + panel (full-series)
    all_for_pct = list(valid_sharpes) + [cat_sharpe]
    cat_percentile = (
        sum(1 for s in all_for_pct if s <= cat_sharpe) / len(all_for_pct) * 100
        if all_for_pct
        else float("nan")
    )
    for sym, sr in panel_sharpes.items():
        print(f"  {sym:>6s}  Sharpe={sr:+.4f}")
    print(f"  panel median = {panel_median:+.4f}")
    print(f"  CAT percentile within panel+CAT = {cat_percentile:.0f}%")
    h6_supported = panel_median >= 0 and cat_percentile <= 75

    # Selection per §3 ------------------------------------------------------
    eligible = [
        r
        for r in cell_rows
        if r.name not in EXCLUDED_FROM_PROMOTION
        and r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT")
        and r.rel_mc_passes
    ]
    selected = max(eligible, key=lambda r: r.ci_lo) if eligible else None

    if selected is not None and selected.ci_lo <= 0:
        verdict_global = (
            f"NOT PROMOTED -- {selected.name} passes 5-axis + L17 but "
            f"CI_lo={selected.ci_lo:+.3f} <= 0 (L46 tighter constraint binds)"
        )
    elif selected is None:
        has_promotable_verdict = any(
            r.name not in EXCLUDED_FROM_PROMOTION
            and r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT")
            for r in cell_rows
        )
        if has_promotable_verdict:
            verdict_global = "RETIRED -- cells pass 5-axis but ALL fail L17 rel-MC (L56 precedent)"
        else:
            verdict_global = "RETIRED -- no cell DEPLOY or CONDITIONAL_WATCHPOINT"
    else:
        h6_tag = "+ H6 ROBUST" if h6_supported else "+ H6 FAIL (CAT-specific)"
        verdict_global = (
            f"PROMOTED -- {selected.name} verdict={selected.verdict}, "
            f"CI_lo={selected.ci_lo:+.3f}, L17 rel_dd={selected.rel_mc_median_dd_reduction:.2f}, {h6_tag}"
        )

    print("\n" + "=" * 72)
    print(f"FINAL VERDICT: {verdict_global}")
    print("=" * 72)

    _write_full_report(
        plateau_spread,
        plateau_l27_pass,
        cell_rows,
        panel_sharpes,
        panel_median,
        cat_percentile,
        h6_supported,
        verdict_global,
    )


def _write_minimal_report(pass1_sharpes, pass1_cis, plateau_spread, verdict_global) -> None:
    fp = REPORTS_DIR / "result_log.md"
    lines = [
        "# turtle V3.6 Re-Audit Result Log",
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
    lines.append(f"## §4.2 L52 plateau pre-flight -- spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append("Pass 2 SKIPPED.")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


def _write_full_report(
    plateau_spread,
    plateau_l27_pass,
    cell_rows,
    panel_sharpes,
    panel_median,
    cat_percentile,
    h6_supported,
    verdict_global,
) -> None:
    fp = REPORTS_DIR / "result_log.md"
    lines = [
        "# turtle V3.6 Re-Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        "**Strategy class:** DAILY_TREND (H1 Donchian breakout)",
        "**Pre-reg:** `directives/Pre-Reg turtle Re-audit 2026-05-16.md`",
        "**Sweep informant:** `.tmp/reports/sweep_turtle/plateau_report.md`",
        f"**periods_per_year:** {PERIODS_PER_YEAR_H1_EQ} (US equity RTH H1, NOT 24/7 FX)",
        f"**Final verdict:** **{verdict_global}**",
        "",
        "## §4.1 Pass 1 -- stitched OOS Sharpe + bootstrap CI",
        "",
        "| Cell | Sharpe | CI95_lo | CI95_hi | n_oos |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | {r.n_oos_bars} |"
        )
    lines.append("")
    lines.append(f"## §4.2 L52 plateau pre-flight -- spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append(f"- H1 gate (<= 50%): **{'PASS' if plateau_spread <= 0.50 else 'FAIL'}**")
    lines.append(f"- L27 strict gate (<= 30%): **{'PASS' if plateau_l27_pass else 'FAIL'}**")
    lines.append("")
    lines.append("## §4.3 Per-cell 5-axis matrix + L17 rel-MC")
    lines.append("")
    lines.append(
        "| Cell | Sharpe | CI_lo | DSR | MC_p | L17 rel_dd | L17 pass | Sanc Sharpe | Sanc %ile | Noise base | Noise axis | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|---|")
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | "
            f"{r.dsr_prob:.3f} | {r.mc_p_maxdd_gt_threshold:.3f} | "
            f"{r.rel_mc_median_dd_reduction:.2f} | {'PASS' if r.rel_mc_passes else 'FAIL'} | "
            f"{r.sanctuary_sharpe:+.3f} | {r.sanctuary_percentile:.2f} | "
            f"{r.noise_base:+.3f} | {r.noise_axis} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## §4.6 H6 Multi-instrument robustness panel")
    lines.append("")
    lines.append("Canonical (entry=20, exit=30) on each ticker's full H1 history, per-bar Sharpe.")
    lines.append("")
    lines.append("| Ticker | Sharpe |")
    lines.append("|---|---:|")
    for sym, sr in panel_sharpes.items():
        lines.append(f"| {sym} | {sr:+.4f} |")
    lines.append(f"| **panel median** | **{panel_median:+.4f}** |")
    lines.append(f"| **CAT percentile** | **{cat_percentile:.0f}%** |")
    lines.append("")
    lines.append(f"H6 (panel median >= 0 AND CAT %ile <= 75): **{'SUPPORTED' if h6_supported else 'REJECTED'}**")
    lines.append("")
    lines.append("## §4.7 Falsification hypothesis verdicts")
    lines.append("")
    canonical = next(r for r in cell_rows if r.name == "C1_canonical")
    live = next(r for r in cell_rows if r.name == "C2_live_canonical")
    peak = next(r for r in cell_rows if r.name == "C3_peak")
    gross = next(r for r in cell_rows if r.name == "C4_gross_no_costs")
    lines.append(
        f"- H1 (plateau holds OOS, spread <= 50%): "
        f"**{'SUPPORTED' if plateau_spread <= 0.50 else 'REJECTED'}** -- spread = {plateau_spread * 100:.2f}%"
    )
    lines.append(
        f"- H2 (canonical sanctuary Sharpe >= +0.30): "
        f"**{'SUPPORTED' if canonical.sanctuary_sharpe >= 0.30 else 'REJECTED'}** -- "
        f"sanctuary Sharpe = {canonical.sanctuary_sharpe:+.3f}"
    )
    lines.append(
        f"- H3 (C1 sanctuary > C2 live): "
        f"**{'SUPPORTED' if canonical.sanctuary_sharpe > live.sanctuary_sharpe else 'REJECTED'}** -- "
        f"C1 {canonical.sanctuary_sharpe:+.3f} vs C2 {live.sanctuary_sharpe:+.3f}"
    )
    lines.append(
        f"- H4 (C1 CI_lo > 0): **{'SUPPORTED' if canonical.ci_lo > 0 else 'REJECTED'}** -- "
        f"CI_lo = {canonical.ci_lo:+.3f}"
    )
    h5_sup = (peak.sanctuary_sharpe > canonical.sanctuary_sharpe) and peak.ci_lo > 0
    lines.append(
        f"- H5 (peak C3 > plateau C1 + C3 CI_lo > 0): "
        f"**{'SUPPORTED' if h5_sup else 'REJECTED'}** -- "
        f"C3 sanc {peak.sanctuary_sharpe:+.3f} vs C1 sanc {canonical.sanctuary_sharpe:+.3f}, "
        f"C3 CI_lo {peak.ci_lo:+.3f}"
    )
    lines.append(
        f"- H6 (multi-instrument): **{'SUPPORTED' if h6_supported else 'REJECTED'}** -- "
        f"panel median {panel_median:+.4f}, CAT %ile {cat_percentile:.0f}%"
    )
    h7_gap = gross.sharpe - canonical.sharpe
    lines.append(
        f"- H7 (cost drag <= 0.25): **{'SUPPORTED' if h7_gap <= 0.25 else 'REJECTED'}** -- "
        f"gross - net = {h7_gap:+.3f}"
    )
    lines.append("")
    lines.append("## §4.8 Promotion verdict")
    lines.append("")
    lines.append(f"**{verdict_global}**")
    lines.append("")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
