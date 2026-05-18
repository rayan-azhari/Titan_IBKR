"""Generic Wave A.2 spot-check audit harness for binary-sized etf_trend variants.

Specified by `directives/Bulk-Retire etf_trend Unleveraged Variants 2026-05-16.md`
Phase 2 (medium-confidence variants — DBC and GLD).

Usage::

    PYTHONIOENCODING=utf-8 uv run python research/etf_trend/run_etf_trend_binary_spotcheck.py DBC
    PYTHONIOENCODING=utf-8 uv run python research/etf_trend/run_etf_trend_binary_spotcheck.py GLD

Pre-reg cells are MINIMAL (4 cells per audit — sufficient for a
confirmation that L56 applies). Verdict criteria same as SPY/TQQQ:
DEPLOY/CONDITIONAL → keep; TIER_UNCONFIRMED/RETIRED → de-allocate.
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

from research.etf_trend.etf_trend_binary_strategy import (  # noqa: E402
    EtfTrendBinaryConfig,
    buy_and_hold_binary_returns,
    etf_trend_binary_assert_causal,
    etf_trend_binary_returns,
)
from titan.research.framework import (  # noqa: E402
    DecisionInputs,
    NoiseConfig,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    pass1_can_clear_any_cell,
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
SANCTUARY_MONTHS = 24

# Per-symbol live config — mirrors `config/etf_trend_<sym>.toml`.
LIVE_CONFIG = {
    "DBC": {"slow_ma": 75, "exit_confirm_days": 1},
    "GLD": {"slow_ma": 250, "exit_confirm_days": 5},
}


def _make_cells(symbol: str) -> dict[str, EtfTrendBinaryConfig]:
    """Generic 4-cell pre-reg for the spot-check.

    C1 = live config (parity baseline)
    P_slow = slower slow_ma (test plateau symmetry)
    P_confirm5 = exit_confirm=5 (test noise mitigation per L56 refined)
    C2_gross = gross (cost drag reference)
    """
    live = LIVE_CONFIG[symbol]
    return {
        "C1_live": EtfTrendBinaryConfig(
            symbol=symbol,
            slow_ma=live["slow_ma"],
            exit_confirm_days=live["exit_confirm_days"],
        ),
        "P_slow_ma_2x": EtfTrendBinaryConfig(
            symbol=symbol,
            slow_ma=live["slow_ma"] * 2,
            exit_confirm_days=live["exit_confirm_days"],
        ),
        "P_confirm5": EtfTrendBinaryConfig(
            symbol=symbol,
            slow_ma=live["slow_ma"],
            exit_confirm_days=max(5, live["exit_confirm_days"]),
        ),
        "C2_gross": EtfTrendBinaryConfig(
            symbol=symbol,
            slow_ma=live["slow_ma"],
            exit_confirm_days=live["exit_confirm_days"],
            apply_costs=False,
        ),
    }


@dataclass
class CellRow:
    name: str
    cfg: EtfTrendBinaryConfig
    n_oos_bars: int
    sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    rel_mc_median_dd_reduction: float
    rel_mc_p_strategy_better: float
    rel_mc_passes: bool
    sanctuary_sharpe: float
    sanctuary_percentile: float
    sanctuary_lucky: bool
    noise_axis: str
    verdict: str


def _load_close(symbol: str) -> pd.Series:
    fp = DATA_DIR / f"{symbol}_D.parquet"
    df = pd.read_parquet(fp)
    s = df["close"].astype(float)
    s.name = symbol
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index().dropna()


def load_universe(symbol: str) -> pd.DataFrame:
    return pd.DataFrame({symbol: _load_close(symbol)})


def _stitched_oos_returns(full_returns: pd.Series, folds: list) -> pd.Series:
    parts = [full_returns.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    return stitched[~stitched.index.duplicated(keep="last")].sort_index()


def run_spotcheck(symbol: str) -> None:
    print("=" * 72)
    print(f"etf_trend {symbol} SPOT-CHECK AUDIT (Wave A.2 follow-up)")
    print("=" * 72)

    reports_dir = PROJECT_ROOT / ".tmp" / "reports" / f"etf_trend_{symbol.lower()}_spotcheck"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cells = _make_cells(symbol)
    canonical_cell = "C1_live"
    plateau_cells = ("C1_live", "P_slow_ma_2x", "P_confirm5")
    excluded = ("C2_gross",)

    closes = load_universe(symbol)
    print(
        f"[load] {symbol}: {closes.shape[0]} bars "
        f"({closes.index[0].date()} -> {closes.index[-1].date()})"
    )

    sanc = slice_sanctuary(closes, months=SANCTUARY_MONTHS)
    visible = sanc.visible
    print(f"[sanctuary] visible: {visible.shape[0]} bars, held out: {sanc.sanctuary.shape[0]} bars")

    etf_trend_binary_assert_causal(closes, cfg=cells[canonical_cell])
    print("[causality] assert_causal: PASS")

    class_def = defaults_for(StrategyClass.DAILY_TREND)
    wfo_cfg = class_def.wfo
    mc_cfg = class_def.mc
    folds = build_folds(visible.index, wfo_cfg, bars_per_year=252)
    print(f"[wfo] {len(folds)} folds")
    if not folds:
        print("[wfo] No folds — abort.")
        return

    # Pass 1 — stitched OOS Sharpe per cell.
    print("\n" + "-" * 72)
    print("Pass 1 — stitched OOS Sharpe + CI")
    print("-" * 72)
    pass1_returns: dict[str, pd.Series] = {}
    pass1_sharpes: dict[str, float] = {}
    pass1_cis: dict[str, tuple[float, float]] = {}
    for name, cfg in cells.items():
        full = etf_trend_binary_returns(visible, cfg=cfg)
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

    # L52 plateau pre-flight.
    plateau_sharpes = [pass1_sharpes[c] for c in plateau_cells]
    plateau_mean = float(np.mean(plateau_sharpes))
    plateau_spread = (max(plateau_sharpes) - min(plateau_sharpes)) / max(abs(plateau_mean), 1e-9)
    print(f"\n[plateau] spread = {plateau_spread * 100:.2f}% (L27 30% gate)")

    # L53 early gate.
    any_can_clear, gate_per_cell = pass1_can_clear_any_cell(
        pass1_returns,
        periods_per_year=252,
        block_size=mc_cfg.block_size_bars,
    )
    print(f"[L53] any_can_clear={any_can_clear}")

    # Pass 2 (L17 rel-MC vs B&H + DSR + sanctuary + noise + decide).
    print("\n" + "-" * 72)
    print("Pass 2 — L17 relative MC + DSR + sanctuary + noise + decide")
    print("-" * 72)

    # Sanctuary divergence.
    sanc_sr_by_cell: dict[str, float] = {}
    sanc_pct_by_cell: dict[str, float] = {}
    sanc_lucky_by_cell: dict[str, bool] = {}
    for name, cfg in cells.items():
        full = etf_trend_binary_returns(closes, cfg=cfg)
        sr = float(sharpe(full.loc[sanc.sanctuary_start :], periods_per_year=252))
        sanc_sr_by_cell[name] = sr
        div = sanctuary_divergence_test(
            historical_returns=pass1_returns[name].dropna(),
            sanctuary_returns=full.loc[sanc.sanctuary_start :].dropna(),
            periods_per_year=252,
        )
        sanc_pct_by_cell[name] = (
            float(div.percentile) if np.isfinite(div.percentile) else float("nan")
        )
        sanc_lucky_by_cell[name] = bool(div.lucky_flag)

    def benchmark_for_mc(df: pd.DataFrame, _sym: str = symbol) -> pd.Series:
        renamed = df.rename(columns={"close": _sym})
        return buy_and_hold_binary_returns(
            renamed,
            cfg=EtfTrendBinaryConfig(symbol=_sym),
        )

    cell_rows: list[CellRow] = []
    for name, cfg in cells.items():
        sr_pass1 = pass1_sharpes[name]
        ci_lo, ci_hi = pass1_cis[name]

        def strategy_for_mc(
            df: pd.DataFrame, _name: str = name, _cfg: EtfTrendBinaryConfig = cfg
        ) -> pd.Series:
            renamed = df.rename(columns={"close": _cfg.symbol})
            return etf_trend_binary_returns(renamed, cfg=_cfg)

        rel_mc = run_relative_block_mc(
            visible[symbol],
            mc_cfg,
            strategy_for_mc,
            benchmark_for_mc,
            periods_per_year=252,
            seed=42,
            n_workers=DEFAULT_MC_WORKERS,
            median_ratio_gate=0.80,
            p_strategy_better_gate=0.50,
        )

        sweep_sharpes_for_dsr = [pass1_sharpes[c] for c in plateau_cells]
        sr_var = sr_var_from_sweep(sweep_sharpes_for_dsr)
        dsr = deflated_sharpe(
            sr_hat=sr_pass1,
            sr_var_across_trials=sr_var,
            returns=pass1_returns[name],
            n_trials=len(plateau_cells),
        )

        def strategy_for_noise(
            closes_subset: pd.DataFrame, _cfg: EtfTrendBinaryConfig = cfg
        ) -> pd.Series:
            return etf_trend_binary_returns(closes_subset, cfg=_cfg)

        noise_res = run_noise_robustness(
            visible,
            strategy_for_noise,
            periods_per_year=252,
            cfg=NoiseConfig(),
        )

        synthetic_mc_p = 0.0 if rel_mc.passes else 1.0
        decision = decide(
            DecisionInputs(
                ci_lo=ci_lo,
                dsr_prob=dsr.dsr_prob,
                p_maxdd_gt_threshold=synthetic_mc_p,
                pass_threshold_prob=mc_cfg.max_dd_pass_prob,
                sanctuary_sharpe=sanc_sr_by_cell[name],
                noise_passes_mean=noise_res.passes,
                noise_passes_worst=noise_res.worst_case_passes,
            )
        )

        cell_rows.append(
            CellRow(
                name=name,
                cfg=cfg,
                n_oos_bars=len(pass1_returns[name].dropna()),
                sharpe=sr_pass1,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                dsr_prob=dsr.dsr_prob,
                rel_mc_median_dd_reduction=rel_mc.median_dd_reduction,
                rel_mc_p_strategy_better=rel_mc.p_strategy_better,
                rel_mc_passes=rel_mc.passes,
                sanctuary_sharpe=sanc_sr_by_cell[name],
                sanctuary_percentile=sanc_pct_by_cell[name],
                sanctuary_lucky=sanc_lucky_by_cell[name],
                noise_axis=decision.noise_axis,
                verdict=decision.verdict.value,
            )
        )
        print(
            f"  {name:>20s}  verdict={decision.verdict.value}  CI_lo={ci_lo:+.3f}  "
            f"rel_dd={rel_mc.median_dd_reduction:.3f}  rel_pass={rel_mc.passes}  "
            f"noise={decision.noise_axis}"
        )

    # Selection.
    eligible = [
        r
        for r in cell_rows
        if r.name not in excluded and r.verdict in ("DEPLOY", "CONDITIONAL_WATCHPOINT")
    ]
    selected = max(eligible, key=lambda r: r.ci_lo) if eligible else None
    if selected is not None and selected.ci_lo <= 0:
        verdict_global = f"NOT PROMOTED — {selected.name} ({selected.verdict}) but CI_lo={selected.ci_lo:+.3f} <= 0"
    elif selected is None:
        verdict_global = "CONFIRMS BULK-RETIRE — no cell promotable (L56 generalises)"
    else:
        verdict_global = (
            f"REFUTES BULK-RETIRE — {selected.name} verdict={selected.verdict}, keep live"
        )

    print("\n" + "=" * 72)
    print(f"FINAL VERDICT: {verdict_global}")
    print("=" * 72)

    _write_report(reports_dir, symbol, plateau_spread, cell_rows, verdict_global)


def _write_report(
    reports_dir: Path,
    symbol: str,
    plateau_spread: float,
    cell_rows: list[CellRow],
    verdict_global: str,
) -> None:
    fp = reports_dir / "result_log.md"
    c1 = next(r for r in cell_rows if r.name == "C1_live")

    lines = [
        f"# etf_trend {symbol} Spot-Check Audit Result Log",
        "",
        "**Run date:** 2026-05-16",
        "**Strategy class:** DAILY_TREND",
        "**Purpose:** Wave A.2 follow-up — confirm or refute L56 bulk-retire for binary-sized variant.",
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
    lines.append(f"## §4.2 Plateau pre-flight — spread {plateau_spread * 100:.2f}%")
    lines.append("")
    lines.append("## §4.3 5-axis matrix (L17 rel-MC vs B&H)")
    lines.append("")
    lines.append(
        "| Cell | Sharpe | CI_lo | Rel-MC DD red | P(strat better) | Rel-MC pass | Sanc Sharpe | Sanc %ile | Lucky | Noise axis | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|---|")
    for r in cell_rows:
        lines.append(
            f"| {r.name} | {r.sharpe:+.4f} | {r.ci_lo:+.3f} | "
            f"{r.rel_mc_median_dd_reduction:.3f} | {r.rel_mc_p_strategy_better:.3f} | "
            f"{'YES' if r.rel_mc_passes else 'no'} | {r.sanctuary_sharpe:+.3f} | "
            f"{r.sanctuary_percentile:.2f} | {'YES' if r.sanctuary_lucky else 'no'} | "
            f"{r.noise_axis} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## §4.4 L56 verdict for this variant")
    lines.append("")
    bulk_retire_confirmed = not c1.rel_mc_passes and c1.noise_axis in ("worst", "mid")
    if bulk_retire_confirmed:
        lines.append("**L56 CONFIRMED on this variant.** Rel-MC fails AND noise axis is not best;")
        lines.append("bulk-retire recommendation stands. De-allocate at next allocator window.")
    else:
        lines.append("**L56 PARTIALLY REFUTED.** Either rel-MC passes OR noise axis is best.")
        lines.append("Check the §3 selection rule — if a cell is DEPLOY/CONDITIONAL, keep it live")
        lines.append("(possibly migrate to that canonical).")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {fp.relative_to(PROJECT_ROOT)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_etf_trend_binary_spotcheck.py <SYMBOL>  (DBC or GLD)")
    symbol = sys.argv[1].upper()
    if symbol not in LIVE_CONFIG:
        raise SystemExit(f"Unknown symbol {symbol}; known: {list(LIVE_CONFIG.keys())}")
    run_spotcheck(symbol)


if __name__ == "__main__":
    main()
