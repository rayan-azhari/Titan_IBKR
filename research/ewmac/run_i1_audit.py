"""I1 — Per-asset HMM regime gate on Carver EWMAC.

Pre-registered in ``directives/Pre-Reg I1 HMM Per-Asset Regime + EWMAC
Gate 2026-05-16.md``. Final L48/L49 escalation: after broad-index regime
gates (B2c trend, B2d vol) both failed, test whether per-asset HMM-based
regime detection rescues the cross-asset universal-trend EWMAC.

Run via::

    uv run python research/ewmac/run_i1_audit.py

§§ Pre-reg cells (V3.1, 8 cells) + exploratory sweep (clearly labelled
separately; DSR penalty applied at the actual cell count).

WFO: CROSS_ASSET_MOMENTUM defaults — but HMM fitting requires a single
IS/OOS split per HMM call. The audit harness performs a **single
holistic-fold** approach: fit the HMM on the visible IS (everything
before the OOS sanctuary), Viterbi-decode forward, then evaluate fold
OOS Sharpes as in B2b/B2c/B2d. The per-fold WFO Sharpe stitching
still applies; only the HMM training stays frozen on the global IS.
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
    ewmac_returns,
)
from research.regime.hmm_gate import (  # noqa: E402
    HMMGateConfig,
    PerAssetRegimeGateConfig,
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
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "i1_ewmac_hmm_gate"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_BASE_EWMAC = dict(
    speeds=((16, 64), (32, 128), (64, 256)),
    fdm=1.35,
    forecast_cap=20.0,
)

# Pre-reg cells (V3.1). 8 cells per Pre-Reg I1 §2.
PRE_REG_CELLS: dict[str, EwmacConfig] = {
    "C1_canonical": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", random_seed=42),
        ),
    ),
    "C2_short_train": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, train_window_bars=252, state_id="autocorr"),
        ),
    ),
    "C3_long_train": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, train_window_bars=252 * 5, state_id="autocorr"),
        ),
    ),
    "C4_vol_based_id": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="low_vol", random_seed=42),
        ),
    ),
    "C5_no_gate_baseline": EwmacConfig(**_BASE_EWMAC, per_asset_regime_gate=None),
    "C7_gross_no_costs": EwmacConfig(
        **_BASE_EWMAC,
        apply_costs=False,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", random_seed=42),
        ),
    ),
    "C8_3_state": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=3, state_id="autocorr", random_seed=42),
        ),
    ),
}
CANONICAL_CELL = "C1_canonical"

# Plateau pre-flight neighbours of C1 (V3.1).
PLATEAU_NEIGHBOURS: dict[str, EwmacConfig] = {
    "P_train_2y": replace(
        PRE_REG_CELLS[CANONICAL_CELL],
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, train_window_bars=252 * 2, state_id="autocorr")
        ),
    ),
    "P_train_3y": replace(
        PRE_REG_CELLS[CANONICAL_CELL],
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, train_window_bars=252 * 3, state_id="autocorr")
        ),
    ),
    "P_seed_alt": replace(
        PRE_REG_CELLS[CANONICAL_CELL],
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", random_seed=11)
        ),
    ),
    "P_smoothing_5d": replace(
        PRE_REG_CELLS[CANONICAL_CELL],
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", smoothing_days=5)
        ),
    ),
}

# Exploratory sweep (BEYOND pre-reg). DSR penalty includes these.
# Documented separately in §4.6 of the result log.
EXPLORATORY_SWEEP: dict[str, EwmacConfig] = {
    "X1_train_full_smooth10": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", smoothing_days=10),
        ),
    ),
    "X2_high_mean_id": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="high_mean"),
        ),
    ),
    "X3_3state_vol_id": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=3, state_id="low_vol"),
        ),
    ),
    "X4_train_1y_smoothing5": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(
                n_states=2, train_window_bars=252, state_id="autocorr", smoothing_days=5
            ),
        ),
    ),
    "X5_seed_99": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", random_seed=99),
        ),
    ),
    "X6_seed_123": EwmacConfig(
        **_BASE_EWMAC,
        per_asset_regime_gate=PerAssetRegimeGateConfig(
            hmm=HMMGateConfig(n_states=2, state_id="autocorr", random_seed=123),
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
    is_sharpe: float = float("nan")  # I1 H4 overfit check


def _stitched_oos_sharpe(closes, cfg, folds, bars_per_year, is_end_idx):
    rets = ewmac_returns(closes, cfg=cfg, is_end_idx=is_end_idx)
    parts = [rets.iloc[f.oos_start : f.oos_end_excl] for f in folds]
    stitched = pd.concat(parts).fillna(0.0)
    return float(sharpe(stitched, periods_per_year=bars_per_year)), stitched


def _is_sharpe(closes, cfg, is_end_idx, bars_per_year):
    """I1 H4 helper: Sharpe computed on the IS window itself."""
    rets = ewmac_returns(closes, cfg=cfg, is_end_idx=is_end_idx)
    return float(sharpe(rets.iloc[:is_end_idx], periods_per_year=bars_per_year))


def _strategy_fn_for_cell(cfg, closes_visible, is_end_idx):
    primary_root = closes_visible.columns[0]
    other_roots = list(closes_visible.columns[1:])

    def strategy_fn(df):
        u = pd.DataFrame(index=df.index)
        u[primary_root] = df["close"]
        for r in other_roots:
            if r in df.columns:
                u[r] = df[r]
        # MC paths may have different length than visible; pass is_end_idx
        # capped to len(u) so HMM training never reads OOS.
        capped = min(is_end_idx, len(u))
        return ewmac_returns(u, cfg=cfg, is_end_idx=capped)

    return strategy_fn


def run_cell(
    cell_name,
    cfg,
    closes_visible,
    closes_sanctuary,
    folds,
    is_end_idx,
    *,
    n_trials_sweep,
    bars_per_year,
    sweep_sharpes,
    mc_n_workers,
):
    cls = StrategyClass.CROSS_ASSET_MOMENTUM
    d = defaults_for(cls)
    sh, stitched = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year, is_end_idx)
    is_sh = _is_sharpe(closes_visible, cfg, is_end_idx, bars_per_year)
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
        strategy_fn=_strategy_fn_for_cell(cfg, closes_visible, is_end_idx),
        periods_per_year=bars_per_year,
        seed=42,
        extra_series=extras,
        n_workers=mc_n_workers,
    )
    full = pd.concat([closes_visible, closes_sanctuary])
    sanc_ret = ewmac_returns(full, cfg=cfg, is_end_idx=is_end_idx).iloc[len(closes_visible) :]
    sanc_sh = float(sharpe(sanc_ret, periods_per_year=bars_per_year))
    div = sanctuary_divergence_test(
        historical_returns=stitched,
        sanctuary_returns=sanc_ret,
        periods_per_year=bars_per_year,
    )

    def _noise_fn(df):
        capped = min(is_end_idx, len(df))
        return ewmac_returns(df, cfg=cfg, is_end_idx=capped)

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
        is_sharpe=round(is_sh, 4),
    )


def main():
    print("=" * 72)
    print("I1 -- Per-Asset HMM Regime Gate on Carver EWMAC (L49 escalation)")
    print("Pre-reg: directives/Pre-Reg I1 HMM Per-Asset Regime + EWMAC Gate 2026-05-16.md")
    print("=" * 72)

    closes = load_universe()
    print(
        f"\nUniverse: {len(closes.columns)} instruments  "
        f"{closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)"
    )

    # Causality smoke test using canonical with a generous is_end_idx.
    smoke_cfg = PRE_REG_CELLS[CANONICAL_CELL]

    # ewmac_assert_causal varies the closes array; pass is_end_idx via
    # closure on a custom returns function so the HMM only sees IS.
    def smoke_returns(closes_df, cfg, is_end_idx=int(len(closes) * 0.8)):
        return ewmac_returns(closes_df, cfg=cfg, is_end_idx=is_end_idx)

    # Manual causality probe (date-aligned, similar to carry_assert_causal).
    base_rets = smoke_returns(closes, cfg=smoke_cfg)
    rng = np.random.default_rng(42)
    is_end = int(len(closes) * 0.8)
    for _ in range(2):
        t_corrupt_idx = int(rng.integers(is_end + 50, len(closes) - 5))
        t_corrupt_date = closes.index[t_corrupt_idx]
        corrupt = closes.copy()
        corrupt.loc[corrupt.index >= t_corrupt_date] = (
            corrupt.loc[corrupt.index >= t_corrupt_date] * 1.5
        )
        cor_rets = smoke_returns(corrupt, cfg=smoke_cfg)
        past_base = base_rets.loc[base_rets.index < t_corrupt_date]
        past_cor = cor_rets.loc[cor_rets.index < t_corrupt_date]
        if not past_base.equals(past_cor):
            n_diff = int((past_base != past_cor).sum())
            raise AssertionError(
                f"i1 causality smoke FAILED: future corruption at {t_corrupt_date.date()} "
                f"changed {n_diff} past returns (is_end={is_end})"
            )
    print("Causality smoke test: PASSED")

    sanc = slice_sanctuary(closes, months=12)
    closes_visible = sanc.visible
    closes_sanctuary = sanc.sanctuary
    print(f"Visible: {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}")

    bars_per_year = BARS_PER_YEAR["D"]
    cls_d = defaults_for(StrategyClass.CROSS_ASSET_MOMENTUM)
    folds = build_folds(closes_visible.index, cls_d.wfo, bars_per_year=bars_per_year)
    print(f"WFO: {len(folds)} folds")

    # HMM IS-cutoff: end of the FIRST OOS fold's IS portion, so all
    # subsequent OOS folds use the same frozen HMM. This satisfies the
    # pre-reg's "IS-frozen training; OOS predictions are Viterbi-decoded".
    # In practice = the OOS start of the first fold.
    is_end_idx = int(folds[0].oos_start) if folds else len(closes_visible) // 2
    print(
        f"HMM IS cutoff: row {is_end_idx} of {len(closes_visible)} ({is_end_idx / len(closes_visible) * 100:.0f}% of visible)"
    )

    # ── Baseline reproduction ──────────────────────────────────────────
    print("\n[Baseline reproduction] C5 (filter disabled)")
    sh_c5, _ = _stitched_oos_sharpe(
        closes_visible, PRE_REG_CELLS["C5_no_gate_baseline"], folds, bars_per_year, is_end_idx
    )
    print(f"  C5_no_gate_baseline: Sharpe={sh_c5:+.4f}  (B2b reference: -0.2817)")

    # ── Plateau pre-flight ──────────────────────────────────────────────
    print("\n[Plateau pre-flight] C1 + 4 HMM-param neighbours...")
    plateau_sharpes: dict[str, float] = {}
    sh_c1, _ = _stitched_oos_sharpe(
        closes_visible, PRE_REG_CELLS[CANONICAL_CELL], folds, bars_per_year, is_end_idx
    )
    plateau_sharpes[CANONICAL_CELL] = sh_c1
    for name, cfg in PLATEAU_NEIGHBOURS.items():
        sh, _ = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year, is_end_idx)
        plateau_sharpes[name] = sh
    for n, s in plateau_sharpes.items():
        print(f"  {n}: Sharpe={s:+.4f}")
    vals = list(plateau_sharpes.values())
    mx, mn = max(vals), min(vals)
    rel_spread = abs(mx - mn) / max(abs(mx), 1e-9)
    print(f"  Relative spread: {rel_spread:.2%}  (gate: < 30%)")
    print("  References: B2b 37.10% (FAIL); B2c 3.40% (degenerate); B2d 107.14% (FAIL)")

    plateau_passed = rel_spread <= 0.30

    if not plateau_passed:
        print("\n  [!] PLATEAU GATE FAILED.")
        report = REPORTS_DIR / "result_log.md"
        with report.open("w", encoding="utf-8") as fh:
            fh.write("# I1 Audit -- ABORTED at plateau pre-flight\n\n")
            fh.write(
                f"**Relative spread:** {rel_spread:.2%}  (B2b 37.10%, B2c 3.40%, B2d 107.14%)\n\n"
            )
            fh.write(f"**C5 baseline:** Sharpe={sh_c5:+.4f}\n\n")
            for n, s in plateau_sharpes.items():
                fh.write(f"- {n}: Sharpe = {s:+.4f}\n")
            fh.write(
                "\nFinal escalation exhausted. B2 line of research closed: "
                "broad-index AND per-asset HMM regime gates all fail to rescue "
                "Carver EWMAC on cross-asset 21y data.\n"
            )
        # Even on abort, run the sweep for diagnostic purposes.
        print("\n[Sweep diagnostic — beyond pre-reg]")
        sweep_results: list[tuple[str, float]] = []
        for name, cfg in EXPLORATORY_SWEEP.items():
            sh, _ = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year, is_end_idx)
            sweep_results.append((name, sh))
            print(f"  {name}: Sharpe={sh:+.4f}")
        # Append to log.
        with report.open("a", encoding="utf-8") as fh:
            fh.write("\n## §4.6 Exploratory sweep (BEYOND pre-reg, diagnostic only)\n\n")
            for name, sh in sweep_results:
                fh.write(f"- {name}: Sharpe = {sh:+.4f}\n")
            fh.write(
                "\nNote: these cells are exploratory; not eligible for promotion per V3.1 "
                "(pre-reg honesty).\n"
            )
        return None

    print("  [OK] plateau gate passed.")

    # ── Pass 1: headline OOS Sharpes ─────────────────────────────────────
    all_cells = {**PRE_REG_CELLS, **EXPLORATORY_SWEEP}
    print(
        f"\n[Pass 1/2] Headline OOS Sharpes ({len(all_cells)} cells total: {len(PRE_REG_CELLS)} pre-reg + {len(EXPLORATORY_SWEEP)} sweep)..."
    )
    sweep_sharpes: list[float] = []
    for name, cfg in all_cells.items():
        sh_v, _ = _stitched_oos_sharpe(closes_visible, cfg, folds, bars_per_year, is_end_idx)
        sweep_sharpes.append(sh_v)
        tag = "[PRE-REG]" if name in PRE_REG_CELLS else "[SWEEP]"
        print(f"  {tag} {name}: Sharpe={sh_v:+.4f}")

    # ── Pass 2: full audit (PRE-REG cells only; sweep cells get only headline) ──
    print(
        f"\n[Pass 2/2] Full 5-axis audit on PRE-REG cells -- MC parallel x{DEFAULT_MC_WORKERS}..."
    )
    results: list[CellResult] = []
    for name, cfg in PRE_REG_CELLS.items():
        print(f"\n  > {name}...")
        r = run_cell(
            name,
            cfg,
            closes_visible,
            closes_sanctuary,
            folds,
            is_end_idx,
            n_trials_sweep=len(all_cells),  # DSR penalty includes sweep cells
            bars_per_year=bars_per_year,
            sweep_sharpes=sweep_sharpes,
            mc_n_workers=DEFAULT_MC_WORKERS,
        )
        results.append(r)
        print(
            f"    Sharpe={r.sharpe:+.4f}  CI=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  IS Sharpe={r.is_sharpe:+.4f}"
        )
        print(f"    DSR={r.dsr_prob:.4f}  Verdict (5-axis): {r.verdict}")

    # ── Result log ──────────────────────────────────────────────────────
    report = REPORTS_DIR / "result_log.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# I1 Audit Result Log -- Per-Asset HMM Regime Gate on Carver EWMAC\n\n")
        fh.write("**Run date:** 2026-05-16\n")
        fh.write(f"**Universe:** {len(closes.columns)} instruments (yf_b2b proxies)\n")
        fh.write(
            f"**Data range:** {closes.index[0].date()} -> {closes.index[-1].date()} ({len(closes)} bars)\n"
        )
        fh.write(
            f"**Visible:** {len(closes_visible)}  Sanctuary: {len(closes_sanctuary)}  "
            f"WFO folds: {len(folds)}  HMM IS cutoff: {is_end_idx} bars\n\n"
        )

        fh.write("## §4.1 Baseline reproduction\n\n")
        fh.write(f"- C5_no_gate_baseline: Sharpe={sh_c5:+.4f}  (B2b ref -0.2817)\n")

        fh.write("\n## §4.2 Plateau pre-flight\n\n")
        fh.write(
            f"**Relative spread:** {rel_spread:.2%}  (B2b 37.10%, B2c 3.40%, B2d 107.14%) -- "
            f"{'PASSED' if plateau_passed else 'FAILED'}\n\n"
        )
        for n, s in plateau_sharpes.items():
            fh.write(f"- {n}: Sharpe = {s:+.4f}\n")

        fh.write("\n## §4.3 Pre-reg cell 5-axis matrix\n\n")
        fh.write(
            "| Cell | OOS Sharpe | IS Sharpe | OOS/IS | CI95 lo | CI95 hi | DSR | MC P | "
            "Sanc Sharpe | Noise axis | Verdict |\n"
        )
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|\n")
        for r in results:
            oos_is_ratio = r.sharpe / r.is_sharpe if abs(r.is_sharpe) > 1e-9 else float("nan")
            fh.write(
                f"| {r.cell} | {r.sharpe:+.4f} | {r.is_sharpe:+.4f} | "
                f"{oos_is_ratio:+.2f} | {r.ci_lo:+.3f} | {r.ci_hi:+.3f} | "
                f"{r.dsr_prob:.4f} | {r.mc_p_maxdd_gt_threshold:.4f} | "
                f"{r.sanctuary_sharpe:+.4f} | {r.noise_axis} | {r.verdict} |\n"
            )

        c1 = next(r for r in results if r.cell == "C1_canonical")
        c5 = next(r for r in results if r.cell == "C5_no_gate_baseline")
        seed_alt_sharpe = plateau_sharpes.get("P_seed_alt", float("nan"))

        fh.write("\n## §4.4 Falsification verdicts\n\n")
        fh.write(
            f"**H1 (plateau ≤ 30%):** spread {rel_spread:.2%} → "
            f"{'SUPPORTED' if plateau_passed else 'REJECTED'}.\n\n"
        )
        h2_ok = c1.sharpe >= 0.30
        fh.write(
            f"**H2 (Sharpe ≥ +0.30):** C1 = {c1.sharpe:+.4f} → "
            f"{'SUPPORTED' if h2_ok else 'REJECTED'}.\n\n"
        )
        h3_ok = c1.sharpe > max(-0.28, -0.22, c5.sharpe)
        fh.write(
            f"**H3 (per-asset > broad gates):** I1 C1={c1.sharpe:+.4f} vs "
            f"B2c=-0.2817, B2d=-0.2224, I1 baseline C5={c5.sharpe:+.4f} → "
            f"{'SUPPORTED' if h3_ok else 'REJECTED'}.\n\n"
        )
        if abs(c1.is_sharpe) > 1e-9:
            h4_ok = (c1.sharpe / c1.is_sharpe) >= 0.5
            fh.write(
                f"**H4 (OOS ≥ 50% of IS — overfit check):** OOS={c1.sharpe:+.4f}, "
                f"IS={c1.is_sharpe:+.4f}, ratio={c1.sharpe / c1.is_sharpe:+.2f} → "
                f"{'SUPPORTED' if h4_ok else 'REJECTED (overfit)'}.\n\n"
            )
        else:
            fh.write("**H4 (overfit check):** IS Sharpe ≈ 0, ratio undefined.\n\n")
        h5_ok = abs(c1.sharpe - seed_alt_sharpe) <= max(abs(c1.sharpe) * 0.1, 0.05)
        fh.write(
            f"**H5 (seed stability):** C1={c1.sharpe:+.4f}, P_seed_alt={seed_alt_sharpe:+.4f}, "
            f"|Δ|={abs(c1.sharpe - seed_alt_sharpe):.4f} → "
            f"{'SUPPORTED' if h5_ok else 'REJECTED (HMM init-sensitive)'}.\n"
        )

        fh.write("\n## §4.5 Pass 1 headline summary (all cells incl. sweep)\n\n")
        fh.write("| Cell | Type | OOS Sharpe |\n|---|---|---:|\n")
        for (name, cfg), sh in zip(all_cells.items(), sweep_sharpes):
            tag = "PRE-REG" if name in PRE_REG_CELLS else "SWEEP"
            fh.write(f"| {name} | {tag} | {sh:+.4f} |\n")

        fh.write("\n## §4.6 Exploratory sweep note\n\n")
        fh.write(
            "Six exploratory cells (X1-X6) were run BEYOND the pre-reg's 8 cells. They are "
            "NOT eligible for V3.1 promotion. Their headline Sharpes are included in the "
            "DSR penalty calculation (N=14 cells total). They serve as diagnostic context "
            "for the methodological discussion only.\n"
        )

        deploy_eligible = [
            r
            for r in results
            if r.cell not in ("C5_no_gate_baseline", "C7_gross_no_costs")
            and (
                r.verdict == "DEPLOY"
                or (r.verdict == "CONDITIONAL_WATCHPOINT" and r.noise_axis == "best")
            )
            and r.ci_lo > 0
        ]
        fh.write("\n## §4.7 Promotion verdict\n\n")
        if deploy_eligible:
            best = max(deploy_eligible, key=lambda r: r.ci_lo)
            fh.write(
                f"**PROMOTE {best.cell}** — per-asset HMM regime gate rescues B2: "
                f"CI_lo={best.ci_lo:+.3f}, Sharpe={best.sharpe:+.4f}, verdict={best.verdict}. "
                f"L49 hypothesis confirmed: regime is per-asset.\n"
            )
        else:
            fh.write(
                "No cell promotion-eligible under strict 5-axis + CI_lo>0 rule. "
                "Per-asset HMM regime gate does NOT rescue B2 either. The full B2/B2b/B2c/B2d/I1 "
                "chain — narrow + broad-trend + broad-vol + per-asset HMM — exhausts the "
                "investigation of L48's regime-artifact diagnosis. Universal-trend Carver "
                "EWMAC is RETIRED across all attempted mitigations.\n"
            )

    print(f"\nResult log: {report}")
    return results


if __name__ == "__main__":
    main()
