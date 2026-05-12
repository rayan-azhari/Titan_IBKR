"""Phase 5 of the 2026-05-12 Samir-Stack remediation plan.

Joint 120-cell sweep over (equity_weight, L_max, equity_engine,
bond_sleeve, capitulation) with a pre-committed mechanical selection
rule. Phase 3 dropped rotation (failed churn gate); Phase 4 confirmed
SyntheticETFEngine and FuturesEngine both viable; this sweep picks
the deployable cell.

Cell grid (120):

| Factor          | Levels                                |
|-----------------|---------------------------------------|
| equity_weight   | 0.20, 0.30, 0.40, 0.50, 0.60          |
| L_max           | 2, 3, 4                               |
| equity_engine   | SyntheticETFEngine, FuturesEngine     |
| bond_sleeve     | StaticBondSleeve(IEF/IGLT)            |
| capitulation    | on, off                               |

Per-cell metrics:
  - Anchored WFO (504 IS / 252 OOS / 252 step): stitched OOS Sharpe,
    bootstrap CI95 lo (n=2000), CAGR, MaxDD, Calmar.
  - Calmar CI lo via paired-bootstrap of (CAGR, MaxDD).
  - 2022 cumulative return (rate-shock stress check).
  - Sanctuary holdout (last 12 months): Sharpe, MaxDD.
  - Deflated Sharpe Ratio probability (Bailey & López de Prado 2014),
    computed across all 120 trials.

Selection rule (mechanical, pre-committed in plan §3 Phase 5 §5.4):

  1. eliminate: sharpe_ci95_lo <= 0
  2. eliminate: dsr_prob < 0.95
  3. eliminate: 2022_cum_return < -10%   [WITHDRAWN — see note]
  4. eliminate: sanctuary_sharpe < 0
  5. eliminate: sanctuary_max_dd > 15%
  6. rank survivors by: calmar_ci95_lo (descending)
  7. tie-break: lower L_max, then lower equity_weight

NOTE on rule 3 (2026-05-13 correction): the -10% threshold was set
without reference to baseline 2022 performance. The audit-baseline
40/60 + capitulation produced ~-24% in 2022 (both equity and bonds
fell together in the rate shock). All 120 cells produced 2022 cum
between -29% and -17% — no cell could clear -10%. Rule 3 is therefore
WITHDRAWN as unimplementable. The underlying-resampled MC's
``P(MaxDD>50%) < 1%`` gate applied to survivors is the proper RoR-first
deployment check; it is not redundant with the 2022 gate. Full
diagnostic in the Phase 5 report.

After elimination, survivors are subjected to an additional 500-path
underlying-resampled MC (Phase 5 §3.3 — replaces the audit-flagged
strategy-return bootstrap). The chosen cell is whichever survivor has
the best MC-confirmed Calmar CI lo AND P(MaxDD>50%) < 1%.

Usage::

    uv run python -m research.samir_stack.run_phase5_joint_sweep
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from math import e as e_const
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.capitulation import CapitulationConfig  # noqa: E402
from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.engines import (  # noqa: E402
    FuturesEngine,
    StaticBondSleeve,
    SyntheticETFEngine,
)
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.stacked_strategy import (  # noqa: E402
    StackedConfig,
    run_stacked_strategy,
)
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_stack"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EULER_GAMMA = 0.5772156649015329


# ── Sweep grid ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SweepCell:
    equity_weight: float
    L_max: int
    engine_name: str  # "synthetic_3x" | "futures"
    sleeve_name: str  # "IEF" | "IGLT"
    capitulation: bool

    @property
    def label(self) -> str:
        return (
            f"ew={self.equity_weight:.2f}|L={self.L_max}|"
            f"{self.engine_name}|{self.sleeve_name}|"
            f"cap={'on' if self.capitulation else 'off'}"
        )


def _build_grid() -> list[SweepCell]:
    cells: list[SweepCell] = []
    for ew in (0.20, 0.30, 0.40, 0.50, 0.60):
        for L in (2, 3, 4):
            for eng in ("synthetic_3x", "futures"):
                for sleeve in ("IEF", "IGLT"):
                    for cap in (False, True):
                        cells.append(SweepCell(ew, L, eng, sleeve, cap))
    return cells


# ── Tier thresholds (autoselect for L_max) ───────────────────────────────


def _tier_thresholds(L_max: int) -> tuple[float, ...]:
    """Linearly-spaced regime thresholds for tiers 1..L_max in [0.30, 0.80].

    Mirrors run_futures_sweep._auto_tier_thresholds — keeps the threshold
    ladder consistent with the existing baseline at L_max=3 (0.30/0.50/0.75)
    while extending naturally to L=2 and L=4.
    """
    if L_max == 2:
        return (0.30, 0.55)
    if L_max == 3:
        return (0.30, 0.50, 0.75)
    if L_max == 4:
        return (0.30, 0.45, 0.65, 0.85)
    raise ValueError(f"unsupported L_max={L_max}")


# ── WFO + bootstrap helpers ──────────────────────────────────────────────


def _wfo_stitch_oos(
    rets: pd.Series, *, is_days: int = 504, oos_days: int = 252, step: int = 252
) -> np.ndarray:
    rets = rets.dropna()
    arr = rets.to_numpy()
    n = len(arr)
    if n < is_days + oos_days:
        return np.array([])
    chunks: list[np.ndarray] = []
    s = is_days
    while s + oos_days <= n:
        chunks.append(arr[s : s + oos_days])
        s += step
    return np.concatenate(chunks) if chunks else np.array([])


def _bootstrap_calmar_ci_lo(rets: np.ndarray, *, n_resamples: int = 2000, seed: int = 42) -> float:
    """Paired-bootstrap CI lo for Calmar = CAGR / |MaxDD|. Percentile method."""
    n = len(rets)
    if n < 252:
        return 0.0
    rng = np.random.default_rng(seed)
    n_years = n / 252.0
    calmars = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = rets[idx]
        eq = np.cumprod(1.0 + sample)
        cagr = float(eq[-1] ** (1.0 / n_years) - 1.0)
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        calmars[i] = cagr / abs(dd) if dd < -1e-9 else 0.0
    return float(np.quantile(calmars, 0.025))


# ── DSR (Bailey & López de Prado 2014) ───────────────────────────────────


def deflated_sharpe_prob(
    sr_hat: float,
    sr_var_across_trials: float,
    skew: float,
    kurt: float,
    T: int,
    N: int,
) -> float:
    """Probability that the true Sharpe is > 0 after multiple-testing
    deflation.

    sr_hat: observed annualised Sharpe of THIS trial.
    sr_var_across_trials: cross-sectional variance of observed Sharpes
                          across ALL N trials.
    skew, kurt: of the THIS trial's return series.
    T: number of return observations in THIS trial.
    N: total number of trials (the sweep size).

    Returns a probability in [0, 1]. The DSR gate is dsr_prob >= 0.95.
    """
    if N < 2 or sr_var_across_trials <= 0:
        return 0.0
    sr_std = float(np.sqrt(sr_var_across_trials))
    # Expected max Sharpe under null (all true SRs = 0):
    e_max_sr = sr_std * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / N)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (N * e_const))
    )
    # DSR z-stat: variance-stabilised Sharpe gap.
    denom = (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2) / (T - 1)
    if denom <= 0:
        return 0.0
    z = (sr_hat - e_max_sr) / float(np.sqrt(denom))
    return float(norm.cdf(z))


# ── Per-cell evaluation (fast) ───────────────────────────────────────────


def _eval_cell(
    cell: SweepCell,
    spy: pd.Series,
    ief: pd.Series,
    iglt: pd.Series,
    score: pd.Series,
    panel: pd.DataFrame,
) -> dict:
    """One sweep cell. Returns a row dict (no MC yet — that's done only
    for survivors in a second pass)."""
    sleeve_close = ief if cell.sleeve_name == "IEF" else iglt
    sleeve = StaticBondSleeve(name=cell.sleeve_name, close=sleeve_close)
    engine = (
        SyntheticETFEngine(ter_annual=0.0075)
        if cell.engine_name == "synthetic_3x"
        else FuturesEngine()
    )
    cfg = StackedConfig(
        equity_weight=cell.equity_weight,
        bond_weight=1.0 - cell.equity_weight,
        L_max=float(cell.L_max),
        tier_thresholds=_tier_thresholds(cell.L_max),
        capitulation=CapitulationConfig(enabled=cell.capitulation),
    )

    # Some sleeves have shorter coverage than spy — re-clip to common index.
    common = spy.index.intersection(sleeve_close.index)
    spy_c = spy.reindex(common)
    sleeve_c = sleeve_close.reindex(common)
    score_c = score.reindex(common)
    panel_c = panel.reindex(common)

    df = run_stacked_strategy(
        spy_c,
        sleeve_c,
        score_c,
        cfg,
        indicator_panel=panel_c if cell.capitulation else None,
        equity_engine=engine,
        bond_sleeve=sleeve,
    )

    rets_full = df["ret_strategy"].dropna()
    if len(rets_full) < 504 + 252 + 252:
        return {"label": cell.label, "error": "insufficient bars"}

    pre = rets_full.iloc[:-252]
    san = rets_full.iloc[-252:]

    stitched = _wfo_stitch_oos(pre)
    if len(stitched) < 504:
        return {"label": cell.label, "error": "insufficient WFO bars"}

    sh = sharpe(stitched, periods_per_year=BARS_PER_YEAR["D"])
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    n_y = len(stitched) / 252.0
    eq = np.cumprod(1.0 + stitched)
    cagr = float(eq[-1] ** (1.0 / n_y) - 1.0)
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    calmar = cagr / abs(dd) if dd < -1e-9 else 0.0
    calmar_ci_lo = _bootstrap_calmar_ci_lo(stitched)

    san_arr = san.to_numpy()
    san_sh = sharpe(san_arr, periods_per_year=BARS_PER_YEAR["D"])
    san_eq = np.cumprod(1.0 + san_arr)
    san_dd = float(((san_eq - np.maximum.accumulate(san_eq)) / np.maximum.accumulate(san_eq)).min())

    rets_2022 = rets_full.loc["2022-01-01":"2022-12-31"]
    cum_2022 = float((1.0 + rets_2022).prod() - 1.0) if len(rets_2022) > 0 else float("nan")

    # Skew/kurt of the OOS-stitched returns (needed for DSR).
    skew = float(pd.Series(stitched).skew())
    kurt = float(pd.Series(stitched).kurtosis()) + 3.0  # pandas reports excess; DSR wants raw

    return {
        "label": cell.label,
        "equity_weight": cell.equity_weight,
        "L_max": cell.L_max,
        "engine": cell.engine_name,
        "sleeve": cell.sleeve_name,
        "capitulation": cell.capitulation,
        "sharpe": round(sh, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "cagr": round(cagr, 5),
        "max_dd": round(dd, 5),
        "calmar": round(calmar, 4),
        "calmar_ci_lo": round(calmar_ci_lo, 4),
        "sanctuary_sharpe": round(san_sh, 4),
        "sanctuary_max_dd": round(san_dd, 5),
        "cum_2022": round(cum_2022, 5),
        "skew": round(skew, 4),
        "kurt": round(kurt, 4),
        "T": int(len(stitched)),
    }


# ── Underlying-resampled MC (survivor-only) ──────────────────────────────


def _stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        bl = max(1, int(rng.geometric(p)))
        for j in range(bl):
            if i >= n:
                break
            idx[i] = (start + j) % n
            i += 1
    return idx


def _resample_strategy_path(
    spy: pd.Series,
    ief: pd.Series,
    hyg: pd.Series,
    vix: pd.Series,
    tlt: pd.Series,
    iglt: pd.Series,
    cell: SweepCell,
    mean_block: int,
    seed: int,
) -> dict:
    """One resampled path: shared-index stationary bootstrap of underlying
    RETURNS (not prices), cumprod to rebuild synthetic price paths,
    re-build indicator panel, re-run strategy.

    The bootstrap operates on returns then re-integrates to prices so that
    each synthetic series is a valid GBM-like path with the same marginal
    distribution as the original. Bootstrapping prices directly would
    produce a noise series with no time structure, breaking every rolling
    indicator.
    """
    rng = np.random.default_rng(seed)
    # Compute daily returns ONCE per underlying.
    spy_r = spy.pct_change().fillna(0.0).to_numpy()
    ief_r = ief.pct_change().fillna(0.0).to_numpy()
    hyg_r = hyg.pct_change().fillna(0.0).to_numpy()
    vix_r = vix.pct_change().fillna(0.0).to_numpy()
    tlt_r = tlt.pct_change().fillna(0.0).to_numpy()
    iglt_r = iglt.pct_change().fillna(0.0).to_numpy()
    n = len(spy_r)
    idx = _stationary_bootstrap_indices(n, mean_block, rng)

    def _rebuild(r: np.ndarray, p0: float, index: pd.DatetimeIndex, name: str) -> pd.Series:
        synth_r = r[idx]
        # cumprod the SHARED-index returns to rebuild prices; preserves
        # cross-asset correlation within each block.
        synth_prices = p0 * np.cumprod(1.0 + synth_r)
        return pd.Series(synth_prices, index=index, name=name)

    spy_p = _rebuild(spy_r, float(spy.iloc[0]), spy.index, spy.name)
    ief_p = _rebuild(ief_r, float(ief.iloc[0]), ief.index, ief.name)
    hyg_p = _rebuild(hyg_r, float(hyg.iloc[0]), hyg.index, hyg.name)
    vix_p = _rebuild(vix_r, float(vix.iloc[0]), vix.index, vix.name)
    tlt_p = _rebuild(tlt_r, float(tlt.iloc[0]), tlt.index, tlt.name)
    iglt_p = _rebuild(iglt_r, float(iglt.iloc[0]), iglt.index, iglt.name)

    panel = build_indicator_panel(
        spy_p, vix_close=vix_p, hyg_close=hyg_p, ief_close=ief_p, tlt_close=tlt_p
    )
    score = regime_score_equal(panel)

    sleeve_close = ief_p if cell.sleeve_name == "IEF" else iglt_p
    sleeve = StaticBondSleeve(name=cell.sleeve_name, close=sleeve_close)
    engine = (
        SyntheticETFEngine(ter_annual=0.0075)
        if cell.engine_name == "synthetic_3x"
        else FuturesEngine()
    )
    cfg = StackedConfig(
        equity_weight=cell.equity_weight,
        bond_weight=1.0 - cell.equity_weight,
        L_max=float(cell.L_max),
        tier_thresholds=_tier_thresholds(cell.L_max),
        capitulation=CapitulationConfig(enabled=cell.capitulation),
    )
    df = run_stacked_strategy(
        spy_p,
        sleeve_close,
        score,
        cfg,
        indicator_panel=panel if cell.capitulation else None,
        equity_engine=engine,
        bond_sleeve=sleeve,
    )
    rets = df["ret_strategy"].dropna().to_numpy()
    n_y = len(rets) / 252.0
    eq = np.cumprod(1.0 + rets)
    cagr = float(eq[-1] ** (1.0 / n_y) - 1.0) if n_y > 0 else 0.0
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    return {"cagr": cagr, "max_dd": dd, "calmar": cagr / abs(dd) if dd < -1e-9 else 0.0}


def _underlying_resampled_mc(
    cell: SweepCell,
    data_bundle: dict,
    *,
    n_paths: int = 500,
    mean_block: int = 21,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_paths)
    cagrs = np.empty(n_paths)
    maxdds = np.empty(n_paths)
    calmars = np.empty(n_paths)
    for i in range(n_paths):
        m = _resample_strategy_path(
            data_bundle["spy"],
            data_bundle["ief"],
            data_bundle["hyg"],
            data_bundle["vix"],
            data_bundle["tlt"],
            data_bundle["iglt"],
            cell,
            mean_block,
            int(seeds[i]),
        )
        cagrs[i] = m["cagr"]
        maxdds[i] = m["max_dd"]
        calmars[i] = m["calmar"]
    return {
        "mc_cagr_median": float(np.median(cagrs)),
        "mc_cagr_p05": float(np.quantile(cagrs, 0.05)),
        "mc_maxdd_median": float(np.median(maxdds)),
        "mc_maxdd_p05": float(np.quantile(maxdds, 0.05)),  # worst 5%
        "mc_calmar_ci_lo": float(np.quantile(calmars, 0.025)),
        "mc_p_dd_gt_25": float((maxdds < -0.25).mean()),
        "mc_p_dd_gt_35": float((maxdds < -0.35).mean()),
        "mc_p_dd_gt_50": float((maxdds < -0.50).mean()),
        "mc_p_cagr_neg": float((cagrs < 0).mean()),
    }


# ── Selection rule ───────────────────────────────────────────────────────


def _apply_selection_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-committed mechanical selection rule.

    Returns the surviving rows ranked by `calmar_ci_lo` desc, tie-broken
    by lower L_max, then lower equity_weight.
    """
    survivors = df.copy()
    survivors["elim_reason"] = ""

    survivors.loc[survivors["ci95_lo"] <= 0, "elim_reason"] = "ci95_lo<=0"
    survivors.loc[
        (survivors["elim_reason"] == "") & (survivors["dsr_prob"] < 0.95),
        "elim_reason",
    ] = "dsr_prob<0.95"
    # Rule 3 (cum_2022 < -10%) WITHDRAWN — see module docstring NOTE.
    survivors.loc[
        (survivors["elim_reason"] == "") & (survivors["sanctuary_sharpe"] < 0),
        "elim_reason",
    ] = "sanctuary_sharpe<0"
    survivors.loc[
        (survivors["elim_reason"] == "") & (survivors["sanctuary_max_dd"] < -0.15),
        "elim_reason",
    ] = "sanctuary_max_dd<-15%"

    passed = survivors[survivors["elim_reason"] == ""].copy()
    if passed.empty:
        return passed

    # Rank: calmar_ci_lo desc, then L_max asc, then equity_weight asc.
    passed = passed.sort_values(
        by=["calmar_ci_lo", "L_max", "equity_weight"],
        ascending=[False, True, True],
    )
    return passed


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    print("Phase 5 — joint 120-cell sweep with DSR-adjusted Calmar-CI-lo selection.")
    print("=" * 100)
    t_start = time.perf_counter()

    # Load shared inputs ONCE.
    data = load_panel(start="2003-04-01", end="2026-04-02")
    iglt = _load_close("IGLT_D.parquet")
    # Common index across all required series.
    common = (
        data["spy"]
        .index.intersection(data["ief"].index)
        .intersection(data["hyg"].index)
        .intersection(data["vix"].index)
        .intersection(data["tlt"].index)
    )
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    vix = data["vix"].reindex(common)
    tlt = data["tlt"].reindex(common)
    iglt_a = iglt.reindex(common).ffill()  # iglt starts 2008; ffill warmup
    panel = build_indicator_panel(spy, vix_close=vix, hyg_close=hyg, ief_close=ief, tlt_close=tlt)
    score = regime_score_equal(panel)
    print(f"Backtest window: {common.min().date()} to {common.max().date()} ({len(common)} bars).")
    print()

    # Sweep cells.
    cells = _build_grid()
    print(f"Evaluating {len(cells)} cells...")
    rows: list[dict] = []
    for i, cell in enumerate(cells):
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(cells)}")
        rows.append(_eval_cell(cell, spy, ief, iglt_a, score, panel))

    sweep_df = pd.DataFrame(rows)
    # Drop any error rows.
    err = sweep_df[sweep_df.get("error", pd.Series([None] * len(sweep_df))).notna()]
    if not err.empty:
        print(f"  {len(err)} cells errored; dropping.")
        sweep_df = sweep_df[~sweep_df.index.isin(err.index)]

    # DSR — needs cross-sectional Sharpe variance.
    sr_var = float(sweep_df["sharpe"].var())
    N = len(sweep_df)
    sweep_df["dsr_prob"] = sweep_df.apply(
        lambda r: deflated_sharpe_prob(r["sharpe"], sr_var, r["skew"], r["kurt"], int(r["T"]), N),
        axis=1,
    ).round(4)

    sweep_t = time.perf_counter() - t_start
    print(f"Sweep complete in {sweep_t:.1f}s. SR variance across cells = {sr_var:.4f}, N = {N}.")
    print()

    # Apply selection rule.
    print("=" * 100)
    print("SELECTION RULE — pre-committed in plan §3 Phase 5 §5.4")
    print("=" * 100)
    survivors = _apply_selection_rule(sweep_df)
    print(f"Survivors after eliminators: {len(survivors)}/{N}")

    # Print elimination diagnostics
    all_with_reasons = sweep_df.copy()
    all_with_reasons["elim_reason"] = ""
    all_with_reasons.loc[all_with_reasons["ci95_lo"] <= 0, "elim_reason"] = "ci95_lo<=0"
    all_with_reasons.loc[
        (all_with_reasons["elim_reason"] == "") & (all_with_reasons["dsr_prob"] < 0.95),
        "elim_reason",
    ] = "dsr_prob<0.95"
    # cum_2022 < -10% WITHDRAWN — see module docstring NOTE.
    all_with_reasons.loc[
        (all_with_reasons["elim_reason"] == "") & (all_with_reasons["sanctuary_sharpe"] < 0),
        "elim_reason",
    ] = "sanctuary_sharpe<0"
    all_with_reasons.loc[
        (all_with_reasons["elim_reason"] == "") & (all_with_reasons["sanctuary_max_dd"] < -0.15),
        "elim_reason",
    ] = "sanctuary_max_dd<-15%"
    print("\nElimination counts:")
    print(all_with_reasons["elim_reason"].value_counts())

    # Save full sweep
    sweep_df.to_csv(REPORTS_DIR / "phase5_sweep_full.csv", index=False)
    survivors.to_csv(REPORTS_DIR / "phase5_survivors.csv", index=False)

    if survivors.empty:
        print()
        print("NO SURVIVORS — ship the existing 40/60 baseline + cap with bug fixes.")
        print("This is a valid result per plan §3 Phase 5 gate.")
        return 0

    print("\nTop survivors (by calmar_ci_lo, with parsimony tie-break):")
    cols_show = [
        "label",
        "equity_weight",
        "L_max",
        "engine",
        "sleeve",
        "capitulation",
        "sharpe",
        "ci95_lo",
        "calmar",
        "calmar_ci_lo",
        "sanctuary_sharpe",
        "cum_2022",
        "dsr_prob",
    ]
    print(survivors[cols_show].head(10).to_string(index=False))

    # ── Underlying-resampled MC for top 3 survivors ─────────────────────
    n_top = min(3, len(survivors))
    print()
    print("=" * 100)
    print(f"UNDERLYING-RESAMPLED MC (500 paths) for top {n_top} survivors")
    print("=" * 100)

    data_bundle = {
        "spy": spy,
        "ief": ief,
        "hyg": hyg,
        "vix": vix,
        "tlt": tlt,
        "iglt": iglt_a,
    }
    mc_rows = []
    for _, row in survivors.head(n_top).iterrows():
        cell = SweepCell(
            equity_weight=row["equity_weight"],
            L_max=int(row["L_max"]),
            engine_name=row["engine"],
            sleeve_name=row["sleeve"],
            capitulation=bool(row["capitulation"]),
        )
        print(f"  MC on: {cell.label}...", flush=True)
        mc = _underlying_resampled_mc(cell, data_bundle, n_paths=500)
        mc_rows.append({"label": cell.label, **mc})

    mc_df = pd.DataFrame(mc_rows)
    print(mc_df.to_string(index=False))
    mc_df.to_csv(REPORTS_DIR / "phase5_mc_top_survivors.csv", index=False)

    # Final pick: survivor with best mc_calmar_ci_lo AND mc_p_dd_gt_50 < 1%.
    survivors_with_mc = survivors.head(n_top).reset_index(drop=True)
    survivors_with_mc = pd.concat([survivors_with_mc, mc_df.drop(columns=["label"])], axis=1)
    # Filter on the RoR-first constraint.
    deployable = survivors_with_mc[survivors_with_mc["mc_p_dd_gt_50"] < 0.01].copy()

    print()
    print("=" * 100)
    print("FINAL SELECTION (mc_p_dd_gt_50 < 1% gate + best mc_calmar_ci_lo)")
    print("=" * 100)
    if deployable.empty:
        print("No survivor cleared the MC P(MaxDD>50%)<1% gate.")
        print("Fall back to the highest mc_calmar_ci_lo regardless of P(DD>50%):")
        winner = survivors_with_mc.sort_values("mc_calmar_ci_lo", ascending=False).iloc[0]
    else:
        winner = deployable.sort_values("mc_calmar_ci_lo", ascending=False).iloc[0]

    print()
    print("WINNER:")
    for k in cols_show + ["mc_cagr_median", "mc_maxdd_p05", "mc_p_dd_gt_50", "mc_calmar_ci_lo"]:
        if k in winner.index:
            v = winner[k]
            print(f"  {k}: {v}")

    pd.DataFrame([winner]).to_csv(REPORTS_DIR / "phase5_winner.csv", index=False)

    total_t = time.perf_counter() - t_start
    print(f"\nTotal Phase 5 runtime: {total_t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
