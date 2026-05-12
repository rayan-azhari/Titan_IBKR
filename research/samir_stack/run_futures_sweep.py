"""Samir-Stack equity-engine sweep: ETF margin vs CFD vs MES futures.

Two comparisons (both 16-fold rolling WFO with bootstrap CI gate):

  Part A — Engine-swap at fixed strategy config (L_max=2, 15/85 split):
    - CSPX margin drift L=2 (current champion baseline)
    - IBKR US500 CFD L=2
    - MES futures L=2
    Isolates the COST-MODEL impact of changing the equity-engine while
    holding strategy mechanics constant. The diff between rows is purely
    engine carrying cost (margin interest vs basis carry vs financing).

  Part B — Futures leverage sweep at fixed 15/85 split:
    - MES futures L=2, 3, 4, 5, 6, 8, 10
    Shows what happens if you crank engine leverage on the same strategy.
    tier_thresholds are auto-generated to span [0.30, 0.80] for the
    requested number of tiers; equity_weight stays at 0.15 throughout
    (so peak notional = L_max × 0.15 = up to 1.5× capital).

Output: per-engine WFO summary + aggregate stitched stats with bootstrap
CIs, plus per-fold CSV. Compare costs by reading off the CAGR/Sharpe
delta vs the CSPX L=2 baseline.

Usage:
    uv run python research/samir_stack/run_futures_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.margin_model import (  # noqa: E402
    cfd_returns,
    drift_margin_returns,
    futures_returns,
)
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.run_samir_improvements import (  # noqa: E402
    bond_rotation_returns,
    compose_with_rate_shock,
)
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


# ── Equity-engine factories (match the (spy, leverage) signature) ─────────


def _engine_cspx_margin_drift(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return (
        drift_margin_returns(spy, initial_leverage=leverage, broker="ibkr_pro")["equity_ret"]
        .reindex(spy.index)
        .fillna(0.0)
    )


def _engine_cfd(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return cfd_returns(spy, leverage=leverage).reindex(spy.index).fillna(0.0)


def _engine_futures(spy: pd.Series, leverage: float) -> pd.Series:
    if leverage <= 0:
        return pd.Series(0.0, index=spy.index)
    return futures_returns(spy, leverage=leverage).reindex(spy.index).fillna(0.0)


# ── Strategy runner ──────────────────────────────────────────────────────


def _auto_tier_thresholds(L_max: int) -> tuple[float, ...]:
    """Linearly-spaced regime thresholds for tiers 1..L_max in [0.30, 0.80]."""
    if L_max <= 1:
        return (0.30,)
    if L_max == 2:
        return (0.30, 0.50)
    if L_max == 3:
        return (0.30, 0.50, 0.75)  # matches existing champion default
    return tuple(round(0.30 + (0.80 - 0.30) * k / (L_max - 1), 3) for k in range(L_max))


def run_with_engine(
    spy: pd.Series,
    efa: pd.Series,
    ief: pd.Series,
    hyg: pd.Series,
    tlt: pd.Series,
    samir_score: pd.Series,
    *,
    equity_engine,
    L_max: float,
    equity_weight: float = 0.15,
    bond_weight: float = 0.85,
) -> pd.DataFrame:
    """Run improved Samir-Stack (I1+I2+I3) with the chosen engine and L_max."""
    common = (
        spy.index.intersection(efa.index)
        .intersection(ief.index)
        .intersection(hyg.index)
        .intersection(tlt.index)
        .intersection(samir_score.index)
    )
    spy_a = spy.reindex(common)
    efa_a = efa.reindex(common)
    ief_a = ief.reindex(common)
    hyg_a = hyg.reindex(common)
    tlt_a = tlt.reindex(common)
    samir_a = samir_score.reindex(common)

    # I3: opt-in EFA
    spy_ret = spy_a.pct_change().fillna(0.0)
    efa_ret = efa_a.pct_change().fillna(0.0)
    spy_12m = spy_a.pct_change(252)
    efa_12m = efa_a.pct_change(252)
    use_efa_mask = (efa_12m - spy_12m) > 0.05
    chosen_ret = pd.Series(
        np.where(use_efa_mask.fillna(False), efa_ret.values, spy_ret.values),
        index=common,
    )
    equity_underlying = (1.0 + chosen_ret).cumprod() * float(spy_a.iloc[0])

    # I2: bond rotation
    rot_rets = bond_rotation_returns(ief_a, hyg_a)
    bond_underlying = (1.0 + rot_rets.reindex(common).fillna(0.0)).cumprod() * float(ief_a.iloc[0])

    # I1: rate-shock score blend
    score = compose_with_rate_shock(samir_a, tlt_a)

    # Monkey-patch the equity engine into stacked_strategy's namespace
    import research.samir_stack.stacked_strategy as ss_mod

    saved = ss_mod.synthetic_leveraged_returns

    def patched(spy_series, leverage, **_kwargs):
        return equity_engine(spy_series, leverage)

    ss_mod.synthetic_leveraged_returns = patched

    cfg = StackedConfig(
        equity_weight=equity_weight,
        bond_weight=bond_weight,
        L_max=L_max,
        tier_thresholds=_auto_tier_thresholds(int(L_max)),
    )
    try:
        return run_stacked_strategy(equity_underlying, bond_underlying, score, cfg)
    finally:
        ss_mod.synthetic_leveraged_returns = saved


# ── Stats helpers ────────────────────────────────────────────────────────


def _fold_stats(rets: np.ndarray) -> dict:
    if len(rets) < 20:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0}
    sh = sharpe(rets, periods_per_year=BARS_PER_YEAR["D"])
    eq = np.cumprod(1.0 + rets)
    n_years = len(rets) / 252.0
    cagr = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {"sharpe": round(sh, 3), "cagr": round(cagr, 4), "max_dd": round(maxdd, 4)}


def _wfo_stitch(
    df: pd.DataFrame, *, is_days: int = 504, oos_days: int = 252, step: int = 252
) -> tuple[np.ndarray, list[dict]]:
    """Slice a full-period equity curve into 16 OOS folds and stitch returns."""
    n = len(df)
    if n < is_days + oos_days:
        return np.array([]), []
    rets = df["ret_strategy"].to_numpy()
    idx = df.index
    fold_rows: list[dict] = []
    stitched: list[np.ndarray] = []
    fold_idx = 0
    oos_start = is_days
    while oos_start + oos_days <= n:
        oos_end = oos_start + oos_days
        slice_rets = rets[oos_start:oos_end]
        stats = _fold_stats(slice_rets)
        fold_rows.append(
            {
                "fold": fold_idx,
                "oos_start": idx[oos_start].strftime("%Y-%m-%d"),
                "oos_end": idx[oos_end - 1].strftime("%Y-%m-%d"),
                **stats,
            }
        )
        stitched.append(slice_rets)
        fold_idx += 1
        oos_start += step
    return np.concatenate(stitched) if stitched else np.array([]), fold_rows


def _summarise(label: str, df: pd.DataFrame) -> dict:
    stitched, fold_rows = _wfo_stitch(df)
    if len(stitched) == 0:
        return {"engine": label, "n_oos_years": 0.0}
    sh = sharpe(stitched, periods_per_year=BARS_PER_YEAR["D"])
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    n_years = len(stitched) / 252.0
    eq = np.cumprod(1.0 + stitched)
    cagr = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    pos_pct = float(np.mean([f["sharpe"] > 0 for f in fold_rows])) if fold_rows else 0.0
    return {
        "engine": label,
        "n_oos_years": round(n_years, 2),
        "stitched_sharpe": round(sh, 3),
        "ci95_lo": round(ci_lo, 3),
        "ci95_hi": round(ci_hi, 3),
        "stitched_cagr": round(cagr, 4),
        "stitched_max_dd": round(dd, 4),
        "pct_pos_folds": round(pos_pct, 3),
        "passes_gate": ci_lo > 0,
    }


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    print("Loading data...", flush=True)
    data = load_panel(start="2003-04-01", end="2026-04-02")
    efa = _load_close("EFA_D.parquet")
    common = (
        data["spy"]
        .index.intersection(efa.index)
        .intersection(data["tlt"].index)
        .intersection(data["hyg"].index)
        .intersection(data["ief"].index)
    )
    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)
    efa = efa.reindex(common)
    print(
        f"Range: {common.min().date()} -> {common.max().date()} ({len(common)} bars)\n",
        flush=True,
    )

    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"].reindex(common),
        hyg_close=hyg,
        ief_close=ief,
        tlt_close=tlt,
    )
    samir_score = regime_score_equal(panel)

    # ── Part A: engine-swap at L_max=2, 15/85 split ──────────────────────
    print("Part A: engine-swap at L_max=2, 15/85 split", flush=True)
    print("-" * 70)

    part_a_cases = [
        ("CSPX margin drift L=2", _engine_cspx_margin_drift, 2.0),
        ("IBKR US500 CFD L=2", _engine_cfd, 2.0),
        ("MES futures L=2", _engine_futures, 2.0),
    ]
    rows_a: list[dict] = []
    for label, engine, L_max in part_a_cases:
        print(f"  {label}...", flush=True)
        df = run_with_engine(
            spy,
            efa,
            ief,
            hyg,
            tlt,
            samir_score,
            equity_engine=engine,
            L_max=L_max,
            equity_weight=0.15,
            bond_weight=0.85,
        )
        rows_a.append(_summarise(label, df))

    summary_a = pd.DataFrame(rows_a).set_index("engine")
    print()
    print("=" * 90)
    print("PART A — Engine-swap WFO summary (16 folds, L_max=2, 15/85 split)")
    print("=" * 90)
    print(summary_a.to_string())
    summary_a.to_csv(REPORTS_DIR / "futures_sweep_part_a.csv")

    # ── Part B: futures leverage sweep at 15/85 split ────────────────────
    print()
    print("Part B: MES futures leverage sweep at 15/85 split")
    print("-" * 70)
    L_grid = [2, 3, 4, 5, 6, 8, 10]
    rows_b: list[dict] = []
    for L_max in L_grid:
        label = f"MES futures L={L_max}"
        print(f"  {label} (tier_thresholds={_auto_tier_thresholds(L_max)})...", flush=True)
        df = run_with_engine(
            spy,
            efa,
            ief,
            hyg,
            tlt,
            samir_score,
            equity_engine=_engine_futures,
            L_max=float(L_max),
            equity_weight=0.15,
            bond_weight=0.85,
        )
        rows_b.append(_summarise(label, df))

    summary_b = pd.DataFrame(rows_b).set_index("engine")
    print()
    print("=" * 90)
    print("PART B — MES futures leverage sweep (16-fold WFO, 15/85 split)")
    print("=" * 90)
    print(summary_b.to_string())
    summary_b.to_csv(REPORTS_DIR / "futures_sweep_part_b.csv")

    # ── Part D: constant-notional sweep ─────────────────────────────────
    # Holds equity NOTIONAL (= L × equity_weight) at 30% while freeing
    # capital for the bond sleeve. Tests the question: does the
    # allocation-sweep finding (more bonds = better Sharpe) compose with
    # higher futures leverage to produce a net win?
    print()
    print("Part D: constant-notional sweep (equity notional = 30%, vary L vs equity_weight)")
    print("-" * 70)
    constant_notional_grid = [
        (2.0, 0.150, 0.850),
        (3.0, 0.100, 0.900),
        (4.0, 0.075, 0.925),
        (5.0, 0.060, 0.940),
        (6.0, 0.050, 0.950),
        (8.0, 0.0375, 0.9625),
        (10.0, 0.030, 0.970),
    ]
    rows_d: list[dict] = []
    for L_max, eq_w, bd_w in constant_notional_grid:
        label = f"MES L={int(L_max)} eq={eq_w:.3f} bd={bd_w:.3f}"
        print(f"  {label}...", flush=True)
        df = run_with_engine(
            spy,
            efa,
            ief,
            hyg,
            tlt,
            samir_score,
            equity_engine=_engine_futures,
            L_max=L_max,
            equity_weight=eq_w,
            bond_weight=bd_w,
        )
        rec = _summarise(label, df)
        rec["equity_notional_pct"] = round(L_max * eq_w * 100, 1)
        rec["bond_capital_pct"] = round(bd_w * 100, 1)
        rows_d.append(rec)
    summary_d = pd.DataFrame(rows_d).set_index("engine")
    print()
    print("=" * 90)
    print("PART D — Constant-notional sweep (equity notional = 30%, more bonds at higher L)")
    print("=" * 90)
    print(summary_d.to_string())
    summary_d.to_csv(REPORTS_DIR / "futures_sweep_part_d.csv")

    # ── Part C: cost-only diagnostic at L=2 ─────────────────────────────
    # Year-1 cost decomposition for the equity sleeve. Keeps the user
    # honest about WHERE the engine differences come from.
    print()
    print("=" * 90)
    print("PART C — Year-1 equity-sleeve cost decomposition at L=2 (synthetic)")
    print("=" * 90)
    spy_test = spy.iloc[-260:]  # Last ~1y of data
    cost_rows = []
    for label, engine, L in part_a_cases:
        rets = engine(spy_test, L)
        spy_rets = spy_test.pct_change().dropna()
        gross_levered = (L * spy_rets).sum()
        engine_total = rets.dropna().sum()
        cost = gross_levered - engine_total
        cost_rows.append(
            {
                "engine": label,
                "spy_ret_1y": round(spy_rets.sum() * 100, 2),
                "gross_levered_ret_1y_pct": round(gross_levered * 100, 2),
                "engine_net_ret_1y_pct": round(engine_total * 100, 2),
                "implied_carry_cost_1y_pct": round(cost * 100, 2),
            }
        )
    print(pd.DataFrame(cost_rows).set_index("engine").to_string())

    print(f"\nSaved: {REPORTS_DIR / 'futures_sweep_part_a.csv'}")
    print(f"Saved: {REPORTS_DIR / 'futures_sweep_part_b.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
