"""Phase 3 of the 2026-05-12 Samir-Stack remediation plan.

Head-to-head test of three bond sleeves on the 40/60 + capitulation
baseline, using ONLY the new engine/sleeve abstractions (no
monkey-patching, no synthetic-close construction):

  1. ``StaticBondSleeve(IEF)``        — USD 7-10y Treasuries (reference).
  2. ``StaticBondSleeve(IGLT)``       — UK 7-10y gilts (GBP-clean baseline).
  3. ``RotationBondSleeve([IGLT, IGLS])``
                                       — UK gilts rotation per operator
                                         decision §0(2) (GBP-clean
                                         momentum-based rotation between
                                         long and short duration plus
                                         cash fallback).

For each sleeve the runner:
  - Computes anchored-WFO stitched OOS Sharpe with bootstrap CI lo (95%).
  - Reports CAGR, MaxDD, Calmar, sanctuary (last 12 months) Sharpe/MaxDD.
  - For the rotation sleeve specifically: counts flips/year and reports
    cumulative 2022 return (the key rate-shock test case).

Phase 3 gate (pre-committed in the remediation plan §3 Phase 3):
  1. Rotation variant has ``ci95_lo > 0``.
  2. Rotation 2022 cum return better than static-IEF baseline.
  3. Rotation churn ≤ 4 flips/year.
  4. Rotation sanctuary OOS Sharpe ≥ 0.

If rotation fails ANY gate, the bond sleeve stays STATIC in Phase 5.

Usage::

    uv run python -m research.samir_stack.run_phase3_bond_rotation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.capitulation import CapitulationConfig  # noqa: E402
from research.samir_stack.data_loader import _load_close, load_panel  # noqa: E402
from research.samir_stack.engines import (  # noqa: E402
    RotationBondSleeve,
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


# ── WFO helpers ──────────────────────────────────────────────────────────


def _wfo_stitch(
    rets: pd.Series, *, is_days: int = 504, oos_days: int = 252, step: int = 252
) -> tuple[np.ndarray, list[dict]]:
    """Anchored-WFO stitched OOS returns.

    IS-window length is informational only (the state machine has no
    per-fold tunable parameters — this is a stability scan with bootstrap
    CI as the deployment gate).
    """
    rets = rets.dropna()
    n = len(rets)
    if n < is_days + oos_days:
        return np.array([]), []
    arr = rets.to_numpy()
    idx = rets.index
    fold_rows: list[dict] = []
    stitched: list[np.ndarray] = []
    fold_idx = 0
    oos_start = is_days
    while oos_start + oos_days <= n:
        oos_end = oos_start + oos_days
        slice_rets = arr[oos_start:oos_end]
        eq = np.cumprod(1.0 + slice_rets)
        peak = np.maximum.accumulate(eq)
        maxdd = float(((eq - peak) / peak).min())
        fold_rows.append(
            {
                "fold": fold_idx,
                "oos_start": idx[oos_start].strftime("%Y-%m-%d"),
                "oos_end": idx[oos_end - 1].strftime("%Y-%m-%d"),
                "sharpe": round(sharpe(slice_rets, periods_per_year=BARS_PER_YEAR["D"]), 3),
                "max_dd": round(maxdd, 4),
            }
        )
        stitched.append(slice_rets)
        fold_idx += 1
        oos_start += step
    return (np.concatenate(stitched) if stitched else np.array([])), fold_rows


def _summary(label: str, df: pd.DataFrame, *, sanctuary_days: int = 252) -> dict:
    """Compute the headline metrics for a sleeve's run."""
    rets_full = df["ret_strategy"].dropna()
    # Split: pre-sanctuary (used for WFO) and sanctuary (last `sanctuary_days`).
    if len(rets_full) < sanctuary_days + 252:
        return {"sleeve": label, "error": f"too few bars ({len(rets_full)})"}
    pre = rets_full.iloc[:-sanctuary_days]
    san = rets_full.iloc[-sanctuary_days:]

    stitched, _folds = _wfo_stitch(pre)
    if len(stitched) == 0:
        return {"sleeve": label, "error": "insufficient for WFO"}

    sh = sharpe(stitched, periods_per_year=BARS_PER_YEAR["D"])
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    n_years = len(stitched) / 252.0
    eq = np.cumprod(1.0 + stitched)
    cagr = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    calmar = cagr / abs(dd) if dd < -1e-9 else 0.0

    # Sanctuary
    san_arr = san.to_numpy()
    san_sh = sharpe(san_arr, periods_per_year=BARS_PER_YEAR["D"]) if len(san_arr) > 1 else 0.0
    san_eq = np.cumprod(1.0 + san_arr)
    san_dd = float(((san_eq - np.maximum.accumulate(san_eq)) / np.maximum.accumulate(san_eq)).min())

    # 2022 rate-shock window
    rets_2022 = rets_full.loc["2022-01-01":"2022-12-31"]
    cum_2022 = float((1.0 + rets_2022).prod() - 1.0) if len(rets_2022) > 0 else float("nan")

    return {
        "sleeve": label,
        "oos_years": round(n_years, 2),
        "stitched_sharpe": round(sh, 3),
        "ci95_lo": round(ci_lo, 3),
        "ci95_hi": round(ci_hi, 3),
        "stitched_cagr": round(cagr, 4),
        "stitched_max_dd": round(dd, 4),
        "calmar": round(calmar, 3),
        "sanctuary_sharpe": round(san_sh, 3),
        "sanctuary_max_dd": round(san_dd, 4),
        "cum_2022": round(cum_2022, 4),
        "passes_ci_gate": ci_lo > 0,
    }


def _rotation_diagnostics(sleeve: RotationBondSleeve, index: pd.DatetimeIndex) -> dict:
    """Phase 3-specific diagnostics for the rotation sleeve."""
    # Re-derive the winner series with the same lag semantics as the
    # sleeve's daily_returns (winner_at_(t-1) decides return_at_t).
    common = None
    for s in sleeve.candidates.values():
        common = s.index if common is None else common.intersection(s.index)
    closes = pd.DataFrame({k: v.reindex(common) for k, v in sleeve.candidates.items()})
    moms = closes.pct_change(sleeve.lookback_days)
    winner = pd.Series("CASH", index=common)
    positive = moms.where(moms > 0)
    any_pos = positive.notna().any(axis=1)
    if any_pos.any():
        valid = positive.loc[any_pos].idxmax(axis=1)
        winner.loc[valid.index] = valid
    winner_lag = winner.shift(1).fillna("CASH")

    # Trim to the strategy's evaluation window
    winner_lag = winner_lag.reindex(index).fillna("CASH")
    flips = (winner_lag != winner_lag.shift(1)).fillna(False).sum()
    n_years = len(winner_lag) / 252.0
    flips_per_year = float(flips) / n_years if n_years > 0 else 0.0

    # Time spent in each state
    distribution = winner_lag.value_counts(normalize=True).to_dict()
    return {
        "total_flips": int(flips),
        "flips_per_year": round(flips_per_year, 2),
        "frac_IGLT": round(float(distribution.get("IGLT", 0.0)), 3),
        "frac_IGLS": round(float(distribution.get("IGLS", 0.0)), 3),
        "frac_CASH": round(float(distribution.get("CASH", 0.0)), 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    print("Phase 3 — bond rotation head-to-head on the 40/60 + capitulation baseline.")
    print("=" * 100)

    # Load the standard panel (SPY/VIX/HYG/IEF/TLT) and the two GBP gilts.
    data = load_panel(start="2003-04-01", end="2026-04-02")
    iglt = _load_close("IGLT_D.parquet")
    igls = _load_close("IGLS_D.parquet")

    # Common date window — limited by IGLS (starts 2010).
    common = (
        data["spy"]
        .index.intersection(data["ief"].index)
        .intersection(iglt.index)
        .intersection(igls.index)
    )
    print(f"Backtest window: {common.min().date()} to {common.max().date()} ({len(common)} bars).")
    print("Note: window is limited by IGLS coverage (starts 2010-01-04). Misses GFC.")
    print()

    spy = data["spy"].reindex(common)
    ief = data["ief"].reindex(common)
    iglt_a = iglt.reindex(common)
    igls_a = igls.reindex(common)
    vix = data["vix"].reindex(common)
    hyg = data["hyg"].reindex(common)
    tlt = data["tlt"].reindex(common)

    # Regime panel + score (no HMM — matches the 40/60 + cap baseline).
    panel = build_indicator_panel(spy, vix_close=vix, hyg_close=hyg, ief_close=ief, tlt_close=tlt)
    score = regime_score_equal(panel)

    cap_cfg = CapitulationConfig(enabled=True)
    cfg = StackedConfig(
        equity_weight=0.40,
        bond_weight=0.60,
        L_max=3.0,
        tier_thresholds=(0.30, 0.50, 0.75),
        capitulation=cap_cfg,
    )
    equity_engine = SyntheticETFEngine(ter_annual=cfg.leverage_ter_annual)

    # Three sleeves.
    sleeves: list[tuple[str, object]] = [
        ("IEF static (USD, audit reference)", StaticBondSleeve(name="IEF", close=ief)),
        ("IGLT static (GBP baseline)", StaticBondSleeve(name="IGLT", close=iglt_a)),
        (
            "IGLT/IGLS rotation (GBP, 60d mom)",
            RotationBondSleeve(
                name="UK_gilts_rotation",
                candidates={"IGLT": iglt_a, "IGLS": igls_a},
                lookback_days=60,
            ),
        ),
    ]

    rows: list[dict] = []
    rotation_diag: dict | None = None
    for label, sleeve in sleeves:
        print(f"  Running: {label}...", flush=True)
        df = run_stacked_strategy(
            spy,
            ief,
            score,
            cfg,
            indicator_panel=panel,
            equity_engine=equity_engine,
            bond_sleeve=sleeve,
        )
        rows.append(_summary(label, df))
        if isinstance(sleeve, RotationBondSleeve):
            rotation_diag = _rotation_diagnostics(sleeve, df.index)

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    summary_df = pd.DataFrame(rows).set_index("sleeve")
    print(summary_df.to_string())
    print()

    if rotation_diag is not None:
        print("Rotation diagnostics (IGLT/IGLS):")
        for k, v in rotation_diag.items():
            print(f"  {k}: {v}")
        print()

    # ── Phase 3 gate evaluation ─────────────────────────────────────────
    rot_row = next(r for r in rows if "rotation" in r["sleeve"])
    ief_row = next(r for r in rows if "IEF static" in r["sleeve"])

    gates = {
        "G1: rotation ci95_lo > 0": rot_row["ci95_lo"] > 0,
        "G2: rotation 2022 cum return > IEF static 2022 cum return": (
            rot_row["cum_2022"] > ief_row["cum_2022"]
        ),
        "G3: rotation churn ≤ 4 flips/year": (
            rotation_diag is not None and rotation_diag["flips_per_year"] <= 4.0
        ),
        "G4: rotation sanctuary Sharpe ≥ 0": rot_row["sanctuary_sharpe"] >= 0,
    }

    print("=" * 100)
    print("PHASE 3 GATE EVALUATION")
    print("=" * 100)
    for name, passed in gates.items():
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}")
    overall = all(gates.values())
    print(
        f"\nOverall: {'PASS — rotation can enter Phase 5 sweep' if overall else 'FAIL — bond sleeve stays static'}"
    )

    # Save outputs
    summary_df.to_csv(REPORTS_DIR / "phase3_bond_rotation_summary.csv")
    out_diag = pd.DataFrame([rotation_diag]) if rotation_diag else pd.DataFrame()
    if not out_diag.empty:
        out_diag.to_csv(REPORTS_DIR / "phase3_rotation_diagnostics.csv", index=False)
    print(f"\nSaved: {REPORTS_DIR / 'phase3_bond_rotation_summary.csv'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
